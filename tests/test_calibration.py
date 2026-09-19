"""Offline tests for PLUMB's blinded human-calibration workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from plumb.calibration import (
    CalibrationError,
    DEVELOPMENT_SPLIT,
    FrozenGateDProtocol,
    FrozenJudgeEvidenceManifest,
    GateDTolerances,
    HELDOUT_SPLIT,
    TrustedJudgeProducer,
    blinded_annotation_rows,
    build_calibration_report,
    calibration_manifest_hash,
    deterministic_annotation_assignments,
    export_annotation_packets,
    import_annotations,
    main,
    parse_annotations,
    validate_manifest,
)


ANNOTATORS = ("annotator-alex", "annotator-blair")


def full_manifest():
    rows = []
    for task_index, task in enumerate(
        ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")
    ):
        for split, count in ((DEVELOPMENT_SPLIT, 20), (HELDOUT_SPLIT, 10)):
            for index in range(count):
                clip_id = "clip-{0}-{1}-{2:02d}".format(task_index, split[:3], index)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "media_ref": "artifact://calibration/{0}.mp4".format(clip_id),
                        "task": task,
                        "split": split,
                        "source_lineage_id": "{0}-{1}-{2:02d}".format(split, task_index, index),
                        # Upstream provenance may retain this; annotation exports must not.
                        "policy": "OpenVLA",
                        "actions": [0.1, 0.2],
                        "condition": "generated-world-seed",
                    }
                )
    return rows


def annotation_row(clip, annotator_id, *, success=False, integrity="intact", collision="none_visible"):
    return {
        "clip_id": clip.clip_id,
        "annotator_id": annotator_id,
        "media_ref": clip.media_ref,
        "task": clip.task,
        "integrity": integrity,
        "collision": collision,
        "progress": 5 if success else 2,
        "completion_evidence": "met" if success else "not_met",
        "evidence_frame_indices": [0, 15],
        "observable_reason": "Visible end state.",
    }


def complete_human_labels(manifest):
    clips = validate_manifest(manifest)
    assignments = deterministic_annotation_assignments(clips, ANNOTATORS)
    rows = []
    for clip in clips:
        for annotator_id in assignments[clip.clip_id]:
            rows.append(annotation_row(clip, annotator_id))
    return rows


def sha256(label):
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def frozen_evidence_manifest(clips):
    heldout = [clip for clip in clips if clip.split == HELDOUT_SPLIT]
    return FrozenJudgeEvidenceManifest(
        calibration_manifest_hash=calibration_manifest_hash(clips),
        heldout_video_hashes={clip.clip_id: sha256("video:" + clip.clip_id) for clip in heldout},
        task_registry_id="bridge-task-registry-v1",
        task_registry_hash=sha256("task-registry"),
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        model_revision="0" * 40,
        processor_revision="1" * 40,
        transformers_version="4.49.0",
        runtime_lock_id="sha256:runtime-lock",
        asset_manifest_id="assets-lock-v1",
        heldout_artifact_hashes={
            clip.clip_id: {
                "video": sha256("artifact-video:" + clip.clip_id),
                "frame_0": sha256("artifact-frame:" + clip.clip_id),
                "reference_source": sha256("artifact-reference:" + clip.clip_id),
            }
            for clip in heldout
        },
        protocol_id="protocol-v1",
        trusted_producer=TrustedJudgeProducer("plumb-judge", "1.0.0", sha256("producer-source")),
    )


def raw_judge_report(clip, frozen, *, test_mode):
    parsed = {
        "integrity": "intact",
        "collision": "none_visible",
        "progress": 2,
        "completion_evidence": "not_met",
        "evidence_frame_indices": [0, 15],
        "observable_reasons": "Visible end state is incomplete.",
    }
    samples = [
        {
            "sample_index": index,
            "seed": 1000 + index,
            "attempts": [{"raw_output": json.dumps(parsed), "parsed": parsed, "failure_reason": None}],
        }
        for index in range(5)
    ]
    return {
        "binary_success": False,
        "progress": 2.0,
        "judge_status": "evaluable",
        "missing_reason": None,
        "agreeing_samples": 5,
        "raw_judge_samples": samples,
        "sampling": {
            "sample_count": 5,
            "quorum": 3,
            "temperature": 0.7,
            "top_p": 1.0,
            "max_new_tokens": 512,
            "retries_per_sample": 1,
        },
        "provenance": {
            "calibration_manifest_hash": frozen.calibration_manifest_hash,
            "clip_id": clip.clip_id,
            "task_id": clip.task,
            "evidence_hashes": {"video_hash": frozen.heldout_video_hashes[clip.clip_id]},
            "task_registry_id": frozen.task_registry_id,
            "task_registry_hash": frozen.task_registry_hash,
            "rubric_hash": "sha256:rubric",
            "sampling_hash": "sha256:sampling",
            "protocol_id": frozen.protocol_id,
            "model": {
                "id": frozen.model_id,
                "model_revision": frozen.model_revision,
                "processor_revision": frozen.processor_revision,
                "transformers_version": frozen.transformers_version,
                "runtime_lock_id": frozen.runtime_lock_id,
                "asset_manifest_id": frozen.asset_manifest_id,
            },
            "artifact_hashes": frozen.heldout_artifact_hashes[clip.clip_id],
            "producer": {
                "name": frozen.trusted_producer.name,
                "version": frozen.trusted_producer.version,
                "source_hash": frozen.trusted_producer.source_hash,
            },
            "trust": {
                "qualified": False,
                "gate_d_eligible": True,
                "test_mode": test_mode,
                "gate_d_unavailable_reasons": [],
            },
        },
    }


class ManifestAndExportTests(unittest.TestCase):
    def test_full_plan_counts_and_deterministic_assignments(self):
        clips = validate_manifest(full_manifest())
        assignments = deterministic_annotation_assignments(clips, ANNOTATORS)
        self.assertEqual(150, len(clips))
        self.assertEqual(200, sum(len(value) for value in assignments.values()))
        self.assertTrue(
            all(len(assignments[clip.clip_id]) == 2 for clip in clips if clip.split == HELDOUT_SPLIT)
        )
        self.assertTrue(
            all(len(assignments[clip.clip_id]) == 1 for clip in clips if clip.split == DEVELOPMENT_SPLIT)
        )

    def test_source_lineage_cannot_cross_development_and_heldout(self):
        rows = full_manifest()
        heldout = next(row for row in rows if row["split"] == HELDOUT_SPLIT)
        development = next(row for row in rows if row["split"] == DEVELOPMENT_SPLIT)
        heldout["source_lineage_id"] = development["source_lineage_id"]
        with self.assertRaises(CalibrationError):
            validate_manifest(rows)

    def test_blinded_export_drops_policy_action_and_condition_metadata(self):
        clips = validate_manifest(full_manifest())
        rows = blinded_annotation_rows(clips, ANNOTATORS[0], ANNOTATORS)
        self.assertGreaterEqual(len(rows), 50)  # every heldout clip is independently assigned
        forbidden = {"policy", "actions", "condition", "split", "source_lineage_id", "world_seed"}
        self.assertTrue(all(forbidden.isdisjoint(row) for row in rows))
        self.assertTrue(all(set(row) <= {
            "schema_version", "annotator_id", "clip_id", "media_ref", "task", "integrity", "collision",
            "progress", "completion_evidence", "evidence_frame_indices", "observable_reason",
        } for row in rows))

    def test_jsonl_and_cli_export_write_only_blinded_packets(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(full_manifest()))
            output_path = root / "alex.jsonl"
            exit_code = main(
                [
                    "export", "--manifest", str(manifest_path), "--annotator", ANNOTATORS[0],
                    "--annotators", *ANNOTATORS, "--output", str(output_path),
                ]
            )
            self.assertEqual(0, exit_code)
            exported = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertTrue(exported)
            self.assertTrue(all("policy" not in row and "condition" not in row for row in exported))


class AnnotationAndReportTests(unittest.TestCase):
    def test_strict_annotation_rejects_hidden_metadata_and_bad_progress(self):
        clips = validate_manifest(full_manifest())
        assignments = deterministic_annotation_assignments(clips, ANNOTATORS)
        clip = clips[0]
        row = annotation_row(clip, assignments[clip.clip_id][0])
        row["policy"] = "leaked"
        with self.assertRaises(CalibrationError):
            parse_annotations([row], clips, assignments, ANNOTATORS)

        row = annotation_row(clip, assignments[clip.clip_id][0])
        row["progress"] = 6
        with self.assertRaises(CalibrationError):
            parse_annotations([row], clips, assignments, ANNOTATORS)

    def test_disagreement_stays_unresolved_and_never_becomes_a_human_label(self):
        clips = validate_manifest(full_manifest())
        rows = complete_human_labels(clips)
        heldout = next(clip for clip in clips if clip.split == HELDOUT_SPLIT)
        target = next(
            row
            for row in rows
            if row["clip_id"] == heldout.clip_id and row["annotator_id"] == ANNOTATORS[1]
        )
        target.update({"progress": 5, "completion_evidence": "met"})
        report = build_calibration_report(clips, rows, ANNOTATORS)
        self.assertEqual(49, report["heldout_consensus"]["binary_success_consensus_count"])
        self.assertEqual(1, report["heldout_consensus"]["unresolved_binary_success_count"])
        self.assertFalse(report["gate_d"]["passed"])
        self.assertEqual("not_evaluable", report["gate_d"]["status"])

    def test_constant_progress_has_undefined_weighted_kappa(self):
        report = build_calibration_report(full_manifest(), complete_human_labels(full_manifest()), ANNOTATORS)
        progress = report["human_human_agreement"]["pooled"]["progress"]
        self.assertEqual(50, progress["n"])
        self.assertEqual(1.0, progress["raw_agreement"])
        self.assertIsNone(progress["weighted_kappa"])
        self.assertTrue(all(
            item["exploratory"] for item in report["human_human_agreement"]["task_exploratory"].values()
        ))

    def test_no_human_or_gate_d_pass_is_fabricated_without_protocol_and_tolerances(self):
        report = build_calibration_report(full_manifest(), [], ANNOTATORS)
        self.assertEqual(0, report["human_annotation_coverage"]["received_annotations"])
        self.assertEqual(0.0, report["human_annotation_coverage"]["coverage"])
        self.assertFalse(report["gate_d"]["passed"])
        self.assertEqual("not_evaluable", report["gate_d"]["status"])
        self.assertIn("frozen_protocol", report["gate_d"]["reason_codes"])
        self.assertIn("preregistered_tolerances", report["gate_d"]["reason_codes"])

    def test_pooled_confusion_leniency_and_null_coverage_require_actual_judge_labels(self):
        clips = validate_manifest(full_manifest())
        human = complete_human_labels(clips)
        judge = [annotation_row(clip, "judge-aggregate") for clip in clips if clip.split == HELDOUT_SPLIT]
        report = build_calibration_report(
            clips,
            human,
            ANNOTATORS,
            judge_annotations=judge,
            protocol=FrozenGateDProtocol("sha256:protocol", "sha256:rubric", "sha256:sampling"),
            tolerances=GateDTolerances(50, 1, 1, 1, 0, 0, 0),
        )
        comparison = report["judge_vs_human"]
        self.assertTrue(comparison["available"])
        self.assertEqual(50, comparison["confusion_matrix"]["true_negative"])
        self.assertEqual(0.0, comparison["leniency_offset"])
        self.assertEqual(0.0, comparison["null_coverage"]["either_unavailable"])
        # Descriptive/copyable Annotation labels cannot stand in for raw judge
        # reports bound to a frozen evidence manifest.
        self.assertEqual("blocked", report["gate_d"]["status"])
        self.assertIn("frozen_judge_evidence_manifest_missing", report["gate_d"]["reason_codes"])

    def test_test_mode_raw_judge_evidence_is_structural_but_cannot_qualify_gate_d(self):
        clips = validate_manifest(full_manifest())
        frozen = frozen_evidence_manifest(clips)
        reports = [raw_judge_report(clip, frozen, test_mode=True) for clip in clips if clip.split == HELDOUT_SPLIT]
        report = build_calibration_report(
            clips,
            complete_human_labels(clips),
            ANNOTATORS,
            judge_reports=reports,
            protocol=FrozenGateDProtocol(
                "sha256:protocol", "sha256:rubric", "sha256:sampling", evidence_manifest=frozen
            ),
            tolerances=GateDTolerances(50, 1, 1, 1, 0, 0, 0),
        )
        self.assertFalse(report["primary_judge_evidence"]["available"])
        self.assertEqual("raw_judge_evidence_invalid", report["primary_judge_evidence"]["reason_code"])
        self.assertEqual("blocked", report["gate_d"]["status"])
        self.assertFalse(report["gate_d"]["passed"])

    def test_bound_raw_judge_evidence_is_reaggregated_before_gate_metrics(self):
        clips = validate_manifest(full_manifest())
        frozen = frozen_evidence_manifest(clips)
        reports = [raw_judge_report(clip, frozen, test_mode=False) for clip in clips if clip.split == HELDOUT_SPLIT]
        report = build_calibration_report(
            clips,
            complete_human_labels(clips),
            ANNOTATORS,
            judge_reports=reports,
            protocol=FrozenGateDProtocol(
                "sha256:protocol", "sha256:rubric", "sha256:sampling", evidence_manifest=frozen
            ),
            tolerances=GateDTolerances(50, 1, 1, 1, 0, 0, 0),
        )
        self.assertTrue(report["primary_judge_evidence"]["available"])
        self.assertEqual(50, report["primary_judge_evidence"]["reaggregated_report_count"])
        self.assertEqual(50, report["primary_judge_evidence"]["judge_comparison"]["confusion_matrix"]["true_negative"])
        # The binding establishes only declared provenance and reaggregation;
        # prevalence makes sensitivity undefined, so a Gate-D pass is still
        # impossible even with structurally complete evidence.
        self.assertEqual("fail", report["gate_d"]["status"])
        self.assertIn("sensitivity_undefined", report["gate_d"]["reason_codes"])

    def test_export_function_can_write_csv_without_any_labels(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "blair.csv"
            count = export_annotation_packets(full_manifest(), output, ANNOTATORS[1], ANNOTATORS)
            self.assertGreaterEqual(count, 50)
            header = output.read_text().splitlines()[0]
            self.assertNotIn("policy", header)
            self.assertNotIn("condition", header)

    def test_csv_label_schema_version_round_trips_as_a_string(self):
        clips = validate_manifest(full_manifest())
        assignments = deterministic_annotation_assignments(clips, ANNOTATORS)
        clip = clips[0]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.csv"
            row = annotation_row(clip, assignments[clip.clip_id][0])
            row["schema_version"] = 1
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            labels = import_annotations(path, clips, assignments, ANNOTATORS)
        self.assertEqual(1, len(labels))
        self.assertEqual(2, labels[0].progress)


class SplitPlanTests(unittest.TestCase):
    def test_the_twenty_and_ten_plan_is_enforced_per_task_not_only_in_total(self):
        rows = full_manifest()
        # Move one held-out clip from open_drawer to close_drawer: the total
        # stays 150 but two task cells are now wrong.
        target = next(
            row for row in rows if row["task"] == "open_drawer" and row["split"] == HELDOUT_SPLIT
        )
        target["task"] = "close_drawer"
        with self.assertRaises(CalibrationError) as raised:
            validate_manifest(rows)
        self.assertIn("requires 10 heldout clips for open_drawer", str(raised.exception))

    def test_a_manifest_short_of_one_hundred_and_fifty_clips_is_refused(self):
        rows = full_manifest()[:-1]
        with self.assertRaises(CalibrationError) as raised:
            validate_manifest(rows)
        self.assertIn("exactly 150 clips", str(raised.exception))

    def test_a_partial_manifest_is_only_accepted_when_the_caller_says_so(self):
        rows = full_manifest()[:5]
        with self.assertRaises(CalibrationError):
            validate_manifest(rows)
        self.assertEqual(5, len(validate_manifest(rows, require_full_plan=False)))


class BlindedFieldContractTests(unittest.TestCase):
    def test_the_allowlist_and_denylist_never_overlap(self):
        from plumb.calibration import BLINDED_EXPORT_FIELDS, _FORBIDDEN_BLIND_FIELDS

        self.assertTrue(_FORBIDDEN_BLIND_FIELDS.isdisjoint(set(BLINDED_EXPORT_FIELDS)))
        for name in ("policy", "backend", "world_seed", "cohort", "condition", "gate_status"):
            self.assertIn(name, _FORBIDDEN_BLIND_FIELDS)

    def test_a_blinded_row_carries_no_field_outside_the_allowlist(self):
        from plumb.calibration import BLINDED_EXPORT_FIELDS

        clips = validate_manifest(full_manifest())
        for row in blinded_annotation_rows(clips, ANNOTATORS[0], ANNOTATORS):
            self.assertEqual(set(BLINDED_EXPORT_FIELDS), set(row))


class BinarySuccessAgreementTests(unittest.TestCase):
    def test_indecisive_labels_are_retained_as_missing_not_recoded_as_failure(self):
        clips = validate_manifest(full_manifest())
        rows = complete_human_labels(clips)
        heldout = next(clip for clip in clips if clip.split == HELDOUT_SPLIT)
        for row in rows:
            if row["clip_id"] == heldout.clip_id:
                row.update({"integrity": "uncertain", "completion_evidence": "uncertain", "progress": 2})
        report = build_calibration_report(clips, rows, ANNOTATORS)
        binary = report["human_human_agreement"]["pooled"]["binary_success"]
        self.assertEqual(50, binary["paired_clips"])
        self.assertEqual(1, binary["indecisive_pairs"])
        self.assertEqual(49, binary["n"])
        self.assertEqual(1.0, binary["raw_agreement"])

    def test_a_degenerate_binary_kappa_is_null_and_never_nan(self):
        report = build_calibration_report(full_manifest(), complete_human_labels(full_manifest()), ANNOTATORS)
        binary = report["human_human_agreement"]["pooled"]["binary_success"]
        # Every label is the same category, so expected agreement is 1 and
        # kappa is undefined. It must be null, not NaN and not zero.
        self.assertIsNone(binary["kappa"])
        self.assertEqual(1.0, binary["raw_agreement"])
        encoded = json.dumps(report, allow_nan=False)
        self.assertNotIn("NaN", encoded)


class CalibrationJsonOutputTests(unittest.TestCase):
    def test_the_report_command_writes_the_artifact_atomically(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(full_manifest()))
            labels_path = root / "labels.jsonl"
            clips = validate_manifest(full_manifest())
            surrogates = ("external:autoeval", "model:qwen")
            assignments = deterministic_annotation_assignments(clips, surrogates)
            rows = [
                annotation_row(clip, annotator_id)
                for clip in clips
                for annotator_id in assignments[clip.clip_id]
            ]
            labels_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            registry_path = root / "annotators.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "annotators": [
                            {
                                "annotator_id": "external:autoeval",
                                "annotator_type": "external_label",
                                "label_source": "AutoEval classifier labels",
                                "source_uri": "hf://datasets/zhouzypaul/auto_eval",
                            },
                            {
                                "annotator_id": "model:qwen",
                                "annotator_type": "model",
                                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                                "model_revision": "c" * 40,
                            },
                        ]
                    }
                )
            )
            output = root / "results" / "judge_calibration.json"
            exit_code = main(
                [
                    "report",
                    "--manifest",
                    str(manifest_path),
                    "--annotations",
                    str(labels_path),
                    "--annotators",
                    "external:autoeval",
                    "model:qwen",
                    "--annotator-registry",
                    str(registry_path),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(0, exit_code)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("external_label + model_reference", payload["calibration_class"])
            self.assertEqual("pass_with_limitations", payload["gate_d"]["status"]) if payload["gate_d"][
                "thresholds_satisfied"
            ] else self.assertFalse(payload["gate_d"]["passed"])
            self.assertEqual(["human_annotation"], payload["gate_d"]["open_dependency_names"])
            self.assertNotIn("NaN", output.read_text(encoding="utf-8"))
            self.assertEqual(
                ["judge_calibration.json"], [item.name for item in output.parent.iterdir()]
            )

    def test_the_output_option_refuses_to_guess_an_annotator_type(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(full_manifest()))
            labels_path = root / "labels.jsonl"
            labels_path.write_text("")
            with self.assertRaises(SystemExit):
                main(
                    [
                        "report",
                        "--manifest",
                        str(manifest_path),
                        "--annotations",
                        str(labels_path),
                        "--annotators",
                        *ANNOTATORS,
                        "--output",
                        str(root / "out.json"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
