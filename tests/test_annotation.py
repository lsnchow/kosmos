"""Offline adversarial tests for the blinded annotation surface and Gate D wiring."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cluster import build_calibration_panel as panel
from plumb.annotation import (
    ANNOTATOR_TYPES,
    EXTERNAL_LABEL_ANNOTATOR,
    HUMAN_ANNOTATION_DEPENDENCY,
    HUMAN_ANNOTATOR,
    MODEL_ANNOTATOR,
    SECOND_HUMAN_DEPENDENCY,
    VIEW_ONLY_FIELDS,
    AnnotationError,
    AnnotationService,
    AnnotationStore,
    AnnotatorIdentity,
    AnnotatorRegistry,
    LabelSubmission,
    assert_blinded,
    assert_finite_json,
    blinded_clip_view,
    build_judge_calibration_payload,
    gate_d_evidence,
    gate_d_measurements,
    gate_d_with_limitations,
    paired_difference_interval,
    wilson_interval,
    write_json_atomic,
    write_judge_calibration_json,
)
from plumb.calibration import (
    DEVELOPMENT_SPLIT,
    HELDOUT_SPLIT,
    FrozenGateDProtocol,
    FrozenJudgeEvidenceManifest,
    GateDTolerances,
    TrustedJudgeProducer,
    _FORBIDDEN_BLIND_FIELDS,
    build_calibration_report,
    calibration_manifest_hash,
    deterministic_annotation_assignments,
    validate_manifest,
)
from plumb.gates import GateRecord


TASKS = ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")


def digest(label):
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def full_manifest():
    rows = []
    for task_index, task in enumerate(TASKS):
        for split, count in ((DEVELOPMENT_SPLIT, 20), (HELDOUT_SPLIT, 10)):
            for index in range(count):
                clip_id = "clip-%d%s%02d" % (task_index, "d" if split == DEVELOPMENT_SPLIT else "h", index)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "media_ref": "artifact://calibration/%s.mp4" % clip_id,
                        "task": task,
                        "split": split,
                        "source_lineage_id": "%s-%d-%02d" % (split, task_index, index),
                    }
                )
    return rows


def human_registry():
    return AnnotatorRegistry(
        (
            AnnotatorIdentity("human:alex", HUMAN_ANNOTATOR),
            AnnotatorIdentity("human:blair", HUMAN_ANNOTATOR),
        )
    )


def surrogate_registry():
    return AnnotatorRegistry(
        (
            AnnotatorIdentity(
                "external:autoeval-classifier",
                EXTERNAL_LABEL_ANNOTATOR,
                label_source="zhouzypaul/auto_eval drawer classifier labels",
                source_uri="hf://datasets/zhouzypaul/auto_eval",
            ),
            AnnotatorIdentity(
                "model:qwen-blind-pass-b",
                MODEL_ANNOTATOR,
                model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                model_revision="c" * 40,
            ),
        )
    )


def _is_success(clip_id):
    """Alternate the ground truth so sensitivity and specificity are defined."""

    return clip_id.endswith(("0", "2", "4", "6", "8"))


def label_rows(clips, registry, *, disagree_on=()):
    assignments = deterministic_annotation_assignments(clips, registry.annotator_ids)
    rows = []
    for clip in clips:
        success = _is_success(clip.clip_id)
        for position, annotator_id in enumerate(assignments[clip.clip_id]):
            flipped = success
            if clip.clip_id in disagree_on and position == 1:
                flipped = not success
            rows.append(
                {
                    "clip_id": clip.clip_id,
                    "annotator_id": annotator_id,
                    "integrity": "intact",
                    "collision": "none_visible",
                    "progress": 5 if flipped else 2,
                    "completion_evidence": "met" if flipped else "not_met",
                    "evidence_frame_indices": [0, 15],
                    "observable_reason": "Visible end state.",
                }
            )
    return rows


def frozen_manifest(clips):
    heldout = [clip for clip in clips if clip.split == HELDOUT_SPLIT]
    return FrozenJudgeEvidenceManifest(
        calibration_manifest_hash=calibration_manifest_hash(clips),
        heldout_video_hashes={clip.clip_id: digest("video:" + clip.clip_id) for clip in heldout},
        task_registry_id="plumb-benchmark-tasks-v1",
        task_registry_hash=digest("task-registry"),
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        model_revision="0" * 40,
        processor_revision="1" * 40,
        transformers_version="4.49.0",
        runtime_lock_id="runtime-lock-v1",
        asset_manifest_id="assets-lock-v1",
        heldout_artifact_hashes={
            clip.clip_id: {"video": digest("artifact:" + clip.clip_id)} for clip in heldout
        },
        protocol_id="protocol-v1",
        trusted_producer=TrustedJudgeProducer("plumb-judge", "1.0.0", digest("producer")),
    )


def raw_judge_report(clip, frozen, *, success):
    parsed = {
        "integrity": "intact",
        "collision": "none_visible",
        "progress": 5 if success else 2,
        "completion_evidence": "met" if success else "not_met",
        "evidence_frame_indices": [0, 15],
        "observable_reasons": "Visible end state.",
    }
    return {
        "binary_success": success,
        "progress": 5.0 if success else 2.0,
        "judge_status": "evaluable",
        "missing_reason": None,
        "agreeing_samples": 5,
        "raw_judge_samples": [
            {
                "sample_index": index,
                "seed": 1000 + index,
                "attempts": [{"raw_output": json.dumps(parsed), "parsed": parsed, "failure_reason": None}],
            }
            for index in range(5)
        ],
        "sampling": dict(panel.PRIMARY_SAMPLING),
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
                "test_mode": False,
                "gate_d_unavailable_reasons": [],
            },
        },
    }


def passing_report(registry):
    """A calibration report whose frozen tolerances are actually satisfied."""

    clips = validate_manifest(full_manifest())
    frozen = frozen_manifest(clips)
    reports = [
        raw_judge_report(clip, frozen, success=_is_success(clip.clip_id))
        for clip in clips
        if clip.split == HELDOUT_SPLIT
    ]
    return clips, build_calibration_report(
        clips,
        label_rows(clips, registry),
        registry.annotator_ids,
        judge_reports=reports,
        protocol=FrozenGateDProtocol(
            "sha256:protocol", "sha256:rubric", "sha256:sampling", evidence_manifest=frozen
        ),
        tolerances=GateDTolerances(50, 1.0, 1.0, 1.0, 0.9, 0.9, 0.1),
    )


class AnnotatorIdentityTests(unittest.TestCase):
    def test_a_model_annotator_can_never_be_serialized_as_human(self):
        model = AnnotatorIdentity(
            "model:qwen", MODEL_ANNOTATOR, model_id="Qwen/Qwen2.5-VL-7B-Instruct", model_revision="c" * 40
        )
        self.assertEqual(MODEL_ANNOTATOR, model.as_mapping()["annotator_type"])
        self.assertFalse(model.is_human)
        with self.assertRaises(AnnotationError):
            AnnotatorIdentity("human:pretending", MODEL_ANNOTATOR, model_id="x", model_revision="y")
        with self.assertRaises(AnnotationError):
            AnnotatorIdentity("model:qwen", HUMAN_ANNOTATOR)
        with self.assertRaises(AnnotationError):
            AnnotatorIdentity("human:alex", HUMAN_ANNOTATOR, model_id="Qwen/Qwen2.5-VL-7B-Instruct")
        with self.assertRaises(AnnotationError):
            AnnotatorIdentity.from_mapping(
                {"annotator_id": "human:alex", "annotator_type": HUMAN_ANNOTATOR, "model_id": "sneaky"}
            )

    def test_a_model_annotator_must_record_its_model_and_revision(self):
        with self.assertRaises(AnnotationError):
            AnnotatorIdentity("model:qwen", MODEL_ANNOTATOR)
        with self.assertRaises(AnnotationError):
            AnnotatorIdentity("model:qwen", MODEL_ANNOTATOR, model_id="Qwen/Qwen2.5-VL-7B-Instruct")

    def test_an_external_label_annotator_must_name_its_third_party_source(self):
        with self.assertRaises(AnnotationError):
            AnnotatorIdentity("external:autoeval", EXTERNAL_LABEL_ANNOTATOR)
        annotator = AnnotatorIdentity(
            "external:autoeval",
            EXTERNAL_LABEL_ANNOTATOR,
            label_source="AutoEval classifier labels",
            source_uri="hf://datasets/zhouzypaul/auto_eval",
        )
        self.assertFalse(annotator.is_human)

    def test_unknown_annotator_types_are_refused(self):
        for value in ("expert", "assistant", "agent", "", None):
            with self.assertRaises(AnnotationError):
                AnnotatorIdentity("human:alex", value)
        self.assertEqual((HUMAN_ANNOTATOR, MODEL_ANNOTATOR, EXTERNAL_LABEL_ANNOTATOR), ANNOTATOR_TYPES)

    def test_calibration_class_names_the_actual_arms(self):
        self.assertEqual("human", human_registry().calibration_class)
        self.assertEqual("external_label + model_reference", surrogate_registry().calibration_class)
        self.assertTrue(human_registry().has_two_blinded_humans)
        self.assertFalse(surrogate_registry().has_two_blinded_humans)
        self.assertEqual(0, surrogate_registry().human_count)

    def test_a_registry_needs_two_distinct_blinded_annotators(self):
        with self.assertRaises(AnnotationError):
            AnnotatorRegistry((AnnotatorIdentity("human:alex", HUMAN_ANNOTATOR),))
        with self.assertRaises(AnnotationError):
            AnnotatorRegistry(
                (
                    AnnotatorIdentity("human:alex", HUMAN_ANNOTATOR),
                    AnnotatorIdentity("human:alex", HUMAN_ANNOTATOR),
                )
            )
        with self.assertRaises(AnnotationError):
            AnnotatorRegistry(
                (
                    AnnotatorIdentity("human:alex", HUMAN_ANNOTATOR, blinded=False),
                    AnnotatorIdentity("human:blair", HUMAN_ANNOTATOR),
                )
            )

    def test_the_registry_round_trips_through_a_file(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotators.json"
            surrogate_registry().save(path)
            loaded = AnnotatorRegistry.load(path)
            self.assertEqual("external_label + model_reference", loaded.calibration_class)
            self.assertEqual(("c" * 40), loaded.get("model:qwen-blind-pass-b").model_revision)


class BlindedViewTests(unittest.TestCase):
    def test_a_blinded_export_leaks_none_of_the_forbidden_fields(self):
        clips = validate_manifest(full_manifest())
        view = blinded_clip_view(clips[0], "human:alex")
        self.assertTrue(_FORBIDDEN_BLIND_FIELDS.isdisjoint(set(view)))
        for name in (
            "policy",
            "backend",
            "actions",
            "action_text",
            "commands",
            "condition",
            "gate_status",
            "reference_rate",
            "world_seed",
            "cohort",
            "split",
            "source_lineage_id",
        ):
            self.assertNotIn(name, view)
        assert_blinded(view)

    def test_view_only_fields_never_collide_with_the_shared_denylist(self):
        self.assertTrue(_FORBIDDEN_BLIND_FIELDS.isdisjoint(set(VIEW_ONLY_FIELDS)))

    def test_a_policy_name_in_a_media_path_is_rejected_as_a_leak(self):
        from plumb.calibration import ClipManifestRow

        leaky = ClipManifestRow(
            clip_id="clip-leak",
            media_ref="artifact://runs/OpenVLA/episode-3.mp4",
            task="open_drawer",
            split=HELDOUT_SPLIT,
            source_lineage_id="lineage-1",
        )
        with self.assertRaises(AnnotationError) as raised:
            blinded_clip_view(leaky, "human:alex")
        self.assertIn("OpenVLA", str(raised.exception))

    def test_a_split_label_in_a_payload_is_rejected_as_a_leak(self):
        with self.assertRaises(AnnotationError):
            assert_blinded({"note": "this clip is from the heldout split"})
        with self.assertRaises(AnnotationError):
            assert_blinded({"nested": [{"backend": "irasim"}]})
        with self.assertRaises(AnnotationError):
            assert_blinded({"world_seed": 7})
        assert_blinded({"note": "the gripper closes on the handle"})

    def test_the_view_shows_the_frozen_rubric_without_a_success_percentage(self):
        clips = validate_manifest(full_manifest())
        view = blinded_clip_view(clips[0], "human:alex")
        self.assertIn("milestone rubric", view["task_rubric"])
        self.assertNotIn("%", view["task_rubric"])
        self.assertNotIn("percent", view["task_rubric"].lower())
        self.assertEqual(6, len(view["progress_scale"]))
        self.assertEqual("", view["progress"])

    def test_the_media_url_is_derived_from_the_blinded_artifact_reference(self):
        clips = validate_manifest(full_manifest())
        view = blinded_clip_view(clips[0], "human:alex", media_url_prefix="/api/artifacts")
        self.assertTrue(view["media_url"].startswith("/api/artifacts/calibration/"))
        self.assertTrue(view["media_url"].endswith(".mp4"))


class LabelCaptureTests(unittest.TestCase):
    def test_a_null_progress_stays_unknown_and_is_not_a_label(self):
        submission = LabelSubmission(
            clip_id="clip-0d00",
            annotator_id="human:alex",
            integrity="intact",
            collision="none_visible",
            progress=None,
            completion_evidence="uncertain",
            evidence_frame_indices=(0,),
        )
        self.assertFalse(submission.complete)
        self.assertIn("progress", submission.missing_fields())
        with self.assertRaises(AnnotationError):
            submission.to_annotation_row()

    def test_a_complete_capture_converts_to_a_strict_calibration_label(self):
        submission = LabelSubmission(
            clip_id="clip-0d00",
            annotator_id="human:alex",
            integrity="intact",
            collision="none_visible",
            progress=5,
            completion_evidence="met",
            evidence_frame_indices=(0, 15),
            observable_reason="Drawer fully closed in the final frame.",
        )
        row = submission.to_annotation_row()
        self.assertEqual(5, row["progress"])
        self.assertEqual("met", row["completion_evidence"])
        self.assertTrue(set(row).isdisjoint(_FORBIDDEN_BLIND_FIELDS))

    def test_inconsistent_progress_and_completion_are_refused(self):
        with self.assertRaises(AnnotationError):
            LabelSubmission(
                clip_id="c", annotator_id="human:alex", progress=3, completion_evidence="met"
            )
        with self.assertRaises(AnnotationError):
            LabelSubmission(
                clip_id="c", annotator_id="human:alex", progress=5, completion_evidence="not_met"
            )

    def test_out_of_range_and_repeated_evidence_frames_are_refused(self):
        with self.assertRaises(AnnotationError):
            LabelSubmission(clip_id="c", annotator_id="human:alex", progress=6)
        with self.assertRaises(AnnotationError):
            LabelSubmission(clip_id="c", annotator_id="human:alex", evidence_frame_indices=(16,))
        with self.assertRaises(AnnotationError):
            LabelSubmission(clip_id="c", annotator_id="human:alex", evidence_frame_indices=(1, 1))

    def test_a_submission_carrying_blinded_metadata_is_refused(self):
        with self.assertRaises(AnnotationError):
            LabelSubmission.from_mapping(
                {"clip_id": "c", "annotator_id": "human:alex", "policy": "OpenVLA"}
            )
        with self.assertRaises(AnnotationError):
            LabelSubmission.from_mapping(
                {"clip_id": "c", "annotator_id": "human:alex", "world_seed": 7}
            )


class AnnotationServiceTests(unittest.TestCase):
    def _service(self, temporary, registry=None):
        registry = registry or human_registry()
        store = AnnotationStore(Path(temporary) / "labels.jsonl")
        return AnnotationService(full_manifest(), registry, store), registry

    def test_the_queue_and_session_never_reveal_the_split(self):
        with TemporaryDirectory() as temporary:
            service, registry = self._service(temporary)
            session = service.session(registry.annotator_ids[0])
            self.assertEqual(HUMAN_ANNOTATOR, session["annotator_type"])
            self.assertNotIn("split", session)
            queue = service.queue(registry.annotator_ids[0])
            self.assertEqual(session["total_assigned"], len(queue["clips"]))
            for view in queue["clips"]:
                assert_blinded(view)
            # every held-out clip is assigned to both annotators
            self.assertGreaterEqual(len(queue["clips"]), 50)

    def test_a_submission_cannot_declare_its_own_annotator_type(self):
        with TemporaryDirectory() as temporary:
            service, registry = self._service(temporary)
            session = service.session(registry.annotator_ids[0])
            with self.assertRaises(AnnotationError) as raised:
                service.submit(
                    {
                        "clip_id": session["next_clip_id"],
                        "annotator_id": registry.annotator_ids[0],
                        "annotator_type": HUMAN_ANNOTATOR,
                    }
                )
            self.assertIn("may not declare annotator_type", str(raised.exception))

    def test_the_store_records_the_registry_type_not_the_submitted_one(self):
        with TemporaryDirectory() as temporary:
            service, registry = self._service(temporary, surrogate_registry())
            annotator_id = registry.annotator_ids[0]
            clip_id = service.session(annotator_id)["next_clip_id"]
            accepted = service.submit(
                {
                    "clip_id": clip_id,
                    "annotator_id": annotator_id,
                    "integrity": "intact",
                    "collision": "none_visible",
                    "progress": 2,
                    "completion_evidence": "not_met",
                    "evidence_frame_indices": [0, 15],
                    "observable_reason": "Incomplete.",
                }
            )
            self.assertNotEqual(HUMAN_ANNOTATOR, accepted["annotator_type"])
            self.assertEqual(registry.get(annotator_id).annotator_type, accepted["annotator_type"])
            self.assertEqual({annotator_id: accepted["annotator_type"]}, service.store.recorded_types())

    def test_a_stored_type_that_contradicts_the_registry_is_refused(self):
        with TemporaryDirectory() as temporary:
            registry = human_registry()
            path = Path(temporary) / "labels.jsonl"
            clips = validate_manifest(full_manifest())
            assignments = deterministic_annotation_assignments(clips, registry.annotator_ids)
            clip = clips[0]
            forged = {
                "record_type": "label",
                "schema_version": 1,
                "clip_id": clip.clip_id,
                "annotator_id": assignments[clip.clip_id][0],
                "annotator_type": MODEL_ANNOTATOR,
                "integrity": "intact",
                "collision": "none_visible",
                "progress": 2,
                "completion_evidence": "not_met",
                "evidence_frame_indices": [0, 15],
                "observable_reason": "x",
                "complete": True,
                "missing_fields": [],
                "submitted_at": "2026-09-19T00:00:00.000Z",
            }
            path.write_text(json.dumps(forged) + "\n")
            service = AnnotationService(full_manifest(), registry, AnnotationStore(path))
            with self.assertRaises(AnnotationError) as raised:
                service.annotation_rows()
            self.assertIn("registry declares", str(raised.exception))

    def test_an_unassigned_clip_is_refused(self):
        with TemporaryDirectory() as temporary:
            service, registry = self._service(temporary)
            development = [
                clip
                for clip in service.clips
                if clip.split == DEVELOPMENT_SPLIT
                and registry.annotator_ids[0] not in service.assignments[clip.clip_id]
            ]
            self.assertTrue(development)
            with self.assertRaises(AnnotationError):
                service.clip(registry.annotator_ids[0], development[0].clip_id)
            with self.assertRaises(AnnotationError):
                service.submit({"clip_id": development[0].clip_id, "annotator_id": registry.annotator_ids[0]})

    def test_incomplete_captures_are_counted_separately_from_labels(self):
        with TemporaryDirectory() as temporary:
            service, registry = self._service(temporary)
            annotator_id = registry.annotator_ids[0]
            clip_id = service.session(annotator_id)["next_clip_id"]
            service.submit(
                {"clip_id": clip_id, "annotator_id": annotator_id, "integrity": "intact"}
            )
            progress = service.progress()
            self.assertEqual(1, progress["by_annotator"][annotator_id]["captured"])
            self.assertEqual(0, progress["by_annotator"][annotator_id]["complete"])
            self.assertEqual(1, progress["by_annotator"][annotator_id]["incomplete_captures"])
            self.assertEqual([], service.annotation_rows())

    def test_export_writes_only_complete_labels_in_the_calibration_shape(self):
        with TemporaryDirectory() as temporary:
            service, registry = self._service(temporary)
            for clip in service.clips:
                for annotator_id in service.assignments[clip.clip_id]:
                    service.submit(
                        {
                            "clip_id": clip.clip_id,
                            "annotator_id": annotator_id,
                            "integrity": "intact",
                            "collision": "none_visible",
                            "progress": 2,
                            "completion_evidence": "not_met",
                            "evidence_frame_indices": [0, 15],
                            "observable_reason": "Incomplete.",
                        }
                    )
            output = Path(temporary) / "labels_export.jsonl"
            result = service.store.export_annotation_rows(output)
            self.assertEqual(200, result["complete_labels"])
            self.assertEqual(0, result["incomplete_captures"])
            first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(set(first).isdisjoint(_FORBIDDEN_BLIND_FIELDS))
            self.assertNotIn("annotator_type", first)


class GateDStatusTests(unittest.TestCase):
    def test_gate_d_without_two_humans_is_pass_with_limitations_and_names_the_dependency(self):
        registry = surrogate_registry()
        _, report = passing_report(registry)
        self.assertEqual("pass", report["gate_d"]["status"])
        wrapped = gate_d_with_limitations(report["gate_d"], registry)
        self.assertEqual("pass_with_limitations", wrapped["status"])
        self.assertFalse(wrapped["passed"])
        self.assertTrue(wrapped["thresholds_satisfied"])
        self.assertEqual([HUMAN_ANNOTATION_DEPENDENCY], wrapped["open_dependency_names"])
        self.assertEqual("external_label + model_reference", wrapped["calibration_class"])
        self.assertIn("30 minutes", wrapped["open_dependencies"][0]["resolution_path"])

    def test_gate_d_with_two_blinded_humans_can_read_pass(self):
        registry = human_registry()
        _, report = passing_report(registry)
        wrapped = gate_d_with_limitations(report["gate_d"], registry)
        self.assertEqual("pass", wrapped["status"])
        self.assertTrue(wrapped["passed"])
        self.assertEqual([], wrapped["open_dependency_names"])
        self.assertEqual("human", wrapped["calibration_class"])

    def test_one_human_still_names_the_missing_second_annotator(self):
        registry = AnnotatorRegistry(
            (
                AnnotatorIdentity("human:alex", HUMAN_ANNOTATOR),
                AnnotatorIdentity(
                    "model:qwen", MODEL_ANNOTATOR, model_id="Qwen/Qwen2.5-VL-7B-Instruct", model_revision="c" * 40
                ),
            )
        )
        _, report = passing_report(registry)
        wrapped = gate_d_with_limitations(report["gate_d"], registry)
        self.assertEqual("pass_with_limitations", wrapped["status"])
        self.assertIn(SECOND_HUMAN_DEPENDENCY, wrapped["open_dependency_names"])
        self.assertEqual("human + model_reference", wrapped["calibration_class"])

    def test_wrapping_an_already_wrapped_record_is_idempotent(self):
        registry = surrogate_registry()
        _, report = passing_report(registry)
        once = gate_d_with_limitations(report["gate_d"], registry)
        twice = gate_d_with_limitations(once, registry)
        self.assertEqual("pass_with_limitations", twice["status"])
        self.assertTrue(twice["thresholds_satisfied"])
        self.assertFalse(twice["passed"])
        self.assertEqual([HUMAN_ANNOTATION_DEPENDENCY], twice["open_dependency_names"])
        # Upgrading to two humans on a re-wrap reaches a real pass.
        upgraded = gate_d_with_limitations(once, human_registry())
        self.assertEqual("pass", upgraded["status"])
        self.assertTrue(upgraded["passed"])

    def test_the_written_artifact_can_be_rewrapped_without_losing_the_statistics(self):
        registry = surrogate_registry()
        clips, report = passing_report(registry)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "judge_calibration.json"
            write_judge_calibration_json(path, report, registry, manifest=clips)
            artifact = json.loads(path.read_text(encoding="utf-8"))
        measurements = gate_d_measurements(artifact, registry)
        self.assertIsNotNone(measurements["leniency_offset"])
        self.assertIsNotNone(measurements["consensus_coverage"])
        self.assertIsNotNone(measurements["binary_kappa"])
        rewrapped = gate_d_with_limitations(artifact["gate_d"], registry)
        self.assertEqual("pass_with_limitations", rewrapped["status"])
        self.assertTrue(rewrapped["thresholds_satisfied"])

    def test_an_empty_gate_d_block_is_not_evaluable_rather_than_null(self):
        wrapped = gate_d_with_limitations({}, human_registry())
        self.assertEqual("not_evaluable", wrapped["status"])
        self.assertFalse(wrapped["passed"])
        self.assertFalse(wrapped["thresholds_satisfied"])

    def test_a_failing_threshold_keeps_its_own_status(self):
        registry = human_registry()
        _, report = passing_report(registry)
        failed = dict(report["gate_d"])
        failed.update({"status": "fail", "passed": False, "reason_codes": ["sensitivity_below_minimum"]})
        wrapped = gate_d_with_limitations(failed, registry)
        self.assertEqual("fail", wrapped["status"])
        self.assertFalse(wrapped["thresholds_satisfied"])

    def test_unresolved_human_disagreement_is_never_broken_by_a_tie_break(self):
        registry = human_registry()
        clips = validate_manifest(full_manifest())
        contested = next(clip for clip in clips if clip.split == HELDOUT_SPLIT)
        rows = label_rows(clips, registry, disagree_on=(contested.clip_id,))
        report = build_calibration_report(clips, rows, registry.annotator_ids)
        self.assertEqual(49, report["heldout_consensus"]["binary_success_consensus_count"])
        self.assertEqual(1, report["heldout_consensus"]["unresolved_binary_success_count"])
        payload = build_judge_calibration_payload(report, registry)
        self.assertIn("No project-team tie-break", payload["unresolved_disagreement_policy"])


class GateDEvidenceTests(unittest.TestCase):
    def test_gate_d_measurements_supply_every_key_the_gate_record_requires(self):
        registry = human_registry()
        _, report = passing_report(registry)
        measurements = gate_d_measurements(
            report,
            registry,
            freeze_created_at="2026-09-19T00:00:00.000Z",
            judge_started_at="2026-09-19T01:00:00.000Z",
        )
        for name in ("binary_kappa", "weighted_progress_kappa", "leniency_offset", "consensus_coverage"):
            self.assertIsNotNone(measurements[name], name)
        self.assertEqual("human", measurements["calibration_class"])
        self.assertTrue(measurements["held_out_frozen_before_evaluation"])

    def test_a_judge_run_before_the_freeze_is_reported_as_not_frozen_first(self):
        registry = human_registry()
        _, report = passing_report(registry)
        measurements = gate_d_measurements(
            report,
            registry,
            freeze_created_at="2026-09-19T02:00:00.000Z",
            judge_started_at="2026-09-19T01:00:00.000Z",
        )
        self.assertFalse(measurements["held_out_frozen_before_evaluation"])

    def test_missing_freeze_timestamps_leave_the_field_null_rather_than_true(self):
        registry = human_registry()
        _, report = passing_report(registry)
        self.assertIsNone(
            gate_d_measurements(report, registry)["held_out_frozen_before_evaluation"]
        )

    def test_the_gate_d_record_passes_the_gate_ledger_evidence_checks_with_humans(self):
        registry = human_registry()
        clips, report = passing_report(registry)
        bundle = gate_d_evidence(
            report,
            registry,
            judge_evidence=panel.gate_d_evidence_bundle(
                {
                    "status": "pass",
                    "calibration_manifest_hash": calibration_manifest_hash(clips),
                    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "model_revision": "0" * 40,
                },
                {"status": "completed", "reports": [], "failed_clips": 0, "scored_clips": 50},
                full_manifest(),
                evidence_uris=["file:///evidence/judge_reports.jsonl"],
                protocol_hash="sha256:" + "b" * 64,
                calibration_class="human",
                human_annotation_present=True,
            ),
            evidence_uris=["file:///evidence/judge_reports.jsonl"],
            protocol_hash="sha256:" + "b" * 64,
            freeze_created_at="2026-09-19T00:00:00.000Z",
            judge_started_at="2026-09-19T01:00:00.000Z",
        )
        self.assertEqual("pass", bundle["status"])
        record = GateRecord.from_mapping(bundle)
        self.assertEqual("D", record.gate_id)
        self.assertEqual((), record.pass_evidence_errors())

    def test_the_gate_d_record_is_blocked_without_human_annotation(self):
        registry = surrogate_registry()
        _, report = passing_report(registry)
        bundle = gate_d_evidence(
            report,
            registry,
            evidence_uris=["file:///evidence/judge_reports.jsonl"],
            protocol_hash="sha256:" + "b" * 64,
        )
        self.assertEqual("blocked", bundle["status"])
        self.assertTrue(any(HUMAN_ANNOTATION_DEPENDENCY in item for item in bundle["reasons"]))
        self.assertEqual("pass_with_limitations", bundle["measurements"]["gate_d_decision_status"])
        record = GateRecord.from_mapping(bundle)
        self.assertEqual("blocked", record.status.value)


class CalibrationJsonWriterTests(unittest.TestCase):
    def test_the_writer_is_atomic_and_leaves_no_partial_file(self):
        registry = surrogate_registry()
        clips, report = passing_report(registry)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "results" / "judge_calibration.json"
            result = write_judge_calibration_json(
                path, report, registry, manifest=clips, manifest_id="cal-v1"
            )
            self.assertEqual(str(path), result["path"])
            self.assertTrue(path.is_file())
            self.assertEqual([], [item for item in path.parent.iterdir() if item.name.startswith(".")])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("external_label + model_reference", payload["calibration_class"])
            # A second write replaces atomically without leaving a temporary file.
            previous = path.read_bytes()
            write_judge_calibration_json(path, report, registry, manifest=clips, manifest_id="cal-v2")
            self.assertEqual(1, len(list(path.parent.iterdir())))
            self.assertNotEqual(previous, path.read_bytes())

    def test_a_write_failure_leaves_the_previous_file_intact(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "out.json"
            write_json_atomic(path, {"value": 1})
            with self.assertRaises(AnnotationError):
                write_json_atomic(path, {"value": float("nan")})
            self.assertEqual({"value": 1}, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual([path.name], [item.name for item in path.parent.iterdir()])

    def test_the_writer_never_emits_nan_or_infinity(self):
        with self.assertRaises(AnnotationError) as raised:
            assert_finite_json({"metrics": {"kappa": float("nan")}})
        self.assertIn("must be null", str(raised.exception))
        with self.assertRaises(AnnotationError):
            assert_finite_json([{"sensitivity": float("inf")}])
        assert_finite_json({"kappa": None, "sensitivity": 1.0})

    def test_the_payload_records_every_required_statistic_and_disclosure(self):
        registry = surrogate_registry()
        clips, report = passing_report(registry)
        payload = build_judge_calibration_payload(
            report, registry, manifest=clips, annotation_rows=label_rows(clips, registry), manifest_id="cal-v1"
        )
        self.assertEqual("external_label + model_reference", payload["calibration_class"])
        self.assertEqual(150, len(payload["split_ids"]["development_clip_ids"]) + 50)
        self.assertEqual(50, len(payload["split_ids"]["heldout_clip_ids"]))
        self.assertEqual(200, payload["annotation_count"])
        self.assertIsNotNone(payload["cohens_kappa"]["binary_success"])
        self.assertIn("weighted_kappa", payload["progress_quadratic_weighted_kappa"])
        self.assertEqual("quadratic", payload["progress_quadratic_weighted_kappa"]["weighting"])
        self.assertIsNotNone(payload["confusion_matrix"])
        self.assertEqual(1.0, payload["sensitivity"])
        self.assertEqual(1.0, payload["specificity"])
        self.assertEqual(0.0, payload["leniency_offset"]["value"])
        self.assertTrue(payload["leniency_offset"]["equals_false_positive_minus_false_negative_over_comparable"])
        self.assertIsNotNone(payload["uncertainty"]["sensitivity_interval"])
        self.assertIsNotNone(payload["uncertainty"]["leniency_offset_interval"])
        self.assertIn("not population estimates", payload["stratification_disclosure"])
        self.assertIn("not a mathematical ceiling", payload["human_human_agreement"]["interpretation"])
        self.assertNotIn("ceiling on", payload["human_human_agreement"]["interpretation"].split("not a")[0])
        self.assertFalse(payload["qualified"])
        self.assertEqual(
            {"clip_id", "annotator_id", "annotator_type"}, set(payload["annotation_ids"][0])
        )

    def test_a_surrogate_annotator_is_never_recorded_as_human_in_the_report(self):
        registry = surrogate_registry()
        clips, report = passing_report(registry)
        payload = build_judge_calibration_payload(
            report, registry, manifest=clips, annotation_rows=label_rows(clips, registry)
        )
        types = {row["annotator_type"] for row in payload["annotation_ids"]}
        self.assertNotIn(HUMAN_ANNOTATOR, types)
        self.assertEqual({EXTERNAL_LABEL_ANNOTATOR, MODEL_ANNOTATOR}, types)
        self.assertFalse(payload["human_annotation_present"])
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn('"annotator_type": "human"', encoded)

    def test_an_artifact_store_hook_reuses_the_shared_writer(self):
        class FakeStore:
            def __init__(self):
                self.payload = None

            def write_judge_calibration(self, payload):
                self.payload = payload
                return "receipt"

        from plumb.annotation import write_judge_calibration_artifact

        registry = surrogate_registry()
        _, report = passing_report(registry)
        store = FakeStore()
        self.assertEqual("receipt", write_judge_calibration_artifact(store, report, registry))
        self.assertEqual("external_label + model_reference", store.payload["calibration_class"])
        with self.assertRaises(AnnotationError):
            write_judge_calibration_artifact(object(), report, registry)


class UncertaintyTests(unittest.TestCase):
    def test_wilson_interval_is_none_without_a_denominator(self):
        self.assertIsNone(wilson_interval(0, 0))
        interval = wilson_interval(9, 10)
        self.assertLess(interval["lower"], 0.9)
        self.assertGreater(interval["upper"], 0.9)
        self.assertTrue(all(math.isfinite(interval[key]) for key in ("point", "lower", "upper")))
        self.assertEqual(0.0, wilson_interval(0, 10)["point"])

    def test_paired_difference_interval_covers_zero_when_discordance_is_balanced(self):
        interval = paired_difference_interval(5, 5, 50)
        self.assertEqual(0.0, interval["point"])
        self.assertLess(interval["lower"], 0.0)
        self.assertGreater(interval["upper"], 0.0)
        self.assertEqual(10, interval["discordant_pairs"])
        self.assertIsNone(paired_difference_interval(0, 0, 0))

    def test_interval_bounds_never_escape_the_unit_or_signed_unit_range(self):
        self.assertGreaterEqual(wilson_interval(1, 1)["lower"], 0.0)
        self.assertLessEqual(wilson_interval(1, 1)["upper"], 1.0)
        interval = paired_difference_interval(50, 0, 50)
        self.assertLessEqual(interval["upper"], 1.0)
        self.assertGreaterEqual(interval["lower"], -1.0)


class PanelDriverTests(unittest.TestCase):
    def _pool(self, per_task=40):
        policies = ("OpenVLA", "OpenPiZero", "Octo", "MiniVLA", "SuSIE", "SuSIE_LL")
        outcomes = ("apparent_success", "apparent_failure", "unknown")
        rows = []
        for task in TASKS:
            for index in range(per_task):
                policy = policies[index % len(policies)]
                rows.append(
                    {
                        "task": task,
                        "policy": policy,
                        "apparent_outcome": outcomes[index % len(outcomes)],
                        "generation_defect": index % 7 == 0,
                        "source_lineage_id": "calib:%s:%s:ep-%04d" % (task, policy, index),
                        "media_ref": "/generated/%s/%s-%04d.mp4" % (policy, task, index),
                    }
                )
        return rows

    def test_selection_is_deterministic_and_matches_the_frozen_plan(self):
        first = panel.select_panel(self._pool())
        second = panel.select_panel(self._pool())
        self.assertEqual("pass", first["status"], first["reason"])
        self.assertEqual(150, first["selected_clips"])
        self.assertEqual(first["manifest_rows"], second["manifest_rows"])
        self.assertEqual(first["calibration_manifest_hash"], second["calibration_manifest_hash"])
        clips = validate_manifest(first["manifest_rows"])
        self.assertEqual(100, sum(1 for clip in clips if clip.split == DEVELOPMENT_SPLIT))
        self.assertEqual(50, sum(1 for clip in clips if clip.split == HELDOUT_SPLIT))
        for task in TASKS:
            self.assertEqual(20, sum(1 for clip in clips if clip.task == task and clip.split == DEVELOPMENT_SPLIT))
            self.assertEqual(10, sum(1 for clip in clips if clip.task == task and clip.split == HELDOUT_SPLIT))

    def test_selected_clip_ids_and_media_refs_carry_no_policy_name(self):
        result = panel.select_panel(self._pool())
        # The operator manifest legitimately keeps split and lineage, and those
        # lineages do name a policy. Only clip_id and media_ref reach the
        # annotator, so those are the two fields that must be opaque.
        self.assertTrue(
            any("OpenVLA" in row["source_lineage_id"] for row in result["manifest_rows"])
        )
        for row in result["manifest_rows"]:
            assert_blinded({"clip_id": row["clip_id"], "media_ref": row["media_ref"]})
            self.assertTrue(row["clip_id"].startswith("clip-"))
            self.assertTrue(row["media_ref"].startswith("artifact://calibration/"))
        clips = validate_manifest(result["manifest_rows"])
        for clip in clips:
            assert_blinded(blinded_clip_view(clip, "human:alex"))

    def test_selection_covers_policies_outcomes_and_generation_defects(self):
        result = panel.select_panel(self._pool())
        coverage = result["strata_coverage"]["open_drawer"]
        self.assertGreater(coverage["distinct_strata"], 1)
        drawn = [cell for cell in coverage["strata"] if cell["drawn"] > 0]
        self.assertGreater(len({cell["policy"] for cell in drawn}), 1)
        self.assertGreater(len({cell["apparent_outcome"] for cell in drawn}), 1)
        self.assertIn(True, {cell["generation_defect"] for cell in drawn})

    def test_a_thin_pool_reports_a_shortfall_instead_of_reusing_a_lineage(self):
        result = panel.select_panel(self._pool(per_task=5))
        self.assertEqual("insufficient_candidate_pool", result["status"])
        self.assertIn("rather than reusing a lineage", result["reason"])
        self.assertTrue(result["shortfall"])

    def test_lineages_reserved_for_other_cohorts_are_excluded(self):
        pool_rows = self._pool()
        reserved = pool_rows[0]["source_lineage_id"]
        result = panel.select_panel(pool_rows, excluded_lineages=[reserved])
        self.assertNotIn(reserved, {row["source_lineage_id"] for row in result["manifest_rows"]})
        self.assertEqual(
            "lineage_reserved_for_another_cohort", result["rejected_candidates"][0]["reason"]
        )

    def test_a_duplicate_lineage_in_the_pool_is_rejected_not_counted_twice(self):
        pool_rows = self._pool()
        clone = dict(pool_rows[0])
        clone["media_ref"] = "/generated/other/clone.mp4"
        result = panel.select_panel(pool_rows + [clone])
        reasons = {item["reason"] for item in result["rejected_candidates"]}
        self.assertIn("duplicate_source_lineage_in_pool", reasons)

    def test_judge_seeds_are_five_distinct_logged_integers_per_clip(self):
        seeds = panel.judge_seeds(20260919, "clip-abc")
        self.assertEqual(5, len(seeds))
        self.assertEqual(5, len(set(seeds)))
        self.assertTrue(all(isinstance(seed, int) and seed >= 0 for seed in seeds))
        self.assertEqual(seeds, panel.judge_seeds(20260919, "clip-abc"))
        self.assertNotEqual(seeds, panel.judge_seeds(20260919, "clip-abd"))
        self.assertNotEqual(seeds, panel.judge_seeds(1, "clip-abc"))

    def test_the_batch_judge_is_a_dry_run_without_execute(self):
        result = panel.select_panel(self._pool())
        with TemporaryDirectory() as temporary:
            media = Path(temporary) / "clips"
            media.mkdir()
            for row in result["manifest_rows"]:
                (media / Path(row["media_ref"]).name).write_bytes(b"clip:" + row["clip_id"].encode())
            frozen = panel.freeze_evidence(
                result["manifest_rows"],
                protocol_id="protocol-v1",
                model_revision="c" * 40,
                processor_revision="d" * 40,
                transformers_version="4.49.0",
                runtime_lock_id="runtime-lock-v1",
                asset_manifest_id="assets-lock-v1",
                media_root=media,
            )
            self.assertEqual("pass", frozen["status"])
            self.assertEqual(50, len(frozen["heldout_video_hashes"]))
            self.assertEqual({}, frozen["heldout_artifact_hashes"])
            self.assertIn("cannot be preregistered", frozen["artifact_hash_note"])
            reference = Path(temporary) / "goal.png"
            reference.write_bytes(b"goal")
            run = panel.run_batch_judge(
                result["manifest_rows"],
                frozen,
                media_root=media,
                references={task: reference for task in TASKS},
                model_path=Path(temporary) / "model",
                base_seed=20260919,
                execute=False,
            )
            self.assertEqual("planned", run["status"])
            self.assertEqual(50, len(run["clip_plan"]))
            self.assertEqual([], run["blockers"])
            self.assertEqual([], run["reports"])
            self.assertEqual(dict(panel.PRIMARY_SAMPLING), run["sampling"])
            self.assertIn("never padded", run["frame_sampling"])

    def test_freeze_blocks_when_heldout_media_is_absent(self):
        result = panel.select_panel(self._pool())
        with TemporaryDirectory() as temporary:
            frozen = panel.freeze_evidence(
                result["manifest_rows"],
                protocol_id="protocol-v1",
                model_revision="c" * 40,
                processor_revision="d" * 40,
                transformers_version="4.49.0",
                runtime_lock_id="runtime-lock-v1",
                asset_manifest_id="assets-lock-v1",
                media_root=Path(temporary) / "missing",
            )
        self.assertEqual("blocked", frozen["status"])
        self.assertEqual(50, len(frozen["missing_heldout_media"]))

    def test_the_batch_judge_runs_against_a_fake_judge_and_binds(self):
        clips = validate_manifest(full_manifest())
        frozen_record = {
            "calibration_manifest_hash": calibration_manifest_hash(clips),
            "heldout_video_hashes": {},
            "protocol_id": "protocol-v1",
            "model_revision": "c" * 40,
            "processor_revision": "d" * 40,
            "transformers_version": "4.49.0",
            "runtime_lock_id": "runtime-lock-v1",
            "asset_manifest_id": "assets-lock-v1",
            "created_at": "2026-09-19T00:00:00.000Z",
        }
        frozen_object = frozen_manifest(clips)
        captured = {}

        class FakeJudgeReport:
            def __init__(self, clip_id):
                self.clip_id = clip_id

            def as_dict(self):
                clip = next(item for item in clips if item.clip_id == self.clip_id)
                return dict(raw_judge_report(clip, frozen_object, success=_is_success(self.clip_id)))

        class FakeJudge:
            def evaluate(self, request, *, seeds):
                captured.setdefault("seeds", []).append(tuple(seeds))
                captured.setdefault("frames", []).append(len(request["frames"]))
                return FakeJudgeReport(request["clip_id"])

        def fake_dependencies():
            def profile(**kwargs):
                return kwargs

            def request(**kwargs):
                return {
                    "frames": kwargs["frames"],
                    "clip_id": kwargs["provenance"]["clip_id"],
                    "task_id": kwargs["task_id"],
                }

            def reference(**kwargs):
                return kwargs

            def provenance(**kwargs):
                return kwargs

            return profile, (lambda _profile: FakeJudge()), request, reference, provenance

        with TemporaryDirectory() as temporary:
            media = Path(temporary) / "clips"
            media.mkdir()
            for clip in clips:
                (media / Path(clip.media_ref).name).write_bytes(b"clip:" + clip.clip_id.encode())
            reference = Path(temporary) / "goal.png"
            reference.write_bytes(b"goal")
            run = panel.run_batch_judge(
                full_manifest(),
                frozen_record,
                media_root=media,
                references={task: reference for task in TASKS},
                model_path=Path(temporary) / "model",
                base_seed=20260919,
                execute=True,
                video_loader=lambda path: tuple(range(24)),
                image_loader=lambda path: "image",
                dependency_loader=fake_dependencies,
                judge_factory=lambda profile, judge_cls: FakeJudge(),
            )
        self.assertEqual("completed", run["status"], run.get("failures"))
        self.assertEqual(50, run["scored_clips"])
        self.assertEqual(0, run["failed_clips"])
        self.assertTrue(all(count == 16 for count in captured["frames"]))
        self.assertTrue(all(len(set(item)) == 5 for item in captured["seeds"]))
        bound = panel.bind_evidence(frozen_record, run["reports"], full_manifest())
        self.assertEqual("pass", bound["status"], bound["binding_problems"])
        self.assertEqual(50, len(bound["heldout_artifact_hashes"]))
        self.assertIn("not a cryptographic attestation", bound["binding_semantics"])

    def test_binding_refuses_a_report_for_a_development_clip(self):
        clips = validate_manifest(full_manifest())
        frozen_object = frozen_manifest(clips)
        development = next(clip for clip in clips if clip.split == DEVELOPMENT_SPLIT)
        heldout = next(clip for clip in clips if clip.split == HELDOUT_SPLIT)
        report = raw_judge_report(heldout, frozen_object, success=False)
        report["provenance"] = dict(report["provenance"])
        report["provenance"]["clip_id"] = development.clip_id
        bound = panel.bind_evidence({"status": "pass"}, [report], full_manifest())
        self.assertEqual("blocked", bound["status"])
        reasons = {item["reason"] for item in bound["binding_problems"]}
        self.assertIn("report_clip_is_not_in_the_heldout_split", reasons)

    def test_the_panel_gate_d_bundle_is_never_pass_on_its_own(self):
        bundle = panel.gate_d_evidence_bundle(
            {"status": "pass", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
            {"status": "completed", "reports": [], "failed_clips": 0, "scored_clips": 50},
            full_manifest(),
            evidence_uris=["file:///evidence/judge.jsonl"],
            protocol_hash="sha256:" + "b" * 64,
            calibration_class="external_label + model_reference",
            human_annotation_present=False,
        )
        self.assertEqual("blocked", bundle["status"])
        self.assertTrue(any(HUMAN_ANNOTATION_DEPENDENCY in item for item in bundle["reasons"]))
        self.assertEqual(50, bundle["measurements"]["heldout_clips"])
        self.assertEqual(100, bundle["measurements"]["development_clips"])
        self.assertEqual(2, bundle["thresholds"]["required_blinded_human_annotators"])


class CliTests(unittest.TestCase):
    def test_register_queue_submit_and_report_run_end_to_end(self):
        from plumb.annotation import main

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "calibration_manifest.jsonl"
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in full_manifest()), encoding="utf-8"
            )
            registry_path = root / "annotators.json"
            store_path = root / "labels.jsonl"
            self.assertEqual(
                0,
                main(
                    [
                        "register",
                        "--output",
                        str(registry_path),
                        "--annotator",
                        json.dumps(
                            {
                                "annotator_id": "external:autoeval",
                                "annotator_type": EXTERNAL_LABEL_ANNOTATOR,
                                "label_source": "AutoEval classifier labels",
                                "source_uri": "hf://datasets/zhouzypaul/auto_eval",
                            }
                        ),
                        "--annotator",
                        json.dumps(
                            {
                                "annotator_id": "model:qwen",
                                "annotator_type": MODEL_ANNOTATOR,
                                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                                "model_revision": "c" * 40,
                            }
                        ),
                    ]
                ),
            )
            shared = [
                "--manifest",
                str(manifest_path),
                "--annotators",
                str(registry_path),
                "--store",
                str(store_path),
            ]
            self.assertEqual(0, main(["session", *shared, "--annotator-id", "external:autoeval"]))
            self.assertEqual(0, main(["status", *shared]))
            registry = AnnotatorRegistry.load(registry_path)
            service = AnnotationService(full_manifest(), registry, AnnotationStore(store_path))
            clip_id = service.session("external:autoeval")["next_clip_id"]
            self.assertEqual(
                0,
                main(
                    [
                        "submit",
                        *shared,
                        "--annotator-id",
                        "external:autoeval",
                        "--clip-id",
                        clip_id,
                        "--integrity",
                        "intact",
                        "--collision",
                        "none_visible",
                        "--progress",
                        "2",
                        "--completion-evidence",
                        "not_met",
                        "--frames",
                        "[0,15]",
                        "--reason",
                        "Incomplete at the final frame.",
                    ]
                ),
            )
            output = root / "results" / "judge_calibration.json"
            self.assertEqual(0, main(["report", *shared, "--output", str(output)]))
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("external_label + model_reference", payload["calibration_class"])
            self.assertEqual(1, payload["annotation_count"])
            self.assertEqual(
                [HUMAN_ANNOTATION_DEPENDENCY], payload["gate_d"]["open_dependency_names"]
            )
            self.assertFalse(payload["gate_d"]["passed"])


if __name__ == "__main__":
    unittest.main()
