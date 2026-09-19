"""Offline tests for judge distillation on Baseten Training Jobs.

No torch, no network, no GPU, no training.  Every test here asserts a refusal
that spec section 5's distillation paragraph requires, or asserts that the
``distilled_judge`` evidence mapping is exactly what ``plumb.gates`` Gate E
reads.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from plumb.calibration import (
    AnnotatorOwnership,
    DEVELOPMENT_SPLIT,
    HELDOUT_SPLIT,
    FrozenGateDProtocol,
    FrozenJudgeEvidenceManifest,
    GateDTolerances,
    TrustedJudgeProducer,
    calibration_manifest_hash,
    deterministic_annotation_assignments,
    validate_manifest,
)
from plumb.distillation import (
    ARM_STATUSES,
    BUDGET_ITEM,
    PUBLISHED_LORA_ARM,
    REPO_DEFAULT_LORA_ARM,
    RUBRIC_JSON_CONTRACT,
    SCALAR_HEAD_CONTRACT,
    SELECTION_SPLIT,
    ArmResult,
    BaseJudgeRevision,
    ClipLineage,
    DistillationDataset,
    DistillationError,
    DistillationGate,
    DistilledJudgeRevision,
    EarlyStoppingRule,
    FrozenVideoScore,
    LineageExclusions,
    LoRAConfig,
    LoRASearchSpace,
    PairedComparisonTolerances,
    StudentModel,
    TrainingBudgetItem,
    TrainingJobSpec,
    TrainingRuntime,
    assert_fresh_heldout_panel,
    assert_scoring_authorised,
    compare_frozen_videos,
    dataset_from_calibration_manifest,
    default_search_space,
    frozen_sampling_for_distilled_judge,
    render_search_payloads,
    run_fresh_heldout_calibration,
    select_arm,
    spoken_claim,
    unvalidated_distilled_judge_evidence,
)
from plumb.gates import GateRecord, GateStatus
from plumb.sweeps import GpuSecondModel


TASKS = ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")
ANNOTATORS = ("annotator-alex", "annotator-blair")
PREREGISTERED_AT = "2026-09-19T00:00:00+00:00"


def sha256(label):
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def full_manifest(panel="original"):
    """A complete 150-clip panel whose clip IDs and lineages carry a panel tag."""

    rows = []
    for task_index, task in enumerate(TASKS):
        for split, count in ((DEVELOPMENT_SPLIT, 20), (HELDOUT_SPLIT, 10)):
            for index in range(count):
                clip_id = "{0}-clip-{1}-{2}-{3:02d}".format(panel, task_index, split[:3], index)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "media_ref": "artifact://{0}/{1}.mp4".format(panel, clip_id),
                        "task": task,
                        "split": split,
                        "source_lineage_id": "{0}-{1}-{2}-{3:02d}".format(panel, split, task_index, index),
                    }
                )
    return rows


def heldout_lineages(panel="original"):
    return tuple(
        row["source_lineage_id"] for row in full_manifest(panel) if row["split"] == HELDOUT_SPLIT
    )


def primary_lineages():
    return tuple("primary-lineage-{0:03d}".format(index) for index in range(10))


def exclusions(fresh=()):
    return LineageExclusions(
        gate_d_heldout=heldout_lineages("original"),
        primary_study=primary_lineages(),
        fresh_heldout=tuple(fresh),
    )


def budget(**overrides):
    values = {
        "label_count": 100,
        "label_source": "original_judge_aggregate",
        "usd_unavailable_reason": "no verified price basis for the training accelerator yet",
    }
    values.update(overrides)
    return TrainingBudgetItem(**values)


def dataset(panel="original", **overrides):
    values = {
        "dataset_id": "distill-dataset-v1",
        "exclusions": exclusions(),
        "label_source": "original_judge_aggregate",
        "budget": budget(),
        "storage_uri": "s3://plumb-distillation/dataset-v1/",
        "frozen_at": PREREGISTERED_AT,
    }
    values.update(overrides)
    return dataset_from_calibration_manifest(full_manifest(panel), **values)


def search():
    return default_search_space(search_id="lora-search-v1", preregistered_at=PREREGISTERED_AT)


def arm_results(space, *, winner="published-lr1e-3-r64-a32-e2", losers_fail=False):
    """One recorded result per preregistered arm, winner included."""

    results = []
    for index, arm in enumerate(space.arms):
        if arm.arm_id == winner:
            results.append(
                ArmResult(
                    arm_id=arm.arm_id,
                    status="early_stopped",
                    selection_metric=space.selection_metric,
                    metric_value=0.41,
                    metric_source="baseten_training_job_log",
                    training_job_id="job-{0}".format(arm.arm_id),
                    adapter_uri="s3://plumb-distillation/adapters/{0}/".format(arm.arm_id),
                    adapter_sha256=sha256("adapter:" + arm.arm_id),
                    epochs_completed=2,
                    early_stopped_at_epoch=2,
                    reported_gpu_seconds=1820.5,
                )
            )
        elif losers_fail:
            results.append(
                ArmResult(
                    arm_id=arm.arm_id,
                    status="failed",
                    selection_metric=space.selection_metric,
                    reason="training container exited on an out-of-memory error",
                )
            )
        else:
            results.append(
                ArmResult(
                    arm_id=arm.arm_id,
                    status="completed",
                    selection_metric=space.selection_metric,
                    metric_value=0.5 + 0.01 * index,
                    metric_source="baseten_training_job_log",
                    training_job_id="job-{0}".format(arm.arm_id),
                    adapter_uri="s3://plumb-distillation/adapters/{0}/".format(arm.arm_id),
                    adapter_sha256=sha256("adapter:" + arm.arm_id),
                    epochs_completed=2,
                    reported_gpu_seconds=1755.25,
                )
            )
    return tuple(results)


def base_judge():
    return BaseJudgeRevision(
        revision_id="qwen2.5-vl-7b-rev-a",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        model_revision="0" * 40,
        rubric_hash=sha256("rubric"),
        sampling_hash=sha256("sampling"),
        gate_d_protocol_hash=sha256("gate-d-protocol"),
        results_ref="results/judge_calibration.json",
        results_sha256=sha256("original-judge-results"),
    )


def student(contract=RUBRIC_JSON_CONTRACT):
    return StudentModel(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        model_revision="0" * 40,
        output_contract=contract,
        parameter_note="LoRA adapter over the frozen base; the rubric schema is unchanged.",
    )


def revision(space, data, *, arm_id="published-lr1e-3-r64-a32-e2", contract=RUBRIC_JSON_CONTRACT, **overrides):
    values = {
        "revision_id": "qwen2.5-vl-7b-distilled-rev-b",
        "base_judge": base_judge(),
        "student": student(contract),
        "adapter_uri": "s3://plumb-distillation/adapters/{0}/".format(arm_id),
        "adapter_sha256": sha256("adapter:" + arm_id),
        "training_dataset_hash": data.content_hash(),
        "search_arm_id": arm_id,
        "search_preregistration_hash": space.preregistration_hash(),
        "training_job_id": "job-" + arm_id,
        "rubric_hash": sha256("rubric"),
        "sampling": frozen_sampling_for_distilled_judge(),
        "results_ref": "results/judge_calibration_distilled_rev_b.json",
        "created_at": PREREGISTERED_AT,
    }
    values.update(overrides)
    return DistilledJudgeRevision(**values)


def runtime(**overrides):
    values = {
        "image": "baseten/training-py311-cu124:2026-09-19",
        "python_version": "3.11",
        "framework": "trl",
        "requirements_ref": "deploy/baseten/training/requirements.txt",
        "start_command": "python train_judge_lora.py --config job.json",
        "accelerator": "H100",
        "environment_variable_names": ("PLUMB_DISTILL_CONFIG", "PLUMB_DISTILL_OUTPUT_DIR"),
        "secret_names": ("hf_access_token",),
    }
    values.update(overrides)
    return TrainingRuntime(**values)


# -- calibration fixtures for the fresh held-out panel ---------------------


def is_success(clip):
    """Half the panel succeeds so sensitivity and specificity are both defined."""

    return clip.clip_id.endswith(("0", "2", "4", "6", "8"))


def human_labels(clips):
    assignments = deterministic_annotation_assignments(clips, ANNOTATORS)
    rows = []
    for clip in clips:
        for annotator_id in assignments[clip.clip_id]:
            success = is_success(clip)
            rows.append(
                {
                    "clip_id": clip.clip_id,
                    "annotator_id": annotator_id,
                    "integrity": "intact",
                    "collision": "none_visible",
                    "progress": 5 if success else 2,
                    "completion_evidence": "met" if success else "not_met",
                    "evidence_frame_indices": [0, 15],
                    "observable_reason": "Visible end state.",
                }
            )
    return rows


def evidence_manifest(clips, *, protocol_id):
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
        runtime_lock_id="runtime-lock-v1",
        asset_manifest_id="assets-lock-v1",
        heldout_artifact_hashes={
            clip.clip_id: {
                "video": sha256("artifact-video:" + clip.clip_id),
                "frame_0": sha256("artifact-frame:" + clip.clip_id),
            }
            for clip in heldout
        },
        protocol_id=protocol_id,
        trusted_producer=TrustedJudgeProducer("plumb-judge", "1.0.0", sha256("producer-source")),
    )


def judge_report(clip, frozen, *, rubric_hash, sampling_hash):
    success = is_success(clip)
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
            "rubric_hash": rubric_hash,
            "sampling_hash": sampling_hash,
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


DISTILLED_PROTOCOL = FrozenGateDProtocol(
    protocol_hash=sha256("distilled-gate-d-protocol"),
    rubric_hash=sha256("rubric"),
    sampling_hash=sha256("distilled-sampling"),
)


def fresh_calibration_inputs():
    """Everything ``run_fresh_heldout_calibration`` needs for a passing panel."""

    fresh = validate_manifest(full_manifest("fresh"))
    frozen = evidence_manifest(fresh, protocol_id="protocol-distilled-v1")
    protocol = FrozenGateDProtocol(
        protocol_hash=DISTILLED_PROTOCOL.protocol_hash,
        rubric_hash=DISTILLED_PROTOCOL.rubric_hash,
        sampling_hash=DISTILLED_PROTOCOL.sampling_hash,
        evidence_manifest=frozen,
    )
    reports = [
        judge_report(
            clip,
            frozen,
            rubric_hash=protocol.rubric_hash,
            sampling_hash=protocol.sampling_hash,
        )
        for clip in fresh
        if clip.split == HELDOUT_SPLIT
    ]
    return {
        "calibration_id": "fresh-heldout-v1",
        "fresh_manifest": full_manifest("fresh"),
        "original_manifest": full_manifest("original"),
        "human_annotations": human_labels(fresh),
        "annotator_ids": ANNOTATORS,
        "distilled_judge_annotations": None,
        "distilled_judge_reports": reports,
        "protocol": protocol,
        "tolerances": GateDTolerances(50, 1.0, 0.9, 0.9, 0.9, 0.9, 0.1),
        "base_judge": base_judge(),
        "annotator_ownership": {
            ANNOTATORS[0]: AnnotatorOwnership(
                annotator_id=ANNOTATORS[0],
                owner_id="human-owner-alex",
                independent_attestation=True,
                attested_at=PREREGISTERED_AT,
            ),
            ANNOTATORS[1]: AnnotatorOwnership(
                annotator_id=ANNOTATORS[1],
                owner_id="human-owner-blair",
                independent_attestation=True,
                attested_at=PREREGISTERED_AT,
            ),
        },
    }


def passing_fresh_calibration():
    result, _ = run_fresh_heldout_calibration(**fresh_calibration_inputs())
    return result


def frozen_video_scores(*, revision_id, flip_clips=(), unevaluable=()):
    """Scores over the fresh held-out clips, with identical frozen frames."""

    fresh = validate_manifest(full_manifest("fresh"))
    rows = []
    for clip in (clip for clip in fresh if clip.split == HELDOUT_SPLIT):
        success = is_success(clip)
        if clip.clip_id in flip_clips:
            success = not success
        rows.append(
            FrozenVideoScore(
                clip_id=clip.clip_id,
                judge_revision=revision_id,
                video_sha256=sha256("video:" + clip.clip_id),
                frame_hashes=tuple(sha256("frame:{0}:{1}".format(clip.clip_id, index)) for index in range(16)),
                binary_success=None if clip.clip_id in unevaluable else success,
                progress=None if clip.clip_id in unevaluable else (5 if success else 2),
                missing_reason="judge disagreement" if clip.clip_id in unevaluable else None,
            )
        )
    return tuple(rows)


def paired_tolerances(**overrides):
    values = {
        "minimum_paired_clips": 40,
        "maximum_absolute_binary_difference": 0.1,
        "maximum_progress_disagreement_rate": 0.2,
        "minimum_paired_coverage": 0.9,
    }
    values.update(overrides)
    return PairedComparisonTolerances(**values)


def passing_paired_comparison():
    return compare_frozen_videos(
        comparison_id="paired-frozen-video-v1",
        original=frozen_video_scores(revision_id="qwen2.5-vl-7b-rev-a"),
        distilled=frozen_video_scores(
            revision_id="qwen2.5-vl-7b-distilled-rev-b",
            flip_clips=("fresh-clip-0-hel-01",),
        ),
        tolerances=paired_tolerances(),
    )


def complete_gate(**overrides):
    space = search()
    data = dataset()
    results = arm_results(space)
    values = {
        "revision": revision(space, data),
        "search": space,
        "arm_results": results,
        "selection": select_arm(space, results),
        "dataset": data,
        "fresh_calibration": passing_fresh_calibration(),
        "paired_comparison": passing_paired_comparison(),
        "requested_for_scoring": True,
    }
    values.update(overrides)
    return DistillationGate(**values)


def gate_e_record(distilled_evidence, status=GateStatus.PASS):
    """A Gate E record that is complete apart from the distillation evidence."""

    return GateRecord(
        gate_id="E",
        status=status,
        protocol_hash=sha256("protocol"),
        evidence_uris=("artifact://results/reliability.json",),
        thresholds={"max_cell_rate_error": 0.1, "min_coverage": 0.9},
        evidence_kind="execution_record",
        measurements={
            "analysis_protocol_frozen": True,
            "planned_episodes": 1500,
            "terminal_episodes": 1500,
            "exclusion_sensitivity": {"status": "computed", "max_rate_shift": 0.02},
            "cost_setting_confirmation": {"confirmed_on_disjoint_panel": True},
            "distilled_judge": distilled_evidence,
        },
    )


class ImportHygieneTests(unittest.TestCase):
    def test_distillation_is_importable_without_torch(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_arm_statuses_start_at_not_run(self):
        self.assertEqual("not_run", ARM_STATUSES[0])


class LineageDisjointnessTests(unittest.TestCase):
    def test_a_training_lineage_from_the_heldout_set_is_refused(self):
        stolen = heldout_lineages("original")[0]
        with self.assertRaises(DistillationError) as caught:
            DistillationDataset(
                dataset_id="leaky",
                train=(ClipLineage("c-1", stolen, "open_drawer"),),
                development_validation=(ClipLineage("c-2", "dev-lineage-2", "open_drawer"),),
                exclusions=exclusions(),
                label_source="original_judge_aggregate",
                budget=budget(),
                storage_uri="s3://plumb-distillation/leaky/",
                frozen_at=PREREGISTERED_AT,
            )
        self.assertIn("not disjoint", str(caught.exception))

    def test_a_training_lineage_from_the_primary_study_is_refused(self):
        with self.assertRaises(DistillationError):
            DistillationDataset(
                dataset_id="leaky-primary",
                train=(ClipLineage("c-1", primary_lineages()[0], "open_drawer"),),
                development_validation=(ClipLineage("c-2", "dev-lineage-2", "open_drawer"),),
                exclusions=exclusions(),
                label_source="original_judge_aggregate",
                budget=budget(),
                storage_uri="s3://plumb-distillation/leaky/",
                frozen_at=PREREGISTERED_AT,
            )

    def test_a_validation_lineage_reused_in_training_is_refused(self):
        with self.assertRaises(DistillationError):
            DistillationDataset(
                dataset_id="leaky-validation",
                train=(ClipLineage("c-1", "shared-lineage", "open_drawer"),),
                development_validation=(ClipLineage("c-2", "shared-lineage", "open_drawer"),),
                exclusions=exclusions(),
                label_source="original_judge_aggregate",
                budget=budget(),
                storage_uri="s3://plumb-distillation/leaky/",
                frozen_at=PREREGISTERED_AT,
            )

    def test_exclusions_cannot_be_left_unknown(self):
        with self.assertRaises(DistillationError):
            LineageExclusions(gate_d_heldout=(), primary_study=primary_lineages())
        with self.assertRaises(DistillationError):
            LineageExclusions(gate_d_heldout=heldout_lineages(), primary_study=())

    def test_the_disjointness_decision_is_delegated_to_the_primary_checker(self):
        report = dataset().lineage_leakage_report()
        self.assertEqual("pass", report["status"])
        self.assertEqual("plumb.measurement.validate_source_lineage_leakage", report["checker"])

    def test_heldout_clips_are_never_training_data(self):
        data = dataset()
        training_clips = {row.clip_id for row in data.train + data.development_validation}
        heldout_clips = {
            row["clip_id"] for row in full_manifest("original") if row["split"] == HELDOUT_SPLIT
        }
        self.assertFalse(training_clips & heldout_clips)
        self.assertEqual(100, len(training_clips))

    def test_a_manifest_whose_heldout_lineages_are_undeclared_is_refused(self):
        with self.assertRaises(DistillationError) as caught:
            dataset_from_calibration_manifest(
                full_manifest("other"),
                dataset_id="undeclared",
                exclusions=exclusions(),
                label_source="original_judge_aggregate",
                budget=budget(),
                storage_uri="s3://plumb-distillation/undeclared/",
                frozen_at=PREREGISTERED_AT,
            )
        self.assertIn("absent from the declared exclusions", str(caught.exception))


class FreshHeldOutPanelTests(unittest.TestCase):
    def test_reusing_the_original_heldout_panel_is_refused(self):
        with self.assertRaises(DistillationError) as caught:
            assert_fresh_heldout_panel(full_manifest("original"), full_manifest("original"))
        self.assertIn("qualified the original judge", str(caught.exception))

    def test_a_relabelled_panel_reusing_the_original_lineages_is_refused(self):
        # New clip IDs, new media refs, same source states: a fresh-looking
        # panel that is the original test set underneath.
        relabelled = full_manifest("fresh")
        original = full_manifest("original")
        for fresh_row, original_row in zip(relabelled, original):
            fresh_row["source_lineage_id"] = original_row["source_lineage_id"]
        with self.assertRaises(DistillationError) as caught:
            assert_fresh_heldout_panel(relabelled, original)
        self.assertIn("not disjoint", str(caught.exception))

    def test_a_panel_reusing_original_clip_ids_is_refused(self):
        relabelled = full_manifest("fresh")
        original = full_manifest("original")
        for fresh_row, original_row in zip(relabelled, original):
            if original_row["split"] == HELDOUT_SPLIT:
                fresh_row["clip_id"] = original_row["clip_id"]
        with self.assertRaises(DistillationError) as caught:
            assert_fresh_heldout_panel(relabelled, original)
        self.assertIn("reuses original held-out clips", str(caught.exception))

    def test_a_genuinely_fresh_panel_is_accepted_and_records_both_hashes(self):
        report = assert_fresh_heldout_panel(full_manifest("fresh"), full_manifest("original"))
        self.assertEqual("pass", report["status"])
        self.assertNotEqual(report["fresh_manifest_hash"], report["original_manifest_hash"])

    def test_a_distilled_judge_cannot_reuse_the_original_frozen_protocol_hash(self):
        inputs = fresh_calibration_inputs()
        inputs["base_judge"] = BaseJudgeRevision(
            revision_id="qwen2.5-vl-7b-rev-a",
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            model_revision="0" * 40,
            rubric_hash=sha256("rubric"),
            sampling_hash=sha256("sampling"),
            gate_d_protocol_hash=inputs["protocol"].protocol_hash,
            results_ref="results/judge_calibration.json",
            results_sha256=sha256("original-judge-results"),
        )
        with self.assertRaises(DistillationError) as caught:
            run_fresh_heldout_calibration(**inputs)
        self.assertIn("own frozen protocol", str(caught.exception))

    def test_the_fresh_calibration_reuses_gate_ds_statistics(self):
        result, report = run_fresh_heldout_calibration(**fresh_calibration_inputs())
        self.assertTrue(result.passed)
        self.assertEqual("pass", result.status)
        self.assertEqual(
            "plumb.calibration.build_calibration_report", result.as_mapping()["statistics_owner"]
        )
        self.assertEqual(1.0, report["gate_d"]["metrics"]["sensitivity"])
        self.assertEqual(50, report["gate_d"]["metrics"]["heldout_overlap"])

    def test_fresh_calibration_requires_declared_independent_human_owners(self):
        inputs = fresh_calibration_inputs()
        inputs.pop("annotator_ownership")
        result, report = run_fresh_heldout_calibration(**inputs)
        self.assertFalse(result.passed)
        self.assertIn("independent_annotator_ownership_missing", result.reason_codes)
        self.assertEqual("missing", report["annotator_ownership"]["status"])

    def test_the_original_judges_inputs_are_never_mutated(self):
        inputs = fresh_calibration_inputs()
        original_panel = inputs["original_manifest"]
        snapshot = copy.deepcopy(original_panel)
        reports_snapshot = copy.deepcopy(inputs["distilled_judge_reports"])
        run_fresh_heldout_calibration(**inputs)
        self.assertEqual(snapshot, original_panel)
        self.assertEqual(reports_snapshot, inputs["distilled_judge_reports"])


class LoRASearchTests(unittest.TestCase):
    def test_both_named_arms_are_preregistered(self):
        space = search()
        signatures = {(arm.learning_rate, arm.rank) for arm in space.arms}
        self.assertIn((1e-3, 64), signatures)
        self.assertIn((5e-4, 32), signatures)
        self.assertEqual(PUBLISHED_LORA_ARM.learning_rate, 1e-3)
        self.assertEqual(REPO_DEFAULT_LORA_ARM.rank, 32)

    def test_a_search_without_the_repository_default_is_refused(self):
        with self.assertRaises(DistillationError) as caught:
            LoRASearchSpace(
                search_id="incomplete",
                arms=(PUBLISHED_LORA_ARM, PUBLISHED_LORA_ARM.__class__(
                    arm_id="another-r64",
                    learning_rate=1e-3,
                    rank=64,
                    alpha=16,
                    epochs=2,
                    provenance="project choice",
                    unsourced_fields=("alpha",),
                )),
                early_stopping=EarlyStoppingRule("development_validation_loss", "min", 2, 0.0, 2),
                selection_metric="development_validation_loss",
                selection_direction="minimize",
                preregistered_at=PREREGISTERED_AT,
            )
        self.assertIn("both named arms", str(caught.exception))

    def test_one_arm_is_not_a_comparison(self):
        with self.assertRaises(DistillationError):
            LoRASearchSpace(
                search_id="single",
                arms=(PUBLISHED_LORA_ARM,),
                early_stopping=EarlyStoppingRule("development_validation_loss", "min", 2, 0.0, 2),
                selection_metric="development_validation_loss",
                selection_direction="minimize",
                preregistered_at=PREREGISTERED_AT,
            )

    def test_the_published_configuration_is_labelled_a_candidate_not_an_optimum(self):
        mapping = PUBLISHED_LORA_ARM.as_mapping()
        self.assertEqual("candidate_configuration_not_a_portable_optimum", mapping["portability"])
        self.assertTrue(mapping["published_optimum_claim"])

    def test_published_optimum_claim_is_reserved_for_the_published_numbers(self):
        with self.assertRaises(DistillationError) as caught:
            LoRAConfig(
                arm_id="pretender",
                learning_rate=5e-4,
                rank=32,
                alpha=32,
                epochs=2,
                provenance="claiming someone else's numbers",
                published_optimum_claim=True,
            )
        self.assertIn("reserved for lr=1e-3", str(caught.exception))

    def test_an_arm_cannot_cite_a_source_for_a_field_it_calls_unsourced(self):
        with self.assertRaises(DistillationError):
            LoRAConfig(
                arm_id="contradictory",
                learning_rate=1e-3,
                rank=64,
                alpha=32,
                epochs=2,
                provenance="published",
                published_optimum_claim=True,
                unsourced_fields=("rank",),
            )

    def test_the_repository_default_arm_declares_which_numbers_it_chose(self):
        self.assertIn("alpha", REPO_DEFAULT_LORA_ARM.unsourced_fields)
        self.assertFalse(REPO_DEFAULT_LORA_ARM.published_optimum_claim)


class SelectionRefusalTests(unittest.TestCase):
    def test_a_search_selecting_on_heldout_data_is_refused(self):
        with self.assertRaises(DistillationError) as caught:
            LoRASearchSpace(
                search_id="tuned-on-test",
                arms=(PUBLISHED_LORA_ARM, REPO_DEFAULT_LORA_ARM),
                early_stopping=EarlyStoppingRule("development_validation_loss", "min", 2, 0.0, 2),
                selection_metric="development_validation_loss",
                selection_direction="minimize",
                preregistered_at=PREREGISTERED_AT,
                selection_split="heldout",
            )
        self.assertIn("tune against a test set", str(caught.exception))

    def test_a_selection_metric_naming_a_heldout_split_is_refused(self):
        with self.assertRaises(DistillationError):
            LoRASearchSpace(
                search_id="tuned-on-test-metric",
                arms=(PUBLISHED_LORA_ARM, REPO_DEFAULT_LORA_ARM),
                early_stopping=EarlyStoppingRule("development_validation_loss", "min", 2, 0.0, 2),
                selection_metric="heldout_binary_accuracy",
                selection_direction="maximize",
                preregistered_at=PREREGISTERED_AT,
            )

    def test_early_stopping_cannot_watch_a_heldout_split(self):
        with self.assertRaises(DistillationError):
            EarlyStoppingRule("development_validation_loss", "min", 2, 0.0, 2, monitored_split="heldout")
        with self.assertRaises(DistillationError):
            EarlyStoppingRule("heldout_loss", "min", 2, 0.0, 2)

    def test_an_arm_result_scored_on_heldout_data_is_refused(self):
        space = search()
        with self.assertRaises(DistillationError) as caught:
            ArmResult(
                arm_id=space.arms[0].arm_id,
                status="completed",
                selection_metric=space.selection_metric,
                metric_value=0.3,
                metric_source="baseten_training_job_log",
                selection_split="heldout",
                training_job_id="job-1",
                adapter_uri="s3://a/",
                adapter_sha256=sha256("a"),
            )
        self.assertIn("cannot select a winner", str(caught.exception))

    def test_an_arm_metric_naming_a_heldout_split_is_refused(self):
        with self.assertRaises(DistillationError):
            ArmResult(arm_id="a", status="not_run", selection_metric="heldout_loss")

    def test_a_completed_arm_must_report_its_metric_and_adapter(self):
        with self.assertRaises(DistillationError):
            ArmResult(arm_id="a", status="completed", selection_metric="development_validation_loss")

    def test_a_metric_without_a_source_is_refused(self):
        with self.assertRaises(DistillationError):
            ArmResult(
                arm_id="a",
                status="running",
                selection_metric="development_validation_loss",
                metric_value=0.2,
            )

    def test_a_failed_arm_must_say_why(self):
        with self.assertRaises(DistillationError):
            ArmResult(arm_id="a", status="failed", selection_metric="development_validation_loss")


class SearchRecordTests(unittest.TestCase):
    def test_every_arm_is_recorded_including_the_losers(self):
        space = search()
        selection = select_arm(space, arm_results(space))
        self.assertEqual("selected", selection.status)
        self.assertEqual("published-lr1e-3-r64-a32-e2", selection.winner_arm_id)
        self.assertEqual(len(space.arms), len(selection.ranking))
        recorded = {entry["arm_id"] for entry in selection.ranking}
        self.assertEqual({arm.arm_id for arm in space.arms}, recorded)
        self.assertEqual(SELECTION_SPLIT, selection.selection_split)

    def test_a_failed_arm_stays_in_the_record_with_its_reason(self):
        space = search()
        results = arm_results(space, losers_fail=True)
        selection = select_arm(space, results)
        self.assertEqual(len(space.arms), len(selection.ranking))
        self.assertTrue(any(code.startswith("arm_failed:") for code in selection.reason_codes))
        failed = [entry for entry in selection.ranking if entry["status"] == "failed"]
        self.assertEqual(2, len(failed))
        self.assertTrue(all(entry["reason"] for entry in failed))

    def test_a_missing_arm_is_named_and_the_selection_is_flagged(self):
        space = search()
        results = arm_results(space)[:2]
        selection = select_arm(space, results)
        self.assertEqual("selected_with_gaps", selection.status)
        self.assertIn("arms_missing_from_record", selection.reason_codes)

    def test_a_tie_refuses_to_name_a_winner(self):
        space = search()
        results = tuple(
            ArmResult(
                arm_id=arm.arm_id,
                status="completed",
                selection_metric=space.selection_metric,
                metric_value=0.5,
                metric_source="baseten_training_job_log",
                training_job_id="job-" + arm.arm_id,
                adapter_uri="s3://a/" + arm.arm_id,
                adapter_sha256=sha256("adapter:" + arm.arm_id),
            )
            for arm in space.arms
        )
        selection = select_arm(space, results)
        self.assertEqual("tied", selection.status)
        self.assertIsNone(selection.winner_arm_id)
        self.assertFalse(selection.reportable)

    def test_no_completed_arm_means_no_selection(self):
        space = search()
        results = tuple(
            ArmResult(arm_id=arm.arm_id, status="not_run", selection_metric=space.selection_metric)
            for arm in space.arms
        )
        selection = select_arm(space, results)
        self.assertEqual("not_selectable", selection.status)
        self.assertIn("no_arm_completed", selection.reason_codes)

    def test_an_unpreregistered_arm_cannot_be_smuggled_into_the_record(self):
        space = search()
        results = arm_results(space) + (
            ArmResult(arm_id="secret-fourth-arm", status="not_run", selection_metric=space.selection_metric),
        )
        with self.assertRaises(DistillationError) as caught:
            select_arm(space, results)
        self.assertIn("not preregistered", str(caught.exception))

    def test_the_spoken_lora_line_requires_the_published_arm_to_win(self):
        space = search()
        won = spoken_claim(space, select_arm(space, arm_results(space)))
        self.assertTrue(won["sayable"])
        self.assertIn("published LoRA optimum", won["sentence"])
        lost = spoken_claim(space, select_arm(space, arm_results(space, winner="repo-default-lr5e-4-r32")))
        self.assertFalse(lost["sayable"])
        self.assertIn("did not win", lost["reason"])
        self.assertIn("not present the published numbers as a portable optimum", lost["sentence"])


class TrainingJobPayloadTests(unittest.TestCase):
    def test_the_payload_is_rendered_and_never_submitted(self):
        space = search()
        spec = TrainingJobSpec(
            job_name="distill-published",
            project_name="plumb-judge-distillation",
            arm=space.arms[0],
            search=space,
            dataset=dataset(),
            base_judge=base_judge(),
            student=student(),
            runtime=runtime(),
        )
        payload = spec.as_payload(env={})
        self.assertEqual("rendered_not_submitted", payload["status"])
        self.assertEqual("unconfirmed_against_training_jobs_reference", payload["route_status"])
        self.assertFalse(hasattr(spec, "submit"))
        self.assertEqual(1e-3, payload["hyperparameters"]["learning_rate"])
        self.assertEqual(64, payload["hyperparameters"]["lora_r"])
        self.assertEqual(32, payload["hyperparameters"]["lora_alpha"])
        self.assertEqual(SELECTION_SPLIT, payload["hyperparameters"]["selection_split"])
        json.dumps(payload, allow_nan=False)

    def test_missing_credentials_and_unchecked_compatibility_are_named_blockers(self):
        space = search()
        spec = TrainingJobSpec(
            job_name="distill-published",
            project_name="plumb-judge-distillation",
            arm=space.arms[0],
            search=space,
            dataset=dataset(),
            base_judge=base_judge(),
            student=student(),
            runtime=runtime(),
        )
        blockers = spec.submission_blockers(env={})
        self.assertIn("BASETEN_API_KEY is unset", blockers)
        self.assertTrue(any("compatibility" in reason for reason in blockers))
        self.assertTrue(any("Training Jobs route" in reason for reason in blockers))

    def test_framework_compatibility_is_unchecked_until_the_container_builds(self):
        self.assertEqual(
            "unchecked_until_the_training_container_builds",
            runtime().as_payload()["framework_compatibility"],
        )
        self.assertEqual(
            "checked_in_training_container",
            runtime(framework_compatibility_checked=True).as_payload()["framework_compatibility"],
        )

    def test_a_secret_value_cannot_be_recorded_in_place_of_its_name(self):
        with self.assertRaises(DistillationError) as caught:
            runtime(secret_names=("hf_access_token=hf_realsecret",))
        self.assertIn("looks like a value", str(caught.exception))

    def test_one_payload_per_preregistered_arm(self):
        space = search()
        payloads = render_search_payloads(
            project_name="plumb-judge-distillation",
            job_name_prefix="distill",
            search=space,
            dataset=dataset(),
            base_judge=base_judge(),
            student=student(),
            runtime=runtime(),
            env={},
        )
        self.assertEqual(len(space.arms), len(payloads))
        self.assertEqual(
            [arm.arm_id for arm in space.arms], [payload["search"]["arm_id"] for payload in payloads]
        )
        self.assertTrue(all(payload["status"] == "rendered_not_submitted" for payload in payloads))

    def test_an_unsupported_training_framework_is_refused(self):
        with self.assertRaises(DistillationError):
            runtime(framework="my-own-trainer")


class BudgetTests(unittest.TestCase):
    def test_a_zero_distillation_cost_is_refused(self):
        with self.assertRaises(DistillationError) as caught:
            budget(estimated_usd=0.0, cost_basis_ref="economics://price-basis-v1")
        self.assertIn("never zero", str(caught.exception))

    def test_an_unknown_cost_needs_a_reason(self):
        with self.assertRaises(DistillationError) as caught:
            TrainingBudgetItem(label_count=100, label_source="human")
        self.assertIn("usd_unavailable_reason", str(caught.exception))

    def test_a_usd_amount_requires_a_recorded_pricing_snapshot(self):
        with self.assertRaises(DistillationError):
            budget(estimated_usd=42.0, usd_unavailable_reason=None)

    def test_the_heldout_clips_never_supply_training_labels(self):
        with self.assertRaises(DistillationError) as caught:
            budget(supplied_by_gate_d_heldout=True)
        self.assertIn("additional acquisition item", str(caught.exception))

    def test_the_cost_is_an_additional_line_item_in_the_cost_model(self):
        item = budget()
        self.assertEqual({BUDGET_ITEM: None}, item.as_prior_cost_report())
        model = GpuSecondModel(prior_cost_reports=item.as_prior_cost_report())
        prior = model.prior_costs()
        self.assertEqual("reported", prior["status"])
        self.assertIsNone(prior["entries"][BUDGET_ITEM])
        self.assertIn("never recorded as zero", prior["note"])

    def test_a_priced_line_item_reports_estimated_not_zero(self):
        item = budget(estimated_usd=37.5, cost_basis_ref="economics://price-basis-v1")
        mapping = item.as_mapping()
        self.assertEqual("estimated", mapping["usd_status"])
        self.assertEqual(37.5, mapping["estimated_usd"])
        self.assertEqual("additional_acquisition_item_reported_separately", mapping["attribution"])
        self.assertEqual({BUDGET_ITEM: 37.5}, item.as_prior_cost_report())

    def test_a_settled_cost_requires_a_reconciliation_reference(self):
        with self.assertRaises(DistillationError):
            budget(settled_usd=31.0)


class RevisionIdentityTests(unittest.TestCase):
    def test_a_distilled_revision_has_its_own_identity_hash(self):
        space = search()
        data = dataset()
        distilled = revision(space, data)
        self.assertNotEqual(distilled.revision_id, distilled.base_judge.revision_id)
        self.assertTrue(distilled.identity_hash().startswith("sha256:"))
        self.assertNotEqual(distilled.identity_hash(), distilled.base_judge.results_sha256)
        self.assertEqual(["D", "E", "F"], distilled.as_mapping()["invalidates_gates"])

    def test_the_original_judges_results_file_cannot_be_reused(self):
        space = search()
        data = dataset()
        with self.assertRaises(DistillationError) as caught:
            revision(space, data, results_ref="results/judge_calibration.json")
        self.assertIn("results are preserved", str(caught.exception))

    def test_the_base_revision_id_cannot_be_reused(self):
        space = search()
        data = dataset()
        with self.assertRaises(DistillationError):
            revision(space, data, revision_id="qwen2.5-vl-7b-rev-a")

    def test_a_different_rubric_is_a_different_measurement(self):
        space = search()
        data = dataset()
        with self.assertRaises(DistillationError) as caught:
            revision(space, data, rubric_hash=sha256("a-nicer-rubric"))
        self.assertIn("rubric is frozen", str(caught.exception))

    def test_changing_the_sample_count_requires_refreezing_the_quorum(self):
        with self.assertRaises(DistillationError) as caught:
            frozen_sampling_for_distilled_judge(sample_count=3)
        self.assertIn("refreezing the quorum", str(caught.exception))
        with self.assertRaises(DistillationError):
            frozen_sampling_for_distilled_judge(sample_count=4, quorum=2)
        refrozen = frozen_sampling_for_distilled_judge(sample_count=3, quorum=2)
        self.assertEqual(3, refrozen["sample_count"])

    def test_an_unknown_sampling_field_needs_a_new_protocol(self):
        with self.assertRaises(DistillationError):
            frozen_sampling_for_distilled_judge(beam_width=4)

    def test_a_revision_trained_on_another_dataset_is_refused(self):
        space = search()
        data = dataset()
        other = dataset(
            dataset_id="other-dataset",
            storage_uri="s3://plumb-distillation/other/",
        )
        results = arm_results(space)
        with self.assertRaises(DistillationError) as caught:
            DistillationGate(
                revision=revision(space, other),
                search=space,
                arm_results=results,
                selection=select_arm(space, results),
                dataset=data,
            )
        self.assertIn("not trained on the recorded dataset", str(caught.exception))


class PairedFrozenVideoTests(unittest.TestCase):
    def test_a_paired_comparison_needs_the_identical_clip_set(self):
        original = frozen_video_scores(revision_id="rev-a")
        distilled = frozen_video_scores(revision_id="rev-b")[:-1]
        with self.assertRaises(DistillationError) as caught:
            compare_frozen_videos(
                comparison_id="short",
                original=original,
                distilled=distilled,
                tolerances=paired_tolerances(),
            )
        self.assertIn("identical clip set", str(caught.exception))

    def test_both_judges_must_see_the_identical_frames(self):
        original = frozen_video_scores(revision_id="rev-a")
        distilled = list(frozen_video_scores(revision_id="rev-b"))
        first = distilled[0]
        distilled[0] = FrozenVideoScore(
            clip_id=first.clip_id,
            judge_revision=first.judge_revision,
            video_sha256=first.video_sha256,
            frame_hashes=tuple(sha256("resampled:{0}".format(index)) for index in range(16)),
            binary_success=first.binary_success,
            progress=first.progress,
        )
        with self.assertRaises(DistillationError) as caught:
            compare_frozen_videos(
                comparison_id="resampled",
                original=original,
                distilled=tuple(distilled),
                tolerances=paired_tolerances(),
            )
        self.assertIn("identical frames", str(caught.exception))

    def test_a_judge_cannot_be_paired_against_itself(self):
        rows = frozen_video_scores(revision_id="rev-a")
        with self.assertRaises(DistillationError) as caught:
            compare_frozen_videos(
                comparison_id="self",
                original=rows,
                distilled=rows,
                tolerances=paired_tolerances(),
            )
        self.assertIn("two different judge revisions", str(caught.exception))

    def test_the_comparison_is_paired_and_reuses_the_paired_interval(self):
        result = passing_paired_comparison()
        self.assertTrue(result.passed)
        mapping = result.as_mapping()
        self.assertTrue(mapping["paired"])
        self.assertEqual("plumb.annotation.paired_difference_interval", mapping["interval_owner"])
        self.assertEqual(50, mapping["metrics"]["both_decisive_pairs"])
        self.assertEqual(49, mapping["metrics"]["binary_agreements"])
        self.assertEqual(1, mapping["metrics"]["positive_only_distilled"])
        self.assertEqual(0, mapping["metrics"]["positive_only_original"])
        self.assertEqual("paired_wald_95", mapping["metrics"]["paired_binary_difference"]["method"])
        json.dumps(mapping, allow_nan=False)

    def test_a_leniently_disagreeing_distilled_judge_fails_its_tolerances(self):
        # Flip only clips the original judge called failures, so the
        # disagreement is one-sided and the paired difference cannot cancel.
        flips = tuple(
            row.clip_id
            for row in frozen_video_scores(revision_id="rev-b")
            if row.binary_success is False
        )[:10]
        result = compare_frozen_videos(
            comparison_id="divergent",
            original=frozen_video_scores(revision_id="rev-a"),
            distilled=frozen_video_scores(revision_id="rev-b", flip_clips=flips),
            tolerances=paired_tolerances(),
        )
        self.assertFalse(result.passed)
        self.assertIn("paired_binary_difference_above_maximum", result.reason_codes)

    def test_thin_paired_coverage_fails_rather_than_being_averaged_away(self):
        unevaluable = tuple(row.clip_id for row in frozen_video_scores(revision_id="rev-b")[:15])
        result = compare_frozen_videos(
            comparison_id="thin",
            original=frozen_video_scores(revision_id="rev-a", unevaluable=unevaluable),
            distilled=frozen_video_scores(revision_id="rev-b", unevaluable=unevaluable),
            tolerances=paired_tolerances(),
        )
        self.assertFalse(result.passed)
        self.assertIn("paired_clips_below_minimum", result.reason_codes)
        self.assertIn("paired_coverage_below_minimum", result.reason_codes)

    def test_an_unevaluable_clip_needs_a_missing_reason(self):
        with self.assertRaises(DistillationError):
            FrozenVideoScore(
                clip_id="c-1",
                judge_revision="rev-a",
                video_sha256=sha256("v"),
                frame_hashes=(sha256("f"),),
                binary_success=None,
            )


class GateEEvidenceTests(unittest.TestCase):
    def test_an_unvalidated_distilled_judge_cannot_be_marked_used_for_scoring(self):
        gate = complete_gate(fresh_calibration=None, paired_comparison=None)
        evidence = gate.evidence()
        self.assertFalse(evidence["used_for_scoring"])
        self.assertFalse(evidence["separately_validated"])
        self.assertTrue(evidence["scoring_request_refused"])
        self.assertTrue(any("fresh held-out calibration" in reason for reason in evidence["blockers"]))
        self.assertTrue(any("paired frozen-video" in reason for reason in evidence["blockers"]))

    def test_a_missing_paired_comparison_alone_still_blocks_scoring(self):
        gate = complete_gate(paired_comparison=None)
        self.assertFalse(gate.separately_validated)
        self.assertFalse(gate.evidence()["used_for_scoring"])

    def test_a_failed_fresh_calibration_blocks_scoring(self):
        inputs = fresh_calibration_inputs()
        inputs["tolerances"] = GateDTolerances(50, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        inputs["distilled_judge_reports"] = None
        failed, _ = run_fresh_heldout_calibration(**inputs)
        self.assertFalse(failed.passed)
        gate = complete_gate(fresh_calibration=failed)
        self.assertFalse(gate.evidence()["used_for_scoring"])
        self.assertTrue(
            any("fresh held-out calibration did not pass" in reason for reason in gate.blockers())
        )

    def test_assert_scoring_authorised_stops_an_unvalidated_burst(self):
        gate = complete_gate(fresh_calibration=None, paired_comparison=None)
        with self.assertRaises(DistillationError) as caught:
            assert_scoring_authorised(gate)
        self.assertIn("may not score a burst", str(caught.exception))

    def test_a_validated_revision_is_authorised_and_scores(self):
        gate = complete_gate()
        assert_scoring_authorised(gate)
        evidence = gate.evidence()
        self.assertTrue(evidence["used_for_scoring"])
        self.assertTrue(evidence["separately_validated"])
        self.assertEqual([], evidence["blockers"])
        self.assertTrue(gate.authorisation()["authorised"])

    def test_a_validated_revision_that_is_not_requested_does_not_score(self):
        gate = complete_gate(requested_for_scoring=False)
        evidence = gate.evidence()
        self.assertTrue(evidence["separately_validated"])
        self.assertFalse(evidence["used_for_scoring"])
        self.assertFalse(evidence["scoring_request_refused"])

    def test_a_scalar_head_student_can_never_score_a_burst(self):
        space = search()
        data = dataset()
        results = arm_results(space)
        gate = DistillationGate(
            revision=revision(space, data, contract=SCALAR_HEAD_CONTRACT),
            search=space,
            arm_results=results,
            selection=select_arm(space, results),
            dataset=data,
            fresh_calibration=passing_fresh_calibration(),
            paired_comparison=passing_paired_comparison(),
            requested_for_scoring=True,
        )
        self.assertFalse(gate.evidence()["used_for_scoring"])
        self.assertTrue(any("frozen rubric schema" in reason for reason in gate.blockers()))

    def test_a_revision_from_a_losing_arm_cannot_score(self):
        space = search()
        data = dataset()
        results = arm_results(space)
        gate = DistillationGate(
            revision=revision(space, data, arm_id="repo-default-lr5e-4-r32"),
            search=space,
            arm_results=results,
            selection=select_arm(space, results),
            dataset=data,
            fresh_calibration=passing_fresh_calibration(),
            paired_comparison=passing_paired_comparison(),
            requested_for_scoring=True,
        )
        self.assertFalse(gate.evidence()["used_for_scoring"])
        self.assertTrue(any("reportable winner is" in reason for reason in gate.blockers()))

    def test_a_gate_cannot_drop_a_losing_arm_from_its_record(self):
        space = search()
        data = dataset()
        results = arm_results(space)
        with self.assertRaises(DistillationError) as caught:
            DistillationGate(
                revision=revision(space, data),
                search=space,
                arm_results=results[:1],
                selection=select_arm(space, results),
                dataset=data,
            )
        self.assertIn("including losers", str(caught.exception))

    def test_the_evidence_records_every_arm_including_losers(self):
        evidence = complete_gate().evidence()
        recorded = {entry["arm_id"] for entry in evidence["lora_search"]["results"]}
        self.assertEqual({arm.arm_id for arm in search().arms}, recorded)
        self.assertEqual(3, len(evidence["lora_search"]["arms"]))
        self.assertEqual(
            search().preregistration_hash(), evidence["lora_search"]["preregistration_hash"]
        )

    def test_the_evidence_preserves_and_separates_the_original_judges_results(self):
        evidence = complete_gate().evidence()
        self.assertTrue(evidence["original_judge_results_preserved"])
        self.assertEqual("results/judge_calibration.json", evidence["original_judge_results_ref"])
        self.assertNotEqual(
            evidence["original_judge_results_ref"], evidence["distilled_judge_results_ref"]
        )

    def test_the_evidence_carries_the_budget_item_and_never_a_zero_cost(self):
        evidence = complete_gate().evidence()
        item = evidence["training_budget_item"]
        self.assertEqual(BUDGET_ITEM, item["budget_item"])
        self.assertEqual("unavailable", item["usd_status"])
        self.assertIsNone(item["estimated_usd"])
        self.assertFalse(item["supplied_by_gate_d_heldout"])

    def test_the_evidence_is_json_serialisable_without_nan(self):
        json.dumps(complete_gate().evidence(), allow_nan=False, sort_keys=True)

    def test_a_complete_mapping_satisfies_gate_e(self):
        patch = complete_gate().gate_e_measurements_patch()
        self.assertEqual({"distilled_judge"}, set(patch))
        record = gate_e_record(patch["distilled_judge"])
        self.assertEqual((), record.pass_evidence_errors())

    def test_a_forged_mapping_fails_gate_e(self):
        record = gate_e_record({"used_for_scoring": True, "separately_validated": False})
        errors = record.pass_evidence_errors()
        self.assertTrue(
            any("separately validated distilled judge" in message for message in errors), errors
        )

    def test_a_blocked_gate_e_mapping_does_not_silently_pass_as_scoring(self):
        gate = complete_gate(fresh_calibration=None, paired_comparison=None)
        record = gate_e_record(gate.gate_e_measurements_patch()["distilled_judge"])
        # Gate E's rule is conditional: refusing to score is a valid Gate E
        # state, and the blockers are on the record for a reviewer to read.
        self.assertEqual((), record.pass_evidence_errors())
        self.assertTrue(record.measurements["distilled_judge"]["blockers"])

    def test_the_no_distilled_judge_entry_is_explicit_rather_than_absent(self):
        evidence = unvalidated_distilled_judge_evidence()
        self.assertFalse(evidence["used_for_scoring"])
        self.assertFalse(evidence["separately_validated"])
        self.assertEqual("not_run", evidence["status"])
        self.assertEqual((), gate_e_record(evidence).pass_evidence_errors())

    def test_a_selection_from_another_preregistered_search_is_refused(self):
        space = search()
        other = default_search_space(search_id="lora-search-v2", preregistered_at=PREREGISTERED_AT)
        data = dataset()
        results = arm_results(space)
        with self.assertRaises(DistillationError) as caught:
            DistillationGate(
                revision=revision(space, data),
                search=space,
                arm_results=results,
                selection=select_arm(other, arm_results(other)),
                dataset=data,
            )
        self.assertIn("different preregistered search", str(caught.exception))


class DeployTrainingDirectoryTests(unittest.TestCase):
    """The rendered job config and the entrypoint, exercised on CPU."""

    def setUp(self):
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import deploy.baseten.training.job_config as job_config
        import deploy.baseten.training.train_judge_lora as entrypoint

        self.job_config = job_config
        self.entrypoint = entrypoint

    def test_the_deploy_modules_import_without_torch_or_transformers(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("trl", sys.modules)
        self.assertNotIn("peft", sys.modules)

    def test_the_pinned_requirements_have_a_digest_and_no_computecanada_wheels(self):
        text = Path(self.job_config.REQUIREMENTS_PATH).read_text(encoding="utf-8")
        self.assertIn("BASETEN IMAGE UNVERIFIED", text)
        self.assertIn("does NOT validate the public-PyPI image", text)
        # The header must *name* the non-transferable Alliance builds; no
        # installable line may actually pin one.
        self.assertIn("+computecanada", text)
        pins = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith(("#", "--"))
        ]
        self.assertTrue(pins)
        for pin in pins:
            self.assertIn("==", pin, pin)
            self.assertNotIn(">=", pin, pin)
            self.assertNotIn("+computecanada", pin, pin)
        self.assertTrue(self.job_config.requirements_digest().startswith("sha256:"))

    def rendered_payloads(self):
        return self.job_config.build_payloads(
            calibration_manifest=full_manifest("original"),
            gate_d_heldout_lineages=heldout_lineages("original"),
            primary_lineages=primary_lineages(),
            dataset_uri="s3://plumb-distillation/dataset-v1/",
            dataset_id="distill-dataset-v1",
            frozen_at=PREREGISTERED_AT,
            preregistered_at=PREREGISTERED_AT,
            search_id="lora-search-v1",
            base_judge_config={
                "revision_id": "qwen2.5-vl-7b-rev-a",
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "model_revision": "0" * 40,
                "rubric_hash": sha256("rubric"),
                "sampling_hash": sha256("sampling"),
                "gate_d_protocol_hash": sha256("gate-d-protocol"),
                "results_ref": "results/judge_calibration.json",
                "results_sha256": sha256("original-judge-results"),
            },
            student_config={
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "model_revision": "0" * 40,
                "output_contract": RUBRIC_JSON_CONTRACT,
            },
            label_source="original_judge_aggregate",
            label_count=100,
            usd_unavailable_reason="no verified price basis for the training accelerator yet",
        )

    def test_the_job_config_renders_one_payload_per_arm_without_submitting(self):
        payloads = self.rendered_payloads()
        self.assertEqual(3, len(payloads))
        for payload in payloads:
            self.assertEqual("rendered_not_submitted", payload["status"])
            self.assertEqual("pass", payload["datasets"]["lineage_disjointness"])
            self.assertEqual(
                self.job_config.requirements_digest(), payload["runtime"]["requirements_sha256"]
            )
            json.dumps(payload, allow_nan=False)

    def test_the_job_config_refuses_a_dataset_that_touches_the_heldout_lineages(self):
        with self.assertRaises(DistillationError):
            self.job_config.build_payloads(
                calibration_manifest=full_manifest("original"),
                gate_d_heldout_lineages=("some-unrelated-lineage",),
                primary_lineages=primary_lineages(),
                dataset_uri="s3://plumb-distillation/dataset-v1/",
                dataset_id="distill-dataset-v1",
                frozen_at=PREREGISTERED_AT,
                preregistered_at=PREREGISTERED_AT,
                search_id="lora-search-v1",
                base_judge_config={
                    "revision_id": "qwen2.5-vl-7b-rev-a",
                    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "model_revision": "0" * 40,
                    "rubric_hash": sha256("rubric"),
                    "sampling_hash": sha256("sampling"),
                    "gate_d_protocol_hash": sha256("gate-d-protocol"),
                    "results_ref": "results/judge_calibration.json",
                    "results_sha256": sha256("original-judge-results"),
                },
                student_config={
                    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "model_revision": "0" * 40,
                    "output_contract": RUBRIC_JSON_CONTRACT,
                },
                label_source="original_judge_aggregate",
                label_count=100,
                usd_unavailable_reason="unknown",
            )

    def dataset_row(self, clip_id, lineage, cohort):
        return {
            "clip_id": clip_id,
            "source_lineage_id": lineage,
            "cohort": cohort,
            "task": "open_drawer",
            "input_profile": self.entrypoint.SERVING_INPUT_PROFILE,
            "instruction": "Open the drawer",
            "rubric": "0 no directed approach ... 5 published final-state criterion met",
            "frames": ["frames/{0}/{1:02d}.png".format(clip_id, index) for index in range(16)],
            "frame_timestamps": [float(index) for index in range(16)],
            "reference_images": [
                {
                    "path": "references/{0}.png".format(clip_id),
                    "sha256": sha256("reference:" + clip_id),
                    "role": "goal",
                    "provenance_uri": "artifact://references/{0}.png".format(clip_id),
                }
            ],
            "label": {
                "integrity": "intact",
                "collision": "none_visible",
                "progress": 2,
                "completion_evidence": "not_met",
                "evidence_frame_indices": [0, 15],
                "observable_reasons": "Handle contact, drawer unmoved.",
            },
        }

    def write_dataset(self, directory, *, train_lineage="dev-lineage-1", validation_lineage="dev-lineage-2"):
        (directory / "train.jsonl").write_text(
            json.dumps(self.dataset_row("train-clip-1", train_lineage, "distillation_train")) + "\n",
            encoding="utf-8",
        )
        (directory / "development_validation.jsonl").write_text(
            json.dumps(
                self.dataset_row("val-clip-1", validation_lineage, "distillation_dev_validation")
            )
            + "\n",
            encoding="utf-8",
        )

    def test_the_entrypoint_check_mode_validates_config_and_dataset_on_cpu(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "job.json"
            config_path.write_text(json.dumps(self.rendered_payloads()[0]), encoding="utf-8")
            self.write_dataset(root)
            summary = self.entrypoint.check(str(config_path), str(root), None)
            self.assertEqual("config_and_dataset_valid", summary["status"])
            self.assertEqual("pass", summary["lineage_disjointness"])
            self.assertEqual(1, summary["splits"]["train"]["rows"])

    def test_the_entrypoint_refuses_a_row_carrying_an_excluded_lineage(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "job.json"
            config_path.write_text(json.dumps(self.rendered_payloads()[0]), encoding="utf-8")
            self.write_dataset(root, train_lineage=heldout_lineages("original")[0])
            excluded = root / "excluded.json"
            excluded.write_text(json.dumps(list(heldout_lineages("original"))), encoding="utf-8")
            with self.assertRaises(self.entrypoint.TrainingConfigError) as caught:
                self.entrypoint.check(str(config_path), str(root), str(excluded))
            self.assertIn("held-out or primary data", str(caught.exception))

    def test_the_entrypoint_refuses_a_config_whose_disjointness_was_edited(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self.rendered_payloads()[0]
            payload["datasets"]["lineage_disjointness"] = "unverifiable"
            config_path = root / "job.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(self.entrypoint.TrainingConfigError) as caught:
                self.entrypoint.check(str(config_path), None, None)
            self.assertIn("not 'pass'", str(caught.exception))

    def test_the_entrypoint_refuses_a_config_that_selects_on_heldout_data(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self.rendered_payloads()[0]
            payload["hyperparameters"]["selection_split"] = "heldout"
            config_path = root / "job.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(self.entrypoint.TrainingConfigError) as caught:
                self.entrypoint.check(str(config_path), None, None)
            self.assertIn("may only be selected on", str(caught.exception))

    def test_the_entrypoint_refuses_a_validation_split_shared_with_training(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "job.json"
            config_path.write_text(json.dumps(self.rendered_payloads()[0]), encoding="utf-8")
            self.write_dataset(root, validation_lineage="dev-lineage-1")
            with self.assertRaises(self.entrypoint.TrainingConfigError) as caught:
                self.entrypoint.check(str(config_path), str(root), None)
            self.assertIn("watch its own", str(caught.exception))

    def test_a_clip_with_the_wrong_frame_count_is_refused(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "job.json"
            config_path.write_text(json.dumps(self.rendered_payloads()[0]), encoding="utf-8")
            row = self.dataset_row("train-clip-1", "dev-lineage-1", "distillation_train")
            row["frames"] = row["frames"][:8]
            (root / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (root / "development_validation.jsonl").write_text(
                json.dumps(self.dataset_row("val-clip-1", "dev-lineage-2", "distillation_dev_validation"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(self.entrypoint.TrainingConfigError) as caught:
                self.entrypoint.check(str(config_path), str(root), None)
            self.assertIn("frame refs", str(caught.exception))

    def test_training_without_the_frameworks_names_them_and_writes_nothing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "job.json"
            config_path.write_text(json.dumps(self.rendered_payloads()[0]), encoding="utf-8")
            self.write_dataset(root)
            exit_code = self.entrypoint.main(
                [
                    "--config",
                    str(config_path),
                    "--dataset-dir",
                    str(root),
                    "--output-dir",
                    str(root / "out"),
                ]
            )
            self.assertEqual(3, exit_code)
            self.assertFalse((root / "out" / self.entrypoint.RESULT_FILENAME).exists())

    def test_a_multi_arm_payload_file_is_refused_as_a_single_job_config(self):
        with TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "all.json"
            config_path.write_text(json.dumps(self.rendered_payloads()), encoding="utf-8")
            with self.assertRaises(self.entrypoint.TrainingConfigError) as caught:
                self.entrypoint.load_job_config(str(config_path))
            self.assertIn("trains one arm", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover - parity with the other suites
    unittest.main()
