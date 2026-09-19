"""Tests for development-only distilled-judge preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from plumb.calibration import Annotation, DEVELOPMENT_SPLIT, HELDOUT_SPLIT, freeze_calibration_selection, load_frozen_calibration_selection
from plumb.distillation import (
    DistillationError,
    build_training_job_readiness,
    deterministic_development_split,
    distilled_revision_readiness,
    freeze_distillation_dataset,
    freeze_training_job_readiness,
    load_frozen_distillation_dataset,
    prepare_distillation_dataset,
)


TASKS = ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def calibration_rows():
    rows = []
    for task in TASKS:
        for split, count in ((DEVELOPMENT_SPLIT, 20), (HELDOUT_SPLIT, 10)):
            for index in range(count):
                clip_id = "clip-{0}-{1}-{2:02d}".format(task, split, index)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "media_ref": "artifact://private-media/{0}.mp4".format(clip_id),
                        "media_hash": digest("media:" + clip_id),
                        "task": task,
                        "split": split,
                        "source_lineage_id": "lineage-{0}-{1}-{2:02d}".format(task, split, index),
                    }
                )
    return rows


def frozen_selection(tmp_path: Path) -> Path:
    path = tmp_path / "calibration-selection.json"
    freeze_calibration_selection(
        calibration_rows(),
        path,
        reserved_cohort_lineages={"primary": ["primary-lineage"], "cost_confirmation": ["cost-lineage"]},
    )
    return path


def protocol():
    return {
        "protocol_id": "distillation-v1",
        "teacher": {
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_revision": "a" * 40,
            "processor_revision": "b" * 40,
            "runtime_lock_id": "runtime-lock-v1",
            "asset_manifest_id": "assets-lock-v1",
            "producer_source_hash": digest("teacher-producer"),
        },
        "rubric_hash": digest("rubric"),
        "sampling_hash": digest("sampling"),
        "development_split_seed": 17,
        "search_plan": {
            "preregistered": True,
            "candidates": [
                {"learning_rate": 1e-3, "rank": 64, "alpha": 32},
                {"learning_rate": 5e-4, "rank": 32, "alpha": 16},
            ],
            "early_stopping": {"enabled": True, "metric": "development_validation_loss", "patience": 3},
        },
    }


def development_annotations(selection_path: Path):
    selection = load_frozen_calibration_selection(selection_path)
    return [
        Annotation(
            clip_id=clip["clip_id"],
            annotator_id="development-human",
            integrity="intact",
            collision="none_visible",
            progress=2,
            completion_evidence="not_met",
            evidence_frame_indices=(0, 15),
            observable_reason="Visible incomplete end state.",
        )
        for clip in selection["clips"]
        if clip["split"] == DEVELOPMENT_SPLIT
    ]


def teacher_reports(selection_path: Path, protocol_value=None):
    selection = load_frozen_calibration_selection(selection_path)
    plan = protocol_value or protocol()
    reports = []
    for clip in selection["clips"]:
        if clip["split"] != DEVELOPMENT_SPLIT:
            continue
        reports.append(
            {
                "raw_outputs": ["raw teacher output for " + clip["clip_id"]],
                "provenance": {
                    "clip_id": clip["clip_id"],
                    "calibration_manifest_hash": selection["calibration_manifest_hash"],
                    "media_hash": clip["media_hash"],
                    "rubric_hash": plan["rubric_hash"],
                    "sampling_hash": plan["sampling_hash"],
                    "teacher": plan["teacher"],
                },
            }
        )
    return reports


def test_freezes_hash_bound_development_only_dataset_with_unselected_lora_candidate(tmp_path: Path) -> None:
    selection = frozen_selection(tmp_path)
    payload = prepare_distillation_dataset(selection, development_annotations(selection), teacher_reports(selection), protocol())
    assert payload["status"] == "prepared_not_trained"
    assert payload["counts"] == {"records": 100, "training": 80, "validation": 20}
    assert payload["distillation_protocol"]["candidate_lora"] == {
        "learning_rate": 1e-3,
        "rank": 64,
        "alpha": 32,
        "status": "candidate_not_optimized",
    }
    assert {record["dataset_split"] for record in payload["records"]} == {"training", "validation"}
    assert all(record["teacher_raw_report"]["raw_outputs"] for record in payload["records"])

    output = tmp_path / "distillation-dataset.json"
    frozen = freeze_distillation_dataset(selection, development_annotations(selection), teacher_reports(selection), protocol(), output)
    loaded = load_frozen_distillation_dataset(output)
    assert loaded["sha256"] == frozen["sha256"]
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(DistillationError, match="overwrite"):
        freeze_distillation_dataset(selection, development_annotations(selection), teacher_reports(selection), protocol(), output)


def test_heldout_labels_or_teacher_reports_cannot_enter_distillation(tmp_path: Path) -> None:
    selection = frozen_selection(tmp_path)
    labels = development_annotations(selection)
    heldout = next(clip for clip in load_frozen_calibration_selection(selection)["clips"] if clip["split"] == HELDOUT_SPLIT)
    labels.append(
        Annotation(
            clip_id=heldout["clip_id"], annotator_id="another-human", integrity="intact", collision="none_visible",
            progress=2, completion_evidence="not_met", evidence_frame_indices=(0,), observable_reason="Heldout label.",
        )
    )
    with pytest.raises(DistillationError, match="heldout human labels"):
        prepare_distillation_dataset(selection, labels, teacher_reports(selection), protocol())

    reports = teacher_reports(selection)
    reports[0]["provenance"]["clip_id"] = heldout["clip_id"]
    with pytest.raises(DistillationError, match="heldout teacher reports"):
        prepare_distillation_dataset(selection, development_annotations(selection), reports, protocol())


def test_teacher_provenance_and_complete_development_labels_are_required(tmp_path: Path) -> None:
    selection = frozen_selection(tmp_path)
    reports = teacher_reports(selection)
    reports[0]["provenance"]["media_hash"] = digest("wrong-media")
    with pytest.raises(DistillationError, match="media_hash"):
        prepare_distillation_dataset(selection, development_annotations(selection), reports, protocol())
    with pytest.raises(DistillationError, match="complete supplied human labels"):
        prepare_distillation_dataset(selection, development_annotations(selection)[:-1], teacher_reports(selection), protocol())


def test_development_split_is_task_stratified_and_lineage_safe(tmp_path: Path) -> None:
    selection = frozen_selection(tmp_path)
    clips = [clip for clip in load_frozen_calibration_selection(selection)["clips"] if clip["split"] == DEVELOPMENT_SPLIT]
    first = deterministic_development_split(clips, 44)
    second = deterministic_development_split(clips, 44)
    assert first == second
    for task in TASKS:
        task_ids = [clip["clip_id"] for clip in clips if clip["task"] == task]
        assert sum(first[clip_id] == "validation" for clip_id in task_ids) == 4
        assert sum(first[clip_id] == "training" for clip_id in task_ids) == 16


def test_job_readiness_never_submits_or_guesses_cloud_payload(tmp_path: Path) -> None:
    selection = frozen_selection(tmp_path)
    dataset_path = tmp_path / "dataset.json"
    freeze_distillation_dataset(selection, development_annotations(selection), teacher_reports(selection), protocol(), dataset_path)
    readiness = build_training_job_readiness(dataset_path, {"job_name": "distilled-judge-candidate"})
    assert readiness["status"] == "blocked"
    assert readiness["launch"]["submission_performed"] is False
    assert readiness["launch"]["endpoint"] is None
    assert "account_id_unknown" in readiness["blockers"]
    with pytest.raises(DistillationError, match="credentials"):
        build_training_job_readiness(dataset_path, {"api_key": "do-not-store"})
    with pytest.raises(DistillationError, match="credentials"):
        build_training_job_readiness(dataset_path, {"environment": {"token": "do-not-store"}})
    with pytest.raises(DistillationError, match="credentials"):
        build_training_job_readiness(dataset_path, {"access_token": "do-not-store"})
    output = tmp_path / "job-readiness.json"
    frozen = freeze_training_job_readiness(dataset_path, {"job_name": "candidate"}, output)
    assert frozen["launch"]["spend_authorized"] is False


def test_distilled_revision_needs_fresh_humans_and_paired_frozen_videos(tmp_path: Path) -> None:
    selection = frozen_selection(tmp_path)
    dataset_path = tmp_path / "dataset.json"
    freeze_distillation_dataset(selection, development_annotations(selection), teacher_reports(selection), protocol(), dataset_path)
    blocked = distilled_revision_readiness(dataset_path, "sha256:" + "c" * 64, {})
    assert blocked["status"] == "blocked"
    assert "fresh_heldout_calibration_missing" in blocked["blockers"]

    structural = distilled_revision_readiness(
        dataset_path,
        "sha256:" + "c" * 64,
        {
            "fresh_heldout_calibration": {
                "status": "pass",
                "calibration_manifest_hash": digest("new-heldout-panel"),
                "human_labels_origin": "fresh_human_annotation",
                "copied_labels": False,
                "source_lineage_disjoint_from_development": True,
                "raw_judge_reports_bound": True,
            },
            "paired_frozen_video_comparison": {"status": "pass", "video_manifest_hash": digest("paired-videos")},
        },
    )
    assert structural["status"] == "fresh_evidence_structurally_complete"
    assert structural["primary_scoring_allowed"] is False
