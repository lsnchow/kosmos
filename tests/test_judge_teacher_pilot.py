"""Offline invariants for unqualified teacher-label collection."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from cluster import judge_teacher_pilot as pilot
from plumb.policies.provenance import image_pixel_hash
from plumb.policies.tasks import BENCHMARK_TASK_REGISTRY, TASK_REGISTRY_HASH


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw(label):
    return json.dumps(
        {
            "integrity": label[0],
            "collision": label[1],
            "progress": label[2],
            "completion_evidence": label[3],
            "evidence_frame_indices": [0, 15],
            "observable_reasons": "visible fixture evidence",
        },
        sort_keys=True,
    )


def _report(labels):
    samples = []
    for index, label in enumerate(labels):
        if label is None:
            attempt = {"sample_index": index, "attempt_index": 0, "raw_output": None, "parsed": None, "failure_reason": "schema_error"}
        else:
            raw = _raw(label)
            attempt = {"sample_index": index, "attempt_index": 0, "raw_output": raw, "parsed": json.loads(raw), "failure_reason": None}
        samples.append({"sample_index": index, "attempts": [attempt]})
    return {"raw_judge_samples": samples}


def test_unique_modal_selection_uses_an_actual_lowest_index_raw_member_without_synthesis():
    modal = ("intact", "none_visible", 4, "not_met")
    report = _report([modal, ("uncertain", "uncertain", None, "uncertain"), modal, None, ("artifact", "visible", 0, "not_met")])

    selection = pilot._selection(report)

    assert selection["status"] == "accepted_unique_modal_tuple"
    assert selection["modal_count"] == 2
    assert selection["quorum_reached"] is False
    assert selection["selected"]["sample_index"] == 0
    assert selection["selected"]["label"] == json.loads(report["raw_judge_samples"][0]["attempts"][0]["raw_output"])


def test_modal_tie_and_no_valid_sample_are_explicit_exclusions():
    first = ("intact", "none_visible", 4, "not_met")
    second = ("artifact", "visible", 0, "not_met")

    assert pilot._selection(_report([first, first, second, second, None]))["status"] == "excluded_modal_tie"
    assert pilot._selection(_report([None, None, None, None, None]))["status"] == "excluded_no_schema_valid_sample"


def test_semantic_agreement_order_handles_null_and_integer_progress_deterministically():
    report = _report(
        [
            ("intact", "none_visible", None, "uncertain"),
            ("intact", "none_visible", 4, "not_met"),
            ("intact", "none_visible", None, "uncertain"),
            ("artifact", "visible", 0, "not_met"),
            None,
        ]
    )

    selection = pilot._selection(report)

    assert selection["status"] == "accepted_unique_modal_tuple"
    assert selection["selected"]["semantic_tuple"] == ["intact", "none_visible", None, "uncertain"]
    assert len(selection["agreement"]) == 3


def test_candidate_manifest_requires_array_pixel_hashes_actual_timestamps_and_fixed_split(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    records = []
    for index in range(16):
        value = np.full((3, 4, 3), index, dtype=np.uint8)
        path = frames_dir / ("%02d.png" % index)
        Image.fromarray(value, "RGB").save(path)
        records.append(
            {
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": _sha(path),
                "pixel_sha256": image_pixel_hash(np.asarray(Image.open(path).convert("RGB")))["sha256"],
                "timestamp": float(index) / 5.0,
            }
        )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"real-video-placeholder")
    reference = frames_dir / "00.png"
    task = BENCHMARK_TASK_REGISTRY.get("close_drawer")
    clips = []
    for index in range(16):
        clips.append(
            {
                "clip_id": "clip-%02d" % index,
                "source_lineage_id": "lineage-%02d" % index,
                "split": "train" if index < 12 else "development_validation",
                "video": {"path": "clip.mp4", "sha256": _sha(video)},
                "frames": records,
                "reference": {
                    "path": reference.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(reference),
                    "pixel_sha256": image_pixel_hash(np.asarray(Image.open(reference).convert("RGB")))["sha256"],
                    "source_uri": "real-bridge://scene-context/clip-%02d" % index,
                    "role": "scene_context_not_goal",
                },
            }
        )
    manifest = {
        "schema": pilot.INPUT_SCHEMA,
        "qualified": False,
        "selection_sha256": pilot.FROZEN_SELECTION_SHA256,
        "task": {"task_id": "close_drawer", "task_registry_hash": TASK_REGISTRY_HASH, "rubric_hash": task.rubric_hash},
        "clips": clips,
    }

    checked = pilot.validate_candidate_manifest(tmp_path, manifest)

    assert len(checked["clips"]) == 16
    assert checked["clips"][0]["frames"][0]["timestamp"] == 0.0
    manifest["clips"][0]["actions"] = [[0.0] * 7]
    with pytest.raises(pilot.TeacherPilotError, match="barred judge-input field"):
        pilot.validate_candidate_manifest(tmp_path, manifest)


def test_label_row_keeps_relative_raw_report_ref_and_full_actual_structured_label(tmp_path):
    root = tmp_path
    raw_path = root / "rawteacher" / "clip-00" / ("a" * 64) / "attempt-0001.json"
    raw_path.parent.mkdir(parents=True)
    label = json.loads(_raw(("intact", "none_visible", 5, "met")))
    clip = {
        "clip_id": "clip-00",
        "source_lineage_id": "lineage-00",
        "cohort": "train",
        "timestamp_semantics": "unverified_nominal_dataset_times",
        "frames": [{"path": "frames/00.png", "sha256": "a" * 64, "timestamp": 0.1}] * 16,
        "reference": {"path": "frames/ref.png", "sha256": "b" * 64, "role": "scene_context_not_goal", "source_uri": "real-bridge://scene"},
    }
    record = {"clip": clip, "teacher_prompt": {"instruction": "Close the drawer", "rubric": pilot._teacher_rubric()}, "selection": {"status": "accepted_unique_modal_tuple", "selected": {"sample_index": 2, "semantic_tuple": ["intact", "none_visible", 5, "met"], "label": label}, "modal_count": 3, "quorum_reached": True}}
    raw_path.write_text(json.dumps(record), encoding="utf-8")

    row = pilot._label_row(root, raw_path, record)

    assert row["cohort"] == "uncalibrated_pilot_train"
    assert row["label"] == label
    assert row["raw_teacher_refs"][0]["uri"] == "rawteacher/clip-00/" + ("a" * 64) + "/attempt-0001.json"
    assert row["reference_images"][0]["role"] == "scene_context_not_goal"
def test_teacher_accepts_real_downloader_manifest_schema_and_checks_bytes(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    config = root / "config.json"
    config.write_text("{}")
    manifest = root / "PLUMB-TRAINING-MANIFEST.json"
    manifest.write_text(json.dumps({"repo": pilot.QWEN_MODEL_ID, "revision": pilot.QWEN_REVISION,
                                    "file_records": [{"path": "config.json", "bytes": 2, "sha256": _sha(config)}]}))
    result = pilot._model_binding(root, manifest)
    assert result["revision"] == pilot.QWEN_REVISION
    config.write_text("changed")
    with pytest.raises(ValueError, match="size"):
        pilot._model_binding(root, manifest)
