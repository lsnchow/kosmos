import json

import numpy as np
from PIL import Image
import pytest

from cluster.prepare_judge_preflight import digest, prepare
from plumb.policies.provenance import image_pixel_hash


def source(tmp_path, monkeypatch):
    # Unit-test transport fixture, never a model report or scientific label.
    frames = tuple(np.full((2, 2, 3), i, dtype=np.uint8) for i in range(16))
    monkeypatch.setattr("cluster.prepare_judge_preflight._load_video_frames", lambda _: frames)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"unit-test-video")
    reference = tmp_path / "ref.png"
    Image.fromarray(frames[0]).save(reference)
    label = {"integrity": "uncertain", "collision": "uncertain", "progress": None,
             "completion_evidence": "uncertain", "evidence_frame_indices": [0], "observable_reasons": "Unclear."}
    attempt = {"parsed": label, "raw_output": json.dumps(label), "attempt_index": 0, "prompt_variant": "base_v2"}
    report = {"kind": "plumb_local_policy_smoke", "command": "judge", "model": {"id": "unit-test-model"},
              "inputs": {"mode": "diagnostic", "reference_role": "scene", "generated_video": {"sha256": digest(video)},
                         "reference": {"sha256": digest(reference)}, "selected_frame_indices": list(range(16)),
                         "source_video_frame_count": 16, "frame_timestamps": [i / 5 for i in range(16)],
                         "task_instruction": "Close the drawer", "task_rubric": "Use only visible evidence."},
              "raw_judge_votes": {"judge_status": "unknown", "missing_reason": "judge_insufficient_quorum",
                                  "provenance": {"evidence_hashes": {"frame_pixel_hashes": [image_pixel_hash(f)["sha256"] for f in frames]}},
                                  "raw_judge_samples": [{"sample_index": 0, "attempts": [attempt]}]}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return path, video, reference, report


def test_real_input_binding_is_preserved_without_turning_vote_into_label(tmp_path, monkeypatch):
    report, video, reference, _ = source(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    row = prepare(report, video, reference, output)
    assert row["purpose"] == "framework_preflight_only"
    assert row["eligible_for_training_dataset"] is False and row["qualified"] is False
    assert row["teacher_aggregation_status"] == "unknown"
    assert row["teacher_missing_reason"] == "judge_insufficient_quorum"
    assert len(row["frames"]) == 16 and len(row["frame_sha256"]) == 16
    assert row["reference_images"][0]["role"] == "scene"
    assert "cohort" not in row
    assert json.loads((output / "row.json").read_text()) == row


def test_video_hash_mismatch_blocks_before_any_write(tmp_path, monkeypatch):
    report, video, reference, _ = source(tmp_path, monkeypatch)
    video.write_bytes(b"different-video")
    with pytest.raises(ValueError, match="bytes do not match"):
        prepare(report, video, reference, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_decoder_pixel_mismatch_blocks_not_silently_resized(tmp_path, monkeypatch):
    report, video, reference, _ = source(tmp_path, monkeypatch)
    monkeypatch.setattr("cluster.prepare_judge_preflight._load_video_frames", lambda _: tuple(np.ones((2, 2, 3), dtype=np.uint8) for _ in range(16)))
    with pytest.raises(ValueError, match="pixels differ"):
        prepare(report, video, reference, tmp_path / "prepared")


def test_raw_sample_and_persisted_parsed_target_must_match(tmp_path, monkeypatch):
    report, video, reference, payload = source(tmp_path, monkeypatch)
    payload["raw_judge_votes"]["raw_judge_samples"][0]["attempts"][0]["parsed"]["observable_reasons"] = "Altered target"
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs from raw"):
        prepare(report, video, reference, tmp_path / "prepared")


def test_existing_fixture_is_never_overwritten(tmp_path, monkeypatch):
    report, video, reference, _ = source(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    prepare(report, video, reference, output)
    before = (output / "row.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        prepare(report, video, reference, output)
    assert before == (output / "row.json").read_bytes()
