import hashlib
import json

from plumb.evidence import experiments_payload


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_closed_loop_is_unqualified_and_media_hash_bound(tmp_path):
    path = tmp_path / "cluster-evidence" / "loop" / "report.json"
    payload = {"kind": "plumb_irasim_openvla_closed_loop_diagnostic", "status": "completed",
               "qualified": True, "binary_success": True, "frames_persisted": 17,
               "ticks_completed": 16, "ticks_requested": 16, "timing": {"total_seconds": 142},
               "video": {"path": "/remote/evidence/loop/movie.mp4", "sha256": hashlib.sha256(b"media").hexdigest()}}
    write(path, payload)
    video = path.parent / "movie.mp4"
    video.write_bytes(b"media")
    report = experiments_payload(tmp_path)
    row = report["experiments"][0]
    assert report["qualified"] is row["qualified"] is False
    assert row["outcome"] == "unknown"
    assert row["latency_seconds"] is None and row["total_seconds"] == 142
    assert row["video_url"] == "/api/artifacts/cluster-evidence/loop/movie.mp4"
    video.write_bytes(b"replaced")
    assert experiments_payload(tmp_path)["experiments"][0]["video_url"] is None


def test_malformed_reports_symlinks_and_nonfinite_numbers(tmp_path):
    evidence = tmp_path / "cluster-evidence"
    write(evidence / "junk.json", {"kind": "synthetic"})
    write(evidence / "null.json", None)
    write(evidence / "cosmos.json", {"kind": "cosmos3_nano_diffusers_smoke", "status": "completed",
                                    "result": {"timing": {"wall_seconds": float("nan")}}})
    external = tmp_path / "outside.json"
    write(external, {"kind": "irasim_original_one_step_smoke"})
    (evidence / "escape.json").symlink_to(external)
    rows = experiments_payload(tmp_path)["experiments"]
    assert len(rows) == 1 and rows[0]["latency_seconds"] is None
    json.dumps(rows, allow_nan=False)


def test_nested_judge_and_irasim_types_keep_timing_scopes(tmp_path):
    write(tmp_path / "cluster-evidence" / "loop" / "judge.json", {
        "kind": "plumb_local_policy_smoke", "command": "judge", "status": "completed_unqualified",
        "timing": {"judge_call_seconds": 29}, "outcome": {"judge_status": "unknown"}})
    write(tmp_path / "cluster-evidence" / "irasim.json", {
        "kind": "irasim_original_one_step_smoke", "status": "completed", "frame_count": 2,
        "timing": {"wall_seconds": 2.55, "model_load_seconds": 23.1}})
    rows = experiments_payload(tmp_path)["experiments"]
    assert {r["stage"] for r in rows} == {"world", "judge"}
    assert {r["timing_scope"] for r in rows} == {
        "world_inference_excludes_model_load", "five_sample_judge_call_including_lazy_load"}
    assert all(r["outcome"] == "unknown" for r in rows)


def test_native_horizon_reference_does_not_imply_causal_feedback(tmp_path):
    write(tmp_path / "cluster-evidence" / "native" / "report.json", {
        "kind": "plumb_irasim_native_open_loop_reference_diagnostic",
        "status": "completed_unqualified_reference", "artifacts": {"mp4_frame_count": 16},
        "timing": {"total_seconds": 34.5}})
    row = experiments_payload(tmp_path)["experiments"][0]
    assert row["stage"] == "open_loop_reference"
    assert row["total_seconds"] == 34.5 and row["latency_seconds"] is None
    assert row["qualified"] is False and row["outcome"] == "unknown"
    assert "14 future action" in " ".join(row["notes"])
