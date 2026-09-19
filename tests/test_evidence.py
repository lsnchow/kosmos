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


def test_causal_history_video_keeps_replay_and_training_limitations(tmp_path):
    report = tmp_path / "cluster-evidence" / "history-job" / "report.json"
    write(report, {"kind": "plumb_irasim_causal_history_replay_diagnostic", "status": "completed",
                   "outcome": {"frame_count": 17, "rows": [{}] * 16,
                               "video": {"path": "/remote/history/history.mp4",
                                         "sha256": hashlib.sha256(b"history").hexdigest()}},
                   "total_seconds": 100})
    video = report.parent / "history" / "history.mp4"
    video.parent.mkdir()
    video.write_bytes(b"history")
    row = experiments_payload(tmp_path)["experiments"][0]
    assert row["video_url"].endswith("/history/history.mp4")
    assert row["ticks_completed"] == 16 and row["qualified"] is False
    assert "policy was not queried" in " ".join(row["notes"])
    video.write_bytes(b"tampered")
    assert experiments_payload(tmp_path)["experiments"][0]["video_url"] is None


def test_octo_native_proposals_are_not_world_frames_or_success(tmp_path):
    write(tmp_path / "cluster-evidence" / "octo.json", {
        "kind": "plumb_octo_small_native_two_observation_diagnostic", "status": "completed_unqualified",
        "native_calls": {"first": {"action": [0.0] * 7, "wall_seconds": 1.2}},
        "total_seconds": 10, "runtime": {"gpu_peak_memory_bytes": 1234}})
    row = experiments_payload(tmp_path)["experiments"][0]
    assert row["action_dimensions"] == 7 and row["stage"] == "policy"
    assert row["frame_count"] is None and row["video_url"] is None
    assert row["outcome"] == "unknown" and row["qualified"] is False


def test_production_octo_profile_is_separate_unqualified_diagnostic(tmp_path):
    write(tmp_path / "cluster-evidence" / "production-octo.json", {
        "kind": "plumb_octo_small_production_241fb_two_observation_smoke",
        "status": "completed_unqualified", "qualified": False,
        "native_calls": {"first": {"actions": [[0.0] * 7] * 4, "wall_seconds": 2}},
        "runtime": {"after_native_calls": {"gpu_peak_memory_bytes": 1234}}})
    row = experiments_payload(tmp_path)["experiments"][0]
    assert "241fb" in row["model"]
    assert row["action_dimensions"] == 7 and row["latency_seconds"] == 2
    assert row["qualified"] is False and row["outcome"] == "unknown"
    assert row["frame_count"] is None and row["video_url"] is None


def test_judge_gradient_check_is_not_a_trained_adapter_or_task_score(tmp_path):
    write(tmp_path / "cluster-evidence" / "judge-preflight.json", {
        "kind": "plumb_judge_lora_framework_preflight", "status": "completed_unqualified_preflight",
        "qualified": False, "loss": 0.7, "wall_seconds": 12, "gpu_peak_memory_bytes": 1024})
    row = experiments_payload(tmp_path)["experiments"][0]
    assert row["stage"] == "judge_training_preflight"
    assert row["qualified"] is False and row["outcome"] == "unknown"
    assert "No optimizer step" in " ".join(row["notes"])
    assert row["latency_seconds"] == 12 and row["video_url"] is None


def test_replica_summary_requires_digest_bound_complete_raw_reports(tmp_path):
    summary = tmp_path / "cluster-evidence" / "replicas" / "summary.json"
    worker = summary.parent / "reports" / "worker.json"
    action = {"action": [0.0] * 7, "proposal": [[0.0] * 7] * 4}
    write(worker, {"kind": "plumb_octo_small_native_two_observation_diagnostic",
                   "status": "completed_unqualified", "source_release": "release", "slurm_process_id": "0",
                   "slurm_node": "node", "native_calls": {"backend_calls": 3, "first": action,
                   "second": action, "after_reset_first": action}})
    write(summary, {"kind": "plumb_octo_worker_reproducibility_diagnostic", "status": "completed_unqualified",
                    "source_release": "release", "reports": [{"relative_path": "reports/worker.json",
                    "sha256": "sha256:" + hashlib.sha256(worker.read_bytes()).hexdigest()}]})
    row = experiments_payload(tmp_path)["experiments"][0]
    assert row["stage"] == "policy_reproducibility" and row["qualified"] is False
    assert "every reset/repeat matches: True" in " ".join(row["notes"])
    worker.write_text("{}")
    assert experiments_payload(tmp_path)["experiments"] == []
