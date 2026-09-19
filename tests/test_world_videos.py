import hashlib
import json

from fastapi.testclient import TestClient

from plumb.api import create_app
from plumb import world_videos


def write_report(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def ref(path, data):
    return {"path": path, "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "frame_count": 17}


def test_catalog_binds_only_generated_output_not_source_media(tmp_path, monkeypatch):
    monkeypatch.setattr(world_videos, "_probe", lambda *_: {"frame_count": 17, "duration_seconds": 3.4})
    folder = tmp_path / "cluster-evidence"
    report = folder / "cosmos.json"
    write_report(report, {
        "kind": "cosmos3_nano_diffusers_smoke", "status": "completed",
        "provenance": {"vision_fixture": ref("/remote/input.mp4", b"input")},
        "result": {"output_artifacts": {"video": ref("/remote/generated.mp4", b"generated")}},
        "qualified": True, "binary_success": True,
    })
    (folder / "input.mp4").write_bytes(b"input")
    (folder / "generated.mp4").write_bytes(b"generated")
    payload = world_videos.world_videos_payload(tmp_path)
    assert payload["total"] == 1 and payload["generation_jobs_submitted"] == 0
    clip = payload["videos"][0]
    assert clip["video_url"].endswith("generated.mp4")
    assert clip["provenance"] == "recorded_model_output" and clip["qualified"] is False
    assert "binary_success" not in clip and clip["duration_seconds"] == 3.4
    (folder / "generated.mp4").write_bytes(b"tampered")
    assert world_videos.world_videos_payload(tmp_path)["total"] == 0


def test_nested_repeat_artifacts_remain_distinct_recordings(tmp_path, monkeypatch):
    monkeypatch.setattr(world_videos, "_probe", lambda *_: {})
    folder = tmp_path / "cluster-evidence" / "history-job"
    write_report(folder / "report.json", {
        "kind": "plumb_irasim_causal_history_replay_diagnostic", "status": "completed",
        "outcome": {"video": ref("/remote/history-job/history/history.mp4", b"video")},
        "same_seed_repeat": {"video": ref("/remote/history-job/repeat/history/history.mp4", b"video")},
    })
    for name in ("history/history.mp4", "repeat/history/history.mp4"):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
    clips = world_videos.world_videos_payload(tmp_path)["videos"]
    assert len(clips) == 2 and len({c["id"] for c in clips}) == 2
    assert len({c["video_url"] for c in clips}) == 2
    assert all(c["duration_seconds"] is None for c in clips)


def test_catalog_excludes_unknown_reports_symlinks_and_source_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(world_videos, "_probe", lambda *_: {})
    folder = tmp_path / "cluster-evidence"
    write_report(folder / "unknown.json", {"kind": "synthetic", "video": ref("/remote/source.mp4", b"media")})
    write_report(folder / "world.json", {"kind": "irasim_original_one_step_smoke", "artifacts": {"video": ref("/remote/source.mp4", b"media")}})
    (tmp_path / "external.mp4").write_bytes(b"media")
    (folder / "source.mp4").symlink_to(tmp_path / "external.mp4")
    assert world_videos.world_videos_payload(tmp_path)["total"] == 0
    assert world_videos._local_video(folder / "world.json", ref("../../external.mp4", b"media"), tmp_path) is None


def test_api_uses_actual_data_root_and_leaves_run_ledger_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(world_videos, "_probe", lambda *_: {})
    root = tmp_path / "served"
    path = root / "cluster-evidence" / "cosmos.json"
    write_report(path, {"kind": "cosmos3_nano_diffusers_smoke", "result": {"output_artifacts": {"video": ref("/remote/output.mp4", b"video")}}})
    (path.parent / "output.mp4").write_bytes(b"video")
    with TestClient(create_app(root)) as client:
        before = client.get("/api/runs").json()
        response = client.get("/api/world-videos")
        assert response.status_code == 200
        clip = response.json()["videos"][0]
        assert client.get(clip["video_url"]).content == b"video"
        assert client.get("/api/runs").json() == before
