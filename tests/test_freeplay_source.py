"""Source-bound free-play refuses MP4-only recordings before generation."""
from __future__ import annotations

import hashlib
import json
import threading

from fastapi.testclient import TestClient

import plumb.api as api_module
from plumb.api import create_app
from plumb.backends.baseten import BasetenBackendSettings, BasetenChainBackend


def _ref(path: str, payload: bytes) -> dict:
    return {"path": path, "sha256": "sha256:" + hashlib.sha256(payload).hexdigest()}


def _recording(root, *, checkpoint: bool = False) -> None:
    evidence = root / "cluster-evidence"
    evidence.mkdir(parents=True)
    video = b"recorded-world-output"
    report = {
        "kind": "cosmos3_nano_diffusers_smoke",
        "status": "completed",
        "result": {"output_artifacts": {"video": _ref("/remote/output.mp4", video)}},
    }
    (evidence / "output.mp4").write_bytes(video)
    if checkpoint:
        state = b"opaque-restorable-world-state"
        (evidence / "checkpoint.bin").write_bytes(state)
        report["source_branch"] = {
            "schema": "plumb-source-branch-v1",
            "source_video_sha256": _ref("ignored", video)["sha256"],
            "adapter": "test-branch-v1",
            "checkpoint": _ref("/remote/checkpoint.bin", state),
        }
    (evidence / "cosmos.json").write_text(json.dumps(report))


def _replace_recording_video(root) -> None:
    """Change the output's content identity while retaining the checkpoint file."""

    evidence = root / "cluster-evidence"
    video = b"recorded-world-output-replaced"
    report_path = evidence / "cosmos.json"
    report = json.loads(report_path.read_text())
    (evidence / "output.mp4").write_bytes(video)
    report["result"]["output_artifacts"]["video"] = _ref("/remote/output.mp4", video)
    report["source_branch"]["source_video_sha256"] = _ref("ignored", video)["sha256"]
    report_path.write_text(json.dumps(report))


def test_mp4_only_recording_reports_both_missing_checkpoint_and_backend(tmp_path):
    root = tmp_path / "data"
    _recording(root)
    with TestClient(create_app(root)) as client:
        video = client.get("/api/world-videos").json()["videos"][0]
        status = client.get("/api/freeplay/status", params={"video_id": video["id"]})
        assert status.status_code == 200
        payload = status.json()
        assert payload["available"] is False
        assert payload["mode"] == "recording_branch"
        assert payload["source"]["video_id"] == video["id"]
        assert payload["source"]["sha256"] == video["sha256"]
        assert payload["binding"]["exact_branch_supported"] is False
        assert "checkpoint" in payload["reason"].lower()
        assert "backend" in payload["reason"].lower()


def test_unsupported_source_rejects_before_generic_freeplay_or_session_fallback(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _recording(root)

    class GenericFreeplayGuard:
        def __init__(self):
            self.calls = 0

        def freeplay_step(self, **_kwargs):
            self.calls += 1
            raise AssertionError("MP4-only source must never call generic free-play")

    guard = GenericFreeplayGuard()
    monkeypatch.setattr(api_module, "_register_baseten_backend", lambda _root: (guard, []))
    with TestClient(create_app(root)) as client:
        video = client.get("/api/world-videos").json()["videos"][0]
        body = {"session_id": "recording1", "direction": "right", "video_id": video["id"], "source_sha256": video["sha256"]}
        refused = client.post("/api/freeplay/step", json=body)
        assert refused.status_code == 409
        assert refused.json()["detail"]["binding"]["exact_branch_supported"] is False
        # Stop is source-bound too: it may not use a generic session to avoid a
        # failed source validation or expose an unrelated frame set.
        assert client.post("/api/freeplay/step", json={**body, "direction": "stop"}).status_code == 409
        assert guard.calls == 0


def test_tampered_checkpoint_is_not_exposed_as_a_source_branch_manifest(tmp_path):
    root = tmp_path / "data"
    _recording(root, checkpoint=True)
    with TestClient(create_app(root)) as client:
        intact = client.get("/api/world-videos").json()["videos"][0]
        assert intact["source_branch"]["checkpoint"]["sha256"].startswith("sha256:")
        (root / "cluster-evidence" / "checkpoint.bin").write_bytes(b"tampered checkpoint")
        video = client.get("/api/world-videos").json()["videos"][0]
        assert video["source_branch"] is None
        status = client.get("/api/freeplay/status", params={"video_id": video["id"]}).json()
        assert status["available"] is False
        assert "checkpoint" in status["reason"].lower()


def test_explicit_source_branch_double_is_the_only_supported_dispatch(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _recording(root, checkpoint=True)

    class SourceBranchDouble:
        """An explicit test adapter, never a production fallback."""

        def __init__(self):
            self.status_calls = []
            self.step_calls = []

        def source_branch_status(self, source):
            self.status_calls.append(source)
            branch = source["source_branch"]
            return {
                "available": True,
                "exact_branch_supported": True,
                "adapter": branch["adapter"],
                "checkpoint_sha256": branch["checkpoint"]["sha256"],
                "reason": "test adapter has verified this checkpoint",
            }

        def source_branch_step(self, **kwargs):
            self.step_calls.append(kwargs)
            return {
                "frame_urls": ["/api/artifacts/freeplay/test/frame.png"],
                "frame_count": 1,
                "requested_resolution": kwargs["resolution"],
                "frame_height": None,
                "frame_width": None,
            }

    adapter = SourceBranchDouble()
    monkeypatch.setattr(api_module, "_register_baseten_backend", lambda _root: (adapter, []))
    with TestClient(create_app(root)) as client:
        video = client.get("/api/world-videos").json()["videos"][0]
        status = client.get("/api/freeplay/status", params={"video_id": video["id"]}).json()
        assert status["available"] is True
        assert status["mode"] == "recording_branch"
        assert status["binding"]["exact_branch_supported"] is True
        body = {"session_id": "sourcebranch1", "direction": "right", "video_id": video["id"], "source_sha256": video["sha256"]}
        response = client.post("/api/freeplay/step", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["mode"] == "recording_branch"
        assert len(adapter.step_calls) == 1
        call = adapter.step_calls[0]
        assert call["source"]["video_id"] == video["id"]
        assert call["source"]["sha256"] == video["sha256"]
        assert len(call["actions"]) == 16
        # A session cannot switch from the exact source branch to legacy
        # generic mode, even for release-to-stop.
        assert client.post("/api/freeplay/step", json={"session_id": "sourcebranch1", "direction": "stop"}).status_code == 409


def test_source_session_rejects_a_new_catalog_hash_after_its_first_step(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _recording(root, checkpoint=True)

    class Adapter:
        def __init__(self):
            self.step_calls = 0

        def source_branch_status(self, source):
            branch = source["source_branch"]
            return {
                "available": True,
                "exact_branch_supported": True,
                "adapter": branch["adapter"],
                "checkpoint_sha256": branch["checkpoint"]["sha256"],
            }

        def source_branch_step(self, **_kwargs):
            self.step_calls += 1
            return {"frame_urls": [], "frame_count": 0}

    adapter = Adapter()
    monkeypatch.setattr(api_module, "_register_baseten_backend", lambda _root: (adapter, []))
    with TestClient(create_app(root)) as client:
        first = client.get("/api/world-videos").json()["videos"][0]
        first_body = {"session_id": "sourcehash1", "direction": "right", "video_id": first["id"], "source_sha256": first["sha256"]}
        assert client.post("/api/freeplay/step", json=first_body).status_code == 200
        _replace_recording_video(root)
        replacement = client.get("/api/world-videos").json()["videos"][0]
        assert replacement["sha256"] != first["sha256"]
        refused = client.post(
            "/api/freeplay/step",
            json={**first_body, "video_id": replacement["id"], "source_sha256": replacement["sha256"]},
        )
        assert refused.status_code == 409
        assert adapter.step_calls == 1


def test_source_session_refuses_overlapping_generation(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _recording(root, checkpoint=True)

    class BlockingAdapter:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def source_branch_status(self, source):
            branch = source["source_branch"]
            return {
                "available": True,
                "exact_branch_supported": True,
                "adapter": branch["adapter"],
                "checkpoint_sha256": branch["checkpoint"]["sha256"],
            }

        def source_branch_step(self, **_kwargs):
            self.started.set()
            assert self.release.wait(2)
            return {"frame_urls": [], "frame_count": 0}

    adapter = BlockingAdapter()
    monkeypatch.setattr(api_module, "_register_baseten_backend", lambda _root: (adapter, []))
    with TestClient(create_app(root)) as client:
        video = client.get("/api/world-videos").json()["videos"][0]
        body = {"session_id": "busybranch1", "direction": "right", "video_id": video["id"], "source_sha256": video["sha256"]}
        result = []
        worker = threading.Thread(target=lambda: result.append(client.post("/api/freeplay/step", json=body)))
        worker.start()
        assert adapter.started.wait(2)
        concurrent = client.post("/api/freeplay/step", json=body)
        assert concurrent.status_code == 409
        assert "in flight" in concurrent.json()["detail"]["reason"]
        adapter.release.set()
        worker.join(timeout=2)
        assert len(result) == 1 and result[0].status_code == 200


def test_malformed_source_adapter_result_clears_busy_and_quarantines_session(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _recording(root, checkpoint=True)

    class BadResultAdapter:
        def source_branch_status(self, source):
            branch = source["source_branch"]
            return {
                "available": True,
                "exact_branch_supported": True,
                "adapter": branch["adapter"],
                "checkpoint_sha256": branch["checkpoint"]["sha256"],
            }

        def source_branch_step(self, **_kwargs):
            return {"frame_urls": "not-a-list"}

    monkeypatch.setattr(api_module, "_register_baseten_backend", lambda _root: (BadResultAdapter(), []))
    with TestClient(create_app(root)) as client:
        video = client.get("/api/world-videos").json()["videos"][0]
        body = {"session_id": "badresult1", "direction": "right", "video_id": video["id"], "source_sha256": video["sha256"]}
        assert client.post("/api/freeplay/step", json=body).status_code == 502
        quarantined = client.post("/api/freeplay/step", json=body)
        assert quarantined.status_code == 409
        assert "cannot be retried automatically" in quarantined.json()["detail"]["reason"]


def test_baseten_backend_forwards_only_an_injected_source_branch_adapter(tmp_path):
    class Adapter:
        def source_branch_status(self, source):
            assert source["sha256"] == "sha256:" + "a" * 64
            return {"available": True, "exact_branch_supported": True}

        def source_branch_step(self, **kwargs):
            return {"frame_count": len(kwargs["actions"])}

    settings = BasetenBackendSettings(
        webhook_endpoint="https://plumb.example.test/api/callbacks",
        operating_point_id="test-op",
    )
    backend = BasetenChainBackend(settings=settings, data_dir=tmp_path, source_branch_adapter=Adapter())
    source = {"sha256": "sha256:" + "a" * 64}
    assert backend.source_branch_status(source)["exact_branch_supported"] is True
    assert backend.source_branch_step(
        session_id="s1", source=source, actions=[[0.0] * 7], protocol_hash="sha256:" + "b" * 64, seed=1, resolution=480
    )["frame_count"] == 1
