"""Local control-plane tests with a deterministic transport, never a GPU."""

from __future__ import annotations

import base64
import hashlib
import shutil
import subprocess
from io import BytesIO
import time
import pytest

from fastapi.testclient import TestClient
from PIL import Image

from plumb.api import create_app
from plumb.live_demo import LiveDemoService, STATE_MODE


_stream = BytesIO()
Image.new("RGB", (2, 2), (17, 34, 51)).save(_stream, format="PNG")
PNG = _stream.getvalue()


def test_catalog_branch_decodes_last_actual_frame_at_low_fps(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for real clip extraction")
    root = tmp_path / "data"
    root.mkdir()
    for index, color in enumerate(((255, 0, 0), (0, 0, 255))):
        Image.new("RGB", (64, 48), color).save(root / ("source-%03d.png" % index))
    clip = root / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "5", "-i", str(root / "source-%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    digest = "sha256:" + hashlib.sha256(clip.read_bytes()).hexdigest()
    monkeypatch.setattr("plumb.world_videos.world_videos_payload", lambda _root: {"videos": [{
        "id": "actual-low-fps", "sha256": digest, "video_url": "/api/artifacts/source.mp4",
    }]})
    service = LiveDemoService(root, transport=FakeWorker())
    try:
        raw, source = service._source_final_frame("actual-low-fps")
        with Image.open(BytesIO(raw)) as image:
            red, green, blue = image.convert("RGB").getpixel((32, 24))
        assert blue > 240 and red < 10 and green < 10
        assert source["source_sha256"] == digest
    finally:
        service.shutdown()


class FakeWorker:
    """A strict in-memory stand-in for the HTTP worker protocol."""

    def __init__(self):
        self.fixture_calls = 0
        self.steps = []

    def health(self):
        return {"status": "ready"}

    def fixture(self):
        self.fixture_calls += 1
        return {"png_base64": base64.b64encode(PNG).decode("ascii"), "prompt": "Move the pot."}

    def step(self, payload):
        self.steps.append(dict(payload))
        if payload["mode"] == "policy":
            assert "direction" not in payload
        else:
            assert payload["direction"] in {"up", "down", "left", "right", "forward", "back"}
        return {
            "predicted_png_base64": base64.b64encode(PNG).decode("ascii"),
            "conditioning_png_base64": payload["current_png_base64"],
            "action7": [0.0] * 7,
            # The actual worker reports no OpenVLA timing for a manual action.
            "timings_ms": {"policy": None if payload["mode"] == "manual" else 1.25, "world": 2.5},
            "model": {"policy": "OpenVLA", "world": "IRASim"},
            "revision": {"openvla": "test", "irasim": "test"},
            "profile": "fake-test-profile",
            "state_mode": STATE_MODE,
            "warning": "Experimental stateless RGB reconditioning.",
        }


def _wait(client: TestClient, session_id: str):
    session = None
    for _ in range(100):
        session = client.get("/api/demo/sessions/" + session_id).json()
        if session["state"] not in {"queued", "initializing", "running"}:
            return session
        time.sleep(0.01)
    raise AssertionError("session did not settle: %r" % session)


def test_policy_session_is_real_worker_protocol_chain_and_is_persisted(tmp_path):
    worker = FakeWorker()
    service = LiveDemoService(tmp_path / "data", transport=worker)
    with TestClient(create_app(tmp_path / "data", live_demo_service=service)) as client:
        assert client.get("/api/demo/status").json()["available"] is True
        created = client.post("/api/demo/sessions", json={"mode": "policy", "steps": 2})
        assert created.status_code == 202
        result = _wait(client, created.json()["id"])

    assert result["state"] == "completed"
    assert result["completed_steps"] == 2
    assert [frame["role"] for frame in result["frames"]] == ["fixture", "predicted", "predicted"]
    assert worker.fixture_calls == 1 and len(worker.steps) == 2
    # The exact raw PNG bytes returned by the first step become the second
    # step's input.  This proves the local service does not substitute a fake
    # image or retain hidden worker state.
    assert worker.steps[1]["current_png_base64"] == base64.b64encode(PNG).decode("ascii")
    second = result["frames"][2]
    assert second["input_sha256"] == result["frames"][1]["sha256"]
    assert second["conditioning_sha256"] == result["frames"][1]["sha256"]
    assert second["model"] == {"policy": "OpenVLA", "world": "IRASim"}
    assert result["state_mode"] == STATE_MODE
    assert result["qualified"] is False


def test_manual_session_fetches_only_fixture_then_deduplicates_one_active_command(tmp_path):
    worker = FakeWorker()
    service = LiveDemoService(tmp_path / "data", transport=worker)
    with TestClient(create_app(tmp_path / "data", live_demo_service=service)) as client:
        created = client.post("/api/demo/sessions", json={"mode": "manual"})
        assert created.status_code == 202
        session_id = created.json()["id"]
        ready = _wait(client, session_id)
        assert ready["state"] == "ready"
        assert worker.fixture_calls == 1
        assert worker.steps == [], "manual creation must not infer"

        first = client.post(
            "/api/demo/sessions/%s/commands" % session_id,
            json={"direction": "backward", "request_id": "same-key"},
        )
        duplicate = client.post(
            "/api/demo/sessions/%s/commands" % session_id,
            json={"direction": "backward", "request_id": "same-key"},
        )
        assert first.status_code == duplicate.status_code == 202
        assert duplicate.json()["idempotent_replay"] is True
        result = _wait(client, session_id)

    assert result["state"] == "ready"
    assert len(worker.steps) == 1
    assert worker.steps[0]["direction"] == "back"
    assert result["frames"][-1]["provenance"]["direction"] == "back"
    assert result["frames"][-1]["timings_ms"]["policy"] is None
    assert len(result["commands"]) == 1


def test_unconfigured_live_demo_reports_unavailable_and_never_creates_fake_session(tmp_path, monkeypatch):
    monkeypatch.delenv("PLUMB_LIVE_DEMO_URL", raising=False)
    monkeypatch.delenv("PLUMB_LIVE_DEMO_TOKEN_FILE", raising=False)
    with TestClient(create_app(tmp_path / "data")) as client:
        status = client.get("/api/demo/status").json()
        assert status["available"] is False
        response = client.post("/api/demo/sessions", json={"mode": "policy"})
        assert response.status_code == 503
        assert client.get("/api/demo/sessions").json()["sessions"] == []
