"""Receipt-journal tests for the Cloudflare-free matched comparison service."""

import base64
import hashlib
import json
import subprocess
import sys
import time
import io
import pytest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from plumb.api import create_app
from plumb.comparisons import (
    COMPARISON_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    STREAM_SCHEMA,
    BasetenCliStreamTransport,
    ComparisonService,
    ComparisonConflict,
    ComparisonNotReady,
)
from plumb.records import canonical_json


# A valid small PNG is enough here: the service checks receipt/file binding, not
# visual quality.  No fixture ever reaches the production HTTP surface because
# the only feed seam is test_mode-gated.
_PNG_BUFFER = io.BytesIO()
Image.new("RGB", (4, 3), (32, 64, 96)).save(_PNG_BUFFER, format="PNG")
PNG = _PNG_BUFFER.getvalue()


def digest(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def state(payload=None):
    payload = payload or {"values": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.4]}
    raw = canonical_json(payload).encode("utf-8")
    return {
        "payload": payload,
        "origin": "source-measured-bridge-v1",
        "conventions_sha256": digest(b"bridge-conventions-v1"),
        "sha256": digest(raw),
    }


def write_manifest(root: Path) -> None:
    (root / "starts").mkdir(parents=True)
    (root / "starts" / "close-drawer.png").write_bytes(PNG)
    evidence = digest(b"manifest-evidence")
    policy = lambda identity: {
        "identity": identity,
        "source_sha256": digest((identity + "source").encode()),
        "normalizer_sha256": digest((identity + "normalizer").encode()),
        "runtime_sha256": digest((identity + "runtime").encode()),
        "readiness": {"status": "ready", "evidence_sha256": evidence},
        "rights": {"status": "approved", "evidence_sha256": evidence},
        "fidelity": {"status": "accepted", "evidence_sha256": evidence},
    }
    workers = {}
    for name in ("controller", "world", "OpenVLA", "MiniVLA", "Octo-Small"):
        workers[name] = {
            "verified_usd_per_hour": 1.0,
            "bound_seconds": 20.0,
            "residency_margin": 1.5,
            "bound_components": {
                "cold_load_seconds": 2.0,
                "execution_seconds": 10.0,
                "scale_down_tail_seconds": 2.0,
            },
            "evidence_sha256": evidence,
        }
    manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "task": "close_drawer",
        "task_instruction": "Close the drawer",
        "start": {
            "id": "real-source-start-001",
            "png_path": "starts/close-drawer.png",
            "sha256": digest(PNG),
            "state": state(),
        },
        "policies": {
            "OpenVLA": policy("openvla-native"),
            "MiniVLA": policy("minivla-native"),
            "Octo-Small": policy("octo-small-v1"),
        },
        "world": {
            "identity": "cosmos-nano-480-30",
            "profile_sha256": digest(b"cosmos-profile"),
            "asset_sha256": digest(b"cosmos-asset"),
            "deployment_sha256": digest(b"cosmos-deployment"),
            "bridge_control_sha256": digest(b"bridge-control"),
            "gripper_semantics_sha256": digest(b"bridge-gripper"),
            "seed_convention_sha256": digest(b"world-seed-convention"),
            "native_action_dim": 7,
            "runtime_profile": {
                "backend": "cosmos", "profile_id": "cosmos-test-480",
                "domain": "bridge_orig_lerobot", "model_id": "nvidia/Cosmos3-Nano",
                "model_revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92",
                "resolution_tier": 480, "inference_steps": 30, "asset_manifest_id": "test-assets",
            },
            "bridge_control_profile_id": "bridge-test",
            "held_gripper_action": 0.4,
            "manual_seed": 104,
            "manual": {"backend": "cosmos", "action_rows": 16, "structural_frames": 17},
            "readiness": {"status": "ready", "evidence_sha256": evidence},
            "rights": {"status": "approved", "evidence_sha256": evidence},
            "fidelity": {"status": "accepted", "evidence_sha256": evidence},
        },
        "chain": {
            "run_remote_url": "https://chain-verified.api.baseten.co/production/run_remote",
            "cli_profile": "plumb-api",
            "timeout_seconds": 60,
        },
        "pricing": {"source_sha256": digest(b"measured-pricing"), "workers": workers},
    }
    (root / "comparison-source-manifest.json").write_text(canonical_json(manifest), encoding="utf-8")


class StreamFixture:
    """A transport, not a direct service feed, so dispatch follows production."""

    def __init__(self, *, terminal=True):
        self.requests = []
        self.terminal = terminal

    def stream(self, *, run_remote_url, cli_profile, payload, timeout_seconds):
        assert run_remote_url.endswith("/run_remote")
        assert cli_profile == "plumb-api"
        self.requests.append(dict(payload))
        dispatch = payload["request"]
        manual = dispatch["kind"] == "manual_world_segment"
        common = {
            "schema": STREAM_SCHEMA,
            "comparison_id": dispatch["comparison_id"],
            "attempt_id": dispatch["attempt_id"],
            "cell_id": dispatch["cell_id"],
            "mode": "manual" if manual else "comparison",
        }
        header = {
            **common,
            "kind": "header",
            "event_id": "header",
            "task": "close_drawer",
            "task_instruction": "Close the drawer",
            "identity_hashes": dispatch["identity_hashes"],
        }
        if manual:
            header.update({
                "command_id": dispatch["command_id"],
                "source_event_id": dispatch["source"]["event_id"],
                "source_png_sha256": dispatch["source"]["png_sha256"],
            })
        else:
            header.update({"policy": dispatch["policy"], "world_seed": dispatch["world_seed"]})
        records = [header]
        count = 16 if manual else 70
        for index in range(1, count + 1):
            record = {
                **common,
                "kind": "frame",
                "event_id": "frame-%03d" % index,
                "segment_id": "manual-segment" if manual else "rollout",
                "frame_index": index,
                "png_base64": base64.b64encode(PNG).decode("ascii"),
                "png_sha256": digest(PNG),
                "state": state({"values": [float(index)] * 7}),
            }
            if manual:
                record["command_id"] = dispatch["command_id"]
            else:
                record["action"] = {"raw_proposal": [0.0] * 7, "executed": [0.0] * 7}
            records.append(record)
        if self.terminal:
            terminal = {**common, "kind": "terminal", "event_id": "terminal", "status": "completed", "frame_count": count}
            if manual:
                terminal["command_id"] = dispatch["command_id"]
            else:
                terminal["action_count"] = 70
            records.append(terminal)
        encoded = b"".join(canonical_json(record).encode("utf-8") + b"\n" for record in records)
        # Arbitrary chunk boundaries exercise the NDJSON parser, including one
        # boundary inside a Base64 field.
        for offset in range(0, len(encoded), 137):
            yield encoded[offset:offset + 137]


def wait_for(service, comparison_id, statuses=("completed", "failed", "interrupted")):
    deadline = time.time() + 10
    while time.time() < deadline:
        value = service.get_comparison(comparison_id)
        if value["status"] in statuses:
            return value
        time.sleep(0.02)
    raise AssertionError("comparison did not settle")


def test_full_fixed_wall_persists_committed_frames_and_manual_branch(tmp_path):
    root = tmp_path / "data"
    write_manifest(root)
    fixture = StreamFixture()
    service = ComparisonService(root, transport=fixture, test_mode=True)
    try:
        readiness = service.readiness().payload()
        assert readiness["available"] is True
        assert readiness["mechanical"]["status"] == "ready"
        assert readiness["demo_quality"]["status"] == "approved"
        quote = service.create_quote()
        created, idempotent = service.create_comparison(quote["id"], "fixed-wall")
        assert idempotent is False and len(created["cells"]) == 12
        finished = wait_for(service, created["id"])
        assert finished["status"] == "completed"
        assert finished["origin"] == finished["execution_origin"] == "fresh"
        assert [(cell["policy"], cell["seed"]) for cell in finished["cells"]] == [
            (policy, seed) for seed in (101, 102, 103, 104) for policy in ("OpenVLA", "MiniVLA", "Octo-Small")
        ]
        assert all(cell["frame_count"] == cell["action_count"] == 70 for cell in finished["cells"])
        assert all(len(cell["frames"]) == 70 and cell["latest_frame"] == cell["frames"][-1] for cell in finished["cells"])
        # Repeated client submissions do not submit another Chain request.
        same, idempotent = service.create_comparison(quote["id"], "fixed-wall")
        assert idempotent is True and same["id"] == created["id"]
        assert len(fixture.requests) == 12
        source = service.create_manual_session(created["id"], finished["cells"][0]["id"])
        command, command_idempotent = service.create_manual_command(source["id"], "right", "right-1")
        assert command_idempotent is False and command["action_rows"] == 16
        deadline = time.time() + 5
        while time.time() < deadline:
            manual = service.get_manual_session(source["id"])
            if manual["commands"][0]["status"] == "completed":
                break
            time.sleep(0.02)
        assert manual["commands"][0]["frame_count"] == 16
        assert len(manual["commands"][0]["frames"]) == 16
        assert all(row[-1] == 0.4 for row in fixture.requests[-1]["request"]["actions"])
        with pytest.raises(ComparisonConflict, match="another direction"):
            service.create_manual_command(source["id"], "left", "right-1")
        first_final = manual["commands"][0]["frames"][-1]
        second, _ = service.create_manual_command(source["id"], "up", "up-1")
        assert second["status"] == "running"
        deadline = time.time() + 5
        while time.time() < deadline:
            manual = service.get_manual_session(source["id"])
            if len(manual["commands"]) == 2 and manual["commands"][1]["status"] == "completed":
                break
            time.sleep(0.02)
        assert manual["source"]["event_id"] == manual["commands"][1]["frames"][-1]["event_id"]
        assert fixture.requests[-1]["request"]["source"]["event_id"] == first_final["event_id"]
        service.promote(created["id"])
        presentation = service.presentation()
        assert presentation["status"] == "available"
        assert presentation["presentation"]["id"] == created["id"]
        assert presentation["presentation"]["origin"] == "precomputed"
        assert all(cell["origin"] == "precomputed" and cell["execution_origin"] == "fresh" for cell in presentation["presentation"]["cells"])
        budget = service.readiness().payload()["budget"]
        assert budget["reserved_usd"] > quote["reservation_usd"]
        service.budget_cap_usd = budget["reserved_usd"]
        with pytest.raises(ComparisonNotReady, match="remaining reserved budget"):
            service.create_manual_command(source["id"], "right", "beyond-cap")
    finally:
        service.shutdown()


def test_missing_terminal_stays_interrupted_and_is_not_promotable(tmp_path):
    root = tmp_path / "data"
    write_manifest(root)
    service = ComparisonService(root, transport=StreamFixture(terminal=False), test_mode=True)
    try:
        quote = service.create_quote()
        created, _ = service.create_comparison(quote["id"], "truncated-wall")
        finished = wait_for(service, created["id"])
        assert finished["status"] == "interrupted"
        assert any(cell["status"] == "interrupted" for cell in finished["cells"])
        assert service.presentation()["status"] == "unavailable"
    finally:
        service.shutdown()


def test_api_is_read_only_without_a_verified_source_manifest(tmp_path):
    root = tmp_path / "data"
    service = ComparisonService(root, transport=StreamFixture(), test_mode=True)
    try:
        with TestClient(create_app(root, comparison_service=service)) as client:
            readiness = client.get("/api/comparisons/readiness")
            assert readiness.status_code == 200
            assert readiness.json()["available"] is False
            assert client.get("/api/comparisons/presentation").json()["status"] == "unavailable"
            # Readiness and presentation never submit a generation request.
            assert fixture_requests(service) == 0
            assert client.post("/api/comparisons/quote", json={}).status_code == 409
    finally:
        # FastAPI lifespan already calls shutdown; repeated shutdown is safe.
        service.shutdown()


def fixture_requests(service):
    return len(getattr(service.transport, "requests", []))


def test_real_service_payload_runs_controller_and_returns_to_durable_receiver(tmp_path):
    """Full CPU boundary test; only policy/world inference are explicit fakes.

    This must not be mistaken for Cosmos supporting native one-action turns.
    It verifies the actual local payload/Chain translator/stream/SQLite path.
    """
    import asyncio
    from test_demo_chain import demo, _Policy, _World

    class ControllerTransport:
        def __init__(self):
            self.world = _World()
            self.policies = {name: _Policy() for name in ("OpenVLA", "MiniVLA", "Octo-Small")}
            self.core = demo.DemoControllerCore(self.policies, self.world)
            self.requests = []

        def stream(self, *, payload, **kwargs):
            self.requests.append(payload)
            typed = demo.service_dispatch_to_stream_request(payload["request"])

            async def collect():
                iterator = self.core.rollout_events(typed.rollout) if typed.rollout else self.core.manual_events(typed.manual)
                return [record.encode() async for record in iterator]

            yield from asyncio.run(collect())

    root = tmp_path / "data"
    write_manifest(root)
    transport = ControllerTransport()
    service = ComparisonService(root, transport=transport, test_mode=True)
    try:
        quote = service.create_quote()
        created, _ = service.create_comparison(quote["id"], "controller-integration")
        detail = wait_for(service, created["id"])
        assert detail["status"] == "completed", detail["cells"][0]
        assert len(transport.world.calls) == 12 * 70
        assert all(len(worker.calls) == 4 * 70 for worker in transport.policies.values())
        assert all(cell["frame_count"] == cell["action_count"] == 70 for cell in detail["cells"])
        assert all(cell["terminal_received"] for cell in detail["cells"])
        branch = service.create_manual_session(created["id"], detail["cells"][0]["id"])
        command, _ = service.create_manual_command(branch["id"], "right", "native-manual")
        deadline = time.time() + 5
        while time.time() < deadline:
            result = service.get_manual_session(branch["id"])
            if result["commands"][0]["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert result["commands"][0]["status"] == "completed", result["commands"][0]
        assert result["commands"][0]["frame_count"] == 16
        assert len(transport.world.calls) == 12 * 70 + 1
        assert all(len(worker.calls) == 4 * 70 for worker in transport.policies.values())
        assert result["source"]["event_id"] == result["commands"][0]["frames"][-1]["event_id"]
        service.promote(created["id"])
    finally:
        service.shutdown()


def test_cli_transport_yields_small_receipt_before_process_closes(monkeypatch):
    """A 64 KiB buffered read would make this first next() take 0.6 seconds."""
    import plumb.comparisons as comparisons

    real_popen = subprocess.Popen
    observed = {}
    script = (
        "import sys,time; sys.stdin.buffer.read(); "
        "sys.stdout.buffer.write(b'first-receipt\\n'); sys.stdout.buffer.flush(); "
        "time.sleep(.6); sys.stdout.buffer.write(b'terminal\\n'); sys.stdout.buffer.flush()"
    )

    def popen(_argv, **kwargs):
        observed["argv"] = list(_argv)
        observed["env"] = dict(kwargs["env"])
        return real_popen([sys.executable, "-u", "-c", script], **kwargs)

    monkeypatch.setattr(comparisons.subprocess, "Popen", popen)
    monkeypatch.setenv("BASETEN_TRUSS_AUTH_API_KEY", "stale-token-must-not-pass")
    monkeypatch.setenv("BASETEN_TRUSS_AUTH_REMOTE_URL", "https://wrong.invalid")
    transport = BasetenCliStreamTransport()
    iterator = transport.stream(
        run_remote_url="https://chain-verified.api.baseten.co/production/run_remote",
        cli_profile="plumb-api",
        payload={"bounded": True},
        timeout_seconds=2,
    )
    started = time.monotonic()
    assert next(iterator) == b"first-receipt\n"
    assert time.monotonic() - started < 0.4
    assert b"".join(iterator) == b"terminal\n"
    assert observed["argv"][:2] == ["baseten", "truss"]
    assert observed["env"]["BASETEN_PROFILE"] == "plumb-api"
    assert "BASETEN_TRUSS_AUTH_API_KEY" not in observed["env"]
    assert "BASETEN_TRUSS_AUTH_REMOTE_URL" not in observed["env"]
