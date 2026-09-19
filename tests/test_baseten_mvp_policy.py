"""CPU protocol tests for the isolated Baseten MVP worker."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "deploy" / "baseten" / "mvp_policy" / "model" / "model.py"


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SHIM = _load(MODEL_PATH, "plumb_mvp_shim")


class FakeStdout:
    def __init__(self, lines):
        self.lines = list(lines)

    def readline(self, size):
        return self.lines.pop(0) if self.lines else ""

    def fileno(self):
        return 77


class FakeStdin:
    def __init__(self):
        self.writes = []

    def write(self, value):
        self.writes.append(value)

    def flush(self):
        pass


class FakeProcess:
    def __init__(self, messages):
        self.stdout = FakeStdout(messages)
        self.chunks = [message.encode("utf-8") for message in messages]
        self.stdin = FakeStdin()
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True

    def wait(self, timeout):
        return 0


def _request(**changes):
    value = {"schema_version": 1, "request_id": "mvp-test-1", "current_png_base64": "a" * 16, "goal_png_base64": "b" * 16, "prompt": "static diagnostic"}
    value.update(changes)
    return value


def test_shim_is_stdlib_only_and_strips_parent_pythonpath_for_worker(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unsafe/server/site-packages")
    environment = SHIM._worker_environment()
    assert "PYTHONPATH" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PLUMB_SOAR_SOURCE_ROOT"] == SHIM.SOURCE_ROOT
    assert "/usr/local/nvidia/lib64" in environment["LD_LIBRARY_PATH"]
    source = MODEL_PATH.read_text(encoding="utf-8")
    for forbidden in ("import tensorflow", "import jax", "import flax", "import numpy", "from PIL"):
        assert forbidden not in source


def test_shim_validates_exact_bounded_request_before_child_write():
    with pytest.raises(SHIM.WorkerProtocolError, match="exactly"):
        SHIM._validate_request(_request(extra="bad"))
    with pytest.raises(SHIM.WorkerProtocolError, match="byte bound"):
        SHIM._validate_request(_request(prompt="p" * (SHIM.MAX_REQUEST_BYTES + 1)))


def test_shim_handshake_forwards_one_request_and_returns_json_safe_unqualified_result(monkeypatch):
    response = {"schema_version": 1, "request_id": "mvp-test-1", "status": "completed_unqualified", "qualified": False, "actions": {"native_model_normalized": [0.0] * 7, "transformed_physical": [[0.0] * 7], "shape": [1, 7], "finite": True}}
    process = FakeProcess([json.dumps({"protocol": SHIM.PROTOCOL, "type": "ready", "runtime": {"protobuf": "4.25.8"}}) + "\n", json.dumps({"protocol": SHIM.PROTOCOL, "type": "result", "request_id": "mvp-test-1", "response": response}) + "\n"])
    monkeypatch.setattr(SHIM.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(SHIM.select, "select", lambda *args, **kwargs: ([process.stdout], [], []))
    monkeypatch.setattr(SHIM.os, "read", lambda fd, size: process.chunks.pop(0) if process.chunks else b"")
    model = SHIM.Model()
    model.load()
    observed = model.predict(_request())
    assert json.loads(json.dumps(observed))["qualified"] is False
    assert json.loads(process.stdin.writes[0].decode("utf-8"))["request_id"] == "mvp-test-1"
    assert observed["serving"]["isolated_ml_venv"] is True


def test_shim_kills_worker_on_timeout_without_replay(monkeypatch):
    process = FakeProcess([json.dumps({"protocol": SHIM.PROTOCOL, "type": "ready", "runtime": {}}) + "\n"])
    monkeypatch.setattr(SHIM.subprocess, "Popen", lambda *args, **kwargs: process)
    model = SHIM.Model()
    monkeypatch.setattr(SHIM.select, "select", lambda *args, **kwargs: ([process.stdout], [], []))
    monkeypatch.setattr(SHIM.os, "read", lambda fd, size: process.chunks.pop(0) if process.chunks else b"")
    model.load()
    monkeypatch.setattr(SHIM.select, "select", lambda *args, **kwargs: ([], [], []))
    with pytest.raises(TimeoutError):
        model.predict(_request())
    assert process.killed is True and model._process is None


def test_shim_partial_protocol_line_is_deadline_bounded_and_contract_error_keeps_worker(monkeypatch):
    error = json.dumps({"protocol": SHIM.PROTOCOL, "type": "error", "request_id": "mvp-test-1", "message": "bad PNG"}) + "\n"
    process = FakeProcess([json.dumps({"protocol": SHIM.PROTOCOL, "type": "ready", "runtime": {}}) + "\n", error])
    process.chunks = [process.chunks[0][:10], process.chunks[0][10:], process.chunks[1]]
    monkeypatch.setattr(SHIM.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(SHIM.select, "select", lambda *args, **kwargs: ([process.stdout], [], []))
    monkeypatch.setattr(SHIM.os, "read", lambda fd, size: process.chunks.pop(0) if process.chunks else b"")
    model = SHIM.Model()
    model.load()
    with pytest.raises(SHIM.WorkerRequestError, match="bad PNG"):
        model.predict(_request())
    assert process.killed is False and model._process is process
    response = {"schema_version": 1, "request_id": "mvp-test-1", "status": "completed_unqualified", "qualified": False}
    process.chunks.append((json.dumps({"protocol": SHIM.PROTOCOL, "type": "result", "request_id": "mvp-test-1", "response": response}) + "\n").encode("utf-8"))
    assert model.predict(_request())["status"] == "completed_unqualified"


def test_shim_kills_worker_on_wrong_response_identity(monkeypatch):
    response = {"schema_version": 1, "request_id": "other", "status": "completed_unqualified", "qualified": False}
    process = FakeProcess([json.dumps({"protocol": SHIM.PROTOCOL, "type": "ready", "runtime": {}}) + "\n", json.dumps({"protocol": SHIM.PROTOCOL, "type": "result", "request_id": "other", "response": response}) + "\n"])
    monkeypatch.setattr(SHIM.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(SHIM.select, "select", lambda *args, **kwargs: ([process.stdout], [], []))
    monkeypatch.setattr(SHIM.os, "read", lambda fd, size: process.chunks.pop(0) if process.chunks else b"")
    model = SHIM.Model()
    model.load()
    with pytest.raises(SHIM.WorkerProtocolError, match="identity"):
        model.predict(_request())
    assert process.killed is True
