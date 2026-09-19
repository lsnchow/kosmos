"""Thin Truss shim: all GPU/framework code is isolated in one venv child."""

import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path


PROTOCOL = "plumb_mvp_ml_worker_v1"
WORKER_PYTHON = "/opt/plumb-ml-venv/bin/python"
WORKER_SCRIPT = "/app/model/ml_worker.py"
SOURCE_ROOT = "/app/source-soar-gcbc"
STARTUP_TIMEOUT_SECONDS = 300.0
INFERENCE_TIMEOUT_SECONDS = 120.0
MAX_REQUEST_BYTES = 12 * 1024 * 1024
REQUIRED_REQUEST_KEYS = {"schema_version", "request_id", "current_png_base64", "goal_png_base64", "prompt"}


class WorkerProtocolError(RuntimeError):
    pass


class WorkerRequestError(WorkerProtocolError):
    """A validated client request was rejected; retain a healthy worker."""


def _worker_environment():
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PLUMB_SOAR_SOURCE_ROOT"] = SOURCE_ROOT
    venv = Path(WORKER_PYTHON).parents[1]
    site_roots = sorted(venv.glob("lib/python*/site-packages"))
    cuda_libs, cuda_bins = [], []
    for site_root in site_roots:
        cuda_libs.extend(sorted(str(path) for path in site_root.glob("nvidia/*/lib") if path.is_dir()))
        cuda_bins.extend(sorted(str(path) for path in site_root.glob("nvidia/*/bin") if path.is_dir()))
    driver_paths = [path for path in ("/usr/local/nvidia/lib64", environment.get("LD_LIBRARY_PATH", "")) if path]
    environment["LD_LIBRARY_PATH"] = ":".join(cuda_libs + driver_paths)
    environment["PATH"] = ":".join([str(venv / "bin")] + cuda_bins + [environment.get("PATH", "")])
    return environment


def _validate_request(request):
    if not isinstance(request, dict) or set(request) != REQUIRED_REQUEST_KEYS:
        raise WorkerProtocolError("request must contain exactly schema_version, request_id, current_png_base64, goal_png_base64, prompt")
    encoded = json.dumps(request, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_REQUEST_BYTES:
        raise WorkerProtocolError("request body exceeds the worker protocol byte bound")
    return encoded


class Model:
    def __init__(self, **kwargs):
        self._process = None
        self._lock = threading.Lock()
        self._ready = None

    def _kill_worker(self):
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _read_line(process, timeout):
        if process.stdout is None:
            raise WorkerProtocolError("worker stdout pipe is unavailable")
        buffer = getattr(process, "_plumb_stdout_buffer", bytearray())
        deadline = time.monotonic() + timeout
        while b"\n" not in buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("worker protocol timed out")
            readable, _, _ = select.select([process.stdout], [], [], remaining)
            if not readable:
                raise TimeoutError("worker protocol timed out")
            chunk = os.read(process.stdout.fileno(), min(65536, MAX_REQUEST_BYTES + 1 - len(buffer)))
            if not chunk:
                raise WorkerProtocolError("worker exited without a protocol response")
            buffer.extend(chunk)
            if len(buffer) > MAX_REQUEST_BYTES:
                raise WorkerProtocolError("worker emitted an oversized protocol line")
        line, _, remainder = buffer.partition(b"\n")
        process._plumb_stdout_buffer = bytearray(remainder)
        if len(line) > MAX_REQUEST_BYTES:
            raise WorkerProtocolError("worker emitted an oversized or unterminated protocol line")
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise WorkerProtocolError("worker stdout is not valid JSON protocol") from error
        if not isinstance(message, dict) or message.get("protocol") != PROTOCOL:
            raise WorkerProtocolError("worker emitted an unknown protocol message")
        return message

    def load(self):
        if self._process is not None:
            return
        process = subprocess.Popen([WORKER_PYTHON, WORKER_SCRIPT], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=False, bufsize=0, env=_worker_environment())
        self._process = process
        try:
            message = self._read_line(process, STARTUP_TIMEOUT_SECONDS)
            if message.get("type") != "ready" or not isinstance(message.get("runtime"), dict):
                if isinstance(message.get("diagnostic"), dict):
                    print("PLUMB_MVP_WORKER_DIAGNOSTIC " + json.dumps(message["diagnostic"], sort_keys=True), file=sys.stderr, flush=True)
                raise WorkerProtocolError("worker failed startup handshake")
            self._ready = message["runtime"]
        except Exception:
            self._kill_worker()
            raise

    def predict(self, request):
        encoded = _validate_request(request)
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None or process.stdin is None:
                self._kill_worker()
                raise WorkerProtocolError("worker is unavailable; requests are never replayed automatically")
            try:
                process.stdin.write(encoded + b"\n")
                process.stdin.flush()
                message = self._read_line(process, INFERENCE_TIMEOUT_SECONDS)
                if message.get("type") == "error":
                    if message.get("request_id") != request["request_id"]:
                        raise WorkerProtocolError("worker error request identity mismatch")
                    raise WorkerRequestError("worker rejected request: %s" % message.get("message", "unknown"))
                if message.get("type") != "result" or message.get("request_id") != request["request_id"]:
                    raise WorkerProtocolError("worker response request identity/protocol mismatch")
                response = message.get("response")
                if not isinstance(response, dict) or response.get("schema_version") != 1 or response.get("request_id") != request["request_id"] or response.get("status") != "completed_unqualified" or response.get("qualified") is not False:
                    raise WorkerProtocolError("worker response does not satisfy the unqualified diagnostic contract")
                json.dumps(response, allow_nan=False)
                response["serving"] = {"worker_protocol": PROTOCOL, "isolated_ml_venv": True, "worker_ready": self._ready}
                return response
            except WorkerRequestError:
                raise
            except Exception:
                self._kill_worker()
                raise

    def __del__(self):
        self._kill_worker()
