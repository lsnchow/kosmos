"""Fixed-venv JSON-lines worker; all ML imports are isolated here."""

import contextlib
import json
import subprocess
import sys

from policy_runtime import ContractError, PolicyRuntime, RuntimeLoadError


PROTOCOL = "plumb_mvp_ml_worker_v1"
MAX_LINE_BYTES = 12 * 1024 * 1024


def _emit(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def _gpu_metadata():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def main():
    runtime = PolicyRuntime()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime.load()
        _emit({"protocol": PROTOCOL, "type": "ready", "runtime": runtime._runtime_payload | {"gpu": _gpu_metadata()}})
    except RuntimeLoadError as error:
        _emit({"protocol": PROTOCOL, "type": "error", "request_id": None, "message": "startup failed: %s" % str(error)[:512], "diagnostic": error.diagnostic})
        return 2
    except Exception as error:
        _emit({"protocol": PROTOCOL, "type": "error", "request_id": None, "message": "startup failed: %s" % str(error)[:512]})
        return 2
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
        if not line:
            return 0
        request_id = None
        try:
            if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
                raise ContractError("worker request line is oversized or unterminated")
            request = json.loads(line)
            if isinstance(request, dict):
                request_id = request.get("request_id")
            with contextlib.redirect_stdout(sys.stderr):
                response = runtime.predict(request)
            _emit({"protocol": PROTOCOL, "type": "result", "request_id": request_id, "response": response})
        except Exception as error:
            _emit({"protocol": PROTOCOL, "type": "error", "request_id": request_id, "message": str(error)[:512]})


if __name__ == "__main__":
    raise SystemExit(main())
