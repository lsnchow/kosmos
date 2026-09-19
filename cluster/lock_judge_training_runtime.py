"""Record the isolated judge-training runtime; no model load or network calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

EXPECTED = {"torch": "2.6.0", "transformers": "4.49.0", "peft": "0.14.0", "trl": "0.14.0",
            "accelerate": "1.3.0", "datasets": "3.2.0", "numpy": "2.2.2", "pyarrow": "19.0.1"}


def inventory() -> dict:
    modules = {}
    for name, version in EXPECTED.items():
        module = importlib.import_module(name)
        actual = str(module.__version__)
        if actual.split("+", 1)[0] != version:
            raise ValueError("%s must import reviewed base %s, got %s" % (name, version, actual))
        modules[name] = {"import_version": actual, "distribution_version": importlib.metadata.version(name),
                         "module_file": str(Path(module.__file__).resolve())}
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=60)
    if check.returncode:
        raise ValueError("training runtime pip check failed: " + check.stdout + check.stderr)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], check=True, capture_output=True, text=True, timeout=60)
    return {"schema": "plumb-judge-training-runtime-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "qualified": False, "model_loaded": False, "training_performed": False,
            "python_executable": sys.executable, "python_version": sys.version, "packages": modules,
            "module_profile": "StdEnv/2023 python/3.11 arrow/19.0.1",
            "pip_check": {"returncode": check.returncode, "stdout": check.stdout, "stderr": check.stderr},
            "pip_freeze_all": freeze.stdout.splitlines(),
            "pip_freeze_sha256": hashlib.sha256(freeze.stdout.encode()).hexdigest(),
            "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": ["Alliance build variants are not equivalent to the Baseten training image.",
                            "Import/dependency checks are not an allocated-GPU model/processor/training check."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.name != "plumb" or not str(root).startswith(("/scratch/", "/global/scratch/")):
        raise ValueError("training runtime locks belong in the cluster scratch plumb root")
    expected = root / "venv-judge-lora-tf449" / "bin" / "python"
    if Path(sys.executable).absolute() != expected:
        raise ValueError("use the separate venv-judge-lora-tf449 interpreter")
    if args.output.absolute().parent != root / "evidence" or args.output.exists() or args.output.is_symlink():
        raise ValueError("output must be a NEW JSON file directly in cluster evidence")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("run import-only inventory with CUDA_VISIBLE_DEVICES empty")
    report = inventory()
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"path": str(args.output), "sha256": hashlib.sha256(encoded.encode()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
