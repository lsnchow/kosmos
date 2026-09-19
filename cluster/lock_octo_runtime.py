#!/usr/bin/env python3
"""Create a no-clobber immutable runtime lock for the isolated Octo v1 env.

Run on a cluster login node after package installation.  It imports no model
and does not initialize JAX/TF devices; it records installed distribution
metadata, the verified staged asset manifest, the clean Octo source commit,
and the one externally sourced pure-Python TFP wheel hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


OCTO_CODE_REVISION = "37951e4e6d708fd76374f6e09e716763fe2673b1"
TFP_WHEEL_SHA256 = "dda5cacfe50cb19ecd96f3ce81e6ff8680d84213bcfe94ca0aaf6e5f51c88061"
PACKAGES = (
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "chex",
    "orbax-checkpoint",
    "tensorflow",
    "tensorflow-probability",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "numpy",
    "scipy",
    "tensorflow-io-gcs-filesystem",
    "imageio",
    "pillow",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(value: Path) -> Path:
    root = value.resolve()
    if not str(root).startswith(("/scratch/", "/global/scratch/")) or root.name != "plumb":
        raise ValueError("Octo runtime locks are permitted only in a cluster scratch/.../plumb directory")
    return root


def _distribution_versions(names: Iterable[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for name in names:
        values[name] = importlib.metadata.version(name)
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/scratch/lchow432/plumb"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-path", type=Path, required=True, help="Versioned, immutable pip freeze snapshot for this lock.")
    args = parser.parse_args(argv)
    try:
        root = _root(args.root)
        output = args.output.resolve()
        if output.parent != root / "evidence":
            raise ValueError("Octo runtime lock output must be directly inside %s" % (root / "evidence"))
        if output.exists():
            raise FileExistsError("Refusing to overwrite a runtime lock: %s" % output)
        source = root / "source-octo"
        head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != OCTO_CODE_REVISION or dirty:
            raise RuntimeError("Octo source checkout is not the clean reviewed commit")
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        assets = root / "models" / "rail-berkeley--octo-small" / "PLUMB-ASSET-MANIFEST.json"
        tfp_wheel = root / "venv-octo" / "wheels" / "tensorflow_probability-0.23.0-py2.py3-none-any.whl"
        freeze = args.freeze_path.resolve()
        freeze.relative_to(root / "venv-octo")
        if _sha256(tfp_wheel) != TFP_WHEEL_SHA256:
            raise RuntimeError("Pinned TensorFlow Probability wheel hash did not match")
        if not freeze.is_file():
            raise RuntimeError("Missing pip freeze snapshot for the isolated Octo runtime")
        payload: Dict[str, Any] = {
            "schema": "plumb-octo-runtime-lock-v1",
            "source_checkout": str(source),
            "source_revision": head,
            "asset_manifest": {"path": str(assets), "sha256": _sha256(assets)},
            "python_executable": sys.executable,
            "python_version": sys.version,
            "pip_freeze": {
                "path": str(freeze),
                "sha256": _sha256(freeze),
                "lines": freeze.read_text(encoding="utf-8").splitlines(),
            },
            "packages": _distribution_versions(PACKAGES),
            "cuda_profile": "Alliance jaxlib 0.4.20+cuda12.cudnn89.computecanada; load cuda/12.2 and cudnn/8.9.5.29",
            "tensorflow_compatibility_deviation": "2.15.1+computecanada replaces source lead 2.15.0; requires fixture/Gate-A/B revalidation.",
            "tfp": {
                "version": "0.23.0",
                "wheel": str(tfp_wheel),
                "sha256": TFP_WHEEL_SHA256,
                "source": "PyPI pure-Python wheel; no public CUDA artifact was installed.",
            },
        }
        encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"path": str(output), "sha256": _sha256(output)}, sort_keys=True))
    except Exception as error:
        print("Octo runtime lock failed: %s" % error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
