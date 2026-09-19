#!/usr/bin/env python3
"""Write an immutable runtime lock for the separate Octo 241fb environment.

Run this only with ``venv-octo-autoeval241fb/bin/python`` and
``PYTHONPATH=/scratch/.../plumb/source-octo-autoeval241fb`` after root has
created the copied environment, installed Octo at the reviewed source, and
installed dlimp from its pinned VCS snapshot.  This helper makes no network
request and refuses to mutate existing locks, checkouts, or environment files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


PRODUCTION_SOURCE_COMMIT = "241fb3514b7c40957a86d869fecb7c7fc353f540"
DLIMP_SOURCE_COMMIT = "5edaa4691567873d495633f2708982b42edf1972"
RUNTIME_SCHEMA = "plumb-octo-runtime-lock-v1"
RUNTIME_PROFILE_ID = "octo-autoeval241fb-v1"
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
    "dlimp",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def _root(value: Path) -> Path:
    root = value.resolve()
    if not str(root).startswith(("/scratch/", "/global/scratch/")) or root.name != "plumb":
        raise ValueError("production Octo runtime locks are permitted only in a cluster scratch/.../plumb directory")
    return root


def _under(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("%s must remain under %s: %s" % (label, root, resolved)) from error
    return resolved


def _packages(names: Iterable[str]) -> Dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def _direct_url(distribution_name: str) -> Mapping[str, Any]:
    distribution = importlib.metadata.distribution(distribution_name)
    raw = distribution.read_text("direct_url.json")
    value = json.loads(raw) if raw is not None else None
    if not isinstance(value, Mapping):
        raise RuntimeError("%s must expose a direct_url.json VCS binding" % distribution_name)
    return value


def _clean_source(root: Path) -> Dict[str, str]:
    source = root / "source-octo-autoeval241fb"
    if not (source / ".git").is_dir():
        raise RuntimeError("missing production Octo checkout: %s" % source)
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != PRODUCTION_SOURCE_COMMIT or dirty:
        raise RuntimeError("production Octo checkout must be clean at %s" % PRODUCTION_SOURCE_COMMIT)
    import octo.model.octo_model as octo_model  # type: ignore

    module = Path(inspect.getfile(octo_model)).resolve()
    try:
        module.relative_to(source)
    except ValueError as error:
        raise RuntimeError("imported Octo module does not come from source-octo-autoeval241fb") from error
    requirements = source / "requirements.txt"
    if not requirements.is_file():
        raise RuntimeError("production Octo checkout lacks requirements.txt")
    return {
        "source_checkout": str(source),
        "source_revision": head,
        "octo_model_module": str(module),
        "octo_model_sha256": _sha256(module),
        "requirements_sha256": _sha256(requirements),
    }


def _live_freeze_lines() -> list[str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not read the live pip freeze from the 241fb environment") from error
    return result.stdout.splitlines()


def _checked_freeze(root: Path, value: Path, live_lines: Sequence[str]) -> Dict[str, Any]:
    venv = root / "venv-octo-autoeval241fb"
    freeze = _under(value, venv, "pip freeze")
    if freeze.is_file():
        lines = freeze.read_text(encoding="utf-8").splitlines()
    else:
        encoded = ("\n".join(live_lines) + "\n").encode("utf-8")
        try:
            descriptor = os.open(freeze, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            lines = freeze.read_text(encoding="utf-8").splitlines()
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            lines = list(live_lines)
    if lines != list(live_lines):
        raise RuntimeError("immutable pip freeze differs from the live 241fb environment")
    # The imported Octo module is bound separately to the clean checkout and
    # file hash above.  A copied venv may retain editable-package metadata that
    # is not the active PYTHONPATH source, so its freeze cannot be used as the
    # code-identity authority.
    if not any("dlimp" in line.lower() and DLIMP_SOURCE_COMMIT in line for line in lines):
        raise RuntimeError("pip freeze must bind dlimp at %s" % DLIMP_SOURCE_COMMIT)
    return {"path": str(freeze), "sha256": _sha256(freeze), "lines": lines}


_PIP_CHECK_DEVIATION = re.compile(
    r"^dlimp(?:\s+\S+)?\s+(?:has requirement|requires)\s+(?P<package>tensorflow|numpy)==(?P<required>[^, ]+),?\s+but you have\s+(?P=package)\s+(?P<actual>\S+?)(?:\s+which is incompatible)?\.?$",
    re.IGNORECASE,
)


def _accepted_pip_check_line(line: str, packages: Mapping[str, str]) -> Dict[str, str] | None:
    match = _PIP_CHECK_DEVIATION.fullmatch(line.strip())
    if match is None:
        return None
    package = match.group("package").lower()
    required = match.group("required")
    actual = match.group("actual")
    expected = {
        "tensorflow": ("2.15.0", "2.15.1"),
        "numpy": ("1.24.3", "1.26.4"),
    }.get(package)
    installed = packages.get(package)
    if (
        expected is None
        or required != expected[0]
        or actual != installed
        or _base_version(actual) != expected[1]
    ):
        return None
    return {"package": package, "source_requirement": required, "actual": actual, "raw_line": line}


def _checked_pip_check(packages: Mapping[str, str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"], check=False, capture_output=True, text=True
        )
    except OSError as error:
        raise RuntimeError("could not run pip check in the 241fb environment") from error
    stdout = result.stdout if isinstance(result.stdout, str) else str(result.stdout)
    stderr = result.stderr if isinstance(result.stderr, str) else str(result.stderr)
    report: Dict[str, Any] = {
        "command": [sys.executable, "-m", "pip", "check"],
        "returncode": int(result.returncode),
        "stdout": stdout,
        "stderr": stderr,
        "accepted_deviations": [],
    }
    if result.returncode == 0:
        return report
    lines = [line.strip() for stream in (stdout, stderr) for line in stream.splitlines() if line.strip()]
    accepted = []
    for line in lines:
        record = _accepted_pip_check_line(line, packages)
        if record is None:
            raise RuntimeError("pip check reported an unreviewed dependency break: %s" % line)
        accepted.append(record)
    names = [record["package"] for record in accepted]
    if not accepted or len(names) != len(set(names)):
        raise RuntimeError("pip check output must contain each reviewed deviation at most once.")
    report["accepted_deviations"] = accepted
    return report


def _asset_manifest(root: Path) -> Dict[str, str]:
    path = root / "models" / "rail-berkeley--octo-small" / "PLUMB-ASSET-MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "plumb-octo-assets-v1"
        or payload.get("model_id") != "rail-berkeley/octo-small"
        or payload.get("checkpoint_revision") != "03d88976c54a58e10480d2043a8c762b35bc2611"
    ):
        raise RuntimeError("Small Octo asset manifest does not bind the reviewed v1.0 checkpoint")
    return {"path": str(path), "sha256": _sha256(path), "staging_code_revision": str(payload.get("code_revision"))}


def _deviations(packages: Mapping[str, str]) -> list[Dict[str, str]]:
    values = {
        "numpy": ("1.24.3", "1.26.4", "Copied Alliance runtime uses NumPy 1.26.4; production source fixture/Gate-A/B revalidation remains required."),
        "tensorflow": ("2.15.0", "2.15.1", "Copied Alliance runtime uses TensorFlow 2.15.1; production source fixture/Gate-A/B revalidation remains required."),
    }
    records: list[Dict[str, str]] = []
    for name, (source_requirement, actual_base, reason) in values.items():
        actual = packages.get(name, "")
        if _base_version(actual) != actual_base:
            raise RuntimeError("%s must have reviewed base version %s, got %r" % (name, actual_base, actual))
        records.append(
            {
                "package": name,
                "source_requirement": source_requirement,
                "actual": actual,
                "actual_base_version": actual_base,
                "reason": reason,
            }
        )
    return records


def _write_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/scratch/lchow432/plumb"))
    parser.add_argument("--output", type=Path, required=True, help="New evidence/octo-runtime-autoeval241fb-v1.json path; overwrite is refused.")
    parser.add_argument(
        "--freeze-path",
        type=Path,
        required=True,
        help="Immutable pip freeze under venv-octo-autoeval241fb; created exclusively from the live inventory if absent.",
    )
    args = parser.parse_args(argv)
    try:
        root = _root(args.root)
        venv = root / "venv-octo-autoeval241fb"
        expected_python = venv / "bin" / "python"
        if Path(sys.executable).resolve() != expected_python.resolve():
            raise RuntimeError("run this helper with %s" % expected_python)
        output = _under(args.output, root / "evidence", "runtime lock output")
        if output.parent != root / "evidence" or output.exists():
            raise FileExistsError("refusing to overwrite or nest a production runtime lock: %s" % output)
        source = _clean_source(root)
        packages = _packages(PACKAGES)
        for name, expected in (("jax", "0.4.20"), ("flax", "0.7.5"), ("tensorflow-probability", "0.23.0")):
            if _base_version(packages[name]) != expected:
                raise RuntimeError("%s must retain source-compatible base %s, got %s" % (name, expected, packages[name]))
        pip_check = _checked_pip_check(packages)
        freeze = _checked_freeze(root, args.freeze_path, _live_freeze_lines())
        direct_url = _direct_url("dlimp")
        vcs = direct_url.get("vcs_info") if isinstance(direct_url.get("vcs_info"), Mapping) else {}
        if vcs.get("commit_id") != DLIMP_SOURCE_COMMIT:
            raise RuntimeError("dlimp direct URL must bind VCS commit %s" % DLIMP_SOURCE_COMMIT)
        payload: Dict[str, Any] = {
            "schema": RUNTIME_SCHEMA,
            "runtime_profile": RUNTIME_PROFILE_ID,
            **source,
            "python_executable": str(expected_python),
            "python_version": sys.version,
            "packages": packages,
            "pip_check": pip_check,
            "pip_freeze": freeze,
            "dlimp": {
                "distribution": "dlimp",
                "source_revision": DLIMP_SOURCE_COMMIT,
                "direct_url": dict(direct_url),
                "direct_url_sha256": _canonical_json_sha256(direct_url),
            },
            "source_requirement_deviations": _deviations(packages),
            "asset_manifest": _asset_manifest(root),
            "cuda_profile": "Alliance JAX CUDA12 runtime copied into a separate 241fb environment; no old venv mutation.",
        }
        _write_no_clobber(output, payload)
        print(json.dumps({"path": str(output), "sha256": _sha256(output), "runtime_profile": RUNTIME_PROFILE_ID}, sort_keys=True))
    except Exception as error:
        print("production Octo runtime lock failed: %s" % error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
