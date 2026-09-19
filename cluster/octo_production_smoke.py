#!/usr/bin/env python3
"""No-clobber GPU smoke for PLUMB's production Octo-Small v1.0 profile.

This entrypoint deliberately exercises ``plumb.policies.octo`` rather than
the separately named 37951 native-v0.1 diagnostic adapter.  The production
adapter is bound to Octo code 241fb and its ``timestep_pad_mask`` plus
``unnormalization_statistics=None`` sampler path.  It emits only raw,
unqualified fixture evidence: two static RGB inputs, three native calls, and
a reset/repeat check.  It is neither a rollout nor certification evidence.

The existing asset manifest retains its original 37951 *staging-code* identity.
The new 241fb runtime lock binds a separate copied environment, its exact
suffix-bearing package inventory, and a pinned dlimp VCS snapshot.  This
program separately verifies the clean 241fb checkout actually imported for
the production profile.  It never modifies a checkout, downloads anything, or
writes except a new no-clobber report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PRODUCTION_SOURCE_COMMIT = "241fb3514b7c40957a86d869fecb7c7fc353f540"
ASSET_STAGING_SOURCE_COMMIT = "37951e4e6d708fd76374f6e09e716763fe2673b1"
MODEL_ID = "rail-berkeley/octo-small"
MODEL_REVISION = "03d88976c54a58e10480d2043a8c762b35bc2611"
CHECKPOINT_STEP = 270000
CHECKPOINT_BYTES = 546696551
CHECKPOINT_SHA256 = "590df097f8a37bbc1c3aac2a488c0fb08e72bae8abaedfb89f0685677d848962"
EXAMPLE_BATCH_BYTES = 738368
EXAMPLE_BATCH_SHA256 = "0ce74dd8e433ce4a8a1534c4ab9687d9fc3e444b3eece750047dcb22125d73ff"
T5_MODEL_ID = "t5-base"
T5_REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
JAX_VERSION = "0.4.20"
FLAX_VERSION = "0.7.5"
NUMPY_SOURCE_VERSION = "1.24.3"
NUMPY_DEVIATION_BASE_VERSION = "1.26.4"
TFP_VERSION = "0.23.0"
TENSORFLOW_SOURCE_VERSION = "2.15.0"
TENSORFLOW_DEVIATION_BASE_VERSION = "2.15.1"
SOURCE_DLiMP_COMMIT = "5edaa4691567873d495633f2708982b42edf1972"
ASSET_SCHEMA = "plumb-octo-assets-v1"
RUNTIME_SCHEMA = "plumb-octo-runtime-lock-v1"
RUNTIME_PROFILE_ID = "octo-autoeval241fb-v1"
RUNTIME_PACKAGE_NAMES = (
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


class OctoProductionSmokeError(RuntimeError):
    """The requested production smoke cannot safely start."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _cluster_root(value: Path) -> Path:
    root = value.resolve()
    if not str(root).startswith(("/scratch/", "/global/scratch/")) or root.name != "plumb":
        raise OctoProductionSmokeError("Octo production smoke is permitted only in a cluster scratch/.../plumb root.")
    return root


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OctoProductionSmokeError("%s is not readable JSON: %s" % (label, path)) from error
    if not isinstance(value, Mapping):
        raise OctoProductionSmokeError("%s must be a JSON object: %s" % (label, path))
    return value


def _path_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise OctoProductionSmokeError("%s must remain under %s: %s" % (label, root, resolved)) from error
    return resolved


def _reserve_report(path: Path) -> Path:
    if path.exists():
        raise FileExistsError("Refusing to overwrite Octo production evidence: %s" % path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    reservation = path.with_name(".%s.reserved" % path.name)
    try:
        descriptor = os.open(reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError("Octo production report name is already reserved: %s" % path) from error
    os.close(descriptor)
    return reservation


def _publish_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError("Refusing to overwrite concurrently published Octo production evidence: %s" % path) from error
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _verify_release(release: Path) -> Dict[str, Any]:
    release = release.resolve()
    expected = release.name
    if not _is_sha256(expected):
        raise OctoProductionSmokeError("release directory name must be a 64-character source release SHA.")
    manifest = release / "RELEASE.json"
    if not manifest.is_file() or _sha256(manifest) != expected:
        raise OctoProductionSmokeError("RELEASE.json must hash to its immutable release directory name.")
    payload = _read_json(manifest, "release manifest")
    records = payload.get("files")
    if not isinstance(records, list):
        raise OctoProductionSmokeError("release manifest must list immutable source files.")
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise OctoProductionSmokeError("release manifest has an invalid file record.")
        path = _path_under(release / record["path"], release, "release file")
        if not path.is_file() or path.stat().st_size != record.get("bytes") or _sha256(path) != record.get("sha256"):
            raise OctoProductionSmokeError("release file did not match its manifest: %s" % record["path"])
    return {"path": str(release), "release_sha256": expected, "manifest_sha256": _sha256(manifest)}


def _verify_checkout(root: Path, checkout: Path) -> Dict[str, str]:
    checkout = _path_under(checkout, root, "production Octo checkout")
    expected_path = root / "source-octo-autoeval241fb"
    if checkout != expected_path.resolve():
        raise OctoProductionSmokeError("production Octo checkout must be exactly %s" % expected_path)
    if (
        not (checkout / ".git").is_dir()
        or not (checkout / "octo" / "model" / "octo_model.py").is_file()
        or not (checkout / "requirements.txt").is_file()
    ):
        raise OctoProductionSmokeError("production Octo checkout is incomplete: %s" % checkout)
    try:
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise OctoProductionSmokeError("could not inspect production Octo checkout") from error
    if head != PRODUCTION_SOURCE_COMMIT or dirty:
        raise OctoProductionSmokeError("production Octo checkout must be clean at %s" % PRODUCTION_SOURCE_COMMIT)
    source_file = checkout / "octo" / "model" / "octo_model.py"
    return {
        "path": str(checkout),
        "revision": head,
        "octo_model_sha256": _sha256(source_file),
        "requirements_sha256": _sha256(checkout / "requirements.txt"),
    }


def _manifest_records(root: Path, records: Any, label: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(records, list):
        raise OctoProductionSmokeError("%s manifest records must be a list." % label)
    result: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("relative_path"), str):
            raise OctoProductionSmokeError("%s manifest contains an invalid file record." % label)
        relative = record["relative_path"]
        candidate = _path_under(root / relative, root, "%s manifest file" % label)
        if not candidate.is_file() or candidate.stat().st_size != record.get("bytes") or _sha256(candidate) != record.get("sha256"):
            raise OctoProductionSmokeError("%s manifest mismatch for %s" % (label, relative))
        result[relative] = dict(record)
    return result


def _verify_assets(root: Path) -> Dict[str, Any]:
    model_root = root / "models" / "rail-berkeley--octo-small"
    manifest_path = model_root / "PLUMB-ASSET-MANIFEST.json"
    payload = _read_json(manifest_path, "Octo asset manifest")
    if payload.get("schema") != ASSET_SCHEMA or payload.get("model_id") != MODEL_ID or payload.get("checkpoint_revision") != MODEL_REVISION:
        raise OctoProductionSmokeError("asset manifest does not bind the required Octo-Small v1.0 snapshot.")
    # The model bytes were staged with the existing 37951 asset helper.  That
    # source identity is recorded, not re-labelled as 241fb: the code checkout
    # for this smoke is separately verified above.
    if payload.get("code_revision") != ASSET_STAGING_SOURCE_COMMIT:
        raise OctoProductionSmokeError("asset manifest must retain its original 37951 staging-code identity.")
    records = _manifest_records(model_root, payload.get("files"), "Octo asset")
    required = {
        "config.json": (None, None),
        "dataset_statistics.json": (None, None),
        "example_batch.msgpack": (EXAMPLE_BATCH_BYTES, EXAMPLE_BATCH_SHA256),
        "%d/commit_success.txt" % CHECKPOINT_STEP: (None, None),
        "%d/default/commit_success.txt" % CHECKPOINT_STEP: (None, None),
        "%d/default/checkpoint" % CHECKPOINT_STEP: (CHECKPOINT_BYTES, CHECKPOINT_SHA256),
    }
    for relative, (size, digest) in required.items():
        record = records.get(relative)
        if record is None:
            raise OctoProductionSmokeError("asset manifest lacks required file %s" % relative)
        if size is not None and (record.get("bytes") != size or record.get("sha256") != digest):
            raise OctoProductionSmokeError("asset manifest does not bind the reviewed %s" % relative)
    t5 = payload.get("t5")
    if not isinstance(t5, Mapping) or t5.get("model_id") != T5_MODEL_ID or t5.get("revision") != T5_REVISION:
        raise OctoProductionSmokeError("asset manifest does not bind the required offline t5-base tokenizer snapshot.")
    t5_root = root / "models" / ".hf-octo" / "hub" / "models--t5-base" / "snapshots" / T5_REVISION
    t5_records = _manifest_records(t5_root, t5.get("files"), "t5 asset")
    for relative in ("config.json", "spiece.model", "tokenizer.json"):
        if relative not in t5_records:
            raise OctoProductionSmokeError("t5 asset manifest lacks required tokenizer file %s" % relative)
    ref = root / "models" / ".hf-octo" / "hub" / "models--t5-base" / "refs" / "main"
    if not ref.is_file() or ref.read_text(encoding="utf-8") != T5_REVISION:
        raise OctoProductionSmokeError("offline t5-base cache ref must contain exactly the pinned revision.")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "staging_code_revision": ASSET_STAGING_SOURCE_COMMIT,
        "model_root": str(model_root),
        "checkpoint_path": str(model_root / ("%d/default/checkpoint" % CHECKPOINT_STEP)),
        "dataset_statistics_path": str(model_root / "dataset_statistics.json"),
        "t5_snapshot": str(t5_root),
    }


def _runtime_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise OctoProductionSmokeError("required runtime distribution is absent: %s" % name) from error


def _base_version(value: str) -> str:
    """Compare source requirements without erasing the recorded site suffix."""

    return value.split("+", 1)[0]


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _distribution_direct_url(name: str) -> Mapping[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
        raw = distribution.read_text("direct_url.json")
        value = json.loads(raw) if raw is not None else None
    except (importlib.metadata.PackageNotFoundError, ValueError) as error:
        raise OctoProductionSmokeError("%s must expose a valid direct_url.json VCS binding." % name) from error
    if not isinstance(value, Mapping):
        raise OctoProductionSmokeError("%s must expose a JSON-object direct_url VCS binding." % name)
    return value


def _required_deviation(
    deviations: Any,
    *,
    package: str,
    source_requirement: str,
    actual: str,
    expected_actual_base: str,
) -> Mapping[str, Any]:
    if not isinstance(deviations, list):
        raise OctoProductionSmokeError("runtime lock must list source requirement deviations explicitly.")
    matches = [item for item in deviations if isinstance(item, Mapping) and item.get("package") == package]
    if len(matches) != 1:
        raise OctoProductionSmokeError("runtime lock must record exactly one explicit %s deviation." % package)
    record = matches[0]
    if (
        record.get("source_requirement") != source_requirement
        or record.get("actual") != actual
        or record.get("actual_base_version") != expected_actual_base
        or not isinstance(record.get("reason"), str)
        or not record["reason"].strip()
    ):
        raise OctoProductionSmokeError("runtime lock %s deviation does not bind the source requirement and installed package exactly." % package)
    return record


def _verify_runtime_lock(root: Path, path: Path, checkout: Path) -> Tuple[Dict[str, Any], Dict[str, str]]:
    payload = _read_json(path, "Octo 241fb runtime lock")
    if payload.get("schema") != RUNTIME_SCHEMA or payload.get("source_revision") != PRODUCTION_SOURCE_COMMIT:
        raise OctoProductionSmokeError("runtime lock must bind the 241fb production source revision.")
    if payload.get("runtime_profile") != RUNTIME_PROFILE_ID:
        raise OctoProductionSmokeError("runtime lock must use profile %s." % RUNTIME_PROFILE_ID)
    if payload.get("source_checkout") != str(checkout.resolve()):
        raise OctoProductionSmokeError("runtime lock source_checkout does not bind source-octo-autoeval241fb.")
    venv_path = root / "venv-octo-autoeval241fb"
    if payload.get("python_executable") != str(venv_path / "bin" / "python"):
        raise OctoProductionSmokeError("runtime lock must bind the separate venv-octo-autoeval241fb interpreter.")
    expected = payload.get("packages")
    if not isinstance(expected, Mapping) or not expected:
        raise OctoProductionSmokeError("runtime lock must contain an exact package inventory.")
    if set(expected) != set(RUNTIME_PACKAGE_NAMES):
        raise OctoProductionSmokeError("runtime lock package inventory must contain exactly the reviewed 241fb runtime package set.")
    actual: Dict[str, str] = {}
    for name, value in expected.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise OctoProductionSmokeError("runtime package inventory is malformed.")
        installed = _runtime_version(name)
        if installed != value:
            raise OctoProductionSmokeError("runtime package %s changed from lock %s to %s" % (name, value, installed))
        actual[name] = installed
    if _base_version(actual.get("jax", "")) != JAX_VERSION:
        raise OctoProductionSmokeError("production Octo requires JAX %s." % JAX_VERSION)
    for name, expected_version in (("flax", FLAX_VERSION), ("tensorflow-probability", TFP_VERSION)):
        if _base_version(actual.get(name, "")) != expected_version:
            raise OctoProductionSmokeError("production Octo requires %s %s." % (name, expected_version))
    numpy = actual.get("numpy", "")
    tensorflow = actual.get("tensorflow", "")
    if _base_version(numpy) != NUMPY_DEVIATION_BASE_VERSION:
        raise OctoProductionSmokeError("241fb runtime must record its reviewed NumPy %s deviation, got %r." % (NUMPY_DEVIATION_BASE_VERSION, numpy))
    if _base_version(tensorflow) != TENSORFLOW_DEVIATION_BASE_VERSION:
        raise OctoProductionSmokeError("241fb runtime must record its reviewed TensorFlow %s deviation, got %r." % (TENSORFLOW_DEVIATION_BASE_VERSION, tensorflow))
    _required_deviation(
        payload.get("source_requirement_deviations"),
        package="numpy",
        source_requirement=NUMPY_SOURCE_VERSION,
        actual=numpy,
        expected_actual_base=NUMPY_DEVIATION_BASE_VERSION,
    )
    _required_deviation(
        payload.get("source_requirement_deviations"),
        package="tensorflow",
        source_requirement=TENSORFLOW_SOURCE_VERSION,
        actual=tensorflow,
        expected_actual_base=TENSORFLOW_DEVIATION_BASE_VERSION,
    )
    freeze = payload.get("pip_freeze")
    if not isinstance(freeze, Mapping) or not isinstance(freeze.get("path"), str) or not _is_sha256(freeze.get("sha256")) or not isinstance(freeze.get("lines"), list):
        raise OctoProductionSmokeError("runtime lock must bind an immutable pip freeze path, SHA-256, and lines.")
    freeze_path = _path_under(Path(freeze["path"]), venv_path, "runtime freeze")
    if not freeze_path.is_file() or _sha256(freeze_path) != freeze["sha256"] or freeze_path.read_text(encoding="utf-8").splitlines() != freeze["lines"]:
        raise OctoProductionSmokeError("runtime freeze does not match the locked 241fb environment.")
    try:
        live_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise OctoProductionSmokeError("could not read the live 241fb pip inventory.") from error
    if live_freeze != freeze["lines"]:
        raise OctoProductionSmokeError("live pip freeze differs from the immutable 241fb runtime lock.")
    dlimp = payload.get("dlimp")
    if not isinstance(dlimp, Mapping) or dlimp.get("distribution") != "dlimp" or dlimp.get("source_revision") != SOURCE_DLiMP_COMMIT:
        raise OctoProductionSmokeError("runtime lock must bind dlimp at %s." % SOURCE_DLiMP_COMMIT)
    locked_direct_url = dlimp.get("direct_url")
    if not isinstance(locked_direct_url, Mapping) or dlimp.get("direct_url_sha256") != _canonical_json_sha256(locked_direct_url):
        raise OctoProductionSmokeError("runtime lock dlimp direct_url binding is malformed.")
    actual_direct_url = _distribution_direct_url("dlimp")
    vcs = actual_direct_url.get("vcs_info") if isinstance(actual_direct_url.get("vcs_info"), Mapping) else {}
    if vcs.get("commit_id") != SOURCE_DLiMP_COMMIT or dict(actual_direct_url) != dict(locked_direct_url):
        raise OctoProductionSmokeError("installed dlimp does not match the exact locked VCS snapshot.")
    return dict(payload), actual


def _frames(frame_zero: Path, frame_one: Path, manifest: Path) -> Tuple[Any, Any, Mapping[str, Any]]:
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as error:
        raise OctoProductionSmokeError("NumPy and Pillow are required to load canonical RGB Octo PNGs.") from error
    verified = _read_json(manifest, "canonical Octo input manifest")
    if verified.get("schema") != "plumb-octo-canonical-inputs-v1" or verified.get("qualified") is not False:
        raise OctoProductionSmokeError("canonical Octo input manifest has an invalid qualification/schema claim.")
    records = verified.get("frames")
    if not isinstance(records, list) or len(records) != 2:
        raise OctoProductionSmokeError("canonical Octo input manifest must bind exactly two ordered frames.")
    expected = (manifest.parent / records[0].get("output_file", ""), manifest.parent / records[1].get("output_file", ""))
    if (frame_zero.resolve(), frame_one.resolve()) != tuple(path.resolve() for path in expected):
        raise OctoProductionSmokeError("canonical PNG arguments must exactly match the input manifest.")
    frames = []
    prior_index = -1
    for index, (path, record) in enumerate(zip((frame_zero, frame_one), records)):
        if not isinstance(record, Mapping):
            raise OctoProductionSmokeError("canonical input manifest has an invalid frame record.")
        source_index = record.get("source_frame_index")
        if not isinstance(source_index, int) or source_index <= prior_index:
            raise OctoProductionSmokeError("canonical Octo input frame ordering is invalid.")
        prior_index = source_index
        if record.get("output_file_sha256") != _sha256(path):
            raise OctoProductionSmokeError("canonical PNG file hash does not match its manifest: %s" % path)
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB").copy())
        if tuple(getattr(frame, "shape", ())) != (256, 256, 3) or str(getattr(frame, "dtype", "")) != "uint8":
            raise OctoProductionSmokeError("canonical PNG frame %d must decode to RGB uint8 (256, 256, 3)." % index)
        if record.get("output_rgb_pixel_sha256") != hashlib.sha256(frame.tobytes()).hexdigest():
            raise OctoProductionSmokeError("canonical PNG RGB pixel hash does not match its manifest: %s" % path)
        frames.append(frame)
    return frames[0], frames[1], verified


def _disable_tensorflow_gpu() -> str:
    try:
        import tensorflow as tf  # type: ignore

        tf.config.set_visible_devices([], "GPU")
    except ImportError as error:
        raise OctoProductionSmokeError("Octo source runtime requires TensorFlow for checkpoint I/O.") from error
    except RuntimeError as error:
        raise OctoProductionSmokeError("TensorFlow GPU visibility was initialized before Octo/JAX setup.") from error
    return str(getattr(tf, "__version__", "unknown"))


def _load_jax_after_tensorflow_setup() -> Any:
    """Import JAX only after TensorFlow has relinquished GPU visibility."""

    try:
        import jax  # type: ignore
    except ImportError as error:
        raise OctoProductionSmokeError("production Octo runtime requires JAX after TensorFlow GPU setup.") from error
    return jax


def _nvidia_smi_metadata() -> Dict[str, Any]:
    """Read physical GPU identity without inferring a CUDA-ordinal mapping."""

    unavailable: Dict[str, Any] = {
        "gpu_uuid": None,
        "gpu_model": None,
        "driver_version": None,
        "gpus": None,
        "returncode": None,
        "error": None,
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        unavailable["error"] = "%s: %s" % (type(error).__name__, error)
        return unavailable
    unavailable["returncode"] = int(result.returncode)
    if result.returncode != 0:
        unavailable["error"] = result.stderr.strip() or "nvidia-smi returned a nonzero status"
        return unavailable
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3 or not all(values):
            unavailable["error"] = "nvidia-smi returned an unparseable GPU identity row"
            return unavailable
        rows.append({"gpu_uuid": values[0], "gpu_model": values[1], "driver_version": values[2]})
    if not rows:
        unavailable["error"] = "nvidia-smi returned no GPU identity rows"
        return unavailable
    unavailable["gpus"] = rows
    if len(rows) == 1:
        unavailable.update(rows[0])
    return unavailable


def _gpu_metadata(jax_module: Any) -> Dict[str, Any]:
    devices = tuple(jax_module.devices())
    gpu_devices = tuple(device for device in devices if getattr(device, "platform", None) == "gpu")
    if not gpu_devices:
        raise OctoProductionSmokeError("production Octo smoke requires a JAX GPU device under Slurm allocation.")
    payload: Dict[str, Any] = {
        "devices": [str(device) for device in devices],
        "gpu_devices": [str(device) for device in gpu_devices],
        "gpu_device_kinds": [getattr(device, "device_kind", None) for device in gpu_devices],
        "gpu_peak_memory_bytes": None,
        "nvidia_smi": _nvidia_smi_metadata(),
    }
    getter = getattr(gpu_devices[0], "memory_stats", None)
    if callable(getter):
        try:
            stats = getter()
            if isinstance(stats, Mapping):
                payload["jax_memory_stats"] = dict(stats)
                if isinstance(stats.get("peak_bytes_in_use"), int):
                    payload["gpu_peak_memory_bytes"] = stats["peak_bytes_in_use"]
        except Exception:
            payload["jax_memory_stats"] = None
    return payload


def _gpu_metadata_for_adapter_runtime(runtime: Any, jax_module: Any) -> Dict[str, Any]:
    """Read GPU state from explicit JAX, never adapter-private runtime state.

    ``plumb.policies.octo._OctoRuntime`` intentionally exposes callbacks plus
    ``jax_version`` rather than a JAX module.  Keeping this boundary explicit
    prevents the smoke harness from widening the production adapter API.
    """

    expected = getattr(runtime, "jax_version", None)
    actual = str(getattr(jax_module, "__version__", ""))
    if not isinstance(expected, str) or not expected:
        raise OctoProductionSmokeError("production adapter runtime did not expose its JAX version.")
    if actual != expected:
        raise OctoProductionSmokeError("explicit JAX module %r does not match adapter runtime JAX %r." % (actual, expected))
    return _gpu_metadata(jax_module)


def _action_mask(dataset_statistics: Path) -> Tuple[Tuple[bool, ...], str]:
    payload = _read_json(dataset_statistics, "Octo dataset statistics")
    try:
        action = payload["bridge_dataset"]["action"]
        mean = action["mean"]
        std = action["std"]
    except (KeyError, TypeError) as error:
        raise OctoProductionSmokeError("dataset_statistics.json must expose bridge_dataset.action mean/std.") from error
    if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 7 or len(std) != 7:
        raise OctoProductionSmokeError("production Octo requires exactly seven action mean/std values.")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in mean + std):
        raise OctoProductionSmokeError("production Octo action mean/std values must be finite.")
    supplied = action.get("mask") if isinstance(action, Mapping) else None
    if supplied is None:
        # At 241fb, Octo's native normal unnormalizer defaults to
        # ones_like(mean).  PLUMB asks it for normalized rows and applies the
        # same explicit seven-channel normalizer at its own single boundary.
        return (True,) * 7, "241fb_sample_actions_default_ones_like_mean"
    if not isinstance(supplied, list) or len(supplied) != 7 or not all(isinstance(value, bool) for value in supplied):
        raise OctoProductionSmokeError("dataset_statistics action mask must be seven booleans when supplied.")
    return tuple(supplied), "dataset_statistics.action.mask"


def _verify_imported_production_source(checkout: Path) -> Dict[str, str]:
    try:
        import octo.model.octo_model as octo_model  # type: ignore
    except ImportError as error:
        raise OctoProductionSmokeError("production Octo source failed to import from the supplied checkout.") from error
    try:
        module_path = Path(inspect.getfile(octo_model)).resolve()
        module_path.relative_to(checkout.resolve())
    except (OSError, TypeError, ValueError) as error:
        raise OctoProductionSmokeError("imported octo.model.octo_model is not loaded from source-octo-autoeval241fb.") from error
    return {"module": str(module_path), "sha256": _sha256(module_path)}


def run(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise OctoProductionSmokeError("refusing production Octo inference outside an allocated Slurm job.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise OctoProductionSmokeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required.")
    root = _cluster_root(Path(args.root))
    hf_home = root / "models" / ".hf-octo"
    if os.environ.get("HF_HOME") != str(hf_home):
        raise OctoProductionSmokeError("HF_HOME must be the immutable Octo T5 cache: %s" % hf_home)
    release = _verify_release(_path_under(Path(args.release_dir), root, "immutable source release"))
    checkout = _verify_checkout(root, Path(args.production_source))
    assets = _verify_assets(root)
    runtime_lock_path = _path_under(Path(args.runtime_lock), root, "241fb runtime lock")
    runtime_lock, packages = _verify_runtime_lock(root, runtime_lock_path, Path(args.production_source))
    first_frame, second_frame, input_manifest = _frames(Path(args.frame_0), Path(args.frame_1), Path(args.input_manifest))
    tensorflow_version = _disable_tensorflow_gpu()
    jax_module = _load_jax_after_tensorflow_setup()

    from plumb.adapters.contracts import PolicyObservation
    from plumb.policies.octo import (
        OCTO_PRODUCTION_SOURCE_COMMIT,
        OctoEnsembleConfig,
        OctoPolicyProfile,
        OctoSmallV1PolicyAdapter,
        load_octo_action_statistics,
    )
    from plumb.policies.contracts import PolicyActionNormalizer, PolicyExecutionMode
    from plumb.policies.tasks import BENCHMARK_TASK_REGISTRY, TASK_REGISTRY_HASH

    if OCTO_PRODUCTION_SOURCE_COMMIT != PRODUCTION_SOURCE_COMMIT:
        raise OctoProductionSmokeError("release main adapter does not bind the reviewed 241fb production source.")
    mask, mask_source = _action_mask(Path(assets["dataset_statistics_path"]))
    statistics = load_octo_action_statistics(
        assets["dataset_statistics_path"],
        normalizer_revision="octo-small-v1-bridge-dataset-%s" % assets["manifest_sha256"],
        mask=mask,
    )
    profile = OctoPolicyProfile(
        profile_id=args.profile_id,
        local_model_path=assets["model_root"],
        model_id=MODEL_ID,
        checkpoint_revision=MODEL_REVISION,
        checkpoint_step=CHECKPOINT_STEP,
        source_revision=PRODUCTION_SOURCE_COMMIT,
        asset_manifest_id="octo-small-assets-staged-37951-data-provenance",
        asset_manifest_sha256=assets["manifest_sha256"],
        runtime_lock_id=RUNTIME_PROFILE_ID,
        runtime_lock_sha256=_sha256(runtime_lock_path),
    )
    adapter = OctoSmallV1PolicyAdapter(
        profile,
        normalizer=PolicyActionNormalizer(statistics),
        ensemble=OctoEnsembleConfig(
            config_revision="octo-v1-temporal-ensemble-241fb-autoeval-rng0-production-smoke-v2",
            exponential_weight=0.0,
            horizon=4,
        ),
        execution_mode=PolicyExecutionMode.FIXTURE_CERTIFICATION,
    )
    capability = adapter.capability()
    if capability.status.value != "blocked" or "no_certification_artifact" not in capability.details.get("certification_errors", []):
        raise OctoProductionSmokeError("production smoke must remain blocked without a golden action certification artifact.")
    runtime = adapter._load_runtime()
    imported_source = _verify_imported_production_source(Path(args.production_source))
    gpu_before = _gpu_metadata_for_adapter_runtime(runtime, jax_module)
    task = BENCHMARK_TASK_REGISTRY.get(args.task_id)
    # The canonical RGB manifest establishes frame order only.  It does not
    # establish physical robot-control or presentation timestamps, so no
    # numeric timestamp is invented for this diagnostic.
    first_observation = PolicyObservation(image_history=(first_frame, first_frame), prompt=task.instruction, timestamp=None)
    second_observation = PolicyObservation(image_history=(first_frame, second_frame), prompt=task.instruction, timestamp=None)
    adapter.reset(args.seed)
    first = adapter.propose(first_observation)
    second = adapter.propose(second_observation)
    adapter.reset(args.seed)
    repeated_first = adapter.propose(first_observation)
    gpu_after = _gpu_metadata_for_adapter_runtime(runtime, jax_module)
    reset_repeat = first.actions == repeated_first.actions and first.executed_actions == repeated_first.executed_actions
    if not reset_repeat:
        raise OctoProductionSmokeError("production Octo reset/repeat was not deterministic for the fixed source/seed/input.")
    return 0, {
        "schema_version": 1,
        "kind": "plumb_octo_small_production_241fb_two_observation_smoke",
        "qualified": False,
        "status": "completed_unqualified",
        "reason": (
            "Static canonical frames exercised PLUMB's production Octo 241fb sampler and reset path. This is not "
            "generated-image feedback, a rollout, Gate-A/B evidence, a certified execute prefix, or a study result."
        ),
        "source_release": release,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "task": {"task_id": task.task_id, "instruction": task.instruction, "registry_hash": TASK_REGISTRY_HASH},
        "production_source": checkout,
        "imported_production_source": imported_source,
        "assets": assets,
        "runtime_lock": {
            "path": str(runtime_lock_path),
            "sha256": _sha256(runtime_lock_path),
            "runtime_profile": runtime_lock["runtime_profile"],
            "source_revision": runtime_lock["source_revision"],
            "package_inventory": packages,
            "source_requirement_deviations": runtime_lock["source_requirement_deviations"],
            "dlimp": runtime_lock["dlimp"],
            "tensorflow_import_version": tensorflow_version,
            "limitation": "This lock binds a separate 241fb environment and explicit source-package deviations. A successful smoke remains unqualified and cannot transfer 37951 diagnostic qualification.",
        },
        "fixture": {
            "input_manifest": str(Path(args.input_manifest).resolve()),
            "input_manifest_sha256": _sha256(Path(args.input_manifest)),
            "frame_files": [str(Path(args.frame_0).resolve()), str(Path(args.frame_1).resolve())],
            "frame_file_sha256": [_sha256(Path(args.frame_0)), _sha256(Path(args.frame_1))],
            "source_frame_indices": [record["source_frame_index"] for record in input_manifest["frames"]],
            "preprocessing": input_manifest["preprocessing"],
            "timestamp_limitation": "Frame order is known; no physical robot-control or video-presentation timestamp is asserted.",
        },
        "profile": {
            "profile_id": profile.profile_id,
            "source_revision": profile.source_revision,
            "checkpoint_revision": profile.checkpoint_revision,
            "checkpoint_step": profile.checkpoint_step,
            "observation_keys_revision": profile.observation_keys.keys_revision,
            "sampler": "sample_actions(observations, tasks, unnormalization_statistics=None, rng=PRNGKey(0))",
            "rng_profile": "autoeval_static_prngkey_0_v2",
            "normalization": "PLUMB source-bound normalizer applies bridge_dataset mean/std once after the 241fb sampler returns normalized rows.",
            "mask_source": mask_source,
            "execution_mode": adapter.execution_mode.value,
            "certified_execute_prefix": None,
        },
        "capability": {"status": capability.status.value, "reason": capability.reason, "details": dict(capability.details)},
        "runtime": {
            "jax_version_full": runtime.jax_version,
            "jax_module_version_full": str(getattr(jax_module, "__version__", "")),
            "tensorflow_gpu_visibility": "disabled before JAX/Octo model imports",
            "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "before_native_calls": gpu_before,
            "after_native_calls": gpu_after,
        },
        "native_calls": {
            "first": {"actions": [list(row) for row in first.actions], "executed_actions": [list(row) for row in first.executed_actions], "metadata": dict(first.metadata), "wall_seconds": first.wall_seconds},
            "second": {"actions": [list(row) for row in second.actions], "executed_actions": [list(row) for row in second.executed_actions], "metadata": dict(second.metadata), "wall_seconds": second.wall_seconds},
            "after_reset_first": {"actions": [list(row) for row in repeated_first.actions], "executed_actions": [list(row) for row in repeated_first.executed_actions], "metadata": dict(repeated_first.metadata), "wall_seconds": repeated_first.wall_seconds},
            "backend_calls": 3,
            "reset_deterministic_repeat": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--release-dir", required=True, help="Immutable release directory whose RELEASE.json hashes to its directory name.")
    parser.add_argument("--production-source", required=True, help="Clean root/source-octo-autoeval241fb checkout; it is never modified.")
    parser.add_argument("--runtime-lock", required=True, help="New immutable 241fb runtime lock for root/venv-octo-autoeval241fb.")
    parser.add_argument("--frame-0", required=True)
    parser.add_argument("--frame-1", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--task-id", required=True, choices=("close_drawer", "open_drawer", "to_basket", "to_sink", "fold_cloth"))
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile-id", default="octo-small-v1-production-241fb-autoeval-rng0-v2-unqualified")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report_path = Path(args.report).resolve()
    try:
        reservation = _reserve_report(report_path)
    except Exception as error:
        print(json.dumps({"status": "failed_before_reservation", "error": {"type": type(error).__name__, "message": str(error)}}, sort_keys=True))
        return 2
    started = time.perf_counter()
    started_at = _utc_now()
    try:
        code, payload = run(args)
    except Exception as error:
        payload = {
            "schema_version": 1,
            "kind": "plumb_octo_small_production_241fb_two_observation_smoke",
            "qualified": False,
            "status": "failed",
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=10)},
        }
        code = 2
    payload.update(
        total_seconds=time.perf_counter() - started,
        started_at=started_at,
        finished_at=_utc_now(),
        producer_sha256=_sha256(Path(__file__)),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_step_id=os.environ.get("SLURM_STEP_ID"),
        slurm_process_id=os.environ.get("SLURM_PROCID"),
        slurm_node=os.environ.get("SLURMD_NODENAME"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        requested_profile_id=args.profile_id,
    )
    try:
        _publish_json_no_clobber(report_path, payload)
    except Exception as error:
        print(json.dumps({"status": "failed_to_publish", "error": {"type": type(error).__name__, "message": str(error)}}, sort_keys=True))
        return 2
    finally:
        try:
            reservation.unlink()
        except FileNotFoundError:
            pass
    print(json.dumps(payload, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
