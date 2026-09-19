#!/usr/bin/env python3
"""One allocated-GPU diagnostic for the exact local Octo-Small v1 path.

This is not a rollout, policy qualification, feedback-fidelity result, or
primary-study row.  It loads the immutable checkpoint once, samples two
distinct observed frames through the released two-frame/four-action wrapper,
then resets and repeats the first observation to check the pinned deterministic
wrapper behavior.  It must run under Slurm on an allocated GPU node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class OctoSmokeError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reserve_report(path: Path) -> Path:
    """Reserve one report name before a GPU/model call can consume evidence."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite Octo smoke evidence: %s" % path)
    reservation = path.parent / (".%s.reservation" % path.name)
    try:
        descriptor = os.open(reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError("Octo smoke report name is already reserved: %s" % path) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("reserved\n")
        stream.flush()
        os.fsync(stream.fileno())
    return reservation


def _publish_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a completed/failure report without replacing an existing file."""

    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError("Refusing to overwrite concurrently published Octo smoke evidence: %s" % path) from error
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    else:
        os.unlink(temporary_name)


def _frames(frame_zero: Path, frame_one: Path, manifest: Path) -> Tuple[Any, Any, Mapping[str, Any]]:
    if not manifest.is_file():
        raise FileNotFoundError("Expected canonical Octo input manifest: %s" % manifest)
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as error:
        raise OctoSmokeError("NumPy and Pillow are required to load canonical RGB Octo PNGs.") from error
    try:
        verified = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OctoSmokeError("Canonical Octo input manifest is not valid JSON.") from error
    if verified.get("schema") != "plumb-octo-canonical-inputs-v1" or verified.get("qualified") is not False:
        raise OctoSmokeError("Canonical Octo input manifest has an invalid qualification/schema claim.")
    records = verified.get("frames")
    if not isinstance(records, list) or len(records) != 2:
        raise OctoSmokeError("Canonical Octo input manifest must bind exactly two ordered frames.")
    expected = (manifest.parent / records[0]["output_file"], manifest.parent / records[1]["output_file"])
    if (frame_zero.resolve(), frame_one.resolve()) != tuple(path.resolve() for path in expected):
        raise OctoSmokeError("Canonical PNG arguments must exactly match the ordered paths bound by the input manifest.")
    frames = []
    previous_source_index = -1
    for index, (path, record) in enumerate(zip((frame_zero, frame_one), records)):
        source_index = record.get("source_frame_index") if isinstance(record, Mapping) else None
        if not isinstance(source_index, int) or source_index <= previous_source_index:
            raise OctoSmokeError("Canonical Octo input manifest frame ordering is invalid.")
        previous_source_index = source_index
        if record.get("output_file_sha256") != _sha256(path):
            raise OctoSmokeError("Canonical PNG file hash does not match the input manifest: %s" % path)
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB").copy())
        shape = tuple(getattr(frame, "shape", ()))
        dtype = str(getattr(frame, "dtype", ""))
        if shape != (256, 256, 3) or dtype != "uint8":
            raise OctoSmokeError(
                "Canonical PNG frame %d must decode to RGB uint8 (256, 256, 3); got shape %r dtype %s." % (index, shape, dtype)
            )
        if record.get("output_rgb_pixel_sha256") != hashlib.sha256(frame.tobytes()).hexdigest():
            raise OctoSmokeError("Canonical PNG RGB pixel hash does not match the input manifest: %s" % path)
        frames.append(frame)
    return frames[0], frames[1], verified


def _gpu_metadata(jax_module: Any) -> Dict[str, Any]:
    devices = tuple(jax_module.devices())
    gpu_devices = tuple(device for device in devices if getattr(device, "platform", None) == "gpu")
    if not gpu_devices:
        raise OctoSmokeError("Octo smoke requires a JAX GPU device under a Slurm allocation; none was visible.")
    details: Dict[str, Any] = {
        "devices": [str(device) for device in devices],
        "gpu_devices": [str(device) for device in gpu_devices],
        "gpu_device_kinds": [getattr(device, "device_kind", None) for device in gpu_devices],
        "gpu_peak_memory_bytes": None,
    }
    memory_stats = getattr(gpu_devices[0], "memory_stats", None)
    if callable(memory_stats):
        try:
            stats = memory_stats()
            if isinstance(stats, Mapping):
                details["jax_memory_stats"] = dict(stats)
                value = stats.get("peak_bytes_in_use")
                if isinstance(value, int):
                    details["gpu_peak_memory_bytes"] = value
        except Exception:
            details["jax_memory_stats"] = None
    return details


def _disable_tensorflow_gpu() -> str:
    """Octo uses TensorFlow only for checkpoint file I/O; keep JAX as GPU owner."""

    try:
        import tensorflow as tf  # type: ignore

        tf.config.set_visible_devices([], "GPU")
    except ImportError as error:
        raise OctoSmokeError("The Octo source runtime requires TensorFlow for local checkpoint I/O.") from error
    except RuntimeError as error:
        raise OctoSmokeError("TensorFlow GPU visibility was initialized before Octo smoke setup.") from error
    return str(getattr(tf, "__version__", "unknown"))


def run(args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    started = time.perf_counter()
    root = Path(args.root).resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        raise OctoSmokeError("Refusing Octo inference outside an allocated Slurm job.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise OctoSmokeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required.")
    hf_home = root / "models" / ".hf-octo"
    if os.environ.get("HF_HOME") != str(hf_home):
        raise OctoSmokeError("HF_HOME must be the immutable Octo T5 cache: %s" % hf_home)

    # This runs before importing plumb.policies.octo, whose lazy model import
    # imports TensorFlow.  It does not mask CUDA from JAX.
    tensorflow_version = _disable_tensorflow_gpu()
    from plumb.adapters.contracts import PolicyObservation
    from plumb.policies.octo import (
        OCTO_SMALL_MODEL_REVISION,
        OCTO_T5_MODEL_REVISION,
        OctoSmallV1Policy,
        OctoV1PolicyProfile,
    )
    from plumb.policies.tasks import BENCHMARK_TASK_REGISTRY, TASK_REGISTRY_HASH

    task = BENCHMARK_TASK_REGISTRY.get(args.task_id)
    model_root = root / "models" / "rail-berkeley--octo-small"
    t5_root = hf_home / "hub" / "models--t5-base" / "snapshots" / OCTO_T5_MODEL_REVISION
    profile = OctoV1PolicyProfile(
        profile_id=args.profile_id,
        local_model_path=str(model_root),
        checkpoint_revision=OCTO_SMALL_MODEL_REVISION,
        source_checkout_path=str(root / "source-octo"),
        asset_manifest_path=str(model_root / "PLUMB-ASSET-MANIFEST.json"),
        t5_tokenizer_path=str(t5_root),
        hf_home=str(hf_home),
        jax_version=args.jax_version,
        compatibility_profile_id=args.compatibility_profile_id,
        runtime_lock_id=args.runtime_lock_id,
        runtime_lock_sha256=args.runtime_lock_sha256,
    )
    adapter = OctoSmallV1Policy(profile)
    capability = adapter.capability()
    if capability.status.value != "ready_unqualified":
        raise OctoSmokeError("Octo profile is not ready: %s" % capability.reason)
    runtime = adapter._load_runtime()  # validates the profile's exact base JAX version; no model call yet.
    gpu_before_calls = _gpu_metadata(runtime.jax)
    first_frame, second_frame, input_manifest = _frames(Path(args.frame_0), Path(args.frame_1), Path(args.input_manifest))
    first = adapter.predict_with_report(
        PolicyObservation(image_history=(first_frame,), prompt=task.instruction, timestamp=None)
    )
    second = adapter.predict_with_report(
        PolicyObservation(image_history=(second_frame,), prompt=task.instruction, timestamp=None)
    )
    adapter.reset()
    repeated_first = adapter.predict_with_report(
        PolicyObservation(image_history=(first_frame,), prompt=task.instruction, timestamp=None)
    )
    gpu_after_calls = _gpu_metadata(runtime.jax)
    deterministic_reset_repeat = first.action == repeated_first.action and first.proposal == repeated_first.proposal
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "plumb_octo_small_native_two_observation_diagnostic",
        "qualified": False,
        "status": "completed_unqualified",
        "reason": (
            "Two static observed fixture frames exercise the source-native policy state and reset path only; "
            "they are not generated-image feedback, a task rollout, Gate A/B evidence, or a primary-study row."
        ),
        "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "task": {"task_id": task.task_id, "instruction": task.instruction, "registry_hash": TASK_REGISTRY_HASH},
        "fixture": {
            "input_manifest": str(Path(args.input_manifest).resolve()),
            "input_manifest_sha256": _sha256(Path(args.input_manifest)),
            "frame_files": [str(Path(args.frame_0).resolve()), str(Path(args.frame_1).resolve())],
            "frame_file_sha256": [_sha256(Path(args.frame_0)), _sha256(Path(args.frame_1))],
            "source_frame_indices": [record["source_frame_index"] for record in input_manifest["frames"]],
            "source_image_timestamps": None,
            "timestamp_limitation": "Frame order is known from the input manifest; no physical robot-control or video-presentation timestamps are asserted.",
            "preprocessing": input_manifest["preprocessing"],
        },
        "profile": {
            "profile_id": profile.profile_id,
            "checkpoint_revision": profile.checkpoint_revision,
            "jax_version_required": profile.jax_version,
            "compatibility_profile_id": profile.compatibility_profile_id,
            "runtime_lock_id": profile.runtime_lock_id,
            "runtime_lock_sha256": profile.runtime_lock_sha256,
            "asset_manifest": profile.asset_manifest_path,
            "source_checkout": profile.source_checkout_path,
            "t5_tokenizer_revision": OCTO_T5_MODEL_REVISION,
        },
        "capability": {"reason": capability.reason, "details": dict(capability.details), "evidence_uris": list(capability.evidence_uris)},
        "runtime": {
            "jax_version_full": runtime.jax_version,
            "tensorflow_version": tensorflow_version,
            "tensorflow_gpu_visibility": "disabled before Octo/JAX model imports",
            "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "before_native_calls": gpu_before_calls,
            "after_native_calls": gpu_after_calls,
        },
        "native_calls": {
            "first": {"action": list(first.action), "proposal": [list(row) for row in first.proposal], "wall_seconds": first.wall_seconds},
            "second": {"action": list(second.action), "proposal": [list(row) for row in second.proposal], "wall_seconds": second.wall_seconds},
            "after_reset_first": {
                "action": list(repeated_first.action),
                "proposal": [list(row) for row in repeated_first.proposal],
                "wall_seconds": repeated_first.wall_seconds,
            },
            "backend_calls": 3,
            "reset_deterministic_repeat": deterministic_reset_repeat,
            "gripper_output": first.gripper_transformation,
        },
    }
    payload["total_seconds"] = time.perf_counter() - started
    return 0, payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--frame-0", required=True, help="Canonical 256x256 uint8 PNG bound as source frame index 0.")
    parser.add_argument("--frame-1", required=True, help="Canonical 256x256 uint8 PNG bound as source frame index 1.")
    parser.add_argument("--input-manifest", required=True, help="No-clobber sidecar binding the two canonical PNGs to source frames/preprocessing.")
    parser.add_argument("--task-id", required=True, choices=("close_drawer", "open_drawer", "to_basket", "to_sink", "fold_cloth"))
    parser.add_argument("--report", required=True, help="New evidence JSON path; overwrites are refused.")
    parser.add_argument("--profile-id", default="octo-small-v1-source-v010-normal-alliance")
    parser.add_argument("--jax-version", default="0.4.20")
    parser.add_argument("--compatibility-profile-id")
    parser.add_argument("--runtime-lock-id")
    parser.add_argument("--runtime-lock-sha256")
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
            "kind": "plumb_octo_small_native_two_observation_diagnostic",
            "qualified": False,
            "status": "failed",
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=10)},
        }
        code = 2
    payload["total_seconds"] = time.perf_counter() - started
    payload["started_at"] = started_at
    payload["finished_at"] = _utc_now()
    payload["source_release"] = os.environ.get("PLUMB_SOURCE_RELEASE")
    payload["producer_sha256"] = _sha256(Path(__file__))
    payload["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    payload["slurm_step_id"] = os.environ.get("SLURM_STEP_ID")
    payload["slurm_process_id"] = os.environ.get("SLURM_PROCID")
    payload["slurm_node"] = os.environ.get("SLURMD_NODENAME")
    payload["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    payload["requested_profile_id"] = args.profile_id
    payload["requested_runtime_lock"] = {"id": args.runtime_lock_id, "sha256": args.runtime_lock_sha256}
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
