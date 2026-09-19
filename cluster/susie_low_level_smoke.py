#!/usr/bin/env python3
"""One allocated-GPU, static-fixture diagnostic for published SuSIE_LL gc_bc.

This invokes only the public ``patreya/gcbc-bridge`` low-level component. It
does not import SuSIE diffusion, Stable Diffusion, any gated component, a world
model, or a judge. Two hash-bound source-video frames are used solely as a
current/goal conditioning pair; the result is explicitly unqualified and is
not a task rollout, benchmark row, Gate result, or success label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


VENDOR_FIXTURE_SHA256 = "a86cfc81633b216891ca26dc58c72193a979c10ad72f123175fa8d61a67cdaec"
INPUT_SCHEMA = "plumb-octo-canonical-inputs-v1"


class SuSIELowLevelSmokeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pixel_sha256(array: Any) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reserve(path: Path) -> Path:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite SuSIE_LL evidence: %s" % path)
    reservation = path.parent / (".%s.reservation" % path.name)
    try:
        descriptor = os.open(reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError("SuSIE_LL evidence name is already reserved: %s" % path) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("reserved\n")
        stream.flush()
        os.fsync(stream.fileno())
    return reservation


def _publish(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError("Refusing to overwrite concurrently published evidence: %s" % path) from error
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _load_inputs(manifest_path: Path, current_path: Path, goal_path: Path) -> Tuple[Any, Any, Mapping[str, Any]]:
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as error:
        raise SuSIELowLevelSmokeError("Pillow and NumPy are required to verify source-fixture PNGs.") from error
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SuSIELowLevelSmokeError("Input manifest is not valid JSON.") from error
    if not isinstance(manifest, Mapping) or manifest.get("schema") != INPUT_SCHEMA or manifest.get("qualified") is not False:
        raise SuSIELowLevelSmokeError("Input manifest must be an explicitly unqualified %s artifact." % INPUT_SCHEMA)
    source = manifest.get("source")
    if not isinstance(source, Mapping) or source.get("sha256") != VENDOR_FIXTURE_SHA256:
        raise SuSIELowLevelSmokeError("Input manifest is not bound to the expected immutable vendor Bridge fixture.")
    records = manifest.get("frames")
    if not isinstance(records, list) or len(records) != 2 or not all(isinstance(item, Mapping) for item in records):
        raise SuSIELowLevelSmokeError("Input manifest must bind exactly current and goal frame records.")
    source_indices = [item.get("source_frame_index") for item in records]
    if not all(isinstance(index, int) for index in source_indices) or source_indices[0] >= source_indices[1]:
        raise SuSIELowLevelSmokeError("Input manifest source frame indices must be strictly increasing.")
    expected_paths = tuple((manifest_path.parent / str(record.get("output_file", ""))).resolve() for record in records)
    if (current_path.resolve(), goal_path.resolve()) != expected_paths:
        raise SuSIELowLevelSmokeError("CLI current/goal paths must exactly match manifest ordering.")
    frames = []
    for path, record in zip((current_path, goal_path), records):
        if record.get("output_file_sha256") != _sha256(path):
            raise SuSIELowLevelSmokeError("Input PNG file SHA-256 differs from manifest: %s" % path)
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB").copy())
        if tuple(frame.shape) != (256, 256, 3) or str(frame.dtype) != "uint8":
            raise SuSIELowLevelSmokeError("Input PNG must decode to RGB uint8 (256, 256, 3): %s" % path)
        if record.get("output_rgb_pixel_sha256") != _pixel_sha256(frame):
            raise SuSIELowLevelSmokeError("Input PNG RGB pixel hash differs from manifest: %s" % path)
        frames.append(frame)
    if _pixel_sha256(frames[0]) == _pixel_sha256(frames[1]):
        raise SuSIELowLevelSmokeError("Goal frame is byte-identical to current frame; refusing a substituted no-op goal.")
    return frames[0], frames[1], manifest


def _runtime_lock(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SuSIELowLevelSmokeError("Runtime lock is not valid JSON.") from error
    if not isinstance(value, Mapping) or value.get("schema") != "plumb-octo-runtime-lock-v1":
        raise SuSIELowLevelSmokeError("This compatibility-deviation probe requires the existing immutable Octo runtime lock.")
    return value


def _disable_tensorflow_gpu() -> str:
    """Run before JAX imports so the allocated GPU belongs to JAX alone."""

    try:
        import tensorflow as tf  # type: ignore
        tf.config.set_visible_devices([], "GPU")
    except ImportError as error:
        raise SuSIELowLevelSmokeError("BridgeData checkpoint runtime requires TensorFlow support.") from error
    except RuntimeError as error:
        raise SuSIELowLevelSmokeError("TensorFlow GPU visibility initialized before SuSIE_LL setup.") from error
    return str(getattr(tf, "__version__", ""))


def _normalizer():
    from plumb.policies.contracts import ActionNormalizationType, PolicyActionNormalizer, PolicyActionNormalizerStatistics
    from plumb.policies.susie import SUSIE_AUTOEVAL_NORMALIZER_SHA256, SUSIE_AUTOEVAL_RELEASED_ARM

    # AutoEval eval_config.py at its pinned commit: ACT_MEAN/ACT_STD. The
    # source wrapper thresholds gripper separately, so mask its seventh value.
    statistics = PolicyActionNormalizerStatistics(
        normalizer_revision="autoeval-gcbc-bridge-meanstd-v1",
        normalization_type=ActionNormalizationType.NORMAL,
        mask=(True, True, True, True, True, True, False),
        mean=(1.9296819e-04, 1.3667766e-04, -1.4583133e-04, -1.8390431e-04, -3.0808983e-04, 2.7425270e-04, 0.0),
        std=(0.00912848, 0.0127196, 0.01229497, 0.02606696, 0.02875283, 0.07807977, 1.0),
        source_uri="https://github.com/zhouzypaul/auto_eval/blob/3ea3ff44c6950433cfbcb4294a3deaa616533745/scripts/configs/eval_config.py#L5-L111",
        statistics_sha256=SUSIE_AUTOEVAL_NORMALIZER_SHA256,
    )
    return PolicyActionNormalizer(statistics), SUSIE_AUTOEVAL_RELEASED_ARM


def _gpu_metadata(jax_module: Any) -> Mapping[str, Any]:
    devices = tuple(jax_module.devices())
    gpu = tuple(device for device in devices if getattr(device, "platform", None) == "gpu")
    if not gpu:
        raise SuSIELowLevelSmokeError("SuSIE_LL diagnostic requires a visible JAX GPU inside a Slurm allocation.")
    return {"devices": [str(item) for item in devices], "gpu_devices": [str(item) for item in gpu], "gpu_device_kinds": [getattr(item, "device_kind", None) for item in gpu]}


def _dependency_deviation(source: Mapping[str, Any], runtime_lock: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compare source requirements to lock metadata without inventing equivalence."""

    required = dict(source.get("source_required_versions", {}))
    actual = dict(runtime_lock.get("packages", {}))
    aliases = {"orbax-checkpoint": "orbax-checkpoint"}
    mismatches = {}
    for name, expected in required.items():
        observed = actual.get(aliases.get(name, name))
        observed_base = None if observed is None else str(observed).split("+", 1)[0]
        if observed_base != expected:
            mismatches[name] = {"source_required": expected, "runtime_lock_actual": observed}
    return {"source_required_versions": required, "runtime_lock_package_versions": actual, "mismatches": mismatches, "full_runtime_tuple_verified": False}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SuSIELowLevelSmokeError("Refusing SuSIE_LL inference outside an allocated Slurm job.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise SuSIELowLevelSmokeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required even for this local-only probe.")
    tensorflow_version = _disable_tensorflow_gpu()
    from plumb.adapters.contracts import PolicyObservation
    from plumb.policies.contracts import NativeWrapperEntryPoint, PolicyExecutionMode
    from plumb.policies.susie import (
        SUSIE_LOW_LEVEL_MODEL_REVISION,
        SUSIE_GCBC_SOURCE_COMMIT,
        SuSIELowLevelPolicyAdapter,
        SuSIEPolicyProfile,
        build_autoeval_gcbc_bridge_low_level,
    )

    current, goal, manifest = _load_inputs(args.input_manifest, args.frame_0, args.goal_frame)
    runtime_lock = _runtime_lock(args.runtime_lock)
    normalizer, arm = _normalizer()
    entry = NativeWrapperEntryPoint(
        module="plumb.policies.susie",
        attribute="build_autoeval_gcbc_bridge_low_level",
        loader_revision=SUSIE_GCBC_SOURCE_COMMIT,
        source_uri="https://github.com/rail-berkeley/soar/blob/%s/model_training/jaxrl_m/agents/continuous/gc_bc.py" % SUSIE_GCBC_SOURCE_COMMIT,
        source_repo_path=str(args.bridge_source),
    )
    profile = SuSIEPolicyProfile(
        profile_id=args.profile_id,
        low_level_model_path=str(args.checkpoint),
        low_level_revision=SUSIE_LOW_LEVEL_MODEL_REVISION,
        arm=arm,
        low_level_entry_point=entry,
        goal_image_source="vendor_bridge_video_first_and_actual_final_frame_static_fixture",
        jax_version=args.jax_version,
        low_level_license="MIT declared in immutable publisher README",
        low_level_license_path=str(args.checkpoint_readme),
        inference_params_only=True,
        runtime_lock_id=args.runtime_lock.stem,
        runtime_lock_sha256=_sha256(args.runtime_lock),
    )
    adapter = SuSIELowLevelPolicyAdapter(profile, normalizer=normalizer, execution_mode=PolicyExecutionMode.FIXTURE_CERTIFICATION)
    adapter.reset(args.seed)
    observation = PolicyObservation(image_history=(current,), goal_image=goal, prompt=args.prompt, timestamp=None)
    started = time.perf_counter()
    first = adapter.propose(observation)
    first_elapsed = time.perf_counter() - started
    model = adapter._low_level
    raw_first = list(getattr(model, "last_model_normalized", ()))
    adapter.reset(args.seed)
    repeated_started = time.perf_counter()
    repeated = adapter.propose(observation)
    repeated_elapsed = time.perf_counter() - repeated_started
    raw_repeat = list(getattr(adapter._low_level, "last_model_normalized", ()))
    if first.actions != repeated.actions or raw_first != raw_repeat:
        raise SuSIELowLevelSmokeError("Same reset seed/current/goal did not reproduce the deterministic gc_bc argmax action.")
    runtime = adapter._load_runtime()
    # The direct loader retains this exact source/runtime deviation record.
    deviation = dict(getattr(model, "source_runtime_deviation", {}))
    if not deviation:
        raise SuSIELowLevelSmokeError("Direct gc_bc loader did not expose source/runtime compatibility metadata.")
    dependency_deviation = _dependency_deviation(deviation, runtime_lock)
    return {
        "schema_version": 1,
        "kind": "plumb_susie_ll_gcbc_static_goal_fixture",
        "qualified": False,
        "status": "completed_unqualified",
        "reason": (
            "One deterministic source-native gc_bc action on a static vendor-video current/goal pair. The goal is the "
            "video's actual final frame, not a task-certified goal; this is not a policy rollout, task success, benchmark, Gate A/B result, or primary row."
        ),
        "source_release": args.source_release,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "profile": {"profile_id": profile.profile_id, "arm": profile.arm.payload(), "checkpoint": str(args.checkpoint), "checkpoint_sha256": _sha256(args.checkpoint), "readme_sha256": _sha256(args.checkpoint_readme), "soar_source": str(args.bridge_source), "soar_source_revision": SUSIE_GCBC_SOURCE_COMMIT, "soar_source_subdirectory": "model_training", "runtime_lock": str(args.runtime_lock), "runtime_lock_sha256": _sha256(args.runtime_lock)},
        "input_fixture": {"manifest": str(args.input_manifest), "manifest_sha256": _sha256(args.input_manifest), "source": dict(manifest["source"]), "frame_files": [str(args.frame_0), str(args.goal_frame)], "frame_file_sha256": [_sha256(args.frame_0), _sha256(args.goal_frame)], "source_frame_indices": [item["source_frame_index"] for item in manifest["frames"]], "goal_role": "actual_vendor_video_final_frame_static_goal_condition_only", "limitation": manifest.get("preprocessing", {}).get("limitation")},
        "runtime": {
            "module_versions": {"jax": runtime.jax_version, "flax": getattr(model, "runtime_flax_version", None), "tensorflow": tensorflow_version},
            "runtime_lock_package_versions": dict(runtime_lock.get("packages", {})),
            "tensorflow_gpu_visible": False,
            "gpu": _gpu_metadata(model._jax),
            "compatibility_deviation": deviation,
            "dependency_deviation": dependency_deviation,
            "runtime_lock_payload": dict(runtime_lock),
        },
        "actions": {"native_model_normalized": raw_first, "transformed_physical": [list(row) for row in first.actions], "repeat_model_normalized": raw_repeat, "repeat_transformed_physical": [list(row) for row in repeated.actions], "finite": True, "shape": [len(first.actions), len(first.actions[0])], "normalizer": normalizer.statistics.payload(), "normalizer_counters": normalizer.counters()},
        "restore": dict(getattr(model, "restore_digest", {})),
        "timing": {
            "first_propose_wall_seconds_including_model_load": first_elapsed,
            "repeat_propose_wall_seconds_model_already_loaded": repeated_elapsed,
            "total_native_calls": 2,
            "first_proposal_backend_calls": first.backend_calls,
            "repeat_proposal_backend_calls": repeated.backend_calls,
        },
        "high_level_subgoal_assets_imported": False,
        "gated_or_unlicensed_components_imported": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-readme", type=Path, required=True)
    parser.add_argument("--bridge-source", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--frame-0", type=Path, required=True)
    parser.add_argument("--goal-frame", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--jax-version", default="0.4.20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Static vendor-video goal-conditioned gc_bc diagnostic")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not str(root).startswith(("/scratch/", "/global/scratch/")) or root.name != "plumb":
        parser.error("--root must be the owning cluster scratch/.../plumb root")
    if len(args.source_release) != 64 or any(char not in "0123456789abcdef" for char in args.source_release.lower()):
        parser.error("--source-release must be a 64-character source release SHA-256")
    reservation = _reserve(args.report)
    started = _utc_now()
    try:
        payload = dict(run(args))
        payload["started_at"] = started
        payload["finished_at"] = _utc_now()
        _publish(args.report, payload)
        return 0
    except Exception as error:
        payload = {"schema_version": 1, "kind": "plumb_susie_ll_gcbc_static_goal_fixture", "qualified": False, "status": "failed", "started_at": started, "finished_at": _utc_now(), "source_release": args.source_release, "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}, "high_level_subgoal_assets_imported": False}
        _publish(args.report, payload)
        return 2
    finally:
        try:
            reservation.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
