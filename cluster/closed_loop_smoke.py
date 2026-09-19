#!/usr/bin/env python3
"""Bounded, real-adapter IRASim/OpenVLA closed-loop diagnostic smoke.

This is deliberately not a PLUMB qualified run.  It drives an injected local
OpenVLA adapter for one native 7-D action at a time, asks the original IRASim
one-step adapter for exactly two frames, drops the echoed condition frame, and
uses the fresh generated frame as the next policy image.  The state path is
explicitly ``reencoded_image_per_tick``: IRASim re-encodes each image and this
is not an upstream latent-carry control protocol or Gate-B evidence.

No models are downloaded.  Heavy ML/image imports happen only in ``main``
after local manifests, the reviewed OpenVLA revision, the safe IRASim tensor
hash, and the original 16/3/1 config have been checked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


DEFAULT_ROOT = Path("/scratch/lchow432/plumb")
REVIEWED_OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
IRASIM_SOURCE_REVISION = "c72b6dade6fcd65971e0aa8ab49ea39b15108c90"
IRASIM_SAFE_TENSOR_SHA256 = "d0ea8e8bec50818e414a278fe58fb8d187f577cfe8b260cadc880cfa6107778d"
DEFAULT_TASK = "Put the pot to the left of the purple item."
WORLD_STATE_MODE = "reencoded_image_per_tick"


class ClosedLoopInputError(ValueError):
    """An input/manifest cannot safely support this bounded diagnostic."""


@dataclass(frozen=True)
class ClosedLoopConfig:
    root: Path
    model_path: Path
    openvla_manifest: Path
    irasim_repo: Path
    irasim_checkpoint: Path
    irasim_checkpoint_sha256: Optional[str]
    irasim_conversion_report: Optional[Path]
    vae_path: Path
    scheduler_path: Path
    irasim_config: Path
    fixture_video: Path
    source_release: Optional[str] = None
    release_manifest: Optional[Path] = None
    task: str = DEFAULT_TASK
    ticks: int = 16
    seed: int = 0
    output_dir: Optional[Path] = None
    report_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not isinstance(self.ticks, int) or isinstance(self.ticks, bool) or not 1 <= self.ticks <= 16:
            raise ClosedLoopInputError("ticks must be an integer from 1 through 16")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ClosedLoopInputError("seed must be an integer")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ClosedLoopInputError("task must be an exact nonempty instruction")

    @property
    def artifacts_dir(self) -> Path:
        # Do not generate a timestamp on every property access: the state,
        # frame, provenance, MP4, and report paths must be one durable bundle.
        # Slurm callers normally pass a job-specific --output-dir.
        return self.output_dir or (self.root / "evidence" / "closed-loop-smoke")

    @property
    def report(self) -> Path:
        return self.report_path or self.artifacts_dir / "report.json"


@dataclass(frozen=True)
class _OutputReservation:
    """Private ownership proof shared only by CLI and internal loop runner."""

    artifact_dir: Path
    report_path: Path
    report_claimed: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, include_hash: bool = True) -> Dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError("required local artifact is absent: %s" % candidate)
    record: Dict[str, Any] = {"path": str(candidate.resolve()), "bytes": candidate.stat().st_size}
    if include_hash:
        record["sha256"] = "sha256:" + _sha256(candidate)
    return record


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _reserve_output(config: ClosedLoopConfig, *, claim_report: bool) -> _OutputReservation:
    """Atomically claim a fresh artifact directory (and CLI report file).

    ``Path.exists()`` followed by ``mkdir(exist_ok=True)`` lets two launchers
    both pass a freshness check. The leaf ``mkdir(exist_ok=False)`` is the
    ownership claim. The CLI also reserves a custom report path with O_EXCL;
    direct loop callers merely reject an existing report because they do not
    write it themselves.
    """

    artifact_dir = config.artifacts_dir
    report_path = config.report
    if report_path.exists():
        raise ClosedLoopInputError("refusing existing closed-loop report path: %s" % report_path)
    try:
        artifact_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ClosedLoopInputError("refusing existing closed-loop artifact directory: %s" % artifact_dir) from exc
    claimed = False
    try:
        if not claim_report:
            # Recheck after the directory claim. This never writes a report,
            # but preserves the direct API's no-preexisting-report contract.
            if report_path.exists():
                raise ClosedLoopInputError("refusing existing closed-loop report path: %s" % report_path)
            return _OutputReservation(artifact_dir, report_path, False)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(report_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        claimed = True
        return _OutputReservation(artifact_dir, report_path, True)
    except BaseException:
        # We alone just created this leaf. Never recursively remove anything;
        # rmdir only succeeds while no artifacts have been written.
        try:
            artifact_dir.rmdir()
        except OSError:
            pass
        raise


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    converter = getattr(value, "tolist", None)
    if callable(converter):
        return _jsonable(converter())
    return repr(value)


def _frame_hash(frame: Any) -> str:
    if isinstance(frame, bytes):
        value = frame
    elif isinstance(frame, str):
        value = frame.encode("utf-8")
    else:
        raw = getattr(frame, "tobytes", None)
        shape = getattr(frame, "shape", None)
        if callable(raw):
            value = (repr(tuple(shape) if shape is not None else "") + ":").encode("utf-8") + raw()
        else:
            value = repr(frame).encode("utf-8")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _timing_dict(timing: Any) -> Dict[str, Any]:
    if callable(getattr(timing, "as_dict", None)):
        return dict(timing.as_dict())
    if is_dataclass(timing):
        return dict(asdict(timing))
    return dict(timing) if isinstance(timing, Mapping) else {"wall_seconds": None}


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClosedLoopInputError("%s is not valid JSON: %s" % (label, path)) from exc
    if not isinstance(payload, dict):
        raise ClosedLoopInputError("%s must be a JSON object" % label)
    return dict(payload)


def _full_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ClosedLoopInputError("%s must be a SHA-256 string" % label)
    normalized = value.lower().removeprefix("sha256:")
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ClosedLoopInputError("%s must be a full 64-character SHA-256" % label)
    return normalized


def verify_openvla_manifest(path: Path) -> Dict[str, Any]:
    manifest = _load_json(path, "OpenVLA manifest")
    revision = manifest.get("revision")
    if revision != REVIEWED_OPENVLA_REVISION:
        raise ClosedLoopInputError(
            "OpenVLA manifest must pin reviewed remote-code/model revision %s" % REVIEWED_OPENVLA_REVISION
        )
    return manifest


def verify_irasim_checkpoint(
    checkpoint: Path, expected_sha256: Optional[str], conversion_report: Optional[Path]
) -> Dict[str, Any]:
    if checkpoint.suffix != ".safetensors":
        raise ClosedLoopInputError("IRASim closed loop accepts only strict-verified .safetensors checkpoints")
    expected = expected_sha256
    conversion: Optional[Dict[str, Any]] = None
    if conversion_report is not None:
        conversion = _load_json(conversion_report, "IRASim conversion report")
        if conversion.get("status") != "completed":
            raise ClosedLoopInputError("IRASim conversion report is not completed")
        report_digest = conversion.get("output_sha256")
        if expected is not None and _full_sha256(expected, "--irasim-checkpoint-sha256") != _full_sha256(report_digest, "conversion output_sha256"):
            raise ClosedLoopInputError("explicit IRASim SHA-256 disagrees with conversion report")
        expected = str(report_digest)
    if expected is None:
        expected = IRASIM_SAFE_TENSOR_SHA256
    expected_normalized = _full_sha256(expected, "IRASim safetensors SHA-256")
    if expected_normalized != IRASIM_SAFE_TENSOR_SHA256:
        raise ClosedLoopInputError(
            "IRASim safetensors SHA-256 must equal the recorded d0ea8… strict-verified checkpoint"
        )
    actual = _sha256(checkpoint)
    if actual != expected_normalized:
        raise ClosedLoopInputError("IRASim safetensors SHA-256 does not match the strict conversion record")
    return {"checkpoint": _file_record(checkpoint), "conversion_report": _file_record(conversion_report) if conversion_report else None,
            "conversion": conversion, "expected_sha256": "sha256:" + expected_normalized}


def verify_irasim_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    expected = {"num_frames": 16, "extras": 3, "mask_frame_num": 1}
    observed: Dict[str, int] = {}
    for key, value in expected.items():
        match = re.search(r"(?m)^\s*%s\s*:\s*(\d+)\s*(?:#.*)?$" % re.escape(key), text)
        if match is None or int(match.group(1)) != value:
            raise ClosedLoopInputError("IRASim config must retain original %s=%d" % (key, value))
        observed[key] = int(match.group(1))
    return {"config": _file_record(path), "verified": observed, "source_revision": IRASIM_SOURCE_REVISION}


def _verify_local_inputs(config: ClosedLoopConfig) -> Dict[str, Any]:
    if not config.root.is_dir():
        raise ClosedLoopInputError("root does not exist: %s" % config.root)
    if not config.model_path.is_dir():
        raise ClosedLoopInputError("OpenVLA model directory is absent: %s" % config.model_path)
    if not config.irasim_repo.is_dir():
        raise ClosedLoopInputError("IRASim source checkout is absent: %s" % config.irasim_repo)
    for path in (config.vae_path, config.scheduler_path):
        if not path.exists():
            raise FileNotFoundError("required local IRASim dependency is absent: %s" % path)
    return {
        "openvla_manifest": verify_openvla_manifest(config.openvla_manifest),
        "irasim_checkpoint": verify_irasim_checkpoint(
            config.irasim_checkpoint, config.irasim_checkpoint_sha256, config.irasim_conversion_report
        ),
        "irasim_config": verify_irasim_config(config.irasim_config),
        "fixture_video": _file_record(config.fixture_video),
        "irasim_source_revision": IRASIM_SOURCE_REVISION,
    }


def _task_binding(task: str) -> Dict[str, str]:
    encoded = task.encode("utf-8")
    return {"task_instruction": task, "task_instruction_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest()}


def _release_binding(config: ClosedLoopConfig) -> Dict[str, Any]:
    """Bind a future evidence bundle to an immutable release when available."""

    release = config.source_release or os.environ.get("PLUMB_SOURCE_RELEASE")
    if not release:
        return {"status": "unavailable", "source_release": None, "release_manifest": None}
    normalized = _full_sha256(release, "source release")
    manifest_path = config.release_manifest or config.root / "releases" / normalized / "RELEASE.json"
    if not manifest_path.is_file():
        return {
            "status": "manifest_unavailable",
            "source_release": normalized,
            "release_manifest": {"path": str(manifest_path)},
        }
    raw = manifest_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    try:
        content = json.loads(raw)
    except ValueError as exc:
        raise ClosedLoopInputError("RELEASE.json is not valid JSON") from exc
    if not isinstance(content, Mapping) or not isinstance(content.get("files"), list):
        raise ClosedLoopInputError("RELEASE.json must contain a files list")
    if actual != normalized:
        raise ClosedLoopInputError("source release hash does not match the exact RELEASE.json bytes")
    return {
        "status": "verified",
        "source_release": normalized,
        "release_manifest": {
            **_file_record(manifest_path),
            "content_sha256": "sha256:" + actual,
            "content": dict(content),
            "release_hash_matches_manifest_bytes": True,
        },
    }


class _OpenVLALoopAdapter:
    """Small bridge from the real adapter's observation contract to this loop."""

    def __init__(self, adapter: Any, observation_cls: Any) -> None:
        self.adapter = adapter
        self.observation_cls = observation_cls

    def action_for(self, image: Any, task: str, timestamp: float) -> Tuple[Tuple[float, ...], Mapping[str, Any]]:
        report = self.adapter.predict_with_report(
            self.observation_cls(image_history=(image,), prompt=task, timestamp=timestamp)
        )
        action = tuple(float(value) for value in report.action)
        return action, {
            "unnorm_key": getattr(report, "unnorm_key", None),
            "source_image_timestamp": getattr(report, "source_image_timestamp", None),
            "backend_calls": getattr(report, "backend_calls", None),
            "wall_seconds": getattr(report, "wall_seconds", None),
            "gpu_peak_memory_bytes": getattr(report, "gpu_peak_memory_bytes", None),
        }


def _load_real_adapters(config: ClosedLoopConfig) -> Tuple[Any, Any]:
    """Import ML stacks only after manifest/asset validation on a cluster node."""

    openvla_module = importlib.import_module("plumb.policies.openvla")
    contracts = importlib.import_module("plumb.adapters.contracts")
    irasim_module = importlib.import_module("plumb.adapters.irasim_runtime")
    profile = openvla_module.OpenVLAProfile if hasattr(openvla_module, "OpenVLAProfile") else openvla_module.OpenVLAPolicyProfile
    openvla_profile = profile(
        profile_id="openvla-irasim-closed-loop-diagnostic",
        local_model_path=str(config.model_path),
        checkpoint_revision=REVIEWED_OPENVLA_REVISION,
        remote_code_revision=REVIEWED_OPENVLA_REVISION,
        reviewed_remote_code_revision=REVIEWED_OPENVLA_REVISION,
        allow_trust_remote_code=True,
    )
    policy = _OpenVLALoopAdapter(openvla_module.OpenVLAPolicyAdapter(openvla_profile), contracts.PolicyObservation)
    irasim_profile = irasim_module.IRASimOneStepProfile(
        "irasim-original-one-step-reencoded-feedback",
        str(config.irasim_repo),
        str(config.irasim_checkpoint),
        str(config.vae_path),
        str(config.scheduler_path),
        str(config.irasim_config),
    )
    return policy, irasim_module.IRASimOneStepAdapter(irasim_profile)


def _load_vendor_frame_zero(video_path: Path) -> Any:
    """Decode only vendor fixture frame 0; imports stay out of CPU/test paths."""

    import imageio.v3 as iio  # type: ignore
    import numpy as np  # type: ignore

    frame = iio.imread(video_path, index=0)
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ClosedLoopInputError("vendor fixture frame 0 must decode as RGB image data")
    return array[..., :3].astype("uint8", copy=False)


def _irasim_frame_to_rgb(frame: Any) -> Any:
    """Convert the generated IRASim tensor/frame to a fresh RGB policy image."""

    import numpy as np  # type: ignore

    value = frame.detach().float().cpu().numpy() if callable(getattr(frame, "detach", None)) else np.asarray(frame)
    if value.ndim == 3 and value.shape[0] in (3, 4):
        value = value[:3].transpose(1, 2, 0)
    elif value.ndim == 3 and value.shape[-1] in (3, 4):
        value = value[..., :3]
    else:
        raise ClosedLoopInputError("IRASim output frame must be CHW/HWC RGB")
    if value.dtype.kind == "f":
        # IRASim's VAE decoder contract is [-1, 1]. Small decode overshoots
        # (for example 1.2) are clipped *after* this fixed mapping, never used
        # to guess a different [0,1] convention that would corrupt negatives.
        if not np.isfinite(value).all():
            raise ClosedLoopInputError("IRASim floating output frame contains non-finite values")
        value = (value + 1.0) * 127.5
    elif value.dtype != np.uint8:
        raise ClosedLoopInputError("IRASim output frame must be uint8 or decoded floating [-1,1] RGB")
    return value.clip(0, 255).astype("uint8")


def _irasim_float_to_uint8(value: Any) -> Any:
    """Apply the fixed decoded-IRASim [-1,1] mapping for focused tests/tools."""

    import numpy as np  # type: ignore

    array = np.asarray(value, dtype="float32")
    if not np.isfinite(array).all():
        raise ClosedLoopInputError("IRASim decoded float values must be finite")
    return ((array + 1.0) * 127.5).clip(0, 255).astype("uint8")


def _video_export_frames(frames: Sequence[Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Resize copies for MP4 only; never mutate policy/world feedback images.

    The first vendor condition can be 480×640 while IRASim's generated image
    is 256×320.  The first generated frame declares the presentation size, so
    the raw vendor condition remains separately persisted and untouched while
    only its MP4 copy is resized.
    """

    if len(frames) < 2:
        raise ClosedLoopInputError("video export needs condition plus at least one generated frame")
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    first_generated = np.asarray(frames[1])
    if first_generated.ndim != 3 or first_generated.shape[-1] != 3 or first_generated.dtype != np.uint8:
        raise ClosedLoopInputError("generated video-export frame must be uint8 HWC RGB")
    target_hw = (int(first_generated.shape[0]), int(first_generated.shape[1]))
    exported = []
    resizes = []
    for index, raw in enumerate(frames):
        array = np.asarray(raw)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ClosedLoopInputError("video-export frames must be uint8 HWC RGB")
        source_hw = (int(array.shape[0]), int(array.shape[1]))
        if source_hw != target_hw:
            resampling = getattr(Image, "Resampling", Image)
            resized = np.asarray(
                Image.fromarray(array).resize((target_hw[1], target_hw[0]), resampling.BILINEAR)
            )
            exported.append(resized)
            resizes.append(
                {
                    "frame_index": index,
                    "from_hw": list(source_hw),
                    "to_hw": list(target_hw),
                    "scope": "video_export_only",
                }
            )
        else:
            exported.append(array.copy())
    return tuple(exported), {
        "target_hw": list(target_hw),
        "video_only_resizes": resizes,
        "raw_policy_world_frames_unchanged": True,
    }


def _write_png(path: Path, frame: Any) -> None:
    import imageio.v3 as iio  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frame)


def _write_mp4(path: Path, frames: Sequence[Any]) -> None:
    import imageio.v3 as iio  # type: ignore
    import numpy as np  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(tuple(frames)), fps=5, codec="libx264")


def _call_action(policy: Any, image: Any, task: str, timestamp: float) -> Tuple[Tuple[float, ...], Mapping[str, Any]]:
    method = getattr(policy, "action_for", None)
    if callable(method):
        result = method(image, task, timestamp)
    elif callable(policy):
        result = policy(image, task, timestamp)
    else:
        raise ClosedLoopInputError("policy adapter must expose action_for(image, task, timestamp)")
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        action, details = result
    else:
        action, details = result, {}
    if isinstance(action, (str, bytes)) or not isinstance(action, Sequence) or len(action) != 7:
        raise ClosedLoopInputError("OpenVLA closed-loop tick must return exactly one raw native 7-D action")
    values = tuple(float(value) for value in action)
    if any(not math.isfinite(value) for value in values):
        raise ClosedLoopInputError("OpenVLA action must be finite")
    return values, dict(details)


def _step_seed(seed: int, index: int) -> int:
    return int.from_bytes(hashlib.sha256(("%d:%d" % (seed, index)).encode("ascii")).digest()[:8], "big") & ((1 << 63) - 1)


def _run_closed_loop_owned(
    config: ClosedLoopConfig,
    policy: Any,
    world: Any,
    initial_frame: Any,
    reservation: _OutputReservation,
    *,
    frame_converter: Callable[[Any], Any] = lambda frame: frame,
    png_writer: Optional[Callable[[Path, Any], None]] = None,
    mp4_writer: Optional[Callable[[Path, Sequence[Any]], None]] = None,
    video_exporter: Callable[[Sequence[Any]], Tuple[Tuple[Any, ...], Mapping[str, Any]]] = _video_export_frames,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run 1..16 true image-feedback ticks and persist state after each one."""

    artifact_dir = reservation.artifact_dir
    state_dir = artifact_dir / "state"
    frames_dir = artifact_dir / "frames"
    png_writer = png_writer or _write_png
    mp4_writer = mp4_writer or _write_mp4
    artifact_dir.mkdir(parents=True, exist_ok=True)
    current = initial_frame
    frames = [current]
    initial_path = frames_dir / "frame-000.png"
    png_writer(initial_path, current)
    current_pixel_sha256 = _frame_hash(current)
    current_file_sha256 = "sha256:" + _sha256(initial_path)
    _atomic_json(
        artifact_dir / "provenance.json",
        {
            "schema_version": 2,
            "kind": "plumb_irasim_openvla_closed_loop_diagnostic",
            "qualification": "experimental_unqualified_not_gate_b",
            "qualified": False,
            "world_state_mode": WORLD_STATE_MODE,
            "action_compatibility": "experimental_not_gate_b_qualified",
            "physical_state_measured": False,
            "policy_feedback": "fresh_generated_image_per_tick",
            "backend_action_contract": "IRASim native raw 7-D action with internal [20,20,20,20,20,20,1] scale; never Cosmos 10-D",
            **_task_binding(config.task),
            "initial_frame_artifact": {
                "path": str(initial_path),
                # Compatibility field: hashes decoded in-memory RGB pixels,
                # not the serialized PNG byte stream.
                "sha256": current_pixel_sha256,
                "sha256_semantics": "decoded_rgb_pixel_sha256",
                "decoded_pixel_sha256": current_pixel_sha256,
                "file_sha256": current_file_sha256,
            },
            "source": dict(provenance or {}),
        },
    )
    records = []
    started = time.perf_counter()
    for index in range(config.ticks):
        seed = _step_seed(config.seed, index)
        record: Dict[str, Any] = {
            "schema_version": 2,
            "kind": "plumb_closed_loop_tick",
            "tick": index,
            "seed": seed,
            "input_frame_index": index,
            "input_frame_sha256": current_pixel_sha256,
            "input_frame_sha256_semantics": "decoded_rgb_pixel_sha256",
            "input_frame_decoded_pixel_sha256": current_pixel_sha256,
            "input_frame_file_sha256": current_file_sha256,
            # This is the RGB frame handed to the policy after the prior
            # IRASim tensor->RGB conversion (or vendor frame-0 decode).
            "preprocessed_policy_input_frame_sha256": current_pixel_sha256,
            **_task_binding(config.task),
            "world_state_mode": WORLD_STATE_MODE,
            "qualification": "experimental_unqualified_not_gate_b",
            "status": "running",
        }
        try:
            policy_started = time.perf_counter()
            action, policy_timing = _call_action(policy, current, config.task, float(index) / 5.0)
            record["policy_timing"] = {"controller_wall_seconds": time.perf_counter() - policy_started, **policy_timing}
            record["raw_native_action_7d"] = list(action)
            world_started = time.perf_counter()
            result = world.generate_one_step(current, action, seed=seed)
            record["world_controller_wall_seconds"] = time.perf_counter() - world_started
            result_frames = tuple(result.frames)
            if len(result_frames) != 2:
                raise ClosedLoopInputError("IRASim one-step adapter must return exactly condition + one future frame")
            # Do not let the repeated condition masquerade as generated
            # feedback or inflate the artifact frame count.
            future = frame_converter(result_frames[1])
            record["world_post_frame0_dropped"] = True
            record["world_returned_frame_count"] = 2
            future_pixel_sha256 = _frame_hash(future)
            record["generated_frame_sha256"] = future_pixel_sha256
            record["generated_frame_sha256_semantics"] = "decoded_rgb_pixel_sha256"
            record["generated_frame_decoded_pixel_sha256"] = future_pixel_sha256
            record["preprocessed_generated_rgb_frame_sha256"] = future_pixel_sha256
            record["native_action_scaled_internal"] = list(getattr(result, "native_action_scaled", ()))
            record["world_timing"] = _timing_dict(getattr(result, "timing", None))
            record["condition_preprocessing"] = _jsonable(getattr(result, "condition_preprocessing", {}))
            frame_path = frames_dir / ("frame-%03d.png" % (index + 1))
            png_writer(frame_path, future)
            future_file_sha256 = "sha256:" + _sha256(frame_path)
            record["generated_frame_artifact"] = {
                "path": str(frame_path),
                # Kept for older readers; explicit semantics disambiguate it.
                "sha256": future_pixel_sha256,
                "sha256_semantics": "decoded_rgb_pixel_sha256",
                "decoded_pixel_sha256": future_pixel_sha256,
                "file_sha256": future_file_sha256,
            }
            record["status"] = "completed"
            _atomic_json(state_dir / ("step-%03d.json" % index), record)
            records.append(record)
            current = future
            current_pixel_sha256 = future_pixel_sha256
            current_file_sha256 = future_file_sha256
            frames.append(future)
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)},
                }
            )
            _atomic_json(state_dir / ("step-%03d.json" % index), record)
            records.append(record)
            return {
                "schema_version": 2,
                "kind": "plumb_irasim_openvla_closed_loop_diagnostic",
                "status": "failed",
                "qualified": False,
                "binary_success": None,
                "progress_score": None,
                "validity": "unknown",
                "missing_reason": "closed_loop_service_failure",
                "world_state_mode": WORLD_STATE_MODE,
                "action_compatibility": "experimental_not_gate_b_qualified",
                **_task_binding(config.task),
                "ticks_requested": config.ticks,
                "ticks_completed": len([item for item in records if item["status"] == "completed"]),
                "frames_persisted": len(frames),
                "step_records": records,
                "timing": {"total_seconds": time.perf_counter() - started},
                "error": record["error"],
            }

    video_path = artifact_dir / "closed-loop.mp4"
    try:
        export_frames, export_metadata = video_exporter(frames)
        mp4_writer(video_path, export_frames)
        video_artifact: Optional[Mapping[str, Any]] = _file_record(video_path)
    except Exception as exc:
        return {
            "schema_version": 2,
            "kind": "plumb_irasim_openvla_closed_loop_diagnostic",
            "status": "failed",
            "qualified": False,
            "binary_success": None,
            "progress_score": None,
            "validity": "unknown",
            "missing_reason": "artifact_write_failure",
            "world_state_mode": WORLD_STATE_MODE,
            "action_compatibility": "experimental_not_gate_b_qualified",
            **_task_binding(config.task),
            "ticks_requested": config.ticks,
            "ticks_completed": config.ticks,
            "frames_persisted": len(frames),
            "step_records": records,
            "timing": {"total_seconds": time.perf_counter() - started},
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    return {
        "schema_version": 2,
        "kind": "plumb_irasim_openvla_closed_loop_diagnostic",
        "status": "completed_unqualified_diagnostic",
        "qualified": False,
        # No validity stage or judge is part of this smoke. Never promote it.
        "binary_success": None,
        "progress_score": None,
        "validity": "unknown",
        "missing_reason": "diagnostic_unqualified_no_judge",
        "world_state_mode": WORLD_STATE_MODE,
        "action_compatibility": "experimental_not_gate_b_qualified",
        **_task_binding(config.task),
        "physical_state_measured": False,
        "ticks_requested": config.ticks,
        "ticks_completed": config.ticks,
        "frames_persisted": len(frames),
        "video": video_artifact,
        "video_export": dict(export_metadata),
        "step_records": records,
        "timing": {"total_seconds": time.perf_counter() - started},
    }


def run_closed_loop(
    config: ClosedLoopConfig,
    policy: Any,
    world: Any,
    initial_frame: Any,
    *,
    frame_converter: Callable[[Any], Any] = lambda frame: frame,
    png_writer: Optional[Callable[[Path, Any], None]] = None,
    mp4_writer: Optional[Callable[[Path, Sequence[Any]], None]] = None,
    video_exporter: Callable[[Sequence[Any]], Tuple[Tuple[Any, ...], Mapping[str, Any]]] = _video_export_frames,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Public direct runner; atomically claims output before its first write."""

    reservation = _reserve_output(config, claim_report=False)
    return _run_closed_loop_owned(
        config,
        policy,
        world,
        initial_frame,
        reservation,
        frame_converter=frame_converter,
        png_writer=png_writer,
        mp4_writer=mp4_writer,
        video_exporter=video_exporter,
        provenance=provenance,
    )


def run_cli(config: ClosedLoopConfig) -> Tuple[int, Dict[str, Any]]:
    """Validate all local pins, run the bounded loop, and always write a report."""

    started = time.perf_counter()
    report_path = config.report
    base: Dict[str, Any] = {
        "schema_version": 2,
        "kind": "plumb_irasim_openvla_closed_loop_diagnostic",
        "status": "failed",
        "qualified": False,
        "binary_success": None,
        "progress_score": None,
        "validity": "unknown",
        "world_state_mode": WORLD_STATE_MODE,
        "action_compatibility": "experimental_not_gate_b_qualified",
        **_task_binding(config.task),
        "ticks_requested": config.ticks,
        "started_at": _utc_now(),
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    reservation: Optional[_OutputReservation] = None
    try:
        # Must precede manifest hashing/imports/model work. If this fails, do
        # not write a setup-failure report over the old requested report.
        reservation = _reserve_output(config, claim_report=True)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        release = _release_binding(config)
        base["source_release"] = release.get("source_release")
        base["source_release_binding"] = release
        provenance = _verify_local_inputs(config)
        provenance["source_release_binding"] = release
        initial = _load_vendor_frame_zero(config.fixture_video)
        policy, world = _load_real_adapters(config)
        outcome = _run_closed_loop_owned(
            config,
            policy,
            world,
            initial,
            reservation,
            frame_converter=_irasim_frame_to_rgb,
            provenance=provenance,
        )
        base.update(outcome)
        code = 0 if outcome["status"] == "completed_unqualified_diagnostic" else 2
    except Exception as exc:
        base.update(
            {
                "status": "failed",
                "missing_reason": "closed_loop_setup_failure",
                "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=10)},
            }
        )
        code = 2
    base["finished_at"] = _utc_now()
    base["timing"] = {**dict(base.get("timing", {})), "total_seconds": time.perf_counter() - started}
    # A CLI report placeholder is O_EXCL-claimed before setup. If reservation
    # failed, leave any existing evidence entirely untouched.
    if reservation is None or not reservation.report_claimed:
        return 2, base
    _atomic_json(report_path, base)
    return code, base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PLUMB_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--ticks", type=int, default=16, help="Bounded number of real one-action feedback ticks (1..16).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--openvla-manifest", type=Path)
    parser.add_argument("--irasim-repo", type=Path)
    parser.add_argument("--irasim-checkpoint", type=Path)
    parser.add_argument("--irasim-checkpoint-sha256", help="Full strict-verified safetensors SHA-256; may come from conversion report.")
    parser.add_argument("--irasim-conversion-report", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--scheduler", type=Path)
    parser.add_argument("--irasim-config", type=Path)
    parser.add_argument("--fixture-video", type=Path)
    parser.add_argument("--source-release", help="Frozen 64-character source release hash, if this run uses a release bundle.")
    parser.add_argument("--release-manifest", type=Path, help="Exact RELEASE.json corresponding to --source-release.")
    return parser


def config_from_args(args: argparse.Namespace) -> ClosedLoopConfig:
    root = Path(args.root)
    return ClosedLoopConfig(
        root=root,
        model_path=args.model_path or root / "models" / "openvla--openvla-7b",
        openvla_manifest=args.openvla_manifest or root / "evidence" / "openvla--openvla-7b-download.json",
        irasim_repo=args.irasim_repo or root / "irasim-source",
        irasim_checkpoint=args.irasim_checkpoint or root / "models" / "irasim" / "frame_ada_0300000.safetensors",
        irasim_checkpoint_sha256=args.irasim_checkpoint_sha256,
        irasim_conversion_report=args.irasim_conversion_report,
        vae_path=args.vae or root / "models" / "stabilityai--stable-diffusion-xl-base-1.0",
        scheduler_path=args.scheduler or root / "irasim-source" / "pretrained_models" / "scheduler",
        irasim_config=args.irasim_config or root / "irasim-source" / "configs" / "evaluation" / "bridge" / "frame_ada.yaml",
        fixture_video=args.fixture_video or root / "fixtures" / "bridge_video.mp4",
        source_release=args.source_release or os.environ.get("PLUMB_SOURCE_RELEASE"),
        release_manifest=args.release_manifest,
        task=args.task,
        ticks=args.ticks,
        seed=args.seed,
        output_dir=args.output_dir or root / "evidence" / ("closed-loop-%s" % _run_stamp()),
        report_path=args.report,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    code, report = run_cli(config)
    print(json.dumps(_jsonable(report), sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
