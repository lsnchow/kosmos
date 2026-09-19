#!/usr/bin/env python3
"""Unqualified paired IRASim state-representation replay intervention.

This command replays exactly the 16 raw 7-D actions and recorded per-tick
seeds from a completed OpenVLA/IRASim closed-loop diagnostic.  It never loads
or queries a policy and never produces a task-success, progress, or fidelity
score.  It compares two state transports only:

* ``image_reencode`` re-encodes the preceding generated uint8 RGB frame;
* ``latent_carry`` carries ``output_latents[:, -1:].clone()``.

The first VAE latent is sampled once and cloned into both branches.  Every
subsequent diffusion call uses an independently constructed, identically
seeded diffusion generator per branch.  VAE sampling uses a different,
recorded seed stream, so skipping an encode in the latent branch cannot shift
the diffusion RNG draws.  This differs intentionally from the source
closed-loop run's shared VAE/diffusion generator protocol.

The implementation follows the checked original IRASim source at
``c72b6dade6fcd65971e0aa8ab49ea39b15108c90``: its autoregressive sample
script carries the final latent, while ``output_type='both'`` returns decoded
video and final denoised latents.  This is a bounded intervention, not a
claim that latent carry fixes image-feedback fidelity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:  # Package import for tests/source-root execution.
    from cluster.closed_loop_smoke import (
        DEFAULT_ROOT,
        IRASIM_SAFE_TENSOR_SHA256,
        IRASIM_SOURCE_REVISION,
        _irasim_frame_to_rgb,
        _load_vendor_frame_zero,
        _write_mp4,
        _write_png,
        _video_export_frames,
        verify_irasim_checkpoint,
        verify_irasim_config,
    )
except ModuleNotFoundError:  # Direct ``python cluster/irasim_state_replay.py`` execution.
    from closed_loop_smoke import (  # type: ignore
        DEFAULT_ROOT,
        IRASIM_SAFE_TENSOR_SHA256,
        IRASIM_SOURCE_REVISION,
        _irasim_frame_to_rgb,
        _load_vendor_frame_zero,
        _write_mp4,
        _write_png,
        _video_export_frames,
        verify_irasim_checkpoint,
        verify_irasim_config,
    )


SOURCE_RUN_ID = "closed-loop-937704"
REPLAY_TICKS = 16
REPLAY_KIND = "plumb_irasim_state_representation_replay_intervention"
QUALIFICATION = "experimental_unqualified_intervention_not_gate_b"


class ReplayInputError(ValueError):
    """A saved source record or local artifact cannot support this replay."""


@dataclass(frozen=True)
class ReplayInput:
    tick: int
    action: Tuple[float, float, float, float, float, float, float]
    recorded_seed: int


@dataclass(frozen=True)
class ReplayConfig:
    root: Path
    source_report: Path
    output_dir: Path
    irasim_repo: Path
    irasim_checkpoint: Path
    irasim_checkpoint_sha256: Optional[str]
    irasim_conversion_report: Optional[Path]
    vae_path: Path
    scheduler_path: Path
    irasim_config: Path
    fixture_video: Path
    source_release: Optional[str] = None
    same_seed_repeat: bool = False

    def __post_init__(self) -> None:
        if self.output_dir.exists():
            raise ReplayInputError("Replay output directory already exists; refusing overwrite: %s" % self.output_dir)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Required local artifact is absent: %s" % path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": "sha256:" + _sha256(path)}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReplayInputError("%s is not valid JSON: %s" % (label, path)) from error
    if not isinstance(payload, dict):
        raise ReplayInputError("%s must be a JSON object" % label)
    return dict(payload)


def _normalised_sha256(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ReplayInputError("%s must be a SHA-256 string" % label)
    digest = value.lower()
    if digest.startswith("sha256:"):
        digest = digest[len("sha256:") :]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ReplayInputError("%s must be a full SHA-256 digest" % label)
    return digest


def load_replay_inputs(report_path: Path) -> Tuple[ReplayInput, ...]:
    """Read only raw saved action rows/seeds; policy model calls are impossible here."""

    report = _load_json(report_path, "source closed-loop report")
    if report.get("kind") != "plumb_irasim_openvla_closed_loop_diagnostic":
        raise ReplayInputError("Source report is not the expected closed-loop diagnostic kind.")
    rows = report.get("step_records")
    if not isinstance(rows, list) or len(rows) != REPLAY_TICKS:
        raise ReplayInputError("Source report must contain exactly 16 saved step_records.")
    values = []
    for expected_tick, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("tick") != expected_tick:
            raise ReplayInputError("Source step_records must be ordered and complete from tick 0 through 15.")
        action = row.get("raw_native_action_7d")
        seed = row.get("seed")
        if not isinstance(action, list) or len(action) != 7:
            raise ReplayInputError("Source tick %d lacks one raw native 7-D action." % expected_tick)
        try:
            parsed_action = tuple(float(item) for item in action)
        except (TypeError, ValueError) as error:
            raise ReplayInputError("Source tick %d action is not numeric." % expected_tick) from error
        if not all(math.isfinite(item) for item in parsed_action):
            raise ReplayInputError("Source tick %d action has non-finite values." % expected_tick)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 1 << 63:
            raise ReplayInputError("Source tick %d seed must be a non-negative signed-63-bit integer." % expected_tick)
        values.append(ReplayInput(expected_tick, parsed_action, seed))  # type: ignore[arg-type]
    return tuple(values)


def source_fixture_provenance(report_path: Path) -> Tuple[Mapping[str, Any], str, Optional[Path]]:
    """Use report-local provenance, or its sibling sidecar in actual 937704 shape."""

    report = _load_json(report_path, "source closed-loop report")
    source = report.get("source")
    origin = "source_report"
    sidecar_path: Optional[Path] = None
    if not isinstance(source, dict):
        sidecar_path = report_path.with_name("provenance.json")
        sidecar = _load_json(sidecar_path, "source closed-loop provenance sidecar")
        if sidecar.get("kind") != report.get("kind"):
            raise ReplayInputError("Source report and sibling provenance sidecar have different diagnostic kinds.")
        source = sidecar.get("source")
        origin = "sibling_provenance_sidecar"
    fixture = source.get("fixture_video") if isinstance(source, dict) else None
    if not isinstance(fixture, dict):
        raise ReplayInputError("Source report/provenance sidecar lacks fixture_video provenance.")
    return dict(fixture), origin, sidecar_path


def source_fixture_sha256(report_path: Path) -> str:
    fixture, _origin, _sidecar = source_fixture_provenance(report_path)
    return _normalised_sha256(fixture.get("sha256"), "source fixture_video.sha256")


def derive_stream_seed(recorded_seed: int, stream: str) -> int:
    """Derive named VAE/diffusion seeds without consuming another generator."""

    if isinstance(recorded_seed, bool) or not isinstance(recorded_seed, int) or recorded_seed < 0:
        raise ReplayInputError("recorded seed must be a non-negative integer")
    if stream not in ("initial_vae", "vae", "diffusion"):
        raise ReplayInputError("Unsupported deterministic replay RNG stream %r." % stream)
    digest = hashlib.sha256(("irasim-state-replay-v1:%d:%s" % (recorded_seed, stream)).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def carry_last_latent(latents: Any) -> Any:
    """Use only the final returned time slice, cloned for the next latent branch tick."""

    shape = tuple(getattr(latents, "shape", ()))
    if len(shape) != 5 or int(shape[1]) != 2:
        raise ReplayInputError("IRASim output latents must have [B,2,4,32,40] shape, got %s." % (shape,))
    final = latents[:, -1:]
    clone = getattr(final, "clone", None)
    if not callable(clone):
        raise ReplayInputError("IRASim final latent must expose clone().")
    return clone()


def tensor_hash(tensor: Any) -> str:
    """Stable dtype/shape/raw-byte hash for a final or conditioning latent tensor."""

    value = tensor.detach() if callable(getattr(tensor, "detach", None)) else tensor
    value = value.contiguous() if callable(getattr(value, "contiguous", None)) else value
    value = value.cpu() if callable(getattr(value, "cpu", None)) else value
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    raw = getattr(value, "tobytes", None)
    shape = getattr(value, "shape", None)
    if not callable(raw) or shape is None:
        raise ReplayInputError("Latent tensor must provide shape and raw bytes for hashing.")
    header = json.dumps({"shape": [int(item) for item in tuple(shape)], "dtype": str(getattr(value, "dtype", "unknown"))}, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(header + b"\0" + bytes(raw())).hexdigest()


def raw_pixel_hash(frame: Any) -> str:
    value = frame
    raw = getattr(value, "tobytes", None)
    shape = getattr(value, "shape", None)
    if not callable(raw) or shape is None:
        raise ReplayInputError("RGB frame must provide shape and raw bytes for hashing.")
    header = json.dumps({"shape": [int(item) for item in tuple(shape)], "dtype": str(getattr(value, "dtype", "unknown"))}, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(header + b"\0" + bytes(raw())).hexdigest()


def _pixel_mae(left: Any, right: Any) -> float:
    """Descriptive RGB difference only; never a task, success, or fidelity score."""

    try:
        import numpy as np  # type: ignore
    except ImportError as error:
        raise ReplayInputError("numpy is required for descriptive replay pixel MAE.") from error
    first = np.asarray(left, dtype="float32")
    second = np.asarray(right, dtype="float32")
    if first.shape != second.shape:
        raise ReplayInputError("Paired branch output frames have different RGB shapes.")
    return float(np.abs(first - second).mean())


def generator_metadata(recorded_seed: int) -> Mapping[str, int]:
    """Describe independent streams; both branch diffusion seeds are equal."""

    return {
        "recorded_seed": recorded_seed,
        "vae_seed": derive_stream_seed(recorded_seed, "vae"),
        "image_branch_diffusion_seed": derive_stream_seed(recorded_seed, "diffusion"),
        "latent_branch_diffusion_seed": derive_stream_seed(recorded_seed, "diffusion"),
    }


def _torch_generator(torch: Any, device: str, seed: int) -> Any:
    return torch.Generator(device=device).manual_seed(int(seed))


def _encode_mask(adapter: Any, torch: Any, pipeline: Any, frame: Any, seed: int) -> Tuple[Any, Mapping[str, Any]]:
    condition, preprocessing = adapter._condition(torch, frame)
    vae = pipeline.vae
    with torch.no_grad():
        latent = vae.encode(condition.unsqueeze(0)).latent_dist.sample(
            generator=_torch_generator(torch, adapter.profile.device, seed)
        ).mul_(vae.config.scaling_factor)
    mask_x = latent.unsqueeze(1)
    expected_tail = (4, 32, 40)
    shape = tuple(int(item) for item in mask_x.shape)
    if shape != (1, 1) + expected_tail:
        raise ReplayInputError("IRASim VAE mask_x must be [B,1,4,32,40], got %s." % (shape,))
    _require_finite(torch, mask_x, "IRASim VAE mask_x")
    return mask_x, preprocessing


def _scaled_action(torch: Any, adapter: Any, action: Sequence[float]) -> Any:
    scale = (20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0)
    values = tuple(float(action[index]) * scale[index] for index in range(7))
    return torch.tensor(values, device=adapter.profile.device, dtype=torch.float32).view(1, 1, 7)


def _pipeline_step(torch: Any, adapter: Any, pipeline: Any, mask_x: Any, action: Sequence[float], diffusion_seed: int) -> Tuple[Any, Any]:
    device = torch.device(adapter.profile.device)
    videos, latents = pipeline(
        _scaled_action(torch, adapter, action),
        mask_x=mask_x,
        video_length=2,
        height=adapter.profile.input_height,
        width=adapter.profile.input_width,
        num_inference_steps=adapter.profile.inference_steps,
        guidance_scale=adapter.profile.guidance_scale,
        generator=_torch_generator(torch, adapter.profile.device, diffusion_seed),
        output_type="both",
        return_dict=False,
        device=device,
    )
    if videos is None or latents is None:
        raise ReplayInputError("IRASim pipeline output_type='both' did not return video and latents.")
    latent_shape = tuple(int(item) for item in latents.shape)
    if latent_shape != (1, 2, 4, 32, 40):
        raise ReplayInputError("IRASim output latents must be [B,2,4,32,40], got %s." % (latent_shape,))
    _require_finite(torch, latents, "IRASim output latents")
    if not bool(torch.equal(latents[:, :1], mask_x)):
        raise ReplayInputError("IRASim pipeline did not retain mask_x at the condition latent position.")
    if int(videos.shape[1]) != 2:
        raise ReplayInputError("IRASim one-action replay must return exactly two video frames.")
    return videos, latents


def _branch_artifact(
    output_dir: Path,
    branch: str,
    frame_index: int,
    frame: Any,
    output_latents: Any,
    final_latent: Optional[Any] = None,
) -> Mapping[str, Any]:
    png_path = output_dir / branch / "frames" / ("frame-%03d.png" % frame_index)
    _write_png(png_path, frame)
    final = final_latent if final_latent is not None else output_latents
    return {
        "frame_index": frame_index,
        "raw_pixel_sha256": raw_pixel_hash(frame),
        "png": _file_record(png_path),
        "output_latent_tensor_sha256": tensor_hash(output_latents),
        "final_latent_tensor_sha256": tensor_hash(final),
    }


def _require_finite(torch: Any, tensor: Any, label: str) -> None:
    isfinite = getattr(torch, "isfinite", None)
    if not callable(isfinite):
        raise ReplayInputError("Runtime cannot verify finite values for %s." % label)
    result = isfinite(tensor).all()
    item = getattr(result, "item", None)
    finite = bool(item()) if callable(item) else bool(result)
    if not finite:
        raise ReplayInputError("%s contains non-finite values." % label)


def verify_irasim_source_checkout(repo: Path) -> Mapping[str, Any]:
    """Bind execution to the pinned checked-out source, not a revision string."""

    tracked = ("sample/sample_autoregressive.py", "sample/pipeline_trajectory2videogen.py")
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReplayInputError("IRASim source checkout must expose a readable git HEAD.") from error
    if head != IRASIM_SOURCE_REVISION:
        raise ReplayInputError("IRASim git HEAD %s does not match pinned %s." % (head, IRASIM_SOURCE_REVISION))
    clean = {}
    for label, prefix in (("unstaged", ["git", "-C", str(repo), "diff", "--quiet", "--"]),
                          ("staged", ["git", "-C", str(repo), "diff", "--cached", "--quiet", "--"])):
        completed = subprocess.run(prefix + list(tracked), capture_output=True, text=True)
        if completed.returncode == 1:
            raise ReplayInputError("IRASim targeted source files have %s git changes." % label)
        if completed.returncode != 0:
            raise ReplayInputError("Could not verify %s IRASim targeted source diff." % label)
        clean[label] = True
    return {"git_head": head, "targeted_source_files": list(tracked), "targeted_git_diff_clean": clean}


def _load_adapter(config: ReplayConfig) -> Any:
    """Load the existing bounded original IRASim adapter only after preflight."""

    from plumb.adapters.irasim_runtime import IRASimOneStepAdapter, IRASimOneStepProfile

    profile = IRASimOneStepProfile(
        "irasim-state-replay-intervention",
        str(config.irasim_repo),
        str(config.irasim_checkpoint),
        str(config.vae_path),
        str(config.scheduler_path),
        str(config.irasim_config),
    )
    return IRASimOneStepAdapter(profile)


def _source_provenance(config: ReplayConfig) -> Mapping[str, Any]:
    if not config.irasim_repo.is_dir():
        raise FileNotFoundError("IRASim source checkout is absent: %s" % config.irasim_repo)
    if not config.vae_path.is_dir() or not config.scheduler_path.is_dir():
        raise FileNotFoundError("IRASim VAE/scheduler local directories are required.")
    source_report_record = _file_record(config.source_report)
    fixture_record, fixture_origin, sidecar_path = source_fixture_provenance(config.source_report)
    expected_fixture_hash = _normalised_sha256(fixture_record.get("sha256"), "source fixture_video.sha256")
    expected_fixture_bytes = fixture_record.get("bytes")
    if expected_fixture_bytes is not None and (
        isinstance(expected_fixture_bytes, bool)
        or not isinstance(expected_fixture_bytes, int)
        or config.fixture_video.stat().st_size != expected_fixture_bytes
    ):
        raise ReplayInputError("Local vendor fixture byte length differs from the saved source provenance.")
    actual_fixture_hash = _sha256(config.fixture_video)
    if actual_fixture_hash != expected_fixture_hash:
        raise ReplayInputError("Local vendor fixture SHA-256 differs from the saved closed-loop source report.")
    checkpoint = verify_irasim_checkpoint(
        config.irasim_checkpoint, config.irasim_checkpoint_sha256, config.irasim_conversion_report
    )
    frame_config = verify_irasim_config(config.irasim_config)
    pipeline_source = config.irasim_repo / "sample" / "pipeline_trajectory2videogen.py"
    autoregressive_source = config.irasim_repo / "sample" / "sample_autoregressive.py"
    return {
        "source_closed_loop_report": source_report_record,
        "source_provenance_sidecar": _file_record(sidecar_path) if sidecar_path is not None else None,
        "fixture_provenance_origin": fixture_origin,
        "source_run_id": SOURCE_RUN_ID,
        "fixture_video": _file_record(config.fixture_video),
        "fixture_hash_matches_source_report": True,
        "irasim_checkpoint": checkpoint,
        "irasim_config": frame_config,
        "irasim_source_revision": IRASIM_SOURCE_REVISION,
        "irasim_source_checkout": verify_irasim_source_checkout(config.irasim_repo),
        "irasim_pipeline_source": _file_record(pipeline_source),
        "irasim_autoregressive_source": _file_record(autoregressive_source),
        "source_contract": {
            "autoregressive_latent_carry": "sample/sample_autoregressive.py lines 239-252: start_image = seg_latents[-1].clone()",
            "pipeline_both_output": "sample/pipeline_trajectory2videogen.py: output_type='both' returns decoded videos and final latents",
            "mask_retention": "sample/pipeline_trajectory2videogen.py re-concats mask_x after each scheduler denoise step",
        },
        "vae_path": str(config.vae_path.resolve()),
        "scheduler_path": str(config.scheduler_path.resolve()),
        "source_release": config.source_release,
    }


def _run_once(
    config: ReplayConfig,
    inputs: Sequence[ReplayInput],
    adapter: Any,
    initial_frame: Any,
    run_dir: Path,
) -> Mapping[str, Any]:
    """Execute the paired intervention with no policy adapter in scope."""

    torch, pipeline = adapter._load()
    if int(adapter.profile.num_frames) != 16:
        raise ReplayInputError("IRASim model must remain instantiated with num_frames=16.")
    initial_seed = derive_stream_seed(inputs[0].recorded_seed, "initial_vae")
    initial_mask, initial_preprocessing = _encode_mask(adapter, torch, pipeline, initial_frame, initial_seed)
    image_mask = initial_mask.clone()
    latent_mask = initial_mask.clone()
    image_current = initial_frame
    latent_current = initial_mask.clone()
    branch_frames = {"image_reencode": [initial_frame], "latent_carry": [initial_frame]}
    branch_records = {"image_reencode": [], "latent_carry": []}
    initial_records = {}
    for branch, mask in (("image_reencode", image_mask), ("latent_carry", latent_mask)):
        initial_records[branch] = _branch_artifact(run_dir, branch, 0, initial_frame, mask, mask)
    expected_next_latent_mask_hash = tensor_hash(initial_mask)
    first_output_assertion: Optional[Mapping[str, bool]] = None
    started = time.perf_counter()
    for item in inputs:
        rng = generator_metadata(item.recorded_seed)
        if item.tick == 0:
            image_mask_for_step = image_mask
            image_preprocessing = dict(initial_preprocessing)
            image_preprocessing["initial_mask_shared_with_latent_branch"] = True
        else:
            image_mask_for_step, image_preprocessing = _encode_mask(
                adapter, torch, pipeline, image_current, rng["vae_seed"]
            )
        latent_mask_for_step = latent_current.clone()
        if tensor_hash(latent_mask_for_step) != expected_next_latent_mask_hash:
            raise ReplayInputError("Latent carry mask does not equal the prior step final latent tensor.")
        with torch.no_grad():
            image_video, image_latents = _pipeline_step(
                torch, adapter, pipeline, image_mask_for_step, item.action, rng["image_branch_diffusion_seed"]
            )
            latent_video, latent_latents = _pipeline_step(
                torch, adapter, pipeline, latent_mask_for_step, item.action, rng["latent_branch_diffusion_seed"]
            )
        image_next = _irasim_frame_to_rgb(image_video[0, 1])
        latent_next = _irasim_frame_to_rgb(latent_video[0, 1])
        image_final_latent = carry_last_latent(image_latents)
        latent_final_latent = carry_last_latent(latent_latents)
        image_final_hash = tensor_hash(image_final_latent)
        latent_final_hash = tensor_hash(latent_final_latent)
        if item.tick == 0:
            first_output_assertion = {
                "pixel_equal": raw_pixel_hash(image_next) == raw_pixel_hash(latent_next),
                "final_latent_equal": image_final_hash == latent_final_hash,
            }
            if not all(first_output_assertion.values()):
                raise ReplayInputError(
                    "Paired branches diverged at tick 0 despite identical initial latent and diffusion seed/draw contract."
                )
        image_current = image_next
        latent_current = latent_final_latent
        expected_next_latent_mask_hash = latent_final_hash
        branch_frames["image_reencode"].append(image_next)
        branch_frames["latent_carry"].append(latent_next)
        image_artifact = _branch_artifact(
            run_dir, "image_reencode", item.tick + 1, image_next, image_latents, image_final_latent
        )
        latent_artifact = _branch_artifact(
            run_dir, "latent_carry", item.tick + 1, latent_next, latent_latents, latent_final_latent
        )
        shared = {
            "tick": item.tick,
            "raw_native_action_7d": list(item.action),
            "recorded_seed": item.recorded_seed,
            "generator_streams": rng,
            "video_length": 2,
            "model_num_frames": 16,
            "future_action_rows": 0,
            "action_padding": False,
            "output_type": "both",
            "descriptive_output_pixel_mae": _pixel_mae(image_next, latent_next),
        }
        branch_records["image_reencode"].append(
            {
                **shared,
                "state_transport": "reencode_previous_generated_uint8_rgb",
                "mask_latent_sha256": tensor_hash(image_mask_for_step),
                "condition_preprocessing": image_preprocessing,
                "output": image_artifact,
            }
        )
        branch_records["latent_carry"].append(
            {
                **shared,
                "state_transport": "carry_previous_output_latents_final_slice_clone",
                "mask_latent_sha256": tensor_hash(latent_mask_for_step),
                "condition_preprocessing": {"vae_encode_skipped": item.tick > 0},
                "output": latent_artifact,
            }
        )
    elapsed = time.perf_counter() - started
    videos = {}
    for branch, frames in branch_frames.items():
        if len(frames) != REPLAY_TICKS + 1:
            raise ReplayInputError("%s branch did not retain 17 frames." % branch)
        video_path = run_dir / branch / (branch + ".mp4")
        # This helper only changes frame copies for MP4 presentation when the
        # vendor condition's size differs; raw frame artifacts remain intact.
        video_frames, video_metadata = _video_export_frames(frames)
        _write_mp4(video_path, video_frames)
        videos[branch] = {"mp4": _file_record(video_path), "frame_count": len(frames), "export": video_metadata}
    comparisons = [
        branch_records["image_reencode"][index]["output"]["raw_pixel_sha256"]
        == branch_records["latent_carry"][index]["output"]["raw_pixel_sha256"]
        for index in range(REPLAY_TICKS)
    ]
    return {
        "initial": {
            "shared_initial_vae_seed": initial_seed,
            "initial_latent_sha256": tensor_hash(initial_mask),
            "branches": initial_records,
        },
        "branches": {"image_reencode": branch_records["image_reencode"], "latent_carry": branch_records["latent_carry"]},
        "videos": videos,
        "paired_output_pixel_hash_equal_by_tick": comparisons,
        "first_output_equality_assertion": first_output_assertion,
        "timing": {"paired_replay_wall_seconds": elapsed},
    }


def run_replay(config: ReplayConfig) -> Tuple[int, Mapping[str, Any]]:
    """Preflight, run, and persist a durable unqualified diagnostic report."""

    # This preflight intentionally happens before ``mkdir`` so an existing
    # target cannot be overwritten even with a failure report.
    config.__post_init__()
    started = time.perf_counter()
    report_path = config.output_dir / "report.json"
    # Existing output targets are rejected above.  Once a fresh directory is
    # claimed, even setup/load failures receive a durable report inside it.
    config.output_dir.mkdir(parents=True, exist_ok=False)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": REPLAY_KIND,
        "qualification": QUALIFICATION,
        "qualified": False,
        "physical_state_measured": False,
        "policy_requery": False,
        "success_scoring": False,
        "binary_success": None,
        "progress_score": None,
        "missing_reason": "unqualified_state_representation_intervention_no_policy_or_success_score",
        "started_at": _utc_now(),
        "protocol_difference_from_source_closed_loop": (
            "Uses explicitly derived independent VAE and diffusion generators; source closed-loop reused one generator "
            "for VAE posterior sampling and diffusion."
        ),
        "offline_environment": {"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE")},
    }
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise ReplayInputError("Replay must run offline with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.")
        inputs = load_replay_inputs(config.source_report)
        provenance = _source_provenance(config)
        initial_frame = _load_vendor_frame_zero(config.fixture_video)
        adapter = _load_adapter(config)
        outcome = _run_once(config, inputs, adapter, initial_frame, config.output_dir)
        report.update(
            {
                "status": "completed_unqualified_intervention",
                "shared_replay_inputs": {
                    "source_report": _file_record(config.source_report),
                    "actions": [list(item.action) for item in inputs],
                    "recorded_seeds": [item.recorded_seed for item in inputs],
                    "ticks": REPLAY_TICKS,
                    "no_future_actions": True,
                    "no_action_padding": True,
                    "no_weight_change": True,
                },
                "provenance": provenance,
                "outcome": outcome,
            }
        )
        if config.same_seed_repeat:
            repeat = _run_once(config, inputs, adapter, initial_frame, config.output_dir / "same-seed-repeat")
            report["same_seed_repeat"] = {
                "requested": True,
                "replay": repeat,
                "note": "Repeat artifact hashes are descriptive determinism evidence only, not fidelity evidence.",
            }
        code = 0
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=10)},
            }
        )
        code = 2
    report["finished_at"] = _utc_now()
    report["timing"] = {"total_seconds": time.perf_counter() - started}
    _atomic_json(report_path, report)
    return code, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PLUMB_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--irasim-repo", type=Path)
    parser.add_argument("--irasim-checkpoint", type=Path)
    parser.add_argument("--irasim-checkpoint-sha256")
    parser.add_argument("--irasim-conversion-report", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--scheduler", type=Path)
    parser.add_argument("--irasim-config", type=Path)
    parser.add_argument("--fixture-video", type=Path)
    parser.add_argument("--source-release", default=os.environ.get("PLUMB_SOURCE_RELEASE"))
    parser.add_argument("--same-seed-repeat", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ReplayConfig:
    root = Path(args.root)
    return ReplayConfig(
        root=root,
        source_report=args.source_report or root / "evidence" / SOURCE_RUN_ID / "report.json",
        output_dir=Path(args.output_dir),
        irasim_repo=args.irasim_repo or root / "irasim-source",
        irasim_checkpoint=args.irasim_checkpoint or root / "models" / "irasim" / "frame_ada_0300000.safetensors",
        irasim_checkpoint_sha256=args.irasim_checkpoint_sha256 or IRASIM_SAFE_TENSOR_SHA256,
        irasim_conversion_report=args.irasim_conversion_report,
        vae_path=args.vae or root / "models" / "stabilityai--stable-diffusion-xl-base-1.0",
        scheduler_path=args.scheduler or root / "irasim-source" / "pretrained_models" / "scheduler",
        irasim_config=args.irasim_config or root / "irasim-source" / "configs" / "evaluation" / "bridge" / "frame_ada.yaml",
        fixture_video=args.fixture_video or root / "fixtures" / "bridge_video.mp4",
        source_release=args.source_release,
        same_seed_repeat=bool(args.same_seed_repeat),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    code, report = run_replay(config)
    print(json.dumps(_jsonable(report), sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
