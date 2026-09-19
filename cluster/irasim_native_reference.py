#!/usr/bin/env python3
"""Unqualified released-horizon IRASim open-loop reference diagnostic.

This is a single original-architecture IRASim call with fifteen saved raw
Bridge 7-D actions and sixteen returned frames.  It uses the same vendor start
and same *initial VAE* seed as the paired one-step state replay, but the
diffusion noise tensor has a different 16-frame shape.  Therefore it is a
qualitative reference only: it cannot support a causal paired comparison with
the one-step branches, a native OpenVLA-control claim, a success score, or any
qualification result.

No policy module is imported or queried.  Actions/seeds come only from the
saved ``closed-loop-937704`` record, and only its first fifteen raw actions are
used.  This is an open-loop full horizon: fourteen supplied rows are future
relative to the first prediction.  No *extra* padding/action suffix is added.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    from cluster import irasim_state_replay as state_replay
    from cluster.closed_loop_smoke import _irasim_frame_to_rgb, _load_vendor_frame_zero, _video_export_frames, _write_mp4, _write_png
except ModuleNotFoundError:  # Direct ``python cluster/irasim_native_reference.py`` execution.
    import irasim_state_replay as state_replay  # type: ignore
    from closed_loop_smoke import _irasim_frame_to_rgb, _load_vendor_frame_zero, _video_export_frames, _write_mp4, _write_png  # type: ignore


NATIVE_ACTION_COUNT = 15
NATIVE_FRAME_COUNT = 16
REFERENCE_KIND = "plumb_irasim_native_open_loop_reference_diagnostic"
QUALIFICATION = "experimental_unqualified_released_horizon_reference_not_gate_b"


class NativeReferenceInputError(ValueError):
    """The bounded native-reference preflight contract was not met."""


@dataclass(frozen=True)
class NativeReferenceConfig:
    replay: state_replay.ReplayConfig

    @property
    def output_dir(self) -> Path:
        return self.replay.output_dir


def _native_actions(torch: Any, adapter: Any, inputs: Sequence[state_replay.ReplayInput]) -> Any:
    if len(inputs) < NATIVE_ACTION_COUNT:
        raise NativeReferenceInputError("Source record must provide at least fifteen saved raw actions.")
    rows = []
    for item in inputs[:NATIVE_ACTION_COUNT]:
        if len(item.action) != 7:
            raise NativeReferenceInputError("Saved source action is not 7-D.")
        rows.append(
            [
                float(item.action[index]) * (20.0 if index < 6 else 1.0)
                for index in range(7)
            ]
        )
    actions = torch.tensor(rows, device=adapter.profile.device, dtype=torch.float32).view(1, NATIVE_ACTION_COUNT, 7)
    if tuple(int(value) for value in actions.shape) != (1, NATIVE_ACTION_COUNT, 7):
        raise NativeReferenceInputError("Released-horizon native IRASim action tensor must be [1,15,7].")
    return actions


def _native_pipeline_call(
    torch: Any, adapter: Any, pipeline: Any, mask_x: Any, inputs: Sequence[state_replay.ReplayInput], diffusion_seed: int
) -> Tuple[Any, Any]:
    if tuple(int(value) for value in mask_x.shape) != (1, 1, 4, 32, 40):
        raise NativeReferenceInputError("Initial IRASim mask_x must be [1,1,4,32,40].")
    state_replay._require_finite(torch, mask_x, "IRASim native-reference mask_x")
    videos, latents = pipeline(
        _native_actions(torch, adapter, inputs),
        mask_x=mask_x,
        video_length=NATIVE_FRAME_COUNT,
        height=adapter.profile.input_height,
        width=adapter.profile.input_width,
        num_inference_steps=adapter.profile.inference_steps,
        guidance_scale=adapter.profile.guidance_scale,
        generator=state_replay._torch_generator(torch, adapter.profile.device, diffusion_seed),
        output_type="both",
        return_dict=False,
        device=torch.device(adapter.profile.device),
    )
    if videos is None or latents is None:
        raise NativeReferenceInputError("IRASim output_type='both' must return both video and latents.")
    if tuple(int(value) for value in latents.shape) != (1, NATIVE_FRAME_COUNT, 4, 32, 40):
        raise NativeReferenceInputError("Released-horizon IRASim latents must be [1,16,4,32,40].")
    state_replay._require_finite(torch, latents, "IRASim native-reference output latents")
    if not bool(torch.equal(latents[:, :1], mask_x)):
        raise NativeReferenceInputError("IRASim native reference did not retain mask_x at frame zero.")
    if int(videos.shape[1]) != NATIVE_FRAME_COUNT:
        raise NativeReferenceInputError("Released-horizon IRASim must return exactly 16 video frames.")
    return videos, latents


def _frame_artifact(output_dir: Path, index: int, frame: Any) -> Mapping[str, Any]:
    path = output_dir / "frames" / ("frame-%03d.png" % index)
    _write_png(path, frame)
    return {
        "frame_index": index,
        "raw_pixel_sha256": state_replay.raw_pixel_hash(frame),
        "png": state_replay._file_record(path),
    }


def _native_final_latent(latents: Any) -> Any:
    """Hash the released-horizon final slice without invoking one-step carry helpers."""

    if tuple(int(value) for value in latents.shape) != (1, NATIVE_FRAME_COUNT, 4, 32, 40):
        raise NativeReferenceInputError("Released-horizon output latents have an unexpected shape.")
    final = latents[:, -1:]
    clone = getattr(final, "clone", None)
    if not callable(clone):
        raise NativeReferenceInputError("Released-horizon final latent slice must expose clone().")
    return clone()


def run_native_reference(
    config: NativeReferenceConfig,
    *,
    adapter_loader: Any = state_replay._load_adapter,
    frame_loader: Any = _load_vendor_frame_zero,
    provenance_loader: Any = state_replay._source_provenance,
) -> Tuple[int, Mapping[str, Any]]:
    """Run exactly one 15-action/16-frame direct pipeline diagnostic."""

    replay_config = config.replay
    replay_config.__post_init__()  # Reject an existing output before any writes.
    started = time.perf_counter()
    report_path = config.output_dir / "report.json"
    config.output_dir.mkdir(parents=True, exist_ok=False)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": REFERENCE_KIND,
        "qualification": QUALIFICATION,
        "qualified": False,
        "policy_requery": False,
        "native_openvla_control_claim": False,
        "success_scoring": False,
        "binary_success": None,
        "progress_score": None,
        "validity": "unknown",
        "missing_reason": "unqualified_released_horizon_reference_no_policy_or_success_score",
        "started_at": state_replay._utc_now(),
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise NativeReferenceInputError("Native reference must run offline with both Hugging Face offline flags set.")
        inputs = state_replay.load_replay_inputs(replay_config.source_report)
        if len(inputs) != state_replay.REPLAY_TICKS:
            raise NativeReferenceInputError("Saved closed-loop source must contain exactly sixteen rows before taking the first fifteen.")
        provenance = provenance_loader(replay_config)
        initial_frame = frame_loader(replay_config.fixture_video)
        adapter = adapter_loader(replay_config)
        torch, pipeline = adapter._load()
        if int(adapter.profile.num_frames) != 16 or int(adapter.profile.extras) != 3 or int(adapter.profile.mask_frame_num) != 1:
            raise NativeReferenceInputError("IRASim adapter profile must retain original 16/3/1 architecture.")
        initial_vae_seed = state_replay.derive_stream_seed(inputs[0].recorded_seed, "initial_vae")
        mask_x, initial_preprocessing = state_replay._encode_mask(
            adapter, torch, pipeline, initial_frame, initial_vae_seed
        )
        diffusion_seed = state_replay.derive_stream_seed(inputs[0].recorded_seed, "diffusion")
        with torch.no_grad():
            videos, latents = _native_pipeline_call(torch, adapter, pipeline, mask_x, inputs, diffusion_seed)
        frames = tuple(_irasim_frame_to_rgb(videos[0, index]) for index in range(NATIVE_FRAME_COUNT))
        artifacts = [_frame_artifact(config.output_dir, index, frame) for index, frame in enumerate(frames)]
        mp4_frames, export = _video_export_frames(frames)
        mp4_path = config.output_dir / "irasim-native-reference.mp4"
        _write_mp4(mp4_path, mp4_frames)
        report.update(
            {
                "status": "completed_unqualified_reference",
                "provenance": provenance,
                "shared_source_inputs": {
                    "source_closed_loop_report": state_replay._file_record(replay_config.source_report),
                    "first_fifteen_raw_native_action_7d": [list(item.action) for item in inputs[:NATIVE_ACTION_COUNT]],
                    "source_recorded_seeds": [item.recorded_seed for item in inputs],
                    "initial_vendor_frame_sha256": state_replay.raw_pixel_hash(initial_frame),
                },
                "rng": {
                    "initial_vae_seed": initial_vae_seed,
                    "diffusion_seed": diffusion_seed,
                    "initial_vae_seed_matches_state_replay": True,
                    "noise_tensor_shape": [1, NATIVE_FRAME_COUNT, 4, 32, 40],
                    "causal_pairing_claim": False,
                    "difference_from_one_step_replay": "16-frame diffusion noise tensor size differs from one-step video_length=2 calls.",
                },
                "native_contract": {
                    "actions_shape": [1, NATIVE_ACTION_COUNT, 7],
                    "action_scaling": [20, 20, 20, 20, 20, 20, 1],
                    "video_length": NATIVE_FRAME_COUNT,
                    "model_num_frames": 16,
                    "extras": 3,
                    "mask_frame_num": 1,
                    "output_type": "both",
                    "supplied_action_rows": NATIVE_ACTION_COUNT,
                    "extra_padding_rows": 0,
                    "future_rows_relative_to_first_prediction": NATIVE_ACTION_COUNT - 1,
                    "open_loop_full_horizon": True,
                    "causal_feedback_claim": False,
                },
                "artifacts": {
                    "frames": artifacts,
                    "mp4": state_replay._file_record(mp4_path),
                    "mp4_frame_count": len(frames),
                    "video_export": export,
                    "mask_latent_tensor_sha256": state_replay.tensor_hash(mask_x),
                    "output_latent_tensor_sha256": state_replay.tensor_hash(latents),
                    "final_latent_tensor_sha256": state_replay.tensor_hash(_native_final_latent(latents)),
                },
            }
        )
        code = 0
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=10)},
            }
        )
        code = 2
    report["finished_at"] = state_replay._utc_now()
    report["timing"] = {"total_seconds": time.perf_counter() - started}
    state_replay._atomic_json(report_path, report)
    return code, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PLUMB_ROOT", state_replay.DEFAULT_ROOT)))
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
    return parser


def config_from_args(args: argparse.Namespace) -> NativeReferenceConfig:
    root = Path(args.root)
    replay = state_replay.ReplayConfig(
        root=root,
        source_report=args.source_report or root / "evidence" / state_replay.SOURCE_RUN_ID / "report.json",
        output_dir=Path(args.output_dir),
        irasim_repo=args.irasim_repo or root / "irasim-source",
        irasim_checkpoint=args.irasim_checkpoint or root / "models" / "irasim" / "frame_ada_0300000.safetensors",
        irasim_checkpoint_sha256=args.irasim_checkpoint_sha256 or state_replay.IRASIM_SAFE_TENSOR_SHA256,
        irasim_conversion_report=args.irasim_conversion_report,
        vae_path=args.vae or root / "models" / "stabilityai--stable-diffusion-xl-base-1.0",
        scheduler_path=args.scheduler or root / "irasim-source" / "pretrained_models" / "scheduler",
        irasim_config=args.irasim_config or root / "irasim-source" / "configs" / "evaluation" / "bridge" / "frame_ada.yaml",
        fixture_video=args.fixture_video or root / "fixtures" / "bridge_video.mp4",
        source_release=args.source_release,
    )
    return NativeReferenceConfig(replay)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    code, report = run_native_reference(config_from_args(args))
    print(json.dumps(state_replay._jsonable(report), sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
