"""Experimental original-IRASim Bridge one-step runtime.

This is a source-pinned, unqualified adapter for a narrow capability probe. It
does not make IRASim a Cosmos-compatible fallback and does not alter the
released 16-frame model/checkpoint architecture.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

from .contracts import CapabilityResult, CapabilityStatus, IRASIM_COMMIT, ServerTiming

IRASIM_SOURCE = "https://github.com/bytedance/IRASim/tree/" + IRASIM_COMMIT
IRASIM_FRAME_CONFIG = IRASIM_SOURCE + "/configs/evaluation/bridge/frame_ada.yaml"
IRASIM_PIPELINE_SOURCE = IRASIM_SOURCE + "/sample/pipeline_trajectory2videogen.py"
IRASIM_ACTION_SCALE = (20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0)


class IRASimRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class IRASimOneStepProfile:
    profile_id: str
    repo_path: str
    checkpoint_path: str
    vae_model_path: str
    scheduler_path: str
    config_path: str
    device: str = "cuda:0"
    num_frames: int = 16
    extras: int = 3
    mask_frame_num: int = 1
    input_height: int = 256
    input_width: int = 320
    inference_steps: int = 50
    guidance_scale: float = 1.0

    def __post_init__(self) -> None:
        if (self.num_frames, self.extras, self.mask_frame_num) != (16, 3, 1):
            raise ValueError("Original Bridge frame_ada checkpoints require unchanged num_frames=16, extras=3, mask_frame_num=1.")


@dataclass(frozen=True)
class IRASimOneStepResult:
    frames: Tuple[Any, ...]
    timing: ServerTiming
    native_action_scaled: Tuple[float, ...]
    capability: CapabilityResult
    condition_preprocessing: dict


class IRASimOneStepAdapter:
    """Use a 16-frame original model with a 1-action/2-frame inference horizon.

    The model itself remains instantiated with the checkpoint's 16-frame
    temporal embedding. The pipeline receives ``video_length=2`` only; model
    forward slices ``temp_embed[:, :frames]`` as released. No action padding,
    state fabrication, extras=5 conversion, or two-frame model construction is
    allowed.
    """

    def __init__(self, profile: IRASimOneStepProfile, loader: Optional[Callable[[IRASimOneStepProfile], Tuple[Any, Any]]] = None) -> None:
        self.profile = profile
        self._loader = loader
        self._runtime: Any = None
        self._pipeline: Any = None
        self._calls = 0

    def capability(self) -> CapabilityResult:
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED if self._loader or Path(self.profile.repo_path).is_dir() else CapabilityStatus.UNAVAILABLE,
            reason=("Experimental original IRASim one-step probe only; 16-frame checkpoint remains unchanged and this cannot qualify a policy/backend cell."),
            source_verified=True,
            evidence_uris=(IRASIM_FRAME_CONFIG, IRASIM_PIPELINE_SOURCE),
            details={"native_action_shape": [1, 7], "video_length": 2, "model_num_frames": 16, "extras": 3, "mask_frame_num": 1},
        )

    def _load(self) -> Tuple[Any, Any]:
        if self._runtime is not None:
            return self._runtime, self._pipeline
        if self._loader is not None:
            self._runtime, self._pipeline = self._loader(self.profile)
            return self._runtime, self._pipeline
        try:
            import torch
            from omegaconf import OmegaConf
            from diffusers.models import AutoencoderKL
            from diffusers.schedulers import PNDMScheduler
        except ImportError as error:
            raise IRASimRuntimeError("IRASim dependencies are absent; imports are intentionally lazy.") from error
        repo = Path(self.profile.repo_path)
        for required in (repo, Path(self.profile.checkpoint_path), Path(self.profile.vae_model_path), Path(self.profile.scheduler_path), Path(self.profile.config_path)):
            if not required.exists():
                raise IRASimRuntimeError("Required local IRASim artifact is absent: %s" % required)
        sys.path.insert(0, str(repo))
        from models import get_models
        from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
        args = OmegaConf.merge(
            OmegaConf.load(repo / "configs/base/data.yaml"),
            OmegaConf.load(repo / "configs/base/diffusion.yaml"),
            OmegaConf.load(self.profile.config_path),
        )
        if (int(args.num_frames), int(args.extras), int(args.mask_frame_num)) != (16, 3, 1):
            raise IRASimRuntimeError("Local config does not match original Bridge frame_ada 16/3/1 contract.")
        args.latent_size = [int(value) // 8 for value in args.video_size]
        model = get_models(args)
        if Path(self.profile.checkpoint_path).suffix == ".safetensors":
            from safetensors.torch import load_file
            checkpoint = load_file(self.profile.checkpoint_path, device="cpu")
        else:
            try:
                checkpoint = torch.load(self.profile.checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError as error:
                raise IRASimRuntimeError("This runtime requires torch.load(weights_only=True); do not unpickle IRASim checkpoints.") from error
        state = checkpoint["ema"] if isinstance(checkpoint, dict) and "ema" in checkpoint else checkpoint
        if not isinstance(state, dict):
            raise IRASimRuntimeError("IRASim checkpoint did not contain a safe state dict.")
        model.load_state_dict(state, strict=True)
        vae = AutoencoderKL.from_pretrained(self.profile.vae_model_path, subfolder="vae", local_files_only=True)
        scheduler = PNDMScheduler.from_pretrained(self.profile.scheduler_path, local_files_only=True)
        device = torch.device(self.profile.device)
        model.to(device).eval()
        vae.to(device).eval().requires_grad_(False)
        self._runtime = torch
        self._pipeline = Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)
        return self._runtime, self._pipeline

    def _condition(self, torch: Any, image: Any) -> Tuple[Any, dict]:
        tensor = torch.as_tensor(image)
        if tensor.ndim == 3 and tensor.shape[-1] in (3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError("IRASim Bridge condition must be RGB CHW/HWC, got %s." % (tuple(tensor.shape),))
        original_size = [int(tensor.shape[1]), int(tensor.shape[2])]
        tensor = tensor.to(self.profile.device, dtype=torch.float32)
        resized = original_size != [self.profile.input_height, self.profile.input_width]
        if resized:
            import torch.nn.functional as F
            tensor = F.interpolate(tensor.unsqueeze(0), size=(self.profile.input_height, self.profile.input_width),
                                   mode="bilinear", align_corners=False, antialias=True).squeeze(0)
        if tensor.max() > 1:
            tensor = tensor / 127.5 - 1.0
        elif tensor.min() >= 0:
            tensor = tensor * 2.0 - 1.0
        return tensor, {"original_size_hw": original_size, "resized_to_hw": [self.profile.input_height, self.profile.input_width],
                        "resize_applied": resized, "interpolation": "bilinear_antialias" if resized else None,
                        "range_after_normalization": "[-1,1]"}

    def generate_one_step(self, image: Any, action: Sequence[float], seed: int = 0) -> IRASimOneStepResult:
        if len(action) != 7 or any(not math.isfinite(float(value)) for value in action):
            raise ValueError("IRASim native one-step action must be finite 7-D.")
        cold_start = self._pipeline is None
        load_started = time.perf_counter()
        torch, pipeline = self._load()
        device = torch.device(self.profile.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_started
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        native = tuple(float(value) for value in action)
        scaled = tuple(value * IRASIM_ACTION_SCALE[index] for index, value in enumerate(native))
        condition, preprocessing = self._condition(torch, image)
        # 2-D VAE: encode the one condition image to [B,1,C,H,W] mask_x.
        vae = pipeline.vae
        with torch.no_grad():
            generator = torch.Generator(device=self.profile.device).manual_seed(int(seed))
            # Use this generator for the posterior draw *and* diffusion noise;
            # otherwise same-seed runs have nondeterministic condition latents.
            latent = vae.encode(condition.unsqueeze(0)).latent_dist.sample(generator=generator).mul_(vae.config.scaling_factor)
            mask_x = latent.unsqueeze(1)
            actions = torch.tensor(scaled, device=self.profile.device, dtype=torch.float32).view(1, 1, 7)
            videos, _ = pipeline(actions, mask_x=mask_x, video_length=2, height=self.profile.input_height,
                                 width=self.profile.input_width, num_inference_steps=self.profile.inference_steps,
                                 guidance_scale=self.profile.guidance_scale, generator=generator, output_type="both",
                                 return_dict=False, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        self._calls += 1
        frames = tuple(videos[0])
        if len(frames) != 2:
            raise IRASimRuntimeError("Original IRASim one-step call returned %d frames; expected exactly 2." % len(frames))
        preprocessing["seed"] = int(seed)
        preprocessing["latent_sampling_generator"] = "same_generator_as_denoising"
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        return IRASimOneStepResult(frames, ServerTiming(1, elapsed, cold_start, gpu_peak_memory_bytes=peak, model_load_seconds=load_seconds), scaled, self.capability(), preprocessing)
