"""World-model adapters with explicit backend-specific action contracts."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from .contracts import (
    BackendProfile,
    CapabilityResult,
    CapabilityStatus,
    COSMOS3_DIFFUSERS_SOURCE,
    FeedbackMode,
    IRASIM_COMMIT,
    ServerTiming,
    WorldRequest,
    WorldResult,
)


DIFFUSERS_COSMOS3_PIN = "a3e0b8ec235c27a6c17a21976daf7fd32d819d05"
COSMOS3_NANO_MODEL_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
COSMOS3_VENDOR_FIXTURE = (
    "https://raw.githubusercontent.com/NVIDIA/cosmos-framework/"
    "c23e51f2f157ae3e51cfcd86ebfb5464850894f2/inputs/omni/action_forward_dynamics_robot.json"
)
COSMOS3_BRIDGE_FIXTURE_ASSET_COMMIT = "2b17a2413bd86b2cf9b03823637108851e4ddf2d"
COSMOS3_BRIDGE_FIXTURE_ACTION_URL = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/"
    + COSMOS3_BRIDGE_FIXTURE_ASSET_COMMIT
    + "/inputs/action/bridge_20260501_0.json"
)
COSMOS3_BRIDGE_FIXTURE_VIDEO_URL = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/"
    + COSMOS3_BRIDGE_FIXTURE_ASSET_COMMIT
    + "/inputs/action/bridge_20260501_0.mp4"
)
# Hash of the small JSON action fixture, not the model checkpoint or MP4.
COSMOS3_BRIDGE_FIXTURE_ACTION_SHA256 = "5c26b3cb84799812a70b534ad939551d2ac308fdc870ea0e66163bb52c9d61da"


class BackendContractError(ValueError):
    """The request is not valid for the selected native backend."""


class BackendUnavailableError(RuntimeError):
    """A local model/dependency/loader required for a real call is missing."""


@dataclass(frozen=True)
class Cosmos3NanoDiffusersProfile:
    """Profile for one exact official Diffusers forward-dynamics serializer.

    It deliberately defaults to the local-only installation root.  The model
    identifier is metadata, not a signal to fetch 35 GB when a runtime starts.
    ``probe_action_lengths`` exists solely for Gate-B causal probes; it cannot
    turn a one-tick call into a qualified feedback implementation.
    """

    profile_id: str
    local_model_path: str
    model_revision: str = COSMOS3_NANO_MODEL_REVISION
    diffusers_revision: str = DIFFUSERS_COSMOS3_PIN
    container_digest: Optional[str] = None
    normalizer_revision: Optional[str] = None
    resolution_tier: int = 256
    fps: float = 5.0
    num_inference_steps: int = 30
    guidance_scale: float = 1.0
    scheduler_flow_shift: Optional[float] = 10.0
    device_map: str = "cuda"
    view_point: str = "ego_view"
    allowed_action_lengths: Tuple[int, ...] = (16,)
    probe_action_lengths: Tuple[int, ...] = (1,)
    local_files_only: bool = True
    use_system_prompt: bool = False
    enable_safety_checker: bool = False

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("Cosmos profile_id is required.")
        if not self.local_model_path:
            raise ValueError("Cosmos local_model_path is required; Hub fetching is intentionally disabled.")
        if self.resolution_tier not in (256, 480, 704, 720):
            raise ValueError("Cosmos action resolution_tier must be 256, 480, 704, or 720.")
        if self.fps <= 0 or self.num_inference_steps < 1:
            raise ValueError("Cosmos fps and num_inference_steps must be positive.")
        if not set(self.allowed_action_lengths).isdisjoint(set(self.probe_action_lengths)):
            raise ValueError("Normal and probe Cosmos action lengths must not overlap.")

    def as_backend_profile(self) -> BackendProfile:
        return BackendProfile(
            profile_id=self.profile_id,
            backend="cosmos3_diffusers",
            model_id="nvidia/Cosmos3-Nano",
            model_revision=self.model_revision,
            code_revision=self.diffusers_revision,
            container_digest=self.container_digest,
            normalizer_revision=self.normalizer_revision,
            source_uri=COSMOS3_DIFFUSERS_SOURCE,
            local_model_path=self.local_model_path,
            local_files_only=self.local_files_only,
            metadata={
                "mode": "forward_dynamics",
                "domain_name": "bridge_orig_lerobot",
                "resolution_tier": self.resolution_tier,
                "fps": self.fps,
                "vendor_fixture": COSMOS3_VENDOR_FIXTURE,
                "vendor_fixture_action": {
                    "uri": COSMOS3_BRIDGE_FIXTURE_ACTION_URL,
                    "sha256": "sha256:" + COSMOS3_BRIDGE_FIXTURE_ACTION_SHA256,
                    "shape": [16, 10],
                },
                "vendor_fixture_video_uri": COSMOS3_BRIDGE_FIXTURE_VIDEO_URL,
                "frame_contract": "N actions -> N+1 returned frames, condition at index 0",
            },
        )


@dataclass(frozen=True)
class _CosmosRuntime:
    torch: Any
    pipeline_cls: Any
    action_condition_cls: Any
    scheduler_cls: Any = None


class Cosmos3NanoDiffusersAdapter:
    """Official ``Cosmos3OmniPipeline`` FD wrapper, imported only on use.

    A pipeline can be injected for fixture tests, but production code calls the
    public Diffusers API directly.  The adapter never downloads weights and
    never pads/repeats policy actions.  Diffusers itself can repeat a short
    sequence when a caller chooses a larger chunk size, so this wrapper sets
    ``chunk_size == len(compiled_actions)`` and records the action length.
    """

    def __init__(
        self,
        profile: Cosmos3NanoDiffusersProfile,
        *,
        pipeline_factory: Optional[Callable[[Cosmos3NanoDiffusersProfile, _CosmosRuntime], Any]] = None,
        runtime_factory: Optional[Callable[[], _CosmosRuntime]] = None,
    ) -> None:
        self.profile = profile
        self._pipeline_factory = pipeline_factory
        self._runtime_factory = runtime_factory
        self._runtime: Optional[_CosmosRuntime] = None
        self._pipeline: Any = None
        self._real_backend_calls = 0
        self._backend_call_attempts = 0

    @property
    def backend_call_attempts(self) -> int:
        """Actual pipeline invocation attempts, including calls that raise."""

        return self._backend_call_attempts

    @property
    def successful_backend_calls(self) -> int:
        """Calls that returned a frame result passing the adapter frame contract."""

        return self._real_backend_calls

    @property
    def backend_profile(self) -> BackendProfile:
        return self.profile.as_backend_profile()

    def capability(self) -> CapabilityResult:
        model_exists = Path(self.profile.local_model_path).is_dir()
        if not model_exists and self._pipeline_factory is None:
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="Cosmos3-Nano local model directory is not present; no Hub download was attempted.",
                source_verified=True,
                evidence_uris=(COSMOS3_DIFFUSERS_SOURCE, COSMOS3_VENDOR_FIXTURE),
                details={"local_model_path": self.profile.local_model_path, "local_files_only": True},
            )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Official Diffusers forward-dynamics call is configured. Vendor and Bridge fixtures plus "
                "actual GPU shape/frame/latency evidence are still required for Gate A; this result is not qualification."
            ),
            source_verified=True,
            evidence_uris=(COSMOS3_DIFFUSERS_SOURCE, COSMOS3_VENDOR_FIXTURE),
            details={
                "model_revision": self.profile.model_revision,
                "diffusers_revision": self.profile.diffusers_revision,
                "allowed_action_lengths": self.profile.allowed_action_lengths,
                "probe_action_lengths": self.profile.probe_action_lengths,
                "expected_frame_rule": "N actions -> N+1 returned frames",
                "safety_checker": self.profile.enable_safety_checker,
            },
        )

    def _load_runtime(self) -> _CosmosRuntime:
        if self._runtime is not None:
            return self._runtime
        if self._runtime_factory is not None:
            self._runtime = self._runtime_factory()
            return self._runtime
        try:
            import torch  # type: ignore
            from diffusers import Cosmos3OmniPipeline, UniPCMultistepScheduler  # type: ignore
            from diffusers import CosmosActionCondition  # type: ignore
        except ImportError as error:
            raise BackendUnavailableError(
                "Cosmos3 Diffusers runtime is unavailable. Install the pinned Diffusers source, torch, "
                "transformers and media dependencies on the GPU worker; imports remain lazy by design."
            ) from error
        self._runtime = _CosmosRuntime(
            torch=torch,
            pipeline_cls=Cosmos3OmniPipeline,
            action_condition_cls=CosmosActionCondition,
            scheduler_cls=UniPCMultistepScheduler,
        )
        return self._runtime

    def _ensure_pipeline(self, runtime: _CosmosRuntime) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_factory is not None:
            self._pipeline = self._pipeline_factory(self.profile, runtime)
            return self._pipeline
        path = Path(self.profile.local_model_path)
        if not path.is_dir():
            raise BackendUnavailableError(
                "Expected local Cosmos3-Nano checkpoint directory at %s; refusing a remote model download."
                % path
            )
        # This is the current official constructor.  ``dtype`` (not top-level
        # torch_dtype) is intentionally aligned with the pinned Diffusers docs.
        pipeline = runtime.pipeline_cls.from_pretrained(
            str(path),
            dtype=runtime.torch.bfloat16,
            device_map=self.profile.device_map,
            local_files_only=True,
            enable_safety_checker=self.profile.enable_safety_checker,
        )
        if self.profile.scheduler_flow_shift is not None and runtime.scheduler_cls is not None:
            pipeline.scheduler = runtime.scheduler_cls.from_config(
                pipeline.scheduler.config,
                flow_shift=self.profile.scheduler_flow_shift,
                use_karras_sigmas=False,
            )
        self._pipeline = pipeline
        return self._pipeline

    def _check_request(self, request: WorldRequest) -> str:
        request.validate_basic()
        if request.compatibility_profile_id != self.profile.profile_id:
            raise BackendContractError("WorldRequest profile ID does not match the Cosmos adapter profile.")
        if request.domain != "bridge_orig_lerobot":
            raise BackendContractError(
                "This Bridge FD profile only accepts domain='bridge_orig_lerobot', not %r." % request.domain
            )
        action_count = len(request.compiled_actions)
        if action_count not in self.profile.allowed_action_lengths + self.profile.probe_action_lengths:
            raise BackendContractError(
                "Cosmos request has %d actions; profile permits normal lengths %s and Gate-B probe lengths %s."
                % (action_count, self.profile.allowed_action_lengths, self.profile.probe_action_lengths)
            )
        for action in request.compiled_actions:
            if len(action) != 10:
                raise BackendContractError(
                    "Cosmos bridge_orig_lerobot forward dynamics requires 10-D rows; do not send IRASim 7-D actions."
                )
        return "probe_unqualified" if action_count in self.profile.probe_action_lengths else "configured_unqualified"

    @staticmethod
    def _frames_from_result(result: Any) -> Tuple[Any, ...]:
        video = getattr(result, "video", None)
        if video is None and isinstance(result, Mapping):
            video = result["video"] if "video" in result else result.get("videos")
        if video is None:
            raise BackendContractError("Cosmos pipeline result did not contain video frames.")
        return tuple(video)

    @staticmethod
    def _peak_memory(runtime: _CosmosRuntime) -> Optional[int]:
        cuda = getattr(runtime.torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                return int(cuda.max_memory_allocated())
        except (AttributeError, RuntimeError):
            return None
        return None

    def generate(self, request: WorldRequest) -> WorldResult:
        action_length_status = self._check_request(request)
        runtime = self._load_runtime()
        cold_start = self._pipeline is None
        load_start = time.perf_counter()
        pipeline = self._ensure_pipeline(runtime)
        model_load_seconds = time.perf_counter() - load_start

        # Peak memory is deliberately sampled only around a genuine backend
        # invocation.  It remains null if the runtime cannot report it.
        cuda = getattr(runtime.torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                cuda.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            pass

        raw_actions = runtime.torch.as_tensor(request.compiled_actions, dtype=runtime.torch.float32)
        action_condition = runtime.action_condition_cls(
            mode="forward_dynamics",
            chunk_size=len(request.compiled_actions),
            domain_name=request.domain,
            resolution_tier=self.profile.resolution_tier,
            raw_actions=raw_actions,
            image=request.conditioning_image,
            view_point=self.profile.view_point,
        )
        generator = None
        try:
            generator = runtime.torch.Generator(device="cuda").manual_seed(int(request.seed))
        except (AttributeError, RuntimeError, TypeError):
            # Some CPU fixture runtimes only support the device-less constructor.
            generator = runtime.torch.Generator().manual_seed(int(request.seed))

        started_unix = time.time()
        start = time.perf_counter()
        self._backend_call_attempts += 1
        try:
            # In action mode no top-level image/video/height/width/num_frames is
            # passed. Diffusers derives N+1 frames from CosmosActionCondition.
            result = pipeline(
                prompt=request.prompt,
                action=action_condition,
                fps=self.profile.fps,
                num_inference_steps=self.profile.num_inference_steps,
                guidance_scale=self.profile.guidance_scale,
                generator=generator,
                output_type="np",
                use_system_prompt=self.profile.use_system_prompt,
            )
        finally:
            finished_unix = time.time()
        elapsed = time.perf_counter() - start
        self._real_backend_calls += 1
        frames = self._frames_from_result(result)
        timestamps = tuple(index / self.profile.fps for index in range(len(frames)))
        timing = ServerTiming(
            backend_calls=1,
            wall_seconds=elapsed,
            cold_start=cold_start,
            gpu_peak_memory_bytes=self._peak_memory(runtime),
            model_load_seconds=model_load_seconds,
            started_at_unix=started_unix,
            finished_at_unix=finished_unix,
        )
        world_result = WorldResult(
            backend="cosmos3_diffusers",
            profile_id=self.profile.profile_id,
            frames=frames,
            nominal_frame_timestamps=timestamps,
            conditioning_frame_included=True,
            timing=timing,
            request_id=request.request_id,
            metadata={
                "model_id": "nvidia/Cosmos3-Nano",
                "model_revision": self.profile.model_revision,
                "diffusers_revision": self.profile.diffusers_revision,
                "mode": "forward_dynamics",
                "domain_name": request.domain,
                "action_length": len(request.compiled_actions),
                "action_length_status": action_length_status,
                "prompt_is_plain_task_string": True,
                "nominal_timestamps_only": True,
                "feedback_mode": request.feedback_mode.value,
            },
        )
        # Never trim, duplicate, or invent frames to make a test pass.
        world_result.validate(len(request.compiled_actions))
        return world_result


@dataclass(frozen=True)
class IRASimBridgeProfile:
    """Official IRASim Bridge contract; intentionally has no implicit loader."""

    profile_id: str
    archive_path: Optional[str] = None
    archive_sha256: Optional[str] = None
    code_revision: str = IRASIM_COMMIT
    loader_revision: Optional[str] = None
    container_digest: Optional[str] = None
    input_height: int = 256
    input_width: int = 320
    num_frames: int = 16
    action_count: int = 15
    action_scale: Tuple[float, ...] = (20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0)

    def __post_init__(self) -> None:
        if self.input_height != 256 or self.input_width != 320:
            raise ValueError("The pinned IRASim Bridge profile is 256x320.")
        if self.num_frames != 16 or self.action_count != 15 or len(self.action_scale) != 7:
            raise ValueError("The pinned IRASim Bridge contract is 16 frames, 15 7-D actions.")


@dataclass(frozen=True)
class IRASimPreparedRequest:
    profile_id: str
    scaled_actions: Tuple[Tuple[float, ...], ...]
    expected_frames: int
    metadata: Mapping[str, Any]


class IRASimBridgeAdapter:
    """Separate IRASim action preparation contract.

    No loader is provided because a known archive and a code checkout do not
    establish a safe/compatible runnable inference stack.  This adapter may
    prepare a fixture request but always reports unsupported execution until a
    reviewed native loader is supplied.
    """

    def __init__(self, profile: IRASimBridgeProfile) -> None:
        self.profile = profile

    def capability(self) -> CapabilityResult:
        return CapabilityResult(
            status=CapabilityStatus.UNSUPPORTED,
            reason=(
                "IRASim Bridge uses its own 256x320 / 16-frame / 15x7-D contract. "
                "No reviewed native loader is implemented, so it cannot be a Cosmos fallback or qualified backend."
            ),
            source_verified=True,
            evidence_uris=(
                "https://github.com/bytedance/IRASim/blob/%s/configs/evaluation/bridge/frame_ada.yaml" % IRASIM_COMMIT,
                "https://github.com/bytedance/IRASim/blob/%s/dataset/dataset_3D.py" % IRASIM_COMMIT,
            ),
            details={
                "input_size": [self.profile.input_height, self.profile.input_width],
                "num_frames": self.profile.num_frames,
                "action_shape": [self.profile.action_count, 7],
                "action_scale": list(self.profile.action_scale),
            },
        )

    def prepare(self, request: WorldRequest) -> IRASimPreparedRequest:
        request.validate_basic()
        if request.compatibility_profile_id != self.profile.profile_id:
            raise BackendContractError("WorldRequest profile ID does not match the IRASim profile.")
        if len(request.compiled_actions) != self.profile.action_count:
            raise BackendContractError(
                "IRASim needs exactly %d action rows, got %d." % (self.profile.action_count, len(request.compiled_actions))
            )
        scaled = []
        for row in request.compiled_actions:
            if len(row) != 7:
                raise BackendContractError("IRASim accepts native 7-D actions only; Cosmos 10-D rows are prohibited.")
            scaled.append(tuple(float(value) * self.profile.action_scale[index] for index, value in enumerate(row)))
        return IRASimPreparedRequest(
            profile_id=self.profile.profile_id,
            scaled_actions=tuple(scaled),
            expected_frames=self.profile.num_frames,
            metadata={
                "native_action_shape": [15, 7],
                "frame_contract": "condition frame + 15 action transitions = 16 frames",
                "source_revision": self.profile.code_revision,
                "execution_status": "unsupported_no_native_loader",
            },
        )

    def generate(self, request: WorldRequest) -> WorldResult:
        self.prepare(request)
        raise BackendUnavailableError(
            "IRASim request is structurally valid but execution is unsupported until a reviewed native loader, "
            "archive hashes/licenses, and independent fixtures are implemented."
        )
