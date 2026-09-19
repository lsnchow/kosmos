"""Pinned, local-only OpenVLA Bridge inference.

The adapter intentionally implements only OpenVLA's native one-step call.  A
world rollout must obtain a new generated image before every call; this class
does not manufacture a proposal by repeating a stale image or a one-step
action.  It is usable for real local inference once the separately reviewed
checkpoint and Transformers 4.40.1 environment are installed, but it does not
claim Gate-B qualification.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from plumb.adapters.contracts import CapabilityResult, CapabilityStatus, PolicyContract, PolicyObservation

from .contracts import OPENVLA_SOURCE_COMMIT, PolicyContractError, PolicyLoadError


OPENVLA_MODEL_ID = "openvla/openvla-7b"
OPENVLA_TRANSFORMERS_VERSION = "4.40.1"
OPENVLA_BRIDGE_UNNORM_KEY = "bridge_orig"
OPENVLA_CONTROL_SOURCE = (
    "https://github.com/openvla/openvla/blob/"
    + OPENVLA_SOURCE_COMMIT
    + "/experiments/robot/bridge/run_bridgev2_eval.py#L245-L280"
)


class OpenVLAUnavailableError(PolicyLoadError):
    """The approved local OpenVLA environment is not available."""


@dataclass(frozen=True)
class OpenVLAPolicyProfile:
    """The approval boundary for one local OpenVLA snapshot.

    Remote model code is never trusted by default.  Loading requires all of:
    an immutable checkpoint revision, the immutable remote-code revision, a
    matching reviewed revision, and an explicit per-profile opt-in.  Keeping
    these separate makes it impossible for an unreviewed ``main`` revision to
    piggy-back on a reviewed weight snapshot.
    """

    profile_id: str
    local_model_path: str
    checkpoint_revision: str
    remote_code_revision: str
    reviewed_remote_code_revision: Optional[str] = None
    allow_trust_remote_code: bool = False
    transformers_version: str = OPENVLA_TRANSFORMERS_VERSION
    unnorm_key: str = OPENVLA_BRIDGE_UNNORM_KEY
    local_files_only: bool = True
    device_map: Optional[str] = "auto"
    torch_dtype: str = "bfloat16"
    container_digest: Optional[str] = None
    asset_manifest_id: Optional[str] = None
    asset_manifest_sha256: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None

    def review_error(self) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        if not self.local_model_path:
            return "local_model_path is required; Hub fetching is disabled"
        if not _immutable_revision(self.checkpoint_revision):
            return "checkpoint_revision must be an immutable 40-character hexadecimal revision"
        if not _immutable_revision(self.remote_code_revision):
            return "remote_code_revision must be an immutable 40-character hexadecimal revision"
        if self.reviewed_remote_code_revision != self.remote_code_revision:
            return "reviewed_remote_code_revision must exactly match remote_code_revision"
        if not self.allow_trust_remote_code:
            return "allow_trust_remote_code=True is required after explicit remote-code review"
        if self.transformers_version != OPENVLA_TRANSFORMERS_VERSION:
            return (
                "OpenVLA requires its pinned Transformers %s runtime, not %r; "
                "modern Transformers 5 compatibility is not assumed."
                % (OPENVLA_TRANSFORMERS_VERSION, self.transformers_version)
            )
        if self.unnorm_key != OPENVLA_BRIDGE_UNNORM_KEY:
            return "OpenVLA Bridge profile must declare unnorm_key='bridge_orig'"
        if not self.local_files_only:
            return "OpenVLA loader is local-only; network retrieval is not permitted"
        if self.torch_dtype not in ("bfloat16", "float16", "float32"):
            return "torch_dtype must be bfloat16, float16, or float32"
        return None


@dataclass(frozen=True)
class OpenVLAActionReport:
    """One actual native model call and its observed resources.

    A missing GPU metric remains ``None``; it is never rewritten as zero.
    """

    action: Tuple[float, float, float, float, float, float, float]
    unnorm_key: str
    source_image_timestamp: Optional[float]
    backend_calls: int
    wall_seconds: Optional[float]
    gpu_peak_memory_bytes: Optional[int]


@dataclass(frozen=True)
class _OpenVLARuntime:
    torch: Any
    model_cls: Any
    processor_cls: Any
    image_fromarray: Callable[[Any], Any]
    transformers_version: str


OPENVLA_CONTRACT = PolicyContract(
    name="OpenVLA",
    required_observation_history=1,
    requires_proprio=False,
    native_proposal_horizon=1,
    certified_execute_prefix=1,
    temporal_ensembling=False,
    preprocessing="OpenVLA native image preprocessing inside reviewed remote model code.",
    normalization="OpenVLA predict_action(..., unnorm_key='bridge_orig') returns Bridge-normalized 7-D action.",
    reset_rule="No action proposal cache; one current generated image per call.",
    rng_rule="Native deterministic predict_action(do_sample=False); world seed is external.",
    implementation_status=CapabilityStatus.READY_UNQUALIFIED,
    limitation=(
        "A one-tick forward-dynamics/image-feedback fixture and Gate B are required before named-policy "
        "results can be qualified."
    ),
)


def _immutable_revision(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


class OpenVLAPolicyAdapter:
    """Local-only, lazily imported OpenVLA predictor.

    ``runtime_factory`` and ``model_factory`` are dependency injection seams
    for fixture tests and isolated deployment images.  They do not weaken the
    profile approval checks: even an injected model cannot be used without the
    explicit reviewed remote-code acknowledgement.
    """

    contract = OPENVLA_CONTRACT

    def __init__(
        self,
        profile: OpenVLAPolicyProfile,
        *,
        runtime_factory: Optional[Callable[[], _OpenVLARuntime]] = None,
        model_factory: Optional[Callable[[OpenVLAPolicyProfile, _OpenVLARuntime], Any]] = None,
        processor_factory: Optional[Callable[[OpenVLAPolicyProfile, _OpenVLARuntime], Any]] = None,
    ) -> None:
        self.profile = profile
        self._runtime_factory = runtime_factory
        self._model_factory = model_factory
        self._processor_factory = processor_factory
        self._runtime: Optional[_OpenVLARuntime] = None
        self._model: Any = None
        self._processor: Any = None
        self._backend_calls = 0
        self.last_report: Optional[OpenVLAActionReport] = None

    def capability(self) -> CapabilityResult:
        review_error = self.profile.review_error()
        if review_error is not None:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="OpenVLA loading is blocked: " + review_error + ".",
                source_verified=True,
                evidence_uris=(OPENVLA_CONTROL_SOURCE,),
                details={"profile_id": self.profile.profile_id, "trust_remote_code": False},
            )
        if not Path(self.profile.local_model_path).is_dir() and (
            self._model_factory is None or self._processor_factory is None
        ):
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="OpenVLA local checkpoint directory is absent; no Hub download was attempted.",
                source_verified=True,
                evidence_uris=(OPENVLA_CONTROL_SOURCE,),
                details={"local_model_path": self.profile.local_model_path, "local_files_only": True},
            )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Pinned local OpenVLA native one-action prediction is configured with bridge_orig normalization. "
                "It remains unqualified until one-tick feedback and Gate-B fixtures pass."
            ),
            source_verified=True,
            evidence_uris=(OPENVLA_CONTROL_SOURCE,),
            details={
                "checkpoint_revision": self.profile.checkpoint_revision,
                "remote_code_revision": self.profile.remote_code_revision,
                "reviewed_remote_code_revision": self.profile.reviewed_remote_code_revision,
                "trust_remote_code": True,
                "transformers_version": self.profile.transformers_version,
                "unnorm_key": self.profile.unnorm_key,
                "native_proposal_horizon": 1,
                "local_files_only": True,
                "asset_manifest_id": self.profile.asset_manifest_id,
                "asset_manifest_sha256": self.profile.asset_manifest_sha256,
                "runtime_lock_id": self.profile.runtime_lock_id,
                "runtime_lock_sha256": self.profile.runtime_lock_sha256,
            },
        )

    def _check_profile(self) -> None:
        error = self.profile.review_error()
        if error is not None:
            raise OpenVLAUnavailableError("Refusing to load OpenVLA: " + error + ".")

    def _load_runtime(self) -> _OpenVLARuntime:
        if self._runtime is not None:
            return self._runtime
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
            if runtime.transformers_version != OPENVLA_TRANSFORMERS_VERSION:
                raise OpenVLAUnavailableError(
                    "Injected OpenVLA runtime is Transformers %r; exactly %s is required."
                    % (runtime.transformers_version, OPENVLA_TRANSFORMERS_VERSION)
                )
            self._runtime = runtime
            return runtime
        try:
            import torch  # type: ignore
            import transformers  # type: ignore
            from PIL import Image  # type: ignore
            from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
        except ImportError as error:
            raise OpenVLAUnavailableError(
                "OpenVLA dependencies are unavailable. Install the pinned Transformers 4.40.1/Torch runtime "
                "in an isolated policy image; imports are lazy."
            ) from error
        installed = str(getattr(transformers, "__version__", ""))
        if installed != OPENVLA_TRANSFORMERS_VERSION:
            raise OpenVLAUnavailableError(
                "Installed Transformers %r is incompatible; OpenVLA requires exactly %s and does not assume "
                "Transformers 5 compatibility."
                % (installed, OPENVLA_TRANSFORMERS_VERSION)
            )
        self._runtime = _OpenVLARuntime(
            torch=torch,
            model_cls=AutoModelForVision2Seq,
            processor_cls=AutoProcessor,
            image_fromarray=Image.fromarray,
            transformers_version=installed,
        )
        return self._runtime

    def _ensure_components(self, runtime: _OpenVLARuntime) -> Tuple[Any, Any]:
        self._check_profile()
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(self.profile, runtime)
            else:
                checkpoint = Path(self.profile.local_model_path)
                if not checkpoint.is_dir():
                    raise OpenVLAUnavailableError(
                        "Expected local OpenVLA checkpoint at %s; refusing network retrieval." % checkpoint
                    )
                dtype = getattr(runtime.torch, self.profile.torch_dtype, None)
                if dtype is None:
                    raise OpenVLAUnavailableError("Pinned OpenVLA Torch runtime lacks dtype %s." % self.profile.torch_dtype)
                kwargs = {
                    "revision": self.profile.checkpoint_revision,
                    "code_revision": self.profile.remote_code_revision,
                    "trust_remote_code": True,
                    "local_files_only": True,
                    "use_safetensors": True,
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": True,
                }
                if self.profile.device_map is not None:
                    kwargs["device_map"] = self.profile.device_map
                self._model = runtime.model_cls.from_pretrained(str(checkpoint), **kwargs)
            if not callable(getattr(self._model, "predict_action", None)):
                raise OpenVLAUnavailableError("Reviewed OpenVLA model does not expose native predict_action().")
            eval_method = getattr(self._model, "eval", None)
            if callable(eval_method):
                maybe_model = eval_method()
                if maybe_model is not None:
                    self._model = maybe_model
        if self._processor is None:
            if self._processor_factory is not None:
                self._processor = self._processor_factory(self.profile, runtime)
            else:
                self._processor = runtime.processor_cls.from_pretrained(
                    self.profile.local_model_path,
                    revision=self.profile.checkpoint_revision,
                    code_revision=self.profile.remote_code_revision,
                    local_files_only=True,
                    trust_remote_code=True,
                )
            if not callable(self._processor):
                raise OpenVLAUnavailableError("Reviewed OpenVLA processor is not callable.")
        return self._model, self._processor

    @staticmethod
    def _check_observation(observation: PolicyObservation) -> Any:
        if not isinstance(observation.prompt, str) or not observation.prompt.strip():
            raise PolicyContractError("OpenVLA requires a nonempty exact task instruction.")
        # Enforce a one-image native interface.  Taking the final item from a
        # longer history would make stale-image use look like proper feedback.
        if len(observation.image_history) != 1 or observation.image_history[0] is None:
            raise PolicyContractError(
                "OpenVLA requires exactly one fresh current image; do not supply padded or stale image history."
            )
        if observation.proprio is not None:
            raise PolicyContractError("OpenVLA's base Bridge API must not receive proprioception.")
        return observation.image_history[0]

    @staticmethod
    def _normalise_action(raw_action: Any) -> Tuple[float, float, float, float, float, float, float]:
        value = raw_action
        for method_name in ("detach", "cpu"):
            method = getattr(value, method_name, None)
            if callable(method):
                value = method()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise PolicyContractError("OpenVLA predict_action() must return one numeric 7-D action.")
        # Some Torch versions return one batched row.  Accept only that one
        # row; a multi-row proposal is never silently truncated.
        if len(value) == 1 and isinstance(value[0], Sequence) and not isinstance(value[0], (str, bytes)):
            value = value[0]
        if len(value) != 7:
            raise PolicyContractError(
                "OpenVLA predict_action() returned %d values; expected exactly one 7-D action (no padding)." % len(value)
            )
        try:
            action = tuple(float(component) for component in value)
        except (TypeError, ValueError) as error:
            raise PolicyContractError("OpenVLA action contains a nonnumeric component.") from error
        if not all(math.isfinite(component) for component in action):
            raise PolicyContractError("OpenVLA action components must all be finite.")
        return action  # type: ignore[return-value]

    @staticmethod
    def _inference_context(torch_module: Any) -> Any:
        no_grad = getattr(torch_module, "inference_mode", None) or getattr(torch_module, "no_grad", None)
        return no_grad() if callable(no_grad) else contextlib.nullcontext()

    @staticmethod
    def _peak_memory(torch_module: Any) -> Optional[int]:
        cuda = getattr(torch_module, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                return int(cuda.max_memory_allocated())
        except (AttributeError, RuntimeError):
            return None
        return None

    @staticmethod
    def _prepare_image(runtime: _OpenVLARuntime, image: Any) -> Any:
        """Turn the canonical RGB frame into the PIL image required by reviewed code."""

        convert = getattr(image, "convert", None)
        if callable(convert):
            return convert("RGB")
        try:
            converted = runtime.image_fromarray(image)
        except Exception as error:
            raise PolicyContractError(
                "OpenVLA requires a current RGB uint8 image or PIL-compatible array for native preprocessing."
            ) from error
        convert = getattr(converted, "convert", None)
        if not callable(convert):
            raise PolicyContractError("OpenVLA image factory did not return a PIL-compatible image.")
        return convert("RGB")

    @staticmethod
    def _prepare_inputs(processor: Any, image: Any, instruction: str, runtime: _OpenVLARuntime, model: Any) -> Any:
        # This exact template follows the pinned OpenVLA Bridge helper for the
        # openvla-7b (non-v0.1) checkpoint.  It lower-cases only within the
        # source-native sentence template; no policy rollout data is appended.
        prompt = "In: What action should the robot take to %s?\nOut:" % instruction.lower()
        try:
            inputs = processor(prompt, image, return_tensors="pt")
        except TypeError:
            # Some reviewed processor versions expose explicit names while
            # preserving exactly the same text/image payload.
            inputs = processor(text=prompt, images=image, return_tensors="pt")
        device = getattr(model, "device", None)
        dtype = getattr(runtime.torch, "bfloat16", None)
        move = getattr(inputs, "to", None)
        if callable(move) and device is not None and dtype is not None:
            moved = move(device, dtype=dtype)
            return inputs if moved is None else moved
        if isinstance(inputs, Mapping) and device is not None:
            moved_inputs = {}
            for key, value in inputs.items():
                move_value = getattr(value, "to", None)
                if callable(move_value):
                    try:
                        moved_inputs[key] = move_value(device, dtype=dtype)
                    except TypeError:
                        moved_inputs[key] = move_value(device)
                else:
                    moved_inputs[key] = value
            return moved_inputs
        return inputs

    def predict_with_report(self, observation: PolicyObservation) -> OpenVLAActionReport:
        """Make exactly one native ``predict_action`` call for a fresh image."""

        image = self._check_observation(observation)
        runtime = self._load_runtime()
        model, processor = self._ensure_components(runtime)
        native_image = self._prepare_image(runtime, image)
        inputs = self._prepare_inputs(processor, native_image, observation.prompt, runtime, model)
        cuda = getattr(runtime.torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                cuda.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            pass
        started = time.perf_counter()
        with self._inference_context(runtime.torch):
            raw_action = model.predict_action(
                **inputs,
                unnorm_key=OPENVLA_BRIDGE_UNNORM_KEY,
                do_sample=False,
            )
        wall_seconds = time.perf_counter() - started
        action = self._normalise_action(raw_action)
        self._backend_calls += 1
        report = OpenVLAActionReport(
            action=action,
            unnorm_key=OPENVLA_BRIDGE_UNNORM_KEY,
            source_image_timestamp=observation.timestamp,
            backend_calls=1,
            wall_seconds=wall_seconds,
            gpu_peak_memory_bytes=self._peak_memory(runtime.torch),
        )
        self.last_report = report
        return report

    def predict_action(self, observation: PolicyObservation) -> Tuple[float, float, float, float, float, float, float]:
        """Return the native one-step Bridge action (never a padded proposal)."""

        return self.predict_with_report(observation).action

    # A short generic alias is useful to the rollout controller while retaining
    # ``predict_action`` as the source-native API in artifacts and tests.
    predict = predict_action
