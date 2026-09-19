"""Pinned, local-only OpenPiZero Bridge inference with refreshed proprioception.

OpenPiZero is the only policy in the matrix whose native API consumes robot
state, so this adapter is deliberately hostile to a stale or absent proprio
vector: it refuses rather than zero-filling, and it refuses an all-zero vector
outright because the Bridge base origin is not a reachable end-effector pose.

It is also the only policy whose declared ``action_normalization_type`` is
``bounds``.  That is an adapter-specific setting, not a universal native-policy
API, so the bounds statistics are an explicit revisioned component with an
invertibility test rather than an assumption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from plumb.adapters.contracts import PolicyContract, PolicyObservation

from .contracts import (
    ActionNormalizationType,
    FrameIdentity,
    GripperPolarityConvention,
    GripperStateValue,
    NativePolicyAdapter,
    NativeProposal,
    NativeWrapperEntryPoint,
    PolicyActionNormalizer,
    PolicyCertification,
    PolicyContractError,
    PolicyExecutionMode,
    PolicyLoadError,
    UNRESOLVED_BRIDGE_GRIPPER_CONVENTION,
    immutable_revision,
    inference_context,
    peak_memory_bytes,
    rows_from_native,
)
from .native import AUTOEVAL_POLICY_SOURCE, OPEN_PI_ZERO_CONTRACT, OPEN_PI_ZERO_SOURCE, OPEN_PI_ZERO_SOURCE_COMMIT


OPEN_PI_ZERO_MODEL_ID = "allenzren/open-pi-zero"
OPEN_PI_ZERO_CHECKPOINT_FILE = "bridge_beta_step19296_2024-12-26_22-30_42.pt"
OPEN_PI_ZERO_ACTION_CHUNK = 4
OPEN_PI_ZERO_NORMALIZATION_TYPE = "bounds"
PALIGEMMA_OFFICIAL_REPO = "google/paligemma-3b-pt-224"
PALIGEMMA_CANDIDATE_MIRROR = "leo009/paligemma-3b-pt-224"
PALIGEMMA_REQUIRED_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "config.json",
)


class OpenPiZeroUnavailableError(PolicyLoadError):
    """The approved local OpenPiZero environment or support files are absent."""


class OpenPiZeroWrapper(Protocol):
    """The reviewed native surface this adapter calls.

    ``infer_action`` must return the checkpoint's *normalized* four-row action
    chunk; PLUMB applies the revisioned bounds denormalization itself so the
    transform happens exactly once at one documented boundary.
    """

    def infer_action(self, *, image: Any, text: str, proprio: Sequence[float]) -> Any:
        ...


@dataclass(frozen=True)
class ProprioLayout:
    """How the canonical 8-D Bridge state maps onto a wrapper's proprio input.

    The canonical vector is ``(x, y, z, rx, ry, rz, 0, gripper)`` where channel
    6 is a structural zero and channel 7 is a *state* gripper in the 0.0-0.39
    range with 0.0 fully closed.  Which channels a wrapper consumes, and in
    which units, is wrapper-specific, so it is declared rather than inferred.
    """

    layout_id: str
    indices: Tuple[int, ...]
    gripper_units: str = "bridge_state"
    source_uri: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.layout_id:
            raise PolicyContractError("ProprioLayout.layout_id is required.")
        if not self.indices:
            raise PolicyContractError("ProprioLayout needs at least one channel index.")
        if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 7 for index in self.indices):
            raise PolicyContractError("ProprioLayout indices must address the canonical 8-D Bridge state.")
        if len(set(self.indices)) != len(self.indices):
            raise PolicyContractError("ProprioLayout indices must be unique.")
        if self.gripper_units not in ("bridge_state", "action_gripper"):
            raise PolicyContractError("ProprioLayout.gripper_units must be 'bridge_state' or 'action_gripper'.")

    @property
    def width(self) -> int:
        return len(self.indices)

    def apply(self, proprio8: Sequence[float], convention: GripperPolarityConvention) -> Tuple[float, ...]:
        if len(proprio8) != 8:
            raise PolicyContractError("ProprioLayout requires the canonical 8-D Bridge state.")
        gripper_state = GripperStateValue(float(proprio8[7]))
        if self.gripper_units == "action_gripper":
            # Explicit, refused-by-default conversion: state units and action
            # gripper units are not interchangeable.
            gripper_value = float(convention.action_from_state(gripper_state).value)
        else:
            gripper_value = float(gripper_state.value)
        values: List[float] = []
        for index in self.indices:
            values.append(gripper_value if index == 7 else float(proprio8[index]))
        return tuple(values)


BRIDGE_8D_PASSTHROUGH_LAYOUT = ProprioLayout(
    layout_id="bridge_8d_xyz_rpy_pad_grip_passthrough",
    indices=(0, 1, 2, 3, 4, 5, 6, 7),
    gripper_units="bridge_state",
    source_uri=AUTOEVAL_POLICY_SOURCE,
)
BRIDGE_7D_DROP_STRUCTURAL_ZERO_LAYOUT = ProprioLayout(
    layout_id="bridge_7d_xyz_rpy_grip_drop_structural_zero",
    indices=(0, 1, 2, 3, 4, 5, 7),
    gripper_units="bridge_state",
    source_uri=AUTOEVAL_POLICY_SOURCE,
)


@dataclass(frozen=True)
class PaliGemmaSupportFiles:
    """The seven required PaliGemma support files and their access terms."""

    local_path: str
    repo_id: str = PALIGEMMA_OFFICIAL_REPO
    revision: Optional[str] = None
    terms_accepted: bool = False
    mirror_comparison_report_sha256: Optional[str] = None

    def review_error(self) -> Optional[str]:
        if not self.local_path:
            return "PaliGemma support files need an explicit local path and application cache directory"
        if self.repo_id not in (PALIGEMMA_OFFICIAL_REPO, PALIGEMMA_CANDIDATE_MIRROR):
            return (
                "PaliGemma support files must come from the official %r, or from the candidate mirror %r after "
                "provenance, hashes, and terms are resolved" % (PALIGEMMA_OFFICIAL_REPO, PALIGEMMA_CANDIDATE_MIRROR)
            )
        if not immutable_revision(self.revision):
            return "PaliGemma revision must be an immutable 40-character hexadecimal revision"
        if not self.terms_accepted:
            return "PaliGemma requires explicitly accepted access terms"
        if self.repo_id == PALIGEMMA_CANDIDATE_MIRROR and not _is_sha256_hex(self.mirror_comparison_report_sha256):
            return (
                "%r is a candidate mirror only: a saved file-by-file comparison report against %r is required, "
                "and byte identity may not be claimed without it"
                % (PALIGEMMA_CANDIDATE_MIRROR, PALIGEMMA_OFFICIAL_REPO)
            )
        return None

    def missing_files(self) -> Tuple[str, ...]:
        root = Path(self.local_path)
        return tuple(name for name in PALIGEMMA_REQUIRED_FILES if not (root / name).is_file())

    def payload(self) -> Dict[str, Any]:
        return {
            "local_path": self.local_path,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "official": self.repo_id == PALIGEMMA_OFFICIAL_REPO,
            "terms_accepted": bool(self.terms_accepted),
            "mirror_comparison_report_sha256": self.mirror_comparison_report_sha256,
            "byte_identity_claimed": False,
            "required_files": list(PALIGEMMA_REQUIRED_FILES),
        }


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value[len("sha256:"):] if value.startswith("sha256:") else value
    return len(candidate) == 64 and all(character in "0123456789abcdefABCDEF" for character in candidate)


@dataclass(frozen=True)
class OpenPiZeroPolicyProfile:
    """The approval boundary for one local OpenPiZero snapshot."""

    profile_id: str
    converted_checkpoint_path: str
    converted_checkpoint_sha256: Optional[str]
    source_pickle_sha256: Optional[str]
    conversion_report_sha256: Optional[str]
    checkpoint_revision: str
    paligemma: PaliGemmaSupportFiles
    proprio_layout: ProprioLayout
    entry_point: NativeWrapperEntryPoint
    application_cache_dir: Optional[str] = None
    model_id: str = OPEN_PI_ZERO_MODEL_ID
    action_chunk: int = OPEN_PI_ZERO_ACTION_CHUNK
    action_normalization_type: str = OPEN_PI_ZERO_NORMALIZATION_TYPE
    state_dict_coverage_verified: bool = False
    torch_dtype: str = "bfloat16"
    local_files_only: bool = True
    native_unnormalizes: bool = False
    asset_manifest_id: Optional[str] = None
    asset_manifest_sha256: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None

    def review_error(self) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        if self.model_id != OPEN_PI_ZERO_MODEL_ID:
            return "model_id must be %r" % OPEN_PI_ZERO_MODEL_ID
        if not immutable_revision(self.checkpoint_revision):
            return "checkpoint_revision must be an immutable 40-character hexadecimal revision"
        if not self.converted_checkpoint_path:
            return "converted_checkpoint_path is required"
        suffix = Path(self.converted_checkpoint_path).suffix.lower()
        if suffix in (".pt", ".pth", ".pkl", ".bin", ".ckpt"):
            return (
                "OpenPiZero's released %s is a legacy pickle; inference consumes only a reviewed converted "
                "artifact produced in an isolated, credential-free, network-free environment"
                % OPEN_PI_ZERO_CHECKPOINT_FILE
            )
        if suffix != ".safetensors":
            return "converted OpenPiZero weights must be a .safetensors file, got %r" % (suffix or "<none>")
        if not _is_sha256_hex(self.converted_checkpoint_sha256):
            return "converted_checkpoint_sha256 is required"
        if not _is_sha256_hex(self.source_pickle_sha256):
            return "source_pickle_sha256 is required so the conversion is traceable"
        if not _is_sha256_hex(self.conversion_report_sha256):
            return (
                "conversion_report_sha256 is required; a pickle scan or strict=True alone does not establish "
                "safety or provenance"
            )
        if not self.state_dict_coverage_verified:
            return (
                "full state-dict coverage must be verified in the pinned loader; strict=True checks keys, not "
                "provenance or deserialization safety"
            )
        paligemma_error = self.paligemma.review_error()
        if paligemma_error is not None:
            return paligemma_error
        if not self.application_cache_dir:
            return "an explicit application cache directory is required"
        entry_error = self.entry_point.review_error()
        if entry_error is not None:
            return entry_error
        if int(self.action_chunk) != OPEN_PI_ZERO_ACTION_CHUNK:
            return "OpenPiZero's supplied configuration is a %d-action chunk, not %r" % (
                OPEN_PI_ZERO_ACTION_CHUNK,
                self.action_chunk,
            )
        if self.action_normalization_type != OPEN_PI_ZERO_NORMALIZATION_TYPE:
            return "OpenPiZero declares action_normalization_type=%r, not %r" % (
                OPEN_PI_ZERO_NORMALIZATION_TYPE,
                self.action_normalization_type,
            )
        if not self.local_files_only:
            return "OpenPiZero loader is local-only; network retrieval is not permitted"
        if self.native_unnormalizes:
            return (
                "PLUMB applies its own revisioned bounds denormalization at exactly one documented boundary; "
                "letting the wrapper also unnormalize would apply the transform twice"
            )
        if self.torch_dtype not in ("bfloat16", "float16", "float32"):
            return "torch_dtype must be bfloat16, float16, or float32"
        return None


@dataclass(frozen=True)
class _OpenPiZeroRuntime:
    """Injection seam for the lazily imported Torch/open-pi-zero stack."""

    torch: Any
    torch_version: str
    image_fromarray: Callable[[Any], Any]
    wrapper_factory: Callable[[OpenPiZeroPolicyProfile], Any]


class OpenPiZeroPolicyAdapter(NativePolicyAdapter):
    """Local-only OpenPiZero four-action chunk predictor with fresh proprioception."""

    base_contract: PolicyContract = OPEN_PI_ZERO_CONTRACT
    source_urls = (OPEN_PI_ZERO_SOURCE, AUTOEVAL_POLICY_SOURCE)

    def __init__(
        self,
        profile: OpenPiZeroPolicyProfile,
        *,
        normalizer: PolicyActionNormalizer,
        certification: Optional[PolicyCertification] = None,
        execution_mode: PolicyExecutionMode = PolicyExecutionMode.CERTIFIED,
        gripper_convention: GripperPolarityConvention = UNRESOLVED_BRIDGE_GRIPPER_CONVENTION,
        runtime_factory: Optional[Callable[[], _OpenPiZeroRuntime]] = None,
        wrapper_factory: Optional[Callable[[OpenPiZeroPolicyProfile, _OpenPiZeroRuntime], Any]] = None,
    ) -> None:
        super().__init__(
            normalizer=normalizer,
            certification=certification,
            execution_mode=execution_mode,
            gripper_convention=gripper_convention,
        )
        if normalizer.statistics.normalization_type is not ActionNormalizationType.BOUNDS:
            raise PolicyContractError(
                "OpenPiZero declares action_normalization_type='bounds'; %r is a different family."
                % normalizer.statistics.normalization_type.value
            )
        self.profile = profile
        self._runtime_factory = runtime_factory
        self._wrapper_factory = wrapper_factory
        self._runtime: Optional[_OpenPiZeroRuntime] = None
        self._wrapper: Any = None
        self._last_proprio: Optional[Tuple[float, ...]] = None
        self._last_native_proprio: Optional[Tuple[float, ...]] = None
        self.last_gpu_peak_memory_bytes: Optional[int] = None

    # -- capability -------------------------------------------------------------

    def _profile_errors(self) -> Tuple[str, ...]:
        problems: List[str] = []
        error = self.profile.review_error()
        if error is not None:
            problems.append(error)
        if self.profile.proprio_layout.gripper_units == "action_gripper" and not self.gripper_convention.resolved:
            problems.append(
                "the declared proprio layout converts the state gripper into action units, which requires a "
                "resolved gripper polarity convention rather than %r" % self.gripper_convention.convention_id
            )
        return tuple(problems)

    def _availability_error(self) -> Optional[str]:
        if self._wrapper_factory is not None:
            return None
        if not Path(self.profile.converted_checkpoint_path).is_file():
            return "reviewed converted checkpoint %s is absent" % self.profile.converted_checkpoint_path
        missing = self.profile.paligemma.missing_files()
        if missing:
            return "PaliGemma support files are missing: %s" % ", ".join(missing)
        return None

    def _capability_details(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile.profile_id,
            "model_id": self.profile.model_id,
            "checkpoint_revision": self.profile.checkpoint_revision,
            "upstream_checkpoint_file": OPEN_PI_ZERO_CHECKPOINT_FILE,
            "converted_checkpoint_path": self.profile.converted_checkpoint_path,
            "converted_checkpoint_sha256": self.profile.converted_checkpoint_sha256,
            "source_pickle_sha256": self.profile.source_pickle_sha256,
            "conversion_report_sha256": self.profile.conversion_report_sha256,
            "state_dict_coverage_verified": bool(self.profile.state_dict_coverage_verified),
            "action_chunk": int(self.profile.action_chunk),
            "action_normalization_type": self.profile.action_normalization_type,
            "proprio_layout_id": self.profile.proprio_layout.layout_id,
            "proprio_width": self.profile.proprio_layout.width,
            "proprio_gripper_units": self.profile.proprio_layout.gripper_units,
            "paligemma": self.profile.paligemma.payload(),
            "application_cache_dir": self.profile.application_cache_dir,
            "entry_point": "%s.%s" % (self.profile.entry_point.module, self.profile.entry_point.attribute),
            "entry_point_loader_revision": self.profile.entry_point.loader_revision,
            "source_commit": OPEN_PI_ZERO_SOURCE_COMMIT,
            "local_files_only": True,
            "asset_manifest_id": self.profile.asset_manifest_id,
            "asset_manifest_sha256": self.profile.asset_manifest_sha256,
            "runtime_lock_id": self.profile.runtime_lock_id,
            "runtime_lock_sha256": self.profile.runtime_lock_sha256,
        }

    # -- durable state ----------------------------------------------------------

    def _reset_native(self, seed: int) -> None:
        self._last_proprio = None
        self._last_native_proprio = None

    def _snapshot_extra(self) -> Dict[str, Any]:
        return {
            "proprio_layout_id": self.profile.proprio_layout.layout_id,
            "last_proprio": None if self._last_proprio is None else list(self._last_proprio),
            "last_native_proprio": None if self._last_native_proprio is None else list(self._last_native_proprio),
        }

    def _restore_extra(self, payload: Mapping[str, Any]) -> None:
        if str(payload.get("proprio_layout_id")) != self.profile.proprio_layout.layout_id:
            raise PolicyContractError(
                "OpenPiZero snapshot proprio layout %r does not match the live layout %r."
                % (payload.get("proprio_layout_id"), self.profile.proprio_layout.layout_id)
            )
        recorded = payload.get("last_proprio")
        self._last_proprio = None if recorded is None else tuple(float(value) for value in recorded)
        native = payload.get("last_native_proprio")
        self._last_native_proprio = None if native is None else tuple(float(value) for value in native)

    # -- observation ------------------------------------------------------------

    def _check_extra_observation(self, observation: PolicyObservation) -> None:
        proprio = tuple(float(value) for value in observation.proprio or ())
        if self._last_proprio is not None and proprio == self._last_proprio and observation.timestamp is None:
            raise PolicyContractError(
                "OpenPiZero received a byte-identical proprioception vector with no advancing control "
                "timestamp, so it cannot establish that the state was refreshed. It will not reuse a stale "
                "pose or substitute a zero fill."
            )

    # -- runtime ----------------------------------------------------------------

    def _check_profile(self) -> None:
        problems = self._profile_errors()
        if problems:
            raise OpenPiZeroUnavailableError("Refusing to load OpenPiZero: " + "; ".join(problems) + ".")

    def _load_runtime(self) -> _OpenPiZeroRuntime:
        if self._runtime is not None:
            return self._runtime
        self._check_profile()
        if self._runtime_factory is not None:
            self._runtime = self._runtime_factory()
            return self._runtime
        try:
            import os

            import torch  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError as error:
            raise OpenPiZeroUnavailableError(
                "OpenPiZero dependencies are unavailable. Install the pinned open-pi-zero uv environment in an "
                "isolated policy container; imports here are lazy."
            ) from error
        cache_dir = self.profile.application_cache_dir
        if cache_dir:
            os.environ.setdefault("HF_HOME", str(Path(cache_dir) / "huggingface"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_dir) / "transformers"))
        factory = self.profile.entry_point.resolve()
        self._runtime = _OpenPiZeroRuntime(
            torch=torch,
            torch_version=str(getattr(torch, "__version__", "")),
            image_fromarray=Image.fromarray,
            wrapper_factory=lambda profile: factory(profile),
        )
        return self._runtime

    def _ensure_wrapper(self, runtime: _OpenPiZeroRuntime) -> Any:
        self._check_profile()
        if self._wrapper is not None:
            return self._wrapper
        if self._wrapper_factory is not None:
            self._wrapper = self._wrapper_factory(self.profile, runtime)
        else:
            if not Path(self.profile.converted_checkpoint_path).is_file():
                raise OpenPiZeroUnavailableError(
                    "Expected the reviewed converted OpenPiZero weights at %s; refusing network retrieval and "
                    "refusing to load the upstream legacy pickle." % self.profile.converted_checkpoint_path
                )
            missing = self.profile.paligemma.missing_files()
            if missing:
                raise OpenPiZeroUnavailableError(
                    "PaliGemma support files are missing from %s: %s"
                    % (self.profile.paligemma.local_path, ", ".join(missing))
                )
            self._wrapper = runtime.wrapper_factory(self.profile)
        if not callable(getattr(self._wrapper, "infer_action", None)):
            raise OpenPiZeroUnavailableError(
                "Reviewed OpenPiZero wrapper does not expose infer_action(image=..., text=..., proprio=...)."
            )
        eval_method = getattr(self._wrapper, "eval", None)
        if callable(eval_method):
            maybe = eval_method()
            if maybe is not None:
                self._wrapper = maybe
        return self._wrapper

    def _prepare_image(self, runtime: _OpenPiZeroRuntime, image: Any) -> Any:
        convert = getattr(image, "convert", None)
        if callable(convert):
            return convert("RGB")
        try:
            converted = runtime.image_fromarray(image)
        except Exception as error:
            raise PolicyContractError(
                "OpenPiZero requires a current RGB uint8 image or PIL-compatible array for native preprocessing."
            ) from error
        convert = getattr(converted, "convert", None)
        if not callable(convert):
            raise PolicyContractError("OpenPiZero image factory did not return a PIL-compatible image.")
        return convert("RGB")

    # -- native call ------------------------------------------------------------

    def _native_proposal(self, observation: PolicyObservation, identity: FrameIdentity) -> NativeProposal:
        runtime = self._load_runtime()
        wrapper = self._ensure_wrapper(runtime)
        canonical = self._validate_proprio(observation.proprio)
        native_proprio = self.profile.proprio_layout.apply(canonical, self.gripper_convention)
        image = self._prepare_image(runtime, observation.image_history[0])
        cuda = getattr(runtime.torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                cuda.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            pass
        started = time.perf_counter()
        with inference_context(runtime.torch):
            raw = wrapper.infer_action(image=image, text=observation.prompt, proprio=native_proprio)
        wall_seconds = time.perf_counter() - started
        self.last_gpu_peak_memory_bytes = peak_memory_bytes(runtime.torch)
        normalized = rows_from_native(
            raw,
            label="OpenPiZero infer_action",
            expected_rows=self.base_contract.native_proposal_horizon,
        )
        actions = tuple(self.normalizer.denormalize(row) for row in normalized)
        self._last_proprio = canonical
        self._last_native_proprio = native_proprio
        return NativeProposal(
            policy_name=self.base_contract.name,
            actions=actions,
            executed_actions=actions,
            history_length=self.base_contract.required_observation_history,
            normalizer_revision=self.normalizer.revision,
            source_revision=self.profile.checkpoint_revision,
            observation_sha256=identity.key,
            observation_identity_stable=identity.stable,
            backend_calls=1,
            wall_seconds=wall_seconds,
            normalizer_counters=self.normalizer.counters(),
            metadata={
                "model_id": self.profile.model_id,
                "action_chunk": int(self.profile.action_chunk),
                "action_normalization_type": self.profile.action_normalization_type,
                "proprio_layout_id": self.profile.proprio_layout.layout_id,
                "proprio_refreshed_at_timestamp": observation.timestamp,
                "proprio_width": len(native_proprio),
                "proprio_gripper_units": self.profile.proprio_layout.gripper_units,
                "denormalization_boundary": "plumb_policy_action_normalizer",
                "clipped_components": self.normalizer.clipped_components,
                "gpu_peak_memory_bytes": self.last_gpu_peak_memory_bytes,
            },
        )
