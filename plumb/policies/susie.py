"""Pinned, local-only SuSIE and SuSIE_LL goal-conditioned Bridge inference.

Both arms are goal-image policies, so both refuse to run without a real goal
image and both refuse a goal image that is byte-identical to the current or
start frame.  Substituting the start frame for a goal would turn a
goal-conditioned policy into a no-op while still producing plausible actions,
which is exactly the class of silent failure this adapter exists to prevent.

``SuSIE`` runs the subgoal diffusion stage on a declared cadence and conditions
its low-level policy on the generated subgoal.  ``SuSIE_LL`` executes the
low-level goal-conditioned policy directly against the task goal image.

AutoEval's released configuration uses ``gc_bc`` while upstream SuSIE specifies
``gc_ddpm_bc``.  That mismatch is a named arm identity here, never a silent
substitution: the released arm is the replication cell and the corrected
upstream arm is a separately named sensitivity cell.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

from plumb.adapters.contracts import PolicyContract, PolicyObservation

from .contracts import (
    ActionNormalizationType,
    FrameIdentity,
    GripperPolarityConvention,
    NativePolicyAdapter,
    NativeProposal,
    NativeWrapperEntryPoint,
    PolicyActionNormalizer,
    PolicyCertification,
    PolicyContractError,
    PolicyExecutionMode,
    PolicyLoadError,
    UNRESOLVED_BRIDGE_GRIPPER_CONVENTION,
    frame_identity,
    immutable_revision,
    rows_from_native,
)
from .native import (
    AUTOEVAL_EVAL_CONFIG_SOURCE,
    AUTOEVAL_POLICY_SOURCE,
    SUSIE_CONTRACT,
    SUSIE_LL_CONTRACT,
    SUSIE_SOURCE,
)


SUSIE_SUBGOAL_MODEL_ID = "kvablack/susie"
SUSIE_LOW_LEVEL_MODEL_ID = "patreya/gcbc-bridge"
SUSIE_LOW_LEVEL_CHECKPOINT_FILE = "checkpoint_75000"
SUSIE_RELEASED_ARM_ID = "autoeval_released_replication"
SUSIE_UPSTREAM_ARM_ID = "upstream_corrected_sensitivity"
SUSIE_RELEASED_AGENT = "gc_bc"
SUSIE_UPSTREAM_AGENT = "gc_ddpm_bc"
_RNG_STREAM_LABELS: Mapping[str, int] = {"subgoal": 1, "low_level": 2}


class SuSIEUnavailableError(PolicyLoadError):
    """The approved local SuSIE JAX/Flax stack or checkpoint is absent."""


class SuSIESubgoalModel(Protocol):
    """The reviewed subgoal diffusion surface this adapter calls."""

    def sample_subgoal(self, *, image: Any, goal_image: Any, prompt: str, rng: Any) -> Any:
        ...


class SuSIELowLevelModel(Protocol):
    """The reviewed goal-conditioned low-level surface this adapter calls.

    ``sample_actions`` must return the checkpoint's *normalized* action rows;
    PLUMB applies the revisioned denormalization once at one boundary.
    """

    def sample_actions(self, *, image: Any, goal_image: Any, rng: Any) -> Any:
        ...


@dataclass(frozen=True)
class SuSIEArmConfiguration:
    """Which published SuSIE configuration this adapter instance replicates."""

    arm_id: str
    low_level_agent: str
    source_uri: Optional[str] = None

    def review_error(self) -> Optional[str]:
        if self.arm_id == SUSIE_RELEASED_ARM_ID:
            if self.low_level_agent != SUSIE_RELEASED_AGENT:
                return (
                    "the AutoEval released replication arm uses %r; recording %r under that arm identity would "
                    "misattribute a corrected configuration" % (SUSIE_RELEASED_AGENT, self.low_level_agent)
                )
            return None
        if self.arm_id == SUSIE_UPSTREAM_ARM_ID:
            if self.low_level_agent != SUSIE_UPSTREAM_AGENT:
                return "the corrected upstream sensitivity arm uses %r, not %r" % (
                    SUSIE_UPSTREAM_AGENT,
                    self.low_level_agent,
                )
            return None
        return "arm_id must be %r or %r" % (SUSIE_RELEASED_ARM_ID, SUSIE_UPSTREAM_ARM_ID)

    def payload(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "low_level_agent": self.low_level_agent,
            "is_released_replication": self.arm_id == SUSIE_RELEASED_ARM_ID,
            "upstream_configuration_mismatch_disclosed": True,
            "source_uri": self.source_uri,
        }


SUSIE_AUTOEVAL_RELEASED_ARM = SuSIEArmConfiguration(
    arm_id=SUSIE_RELEASED_ARM_ID,
    low_level_agent=SUSIE_RELEASED_AGENT,
    source_uri=AUTOEVAL_EVAL_CONFIG_SOURCE,
)
SUSIE_UPSTREAM_CORRECTED_ARM = SuSIEArmConfiguration(
    arm_id=SUSIE_UPSTREAM_ARM_ID,
    low_level_agent=SUSIE_UPSTREAM_AGENT,
    source_uri=SUSIE_SOURCE,
)


@dataclass(frozen=True)
class SuSIEPolicyProfile:
    """The approval boundary for one local SuSIE or SuSIE_LL snapshot."""

    profile_id: str
    low_level_model_path: str
    low_level_revision: str
    arm: SuSIEArmConfiguration
    low_level_entry_point: NativeWrapperEntryPoint
    goal_image_source: Optional[str] = None
    subgoal_model_path: Optional[str] = None
    subgoal_revision: Optional[str] = None
    subgoal_entry_point: Optional[NativeWrapperEntryPoint] = None
    subgoal_cadence: Optional[int] = None
    jax_version: Optional[str] = None
    low_level_model_id: str = SUSIE_LOW_LEVEL_MODEL_ID
    low_level_license: Optional[str] = None
    subgoal_model_id: str = SUSIE_SUBGOAL_MODEL_ID
    local_files_only: bool = True
    native_unnormalizes: bool = False
    asset_manifest_id: Optional[str] = None
    asset_manifest_sha256: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None

    def review_error(self, *, requires_subgoal: bool) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        if not self.low_level_model_path:
            return "low_level_model_path is required; Hub fetching is disabled"
        if not immutable_revision(self.low_level_revision):
            return "low_level_revision must be an immutable 40-character hexadecimal revision"
        arm_error = self.arm.review_error()
        if arm_error is not None:
            return arm_error
        entry_error = self.low_level_entry_point.review_error()
        if entry_error is not None:
            return entry_error
        if not self.jax_version:
            return "the pinned JAX/Flax (and, for SuSIE, Stable Diffusion) stack version must be recorded"
        if not self.goal_image_source:
            return (
                "susie_goal_image_source_unavailable: a goal-conditioned policy needs a declared goal-image "
                "producer bound to Scenario.goal_image. PLUMB will not substitute the start frame for a goal"
            )
        if requires_subgoal:
            if not self.subgoal_model_path:
                return "SuSIE needs a separate subgoal diffusion checkpoint path"
            if not immutable_revision(self.subgoal_revision):
                return "subgoal_revision must be an immutable 40-character hexadecimal revision"
            if self.subgoal_entry_point is None:
                return "SuSIE needs a pinned reviewed subgoal diffusion entry point"
            subgoal_error = self.subgoal_entry_point.review_error()
            if subgoal_error is not None:
                return subgoal_error
            if (
                isinstance(self.subgoal_cadence, bool)
                or not isinstance(self.subgoal_cadence, int)
                or self.subgoal_cadence < 1
            ):
                return (
                    "SuSIE's subgoal cadence must come from the pinned released wrapper as a positive integer, "
                    "not be inferred at runtime"
                )
        else:
            if self.subgoal_model_path or self.subgoal_revision or self.subgoal_entry_point is not None:
                return (
                    "SuSIE_LL executes the low-level goal-conditioned policy directly and has no subgoal "
                    "diffusion stage; remove the subgoal configuration or use the SuSIE adapter"
                )
            if self.subgoal_cadence is not None:
                return "SuSIE_LL has no subgoal cadence"
        if not self.local_files_only:
            return "SuSIE loader is local-only; network retrieval is not permitted"
        if self.native_unnormalizes:
            return (
                "PLUMB applies its own revisioned action normalizer at exactly one documented boundary; letting "
                "the released wrapper also unnormalize would apply the transform twice"
            )
        return None


@dataclass(frozen=True)
class _SuSIERuntime:
    """Injection seam for the lazily imported JAX/Flax stack."""

    jax_version: str
    prng_key: Callable[[int], Any]
    fold_in: Callable[[Any, int], Any]
    low_level_factory: Callable[[SuSIEPolicyProfile], Any]
    subgoal_factory: Optional[Callable[[SuSIEPolicyProfile], Any]] = None


class _SuSIEAdapterBase(NativePolicyAdapter):
    """Shared goal-image plumbing for both SuSIE arms."""

    source_urls = (SUSIE_SOURCE, AUTOEVAL_POLICY_SOURCE, AUTOEVAL_EVAL_CONFIG_SOURCE)
    requires_subgoal: bool = False

    def __init__(
        self,
        profile: SuSIEPolicyProfile,
        *,
        normalizer: PolicyActionNormalizer,
        certification: Optional[PolicyCertification] = None,
        execution_mode: PolicyExecutionMode = PolicyExecutionMode.CERTIFIED,
        gripper_convention: GripperPolarityConvention = UNRESOLVED_BRIDGE_GRIPPER_CONVENTION,
        runtime_factory: Optional[Callable[[], _SuSIERuntime]] = None,
        low_level_factory: Optional[Callable[[SuSIEPolicyProfile, _SuSIERuntime], Any]] = None,
        subgoal_factory: Optional[Callable[[SuSIEPolicyProfile, _SuSIERuntime], Any]] = None,
    ) -> None:
        super().__init__(
            normalizer=normalizer,
            certification=certification,
            execution_mode=execution_mode,
            gripper_convention=gripper_convention,
        )
        if normalizer.statistics.normalization_type is not ActionNormalizationType.NORMAL:
            raise PolicyContractError(
                "SuSIE's low-level statistics are the mean/standard-deviation family; 'bounds' is OpenPiZero's "
                "adapter-specific setting."
            )
        self.profile = profile
        self._runtime_factory = runtime_factory
        self._low_level_factory = low_level_factory
        self._subgoal_factory = subgoal_factory
        self._runtime: Optional[_SuSIERuntime] = None
        self._low_level: Any = None
        self._subgoal_model: Any = None
        self._goal_key: Optional[str] = None
        self._start_frame_key: Optional[str] = None
        self._subgoal: Any = None
        self._subgoal_identity: Optional[str] = None
        self._subgoal_step: Optional[int] = None
        self._subgoal_refreshes = 0

    # -- capability -------------------------------------------------------------

    def _profile_errors(self) -> Tuple[str, ...]:
        error = self.profile.review_error(requires_subgoal=self.requires_subgoal)
        return () if error is None else (error,)

    def _availability_error(self) -> Optional[str]:
        missing: List[str] = []
        if self._low_level_factory is None and not Path(self.profile.low_level_model_path).exists():
            missing.append("low-level checkpoint %s" % self.profile.low_level_model_path)
        if (
            self.requires_subgoal
            and self._subgoal_factory is None
            and self.profile.subgoal_model_path
            and not Path(self.profile.subgoal_model_path).exists()
        ):
            missing.append("subgoal diffusion checkpoint %s" % self.profile.subgoal_model_path)
        return None if not missing else "absent local components: " + ", ".join(missing)

    def _capability_details(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile.profile_id,
            "arm": self.profile.arm.payload(),
            "goal_image_required": True,
            "goal_image_source": self.profile.goal_image_source,
            "start_frame_substituted_for_goal": False,
            "low_level_model_id": self.profile.low_level_model_id,
            "low_level_model_path": self.profile.low_level_model_path,
            "low_level_revision": self.profile.low_level_revision,
            "low_level_license": self.profile.low_level_license,
            "low_level_checkpoint_file": SUSIE_LOW_LEVEL_CHECKPOINT_FILE,
            "subgoal_stage": bool(self.requires_subgoal),
            "subgoal_model_id": self.profile.subgoal_model_id if self.requires_subgoal else None,
            "subgoal_model_path": self.profile.subgoal_model_path,
            "subgoal_revision": self.profile.subgoal_revision,
            "subgoal_cadence": self.profile.subgoal_cadence,
            "jax_version": self.profile.jax_version,
            "local_files_only": True,
            "asset_manifest_id": self.profile.asset_manifest_id,
            "asset_manifest_sha256": self.profile.asset_manifest_sha256,
            "runtime_lock_id": self.profile.runtime_lock_id,
            "runtime_lock_sha256": self.profile.runtime_lock_sha256,
        }

    # -- durable state ----------------------------------------------------------

    def _reset_native(self, seed: int) -> None:
        self._goal_key = None
        self._start_frame_key = None
        self._subgoal = None
        self._subgoal_identity = None
        self._subgoal_step = None
        self._subgoal_refreshes = 0

    @property
    def subgoal_pixels_available(self) -> bool:
        """False only after a resume dropped an opaque generated subgoal image."""

        return self._subgoal is not None or self._subgoal_identity is None

    def _snapshot_extra(self) -> Dict[str, Any]:
        return {
            "arm_id": self.profile.arm.arm_id,
            "low_level_agent": self.profile.arm.low_level_agent,
            "goal_identity": self._goal_key,
            "start_frame_identity": self._start_frame_key,
            "subgoal": {
                "identity": self._subgoal_identity,
                "generated_at_step": self._subgoal_step,
                "refreshes": int(self._subgoal_refreshes),
                "cadence": self.profile.subgoal_cadence,
                "pixels_available": self.subgoal_pixels_available,
            },
        }

    def _restore_extra(self, payload: Mapping[str, Any]) -> None:
        if str(payload.get("arm_id")) != self.profile.arm.arm_id:
            raise PolicyContractError(
                "SuSIE snapshot arm %r does not match the live arm %r; arms are separate policy identities."
                % (payload.get("arm_id"), self.profile.arm.arm_id)
            )
        if str(payload.get("low_level_agent")) != self.profile.arm.low_level_agent:
            raise PolicyContractError("SuSIE snapshot low-level agent does not match the live configuration.")
        goal = payload.get("goal_identity")
        self._goal_key = None if goal is None else str(goal)
        start = payload.get("start_frame_identity")
        self._start_frame_key = None if start is None else str(start)
        subgoal = payload.get("subgoal")
        if not isinstance(subgoal, Mapping):
            raise PolicyContractError("SuSIE snapshot must carry its subgoal record.")
        if subgoal.get("cadence") != self.profile.subgoal_cadence:
            raise PolicyContractError("SuSIE snapshot subgoal cadence does not match the live profile.")
        identity = subgoal.get("identity")
        self._subgoal_identity = None if identity is None else str(identity)
        step = subgoal.get("generated_at_step")
        self._subgoal_step = None if step is None else int(step)
        self._subgoal_refreshes = int(subgoal.get("refreshes", 0))
        # A diffusion subgoal is an opaque image: a JSON checkpoint records its
        # hash, never its pixels.  Dropping it here is what makes the
        # mid-interval refusal below honest instead of silently regenerating a
        # different subgoal.
        self._subgoal = None

    # -- goal-image rules -------------------------------------------------------

    def _check_goal_image(self, observation: PolicyObservation) -> None:
        name = self.base_contract.name
        goal = observation.goal_image
        if goal is None:
            raise PolicyContractError(
                "%s requires an explicit goal image on every observation. PLUMB will not substitute the start "
                "frame, the current frame, or a blank image for a goal." % name
            )
        goal_identity = frame_identity(goal)
        current_identity = frame_identity(observation.image_history[-1])
        if goal_identity.key == current_identity.key:
            raise PolicyContractError(
                "%s received a goal image identical to its current observation frame. A start or current frame "
                "is never a goal image." % name
            )
        if self._start_frame_key is not None and goal_identity.key == self._start_frame_key:
            raise PolicyContractError(
                "%s received a goal image identical to this episode's start frame; that is a substituted goal, "
                "not a goal." % name
            )
        if self._goal_key is not None and goal_identity.key != self._goal_key:
            raise PolicyContractError(
                "%s goal image changed mid-episode (%s -> %s); a changed goal is a different scenario."
                % (name, self._goal_key, goal_identity.key)
            )

    # -- runtime ----------------------------------------------------------------

    def _check_profile(self) -> None:
        error = self.profile.review_error(requires_subgoal=self.requires_subgoal)
        if error is not None:
            raise SuSIEUnavailableError("Refusing to load %s: %s." % (self.base_contract.name, error))

    def _load_runtime(self) -> _SuSIERuntime:
        if self._runtime is not None:
            return self._runtime
        self._check_profile()
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
            if runtime.jax_version != self.profile.jax_version:
                raise SuSIEUnavailableError(
                    "Injected SuSIE runtime is JAX %r; the profile pins %r."
                    % (runtime.jax_version, self.profile.jax_version)
                )
            self._runtime = runtime
            return runtime
        try:
            import jax  # type: ignore
        except ImportError as error:
            raise SuSIEUnavailableError(
                "SuSIE dependencies are unavailable. Install the pinned JAX/Flax (and Stable Diffusion, for the "
                "subgoal stage) image in an isolated policy container; imports here are lazy."
            ) from error
        installed = str(getattr(jax, "__version__", ""))
        if installed != self.profile.jax_version:
            raise SuSIEUnavailableError(
                "Installed JAX %r is incompatible; this SuSIE profile pins %r." % (installed, self.profile.jax_version)
            )
        low_level_factory = self.profile.low_level_entry_point.resolve()
        subgoal_factory = None
        if self.requires_subgoal and self.profile.subgoal_entry_point is not None:
            subgoal_factory = self.profile.subgoal_entry_point.resolve()
        self._runtime = _SuSIERuntime(
            jax_version=installed,
            prng_key=lambda seed: jax.random.PRNGKey(seed),
            fold_in=lambda key, value: jax.random.fold_in(key, value),
            low_level_factory=lambda profile: low_level_factory(profile),
            subgoal_factory=None if subgoal_factory is None else (lambda profile: subgoal_factory(profile)),
        )
        return self._runtime

    def _ensure_models(self, runtime: _SuSIERuntime) -> Tuple[Any, Any]:
        self._check_profile()
        if self._low_level is None:
            self._low_level = (
                self._low_level_factory(self.profile, runtime)
                if self._low_level_factory is not None
                else runtime.low_level_factory(self.profile)
            )
            if not callable(getattr(self._low_level, "sample_actions", None)):
                raise SuSIEUnavailableError(
                    "Reviewed SuSIE low-level policy does not expose sample_actions(image=..., goal_image=..., rng=...)."
                )
        if self.requires_subgoal and self._subgoal_model is None:
            if self._subgoal_factory is not None:
                self._subgoal_model = self._subgoal_factory(self.profile, runtime)
            elif runtime.subgoal_factory is not None:
                self._subgoal_model = runtime.subgoal_factory(self.profile)
            else:
                raise SuSIEUnavailableError("SuSIE's reviewed subgoal diffusion factory is unavailable.")
            if not callable(getattr(self._subgoal_model, "sample_subgoal", None)):
                raise SuSIEUnavailableError(
                    "Reviewed SuSIE subgoal model does not expose sample_subgoal(image=..., goal_image=..., "
                    "prompt=..., rng=...)."
                )
        return self._subgoal_model, self._low_level

    def _stream_rng(self, runtime: _SuSIERuntime, stream: str) -> Any:
        if self._seed is None:
            raise PolicyContractError(
                "%s requires reset(seed) before a native call so its JAX RNG streams are pinned and restorable."
                % self.base_contract.name
            )
        label = _RNG_STREAM_LABELS[stream]
        return runtime.fold_in(runtime.fold_in(runtime.prng_key(int(self._seed)), int(self._step)), label)

    def _conditioning_image(self, runtime: _SuSIERuntime, subgoal_model: Any, observation: PolicyObservation) -> Tuple[Any, Dict[str, Any]]:
        """The image the low-level policy is conditioned on for this tick."""

        goal = observation.goal_image
        if not self.requires_subgoal:
            return goal, {"conditioning": "task_goal_image", "subgoal_refreshed": False}
        cadence = int(self.profile.subgoal_cadence or 0)
        if cadence < 1:
            raise PolicyContractError("SuSIE subgoal cadence is not configured.")
        if self._step % cadence == 0:
            rng = self._stream_rng(runtime, "subgoal")
            subgoal = subgoal_model.sample_subgoal(
                image=observation.image_history[-1], goal_image=goal, prompt=observation.prompt, rng=rng
            )
            if subgoal is None:
                raise PolicyContractError("SuSIE subgoal diffusion returned no subgoal image.")
            self._subgoal = subgoal
            self._subgoal_identity = frame_identity(subgoal).key
            self._subgoal_step = int(self._step)
            self._subgoal_refreshes += 1
            return subgoal, {
                "conditioning": "generated_subgoal",
                "subgoal_refreshed": True,
                "subgoal_step": self._subgoal_step,
            }
        if self._subgoal is None:
            if self._subgoal_identity is None:
                raise PolicyContractError(
                    "SuSIE reached control step %d without a subgoal; the first call of an episode must land on "
                    "a cadence boundary." % self._step
                )
            raise PolicyContractError(
                "SuSIE resumed at control step %d, inside the subgoal interval that began at step %s. The "
                "generated subgoal image (%s) is not restorable from a JSON checkpoint, and regenerating one "
                "here would silently change the policy; resume only on a subgoal cadence boundary."
                % (self._step, self._subgoal_step, self._subgoal_identity)
            )
        return self._subgoal, {
            "conditioning": "cached_subgoal",
            "subgoal_refreshed": False,
            "subgoal_step": self._subgoal_step,
        }

    # -- native call ------------------------------------------------------------

    def _native_proposal(self, observation: PolicyObservation, identity: FrameIdentity) -> NativeProposal:
        runtime = self._load_runtime()
        subgoal_model, low_level = self._ensure_models(runtime)
        if self._start_frame_key is None:
            self._start_frame_key = identity.key
        if self._goal_key is None:
            self._goal_key = frame_identity(observation.goal_image).key
        conditioning, conditioning_metadata = self._conditioning_image(runtime, subgoal_model, observation)
        rng = self._stream_rng(runtime, "low_level")
        started = time.perf_counter()
        raw = low_level.sample_actions(image=observation.image_history[-1], goal_image=conditioning, rng=rng)
        wall_seconds = time.perf_counter() - started
        normalized = rows_from_native(
            raw,
            label="%s sample_actions" % self.base_contract.name,
            expected_rows=self.base_contract.native_proposal_horizon,
        )
        actions = tuple(self.normalizer.denormalize(row) for row in normalized)
        metadata: Dict[str, Any] = {
            "arm_id": self.profile.arm.arm_id,
            "low_level_agent": self.profile.arm.low_level_agent,
            "goal_identity": self._goal_key,
            "goal_image_source": self.profile.goal_image_source,
            "subgoal_identity": self._subgoal_identity,
            "subgoal_refreshes": int(self._subgoal_refreshes),
            "subgoal_cadence": self.profile.subgoal_cadence,
            "denormalization_boundary": "plumb_policy_action_normalizer",
        }
        metadata.update(conditioning_metadata)
        return NativeProposal(
            policy_name=self.base_contract.name,
            actions=actions,
            executed_actions=actions,
            history_length=self.base_contract.required_observation_history,
            normalizer_revision=self.normalizer.revision,
            source_revision=self.profile.low_level_revision,
            observation_sha256=identity.key,
            observation_identity_stable=identity.stable,
            backend_calls=1 + (1 if conditioning_metadata.get("subgoal_refreshed") else 0),
            wall_seconds=wall_seconds,
            normalizer_counters=self.normalizer.counters(),
            metadata=metadata,
        )


class SuSIEPolicyAdapter(_SuSIEAdapterBase):
    """SuSIE: subgoal diffusion on a declared cadence plus the low-level policy."""

    base_contract: PolicyContract = SUSIE_CONTRACT
    requires_subgoal = True


class SuSIELowLevelPolicyAdapter(_SuSIEAdapterBase):
    """SuSIE_LL: the low-level goal-conditioned policy executed directly."""

    base_contract: PolicyContract = SUSIE_LL_CONTRACT
    requires_subgoal = False
