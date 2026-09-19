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

import hashlib
import inspect
import os
import subprocess
import sys
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
# The Hub snapshot preserves the historical GCS step only in
# checkpoint/commit_success.txt. Its actual restore target is this direct file;
# pointing Flax at the containing directory with default prefix="checkpoint_"
# silently returns the initialized target.
SUSIE_LOW_LEVEL_CHECKPOINT_FILE = "checkpoint/checkpoint"
SUSIE_LOW_LEVEL_MODEL_REVISION = "1a4c15dd9ad780a257e9494f0fac79cbe8e64793"
SUSIE_LOW_LEVEL_CHECKPOINT_SHA256 = "80b354db7a05d514d6df5b5a4395469902b0ff362d383edeeb4abb8c1c9d9e33"
SUSIE_LOW_LEVEL_README_SHA256 = "d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17"
# AutoEval's pinned README names rail-berkeley/soar/model_training—not the
# similarly shaped BridgeData V2 jaxrl_m tree. The latter rejects AutoEval's
# std_parameterization policy keyword and cannot be substituted.
SUSIE_GCBC_SOURCE_COMMIT = "eabd5f16a856e484884a22e257a941bb358cea08"
SUSIE_GCBC_SOURCE_SUBDIR = "model_training"
SUSIE_GCBC_SOURCE_REQUIREMENTS = {
    "jax": "0.4.20",
    "flax": "0.7.5",
    "distrax": "0.1.5",
    "numpy": "1.24.3",
    "tensorflow": "2.15.0",
    "orbax-checkpoint": "0.3.5",
    "scipy": "1.12.0",
}
# This remains the source for the separate upstream gc_ddpm_bc sensitivity
# arm, not for the released AutoEval gc_bc checkpoint.
SUSIE_BRIDGE_DATA_SOURCE_COMMIT = "bc60a35b701a12021c8c95e9d8601274d3acd928"
SUSIE_AUTOEVAL_CONFIG_SHA256 = "4a474fdb183956e674a45450f12afb924a91195bffbd009e67b5e95dc7b16564"
SUSIE_AUTOEVAL_NORMALIZER_SHA256 = "9c8d48caa20c0e8e43888819d0410918a107b3be971cc2ec046acbb373ba7f3d"
SUSIE_RELEASED_ARM_ID = "autoeval_released_replication"
SUSIE_UPSTREAM_ARM_ID = "upstream_corrected_sensitivity"
SUSIE_RELEASED_AGENT = "gc_bc"
SUSIE_UPSTREAM_AGENT = "gc_ddpm_bc"
_RNG_STREAM_LABELS: Mapping[str, int] = {"subgoal": 1, "low_level": 2}


class SuSIEUnavailableError(PolicyLoadError):
    """The approved local SuSIE JAX/Flax stack or checkpoint is absent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _verify_clean_source(root: Path) -> Path:
    """Validate the exact SOAR checkout that supplies AutoEval's ``jaxrl_m``."""

    module_root = root / SUSIE_GCBC_SOURCE_SUBDIR
    required = ("jaxrl_m/agents/__init__.py", "jaxrl_m/agents/continuous/gc_bc.py", "jaxrl_m/vision/__init__.py")
    if not root.is_dir() or any(not (module_root / name).is_file() for name in required):
        raise SuSIEUnavailableError("Pinned SOAR model_training source checkout is incomplete at %s." % root)
    try:
        head = subprocess.run(("git", "-C", str(root), "rev-parse", "HEAD"), capture_output=True, text=True, check=False, timeout=10)
        status = subprocess.run(("git", "-C", str(root), "status", "--porcelain"), capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise SuSIEUnavailableError("Cannot inspect SOAR source checkout %s." % root) from error
    if head.returncode != 0 or head.stdout.strip() != SUSIE_GCBC_SOURCE_COMMIT:
        raise SuSIEUnavailableError("SOAR checkout must be pinned to %s." % SUSIE_GCBC_SOURCE_COMMIT)
    if status.returncode != 0 or status.stdout.strip():
        raise SuSIEUnavailableError("SOAR checkout is dirty; refusing mutable gc_bc source.")
    return module_root.resolve()


def _tree_digest(jax_module: Any, value: Any) -> str:
    """Hash initialized/restored tensor leaves without serializing code objects."""

    digest = hashlib.sha256()
    leaves = jax_module.tree_util.tree_leaves(value)
    for leaf in leaves:
        array = jax_module.device_get(leaf)
        shape = tuple(getattr(array, "shape", ()))
        dtype = str(getattr(array, "dtype", ""))
        digest.update((repr(shape) + "|" + dtype + "|").encode("utf-8"))
        raw = getattr(array, "tobytes", None)
        if callable(raw):
            digest.update(raw())
        else:
            digest.update(repr(array).encode("utf-8"))
    return digest.hexdigest()


def _strict_state_tree(expected: Any, actual: Any, *, path: str = "params") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise SuSIEUnavailableError("Checkpoint %s must be a mapping." % path)
        expected_keys = set(str(key) for key in expected)
        actual_keys = set(str(key) for key in actual)
        if expected_keys != actual_keys:
            raise SuSIEUnavailableError(
                "Checkpoint %s keys differ from initialized source model; missing=%s extra=%s."
                % (path, sorted(expected_keys - actual_keys), sorted(actual_keys - expected_keys))
            )
        for key, value in expected.items():
            _strict_state_tree(value, actual[str(key)], path=path + "." + str(key))


def _strict_param_shapes(jax_module: Any, expected: Any, restored: Any) -> None:
    expected_leaves, expected_tree = jax_module.tree_util.tree_flatten(expected)
    restored_leaves, restored_tree = jax_module.tree_util.tree_flatten(restored)
    if expected_tree != restored_tree or len(expected_leaves) != len(restored_leaves):
        raise SuSIEUnavailableError("Restored parameter pytree does not exactly match initialized source structure.")
    for index, (before, after) in enumerate(zip(expected_leaves, restored_leaves)):
        before_value, after_value = jax_module.device_get(before), jax_module.device_get(after)
        if tuple(getattr(before_value, "shape", ())) != tuple(getattr(after_value, "shape", ())):
            raise SuSIEUnavailableError("Restored parameter leaf %d shape differs from initialized source model." % index)
        if str(getattr(before_value, "dtype", "")) != str(getattr(after_value, "dtype", "")):
            raise SuSIEUnavailableError("Restored parameter leaf %d dtype differs from initialized source model." % index)


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
    low_level_license_path: Optional[str] = None
    subgoal_model_id: str = SUSIE_SUBGOAL_MODEL_ID
    local_files_only: bool = True
    native_unnormalizes: bool = False
    inference_params_only: bool = False
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


class _AutoEvalGCBCBridgeLowLevel:
    """Direct, reviewed AutoEval ``gc_bc`` inference over local Bridge assets.

    It deliberately contains no SuSIE diffusion dependency.  The object
    exposes the production adapter's normalized-row boundary: dimensions 0–5
    remain model-normalized while dimension 6 is source-thresholded to the
    AutoEval physical binary gripper convention.  A normalizer with a false
    seventh mask then applies the one allowed affine transform.
    """

    def __init__(self, profile: "SuSIEPolicyProfile") -> None:
        if profile.arm.arm_id != SUSIE_RELEASED_ARM_ID or profile.arm.low_level_agent != SUSIE_RELEASED_AGENT:
            raise SuSIEUnavailableError("The direct gc_bc loader is only valid for the named AutoEval released replication arm.")
        if profile.low_level_model_id != SUSIE_LOW_LEVEL_MODEL_ID or profile.low_level_revision != SUSIE_LOW_LEVEL_MODEL_REVISION:
            raise SuSIEUnavailableError("Direct gc_bc loader requires the pinned patreya/gcbc-bridge artifact revision.")
        checkpoint = Path(profile.low_level_model_path)
        if checkpoint.name != "checkpoint" or checkpoint.parent.name != "checkpoint":
            raise SuSIEUnavailableError(
                "gc_bc local_model_path must be the immutable direct file .../checkpoint/checkpoint, not checkpoint_75000 or a directory."
            )
        if not checkpoint.is_file() or _sha256_file(checkpoint) != SUSIE_LOW_LEVEL_CHECKPOINT_SHA256:
            raise SuSIEUnavailableError("gc_bc checkpoint is absent or does not match its immutable SHA-256.")
        readme = Path(profile.low_level_license_path or checkpoint.parents[1] / "README.md")
        if not readme.is_file() or _sha256_file(readme) != SUSIE_LOW_LEVEL_README_SHA256:
            raise SuSIEUnavailableError("gc_bc publisher MIT README is absent or does not match its immutable snapshot hash.")
        if "license: mit" not in readme.read_text(encoding="utf-8").lower():
            raise SuSIEUnavailableError("gc_bc immutable README does not declare the publisher's MIT license.")
        source_value = profile.low_level_entry_point.source_repo_path
        if not source_value:
            raise SuSIEUnavailableError("Direct gc_bc loader requires a clean pinned SOAR source checkout path.")
        source_root = Path(str(source_value)).resolve()
        module_root = _verify_clean_source(source_root)
        if str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))
        try:
            import flax  # type: ignore
            import jax  # type: ignore
            import numpy as np  # type: ignore
            from flax import serialization  # type: ignore
            from flax.training import checkpoints  # type: ignore
            from jaxrl_m.agents import agents  # type: ignore
            from jaxrl_m.vision import encoders  # type: ignore
        except ImportError as error:
            raise SuSIEUnavailableError("Direct gc_bc runtime lacks pinned BridgeData JAX/Flax dependencies.") from error
        if str(getattr(jax, "__version__", "")).split("+", 1)[0] != str(profile.jax_version or "").split("+", 1)[0]:
            raise SuSIEUnavailableError("Installed JAX does not match the profile's recorded runtime version.")
        agent_class = agents.get(SUSIE_RELEASED_AGENT) if hasattr(agents, "get") else agents[SUSIE_RELEASED_AGENT]
        encoder_factory = encoders["resnetv1-34"]
        encoder_source = encoder_factory
        while hasattr(encoder_source, "func"):
            encoder_source = encoder_source.func
        if not _under(Path(inspect.getfile(agent_class)), module_root) or not _under(Path(inspect.getfile(encoder_source)), module_root):
            raise SuSIEUnavailableError("Imported jaxrl_m modules do not resolve inside the pinned SOAR model_training checkout.")
        self._jax = jax
        self._np = np
        self.runtime_jax_version = str(getattr(jax, "__version__", ""))
        self.runtime_flax_version = str(getattr(flax, "__version__", ""))
        self.source_runtime_deviation = {
            "source_repository": "rail-berkeley/soar",
            "source_commit": SUSIE_GCBC_SOURCE_COMMIT,
            "source_subdirectory": SUSIE_GCBC_SOURCE_SUBDIR,
            "source_required_versions": dict(SUSIE_GCBC_SOURCE_REQUIREMENTS),
            "actual_jax_version": self.runtime_jax_version,
            "actual_flax_version": self.runtime_flax_version,
            "core_jax_flax_match": (
                self.runtime_jax_version.split("+", 1)[0] == SUSIE_GCBC_SOURCE_REQUIREMENTS["jax"]
                and self.runtime_flax_version.split("+", 1)[0] == SUSIE_GCBC_SOURCE_REQUIREMENTS["flax"]
            ),
            "full_runtime_tuple_verified": False,
        }
        # Exact AutoEval jaxrl_gc_policy_kwargs construction.
        encoder = encoder_factory(pooling_method="avg", add_spatial_coordinates=False, act="swish")
        batch = {
            "observations": {"proprio": np.zeros((1, 7)), "image": np.zeros((1, 256, 256, 3))},
            "goals": {"image": np.zeros((1, 256, 256, 3))},
            "actions": np.zeros((1, 7)),
        }
        root_rng = jax.random.PRNGKey(42)
        _, construct_rng = jax.random.split(root_rng)
        agent = agent_class.create(
            rng=construct_rng,
            observations=batch["observations"],
            goals=batch["goals"],
            actions=batch["actions"],
            encoder_def=encoder,
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
            decay_steps=int(2e6),
            network_kwargs={"hidden_dims": (256, 256, 256), "dropout_rate": 0.1},
            policy_kwargs={
                "tanh_squash_distribution": False,
                "std_parameterization": "fixed",
                "fixed_std": [1, 1, 1, 1, 1, 1, 0.1],
            },
        )
        before = _tree_digest(jax, agent.state.params)
        if profile.inference_params_only:
            raw_checkpoint = checkpoints.restore_checkpoint(str(checkpoint), target=None)
            if not isinstance(raw_checkpoint, Mapping) or set(raw_checkpoint) != {"state"}:
                raise SuSIEUnavailableError("Inference-only gc_bc restore requires raw checkpoint top-level keys exactly {'state'}.")
            raw_state = raw_checkpoint["state"]
            expected_state_keys = {"opt_states", "params", "rng", "step", "target_params"}
            if not isinstance(raw_state, Mapping) or set(raw_state) != expected_state_keys:
                raise SuSIEUnavailableError("Inference-only gc_bc restore raw state keys do not match the reviewed checkpoint schema.")
            if raw_state["target_params"] is not None or agent.state.target_params is not None:
                raise SuSIEUnavailableError("Inference-only gc_bc restore requires reviewed target_params=None on both checkpoint and initialized source state.")
            expected_params = serialization.to_state_dict(agent.state.params)
            _strict_state_tree(expected_params, raw_state["params"])
            restored_params = serialization.from_state_dict(agent.state.params, raw_state["params"])
            _strict_param_shapes(jax, agent.state.params, restored_params)
            restored = agent.replace(state=agent.state.replace(params=restored_params, target_params=None))
            restore_method = "inference_params_only_source_checkpoint_no_optimizer_restore"
            raw_keys = {"top_level": sorted(raw_checkpoint), "state": sorted(raw_state)}
            optimizer_excluded = True
        else:
            restored = checkpoints.restore_checkpoint(str(checkpoint), target=agent)
            restore_method = "full_source_checkpoint_restore"
            raw_keys = None
            optimizer_excluded = False
        after = _tree_digest(jax, restored.state.params)
        if before == after:
            raise SuSIEUnavailableError(
                "gc_bc restore left the initialized target byte-identical; refusing a silent no-checkpoint diagnostic."
            )
        self._agent = restored
        self.restore_digest = {
            "initialized_params": before,
            "restored_params": after,
            "changed": True,
            "method": restore_method,
            "raw_checkpoint_keys": raw_keys,
            "optimizer_state_excluded": optimizer_excluded,
            "inference_only_not_resumable": bool(profile.inference_params_only),
        }
        self.last_model_normalized: Optional[Tuple[float, ...]] = None
        self.last_wrapper_boundary: Optional[Tuple[float, ...]] = None

    def sample_actions(self, *, image: Any, goal_image: Any, rng: Any) -> Any:
        current = self._np.asarray(image)
        goal = self._np.asarray(goal_image)
        for label, value in (("current", current), ("goal", goal)):
            if tuple(getattr(value, "shape", ())) != (256, 256, 3) or str(getattr(value, "dtype", "")) != "uint8":
                raise PolicyContractError("AutoEval gc_bc %s image must be exact uint8 (256, 256, 3)." % label)
        # This is AutoEval GCPolicy's deterministic path. The agent's argmax
        # branch does not consume a PRNG seed; the adapter's RNG remains logged
        # at the call boundary but is intentionally not substituted here.
        sampler = getattr(self._agent, "sample_actions")
        raw = sampler(
            {"image": current[self._np.newaxis, ...]},
            {"image": goal[self._np.newaxis, ...]},
            temperature=0.0,
            argmax=True,
            seed=None,
        )
        if isinstance(raw, tuple):
            if len(raw) != 2:
                raise PolicyContractError("AutoEval gc_bc tuple output must be exactly (actions, action_mode).")
            raw, action_mode = raw
            mode_values = self._np.asarray(self._jax.device_get(action_mode))
            if tuple(getattr(mode_values, "shape", ())) != (1, 7) or not self._np.isfinite(mode_values).all():
                raise PolicyContractError("AutoEval gc_bc action_mode must be one finite (1, 7) row.")
        values = self._np.asarray(self._jax.device_get(raw))
        if tuple(getattr(values, "shape", ())) != (1, 7) or not self._np.isfinite(values).all():
            raise PolicyContractError("AutoEval gc_bc must return one finite normalized (1, 7) action.")
        row = values[0]
        self.last_model_normalized = tuple(float(value) for value in row)
        # Match AutoEval unnormalize_actions exactly at the gripper boundary;
        # first six values are left for PLUMB's revisioned normalizer.
        boundary = tuple(float(value) for value in row[:6]) + (1.0 if float(row[6]) > 0.0 else 0.0,)
        self.last_wrapper_boundary = boundary
        return [list(boundary)]


def build_autoeval_gcbc_bridge_low_level(profile: "SuSIEPolicyProfile") -> _AutoEvalGCBCBridgeLowLevel:
    """Pinned real loader for the published AutoEval ``gc_bc`` component."""

    return _AutoEvalGCBCBridgeLowLevel(profile)


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
            "inference_params_only": self.profile.inference_params_only,
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
