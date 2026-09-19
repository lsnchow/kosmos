"""Pinned local SuSIE and SuSIE_LL source paths.

The two policy identities that are often both called "SuSIE" are not
interchangeable.  AutoEval's released replication wrapper couples SuSIE image
subgoals with a ``gc_bc`` low-level policy.  SuSIE's own repository instead
states that its low-level policy is ``gc_ddpm_bc`` with a four-action horizon
and ``delta_goals``.  This module represents those as different immutable
arms; it never upgrades the released replication arm in place.

All imports are lazy and every source loader is local-only.  The high-level
weights use the published Flax msgpack path; the low-level checkpoint uses
the reviewed Flax checkpoint restore API rather than Python pickle.  Neither
arm is a Gate-A/B-qualified policy evaluation merely because it loads.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Mapping, Optional, Tuple

from plumb.adapters.contracts import CapabilityResult, CapabilityStatus, PolicyContract, PolicyObservation

from ..contracts import AUTOEVAL_POLICY_SOURCE_COMMIT, PolicyContractError, PolicyLoadError


SUSIE_SOURCE_COMMIT = "8177f63332a3905202c122310a92c51b8ff14280"
SUSIE_BRIDGE_DATA_SOURCE_COMMIT = "bc60a35b701a12021c8c95e9d8601274d3acd928"
SUSIE_HIGH_LEVEL_MODEL_ID = "kvablack/susie"
SUSIE_HIGH_LEVEL_MODEL_REVISION = "83d3c21fda79550fc5431f8a42c07835c9ec3f95"
SUSIE_HIGH_LEVEL_UNET_SHA256 = "371d791b6e605a4a6d927fe704172e5023e83661affb1693ff421c2095f437a2"
SUSIE_HIGH_LEVEL_UNET_CONFIG_SHA256 = "5dd4c5adb5bb76946e821c1a4ab362c0686861311ddb152926cf66052d0b2cf4"
SUSIE_LOW_LEVEL_GCBC_MODEL_ID = "patreya/gcbc-bridge"
SUSIE_LOW_LEVEL_GCBC_MODEL_REVISION = "1a4c15dd9ad780a257e9494f0fac79cbe8e64793"
SUSIE_LOW_LEVEL_GCBC_CHECKPOINT_SHA256 = "80b354db7a05d514d6df5b5a4395469902b0ff362d383edeeb4abb8c1c9d9e33"
SUSIE_REPLICATION_ARM_ID = "susie-autoeval-gc-bc-replication"
SUSIE_CORRECTED_SENSITIVITY_ARM_ID = "susie-upstream-gc-ddpm-bc-sensitivity"
SUSIE_LL_REPLICATION_ARM_ID = "susie-ll-autoeval-gc-bc-replication"
SUSIE_LL_CORRECTED_SENSITIVITY_ARM_ID = "susie-ll-upstream-gc-ddpm-bc-sensitivity"
SUSIE_AUTOEVAL_ALGORITHM = "gc_bc"
SUSIE_UPSTREAM_ALGORITHM = "gc_ddpm_bc"
SUSIE_IMAGE_SIZE = 256
SUSIE_SUBGOAL_CADENCE = 20
SUSIE_UPSTREAM_ACTION_HORIZON = 4
SUSIE_JAX_VERSION = "0.4.11"
SUSIE_FLAX_VERSION = "0.7.0"
SUSIE_DIFFUSERS_VERSION = "0.18.2"
SUSIE_TRANSFORMERS_VERSION = "4.33.1"
SUSIE_STABLE_DIFFUSION_MODEL_ID = "lodestones/stable-diffusion-v1-5-flax"

SUSIE_AUTOEVAL_POLICY_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + AUTOEVAL_POLICY_SOURCE_COMMIT
    + "/auto_eval/robot/policy.py#L454-L652"
)
SUSIE_AUTOEVAL_CONFIG_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + AUTOEVAL_POLICY_SOURCE_COMMIT
    + "/scripts/configs/eval_config.py#L56-L111"
)
SUSIE_UPSTREAM_README_SOURCE = (
    "https://github.com/kvablack/susie/blob/" + SUSIE_SOURCE_COMMIT + "/README.md#L1-L28"
)
SUSIE_HIGH_LEVEL_SOURCE = (
    "https://github.com/kvablack/susie/blob/" + SUSIE_SOURCE_COMMIT + "/susie/model.py#L150-L227"
)
SUSIE_UPSTREAM_AGENT_SOURCE = (
    "https://github.com/rail-berkeley/bridge_data_v2/blob/"
    + SUSIE_BRIDGE_DATA_SOURCE_COMMIT
    + "/jaxrl_m/agents/continuous/gc_ddpm_bc.py#L41-L204"
)
SUSIE_UPSTREAM_CONFIG_SOURCE = (
    "https://github.com/rail-berkeley/bridge_data_v2/blob/"
    + SUSIE_BRIDGE_DATA_SOURCE_COMMIT
    + "/experiments/susie/calvin/configs/gcbc_train_config.py#L38-L76"
)
SUSIE_REPLICATION_AGENT_SOURCE = (
    "https://github.com/rail-berkeley/bridge_data_v2/blob/"
    + SUSIE_BRIDGE_DATA_SOURCE_COMMIT
    + "/jaxrl_m/agents/continuous/gc_bc.py#L47-L141"
)
SUSIE_HIGH_LEVEL_ARTIFACT_SOURCE = "https://huggingface.co/%s/tree/%s" % (
    SUSIE_HIGH_LEVEL_MODEL_ID,
    SUSIE_HIGH_LEVEL_MODEL_REVISION,
)
SUSIE_LOW_LEVEL_GCBC_ARTIFACT_SOURCE = "https://huggingface.co/%s/tree/%s" % (
    SUSIE_LOW_LEVEL_GCBC_MODEL_ID,
    SUSIE_LOW_LEVEL_GCBC_MODEL_REVISION,
)
SUSIE_CORRECTED_CHECKPOINT_BLOCKER = (
    "The advertised patreya/gcbc-bridge artifact is a gc_bc checkpoint. It cannot be loaded as the distinct "
    "upstream gc_ddpm_bc four-action sensitivity arm. A separately pinned, license-recorded gc_ddpm_bc checkpoint "
    "and its complete architecture/configuration manifest are required."
)

# Exact released AutoEval low-level normalizer.  Gripper conversion is part of
# the wrapper: the first six dimensions use normal statistics, and normalized
# gripper values become {0, 1} at zero.
_AUTOEVAL_ACTION_MEAN = (
    1.9296819e-04,
    1.3667766e-04,
    -1.4583133e-04,
    -1.8390431e-04,
    -3.0808983e-04,
    2.7425270e-04,
    5.9716219e-01,
)
_AUTOEVAL_ACTION_STD = (
    0.00912848,
    0.0127196,
    0.01229497,
    0.02606696,
    0.02875283,
    0.07807977,
    0.48710242,
)


class SuSIEUnavailableError(PolicyLoadError):
    """A requested source-pinned SuSIE arm is unavailable or unsafe to run."""


def _immutable_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(char in "0123456789abcdef" for char in value.lower())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version_without_build(value: object) -> str:
    return str(value).split("+", 1)[0]


def _is_hex_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _checked_git_checkout(root_value: str, revision: str, anchors: Tuple[str, ...], *, label: str) -> Path:
    """Prove an on-disk source checkout rather than trusting a profile string."""

    root = Path(root_value)
    if not root.is_dir():
        raise SuSIEUnavailableError("Reviewed %s source checkout is absent at %s." % (label, root))
    try:
        head = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SuSIEUnavailableError("Could not inspect reviewed %s source checkout." % label) from error
    if head.returncode != 0 or head.stdout.strip() != revision:
        raise SuSIEUnavailableError("%s source checkout HEAD does not match reviewed commit %s." % (label, revision))
    if status.returncode != 0 or status.stdout.strip():
        raise SuSIEUnavailableError("%s source checkout is dirty; refusing mutable code." % label)
    missing = tuple(anchor for anchor in anchors if not (root / anchor).is_file())
    if missing:
        raise SuSIEUnavailableError("%s source checkout lacks reviewed files: %s." % (label, ", ".join(missing)))
    return root.resolve()


@dataclass(frozen=True)
class SuSIEPolicyProfile:
    """A local asset manifest for one named SuSIE configuration arm.

    ``stable_diffusion_revision`` intentionally has no convenience default:
    the AutoEval configuration names an unpinned dependency.  An operator must
    bind a reviewed local SD v1.5 Flax snapshot before high-level generation.
    """

    profile_id: str
    arm_id: str
    local_low_level_checkpoint_dir: str
    low_level_checkpoint_sha256: str
    local_high_level_model_path: Optional[str] = None
    local_stable_diffusion_path: Optional[str] = None
    stable_diffusion_revision: Optional[str] = None
    stable_diffusion_asset_sha256: Optional[str] = None
    stable_diffusion_license_evidence_uri: Optional[str] = None
    local_stable_diffusion_manifest_path: Optional[str] = None
    local_susie_source_path: Optional[str] = None
    local_bridge_data_source_path: Optional[str] = None
    local_autoeval_source_path: Optional[str] = None
    low_level_model_id: str = SUSIE_LOW_LEVEL_GCBC_MODEL_ID
    low_level_model_revision: str = SUSIE_LOW_LEVEL_GCBC_MODEL_REVISION
    high_level_model_revision: str = SUSIE_HIGH_LEVEL_MODEL_REVISION
    high_level_unet_sha256: str = SUSIE_HIGH_LEVEL_UNET_SHA256
    high_level_unet_config_sha256: str = SUSIE_HIGH_LEVEL_UNET_CONFIG_SHA256
    source_revision: str = SUSIE_SOURCE_COMMIT
    bridge_data_source_revision: str = SUSIE_BRIDGE_DATA_SOURCE_COMMIT
    autoeval_source_revision: str = AUTOEVAL_POLICY_SOURCE_COMMIT
    jax_version: str = SUSIE_JAX_VERSION
    flax_version: str = SUSIE_FLAX_VERSION
    diffusers_version: str = SUSIE_DIFFUSERS_VERSION
    transformers_version: str = SUSIE_TRANSFORMERS_VERSION
    local_files_only: bool = True
    container_digest: Optional[str] = None
    asset_manifest_id: Optional[str] = None
    asset_manifest_sha256: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None

    def algorithm(self) -> str:
        if self.arm_id in (SUSIE_REPLICATION_ARM_ID, SUSIE_LL_REPLICATION_ARM_ID):
            return SUSIE_AUTOEVAL_ALGORITHM
        if self.arm_id in (SUSIE_CORRECTED_SENSITIVITY_ARM_ID, SUSIE_LL_CORRECTED_SENSITIVITY_ARM_ID):
            return SUSIE_UPSTREAM_ALGORITHM
        return ""

    def is_low_level_only(self) -> bool:
        return self.arm_id in (SUSIE_LL_REPLICATION_ARM_ID, SUSIE_LL_CORRECTED_SENSITIVITY_ARM_ID)

    def review_error(self, *, require_high_level: bool) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        if self.arm_id not in (
            SUSIE_REPLICATION_ARM_ID,
            SUSIE_CORRECTED_SENSITIVITY_ARM_ID,
            SUSIE_LL_REPLICATION_ARM_ID,
            SUSIE_LL_CORRECTED_SENSITIVITY_ARM_ID,
        ):
            return "arm_id must be one of the separately named released or corrected SuSIE arms"
        if require_high_level != (not self.is_low_level_only()):
            return "adapter and profile disagree about whether this SuSIE arm includes the high-level subgoal model"
        if not self.local_low_level_checkpoint_dir:
            return "local_low_level_checkpoint_dir is required; Hub fetching is disabled"
        if not _is_hex_digest(self.low_level_checkpoint_sha256):
            return "low_level_checkpoint_sha256 must be an actual SHA-256 digest"
        if self.source_revision != SUSIE_SOURCE_COMMIT or self.bridge_data_source_revision != SUSIE_BRIDGE_DATA_SOURCE_COMMIT:
            return "SuSIE and BridgeData source revisions must match the reviewed immutable commits"
        if self.autoeval_source_revision != AUTOEVAL_POLICY_SOURCE_COMMIT:
            return "autoeval_source_revision must match the reviewed AutoEval commit"
        if self.jax_version != SUSIE_JAX_VERSION or self.flax_version != SUSIE_FLAX_VERSION:
            return "SuSIE requires JAX %s and Flax %s" % (SUSIE_JAX_VERSION, SUSIE_FLAX_VERSION)
        if require_high_level and (
            self.diffusers_version != SUSIE_DIFFUSERS_VERSION or self.transformers_version != SUSIE_TRANSFORMERS_VERSION
        ):
            return "SuSIE high-level sampling requires Diffusers %s and Transformers %s" % (
                SUSIE_DIFFUSERS_VERSION,
                SUSIE_TRANSFORMERS_VERSION,
            )
        if not self.local_files_only:
            return "SuSIE loader is local-only; network retrieval is not permitted"
        if self.algorithm() == SUSIE_AUTOEVAL_ALGORITHM:
            if self.low_level_model_id != SUSIE_LOW_LEVEL_GCBC_MODEL_ID or self.low_level_model_revision != SUSIE_LOW_LEVEL_GCBC_MODEL_REVISION:
                return "released AutoEval gc_bc arm must use the pinned patreya/gcbc-bridge artifact identity"
            if self.low_level_checkpoint_sha256 != SUSIE_LOW_LEVEL_GCBC_CHECKPOINT_SHA256:
                return "released AutoEval gc_bc arm must use the pinned low-level checkpoint SHA-256"
        else:
            # The reviewed inputs identify the *configuration* of the
            # corrected arm, not a compatible checkpoint nor its action
            # normalizer.  Do not make a permissive profile field look like
            # evidence that either has been supplied.
            return SUSIE_CORRECTED_CHECKPOINT_BLOCKER
        if require_high_level:
            if not self.local_high_level_model_path or not self.local_stable_diffusion_path:
                return "SuSIE needs separate local high-level and Stable Diffusion Flax snapshots"
            if self.high_level_model_revision != SUSIE_HIGH_LEVEL_MODEL_REVISION:
                return "high_level_model_revision must match the pinned kvablack/susie revision"
            if self.high_level_unet_sha256 != SUSIE_HIGH_LEVEL_UNET_SHA256:
                return "high_level_unet_sha256 must match the pinned kvablack/susie UNet"
            if self.high_level_unet_config_sha256 != SUSIE_HIGH_LEVEL_UNET_CONFIG_SHA256:
                return "high_level_unet_config_sha256 must match the pinned kvablack/susie UNet config"
            if not _immutable_revision(self.stable_diffusion_revision):
                return "Stable Diffusion Flax dependency must have an immutable recorded revision"
            if (
                not _is_hex_digest(self.stable_diffusion_asset_sha256)
                or not self.stable_diffusion_license_evidence_uri
                or not self.local_stable_diffusion_manifest_path
            ):
                return "Stable Diffusion Flax dependency needs a local asset manifest, verified manifest SHA-256, and license evidence before use"
        return None


@dataclass(frozen=True)
class SuSIEActionReport:
    """One source-wrapper action with goal/cadence identity retained."""

    action: Tuple[float, float, float, float, float, float, float]
    proposal: Tuple[Tuple[float, float, float, float, float, float, float], ...]
    arm_id: str
    algorithm: str
    source_image_timestamp: Optional[float]
    backend_calls: int
    wall_seconds: Optional[float]
    policy_step: int
    goal_refreshed: bool
    subgoal_cadence: Optional[int]
    high_level_rng_rule: Optional[str]


@dataclass(frozen=True)
class _SuSIERuntime:
    """Lazy source imports. Test factories do not change production behavior."""

    jax: Any
    numpy: Any
    agent_registry: Mapping[str, Any]
    encoder_registry: Mapping[str, Any]
    restore_checkpoint: Callable[..., Any]
    create_sample_fn: Callable[..., Any]
    jax_version: str
    flax_version: str
    diffusers_version: str
    transformers_version: str


SUSIE_REPLICATION_CONTRACT = PolicyContract(
    name="SuSIE (AutoEval gc_bc replication)",
    required_observation_history=1,
    requires_proprio=False,
    native_proposal_horizon=1,
    certified_execute_prefix=1,
    temporal_ensembling=False,
    preprocessing="Current and goal RGB 256×256 images; source high-level SuSIE image editor refreshes a subgoal every 20 policy steps.",
    normalization="Released AutoEval gc_bc first six dimensions use ACT_STD*action+ACT_MEAN; gripper is thresholded to {0,1}.",
    reset_rule="Clear source-generated subgoal and reset the 20-step cadence at every trajectory boundary.",
    rng_rule="AutoEval gc_bc uses deterministic argmax; upstream high-level create_sample_fn seeds its internal JAX RNG from wall clock.",
    implementation_status=CapabilityStatus.READY_UNQUALIFIED,
    limitation="Source high-level sampling is wall-clock seeded and the complete component chain still needs fixture/Gate-B evidence.",
)

SUSIE_LL_REPLICATION_CONTRACT = PolicyContract(
    name="SuSIE_LL (AutoEval gc_bc replication)",
    required_observation_history=1,
    requires_proprio=False,
    native_proposal_horizon=1,
    certified_execute_prefix=1,
    temporal_ensembling=False,
    preprocessing="Current and scenario-supplied goal RGB 256×256 images; no high-level text-to-subgoal stage.",
    normalization="Released AutoEval gc_bc ACT_MEAN/ACT_STD plus binary gripper conversion.",
    reset_rule="No action cache; retain no goal state across trajectories.",
    rng_rule="AutoEval gc_bc deterministic argmax call; Gate-A/B fixtures remain required.",
    implementation_status=CapabilityStatus.READY_UNQUALIFIED,
    limitation="Goal-image provenance, exact low-level execution, and fixture evidence remain required before qualification.",
)

SUSIE_CORRECTED_SENSITIVITY_CONTRACT = PolicyContract(
    name="SuSIE (upstream gc_ddpm_bc sensitivity)",
    required_observation_history=1,
    requires_proprio=False,
    native_proposal_horizon=SUSIE_UPSTREAM_ACTION_HORIZON,
    certified_execute_prefix=None,
    temporal_ensembling=True,
    preprocessing="Upstream gc_ddpm_bc goal-conditioned RGB observation history; separate from AutoEval gc_bc.",
    normalization="Corrected arm requires the source-compatible gc_ddpm_bc checkpoint's declared action normalization.",
    reset_rule="Clear image/action histories, diffusion RNG, and high-level subgoal state at a trajectory boundary.",
    rng_rule="gc_ddpm_bc samples a JAX diffusion trajectory; all seeds and calls must be retained.",
    implementation_status=CapabilityStatus.BLOCKED,
    limitation=SUSIE_CORRECTED_CHECKPOINT_BLOCKER,
)


class _SuSIEBaseAdapter:
    """Shared local source implementation for subgoal and low-level-only arms."""

    contract: PolicyContract

    def __init__(
        self,
        profile: SuSIEPolicyProfile,
        *,
        require_high_level: bool,
        runtime_factory: Optional[Callable[[], _SuSIERuntime]] = None,
    ) -> None:
        self.profile = profile
        self._require_high_level = require_high_level
        self._runtime_factory = runtime_factory
        self._runtime: Optional[_SuSIERuntime] = None
        self._agent: Any = None
        self._goal_generator: Any = None
        self._current_goal: Any = None
        self._policy_step = 0
        self._action_history: Deque[Any] = deque(maxlen=SUSIE_UPSTREAM_ACTION_HORIZON)
        self._ddpm_rng: Any = None
        self._backend_calls = 0
        self.last_report: Optional[SuSIEActionReport] = None

    def _sources(self) -> Tuple[str, ...]:
        return (
            SUSIE_AUTOEVAL_POLICY_SOURCE,
            SUSIE_AUTOEVAL_CONFIG_SOURCE,
            SUSIE_UPSTREAM_README_SOURCE,
            SUSIE_HIGH_LEVEL_SOURCE,
            SUSIE_REPLICATION_AGENT_SOURCE,
            SUSIE_UPSTREAM_AGENT_SOURCE,
            SUSIE_UPSTREAM_CONFIG_SOURCE,
            SUSIE_HIGH_LEVEL_ARTIFACT_SOURCE,
            SUSIE_LOW_LEVEL_GCBC_ARTIFACT_SOURCE,
        )

    def _profile_error(self) -> Optional[str]:
        return self.profile.review_error(require_high_level=self._require_high_level)

    def _source_paths_missing(self) -> Tuple[str, ...]:
        missing = []
        if not self.profile.local_susie_source_path:
            missing.append("local_susie_source_path (clean kvablack/susie checkout)")
        if not self.profile.local_bridge_data_source_path:
            missing.append("local_bridge_data_source_path (clean rail-berkeley/bridge_data_v2 checkout)")
        if not self.profile.local_autoeval_source_path:
            missing.append("local_autoeval_source_path (clean zhouzypaul/auto_eval checkout)")
        return tuple(missing)

    def _low_checkpoint_file(self) -> Path:
        return Path(self.profile.local_low_level_checkpoint_dir) / "checkpoint"

    def _required_paths(self) -> Tuple[Tuple[Path, Optional[str]], ...]:
        paths: list[tuple[Path, Optional[str]]] = [(self._low_checkpoint_file(), self.profile.low_level_checkpoint_sha256)]
        if self._require_high_level:
            high_root = Path(self.profile.local_high_level_model_path or "")
            stable_root = Path(self.profile.local_stable_diffusion_path or "")
            paths.extend(
                (
                    (high_root / "unet" / "config.json", self.profile.high_level_unet_config_sha256),
                    (high_root / "unet" / "diffusion_flax_model.msgpack", self.profile.high_level_unet_sha256),
                    (stable_root / "vae", None),
                    (stable_root / "text_encoder", None),
                    (stable_root / "tokenizer", None),
                    (Path(self.profile.local_stable_diffusion_manifest_path or ""), self.profile.stable_diffusion_asset_sha256),
                )
            )
        return tuple(paths)

    def capability(self) -> CapabilityResult:
        error = self._profile_error()
        if error is not None:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="SuSIE loading is blocked: %s." % error,
                source_verified=True,
                evidence_uris=self._sources(),
                details={
                    "profile_id": self.profile.profile_id,
                    "arm_id": self.profile.arm_id,
                    "algorithm": self.profile.algorithm(),
                    "corrected_checkpoint_blocker": SUSIE_CORRECTED_CHECKPOINT_BLOCKER,
                    "local_files_only": self.profile.local_files_only,
                },
            )
        source_paths_missing = self._source_paths_missing()
        if source_paths_missing and self._runtime_factory is None:
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="SuSIE reviewed local source checkouts are required before import; no source was fetched.",
                source_verified=True,
                evidence_uris=self._sources(),
                details={"missing_source_paths": source_paths_missing, "local_files_only": True},
            )
        missing = tuple(str(path) for path, _ in self._required_paths() if not path.exists())
        if missing:
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="SuSIE local immutable artifacts are incomplete; no Hub download was attempted.",
                source_verified=True,
                evidence_uris=self._sources(),
                details={"arm_id": self.profile.arm_id, "missing_files": missing, "local_files_only": True},
            )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Pinned local %s source path is configured. It remains unqualified until its component-chain fixture, "
                "goal provenance, feedback, action normalization, and Gate-B tests pass."
            ) % self.contract.name,
            source_verified=True,
            evidence_uris=self._sources(),
            details={
                "arm_id": self.profile.arm_id,
                "algorithm": self.profile.algorithm(),
                "low_level_model_id": self.profile.low_level_model_id,
                "low_level_model_revision": self.profile.low_level_model_revision,
                "high_level_model_id": SUSIE_HIGH_LEVEL_MODEL_ID if self._require_high_level else None,
                "high_level_model_revision": self.profile.high_level_model_revision if self._require_high_level else None,
                "subgoal_cadence": SUSIE_SUBGOAL_CADENCE if self._require_high_level else None,
                "native_proposal_horizon": self.contract.native_proposal_horizon,
                "certified_execute_prefix": self.contract.certified_execute_prefix,
                "local_files_only": True,
                "asset_manifest_id": self.profile.asset_manifest_id,
                "asset_manifest_sha256": self.profile.asset_manifest_sha256,
                "runtime_lock_id": self.profile.runtime_lock_id,
                "runtime_lock_sha256": self.profile.runtime_lock_sha256,
            },
        )

    def _check_profile(self) -> None:
        error = self._profile_error()
        if error is not None:
            raise SuSIEUnavailableError("Refusing to load SuSIE: %s." % error)

    def _load_runtime(self) -> _SuSIERuntime:
        if self._runtime is not None:
            return self._runtime
        susie_source_root: Optional[Path] = None
        bridge_source_root: Optional[Path] = None
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
        else:
            source_paths_missing = self._source_paths_missing()
            if source_paths_missing:
                raise SuSIEUnavailableError(
                    "SuSIE requires reviewed local source checkouts before import: %s."
                    % ", ".join(source_paths_missing)
                )
            autoeval_source_root = _checked_git_checkout(
                str(self.profile.local_autoeval_source_path),
                AUTOEVAL_POLICY_SOURCE_COMMIT,
                ("auto_eval/robot/policy.py", "scripts/configs/eval_config.py"),
                label="AutoEval",
            )
            # Retain the value to make the code-source association explicit;
            # AutoEval config is re-expressed below rather than imported into
            # this policy process and therefore cannot perform network I/O.
            _ = autoeval_source_root
            susie_source_root = _checked_git_checkout(
                str(self.profile.local_susie_source_path),
                SUSIE_SOURCE_COMMIT,
                ("susie/model.py", "requirements.txt"),
                label="SuSIE",
            )
            bridge_source_root = _checked_git_checkout(
                str(self.profile.local_bridge_data_source_path),
                SUSIE_BRIDGE_DATA_SOURCE_COMMIT,
                ("jaxrl_m/agents/__init__.py", "jaxrl_m/agents/continuous/gc_bc.py", "jaxrl_m/agents/continuous/gc_ddpm_bc.py"),
                label="BridgeData V2",
            )
            try:
                import flax  # type: ignore
                import jax  # type: ignore
                import numpy as np  # type: ignore
                from flax.training import checkpoints  # type: ignore
                from jaxrl_m.agents import agents  # type: ignore
                from jaxrl_m.vision import encoders  # type: ignore
                if self._require_high_level:
                    import diffusers  # type: ignore
                    import transformers  # type: ignore
                    from susie.model import create_sample_fn  # type: ignore
                else:
                    diffusers = None
                    transformers = None
                    create_sample_fn = None
            except ImportError as error:
                raise SuSIEUnavailableError(
                    "SuSIE dependencies are unavailable. Install the pinned JAX/Flax/Diffusers source stack in an "
                    "isolated policy image; imports are lazy."
                ) from error
            runtime = _SuSIERuntime(
                jax=jax,
                numpy=np,
                agent_registry=agents,
                encoder_registry=encoders,
                restore_checkpoint=checkpoints.restore_checkpoint,
                create_sample_fn=create_sample_fn,
                jax_version=str(getattr(jax, "__version__", "")),
                flax_version=str(getattr(flax, "__version__", "")),
                diffusers_version=str(getattr(diffusers, "__version__", "")),
                transformers_version=str(getattr(transformers, "__version__", "")),
            )
            agent_class = agents.get(self.profile.algorithm()) if hasattr(agents, "get") else agents[self.profile.algorithm()]
            if not _is_within(Path(inspect.getfile(agent_class)), bridge_source_root):
                raise SuSIEUnavailableError("Imported JAXRL agent does not resolve inside local_bridge_data_source_path.")
            if self._require_high_level and not _is_within(Path(inspect.getfile(create_sample_fn)), susie_source_root):
                raise SuSIEUnavailableError("Imported SuSIE sampler does not resolve inside local_susie_source_path.")
        versions = [
            ("JAX", _version_without_build(runtime.jax_version), SUSIE_JAX_VERSION),
            ("Flax", runtime.flax_version, SUSIE_FLAX_VERSION),
        ]
        if self._require_high_level:
            versions.extend(
                (
                    ("Diffusers", runtime.diffusers_version, SUSIE_DIFFUSERS_VERSION),
                    ("Transformers", runtime.transformers_version, SUSIE_TRANSFORMERS_VERSION),
                )
            )
        for name, actual, expected in versions:
            if actual != expected:
                raise SuSIEUnavailableError("SuSIE runtime %s %r is incompatible; exactly %s is required." % (name, actual, expected))
        if self.profile.algorithm() not in runtime.agent_registry:
            raise SuSIEUnavailableError("Pinned SuSIE runtime does not expose low-level agent %r." % self.profile.algorithm())
        if not callable(runtime.restore_checkpoint) or (self._require_high_level and not callable(runtime.create_sample_fn)):
            raise SuSIEUnavailableError("Pinned SuSIE runtime omitted a required reviewed source API.")
        self._runtime = runtime
        return runtime

    def _validate_artifacts(self) -> None:
        for path, expected_digest in self._required_paths():
            if not path.exists():
                raise SuSIEUnavailableError("Expected local immutable SuSIE artifact at %s." % path)
            if expected_digest is not None and _sha256_file(path) != expected_digest:
                raise SuSIEUnavailableError("SHA-256 mismatch for %s; refusing to load it." % path)
        if self._require_high_level:
            self._validate_stable_diffusion_manifest()

    def _validate_stable_diffusion_manifest(self) -> None:
        """Bind every source-loaded SD component file, not merely its manifest.

        ``susie.model.create_sample_fn`` opens the local ``vae``,
        ``text_encoder``, and ``tokenizer`` subtrees.  A hash of a declaration
        alone would not prove those bytes.  The deliberately small manifest
        schema below records every regular file in precisely those subtrees;
        no path can escape the configured local snapshot.
        """

        root = Path(self.profile.local_stable_diffusion_path or "").resolve()
        manifest_path = Path(self.profile.local_stable_diffusion_manifest_path or "")
        if not manifest_path.is_file():
            raise SuSIEUnavailableError("Stable Diffusion asset manifest is absent at %s." % manifest_path)
        expected_manifest_digest = self.profile.stable_diffusion_asset_sha256
        if not _is_hex_digest(expected_manifest_digest) or _sha256_file(manifest_path) != expected_manifest_digest:
            raise SuSIEUnavailableError("Stable Diffusion asset manifest SHA-256 does not match the profile.")
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, ValueError) as error:
            raise SuSIEUnavailableError("Stable Diffusion asset manifest is not valid JSON.") from error
        if not isinstance(manifest, Mapping):
            raise SuSIEUnavailableError("Stable Diffusion asset manifest must be a JSON object.")
        if manifest.get("model_id") != SUSIE_STABLE_DIFFUSION_MODEL_ID:
            raise SuSIEUnavailableError("Stable Diffusion asset manifest model_id does not match the reviewed source dependency.")
        if manifest.get("revision") != self.profile.stable_diffusion_revision:
            raise SuSIEUnavailableError("Stable Diffusion asset manifest revision does not match the profile.")
        raw_records = manifest.get("file_records")
        if not isinstance(raw_records, list) or not raw_records:
            raise SuSIEUnavailableError("Stable Diffusion asset manifest needs nonempty file_records.")
        allowed_prefixes = ("vae/", "text_encoder/", "tokenizer/")
        recorded: dict[str, str] = {}
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise SuSIEUnavailableError("Stable Diffusion asset manifest records must be objects.")
            relative = raw_record.get("path")
            digest = raw_record.get("sha256")
            if not isinstance(relative, str) or not relative or relative.startswith("/") or "\\" in relative:
                raise SuSIEUnavailableError("Stable Diffusion asset manifest record path is invalid.")
            relative_path = Path(relative)
            if relative_path.as_posix() != relative or ".." in relative_path.parts or not relative.startswith(allowed_prefixes):
                raise SuSIEUnavailableError("Stable Diffusion asset manifest may only record files under vae/, text_encoder/, or tokenizer/.")
            if not _is_hex_digest(digest) or relative in recorded:
                raise SuSIEUnavailableError("Stable Diffusion asset manifest has duplicate or non-SHA-256 file records.")
            target = (root / relative_path).resolve()
            if not _is_within(target, root) or not target.is_file() or target.is_symlink():
                raise SuSIEUnavailableError("Stable Diffusion asset manifest record does not resolve to a local regular file.")
            if _sha256_file(target) != digest:
                raise SuSIEUnavailableError("Stable Diffusion asset file SHA-256 mismatch for %s." % relative)
            recorded[relative] = digest
        actual_files: set[str] = set()
        for prefix in allowed_prefixes:
            directory = root / prefix.rstrip("/")
            if not directory.is_dir() or directory.is_symlink():
                raise SuSIEUnavailableError("Stable Diffusion required component directory is absent or symlinked: %s." % directory)
            for candidate in directory.rglob("*"):
                if candidate.is_symlink():
                    raise SuSIEUnavailableError("Stable Diffusion component tree contains a symlink: %s." % candidate)
                if candidate.is_file():
                    actual_files.add(candidate.relative_to(root).as_posix())
        if actual_files != set(recorded):
            missing = sorted(actual_files.difference(recorded))
            extra = sorted(set(recorded).difference(actual_files))
            raise SuSIEUnavailableError(
                "Stable Diffusion asset manifest must cover exactly every source-loaded component file; missing=%s extra=%s."
                % (missing, extra)
            )

    def _agent_specification(self) -> Tuple[str, Mapping[str, Any], Mapping[str, Any], Tuple[int, ...], Tuple[int, ...]]:
        if self.profile.algorithm() == SUSIE_AUTOEVAL_ALGORITHM:
            return (
                "resnetv1-34",
                {"pooling_method": "avg", "add_spatial_coordinates": False, "act": "swish"},
                {
                    "shared_goal_encoder": True,
                    "early_goal_concat": True,
                    "use_proprio": False,
                    "learning_rate": 3e-4,
                    "warmup_steps": 2000,
                    "decay_steps": int(2e6),
                    "network_kwargs": {"hidden_dims": (256, 256, 256), "dropout_rate": 0.1},
                    "policy_kwargs": {
                        "tanh_squash_distribution": False,
                        "std_parameterization": "fixed",
                        "fixed_std": [1, 1, 1, 1, 1, 1, 0.1],
                    },
                },
                (1, SUSIE_IMAGE_SIZE, SUSIE_IMAGE_SIZE, 3),
                (1, 7),
            )
        return (
            "resnetv1-34-bridge",
            {"pooling_method": "avg", "add_spatial_coordinates": True, "act": "swish"},
            {
                "score_network_kwargs": {
                    "time_dim": 32,
                    "num_blocks": 3,
                    "dropout_rate": 0.1,
                    "hidden_dim": 256,
                    "use_layer_norm": True,
                },
                "early_goal_concat": True,
                "shared_goal_encoder": True,
                "use_proprio": False,
                "beta_schedule": "cosine",
                "diffusion_steps": 20,
                "action_samples": 1,
                "repeat_last_step": 0,
                "learning_rate": 3e-4,
                "warmup_steps": 2000,
                "actor_decay_steps": int(2e6),
            },
            (1, 1, SUSIE_IMAGE_SIZE, SUSIE_IMAGE_SIZE, 3),
            (1, SUSIE_UPSTREAM_ACTION_HORIZON, 7),
        )

    def _ensure_agent(self, runtime: _SuSIERuntime) -> Any:
        self._check_profile()
        if self._agent is not None:
            return self._agent
        self._validate_artifacts()
        encoder_name, encoder_kwargs, agent_kwargs, image_shape, action_shape = self._agent_specification()
        try:
            encoder_def = runtime.encoder_registry[encoder_name](**encoder_kwargs)
            observations = {"image": runtime.numpy.zeros(image_shape)}
            goals = {"image": runtime.numpy.zeros((1, SUSIE_IMAGE_SIZE, SUSIE_IMAGE_SIZE, 3))}
            actions = runtime.numpy.zeros(action_shape)
            initial_rng = runtime.jax.random.PRNGKey(42)
            _, construct_rng = runtime.jax.random.split(initial_rng)
            agent = runtime.agent_registry[self.profile.algorithm()].create(
                rng=construct_rng,
                observations=observations,
                goals=goals,
                actions=actions,
                encoder_def=encoder_def,
                **agent_kwargs,
            )
            # Flax's target-aware checkpoint restore is the source API used by
            # both AutoEval and BridgeData.  It receives a typed initialized
            # target rather than arbitrary Python objects.
            agent = runtime.restore_checkpoint(self.profile.local_low_level_checkpoint_dir, target=agent)
        except Exception as error:
            raise SuSIEUnavailableError("Source-compatible SuSIE low-level construction/restore failed.") from error
        if not callable(getattr(agent, "sample_actions", None)):
            raise SuSIEUnavailableError("SuSIE checkpoint did not restore an agent with sample_actions().")
        self._agent = agent
        self._ddpm_rng = runtime.jax.random.PRNGKey(42)
        return agent

    def _ensure_goal_generator(self, runtime: _SuSIERuntime) -> Any:
        if not self._require_high_level:
            return None
        if self._goal_generator is None:
            self._validate_artifacts()
            high_root = Path(self.profile.local_high_level_model_path or "")
            # ``create_sample_fn`` only touches W&B for its Orbax checkpoint
            # branch. The released Hub snapshot is an ``unet/`` directory;
            # reject any ambiguous local root rather than letting it select
            # that networked branch.
            if (high_root / "checkpoint").exists():
                raise SuSIEUnavailableError(
                    "High-level SuSIE root contains 'checkpoint'; source create_sample_fn would take its W&B-backed "
                    "Orbax branch, so local-only loading is blocked."
                )
            try:
                self._goal_generator = runtime.create_sample_fn(
                    str(high_root),
                    None,
                    50,
                    7.5,
                    4.0,
                    0.0,
                    self.profile.local_stable_diffusion_path,
                )
            except Exception as error:
                raise SuSIEUnavailableError("Source-compatible SuSIE high-level sampler construction failed.") from error
        if not callable(self._goal_generator):
            raise SuSIEUnavailableError("SuSIE create_sample_fn() did not return a callable subgoal sampler.")
        return self._goal_generator

    @staticmethod
    def _image(value: Any, *, role: str, runtime: _SuSIERuntime) -> Any:
        if value is None:
            raise PolicyContractError("SuSIE requires a %s image." % role)
        array = runtime.numpy.asarray(value)
        if tuple(getattr(array, "shape", ())) != (SUSIE_IMAGE_SIZE, SUSIE_IMAGE_SIZE, 3):
            raise PolicyContractError("SuSIE %s image must have exact shape (256, 256, 3)." % role)
        try:
            dtype_name = str(array.dtype)
        except AttributeError as error:
            raise PolicyContractError("SuSIE %s image must be a numeric array." % role) from error
        if dtype_name != "uint8":
            raise PolicyContractError("SuSIE %s image must be uint8; do not silently rescale/cast observations." % role)
        return array

    def _observation(self, observation: PolicyObservation, runtime: _SuSIERuntime) -> Any:
        if not isinstance(observation.prompt, str) or not observation.prompt.strip():
            raise PolicyContractError("SuSIE requires a nonempty exact task instruction.")
        if len(observation.image_history) != 1:
            raise PolicyContractError("SuSIE source wrapper takes exactly one fresh current image per tick.")
        return self._image(observation.image_history[0], role="current", runtime=runtime)

    @staticmethod
    def _rows(raw: Any, runtime: _SuSIERuntime, expected_count: int) -> Tuple[Tuple[float, float, float, float, float, float, float], ...]:
        array = runtime.numpy.asarray(runtime.jax.device_get(raw))
        if expected_count == 1 and tuple(getattr(array, "shape", ())) == (1, 7):
            array = array[0]
        if expected_count == 1 and tuple(getattr(array, "shape", ())) == (7,):
            rows_source = [array.tolist()]
        elif tuple(getattr(array, "shape", ())) == (expected_count, 7):
            rows_source = array.tolist()
        else:
            raise PolicyContractError(
                "SuSIE %s returned shape %r; expected %s."
                % ("gc_bc" if expected_count == 1 else "gc_ddpm_bc", getattr(array, "shape", None), (expected_count, 7))
            )
        rows = []
        for row in rows_source:
            try:
                normalized = tuple(float(value) for value in row)
            except (TypeError, ValueError) as error:
                raise PolicyContractError("SuSIE low-level output contains a nonnumeric action.") from error
            if len(normalized) != 7 or not all(math.isfinite(value) for value in normalized):
                raise PolicyContractError("SuSIE low-level output must be finite 7-D actions.")
            # AutoEval's released wrapper applies this exact normalizer to all
            # non-PiZero policies.  A future corrected checkpoint must carry a
            # separate manifest before it can alter this boundary.
            action = tuple(
                _AUTOEVAL_ACTION_STD[index] * normalized[index] + _AUTOEVAL_ACTION_MEAN[index]
                for index in range(6)
            ) + (1.0 if normalized[6] > 0 else 0.0,)
            rows.append(action)
        return tuple(rows)  # type: ignore[return-value]

    def _sample_low_level(self, agent: Any, current: Any, goal: Any, runtime: _SuSIERuntime) -> Tuple[Tuple[float, ...], ...]:
        if self.profile.algorithm() == SUSIE_AUTOEVAL_ALGORITHM:
            raw = agent.sample_actions(
                {"image": current[None, ...]},
                {"image": goal[None, ...]},
                temperature=0.0,
                argmax=True,
                seed=None,
            )
            return self._rows(raw, runtime, 1)
        # This path is intentionally a separately named sensitivity arm.  The
        # upstream GCDDPMBCAgent needs an observation-history axis and samples
        # its diffusion trajectory using an explicit JAX key.
        history = runtime.numpy.stack((current,))
        self._ddpm_rng, sample_rng = runtime.jax.random.split(self._ddpm_rng)
        raw = agent.sample_actions(
            {"image": history},
            {"image": goal},
            seed=sample_rng,
            temperature=1.0,
            argmax=False,
        )
        return self._rows(raw, runtime, SUSIE_UPSTREAM_ACTION_HORIZON)

    def _select_action(self, proposal: Tuple[Tuple[float, ...], ...], runtime: _SuSIERuntime) -> Tuple[float, float, float, float, float, float, float]:
        if len(proposal) == 1:
            return proposal[0]  # type: ignore[return-value]
        # Follow the upstream robot evaluator's rolling proposal aggregation.
        # It is reachable only after a separately pinned gc_ddpm_bc artifact
        # exists; it never changes the released gc_bc arm.
        proposal_array = runtime.numpy.asarray(proposal)
        if not self._action_history:
            self._action_history.append(proposal_array)
            weights = 1.0 / (runtime.numpy.arange(len(proposal_array)) + 1)
            accumulated = proposal_array * weights[:, None]
        else:
            previous = self._action_history[-1]
            shifted = runtime.numpy.concatenate((previous[1:], runtime.numpy.zeros_like(previous[-1:])))
            self._action_history.append(proposal_array)
            weights = runtime.numpy.concatenate((1.0 / (runtime.numpy.arange(len(proposal_array) - 1) + 2), runtime.numpy.asarray([1.0 / len(proposal_array)])))
            accumulated = shifted * weights[:, None] + proposal_array * weights[:, None]
        selected = tuple(float(value) for value in runtime.numpy.asarray(accumulated[0]).tolist())
        if len(selected) != 7 or not all(math.isfinite(value) for value in selected):
            raise PolicyContractError("SuSIE gc_ddpm_bc temporal aggregation did not yield a finite 7-D action.")
        return selected  # type: ignore[return-value]

    def reset(self) -> None:
        """Reset cadence, cached goal, action aggregation and local diffusion key."""

        self._current_goal = None
        self._policy_step = 0
        self._action_history.clear()
        if self._runtime is not None:
            self._ddpm_rng = self._runtime.jax.random.PRNGKey(42)
        self.last_report = None

    def predict_with_report(self, observation: PolicyObservation) -> SuSIEActionReport:
        # Reject licensing/source/profile failures before importing JAX,
        # Diffusers, or any model-loader dependency.
        self._check_profile()
        runtime = self._load_runtime()
        current = self._observation(observation, runtime)
        agent = self._ensure_agent(runtime)
        goal_refreshed = False
        if self._require_high_level:
            if self._policy_step % SUSIE_SUBGOAL_CADENCE == 0:
                goal_generator = self._ensure_goal_generator(runtime)
                generated_goal = goal_generator(current, observation.prompt)
                self._current_goal = self._image(generated_goal, role="generated subgoal", runtime=runtime)
                goal_refreshed = True
            goal = self._current_goal
        else:
            goal = self._image(observation.goal_image, role="scenario-supplied goal", runtime=runtime)
        started = time.perf_counter()
        proposal = self._sample_low_level(agent, current, goal, runtime)
        wall_seconds = time.perf_counter() - started
        action = self._select_action(proposal, runtime)
        step = self._policy_step
        self._policy_step += 1
        self._backend_calls += 1
        report = SuSIEActionReport(
            action=action,
            proposal=proposal,  # type: ignore[arg-type]
            arm_id=self.profile.arm_id,
            algorithm=self.profile.algorithm(),
            source_image_timestamp=observation.timestamp,
            backend_calls=1,
            wall_seconds=wall_seconds,
            policy_step=step,
            goal_refreshed=goal_refreshed,
            subgoal_cadence=SUSIE_SUBGOAL_CADENCE if self._require_high_level else None,
            high_level_rng_rule=("source create_sample_fn initializes a JAX key from wall-clock time" if self._require_high_level else None),
        )
        self.last_report = report
        return report

    def predict_action(self, observation: PolicyObservation) -> Tuple[float, float, float, float, float, float, float]:
        return self.predict_with_report(observation).action

    predict = predict_action


class SuSIEPolicyAdapter(_SuSIEBaseAdapter):
    """SuSIE high-level image subgoal plus source-matched low-level arm."""

    def __init__(self, profile: SuSIEPolicyProfile, *, runtime_factory: Optional[Callable[[], _SuSIERuntime]] = None) -> None:
        contract = SUSIE_REPLICATION_CONTRACT if profile.algorithm() == SUSIE_AUTOEVAL_ALGORITHM else SUSIE_CORRECTED_SENSITIVITY_CONTRACT
        self.contract = contract
        super().__init__(profile, require_high_level=True, runtime_factory=runtime_factory)


class SuSIELowLevelPolicyAdapter(_SuSIEBaseAdapter):
    """SuSIE_LL: source low-level goal policy, never a fake language shortcut."""

    def __init__(self, profile: SuSIEPolicyProfile, *, runtime_factory: Optional[Callable[[], _SuSIERuntime]] = None) -> None:
        contract = SUSIE_LL_REPLICATION_CONTRACT if profile.algorithm() == SUSIE_AUTOEVAL_ALGORITHM else SUSIE_CORRECTED_SENSITIVITY_CONTRACT
        self.contract = contract
        super().__init__(profile, require_high_level=False, runtime_factory=runtime_factory)
