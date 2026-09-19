"""Pinned, local-only Octo v1.0 Bridge inference with released temporal ensembling.

Octo's native contract is a two-image history and a four-action proposal.  Its
released ``TemporalEnsembleWrapper`` averages the overlapping predictions for
the current control tick across the last ``pred_horizon`` model calls, which
means the wrapper consumes a *fresh* prediction at every tick and therefore
executes exactly one ensembled action per call.  This adapter implements that
shape and refuses to execute the un-ensembled tail rows of a proposal.

The production sampler profile is bound to Octo source commit
``241fb3514b7c40957a86d869fecb7c7fc353f540``.  Its
``sample_actions`` implementation reads ``timestep_pad_mask`` from the
observation when no explicit keyword is supplied.  It is intentionally
different from the separately named 37951 native-v0.1 diagnostic profile.

Nothing here claims Gate-B qualification, and nothing produces an action until
a golden action fixture certifies the executed prefix.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from plumb.adapters.contracts import PolicyContract, PolicyObservation

from .contracts import (
    ActionNormalizationType,
    FrameIdentity,
    GripperPolarityConvention,
    NativePolicyAdapter,
    NativeProposal,
    PolicyActionNormalizer,
    PolicyActionNormalizerStatistics,
    PolicyCertification,
    PolicyContractError,
    PolicyExecutionMode,
    PolicyLoadError,
    UNRESOLVED_BRIDGE_GRIPPER_CONVENTION,
    immutable_revision,
    rows_from_native,
)
from .native import AUTOEVAL_POLICY_SOURCE, OCTO_BASE_V1_CONTRACT, OCTO_SMALL_V1_CONTRACT, OCTO_SOURCE
from .provenance import file_source_sha256


OCTO_SMALL_MODEL_ID = "rail-berkeley/octo-small"
OCTO_BASE_MODEL_ID = "rail-berkeley/octo-base"
OCTO_FORBIDDEN_MODEL_IDS = ("rail-berkeley/octo-small-1.5", "rail-berkeley/octo-base-1.5")
OCTO_SMALL_CHECKPOINT_STEP = 270000
OCTO_BASE_CHECKPOINT_STEP = 300000
OCTO_JAX_VERSION = "0.4.20"
OCTO_DEFAULT_DATASET_KEY = "bridge_dataset"
OCTO_PRODUCTION_SOURCE_COMMIT = "241fb3514b7c40957a86d869fecb7c7fc353f540"
OCTO_PRODUCTION_SOURCE = "https://github.com/octo-models/octo/tree/" + OCTO_PRODUCTION_SOURCE_COMMIT
OCTO_PRODUCTION_MODEL_SOURCE = "https://github.com/octo-models/octo/blob/" + OCTO_PRODUCTION_SOURCE_COMMIT + "/octo/model/octo_model.py"
OCTO_ENSEMBLE_SOURCE = "https://github.com/octo-models/octo/blob/" + OCTO_PRODUCTION_SOURCE_COMMIT + "/octo/utils/gym_wrappers.py"
OCTO_AUTOEVAL_SOURCE_COMMIT = "3ea3ff44c6950433cfbcb4294a3deaa616533745"
OCTO_AUTOEVAL_OCTO_POLICY_SOURCE = (
    "https://github.com/zhouzypaul/auto_eval/blob/"
    + OCTO_AUTOEVAL_SOURCE_COMMIT
    + "/auto_eval/robot/policy.py#L134-L164"
)
OCTO_AUTOEVAL_STATIC_RNG_SEED = 0

# octo's released ``TemporalEnsembleWrapper`` defaults to ``exp_weight=0``,
# which is a uniform mean over the overlapping predictions.  AutoEval's value
# is a wrapper detail: it must be read out of pinned source and confirmed by a
# golden action fixture, which is why no profile gets this as a silent default.
OCTO_RELEASED_ENSEMBLE_EXP_WEIGHT = 0.0


class OctoUnavailableError(PolicyLoadError):
    """The approved local Octo v1.0 JAX environment or checkpoint is absent."""


@dataclass(frozen=True)
class OctoObservationKeys:
    """The exact observation dictionary keys the pinned Octo release expects.

    Octo renamed its timestep padding key between releases, so the key set is a
    revisioned, declared value rather than a guess buried in the call site.
    """

    keys_revision: str
    image_key: str
    pad_mask_key: str
    source_uri: Optional[str] = None
    evidence: str = "declared_from_pinned_release_pending_fixture"

    def __post_init__(self) -> None:
        if not self.keys_revision:
            raise PolicyContractError("OctoObservationKeys.keys_revision is required.")
        if not self.image_key or not self.pad_mask_key:
            raise PolicyContractError("Octo observation keys must both be nonempty.")


OCTO_V1_0_OBSERVATION_KEYS = OctoObservationKeys(
    keys_revision="octo-v1.0-image_primary-timestep_pad_mask-241fb",
    image_key="image_primary",
    pad_mask_key="timestep_pad_mask",
    source_uri=OCTO_PRODUCTION_MODEL_SOURCE,
)


@dataclass(frozen=True)
class OctoEnsembleConfig:
    """Released temporal-ensembling configuration, explicitly revisioned."""

    config_revision: str
    exponential_weight: float
    horizon: int
    source_uri: Optional[str] = OCTO_ENSEMBLE_SOURCE

    def __post_init__(self) -> None:
        if not self.config_revision:
            raise PolicyContractError("OctoEnsembleConfig.config_revision is required.")
        if isinstance(self.exponential_weight, bool) or not isinstance(self.exponential_weight, (int, float)):
            raise PolicyContractError("Octo exponential_weight must be a real number.")
        if float(self.exponential_weight) < 0.0:
            raise PolicyContractError("Octo exponential_weight must be nonnegative.")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise PolicyContractError("Octo ensemble horizon must be a positive integer.")


class OctoTemporalEnsembler:
    """Transcription of octo's released overlapping-prediction average.

    The released wrapper pairs the oldest retained prediction with the *latest*
    index of its chunk and the newest prediction with index 0, then weights by
    ``exp(-exp_weight * i)`` with ``i=0`` on the oldest prediction — so more
    recent predictions get exponentially *less* weight.  That ordering is
    preserved verbatim here; at ``exp_weight=0`` it is a uniform mean.
    """

    def __init__(self, config: OctoEnsembleConfig) -> None:
        self.config = config
        self._history: List[Tuple[Tuple[float, ...], ...]] = []

    @property
    def depth(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history = []

    def snapshot(self) -> Dict[str, Any]:
        return {
            "config_revision": self.config.config_revision,
            "exponential_weight": float(self.config.exponential_weight),
            "horizon": int(self.config.horizon),
            "history": [[list(row) for row in proposal] for proposal in self._history],
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise PolicyContractError("Octo ensembler state must be a mapping.")
        if str(payload.get("config_revision")) != self.config.config_revision:
            raise PolicyContractError(
                "Octo ensembler snapshot is revision %r, live configuration is %r."
                % (payload.get("config_revision"), self.config.config_revision)
            )
        if float(payload.get("exponential_weight", -1.0)) != float(self.config.exponential_weight):
            raise PolicyContractError("Octo ensembler snapshot exponential_weight does not match the live value.")
        raw = payload.get("history", [])
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise PolicyContractError("Octo ensembler history must be a sequence of proposals.")
        history: List[Tuple[Tuple[float, ...], ...]] = []
        for proposal in raw:
            if isinstance(proposal, (str, bytes)) or not isinstance(proposal, Sequence):
                raise PolicyContractError("Each Octo ensembler proposal must be a sequence of rows.")
            rows: List[Tuple[float, ...]] = []
            for row in proposal:
                if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 7:
                    raise PolicyContractError("Octo ensembler rows must be 7-D actions.")
                rows.append(tuple(float(value) for value in row))
            history.append(tuple(rows))
        if len(history) > self.config.horizon:
            raise PolicyContractError("Octo ensembler history is deeper than its declared horizon.")
        self._history = history

    def ensemble(self, proposal: Tuple[Tuple[float, ...], ...]) -> Tuple[float, ...]:
        horizon = self.config.horizon
        if len(proposal) < horizon:
            raise PolicyContractError(
                "Octo temporal ensembling needs at least %d proposed rows, got %d; it will not pad the chunk."
                % (horizon, len(proposal))
            )
        self._history.append(tuple(proposal[:horizon]))
        if len(self._history) > horizon:
            self._history.pop(0)
        depth = len(self._history)
        weights: List[float] = []
        rows: List[Tuple[float, ...]] = []
        for index in range(depth):
            # ``index == 0`` is the oldest retained prediction; it contributes
            # the row it forecast for the current tick, which is ``depth-1-index``.
            rows.append(self._history[index][depth - 1 - index])
            weights.append(math.exp(-float(self.config.exponential_weight) * index))
        total = sum(weights)
        if total <= 0.0:
            raise PolicyContractError("Octo ensemble weights summed to zero; refusing to divide.")
        return tuple(
            sum(weight * row[channel] for weight, row in zip(weights, rows)) / total for channel in range(7)
        )


@dataclass(frozen=True)
class OctoPolicyProfile:
    """The approval boundary for one local Octo v1.0 snapshot."""

    profile_id: str
    local_model_path: str
    model_id: str
    checkpoint_revision: str
    checkpoint_step: int
    source_revision: str = OCTO_PRODUCTION_SOURCE_COMMIT
    observation_keys: OctoObservationKeys = OCTO_V1_0_OBSERVATION_KEYS
    dataset_statistics_key: str = OCTO_DEFAULT_DATASET_KEY
    jax_version: str = OCTO_JAX_VERSION
    local_files_only: bool = True
    native_unnormalizes: bool = False
    asset_manifest_id: Optional[str] = None
    asset_manifest_sha256: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None

    @property
    def checkpoint_relative_path(self) -> str:
        """Octo checkpoints are not at the repository root.

        ``resolve/main/checkpoint`` returns 404; the real path is
        ``<step>/default/checkpoint``.
        """

        return "%d/default/checkpoint" % int(self.checkpoint_step)

    def review_error(self) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        if not self.local_model_path:
            return "local_model_path is required; Hub fetching is disabled"
        if self.model_id in OCTO_FORBIDDEN_MODEL_IDS or "1.5" in self.model_id:
            return (
                "Octo must be v1.0; %r is a different model and octo-small-1.5 is explicitly excluded"
                % self.model_id
            )
        if self.model_id not in (OCTO_SMALL_MODEL_ID, OCTO_BASE_MODEL_ID):
            return "model_id must be %r or %r" % (OCTO_SMALL_MODEL_ID, OCTO_BASE_MODEL_ID)
        expected_step = OCTO_SMALL_CHECKPOINT_STEP if self.model_id == OCTO_SMALL_MODEL_ID else OCTO_BASE_CHECKPOINT_STEP
        if int(self.checkpoint_step) != expected_step:
            return "%s pins checkpoint step %d, not %r" % (self.model_id, expected_step, self.checkpoint_step)
        if not immutable_revision(self.checkpoint_revision):
            return "checkpoint_revision must be an immutable 40-character hexadecimal revision"
        if self.source_revision != OCTO_PRODUCTION_SOURCE_COMMIT:
            return "source_revision must equal the pinned production Octo source commit"
        if not self.local_files_only:
            return "Octo loader is local-only; network retrieval is not permitted"
        if self.native_unnormalizes:
            return (
                "PLUMB applies its own revisioned action normalizer at exactly one documented boundary; "
                "letting Octo also unnormalize would apply the transform twice"
            )
        if self.jax_version != OCTO_JAX_VERSION:
            return "Octo v1.0 requires its pinned JAX %s runtime, not %r" % (OCTO_JAX_VERSION, self.jax_version)
        if not self.dataset_statistics_key:
            return "dataset_statistics_key is required to bind the action statistics"
        return None


@dataclass(frozen=True)
class _OctoRuntime:
    """Injection seam for the lazily imported JAX/Octo stack."""

    jax_version: str
    array_factory: Callable[[Any, str], Any]
    prng_key: Callable[[int], Any]
    fold_in: Callable[[Any, int], Any]
    model_loader: Callable[[str, int], Any]


def load_octo_action_statistics(
    path: Any,
    *,
    normalizer_revision: str,
    dataset_statistics_key: str = OCTO_DEFAULT_DATASET_KEY,
    mask: Optional[Sequence[bool]] = None,
) -> PolicyActionNormalizerStatistics:
    """Build Octo's mean/std action statistics from its shipped JSON.

    The numbers are never hardcoded in PLUMB: they come from the checkpoint's
    own ``dataset_statistics.json``, and the file's hash is recorded so the lock
    can bind the normalizer to the weights.
    """

    candidate = Path(str(path))
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except OSError as error:
        raise OctoUnavailableError("Cannot read Octo dataset statistics %s: %s" % (candidate, error)) from error
    except ValueError as error:
        raise OctoUnavailableError("Octo dataset statistics %s is not valid JSON." % candidate) from error
    scoped = payload.get(dataset_statistics_key, payload) if isinstance(payload, Mapping) else None
    if not isinstance(scoped, Mapping) or not isinstance(scoped.get("action"), Mapping):
        raise OctoUnavailableError(
            "Octo dataset statistics %s has no %r action block." % (candidate, dataset_statistics_key)
        )
    action = scoped["action"]
    declared_mask = mask if mask is not None else action.get("mask")
    if declared_mask is None:
        raise OctoUnavailableError(
            "Octo action statistics declare no normalization mask. Supply it explicitly from pinned source "
            "rather than assuming which channels are normalized."
        )
    try:
        mean = tuple(float(value) for value in action["mean"])
        std = tuple(float(value) for value in action["std"])
    except (KeyError, TypeError, ValueError) as error:
        raise OctoUnavailableError("Octo action statistics lack usable mean/std vectors.") from error
    return PolicyActionNormalizerStatistics(
        normalizer_revision=normalizer_revision,
        normalization_type=ActionNormalizationType.NORMAL,
        mask=tuple(bool(flag) for flag in declared_mask),
        mean=mean,
        std=std,
        source_uri=str(candidate),
        statistics_sha256=file_source_sha256(candidate),
    )


class _OctoAdapterBase(NativePolicyAdapter):
    """Shared Octo v1.0 loader; subclasses bind one declared contract."""

    source_urls = (OCTO_PRODUCTION_SOURCE, AUTOEVAL_POLICY_SOURCE, OCTO_ENSEMBLE_SOURCE)

    def __init__(
        self,
        profile: OctoPolicyProfile,
        *,
        normalizer: PolicyActionNormalizer,
        ensemble: OctoEnsembleConfig,
        certification: Optional[PolicyCertification] = None,
        execution_mode: PolicyExecutionMode = PolicyExecutionMode.CERTIFIED,
        gripper_convention: GripperPolarityConvention = UNRESOLVED_BRIDGE_GRIPPER_CONVENTION,
        runtime_factory: Optional[Callable[[], _OctoRuntime]] = None,
        model_factory: Optional[Callable[[OctoPolicyProfile, _OctoRuntime], Any]] = None,
    ) -> None:
        super().__init__(
            normalizer=normalizer,
            certification=certification,
            execution_mode=execution_mode,
            gripper_convention=gripper_convention,
        )
        if normalizer.statistics.normalization_type is not ActionNormalizationType.NORMAL:
            raise PolicyContractError(
                "Octo's released statistics are a mean/standard-deviation family; %r is not that family."
                % normalizer.statistics.normalization_type.value
            )
        if ensemble.horizon != self.base_contract.native_proposal_horizon:
            raise PolicyContractError(
                "Octo temporal ensembling depth must equal its %d-action native proposal, not %d."
                % (self.base_contract.native_proposal_horizon, ensemble.horizon)
            )
        self.profile = profile
        self.ensembler = OctoTemporalEnsembler(ensemble)
        self._runtime_factory = runtime_factory
        self._model_factory = model_factory
        self._runtime: Optional[_OctoRuntime] = None
        self._model: Any = None
        self._task: Any = None
        self._task_instruction: Optional[str] = None

    # -- capability -------------------------------------------------------------

    def _profile_errors(self) -> Tuple[str, ...]:
        error = self.profile.review_error()
        return () if error is None else (error,)

    def _availability_error(self) -> Optional[str]:
        if self._model_factory is not None:
            return None
        root = Path(self.profile.local_model_path)
        if not root.is_dir():
            return "local checkpoint directory %s is absent; no Hub download was attempted" % root
        checkpoint = root / self.profile.checkpoint_relative_path
        if not checkpoint.exists():
            return (
                "expected the Octo checkpoint at %s (it is not at the repository root; resolve/main/checkpoint "
                "returns 404)" % checkpoint
            )
        return None

    def _capability_details(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile.profile_id,
            "model_id": self.profile.model_id,
            "octo_release": "v1.0",
            "checkpoint_revision": self.profile.checkpoint_revision,
            "source_revision": self.profile.source_revision,
            "autoeval_wrapper_revision": OCTO_AUTOEVAL_SOURCE_COMMIT,
            "autoeval_sampler_rng_seed": OCTO_AUTOEVAL_STATIC_RNG_SEED,
            "checkpoint_step": int(self.profile.checkpoint_step),
            "checkpoint_relative_path": self.profile.checkpoint_relative_path,
            "forbidden_model_ids": list(OCTO_FORBIDDEN_MODEL_IDS),
            "observation_keys_revision": self.profile.observation_keys.keys_revision,
            "observation_keys": {
                "image": self.profile.observation_keys.image_key,
                "pad_mask": self.profile.observation_keys.pad_mask_key,
            },
            "ensemble_config_revision": self.ensembler.config.config_revision,
            "ensemble_exponential_weight": float(self.ensembler.config.exponential_weight),
            "jax_version": self.profile.jax_version,
            "dataset_statistics_key": self.profile.dataset_statistics_key,
            "normalizer_statistics_sha256": self.normalizer.statistics.statistics_sha256,
            "local_files_only": True,
            "asset_manifest_id": self.profile.asset_manifest_id,
            "asset_manifest_sha256": self.profile.asset_manifest_sha256,
            "runtime_lock_id": self.profile.runtime_lock_id,
            "runtime_lock_sha256": self.profile.runtime_lock_sha256,
        }

    # -- durable state ----------------------------------------------------------

    def _reset_native(self, seed: int) -> None:
        self.ensembler.reset()
        self._task = None
        self._task_instruction = None

    def _snapshot_extra(self) -> Dict[str, Any]:
        return {
            "ensembler": self.ensembler.snapshot(),
            # The encoded task pytree is runtime-specific and cannot be safely
            # serialized as JSON. Bind its exact source instruction instead;
            # the pinned model recreates it lazily after restore.
            "task_instruction": self._task_instruction,
        }

    def _restore_extra(self, payload: Mapping[str, Any]) -> None:
        ensembler = payload.get("ensembler")
        if ensembler is None:
            raise PolicyContractError("Octo snapshot must carry its temporal-ensembling state.")
        self.ensembler.restore(ensembler)
        instruction = payload.get("task_instruction")
        if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
            raise PolicyContractError("Octo snapshot task_instruction must be a nonempty string or null.")
        if self._step > 0 and instruction is None:
            raise PolicyContractError(
                "Octo snapshot after a native call must bind task_instruction so a restored AutoEval task cannot change silently."
            )
        self._task = None
        self._task_instruction = instruction

    # -- runtime ----------------------------------------------------------------

    def _check_profile(self) -> None:
        error = self.profile.review_error()
        if error is not None:
            raise OctoUnavailableError("Refusing to load Octo: " + error + ".")

    def _load_runtime(self) -> _OctoRuntime:
        if self._runtime is not None:
            return self._runtime
        self._check_profile()
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
            if runtime.jax_version != self.profile.jax_version:
                raise OctoUnavailableError(
                    "Injected Octo runtime is JAX %r; exactly %s is required."
                    % (runtime.jax_version, self.profile.jax_version)
                )
            self._runtime = runtime
            return runtime
        try:
            import jax  # type: ignore
            import numpy  # type: ignore
            from octo.model.octo_model import OctoModel  # type: ignore
        except ImportError as error:
            raise OctoUnavailableError(
                "Octo v1.0 dependencies are unavailable. Install the pinned python 3.10 / JAX %s image in an "
                "isolated policy container; imports here are lazy." % OCTO_JAX_VERSION
            ) from error
        installed = str(getattr(jax, "__version__", ""))
        if installed != self.profile.jax_version:
            raise OctoUnavailableError(
                "Installed JAX %r is incompatible; Octo v1.0 pins exactly %s."
                % (installed, self.profile.jax_version)
            )
        self._runtime = _OctoRuntime(
            jax_version=installed,
            array_factory=lambda value, dtype: numpy.asarray(value, dtype=dtype),
            prng_key=lambda seed: jax.random.PRNGKey(seed),
            fold_in=lambda key, step: jax.random.fold_in(key, step),
            model_loader=lambda path, step: OctoModel.load_pretrained(path, step),
        )
        return self._runtime

    def _ensure_model(self, runtime: _OctoRuntime) -> Any:
        self._check_profile()
        if self._model is not None:
            return self._model
        if self._model_factory is not None:
            self._model = self._model_factory(self.profile, runtime)
        else:
            root = Path(self.profile.local_model_path)
            checkpoint = root / self.profile.checkpoint_relative_path
            if not checkpoint.exists():
                raise OctoUnavailableError(
                    "Expected the Octo v1.0 checkpoint at %s; refusing network retrieval." % checkpoint
                )
            self._model = runtime.model_loader(str(root), int(self.profile.checkpoint_step))
        for method_name in ("create_tasks", "sample_actions"):
            if not callable(getattr(self._model, method_name, None)):
                raise OctoUnavailableError("Loaded Octo model does not expose native %s()." % method_name)
        return self._model

    # -- native call ------------------------------------------------------------

    def timestep_pad_mask(self) -> Tuple[bool, ...]:
        """Octo's per-timestep padding mask for the declared two-image history.

        Before the first generated frame arrives, the controller's reset
        bootstrap legitimately repeats the start frame, so the older slot is
        marked as padding exactly as the released history wrapper does.  It is
        never used to disguise a stale frame later in the episode.
        """

        required = self.base_contract.required_observation_history
        if self._step == 0:
            return tuple(index == required - 1 for index in range(required))
        return tuple(True for _ in range(required))

    def _episode_rng(self, runtime: _OctoRuntime) -> Any:
        if self._seed is None:
            raise PolicyContractError(
                "Octo requires reset(seed) before a native call so its wrapper history is reset and restorable."
            )
        # The pinned AutoEval OctoPolicy invokes PRNGKey(0) on every policy
        # call.  Preserve that source behavior; the controller reset seed
        # governs PLUMB-owned history state, not the upstream sampler key.
        return runtime.prng_key(OCTO_AUTOEVAL_STATIC_RNG_SEED)

    def _task_for_instruction(self, model: Any, instruction: str) -> Any:
        """Mirror AutoEval's one-language-task cache for a rollout/reset span."""

        if self._task_instruction is None:
            self._task_instruction = instruction
        elif instruction != self._task_instruction:
            raise PolicyContractError(
                "Octo AutoEval source binds the language task on its first call; call reset(seed) before changing instructions."
            )
        if self._task is None:
            self._task = model.create_tasks(texts=[instruction])
        return self._task

    def _native_proposal(self, observation: PolicyObservation, identity: FrameIdentity) -> NativeProposal:
        runtime = self._load_runtime()
        model = self._ensure_model(runtime)
        keys = self.profile.observation_keys
        pad_mask = self.timestep_pad_mask()
        native_observation = {
            keys.image_key: runtime.array_factory([list(observation.image_history)], "uint8"),
            keys.pad_mask_key: runtime.array_factory([list(pad_mask)], "bool"),
        }
        task = self._task_for_instruction(model, observation.prompt)
        rng = self._episode_rng(runtime)
        started = time.perf_counter()
        # ``unnormalization_statistics=None`` asks the released sampler for
        # model-space actions so PLUMB's revisioned normalizer is the single
        # denormalization boundary.  That keyword, and the observation key names
        # above, are pinned-release details: both must be confirmed against the
        # reviewed commit and the golden action fixture before Gate B.
        raw = model.sample_actions(native_observation, task, unnormalization_statistics=None, rng=rng)
        wall_seconds = time.perf_counter() - started
        normalized = rows_from_native(
            raw,
            label="Octo sample_actions",
            expected_rows=self.base_contract.native_proposal_horizon,
        )
        # Exactly one denormalization per proposed row, at this one boundary.
        actions = tuple(self.normalizer.denormalize(row) for row in normalized)
        ensembled = self.ensembler.ensemble(actions)
        return NativeProposal(
            policy_name=self.base_contract.name,
            actions=actions,
            executed_actions=(ensembled,),
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
                "source_revision": self.profile.source_revision,
                "checkpoint_step": int(self.profile.checkpoint_step),
                "timestep_pad_mask": list(pad_mask),
                "ensemble_depth": self.ensembler.depth,
                "ensemble_exponential_weight": float(self.ensembler.config.exponential_weight),
                "ensemble_config_revision": self.ensembler.config.config_revision,
                "observation_keys_revision": keys.keys_revision,
                "denormalization_boundary": "plumb_policy_action_normalizer",
                "sampler_rng": "autoeval_static_prngkey_0",
                "controller_reset_seed": self._seed,
            },
        )


class OctoSmallV1PolicyAdapter(_OctoAdapterBase):
    """Octo-Small v1.0, the primary-matrix Octo policy."""

    base_contract: PolicyContract = OCTO_SMALL_V1_CONTRACT


class OctoBaseV1PolicyAdapter(_OctoAdapterBase):
    """Octo-Base v1.0, a separate diagnostic and not a seventh benchmark policy."""

    base_contract: PolicyContract = OCTO_BASE_V1_CONTRACT
