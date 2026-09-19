"""Dependency-light policy contracts and safe lazy-loader primitives.

Nothing in :mod:`plumb.policies` imports a model framework at module import
time.  That is intentional: a CPU control-plane host should be able to read
the declared native contracts without accidentally importing Torch, JAX, or
Transformers, and should never cause a checkpoint download.

This module also holds the pieces that every real multi-action adapter shares:
an explicitly revisioned action normalizer, distinct Bridge gripper *state* and
*action* unit types, and :class:`PolicyCertification` — the golden-action
fixture that is the only thing allowed to turn ``certified_execute_prefix``
from ``None`` into a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from plumb.adapters.contracts import CapabilityResult, CapabilityStatus, PolicyContract, PolicyObservation

from .provenance import canonical_json_sha256, image_pixel_hash, sha256_bytes


OPENVLA_SOURCE_COMMIT = "c8f03f48af692657d3060c19588038c7220e9af9"
AUTOEVAL_POLICY_SOURCE_COMMIT = "3ea3ff44c6950433cfbcb4294a3deaa616533745"
OPEN_PI_ZERO_SOURCE_COMMIT = "c3df7fb062175c16f69d7ca4ce042958ea238fb7"


class PolicyLoadError(RuntimeError):
    """A requested local policy cannot safely be loaded in this runtime."""


class PolicyContractError(ValueError):
    """A caller supplied an observation or action outside a native contract."""


class PolicyCertificationError(PolicyLoadError):
    """A certified action path was requested without a valid golden fixture."""


class NativePolicyCallable(Protocol):
    """Minimal protocol for an explicitly supplied, separately certified wrapper."""

    def __call__(self, observation: PolicyObservation) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class ExternalPolicyProfile:
    """Identity for a policy that has no bundled implementation yet.

    ``loader_hook`` is deliberately an injection point rather than an import
    of a guessed third-party API.  An operator may attach a source-reviewed
    wrapper in its own isolated image.  Doing so changes neither this adapter's
    unqualified status nor its need for fixture certification.
    """

    profile_id: str
    local_model_path: Optional[str] = None
    checkpoint_revision: Optional[str] = None
    loader_revision: Optional[str] = None
    local_files_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ExternalPolicyHook:
    """Honest placeholder for incompatible native policy environments.

    It never synthesizes an action.  A callable can be injected only by the
    deployment which owns a reviewed native wrapper; tests can use the same
    hook without importing an ML stack.
    """

    def __init__(
        self,
        contract: PolicyContract,
        profile: ExternalPolicyProfile,
        *,
        loader_hook: Optional[Callable[[ExternalPolicyProfile], NativePolicyCallable]] = None,
        source_urls: Tuple[str, ...] = (),
    ) -> None:
        self.contract = contract
        self.profile = profile
        self._loader_hook = loader_hook
        self._policy: Optional[NativePolicyCallable] = None
        self._source_urls = source_urls

    def capability(self) -> CapabilityResult:
        if self._loader_hook is None:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason=(
                    "%s has a declared native contract but no source-reviewed local loader in PLUMB. "
                    "No inferred or placeholder actions will be emitted."
                )
                % self.contract.name,
                source_verified=bool(self._source_urls),
                evidence_uris=self._source_urls,
                details={
                    "profile_id": self.profile.profile_id,
                    "native_proposal_horizon": self.contract.native_proposal_horizon,
                    "certified_execute_prefix": self.contract.certified_execute_prefix,
                    "local_files_only": self.profile.local_files_only,
                },
            )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "%s has an explicitly injected local native wrapper. Fixture validation and Gate B are "
                "still required; this adapter cannot mark it qualified."
            )
            % self.contract.name,
            source_verified=bool(self._source_urls),
            evidence_uris=self._source_urls,
            details={"profile_id": self.profile.profile_id, "loader_revision": self.profile.loader_revision},
        )

    def _ensure_policy(self) -> NativePolicyCallable:
        if self._policy is not None:
            return self._policy
        if self._loader_hook is None:
            raise PolicyLoadError(
                "%s is intentionally unimplemented until a pinned native wrapper and fixture evidence are supplied."
                % self.contract.name
            )
        policy = self._loader_hook(self.profile)
        if not callable(policy):
            raise PolicyLoadError("The injected %s loader did not return a callable native policy." % self.contract.name)
        self._policy = policy
        return policy

    def predict(self, observation: PolicyObservation) -> Tuple[float, ...]:
        """Call an injected wrapper without filling or extending its proposal."""

        action = self._ensure_policy()(observation)
        if isinstance(action, (str, bytes)):
            raise PolicyContractError("Native policy action must be a numeric sequence, not text.")
        values = tuple(float(value) for value in action)
        if len(values) != 7:
            raise PolicyContractError(
                "%s injected wrapper returned %d values; PLUMB canonical physical actions are 7-D."
                % (self.contract.name, len(values))
            )
        return values


# ---------------------------------------------------------------------------
# Bridge gripper units.  State units and action units are not interchangeable.
# ---------------------------------------------------------------------------

BRIDGE_STATE_GRIPPER_CLOSED = 0.0
BRIDGE_STATE_GRIPPER_OPEN_LIMIT = 0.39
BRIDGE_ACTION_GRIPPER_LOW = 0.0
BRIDGE_ACTION_GRIPPER_HIGH = 0.996
BRIDGE_GRIPPER_RANGE_EVIDENCE = "handoff_observed_range_pending_source_data_fixture"
_GRIPPER_TOLERANCE = 1e-6


@dataclass(frozen=True)
class GripperStateValue:
    """The Bridge proprioception gripper channel.

    The handoff's convention is a 0.0-0.39 continuous range in which **0.0
    means fully closed**.  That range still has to pass a source-data fixture
    (spec section 2), so ``range_evidence`` records that it is observed rather
    than certified.  This type exists so a state value can never be handed to
    something expecting an action gripper value.
    """

    value: float
    range_evidence: str = BRIDGE_GRIPPER_RANGE_EVIDENCE

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise PolicyContractError("Bridge gripper state must be a real number.")
        if not math.isfinite(float(self.value)):
            raise PolicyContractError("Bridge gripper state must be finite.")
        if not (
            BRIDGE_STATE_GRIPPER_CLOSED - _GRIPPER_TOLERANCE
            <= float(self.value)
            <= BRIDGE_STATE_GRIPPER_OPEN_LIMIT + _GRIPPER_TOLERANCE
        ):
            raise PolicyContractError(
                "Bridge gripper *state* %r is outside the observed 0.0-%.2f range; an action gripper value "
                "(0.0-%.3f) is not a state value." % (self.value, BRIDGE_STATE_GRIPPER_OPEN_LIMIT, BRIDGE_ACTION_GRIPPER_HIGH)
            )

    @property
    def is_fully_closed(self) -> bool:
        return abs(float(self.value) - BRIDGE_STATE_GRIPPER_CLOSED) <= _GRIPPER_TOLERANCE

    def open_fraction(self) -> float:
        """Fraction open in state units: 0.0 is fully closed, 1.0 is the open limit."""

        return float(self.value) / BRIDGE_STATE_GRIPPER_OPEN_LIMIT


@dataclass(frozen=True)
class GripperActionValue:
    """The seventh channel of a physical 7-D Bridge action.

    Observed values are quantised over 0.0-0.996.  Whether the high end means
    open or closed is wrapper-specific and is *not* asserted here; converting
    to or from a state value requires an explicitly resolved
    :class:`GripperPolarityConvention`.
    """

    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise PolicyContractError("Bridge gripper action must be a real number.")
        if not math.isfinite(float(self.value)):
            raise PolicyContractError("Bridge gripper action must be finite.")
        if not (
            BRIDGE_ACTION_GRIPPER_LOW - _GRIPPER_TOLERANCE
            <= float(self.value)
            <= BRIDGE_ACTION_GRIPPER_HIGH + _GRIPPER_TOLERANCE
        ):
            raise PolicyContractError(
                "Bridge gripper *action* %r is outside the observed 0.0-%.3f range."
                % (self.value, BRIDGE_ACTION_GRIPPER_HIGH)
            )


@dataclass(frozen=True)
class GripperPolarityConvention:
    """One wrapper's gripper transformation, or an explicit unresolved marker.

    ``action_high_means_open=None`` means the polarity has not been read out of
    pinned source plus a golden fixture.  Every conversion then refuses instead
    of guessing, because a silently inverted gripper destroys a policy without
    producing any error.
    """

    convention_id: str
    source_uri: Optional[str] = None
    source_revision: Optional[str] = None
    action_high_means_open: Optional[bool] = None

    @property
    def resolved(self) -> bool:
        return self.action_high_means_open is not None

    def _require_resolved(self) -> bool:
        if self.action_high_means_open is None:
            raise PolicyContractError(
                "Gripper convention %r has not resolved whether a high action value means open. "
                "Verify the wrapper's gripper transformation against pinned source and a golden "
                "action fixture before converting between state and action units."
                % self.convention_id
            )
        return bool(self.action_high_means_open)

    def action_open_fraction(self, action: GripperActionValue) -> float:
        if not isinstance(action, GripperActionValue):
            raise PolicyContractError(
                "action_open_fraction requires a GripperActionValue; state units and action gripper units "
                "are not interchangeable."
            )
        high_is_open = self._require_resolved()
        fraction = float(action.value) / BRIDGE_ACTION_GRIPPER_HIGH
        return fraction if high_is_open else 1.0 - fraction

    def action_from_state(self, state: GripperStateValue) -> GripperActionValue:
        if not isinstance(state, GripperStateValue):
            raise PolicyContractError(
                "action_from_state requires a GripperStateValue; state units and action gripper units "
                "are not interchangeable."
            )
        high_is_open = self._require_resolved()
        fraction = state.open_fraction()
        scaled = fraction if high_is_open else 1.0 - fraction
        return GripperActionValue(min(max(scaled, 0.0), 1.0) * BRIDGE_ACTION_GRIPPER_HIGH)

    def state_from_action(self, action: GripperActionValue) -> GripperStateValue:
        fraction = self.action_open_fraction(action)
        return GripperStateValue(min(max(fraction, 0.0), 1.0) * BRIDGE_STATE_GRIPPER_OPEN_LIMIT)


UNRESOLVED_BRIDGE_GRIPPER_CONVENTION = GripperPolarityConvention(
    convention_id="bridge_gripper_polarity_unresolved",
    source_uri=(
        "https://github.com/zhouzypaul/auto_eval/blob/" + AUTOEVAL_POLICY_SOURCE_COMMIT + "/auto_eval/robot/policy.py"
    ),
    source_revision=AUTOEVAL_POLICY_SOURCE_COMMIT,
    action_high_means_open=None,
)


# ---------------------------------------------------------------------------
# Revisioned, invertible policy-side action normalization.
# ---------------------------------------------------------------------------


class ActionNormalizationType(str, Enum):
    """Per-adapter denormalization family.

    ``bounds`` is OpenPiZero's declared setting and ``normal`` is the
    mean/standard-deviation family the other wrappers use.  Neither is a
    universal native-policy API: each adapter's statistics, denormalization,
    gripper transformation, and clipping must be checked against pinned source
    and a golden fixture.
    """

    BOUNDS = "bounds"
    NORMAL = "normal"


@dataclass(frozen=True)
class PolicyActionNormalizerStatistics:
    """Immutable, revisioned statistics for exactly one checkpoint."""

    normalizer_revision: str
    normalization_type: ActionNormalizationType
    mask: Tuple[bool, ...]
    low: Optional[Tuple[float, ...]] = None
    high: Optional[Tuple[float, ...]] = None
    mean: Optional[Tuple[float, ...]] = None
    std: Optional[Tuple[float, ...]] = None
    clip_normalized_to_unit: bool = False
    source_uri: Optional[str] = None
    statistics_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.normalizer_revision:
            raise PolicyContractError("normalizer_revision is required; an unrevisioned normalizer is unreviewable.")
        if len(self.mask) != 7:
            raise PolicyContractError("Action normalizer mask must cover all seven physical action channels.")
        if any(not isinstance(flag, bool) for flag in self.mask):
            raise PolicyContractError("Action normalizer mask entries must be booleans.")
        if self.normalization_type is ActionNormalizationType.BOUNDS:
            if self.low is None or self.high is None:
                raise PolicyContractError("bounds normalization requires explicit low/high statistics.")
            if self.mean is not None or self.std is not None:
                raise PolicyContractError("bounds normalization must not also declare mean/std statistics.")
            _check_width("low", self.low)
            _check_width("high", self.high)
            for index, (low, high) in enumerate(zip(self.low, self.high)):
                if self.mask[index] and not high > low:
                    raise PolicyContractError("bounds normalization needs high > low on masked channel %d." % index)
        else:
            if self.mean is None or self.std is None:
                raise PolicyContractError("normal normalization requires explicit mean/std statistics.")
            if self.low is not None or self.high is not None:
                raise PolicyContractError("normal normalization must not also declare low/high statistics.")
            _check_width("mean", self.mean)
            _check_width("std", self.std)
            for index, deviation in enumerate(self.std):
                if self.mask[index] and not deviation > 0.0:
                    raise PolicyContractError("normal normalization needs std > 0 on masked channel %d." % index)
            if self.clip_normalized_to_unit:
                raise PolicyContractError("unit clipping is a bounds-normalization behaviour, not a normal one.")

    def payload(self) -> Dict[str, Any]:
        return {
            "normalizer_revision": self.normalizer_revision,
            "normalization_type": self.normalization_type.value,
            "mask": [bool(flag) for flag in self.mask],
            "low": list(self.low) if self.low is not None else None,
            "high": list(self.high) if self.high is not None else None,
            "mean": list(self.mean) if self.mean is not None else None,
            "std": list(self.std) if self.std is not None else None,
            "clip_normalized_to_unit": bool(self.clip_normalized_to_unit),
            "source_uri": self.source_uri,
            "statistics_sha256": self.statistics_sha256,
        }


def _check_width(name: str, values: Tuple[float, ...]) -> None:
    if len(values) != 7:
        raise PolicyContractError("Action normalizer %s must have seven channels, got %d." % (name, len(values)))
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise PolicyContractError("Action normalizer %s must contain finite real numbers." % name)


class PolicyActionNormalizer:
    """Apply one checkpoint's denormalization exactly once, at one boundary.

    Call counters are part of the public surface so a test can prove the
    transform was applied once per emitted row rather than twice (a doubled
    affine transform silently rescales every action).
    """

    def __init__(self, statistics: PolicyActionNormalizerStatistics) -> None:
        self.statistics = statistics
        self.denormalize_calls = 0
        self.normalize_calls = 0
        self.clipped_components = 0

    @property
    def revision(self) -> str:
        return self.statistics.normalizer_revision

    def counters(self) -> Dict[str, int]:
        return {
            "denormalize_calls": int(self.denormalize_calls),
            "normalize_calls": int(self.normalize_calls),
            "clipped_components": int(self.clipped_components),
        }

    def restore_counters(self, payload: Optional[Mapping[str, Any]]) -> None:
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            raise PolicyContractError("normalizer counters must be a mapping.")
        self.denormalize_calls = int(payload.get("denormalize_calls", 0))
        self.normalize_calls = int(payload.get("normalize_calls", 0))
        self.clipped_components = int(payload.get("clipped_components", 0))

    @staticmethod
    def _row(values: Sequence[float], direction: str) -> List[float]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise PolicyContractError("Action normalizer %s requires a numeric 7-D sequence." % direction)
        if len(values) != 7:
            raise PolicyContractError(
                "Action normalizer %s requires exactly seven channels, got %d; padding is forbidden."
                % (direction, len(values))
            )
        row: List[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PolicyContractError("Action normalizer %s received a nonnumeric channel." % direction)
            number = float(value)
            if not math.isfinite(number):
                raise PolicyContractError("Action normalizer %s received a nonfinite channel." % direction)
            row.append(number)
        return row

    def denormalize(self, values: Sequence[float]) -> Tuple[float, ...]:
        """Model-space row to physical 7-D Bridge action."""

        row = self._row(values, "denormalize")
        statistics = self.statistics
        output: List[float] = []
        for index, value in enumerate(row):
            if not statistics.mask[index]:
                output.append(value)
                continue
            if statistics.normalization_type is ActionNormalizationType.BOUNDS:
                assert statistics.low is not None and statistics.high is not None  # validated in __post_init__
                candidate = value
                if statistics.clip_normalized_to_unit:
                    clipped = min(max(value, -1.0), 1.0)
                    if clipped != value:
                        self.clipped_components += 1
                    candidate = clipped
                low = statistics.low[index]
                high = statistics.high[index]
                output.append(0.5 * (candidate + 1.0) * (high - low) + low)
            else:
                assert statistics.mean is not None and statistics.std is not None  # validated in __post_init__
                output.append(value * statistics.std[index] + statistics.mean[index])
        self.denormalize_calls += 1
        return tuple(output)

    def normalize(self, values: Sequence[float]) -> Tuple[float, ...]:
        """Physical 7-D Bridge action back to model space (the exact inverse)."""

        row = self._row(values, "normalize")
        statistics = self.statistics
        output: List[float] = []
        for index, value in enumerate(row):
            if not statistics.mask[index]:
                output.append(value)
                continue
            if statistics.normalization_type is ActionNormalizationType.BOUNDS:
                assert statistics.low is not None and statistics.high is not None  # validated in __post_init__
                low = statistics.low[index]
                high = statistics.high[index]
                output.append(2.0 * (value - low) / (high - low) - 1.0)
            else:
                assert statistics.mean is not None and statistics.std is not None  # validated in __post_init__
                output.append((value - statistics.mean[index]) / statistics.std[index])
        self.normalize_calls += 1
        return tuple(output)

    def denormalize_rows(self, rows: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise PolicyContractError("Action normalizer requires a sequence of 7-D rows.")
        return tuple(self.denormalize(row) for row in rows)


# ---------------------------------------------------------------------------
# Frame identity used only for stale-observation detection.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameIdentity:
    """An in-process identity for one observation frame.

    ``stable`` is False when the frame object exposed no pixel bytes, so the
    key falls back to a repr.  An unstable key is never written into durable
    provenance; it is only good enough to notice that the *same object* was
    presented twice in a row.
    """

    key: str
    stable: bool


def _jsonable_frame(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable_frame(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable_frame(item) for key, item in value.items()}
    raise TypeError("frame is not JSON-describable")


def frame_identity(frame: Any) -> FrameIdentity:
    """Best-effort stable identity for a frame, with the fallback flagged."""

    if frame is None:
        return FrameIdentity("absent", True)
    try:
        return FrameIdentity(str(image_pixel_hash(frame)["sha256"]), True)
    except TypeError:
        pass
    if isinstance(frame, str):
        return FrameIdentity(sha256_bytes(frame.encode("utf-8")), True)
    try:
        return FrameIdentity(canonical_json_sha256(_jsonable_frame(frame)), True)
    except TypeError:
        pass
    descriptor = "%s.%s:%s" % (type(frame).__module__, type(frame).__qualname__, repr(frame))
    return FrameIdentity("unstable:" + descriptor, False)


# ---------------------------------------------------------------------------
# Native proposals and golden-fixture certification.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeProposal:
    """One real native wrapper call and the rows it actually proposed.

    ``actions`` is the native proposal exactly as the model produced it, after
    the checkpoint's denormalization.  ``executed_actions`` is what the
    released wrapper would execute from that proposal: identical for a plain
    action-chunk policy, and a single temporally ensembled row for a wrapper
    whose ensembler consumes a fresh prediction every control tick.
    """

    policy_name: str
    actions: Tuple[Tuple[float, ...], ...]
    executed_actions: Tuple[Tuple[float, ...], ...]
    history_length: int
    normalizer_revision: str
    source_revision: Optional[str]
    observation_sha256: str
    observation_identity_stable: bool
    backend_calls: int
    wall_seconds: Optional[float]
    normalizer_counters: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_rows("native proposal", self.actions)
        _check_rows("executed", self.executed_actions)
        if self.history_length < 1:
            raise PolicyContractError("NativeProposal.history_length must be positive.")
        if self.backend_calls < 1:
            raise PolicyContractError("NativeProposal must report at least one actual backend call.")


def _check_rows(label: str, rows: Tuple[Tuple[float, ...], ...]) -> None:
    if not rows:
        raise PolicyContractError("%s rows cannot be empty; PLUMB never pads a proposal." % label)
    for row in rows:
        if len(row) != 7:
            raise PolicyContractError("%s rows must be physical 7-D Bridge actions, got width %d." % (label, len(row)))
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise PolicyContractError("%s rows must contain finite real numbers." % label)


def _normalise_sha256(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value[len("sha256:"):] if value.startswith("sha256:") else value
    candidate = candidate.strip().lower()
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        return None
    return "sha256:" + candidate


@dataclass(frozen=True)
class CertificationCheck:
    """The outcome of replaying a golden observation through a real wrapper."""

    policy_name: str
    passed: bool
    reasons: Tuple[str, ...]
    max_abs_error: Optional[float]
    observed_actions: Tuple[Tuple[float, ...], ...]
    expected_actions: Tuple[Tuple[float, ...], ...]
    observation_matches: Optional[bool]

    def payload(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
            "max_abs_error": self.max_abs_error,
            "observed_actions": [list(row) for row in self.observed_actions],
            "expected_actions": [list(row) for row in self.expected_actions],
            "observation_matches": self.observation_matches,
        }


CERTIFICATION_SCHEMA_KIND = "plumb_policy_action_certification"
CERTIFICATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PolicyCertification:
    """A golden action fixture plus its hash, loaded from JSON.

    Until one of these is attached, an adapter's ``certified_execute_prefix``
    stays ``None`` and its capability stays ``BLOCKED``.  The fixture records
    the golden observation hash, the expected physical action rows, and the
    source revision that produced them, so a later adapter, normalizer, or
    checkpoint change cannot inherit an older certification silently.
    """

    policy_name: str
    source_revision: str
    normalizer_revision: str
    golden_observation_sha256: str
    expected_actions: Tuple[Tuple[float, ...], ...]
    certified_execute_prefix: int
    native_proposal_horizon: int
    required_observation_history: int
    verified_terminal_prefixes: Tuple[int, ...] = ()
    temporal_ensembling: Optional[bool] = None
    expected_executed_actions: Optional[Tuple[Tuple[float, ...], ...]] = None
    tolerance: float = 1e-6
    recorded_at: Optional[str] = None
    fixture_path: Optional[str] = None
    fixture_sha256: Optional[str] = None
    certification_sha256: Optional[str] = None
    evidence_uris: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> Dict[str, Any]:
        """The canonically hashed body, excluding self-referential hashes."""

        return {
            "schema_version": CERTIFICATION_SCHEMA_VERSION,
            "kind": CERTIFICATION_SCHEMA_KIND,
            "policy_name": self.policy_name,
            "source_revision": self.source_revision,
            "normalizer_revision": self.normalizer_revision,
            "golden_observation_sha256": self.golden_observation_sha256,
            "expected_actions": [list(row) for row in self.expected_actions],
            "expected_executed_actions": (
                None if self.expected_executed_actions is None else [list(row) for row in self.expected_executed_actions]
            ),
            "certified_execute_prefix": int(self.certified_execute_prefix),
            "native_proposal_horizon": int(self.native_proposal_horizon),
            "required_observation_history": int(self.required_observation_history),
            "verified_terminal_prefixes": [int(value) for value in self.verified_terminal_prefixes],
            "temporal_ensembling": self.temporal_ensembling,
            "tolerance": float(self.tolerance),
            "recorded_at": self.recorded_at,
            "evidence_uris": list(self.evidence_uris),
            "metadata": dict(self.metadata),
        }

    def canonical_sha256(self) -> str:
        return canonical_json_sha256(self.payload())

    def errors(self) -> Tuple[str, ...]:
        """Every reason this fixture may not certify an execute prefix."""

        problems: List[str] = []
        if not self.policy_name:
            problems.append("policy_name is required")
        if not self.source_revision:
            problems.append("source_revision is required")
        if not self.normalizer_revision:
            problems.append("normalizer_revision is required")
        if _normalise_sha256(self.golden_observation_sha256) is None:
            problems.append("golden_observation_sha256 must be a sha256 digest")
        try:
            _check_rows("certified expected", self.expected_actions)
        except PolicyContractError as error:
            problems.append(str(error))
        if self.expected_executed_actions is not None:
            try:
                _check_rows("certified executed", self.expected_executed_actions)
            except PolicyContractError as error:
                problems.append(str(error))
        if self.native_proposal_horizon < 1:
            problems.append("native_proposal_horizon must be positive")
        elif len(self.expected_actions) != self.native_proposal_horizon:
            problems.append(
                "expected_actions has %d rows but native_proposal_horizon is %d; a fixture never pads a proposal"
                % (len(self.expected_actions), self.native_proposal_horizon)
            )
        if not 1 <= self.certified_execute_prefix <= max(1, self.native_proposal_horizon):
            problems.append("certified_execute_prefix must select a nonempty prefix of the native proposal")
        if self.required_observation_history < 1:
            problems.append("required_observation_history must be positive")
        for prefix in self.verified_terminal_prefixes:
            if not 1 <= prefix <= max(1, self.native_proposal_horizon):
                problems.append("verified terminal prefix %d is outside the native proposal" % prefix)
        if not isinstance(self.tolerance, (int, float)) or not math.isfinite(float(self.tolerance)) or self.tolerance <= 0:
            problems.append("tolerance must be a positive finite number")
        if self.temporal_ensembling is None:
            problems.append("temporal_ensembling must be recorded explicitly as true or false")
        elif self.temporal_ensembling and self.certified_execute_prefix != 1:
            problems.append(
                "temporal ensembling consumes a fresh prediction at every control tick, so the certified "
                "execute prefix must be 1, not %d" % self.certified_execute_prefix
            )
        if _normalise_sha256(self.fixture_sha256) is None:
            problems.append("fixture_sha256 (the hash of the certification artifact) is required")
        declared = _normalise_sha256(self.certification_sha256)
        if declared is None:
            problems.append("certification_sha256 is required")
        elif declared != self.canonical_sha256():
            problems.append("certification_sha256 does not match the canonical certification body")
        return tuple(problems)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        fixture_path: Optional[str] = None,
        fixture_sha256: Optional[str] = None,
    ) -> "PolicyCertification":
        if not isinstance(payload, Mapping):
            raise PolicyCertificationError("A policy certification fixture must be a JSON object.")
        if str(payload.get("kind")) != CERTIFICATION_SCHEMA_KIND:
            raise PolicyCertificationError(
                "Certification fixture kind must be %r, got %r." % (CERTIFICATION_SCHEMA_KIND, payload.get("kind"))
            )
        if int(payload.get("schema_version", 0)) != CERTIFICATION_SCHEMA_VERSION:
            raise PolicyCertificationError(
                "Unsupported certification schema_version %r." % payload.get("schema_version")
            )
        try:
            return cls(
                policy_name=str(payload["policy_name"]),
                source_revision=str(payload["source_revision"]),
                normalizer_revision=str(payload["normalizer_revision"]),
                golden_observation_sha256=str(payload["golden_observation_sha256"]),
                expected_actions=_rows_from_json(payload["expected_actions"]),
                expected_executed_actions=(
                    None
                    if payload.get("expected_executed_actions") is None
                    else _rows_from_json(payload["expected_executed_actions"])
                ),
                certified_execute_prefix=int(payload["certified_execute_prefix"]),
                native_proposal_horizon=int(payload["native_proposal_horizon"]),
                required_observation_history=int(payload["required_observation_history"]),
                verified_terminal_prefixes=tuple(int(value) for value in payload.get("verified_terminal_prefixes", ())),
                temporal_ensembling=(
                    None if payload.get("temporal_ensembling") is None else bool(payload["temporal_ensembling"])
                ),
                tolerance=float(payload.get("tolerance", 1e-6)),
                recorded_at=(None if payload.get("recorded_at") is None else str(payload["recorded_at"])),
                fixture_path=fixture_path,
                fixture_sha256=fixture_sha256,
                certification_sha256=(
                    None if payload.get("certification_sha256") is None else str(payload["certification_sha256"])
                ),
                evidence_uris=tuple(str(value) for value in payload.get("evidence_uris", ())),
                metadata=dict(payload.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PolicyCertificationError("Malformed policy certification fixture: %s" % error) from error

    @classmethod
    def from_json_file(cls, path: Any) -> "PolicyCertification":
        """Load a fixture and bind the artifact's own byte hash."""

        import json

        candidate = Path(str(path))
        try:
            raw = candidate.read_bytes()
        except OSError as error:
            raise PolicyCertificationError("Cannot read certification fixture %s: %s" % (candidate, error)) from error
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise PolicyCertificationError("Certification fixture %s is not valid JSON." % candidate) from error
        return cls.from_mapping(payload, fixture_path=str(candidate), fixture_sha256=sha256_bytes(raw))

    def matches(self, proposal: NativeProposal) -> CertificationCheck:
        """Compare a live native proposal against the golden fixture."""

        reasons: List[str] = []
        problems = self.errors()
        if problems:
            reasons.extend("invalid_certification:" + problem for problem in problems)
        expected_hash = _normalise_sha256(self.golden_observation_sha256)
        observed_hash = _normalise_sha256(proposal.observation_sha256)
        observation_matches: Optional[bool]
        if expected_hash is None or observed_hash is None:
            observation_matches = None
            reasons.append("observation_hash_unverifiable")
        else:
            observation_matches = expected_hash == observed_hash
            if not observation_matches:
                reasons.append("golden_observation_hash_mismatch")
        if proposal.normalizer_revision != self.normalizer_revision:
            reasons.append(
                "normalizer_revision_mismatch:%s!=%s" % (proposal.normalizer_revision, self.normalizer_revision)
            )
        if proposal.source_revision is not None and proposal.source_revision != self.source_revision:
            reasons.append("source_revision_mismatch:%s!=%s" % (proposal.source_revision, self.source_revision))
        if proposal.history_length != self.required_observation_history:
            reasons.append(
                "history_length_mismatch:%d!=%d" % (proposal.history_length, self.required_observation_history)
            )
        max_error: Optional[float] = None
        if len(proposal.actions) != len(self.expected_actions):
            reasons.append("row_count_mismatch:%d!=%d" % (len(proposal.actions), len(self.expected_actions)))
        else:
            max_error = 0.0
            for observed_row, expected_row in zip(proposal.actions, self.expected_actions):
                for observed, expected in zip(observed_row, expected_row):
                    max_error = max(max_error, abs(float(observed) - float(expected)))
            if max_error > float(self.tolerance):
                reasons.append("action_mismatch:max_abs_error=%.12g>%.12g" % (max_error, float(self.tolerance)))
        if self.expected_executed_actions is not None:
            if len(proposal.executed_actions) != len(self.expected_executed_actions):
                reasons.append(
                    "executed_row_count_mismatch:%d!=%d"
                    % (len(proposal.executed_actions), len(self.expected_executed_actions))
                )
            else:
                for observed_row, expected_row in zip(proposal.executed_actions, self.expected_executed_actions):
                    for observed, expected in zip(observed_row, expected_row):
                        if abs(float(observed) - float(expected)) > float(self.tolerance):
                            reasons.append("executed_action_mismatch")
                            break
        return CertificationCheck(
            policy_name=self.policy_name,
            passed=not reasons,
            reasons=tuple(reasons),
            max_abs_error=max_error,
            observed_actions=proposal.actions,
            expected_actions=self.expected_actions,
            observation_matches=observation_matches,
        )


def rows_from_native(value: Any, *, label: str, expected_rows: int, width: int = 7) -> Tuple[Tuple[float, ...], ...]:
    """Turn a native array/tensor proposal into exact physical rows.

    Only an unambiguous leading singleton batch dimension is removed, and the
    row count is never padded or truncated to fit ``expected_rows``.
    """

    candidate: Any = value
    for method_name in ("detach", "cpu", "numpy", "tolist"):
        method = getattr(candidate, method_name, None)
        if callable(method):
            try:
                result = method()
            except (TypeError, RuntimeError) as error:
                raise PolicyContractError("%s output could not be materialised: %s" % (label, error)) from error
            if result is not None:
                candidate = result
    while (
        isinstance(candidate, Sequence)
        and not isinstance(candidate, (str, bytes))
        and len(candidate) == 1
        and isinstance(candidate[0], Sequence)
        and not isinstance(candidate[0], (str, bytes))
        and len(candidate[0]) > 0
        and isinstance(candidate[0][0], Sequence)
        and not isinstance(candidate[0][0], (str, bytes))
    ):
        candidate = candidate[0]
    if isinstance(candidate, (str, bytes)) or not isinstance(candidate, Sequence):
        raise PolicyContractError("%s must return a numeric action array." % label)
    if len(candidate) != expected_rows:
        raise PolicyContractError(
            "%s returned %d action rows; exactly %d were declared. PLUMB never pads or truncates a native "
            "proposal." % (label, len(candidate), expected_rows)
        )
    rows: List[Tuple[float, ...]] = []
    for row in candidate:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise PolicyContractError("%s rows must be numeric sequences." % label)
        if len(row) != width:
            raise PolicyContractError("%s rows must have width %d, got %d." % (label, width, len(row)))
        values: List[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise PolicyContractError("%s contains a nonnumeric action component." % label)
            number = float(item)
            if not math.isfinite(number):
                raise PolicyContractError("%s contains a nonfinite action component." % label)
            values.append(number)
        rows.append(tuple(values))
    return tuple(rows)


@dataclass(frozen=True)
class NativeWrapperEntryPoint:
    """A pinned, reviewed entry point into a third-party policy wrapper.

    Some released wrappers (open-pi-zero, SuSIE) have constructor APIs that are
    repository-local rather than a documented public function, so PLUMB records
    the exact module/attribute and its reviewed revision instead of guessing a
    call signature.  The import happens only inside the load path.
    """

    module: str
    attribute: str
    loader_revision: Optional[str] = None
    source_uri: Optional[str] = None
    source_repo_path: Optional[str] = None

    def review_error(self) -> Optional[str]:
        if not self.module or not self.attribute:
            return "native wrapper entry point needs both a module and an attribute"
        if not self.loader_revision:
            return "native wrapper entry point needs a pinned reviewed loader revision"
        if self.source_repo_path is not None and not str(self.source_repo_path):
            return "source_repo_path must be a real path when supplied"
        return None

    def resolve(self) -> Any:
        """Import the reviewed factory lazily and return it."""

        import importlib
        import sys

        error = self.review_error()
        if error is not None:
            raise PolicyLoadError("Refusing to import an unpinned native wrapper: " + error + ".")
        if self.source_repo_path:
            repository = str(Path(str(self.source_repo_path)))
            if not Path(repository).is_dir():
                raise PolicyLoadError("Reviewed wrapper source repository %s is absent." % repository)
            if repository not in sys.path:
                sys.path.insert(0, repository)
        try:
            module = importlib.import_module(self.module)
        except ImportError as import_error:
            raise PolicyLoadError(
                "Reviewed native wrapper module %r is unavailable in this runtime; install the policy's own "
                "isolated image. Imports here are lazy." % self.module
            ) from import_error
        factory = getattr(module, self.attribute, None)
        if not callable(factory):
            raise PolicyLoadError(
                "Reviewed native wrapper %s.%s is not callable." % (self.module, self.attribute)
            )
        return factory


def inference_context(torch_module: Any) -> Any:
    """``inference_mode``/``no_grad`` if the runtime has one, else a no-op."""

    import contextlib

    factory = getattr(torch_module, "inference_mode", None) or getattr(torch_module, "no_grad", None)
    return factory() if callable(factory) else contextlib.nullcontext()


def peak_memory_bytes(torch_module: Any) -> Optional[int]:
    """Observed CUDA peak memory, or ``None``.  Never rewritten as zero."""

    cuda = getattr(torch_module, "cuda", None)
    try:
        if cuda is not None and cuda.is_available():
            return int(cuda.max_memory_allocated())
    except (AttributeError, RuntimeError):
        return None
    return None


def immutable_revision(value: Any) -> bool:
    """Whether ``value`` is a full 40-character hexadecimal commit."""

    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def mint_certification_payload(
    proposal: NativeProposal,
    *,
    certified_execute_prefix: int,
    temporal_ensembling: bool,
    required_observation_history: Optional[int] = None,
    source_revision: Optional[str] = None,
    verified_terminal_prefixes: Sequence[int] = (),
    tolerance: float = 1e-6,
    recorded_at: Optional[str] = None,
    evidence_uris: Sequence[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON body of a golden action fixture from one real native call.

    This is the only supported way to create a certification: the rows, the
    golden observation hash, the normalizer revision, and the source revision
    all come from an actual wrapper call rather than from hand-edited JSON.  The
    caller writes the returned object to disk and loads it back with
    :meth:`PolicyCertification.from_json_file`, which binds the artifact hash.
    """

    if not isinstance(proposal, NativeProposal):
        raise PolicyCertificationError("A golden fixture must be minted from a real NativeProposal.")
    revision = source_revision if source_revision is not None else proposal.source_revision
    if not revision:
        raise PolicyCertificationError(
            "A golden fixture needs the source revision that produced it; refusing to record an unpinned "
            "certification."
        )
    if not proposal.observation_identity_stable:
        raise PolicyCertificationError(
            "The golden observation had no stable pixel identity, so its hash cannot be recorded durably. "
            "Supply a real image array or bytes."
        )
    draft = PolicyCertification(
        policy_name=proposal.policy_name,
        source_revision=str(revision),
        normalizer_revision=proposal.normalizer_revision,
        golden_observation_sha256=proposal.observation_sha256,
        expected_actions=proposal.actions,
        expected_executed_actions=proposal.executed_actions,
        certified_execute_prefix=int(certified_execute_prefix),
        native_proposal_horizon=len(proposal.actions),
        required_observation_history=(
            proposal.history_length if required_observation_history is None else int(required_observation_history)
        ),
        verified_terminal_prefixes=tuple(int(value) for value in verified_terminal_prefixes),
        temporal_ensembling=bool(temporal_ensembling),
        tolerance=float(tolerance),
        recorded_at=recorded_at,
        evidence_uris=tuple(str(value) for value in evidence_uris),
        metadata=dict(metadata or {}),
    )
    payload = draft.payload()
    payload["certification_sha256"] = draft.canonical_sha256()
    return payload


def _rows_from_json(value: Any) -> Tuple[Tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PolicyCertificationError("Certified action rows must be a JSON array of arrays.")
    rows: List[Tuple[float, ...]] = []
    for row in value:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise PolicyCertificationError("Each certified action row must be a JSON array.")
        rows.append(tuple(float(item) for item in row))
    return tuple(rows)


# ---------------------------------------------------------------------------
# Shared base class for real native multi-action adapters.
# ---------------------------------------------------------------------------


class PolicyExecutionMode(str, Enum):
    """Whether this adapter instance may drive a rollout.

    ``CERTIFIED`` is the default and requires a valid golden fixture before any
    action is produced.  ``FIXTURE_CERTIFICATION`` exists solely so the golden
    fixture can be minted in the policy's own isolated image; in that mode the
    adapter refuses to emit a :class:`~plumb.rollout.PolicyControlPlan`,
    because its verified prefix would be fabricated.
    """

    CERTIFIED = "certified"
    FIXTURE_CERTIFICATION = "fixture_certification"


NATIVE_STATE_SCHEMA_KIND = "plumb_native_policy_state"
NATIVE_STATE_SCHEMA_VERSION = 1


class NativePolicyAdapter:
    """Certified-execution scaffolding shared by the real native adapters.

    Subclasses supply a lazily imported runtime and one ``_native_proposal``
    implementation.  This class owns the parts that are identical — and easy to
    get dangerously wrong — across policies: refusing to act without a golden
    fixture, refusing a stale observation, refusing to invent an execute
    prefix, and producing durable, restorable state for resume.
    """

    base_contract: PolicyContract
    source_urls: Tuple[str, ...] = ()

    def __init__(
        self,
        *,
        normalizer: PolicyActionNormalizer,
        certification: Optional[PolicyCertification] = None,
        execution_mode: PolicyExecutionMode = PolicyExecutionMode.CERTIFIED,
        gripper_convention: GripperPolarityConvention = UNRESOLVED_BRIDGE_GRIPPER_CONVENTION,
    ) -> None:
        if not isinstance(normalizer, PolicyActionNormalizer):
            raise PolicyContractError("A revisioned PolicyActionNormalizer is required.")
        if not isinstance(execution_mode, PolicyExecutionMode):
            raise PolicyContractError("execution_mode must be a PolicyExecutionMode.")
        self.normalizer = normalizer
        self.certification = certification
        self.execution_mode = execution_mode
        self.gripper_convention = gripper_convention
        self._seed: Optional[int] = None
        self._step = 0
        self._backend_calls = 0
        self._last_frame_key: Optional[str] = None
        self._last_frame_stable = True
        self._last_timestamp: Optional[float] = None
        self.last_proposal: Optional[NativeProposal] = None

    # -- contract and capability ------------------------------------------------

    @property
    def policy_name(self) -> str:
        return self.base_contract.name

    def certification_errors(self) -> Tuple[str, ...]:
        """Why the attached fixture cannot certify this adapter, if it cannot."""

        certification = self.certification
        if certification is None:
            return ("no_certification_artifact",)
        problems = list(certification.errors())
        base = self.base_contract
        if certification.policy_name != base.name:
            problems.append("policy_name_mismatch:%s!=%s" % (certification.policy_name, base.name))
        if certification.native_proposal_horizon != base.native_proposal_horizon:
            problems.append(
                "native_proposal_horizon_mismatch:%d!=%d"
                % (certification.native_proposal_horizon, base.native_proposal_horizon)
            )
        if certification.required_observation_history != base.required_observation_history:
            problems.append(
                "required_observation_history_mismatch:%d!=%d"
                % (certification.required_observation_history, base.required_observation_history)
            )
        if certification.temporal_ensembling is not None and certification.temporal_ensembling != base.temporal_ensembling:
            problems.append(
                "temporal_ensembling_mismatch:%s!=%s"
                % (certification.temporal_ensembling, base.temporal_ensembling)
            )
        if certification.normalizer_revision != self.normalizer.revision:
            problems.append(
                "normalizer_revision_mismatch:%s!=%s" % (certification.normalizer_revision, self.normalizer.revision)
            )
        return tuple(problems)

    def valid_certification(self) -> Optional[PolicyCertification]:
        return None if self.certification_errors() else self.certification

    @property
    def contract(self) -> PolicyContract:
        """The declared contract, certified only when a valid fixture exists."""

        certification = self.valid_certification()
        if certification is None:
            return self.base_contract
        limitation = (
            "Golden action fixture %s certifies execute prefix %d for source revision %s. Gate B and a "
            "one-tick feedback fixture are still required before any named-policy result is qualified."
            % (certification.fixture_sha256, certification.certified_execute_prefix, certification.source_revision)
        )
        return replace(
            self.base_contract,
            certified_execute_prefix=certification.certified_execute_prefix,
            implementation_status=CapabilityStatus.READY_UNQUALIFIED,
            limitation=limitation,
        )

    def _profile_errors(self) -> Tuple[str, ...]:
        """Subclass hook: pinned-asset and review problems that block loading."""

        return ()

    def _availability_error(self) -> Optional[str]:
        """Subclass hook: why the pinned local assets are absent right now."""

        return None

    def _capability_details(self) -> Dict[str, Any]:
        """Subclass hook: extra, non-fabricated capability evidence."""

        return {}

    def _base_details(self) -> Dict[str, Any]:
        certification = self.certification
        return {
            "policy": self.base_contract.name,
            "execution_mode": self.execution_mode.value,
            "native_proposal_horizon": self.base_contract.native_proposal_horizon,
            "required_observation_history": self.base_contract.required_observation_history,
            "requires_proprio": self.base_contract.requires_proprio,
            "temporal_ensembling": self.base_contract.temporal_ensembling,
            "certified_execute_prefix": self.contract.certified_execute_prefix,
            "normalizer_revision": self.normalizer.revision,
            "normalization_type": self.normalizer.statistics.normalization_type.value,
            "gripper_convention_id": self.gripper_convention.convention_id,
            "gripper_polarity_resolved": self.gripper_convention.resolved,
            "certification_fixture_sha256": None if certification is None else certification.fixture_sha256,
            "certification_sha256": None if certification is None else certification.certification_sha256,
            "certification_errors": list(self.certification_errors()),
        }

    def capability(self) -> CapabilityResult:
        details = self._base_details()
        details.update(self._capability_details())
        profile_problems = self._profile_errors()
        if profile_problems:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="%s loading is blocked: %s." % (self.base_contract.name, "; ".join(profile_problems)),
                source_verified=bool(self.source_urls),
                evidence_uris=self.source_urls,
                details=details,
            )
        certification_problems = self.certification_errors()
        if certification_problems:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason=(
                    "%s has no valid golden action fixture, so its executed prefix stays uncertified and it "
                    "will not emit actions: %s."
                )
                % (self.base_contract.name, "; ".join(certification_problems)),
                source_verified=bool(self.source_urls),
                evidence_uris=self.source_urls,
                details=details,
            )
        availability = self._availability_error()
        if availability is not None:
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="%s pinned local assets are unavailable: %s." % (self.base_contract.name, availability),
                source_verified=bool(self.source_urls),
                evidence_uris=self.source_urls,
                details=details,
            )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "%s native execution is configured against a pinned checkpoint, a revisioned normalizer, and a "
                "golden action fixture. It remains unqualified until Gate B and one-tick feedback fixtures pass."
            )
            % self.base_contract.name,
            source_verified=bool(self.source_urls),
            evidence_uris=self.source_urls,
            details=details,
        )

    # -- reset and durable state ------------------------------------------------

    def _reset_native(self, seed: int) -> None:
        """Subclass hook: clear wrapper-owned state for a new episode."""

    def reset(self, seed: int) -> Dict[str, Any]:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PolicyContractError("%s reset requires an integer seed." % self.base_contract.name)
        self._seed = int(seed)
        self._step = 0
        self._backend_calls = 0
        self._last_frame_key = None
        self._last_frame_stable = True
        self._last_timestamp = None
        self.last_proposal = None
        self.normalizer.denormalize_calls = 0
        self.normalizer.normalize_calls = 0
        self.normalizer.clipped_components = 0
        self._reset_native(int(seed))
        certification = self.certification
        return {
            "policy": self.base_contract.name,
            "seed": int(seed),
            "reset_rule": self.base_contract.reset_rule,
            "rng_rule": self.base_contract.rng_rule,
            "normalizer_revision": self.normalizer.revision,
            "certification_sha256": None if certification is None else certification.certification_sha256,
            "execution_mode": self.execution_mode.value,
        }

    def _snapshot_extra(self) -> Dict[str, Any]:
        """Subclass hook: JSON-restorable wrapper state."""

        return {}

    def _restore_extra(self, payload: Mapping[str, Any]) -> None:
        """Subclass hook: restore what ``_snapshot_extra`` recorded."""

    def snapshot_state(self) -> Dict[str, Any]:
        certification = self.certification
        return {
            "schema_version": NATIVE_STATE_SCHEMA_VERSION,
            "kind": NATIVE_STATE_SCHEMA_KIND,
            "policy": self.base_contract.name,
            "seed": self._seed,
            "step": int(self._step),
            "backend_calls": int(self._backend_calls),
            "last_frame_identity": self._last_frame_key if self._last_frame_stable else None,
            "last_frame_identity_stable": bool(self._last_frame_stable),
            "last_timestamp": self._last_timestamp,
            "normalizer_revision": self.normalizer.revision,
            "normalizer_counters": self.normalizer.counters(),
            "certification_sha256": None if certification is None else certification.certification_sha256,
            "extra": self._snapshot_extra(),
        }

    def restore_state(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            raise PolicyContractError("%s restore_state requires a snapshot mapping." % self.base_contract.name)
        if str(payload.get("kind")) != NATIVE_STATE_SCHEMA_KIND:
            raise PolicyContractError("Snapshot kind must be %r." % NATIVE_STATE_SCHEMA_KIND)
        if int(payload.get("schema_version", 0)) != NATIVE_STATE_SCHEMA_VERSION:
            raise PolicyContractError("Unsupported native policy snapshot schema_version.")
        if str(payload.get("policy")) != self.base_contract.name:
            raise PolicyContractError(
                "Snapshot belongs to %r, not %r; refusing a cross-policy resume."
                % (payload.get("policy"), self.base_contract.name)
            )
        if str(payload.get("normalizer_revision")) != self.normalizer.revision:
            raise PolicyContractError(
                "Snapshot normalizer revision %r does not match the live normalizer %r."
                % (payload.get("normalizer_revision"), self.normalizer.revision)
            )
        certification = self.certification
        recorded = payload.get("certification_sha256")
        live = None if certification is None else certification.certification_sha256
        if recorded != live:
            raise PolicyContractError(
                "Snapshot certification %r does not match the attached certification %r." % (recorded, live)
            )
        seed = payload.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise PolicyContractError("Snapshot seed must be an integer or null.")
        self._seed = None if seed is None else int(seed)
        self._step = int(payload.get("step", 0))
        self._backend_calls = int(payload.get("backend_calls", 0))
        identity = payload.get("last_frame_identity")
        self._last_frame_key = None if identity is None else str(identity)
        self._last_frame_stable = bool(payload.get("last_frame_identity_stable", True))
        timestamp = payload.get("last_timestamp")
        self._last_timestamp = None if timestamp is None else float(timestamp)
        self.normalizer.restore_counters(payload.get("normalizer_counters"))
        extra = payload.get("extra", {})
        if not isinstance(extra, Mapping):
            raise PolicyContractError("Snapshot 'extra' must be a mapping.")
        self._restore_extra(extra)

    # -- observation validation -------------------------------------------------

    def _check_goal_image(self, observation: PolicyObservation) -> None:
        """Subclass hook: goal-conditioned policies require a real goal image."""

    def _check_extra_observation(self, observation: PolicyObservation) -> None:
        """Subclass hook: per-policy observation rules."""

    def _validate_observation(self, observation: PolicyObservation) -> FrameIdentity:
        name = self.base_contract.name
        if not isinstance(observation, PolicyObservation):
            raise PolicyContractError("%s requires a canonical PolicyObservation." % name)
        if not isinstance(observation.prompt, str) or not observation.prompt.strip():
            raise PolicyContractError("%s requires a nonempty exact task instruction." % name)
        required = self.base_contract.required_observation_history
        if len(observation.image_history) != required:
            raise PolicyContractError(
                "%s requires exactly %d observation frames, got %d; the controller never pads or truncates a "
                "native history." % (name, required, len(observation.image_history))
            )
        if any(frame is None for frame in observation.image_history):
            raise PolicyContractError("%s observation history contains an absent frame." % name)
        if self.base_contract.requires_proprio:
            self._validate_proprio(observation.proprio)
        elif observation.proprio is not None:
            raise PolicyContractError(
                "%s declares requires_proprio=False; its native API must not receive proprioception." % name
            )
        timestamp = observation.timestamp
        if timestamp is not None:
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)):
                raise PolicyContractError("%s observation timestamp must be a finite number." % name)
            if self._last_timestamp is not None and float(timestamp) <= self._last_timestamp:
                raise PolicyContractError(
                    "%s received observation timestamp %r which does not advance past the previously consumed "
                    "tick %r; a repeated tick means the observation was not refreshed."
                    % (name, float(timestamp), self._last_timestamp)
                )
        self._check_goal_image(observation)
        self._check_extra_observation(observation)
        identity = frame_identity(observation.image_history[-1])
        if self._last_frame_key is not None and identity.key == self._last_frame_key:
            raise PolicyContractError(
                "%s was handed the same newest observation frame as its previous native call (%s). A native "
                "multi-action proposal must never be produced by requerying a policy against a stale image; "
                "obtain a freshly generated frame first." % (name, identity.key)
            )
        return identity

    def _validate_proprio(self, proprio: Optional[Tuple[float, ...]]) -> Tuple[float, ...]:
        name = self.base_contract.name
        if proprio is None:
            raise PolicyContractError(
                "%s declares requires_proprio=True and refuses to run without a refreshed proprioception "
                "vector; a missing value is never zero-filled." % name
            )
        if isinstance(proprio, (str, bytes)) or not isinstance(proprio, Sequence):
            raise PolicyContractError("%s proprioception must be a numeric sequence." % name)
        if len(proprio) != 8:
            raise PolicyContractError(
                "%s expects the canonical 8-D Bridge state (x,y,z,rx,ry,rz,0,gripper), got %d values."
                % (name, len(proprio))
            )
        values: List[float] = []
        for value in proprio:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise PolicyContractError("%s proprioception must contain finite real numbers." % name)
            values.append(float(value))
        if values[6] != 0.0:
            raise PolicyContractError(
                "%s proprioception channel 6 is the Bridge convention's structural zero, got %r." % (name, values[6])
            )
        if all(value == 0.0 for value in values):
            raise PolicyContractError(
                "%s received an all-zero proprioception vector. The Bridge base origin is not a reachable "
                "end-effector pose, so this is a zero-fill, not a measurement; refusing to run." % name
            )
        # Validate the state gripper in state units so an action gripper value
        # cannot be smuggled into the proprio channel.
        GripperStateValue(values[7])
        return tuple(values)

    def state_gripper(self, proprio: Sequence[float]) -> GripperStateValue:
        """Read the proprio gripper channel as an explicit state-unit value."""

        if len(proprio) != 8:
            raise PolicyContractError("Bridge proprioception must be 8-D to read its gripper channel.")
        return GripperStateValue(float(proprio[7]))

    @staticmethod
    def action_gripper(action: Sequence[float]) -> GripperActionValue:
        """Read a 7-D action's gripper channel as an explicit action-unit value."""

        if len(action) != 7:
            raise PolicyContractError("A physical Bridge action must be 7-D to read its gripper channel.")
        return GripperActionValue(float(action[6]))

    # -- native proposal --------------------------------------------------------

    def _native_proposal(self, observation: PolicyObservation, identity: FrameIdentity) -> NativeProposal:
        raise NotImplementedError("Subclasses implement one real native wrapper call.")

    def propose(self, observation: PolicyObservation) -> NativeProposal:
        """Make exactly one native wrapper call for a fresh observation.

        Refused in ``CERTIFIED`` mode without a valid fixture, so an
        uncertified adapter cannot emit an action through any path.
        """

        if self.execution_mode is PolicyExecutionMode.CERTIFIED:
            problems = self.certification_errors()
            if problems:
                raise PolicyCertificationError(
                    "%s refuses to produce actions in certified mode without a valid golden action fixture: %s."
                    % (self.base_contract.name, "; ".join(problems))
                )
        identity = self._validate_observation(observation)
        proposal = self._native_proposal(observation, identity)
        self._check_proposal(proposal)
        self._last_frame_key = identity.key
        self._last_frame_stable = identity.stable
        if observation.timestamp is not None:
            self._last_timestamp = float(observation.timestamp)
        self._step += 1
        self._backend_calls += proposal.backend_calls
        self.last_proposal = proposal
        return proposal

    def _check_proposal(self, proposal: NativeProposal) -> None:
        name = self.base_contract.name
        if not isinstance(proposal, NativeProposal):
            raise PolicyContractError("%s native call must return a NativeProposal." % name)
        if proposal.policy_name != name:
            raise PolicyContractError("Native proposal is labelled %r, not %r." % (proposal.policy_name, name))
        expected = self.base_contract.native_proposal_horizon
        if len(proposal.actions) != expected:
            raise PolicyContractError(
                "%s declares a %d-action native proposal but its wrapper returned %d rows; PLUMB neither pads "
                "nor truncates a native horizon." % (name, expected, len(proposal.actions))
            )
        if proposal.history_length != self.base_contract.required_observation_history:
            raise PolicyContractError(
                "%s native proposal declares history %d, contract requires %d."
                % (name, proposal.history_length, self.base_contract.required_observation_history)
            )
        if proposal.normalizer_revision != self.normalizer.revision:
            raise PolicyContractError(
                "%s native proposal reports normalizer %r, live normalizer is %r."
                % (name, proposal.normalizer_revision, self.normalizer.revision)
            )
        if self.base_contract.temporal_ensembling:
            if len(proposal.executed_actions) != 1:
                raise PolicyContractError(
                    "%s runs temporal ensembling, so exactly one ensembled row may be executed per native call, "
                    "not %d." % (name, len(proposal.executed_actions))
                )
        elif proposal.executed_actions != proposal.actions:
            raise PolicyContractError(
                "%s does not run temporal ensembling, so its executed rows must be its native proposal rows." % name
            )

    def certify(self, golden_observation: PolicyObservation) -> CertificationCheck:
        """Replay the golden observation and compare it with the fixture."""

        certification = self.certification
        if certification is None:
            raise PolicyCertificationError(
                "%s has no certification fixture to check against." % self.base_contract.name
            )
        identity = self._validate_observation(golden_observation)
        proposal = self._native_proposal(golden_observation, identity)
        self._check_proposal(proposal)
        self.last_proposal = proposal
        return certification.matches(proposal)

    # -- rollout interface ------------------------------------------------------

    def plan_control(self, observation: PolicyObservation, remaining: int) -> Any:
        """Return a :class:`~plumb.rollout.PolicyControlPlan` for the controller."""

        from plumb.rollout import PolicyControlPlan  # local import keeps plumb.policies leaf-light

        name = self.base_contract.name
        if self.execution_mode is not PolicyExecutionMode.CERTIFIED:
            raise PolicyCertificationError(
                "%s is in %s mode, which mints golden fixtures. It will not hand a rollout a verified prefix "
                "it has not certified." % (name, self.execution_mode.value)
            )
        problems = self.certification_errors()
        if problems:
            raise PolicyCertificationError(
                "%s cannot supply a verified control plan without a valid golden action fixture: %s."
                % (name, "; ".join(problems))
            )
        certification = self.certification
        assert certification is not None  # guaranteed by certification_errors()
        if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 1:
            raise PolicyContractError("%s plan_control requires a positive remaining action count." % name)
        proposal = self.propose(observation)
        rows = proposal.executed_actions
        prefix = certification.certified_execute_prefix
        if len(rows) < prefix:
            raise PolicyContractError(
                "%s certified execute prefix %d exceeds the %d executable rows its wrapper produced."
                % (name, prefix, len(rows))
            )
        # The native call's own record travels with the plan so the rollout
        # segment artifact keeps the wrapper-level evidence; plan-level keys win
        # on any collision.
        metadata: Dict[str, Any] = dict(proposal.metadata)
        metadata.update(
            {
                "policy": name,
                "native_proposal_horizon": self.base_contract.native_proposal_horizon,
                "native_proposal_rows": len(proposal.actions),
                "temporal_ensembling": self.base_contract.temporal_ensembling,
                "normalizer_revision": proposal.normalizer_revision,
                "normalization_type": self.normalizer.statistics.normalization_type.value,
                "source_revision": proposal.source_revision,
                "certification_sha256": certification.certification_sha256,
                "certification_fixture_sha256": certification.fixture_sha256,
                "observation_sha256": proposal.observation_sha256,
                "observation_identity_stable": proposal.observation_identity_stable,
                "backend_calls": proposal.backend_calls,
                "remaining_actions_at_call": int(remaining),
                "qualified": False,
            }
        )
        return PolicyControlPlan(
            actions=rows,
            verified_prefix=prefix,
            history_length=self.base_contract.required_observation_history,
            verified_terminal_prefixes=certification.verified_terminal_prefixes,
            metadata=metadata,
        )
