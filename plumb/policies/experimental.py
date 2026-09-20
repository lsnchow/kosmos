"""Source-shaped one-turn policy calls for the unscored comparison wall.

This module deliberately sits beside, rather than inside, the certified study
adapters.  It is the narrow bridge used by the Baseten comparison controller:
one request supplies one *current* RGB observation and receives one action to
send to the world model.  It never fills a policy horizon with repeated calls
or with a stale frame.

The three supported policies have materially different source boundaries:

* OpenVLA exposes one native 7-D action for one RGB observation.
* The reviewed MiniVLA source decodes a VQ latent internally but exposes only
  ``ret_action[:, 0]`` -- one 7-D action, not seven executable actions.
* Octo-Small exposes four physical proposal rows but source temporal
  ensembling chooses exactly one executed row.

No result from this module is a certification, score, or qualification.  In
particular, a blocked asset/license gate is returned as ``blocked`` and is
never bypassed by a test adapter or a guessed state vector.

The router has no mutable per-cell policy cache.  A caller supplies a fresh
adapter factory for each turn; immutable model weights may be shared by that
factory's deployment-owned loader, but source task/history/sampler state is
restored from the supplied serialised snapshot every time.  That avoids the
common error where a replica-global Octo history leaks from one wall cell to
another.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from plumb.adapters.contracts import CapabilityStatus, PolicyObservation

from .provenance import image_pixel_hash


EXPERIMENTAL_POLICY_TURN_SCHEMA = "plumb-experimental-policy-turn-v1"
_OCTO_SOURCE_STATE_SCHEMA = "plumb-octo-v0.1-source-state-v1"
_RGB_SNAPSHOT_ENCODING = "rgb_uint8_base64"
_ACTION_DIMENSIONS = 7
_OCTO_HISTORY = 2
_OCTO_PROPOSAL_HISTORY = 4


class ExperimentalPolicyTurnStatus(str, Enum):
    """Operational state of one request; this is not an outcome score."""

    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class ExperimentalPolicyTurnError(ValueError):
    """A request/snapshot cannot represent a source-faithful policy turn."""


class ExperimentalPolicyId(str, Enum):
    OPENVLA = "openvla"
    MINIVLA = "minivla"
    OCTO_SMALL = "octo-small"


_POLICY_ALIASES = {
    "openvla": ExperimentalPolicyId.OPENVLA,
    "openvla-7b": ExperimentalPolicyId.OPENVLA,
    "minivla": ExperimentalPolicyId.MINIVLA,
    "octo-small": ExperimentalPolicyId.OCTO_SMALL,
    "octo-small-v1": ExperimentalPolicyId.OCTO_SMALL,
    "octo-small-v1.0": ExperimentalPolicyId.OCTO_SMALL,
}


def canonical_experimental_policy_id(value: object) -> ExperimentalPolicyId:
    """Return a stable wall policy ID without accepting a silent substitute."""

    if isinstance(value, ExperimentalPolicyId):
        return value
    key = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    try:
        return _POLICY_ALIASES[key]
    except KeyError as error:
        raise ExperimentalPolicyTurnError(
            "Experimental policy_id must be one of OpenVLA, MiniVLA, or Octo-Small; got %r." % value
        ) from error


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite_action_row(value: object, *, label: str) -> Tuple[float, float, float, float, float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ExperimentalPolicyTurnError("%s must be a numeric 7-D action, not text." % label)
    try:
        row = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ExperimentalPolicyTurnError("%s must be a numeric 7-D action." % label) from error
    if len(row) != _ACTION_DIMENSIONS or not all(math.isfinite(item) for item in row):
        raise ExperimentalPolicyTurnError("%s must contain exactly seven finite values." % label)
    return row  # type: ignore[return-value]


def _finite_action_rows(value: object, *, label: str, expected_rows: Optional[int] = None) -> Tuple[Tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)):
        raise ExperimentalPolicyTurnError("%s must be rows of numeric actions, not text." % label)
    try:
        rows = tuple(_finite_action_row(row, label="%s row" % label) for row in value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ExperimentalPolicyTurnError("%s must be a sequence of action rows." % label) from error
    if not rows:
        raise ExperimentalPolicyTurnError("%s cannot be empty." % label)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ExperimentalPolicyTurnError("%s must contain exactly %d rows, got %d." % (label, expected_rows, len(rows)))
    return rows


def _json_copy_mapping(value: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    """Copy a mapping only when it is safe to put in a streamed NDJSON record."""

    try:
        copied = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError) as error:
        raise ExperimentalPolicyTurnError("%s must contain JSON-serialisable values." % label) from error
    if not isinstance(copied, dict):  # Defensive; json object decoding always gives dict here.
        raise ExperimentalPolicyTurnError("%s must be a JSON object." % label)
    return copied


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentalPolicyTurnError("%s is required." % label)
    return value


def _frame_hash(image: Any) -> str:
    try:
        value = image_pixel_hash(image).get("sha256")
    except (TypeError, ValueError) as error:
        raise ExperimentalPolicyTurnError(
            "A source-current RGB frame with stable array/pixel bytes is required; a frame hash could not be produced."
        ) from error
    if not isinstance(value, str) or not value:
        raise ExperimentalPolicyTurnError("A stable source-current RGB frame hash is required.")
    return value


def _rgb_snapshot(image: Any, *, frame_sha256: str) -> Dict[str, Any]:
    """Encode an already-decoded uint8 HWC RGB frame without transforming it.

    Octo needs the exact last two observations after a worker process/restart.
    A JSON metadata reference is insufficient to reconstruct source history;
    this stores those *received* RGB bytes and their shape.  It refuses PIL,
    float, CHW, or inferred frames instead of changing preprocessing at this
    boundary.
    """

    shape = getattr(image, "shape", None)
    dtype = getattr(image, "dtype", None)
    tobytes = getattr(image, "tobytes", None)
    if not isinstance(shape, tuple) or len(shape) != 3 or tuple(int(item) for item in shape)[2] != 3:
        raise ExperimentalPolicyTurnError("Octo state snapshots require an HWC RGB array with shape [height, width, 3].")
    if str(dtype) != "uint8" or not callable(tobytes):
        raise ExperimentalPolicyTurnError("Octo state snapshots require an unmodified uint8 RGB array.")
    dimensions = tuple(int(item) for item in shape)
    if dimensions[0] < 1 or dimensions[1] < 1:
        raise ExperimentalPolicyTurnError("Octo state snapshots require a nonempty RGB image.")
    raw = bytes(tobytes())
    expected_bytes = dimensions[0] * dimensions[1] * dimensions[2]
    if len(raw) != expected_bytes:
        raise ExperimentalPolicyTurnError("Octo RGB frame byte length does not match its declared shape.")
    return {
        "encoding": _RGB_SNAPSHOT_ENCODING,
        "shape": list(dimensions),
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "frame_sha256": frame_sha256,
    }


def _decode_rgb_snapshot(value: object) -> Any:
    """Restore the exact bytes held in an Octo snapshot as a uint8 HWC array."""

    if not isinstance(value, Mapping):
        raise ExperimentalPolicyTurnError("Octo snapshot history must carry an RGB frame object.")
    if value.get("encoding") != _RGB_SNAPSHOT_ENCODING:
        raise ExperimentalPolicyTurnError("Octo snapshot history uses an unsupported RGB encoding.")
    shape = value.get("shape")
    data = value.get("data_base64")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) != 3:
        raise ExperimentalPolicyTurnError("Octo snapshot RGB shape must have exactly three dimensions.")
    try:
        dimensions = tuple(int(item) for item in shape)
        raw = base64.b64decode(data, validate=True) if isinstance(data, str) else b""
    except (TypeError, ValueError) as error:
        raise ExperimentalPolicyTurnError("Octo snapshot RGB encoding is malformed.") from error
    if dimensions[0] < 1 or dimensions[1] < 1 or dimensions[2] != 3 or len(raw) != dimensions[0] * dimensions[1] * 3:
        raise ExperimentalPolicyTurnError("Octo snapshot RGB bytes do not match an HWC RGB uint8 image.")
    try:
        import numpy as np  # Imported only when a real Octo history is restored.
    except ImportError as error:
        raise ExperimentalPolicyTurnError("Octo source history restore requires NumPy in the policy worker.") from error
    image = np.frombuffer(raw, dtype=np.uint8).reshape(dimensions).copy()
    if _frame_hash(image) != value.get("frame_sha256"):
        raise ExperimentalPolicyTurnError("Octo snapshot RGB hash does not match its stored source frame hash.")
    return image


@dataclass(frozen=True)
class ExperimentalPolicyTurnRequest:
    """One policy feedback boundary.

    ``image_history`` is intentionally named to match worker payloads but must
    contain exactly one fresh image.  Octo's source history comes solely from
    ``prior_snapshot``; callers cannot inject a made-up padded history.
    """

    policy_id: object
    prompt: str
    image_history: Tuple[Any, ...]
    proprio: Optional[Tuple[float, ...]] = None
    policy_seed: Optional[int] = None
    prior_snapshot: Optional[Mapping[str, Any]] = None
    identifiers: Mapping[str, Any] = field(default_factory=dict)

    def canonical_policy_id(self) -> ExperimentalPolicyId:
        return canonical_experimental_policy_id(self.policy_id)


@dataclass(frozen=True)
class ExperimentalPolicyTurnResult:
    """Auditable source action rows plus a JSON-safe continuation snapshot."""

    status: str
    raw_proposal_rows: Tuple[Tuple[float, ...], ...] = ()
    source_executed_rows: Tuple[Tuple[float, ...], ...] = ()
    snapshot: Optional[Mapping[str, Any]] = None
    identifiers: Mapping[str, Any] = field(default_factory=dict)
    timing: Mapping[str, Any] = field(default_factory=dict)
    reasons: Tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reason(self) -> Optional[str]:
        """Convenience for consumers that have a singular result-reason field."""

        return self.reasons[0] if self.reasons else None

    def as_dict(self) -> Dict[str, Any]:
        """Return a fully JSON-safe record suitable for Chain streaming."""

        return {
            "status": self.status,
            "raw_proposal_rows": [list(row) for row in self.raw_proposal_rows],
            "source_executed_rows": [list(row) for row in self.source_executed_rows],
            "snapshot": None if self.snapshot is None else _json_copy_mapping(self.snapshot, label="snapshot"),
            "identifiers": _json_copy_mapping(self.identifiers, label="identifiers"),
            "timing": _json_copy_mapping(self.timing, label="timing"),
            "reasons": list(self.reasons),
            "reason": self.reason,
            "provenance": _json_copy_mapping(self.provenance, label="provenance"),
        }


PolicyAdapterFactory = Callable[[], Any]


class ExperimentalPolicyTurnRouter:
    """Route one source-faithful policy turn from injected fresh adapters.

    Factories must return a fresh wrapper object per invocation.  A deployment
    may cache immutable weights *behind* those factories, but returning a
    stateful policy wrapper twice is rejected: it would retain Octo task,
    history, or temporal-ensemble state outside the request snapshot.
    """

    def __init__(self, policy_factories: Mapping[object, PolicyAdapterFactory], *, require_all_policies: bool = True) -> None:
        canonical: Dict[ExperimentalPolicyId, PolicyAdapterFactory] = {}
        for policy_id, factory in policy_factories.items():
            policy = canonical_experimental_policy_id(policy_id)
            if policy in canonical:
                raise ExperimentalPolicyTurnError("Duplicate experimental adapter factory for %s." % policy.value)
            if not callable(factory):
                raise ExperimentalPolicyTurnError("Experimental adapter factory for %s must be callable." % policy.value)
            canonical[policy] = factory
        missing = tuple(policy.value for policy in ExperimentalPolicyId if policy not in canonical)
        if require_all_policies and missing:
            raise ExperimentalPolicyTurnError("Missing experimental adapter factories for: %s." % ", ".join(missing))
        if not canonical:
            raise ExperimentalPolicyTurnError("Experimental policy router requires at least one adapter factory.")
        self._factories = canonical
        # Keep only the immediately prior wrapper. This catches the dangerous
        # cached-wrapper pattern without retaining every heavyweight adapter.
        self._last_worker: Dict[ExperimentalPolicyId, Any] = {}

    @classmethod
    def for_policy(cls, policy_id: object, adapter_factory: PolicyAdapterFactory) -> "ExperimentalPolicyTurnRouter":
        """Build the one-policy variant used inside an isolated Chain worker.

        An OpenVLA, MiniVLA, or Octo worker must not import/build the other
        two model stacks simply to satisfy controller dispatch.  The returned
        router retains every request/snapshot validation rule, but rejects a
        request addressed to another policy before any factory is called.
        """

        policy = canonical_experimental_policy_id(policy_id)
        return cls({policy: adapter_factory}, require_all_policies=False)

    def __call__(self, request: ExperimentalPolicyTurnRequest) -> ExperimentalPolicyTurnResult:
        return self.predict(request)

    def predict(self, request: ExperimentalPolicyTurnRequest) -> ExperimentalPolicyTurnResult:
        """Process one request and return completed/blocked/failed explicitly."""

        try:
            return self._predict(request)
        except ExperimentalPolicyTurnError as error:
            return self._failed(request, str(error))
        except Exception as error:  # Source runtime failures must not grow a synthetic continuation.
            return self._failed(request, "source_policy_call_failed: %s" % str(error))

    turn = predict

    def _failed(self, request: ExperimentalPolicyTurnRequest, reason: str) -> ExperimentalPolicyTurnResult:
        identifiers = dict(request.identifiers) if isinstance(request.identifiers, Mapping) else {}
        snapshot = None
        if isinstance(request.prior_snapshot, Mapping):
            try:
                snapshot = _json_copy_mapping(request.prior_snapshot, label="prior_snapshot")
            except ExperimentalPolicyTurnError:
                # Do not recirculate an invalid state artifact as if it were usable.
                snapshot = None
        return ExperimentalPolicyTurnResult(
            status=ExperimentalPolicyTurnStatus.FAILED.value,
            snapshot=snapshot,
            identifiers=identifiers,
            reasons=(reason,),
            provenance={"schema": EXPERIMENTAL_POLICY_TURN_SCHEMA, "experimental": True, "outcome": "not_scored"},
        )

    def _predict(self, request: ExperimentalPolicyTurnRequest) -> ExperimentalPolicyTurnResult:
        if not isinstance(request, ExperimentalPolicyTurnRequest):
            raise ExperimentalPolicyTurnError("ExperimentalPolicyTurnRouter requires an ExperimentalPolicyTurnRequest.")
        policy = request.canonical_policy_id()
        prompt = _require_text(request.prompt, label="prompt")
        image = self._one_fresh_rgb(request)
        identifiers = _json_copy_mapping(request.identifiers, label="identifiers")
        identifiers["source_frame_sha256"] = _frame_hash(image)
        state_scope = self._state_scope(policy, prompt, identifiers)
        prior = self._parse_prior_snapshot(request.prior_snapshot, state_scope)
        observation_id = identifiers.get("observation_id")
        if prior is not None and not isinstance(observation_id, str):
            raise ExperimentalPolicyTurnError(
                "observation_id is required after the first policy turn; an image hash alone cannot prove a fresh observation."
            )
        if prior is not None and observation_id == prior["last_observation_id"]:
            raise ExperimentalPolicyTurnError("The current observation_id repeats the prior policy boundary; stale RGB reuse is refused.")
        if prior is None and not isinstance(observation_id, str):
            # An initial one-shot call is safe for stateless source policies,
            # but cannot become an Octo trajectory without a durable boundary.
            if policy is ExperimentalPolicyId.OCTO_SMALL:
                raise ExperimentalPolicyTurnError("Octo requires an observation_id at its first source feedback boundary.")
            observation_id = None
        elif isinstance(observation_id, str) and not observation_id.strip():
            raise ExperimentalPolicyTurnError("observation_id cannot be blank when supplied.")

        state = self._state_metadata(identifiers)
        proprio = self._checked_proprio(request.proprio, policy=policy, identifiers=identifiers)
        adapter = self._fresh_adapter(policy)
        capability = self._capability(adapter, policy)
        if capability["status"] != CapabilityStatus.READY_UNQUALIFIED.value:
            return ExperimentalPolicyTurnResult(
                status=ExperimentalPolicyTurnStatus.BLOCKED.value,
                snapshot=None if prior is None else prior,
                identifiers=self._result_identifiers(identifiers, policy, state_scope, observation_id),
                timing={},
                reasons=("policy_capability_blocked: %s" % capability["reason"],),
                provenance={
                    "schema": EXPERIMENTAL_POLICY_TURN_SCHEMA,
                    "experimental": True,
                    "outcome": "not_scored",
                    "policy_id": policy.value,
                    "capability": capability,
                },
            )

        observation = PolicyObservation(
            image_history=(image,),
            prompt=prompt,
            proprio=proprio,
            timestamp=self._timestamp(identifiers),
            proprio_convention=(None if policy is not ExperimentalPolicyId.OCTO_SMALL else str(identifiers["proprio_converter_revision"])),
        )
        if policy is ExperimentalPolicyId.OCTO_SMALL:
            self._restore_octo_source_state(adapter, prior, prompt)
            raw_rows, executed_rows, timing, source_metadata = self._octo_turn(adapter, observation)
            snapshot = self._next_octo_snapshot(
                prior=prior,
                scope=state_scope,
                observation_id=observation_id,
                image=image,
                prompt=prompt,
                state=state,
                proprio=proprio,
                identifiers=identifiers,
                raw_rows=raw_rows,
            )
        elif policy is ExperimentalPolicyId.MINIVLA:
            raw_rows, executed_rows, timing, source_metadata = self._minivla_turn(adapter, observation)
            snapshot = self._stateless_snapshot(scope=state_scope, observation_id=observation_id, prior=prior)
        else:
            raw_rows, executed_rows, timing, source_metadata = self._openvla_turn(adapter, observation)
            snapshot = self._stateless_snapshot(scope=state_scope, observation_id=observation_id, prior=prior)

        # The wall sends *only* this row to the world model.  The raw Octo
        # proposal remains in provenance for review, never as a four-action
        # controller command.
        if len(executed_rows) != 1:
            raise ExperimentalPolicyTurnError("Source policy must expose exactly one executed action for this feedback boundary.")
        return ExperimentalPolicyTurnResult(
            status=ExperimentalPolicyTurnStatus.COMPLETED.value,
            raw_proposal_rows=raw_rows,
            source_executed_rows=executed_rows,
            snapshot=snapshot,
            identifiers=self._result_identifiers(identifiers, policy, state_scope, observation_id),
            timing=timing,
            provenance={
                "schema": EXPERIMENTAL_POLICY_TURN_SCHEMA,
                "experimental": True,
                "outcome": "not_scored",
                "policy_id": policy.value,
                "source_metadata": source_metadata,
                "capability": capability,
                "state": state,
                "policy_seed": request.policy_seed,
                "policy_seed_semantics": (
                    "controller_provenance_only; source sampler uses its pinned behavior"
                    if policy is ExperimentalPolicyId.OCTO_SMALL
                    else "controller_provenance_only; no source sampler override was applied"
                ),
            },
        )

    @staticmethod
    def _one_fresh_rgb(request: ExperimentalPolicyTurnRequest) -> Any:
        if (
            not isinstance(request.image_history, Sequence)
            or isinstance(request.image_history, (str, bytes))
            or len(request.image_history) != 1
            or request.image_history[0] is None
        ):
            raise ExperimentalPolicyTurnError(
                "Experimental policy turns require exactly one current RGB image; pre-padded/stale image histories are forbidden."
            )
        image = request.image_history[0]
        _frame_hash(image)
        return image

    @staticmethod
    def _timestamp(identifiers: Mapping[str, Any]) -> Optional[float]:
        value = identifiers.get("timestamp")
        if value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ExperimentalPolicyTurnError("identifiers.timestamp must be finite when supplied.")
        return float(value)

    @staticmethod
    def _state_metadata(identifiers: Mapping[str, Any]) -> Dict[str, str]:
        kind = identifiers.get("state_kind")
        lineage = identifiers.get("state_lineage_id")
        if kind not in ("source_measured", "forecast_integrated"):
            raise ExperimentalPolicyTurnError(
                "identifiers.state_kind must explicitly be 'source_measured' or 'forecast_integrated'; inferred state is forbidden."
            )
        if not isinstance(lineage, str) or not lineage.strip():
            raise ExperimentalPolicyTurnError("identifiers.state_lineage_id is required; no zero/invented source state is permitted.")
        return {"kind": str(kind), "lineage_id": lineage}

    @staticmethod
    def _checked_proprio(
        value: Optional[Tuple[float, ...]], *, policy: ExperimentalPolicyId, identifiers: Mapping[str, Any]
    ) -> Optional[Tuple[float, ...]]:
        if policy is not ExperimentalPolicyId.OCTO_SMALL:
            if value is None:
                return None
            # The controller may retain an explicit measured/forecast state
            # for the world-model lineage, but neither of these source policy
            # APIs accepts proprioception. Validate that it is not malformed,
            # then deliberately keep it out of the policy observation rather
            # than accidentally changing OpenVLA/MiniVLA preprocessing.
            try:
                retained_state = tuple(float(item) for item in value)
            except (TypeError, ValueError) as error:
                raise ExperimentalPolicyTurnError("Retained non-policy proprio state must be numeric.") from error
            if len(retained_state) not in (7, 8) or not all(math.isfinite(item) for item in retained_state):
                raise ExperimentalPolicyTurnError("Retained non-policy proprio state must be finite 7-D or 8-D state.")
            return None
        if value is None:
            raise ExperimentalPolicyTurnError("Octo-Small requires current source-converted proprioception; do not inject zero/default proprio.")
        try:
            proprio = tuple(float(item) for item in value)
        except (TypeError, ValueError) as error:
            raise ExperimentalPolicyTurnError("Octo source proprioception must be numeric.") from error
        if len(proprio) not in (7, 8) or not all(math.isfinite(item) for item in proprio):
            raise ExperimentalPolicyTurnError("Octo source proprioception must be a finite source-converted 7-D or 8-D vector.")
        if not any(item != 0.0 for item in proprio):
            raise ExperimentalPolicyTurnError("all-zero Octo proprioception is an invented default, not source-converted state.")
        for key in ("proprio_lineage_id", "proprio_converter_revision"):
            if not isinstance(identifiers.get(key), str) or not str(identifiers[key]).strip():
                raise ExperimentalPolicyTurnError("Octo identifiers.%s is required to bind source-converted proprioception." % key)
        return proprio

    @staticmethod
    def _state_scope(policy: ExperimentalPolicyId, prompt: str, identifiers: Mapping[str, Any]) -> Dict[str, Any]:
        cell_id = identifiers.get("cell_id")
        policy_revision = identifiers.get("policy_revision")
        task_id = identifiers.get("task_id")
        if policy is ExperimentalPolicyId.OCTO_SMALL:
            for key, value in (("cell_id", cell_id), ("policy_revision", policy_revision), ("task_id", task_id)):
                if not isinstance(value, str) or not value.strip():
                    raise ExperimentalPolicyTurnError("Octo requires identifiers.%s to scope source task/history state." % key)
        # Stateless policies can run one requested turn without a cell binding;
        # their snapshot remains deliberately unresumable across a different
        # scope.  The wall always supplies all three IDs.
        return {
            "cell_id": cell_id if isinstance(cell_id, str) and cell_id.strip() else None,
            "policy_id": policy.value,
            "policy_revision": policy_revision if isinstance(policy_revision, str) and policy_revision.strip() else None,
            "task_id": task_id if isinstance(task_id, str) and task_id.strip() else None,
            "prompt_sha256": _sha256_text(prompt),
        }

    @staticmethod
    def _parse_prior_snapshot(
        value: Optional[Mapping[str, Any]], scope: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ExperimentalPolicyTurnError("prior_snapshot must be a prior router snapshot object.")
        snapshot = _json_copy_mapping(value, label="prior_snapshot")
        if snapshot.get("schema") != EXPERIMENTAL_POLICY_TURN_SCHEMA:
            raise ExperimentalPolicyTurnError("prior_snapshot is not an experimental policy-turn snapshot.")
        if snapshot.get("scope") != dict(scope):
            raise ExperimentalPolicyTurnError("prior_snapshot belongs to another cell, policy revision, or task; state mixing is refused.")
        if not isinstance(snapshot.get("turn_index"), int) or int(snapshot["turn_index"]) < 0:
            raise ExperimentalPolicyTurnError("prior_snapshot.turn_index must be a nonnegative integer.")
        last_observation_id = snapshot.get("last_observation_id")
        if last_observation_id is not None and (not isinstance(last_observation_id, str) or not last_observation_id):
            raise ExperimentalPolicyTurnError("prior_snapshot.last_observation_id is malformed.")
        return snapshot

    @staticmethod
    def _stateless_snapshot(
        *, scope: Mapping[str, Any], observation_id: Optional[str], prior: Optional[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "schema": EXPERIMENTAL_POLICY_TURN_SCHEMA,
            "scope": dict(scope),
            "turn_index": 0 if prior is None else int(prior["turn_index"]) + 1,
            "last_observation_id": observation_id,
            "source_state": {"kind": "stateless_single_turn"},
        }

    @staticmethod
    def _result_identifiers(
        identifiers: Mapping[str, Any], policy: ExperimentalPolicyId, scope: Mapping[str, Any], observation_id: Optional[str]
    ) -> Dict[str, Any]:
        result = dict(identifiers)
        result.update({"policy_id": policy.value, "state_scope": dict(scope), "observation_id": observation_id})
        return _json_copy_mapping(result, label="result identifiers")

    def _fresh_adapter(self, policy: ExperimentalPolicyId) -> Any:
        factory = self._factories.get(policy)
        if factory is None:
            raise ExperimentalPolicyTurnError(
                "This isolated experimental worker is configured for %s, not %s."
                % (", ".join(item.value for item in self._factories), policy.value)
            )
        adapter = factory()
        if adapter is None:
            raise ExperimentalPolicyTurnError("Experimental %s factory returned no source adapter." % policy.value)
        if self._last_worker.get(policy) is adapter:
            raise ExperimentalPolicyTurnError(
                "Experimental %s factory returned a prior stateful adapter; factories must create a fresh wrapper per turn."
                % policy.value
            )
        self._last_worker[policy] = adapter
        return adapter

    @staticmethod
    def _capability(adapter: Any, policy: ExperimentalPolicyId) -> Dict[str, Any]:
        call = getattr(adapter, "capability", None)
        if not callable(call):
            raise ExperimentalPolicyTurnError("Experimental %s adapter does not expose capability()." % policy.value)
        capability = call()
        status = getattr(capability, "status", None)
        status_value = getattr(status, "value", status)
        reason = getattr(capability, "reason", None)
        if not isinstance(status_value, str) or not isinstance(reason, str):
            raise ExperimentalPolicyTurnError("Experimental %s adapter returned an invalid capability report." % policy.value)
        details = getattr(capability, "details", {})
        evidence_uris = getattr(capability, "evidence_uris", ())
        return _json_copy_mapping(
            {
                "status": status_value,
                "reason": reason,
                "source_verified": bool(getattr(capability, "source_verified", False)),
                "evidence_uris": list(evidence_uris) if not isinstance(evidence_uris, str) else [evidence_uris],
                "details": dict(details) if isinstance(details, Mapping) else {},
            },
            label="capability",
        )

    @staticmethod
    def _openvla_turn(adapter: Any, observation: PolicyObservation) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[Tuple[float, ...], ...], Dict[str, Any], Dict[str, Any]]:
        call = getattr(adapter, "predict_with_report", None)
        if not callable(call):
            raise ExperimentalPolicyTurnError("OpenVLA source adapter must expose predict_with_report().")
        report = call(observation)
        action = _finite_action_row(getattr(report, "action", None), label="OpenVLA source action")
        calls = getattr(report, "backend_calls", None)
        if calls != 1:
            raise ExperimentalPolicyTurnError("OpenVLA must make exactly one source call per fresh observation.")
        timing = {"backend_calls": 1, "wall_seconds": getattr(report, "wall_seconds", None), "gpu_peak_memory_bytes": getattr(report, "gpu_peak_memory_bytes", None)}
        metadata = {
            "source_boundary": "openvla.predict_action(one_fresh_rgb)",
            "unnorm_key": getattr(report, "unnorm_key", None),
            "source_image_timestamp": getattr(report, "source_image_timestamp", None),
        }
        return (action,), (action,), _json_copy_mapping(timing, label="OpenVLA timing"), _json_copy_mapping(metadata, label="OpenVLA metadata")

    @staticmethod
    def _minivla_turn(adapter: Any, observation: PolicyObservation) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[Tuple[float, ...], ...], Dict[str, Any], Dict[str, Any]]:
        call = getattr(adapter, "predict_with_report", None)
        if not callable(call):
            raise ExperimentalPolicyTurnError("MiniVLA source adapter must expose predict_with_report().")
        report = call(observation)
        action = _finite_action_row(getattr(report, "action", None), label="MiniVLA source ret_action[:, 0]")
        if getattr(report, "backend_calls", None) != 1 or getattr(report, "returned_action_count", None) != 1:
            raise ExperimentalPolicyTurnError("MiniVLA source boundary must return exactly one action from one source call.")
        if getattr(report, "vq_future_action_horizon", None) != 7 or getattr(report, "vq_input_horizon", None) != 8:
            raise ExperimentalPolicyTurnError("MiniVLA source report must preserve its VQ H=8 / future-H=7 provenance.")
        timing = {"backend_calls": 1, "wall_seconds": getattr(report, "wall_seconds", None)}
        metadata = {
            "source_boundary": "released_minivla.predict_action(...).ret_action[:, 0]",
            "unnorm_key": getattr(report, "unnorm_key", None),
            "vq_input_horizon": 8,
            "vq_future_action_horizon": 7,
            "returned_action_count": 1,
            "source_image_timestamp": getattr(report, "source_image_timestamp", None),
        }
        return (action,), (action,), _json_copy_mapping(timing, label="MiniVLA timing"), _json_copy_mapping(metadata, label="MiniVLA metadata")

    @staticmethod
    def _octo_turn(adapter: Any, observation: PolicyObservation) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[Tuple[float, ...], ...], Dict[str, Any], Dict[str, Any]]:
        call = getattr(adapter, "predict_with_report", None)
        if not callable(call):
            raise ExperimentalPolicyTurnError("Octo source adapter must expose predict_with_report().")
        report = call(observation)
        proposal = _finite_action_rows(getattr(report, "proposal", None), label="Octo source proposal", expected_rows=4)
        action = _finite_action_row(getattr(report, "action", None), label="Octo source temporal-ensemble action")
        if getattr(report, "backend_calls", None) != 1 or not bool(getattr(report, "temporal_ensembling", False)):
            raise ExperimentalPolicyTurnError("Octo requires one source call and its source temporal-ensemble action.")
        timing = {"backend_calls": 1, "wall_seconds": getattr(report, "wall_seconds", None)}
        metadata = {
            "source_boundary": "octo-v0.1.sample_actions(4x7) + source_temporal_ensemble",
            "observation_count": getattr(report, "observation_count", None),
            "rng_seed": getattr(report, "rng_seed", None),
            "temporal_ensembling": True,
            "action_exp_weight": getattr(report, "action_exp_weight", None),
            "gripper_transformation": getattr(report, "gripper_transformation", None),
            "normalization": getattr(report, "normalization", None),
            "source_image_timestamp": getattr(report, "source_image_timestamp", None),
        }
        return proposal, (action,), _json_copy_mapping(timing, label="Octo timing"), _json_copy_mapping(metadata, label="Octo metadata")

    @staticmethod
    def _restore_octo_source_state(adapter: Any, prior: Optional[Mapping[str, Any]], prompt: str) -> None:
        if prior is None:
            return
        source_state = prior.get("source_state")
        if not isinstance(source_state, Mapping) or source_state.get("schema") != _OCTO_SOURCE_STATE_SCHEMA:
            raise ExperimentalPolicyTurnError("Octo prior_snapshot lacks its source task/history/temporal-ensemble state.")
        if source_state.get("task_instruction_sha256") != _sha256_text(prompt):
            raise ExperimentalPolicyTurnError("Octo prior_snapshot task does not match the current exact instruction.")
        if source_state.get("task_instruction") != prompt:
            raise ExperimentalPolicyTurnError("Octo prior_snapshot does not retain the current exact source instruction.")
        restore = getattr(adapter, "restore_source_state", None)
        if not callable(restore):
            raise ExperimentalPolicyTurnError(
                "Octo source adapter must expose restore_source_state(); replica-global history is forbidden."
            )
        history = source_state.get("observation_history")
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise ExperimentalPolicyTurnError("Octo prior source state lacks its received observation history.")
        decoded_history = []
        for entry in history:
            if not isinstance(entry, Mapping):
                raise ExperimentalPolicyTurnError("Octo prior source history entry is malformed.")
            if entry.get("state_kind") not in ("source_measured", "forecast_integrated"):
                raise ExperimentalPolicyTurnError("Octo prior source history lacks explicit measured/forecast state provenance.")
            for key in ("state_lineage_id", "proprio_lineage_id", "proprio_converter_revision"):
                if not isinstance(entry.get(key), str) or not str(entry[key]).strip():
                    raise ExperimentalPolicyTurnError("Octo prior source history lacks %s provenance." % key)
            decoded_history.append(
                {
                    "decoded_rgb": _decode_rgb_snapshot(entry.get("rgb")),
                    "proprio": entry.get("proprio"),
                }
            )
        restore_payload = dict(source_state)
        restore_payload["observation_history"] = decoded_history
        restore(restore_payload)

    @staticmethod
    def _next_octo_snapshot(
        *,
        prior: Optional[Mapping[str, Any]],
        scope: Mapping[str, Any],
        observation_id: Optional[str],
        image: Any,
        prompt: str,
        state: Mapping[str, str],
        proprio: Optional[Tuple[float, ...]],
        identifiers: Mapping[str, Any],
        raw_rows: Tuple[Tuple[float, ...], ...],
    ) -> Dict[str, Any]:
        if observation_id is None or proprio is None:
            raise ExperimentalPolicyTurnError("Octo cannot create a resumable source state without an observation ID and source-converted proprioception.")
        old_state = None if prior is None else prior.get("source_state")
        old_history = []
        old_proposals = []
        old_count = 0
        if old_state is not None:
            if not isinstance(old_state, Mapping) or old_state.get("schema") != _OCTO_SOURCE_STATE_SCHEMA:
                raise ExperimentalPolicyTurnError("Octo prior source state is malformed.")
            old_history = list(old_state.get("observation_history", []))
            old_proposals = list(old_state.get("proposal_history", []))
            old_count = old_state.get("observation_count", 0)
            if not isinstance(old_count, int) or old_count < 1:
                raise ExperimentalPolicyTurnError("Octo prior source state has no valid observation count.")
        entry = {
            "observation_id": observation_id,
            "rgb": _rgb_snapshot(image, frame_sha256=_frame_hash(image)),
            "proprio": list(proprio),
            "state_kind": state["kind"],
            "state_lineage_id": state["lineage_id"],
            "proprio_lineage_id": identifiers["proprio_lineage_id"],
            "proprio_converter_revision": identifiers["proprio_converter_revision"],
        }
        history = (old_history + [entry])[-_OCTO_HISTORY:]
        proposals = (old_proposals + [[list(row) for row in raw_rows]])[-_OCTO_PROPOSAL_HISTORY:]
        next_count = int(old_count) + 1
        source_state = {
            "schema": _OCTO_SOURCE_STATE_SCHEMA,
            "task_instruction": prompt,
            "task_instruction_sha256": scope["prompt_sha256"],
            "observation_count": next_count,
            "observation_history": history,
            "proposal_history": proposals,
        }
        snapshot = {
            "schema": EXPERIMENTAL_POLICY_TURN_SCHEMA,
            "scope": dict(scope),
            "turn_index": 0 if prior is None else int(prior["turn_index"]) + 1,
            "last_observation_id": observation_id,
            "source_state": source_state,
        }
        # This assertion is intentionally local. A caller can persist exactly
        # this value, but cannot sneak opaque worker state into an artifact.
        _json_copy_mapping(snapshot, label="Octo continuation snapshot")
        return snapshot


def make_experimental_policy_router(
    policy_id: object, adapter_factory: PolicyAdapterFactory
) -> ExperimentalPolicyTurnRouter:
    """Convenience factory for a single isolated policy worker.

    ``adapter_factory`` must still return a fresh source wrapper per turn;
    this helper intentionally does not construct profiles, assets, or model
    permissions on behalf of a worker.
    """

    return ExperimentalPolicyTurnRouter.for_policy(policy_id, adapter_factory)
