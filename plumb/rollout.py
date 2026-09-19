"""Backend-agnostic, checkpointed PLUMB control rollouts.

This module deliberately sits below the fixture-only :mod:`plumb.engine`.
It can connect reviewed policy/world adapters when they are injected by a
deployment, but it does not make those adapters qualified or synthesize action
chunks to fit a backend.  In particular, a one-step native policy is never
repeated to fill a 16-action Cosmos request.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .adapters.bridge import BridgeState
from .adapters.contracts import FeedbackMode, PolicyContract, PolicyObservation, WorldRequest, WorldResult
from .adapters.worlds import (
    GATE_A_CERTIFIED_CLASS,
    PROTOCOL_CERTIFIED_TERMINAL_PADDING,
    PROTOCOL_EXACT_TERMINAL_HORIZON,
    UNCERTIFIED_DEFAULT_CLASS,
    BackendUnavailableError,
    TerminalPaddingCertificate,
)


#: Stage B receives exactly this many frames, endpoints included (spec 5).
#: This is the single authoritative sampling rule in the controller: there is no
#: second, looser rule that silently returns a shorter clip.
JUDGE_FRAME_COUNT = 16
JUDGE_SAMPLE_COUNT = 5
JUDGE_PROTOCOL_REQUEST = "judge_request_v1_16_frame_5_seed"
JUDGE_PROTOCOL_LEGACY_PAYLOAD = "legacy_rollout_payload_diagnostic"

#: Stage A is a separate deterministic component.  When no gate is supplied the
#: controller records this reason code and ``validity="unknown"``.  It never
#: defaults to ``"valid"``: an unrun check is not a passed check.
VALIDITY_GATE_NOT_CONFIGURED = "validity_gate_not_configured"
VALIDITY_GATE_ERROR = "validity_gate_error"
VALIDITY_GATE_CONTRACT_VIOLATION = "validity_gate_contract_violation"


TASK_HORIZONS: Mapping[str, int] = {
    "open_drawer": 70,
    "close_drawer": 70,
    "to_basket": 100,
    "to_sink": 100,
    "fold_cloth": 80,
}
TASK_PROMPTS: Mapping[str, str] = {
    "open_drawer": "Open the drawer",
    "close_drawer": "Close the drawer",
    "to_basket": "Put the eggplant in the yellow basket",
    "to_sink": "Put the eggplant in the blue sink",
    "fold_cloth": "fold the cloth from top right to bottom left",
}


class RolloutConfigurationError(ValueError):
    """The supplied adapters cannot support the requested honest rollout."""


class UnsupportedActionLengthError(RolloutConfigurationError):
    """The native proposal/prefix cannot be sent without forbidden padding."""


class ResumeUnsupportedError(RolloutConfigurationError):
    """A checkpoint lacks restorable policy state or observation history."""


class JudgeWiringError(RolloutConfigurationError):
    """The controller and the judge disagree about the call contract.

    This is deliberately loud.  A signature or schema mismatch between the
    controller and the judge is a build defect, not a missing measurement, and
    silently recording it as ``judge_failure`` is how the previous
    implementation hid a broken judge connection for every episode.
    """


class ClipTooShortError(RolloutConfigurationError):
    """A clip has fewer frames than the frozen judge protocol requires.

    Spec 5: reject malformed/too-short clips rather than silently altering
    sampling.  The controller converts this into an explicit unevaluable
    outcome instead of resampling.
    """


class UncertifiedPaddingError(UnsupportedActionLengthError):
    """Terminal padding was required but no prefix-invariance certificate exists."""


class _UnreachableJudgeError(Exception):
    """Placeholder keeping typed except clauses inert if judge types are absent."""


class ArtifactStore(Protocol):
    def write_json(self, key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def read_json(self, reference: Any) -> Mapping[str, Any]:
        ...

    def list_json(self, prefix: str) -> Sequence[Any]:
        ...


@dataclass(frozen=True)
class ReferenceImageProvenance:
    """A goal/reference image plus the provenance the judge protocol requires.

    The digest must come from the source artifact.  Hashing an in-memory object
    would produce a value that looks like provenance without establishing any,
    so the caller supplies it and this type only checks its shape.
    """

    image: Any
    source_uri: str
    sha256: str

    def __post_init__(self) -> None:
        if self.image is None:
            raise RolloutConfigurationError("reference image cannot be null")
        if not isinstance(self.source_uri, str) or not self.source_uri:
            raise RolloutConfigurationError("reference image requires a source_uri")
        digest = str(self.sha256 or "")
        if digest.startswith("sha256:"):
            digest = digest[len("sha256:") :]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
            raise RolloutConfigurationError("reference image requires a 64-character SHA-256 digest")
        object.__setattr__(self, "sha256", digest.lower())


@dataclass(frozen=True)
class Scenario:
    """A concrete start state; hashes identify input provenance, not measurements."""

    initial_rgb: Any
    state: BridgeState
    task: str
    horizon_actions: int
    hashes: Mapping[str, str]
    start_lineage_id: str = ""
    seed: int = 0
    goal_image: Any = None
    #: Provenance-backed goal/reference images for the Stage B judge request.
    #: Empty means the judge cannot be called for this scenario; it is never
    #: back-filled from ``goal_image`` without a source digest.
    reference_images: Tuple[ReferenceImageProvenance, ...] = ()

    def __post_init__(self) -> None:
        if self.task not in TASK_HORIZONS:
            raise RolloutConfigurationError("unknown benchmark task %r" % self.task)
        if self.horizon_actions != TASK_HORIZONS[self.task]:
            raise RolloutConfigurationError(
                "%s must use its exact %d-action horizon, not %d"
                % (self.task, TASK_HORIZONS[self.task], self.horizon_actions)
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RolloutConfigurationError("scenario seed must be an integer")
        if not isinstance(self.hashes, Mapping) or not self.hashes:
            raise RolloutConfigurationError("scenario hashes are required for input provenance")
        if any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in self.hashes.items()):
            raise RolloutConfigurationError("scenario hashes must be nonempty string mappings")

    @property
    def initialRGB(self) -> Any:  # Compatibility spelling used in the design brief.
        return self.initial_rgb

    @property
    def identity(self) -> str:
        payload = {
            "task": self.task,
            "horizon_actions": self.horizon_actions,
            "hashes": dict(sorted(self.hashes.items())),
            "start_lineage_id": self.start_lineage_id,
            "seed": self.seed,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class PolicyControlPlan:
    """A native proposal plus the exact prefixes certified by its wrapper.

    ``verified_prefix`` is the ordinary replan boundary. At a non-divisible
    task horizon, a shorter final prefix is legal only when it occurs in
    ``verified_terminal_prefixes``. The controller never invents one.
    """

    actions: Tuple[Tuple[float, ...], ...]
    verified_prefix: int
    history_length: int
    verified_terminal_prefixes: Tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actions:
            raise RolloutConfigurationError("native policy returned an empty control plan")
        for action in self.actions:
            if len(action) != 7:
                raise RolloutConfigurationError("native policy actions must be physical 7-D Bridge actions")
            if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in action):
                raise RolloutConfigurationError("native policy actions must be finite numeric values")
        if not 1 <= self.verified_prefix <= len(self.actions):
            raise RolloutConfigurationError("verified_prefix must select a nonempty prefix of the native proposal")
        if self.history_length < 1:
            raise RolloutConfigurationError("history_length must be positive")
        for prefix in self.verified_terminal_prefixes:
            if not 1 <= prefix <= len(self.actions):
                raise RolloutConfigurationError("every verified terminal prefix must be inside the native proposal")

    def select_prefix(self, remaining_actions: int) -> int:
        if remaining_actions >= self.verified_prefix:
            return self.verified_prefix
        if remaining_actions in self.verified_terminal_prefixes:
            return remaining_actions
        raise UnsupportedActionLengthError(
            "remaining horizon is %d but the native policy only certified prefix %d and terminal prefixes %s; "
            "the controller will not truncate or pad it"
            % (remaining_actions, self.verified_prefix, self.verified_terminal_prefixes)
        )


@dataclass(frozen=True)
class WorldActionProfile:
    """The explicitly normal (not probe) action lengths a backend can execute.

    ``supported_action_lengths`` is evidence.  ``certification_class`` says
    whether it came from a Gate-A certification artifact or is the single
    uncertified default, and ``certification_source_hash`` names the artifact.
    Qualification mode refuses an uncertified profile outright.
    """

    profile_id: str
    domain: str
    supported_action_lengths: Tuple[int, ...]
    control_hz: float = 5.0
    certification_class: str = UNCERTIFIED_DEFAULT_CLASS
    certification_source_hash: Optional[str] = None
    certification_source_uri: Optional[str] = None
    certification_id: Optional[str] = None
    terminal_padding_certificate: Optional[TerminalPaddingCertificate] = None

    def __post_init__(self) -> None:
        if not self.profile_id or not self.domain:
            raise RolloutConfigurationError("world action profile needs profile_id and domain")
        if not self.supported_action_lengths or any(
            isinstance(length, bool) or not isinstance(length, int) or length < 1
            for length in self.supported_action_lengths
        ):
            raise RolloutConfigurationError("world action profile needs explicit positive normal action lengths")
        if self.control_hz <= 0:
            raise RolloutConfigurationError("control_hz must be positive")
        if self.certification_class not in (UNCERTIFIED_DEFAULT_CLASS, GATE_A_CERTIFIED_CLASS):
            raise RolloutConfigurationError(
                "certification_class must be %r or %r" % (UNCERTIFIED_DEFAULT_CLASS, GATE_A_CERTIFIED_CLASS)
            )
        if self.certification_class == GATE_A_CERTIFIED_CLASS and not self.certification_source_hash:
            raise RolloutConfigurationError(
                "a gate_a_certified action-length profile must name its certification artifact hash"
            )
        certificate = self.terminal_padding_certificate
        if certificate is not None and certificate.profile_id != self.profile_id:
            raise RolloutConfigurationError(
                "terminal padding certificate %r belongs to profile %r, not %r"
                % (certificate.certificate_id, certificate.profile_id, self.profile_id)
            )

    def reachable_totals(self, horizon: int) -> Tuple[bool, ...]:
        """Subset-sum reachability of every total up to ``horizon``."""

        reachable = [False] * (max(int(horizon), 0) + 1)
        reachable[0] = True
        for total in range(1, len(reachable)):
            reachable[total] = any(
                total >= length and reachable[total - length] for length in self.supported_action_lengths
            )
        return tuple(reachable)

    def represents_exactly(self, horizon: int) -> bool:
        return self.reachable_totals(horizon)[horizon]

    def unblocking_lengths(self, horizon: int, *, limit: int = 12) -> Tuple[int, ...]:
        """Action lengths that, if certified, would make ``horizon`` exact.

        This is the actionable part of the error message: it names the lengths
        worth probing instead of leaving a task permanently blocked.
        """

        existing = set(self.supported_action_lengths)
        found: List[int] = []
        for candidate in range(1, int(horizon) + 1):
            if candidate in existing:
                continue
            extended = WorldActionProfile(
                profile_id=self.profile_id,
                domain=self.domain,
                supported_action_lengths=tuple(sorted(existing | {candidate})),
                control_hz=self.control_hz,
            )
            if extended.represents_exactly(horizon):
                found.append(candidate)
                if len(found) >= limit:
                    break
        return tuple(found)

    def padding_prefixes(self, horizon: int) -> Tuple[int, ...]:
        """Certified terminal prefixes that close an otherwise unreachable horizon."""

        certificate = self.terminal_padding_certificate
        if certificate is None or certificate.errors():
            return ()
        if certificate.padded_action_length not in self.supported_action_lengths:
            return ()
        reachable = self.reachable_totals(horizon)
        return tuple(
            prefix
            for prefix in sorted(int(value) for value in certificate.certified_prefix_lengths)
            if 0 < prefix <= horizon and reachable[horizon - prefix]
        )

    def certification_record(self) -> Dict[str, Any]:
        certificate = self.terminal_padding_certificate
        return {
            "profile_id": self.profile_id,
            "supported_action_lengths": list(self.supported_action_lengths),
            "certification_class": self.certification_class,
            "certification_id": self.certification_id,
            "certification_source_hash": self.certification_source_hash,
            "certification_source_uri": self.certification_source_uri,
            "terminal_padding_certificate": None if certificate is None else certificate.as_dict(),
        }


@dataclass(frozen=True)
class QualificationEvidence:
    """Small injected evidence boundary for a controller invocation.

    Gate records remain owned by :mod:`plumb.gates`; this value makes a caller
    explicitly supply the result of that review instead of inferring it from a
    successful adapter call.
    """

    passed_gates: Tuple[str, ...]
    protocol_hash: Optional[str] = None
    evidence_uris: Tuple[str, ...] = ()

    def errors(self) -> Tuple[str, ...]:
        missing = tuple(gate for gate in ("A", "B", "C", "D") if gate not in set(self.passed_gates))
        errors: List[str] = []
        if missing:
            errors.append("qualification requires passed gates %s" % ", ".join(missing))
        if not self.protocol_hash:
            errors.append("qualification requires a frozen protocol hash")
        if not self.evidence_uris:
            errors.append("qualification requires evidence URIs")
        return tuple(errors)


@dataclass(frozen=True)
class RolloutJudgeInput:
    """Actions and policy identity are intentionally absent from judge input."""

    task: str
    frames: Tuple[Any, ...]
    nominal_timestamps: Tuple[float, ...]
    goal_image: Any
    frame_hashes: Tuple[str, ...]


@dataclass(frozen=True)
class SegmentRecord:
    index: int
    action_offset: int
    action_count: int
    policy_history_hashes: Tuple[str, ...]
    native_actions: Tuple[Tuple[float, ...], ...]
    compiled_actions: Tuple[Tuple[float, ...], ...]
    world_request_id: str
    world_timing: Mapping[str, Any]
    generated_frame_hashes: Tuple[str, ...]
    checkpoint_ref: Optional[Mapping[str, Any]]
    status: str = "completed"
    error: Optional[Mapping[str, str]] = None
    #: Present only when a certified terminal padding request was sent. It names
    #: the certificate, the padded request length, the executed prefix, and the
    #: discarded post-horizon frames.
    padding: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class RolloutReport:
    rollout_id: str
    scenario_id: str
    status: str
    mode: str
    qualified: bool
    feedback_mode: str
    physical_state_measured: bool
    horizon_actions: int
    executed_actions: int
    segments: Tuple[SegmentRecord, ...]
    native_actions: Tuple[Tuple[float, ...], ...]
    compiled_actions: Tuple[Tuple[float, ...], ...]
    frame_hashes: Tuple[str, ...]
    nominal_timestamps: Tuple[float, ...]
    validity: str
    binary_success: Optional[bool]
    progress_score: Optional[int]
    missing_reason: Optional[str]
    checkpoint_refs: Tuple[Mapping[str, Any], ...]
    world_calls: int
    timings: Mapping[str, Any]
    error: Optional[Mapping[str, str]] = None
    #: ``exact_terminal_horizon`` or ``certified_terminal_padding``. A padded run
    #: never shares the identity of an exact horizon match.
    protocol_identity: str = PROTOCOL_EXACT_TERMINAL_HORIZON
    padding_events: Tuple[Mapping[str, Any], ...] = ()
    world_action_length_certification: Mapping[str, Any] = field(default_factory=dict)
    #: Stage A results. ``validity_gate_configured=False`` with
    #: ``validity="unknown"`` is the loud record of an absent gate.
    validity_gate_configured: bool = False
    validity_reason_codes: Tuple[str, ...] = ()
    validity_report: Optional[Mapping[str, Any]] = None
    #: Stage B wiring evidence.
    judge_protocol: Optional[str] = None
    judge_seeds: Tuple[int, ...] = ()
    judge_frame_indices: Tuple[int, ...] = ()
    judge_frame_hashes: Tuple[str, ...] = ()
    judge_rubric_hash: Optional[str] = None
    judge_task_registry_hash: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


class FileArtifactStore:
    """Minimal durable JSON checkpoint store for local/offline controller use."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if self.root.resolve() not in candidate.parents:
            raise RolloutConfigurationError("artifact key escapes the configured store")
        return candidate

    def write_json(self, key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _canonical_json(_jsonable(dict(payload))).encode("utf-8") + b"\n"
        temporary = path.with_name(".%s.tmp-%s" % (path.name, uuid.uuid4().hex))
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            # Segment keys are immutable. A retry returns the pre-existing
            # artifact rather than replacing a durable checkpoint.
            pass
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "uri": "artifact://%s" % key,
            "key": key,
            "sha256": "sha256:" + digest,
            "media_type": "application/json",
        }

    def read_json(self, reference: Any) -> Mapping[str, Any]:
        key = reference.get("key") if isinstance(reference, Mapping) else str(reference)
        with self._path(str(key)).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ResumeUnsupportedError("checkpoint JSON must be an object")
        return payload

    def list_json(self, prefix: str) -> Sequence[Mapping[str, Any]]:
        directory = self._path(prefix)
        if not directory.exists():
            return ()
        return tuple({"key": path.relative_to(self.root).as_posix()} for path in sorted(directory.glob("*.json")))


class RolloutController:
    """Execute real injected adapter calls with strict control-feedback rules."""

    def __init__(
        self,
        *,
        mode: str = "diagnostic",
        qualification_evidence: Optional[QualificationEvidence] = None,
        feedback_mode: FeedbackMode = FeedbackMode.FORECAST_STATE,
    ) -> None:
        if mode not in ("diagnostic", "qualification"):
            raise RolloutConfigurationError("mode must be diagnostic or qualification")
        self.mode = mode
        self.qualification_evidence = qualification_evidence
        self.feedback_mode = feedback_mode

    def execute(
        self,
        scenario: Scenario,
        policy: Any,
        world: Any,
        compiler: Any,
        judge: Any,
        artifact_store: Any,
        *,
        validity_gate: Any = None,
        rollout_id: Optional[str] = None,
        resume: bool = False,
    ) -> RolloutReport:
        """Run the exact scenario horizon or return an explicit partial record.

        Unsupported action lengths are raised before a world call. Runtime
        world errors are accounted as nullable, non-evaluable reports because
        an unavailable output is neither a success nor a known failure.

        ``validity_gate`` is the Stage A deterministic gate. Omitting it does
        not make episodes valid: the report records ``validity="unknown"``,
        ``validity_gate_configured=False`` and an explicit reason code.
        """

        store = self._coerce_store(artifact_store)
        profile = self._world_profile(world)
        self._check_qualification(profile)
        padding_prefixes = self._check_horizon_representable(scenario.horizon_actions, profile)
        rollout_id = rollout_id or ("rollout-" + uuid.uuid4().hex)
        prefix = "rollouts/%s" % rollout_id

        if resume:
            (
                action_offset,
                state,
                history,
                reset_state,
                checkpoint_refs,
                prior_native_actions,
                prior_compiled_actions,
                prior_frame_hashes,
            ) = self._restore_checkpoint(
                store, prefix, policy
            )
        else:
            action_offset = 0
            state = scenario.state
            history = self._initial_history(scenario.initial_rgb, self._policy_contract(policy))
            reset_state = self._reset_policy(policy, scenario.seed)
            checkpoint_refs = []
            prior_native_actions = []
            prior_compiled_actions = []
            prior_frame_hashes = []

        judge_frames: List[Any] = [scenario.initial_rgb]
        # The source image must appear once in the rollout timeline; bootstrap
        # copies only satisfy a policy's declared initial history contract.
        frame_hashes: List[str] = [_frame_hash(scenario.initial_rgb)] + list(prior_frame_hashes)
        timestamps: List[float] = [index / profile.control_hz for index in range(action_offset + 1)]
        all_native_actions: List[Tuple[float, ...]] = list(prior_native_actions)
        all_compiled_actions: List[Tuple[float, ...]] = list(prior_compiled_actions)
        forecast_state_rows: List[Tuple[float, ...]] = [tuple(scenario.state.values8())]
        segments: List[SegmentRecord] = []
        padding_events: List[Mapping[str, Any]] = []
        world_calls = 0
        started = time.perf_counter()
        effective_feedback_mode: Optional[FeedbackMode] = None

        while action_offset < scenario.horizon_actions:
            observation = self._observation(scenario, policy, history, state, action_offset, profile.control_hz)
            plan = self._plan(policy, observation, scenario.horizon_actions - action_offset)
            self._validate_plan_history(plan, policy, history)
            count = plan.select_prefix(scenario.horizon_actions - action_offset)
            # ``count`` is the executed control prefix. ``request_length`` is what
            # the backend is asked for, which differs only under a certified
            # terminal padding protocol.
            request_length = count
            padding_certificate: Optional[TerminalPaddingCertificate] = None
            if count not in profile.supported_action_lengths:
                padding_certificate = self._padding_certificate_for(profile, count, padding_prefixes)
                if padding_certificate is None:
                    raise self._unsupported_length_error(policy, profile, count, padding_prefixes)
                request_length = int(padding_certificate.padded_action_length)

            native_actions = tuple(tuple(float(value) for value in row) for row in plan.actions[:count])
            if padding_certificate is not None:
                filler = padding_certificate.padding_actions(native_actions, request_length - count)
                request_actions = native_actions + tuple(
                    tuple(float(value) for value in row) for row in filler
                )
            else:
                filler = ()
                request_actions = native_actions
            world_called = False
            try:
                compiled = compiler.compile(state, request_actions)
                backend_actions = tuple(tuple(float(value) for value in row) for row in compiled.backend_actions)
                forecast_states = tuple(compiled.forecast_states)
                if len(backend_actions) != request_length or len(forecast_states) != request_length + 1:
                    raise RolloutConfigurationError("compiler must preserve the selected prefix and return N+1 forecast states")
                compiled_feedback = getattr(compiled, "feedback_mode", self.feedback_mode)
                if not isinstance(compiled_feedback, FeedbackMode):
                    compiled_feedback = FeedbackMode(str(compiled_feedback))
                if effective_feedback_mode is None:
                    effective_feedback_mode = compiled_feedback
                elif effective_feedback_mode is not compiled_feedback:
                    raise RolloutConfigurationError("compiler changed feedback_mode within one logical rollout")
                if self.mode == "qualification" and compiled_feedback is not FeedbackMode.NATIVE_FEEDBACK:
                    raise RolloutConfigurationError(
                        "qualification mode refuses compiler feedback_mode=%s; forecast state is not physical state"
                        % compiled_feedback.value
                    )
                request = WorldRequest(
                    conditioning_image=history[-1],
                    prompt=TASK_PROMPTS[scenario.task],
                    domain=profile.domain,
                    compiled_actions=backend_actions,
                    nominal_control_timestamps=tuple(
                        (action_offset + index + 1) / profile.control_hz for index in range(request_length)
                    ),
                    seed=_segment_seed(scenario.seed, action_offset),
                    compatibility_profile_id=profile.profile_id,
                    feedback_mode=compiled_feedback,
                    source_state_lineage_id=scenario.start_lineage_id or None,
                    request_id="%s:segment-%04d" % (rollout_id, len(segments)),
                )
                before = time.perf_counter()
                world_called = True
                result = world.generate(request)
                call_elapsed = time.perf_counter() - before
                world_calls += 1
                if not isinstance(result, WorldResult):
                    raise RolloutConfigurationError("world adapter must return WorldResult")
                result.validate(request_length)
                generated_all = result.future_frames
                if len(generated_all) != request_length:
                    raise RolloutConfigurationError("world result did not provide exactly one future frame per action")
                # Post-horizon frames from a padded request are recorded as
                # discarded evidence, never scored and never stitched in.
                generated = generated_all[:count]
                discarded_frames = generated_all[count:]
            except UnsupportedActionLengthError:
                raise
            except RolloutConfigurationError:
                # Controller/qualification contract failures are detected
                # before a valid world outcome exists; do not disguise them as
                # a backend service failure.
                raise
            except Exception as exc:
                world_calls += 1 if world_called else 0
                error = {"type": type(exc).__name__, "message": str(exc)}
                failed = SegmentRecord(
                    index=len(segments),
                    action_offset=action_offset,
                    action_count=count,
                    policy_history_hashes=tuple(_frame_hash(frame) for frame in observation.image_history),
                    native_actions=native_actions,
                    compiled_actions=(),
                    world_request_id="%s:segment-%04d" % (rollout_id, len(segments)),
                    world_timing={"wall_seconds": time.perf_counter() - started, "backend_call_confirmed": False},
                    generated_frame_hashes=(),
                    checkpoint_ref=None,
                    status="failed",
                    error=error,
                )
                segments.append(failed)
                report = self._report(
                    rollout_id, scenario, "world_unavailable" if isinstance(exc, BackendUnavailableError) else "failed",
                    action_offset, segments, all_native_actions, all_compiled_actions, frame_hashes, timestamps,
                    checkpoint_refs, world_calls, started, "unknown", None, None, "world_failure", error,
                    feedback_mode=effective_feedback_mode,
                    profile=profile,
                    padding_events=padding_events,
                    validity_gate_configured=validity_gate is not None,
                )
                self._write_terminal(store, prefix, report)
                return report

            # ``future_frames`` removes an echoed conditioning frame exactly
            # once. The next native policy call gets a freshly generated image.
            old_offset = action_offset
            history.append(generated[-1])
            required = self._policy_contract(policy).required_observation_history if self._policy_contract(policy) else plan.history_length
            history = history[-max(1, required):]
            # Feedback continues from the executed prefix, not from the padded
            # request's final forecast state.
            state = forecast_states[count]
            action_offset += count
            all_native_actions.extend(native_actions)
            executed_backend_actions = backend_actions[:count]
            all_compiled_actions.extend(executed_backend_actions)
            forecast_state_rows.extend(tuple(row.values8()) for row in forecast_states[1 : count + 1])
            generated_hashes = tuple(_frame_hash(frame) for frame in generated)
            frame_hashes.extend(generated_hashes)
            judge_frames.extend(generated)
            timestamps.extend((old_offset + index + 1) / profile.control_hz for index in range(count))
            padding_record: Optional[Mapping[str, Any]] = None
            if padding_certificate is not None:
                padding_record = {
                    "protocol_identity": padding_certificate.protocol_identity,
                    "certificate_id": padding_certificate.certificate_id,
                    "certificate_hash": padding_certificate.certificate_hash,
                    "padding_action_policy": padding_certificate.padding_action_policy,
                    "requested_action_length": request_length,
                    "executed_prefix": count,
                    "padding_action_count": request_length - count,
                    "padding_actions": [list(row) for row in filler],
                    "discarded_post_horizon_frame_hashes": [_frame_hash(frame) for frame in discarded_frames],
                    "segment_index": len(segments),
                    "action_offset": old_offset,
                }
                padding_events.append(padding_record)

            checkpoint_payload, resumable = self._checkpoint_payload(
                scenario, action_offset, state, history, policy, reset_state, len(segments), request, native_actions,
                executed_backend_actions, generated_hashes, result, padding=padding_record,
            )
            checkpoint_ref = self._write_json(store, "%s/checkpoints/segment-%04d.json" % (prefix, len(segments)), checkpoint_payload)
            checkpoint_refs.append(checkpoint_ref)
            segments.append(
                SegmentRecord(
                    index=len(segments),
                    action_offset=action_offset - count,
                    action_count=count,
                    policy_history_hashes=tuple(_frame_hash(frame) for frame in observation.image_history),
                    native_actions=native_actions,
                    compiled_actions=executed_backend_actions,
                    world_request_id=request.request_id or "",
                    world_timing={**_timing_dict(result.timing), "controller_wall_seconds": call_elapsed, "resumable": resumable},
                    generated_frame_hashes=generated_hashes,
                    checkpoint_ref=checkpoint_ref,
                    padding=padding_record,
                )
            )

        # Stage A always runs, or is always explicitly recorded as not run.
        if resume:
            # Segment checkpoints retain the current history, action records and
            # frame hashes. They do not reconstruct every raw prior frame, so
            # Stage A cannot see the whole clip either. Reporting "unknown" is
            # correct here; running the deterministic checks on a truncated
            # frame stream would manufacture a frame-count failure.
            validity_result = _ControllerValidity(
                "unknown",
                ("validity_requires_full_frame_artifacts_after_resume",),
                {
                    "validity": "unknown",
                    "reason_codes": ["validity_requires_full_frame_artifacts_after_resume"],
                    "gate_configured": validity_gate is not None,
                    "note": (
                        "A resumed rollout has only the frames generated after the resume boundary. Stage A needs the "
                        "whole clip, so it is recorded as not run rather than evaluated on a partial stream."
                    ),
                },
            )
        else:
            validity_result = self._validity(
                validity_gate,
                scenario=scenario,
                frames=tuple(judge_frames),
                actions=tuple(all_native_actions),
                timestamps=tuple(timestamps),
                states=tuple(forecast_state_rows),
                rollout_id=rollout_id,
            )
        judge_evidence: Dict[str, Any] = {}
        if resume:
            success, progress, missing_reason = (None, None, "judge_requires_full_frame_artifacts_after_resume")
        elif validity_result.validity != "valid":
            # Stage A invalid/unknown episodes stay unevaluable for primary
            # rates (spec 5). They are not deleted and not counted as failures.
            success, progress, missing_reason = (None, None, "validity_%s" % validity_result.validity)
        else:
            success, progress, missing_reason, judge_evidence = self._judge(
                scenario, judge, tuple(judge_frames), frame_hashes, timestamps, rollout_id=rollout_id
            )
        report = self._report(
            rollout_id, scenario, "completed", action_offset, segments, all_native_actions, all_compiled_actions,
            frame_hashes, timestamps, checkpoint_refs, world_calls, started, validity_result.validity, success, progress,
            missing_reason, None,
            feedback_mode=effective_feedback_mode,
            profile=profile,
            padding_events=padding_events,
            validity_gate_configured=validity_gate is not None,
            validity_reason_codes=validity_result.reason_codes,
            validity_report=validity_result.report,
            judge_evidence=judge_evidence,
        )
        self._write_terminal(store, prefix, report)
        return report

    def _coerce_store(self, artifact_store: Any) -> Any:
        if isinstance(artifact_store, (str, Path)):
            return FileArtifactStore(Path(artifact_store))
        if not callable(getattr(artifact_store, "write_json", None)):
            raise RolloutConfigurationError("artifact_store must provide write_json for durable checkpoints")
        return artifact_store

    @staticmethod
    def _policy_contract(policy: Any) -> Optional[PolicyContract]:
        contract = getattr(policy, "contract", None)
        return contract if isinstance(contract, PolicyContract) else None

    def _world_profile(self, world: Any) -> WorldActionProfile:
        candidate = getattr(world, "rollout_profile", None)
        if isinstance(candidate, WorldActionProfile):
            return candidate
        if isinstance(candidate, Mapping):
            return WorldActionProfile(
                profile_id=str(candidate["profile_id"]),
                domain=str(candidate.get("domain", "bridge_orig_lerobot")),
                supported_action_lengths=tuple(int(value) for value in candidate["supported_action_lengths"]),
                control_hz=float(candidate.get("control_hz", 5.0)),
            )
        native_profile = getattr(world, "profile", None)
        lengths = getattr(native_profile, "allowed_action_lengths", None)
        if lengths is None:
            lengths = getattr(world, "supported_action_lengths", None)
        profile_id = getattr(native_profile, "profile_id", None) or getattr(world, "profile_id", None)
        domain = getattr(world, "domain", None) or "bridge_orig_lerobot"
        control_hz = getattr(native_profile, "fps", None) or getattr(world, "control_hz", 5.0)
        if not profile_id or lengths is None:
            raise RolloutConfigurationError(
                "world adapter must expose an explicit normal supported_action_lengths profile; probe lengths are never inferred"
            )
        certification = getattr(native_profile, "action_length_certification", None)
        padding = getattr(native_profile, "terminal_padding_certificate", None)
        if not isinstance(padding, TerminalPaddingCertificate):
            padding = None
        return WorldActionProfile(
            profile_id=str(profile_id),
            domain=str(domain),
            supported_action_lengths=tuple(int(value) for value in lengths),
            control_hz=float(control_hz),
            certification_class=str(
                getattr(native_profile, "action_length_certification_class", UNCERTIFIED_DEFAULT_CLASS)
            ),
            certification_source_hash=getattr(native_profile, "action_length_certification_source_hash", None),
            certification_source_uri=getattr(certification, "source_uri", None),
            certification_id=getattr(certification, "certification_id", None),
            terminal_padding_certificate=padding,
        )

    def _check_qualification(self, profile: WorldActionProfile) -> None:
        if self.mode != "qualification":
            return
        if self.feedback_mode is not FeedbackMode.NATIVE_FEEDBACK:
            raise RolloutConfigurationError(
                "qualification mode refuses forecast/frozen state feedback; forecast state is not measured physical state"
            )
        if self.qualification_evidence is None:
            raise RolloutConfigurationError("qualification mode requires explicit passed gate evidence")
        errors = self.qualification_evidence.errors()
        if errors:
            raise RolloutConfigurationError("qualification blocked: " + "; ".join(errors))
        if not profile.supported_action_lengths:
            raise RolloutConfigurationError("qualification requires an explicitly supported action-length profile")
        if profile.certification_class != GATE_A_CERTIFIED_CLASS:
            raise RolloutConfigurationError(
                "qualification requires Gate-A certified action lengths for %s; the profile is %r, so its "
                "supported lengths %s are an uncertified default rather than probed evidence"
                % (profile.profile_id, profile.certification_class, profile.supported_action_lengths)
            )

    @staticmethod
    def _check_horizon_representable(horizon: int, profile: WorldActionProfile) -> Tuple[int, ...]:
        """Return the certified terminal prefixes needed, or ``()`` for an exact fit.

        Raises before any world call when the horizon is unrepresentable and no
        prefix-invariance certificate authorises padding, naming the exact
        action lengths that would unblock it.
        """

        if profile.represents_exactly(horizon):
            return ()
        prefixes = profile.padding_prefixes(horizon)
        if prefixes:
            return prefixes
        unblocking = profile.unblocking_lengths(horizon)
        certificate = profile.terminal_padding_certificate
        detail = ""
        if certificate is not None:
            reasons = certificate.errors()
            if reasons:
                detail = " Its terminal padding certificate %r cannot authorise padding: %s." % (
                    certificate.certificate_id,
                    "; ".join(reasons),
                )
            elif certificate.padded_action_length not in profile.supported_action_lengths:
                detail = (
                    " Its terminal padding certificate pads to %d actions, which is not a supported request length %s."
                    % (certificate.padded_action_length, profile.supported_action_lengths)
                )
            else:
                detail = (
                    " Its certified terminal prefixes %s do not leave a reachable remainder for horizon %d."
                    % (tuple(certificate.certified_prefix_lengths), horizon)
                )
        raise UnsupportedActionLengthError(
            "%s cannot execute the exact %d-action terminal horizon from certified action lengths %s "
            "(certification_class=%s). Certifying any one of these action lengths would make it exact: %s. "
            "Otherwise supply a TerminalPaddingCertificate with measured prefix invariance for a terminal prefix "
            "congruent to %d modulo the supported lengths. No padding or action repetition is applied without one.%s"
            % (
                profile.profile_id,
                horizon,
                profile.supported_action_lengths,
                profile.certification_class,
                unblocking or "none at or below the horizon",
                horizon,
                detail,
            )
        )

    @staticmethod
    def _padding_certificate_for(
        profile: WorldActionProfile, count: int, padding_prefixes: Sequence[int]
    ) -> Optional[TerminalPaddingCertificate]:
        certificate = profile.terminal_padding_certificate
        if certificate is None or count not in tuple(int(value) for value in padding_prefixes):
            return None
        if certificate.padded_action_length not in profile.supported_action_lengths:
            return None
        if not certificate.certifies(
            padded_length=certificate.padded_action_length, consumed_prefix=count, profile_id=profile.profile_id
        ):
            return None
        return certificate

    def _unsupported_length_error(
        self, policy: Any, profile: WorldActionProfile, count: int, padding_prefixes: Sequence[int]
    ) -> UnsupportedActionLengthError:
        contract = self._policy_contract(policy)
        policy_name = contract.name if contract else type(policy).__name__
        if count == 1 and 16 in profile.supported_action_lengths:
            return UnsupportedActionLengthError(
                "one-step native proposal from %s is unsupported by %s normal lengths %s; "
                "it will not be repeated or padded to 16"
                % (policy_name, profile.profile_id, profile.supported_action_lengths)
            )
        certificate = profile.terminal_padding_certificate
        if certificate is not None:
            return UncertifiedPaddingError(
                "native verified prefix %d is unsupported by %s certified action lengths %s, and terminal padding "
                "certificate %r does not cover a consumed prefix of %d (it certifies %s at padded length %d). "
                "No world call was made."
                % (
                    count,
                    profile.profile_id,
                    profile.supported_action_lengths,
                    certificate.certificate_id,
                    count,
                    tuple(certificate.certified_prefix_lengths),
                    certificate.padded_action_length,
                )
            )
        return UncertifiedPaddingError(
            "native verified prefix %d is unsupported by %s certified action lengths %s and no "
            "TerminalPaddingCertificate was supplied, so prefix invariance to post-horizon actions is unproven. "
            "Certifying action length %d, or supplying a padding certificate for a consumed prefix of %d, would "
            "unblock it. No world call was made."
            % (count, profile.profile_id, profile.supported_action_lengths, count, count)
        )

    @staticmethod
    def _initial_history(initial_rgb: Any, contract: Optional[PolicyContract]) -> List[Any]:
        required = contract.required_observation_history if contract else 1
        # This is a declared reset bootstrap, never an attempt to manufacture a
        # future feedback image or extend a control proposal.
        return [initial_rgb for _ in range(required)]

    @staticmethod
    def _call_with_supported_arity(function: Any, *values: Any) -> Any:
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return function(*values)
        params = [parameter for parameter in signature.parameters.values() if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD
        )]
        if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
            return function(*values)
        return function(*values[:len(params)])

    def _reset_policy(self, policy: Any, seed: int) -> Any:
        reset = getattr(policy, "reset", None)
        return self._call_with_supported_arity(reset, seed) if callable(reset) else None

    def _observation(
        self, scenario: Scenario, policy: Any, history: Sequence[Any], state: BridgeState, action_offset: int, control_hz: float
    ) -> PolicyObservation:
        contract = self._policy_contract(policy)
        proprio = state.values8()
        return PolicyObservation(
            image_history=tuple(history),
            prompt=TASK_PROMPTS[scenario.task],
            proprio=proprio if contract and contract.requires_proprio else None,
            goal_image=scenario.goal_image,
            timestamp=action_offset / control_hz,
        )

    def _plan(self, policy: Any, observation: PolicyObservation, remaining: int) -> PolicyControlPlan:
        method = getattr(policy, "plan_control", None) or getattr(policy, "plan", None)
        if callable(method):
            plan = self._call_with_supported_arity(method, observation, remaining)
            if not isinstance(plan, PolicyControlPlan):
                raise RolloutConfigurationError("policy plan method must return PolicyControlPlan with verified prefix metadata")
            return plan
        contract = self._policy_contract(policy)
        predict = getattr(policy, "predict_action", None) or getattr(policy, "predict", None)
        if callable(predict) and contract is not None and contract.native_proposal_horizon == 1 and contract.certified_execute_prefix == 1:
            action = self._call_with_supported_arity(predict, observation)
            return PolicyControlPlan(
                actions=(tuple(float(value) for value in action),),
                verified_prefix=1,
                history_length=contract.required_observation_history,
                metadata={"derived_from_native_one_step": True},
            )
        raise RolloutConfigurationError(
            "policy must expose plan_control()/plan() returning a verified PolicyControlPlan; "
            "only explicitly certified native one-step adapters may use predict_action()"
        )

    def _validate_plan_history(self, plan: PolicyControlPlan, policy: Any, history: Sequence[Any]) -> None:
        contract = self._policy_contract(policy)
        expected = contract.required_observation_history if contract else plan.history_length
        if plan.history_length != expected or len(history) != expected:
            raise RolloutConfigurationError(
                "policy history mismatch: plan declares %d, adapter requires %d, controller has %d"
                % (plan.history_length, expected, len(history))
            )

    def _checkpoint_payload(
        self,
        scenario: Scenario,
        action_offset: int,
        state: BridgeState,
        history: Sequence[Any],
        policy: Any,
        reset_state: Any,
        segment_index: int,
        request: WorldRequest,
        native_actions: Sequence[Sequence[float]],
        backend_actions: Sequence[Sequence[float]],
        generated_hashes: Sequence[str],
        result: WorldResult,
        padding: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        snapshot = getattr(policy, "snapshot_state", None)
        restore = getattr(policy, "restore_state", None)
        policy_state: Any = None
        state_ok = callable(snapshot) and callable(restore)
        if state_ok:
            try:
                policy_state = snapshot()
                _canonical_json(_jsonable(policy_state))
            except (TypeError, ValueError):
                state_ok = False
                policy_state = None
        try:
            history_json = _jsonable(list(history))
            _canonical_json(history_json)
            history_ok = True
        except (TypeError, ValueError):
            history_json = None
            history_ok = False
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "kind": "plumb_rollout_segment_checkpoint",
            "scenario_id": scenario.identity,
            "segment_index": segment_index,
            "status": "completed",
            "action_offset": action_offset,
            "state_values8": list(state.values8()),
            "state_source": state.source,
            "history": history_json,
            "history_hashes": [_frame_hash(frame) for frame in history],
            "policy_reset_state": _safe_jsonable(reset_state),
            "policy_state": _safe_jsonable(policy_state),
            "rng": {"scenario_seed": scenario.seed, "next_segment_seed": _segment_seed(scenario.seed, action_offset)},
            "request": {
                "request_id": request.request_id,
                "seed": request.seed,
                "action_count": len(request.compiled_actions),
                "profile_id": request.compatibility_profile_id,
                "feedback_mode": request.feedback_mode.value,
            },
            "native_actions": [list(row) for row in native_actions],
            "compiled_actions": [list(row) for row in backend_actions],
            "generated_frame_hashes": list(generated_hashes),
            "world": {"backend": result.backend, "profile_id": result.profile_id, "timing": _timing_dict(result.timing)},
            "protocol_identity": (
                PROTOCOL_CERTIFIED_TERMINAL_PADDING if padding else PROTOCOL_EXACT_TERMINAL_HORIZON
            ),
            "padding": dict(padding) if padding else None,
            "resumable": bool(state_ok and history_ok),
        }
        return payload, bool(state_ok and history_ok)

    def _restore_checkpoint(
        self, store: Any, prefix: str, policy: Any
    ) -> Tuple[
        int,
        BridgeState,
        List[Any],
        Any,
        List[Mapping[str, Any]],
        List[Tuple[float, ...]],
        List[Tuple[float, ...]],
        List[str],
    ]:
        if not callable(getattr(store, "list_json", None)) or not callable(getattr(store, "read_json", None)):
            raise ResumeUnsupportedError("artifact store cannot list/read durable checkpoints")
        references = list(store.list_json(prefix + "/checkpoints"))
        if not references:
            raise ResumeUnsupportedError("no completed segment checkpoint exists for this rollout")
        completed: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for reference in references:
            payload = store.read_json(reference)
            if payload.get("status") == "completed":
                completed.append((reference, payload))
        if not completed:
            raise ResumeUnsupportedError("no completed segment checkpoint exists for this rollout")
        ordered = sorted(completed, key=lambda pair: int(pair[1].get("segment_index", -1)))
        reference, checkpoint = ordered[-1]
        if not checkpoint.get("resumable"):
            raise ResumeUnsupportedError("checkpoint cannot restore adapter state/history; refusing nondeterministic resume")
        restore = getattr(policy, "restore_state", None)
        if not callable(restore):
            raise ResumeUnsupportedError("policy adapter lacks restore_state; refusing resume")
        try:
            restore(checkpoint.get("policy_state"))
            state = BridgeState.from_values(tuple(checkpoint["state_values8"]), source=str(checkpoint.get("state_source", "forecast")))
            history = list(checkpoint["history"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ResumeUnsupportedError("checkpoint cannot restore policy state/history") from exc
        if not history:
            raise ResumeUnsupportedError("checkpoint has empty observation history")
        native_actions: List[Tuple[float, ...]] = []
        compiled_actions: List[Tuple[float, ...]] = []
        frame_hashes: List[str] = []
        for _, prior in ordered:
            try:
                native_actions.extend(tuple(float(value) for value in row) for row in prior["native_actions"])
                compiled_actions.extend(tuple(float(value) for value in row) for row in prior["compiled_actions"])
                frame_hashes.extend(str(value) for value in prior["generated_frame_hashes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ResumeUnsupportedError("checkpoint has malformed action/frame lineage") from exc
        if len(native_actions) != int(checkpoint["action_offset"]) or len(compiled_actions) != len(native_actions):
            raise ResumeUnsupportedError("checkpoint action lineage does not match durable action offset")
        return (
            int(checkpoint["action_offset"]),
            state,
            history,
            checkpoint.get("policy_reset_state"),
            [dict(ref) for ref, _ in ordered],
            native_actions,
            compiled_actions,
            frame_hashes,
        )

    def _validity(
        self,
        validity_gate: Any,
        *,
        scenario: Scenario,
        frames: Tuple[Any, ...],
        actions: Tuple[Tuple[float, ...], ...],
        timestamps: Tuple[float, ...],
        states: Tuple[Tuple[float, ...], ...],
        rollout_id: str,
    ) -> "_ControllerValidity":
        """Run the injected Stage A gate, or record loudly that none exists.

        The previous implementation called ``judge.validate`` and fell back to
        the string ``"valid"`` when the judge had no such method, which marked
        every episode valid.  There is no fallback to ``"valid"`` here: an
        absent gate, a raising gate and an off-contract gate all produce
        ``"unknown"`` with a reason code.
        """

        if validity_gate is None:
            return _ControllerValidity(
                "unknown",
                (VALIDITY_GATE_NOT_CONFIGURED,),
                {
                    "validity": "unknown",
                    "reason_codes": [VALIDITY_GATE_NOT_CONFIGURED],
                    "gate_configured": False,
                    "note": (
                        "No Stage A deterministic validity gate was supplied. An unrun check is not a passed check, "
                        "so this episode is unevaluable rather than valid."
                    ),
                },
            )
        evaluate = getattr(validity_gate, "evaluate", None)
        if not callable(evaluate):
            if callable(validity_gate):
                evaluate = validity_gate
            else:
                raise JudgeWiringError(
                    "validity_gate %r exposes no callable evaluate(); supply a StageAValidityGate or a callable"
                    % type(validity_gate).__name__
                )
        # The signature is checked *before* the call so a wiring defect raises
        # loudly while a data-dependent runtime error stays a recorded unknown.
        missing_parameters = _missing_stage_a_parameters(evaluate)
        if missing_parameters:
            raise JudgeWiringError(
                "validity gate %s does not accept the Stage A call contract; missing keyword parameters %s. "
                "The contract is evaluate(*, frames, actions, nominal_timestamps, conditioning_frame, states, episode_id)."
                % (type(validity_gate).__name__, ", ".join(missing_parameters))
            )
        try:
            outcome = evaluate(
                frames=frames,
                actions=actions,
                nominal_timestamps=timestamps,
                conditioning_frame=scenario.initial_rgb,
                states=states,
                episode_id="%s:%s" % (rollout_id, scenario.identity),
            )
        except Exception as exc:
            return _ControllerValidity(
                "unknown",
                (VALIDITY_GATE_ERROR,),
                {
                    "validity": "unknown",
                    "reason_codes": [VALIDITY_GATE_ERROR],
                    "gate_configured": True,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        return _coerce_validity_outcome(outcome)

    def _judge(
        self,
        scenario: Scenario,
        judge: Any,
        frames: Tuple[Any, ...],
        frame_hashes: Sequence[str],
        timestamps: Sequence[float],
        *,
        rollout_id: str,
    ) -> Tuple[Optional[bool], Optional[int], Optional[str], Dict[str, Any]]:
        """Call the Stage B rubric judge under its own frozen contract.

        Wiring and schema defects raise :class:`JudgeWiringError`.  Refusals,
        transport failures and schema failures become distinct explicit missing
        reasons, so a broken connection can never be mistaken for a judge that
        declined to answer.
        """

        evidence: Dict[str, Any] = {}
        if judge is None:
            return None, None, "judge_not_configured", evidence
        evaluate = judge if callable(judge) and not hasattr(judge, "evaluate") else getattr(judge, "evaluate", None)
        if not callable(evaluate):
            raise JudgeWiringError(
                "judge %r exposes no callable evaluate(); the controller will not guess a scoring entry point"
                % type(judge).__name__
            )

        try:
            indices = judge_frame_indices(len(frames))
        except ClipTooShortError:
            evidence["judge_protocol"] = JUDGE_PROTOCOL_REQUEST
            evidence["judge_clip_frame_count"] = len(frames)
            return None, None, "clip_too_short_for_judge_protocol", evidence
        sampled_frames = tuple(frames[index] for index in indices)
        sampled_hashes = tuple(frame_hashes[index] for index in indices)
        sampled_timestamps = tuple(timestamps[index] for index in indices)
        evidence["judge_frame_indices"] = indices
        evidence["judge_frame_hashes"] = sampled_hashes

        if not _judge_accepts_seeds(evaluate):
            evidence["judge_protocol"] = JUDGE_PROTOCOL_LEGACY_PAYLOAD
            return self._judge_legacy(
                scenario, evaluate, sampled_frames, sampled_hashes, sampled_timestamps, evidence
            )

        evidence["judge_protocol"] = JUDGE_PROTOCOL_REQUEST
        input_error, load_error, refusal_error, schema_error = _judge_error_types()
        seeds = derive_judge_seeds(_judge_seed_material(scenario, rollout_id))
        evidence["judge_seeds"] = seeds
        if not scenario.reference_images:
            return None, None, "judge_reference_images_unavailable", evidence
        try:
            request, metadata = build_judge_request(
                scenario,
                sampled_frames,
                sampled_timestamps,
                rollout_id=rollout_id,
                frame_indices=indices,
            )
        except ClipTooShortError:
            return None, None, "clip_too_short_for_judge_protocol", evidence
        except input_error as exc:
            raise JudgeWiringError(
                "the controller built a JudgeRequest the frozen judge protocol rejects: %s" % exc
            ) from exc
        evidence.update(metadata)

        try:
            report = evaluate(request, seeds=seeds)
        except TypeError as exc:
            raise JudgeWiringError(
                "judge %s does not accept evaluate(request, *, seeds=...); this is the controller/judge wiring "
                "contract, not a judge refusal: %s" % (type(judge).__name__, exc)
            ) from exc
        except AttributeError as exc:
            raise JudgeWiringError("judge %s is incompletely wired: %s" % (type(judge).__name__, exc)) from exc
        except refusal_error as exc:
            evidence["judge_error"] = {"type": type(exc).__name__, "message": str(exc)}
            return None, None, "judge_refusal", evidence
        except schema_error as exc:
            evidence["judge_error"] = {"type": type(exc).__name__, "message": str(exc)}
            return None, None, "judge_schema_failure", evidence
        except input_error as exc:
            raise JudgeWiringError(
                "the frozen judge rejected the controller's request or seeds: %s" % exc
            ) from exc
        except (load_error, BackendUnavailableError, OSError, ConnectionError, TimeoutError) as exc:
            evidence["judge_error"] = {"type": type(exc).__name__, "message": str(exc)}
            return None, None, "judge_transport_failure:%s" % type(exc).__name__, evidence

        return self._judge_outcome(report, evidence)

    @staticmethod
    def _judge_outcome(
        report: Any, evidence: Dict[str, Any]
    ) -> Tuple[Optional[bool], Optional[int], Optional[str], Dict[str, Any]]:
        success = getattr(report, "binary_success", None)
        progress = getattr(report, "progress", getattr(report, "progress_score", None))
        status = getattr(report, "judge_status", None)
        missing = getattr(report, "missing_reason", None)
        if success not in (None, True, False):
            raise JudgeWiringError("judge binary_success must be bool or null, got %r" % (success,))
        if progress is not None and (
            isinstance(progress, bool) or not isinstance(progress, int) or not 0 <= progress <= 5
        ):
            raise JudgeWiringError("judge progress must be an integer 0..5 or null, got %r" % (progress,))
        if status is not None:
            evidence["judge_status"] = str(status)
        if success is None:
            return None, progress, str(missing or "judge_unevaluable"), evidence
        return bool(success), progress, None, evidence

    def _judge_legacy(
        self,
        scenario: Scenario,
        evaluate: Any,
        sampled_frames: Tuple[Any, ...],
        sampled_hashes: Tuple[str, ...],
        sampled_timestamps: Tuple[float, ...],
        evidence: Dict[str, Any],
    ) -> Tuple[Optional[bool], Optional[int], Optional[str], Dict[str, Any]]:
        """Diagnostic path for a fixture judge that takes one payload object.

        It is separately named in the record so a fixture result can never be
        confused with a frozen-protocol judge result.  A signature mismatch
        still raises rather than becoming a silent missing outcome.
        """

        payload = RolloutJudgeInput(
            task=scenario.task,
            frames=sampled_frames,
            nominal_timestamps=sampled_timestamps,
            goal_image=scenario.goal_image,
            frame_hashes=sampled_hashes,
        )
        try:
            outcome = evaluate(payload)
        except TypeError as exc:
            raise JudgeWiringError(
                "judge %r accepts neither evaluate(request, *, seeds=...) nor evaluate(payload): %s"
                % (getattr(evaluate, "__qualname__", evaluate), exc)
            ) from exc
        if isinstance(outcome, Mapping):
            success = outcome.get("binary_success")
            progress = outcome.get("progress_score")
            missing = outcome.get("missing_reason")
        else:
            success = getattr(outcome, "binary_success", None)
            progress = getattr(outcome, "progress_score", None)
            missing = getattr(outcome, "missing_reason", None)
        if success not in (None, True, False):
            raise JudgeWiringError("judge binary_success must be bool or null, got %r" % (success,))
        if progress is not None and (
            isinstance(progress, bool) or not isinstance(progress, int) or not 0 <= progress <= 5
        ):
            raise JudgeWiringError("judge progress_score must be an integer 0..5 or null, got %r" % (progress,))
        if success is None:
            return None, progress, str(missing or "judge_unevaluable"), evidence
        return bool(success), progress, None, evidence

    def _report(
        self,
        rollout_id: str,
        scenario: Scenario,
        status: str,
        executed_actions: int,
        segments: Sequence[SegmentRecord],
        native_actions: Sequence[Tuple[float, ...]],
        compiled_actions: Sequence[Tuple[float, ...]],
        frame_hashes: Sequence[str],
        timestamps: Sequence[float],
        checkpoints: Sequence[Mapping[str, Any]],
        world_calls: int,
        started: float,
        validity: str,
        success: Optional[bool],
        progress: Optional[int],
        missing_reason: Optional[str],
        error: Optional[Mapping[str, str]],
        feedback_mode: Optional[FeedbackMode] = None,
        profile: Optional[WorldActionProfile] = None,
        padding_events: Sequence[Mapping[str, Any]] = (),
        validity_gate_configured: bool = False,
        validity_reason_codes: Sequence[str] = (),
        validity_report: Optional[Mapping[str, Any]] = None,
        judge_evidence: Optional[Mapping[str, Any]] = None,
    ) -> RolloutReport:
        feedback = (feedback_mode or self.feedback_mode).value
        evidence = dict(judge_evidence or {})
        return RolloutReport(
            rollout_id=rollout_id,
            scenario_id=scenario.identity,
            status=status,
            mode=self.mode,
            qualified=self.mode == "qualification",
            feedback_mode=feedback,
            # A state forecast is a controller estimate, not a robot sensor.
            physical_state_measured=False,
            horizon_actions=scenario.horizon_actions,
            executed_actions=executed_actions,
            segments=tuple(segments),
            native_actions=tuple(native_actions),
            compiled_actions=tuple(compiled_actions),
            frame_hashes=tuple(frame_hashes),
            nominal_timestamps=tuple(timestamps),
            validity=validity,
            binary_success=success,
            progress_score=progress,
            missing_reason=missing_reason,
            checkpoint_refs=tuple(checkpoints),
            world_calls=world_calls,
            timings={"controller_wall_seconds": time.perf_counter() - started},
            error=dict(error) if error else None,
            protocol_identity=(
                PROTOCOL_CERTIFIED_TERMINAL_PADDING if padding_events else PROTOCOL_EXACT_TERMINAL_HORIZON
            ),
            padding_events=tuple(dict(item) for item in padding_events),
            world_action_length_certification=(profile.certification_record() if profile is not None else {}),
            validity_gate_configured=bool(validity_gate_configured),
            validity_reason_codes=tuple(str(code) for code in validity_reason_codes),
            validity_report=dict(validity_report) if validity_report else None,
            judge_protocol=evidence.get("judge_protocol"),
            judge_seeds=tuple(int(value) for value in evidence.get("judge_seeds", ())),
            judge_frame_indices=tuple(int(value) for value in evidence.get("judge_frame_indices", ())),
            judge_frame_hashes=tuple(str(value) for value in evidence.get("judge_frame_hashes", ())),
            judge_rubric_hash=evidence.get("judge_rubric_hash"),
            judge_task_registry_hash=evidence.get("judge_task_registry_hash"),
        )

    def _write_json(self, store: Any, key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        reference = store.write_json(key, payload)
        if not isinstance(reference, Mapping):
            raise RolloutConfigurationError("artifact_store.write_json must return an artifact mapping")
        return dict(reference)

    def _write_terminal(self, store: Any, prefix: str, report: RolloutReport) -> None:
        self._write_json(store, prefix + "/terminal_report.json", report.as_dict())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Do not pretend an opaque frame/object is durably restorable.
    raise TypeError("value is not JSON-restorable")


def _safe_jsonable(value: Any) -> Any:
    try:
        return _jsonable(value)
    except TypeError:
        return None


def _frame_hash(frame: Any) -> str:
    if isinstance(frame, bytes):
        encoded = frame
    elif isinstance(frame, str):
        encoded = frame.encode("utf-8")
    else:
        try:
            encoded = _canonical_json(_jsonable(frame)).encode("utf-8")
        except (TypeError, ValueError):
            encoded = (type(frame).__module__ + ":" + type(frame).__qualname__ + ":" + repr(frame)).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _segment_seed(seed: int, action_offset: int) -> int:
    payload = ("%d:%d" % (seed, action_offset)).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _timing_dict(timing: Any) -> Dict[str, Any]:
    if callable(getattr(timing, "as_dict", None)):
        return dict(timing.as_dict())
    if is_dataclass(timing):
        return dict(asdict(timing))
    if isinstance(timing, Mapping):
        return dict(timing)
    return {"timing": None}


@dataclass(frozen=True)
class _ControllerValidity:
    validity: str
    reason_codes: Tuple[str, ...]
    report: Optional[Mapping[str, Any]]


def _coerce_validity_outcome(outcome: Any) -> _ControllerValidity:
    """Accept a ValidityReport, a mapping, or a bare validity string.

    Anything that does not carry one of ``valid``/``invalid``/``unknown`` is a
    contract violation and becomes ``unknown``, never ``valid``.
    """

    report: Optional[Mapping[str, Any]] = None
    if isinstance(outcome, Mapping):
        validity = outcome.get("validity")
        codes = outcome.get("reason_codes") or ()
        report = dict(outcome)
    elif hasattr(outcome, "validity"):
        validity = getattr(outcome, "validity")
        codes = getattr(outcome, "reason_codes", ()) or ()
        as_dict = getattr(outcome, "as_dict", None)
        report = dict(as_dict()) if callable(as_dict) else None
    elif isinstance(outcome, str):
        validity = outcome
        codes = ()
    else:
        return _ControllerValidity(
            "unknown",
            (VALIDITY_GATE_CONTRACT_VIOLATION,),
            {
                "validity": "unknown",
                "reason_codes": [VALIDITY_GATE_CONTRACT_VIOLATION],
                "returned_type": type(outcome).__name__,
            },
        )
    if validity not in ("valid", "invalid", "unknown"):
        return _ControllerValidity(
            "unknown",
            (VALIDITY_GATE_CONTRACT_VIOLATION,),
            {
                "validity": "unknown",
                "reason_codes": [VALIDITY_GATE_CONTRACT_VIOLATION],
                "returned_validity": None if validity is None else str(validity),
            },
        )
    if isinstance(codes, str):
        codes = (codes,)
    return _ControllerValidity(str(validity), tuple(str(code) for code in codes), report)


def judge_frame_indices(frame_count: int) -> Tuple[int, ...]:
    """The single authoritative Stage B sampling rule.

    Exactly 16 uniformly spaced indices from the original start through the
    exact final control tick, both endpoints included.  A clip with fewer than
    16 frames is rejected (spec 5) instead of being resampled into a shorter
    view that ``JudgeRequest.validate`` would refuse anyway.
    """

    count = int(frame_count)
    if count < JUDGE_FRAME_COUNT:
        raise ClipTooShortError(
            "the frozen judge protocol needs exactly %d frames including both endpoints; this clip has %d. "
            "Reject it as unevaluable rather than altering the sampling rule."
            % (JUDGE_FRAME_COUNT, count)
        )
    indices = tuple(round(index * (count - 1) / (JUDGE_FRAME_COUNT - 1)) for index in range(JUDGE_FRAME_COUNT))
    if indices[0] != 0 or indices[-1] != count - 1 or len(set(indices)) != JUDGE_FRAME_COUNT:
        raise ClipTooShortError(
            "uniform 16-frame sampling of a %d-frame clip did not produce 16 distinct endpoint-inclusive indices"
            % count
        )
    return indices


def _judge_seed_material(scenario: Scenario, rollout_id: str) -> str:
    return _canonical_json(
        {
            "kind": "plumb_judge_seed_material_v1",
            "scenario_id": scenario.identity,
            "scenario_seed": scenario.seed,
            "start_lineage_id": scenario.start_lineage_id,
            "task": scenario.task,
            "rollout_id": rollout_id,
        }
    )


def derive_judge_seeds(seed_material: str, count: int = JUDGE_SAMPLE_COUNT) -> Tuple[int, ...]:
    """Derive the five logged judge seeds deterministically from episode material.

    The judge requires five *distinct* integer seeds, so the counter advances
    until the derived values are distinct instead of silently reusing one.
    """

    seeds: List[int] = []
    counter = 0
    while len(seeds) < count:
        digest = hashlib.sha256(("plumb-judge-seed:%s:%d" % (seed_material, counter)).encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        if value not in seeds:
            seeds.append(value)
        counter += 1
        if counter > 10000:  # pragma: no cover - defensive; collisions are astronomically unlikely.
            raise RolloutConfigurationError("could not derive %d distinct judge seeds" % count)
    return tuple(seeds)


def _judge_error_types() -> Tuple[type, type, type, type]:
    """Resolve the judge's typed exceptions lazily.

    A missing judge module yields inert placeholder types so the typed except
    clauses stay typed rather than collapsing into a bare ``except Exception``.
    """

    try:
        from .policies.judge import JudgeInputError, JudgeLoadError, JudgeRefusalError, JudgeSchemaError
    except ImportError:  # pragma: no cover - the judge module ships with the package.
        placeholder = _UnreachableJudgeError
        return (placeholder, placeholder, placeholder, placeholder)
    return (JudgeInputError, JudgeLoadError, JudgeRefusalError, JudgeSchemaError)


#: The Stage A call contract the controller uses for every validity gate.
STAGE_A_CALL_PARAMETERS: Tuple[str, ...] = (
    "frames",
    "actions",
    "nominal_timestamps",
    "conditioning_frame",
    "states",
    "episode_id",
)


def _missing_stage_a_parameters(evaluate: Any) -> Tuple[str, ...]:
    """Stage A keywords this callable cannot accept, checked before calling it."""

    try:
        signature = inspect.signature(evaluate)
    except (TypeError, ValueError):
        return ()
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return ()
    accepted = {
        name
        for name, parameter in parameters.items()
        if parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    return tuple(name for name in STAGE_A_CALL_PARAMETERS if name not in accepted)


def _judge_accepts_seeds(evaluate: Any) -> bool:
    """Whether this callable implements ``evaluate(request, *, seeds=...)``."""

    try:
        signature = inspect.signature(evaluate)
    except (TypeError, ValueError):
        return False
    return "seeds" in signature.parameters


def build_judge_request(
    scenario: Scenario,
    frames: Sequence[Any],
    timestamps: Sequence[float],
    *,
    rollout_id: str,
    frame_indices: Sequence[int] = (),
) -> Tuple[Any, Dict[str, Any]]:
    """Build a real ``JudgeRequest`` plus the provenance the ledger records.

    Primary scoring passes the canonical ``task_id`` only: the judge resolves
    the frozen instruction and rubric from the task registry, so no free-text
    rubric, policy name, action row, or reference rate can reach Stage B.  The
    resolved rubric hash travels in the returned metadata instead.
    """

    from .policies.judge import JudgeInputProvenance, JudgeRequest, ReferenceImage
    from .policies.tasks import BENCHMARK_TASK_REGISTRY, TASK_REGISTRY_HASH

    if len(frames) != JUDGE_FRAME_COUNT or len(timestamps) != JUDGE_FRAME_COUNT:
        raise ClipTooShortError(
            "build_judge_request needs exactly %d sampled frames and timestamps" % JUDGE_FRAME_COUNT
        )
    if not scenario.reference_images:
        raise RolloutConfigurationError(
            "the judge protocol requires at least one provenance-backed reference image; "
            "Scenario.reference_images is empty and a digest will not be invented"
        )
    references = tuple(
        ReferenceImage(image=item.image, source_uri=item.source_uri, sha256=item.sha256)
        for item in scenario.reference_images
    )
    task = BENCHMARK_TASK_REGISTRY.get(scenario.task)
    request = JudgeRequest(
        frames=tuple(frames),
        frame_timestamps=tuple(float(value) for value in timestamps),
        reference_images=references,
        task_id=scenario.task,
        provenance=JudgeInputProvenance(clip_id="%s:%s" % (rollout_id, scenario.identity)),
    )
    request.validate()
    metadata = {
        "judge_rubric_hash": task.rubric_hash,
        "judge_task_registry_hash": TASK_REGISTRY_HASH,
        "judge_frame_indices": tuple(int(value) for value in frame_indices),
        "judge_reference_image_sources": tuple(item.source_uri for item in scenario.reference_images),
    }
    return request, metadata


__all__ = [
    "ArtifactStore",
    "ClipTooShortError",
    "FileArtifactStore",
    "JUDGE_FRAME_COUNT",
    "JUDGE_PROTOCOL_LEGACY_PAYLOAD",
    "JUDGE_PROTOCOL_REQUEST",
    "JUDGE_SAMPLE_COUNT",
    "JudgeWiringError",
    "PolicyControlPlan",
    "QualificationEvidence",
    "ReferenceImageProvenance",
    "ResumeUnsupportedError",
    "RolloutConfigurationError",
    "RolloutController",
    "RolloutJudgeInput",
    "RolloutReport",
    "Scenario",
    "SegmentRecord",
    "UncertifiedPaddingError",
    "UnsupportedActionLengthError",
    "VALIDITY_GATE_CONTRACT_VIOLATION",
    "VALIDITY_GATE_ERROR",
    "VALIDITY_GATE_NOT_CONFIGURED",
    "WorldActionProfile",
    "build_judge_request",
    "derive_judge_seeds",
    "judge_frame_indices",
]
