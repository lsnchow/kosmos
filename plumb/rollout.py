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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from .adapters.bridge import BridgeState
from .adapters.contracts import FeedbackMode, PolicyContract, PolicyObservation, WorldRequest, WorldResult
from .adapters.worlds import BackendUnavailableError


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


class ArtifactStore(Protocol):
    def write_json(self, key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def read_json(self, reference: Any) -> Mapping[str, Any]:
        ...

    def list_json(self, prefix: str) -> Sequence[Any]:
        ...


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
    """The explicitly normal (not probe) action lengths a backend can execute."""

    profile_id: str
    domain: str
    supported_action_lengths: Tuple[int, ...]
    control_hz: float = 5.0

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
        rollout_id: Optional[str] = None,
        resume: bool = False,
    ) -> RolloutReport:
        """Run the exact scenario horizon or return an explicit partial record.

        Unsupported action lengths are raised before a world call. Runtime
        world errors are accounted as nullable, non-evaluable reports because
        an unavailable output is neither a success nor a known failure.
        """

        store = self._coerce_store(artifact_store)
        profile = self._world_profile(world)
        self._check_qualification(profile)
        self._check_horizon_representable(scenario.horizon_actions, profile)
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
        segments: List[SegmentRecord] = []
        world_calls = 0
        started = time.perf_counter()
        effective_feedback_mode: Optional[FeedbackMode] = None

        while action_offset < scenario.horizon_actions:
            observation = self._observation(scenario, policy, history, state, action_offset, profile.control_hz)
            plan = self._plan(policy, observation, scenario.horizon_actions - action_offset)
            self._validate_plan_history(plan, policy, history)
            count = plan.select_prefix(scenario.horizon_actions - action_offset)
            if count not in profile.supported_action_lengths:
                policy_name = self._policy_contract(policy).name if self._policy_contract(policy) else type(policy).__name__
                if count == 1 and 16 in profile.supported_action_lengths:
                    raise UnsupportedActionLengthError(
                        "one-step native proposal from %s is unsupported by %s normal lengths %s; "
                        "it will not be repeated or padded to 16"
                        % (policy_name, profile.profile_id, profile.supported_action_lengths)
                    )
                raise UnsupportedActionLengthError(
                    "native verified prefix %d is unsupported by %s normal action lengths %s; no world call was made"
                    % (count, profile.profile_id, profile.supported_action_lengths)
                )

            native_actions = tuple(tuple(float(value) for value in row) for row in plan.actions[:count])
            world_called = False
            try:
                compiled = compiler.compile(state, native_actions)
                backend_actions = tuple(tuple(float(value) for value in row) for row in compiled.backend_actions)
                forecast_states = tuple(compiled.forecast_states)
                if len(backend_actions) != count or len(forecast_states) != count + 1:
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
                        (action_offset + index + 1) / profile.control_hz for index in range(count)
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
                result.validate(count)
                generated = result.future_frames
                if len(generated) != count:
                    raise RolloutConfigurationError("world result did not provide exactly one future frame per action")
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
                )
                self._write_terminal(store, prefix, report)
                return report

            # ``future_frames`` removes an echoed conditioning frame exactly
            # once. The next native policy call gets a freshly generated image.
            old_offset = action_offset
            history.append(generated[-1])
            required = self._policy_contract(policy).required_observation_history if self._policy_contract(policy) else plan.history_length
            history = history[-max(1, required):]
            state = forecast_states[-1]
            action_offset += count
            all_native_actions.extend(native_actions)
            all_compiled_actions.extend(backend_actions)
            generated_hashes = tuple(_frame_hash(frame) for frame in generated)
            frame_hashes.extend(generated_hashes)
            judge_frames.extend(generated)
            timestamps.extend((old_offset + index + 1) / profile.control_hz for index in range(count))

            checkpoint_payload, resumable = self._checkpoint_payload(
                scenario, action_offset, state, history, policy, reset_state, len(segments), request, native_actions,
                backend_actions, generated_hashes, result,
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
                    compiled_actions=backend_actions,
                    world_request_id=request.request_id or "",
                    world_timing={**_timing_dict(result.timing), "controller_wall_seconds": call_elapsed, "resumable": resumable},
                    generated_frame_hashes=generated_hashes,
                    checkpoint_ref=checkpoint_ref,
                )
            )

        if resume and judge is not None:
            # Segment checkpoints retain the current history, action records,
            # and frame hashes. They do not pretend to reconstruct every raw
            # prior frame, so a resumed run is explicitly unevaluable unless a
            # future artifact-store profile can restore the full frame stream.
            validity, success, progress, missing_reason = (
                "unknown", None, None, "judge_requires_full_frame_artifacts_after_resume"
            )
        else:
            validity, success, progress, missing_reason = self._judge(
                scenario, judge, tuple(judge_frames), frame_hashes, timestamps
            )
        report = self._report(
            rollout_id, scenario, "completed", action_offset, segments, all_native_actions, all_compiled_actions,
            frame_hashes, timestamps, checkpoint_refs, world_calls, started, validity, success, progress,
            missing_reason, None,
            feedback_mode=effective_feedback_mode,
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
        return WorldActionProfile(str(profile_id), str(domain), tuple(int(value) for value in lengths), float(control_hz))

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

    @staticmethod
    def _check_horizon_representable(horizon: int, profile: WorldActionProfile) -> None:
        reachable = [False] * (horizon + 1)
        reachable[0] = True
        for total in range(1, horizon + 1):
            reachable[total] = any(total >= length and reachable[total - length] for length in profile.supported_action_lengths)
        if not reachable[horizon]:
            raise UnsupportedActionLengthError(
                "%s cannot execute the exact %d-action terminal horizon from normal supported action lengths %s; "
                "probe/action padding is forbidden"
                % (profile.profile_id, horizon, profile.supported_action_lengths)
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

    def _judge(
        self,
        scenario: Scenario,
        judge: Any,
        frames: Tuple[Any, ...],
        frame_hashes: Sequence[str],
        timestamps: Sequence[float],
    ) -> Tuple[str, Optional[bool], Optional[int], Optional[str]]:
        if judge is None:
            return "unknown", None, None, "judge_not_configured"
        sampled_frames, sampled_hashes, sampled_timestamps = _judge_sample(frames, frame_hashes, timestamps)
        payload = RolloutJudgeInput(
            task=scenario.task,
            frames=sampled_frames,
            nominal_timestamps=sampled_timestamps,
            goal_image=scenario.goal_image,
            frame_hashes=sampled_hashes,
        )
        try:
            validity_result = judge.validate(payload) if callable(getattr(judge, "validate", None)) else "valid"
            validity = validity_result.get("validity", "unknown") if isinstance(validity_result, Mapping) else str(validity_result)
            if validity not in ("valid", "invalid", "unknown"):
                raise ValueError("judge validity must be valid, invalid, or unknown")
            if validity != "valid":
                return validity, None, None, "validity_%s" % validity
            evaluate = getattr(judge, "evaluate", None) if not callable(judge) else judge
            if not callable(evaluate):
                return "valid", None, None, "judge_has_no_evaluate_method"
            outcome = evaluate(payload)
            if isinstance(outcome, Mapping):
                success = outcome.get("binary_success")
                progress = outcome.get("progress_score")
                missing = outcome.get("missing_reason")
            else:
                success = getattr(outcome, "binary_success", None)
                progress = getattr(outcome, "progress_score", None)
                missing = getattr(outcome, "missing_reason", None)
            if success not in (None, True, False, 0, 1):
                raise ValueError("judge binary_success must be bool or null")
            if progress is not None and (isinstance(progress, bool) or not isinstance(progress, int) or not 0 <= progress <= 5):
                raise ValueError("judge progress_score must be 0..5 or null")
            return "valid", None if success is None else bool(success), progress, None if success is not None else str(missing or "judge_unevaluable")
        except Exception as exc:
            return "unknown", None, None, "judge_failure:%s" % type(exc).__name__

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
    ) -> RolloutReport:
        feedback = (feedback_mode or self.feedback_mode).value
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


def _judge_sample(
    frames: Sequence[Any], frame_hashes: Sequence[str], timestamps: Sequence[float]
) -> Tuple[Tuple[Any, ...], Tuple[str, ...], Tuple[float, ...]]:
    """Use the frozen 16-frame endpoint-inclusive judge view when possible."""

    if not (len(frames) == len(frame_hashes) == len(timestamps)):
        raise RolloutConfigurationError("judge frame data must have aligned frames, hashes, and timestamps")
    if len(frames) <= 16:
        return tuple(frames), tuple(frame_hashes), tuple(timestamps)
    indices = tuple(round(index * (len(frames) - 1) / 15) for index in range(16))
    return (
        tuple(frames[index] for index in indices),
        tuple(frame_hashes[index] for index in indices),
        tuple(timestamps[index] for index in indices),
    )


__all__ = [
    "ArtifactStore",
    "FileArtifactStore",
    "PolicyControlPlan",
    "QualificationEvidence",
    "ResumeUnsupportedError",
    "RolloutConfigurationError",
    "RolloutController",
    "RolloutJudgeInput",
    "RolloutReport",
    "Scenario",
    "SegmentRecord",
    "UnsupportedActionLengthError",
    "WorldActionProfile",
]
