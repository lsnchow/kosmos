"""Controller contracts with fake adapters; no model assets or physics claims."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from plumb.adapters.bridge import BridgeState
from plumb.adapters.contracts import CapabilityStatus, FeedbackMode, PolicyContract, ServerTiming, WorldResult
from plumb.adapters.worlds import (
    GATE_A_CERTIFIED_CLASS,
    PROTOCOL_CERTIFIED_TERMINAL_PADDING,
    PROTOCOL_EXACT_TERMINAL_HORIZON,
    BackendUnavailableError,
    TerminalPaddingCertificate,
)
from plumb.policies.judge import (
    JudgeInputError,
    JudgeLoadError,
    JudgeRefusalError,
    JudgeReport,
    JudgeRequest,
    JudgeSchemaError,
)
from plumb.policies.tasks import BENCHMARK_TASK_REGISTRY, TASK_REGISTRY_HASH
from plumb.rollout import (
    JUDGE_FRAME_COUNT,
    JUDGE_PROTOCOL_LEGACY_PAYLOAD,
    JUDGE_PROTOCOL_REQUEST,
    VALIDITY_GATE_CONTRACT_VIOLATION,
    VALIDITY_GATE_ERROR,
    VALIDITY_GATE_NOT_CONFIGURED,
    ClipTooShortError,
    FileArtifactStore,
    JudgeWiringError,
    PolicyControlPlan,
    QualificationEvidence,
    ReferenceImageProvenance,
    ResumeUnsupportedError,
    RolloutController,
    Scenario,
    UncertifiedPaddingError,
    UnsupportedActionLengthError,
    WorldActionProfile,
    derive_judge_seeds,
    judge_frame_indices,
)
from plumb.validity import CALIBRATION_DEVELOPMENT, StageAParameters, StageAValidityGate


CONTRACT = PolicyContract(
    name="Fake native policy",
    required_observation_history=1,
    requires_proprio=True,
    native_proposal_horizon=4,
    certified_execute_prefix=4,
    temporal_ensembling=False,
    preprocessing="test",
    normalization="test",
    reset_rule="test",
    rng_rule="test",
    implementation_status=CapabilityStatus.READY_UNQUALIFIED,
)

REFERENCE_IMAGE = ReferenceImageProvenance(
    image="goal-reference-frame", source_uri="fixture://goal/open_drawer.png", sha256="a" * 64
)


def scenario(task="open_drawer", *, reference_images=(REFERENCE_IMAGE,)):
    horizon = {"open_drawer": 70, "to_basket": 100, "fold_cloth": 80}[task]
    return Scenario(
        initial_rgb="initial:%s" % task,
        state=BridgeState.from_values((0, 0, 0, 0, 0, 0, 0, 0), source="measured"),
        task=task,
        horizon_actions=horizon,
        hashes={"image": "sha256:initial", "state": "sha256:state"},
        seed=13,
        reference_images=tuple(reference_images),
    )


class FakePolicy:
    contract = CONTRACT

    def __init__(self, terminal_prefixes=(2,)):
        self.observations = []
        self.terminal_prefixes = terminal_prefixes
        self.counter = 0
        self.restored = False

    def reset(self, seed):
        self.counter = 0
        return {"seed": seed}

    def plan(self, observation, remaining):
        self.observations.append(observation)
        self.counter += 1
        return PolicyControlPlan(
            actions=((1, 0, 0, 0, 0, 0, 0),) * 4,
            verified_prefix=4,
            history_length=1,
            verified_terminal_prefixes=self.terminal_prefixes,
        )


class StatefulFakePolicy(FakePolicy):
    def snapshot_state(self):
        return {"counter": self.counter}

    def restore_state(self, value):
        self.counter = value["counter"]
        self.restored = True


class SixteenStepPolicy(FakePolicy):
    contract = PolicyContract(
        name="Fake sixteen-step policy",
        required_observation_history=1,
        requires_proprio=True,
        native_proposal_horizon=16,
        certified_execute_prefix=16,
        temporal_ensembling=False,
        preprocessing="test",
        normalization="test",
        reset_rule="test",
        rng_rule="test",
        implementation_status=CapabilityStatus.READY_UNQUALIFIED,
    )

    def plan(self, observation, remaining):
        self.observations.append(observation)
        return PolicyControlPlan(
            actions=((1, 0, 0, 0, 0, 0, 0),) * 16, verified_prefix=16, history_length=1
        )


class OneStepPolicy(FakePolicy):
    contract = PolicyContract(
        name="Fake one-step policy",
        required_observation_history=1,
        requires_proprio=False,
        native_proposal_horizon=1,
        certified_execute_prefix=1,
        temporal_ensembling=False,
        preprocessing="test",
        normalization="test",
        reset_rule="test",
        rng_rule="test",
        implementation_status=CapabilityStatus.READY_UNQUALIFIED,
    )

    def plan(self, observation, remaining):
        self.observations.append(observation)
        return PolicyControlPlan(actions=((0, 0, 0, 0, 0, 0, 0),), verified_prefix=1, history_length=1)


@dataclass(frozen=True)
class FakeCompiled:
    backend_actions: tuple
    forecast_states: tuple
    feedback_mode: FeedbackMode = FeedbackMode.FORECAST_STATE


class FakeCompiler:
    def compile(self, state, actions):
        states = [state]
        current = state
        for action in actions:
            values = list(current.values8())
            values[0] += action[0]
            current = BridgeState.from_values(values, source="forecast")
            states.append(current)
        return FakeCompiled(tuple(tuple(float(x) for x in action) for action in actions), tuple(states))


class FakeWorld:
    def __init__(self, lengths=(4, 2), fail_call=None, profile=None):
        self.rollout_profile = profile or WorldActionProfile("fake-world", "bridge_orig_lerobot", tuple(lengths))
        self.requests = []
        self.fail_call = fail_call

    def generate(self, request):
        self.requests.append(request)
        if self.fail_call == len(self.requests):
            raise BackendUnavailableError("offline fake backend")
        frames = (request.conditioning_image,) + tuple(
            "generated:%d:%d" % (len(self.requests), index) for index in range(len(request.compiled_actions))
        )
        return WorldResult(
            backend="fake",
            profile_id=self.rollout_profile.profile_id,
            frames=frames,
            nominal_frame_timestamps=tuple(index / 5.0 for index in range(len(frames))),
            conditioning_frame_included=True,
            timing=ServerTiming(backend_calls=1, wall_seconds=0.01, cold_start=False),
        )


class NumpyWorld(FakeWorld):
    """Emits real uint8 frames so the deterministic Stage A gate can run."""

    def __init__(self, *, size=64, constant=False, lengths=(4, 2)):
        super().__init__(lengths=lengths)
        self.size = size
        self.constant = constant
        self._rng = np.random.default_rng(0)

    def _frame(self):
        if self.constant:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        return self._rng.integers(0, 255, size=(self.size, self.size, 3), dtype=np.uint8)

    def generate(self, request):
        self.requests.append(request)
        frames = (request.conditioning_image,) + tuple(self._frame() for _ in request.compiled_actions)
        return WorldResult(
            backend="fake-numpy",
            profile_id=self.rollout_profile.profile_id,
            frames=frames,
            nominal_frame_timestamps=tuple(index / 5.0 for index in range(len(frames))),
            conditioning_frame_included=True,
            timing=ServerTiming(backend_calls=1, wall_seconds=0.01, cold_start=False),
        )


class FakeJudge:
    """Legacy one-payload fixture judge; explicitly not the frozen protocol."""

    def __init__(self):
        self.inputs = []

    def evaluate(self, request):
        self.inputs.append(request)
        return {"binary_success": False, "progress_score": 3}


class ProtocolJudge:
    """A judge with the real ``evaluate(request, *, seeds)`` signature."""

    def __init__(self, report=None, error=None):
        self.calls = []
        self.error = error
        self.report = report or JudgeReport(
            binary_success=True,
            progress=5,
            judge_status="evaluable",
            missing_reason=None,
            agreeing_samples=4,
            sample_reports=(),
            wall_seconds=0.5,
            gpu_peak_memory_bytes=None,
        )

    def evaluate(self, request, *, seeds):
        self.calls.append((request, tuple(seeds)))
        if self.error is not None:
            raise self.error
        return self.report


class BrokenProtocolJudge:
    def evaluate(self, request, *, seeds, extra_required_argument):  # pragma: no cover - raises on call
        raise AssertionError("never reached")


class StubGate:
    def __init__(self, outcome="valid", error=None):
        self.outcome = outcome
        self.error = error
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.outcome


def valid_gate():
    return StubGate({"validity": "valid", "reason_codes": (), "parameters_hash": "sha256:" + "0" * 64})


def certified_profile(lengths=(4, 2), *, padding=None, profile_id="fake-world"):
    return WorldActionProfile(
        profile_id=profile_id,
        domain="bridge_orig_lerobot",
        supported_action_lengths=tuple(lengths),
        certification_class=GATE_A_CERTIFIED_CLASS,
        certification_id="cert-937398",
        certification_source_hash="sha256:" + "1" * 64,
        certification_source_uri="fixture://evidence/937398.json",
        terminal_padding_certificate=padding,
    )


def padding_certificate(profile_id="fake-world", *, padded=4, prefixes=(2,), residual=0.02):
    return TerminalPaddingCertificate(
        certificate_id="tpc-fixture",
        profile_id=profile_id,
        padded_action_length=padded,
        certified_prefix_lengths=prefixes,
        prefix_invariance_trials=12,
        max_observed_prefix_frame_mae=residual,
        tolerance_frame_mae=0.25,
        evidence_uri="fixture://evidence/prefix-invariance.json",
        evidence_sha256="b" * 64,
        tolerance_source="protocol.Tolerances.max_suffix_invariance_mae",
    )


# --- existing controller contracts ----------------------------------------


@pytest.mark.parametrize("task", ["open_drawer", "to_basket", "fold_cloth"])
def test_exact_horizons_use_verified_prefixes_and_fresh_generated_feedback(tmp_path, task):
    policy = FakePolicy()
    world = FakeWorld()
    judge = FakeJudge()
    report = RolloutController().execute(
        scenario(task), policy, world, FakeCompiler(), judge, FileArtifactStore(tmp_path), validity_gate=valid_gate()
    )

    assert report.status == "completed"
    assert report.executed_actions == report.horizon_actions
    assert sum(segment.action_count for segment in report.segments) == report.horizon_actions
    assert all(segment.action_count in (4, 2) for segment in report.segments)
    assert report.frame_hashes and len(report.frame_hashes) == report.horizon_actions + 1
    assert report.feedback_mode == "forecast_state"
    assert report.physical_state_measured is False
    assert report.validity == "valid"
    assert report.binary_success is False and report.progress_score == 3
    assert report.protocol_identity == PROTOCOL_EXACT_TERMINAL_HORIZON
    assert report.padding_events == ()
    assert all(len(request.compiled_actions) in (4, 2) for request in world.requests)
    # Every feedback request uses the final newly generated frame from the
    # prior segment, never the condition frame echoed by WorldResult.
    assert [request.conditioning_image for request in world.requests][0].startswith("initial:")
    assert all(
        world.requests[index].conditioning_image == "generated:%d:%d" % (index, len(world.requests[index - 1].compiled_actions) - 1)
        for index in range(1, len(world.requests))
    )
    assert all(observation.proprio is not None for observation in policy.observations)
    assert len(judge.inputs[0].frames) == JUDGE_FRAME_COUNT
    assert report.judge_protocol == JUDGE_PROTOCOL_LEGACY_PAYLOAD


def test_one_step_policy_is_not_padded_to_cosmos_16_before_any_world_call(tmp_path):
    world = FakeWorld(lengths=(16, 6))  # 70 is representable, a 1-step proposal is not.
    with pytest.raises(UnsupportedActionLengthError, match="one-step.*not be repeated or padded to 16"):
        RolloutController().execute(scenario(), OneStepPolicy(), world, FakeCompiler(), None, FileArtifactStore(tmp_path))
    assert world.requests == []


def test_world_unavailable_is_a_nullable_partial_record(tmp_path):
    report = RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(fail_call=1), FakeCompiler(), None, FileArtifactStore(tmp_path)
    )
    assert report.status == "world_unavailable"
    assert report.executed_actions == 0
    assert report.binary_success is None
    assert report.validity == "unknown"
    assert report.missing_reason == "world_failure"
    assert report.world_calls == 1
    assert report.segments[-1].status == "failed"


def test_resume_restores_policy_state_and_history_from_durable_segment_checkpoint(tmp_path):
    store = FileArtifactStore(tmp_path)
    policy = StatefulFakePolicy()
    first_world = FakeWorld(fail_call=2)
    first = RolloutController().execute(scenario(), policy, first_world, FakeCompiler(), None, store, rollout_id="resume-me")
    assert first.status == "world_unavailable" and first.executed_actions == 4

    recovered_policy = StatefulFakePolicy()
    recovered = RolloutController().execute(
        scenario(), recovered_policy, FakeWorld(), FakeCompiler(), None, store, rollout_id="resume-me", resume=True
    )
    assert recovered.status == "completed"
    assert recovered.executed_actions == 70
    assert len(recovered.native_actions) == len(recovered.compiled_actions) == 70
    assert len(recovered.frame_hashes) == len(recovered.nominal_timestamps) == 71
    assert recovered_policy.restored is True
    assert recovered_policy.observations[0].image_history == ("generated:1:3",)
    # A resumed clip is partial, so Stage A is recorded as not run rather than
    # evaluated into a spurious frame-count failure.
    assert recovered.validity == "unknown"
    assert recovered.validity_reason_codes == ("validity_requires_full_frame_artifacts_after_resume",)
    assert recovered.missing_reason == "judge_requires_full_frame_artifacts_after_resume"


def test_a_resumed_rollout_does_not_run_stage_a_on_a_partial_clip(tmp_path):
    store = FileArtifactStore(tmp_path)
    gate = valid_gate()
    RolloutController().execute(
        scenario(), StatefulFakePolicy(), FakeWorld(fail_call=2), FakeCompiler(), None, store, rollout_id="partial"
    )
    recovered = RolloutController().execute(
        scenario(), StatefulFakePolicy(), FakeWorld(), FakeCompiler(), FakeJudge(), store,
        rollout_id="partial", resume=True, validity_gate=gate,
    )
    assert recovered.status == "completed"
    assert recovered.validity == "unknown"
    assert gate.calls == []  # the gate was not handed a truncated frame stream
    assert recovered.binary_success is None


def test_resume_refuses_policy_without_restorable_state(tmp_path):
    store = FileArtifactStore(tmp_path)
    first = RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(fail_call=2), FakeCompiler(), None, store, rollout_id="cannot-resume"
    )
    assert first.status == "world_unavailable"
    with pytest.raises(ResumeUnsupportedError, match="cannot restore"):
        RolloutController().execute(
            scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), None, store, rollout_id="cannot-resume", resume=True
        )


def test_qualification_requires_explicit_passed_evidence_and_native_feedback(tmp_path):
    certified = FakeWorld(profile=certified_profile())
    with pytest.raises(ValueError, match="requires explicit passed gate evidence"):
        RolloutController(mode="qualification", feedback_mode=FeedbackMode.NATIVE_FEEDBACK).execute(
            scenario(), FakePolicy(), certified, FakeCompiler(), None, FileArtifactStore(tmp_path)
        )
    with pytest.raises(ValueError, match="gate names alone are not evidence"):
        RolloutController(
            mode="qualification",
            feedback_mode=FeedbackMode.NATIVE_FEEDBACK,
            qualification_evidence=QualificationEvidence(("A", "B", "C", "D"), "sha256:frozen", ("https://evidence.test",)),
        ).execute(scenario(), FakePolicy(), certified, FakeCompiler(), None, FileArtifactStore(tmp_path))


# --- Task 2: the fail-open validity default -------------------------------


def test_absent_validity_gate_is_unknown_and_loud_never_valid(tmp_path):
    judge = FakeJudge()
    report = RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), judge, FileArtifactStore(tmp_path)
    )
    assert report.status == "completed"
    assert report.validity == "unknown"
    assert report.validity_gate_configured is False
    assert report.validity_reason_codes == (VALIDITY_GATE_NOT_CONFIGURED,)
    assert report.validity_report["reason_codes"] == [VALIDITY_GATE_NOT_CONFIGURED]
    assert report.missing_reason == "validity_unknown"
    assert report.binary_success is None and report.progress_score is None
    # Stage A did not pass, so Stage B was never asked.
    assert judge.inputs == []


def test_a_judge_without_a_validate_method_no_longer_marks_episodes_valid(tmp_path):
    """The exact regression: QwenRubricJudge has no ``validate`` method."""

    class JudgeWithoutValidate:
        def __init__(self):
            self.inputs = []

        def evaluate(self, request):
            self.inputs.append(request)
            return {"binary_success": True, "progress_score": 5}

    judge = JudgeWithoutValidate()
    report = RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), judge, FileArtifactStore(tmp_path)
    )
    assert not hasattr(judge, "validate")
    assert report.validity == "unknown"
    assert report.binary_success is None
    assert judge.inputs == []


def test_raising_validity_gate_is_unknown_with_an_error_reason(tmp_path):
    gate = StubGate(error=RuntimeError("flow backend exploded"))
    report = RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), FakeJudge(), FileArtifactStore(tmp_path), validity_gate=gate
    )
    assert report.validity == "unknown"
    assert report.validity_gate_configured is True
    assert report.validity_reason_codes == (VALIDITY_GATE_ERROR,)
    assert report.validity_report["error"]["type"] == "RuntimeError"
    assert report.missing_reason == "validity_unknown"


@pytest.mark.parametrize("outcome", ["probably_ok", None, 17, {"validity": "great"}])
def test_off_contract_validity_gate_is_unknown_not_valid(tmp_path, outcome):
    report = RolloutController().execute(
        scenario(),
        FakePolicy(),
        FakeWorld(),
        FakeCompiler(),
        FakeJudge(),
        FileArtifactStore(tmp_path),
        validity_gate=StubGate(outcome),
    )
    assert report.validity == "unknown"
    assert report.validity_reason_codes == (VALIDITY_GATE_CONTRACT_VIOLATION,)


def test_invalid_validity_keeps_the_episode_but_skips_the_judge(tmp_path):
    judge = FakeJudge()
    gate = StubGate({"validity": "invalid", "reason_codes": ("constant_frame", "decode_failure")})
    report = RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), judge, FileArtifactStore(tmp_path), validity_gate=gate
    )
    assert report.validity == "invalid"
    assert report.validity_reason_codes == ("constant_frame", "decode_failure")
    assert report.missing_reason == "validity_invalid"
    assert report.binary_success is None
    assert report.executed_actions == 70  # retained in coverage, not deleted
    assert judge.inputs == []


def test_validity_gate_receives_every_frame_action_state_and_the_conditioning_frame(tmp_path):
    gate = valid_gate()
    RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), None, FileArtifactStore(tmp_path), validity_gate=gate
    )
    call = gate.calls[0]
    assert len(call["frames"]) == 71
    assert len(call["actions"]) == 70
    assert len(call["nominal_timestamps"]) == 71
    assert len(call["states"]) == 71
    assert call["conditioning_frame"] == "initial:open_drawer"
    assert call["frames"][0] == "initial:open_drawer"


def test_validity_gate_with_a_bad_signature_raises_before_any_world_call(tmp_path):
    class WrongGate:
        def evaluate(self, payload):  # pragma: no cover - never called
            return "valid"

    with pytest.raises(JudgeWiringError, match="missing keyword parameters"):
        RolloutController().execute(
            scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), None, FileArtifactStore(tmp_path),
            validity_gate=WrongGate(),
        )


def test_a_data_dependent_gate_error_is_unknown_rather_than_a_wiring_error(tmp_path):
    """A runtime fault inside a correctly wired gate must not crash the run."""

    class FlakyGate:
        def evaluate(self, *, frames, actions, nominal_timestamps, conditioning_frame, states, episode_id):
            raise TypeError("unsupported operand type(s) inside the flow estimator")

    report = RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), FakeJudge(), FileArtifactStore(tmp_path),
        validity_gate=FlakyGate(),
    )
    assert report.validity == "unknown"
    assert report.validity_reason_codes == (VALIDITY_GATE_ERROR,)
    assert report.validity_report["error"]["type"] == "TypeError"


def test_a_callable_gate_and_a_kwargs_gate_both_satisfy_the_contract(tmp_path):
    def functional_gate(**kwargs):
        return {"validity": "valid", "reason_codes": ()}

    report = RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), None, FileArtifactStore(tmp_path),
        validity_gate=functional_gate,
    )
    assert report.validity == "valid"
    assert report.validity_gate_configured is True


def test_real_stage_a_gate_catches_all_black_generated_frames_end_to_end(tmp_path):
    parameters = StageAParameters(
        frame_height=64, frame_width=64, flow_downsample_factor=1, grid_rows=8, grid_cols=8, block_search_radius=2
    )
    gate = StageAValidityGate(parameters, calibration_class=CALIBRATION_DEVELOPMENT)
    judge = FakeJudge()
    world = NumpyWorld(constant=True)
    start = np.zeros((64, 64, 3), dtype=np.uint8)
    base = scenario()
    black_start = Scenario(
        initial_rgb=start,
        state=base.state,
        task=base.task,
        horizon_actions=base.horizon_actions,
        hashes=base.hashes,
        seed=base.seed,
        reference_images=base.reference_images,
    )
    report = RolloutController().execute(
        black_start, FakePolicy(), world, FakeCompiler(), judge, FileArtifactStore(tmp_path), validity_gate=gate
    )
    assert report.validity == "invalid"
    assert "constant_frame" in report.validity_reason_codes
    assert report.validity_report["parameters_hash"] == gate.parameters_hash
    assert report.validity_report["calibration_reference_hash"] is None
    assert report.validity_report["calibration_class"] == CALIBRATION_DEVELOPMENT
    assert judge.inputs == []


def test_real_stage_a_gate_without_a_reference_is_unknown_not_valid(tmp_path):
    parameters = StageAParameters(
        frame_height=64, frame_width=64, flow_downsample_factor=1, grid_rows=8, grid_cols=8, block_search_radius=2
    )
    gate = StageAValidityGate(parameters)
    base = scenario()
    textured = Scenario(
        initial_rgb=np.random.default_rng(1).integers(0, 255, size=(64, 64, 3), dtype=np.uint8),
        state=base.state,
        task=base.task,
        horizon_actions=base.horizon_actions,
        hashes=base.hashes,
        seed=base.seed,
        reference_images=base.reference_images,
    )
    report = RolloutController().execute(
        textured, FakePolicy(), NumpyWorld(), FakeCompiler(), FakeJudge(), FileArtifactStore(tmp_path), validity_gate=gate
    )
    assert report.validity == "unknown"
    assert "no_motion_reference" in report.validity_reason_codes
    assert report.missing_reason == "validity_unknown"


# --- Task 3: connecting the real judge ------------------------------------


def test_judge_frame_sampling_is_one_authoritative_endpoint_inclusive_rule():
    assert judge_frame_indices(16) == tuple(range(16))
    indices = judge_frame_indices(71)
    assert len(indices) == JUDGE_FRAME_COUNT
    assert indices[0] == 0 and indices[-1] == 70
    assert list(indices) == sorted(set(indices))
    for count in (0, 1, 2, 15):
        with pytest.raises(ClipTooShortError, match="exactly 16 frames"):
            judge_frame_indices(count)


def test_derived_judge_seeds_are_deterministic_distinct_and_recorded():
    seeds = derive_judge_seeds("material")
    assert len(seeds) == 5 and len(set(seeds)) == 5
    assert all(isinstance(seed, int) and 0 <= seed < 2 ** 63 for seed in seeds)
    assert seeds == derive_judge_seeds("material")
    assert seeds != derive_judge_seeds("other-material")


def test_real_judge_request_is_built_and_scored_through_the_controller(tmp_path):
    judge = ProtocolJudge()
    report = RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), judge, FileArtifactStore(tmp_path), validity_gate=valid_gate()
    )
    assert report.judge_protocol == JUDGE_PROTOCOL_REQUEST
    assert report.binary_success is True and report.progress_score == 5
    assert report.missing_reason is None

    request, seeds = judge.calls[0]
    assert isinstance(request, JudgeRequest)
    request.validate()
    assert len(request.frames) == JUDGE_FRAME_COUNT
    assert len(request.frame_timestamps) == JUDGE_FRAME_COUNT
    assert request.frame_timestamps[0] == 0.0
    assert request.frame_timestamps[-1] == pytest.approx(70 / 5.0)
    assert request.task_id == "open_drawer"
    assert request.task_instruction is None and request.task_rubric is None
    assert request.reference_images[0].sha256 == "a" * 64
    assert request.frames[0] == "initial:open_drawer"
    assert len(seeds) == 5 and len(set(seeds)) == 5
    assert report.judge_seeds == seeds
    assert report.judge_frame_indices[0] == 0 and report.judge_frame_indices[-1] == 70
    assert report.judge_rubric_hash == BENCHMARK_TASK_REGISTRY.get("open_drawer").rubric_hash
    assert report.judge_task_registry_hash == TASK_REGISTRY_HASH
    assert len(report.judge_frame_hashes) == JUDGE_FRAME_COUNT


def test_no_action_text_or_policy_identity_reaches_the_judge_request(tmp_path):
    judge = ProtocolJudge()
    RolloutController().execute(
        scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), judge, FileArtifactStore(tmp_path), validity_gate=valid_gate()
    )
    request, _seeds = judge.calls[0]
    fields = set(vars(request))
    assert not fields & {"actions", "compiled_actions", "native_actions", "policy", "policy_name", "validity"}


def test_missing_provenance_backed_reference_images_make_the_episode_unevaluable(tmp_path):
    judge = ProtocolJudge()
    report = RolloutController().execute(
        scenario(reference_images=()),
        FakePolicy(),
        FakeWorld(),
        FakeCompiler(),
        judge,
        FileArtifactStore(tmp_path),
        validity_gate=valid_gate(),
    )
    assert report.missing_reason == "judge_reference_images_unavailable"
    assert report.binary_success is None
    assert judge.calls == []


def test_reference_image_provenance_requires_a_real_digest():
    with pytest.raises(ValueError, match="SHA-256"):
        ReferenceImageProvenance(image="x", source_uri="fixture://x", sha256="not-a-digest")
    with pytest.raises(ValueError, match="source_uri"):
        ReferenceImageProvenance(image="x", source_uri="", sha256="a" * 64)
    prefixed = ReferenceImageProvenance(image="x", source_uri="fixture://x", sha256="sha256:" + "A" * 64)
    assert prefixed.sha256 == "a" * 64


def test_a_judge_signature_mismatch_raises_instead_of_recording_judge_failure(tmp_path):
    with pytest.raises(JudgeWiringError, match="does not accept evaluate"):
        RolloutController().execute(
            scenario(),
            FakePolicy(),
            FakeWorld(),
            FakeCompiler(),
            BrokenProtocolJudge(),
            FileArtifactStore(tmp_path),
            validity_gate=valid_gate(),
        )


def test_a_judge_rejecting_our_request_schema_raises_instead_of_hiding(tmp_path):
    judge = ProtocolJudge(error=JudgeInputError("frames must be 16"))
    with pytest.raises(JudgeWiringError, match="rejected the controller's request or seeds"):
        RolloutController().execute(
            scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), judge, FileArtifactStore(tmp_path),
            validity_gate=valid_gate(),
        )


@pytest.mark.parametrize(
    "error,expected",
    [
        (JudgeRefusalError("the model declined"), "judge_refusal"),
        (JudgeSchemaError("not JSON"), "judge_schema_failure"),
        (JudgeLoadError("no local weights"), "judge_transport_failure:JudgeLoadError"),
        (BackendUnavailableError("offline"), "judge_transport_failure:BackendUnavailableError"),
        (TimeoutError("deadline"), "judge_transport_failure:TimeoutError"),
    ],
)
def test_judge_refusal_transport_and_schema_failures_are_distinguishable(tmp_path, error, expected):
    report = RolloutController().execute(
        scenario(),
        FakePolicy(),
        FakeWorld(),
        FakeCompiler(),
        ProtocolJudge(error=error),
        FileArtifactStore(tmp_path),
        validity_gate=valid_gate(),
    )
    assert report.missing_reason == expected
    assert report.binary_success is None
    assert report.validity == "valid"


def test_an_unevaluable_judge_report_keeps_its_own_missing_reason(tmp_path):
    unevaluable = JudgeReport(
        binary_success=None,
        progress=None,
        judge_status="unknown",
        missing_reason="insufficient_decisive_quorum",
        agreeing_samples=2,
        sample_reports=(),
        wall_seconds=0.2,
        gpu_peak_memory_bytes=None,
    )
    report = RolloutController().execute(
        scenario(),
        FakePolicy(),
        FakeWorld(),
        FakeCompiler(),
        ProtocolJudge(report=unevaluable),
        FileArtifactStore(tmp_path),
        validity_gate=valid_gate(),
    )
    assert report.binary_success is None
    assert report.missing_reason == "insufficient_decisive_quorum"


def test_a_judge_returning_an_out_of_range_progress_raises(tmp_path):
    broken = JudgeReport(
        binary_success=True,
        progress=9,
        judge_status="evaluable",
        missing_reason=None,
        agreeing_samples=5,
        sample_reports=(),
        wall_seconds=0.2,
        gpu_peak_memory_bytes=None,
    )
    with pytest.raises(JudgeWiringError, match="progress must be an integer 0..5"):
        RolloutController().execute(
            scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), ProtocolJudge(report=broken),
            FileArtifactStore(tmp_path), validity_gate=valid_gate(),
        )


def test_a_judge_with_no_evaluate_entry_point_raises(tmp_path):
    class NotAJudge:
        pass

    with pytest.raises(JudgeWiringError, match="no callable evaluate"):
        RolloutController().execute(
            scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), NotAJudge(), FileArtifactStore(tmp_path),
            validity_gate=valid_gate(),
        )


# --- Task 4: certified action lengths and terminal padding -----------------


@pytest.mark.parametrize(
    "task,horizon",
    [("open_drawer", 70), ("to_basket", 100), ("fold_cloth", 80)],
)
def test_cosmos_sixteen_only_blocks_the_non_multiple_horizons_by_name(tmp_path, task, horizon):
    world = FakeWorld(lengths=(16,))
    if horizon % 16 == 0:
        report = RolloutController().execute(
            scenario(task), SixteenStepPolicy(), world, FakeCompiler(), None, FileArtifactStore(tmp_path)
        )
        assert report.executed_actions == horizon
        assert report.protocol_identity == PROTOCOL_EXACT_TERMINAL_HORIZON
        return
    with pytest.raises(UnsupportedActionLengthError) as raised:
        RolloutController().execute(
            scenario(task), FakePolicy(), world, FakeCompiler(), None, FileArtifactStore(tmp_path)
        )
    message = str(raised.value)
    assert "cannot execute the exact %d-action terminal horizon" % horizon in message
    assert "uncertified_default" in message
    assert "Certifying any one of these action lengths would make it exact" in message
    assert world.requests == []


def test_seventy_is_unreachable_from_any_multiple_of_four_and_the_error_names_the_fix(tmp_path):
    profile = WorldActionProfile("cosmos", "bridge_orig_lerobot", (4, 8, 16))
    assert profile.represents_exactly(70) is False  # 70 mod 4 == 2
    assert profile.represents_exactly(80) is True
    unblocking = profile.unblocking_lengths(70)
    assert 2 in unblocking and 6 in unblocking
    assert all(length % 4 != 0 for length in unblocking)
    with pytest.raises(UnsupportedActionLengthError, match="2"):
        RolloutController().execute(
            scenario(), FakePolicy(), FakeWorld(profile=profile), FakeCompiler(), None, FileArtifactStore(tmp_path)
        )


def test_uncertified_action_lengths_are_refused_for_qualification(tmp_path):
    evidence = QualificationEvidence(("A", "B", "C", "D"), "sha256:frozen", ("https://evidence.test",))
    with pytest.raises(ValueError, match="requires Gate-A certified action lengths"):
        RolloutController(
            mode="qualification", feedback_mode=FeedbackMode.NATIVE_FEEDBACK, qualification_evidence=evidence
        ).execute(scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), None, FileArtifactStore(tmp_path))


def test_certified_profile_records_its_certification_source_hash(tmp_path):
    world = FakeWorld(profile=certified_profile())
    report = RolloutController().execute(
        scenario(), FakePolicy(), world, FakeCompiler(), None, FileArtifactStore(tmp_path)
    )
    record = report.world_action_length_certification
    assert record["certification_class"] == GATE_A_CERTIFIED_CLASS
    assert record["certification_source_hash"] == "sha256:" + "1" * 64
    assert record["certification_id"] == "cert-937398"
    assert record["supported_action_lengths"] == [4, 2]


def test_a_certified_profile_must_name_its_artifact_hash():
    with pytest.raises(ValueError, match="must name its certification artifact hash"):
        WorldActionProfile(
            "cosmos", "bridge_orig_lerobot", (16,), certification_class=GATE_A_CERTIFIED_CLASS
        )


def test_terminal_padding_needs_a_certificate_and_carries_a_distinct_protocol(tmp_path):
    # 70 = 4 * 17 + 2, so the terminal segment needs a consumed prefix of 2.
    without = WorldActionProfile("fake-world", "bridge_orig_lerobot", (4,))
    with pytest.raises(UnsupportedActionLengthError, match="TerminalPaddingCertificate"):
        RolloutController().execute(
            scenario(), FakePolicy(), FakeWorld(profile=without), FakeCompiler(), None, FileArtifactStore(tmp_path)
        )

    certified = certified_profile(lengths=(4,), padding=padding_certificate())
    world = FakeWorld(profile=certified)
    report = RolloutController().execute(
        scenario(), FakePolicy(), world, FakeCompiler(), FakeJudge(), FileArtifactStore(tmp_path),
        validity_gate=valid_gate(),
    )
    assert report.executed_actions == 70
    assert report.protocol_identity == PROTOCOL_CERTIFIED_TERMINAL_PADDING
    assert len(report.padding_events) == 1
    event = report.padding_events[0]
    assert event["requested_action_length"] == 4
    assert event["executed_prefix"] == 2
    assert event["padding_action_count"] == 2
    assert len(event["discarded_post_horizon_frame_hashes"]) == 2
    assert event["certificate_hash"].startswith("sha256:")
    # Every request the backend saw was a supported length.
    assert {len(request.compiled_actions) for request in world.requests} == {4}
    # The executed control record holds only the 70 executed actions.
    assert len(report.native_actions) == len(report.compiled_actions) == 70
    assert len(report.frame_hashes) == 71
    assert report.segments[-1].action_count == 2
    assert report.segments[-1].padding["executed_prefix"] == 2
    assert report.segments[0].padding is None


def test_an_uncertified_or_failing_padding_certificate_is_refused(tmp_path):
    unmeasured = padding_certificate(residual=9.0)
    assert unmeasured.errors()
    profile = certified_profile(lengths=(4,), padding=unmeasured)
    with pytest.raises(UnsupportedActionLengthError, match="cannot authorise padding"):
        RolloutController().execute(
            scenario(), FakePolicy(), FakeWorld(profile=profile), FakeCompiler(), None, FileArtifactStore(tmp_path)
        )


def test_a_padding_certificate_for_the_wrong_prefix_does_not_unblock(tmp_path):
    profile = certified_profile(lengths=(4,), padding=padding_certificate(prefixes=(3,)))
    with pytest.raises(UnsupportedActionLengthError, match="do not leave a reachable remainder"):
        RolloutController().execute(
            scenario(), FakePolicy(), FakeWorld(profile=profile), FakeCompiler(), None, FileArtifactStore(tmp_path)
        )


def test_padding_is_never_applied_to_an_ordinary_mid_episode_prefix(tmp_path):
    """A certificate for the terminal prefix does not license padding elsewhere."""

    profile = certified_profile(lengths=(4,), padding=padding_certificate(prefixes=(2,)))
    policy = FakePolicy(terminal_prefixes=(2, 3))
    world = FakeWorld(profile=profile)

    class ThreeStepPolicy(FakePolicy):
        def plan(self, observation, remaining):
            self.observations.append(observation)
            return PolicyControlPlan(
                actions=((1, 0, 0, 0, 0, 0, 0),) * 3, verified_prefix=3, history_length=1
            )

    with pytest.raises(UncertifiedPaddingError, match="does not cover a consumed prefix of 3"):
        RolloutController().execute(
            scenario(), ThreeStepPolicy(), world, FakeCompiler(), None, FileArtifactStore(tmp_path)
        )
    assert policy.observations == []
    assert world.requests == []


def test_padding_uses_only_the_certified_declared_filler_policy(tmp_path):
    certificate = padding_certificate()
    profile = certified_profile(lengths=(4,), padding=certificate)
    world = FakeWorld(profile=profile)
    report = RolloutController().execute(
        scenario(), FakePolicy(), world, FakeCompiler(), None, FileArtifactStore(tmp_path)
    )
    event = report.padding_events[0]
    assert event["padding_action_policy"] == "zero_delta_hold_gripper"
    # The declared filler is a zero delta holding the last commanded gripper.
    assert event["padding_actions"] == [[0.0] * 7, [0.0] * 7]
    terminal_request = world.requests[-1]
    assert [row[0] for row in terminal_request.compiled_actions] == [1.0, 1.0, 0.0, 0.0]
