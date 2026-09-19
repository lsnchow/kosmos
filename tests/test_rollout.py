"""Controller contracts with fake adapters; no model assets or physics claims."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from plumb.adapters.bridge import BridgeState
from plumb.adapters.contracts import CapabilityStatus, FeedbackMode, PolicyContract, ServerTiming, WorldResult
from plumb.adapters.worlds import BackendUnavailableError
from plumb.rollout import (
    FileArtifactStore,
    PolicyControlPlan,
    QualificationEvidence,
    ResumeUnsupportedError,
    RolloutController,
    Scenario,
    UnsupportedActionLengthError,
    WorldActionProfile,
)


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


def scenario(task="open_drawer"):
    horizon = {"open_drawer": 70, "to_basket": 100, "fold_cloth": 80}[task]
    return Scenario(
        initial_rgb="initial:%s" % task,
        state=BridgeState.from_values((0, 0, 0, 0, 0, 0, 0, 0), source="measured"),
        task=task,
        horizon_actions=horizon,
        hashes={"image": "sha256:initial", "state": "sha256:state"},
        seed=13,
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

    def snapshot_state(self):
        return {"counter": self.counter}

    def restore_state(self, value):
        self.counter = value["counter"]
        self.restored = True


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
    def __init__(self, lengths=(4, 2), fail_call=None):
        self.rollout_profile = WorldActionProfile("fake-world", "bridge_orig_lerobot", tuple(lengths))
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
            profile_id="fake-world",
            frames=frames,
            nominal_frame_timestamps=tuple(index / 5.0 for index in range(len(frames))),
            conditioning_frame_included=True,
            timing=ServerTiming(backend_calls=1, wall_seconds=0.01, cold_start=False),
        )


class FakeJudge:
    def __init__(self):
        self.inputs = []

    def validate(self, request):
        return "valid"

    def evaluate(self, request):
        self.inputs.append(request)
        return {"binary_success": False, "progress_score": 3}


@pytest.mark.parametrize("task", ["open_drawer", "to_basket", "fold_cloth"])
def test_exact_horizons_use_verified_prefixes_and_fresh_generated_feedback(tmp_path, task):
    policy = FakePolicy()
    world = FakeWorld()
    judge = FakeJudge()
    report = RolloutController().execute(scenario(task), policy, world, FakeCompiler(), judge, FileArtifactStore(tmp_path))

    assert report.status == "completed"
    assert report.executed_actions == report.horizon_actions
    assert sum(segment.action_count for segment in report.segments) == report.horizon_actions
    assert all(segment.action_count in (4, 2) for segment in report.segments)
    assert report.frame_hashes and len(report.frame_hashes) == report.horizon_actions + 1
    assert report.feedback_mode == "forecast_state"
    assert report.physical_state_measured is False
    assert report.binary_success is False and report.progress_score == 3
    assert all(len(request.compiled_actions) in (4, 2) for request in world.requests)
    # Every feedback request uses the final newly generated frame from the
    # prior segment, never the condition frame echoed by WorldResult.
    assert [request.conditioning_image for request in world.requests][0].startswith("initial:")
    assert all(
        world.requests[index].conditioning_image == "generated:%d:%d" % (index, len(world.requests[index - 1].compiled_actions) - 1)
        for index in range(1, len(world.requests))
    )
    assert all(observation.proprio is not None for observation in policy.observations)
    assert len(judge.inputs[0].frames) == 16


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
    policy = FakePolicy()
    first_world = FakeWorld(fail_call=2)
    first = RolloutController().execute(scenario(), policy, first_world, FakeCompiler(), None, store, rollout_id="resume-me")
    assert first.status == "world_unavailable" and first.executed_actions == 4

    recovered_policy = FakePolicy()
    recovered = RolloutController().execute(
        scenario(), recovered_policy, FakeWorld(), FakeCompiler(), None, store, rollout_id="resume-me", resume=True
    )
    assert recovered.status == "completed"
    assert recovered.executed_actions == 70
    assert len(recovered.native_actions) == len(recovered.compiled_actions) == 70
    assert len(recovered.frame_hashes) == len(recovered.nominal_timestamps) == 71
    assert recovered_policy.restored is True
    assert recovered_policy.observations[0].image_history == ("generated:1:3",)


def test_resume_refuses_policy_without_restorable_state(tmp_path):
    class StatelessPolicy(FakePolicy):
        snapshot_state = None
        restore_state = None

    store = FileArtifactStore(tmp_path)
    first = RolloutController().execute(
        scenario(), StatelessPolicy(), FakeWorld(fail_call=2), FakeCompiler(), None, store, rollout_id="cannot-resume"
    )
    assert first.status == "world_unavailable"
    with pytest.raises(ResumeUnsupportedError, match="cannot restore"):
        RolloutController().execute(
            scenario(), StatelessPolicy(), FakeWorld(), FakeCompiler(), None, store, rollout_id="cannot-resume", resume=True
        )


def test_qualification_requires_explicit_passed_evidence_and_native_feedback(tmp_path):
    with pytest.raises(ValueError, match="requires explicit passed gate evidence"):
        RolloutController(mode="qualification", feedback_mode=FeedbackMode.NATIVE_FEEDBACK).execute(
            scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), None, FileArtifactStore(tmp_path)
        )
    with pytest.raises(ValueError, match="refuses compiler feedback_mode=forecast_state"):
        RolloutController(
            mode="qualification",
            feedback_mode=FeedbackMode.NATIVE_FEEDBACK,
            qualification_evidence=QualificationEvidence(("A", "B", "C", "D"), "sha256:frozen", ("https://evidence.test",)),
        ).execute(scenario(), FakePolicy(), FakeWorld(), FakeCompiler(), None, FileArtifactStore(tmp_path))
