"""Source-profile guards for AutoEval and production Octo conformance."""

from __future__ import annotations

import pytest

from plumb.adapters.contracts import PolicyObservation
from plumb.policies import octo_autoeval_conformance as conformance
from plumb.policies.contracts import ActionNormalizationType, PolicyActionNormalizer, PolicyActionNormalizerStatistics, PolicyContractError, PolicyExecutionMode
from plumb.policies.octo import (
    OCTO_SMALL_CHECKPOINT_STEP,
    OCTO_SMALL_MODEL_ID,
    OctoEnsembleConfig,
    OctoPolicyProfile,
    OctoSmallV1PolicyAdapter,
    _OctoRuntime,
)


class FakeOctoModel:
    def __init__(self):
        self.task_calls = []
        self.sample_calls = []

    def create_tasks(self, *, texts):
        self.task_calls.append(tuple(texts))
        return {"task": tuple(texts)}

    def sample_actions(self, observations, task, unnormalization_statistics=None, rng=None):
        self.sample_calls.append(
            {
                "observations": observations,
                "task": task,
                "unnormalization_statistics": unnormalization_statistics,
                "rng": rng,
            }
        )
        return [[[float(index)] * 7 for index in range(4)]]


def _adapter(model, *, folds):
    profile = OctoPolicyProfile(
        profile_id="octo-autoeval-conformance-fixture",
        local_model_path="/not/loaded",
        model_id=OCTO_SMALL_MODEL_ID,
        checkpoint_revision="a" * 40,
        checkpoint_step=OCTO_SMALL_CHECKPOINT_STEP,
    )
    runtime = _OctoRuntime(
        jax_version="0.4.20",
        array_factory=lambda value, dtype: {"dtype": dtype, "value": value},
        prng_key=lambda seed: ("prng", seed),
        fold_in=lambda key, step: folds.append((key, step)),
        model_loader=lambda path, step: pytest.fail("fixture must not load a model"),
    )
    normalizer = PolicyActionNormalizer(
        PolicyActionNormalizerStatistics(
            normalizer_revision="octo-autoeval-conformance-stats",
            normalization_type=ActionNormalizationType.NORMAL,
            mask=(True,) * 7,
            mean=(0.0,) * 7,
            std=(1.0,) * 7,
        )
    )
    return OctoSmallV1PolicyAdapter(
        profile,
        normalizer=normalizer,
        ensemble=OctoEnsembleConfig(config_revision="octo-autoeval-source-exp0", exponential_weight=0.0, horizon=4),
        execution_mode=PolicyExecutionMode.FIXTURE_CERTIFICATION,
        runtime_factory=lambda: runtime,
        model_factory=lambda profile, runtime: model,
    )


def test_production_adapter_uses_autoeval_static_rng_and_task_cache_until_reset():
    model = FakeOctoModel()
    folds = []
    adapter = _adapter(model, folds=folds)
    adapter.reset(91)

    adapter.propose(PolicyObservation(image_history=(b"first", b"first"), prompt="Close the drawer"))
    adapter.propose(PolicyObservation(image_history=(b"first", b"second"), prompt="Close the drawer"))

    assert model.task_calls == [("Close the drawer",)]
    assert [call["rng"] for call in model.sample_calls] == [("prng", 0), ("prng", 0)]
    assert folds == []
    assert all(call["unnormalization_statistics"] is None for call in model.sample_calls)
    with pytest.raises(PolicyContractError, match="call reset"):
        adapter.propose(PolicyObservation(image_history=(b"second", b"third"), prompt="Open the drawer"))

    adapter.reset(12)
    adapter.propose(PolicyObservation(image_history=(b"third", b"third"), prompt="Open the drawer"))
    assert model.task_calls == [("Close the drawer",), ("Open the drawer",)]


def test_snapshot_restore_rebuilds_the_bound_task_and_refuses_a_task_swap():
    original = FakeOctoModel()
    adapter = _adapter(original, folds=[])
    adapter.reset(4)
    adapter.propose(PolicyObservation(image_history=(b"first", b"first"), prompt="Close the drawer"))
    snapshot = adapter.snapshot_state()
    assert snapshot["extra"]["task_instruction"] == "Close the drawer"

    resumed_model = FakeOctoModel()
    resumed = _adapter(resumed_model, folds=[])
    resumed.restore_state(snapshot)
    resumed.propose(PolicyObservation(image_history=(b"first", b"second"), prompt="Close the drawer"))
    assert resumed_model.task_calls == [("Close the drawer",)]
    with pytest.raises(PolicyContractError, match="call reset"):
        resumed.propose(PolicyObservation(image_history=(b"second", b"third"), prompt="Open the drawer"))

    legacy = {**snapshot, "extra": {"ensembler": snapshot["extra"]["ensembler"]}}
    with pytest.raises(PolicyContractError, match="must bind task_instruction"):
        _adapter(FakeOctoModel(), folds=[]).restore_state(legacy)


def test_source_fragment_harness_recognizes_pinned_autoeval_contract():
    findings = conformance._required_source_fragments(
        policy_text=(
            'assert all(key in obs_dict.keys() for key in ["image_primary", "proprio"])\n'
            "if self.task is None:\n"
            "self.task = self.agent.create_tasks(texts=[language_instruction])\n"
            'actions = self.agent.sample_actions(unnormalization_statistics=self.agent.dataset_statistics["bridge_dataset"]["action"], rng=jax.random.PRNGKey(0))'
        ),
        run_eval_text=(
            "env = ManipulatorEnv(state_encoding=StateEncoding.POS_EULER)\n"
            "env = ConvertState2Proprio(env)\n"
            "env = HistoryWrapper(env, horizon=2)\n"
            "env = TemporalEnsembleWrapper(env, 4)"
        ),
        wrappers_text=(
            "self.history.extend([obs] * self.horizon)\n"
            "timestep_pad_mask[:pad_length] = 0\n"
            "def __init__(self, env: gym.Env, pred_horizon: int, exp_weight: int = 0):\n"
            "range(num_actions - 1, -1, -1)\n"
            "np.exp(-self.exp_weight * np.arange(num_actions))"
        ),
        model_text=(
            'timestep_pad_mask = observations["timestep_pad_mask"]\n'
            '(action * unnormalization_statistics["std"])\n'
            'jnp.ones_like(unnormalization_statistics["mean"], dtype=bool)'
        ),
    )

    assert all(findings.values())
    broken = conformance._required_source_fragments(policy_text="", run_eval_text="", wrappers_text="", model_text="")
    assert not any(broken.values())
