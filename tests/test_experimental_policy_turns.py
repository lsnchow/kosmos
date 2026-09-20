"""Offline contract tests for the unscored source-policy turn router.

These are adapter-bound fakes: no test pretends to be a model checkpoint or a
production worker.  The point is to make the controller's handling of the
three *reported source boundaries* strict and inspectable on a CPU host.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from plumb.adapters.contracts import CapabilityResult, CapabilityStatus
from plumb.policies.experimental import (
    EXPERIMENTAL_POLICY_TURN_SCHEMA,
    ExperimentalPolicyTurnRequest,
    ExperimentalPolicyTurnRouter,
    make_experimental_policy_router,
)
from plumb.policies.diagnostics.octo_v0_1 import (
    OCTO_SMALL_MODEL_REVISION,
    OCTO_V1_SOURCE_COMMIT,
    OctoSmallV1Policy,
    OctoV1PolicyProfile,
    _OctoRuntime,
)


def _ready() -> CapabilityResult:
    return CapabilityResult(
        status=CapabilityStatus.READY_UNQUALIFIED,
        reason="fixture source adapter only; no qualification asserted",
        source_verified=True,
        evidence_uris=("fixture://source",),
        details={"fixture": True},
    )


def _blocked(reason: str) -> CapabilityResult:
    return CapabilityResult(status=CapabilityStatus.BLOCKED, reason=reason, source_verified=True)


@dataclass
class OpenVLABoundary:
    action: tuple[float, ...]
    capability_result: CapabilityResult
    calls: list[Any]

    def capability(self):
        return self.capability_result

    def predict_with_report(self, observation):
        self.calls.append(observation)
        return SimpleNamespace(
            action=self.action,
            backend_calls=1,
            wall_seconds=0.125,
            gpu_peak_memory_bytes=None,
            unnorm_key="bridge_orig",
            source_image_timestamp=observation.timestamp,
        )


@dataclass
class MiniVLABoundary:
    action: tuple[float, ...]
    capability_result: CapabilityResult
    calls: list[Any]

    def capability(self):
        return self.capability_result

    def predict_with_report(self, observation):
        self.calls.append(observation)
        return SimpleNamespace(
            action=self.action,
            backend_calls=1,
            wall_seconds=0.25,
            unnorm_key="bridge_dataset",
            source_image_timestamp=observation.timestamp,
            # The source internals have H=8 / seven future elements but the
            # actual exposed source return is ret_action[:, 0], a single row.
            vq_input_horizon=8,
            vq_future_action_horizon=7,
            returned_action_count=1,
        )


@dataclass
class OctoBoundary:
    proposal: tuple[tuple[float, ...], ...]
    executed: tuple[float, ...]
    capability_result: CapabilityResult
    calls: list[Any]
    restored: list[dict[str, Any]]

    def capability(self):
        return self.capability_result

    def restore_source_state(self, payload):
        self.restored.append(payload)

    def predict_with_report(self, observation):
        self.calls.append(observation)
        return SimpleNamespace(
            action=self.executed,
            proposal=self.proposal,
            backend_calls=1,
            wall_seconds=0.5,
            observation_count=len(self.restored) + 1,
            rng_seed=0,
            temporal_ensembling=True,
            action_exp_weight=0.0,
            gripper_transformation="source normalises all 7 dimensions",
            normalization="source mean/std",
            source_image_timestamp=observation.timestamp,
        )


def _row(value: float) -> tuple[float, ...]:
    return (value,) * 7


def _ids(*, cell_id="cell-a", revision="policy-rev-a", task_id="close-drawer", observation_id="obs-0", **extra):
    result = {
        "cell_id": cell_id,
        "policy_revision": revision,
        "task_id": task_id,
        "observation_id": observation_id,
        "state_kind": "source_measured",
        "state_lineage_id": "start-state-v1",
        "timestamp": 1.0,
    }
    result.update(extra)
    return result


def _router(*, open_factory, mini_factory, octo_factory):
    return ExperimentalPolicyTurnRouter(
        {"openvla": open_factory, "minivla": mini_factory, "octo-small": octo_factory}
    )


def test_openvla_is_one_fresh_rgb_to_one_native_action_and_never_an_action_chunk():
    created = []

    def build_open():
        adapter = OpenVLABoundary(_row(1.0), _ready(), [])
        created.append(adapter)
        return adapter

    router = _router(
        open_factory=build_open,
        mini_factory=lambda: MiniVLABoundary(_row(2.0), _ready(), []),
        octo_factory=lambda: OctoBoundary((_row(3.0),) * 4, _row(4.0), _ready(), [], []),
    )
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    result = router.predict(
        ExperimentalPolicyTurnRequest(
            policy_id="OpenVLA",
            prompt="Close the drawer",
            image_history=(image,),
            # The world/controller retains this state lineage, but the source
            # OpenVLA call must not receive it as policy proprioception.
            proprio=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            identifiers=_ids(),
        )
    )

    assert result.status == "completed"
    assert result.raw_proposal_rows == (_row(1.0),)
    assert result.source_executed_rows == (_row(1.0),)
    assert len(created) == 1 and created[0].calls[0].image_history == (image,)
    assert created[0].calls[0].proprio is None
    assert result.provenance["source_metadata"]["unnorm_key"] == "bridge_orig"
    assert result.provenance["outcome"] == "not_scored"


def test_minivla_uses_exposed_ret_action_zero_once_not_an_internal_vq_future_chunk():
    created = []

    def build_mini():
        adapter = MiniVLABoundary(_row(7.0), _ready(), [])
        created.append(adapter)
        return adapter

    router = _router(
        open_factory=lambda: OpenVLABoundary(_row(1.0), _ready(), []),
        mini_factory=build_mini,
        octo_factory=lambda: OctoBoundary((_row(3.0),) * 4, _row(4.0), _ready(), [], []),
    )

    result = router.predict(
        ExperimentalPolicyTurnRequest(
            policy_id="minivla",
            prompt="Close the drawer",
            image_history=(np.zeros((2, 2, 3), dtype=np.uint8),),
            identifiers=_ids(),
        )
    )

    assert result.status == "completed"
    assert result.raw_proposal_rows == (_row(7.0),)
    assert result.source_executed_rows == (_row(7.0),)
    assert result.provenance["source_metadata"]["vq_future_action_horizon"] == 7
    assert result.provenance["source_metadata"]["returned_action_count"] == 1
    assert len(created[0].calls) == 1


def test_minivla_license_or_asset_gate_blocks_before_a_source_call_and_never_mints_permission():
    created = []

    def build_mini():
        adapter = MiniVLABoundary(_row(7.0), _blocked("VQ license is unresolved; no source call allowed"), [])
        created.append(adapter)
        return adapter

    router = _router(
        open_factory=lambda: OpenVLABoundary(_row(1.0), _ready(), []),
        mini_factory=build_mini,
        octo_factory=lambda: OctoBoundary((_row(3.0),) * 4, _row(4.0), _ready(), [], []),
    )
    result = router.predict(
        ExperimentalPolicyTurnRequest(
            policy_id="MiniVLA",
            prompt="Close the drawer",
            image_history=(np.zeros((2, 2, 3), dtype=np.uint8),),
            identifiers=_ids(),
        )
    )

    assert result.status == "blocked"
    assert result.raw_proposal_rows == ()
    assert result.source_executed_rows == ()
    assert created[0].calls == []
    assert "license is unresolved" in result.reason


def test_octo_reports_four_raw_rows_but_executes_only_the_source_temporal_ensemble_row_and_serializes_state():
    created = []
    proposal = (_row(10.0), _row(20.0), _row(30.0), _row(40.0))

    def build_octo():
        adapter = OctoBoundary(proposal, _row(99.0), _ready(), [], [])
        created.append(adapter)
        return adapter

    router = _router(
        open_factory=lambda: OpenVLABoundary(_row(1.0), _ready(), []),
        mini_factory=lambda: MiniVLABoundary(_row(2.0), _ready(), []),
        octo_factory=build_octo,
    )
    first = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    identifiers = _ids(
        proprio_lineage_id="bridge-convert-state-0",
        proprio_converter_revision="manipulator-gym-commit-a",
    )

    result = router.predict(
        ExperimentalPolicyTurnRequest(
            policy_id="octo-small",
            prompt="Close the drawer",
            image_history=(first,),
            proprio=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            policy_seed=101,
            identifiers=identifiers,
        )
    )

    assert result.status == "completed"
    assert result.raw_proposal_rows == proposal
    assert result.source_executed_rows == (_row(99.0),)
    assert result.source_executed_rows[0] != proposal[0]
    assert created[0].restored == []
    assert result.snapshot is not None
    assert result.snapshot["schema"] == EXPERIMENTAL_POLICY_TURN_SCHEMA
    assert result.snapshot["source_state"]["observation_count"] == 1
    history_entry = result.snapshot["source_state"]["observation_history"][0]
    assert history_entry["state_kind"] == "source_measured"
    assert history_entry["rgb"]["encoding"] == "rgb_uint8_base64"
    # This is the contract sent through the byte stream; it contains no live
    # adapter/model/task object and is safe to journal/replay.
    assert json.loads(json.dumps(result.as_dict()))["snapshot"]["source_state"]["proposal_history"] == [
        [list(row) for row in proposal]
    ]


def test_octo_next_turn_restores_only_that_cells_real_history_and_raw_proposals_into_a_fresh_adapter():
    created = []

    def build_octo():
        index = len(created)
        adapter = OctoBoundary(
            (_row(1.0 + index), _row(2.0 + index), _row(3.0 + index), _row(4.0 + index)),
            _row(50.0 + index),
            _ready(),
            [],
            [],
        )
        created.append(adapter)
        return adapter

    router = _router(
        open_factory=lambda: OpenVLABoundary(_row(1.0), _ready(), []),
        mini_factory=lambda: MiniVLABoundary(_row(2.0), _ready(), []),
        octo_factory=build_octo,
    )
    base_ids = _ids(proprio_lineage_id="p0", proprio_converter_revision="convert-a")
    first_image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    first = router.predict(
        ExperimentalPolicyTurnRequest(
            policy_id="octo-small",
            prompt="Close the drawer",
            image_history=(first_image,),
            proprio=(1.0,) * 8,
            identifiers=base_ids,
        )
    )
    assert first.status == "completed"
    second_ids = _ids(
        observation_id="obs-1",
        state_kind="forecast_integrated",
        state_lineage_id="world-integrated-0",
        proprio_lineage_id="p1",
        proprio_converter_revision="convert-a",
    )
    second = router.predict(
        ExperimentalPolicyTurnRequest(
            policy_id="octo-small",
            prompt="Close the drawer",
            image_history=(np.full((2, 2, 3), 11, dtype=np.uint8),),
            proprio=(2.0,) * 8,
            prior_snapshot=first.snapshot,
            identifiers=second_ids,
        )
    )

    assert second.status == "completed"
    assert len(created) == 2 and created[0] is not created[1]
    restored = created[1].restored[0]
    assert restored["task_instruction"] == "Close the drawer"
    assert restored["observation_count"] == 1
    np.testing.assert_array_equal(restored["observation_history"][0]["decoded_rgb"], first_image)
    assert restored["observation_history"][0]["proprio"] == [1.0] * 8
    assert restored["proposal_history"] == [[list(row) for row in first.raw_proposal_rows]]
    assert second.snapshot["source_state"]["observation_count"] == 2
    assert [item["state_kind"] for item in second.snapshot["source_state"]["observation_history"]] == [
        "source_measured",
        "forecast_integrated",
    ]


def test_octo_rejects_missing_or_all_zero_proprio_and_rejects_a_snapshot_from_another_cell():
    created = []

    def build_octo():
        adapter = OctoBoundary((_row(1.0),) * 4, _row(8.0), _ready(), [], [])
        created.append(adapter)
        return adapter

    router = _router(
        open_factory=lambda: OpenVLABoundary(_row(1.0), _ready(), []),
        mini_factory=lambda: MiniVLABoundary(_row(2.0), _ready(), []),
        octo_factory=build_octo,
    )
    common = dict(policy_id="octo-small", prompt="Close the drawer", image_history=(np.zeros((2, 2, 3), dtype=np.uint8),))
    missing = router.predict(
        ExperimentalPolicyTurnRequest(**common, identifiers=_ids(proprio_lineage_id="p", proprio_converter_revision="c"))
    )
    zeros = router.predict(
        ExperimentalPolicyTurnRequest(
            **common,
            proprio=(0.0,) * 8,
            identifiers=_ids(proprio_lineage_id="p", proprio_converter_revision="c"),
        )
    )
    assert missing.status == zeros.status == "failed"
    assert "source-converted proprioception" in missing.reason
    assert "all-zero" in zeros.reason
    assert created == []

    first = router.predict(
        ExperimentalPolicyTurnRequest(
            **common,
            proprio=(1.0,) * 8,
            identifiers=_ids(proprio_lineage_id="p", proprio_converter_revision="c"),
        )
    )
    wrong_cell = router.predict(
        ExperimentalPolicyTurnRequest(
            policy_id="octo-small",
            prompt="Close the drawer",
            image_history=(np.ones((2, 2, 3), dtype=np.uint8),),
            proprio=(1.0,) * 8,
            prior_snapshot=first.snapshot,
            identifiers=_ids(
                cell_id="other-cell",
                observation_id="obs-1",
                proprio_lineage_id="p1",
                proprio_converter_revision="c",
            ),
        )
    )
    assert wrong_cell.status == "failed"
    assert "another cell" in wrong_cell.reason


def test_router_rejects_a_reused_stateful_adapter_instead_of_leaking_history_across_turns():
    shared = OpenVLABoundary(_row(1.0), _ready(), [])
    router = _router(
        open_factory=lambda: shared,
        mini_factory=lambda: MiniVLABoundary(_row(2.0), _ready(), []),
        octo_factory=lambda: OctoBoundary((_row(3.0),) * 4, _row(4.0), _ready(), [], []),
    )
    first = router.predict(
        ExperimentalPolicyTurnRequest(
            policy_id="openvla",
            prompt="Close the drawer",
            image_history=(np.zeros((2, 2, 3), dtype=np.uint8),),
            identifiers=_ids(),
        )
    )
    second = router.predict(
        ExperimentalPolicyTurnRequest(
            policy_id="openvla",
            prompt="Close the drawer",
            image_history=(np.ones((2, 2, 3), dtype=np.uint8),),
            prior_snapshot=first.snapshot,
            identifiers=_ids(observation_id="obs-1"),
        )
    )
    assert first.status == "completed"
    assert second.status == "failed"
    assert "fresh wrapper per turn" in second.reason


def test_single_policy_router_keeps_isolated_workers_from_loading_unrelated_policy_stacks():
    router = make_experimental_policy_router(
        "openvla", lambda: OpenVLABoundary(_row(1.0), _ready(), [])
    )
    result = router.predict(
        ExperimentalPolicyTurnRequest(
            policy_id="MiniVLA",
            prompt="Close the drawer",
            image_history=(np.zeros((2, 2, 3), dtype=np.uint8),),
            identifiers=_ids(),
        )
    )
    assert result.status == "failed"
    assert "configured for openvla" in result.reason


class _RestoreRandom:
    def PRNGKey(self, seed):
        return ("prng", seed)


class _RestoreJax:
    random = _RestoreRandom()

    @staticmethod
    def tree_map(function, value):
        if isinstance(value, dict):
            return {key: _RestoreJax.tree_map(function, item) for key, item in value.items()}
        return function(value)


class _RestoreOctoModel:
    def __init__(self):
        self.create_calls = []
        self.sample_calls = []
        self.dataset_statistics = {"bridge_dataset": {"action": {"mean": np.zeros(7), "std": np.ones(7)}}}

    def create_tasks(self, *, texts):
        self.create_calls.append(texts)
        return {"texts": texts}

    def sample_actions(self, observations, tasks, *, rng):
        self.sample_calls.append((observations, tasks, rng))
        return np.asarray([[_row(20.0), _row(21.0), _row(22.0), _row(23.0)]], dtype=np.float32)


def test_diagnostic_octo_restores_real_history_and_raw_proposals_without_replaying_model_calls():
    model = _RestoreOctoModel()
    runtime = _OctoRuntime(
        jax=_RestoreJax(),
        numpy=np,
        model_cls=None,
        stack_and_pad=lambda history, count: {
            "image_primary": np.stack([item["image_primary"] for item in history]),
            "proprio": np.asarray([item["proprio"] for item in history]),
            "pad_mask": np.ones((len(history),), dtype=bool),
        },
        jax_version="0.4.20",
    )
    profile = OctoV1PolicyProfile(
        profile_id="octo-source-state-fixture",
        local_model_path="/not-used-by-fixture",
        checkpoint_revision=OCTO_SMALL_MODEL_REVISION,
        code_revision=OCTO_V1_SOURCE_COMMIT,
    )
    adapter = OctoSmallV1Policy(
        profile,
        runtime_factory=lambda: runtime,
        model_factory=lambda profile, runtime: model,
    )
    prior_proposal = [list(_row(10.0)), list(_row(11.0)), list(_row(12.0)), list(_row(13.0))]
    adapter.restore_source_state(
        {
            "schema": "plumb-octo-v0.1-source-state-v1",
            "task_instruction": "Close the drawer",
            "observation_count": 1,
            "observation_history": [
                {"decoded_rgb": np.zeros((2, 2, 3), dtype=np.uint8), "proprio": [1.0] * 8}
            ],
            "proposal_history": [prior_proposal],
        }
    )
    assert model.create_calls == model.sample_calls == []

    report = adapter.predict_with_report(
        SimpleNamespace(
            image_history=(np.ones((2, 2, 3), dtype=np.uint8),),
            prompt="Close the drawer",
            proprio=(2.0,) * 8,
            timestamp=2.0,
        )
    )

    assert len(model.sample_calls) == 1
    assert model.create_calls == [["Close the drawer"]]
    # At the second source tick, uniform temporal ensembling selects prior
    # proposal row 1 and current proposal row 0. The first row is not executed.
    assert report.action == _row(15.5)
