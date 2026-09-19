"""CPU-only tests for the Baseten Chains topology.

Nothing here imports ``truss_chains``, torch, diffusers, or JAX, and nothing here
touches a GPU.  The module under test is loaded by path because
``deploy/baseten/chain.py`` deliberately lives outside an importable package
namespace used at runtime: ``truss chains push`` treats its directory as the
chain workspace.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHAIN_PATH = REPO_ROOT / "deploy" / "baseten" / "chain.py"
REQUIREMENTS_DIR = REPO_ROOT / "deploy" / "baseten" / "requirements"


def _load_chain() -> Any:
    spec = importlib.util.spec_from_file_location("plumb_deploy_chain", CHAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing: when truss_chains IS installed, its
    # ``__init_subclass__`` hook calls ``inspect.getfile(cls)``, which needs the
    # defining module to be resolvable through sys.modules.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


chain = _load_chain()

# The project venv deliberately has no truss_chains; the dedicated deploy venv
# does. A handful of tests are specific to one of those two worlds.
CHAINS_SDK_INSTALLED = importlib.util.find_spec("truss_chains") is not None
requires_no_sdk = pytest.mark.skipif(
    CHAINS_SDK_INSTALLED, reason="asserts the no-SDK import path; truss_chains is installed here"
)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


@requires_no_sdk
def test_module_imports_without_the_chains_sdk():
    """A normal checkout has no truss_chains, and importing must still work."""

    assert chain.CHAINS_RUNTIME_AVAILABLE is False
    for name in (
        "WorldWorker",
        "OpenVLAWorker",
        "OctoWorker",
        "MiniVLAWorker",
        "OpenPiZeroWorker",
        "SusieWorker",
        "ValidityWorker",
        "JudgeWorker",
        "PolicyRouter",
        "RolloutController",
    ):
        assert getattr(chain, name) is None, "%s must not be a deployable class locally" % name


def test_combined_policy_worker_is_retired():
    """One shared policy image cannot hold torch 2.2, torch 2.6, and JAX 0.4.20."""

    assert chain.PolicyWorker is None
    assert set(chain.POLICY_WORKER_CLASS_NAMES) == {
        "OpenVLAWorker",
        "OctoWorker",
        "MiniVLAWorker",
        "OpenPiZeroWorker",
        "SusieWorker",
    }


@requires_no_sdk
def test_require_chains_runtime_refuses_to_pretend():
    with pytest.raises(chain.ChainRuntimeUnavailable):
        chain.require_chains_runtime()


def test_chain_module_avoids_postponed_annotations():
    """PEP 563 would make every endpoint annotation a string.

    ``truss_chains.framework._validate_io_type`` rejects string annotations
    outright, so the future import would make this Chain unpushable.
    """

    statements = [
        line.strip()
        for line in CHAIN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("from __future__ import")
    ]
    assert statements == [], "chain.py must not use postponed annotations: %r" % statements


# ---------------------------------------------------------------------------
# Wire contracts
# ---------------------------------------------------------------------------


def _frame(**overrides: Any) -> Dict[str, Any]:
    payload = {"encoding": "png_base64", "data": "aGVsbG8="}
    payload.update(overrides)
    return payload


def test_stage_request_requires_identity_fields():
    request = chain.StageRequest(episode_id="ep-1", protocol_hash="ph-1", payload={"a": 1})
    assert request.payload == {"a": 1}
    with pytest.raises(Exception):
        chain.StageRequest(episode_id="", protocol_hash="ph-1")
    with pytest.raises(Exception):
        chain.StageRequest(episode_id="ep-1", protocol_hash="ph-1", surprise="no")


def test_rollout_request_detects_identity_mismatch():
    def stage(episode_id: str = "ep-1", protocol_hash: str = "ph-1") -> chain.StageRequest:
        return chain.StageRequest(episode_id=episode_id, protocol_hash=protocol_hash)

    good = chain.RolloutRequest(
        run_id="run-1",
        episode_id="ep-1",
        protocol_hash="ph-1",
        policy=stage(),
        world=stage(),
        validity=stage(),
        judge=stage(),
    )
    assert good.matching_episode_requests() is True

    mismatched = chain.RolloutRequest(
        run_id="run-1",
        episode_id="ep-1",
        protocol_hash="ph-1",
        policy=stage(),
        world=stage(episode_id="ep-2"),
        validity=stage(),
        judge=stage(),
    )
    assert mismatched.matching_episode_requests() is False

    wrong_protocol = chain.RolloutRequest(
        run_id="run-1",
        episode_id="ep-1",
        protocol_hash="ph-1",
        policy=stage(),
        world=stage(),
        validity=stage(),
        judge=stage(protocol_hash="ph-2"),
    )
    assert wrong_protocol.matching_episode_requests() is False


def test_stage_result_rejects_unknown_stage_or_status():
    ok = chain.StageResult(stage="world", status="blocked", unresolved_contracts=["x"])
    assert ok.output is None
    with pytest.raises(Exception):
        chain.StageResult(stage="physics", status="blocked")
    with pytest.raises(Exception):
        chain.StageResult(stage="world", status="probably_fine")


def test_rollout_result_defaults_are_null_not_zero():
    result = chain.RolloutResult(
        run_id="run-1", episode_id="ep-1", protocol_hash="ph-1", status="blocked"
    )
    assert result.binary_success is None
    assert result.progress_score is None
    assert result.validity is None
    assert result.judge_status is None
    assert result.qualified is False
    assert result.physical_state_measured is False
    assert result.executed_actions == 0
    assert result.stages == {}
    assert result.stage_history == []


def test_rollout_result_field_names_match_the_plumb_consumer():
    """plumb/backends/baseten.py reads these exact names off the Chain result."""

    fields = chain.RolloutResult.model_fields
    # Rollout-level progress is `progress_score` in plumb.rollout.RolloutReport.
    assert "progress_score" in fields
    assert "progress" not in fields
    # `stages` is indexed by stage name by the backend, so it must be a mapping.
    assert chain.RolloutResult(
        run_id="r", episode_id="e", protocol_hash="p", status="blocked"
    ).stages == {}
    for name in ("run_id", "episode_id", "protocol_hash", "status", "missing_reason", "validity", "binary_success"):
        assert name in fields


def test_rollout_result_stage_mapping_keeps_the_last_of_each_stage():
    """A rollout makes many policy/world calls, so the mapping keeps the last."""

    history = [
        chain.StageResult(stage="policy", status="completed"),
        chain.StageResult(stage="world", status="completed"),
        chain.StageResult(stage="policy", status="blocked", unresolved_contracts=["second call blocked"]),
    ]
    terminal = {entry.stage: entry for entry in history}
    result = chain.RolloutResult(
        run_id="r",
        episode_id="e",
        protocol_hash="p",
        status="blocked",
        stages=terminal,
        stage_history=history,
    )
    assert set(result.stages) == {"policy", "world"}
    assert result.stages["policy"].status == "blocked"
    assert len(result.stage_history) == 3


def test_judge_payload_forbids_unblinded_fields():
    """Structural blinding: the leaky fields have no place to land."""

    base = {
        "frames": [_frame() for _ in range(16)],
        "frame_timestamps": [float(index) for index in range(16)],
        "reference_images": [{"image": _frame(), "source_uri": "s3://panel", "sha256": "a" * 64}],
        "task_id": "open_drawer",
        "seeds": [1, 2, 3, 4, 5],
    }
    assert chain.JudgeStagePayload.model_validate(base).task_id == "open_drawer"

    for leak in (
        "policy",
        "policy_arm",
        "native_actions",
        "actions",
        "commands",
        "reference_success_rate",
        "condition_label",
        "validity",
        "stage_a_outcome",
        "task_instruction",
        "task_rubric",
    ):
        payload = dict(base)
        payload[leak] = "anything"
        with pytest.raises(Exception):
            chain.JudgeStagePayload.model_validate(payload)
        assert leak not in chain.JudgeStagePayload.model_fields


def test_judge_payload_requires_exactly_sixteen_frames_and_five_seeds():
    def build(frames: int, seeds: List[int]) -> Dict[str, Any]:
        return {
            "frames": [_frame() for _ in range(frames)],
            "frame_timestamps": [float(index) for index in range(frames)],
            "reference_images": [{"image": _frame(), "source_uri": "s3://panel", "sha256": "b" * 64}],
            "task_id": "to_sink",
            "seeds": seeds,
        }

    chain.JudgeStagePayload.model_validate(build(16, [1, 2, 3, 4, 5]))
    with pytest.raises(Exception):
        chain.JudgeStagePayload.model_validate(build(15, [1, 2, 3, 4, 5]))
    with pytest.raises(Exception):
        chain.JudgeStagePayload.model_validate(build(17, [1, 2, 3, 4, 5]))
    with pytest.raises(Exception):
        chain.JudgeStagePayload.model_validate(build(16, [1, 2, 3, 4]))


def test_world_payload_requires_actions_and_timestamps():
    good = chain.WorldStagePayload.model_validate(
        {
            "compatibility_profile_id": "plumb-cosmos3_nano-fd-r256",
            "domain": "bridge_orig_lerobot",
            "prompt": "Open the drawer",
            "conditioning_image": _frame(),
            "compiled_actions": [[0.0] * 10],
            "nominal_control_timestamps": [0.2],
            "seed": 7,
        }
    )
    assert good.return_frames is True
    with pytest.raises(Exception):
        chain.WorldStagePayload.model_validate(
            {
                "compatibility_profile_id": "p",
                "domain": "bridge_orig_lerobot",
                "prompt": "Open the drawer",
                "conditioning_image": _frame(),
                "compiled_actions": [],
                "nominal_control_timestamps": [],
                "seed": 7,
            }
        )


def test_episode_payload_accepts_seven_or_eight_dimensional_state():
    def build(state: List[float]) -> Dict[str, Any]:
        return {
            "policy": "OpenVLA",
            "task_id": "open_drawer",
            "prompt": "Open the drawer",
            "horizon_actions": 70,
            "initial_frame": _frame(),
            "initial_state": state,
            "bridge_control_profile_id": "bridge-v1",
        }

    assert chain.EpisodeControlPayload.model_validate(build([0.0] * 7)).horizon_actions == 70
    assert chain.EpisodeControlPayload.model_validate(build([0.0] * 8)).control_hz == 5.0
    with pytest.raises(Exception):
        chain.EpisodeControlPayload.model_validate(build([0.0] * 6))
    with pytest.raises(Exception):
        chain.EpisodeControlPayload.model_validate(build([0.0] * 9))


# ---------------------------------------------------------------------------
# Policy routing table
# ---------------------------------------------------------------------------


def test_policy_arm_normalisation_is_an_explicit_table():
    assert chain.normalise_policy_arm("OpenVLA") == "OpenVLA"
    assert chain.normalise_policy_arm("openvla-7b") == "OpenVLA"
    assert chain.normalise_policy_arm("octo") == "Octo-Small"
    assert chain.normalise_policy_arm("susie_ll") == "SuSIE_LL"
    # No fuzzy matching: an unknown or near-miss name must not be routed.
    assert chain.normalise_policy_arm("OpenVLA-v2") is None
    assert chain.normalise_policy_arm("Octo-Tiny") is None
    assert chain.normalise_policy_arm("") is None
    assert chain.normalise_policy_arm(None) is None


def test_every_canonical_arm_maps_to_exactly_one_worker():
    seen: Dict[str, str] = {}
    for worker, arms in chain.WORKER_ARMS.items():
        for arm in arms:
            assert arm not in seen, "%s is served by two workers" % arm
            seen[arm] = worker
    assert set(seen) == set(chain.CANONICAL_POLICY_ARMS)
    # SuSIE and SuSIE_LL share one image but stay distinct arms.
    assert seen["SuSIE"] == seen["SuSIE_LL"] == "plumb-susie-worker"
    assert seen["Octo-Small"] == seen["Octo-Base"] == "plumb-octo-worker"


def test_every_worker_has_its_own_requirements_file():
    files = list(chain.WORKER_REQUIREMENTS_FILES.values())
    assert len(files) == len(set(files)), "two Chainlets share a requirements file"
    for relative in files:
        assert (REPO_ROOT / "deploy" / "baseten" / relative).is_file()


# ---------------------------------------------------------------------------
# Autoscaling declarations
# ---------------------------------------------------------------------------


def test_autoscaling_covers_every_chainlet_and_labels_itself_a_hypothesis():
    payloads = chain.autoscaling_patch_payloads()
    names = {item["chainlet_name"] for item in payloads}
    assert names == set(chain.WORKER_DECLARED_GPU_COUNT)
    for item in payloads:
        assert item["status"] == "initial_hypothesis_not_measured"
        assert item["rationale"]
        settings = item["autoscaling_settings"]
        assert set(settings) == {"min_replica", "max_replica", "concurrency_target", "autoscaling_window"}
        assert settings["max_replica"] >= max(1, settings["min_replica"])


def test_world_and_judge_default_to_the_hundred_replica_hypothesis():
    settings = {item.chainlet_name: item for item in chain.AUTOSCALING_HYPOTHESES}
    world = settings["plumb-world-worker"]
    judge = settings["plumb-judge-worker"]
    assert world.max_replica == 100
    assert judge.max_replica == 100
    # Diffusion saturates the GPU, so the spec's starting point is 1.
    assert world.concurrency_target == 1
    assert judge.concurrency_target == 1


def test_unloadable_policy_arms_scale_from_zero():
    """A warm replica that can only return blocked would burn GPU-hours."""

    settings = {item.chainlet_name: item for item in chain.AUTOSCALING_HYPOTHESES}
    for name in (
        "plumb-octo-worker",
        "plumb-minivla-worker",
        "plumb-openpizero-worker",
        "plumb-susie-worker",
    ):
        assert settings[name].min_replica == 0


def test_declared_gpu_counts_match_the_cpu_only_workers():
    assert chain.WORKER_DECLARED_GPU_COUNT["plumb-validity-worker"] == 0
    assert chain.WORKER_DECLARED_GPU_COUNT["plumb-rollout-controller"] == 0
    assert chain.WORKER_DECLARED_GPU_COUNT["plumb-world-worker"] == 1
    assert chain.WORKER_DECLARED_GPU_COUNT["plumb-judge-worker"] == 1


def test_world_variants_never_invent_an_edge_revision():
    nano = chain.WORLD_VARIANTS["cosmos3_nano"]
    edge = chain.WORLD_VARIANTS["cosmos3_edge"]
    assert nano.revision == "e59a53c25979a090fa8706c9acc0c254a6e89b92"
    assert len(nano.revision) == 40
    # Unset unless the operator supplies PLUMB_WORLD_EDGE_REVISION at push time.
    assert edge.revision == ""
    assert edge.metadata_model_id_is_profile_default is True
    assert chain.selected_world_variant().variant_id == "cosmos3_nano"


# ---------------------------------------------------------------------------
# Micro-batch planner: grouping, timeout flush, partial batches, isolation
# ---------------------------------------------------------------------------


def _ticket(planner: Any, key: str, payload: Any, submitted_at: float) -> Any:
    ticket = chain.MicroBatchTicket(
        ticket_id=planner.next_ticket_id(),
        batch_key=key,
        payload=payload,
        submitted_at=submitted_at,
    )
    planner.submit(ticket)
    return ticket


def test_planner_rejects_impossible_configuration():
    with pytest.raises(ValueError):
        chain.MicroBatchPlanner(batch_max=0, batch_window_ms=10)
    with pytest.raises(ValueError):
        chain.MicroBatchPlanner(batch_max=4, batch_window_ms=-1)


def test_planner_emits_full_batches_immediately():
    planner = chain.MicroBatchPlanner(batch_max=3, batch_window_ms=50)
    for index in range(7):
        _ticket(planner, "k", index, 1000.0)
    batches = planner.take_ready(1000.0)
    assert [[ticket.payload for ticket in batch] for batch in batches] == [[0, 1, 2], [3, 4, 5]]
    # The seventh is a partial batch whose window has not expired.
    assert planner.pending_count == 1


def test_planner_holds_a_partial_batch_until_the_window_expires():
    planner = chain.MicroBatchPlanner(batch_max=8, batch_window_ms=15)
    _ticket(planner, "k", "a", 1000.0)
    _ticket(planner, "k", "b", 1000.005)
    assert planner.take_ready(1000.010) == []
    assert planner.pending_count == 2
    batches = planner.take_ready(1000.020)
    assert [[ticket.payload for ticket in batch] for batch in batches] == [["a", "b"]]
    assert planner.pending_count == 0


def test_planner_partial_batch_is_not_padded_to_batch_max():
    planner = chain.MicroBatchPlanner(batch_max=16, batch_window_ms=0)
    _ticket(planner, "k", "only", 1000.0)
    batches = planner.take_ready(1000.0)
    assert len(batches) == 1
    assert len(batches[0]) == 1, "a partial batch must never be padded with duplicated work"


def test_planner_force_flush_releases_an_unexpired_partial_batch():
    planner = chain.MicroBatchPlanner(batch_max=8, batch_window_ms=1000)
    _ticket(planner, "k", "a", 1000.0)
    assert planner.take_ready(1000.0) == []
    batches = planner.take_ready(1000.0, force_flush=True)
    assert [[ticket.payload for ticket in batch] for batch in batches] == [["a"]]


def test_planner_never_mixes_incompatible_batch_keys():
    planner = chain.MicroBatchPlanner(batch_max=2, batch_window_ms=0)
    for index, key in enumerate(["a", "b", "a", "b", "a"]):
        _ticket(planner, key, (key, index), 1000.0)
    batches = planner.take_ready(1000.0)
    for batch in batches:
        assert len({ticket.batch_key for ticket in batch}) == 1
    assert [[ticket.payload for ticket in batch] for batch in batches] == [
        [("a", 0), ("a", 2)],
        [("a", 4)],
        [("b", 1), ("b", 3)],
    ]


def test_planner_preserves_fifo_order_within_a_key():
    planner = chain.MicroBatchPlanner(batch_max=4, batch_window_ms=0)
    for index in range(4):
        _ticket(planner, "k", index, 1000.0 + index)
    batch = planner.take_ready(1000.0 + 3)[0]
    assert [ticket.payload for ticket in batch] == [0, 1, 2, 3]
    assert [ticket.ticket_id for ticket in batch] == sorted(ticket.ticket_id for ticket in batch)


def test_planner_reports_time_until_the_next_flush():
    planner = chain.MicroBatchPlanner(batch_max=8, batch_window_ms=20)
    assert planner.seconds_until_next_flush(1000.0) is None
    _ticket(planner, "k", "a", 1000.0)
    assert planner.seconds_until_next_flush(1000.0) == pytest.approx(0.020)
    assert planner.seconds_until_next_flush(1000.030) == 0.0


def test_planner_drops_tickets_whose_caller_went_away():
    planner = chain.MicroBatchPlanner(batch_max=4, batch_window_ms=0)
    keep = _ticket(planner, "k", "keep", 1000.0)
    gone = _ticket(planner, "k", "gone", 1000.0)
    dropped = planner.drop(lambda ticket: ticket.payload == "gone")
    assert dropped == [gone]
    assert planner.pending_count == 1
    assert planner.take_ready(1000.0)[0] == [keep]


def test_scatter_pairs_results_positionally():
    planner = chain.MicroBatchPlanner(batch_max=4, batch_window_ms=0)
    tickets = [_ticket(planner, "k", index, 1000.0) for index in range(3)]
    outcomes = chain.MicroBatchPlanner.scatter(tickets, ["r0", "r1", "r2"])
    assert [outcome.value for outcome in outcomes] == ["r0", "r1", "r2"]
    assert all(outcome.ok for outcome in outcomes)


def test_scatter_isolates_one_items_exception():
    planner = chain.MicroBatchPlanner(batch_max=4, batch_window_ms=0)
    tickets = [_ticket(planner, "k", index, 1000.0) for index in range(3)]
    boom = ValueError("only the middle item failed")
    outcomes = chain.MicroBatchPlanner.scatter(tickets, ["r0", boom, "r2"])
    assert outcomes[0].value == "r0" and outcomes[0].ok
    assert outcomes[1].error is boom and not outcomes[1].ok
    assert outcomes[2].value == "r2" and outcomes[2].ok


def test_scatter_propagates_a_whole_batch_failure_to_every_caller():
    planner = chain.MicroBatchPlanner(batch_max=4, batch_window_ms=0)
    tickets = [_ticket(planner, "k", index, 1000.0) for index in range(3)]
    boom = RuntimeError("cuda oom")
    outcomes = chain.MicroBatchPlanner.scatter(tickets, boom)
    assert all(outcome.error is boom for outcome in outcomes)


def test_scatter_refuses_to_reuse_a_result_on_a_length_mismatch():
    """Reusing a result would attribute one episode's frames to another."""

    planner = chain.MicroBatchPlanner(batch_max=4, batch_window_ms=0)
    tickets = [_ticket(planner, "k", index, 1000.0) for index in range(3)]
    outcomes = chain.MicroBatchPlanner.scatter(tickets, ["only-one"])
    assert len(outcomes) == 3
    assert all(isinstance(outcome.error, chain.BatchContractError) for outcome in outcomes)
    assert all(outcome.value is None for outcome in outcomes)
    assert "never reused" in str(outcomes[0].error)


def test_scatter_rejects_a_non_iterable_result_set():
    planner = chain.MicroBatchPlanner(batch_max=4, batch_window_ms=0)
    tickets = [_ticket(planner, "k", 0, 1000.0)]
    outcomes = chain.MicroBatchPlanner.scatter(tickets, 42)
    assert isinstance(outcomes[0].error, chain.BatchContractError)


# ---------------------------------------------------------------------------
# Micro-batch queue: the asyncio collector around the pure planner
# ---------------------------------------------------------------------------


def test_queue_packs_concurrent_callers_into_one_executor_call():
    seen: List[List[Any]] = []

    def executor(payloads: List[Any]) -> List[Any]:
        seen.append(list(payloads))
        return [payload * 10 for payload in payloads]

    async def scenario() -> List[Any]:
        queue = chain.MicroBatchQueue(executor, batch_max=4, batch_window_ms=5)
        return await asyncio.gather(*[queue.submit("k", index) for index in range(6)])

    results = asyncio.run(scenario())
    assert results == [0, 10, 20, 30, 40, 50]
    assert seen[0] == [0, 1, 2, 3], "the first four callers must share one executor call"
    assert sum(len(batch) for batch in seen) == 6


def test_queue_delivers_a_per_item_exception_only_to_that_caller():
    def executor(payloads: List[Any]) -> List[Any]:
        return [ValueError("item %s failed" % payload) if payload == 2 else payload for payload in payloads]

    async def scenario() -> List[Any]:
        queue = chain.MicroBatchQueue(executor, batch_max=4, batch_window_ms=1)

        async def one(value: int) -> Any:
            try:
                return await queue.submit("k", value)
            except ValueError as error:
                return "raised:%s" % error

        return await asyncio.gather(*[one(index) for index in range(4)])

    assert asyncio.run(scenario()) == [0, 1, "raised:item 2 failed", 3]


def test_queue_raises_the_executor_failure_for_every_caller_in_the_batch():
    def executor(payloads: List[Any]) -> List[Any]:
        raise RuntimeError("backend unavailable")

    async def scenario() -> List[str]:
        queue = chain.MicroBatchQueue(executor, batch_max=2, batch_window_ms=1)

        async def one(value: int) -> str:
            try:
                await queue.submit("k", value)
            except RuntimeError as error:
                return str(error)
            return "unexpected success"

        return await asyncio.gather(*[one(index) for index in range(2)])

    assert asyncio.run(scenario()) == ["backend unavailable", "backend unavailable"]


def test_queue_keeps_incompatible_keys_in_separate_executor_calls():
    seen: List[List[Any]] = []

    def executor(payloads: List[Any]) -> List[Any]:
        seen.append(list(payloads))
        return list(payloads)

    async def scenario() -> None:
        queue = chain.MicroBatchQueue(executor, batch_max=8, batch_window_ms=2)
        await asyncio.gather(
            queue.submit("shape-a", "a1"),
            queue.submit("shape-b", "b1"),
            queue.submit("shape-a", "a2"),
        )

    asyncio.run(scenario())
    for batch in seen:
        assert set(batch) <= {"a1", "a2"} or set(batch) <= {"b1"}


def test_queue_survives_a_timeout_flush_of_a_single_request():
    def executor(payloads: List[Any]) -> List[Any]:
        return ["ok:%s" % payload for payload in payloads]

    async def scenario() -> Any:
        queue = chain.MicroBatchQueue(executor, batch_max=16, batch_window_ms=5)
        return await queue.submit("k", "lonely")

    assert asyncio.run(scenario()) == "ok:lonely"


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


def test_uniform_frame_indices_include_both_endpoints_without_repeats():
    for available in (16, 17, 20, 71, 101):
        indices = chain._uniform_frame_indices(available, 16)
        assert len(indices) == 16
        assert len(set(indices)) == 16
        assert indices[0] == 0
        assert indices[-1] == available - 1
        assert indices == sorted(indices)


def test_uniform_frame_indices_rejects_a_short_clip_instead_of_padding():
    with pytest.raises(ValueError):
        chain._uniform_frame_indices(15, 16)


# ---------------------------------------------------------------------------
# Requirements files
# ---------------------------------------------------------------------------

EXPECTED_REQUIREMENTS = {
    "world-cosmos.txt",
    "policy-openvla.txt",
    "policy-octo.txt",
    "policy-minivla.txt",
    "policy-openpizero.txt",
    "policy-susie.txt",
    "judge-qwen.txt",
    "validity.txt",
    "controller.txt",
}

PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==([^\s;]+)$")
GIT_PATTERN = re.compile(r"^([A-Za-z0-9_.\-]+) @ git\+https://[^\s@]+@([0-9a-f]{40})$")
ALLOWED_OPTION_PREFIXES = ("--extra-index-url ", "-f ", "--find-links ", "--index-url ")


def _requirement_lines(path: Path) -> List[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def test_every_expected_requirements_file_exists():
    present = {path.name for path in REQUIREMENTS_DIR.glob("*.txt")}
    assert present == EXPECTED_REQUIREMENTS


@pytest.mark.parametrize("name", sorted(EXPECTED_REQUIREMENTS))
def test_requirements_file_parses_as_pip_requirements(name: str):
    path = REQUIREMENTS_DIR / name
    lines = _requirement_lines(path)
    assert lines, "%s has no installable requirement" % name
    for line in lines:
        if line.startswith(ALLOWED_OPTION_PREFIXES):
            continue
        assert PIN_PATTERN.match(line) or GIT_PATTERN.match(line), "%s: unparseable line %r" % (name, line)


@pytest.mark.parametrize("name", sorted(EXPECTED_REQUIREMENTS))
def test_requirements_file_has_no_computecanada_wheel(name: str):
    """No installable line may use an Alliance local build.

    The check is scoped to requirement lines rather than the whole file on
    purpose: each header *names* the ``+computecanada`` pins recorded in
    HANDOFF.md in order to explain why the recorded evidence does not transfer.
    Documenting the trap is required; installing it is what must not happen.
    """

    path = REQUIREMENTS_DIR / name
    for line in _requirement_lines(path):
        assert "computecanada" not in line, "%s: %r installs an Alliance local build" % (name, line)
    assert "+computecanada" in path.read_text(encoding="utf-8"), (
        "%s must explain that the recorded ComputeCanada pins do not transfer" % name
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_REQUIREMENTS))
def test_requirements_file_states_that_cluster_evidence_does_not_transfer(name: str):
    text = (REQUIREMENTS_DIR / name).read_text(encoding="utf-8")
    assert "DOES NOT TRANSFER" in text
    assert "cluster-runtime-evidence.json" in text or "ComputeCanada" in text


@pytest.mark.parametrize("name", sorted(EXPECTED_REQUIREMENTS))
def test_requirements_pins_are_exact_not_floating(name: str):
    for line in _requirement_lines(REQUIREMENTS_DIR / name):
        if line.startswith(ALLOWED_OPTION_PREFIXES) or GIT_PATTERN.match(line):
            continue
        assert "==" in line, "%s: %r is not an exact pin" % (name, line)
        for loose in (">=", "<=", "~=", ">", "<"):
            assert loose not in line, "%s: %r uses a floating specifier" % (name, line)


@pytest.mark.parametrize("name", sorted(EXPECTED_REQUIREMENTS))
def test_git_requirements_use_immutable_forty_hex_revisions(name: str):
    for line in _requirement_lines(REQUIREMENTS_DIR / name):
        if "git+" not in line:
            continue
        assert GIT_PATTERN.match(line), "%s: %r is not pinned to a 40-hex commit" % (name, line)
        assert "@main" not in line and "@master" not in line


def test_cosmos_requirements_pin_the_recorded_diffusers_revision():
    lines = _requirement_lines(REQUIREMENTS_DIR / "world-cosmos.txt")
    assert any("a3e0b8ec235c27a6c17a21976daf7fd32d819d05" in line for line in lines)
    assert "transformers==5.17.0" in lines
    assert "huggingface-hub==1.32.0" in lines


def test_openvla_requirements_pin_the_enforced_transformers_version():
    lines = _requirement_lines(REQUIREMENTS_DIR / "policy-openvla.txt")
    assert "transformers==4.40.1" in lines
    assert "tokenizers==0.19.1" in lines
    assert "timm==0.9.10" in lines


def test_octo_requirements_record_the_named_cuda_deviation():
    path = REQUIREMENTS_DIR / "policy-octo.txt"
    assert "jax[cuda12_pip]==0.4.20" in _requirement_lines(path)
    text = path.read_text(encoding="utf-8")
    assert "cuda11_pip" in text, "the deviation from the spec's cuda11 pin must be stated"
    assert "DEVIATION" in text


def test_minivla_requirements_keep_the_pinned_torch_and_explain_flash_attn():
    path = REQUIREMENTS_DIR / "policy-minivla.txt"
    lines = _requirement_lines(path)
    assert "torch==2.2.0" in lines
    assert "torchvision==0.17.0" in lines
    text = path.read_text(encoding="utf-8")
    # flash-attn is installed by build_commands, not by this file, because pip
    # does not accept --no-build-isolation inside a requirements file.
    assert "flash-attn==2.5.5" not in lines
    assert "flash-attn==2.5.5" in text
    assert "--no-build-isolation" in text


def test_validity_requirements_are_cpu_only():
    lines = _requirement_lines(REQUIREMENTS_DIR / "validity.txt")
    joined = "\n".join(lines)
    assert "opencv-python-headless==" in joined
    assert "numpy==" in joined
    assert "imageio==" in joined
    for banned in ("torch", "jax", "tensorflow", "diffusers", "transformers", "vjepa"):
        for line in lines:
            assert banned not in line.lower(), (
                "the deterministic Stage-A image must not install %s" % banned
            )


def test_judge_requirements_avoid_a_second_serving_engine():
    lines = _requirement_lines(REQUIREMENTS_DIR / "judge-qwen.txt")
    assert "transformers==4.49.0" in lines
    for line in lines:
        for banned in ("vllm", "sglang", "tensorrt"):
            assert banned not in line.lower(), (
                "a different serving stack would be a different judge revision, needing a fresh Gate D"
            )


def test_superseded_index_points_at_the_real_files():
    text = (REPO_ROOT / "deploy" / "baseten" / "requirements.lock-inputs.txt").read_text(encoding="utf-8")
    assert "SUPERSEDED" in text
    for name in EXPECTED_REQUIREMENTS:
        assert "requirements/%s" % name in text


# ---------------------------------------------------------------------------
# Frame codec
# ---------------------------------------------------------------------------


def test_frame_round_trip_preserves_pixels_and_binds_both_digests():
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    rng = numpy.random.default_rng(7)
    original = rng.integers(0, 256, size=(8, 11, 3), dtype=numpy.uint8)
    payload = chain.encode_frame(original, nominal_timestamp=0.4)

    assert payload.encoding == "png_base64"
    assert payload.height == 8 and payload.width == 11
    assert payload.nominal_timestamp == 0.4
    assert payload.png_sha256 != payload.pixels_sha256, (
        "PNG file bytes and decoded pixels are different artifacts and must not share one ambiguous hash"
    )
    assert numpy.array_equal(chain.decode_frame(payload), original)


def test_frame_decode_rejects_a_tampered_pixel_digest():
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    payload = chain.encode_frame(numpy.zeros((4, 4, 3), dtype=numpy.uint8))
    tampered = payload.model_copy(update={"pixels_sha256": "sha256:" + "0" * 64})
    with pytest.raises(ValueError, match="pixels_sha256"):
        chain.decode_frame(tampered)


def test_frame_decode_rejects_a_tampered_png_digest_and_bad_base64():
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    payload = chain.encode_frame(numpy.zeros((4, 4, 3), dtype=numpy.uint8))
    with pytest.raises(ValueError, match="png_sha256"):
        chain.decode_frame(payload.model_copy(update={"png_sha256": "sha256:" + "1" * 64}))
    with pytest.raises(ValueError, match="base64"):
        chain.decode_frame(payload.model_copy(update={"data": "not base64 !!!"}))


def test_frame_decode_rejects_a_declared_size_mismatch():
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    payload = chain.encode_frame(numpy.zeros((4, 5, 3), dtype=numpy.uint8))
    with pytest.raises(ValueError, match="height"):
        chain.decode_frame(payload.model_copy(update={"height": 9}))
    with pytest.raises(ValueError, match="width"):
        chain.decode_frame(payload.model_copy(update={"width": 9}))


def test_encode_frame_rejects_non_finite_float_pixels():
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    broken = numpy.zeros((4, 4, 3), dtype=numpy.float32)
    broken[0, 0, 0] = numpy.nan
    with pytest.raises(ValueError, match="non-finite"):
        chain.encode_frame(broken)


def test_encode_frame_scales_diffusers_float_output():
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    frame = numpy.zeros((2, 2, 3), dtype=numpy.float32)
    frame[0, 0] = 1.0
    decoded = chain.decode_frame(chain.encode_frame(frame))
    assert decoded[0, 0].tolist() == [255, 255, 255]
    assert decoded[1, 1].tolist() == [0, 0, 0]


def test_encode_frame_rejects_a_stack_of_frames():
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    with pytest.raises(ValueError, match="HxWx3"):
        chain.encode_frame(numpy.zeros((17, 4, 4, 3), dtype=numpy.uint8))


# ---------------------------------------------------------------------------
# Segment seed parity with the local controller
# ---------------------------------------------------------------------------


def test_segment_world_seed_matches_the_plumb_controller_exactly():
    """A drift here would silently change every generated frame."""

    from plumb.rollout import _segment_seed

    for seed in (0, 1, 7, 42, 123456789, 2**31 - 1):
        for offset in (0, 1, 4, 7, 16, 70, 100):
            assert chain.segment_world_seed(seed, offset) == _segment_seed(seed, offset)


def test_segment_world_seed_varies_with_offset_and_stays_positive():
    values = [chain.segment_world_seed(99, offset) for offset in range(8)]
    assert len(set(values)) == 8
    assert all(0 <= value < 2**63 for value in values)


# ---------------------------------------------------------------------------
# Controller helpers
# ---------------------------------------------------------------------------


def _rollout_request(world_payload: Dict[str, Any], **stage_payloads: Dict[str, Any]) -> Any:
    def stage(payload: Dict[str, Any]) -> Any:
        return chain.StageRequest(episode_id="ep-1", protocol_hash="ph-1", payload=payload)

    return chain.RolloutRequest(
        run_id="run-1",
        episode_id="ep-1",
        protocol_hash="ph-1",
        policy=stage(stage_payloads.get("policy", {})),
        world=stage(world_payload),
        validity=stage(stage_payloads.get("validity", {})),
        judge=stage(stage_payloads.get("judge", {})),
    )


def test_world_setup_requires_the_protocol_seed():
    complete = {"compatibility_profile_id": "p", "domain": "bridge_orig_lerobot", "seed": 11}
    profile_id, domain, seed, lineage = chain._world_setup(_rollout_request(complete))
    assert (profile_id, domain, seed, lineage) == ("p", "bridge_orig_lerobot", 11, None)

    missing_seed = dict(complete)
    missing_seed.pop("seed")
    assert "world_payload_missing_seed" in chain._world_setup(_rollout_request(missing_seed))

    # A boolean is not a seed, and defaulting to 0 would replace the RNG lineage.
    bool_seed = dict(complete, seed=True)
    assert "world_payload_missing_seed" in chain._world_setup(_rollout_request(bool_seed))

    no_profile = dict(complete)
    no_profile["compatibility_profile_id"] = ""
    assert chain._world_setup(_rollout_request(no_profile)) == "world_payload_missing_compatibility_profile_id"

    no_domain = dict(complete)
    no_domain.pop("domain")
    assert chain._world_setup(_rollout_request(no_domain)) == "world_payload_missing_domain"


def test_select_prefix_blocks_when_no_prefix_is_certified():
    proposal = [[0.0] * 7 for _ in range(4)]
    reason = chain._select_prefix("Octo-Small", proposal, None, remaining=70)
    assert isinstance(reason, str)
    assert "certified_execute_prefix_unresolved" in reason
    assert "never assumes five chunks" in reason or "five chunks" in reason


def test_select_prefix_uses_the_certified_prefix_when_the_horizon_allows():
    proposal = [[0.0] * 7 for _ in range(4)]
    assert chain._select_prefix("OpenPiZero", proposal, 4, remaining=70) == 4
    assert chain._select_prefix("OpenVLA", [[0.0] * 7], 1, remaining=1) == 1


def test_select_prefix_refuses_to_truncate_at_a_non_divisible_horizon():
    proposal = [[0.0] * 7 for _ in range(4)]
    reason = chain._select_prefix("OpenPiZero", proposal, 4, remaining=2)
    assert isinstance(reason, str)
    assert "terminal_prefix_2_not_demonstrated" in reason
    assert "padding" in reason


def test_select_prefix_refuses_a_prefix_longer_than_the_proposal():
    reason = chain._select_prefix("MiniVLA", [[0.0] * 7], 7, remaining=70)
    assert isinstance(reason, str)
    assert "exceeds_proposal" in reason


def test_select_prefix_rejects_an_empty_proposal_and_boolean_prefix():
    assert "empty_proposal" in chain._select_prefix("OpenVLA", [], 1, remaining=5)
    assert "unresolved" in chain._select_prefix("OpenVLA", [[0.0] * 7], True, remaining=5)


def test_accumulate_segment_frames_drops_the_echoed_conditioning_frame():
    output = {
        "frames": [_frame(pixels_sha256=None) for _ in range(5)],
        "conditioning_frame_included": True,
    }
    frames, dropped = chain._accumulate_segment_frames(output, expected_actions=4)
    assert dropped is True
    assert len(frames) == 4


def test_accumulate_segment_frames_keeps_all_frames_when_none_is_echoed():
    output = {"frames": [_frame() for _ in range(4)], "conditioning_frame_included": False}
    frames, dropped = chain._accumulate_segment_frames(output, expected_actions=4)
    assert dropped is False
    assert len(frames) == 4


def test_accumulate_segment_frames_rejects_a_frame_count_mismatch():
    output = {"frames": [_frame() for _ in range(3)], "conditioning_frame_included": True}
    reason = chain._accumulate_segment_frames(output, expected_actions=4)
    assert isinstance(reason, str)
    assert "world_returned_3_frames_expected_5" == reason


def test_accumulate_segment_frames_refuses_to_reuse_the_previous_frame():
    reason = chain._accumulate_segment_frames({"frames": [], "conditioning_frame_included": True}, 4)
    assert isinstance(reason, str)
    assert "will not reuse the previous frame" in reason


def test_next_history_follows_the_declared_observation_window():
    history = [_payload_frame("h0")]
    future = [_payload_frame("f0"), _payload_frame("f1")]
    one = chain._next_history({"contract": {"required_observation_history": 1}}, history, future)
    assert [frame.data for frame in one] == ["f1"]
    two = chain._next_history({"contract": {"required_observation_history": 2}}, history, future)
    assert [frame.data for frame in two] == ["f0", "f1"]
    # A missing or nonsensical contract value falls back to one fresh image.
    assert len(chain._next_history({}, history, future)) == 1
    assert len(chain._next_history({"contract": {"required_observation_history": 0}}, history, future)) == 1


def _payload_frame(tag: str) -> Any:
    return chain.FramePayload(encoding="png_base64", data=tag)


def test_judge_payload_samples_sixteen_frames_from_a_long_clip():
    frames = [_payload_frame("f%02d" % index) for index in range(71)]
    timestamps = [index * 0.2 for index in range(71)]
    request = _rollout_request(
        {"compatibility_profile_id": "p", "domain": "d", "seed": 1},
        judge={
            "task_id": "open_drawer",
            "reference_images": [{"image": _frame(), "source_uri": "s3://p", "sha256": "c" * 64}],
            "seeds": [11, 12, 13, 14, 15],
            "clip_id": "clip-1",
        },
    )
    payload = chain._judge_payload(request, frames, timestamps)
    assert isinstance(payload, dict)
    assert len(payload["frames"]) == 16
    assert payload["frames"][0]["data"] == "f00"
    assert payload["frames"][-1]["data"] == "f70"
    assert payload["frame_timestamps"] == sorted(payload["frame_timestamps"])
    assert payload["clip_id"] == "clip-1"
    # The sampled payload must still satisfy the blinded contract.
    chain.JudgeStagePayload.model_validate(payload)


def test_judge_payload_blocks_on_missing_protocol_inputs():
    frames = [_payload_frame("f%02d" % index) for index in range(20)]
    timestamps = [index * 0.2 for index in range(20)]

    def build(judge: Dict[str, Any]) -> Any:
        return _rollout_request({"compatibility_profile_id": "p", "domain": "d", "seed": 1}, judge=judge)

    references = [{"image": _frame(), "source_uri": "s3://p", "sha256": "d" * 64}]
    assert "judge_task_id_missing" in chain._judge_payload(
        build({"reference_images": references, "seeds": [1, 2, 3, 4, 5]}), frames, timestamps
    )
    assert "judge_reference_images_missing" in chain._judge_payload(
        build({"task_id": "to_sink", "seeds": [1, 2, 3, 4, 5]}), frames, timestamps
    )
    assert "judge_seeds_missing" in chain._judge_payload(
        build({"task_id": "to_sink", "reference_images": references, "seeds": [1, 1, 2, 3, 4]}), frames, timestamps
    )
    assert "judge_seeds_missing" in chain._judge_payload(
        build({"task_id": "to_sink", "reference_images": references}), frames, timestamps
    )


def test_judge_payload_rejects_a_clip_shorter_than_the_rubric_window():
    frames = [_payload_frame("f%02d" % index) for index in range(9)]
    timestamps = [index * 0.2 for index in range(9)]
    request = _rollout_request(
        {"compatibility_profile_id": "p", "domain": "d", "seed": 1},
        judge={
            "task_id": "fold_cloth",
            "reference_images": [{"image": _frame(), "source_uri": "s3://p", "sha256": "e" * 64}],
            "seeds": [1, 2, 3, 4, 5],
        },
    )
    reason = chain._judge_payload(request, frames, timestamps)
    assert isinstance(reason, str)
    assert "judge_clip_too_short" in reason


def test_world_payload_derives_a_per_segment_seed():
    episode = chain.EpisodeControlPayload.model_validate(
        {
            "policy": "OpenVLA",
            "task_id": "open_drawer",
            "prompt": "Open the drawer",
            "horizon_actions": 70,
            "initial_frame": _frame(),
            "initial_state": [0.0] * 8,
            "bridge_control_profile_id": "bridge-v1",
        }
    )
    request = _rollout_request({"compatibility_profile_id": "p", "domain": "bridge_orig_lerobot", "seed": 5})
    first = chain._world_payload(
        episode, request, _payload_frame("c0"), [[0.0] * 10], [0.2], 0,
        profile_id="p", domain="bridge_orig_lerobot", episode_seed=5, action_offset=0, lineage=None,
    )
    second = chain._world_payload(
        episode, request, _payload_frame("c1"), [[0.0] * 10], [0.4], 1,
        profile_id="p", domain="bridge_orig_lerobot", episode_seed=5, action_offset=1, lineage="src-1",
    )
    assert first["seed"] == chain.segment_world_seed(5, 0)
    assert second["seed"] == chain.segment_world_seed(5, 1)
    assert first["seed"] != second["seed"]
    assert "source_state_lineage_id" not in first
    assert second["source_state_lineage_id"] == "src-1"
    chain.WorldStagePayload.model_validate(first)


def test_build_compiler_blocks_external_normalization_without_a_normalizer():
    episode = chain.EpisodeControlPayload.model_validate(
        {
            "policy": "OpenVLA",
            "task_id": "open_drawer",
            "prompt": "Open the drawer",
            "horizon_actions": 70,
            "initial_frame": _frame(),
            "initial_state": [0.0] * 8,
            "bridge_control_profile_id": "bridge-v1",
            "normalization_boundary": "external",
        }
    )
    reason = chain._build_compiler(episode)
    assert isinstance(reason, str)
    assert "will not infer normalizer bounds" in reason


def test_build_compiler_and_state_use_the_real_bridge_adapters():
    episode = chain.EpisodeControlPayload.model_validate(
        {
            "policy": "OpenVLA",
            "task_id": "open_drawer",
            "prompt": "Open the drawer",
            "horizon_actions": 4,
            "initial_frame": _frame(),
            "initial_state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.25],
            "bridge_control_profile_id": "bridge-v1",
        }
    )
    state = chain._build_initial_state(episode)
    assert not isinstance(state, str), state
    compiler = chain._build_compiler(episode)
    assert not isinstance(compiler, str), compiler
    compiled = compiler.compile(state, [(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)])
    assert len(compiled.backend_actions) == 1
    assert len(compiled.backend_actions[0]) == 10, "Cosmos requires 10-D rows"
    assert len(compiled.forecast_states) == 2, "the compiler must return N+1 forecast states"
    assert compiled.normalization_applications == 0, "BACKEND boundary applies the normalizer inside the backend"


def test_build_initial_state_rejects_a_malformed_state():
    episode = chain.EpisodeControlPayload.model_validate(
        {
            "policy": "OpenVLA",
            "task_id": "open_drawer",
            "prompt": "Open the drawer",
            "horizon_actions": 4,
            "initial_frame": _frame(),
            "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
            "bridge_control_profile_id": "bridge-v1",
        }
    )
    reason = chain._build_initial_state(episode)
    assert isinstance(reason, str)
    assert "initial_bridge_state_rejected" in reason


# ---------------------------------------------------------------------------
# GPU-second accounting
# ---------------------------------------------------------------------------


def test_gpu_seconds_labels_allocation_as_unknown_not_zero():
    timings = [
        chain.StageTiming(stage="policy", worker="w", wall_seconds=0.8, declared_gpu_count=1),
        chain.StageTiming(stage="world", worker="w", wall_seconds=5.0, server_seconds=4.5, declared_gpu_count=1),
        chain.StageTiming(stage="judge", worker="w", wall_seconds=30.0, declared_gpu_count=1),
        chain.StageTiming(stage="compile", worker="c", wall_seconds=0.01, declared_gpu_count=0),
        chain.StageTiming(stage="controller", worker="c", wall_seconds=40.0, declared_gpu_count=0),
    ]
    accounting = chain._gpu_seconds(timings)
    assert accounting.policy_gpu_seconds == pytest.approx(0.8)
    # server_seconds wins over wall_seconds when the worker reported it.
    assert accounting.world_gpu_seconds == pytest.approx(4.5)
    assert accounting.judge_gpu_seconds == pytest.approx(30.0)
    assert accounting.total_instrumented_gpu_seconds == pytest.approx(35.3)
    assert accounting.allocated_gpu_seconds is None, "null means unknown, never zero"
    assert accounting.accounting_basis == "instrumented_stage_wall_seconds_times_declared_gpu_count"
    assert any("null means unknown" in note for note in accounting.limitations)
    assert not hasattr(accounting, "estimated_usd")


def test_gpu_seconds_reports_null_when_nothing_was_observed():
    accounting = chain._gpu_seconds(
        [chain.StageTiming(stage="controller", worker="c", wall_seconds=1.0, declared_gpu_count=0)]
    )
    assert accounting.policy_gpu_seconds is None
    assert accounting.world_gpu_seconds is None
    assert accounting.judge_gpu_seconds is None
    assert accounting.total_instrumented_gpu_seconds is None


def test_gpu_seconds_ignores_a_stage_with_no_timing():
    accounting = chain._gpu_seconds(
        [chain.StageTiming(stage="world", worker="w", wall_seconds=None, declared_gpu_count=1)]
    )
    assert accounting.world_gpu_seconds is None


# ---------------------------------------------------------------------------
# Payload builders that the controller sends to the policy workers
# ---------------------------------------------------------------------------


def test_policy_payload_is_a_valid_observation_and_carries_no_actions():
    episode = chain.EpisodeControlPayload.model_validate(
        {
            "policy": "OpenVLA",
            "task_id": "open_drawer",
            "prompt": "Open the drawer",
            "horizon_actions": 70,
            "control_hz": 5.0,
            "initial_frame": _frame(),
            "initial_state": [0.0] * 8,
            "bridge_control_profile_id": "bridge-v1",
            "policy_seed": 3,
        }
    )
    payload = chain._policy_payload(episode, "OpenVLA", [_payload_frame("f0")], action_offset=10)
    parsed = chain.PolicyObservationPayload.model_validate(payload)
    assert parsed.policy == "OpenVLA"
    assert parsed.prompt == "Open the drawer"
    assert parsed.timestamp == pytest.approx(2.0)
    assert parsed.remaining_actions == 60
    assert parsed.policy_seed == 3
    assert "native_actions" not in payload


def test_validity_payload_carries_frames_actions_and_no_judge_fields():
    request = _rollout_request(
        {"compatibility_profile_id": "p", "domain": "d", "seed": 1},
        validity={"task_id": "open_drawer", "parameters_hash": "sha256:" + "a" * 64, "expected_static": False},
    )
    frames = [_payload_frame("f0"), _payload_frame("f1")]
    payload = chain._validity_payload(
        request,
        frames,
        [0.0, 0.2],
        [[0.0] * 7],
        conditioning_frame=frames[0],
        forecast_states=[[0.0] * 8, [0.1] * 8],
        episode_id="ep-1",
    )
    parsed = chain.ValidityStagePayload.model_validate(payload)
    assert parsed.expected_frame_count == 2
    assert parsed.task_id == "open_drawer"
    assert len(parsed.native_actions) == 1
    # The gate needs len(frames) == len(actions) + 1, plus the start frame again
    # as conditioning_frame so it can verify frames[0].
    assert len(parsed.frames) == len(parsed.native_actions) + 1
    assert parsed.conditioning_frame is not None
    assert parsed.forecast_states is not None and len(parsed.forecast_states) == 2
    assert parsed.episode_id == "ep-1"
    assert parsed.expected_static is False
    assert "binary_success" not in payload and "progress" not in payload
    assert "progress_score" not in payload


def test_validity_payload_omits_expected_static_unless_the_protocol_declares_it():
    request = _rollout_request({"compatibility_profile_id": "p", "domain": "d", "seed": 1}, validity={})
    payload = chain._validity_payload(request, [_payload_frame("f0")], [0.0], [])
    assert "expected_static" not in payload, "staticness is a protocol declaration, never inferred from video"


def test_stage_a_keyword_contract_matches_the_real_gate_signature():
    """A drift here would silently block every episode at the validity stage."""

    validity = pytest.importorskip("plumb.validity")
    import inspect

    signature = inspect.signature(validity.StageAValidityGate.evaluate)
    accepted = set(signature.parameters) - {"self"}
    declared = set(chain.ValidityGateCore.CANONICAL_KEYWORDS)
    assert declared <= accepted, "chain declares keywords the gate does not accept: %s" % sorted(declared - accepted)
    required = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    assert required <= declared, "the gate requires keywords chain cannot supply: %s" % sorted(required - declared)


def test_validity_gate_core_binds_the_real_gate():
    pytest.importorskip("plumb.validity")
    core = chain.ValidityGateCore()
    assert core.unresolved == [], core.unresolved
    assert core.ready is True
    assert "actions" in core.accepted_keywords
    assert "frames" in core.accepted_keywords


def test_validity_report_normalises_through_as_dict():
    validity = pytest.importorskip("plumb.validity")
    gate = validity.StageAValidityGate(calibration_class="uncalibrated_development")
    report = gate.evaluate(frames=[], actions=[], nominal_timestamps=[])
    label, reasons, details = chain._normalise_validity_result(report)
    assert label in ("valid", "invalid", "unknown")
    assert "uncalibrated_development_mode" in reasons
    assert "parameters_hash" in details
    assert "validity" not in details and "reason_codes" not in details


# ---------------------------------------------------------------------------
# Collector crash safety
# ---------------------------------------------------------------------------


def test_queue_failure_inside_the_collector_never_hangs_a_caller():
    """A hung episode consumes the run deadline while producing no record."""

    class Hostile:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, payloads: List[Any]) -> List[Any]:
            self.calls += 1
            # Returning a bad-length result set is a contract error, not a crash.
            return ["only-one"]

    executor = Hostile()

    async def scenario() -> List[str]:
        queue = chain.MicroBatchQueue(executor, batch_max=2, batch_window_ms=1)

        async def one(value: int) -> str:
            try:
                await queue.submit("k", value)
            except chain.BatchContractError as error:
                return "contract:%s" % type(error).__name__
            return "unexpected"

        return await asyncio.wait_for(asyncio.gather(*[one(index) for index in range(2)]), timeout=5.0)

    assert asyncio.run(scenario()) == ["contract:BatchContractError", "contract:BatchContractError"]
    assert executor.calls == 1


# ---------------------------------------------------------------------------
# JSON coercion of adapter output
# ---------------------------------------------------------------------------


def test_json_safe_unwraps_enums_and_nulls_non_finite_floats():
    import enum
    import json

    class Label(str, enum.Enum):
        VALID = "valid"

    class Count(enum.IntEnum):
        TWO = 2

    coerced = chain._json_safe(
        {
            "label": Label.VALID,
            "count": Count.TWO,
            "nan": float("nan"),
            "inf": float("inf"),
            "tuple": (1, 2.5),
            "set": {3},
            "nested": {"ok": True, "none": None},
        }
    )
    assert coerced["label"] == "valid"
    assert type(coerced["label"]) is str, "an Enum object must not leak into a JSON payload"
    assert coerced["count"] == 2
    assert coerced["nan"] is None, "null means unknown, never zero"
    assert coerced["inf"] is None
    assert coerced["tuple"] == [1, 2.5]
    assert coerced["set"] == [3]
    assert coerced["nested"] == {"ok": True, "none": None}
    json.dumps(coerced)


def test_json_safe_uses_as_dict_when_an_adapter_provides_one():
    class Report:
        def as_dict(self):
            return {"wall_seconds": 1.5, "gpu_peak_memory_bytes": None}

    assert chain._json_safe(Report()) == {"wall_seconds": 1.5, "gpu_peak_memory_bytes": None}


def test_validity_result_normalisation_accepts_mappings_and_objects():
    assert chain._normalise_validity_result({"validity": "valid", "reason_codes": ["ok"]})[:2] == ("valid", ["ok"])

    class Result:
        validity = "unknown"
        reason_codes = ("ambiguous_motion",)

    assert chain._normalise_validity_result(Result())[:2] == ("unknown", ["ambiguous_motion"])


def test_validity_result_normalisation_rejects_anything_outside_three_labels():
    for bad in ({"validity": "probably"}, {"validity": True}, {"validity": None}, {"validity": 1}):
        with pytest.raises(ValueError, match="valid"):
            chain._normalise_validity_result(bad)


# ---------------------------------------------------------------------------
# Optional: full SDK validation.
#
# Always skipped in the project venv, which has no truss_chains by design. It
# runs in the dedicated deploy venv (`pip install truss==0.18.30`) and is the
# strongest check available without an actual push: it constructs all nine
# RemoteConfigs and runs the framework's own endpoint/IO validator.
# ---------------------------------------------------------------------------


def test_chainlets_pass_the_real_framework_validator():
    pytest.importorskip("truss_chains", reason="only runs in the dedicated deploy venv")

    # Reuse the module-level load: importing the same source a second time would
    # re-register every Chainlet and the framework registry rejects duplicates.
    module = chain

    from truss_chains import framework

    # Collected (not raised) during class creation; this surfaces them.
    framework.raise_validation_errors()

    assert module.CHAINS_RUNTIME_AVAILABLE is True
    descriptor = framework.get_descriptor(module.RolloutController)
    assert descriptor.endpoint.is_async
    assert [arg.type.raw.__name__ for arg in descriptor.endpoint.input_args] == ["RolloutRequest"]
    assert [out.raw.__name__ for out in descriptor.endpoint.output_types] == ["RolloutResult"]
    # Every dependency must be retries=0: a transport retry is an attempt, not a
    # new statistical episode, and attempt accounting belongs to the ledger.
    assert set(dependency.options.retries for dependency in descriptor.dependencies.values()) == {0}
    assert len(descriptor.dependencies) == 8

    world = module.WorldWorker.remote_config
    assert world.get_compute_spec().predict_concurrency == module.BATCH_MAX
    assert str(world.get_compute_spec().accelerator.accelerator).endswith("H100")
    assert sorted(world.get_asset_spec().secrets) == ["hf_access_token"]
    cached = world.get_asset_spec().cached
    assert [repo.repo_id for repo in cached] == ["nvidia/Cosmos3-Nano"]
    assert cached[0].revision == "e59a53c25979a090fa8706c9acc0c254a6e89b92"
    assert cached[0].use_volume is True

    judge = module.JudgeWorker.remote_config
    assert str(judge.get_compute_spec().accelerator.accelerator).endswith("H100"), (
        "the judge is GPU: cluster evidence records a 16.9 GB peak for Qwen2.5-VL"
    )
    validity = module.ValidityWorker.remote_config
    assert validity.get_compute_spec().accelerator.accelerator is None, "Stage A stays CPU-only"
    controller = module.RolloutController.remote_config
    assert controller.get_compute_spec().accelerator.accelerator is None

    # Nine distinct images, one per incompatible dependency stack.
    names = set()
    requirement_files = set()
    for class_name in module.WORKER_REQUIREMENTS_FILES:
        config = getattr(module, class_name).remote_config
        names.add(config.name)
        image = config.docker_image
        path = getattr(image, "requirements_file", None) or getattr(image, "pip_requirements_file", None)
        assert path is not None, "%s has no requirements file" % class_name
        requirement_files.add(path.abs_path)
    assert len(names) == 9
    assert len(requirement_files) == 9


# ---------------------------------------------------------------------------
# Certified-adapter awareness
# ---------------------------------------------------------------------------


def test_certified_adapter_requirements_are_discovered_from_live_classes():
    policies = pytest.importorskip("plumb.policies")

    for arm in ("Octo-Small", "Octo-Base", "MiniVLA", "OpenPiZero", "SuSIE", "SuSIE_LL"):
        requirements = chain.certified_adapter_requirements(policies, arm)
        assert requirements is not None, "%s has a certified adapter in plumb.policies" % arm
        assert requirements["adapter"].startswith("plumb.policies.")
        assert requirements["required_profile_fields"], "the required profile fields must be enumerated"
        assert "profile_id" in requirements["required_profile_fields"]
        assert "will not invent them" in requirements["why_the_chain_does_not_construct_it"]

    # OpenVLA is loaded through its own bundled adapter, not this table.
    assert chain.certified_adapter_requirements(policies, "OpenVLA") is None
    assert chain.certified_adapter_requirements(policies, "Nonexistent") is None


def test_certified_adapter_table_covers_every_non_openvla_arm():
    covered = set(chain.NATIVE_POLICY_CERTIFIED_ADAPTERS)
    assert covered == set(chain.CANONICAL_POLICY_ARMS) - {"OpenVLA"}


def test_blocked_policy_arms_name_the_certified_adapter_and_its_missing_evidence():
    pytest.importorskip("plumb.policies")
    core = chain.NativePolicyCore("plumb-susie-worker", "policy-susie.txt")
    assert core.arms == ("SuSIE", "SuSIE_LL")
    for arm in core.arms:
        reasons = core.blocked_reasons.get(arm)
        assert reasons, "%s must be blocked: no source-reviewed loader exists" % arm
        joined = " ".join(reasons)
        assert "no source-reviewed local loader" in joined
        assert "A certified loader exists as plumb.policies." in joined
        assert "action normalizer" in joined
        assert core.certified_adapter[arm]["required_profile_fields"]


def test_openvla_core_blocks_without_the_remote_code_acknowledgement(monkeypatch):
    pytest.importorskip("plumb.policies")
    monkeypatch.delenv("PLUMB_OPENVLA_REVIEWED_REMOTE_CODE_ACK", raising=False)
    core = chain.NativePolicyCore("plumb-openvla-worker", "policy-openvla.txt")
    reasons = core.blocked_reasons.get("OpenVLA")
    assert reasons, "trust_remote_code must not be enabled without a recorded review"
    assert "47a0ec7fc4ec123775a391911046cf33cf9ed83f" in " ".join(reasons)
    assert "OpenVLA" not in core.adapters


def test_policy_core_records_its_runtime_lock_digest():
    pytest.importorskip("plumb.policies")
    core = chain.NativePolicyCore("plumb-octo-worker", "policy-octo.txt")
    assert core.runtime_lock_path.name == "policy-octo.txt"
    assert core.runtime_lock_sha256 is not None
    assert core.runtime_lock_sha256.startswith("sha256:")


# ---------------------------------------------------------------------------
# Stage-A calibration class
# ---------------------------------------------------------------------------


def test_validity_core_defaults_to_the_primary_calibration_class():
    pytest.importorskip("plumb.validity")
    core = chain.ValidityGateCore()
    assert chain.VALIDITY_CALIBRATION_CLASS == "calibrated_primary"
    assert core.calibration_class == "calibrated_primary"
    assert core.ready is True
    # A real parameter pin from the gate's own declared parameters.
    assert core.parameters_hash is not None and core.parameters_hash.startswith("sha256:")
    # No MotionReference is mounted, which the worker records rather than hides.
    assert core.motion_reference_missing is True


def test_validity_core_accepts_the_development_class_which_self_labels():
    validity = pytest.importorskip("plumb.validity")
    core = chain.ValidityGateCore(calibration_class="uncalibrated_development")
    assert core.ready is True
    assert core.calibration_class == "uncalibrated_development"
    report = core.gate.evaluate(frames=[], actions=[], nominal_timestamps=[])
    # Safe to expose only because the gate itself marks such reports ineligible.
    assert report.primary_scoring_eligible is False
    assert validity.ValidityReasonCode.UNCALIBRATED_DEVELOPMENT_MODE.value in report.reason_codes


def test_validity_core_blocks_on_an_unknown_calibration_class():
    pytest.importorskip("plumb.validity")
    core = chain.ValidityGateCore(calibration_class="whatever_sounds_good")
    assert core.ready is False
    assert core.gate is None
    assert "calibration_class must be one of" in " ".join(core.unresolved)


def test_validity_gate_returns_a_real_verdict_for_a_real_clip():
    """Not blocked: plumb.validity landed, so this stage genuinely runs."""

    numpy = pytest.importorskip("numpy")
    pytest.importorskip("plumb.validity")
    core = chain.ValidityGateCore()
    frames = [numpy.zeros((64, 64, 3), dtype=numpy.uint8) for _ in range(3)]
    label, reasons, details = core.evaluate(
        frames=frames,
        actions=[[0.0] * 7, [0.0] * 7],
        nominal_timestamps=[0.0, 0.2, 0.4],
        conditioning_frame=frames[0],
        states=[[0.0] * 8, [0.0] * 8, [0.0] * 8],
        expected_static=True,
        episode_id="ep-1",
    )
    assert label in ("valid", "invalid", "unknown")
    assert isinstance(reasons, list)
    # The evidence record travels with the verdict.
    assert details.get("parameters_hash")
    assert details.get("primary_scoring_eligible") in (True, False)
    assert "no_motion_reference" in reasons, "an unmounted motion reference must be named, not assumed away"


# ---------------------------------------------------------------------------
# Static guards for the branch that never executes locally.
#
# Everything under `if CHAINS_RUNTIME_AVAILABLE:` is unreachable in this venv, so
# a wrong keyword there would only surface on a deployed replica. These two tests
# read the source instead of running it.
# ---------------------------------------------------------------------------


def _chain_ast():
    import ast

    return ast.parse(CHAIN_PATH.read_text(encoding="utf-8"))


def test_pydantic_constructions_only_use_declared_fields():
    """A stray keyword on an extra="forbid" model is a runtime ValidationError."""

    import ast

    models = {
        name: getattr(chain, name)
        for name in dir(chain)
        if isinstance(getattr(chain, name, None), type)
        and hasattr(getattr(chain, name), "model_fields")
        and getattr(chain, name).__module__ == chain.__name__
    }
    assert "RolloutResult" in models and "StageResult" in models, sorted(models)

    problems = []
    for node in ast.walk(_chain_ast()):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        model = models.get(node.func.id)
        if model is None:
            continue
        declared = set(model.model_fields)
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            if keyword.arg not in declared:
                problems.append(
                    "line %d: %s(%s=...) is not a declared field; declared: %s"
                    % (node.lineno, node.func.id, keyword.arg, sorted(declared))
                )
    assert not problems, "\n".join(problems)


def test_local_calls_only_use_declared_keywords():
    """Catches a call-site keyword that no longer matches its definition."""

    import ast

    tree = _chain_ast()
    signatures = {}
    duplicated = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = node.args
        accepted = {argument.arg for argument in arguments.args}
        accepted |= {argument.arg for argument in arguments.posonlyargs}
        accepted |= {argument.arg for argument in arguments.kwonlyargs}
        entry = (accepted, arguments.kwarg is not None)
        if node.name in signatures and signatures[node.name] != entry:
            duplicated.add(node.name)
        signatures[node.name] = entry

    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        if name in duplicated or name not in signatures:
            continue
        accepted, takes_kwargs = signatures[name]
        if takes_kwargs:
            continue
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            if keyword.arg not in accepted:
                problems.append(
                    "line %d: %s(%s=...) is not a parameter of the local definition; accepts: %s"
                    % (node.lineno, name, keyword.arg, sorted(accepted))
                )
    assert not problems, "\n".join(problems)


def test_terminal_helper_and_result_agree_on_progress_score():
    """The exact drift this pair of guards was written to catch."""

    import ast

    fields = set(chain.RolloutResult.model_fields)
    assert "progress_score" in fields and "progress" not in fields
    for node in ast.walk(_chain_ast()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_terminal":
            names = {argument.arg for argument in node.args.args} | {
                argument.arg for argument in node.args.kwonlyargs
            }
            assert "progress_score" in names, "the helper must use the model's field name"
            assert "progress" not in names
            return
    raise AssertionError("_terminal helper not found in chain.py")


# ---------------------------------------------------------------------------
# Adapter cores off-GPU: they must bind the real adapters and block precisely,
# never crash and never fabricate.
# ---------------------------------------------------------------------------


def test_world_core_binds_the_real_cosmos_adapter_and_blocks_on_absent_weights():
    pytest.importorskip("plumb.adapters.worlds")
    core = chain.WorldAdapterCore()
    assert core.ready is False
    assert core.variant.variant_id == "cosmos3_nano"
    assert core.profile_id == "plumb-cosmos3_nano-fd-r256"
    assert core.local_model_path == "/app/model_cache/cosmos3-nano"
    # The adapter was constructed; only the mounted weights are missing.
    assert core.adapter is not None
    assert core.profile is not None
    assert core.profile.model_revision == "e59a53c25979a090fa8706c9acc0c254a6e89b92"
    reason = " ".join(core.unresolved)
    assert "local model directory is not present" in reason
    assert "no Hub download was attempted" in reason
    assert core.runtime_lock_sha256 is not None and core.runtime_lock_sha256.startswith("sha256:")


def test_world_core_batch_key_separates_incompatible_shapes():
    pytest.importorskip("plumb.adapters.worlds")
    core = chain.WorldAdapterCore()
    base = core.batch_key(16, 10, "bridge_orig_lerobot")
    assert base == core.batch_key(16, 10, "bridge_orig_lerobot")
    assert base != core.batch_key(15, 10, "bridge_orig_lerobot"), "action count changes tensor shape"
    assert base != core.batch_key(16, 7, "bridge_orig_lerobot"), "action width changes tensor shape"
    assert base != core.batch_key(16, 10, "other_domain"), "domain changes the serializer"
    assert core.profile_id in base


def test_judge_core_binds_the_real_judge_and_blocks_on_absent_weights():
    pytest.importorskip("plumb.policies")
    core = chain.JudgeCore()
    assert core.ready is False
    assert core.local_model_path == "/app/model_cache/qwen2p5-vl-7b-instruct"
    assert core.judge is not None, "the judge object must be constructed, which validates the frozen sampling"
    assert core.request_cls is not None and core.reference_cls is not None
    # The frozen primary protocol, enforced inside plumb.policies.judge.
    sampling = core.judge.sampling
    assert (sampling.sample_count, sampling.quorum) == (5, 3)
    assert (sampling.temperature, sampling.top_p) == (0.7, 1.0)
    assert sampling.max_new_tokens == 512
    assert sampling.retries_per_sample == 1
    assert core.judge.profile.model_revision == "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    assert core.judge.profile.trust_remote_code is False
    assert core.judge.profile.local_files_only is True
    assert "local model directory is absent" in " ".join(core.unresolved)


def test_judge_core_reports_gate_d_unavailable_without_an_asset_manifest():
    pytest.importorskip("plumb.policies")
    core = chain.JudgeCore()
    # No assets.lock.json exists, so the profile carries no manifest identity and
    # every report is correctly Gate-D unavailable rather than silently eligible.
    assert core.judge.profile.asset_manifest_id is None
    assert core.judge.profile.asset_manifest_sha256 is None
    # The runtime lock IS real: a digest of the file shipped in the image.
    assert core.judge.profile.runtime_lock_id == "requirements/judge-qwen.txt"
    assert core.judge.profile.runtime_lock_sha256 == core.runtime_lock_sha256


@requires_no_sdk
def test_openvla_core_constructs_the_adapter_once_the_review_is_recorded(monkeypatch):
    # Re-loads the module to pick up the env var, which would duplicate-register
    # every Chainlet if truss_chains were installed; hence the no-SDK gate.
    pytest.importorskip("plumb.policies")
    monkeypatch.setenv("PLUMB_OPENVLA_REVIEWED_REMOTE_CODE_ACK", "47a0ec7fc4ec123775a391911046cf33cf9ed83f")
    module = _load_chain()
    try:
        core = module.NativePolicyCore("plumb-openvla-worker", "policy-openvla.txt")
        assert "OpenVLA" in core.adapters, "the adapter must build once the remote-code review is recorded"
        profile = core.adapters["OpenVLA"].profile
        assert profile.allow_trust_remote_code is True
        assert profile.reviewed_remote_code_revision == profile.remote_code_revision
        assert profile.transformers_version == "4.40.1"
        assert profile.unnorm_key == "bridge_orig"
        assert profile.local_files_only is True
        # Only the absent mounted checkpoint blocks it now.
        assert "local checkpoint directory is absent" in " ".join(core.blocked_reasons["OpenVLA"])
        contract = core.contract_summary("OpenVLA")
        assert contract["certified_execute_prefix"] == 1
        assert contract["native_proposal_horizon"] == 1
        assert contract["requires_proprio"] is False
    finally:
        sys.modules.pop("plumb_deploy_chain", None)


def test_queue_never_strands_an_in_flight_ticket_when_the_collector_dies():
    """take_ready removes tickets from the planner, so a mid-execution crash
    would otherwise leave their callers awaiting until the run deadline."""

    class ExplodingScatter(Exception):
        pass

    def executor(payloads: List[Any]) -> List[Any]:
        return list(payloads)

    async def scenario() -> List[str]:
        queue = chain.MicroBatchQueue(executor, batch_max=2, batch_window_ms=1)

        # Break the resolution step itself, after take_ready has already handed
        # the tickets to _execute.
        def exploding_scatter(tickets, outcomes):
            raise ExplodingScatter("scatter blew up after the batch was taken")

        monkey = chain.MicroBatchPlanner.scatter
        chain.MicroBatchPlanner.scatter = staticmethod(exploding_scatter)
        try:

            async def one(value: int) -> str:
                try:
                    await queue.submit("k", value)
                except ExplodingScatter:
                    return "failed-not-hung"
                except BaseException as error:  # noqa: BLE001
                    return "other:%s" % type(error).__name__
                return "unexpected success"

            return await asyncio.wait_for(asyncio.gather(*[one(index) for index in range(2)]), timeout=5.0)
        finally:
            chain.MicroBatchPlanner.scatter = monkey

    assert asyncio.run(scenario()) == ["failed-not-hung", "failed-not-hung"]


def test_queue_clears_in_flight_tracking_after_a_normal_batch():
    def executor(payloads: List[Any]) -> List[Any]:
        return [payload + 1 for payload in payloads]

    async def scenario() -> Any:
        queue = chain.MicroBatchQueue(executor, batch_max=2, batch_window_ms=1)
        results = await asyncio.gather(queue.submit("k", 1), queue.submit("k", 2))
        return results, list(queue._inflight)

    results, inflight = asyncio.run(scenario())
    assert results == [2, 3]
    assert inflight == [], "a completed batch must not stay marked in flight"


def test_queue_records_a_collector_failure_without_orphaning_the_task_exception():
    """The error reaches every caller, so re-raising would only add log noise."""

    class Boom(Exception):
        pass

    def executor(payloads: List[Any]) -> List[Any]:
        return list(payloads)

    async def scenario():
        queue = chain.MicroBatchQueue(executor, batch_max=2, batch_window_ms=1)

        def exploding_scatter(tickets, outcomes):
            raise Boom("collector died")

        original = chain.MicroBatchPlanner.scatter
        chain.MicroBatchPlanner.scatter = staticmethod(exploding_scatter)
        try:
            try:
                await asyncio.wait_for(queue.submit("k", 1), timeout=5.0)
            except Boom:
                delivered = True
            else:
                delivered = False
        finally:
            chain.MicroBatchPlanner.scatter = original
        # Let the drain task settle before inspecting it.
        await asyncio.sleep(0)
        return delivered, queue.last_drain_error, queue._drain_task

    delivered, recorded, task = asyncio.run(scenario())
    assert delivered, "the caller must receive the failure, not hang"
    assert isinstance(recorded, Boom), "the failure is recorded for diagnostics"
    # The task returned rather than raising, so asyncio has no unretrieved
    # exception to warn about in the deployment logs.
    assert task.done()
    assert task.exception() is None


def test_queue_recovers_after_a_collector_failure():
    calls = {"n": 0}

    def executor(payloads: List[Any]) -> List[Any]:
        calls["n"] += 1
        return [payload * 2 for payload in payloads]

    async def scenario():
        queue = chain.MicroBatchQueue(executor, batch_max=2, batch_window_ms=1)

        def exploding_scatter(tickets, outcomes):
            raise RuntimeError("first collector dies")

        original = chain.MicroBatchPlanner.scatter
        chain.MicroBatchPlanner.scatter = staticmethod(exploding_scatter)
        try:
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(queue.submit("k", 1), timeout=5.0)
        finally:
            chain.MicroBatchPlanner.scatter = original
        await asyncio.sleep(0)
        # A fresh submit must start a new collector and succeed.
        return await asyncio.wait_for(queue.submit("k", 21), timeout=5.0)

    assert asyncio.run(scenario()) == 42
    assert calls["n"] == 2


def test_missing_reason_uses_the_stage_name_status_order_the_consumer_expects():
    """plumb/backends/baseten.py composes `stage_<name>_<status>`; match it."""

    import ast
    import re

    source = CHAIN_PATH.read_text(encoding="utf-8")
    composed = re.findall(r'"stage_(policy|world|validity|judge)_%s"', source)
    assert set(composed) == {"policy", "world", "validity", "judge"}
    # And no leftover reversed form.
    assert not re.search(r'"%s_stage_(policy|world|validity|judge)"', source)
    ast.parse(source)
