"""Capacity planning: which limit binds, and refusing to answer when it cannot."""
from __future__ import annotations

import pytest

from plumb.capacity import (
    NATIVE_FEEDBACK,
    TARGET_SECONDS,
    CapacityError,
    DeploymentCapacity,
    MeasuredLatencies,
    chunked_prefix,
    fallback_episode_count,
    plan_capacity,
    required_world_latency,
    smallest_sufficient_cap,
    sweep_replicas,
)


def _lat(**kw):
    values = {"policy_seconds": 0.25, "world_seconds": 0.30}
    values.update(kw)
    return MeasuredLatencies(**values)


def test_latencies_must_be_positive_and_finite():
    with pytest.raises(CapacityError, match="positive"):
        MeasuredLatencies(policy_seconds=0.0, world_seconds=1.0)
    with pytest.raises(CapacityError, match="finite"):
        MeasuredLatencies(policy_seconds=1.0, world_seconds=1.0, judge_seconds=float("inf"))


def test_an_estimate_is_never_reported_as_measured():
    assert _lat().measured is False
    assert _lat(source="gate_a_report_937372").measured is True
    plan = plan_capacity(_lat(), DeploymentCapacity(max_replicas=10), NATIVE_FEEDBACK)
    assert any("estimates" in w for w in plan.warnings)


def test_replicas_cannot_beat_the_per_episode_latency_floor():
    """The central finding: at native cadence the cap is not the lever."""

    slow = _lat(policy_seconds=0.836, world_seconds=4.52)
    plans = sweep_replicas(slow, NATIVE_FEEDBACK, [1, 10, 100, 1000])
    floors = {p.episode_latency_seconds for p in plans}
    assert len(floors) == 1, "the per-episode floor must not move with replica count"
    assert plans[0].episode_latency_seconds > TARGET_SECONDS
    # No cap is sufficient when one episode alone exceeds the window.
    assert smallest_sufficient_cap(slow, NATIVE_FEEDBACK) is None


def test_a_sufficient_cap_of_none_means_impossible_not_merely_large():
    slow = _lat(policy_seconds=0.836, world_seconds=4.52)
    plan = plan_capacity(slow, DeploymentCapacity(max_replicas=100), NATIVE_FEEDBACK)
    assert plan.replicas_needed_for_target() is None
    assert plan.episodes_achievable_in_target() == 0


def test_the_executed_prefix_changes_the_regime_not_the_hardware():
    fast = _lat()
    native = plan_capacity(fast, DeploymentCapacity(max_replicas=10), NATIVE_FEEDBACK)
    chunked = plan_capacity(fast, DeploymentCapacity(max_replicas=10), chunked_prefix(16))
    assert chunked.rounds_per_episode < native.rounds_per_episode
    assert chunked.episode_latency_seconds < native.episode_latency_seconds
    # Identical hardware; only Gate B's decision differs.
    assert chunked.capacity == native.capacity


def test_rounds_are_ceiled_per_task_not_on_the_mean():
    """A 100-tick task at chunk 16 is 7 rounds, not 6.25."""

    plan = plan_capacity(_lat(), DeploymentCapacity(max_replicas=1), chunked_prefix(16))
    # 70->5, 70->5, 100->7, 100->7, 80->5  => mean 5.8
    assert plan.rounds_per_episode == pytest.approx(5.8)


def test_the_slowest_task_sets_the_floor_not_the_average():
    plan = plan_capacity(_lat(), DeploymentCapacity(max_replicas=1), chunked_prefix(16))
    # worst task is 7 rounds, so the floor uses 7, not the 5.8 mean
    assert plan.episode_latency_seconds == pytest.approx(7 * (0.25 + 0.30))


def test_batch_size_is_capped_by_measured_memory_and_says_so():
    """A recorded 36.5 GB peak means a batch of 16 does not fit in 80 GB."""

    lat = _lat(world_peak_memory_bytes=int(36.52 * 1024**3))
    plan = plan_capacity(
        lat, DeploymentCapacity(max_replicas=10, world_batch_size=16), chunked_prefix(16)
    )
    assert plan.memory_batch_ceiling == 2
    assert any("exceeds the 2" in w for w in plan.warnings)


def test_an_unmeasured_gpu_peak_makes_the_batch_unchecked_and_warns():
    plan = plan_capacity(
        _lat(), DeploymentCapacity(max_replicas=10, world_batch_size=16), chunked_prefix(16)
    )
    assert plan.memory_batch_ceiling is None
    assert any("unchecked against memory" in w for w in plan.warnings)


def test_batched_throughput_assumption_is_disclosed():
    plan = plan_capacity(
        _lat(), DeploymentCapacity(max_replicas=10, world_batch_size=4), chunked_prefix(16)
    )
    assert any("fused forward" in w for w in plan.warnings)


def test_the_binding_limit_is_named():
    latency_bound = plan_capacity(
        _lat(policy_seconds=5.0, world_seconds=5.0),
        DeploymentCapacity(max_replicas=100000),
        NATIVE_FEEDBACK,
    )
    assert latency_bound.binding_limit == "per_episode_latency"
    throughput_bound = plan_capacity(
        _lat(), DeploymentCapacity(max_replicas=1), chunked_prefix(16)
    )
    assert throughput_bound.binding_limit == "aggregate_throughput"


def test_required_latency_refuses_when_policy_time_alone_blows_the_window():
    result = required_world_latency(
        DeploymentCapacity(max_replicas=100), NATIVE_FEEDBACK, policy_seconds=1.0
    )
    assert result["feasible"] is False
    assert result["required_world_seconds"] is None
    assert "policy latency alone" in result["reason"]


def test_required_latency_reports_the_tighter_of_the_two_budgets():
    result = required_world_latency(
        DeploymentCapacity(max_replicas=1, world_batch_size=2), chunked_prefix(16), policy_seconds=0.25
    )
    assert result["required_world_seconds"] == pytest.approx(
        min(result["latency_limited_budget_seconds"], result["throughput_limited_budget_seconds"])
    )
    assert result["binding_limit"] == "aggregate_throughput"


def test_the_study_is_never_reduced_only_the_live_burst():
    plan = plan_capacity(
        _lat(policy_seconds=0.836, world_seconds=4.52),
        DeploymentCapacity(max_replicas=1),
        chunked_prefix(16),
    )
    fallback = fallback_episode_count(plan)
    assert fallback["full_study_episodes"] == 1500
    assert fallback["study_is_reduced"] is False
    assert fallback["live_is_a_subset"] is True
    assert "still runs in full" in fallback["reason"]
    assert "never present a subset as the full study" in fallback["note"].lower()


def test_an_executed_prefix_below_one_tick_is_rejected():
    with pytest.raises(CapacityError, match="at least 1 control tick"):
        chunked_prefix(0)


def test_capacity_rejects_impossible_configurations():
    with pytest.raises(CapacityError, match="max_replicas"):
        DeploymentCapacity(max_replicas=0)
    with pytest.raises(CapacityError, match="world_batch_size"):
        DeploymentCapacity(max_replicas=1, world_batch_size=0)


def test_plan_serialises_json_safely_and_names_its_assumptions():
    import json

    plan = plan_capacity(_lat(), DeploymentCapacity(max_replicas=10), chunked_prefix(16))
    payload = plan.to_mapping()
    json.dumps(payload, allow_nan=False)
    assert payload["feedback_qualified"] is False
    assert payload["latencies_are_measured"] is False
    assert payload["binding_limit"] in ("per_episode_latency", "aggregate_throughput")
