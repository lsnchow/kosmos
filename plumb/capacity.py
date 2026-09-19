"""Capacity planning for the burst: what is actually achievable, and why.

The project's public target is 1,500 episodes in 60 seconds for about $11, with
"100 replicas" recorded in the spec as a *capacity hypothesis*.  That hypothesis
was formed before anyone worked out which limit binds, and the arithmetic says it
is usually the wrong knob.

There are two independent limits, and the slower one wins:

**Per-episode latency.**  Spec section 7: each episode's policy/world feedback is
sequential; only episodes parallelise.  So one episode takes
``rounds x (policy_latency + world_latency)``, where ``rounds`` is the number of
times the controller must stop and ask the policy what to do.  **No number of
replicas reduces this.**  If one episode takes 90 seconds, 1,500 episodes cannot
finish in 60.

**Aggregate throughput.**  Total world-model GPU-seconds divided by how much
parallel capacity exists.  Replicas help here, and so does in-container batch
packing, which spec section 7 names explicitly.

Which limit binds is decided by Gate B, not by the account cap:

* At OpenVLA's **native cadence** — one action per fresh generated frame — a
  70-to-100-tick task is 70 to 100 sequential rounds.  The per-episode floor is
  large and replicas barely matter.
* Under a **chunked-prefix approximation**, the controller consumes a whole
  action chunk per round, so a 100-tick task is ~7 rounds.  The floor collapses
  and modest capacity suffices.

This module computes both limits from measured inputs and reports which one
binds.  It deliberately requires latencies to be passed in: there is no default,
because a made-up latency produces a made-up plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .protocol import TASK_HORIZONS

#: Spec section 7's operational definitions of the public claim.
TARGET_EPISODES = 1500
TARGET_SECONDS = 60.0
TARGET_USD = 11.25


class CapacityError(ValueError):
    """A capacity plan cannot be computed from the supplied measurements."""


@dataclass(frozen=True)
class MeasuredLatencies:
    """Latencies measured on the actual deployment, per Gate A.

    ``source`` names where each number came from.  A plan built from estimates
    is still useful for deciding what to try, but it must never be reported as a
    measured result, so the label travels with it.
    """

    policy_seconds: float
    world_seconds: float
    judge_seconds: float = 0.0
    validity_seconds: float = 0.0
    source: str = "estimate"
    world_peak_memory_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("policy_seconds", "world_seconds", "judge_seconds", "validity_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise CapacityError("%s must be a finite non-negative number" % name)
        if self.policy_seconds <= 0 or self.world_seconds <= 0:
            raise CapacityError("policy and world latency must be positive to plan capacity")

    @property
    def measured(self) -> bool:
        return self.source not in ("estimate", "", "unknown")


@dataclass(frozen=True)
class DeploymentCapacity:
    """What the account and the hardware actually allow."""

    max_replicas: int
    #: Requests fused into one diffusion forward pass. Bounded by GPU memory,
    #: not by preference: see ``max_batch_for_memory``.
    world_batch_size: int = 1
    gpu_memory_bytes: int = 80 * 1024**3
    #: Controller/policy concurrency. Usually not the bottleneck, but a low value
    #: silently caps how many episodes can be in flight.
    max_concurrent_episodes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_replicas < 1:
            raise CapacityError("max_replicas must be at least 1")
        if self.world_batch_size < 1:
            raise CapacityError("world_batch_size must be at least 1")

    def max_batch_for_memory(self, per_call_peak_bytes: Optional[int]) -> Optional[int]:
        """Largest batch the GPU can hold, or ``None`` if peak memory is unknown.

        Batch packing is the lever the 60-second claim depends on, and it is the
        one most easily over-assumed: a recorded 36.5 GB peak for a single call
        means a batch of 16 does not fit on an 80 GB card, whatever the config
        file says.
        """

        if not per_call_peak_bytes or per_call_peak_bytes <= 0:
            return None
        return max(1, int(self.gpu_memory_bytes // per_call_peak_bytes))


@dataclass(frozen=True)
class FeedbackProfile:
    """How often the controller must stop and re-ask the policy.

    ``executed_prefix`` is the number of control ticks consumed per policy call.
    It is a property of the policy wrapper certified by Gate B, never a tuning
    knob: setting it higher than what the wrapper certifies is the silent
    protocol substitution spec section 10 forbids.
    """

    name: str
    executed_prefix: int
    qualified: bool
    note: str = ""

    def __post_init__(self) -> None:
        if self.executed_prefix < 1:
            raise CapacityError("executed_prefix must be at least 1 control tick")

    def rounds_for(self, horizon_ticks: int) -> int:
        return math.ceil(horizon_ticks / self.executed_prefix)


#: The two feedback regimes Gate B decides between.
NATIVE_FEEDBACK = FeedbackProfile(
    name="native_feedback",
    executed_prefix=1,
    qualified=True,
    note="One action per fresh generated frame. The policy's real contract; no approximation.",
)


def chunked_prefix(chunk: int) -> FeedbackProfile:
    """A chunked-prefix approximation of ``chunk`` ticks per policy call."""

    return FeedbackProfile(
        name="chunked_prefix_approximation",
        executed_prefix=chunk,
        qualified=False,
        note=(
            "Consumes a whole chunk per policy call. Spec section 2 calls this an "
            "approximation even if the suffix-causality test passes; it carries its own "
            "protocol identity and its residual sensitivity must be recorded."
        ),
    )


@dataclass
class CapacityPlan:
    """The computed plan, with the binding limit named."""

    episodes: int
    feedback: FeedbackProfile
    latencies: MeasuredLatencies
    capacity: DeploymentCapacity
    rounds_per_episode: float
    episode_latency_seconds: float
    world_calls: int
    batched_world_calls: int
    world_gpu_seconds: float
    throughput_seconds: float
    wall_seconds: float
    binding_limit: str
    memory_batch_ceiling: Optional[int]
    warnings: List[str] = field(default_factory=list)

    @property
    def meets_target(self) -> bool:
        return self.wall_seconds <= TARGET_SECONDS

    def replicas_needed_for_target(self) -> Optional[int]:
        """Replicas needed to hit 60 s, or ``None`` if latency makes it impossible.

        Returning ``None`` rather than a large number is the point: when the
        per-episode floor already exceeds the target, more capacity buys nothing
        and quoting a replica count would be misleading.
        """

        if self.episode_latency_seconds > TARGET_SECONDS:
            return None
        if self.world_gpu_seconds <= 0:
            return 1
        return max(1, math.ceil(self.world_gpu_seconds / TARGET_SECONDS))

    def episodes_achievable_in_target(self) -> int:
        """How many episodes *do* fit in 60 s at this capacity."""

        if self.episode_latency_seconds > TARGET_SECONDS:
            return 0
        per_episode_gpu = self.world_gpu_seconds / self.episodes if self.episodes else 0.0
        if per_episode_gpu <= 0:
            return self.episodes
        budget = TARGET_SECONDS * self.capacity.max_replicas
        return max(0, int(budget // per_episode_gpu))

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "episodes": self.episodes,
            "feedback_mode": self.feedback.name,
            "feedback_qualified": self.feedback.qualified,
            "executed_prefix_ticks": self.feedback.executed_prefix,
            "latency_source": self.latencies.source,
            "latencies_are_measured": self.latencies.measured,
            "max_replicas": self.capacity.max_replicas,
            "world_batch_size": self.capacity.world_batch_size,
            "memory_batch_ceiling": self.memory_batch_ceiling,
            "rounds_per_episode": round(self.rounds_per_episode, 3),
            "episode_latency_seconds": round(self.episode_latency_seconds, 3),
            "world_calls": self.world_calls,
            "batched_world_calls": self.batched_world_calls,
            "world_gpu_seconds": round(self.world_gpu_seconds, 3),
            "throughput_seconds": round(self.throughput_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "binding_limit": self.binding_limit,
            "meets_60s_target": self.meets_target,
            "replicas_needed_for_target": self.replicas_needed_for_target(),
            "episodes_achievable_in_60s": self.episodes_achievable_in_target(),
            "warnings": list(self.warnings),
        }


def plan_capacity(
    latencies: MeasuredLatencies,
    capacity: DeploymentCapacity,
    feedback: FeedbackProfile = NATIVE_FEEDBACK,
    episodes: int = TARGET_EPISODES,
    horizons: Optional[Mapping[str, int]] = None,
) -> CapacityPlan:
    """Compute the burst plan and name the limit that binds."""

    if episodes < 1:
        raise CapacityError("episodes must be at least 1")
    task_horizons = dict(horizons or TASK_HORIZONS)
    if not task_horizons:
        raise CapacityError("at least one task horizon is required")

    warnings: List[str] = []

    # Rounds are computed per task and averaged, because ceil() per task is not
    # the same as ceil() of the mean -- a 100-tick task at chunk 16 is 7 rounds,
    # not 6.25.
    rounds_by_task = {task: feedback.rounds_for(ticks) for task, ticks in task_horizons.items()}
    rounds_per_episode = sum(rounds_by_task.values()) / len(rounds_by_task)
    max_rounds = max(rounds_by_task.values())

    per_round = latencies.policy_seconds + latencies.world_seconds
    # The slowest task sets the per-episode floor, not the average: the burst is
    # not finished until its longest episode is.
    episode_latency = max_rounds * per_round + latencies.validity_seconds + latencies.judge_seconds

    memory_ceiling = capacity.max_batch_for_memory(latencies.world_peak_memory_bytes)
    effective_batch = capacity.world_batch_size
    if memory_ceiling is not None and effective_batch > memory_ceiling:
        warnings.append(
            "requested world batch %d exceeds the %d that fits in %.0f GB at the measured "
            "%.1f GB peak per call; planning with %d"
            % (
                capacity.world_batch_size,
                memory_ceiling,
                capacity.gpu_memory_bytes / 1024**3,
                (latencies.world_peak_memory_bytes or 0) / 1024**3,
                memory_ceiling,
            )
        )
        effective_batch = memory_ceiling
    if latencies.world_peak_memory_bytes is None and capacity.world_batch_size > 1:
        warnings.append(
            "no measured per-call GPU peak, so the batch size is unchecked against memory; "
            "Gate A must record it before this plan is trusted"
        )

    world_calls = int(round(episodes * rounds_per_episode))
    batched_world_calls = math.ceil(world_calls / effective_batch)
    # A fused forward is not free: assume it costs one call's latency. This is
    # optimistic and is flagged as such.
    world_gpu_seconds = batched_world_calls * latencies.world_seconds
    throughput_seconds = world_gpu_seconds / capacity.max_replicas

    if effective_batch > 1:
        warnings.append(
            "batched throughput assumes a fused forward of %d costs the same wall time as one "
            "call; measure the real batch scaling in Gate A before relying on it" % effective_batch
        )

    wall = max(episode_latency, throughput_seconds)
    binding = "per_episode_latency" if episode_latency >= throughput_seconds else "aggregate_throughput"

    if binding == "per_episode_latency" and not feedback.qualified:
        warnings.append(
            "latency-bound under an unqualified feedback mode: adding replicas will not help, "
            "and the approximation is what is buying the speed"
        )
    if binding == "per_episode_latency" and feedback.qualified:
        warnings.append(
            "latency-bound at the certified native cadence: replicas cannot reduce this, so the "
            "target needs a faster per-round latency or a Gate-B-qualified larger executed prefix"
        )
    if not latencies.measured:
        warnings.append(
            "latencies are estimates, not Gate A measurements; this plan chooses what to try, "
            "it does not report what was achieved"
        )

    return CapacityPlan(
        episodes=episodes,
        feedback=feedback,
        latencies=latencies,
        capacity=capacity,
        rounds_per_episode=rounds_per_episode,
        episode_latency_seconds=episode_latency,
        world_calls=world_calls,
        batched_world_calls=batched_world_calls,
        world_gpu_seconds=world_gpu_seconds,
        throughput_seconds=throughput_seconds,
        wall_seconds=wall,
        binding_limit=binding,
        memory_batch_ceiling=memory_ceiling,
        warnings=warnings,
    )


def sweep_replicas(
    latencies: MeasuredLatencies,
    feedback: FeedbackProfile,
    replica_options: Sequence[int],
    world_batch_size: int = 1,
    episodes: int = TARGET_EPISODES,
    gpu_memory_bytes: int = 80 * 1024**3,
) -> List[CapacityPlan]:
    """Plan across candidate replica caps, to show where the cap stops mattering."""

    plans = []
    for replicas in replica_options:
        plans.append(
            plan_capacity(
                latencies,
                DeploymentCapacity(
                    max_replicas=replicas,
                    world_batch_size=world_batch_size,
                    gpu_memory_bytes=gpu_memory_bytes,
                ),
                feedback=feedback,
                episodes=episodes,
            )
        )
    return plans


def smallest_sufficient_cap(
    latencies: MeasuredLatencies,
    feedback: FeedbackProfile,
    world_batch_size: int = 1,
    episodes: int = TARGET_EPISODES,
    ceiling: int = 1000,
) -> Optional[int]:
    """The smallest replica cap that meets the target, or ``None`` if unreachable.

    ``None`` means the per-episode latency floor already exceeds the target, so
    no cap is sufficient -- the honest answer rather than a very large number.
    """

    probe = plan_capacity(
        latencies,
        DeploymentCapacity(max_replicas=1, world_batch_size=world_batch_size),
        feedback=feedback,
        episodes=episodes,
    )
    if probe.episode_latency_seconds > TARGET_SECONDS:
        return None
    for replicas in range(1, ceiling + 1):
        plan = plan_capacity(
            latencies,
            DeploymentCapacity(max_replicas=replicas, world_batch_size=world_batch_size),
            feedback=feedback,
            episodes=episodes,
        )
        if plan.meets_target:
            return replicas
    return None


def required_world_latency(
    capacity: DeploymentCapacity,
    feedback: FeedbackProfile,
    policy_seconds: float,
    episodes: int = TARGET_EPISODES,
    horizons: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Per-call world latency needed to hit the target at a given capacity.

    This is the actionable direction when the replica cap is fixed: the question
    stops being "how many replicas can we get" and becomes "how fast must one
    forward pass be, and is that reachable by lowering resolution and denoising
    steps".  Both limits are solved, and the tighter one is reported.
    """

    task_horizons = dict(horizons or TASK_HORIZONS)
    rounds = {task: feedback.rounds_for(ticks) for task, ticks in task_horizons.items()}
    mean_rounds = sum(rounds.values()) / len(rounds)
    max_rounds = max(rounds.values())

    # Latency limit: the slowest episode must fit in the window on its own.
    latency_budget = TARGET_SECONDS / max_rounds - policy_seconds
    # Throughput limit: total batched world seconds must fit across replicas.
    batched_calls = math.ceil(episodes * mean_rounds / max(1, capacity.world_batch_size))
    throughput_budget = (
        TARGET_SECONDS * capacity.max_replicas / batched_calls if batched_calls else float("inf")
    )

    feasible = latency_budget > 0
    required = min(latency_budget, throughput_budget) if feasible else None
    return {
        "feasible": feasible and required is not None and required > 0,
        "required_world_seconds": None if not feasible or required is None or required <= 0 else round(required, 4),
        "latency_limited_budget_seconds": round(latency_budget, 4),
        "throughput_limited_budget_seconds": round(throughput_budget, 4),
        "binding_limit": (
            "per_episode_latency" if feasible and latency_budget <= throughput_budget else "aggregate_throughput"
        ),
        "rounds_per_episode_mean": round(mean_rounds, 3),
        "rounds_worst_task": max_rounds,
        "batched_world_calls": batched_calls,
        "reason": (
            None
            if feasible
            else "policy latency alone (%.3f s x %d rounds) already exceeds the %.0f s window"
            % (policy_seconds, max_rounds, TARGET_SECONDS)
        ),
    }


def fallback_episode_count(plan: CapacityPlan) -> Dict[str, Any]:
    """What to run live when the full matrix will not fit in the target window.

    Spec section 7 forbids silently reducing the study to fit. The study stays
    1,500; this describes a smaller *live demonstration* and says plainly that it
    is a subset, so the two are never conflated.
    """

    achievable = plan.episodes_achievable_in_target()
    return {
        "full_study_episodes": plan.episodes,
        "study_is_reduced": False,
        "live_demonstration_episodes": min(achievable, plan.episodes),
        "live_is_a_subset": achievable < plan.episodes,
        "reason": (
            "The %d-episode study still runs in full and is reported in full. Only the live "
            "on-stage burst is a subset, because the per-episode floor or the available capacity "
            "will not fit the whole matrix in %.0f s." % (plan.episodes, TARGET_SECONDS)
        ),
        "note": (
            "Never present a subset as the full study, and never present the study's runtime as "
            "the burst's runtime."
        ),
    }


__all__ = [
    "NATIVE_FEEDBACK",
    "TARGET_EPISODES",
    "TARGET_SECONDS",
    "TARGET_USD",
    "CapacityError",
    "CapacityPlan",
    "DeploymentCapacity",
    "FeedbackProfile",
    "MeasuredLatencies",
    "chunked_prefix",
    "fallback_episode_count",
    "plan_capacity",
    "required_world_latency",
    "smallest_sufficient_cap",
    "sweep_replicas",
]
