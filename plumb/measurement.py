"""Honest, dependency-light measurement summaries for PLUMB.

This module deliberately separates descriptive quantities that can be computed
from a rollout ledger from inference that still needs a frozen, lineage-aware
protocol.  In particular, a Wilson interval is not presented as a substitute
for the specified clustered bootstrap or simultaneous endpoint envelopes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, permutations
import hashlib
import math
import random
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .reference import (
    HUMAN,
    POLICY_IDS,
    REFERENCE_TRIALS_PER_CELL,
    TASK_IDS,
    canonical_policy,
    canonical_task,
    reference_payload,
    reference_rate,
    reference_successes,
)


TERMINAL_STATUSES = frozenset(("completed", "failed", "cancelled"))
SERVICE_FAILURE_STATUSES = frozenset(("failed",))
PRIMARY_POLICY_IDS = tuple(POLICY_IDS)
PRIMARY_TASK_IDS = tuple(TASK_IDS)


def _is_bool(value: Any) -> bool:
    # bool is intentionally stricter than 0/1 so malformed judge records do
    # not become failures by accident.
    return isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _json_number(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(present) / len(present) if present else None


def _stable_sort_key(value: Any) -> Tuple[str, str]:
    return (type(value).__name__, str(value))


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Optional[Dict[str, Any]]:
    """Return the two-sided Wilson interval, or ``None`` for an empty sample.

    This is descriptive when starts are correlated or repeatedly seeded.  It
    intentionally performs no 0/0 substitution.
    """

    if n <= 0:
        return None
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / float(n)
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return {
        "method": "wilson",
        "confidence": confidence,
        "successes": successes,
        "n": n,
        "lower": max(0.0, centre - radius),
        "upper": min(1.0, centre + radius),
    }


def exact_binomial_interval(successes: int, n: int, confidence: float = 0.95) -> Dict[str, Any]:
    """Optionally calculate Clopper--Pearson bounds with SciPy.

    SciPy is not a PLUMB runtime dependency.  Returning an explicit
    unavailable state when it is absent is preferable to labelling a different
    approximation as an exact beta interval.
    """

    if n <= 0:
        return {"status": "unavailable", "reason": "Exact binomial intervals are undefined for n=0."}
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    try:
        from scipy.stats import beta  # type: ignore
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "SciPy is not installed; no exact-beta interval was substituted.",
        }
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes))
    return {
        "status": "computed",
        "method": "clopper_pearson",
        "confidence": confidence,
        "successes": successes,
        "n": n,
        "lower": lower,
        "upper": upper,
    }


def exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p value from the two discordant counts."""

    if a_only < 0 or b_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = a_only + b_only
    if discordant == 0:
        # The conditional binomial experiment has no trials here.  The usual
        # exact-test convention is p=1, not an undefined 0/0 calculation.
        return 1.0
    tail_count = sum(math.comb(discordant, k) for k in range(0, min(a_only, b_only) + 1))
    return min(1.0, 2.0 * (tail_count / float(1 << discordant)))


def mean_maximum_rank_violation(perf_sim: Sequence[float], perf_real: Sequence[float]) -> float:
    """The pinned SIMPLER MMRV definition: ``(perf_sim, perf_real)``.

    The order comparison is deliberately strict.  Thus a tie in one vector
    and a strict ordering in the other is a violation, matching SIMPLER's
    ``(a > b) != (c > d)`` implementation.  Violation magnitude is the real
    performance margin, and each policy contributes its maximum violation.
    """

    if len(perf_sim) != len(perf_real):
        raise ValueError("perf_sim and perf_real must have the same length")
    if not perf_sim:
        raise ValueError("MMRV is undefined for an empty policy vector")
    sim = [float(value) for value in perf_sim]
    real = [float(value) for value in perf_real]
    if not all(math.isfinite(value) for value in sim + real):
        raise ValueError("MMRV requires finite performance values")
    violations: List[float] = []
    for i in range(len(sim)):
        maximum = 0.0
        for j in range(len(sim)):
            if (sim[i] > sim[j]) != (real[i] > real[j]):
                maximum = max(maximum, abs(real[i] - real[j]))
        violations.append(maximum)
    return sum(violations) / len(violations)


# Convenient short name for callers and tests.
mmrv = mean_maximum_rank_violation


def mmrv_random_order_baseline(perf_sim: Sequence[float], perf_real: Sequence[float]) -> Dict[str, Any]:
    """Enumerate the random-order MMRV baseline, retaining tied values."""

    if len(perf_sim) != len(perf_real):
        raise ValueError("perf_sim and perf_real must have the same length")
    if not perf_sim:
        return {"status": "unavailable", "reason": "An empty policy vector has no random-order baseline."}
    # Do not de-duplicate tied permutations.  A random assignment permutes
    # policy labels, even when two values happen to be equal.
    values = [mean_maximum_rank_violation(candidate, perf_real) for candidate in permutations(perf_sim)]
    return {
        "status": "computed",
        "permutations": math.factorial(len(perf_sim)),
        "mean": sum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def validate_source_lineage_leakage(
    episodes: Sequence[Mapping[str, Any]], cohort_field: str = "cohort"
) -> Dict[str, Any]:
    """Detect source lineages that cross disjoint study cohorts.

    Pairing policies inside one cohort is expected; only a lineage appearing in
    two different cohort labels is a leakage violation.  Missing lineage IDs
    are reported as an inability to establish the required guarantee.
    """

    cohorts_by_lineage: Dict[str, set] = defaultdict(set)
    missing_lineage_records: List[int] = []
    cohort_records = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise TypeError("episodes must contain mappings")
        lineage = episode.get("start_lineage_id")
        cohort = episode.get(cohort_field, episode.get("split"))
        if cohort is None or str(cohort).strip() == "":
            continue
        cohort_records += 1
        if lineage is None or str(lineage).strip() == "":
            missing_lineage_records.append(index)
            continue
        cohorts_by_lineage[str(lineage)].add(str(cohort))
    violations = [
        {"start_lineage_id": lineage, "cohorts": sorted(values)}
        for lineage, values in sorted(cohorts_by_lineage.items())
        if len(values) > 1
    ]
    if violations:
        status = "fail"
    elif missing_lineage_records:
        status = "unverifiable"
    elif cohort_records:
        status = "pass"
    else:
        status = "not_assessed"
    return {
        "status": status,
        "lineage_count": len(cohorts_by_lineage),
        "cohort_record_count": cohort_records,
        "violations": violations,
        "missing_lineage_record_indices": missing_lineage_records,
        "reason": (
            "A source lineage appears in more than one cohort."
            if violations
            else "Some cohort records have no start_lineage_id."
            if missing_lineage_records
            else None
        ),
    }


def _normalise_policy(value: Any) -> str:
    policy = canonical_policy(value)
    return policy if policy is not None else "__missing_policy__"


def _normalise_task(value: Any) -> str:
    task = canonical_task(value)
    return task if task is not None else "__missing_task__"


def _episode_variant(episode: Mapping[str, Any]) -> Optional[str]:
    policy = episode.get("policy")
    variant = policy.get("variant") if isinstance(policy, Mapping) else None
    if variant is None:
        variant = episode.get("policy_variant", episode.get("variant"))
    if variant is None or str(variant).strip() == "":
        return None
    return str(variant).strip()


def _episode_identity(episode: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    episode_id = episode.get("episode_id", episode.get("id"))
    return {
        "episode_id": str(episode_id) if episode_id is not None and str(episode_id).strip() else None,
        "policy": _normalise_policy(episode.get("policy")),
        "policy_variant": _episode_variant(episode),
        "task": _normalise_task(episode.get("task")),
        "cohort": str(episode["cohort"]).strip() if episode.get("cohort") is not None else None,
        "protocol_hash": str(episode["protocol_hash"]).strip() if episode.get("protocol_hash") is not None else None,
        "start_id": str(episode["start_id"]).strip() if episode.get("start_id") is not None else None,
    }


def _primary_analysis_requested(episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> bool:
    mode = str(protocol.get("analysis_mode", protocol.get("mode", ""))).strip().lower()
    if mode in ("primary", "qualified"):
        return True
    if isinstance(protocol.get("analysis_manifest"), Mapping) or isinstance(protocol.get("frozen_analysis_manifest"), Mapping):
        return True
    return any(str(episode.get("cohort", "")).strip().lower() == "primary" for episode in episodes)


def _validate_analysis_manifest(episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Require a frozen identity manifest before a primary analysis is scored."""

    primary_requested = _primary_analysis_requested(episodes, protocol)
    manifest = protocol.get("analysis_manifest", protocol.get("frozen_analysis_manifest"))
    if not primary_requested:
        return {
            "status": "diagnostic",
            "primary_requested": False,
            "reason": "No primary manifest was requested; output is diagnostic and never a qualified claim.",
        }
    if not isinstance(manifest, Mapping):
        return {
            "status": "rejected",
            "primary_requested": True,
            "reason": "Primary analysis requires a frozen analysis_manifest with allowed episode IDs and identity fields.",
            "errors": [{"reason": "missing_analysis_manifest"}],
        }
    allowed_raw = manifest.get("allowed_episode_ids")
    allowed = set(str(value) for value in allowed_raw) if isinstance(allowed_raw, (list, tuple, set)) else set()
    identity = manifest.get("identity") if isinstance(manifest.get("identity"), Mapping) else {}
    expected_cohort = identity.get("cohort", manifest.get("cohort"))
    expected_protocol_hash = identity.get("protocol_hash", manifest.get("protocol_hash"))
    variants = manifest.get("policy_variants")
    errors: List[Dict[str, Any]] = []
    if not allowed:
        errors.append({"reason": "missing_allowed_episode_ids"})
    if expected_cohort is None or not str(expected_cohort).strip():
        errors.append({"reason": "missing_manifest_cohort"})
    if expected_protocol_hash is None or not str(expected_protocol_hash).strip():
        errors.append({"reason": "missing_manifest_protocol_hash"})
    if not isinstance(variants, Mapping) or not variants:
        errors.append({"reason": "missing_manifest_policy_variants"})
    identities_by_id = manifest.get("episode_identities")
    if not isinstance(identities_by_id, Mapping):
        identities_by_id = {}
    observed_ids: List[str] = []
    for index, episode in enumerate(episodes):
        observed = _episode_identity(episode)
        episode_id = observed["episode_id"]
        if episode_id is None:
            errors.append({"record_index": index, "reason": "missing_episode_id"})
            continue
        observed_ids.append(episode_id)
        if episode_id not in allowed:
            errors.append({"episode_id": episode_id, "reason": "episode_not_allowed_by_manifest"})
        if observed["cohort"] != str(expected_cohort).strip():
            errors.append({"episode_id": episode_id, "reason": "cohort_identity_mismatch"})
        if observed["protocol_hash"] != str(expected_protocol_hash).strip():
            errors.append({"episode_id": episode_id, "reason": "protocol_identity_mismatch"})
        expected_variant = variants.get(observed["policy"]) if isinstance(variants, Mapping) else None
        if expected_variant is None or observed["policy_variant"] != str(expected_variant).strip():
            errors.append({"episode_id": episode_id, "reason": "policy_variant_identity_mismatch"})
        per_episode = identities_by_id.get(episode_id)
        if isinstance(per_episode, Mapping):
            for field in ("policy", "policy_variant", "task", "cohort", "protocol_hash", "start_id"):
                if field in per_episode and observed[field] != str(per_episode[field]).strip():
                    errors.append({"episode_id": episode_id, "reason": "episode_identity_mismatch", "field": field})
    duplicate_ids = sorted(episode_id for episode_id, count in Counter(observed_ids).items() if count > 1)
    if duplicate_ids:
        errors.append({"reason": "duplicate_episode_ids", "episode_ids": duplicate_ids})
    missing_ids = sorted(allowed - set(observed_ids))
    if missing_ids:
        errors.append({"reason": "manifest_allowed_episode_missing_from_ledger", "episode_ids": missing_ids})
    if errors:
        return {
            "status": "rejected",
            "primary_requested": True,
            "reason": "Primary analysis identities do not match the frozen manifest.",
            "errors": errors,
        }
    return {
        "status": "accepted",
        "primary_requested": True,
        "manifest_episode_count": len(allowed),
        "cohort": str(expected_cohort).strip(),
        "protocol_hash": str(expected_protocol_hash).strip(),
        "policy_variants": {str(key): str(value).strip() for key, value in variants.items()},
    }


def _nonterminal_ledger_status(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    nonterminal = Counter()
    run_ids = set()
    for episode in episodes:
        status = str(episode.get("status", "")).strip().lower()
        if status not in TERMINAL_STATUSES:
            nonterminal[status or "missing"] += 1
            run_id = episode.get("run_id")
            if run_id is not None:
                run_ids.add(str(run_id))
    return {
        "active": bool(nonterminal),
        "nonterminal_status_counts": dict(sorted(nonterminal.items())),
        "run_ids": sorted(run_ids),
    }


def _arm_id(
    policy: str,
    variant: Optional[str],
    cohort: Optional[str],
    protocol_hash: Optional[str],
    primary_manifest_accepted: bool,
) -> str:
    # A frozen primary manifest has already established one declared variant
    # per policy.  Diagnostics must retain a variant suffix to prevent pooled
    # released/corrected sensitivity arms from becoming one n=100 cell.
    if primary_manifest_accepted or (variant is None and cohort is None and protocol_hash is None):
        return policy
    parts = []
    if variant is not None:
        parts.append("variant=" + variant)
    if cohort is not None:
        parts.append("cohort=" + cohort)
    if protocol_hash is not None:
        parts.append("protocol=" + protocol_hash)
    return policy + " [" + "; ".join(parts) + "]"


def _episode_outcome(episode: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify one logical episode without converting missingness to failure."""

    raw_status = episode.get("status")
    status = str(raw_status).strip().lower() if raw_status is not None else ""
    raw_validity = episode.get("validity")
    validity = str(raw_validity).strip().lower() if raw_validity is not None else ""
    success = episode.get("binary_success")
    explicit_reason = episode.get("missing_reason")
    explicit_reason = str(explicit_reason) if explicit_reason is not None else None

    service_failure = status in SERVICE_FAILURE_STATUSES or explicit_reason == "service_failure"
    if status != "completed":
        if service_failure:
            reason = "service_failure"
        elif status == "cancelled":
            reason = "cancelled"
        elif status:
            reason = "execution_" + status
        else:
            reason = "execution_status_missing"
        return {
            "evaluable": False,
            "success": False,
            "reason": reason,
            "status": status or None,
            "validity": validity or None,
            "service_failure": service_failure,
        }
    if validity != "valid":
        if validity == "invalid":
            reason = "invalid"
        elif validity == "unknown":
            reason = "unknown"
        elif explicit_reason:
            reason = explicit_reason
        else:
            reason = "validity_missing" if not validity else "validity_" + validity
        return {
            "evaluable": False,
            "success": False,
            "reason": reason,
            "status": status,
            "validity": validity or None,
            "service_failure": False,
        }
    if not _is_bool(success):
        return {
            "evaluable": False,
            "success": False,
            "reason": explicit_reason or "judge_unevaluable",
            "status": status,
            "validity": validity,
            "service_failure": False,
        }
    return {
        "evaluable": True,
        "success": bool(success),
        "reason": None,
        "status": status,
        "validity": validity,
        "service_failure": False,
    }


def _protocol_identifiers(protocol: Mapping[str, Any], key: str, defaults: Sequence[str], normaliser: Any) -> List[str]:
    values = protocol.get(key)
    if values is None:
        return list(defaults)
    if isinstance(values, Mapping):
        values = list(values.keys())
    if isinstance(values, (str, bytes)):
        values = [values]
    result: List[str] = []
    for value in values:
        identifier = normaliser(value)
        if identifier not in result:
            result.append(identifier)
    return result or list(defaults)


def _planned_count(protocol: Mapping[str, Any], policy: str, task: str, observed: int) -> Optional[int]:
    """Read optional expected ledger cardinality without assuming unrecorded trials."""

    configured: Any = None
    planned_counts = protocol.get("planned_counts")
    if isinstance(planned_counts, Mapping):
        configured = planned_counts.get(policy + "/" + task)
        if configured is None:
            row = planned_counts.get(policy)
            if isinstance(row, Mapping):
                configured = row.get(task)
    if configured is None:
        configured = protocol.get("planned_per_cell", protocol.get("planned_n"))
    if configured is None:
        return None
    try:
        configured_int = int(configured)
    except (TypeError, ValueError):
        return None
    if configured_int < 0:
        raise ValueError("planned episode counts must be non-negative")
    # A malformed partial export must not make observed logical records vanish.
    return max(observed, configured_int)


def _cell_from_records(
    policy: str,
    task: str,
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    reference_policy: Optional[str] = None,
    policy_variant: Optional[str] = None,
) -> Dict[str, Any]:
    outcomes = [_episode_outcome(record) for record in records]
    observed_records = len(records)
    planned = _planned_count(protocol, policy, task, observed_records)
    n = max(observed_records, planned or 0)
    implicit_missing = n - observed_records
    valid = sum(1 for outcome in outcomes if outcome["evaluable"])
    successes = sum(1 for outcome in outcomes if outcome["evaluable"] and outcome["success"])
    status_counts = Counter(outcome["status"] or "missing" for outcome in outcomes)
    validity_counts = Counter(outcome["validity"] or "missing" for outcome in outcomes)
    reason_counts = Counter(outcome["reason"] for outcome in outcomes if outcome["reason"])
    if implicit_missing:
        reason_counts["unrecorded_planned"] += implicit_missing
    service_failures = sum(1 for outcome in outcomes if outcome["service_failure"])
    missing = n - valid
    conditional_rate = successes / float(valid) if valid else None
    positive_rate = successes / float(n) if n else None
    coverage = valid / float(n) if n else None
    bounds = None
    if n:
        bounds = {
            "lower": successes / float(n),
            "upper": (successes + n - valid) / float(n),
            "width": (n - valid) / float(n),
            "method": "horowitz_manski_no_assumption",
        }
    source_policy = reference_policy or policy
    ref_successes = reference_successes(source_policy, task, "human")
    ref_rate = reference_rate(source_policy, task, "human")
    simpler_successes = reference_successes(source_policy, task, "simpler")
    simpler_rate = reference_rate(source_policy, task, "simpler")
    progress_values = [float(record["progress_score"]) for record in records if _finite_number(record.get("progress_score"))]
    is_complete_ledger = bool(n) and implicit_missing == 0 and all(
        outcome["status"] in TERMINAL_STATUSES for outcome in outcomes
    )
    conditional_wilson = wilson_interval(successes, valid)
    positive_wilson = wilson_interval(successes, n)
    published_wilson = (
        wilson_interval(ref_successes, REFERENCE_TRIALS_PER_CELL) if ref_successes is not None else None
    )
    return {
        "policy": policy,
        "policy_base": source_policy,
        "policy_variant": policy_variant,
        "task": task,
        "n": n,
        "reported_records": observed_records,
        "planned_records": planned,
        "valid": valid,
        "successes": successes,
        "missing": missing,
        "coverage": coverage,
        # `rate` is deliberately conditional, and is null rather than 0/0.
        "rate": conditional_rate,
        "positive_rate": positive_rate,
        # Compact arrays are convenient for scoreboards.  The method, n and
        # width stay alongside them so no interval is stripped of context.
        "wilson": [conditional_wilson["lower"], conditional_wilson["upper"]] if conditional_wilson else None,
        "wilson_detail": conditional_wilson,
        "positive_wilson": [positive_wilson["lower"], positive_wilson["upper"]] if positive_wilson else None,
        "positive_wilson_detail": positive_wilson,
        "missing_bounds": [bounds["lower"], bounds["upper"]] if bounds else None,
        "missing_bounds_detail": bounds,
        "missingness_rate": (n - valid) / float(n) if n else None,
        "missing_bounds_width": bounds["width"] if bounds else None,
        "status_counts": dict(sorted(status_counts.items())),
        "validity_counts": dict(sorted(validity_counts.items())),
        "missing_reason_counts": dict(sorted(reason_counts.items())),
        "service_failures": service_failures,
        "mean_progress_score": _safe_mean(progress_values),
        "progress_n": len(progress_values),
        "ledger_status": "complete" if is_complete_ledger else "incomplete" if n else "not_started",
        "reference_successes": ref_successes,
        "reference_n": REFERENCE_TRIALS_PER_CELL if ref_successes is not None else None,
        "reference_rate": ref_rate,
        "reference_wilson": [published_wilson["lower"], published_wilson["upper"]] if published_wilson else None,
        "reference_wilson_detail": published_wilson,
        "simpler_successes": simpler_successes,
        "simpler_rate": simpler_rate,
        "conditional_reference_difference": conditional_rate - ref_rate
        if conditional_rate is not None and ref_rate is not None
        else None,
        "observed_positive_reference_difference": positive_rate - ref_rate
        if positive_rate is not None and ref_rate is not None
        else None,
        "interval_caveat": "Wilson intervals are descriptive only when starts are correlated or repeated.",
    }


def _cell_index(cells: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(str(cell["policy"]), str(cell["task"])): cell for cell in cells}


def _policy_macro(policy: str, tasks: Sequence[str], cells: Mapping[Tuple[str, str], Mapping[str, Any]]) -> Dict[str, Any]:
    task_cells = [cells[(policy, task)] for task in tasks if (policy, task) in cells]
    positive_values = [cell["positive_rate"] for cell in task_cells]
    conditional_values = [cell["rate"] for cell in task_cells]
    reference_values = [cell["reference_rate"] for cell in task_cells]
    all_tasks_present = len(task_cells) == len(tasks)
    return {
        "policy": policy,
        "task_count": len(task_cells),
        "expected_task_count": len(tasks),
        "observed_positive_rate": _safe_mean(positive_values) if all_tasks_present and all(value is not None for value in positive_values) else None,
        "conditional_rate": _safe_mean(conditional_values) if all_tasks_present and all(value is not None for value in conditional_values) else None,
        "reference_rate": _safe_mean(reference_values) if all_tasks_present and all(value is not None for value in reference_values) else None,
        "status": "computed"
        if all_tasks_present and all(value is not None for value in positive_values)
        else "incomplete",
        "note": "Macro rates are equal-weight task means; they do not generalize to a population of tasks.",
    }


def _summary(cells: Sequence[Mapping[str, Any]], policies: Sequence[str], tasks: Sequence[str]) -> Dict[str, Any]:
    total_n = sum(int(cell["n"]) for cell in cells)
    total_valid = sum(int(cell["valid"]) for cell in cells)
    total_successes = sum(int(cell["successes"]) for cell in cells)
    status_counts: Counter = Counter()
    reason_counts: Counter = Counter()
    service_failures = 0
    for cell in cells:
        status_counts.update(cell["status_counts"])
        reason_counts.update(cell["missing_reason_counts"])
        service_failures += int(cell["service_failures"])
    indexed = _cell_index(cells)
    macros = [_policy_macro(policy, tasks, indexed) for policy in policies]
    per_task: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        rows = [indexed[(policy, task)] for policy in policies if (policy, task) in indexed]
        per_task[task] = {
            "n": sum(int(row["n"]) for row in rows),
            "valid": sum(int(row["valid"]) for row in rows),
            "successes": sum(int(row["successes"]) for row in rows),
            "coverage": (
                sum(int(row["valid"]) for row in rows) / float(sum(int(row["n"]) for row in rows))
                if sum(int(row["n"]) for row in rows)
                else None
            ),
            "policies": [
                {
                    "policy": row["policy"],
                    "positive_rate": row["positive_rate"],
                    "rate": row["rate"],
                    "reference_rate": row["reference_rate"],
                }
                for row in rows
            ],
        }
    return {
        "n": total_n,
        "valid": total_valid,
        "successes": total_successes,
        "missing": total_n - total_valid,
        "coverage": total_valid / float(total_n) if total_n else None,
        "rate": total_successes / float(total_valid) if total_valid else None,
        "positive_rate": total_successes / float(total_n) if total_n else None,
        "missing_bounds": {
            "lower": total_successes / float(total_n),
            "upper": (total_successes + total_n - total_valid) / float(total_n),
            "width": (total_n - total_valid) / float(total_n),
            "method": "horowitz_manski_no_assumption",
        }
        if total_n
        else None,
        "status_counts": dict(sorted(status_counts.items())),
        "missing_reason_counts": dict(sorted(reason_counts.items())),
        "service_failures": service_failures,
        "macro": macros,
        "per_task": per_task,
    }


def _operational_summary(cells: Sequence[Mapping[str, Any]], reason: str) -> Dict[str, Any]:
    status_counts: Counter = Counter()
    for cell in cells:
        status_counts.update(cell["status_counts"])
    terminal = sum(count for status, count in status_counts.items() if status in TERMINAL_STATUSES)
    reported = sum(int(cell["reported_records"]) for cell in cells)
    return {
        "status": "provisional_operational_only",
        "reason": reason,
        "reported_records": reported,
        "planned_records": sum(int(cell["n"]) for cell in cells),
        "terminal_records": terminal,
        "nonterminal_records": reported - terminal,
        "status_counts": dict(sorted(status_counts.items())),
        "scientific_rates": None,
    }


def _redact_scientific_cell_fields(cells: Sequence[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
    """Keep durable operational counts while withholding live/rejected estimates."""

    scientific_fields = (
        "valid",
        "successes",
        "missing",
        "coverage",
        "rate",
        "positive_rate",
        "wilson",
        "wilson_detail",
        "positive_wilson",
        "positive_wilson_detail",
        "missing_bounds",
        "missing_bounds_detail",
        "missingness_rate",
        "missing_bounds_width",
        "mean_progress_score",
        "progress_n",
        "reference_successes",
        "reference_n",
        "reference_rate",
        "reference_wilson",
        "reference_wilson_detail",
        "simpler_successes",
        "simpler_rate",
        "conditional_reference_difference",
        "observed_positive_reference_difference",
    )
    output: List[Dict[str, Any]] = []
    for cell in cells:
        copied = dict(cell)
        for field in scientific_fields:
            copied[field] = None
        copied["ledger_status"] = "provisional" if copied["reported_records"] else "not_started"
        copied["scientific_status"] = "blocked"
        copied["scientific_reason"] = reason
        output.append(copied)
    return output


def _paired_start_values(
    episodes: Sequence[Mapping[str, Any]], policy_a: str, policy_b: str, task: str
) -> Dict[str, Any]:
    """Pair lower-bound endpoint Y by start ID, rejecting duplicate bundles."""

    values: Dict[str, Dict[str, List[int]]] = {policy_a: defaultdict(list), policy_b: defaultdict(list)}
    missing_start_records = 0
    for episode in episodes:
        if _normalise_task(episode.get("task")) != task:
            continue
        policy = _normalise_policy(episode.get("policy"))
        if policy not in values:
            continue
        start_id = episode.get("start_id")
        if start_id is None or str(start_id).strip() == "":
            missing_start_records += 1
            continue
        outcome = _episode_outcome(episode)
        values[policy][str(start_id)].append(1 if outcome["evaluable"] and outcome["success"] else 0)
    starts_a = set(values[policy_a])
    starts_b = set(values[policy_b])
    paired_ids = sorted(starts_a & starts_b)
    b = c = concordant_positive = concordant_negative = paired_n = 0
    duplicate_start_ids: List[str] = []
    for start_id in paired_ids:
        a_values = values[policy_a][start_id]
        b_values = values[policy_b][start_id]
        if len(a_values) != 1 or len(b_values) != 1:
            duplicate_start_ids.append(start_id)
            continue
        a_value, b_value = a_values[0], b_values[0]
        paired_n += 1
        if a_value == 1 and b_value == 0:
            b += 1
        elif a_value == 0 and b_value == 1:
            c += 1
        elif a_value == 1:
            concordant_positive += 1
        else:
            concordant_negative += 1
    return {
        "a_only": b,
        "b_only": c,
        "paired_n": paired_n,
        "concordant_positive": concordant_positive,
        "concordant_negative": concordant_negative,
        "unpaired_a": len(starts_a - starts_b),
        "unpaired_b": len(starts_b - starts_a),
        "duplicate_start_ids": duplicate_start_ids,
        "missing_start_records": missing_start_records,
    }


def _task_pairwise(
    episodes: Sequence[Mapping[str, Any]],
    cells: Mapping[Tuple[str, str], Mapping[str, Any]],
    policy_a: str,
    policy_b: str,
    task: str,
) -> Dict[str, Any]:
    a_cell = cells[(policy_a, task)]
    b_cell = cells[(policy_b, task)]
    paired = _paired_start_values(episodes, policy_a, policy_b, task)
    pvalue = exact_mcnemar_pvalue(paired["a_only"], paired["b_only"])
    available = paired["paired_n"] > 0
    return {
        "scope": task,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "observed_positive_difference": (
            a_cell["positive_rate"] - b_cell["positive_rate"]
            if a_cell["positive_rate"] is not None and b_cell["positive_rate"] is not None
            else None
        ),
        "paired": {
            **paired,
            "method": "two_sided_exact_mcnemar",
            "p_value": pvalue,
            "family_alpha": 0.05 / 75.0,
            "passes_bonferroni_exact_test": pvalue <= 0.05 / 75.0,
        },
        "status": "computed" if available else "unavailable",
        "ordering": "indeterminate",
        "supported": False,
        "ordering_reason": (
            "Formal ordering additionally requires the conservative lineage-aware bootstrap and simultaneous endpoint envelopes; those are unavailable here."
        ),
    }


def _lineage_bundle_design(
    episodes: Sequence[Mapping[str, Any]],
    policies: Sequence[str],
    tasks: Sequence[str],
    min_lineage_clusters: int = 10,
) -> Dict[str, Any]:
    """Construct matched start bundles, retaining shared source lineages.

    This intentionally refuses partial cells, duplicate logical records, absent
    start IDs, and absent/inconsistent source lineages.  A bootstrap can only
    claim to resample a matched design when each selected start has precisely
    one logical record for every selected policy.
    """

    selected_policies = list(dict.fromkeys(_normalise_policy(policy) for policy in policies))
    selected_tasks = list(dict.fromkeys(_normalise_task(task) for task in tasks))
    if not selected_policies or not selected_tasks:
        return {"status": "unavailable", "reason": "At least one policy and task are required.", "errors": []}
    if min_lineage_clusters < 2:
        raise ValueError("min_lineage_clusters must be at least two")
    records: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    lineages: Dict[Tuple[str, str], set] = defaultdict(set)
    errors: List[Dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise TypeError("episodes must contain mappings")
        policy = _normalise_policy(episode.get("policy"))
        task = _normalise_task(episode.get("task"))
        if policy not in selected_policies or task not in selected_tasks:
            continue
        start = episode.get("start_id")
        lineage = episode.get("start_lineage_id")
        if start is None or str(start).strip() == "":
            errors.append({"record_index": index, "reason": "missing_start_id"})
            continue
        if lineage is None or str(lineage).strip() == "":
            errors.append({"record_index": index, "reason": "missing_start_lineage_id"})
            continue
        start_id = str(start)
        records[(policy, task, start_id)].append(episode)
        lineages[(task, start_id)].add(str(lineage))
    tasks_by_lineage: Dict[str, set] = defaultdict(set)
    for (task, _), values in lineages.items():
        for lineage in values:
            tasks_by_lineage[lineage].add(task)
    cross_task_lineages = [
        {"start_lineage_id": lineage, "tasks": sorted(values)}
        for lineage, values in sorted(tasks_by_lineage.items())
        if len(values) > 1
    ]
    if cross_task_lineages:
        errors.append({"reason": "source_lineage_crosses_tasks", "lineages": cross_task_lineages})
    leakage = validate_source_lineage_leakage(episodes)
    if leakage["status"] == "fail":
        errors.append({"reason": "source_lineage_crosses_cohorts", "violations": leakage["violations"]})

    task_designs: Dict[str, Any] = {}
    for task in selected_tasks:
        starts = sorted(start for current_task, start in lineages if current_task == task)
        if not starts:
            errors.append({"task": task, "reason": "no_matched_start_bundles"})
            continue
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for start_id in starts:
            source_lineages = lineages[(task, start_id)]
            if len(source_lineages) != 1:
                errors.append(
                    {
                        "task": task,
                        "start_id": start_id,
                        "reason": "inconsistent_start_lineage",
                        "lineages": sorted(source_lineages),
                    }
                )
                continue
            outcomes: Dict[str, Dict[str, int]] = {}
            complete = True
            for policy in selected_policies:
                logical_records = records.get((policy, task, start_id), [])
                if len(logical_records) != 1:
                    errors.append(
                        {
                            "task": task,
                            "start_id": start_id,
                            "policy": policy,
                            "reason": "missing_policy_record" if not logical_records else "duplicate_policy_record",
                            "records": len(logical_records),
                        }
                    )
                    complete = False
                    continue
                outcome = _episode_outcome(logical_records[0])
                # Y is the labelled all-attempt lower endpoint.  Z supplies
                # the missingness upper endpoint: success OR missing.
                y = int(outcome["evaluable"] and outcome["success"])
                z = int(y or not outcome["evaluable"])
                outcomes[policy] = {"y": y, "z": z}
            if complete:
                grouped[next(iter(source_lineages))].append({"start_id": start_id, "outcomes": outcomes})
        groups: List[Dict[str, Any]] = []
        for lineage, bundles in sorted(grouped.items()):
            groups.append(
                {
                    "lineage": lineage,
                    "n": len(bundles),
                    "bundles": bundles,
                    "y": {policy: sum(bundle["outcomes"][policy]["y"] for bundle in bundles) for policy in selected_policies},
                    "z": {policy: sum(bundle["outcomes"][policy]["z"] for bundle in bundles) for policy in selected_policies},
                }
            )
        if not groups:
            errors.append({"task": task, "reason": "no_complete_matched_bundles"})
        task_designs[task] = {"groups": groups, "n_starts": sum(group["n"] for group in groups)}
    if errors:
        return {
            "status": "unavailable",
            "reason": "Lineage-bundle inference requires complete, uniquely matched logical records.",
            "errors": errors,
            "leakage": leakage,
        }
    clustered_tasks = [
        task for task, design in task_designs.items() if any(group["n"] != 1 for group in design["groups"])
    ]
    weak_tasks = [task for task, design in task_designs.items() if len(design["groups"]) < min_lineage_clusters]
    return {
        "status": "computed",
        "policies": selected_policies,
        "tasks": selected_tasks,
        "task_designs": task_designs,
        "independent_starts": not clustered_tasks,
        "clustered_tasks": clustered_tasks,
        "weak_lineage_tasks": weak_tasks,
        "min_lineage_clusters": min_lineage_clusters,
        "leakage": leakage,
    }


def _design_summary(design: Mapping[str, Any]) -> Dict[str, Any]:
    if design.get("status") != "computed":
        return {"status": design.get("status"), "reason": design.get("reason"), "errors": design.get("errors", [])}
    return {
        "status": "computed",
        "independent_starts": bool(design["independent_starts"]),
        "clustered_tasks": list(design["clustered_tasks"]),
        "weak_lineage_tasks": list(design["weak_lineage_tasks"]),
        "min_lineage_clusters": int(design["min_lineage_clusters"]),
        "tasks": {
            task: {
                "starts": int(task_design["n_starts"]),
                "lineages": len(task_design["groups"]),
                "lineage_sizes": [int(group["n"]) for group in task_design["groups"]],
            }
            for task, task_design in design["task_designs"].items()
        },
    }


def _one_sided_bootstrap_lower(values: Sequence[float], alpha: float) -> Optional[float]:
    if not values or not 0.0 < alpha < 1.0:
        return None
    ordered = sorted(float(value) for value in values)
    # A conservative empirical lower percentile.  It is not called an exact
    # confidence bound when the lineage design is weak or clustered.
    index = max(0, math.ceil(alpha * len(ordered)) - 1)
    return ordered[index]


def _bootstrap_endpoint_rates(
    design: Mapping[str, Any], bootstrap_replicates: int, seed: int
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Generate task-stratified lineage resamples once for every policy."""

    samples: Dict[str, Dict[str, Dict[str, Any]]] = {}
    policies = design["policies"]
    for task in design["tasks"]:
        groups = design["task_designs"][task]["groups"]
        point_n = sum(group["n"] for group in groups)
        task_samples: Dict[str, Dict[str, Any]] = {
            policy: {
                "point_lower": sum(group["y"][policy] for group in groups) / float(point_n),
                "point_upper": sum(group["z"][policy] for group in groups) / float(point_n),
                "lower": [],
                "upper": [],
            }
            for policy in policies
        }
        rng = random.Random(_stable_seed(seed, "lineage_bundle_bootstrap", task))
        for _ in range(bootstrap_replicates):
            total_n = 0
            total_y = {policy: 0 for policy in policies}
            total_z = {policy: 0 for policy in policies}
            for _ in range(len(groups)):
                group = groups[rng.randrange(len(groups))]
                total_n += group["n"]
                for policy in policies:
                    total_y[policy] += group["y"][policy]
                    total_z[policy] += group["z"][policy]
            # At least one source lineage is required by the design builder.
            for policy in policies:
                task_samples[policy]["lower"].append(total_y[policy] / float(total_n))
                task_samples[policy]["upper"].append(total_z[policy] / float(total_n))
        samples[task] = task_samples
    return samples


def _task_stratified_bootstrap_from_design(
    design: Mapping[str, Any], bootstrap_replicates: int, seed: int, task_alpha: float, macro_alpha: float
) -> Tuple[Dict[str, Any], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < task_alpha < 1.0 or not 0.0 < macro_alpha < 1.0:
        raise ValueError("bootstrap alpha values must be strictly between zero and one")
    samples = _bootstrap_endpoint_rates(design, bootstrap_replicates, seed)
    rows: List[Dict[str, Any]] = []
    index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    policies = design["policies"]
    tasks = design["tasks"]
    for task in tasks:
        for policy_a, policy_b in combinations(policies, 2):
            a_over_b = [
                a_lower - b_upper
                for a_lower, b_upper in zip(samples[task][policy_a]["lower"], samples[task][policy_b]["upper"])
            ]
            b_over_a = [
                b_lower - a_upper
                for b_lower, a_upper in zip(samples[task][policy_b]["lower"], samples[task][policy_a]["upper"])
            ]
            row = {
                "scope": task,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "method": "task_stratified_lineage_bundle_bootstrap",
                "bootstrap_replicates": bootstrap_replicates,
                "one_sided_alpha": task_alpha,
                "a_over_b": {
                    "point_contrast": samples[task][policy_a]["point_lower"] - samples[task][policy_b]["point_upper"],
                    "lower_confidence_bound": _one_sided_bootstrap_lower(a_over_b, task_alpha),
                },
                "b_over_a": {
                    "point_contrast": samples[task][policy_b]["point_lower"] - samples[task][policy_a]["point_upper"],
                    "lower_confidence_bound": _one_sided_bootstrap_lower(b_over_a, task_alpha),
                },
            }
            rows.append(row)
            index[(task, policy_a, policy_b)] = row
    for policy_a, policy_b in combinations(policies, 2):
        a_over_b = [
            sum(samples[task][policy_a]["lower"][replicate] - samples[task][policy_b]["upper"][replicate] for task in tasks)
            / float(len(tasks))
            for replicate in range(bootstrap_replicates)
        ]
        b_over_a = [
            sum(samples[task][policy_b]["lower"][replicate] - samples[task][policy_a]["upper"][replicate] for task in tasks)
            / float(len(tasks))
            for replicate in range(bootstrap_replicates)
        ]
        row = {
            "scope": "macro",
            "policy_a": policy_a,
            "policy_b": policy_b,
            "method": "task_stratified_lineage_bundle_bootstrap_equal_task_macro",
            "bootstrap_replicates": bootstrap_replicates,
            "one_sided_alpha": macro_alpha,
            "a_over_b": {
                "point_contrast": sum(
                    samples[task][policy_a]["point_lower"] - samples[task][policy_b]["point_upper"] for task in tasks
                )
                / float(len(tasks)),
                "lower_confidence_bound": _one_sided_bootstrap_lower(a_over_b, macro_alpha),
            },
            "b_over_a": {
                "point_contrast": sum(
                    samples[task][policy_b]["point_lower"] - samples[task][policy_a]["point_upper"] for task in tasks
                )
                / float(len(tasks)),
                "lower_confidence_bound": _one_sided_bootstrap_lower(b_over_a, macro_alpha),
            },
        }
        rows.append(row)
        index[("macro", policy_a, policy_b)] = row
    return {
        "status": "computed",
        "method": "task_stratified_lineage_bundle_bootstrap",
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "task_alpha": task_alpha,
        "macro_alpha": macro_alpha,
        "design": _design_summary(design),
        "pairwise": rows,
    }, index


def task_stratified_lineage_bundle_bootstrap(
    episodes: Sequence[Mapping[str, Any]],
    policies: Sequence[str],
    tasks: Sequence[str],
    bootstrap_replicates: int = 10000,
    seed: int = 20260918,
    task_alpha: float = 0.05 / 75.0,
    macro_alpha: float = 0.05 / 15.0,
    min_lineage_clusters: int = 10,
) -> Dict[str, Any]:
    """Bootstrap conservative ``L_A-U_B`` contrasts by source lineage.

    Resampling happens independently within each fixed task and carries every
    policy's outcomes for a selected start bundle together.  The result is
    descriptive when source clusters contain multiple starts or the number of
    lineages is weak; it does not itself grant a supported ordering.
    """

    design = _lineage_bundle_design(episodes, policies, tasks, min_lineage_clusters)
    if design["status"] != "computed":
        return _design_summary(design)
    report, _ = _task_stratified_bootstrap_from_design(
        design, bootstrap_replicates, seed, task_alpha, macro_alpha
    )
    return report


def _one_sided_exact_binomial_endpoint(successes: int, n: int, alpha: float, side: str) -> Dict[str, Any]:
    """Optional one-sided Clopper--Pearson endpoint, never approximated."""

    if n <= 0:
        return {"status": "unavailable", "reason": "Exact endpoint is undefined for n=0."}
    if successes < 0 or successes > n:
        raise ValueError("successes must be between zero and n")
    if side not in ("lower", "upper") or not 0.0 < alpha < 1.0:
        raise ValueError("side must be lower/upper and alpha must be strictly between zero and one")
    try:
        from scipy.stats import beta  # type: ignore
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "SciPy is not installed; no exact Clopper--Pearson endpoint was substituted.",
        }
    if side == "lower":
        value = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, n - successes + 1))
    else:
        value = 1.0 if successes == n else float(beta.ppf(1.0 - alpha, successes + 1, n - successes))
    return {
        "status": "computed",
        "method": "one_sided_clopper_pearson",
        "side": side,
        "alpha": alpha,
        "successes": successes,
        "n": n,
        "value": value,
    }


def _endpoint_envelopes_from_design(
    design: Mapping[str, Any], endpoint_alpha: float
) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]:
    if design["status"] != "computed":
        return _design_summary(design), {}
    if not design["independent_starts"]:
        return (
            {
                "status": "unavailable",
                "reason": "Simultaneous exact binomial endpoint envelopes require independent starts; clustered lineages remain indeterminate.",
                "design": _design_summary(design),
            },
            {},
        )
    rows: List[Dict[str, Any]] = []
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for task in design["tasks"]:
        groups = design["task_designs"][task]["groups"]
        n = sum(group["n"] for group in groups)
        for policy in design["policies"]:
            lower_y = _one_sided_exact_binomial_endpoint(
                sum(group["y"][policy] for group in groups), n, endpoint_alpha, "lower"
            )
            upper_z = _one_sided_exact_binomial_endpoint(
                sum(group["z"][policy] for group in groups), n, endpoint_alpha, "upper"
            )
            if lower_y["status"] != "computed" or upper_z["status"] != "computed":
                return (
                    {
                        "status": "unavailable",
                        "reason": lower_y.get("reason") or upper_z.get("reason"),
                        "design": _design_summary(design),
                    },
                    {},
                )
            row = {
                "policy": policy,
                "task": task,
                "n": n,
                "lower_y": lower_y,
                "upper_z": upper_z,
            }
            rows.append(row)
            index[(policy, task)] = row
    return (
        {
            "status": "computed",
            "method": "simultaneous_one_sided_clopper_pearson_endpoint_envelopes",
            "family_tail_alpha": endpoint_alpha,
            "family": "30 cells x lower/upper endpoints",
            "design": _design_summary(design),
            "cells": rows,
        },
        index,
    )


def bonferroni_endpoint_envelopes(
    episodes: Sequence[Mapping[str, Any]],
    policies: Sequence[str],
    tasks: Sequence[str],
    endpoint_alpha: float = 0.05 / 60.0,
    min_lineage_clusters: int = 10,
) -> Dict[str, Any]:
    """Return optional SciPy exact endpoint envelopes for independent starts."""

    design = _lineage_bundle_design(episodes, policies, tasks, min_lineage_clusters)
    report, _ = _endpoint_envelopes_from_design(design, endpoint_alpha)
    return report


def paired_macro_sign_test(
    episodes: Sequence[Mapping[str, Any]],
    policy_a: str,
    policy_b: str,
    tasks: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Exact fixed-five-task paired label-swap test for the Y endpoint.

    With five equally sized task panels, every discordant matched start has the
    same signed contribution to the equal-task macro difference.  The exact
    sign-swap null is therefore binomial over pooled discordant pairs.
    """

    selected_tasks = list(tasks or PRIMARY_TASK_IDS)
    selected_tasks = list(dict.fromkeys(_normalise_task(task) for task in selected_tasks))
    if set(selected_tasks) != set(PRIMARY_TASK_IDS) or len(selected_tasks) != len(PRIMARY_TASK_IDS):
        return {
            "status": "unavailable",
            "reason": "The exact macro sign-test helper is defined for the fixed five benchmark tasks.",
        }
    policy_a = _normalise_policy(policy_a)
    policy_b = _normalise_policy(policy_b)
    design = _lineage_bundle_design(episodes, [policy_a, policy_b], selected_tasks, min_lineage_clusters=2)
    if design["status"] != "computed":
        return _design_summary(design)
    if not design["independent_starts"]:
        return {
            "status": "unavailable",
            "reason": "The exact individual-sign null is not valid for multi-start source lineages.",
            "design": _design_summary(design),
        }
    per_task: Dict[str, Any] = {}
    task_sizes: List[int] = []
    a_only = b_only = 0
    for task in selected_tasks:
        groups = design["task_designs"][task]["groups"]
        task_a_only = task_b_only = task_n = 0
        for group in groups:
            # independent_starts means this group has precisely one bundle.
            outcomes = group["bundles"][0]["outcomes"]
            a_value = outcomes[policy_a]["y"]
            b_value = outcomes[policy_b]["y"]
            task_n += 1
            if a_value == 1 and b_value == 0:
                task_a_only += 1
            elif a_value == 0 and b_value == 1:
                task_b_only += 1
        task_sizes.append(task_n)
        a_only += task_a_only
        b_only += task_b_only
        per_task[task] = {"paired_n": task_n, "a_only": task_a_only, "b_only": task_b_only}
    if not task_sizes or len(set(task_sizes)) != 1:
        return {
            "status": "unavailable",
            "reason": "Equal task panel sizes are required for the pooled exact macro sign-swap null.",
            "per_task": per_task,
            "design": _design_summary(design),
        }
    paired_n = sum(task_sizes)
    return {
        "status": "computed",
        "method": "task_stratified_within_start_label_swap_exact_binomial",
        "policy_a": policy_a,
        "policy_b": policy_b,
        "tasks": selected_tasks,
        "n_per_task": task_sizes[0],
        "paired_n": paired_n,
        "a_only": a_only,
        "b_only": b_only,
        "discordant": a_only + b_only,
        "macro_observed_positive_difference": (a_only - b_only) / float(paired_n),
        "p_value": exact_mcnemar_pvalue(a_only, b_only),
        "family_alpha": 0.05 / 15.0,
        "passes_bonferroni_exact_test": exact_mcnemar_pvalue(a_only, b_only) <= 0.05 / 15.0,
        "per_task": per_task,
        "design": _design_summary(design),
        "note": "A significant paired endpoint contrast alone does not establish unconditional virtual-success ordering.",
    }


def _macro_endpoint_separation(
    endpoint_index: Mapping[Tuple[str, str], Mapping[str, Any]],
    policy_a: str,
    policy_b: str,
    tasks: Sequence[str],
) -> Dict[str, Any]:
    if not endpoint_index:
        return {"status": "unavailable", "reason": "Exact endpoint envelopes are unavailable."}
    try:
        lower_a = _safe_mean(endpoint_index[(policy_a, task)]["lower_y"]["value"] for task in tasks)
        upper_b = _safe_mean(endpoint_index[(policy_b, task)]["upper_z"]["value"] for task in tasks)
        lower_b = _safe_mean(endpoint_index[(policy_b, task)]["lower_y"]["value"] for task in tasks)
        upper_a = _safe_mean(endpoint_index[(policy_a, task)]["upper_z"]["value"] for task in tasks)
    except KeyError:
        return {"status": "unavailable", "reason": "A task endpoint envelope is missing."}
    return {
        "status": "computed",
        "a_over_b": {"lower_l": lower_a, "upper_u": upper_b, "separated": lower_a is not None and upper_b is not None and lower_a > upper_b},
        "b_over_a": {"lower_l": lower_b, "upper_u": upper_a, "separated": lower_b is not None and upper_a is not None and lower_b > upper_a},
    }


def _advanced_pairwise_inference(
    episodes: Sequence[Mapping[str, Any]], policies: Sequence[str], tasks: Sequence[str], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Run the expensive, explicitly requested ranking inference bundle."""

    bootstrap_replicates = int(config.get("bootstrap_replicates", 10000))
    seed = int(config.get("seed", 20260918))
    min_lineage_clusters = int(config.get("min_lineage_clusters", 10))
    endpoint_alpha = float(config.get("endpoint_alpha", 0.05 / 60.0))
    design = _lineage_bundle_design(episodes, policies, tasks, min_lineage_clusters)
    if design["status"] != "computed":
        return {"status": "unavailable", "reason": design.get("reason"), "design": _design_summary(design), "pairwise": []}
    bootstrap, _ = _task_stratified_bootstrap_from_design(
        design, bootstrap_replicates, seed, 0.05 / 75.0, 0.05 / 15.0
    )
    envelopes, endpoint_index = _endpoint_envelopes_from_design(design, endpoint_alpha)
    support_replicates = bootstrap_replicates >= 10000
    rows: List[Dict[str, Any]] = []
    for bootstrap_row in bootstrap["pairwise"]:
        scope = bootstrap_row["scope"]
        policy_a = bootstrap_row["policy_a"]
        policy_b = bootstrap_row["policy_b"]
        if scope == "macro":
            endpoint_guard = _macro_endpoint_separation(endpoint_index, policy_a, policy_b, design["tasks"])
            cluster_or_weak = bool(design["clustered_tasks"] or design["weak_lineage_tasks"])
            sign_test = paired_macro_sign_test(episodes, policy_a, policy_b, design["tasks"])
        else:
            if endpoint_index:
                a_cell = endpoint_index[(policy_a, scope)]
                b_cell = endpoint_index[(policy_b, scope)]
                endpoint_guard = {
                    "status": "computed",
                    "a_over_b": {
                        "lower_l": a_cell["lower_y"]["value"],
                        "upper_u": b_cell["upper_z"]["value"],
                        "separated": a_cell["lower_y"]["value"] > b_cell["upper_z"]["value"],
                    },
                    "b_over_a": {
                        "lower_l": b_cell["lower_y"]["value"],
                        "upper_u": a_cell["upper_z"]["value"],
                        "separated": b_cell["lower_y"]["value"] > a_cell["upper_z"]["value"],
                    },
                }
            else:
                endpoint_guard = {"status": "unavailable", "reason": envelopes.get("reason", "Exact endpoint envelopes are unavailable.")}
            cluster_or_weak = scope in design["clustered_tasks"] or scope in design["weak_lineage_tasks"]
            sign_test = None
        a_supported = bool(
            support_replicates
            and not cluster_or_weak
            and endpoint_guard.get("status") == "computed"
            and bootstrap_row["a_over_b"]["lower_confidence_bound"] is not None
            and bootstrap_row["a_over_b"]["lower_confidence_bound"] > 0.0
            and endpoint_guard["a_over_b"]["separated"]
        )
        b_supported = bool(
            support_replicates
            and not cluster_or_weak
            and endpoint_guard.get("status") == "computed"
            and bootstrap_row["b_over_a"]["lower_confidence_bound"] is not None
            and bootstrap_row["b_over_a"]["lower_confidence_bound"] > 0.0
            and endpoint_guard["b_over_a"]["separated"]
        )
        if a_supported:
            ordering = policy_a + "_over_" + policy_b
        elif b_supported:
            ordering = policy_b + "_over_" + policy_a
        else:
            ordering = "indeterminate"
        reason = None
        if ordering == "indeterminate":
            if cluster_or_weak:
                reason = "Clustered or weak lineage designs remain indeterminate without a separately validated boundary-aware method."
            elif not support_replicates:
                reason = "At least 10,000 bootstrap replicates are required for a supported ordering."
            elif endpoint_guard.get("status") != "computed":
                reason = endpoint_guard.get("reason", "Exact endpoint guard is unavailable.")
            else:
                reason = "The conservative bootstrap lower bound and simultaneous endpoint guard do not both separate an ordering."
        row = {
            "scope": scope,
            "policy_a": policy_a,
            "policy_b": policy_b,
            "bootstrap": bootstrap_row,
            "endpoint_envelope_guard": endpoint_guard,
            "macro_sign_test": sign_test,
            "ordering": ordering,
            "supported": ordering != "indeterminate",
            "reason": reason,
        }
        rows.append(row)
    return {
        "status": "computed" if envelopes.get("status") == "computed" else "computed_with_unavailable_endpoint_guard",
        "bootstrap": {key: value for key, value in bootstrap.items() if key != "pairwise"},
        "endpoint_envelopes": envelopes,
        "pairwise": rows,
    }


def _merge_advanced_pairwise(
    pairwise: Sequence[Dict[str, Any]], advanced: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    advanced_rows = {
        (row["scope"], row["policy_a"], row["policy_b"]): row for row in advanced.get("pairwise", [])
    }
    merged: List[Dict[str, Any]] = []
    for row in pairwise:
        copied = dict(row)
        evidence = advanced_rows.get((row["scope"], row["policy_a"], row["policy_b"]))
        if evidence is not None:
            copied["advanced_inference"] = evidence
            copied["ordering"] = evidence["ordering"]
            copied["supported"] = evidence["supported"]
            copied["ordering_reason"] = evidence["reason"]
            if row["scope"] == "macro" and evidence.get("macro_sign_test") is not None:
                copied["paired"] = evidence["macro_sign_test"]
        merged.append(copied)
    return merged


def _macro_pairwise(
    cells: Mapping[Tuple[str, str], Mapping[str, Any]],
    policy_a: str,
    policy_b: str,
    tasks: Sequence[str],
) -> Dict[str, Any]:
    a_macro = _policy_macro(policy_a, tasks, cells)
    b_macro = _policy_macro(policy_b, tasks, cells)
    difference = None
    if a_macro["observed_positive_rate"] is not None and b_macro["observed_positive_rate"] is not None:
        difference = a_macro["observed_positive_rate"] - b_macro["observed_positive_rate"]
    return {
        "scope": "macro",
        "policy_a": policy_a,
        "policy_b": policy_b,
        "observed_positive_difference": difference,
        "paired": {
            "status": "unavailable",
            "reason": "The required task-stratified within-start label-swap test is not run in live summaries.",
            "family_alpha": 0.05 / 15.0,
        },
        "status": "unavailable",
        "ordering": "indeterminate",
        "supported": False,
        "ordering_reason": "Macro ordering requires the frozen task-stratified randomization and boundary-aware inference.",
    }


def _rankings(cells: Mapping[Tuple[str, str], Mapping[str, Any]], policies: Sequence[str], tasks: Sequence[str]) -> List[Dict[str, Any]]:
    rankings: List[Dict[str, Any]] = []
    scopes = list(tasks) + ["macro"]
    macro_rows = {row["policy"]: row for row in (_policy_macro(policy, tasks, cells) for policy in policies)}
    for scope in scopes:
        rows: List[Dict[str, Any]] = []
        for policy in policies:
            if scope == "macro":
                row = macro_rows[policy]
                estimate = row["observed_positive_rate"]
                reference = row["reference_rate"]
            else:
                cell = cells[(policy, scope)]
                estimate = cell["positive_rate"]
                reference = cell["reference_rate"]
            rows.append({"policy": policy, "estimate": estimate, "reference_rate": reference})
        visual = sorted(rows, key=lambda row: (row["estimate"] is None, -(row["estimate"] or 0.0), row["policy"]))
        rankings.append(
            {
                "scope": scope,
                "visual_order": [row["policy"] for row in visual],
                "estimates": rows,
                "ordering": "indeterminate",
                "supported": False,
                "reason": "Visual sorting is not a supported policy ordering without the frozen conservative inference procedure.",
            }
        )
    return rankings


def _human_aggregate_order(policies: Sequence[str]) -> List[str]:
    frozen_position = {policy: index for index, policy in enumerate(PRIMARY_POLICY_IDS)}
    return sorted(
        policies,
        key=lambda policy: (
            -sum(HUMAN[policy].values()) if policy in HUMAN else 0,
            frozen_position.get(policy, len(frozen_position)),
            policy,
        ),
    )


def _mmrv_vectors(
    cells: Mapping[Tuple[str, str], Mapping[str, Any]],
    policies: Sequence[str],
    tasks: Sequence[str],
    scenario: str,
    complete_case: bool = False,
) -> Optional[Dict[str, Any]]:
    if any(policy not in HUMAN for policy in policies) or any(task not in TASK_IDS for task in tasks):
        return None
    top_three = set(_human_aggregate_order(list(policies))[:3])
    sim_by_task: Dict[str, List[float]] = {}
    real_by_task: Dict[str, List[float]] = {}
    for task in tasks:
        sim_values: List[float] = []
        real_values: List[float] = []
        for policy in policies:
            cell = cells.get((policy, task))
            if cell is None or cell["n"] == 0:
                return None
            if complete_case:
                value = cell["rate"]
            else:
                missing = int(cell["n"]) - int(cell["valid"])
                numerator = int(cell["successes"])
                if scenario == "all_missing_succeed":
                    numerator += missing
                elif scenario == "missing_favors_human_order" and policy in top_three:
                    numerator += missing
                elif scenario == "missing_reverses_human_order" and policy not in top_three:
                    numerator += missing
                value = numerator / float(cell["n"])
            if value is None:
                return None
            sim_values.append(float(value))
            real_values.append(HUMAN[policy][task] / float(REFERENCE_TRIALS_PER_CELL))
        sim_by_task[task] = sim_values
        real_by_task[task] = real_values
    macro_sim = [_safe_mean(sim_by_task[task][i] for task in tasks) for i in range(len(policies))]
    macro_real = [_safe_mean(real_by_task[task][i] for task in tasks) for i in range(len(policies))]
    if any(value is None for value in macro_sim + macro_real):
        return None
    per_task = {
        task: mean_maximum_rank_violation(sim_by_task[task], real_by_task[task]) for task in tasks
    }
    return {
        "macro": mean_maximum_rank_violation(macro_sim, macro_real),
        "per_task": per_task,
        "task_summary_mean": _safe_mean(per_task.values()),
        "macro_sim": macro_sim,
        "macro_real": macro_real,
        "sim_by_task": sim_by_task,
        "real_by_task": real_by_task,
    }


def _mmrv_report(
    cells: Mapping[Tuple[str, str], Mapping[str, Any]],
    policies: Sequence[str],
    tasks: Sequence[str],
    include_completion_sensitivity: bool = False,
    include_random_order_baseline: bool = False,
) -> Dict[str, Any]:
    if len(policies) != 6 or set(policies) != set(PRIMARY_POLICY_IDS) or not tasks:
        return {
            "status": "unavailable",
            "reason": "The prescribed MMRV report requires the six frozen benchmark policies and at least one reference task.",
        }
    primary = _mmrv_vectors(cells, policies, tasks, "all_missing_fail")
    complete_case = _mmrv_vectors(cells, policies, tasks, "all_missing_fail", complete_case=True)
    if primary is None:
        return {
            "status": "unavailable",
            "reason": "At least one required primary cell has no denominator.",
            "completion_sensitivity": {
                "status": "not_computed",
                "reason": "No complete primary vector is available for completion sensitivity.",
            },
        }
    result: Dict[str, Any] = {
        "status": "computed",
        "definition": "SIMPLER mean_maximum_rank_violation(perf_sim, perf_real)",
        "policy_order": list(policies),
        "primary_observed_positive": {
            key: value
            for key, value in primary.items()
            if key not in ("macro_sim", "macro_real", "sim_by_task", "real_by_task")
        },
        "complete_case": (
            {
                key: value
                for key, value in complete_case.items()
                if key not in ("macro_sim", "macro_real", "sim_by_task", "real_by_task")
            }
            if complete_case is not None
            else {"status": "unavailable", "reason": "Complete-case MMRV is undefined when a required cell has V=0."}
        ),
    }
    if include_completion_sensitivity:
        scenarios = (
            "all_missing_fail",
            "all_missing_succeed",
            "missing_favors_human_order",
            "missing_reverses_human_order",
        )
        sensitivity: Dict[str, Any] = {}
        for scenario in scenarios:
            report = _mmrv_vectors(cells, policies, tasks, scenario)
            sensitivity[scenario] = (
                {key: value for key, value in report.items() if key not in ("macro_sim", "macro_real", "sim_by_task", "real_by_task")}
                if report is not None
                else {"status": "unavailable", "reason": "A required cell has no planned denominator."}
            )
        result["completion_sensitivity"] = sensitivity
    else:
        result["completion_sensitivity"] = {
            "status": "not_computed",
            "reason": "Set protocol.mmrv.completion_sensitivity=true for deterministic missing-outcome scenarios.",
        }
    if include_random_order_baseline:
        macro_baseline = mmrv_random_order_baseline(primary["macro_sim"], primary["macro_real"])
        task_baselines = {
            task: mmrv_random_order_baseline(primary["sim_by_task"][task], primary["real_by_task"][task])
            for task in tasks
        }
        result["random_order_baseline"] = {
            "macro": macro_baseline,
            "per_task": task_baselines,
            "task_summary_mean": _safe_mean(baseline["mean"] for baseline in task_baselines.values()),
            "note": "Each task baseline enumerates 6! policy-label permutations with its fixed human vector.",
        }
    else:
        result["random_order_baseline"] = {
            "status": "not_computed",
            "reason": "Set protocol.mmrv.random_order_baseline=true to enumerate 6! permutations per task and macro.",
        }
    return result


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(seed)] + [str(part) for part in parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in left_delta) * sum(value * value for value in right_delta))
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def _balanced_lineage_split(lineages: Mapping[str, Sequence[str]], seed: int) -> Tuple[List[str], List[str]]:
    """Find a deterministic closest-to-half split while keeping lineages whole."""

    items = [(lineage, sorted(starts)) for lineage, starts in lineages.items()]
    random.Random(seed).shuffle(items)
    total = sum(len(starts) for _, starts in items)
    # subset sum is tiny (normally <=50 starts) and avoids a greedy split that
    # needlessly violates the requested balance.
    reachable: Dict[int, Tuple[int, ...]] = {0: ()}
    for index, (_, starts) in enumerate(items):
        size = len(starts)
        for count, selected in list(reachable.items()):
            candidate = count + size
            if candidate not in reachable:
                reachable[candidate] = selected + (index,)
    candidates = [count for count in reachable if 0 < count < total]
    if not candidates:
        return [], []
    selected_count = min(candidates, key=lambda count: (abs(2 * count - total), count))
    selected = set(reachable[selected_count])
    half_a = sorted(start for index, (_, starts) in enumerate(items) if index in selected for start in starts)
    half_b = sorted(start for index, (_, starts) in enumerate(items) if index not in selected for start in starts)
    return half_a, half_b


def split_half_reliability(
    episodes: Sequence[Mapping[str, Any]],
    policies: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    repetitions: int = 1000,
    seed: int = 20260918,
) -> Dict[str, Any]:
    """Compute deterministic lineage-safe split-half descriptive reliability.

    Repeated split quantiles are a distribution of resamplings, not a confidence
    interval.  Constant score vectors yield ``undefined`` rather than a false
    perfect correlation.
    """

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    selected_policies = [_normalise_policy(policy) for policy in (policies or PRIMARY_POLICY_IDS)]
    selected_tasks = [_normalise_task(task) for task in (tasks or PRIMARY_TASK_IDS)]
    selected_policies = list(dict.fromkeys(selected_policies))
    selected_tasks = list(dict.fromkeys(selected_tasks))
    outcomes: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    lineages_by_task_start: Dict[Tuple[str, str], str] = {}
    missing_lineage = 0
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise TypeError("episodes must contain mappings")
        policy = _normalise_policy(episode.get("policy"))
        task = _normalise_task(episode.get("task"))
        start = episode.get("start_id")
        if policy not in selected_policies or task not in selected_tasks or start is None or str(start).strip() == "":
            continue
        start_id = str(start)
        lineage = episode.get("start_lineage_id")
        if lineage is None or str(lineage).strip() == "":
            missing_lineage += 1
            continue
        lineage_by_key = lineages_by_task_start.get((task, start_id))
        if lineage_by_key is not None and lineage_by_key != str(lineage):
            return {
                "status": "unavailable",
                "reason": "One task/start_id maps to multiple source lineages.",
                "task": task,
                "start_id": start_id,
            }
        lineages_by_task_start[(task, start_id)] = str(lineage)
        outcome = _episode_outcome(episode)
        outcomes[(policy, task, start_id)].append(1 if outcome["evaluable"] and outcome["success"] else 0)
    if missing_lineage:
        return {
            "status": "unavailable",
            "reason": "Split-half reliability cannot guarantee no lineage crosses halves while source lineages are missing.",
            "missing_lineage_records": missing_lineage,
        }
    tasks_by_lineage: Dict[str, set] = defaultdict(set)
    for (task, _), lineage in lineages_by_task_start.items():
        tasks_by_lineage[lineage].add(task)
    cross_task_lineages = [
        {"start_lineage_id": lineage, "tasks": sorted(task_set)}
        for lineage, task_set in sorted(tasks_by_lineage.items())
        if len(task_set) > 1
    ]
    if cross_task_lineages:
        return {
            "status": "unavailable",
            "reason": "A source lineage spans selected tasks, so independent per-task half assignments could leak it across halves.",
            "cross_task_lineages": cross_task_lineages,
        }

    starts_by_task: Dict[str, List[str]] = {}
    lineage_groups: Dict[str, Dict[str, List[str]]] = {}
    duplicates: List[Dict[str, str]] = []
    incomplete_bundles: List[Dict[str, Any]] = []
    for task in selected_tasks:
        starts = sorted(start for current_task, start in lineages_by_task_start if current_task == task)
        starts_by_task[task] = starts
        groups: Dict[str, List[str]] = defaultdict(list)
        for start in starts:
            groups[lineages_by_task_start[(task, start)]].append(start)
            for policy in selected_policies:
                values = outcomes.get((policy, task, start), [])
                if len(values) != 1:
                    target = {"task": task, "start_id": start, "policy": policy, "records": len(values)}
                    if len(values) > 1:
                        duplicates.append(target)
                    else:
                        incomplete_bundles.append(target)
        lineage_groups[task] = groups
    too_small = [task for task, starts in starts_by_task.items() if len(starts) < 2]
    if duplicates or incomplete_bundles or too_small:
        return {
            "status": "unavailable",
            "reason": "Each selected task needs at least two complete, unique policy bundles before split-half reliability is meaningful.",
            "duplicate_records": duplicates,
            "incomplete_bundles": incomplete_bundles,
            "too_small_tasks": too_small,
        }

    correlations: List[float] = []
    undefined_splits = 0
    disagreements: List[float] = []
    stable_orders = 0
    comparable_orders = 0
    half_sizes: List[Dict[str, Dict[str, int]]] = []
    for repetition in range(repetitions):
        vector_a: List[float] = []
        vector_b: List[float] = []
        policy_task_rates_a: Dict[Tuple[str, str], float] = {}
        policy_task_rates_b: Dict[Tuple[str, str], float] = {}
        sizes: Dict[str, Dict[str, int]] = {}
        for task in selected_tasks:
            half_a, half_b = _balanced_lineage_split(lineage_groups[task], _stable_seed(seed, repetition, task))
            if not half_a or not half_b:
                return {
                    "status": "unavailable",
                    "reason": "A task cannot be divided into two non-empty lineage-safe halves.",
                    "task": task,
                }
            sizes[task] = {"a": len(half_a), "b": len(half_b)}
            for policy in selected_policies:
                rate_a = sum(outcomes[(policy, task, start)][0] for start in half_a) / float(len(half_a))
                rate_b = sum(outcomes[(policy, task, start)][0] for start in half_b) / float(len(half_b))
                vector_a.append(rate_a)
                vector_b.append(rate_b)
                policy_task_rates_a[(policy, task)] = rate_a
                policy_task_rates_b[(policy, task)] = rate_b
        correlation = _pearson(vector_a, vector_b)
        if correlation is None:
            undefined_splits += 1
        else:
            correlations.append(correlation)
        disagreements.append(sum(abs(a - b) for a, b in zip(vector_a, vector_b)) / len(vector_a))
        policy_macro_a = {policy: _safe_mean(policy_task_rates_a[(policy, task)] for task in selected_tasks) for policy in selected_policies}
        policy_macro_b = {policy: _safe_mean(policy_task_rates_b[(policy, task)] for task in selected_tasks) for policy in selected_policies}
        for policy_a, policy_b in combinations(selected_policies, 2):
            direction_a = (policy_macro_a[policy_a] or 0.0) - (policy_macro_a[policy_b] or 0.0)
            direction_b = (policy_macro_b[policy_a] or 0.0) - (policy_macro_b[policy_b] or 0.0)
            if direction_a == 0.0 or direction_b == 0.0:
                continue
            comparable_orders += 1
            if (direction_a > 0.0) == (direction_b > 0.0):
                stable_orders += 1
        half_sizes.append(sizes)
    if not correlations:
        status = "undefined"
    elif undefined_splits:
        status = "computed_with_undefined_splits"
    else:
        status = "computed"
    return {
        "status": status,
        "method": "deterministic_balanced_lineage_split_half",
        "repetitions": repetitions,
        "seed": seed,
        "cell_vector_length": len(selected_policies) * len(selected_tasks),
        "correlations": correlations,
        "undefined_splits": undefined_splits,
        "mean_correlation": _safe_mean(correlations),
        "rate_disagreements": disagreements,
        "mean_rate_disagreement": _safe_mean(disagreements),
        "pairwise_order_stability": stable_orders / float(comparable_orders) if comparable_orders else None,
        "comparable_pairwise_orders": comparable_orders,
        "half_sizes": half_sizes,
        "lineage_leakage_detected": False,
        "confidence_interval": {
            "status": "unavailable",
            "reason": "An outer lineage bootstrap is required; split quantiles are not confidence intervals.",
        },
        "note": "Repeated splits reuse episodes and are not independent studies.",
    }


def _percentile_interval(values: Sequence[float], confidence: float = 0.95) -> Optional[Dict[str, Any]]:
    if not values or not 0.0 < confidence < 1.0:
        return None
    ordered = sorted(float(value) for value in values)
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, math.ceil(tail * len(ordered)) - 1)
    upper_index = min(len(ordered) - 1, math.floor((1.0 - tail) * len(ordered)) - 1)
    return {
        "method": "outer_lineage_bootstrap_percentile",
        "confidence": confidence,
        "replicates": len(ordered),
        "lower": ordered[lower_index],
        "upper": ordered[upper_index],
    }


def outer_lineage_bootstrap_reliability(
    episodes: Sequence[Mapping[str, Any]],
    policies: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    outer_replicates: int = 1000,
    inner_repetitions: int = 1000,
    seed: int = 20260918,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Outer lineage bootstrap for split-half uncertainty (explicitly opt-in).

    Each outer draw resamples whole source lineages within task.  Repeatedly
    selected lineages are assigned a fresh bootstrap lineage identity, while
    all policies and starts within that selected lineage remain together.  This
    prevents accidental splitting of a source cluster and avoids treating the
    existing 1,000 split quantiles as a confidence interval.
    """

    if outer_replicates <= 0 or inner_repetitions <= 0:
        raise ValueError("outer_replicates and inner_repetitions must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    selected_policies = list(dict.fromkeys(_normalise_policy(policy) for policy in (policies or PRIMARY_POLICY_IDS)))
    selected_tasks = list(dict.fromkeys(_normalise_task(task) for task in (tasks or PRIMARY_TASK_IDS)))
    design = _lineage_bundle_design(episodes, selected_policies, selected_tasks, min_lineage_clusters=2)
    if design["status"] != "computed":
        return _design_summary(design)
    too_few = [task for task in selected_tasks if len(design["task_designs"][task]["groups"]) < 2]
    if too_few:
        return {
            "status": "unavailable",
            "reason": "At least two source lineages per task are required for an outer lineage bootstrap.",
            "too_few_lineage_tasks": too_few,
            "design": _design_summary(design),
        }
    base = split_half_reliability(
        episodes, policies=selected_policies, tasks=selected_tasks, repetitions=inner_repetitions, seed=seed
    )
    if base.get("status") == "undefined":
        return {
            "status": "undefined",
            "reason": "The original split-half score vector is constant; an outer bootstrap cannot turn it into reliability.",
            "base": base,
            "design": _design_summary(design),
        }
    if base.get("status") == "unavailable":
        return {"status": "unavailable", "reason": base.get("reason"), "base": base, "design": _design_summary(design)}

    statistic_values: Dict[str, List[float]] = {
        "mean_correlation": [],
        "mean_rate_disagreement": [],
        "pairwise_order_stability": [],
    }
    undefined_outer = 0
    for outer_index in range(outer_replicates):
        bootstrap_records: List[Dict[str, Any]] = []
        for task in selected_tasks:
            groups = design["task_designs"][task]["groups"]
            rng = random.Random(_stable_seed(seed, "outer_lineage_bootstrap", outer_index, task))
            for draw_index in range(len(groups)):
                group = groups[rng.randrange(len(groups))]
                lineage_id = "outer:%s:%d:%d" % (task, outer_index, draw_index)
                for bundle_index, bundle in enumerate(group["bundles"]):
                    start_id = "outer:%s:%d:%d:%d" % (task, outer_index, draw_index, bundle_index)
                    for policy in selected_policies:
                        bootstrap_records.append(
                            {
                                "policy": policy,
                                "task": task,
                                "start_id": start_id,
                                "start_lineage_id": lineage_id,
                                "status": "completed",
                                "validity": "valid",
                                "binary_success": bool(bundle["outcomes"][policy]["y"]),
                                "progress_score": None,
                            }
                        )
        inner = split_half_reliability(
            bootstrap_records,
            policies=selected_policies,
            tasks=selected_tasks,
            repetitions=inner_repetitions,
            seed=_stable_seed(seed, "outer_inner", outer_index),
        )
        if inner.get("status") in ("unavailable", "undefined"):
            undefined_outer += 1
            continue
        any_value = False
        for key, values in statistic_values.items():
            value = inner.get(key)
            if value is not None:
                values.append(float(value))
                any_value = True
        if not any_value:
            undefined_outer += 1
    intervals = {key: _percentile_interval(values, confidence) for key, values in statistic_values.items()}
    if not statistic_values["mean_correlation"]:
        status = "undefined"
    elif undefined_outer:
        status = "computed_with_undefined_outer_replicates"
    elif outer_replicates < 200:
        status = "exploratory_low_replicate_count"
    else:
        status = "computed"
    return {
        "status": status,
        "method": "outer_task_stratified_lineage_bootstrap_of_split_half_reliability",
        "outer_replicates": outer_replicates,
        "inner_repetitions": inner_repetitions,
        "seed": seed,
        "confidence": confidence,
        "base": {
            "status": base.get("status"),
            "mean_correlation": base.get("mean_correlation"),
            "mean_rate_disagreement": base.get("mean_rate_disagreement"),
            "pairwise_order_stability": base.get("pairwise_order_stability"),
        },
        "intervals": intervals,
        "undefined_outer_replicates": undefined_outer,
        "design": _design_summary(design),
        "note": "Outer resampling quantifies sampling uncertainty for split-half summaries; it does not establish measurement validity.",
    }


def _joint_table(p_a: float, p_b: float, p11: float) -> Dict[str, float]:
    return {
        "both_success": p11,
        "a_only": p_a - p11,
        "b_only": p_b - p11,
        "both_failure": 1.0 - p_a - p_b + p11,
    }


def _mdd_scenarios(p_a: float, p_b: float) -> List[Tuple[str, float]]:
    lower = max(0.0, p_a + p_b - 1.0)
    upper = min(p_a, p_b)
    return [
        ("maximum_discordance", lower),
        ("midpoint_discordance", (lower + upper) / 2.0),
        ("minimum_discordance", upper),
    ]


def _simulate_mcnemar_power(
    tables: Sequence[Dict[str, float]],
    n_per_task: Sequence[int],
    alpha: float,
    simulations: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    rejects = 0
    for _ in range(simulations):
        a_only = b_only = 0
        for table, n in zip(tables, n_per_task):
            cutoff_both = table["both_success"]
            cutoff_a = cutoff_both + table["a_only"]
            cutoff_b = cutoff_a + table["b_only"]
            for _ in range(n):
                draw = rng.random()
                if draw < cutoff_both:
                    continue
                if draw < cutoff_a:
                    a_only += 1
                elif draw < cutoff_b:
                    b_only += 1
        pvalue = exact_mcnemar_pvalue(a_only, b_only)
        if pvalue <= alpha:
            rejects += 1
    return rejects / float(simulations)


def paired_mdd_power_curve(
    baseline_rates: Any,
    gap_grid: Optional[Sequence[float]] = None,
    n_per_task: Any = 50,
    alpha: float = 0.05 / 75.0,
    simulations: int = 10000,
    seed: int = 20260918,
) -> Dict[str, Any]:
    """Simulate an exact-McNemar MDD curve under feasible joint alternatives.

    A scalar baseline describes a named 50-start cell.  A five-rate vector
    describes the fixed-five-task, equal-size macro sign-swap setting.  It is
    deliberately opt-in because the required >=10,000 simulations per one
    percentage-point gap and discordance scenario are not dashboard-poll work.
    """

    if simulations < 10000:
        raise ValueError("The frozen MDD protocol requires at least 10,000 simulations per gap and scenario")
    if isinstance(baseline_rates, (int, float)) and not isinstance(baseline_rates, bool):
        baseline = [float(baseline_rates)]
        design = "paired_cell_exact_mcnemar"
    else:
        baseline = [float(value) for value in baseline_rates]
        design = "fixed_task_macro_label_swap_equal_task_sizes"
    if not baseline or not all(0.0 <= value <= 1.0 and math.isfinite(value) for value in baseline):
        raise ValueError("baseline rates must be finite probabilities")
    if isinstance(n_per_task, int):
        sample_sizes = [n_per_task] * len(baseline)
    else:
        sample_sizes = [int(value) for value in n_per_task]
    if len(sample_sizes) != len(baseline) or any(value <= 0 for value in sample_sizes):
        raise ValueError("n_per_task must be positive and match baseline_rates")
    if len(baseline) > 1 and len(set(sample_sizes)) != 1:
        raise ValueError("The macro label-swap shortcut requires equal task sample sizes")
    gaps = list(gap_grid) if gap_grid is not None else [step / 100.0 for step in range(1, 101)]
    if not gaps or any(not _finite_number(gap) or float(gap) <= 0.0 for gap in gaps):
        raise ValueError("gap_grid must contain positive finite proportions")
    rows: List[Dict[str, Any]] = []
    for gap in gaps:
        gap = float(gap)
        comparison = [rate + gap for rate in baseline]
        if any(rate > 1.0 for rate in comparison):
            rows.append({"gap": gap, "status": "infeasible", "reason": "The comparison marginal would exceed one."})
            continue
        scenario_rows: List[Dict[str, Any]] = []
        for scenario_index, scenario_name in enumerate(("maximum_discordance", "midpoint_discordance", "minimum_discordance")):
            p11_values = [_mdd_scenarios(a, b)[scenario_index][1] for a, b in zip(baseline, comparison)]
            tables = [_joint_table(a, b, p11) for a, b, p11 in zip(baseline, comparison, p11_values)]
            power = _simulate_mcnemar_power(
                tables,
                sample_sizes,
                alpha,
                simulations,
                _stable_seed(seed, gap, scenario_name),
            )
            scenario_rows.append(
                {
                    "scenario": scenario_name,
                    "power": power,
                    "monte_carlo_standard_error": math.sqrt(power * (1.0 - power) / simulations),
                    "alternative_table": tables,
                }
            )
        rows.append(
            {
                "gap": gap,
                "status": "computed",
                "scenarios": scenario_rows,
                "minimum_scenario_power": min(row["power"] for row in scenario_rows),
                "maximum_scenario_power": max(row["power"] for row in scenario_rows),
            }
        )
    passing = [row["gap"] for row in rows if row["status"] == "computed" and row["minimum_scenario_power"] >= 0.8]
    return {
        "status": "computed",
        "design": design,
        "test": "two_sided_exact_mcnemar",
        "alpha": alpha,
        "baseline_rates": baseline,
        "comparison_rule": "comparison marginal equals baseline marginal plus tested gap",
        "n_per_task": sample_sizes,
        "simulations_per_gap_scenario": simulations,
        "seed": seed,
        "rows": rows,
        "conservative_mdd": min(passing) if passing else None,
        "mdd_status": "attained" if passing else "unattainable_on_tested_grid",
        "assumptions": [
            "Discordance is varied over feasible joint-probability scenarios.",
            "The macro shortcut uses equal task sample sizes and task-stratified within-start label exchangeability.",
            "This is an observed-positive endpoint MDD, not an unconditional real-success MDD.",
        ],
    }


def _advanced_inference_status(advanced: Mapping[str, Any], reliability: Mapping[str, Any]) -> Dict[str, Any]:
    bootstrap = advanced.get("bootstrap", {}) if isinstance(advanced.get("bootstrap"), Mapping) else {}
    endpoint = advanced.get("endpoint_envelopes", {}) if isinstance(advanced.get("endpoint_envelopes"), Mapping) else {}
    outer = reliability.get("outer_bootstrap", {}) if isinstance(reliability.get("outer_bootstrap"), Mapping) else {}
    available_statuses = ("computed", "computed_with_unavailable_endpoint_guard", "computed_with_undefined_outer_replicates", "exploratory_low_replicate_count")
    overall_status = (
        "partially_available"
        if advanced.get("status") in available_statuses or outer.get("status") in available_statuses
        else "unavailable"
    )
    return {
        "status": overall_status,
        "reason": "Only explicitly requested methods below are run; no result is substituted for a method that remains unavailable.",
        "methods": [
            {
                "name": "lineage_block_bootstrap",
                "status": bootstrap.get("status", advanced.get("status", "not_computed")),
                "reason": bootstrap.get("reason", advanced.get("reason")),
            },
            {
                "name": "conservative_missingness_ordering",
                "status": advanced.get("status", "not_computed"),
                "reason": advanced.get("reason"),
            },
            {
                "name": "simultaneous_exact_endpoint_envelopes",
                "status": endpoint.get("status", "not_computed"),
                "reason": endpoint.get("reason"),
            },
            {
                "name": "reference_uncertainty_parametric_bootstrap",
                "status": "unavailable",
                "reason": "Reference trial identities and correlations are unavailable; no generated/human pairing is fabricated.",
            },
            {
                "name": "outer_lineage_bootstrap_for_split_half",
                "status": outer.get("status", "not_computed"),
                "reason": outer.get("reason"),
            },
        ],
    }


def _opted_in(protocol: Mapping[str, Any], key: str) -> bool:
    value = protocol.get(key)
    if isinstance(value, Mapping):
        return bool(value.get("enabled", True))
    return bool(value)


def analyze(episodes: List[Dict[str, Any]], protocol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Summarize a logical-episode ledger as JSON-safe measurement output.

    ``episodes`` must include one logical record per planned episode.  Supplying
    ``planned_per_cell`` (or ``planned_counts``) in ``protocol`` makes absent
    ledger rows explicit missing planned records.  No default silently turns an
    empty export into 50 failures.
    """

    if not isinstance(episodes, list):
        raise TypeError("episodes must be a list of dictionaries")
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise TypeError("episodes must contain dictionaries")
    raw_protocol: Mapping[str, Any] = protocol if isinstance(protocol, Mapping) else {}
    analysis_identity = _validate_analysis_manifest(episodes, raw_protocol)
    ledger_state = _nonterminal_ledger_status(episodes)
    base_policies = _protocol_identifiers(raw_protocol, "policies", PRIMARY_POLICY_IDS, _normalise_policy)
    tasks = _protocol_identifiers(raw_protocol, "tasks", PRIMARY_TASK_IDS, _normalise_task)
    primary_manifest_accepted = analysis_identity["status"] == "accepted"
    policies = list(base_policies)
    arm_metadata: Dict[str, Dict[str, Optional[str]]] = {
        policy: {"base_policy": policy, "policy_variant": None} for policy in base_policies
    }
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    observed_unarmed: Dict[str, bool] = defaultdict(bool)
    observed_arms: Dict[str, set] = defaultdict(set)
    for episode in episodes:
        identity = _episode_identity(episode)
        base_policy = identity["policy"] or "__missing_policy__"
        task = identity["task"] or "__missing_task__"
        arm = _arm_id(
            base_policy,
            identity["policy_variant"],
            None if primary_manifest_accepted else identity["cohort"],
            None if primary_manifest_accepted else identity["protocol_hash"],
            primary_manifest_accepted,
        )
        grouped[(arm, task)].append(episode)
        observed_arms[base_policy].add(arm)
        if arm == base_policy:
            observed_unarmed[base_policy] = True
        if arm not in policies:
            policies.append(arm)
        arm_metadata[arm] = {"base_policy": base_policy, "policy_variant": identity["policy_variant"]}
        if task not in tasks:
            tasks.append(task)
    # Do not emit an empty base-policy row beside a named sensitivity arm when
    # the ledger has no base/default identity for that policy.
    policies = [
        policy
        for policy in policies
        if not (
            policy in observed_arms
            and policy in base_policies
            and not observed_unarmed[policy]
            and any(arm != policy for arm in observed_arms[policy])
        )
    ]
    cells = [
        _cell_from_records(
            policy,
            task,
            grouped.get((policy, task), []),
            raw_protocol,
            reference_policy=arm_metadata.get(policy, {}).get("base_policy"),
            policy_variant=arm_metadata.get(policy, {}).get("policy_variant"),
        )
        for policy in policies
        for task in tasks
    ]
    block_reason = None
    if analysis_identity["status"] == "rejected":
        block_reason = analysis_identity["reason"]
    elif ledger_state["active"]:
        block_reason = "The ledger has nonterminal planned/submitted/running records; only provisional operational counts are available."
    if block_reason is not None:
        redacted_cells = _redact_scientific_cell_fields(cells, block_reason)
        blocked_reliability = {"status": "not_computed", "reason": block_reason, "outer_bootstrap": {"status": "not_computed", "reason": block_reason}}
        blocked_advanced = {"status": "not_computed", "reason": block_reason, "pairwise": []}
        result = {
            "schema_version": 1,
            "endpoint": {
                "name": "observed_positive_lower_bound",
                "definition": "Scientific endpoint withheld until the frozen identity gate passes and all logical rows are terminal.",
            },
            "analysis_identity": analysis_identity,
            "ledger_state": ledger_state,
            "matrix": {"policies": policies, "tasks": tasks},
            "cells": redacted_cells,
            "summary": _operational_summary(redacted_cells, block_reason),
            "rankings": [],
            "pairwise": [],
            "advanced_inference": blocked_advanced,
            "lineage_leakage": validate_source_lineage_leakage(episodes),
            "reliability": blocked_reliability,
            "mmrv": {"status": "not_computed", "reason": block_reason},
            "mdd": {"status": "not_computed", "reason": block_reason},
            "unimplemented_advanced_inference": _advanced_inference_status(blocked_advanced, blocked_reliability),
        }
        _assert_json_finite(result)
        return result
    for cell in cells:
        cell["scientific_status"] = "primary_manifest_bound" if primary_manifest_accepted else "diagnostic"
        if not primary_manifest_accepted:
            cell["scientific_reason"] = analysis_identity["reason"]
    has_disaggregated_arms = any(policy != arm_metadata.get(policy, {}).get("base_policy") for policy in policies)
    indexed = _cell_index(cells)
    if has_disaggregated_arms:
        pairwise = []
        advanced_inference = {
            "status": "unavailable",
            "reason": "Diagnostic policy variants/cohorts/protocol identities are displayed as separate arms and are not pooled into pairwise inference.",
            "pairwise": [],
        }
    else:
        pairwise = [
            _task_pairwise(episodes, indexed, policy_a, policy_b, task)
            for task in tasks
            for policy_a, policy_b in combinations(policies, 2)
        ]
        pairwise.extend(
            _macro_pairwise(indexed, policy_a, policy_b, tasks) for policy_a, policy_b in combinations(policies, 2)
        )
        advanced_config = (
            raw_protocol.get("advanced_inference")
            if isinstance(raw_protocol.get("advanced_inference"), Mapping)
            else {"enabled": True}
            if raw_protocol.get("advanced_inference") is True
            else {}
        )
        if bool(advanced_config.get("enabled")):
            advanced_inference = _advanced_pairwise_inference(episodes, policies, tasks, advanced_config)
            pairwise = _merge_advanced_pairwise(pairwise, advanced_inference)
        else:
            advanced_inference = {
                "status": "not_computed",
                "reason": "Set protocol.advanced_inference.enabled=true to run the expensive lineage-bootstrap ordering analysis.",
                "pairwise": [],
            }
    reliability: Dict[str, Any]
    if has_disaggregated_arms:
        reliability = {
            "status": "unavailable",
            "reason": "Reliability requires one frozen identity per policy; diagnostic arms remain separate.",
            "outer_bootstrap": {
                "status": "not_computed",
                "reason": "Reliability is unavailable for disaggregated diagnostic arms.",
            },
        }
    elif _opted_in(raw_protocol, "compute_reliability") or _opted_in(raw_protocol, "reliability"):
        config = raw_protocol.get("reliability") if isinstance(raw_protocol.get("reliability"), Mapping) else {}
        reliability = split_half_reliability(
            episodes,
            policies=policies,
            tasks=tasks,
            repetitions=int(config.get("repetitions", 1000)),
            seed=int(config.get("seed", 20260918)),
        )
        outer_config = config.get("outer_bootstrap") if isinstance(config.get("outer_bootstrap"), Mapping) else {}
        if bool(outer_config.get("enabled")):
            reliability["outer_bootstrap"] = outer_lineage_bootstrap_reliability(
                episodes,
                policies=policies,
                tasks=tasks,
                outer_replicates=int(outer_config.get("outer_replicates", 1000)),
                inner_repetitions=int(outer_config.get("inner_repetitions", config.get("repetitions", 1000))),
                seed=int(outer_config.get("seed", config.get("seed", 20260918))),
                confidence=float(outer_config.get("confidence", 0.95)),
            )
        else:
            reliability["outer_bootstrap"] = {
                "status": "not_computed",
                "reason": "Set protocol.reliability.outer_bootstrap.enabled=true for outer lineage-bootstrap uncertainty.",
            }
    else:
        reliability = {
            "status": "not_computed",
            "reason": "Split-half reliability is opt-in because 1,000 lineage-safe splits are not dashboard-poll work.",
            "outer_bootstrap": {
                "status": "not_computed",
                "reason": "Outer uncertainty requires protocol.reliability.enabled and protocol.reliability.outer_bootstrap.enabled.",
            },
        }
    # The primary MMRV is a small six-policy calculation and remains useful in
    # live views.  The four completion scenarios and 720-permutation baselines
    # are intentionally explicit, so polling cannot turn into an analysis job.
    mmrv_config = raw_protocol.get("mmrv") if isinstance(raw_protocol.get("mmrv"), Mapping) else {}
    full_mmrv = _opted_in(raw_protocol, "compute_mmrv") or raw_protocol.get("mmrv") is True
    mmrv_result = (
        {
            "status": "unavailable",
            "reason": "MMRV is not pooled across separately labelled diagnostic policy arms.",
        }
        if has_disaggregated_arms
        else _mmrv_report(
            indexed,
            policies,
            tasks,
            include_completion_sensitivity=full_mmrv or bool(mmrv_config.get("completion_sensitivity")),
            include_random_order_baseline=full_mmrv or bool(mmrv_config.get("random_order_baseline")),
        )
    )
    mdd_config = raw_protocol.get("mdd")
    if has_disaggregated_arms:
        mdd_result = {"status": "not_computed", "reason": "MDD is withheld for disaggregated diagnostic arms."}
    elif isinstance(mdd_config, Mapping) and mdd_config.get("enabled", True):
        baseline = mdd_config.get("baseline_rates")
        if baseline is None:
            mdd_result: Dict[str, Any] = {
                "status": "unavailable",
                "reason": "Set mdd.baseline_rates explicitly; MDD is not inferred from a dashboard poll.",
            }
        else:
            options = {
                key: mdd_config[key]
                for key in ("gap_grid", "n_per_task", "alpha", "simulations", "seed")
                if key in mdd_config
            }
            mdd_result = paired_mdd_power_curve(baseline, **options)
    else:
        mdd_result = {
            "status": "not_computed",
            "reason": "MDD power simulation is opt-in and intentionally not run on every analysis poll.",
        }
    result = {
        "schema_version": 1,
        "endpoint": {
            "name": "observed_positive_lower_bound",
            "definition": "Y = 1{completed, validity=valid, evaluable binary success}; unevaluable planned episodes contribute Y=0 only for this labelled lower-bound endpoint.",
        },
        "analysis_identity": analysis_identity,
        "ledger_state": ledger_state,
        "matrix": {"policies": policies, "tasks": tasks},
        "cells": cells,
        "summary": _summary(cells, policies, tasks),
        "rankings": _rankings(indexed, policies, tasks),
        "pairwise": pairwise,
        "advanced_inference": advanced_inference,
        "lineage_leakage": validate_source_lineage_leakage(episodes),
        "reliability": reliability,
        "mmrv": mmrv_result,
        "mdd": mdd_result,
        "unimplemented_advanced_inference": _advanced_inference_status(advanced_inference, reliability),
    }
    # Fail locally if a future change introduces a NaN/inf that JSON would
    # serialize non-portably.  The returned object otherwise contains only
    # stdlib JSON types.
    _assert_json_finite(result)
    return result


def _assert_json_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("analysis produced a non-finite JSON number")
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_json_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_json_finite(child)


__all__ = [
    "analyze",
    "wilson_interval",
    "exact_binomial_interval",
    "exact_mcnemar_pvalue",
    "mean_maximum_rank_violation",
    "mmrv",
    "mmrv_random_order_baseline",
    "validate_source_lineage_leakage",
    "split_half_reliability",
    "outer_lineage_bootstrap_reliability",
    "task_stratified_lineage_bundle_bootstrap",
    "bonferroni_endpoint_envelopes",
    "paired_macro_sign_test",
    "paired_mdd_power_curve",
    "reference_payload",
]
