"""Frozen full-study planning and evidence checks for PLUMB.

This module is deliberately an *offline control-plane* contract.  It creates
the exact 6 x 5 x 50 primary plan from supplied scenario provenance, and it
checks records produced elsewhere.  It never generates a scenario, a label, a
cost, a gate pass, or a rollout.  In particular, a locally constructed plan is
not a preregistration and a structurally valid rehearsal record is not proof
that a platform executed it.

The module has four public entry points useful to CLI/API callers:

* :func:`build_full_study_plan` freezes the matrix and deterministic streams.
* :func:`assess_primary_readiness` checks A--D evidence for execution.
* :func:`build_cost_fidelity_design` / :func:`build_drift_design` create
  fixed-horizon comparison schedules without selecting a winner.
* :func:`validate_burst_rehearsals` and :func:`assess_burst_target_claim`
  check three fresh full-matrix rehearsal records before a speed/cost target
  can be displayed as achieved.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .adapters.contracts import FeedbackMode
from .gates import (
    GateLedger,
    GateRecord,
    GateStatus,
    QualificationValidator,
    ScenarioManifestValidator,
    canonical_json_sha256,
)
from .measurement import validate_source_lineage_leakage
from .policies.tasks import BENCHMARK_TASK_REGISTRY
from .records import DEFAULT_POLICIES, DEFAULT_TASKS, canonical_json, deterministic_seed, json_digest
from .scenarios import ScenarioError, validate_scenario_manifest


STUDY_SCHEMA_VERSION = 1
PRIMARY_COHORT = "primary"
DEVELOPMENT_COHORT = "development"
CALIBRATION_DEVELOPMENT_COHORT = "calibration_development"
CALIBRATION_HELDOUT_COHORT = "calibration_heldout"
COST_DEVELOPMENT_COHORT = "cost_development"
COST_CONFIRMATION_COHORT = "cost_confirmation"
TRANSPORT_SENSITIVITY_COHORT = "transport_sensitivity"
DRIFT_HELDOUT_COHORT = "drift_heldout"
BURST_REHEARSAL_COHORT = "burst_rehearsal"

PRIMARY_POLICIES: Tuple[str, ...] = tuple(DEFAULT_POLICIES)
PRIMARY_TASKS: Tuple[str, ...] = tuple(DEFAULT_TASKS)
PRIMARY_STARTS_PER_TASK = 50
PRIMARY_EPISODE_COUNT = len(PRIMARY_POLICIES) * len(PRIMARY_TASKS) * PRIMARY_STARTS_PER_TASK
BURST_REHEARSAL_COUNT = 3
BURST_TARGET_SECONDS = 60.0
BURST_TARGET_USD = 11.25
PRIMARY_TASK_HORIZONS = {
    task.task_id: task.max_steps for task in BENCHMARK_TASK_REGISTRY.tasks if task.task_id in PRIMARY_TASKS
}


class StudyValidationError(ValueError):
    """Raised for malformed plans or comparison inputs, before any work starts."""


def _json_mapping(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Make a detached JSON-compatible mapping or raise a useful error."""

    if not isinstance(value, Mapping):
        raise StudyValidationError("value must be an object")
    try:
        result = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as error:
        raise StudyValidationError("value must be JSON serializable") from error
    if not isinstance(result, dict):  # defensive: canonical_json preserves mappings.
        raise StudyValidationError("value must serialize to an object")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudyValidationError("%s must be a non-empty string" % field)
    return value.strip()


def _int(value: Any, field: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StudyValidationError("%s must be an integer" % field)
    if minimum is not None and value < minimum:
        raise StudyValidationError("%s must be at least %d" % (field, minimum))
    return value


def _finite_number(value: Any, field: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise StudyValidationError("%s must be a finite number" % field)
    number = float(value)
    if minimum is not None and number < minimum:
        raise StudyValidationError("%s must be at least %s" % (field, minimum))
    return number


def _sha256(value: Any, field: str) -> str:
    text = _text(value, field)
    if not text.startswith("sha256:") or len(text) != len("sha256:") + 64:
        raise StudyValidationError("%s must be a sha256: digest" % field)
    digest = text[len("sha256:") :]
    if any(character not in "0123456789abcdef" for character in digest.lower()):
        raise StudyValidationError("%s must be a hexadecimal SHA-256 digest" % field)
    return text.lower()


def _contains_synthetic(value: Any) -> bool:
    """Reject obvious synthetic/mock evidence from a qualification route."""

    # Gate A/B require real vendor/Bridge fixtures, so "fixture" alone is not
    # synthetic evidence. Reject only explicit simulated/mock labels.
    forbidden = {"synthetic", "simulated", "mock", "fake"}
    if isinstance(value, str):
        lowered = value.lower()
        return any(word in lowered for word in forbidden)
    if isinstance(value, Mapping):
        return any(_contains_synthetic(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_synthetic(item) for item in value)
    return False


def _parse_timestamp(value: Any, field: str) -> datetime:
    text = _text(value, field)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise StudyValidationError("%s must be an ISO-8601 timestamp" % field) from error
    if parsed.tzinfo is None:
        raise StudyValidationError("%s must include a UTC offset" % field)
    return parsed.astimezone(timezone.utc)


def _as_utc_text(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _normalise_feedback_mode(value: Union[FeedbackMode, str, None]) -> FeedbackMode:
    if value is None:
        return FeedbackMode.UNQUALIFIED
    if isinstance(value, FeedbackMode):
        return value
    try:
        return FeedbackMode(str(value))
    except ValueError as error:
        raise StudyValidationError("feedback_mode must be a declared FeedbackMode") from error


def _require_policy_revision_bindings(value: Any) -> Dict[str, Any]:
    """Require each named primary wrapper to have an immutable identity.

    ``GateRecord.status='pass'`` is intentionally insufficient.  The same
    identity object must be copied into Gate B evidence, so an adapter or asset
    change cannot inherit an old pass by policy name alone.
    """

    if not isinstance(value, Mapping) or set(value) != set(PRIMARY_POLICIES):
        raise StudyValidationError("policy_revisions must contain exactly the six canonical primary policy IDs")
    bindings: Dict[str, Any] = {}
    for policy in PRIMARY_POLICIES:
        item = value[policy]
        if not isinstance(item, Mapping):
            raise StudyValidationError("policy_revisions.%s must be an object" % policy)
        asset = _text(item.get("asset_manifest_id"), "policy_revisions.%s.asset_manifest_id" % policy)
        adapter = _sha256(item.get("adapter_hash"), "policy_revisions.%s.adapter_hash" % policy)
        normalized = _json_mapping(item)
        normalized["asset_manifest_id"] = asset
        normalized["adapter_hash"] = adapter
        if policy == "Octo" and normalized.get("display_name") != "Octo-Small v1.0":
            raise StudyValidationError("policy_revisions.Octo.display_name must bind the Octo alias to Octo-Small v1.0")
        bindings[policy] = normalized
    return bindings


def _require_world_revision_binding(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyValidationError("world_model_revision must be an object")
    binding = _json_mapping(value)
    _text(binding.get("asset_manifest_id"), "world_model_revision.asset_manifest_id")
    _sha256(binding.get("backend_profile_hash"), "world_model_revision.backend_profile_hash")
    _text(binding.get("model_revision"), "world_model_revision.model_revision")
    _text(binding.get("code_revision"), "world_model_revision.code_revision")
    _text(binding.get("container_digest"), "world_model_revision.container_digest")
    return binding


def _require_judge_revision_binding(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyValidationError("judge_revision must be an object")
    binding = _json_mapping(value)
    _text(binding.get("asset_manifest_id"), "judge_revision.asset_manifest_id")
    _text(binding.get("model_revision"), "judge_revision.model_revision")
    _text(binding.get("processor_revision"), "judge_revision.processor_revision")
    _sha256(binding.get("runtime_lock_hash"), "judge_revision.runtime_lock_hash")
    _sha256(binding.get("rubric_hash"), "judge_revision.rubric_hash")
    _sha256(binding.get("sampling_hash"), "judge_revision.sampling_hash")
    return binding


def _start_rows_from_manifest(manifest: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    rows: List[Dict[str, Any]] = []
    tasks = manifest.get("tasks")
    if isinstance(tasks, Mapping):
        for task in PRIMARY_TASKS:
            entry = tasks.get(task)
            if isinstance(entry, Mapping) and isinstance(entry.get("starts"), list):
                for start in entry["starts"]:
                    if isinstance(start, Mapping):
                        item = dict(start)
                        item.setdefault("task", task)
                        rows.append(item)
        return tuple(rows)
    starts = manifest.get("starts")
    if isinstance(starts, list):
        return tuple(dict(start) for start in starts if isinstance(start, Mapping))
    return ()


def _canonical_primary_start_rows(scenario_manifest: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    errors = ScenarioManifestValidator.validate(scenario_manifest, require_primary_panel=True)
    if errors:
        raise StudyValidationError("primary scenario manifest is invalid: " + "; ".join(errors))
    rows = _start_rows_from_manifest(scenario_manifest)
    by_task: Dict[str, List[Dict[str, Any]]] = {task: [] for task in PRIMARY_TASKS}
    for row in rows:
        task = row.get("task")
        if task in by_task:
            by_task[task].append(row)
    ordered: List[Dict[str, Any]] = []
    for task in PRIMARY_TASKS:
        task_rows = by_task[task]
        if len(task_rows) != PRIMARY_STARTS_PER_TASK:
            raise StudyValidationError("task %s needs exactly %d primary starts" % (task, PRIMARY_STARTS_PER_TASK))
        ordered.extend(sorted(task_rows, key=lambda row: str(row["start_id"])))
    return tuple(ordered)


def _seed(base_seed: int, stream: str, policy: str, task: str, start_id: str, index: Optional[int] = None) -> int:
    """Use the established deterministic stream but domain-separate every role."""

    suffix = stream if index is None else "%s:%d" % (stream, index)
    return deterministic_seed(base_seed, "%s|%s" % (policy, suffix), task, start_id)


@dataclass(frozen=True)
class StudyEpisode:
    """One immutable logical episode in the primary 1,500-row matrix."""

    episode_id: str
    policy: str
    task: str
    start_id: str
    start_lineage_id: str
    policy_seed: int
    world_seed: int
    judge_sample_seeds: Tuple[int, ...]
    horizon_actions: int

    def to_mapping(self, protocol_hash: str) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "cohort": PRIMARY_COHORT,
            "protocol_hash": protocol_hash,
            "policy": self.policy,
            "task": self.task,
            "start_id": self.start_id,
            "start_lineage_id": self.start_lineage_id,
            "horizon_actions": self.horizon_actions,
            "seeds": {
                "policy": self.policy_seed,
                "world": self.world_seed,
                "judge_samples": list(self.judge_sample_seeds),
            },
        }


@dataclass(frozen=True)
class FrozenStudyPlan:
    """JSON-detached immutable primary protocol and exact episode schedule."""

    protocol_hash: str
    plan_hash: str
    scenario_manifest_hash: str
    protocol_json: str
    episodes: Tuple[StudyEpisode, ...]

    @property
    def protocol(self) -> Dict[str, Any]:
        return json.loads(self.protocol_json)

    def episode_rows(self) -> List[Dict[str, Any]]:
        return [episode.to_mapping(self.protocol_hash) for episode in self.episodes]

    def analysis_manifest(self) -> Dict[str, Any]:
        protocol = self.protocol
        return dict(protocol["analysis_manifest"])

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema_version": STUDY_SCHEMA_VERSION,
            "plan_hash": self.plan_hash,
            "protocol_hash": self.protocol_hash,
            "scenario_manifest_hash": self.scenario_manifest_hash,
            "protocol": self.protocol,
            "episodes": self.episode_rows(),
        }


def study_plan_from_mapping(payload: Mapping[str, Any]) -> FrozenStudyPlan:
    """Re-open a serialized plan only when all local immutable bindings match.

    This is a JSON integrity check, not proof of a remote preregistration.  It
    lets a CLI validate a plan emitted on an earlier machine without accepting
    a changed episode schedule or quietly re-deriving seeds.
    """

    if not isinstance(payload, Mapping):
        raise StudyValidationError("study plan must be an object")
    protocol = _json_mapping(payload.get("protocol"))
    protocol_hash = _sha256(payload.get("protocol_hash", protocol.get("sha256")), "protocol_hash")
    if protocol.get("sha256") != protocol_hash:
        raise StudyValidationError("plan protocol.sha256 does not match protocol_hash")
    core = dict(protocol)
    core.pop("sha256", None)
    analysis_manifest = core.pop("analysis_manifest", None)
    if canonical_json_sha256(core) != protocol_hash:
        raise StudyValidationError("plan protocol content does not match protocol_hash")
    if protocol.get("status") != "frozen" or protocol.get("frozen") is not True:
        raise StudyValidationError("serialized primary plan protocol is not frozen")
    if not isinstance(protocol.get("remote_record"), str) or not protocol["remote_record"].strip():
        raise StudyValidationError("serialized primary plan lacks an externally auditable remote record")
    bindings = protocol.get("revision_bindings")
    if not isinstance(bindings, Mapping):
        raise StudyValidationError("serialized primary plan lacks revision_bindings")
    _require_policy_revision_bindings(bindings.get("policy"))
    _require_world_revision_binding(bindings.get("world_model"))
    _require_judge_revision_binding(bindings.get("judge"))
    if not isinstance(analysis_manifest, Mapping):
        raise StudyValidationError("plan protocol lacks an analysis_manifest")
    identity = analysis_manifest.get("identity")
    if not isinstance(identity, Mapping) or identity.get("cohort") != PRIMARY_COHORT or identity.get("protocol_hash") != protocol_hash:
        raise StudyValidationError("analysis_manifest does not bind the primary cohort/protocol")
    scenario_hash = _sha256(payload.get("scenario_manifest_hash"), "scenario_manifest_hash")
    if protocol.get("scenario_manifest_hash") != scenario_hash:
        raise StudyValidationError("plan scenario_manifest_hash does not match protocol")
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != PRIMARY_EPISODE_COUNT:
        raise StudyValidationError("serialized plan needs exactly %d episodes" % PRIMARY_EPISODE_COUNT)
    episodes: List[StudyEpisode] = []
    normalized_rows: List[Dict[str, Any]] = []
    seen = set()
    base_seed = _int(protocol.get("seeds", {}).get("base_seed") if isinstance(protocol.get("seeds"), Mapping) else None, "protocol.seeds.base_seed")
    cell_counts: Dict[Tuple[str, str], int] = {}
    starts_by_policy_task: Dict[Tuple[str, str], set] = {}
    for index, raw in enumerate(raw_episodes):
        if not isinstance(raw, Mapping):
            raise StudyValidationError("episodes[%d] must be an object" % index)
        episode_id = _text(raw.get("episode_id"), "episodes[%d].episode_id" % index)
        policy = _text(raw.get("policy"), "episodes[%d].policy" % index)
        task = _text(raw.get("task"), "episodes[%d].task" % index)
        start_id = _text(raw.get("start_id"), "episodes[%d].start_id" % index)
        lineage = _text(raw.get("start_lineage_id"), "episodes[%d].start_lineage_id" % index)
        if policy not in PRIMARY_POLICIES or task not in PRIMARY_TASKS:
            raise StudyValidationError("episodes[%d] has an unknown primary policy or task" % index)
        if raw.get("cohort") != PRIMARY_COHORT or raw.get("protocol_hash") != protocol_hash:
            raise StudyValidationError("episodes[%d] does not bind primary/protocol identity" % index)
        seeds = raw.get("seeds")
        if not isinstance(seeds, Mapping):
            raise StudyValidationError("episodes[%d].seeds must be an object" % index)
        policy_seed = _int(seeds.get("policy"), "episodes[%d].seeds.policy" % index)
        world_seed = _int(seeds.get("world"), "episodes[%d].seeds.world" % index)
        judge_seeds = seeds.get("judge_samples")
        if not isinstance(judge_seeds, list) or len(judge_seeds) != 5:
            raise StudyValidationError("episodes[%d] needs exactly five judge sample seeds" % index)
        normalized_judge = tuple(_int(value, "episodes[%d].judge seed" % index) for value in judge_seeds)
        horizon = _int(raw.get("horizon_actions"), "episodes[%d].horizon_actions" % index, minimum=1)
        if horizon != PRIMARY_TASK_HORIZONS[task]:
            raise StudyValidationError("episodes[%d] changes the frozen task horizon" % index)
        if policy_seed != _seed(base_seed, "policy", policy, task, start_id):
            raise StudyValidationError("episodes[%d] policy seed does not match deterministic frozen derivation" % index)
        if world_seed != _seed(base_seed, "world", "matched-start", task, start_id):
            raise StudyValidationError("episodes[%d] world seed does not match deterministic matched-start derivation" % index)
        if normalized_judge != tuple(_seed(base_seed, "judge", policy, task, start_id, item) for item in range(5)):
            raise StudyValidationError("episodes[%d] judge seeds do not match deterministic frozen derivation" % index)
        identity_key = (policy, task, start_id)
        if identity_key in seen:
            raise StudyValidationError("serialized plan duplicates a policy/task/start identity")
        seen.add(identity_key)
        cell_key = (policy, task)
        cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1
        starts_by_policy_task.setdefault(cell_key, set()).add(start_id)
        episode = StudyEpisode(episode_id, policy, task, start_id, lineage, policy_seed, world_seed, normalized_judge, horizon)
        episodes.append(episode)
        normalized_rows.append(episode.to_mapping(protocol_hash))
    if set(cell_counts) != {(policy, task) for policy in PRIMARY_POLICIES for task in PRIMARY_TASKS} or any(
        count != PRIMARY_STARTS_PER_TASK for count in cell_counts.values()
    ):
        raise StudyValidationError("serialized plan must retain exactly 50 starts in every six-policy/five-task cell")
    for task in PRIMARY_TASKS:
        paired = [starts_by_policy_task[(policy, task)] for policy in PRIMARY_POLICIES]
        if any(starts != paired[0] for starts in paired[1:]):
            raise StudyValidationError("serialized plan does not retain matched start IDs across policies for task %s" % task)
    allowed = analysis_manifest.get("allowed_episode_ids")
    if not isinstance(allowed, list) or set(allowed) != {episode.episode_id for episode in episodes}:
        raise StudyValidationError("analysis_manifest allowed episode IDs do not match serialized plan")
    expected_plan_hash = json_digest(
        {
            "schema_version": STUDY_SCHEMA_VERSION,
            "protocol_hash": protocol_hash,
            "scenario_manifest_hash": scenario_hash,
            "episodes": normalized_rows,
        }
    )
    plan_hash = _sha256(payload.get("plan_hash"), "plan_hash")
    if plan_hash != expected_plan_hash:
        raise StudyValidationError("serialized plan content does not match plan_hash")
    return FrozenStudyPlan(
        protocol_hash=protocol_hash,
        plan_hash=plan_hash,
        scenario_manifest_hash=scenario_hash,
        protocol_json=canonical_json(protocol),
        episodes=tuple(episodes),
    )


def build_full_study_plan(
    scenario_manifest: Mapping[str, Any],
    protocol_metadata: Mapping[str, Any],
    *,
    base_seed: int,
    feedback_mode: Union[FeedbackMode, str, None] = None,
) -> FrozenStudyPlan:
    """Freeze the full primary matrix from an externally supplied scenario panel.

    ``scenario_manifest`` must already describe all 250 real candidate starts;
    this routine neither supplies starts nor accepts a shorter surrogate.  The
    returned plan is immutable in memory, but only a metadata record marked
    ``status='frozen'`` *and* carrying an external preregistration URI can
    satisfy the existing qualification validator.
    """

    _int(base_seed, "base_seed")
    scenario = _json_mapping(scenario_manifest)
    try:
        # The scenario module verifies its own content hash, explicit panel
        # structure, real-robot provenance, and the canonical Octo alias.
        validate_scenario_manifest(scenario, require_full_primary=True)
    except ScenarioError as error:
        raise StudyValidationError("frozen scenario manifest is invalid: " + str(error)) from error
    if scenario.get("status") != "frozen" or scenario.get("frozen") is not True:
        raise StudyValidationError("a full primary plan requires a frozen scenario manifest")
    starts = _canonical_primary_start_rows(scenario)
    scenario_hash = _sha256(scenario.get("sha256"), "scenario_manifest.sha256")
    metadata = _json_mapping(protocol_metadata)
    protocol_id = _text(metadata.get("protocol_id"), "protocol_metadata.protocol_id")
    for field in ("policy", "world_model", "judge", "thresholds"):
        if not isinstance(metadata.get(field), Mapping) or not metadata[field]:
            raise StudyValidationError("protocol_metadata.%s must be a non-empty object" % field)
    status = str(metadata.get("status", "draft")).strip().lower()
    if status != "frozen":
        raise StudyValidationError("a full primary plan requires protocol_metadata.status='frozen'")
    remote_record = metadata.get("remote_record", metadata.get("preregistration_uri"))
    if not isinstance(remote_record, str):
        raise StudyValidationError("a frozen protocol needs an externally auditable remote_record or preregistration_uri")
    if _contains_synthetic(metadata):
        raise StudyValidationError("synthetic/mock metadata may not define a full-study protocol")
    policy_revisions = _require_policy_revision_bindings(metadata.get("policy_revisions"))
    world_revision = _require_world_revision_binding(metadata.get("world_model_revision"))
    judge_revision = _require_judge_revision_binding(metadata.get("judge_revision"))
    minimum_coverage = metadata["thresholds"].get("minimum_scientific_coverage")
    _finite_number(minimum_coverage, "thresholds.minimum_scientific_coverage", minimum=0.0)
    if float(minimum_coverage) > 1.0:
        raise StudyValidationError("thresholds.minimum_scientific_coverage must not exceed 1")

    selected_feedback = _normalise_feedback_mode(feedback_mode if feedback_mode is not None else metadata.get("feedback_mode"))
    starts_by_task: Dict[str, Tuple[Dict[str, Any], ...]] = {
        task: tuple(row for row in starts if row["task"] == task) for task in PRIMARY_TASKS
    }
    episodes: List[StudyEpisode] = []
    for policy in PRIMARY_POLICIES:
        for task in PRIMARY_TASKS:
            for start in starts_by_task[task]:
                start_id = _text(start.get("start_id"), "start_id")
                lineage = _text(start.get("start_lineage_id"), "start_lineage_id")
                episode_id = "primary-" + json_digest(
                    {"policy": policy, "task": task, "start_id": start_id}
                ).split(":", 1)[1][:24]
                episodes.append(
                    StudyEpisode(
                        episode_id=episode_id,
                        policy=policy,
                        task=task,
                        start_id=start_id,
                        start_lineage_id=lineage,
                        policy_seed=_seed(base_seed, "policy", policy, task, start_id),
                        # World stochasticity is deliberately matched across
                        # policies for one task/start.  Policy and judge RNGs
                        # remain policy-specific streams.
                        world_seed=_seed(base_seed, "world", "matched-start", task, start_id),
                        judge_sample_seeds=tuple(
                            _seed(base_seed, "judge", policy, task, start_id, index) for index in range(5)
                        ),
                        horizon_actions=PRIMARY_TASK_HORIZONS[task],
                    )
                )
    if len(episodes) != PRIMARY_EPISODE_COUNT:
        raise StudyValidationError("full plan must contain exactly %d logical episodes" % PRIMARY_EPISODE_COUNT)
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise StudyValidationError("primary episode IDs unexpectedly collide")

    protocol_core: Dict[str, Any] = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "status": status,
        "frozen": status == "frozen",
        "scenario_manifest_hash": scenario_hash,
        "policy": metadata["policy"],
        "world_model": metadata["world_model"],
        "judge": metadata["judge"],
        "revision_bindings": {
            "policy": policy_revisions,
            "world_model": world_revision,
            "judge": judge_revision,
        },
        "thresholds": metadata["thresholds"],
        "feedback_mode": selected_feedback.value,
        "matrix": {
            "policies": list(PRIMARY_POLICIES),
            "tasks": list(PRIMARY_TASKS),
            "starts_per_task": PRIMARY_STARTS_PER_TASK,
            "planned_episodes": PRIMARY_EPISODE_COUNT,
            "total_planned_episodes": PRIMARY_EPISODE_COUNT,
        },
        "task_horizons": dict(PRIMARY_TASK_HORIZONS),
        "seeds": {
            "base_seed": base_seed,
            "derivation": "records.deterministic_seed with a domain-separated policy/world/judge stream",
            "world_seed_pairing": "one shared world seed per task/start across all six policies",
            "judge_sample_count": 5,
        },
    }
    if remote_record is not None:
        protocol_core["remote_record"] = _text(remote_record, "protocol_metadata.remote_record")
    # Preserve declared non-identity protocol metadata, but never an untrusted
    # caller-supplied hash or a caller-supplied matrix/seeds replacement.
    for field in (
        "asset_manifest_hash",
        "asset_lock_hash",
        "qualification_assets",
        "tolerances",
        "judge_sampling",
        "notes",
        "revisions",
        "operating_point_id",
    ):
        if field in metadata:
            protocol_core[field] = metadata[field]
    protocol_hash = canonical_json_sha256(protocol_core)
    episode_rows = [episode.to_mapping(protocol_hash) for episode in episodes]
    analysis_manifest = {
        "manifest_id": "analysis-" + protocol_hash.split(":", 1)[1][:16],
        "allowed_episode_ids": [episode.episode_id for episode in episodes],
        "identity": {"cohort": PRIMARY_COHORT, "protocol_hash": protocol_hash},
        "policy_variants": {policy: "native-wrapper" for policy in PRIMARY_POLICIES},
        "episode_identities": {
            episode["episode_id"]: {
                "policy": episode["policy"],
                "task": episode["task"],
                "start_id": episode["start_id"],
                "start_lineage_id": episode["start_lineage_id"],
                "cohort": PRIMARY_COHORT,
                "protocol_hash": protocol_hash,
            }
            for episode in episode_rows
        },
    }
    protocol = dict(protocol_core)
    protocol["sha256"] = protocol_hash
    protocol["analysis_manifest"] = analysis_manifest
    plan_hash = json_digest(
        {
            "schema_version": STUDY_SCHEMA_VERSION,
            "protocol_hash": protocol_hash,
            "scenario_manifest_hash": scenario_hash,
            "episodes": episode_rows,
        }
    )
    return FrozenStudyPlan(
        protocol_hash=protocol_hash,
        plan_hash=plan_hash,
        scenario_manifest_hash=scenario_hash,
        protocol_json=canonical_json(protocol),
        episodes=tuple(episodes),
    )


def materialize_study_run(
    plan: FrozenStudyPlan, run_id: str, *, cohort: str = PRIMARY_COHORT
) -> List[Dict[str, Any]]:
    """Give stable plan slots run-unique ledger episode IDs.

    ``StudyEpisode.episode_id`` is a stable matching slot, deliberately shared
    by plan revisions with identical policy/task/start identities.  SQLite
    ledger episode IDs are global, however, so a fresh primary or rehearsal
    run must use a run-namespaced ID.  The original slot is retained in
    ``plan_slot_id`` for paired analysis and rehearsal validation.
    """

    selected_run = _text(run_id, "run_id")
    selected_cohort = _text(cohort, "cohort")
    if selected_cohort not in {PRIMARY_COHORT, BURST_REHEARSAL_COHORT}:
        raise StudyValidationError("materialized full-study runs must be primary or burst_rehearsal cohorts")
    rows: List[Dict[str, Any]] = []
    # ``FrozenStudyPlan.protocol`` intentionally returns a detached JSON
    # object. Parse it once for the full matrix, never once per ledger row.
    bindings = plan.protocol["revision_bindings"]
    for episode in plan.episodes:
        row = episode.to_mapping(plan.protocol_hash)
        slot_id = row["episode_id"]
        row["episode_id"] = "run-" + json_digest(
            {"plan_hash": plan.plan_hash, "run_id": selected_run, "plan_slot_id": slot_id}
        ).split(":", 1)[1][:24]
        row["plan_slot_id"] = slot_id
        row["run_id"] = selected_run
        row["cohort"] = selected_cohort
        row["plan_hash"] = plan.plan_hash
        row["policy_revision"] = bindings["policy"][episode.policy]
        row["world_model_revision"] = bindings["world_model"]
        row["judge_revision"] = bindings["judge"]
        row["status"] = "planned"
        rows.append(row)
    return rows


def validate_cohort_lineage_separation(
    cohorts: Mapping[str, Iterable[Mapping[str, Any]]]
) -> Dict[str, Any]:
    """Check source-lineage separation while allowing policy pairing in a cohort."""

    if not isinstance(cohorts, Mapping) or not cohorts:
        raise StudyValidationError("cohorts must be a non-empty mapping of cohort names to rows")
    flattened: List[Dict[str, Any]] = []
    malformed: List[str] = []
    for cohort, rows in cohorts.items():
        cohort_name = _text(cohort, "cohort name")
        if isinstance(rows, (str, bytes)):
            raise StudyValidationError("cohort %s rows must be iterable mappings" % cohort_name)
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                malformed.append("%s[%d] is not an object" % (cohort_name, index))
                continue
            lineage = row.get("start_lineage_id", row.get("source_lineage_id"))
            flattened.append({"cohort": cohort_name, "start_lineage_id": lineage})
    if malformed:
        return {
            "status": "unverifiable",
            "reason": "Malformed cohort rows: " + "; ".join(malformed),
            "violations": [],
            "missing_lineage_record_indices": [],
        }
    report = validate_source_lineage_leakage(flattened)
    report["cohorts"] = sorted(str(name) for name in cohorts)
    return report


def _gate_d_report_passed(record: GateRecord) -> bool:
    measurements = record.measurements
    candidates = (
        measurements.get("calibration_report"),
        measurements.get("gate_d"),
        measurements.get("judge_calibration"),
    )
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        gate_d = candidate.get("gate_d", candidate)
        if isinstance(gate_d, Mapping) and gate_d.get("passed") is True and gate_d.get("status") == "pass":
            return True
    return False


def _gate_actual_evidence_errors(record: GateRecord, gate_id: str, plan: FrozenStudyPlan) -> List[str]:
    errors = list(record.pass_evidence_errors())
    if record.status is not GateStatus.PASS:
        errors.append("Gate %s is %s, not pass" % (gate_id, record.status.value))
        return errors
    if record.protocol_hash != plan.protocol_hash:
        errors.append("Gate %s protocol hash does not match this plan" % gate_id)
    if _contains_synthetic(record.evidence_kind) or _contains_synthetic(record.measurements):
        errors.append("Gate %s has synthetic/mock evidence" % gate_id)
    bindings = plan.protocol.get("revision_bindings", {})
    expected_policy = bindings.get("policy") if isinstance(bindings, Mapping) else None
    expected_world = bindings.get("world_model") if isinstance(bindings, Mapping) else None
    expected_judge = bindings.get("judge") if isinstance(bindings, Mapping) else None
    if gate_id == "A":
        if not record.backend_revisions:
            errors.append("Gate A lacks a pinned backend revision")
        elif record.backend_revisions != expected_world:
            errors.append("Gate A backend revision does not exactly bind the frozen world-model revision")
    if gate_id == "B":
        missing = sorted(set(PRIMARY_POLICIES) - set(record.policy_revisions))
        if missing:
            errors.append("Gate B lacks policy revisions for %s" % ", ".join(missing))
        elif record.policy_revisions != expected_policy:
            errors.append("Gate B policy revisions do not exactly bind the frozen six-wrapper identities")
    if gate_id == "C":
        expected = {(episode.task, episode.start_id) for episode in plan.episodes}
        observed = set()
        for start_id in record.start_ids:
            if "/" in start_id:
                task, identifier = start_id.split("/", 1)
                observed.add((task, identifier))
        if expected - observed:
            errors.append("Gate C evidence does not enumerate all 250 task/start identities")
    if gate_id == "D":
        if not record.judge_revisions:
            errors.append("Gate D lacks a pinned judge revision")
        elif record.judge_revisions != expected_judge:
            errors.append("Gate D judge revision does not exactly bind the frozen judge identity")
        if not _gate_d_report_passed(record):
            errors.append("Gate D lacks a passed provenance-bound held-out calibration report")
    return errors


def _coerce_gate_ledger(value: Union[GateLedger, Mapping[str, Any]]) -> GateLedger:
    if isinstance(value, GateLedger):
        return value
    if isinstance(value, Mapping):
        return GateLedger.from_mapping(value)
    raise StudyValidationError("gate_ledger must be a GateLedger or gates.json object")


def assess_primary_readiness(
    plan: FrozenStudyPlan,
    scenario_manifest: Mapping[str, Any],
    gate_ledger: Union[GateLedger, Mapping[str, Any]],
    *,
    cohort_rows: Optional[Mapping[str, Iterable[Mapping[str, Any]]]] = None,
    asset_lock: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a fail-closed primary-execution readiness decision.

    Gate E is intentionally not required here: it depends on the completed
    primary matrix.  A ready result only authorizes the *planned primary
    execution route*; it says nothing about reliability, cheap-setting, or the
    public burst target.
    """

    scenario = _json_mapping(scenario_manifest)
    ledger = _coerce_gate_ledger(gate_ledger)
    errors: List[str] = []
    try:
        validate_scenario_manifest(scenario, require_full_primary=True)
    except ScenarioError as error:
        errors.append("scenario manifest integrity: " + str(error))
    if scenario.get("status") != "frozen" or scenario.get("frozen") is not True:
        errors.append("primary readiness requires a frozen scenario manifest")
    if scenario.get("sha256") != plan.scenario_manifest_hash:
        errors.append("scenario manifest hash does not match frozen plan")
    if _contains_synthetic(scenario.get("provenance_kind")):
        errors.append("synthetic scenario provenance cannot qualify the primary study")
    primary_rows = plan.episode_rows()
    if cohort_rows is not None:
        supplied = {str(name): list(rows) for name, rows in cohort_rows.items()}
        supplied.setdefault(PRIMARY_COHORT, primary_rows)
        lineage = validate_cohort_lineage_separation(supplied)
        if lineage["status"] != "pass":
            errors.append("cohort source-lineage separation is %s" % lineage["status"])
    else:
        lineage = validate_cohort_lineage_separation({PRIMARY_COHORT: primary_rows})
    for gate_id in ("A", "B", "C", "D"):
        errors.extend(_gate_actual_evidence_errors(ledger.records[gate_id], gate_id, plan))
    feedback = _normalise_feedback_mode(plan.protocol.get("feedback_mode"))
    # ``analysis_manifest`` is derived after the protocol core receives its
    # self-hash (it embeds that hash in every allowed episode identity).  The
    # qualification boundary validates the frozen core, not that derived view.
    qualification_protocol = dict(plan.protocol)
    qualification_protocol.pop("analysis_manifest", None)
    for task in PRIMARY_TASKS:
        decision = QualificationValidator.validate(
            ledger,
            scenario,
            qualification_protocol,
            task=task,
            feedback_mode=feedback,
            requires_judge=True,
            require_primary_panel=True,
            asset_lock=asset_lock,
        )
        errors.extend("%s: %s" % (task, error) for error in decision.errors)
    deduplicated = tuple(dict.fromkeys(errors))
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "status": "ready" if not deduplicated else "blocked",
        "route": "primary_execution",
        "plan_hash": plan.plan_hash,
        "protocol_hash": plan.protocol_hash,
        "required_gates": ["A", "B", "C", "D"],
        "asset_lock_supplied": asset_lock is not None,
        "lineage_separation": lineage,
        "errors": list(deduplicated),
        "limitations": [
            "Gate E remains required for reliability and selected-cost-setting claims.",
            "Gate F and three fresh rehearsals remain required for the 60-second/$11.25 target claim.",
        ],
    }


def partition_horizon(horizon_actions: int, request_count: int) -> Tuple[int, ...]:
    """Split an exact horizon into deterministic near-equal request lengths."""

    horizon = _int(horizon_actions, "horizon_actions", minimum=1)
    requests = _int(request_count, "request_count", minimum=1)
    if requests > 6:
        raise StudyValidationError("request_count must be between 1 and 6 for the declared partition sweep")
    if requests > horizon:
        raise StudyValidationError("request_count cannot exceed exact action horizon")
    quotient, remainder = divmod(horizon, requests)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(requests))


@dataclass(frozen=True)
class OperatingPoint:
    """One explicit native or unqualified transport candidate.

    Native candidates carry the independently certified policy feedback
    cadence.  A display partition is never allowed to replace that cadence.
    Transport candidates are intentionally labelled unqualified and are the
    only route that may use the 1--6 request display sweep.
    """

    operating_point_id: str
    resolution: Tuple[int, int]
    diffusion_steps: int
    backend_profile_id: str
    judge_profile_id: str
    feedback_mode: FeedbackMode
    comparison_kind: str
    policy_cadences: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    backend_supported_action_lengths: Tuple[int, ...] = ()
    request_count: Optional[int] = None

    def __post_init__(self) -> None:
        _text(self.operating_point_id, "operating_point_id")
        if len(self.resolution) != 2:
            raise StudyValidationError("resolution must contain width and height")
        for axis in self.resolution:
            _int(axis, "resolution axis", minimum=1)
        _int(self.diffusion_steps, "diffusion_steps", minimum=1)
        _text(self.backend_profile_id, "backend_profile_id")
        _text(self.judge_profile_id, "judge_profile_id")
        if not isinstance(self.feedback_mode, FeedbackMode):
            raise StudyValidationError("feedback_mode must be a FeedbackMode")
        if self.comparison_kind not in {"native_cost_fidelity", "transport_sensitivity"}:
            raise StudyValidationError("comparison_kind must be native_cost_fidelity or transport_sensitivity")
        if self.comparison_kind == "native_cost_fidelity":
            if self.feedback_mode is not FeedbackMode.NATIVE_FEEDBACK:
                raise StudyValidationError("native cost/fidelity candidates require native_feedback")
            if self.request_count is not None:
                raise StudyValidationError(
                    "native cost/fidelity candidates may not set request_count; use certified policy cadence instead"
                )
            if set(self.policy_cadences) != set(PRIMARY_POLICIES):
                raise StudyValidationError("native candidates need certified cadence for exactly all six policies")
            supported = tuple(_int(length, "backend_supported_action_lengths", minimum=1) for length in self.backend_supported_action_lengths)
            if not supported or len(set(supported)) != len(supported):
                raise StudyValidationError("native candidates need unique supported backend action lengths")
            for policy in PRIMARY_POLICIES:
                cadence = self.policy_cadences[policy]
                if not isinstance(cadence, Mapping):
                    raise StudyValidationError("policy cadence for %s must be an object" % policy)
                prefix = _int(cadence.get("certified_execute_prefix"), "%s certified_execute_prefix" % policy, minimum=1)
                world_length = _int(cadence.get("world_request_action_length"), "%s world_request_action_length" % policy, minimum=1)
                if prefix != world_length:
                    raise StudyValidationError(
                        "%s world request length must equal its certified execute prefix; requests cannot cross requery boundaries"
                        % policy
                    )
                if policy == "OpenVLA" and prefix != 1:
                    raise StudyValidationError("OpenVLA native feedback requires certified_execute_prefix=1")
                if prefix not in supported:
                    raise StudyValidationError("backend does not declare support for %s's native request length" % policy)
        else:
            if self.feedback_mode is FeedbackMode.NATIVE_FEEDBACK:
                raise StudyValidationError("1--6 transport sensitivity is not a native-feedback candidate")
            requests = _int(self.request_count, "request_count", minimum=1)
            if requests > 6:
                raise StudyValidationError("transport request_count must be between 1 and 6")
            if self.policy_cadences or self.backend_supported_action_lengths:
                raise StudyValidationError("transport sensitivity may not declare native policy cadence/support")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatingPoint":
        if not isinstance(payload, Mapping):
            raise StudyValidationError("operating point must be an object")
        resolution = payload.get("resolution")
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise StudyValidationError("operating point resolution must be [width, height]")
        return cls(
            operating_point_id=_text(payload.get("operating_point_id"), "operating_point_id"),
            resolution=(_int(resolution[0], "resolution[0]", minimum=1), _int(resolution[1], "resolution[1]", minimum=1)),
            diffusion_steps=_int(payload.get("diffusion_steps"), "diffusion_steps", minimum=1),
            backend_profile_id=_text(payload.get("backend_profile_id"), "backend_profile_id"),
            judge_profile_id=_text(payload.get("judge_profile_id"), "judge_profile_id"),
            feedback_mode=_normalise_feedback_mode(payload.get("feedback_mode")),
            comparison_kind=_text(payload.get("comparison_kind"), "comparison_kind"),
            policy_cadences=_json_mapping(payload.get("policy_cadences", {})),
            backend_supported_action_lengths=tuple(payload.get("backend_supported_action_lengths", ())),
            request_count=(
                None
                if payload.get("request_count") is None
                else _int(payload.get("request_count"), "request_count", minimum=1)
            ),
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "operating_point_id": self.operating_point_id,
            "resolution": list(self.resolution),
            "diffusion_steps": self.diffusion_steps,
            "backend_profile_id": self.backend_profile_id,
            "judge_profile_id": self.judge_profile_id,
            "feedback_mode": self.feedback_mode.value,
            "comparison_kind": self.comparison_kind,
            "policy_cadences": {policy: dict(value) for policy, value in self.policy_cadences.items()},
            "backend_supported_action_lengths": list(self.backend_supported_action_lengths),
            "request_count": self.request_count,
        }


def _cohort_starts(rows: Iterable[Mapping[str, Any]], cohort: str) -> Tuple[Dict[str, Any], ...]:
    parsed: List[Dict[str, Any]] = []
    seen: set = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise StudyValidationError("%s[%d] must be an object" % (cohort, index))
        task = _text(raw.get("task"), "%s task" % cohort)
        if task not in PRIMARY_TASKS:
            raise StudyValidationError("%s has an unknown benchmark task %s" % (cohort, task))
        start_id = _text(raw.get("start_id"), "%s start_id" % cohort)
        lineage = _text(raw.get("start_lineage_id", raw.get("source_lineage_id")), "%s start_lineage_id" % cohort)
        key = (task, start_id)
        if key in seen:
            raise StudyValidationError("%s duplicates task/start %s/%s" % (cohort, task, start_id))
        seen.add(key)
        parsed.append({"cohort": cohort, "task": task, "start_id": start_id, "start_lineage_id": lineage})
    if not parsed:
        raise StudyValidationError("%s must contain at least one start" % cohort)
    missing_tasks = sorted(set(PRIMARY_TASKS) - {row["task"] for row in parsed})
    if missing_tasks:
        raise StudyValidationError("%s lacks starts for %s" % (cohort, ", ".join(missing_tasks)))
    return tuple(sorted(parsed, key=lambda row: (row["task"], row["start_id"])))


@dataclass(frozen=True)
class CostFidelityDesign:
    """Fixed-horizon development and confirmation comparison schedule."""

    design_hash: str
    plan_hash: str
    conditions_json: str

    @property
    def conditions(self) -> List[Dict[str, Any]]:
        return json.loads(self.conditions_json)

    def to_mapping(self) -> Dict[str, Any]:
        conditions = self.conditions
        return {
            "schema_version": STUDY_SCHEMA_VERSION,
            "design_hash": self.design_hash,
            "plan_hash": self.plan_hash,
            "selection_status": "not_selected",
            "conditions": conditions,
            "invariants": {
                "task_horizons": dict(PRIMARY_TASK_HORIZONS),
                "same_seed_within_cohort_policy_task_start": True,
                "same_judge_profile_across_candidates": True,
                "world_requests_cut_at_certified_policy_feedback_boundaries": True,
                "no_cost_or_fidelity_result_is_embedded": True,
            },
        }


def build_cost_fidelity_design(
    plan: FrozenStudyPlan,
    development_starts: Iterable[Mapping[str, Any]],
    confirmation_starts: Iterable[Mapping[str, Any]],
    operating_points: Iterable[Union[OperatingPoint, Mapping[str, Any]]],
) -> CostFidelityDesign:
    """Build matched cost/fidelity conditions without selecting an operating point.

    Every candidate keeps the task's full horizon. World requests are cut at
    independently certified policy feedback boundaries; the 1--6 request
    display is a separate unqualified transport-sensitivity schedule. A
    policy/task/start receives the same policy/world/judge seeds across
    candidates in its cohort.
    """

    development = _cohort_starts(development_starts, COST_DEVELOPMENT_COHORT)
    confirmation = _cohort_starts(confirmation_starts, COST_CONFIRMATION_COHORT)
    points = tuple(item if isinstance(item, OperatingPoint) else OperatingPoint.from_mapping(item) for item in operating_points)
    if not points:
        raise StudyValidationError("at least one operating point is required")
    point_ids = [point.operating_point_id for point in points]
    if len(set(point_ids)) != len(point_ids):
        raise StudyValidationError("operating_point_id values must be unique")
    backend_ids = {point.backend_profile_id for point in points}
    judge_ids = {point.judge_profile_id for point in points}
    feedback_modes = {point.feedback_mode for point in points}
    if len(backend_ids) != 1 or len(judge_ids) != 1 or len(feedback_modes) != 1:
        raise StudyValidationError(
            "a cost/fidelity comparison must hold backend profile, judge profile, and feedback mode constant"
        )
    if next(iter(feedback_modes)) is not FeedbackMode.NATIVE_FEEDBACK:
        raise StudyValidationError("cost/fidelity candidates require declared native_feedback; approximations need a separate cohort")
    if any(point.comparison_kind != "native_cost_fidelity" for point in points):
        raise StudyValidationError("1--6 transport points belong to build_transport_sensitivity_design")
    cadence_identities = {
        canonical_json(
            {
                "policy_cadences": {policy: dict(point.policy_cadences[policy]) for policy in PRIMARY_POLICIES},
                "backend_supported_action_lengths": list(point.backend_supported_action_lengths),
            }
        )
        for point in points
    }
    if len(cadence_identities) != 1:
        raise StudyValidationError("native cost/fidelity candidates must hold certified cadence and backend support constant")
    lineage = validate_cohort_lineage_separation(
        {
            PRIMARY_COHORT: plan.episode_rows(),
            COST_DEVELOPMENT_COHORT: development,
            COST_CONFIRMATION_COHORT: confirmation,
        }
    )
    if lineage["status"] != "pass":
        raise StudyValidationError("cost panels do not have source-lineage separation: %s" % lineage["status"])
    base_seed = plan.protocol["seeds"]["base_seed"]
    conditions: List[Dict[str, Any]] = []
    for cohort_rows in (development, confirmation):
        for point in points:
            for row in cohort_rows:
                horizon = PRIMARY_TASK_HORIZONS[row["task"]]
                for policy in PRIMARY_POLICIES:
                    prefix = int(point.policy_cadences[policy]["certified_execute_prefix"])
                    partitions = tuple(min(prefix, horizon - offset) for offset in range(0, horizon, prefix))
                    if any(length not in point.backend_supported_action_lengths for length in partitions):
                        raise StudyValidationError(
                            "%s native terminal partition is not declared supported by the backend" % policy
                        )
                    boundaries: List[int] = []
                    total = 0
                    for length in partitions:
                        total += length
                        boundaries.append(total)
                    condition_base = {
                        "cohort": row["cohort"],
                        "policy": policy,
                        "task": row["task"],
                        "start_id": row["start_id"],
                        "operating_point_id": point.operating_point_id,
                    }
                    conditions.append(
                        {
                            "condition_id": "cost-" + json_digest(condition_base).split(":", 1)[1][:24],
                            **condition_base,
                            "start_lineage_id": row["start_lineage_id"],
                            "protocol_hash": plan.protocol_hash,
                            "plan_hash": plan.plan_hash,
                            "horizon_actions": horizon,
                            "request_action_lengths": list(partitions),
                            "autoregressive_boundaries": list(boundaries),
                            "policy_feedback_boundaries": list(boundaries),
                            "policy_query_count": len(boundaries),
                            "certified_execute_prefix": prefix,
                            "operating_point": point.to_mapping(),
                            "seeds": {
                                "policy": _seed(base_seed, "cost-policy", policy, row["task"], row["start_id"]),
                                "world": _seed(
                                    base_seed, "cost-world", "matched-start", row["task"], row["start_id"]
                                ),
                                "judge_samples": [
                                    _seed(base_seed, "cost-judge", policy, row["task"], row["start_id"], index)
                                    for index in range(5)
                                ],
                            },
                        }
                    )
    canonical_conditions = json.loads(canonical_json(conditions))
    return CostFidelityDesign(
        design_hash=json_digest(
            {"plan_hash": plan.plan_hash, "operating_points": [point.to_mapping() for point in points], "conditions": canonical_conditions}
        ),
        plan_hash=plan.plan_hash,
        conditions_json=canonical_json(canonical_conditions),
    )


@dataclass(frozen=True)
class TransportSensitivityDesign:
    """Explicitly unqualified 1--6 request transport/horizon sensitivity."""

    design_hash: str
    plan_hash: str
    conditions_json: str

    @property
    def conditions(self) -> List[Dict[str, Any]]:
        return json.loads(self.conditions_json)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema_version": STUDY_SCHEMA_VERSION,
            "design_hash": self.design_hash,
            "plan_hash": self.plan_hash,
            "classification": "unqualified_transport_horizon_sensitivity",
            "conditions": self.conditions,
            "limitations": [
                "A 1--6 world-request partition is transport/horizon sensitivity, not native policy feedback.",
                "No policy query or feedback boundary is inferred from the FD request partition.",
                "This design may not select a cost/fidelity setting or populate a qualified primary cell.",
            ],
        }


def build_transport_sensitivity_design(
    plan: FrozenStudyPlan,
    starts: Iterable[Mapping[str, Any]],
    operating_points: Iterable[Union[OperatingPoint, Mapping[str, Any]]],
) -> TransportSensitivityDesign:
    """Build the separate unqualified 1--6 request display schedule.

    The emitted request boundaries describe only the world-model transport. A
    policy's requery cadence is unknown in this arm and is deliberately not
    overwritten by the request partition.
    """

    rows = _cohort_starts(starts, TRANSPORT_SENSITIVITY_COHORT)
    points = tuple(item if isinstance(item, OperatingPoint) else OperatingPoint.from_mapping(item) for item in operating_points)
    if not points or any(point.comparison_kind != "transport_sensitivity" for point in points):
        raise StudyValidationError("transport sensitivity requires one or more transport_sensitivity operating points")
    point_ids = [point.operating_point_id for point in points]
    if len(point_ids) != len(set(point_ids)):
        raise StudyValidationError("operating_point_id values must be unique")
    if validate_cohort_lineage_separation({PRIMARY_COHORT: plan.episode_rows(), TRANSPORT_SENSITIVITY_COHORT: rows})[
        "status"
    ] != "pass":
        raise StudyValidationError("transport sensitivity starts must be disjoint from primary source lineages")
    base_seed = plan.protocol["seeds"]["base_seed"]
    conditions: List[Dict[str, Any]] = []
    for point in points:
        for row in rows:
            horizon = PRIMARY_TASK_HORIZONS[row["task"]]
            partitions = partition_horizon(horizon, int(point.request_count or 0))
            boundaries: List[int] = []
            total = 0
            for length in partitions:
                total += length
                boundaries.append(total)
            for policy in PRIMARY_POLICIES:
                base = {
                    "cohort": TRANSPORT_SENSITIVITY_COHORT,
                    "policy": policy,
                    "task": row["task"],
                    "start_id": row["start_id"],
                    "operating_point_id": point.operating_point_id,
                }
                conditions.append(
                    {
                        "condition_id": "transport-" + json_digest(base).split(":", 1)[1][:24],
                        **base,
                        "start_lineage_id": row["start_lineage_id"],
                        "protocol_hash": plan.protocol_hash,
                        "plan_hash": plan.plan_hash,
                        "horizon_actions": horizon,
                        "world_request_action_lengths": list(partitions),
                        "generated_autoregressive_boundaries": list(boundaries),
                        "policy_feedback_boundaries": None,
                        "policy_query_count": None,
                        "feedback_mode": point.feedback_mode.value,
                        "operating_point": point.to_mapping(),
                        "seeds": {
                            "policy": _seed(base_seed, "transport-policy", policy, row["task"], row["start_id"]),
                            "world": _seed(base_seed, "transport-world", "matched-start", row["task"], row["start_id"]),
                        },
                    }
                )
    canonical_conditions = json.loads(canonical_json(conditions))
    return TransportSensitivityDesign(
        design_hash=json_digest({"plan_hash": plan.plan_hash, "conditions": canonical_conditions}),
        plan_hash=plan.plan_hash,
        conditions_json=canonical_json(canonical_conditions),
    )


@dataclass(frozen=True)
class DriftDesign:
    """Teacher-forced/free-running pairs with action and generation settings fixed."""

    design_hash: str
    plan_hash: str
    comparisons_json: str

    @property
    def comparisons(self) -> List[Dict[str, Any]]:
        return json.loads(self.comparisons_json)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema_version": STUDY_SCHEMA_VERSION,
            "design_hash": self.design_hash,
            "plan_hash": self.plan_hash,
            "selection_status": "not_measured",
            "comparisons": self.comparisons,
            "required_measurements": ["action_alignment", "integrity", "visual_divergence"],
            "limitation": "Partition sensitivity is not isolated drift when feedback cadence changes.",
        }


def build_drift_design(
    plan: FrozenStudyPlan,
    heldout_sequences: Iterable[Mapping[str, Any]],
    operating_points: Iterable[Union[OperatingPoint, Mapping[str, Any]]],
) -> DriftDesign:
    """Schedule paired teacher-forced/free-running held-out comparisons.

    Each source sequence needs immutable action/video references.  Both modes
    share the exact source actions, generation seed, full task horizon and
    operating point; only the declared feedback regime varies.
    """

    points = tuple(item if isinstance(item, OperatingPoint) else OperatingPoint.from_mapping(item) for item in operating_points)
    if not points:
        raise StudyValidationError("at least one operating point is required")
    if any(point.comparison_kind != "transport_sensitivity" for point in points):
        raise StudyValidationError(
            "recorded-action drift replay uses explicit transport_sensitivity points; it must not invent native policy boundaries"
        )
    sequences: List[Dict[str, Any]] = []
    seen = set()
    for index, row in enumerate(heldout_sequences):
        if not isinstance(row, Mapping):
            raise StudyValidationError("heldout_sequences[%d] must be an object" % index)
        task = _text(row.get("task"), "drift sequence task")
        if task not in PRIMARY_TASKS:
            raise StudyValidationError("drift sequence has an unknown task")
        sequence_id = _text(row.get("sequence_id"), "drift sequence_id")
        lineage = _text(row.get("source_lineage_id", row.get("start_lineage_id")), "drift source_lineage_id")
        actions_hash = _sha256(row.get("actions_hash"), "drift actions_hash")
        video_hash = _sha256(row.get("video_hash"), "drift video_hash")
        control_ticks = _int(row.get("recorded_control_ticks"), "drift recorded_control_ticks", minimum=1)
        if control_ticks < PRIMARY_TASK_HORIZONS[task]:
            raise StudyValidationError("drift source sequence is shorter than the frozen task horizon")
        timestamps_hash = row.get("control_timestamps_hash")
        if timestamps_hash is not None:
            timestamps_hash = _sha256(timestamps_hash, "drift control_timestamps_hash")
        key = (task, sequence_id)
        if key in seen:
            raise StudyValidationError("drift sequences duplicate task/sequence_id")
        seen.add(key)
        sequences.append(
            {
                "cohort": DRIFT_HELDOUT_COHORT,
                "task": task,
                "sequence_id": sequence_id,
                "start_lineage_id": lineage,
                "actions_hash": actions_hash,
                "video_hash": video_hash,
                "recorded_control_ticks": control_ticks,
                "control_timestamps_hash": timestamps_hash,
            }
        )
    if not sequences:
        raise StudyValidationError("at least one held-out drift sequence is required")
    lineage = validate_cohort_lineage_separation(
        {PRIMARY_COHORT: plan.episode_rows(), DRIFT_HELDOUT_COHORT: sequences}
    )
    if lineage["status"] != "pass":
        raise StudyValidationError("drift heldout sequences do not have source-lineage separation")
    base_seed = plan.protocol["seeds"]["base_seed"]
    comparisons: List[Dict[str, Any]] = []
    for point in points:
        for sequence in sorted(sequences, key=lambda item: (item["task"], item["sequence_id"])):
            horizon = PRIMARY_TASK_HORIZONS[sequence["task"]]
            partitions = partition_horizon(horizon, int(point.request_count or 0))
            boundaries: List[int] = []
            total = 0
            for length in partitions:
                total += length
                boundaries.append(total)
            comparison_base = {
                "task": sequence["task"],
                "sequence_id": sequence["sequence_id"],
                "operating_point_id": point.operating_point_id,
            }
            comparisons.append(
                {
                    "comparison_id": "drift-" + json_digest(comparison_base).split(":", 1)[1][:24],
                    **comparison_base,
                    "cohort": DRIFT_HELDOUT_COHORT,
                    "protocol_hash": plan.protocol_hash,
                    "plan_hash": plan.plan_hash,
                    "source_lineage_id": sequence["start_lineage_id"],
                    "source_actions_hash": sequence["actions_hash"],
                    "source_video_hash": sequence["video_hash"],
                    "source_recorded_control_ticks": sequence["recorded_control_ticks"],
                    "source_control_timestamps_hash": sequence["control_timestamps_hash"],
                    "horizon_actions": horizon,
                    "world_request_action_lengths": list(partitions),
                    "generated_autoregressive_boundaries": list(boundaries),
                    "policy_feedback_boundaries": None,
                    "policy_queries": 0,
                    "recorded_action_replay": True,
                    "operating_point": point.to_mapping(),
                    "generation_seed": _seed(
                        base_seed, "drift-world", "heldout", sequence["task"], sequence["sequence_id"]
                    ),
                    "arms": [
                        {"mode": "teacher_forced", "same_actions": True, "same_generation_settings": True},
                        {"mode": "free_running", "same_actions": True, "same_generation_settings": True},
                    ],
                }
            )
    canonical_comparisons = json.loads(canonical_json(comparisons))
    return DriftDesign(
        design_hash=json_digest({"plan_hash": plan.plan_hash, "comparisons": canonical_comparisons}),
        plan_hash=plan.plan_hash,
        comparisons_json=canonical_json(canonical_comparisons),
    )


def _artifact_ref(value: Any, field: str, *, require_created_at: bool = False) -> Tuple[Optional[str], List[str], Optional[datetime]]:
    """Validate an immutable artifact binding and return a de-duplication key."""

    errors: List[str] = []
    if not isinstance(value, Mapping):
        return None, ["%s must be an artifact object with uri and sha256" % field], None
    uri = value.get("uri", value.get("artifact_uri"))
    digest = value.get("sha256", value.get("digest"))
    try:
        normalized_uri = _text(uri, "%s.uri" % field)
        normalized_digest = _sha256(digest, "%s.sha256" % field)
    except StudyValidationError as error:
        return None, [str(error)], None
    created = value.get("created_at", value.get("generated_at"))
    parsed_created = None
    if require_created_at:
        try:
            parsed_created = _parse_timestamp(created, "%s.created_at" % field)
        except StudyValidationError as error:
            errors.append(str(error))
    elif created is not None:
        try:
            parsed_created = _parse_timestamp(created, "%s.created_at" % field)
        except StudyValidationError as error:
            errors.append(str(error))
    return "%s|%s" % (normalized_uri, normalized_digest), errors, parsed_created


def _artifact_uri(value: Any) -> Optional[str]:
    """Return the immutable URI/version identity without treating a new hash as a new artifact."""

    if not isinstance(value, Mapping):
        return None
    uri = value.get("uri", value.get("artifact_uri"))
    return uri.strip() if isinstance(uri, str) and uri.strip() else None


def _episode_policy(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        value = value.get("name")
    return str(value).strip() if value is not None and str(value).strip() else None


def _episode_seed(row: Mapping[str, Any]) -> Optional[int]:
    seeds = row.get("seeds")
    value = seeds.get("world") if isinstance(seeds, Mapping) else row.get("world_seed")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _episode_artifact(row: Mapping[str, Any]) -> Any:
    return row.get("generation_artifact_ref", row.get("video_ref", row.get("segments_manifest_ref")))


def _rehearsal_threshold(plan: FrozenStudyPlan) -> float:
    thresholds = plan.protocol.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise StudyValidationError("frozen plan has no thresholds object")
    value = thresholds.get("minimum_scientific_coverage")
    number = _finite_number(value, "thresholds.minimum_scientific_coverage", minimum=0.0)
    if number > 1:
        raise StudyValidationError("thresholds.minimum_scientific_coverage must not exceed 1")
    return number


def _validate_one_rehearsal(plan: FrozenStudyPlan, raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate one fresh 1,500-row rehearsal without awarding Gate F."""

    errors: List[str] = []
    if not isinstance(raw, Mapping):
        return {"status": "invalid", "errors": ["rehearsal must be an object"]}
    run_id = str(raw.get("run_id", "")).strip()
    if not run_id:
        errors.append("run_id is required")
    if raw.get("plan_hash") != plan.plan_hash:
        errors.append("plan_hash does not match frozen full-study plan")
    if raw.get("protocol_hash") != plan.protocol_hash:
        errors.append("protocol_hash does not match frozen plan")
    if raw.get("fresh_execution") is not True:
        errors.append("fresh_execution must be true; cached/replayed work cannot qualify")
    if _contains_synthetic(raw.get("evidence_kind")):
        errors.append("synthetic/mock rehearsal evidence cannot qualify")
    try:
        accepted = _parse_timestamp(raw.get("run_accepted_at"), "run_accepted_at")
        finalized = _parse_timestamp(raw.get("finalized_at"), "finalized_at")
        if finalized < accepted:
            errors.append("finalized_at precedes run_accepted_at")
        elapsed_seconds = (finalized - accepted).total_seconds()
    except StudyValidationError as error:
        errors.append(str(error))
        accepted = finalized = None
        elapsed_seconds = None
    records = raw.get("episodes")
    if not isinstance(records, list):
        errors.append("episodes must be a list of final logical records")
        records = []
    expected = {
        (episode.policy, episode.task, episode.start_id, episode.world_seed): episode for episode in plan.episodes
    }
    # See materialize_study_run: this property is a defensive JSON copy, so
    # parsing it in the 1,500-row loop would turn validation quadratic.
    revision_bindings = plan.protocol["revision_bindings"]
    observed: set = set()
    request_ids: List[str] = []
    artifact_keys: List[str] = []
    artifact_uris: List[str] = []
    judge_keys: List[str] = []
    judge_uris: List[str] = []
    eligible = 0
    judged_eligible = 0
    evaluable_outcomes = 0
    terminal_completed = 0
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            errors.append("episodes[%d] is not an object" % index)
            continue
        policy = _episode_policy(row.get("policy"))
        task = row.get("task")
        start_id = row.get("start_id")
        seed = _episode_seed(row)
        identity = (policy, task, start_id, seed)
        expected_episode = expected.get(identity)
        if expected_episode is None:
            errors.append("episodes[%d] is not an expected plan identity" % index)
        elif identity in observed:
            errors.append("episodes[%d] duplicates a logical plan identity" % index)
        else:
            observed.add(identity)
        if row.get("run_id") != run_id:
            errors.append("episodes[%d] run_id does not match rehearsal" % index)
        if row.get("cohort") != BURST_REHEARSAL_COHORT:
            errors.append("episodes[%d] must be labelled burst_rehearsal" % index)
        if row.get("protocol_hash") != plan.protocol_hash:
            errors.append("episodes[%d] protocol_hash does not match rehearsal plan" % index)
        if expected_episode is not None:
            if row.get("horizon_actions") != expected_episode.horizon_actions:
                errors.append("episodes[%d] changes the frozen task horizon" % index)
            if row.get("executed_actions") != expected_episode.horizon_actions:
                errors.append("episodes[%d] does not execute the exact frozen task horizon" % index)
            seeds = row.get("seeds")
            if not isinstance(seeds, Mapping) or seeds.get("policy") != expected_episode.policy_seed:
                errors.append("episodes[%d] policy seed does not match frozen plan" % index)
            elif seeds.get("world") != expected_episode.world_seed:
                errors.append("episodes[%d] world seed does not match frozen matched-start plan" % index)
            elif seeds.get("judge_samples") != list(expected_episode.judge_sample_seeds):
                errors.append("episodes[%d] judge sample seeds do not match frozen plan" % index)
            if row.get("policy_revision") != revision_bindings["policy"][expected_episode.policy]:
                errors.append("episodes[%d] policy revision does not match frozen wrapper binding" % index)
            if row.get("world_model_revision") != revision_bindings["world_model"]:
                errors.append("episodes[%d] world-model revision does not match frozen backend binding" % index)
            if row.get("judge_revision") != revision_bindings["judge"]:
                errors.append("episodes[%d] judge revision does not match frozen judge binding" % index)
        if row.get("status") != "completed":
            errors.append("episodes[%d] is %r, not completed" % (index, row.get("status")))
        else:
            terminal_completed += 1
        artifact_key, artifact_errors, created_at = _artifact_ref(
            _episode_artifact(row), "episodes[%d].generation_artifact_ref" % index, require_created_at=True
        )
        errors.extend(artifact_errors)
        if artifact_key is not None:
            artifact_keys.append(artifact_key)
        artifact_uri = _artifact_uri(_episode_artifact(row))
        if artifact_uri is not None:
            artifact_uris.append(artifact_uri)
        if accepted is not None and created_at is not None and created_at < accepted:
            errors.append("episodes[%d] generation artifact predates run acceptance" % index)
        if finalized is not None and created_at is not None and created_at > finalized:
            errors.append("episodes[%d] generation artifact postdates finalization" % index)
        platform_ids = row.get("platform_request_ids")
        if not isinstance(platform_ids, list) or not platform_ids or any(not isinstance(item, str) or not item.strip() for item in platform_ids):
            errors.append("episodes[%d] needs non-empty platform_request_ids" % index)
        else:
            request_ids.extend(item.strip() for item in platform_ids)
        validity = row.get("validity")
        if validity not in {"valid", "invalid", "unknown"}:
            errors.append("episodes[%d] validity must be valid, invalid, or unknown" % index)
        judge_ref = row.get("raw_judge_samples_ref", row.get("judge_artifact_ref"))
        if validity == "valid":
            eligible += 1
            judge_key, judge_errors, judge_created = _artifact_ref(
                judge_ref, "episodes[%d].raw_judge_samples_ref" % index, require_created_at=True
            )
            errors.extend(judge_errors)
            if judge_key is not None:
                judged_eligible += 1
                judge_keys.append(judge_key)
            judge_uri = _artifact_uri(judge_ref)
            if judge_uri is not None:
                judge_uris.append(judge_uri)
            if accepted is not None and judge_created is not None and judge_created < accepted:
                errors.append("episodes[%d] judge artifact predates run acceptance" % index)
            if finalized is not None and judge_created is not None and judge_created > finalized:
                errors.append("episodes[%d] judge artifact postdates finalization" % index)
            binary_success = row.get("binary_success")
            judge_status = row.get("judge_status")
            sampling = row.get("judge_sampling")
            agreeing = row.get("agreeing_samples")
            if isinstance(binary_success, bool):
                if judge_status != "evaluable":
                    errors.append("episodes[%d] has a binary judge outcome without evaluable judge_status" % index)
                elif (
                    not isinstance(sampling, Mapping)
                    or sampling.get("sample_count") != 5
                    or sampling.get("quorum") != 3
                    or isinstance(agreeing, bool)
                    or not isinstance(agreeing, int)
                    or agreeing < 3
                ):
                    errors.append("episodes[%d] binary outcome lacks a bound five-sample/three-quorum judge result" % index)
                else:
                    evaluable_outcomes += 1
            elif binary_success is not None:
                errors.append("episodes[%d] binary_success must be strictly boolean or null" % index)
            else:
                if judge_status == "evaluable":
                    errors.append("episodes[%d] evaluable judge_status requires a boolean binary_success" % index)
                if not isinstance(row.get("missing_reason"), str) or not row.get("missing_reason").strip():
                    errors.append("episodes[%d] unevaluable valid video needs an explicit missing_reason" % index)
        else:
            if row.get("binary_success") is not None:
                errors.append("episodes[%d] invalid/unknown video may not carry a binary outcome" % index)
            if judge_ref is None and (not isinstance(row.get("missing_reason"), str) or not row.get("missing_reason").strip()):
                errors.append("episodes[%d] needs a scoring artifact or explicit unevaluable missing_reason" % index)
    if len(records) != PRIMARY_EPISODE_COUNT:
        errors.append("rehearsal has %d records; requires %d" % (len(records), PRIMARY_EPISODE_COUNT))
    missing = len(set(expected) - observed)
    if missing:
        errors.append("rehearsal is missing %d planned logical identities" % missing)
    if len(set(request_ids)) != len(request_ids):
        errors.append("platform request IDs repeat within rehearsal")
    if len(set(artifact_keys)) != len(artifact_keys):
        errors.append("generation artifact references repeat within rehearsal")
    if len(set(artifact_uris)) != len(artifact_uris):
        errors.append("generation artifact URI/version identities repeat within rehearsal")
    if len(set(judge_keys)) != len(judge_keys):
        errors.append("judge artifact references repeat within rehearsal")
    if len(set(judge_uris)) != len(judge_uris):
        errors.append("judge artifact URI/version identities repeat within rehearsal")
    judge_attempt_coverage = judged_eligible / float(eligible) if eligible else 0.0
    scientific_coverage = evaluable_outcomes / float(PRIMARY_EPISODE_COUNT)
    required_coverage = _rehearsal_threshold(plan)
    cost = raw.get("cost")
    estimated_usd: Optional[float] = None
    if not isinstance(cost, Mapping):
        errors.append("cost must be a total demonstration-run cost object")
    else:
        if cost.get("kind") != "total_demonstration":
            errors.append("cost.kind must be total_demonstration")
        if cost.get("status") not in {"estimated", "settled"}:
            errors.append("cost.status must be estimated or settled")
        try:
            estimated_usd = _finite_number(cost.get("usd"), "cost.usd", minimum=0.0)
        except StudyValidationError as error:
            errors.append(str(error))
        for field in ("pricing_snapshot_ref", "allocation_ledger_ref"):
            _, reference_errors, _ = _artifact_ref(cost.get(field), "cost.%s" % field)
            errors.extend(reference_errors)
    target_errors: List[str] = []
    if elapsed_seconds is None or elapsed_seconds > BURST_TARGET_SECONDS:
        target_errors.append("elapsed time is unavailable or exceeds %.0f seconds" % BURST_TARGET_SECONDS)
    if estimated_usd is None or estimated_usd > BURST_TARGET_USD:
        target_errors.append("total demonstration cost is unavailable or exceeds $%.2f" % BURST_TARGET_USD)
    if terminal_completed != PRIMARY_EPISODE_COUNT:
        target_errors.append("not all 1,500 planned episodes completed generation")
    if eligible == 0 or judge_attempt_coverage < 1.0:
        target_errors.append("not every eligible valid video received a judge attempt")
    if scientific_coverage < required_coverage:
        target_errors.append("evaluable scientific coverage V/N is below the frozen threshold")
    return {
        "run_id": run_id or None,
        "status": "invalid" if errors else "valid_evidence" if not target_errors else "valid_evidence_target_missed",
        "errors": list(dict.fromkeys(errors)),
        "target_errors": target_errors,
        "elapsed_seconds": elapsed_seconds,
        "estimated_usd": estimated_usd,
        "cost_status": cost.get("status") if isinstance(cost, Mapping) else None,
        "planned_records": PRIMARY_EPISODE_COUNT,
        "observed_records": len(records),
        "completed_records": terminal_completed,
        "eligible_videos": eligible,
        "judged_eligible_videos": judged_eligible,
        "judge_attempt_coverage": judge_attempt_coverage,
        "evaluable_outcomes": evaluable_outcomes,
        "scientific_coverage": scientific_coverage,
        "minimum_scientific_coverage": required_coverage,
        "platform_request_ids": request_ids,
        "generation_artifact_keys": artifact_keys,
        "generation_artifact_uris": artifact_uris,
        "judge_artifact_keys": judge_keys,
        "judge_artifact_uris": judge_uris,
        "run_accepted_at": _as_utc_text(accepted) if accepted is not None else None,
        "finalized_at": _as_utc_text(finalized) if finalized is not None else None,
    }


def validate_burst_rehearsals(plan: FrozenStudyPlan, rehearsals: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Validate exactly three fresh full-matrix rehearsal records.

    The report is evidence validation, not a Gate-F mutation.  Callers must
    still bind it to an actual Gate F record with real evidence URIs.
    """

    reports = [_validate_one_rehearsal(plan, raw) for raw in rehearsals]
    errors: List[str] = []
    if len(reports) != BURST_REHEARSAL_COUNT:
        errors.append("requires exactly %d fresh rehearsal records" % BURST_REHEARSAL_COUNT)
    run_ids = [report.get("run_id") for report in reports]
    if None in run_ids or len(set(run_ids)) != len(run_ids):
        errors.append("rehearsal run_id values must be non-empty and unique")
    for field, label in (
        ("platform_request_ids", "platform request IDs"),
        ("generation_artifact_uris", "generation artifact URI/version identities"),
        ("judge_artifact_uris", "judge artifact URI/version identities"),
    ):
        seen: set = set()
        duplicate = False
        for report in reports:
            for value in report.get(field, []):
                if value in seen:
                    duplicate = True
                seen.add(value)
        if duplicate:
            errors.append("%s are reused across rehearsal runs" % label)
    invalid = [report.get("run_id") for report in reports if report.get("status") == "invalid"]
    missed = [report.get("run_id") for report in reports if report.get("target_errors")]
    if invalid:
        errors.append("invalid rehearsal evidence for %s" % ", ".join(str(value) for value in invalid))
    if missed:
        errors.append("burst target not met by %s" % ", ".join(str(value) for value in missed))
    public_report = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "plan_hash": plan.plan_hash,
        "protocol_hash": plan.protocol_hash,
        "required_rehearsal_count": BURST_REHEARSAL_COUNT,
        "target": {"elapsed_seconds": BURST_TARGET_SECONDS, "total_demonstration_usd": BURST_TARGET_USD},
        "rehearsals": [
            {
                key: value
                for key, value in report.items()
                if key
                not in {
                    "platform_request_ids",
                    "generation_artifact_keys",
                    "generation_artifact_uris",
                    "judge_artifact_keys",
                    "judge_artifact_uris",
                }
            }
            for report in reports
        ],
        "status": "evidence_validated" if not errors else "not_qualified",
        "errors": list(dict.fromkeys(errors)),
        "limitations": [
            "This validates supplied record bindings; it does not attest to remote platform execution.",
            "An estimated cost remains an estimate until billing reconciliation.",
        ],
    }
    public_report["report_hash"] = json_digest(public_report)
    return public_report


def assess_burst_target_claim(
    plan: FrozenStudyPlan,
    scenario_manifest: Mapping[str, Any],
    gate_ledger: Union[GateLedger, Mapping[str, Any]],
    rehearsal_report: Mapping[str, Any],
    *,
    asset_lock: Optional[Any] = None,
) -> Dict[str, Any]:
    """Fail closed unless Gates A--F and validated fresh evidence all bind."""

    errors: List[str] = []
    primary = assess_primary_readiness(plan, scenario_manifest, gate_ledger, asset_lock=asset_lock)
    if primary["status"] != "ready":
        errors.extend("primary readiness: " + error for error in primary["errors"])
    if not isinstance(rehearsal_report, Mapping):
        raise StudyValidationError("rehearsal_report must be a mapping from validate_burst_rehearsals")
    if rehearsal_report.get("plan_hash") != plan.plan_hash or rehearsal_report.get("protocol_hash") != plan.protocol_hash:
        errors.append("rehearsal report does not bind this frozen plan/protocol")
    if rehearsal_report.get("status") != "evidence_validated":
        errors.extend("rehearsal evidence: " + str(error) for error in rehearsal_report.get("errors", []))
    report_hash = rehearsal_report.get("report_hash")
    if not isinstance(report_hash, str) or not report_hash.startswith("sha256:"):
        errors.append("rehearsal report lacks an immutable report_hash")
    ledger = _coerce_gate_ledger(gate_ledger)
    for gate_id in ("E", "F"):
        errors.extend(_gate_actual_evidence_errors(ledger.records[gate_id], gate_id, plan))
    gate_e = ledger.records["E"]
    if not gate_e.measurements.get("full_matrix_completed"):
        errors.append("Gate E does not record a completed full matrix")
    gate_f = ledger.records["F"]
    measurement = gate_f.measurements
    run_ids = [entry.get("run_id") for entry in rehearsal_report.get("rehearsals", []) if isinstance(entry, Mapping)]
    if measurement.get("rehearsal_validation_hash") != report_hash:
        errors.append("Gate F does not bind this rehearsal validation hash")
    if set(measurement.get("rehearsal_run_ids", [])) != set(run_ids) or len(run_ids) != BURST_REHEARSAL_COUNT:
        errors.append("Gate F does not bind exactly these three rehearsal run IDs")
    deduplicated = list(dict.fromkeys(errors))
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "status": "claim_permitted" if not deduplicated else "blocked",
        "claim": "1,500 fresh episodes in <=60 seconds at <=$11.25 total demonstration-run cost",
        "plan_hash": plan.plan_hash,
        "protocol_hash": plan.protocol_hash,
        "rehearsal_report_hash": report_hash,
        "asset_lock_supplied": asset_lock is not None,
        "errors": deduplicated,
        "cost_qualification": "estimated unless the three rehearsal records report settled billing",
    }


def _read_json_file(path: str, description: str) -> Any:
    candidate = Path(path)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StudyValidationError("unable to read %s JSON: %s" % (description, candidate)) from error
    return value


def _read_object(path: str, description: str) -> Dict[str, Any]:
    value = _read_json_file(path, description)
    if not isinstance(value, Mapping):
        raise StudyValidationError("%s JSON must contain one object" % description)
    return dict(value)


def _read_rows(path: str, description: str) -> List[Dict[str, Any]]:
    value = _read_json_file(path, description)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise StudyValidationError("%s JSON must contain a list of objects" % description)
    return [dict(item) for item in value]


def _emit_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Offline CLI wrapper intended for ``plumb study ...`` routing.

    It writes only JSON to stdout.  A caller that wants durable storage should
    use its own write-once release/record workflow rather than treating this
    convenience command as a source of external evidence.
    """

    parser = argparse.ArgumentParser(description="Freeze and validate PLUMB full-study control-plane records.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    plan_parser = subcommands.add_parser("plan", help="emit a full immutable 6x5x50 study plan")
    plan_parser.add_argument("--scenario", required=True, help="frozen scenario manifest JSON")
    plan_parser.add_argument("--protocol", required=True, help="protocol metadata JSON")
    plan_parser.add_argument("--base-seed", required=True, type=int)
    plan_parser.add_argument("--feedback-mode", default=None, choices=[mode.value for mode in FeedbackMode])

    materialize_parser = subcommands.add_parser("materialize", help="emit run-unique ledger rows for a frozen plan")
    materialize_parser.add_argument("--plan", required=True)
    materialize_parser.add_argument("--run-id", required=True)
    materialize_parser.add_argument("--cohort", default=PRIMARY_COHORT, choices=[PRIMARY_COHORT, BURST_REHEARSAL_COHORT])

    readiness_parser = subcommands.add_parser("readiness", help="fail-closed A-D execution readiness")
    readiness_parser.add_argument("--plan", required=True)
    readiness_parser.add_argument("--scenario", required=True)
    readiness_parser.add_argument("--gates", required=True)
    readiness_parser.add_argument("--cohorts", help="optional object mapping cohort names to start rows")
    readiness_parser.add_argument("--asset-lock", help="verified assets.lock.json bound by the frozen protocol")

    cost_parser = subcommands.add_parser("cost-design", help="emit matched fixed-horizon cost/fidelity conditions")
    cost_parser.add_argument("--plan", required=True)
    cost_parser.add_argument("--development-starts", required=True)
    cost_parser.add_argument("--confirmation-starts", required=True)
    cost_parser.add_argument("--operating-points", required=True)

    transport_parser = subcommands.add_parser("transport-design", help="emit unqualified 1-6 request transport sensitivity conditions")
    transport_parser.add_argument("--plan", required=True)
    transport_parser.add_argument("--starts", required=True)
    transport_parser.add_argument("--operating-points", required=True)

    drift_parser = subcommands.add_parser("drift-design", help="emit paired held-out drift conditions")
    drift_parser.add_argument("--plan", required=True)
    drift_parser.add_argument("--heldout-sequences", required=True)
    drift_parser.add_argument("--operating-points", required=True)

    rehearsal_parser = subcommands.add_parser("validate-rehearsals", help="validate three fresh burst evidence records")
    rehearsal_parser.add_argument("--plan", required=True)
    rehearsal_parser.add_argument("--rehearsals", required=True)

    claim_parser = subcommands.add_parser("claim", help="check whether evidence permits the public burst target claim")
    claim_parser.add_argument("--plan", required=True)
    claim_parser.add_argument("--scenario", required=True)
    claim_parser.add_argument("--gates", required=True)
    claim_parser.add_argument("--rehearsal-report", required=True)
    claim_parser.add_argument("--asset-lock", help="verified assets.lock.json bound by the frozen protocol")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "plan":
            plan = build_full_study_plan(
                _read_object(arguments.scenario, "scenario"),
                _read_object(arguments.protocol, "protocol metadata"),
                base_seed=arguments.base_seed,
                feedback_mode=arguments.feedback_mode,
            )
            _emit_json(plan.to_mapping())
            return 0
        plan = study_plan_from_mapping(_read_object(arguments.plan, "study plan"))
        if arguments.command == "materialize":
            _emit_json({"episodes": materialize_study_run(plan, arguments.run_id, cohort=arguments.cohort)})
            return 0
        if arguments.command == "readiness":
            cohorts = _read_object(arguments.cohorts, "cohorts") if arguments.cohorts else None
            _emit_json(
                assess_primary_readiness(
                    plan,
                    _read_object(arguments.scenario, "scenario"),
                    _read_object(arguments.gates, "gates"),
                    cohort_rows=cohorts,
                    asset_lock=_read_object(arguments.asset_lock, "asset lock") if arguments.asset_lock else None,
                )
            )
            return 0
        if arguments.command == "cost-design":
            design = build_cost_fidelity_design(
                plan,
                _read_rows(arguments.development_starts, "development starts"),
                _read_rows(arguments.confirmation_starts, "confirmation starts"),
                _read_rows(arguments.operating_points, "operating points"),
            )
            _emit_json(design.to_mapping())
            return 0
        if arguments.command == "drift-design":
            design = build_drift_design(
                plan,
                _read_rows(arguments.heldout_sequences, "heldout sequences"),
                _read_rows(arguments.operating_points, "operating points"),
            )
            _emit_json(design.to_mapping())
            return 0
        if arguments.command == "transport-design":
            design = build_transport_sensitivity_design(
                plan,
                _read_rows(arguments.starts, "transport starts"),
                _read_rows(arguments.operating_points, "operating points"),
            )
            _emit_json(design.to_mapping())
            return 0
        if arguments.command == "validate-rehearsals":
            _emit_json(validate_burst_rehearsals(plan, _read_rows(arguments.rehearsals, "rehearsals")))
            return 0
        if arguments.command == "claim":
            _emit_json(
                assess_burst_target_claim(
                    plan,
                    _read_object(arguments.scenario, "scenario"),
                    _read_object(arguments.gates, "gates"),
                    _read_object(arguments.rehearsal_report, "rehearsal report"),
                    asset_lock=_read_object(arguments.asset_lock, "asset lock") if arguments.asset_lock else None,
                )
            )
            return 0
    except StudyValidationError as error:
        parser.error(str(error))
    return 2


__all__ = [
    "BURST_REHEARSAL_COHORT",
    "BURST_REHEARSAL_COUNT",
    "BURST_TARGET_SECONDS",
    "BURST_TARGET_USD",
    "COST_CONFIRMATION_COHORT",
    "COST_DEVELOPMENT_COHORT",
    "CostFidelityDesign",
    "DriftDesign",
    "FrozenStudyPlan",
    "OperatingPoint",
    "PRIMARY_EPISODE_COUNT",
    "PRIMARY_POLICIES",
    "PRIMARY_STARTS_PER_TASK",
    "PRIMARY_TASK_HORIZONS",
    "PRIMARY_TASKS",
    "StudyEpisode",
    "StudyValidationError",
    "TransportSensitivityDesign",
    "assess_burst_target_claim",
    "assess_primary_readiness",
    "build_cost_fidelity_design",
    "build_drift_design",
    "build_full_study_plan",
    "build_transport_sensitivity_design",
    "materialize_study_run",
    "partition_horizon",
    "study_plan_from_mapping",
    "validate_burst_rehearsals",
    "validate_cohort_lineage_separation",
    "main",
]
