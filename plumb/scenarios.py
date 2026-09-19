"""Safe scenario-manifest ingestion and readiness checks for PLUMB.

This module prepares evidence for Gate C; it cannot mark Gate C passed.  A
complete, frozen manifest establishes that a proposed panel has the required
shape and local artifact hashes.  Scene parity, reset-distribution comparison,
and controller comparability still require separately recorded evidence.

Scenario import is deliberately narrow.  It accepts JSON manifests supplied by
the caller, verifies referenced *local* image/JSON files against SHA-256 values,
and never imports pickle, torch checkpoints, remote URLs, or arbitrary source
objects.  It does not download, decode, or execute source artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .policies.tasks import BENCHMARK_TASK_REGISTRY


SCHEMA_VERSION = 1
PRIMARY_PANEL = "primary"
DEVELOPMENT_PANEL = "development"
HELDOUT_CALIBRATION_PANEL = "heldout_calibration"
COST_CONFIRMATION_PANEL = "cost_confirmation"
PANELS = (
    PRIMARY_PANEL,
    DEVELOPMENT_PANEL,
    HELDOUT_CALIBRATION_PANEL,
    COST_CONFIRMATION_PANEL,
)
PRIMARY_STARTS_PER_TASK = 50


@dataclass(frozen=True)
class TaskDefinition:
    """The benchmark task contract from the build specification."""

    task_id: str
    instruction: str
    success_criterion: str
    max_steps: int


# Prompts and horizons derive from the frozen primary-judge registry, avoiding
# a second copy that can silently drift.  Success criteria are scenario-panel
# metadata rather than judge rubrics, so they stay here as the literal §3 text.
_SUCCESS_CRITERIA = {
    "close_drawer": "completely closed",
    "open_drawer": "opened >= 1.5 cm",
    "to_basket": "fully inside at episode end",
    "to_sink": "fully inside at episode end",
    "fold_cloth": "folded >= 1/4 diagonally",
}
TASK_REGISTRY: Tuple[TaskDefinition, ...] = tuple(
    TaskDefinition(task.task_id, task.instruction, _SUCCESS_CRITERIA[task.task_id], task.max_steps)
    for task in BENCHMARK_TASK_REGISTRY.tasks
)
TASKS: Tuple[str, ...] = tuple(task.task_id for task in TASK_REGISTRY)
TASK_BY_ID = {task.task_id: task for task in TASK_REGISTRY}

# These IDs intentionally use the already-canonical measurement/reference key
# ``Octo``.  The manifest below binds it to the required Octo-Small v1.0
# runtime, preventing a quiet replacement with another Octo checkpoint.
PRIMARY_POLICY_IDS = (
    "OpenVLA",
    "OpenPiZero",
    "Octo",
    "MiniVLA",
    "SuSIE",
    "SuSIE_LL",
)
PRIMARY_POLICY_IDENTITIES = (
    {"policy_id": "OpenVLA", "display_name": "OpenVLA"},
    {"policy_id": "OpenPiZero", "display_name": "OpenPiZero"},
    {"policy_id": "Octo", "display_name": "Octo-Small v1.0", "runtime_name": "OctoSmall"},
    {"policy_id": "MiniVLA", "display_name": "MiniVLA"},
    {"policy_id": "SuSIE", "display_name": "SuSIE"},
    {"policy_id": "SuSIE_LL", "display_name": "SuSIE_LL"},
)

_SAFE_ARTIFACT_TYPES = {
    "image/png": {".png"},
    "image/jpeg": {".jpg", ".jpeg"},
    "image/webp": {".webp"},
    "application/json": {".json"},
}
_UNSAFE_SUFFIXES = {".pkl", ".pickle", ".pt", ".pth", ".bin", ".npy", ".npz", ".joblib"}


class ScenarioError(ValueError):
    """Raised when a scenario or source manifest is unsafe or inconsistent."""


@dataclass(frozen=True)
class ScenarioSourceRecord:
    """A locally hash-verified source record used by a scenario start."""

    source_dataset: str
    source_revision: str
    episode_id: str
    frame_id: str
    start_lineage_id: str
    image_hash: str
    state_hash: str
    goal_reference_hashes: Tuple[str, ...]
    image_timestamp: Union[str, float, int]
    state_timestamp: Union[str, float, int]
    state_convention: str

    @property
    def identity(self) -> Tuple[str, str, str, str]:
        return (self.source_dataset, self.source_revision, self.episode_id, self.frame_id)


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used by manifest digests."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ScenarioError("manifest must be JSON serializable without NaN values") from error


def sha256_file(path: Union[str, Path]) -> str:
    """Digest bytes without parsing or executing the referenced artifact."""

    candidate = Path(path)
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ScenarioError("unable to read source artifact: {0}".format(candidate)) from error
    return "sha256:" + digest.hexdigest()


def scenario_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash every manifest field except its self-referential SHA-256 field."""

    if not isinstance(manifest, Mapping):
        raise ScenarioError("scenario manifest must be an object")
    payload = dict(manifest)
    payload.pop("sha256", None)
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def read_json_manifest(path: Union[str, Path], *, description: str) -> Dict[str, Any]:
    """Read one explicit JSON manifest, rejecting executable/pickle formats."""

    candidate = Path(path)
    if candidate.suffix.lower() != ".json":
        raise ScenarioError("{0} must be an explicit .json manifest".format(description))
    if not candidate.is_file():
        raise ScenarioError("{0} does not exist: {1}".format(description, candidate))
    try:
        parsed = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScenarioError("unable to parse {0} as JSON".format(description)) from error
    if not isinstance(parsed, Mapping):
        raise ScenarioError("{0} must contain one JSON object".format(description))
    return dict(parsed)


def load_explicit_source_manifest(path: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[Tuple[str, str, str, str], ScenarioSourceRecord]]:
    """Load a source manifest and verify every declared safe local artifact.

    The input schema is intentionally small::

        {"schema_version": 1, "source_manifest_id": "...", "records": [
          {"source_dataset": "...", "source_revision": "...",
           "episode_id": "...", "frame_id": "...", "start_lineage_id": "...",
           "image": {"path": "frame.png", "sha256": "sha256:...", "media_type": "image/png"},
           "state": {"path": "state.json", "sha256": "sha256:...", "media_type": "application/json"},
           "goal_references": [{"path": "goal.png", "sha256": "sha256:...", "media_type": "image/png"}]}
        ]}

    Paths are relative to this source-manifest file.  The function hashes bytes
    only; it never opens image/video/model formats with a deserializer.
    """

    source_path = Path(path).resolve()
    manifest = read_json_manifest(source_path, description="source manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ScenarioError("source manifest schema_version must be {0}".format(SCHEMA_VERSION))
    _require_string(manifest.get("source_manifest_id"), "source_manifest_id")
    if manifest.get("provenance_kind") != "real_robot":
        raise ScenarioError("source manifest provenance_kind must be real_robot")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ScenarioError("source manifest records must be a non-empty list")

    base = source_path.parent
    indexed: Dict[Tuple[str, str, str, str], ScenarioSourceRecord] = {}
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ScenarioError("source manifest record {0} must be an object".format(index))
        record = dict(raw)
        identity = tuple(
            _require_string(record.get(field), "source record {0}.{1}".format(index, field))
            for field in ("source_dataset", "source_revision", "episode_id", "frame_id")
        )
        if identity in indexed:
            raise ScenarioError("source manifest repeats dataset/revision/episode/frame identity")
        lineage = _require_string(record.get("start_lineage_id"), "source record start_lineage_id")
        image_timestamp = _timestamp(record.get("image_timestamp"), "source record image_timestamp")
        state_timestamp = _timestamp(record.get("state_timestamp"), "source record state_timestamp")
        state_convention = _require_string(record.get("state_convention"), "source record state_convention")
        image = _verified_artifact(record.get("image"), base, "source record image")
        state = _verified_artifact(record.get("state"), base, "source record state")
        if image["media_type"] not in {"image/png", "image/jpeg", "image/webp"}:
            raise ScenarioError("source record image must be an allowlisted image type")
        if state["media_type"] != "application/json":
            raise ScenarioError("source record state must be application/json; pickle/state-dict input is forbidden")
        _parse_bridge_state_json(Path(state["path"]))
        raw_goals = record.get("goal_references", record.get("goals"))
        if not isinstance(raw_goals, list) or not raw_goals:
            raise ScenarioError("source record goal_references must be a non-empty list")
        goals = tuple(_verified_artifact(goal, base, "source record goal_reference") for goal in raw_goals)
        if any(goal["media_type"] not in {"image/png", "image/jpeg", "image/webp"} for goal in goals):
            raise ScenarioError("source record goal_references must be allowlisted image types")
        indexed[identity] = ScenarioSourceRecord(
            source_dataset=identity[0],
            source_revision=identity[1],
            episode_id=identity[2],
            frame_id=identity[3],
            start_lineage_id=lineage,
            image_hash=image["sha256"],
            state_hash=state["sha256"],
            goal_reference_hashes=tuple(goal["sha256"] for goal in goals),
            image_timestamp=image_timestamp,
            state_timestamp=state_timestamp,
            state_convention=state_convention,
        )
    return manifest, indexed


def import_scenario_manifest(
    scenario_manifest_path: Union[str, Path], source_manifest_path: Union[str, Path]
) -> Dict[str, Any]:
    """Import a scenario proposal using a separately explicit source manifest.

    No source location encoded inside the scenario file is followed.  Requiring
    the caller to pass ``source_manifest_path`` makes the reviewed source set
    explicit and avoids remote or accidental object loading.
    """

    scenario_path = Path(scenario_manifest_path).resolve()
    manifest = read_json_manifest(scenario_path, description="scenario manifest")
    source_path = Path(source_manifest_path).resolve()
    source_manifest, source_records = load_explicit_source_manifest(source_path)
    expected_source_hash = manifest.get("source_manifest_sha256")
    if expected_source_hash is None and isinstance(manifest.get("source_manifest"), Mapping):
        expected_source_hash = manifest["source_manifest"].get("sha256")
    if expected_source_hash is None:
        raise ScenarioError("scenario manifest must bind source_manifest_sha256 from the explicit source manifest")
    _require_hash(expected_source_hash, "source_manifest_sha256")
    actual_source_hash = sha256_file(source_path)
    if not _hash_equal(expected_source_hash, actual_source_hash):
        raise ScenarioError("scenario manifest source_manifest_sha256 does not match explicit source manifest bytes")

    _validate_manifest_shape(manifest, require_full_primary=False, source_records=source_records)
    imported = json.loads(canonical_json(manifest))
    imported["source_manifest_sha256"] = actual_source_hash
    imported["source_manifest_id"] = source_manifest["source_manifest_id"]
    imported["source_manifest_path"] = str(source_path)
    imported.pop("sha256", None)
    return imported


def validate_scenario_manifest(
    manifest: Mapping[str, Any], *, require_full_primary: bool = False
) -> Dict[str, Any]:
    """Validate a manifest without opening source files or asserting parity.

    Use :func:`import_scenario_manifest` for an actual source-artifact import.
    This function is useful after a frozen source manifest has been moved to an
    offline review machine where its byte-level source files are unavailable.
    """

    _validate_manifest_shape(manifest, require_full_primary=require_full_primary, source_records=None)
    return json.loads(canonical_json(dict(manifest)))


def freeze_scenario_manifest(
    manifest: Mapping[str, Any], output_path: Union[str, Path], *, require_full_primary: bool = True
) -> Dict[str, Any]:
    """Write a hash-verifiable scenario freeze exactly once.

    Existing freezes are never overwritten.  Repeating the same call returns
    the verified existing payload; changing any content requires a new path and
    produces a new manifest ID/hash, which keeps amendments visible.
    """

    prepared = validate_scenario_manifest(manifest, require_full_primary=require_full_primary)
    _require_hash(prepared.get("source_manifest_sha256"), "source_manifest_sha256")
    prepared.pop("source_manifest_path", None)  # local convenience path is not a portable immutable identity.
    prepared["status"] = "frozen"
    prepared["frozen"] = True
    prepared.pop("sha256", None)
    prepared["sha256"] = scenario_manifest_hash(prepared)
    encoded = (canonical_json(prepared) + "\n").encode("utf-8")
    target = Path(output_path)
    if target.suffix.lower() != ".json":
        raise ScenarioError("frozen scenario output must have a .json suffix")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        existing = load_frozen_scenario_manifest(target)
        if existing.get("sha256") != prepared["sha256"]:
            raise ScenarioError("refusing to overwrite a different frozen scenario manifest")
        return existing
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except BaseException:
        # An incomplete newly-created file cannot be a valid freeze.  This is
        # the one recoverable cleanup path; it never touches an older freeze.
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return prepared


def load_frozen_scenario_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Read and verify an immutable freeze before it is attached to a protocol."""

    manifest = read_json_manifest(path, description="frozen scenario manifest")
    if manifest.get("status") != "frozen" or manifest.get("frozen") is not True:
        raise ScenarioError("scenario manifest is not marked frozen")
    expected = manifest.get("sha256")
    _require_hash(expected, "frozen scenario sha256")
    if not _hash_equal(expected, scenario_manifest_hash(manifest)):
        raise ScenarioError("frozen scenario manifest SHA-256 does not match its content")
    validate_scenario_manifest(manifest, require_full_primary=True)
    return manifest


def scenario_readiness_report(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Describe concrete prerequisites without qualifying a scenario panel.

    A structurally complete panel is reported as ``ready_for_gate_c_evidence``.
    It remains ``not_run`` for Gate C because comparison to source scene/reset
    distributions, control timing, and policy-wrapper configuration is external
    empirical work rather than a property of a JSON file.
    """

    errors: List[str] = []
    try:
        _validate_manifest_shape(manifest, require_full_primary=False, source_records=None)
    except ScenarioError as error:
        errors.append(str(error))
    task_groups = manifest.get("tasks") if isinstance(manifest, Mapping) else None
    task_rows: Dict[str, Dict[str, Any]] = {}
    seen_lineages: Dict[str, List[Dict[str, str]]] = {}
    for task in TASKS:
        group = task_groups.get(task, {}) if isinstance(task_groups, Mapping) else {}
        task_blockers: List[str] = []
        try:
            panel_rows = _panel_rows(group) if isinstance(group, Mapping) else {panel: [] for panel in PANELS}
        except ScenarioError as error:
            panel_rows = {panel: [] for panel in PANELS}
            task_blockers.append(str(error))
        counts = {panel: len(panel_rows[panel]) for panel in PANELS}
        if counts[PRIMARY_PANEL] != PRIMARY_STARTS_PER_TASK:
            task_blockers.append(
                "primary panel has {0} starts; requires exactly {1}".format(
                    counts[PRIMARY_PANEL], PRIMARY_STARTS_PER_TASK
                )
            )
        for panel in (DEVELOPMENT_PANEL, HELDOUT_CALIBRATION_PANEL, COST_CONFIRMATION_PANEL):
            if not counts[panel]:
                task_blockers.append("{0} panel has no real source starts".format(panel))
        for panel, rows in panel_rows.items():
            for start in rows:
                if isinstance(start, Mapping) and _nonempty_string(start.get("start_lineage_id")):
                    seen_lineages.setdefault(str(start["start_lineage_id"]), []).append(
                        {"task": task, "panel": panel, "start_id": str(start.get("start_id", ""))}
                    )
        task_rows[task] = {
            "instruction": TASK_BY_ID[task].instruction,
            "max_steps": TASK_BY_ID[task].max_steps,
            "success_criterion": TASK_BY_ID[task].success_criterion,
            "panel_counts": counts,
            "blockers": task_blockers,
        }
        errors.extend("{0}: {1}".format(task, blocker) for blocker in task_blockers)
    lineage_conflicts = [
        {"start_lineage_id": lineage, "uses": uses}
        for lineage, uses in sorted(seen_lineages.items())
        if len(uses) > 1
    ]
    if lineage_conflicts:
        errors.append("source lineage is reused across starts, tasks, or disjoint panels")
    hash_status = "missing"
    if isinstance(manifest, Mapping) and manifest.get("sha256"):
        try:
            hash_status = "valid" if _hash_equal(manifest.get("sha256"), scenario_manifest_hash(manifest)) else "mismatch"
        except ScenarioError:
            hash_status = "invalid"
        if hash_status != "valid":
            errors.append("declared scenario manifest SHA-256 is not valid")
    policy_matrix = manifest.get("shared_policy_ids") if isinstance(manifest, Mapping) else None
    policy_status = "missing"
    if isinstance(policy_matrix, list):
        policy_status = "valid" if tuple(policy_matrix) == PRIMARY_POLICY_IDS else "mismatch"
        if policy_status != "valid":
            errors.append("shared_policy_ids must name the frozen six-policy primary matrix")
    elif policy_matrix is not None:
        policy_status = "invalid"
        errors.append("shared_policy_ids must be a list when supplied")
    frozen = bool(isinstance(manifest, Mapping) and manifest.get("frozen") is True and manifest.get("status") == "frozen")
    if not frozen:
        errors.append("scenario selection is not frozen")
    if hash_status != "valid":
        errors.append("scenario freeze lacks a content-valid SHA-256")
    source_hash = manifest.get("source_manifest_sha256") if isinstance(manifest, Mapping) else None
    if not _is_hash(source_hash):
        errors.append("source-manifest byte hash is missing or invalid")
    structural_ready = not errors
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest.get("manifest_id") if isinstance(manifest, Mapping) else None,
        "declared_provenance_kind": manifest.get("provenance_kind") if isinstance(manifest, Mapping) else None,
        "freeze": {"frozen": frozen, "sha256_status": hash_status},
        "policy_matrix": {"status": policy_status, "required_policy_ids": list(PRIMARY_POLICY_IDS)},
        "tasks": task_rows,
        "lineage": {
            "status": "fail" if lineage_conflicts else "pass",
            "conflicts": lineage_conflicts,
        },
        "source_provenance": {
            "source_manifest_id": manifest.get("source_manifest_id") if isinstance(manifest, Mapping) else None,
            "source_manifest_sha256": source_hash,
            "hash_format_valid": _is_hash(source_hash),
        },
        "status": "ready_for_gate_c_evidence" if structural_ready else "blocked",
        "gate_c": {
            "status": "not_run",
            "qualified": False,
            "reason": (
                "Manifest structure and local provenance are prepared; Gate C still needs recorded scene, reset, "
                "control-timing, and policy-wrapper parity evidence."
                if structural_ready
                else "Scenario manifest has unresolved structural/provenance blockers."
            ),
        },
        "blockers": errors,
    }


def _validate_manifest_shape(
    manifest: Mapping[str, Any], *, require_full_primary: bool, source_records: Optional[Mapping[Tuple[str, str, str, str], ScenarioSourceRecord]]
) -> None:
    if not isinstance(manifest, Mapping):
        raise ScenarioError("scenario manifest must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ScenarioError("scenario manifest schema_version must be {0}".format(SCHEMA_VERSION))
    _require_string(manifest.get("manifest_id"), "manifest_id")
    if "sha256" in manifest:
        _require_hash(manifest.get("sha256"), "scenario manifest sha256")
        if not _hash_equal(manifest.get("sha256"), scenario_manifest_hash(manifest)):
            raise ScenarioError("scenario manifest sha256 does not match its content")
    if manifest.get("provenance_kind") != "real_robot":
        raise ScenarioError("scenario manifest provenance_kind must be exactly real_robot; fixtures cannot qualify")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ScenarioError("scenario manifest tasks must be an object keyed by the five task IDs")
    unexpected_tasks = sorted(set(tasks) - set(TASKS))
    if unexpected_tasks:
        raise ScenarioError("scenario manifest has unknown task IDs: {0}".format(", ".join(str(value) for value in unexpected_tasks)))

    all_ids: Dict[str, Tuple[str, str]] = {}
    all_lineages: Dict[str, Tuple[str, str]] = {}
    for task in TASKS:
        group = tasks.get(task)
        if group is None:
            if require_full_primary:
                raise ScenarioError("scenario manifest is missing task {0}".format(task))
            continue
        if not isinstance(group, Mapping):
            raise ScenarioError("task {0} must be an object".format(task))
        panel_rows = _panel_rows(group)
        if require_full_primary and len(panel_rows[PRIMARY_PANEL]) != PRIMARY_STARTS_PER_TASK:
            raise ScenarioError(
                "task {0} requires exactly {1} primary starts".format(task, PRIMARY_STARTS_PER_TASK)
            )
        if require_full_primary:
            for panel in (DEVELOPMENT_PANEL, HELDOUT_CALIBRATION_PANEL, COST_CONFIRMATION_PANEL):
                if not panel_rows[panel]:
                    raise ScenarioError("task {0} requires a disjoint {1} panel".format(task, panel))
        for panel, rows in panel_rows.items():
            if not isinstance(rows, list):
                raise ScenarioError("task {0} panel {1} must be a list".format(task, panel))
            for index, raw_start in enumerate(rows):
                start = _validate_start(task, panel, raw_start, index, source_records)
                start_id = start["start_id"]
                lineage = start["start_lineage_id"]
                if start_id in all_ids:
                    previous = all_ids[start_id]
                    raise ScenarioError(
                        "start_id {0} is reused by {1}/{2} and {3}/{4}".format(
                            start_id, previous[0], previous[1], task, panel
                        )
                    )
                if lineage in all_lineages:
                    previous = all_lineages[lineage]
                    raise ScenarioError(
                        "source lineage {0} crosses {1}/{2} and {3}/{4}".format(
                            lineage, previous[0], previous[1], task, panel
                        )
                    )
                all_ids[start_id] = (task, panel)
                all_lineages[lineage] = (task, panel)
    if require_full_primary:
        matrix = manifest.get("shared_policy_ids")
        if not isinstance(matrix, list) or tuple(matrix) != PRIMARY_POLICY_IDS:
            raise ScenarioError("full primary freeze requires ordered shared_policy_ids for all six primary policies")
        _require_hash(manifest.get("source_manifest_sha256"), "source_manifest_sha256")
        _require_string(manifest.get("source_manifest_id"), "source_manifest_id")


def _panel_rows(group: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """Return the explicit primary panel plus ancillary disjoint panels."""

    starts = group.get("starts", group.get(PRIMARY_PANEL, []))
    panels = group.get("panels", {})
    if "starts" in group and PRIMARY_PANEL in group and group[PRIMARY_PANEL] != starts:
        raise ScenarioError("task group cannot provide conflicting starts and primary panels")
    if not isinstance(starts, list):
        raise ScenarioError("task primary starts must be a list")
    if not isinstance(panels, Mapping):
        raise ScenarioError("task panels must be an object")
    unknown = set(panels) - {DEVELOPMENT_PANEL, HELDOUT_CALIBRATION_PANEL, COST_CONFIRMATION_PANEL}
    if unknown:
        raise ScenarioError("task panels include unsupported cohort names: {0}".format(", ".join(sorted(unknown))))
    output: Dict[str, List[Any]] = {PRIMARY_PANEL: starts}
    for panel in (DEVELOPMENT_PANEL, HELDOUT_CALIBRATION_PANEL, COST_CONFIRMATION_PANEL):
        rows = panels.get(panel, [])
        if not isinstance(rows, list):
            raise ScenarioError("task panel {0} must be a list".format(panel))
        output[panel] = rows
    return output


def _validate_start(
    task: str,
    panel: str,
    raw: Any,
    index: int,
    source_records: Optional[Mapping[Tuple[str, str, str, str], ScenarioSourceRecord]],
) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ScenarioError("task {0} {1} start {2} must be an object".format(task, panel, index))
    start = dict(raw)
    for field in (
        "start_id",
        "start_lineage_id",
        "source_dataset",
        "source_revision",
        "episode_id",
        "frame_id",
        "image_timestamp",
        "state_timestamp",
        "state_convention",
        "camera",
        "crop",
        "camera_calibration",
        "scene",
        "scene_configuration",
        "initial_state_stratum",
        "instruction",
    ):
        _require_nonempty_value(start.get(field), "task {0} start {1}.{2}".format(task, index, field))
    if start["instruction"] != TASK_BY_ID[task].instruction:
        raise ScenarioError("task {0} start {1} instruction is not the verbatim benchmark prompt".format(task, start["start_id"]))
    for field in ("image_hash", "state_hash"):
        _require_hash(start.get(field), "task {0} start {1}.{2}".format(task, start["start_id"], field))
    goals = start.get("goal_reference_hashes")
    if not isinstance(goals, list) or not goals:
        raise ScenarioError("task {0} start {1} goal_reference_hashes must be a non-empty list".format(task, start["start_id"]))
    for goal in goals:
        _require_hash(goal, "goal_reference_hashes")
    provenance = start.get("goal_reference_provenance")
    if not isinstance(provenance, (Mapping, list)) or not provenance:
        raise ScenarioError("task {0} start {1} needs goal_reference_provenance".format(task, start["start_id"]))
    _timestamp(start["image_timestamp"], "task {0} start {1}.image_timestamp".format(task, start["start_id"]))
    _timestamp(start["state_timestamp"], "task {0} start {1}.state_timestamp".format(task, start["start_id"]))
    if start.get("task", task) != task:
        raise ScenarioError("task {0} start {1} task field disagrees with its task group".format(task, start["start_id"]))
    if source_records is not None:
        identity = tuple(start[field] for field in ("source_dataset", "source_revision", "episode_id", "frame_id"))
        source = source_records.get(identity)
        if source is None:
            raise ScenarioError("task {0} start {1} is absent from the explicit source manifest".format(task, start["start_id"]))
        if source.start_lineage_id != start["start_lineage_id"]:
            raise ScenarioError("task {0} start {1} lineage does not match the explicit source record".format(task, start["start_id"]))
        if not _hash_equal(source.image_hash, start["image_hash"]) or not _hash_equal(source.state_hash, start["state_hash"]):
            raise ScenarioError("task {0} start {1} image/state hash does not match the explicit source record".format(task, start["start_id"]))
        if tuple(start["goal_reference_hashes"]) != source.goal_reference_hashes:
            raise ScenarioError("task {0} start {1} goal hashes do not match the explicit source record".format(task, start["start_id"]))
        if source.image_timestamp != _timestamp(start["image_timestamp"], "start image_timestamp"):
            raise ScenarioError("task {0} start {1} image timestamp does not match the explicit source record".format(task, start["start_id"]))
        if source.state_timestamp != _timestamp(start["state_timestamp"], "start state_timestamp"):
            raise ScenarioError("task {0} start {1} state timestamp does not match the explicit source record".format(task, start["start_id"]))
        if source.state_convention != start["state_convention"]:
            raise ScenarioError("task {0} start {1} state convention does not match the explicit source record".format(task, start["start_id"]))
    return start


def _verified_artifact(raw: Any, base: Path, field_name: str) -> Dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ScenarioError("{0} must be an artifact object".format(field_name))
    path_value = _require_string(raw.get("path"), field_name + ".path")
    media_type = _require_string(raw.get("media_type"), field_name + ".media_type")
    expected_hash = _require_hash(raw.get("sha256"), field_name + ".sha256")
    if media_type not in _SAFE_ARTIFACT_TYPES:
        raise ScenarioError("{0} media_type is not an allowlisted source type".format(field_name))
    candidate = _safe_relative_artifact_path(base, path_value)
    if candidate.suffix.lower() not in _SAFE_ARTIFACT_TYPES[media_type]:
        raise ScenarioError("{0} filename extension does not match its allowlisted media type".format(field_name))
    actual_hash = sha256_file(candidate)
    if not _hash_equal(expected_hash, actual_hash):
        raise ScenarioError("{0} SHA-256 does not match local artifact bytes".format(field_name))
    return {"path": str(candidate), "media_type": media_type, "sha256": actual_hash}


def _safe_relative_artifact_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ScenarioError("source artifact paths must be relative to the explicit source manifest")
    if candidate.suffix.lower() in _UNSAFE_SUFFIXES:
        raise ScenarioError("pickle/checkpoint-like source artifact paths are forbidden")
    resolved_base = base.resolve()
    resolved = (resolved_base / candidate).resolve()
    if resolved_base != resolved and resolved_base not in resolved.parents:
        raise ScenarioError("source artifact path escapes the source-manifest directory")
    if not resolved.is_file():
        raise ScenarioError("source artifact does not exist: {0}".format(candidate))
    return resolved


def _parse_bridge_state_json(path: Path) -> Tuple[float, ...]:
    """Parse only standard JSON state vectors and validate Bridge's 8-D shape."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScenarioError("source state must be valid UTF-8 JSON, never a pickle/object payload") from error
    if isinstance(value, Mapping):
        values = value.get("bridge_state", value.get("state", value.get("proprio")))
    else:
        values = value
    if not isinstance(values, list) or len(values) != 8:
        raise ScenarioError("source state JSON must contain exactly one finite 8-D Bridge state vector")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in values):
        raise ScenarioError("source state JSON Bridge vector must contain only finite numeric values")
    return tuple(float(item) for item in values)


def _timestamp(value: Any, field_name: str) -> Union[str, float, int]:
    """Accept an explicit UTC/text timestamp or a finite numeric source time."""

    if isinstance(value, str):
        if value.strip():
            return value
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return value
    raise ScenarioError("{0} must be a non-empty text timestamp or finite numeric time".format(field_name))


def _write_new_json(path: Path, payload: Mapping[str, Any], *, description: str) -> None:
    """Create an import output once so concurrent workers cannot race an overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ScenarioError("refusing to overwrite existing {0}: {1}".format(description, path)) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _require_string(value: Any, field_name: str) -> str:
    if not _nonempty_string(value):
        raise ScenarioError("{0} must be a non-empty string".format(field_name))
    return str(value)


def _require_nonempty_value(value: Any, field_name: str) -> None:
    if value is None or value == "" or value == [] or value == {}:
        raise ScenarioError("{0} is required".format(field_name))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_hash(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    digest = value[7:] if value.startswith("sha256:") else value
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest.lower())


def _require_hash(value: Any, field_name: str) -> str:
    if not _is_hash(value):
        raise ScenarioError("{0} must be a SHA-256 digest".format(field_name))
    return str(value)


def _hash_equal(first: Any, second: Any) -> bool:
    if not _is_hash(first) or not _is_hash(second):
        return False
    first_digest = str(first)[7:] if str(first).startswith("sha256:") else str(first)
    second_digest = str(second)[7:] if str(second).startswith("sha256:") else str(second)
    return first_digest.lower() == second_digest.lower()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI for offline scenario import, freeze, and readiness review."""

    parser = argparse.ArgumentParser(description="PLUMB safe scenario-manifest workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    import_command = commands.add_parser("import", help="verify local sources and emit a scenario proposal")
    import_command.add_argument("--scenario", required=True)
    import_command.add_argument("--sources", required=True)
    import_command.add_argument("--output", required=True)
    freeze_command = commands.add_parser("freeze", help="write a scenario selection exactly once")
    freeze_command.add_argument("--scenario", required=True)
    freeze_command.add_argument("--output", required=True)
    freeze_command.add_argument("--allow-incomplete-primary", action="store_true")
    report_command = commands.add_parser("readiness", help="report structural scenario blockers")
    report_command.add_argument("--scenario", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            imported = import_scenario_manifest(args.scenario, args.sources)
            target = Path(args.output)
            _write_new_json(target, imported, description="scenario import output")
            print(json.dumps({"output": str(target), "status": "imported_unfrozen"}, sort_keys=True))
            return 0
        manifest = read_json_manifest(args.scenario, description="scenario manifest")
        if args.command == "freeze":
            frozen = freeze_scenario_manifest(
                manifest, args.output, require_full_primary=not args.allow_incomplete_primary
            )
            print(json.dumps({"output": str(args.output), "sha256": frozen["sha256"], "status": "frozen"}, sort_keys=True))
            return 0
        print(json.dumps(scenario_readiness_report(manifest), ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except ScenarioError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COST_CONFIRMATION_PANEL",
    "DEVELOPMENT_PANEL",
    "HELDOUT_CALIBRATION_PANEL",
    "PANELS",
    "PRIMARY_PANEL",
    "PRIMARY_POLICY_IDS",
    "PRIMARY_POLICY_IDENTITIES",
    "PRIMARY_STARTS_PER_TASK",
    "SCHEMA_VERSION",
    "TASK_REGISTRY",
    "TASKS",
    "ScenarioError",
    "ScenarioSourceRecord",
    "TaskDefinition",
    "canonical_json",
    "freeze_scenario_manifest",
    "import_scenario_manifest",
    "load_explicit_source_manifest",
    "load_frozen_scenario_manifest",
    "main",
    "read_json_manifest",
    "scenario_manifest_hash",
    "scenario_readiness_report",
    "sha256_file",
    "validate_scenario_manifest",
]
