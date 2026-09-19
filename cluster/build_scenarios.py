#!/usr/bin/env python3
"""Acquire real starting states and assemble the Gate C scenario manifest.

Dry-run is the default for every subcommand; nothing is downloaded, converted,
or written without ``--execute``.  The pipeline is four explicit steps:

``plan``
    Emit the acquisition plan: source dataset and pinned revision per task, the
    language-instruction selection rule, cohort counts, the size ceiling, and
    the exact ``cluster/download_assets.py`` commands to run.  No network.
``convert-auto-eval``
    Convert one reviewed ``zhouzypaul/auto_eval`` pickle with restricted
    loading.  The allowlist, ``weights_only=True``, the refuse-to-overwrite
    rule, and the output hash follow ``cluster/convert_irasim_checkpoint.py``.
    ``robot_eval_logger`` and ``wandb`` globals are refused outright: a stub
    unpickler is not a security boundary, so there is no stub.
``select``
    Choose episodes and start frames from *local* source indices, enforcing the
    independence policy, and write per-start provenance records.
``assemble``
    Build the ``ScenarioManifest``, validate it against
    ``plumb.gates.ScenarioManifestValidator``, write ``scenarios.jsonl``, and
    emit the Gate C evidence bundle.

Drawer starts come from ``zhouzypaul/auto_eval`` (real WidowX video, 7-D
actions, 8-D proprio, classifier labels); it is used for drawer validation and
human reannotation, never as human-labelled five-task ground truth.  Basket,
sink, and cloth starts come from ``IPEC-COMMUNITY/bridge_orig_lerobot`` by
language-instruction match, with only manifest-selected episodes pulled.  That
dataset is not automatically the same scene distribution as AutoEval, so those
panels are matched-distribution comparisons with documented limitations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from plumb.scenarios import (  # noqa: E402  (path bootstrap must precede this import)
    AUTO_EVAL_DATASET,
    BRIDGE_LEROBOT_DATASET,
    BRIDGE_STATE_CONVENTION,
    CALIBRATION_COHORT,
    COST_CONFIRMATION_COHORT,
    DEVELOPMENT_COHORT,
    PRIMARY_COHORT,
    PRIMARY_STARTS_PER_TASK,
    TASK_SCENE_FAMILY,
    TASKS,
    IndependencePolicy,
    ScenarioError,
    SourceDataset,
    StartRecord,
    build_manifest,
    gate_c_evidence,
    independence_violations,
    verbatim_prompt,
)

DEFAULT_COHORT_STARTS: Mapping[str, int] = {
    PRIMARY_COHORT: PRIMARY_STARTS_PER_TASK,
    DEVELOPMENT_COHORT: 10,
    # 30 distinct calibration lineages per task supply the 20 development plus
    # 10 held-out calibration clips without any lineage crossing the split.
    CALIBRATION_COHORT: 30,
    COST_CONFIRMATION_COHORT: 10,
}

#: Cohorts filled in this fixed order so a shortfall is visible in the cohort
#: that ran out, not spread silently across all of them.
COHORT_FILL_ORDER: Tuple[str, ...] = (
    PRIMARY_COHORT,
    CALIBRATION_COHORT,
    DEVELOPMENT_COHORT,
    COST_CONFIRMATION_COHORT,
)

TASK_SOURCE_DATASET: Mapping[str, str] = {
    task: AUTO_EVAL_DATASET if TASK_SCENE_FAMILY[task] == "drawer" else BRIDGE_LEROBOT_DATASET
    for task in TASKS
}

AUTO_EVAL_ROLE = "scenario_and_label_source"
BRIDGE_ROLE = "scenario_source"

#: ``cluster.asset_plan`` owns the pinned revision, size ceiling, license status,
#: and allow patterns for both source datasets.  This driver reads them instead
#: of keeping a second copy that could drift.
ASSET_PLAN_NAMES: Mapping[str, str] = {
    AUTO_EVAL_DATASET: "auto-eval",
    BRIDGE_LEROBOT_DATASET: "bridge-orig-lerobot",
}

AUTO_EVAL_USE_LIMIT = (
    "Use zhouzypaul/auto_eval for drawer validation and human reannotation, not as human-labelled "
    "five-task ground truth."
)
BRIDGE_USE_LIMIT = (
    "IPEC-COMMUNITY/bridge_orig_lerobot is not automatically the same scene distribution as "
    "AutoEval; pull only manifest-selected episodes."
)

#: Pickle globals permitted when converting an ``auto_eval`` artifact.  Anything
#: outside this set, and every ``robot_eval_logger``/``wandb`` object, is
#: refused rather than stubbed.
APPROVED_AUTO_EVAL_GLOBALS = frozenset(
    {
        "builtins.bool",
        "builtins.bytes",
        "builtins.complex",
        "builtins.dict",
        "builtins.float",
        "builtins.int",
        "builtins.list",
        "builtins.set",
        "builtins.str",
        "builtins.tuple",
        "collections.OrderedDict",
        "collections.defaultdict",
        "numpy.core.multiarray._reconstruct",
        "numpy.dtype",
        "numpy.ndarray",
    }
)

#: Module prefixes that make an artifact non-convertible here.  These objects
#: execute third-party code on unpickling.
REFUSED_GLOBAL_MODULE_PREFIXES: Tuple[str, ...] = (
    "robot_eval_logger",
    "wandb",
    "torch.nn",
    "functools",
    "posix",
    "os",
    "subprocess",
    "builtins.eval",
    "builtins.exec",
    "builtins.getattr",
)

_SCRATCH_PREFIXES = ("/scratch/", "/global/scratch/")


class AcquisitionError(RuntimeError):
    """A scenario acquisition step cannot proceed honestly."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise AcquisitionError("expected a local source file: %s" % candidate)
    return {
        "path": str(candidate.resolve()),
        "bytes": candidate.stat().st_size,
        "sha256": sha256_file(candidate),
    }


def _is_immutable_revision(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _require_scratch_root(root: Path) -> Path:
    resolved = Path(root).resolve()
    if not str(resolved).startswith(_SCRATCH_PREFIXES) or resolved.name != "plumb":
        raise AcquisitionError(
            "large source pulls are permitted only in a cluster scratch/.../plumb directory, not %s"
            % resolved
        )
    return resolved


def _atomic_write_text(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + destination.name + "-", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(destination: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> Dict[str, Any]:
    target = Path(destination)
    if target.exists() and not overwrite:
        raise AcquisitionError("refusing to overwrite %s" % target)
    _atomic_write_text(target, json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return {"path": str(target), "sha256": sha256_file(target)}


def write_jsonl(destination: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool = False) -> Dict[str, Any]:
    target = Path(destination)
    if target.exists() and not overwrite:
        raise AcquisitionError("refusing to overwrite %s" % target)
    text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    _atomic_write_text(target, text)
    return {"path": str(target), "rows": len(rows), "sha256": sha256_file(target)}


# ---------------------------------------------------------------- plan step


def asset_plan_entry(dataset: str) -> Optional[Mapping[str, Any]]:
    """Return this dataset's asset-plan facts, or ``None`` when unavailable.

    ``cluster.asset_plan`` is owned elsewhere, so this reads it defensively: a
    missing module or renamed entry degrades to ``None`` rather than crashing an
    otherwise valid plan.
    """

    name = ASSET_PLAN_NAMES.get(dataset)
    if name is None:
        return None
    try:
        from cluster.asset_plan import get_plan  # noqa: PLC0415 - optional, evolving module

        entry = get_plan(name)
    except Exception:  # noqa: BLE001 - never let a planning read break the plan
        return None
    license_status = getattr(entry, "license_status", None)
    access = getattr(entry, "access", None)
    return {
        "asset_plan_name": name,
        "revision": getattr(entry, "revision", None),
        "max_gb": getattr(entry, "max_gb", None),
        "license": getattr(entry, "license", None),
        "license_status": getattr(license_status, "value", license_status),
        "access": getattr(access, "value", access),
        "blockers": list(getattr(entry, "blockers", ()) or ()),
        "traps": list(getattr(entry, "traps", ()) or ()),
    }


def cohort_counts(
    primary: int = PRIMARY_STARTS_PER_TASK,
    development: int = DEFAULT_COHORT_STARTS[DEVELOPMENT_COHORT],
    calibration: int = DEFAULT_COHORT_STARTS[CALIBRATION_COHORT],
    cost_confirmation: int = DEFAULT_COHORT_STARTS[COST_CONFIRMATION_COHORT],
) -> Dict[str, int]:
    counts = {
        PRIMARY_COHORT: int(primary),
        DEVELOPMENT_COHORT: int(development),
        CALIBRATION_COHORT: int(calibration),
        COST_CONFIRMATION_COHORT: int(cost_confirmation),
    }
    for name, value in counts.items():
        if value < 1:
            raise AcquisitionError("cohort %s needs at least one start" % name)
    return counts


def build_plan(
    *,
    manifest_id: str,
    revisions: Mapping[str, str],
    counts: Optional[Mapping[str, int]] = None,
    starts_per_episode: int = 1,
    min_source_frame_gap: int = IndependencePolicy().min_source_frame_gap,
    max_gb: float = 40.0,
    root: str = "/scratch/<user>/plumb",
    patterns: Optional[Mapping[str, str]] = None,
    tasks: Sequence[str] = TASKS,
) -> Dict[str, Any]:
    """Return the acquisition plan without contacting any network service."""

    resolved_counts = dict(counts or cohort_counts())
    if "<" not in root:
        # A concrete root is checked now, so the emitted download commands can
        # never point a multi-gigabyte dataset pull outside cluster scratch.
        _require_scratch_root(Path(root))
    selected = [task for task in tasks if task in TASK_SCENE_FAMILY]
    unknown = sorted(set(tasks) - set(selected))
    if unknown:
        raise AcquisitionError("unknown benchmark tasks: %s" % ", ".join(unknown))
    if starts_per_episode < 1:
        raise AcquisitionError("starts_per_episode must be at least 1")
    per_task_episodes = -(-sum(resolved_counts.values()) // starts_per_episode)
    plan_entries = {
        dataset: asset_plan_entry(dataset)
        for dataset in {TASK_SOURCE_DATASET[task] for task in selected}
    }
    task_plans = []
    for task in selected:
        dataset = TASK_SOURCE_DATASET[task]
        entry = plan_entries.get(dataset) or {}
        revision = revisions.get(dataset) or entry.get("revision")
        pattern = (patterns or {}).get(task, verbatim_prompt(task))
        task_plans.append(
            {
                "task": task,
                "instruction": verbatim_prompt(task),
                "scene_family": TASK_SCENE_FAMILY[task],
                "source_dataset": dataset,
                "source_revision": revision,
                "revision_source": "command_line"
                if revisions.get(dataset)
                else ("asset_plan" if entry.get("revision") else "unresolved"),
                "asset_plan": entry or None,
                "revision_immutable": _is_immutable_revision(revision),
                "selection_rule": "case-insensitive full match of the language instruction",
                "selection_pattern": pattern,
                "cohort_starts": dict(resolved_counts),
                "starts_per_episode": starts_per_episode,
                "distinct_episodes_required": per_task_episodes,
                "use_limit": AUTO_EVAL_USE_LIMIT if dataset == AUTO_EVAL_DATASET else BRIDGE_USE_LIMIT,
                "comparability": "matched_provenance"
                if TASK_SCENE_FAMILY[task] == "drawer"
                else "matched_distribution_with_limitations",
            }
        )
    download_commands = []
    for dataset in sorted({plan["source_dataset"] for plan in task_plans}):
        entry = plan_entries.get(dataset) or {}
        revision = revisions.get(dataset) or entry.get("revision")
        command = [
            "python",
            "-m",
            "cluster.download_assets",
            "--from-plan",
            str(entry.get("asset_plan_name") or ASSET_PLAN_NAMES.get(dataset, dataset)),
            "--root",
            root,
            "--max-gb",
            str(entry.get("max_gb") or max_gb),
            "--allow",
            "meta/*",
            "--execute",
        ]
        download_commands.append(
            {
                "dataset": dataset,
                "revision": revision,
                "asset_plan_name": entry.get("asset_plan_name"),
                "license_status": entry.get("license_status"),
                "blockers": entry.get("blockers", []),
                "traps": entry.get("traps", []),
                "command": command,
                "note": (
                    "Pull the metadata index first, run `select` to choose episodes, then re-run with "
                    "one --allow pattern per selected episode file. Never pull the whole dataset."
                ),
            }
        )
    unresolved = [
        "unpinned_source_revision:%s" % plan["source_dataset"]
        for plan in task_plans
        if not plan["revision_immutable"]
    ]
    for dataset, entry in sorted(plan_entries.items()):
        for blocker in (entry or {}).get("blockers", ()):  # asset-plan blockers are Gate C dependencies
            unresolved.append("%s: %s" % (dataset, blocker))
    return {
        "schema_version": 1,
        "kind": "plumb_scenario_acquisition_plan",
        "created_at": _utc_now(),
        "manifest_id": manifest_id,
        "root": root,
        "max_gb": max_gb,
        "independence_policy": IndependencePolicy(
            min_source_frame_gap=min_source_frame_gap,
            max_starts_per_episode=starts_per_episode,
        ).as_mapping(),
        "tasks": task_plans,
        "download_commands": download_commands,
        "safe_conversion": {
            "required_for": AUTO_EVAL_DATASET,
            "reason": (
                "Some auto_eval pickle objects refer to robot_eval_logger and wandb. A stub "
                "unpickler is not a security boundary."
            ),
            "command": [
                "python",
                "cluster/build_scenarios.py",
                "convert-auto-eval",
                "--pickle",
                "<episode.pkl>",
                "--output",
                "<episode.json>",
                "--source-sha256",
                "<sha256>",
                "--report",
                "<report.json>",
                "--execute",
            ],
            "approved_globals": sorted(APPROVED_AUTO_EVAL_GLOBALS),
            "refused_module_prefixes": list(REFUSED_GLOBAL_MODULE_PREFIXES),
        },
        "open_dependencies": sorted(set(unresolved)),
        "qualified": False,
    }


# ------------------------------------------------- safe auto_eval conversion


def classify_pickle_globals(found: Iterable[str]) -> Dict[str, List[str]]:
    """Split observed pickle globals into approved, refused, and unreviewed."""

    approved: List[str] = []
    refused: List[str] = []
    unreviewed: List[str] = []
    for name in sorted({str(value) for value in found}):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in REFUSED_GLOBAL_MODULE_PREFIXES):
            refused.append(name)
        elif name in APPROVED_AUTO_EVAL_GLOBALS:
            approved.append(name)
        else:
            unreviewed.append(name)
    return {"approved": approved, "refused": refused, "unreviewed": unreviewed}


def validate_pickle_globals(found: Iterable[str]) -> List[str]:
    """Raise unless every observed global is on the reviewed allowlist."""

    classified = classify_pickle_globals(found)
    if classified["refused"]:
        raise AcquisitionError(
            "refusing this artifact: it unpickles third-party objects that execute code (%s). "
            "A stub unpickler is not a security boundary; obtain a safe export instead."
            % ", ".join(classified["refused"])
        )
    if classified["unreviewed"]:
        raise AcquisitionError(
            "refusing this artifact: unreviewed pickle globals %s" % ", ".join(classified["unreviewed"])
        )
    return classified["approved"]


def _torch_module() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as error:  # pragma: no cover - cluster-only dependency
        raise AcquisitionError(
            "torch is required for restricted deserialization; run this in the isolated "
            "conversion environment"
        ) from error
    return torch


def convert_auto_eval_pickle(
    pickle_path: Path,
    output_path: Path,
    *,
    expected_sha256: str,
    execute: bool,
    torch_loader: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Convert one auto_eval pickle under restricted loading, or refuse.

    Mirrors ``cluster/convert_irasim_checkpoint.py``: verify the source digest,
    inspect the checkpoint's unsafe globals against an allowlist, load with
    ``weights_only=True`` inside ``safe_globals``, refuse to overwrite, and hash
    the output.  The report is returned for both success and failure.
    """

    source = Path(pickle_path)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "auto_eval_safe_conversion",
        "created_at": _utc_now(),
        "source": str(source),
        "expected_source_sha256": expected_sha256,
        "output": str(output_path),
        "status": "planned" if not execute else "failed",
        "isolation_requirement": (
            "Run in a disposable unprivileged environment with no credentials and no network."
        ),
    }
    if not source.is_file():
        report["error"] = {"type": "AcquisitionError", "message": "source pickle is absent"}
        report["status"] = "failed"
        return report
    actual = sha256_file(source)
    report["source_sha256"] = actual
    report["source_bytes"] = source.stat().st_size
    if actual.split(":")[-1].lower() != str(expected_sha256).split(":")[-1].lower():
        report["error"] = {"type": "AcquisitionError", "message": "source checksum mismatch"}
        report["status"] = "failed"
        return report
    if not execute:
        report["note"] = "Dry run: no deserialization was attempted. Re-run with --execute."
        return report
    try:
        torch = (torch_loader or _torch_module)()
        observed = torch.serialization.get_unsafe_globals_in_checkpoint(str(source))
        report["observed_unsafe_globals"] = sorted({str(value) for value in observed})
        report["classified_globals"] = classify_pickle_globals(observed)
        report["approved_unsafe_globals"] = validate_pickle_globals(observed)
        with torch.serialization.safe_globals(_safe_alias_list()):
            payload = torch.load(str(source), map_location="cpu", weights_only=True)
        record = _validate_auto_eval_payload(payload)
        destination = Path(output_path)
        if destination.exists():
            raise AcquisitionError("refusing to overwrite an existing converted artifact")
        _atomic_write_text(
            destination, json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        report.update(
            {
                "status": "completed",
                "output_sha256": sha256_file(destination),
                "frame_count": record["frame_count"],
                "action_dimension": record["action_dimension"],
                "proprio_dimension": record["proprio_dimension"],
                "classifier_label_present": record["classifier_label"] is not None,
            }
        )
    except Exception as error:  # noqa: BLE001 - the report must survive any failure
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    return report


def _safe_alias_list() -> List[Any]:
    """Allowlist only inert container and array types for ``safe_globals``."""

    import collections

    aliases: List[Any] = [
        collections.OrderedDict,
        collections.defaultdict,
        dict,
        list,
        tuple,
        set,
        int,
        float,
        bool,
        str,
        bytes,
        complex,
    ]
    try:
        import numpy  # type: ignore

        aliases.extend([numpy.ndarray, numpy.dtype, numpy.core.multiarray._reconstruct])
    except (ImportError, AttributeError):  # pragma: no cover - numpy is optional here
        pass
    return aliases


def _validate_auto_eval_payload(payload: Any) -> Dict[str, Any]:
    """Validate the converted episode schema; absent fields stay null."""

    if not isinstance(payload, Mapping):
        raise AcquisitionError("converted auto_eval payload must be a mapping")
    actions = payload.get("actions")
    proprio = payload.get("proprio", payload.get("observations/state"))
    frames = payload.get("images", payload.get("observations/images0"))
    action_rows = _numeric_rows(actions, "actions")
    proprio_rows = _numeric_rows(proprio, "proprio") if proprio is not None else []
    action_dimension = len(action_rows[0]) if action_rows else None
    proprio_dimension = len(proprio_rows[0]) if proprio_rows else None
    if action_dimension is not None and action_dimension != 7:
        raise AcquisitionError("auto_eval actions must be 7-D, found %d" % action_dimension)
    if proprio_dimension is not None and proprio_dimension != 8:
        raise AcquisitionError("auto_eval proprio must be 8-D, found %d" % proprio_dimension)
    frame_count = len(frames) if isinstance(frames, (list, tuple)) else None
    label = payload.get("classifier_label", payload.get("success"))
    if isinstance(label, (list, tuple)):
        label = list(label)
    return {
        "schema_version": 1,
        "kind": "auto_eval_converted_episode",
        "frame_count": frame_count,
        "action_count": len(action_rows) or None,
        "action_dimension": action_dimension,
        "proprio_dimension": proprio_dimension,
        "actions": action_rows,
        "proprio": proprio_rows,
        "classifier_label": label,
        "classifier_label_provenance": (
            "AutoEval's own classifier output on real WidowX drawer video. It is a third-party "
            "external label, not a PLUMB human annotation and not five-task ground truth."
        ),
    }


def _numeric_rows(value: Any, field_name: str) -> List[List[float]]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, (list, tuple)):
        raise AcquisitionError("%s must be a sequence of numeric rows" % field_name)
    rows: List[List[float]] = []
    for item in value:
        item_list = item.tolist() if callable(getattr(item, "tolist", None)) else item
        if not isinstance(item_list, (list, tuple)):
            raise AcquisitionError("%s rows must be numeric sequences" % field_name)
        row: List[float] = []
        for element in item_list:
            if isinstance(element, bool) or not isinstance(element, (int, float)):
                raise AcquisitionError("%s must contain finite numbers only" % field_name)
            row.append(float(element))
        rows.append(row)
    widths = {len(row) for row in rows}
    if len(widths) > 1:
        raise AcquisitionError("%s rows have inconsistent width" % field_name)
    return rows


# -------------------------------------------------------------- select step


def load_index_rows(path: Path) -> List[Dict[str, Any]]:
    """Read a local source index (JSONL or JSON list/object)."""

    source = Path(path)
    if not source.is_file():
        raise AcquisitionError("source index is absent: %s" % source)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as error:
                raise AcquisitionError("invalid JSONL on line %d of %s" % (number, source)) from error
            if not isinstance(parsed, Mapping):
                raise AcquisitionError("source index rows must be objects")
            rows.append(dict(parsed))
        return rows
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise AcquisitionError("invalid JSON in %s" % source) from error
    if isinstance(parsed, Mapping):
        for key in ("episodes", "rows", "data"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        raise AcquisitionError("%s must hold a list of episode records" % source)
    return [dict(row) for row in parsed if isinstance(row, Mapping)]


def _episode_instruction(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("instruction", "language_instruction", "task", "tasks", "language"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
            return value[0]
    return None


def matching_episodes(
    rows: Sequence[Mapping[str, Any]], pattern: str, *, regex: bool = False
) -> List[Dict[str, Any]]:
    """Return index rows whose language instruction matches the selection rule."""

    compiled = re.compile(pattern, re.IGNORECASE) if regex else None
    matched: List[Dict[str, Any]] = []
    for row in rows:
        instruction = _episode_instruction(row)
        if instruction is None:
            continue
        if compiled is not None:
            if compiled.search(instruction):
                matched.append(dict(row))
        elif instruction.strip().casefold() == pattern.strip().casefold():
            matched.append(dict(row))
    return matched


def _episode_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("episode_id", "episode_index", "episode", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return "episode_%06d" % value
    return "episode_row_%06d" % index


def select_starts(
    *,
    task: str,
    index_rows: Sequence[Mapping[str, Any]],
    source_dataset: str,
    source_revision: str,
    counts: Mapping[str, int],
    pattern: Optional[str] = None,
    regex: bool = False,
    starts_per_episode: int = 1,
    min_source_frame_gap: int = IndependencePolicy().min_source_frame_gap,
    camera_id: str = "unresolved_source_camera",
) -> Dict[str, Any]:
    """Assign matched source episodes to cohorts as independent start records.

    Returns a status/reason mapping: a shortfall is reported, never padded with
    a neighbouring frame or a repeated episode.
    """

    if task not in TASK_SCENE_FAMILY:
        raise AcquisitionError("unknown benchmark task %r" % (task,))
    selection_pattern = pattern or verbatim_prompt(task)
    matched = matching_episodes(index_rows, selection_pattern, regex=regex)
    scene_family = TASK_SCENE_FAMILY[task]
    required = sum(counts.get(cohort, 0) for cohort in COHORT_FILL_ORDER)
    available = len(matched) * max(1, starts_per_episode)
    records: List[StartRecord] = []
    provenance: List[Dict[str, Any]] = []
    cursor = 0
    shortfall: Dict[str, int] = {}
    for cohort in COHORT_FILL_ORDER:
        wanted = counts.get(cohort, 0)
        assigned = 0
        while assigned < wanted and cursor < len(matched) * max(1, starts_per_episode):
            episode_position, offset = divmod(cursor, max(1, starts_per_episode))
            row = matched[episode_position]
            episode_id = _episode_id(row, episode_position)
            frame_index = offset * max(min_source_frame_gap, 1)
            identity = "%s\x1f%s\x1f%s\x1f%d" % (source_dataset, source_revision, episode_id, frame_index)
            lineage = "%s@%s:%s:frame-%06d" % (source_dataset, source_revision, episode_id, frame_index)
            image_hash = row.get("image_sha256") or row.get("frame_sha256")
            state_hash = row.get("state_sha256")
            goal_hash = row.get("goal_reference_sha256")
            records.append(
                StartRecord(
                    start_id="start-%s-%s-%03d" % (task, cohort, assigned),
                    task=task,
                    cohort=cohort,
                    start_lineage_id=lineage,
                    source_dataset=source_dataset,
                    source_revision=source_revision,
                    episode_id=episode_id,
                    frame_id="frame-%06d" % frame_index,
                    frame_index=frame_index,
                    image_hash=image_hash or sha256_text("pending-image:" + identity),
                    state_hash=state_hash or sha256_text("pending-state:" + identity),
                    image_timestamp=str(row.get("image_timestamp") or row.get("timestamp") or "unresolved"),
                    state_timestamp=str(row.get("state_timestamp") or row.get("timestamp") or "unresolved"),
                    state_convention=str(row.get("state_convention") or BRIDGE_STATE_CONVENTION),
                    camera={
                        "camera_id": str(row.get("camera_id") or camera_id),
                        "crop": dict(row.get("crop") or {"kind": "identity", "width": 256, "height": 256}),
                        "calibration": dict(
                            row.get("calibration")
                            or {
                                "status": "unresolved",
                                "reason": (
                                    "No scene-specific metric reference recovered; the 1.5 cm drawer "
                                    "threshold is not estimated from pixels."
                                ),
                            }
                        ),
                    },
                    scene={
                        "scene_family": scene_family,
                        "objects": list(row.get("objects") or [scene_family]),
                        "layout": str(row.get("layout") or "source_episode_only"),
                        "reset_randomization": dict(
                            row.get("reset_randomization") or {"status": "unresolved"}
                        ),
                    },
                    initial_state_stratum=str(row.get("initial_state_stratum") or "unstratified"),
                    instruction=verbatim_prompt(task),
                    goal_reference_hashes={
                        "goal_image": goal_hash or sha256_text("pending-goal:" + identity)
                    },
                    goal_reference_provenance={
                        "source_uri": str(
                            row.get("source_uri")
                            or "hf://datasets/%s@%s/%s" % (source_dataset, source_revision, episode_id)
                        ),
                        "source_dataset": source_dataset,
                        "source_revision": source_revision,
                        "resolved": bool(goal_hash),
                    },
                    source_scene_family=scene_family,
                    notes=()
                    if image_hash and state_hash and goal_hash
                    else ("placeholder content hashes: rerun select against downloaded episode files",),
                )
            )
            provenance.append(
                {
                    "record_type": "start_provenance",
                    "start_id": records[-1].start_id,
                    "task": task,
                    "cohort": cohort,
                    "source_dataset": source_dataset,
                    "source_revision": source_revision,
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "selection_rule": "regex" if regex else "case_insensitive_full_instruction_match",
                    "selection_pattern": selection_pattern,
                    "matched_instruction": _episode_instruction(row),
                    "content_hashes_resolved": bool(image_hash and state_hash and goal_hash),
                    "source_index_row_keys": sorted(str(key) for key in row),
                }
            )
            assigned += 1
            cursor += 1
        if assigned < wanted:
            shortfall[cohort] = wanted - assigned
    violations = independence_violations(
        records,
        IndependencePolicy(
            min_source_frame_gap=min_source_frame_gap, max_starts_per_episode=starts_per_episode
        ),
    )
    status = "pass"
    reason: Optional[str] = None
    if violations:
        status = "fail"
        reason = "selected starts are not independent physical starts"
    elif shortfall:
        status = "insufficient_source_episodes"
        reason = (
            "matched %d episodes for %s, which supplies %d of %d required starts; do not pad this "
            "cell with neighbouring frames or another scene"
            % (len(matched), task, len(records), required)
        )
    return {
        "status": status,
        "reason": reason,
        "task": task,
        "selection_pattern": selection_pattern,
        "matched_episodes": len(matched),
        "available_starts": available,
        "required_starts": required,
        "selected_starts": len(records),
        "shortfall": shortfall,
        "independence_violations": list(violations),
        "starts": records,
        "provenance": provenance,
    }


# ------------------------------------------------------------ assemble step


def source_datasets(revisions: Mapping[str, str], tasks: Sequence[str] = TASKS) -> Tuple[SourceDataset, ...]:
    """Build the declared source records for the datasets these tasks need."""

    needed: Dict[str, List[str]] = {}
    for task in tasks:
        dataset = TASK_SOURCE_DATASET[task]
        needed.setdefault(dataset, [])
        family = TASK_SCENE_FAMILY[task]
        if family not in needed[dataset]:
            needed[dataset].append(family)
    records = []
    for dataset, families in sorted(needed.items()):
        revision = revisions.get(dataset)
        if not revision:
            raise AcquisitionError("no pinned revision supplied for %s" % dataset)
        records.append(
            SourceDataset(
                dataset_id=dataset,
                revision=revision,
                role=AUTO_EVAL_ROLE if dataset == AUTO_EVAL_DATASET else BRIDGE_ROLE,
                scene_families=tuple(families),
                selection_rule="case-insensitive full match of the verbatim language instruction",
                license="advertised Apache-2.0; verify the archive notices"
                if dataset == BRIDGE_LEROBOT_DATASET
                else "unresolved; verify dataset card and notices before redistribution",
                url="https://huggingface.co/datasets/%s" % dataset,
                notes=(AUTO_EVAL_USE_LIMIT,) if dataset == AUTO_EVAL_DATASET else (BRIDGE_USE_LIMIT,),
            )
        )
    return tuple(records)


def assemble(
    *,
    manifest_id: str,
    start_rows: Sequence[Mapping[str, Any]],
    revisions: Mapping[str, str],
    evidence_uris: Sequence[str],
    protocol_hash: Optional[str] = None,
    min_source_frame_gap: int = IndependencePolicy().min_source_frame_gap,
    starts_per_episode: int = 1,
) -> Dict[str, Any]:
    """Build and validate the manifest plus its Gate C evidence bundle."""

    records = [StartRecord.from_mapping(row) for row in start_rows]
    tasks = [task for task in TASKS if any(record.task == task for record in records)]
    if not tasks:
        raise AcquisitionError("no start records were supplied")
    manifest = build_manifest(
        manifest_id,
        records,
        source_datasets(revisions, tasks),
        independence_policy=IndependencePolicy(
            min_source_frame_gap=min_source_frame_gap, max_starts_per_episode=starts_per_episode
        ),
    )
    evidence = gate_c_evidence(
        manifest,
        evidence_uris=evidence_uris,
        protocol_hash=protocol_hash,
        evidence_kind="real_source_frames",
    )
    return {"manifest": manifest, "gate_c_evidence": evidence}


# ---------------------------------------------------------------------- CLI


def _parse_revisions(values: Sequence[str]) -> Dict[str, str]:
    revisions: Dict[str, str] = {}
    for item in values or ():
        if "=" not in item:
            raise AcquisitionError("--revision must be DATASET=COMMIT, got %r" % item)
        dataset, revision = item.split("=", 1)
        revisions[dataset.strip()] = revision.strip()
    return revisions


def _parse_patterns(values: Sequence[str]) -> Dict[str, str]:
    patterns: Dict[str, str] = {}
    for item in values or ():
        if "=" not in item:
            raise AcquisitionError("--pattern must be TASK=PATTERN, got %r" % item)
        task, pattern = item.split("=", 1)
        if task.strip() not in TASK_SCENE_FAMILY:
            raise AcquisitionError("unknown task in --pattern: %r" % task)
        patterns[task.strip()] = pattern
    return patterns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="Emit the acquisition plan (no network)")
    plan.add_argument("--manifest-id", required=True)
    plan.add_argument("--revision", action="append", default=[], metavar="DATASET=COMMIT")
    plan.add_argument("--pattern", action="append", default=[], metavar="TASK=PATTERN")
    plan.add_argument("--task", action="append", default=[], choices=list(TASKS))
    plan.add_argument("--primary-starts", type=int, default=PRIMARY_STARTS_PER_TASK)
    plan.add_argument("--development-starts", type=int, default=DEFAULT_COHORT_STARTS[DEVELOPMENT_COHORT])
    plan.add_argument("--calibration-starts", type=int, default=DEFAULT_COHORT_STARTS[CALIBRATION_COHORT])
    plan.add_argument(
        "--cost-confirmation-starts", type=int, default=DEFAULT_COHORT_STARTS[COST_CONFIRMATION_COHORT]
    )
    plan.add_argument("--starts-per-episode", type=int, default=1)
    plan.add_argument("--min-source-frame-gap", type=int, default=IndependencePolicy().min_source_frame_gap)
    plan.add_argument("--max-gb", type=float, default=40.0)
    plan.add_argument("--root", default="/scratch/<user>/plumb")
    plan.add_argument("--output")
    plan.add_argument("--execute", action="store_true", help="Write --output instead of printing only.")

    convert = commands.add_parser("convert-auto-eval", help="Restricted-load one auto_eval pickle")
    convert.add_argument("--pickle", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--source-sha256", required=True)
    convert.add_argument("--report", required=True)
    convert.add_argument("--execute", action="store_true")

    select = commands.add_parser("select", help="Choose independent starts from a local source index")
    select.add_argument("--task", required=True, choices=list(TASKS))
    select.add_argument("--index", required=True, help="Local source episode index (JSON or JSONL).")
    select.add_argument("--revision", action="append", default=[], metavar="DATASET=COMMIT", required=True)
    select.add_argument("--pattern")
    select.add_argument("--regex", action="store_true")
    select.add_argument("--primary-starts", type=int, default=PRIMARY_STARTS_PER_TASK)
    select.add_argument("--development-starts", type=int, default=DEFAULT_COHORT_STARTS[DEVELOPMENT_COHORT])
    select.add_argument("--calibration-starts", type=int, default=DEFAULT_COHORT_STARTS[CALIBRATION_COHORT])
    select.add_argument(
        "--cost-confirmation-starts", type=int, default=DEFAULT_COHORT_STARTS[COST_CONFIRMATION_COHORT]
    )
    select.add_argument("--starts-per-episode", type=int, default=1)
    select.add_argument("--min-source-frame-gap", type=int, default=IndependencePolicy().min_source_frame_gap)
    select.add_argument("--output", help="Start records JSONL.")
    select.add_argument("--provenance", help="Per-start provenance JSONL.")
    select.add_argument("--execute", action="store_true")

    assemble_command = commands.add_parser("assemble", help="Write scenarios.jsonl and Gate C evidence")
    assemble_command.add_argument("--manifest-id", required=True)
    assemble_command.add_argument("--starts", action="append", required=True, help="Start records JSONL.")
    assemble_command.add_argument("--revision", action="append", default=[], required=True)
    assemble_command.add_argument("--evidence-uri", action="append", default=[])
    assemble_command.add_argument("--protocol-hash")
    assemble_command.add_argument("--output", help="scenarios.jsonl destination.")
    assemble_command.add_argument("--gate-evidence", help="Gate C evidence bundle destination.")
    assemble_command.add_argument("--starts-per-episode", type=int, default=1)
    assemble_command.add_argument(
        "--min-source-frame-gap", type=int, default=IndependencePolicy().min_source_frame_gap
    )
    assemble_command.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a manifest that is missing whole task panels, marked incomplete.",
    )
    assemble_command.add_argument("--overwrite", action="store_true")
    assemble_command.add_argument("--execute", action="store_true")

    download = commands.add_parser(
        "download-commands", help="Print the exact pinned download commands for a plan"
    )
    download.add_argument("--plan", required=True)
    return parser


def run(argv: Optional[Sequence[str]] = None) -> Tuple[int, Dict[str, Any]]:
    """Execute one subcommand and return its exit code plus report."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_plan(
            manifest_id=args.manifest_id,
            revisions=_parse_revisions(args.revision),
            counts=cohort_counts(
                args.primary_starts,
                args.development_starts,
                args.calibration_starts,
                args.cost_confirmation_starts,
            ),
            starts_per_episode=args.starts_per_episode,
            min_source_frame_gap=args.min_source_frame_gap,
            max_gb=args.max_gb,
            root=args.root,
            patterns=_parse_patterns(args.pattern),
            tasks=args.task or TASKS,
        )
        if args.execute and args.output:
            plan["written"] = write_json(Path(args.output), plan, overwrite=True)
        elif args.output:
            plan["dry_run"] = "Pass --execute to write %s" % args.output
        return 0, plan

    if args.command == "convert-auto-eval":
        report = convert_auto_eval_pickle(
            Path(args.pickle),
            Path(args.output),
            expected_sha256=args.source_sha256,
            execute=args.execute,
        )
        write_json(Path(args.report), report, overwrite=True)
        return (0 if report["status"] in ("completed", "planned") else 2), report

    if args.command == "select":
        revisions = _parse_revisions(args.revision)
        dataset = TASK_SOURCE_DATASET[args.task]
        revision = revisions.get(dataset)
        if not revision:
            raise AcquisitionError("supply --revision %s=<commit> for task %s" % (dataset, args.task))
        index_record = file_record(Path(args.index))
        result = select_starts(
            task=args.task,
            index_rows=load_index_rows(Path(args.index)),
            source_dataset=dataset,
            source_revision=revision,
            counts=cohort_counts(
                args.primary_starts,
                args.development_starts,
                args.calibration_starts,
                args.cost_confirmation_starts,
            ),
            pattern=args.pattern,
            regex=args.regex,
            starts_per_episode=args.starts_per_episode,
            min_source_frame_gap=args.min_source_frame_gap,
        )
        report = {key: value for key, value in result.items() if key not in ("starts", "provenance")}
        report["source_index"] = index_record
        report["start_rows"] = [start.as_mapping() for start in result["starts"]]
        report["provenance_rows"] = result["provenance"]
        if args.execute:
            if args.output:
                report["written_starts"] = write_jsonl(
                    Path(args.output), report["start_rows"], overwrite=True
                )
            if args.provenance:
                report["written_provenance"] = write_jsonl(
                    Path(args.provenance), result["provenance"], overwrite=True
                )
        else:
            report["dry_run"] = "Pass --execute to write the start and provenance records."
        return (0 if result["status"] == "pass" else 2), report

    if args.command == "assemble":
        rows: List[Mapping[str, Any]] = []
        for path in args.starts:
            rows.extend(load_index_rows(Path(path)))
        built = assemble(
            manifest_id=args.manifest_id,
            start_rows=rows,
            revisions=_parse_revisions(args.revision),
            evidence_uris=args.evidence_uri,
            protocol_hash=args.protocol_hash,
            min_source_frame_gap=args.min_source_frame_gap,
            starts_per_episode=args.starts_per_episode,
        )
        manifest = built["manifest"]
        report: Dict[str, Any] = {
            "manifest_id": manifest.manifest_id,
            "scenario_manifest_sha256": manifest.sha256,
            "manifest_status": manifest.manifest_status(),
            "unacquired_task_panels": list(manifest.absent_tasks()),
            "validator_errors": list(manifest.validate()),
            "gate_c_evidence": built["gate_c_evidence"],
        }
        if args.execute:
            if args.output:
                report["written_manifest"] = manifest.write_jsonl(
                    Path(args.output),
                    overwrite=args.overwrite,
                    allow_incomplete=args.allow_incomplete,
                )
            if args.gate_evidence:
                report["written_evidence"] = write_json(
                    Path(args.gate_evidence), built["gate_c_evidence"], overwrite=True
                )
        else:
            report["dry_run"] = "Pass --execute to write scenarios.jsonl and the Gate C bundle."
        return (0 if report["gate_c_evidence"]["status"] != "fail" else 2), report

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    return 0, {"download_commands": plan.get("download_commands", [])}


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        code, report = run(argv)
    except (AcquisitionError, ScenarioError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
