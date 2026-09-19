"""Audit pinned LeRobot metadata without selecting starts or inventing labels.

Consumes the existing downloader's hash manifest and four metadata files.
No network, video decoding, model imports, primary-lock edits or qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping

from plumb.scenarios import TASKS, verbatim_prompt

DATASET = "IPEC-COMMUNITY/bridge_orig_lerobot"
METADATA_FILES = ("meta/info.json", "meta/stats.json", "meta/tasks.jsonl", "meta/episodes.jsonl")
# Discovery only: these are deliberately NOT benchmark eligibility rules.
DISCOVERY_PATTERNS = {
    "open_drawer": r"(?=.*\bopen\b)(?=.*\bdrawer\b)",
    "close_drawer": r"(?=.*\bclose\b)(?=.*\bdrawer\b)",
    "to_basket": r"\bbasket\b",
    "to_sink": r"\bsink\b",
    "fold_cloth": r"(?=.*\bfold\b)(?=.*\b(?:cloth|towel)\b)",
}


class MetadataAuditError(ValueError):
    pass


def _json(raw: bytes) -> Any:
    def reject(value: str) -> None:
        raise MetadataAuditError("non-finite JSON constant: " + value)
    return json.loads(raw, parse_constant=reject)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_bound(root: Path, relative: str, records: Mapping[str, Any]) -> bytes:
    path = root / relative
    if (root / Path(relative).parts[0]).is_symlink() or path.is_symlink():
        raise MetadataAuditError("metadata symlinks are not accepted: " + relative)
    if not path.is_file() or root.resolve() not in path.resolve().parents:
        raise MetadataAuditError("missing or escaping metadata: " + relative)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise MetadataAuditError("metadata exceeds 32 MiB bound: " + relative)
    raw = path.read_bytes()
    expected = records.get(relative)
    if not isinstance(expected, Mapping) or expected.get("bytes") != len(raw) or expected.get("sha256") != _sha(raw):
        raise MetadataAuditError("metadata does not match download manifest: " + relative)
    return raw


def _rows(raw: bytes, label: str) -> list:
    rows = [_json(line) for line in raw.splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise MetadataAuditError(label + " must contain object records")
    return rows


def _index(rows: list, key: str, label: str) -> dict:
    result = {}
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value in result:
            raise MetadataAuditError(label + " has duplicate/invalid " + key)
        result[value] = row
    return result


def audit(root: Path, manifest_path: Path, *, expected_revision: str) -> Dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        raise MetadataAuditError("expected revision must be a full immutable commit")
    if root.is_symlink() or manifest_path.is_symlink() or manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise MetadataAuditError("manifest/root symlink or oversized manifest")
    manifest_raw = manifest_path.read_bytes()
    manifest = _json(manifest_raw)
    if not isinstance(manifest, dict):
        raise MetadataAuditError("download manifest must be an object")
    if manifest.get("repo") != DATASET or manifest.get("repo_type") != "dataset" or manifest.get("revision") != expected_revision:
        raise MetadataAuditError("download manifest dataset/revision mismatch")
    record_rows = manifest.get("file_records", [])
    if not isinstance(record_rows, list) or not all(isinstance(row, dict) for row in record_rows):
        raise MetadataAuditError("invalid download manifest file records")
    records = {row.get("path"): row for row in record_rows}
    if len(records) != len(record_rows):
        raise MetadataAuditError("duplicate manifest file records")
    bound = {relative: _read_bound(root, relative, records) for relative in METADATA_FILES}
    info, stats = _json(bound["meta/info.json"]), _json(bound["meta/stats.json"])
    if not isinstance(info, dict) or not isinstance(stats, dict):
        raise MetadataAuditError("info and stats must be objects")
    tasks = _index(_rows(bound["meta/tasks.jsonl"], "tasks"), "task_index", "tasks")
    episodes = _index(_rows(bound["meta/episodes.jsonl"], "episodes"), "episode_index", "episodes")
    task_texts = set()
    for row in tasks.values():
        value = row.get("task")
        if not isinstance(value, str):
            raise MetadataAuditError("task table contains non-text instruction")
        task_texts.add(value)
    total_frames = 0
    instructions = {}
    for index, row in episodes.items():
        labels, length = row.get("tasks"), row.get("length")
        if not isinstance(labels, list) or not labels or not all(isinstance(v, str) and v in task_texts for v in labels):
            raise MetadataAuditError("episode instruction is absent from the task table")
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise MetadataAuditError("invalid episode length")
        total_frames += length
        instructions[index] = tuple(dict.fromkeys(labels))
    blockers = ["scene_reset_camera_and_object_parity_unreviewed", "no_goal_or_metric_calibration_references",
                "no_source_episode_state_or_video_files_examined", "no_human_annotations"]
    unlabeled = sum(not any(value.strip() for value in labels) for labels in instructions.values())
    if unlabeled:
        blockers.append("source_contains_unlabeled_episodes")
    for key, actual in (("total_episodes", len(episodes)), ("total_tasks", len(tasks)), ("total_frames", total_frames)):
        if info.get(key) != actual:
            blockers.append("metadata_count_mismatch:" + key)
    features = info.get("features", {})
    if not isinstance(features, dict) or not all(isinstance(v, dict) for v in features.values()):
        raise MetadataAuditError("features must map names to objects")
    if features.get("observation.state", {}).get("shape") != [8] or features.get("action", {}).get("shape") != [7]:
        blockers.append("state_or_action_dimension_mismatch")
    state_stats = stats.get("observation.state", {})
    if not isinstance(state_stats, dict):
        raise MetadataAuditError("state statistics must be an object")
    extrema = {}
    for key in ("min", "max"):
        values = state_stats.get(key)
        if not isinstance(values, list) or len(values) != 8 or not all(
            isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in values
        ):
            raise MetadataAuditError("state statistics require finite eight-dimensional extrema")
        extrema[key] = values
    if any(a > b for a, b in zip(extrema["min"], extrema["max"])):
        raise MetadataAuditError("state minimum exceeds maximum")
    # The current 0..0.39 type is explicitly an uncertified handoff convention.
    # Do not rescale, clamp, or replace it based only on aggregate statistics.
    gripper_range = [extrema["min"][7], extrema["max"][7]]
    if gripper_range[0] < 0 or gripper_range[1] > 0.39:
        blockers.append("source_gripper_state_range_conflicts_with_unverified_0_to_0_39_profile")
    coverage = {}
    for task in TASKS:
        prompt = verbatim_prompt(task)
        exact = [i for i, labels in instructions.items() if any(v.strip().casefold() == prompt.strip().casefold() for v in labels)]
        regex = re.compile(DISCOVERY_PATTERNS[task], re.IGNORECASE)
        candidates = [i for i, labels in instructions.items() if any(regex.search(v) for v in labels)]
        examples = Counter(v for i in candidates for v in instructions[i] if regex.search(v))
        coverage[task] = {"benchmark_prompt": prompt, "exact_instruction_episode_count": len(exact),
                          "exact_episode_ids": sorted(exact), "discovery_regex": DISCOVERY_PATTERNS[task],
                          "discovery_episode_count": len(candidates), "discovery_is_eligibility": False,
                          "discovery_instruction_examples": [{"instruction": value, "episodes": count}
                                                              for value, count in examples.most_common(15)]}
    return {"schema": "plumb-bridge-metadata-audit-v1", "status": "needs_review", "qualified": False,
            "dataset": DATASET, "revision": expected_revision, "download_manifest_sha256": _sha(manifest_raw),
            "input_files": {name: {"sha256": _sha(raw), "bytes": len(raw)} for name, raw in bound.items()},
            "source_counts": {"episodes": len(episodes), "tasks": len(tasks), "frames": total_frames},
            "unlabeled_episode_count": unlabeled,
            "declared_counts": {key: info.get(key) for key in ("total_episodes", "total_tasks", "total_frames", "total_videos")},
            "source_fps": info.get("fps"), "video_features": {key: value for key, value in features.items() if value.get("dtype") == "video"},
            "state_feature": features.get("observation.state"), "source_gripper_state_range": gripper_range,
            "gripper_conversion_applied": False, "selected_starts": [], "task_coverage": coverage,
            "blockers": blockers, "limitation": "Metadata language matches are not matched benchmark starts, successful demonstrations, or human labels."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.dataset_root, args.download_manifest, expected_revision=args.revision)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"output": str(args.output), "status": result["status"], "source_counts": result["source_counts"],
                          "exact_task_counts": {task: row["exact_instruction_episode_count"] for task, row in result["task_coverage"].items()},
                          "blockers": result["blockers"]}))
        return 0  # A successful audit is not a successful Gate C.
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error), "qualified": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
