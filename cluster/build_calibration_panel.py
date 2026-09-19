#!/usr/bin/env python3
"""Freeze the 150-clip calibration panel and run the judge over it in batch.

Dry-run is the default for every subcommand; nothing is written, staged, or sent
to a model without ``--execute``.  The pipeline is four ordered steps.

``select``
    Stratified deterministic selection of 150 clips (100 development, 50
    held-out; 20 and 10 per task) from a candidate pool, with source lineages
    disjoint across the split and disjoint from the primary and
    cost-confirmation cohorts.  Clip IDs and media references are rewritten to
    opaque values so the blinded annotation view cannot leak a policy from a
    file path.  ``plumb.calibration.validate_manifest`` enforces the counts.
``freeze``
    Pin everything knowable before the judge runs: the calibration manifest
    hash, the held-out video hashes, the task registry, the model identity, the
    protocol ID, and the producer.  Refuses to overwrite.
``judge``
    Run the frozen primary protocol over the manifest: five samples, T=0.7,
    top_p=1.0, a 512-token cap, logged seeds, and the one bounded retry the
    judge itself applies for schema/transport failure only.  Sixteen frames are
    sampled uniformly from the original start through the final control tick,
    including both endpoints; a clip that cannot supply them is rejected, never
    padded.  Output is the raw per-clip report that
    ``plumb.calibration.validate_primary_judge_evidence`` re-aggregates.
``bind``
    Complete the frozen evidence manifest with the per-clip artifact hashes the
    run produced and emit the Gate D evidence bundle.  Pixel-hash binding is
    post-hoc consistency checking, not preregistration: the judge computes frame
    hashes at call time, so they cannot be pinned in advance.  This is stated in
    the emitted record rather than glossed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from cluster.policy_smoke import uniform_video_indices  # noqa: E402
from plumb.calibration import (  # noqa: E402
    DEVELOPMENT_SPLIT,
    FULL_PLAN_COUNTS,
    HELDOUT_SPLIT,
    TASKS,
    CalibrationError,
    ClipManifestRow,
    calibration_manifest_hash,
    validate_manifest,
)
from plumb.policies.tasks import TASK_REGISTRY_HASH, TASK_REGISTRY_ID  # noqa: E402

QWEN_JUDGE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SELECTION_SALT = "plumb-calibration-panel-selection-v1"
CLIP_ID_SALT = "plumb-calibration-clip-id-v1"
JUDGE_SEED_SALT = "plumb-calibration-judge-seed-v1"
BLINDED_MEDIA_PREFIX = "artifact://calibration"

#: Frozen primary sampling operating point, echoed here so a driver change is
#: visible in a diff rather than only inside the judge.
PRIMARY_SAMPLING = {
    "sample_count": 5,
    "quorum": 3,
    "temperature": 0.7,
    "top_p": 1.0,
    "max_new_tokens": 512,
    "retries_per_sample": 1,
}

APPARENT_OUTCOMES: Tuple[str, ...] = ("apparent_success", "apparent_failure", "unknown")
VIDEO_SUFFIXES = frozenset((".mp4", ".mov", ".mkv", ".avi", ".webm"))


class PanelError(RuntimeError):
    """A calibration panel step cannot proceed honestly."""


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


def _digest_int(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


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
        raise PanelError("refusing to overwrite %s" % target)
    _atomic_write_text(target, json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return {"path": str(target), "sha256": sha256_file(target)}


def write_jsonl(destination: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool = False) -> Dict[str, Any]:
    target = Path(destination)
    if target.exists() and not overwrite:
        raise PanelError("refusing to overwrite %s" % target)
    _atomic_write_text(
        target,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
    )
    return {"path": str(target), "rows": len(rows), "sha256": sha256_file(target)}


def read_rows(path: Path) -> List[Dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise PanelError("input does not exist: %s" % source)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as error:
                raise PanelError("invalid JSONL on line %d of %s" % (number, source)) from error
            if not isinstance(parsed, Mapping):
                raise PanelError("JSONL rows must be objects")
            rows.append(dict(parsed))
        return rows
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise PanelError("invalid JSON in %s" % source) from error
    if isinstance(parsed, Mapping):
        for key in ("clips", "rows", "candidates", "reports"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        raise PanelError("%s must hold a list of objects" % source)
    return [dict(row) for row in parsed if isinstance(row, Mapping)]


# -------------------------------------------------------------- select step


def blinded_clip_id(source_lineage_id: str, episode_ref: str) -> str:
    """Opaque clip ID derived from the source identity, not from a policy name."""

    digest = hashlib.sha256(
        ("%s\x1f%s\x1f%s" % (CLIP_ID_SALT, source_lineage_id, episode_ref)).encode("utf-8")
    ).hexdigest()
    return "clip-" + digest[:12]


def _candidate_key(row: Mapping[str, Any], index: int) -> Tuple[str, str, str, bool, str]:
    """Stratum key: task, policy, apparent outcome, defect flag, lineage."""

    task = str(row.get("task", ""))
    policy = str(row.get("policy", "unknown_policy"))
    outcome = str(row.get("apparent_outcome", "unknown"))
    if outcome not in APPARENT_OUTCOMES:
        raise PanelError(
            "candidate %d has apparent_outcome %r; use one of %s"
            % (index, outcome, ", ".join(APPARENT_OUTCOMES))
        )
    defect = bool(row.get("generation_defect", False))
    lineage = str(row.get("source_lineage_id", ""))
    if not task or task not in TASKS:
        raise PanelError("candidate %d has task %r, which is not one of the five tasks" % (index, task))
    if not lineage:
        raise PanelError("candidate %d has no source_lineage_id" % index)
    return (task, policy, outcome, defect, lineage)


def select_panel(
    pool: Sequence[Mapping[str, Any]],
    *,
    excluded_lineages: Iterable[str] = (),
    counts: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Deterministically select the stratified 150-clip panel.

    Selection is a pure function of the candidate pool: every ordering comes
    from a salted SHA-256, so two operators on two machines freeze the same
    panel.  A stratum cell that cannot be filled is reported as a shortfall; no
    clip and no lineage is ever reused to reach a count.
    """

    plan = dict(counts or FULL_PLAN_COUNTS)
    excluded = {str(value) for value in excluded_lineages}
    by_task: Dict[str, List[Dict[str, Any]]] = {task: [] for task in TASKS}
    seen_lineages: Dict[str, int] = {}
    rejected: List[Dict[str, Any]] = []
    for index, row in enumerate(pool):
        task, policy, outcome, defect, lineage = _candidate_key(row, index)
        if lineage in excluded:
            rejected.append({"index": index, "reason": "lineage_reserved_for_another_cohort", "lineage": lineage})
            continue
        if lineage in seen_lineages:
            rejected.append({"index": index, "reason": "duplicate_source_lineage_in_pool", "lineage": lineage})
            continue
        media_ref = row.get("media_ref") or row.get("video_path")
        if not isinstance(media_ref, str) or not media_ref.strip():
            raise PanelError("candidate %d has no media_ref" % index)
        seen_lineages[lineage] = index
        by_task[task].append(
            {
                "source_lineage_id": lineage,
                "source_media_ref": media_ref,
                "task": task,
                "stratum": {
                    "policy": policy,
                    "apparent_outcome": outcome,
                    "generation_defect": defect,
                },
                "order_key": _digest_int(SELECTION_SALT, task, lineage),
                "video_sha256": row.get("video_sha256"),
            }
        )

    selected: List[Dict[str, Any]] = []
    shortfall: Dict[str, int] = {}
    strata_coverage: Dict[str, Dict[str, Any]] = {}
    for task in TASKS:
        candidates = sorted(by_task[task], key=lambda item: (item["order_key"], item["source_lineage_id"]))
        buckets: Dict[Tuple[str, str, bool], List[Dict[str, Any]]] = {}
        for candidate in candidates:
            stratum = candidate["stratum"]
            buckets.setdefault(
                (stratum["policy"], stratum["apparent_outcome"], stratum["generation_defect"]), []
            ).append(candidate)
        ordered_cells = sorted(buckets, key=lambda cell: (_digest_int(SELECTION_SALT, task, *cell), cell))
        # Round-robin across strata so the panel covers policies, apparent
        # outcomes, and defects instead of filling from one dominant cell.
        drawn: List[Dict[str, Any]] = []
        cursor = {cell: 0 for cell in ordered_cells}
        while len(drawn) < sum(plan.values()):
            progressed = False
            for cell in ordered_cells:
                if len(drawn) >= sum(plan.values()):
                    break
                position = cursor[cell]
                if position < len(buckets[cell]):
                    drawn.append(buckets[cell][position])
                    cursor[cell] = position + 1
                    progressed = True
            if not progressed:
                break
        strata_coverage[task] = {
            "distinct_strata": len(ordered_cells),
            "strata": [
                {
                    "policy": cell[0],
                    "apparent_outcome": cell[1],
                    "generation_defect": cell[2],
                    "available": len(buckets[cell]),
                    "drawn": cursor[cell],
                }
                for cell in ordered_cells
            ],
            "available_candidates": len(candidates),
        }
        cursor_index = 0
        for split in (DEVELOPMENT_SPLIT, HELDOUT_SPLIT):
            wanted = plan[split]
            taken = drawn[cursor_index : cursor_index + wanted]
            cursor_index += len(taken)
            if len(taken) < wanted:
                shortfall["%s/%s" % (task, split)] = wanted - len(taken)
            for candidate in taken:
                clip_id = blinded_clip_id(candidate["source_lineage_id"], candidate["source_media_ref"])
                selected.append(
                    {
                        "clip_id": clip_id,
                        "media_ref": "%s/%s%s"
                        % (
                            BLINDED_MEDIA_PREFIX,
                            clip_id,
                            Path(candidate["source_media_ref"]).suffix.lower() or ".mp4",
                        ),
                        "task": task,
                        "split": split,
                        "source_lineage_id": candidate["source_lineage_id"],
                        "source_media_ref": candidate["source_media_ref"],
                        "stratum": candidate["stratum"],
                        "source_video_sha256": candidate["video_sha256"],
                    }
                )
    clip_ids = [row["clip_id"] for row in selected]
    if len(set(clip_ids)) != len(clip_ids):
        raise PanelError("blinded clip IDs collided; two candidates share a source identity")
    manifest_rows = [
        {
            "clip_id": row["clip_id"],
            "media_ref": row["media_ref"],
            "task": row["task"],
            "split": row["split"],
            "source_lineage_id": row["source_lineage_id"],
        }
        for row in selected
    ]
    status = "pass"
    reason: Optional[str] = None
    validated: Tuple[ClipManifestRow, ...] = ()
    manifest_hash: Optional[str] = None
    if shortfall:
        status = "insufficient_candidate_pool"
        reason = (
            "the candidate pool cannot fill %d stratum cells; generate more clips rather than "
            "reusing a lineage" % len(shortfall)
        )
    else:
        try:
            validated = validate_manifest(manifest_rows, require_full_plan=plan == dict(FULL_PLAN_COUNTS))
            manifest_hash = calibration_manifest_hash(validated)
        except CalibrationError as error:
            status = "fail"
            reason = str(error)
    return {
        "status": status,
        "reason": reason,
        "selection_salt": SELECTION_SALT,
        "clip_id_salt": CLIP_ID_SALT,
        "plan": plan,
        "selected_clips": len(selected),
        "shortfall": shortfall,
        "rejected_candidates": rejected,
        "strata_coverage": strata_coverage,
        "manifest_rows": manifest_rows,
        "selection_records": selected,
        "calibration_manifest_hash": manifest_hash,
        "pool_size": len(pool),
    }


def stage_media(
    selection_records: Sequence[Mapping[str, Any]],
    media_root: Path,
    *,
    execute: bool,
    link: bool = True,
) -> Dict[str, Any]:
    """Place each selected clip at its blinded path and record both hashes.

    The blinded path is what an annotator's browser requests, so the real
    generated path (which can contain a policy directory) never reaches the
    view.  Hashes before and after staging must match.
    """

    root = Path(media_root)
    records: List[Dict[str, Any]] = []
    problems: List[Dict[str, Any]] = []
    for record in selection_records:
        source = Path(str(record["source_media_ref"]))
        target = root / Path(str(record["media_ref"])).name
        entry: Dict[str, Any] = {
            "clip_id": record["clip_id"],
            "blinded_path": str(target),
            "source_path": str(source),
            "staged": False,
        }
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            problems.append({"clip_id": record["clip_id"], "reason": "source_is_not_a_video"})
            records.append(entry)
            continue
        if not source.is_file():
            problems.append({"clip_id": record["clip_id"], "reason": "source_media_absent"})
            records.append(entry)
            continue
        entry["source_sha256"] = sha256_file(source)
        declared = record.get("source_video_sha256")
        if declared and str(declared).split(":")[-1].lower() != entry["source_sha256"].split(":")[-1].lower():
            problems.append({"clip_id": record["clip_id"], "reason": "declared_video_sha256_mismatch"})
            records.append(entry)
            continue
        if not execute:
            records.append(entry)
            continue
        root.mkdir(parents=True, exist_ok=True)
        if target.exists():
            entry["staged"] = True
            entry["blinded_sha256"] = sha256_file(target)
            if entry["blinded_sha256"] != entry["source_sha256"]:
                problems.append({"clip_id": record["clip_id"], "reason": "existing_blinded_media_differs"})
            records.append(entry)
            continue
        if link:
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        else:
            shutil.copy2(source, target)
        entry["staged"] = True
        entry["blinded_sha256"] = sha256_file(target)
        if entry["blinded_sha256"] != entry["source_sha256"]:
            problems.append({"clip_id": record["clip_id"], "reason": "staged_media_hash_mismatch"})
        records.append(entry)
    return {
        "status": "pass" if not problems else "fail",
        "media_root": str(root),
        "records": records,
        "problems": problems,
        "staged": sum(1 for item in records if item["staged"]),
    }


# --------------------------------------------------------------- freeze step


def judge_source_hash() -> str:
    """SHA-256 of the local judge module bytes; a snapshot, not an attestation."""

    module_path = _REPOSITORY_ROOT / "plumb" / "policies" / "judge.py"
    return sha256_file(module_path)


def freeze_evidence(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    protocol_id: str,
    model_revision: str,
    processor_revision: str,
    transformers_version: str,
    runtime_lock_id: str,
    asset_manifest_id: str,
    media_root: Path,
    producer_version: str = "plumb-judge-report-v1",
) -> Dict[str, Any]:
    """Pin everything knowable before the judge runs over the held-out clips."""

    clips = validate_manifest(manifest_rows)
    heldout = [clip for clip in clips if clip.split == HELDOUT_SPLIT]
    video_hashes: Dict[str, str] = {}
    missing: List[str] = []
    for clip in heldout:
        path = Path(media_root) / Path(clip.media_ref).name
        if not path.is_file():
            missing.append(clip.clip_id)
            continue
        video_hashes[clip.clip_id] = sha256_file(path)
    status = "pass" if not missing else "blocked"
    return {
        "schema_version": 1,
        "kind": "plumb_frozen_judge_evidence_prerun",
        "status": status,
        "created_at": _utc_now(),
        "reason": None
        if not missing
        else "held-out media absent for %d clips; stage every clip before freezing" % len(missing),
        "missing_heldout_media": missing,
        "calibration_manifest_hash": calibration_manifest_hash(clips),
        "heldout_video_hashes": video_hashes,
        "task_registry_id": TASK_REGISTRY_ID,
        "task_registry_hash": TASK_REGISTRY_HASH,
        "model_id": QWEN_JUDGE_MODEL_ID,
        "model_revision": model_revision,
        "processor_revision": processor_revision,
        "transformers_version": transformers_version,
        "runtime_lock_id": runtime_lock_id,
        "asset_manifest_id": asset_manifest_id,
        "protocol_id": protocol_id,
        "trusted_producer": {
            "name": "plumb.policies.judge.QwenRubricJudge",
            "version": producer_version,
            "source_hash": judge_source_hash(),
        },
        "sampling": dict(PRIMARY_SAMPLING),
        "heldout_artifact_hashes": {},
        "artifact_hash_note": (
            "heldout_artifact_hashes stays empty here. The judge computes frame pixel hashes at call "
            "time, so they cannot be preregistered; `bind` fills them from the produced reports as "
            "post-hoc consistency checking, not preregistration."
        ),
        "qualified": False,
    }


# ---------------------------------------------------------------- judge step


def judge_seeds(base_seed: int, clip_id: str, count: int = 5) -> List[int]:
    """Five distinct logged seeds derived deterministically from clip identity."""

    seeds: List[int] = []
    counter = 0
    while len(seeds) < count:
        value = _digest_int(JUDGE_SEED_SALT, base_seed, clip_id, counter) & 0x7FFFFFFF
        if value not in seeds:
            seeds.append(value)
        counter += 1
        if counter > 1000:  # pragma: no cover - collision-bounded by construction
            raise PanelError("unable to derive %d distinct seeds for %s" % (count, clip_id))
    return seeds


def _load_video_frames(path: Path) -> Tuple[Any, ...]:
    if Path(path).suffix.lower() not in VIDEO_SUFFIXES:
        raise PanelError("judge input must be a generated video, not %s" % path)
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError as error:  # pragma: no cover - cluster-only dependency
        raise PanelError("imageio is required to decode the generated clips") from error
    frames = tuple(iio.imiter(path))
    if not frames:
        raise PanelError("generated clip has no decodable frames: %s" % path)
    return frames


def _load_image(path: Path) -> Any:
    try:
        from PIL import Image  # type: ignore
    except ImportError as error:  # pragma: no cover - cluster-only dependency
        raise PanelError("Pillow is required to load a goal/reference image") from error
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _judge_dependencies() -> Tuple[Any, Any, Any, Any, Any]:
    module = importlib.import_module("plumb.policies.judge")
    return (
        module.QwenJudgeProfile,
        module.QwenRubricJudge,
        module.JudgeRequest,
        module.ReferenceImage,
        module.JudgeInputProvenance,
    )


def run_batch_judge(
    manifest_rows: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
    *,
    media_root: Path,
    references: Mapping[str, Path],
    model_path: Path,
    base_seed: int,
    splits: Sequence[str] = (HELDOUT_SPLIT,),
    execute: bool,
    existing_clip_ids: Iterable[str] = (),
    video_loader: Callable[[Path], Tuple[Any, ...]] = _load_video_frames,
    image_loader: Callable[[Path], Any] = _load_image,
    dependency_loader: Callable[[], Tuple[Any, Any, Any, Any, Any]] = _judge_dependencies,
    judge_factory: Optional[Callable[[Any, Any], Any]] = None,
    fps: float = 5.0,
) -> Dict[str, Any]:
    """Run the frozen primary judge protocol over every selected clip.

    The per-clip plan is built and checked first, so a dry run reports exactly
    which clips would be scored, with which seeds, and against which frozen
    video hash.  Only ``--execute`` calls the model.
    """

    clips = validate_manifest(manifest_rows)
    wanted_splits = set(splits)
    targets = [clip for clip in clips if clip.split in wanted_splits]
    if not targets:
        raise PanelError("no clips in the requested splits %s" % ", ".join(sorted(wanted_splits)))
    done = {str(value) for value in existing_clip_ids}
    missing_reference = sorted(
        {clip.task for clip in targets if clip.task not in references}
    )
    plan: List[Dict[str, Any]] = []
    for clip in targets:
        video_path = Path(media_root) / Path(clip.media_ref).name
        entry = {
            "clip_id": clip.clip_id,
            "task": clip.task,
            "split": clip.split,
            "video_path": str(video_path),
            "video_present": video_path.is_file(),
            "frozen_video_hash": frozen.get("heldout_video_hashes", {}).get(clip.clip_id),
            "seeds": judge_seeds(base_seed, clip.clip_id),
            "already_scored": clip.clip_id in done,
            "reference": str(references.get(clip.task)) if clip.task in references else None,
        }
        plan.append(entry)
    blockers: List[str] = []
    if missing_reference:
        blockers.append("missing_goal_reference_for_tasks:%s" % ",".join(missing_reference))
    absent = [item["clip_id"] for item in plan if not item["video_present"]]
    if absent:
        blockers.append("missing_media_for_%d_clips" % len(absent))
    report: Dict[str, Any] = {
        "status": "planned",
        "created_at": _utc_now(),
        "sampling": dict(PRIMARY_SAMPLING),
        "frame_sampling": (
            "16 frames sampled uniformly from the original start through the exact final control "
            "tick, both endpoints included; a shorter clip is rejected, never padded."
        ),
        "base_seed": base_seed,
        "splits": sorted(wanted_splits),
        "clip_plan": plan,
        "blockers": blockers,
        "reports": [],
        "failures": [],
        "qualified": False,
    }
    if not execute:
        report["dry_run"] = "Pass --execute to call the judge."
        return report
    if blockers:
        report["status"] = "blocked"
        return report

    profile_cls, judge_cls, request_cls, reference_cls, provenance_cls = dependency_loader()
    profile = profile_cls(
        profile_id="qwen-calibration-panel-primary",
        local_model_path=str(model_path),
        model_revision=frozen["model_revision"],
        processor_revision=frozen["processor_revision"],
        transformers_version=frozen["transformers_version"],
        runtime_lock_id=frozen.get("runtime_lock_id"),
        runtime_lock_sha256=frozen.get("runtime_lock_sha256"),
        asset_manifest_id=frozen.get("asset_manifest_id"),
        asset_manifest_sha256=frozen.get("asset_manifest_sha256"),
    )
    judge = judge_factory(profile, judge_cls) if judge_factory is not None else judge_cls(profile)
    reference_cache: Dict[str, Any] = {}
    for entry in plan:
        if entry["already_scored"]:
            continue
        started = time.perf_counter()
        try:
            video_path = Path(entry["video_path"])
            video_hash = sha256_file(video_path)
            frozen_hash = entry["frozen_video_hash"]
            if frozen_hash and frozen_hash.split(":")[-1].lower() != video_hash.split(":")[-1].lower():
                raise PanelError("clip media does not match the frozen held-out video hash")
            frames = video_loader(video_path)
            indices = uniform_video_indices(len(frames), 16)
            reference_path = Path(str(entry["reference"]))
            if entry["task"] not in reference_cache:
                reference_cache[entry["task"]] = (
                    image_loader(reference_path),
                    sha256_file(reference_path),
                )
            reference_image, reference_hash = reference_cache[entry["task"]]
            request = request_cls(
                frames=tuple(frames[index] for index in indices),
                frame_timestamps=tuple(float(index) / float(fps) for index in indices),
                task_id=entry["task"],
                reference_images=(
                    reference_cls(
                        image=reference_image,
                        source_uri="artifact://goal/%s%s" % (entry["task"], reference_path.suffix.lower()),
                        sha256=reference_hash.split(":")[-1],
                    ),
                ),
                provenance=provenance_cls(
                    clip_id=entry["clip_id"],
                    video_sha256=video_hash,
                    protocol_id=frozen["protocol_id"],
                    calibration_manifest_hash=frozen["calibration_manifest_hash"],
                ),
            )
            result = judge.evaluate(request, seeds=entry["seeds"])
            payload = dict(result.as_dict())
            payload["clip_id"] = entry["clip_id"]
            payload["selected_frame_indices"] = list(indices)
            payload["source_video_frame_count"] = len(frames)
            payload["driver_wall_seconds"] = time.perf_counter() - started
            report["reports"].append(payload)
        except Exception as error:  # noqa: BLE001 - one clip must not lose the batch
            report["failures"].append(
                {
                    "clip_id": entry["clip_id"],
                    "task": entry["task"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    report["status"] = "completed" if not report["failures"] else "partial"
    report["scored_clips"] = len(report["reports"])
    report["failed_clips"] = len(report["failures"])
    return report


# ----------------------------------------------------------------- bind step


def bind_evidence(
    frozen: Mapping[str, Any],
    judge_reports: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Complete the frozen evidence manifest from the produced judge reports."""

    clips = validate_manifest(manifest_rows)
    heldout = {clip.clip_id for clip in clips if clip.split == HELDOUT_SPLIT}
    artifact_hashes: Dict[str, Dict[str, str]] = {}
    problems: List[Dict[str, Any]] = []
    for report in judge_reports:
        provenance = report.get("provenance")
        if not isinstance(provenance, Mapping):
            problems.append({"reason": "judge_report_has_no_provenance"})
            continue
        clip_id = provenance.get("clip_id")
        if clip_id not in heldout:
            problems.append({"clip_id": clip_id, "reason": "report_clip_is_not_in_the_heldout_split"})
            continue
        hashes = provenance.get("artifact_hashes")
        if not isinstance(hashes, Mapping) or not hashes:
            problems.append({"clip_id": clip_id, "reason": "report_has_no_artifact_hashes"})
            continue
        unresolved = sorted(name for name, value in hashes.items() if not isinstance(value, str) or not value)
        if unresolved:
            problems.append(
                {"clip_id": clip_id, "reason": "null_artifact_hashes", "artifacts": unresolved}
            )
            continue
        artifact_hashes[str(clip_id)] = {str(name): str(value) for name, value in hashes.items()}
    uncovered = sorted(heldout - set(artifact_hashes))
    if uncovered:
        problems.append({"reason": "heldout_clips_without_bound_artifact_hashes", "clip_ids": uncovered})
    complete = dict(frozen)
    complete.update(
        {
            "kind": "plumb_frozen_judge_evidence",
            "status": "pass" if not problems else "blocked",
            "bound_at": _utc_now(),
            "heldout_artifact_hashes": artifact_hashes,
            "binding_problems": problems,
            "binding_semantics": (
                "Immutable identifiers and a declared producer only. This is not a cryptographic "
                "attestation that a model ran, and it is not a calibration result."
            ),
        }
    )
    return complete


def gate_d_evidence_bundle(
    frozen: Mapping[str, Any],
    judge_run: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    evidence_uris: Sequence[str],
    protocol_hash: Optional[str] = None,
    calibration_class: Optional[str] = None,
    human_annotation_present: bool = False,
) -> Dict[str, Any]:
    """Shape a Gate D record for ``plumb.gates.GateRecord``.

    The status is never ``pass`` here: this bundle proves the judge ran over a
    frozen panel, and Gate D additionally needs two blinded human annotators and
    the numerical tolerances met.  ``plumb.annotation`` makes that decision.
    """

    clips = validate_manifest(manifest_rows)
    heldout = [clip for clip in clips if clip.split == HELDOUT_SPLIT]
    reasons: List[str] = []
    if not human_annotation_present:
        reasons.append("open dependency: human_annotation (two blinded human annotators)")
    if frozen.get("status") not in ("pass", None):
        reasons.append("frozen judge evidence is %s" % frozen.get("status"))
    if judge_run.get("status") not in ("completed",):
        reasons.append("judge batch status is %s" % judge_run.get("status"))
    if judge_run.get("failed_clips"):
        reasons.append("%d clips failed the judge batch" % int(judge_run["failed_clips"]))
    if not evidence_uris:
        reasons.append("Gate D evidence bundle needs at least one evidence URI")
    if not protocol_hash:
        reasons.append("Gate D evidence bundle needs the frozen protocol hash")
    statuses: Dict[str, int] = {}
    for report in judge_run.get("reports", ()):  # counted, never inferred
        key = str(report.get("judge_status", "unknown"))
        statuses[key] = statuses.get(key, 0) + 1
    return {
        "gate_id": "D",
        "status": "blocked" if reasons else "not_run",
        "protocol_hash": protocol_hash,
        "fixture_ids": [clip.clip_id for clip in heldout],
        "start_ids": [],
        "backend_revisions": {},
        "policy_revisions": {},
        "judge_revisions": {
            "model_id": frozen.get("model_id"),
            "model_revision": frozen.get("model_revision"),
            "processor_revision": frozen.get("processor_revision"),
            "transformers_version": frozen.get("transformers_version"),
            "task_registry_id": frozen.get("task_registry_id"),
            "task_registry_hash": frozen.get("task_registry_hash"),
            "producer_source_hash": (frozen.get("trusted_producer") or {}).get("source_hash"),
        },
        "evidence_uris": list(evidence_uris),
        "measurements": {
            "calibration_manifest_hash": frozen.get("calibration_manifest_hash"),
            "development_clips": sum(1 for clip in clips if clip.split == DEVELOPMENT_SPLIT),
            "heldout_clips": len(heldout),
            "scored_clips": int(judge_run.get("scored_clips", 0) or 0),
            "failed_clips": int(judge_run.get("failed_clips", 0) or 0),
            "judge_status_counts": statuses,
            "calibration_class": calibration_class,
            "human_annotation_present": human_annotation_present,
        },
        "thresholds": {
            "sampling": dict(PRIMARY_SAMPLING),
            "development_clips_per_task": FULL_PLAN_COUNTS[DEVELOPMENT_SPLIT],
            "heldout_clips_per_task": FULL_PLAN_COUNTS[HELDOUT_SPLIT],
            "required_heldout_overlap": 50,
            "required_blinded_human_annotators": 2,
            "split_lineages_disjoint": True,
        },
        "reasons": reasons,
        "evidence_kind": "real_vlm_judge_reports_over_generated_clips",
    }


# ---------------------------------------------------------------------- CLI


def _parse_references(values: Sequence[str]) -> Dict[str, Path]:
    references: Dict[str, Path] = {}
    for item in values or ():
        if "=" not in item:
            raise PanelError("--reference must be TASK=PATH, got %r" % item)
        task, path = item.split("=", 1)
        if task.strip() not in TASKS:
            raise PanelError("unknown task in --reference: %r" % task)
        references[task.strip()] = Path(path)
    return references


#: Scenario cohorts whose lineages a calibration clip may never reuse.  The
#: panel is built from ``calibration``-cohort starts, so every other cohort is
#: reserved.  Spec section 5 requires at least ``primary`` and
#: ``cost_confirmation``; reserving the rest costs nothing and cannot leak.
RESERVED_SCENARIO_COHORTS: Tuple[str, ...] = (
    "primary",
    "cost_confirmation",
    "development",
    "reverse_validation",
)


def _lineages_from_files(
    paths: Sequence[str], reserved_cohorts: Sequence[str] = RESERVED_SCENARIO_COHORTS
) -> List[str]:
    """Collect reserved lineages from ``scenarios.jsonl`` or a plain lineage list.

    A row that declares a ``cohort`` contributes only when that cohort is
    reserved, so passing the whole scenario manifest does not accidentally
    exclude the ``calibration``-cohort lineages this panel is built from.  A row
    with no cohort is always treated as reserved.
    """

    reserved = {str(value) for value in reserved_cohorts}
    lineages: List[str] = []
    for path in paths or ():
        for row in read_rows(Path(path)):
            value = row.get("start_lineage_id") or row.get("source_lineage_id")
            if not isinstance(value, str) or not value.strip():
                continue
            cohort = row.get("cohort")
            if isinstance(cohort, str) and cohort.strip() and cohort not in reserved:
                continue
            lineages.append(value)
    return lineages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select", help="Freeze the stratified 150-clip panel")
    select.add_argument("--pool", required=True, help="Candidate generated-clip pool (JSON/JSONL).")
    select.add_argument(
        "--exclude-lineages",
        action="append",
        default=[],
        help="scenarios.jsonl or any JSONL of lineages reserved for other cohorts.",
    )
    select.add_argument(
        "--reserve-cohort",
        action="append",
        default=[],
        help="Scenario cohort whose lineages this panel may not reuse (default: everything but calibration).",
    )
    select.add_argument("--manifest-out", required=True)
    select.add_argument("--freeze-out", required=True)
    select.add_argument("--media-root", help="Directory that receives the blinded clip copies.")
    select.add_argument("--stage-media", action="store_true")
    select.add_argument("--copy-media", action="store_true", help="Copy instead of hard-linking.")
    select.add_argument("--execute", action="store_true")

    freeze = commands.add_parser("freeze", help="Pin pre-run judge evidence")
    freeze.add_argument("--manifest", required=True)
    freeze.add_argument("--media-root", required=True)
    freeze.add_argument("--protocol-id", required=True)
    freeze.add_argument("--model-revision", required=True)
    freeze.add_argument("--processor-revision", required=True)
    freeze.add_argument("--transformers-version", required=True)
    freeze.add_argument("--runtime-lock-id", required=True)
    freeze.add_argument("--asset-manifest-id", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--execute", action="store_true")

    judge = commands.add_parser("judge", help="Batch-run the frozen primary judge protocol")
    judge.add_argument("--manifest", required=True)
    judge.add_argument("--frozen", required=True)
    judge.add_argument("--media-root", required=True)
    judge.add_argument("--model-path", required=True)
    judge.add_argument("--reference", action="append", default=[], metavar="TASK=PATH", required=True)
    judge.add_argument("--base-seed", type=int, required=True)
    judge.add_argument("--split", action="append", default=[], choices=[DEVELOPMENT_SPLIT, HELDOUT_SPLIT])
    judge.add_argument("--fps", type=float, default=5.0)
    judge.add_argument("--reports-out", required=True, help="Append-only judge report JSONL.")
    judge.add_argument("--run-report", help="Batch run summary JSON.")
    judge.add_argument("--execute", action="store_true")

    bind = commands.add_parser("bind", help="Complete frozen evidence and emit the Gate D bundle")
    bind.add_argument("--manifest", required=True)
    bind.add_argument("--frozen", required=True)
    bind.add_argument("--reports", action="append", required=True)
    bind.add_argument("--run-report")
    bind.add_argument("--output", required=True)
    bind.add_argument("--gate-evidence")
    bind.add_argument("--evidence-uri", action="append", default=[])
    bind.add_argument("--protocol-hash")
    bind.add_argument("--calibration-class")
    bind.add_argument("--human-annotation-present", action="store_true")
    bind.add_argument("--execute", action="store_true")
    return parser


def run(argv: Optional[Sequence[str]] = None) -> Tuple[int, Dict[str, Any]]:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "select":
        result = select_panel(
            read_rows(Path(args.pool)),
            excluded_lineages=_lineages_from_files(
                args.exclude_lineages, args.reserve_cohort or RESERVED_SCENARIO_COHORTS
            ),
        )
        report = {key: value for key, value in result.items() if key != "selection_records"}
        if args.stage_media:
            if not args.media_root:
                raise PanelError("--stage-media requires --media-root")
            report["staging"] = stage_media(
                result["selection_records"],
                Path(args.media_root),
                execute=args.execute,
                link=not args.copy_media,
            )
        if args.execute and result["status"] == "pass":
            report["written_manifest"] = write_jsonl(Path(args.manifest_out), result["manifest_rows"])
            report["written_freeze"] = write_json(
                Path(args.freeze_out),
                {
                    "schema_version": 1,
                    "kind": "plumb_calibration_panel_freeze",
                    "frozen_at": _utc_now(),
                    "selection_salt": result["selection_salt"],
                    "clip_id_salt": result["clip_id_salt"],
                    "plan": result["plan"],
                    "calibration_manifest_hash": result["calibration_manifest_hash"],
                    "pool_size": result["pool_size"],
                    "strata_coverage": result["strata_coverage"],
                    "rejected_candidates": result["rejected_candidates"],
                    "selection_records": result["selection_records"],
                    "note": (
                        "Selection is frozen before any annotation. Re-running select on the same "
                        "pool reproduces this panel exactly."
                    ),
                    "qualified": False,
                },
            )
        elif not args.execute:
            report["dry_run"] = "Pass --execute to write the manifest and freeze record."
        return (0 if result["status"] == "pass" else 2), report

    if args.command == "freeze":
        frozen = freeze_evidence(
            read_rows(Path(args.manifest)),
            protocol_id=args.protocol_id,
            model_revision=args.model_revision,
            processor_revision=args.processor_revision,
            transformers_version=args.transformers_version,
            runtime_lock_id=args.runtime_lock_id,
            asset_manifest_id=args.asset_manifest_id,
            media_root=Path(args.media_root),
        )
        if args.execute:
            frozen["written"] = write_json(Path(args.output), frozen)
        else:
            frozen["dry_run"] = "Pass --execute to write %s" % args.output
        return (0 if frozen["status"] == "pass" else 2), frozen

    if args.command == "judge":
        frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
        reports_path = Path(args.reports_out)
        existing = [row.get("clip_id") for row in read_rows(reports_path)] if reports_path.is_file() else []
        result = run_batch_judge(
            read_rows(Path(args.manifest)),
            frozen,
            media_root=Path(args.media_root),
            references=_parse_references(args.reference),
            model_path=Path(args.model_path),
            base_seed=args.base_seed,
            splits=tuple(args.split) or (HELDOUT_SPLIT,),
            execute=args.execute,
            existing_clip_ids=[value for value in existing if value],
            fps=args.fps,
        )
        if args.execute and result["reports"]:
            reports_path.parent.mkdir(parents=True, exist_ok=True)
            with reports_path.open("a", encoding="utf-8") as handle:
                for payload in result["reports"]:
                    handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            result["reports_path"] = str(reports_path)
        summary = {key: value for key, value in result.items() if key != "reports"}
        summary["report_clip_ids"] = [payload.get("clip_id") for payload in result["reports"]]
        if args.run_report:
            if args.execute:
                summary["written"] = write_json(Path(args.run_report), summary, overwrite=True)
            else:
                summary["dry_run_run_report"] = args.run_report
        return (0 if result["status"] in ("completed", "planned") else 2), summary

    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    reports: List[Mapping[str, Any]] = []
    for path in args.reports:
        reports.extend(read_rows(Path(path)))
    manifest_rows = read_rows(Path(args.manifest))
    complete = bind_evidence(frozen, reports, manifest_rows)
    judge_run = (
        json.loads(Path(args.run_report).read_text(encoding="utf-8"))
        if args.run_report and Path(args.run_report).is_file()
        else {"status": "completed" if reports else "not_run", "reports": reports, "failed_clips": 0}
    )
    bundle = gate_d_evidence_bundle(
        complete,
        judge_run,
        manifest_rows,
        evidence_uris=args.evidence_uri,
        protocol_hash=args.protocol_hash,
        calibration_class=args.calibration_class,
        human_annotation_present=args.human_annotation_present,
    )
    report = {
        "frozen_status": complete["status"],
        "binding_problems": complete["binding_problems"],
        "gate_d_evidence": bundle,
    }
    if args.execute:
        report["written_frozen"] = write_json(Path(args.output), complete, overwrite=True)
        if args.gate_evidence:
            report["written_gate_evidence"] = write_json(Path(args.gate_evidence), bundle, overwrite=True)
    else:
        report["dry_run"] = "Pass --execute to write the frozen evidence and Gate D bundle."
    return (0 if complete["status"] == "pass" else 2), report


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        code, report = run(argv)
    except (PanelError, CalibrationError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
