#!/usr/bin/env python3
"""Collect unqualified Qwen teacher samples for the frozen LoRA pilot.

This collector is deliberately narrower than primary judging.  It receives
only 16 manifest-bound real close-drawer clips, their 16 already-selected RGB
frames, actual recorded control timestamps, and a scene-context reference.
It gives Qwen exactly the canonical close-drawer instruction/rubric plus a
fixed scene-context disclaimer under ``diagnostic_mode=True``.  No policy,
action, gate, or success-rate field reaches the VLM.

Every clip retains its full raw five-sample judge report.  A training target is
selected only from an actual schema-valid final sample whose semantic tuple is
the unique mode.  Ties and invalid samples are exclusions, never repaired or
turned into a synthesized majority JSON.  All outputs remain unqualified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
TRANSFORMERS_VERSION = "4.49.0"
INPUT_SCHEMA = "plumb-judge-teacher-pilot-input-v1"
FINAL_INPUTS_SCHEMA = "plumb-judge-teacher-pilot-inputs-v1"
CLIP_REPORT_SCHEMA = "plumb-judge-teacher-pilot-clip-v1"
LABEL_SCHEMA = "plumb-judge-teacher-pilot-label-v1"
SELECTION_RULE_ID = "unique_modal_semantic_tuple_actual_member_v1"
INPUT_PROFILE = "qwen_rubric_serving_messages_v1"
PURPOSE = "uncalibrated_judge_distillation_pilot"
LABEL_SOURCE = "uncalibrated_teacher_structured_sample"
TEACHER_SEEDS = (410001, 410002, 410003, 410004, 410005)
FROZEN_SELECTION_SHA256 = "18717375576b9fb3e08606a7e09b30d73f2dd731b167b3e99c0866f122cf5bf8"
SCENE_CONTEXT_DISCLAIMER = "Reference image is scene context only; it is not a goal image, target state, or label."
_SAFE_CLIP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BANNED_JUDGE_INPUT_KEYS = frozenset(
    {
        "action",
        "actions",
        "action_text",
        "command",
        "commands",
        "policy",
        "policy_id",
        "policy_name",
        "policy_metadata",
        "gate",
        "gate_status",
        "condition",
        "world_condition",
        "success_rate",
        "reference_percentage",
    }
)


class TeacherPilotError(ValueError):
    """An input cannot safely support the unqualified teacher collector."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TeacherPilotError("%s must be a SHA-256 digest" % label)
    candidate = value.lower().strip()
    if candidate.startswith("sha256:"):
        candidate = candidate[len("sha256:") :]
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        raise TeacherPilotError("%s must be a SHA-256 digest" % label)
    return candidate


def _json(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TeacherPilotError("unable to read %s" % label) from error
    if not isinstance(value, Mapping):
        raise TeacherPilotError("%s must be a JSON object" % label)
    return dict(value)


def _regular_relative(root: Path, value: Any, label: str) -> Tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise TeacherPilotError("%s must be a nonempty dataset-root-relative path" % label)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise TeacherPilotError("%s escapes dataset root" % label)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise TeacherPilotError("%s contains a symlink: %s" % (label, current))
    try:
        resolved = (root / relative).resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise TeacherPilotError("%s is not a regular file under dataset root" % label) from error
    if not resolved.is_file() or resolved.is_symlink():
        raise TeacherPilotError("%s is not a regular source file" % label)
    return resolved, relative.as_posix()


def _assert_no_barred_judge_input(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TeacherPilotError("input manifest keys must be strings")
            lowered = key.lower()
            if lowered in _BANNED_JUDGE_INPUT_KEYS:
                raise TeacherPilotError("barred judge-input field %s" % ((path + ".") if path else "") + key)
            _assert_no_barred_judge_input(item, (path + "." if path else "") + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_barred_judge_input(item, "%s[%d]" % (path, index))


def _load_rgb(path: Path, expected_pixel_sha256: str, label: str) -> Any:
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as error:
        raise TeacherPilotError("Pillow and NumPy are required to decode teacher-pilot PNGs") from error
    from plumb.policies.provenance import image_pixel_hash

    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
    except OSError as error:
        raise TeacherPilotError("unable to decode %s" % label) from error
    record = image_pixel_hash(array)
    if record.get("semantics") != "array_pixels" or _digest(record.get("sha256"), label + ".computed_pixel_sha256") != expected_pixel_sha256:
        raise TeacherPilotError("%s decoded RGB array pixels do not match the manifest" % label)
    return array


def _finite_timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TeacherPilotError("%s must be a finite actual control timestamp" % label)
    return float(value)


def _canonical_task() -> Any:
    from plumb.policies.tasks import BENCHMARK_TASK_REGISTRY

    return BENCHMARK_TASK_REGISTRY.get("close_drawer")


def _teacher_rubric() -> str:
    return _canonical_task().rubric + "\n\n" + SCENE_CONTEXT_DISCLAIMER


def _collection_config() -> Dict[str, Any]:
    return {
        "purpose": PURPOSE,
        "model_id": QWEN_MODEL_ID,
        "model_revision": QWEN_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "task_id": "close_drawer",
        "instruction": _canonical_task().instruction,
        "rubric": _teacher_rubric(),
        "diagnostic_mode": True,
        "reference_role": "scene_context_not_goal",
        "teacher_seeds": list(TEACHER_SEEDS),
        "selection_rule": SELECTION_RULE_ID,
        "frozen_selection_sha256": FROZEN_SELECTION_SHA256,
    }


def _validated_clip(root: Path, clip: Mapping[str, Any], index: int) -> Dict[str, Any]:
    clip_id = clip.get("clip_id")
    if not isinstance(clip_id, str) or not _SAFE_CLIP_ID.fullmatch(clip_id):
        raise TeacherPilotError("clips[%d].clip_id must be a safe immutable identifier" % index)
    lineage = clip.get("source_lineage_id")
    if not isinstance(lineage, str) or not lineage.strip():
        raise TeacherPilotError("clips[%d].source_lineage_id is required" % index)
    cohort = clip.get("cohort", clip.get("split"))
    if cohort not in ("train", "development_validation"):
        raise TeacherPilotError("clips[%d].cohort must be train or development_validation" % index)
    timestamp_semantics = clip.get("timestamp_semantics", "unverified_nominal_dataset_times")
    if timestamp_semantics not in ("unverified_nominal_dataset_times", "source_parquet_nominal_timestamp"):
        raise TeacherPilotError(
            "clips[%d].timestamp_semantics must be unverified_nominal_dataset_times or source_parquet_nominal_timestamp" % index
        )
    video = clip.get("video")
    if not isinstance(video, Mapping):
        raise TeacherPilotError("clips[%d].video must be an object" % index)
    video_path, video_relative = _regular_relative(root, video.get("path"), "clips[%d].video.path" % index)
    if _sha256(video_path) != _digest(video.get("sha256"), "clips[%d].video.sha256" % index):
        raise TeacherPilotError("clips[%d] video hash changed" % index)
    frames = clip.get("frames")
    if not isinstance(frames, list) or len(frames) != 16:
        raise TeacherPilotError("clips[%d] must bind exactly 16 preselected frames" % index)
    checked_frames = []
    timestamps = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise TeacherPilotError("clips[%d].frames[%d] must be an object" % (index, frame_index))
        path, relative = _regular_relative(root, frame.get("path"), "clips[%d].frames[%d].path" % (index, frame_index))
        file_sha = _digest(frame.get("sha256"), "clips[%d].frames[%d].sha256" % (index, frame_index))
        pixel_sha = _digest(frame.get("pixel_sha256"), "clips[%d].frames[%d].pixel_sha256" % (index, frame_index))
        if _sha256(path) != file_sha:
            raise TeacherPilotError("clips[%d] frame %d file hash changed" % (index, frame_index))
        timestamp = _finite_timestamp(frame.get("timestamp"), "clips[%d].frames[%d].timestamp" % (index, frame_index))
        if timestamps and timestamp <= timestamps[-1]:
            raise TeacherPilotError("clips[%d] frame timestamps must be finite strictly increasing source dataset times" % index)
        timestamps.append(timestamp)
        checked_frames.append({"path": relative, "sha256": file_sha, "pixel_sha256": pixel_sha, "timestamp": timestamp})
    reference = clip.get("reference")
    if not isinstance(reference, Mapping) or reference.get("role") != "scene_context_not_goal":
        raise TeacherPilotError("clips[%d] requires one reference with role=scene_context_not_goal" % index)
    reference_path, reference_relative = _regular_relative(root, reference.get("path"), "clips[%d].reference.path" % index)
    reference_sha = _digest(reference.get("sha256"), "clips[%d].reference.sha256" % index)
    reference_pixel_sha = _digest(reference.get("pixel_sha256"), "clips[%d].reference.pixel_sha256" % index)
    source_uri = reference.get("source_uri")
    if not isinstance(source_uri, str) or not source_uri.strip():
        raise TeacherPilotError("clips[%d].reference.source_uri is required" % index)
    if _sha256(reference_path) != reference_sha:
        raise TeacherPilotError("clips[%d] reference file hash changed" % index)
    checked = {
        "clip_id": clip_id,
        "source_lineage_id": lineage,
        "cohort": cohort,
        "timestamp_semantics": timestamp_semantics,
        "video": {"path": video_relative, "sha256": _digest(video.get("sha256"), "video.sha256")},
        "frames": checked_frames,
        "reference": {
            "path": reference_relative,
            "sha256": reference_sha,
            "pixel_sha256": reference_pixel_sha,
            "source_uri": source_uri,
            "role": "scene_context_not_goal",
        },
    }
    checked["input_binding_sha256"] = _canonical_sha256(checked)
    return checked


def validate_candidate_manifest(root: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    _assert_no_barred_judge_input(payload)
    if payload.get("schema") != INPUT_SCHEMA or payload.get("qualified") is not False:
        raise TeacherPilotError("candidate manifest must be explicitly unqualified %s" % INPUT_SCHEMA)
    task = _canonical_task()
    declared_task = payload.get("task")
    if not isinstance(declared_task, Mapping):
        raise TeacherPilotError("candidate manifest must bind the canonical close_drawer task")
    if declared_task.get("task_id") != task.task_id:
        raise TeacherPilotError("teacher pilot only accepts canonical task close_drawer")
    if declared_task.get("task_registry_hash") != __import__("plumb.policies.tasks", fromlist=["TASK_REGISTRY_HASH"]).TASK_REGISTRY_HASH:
        raise TeacherPilotError("candidate manifest task_registry_hash does not match the canonical registry")
    if declared_task.get("rubric_hash") != task.rubric_hash:
        raise TeacherPilotError("candidate manifest rubric_hash does not match canonical close_drawer")
    if _digest(payload.get("selection_sha256"), "candidate manifest selection_sha256") != FROZEN_SELECTION_SHA256:
        raise TeacherPilotError("candidate manifest does not bind the frozen 16-clip selection")
    clips = payload.get("clips")
    if not isinstance(clips, list) or len(clips) != 16:
        raise TeacherPilotError("teacher pilot requires exactly 16 frozen candidate clips")
    checked = [_validated_clip(root, clip, index) if isinstance(clip, Mapping) else None for index, clip in enumerate(clips)]
    if any(clip is None for clip in checked):
        raise TeacherPilotError("candidate clips must all be objects")
    values = [clip for clip in checked if clip is not None]
    ids = [clip["clip_id"] for clip in values]
    lineages = [clip["source_lineage_id"] for clip in values]
    if len(ids) != len(set(ids)) or len(lineages) != len(set(lineages)):
        raise TeacherPilotError("teacher pilot clips and source lineages must be unique")
    if sum(clip["cohort"] == "train" for clip in values) != 12 or sum(clip["cohort"] == "development_validation" for clip in values) != 4:
        raise TeacherPilotError("teacher pilot split is frozen at 12 train and 4 development_validation clips")
    return {
        "task": {"task_id": task.task_id, "instruction": task.instruction, "canonical_rubric": task.rubric, "rubric_hash": task.rubric_hash},
        "selection_sha256": FROZEN_SELECTION_SHA256,
        "clips": values,
    }


def _model_binding(model_root: Path, manifest_path: Path) -> Dict[str, str]:
    from deploy.baseten.training.train_judge_lora import _local_model_binding

    binding = _local_model_binding(str(model_root), str(manifest_path),
                                   {"model_id": QWEN_MODEL_ID, "model_revision": QWEN_REVISION})
    return {"root": binding["model_root"], "manifest": binding["model_manifest"],
            "manifest_sha256": binding["model_manifest_sha256"], "revision": QWEN_REVISION}


def _runtime_binding(path: Path) -> Dict[str, str]:
    lock = _json(path, "judge runtime lock")
    if lock.get("schema") != "plumb-judge-training-runtime-v1":
        raise TeacherPilotError("judge runtime lock schema is not plumb-judge-training-runtime-v1")
    packages = lock.get("packages")
    transformers = packages.get("transformers") if isinstance(packages, Mapping) else None
    version = transformers.get("import_version") if isinstance(transformers, Mapping) else None
    if not isinstance(version, str) or version.split("+", 1)[0] != TRANSFORMERS_VERSION:
        raise TeacherPilotError("judge runtime lock must bind Transformers %s" % TRANSFORMERS_VERSION)
    try:
        installed = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as error:
        raise TeacherPilotError("judge runtime lacks Transformers") from error
    if installed.split("+", 1)[0] != TRANSFORMERS_VERSION:
        raise TeacherPilotError("installed Transformers %r does not match pilot runtime %s" % (installed, TRANSFORMERS_VERSION))
    return {"path": str(path.resolve()), "sha256": _sha256(path), "transformers_version": installed}


def _next_attempt_path(root: Path, clip: Mapping[str, Any]) -> Path:
    directory = root / "rawteacher" / clip["clip_id"] / clip["input_binding_sha256"]
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    index = 1
    while (directory / ("attempt-%04d.json" % index)).exists():
        index += 1
    return directory / ("attempt-%04d.json" % index)


def _write_no_clobber_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite teacher-pilot evidence: %s" % path) from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _selection(raw_report: Mapping[str, Any]) -> Dict[str, Any]:
    from plumb.policies.judge import parse_rubric_json

    reports = raw_report.get("raw_judge_samples")
    if not isinstance(reports, list) or len(reports) != 5:
        raise TeacherPilotError("teacher report must retain exactly five raw sample reports")
    indexes = [sample.get("sample_index") if isinstance(sample, Mapping) else None for sample in reports]
    if indexes != list(range(5)):
        raise TeacherPilotError("teacher report sample indexes must be exactly 0 through 4 in order")
    valid = []
    invalid = []
    for sample in reports:
        if not isinstance(sample, Mapping) or not isinstance(sample.get("sample_index"), int):
            raise TeacherPilotError("raw teacher sample identity is malformed")
        attempts = sample.get("attempts")
        if not isinstance(attempts, list) or not attempts or not isinstance(attempts[-1], Mapping):
            raise TeacherPilotError("raw teacher sample attempts are malformed")
        final = attempts[-1]
        if final.get("sample_index") != sample["sample_index"] or not isinstance(final.get("attempt_index"), int):
            raise TeacherPilotError("raw teacher final attempt identity is malformed")
        raw = final.get("raw_output")
        parsed = final.get("parsed")
        if not isinstance(raw, str) or not isinstance(parsed, Mapping):
            invalid.append({"sample_index": sample["sample_index"], "reason": final.get("failure_reason") or "no_schema_valid_final_sample"})
            continue
        try:
            actual = parse_rubric_json(raw).as_dict()
        except Exception as error:
            invalid.append({"sample_index": sample["sample_index"], "reason": "raw_schema_recheck_failed:%s" % type(error).__name__})
            continue
        if dict(parsed) != dict(actual):
            invalid.append({"sample_index": sample["sample_index"], "reason": "parsed_record_differs_from_actual_raw_sample"})
            continue
        semantic = (actual["integrity"], actual["collision"], actual["progress"], actual["completion_evidence"])
        valid.append(
            {
                "sample_index": sample["sample_index"],
                "attempt_index": final.get("attempt_index"),
                "semantic_tuple": list(semantic),
                "label": actual,
                "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            }
        )
    if not valid:
        return {"status": "excluded_no_schema_valid_sample", "valid_samples": [], "invalid_samples": invalid, "rule": SELECTION_RULE_ID}
    groups: Dict[Tuple[Any, ...], list[Dict[str, Any]]] = {}
    for candidate in valid:
        groups.setdefault(tuple(candidate["semantic_tuple"]), []).append(candidate)
    peak = max(len(members) for members in groups.values())
    modes = [key for key, members in groups.items() if len(members) == peak]
    agreement = [
        {"semantic_tuple": list(key), "count": len(members), "sample_indexes": sorted(member["sample_index"] for member in members)}
        for key, members in sorted(
            groups.items(), key=lambda item: json.dumps(list(item[0]), separators=(",", ":"), ensure_ascii=True)
        )
    ]
    if len(modes) != 1:
        return {"status": "excluded_modal_tie", "rule": SELECTION_RULE_ID, "valid_samples": valid, "invalid_samples": invalid, "agreement": agreement, "modal_count": peak}
    members = sorted(groups[modes[0]], key=lambda value: value["sample_index"])
    selected = members[0]
    return {
        "status": "accepted_unique_modal_tuple",
        "rule": SELECTION_RULE_ID,
        "selected": selected,
        "modal_count": peak,
        "quorum_reached": peak >= 3,
        "valid_samples": valid,
        "invalid_samples": invalid,
        "agreement": agreement,
    }


def _completed_record(root: Path, clip: Mapping[str, Any]) -> Optional[Tuple[Path, Dict[str, Any]]]:
    directory = root / "rawteacher" / clip["clip_id"] / clip["input_binding_sha256"]
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("attempt-*.json"), reverse=True):
        try:
            payload = _json(path, "existing teacher clip report")
        except TeacherPilotError:
            continue
        if (
            payload.get("schema") == CLIP_REPORT_SCHEMA
            and payload.get("status") == "completed_unqualified"
            and payload.get("input_binding_sha256") == clip["input_binding_sha256"]
            and payload.get("collection_config_sha256") == _canonical_sha256(_collection_config())
            and isinstance(payload.get("selection"), Mapping)
        ):
            return path, payload
    return None


def _judge_clip(clip: Mapping[str, Any], *, dataset_root: Path, judge: Any, model: Mapping[str, str], runtime: Mapping[str, str]) -> Dict[str, Any]:
    from plumb.policies.judge import JudgeInputProvenance, JudgeRequest, ReferenceImage

    frames = tuple(_load_rgb(dataset_root / item["path"], item["pixel_sha256"], "frame %d" % index) for index, item in enumerate(clip["frames"]))
    reference_data = clip["reference"]
    reference_image = _load_rgb(dataset_root / reference_data["path"], reference_data["pixel_sha256"], "scene context reference")
    task = _canonical_task()
    request = JudgeRequest(
        frames=frames,
        frame_timestamps=tuple(item["timestamp"] for item in clip["frames"]),
        reference_images=(ReferenceImage(image=reference_image, source_uri=reference_data["source_uri"], sha256=reference_data["sha256"]),),
        diagnostic_mode=True,
        task_instruction=task.instruction,
        task_rubric=_teacher_rubric(),
        provenance=JudgeInputProvenance(
            clip_id=clip["clip_id"],
            video_sha256=clip["video"]["sha256"],
            protocol_id="uncalibrated_judge_distillation_pilot",
        ),
    )
    report = judge.evaluate(request, seeds=TEACHER_SEEDS).as_dict()
    selection = _selection(report)
    return {
        "schema": CLIP_REPORT_SCHEMA,
        "qualified": False,
        "status": "completed_unqualified",
        "purpose": PURPOSE,
        "collection_config": _collection_config(),
        "collection_config_sha256": _canonical_sha256(_collection_config()),
        "input_binding_sha256": clip["input_binding_sha256"],
        "clip": clip,
        "reference_role": "scene_context_not_goal",
        "teacher_prompt": {"task_id": task.task_id, "instruction": task.instruction, "rubric": _teacher_rubric(), "diagnostic_mode": True},
        "model": model,
        "runtime": runtime,
        "teacher_seeds": list(TEACHER_SEEDS),
        "raw_judge_report": report,
        "selection": selection,
        "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
        "finished_at": _utc_now(),
    }


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _label_row(root: Path, clip_record_path: Path, clip_record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    selection = clip_record["selection"]
    if selection.get("status") != "accepted_unique_modal_tuple":
        return None
    clip = clip_record["clip"]
    selected = selection["selected"]
    cohort = "uncalibrated_pilot_train" if clip["cohort"] == "train" else "uncalibrated_pilot_validation"
    raw_sha = _sha256(clip_record_path)
    reference = clip["reference"]
    ref = {"path": reference["path"], "sha256": reference["sha256"], "role": reference["role"], "provenance_uri": reference["source_uri"]}
    return {
        "schema": LABEL_SCHEMA,
        "purpose": PURPOSE,
        "qualified": False,
        "label_source": LABEL_SOURCE,
        "input_profile": INPUT_PROFILE,
        "clip_id": clip["clip_id"],
        "source_lineage_id": clip["source_lineage_id"],
        "cohort": cohort,
        "task": "close_drawer",
        "instruction": clip_record["teacher_prompt"]["instruction"],
        "rubric": clip_record["teacher_prompt"]["rubric"],
        "frames": [item["path"] for item in clip["frames"]],
        "frame_sha256": [item["sha256"] for item in clip["frames"]],
        "frame_timestamps": [item["timestamp"] for item in clip["frames"]],
        "timestamp_semantics": clip["timestamp_semantics"],
        "frame_timestamp_semantics": clip["timestamp_semantics"],
        "reference_images": [ref],
        "refs": [ref],
        "label": selected["label"],
        "teacher_selection": {"rule": SELECTION_RULE_ID, "modal_count": selection["modal_count"], "quorum_reached": selection["quorum_reached"], "semantic_tuple": selected["semantic_tuple"]},
        "raw_teacher_refs": [{"uri": _relative(root, clip_record_path), "sha256": raw_sha, "sample_index": selected["sample_index"]}],
    }


def _write_jsonl_match_or_create(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise TeacherPilotError("existing immutable output differs: %s" % path)
        return _sha256(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise TeacherPilotError("concurrent immutable output differs: %s" % path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _sha256(path)


def _write_json_match_or_create(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise TeacherPilotError("existing immutable output differs: %s" % path)
        return _sha256(path)
    _write_no_clobber_json(path, payload)
    return _sha256(path)


def _completed_records(root: Path, clips: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
    records: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
    for clip in clips:
        record = _completed_record(root, clip)
        if record is not None:
            records[clip["clip_id"]] = record
    return records


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise TeacherPilotError("teacher collection requires an allocated GPU job")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise TeacherPilotError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
    dataset_root = Path(args.dataset_root).resolve(strict=True)
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise TeacherPilotError("dataset root must be an existing non-symlink directory")
    candidate_path, _ = _regular_relative(dataset_root, args.input_manifest, "input manifest")
    candidate = validate_candidate_manifest(dataset_root, _json(candidate_path, "candidate manifest"))
    model = _model_binding(Path(args.model_root), Path(args.model_manifest))
    runtime = _runtime_binding(Path(args.runtime_lock).resolve(strict=True))
    from plumb.policies.judge import QwenJudgeProfile, QwenRubricJudge

    profile = QwenJudgeProfile(
        profile_id="qwen-teacher-pilot-cc594-serving-v1",
        local_model_path=model["root"],
        model_revision=QWEN_REVISION,
        processor_revision=QWEN_REVISION,
        transformers_version=TRANSFORMERS_VERSION,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype="bfloat16",
        device_map="auto",
        asset_manifest_id="qwen-teacher-pilot-model-manifest",
        asset_manifest_sha256=model["manifest_sha256"],
        runtime_lock_id="judge-training-runtime-tf449-v1",
        runtime_lock_sha256=runtime["sha256"],
    )
    judge = QwenRubricJudge(profile)
    capability = judge.capability()
    if capability.status.value != "ready_unqualified":
        raise TeacherPilotError("teacher judge is unavailable: %s" % capability.reason)
    if not isinstance(args.max_clips, int) or isinstance(args.max_clips, bool) or not 1 <= args.max_clips <= 16:
        raise TeacherPilotError("--max-clips must be an integer from 1 through 16")
    completed = _completed_records(dataset_root, candidate["clips"])
    pending = [clip for clip in candidate["clips"] if clip["clip_id"] not in completed]
    for clip in pending[: args.max_clips]:
        output_path = _next_attempt_path(dataset_root, clip)
        started = time.perf_counter()
        started_at = _utc_now()
        try:
            payload = _judge_clip(clip, dataset_root=dataset_root, judge=judge, model=model, runtime=runtime)
            payload["started_at"] = started_at
            payload["total_seconds"] = time.perf_counter() - started
        except Exception as error:
            payload = {
                "schema": CLIP_REPORT_SCHEMA,
                "qualified": False,
                "status": "failed",
                "purpose": PURPOSE,
                "input_binding_sha256": clip["input_binding_sha256"],
                "clip": clip,
                "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=8)},
                "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
                "finished_at": _utc_now(),
                "total_seconds": time.perf_counter() - started,
            }
        _write_no_clobber_json(output_path, payload)
    by_clip = _completed_records(dataset_root, candidate["clips"])
    if len(by_clip) != len(candidate["clips"]):
        return {
            "schema": "plumb-judge-teacher-pilot-summary-v1",
            "purpose": PURPOSE,
            "qualified": False,
            "status": "partial",
            "candidate_count": len(candidate["clips"]),
            "completed_clip_count": len(by_clip),
            "pending_clip_ids": [clip["clip_id"] for clip in candidate["clips"] if clip["clip_id"] not in by_clip],
            "max_clips": args.max_clips,
            "final_outputs_written": False,
            "filled_gates": [],
            "gate_d_eligible": False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
        }
    labels = []
    exclusions = []
    for clip in candidate["clips"]:
        record = by_clip.get(clip["clip_id"])
        if record is None:
            exclusions.append({"clip_id": clip["clip_id"], "source_lineage_id": clip["source_lineage_id"], "cohort": clip["cohort"], "reason": "no_hash_matched_completed_teacher_record", "input_binding_sha256": clip["input_binding_sha256"]})
            continue
        row = _label_row(dataset_root, record[0], record[1])
        if row is None:
            exclusions.append({"clip_id": clip["clip_id"], "source_lineage_id": clip["source_lineage_id"], "cohort": clip["cohort"], "reason": record[1]["selection"]["status"], "selection": record[1]["selection"], "raw_teacher_ref": {"uri": _relative(dataset_root, record[0]), "sha256": _sha256(record[0])}})
        else:
            labels.append(row)
    labels.sort(key=lambda row: (row["cohort"], row["clip_id"]))
    labels_sha = _write_jsonl_match_or_create(dataset_root / "teacher-pilot-labels.jsonl", labels)
    exclusions_sha = _write_jsonl_match_or_create(dataset_root / "teacher-pilot-exclusions.jsonl", exclusions)
    train_rows = [row for row in labels if row["cohort"] == "uncalibrated_pilot_train"]
    validation_rows = [row for row in labels if row["cohort"] == "uncalibrated_pilot_validation"]
    train_sha = _write_jsonl_match_or_create(dataset_root / "train.jsonl", train_rows)
    validation_sha = _write_jsonl_match_or_create(dataset_root / "development_validation.jsonl", validation_rows)
    candidate_states = []
    teacher_hashes = {}
    for clip in candidate["clips"]:
        path, record = by_clip[clip["clip_id"]]
        selection = record["selection"]
        accepted = selection.get("status") == "accepted_unique_modal_tuple"
        candidate_state = {
            "clip_id": clip["clip_id"],
            "source_lineage_id": clip["source_lineage_id"],
            "split": clip["cohort"],
            "status": "accepted" if accepted else "excluded",
        }
        if not accepted:
            candidate_state["exclusion_reason"] = selection.get("status")
        candidate_states.append(candidate_state)
        teacher_hashes[clip["clip_id"]] = {"uri": _relative(dataset_root, path), "sha256": _sha256(path)}
    final_inputs = {
        "schema_version": 1,
        "schema": FINAL_INPUTS_SCHEMA,
        "purpose": PURPOSE,
        "qualified": False,
        "candidates": candidate_states,
        "raw_source_hashes": {
            "candidate_manifest": {"uri": _relative(dataset_root, candidate_path), "sha256": _sha256(candidate_path)},
            "model_manifest": {"uri": model["manifest"], "sha256": model["manifest_sha256"]},
            "runtime_lock": {"uri": runtime["path"], "sha256": runtime["sha256"]},
            "collection_config_sha256": _canonical_sha256(_collection_config()),
            "teacher_reports": teacher_hashes,
            "labels": {"uri": "teacher-pilot-labels.jsonl", "sha256": labels_sha},
            "exclusions": {"uri": "teacher-pilot-exclusions.jsonl", "sha256": exclusions_sha},
            "splits": {
                "train": {"uri": "train.jsonl", "sha256": train_sha},
                "development_validation": {"uri": "development_validation.jsonl", "sha256": validation_sha},
            },
        },
    }
    final_inputs_sha = _write_json_match_or_create(dataset_root / args.output_inputs, final_inputs)
    return {
        "schema": "plumb-judge-teacher-pilot-summary-v1",
        "purpose": PURPOSE,
        "qualified": False,
        "status": "partial" if not labels else "completed_unqualified",
        "candidate_count": len(candidate["clips"]),
        "completed_clip_count": len(by_clip),
        "accepted_count": len(labels),
        "excluded_count": len(exclusions),
        "labels": {"uri": "teacher-pilot-labels.jsonl", "sha256": labels_sha},
        "exclusions": {"uri": "teacher-pilot-exclusions.jsonl", "sha256": exclusions_sha},
        "splits": {
            "train": {"uri": "train.jsonl", "sha256": train_sha, "accepted_count": len(train_rows)},
            "development_validation": {"uri": "development_validation.jsonl", "sha256": validation_sha, "accepted_count": len(validation_rows)},
        },
        "inputs": {"uri": args.output_inputs, "sha256": final_inputs_sha},
        "filled_gates": [],
        "gate_d_eligible": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--input-manifest", default="candidate-inputs.json", help="Candidate-only immutable manifest relative to dataset root.")
    parser.add_argument("--output-inputs", default="inputs.json", help="New immutable final inputs summary relative to dataset root.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--max-clips", type=int, default=16, help="Process at most this many currently incomplete clips; completed hash-matched clips always resume-skip.")
    parser.add_argument("--summary", default="teacher-pilot-summary.json", help="New immutable summary relative to dataset root.")
    parser.add_argument("--shard-summary", help="Optional immutable partial-shard summary relative to dataset root; omitted partial runs rely on per-clip raw reports plus stdout.")
    args = parser.parse_args(argv)
    summary_path: Optional[Path] = None
    try:
        summary = run(args)
        root = Path(args.dataset_root).resolve(strict=True)
        if summary["status"] == "partial":
            if args.shard_summary:
                summary_path = root / args.shard_summary
                _write_no_clobber_json(summary_path, summary)
        else:
            summary_path = root / args.summary
            _write_no_clobber_json(summary_path, summary)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except Exception as error:
        failure = {"schema": "plumb-judge-teacher-pilot-summary-v1", "purpose": PURPOSE, "qualified": False, "status": "failed", "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=8)}, "filled_gates": [], "gate_d_eligible": False, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "source_release": os.environ.get("PLUMB_SOURCE_RELEASE")}
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
