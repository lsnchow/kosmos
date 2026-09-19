"""Preparation-only workflow for a separately revised distilled judge.

The module creates immutable, provenance-bound *inputs* for a future Training
Job.  It never calls a cloud API, uploads data, submits a job, starts a GPU, or
claims that a distilled model is calibrated.  Development clips and labels are
the sole exportable training/validation population.  Held-out calibration,
primary, and cost-confirmation lineages remain forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .calibration import (
    Annotation,
    CalibrationError,
    ClipManifestRow,
    DEVELOPMENT_SPLIT,
    HELDOUT_SPLIT,
    calibration_manifest_hash,
    deterministic_annotation_assignments,
    import_annotations,
    load_frozen_calibration_selection,
    parse_annotations,
    validate_manifest,
)


SCHEMA_VERSION = 1
DATASET_KIND = "plumb_distillation_dataset"
JOB_READINESS_KIND = "plumb_distillation_training_job_readiness"
TRAINING_SPLIT = "training"
VALIDATION_SPLIT = "validation"
DEVELOPMENT_CLIPS_PER_TASK = 20
VALIDATION_CLIPS_PER_TASK = 4
TRAINING_CLIPS_PER_TASK = DEVELOPMENT_CLIPS_PER_TASK - VALIDATION_CLIPS_PER_TASK
REQUIRED_RESERVED_COHORTS = ("primary", "cost_confirmation")


class DistillationError(ValueError):
    """Raised when a proposed distillation input is unsafe or incomplete."""


@dataclass(frozen=True)
class LoraConfiguration:
    """One unselected LoRA candidate; this is never an asserted optimum."""

    learning_rate: float = 1e-3
    rank: int = 64
    alpha: int = 32
    status: str = "candidate_not_optimized"

    def __post_init__(self) -> None:
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(float(self.learning_rate))
            or float(self.learning_rate) <= 0
        ):
            raise DistillationError("LoRA learning_rate must be a positive finite number")
        for field in ("rank", "alpha"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DistillationError("LoRA {0} must be a positive integer".format(field))
        if self.status != "candidate_not_optimized":
            raise DistillationError("LoRA status must remain candidate_not_optimized before a frozen search")


DEFAULT_LORA_CANDIDATE = LoraConfiguration()


@dataclass(frozen=True)
class TeacherIdentity:
    """Pinned teacher/runtime identities to bind every raw teacher response."""

    model_id: str
    model_revision: str
    processor_revision: str
    runtime_lock_id: str
    asset_manifest_id: str
    producer_source_hash: str

    def __post_init__(self) -> None:
        for field in ("model_id", "model_revision", "processor_revision", "runtime_lock_id", "asset_manifest_id"):
            _require_string(getattr(self, field), "teacher " + field)
        _require_hash(self.producer_source_hash, "teacher producer_source_hash")


@dataclass(frozen=True)
class DistillationProtocol:
    """Frozen development-only training plan, before any job is submitted."""

    protocol_id: str
    teacher: TeacherIdentity
    rubric_hash: str
    sampling_hash: str
    search_plan: Mapping[str, Any]
    development_split_seed: int
    candidate_lora: LoraConfiguration = DEFAULT_LORA_CANDIDATE

    def __post_init__(self) -> None:
        _require_string(self.protocol_id, "distillation protocol_id")
        if not isinstance(self.teacher, TeacherIdentity):
            raise DistillationError("distillation protocol teacher must be a TeacherIdentity")
        _require_hash(self.rubric_hash, "distillation rubric_hash")
        _require_hash(self.sampling_hash, "distillation sampling_hash")
        if isinstance(self.development_split_seed, bool) or not isinstance(self.development_split_seed, int):
            raise DistillationError("development_split_seed must be an integer")
        if not isinstance(self.candidate_lora, LoraConfiguration):
            raise DistillationError("candidate_lora must be a LoraConfiguration")
        _validate_search_plan(self.search_plan, self.candidate_lora)


def prepare_distillation_dataset(
    calibration_selection: Union[str, Path],
    human_annotations: Iterable[Union[Annotation, Mapping[str, Any]]],
    teacher_reports: Iterable[Mapping[str, Any]],
    protocol: Union[DistillationProtocol, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Create an in-memory development-only dataset ready to freeze.

    This function demands all 100 development labels and raw teacher reports.
    It rejects any held-out label/report or a lineage that was reserved by the
    frozen calibration selection. No target labels are inferred or synthesized.
    """

    selection = load_frozen_calibration_selection(calibration_selection)
    clips = validate_manifest(selection["clips"])
    protocol_value = _coerce_protocol(protocol)
    _require_passing_calibration_lineage_partition(selection)
    development = tuple(clip for clip in clips if clip.split == DEVELOPMENT_SPLIT)
    heldout = tuple(clip for clip in clips if clip.split == HELDOUT_SPLIT)
    _validate_development_plan(development, heldout)
    labels = _coerce_annotations(human_annotations, clips)
    labels_by_clip = _development_labels(labels, development, heldout)
    split = deterministic_development_split(development, protocol_value.development_split_seed)
    reports_by_clip = _bind_teacher_reports(
        teacher_reports,
        development,
        heldout,
        selection,
        protocol_value,
    )

    records = []
    for clip in sorted(development, key=lambda item: item.clip_id):
        label = labels_by_clip[clip.clip_id]
        report = reports_by_clip[clip.clip_id]
        records.append(
            {
                "clip_id": clip.clip_id,
                "source_lineage_id": clip.source_lineage_id,
                "media_ref": clip.media_ref,
                "media_hash": clip.media_hash,
                "task": clip.task,
                "dataset_split": split[clip.clip_id],
                "human_annotation": _annotation_mapping(label),
                # Preserve the exact raw report object rather than a copied
                # aggregate label; the teacher target can be audited later.
                "teacher_raw_report": report,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": DATASET_KIND,
        "status": "prepared_not_trained",
        "calibration_selection_sha256": selection["sha256"],
        "calibration_manifest_hash": calibration_manifest_hash(clips),
        "distillation_protocol": _protocol_mapping(protocol_value),
        "source_population": {
            "allowed": DEVELOPMENT_SPLIT,
            "excluded": [HELDOUT_SPLIT, *REQUIRED_RESERVED_COHORTS],
            "development_clip_count": len(development),
            "heldout_clip_count_excluded": len(heldout),
        },
        "records": records,
        "counts": {
            "records": len(records),
            TRAINING_SPLIT: sum(1 for record in records if record["dataset_split"] == TRAINING_SPLIT),
            VALIDATION_SPLIT: sum(1 for record in records if record["dataset_split"] == VALIDATION_SPLIT),
        },
        "post_training_requirements": {
            "fresh_heldout_calibration_required": True,
            "copied_labels_cannot_qualify": True,
            "paired_frozen_video_comparison_required": True,
            "original_judge_results_preserved": True,
        },
    }


def freeze_distillation_dataset(
    calibration_selection: Union[str, Path],
    human_annotations: Iterable[Union[Annotation, Mapping[str, Any]]],
    teacher_reports: Iterable[Mapping[str, Any]],
    protocol: Union[DistillationProtocol, Mapping[str, Any]],
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """Write a content-addressed development dataset once, without media copy/upload."""

    payload = prepare_distillation_dataset(calibration_selection, human_annotations, teacher_reports, protocol)
    payload["frozen"] = True
    payload["sha256"] = _self_hash(payload)
    _write_new_immutable_json(Path(output_path), payload, "distillation dataset")
    return payload


def load_frozen_distillation_dataset(path: Union[str, Path]) -> Dict[str, Any]:
    """Read and validate a frozen development-only dataset artifact."""

    payload = _load_json_object(path, "frozen distillation dataset")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != DATASET_KIND:
        raise DistillationError("unrecognized distillation dataset schema")
    if payload.get("frozen") is not True or not _hash_equal(payload.get("sha256"), _self_hash(payload)):
        raise DistillationError("distillation dataset SHA-256 does not match its content")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise DistillationError("distillation dataset records are missing")
    _validate_frozen_dataset_records(records, payload)
    return payload


def deterministic_development_split(
    clips: Iterable[Union[ClipManifestRow, Mapping[str, Any]]], seed: int
) -> Dict[str, str]:
    """Make a deterministic 16/4 train/validation split inside each task."""

    rows = validate_manifest(clips, require_full_plan=False)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DistillationError("development split seed must be an integer")
    if any(clip.split != DEVELOPMENT_SPLIT for clip in rows):
        raise DistillationError("development split accepts development clips only")
    by_task: Dict[str, List[ClipManifestRow]] = {}
    for clip in rows:
        by_task.setdefault(clip.task, []).append(clip)
    assignments: Dict[str, str] = {}
    for task, task_clips in by_task.items():
        if len(task_clips) != DEVELOPMENT_CLIPS_PER_TASK:
            raise DistillationError(
                "development task {0} needs exactly {1} clips for its frozen train/validation split".format(
                    task, DEVELOPMENT_CLIPS_PER_TASK
                )
            )
        lineages = [clip.source_lineage_id for clip in task_clips]
        if len(set(lineages)) != len(lineages):
            raise DistillationError("development source lineages must be unique within each task")
        ranked = sorted(
            task_clips,
            key=lambda clip: hashlib.sha256(
                "plumb-distillation-dev-split-v1\x1f{0}\x1f{1}\x1f{2}".format(seed, task, clip.clip_id).encode("utf-8")
            ).hexdigest(),
        )
        for index, clip in enumerate(ranked):
            assignments[clip.clip_id] = VALIDATION_SPLIT if index < VALIDATION_CLIPS_PER_TASK else TRAINING_SPLIT
    return assignments


def build_training_job_readiness(
    frozen_dataset: Union[str, Path], job_inputs: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Prepare no-submit Training Job inputs while exact cloud schema is unknown.

    There is deliberately no endpoint, API payload, network call, or submit
    method here.  Baseten account/runtime/image details are deployment inputs,
    not values this repository can safely guess.
    """

    dataset = load_frozen_distillation_dataset(frozen_dataset)
    inputs = _safe_job_inputs(job_inputs or {})
    blockers = []
    for field in ("account_id", "runtime_image", "runtime_lock_id"):
        if not _nonempty_string(inputs.get(field)):
            blockers.append(field + "_unknown")
    blockers.append("official_baseten_training_jobs_payload_unverified_no_submission")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": JOB_READINESS_KIND,
        "status": "blocked",
        "dataset_sha256": dataset["sha256"],
        "job_inputs": inputs,
        "launch": {
            "submission_performed": False,
            "endpoint": None,
            "api_payload": None,
            "spend_authorized": False,
        },
        "blockers": blockers,
        "required_after_training": dataset["post_training_requirements"],
    }


def freeze_training_job_readiness(
    frozen_dataset: Union[str, Path], job_inputs: Optional[Mapping[str, Any]], output_path: Union[str, Path]
) -> Dict[str, Any]:
    """Persist a no-submit readiness artifact exactly once for later review."""

    payload = build_training_job_readiness(frozen_dataset, job_inputs)
    payload["frozen"] = True
    payload["sha256"] = _self_hash(payload)
    _write_new_immutable_json(Path(output_path), payload, "training-job readiness")
    return payload


def distilled_revision_readiness(
    frozen_dataset: Union[str, Path], revision_id: str, qualification_evidence: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Require new held-out humans and paired videos for each distilled revision.

    This is a structural check of declared evidence.  It never records Gate D
    as passed and deliberately treats copied/no-new labels as blockers.
    """

    dataset = load_frozen_distillation_dataset(frozen_dataset)
    _require_string(revision_id, "distilled revision_id")
    evidence = dict(qualification_evidence or {})
    blockers = []
    fresh = evidence.get("fresh_heldout_calibration")
    if not isinstance(fresh, Mapping):
        blockers.append("fresh_heldout_calibration_missing")
    else:
        if fresh.get("status") != "pass":
            blockers.append("fresh_heldout_calibration_not_passed")
        fresh_hash = fresh.get("calibration_manifest_hash")
        if not _is_hash(fresh_hash) or _hash_equal(fresh_hash, dataset["calibration_manifest_hash"]):
            blockers.append("fresh_heldout_calibration_not_distinct")
        if fresh.get("human_labels_origin") != "fresh_human_annotation" or fresh.get("copied_labels") is not False:
            blockers.append("copied_or_unverifiable_human_labels")
        if fresh.get("source_lineage_disjoint_from_development") is not True:
            blockers.append("fresh_heldout_lineage_disjointness_unverified")
        if fresh.get("raw_judge_reports_bound") is not True:
            blockers.append("fresh_heldout_raw_judge_provenance_unverified")
    paired = evidence.get("paired_frozen_video_comparison")
    if not isinstance(paired, Mapping) or paired.get("status") != "pass" or not _is_hash(paired.get("video_manifest_hash")):
        blockers.append("paired_frozen_video_comparison_missing_or_failed")
    return {
        "schema_version": SCHEMA_VERSION,
        "distilled_revision_id": revision_id,
        "dataset_sha256": dataset["sha256"],
        "status": "blocked" if blockers else "fresh_evidence_structurally_complete",
        "primary_scoring_allowed": False,
        "gate_d_recording_required": True,
        "original_judge_results_preserved": True,
        "blockers": blockers,
    }


def _coerce_protocol(value: Union[DistillationProtocol, Mapping[str, Any]]) -> DistillationProtocol:
    if isinstance(value, DistillationProtocol):
        return value
    if not isinstance(value, Mapping):
        raise DistillationError("distillation protocol must be an object")
    try:
        data = dict(value)
        teacher = data.get("teacher")
        if isinstance(teacher, Mapping):
            data["teacher"] = TeacherIdentity(**dict(teacher))
        candidate = data.get("candidate_lora", DEFAULT_LORA_CANDIDATE)
        if isinstance(candidate, Mapping):
            data["candidate_lora"] = LoraConfiguration(**dict(candidate))
        return DistillationProtocol(**data)
    except (TypeError, DistillationError) as error:
        raise DistillationError("distillation protocol is invalid") from error


def _protocol_mapping(protocol: DistillationProtocol) -> Dict[str, Any]:
    return {
        "protocol_id": protocol.protocol_id,
        "teacher": asdict(protocol.teacher),
        "rubric_hash": protocol.rubric_hash,
        "sampling_hash": protocol.sampling_hash,
        "search_plan": json.loads(_canonical_json(protocol.search_plan)),
        "development_split_seed": protocol.development_split_seed,
        "candidate_lora": asdict(protocol.candidate_lora),
    }


def _validate_search_plan(value: Any, candidate: LoraConfiguration) -> None:
    if not isinstance(value, Mapping) or value.get("preregistered") is not True:
        raise DistillationError("distillation search_plan must declare preregistered=true")
    configurations = value.get("candidates")
    if not isinstance(configurations, list) or len(configurations) < 2:
        raise DistillationError("distillation search_plan needs at least two preregistered candidate configurations")
    parsed = []
    for configuration in configurations:
        if not isinstance(configuration, Mapping):
            raise DistillationError("distillation search candidates must be objects")
        parsed.append(LoraConfiguration(**dict(configuration)))
    if candidate not in parsed:
        raise DistillationError("the lr=1e-3/r=64/alpha=32 candidate must be retained in the preregistered search")
    early = value.get("early_stopping")
    if not isinstance(early, Mapping) or early.get("enabled") is not True or not _nonempty_string(early.get("metric")):
        raise DistillationError("distillation search_plan requires enabled early stopping and a development metric")
    patience = early.get("patience")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise DistillationError("distillation early-stopping patience must be a positive integer")


def _require_passing_calibration_lineage_partition(selection: Mapping[str, Any]) -> None:
    lineage = selection.get("lineage_partition")
    if not isinstance(lineage, Mapping) or lineage.get("status") != "pass":
        raise DistillationError("frozen calibration selection has no passing primary/cost lineage partition")
    counts = lineage.get("reserved_cohorts")
    if not isinstance(counts, Mapping) or any(cohort not in counts for cohort in REQUIRED_RESERVED_COHORTS):
        raise DistillationError("frozen calibration selection lacks explicit primary and cost-confirmation lineage inputs")


def _validate_development_plan(development: Sequence[ClipManifestRow], heldout: Sequence[ClipManifestRow]) -> None:
    if len(development) != DEVELOPMENT_CLIPS_PER_TASK * 5 or len(heldout) != 10 * 5:
        raise DistillationError("distillation requires the frozen 100-development/50-heldout calibration selection")
    development_lineages = {clip.source_lineage_id for clip in development}
    heldout_lineages = {clip.source_lineage_id for clip in heldout}
    if len(development_lineages) != len(development):
        raise DistillationError("development calibration clips must have unique source lineages")
    if development_lineages & heldout_lineages:
        raise DistillationError("development and heldout calibration lineages overlap")


def _coerce_annotations(
    values: Iterable[Union[Annotation, Mapping[str, Any]]], clips: Sequence[ClipManifestRow]
) -> Tuple[Annotation, ...]:
    raw = list(values)
    if all(isinstance(value, Annotation) for value in raw):
        return tuple(raw)  # type: ignore[return-value]
    if any(isinstance(value, Annotation) for value in raw):
        raise DistillationError("human annotations must be all Annotation objects or all packet mappings")
    assignments = deterministic_annotation_assignments(clips, ("annotator-a", "annotator-b"))
    # Packet mappings normally carry real annotator IDs.  Their source parsing
    # is performed by the CLI with the actual pair; direct callers should pass
    # Annotation objects to avoid replacing ownership information here.
    try:
        return parse_annotations(raw, clips, assignments=assignments, annotator_ids=("annotator-a", "annotator-b"))
    except CalibrationError as error:
        raise DistillationError("unable to parse human annotation mappings; use imported Annotation values") from error


def _development_labels(
    labels: Sequence[Annotation], development: Sequence[ClipManifestRow], heldout: Sequence[ClipManifestRow]
) -> Dict[str, Annotation]:
    development_ids = {clip.clip_id for clip in development}
    heldout_ids = {clip.clip_id for clip in heldout}
    result: Dict[str, Annotation] = {}
    for label in labels:
        if label.clip_id in heldout_ids:
            raise DistillationError("heldout human labels must never enter distillation data")
        if label.clip_id not in development_ids:
            raise DistillationError("annotation references a clip outside frozen development calibration")
        if label.clip_id in result:
            raise DistillationError("distillation needs exactly one human development annotation per clip")
        result[label.clip_id] = label
    if set(result) != development_ids:
        raise DistillationError("distillation needs complete supplied human labels for all 100 development clips")
    return result


def _bind_teacher_reports(
    reports: Iterable[Mapping[str, Any]],
    development: Sequence[ClipManifestRow],
    heldout: Sequence[ClipManifestRow],
    selection: Mapping[str, Any],
    protocol: DistillationProtocol,
) -> Dict[str, Dict[str, Any]]:
    development_by_id = {clip.clip_id: clip for clip in development}
    heldout_ids = {clip.clip_id for clip in heldout}
    result: Dict[str, Dict[str, Any]] = {}
    for raw in reports:
        if not isinstance(raw, Mapping):
            raise DistillationError("teacher reports must be objects")
        report = json.loads(_canonical_json(raw))
        provenance = report.get("provenance")
        if not isinstance(provenance, Mapping):
            raise DistillationError("teacher report provenance is required")
        clip_id = _require_string(provenance.get("clip_id"), "teacher report provenance clip_id")
        if clip_id in heldout_ids:
            raise DistillationError("heldout teacher reports must never enter distillation data")
        clip = development_by_id.get(clip_id)
        if clip is None:
            raise DistillationError("teacher report references a clip outside frozen development calibration")
        if clip_id in result:
            raise DistillationError("duplicate teacher report clip_id")
        _require_exact_hash(provenance, "calibration_manifest_hash", selection["calibration_manifest_hash"], "teacher provenance")
        _require_exact_hash(provenance, "media_hash", clip.media_hash, "teacher provenance")
        _require_exact_hash(provenance, "rubric_hash", protocol.rubric_hash, "teacher provenance")
        _require_exact_hash(provenance, "sampling_hash", protocol.sampling_hash, "teacher provenance")
        teacher = provenance.get("teacher")
        if not isinstance(teacher, Mapping):
            raise DistillationError("teacher report provenance teacher identity is required")
        for field, expected in asdict(protocol.teacher).items():
            if field == "producer_source_hash":
                _require_exact_hash(teacher, field, expected, "teacher provenance identity")
            else:
                _require_exact_string(teacher, field, expected, "teacher provenance identity")
        if not _raw_outputs(report):
            raise DistillationError("teacher report must retain one or more raw outputs")
        result[clip_id] = report
    if set(result) != set(development_by_id):
        raise DistillationError("distillation needs raw teacher reports for every development clip")
    return result


def _raw_outputs(report: Mapping[str, Any]) -> List[str]:
    direct = report.get("raw_outputs")
    if isinstance(direct, list) and direct and all(_nonempty_string(item) for item in direct):
        return list(direct)
    outputs: List[str] = []
    samples = report.get("raw_judge_samples")
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            attempts = sample.get("attempts")
            if not isinstance(attempts, list):
                continue
            for attempt in attempts:
                if isinstance(attempt, Mapping) and _nonempty_string(attempt.get("raw_output")):
                    outputs.append(attempt["raw_output"])
    return outputs


def _annotation_mapping(annotation: Annotation) -> Dict[str, Any]:
    return {
        "annotator_id": annotation.annotator_id,
        "integrity": annotation.integrity,
        "collision": annotation.collision,
        "progress": annotation.progress,
        "completion_evidence": annotation.completion_evidence,
        "evidence_frame_indices": list(annotation.evidence_frame_indices),
        "observable_reason": annotation.observable_reason,
    }


def _validate_frozen_dataset_records(records: Sequence[Any], payload: Mapping[str, Any]) -> None:
    seen = set()
    counts = {TRAINING_SPLIT: 0, VALIDATION_SPLIT: 0}
    lineage_by_split = {TRAINING_SPLIT: set(), VALIDATION_SPLIT: set()}
    for record in records:
        if not isinstance(record, Mapping):
            raise DistillationError("distillation dataset records must be objects")
        clip_id = _require_string(record.get("clip_id"), "dataset clip_id")
        if clip_id in seen:
            raise DistillationError("distillation dataset repeats clip_id")
        seen.add(clip_id)
        split = record.get("dataset_split")
        if split not in counts:
            raise DistillationError("dataset split must be training or validation")
        lineage = _require_string(record.get("source_lineage_id"), "dataset source_lineage_id")
        if lineage in lineage_by_split[split]:
            raise DistillationError("dataset split repeats a source lineage")
        lineage_by_split[split].add(lineage)
        counts[split] += 1
        _require_hash(record.get("media_hash"), "dataset media_hash")
        if not isinstance(record.get("human_annotation"), Mapping) or not isinstance(record.get("teacher_raw_report"), Mapping):
            raise DistillationError("dataset records require human annotation and preserved raw teacher report")
    if lineage_by_split[TRAINING_SPLIT] & lineage_by_split[VALIDATION_SPLIT]:
        raise DistillationError("train/validation splits share a source lineage")
    if counts != {TRAINING_SPLIT: 80, VALIDATION_SPLIT: 20}:
        raise DistillationError("distillation dataset must contain the frozen 80/20 development split")
    declared = payload.get("counts")
    if not isinstance(declared, Mapping) or declared.get(TRAINING_SPLIT) != 80 or declared.get(VALIDATION_SPLIT) != 20:
        raise DistillationError("distillation dataset count summary is inconsistent")


def _safe_job_inputs(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DistillationError("training job inputs must be an object")
    blocked_names = {"api_key", "token", "secret", "password", "authorization"}
    _reject_sensitive_fields(value, blocked_names)
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as error:
        raise DistillationError("training job inputs must be JSON serializable") from error


def _reject_sensitive_fields(value: Any, blocked_names: set) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_name = str(key).lower()
            if key_name in blocked_names or any(marker in key_name for marker in blocked_names):
                raise DistillationError("training job readiness must not persist credentials")
            _reject_sensitive_fields(nested, blocked_names)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_sensitive_fields(nested, blocked_names)


def _write_new_immutable_json(path: Path, payload: Mapping[str, Any], description: str) -> None:
    if path.suffix.lower() != ".json":
        raise DistillationError("{0} output must have a .json suffix".format(description))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise DistillationError("refusing to overwrite existing {0}".format(description)) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _load_json_object(path: Union[str, Path], description: str) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DistillationError("unable to read {0}".format(description)) from error
    if not isinstance(value, Mapping):
        raise DistillationError("{0} must be a JSON object".format(description))
    return dict(value)


def _load_reports(path: Union[str, Path]) -> List[Dict[str, Any]]:
    candidate = Path(path)
    try:
        if candidate.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            value = _load_json_object(candidate, "teacher reports")
            values = value.get("reports", [value])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DistillationError("unable to read teacher reports") from error
    if not isinstance(values, list) or not all(isinstance(value, Mapping) for value in values):
        raise DistillationError("teacher reports must be JSON objects")
    return [dict(value) for value in values]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("sha256", None)
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_hash(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    digest = value[7:] if value.startswith("sha256:") else value
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest.lower())


def _hash_equal(first: Any, second: Any) -> bool:
    if not _is_hash(first) or not _is_hash(second):
        return False
    first_digest = first[7:] if str(first).startswith("sha256:") else first
    second_digest = second[7:] if str(second).startswith("sha256:") else second
    return str(first_digest).lower() == str(second_digest).lower()


def _require_hash(value: Any, field: str) -> str:
    if not _is_hash(value):
        raise DistillationError("{0} must be a SHA-256 digest".format(field))
    return str(value)


def _require_string(value: Any, field: str) -> str:
    if not _nonempty_string(value):
        raise DistillationError("{0} must be a non-empty string".format(field))
    return str(value)


def _require_exact_hash(value: Mapping[str, Any], field: str, expected: Any, context: str) -> None:
    if not _hash_equal(value.get(field), expected):
        raise DistillationError("{0}.{1} does not match frozen provenance".format(context, field))


def _require_exact_string(value: Mapping[str, Any], field: str, expected: str, context: str) -> None:
    if value.get(field) != expected:
        raise DistillationError("{0}.{1} does not match frozen provenance".format(context, field))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI for local-only dataset/job preparation; no command can submit."""

    parser = argparse.ArgumentParser(description="PLUMB distilled-judge preparation (no cloud submission)")
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("freeze-dataset", help="freeze development-only distillation data")
    dataset.add_argument("--calibration-selection", required=True)
    dataset.add_argument("--annotations", required=True, action="append")
    dataset.add_argument("--annotators", required=True, nargs=2)
    dataset.add_argument("--teacher-reports", required=True, action="append")
    dataset.add_argument("--protocol", required=True)
    dataset.add_argument("--output", required=True)
    readiness = commands.add_parser("job-readiness", help="freeze no-submit Training Job inputs")
    readiness.add_argument("--dataset", required=True)
    readiness.add_argument("--job-inputs", required=True)
    readiness.add_argument("--output", required=True)
    revision = commands.add_parser("revision-readiness", help="check fresh-heldout requirements for a distilled revision")
    revision.add_argument("--dataset", required=True)
    revision.add_argument("--revision-id", required=True)
    revision.add_argument("--evidence", required=True)
    revision.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-dataset":
            selection = load_frozen_calibration_selection(args.calibration_selection)
            clips = validate_manifest(selection["clips"])
            assignments = deterministic_annotation_assignments(clips, args.annotators)
            labels: List[Annotation] = []
            for path in args.annotations:
                labels.extend(import_annotations(path, clips, assignments, args.annotators))
            reports: List[Dict[str, Any]] = []
            for path in args.teacher_reports:
                reports.extend(_load_reports(path))
            protocol = _load_json_object(args.protocol, "distillation protocol")
            frozen = freeze_distillation_dataset(args.calibration_selection, labels, reports, protocol, args.output)
            print(json.dumps({"output": str(args.output), "sha256": frozen["sha256"], "status": frozen["status"]}, sort_keys=True))
            return 0
        if args.command == "job-readiness":
            inputs = _load_json_object(args.job_inputs, "training job inputs")
            frozen = freeze_training_job_readiness(args.dataset, inputs, args.output)
            print(json.dumps({"output": str(args.output), "sha256": frozen["sha256"], "status": frozen["status"]}, sort_keys=True))
            return 0
        evidence = _load_json_object(args.evidence, "distilled revision evidence")
        report = distilled_revision_readiness(args.dataset, args.revision_id, evidence)
        report["frozen"] = True
        report["sha256"] = _self_hash(report)
        _write_new_immutable_json(Path(args.output), report, "distilled revision readiness")
        print(json.dumps({"output": str(args.output), "sha256": report["sha256"], "status": report["status"]}, sort_keys=True))
        return 0
    except (CalibrationError, DistillationError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATASET_KIND",
    "DEFAULT_LORA_CANDIDATE",
    "DEVELOPMENT_CLIPS_PER_TASK",
    "DistillationError",
    "DistillationProtocol",
    "JOB_READINESS_KIND",
    "LoraConfiguration",
    "TeacherIdentity",
    "TRAINING_SPLIT",
    "VALIDATION_SPLIT",
    "build_training_job_readiness",
    "deterministic_development_split",
    "distilled_revision_readiness",
    "freeze_distillation_dataset",
    "freeze_training_job_readiness",
    "load_frozen_distillation_dataset",
    "main",
    "prepare_distillation_dataset",
]
