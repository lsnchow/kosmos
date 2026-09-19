"""Offline, blinded human-calibration workflow for the PLUMB judge.

This module creates annotation packets and summarizes supplied labels.  It does
not render clips, infer labels, call a model, or mark Gate D passed merely
because a manifest exists.  A Gate-D pass needs a complete frozen protocol,
explicit numerical tolerances, actual annotation coverage, and raw five-sample
judge reports that bind to a frozen evidence manifest. These bindings are not
cryptographic proof that a model ran.

The full plan is fixed at 150 generated clips: 20 development and 10 held-out
clips for each of five tasks.  Development and held-out source lineages must be
disjoint.  Development clips receive one deterministic human assignment;
every held-out clip is independently assigned to both annotators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


SCHEMA_VERSION = 1
TASKS = (
    "open_drawer",
    "close_drawer",
    "to_basket",
    "to_sink",
    "fold_cloth",
)
DEVELOPMENT_SPLIT = "development"
HELDOUT_SPLIT = "heldout"
FULL_PLAN_COUNTS = {DEVELOPMENT_SPLIT: 20, HELDOUT_SPLIT: 10}
INTEGRITY_VALUES = ("intact", "artifact", "uncertain")
COLLISION_VALUES = ("none_visible", "visible", "uncertain")
COMPLETION_VALUES = ("met", "not_met", "uncertain")

# A blind packet may carry the task-specific rubric context, but never the
# policy, actions, condition, score target, gate status, or experimental arm.
BLINDED_EXPORT_FIELDS = (
    "schema_version",
    "annotator_id",
    "clip_id",
    "media_ref",
    "task",
    "integrity",
    "collision",
    "progress",
    "completion_evidence",
    "evidence_frame_indices",
    "observable_reason",
)
_IMPORT_ALLOWED_FIELDS = set(BLINDED_EXPORT_FIELDS) | {"completion"}
_FORBIDDEN_BLIND_FIELDS = {
    "policy",
    "policy_name",
    "policy_variant",
    "actions",
    "action",
    "action_text",
    "commands",
    "condition",
    "condition_id",
    "condition_label",
    "gate_status",
    "reference_success",
    "reference_rate",
    "world_seed",
    "backend",
    "cohort",
}


class CalibrationError(ValueError):
    """Raised for invalid calibration evidence, labels, or gate inputs."""


@dataclass(frozen=True)
class ClipManifestRow:
    """The minimal clip identity retained by the blinded calibration workflow."""

    clip_id: str
    media_ref: str
    task: str
    split: str
    source_lineage_id: str


@dataclass(frozen=True)
class Annotation:
    """A strict human or aggregated-judge label for one generated clip."""

    clip_id: str
    annotator_id: str
    integrity: str
    collision: str
    progress: int
    completion_evidence: str
    evidence_frame_indices: Tuple[int, ...]
    observable_reason: str

    def __post_init__(self) -> None:
        """Keep programmatically constructed labels as strict as imported ones."""

        _strict_string(self.clip_id, "annotation clip_id")
        _strict_string(self.annotator_id, "annotation annotator_id")
        _strict_enum(self.integrity, INTEGRITY_VALUES, "integrity")
        _strict_enum(self.collision, COLLISION_VALUES, "collision")
        _strict_progress(self.progress)
        _strict_enum(self.completion_evidence, COMPLETION_VALUES, "completion_evidence")
        if self.completion_evidence == "met" and self.progress != 5:
            raise CalibrationError("completion_evidence='met' requires progress=5")
        if self.completion_evidence == "not_met" and self.progress > 4:
            raise CalibrationError("completion_evidence='not_met' requires progress from 0 through 4")
        _parse_evidence_frame_indices(list(self.evidence_frame_indices))
        _parse_observable_reason(self.observable_reason)

    @property
    def binary_success(self) -> Optional[bool]:
        """Return a visible completion label only when this sample is decisive."""

        if self.integrity != "intact" or self.completion_evidence == "uncertain":
            return None
        if self.completion_evidence == "met":
            return True
        return False


@dataclass(frozen=True)
class FrozenGateDProtocol:
    """Identity of the frozen judge protocol required for a Gate-D decision."""

    protocol_hash: str
    rubric_hash: str
    sampling_hash: str
    label_schema_version: int = SCHEMA_VERSION
    evidence_manifest: Optional["FrozenJudgeEvidenceManifest"] = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("protocol_hash", self.protocol_hash),
            ("rubric_hash", self.rubric_hash),
            ("sampling_hash", self.sampling_hash),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CalibrationError("{0} must be a non-empty frozen identifier".format(field_name))
        if self.label_schema_version != SCHEMA_VERSION:
            raise CalibrationError("label_schema_version does not match the calibration schema")


@dataclass(frozen=True)
class TrustedJudgeProducer:
    """A declared trusted producer identity; this is not an attestation system."""

    name: str
    version: str
    source_hash: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("name", self.name),
            ("version", self.version),
            ("source_hash", self.source_hash),
        ):
            if not _nonempty_string(value):
                raise CalibrationError("trusted producer {0} must be non-empty".format(field_name))
        _require_hash(self.source_hash, "trusted producer source_hash")


@dataclass(frozen=True)
class FrozenJudgeEvidenceManifest:
    """Frozen identities and hashes to which every primary judge report binds.

    This checks immutable identifiers and a declared producer, not that a model
    really executed. It is therefore evidence binding/reaggregation only; the
    operational trust decision remains an explicit project responsibility.
    """

    calibration_manifest_hash: str
    heldout_video_hashes: Mapping[str, str]
    task_registry_id: str
    task_registry_hash: str
    model_id: str
    model_revision: str
    processor_revision: str
    transformers_version: str
    runtime_lock_id: str
    asset_manifest_id: str
    heldout_artifact_hashes: Mapping[str, Mapping[str, str]]
    protocol_id: str
    trusted_producer: TrustedJudgeProducer

    def __post_init__(self) -> None:
        for field_name, value in (
            ("calibration_manifest_hash", self.calibration_manifest_hash),
            ("task_registry_id", self.task_registry_id),
            ("task_registry_hash", self.task_registry_hash),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("processor_revision", self.processor_revision),
            ("transformers_version", self.transformers_version),
            ("runtime_lock_id", self.runtime_lock_id),
            ("asset_manifest_id", self.asset_manifest_id),
            ("protocol_id", self.protocol_id),
        ):
            if not _nonempty_string(value):
                raise CalibrationError("frozen judge evidence {0} must be non-empty".format(field_name))
        _require_hash(self.calibration_manifest_hash, "calibration_manifest_hash")
        _require_hash(self.task_registry_hash, "task_registry_hash")
        if not isinstance(self.heldout_video_hashes, Mapping) or not self.heldout_video_hashes:
            raise CalibrationError("heldout_video_hashes must be a non-empty mapping")
        for clip_id, digest in self.heldout_video_hashes.items():
            if not _nonempty_string(clip_id):
                raise CalibrationError("heldout_video_hashes clip IDs must be non-empty")
            _require_hash(digest, "heldout video hash")
        if set(self.heldout_artifact_hashes) != set(self.heldout_video_hashes):
            raise CalibrationError("heldout_artifact_hashes must cover exactly the frozen heldout clips")
        for clip_id, artifact_hashes in self.heldout_artifact_hashes.items():
            if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
                raise CalibrationError("heldout artifact hashes must be non-empty mappings")
            for artifact_name, digest in artifact_hashes.items():
                if not _nonempty_string(artifact_name):
                    raise CalibrationError("artifact hash names must be non-empty")
                _require_hash(digest, "heldout_artifact_hashes.{0}.{1}".format(clip_id, artifact_name))
        if not isinstance(self.trusted_producer, TrustedJudgeProducer):
            raise CalibrationError("trusted_producer must be a TrustedJudgeProducer")


@dataclass(frozen=True)
class GateDTolerances:
    """Numerical acceptance thresholds chosen before held-out labels are exposed."""

    minimum_heldout_overlap: int
    minimum_human_annotation_coverage: float
    minimum_human_consensus_coverage: float
    minimum_judge_comparison_coverage: float
    minimum_sensitivity: float
    minimum_specificity: float
    maximum_absolute_leniency_offset: float

    def __post_init__(self) -> None:
        if self.minimum_heldout_overlap < 1:
            raise CalibrationError("minimum_heldout_overlap must be positive")
        for field_name in (
            "minimum_human_annotation_coverage",
            "minimum_human_consensus_coverage",
            "minimum_judge_comparison_coverage",
            "minimum_sensitivity",
            "minimum_specificity",
        ):
            _require_probability(getattr(self, field_name), field_name)
        if (
            not isinstance(self.maximum_absolute_leniency_offset, (int, float))
            or isinstance(self.maximum_absolute_leniency_offset, bool)
            or not math.isfinite(float(self.maximum_absolute_leniency_offset))
            or not 0 <= float(self.maximum_absolute_leniency_offset) <= 1
        ):
            raise CalibrationError("maximum_absolute_leniency_offset must be a finite value from 0 through 1")


def validate_manifest(
    rows: Iterable[Union[ClipManifestRow, Mapping[str, Any]]], require_full_plan: bool = True
) -> Tuple[ClipManifestRow, ...]:
    """Validate the source-independent calibration manifest deterministically.

    Extra manifest metadata is intentionally discarded after validation.  This
    preserves useful provenance in the upstream manifest without letting
    policy/action/condition values escape into human annotation packets.
    """

    clips: List[ClipManifestRow] = []
    seen_clip_ids = set()
    seen_media_refs = set()
    for index, raw in enumerate(rows):
        clip = raw if isinstance(raw, ClipManifestRow) else _parse_manifest_row(raw, index)
        if not _nonempty_string(clip.clip_id):
            raise CalibrationError("manifest clip_id must be a non-empty string")
        if not _nonempty_string(clip.media_ref):
            raise CalibrationError("manifest media_ref must be a non-empty string")
        if clip.task not in TASKS:
            raise CalibrationError("manifest task must be one of the five planned tasks")
        split = _normalise_split(clip.split)
        if not _nonempty_string(clip.source_lineage_id):
            raise CalibrationError("manifest source_lineage_id must be a non-empty string")
        normalised = ClipManifestRow(
            clip_id=clip.clip_id,
            media_ref=clip.media_ref,
            task=clip.task,
            split=split,
            source_lineage_id=clip.source_lineage_id,
        )
        if normalised.clip_id in seen_clip_ids:
            raise CalibrationError("manifest clip_id values must be unique")
        if normalised.media_ref in seen_media_refs:
            raise CalibrationError("manifest media_ref values must be unique")
        seen_clip_ids.add(normalised.clip_id)
        seen_media_refs.add(normalised.media_ref)
        clips.append(normalised)

    if not clips:
        raise CalibrationError("manifest must contain at least one clip")
    _validate_split_lineages(clips)
    if require_full_plan:
        _validate_full_plan_counts(clips)
    return tuple(sorted(clips, key=lambda clip: clip.clip_id))


def calibration_manifest_hash(
    manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]]
) -> str:
    """Hash the exact identity fields used by the calibration evidence binder."""

    clips = validate_manifest(manifest)
    payload = [
        {
            "clip_id": clip.clip_id,
            "media_ref": clip.media_ref,
            "task": clip.task,
            "split": clip.split,
            "source_lineage_id": clip.source_lineage_id,
        }
        for clip in clips
    ]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deterministic_annotation_assignments(
    manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]], annotator_ids: Sequence[str]
) -> Dict[str, Tuple[str, ...]]:
    """Assign development once and held-out twice using a stable SHA-256 rule."""

    clips = validate_manifest(manifest)
    annotators = _validate_annotators(annotator_ids)
    assignments: Dict[str, Tuple[str, ...]] = {}
    for clip in clips:
        if clip.split == HELDOUT_SPLIT:
            assignments[clip.clip_id] = tuple(annotators)
        else:
            digest = hashlib.sha256(
                ("plumb-calibration-development-assignment-v1\x1f" + clip.clip_id).encode("utf-8")
            ).digest()
            assignments[clip.clip_id] = (annotators[int.from_bytes(digest[:8], "big") % len(annotators)],)
    return assignments


def blinded_annotation_rows(
    manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    annotator_id: str,
    annotator_ids: Sequence[str],
    assignments: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[Dict[str, Any]]:
    """Return one annotator's blank packet without experimental metadata."""

    clips = validate_manifest(manifest)
    annotators = _validate_annotators(annotator_ids)
    if annotator_id not in annotators:
        raise CalibrationError("annotator_id is not one of the two planned annotators")
    resolved_assignments = (
        _validate_assignments(clips, assignments, annotators)
        if assignments is not None
        else deterministic_annotation_assignments(clips, annotators)
    )
    packets: List[Dict[str, Any]] = []
    for clip in clips:
        if annotator_id not in resolved_assignments[clip.clip_id]:
            continue
        # Do not add split, lineage, policy, action, condition, backend, seed,
        # ground truth, gate, or protocol context to the blinded form.
        packets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "annotator_id": annotator_id,
                "clip_id": clip.clip_id,
                "media_ref": clip.media_ref,
                "task": clip.task,
                "integrity": "",
                "collision": "",
                "progress": "",
                "completion_evidence": "",
                "evidence_frame_indices": "",
                "observable_reason": "",
            }
        )
    return packets


def export_annotation_packets(
    manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    output_path: Union[str, Path],
    annotator_id: str,
    annotator_ids: Sequence[str],
    output_format: Optional[str] = None,
    assignments: Optional[Mapping[str, Sequence[str]]] = None,
) -> int:
    """Write a blinded JSONL or CSV packet and return its number of rows."""

    rows = blinded_annotation_rows(manifest, annotator_id, annotator_ids, assignments)
    path = Path(output_path)
    fmt = _resolve_format(path, output_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
                handle.write("\n")
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(BLINDED_EXPORT_FIELDS), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def import_annotations(
    input_path: Union[str, Path],
    manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    assignments: Optional[Mapping[str, Sequence[str]]] = None,
    annotator_ids: Optional[Sequence[str]] = None,
    expected_assignment: bool = True,
) -> Tuple[Annotation, ...]:
    """Read strict labels from JSONL or CSV; blank export fields are not labels."""

    path = Path(input_path)
    rows = _read_rows(path)
    return parse_annotations(
        rows,
        manifest,
        assignments=assignments,
        annotator_ids=annotator_ids,
        expected_assignment=expected_assignment,
    )


def parse_annotations(
    rows: Iterable[Mapping[str, Any]],
    manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    assignments: Optional[Mapping[str, Sequence[str]]] = None,
    annotator_ids: Optional[Sequence[str]] = None,
    expected_assignment: bool = True,
) -> Tuple[Annotation, ...]:
    """Parse labels without filling in a missing human judgement."""

    clips = validate_manifest(manifest)
    clips_by_id = {clip.clip_id: clip for clip in clips}
    resolved_assignments: Optional[Dict[str, Tuple[str, ...]]] = None
    if expected_assignment:
        if annotator_ids is None:
            raise CalibrationError("annotator_ids are required when validating human annotation assignments")
        annotators = _validate_annotators(annotator_ids)
        resolved_assignments = (
            _validate_assignments(clips, assignments, annotators)
            if assignments is not None
            else deterministic_annotation_assignments(clips, annotators)
        )

    parsed: List[Annotation] = []
    seen = set()
    for row_index, row in enumerate(rows):
        annotation = _parse_annotation_row(row, clips_by_id, row_index)
        key = (annotation.annotator_id, annotation.clip_id)
        if key in seen:
            raise CalibrationError("duplicate annotation for annotator_id and clip_id")
        seen.add(key)
        if resolved_assignments is not None and annotation.annotator_id not in resolved_assignments[annotation.clip_id]:
            raise CalibrationError("annotation is not assigned to this annotator")
        parsed.append(annotation)
    return tuple(sorted(parsed, key=lambda label: (label.clip_id, label.annotator_id)))


def validate_human_annotation_coverage(
    manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    annotations: Iterable[Annotation],
    annotator_ids: Sequence[str],
    assignments: Optional[Mapping[str, Sequence[str]]] = None,
    require_complete: bool = False,
) -> Dict[str, Any]:
    """Validate human assignments and return coverage without manufacturing rows."""

    clips = validate_manifest(manifest)
    annotators = _validate_annotators(annotator_ids)
    resolved_assignments = (
        _validate_assignments(clips, assignments, annotators)
        if assignments is not None
        else deterministic_annotation_assignments(clips, annotators)
    )
    labels = tuple(annotations)
    _validate_annotation_objects(labels, clips, resolved_assignments)
    expected_pairs = {
        (annotator_id, clip_id)
        for clip_id, assigned in resolved_assignments.items()
        for annotator_id in assigned
    }
    observed_pairs = {(label.annotator_id, label.clip_id) for label in labels}
    unexpected = observed_pairs - expected_pairs
    if unexpected:
        raise CalibrationError("annotation is not in the deterministic assignment plan")
    missing = expected_pairs - observed_pairs
    if require_complete and missing:
        raise CalibrationError("human annotations are incomplete")
    counts_by_split = {DEVELOPMENT_SPLIT: 0, HELDOUT_SPLIT: 0}
    clips_by_id = {clip.clip_id: clip for clip in clips}
    for _, clip_id in observed_pairs:
        counts_by_split[clips_by_id[clip_id].split] += 1
    expected_by_split = {
        split: sum(len(resolved_assignments[clip.clip_id]) for clip in clips if clip.split == split)
        for split in (DEVELOPMENT_SPLIT, HELDOUT_SPLIT)
    }
    return {
        "expected_annotations": len(expected_pairs),
        "received_annotations": len(observed_pairs),
        "coverage": _ratio(len(observed_pairs), len(expected_pairs)),
        "missing_annotations": len(missing),
        "by_split": {
            split: {
                "expected_annotations": expected_by_split[split],
                "received_annotations": counts_by_split[split],
                "coverage": _ratio(counts_by_split[split], expected_by_split[split]),
            }
            for split in (DEVELOPMENT_SPLIT, HELDOUT_SPLIT)
        },
    }


def build_calibration_report(
    manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    human_annotations: Iterable[Union[Annotation, Mapping[str, Any]]],
    annotator_ids: Sequence[str],
    judge_annotations: Optional[Iterable[Union[Annotation, Mapping[str, Any]]]] = None,
    judge_reports: Optional[Iterable[Any]] = None,
    protocol: Optional[Union[FrozenGateDProtocol, Mapping[str, Any]]] = None,
    tolerances: Optional[Union[GateDTolerances, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Summarize supplied calibration evidence and conservatively assess Gate D.

    Per-task metrics are always marked exploratory: the held-out plan has only
    10 clips per task.  The pooled metrics are the only possible inputs to the
    initial Gate-D decision.
    """

    clips = validate_manifest(manifest)
    annotators = _validate_annotators(annotator_ids)
    assignments = deterministic_annotation_assignments(clips, annotators)
    human_labels = _coerce_annotations(human_annotations, clips, assignments, annotators, True)
    human_coverage = validate_human_annotation_coverage(clips, human_labels, annotators, assignments)

    heldout_clips = tuple(clip for clip in clips if clip.split == HELDOUT_SPLIT)
    labels_by_clip: Dict[str, Dict[str, Annotation]] = defaultdict(dict)
    for label in human_labels:
        labels_by_clip[label.clip_id][label.annotator_id] = label
    pairs_by_task: Dict[str, List[Tuple[Annotation, Annotation]]] = {task: [] for task in TASKS}
    all_pairs: List[Tuple[Annotation, Annotation]] = []
    consensus_by_clip: Dict[str, Dict[str, Any]] = {}
    for clip in heldout_clips:
        per_clip = labels_by_clip.get(clip.clip_id, {})
        if all(annotator in per_clip for annotator in annotators):
            first, second = per_clip[annotators[0]], per_clip[annotators[1]]
            all_pairs.append((first, second))
            pairs_by_task[clip.task].append((first, second))
            consensus_by_clip[clip.clip_id] = _consensus(first, second)
        else:
            consensus_by_clip[clip.clip_id] = _unresolved_consensus()

    agreement = {
        "pooled": _agreement_metrics(all_pairs),
        "task_exploratory": {
            task: {"exploratory": True, **_agreement_metrics(pairs_by_task[task])} for task in TASKS
        },
    }
    consensus_summary = _consensus_summary(consensus_by_clip, heldout_clips)
    judge_labels = _coerce_judge_annotations(judge_annotations, clips)
    judge_comparison = _judge_human_comparison(heldout_clips, consensus_by_clip, judge_labels)
    judge_comparison["descriptive_only"] = True
    protocol_value = _coerce_protocol(protocol)
    tolerance_value = _coerce_tolerances(tolerances)
    primary_judge_evidence = validate_primary_judge_evidence(
        clips,
        consensus_by_clip,
        protocol_value,
        judge_reports,
    )
    gate_d = evaluate_gate_d(
        heldout_overlap=len(all_pairs),
        human_annotation_coverage=human_coverage["coverage"],
        human_consensus_coverage=consensus_summary["binary_success_coverage"],
        primary_judge_evidence=primary_judge_evidence,
        protocol=protocol_value,
        tolerances=tolerance_value,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": {
            "total_clips": len(clips),
            "development_clips": sum(1 for clip in clips if clip.split == DEVELOPMENT_SPLIT),
            "heldout_clips": len(heldout_clips),
            "tasks": list(TASKS),
            "lineage_split_disjoint": True,
        },
        "human_annotation_coverage": human_coverage,
        "human_human_agreement": agreement,
        "heldout_consensus": consensus_summary,
        "judge_vs_human": judge_comparison,
        "primary_judge_evidence": primary_judge_evidence,
        "gate_d": gate_d,
    }


def evaluate_gate_d(
    *,
    heldout_overlap: int,
    human_annotation_coverage: Optional[float],
    human_consensus_coverage: Optional[float],
    primary_judge_evidence: Mapping[str, Any],
    protocol: Optional[FrozenGateDProtocol],
    tolerances: Optional[GateDTolerances],
) -> Dict[str, Any]:
    """Return ``pass`` only with all required evidence and frozen thresholds."""

    if protocol is None or tolerances is None:
        missing = []
        if protocol is None:
            missing.append("frozen_protocol")
        if tolerances is None:
            missing.append("preregistered_tolerances")
        return {
            "status": "not_evaluable",
            "passed": False,
            "reason_codes": missing,
            "protocol": None if protocol is None else asdict(protocol),
            "tolerances": None if tolerances is None else asdict(tolerances),
        }
    if not primary_judge_evidence.get("available", False):
        return {
            "status": "blocked",
            "passed": False,
            "reason_codes": [primary_judge_evidence.get("reason_code", "primary_judge_evidence_missing")],
            "protocol": asdict(protocol),
            "tolerances": asdict(tolerances),
        }

    judge_comparison = primary_judge_evidence["judge_comparison"]

    reasons: List[str] = []
    if heldout_overlap < tolerances.minimum_heldout_overlap:
        reasons.append("heldout_overlap_below_minimum")
    if human_annotation_coverage is None or human_annotation_coverage < tolerances.minimum_human_annotation_coverage:
        reasons.append("human_annotation_coverage_below_minimum")
    if human_consensus_coverage is None or human_consensus_coverage < tolerances.minimum_human_consensus_coverage:
        reasons.append("human_consensus_coverage_below_minimum")
    comparison_coverage = judge_comparison.get("comparison_coverage")
    if comparison_coverage is None or comparison_coverage < tolerances.minimum_judge_comparison_coverage:
        reasons.append("judge_comparison_coverage_below_minimum")
    sensitivity = judge_comparison.get("sensitivity")
    specificity = judge_comparison.get("specificity")
    leniency = judge_comparison.get("leniency_offset")
    if sensitivity is None:
        reasons.append("sensitivity_undefined")
    elif sensitivity < tolerances.minimum_sensitivity:
        reasons.append("sensitivity_below_minimum")
    if specificity is None:
        reasons.append("specificity_undefined")
    elif specificity < tolerances.minimum_specificity:
        reasons.append("specificity_below_minimum")
    if leniency is None:
        reasons.append("leniency_offset_undefined")
    elif abs(leniency) > tolerances.maximum_absolute_leniency_offset:
        reasons.append("leniency_offset_above_maximum")
    return {
        "status": "pass" if not reasons else "fail",
        "passed": not reasons,
        "reason_codes": reasons,
        "protocol": asdict(protocol),
        "tolerances": asdict(tolerances),
        "metrics": {
            "heldout_overlap": heldout_overlap,
            "human_annotation_coverage": human_annotation_coverage,
            "human_consensus_coverage": human_consensus_coverage,
            "judge_comparison_coverage": comparison_coverage,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "leniency_offset": leniency,
        },
    }


def _parse_manifest_row(raw: Mapping[str, Any], index: int) -> ClipManifestRow:
    if not isinstance(raw, Mapping):
        raise CalibrationError("manifest row {0} must be an object".format(index))
    required = ("clip_id", "media_ref", "task", "split", "source_lineage_id")
    missing = [field for field in required if field not in raw]
    if missing:
        raise CalibrationError("manifest row {0} is missing {1}".format(index, ", ".join(missing)))
    return ClipManifestRow(
        clip_id=_strict_string(raw["clip_id"], "manifest clip_id"),
        media_ref=_strict_string(raw["media_ref"], "manifest media_ref"),
        task=_strict_string(raw["task"], "manifest task"),
        split=_strict_string(raw["split"], "manifest split"),
        source_lineage_id=_strict_string(raw["source_lineage_id"], "manifest source_lineage_id"),
    )


def _normalise_split(value: str) -> str:
    if value == "dev":
        return DEVELOPMENT_SPLIT
    if value not in (DEVELOPMENT_SPLIT, HELDOUT_SPLIT):
        raise CalibrationError("manifest split must be development/dev or heldout")
    return value


def _validate_split_lineages(clips: Sequence[ClipManifestRow]) -> None:
    development = {clip.source_lineage_id for clip in clips if clip.split == DEVELOPMENT_SPLIT}
    heldout = {clip.source_lineage_id for clip in clips if clip.split == HELDOUT_SPLIT}
    overlap = sorted(development & heldout)
    if overlap:
        raise CalibrationError("source lineages cross development and heldout splits: {0}".format(", ".join(overlap)))


def _validate_full_plan_counts(clips: Sequence[ClipManifestRow]) -> None:
    if len(clips) != 150:
        raise CalibrationError("full calibration manifest must contain exactly 150 clips")
    counts: Dict[Tuple[str, str], int] = Counter((clip.task, clip.split) for clip in clips)
    for task in TASKS:
        for split, expected in FULL_PLAN_COUNTS.items():
            observed = counts[(task, split)]
            if observed != expected:
                raise CalibrationError(
                    "full calibration manifest requires {0} {1} clips for {2}; found {3}".format(
                        expected, split, task, observed
                    )
                )


def _validate_annotators(annotator_ids: Sequence[str]) -> Tuple[str, str]:
    if len(annotator_ids) != 2:
        raise CalibrationError("exactly two annotator IDs are required for heldout overlap")
    values = tuple(_strict_string(value, "annotator_id") for value in annotator_ids)
    if values[0] == values[1]:
        raise CalibrationError("annotator IDs must be distinct")
    return tuple(sorted(values))  # type: ignore[return-value]


def _validate_assignments(
    clips: Sequence[ClipManifestRow],
    assignments: Mapping[str, Sequence[str]],
    annotators: Sequence[str],
) -> Dict[str, Tuple[str, ...]]:
    expected_ids = {clip.clip_id for clip in clips}
    if set(assignments) != expected_ids:
        raise CalibrationError("assignment keys must exactly match manifest clip IDs")
    resolved: Dict[str, Tuple[str, ...]] = {}
    for clip in clips:
        assigned = tuple(assignments[clip.clip_id])
        if any(annotator not in annotators for annotator in assigned):
            raise CalibrationError("assignment contains an unknown annotator")
        if len(set(assigned)) != len(assigned):
            raise CalibrationError("assignment contains a duplicate annotator")
        expected_count = 2 if clip.split == HELDOUT_SPLIT else 1
        if len(assigned) != expected_count:
            raise CalibrationError("heldout clips need two assignments and development clips need one")
        if clip.split == HELDOUT_SPLIT and set(assigned) != set(annotators):
            raise CalibrationError("every heldout clip must be assigned to both annotators")
        resolved[clip.clip_id] = assigned
    return resolved


def _parse_annotation_row(
    raw: Mapping[str, Any], clips_by_id: Mapping[str, ClipManifestRow], row_index: int
) -> Annotation:
    if not isinstance(raw, Mapping):
        raise CalibrationError("annotation row {0} must be an object".format(row_index))
    unexpected = set(raw) - _IMPORT_ALLOWED_FIELDS
    forbidden = set(raw) & _FORBIDDEN_BLIND_FIELDS
    if forbidden:
        raise CalibrationError("annotation contains blinded metadata: {0}".format(", ".join(sorted(forbidden))))
    if unexpected:
        raise CalibrationError("annotation contains unsupported fields: {0}".format(", ".join(sorted(unexpected))))
    if "completion" in raw and "completion_evidence" in raw:
        raise CalibrationError("annotation may use completion or completion_evidence, not both")
    if "schema_version" in raw and _strict_schema_version(raw["schema_version"]) != SCHEMA_VERSION:
        raise CalibrationError("annotation schema_version must match the calibration schema")
    completion_key = "completion_evidence" if "completion_evidence" in raw else "completion"
    required = ("clip_id", "annotator_id", "integrity", "collision", "progress", completion_key)
    missing = [field for field in required if field not in raw]
    if missing:
        raise CalibrationError("annotation row {0} is missing {1}".format(row_index, ", ".join(missing)))
    clip_id = _strict_string(raw["clip_id"], "annotation clip_id")
    if clip_id not in clips_by_id:
        raise CalibrationError("annotation references a clip not in the manifest")
    clip = clips_by_id[clip_id]
    if "task" in raw and _strict_string(raw["task"], "annotation task") != clip.task:
        raise CalibrationError("annotation task does not match manifest")
    if "media_ref" in raw and _strict_string(raw["media_ref"], "annotation media_ref") != clip.media_ref:
        raise CalibrationError("annotation media_ref does not match manifest")
    integrity = _strict_enum(raw["integrity"], INTEGRITY_VALUES, "integrity")
    collision = _strict_enum(raw["collision"], COLLISION_VALUES, "collision")
    progress = _strict_progress(raw["progress"])
    completion = _strict_enum(raw[completion_key], COMPLETION_VALUES, "completion_evidence")
    if completion == "met" and progress != 5:
        raise CalibrationError("completion_evidence='met' requires progress=5")
    if completion == "not_met" and progress > 4:
        raise CalibrationError("completion_evidence='not_met' requires progress from 0 through 4")
    return Annotation(
        clip_id=clip_id,
        annotator_id=_strict_string(raw["annotator_id"], "annotation annotator_id"),
        integrity=integrity,
        collision=collision,
        progress=progress,
        completion_evidence=completion,
        evidence_frame_indices=_parse_evidence_frame_indices(raw.get("evidence_frame_indices", [])),
        observable_reason=_parse_observable_reason(raw.get("observable_reason", "")),
    )


def _validate_annotation_objects(
    labels: Sequence[Annotation],
    clips: Sequence[ClipManifestRow],
    assignments: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    clips_by_id = {clip.clip_id: clip for clip in clips}
    seen = set()
    for label in labels:
        if not isinstance(label, Annotation):
            raise CalibrationError("annotations must be Annotation objects or parsed mappings")
        if label.clip_id not in clips_by_id:
            raise CalibrationError("annotation references a clip not in the manifest")
        key = (label.annotator_id, label.clip_id)
        if key in seen:
            raise CalibrationError("duplicate annotation for annotator_id and clip_id")
        seen.add(key)
        if assignments is not None and label.annotator_id not in assignments[label.clip_id]:
            raise CalibrationError("annotation is not in the deterministic assignment plan")


def _coerce_annotations(
    labels: Iterable[Union[Annotation, Mapping[str, Any]]],
    clips: Sequence[ClipManifestRow],
    assignments: Mapping[str, Sequence[str]],
    annotators: Sequence[str],
    expected_assignment: bool,
) -> Tuple[Annotation, ...]:
    values = list(labels)
    if all(isinstance(value, Annotation) for value in values):
        typed = tuple(values)  # type: ignore[assignment]
        _validate_annotation_objects(typed, clips, assignments if expected_assignment else None)
        return tuple(sorted(typed, key=lambda label: (label.clip_id, label.annotator_id)))
    if any(isinstance(value, Annotation) for value in values):
        raise CalibrationError("do not mix Annotation objects and raw annotation mappings")
    return parse_annotations(
        values, clips, assignments=assignments, annotator_ids=annotators, expected_assignment=expected_assignment
    )


def _coerce_judge_annotations(
    labels: Optional[Iterable[Union[Annotation, Mapping[str, Any]]]], clips: Sequence[ClipManifestRow]
) -> Dict[str, Annotation]:
    if labels is None:
        return {}
    values = list(labels)
    if all(isinstance(value, Annotation) for value in values):
        typed = tuple(values)  # type: ignore[assignment]
        _validate_annotation_objects(typed, clips)
    elif any(isinstance(value, Annotation) for value in values):
        raise CalibrationError("do not mix Annotation objects and raw judge mappings")
    else:
        typed = parse_annotations(values, clips, expected_assignment=False)
    by_clip: Dict[str, Annotation] = {}
    for label in typed:
        if label.clip_id in by_clip:
            raise CalibrationError("judge labels must contain at most one aggregated label per clip")
        by_clip[label.clip_id] = label
    return by_clip


def _consensus(first: Annotation, second: Annotation) -> Dict[str, Any]:
    integrity = first.integrity if first.integrity == second.integrity else None
    collision = first.collision if first.collision == second.collision else None
    progress = first.progress if first.progress == second.progress else None
    completion = (
        first.completion_evidence if first.completion_evidence == second.completion_evidence else None
    )
    binary: Optional[bool] = None
    if integrity == "intact" and progress is not None and completion in ("met", "not_met"):
        binary = completion == "met"
    return {
        "integrity": integrity,
        "collision": collision,
        "progress": progress,
        "completion_evidence": completion,
        "binary_success": binary,
    }


def _unresolved_consensus() -> Dict[str, Any]:
    return {
        "integrity": None,
        "collision": None,
        "progress": None,
        "completion_evidence": None,
        "binary_success": None,
    }


def _consensus_summary(
    consensus_by_clip: Mapping[str, Mapping[str, Any]], heldout_clips: Sequence[ClipManifestRow]
) -> Dict[str, Any]:
    field_counts = {
        field: sum(1 for value in consensus_by_clip.values() if value.get(field) is not None)
        for field in ("integrity", "collision", "progress", "completion_evidence", "binary_success")
    }
    total = len(heldout_clips)
    return {
        "heldout_clips": total,
        "field_consensus_counts": field_counts,
        "field_consensus_coverage": {field: _ratio(count, total) for field, count in field_counts.items()},
        "binary_success_consensus_count": field_counts["binary_success"],
        "binary_success_coverage": _ratio(field_counts["binary_success"], total),
        "unresolved_binary_success_count": total - field_counts["binary_success"],
    }


def _agreement_metrics(pairs: Sequence[Tuple[Annotation, Annotation]]) -> Dict[str, Any]:
    return {
        "paired_clips": len(pairs),
        "integrity": _categorical_agreement(
            [(first.integrity, second.integrity) for first, second in pairs], INTEGRITY_VALUES
        ),
        "collision": _categorical_agreement(
            [(first.collision, second.collision) for first, second in pairs], COLLISION_VALUES
        ),
        "completion_evidence": _categorical_agreement(
            [(first.completion_evidence, second.completion_evidence) for first, second in pairs], COMPLETION_VALUES
        ),
        "progress": _weighted_progress_agreement([(first.progress, second.progress) for first, second in pairs]),
        "binary_success": _binary_success_agreement(pairs),
    }


def _binary_success_agreement(pairs: Sequence[Tuple[Annotation, Annotation]]) -> Dict[str, Any]:
    """Cohen's kappa on the two-category visible-completion label.

    Only pairs where *both* annotators were decisive contribute.  An indecisive
    label is not recoded as a failure, so ``n`` is smaller than the paired clip
    count and ``indecisive_pairs`` records the difference rather than hiding it.
    """

    decisive = [
        (first.binary_success, second.binary_success)
        for first, second in pairs
        if first.binary_success is not None and second.binary_success is not None
    ]
    metrics = _categorical_agreement(
        [("met" if first else "not_met", "met" if second else "not_met") for first, second in decisive],
        ("met", "not_met"),
    )
    metrics["indecisive_pairs"] = len(pairs) - len(decisive)
    metrics["paired_clips"] = len(pairs)
    return metrics


def _categorical_agreement(pairs: Sequence[Tuple[str, str]], categories: Sequence[str]) -> Dict[str, Any]:
    n = len(pairs)
    if n == 0:
        return {"n": 0, "raw_agreement": None, "kappa": None}
    raw = sum(1 for first, second in pairs if first == second) / n
    first_counts = Counter(first for first, _ in pairs)
    second_counts = Counter(second for _, second in pairs)
    expected = sum((first_counts[value] / n) * (second_counts[value] / n) for value in categories)
    denominator = 1 - expected
    kappa = None if math.isclose(denominator, 0.0, abs_tol=1e-15) else (raw - expected) / denominator
    return {"n": n, "raw_agreement": raw, "kappa": kappa}


def _weighted_progress_agreement(pairs: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    """Quadratic-weighted Cohen's kappa over the frozen 0--5 ordinal scale."""

    n = len(pairs)
    if n == 0:
        return {"n": 0, "raw_agreement": None, "weighted_kappa": None, "weighting": "quadratic"}
    raw = sum(1 for first, second in pairs if first == second) / n
    first_counts = Counter(first for first, _ in pairs)
    second_counts = Counter(second for _, second in pairs)

    def weight(first: int, second: int) -> float:
        return 1.0 - ((first - second) / 5.0) ** 2

    observed = sum(weight(first, second) for first, second in pairs) / n
    expected = sum(
        weight(first, second) * (first_counts[first] / n) * (second_counts[second] / n)
        for first in range(6)
        for second in range(6)
    )
    denominator = 1 - expected
    kappa = None if math.isclose(denominator, 0.0, abs_tol=1e-15) else (observed - expected) / denominator
    return {"n": n, "raw_agreement": raw, "weighted_kappa": kappa, "weighting": "quadratic"}


def _judge_human_comparison(
    heldout_clips: Sequence[ClipManifestRow],
    consensus_by_clip: Mapping[str, Mapping[str, Any]],
    judge_by_clip: Mapping[str, Annotation],
) -> Dict[str, Any]:
    return _binary_judge_human_comparison(
        heldout_clips,
        consensus_by_clip,
        {clip_id: label.binary_success for clip_id, label in judge_by_clip.items()},
        available=bool(judge_by_clip),
        unavailable_reason="No aggregated judge labels were supplied; human-only evidence cannot measure judge accuracy.",
    )


def validate_primary_judge_evidence(
    manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    consensus_by_clip: Mapping[str, Mapping[str, Any]],
    protocol: Optional[FrozenGateDProtocol],
    judge_reports: Optional[Iterable[Any]],
) -> Dict[str, Any]:
    """Bind and re-aggregate primary judge reports before a Gate-D decision.

    This is deliberately stricter than the descriptive ``judge_annotations``
    input. Human-labelled or copied ``Annotation`` values never substitute for
    raw five-sample judge reports. The checks bind declared immutable evidence
    and a declared trusted producer; they do **not** cryptographically prove a
    model ran or establish scientific validity.
    """

    clips = validate_manifest(manifest)
    heldout_clips = tuple(clip for clip in clips if clip.split == HELDOUT_SPLIT)
    if protocol is None:
        return _blocked_primary_evidence("frozen_protocol_missing")
    frozen = protocol.evidence_manifest
    if frozen is None:
        return _blocked_primary_evidence("frozen_judge_evidence_manifest_missing")
    if not _hash_equal(frozen.calibration_manifest_hash, calibration_manifest_hash(clips)):
        return _blocked_primary_evidence("calibration_manifest_hash_mismatch")
    expected_video_hashes = {clip.clip_id: frozen.heldout_video_hashes.get(clip.clip_id) for clip in heldout_clips}
    if set(frozen.heldout_video_hashes) != {clip.clip_id for clip in heldout_clips} or any(
        value is None for value in expected_video_hashes.values()
    ):
        return _blocked_primary_evidence("frozen_heldout_video_hashes_incomplete")
    if judge_reports is None:
        return _blocked_primary_evidence("raw_judge_reports_missing")

    reports = list(judge_reports)
    if len(reports) != len(heldout_clips):
        return _blocked_primary_evidence("raw_judge_report_count_mismatch")
    observed_votes: Dict[str, Optional[bool]] = {}
    heldout_tasks = {clip.clip_id: clip.task for clip in heldout_clips}
    try:
        for report in reports:
            report_map = _judge_report_mapping(report)
            clip_id, vote = _bind_and_reaggregate_judge_report(report_map, frozen, protocol, heldout_tasks)
            if clip_id in observed_votes:
                raise CalibrationError("duplicate raw judge report clip_id")
            observed_votes[clip_id] = vote
    except CalibrationError as error:
        return _blocked_primary_evidence("raw_judge_evidence_invalid", detail=str(error))
    if set(observed_votes) != {clip.clip_id for clip in heldout_clips}:
        return _blocked_primary_evidence("raw_judge_reports_do_not_cover_heldout_manifest")
    for clip_id, expected_hash in expected_video_hashes.items():
        # The direct report check happens in _bind..., and this keeps the
        # expected binding explicit in the returned summary as well.
        if not _is_hash(expected_hash):
            return _blocked_primary_evidence("frozen_heldout_video_hash_invalid")
    comparison = _binary_judge_human_comparison(
        heldout_clips,
        consensus_by_clip,
        observed_votes,
        available=True,
        unavailable_reason=None,
    )
    return {
        "available": True,
        "binding": "immutable identifiers and declared producer only; not a cryptographic execution attestation",
        "reaggregated_report_count": len(observed_votes),
        "judge_comparison": comparison,
        "reason_code": None,
    }


def _blocked_primary_evidence(reason_code: str, detail: Optional[str] = None) -> Dict[str, Any]:
    return {
        "available": False,
        "reason_code": reason_code,
        "detail": detail,
        "binding": "unavailable",
        "reaggregated_report_count": 0,
        "judge_comparison": None,
    }


def _judge_report_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        mapped = as_dict()
        if isinstance(mapped, Mapping):
            return mapped
    raise CalibrationError("raw judge report must be a mapping or expose as_dict()")


def _bind_and_reaggregate_judge_report(
    report: Mapping[str, Any],
    frozen: FrozenJudgeEvidenceManifest,
    protocol: FrozenGateDProtocol,
    heldout_tasks: Mapping[str, str],
) -> Tuple[str, Optional[bool]]:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CalibrationError("raw judge report provenance is missing")
    clip_id = _required_mapping_string(provenance, "clip_id", "judge provenance")
    _require_exact_string(provenance, "calibration_manifest_hash", frozen.calibration_manifest_hash, "judge provenance")
    _require_exact_string(provenance, "task_registry_id", frozen.task_registry_id, "judge provenance")
    _require_exact_hash(provenance, "task_registry_hash", frozen.task_registry_hash, "judge provenance")
    task_id = heldout_tasks.get(clip_id)
    if task_id is None:
        raise CalibrationError("judge report clip_id is not in the heldout manifest")
    _require_exact_string(provenance, "task_id", task_id, "judge provenance")
    _require_exact_string(provenance, "rubric_hash", protocol.rubric_hash, "judge provenance")
    _require_exact_string(provenance, "sampling_hash", protocol.sampling_hash, "judge provenance")
    _require_exact_string(provenance, "protocol_id", frozen.protocol_id, "judge provenance")
    expected_video_hash = frozen.heldout_video_hashes.get(clip_id)
    if expected_video_hash is None:
        raise CalibrationError("judge report clip_id is not in the frozen heldout manifest")
    evidence_hashes = provenance.get("evidence_hashes")
    if not isinstance(evidence_hashes, Mapping):
        raise CalibrationError("judge provenance evidence_hashes is missing")
    _require_exact_hash(evidence_hashes, "video_hash", expected_video_hash, "judge provenance evidence_hashes")
    model = provenance.get("model")
    if not isinstance(model, Mapping):
        raise CalibrationError("judge provenance model identity is missing")
    for field_name, expected in (
        ("id", frozen.model_id),
        ("model_revision", frozen.model_revision),
        ("processor_revision", frozen.processor_revision),
        ("transformers_version", frozen.transformers_version),
        ("runtime_lock_id", frozen.runtime_lock_id),
        ("asset_manifest_id", frozen.asset_manifest_id),
    ):
        _require_exact_string(model, field_name, expected, "judge provenance model")
    artifact_hashes = provenance.get("artifact_hashes")
    expected_artifact_hashes = frozen.heldout_artifact_hashes.get(clip_id)
    if (
        not isinstance(artifact_hashes, Mapping)
        or expected_artifact_hashes is None
        or set(artifact_hashes) != set(expected_artifact_hashes)
    ):
        raise CalibrationError("judge provenance artifact_hashes do not match the frozen manifest")
    for field_name, expected in expected_artifact_hashes.items():
        _require_exact_hash(artifact_hashes, field_name, expected, "judge provenance artifact_hashes")
    producer = provenance.get("producer")
    if not isinstance(producer, Mapping):
        raise CalibrationError("judge provenance producer is missing")
    for field_name, expected in (
        ("name", frozen.trusted_producer.name),
        ("version", frozen.trusted_producer.version),
    ):
        _require_exact_string(producer, field_name, expected, "judge provenance producer")
    _require_exact_hash(producer, "source_hash", frozen.trusted_producer.source_hash, "judge provenance producer")
    trust = provenance.get("trust")
    if not isinstance(trust, Mapping) or trust.get("gate_d_eligible") is not True:
        raise CalibrationError("judge provenance does not declare gate_d_eligible")
    if trust.get("test_mode") is not False:
        raise CalibrationError("test or unspecified producer evidence cannot qualify Gate D")
    if trust.get("gate_d_unavailable_reasons") not in (None, [], ()):
        raise CalibrationError("judge provenance declares an unavailable reason")
    _validate_primary_sampling(report.get("sampling"))
    samples = report.get("raw_judge_samples", report.get("samples"))
    expected_binary, expected_progress, expected_status, expected_agreeing = _reaggregate_raw_judge_samples(samples)
    if report.get("binary_success") is not expected_binary:
        raise CalibrationError("raw judge report binary_success does not match reaggregation")
    reported_progress = report.get("progress")
    if expected_progress is None:
        if reported_progress is not None:
            raise CalibrationError("raw judge report progress does not match reaggregation")
    elif (
        isinstance(reported_progress, bool)
        or not isinstance(reported_progress, (int, float))
        or not math.isclose(float(reported_progress), expected_progress, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise CalibrationError("raw judge report progress does not match reaggregation")
    if report.get("judge_status") != expected_status:
        raise CalibrationError("raw judge report judge_status does not match reaggregation")
    if report.get("agreeing_samples") != expected_agreeing:
        raise CalibrationError("raw judge report agreeing_samples does not match reaggregation")
    if expected_status == "evaluable" and report.get("missing_reason") is not None:
        raise CalibrationError("evaluable judge report must have null missing_reason")
    if expected_status == "unknown" and not _nonempty_string(report.get("missing_reason")):
        raise CalibrationError("unevaluable judge report must retain a missing_reason")
    return clip_id, expected_binary


def _validate_primary_sampling(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise CalibrationError("raw judge report sampling configuration is missing")
    expected = {
        "sample_count": 5,
        "quorum": 3,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_new_tokens": 512,
        "retries_per_sample": 1,
    }
    if not set(expected).issubset(value):
        raise CalibrationError("raw judge report sampling fields do not match the frozen primary protocol")
    for field_name, expected_value in expected.items():
        actual = value[field_name]
        if isinstance(expected_value, float):
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or float(actual) != expected_value:
                raise CalibrationError("raw judge report sampling configuration does not match the primary protocol")
        elif actual != expected_value or isinstance(actual, bool):
            raise CalibrationError("raw judge report sampling configuration does not match the primary protocol")


def _reaggregate_raw_judge_samples(value: Any) -> Tuple[Optional[bool], Optional[float], str, int]:
    if not isinstance(value, list) or len(value) != 5:
        raise CalibrationError("raw_judge_samples must contain exactly five sample reports")
    sample_indices = set()
    seeds = set()
    decisive: Dict[bool, List[int]] = {True: [], False: []}
    parsed_count = 0
    saw_refusal = False
    for sample in value:
        if not isinstance(sample, Mapping):
            raise CalibrationError("raw judge sample report must be an object")
        sample_index = sample.get("sample_index")
        seed = sample.get("seed")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index in sample_indices:
            raise CalibrationError("raw judge sample indexes must be unique integers")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in seeds:
            raise CalibrationError("raw judge sample seeds must be unique integers")
        sample_indices.add(sample_index)
        seeds.add(seed)
        attempts = sample.get("attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2 or not all(
            isinstance(attempt, Mapping) for attempt in attempts
        ):
            raise CalibrationError("each raw judge sample needs one or two raw attempts")
        final = attempts[-1]
        parsed = final.get("parsed")
        if parsed is None:
            if final.get("failure_reason") == "judge_refusal":
                saw_refusal = True
            continue
        if not isinstance(final.get("raw_output"), str) or not final.get("raw_output"):
            raise CalibrationError("a parsed judge sample must retain its raw_output")
        binary, progress = _parse_raw_judge_sample(parsed)
        parsed_count += 1
        if binary is not None:
            decisive[binary].append(progress)  # type: ignore[arg-type]
    if set(sample_indices) != set(range(5)):
        raise CalibrationError("raw judge sample indexes must be exactly 0 through 4")
    winner: Optional[bool] = None
    for candidate in (True, False):
        if len(decisive[candidate]) >= 3:
            winner = candidate
            break
    if winner is not None:
        values = sorted(decisive[winner])
        middle = len(values) // 2
        progress = float(values[middle]) if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
        return winner, progress, "evaluable", len(values)
    agreeing = max(len(decisive[True]), len(decisive[False]))
    # The exact diagnostic reason is produced by the judge worker. Requiring a
    # non-null summary reason above preserves it without trying to repair it.
    del parsed_count, saw_refusal
    return None, None, "unknown", agreeing


def _parse_raw_judge_sample(value: Any) -> Tuple[Optional[bool], Optional[int]]:
    if not isinstance(value, Mapping):
        raise CalibrationError("parsed raw judge sample must be an object")
    allowed = {
        "integrity",
        "collision",
        "progress",
        "completion_evidence",
        "evidence_frame_indices",
        "observable_reasons",
    }
    if set(value) != allowed:
        raise CalibrationError("parsed raw judge sample does not match the frozen rubric schema")
    integrity = _strict_enum(value["integrity"], INTEGRITY_VALUES, "raw judge integrity")
    _strict_enum(value["collision"], COLLISION_VALUES, "raw judge collision")
    completion = _strict_enum(value["completion_evidence"], COMPLETION_VALUES, "raw judge completion_evidence")
    progress_value = value["progress"]
    if progress_value is not None:
        progress = _strict_progress(progress_value)
    else:
        progress = None
    if completion == "met" and progress != 5:
        raise CalibrationError("raw judge met completion requires progress=5")
    if completion == "not_met" and (progress is None or progress > 4):
        raise CalibrationError("raw judge not_met completion requires progress 0 through 4")
    frames = value["evidence_frame_indices"]
    if not isinstance(frames, list) or not frames or any(
        isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < 16 for frame in frames
    ):
        raise CalibrationError("raw judge evidence_frame_indices must be non-empty indexes 0 through 15")
    if not _nonempty_string(value["observable_reasons"]):
        raise CalibrationError("raw judge observable_reasons must be non-empty")
    if integrity != "intact" or progress is None or completion == "uncertain":
        return None, None
    return completion == "met", progress


def _binary_judge_human_comparison(
    heldout_clips: Sequence[ClipManifestRow],
    consensus_by_clip: Mapping[str, Mapping[str, Any]],
    judge_votes: Mapping[str, Optional[bool]],
    *,
    available: bool,
    unavailable_reason: Optional[str],
) -> Dict[str, Any]:
    total = len(heldout_clips)
    if not available:
        return {
            "available": False,
            "reason": unavailable_reason,
            "heldout_clips": total,
            "human_binary_consensus_count": sum(
                1 for value in consensus_by_clip.values() if value.get("binary_success") is not None
            ),
            "judge_decisive_count": 0,
            "comparable_count": 0,
            "comparison_coverage": 0.0,
            "null_coverage": {"human_unresolved": None, "judge_unevaluable": None, "either_unavailable": None},
            "confusion_matrix": None,
            "sensitivity": None,
            "specificity": None,
            "leniency_offset": None,
        }
    human_known = 0
    judge_known = 0
    comparable: List[Tuple[bool, bool]] = []
    human_unresolved = 0
    judge_unevaluable = 0
    either_unavailable = 0
    for clip in heldout_clips:
        human = consensus_by_clip[clip.clip_id].get("binary_success")
        judge = judge_votes.get(clip.clip_id)
        if human is None:
            human_unresolved += 1
        else:
            human_known += 1
        if judge is None:
            judge_unevaluable += 1
        else:
            judge_known += 1
        if human is None or judge is None:
            either_unavailable += 1
        else:
            comparable.append((bool(human), bool(judge)))
    true_positive = sum(1 for human, judge in comparable if human and judge)
    false_negative = sum(1 for human, judge in comparable if human and not judge)
    false_positive = sum(1 for human, judge in comparable if not human and judge)
    true_negative = sum(1 for human, judge in comparable if not human and not judge)
    sensitivity = _ratio_or_none(true_positive, true_positive + false_negative)
    specificity = _ratio_or_none(true_negative, true_negative + false_positive)
    human_positive = sum(1 for human, _ in comparable if human)
    judge_positive = sum(1 for _, judge in comparable if judge)
    return {
        "available": True,
        "heldout_clips": total,
        "human_binary_consensus_count": human_known,
        "judge_decisive_count": judge_known,
        "comparable_count": len(comparable),
        "comparison_coverage": _ratio(len(comparable), total),
        "null_coverage": {
            "human_unresolved": _ratio(human_unresolved, total),
            "judge_unevaluable": _ratio(judge_unevaluable, total),
            "either_unavailable": _ratio(either_unavailable, total),
        },
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "sensitivity": sensitivity,
        "specificity": specificity,
        "leniency_offset": _ratio(judge_positive, len(comparable)) - _ratio(human_positive, len(comparable))
        if comparable
        else None,
    }


def _coerce_protocol(value: Optional[Union[FrozenGateDProtocol, Mapping[str, Any]]]) -> Optional[FrozenGateDProtocol]:
    if value is None:
        return None
    if isinstance(value, FrozenGateDProtocol):
        return value
    if not isinstance(value, Mapping):
        raise CalibrationError("protocol must be a FrozenGateDProtocol or object")
    try:
        data = dict(value)
        evidence = data.get("evidence_manifest")
        if isinstance(evidence, Mapping):
            evidence_data = dict(evidence)
            producer = evidence_data.get("trusted_producer")
            if isinstance(producer, Mapping):
                evidence_data["trusted_producer"] = TrustedJudgeProducer(**dict(producer))
            data["evidence_manifest"] = FrozenJudgeEvidenceManifest(**evidence_data)
        return FrozenGateDProtocol(**data)
    except (TypeError, CalibrationError) as error:
        raise CalibrationError("protocol does not match FrozenGateDProtocol") from error


def _coerce_tolerances(value: Optional[Union[GateDTolerances, Mapping[str, Any]]]) -> Optional[GateDTolerances]:
    if value is None:
        return None
    if isinstance(value, GateDTolerances):
        return value
    if not isinstance(value, Mapping):
        raise CalibrationError("tolerances must be a GateDTolerances or object")
    try:
        return GateDTolerances(**dict(value))
    except TypeError as error:
        raise CalibrationError("tolerances do not match GateDTolerances") from error


def _read_rows(path: Path) -> List[Mapping[str, Any]]:
    if not path.is_file():
        raise CalibrationError("annotation input does not exist: {0}".format(path))
    fmt = _resolve_format(path, None)
    rows: List[Mapping[str, Any]] = []
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CalibrationError("invalid JSONL on line {0}".format(line_number)) from error
                if not isinstance(value, Mapping):
                    raise CalibrationError("JSONL annotation rows must be objects")
                rows.append(dict(value))
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise CalibrationError("CSV annotation input requires a header row")
            rows.extend(dict(row) for row in reader)
    return rows


def _resolve_format(path: Path, requested: Optional[str]) -> str:
    fmt = requested.lower() if requested else path.suffix.lower().lstrip(".")
    if fmt not in ("jsonl", "csv"):
        raise CalibrationError("format must be jsonl or csv (or use a matching file suffix)")
    return fmt


def _is_hash(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    digest = value[7:] if value.startswith("sha256:") else value
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest.lower())


def _require_hash(value: Any, field_name: str) -> str:
    if not _is_hash(value):
        raise CalibrationError("{0} must be a SHA-256 digest".format(field_name))
    return value


def _hash_equal(first: Any, second: Any) -> bool:
    if not _is_hash(first) or not _is_hash(second):
        return False
    first_digest = first[7:] if first.startswith("sha256:") else first
    second_digest = second[7:] if second.startswith("sha256:") else second
    return first_digest.lower() == second_digest.lower()


def _required_mapping_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    if key not in payload:
        raise CalibrationError("{0}.{1} is required".format(context, key))
    return _strict_string(payload[key], "{0}.{1}".format(context, key))


def _require_exact_string(payload: Mapping[str, Any], key: str, expected: str, context: str) -> None:
    actual = _required_mapping_string(payload, key, context)
    if actual != expected:
        raise CalibrationError("{0}.{1} does not match the frozen manifest".format(context, key))


def _require_exact_hash(payload: Mapping[str, Any], key: str, expected: str, context: str) -> None:
    if key not in payload:
        raise CalibrationError("{0}.{1} is required".format(context, key))
    _require_hash(payload[key], "{0}.{1}".format(context, key))
    if not _hash_equal(payload[key], expected):
        raise CalibrationError("{0}.{1} does not match the frozen manifest".format(context, key))


def _strict_progress(value: Any) -> int:
    if isinstance(value, bool):
        raise CalibrationError("progress must be an integer from 0 through 5")
    if isinstance(value, str):
        if not value or not value.isdigit():
            raise CalibrationError("progress must be an integer from 0 through 5")
        value = int(value)
    if not isinstance(value, int) or not 0 <= value <= 5:
        raise CalibrationError("progress must be an integer from 0 through 5")
    return value


def _strict_schema_version(value: Any) -> int:
    """Accept the CSV spelling ``"1"`` but not floating or boolean versions."""

    if isinstance(value, bool):
        raise CalibrationError("annotation schema_version must be an integer")
    if isinstance(value, str):
        if not value or not value.isdigit():
            raise CalibrationError("annotation schema_version must be an integer")
        value = int(value)
    if not isinstance(value, int):
        raise CalibrationError("annotation schema_version must be an integer")
    return value


def _strict_enum(value: Any, allowed: Sequence[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise CalibrationError("{0} must be one of {1}".format(field_name, ", ".join(allowed)))
    return value


def _parse_evidence_frame_indices(value: Any) -> Tuple[int, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise CalibrationError("evidence_frame_indices must be a JSON array in CSV") from error
    if not isinstance(value, list):
        raise CalibrationError("evidence_frame_indices must be a list of non-negative integers")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise CalibrationError("evidence_frame_indices must be a list of non-negative integers")
    if len(set(value)) != len(value):
        raise CalibrationError("evidence_frame_indices must not contain duplicates")
    return tuple(value)


def _parse_observable_reason(value: Any) -> str:
    if not isinstance(value, str):
        raise CalibrationError("observable_reason must be a string")
    if len(value) > 2000:
        raise CalibrationError("observable_reason is too long")
    return value


def _strict_string(value: Any, field_name: str) -> str:
    if not _nonempty_string(value):
        raise CalibrationError("{0} must be a non-empty string".format(field_name))
    return value


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_probability(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise CalibrationError("{0} must be a finite value from 0 through 1".format(field_name))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _ratio_or_none(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _load_manifest(path: Union[str, Path]) -> Tuple[ClipManifestRow, ...]:
    candidate = Path(path)
    if not candidate.is_file():
        raise CalibrationError("manifest does not exist: {0}".format(candidate))
    if candidate.suffix.lower() == ".jsonl":
        rows = _read_rows(candidate)
    elif candidate.suffix.lower() == ".json":
        try:
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise CalibrationError("manifest JSON is invalid") from error
        if isinstance(parsed, Mapping):
            parsed = parsed.get("clips")
        if not isinstance(parsed, list):
            raise CalibrationError("manifest JSON must be a list or an object with clips")
        rows = parsed
    elif candidate.suffix.lower() == ".csv":
        rows = _read_rows(candidate)
    else:
        raise CalibrationError("manifest must be JSON, JSONL, or CSV")
    return validate_manifest(rows)


def _load_json_object(path: Union[str, Path], description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError("unable to read {0}".format(description)) from error
    if not isinstance(value, Mapping):
        raise CalibrationError("{0} must be a JSON object".format(description))
    return value


def _load_judge_reports(path: Union[str, Path]) -> List[Mapping[str, Any]]:
    """Read persisted judge reports from JSONL or a JSON object/list artifact."""

    candidate = Path(path)
    if not candidate.is_file():
        raise CalibrationError("judge report input does not exist: {0}".format(candidate))
    if candidate.suffix.lower() == ".jsonl":
        return _read_rows(candidate)
    if candidate.suffix.lower() != ".json":
        raise CalibrationError("judge reports must be JSONL or JSON")
    try:
        parsed = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError("unable to read judge reports") from error
    if isinstance(parsed, Mapping):
        if isinstance(parsed.get("reports"), list):
            parsed = parsed["reports"]
        else:
            parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(value, Mapping) for value in parsed):
        raise CalibrationError("judge report JSON must be an object, a list, or an object with reports")
    return [dict(value) for value in parsed]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint: ``python -m plumb.calibration export|report``."""

    parser = argparse.ArgumentParser(description="PLUMB blinded human-calibration workflow")
    subcommands = parser.add_subparsers(dest="command", required=True)
    export = subcommands.add_parser("export", help="write one blinded annotation packet")
    export.add_argument("--manifest", required=True)
    export.add_argument("--annotator", required=True)
    export.add_argument("--annotators", required=True, nargs=2, metavar=("ANNOTATOR_A", "ANNOTATOR_B"))
    export.add_argument("--output", required=True)
    export.add_argument("--format", choices=("jsonl", "csv"))

    report = subcommands.add_parser("report", help="summarize supplied calibration labels")
    report.add_argument("--manifest", required=True)
    report.add_argument("--annotations", required=True, action="append")
    report.add_argument("--annotators", required=True, nargs=2, metavar=("ANNOTATOR_A", "ANNOTATOR_B"))
    report.add_argument("--judge-annotations", action="append")
    report.add_argument("--judge-reports", action="append")
    report.add_argument("--protocol")
    report.add_argument("--tolerances")
    report.add_argument(
        "--annotator-registry",
        help="Annotator registry JSON declaring each annotator_type (human/model/external_label).",
    )
    report.add_argument(
        "--output",
        help="Write results/judge_calibration.json atomically; requires --annotator-registry.",
    )

    args = parser.parse_args(argv)
    try:
        manifest = _load_manifest(args.manifest)
        if args.command == "export":
            count = export_annotation_packets(
                manifest,
                args.output,
                args.annotator,
                args.annotators,
                output_format=args.format,
            )
            print(json.dumps({"exported_annotations": count, "output": str(args.output)}, sort_keys=True))
            return 0
        assignments = deterministic_annotation_assignments(manifest, args.annotators)
        human_labels: List[Annotation] = []
        for path in args.annotations:
            human_labels.extend(import_annotations(path, manifest, assignments, args.annotators))
        judge_labels: List[Annotation] = []
        for path in args.judge_annotations or []:
            judge_labels.extend(import_annotations(path, manifest, expected_assignment=False))
        judge_reports: List[Mapping[str, Any]] = []
        for path in args.judge_reports or []:
            judge_reports.extend(_load_judge_reports(path))
        protocol = _load_json_object(args.protocol, "protocol") if args.protocol else None
        tolerances = _load_json_object(args.tolerances, "tolerances") if args.tolerances else None
        result = build_calibration_report(
            manifest,
            human_labels,
            args.annotators,
            judge_annotations=judge_labels if args.judge_annotations else None,
            judge_reports=judge_reports if args.judge_reports else None,
            protocol=protocol,
            tolerances=tolerances,
        )
        if args.output:
            if not args.annotator_registry:
                raise CalibrationError(
                    "--output requires --annotator-registry: the report records each annotator's "
                    "declared type, and a missing type is never treated as human"
                )
            # Imported lazily so this module stays importable without the
            # annotation surface, and so there is no import cycle.
            from .annotation import AnnotationError, AnnotatorRegistry, write_judge_calibration_json

            try:
                registry = AnnotatorRegistry.load(args.annotator_registry)
                if set(registry.annotator_ids) != set(args.annotators):
                    raise CalibrationError("--annotators must match the annotator registry IDs exactly")
                written = write_judge_calibration_json(
                    args.output,
                    result,
                    registry,
                    manifest=manifest,
                    annotation_rows=[
                        {"clip_id": label.clip_id, "annotator_id": label.annotator_id}
                        for label in human_labels
                    ],
                )
            except AnnotationError as error:
                raise CalibrationError(str(error)) from error
            print(json.dumps(written, ensure_ascii=False, sort_keys=True, allow_nan=False))
            return 0
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except CalibrationError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Annotation",
    "BLINDED_EXPORT_FIELDS",
    "CalibrationError",
    "ClipManifestRow",
    "COMPLETION_VALUES",
    "COLLISION_VALUES",
    "DEVELOPMENT_SPLIT",
    "FrozenGateDProtocol",
    "FrozenJudgeEvidenceManifest",
    "GateDTolerances",
    "HELDOUT_SPLIT",
    "INTEGRITY_VALUES",
    "SCHEMA_VERSION",
    "TASKS",
    "TrustedJudgeProducer",
    "blinded_annotation_rows",
    "build_calibration_report",
    "calibration_manifest_hash",
    "deterministic_annotation_assignments",
    "evaluate_gate_d",
    "export_annotation_packets",
    "import_annotations",
    "main",
    "parse_annotations",
    "validate_human_annotation_coverage",
    "validate_manifest",
    "validate_primary_judge_evidence",
]
