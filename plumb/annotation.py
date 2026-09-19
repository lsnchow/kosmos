"""Blinded annotation surface and the ``results/judge_calibration.json`` writer.

``plumb.calibration`` already owns the statistics, the split plan, the
deterministic assignment rule, the blinded-field allowlist/denylist, and the
Gate-D threshold decision.  This module supplies the three things it lacks: a
place for a blinded annotator to actually see a clip and record a label, an
honest record of *who or what* produced each label, and a durable report file.

Annotator identity is the one thing that must never be rounded up.  Gate D
requires two blinded *human* annotators.  Three explicitly typed arms exist:

``human``
    A person.  Only this arm can satisfy Gate D.
``model``
    A blinded model pass, with model id and revision recorded.
``external_label``
    Third-party labels this project did not produce, such as AutoEval's own
    classifier labels on real drawer video.

A ``model`` or ``external_label`` annotator can never be serialized as
``human``: the type is validated on construction, the ``annotator_id`` must
carry the matching prefix, and the report writer recomputes the type from the
registry rather than trusting a submitted row.  Whenever two blinded humans are
absent, Gate D reads ``pass_with_limitations`` with ``human_annotation`` as a
named open dependency, and ``passed`` stays false.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .calibration import (
    Annotation,
    AnnotatorOwnership,
    BLINDED_EXPORT_FIELDS,
    COLLISION_VALUES,
    COMPLETION_VALUES,
    DEVELOPMENT_SPLIT,
    HELDOUT_SPLIT,
    INTEGRITY_VALUES,
    SCHEMA_VERSION,
    CalibrationError,
    ClipManifestRow,
    _FORBIDDEN_BLIND_FIELDS,
    build_calibration_report,
    blinded_annotation_rows,
    blinded_clip_id,
    blinded_media_ref,
    blinded_media_resolver,
    calibration_manifest_hash,
    deterministic_annotation_assignments,
    export_annotation_packets,
    freeze_calibration_selection,
    import_annotations,
    load_frozen_calibration_selection,
    main as _calibration_main,
    validate_manifest,
    validate_calibration_lineage_partition,
    validate_annotator_ownership,
)
from .policies.tasks import BENCHMARK_TASK_REGISTRY


HUMAN_ANNOTATOR = "human"
MODEL_ANNOTATOR = "model"
EXTERNAL_LABEL_ANNOTATOR = "external_label"
ANNOTATOR_TYPES: Tuple[str, ...] = (HUMAN_ANNOTATOR, MODEL_ANNOTATOR, EXTERNAL_LABEL_ANNOTATOR)

#: Required ``annotator_id`` prefix per type.  A mislabelled row in a CSV is
#: then visible in the identifier itself, not only in a metadata column.
ANNOTATOR_ID_PREFIXES: Mapping[str, str] = {
    HUMAN_ANNOTATOR: "human:",
    MODEL_ANNOTATOR: "model:",
    EXTERNAL_LABEL_ANNOTATOR: "external:",
}

#: Fields a blinded view may add on top of the label allowlist.  None of them
#: may collide with ``plumb.calibration._FORBIDDEN_BLIND_FIELDS``.
VIEW_ONLY_FIELDS: Tuple[str, ...] = (
    "media_url",
    "task_instruction",
    "task_rubric",
    "progress_scale",
    "position",
    "remaining",
    "total_assigned",
    "already_labelled",
)

#: Policy, backend, source-condition, and split labels that must never appear
#: in a blinded payload.  These are experimental arm labels, not content, so a
#: hit is a leak rather than a coincidence.  Generic cohort words such as
#: ``calibration`` are deliberately *not* scanned: they are already forbidden as
#: field names by the shared denylist, and scanning them as free text would
#: reject legitimate artifact paths.
_FORBIDDEN_BLIND_VALUE_TOKENS: Tuple[str, ...] = (
    "openvla",
    "openpizero",
    "open_pi_zero",
    "octo",
    "octo_small",
    "octo_base",
    "minivla",
    "susie",
    "susie_ll",
    "cosmos",
    "irasim",
    "baseten",
    "synthetic",
    "development",
    "heldout",
    "held_out",
    "world_seed",
)
_FORBIDDEN_VALUE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(token) for token in _FORBIDDEN_BLIND_VALUE_TOKENS) + r")\b",
    re.IGNORECASE,
)

#: Frozen 0--5 milestone scale text shown to a blinded annotator.  It carries no
#: policy, condition, or published rate.
PROGRESS_SCALE_LABELS: Tuple[str, ...] = (
    "0 - no directed approach",
    "1 - approach",
    "2 - contact or grasp established",
    "3 - moved toward target",
    "4 - partial target state",
    "5 - published final-state criterion met in the visible end state",
)

HUMAN_ANNOTATION_DEPENDENCY = "human_annotation"
SECOND_HUMAN_DEPENDENCY = "second_blinded_human_annotator"

STRATIFICATION_DISCLOSURE = (
    "The 150-clip panel is stratified across tasks, policies, apparent successes/failures, and "
    "generation defects. Prevalence-sensitive statistics from an enriched calibration sample are "
    "not population estimates without sampling weights."
)

HUMAN_AGREEMENT_INTERPRETATION = (
    "Human-human agreement is a reproducibility reference for this two-annotator protocol. It is "
    "not a mathematical ceiling on judge accuracy."
)

LENIENCY_DEFINITION = (
    "judge-positive proportion minus human-positive proportion on the same consensus-labelled cases"
)

#: Two-sided 95% normal quantile, inlined so this module needs no SciPy.
_Z_95 = 1.959963984540054


class AnnotationError(ValueError):
    """Raised for an invalid annotator identity, blinded view, or label."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_string(value: Any, field_name: str) -> str:
    if not _nonempty_string(value):
        raise AnnotationError("%s must be a non-empty string" % field_name)
    return value


@dataclass(frozen=True)
class AnnotatorIdentity:
    """Who or what produced a label, recorded so it cannot be rounded up."""

    annotator_id: str
    annotator_type: str
    blinded: bool = True
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    label_source: Optional[str] = None
    source_uri: Optional[str] = None
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.annotator_id, "annotator_id")
        if self.annotator_type not in ANNOTATOR_TYPES:
            raise AnnotationError("annotator_type must be one of %s" % ", ".join(ANNOTATOR_TYPES))
        if not isinstance(self.blinded, bool):
            raise AnnotationError("annotator blinded must be a boolean")
        prefix = ANNOTATOR_ID_PREFIXES[self.annotator_type]
        if not self.annotator_id.startswith(prefix):
            raise AnnotationError(
                "an annotator of type %r must use an annotator_id beginning %r so a label can never "
                "be read as a different kind of annotator" % (self.annotator_type, prefix)
            )
        for other_type, other_prefix in ANNOTATOR_ID_PREFIXES.items():
            if other_type != self.annotator_type and self.annotator_id.startswith(other_prefix):
                raise AnnotationError("annotator_id prefix contradicts annotator_type")
        if self.annotator_type == MODEL_ANNOTATOR:
            _require_string(self.model_id, "model annotator model_id")
            _require_string(self.model_revision, "model annotator model_revision")
        elif self.model_id is not None or self.model_revision is not None:
            raise AnnotationError(
                "only a model annotator records model_id/model_revision; a human or external-label "
                "annotator carrying model identity is a mislabelled arm"
            )
        if self.annotator_type == EXTERNAL_LABEL_ANNOTATOR:
            _require_string(self.label_source, "external_label annotator label_source")
            _require_string(self.source_uri, "external_label annotator source_uri")
        if self.annotator_type == HUMAN_ANNOTATOR and self.label_source is not None:
            raise AnnotationError("a human annotator does not carry a third-party label_source")

    @property
    def is_human(self) -> bool:
        return self.annotator_type == HUMAN_ANNOTATOR

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "annotator_id": self.annotator_id,
            "annotator_type": self.annotator_type,
            "blinded": self.blinded,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "label_source": self.label_source,
            "source_uri": self.source_uri,
            "notes": list(self.notes),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AnnotatorIdentity":
        if not isinstance(payload, Mapping):
            raise AnnotationError("an annotator record must be an object")
        allowed = {
            "annotator_id",
            "annotator_type",
            "blinded",
            "model_id",
            "model_revision",
            "label_source",
            "source_uri",
            "notes",
        }
        unexpected = sorted(set(payload) - allowed)
        if unexpected:
            raise AnnotationError("annotator record has unsupported fields: %s" % ", ".join(unexpected))
        notes = payload.get("notes", ())
        return cls(
            annotator_id=payload.get("annotator_id"),
            annotator_type=payload.get("annotator_type"),
            blinded=payload.get("blinded", True),
            model_id=payload.get("model_id"),
            model_revision=payload.get("model_revision"),
            label_source=payload.get("label_source"),
            source_uri=payload.get("source_uri"),
            notes=tuple(notes) if isinstance(notes, (list, tuple)) else (str(notes),),
        )


#: Deterministic composed labels for the report's ``calibration_class`` field.
_CALIBRATION_CLASS_LABELS: Mapping[Tuple[str, str], str] = {
    (HUMAN_ANNOTATOR, HUMAN_ANNOTATOR): "human",
    (HUMAN_ANNOTATOR, MODEL_ANNOTATOR): "human + model_reference",
    (EXTERNAL_LABEL_ANNOTATOR, HUMAN_ANNOTATOR): "human + external_label",
    (MODEL_ANNOTATOR, MODEL_ANNOTATOR): "model_reference",
    (EXTERNAL_LABEL_ANNOTATOR, MODEL_ANNOTATOR): "external_label + model_reference",
    (EXTERNAL_LABEL_ANNOTATOR, EXTERNAL_LABEL_ANNOTATOR): "external_label",
}


@dataclass(frozen=True)
class AnnotatorRegistry:
    """Exactly the two annotators the held-out overlap protocol needs."""

    annotators: Tuple[AnnotatorIdentity, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.annotators) != 2:
            raise AnnotationError("exactly two annotator identities are required for the heldout overlap")
        for annotator in self.annotators:
            if not isinstance(annotator, AnnotatorIdentity):
                raise AnnotationError("registry entries must be AnnotatorIdentity values")
        ids = [annotator.annotator_id for annotator in self.annotators]
        if len(set(ids)) != 2:
            raise AnnotationError("annotator IDs must be distinct")
        if any(not annotator.blinded for annotator in self.annotators):
            raise AnnotationError("every calibration annotator must be blinded")

    @property
    def annotator_ids(self) -> Tuple[str, str]:
        values = tuple(sorted(annotator.annotator_id for annotator in self.annotators))
        return (values[0], values[1])

    def get(self, annotator_id: str) -> AnnotatorIdentity:
        for annotator in self.annotators:
            if annotator.annotator_id == annotator_id:
                return annotator
        raise AnnotationError("unknown annotator_id %r" % (annotator_id,))

    @property
    def human_count(self) -> int:
        return sum(1 for annotator in self.annotators if annotator.is_human)

    @property
    def has_two_blinded_humans(self) -> bool:
        return self.human_count == 2

    @property
    def calibration_class(self) -> str:
        key = tuple(sorted(annotator.annotator_type for annotator in self.annotators))
        label = _CALIBRATION_CLASS_LABELS.get((key[0], key[1]))
        if label is None:  # pragma: no cover - the table covers every pair
            raise AnnotationError("no calibration_class label for annotator types %s" % ", ".join(key))
        return label

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "calibration_class": self.calibration_class,
            "human_annotator_count": self.human_count,
            "has_two_blinded_humans": self.has_two_blinded_humans,
            "annotators": [
                annotator.as_mapping()
                for annotator in sorted(self.annotators, key=lambda item: item.annotator_id)
            ],
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AnnotatorRegistry":
        if not isinstance(payload, Mapping):
            raise AnnotationError("an annotator registry must be an object")
        rows = payload.get("annotators")
        if not isinstance(rows, (list, tuple)):
            raise AnnotationError("annotator registry needs an annotators list")
        return cls(annotators=tuple(AnnotatorIdentity.from_mapping(row) for row in rows))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnnotatorRegistry":
        source = Path(path)
        if not source.is_file():
            raise AnnotationError("annotator registry does not exist: %s" % source)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AnnotationError("unable to read annotator registry %s" % source) from error
        return cls.from_mapping(payload)

    def save(self, path: Union[str, Path]) -> Dict[str, Any]:
        destination = Path(path)
        write_json_atomic(destination, self.as_mapping())
        return {"path": str(destination), "calibration_class": self.calibration_class}


def assert_blinded(payload: Any, *, context: str = "blinded payload") -> None:
    """Refuse a payload that carries a forbidden field name or arm label.

    Field names are checked against ``plumb.calibration._FORBIDDEN_BLIND_FIELDS``
    so there is exactly one denylist.  String values are additionally scanned:
    an opaque ``media_ref`` that happens to contain a policy directory name
    would otherwise unblind the annotator.
    """

    if isinstance(payload, Mapping):
        forbidden = sorted(set(str(key) for key in payload) & _FORBIDDEN_BLIND_FIELDS)
        if forbidden:
            raise AnnotationError(
                "%s carries blinded metadata fields: %s" % (context, ", ".join(forbidden))
            )
        for key, value in payload.items():
            assert_blinded(value, context="%s.%s" % (context, key))
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_blinded(value, context="%s[%d]" % (context, index))
        return
    if isinstance(payload, str):
        match = _FORBIDDEN_VALUE_PATTERN.search(payload)
        if match is not None:
            raise AnnotationError(
                "%s leaks the experimental arm label %r; blinded views need opaque identifiers"
                % (context, match.group(0))
            )


def _media_url(media_ref: str, prefix: str) -> str:
    if media_ref.startswith("artifact://"):
        return prefix.rstrip("/") + "/" + media_ref[len("artifact://") :].lstrip("/")
    if media_ref.startswith(("http://", "https://", "/")):
        return media_ref
    return prefix.rstrip("/") + "/" + media_ref.lstrip("/")


def blinded_clip_view(
    clip: ClipManifestRow,
    annotator_id: str,
    *,
    media_url_prefix: str = "/api/artifacts",
    position: Optional[int] = None,
    total_assigned: Optional[int] = None,
    remaining: Optional[int] = None,
    already_labelled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Render one clip for a blinded annotator and verify it leaks nothing.

    The split, lineage, policy, backend, action text, condition, gate status,
    reference rate, world seed, and cohort are all absent by construction: the
    payload is assembled from the label allowlist plus ``VIEW_ONLY_FIELDS``, and
    then re-checked against the shared denylist.
    """

    task = BENCHMARK_TASK_REGISTRY.get(clip.task)
    view: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "annotator_id": annotator_id,
        "clip_id": clip.clip_id,
        "media_ref": clip.media_ref,
        "media_url": _media_url(clip.media_ref, media_url_prefix),
        "task": clip.task,
        "task_instruction": task.instruction,
        "task_rubric": task.rubric,
        "progress_scale": list(PROGRESS_SCALE_LABELS),
        "integrity": "",
        "collision": "",
        "progress": "",
        "completion_evidence": "",
        "evidence_frame_indices": "",
        "observable_reason": "",
    }
    if position is not None:
        view["position"] = position
    if total_assigned is not None:
        view["total_assigned"] = total_assigned
    if remaining is not None:
        view["remaining"] = remaining
    if already_labelled is not None:
        view["already_labelled"] = already_labelled
    allowed = set(BLINDED_EXPORT_FIELDS) | set(VIEW_ONLY_FIELDS)
    unexpected = sorted(set(view) - allowed)
    if unexpected:
        raise AnnotationError("blinded clip view has unsupported fields: %s" % ", ".join(unexpected))
    assert_blinded(view, context="blinded clip view")
    return view


@dataclass(frozen=True)
class LabelSubmission:
    """One captured label.  ``None`` means unknown and never becomes a guess."""

    clip_id: str
    annotator_id: str
    integrity: Optional[str] = None
    collision: Optional[str] = None
    progress: Optional[int] = None
    completion_evidence: Optional[str] = None
    evidence_frame_indices: Tuple[int, ...] = ()
    observable_reason: str = ""
    submitted_at: Optional[str] = None

    def __post_init__(self) -> None:
        _require_string(self.clip_id, "label clip_id")
        _require_string(self.annotator_id, "label annotator_id")
        if self.integrity is not None and self.integrity not in INTEGRITY_VALUES:
            raise AnnotationError("integrity must be one of %s or null" % ", ".join(INTEGRITY_VALUES))
        if self.collision is not None and self.collision not in COLLISION_VALUES:
            raise AnnotationError("collision must be one of %s or null" % ", ".join(COLLISION_VALUES))
        if self.completion_evidence is not None and self.completion_evidence not in COMPLETION_VALUES:
            raise AnnotationError(
                "completion_evidence must be one of %s or null" % ", ".join(COMPLETION_VALUES)
            )
        if self.progress is not None and (
            isinstance(self.progress, bool) or not isinstance(self.progress, int) or not 0 <= self.progress <= 5
        ):
            raise AnnotationError("progress must be an integer from 0 through 5, or null for unknown")
        if not isinstance(self.evidence_frame_indices, tuple):
            raise AnnotationError("evidence_frame_indices must be a tuple of frame indices")
        for index in self.evidence_frame_indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 16:
                raise AnnotationError("evidence_frame_indices must be frame indexes 0 through 15")
        if len(set(self.evidence_frame_indices)) != len(self.evidence_frame_indices):
            raise AnnotationError("evidence_frame_indices must not repeat a frame")
        if not isinstance(self.observable_reason, str) or len(self.observable_reason) > 2000:
            raise AnnotationError("observable_reason must be a string of at most 2000 characters")
        if self.completion_evidence == "met" and self.progress is not None and self.progress != 5:
            raise AnnotationError("completion_evidence='met' requires progress=5")
        if self.completion_evidence == "not_met" and self.progress is not None and self.progress > 4:
            raise AnnotationError("completion_evidence='not_met' requires progress from 0 through 4")

    def missing_fields(self) -> Tuple[str, ...]:
        missing: List[str] = []
        if self.integrity is None:
            missing.append("integrity")
        if self.collision is None:
            missing.append("collision")
        if self.progress is None:
            missing.append("progress")
        if self.completion_evidence is None:
            missing.append("completion_evidence")
        if not self.evidence_frame_indices:
            missing.append("evidence_frame_indices")
        return tuple(missing)

    @property
    def complete(self) -> bool:
        return not self.missing_fields()

    def to_annotation_row(self) -> Dict[str, Any]:
        """Return the strict label row ``plumb.calibration`` accepts.

        An incomplete capture raises rather than being completed with a default;
        it stays in the store as an unfinished capture and in the missingness
        counts.
        """

        missing = self.missing_fields()
        if missing:
            raise AnnotationError(
                "this capture is incomplete and is not a label; missing %s" % ", ".join(missing)
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "clip_id": self.clip_id,
            "annotator_id": self.annotator_id,
            "integrity": self.integrity,
            "collision": self.collision,
            "progress": self.progress,
            "completion_evidence": self.completion_evidence,
            "evidence_frame_indices": list(self.evidence_frame_indices),
            "observable_reason": self.observable_reason,
        }

    def as_store_row(self, annotator_type: str) -> Dict[str, Any]:
        if annotator_type not in ANNOTATOR_TYPES:
            raise AnnotationError("annotator_type must be one of %s" % ", ".join(ANNOTATOR_TYPES))
        return {
            "record_type": "label",
            "schema_version": SCHEMA_VERSION,
            "clip_id": self.clip_id,
            "annotator_id": self.annotator_id,
            "annotator_type": annotator_type,
            "integrity": self.integrity,
            "collision": self.collision,
            "progress": self.progress,
            "completion_evidence": self.completion_evidence,
            "evidence_frame_indices": list(self.evidence_frame_indices),
            "observable_reason": self.observable_reason,
            "complete": self.complete,
            "missing_fields": list(self.missing_fields()),
            "submitted_at": self.submitted_at or utc_now(),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LabelSubmission":
        if not isinstance(payload, Mapping):
            raise AnnotationError("a label submission must be an object")
        forbidden = sorted(set(str(key) for key in payload) & _FORBIDDEN_BLIND_FIELDS)
        if forbidden:
            raise AnnotationError("label submission carries blinded metadata: %s" % ", ".join(forbidden))
        allowed = {
            "schema_version",
            "clip_id",
            "annotator_id",
            "integrity",
            "collision",
            "progress",
            "completion_evidence",
            "evidence_frame_indices",
            "observable_reason",
            "submitted_at",
            "record_type",
            "annotator_type",
            "complete",
            "missing_fields",
        }
        unexpected = sorted(set(payload) - allowed)
        if unexpected:
            raise AnnotationError("label submission has unsupported fields: %s" % ", ".join(unexpected))
        if "schema_version" in payload and _coerce_int(payload["schema_version"]) != SCHEMA_VERSION:
            raise AnnotationError("label schema_version must match the calibration schema")
        frames = payload.get("evidence_frame_indices", ())
        if isinstance(frames, str):
            if not frames.strip():
                frames = ()
            else:
                try:
                    frames = json.loads(frames)
                except json.JSONDecodeError as error:
                    raise AnnotationError("evidence_frame_indices must be a JSON array") from error
        if frames is None:
            frames = ()
        if not isinstance(frames, (list, tuple)):
            raise AnnotationError("evidence_frame_indices must be a list of frame indexes")
        return cls(
            clip_id=payload.get("clip_id"),
            annotator_id=payload.get("annotator_id"),
            integrity=_blank_to_none(payload.get("integrity")),
            collision=_blank_to_none(payload.get("collision")),
            progress=_optional_int(payload.get("progress")),
            completion_evidence=_blank_to_none(payload.get("completion_evidence")),
            evidence_frame_indices=tuple(frames),
            observable_reason=payload.get("observable_reason") or "",
            submitted_at=_blank_to_none(payload.get("submitted_at")),
        )


def _blank_to_none(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        raise AnnotationError("expected an integer, not a boolean")
    if isinstance(value, str):
        if not value.strip().isdigit():
            raise AnnotationError("expected an integer")
        return int(value)
    if not isinstance(value, int):
        raise AnnotationError("expected an integer")
    return value


def _optional_int(value: Any) -> Optional[int]:
    blank = _blank_to_none(value)
    return None if blank is None else _coerce_int(blank)


class AnnotationStore:
    """Append-only JSONL label store; the latest row per pair wins.

    Every row records the annotator type, so a model pass can never be reread
    as a human label even if the file is edited by hand: the type must still
    match the registry when the report is built.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    def _rows(self) -> List[Dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: List[Dict[str, Any]] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as error:
                raise AnnotationError("invalid JSONL on line %d of %s" % (number, self.path)) from error
            if not isinstance(parsed, Mapping):
                raise AnnotationError("label store rows must be objects")
            rows.append(dict(parsed))
        return rows

    def latest_rows(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in self._rows():
            annotator_id = row.get("annotator_id")
            clip_id = row.get("clip_id")
            if not _nonempty_string(annotator_id) or not _nonempty_string(clip_id):
                raise AnnotationError("label store rows need annotator_id and clip_id")
            latest[(str(annotator_id), str(clip_id))] = row
        return latest

    def submissions(self) -> Tuple[LabelSubmission, ...]:
        return tuple(
            LabelSubmission.from_mapping(row)
            for _, row in sorted(self.latest_rows().items(), key=lambda item: item[0])
        )

    def complete_submissions(self) -> Tuple[LabelSubmission, ...]:
        return tuple(item for item in self.submissions() if item.complete)

    def recorded_types(self) -> Dict[str, str]:
        types: Dict[str, str] = {}
        for (annotator_id, _), row in sorted(self.latest_rows().items()):
            recorded = row.get("annotator_type")
            if not _nonempty_string(recorded):
                raise AnnotationError(
                    "label store row for %s has no annotator_type; an unlabelled arm is not a human"
                    % annotator_id
                )
            previous = types.setdefault(annotator_id, str(recorded))
            if previous != recorded:
                raise AnnotationError("label store records two annotator types for %s" % annotator_id)
        return types

    def append(self, submission: LabelSubmission, annotator_type: str) -> Dict[str, Any]:
        row = submission.as_store_row(annotator_type)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return row

    def export_annotation_rows(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Write only the complete labels, in the exact shape calibration parses."""

        complete = self.complete_submissions()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(item.to_annotation_row(), sort_keys=True, ensure_ascii=False, allow_nan=False)
            for item in complete
        ]
        _atomic_write_text(destination, "".join(line + "\n" for line in lines))
        return {
            "path": str(destination),
            "complete_labels": len(complete),
            "incomplete_captures": len(self.submissions()) - len(complete),
        }


class AnnotationService:
    """Server-side data layer for a blinded ``/annotate`` view.

    Each method is a pure function of the frozen manifest, the annotator
    registry, and the label store, and returns JSON-safe values.  An HTTP layer
    only has to map a route to a method and serialize the result.
    """

    def __init__(
        self,
        manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
        registry: AnnotatorRegistry,
        store: AnnotationStore,
        *,
        media_url_prefix: str = "/api/artifacts",
        require_full_plan: bool = True,
    ) -> None:
        self.clips = validate_manifest(manifest, require_full_plan=require_full_plan)
        self.registry = registry
        self.store = store
        self.media_url_prefix = media_url_prefix
        self.assignments = deterministic_annotation_assignments(self.clips, registry.annotator_ids)
        self._clips_by_id = {clip.clip_id: clip for clip in self.clips}

    def _assigned_clips(self, annotator_id: str) -> Tuple[ClipManifestRow, ...]:
        self.registry.get(annotator_id)
        return tuple(
            clip for clip in self.clips if annotator_id in self.assignments[clip.clip_id]
        )

    def _labelled_clip_ids(self, annotator_id: str) -> set:
        return {
            clip_id
            for (stored_annotator, clip_id), row in self.store.latest_rows().items()
            if stored_annotator == annotator_id and row.get("complete") is True
        }

    def session(self, annotator_id: str) -> Dict[str, Any]:
        """Blinded session summary; carries no split or cohort information."""

        annotator = self.registry.get(annotator_id)
        assigned = self._assigned_clips(annotator_id)
        labelled = self._labelled_clip_ids(annotator_id)
        remaining = [clip.clip_id for clip in assigned if clip.clip_id not in labelled]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "annotator_id": annotator.annotator_id,
            "annotator_type": annotator.annotator_type,
            "blinded": annotator.blinded,
            "total_assigned": len(assigned),
            "completed": len(labelled),
            "remaining": len(remaining),
            "next_clip_id": remaining[0] if remaining else None,
            "progress_scale": list(PROGRESS_SCALE_LABELS),
            "integrity_values": list(INTEGRITY_VALUES),
            "collision_values": list(COLLISION_VALUES),
            "completion_evidence_values": list(COMPLETION_VALUES),
            "instructions": (
                "Score only what is visible. Leave a field blank when the video does not show it; a "
                "blank field stays unknown and is never counted as a failure."
            ),
        }
        # ``annotator_type`` is deliberately shown: an annotator knowing what
        # they are does not unblind the experimental arm of the clip.
        assert_blinded(
            {key: value for key, value in payload.items() if key != "annotator_type"},
            context="annotation session",
        )
        return payload

    def queue(self, annotator_id: str) -> Dict[str, Any]:
        assigned = self._assigned_clips(annotator_id)
        labelled = self._labelled_clip_ids(annotator_id)
        views = []
        for position, clip in enumerate(assigned, 1):
            views.append(
                blinded_clip_view(
                    clip,
                    annotator_id,
                    media_url_prefix=self.media_url_prefix,
                    position=position,
                    total_assigned=len(assigned),
                    remaining=len(assigned) - len(labelled),
                    already_labelled=clip.clip_id in labelled,
                )
            )
        return {"schema_version": SCHEMA_VERSION, "annotator_id": annotator_id, "clips": views}

    def clip(self, annotator_id: str, clip_id: str) -> Dict[str, Any]:
        assigned = self._assigned_clips(annotator_id)
        ordered = {clip.clip_id: index for index, clip in enumerate(assigned, 1)}
        if clip_id not in ordered:
            raise AnnotationError("clip %r is not assigned to %r" % (clip_id, annotator_id))
        labelled = self._labelled_clip_ids(annotator_id)
        return blinded_clip_view(
            self._clips_by_id[clip_id],
            annotator_id,
            media_url_prefix=self.media_url_prefix,
            position=ordered[clip_id],
            total_assigned=len(assigned),
            remaining=len(assigned) - len(labelled),
            already_labelled=clip_id in labelled,
        )

    def submit(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if isinstance(payload, Mapping) and "annotator_type" in payload:
            raise AnnotationError(
                "a submission may not declare annotator_type; the server reads it from the "
                "registry so a model pass can never post itself as human"
            )
        submission = LabelSubmission.from_mapping(payload)
        annotator = self.registry.get(submission.annotator_id)
        if submission.clip_id not in self._clips_by_id:
            raise AnnotationError("label references a clip outside the frozen manifest")
        if submission.annotator_id not in self.assignments[submission.clip_id]:
            raise AnnotationError("this clip is not assigned to this annotator")
        row = self.store.append(submission, annotator.annotator_type)
        return {
            "schema_version": SCHEMA_VERSION,
            "clip_id": submission.clip_id,
            "annotator_id": submission.annotator_id,
            "annotator_type": annotator.annotator_type,
            "accepted": True,
            "complete": submission.complete,
            "missing_fields": list(submission.missing_fields()),
            "submitted_at": row["submitted_at"],
        }

    def progress(self) -> Dict[str, Any]:
        """Coverage per annotator and per split, for an operator dashboard."""

        latest = self.store.latest_rows()
        by_annotator: Dict[str, Dict[str, Any]] = {}
        for annotator_id in self.registry.annotator_ids:
            assigned = self._assigned_clips(annotator_id)
            complete = sum(
                1
                for clip in assigned
                if latest.get((annotator_id, clip.clip_id), {}).get("complete") is True
            )
            captured = sum(1 for clip in assigned if (annotator_id, clip.clip_id) in latest)
            by_annotator[annotator_id] = {
                "annotator_type": self.registry.get(annotator_id).annotator_type,
                "assigned": len(assigned),
                "captured": captured,
                "complete": complete,
                "incomplete_captures": captured - complete,
                "outstanding": len(assigned) - complete,
            }
        by_split = {}
        for split in (DEVELOPMENT_SPLIT, HELDOUT_SPLIT):
            split_clips = [clip for clip in self.clips if clip.split == split]
            expected = sum(len(self.assignments[clip.clip_id]) for clip in split_clips)
            observed = sum(
                1
                for clip in split_clips
                for annotator_id in self.assignments[clip.clip_id]
                if latest.get((annotator_id, clip.clip_id), {}).get("complete") is True
            )
            by_split[split] = {"expected_labels": expected, "complete_labels": observed}
        return {
            "schema_version": SCHEMA_VERSION,
            "calibration_class": self.registry.calibration_class,
            "has_two_blinded_humans": self.registry.has_two_blinded_humans,
            "by_annotator": by_annotator,
            "by_split": by_split,
        }

    def annotation_rows(self) -> List[Dict[str, Any]]:
        """Complete labels in the exact shape ``plumb.calibration`` parses."""

        recorded = self.store.recorded_types()
        for annotator_id, annotator_type in recorded.items():
            declared = self.registry.get(annotator_id).annotator_type
            if declared != annotator_type:
                raise AnnotationError(
                    "stored labels for %s record type %r but the registry declares %r"
                    % (annotator_id, annotator_type, declared)
                )
        return [item.to_annotation_row() for item in self.store.complete_submissions()]


def wilson_interval(successes: int, total: int, *, z: float = _Z_95) -> Optional[Dict[str, Any]]:
    """Return a Wilson score interval, or ``None`` when there is no denominator."""

    if total <= 0:
        return None
    if successes < 0 or successes > total:
        raise AnnotationError("Wilson interval needs 0 <= successes <= total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    )
    return {
        "point": proportion,
        "lower": max(0.0, centre - margin),
        "upper": min(1.0, centre + margin),
        "n": total,
        "method": "wilson_score_95",
    }


def paired_difference_interval(
    positive_only_judge: int, positive_only_human: int, total: int, *, z: float = _Z_95
) -> Optional[Dict[str, Any]]:
    """Wald interval for a paired proportion difference on the same clips."""

    if total <= 0:
        return None
    discordant = positive_only_judge + positive_only_human
    if discordant > total:
        raise AnnotationError("discordant pairs cannot exceed the paired sample size")
    difference = (positive_only_judge - positive_only_human) / total
    variance = (discordant - (positive_only_judge - positive_only_human) ** 2 / total) / (total * total)
    standard_error = math.sqrt(max(0.0, variance))
    return {
        "point": difference,
        "lower": max(-1.0, difference - z * standard_error),
        "upper": min(1.0, difference + z * standard_error),
        "n": total,
        "discordant_pairs": discordant,
        "method": "paired_wald_95",
    }


def gate_d_with_limitations(
    gate_d: Mapping[str, Any], registry: AnnotatorRegistry
) -> Dict[str, Any]:
    """Downgrade a threshold pass to ``pass_with_limitations`` without humans.

    ``plumb.calibration.evaluate_gate_d`` answers only "were the frozen
    numerical tolerances met".  Gate D additionally requires two blinded human
    annotators.  When they are absent the thresholds may still be satisfied, but
    the gate is not passed and ``human_annotation`` is named as the dependency.
    """

    if not isinstance(gate_d, Mapping):
        raise AnnotationError("gate_d must be the mapping returned by evaluate_gate_d")
    inner_status = gate_d.get("status") or "not_evaluable"
    # Idempotent: re-wrapping an already-wrapped record (for example the
    # ``gate_d`` block of a written judge_calibration.json) must not downgrade
    # ``thresholds_satisfied``, whose ``passed`` was already set to false.
    thresholds_satisfied = (
        bool(gate_d["thresholds_satisfied"])
        if "thresholds_satisfied" in gate_d
        else bool(gate_d.get("passed", False))
    )
    if inner_status == "pass_with_limitations":
        inner_status = "pass" if thresholds_satisfied else "fail"
    dependencies: List[Dict[str, Any]] = []
    if not registry.has_two_blinded_humans:
        dependencies.append(
            {
                "name": HUMAN_ANNOTATION_DEPENDENCY,
                "blocks": "Gate D",
                "detail": (
                    "Gate D requires two blinded human annotators. This run used %s."
                    % registry.calibration_class
                ),
                "resolution_path": (
                    "Two blinded people each label the 50 held-out clips (about 30 minutes); see "
                    "docs/GATE_D_RUNBOOK.md."
                ),
            }
        )
        if registry.human_count == 1:
            dependencies.append(
                {
                    "name": SECOND_HUMAN_DEPENDENCY,
                    "blocks": "Gate D",
                    "detail": "Exactly one blinded human annotator is registered; the overlap needs two.",
                    "resolution_path": "Register a second blinded human and relabel the 50 held-out clips.",
                }
            )
    if thresholds_satisfied and dependencies:
        status = "pass_with_limitations"
        passed = False
    elif thresholds_satisfied:
        status = "pass"
        passed = True
    else:
        status = inner_status
        passed = False
    result = dict(gate_d)
    result.update(
        {
            "status": status,
            "passed": passed,
            "thresholds_satisfied": thresholds_satisfied,
            "calibration_class": registry.calibration_class,
            "has_two_blinded_humans": registry.has_two_blinded_humans,
            "annotators": [
                annotator.as_mapping()
                for annotator in sorted(registry.annotators, key=lambda item: item.annotator_id)
            ],
            "open_dependencies": dependencies,
            "open_dependency_names": [item["name"] for item in dependencies],
        }
    )
    return result


def gate_d_measurements(
    report: Mapping[str, Any],
    registry: AnnotatorRegistry,
    *,
    freeze_created_at: Optional[str] = None,
    judge_started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract exactly the measurements a Gate D record must carry.

    ``plumb.gates.GateRecord.pass_evidence_errors`` requires ``binary_kappa``,
    ``weighted_progress_kappa``, ``leniency_offset``, ``consensus_coverage``,
    ``calibration_class``, and ``held_out_frozen_before_evaluation``.  Each one
    is read from measured evidence or left ``None``; a null blocks the pass,
    which is the correct outcome when the quantity was not measured.
    """

    agreement = report.get("human_human_agreement", {})
    pooled = agreement.get("pooled", {}) if isinstance(agreement, Mapping) else {}
    comparison, comparison_source = _authoritative_comparison(report)
    consensus = report.get("heldout_consensus")
    frozen_before: Optional[bool] = None
    if _nonempty_string(freeze_created_at) and _nonempty_string(judge_started_at):
        frozen_before = str(freeze_created_at) <= str(judge_started_at)
    return {
        "binary_kappa": _kappa_of(pooled, "binary_success"),
        "weighted_progress_kappa": (pooled.get("progress") or {}).get("weighted_kappa")
        if isinstance(pooled.get("progress"), Mapping)
        else None,
        "leniency_offset": comparison.get("leniency_offset") if isinstance(comparison, Mapping) else None,
        "consensus_coverage": consensus.get("binary_success_coverage")
        if isinstance(consensus, Mapping)
        else None,
        "calibration_class": registry.calibration_class,
        "human_annotator_count": registry.human_count,
        "has_two_blinded_humans": registry.has_two_blinded_humans,
        "held_out_frozen_before_evaluation": frozen_before,
        "freeze_created_at": freeze_created_at,
        "judge_started_at": judge_started_at,
        "judge_comparison_source": comparison_source,
        "binary_kappa_basis": (
            "Cohen's kappa on the two annotators' visible-completion labels over held-out clips "
            "where both were decisive; indecisive labels are retained as missing, not recoded."
        ),
    }


def gate_d_evidence(
    report: Mapping[str, Any],
    registry: AnnotatorRegistry,
    *,
    judge_evidence: Optional[Mapping[str, Any]] = None,
    evidence_uris: Sequence[str] = (),
    protocol_hash: Optional[str] = None,
    freeze_created_at: Optional[str] = None,
    judge_started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge calibration statistics into a Gate D record for ``GateRecord``.

    ``status`` is never ``pass`` without two blinded human annotators, because
    the gate's own definition requires them.  ``pass_with_limitations`` is
    reported inside ``measurements`` since ``GateStatus`` has no such member.
    """

    base = dict(judge_evidence or {})
    measurements = dict(base.get("measurements") or {})
    measurements.update(
        gate_d_measurements(
            report,
            registry,
            freeze_created_at=freeze_created_at,
            judge_started_at=judge_started_at,
        )
    )
    wrapped = gate_d_with_limitations(report.get("gate_d") or {}, registry)
    measurements["gate_d_decision_status"] = wrapped["status"]
    measurements["thresholds_satisfied"] = wrapped["thresholds_satisfied"]
    reasons = [item for item in base.get("reasons") or () if isinstance(item, str)]
    for dependency in wrapped["open_dependency_names"]:
        marker = "open dependency: %s" % dependency
        if not any(marker in item for item in reasons):
            reasons.append(marker)
    for code in wrapped.get("reason_codes") or ():
        reasons.append("gate_d_reason: %s" % code)
    uris = list(evidence_uris) or list(base.get("evidence_uris") or ())
    if not uris:
        reasons.append("Gate D evidence bundle needs at least one evidence URI")
    resolved_hash = protocol_hash or base.get("protocol_hash")
    if not resolved_hash:
        reasons.append("Gate D evidence bundle needs the frozen protocol hash")
    status = "pass" if (wrapped["passed"] and not reasons) else "blocked"
    record = dict(base)
    record.update(
        {
            "gate_id": "D",
            "status": status,
            "protocol_hash": resolved_hash,
            "evidence_uris": uris,
            "measurements": measurements,
            "thresholds": dict(base.get("thresholds") or {}) or dict(wrapped.get("tolerances") or {}),
            "reasons": reasons,
            "evidence_kind": base.get("evidence_kind", "real_vlm_judge_reports_over_generated_clips"),
            "open_dependencies": wrapped["open_dependencies"],
        }
    )
    return record


def _authoritative_comparison(report: Mapping[str, Any]) -> Tuple[Optional[Mapping[str, Any]], str]:
    primary = report.get("primary_judge_evidence")
    if isinstance(primary, Mapping) and primary.get("available") and isinstance(
        primary.get("judge_comparison"), Mapping
    ):
        return primary["judge_comparison"], "primary_judge_evidence.judge_comparison"
    # ``judge_vs_human_descriptive`` is the key a written judge_calibration.json
    # uses, so re-reading that artifact resolves the same comparison.
    for key in ("judge_vs_human", "judge_vs_human_descriptive"):
        descriptive = report.get(key)
        if isinstance(descriptive, Mapping) and descriptive.get("available"):
            return descriptive, "%s (descriptive only; not bound to a frozen evidence manifest)" % key
    return None, "unavailable"


def build_judge_calibration_payload(
    report: Mapping[str, Any],
    registry: AnnotatorRegistry,
    *,
    manifest: Optional[Iterable[Union[ClipManifestRow, Mapping[str, Any]]]] = None,
    annotation_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    manifest_id: Optional[str] = None,
    scenario_manifest_sha256: Optional[str] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble ``results/judge_calibration.json`` from a calibration report."""

    if not isinstance(report, Mapping):
        raise AnnotationError("report must be the mapping returned by build_calibration_report")
    gate_d = report.get("gate_d")
    if not isinstance(gate_d, Mapping):
        raise AnnotationError("calibration report has no gate_d block")
    agreement = report.get("human_human_agreement", {})
    pooled = agreement.get("pooled", {}) if isinstance(agreement, Mapping) else {}
    comparison, comparison_source = _authoritative_comparison(report)
    confusion = comparison.get("confusion_matrix") if isinstance(comparison, Mapping) else None
    sensitivity = comparison.get("sensitivity") if isinstance(comparison, Mapping) else None
    specificity = comparison.get("specificity") if isinstance(comparison, Mapping) else None
    leniency = comparison.get("leniency_offset") if isinstance(comparison, Mapping) else None
    comparable = int(comparison.get("comparable_count", 0)) if isinstance(comparison, Mapping) else 0

    sensitivity_interval: Optional[Dict[str, Any]] = None
    specificity_interval: Optional[Dict[str, Any]] = None
    leniency_interval: Optional[Dict[str, Any]] = None
    leniency_identity: Optional[bool] = None
    if isinstance(confusion, Mapping):
        true_positive = int(confusion.get("true_positive", 0))
        false_negative = int(confusion.get("false_negative", 0))
        false_positive = int(confusion.get("false_positive", 0))
        true_negative = int(confusion.get("true_negative", 0))
        sensitivity_interval = wilson_interval(true_positive, true_positive + false_negative)
        specificity_interval = wilson_interval(true_negative, true_negative + false_positive)
        leniency_interval = paired_difference_interval(false_positive, false_negative, comparable)
        if leniency is not None and comparable:
            leniency_identity = math.isclose(
                float(leniency), (false_positive - false_negative) / comparable, rel_tol=0.0, abs_tol=1e-12
            )

    clips = tuple(validate_manifest(manifest)) if manifest is not None else ()
    split_ids: Dict[str, Any] = {
        "manifest_id": manifest_id,
        "scenario_manifest_sha256": scenario_manifest_sha256,
        "calibration_manifest_hash": calibration_manifest_hash(clips) if clips else None,
        "development_clip_ids": [clip.clip_id for clip in clips if clip.split == DEVELOPMENT_SPLIT],
        "heldout_clip_ids": [clip.clip_id for clip in clips if clip.split == HELDOUT_SPLIT],
    }
    annotation_ids = [
        {"clip_id": row.get("clip_id"), "annotator_id": row.get("annotator_id")}
        for row in sorted(
            annotation_rows or (),
            key=lambda item: (str(item.get("clip_id")), str(item.get("annotator_id"))),
        )
    ]
    for entry in annotation_ids:
        annotator_id = entry.get("annotator_id")
        if _nonempty_string(annotator_id):
            entry["annotator_type"] = registry.get(str(annotator_id)).annotator_type

    wrapped_gate_d = gate_d_with_limitations(gate_d, registry)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now(),
        "calibration_class": registry.calibration_class,
        "annotator_registry": registry.as_mapping(),
        "human_annotation_present": registry.human_count > 0,
        "has_two_blinded_humans": registry.has_two_blinded_humans,
        "split_ids": split_ids,
        "annotation_ids": annotation_ids,
        "annotation_count": len(annotation_ids),
        "human_annotation_coverage": report.get("human_annotation_coverage"),
        "human_human_agreement": {
            "pooled": pooled,
            "task_exploratory": agreement.get("task_exploratory") if isinstance(agreement, Mapping) else None,
            "interpretation": HUMAN_AGREEMENT_INTERPRETATION,
        },
        "cohens_kappa": {
            "binary_success": _kappa_of(pooled, "binary_success"),
            "integrity": _kappa_of(pooled, "integrity"),
            "collision": _kappa_of(pooled, "collision"),
            "completion_evidence": _kappa_of(pooled, "completion_evidence"),
            "note": (
                "Cohen's kappa on the two independent annotators' held-out labels. A null value means "
                "the expected-agreement denominator was degenerate, not that agreement was zero."
            ),
        },
        "progress_quadratic_weighted_kappa": pooled.get("progress")
        if isinstance(pooled, Mapping)
        else None,
        "heldout_consensus": report.get("heldout_consensus"),
        "judge_comparison_source": comparison_source,
        "confusion_matrix": confusion,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "leniency_offset": {
            "value": leniency,
            "definition": LENIENCY_DEFINITION,
            "equals_false_positive_minus_false_negative_over_comparable": leniency_identity,
        },
        "uncertainty": {
            "sensitivity_interval": sensitivity_interval,
            "specificity_interval": specificity_interval,
            "leniency_offset_interval": leniency_interval,
            "null_coverage": comparison.get("null_coverage") if isinstance(comparison, Mapping) else None,
            "comparable_count": comparable,
            "note": (
                "Intervals are sampling intervals for this enriched panel only. No interval is "
                "reported for kappa: this protocol preregistered no kappa variance estimator."
            ),
        },
        "stratification_disclosure": STRATIFICATION_DISCLOSURE,
        "unresolved_disagreement_policy": (
            "Identical labels define consensus. Disagreement stays unresolved and remains in "
            "coverage/missingness counts. No project-team tie-break after seeing VLM output."
        ),
        "gate_d": wrapped_gate_d,
        "open_dependencies": wrapped_gate_d["open_dependencies"],
        "primary_judge_evidence": report.get("primary_judge_evidence"),
        "manifest": report.get("manifest"),
        "judge_vs_human_descriptive": report.get("judge_vs_human"),
        "qualified": False,
        "qualification_note": (
            "This file records calibration evidence. It does not by itself qualify Gate D or the "
            "primary scoring that depends on it."
        ),
    }
    assert_finite_json(payload)
    return payload


def _kappa_of(pooled: Any, field_name: str) -> Optional[float]:
    if not isinstance(pooled, Mapping):
        return None
    block = pooled.get(field_name)
    if not isinstance(block, Mapping):
        return None
    value = block.get("kappa")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def assert_finite_json(payload: Any, *, context: str = "payload") -> None:
    """Refuse NaN/infinity before writing; an undefined value must be null."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            assert_finite_json(value, context="%s.%s" % (context, key))
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_finite_json(value, context="%s[%d]" % (context, index))
        return
    if isinstance(payload, float) and not math.isfinite(payload):
        raise AnnotationError(
            "%s is %r; an undefined statistic must be null, never NaN or infinity" % (context, payload)
        )


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


def write_json_atomic(path: Union[str, Path], payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Serialize strictly, then replace the destination in one atomic step."""

    assert_finite_json(payload)
    destination = Path(path)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _atomic_write_text(destination, encoded)
    return {"path": str(destination), "bytes": len(encoded.encode("utf-8"))}


def write_judge_calibration_json(
    path: Union[str, Path],
    report: Mapping[str, Any],
    registry: AnnotatorRegistry,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build and atomically write ``results/judge_calibration.json``."""

    payload = build_judge_calibration_payload(report, registry, **kwargs)
    written = write_json_atomic(path, payload)
    return {
        "path": written["path"],
        "bytes": written["bytes"],
        "calibration_class": payload["calibration_class"],
        "gate_d_status": payload["gate_d"]["status"],
        "open_dependencies": payload["gate_d"]["open_dependency_names"],
    }


def write_judge_calibration_artifact(
    store: Any, report: Mapping[str, Any], registry: AnnotatorRegistry, **kwargs: Any
) -> Any:
    """Hand the payload to an artifact store's ``write_judge_calibration``.

    ``store`` is duck-typed on purpose: ``plumb.artifacts.ArtifactStore`` already
    owns ``results/judge_calibration.json`` for ``plumb publish``, and this keeps
    the content builder here without importing that module.
    """

    writer = getattr(store, "write_judge_calibration", None)
    if not callable(writer):
        raise AnnotationError("store must expose write_judge_calibration(payload)")
    return writer(build_judge_calibration_payload(report, registry, **kwargs))


def _load_manifest_rows(path: Union[str, Path]) -> List[Dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise AnnotationError("calibration manifest does not exist: %s" % source)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as error:
                raise AnnotationError("invalid JSONL on line %d of %s" % (number, source)) from error
            if not isinstance(parsed, Mapping):
                raise AnnotationError("calibration manifest rows must be objects")
            rows.append(dict(parsed))
        return rows
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise AnnotationError("calibration manifest JSON is invalid") from error
    if isinstance(parsed, Mapping):
        parsed = parsed.get("clips")
    if not isinstance(parsed, list):
        raise AnnotationError("calibration manifest JSON must be a list or an object with clips")
    return [dict(row) for row in parsed]


def _service_from_args(args: argparse.Namespace) -> AnnotationService:
    registry = AnnotatorRegistry.load(args.annotators)
    return AnnotationService(
        _load_manifest_rows(args.manifest),
        registry,
        AnnotationStore(args.store),
        require_full_plan=not args.allow_partial_manifest,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint: ``python -m plumb.annotation <command>``."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    # The interactive app surface owns sessions and its append-only store.  The
    # offline packet/freeze workflow owns immutable selected manifests and is
    # intentionally routed to calibration's stricter CLI.  ``--store`` is the
    # unambiguous legacy/app marker for the overlapping export/report verbs.
    if arguments and (
        arguments[0] == "freeze" or (arguments[0] in {"export", "report"} and "--store" not in arguments)
    ):
        return _calibration_main(arguments)

    parser = argparse.ArgumentParser(description="Nightshift blinded annotation surface")
    subcommands = parser.add_subparsers(dest="command", required=True)

    def shared(command: argparse.ArgumentParser) -> None:
        command.add_argument("--manifest", required=True, help="Frozen 150-clip calibration manifest.")
        command.add_argument("--annotators", required=True, help="Annotator registry JSON.")
        command.add_argument("--store", required=True, help="Append-only label store JSONL.")
        command.add_argument(
            "--allow-partial-manifest",
            action="store_true",
            help="Permit a manifest that is not yet the full 150-clip plan (development only).",
        )

    register = subcommands.add_parser("register", help="Write a two-annotator registry with explicit types")
    register.add_argument("--output", required=True)
    register.add_argument(
        "--annotator",
        action="append",
        required=True,
        metavar="JSON",
        help='One annotator object, e.g. \'{"annotator_id":"human:alex","annotator_type":"human"}\'.',
    )

    session = subcommands.add_parser("session", help="Blinded session summary for one annotator")
    shared(session)
    session.add_argument("--annotator-id", required=True)

    queue = subcommands.add_parser("queue", help="All blinded clip views assigned to one annotator")
    shared(queue)
    queue.add_argument("--annotator-id", required=True)

    show = subcommands.add_parser("show", help="One blinded clip view")
    shared(show)
    show.add_argument("--annotator-id", required=True)
    show.add_argument("--clip-id", required=True)

    submit = subcommands.add_parser("submit", help="Record one blinded label")
    shared(submit)
    submit.add_argument("--annotator-id", required=True)
    submit.add_argument("--clip-id", required=True)
    submit.add_argument("--integrity", choices=INTEGRITY_VALUES)
    submit.add_argument("--collision", choices=COLLISION_VALUES)
    submit.add_argument("--progress", type=int, choices=(0, 1, 2, 3, 4, 5))
    submit.add_argument("--completion-evidence", choices=COMPLETION_VALUES)
    submit.add_argument("--frames", help="JSON array of evidence frame indexes, e.g. [0,15].")
    submit.add_argument("--reason", default="", help="Concise observable reason.")

    status = subcommands.add_parser("status", help="Annotation coverage per annotator and split")
    shared(status)

    export = subcommands.add_parser("export", help="Write complete labels in calibration's label shape")
    shared(export)
    export.add_argument("--output", required=True)

    report_command = subcommands.add_parser("report", help="Write results/judge_calibration.json")
    shared(report_command)
    report_command.add_argument("--output", required=True)
    report_command.add_argument("--judge-reports", action="append")
    report_command.add_argument("--judge-annotations", action="append")
    report_command.add_argument("--protocol")
    report_command.add_argument("--tolerances")
    report_command.add_argument("--manifest-id")
    report_command.add_argument("--scenario-manifest-sha256")

    args = parser.parse_args(arguments)
    try:
        if args.command == "register":
            annotators = []
            for raw in args.annotator:
                try:
                    annotators.append(json.loads(raw))
                except json.JSONDecodeError as error:
                    raise AnnotationError("--annotator must be a JSON object: %s" % raw) from error
            registry = AnnotatorRegistry.from_mapping({"annotators": annotators})
            print(json.dumps(registry.save(args.output), sort_keys=True))
            return 0

        service = _service_from_args(args)
        if args.command == "session":
            print(json.dumps(service.session(args.annotator_id), sort_keys=True, indent=2))
        elif args.command == "queue":
            print(json.dumps(service.queue(args.annotator_id), sort_keys=True, indent=2))
        elif args.command == "show":
            print(json.dumps(service.clip(args.annotator_id, args.clip_id), sort_keys=True, indent=2))
        elif args.command == "submit":
            payload = {
                "clip_id": args.clip_id,
                "annotator_id": args.annotator_id,
                "integrity": args.integrity,
                "collision": args.collision,
                "progress": args.progress,
                "completion_evidence": args.completion_evidence,
                "evidence_frame_indices": args.frames or "",
                "observable_reason": args.reason,
            }
            print(json.dumps(service.submit(payload), sort_keys=True, indent=2))
        elif args.command == "status":
            print(json.dumps(service.progress(), sort_keys=True, indent=2))
        elif args.command == "export":
            print(json.dumps(service.store.export_annotation_rows(args.output), sort_keys=True))
        elif args.command == "report":
            judge_reports = _load_report_list(args.judge_reports)
            judge_annotations = _load_report_list(args.judge_annotations)
            protocol = _load_json_object(args.protocol) if args.protocol else None
            tolerances = _load_json_object(args.tolerances) if args.tolerances else None
            rows = service.annotation_rows()
            calibration_report = build_calibration_report(
                service.clips,
                rows,
                service.registry.annotator_ids,
                judge_annotations=judge_annotations or None,
                judge_reports=judge_reports or None,
                protocol=protocol,
                tolerances=tolerances,
            )
            written = write_judge_calibration_json(
                args.output,
                calibration_report,
                service.registry,
                manifest=service.clips,
                annotation_rows=rows,
                manifest_id=args.manifest_id,
                scenario_manifest_sha256=args.scenario_manifest_sha256,
            )
            print(json.dumps(written, sort_keys=True, indent=2))
        return 0
    except (AnnotationError, CalibrationError) as error:
        parser.error(str(error))
    return 2


def _load_json_object(path: Union[str, Path]) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnnotationError("unable to read JSON object %s" % path) from error
    if not isinstance(value, Mapping):
        raise AnnotationError("%s must contain a JSON object" % path)
    return value


def _load_report_list(paths: Optional[Sequence[str]]) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    for path in paths or ():
        source = Path(path)
        if not source.is_file():
            raise AnnotationError("input does not exist: %s" % source)
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".jsonl":
            for number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AnnotationError("invalid JSONL on line %d of %s" % (number, source)) from error
                if not isinstance(parsed, Mapping):
                    raise AnnotationError("JSONL rows must be objects")
                rows.append(dict(parsed))
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise AnnotationError("invalid JSON in %s" % source) from error
        if isinstance(parsed, Mapping):
            if isinstance(parsed.get("reports"), list):
                rows.extend(dict(item) for item in parsed["reports"] if isinstance(item, Mapping))
            else:
                rows.append(dict(parsed))
        elif isinstance(parsed, list):
            rows.extend(dict(item) for item in parsed if isinstance(item, Mapping))
        else:
            raise AnnotationError("%s must hold an object or a list of objects" % source)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANNOTATOR_ID_PREFIXES",
    "ANNOTATOR_TYPES",
    "AnnotationError",
    "AnnotationService",
    "AnnotationStore",
    "AnnotatorIdentity",
    "AnnotatorRegistry",
    "EXTERNAL_LABEL_ANNOTATOR",
    "HUMAN_ANNOTATION_DEPENDENCY",
    "HUMAN_ANNOTATOR",
    "HUMAN_AGREEMENT_INTERPRETATION",
    "LENIENCY_DEFINITION",
    "LabelSubmission",
    "MODEL_ANNOTATOR",
    "PROGRESS_SCALE_LABELS",
    "SECOND_HUMAN_DEPENDENCY",
    "STRATIFICATION_DISCLOSURE",
    "VIEW_ONLY_FIELDS",
    "assert_blinded",
    "assert_finite_json",
    "blinded_clip_view",
    "build_judge_calibration_payload",
    "gate_d_evidence",
    "gate_d_measurements",
    "gate_d_with_limitations",
    "main",
    "paired_difference_interval",
    "utc_now",
    "wilson_interval",
    "write_json_atomic",
    "write_judge_calibration_artifact",
    "write_judge_calibration_json",
    # Offline immutable-selection and opaque-media helpers, re-exported so
    # callers can remain on the public annotation namespace.
    "Annotation",
    "AnnotatorOwnership",
    "CalibrationError",
    "ClipManifestRow",
    "blinded_annotation_rows",
    "blinded_clip_id",
    "blinded_media_ref",
    "blinded_media_resolver",
    "build_calibration_report",
    "deterministic_annotation_assignments",
    "export_annotation_packets",
    "freeze_calibration_selection",
    "import_annotations",
    "load_frozen_calibration_selection",
    "main",
    "validate_calibration_lineage_partition",
    "validate_annotator_ownership",
]
