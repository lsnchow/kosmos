"""Judge distillation on Baseten Training Jobs as a separate judge revision.

Spec section 5 closes with the paragraph this module implements:

    Distill on Baseten Training Jobs as a separate judge revision.  Keep
    development/training, held-out calibration and primary evaluation lineages
    disjoint.  LoRA ``lr=1e-3, r=64, alpha=32`` is one candidate configuration,
    not a portable optimum; compare a preregistered small search with early
    stopping on development validation.  Training labels/compute are an
    additional acquisition/budget item, not supplied by the 50 held-out clips.
    A distilled judge must pass fresh held-out calibration and paired
    frozen-video comparison before scoring a burst; preserve the original
    judge's results.

Every clause above is a refusal here, not a comment:

``DistillationDataset``
    Cannot be constructed if a training or development-validation source
    lineage also appears in the Gate-D held-out set, a fresh held-out set, or
    the primary study.  The check is ``plumb.measurement.
    validate_source_lineage_leakage``, the same one the primary study uses.
``LoRASearchSpace``
    Must contain both named arms -- the published ``lr=1e-3, r=64, alpha=32``
    and the policy-repository default ``lr=5e-4, r=32`` -- and must select on
    development validation.  ``select_arm`` refuses to name a winner from a
    result that was scored on any held-out split, and refuses to drop a losing
    or failed arm from the record.
``TrainingJobSpec``
    Renders the Training Jobs payload and never submits it.  The route and
    accepted body must be confirmed against the Training Jobs reference at
    deploy time, and framework/model compatibility is checked in the training
    container -- not asserted here (spec section 7).
``DistilledJudgeRevision``
    A new revision with its own identity hash, its own results reference, and a
    recorded base-judge revision.  A revision whose results reference equals
    the base judge's is refused: the original judge's results are preserved.
``DistillationGate``
    Emits the ``distilled_judge`` mapping that ``plumb.gates`` Gate E reads,
    with ``separately_validated`` false until *both* a fresh held-out
    calibration and a paired frozen-video comparison exist and pass.  It never
    emits ``used_for_scoring: true`` while any blocker stands.

Nothing here trains, submits, or measures.  No loss, accuracy, adapter hash,
GPU-second or dollar amount is ever produced by this module: every such value
arrives from a recorded training job or stays ``None``, which means unknown and
never zero.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .annotation import paired_difference_interval
from .calibration import (
    DEVELOPMENT_SPLIT,
    HELDOUT_SPLIT,
    ClipManifestRow,
    FrozenGateDProtocol,
    GateDTolerances,
    build_calibration_report,
    calibration_manifest_hash,
    validate_manifest,
)
from .gates import canonical_json_sha256
from .measurement import validate_source_lineage_leakage
from .protocol import FROZEN_JUDGE_SAMPLING


SCHEMA_VERSION = 1

#: Cohort labels handed to ``validate_source_lineage_leakage``.  A source
#: lineage that appears under two of these is a leakage violation, which is the
#: whole point: these five cohorts must stay disjoint (spec section 5).
TRAIN_COHORT = "distillation_train"
DEV_VALIDATION_COHORT = "distillation_dev_validation"
GATE_D_HELDOUT_COHORT = "gate_d_heldout"
FRESH_HELDOUT_COHORT = "distillation_fresh_heldout"
PRIMARY_COHORT = "primary"
DISJOINT_COHORTS: Tuple[str, ...] = (
    TRAIN_COHORT,
    DEV_VALIDATION_COHORT,
    GATE_D_HELDOUT_COHORT,
    FRESH_HELDOUT_COHORT,
    PRIMARY_COHORT,
)

#: The only split a search arm may be selected on.  Anything else is tuning
#: against a test set.
SELECTION_SPLIT = "development_validation"
_FORBIDDEN_SELECTION_TOKENS: Tuple[str, ...] = ("heldout", "held_out", "held-out", "test", "primary")

#: ``GpuSecondModel(prior_cost_reports={...})`` already reserves this key for
#: "prior development/training/calibration spend, reported separately and never
#: recorded as zero" (spec section 7).  Distillation is that line item.
BUDGET_ITEM = "judge_distillation"

#: Documentation the spec cites for these numbers.  Recorded so a reader can
#: check the provenance of every hyperparameter rather than trusting this file.
PUBLISHED_LORA_REFERENCE = "https://www.baseten.co/blog/practical-lora-research/"
TRAINING_JOBS_REFERENCE = "https://docs.baseten.co/training/overview"

#: Frameworks BUILD-SPEC.md names for Training Jobs.  Compatibility with a
#: given model is checked in the training container, not asserted here.
SUPPORTED_FRAMEWORKS: Tuple[str, ...] = ("trl", "axolotl")

#: A distilled judge may only substitute for the primary judge if it emits the
#: frozen rubric schema.  A scalar/feature head (for example a V-JEPA video
#: encoder with a regression head) is a separate exploratory analysis and can
#: never be authorised to score a burst: the schema, enums and label mapping are
#: part of the frozen judge protocol (spec section 5).
RUBRIC_JSON_CONTRACT = "rubric_json_schema"
SCALAR_HEAD_CONTRACT = "scalar_head"
OUTPUT_CONTRACTS: Tuple[str, ...] = (RUBRIC_JSON_CONTRACT, SCALAR_HEAD_CONTRACT)

#: Where training labels may come from.  The 50 held-out clips are not on this
#: list of *sources* and are excluded by lineage besides.
LABEL_SOURCES: Tuple[str, ...] = (
    "original_judge_aggregate",
    "original_judge_raw_samples",
    "human",
    "external_label",
)

#: A training job's own lifecycle, separate from any measurement.  ``not_run``
#: is the honest starting value; ``failed`` and ``cancelled`` arms stay in the
#: search record rather than disappearing from it.
ARM_STATUSES: Tuple[str, ...] = (
    "not_run",
    "running",
    "completed",
    "early_stopped",
    "failed",
    "cancelled",
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OFFSET_TIMESTAMP_RE = re.compile(r"(?:[+-]\d{2}:?\d{2}|Z)$")


class DistillationError(ValueError):
    """Raised for a leakage, provenance, or selection-integrity violation."""


# --------------------------------------------------------------------------
# Small strict validators.  These exist so a malformed record cannot reach a
# gate mapping, not to duplicate the statistics in plumb/calibration.py.
# --------------------------------------------------------------------------


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DistillationError("{0} must be a non-empty string".format(field_name))
    return value


def _sha256(value: Any, field_name: str) -> str:
    text = _text(value, field_name)
    if not _SHA256_RE.match(text):
        raise DistillationError("{0} must be a 'sha256:<64 hex>' digest".format(field_name))
    return text


def _optional_sha256(value: Any, field_name: str) -> Optional[str]:
    return None if value is None else _sha256(value, field_name)


def _timestamp(value: Any, field_name: str) -> str:
    text = _text(value, field_name)
    if not _OFFSET_TIMESTAMP_RE.search(text):
        raise DistillationError("{0} must be an ISO-8601 timestamp with a UTC offset".format(field_name))
    return text


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DistillationError("{0} must be a positive integer".format(field_name))
    return value


def _positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DistillationError("{0} must be a number".format(field_name))
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise DistillationError("{0} must be a finite positive number".format(field_name))
    return number


def _nonnegative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DistillationError("{0} must be a number".format(field_name))
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise DistillationError("{0} must be a finite non-negative number".format(field_name))
    return number


def _optional_nonnegative_float(value: Any, field_name: str) -> Optional[float]:
    return None if value is None else _nonnegative_float(value, field_name)


def _optional_finite_float(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DistillationError("{0} must be a number".format(field_name))
    number = float(value)
    if not math.isfinite(number):
        raise DistillationError("{0} must be finite; NaN/inf cannot reach an artifact".format(field_name))
    return number


def _optional_positive_int(value: Any, field_name: str) -> Optional[int]:
    return None if value is None else _positive_int(value, field_name)


def _enum(value: Any, allowed: Sequence[str], field_name: str) -> str:
    text = _text(value, field_name)
    if text not in allowed:
        raise DistillationError(
            "{0} must be one of {1}".format(field_name, ", ".join(allowed))
        )
    return text


def _string_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise DistillationError("{0} must be a sequence of strings, not one string".format(field_name))
    return tuple(_text(item, "{0} entry".format(field_name)) for item in value)


def _mentions_forbidden_split(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _FORBIDDEN_SELECTION_TOKENS)


# --------------------------------------------------------------------------
# Dataset provenance and the hard disjointness assertion
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipLineage:
    """One clip's identity and the source state it descends from."""

    clip_id: str
    source_lineage_id: str
    task: str

    def __post_init__(self) -> None:
        _text(self.clip_id, "clip_id")
        _text(self.source_lineage_id, "source_lineage_id")
        _text(self.task, "task")

    def as_cohort_record(self, cohort: str) -> Dict[str, Any]:
        """Shape this row the way ``validate_source_lineage_leakage`` reads it."""

        return {
            "start_lineage_id": self.source_lineage_id,
            "cohort": cohort,
            "clip_id": self.clip_id,
            "task": self.task,
        }

    def as_mapping(self) -> Dict[str, Any]:
        return {"clip_id": self.clip_id, "source_lineage_id": self.source_lineage_id, "task": self.task}


@dataclass(frozen=True)
class LineageExclusions:
    """Lineages distillation data must never touch, named by their cohort.

    The Gate-D held-out lineages are mandatory: the 50 clips that qualified the
    original judge do not supply training labels (spec section 5).  The primary
    study lineages are mandatory too, because a judge trained on a lineage it
    later scores is not measuring that lineage.
    """

    gate_d_heldout: Tuple[str, ...]
    primary_study: Tuple[str, ...]
    fresh_heldout: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.gate_d_heldout:
            raise DistillationError(
                "the Gate-D held-out lineages must be supplied; disjointness cannot be asserted against an unknown set"
            )
        if not self.primary_study:
            raise DistillationError(
                "the primary-study lineages must be supplied; disjointness cannot be asserted against an unknown set"
            )
        _string_tuple(self.gate_d_heldout, "gate_d_heldout")
        _string_tuple(self.primary_study, "primary_study")
        _string_tuple(self.fresh_heldout, "fresh_heldout")

    def cohort_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for cohort, lineages in (
            (GATE_D_HELDOUT_COHORT, self.gate_d_heldout),
            (PRIMARY_COHORT, self.primary_study),
            (FRESH_HELDOUT_COHORT, self.fresh_heldout),
        ):
            for lineage in lineages:
                records.append({"start_lineage_id": lineage, "cohort": cohort})
        return records

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "gate_d_heldout_lineages": len(self.gate_d_heldout),
            "primary_study_lineages": len(self.primary_study),
            "fresh_heldout_lineages": len(self.fresh_heldout),
        }


@dataclass(frozen=True)
class TrainingBudgetItem:
    """Training labels and compute as an additional acquisition/budget item.

    Spec section 5: training labels/compute are an additional acquisition and
    budget item, not supplied by the 50 held-out clips.  Spec section 7: prior
    development/training/calibration costs are reported separately, not as
    zero.  ``as_prior_cost_report`` produces the exact mapping
    ``plumb.sweeps.GpuSecondModel(prior_cost_reports=...)`` expects.
    """

    label_count: int
    label_source: str
    budget_item: str = BUDGET_ITEM
    supplied_by_gate_d_heldout: bool = False
    reported_gpu_seconds: Optional[float] = None
    estimated_usd: Optional[float] = None
    cost_basis_ref: Optional[str] = None
    usd_unavailable_reason: Optional[str] = None
    settled_usd: Optional[float] = None
    settled_reconciliation_ref: Optional[str] = None

    def __post_init__(self) -> None:
        _positive_int(self.label_count, "label_count")
        _enum(self.label_source, LABEL_SOURCES, "label_source")
        _text(self.budget_item, "budget_item")
        if self.supplied_by_gate_d_heldout:
            raise DistillationError(
                "training labels are an additional acquisition item; the 50 held-out clips do not supply them"
            )
        _optional_nonnegative_float(self.reported_gpu_seconds, "reported_gpu_seconds")
        _optional_nonnegative_float(self.estimated_usd, "estimated_usd")
        _optional_nonnegative_float(self.settled_usd, "settled_usd")
        if self.estimated_usd is not None:
            if float(self.estimated_usd) == 0.0:
                raise DistillationError(
                    "distillation cost is never zero; report it as unknown with a reason instead"
                )
            if self.cost_basis_ref is None:
                raise DistillationError("a USD amount requires a recorded cost_basis_ref pricing snapshot")
        elif not self.usd_unavailable_reason:
            raise DistillationError(
                "an unknown distillation cost needs a usd_unavailable_reason; silence would read as zero"
            )
        if self.settled_usd is not None:
            if float(self.settled_usd) == 0.0:
                raise DistillationError("a settled distillation cost of zero is not a reconciliation result")
            if self.settled_reconciliation_ref is None:
                raise DistillationError("a settled cost requires a billing reconciliation reference")

    @property
    def usd_status(self) -> str:
        return "estimated" if self.estimated_usd is not None else "unavailable"

    def as_prior_cost_report(self) -> Dict[str, Optional[float]]:
        """Return ``{"judge_distillation": <usd or None>}`` for the cost model."""

        return {self.budget_item: self.estimated_usd}

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "budget_item": self.budget_item,
            "attribution": "additional_acquisition_item_reported_separately",
            "label_count": self.label_count,
            "label_source": self.label_source,
            "supplied_by_gate_d_heldout": False,
            "reported_gpu_seconds": self.reported_gpu_seconds,
            "gpu_seconds_status": "reported" if self.reported_gpu_seconds is not None else "unavailable",
            "estimated_usd": self.estimated_usd,
            "usd_status": self.usd_status,
            "usd_unavailable_reason": self.usd_unavailable_reason,
            "cost_basis_ref": self.cost_basis_ref,
            "settled_usd": self.settled_usd,
            "settled_reconciliation_ref": self.settled_reconciliation_ref,
            "note": (
                "Distillation labels and compute are a separate budget line; they are never folded into the "
                "burst's marginal execution estimate and never recorded as zero."
            ),
        }


@dataclass(frozen=True)
class DistillationDataset:
    """Training and development-validation clips with lineage provenance.

    Construction runs ``plumb.measurement.validate_source_lineage_leakage``
    over the training split, the development-validation split, and the supplied
    exclusion cohorts.  Anything other than a ``pass`` raises: an
    ``unverifiable`` result (a row with no lineage ID) is not permission to
    proceed, it is an inability to establish the guarantee.
    """

    dataset_id: str
    train: Tuple[ClipLineage, ...]
    development_validation: Tuple[ClipLineage, ...]
    exclusions: LineageExclusions
    label_source: str
    budget: TrainingBudgetItem
    storage_uri: str
    frozen_at: str
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.dataset_id, "dataset_id")
        _text(self.storage_uri, "storage_uri")
        _timestamp(self.frozen_at, "frozen_at")
        _enum(self.label_source, LABEL_SOURCES, "label_source")
        _string_tuple(self.notes, "notes")
        if not isinstance(self.exclusions, LineageExclusions):
            raise DistillationError("exclusions must be a LineageExclusions")
        if not isinstance(self.budget, TrainingBudgetItem):
            raise DistillationError("budget must be a TrainingBudgetItem")
        if self.budget.label_source != self.label_source:
            raise DistillationError("dataset label_source and budget label_source disagree")
        for name, rows in (("train", self.train), ("development_validation", self.development_validation)):
            if not rows:
                raise DistillationError("{0} split must contain at least one clip".format(name))
            for row in rows:
                if not isinstance(row, ClipLineage):
                    raise DistillationError("{0} split must contain ClipLineage rows".format(name))
        clip_ids = [row.clip_id for row in self.train + self.development_validation]
        duplicates = sorted({clip_id for clip_id in clip_ids if clip_ids.count(clip_id) > 1})
        if duplicates:
            raise DistillationError("clip IDs must be unique across splits: {0}".format(", ".join(duplicates)))
        leakage = self.lineage_leakage_report()
        if leakage["status"] != "pass":
            raise DistillationError(
                "distillation lineages are not disjoint ({0}): {1}".format(
                    leakage["status"], leakage.get("reason") or "no reason reported"
                )
            )

    # -- provenance ------------------------------------------------------

    def cohort_records(self) -> List[Dict[str, Any]]:
        records = [row.as_cohort_record(TRAIN_COHORT) for row in self.train]
        records.extend(row.as_cohort_record(DEV_VALIDATION_COHORT) for row in self.development_validation)
        records.extend(self.exclusions.cohort_records())
        return records

    def lineage_leakage_report(self) -> Dict[str, Any]:
        """Delegate the disjointness decision to the primary-study checker."""

        report = dict(validate_source_lineage_leakage(self.cohort_records()))
        report["checker"] = "plumb.measurement.validate_source_lineage_leakage"
        report["cohorts_checked"] = list(DISJOINT_COHORTS)
        return report

    def content_hash(self) -> str:
        """Hash the identity fields a training job actually consumes."""

        return canonical_json_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": self.dataset_id,
                "label_source": self.label_source,
                "storage_uri": self.storage_uri,
                "frozen_at": self.frozen_at,
                "train": sorted((row.as_mapping() for row in self.train), key=lambda row: row["clip_id"]),
                "development_validation": sorted(
                    (row.as_mapping() for row in self.development_validation),
                    key=lambda row: row["clip_id"],
                ),
            }
        )

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "content_hash": self.content_hash(),
            "storage_uri": self.storage_uri,
            "frozen_at": self.frozen_at,
            "label_source": self.label_source,
            "train_clips": len(self.train),
            "development_validation_clips": len(self.development_validation),
            "train_lineages": len({row.source_lineage_id for row in self.train}),
            "development_validation_lineages": len(
                {row.source_lineage_id for row in self.development_validation}
            ),
            "exclusions": self.exclusions.as_mapping(),
            "lineage_disjointness": self.lineage_leakage_report(),
            "budget": self.budget.as_mapping(),
            "notes": list(self.notes),
        }


def dataset_from_calibration_manifest(
    rows: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    *,
    dataset_id: str,
    exclusions: LineageExclusions,
    label_source: str,
    budget: TrainingBudgetItem,
    storage_uri: str,
    frozen_at: str,
    require_full_plan: bool = True,
) -> DistillationDataset:
    """Build a dataset from a calibration manifest's *development* clips only.

    ``plumb.calibration.validate_manifest`` owns the manifest contract, so this
    reuses it rather than reparsing rows.  Held-out rows are dropped here and
    the held-out lineages must still appear in ``exclusions``: dropping them
    from the training file is not the same as proving they never leaked.
    """

    clips = validate_manifest(rows, require_full_plan=require_full_plan)
    development = [clip for clip in clips if clip.split == DEVELOPMENT_SPLIT]
    if not development:
        raise DistillationError("a distillation dataset needs development clips; the held-out split is not training data")
    heldout_lineages = {clip.source_lineage_id for clip in clips if clip.split == HELDOUT_SPLIT}
    missing = sorted(heldout_lineages - set(exclusions.gate_d_heldout) - set(exclusions.fresh_heldout))
    if missing:
        raise DistillationError(
            "held-out lineages from this manifest are absent from the declared exclusions: {0}".format(
                ", ".join(missing)
            )
        )
    lineages = [ClipLineage(clip.clip_id, clip.source_lineage_id, clip.task) for clip in development]
    # Deterministic split: every fifth development clip, by sorted clip ID,
    # validates.  Early stopping needs a validation split the trainer never
    # fits on, and the rule must not depend on dict ordering.
    train = tuple(row for index, row in enumerate(lineages) if index % 5 != 4)
    validation = tuple(row for index, row in enumerate(lineages) if index % 5 == 4)
    if not validation:
        raise DistillationError(
            "too few development clips to hold out a validation split; early stopping would have nothing to monitor"
        )
    return DistillationDataset(
        dataset_id=dataset_id,
        train=train,
        development_validation=validation,
        exclusions=exclusions,
        label_source=label_source,
        budget=budget,
        storage_uri=storage_uri,
        frozen_at=frozen_at,
        notes=(
            "Training/validation split rule: sorted development clip IDs, every fifth clip validates.",
            "Held-out and primary lineages are excluded by declaration, not by omission.",
        ),
    )


# --------------------------------------------------------------------------
# The preregistered LoRA search
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRAConfig:
    """One candidate LoRA arm, with the provenance of each number recorded.

    ``published_optimum_claim`` may only be set on the exact configuration the
    published write-up reports.  Spec section 5 calls that configuration one
    candidate, not a portable optimum, so the flag records *where the numbers
    came from* and never that this arm is best here.  Fields whose value this
    project chose rather than read from a source are listed in
    ``unsourced_fields`` and travel into the preregistration hash.
    """

    arm_id: str
    learning_rate: float
    rank: int
    alpha: int
    epochs: int
    provenance: str
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    dropout: float = 0.0
    published_optimum_claim: bool = False
    unsourced_fields: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.arm_id, "arm_id")
        _text(self.provenance, "provenance")
        _positive_float(self.learning_rate, "learning_rate")
        _positive_int(self.rank, "rank")
        _positive_int(self.alpha, "alpha")
        _positive_int(self.epochs, "epochs")
        _string_tuple(self.target_modules, "target_modules")
        if not self.target_modules:
            raise DistillationError("target_modules must name at least one module")
        _nonnegative_float(self.dropout, "dropout")
        if float(self.dropout) >= 1:
            raise DistillationError("dropout must be below 1")
        known = {
            "learning_rate",
            "rank",
            "alpha",
            "epochs",
            "target_modules",
            "dropout",
        }
        unknown = sorted(set(self.unsourced_fields) - known)
        if unknown:
            raise DistillationError(
                "unsourced_fields must name real hyperparameters; unknown: {0}".format(", ".join(unknown))
            )
        if self.published_optimum_claim:
            if (float(self.learning_rate), int(self.rank), int(self.alpha)) != (1e-3, 64, 32):
                raise DistillationError(
                    "published_optimum_claim is reserved for lr=1e-3, r=64, alpha=32; this arm is not that configuration"
                )
            claimed = sorted({"learning_rate", "rank", "alpha", "epochs"} & set(self.unsourced_fields))
            if claimed:
                raise DistillationError(
                    "an arm cannot cite a published source for fields it also lists as unsourced: {0}".format(
                        ", ".join(claimed)
                    )
                )

    def hyperparameters(self) -> Dict[str, Any]:
        """The values a training job needs, and nothing else."""

        return {
            "learning_rate": float(self.learning_rate),
            "lora_r": int(self.rank),
            "lora_alpha": int(self.alpha),
            "lora_dropout": float(self.dropout),
            "lora_target_modules": list(self.target_modules),
            "num_train_epochs": int(self.epochs),
        }

    def fingerprint(self) -> str:
        return canonical_json_sha256({"arm_id": self.arm_id, **self.hyperparameters()})

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "fingerprint": self.fingerprint(),
            "hyperparameters": self.hyperparameters(),
            "provenance": self.provenance,
            "published_optimum_claim": self.published_optimum_claim,
            "unsourced_fields": list(self.unsourced_fields),
            "portability": "candidate_configuration_not_a_portable_optimum",
        }


#: The configuration BUILD-SPEC.md quotes out loud: "We used your published
#: LoRA optimum -- learning rate 1e-3, r=64, alpha 32, two epochs -- rather
#: than the repository default."  That sentence is only sayable if this arm
#: actually wins the development-validation search; ``select_arm`` is what
#: decides, and ``spoken_claim`` is what renders the honest version.
PUBLISHED_LORA_ARM = LoRAConfig(
    arm_id="published-lr1e-3-r64-a32-e2",
    learning_rate=1e-3,
    rank=64,
    alpha=32,
    epochs=2,
    provenance=(
        "AGENT-BUILD-SPEC.md section 5 and BUILD-SPEC.md 'Things to say, verbatim', citing "
        + PUBLISHED_LORA_REFERENCE
    ),
    published_optimum_claim=True,
    unsourced_fields=("target_modules", "dropout"),
)

#: The policy-repository default, the arm the published configuration is being
#: compared *against*.  Only lr and r come from that default; the remaining
#: values are this project's choices and say so.
REPO_DEFAULT_LORA_ARM = LoRAConfig(
    arm_id="repo-default-lr5e-4-r32",
    learning_rate=5e-4,
    rank=32,
    alpha=32,
    epochs=2,
    provenance="policy repository default (lr=5e-4, r=32); the comparison arm named in AGENT-BUILD-SPEC.md section 5",
    unsourced_fields=("alpha", "epochs", "target_modules", "dropout"),
)

_REQUIRED_ARM_SIGNATURES: Tuple[Tuple[float, int], ...] = (
    (1e-3, 64),
    (5e-4, 32),
)


@dataclass(frozen=True)
class EarlyStoppingRule:
    """Early stopping, monitored on development validation and nowhere else."""

    monitor: str
    mode: str
    patience: int
    min_delta: float
    max_epochs: int
    monitored_split: str = SELECTION_SPLIT

    def __post_init__(self) -> None:
        _text(self.monitor, "monitor")
        _enum(self.mode, ("min", "max"), "mode")
        _positive_int(self.patience, "patience")
        _nonnegative_float(self.min_delta, "min_delta")
        _positive_int(self.max_epochs, "max_epochs")
        if self.monitored_split != SELECTION_SPLIT:
            raise DistillationError(
                "early stopping must be monitored on {0}; watching a held-out split is tuning against it".format(
                    SELECTION_SPLIT
                )
            )
        if _mentions_forbidden_split(self.monitor):
            raise DistillationError(
                "early-stopping monitor {0!r} names a held-out/primary split".format(self.monitor)
            )

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": float(self.min_delta),
            "max_epochs": self.max_epochs,
            "monitored_split": self.monitored_split,
        }


@dataclass(frozen=True)
class LoRASearchSpace:
    """A small preregistered search: named arms, one selection rule, one hash.

    The hash is computed from the arms, the early-stopping rule, the selection
    rule and the preregistration timestamp.  It is a local content hash, not an
    independently timestamped preregistration; spec section 6 requires the
    remote record separately, and ``as_mapping`` says so.
    """

    search_id: str
    arms: Tuple[LoRAConfig, ...]
    early_stopping: EarlyStoppingRule
    selection_metric: str
    selection_direction: str
    preregistered_at: str
    selection_split: str = SELECTION_SPLIT
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.search_id, "search_id")
        _text(self.selection_metric, "selection_metric")
        _enum(self.selection_direction, ("minimize", "maximize"), "selection_direction")
        _timestamp(self.preregistered_at, "preregistered_at")
        _string_tuple(self.notes, "notes")
        if not isinstance(self.early_stopping, EarlyStoppingRule):
            raise DistillationError("early_stopping must be an EarlyStoppingRule")
        if self.selection_split != SELECTION_SPLIT:
            raise DistillationError(
                "a search winner may only be selected on {0}; {1!r} would tune against a test set".format(
                    SELECTION_SPLIT, self.selection_split
                )
            )
        if _mentions_forbidden_split(self.selection_metric):
            raise DistillationError(
                "selection_metric {0!r} names a held-out/primary split".format(self.selection_metric)
            )
        if len(self.arms) < 2:
            raise DistillationError("a search compares at least two arms; one arm is not a comparison")
        for arm in self.arms:
            if not isinstance(arm, LoRAConfig):
                raise DistillationError("arms must contain LoRAConfig objects")
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise DistillationError("arm_id values must be unique")
        signatures = {(float(arm.learning_rate), int(arm.rank)) for arm in self.arms}
        missing = [pair for pair in _REQUIRED_ARM_SIGNATURES if pair not in signatures]
        if missing:
            raise DistillationError(
                "the preregistered search must include both named arms; missing lr/r: {0}".format(
                    ", ".join("lr={0:g}/r={1}".format(lr, rank) for lr, rank in missing)
                )
            )

    def arm(self, arm_id: str) -> LoRAConfig:
        for candidate in self.arms:
            if candidate.arm_id == arm_id:
                return candidate
        raise DistillationError("unknown arm_id {0!r}".format(arm_id))

    def preregistration_hash(self) -> str:
        return canonical_json_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "search_id": self.search_id,
                "arms": [arm.as_mapping() for arm in sorted(self.arms, key=lambda item: item.arm_id)],
                "early_stopping": self.early_stopping.as_mapping(),
                "selection_metric": self.selection_metric,
                "selection_direction": self.selection_direction,
                "selection_split": self.selection_split,
                "preregistered_at": self.preregistered_at,
            }
        )

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "search_id": self.search_id,
            "preregistration_hash": self.preregistration_hash(),
            "preregistration_status": "local_content_hash_only_no_external_timestamp",
            "preregistered_at": self.preregistered_at,
            "arms": [arm.as_mapping() for arm in self.arms],
            "early_stopping": self.early_stopping.as_mapping(),
            "selection_metric": self.selection_metric,
            "selection_direction": self.selection_direction,
            "selection_split": self.selection_split,
            "notes": list(self.notes),
        }


def default_search_space(*, search_id: str, preregistered_at: str) -> LoRASearchSpace:
    """The two named arms plus one interpolating arm, selecting on dev validation.

    The third arm exists so the search can distinguish "the published optimum
    transfers" from "any larger rank helps"; its numbers are this project's
    choice and are marked unsourced.
    """

    middle = LoRAConfig(
        arm_id="interpolating-lr1e-3-r32-a32",
        learning_rate=1e-3,
        rank=32,
        alpha=32,
        epochs=2,
        provenance="project-chosen interpolation between the published arm and the repository default",
        unsourced_fields=("learning_rate", "rank", "alpha", "epochs", "target_modules", "dropout"),
    )
    return LoRASearchSpace(
        search_id=search_id,
        arms=(PUBLISHED_LORA_ARM, REPO_DEFAULT_LORA_ARM, middle),
        early_stopping=EarlyStoppingRule(
            monitor="development_validation_loss",
            mode="min",
            patience=2,
            min_delta=0.0,
            max_epochs=2,
        ),
        selection_metric="development_validation_loss",
        selection_direction="minimize",
        preregistered_at=preregistered_at,
        notes=(
            "LoRA lr=1e-3, r=64, alpha=32 is one candidate configuration, not a portable optimum.",
            "No arm may be compared on the held-out calibration set or the primary study.",
            "Every arm is recorded, including the ones that lose and the ones that fail.",
        ),
    )


@dataclass(frozen=True)
class ArmResult:
    """What one training job actually reported for one arm.

    Every number here is copied from a recorded training job.  ``None`` means
    the job did not report it, which is unknown -- never zero and never
    inferred.  A ``completed`` arm must carry both an adapter digest and a
    metric value, because an arm with neither did not complete.
    """

    arm_id: str
    status: str
    selection_metric: str
    metric_value: Optional[float] = None
    metric_source: Optional[str] = None
    selection_split: str = SELECTION_SPLIT
    training_job_id: Optional[str] = None
    adapter_uri: Optional[str] = None
    adapter_sha256: Optional[str] = None
    epochs_completed: Optional[int] = None
    early_stopped_at_epoch: Optional[int] = None
    reported_gpu_seconds: Optional[float] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.arm_id, "arm_id")
        _enum(self.status, ARM_STATUSES, "status")
        _text(self.selection_metric, "selection_metric")
        if self.selection_split != SELECTION_SPLIT:
            raise DistillationError(
                "an arm result scored on {0!r} cannot select a winner; only {1} may".format(
                    self.selection_split, SELECTION_SPLIT
                )
            )
        if _mentions_forbidden_split(self.selection_metric):
            raise DistillationError(
                "arm metric {0!r} names a held-out/primary split; a winner chosen on it would be invalid".format(
                    self.selection_metric
                )
            )
        _optional_finite_float(self.metric_value, "metric_value")
        _optional_sha256(self.adapter_sha256, "adapter_sha256")
        _optional_positive_int(self.epochs_completed, "epochs_completed")
        _optional_positive_int(self.early_stopped_at_epoch, "early_stopped_at_epoch")
        _optional_nonnegative_float(self.reported_gpu_seconds, "reported_gpu_seconds")
        if self.metric_value is not None and not self.metric_source:
            raise DistillationError("a metric value requires a metric_source naming where it was read")
        if self.completed:
            if self.metric_value is None:
                raise DistillationError(
                    "a completed arm must report its development-validation metric; an unreported metric is not a result"
                )
            if self.adapter_sha256 is None or not self.adapter_uri:
                raise DistillationError("a completed arm must carry the adapter URI and digest it produced")
            if not self.training_job_id:
                raise DistillationError("a completed arm must record the training job that produced it")
        if self.status in ("failed", "cancelled") and not self.reason:
            raise DistillationError("a failed or cancelled arm must record its reason")

    @property
    def completed(self) -> bool:
        return self.status in ("completed", "early_stopped")

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "status": self.status,
            "selection_metric": self.selection_metric,
            "selection_split": self.selection_split,
            "metric_value": self.metric_value,
            "metric_status": "reported" if self.metric_value is not None else "unavailable",
            "metric_source": self.metric_source,
            "training_job_id": self.training_job_id,
            "adapter_uri": self.adapter_uri,
            "adapter_sha256": self.adapter_sha256,
            "epochs_completed": self.epochs_completed,
            "early_stopped_at_epoch": self.early_stopped_at_epoch,
            "reported_gpu_seconds": self.reported_gpu_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SearchSelection:
    """The winner, the losers, and why the choice is or is not reportable."""

    search_id: str
    preregistration_hash: str
    status: str
    winner_arm_id: Optional[str]
    ranking: Tuple[Dict[str, Any], ...]
    reason_codes: Tuple[str, ...]
    selection_split: str = SELECTION_SPLIT

    @property
    def reportable(self) -> bool:
        return self.status == "selected" and self.winner_arm_id is not None

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "search_id": self.search_id,
            "preregistration_hash": self.preregistration_hash,
            "status": self.status,
            "winner_arm_id": self.winner_arm_id,
            "selection_split": self.selection_split,
            "reason_codes": list(self.reason_codes),
            "arms": [dict(entry) for entry in self.ranking],
            "note": (
                "Arms are ranked on development validation only. Losing and failed arms are retained; "
                "a search reported without them is not the search that was preregistered."
            ),
        }


def select_arm(search: LoRASearchSpace, results: Sequence[ArmResult]) -> SearchSelection:
    """Pick the development-validation winner, or refuse and say why.

    Refusals, all as ``status``/``reason_codes`` rather than a silent answer:
    an arm from the preregistered search is missing from the record; two arms
    tie on the selection metric; no arm completed.  ``ArmResult`` has already
    refused any result scored on a held-out split, so a winner chosen on
    held-out data cannot reach this function.
    """

    if not isinstance(search, LoRASearchSpace):
        raise DistillationError("search must be a LoRASearchSpace")
    for result in results:
        if not isinstance(result, ArmResult):
            raise DistillationError("results must contain ArmResult objects")
    seen = [result.arm_id for result in results]
    duplicates = sorted({arm_id for arm_id in seen if seen.count(arm_id) > 1})
    if duplicates:
        raise DistillationError("one result per arm; duplicated: {0}".format(", ".join(duplicates)))
    unknown = sorted(set(seen) - {arm.arm_id for arm in search.arms})
    if unknown:
        raise DistillationError(
            "results contain arms that were not preregistered: {0}".format(", ".join(unknown))
        )
    for result in results:
        if result.selection_metric != search.selection_metric:
            raise DistillationError(
                "arm {0} reports {1!r} but the search preregistered {2!r}".format(
                    result.arm_id, result.selection_metric, search.selection_metric
                )
            )

    reasons: List[str] = []
    missing = sorted({arm.arm_id for arm in search.arms} - set(seen))
    if missing:
        reasons.append("arms_missing_from_record")
    ranking: List[Dict[str, Any]] = []
    by_id = {result.arm_id: result for result in results}
    for arm in search.arms:
        result = by_id.get(arm.arm_id)
        entry = {
            "arm_id": arm.arm_id,
            "hyperparameters": arm.hyperparameters(),
            "published_optimum_claim": arm.published_optimum_claim,
        }
        entry.update(
            result.as_mapping()
            if result is not None
            else {"status": "not_recorded", "metric_value": None, "metric_status": "unavailable"}
        )
        ranking.append(entry)
    for arm_id in missing:
        reasons.append("arm_not_recorded:{0}".format(arm_id))
    for result in results:
        if result.status in ("failed", "cancelled"):
            reasons.append("arm_{0}:{1}".format(result.status, result.arm_id))

    finished = [result for result in results if result.completed]
    sign = -1.0 if search.selection_direction == "maximize" else 1.0

    def ordered(value: Optional[float]) -> float:
        """Rank better-first for either direction; unranked arms sort last."""

        return 0.0 if value is None else sign * float(value)

    ranking.sort(key=lambda entry: (entry.get("metric_value") is None, ordered(entry.get("metric_value")), entry["arm_id"]))
    if not finished:
        return SearchSelection(
            search_id=search.search_id,
            preregistration_hash=search.preregistration_hash(),
            status="not_selectable",
            winner_arm_id=None,
            ranking=tuple(ranking),
            reason_codes=tuple(reasons + ["no_arm_completed"]),
        )
    best = min(finished, key=lambda result: (ordered(result.metric_value), result.arm_id))
    tied = [
        result.arm_id
        for result in finished
        if result.arm_id != best.arm_id and ordered(result.metric_value) == ordered(best.metric_value)
    ]
    if tied:
        return SearchSelection(
            search_id=search.search_id,
            preregistration_hash=search.preregistration_hash(),
            status="tied",
            winner_arm_id=None,
            ranking=tuple(ranking),
            reason_codes=tuple(reasons + ["tie_on_selection_metric:" + ",".join(sorted([best.arm_id] + tied))]),
        )
    return SearchSelection(
        search_id=search.search_id,
        preregistration_hash=search.preregistration_hash(),
        status="selected" if not missing else "selected_with_gaps",
        winner_arm_id=best.arm_id,
        ranking=tuple(ranking),
        reason_codes=tuple(reasons),
    )


def spoken_claim(search: LoRASearchSpace, selection: SearchSelection) -> Dict[str, Any]:
    """Return the sentence the demo may say about the LoRA configuration.

    BUILD-SPEC.md lists "We used your published LoRA optimum -- learning rate
    1e-3, r=64, alpha 32, two epochs -- rather than the repository default" as a
    verbatim line.  It is only true if that arm won the development-validation
    search.  This returns the sanctioned wording for whichever outcome actually
    happened, so the script cannot outrun the result.
    """

    if not selection.reportable:
        return {
            "sayable": False,
            "reason": "no winner is reportable: " + (", ".join(selection.reason_codes) or selection.status),
            "sentence": None,
        }
    winner = search.arm(str(selection.winner_arm_id))
    if winner.published_optimum_claim:
        return {
            "sayable": True,
            "winner_arm_id": winner.arm_id,
            "sentence": (
                "We used your published LoRA optimum - learning rate 1e-3, r=64, alpha 32, two epochs - "
                "rather than the repository default, and it won our preregistered development-validation "
                "comparison against that default."
            ),
        }
    return {
        "sayable": False,
        "winner_arm_id": winner.arm_id,
        "reason": "the published configuration did not win this search",
        "sentence": (
            "We compared the published LoRA configuration - learning rate 1e-3, r=64, alpha 32 - against the "
            "repository default on a preregistered development-validation search. Arm {0} won here, which is "
            "why we do not present the published numbers as a portable optimum.".format(winner.arm_id)
        ),
    }


# --------------------------------------------------------------------------
# Judge revisions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentModel:
    """The model a LoRA adapter is trained onto, and what it can output."""

    model_id: str
    model_revision: str
    output_contract: str
    parameter_note: str = ""

    def __post_init__(self) -> None:
        _text(self.model_id, "model_id")
        _text(self.model_revision, "model_revision")
        _enum(self.output_contract, OUTPUT_CONTRACTS, "output_contract")

    @property
    def can_substitute_for_primary_judge(self) -> bool:
        """Only a model emitting the frozen rubric schema can score a burst."""

        return self.output_contract == RUBRIC_JSON_CONTRACT

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "output_contract": self.output_contract,
            "can_substitute_for_primary_judge": self.can_substitute_for_primary_judge,
            "parameter_note": self.parameter_note,
        }


@dataclass(frozen=True)
class BaseJudgeRevision:
    """The judge that qualified Gate D, and where its results live.

    ``results_ref`` is recorded so the distilled revision can be refused if it
    points at the same artifact: the original judge's results are preserved.
    """

    revision_id: str
    model_id: str
    model_revision: str
    rubric_hash: str
    sampling_hash: str
    gate_d_protocol_hash: str
    results_ref: str
    results_sha256: str

    def __post_init__(self) -> None:
        for name in ("revision_id", "model_id", "model_revision", "results_ref"):
            _text(getattr(self, name), name)
        _sha256(self.rubric_hash, "rubric_hash")
        _sha256(self.sampling_hash, "sampling_hash")
        _sha256(self.gate_d_protocol_hash, "gate_d_protocol_hash")
        _sha256(self.results_sha256, "results_sha256")

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "rubric_hash": self.rubric_hash,
            "sampling_hash": self.sampling_hash,
            "gate_d_protocol_hash": self.gate_d_protocol_hash,
            "results_ref": self.results_ref,
            "results_sha256": self.results_sha256,
        }


def frozen_sampling_for_distilled_judge(**overrides: Any) -> Dict[str, Any]:
    """Start from ``FROZEN_JUDGE_SAMPLING`` and require a quorum majority.

    Spec section 5 allows a revision to freeze a different sample count, but
    only together with its quorum, schema and cost model.  Overriding the count
    without the quorum is refused here rather than discovered later.
    """

    sampling: Dict[str, Any] = dict(FROZEN_JUDGE_SAMPLING)
    unknown = sorted(set(overrides) - set(sampling))
    if unknown:
        raise DistillationError(
            "unknown sampling field(s) {0}; a new field is a new schema and needs a new protocol".format(
                ", ".join(unknown)
            )
        )
    sampling.update(overrides)
    count = _positive_int(sampling["sample_count"], "sample_count")
    quorum = _positive_int(sampling["quorum"], "quorum")
    if quorum * 2 <= count:
        raise DistillationError("quorum must be a strict majority of sample_count")
    if quorum > count:
        raise DistillationError("quorum cannot exceed sample_count")
    if "sample_count" in overrides and "quorum" not in overrides:
        raise DistillationError(
            "changing sample_count requires explicitly refreezing the quorum alongside it"
        )
    return sampling


@dataclass(frozen=True)
class DistilledJudgeRevision:
    """A new judge revision with its own identity hash.

    It never replaces the base judge: ``base_judge`` is recorded, the base
    judge's ``results_ref`` is refused as this revision's own, and
    ``identity_hash`` differs from the base revision by construction.
    """

    revision_id: str
    base_judge: BaseJudgeRevision
    student: StudentModel
    adapter_uri: str
    adapter_sha256: str
    training_dataset_hash: str
    search_arm_id: str
    search_preregistration_hash: str
    training_job_id: str
    rubric_hash: str
    sampling: Mapping[str, Any]
    results_ref: str
    created_at: str

    def __post_init__(self) -> None:
        for name in ("revision_id", "adapter_uri", "search_arm_id", "training_job_id", "results_ref"):
            _text(getattr(self, name), name)
        if not isinstance(self.base_judge, BaseJudgeRevision):
            raise DistillationError("base_judge must be a BaseJudgeRevision")
        if not isinstance(self.student, StudentModel):
            raise DistillationError("student must be a StudentModel")
        _sha256(self.adapter_sha256, "adapter_sha256")
        _sha256(self.training_dataset_hash, "training_dataset_hash")
        _sha256(self.search_preregistration_hash, "search_preregistration_hash")
        _sha256(self.rubric_hash, "rubric_hash")
        _timestamp(self.created_at, "created_at")
        if self.revision_id == self.base_judge.revision_id:
            raise DistillationError(
                "a distilled judge is a separate revision; it cannot reuse the base judge's revision ID"
            )
        if self.results_ref == self.base_judge.results_ref:
            raise DistillationError(
                "the original judge's results are preserved; a distilled revision needs its own results reference"
            )
        if self.rubric_hash != self.base_judge.rubric_hash:
            raise DistillationError(
                "the rubric is frozen; a distilled judge scoring a different rubric is a different measurement"
            )
        if not isinstance(self.sampling, Mapping) or not self.sampling:
            raise DistillationError("sampling must be a non-empty mapping")
        frozen_sampling_for_distilled_judge(**dict(self.sampling))

    @property
    def sampling_hash(self) -> str:
        return canonical_json_sha256(dict(self.sampling))

    def identity_hash(self) -> str:
        """This revision's own hash, distinct from the base judge's."""

        return canonical_json_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "revision_id": self.revision_id,
                "base_judge_revision": self.base_judge.revision_id,
                "student": self.student.as_mapping(),
                "adapter_sha256": self.adapter_sha256,
                "training_dataset_hash": self.training_dataset_hash,
                "search_arm_id": self.search_arm_id,
                "search_preregistration_hash": self.search_preregistration_hash,
                "rubric_hash": self.rubric_hash,
                "sampling_hash": self.sampling_hash,
            }
        )

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "identity_hash": self.identity_hash(),
            "base_judge": self.base_judge.as_mapping(),
            "student": self.student.as_mapping(),
            "adapter_uri": self.adapter_uri,
            "adapter_sha256": self.adapter_sha256,
            "training_dataset_hash": self.training_dataset_hash,
            "search_arm_id": self.search_arm_id,
            "search_preregistration_hash": self.search_preregistration_hash,
            "training_job_id": self.training_job_id,
            "rubric_hash": self.rubric_hash,
            "sampling": dict(self.sampling),
            "sampling_hash": self.sampling_hash,
            "results_ref": self.results_ref,
            "created_at": self.created_at,
            "invalidates_gates": ["D", "E", "F"],
            "invalidation_note": (
                "A changed judge invalidates its dependent gates (spec section 0). This revision does not "
                "inherit the base judge's Gate D; it needs a fresh held-out calibration of its own."
            ),
        }


# --------------------------------------------------------------------------
# Training Jobs payload rendering.  Nothing here submits anything.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRuntime:
    """The training container, its accelerator, and what is still unverified.

    Spec section 7: framework/model compatibility is checked in its own
    container.  ``framework_compatibility_checked`` starts false and only a
    recorded container build may set it.
    """

    image: str
    python_version: str
    framework: str
    requirements_ref: str
    start_command: str
    accelerator: str
    accelerator_count: int = 1
    node_count: int = 1
    requirements_sha256: Optional[str] = None
    environment_variable_names: Tuple[str, ...] = ()
    secret_names: Tuple[str, ...] = ()
    framework_compatibility_checked: bool = False

    def __post_init__(self) -> None:
        for name in ("image", "python_version", "requirements_ref", "start_command", "accelerator"):
            _text(getattr(self, name), name)
        _enum(self.framework, SUPPORTED_FRAMEWORKS, "framework")
        _positive_int(self.accelerator_count, "accelerator_count")
        _positive_int(self.node_count, "node_count")
        _optional_sha256(self.requirements_sha256, "requirements_sha256")
        _string_tuple(self.environment_variable_names, "environment_variable_names")
        _string_tuple(self.secret_names, "secret_names")
        for name in self.environment_variable_names + self.secret_names:
            if "=" in name or name != name.strip():
                raise DistillationError(
                    "record environment variable and secret *names* only; {0!r} looks like a value".format(name)
                )

    def as_payload(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "python_version": self.python_version,
            "framework": self.framework,
            "requirements_ref": self.requirements_ref,
            "requirements_sha256": self.requirements_sha256,
            "start_command": self.start_command,
            "environment_variable_names": list(self.environment_variable_names),
            "secret_names": list(self.secret_names),
            "framework_compatibility": (
                "checked_in_training_container"
                if self.framework_compatibility_checked
                else "unchecked_until_the_training_container_builds"
            ),
        }

    def as_compute_payload(self) -> Dict[str, Any]:
        return {
            "accelerator": self.accelerator,
            "accelerator_count": self.accelerator_count,
            "node_count": self.node_count,
            "status": "requested_capacity_not_reserved_capacity",
        }


@dataclass(frozen=True)
class TrainingJobSpec:
    """One arm's Training Jobs request, rendered and never submitted.

    ``as_payload`` supplies the *values*; the exact route and accepted body
    must be confirmed against the Training Jobs reference at deploy time, the
    same discipline ``deploy/baseten/chain.py`` uses for its autoscaling
    patches.  There is deliberately no submit method in this module.
    """

    job_name: str
    project_name: str
    arm: LoRAConfig
    search: LoRASearchSpace
    dataset: DistillationDataset
    base_judge: BaseJudgeRevision
    student: StudentModel
    runtime: TrainingRuntime
    entrypoint: str = "deploy/baseten/training/train_judge_lora.py"

    def __post_init__(self) -> None:
        for name in ("job_name", "project_name", "entrypoint"):
            _text(getattr(self, name), name)
        if not isinstance(self.arm, LoRAConfig):
            raise DistillationError("arm must be a LoRAConfig")
        if not isinstance(self.search, LoRASearchSpace):
            raise DistillationError("search must be a LoRASearchSpace")
        if not isinstance(self.dataset, DistillationDataset):
            raise DistillationError("dataset must be a DistillationDataset")
        if not isinstance(self.base_judge, BaseJudgeRevision):
            raise DistillationError("base_judge must be a BaseJudgeRevision")
        if not isinstance(self.student, StudentModel):
            raise DistillationError("student must be a StudentModel")
        if not isinstance(self.runtime, TrainingRuntime):
            raise DistillationError("runtime must be a TrainingRuntime")
        self.search.arm(self.arm.arm_id)

    def submission_blockers(self, env: Optional[Mapping[str, str]] = None) -> Tuple[str, ...]:
        """What still stands between this rendered payload and a real job."""

        environment = os.environ if env is None else env
        blockers: List[str] = []
        if not str(environment.get("BASETEN_API_KEY", "")).strip():
            blockers.append("BASETEN_API_KEY is unset")
        if not self.runtime.framework_compatibility_checked:
            blockers.append(
                "framework/model compatibility has not been checked in the training container"
            )
        if self.runtime.requirements_sha256 is None:
            blockers.append("the training requirements file has no recorded digest")
        if not self.dataset.storage_uri.startswith(("s3://", "gs://", "https://")):
            blockers.append(
                "dataset storage_uri is not a remote object-store URI the training job can read"
            )
        blockers.append(
            "the Training Jobs route and body are unconfirmed against " + TRAINING_JOBS_REFERENCE
        )
        return tuple(blockers)

    def as_payload(self, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "rendered_not_submitted",
            "project_name": self.project_name,
            "job_name": self.job_name,
            "entrypoint": self.entrypoint,
            "compute": self.runtime.as_compute_payload(),
            "runtime": self.runtime.as_payload(),
            "hyperparameters": {
                **self.arm.hyperparameters(),
                "early_stopping": self.search.early_stopping.as_mapping(),
                "selection_metric": self.search.selection_metric,
                "selection_split": self.search.selection_split,
            },
            "datasets": {
                "train": {
                    "uri": self.dataset.storage_uri,
                    "split": "train",
                    "clips": len(self.dataset.train),
                    "cohort": TRAIN_COHORT,
                },
                "development_validation": {
                    "uri": self.dataset.storage_uri,
                    "split": SELECTION_SPLIT,
                    "clips": len(self.dataset.development_validation),
                    "cohort": DEV_VALIDATION_COHORT,
                },
                "dataset_id": self.dataset.dataset_id,
                "content_hash": self.dataset.content_hash(),
                "lineage_disjointness": self.dataset.lineage_leakage_report()["status"],
            },
            "base_judge": self.base_judge.as_mapping(),
            "student_model": self.student.as_mapping(),
            "search": {
                "search_id": self.search.search_id,
                "arm_id": self.arm.arm_id,
                "preregistration_hash": self.search.preregistration_hash(),
                "portability": "candidate_configuration_not_a_portable_optimum",
            },
            "budget": self.dataset.budget.as_mapping(),
            "route_status": "unconfirmed_against_training_jobs_reference",
            "route_reference": TRAINING_JOBS_REFERENCE,
            "rationale": (
                "Judge distillation runs on Training Jobs, not the RL SDK and not the serving Chain "
                "(AGENT-BUILD-SPEC.md section 7). Loops' access and model catalogue are not a dependency."
            ),
            "submission_blockers": list(self.submission_blockers(env)),
        }


def render_search_payloads(
    *,
    project_name: str,
    job_name_prefix: str,
    search: LoRASearchSpace,
    dataset: DistillationDataset,
    base_judge: BaseJudgeRevision,
    student: StudentModel,
    runtime: TrainingRuntime,
    env: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Render one payload per preregistered arm, in preregistration order."""

    _text(project_name, "project_name")
    _text(job_name_prefix, "job_name_prefix")
    return [
        TrainingJobSpec(
            job_name="{0}-{1}".format(job_name_prefix, arm.arm_id),
            project_name=project_name,
            arm=arm,
            search=search,
            dataset=dataset,
            base_judge=base_judge,
            student=student,
            runtime=runtime,
        ).as_payload(env)
        for arm in search.arms
    ]


# --------------------------------------------------------------------------
# Fresh held-out calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FreshHeldOutResult:
    """A distilled judge's own held-out calibration, on a set it never saw."""

    calibration_id: str
    status: str
    passed: bool
    fresh_manifest_hash: str
    original_manifest_hash: str
    disjointness: Mapping[str, Any]
    gate_d: Mapping[str, Any]
    reason_codes: Tuple[str, ...] = ()
    report_ref: Optional[str] = None

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "status": self.status,
            "passed": self.passed,
            "fresh_manifest_hash": self.fresh_manifest_hash,
            "original_manifest_hash": self.original_manifest_hash,
            "disjoint_from_original_heldout": self.disjointness,
            "gate_d": dict(self.gate_d),
            "reason_codes": list(self.reason_codes),
            "report_ref": self.report_ref,
            "statistics_owner": "plumb.calibration.build_calibration_report",
        }


def assert_fresh_heldout_panel(
    fresh_manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    original_manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    *,
    require_full_plan: bool = True,
) -> Dict[str, Any]:
    """Refuse a held-out panel that reuses the set which qualified the judge.

    Reusing those clips is tuning against the test set, so this raises rather
    than reporting a status.  The disjointness decision is delegated to
    ``plumb.measurement.validate_source_lineage_leakage``: the two held-out
    panels are handed to it as two cohorts, and only ``pass`` is accepted.
    """

    fresh = validate_manifest(fresh_manifest, require_full_plan=require_full_plan)
    original = validate_manifest(original_manifest, require_full_plan=require_full_plan)
    fresh_hash = calibration_manifest_hash(fresh)
    original_hash = calibration_manifest_hash(original)
    if fresh_hash == original_hash:
        raise DistillationError(
            "the fresh held-out panel is the panel that qualified the original judge; "
            "obtain a fresh held-out set instead of reusing the same test set"
        )
    fresh_heldout = [clip for clip in fresh if clip.split == HELDOUT_SPLIT]
    original_heldout = [clip for clip in original if clip.split == HELDOUT_SPLIT]
    if not fresh_heldout or not original_heldout:
        raise DistillationError("both panels must contain held-out clips to be compared")
    shared_clips = sorted({clip.clip_id for clip in fresh_heldout} & {clip.clip_id for clip in original_heldout})
    if shared_clips:
        raise DistillationError(
            "fresh held-out panel reuses original held-out clips: {0}".format(", ".join(shared_clips))
        )
    records = [
        {"start_lineage_id": clip.source_lineage_id, "cohort": FRESH_HELDOUT_COHORT, "clip_id": clip.clip_id}
        for clip in fresh_heldout
    ]
    records.extend(
        {"start_lineage_id": clip.source_lineage_id, "cohort": GATE_D_HELDOUT_COHORT, "clip_id": clip.clip_id}
        for clip in original_heldout
    )
    leakage = dict(validate_source_lineage_leakage(records))
    if leakage["status"] != "pass":
        raise DistillationError(
            "fresh held-out lineages are not disjoint from the original held-out set ({0}): {1}".format(
                leakage["status"], leakage.get("reason") or "no reason reported"
            )
        )
    leakage["checker"] = "plumb.measurement.validate_source_lineage_leakage"
    leakage["fresh_manifest_hash"] = fresh_hash
    leakage["original_manifest_hash"] = original_hash
    return leakage


def run_fresh_heldout_calibration(
    *,
    calibration_id: str,
    fresh_manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    original_manifest: Iterable[Union[ClipManifestRow, Mapping[str, Any]]],
    human_annotations: Iterable[Any],
    annotator_ids: Sequence[str],
    distilled_judge_annotations: Iterable[Any],
    distilled_judge_reports: Optional[Iterable[Any]],
    protocol: FrozenGateDProtocol,
    tolerances: GateDTolerances,
    base_judge: BaseJudgeRevision,
    require_full_plan: bool = True,
    report_ref: Optional[str] = None,
) -> Tuple[FreshHeldOutResult, Dict[str, Any]]:
    """Calibrate the distilled judge on a fresh panel, reusing Gate D's maths.

    ``plumb.calibration.build_calibration_report`` computes every statistic and
    ``plumb.calibration.evaluate_gate_d`` makes the threshold decision; nothing
    is recomputed here.  This function adds the two things the Gate-D workflow
    has no reason to check: that the panel is genuinely fresh, and that the
    frozen protocol being calibrated belongs to the distilled revision rather
    than the judge it replaces.
    """

    _text(calibration_id, "calibration_id")
    if not isinstance(protocol, FrozenGateDProtocol):
        raise DistillationError("protocol must be a FrozenGateDProtocol")
    if not isinstance(tolerances, GateDTolerances):
        raise DistillationError("tolerances must be a GateDTolerances")
    if not isinstance(base_judge, BaseJudgeRevision):
        raise DistillationError("base_judge must be a BaseJudgeRevision")
    if protocol.protocol_hash == base_judge.gate_d_protocol_hash:
        raise DistillationError(
            "a distilled judge cannot be calibrated under the original judge's frozen protocol hash; "
            "a new judge revision needs its own frozen protocol"
        )
    disjointness = assert_fresh_heldout_panel(
        fresh_manifest, original_manifest, require_full_plan=require_full_plan
    )
    report = build_calibration_report(
        fresh_manifest,
        human_annotations,
        annotator_ids,
        judge_annotations=distilled_judge_annotations,
        judge_reports=distilled_judge_reports,
        protocol=protocol,
        tolerances=tolerances,
    )
    gate_d = dict(report.get("gate_d") or {})
    passed = bool(gate_d.get("passed"))
    result = FreshHeldOutResult(
        calibration_id=calibration_id,
        status=str(gate_d.get("status") or "not_evaluable"),
        passed=passed,
        fresh_manifest_hash=str(disjointness["fresh_manifest_hash"]),
        original_manifest_hash=str(disjointness["original_manifest_hash"]),
        disjointness=disjointness,
        gate_d={
            "status": gate_d.get("status"),
            "passed": passed,
            "reason_codes": list(gate_d.get("reason_codes") or ()),
            "metrics": dict(gate_d.get("metrics") or {}),
        },
        reason_codes=tuple(str(code) for code in (gate_d.get("reason_codes") or ())),
        report_ref=report_ref,
    )
    return result, report


# --------------------------------------------------------------------------
# Paired frozen-video comparison
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenVideoScore:
    """One judge's label for one clip, with the frames it actually saw."""

    clip_id: str
    judge_revision: str
    video_sha256: str
    frame_hashes: Tuple[str, ...]
    binary_success: Optional[bool]
    progress: Optional[int] = None
    missing_reason: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.clip_id, "clip_id")
        _text(self.judge_revision, "judge_revision")
        _sha256(self.video_sha256, "video_sha256")
        if not self.frame_hashes:
            raise DistillationError("frame_hashes must record the frames the judge saw")
        for digest in self.frame_hashes:
            _sha256(digest, "frame hash")
        if self.binary_success is not None and not isinstance(self.binary_success, bool):
            raise DistillationError("binary_success must be True, False, or None")
        if self.progress is not None:
            if isinstance(self.progress, bool) or not isinstance(self.progress, int):
                raise DistillationError("progress must be an integer 0-5 or None")
            if not 0 <= self.progress <= 5:
                raise DistillationError("progress must be an integer 0-5 or None")
        if self.binary_success is None and not self.missing_reason:
            raise DistillationError("an unevaluable clip needs a missing_reason; silence is not a label")

    @property
    def frames_fingerprint(self) -> str:
        return canonical_json_sha256({"video": self.video_sha256, "frames": list(self.frame_hashes)})


@dataclass(frozen=True)
class PairedComparisonTolerances:
    """Thresholds frozen before the paired comparison is computed."""

    minimum_paired_clips: int
    maximum_absolute_binary_difference: float
    maximum_progress_disagreement_rate: float
    minimum_paired_coverage: float

    def __post_init__(self) -> None:
        _positive_int(self.minimum_paired_clips, "minimum_paired_clips")
        for name in (
            "maximum_absolute_binary_difference",
            "maximum_progress_disagreement_rate",
            "minimum_paired_coverage",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DistillationError("{0} must be a number".format(name))
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise DistillationError("{0} must be a probability from 0 through 1".format(name))

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "minimum_paired_clips": self.minimum_paired_clips,
            "maximum_absolute_binary_difference": float(self.maximum_absolute_binary_difference),
            "maximum_progress_disagreement_rate": float(self.maximum_progress_disagreement_rate),
            "minimum_paired_coverage": float(self.minimum_paired_coverage),
        }

    def fingerprint(self) -> str:
        return canonical_json_sha256(self.as_mapping())


@dataclass(frozen=True)
class PairedComparisonResult:
    """A paired comparison on identical clips with identical frames."""

    comparison_id: str
    status: str
    passed: bool
    paired_clips: int
    clip_set_hash: str
    tolerances_hash: str
    metrics: Mapping[str, Any]
    reason_codes: Tuple[str, ...] = ()

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "status": self.status,
            "passed": self.passed,
            "paired": True,
            "paired_clips": self.paired_clips,
            "clip_set_hash": self.clip_set_hash,
            "tolerances_hash": self.tolerances_hash,
            "metrics": dict(self.metrics),
            "reason_codes": list(self.reason_codes),
            "frames": "identical_frozen_frames_for_both_revisions",
            "interval_owner": "plumb.annotation.paired_difference_interval",
        }


def compare_frozen_videos(
    *,
    comparison_id: str,
    original: Sequence[FrozenVideoScore],
    distilled: Sequence[FrozenVideoScore],
    tolerances: PairedComparisonTolerances,
) -> PairedComparisonResult:
    """Compare two judge revisions on one frozen clip set, clip by clip.

    Refusals: a clip present for one revision and not the other, a clip whose
    video or frame digests differ between the revisions, and a comparison of a
    revision against itself.  Without those three the comparison is not paired
    and not frozen-video, whatever it is called.
    """

    _text(comparison_id, "comparison_id")
    if not isinstance(tolerances, PairedComparisonTolerances):
        raise DistillationError("tolerances must be a PairedComparisonTolerances")
    for name, rows in (("original", original), ("distilled", distilled)):
        if not rows:
            raise DistillationError("{0} scores must not be empty".format(name))
        for row in rows:
            if not isinstance(row, FrozenVideoScore):
                raise DistillationError("{0} scores must contain FrozenVideoScore rows".format(name))
    original_revisions = {row.judge_revision for row in original}
    distilled_revisions = {row.judge_revision for row in distilled}
    if len(original_revisions) != 1 or len(distilled_revisions) != 1:
        raise DistillationError("each side of the comparison must come from exactly one judge revision")
    if original_revisions == distilled_revisions:
        raise DistillationError("a paired comparison needs two different judge revisions")

    original_by_clip = {row.clip_id: row for row in original}
    distilled_by_clip = {row.clip_id: row for row in distilled}
    if len(original_by_clip) != len(original) or len(distilled_by_clip) != len(distilled):
        raise DistillationError("each revision may score each clip once in a paired comparison")
    if set(original_by_clip) != set(distilled_by_clip):
        only_original = sorted(set(original_by_clip) - set(distilled_by_clip))
        only_distilled = sorted(set(distilled_by_clip) - set(original_by_clip))
        raise DistillationError(
            "a paired comparison needs the identical clip set; original-only: {0}; distilled-only: {1}".format(
                ", ".join(only_original) or "none", ", ".join(only_distilled) or "none"
            )
        )
    mismatched = sorted(
        clip_id
        for clip_id in original_by_clip
        if original_by_clip[clip_id].frames_fingerprint != distilled_by_clip[clip_id].frames_fingerprint
    )
    if mismatched:
        raise DistillationError(
            "both revisions must score the identical frames; differing clips: {0}".format(", ".join(mismatched))
        )

    clip_ids = sorted(original_by_clip)
    both_decisive = 0
    positive_only_distilled = 0
    positive_only_original = 0
    binary_agreements = 0
    unevaluable_pairs = 0
    progress_comparable = 0
    progress_disagreements = 0
    for clip_id in clip_ids:
        first = original_by_clip[clip_id]
        second = distilled_by_clip[clip_id]
        if first.binary_success is None or second.binary_success is None:
            unevaluable_pairs += 1
        else:
            both_decisive += 1
            if first.binary_success == second.binary_success:
                binary_agreements += 1
            elif second.binary_success:
                positive_only_distilled += 1
            else:
                positive_only_original += 1
        if first.progress is not None and second.progress is not None:
            progress_comparable += 1
            if first.progress != second.progress:
                progress_disagreements += 1

    coverage = both_decisive / len(clip_ids)
    difference = paired_difference_interval(positive_only_distilled, positive_only_original, both_decisive)
    progress_disagreement_rate = (
        progress_disagreements / progress_comparable if progress_comparable else None
    )
    metrics: Dict[str, Any] = {
        "clips": len(clip_ids),
        "both_decisive_pairs": both_decisive,
        "unevaluable_pairs": unevaluable_pairs,
        "paired_coverage": coverage,
        "binary_agreements": binary_agreements,
        "binary_agreement_rate": binary_agreements / both_decisive if both_decisive else None,
        "positive_only_distilled": positive_only_distilled,
        "positive_only_original": positive_only_original,
        "paired_binary_difference": difference,
        "progress_comparable_pairs": progress_comparable,
        "progress_disagreements": progress_disagreements,
        "progress_disagreement_rate": progress_disagreement_rate,
        "original_revision": sorted(original_revisions)[0],
        "distilled_revision": sorted(distilled_revisions)[0],
        "tolerances": tolerances.as_mapping(),
    }

    reasons: List[str] = []
    if both_decisive < tolerances.minimum_paired_clips:
        reasons.append("paired_clips_below_minimum")
    if coverage < float(tolerances.minimum_paired_coverage):
        reasons.append("paired_coverage_below_minimum")
    if difference is None:
        reasons.append("paired_difference_undefined")
    elif abs(float(difference["point"])) > float(tolerances.maximum_absolute_binary_difference):
        reasons.append("paired_binary_difference_above_maximum")
    if progress_disagreement_rate is None:
        reasons.append("progress_disagreement_rate_undefined")
    elif progress_disagreement_rate > float(tolerances.maximum_progress_disagreement_rate):
        reasons.append("progress_disagreement_rate_above_maximum")

    return PairedComparisonResult(
        comparison_id=comparison_id,
        status="pass" if not reasons else "fail",
        passed=not reasons,
        paired_clips=both_decisive,
        clip_set_hash=canonical_json_sha256(
            {"clips": [{"clip_id": cid, "frames": original_by_clip[cid].frames_fingerprint} for cid in clip_ids]}
        ),
        tolerances_hash=tolerances.fingerprint(),
        metrics=metrics,
        reason_codes=tuple(reasons),
    )


# --------------------------------------------------------------------------
# Gate E evidence
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DistillationGate:
    """Produce the ``distilled_judge`` mapping Gate E reads, and nothing more.

    ``plumb.gates`` Gate E enforces exactly one rule about distillation: if
    ``distilled_judge.used_for_scoring`` is truthy then
    ``distilled_judge.separately_validated`` must be truthy too.  This class is
    the only sanctioned producer of that mapping.  ``separately_validated`` is
    false until a fresh held-out calibration *and* a paired frozen-video
    comparison both exist and both pass, and ``used_for_scoring`` is never
    emitted as true while any blocker stands -- so the mapping cannot be made to
    lie by setting ``requested_for_scoring``.
    """

    revision: DistilledJudgeRevision
    search: LoRASearchSpace
    arm_results: Tuple[ArmResult, ...]
    selection: SearchSelection
    dataset: DistillationDataset
    fresh_calibration: Optional[FreshHeldOutResult] = None
    paired_comparison: Optional[PairedComparisonResult] = None
    requested_for_scoring: bool = False
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.revision, DistilledJudgeRevision):
            raise DistillationError("revision must be a DistilledJudgeRevision")
        if not isinstance(self.search, LoRASearchSpace):
            raise DistillationError("search must be a LoRASearchSpace")
        if not isinstance(self.selection, SearchSelection):
            raise DistillationError("selection must be a SearchSelection")
        if not isinstance(self.dataset, DistillationDataset):
            raise DistillationError("dataset must be a DistillationDataset")
        for result in self.arm_results:
            if not isinstance(result, ArmResult):
                raise DistillationError("arm_results must contain ArmResult objects")
        if self.fresh_calibration is not None and not isinstance(self.fresh_calibration, FreshHeldOutResult):
            raise DistillationError("fresh_calibration must be a FreshHeldOutResult")
        if self.paired_comparison is not None and not isinstance(
            self.paired_comparison, PairedComparisonResult
        ):
            raise DistillationError("paired_comparison must be a PairedComparisonResult")
        _string_tuple(self.notes, "notes")
        if self.selection.preregistration_hash != self.search.preregistration_hash():
            raise DistillationError(
                "the selection was made against a different preregistered search than the one recorded"
            )
        if self.revision.search_preregistration_hash != self.search.preregistration_hash():
            raise DistillationError(
                "the distilled revision does not belong to the recorded preregistered search"
            )
        if self.revision.training_dataset_hash != self.dataset.content_hash():
            raise DistillationError("the distilled revision was not trained on the recorded dataset")
        recorded = {result.arm_id for result in self.arm_results}
        missing = sorted({arm.arm_id for arm in self.search.arms} - recorded)
        if missing:
            raise DistillationError(
                "every preregistered arm must be recorded, including losers: missing {0}".format(
                    ", ".join(missing)
                )
            )

    # -- decisions -------------------------------------------------------

    @property
    def separately_validated(self) -> bool:
        """Both independent validations exist and both passed."""

        return bool(
            self.fresh_calibration is not None
            and self.fresh_calibration.passed
            and self.paired_comparison is not None
            and self.paired_comparison.passed
        )

    def blockers(self) -> Tuple[str, ...]:
        """Every reason this revision may not score a burst."""

        reasons: List[str] = []
        if self.fresh_calibration is None:
            reasons.append("no fresh held-out calibration has been run for this revision")
        elif not self.fresh_calibration.passed:
            reasons.append(
                "fresh held-out calibration did not pass: "
                + (", ".join(self.fresh_calibration.reason_codes) or self.fresh_calibration.status)
            )
        if self.paired_comparison is None:
            reasons.append("no paired frozen-video comparison against the original judge exists")
        elif not self.paired_comparison.passed:
            reasons.append(
                "paired frozen-video comparison did not pass: "
                + (", ".join(self.paired_comparison.reason_codes) or self.paired_comparison.status)
            )
        if not self.selection.reportable:
            reasons.append(
                "no development-validation winner is reportable: "
                + (", ".join(self.selection.reason_codes) or self.selection.status)
            )
        elif self.selection.winner_arm_id != self.revision.search_arm_id:
            reasons.append(
                "this revision came from arm {0}, but the reportable winner is {1}".format(
                    self.revision.search_arm_id, self.selection.winner_arm_id
                )
            )
        if not self.revision.student.can_substitute_for_primary_judge:
            reasons.append(
                "student output contract {0!r} is not the frozen rubric schema; this revision is an "
                "exploratory analysis and cannot score a burst".format(self.revision.student.output_contract)
            )
        if self.dataset.lineage_leakage_report()["status"] != "pass":
            reasons.append("training lineages are not provably disjoint from held-out and primary cohorts")
        return tuple(reasons)

    def authorisation(self) -> Dict[str, Any]:
        """Whether this revision may score, as a status/reason pair."""

        blockers = self.blockers()
        return {
            "authorised": bool(not blockers and self.requested_for_scoring),
            "requested_for_scoring": bool(self.requested_for_scoring),
            "separately_validated": self.separately_validated,
            "blockers": list(blockers),
        }

    # -- evidence --------------------------------------------------------

    def evidence(self) -> Dict[str, Any]:
        """The exact ``measurements['distilled_judge']`` mapping for Gate E."""

        blockers = self.blockers()
        used_for_scoring = bool(self.requested_for_scoring and not blockers)
        refused = bool(self.requested_for_scoring and blockers)
        return {
            "schema_version": SCHEMA_VERSION,
            # The two keys plumb/gates.py Gate E actually reads.
            "used_for_scoring": used_for_scoring,
            "separately_validated": self.separately_validated,
            # Everything a reviewer needs to check that decision.
            "scoring_request_refused": refused,
            "blockers": list(blockers),
            "revision": self.revision.as_mapping(),
            "original_judge_results_preserved": True,
            "original_judge_results_ref": self.revision.base_judge.results_ref,
            "original_judge_results_sha256": self.revision.base_judge.results_sha256,
            "distilled_judge_results_ref": self.revision.results_ref,
            "training_data": self.dataset.as_mapping(),
            "lora_search": {
                **self.search.as_mapping(),
                "results": [result.as_mapping() for result in self.arm_results],
                "selection": self.selection.as_mapping(),
                "spoken_claim": spoken_claim(self.search, self.selection),
            },
            "fresh_heldout_calibration": (
                None if self.fresh_calibration is None else self.fresh_calibration.as_mapping()
            ),
            "paired_frozen_video_comparison": (
                None if self.paired_comparison is None else self.paired_comparison.as_mapping()
            ),
            "training_budget_item": self.dataset.budget.as_mapping(),
            "notes": list(self.notes),
        }

    def gate_e_measurements_patch(self) -> Dict[str, Any]:
        """The fragment to merge into an existing Gate E ``measurements``."""

        return {"distilled_judge": self.evidence()}


def assert_scoring_authorised(gate: DistillationGate) -> None:
    """Raise unless this distilled revision may actually score a burst.

    Call this immediately before a burst that names a distilled judge.  Gate E
    catches a dishonest ledger entry after the fact; this stops the run.
    """

    if not isinstance(gate, DistillationGate):
        raise DistillationError("gate must be a DistillationGate")
    blockers = gate.blockers()
    if blockers:
        raise DistillationError(
            "distilled judge {0} may not score a burst: {1}".format(
                gate.revision.revision_id, "; ".join(blockers)
            )
        )


def unvalidated_distilled_judge_evidence() -> Dict[str, Any]:
    """The honest Gate E entry when no distilled judge exists yet.

    Gate E's rule is conditional: a distilled judge that is not used for
    scoring passes trivially.  Recording this mapping is how a ledger says "the
    primary judge scored this run" without leaving the field absent.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "used_for_scoring": False,
        "separately_validated": False,
        "status": "not_run",
        "reason": "no distilled judge revision exists; the original judge scored this run",
        "original_judge_results_preserved": True,
    }


__all__ = [
    "ArmResult",
    "BUDGET_ITEM",
    "BaseJudgeRevision",
    "ClipLineage",
    "DEV_VALIDATION_COHORT",
    "DISJOINT_COHORTS",
    "DistillationDataset",
    "DistillationError",
    "DistillationGate",
    "DistilledJudgeRevision",
    "EarlyStoppingRule",
    "FRESH_HELDOUT_COHORT",
    "FreshHeldOutResult",
    "FrozenVideoScore",
    "GATE_D_HELDOUT_COHORT",
    "LABEL_SOURCES",
    "LineageExclusions",
    "LoRAConfig",
    "LoRASearchSpace",
    "PUBLISHED_LORA_ARM",
    "PUBLISHED_LORA_REFERENCE",
    "PairedComparisonResult",
    "PairedComparisonTolerances",
    "REPO_DEFAULT_LORA_ARM",
    "RUBRIC_JSON_CONTRACT",
    "SCALAR_HEAD_CONTRACT",
    "SCHEMA_VERSION",
    "SELECTION_SPLIT",
    "SearchSelection",
    "StudentModel",
    "TRAIN_COHORT",
    "TRAINING_JOBS_REFERENCE",
    "TrainingBudgetItem",
    "TrainingJobSpec",
    "TrainingRuntime",
    "assert_fresh_heldout_panel",
    "assert_scoring_authorised",
    "compare_frozen_videos",
    "dataset_from_calibration_manifest",
    "default_search_space",
    "frozen_sampling_for_distilled_judge",
    "render_search_payloads",
    "run_fresh_heldout_calibration",
    "select_arm",
    "spoken_claim",
    "unvalidated_distilled_judge_evidence",
]
