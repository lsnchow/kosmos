"""Gate C scenario panels: real-provenance starting states and their limits.

This module produces the ``ScenarioManifest`` that ``plumb.gates`` validates.
It is a *bookkeeping and refusal* layer, not an acquisition layer: it cannot
invent a start, a scene fact, a camera calibration, or a comparability claim.
Every field is supplied by ``cluster/build_scenarios.py`` from a real source
dataset, and everything still unresolved is carried as an explicit open
dependency rather than filled with a plausible value.

Three refusals are enforced in code because they are the ways this panel could
silently become fiction:

1. A start whose source scene family does not match its task is rejected.  In
   particular a drawer-sourced start can never populate the basket, sink, or
   cloth cells; that absence is an acquisition dependency, not a substitution.
2. A ``start_lineage_id`` may appear in exactly one cohort.  Cohort separation
   is checked with ``plumb.measurement.validate_source_lineage_leakage`` rather
   than a second private implementation.
3. Neighbouring frames from one source episode, a repeated source image/state,
   and a re-seeded replay of one physical start are not independent new starts.

Scene details from the original handoff (drawer and cloth taped to the table, a
thin plastic wrap over the sink) are carried as ``scene_claims`` with
``verified=False`` until a source scene record is named.  The published
``Open the drawer`` criterion is a 1.5 cm metric threshold; its scene-specific
calibrated reference is recorded as required and unresolved instead of being
estimated from pixels.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .gates import EXACT_TASK_PROMPTS, ScenarioManifestValidator, canonical_json_sha256
from .measurement import validate_source_lineage_leakage
from .policies.tasks import BENCHMARK_TASK_REGISTRY


SCHEMA_VERSION = 1

#: Task IDs in the frozen order used by the rest of the project.
TASKS: Tuple[str, ...] = ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")

PRIMARY_COHORT = "primary"
DEVELOPMENT_COHORT = "development"
CALIBRATION_COHORT = "calibration"
COST_CONFIRMATION_COHORT = "cost_confirmation"
REVERSE_VALIDATION_COHORT = "reverse_validation"

#: Cohorts recognised by this manifest.  ``primary`` is the 50-start panel
#: shared across all six policies; the others must stay lineage-disjoint from it.
COHORTS: Tuple[str, ...] = (
    PRIMARY_COHORT,
    DEVELOPMENT_COHORT,
    CALIBRATION_COHORT,
    COST_CONFIRMATION_COHORT,
    REVERSE_VALIDATION_COHORT,
)
#: Cohorts the specification requires to exist before Gate C can be considered.
REQUIRED_COHORTS: Tuple[str, ...] = (
    PRIMARY_COHORT,
    DEVELOPMENT_COHORT,
    CALIBRATION_COHORT,
    COST_CONFIRMATION_COHORT,
)

PRIMARY_STARTS_PER_TASK = 50

DRAWER_SCENE = "drawer"
BASKET_SCENE = "basket"
SINK_SCENE = "sink"
CLOTH_SCENE = "cloth"
SCENE_FAMILIES: Tuple[str, ...] = (DRAWER_SCENE, BASKET_SCENE, SINK_SCENE, CLOTH_SCENE)

#: The scene family a task's starts must come from.  This mapping is the code
#: form of "do not fill these cells with drawer starts".
TASK_SCENE_FAMILY: Mapping[str, str] = {
    "open_drawer": DRAWER_SCENE,
    "close_drawer": DRAWER_SCENE,
    "to_basket": BASKET_SCENE,
    "to_sink": SINK_SCENE,
    "fold_cloth": CLOTH_SCENE,
}

MATCHED_PROVENANCE = "matched_provenance"
MATCHED_DISTRIBUTION = "matched_distribution_with_limitations"
COMPARABILITY_VALUES: Tuple[str, ...] = (MATCHED_PROVENANCE, MATCHED_DISTRIBUTION)

AUTO_EVAL_DATASET = "zhouzypaul/auto_eval"
BRIDGE_LEROBOT_DATASET = "IPEC-COMMUNITY/bridge_orig_lerobot"

#: Scene families a known source dataset may legitimately supply.  The public
#: ``auto_eval`` index exposes the drawer scene only; it does not establish
#: matched sink/cloth panels.
DATASET_SCENE_FAMILIES: Mapping[str, Tuple[str, ...]] = {
    AUTO_EVAL_DATASET: (DRAWER_SCENE,),
    BRIDGE_LEROBOT_DATASET: (BASKET_SCENE, SINK_SCENE, CLOTH_SCENE, DRAWER_SCENE),
}

#: Declared Bridge state convention from the build specification.  Recorded per
#: start so a later convention change is visible instead of silent.
BRIDGE_STATE_CONVENTION = "(x, y, z, rx, ry, rz, 0, gripper)"

#: Minimum separation, in source frames, between two starts drawn from one
#: episode.  This is a declared protocol parameter frozen before acquisition,
#: not an empirical independence measurement.
DEFAULT_MIN_SOURCE_FRAME_GAP = 10

SYNTHETIC_PROVENANCE_VALUES = frozenset({"synthetic", "simulated", "mock", "fake"})

_CALIBRATION_STATUS_VALUES = ("not_required", "unresolved", "resolved")
_AXIS_STATUS_VALUES = ("matched", "mismatched", "unresolved")

#: The five Gate C comparison axes named by the specification.
GATE_C_COMPARISON_AXES: Tuple[str, ...] = (
    "scene_layout",
    "objects",
    "reset_randomization_distribution",
    "policy_checkpoint_wrapper",
    "control_timing_and_task_horizon",
)

#: Applies to every task: AutoEval publishes aggregate per-cell tables, so no
#: start in this manifest is a paired reconstruction of an original trial.
UNPAIRED_REFERENCE_LIMITATION = (
    "AutoEval publishes aggregate per-cell tables only; the original trial-level starting states "
    "are not released, so no start here reconstructs a specific original trial."
)

SINK_CLOTH_ACQUISITION_DEPENDENCY = (
    "autoeval_sink_cloth_basket_scene_assets: no AutoEval scene asset or scene record for this task "
    "has been acquired; the cell stays empty rather than being filled with drawer starts."
)

CALIBRATION_REFERENCE_DEPENDENCY = (
    "scene_metric_calibration_reference: the published 1.5 cm open-drawer threshold needs a "
    "scene-specific calibrated reference; it is unresolved and is not estimated from pixels."
)

SCENE_CLAIM_VERIFICATION_DEPENDENCY = (
    "source_scene_record_verification: taped drawer/cloth and sink plastic wrap are handoff claims "
    "and stay verified=false until a named source scene record is checked."
)


class ScenarioError(ValueError):
    """Raised when a scenario panel would be invalid, unpaired, or fabricated."""


def utc_now() -> str:
    """Return an unambiguous UTC timestamp for a manifest record."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def verbatim_prompt(task: str) -> str:
    """Return the verbatim task prompt from the single source of truth."""

    if task not in EXACT_TASK_PROMPTS:
        raise ScenarioError("unknown benchmark task %r" % (task,))
    return EXACT_TASK_PROMPTS[task]


def verbatim_prompt_consistency() -> Dict[str, Any]:
    """Compare ``EXACT_TASK_PROMPTS`` with the frozen judge task registry.

    Two in-repo constants encode the five prompts.  Reporting their agreement
    here makes drift a visible failure instead of a silent paraphrase.
    """

    mismatches: List[Dict[str, Any]] = []
    for task in sorted(EXACT_TASK_PROMPTS):
        expected = EXACT_TASK_PROMPTS[task]
        try:
            registered = BENCHMARK_TASK_REGISTRY.get(task).instruction
        except KeyError:
            mismatches.append({"task": task, "reason": "absent_from_task_registry", "expected": expected})
            continue
        if registered != expected:
            mismatches.append(
                {"task": task, "reason": "instruction_differs", "expected": expected, "registry": registered}
            )
    lowercase_fold = EXACT_TASK_PROMPTS.get("fold_cloth") == "fold the cloth from top right to bottom left"
    if not lowercase_fold:
        mismatches.append(
            {
                "task": "fold_cloth",
                "reason": "task_five_prompt_is_not_the_verbatim_lowercase_fold_string",
                "expected": "fold the cloth from top right to bottom left",
                "actual": EXACT_TASK_PROMPTS.get("fold_cloth"),
            }
        )
    missing_tasks = [task for task in TASKS if task not in EXACT_TASK_PROMPTS]
    for task in missing_tasks:
        mismatches.append({"task": task, "reason": "absent_from_exact_task_prompts"})
    return {
        "status": "fail" if mismatches else "pass",
        "lowercase_fold_preserved": lowercase_fold,
        "mismatches": mismatches,
        "source_of_truth": "plumb.gates.EXACT_TASK_PROMPTS",
    }


def assert_verbatim_prompts() -> None:
    """Refuse to build a manifest when the five prompts are not verbatim."""

    result = verbatim_prompt_consistency()
    if result["status"] != "pass":
        raise ScenarioError(
            "verbatim task prompts disagree with plumb.gates.EXACT_TASK_PROMPTS: %s"
            % json.dumps(result["mismatches"], sort_keys=True)
        )


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_string(value: Any, field_name: str) -> str:
    if not _nonempty_string(value):
        raise ScenarioError("%s must be a non-empty string" % field_name)
    return value


def _is_hash(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    digest = value[7:] if value.startswith("sha256:") else value
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest.lower())


def _require_hash(value: Any, field_name: str) -> str:
    if not _is_hash(value):
        raise ScenarioError("%s must be a SHA-256 digest" % field_name)
    return value


def _require_index(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScenarioError("%s must be a non-negative integer" % field_name)
    return value


def _require_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ScenarioError("%s must be a non-empty object" % field_name)
    return {str(key): item for key, item in value.items()}


def _is_immutable_revision(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _brief(items: Sequence[Any], limit: int = 8) -> str:
    """Render a bounded list for an error message; long lists stay readable."""

    values = [str(item) for item in items]
    if len(values) <= limit:
        return ", ".join(values)
    return "%s, ... and %d more" % (", ".join(values[:limit]), len(values) - limit)


def _json_ready(value: Any) -> Any:
    """Convert a nested structure to JSON-safe values without inventing any."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class SceneClaim:
    """A scene detail that must be verified against a source record first.

    ``verified`` defaults to ``False`` and cannot be set true without naming the
    source record and the verification method, so a handoff sentence can never
    become a fact by assertion.
    """

    claim_id: str
    statement: str
    scene_family: str
    verified: bool = False
    source_record_uri: Optional[str] = None
    verification_method: Optional[str] = None

    def __post_init__(self) -> None:
        _require_string(self.claim_id, "scene claim claim_id")
        _require_string(self.statement, "scene claim statement")
        if self.scene_family not in SCENE_FAMILIES:
            raise ScenarioError("scene claim scene_family must be one of %s" % ", ".join(SCENE_FAMILIES))
        if not isinstance(self.verified, bool):
            raise ScenarioError("scene claim verified must be a boolean")
        if self.verified and not (
            _nonempty_string(self.source_record_uri) and _nonempty_string(self.verification_method)
        ):
            raise ScenarioError(
                "a verified scene claim requires source_record_uri and verification_method; "
                "an unsupported claim stays verified=false"
            )

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "scene_family": self.scene_family,
            "verified": self.verified,
            "source_record_uri": self.source_record_uri,
            "verification_method": self.verification_method,
        }


def default_scene_claims(scene_family: str) -> Tuple[SceneClaim, ...]:
    """Return the unverified handoff scene claims for one scene family."""

    if scene_family == DRAWER_SCENE:
        return (
            SceneClaim(
                claim_id="drawer_taped_to_table",
                statement="The drawer unit is taped to the table.",
                scene_family=DRAWER_SCENE,
            ),
        )
    if scene_family == CLOTH_SCENE:
        return (
            SceneClaim(
                claim_id="cloth_taped_to_table",
                statement="The cloth is taped to the table.",
                scene_family=CLOTH_SCENE,
            ),
        )
    if scene_family == SINK_SCENE:
        return (
            SceneClaim(
                claim_id="sink_thin_plastic_wrap",
                statement="A thin plastic wrap covers the sink.",
                scene_family=SINK_SCENE,
            ),
        )
    # The handoff records no basket-specific fixture detail.  Inventing one
    # would be exactly the failure this structure exists to prevent.
    return ()


@dataclass(frozen=True)
class SuccessCriterion:
    """The published success rule plus whether it is visible or needs metrology."""

    task: str
    criterion: str
    max_steps: int
    measurement_kind: str
    calibration_reference_required: bool
    calibration_reference_status: str = "unresolved"
    calibration_reference_uri: Optional[str] = None
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task not in TASK_SCENE_FAMILY:
            raise ScenarioError("success criterion task %r is not a benchmark task" % (self.task,))
        _require_string(self.criterion, "success criterion text")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ScenarioError("success criterion max_steps must be a positive integer")
        _require_string(self.measurement_kind, "success criterion measurement_kind")
        if self.calibration_reference_status not in _CALIBRATION_STATUS_VALUES:
            raise ScenarioError(
                "calibration_reference_status must be one of %s" % ", ".join(_CALIBRATION_STATUS_VALUES)
            )
        if not self.calibration_reference_required:
            if self.calibration_reference_status != "not_required":
                raise ScenarioError(
                    "a criterion that needs no calibrated reference must record status 'not_required'"
                )
        elif self.calibration_reference_status == "not_required":
            raise ScenarioError("a required calibration reference cannot be status 'not_required'")
        if self.calibration_reference_status == "resolved" and not _nonempty_string(self.calibration_reference_uri):
            raise ScenarioError(
                "a resolved calibration reference must name calibration_reference_uri; "
                "an unsupported metric estimate is not a resolution"
            )

    @property
    def resolved(self) -> bool:
        return not self.calibration_reference_required or self.calibration_reference_status == "resolved"

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "criterion": self.criterion,
            "max_steps": self.max_steps,
            "measurement_kind": self.measurement_kind,
            "calibration_reference_required": self.calibration_reference_required,
            "calibration_reference_status": self.calibration_reference_status,
            "calibration_reference_uri": self.calibration_reference_uri,
            "notes": list(self.notes),
        }


#: Verbatim §3 success rules and horizons, with the metrology requirement made
#: explicit.  ``open_drawer`` is the one metric threshold; it stays unresolved.
DEFAULT_SUCCESS_CRITERIA: Mapping[str, SuccessCriterion] = {
    "close_drawer": SuccessCriterion(
        task="close_drawer",
        criterion="completely closed",
        max_steps=70,
        measurement_kind="visible_final_state",
        calibration_reference_required=False,
        calibration_reference_status="not_required",
    ),
    "open_drawer": SuccessCriterion(
        task="open_drawer",
        criterion="opened >= 1.5 cm",
        max_steps=70,
        measurement_kind="metric_threshold",
        calibration_reference_required=True,
        calibration_reference_status="unresolved",
        notes=(
            "A 1.5 cm threshold needs a scene-specific calibrated reference (known-size fiducial or "
            "measured drawer travel) before it is measurable; it is not estimated from pixel counts.",
        ),
    ),
    "to_basket": SuccessCriterion(
        task="to_basket",
        criterion="fully inside at episode end",
        max_steps=100,
        measurement_kind="visible_final_state",
        calibration_reference_required=False,
        calibration_reference_status="not_required",
    ),
    "to_sink": SuccessCriterion(
        task="to_sink",
        criterion="fully inside at episode end",
        max_steps=100,
        measurement_kind="visible_final_state",
        calibration_reference_required=False,
        calibration_reference_status="not_required",
    ),
    "fold_cloth": SuccessCriterion(
        task="fold_cloth",
        criterion="folded >= 1/4 diagonally",
        max_steps=80,
        measurement_kind="relative_geometric",
        calibration_reference_required=True,
        calibration_reference_status="unresolved",
        notes=(
            "The quarter-diagonal rule is relative to the cloth's own diagonal, so it needs a scene "
            "record fixing the cloth extent and the designated top-right corner before it is measurable.",
        ),
    ),
}


@dataclass(frozen=True)
class ComparisonAxis:
    """One Gate C comparability axis against AutoEval; ``unresolved`` by default."""

    axis: str
    status: str = "unresolved"
    evidence_uri: Optional[str] = None
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if self.axis not in GATE_C_COMPARISON_AXES:
            raise ScenarioError("unknown Gate C comparison axis %r" % (self.axis,))
        if self.status not in _AXIS_STATUS_VALUES:
            raise ScenarioError("comparison axis status must be one of %s" % ", ".join(_AXIS_STATUS_VALUES))
        if self.status == "matched" and not _nonempty_string(self.evidence_uri):
            raise ScenarioError("a matched comparison axis must name its evidence_uri")

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "axis": self.axis,
            "status": self.status,
            "evidence_uri": self.evidence_uri,
            "detail": self.detail,
        }


def default_comparison_axes() -> Tuple[ComparisonAxis, ...]:
    """Return all five Gate C axes unresolved; comparability is not assumed."""

    return tuple(ComparisonAxis(axis=axis) for axis in GATE_C_COMPARISON_AXES)


@dataclass(frozen=True)
class SourceDataset:
    """A real source dataset, its pinned revision, and its permitted scenes."""

    dataset_id: str
    revision: str
    role: str
    scene_families: Tuple[str, ...]
    selection_rule: str
    license: Optional[str] = None
    url: Optional[str] = None
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.dataset_id, "source dataset_id")
        _require_string(self.revision, "source revision")
        _require_string(self.role, "source role")
        _require_string(self.selection_rule, "source selection_rule")
        if not self.scene_families:
            raise ScenarioError("a source dataset must declare at least one scene family")
        unknown = [value for value in self.scene_families if value not in SCENE_FAMILIES]
        if unknown:
            raise ScenarioError("unknown source scene families: %s" % ", ".join(sorted(unknown)))
        allowed = DATASET_SCENE_FAMILIES.get(self.dataset_id)
        if allowed is not None:
            outside = [value for value in self.scene_families if value not in allowed]
            if outside:
                raise ScenarioError(
                    "dataset %s does not establish a %s scene panel"
                    % (self.dataset_id, ", ".join(sorted(outside)))
                )

    @property
    def revision_immutable(self) -> bool:
        return _is_immutable_revision(self.revision)

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "revision_immutable": self.revision_immutable,
            "role": self.role,
            "scene_families": list(self.scene_families),
            "selection_rule": self.selection_rule,
            "license": self.license,
            "url": self.url,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class StartRecord:
    """One real starting state.

    Every field ``plumb.gates.ScenarioManifestValidator.REQUIRED_START_FIELDS``
    demands is present and non-empty by construction.  ``frame_id`` is the
    source-native frame identifier (a string, so frame zero is never falsey) and
    ``frame_index`` is the integer position the independence guard compares.
    """

    start_id: str
    task: str
    cohort: str
    start_lineage_id: str
    source_dataset: str
    source_revision: str
    episode_id: str
    frame_id: str
    frame_index: int
    image_hash: str
    state_hash: str
    image_timestamp: str
    state_timestamp: str
    state_convention: str
    camera: Mapping[str, Any]
    scene: Mapping[str, Any]
    initial_state_stratum: str
    instruction: str
    goal_reference_hashes: Mapping[str, str]
    goal_reference_provenance: Mapping[str, Any]
    source_scene_family: str
    world_seed: Optional[int] = None
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task not in EXACT_TASK_PROMPTS:
            raise ScenarioError("start task %r is not one of the five benchmark tasks" % (self.task,))
        if self.cohort not in COHORTS:
            raise ScenarioError("start cohort must be one of %s" % ", ".join(COHORTS))
        for field_name in (
            "start_id",
            "start_lineage_id",
            "source_dataset",
            "source_revision",
            "episode_id",
            "frame_id",
            "image_timestamp",
            "state_timestamp",
            "state_convention",
            "initial_state_stratum",
        ):
            _require_string(getattr(self, field_name), "start %s" % field_name)
        _require_index(self.frame_index, "start frame_index")
        _require_hash(self.image_hash, "start image_hash")
        _require_hash(self.state_hash, "start state_hash")
        if self.instruction != EXACT_TASK_PROMPTS[self.task]:
            raise ScenarioError(
                "start %s instruction is not the verbatim prompt for %s" % (self.start_id, self.task)
            )
        if self.source_scene_family not in SCENE_FAMILIES:
            raise ScenarioError("start source_scene_family must be one of %s" % ", ".join(SCENE_FAMILIES))
        required_family = TASK_SCENE_FAMILY[self.task]
        if self.source_scene_family != required_family:
            raise ScenarioError(
                "start %s is a %s-scene start and cannot populate task %s, which needs %s-scene "
                "starts. %s"
                % (
                    self.start_id,
                    self.source_scene_family,
                    self.task,
                    required_family,
                    SINK_CLOTH_ACQUISITION_DEPENDENCY,
                )
            )
        allowed = DATASET_SCENE_FAMILIES.get(self.source_dataset)
        if allowed is not None and self.source_scene_family not in allowed:
            raise ScenarioError(
                "dataset %s does not expose a %s scene; it cannot supply this start"
                % (self.source_dataset, self.source_scene_family)
            )
        camera = _require_mapping(self.camera, "start camera")
        _require_string(camera.get("camera_id"), "start camera.camera_id")
        if not isinstance(camera.get("crop"), Mapping):
            raise ScenarioError("start camera.crop must be an object (record the exact crop, even if identity)")
        calibration = camera.get("calibration")
        if not isinstance(calibration, Mapping) or calibration.get("status") not in ("unresolved", "resolved"):
            raise ScenarioError(
                "start camera.calibration must record status 'unresolved' or 'resolved'; "
                "an absent calibration is unresolved, not zero"
            )
        if calibration.get("status") == "resolved" and not _nonempty_string(calibration.get("reference_uri")):
            raise ScenarioError("a resolved camera calibration must name its reference_uri")
        scene = _require_mapping(self.scene, "start scene")
        if scene.get("scene_family") != self.source_scene_family:
            raise ScenarioError("start scene.scene_family must equal source_scene_family")
        objects = scene.get("objects")
        if not isinstance(objects, (list, tuple)) or not objects:
            raise ScenarioError("start scene.objects must be a non-empty list of observed objects")
        hashes = _require_mapping(self.goal_reference_hashes, "start goal_reference_hashes")
        for name, digest in hashes.items():
            _require_hash(digest, "start goal_reference_hashes.%s" % name)
        provenance = _require_mapping(self.goal_reference_provenance, "start goal_reference_provenance")
        _require_string(provenance.get("source_uri"), "start goal_reference_provenance.source_uri")
        if self.world_seed is not None and (isinstance(self.world_seed, bool) or not isinstance(self.world_seed, int)):
            raise ScenarioError("start world_seed must be an integer or null")
        if not isinstance(self.notes, tuple):
            raise ScenarioError("start notes must be a tuple of strings")

    @property
    def physical_start_key(self) -> Tuple[str, str, str, int]:
        """Identity of the physical start: dataset, revision, episode, frame."""

        return (self.source_dataset, self.source_revision, self.episode_id, self.frame_index)

    @property
    def episode_key(self) -> Tuple[str, str, str]:
        return (self.source_dataset, self.source_revision, self.episode_id)

    def as_mapping(self) -> Dict[str, Any]:
        """Serialize exactly the fields the scenario validator and audit need."""

        return {
            "start_id": self.start_id,
            "task": self.task,
            "cohort": self.cohort,
            "start_lineage_id": self.start_lineage_id,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "image_hash": self.image_hash,
            "state_hash": self.state_hash,
            "image_timestamp": self.image_timestamp,
            "state_timestamp": self.state_timestamp,
            "state_convention": self.state_convention,
            "camera": _json_ready(self.camera),
            "scene": _json_ready(self.scene),
            "initial_state_stratum": self.initial_state_stratum,
            "instruction": self.instruction,
            "goal_reference_hashes": dict(self.goal_reference_hashes),
            "goal_reference_provenance": _json_ready(self.goal_reference_provenance),
            "source_scene_family": self.source_scene_family,
            "world_seed": self.world_seed,
            "notes": list(self.notes),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StartRecord":
        if not isinstance(payload, Mapping):
            raise ScenarioError("a start record must be an object")
        missing = [
            name
            for name in (
                "start_id",
                "task",
                "cohort",
                "start_lineage_id",
                "source_dataset",
                "source_revision",
                "episode_id",
                "frame_id",
                "frame_index",
                "image_hash",
                "state_hash",
                "image_timestamp",
                "state_timestamp",
                "state_convention",
                "camera",
                "scene",
                "initial_state_stratum",
                "instruction",
                "goal_reference_hashes",
                "goal_reference_provenance",
                "source_scene_family",
            )
            if name not in payload
        ]
        if missing:
            raise ScenarioError("start record is missing %s" % ", ".join(missing))
        notes = payload.get("notes", ())
        return cls(
            start_id=payload["start_id"],
            task=payload["task"],
            cohort=payload["cohort"],
            start_lineage_id=payload["start_lineage_id"],
            source_dataset=payload["source_dataset"],
            source_revision=payload["source_revision"],
            episode_id=payload["episode_id"],
            frame_id=payload["frame_id"],
            frame_index=payload["frame_index"],
            image_hash=payload["image_hash"],
            state_hash=payload["state_hash"],
            image_timestamp=payload["image_timestamp"],
            state_timestamp=payload["state_timestamp"],
            state_convention=payload["state_convention"],
            camera=dict(payload["camera"]) if isinstance(payload["camera"], Mapping) else payload["camera"],
            scene=dict(payload["scene"]) if isinstance(payload["scene"], Mapping) else payload["scene"],
            initial_state_stratum=payload["initial_state_stratum"],
            instruction=payload["instruction"],
            goal_reference_hashes=dict(payload["goal_reference_hashes"])
            if isinstance(payload["goal_reference_hashes"], Mapping)
            else payload["goal_reference_hashes"],
            goal_reference_provenance=dict(payload["goal_reference_provenance"])
            if isinstance(payload["goal_reference_provenance"], Mapping)
            else payload["goal_reference_provenance"],
            source_scene_family=payload["source_scene_family"],
            world_seed=payload.get("world_seed"),
            notes=tuple(notes) if isinstance(notes, (list, tuple)) else (str(notes),),
        )


@dataclass(frozen=True)
class IndependencePolicy:
    """Declared, frozen-before-acquisition independence rules for new starts."""

    min_source_frame_gap: int = DEFAULT_MIN_SOURCE_FRAME_GAP
    max_starts_per_episode: Optional[int] = None
    reject_repeated_world_seed: bool = True
    reject_repeated_source_pixels: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_source_frame_gap, bool)
            or not isinstance(self.min_source_frame_gap, int)
            or self.min_source_frame_gap < 1
        ):
            raise ScenarioError("min_source_frame_gap must be a positive integer")
        if self.max_starts_per_episode is not None and (
            isinstance(self.max_starts_per_episode, bool)
            or not isinstance(self.max_starts_per_episode, int)
            or self.max_starts_per_episode < 1
        ):
            raise ScenarioError("max_starts_per_episode must be a positive integer or null")

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "min_source_frame_gap": self.min_source_frame_gap,
            "max_starts_per_episode": self.max_starts_per_episode,
            "reject_repeated_world_seed": self.reject_repeated_world_seed,
            "reject_repeated_source_pixels": self.reject_repeated_source_pixels,
            "basis": (
                "Declared protocol parameters frozen before acquisition. They bound obvious "
                "non-independence; they do not measure statistical independence."
            ),
        }


def independence_violations(
    starts: Sequence[StartRecord], policy: Optional[IndependencePolicy] = None
) -> Tuple[Dict[str, Any], ...]:
    """Return every reason these starts are not distinct physical starts.

    This is a pure report so a caller can inspect a candidate pool before
    committing it.  ``ScenarioManifest`` refuses to exist when it is non-empty.
    """

    rules = policy or IndependencePolicy()
    violations: List[Dict[str, Any]] = []
    by_physical: Dict[Tuple[str, str, str, int], List[StartRecord]] = {}
    by_frame_id: Dict[Tuple[str, str, str, str], List[StartRecord]] = {}
    by_episode: Dict[Tuple[str, str, str], List[StartRecord]] = {}
    by_image: Dict[str, List[StartRecord]] = {}
    by_state: Dict[str, List[StartRecord]] = {}
    by_seed: Dict[int, List[StartRecord]] = {}
    for start in starts:
        by_physical.setdefault(start.physical_start_key, []).append(start)
        by_frame_id.setdefault(start.episode_key + (start.frame_id,), []).append(start)
        by_episode.setdefault(start.episode_key, []).append(start)
        by_image.setdefault(start.image_hash.lower(), []).append(start)
        by_state.setdefault(start.state_hash.lower(), []).append(start)
        if start.world_seed is not None:
            by_seed.setdefault(start.world_seed, []).append(start)

    for key, group in sorted(by_physical.items(), key=lambda item: str(item[0])):
        if len(group) > 1:
            seeds = sorted({start.world_seed for start in group if start.world_seed is not None})
            violations.append(
                {
                    "reason": "repeated_world_seed_is_not_a_new_start"
                    if len(seeds) > 1
                    else "repeated_physical_start",
                    "start_ids": sorted(start.start_id for start in group),
                    "detail": {
                        "source_dataset": key[0],
                        "source_revision": key[1],
                        "episode_id": key[2],
                        "frame_index": key[3],
                        "world_seeds": seeds,
                    },
                }
            )
    for key, group in sorted(by_frame_id.items(), key=lambda item: str(item[0])):
        if len(group) > 1 and len({start.frame_index for start in group}) == 1:
            continue  # already reported as a repeated physical start
        if len(group) > 1:
            violations.append(
                {
                    "reason": "duplicate_source_frame_id",
                    "start_ids": sorted(start.start_id for start in group),
                    "detail": {"episode_id": key[2], "frame_id": key[3]},
                }
            )
    for key, group in sorted(by_episode.items(), key=lambda item: str(item[0])):
        ordered = sorted(group, key=lambda start: (start.frame_index, start.start_id))
        for first, second in zip(ordered, ordered[1:]):
            gap = second.frame_index - first.frame_index
            if 0 < gap < rules.min_source_frame_gap:
                violations.append(
                    {
                        "reason": "neighbouring_source_frames_are_not_independent_starts",
                        "start_ids": sorted((first.start_id, second.start_id)),
                        "detail": {
                            "episode_id": key[2],
                            "frame_gap": gap,
                            "min_source_frame_gap": rules.min_source_frame_gap,
                        },
                    }
                )
        if rules.max_starts_per_episode is not None and len(ordered) > rules.max_starts_per_episode:
            violations.append(
                {
                    "reason": "too_many_starts_from_one_source_episode",
                    "start_ids": sorted(start.start_id for start in ordered),
                    "detail": {
                        "episode_id": key[2],
                        "count": len(ordered),
                        "max_starts_per_episode": rules.max_starts_per_episode,
                    },
                }
            )
    if rules.reject_repeated_source_pixels:
        for digest, group in sorted(by_image.items()):
            if len(group) > 1:
                violations.append(
                    {
                        "reason": "repeated_source_image",
                        "start_ids": sorted(start.start_id for start in group),
                        "detail": {"image_hash": digest},
                    }
                )
        for digest, group in sorted(by_state.items()):
            if len(group) > 1:
                violations.append(
                    {
                        "reason": "repeated_source_state",
                        "start_ids": sorted(start.start_id for start in group),
                        "detail": {"state_hash": digest},
                    }
                )
    if rules.reject_repeated_world_seed:
        for seed, group in sorted(by_seed.items()):
            if len(group) > 1:
                violations.append(
                    {
                        "reason": "repeated_world_seed",
                        "start_ids": sorted(start.start_id for start in group),
                        "detail": {"world_seed": seed},
                    }
                )
    return tuple(violations)


@dataclass(frozen=True)
class TaskPanel:
    """One task's cohort panels, comparability status, and documented limits."""

    task: str
    comparability: str
    starts_by_cohort: Mapping[str, Tuple[StartRecord, ...]]
    limitations: Tuple[str, ...] = ()
    scene_claims: Tuple[SceneClaim, ...] = ()
    success_criterion: Optional[SuccessCriterion] = None
    autoeval_comparison: Tuple[ComparisonAxis, ...] = ()
    acquisition_dependencies: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task not in EXACT_TASK_PROMPTS:
            raise ScenarioError("panel task %r is not one of the five benchmark tasks" % (self.task,))
        if self.comparability not in COMPARABILITY_VALUES:
            raise ScenarioError("panel comparability must be one of %s" % ", ".join(COMPARABILITY_VALUES))
        if not isinstance(self.starts_by_cohort, Mapping):
            raise ScenarioError("panel starts_by_cohort must be a mapping of cohort to starts")
        for cohort, group in self.starts_by_cohort.items():
            if cohort not in COHORTS:
                raise ScenarioError("panel cohort %r is not a known cohort" % (cohort,))
            for start in group:
                if not isinstance(start, StartRecord):
                    raise ScenarioError("panel starts must be StartRecord values")
                if start.task != self.task:
                    raise ScenarioError("start %s is not a %s start" % (start.start_id, self.task))
                if start.cohort != cohort:
                    raise ScenarioError(
                        "start %s declares cohort %s but sits in the %s panel"
                        % (start.start_id, start.cohort, cohort)
                    )
        if UNPAIRED_REFERENCE_LIMITATION not in self.limitations:
            raise ScenarioError(
                "every task panel must carry the unpaired-reference limitation: "
                "AutoEval's trial-level starts are not published"
            )
        if self.comparability == MATCHED_DISTRIBUTION and len(self.limitations) < 2:
            raise ScenarioError(
                "a matched-distribution panel must enumerate its own limitations, not only the "
                "shared unpaired-reference statement"
            )
        if self.comparability == MATCHED_PROVENANCE and self.task not in ("open_drawer", "close_drawer"):
            raise ScenarioError(
                "only the drawer tasks have a real AutoEval scene recording; %s cannot claim "
                "matched provenance" % self.task
            )
        for claim in self.scene_claims:
            if not isinstance(claim, SceneClaim):
                raise ScenarioError("panel scene_claims must be SceneClaim values")
            if claim.scene_family != TASK_SCENE_FAMILY[self.task]:
                raise ScenarioError(
                    "scene claim %s belongs to the %s scene, not %s"
                    % (claim.claim_id, claim.scene_family, self.task)
                )
        if self.success_criterion is not None and self.success_criterion.task != self.task:
            raise ScenarioError("panel success_criterion task does not match the panel task")
        axes = [axis.axis for axis in self.autoeval_comparison]
        if len(set(axes)) != len(axes):
            raise ScenarioError("panel autoeval_comparison repeats an axis")

    @property
    def starts(self) -> Tuple[StartRecord, ...]:
        ordered: List[StartRecord] = []
        for cohort in COHORTS:
            ordered.extend(self.starts_by_cohort.get(cohort, ()))
        return tuple(ordered)

    def primary_starts(self) -> Tuple[StartRecord, ...]:
        return tuple(self.starts_by_cohort.get(PRIMARY_COHORT, ()))

    def cohort_counts(self) -> Dict[str, int]:
        return {cohort: len(self.starts_by_cohort.get(cohort, ())) for cohort in COHORTS}

    def unresolved_axes(self) -> Tuple[str, ...]:
        recorded = {axis.axis: axis.status for axis in self.autoeval_comparison}
        return tuple(
            axis for axis in GATE_C_COMPARISON_AXES if recorded.get(axis, "unresolved") != "matched"
        )

    def unverified_claims(self) -> Tuple[str, ...]:
        return tuple(claim.claim_id for claim in self.scene_claims if not claim.verified)

    def task_status(self) -> str:
        """Per-task Gate C status.

        ``GateStatus`` has no ``pass_with_limitations`` member, so this richer
        per-task value is reported inside the Gate C measurements instead of
        being forced into the gate ledger's enum.
        """

        if len(self.primary_starts()) != PRIMARY_STARTS_PER_TASK:
            return "blocked"
        if self.comparability == MATCHED_DISTRIBUTION:
            return "pass_with_limitations"
        if self.unresolved_axes() or self.unverified_claims():
            return "pass_with_limitations"
        if self.success_criterion is not None and not self.success_criterion.resolved:
            return "pass_with_limitations"
        return "pass"

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "instruction": EXACT_TASK_PROMPTS[self.task],
            "comparability": self.comparability,
            "comparability_meaning": (
                "Matched source provenance for this scene; still not a paired reconstruction of an "
                "original AutoEval trial."
                if self.comparability == MATCHED_PROVENANCE
                else "Matched-distribution comparison with documented limitations; not paired "
                "reconstruction of the original trials."
            ),
            "limitations": list(self.limitations),
            "scene_claims": [claim.as_mapping() for claim in self.scene_claims],
            "success_criterion": None
            if self.success_criterion is None
            else self.success_criterion.as_mapping(),
            "autoeval_comparison": [axis.as_mapping() for axis in self.autoeval_comparison],
            "acquisition_dependencies": list(self.acquisition_dependencies),
            "cohort_counts": self.cohort_counts(),
            "task_status": self.task_status(),
            "starts": [start.as_mapping() for start in self.primary_starts()],
        }


def default_limitations(task: str) -> Tuple[str, ...]:
    """Return the enumerated limitations for one task's comparability claim."""

    shared = (UNPAIRED_REFERENCE_LIMITATION,)
    if TASK_SCENE_FAMILY[task] == DRAWER_SCENE:
        return shared + (
            "Scene comparability rests on the public zhouzypaul/auto_eval drawer recording; "
            "policy checkpoint/wrapper, control timing, and reset randomization are compared "
            "separately per axis and stay unresolved until each is checked.",
        )
    return shared + (
        "No AutoEval scene asset or scene record for this task has been acquired, so scene layout, "
        "object set, and reset/randomization distribution are matched only by source-dataset "
        "language instruction.",
        "IPEC-COMMUNITY/bridge_orig_lerobot is not automatically the same scene distribution as "
        "AutoEval; instruction-matched episodes can differ in table, lighting, camera pose, and "
        "object instance.",
        "Per-cell comparison against the published table is distribution-level only; a discrepancy "
        "cannot be attributed to the world model without resolving these axes.",
    )


def default_task_panel(
    task: str,
    starts_by_cohort: Mapping[str, Sequence[StartRecord]],
    *,
    scene_claims: Optional[Sequence[SceneClaim]] = None,
    autoeval_comparison: Optional[Sequence[ComparisonAxis]] = None,
) -> TaskPanel:
    """Build one task panel with the conservative default status for that task."""

    family = TASK_SCENE_FAMILY[task]
    comparability = MATCHED_PROVENANCE if family == DRAWER_SCENE else MATCHED_DISTRIBUTION
    dependencies: List[str] = []
    if family != DRAWER_SCENE:
        dependencies.append(SINK_CLOTH_ACQUISITION_DEPENDENCY)
    criterion = DEFAULT_SUCCESS_CRITERIA[task]
    if criterion.calibration_reference_required and criterion.calibration_reference_status != "resolved":
        dependencies.append(CALIBRATION_REFERENCE_DEPENDENCY)
    claims = tuple(scene_claims) if scene_claims is not None else default_scene_claims(family)
    if any(not claim.verified for claim in claims):
        dependencies.append(SCENE_CLAIM_VERIFICATION_DEPENDENCY)
    return TaskPanel(
        task=task,
        comparability=comparability,
        starts_by_cohort={
            cohort: tuple(group) for cohort, group in starts_by_cohort.items() if group
        },
        limitations=default_limitations(task),
        scene_claims=claims,
        success_criterion=criterion,
        autoeval_comparison=tuple(autoeval_comparison)
        if autoeval_comparison is not None
        else default_comparison_axes(),
        acquisition_dependencies=tuple(dict.fromkeys(dependencies)),
    )


@dataclass(frozen=True)
class ScenarioManifest:
    """The validated scenario manifest written to ``scenarios.jsonl``."""

    manifest_id: str
    provenance_kind: str
    panels: Tuple[TaskPanel, ...]
    sources: Tuple[SourceDataset, ...]
    created_at: str = field(default_factory=utc_now)
    independence_policy: IndependencePolicy = field(default_factory=IndependencePolicy)
    schema_version: int = SCHEMA_VERSION
    open_dependencies: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert_verbatim_prompts()
        _require_string(self.manifest_id, "manifest_id")
        _require_string(self.provenance_kind, "provenance_kind")
        _require_string(self.created_at, "created_at")
        if self.provenance_kind.strip().lower() in SYNTHETIC_PROVENANCE_VALUES:
            raise ScenarioError("synthetic scenario provenance cannot qualify a real-reference cell")
        if int(self.schema_version) < 1:
            raise ScenarioError("schema_version must be at least 1")
        if not self.panels:
            raise ScenarioError("a scenario manifest needs at least one task panel")
        tasks = [panel.task for panel in self.panels]
        if len(set(tasks)) != len(tasks):
            raise ScenarioError("a scenario manifest may hold only one panel per task")
        if not self.sources:
            raise ScenarioError("a scenario manifest must name its real source datasets")
        declared_sources = {source.dataset_id for source in self.sources}
        starts = self.all_starts()
        if not starts:
            raise ScenarioError("a scenario manifest needs at least one start")
        start_ids = [start.start_id for start in starts]
        duplicate_ids = sorted({value for value in start_ids if start_ids.count(value) > 1})
        if duplicate_ids:
            raise ScenarioError("duplicate start_id values: %s" % _brief(duplicate_ids))
        lineages: Dict[str, List[StartRecord]] = {}
        for start in starts:
            if start.source_dataset not in declared_sources:
                raise ScenarioError(
                    "start %s cites undeclared source dataset %s" % (start.start_id, start.source_dataset)
                )
            lineages.setdefault(start.start_lineage_id, []).append(start)
        repeated_lineages = sorted(key for key, group in lineages.items() if len(group) > 1)
        if repeated_lineages:
            raise ScenarioError(
                "a start_lineage_id identifies one physical start and may appear once: %s"
                % _brief(repeated_lineages)
            )
        leakage = self.lineage_separation()
        if leakage["status"] == "fail":
            raise ScenarioError(
                "start_lineage_id crosses cohorts: %s"
                % _brief([json.dumps(item, sort_keys=True) for item in leakage["violations"]])
            )
        if leakage["status"] == "unverifiable":
            raise ScenarioError("cohort separation is unverifiable: %s" % leakage["reason"])
        violations = independence_violations(starts, self.independence_policy)
        if violations:
            raise ScenarioError(
                "starts are not independent physical starts: %s"
                % _brief([json.dumps(item, sort_keys=True) for item in violations])
            )

    def all_starts(self) -> Tuple[StartRecord, ...]:
        ordered: List[StartRecord] = []
        for panel in self.panels:
            ordered.extend(panel.starts)
        return tuple(ordered)

    def panel(self, task: str) -> TaskPanel:
        for panel in self.panels:
            if panel.task == task:
                return panel
        raise ScenarioError("no panel for task %r" % (task,))

    def lineage_separation(self) -> Dict[str, Any]:
        """Check cohort separation with the shared measurement guard."""

        rows = [
            {"start_lineage_id": start.start_lineage_id, "cohort": start.cohort}
            for start in self.all_starts()
        ]
        return validate_source_lineage_leakage(rows)

    def cohort_counts(self) -> Dict[str, Dict[str, int]]:
        return {
            cohort: {panel.task: len(panel.starts_by_cohort.get(cohort, ())) for panel in self.panels}
            for cohort in COHORTS
        }

    def missing_cohorts(self) -> Tuple[str, ...]:
        counts = self.cohort_counts()
        return tuple(
            cohort for cohort in REQUIRED_COHORTS if not any(counts[cohort].get(panel.task) for panel in self.panels)
        )

    def per_episode_start_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for start in self.all_starts():
            key = "%s@%s:%s" % (start.source_dataset, start.source_revision, start.episode_id)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def measurements(self) -> Dict[str, Any]:
        """Actual counted properties of this manifest; no estimates."""

        per_episode = self.per_episode_start_counts()
        starts = self.all_starts()
        return {
            "total_starts": len(starts),
            # ``starts_per_task`` and ``comparability`` are the exact keys
            # ``plumb.gates.GateRecord.pass_evidence_errors`` reads for Gate C.
            "starts_per_task": {panel.task: len(panel.primary_starts()) for panel in self.panels},
            "comparability": {
                panel.task: {
                    "comparability": panel.comparability,
                    "limitations": list(panel.limitations),
                    "task_status": panel.task_status(),
                    "unresolved_comparison_axes": list(panel.unresolved_axes()),
                    "unverified_scene_claims": list(panel.unverified_claims()),
                    "acquisition_dependencies": list(panel.acquisition_dependencies),
                }
                for panel in self.panels
            },
            "cohort_counts": self.cohort_counts(),
            "distinct_source_episodes": len(per_episode),
            "max_starts_per_source_episode": max(per_episode.values()) if per_episode else 0,
            "distinct_state_conventions": sorted({start.state_convention for start in starts}),
            "declared_bridge_state_convention": BRIDGE_STATE_CONVENTION,
            "task_status": {panel.task: panel.task_status() for panel in self.panels},
            "unresolved_comparison_axes": {
                panel.task: list(panel.unresolved_axes()) for panel in self.panels
            },
            "unverified_scene_claims": {
                panel.task: list(panel.unverified_claims()) for panel in self.panels
            },
            "calibration_reference_status": {
                panel.task: None
                if panel.success_criterion is None
                else panel.success_criterion.calibration_reference_status
                for panel in self.panels
            },
            "camera_calibration_status": {
                start.start_id: start.camera.get("calibration", {}).get("status")
                for start in starts
                if isinstance(start.camera.get("calibration"), Mapping)
            },
            "lineage_separation_status": self.lineage_separation()["status"],
            "missing_required_cohorts": list(self.missing_cohorts()),
            "verbatim_prompts": verbatim_prompt_consistency()["status"],
        }

    def thresholds(self) -> Dict[str, Any]:
        """Thresholds frozen before Gate C evaluation."""

        return {
            "primary_starts_per_task": PRIMARY_STARTS_PER_TASK,
            "required_cohorts": list(REQUIRED_COHORTS),
            "lineage_cohort_crossings_allowed": 0,
            "verbatim_instruction_match_required": True,
            "independence_policy": self.independence_policy.as_mapping(),
            "comparability_values": list(COMPARABILITY_VALUES),
            "scene_claims_must_be_verified_for_pass": True,
            "calibration_reference_must_be_resolved_for_pass": True,
        }

    def resolved_open_dependencies(self) -> Tuple[str, ...]:
        """Union of explicitly declared and structurally implied dependencies."""

        dependencies: List[str] = list(self.open_dependencies)
        for panel in self.panels:
            dependencies.extend(panel.acquisition_dependencies)
        for cohort in self.missing_cohorts():
            dependencies.append("missing_cohort_panel:%s" % cohort)
        for panel in self.panels:
            if len(panel.primary_starts()) != PRIMARY_STARTS_PER_TASK:
                dependencies.append(
                    "incomplete_primary_panel:%s (%d of %d starts)"
                    % (panel.task, len(panel.primary_starts()), PRIMARY_STARTS_PER_TASK)
                )
        for source in self.sources:
            if not source.revision_immutable:
                dependencies.append("unpinned_source_revision:%s" % source.dataset_id)
        return tuple(dict.fromkeys(dependencies))

    def _payload(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "manifest_id": self.manifest_id,
            "provenance_kind": self.provenance_kind,
            "created_at": self.created_at,
            "manifest_status": self.manifest_status(),
            "unacquired_task_panels": list(self.absent_tasks()),
            "tasks": {panel.task: panel.as_mapping() for panel in self.panels},
            "cohorts": {
                cohort: {
                    "tasks": {
                        panel.task: {
                            "starts": [
                                start.as_mapping() for start in panel.starts_by_cohort.get(cohort, ())
                            ]
                        }
                        for panel in self.panels
                    }
                }
                for cohort in COHORTS
            },
            "sources": [source.as_mapping() for source in self.sources],
            "independence_policy": self.independence_policy.as_mapping(),
            "lineage_separation": self.lineage_separation(),
            "measurements": self.measurements(),
            "thresholds": self.thresholds(),
            "open_dependencies": list(self.resolved_open_dependencies()),
            "notes": list(self.notes),
        }

    def to_manifest_mapping(self) -> Dict[str, Any]:
        """Return the mapping ``ScenarioManifestValidator`` consumes.

        ``tasks.<task>.starts`` holds the primary cohort only, so the
        validator's ``require_primary_panel`` count means what it says.  Other
        cohorts live under ``cohorts``.
        """

        payload = self._payload()
        payload["sha256"] = canonical_json_sha256(payload)
        return payload

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self._payload())

    def validate(self, *, task: Optional[str] = None, require_primary_panel: bool = True) -> Tuple[str, ...]:
        """Return the gate validator's errors for this manifest."""

        return ScenarioManifestValidator.validate(
            self.to_manifest_mapping(), task=task, require_primary_panel=require_primary_panel
        )

    def absent_tasks(self) -> Tuple[str, ...]:
        """Tasks with no panel at all, in the frozen task order."""

        present = {panel.task for panel in self.panels}
        return tuple(task for task in TASKS if task not in present)

    def partition_errors(self, errors: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
        """Split validator errors into unacquired-panel and blocking groups.

        A task with no acquired scene asset is an *acquisition dependency*, not
        a malformed manifest.  Separating the two lets the drawer panel be
        written and audited without pretending the sink and cloth cells exist.
        """

        deferrable_messages = set()
        for task in self.absent_tasks():
            deferrable_messages.add("task %s has no starts" % task)
            deferrable_messages.add("task %s needs exactly 50 distinct primary starts" % task)
        deferrable = tuple(message for message in errors if message in deferrable_messages)
        blocking = tuple(message for message in errors if message not in deferrable_messages)
        return {"unacquired_task_panels": deferrable, "blocking": blocking}

    def manifest_status(self) -> str:
        """``complete`` only when all five tasks carry a full primary panel."""

        if self.absent_tasks():
            return "incomplete"
        if any(len(panel.primary_starts()) != PRIMARY_STARTS_PER_TASK for panel in self.panels):
            return "incomplete"
        return "complete"

    def jsonl_lines(
        self, *, require_primary_panel: bool = True, allow_incomplete: bool = False
    ) -> Tuple[str, ...]:
        """Validate, then render the header and one line per start record."""

        errors = self.validate(require_primary_panel=require_primary_panel)
        split = self.partition_errors(errors)
        if split["blocking"] or (errors and not allow_incomplete):
            raise ScenarioError("scenario manifest is invalid: %s" % "; ".join(errors))
        mapping = self.to_manifest_mapping()
        header = {key: value for key, value in mapping.items() if key not in ("tasks", "cohorts")}
        header["record_type"] = "manifest_header"
        header["tasks"] = {
            panel.task: {
                key: value
                for key, value in mapping["tasks"][panel.task].items()
                if key != "starts"
            }
            for panel in self.panels
        }
        header["start_record_count"] = len(self.all_starts())
        # Derived audit context: intentionally outside the hashed payload.
        header["scenario_validator_errors"] = list(errors)
        lines = [json.dumps(header, sort_keys=True, ensure_ascii=False, allow_nan=False)]
        for start in self.all_starts():
            row = dict(start.as_mapping())
            row["record_type"] = "start"
            row["manifest_id"] = self.manifest_id
            row["manifest_sha256"] = mapping["sha256"]
            lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return tuple(lines)

    def write_jsonl(
        self,
        path: Union[str, Path],
        *,
        require_primary_panel: bool = True,
        overwrite: bool = False,
        allow_incomplete: bool = False,
    ) -> Dict[str, Any]:
        """Validate then atomically write ``scenarios.jsonl``; refuse to clobber."""

        lines = self.jsonl_lines(
            require_primary_panel=require_primary_panel, allow_incomplete=allow_incomplete
        )
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise ScenarioError("refusing to overwrite an existing scenario manifest: %s" % destination)
        _atomic_write_text(destination, "\n".join(lines) + "\n")
        return {
            "path": str(destination),
            "manifest_id": self.manifest_id,
            "sha256": self.sha256,
            "start_records": len(self.all_starts()),
            "manifest_status": self.manifest_status(),
            "unacquired_task_panels": list(self.absent_tasks()),
            "require_primary_panel": require_primary_panel,
        }

    def write_json(
        self,
        path: Union[str, Path],
        *,
        require_primary_panel: bool = True,
        overwrite: bool = False,
        allow_incomplete: bool = False,
    ) -> Dict[str, Any]:
        """Write the validator-shaped mapping for the qualification validator."""

        errors = self.validate(require_primary_panel=require_primary_panel)
        split = self.partition_errors(errors)
        if split["blocking"] or (errors and not allow_incomplete):
            raise ScenarioError("scenario manifest is invalid: %s" % "; ".join(errors))
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise ScenarioError("refusing to overwrite an existing scenario manifest: %s" % destination)
        mapping = self.to_manifest_mapping()
        _atomic_write_text(
            destination, json.dumps(mapping, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        return {
            "path": str(destination),
            "manifest_id": self.manifest_id,
            "sha256": self.sha256,
            "manifest_status": self.manifest_status(),
        }


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


def read_manifest_jsonl(path: Union[str, Path]) -> Dict[str, Any]:
    """Read ``scenarios.jsonl`` back and re-verify its recorded hash.

    Returns a status/reason mapping rather than raising for a hash mismatch, so
    an auditor sees the disagreement instead of a traceback.
    """

    source = Path(path)
    if not source.is_file():
        raise ScenarioError("scenario manifest does not exist: %s" % source)
    header: Optional[Dict[str, Any]] = None
    starts: List[Dict[str, Any]] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise ScenarioError("invalid JSONL on line %d of %s" % (number, source)) from error
        if not isinstance(parsed, Mapping):
            raise ScenarioError("scenario JSONL rows must be objects")
        kind = parsed.get("record_type")
        if kind == "manifest_header":
            if header is not None:
                raise ScenarioError("scenario manifest has more than one header record")
            header = dict(parsed)
        elif kind == "start":
            starts.append(dict(parsed))
        else:
            raise ScenarioError("unknown scenario record_type %r on line %d" % (kind, number))
    if header is None:
        raise ScenarioError("scenario manifest has no manifest_header record")
    declared = header.get("sha256")
    expected_count = header.get("start_record_count")
    count_matches = expected_count == len(starts)
    referenced = {start.get("manifest_sha256") for start in starts}
    binding = referenced <= {declared} if starts else True
    return {
        "status": "pass" if count_matches and binding else "fail",
        "header": header,
        "starts": starts,
        "declared_sha256": declared,
        "start_record_count": len(starts),
        "reason": None
        if count_matches and binding
        else "start record count or manifest_sha256 binding does not match the header",
    }


def gate_c_evidence(
    manifest: ScenarioManifest,
    *,
    evidence_uris: Sequence[str],
    protocol_hash: Optional[str] = None,
    evidence_kind: str = "real_source_frames",
    backend_revisions: Optional[Mapping[str, Any]] = None,
    policy_revisions: Optional[Mapping[str, Any]] = None,
    require_primary_panel: bool = True,
) -> Dict[str, Any]:
    """Build a Gate C evidence bundle shaped for ``plumb.gates.GateRecord``.

    The status is never ``pass`` while a comparability axis, scene claim, or
    calibration reference is unresolved.  ``GateStatus`` has no
    ``pass_with_limitations`` member, so the richer per-task statuses are
    reported inside ``measurements``.
    """

    errors = manifest.validate(require_primary_panel=require_primary_panel)
    reasons: List[str] = list(errors)
    dependencies = manifest.resolved_open_dependencies()
    reasons.extend("open dependency: %s" % item for item in dependencies)
    if not evidence_uris:
        reasons.append("Gate C evidence bundle needs at least one evidence URI")
    if not protocol_hash:
        reasons.append("Gate C evidence bundle needs the frozen protocol hash")
    statuses = {panel.task: panel.task_status() for panel in manifest.panels}
    if reasons or any(value != "pass" for value in statuses.values()):
        status = "blocked"
    else:
        status = "pass"
    measurements = dict(manifest.measurements())
    measurements["per_task_gate_status"] = statuses
    measurements["scenario_manifest_sha256"] = manifest.sha256
    measurements["scenario_manifest_validator_errors"] = list(errors)
    return {
        "gate_id": "C",
        "status": status,
        "protocol_hash": protocol_hash,
        "fixture_ids": [],
        "start_ids": [start.start_id for start in manifest.all_starts()],
        "backend_revisions": dict(backend_revisions or {}),
        "policy_revisions": dict(policy_revisions or {}),
        "judge_revisions": {},
        "evidence_uris": list(evidence_uris),
        "measurements": measurements,
        "thresholds": manifest.thresholds(),
        "reasons": reasons,
        "evidence_kind": evidence_kind,
        "open_dependencies": list(dependencies),
    }


def build_manifest(
    manifest_id: str,
    starts: Iterable[StartRecord],
    sources: Sequence[SourceDataset],
    *,
    provenance_kind: str = "real_source_frames",
    independence_policy: Optional[IndependencePolicy] = None,
    scene_claims: Optional[Mapping[str, Sequence[SceneClaim]]] = None,
    autoeval_comparison: Optional[Mapping[str, Sequence[ComparisonAxis]]] = None,
    created_at: Optional[str] = None,
    open_dependencies: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> ScenarioManifest:
    """Group real start records into task panels with conservative defaults."""

    grouped: Dict[str, Dict[str, List[StartRecord]]] = {}
    for start in starts:
        if not isinstance(start, StartRecord):
            raise ScenarioError("build_manifest requires StartRecord values")
        grouped.setdefault(start.task, {}).setdefault(start.cohort, []).append(start)
    if not grouped:
        raise ScenarioError("build_manifest needs at least one start record")
    panels = tuple(
        default_task_panel(
            task,
            grouped[task],
            scene_claims=None if scene_claims is None else scene_claims.get(task),
            autoeval_comparison=None if autoeval_comparison is None else autoeval_comparison.get(task),
        )
        for task in TASKS
        if task in grouped
    )
    return ScenarioManifest(
        manifest_id=manifest_id,
        provenance_kind=provenance_kind,
        panels=panels,
        sources=tuple(sources),
        created_at=created_at or utc_now(),
        independence_policy=independence_policy or IndependencePolicy(),
        open_dependencies=tuple(open_dependencies),
        notes=tuple(notes),
    )


__all__ = [
    "AUTO_EVAL_DATASET",
    "BRIDGE_LEROBOT_DATASET",
    "BRIDGE_STATE_CONVENTION",
    "CALIBRATION_COHORT",
    "CALIBRATION_REFERENCE_DEPENDENCY",
    "COHORTS",
    "COMPARABILITY_VALUES",
    "COST_CONFIRMATION_COHORT",
    "ComparisonAxis",
    "DEFAULT_MIN_SOURCE_FRAME_GAP",
    "DEFAULT_SUCCESS_CRITERIA",
    "DEVELOPMENT_COHORT",
    "GATE_C_COMPARISON_AXES",
    "IndependencePolicy",
    "MATCHED_DISTRIBUTION",
    "MATCHED_PROVENANCE",
    "PRIMARY_COHORT",
    "PRIMARY_STARTS_PER_TASK",
    "REQUIRED_COHORTS",
    "REVERSE_VALIDATION_COHORT",
    "SCENE_FAMILIES",
    "SINK_CLOTH_ACQUISITION_DEPENDENCY",
    "SCHEMA_VERSION",
    "SceneClaim",
    "ScenarioError",
    "ScenarioManifest",
    "SourceDataset",
    "StartRecord",
    "SuccessCriterion",
    "TASKS",
    "TASK_SCENE_FAMILY",
    "TaskPanel",
    "UNPAIRED_REFERENCE_LIMITATION",
    "assert_verbatim_prompts",
    "build_manifest",
    "default_comparison_axes",
    "default_limitations",
    "default_scene_claims",
    "default_task_panel",
    "gate_c_evidence",
    "independence_violations",
    "read_manifest_jsonl",
    "utc_now",
    "verbatim_prompt",
    "verbatim_prompt_consistency",
]
