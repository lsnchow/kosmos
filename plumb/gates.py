"""Qualification-gate persistence and conservative manifest validation.

The ledger is intentionally unable to infer a qualified cell from a synthetic
fixture or from a backend process exiting successfully.  It records evidence;
``QualificationValidator`` makes a separate, inspectable decision from frozen
manifests and passed gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .adapters.contracts import CapabilityResult, CapabilityStatus, FeedbackMode


class GateStatus(str, Enum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"


GATE_IDS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

GATE_DESCRIPTIONS: Dict[str, str] = {
    "A": "Pinned model/backend conformance and measured GPU inference",
    "B": "Action grounding, native feedback, and state conversion",
    "C": "Matched real starting states for all five tasks",
    "D": "Blinded judge calibration against independent labels",
    "E": "Primary measurement and independent cost confirmation",
    "F": "Three fresh full-scale latency/cost rehearsals",
}

#: Spec section 0 replaces the old single shuffled-action rule with five paired
#: control arms plus legitimate-stationary and already-successful controls.
REQUIRED_GATE_B_ARMS: Tuple[str, ...] = (
    "original",
    "zero",
    "temporally_permuted",
    "sign_reversed",
    "cross_episode",
)

#: Spec section 7 fixes these as the operational definition of the public claim.
BURST_EPISODE_TARGET = 1500
BURST_SECONDS_TARGET = 60.0
BURST_USD_TARGET = 11.25


def _as_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _contains_synthetic(value: Any) -> bool:
    """Only real acquisition/execution evidence can support a passed gate."""

    if isinstance(value, str):
        return value.strip().lower() in {"synthetic", "simulated", "mock", "fake"}
    if isinstance(value, Mapping):
        return any(_contains_synthetic(item) for item in value.values())
    if isinstance(value, (tuple, list, set)):
        return any(_contains_synthetic(item) for item in value)
    return False


@dataclass(frozen=True)
class GateRecord:
    gate_id: str
    status: GateStatus = GateStatus.NOT_RUN
    protocol_hash: Optional[str] = None
    fixture_ids: Tuple[str, ...] = ()
    start_ids: Tuple[str, ...] = ()
    backend_revisions: Mapping[str, Any] = field(default_factory=dict)
    policy_revisions: Mapping[str, Any] = field(default_factory=dict)
    judge_revisions: Mapping[str, Any] = field(default_factory=dict)
    evidence_uris: Tuple[str, ...] = ()
    measurements: Mapping[str, Any] = field(default_factory=dict)
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    reasons: Tuple[str, ...] = ()
    evidence_kind: str = "unknown"

    def __post_init__(self) -> None:
        if self.gate_id not in GATE_IDS:
            raise ValueError("Unknown gate ID %r." % self.gate_id)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], gate_id: Optional[str] = None) -> "GateRecord":
        selected_id = gate_id or str(payload.get("gate_id") or payload.get("gate") or "")
        return cls(
            gate_id=selected_id,
            status=GateStatus(payload.get("status", GateStatus.NOT_RUN.value)),
            protocol_hash=payload.get("protocol_hash"),
            fixture_ids=_as_tuple(payload.get("fixture_ids")),
            start_ids=_as_tuple(payload.get("start_ids")),
            backend_revisions=dict(payload.get("backend_revisions", {})),
            policy_revisions=dict(payload.get("policy_revisions", {})),
            judge_revisions=dict(payload.get("judge_revisions", {})),
            evidence_uris=_as_tuple(payload.get("evidence_uris")),
            measurements=dict(payload.get("measurements", {})),
            thresholds=dict(payload.get("thresholds", {})),
            reasons=_as_tuple(payload.get("reasons")),
            evidence_kind=str(payload.get("evidence_kind", "unknown")),
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status.value,
            "protocol_hash": self.protocol_hash,
            "fixture_ids": list(self.fixture_ids),
            "start_ids": list(self.start_ids),
            "backend_revisions": dict(self.backend_revisions),
            "policy_revisions": dict(self.policy_revisions),
            "judge_revisions": dict(self.judge_revisions),
            "evidence_uris": list(self.evidence_uris),
            "measurements": dict(self.measurements),
            "thresholds": dict(self.thresholds),
            "reasons": list(self.reasons),
            "evidence_kind": self.evidence_kind,
        }

    def pass_evidence_errors(self) -> Tuple[str, ...]:
        """Return missing prerequisites for a pass; do not mutate the record."""

        if self.status is not GateStatus.PASS:
            return ()
        errors: List[str] = []
        if not self.protocol_hash:
            errors.append("passed gate lacks protocol_hash")
        if not self.fixture_ids and self.gate_id in ("A", "B", "D"):
            errors.append("passed gate lacks fixture IDs")
        if not self.start_ids and self.gate_id == "C":
            errors.append("passed scenario gate lacks start IDs")
        if not self.evidence_uris:
            errors.append("passed gate lacks evidence URIs")
        if not self.thresholds:
            errors.append("passed gate lacks frozen thresholds")
        if _contains_synthetic(self.evidence_kind) or _contains_synthetic(self.measurements):
            errors.append("synthetic/mock evidence may not support a passed qualification gate")
        if self.gate_id == "A":
            if int(self.measurements.get("backend_calls", 0) or 0) < 1:
                errors.append("Gate A needs at least one actual backend call")
            if self.measurements.get("wall_seconds") is None:
                errors.append("Gate A needs measured wall_seconds (null is not a pass)")
            if self.measurements.get("gpu_peak_memory_bytes") is None:
                errors.append("Gate A needs actual GPU peak memory")
        if self.gate_id == "B":
            interventions = self.measurements.get("interventions")
            if not interventions:
                errors.append("Gate B needs paired original/zero/permuted/sign/cross intervention evidence")
            else:
                present = {str(name) for name in interventions}
                missing = [arm for arm in REQUIRED_GATE_B_ARMS if arm not in present]
                if missing:
                    errors.append(
                        "Gate B is missing required control arms: %s" % ", ".join(sorted(missing))
                    )
            if self.measurements.get("suffix_causality") is None:
                errors.append(
                    "Gate B needs a suffix-causality result; it decides the feedback mode and cannot be skipped"
                )
            if not self.measurements.get("feedback_mode"):
                errors.append("Gate B must record the feedback_mode it certifies")
        if self.gate_id == "C":
            per_task = self.measurements.get("starts_per_task")
            if not isinstance(per_task, Mapping) or not per_task:
                errors.append("Gate C needs a per-task start count")
            else:
                thin = sorted(task for task, count in per_task.items() if int(count or 0) < 50)
                if thin:
                    errors.append("Gate C needs 50 starts per task; short: %s" % ", ".join(thin))
            comparability = self.measurements.get("comparability")
            if not isinstance(comparability, Mapping) or not comparability:
                errors.append("Gate C must record per-task comparability with its limitations")
        if self.gate_id == "D":
            for name in ("binary_kappa", "weighted_progress_kappa", "leniency_offset", "consensus_coverage"):
                if self.measurements.get(name) is None:
                    errors.append("Gate D needs a measured %s" % name)
            if not self.measurements.get("calibration_class"):
                errors.append("Gate D must record its calibration_class (human / external_label / model)")
            if not self.measurements.get("held_out_frozen_before_evaluation"):
                errors.append(
                    "Gate D requires the judge to be frozen before the held-out split was evaluated"
                )
        if self.gate_id == "E":
            # Spec section 0: Gate E is frozen analysis protocol + full matrix +
            # exclusion sensitivity + held-out cost confirmation + a separately
            # validated distilled judge.
            if not self.measurements.get("analysis_protocol_frozen"):
                errors.append("Gate E needs a frozen analysis protocol")
            planned = self.measurements.get("planned_episodes")
            terminal = self.measurements.get("terminal_episodes")
            if planned is None or terminal is None:
                errors.append("Gate E needs planned and terminal episode counts")
            elif int(planned) <= 0 or int(terminal) != int(planned):
                errors.append(
                    "Gate E needs every planned episode accounted for (%s of %s terminal)"
                    % (terminal, planned)
                )
            if self.measurements.get("exclusion_sensitivity") is None:
                errors.append("Gate E needs exclusion sensitivity results")
            confirmation = self.measurements.get("cost_setting_confirmation")
            if not isinstance(confirmation, Mapping) or not confirmation.get("confirmed_on_disjoint_panel"):
                errors.append(
                    "Gate E needs the selected cost setting confirmed on a disjoint panel"
                )
            distilled = self.measurements.get("distilled_judge")
            if isinstance(distilled, Mapping) and distilled.get("used_for_scoring"):
                if not distilled.get("separately_validated"):
                    errors.append(
                        "Gate E requires a separately validated distilled judge before it scores anything"
                    )
                else:
                    # ``separately_validated`` is a claim. Spec section 5 says what
                    # backs it: "A distilled judge must pass fresh held-out
                    # calibration and paired frozen-video comparison before
                    # scoring a burst." Require both records, so a hand-edited
                    # ledger cannot assert validation it never performed.
                    for record_name, label in (
                        ("fresh_heldout_calibration", "fresh held-out calibration"),
                        ("paired_frozen_video_comparison", "paired frozen-video comparison"),
                    ):
                        record = distilled.get(record_name)
                        if not isinstance(record, Mapping):
                            errors.append(
                                "Gate E claims a validated distilled judge but records no %s" % label
                            )
                        elif not record.get("passed"):
                            errors.append(
                                "Gate E's distilled judge did not pass its %s" % label
                            )
                    if not distilled.get("original_judge_results_preserved"):
                        errors.append(
                            "Gate E must preserve the original judge's results alongside the distilled ones"
                        )
        if self.gate_id == "F":
            rehearsals = self.measurements.get("rehearsals")
            if not isinstance(rehearsals, (list, tuple)) or len(rehearsals) < 3:
                errors.append("Gate F needs three complete full-scale rehearsals, not best-of-three")
            else:
                for index, rehearsal in enumerate(rehearsals):
                    if not isinstance(rehearsal, Mapping):
                        errors.append("Gate F rehearsal %d is malformed" % index)
                        continue
                    label = str(rehearsal.get("run_id") or index)
                    planned = int(rehearsal.get("planned_episodes") or 0)
                    finished = int(rehearsal.get("generated_episodes") or 0)
                    if planned != BURST_EPISODE_TARGET or finished != planned:
                        errors.append(
                            "Gate F rehearsal %s must complete all %d planned episodes (%d of %d)"
                            % (label, BURST_EPISODE_TARGET, finished, planned)
                        )
                    if int(rehearsal.get("terminal_service_failures") or 0) > 0:
                        errors.append(
                            "Gate F rehearsal %s had terminal service failures" % label
                        )
                    seconds = rehearsal.get("elapsed_seconds")
                    if seconds is None or float(seconds) > BURST_SECONDS_TARGET:
                        errors.append(
                            "Gate F rehearsal %s elapsed %s s against the %.0f s target"
                            % (label, seconds, BURST_SECONDS_TARGET)
                        )
                    usd = rehearsal.get("total_demonstration_usd")
                    if usd is None or float(usd) > BURST_USD_TARGET:
                        errors.append(
                            "Gate F rehearsal %s total demonstration cost %s against the $%.2f definition"
                            % (label, usd, BURST_USD_TARGET)
                        )
                    if rehearsal.get("reused_cached_outputs"):
                        errors.append(
                            "Gate F rehearsal %s reused cached segments/videos/judge outputs" % label
                        )
            if not self.measurements.get("displayed_capacity"):
                errors.append("Gate F must record the capacity and configuration it displayed")
            if self.measurements.get("cost_basis") is None:
                errors.append("Gate F needs the pricing basis used for its cost figures")
        return tuple(errors)


@dataclass
class GateLedger:
    protocol_hash: Optional[str] = None
    records: Dict[str, GateRecord] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        for gate_id in GATE_IDS:
            self.records.setdefault(gate_id, GateRecord(gate_id=gate_id, protocol_hash=self.protocol_hash))

    @classmethod
    def new(cls, protocol_hash: Optional[str] = None) -> "GateLedger":
        return cls(protocol_hash=protocol_hash)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GateLedger":
        raw = payload.get("gates", payload.get("records", {}))
        records: Dict[str, GateRecord] = {}
        if isinstance(raw, Mapping):
            for gate_id, item in raw.items():
                records[str(gate_id)] = GateRecord.from_mapping(item, str(gate_id))
        elif isinstance(raw, list):
            for item in raw:
                record = GateRecord.from_mapping(item)
                records[record.gate_id] = record
        else:
            raise ValueError("gates.json gates must be a mapping or list.")
        return cls(
            protocol_hash=payload.get("protocol_hash"),
            records=records,
            schema_version=int(payload.get("schema_version", 1)),
        )

    @classmethod
    def load(cls, path: str) -> "GateLedger":
        source = Path(path)
        if not source.exists():
            return cls.new()
        with source.open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_hash": self.protocol_hash,
            "gates": {gate_id: self.records[gate_id].to_mapping() for gate_id in GATE_IDS},
        }

    def save(self, path: str) -> None:
        """Atomically save a project-owned ledger; remote preregistration is separate."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(self.to_mapping(), sort_keys=True, indent=2) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=".gates-", suffix=".json", dir=str(destination.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def record(self, record: GateRecord) -> None:
        if self.protocol_hash and record.protocol_hash and self.protocol_hash != record.protocol_hash:
            raise ValueError("Refusing to mix evidence from a different protocol hash into this gate ledger.")
        if self.protocol_hash is None and record.protocol_hash:
            self.protocol_hash = record.protocol_hash
        self.records[record.gate_id] = record

    def invalidate_for_change(self, changed: str, old_revision: Any, new_revision: Any) -> Tuple[str, ...]:
        """Invalidate dependent gates; retain the prior revision in the reason."""

        if old_revision == new_revision:
            return ()
        dependencies = {
            "backend": ("A", "B", "C", "E", "F"),
            "adapter": ("A", "B", "C", "E", "F"),
            "policy": ("B", "C", "E", "F"),
            "judge": ("D", "E", "F"),
            "scenario": ("C", "E", "F"),
            "operating_point": ("A", "B", "C", "D", "E", "F"),
        }
        if changed not in dependencies:
            raise ValueError("Unknown revision type %r." % changed)
        invalidated = []
        reason = "invalidated because %s revision changed from %r to %r" % (changed, old_revision, new_revision)
        for gate_id in dependencies[changed]:
            previous = self.records[gate_id]
            self.records[gate_id] = GateRecord(
                gate_id=gate_id,
                status=GateStatus.NOT_RUN,
                protocol_hash=self.protocol_hash,
                reasons=previous.reasons + (reason,),
            )
            invalidated.append(gate_id)
        return tuple(invalidated)


EXACT_TASK_PROMPTS = {
    "close_drawer": "Close the drawer",
    "open_drawer": "Open the drawer",
    "to_basket": "Put the eggplant in the yellow basket",
    "to_sink": "Put the eggplant in the blue sink",
    "fold_cloth": "fold the cloth from top right to bottom left",
}


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Local content hash; not an independently timestamped preregistration."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class ScenarioManifestValidator:
    REQUIRED_START_FIELDS = (
        "start_id",
        "start_lineage_id",
        "source_dataset",
        "source_revision",
        "episode_id",
        "frame_id",
        "image_hash",
        "state_hash",
        "instruction",
        "state_convention",
        "camera",
        "scene",
        "initial_state_stratum",
        "goal_reference_hashes",
    )

    @classmethod
    def _starts_for_task(cls, manifest: Mapping[str, Any], task: str) -> Sequence[Mapping[str, Any]]:
        tasks = manifest.get("tasks")
        if isinstance(tasks, Mapping):
            group = tasks.get(task, {})
            if isinstance(group, Mapping):
                return group.get("starts", ())
        starts = manifest.get("starts", ())
        return [item for item in starts if isinstance(item, Mapping) and item.get("task") == task]

    @classmethod
    def validate(
        cls,
        manifest: Mapping[str, Any],
        *,
        task: Optional[str] = None,
        require_primary_panel: bool = False,
    ) -> Tuple[str, ...]:
        errors: List[str] = []
        if int(manifest.get("schema_version", 0) or 0) < 1:
            errors.append("scenario manifest needs schema_version >= 1")
        if not manifest.get("manifest_id"):
            errors.append("scenario manifest needs manifest_id")
        if not str(manifest.get("sha256", "")).startswith("sha256:"):
            errors.append("scenario manifest needs immutable sha256")
        if _contains_synthetic(manifest.get("provenance_kind")):
            errors.append("synthetic scenario provenance cannot qualify a real-reference cell")
        tasks = (task,) if task else tuple(EXACT_TASK_PROMPTS)
        for task_name in tasks:
            if task_name not in EXACT_TASK_PROMPTS:
                errors.append("unknown benchmark task %r" % task_name)
                continue
            starts = cls._starts_for_task(manifest, task_name)
            if require_primary_panel and len(starts) != 50:
                errors.append("task %s needs exactly 50 distinct primary starts" % task_name)
            if not starts:
                errors.append("task %s has no starts" % task_name)
                continue
            ids = set()
            lineages = set()
            for index, start in enumerate(starts):
                missing = [field for field in cls.REQUIRED_START_FIELDS if not start.get(field)]
                if missing:
                    errors.append("task %s start %d lacks %s" % (task_name, index, ", ".join(missing)))
                    continue
                if start["instruction"] != EXACT_TASK_PROMPTS[task_name]:
                    errors.append("task %s start %s has a non-verbatim instruction" % (task_name, start["start_id"]))
                if start["start_id"] in ids:
                    errors.append("task %s duplicates start_id %s" % (task_name, start["start_id"]))
                ids.add(start["start_id"])
                if start["start_lineage_id"] in lineages:
                    errors.append("task %s repeats start_lineage_id %s" % (task_name, start["start_lineage_id"]))
                lineages.add(start["start_lineage_id"])
        return tuple(errors)


class ProtocolManifestValidator:
    @staticmethod
    def validate(protocol: Mapping[str, Any], scenario_manifest: Mapping[str, Any]) -> Tuple[str, ...]:
        errors: List[str] = []
        if not protocol.get("protocol_id"):
            errors.append("protocol needs protocol_id")
        if not str(protocol.get("sha256", "")).startswith("sha256:"):
            errors.append("protocol needs immutable sha256")
        if protocol.get("status") != "frozen" and protocol.get("frozen") is not True:
            errors.append("protocol is not frozen")
        expected = scenario_manifest.get("sha256")
        if expected and protocol.get("scenario_manifest_hash") != expected:
            errors.append("protocol scenario_manifest_hash does not match supplied scenario manifest")
        for field in ("policy", "world_model", "judge", "thresholds", "seeds"):
            if field not in protocol:
                errors.append("protocol lacks %s" % field)
        remote = protocol.get("remote_record") or protocol.get("preregistration_uri")
        if not remote:
            errors.append("protocol lacks externally auditable remote record URI")
        return tuple(errors)


@dataclass(frozen=True)
class QualificationDecision:
    capability: CapabilityResult
    required_gates: Tuple[str, ...]
    errors: Tuple[str, ...]


class QualificationValidator:
    """Conservative route from recorded evidence to a named qualified cell."""

    @staticmethod
    def validate(
        ledger: GateLedger,
        scenario_manifest: Mapping[str, Any],
        protocol: Mapping[str, Any],
        *,
        task: str,
        feedback_mode: FeedbackMode,
        requires_judge: bool = True,
        require_primary_panel: bool = False,
    ) -> QualificationDecision:
        required = ("A", "B", "C", "D") if requires_judge else ("A", "B", "C")
        errors: List[str] = []
        if feedback_mode is not FeedbackMode.NATIVE_FEEDBACK:
            errors.append(
                "feedback_mode=%s is an approximation cohort; only Gate-B-qualified native_feedback may populate a qualified cell"
                % feedback_mode.value
            )
        errors.extend(ScenarioManifestValidator.validate(
            scenario_manifest, task=task, require_primary_panel=require_primary_panel
        ))
        errors.extend(ProtocolManifestValidator.validate(protocol, scenario_manifest))
        protocol_hash = protocol.get("sha256")
        for gate_id in required:
            record = ledger.records[gate_id]
            if record.status is not GateStatus.PASS:
                errors.append("Gate %s is %s, not pass" % (gate_id, record.status.value))
                continue
            if record.protocol_hash != protocol_hash:
                errors.append("Gate %s protocol hash does not match frozen protocol" % gate_id)
            errors.extend("Gate %s: %s" % (gate_id, message) for message in record.pass_evidence_errors())
        if errors:
            capability = CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="Qualification blocked: " + "; ".join(errors),
                source_verified=False,
                details={"task": task, "feedback_mode": feedback_mode.value, "required_gates": list(required)},
            )
        else:
            capability = CapabilityResult(
                status=CapabilityStatus.QUALIFIED,
                reason="Manifest and actual gate evidence satisfy the selected qualification route.",
                source_verified=True,
                details={"task": task, "feedback_mode": feedback_mode.value, "required_gates": list(required)},
            )
        return QualificationDecision(capability=capability, required_gates=required, errors=tuple(errors))
