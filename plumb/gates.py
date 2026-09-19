"""Qualification-gate persistence and conservative manifest validation.

The ledger is intentionally unable to infer a qualified cell from a synthetic
fixture or from a backend process exiting successfully.  It records evidence;
``QualificationValidator`` makes a separate, inspectable decision from frozen
manifests and passed gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
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

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


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


def _contains_nonqualifying_execution(value: Any) -> bool:
    """Whether a value explicitly labels a run as non-qualifying execution.

    A vendor *fixture* may be genuine Gate A/B evidence, so ``fixture`` is not
    forbidden.  In contrast a simulated transport, rehearsal or diagnostic is
    deliberately useful engineering evidence but cannot survive an environment
    switch and become a real-backend gate pass.
    """

    labels = ("synthetic", "simulated", "mock", "fake", "rehearsal", "diagnostic")
    if isinstance(value, str):
        lowered = value.strip().lower()
        return any(label in lowered for label in labels)
    if isinstance(value, Mapping):
        return any(_contains_nonqualifying_execution(item) for item in value.values())
    if isinstance(value, (tuple, list, set)):
        return any(_contains_nonqualifying_execution(item) for item in value)
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
        if _contains_nonqualifying_execution(self.evidence_kind) or _contains_nonqualifying_execution(self.measurements):
            errors.append(
                "synthetic/simulated/rehearsal/diagnostic evidence may not support a passed qualification gate"
            )
        if self.gate_id == "A":
            backend_calls = self.measurements.get("backend_calls")
            if isinstance(backend_calls, bool) or not isinstance(backend_calls, int) or backend_calls < 1:
                errors.append("Gate A needs at least one actual backend call")
            if not _finite_positive(self.measurements.get("wall_seconds")):
                errors.append("Gate A needs a finite positive measured wall_seconds")
            gpu_peak = self.measurements.get("gpu_peak_memory_bytes")
            if isinstance(gpu_peak, bool) or not isinstance(gpu_peak, int) or gpu_peak < 1:
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
            if self.measurements.get("feedback_mode") != FeedbackMode.NATIVE_FEEDBACK.value:
                errors.append("Gate B must certify native_feedback; approximation modes stay unqualified")
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
            if self.measurements.get("calibration_class") != "human":
                errors.append("Gate D requires human calibration; model/external labels stay unqualified")
            if self.measurements.get("has_two_blinded_humans") is not True or self.measurements.get("human_annotator_count") != 2:
                errors.append("Gate D requires declarations of exactly two blinded human annotators")
            if self.measurements.get("gate_d_decision_status") != "pass":
                errors.append("Gate D pass_with_limitations is blocked until human calibration passes")
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
        protocol_hash = protocol.get("sha256") or protocol.get("protocol_sha256")
        if not _is_sha256(protocol_hash):
            errors.append("protocol needs immutable sha256")
        # The legacy mapping form stores its content hash under ``sha256``.
        # Do not merely check the prefix: otherwise an edited threshold or
        # scope can borrow the hash from the protocol it replaced.  The modern
        # ProtocolDocument form has a distinct ``protocol_sha256`` body and is
        # validated by ``ProtocolDocument.load`` before reaching this boundary.
        if "sha256" in protocol and _is_sha256(protocol.get("sha256")):
            computed = canonical_json_sha256({key: value for key, value in protocol.items() if key != "sha256"})
            if protocol["sha256"] != computed:
                errors.append("protocol sha256 does not match its supplied frozen content")
        if protocol.get("status") != "frozen" and protocol.get("frozen") is not True:
            errors.append("protocol is not frozen")
        expected = scenario_manifest.get("sha256")
        if expected and protocol.get("scenario_manifest_hash") != expected:
            errors.append("protocol scenario_manifest_hash does not match supplied scenario manifest")
        # Older, study-local records carry direct revision groups; the current
        # ProtocolDocument carries a frozen matrix/tolerances/judge-sampling
        # body instead.  Either shape is inspectable here.  The qualification
        # validator below requires the stricter execution bindings for both.
        for field in ("seeds",):
            if field not in protocol:
                errors.append("protocol lacks %s" % field)
        legacy_identity = all(field in protocol for field in ("policy", "world_model", "judge", "thresholds"))
        document_identity = all(field in protocol for field in ("matrix", "tolerances", "judge_sampling"))
        if not legacy_identity and not document_identity:
            errors.append("protocol lacks either direct revision groups or a frozen matrix/tolerances/judge-sampling body")
        preregistration = protocol.get("preregistration")
        remote = protocol.get("remote_record") or protocol.get("preregistration_uri")
        if not remote and isinstance(preregistration, Mapping):
            remote = preregistration.get("uri")
        if not remote:
            errors.append("protocol lacks externally auditable remote record URI")
        return tuple(errors)


@dataclass(frozen=True)
class QualificationDecision:
    capability: CapabilityResult
    required_gates: Tuple[str, ...]
    errors: Tuple[str, ...]


def _string_list(value: Any, label: str, errors: List[str]) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        errors.append("%s must be a non-empty list of asset IDs" % label)
        return ()
    items = tuple(str(item) for item in value)
    if any(not _nonempty_string(item) for item in items):
        errors.append("%s contains an empty asset ID" % label)
    if len(set(items)) != len(items):
        errors.append("%s contains duplicate asset IDs" % label)
    return items


def _qualification_scope_and_assets(protocol: Mapping[str, Any]) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]], List[str]]:
    """Read the explicit full-study asset binding without inventing defaults."""

    errors: List[str] = []
    matrix = protocol.get("matrix")
    if not isinstance(matrix, Mapping):
        return (), {}, ["qualification requires a frozen full-study matrix"]
    from .reference import POLICY_IDS, TASK_IDS

    policies = tuple(str(policy) for policy in matrix.get("policies", ()))
    tasks = tuple(str(task) for task in matrix.get("tasks", ()))
    if policies != tuple(POLICY_IDS):
        errors.append("qualification matrix must retain all six canonical policies in frozen order")
    if tasks != tuple(TASK_IDS):
        errors.append("qualification matrix must retain all five canonical tasks in frozen order")
    if matrix.get("starts_per_task") != 50 or matrix.get("total_planned_episodes") != 1500:
        errors.append("qualification matrix must bind the complete 6 x 5 x 50 (1500 episode) scope")

    declared = protocol.get("qualification_assets")
    if not isinstance(declared, Mapping):
        return policies, {}, errors + ["protocol lacks explicit qualification_assets binding"]
    groups: Dict[str, Tuple[str, ...]] = {}
    for group in ("world_model", "judge", "scenario"):
        groups[group] = _string_list(declared.get(group), "qualification_assets.%s" % group, errors)
    raw_policies = declared.get("policies")
    if not isinstance(raw_policies, Mapping):
        errors.append("qualification_assets.policies must bind every canonical policy")
        groups["policies"] = ()
    else:
        if set(raw_policies) != set(POLICY_IDS):
            errors.append("qualification_assets.policies must contain exactly the six canonical policies")
        policy_assets: List[str] = []
        for policy in POLICY_IDS:
            policy_assets.extend(
                _string_list(raw_policies.get(policy), "qualification_assets.policies.%s" % policy, errors)
            )
        if len(policy_assets) != len(set(policy_assets)):
            errors.append("qualification policy asset bindings reuse an asset ID across policies")
        groups["policies"] = tuple(policy_assets)
    return policies, groups, errors


def _load_qualification_asset_lock(
    protocol: Mapping[str, Any], asset_lock: Any, required_asset_ids: Sequence[str]
) -> Tuple[Any, List[str]]:
    """Load and bind the exact asset lock at the final qualification boundary.

    ``AssetLock.summary`` is a display convenience, so it is never trusted as
    qualification evidence.  The row-level checks and canonical lock hash are
    recomputed by ``AssetLock`` instead.
    """

    errors: List[str] = []
    expected_hash = protocol.get("asset_lock_hash")
    if not _is_sha256(expected_hash):
        errors.append("protocol lacks an immutable asset_lock_hash")
    if asset_lock is None:
        errors.append("qualification requires the asset lock named by protocol.asset_lock_hash")
        return None, errors
    try:
        # Local import avoids the protocol -> gates import cycle at module load.
        from .protocol import AssetLock, canonical_sha256

        lock = asset_lock if isinstance(asset_lock, AssetLock) else AssetLock.from_mapping(asset_lock)
        actual_hash = canonical_sha256(lock.to_mapping())
        if expected_hash and actual_hash != expected_hash:
            errors.append("supplied asset lock hash does not match protocol.asset_lock_hash")
        errors.extend(lock.qualification_errors(required_asset_ids))
        return lock, errors
    except (TypeError, ValueError, KeyError) as exc:
        errors.append("qualification asset lock is invalid: %s" % exc)
        return None, errors


def _profile_errors(
    profile: Any,
    *,
    label: str,
    asset_ids: Sequence[str],
    asset_hashes: Optional[Mapping[str, str]] = None,
    required_keys: Sequence[str],
    require_gpu: bool = False,
) -> List[str]:
    """Validate a measured executable profile, not a caller-supplied flag."""

    errors: List[str] = []
    if not isinstance(profile, Mapping):
        return ["%s must be a structured measured profile" % label]
    for key in required_keys:
        if not _nonempty_string(profile.get(key)):
            errors.append("%s lacks %s" % (label, key))
    for key in ("model_asset_id", "code_asset_id"):
        value = profile.get(key)
        if not _nonempty_string(value):
            errors.append("%s lacks %s" % (label, key))
        elif value not in set(asset_ids):
            errors.append("%s %s is not in the protocol-bound qualification asset set" % (label, key))
        elif asset_hashes is not None:
            expected_hash = asset_hashes.get(str(value))
            observed_hash = profile.get(key.replace("_id", "_sha256"))
            if not _is_sha256(observed_hash) or observed_hash != expected_hash:
                errors.append("%s %s does not bind the locked asset SHA-256" % (label, key))
    container = profile.get("container_digest")
    if not _is_sha256(container):
        errors.append("%s container_digest must be a sha256 digest" % label)
    if require_gpu:
        gpu = profile.get("gpu_profile")
        if not isinstance(gpu, Mapping):
            errors.append("%s lacks a measured gpu_profile" % label)
        else:
            for key in ("gpu_model", "device_uuid", "driver_version", "runtime_version", "measured_at", "evidence_uri"):
                if not _nonempty_string(gpu.get(key)):
                    errors.append("%s.gpu_profile lacks %s" % (label, key))
            if not _finite_positive(gpu.get("peak_memory_bytes")):
                errors.append("%s.gpu_profile peak_memory_bytes must be a measured positive number" % label)
    return errors


def _strict_gate_evidence_errors(
    ledger: GateLedger,
    protocol: Mapping[str, Any],
    scenario_manifest: Mapping[str, Any],
    *,
    task: str,
    asset_lock: Any,
) -> List[str]:
    """Evidence that is required *in addition to* a PASS enum.

    This routine is intentionally at the qualification boundary rather than
    ``GateRecord.from_mapping``: a ledger remains able to preserve incomplete,
    diagnostic, and ``pass_with_limitations`` artifacts without relabelling
    them as qualified.  A result is qualified only when every reference below
    can be followed to the frozen asset/profile/evidence record.
    """

    errors: List[str] = []
    policies, asset_groups, scope_errors = _qualification_scope_and_assets(protocol)
    errors.extend(scope_errors)
    all_assets = tuple(asset for group in asset_groups.values() for asset in group)
    lock, lock_errors = _load_qualification_asset_lock(protocol, asset_lock, all_assets)
    asset_hashes = (
        {asset_id: str(record.sha256) for asset_id, record in lock.assets.items()}
        if lock is not None
        else {}
    )
    errors.extend(lock_errors)

    # All five panels must be real, frozen, and fully enumerated.  Checking
    # only the task passed to this call would make a one-task convenient subset
    # look like the preregistered six-by-five study.
    errors.extend(
        "full-scope scenario: " + message
        for message in ScenarioManifestValidator.validate(scenario_manifest, require_primary_panel=True)
    )
    scenario_hash = scenario_manifest.get("sha256")
    if _is_sha256(scenario_hash):
        computed_scenario_hash = canonical_json_sha256(
            {key: value for key, value in scenario_manifest.items() if key != "sha256"}
        )
        if scenario_hash != computed_scenario_hash:
            errors.append("full-scope scenario: sha256 does not match supplied start content")

    gate_a = ledger.records["A"]
    backend_profile = gate_a.measurements.get("backend_profile")
    errors.extend(
        "Gate A: " + message
        for message in _profile_errors(
            backend_profile,
            label="backend_profile",
            asset_ids=asset_groups.get("world_model", ()),
            asset_hashes=asset_hashes,
            required_keys=("profile_id", "model_revision", "code_revision", "normalizer_revision"),
            require_gpu=True,
        )
    )
    if isinstance(backend_profile, Mapping):
        gpu = backend_profile.get("gpu_profile")
        if isinstance(gpu, Mapping) and gate_a.measurements.get("gpu_peak_memory_bytes") != gpu.get("peak_memory_bytes"):
            errors.append("Gate A: gpu_peak_memory_bytes must equal the backend profile's measured GPU peak")
    if not _finite_positive(gate_a.measurements.get("wall_seconds")):
        errors.append("Gate A: wall_seconds must be an actual positive measurement")
    if not isinstance(gate_a.measurements.get("backend_calls"), int) or isinstance(
        gate_a.measurements.get("backend_calls"), bool
    ) or gate_a.measurements.get("backend_calls", 0) < 1:
        errors.append("Gate A: backend_calls must count at least one actual invocation")

    gate_b = ledger.records["B"]
    feedback = gate_b.measurements.get("feedback_certification")
    if not isinstance(feedback, Mapping):
        errors.append("Gate B: needs a structured feedback_certification, not a feedback_mode flag")
    else:
        if feedback.get("feedback_mode") != FeedbackMode.NATIVE_FEEDBACK.value:
            errors.append("Gate B: feedback_certification must explicitly be native_feedback")
        profile_id = backend_profile.get("profile_id") if isinstance(backend_profile, Mapping) else None
        if not _nonempty_string(feedback.get("world_profile_id")) or feedback.get("world_profile_id") != profile_id:
            errors.append("Gate B: feedback certification must bind the Gate-A backend profile")
        if feedback.get("terminal_padding") is not False:
            errors.append("Gate B: terminal padding is an alternate protocol, not native qualification evidence")
        if not _is_sha256(feedback.get("evidence_hash")) or not _nonempty_string(feedback.get("evidence_uri")):
            errors.append("Gate B: feedback certification needs immutable evidence_hash and evidence_uri")
        prefixes = feedback.get("policy_executed_prefixes")
        if not isinstance(prefixes, Mapping) or set(prefixes) != set(policies):
            errors.append("Gate B: must record a certified executed prefix for every policy in the frozen scope")
        else:
            for policy in policies:
                prefix = prefixes.get(policy)
                if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 1:
                    errors.append("Gate B: %s lacks a positive certified executed prefix" % policy)
            if prefixes.get("OpenVLA") != 1:
                errors.append("Gate B: OpenVLA native feedback requires exactly one fresh action per generated frame")
            policy_profiles = feedback.get("policy_profiles")
            if not isinstance(policy_profiles, Mapping) or set(policy_profiles) != set(policies):
                errors.append("Gate B: needs a measured executable policy profile for every frozen policy")
            else:
                raw_policy_assets = protocol.get("qualification_assets", {}).get("policies", {})
                for policy in policies:
                    profile = policy_profiles.get(policy)
                    errors.extend(
                        "Gate B: " + message
                        for message in _profile_errors(
                            profile,
                            label="policy_profiles.%s" % policy,
                            asset_ids=raw_policy_assets.get(policy, ()) if isinstance(raw_policy_assets, Mapping) else (),
                            asset_hashes=asset_hashes,
                            required_keys=("profile_id", "model_revision", "code_revision", "normalizer_revision"),
                        )
                    )
                    if isinstance(profile, Mapping):
                        if profile.get("executed_prefix_ticks") != prefixes.get(policy):
                            errors.append("Gate B: %s profile prefix does not match its feedback certificate" % policy)
                        terminal = profile.get("certified_terminal_prefixes")
                        if not isinstance(terminal, Mapping) or set(terminal) != set(EXACT_TASK_PROMPTS):
                            errors.append("Gate B: %s profile lacks terminal support for every frozen task" % policy)
                        else:
                            for task_name, terminal_prefix in terminal.items():
                                if (
                                    isinstance(terminal_prefix, bool)
                                    or not isinstance(terminal_prefix, int)
                                    or terminal_prefix < 1
                                    or terminal_prefix > prefixes.get(policy, 0)
                                ):
                                    errors.append(
                                        "Gate B: %s terminal support for %s is not a certified native prefix"
                                        % (policy, task_name)
                                    )
                    revision = gate_b.policy_revisions.get(policy)
                    if not isinstance(revision, Mapping) or not isinstance(profile, Mapping) or revision.get("profile_id") != profile.get("profile_id"):
                        errors.append("Gate B: %s policy revision does not bind its measured executable profile" % policy)
    suffix = gate_b.measurements.get("suffix_causality")
    if not isinstance(suffix, Mapping):
        errors.append("Gate B: suffix_causality must be a structured paired result")
    else:
        if suffix.get("status") != "pass":
            errors.append("Gate B: suffix_causality has not passed")
        if suffix.get("first_action_fixed_against_baseline") is not True:
            errors.append("Gate B: suffix evidence must hold the first action fixed")
        if suffix.get("future_suffix_changed_first_generated_frame") is not False:
            errors.append("Gate B: future suffix sensitivity blocks native feedback qualification")
        if not _is_sha256(suffix.get("evidence_hash")) or not _nonempty_string(suffix.get("evidence_uri")):
            errors.append("Gate B: suffix evidence needs immutable evidence_hash and evidence_uri")
    if any("padding" in str(value).lower() for value in (gate_b.measurements.get("protocol_identity"), gate_b.measurements.get("feedback_mode"))):
        errors.append("Gate B: a padded protocol identity may not populate a native qualified cell")

    gate_c = ledger.records["C"]
    per_task = gate_c.measurements.get("starts_per_task")
    comparability = gate_c.measurements.get("comparability")
    for label, value in (("starts_per_task", per_task), ("comparability", comparability)):
        if not isinstance(value, Mapping):
            errors.append("Gate C: %s must cover every frozen task" % label)
            continue
        missing = [name for name in EXACT_TASK_PROMPTS if name not in value]
        if missing:
            errors.append("Gate C: %s is missing %s" % (label, ", ".join(missing)))
    if isinstance(per_task, Mapping):
        for name in EXACT_TASK_PROMPTS:
            count = per_task.get(name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 50:
                errors.append("Gate C: %s needs at least 50 measured starts" % name)
    if isinstance(comparability, Mapping):
        for name in EXACT_TASK_PROMPTS:
            result = comparability.get(name)
            if not isinstance(result, Mapping) or result.get("status") != "pass":
                errors.append("Gate C: %s lacks a passed real-panel comparability record" % name)

    gate_d = ledger.records["D"]
    measurements = gate_d.measurements
    if measurements.get("calibration_class") != "human":
        errors.append("Gate D: model/external labels cannot substitute for the required human calibration")
    if measurements.get("has_two_blinded_humans") is not True or measurements.get("human_annotator_count") != 2:
        errors.append("Gate D: needs declarations of exactly two blinded human annotators")
    if measurements.get("gate_d_decision_status") != "pass" or measurements.get("thresholds_satisfied") is not True:
        errors.append("Gate D: pass_with_limitations or a threshold flag alone is not a passed human calibration")
    human_evidence = measurements.get("human_annotation_evidence")
    if not isinstance(human_evidence, Mapping):
        errors.append("Gate D: needs a held-out human annotation evidence manifest")
    else:
        ids = human_evidence.get("annotator_ids")
        if not isinstance(ids, (list, tuple)) or len(ids) != 2 or len({str(item) for item in ids}) != 2:
            errors.append("Gate D: annotation manifest must declare two distinct annotator IDs")
        if human_evidence.get("heldout_clip_count") != 50 or human_evidence.get("double_labeled_clip_count") != 50:
            errors.append("Gate D: annotation manifest must bind all 50 double-labelled held-out clips")
        if not _is_sha256(human_evidence.get("manifest_hash")) or not _nonempty_string(human_evidence.get("evidence_uri")):
            errors.append("Gate D: annotation manifest needs immutable manifest_hash and evidence_uri")
    judge_profile = measurements.get("judge_profile")
    errors.extend(
        "Gate D: " + message
        for message in _profile_errors(
            judge_profile,
            label="judge_profile",
            asset_ids=asset_groups.get("judge", ()),
            asset_hashes=asset_hashes,
            required_keys=("profile_id", "model_revision", "code_revision", "normalizer_revision", "runtime_lock_hash"),
        )
    )
    tolerance_block = protocol.get("tolerances")
    gate_d_tolerances = tolerance_block.get("gate_d") if isinstance(tolerance_block, Mapping) else None
    if not isinstance(gate_d_tolerances, Mapping):
        errors.append("Gate D: frozen protocol lacks gate_d tolerances")
    else:
        comparisons = (
            ("binary_kappa", "min_binary_kappa", lambda actual, limit: actual >= limit),
            ("weighted_progress_kappa", "min_weighted_progress_kappa", lambda actual, limit: actual >= limit),
            ("leniency_offset", "max_leniency_offset", lambda actual, limit: actual <= limit),
            ("consensus_coverage", "min_consensus_coverage", lambda actual, limit: actual >= limit),
        )
        for observed, threshold, predicate in comparisons:
            actual = measurements.get(observed)
            limit = gate_d_tolerances.get(threshold)
            if not _finite_positive(actual) and not (observed == "leniency_offset" and isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(float(actual))):
                errors.append("Gate D: %s must be a finite measured number" % observed)
            elif not isinstance(limit, (int, float)) or isinstance(limit, bool) or not predicate(float(actual), float(limit)):
                errors.append("Gate D: %s does not satisfy frozen %s" % (observed, threshold))
    # These fields establish a provenance chain, not the real-world identity of
    # an annotator.  Human identity remains an operational review obligation.
    return errors


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
        asset_lock: Optional[Any] = None,
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
        protocol_hash = protocol.get("sha256") or protocol.get("protocol_sha256")
        for gate_id in required:
            record = ledger.records[gate_id]
            if record.status is not GateStatus.PASS:
                errors.append("Gate %s is %s, not pass" % (gate_id, record.status.value))
                continue
            if record.protocol_hash != protocol_hash:
                errors.append("Gate %s protocol hash does not match frozen protocol" % gate_id)
            errors.extend("Gate %s: %s" % (gate_id, message) for message in record.pass_evidence_errors())
        # A PASS enum means evidence was recorded; it is not an assertion that
        # the recorded bytes, profiles and calibration support a public cell.
        # Run this even when a gate is blocked so callers get the complete
        # actionable evidence gap rather than a misleading first-error result.
        errors.extend(
            _strict_gate_evidence_errors(
                ledger,
                protocol,
                scenario_manifest,
                task=task,
                asset_lock=asset_lock,
            )
        )
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
