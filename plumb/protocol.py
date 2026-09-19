"""Frozen evaluation protocol, asset lock, and preregistration records.

Spec section 6 requires the scenario/policy matrix, split membership, RNG seeds,
judge identity, endpoints, thresholds, exclusion policy, cost-selection rule and
analysis-code revision to be frozen *before* primary evaluation, and it is
explicit that a local content hash is not preregistration:

    "Store ``protocol.json``, its SHA-256 and the analysis-code revision in an
    append-only project-owned remote artifact store with an externally auditable
    timestamp ... A local hash alone does not supply independent timing
    evidence."

So this module separates three things that are easy to conflate:

``ProtocolDocument``
    The frozen content plus its canonical SHA-256.  Local, reproducible, and by
    itself **not** preregistration.
``PreregistrationRecord``
    A pointer to an external, independently timestamped artifact (a signed git
    tag pushed to a remote is the supported form).  ``external_timestamp`` is
    the remote's timestamp, never this machine's clock.  Absent evidence is
    ``None`` and reports ``status="unregistered"`` rather than borrowing the
    document hash as proof of timing.
``AssetLock``
    Immutable asset/code/runtime/access records (spec section 1).  A row with an
    unknown hash, license or container digest stays ``verified=False``; the lock
    refuses to report ``verified`` status for an incomplete row.

Amendments receive a new protocol ID and never overwrite an older document.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .gates import EXACT_TASK_PROMPTS
from .reference import POLICY_IDS, TASK_IDS


PROTOCOL_SCHEMA_VERSION = 1
ASSET_LOCK_SCHEMA_VERSION = 1

#: Task horizons in Bridge control ticks (spec section 3).  These are exact
#: terminal horizons, not suggestions; a shorter run is a different protocol.
TASK_HORIZONS: Dict[str, int] = {
    "close_drawer": 70,
    "open_drawer": 70,
    "to_basket": 100,
    "to_sink": 100,
    "fold_cloth": 80,
}

#: Analysis families and their Bonferroni denominators (spec section 6).  The
#: two families are reported separately and do **not** compose into a joint
#: guarantee across both.
ALPHA_FAMILIES: Dict[str, Dict[str, Any]] = {
    "per_task_pairwise": {"alpha": 0.05 / 75, "members": 75, "description": "15 policy pairs x 5 tasks"},
    "macro_pairwise": {"alpha": 0.05 / 15, "members": 15, "description": "15 policy pairs, equal-task macro"},
    "cell_endpoint_envelopes": {"alpha": 0.05 / 60, "members": 60, "description": "30 cells x lower/upper endpoint"},
}

#: Judge sampling is frozen together with its quorum, schema and cost model
#: (spec section 5).  Changing the sample count requires a new protocol ID.
FROZEN_JUDGE_SAMPLING: Dict[str, Any] = {
    "sample_count": 5,
    "quorum": 3,
    "temperature": 0.7,
    "top_p": 1.0,
    "max_new_tokens": 512,
    "retries_per_sample": 1,
    "retry_eligible": ["schema_error", "transient_transport_error"],
    "progress_aggregation": "lower_median_of_agreeing_decisive_samples",
}

#: Opt-in analysis switches.  ``plumb/measurement.py`` deliberately returns
#: ``not_computed`` for these unless the caller asks, so that an exploratory
#: dashboard request cannot silently publish a headline statistic.  A frozen
#: protocol is what authorises them.
ANALYSIS_OPT_INS: Tuple[str, ...] = (
    "reliability",
    "mdd",
    "mmrv",
    "advanced_inference",
    "cost_fidelity",
    "drift",
)

_SHA256_RE = re.compile(r"^(sha256:)?[0-9a-f]{64}$")
_OFFSET_TIMESTAMP_RE = re.compile(r"[+-]\d{2}:?\d{2}$|Z$")


class ProtocolError(ValueError):
    """A protocol document, asset lock, or preregistration record is invalid."""


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Deterministic JSON encoding used for every hash in this module."""

    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git(args: Sequence[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Run a read-only git command, returning ``None`` when git is unavailable."""

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerances:
    """Preregistered numerical thresholds.

    Every field must be set explicitly.  Spec section 0 requires tolerances to
    be defined from independent development fixtures and frozen *before*
    held-out gate evaluation, so there are no defaults that could be quietly
    relaxed after seeing a result.
    """

    # Gate A - backend conformance
    frame_count_exact: bool
    max_nonfinite_fraction: float
    # Gate B - action and feedback fidelity
    min_intervention_effect_mae: float
    max_suffix_invariance_mae: float
    repeat_determinism_max_mae: float
    # Gate C - scenario parity
    min_starts_per_task: int
    min_distinct_lineages_per_task: int
    # Gate D - judge calibration
    min_binary_kappa: float
    min_weighted_progress_kappa: float
    max_leniency_offset: float
    min_consensus_coverage: float
    # Gate E - measurement confirmation
    max_cell_rate_error: float
    min_coverage: float
    min_pairwise_order_agreement: float
    # Gate F - burst qualification
    max_burst_seconds: float
    max_total_demonstration_usd: float
    required_rehearsals: int
    # Power
    power_target: float
    mdd_simulations_per_gap: int
    bootstrap_replicates: int
    split_half_repetitions: int

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "gate_a": {
                "frame_count_exact": self.frame_count_exact,
                "max_nonfinite_fraction": self.max_nonfinite_fraction,
            },
            "gate_b": {
                "min_intervention_effect_mae": self.min_intervention_effect_mae,
                "max_suffix_invariance_mae": self.max_suffix_invariance_mae,
                "repeat_determinism_max_mae": self.repeat_determinism_max_mae,
            },
            "gate_c": {
                "min_starts_per_task": self.min_starts_per_task,
                "min_distinct_lineages_per_task": self.min_distinct_lineages_per_task,
            },
            "gate_d": {
                "min_binary_kappa": self.min_binary_kappa,
                "min_weighted_progress_kappa": self.min_weighted_progress_kappa,
                "max_leniency_offset": self.max_leniency_offset,
                "min_consensus_coverage": self.min_consensus_coverage,
            },
            "gate_e": {
                "max_cell_rate_error": self.max_cell_rate_error,
                "min_coverage": self.min_coverage,
                "min_pairwise_order_agreement": self.min_pairwise_order_agreement,
            },
            "gate_f": {
                "max_burst_seconds": self.max_burst_seconds,
                "max_total_demonstration_usd": self.max_total_demonstration_usd,
                "required_rehearsals": self.required_rehearsals,
            },
            "power": {
                "power_target": self.power_target,
                "mdd_simulations_per_gap": self.mdd_simulations_per_gap,
                "bootstrap_replicates": self.bootstrap_replicates,
                "split_half_repetitions": self.split_half_repetitions,
            },
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Tolerances":
        try:
            gate_a = payload["gate_a"]
            gate_b = payload["gate_b"]
            gate_c = payload["gate_c"]
            gate_d = payload["gate_d"]
            gate_e = payload["gate_e"]
            gate_f = payload["gate_f"]
            power = payload["power"]
            return cls(
                frame_count_exact=bool(gate_a["frame_count_exact"]),
                max_nonfinite_fraction=float(gate_a["max_nonfinite_fraction"]),
                min_intervention_effect_mae=float(gate_b["min_intervention_effect_mae"]),
                max_suffix_invariance_mae=float(gate_b["max_suffix_invariance_mae"]),
                repeat_determinism_max_mae=float(gate_b["repeat_determinism_max_mae"]),
                min_starts_per_task=int(gate_c["min_starts_per_task"]),
                min_distinct_lineages_per_task=int(gate_c["min_distinct_lineages_per_task"]),
                min_binary_kappa=float(gate_d["min_binary_kappa"]),
                min_weighted_progress_kappa=float(gate_d["min_weighted_progress_kappa"]),
                max_leniency_offset=float(gate_d["max_leniency_offset"]),
                min_consensus_coverage=float(gate_d["min_consensus_coverage"]),
                max_cell_rate_error=float(gate_e["max_cell_rate_error"]),
                min_coverage=float(gate_e["min_coverage"]),
                min_pairwise_order_agreement=float(gate_e["min_pairwise_order_agreement"]),
                max_burst_seconds=float(gate_f["max_burst_seconds"]),
                max_total_demonstration_usd=float(gate_f["max_total_demonstration_usd"]),
                required_rehearsals=int(gate_f["required_rehearsals"]),
                power_target=float(power["power_target"]),
                mdd_simulations_per_gap=int(power["mdd_simulations_per_gap"]),
                bootstrap_replicates=int(power["bootstrap_replicates"]),
                split_half_repetitions=int(power["split_half_repetitions"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("tolerances payload is incomplete: %s" % exc) from exc


def default_tolerances() -> Tolerances:
    """The v1 candidate tolerance set.

    These are the values the project intends to freeze.  They are derived from
    the spec's own stated numbers where it gives them (80% power, >=10,000
    simulations per gap, 10,000 bootstrap replicates, 1,000 split-half
    repetitions, 60 seconds, $11.25, three rehearsals, 50 starts per task) and
    are otherwise explicit development-fixture choices.  They are *candidates*
    until a ``PreregistrationRecord`` with an external timestamp exists; see
    ``ProtocolDocument.preregistration_status``.
    """

    return Tolerances(
        frame_count_exact=True,
        max_nonfinite_fraction=0.0,
        min_intervention_effect_mae=0.5,
        max_suffix_invariance_mae=0.25,
        repeat_determinism_max_mae=0.0,
        min_starts_per_task=50,
        min_distinct_lineages_per_task=50,
        min_binary_kappa=0.6,
        min_weighted_progress_kappa=0.6,
        max_leniency_offset=0.15,
        min_consensus_coverage=0.8,
        max_cell_rate_error=0.1,
        min_coverage=0.8,
        min_pairwise_order_agreement=0.9,
        max_burst_seconds=60.0,
        # Spec section 7 fixes this as the operational definition of
        # "approximately $11".
        max_total_demonstration_usd=11.25,
        required_rehearsals=3,
        power_target=0.8,
        mdd_simulations_per_gap=10_000,
        bootstrap_replicates=10_000,
        split_half_repetitions=1_000,
    )


# ---------------------------------------------------------------------------
# Preregistration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreregistrationRecord:
    """A pointer to externally timestamped evidence that a protocol was frozen.

    ``external_timestamp`` must come from the remote artifact store, not from
    this process.  ``uri`` must be a remote URI; a local path is rejected
    because spec section 6 says a local hash supplies no independent timing
    evidence.
    """

    protocol_sha256: str
    uri: Optional[str] = None
    kind: str = "signed_git_tag"
    tag: Optional[str] = None
    commit: Optional[str] = None
    signed: bool = False
    external_timestamp: Optional[str] = None
    verified_at: Optional[str] = None
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SHA256_RE.match(self.protocol_sha256 or ""):
            raise ProtocolError("protocol_sha256 must be a sha256 digest")
        if self.uri is not None and not str(self.uri).startswith(("https://", "ssh://", "git@")):
            raise ProtocolError(
                "preregistration uri must be a remote artifact URI; a local path is not "
                "independent timing evidence"
            )
        if self.external_timestamp is not None and not _OFFSET_TIMESTAMP_RE.search(self.external_timestamp):
            raise ProtocolError("external_timestamp must carry an explicit UTC offset or Z")

    @property
    def status(self) -> str:
        """``registered`` only when remote, externally timestamped evidence exists."""

        if self.uri and self.external_timestamp and self.commit:
            return "registered" if self.signed else "registered_unsigned"
        return "unregistered"

    def blocking_reasons(self) -> Tuple[str, ...]:
        reasons: List[str] = []
        if not self.uri:
            reasons.append("no remote preregistration URI")
        if not self.commit:
            reasons.append("no commit bound to the protocol hash")
        if not self.external_timestamp:
            reasons.append("no externally auditable timestamp")
        if not self.signed:
            reasons.append("preregistration artifact is not signed")
        return tuple(reasons)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "uri": self.uri,
            "kind": self.kind,
            "tag": self.tag,
            "commit": self.commit,
            "signed": self.signed,
            "external_timestamp": self.external_timestamp,
            "verified_at": self.verified_at,
            "status": self.status,
            "blocking_reasons": list(self.blocking_reasons()),
            "notes": list(self.notes),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PreregistrationRecord":
        return cls(
            protocol_sha256=str(payload.get("protocol_sha256", "")),
            uri=payload.get("uri"),
            kind=str(payload.get("kind", "signed_git_tag")),
            tag=payload.get("tag"),
            commit=payload.get("commit"),
            signed=bool(payload.get("signed", False)),
            external_timestamp=payload.get("external_timestamp"),
            verified_at=payload.get("verified_at"),
            notes=tuple(str(item) for item in payload.get("notes", ())),
        )

    @classmethod
    def unregistered(cls, protocol_sha256: str, note: str) -> "PreregistrationRecord":
        return cls(protocol_sha256=protocol_sha256, notes=(note,))


def discover_preregistration(
    protocol_sha256: str,
    tag: str,
    repo_root: Optional[Path] = None,
) -> PreregistrationRecord:
    """Read a signed annotated tag as preregistration evidence.

    The tag must (a) exist, (b) contain the protocol hash in its message, and
    (c) be present on a remote.  A tag that exists only locally is *not*
    preregistration, because nothing outside this machine witnessed its time.
    """

    root = repo_root or Path(__file__).resolve().parents[1]
    notes: List[str] = []

    message = _git(["tag", "-l", tag, "--format=%(contents)"], cwd=root)
    if not message:
        return PreregistrationRecord.unregistered(protocol_sha256, "tag %s does not exist locally" % tag)

    bare = protocol_sha256.split(":", 1)[-1]
    if bare not in message:
        return PreregistrationRecord.unregistered(
            protocol_sha256, "tag %s does not contain the current protocol hash" % tag
        )

    commit = _git(["rev-list", "-n", "1", tag], cwd=root)
    signature = _git(["tag", "-l", tag, "--format=%(contents:signature)"], cwd=root)
    signed = bool(signature and "PGP SIGNATURE" in signature)
    if not signed:
        notes.append("tag exists but carries no PGP signature")

    remote_listing = _git(["ls-remote", "--tags", "origin", "refs/tags/" + tag], cwd=root)
    if not remote_listing:
        return PreregistrationRecord(
            protocol_sha256=protocol_sha256,
            tag=tag,
            commit=commit,
            signed=signed,
            notes=tuple(notes + ["tag is not present on the origin remote; no external witness"]),
        )

    remote_url = _git(["remote", "get-url", "origin"], cwd=root)
    uri = None
    if remote_url and remote_url.startswith(("https://", "ssh://", "git@")):
        uri = "%s#refs/tags/%s" % (remote_url, tag)

    # ``%(taggerdate:iso-strict)`` carries an explicit offset.  It is the
    # tagger's clock rather than the remote's receipt time; record it as the
    # best available external anchor and say so.
    tagger_date = _git(["tag", "-l", tag, "--format=%(taggerdate:iso-strict)"], cwd=root)
    if tagger_date:
        notes.append("external_timestamp is the annotated tag date as published to origin")

    return PreregistrationRecord(
        protocol_sha256=protocol_sha256,
        uri=uri,
        tag=tag,
        commit=commit,
        signed=signed,
        external_timestamp=tagger_date,
        verified_at=datetime.now(timezone.utc).isoformat(),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Asset lock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetRecord:
    """One immutable asset row (spec section 1, "Asset manifest and loading").

    ``verified`` is computed, never supplied: an unknown hash, license or
    container digest keeps the row unverified.  Spec section 1: "Use actual
    downloaded hashes; unknown values stay null and block a ``verified``
    status."
    """

    asset_id: str
    repo_id: str
    repo_type: str
    revision: Optional[str] = None
    file_path: Optional[str] = None
    byte_length: Optional[int] = None
    #: A size quoted in a spec table or model card. A discovery aid, never
    #: evidence: ``byte_length`` means bytes we actually retrieved and verified.
    historical_size_estimate: Optional[int] = None
    sha256: Optional[str] = None
    source_url: Optional[str] = None
    retrieved_at: Optional[str] = None
    license: Optional[str] = None
    access_status: str = "unresolved"
    redistribution: str = "unresolved"
    notices: Tuple[str, ...] = ()
    loader_revision: Optional[str] = None
    container_digest: Optional[str] = None
    compatibility_profile_id: Optional[str] = None
    role: str = "unknown"

    @property
    def verified(self) -> bool:
        return bool(
            self.revision
            and self.sha256
            and _SHA256_RE.match(self.sha256)
            and self.byte_length is not None
            and self.retrieved_at
            and self.license is not None
            and self.access_status == "resolved"
        )

    def unresolved_fields(self) -> Tuple[str, ...]:
        missing: List[str] = []
        for name in ("revision", "sha256", "byte_length", "retrieved_at", "license"):
            if getattr(self, name) in (None, ""):
                missing.append(name)
        if self.access_status != "resolved":
            missing.append("access_status")
        if self.sha256 and not _SHA256_RE.match(self.sha256):
            missing.append("sha256_malformed")
        return tuple(missing)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "file_path": self.file_path,
            "byte_length": self.byte_length,
            "historical_size_estimate": self.historical_size_estimate,
            "sha256": self.sha256,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "license": self.license,
            "access_status": self.access_status,
            "redistribution": self.redistribution,
            "notices": list(self.notices),
            "loader_revision": self.loader_revision,
            "container_digest": self.container_digest,
            "compatibility_profile_id": self.compatibility_profile_id,
            "role": self.role,
            "verified": self.verified,
            "unresolved_fields": list(self.unresolved_fields()),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AssetRecord":
        return cls(
            asset_id=str(payload["asset_id"]),
            repo_id=str(payload["repo_id"]),
            repo_type=str(payload.get("repo_type", "model")),
            revision=payload.get("revision"),
            file_path=payload.get("file_path"),
            byte_length=payload.get("byte_length"),
            historical_size_estimate=payload.get("historical_size_estimate"),
            sha256=payload.get("sha256"),
            source_url=payload.get("source_url"),
            retrieved_at=payload.get("retrieved_at"),
            license=payload.get("license"),
            access_status=str(payload.get("access_status", "unresolved")),
            redistribution=str(payload.get("redistribution", "unresolved")),
            notices=tuple(str(item) for item in payload.get("notices", ())),
            loader_revision=payload.get("loader_revision"),
            container_digest=payload.get("container_digest"),
            compatibility_profile_id=payload.get("compatibility_profile_id"),
            role=str(payload.get("role", "unknown")),
        )


@dataclass
class AssetLock:
    """``assets.lock.json`` — the pre-Gate-A asset record."""

    assets: Dict[str, AssetRecord] = field(default_factory=dict)
    schema_version: int = ASSET_LOCK_SCHEMA_VERSION

    def add(self, record: AssetRecord) -> None:
        self.assets[record.asset_id] = record

    @property
    def verified_asset_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(key for key, value in self.assets.items() if value.verified))

    @property
    def unresolved_asset_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(key for key, value in self.assets.items() if not value.verified))

    def to_mapping(self) -> Dict[str, Any]:
        body = {
            "schema_version": self.schema_version,
            "assets": {key: self.assets[key].to_mapping() for key in sorted(self.assets)},
        }
        body["summary"] = {
            "total": len(self.assets),
            "verified": len(self.verified_asset_ids),
            "unresolved": len(self.unresolved_asset_ids),
            "unresolved_asset_ids": list(self.unresolved_asset_ids),
            "status": "verified" if self.assets and not self.unresolved_asset_ids else "incomplete",
        }
        return body

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AssetLock":
        raw = payload.get("assets", {})
        if not isinstance(raw, Mapping):
            raise ProtocolError("assets.lock.json assets must be a mapping")
        lock = cls(schema_version=int(payload.get("schema_version", ASSET_LOCK_SCHEMA_VERSION)))
        for key, item in raw.items():
            record = AssetRecord.from_mapping({**item, "asset_id": item.get("asset_id", key)})
            lock.add(record)
        return lock

    @classmethod
    def load(cls, path: Path) -> "AssetLock":
        source = Path(path)
        if not source.is_file():
            return cls()
        with source.open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def save(self, path: Path) -> str:
        payload = self.to_mapping()
        _atomic_write_json(Path(path), payload)
        return canonical_sha256(payload)


def asset_plan_records() -> Tuple[AssetRecord, ...]:
    """Discovery-stage asset rows for every asset spec section 1 names.

    Byte lengths and revisions that appear in the spec are *discovery aids*, not
    proof of a local download, so they are recorded as ``byte_length`` hints
    with ``sha256=None`` and ``access_status="unresolved"``.  Every row is
    therefore unverified until an actual retrieval fills it in.  ``cluster``
    tooling owns the retrieval; this function owns the shape.
    """

    def record(**kwargs: Any) -> AssetRecord:
        return AssetRecord(**kwargs)

    return (
        record(
            asset_id="world.cosmos3_nano",
            repo_id="nvidia/Cosmos3-Nano",
            repo_type="model",
            revision="e59a53c25979a090fa8706c9acc0c254a6e89b92",
            license="OpenMDW-1.1",
            role="world_model",
            notices=("guardrail model is separately gated and deliberately not fetched",),
        ),
        record(
            asset_id="world.cosmos3_edge",
            repo_id="nvidia/Cosmos3-Edge",
            repo_type="model",
            license="OpenMDW-1.1",
            role="world_model_speed_arm",
        ),
        record(
            asset_id="policy.openvla",
            repo_id="openvla/openvla-7b",
            repo_type="model",
            revision="47a0ec7fc4ec123775a391911046cf33cf9ed83f",
            historical_size_estimate=15_085_000_000,
            role="policy",
            notices=("reviewed trust_remote_code opt-in is mandatory",),
        ),
        record(
            asset_id="policy.octo_small",
            repo_id="rail-berkeley/octo-small",
            repo_type="model",
            file_path="270000/default/checkpoint",
            historical_size_estimate=547_000_000,
            role="policy",
            notices=(
                "checkpoint is NOT at repo root; resolve/main/checkpoint returns 404",
                "must be v1.0; do not use octo-small-1.5",
            ),
        ),
        record(
            asset_id="policy.octo_base",
            repo_id="rail-berkeley/octo-base",
            repo_type="model",
            file_path="300000/default/checkpoint",
            historical_size_estimate=811_000_000,
            role="diagnostic_policy",
            notices=("separate diagnostic, not a seventh benchmark policy",),
        ),
        record(
            asset_id="policy.minivla",
            repo_id="Stanford-ILIAD/minivla-vq-bridge-prismatic",
            repo_type="model",
            file_path="checkpoints/step-362500-epoch-21-loss%3D0.2259.pt",
            historical_size_estimate=5_550_000_000,
            role="policy",
            notices=("filename contains '='; URL-encode as %3D", "skip the 76 MB training-log .jsonl"),
        ),
        record(
            asset_id="policy.minivla_vq",
            repo_id="Stanford-ILIAD/pretrain_vq",
            repo_type="model",
            file_path=(
                "pretrain_modvq+mx-bridge_dataset+fach-7+ng-7+nemb-256+nlatent-512/checkpoints/model.pt"
            ),
            historical_size_estimate=9_500_000,
            license=None,
            redistribution="prohibited_pending_resolution",
            role="policy_component",
            notices=(
                "repo declares no license (cardData: null)",
                "do not redistribute; resolve access terms before use",
            ),
        ),
        record(
            asset_id="policy.open_pi_zero",
            repo_id="allenzren/open-pi-zero",
            repo_type="model",
            file_path="bridge_beta_step19296_2024-12-26_22-30_42.pt",
            historical_size_estimate=11_773_000_000,
            role="policy",
            notices=("verify full state-dict coverage in the pinned loader",),
        ),
        record(
            asset_id="policy.paligemma_support",
            repo_id="google/paligemma-3b-pt-224",
            repo_type="model",
            historical_size_estimate=21_900_000,
            access_status="unresolved",
            role="policy_component",
            notices=(
                "prefer the official repo under accepted terms",
                "leo009 mirror is a candidate only after provenance/hash/terms comparison is saved",
                "Gemma license governs the weights",
            ),
        ),
        record(
            asset_id="policy.susie_subgoal",
            repo_id="kvablack/susie",
            repo_type="model",
            historical_size_estimate=3_438_000_000,
            role="policy_component",
            notices=("needs a pinned JAX/Flax Stable Diffusion stack",),
        ),
        record(
            asset_id="policy.susie_lowlevel",
            repo_id="patreya/gcbc-bridge",
            repo_type="model",
            file_path="checkpoint_75000",
            historical_size_estimate=258_718_956,
            license="MIT (advertised)",
            role="policy_component",
            notices=("advertised MIT; confirm against repo contents before relying on it",),
        ),
        record(
            asset_id="judge.qwen2_5_vl_7b",
            repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
            repo_type="model",
            revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
            historical_size_estimate=16_600_000_000,
            role="judge",
        ),
        record(
            asset_id="judge.vjepa2_features",
            repo_id="facebook/vjepa2-vitl-fpc64-256",
            repo_type="model",
            historical_size_estimate=1_303_947_864,
            role="feature_diagnostic",
            notices=(
                "encoder, not a text-following rubric judge",
                "skip original/model.pth (5.1 GB duplicate)",
                "separate analysis; must not alter the deterministic validity gate",
            ),
        ),
        record(
            asset_id="data.bridge_orig_lerobot",
            repo_id="IPEC-COMMUNITY/bridge_orig_lerobot",
            repo_type="dataset",
            license="Apache-2.0 (advertised)",
            role="scenario_source",
            notices=(
                "pull only manifest-selected episodes",
                "not automatically the same scene distribution as AutoEval",
            ),
        ),
        record(
            asset_id="data.auto_eval",
            repo_id="zhouzypaul/auto_eval",
            repo_type="dataset",
            license="MIT (advertised)",
            role="scenario_and_label_source",
            notices=(
                "drawer scene only; labels are classifier-generated, not human",
                "no policy identity; never use as five-task human ground truth",
                "pickles import robot_eval_logger/wandb; a stub unpickler is not a security boundary",
            ),
        ),
        record(
            asset_id="world.irasim_bridge",
            repo_id="bytedance/IRASim",
            repo_type="source",
            revision="c72b6dade6fcd65971e0aa8ab49ea39b15108c90",
            license="Apache-2.0",
            role="alternate_world_model",
            notices=("archive is uncompressed TAR despite the .tar.gz suffix",),
        ),
        record(
            asset_id="world.sdxl_vae",
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            repo_type="model",
            revision="462165984030d82259a11f4367a4eed129e94a7b",
            file_path="vae/",
            license="OpenRAIL++",
            role="alternate_world_component",
            notices=("licensing is distinct from IRASim's Apache-2.0 source",),
        ),
    )


def _plan_record_from_cluster_entry(entry: Mapping[str, Any]) -> AssetRecord:
    """Adapt one ``cluster.asset_plan`` entry into an ``AssetRecord``.

    The cluster plan is the richer, authoritative source: it carries pinned
    revisions, per-file selections, license *status* (not just a license string)
    and redistribution terms.  Two of its distinctions are preserved carefully
    here rather than flattened:

    ``license_status``
        ``advertised_unverified`` means a repo *claims* a license that nobody has
        checked.  That is not a resolved license, so ``access_status`` stays
        unresolved and the row cannot report ``verified``.
    ``historical_total_estimate_bytes``
        A size from a spec table is a discovery aid, not a measurement.  It is
        deliberately **not** mapped onto ``byte_length``, which means "bytes we
        actually retrieved".
    """

    files = entry.get("files") or []
    file_path = None
    if isinstance(files, (list, tuple)) and files:
        first = files[0]
        file_path = str(first.get("path")) if isinstance(first, Mapping) else str(first)
        if len(files) > 1:
            file_path = "%s (+%d more)" % (file_path, len(files) - 1)

    license_status = str(entry.get("license_status") or "unresolved")
    access = str(entry.get("access") or "unknown")
    # Only a checked license on a publicly reachable asset counts as resolved.
    access_status = "resolved" if (license_status == "verified" and access == "public") else "unresolved"

    notices = []
    for key in ("notes", "notices", "traps"):
        value = entry.get(key)
        if isinstance(value, str):
            notices.append(value)
        elif isinstance(value, (list, tuple)):
            notices.extend(str(item) for item in value)
    if license_status == "advertised_unverified":
        notices.append("license is advertised by the repo but has not been verified")
    if not entry.get("revision_immutable", False):
        notices.append("revision is not pinned to an immutable commit")

    return AssetRecord(
        asset_id=str(entry.get("name") or entry.get("repo_id")),
        repo_id=str(entry.get("repo_id")),
        repo_type=str(entry.get("repo_type") or "model"),
        revision=entry.get("revision") if entry.get("revision_immutable") else None,
        file_path=file_path,
        # Never a historical estimate: byte_length means bytes actually retrieved.
        byte_length=None,
        historical_size_estimate=entry.get("historical_total_estimate_bytes"),
        sha256=None,
        source_url=entry.get("source_url"),
        retrieved_at=None,
        license=entry.get("license") if license_status == "verified" else None,
        access_status=access_status,
        redistribution=str(entry.get("redistribution") or "unresolved"),
        notices=tuple(dict.fromkeys(notices)),
        loader_revision=entry.get("loader_revision"),
        container_digest=entry.get("container_digest"),
        compatibility_profile_id=entry.get("compatibility_profile_id"),
        role="primary_matrix" if entry.get("required_for_primary_matrix") else "supporting",
    )


def cluster_plan_records() -> Tuple[AssetRecord, ...]:
    """The cluster acquisition plan, adapted into lock rows.

    ``cluster`` is a top-level package in the repository root, which is not on
    ``sys.path`` when PLUMB runs from its installed console script.  The root is
    therefore added explicitly rather than letting the import fail silently and
    quietly produce a thinner lock than the operator expects.
    """

    import sys

    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from cluster.asset_plan import asset_plan_entries

    records: List[AssetRecord] = []
    problems: List[str] = []
    for index, entry in enumerate(asset_plan_entries()):
        if not isinstance(entry, Mapping):
            problems.append("entry %d is not a mapping" % index)
            continue
        try:
            records.append(_plan_record_from_cluster_entry(entry))
        except Exception as exc:  # noqa: BLE001
            # Never silently drop a planned asset: a missing row would make the
            # lock look more complete than it is.
            problems.append("%s: %s" % (entry.get("name", index), exc))
    if problems:
        raise ProtocolError(
            "%d acquisition-plan entries could not be adapted: %s"
            % (len(problems), "; ".join(problems[:5]))
        )
    return tuple(records)


def build_asset_lock(existing: Optional[AssetLock] = None) -> AssetLock:
    """Merge the acquisition plan with any already-retrieved rows.

    Retrieved rows win: a plan row never overwrites a real hash, because the
    plan describes what we intend to fetch and the retrieved row describes what
    we actually got.
    """

    lock = AssetLock()
    # Two plans describe the same assets under different names (this module's
    # discovery list and the richer cluster acquisition plan). Key on the actual
    # artifact -- repo plus file -- so one asset produces one row, and let the
    # cluster entry win because it carries pinned revisions and license status.
    by_artifact: Dict[Tuple[str, str], AssetRecord] = {}

    def _key(record: AssetRecord) -> Tuple[str, str]:
        return (record.repo_id.strip().lower(), (record.file_path or "").strip().lower())

    for record in asset_plan_records():
        by_artifact[_key(record)] = record
    for record in cluster_plan_records():
        by_artifact[_key(record)] = record
    for record in by_artifact.values():
        lock.add(record)

    if existing is not None:
        # A retrieved row always wins: the plan says what we meant to fetch, the
        # retrieved row says what we actually got.
        retrieved = {_key(r): r for r in existing.assets.values() if r.verified or r.sha256}
        for key, record in retrieved.items():
            for asset_id, planned in list(lock.assets.items()):
                if _key(planned) == key:
                    lock.assets.pop(asset_id)
            lock.add(record)
    return lock


# ---------------------------------------------------------------------------
# Protocol document
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolDocument:
    """The frozen protocol plus its canonical hash."""

    protocol_id: str
    created_at: str
    policies: Tuple[str, ...]
    tasks: Tuple[str, ...]
    starts_per_task: int
    seed: int
    tolerances: Tolerances
    analysis_code_revision: Optional[str]
    judge_sampling: Mapping[str, Any]
    exclusion_policy: Mapping[str, Any]
    cost_selection_rule: Mapping[str, Any]
    cohorts: Tuple[str, ...]
    scenario_manifest_hash: Optional[str] = None
    asset_lock_hash: Optional[str] = None
    operating_point_id: Optional[str] = None
    frozen: bool = False
    amends: Optional[str] = None
    notes: Tuple[str, ...] = ()

    @property
    def total_planned_episodes(self) -> int:
        return len(self.policies) * len(self.tasks) * self.starts_per_task

    def body(self) -> Dict[str, Any]:
        """The hashed content.  Deliberately excludes the hash itself."""

        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "protocol_id": self.protocol_id,
            "created_at": self.created_at,
            "amends": self.amends,
            "frozen": self.frozen,
            "matrix": {
                "policies": list(self.policies),
                "tasks": list(self.tasks),
                "starts_per_task": self.starts_per_task,
                "total_planned_episodes": self.total_planned_episodes,
                "task_horizons": {task: TASK_HORIZONS[task] for task in self.tasks},
                "task_prompts": {task: EXACT_TASK_PROMPTS[task] for task in self.tasks},
            },
            "seeds": {"root": self.seed, "derivation": "sha256(root, policy, task, start_id)"},
            "cohorts": list(self.cohorts),
            "alpha_families": ALPHA_FAMILIES,
            "tolerances": self.tolerances.to_mapping(),
            "judge_sampling": dict(self.judge_sampling),
            "exclusion_policy": dict(self.exclusion_policy),
            "cost_selection_rule": dict(self.cost_selection_rule),
            "analysis": {
                "code_revision": self.analysis_code_revision,
                "opt_ins": list(ANALYSIS_OPT_INS),
                "mmrv_implementation": "simpler_env.utils.metrics (perf_sim, perf_real)",
                "bootstrap_unit": "start_state_bundle_within_fixed_task",
            },
            "scenario_manifest_hash": self.scenario_manifest_hash,
            "asset_lock_hash": self.asset_lock_hash,
            "operating_point_id": self.operating_point_id,
            "notes": list(self.notes),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.body())

    def analysis_options(self) -> Dict[str, Any]:
        """Opt-in switches for ``plumb.measurement.analyze``.

        ``measurement`` returns ``not_computed`` for the headline statistics
        unless a caller asks for them.  A frozen protocol is the thing that
        authorises asking, which is why this lives here and not in the API.
        """

        options: Dict[str, Any] = {name: True for name in ANALYSIS_OPT_INS}
        options.update(
            {
                "protocol_hash": self.sha256,
                "protocol_id": self.protocol_id,
                "bootstrap_replicates": self.tolerances.bootstrap_replicates,
                "split_half_repetitions": self.tolerances.split_half_repetitions,
                "mdd_simulations_per_gap": self.tolerances.mdd_simulations_per_gap,
                "power_target": self.tolerances.power_target,
                "alpha_families": ALPHA_FAMILIES,
            }
        )
        return options

    def preregistration_status(self, repo_root: Optional[Path] = None) -> PreregistrationRecord:
        return discover_preregistration(self.sha256, self.preregistration_tag, repo_root=repo_root)

    @property
    def preregistration_tag(self) -> str:
        return "prereg/" + self.protocol_id

    def to_mapping(self, repo_root: Optional[Path] = None) -> Dict[str, Any]:
        body = self.body()
        return {
            **body,
            "protocol_sha256": self.sha256,
            "preregistration": self.preregistration_status(repo_root=repo_root).to_mapping(),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProtocolDocument":
        try:
            matrix = payload["matrix"]
            analysis = payload.get("analysis", {})
            document = cls(
                protocol_id=str(payload["protocol_id"]),
                created_at=str(payload["created_at"]),
                policies=tuple(str(item) for item in matrix["policies"]),
                tasks=tuple(str(item) for item in matrix["tasks"]),
                starts_per_task=int(matrix["starts_per_task"]),
                seed=int(payload["seeds"]["root"]),
                tolerances=Tolerances.from_mapping(payload["tolerances"]),
                analysis_code_revision=analysis.get("code_revision"),
                judge_sampling=dict(payload.get("judge_sampling", {})),
                exclusion_policy=dict(payload.get("exclusion_policy", {})),
                cost_selection_rule=dict(payload.get("cost_selection_rule", {})),
                cohorts=tuple(str(item) for item in payload.get("cohorts", ())),
                scenario_manifest_hash=payload.get("scenario_manifest_hash"),
                asset_lock_hash=payload.get("asset_lock_hash"),
                operating_point_id=payload.get("operating_point_id"),
                frozen=bool(payload.get("frozen", False)),
                amends=payload.get("amends"),
                notes=tuple(str(item) for item in payload.get("notes", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("protocol.json is incomplete: %s" % exc) from exc

        recorded = payload.get("protocol_sha256")
        if recorded and recorded != document.sha256:
            raise ProtocolError(
                "protocol.json hash mismatch: recorded %s, recomputed %s" % (recorded, document.sha256)
            )
        return document

    @classmethod
    def load(cls, path: Path) -> "ProtocolDocument":
        source = Path(path)
        if not source.is_file():
            raise ProtocolError("protocol document not found at %s" % source)
        with source.open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def save(self, path: Path, repo_root: Optional[Path] = None) -> str:
        _atomic_write_json(Path(path), self.to_mapping(repo_root=repo_root))
        return self.sha256


def default_exclusion_policy() -> Dict[str, Any]:
    """Spec section 5/6: exclude from *primary rates*, never from coverage."""

    return {
        "validity_invalid": "unevaluable_for_primary_rates_retained_in_coverage",
        "validity_unknown": "unevaluable_for_primary_rates_retained_in_coverage",
        "judge_no_quorum": "unevaluable_for_primary_rates_retained_in_coverage",
        "judge_refusal": "explicit_missing_outcome",
        "terminal_service_failure": "retained_in_intent_to_evaluate_denominator",
        "never_executed_cell": "incomplete_not_a_zero_success_measurement",
        "deletion_permitted": False,
        "partial_identification": "horowitz_manski_bounds_reported_alongside_complete_case",
        "primary_endpoint": "Y = 1{evaluable AND success}; missing outcomes are 0 only for this labelled lower bound",
    }


def default_cost_selection_rule() -> Dict[str, Any]:
    """Spec section 6.4 selection rule, stated so it cannot drift later."""

    return {
        "search_space": ["resolution", "denoise_steps", "chunk_partition", "batch_size"],
        "held_fixed": ["task_caps", "starts", "seeds", "judge"],
        "rule": "cheapest development-panel point satisfying all preregistered tolerances",
        "confirmation": "required on a disjoint cost-confirmation panel before any qualified claim",
        "preserve_failures": True,
        "horizon_guard": "fixed task horizon and coverage travel with every point; a shorter task is not a cheaper equivalent",
    }


def default_protocol(
    *,
    protocol_id: str = "protocol-v1",
    seed: int = 20260919,
    starts_per_task: int = 50,
    analysis_code_revision: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> ProtocolDocument:
    """Build the v1 protocol document from the spec's own constants."""

    root = repo_root or Path(__file__).resolve().parents[1]
    revision = analysis_code_revision or _git(["rev-parse", "HEAD"], cwd=root)
    return ProtocolDocument(
        protocol_id=protocol_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        policies=tuple(POLICY_IDS),
        tasks=tuple(TASK_IDS),
        starts_per_task=starts_per_task,
        seed=seed,
        tolerances=default_tolerances(),
        analysis_code_revision=revision,
        judge_sampling=dict(FROZEN_JUDGE_SAMPLING),
        exclusion_policy=default_exclusion_policy(),
        cost_selection_rule=default_cost_selection_rule(),
        cohorts=("primary", "development", "calibration", "cost_confirmation", "reverse_validation"),
        notes=(
            "Tolerances are candidates until a PreregistrationRecord reports status=registered.",
            "A changed adapter, backend, judge or operating point invalidates its dependent gates.",
        ),
    )


def freeze(document: ProtocolDocument, **updates: Any) -> ProtocolDocument:
    """Return a frozen copy, optionally binding manifest and asset hashes.

    Freezing changes the hash, which is intentional: the hash that gets
    preregistered must be the hash of the frozen document.
    """

    from dataclasses import replace as _replace

    return _replace(document, frozen=True, **updates)


def amend(document: ProtocolDocument, new_protocol_id: str, **updates: Any) -> ProtocolDocument:
    """Create an amendment with a new ID; never overwrite the original."""

    from dataclasses import replace as _replace

    if new_protocol_id == document.protocol_id:
        raise ProtocolError("an amendment requires a new protocol_id")
    return _replace(
        document,
        protocol_id=new_protocol_id,
        amends=document.protocol_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        frozen=False,
        **updates,
    )


def build_analysis_manifest(
    episodes: Sequence[Mapping[str, Any]],
    *,
    protocol_hash: str,
    cohort: str = "primary",
) -> Dict[str, Any]:
    """Freeze the identity manifest ``plumb.measurement`` requires.

    ``measurement._validate_analysis_manifest`` refuses to score a primary
    analysis without a frozen manifest naming every allowed episode ID and its
    identity.  The manifest must be built from the *planned* ledger rows at run
    creation, not from whatever survived execution -- otherwise a run could
    quietly drop its failures and still look complete.

    Every episode is therefore included regardless of its status, and the
    resulting membership is what the analysis later checks the ledger against.
    """

    if not episodes:
        raise ProtocolError("an analysis manifest needs at least one planned episode")

    allowed: List[str] = []
    identities: Dict[str, Dict[str, Any]] = {}
    variants: Dict[str, str] = {}
    mismatched: List[str] = []

    for episode in episodes:
        episode_id = episode.get("episode_id")
        if not episode_id:
            raise ProtocolError("every planned episode needs an episode_id to be manifested")
        episode_id = str(episode_id)
        recorded_cohort = str(episode.get("cohort") or "").strip()
        recorded_hash = str(episode.get("protocol_hash") or "").strip()
        if recorded_cohort != cohort or recorded_hash != protocol_hash:
            mismatched.append(episode_id)
            continue
        policy = str(episode.get("policy"))
        variant = str(episode.get("policy_variant") or policy)
        existing = variants.setdefault(policy, variant)
        if existing != variant:
            raise ProtocolError(
                "policy %s appears with two variants (%s, %s); a manifest cannot pool them"
                % (policy, existing, variant)
            )
        allowed.append(episode_id)
        identities[episode_id] = {
            "policy": policy,
            "policy_variant": variant,
            "task": str(episode.get("task")),
            "cohort": recorded_cohort,
            "protocol_hash": recorded_hash,
            "start_id": str(episode.get("start_id")),
        }

    if mismatched:
        raise ProtocolError(
            "%d episode(s) do not carry cohort=%s and the frozen protocol hash; refusing to "
            "manifest a partially-identified run" % (len(mismatched), cohort)
        )

    manifest = {
        "frozen": True,
        "cohort": cohort,
        "protocol_hash": protocol_hash,
        "identity": {"cohort": cohort, "protocol_hash": protocol_hash},
        "policy_variants": variants,
        "allowed_episode_ids": sorted(allowed),
        "episode_identities": identities,
    }
    manifest["manifest_sha256"] = canonical_sha256(
        {
            "cohort": cohort,
            "protocol_hash": protocol_hash,
            "policy_variants": variants,
            "allowed_episode_ids": manifest["allowed_episode_ids"],
        }
    )
    return manifest


def analysis_options_for_run(
    document: ProtocolDocument,
    episodes: Sequence[Mapping[str, Any]],
    *,
    cohort: str = "primary",
    sweeps: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Protocol opt-ins plus the frozen manifest, ready for ``analyze``.

    This is the call that turns the already-implemented split-half and MDD paths
    from ``not_computed`` into real numbers: they are gated on opt-in flags that
    nothing in the repository previously set.
    """

    options = document.analysis_options()
    options["analysis_mode"] = "primary" if cohort == "primary" else "diagnostic"
    options["analysis_manifest"] = build_analysis_manifest(
        episodes, protocol_hash=document.sha256, cohort=cohort
    )
    options["mdd"] = mdd_options(document, episodes)
    if sweeps:
        options["sweeps"] = dict(sweeps)
    return options


def mdd_options(
    document: ProtocolDocument,
    episodes: Sequence[Mapping[str, Any]],
    reference_policy: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a retrospective MDD configuration from the observed matrix.

    ``measurement.paired_mdd_power_curve`` refuses to guess baseline rates,
    which is correct: an MDD is meaningless without the marginal it is computed
    against.  Spec section 6 also draws a hard line between a *planning* curve
    from development data and a *retrospective* curve whose nuisance
    probabilities come from the completed matrix, so this returns the
    retrospective form and labels it as such.

    The baseline is the designated policy's per-task observed-positive marginal,
    ordered by the protocol's task order so the macro label-swap null sees equal
    task sizes.  Returns a ``disabled`` config when the matrix cannot support
    one, rather than inventing a rate.
    """

    policy = reference_policy or (document.policies[0] if document.policies else None)
    if policy is None:
        return {"enabled": False, "reason": "no reference policy in the protocol"}

    per_task_total: Dict[str, int] = {task: 0 for task in document.tasks}
    per_task_positive: Dict[str, int] = {task: 0 for task in document.tasks}
    for episode in episodes:
        if str(episode.get("policy")) != policy:
            continue
        task = str(episode.get("task"))
        if task not in per_task_total:
            continue
        per_task_total[task] += 1
        # The primary endpoint is the observed-positive lower bound: an
        # unevaluable episode counts as a non-success for this endpoint only.
        if episode.get("status") == "completed" and episode.get("validity") == "valid":
            if episode.get("binary_success") is True:
                per_task_positive[task] += 1

    sizes = [per_task_total[task] for task in document.tasks]
    if any(size <= 0 for size in sizes):
        return {
            "enabled": False,
            "reason": "at least one task has no planned episodes for the reference policy",
        }
    if len(set(sizes)) != 1:
        return {
            "enabled": False,
            "reason": "the macro label-swap null requires equal task sample sizes",
        }

    baseline = [per_task_positive[task] / float(per_task_total[task]) for task in document.tasks]
    return {
        "enabled": True,
        "design": "retrospective",
        "reference_policy": policy,
        "endpoint": "observed_positive_lower_bound",
        "baseline_rates": baseline,
        "n_per_task": sizes[0],
        "alpha": ALPHA_FAMILIES["macro_pairwise"]["alpha"],
        "simulations": document.tolerances.mdd_simulations_per_gap,
        "seed": document.seed,
        "note": (
            "Retrospective curve: nuisance and joint probabilities come from the completed matrix, "
            "not from development planning data."
        ),
    }


def protocol_payload(document: ProtocolDocument, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """API-shaped protocol summary including the revisions the console needs."""

    prereg = document.preregistration_status(repo_root=repo_root)
    return {
        "protocol_id": document.protocol_id,
        "protocol_hash": document.sha256,
        "frozen": document.frozen,
        "preregistration": prereg.to_mapping(),
        "policies": list(document.policies),
        "tasks": list(document.tasks),
        "starts_per_task": document.starts_per_task,
        "total": document.total_planned_episodes,
        "task_horizons": {task: TASK_HORIZONS[task] for task in document.tasks},
        "analysis_code_revision": document.analysis_code_revision,
        "scenario_manifest_hash": document.scenario_manifest_hash,
        "asset_lock_hash": document.asset_lock_hash,
        "operating_point_id": document.operating_point_id,
        "judge_sampling": dict(document.judge_sampling),
        "alpha_families": ALPHA_FAMILIES,
    }


__all__ = [
    "ALPHA_FAMILIES",
    "ANALYSIS_OPT_INS",
    "ASSET_LOCK_SCHEMA_VERSION",
    "AssetLock",
    "AssetRecord",
    "FROZEN_JUDGE_SAMPLING",
    "PROTOCOL_SCHEMA_VERSION",
    "PreregistrationRecord",
    "ProtocolDocument",
    "ProtocolError",
    "TASK_HORIZONS",
    "Tolerances",
    "amend",
    "analysis_options_for_run",
    "asset_plan_records",
    "build_analysis_manifest",
    "build_asset_lock",
    "canonical_bytes",
    "canonical_sha256",
    "cluster_plan_records",
    "default_cost_selection_rule",
    "default_exclusion_policy",
    "default_protocol",
    "default_tolerances",
    "discover_preregistration",
    "freeze",
    "mdd_options",
    "protocol_payload",
]
