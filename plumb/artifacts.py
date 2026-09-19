"""Writers for the spec section 8 output contract.

Section 8 names fourteen artifacts.  Before this module nothing wrote any of
them: ``GateLedger.save`` existed with no call site, calibration printed to
stdout, and the episode export only streamed to an HTTP client.  Everything
lived under ``data/`` and ``results/`` was mentioned only in ``.gitignore``.

Two rules shape the code below.

**Null means unknown, never zero.**  A writer never substitutes ``0`` for an
absent measurement and never omits a planned episode from a report to make the
arithmetic tidier.  ``_json_safe`` rejects NaN/inf so a non-portable float can
never reach an artifact.

**An artifact records what happened, including that nothing happened.**  Writing
``reliability.json`` for a run whose statistics were withheld produces a file
that says they were withheld and why, rather than no file at all.  A reader
should never have to infer meaning from a missing path.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .records import utc_now
from .reference import HUMAN, SIMPLER, reference_payload


ARTIFACT_SCHEMA_VERSION = 1

#: The section 8 tree, relative to the results root.  Directories end in "/".
SECTION_8_ARTIFACTS: Tuple[str, ...] = (
    "gates.json",
    "run_ledger.jsonl",
    "per_episode/",
    "segments/",
    "leaderboard.md",
    "reliability.json",
    "judge_calibration.json",
    "exclusions.json",
    "load_test.json",
    "economics.json",
    "provenance/",
)

#: Artifacts that live at the repository root rather than under ``results/``.
ROOT_ARTIFACTS: Tuple[str, ...] = ("assets.lock.json", "protocol.json", "scenarios.jsonl")


class ArtifactError(RuntimeError):
    """An artifact could not be written or would have been misleading."""


def _json_safe(value: Any, path: str = "$") -> Any:
    """Recursively verify JSON portability, rejecting NaN and infinity."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactError("non-finite float at %s would not serialize portably" % path)
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, "%s.%s" % (path, key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, "%s[%d]" % (path, index)) for index, item in enumerate(value)]
    raise ArtifactError("value at %s is not JSON-representable: %r" % (path, type(value).__name__))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(_json_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n")


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    lines: List[str] = []
    for row in rows:
        lines.append(json.dumps(_json_safe(row), sort_keys=True, separators=(",", ":"), allow_nan=False))
        count += 1
    _atomic_write(path, "".join(line + "\n" for line in lines))
    return count


def _percent(value: Optional[float]) -> str:
    """Render a rate as a percentage, or an explicit dash for unknown."""

    if value is None:
        return "—"
    return "%.1f%%" % (100.0 * float(value))


def _interval(pair: Optional[Sequence[float]]) -> str:
    if not pair or pair[0] is None or pair[1] is None:
        return "—"
    return "[%.1f, %.1f]%%" % (100.0 * float(pair[0]), 100.0 * float(pair[1]))


@dataclass(frozen=True)
class WriteReceipt:
    """What a writer produced, for the gate ledger's ``evidence_uris``."""

    path: Path
    rows: Optional[int] = None

    def to_mapping(self) -> Dict[str, Any]:
        return {"uri": self.path.as_posix(), "rows": self.rows}


class ArtifactStore:
    """Owns the ``results/`` tree and the three root artifacts."""

    def __init__(self, root: Path, results_dirname: str = "results") -> None:
        self.root = Path(root)
        self.results = self.root / results_dirname
        self.results.mkdir(parents=True, exist_ok=True)
        (self.results / "per_episode").mkdir(exist_ok=True)
        (self.results / "segments").mkdir(exist_ok=True)
        (self.results / "provenance").mkdir(exist_ok=True)

    # -- paths ----------------------------------------------------------------

    @property
    def gates_path(self) -> Path:
        return self.results / "gates.json"

    @property
    def run_ledger_path(self) -> Path:
        return self.results / "run_ledger.jsonl"

    def per_episode_path(self, run_id: str) -> Path:
        return self.results / "per_episode" / ("%s.jsonl" % _safe_component(run_id))

    def segments_dir(self, run_id: str) -> Path:
        path = self.results / "segments" / _safe_component(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def leaderboard_path(self) -> Path:
        return self.results / "leaderboard.md"

    @property
    def reliability_path(self) -> Path:
        return self.results / "reliability.json"

    @property
    def judge_calibration_path(self) -> Path:
        return self.results / "judge_calibration.json"

    @property
    def exclusions_path(self) -> Path:
        return self.results / "exclusions.json"

    @property
    def load_test_path(self) -> Path:
        return self.results / "load_test.json"

    @property
    def economics_path(self) -> Path:
        return self.results / "economics.json"

    @property
    def reverse_validation_path(self) -> Path:
        return self.results / "reverse_validation.json"

    @property
    def provenance_dir(self) -> Path:
        return self.results / "provenance"

    # -- writers --------------------------------------------------------------

    def write_gates(self, gate_ledger: Any) -> WriteReceipt:
        """Persist the gate ledger to the section 8 path.

        ``GateLedger.save`` is already atomic; this is the missing call site.
        """

        gate_ledger.save(str(self.gates_path))
        return WriteReceipt(self.gates_path)

    def write_run_ledger(self, runs: Sequence[Mapping[str, Any]]) -> WriteReceipt:
        rows = [{"schema_version": ARTIFACT_SCHEMA_VERSION, "exported_at": utc_now(), **dict(run)} for run in runs]
        count = _atomic_write_jsonl(self.run_ledger_path, rows)
        return WriteReceipt(self.run_ledger_path, count)

    def write_per_episode(
        self,
        run_id: str,
        episodes: Sequence[Mapping[str, Any]],
        run: Optional[Mapping[str, Any]] = None,
    ) -> WriteReceipt:
        """One final record per *planned* logical episode.

        Spec section 8: "Finalize every planned episode, including failures."
        So this writer refuses to drop a planned row, and it does not invent a
        measurement for one that never ran -- ``status`` stays ``planned`` and
        every measurement stays null.
        """

        planned = len(episodes)
        if planned == 0:
            raise ArtifactError("refusing to write an empty per-episode record for %s" % run_id)
        rows = [self._episode_record(run_id, episode, run) for episode in episodes]
        count = _atomic_write_jsonl(self.per_episode_path(run_id), rows)
        if count != planned:
            raise ArtifactError("per-episode export dropped rows: wrote %d of %d" % (count, planned))
        return WriteReceipt(self.per_episode_path(run_id), count)

    @staticmethod
    def _episode_record(
        run_id: str,
        episode: Mapping[str, Any],
        run: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        config = dict((run or {}).get("config") or {})
        missing_reason = episode.get("missing_reason")
        if missing_reason is None and episode.get("status") == "planned":
            missing_reason = "not_run"
        return {
            "schema_version": int(episode.get("schema_version", ARTIFACT_SCHEMA_VERSION)),
            "run_id": run_id,
            "episode_id": episode.get("episode_id"),
            "cohort": episode.get("cohort"),
            "protocol_hash": episode.get("protocol_hash"),
            "task": episode.get("task"),
            "start_id": episode.get("start_id"),
            "start_lineage_id": episode.get("start_lineage_id"),
            "scenario_manifest_hash": episode.get("scenario_manifest_hash"),
            "status": episode.get("status"),
            "policy": {
                "name": episode.get("policy"),
                "variant": episode.get("policy_variant"),
                **dict(episode.get("policy_identity") or {}),
            },
            "world_model": dict(episode.get("world_identity") or {}),
            "judge": dict(episode.get("judge_identity") or {}),
            "seeds": dict(episode.get("seeds") or {}),
            "feedback_mode": episode.get("feedback_mode"),
            "parity_status": episode.get("parity_status"),
            "horizon_actions": episode.get("horizon_actions"),
            "executed_actions": episode.get("executed_actions"),
            "n_segments": episode.get("n_segments"),
            "validity": episode.get("validity"),
            "progress_score": episode.get("progress_score"),
            "binary_success": episode.get("binary_success"),
            "missing_reason": missing_reason,
            "exclusion_reason": episode.get("exclusion_reason"),
            "raw_judge_samples_ref": episode.get("raw_judge_samples_ref"),
            "attempt_ids": list(episode.get("attempt_ids") or []),
            "platform_request_ids": list(episode.get("platform_request_ids") or []),
            "segments_manifest_ref": episode.get("segments_manifest_ref"),
            "video_ref": episode.get("video_ref") or episode.get("video_url"),
            "timing_ref": episode.get("timing_ref"),
            "timing": dict(episode.get("timing") or {}),
            "compute_gpu_seconds": episode.get("compute_gpu_seconds"),
            "allocated_gpu_seconds": episode.get("allocated_gpu_seconds"),
            "estimated_usd": episode.get("estimated_usd"),
            "cost_basis_ref": episode.get("cost_basis_ref"),
            "mode": episode.get("mode"),
            "run_backend": config.get("backend"),
            "artifact_refs": dict(episode.get("artifact_refs") or {}),
        }

    def write_leaderboard(
        self,
        analysis: Mapping[str, Any],
        run: Optional[Mapping[str, Any]] = None,
        protocol: Optional[Mapping[str, Any]] = None,
    ) -> WriteReceipt:
        """Human-readable rates, reference, bounds, coverage, n and parity."""

        _atomic_write(self.leaderboard_path, _render_leaderboard(analysis, run, protocol))
        return WriteReceipt(self.leaderboard_path)

    def write_reliability(
        self,
        analysis: Mapping[str, Any],
        sweeps: Optional[Mapping[str, Any]] = None,
    ) -> WriteReceipt:
        """Split-half, MDD, MMRV, drift and cost, each with its own status.

        A withheld statistic is written as ``not_computed`` with its reason
        rather than omitted, so the absence is auditable.
        """

        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "generated_at": utc_now(),
            "analysis_identity": analysis.get("analysis_identity"),
            "ledger_state": analysis.get("ledger_state"),
            "endpoint": analysis.get("endpoint"),
            "split_half_reliability": analysis.get("reliability", {"status": "not_computed"}),
            "minimum_detectable_difference": analysis.get("mdd", {"status": "not_computed"}),
            "mmrv": analysis.get("mmrv", {"status": "not_computed"}),
            "advanced_inference": analysis.get("advanced_inference", {"status": "not_computed"}),
            "lineage_leakage": analysis.get("lineage_leakage"),
            "cost_fidelity": (sweeps or {}).get("cost_fidelity", {"status": "not_computed", "reason": "no sweep artifact supplied"}),
            "drift": (sweeps or {}).get("drift", {"status": "not_computed", "reason": "no drift artifact supplied"}),
            "notes": [
                "Repeatability does not establish validity.",
                "Split repetitions reuse episodes across repetitions and are not independent studies.",
                "A drift result is the last tested horizon, never an unlimited guarantee.",
            ],
        }
        _atomic_write_json(self.reliability_path, payload)
        return WriteReceipt(self.reliability_path)

    def write_exclusions(self, analysis: Mapping[str, Any]) -> WriteReceipt:
        """Missing/invalid reasons and sensitivity, per cell.

        Horowitz-Manski point-bound width equals the missing-outcome rate
        exactly (spec section 6); this writer asserts that identity rather than
        trusting it, because a mismatch would mean the bounds are wrong.
        """

        cells = list(analysis.get("cells") or [])
        per_cell: List[Dict[str, Any]] = []
        for cell in cells:
            width = cell.get("missing_bounds_width")
            missingness = cell.get("missingness_rate")
            if width is not None and missingness is not None:
                if abs(float(width) - float(missingness)) > 1e-9:
                    raise ArtifactError(
                        "missing-bound width %r does not equal missingness %r for %s/%s"
                        % (width, missingness, cell.get("policy"), cell.get("task"))
                    )
            per_cell.append(
                {
                    "policy": cell.get("policy"),
                    "policy_variant": cell.get("policy_variant"),
                    "task": cell.get("task"),
                    "n": cell.get("n"),
                    "valid": cell.get("valid"),
                    "missing": cell.get("missing"),
                    "coverage": cell.get("coverage"),
                    "missingness_rate": missingness,
                    "missing_bounds": cell.get("missing_bounds"),
                    "missing_bounds_width": width,
                    "status_counts": cell.get("status_counts"),
                    "validity_counts": cell.get("validity_counts"),
                    "missing_reason_counts": cell.get("missing_reason_counts"),
                    "service_failures": cell.get("service_failures"),
                }
            )
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "generated_at": utc_now(),
            "convention": {
                "bounds": "horowitz_manski_no_assumption",
                "identity": "point-bound width == missing-outcome rate",
                "policy": "excluded episodes stay in coverage and partial identification; they are never deleted",
                "unevaluable": "an unevaluable episode is not a known physical failure",
            },
            "cells": per_cell,
            "summary": analysis.get("summary"),
        }
        _atomic_write_json(self.exclusions_path, payload)
        return WriteReceipt(self.exclusions_path)

    def write_judge_calibration(self, report: Mapping[str, Any]) -> WriteReceipt:
        payload = {"schema_version": ARTIFACT_SCHEMA_VERSION, "generated_at": utc_now(), **dict(report)}
        _atomic_write_json(self.judge_calibration_path, payload)
        return WriteReceipt(self.judge_calibration_path)

    def write_load_test(self, rehearsals: Sequence[Mapping[str, Any]], target: Mapping[str, Any]) -> WriteReceipt:
        """All rehearsals, not best-of-three (spec section 7).

        The writer refuses to drop a failed rehearsal, because reporting only
        the successful ones is the exact failure mode the spec calls out.
        """

        rows = [dict(item) for item in rehearsals]
        passing = [row for row in rows if row.get("passed") is True]
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "generated_at": utc_now(),
            "target": dict(target),
            "rehearsal_count": len(rows),
            "rehearsals": rows,
            "all_rehearsals_reported": True,
            "passing_count": len(passing),
            "qualified": len(rows) >= 3 and len(passing) == len(rows),
            "reason": (
                "all rehearsals met the displayed target"
                if rows and len(passing) == len(rows) and len(rows) >= 3
                else "target not met or fewer than three complete rehearsals; the measured values stand as reported"
            ),
        }
        _atomic_write_json(self.load_test_path, payload)
        return WriteReceipt(self.load_test_path)

    def write_economics(self, economics: Mapping[str, Any]) -> WriteReceipt:
        """Pricing basis, allocations, estimates and reconciliation status.

        USD stays ``None`` when the price basis is unknown; spec section 7 is
        explicit that an API field named ``price`` does not imply hourly USD.
        """

        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "generated_at": utc_now(),
            "views": {
                "marginal_execution_estimate": "attributed stage work, retries and transfers for this run only",
                "total_demonstration_run_cost": "every allocation from prewarm start through cooldown, counted once",
            },
            "formula": "estimated_usd = sum(resource_rate_per_hour * allocated_resource_hours) + other_charges",
            **dict(economics),
        }
        if payload.get("price_basis") in (None, {}, ""):
            payload.setdefault("reason", "USD unavailable: no verified price basis (currency/unit) was resolved")
        _atomic_write_json(self.economics_path, payload)
        return WriteReceipt(self.economics_path)

    def write_reverse_validation(self, report: Mapping[str, Any]) -> WriteReceipt:
        payload = {"schema_version": ARTIFACT_SCHEMA_VERSION, "generated_at": utc_now(), **dict(report)}
        _atomic_write_json(self.reverse_validation_path, payload)
        return WriteReceipt(self.reverse_validation_path)

    def record_provenance(self, name: str, payload: Mapping[str, Any]) -> WriteReceipt:
        path = self.provenance_dir / ("%s.json" % _safe_component(name))
        _atomic_write_json(path, {"schema_version": ARTIFACT_SCHEMA_VERSION, "recorded_at": utc_now(), **dict(payload)})
        return WriteReceipt(path)

    def copy_segment_manifest(self, run_id: str, episode_id: str, source: Path) -> WriteReceipt:
        destination = self.segments_dir(run_id) / ("%s.json" % _safe_component(episode_id))
        shutil.copy2(source, destination)
        return WriteReceipt(destination)

    # -- inventory ------------------------------------------------------------

    def inventory(self) -> Dict[str, Any]:
        """Which section 8 artifacts exist, for the smoke test and the console."""

        entries: Dict[str, Any] = {}
        for relative in SECTION_8_ARTIFACTS:
            path = self.results / relative.rstrip("/")
            if relative.endswith("/"):
                present = path.is_dir()
                children = sorted(item.name for item in path.iterdir()) if present else []
                entries[relative] = {
                    "uri": path.as_posix(),
                    "present": present,
                    "kind": "directory",
                    "entries": children,
                    "count": len(children),
                }
            else:
                entries[relative] = {
                    "uri": path.as_posix(),
                    "present": path.is_file(),
                    "kind": "file",
                    "bytes": path.stat().st_size if path.is_file() else None,
                }
        for relative in ROOT_ARTIFACTS:
            path = self.root / relative
            entries[relative] = {
                "uri": path.as_posix(),
                "present": path.is_file(),
                "kind": "file",
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        missing = sorted(key for key, value in entries.items() if not value["present"])
        return {
            "root": self.root.as_posix(),
            "results": self.results.as_posix(),
            "artifacts": entries,
            "missing": missing,
            "complete": not missing,
        }


def _safe_component(value: str) -> str:
    """Reduce an identifier to a safe single path component."""

    cleaned = "".join(character if character.isalnum() or character in "-_." else "-" for character in str(value))
    cleaned = cleaned.strip("-.") or "unnamed"
    if len(cleaned) > 120:
        cleaned = cleaned[:120]
    return cleaned


def _render_leaderboard(
    analysis: Mapping[str, Any],
    run: Optional[Mapping[str, Any]],
    protocol: Optional[Mapping[str, Any]],
) -> str:
    """Render ``leaderboard.md``.

    Every row carries its ``n``, its coverage, its missingness bounds and its
    parity status.  Ordering is by estimate for readability, but the file states
    the indeterminacy rule so a visual sort is never read as a resolved ranking.
    """

    identity = dict(analysis.get("analysis_identity") or {})
    ledger_state = dict(analysis.get("ledger_state") or {})
    config = dict((run or {}).get("config") or {})
    proto = dict(protocol or {})

    lines: List[str] = []
    lines.append("# Nightshift leaderboard")
    lines.append("")
    lines.append("Generated %s." % utc_now())
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append("| Run | `%s` |" % ((run or {}).get("id") or (run or {}).get("run_id") or "—"))
    lines.append("| Analysis status | %s |" % (identity.get("status") or "unknown"))
    lines.append("| Ledger state | %s |" % (ledger_state.get("status") or "unknown"))
    lines.append("| Backend | %s |" % (config.get("backend") or "—"))
    lines.append("| Mode | %s |" % (config.get("mode") or "—"))
    lines.append("| Cohort | %s |" % (config.get("cohort") or "—"))
    lines.append("| Protocol hash | `%s` |" % (proto.get("protocol_hash") or config.get("protocol_hash") or "—"))
    lines.append("| Protocol frozen | %s |" % proto.get("frozen", "—"))
    lines.append(
        "| Preregistration | %s |" % ((proto.get("preregistration") or {}).get("status") or "unregistered")
    )
    lines.append("| Seed | %s |" % (config.get("seed") if config.get("seed") is not None else "—"))
    lines.append("| Judge revision | %s |" % (proto.get("judge_revision") or "—"))
    lines.append("| Operating point | %s |" % (proto.get("operating_point_id") or "—"))
    lines.append("")

    if identity.get("status") != "accepted":
        lines.append(
            "> **Not a primary result.** The analysis manifest status is `%s`%s. "
            "Scientific fields are withheld rather than estimated."
            % (identity.get("status") or "unknown", _reason_suffix(identity.get("reason")))
        )
        lines.append("")

    cells = list(analysis.get("cells") or [])
    lines.append("## Per-cell virtual rates against the published human reference")
    lines.append("")
    lines.append(
        "| Policy | Variant | Task | n | valid | successes | S/V | S/N | Wilson (S/N) | Missing bounds | Coverage "
        "| Human | SIMPLER | Parity | Feedback |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|")
    ordered = sorted(
        cells,
        key=lambda cell: (
            -(cell.get("positive_rate") if cell.get("positive_rate") is not None else -1.0),
            str(cell.get("policy")),
            str(cell.get("task")),
        ),
    )
    for cell in ordered:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                cell.get("policy") or "—",
                cell.get("policy_variant") or "—",
                cell.get("task") or "—",
                cell.get("n") if cell.get("n") is not None else "—",
                cell.get("valid") if cell.get("valid") is not None else "—",
                cell.get("successes") if cell.get("successes") is not None else "—",
                _percent(cell.get("rate")),
                _percent(cell.get("positive_rate")),
                _interval(cell.get("positive_wilson")),
                _interval(cell.get("missing_bounds")),
                _percent(cell.get("coverage")),
                _percent(cell.get("reference_rate")),
                _percent(cell.get("simpler_rate")),
                cell.get("parity_status") or "—",
                cell.get("feedback_mode") or "—",
            )
        )
    if not cells:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
    lines.append("")

    lines.append("## Published reference (AutoEval Table 2 / Table 3)")
    lines.append("")
    lines.append("These are the paper's real-robot and simulator results, not Nightshift outcomes.")
    lines.append("")
    lines.append("| Policy | open_drawer | close_drawer | to_basket | to_sink | fold_cloth | total /250 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for policy, row in HUMAN.items():
        lines.append(
            "| %s | %d | %d | %d | %d | %d | %d |"
            % (
                policy,
                row["open_drawer"],
                row["close_drawer"],
                row["to_basket"],
                row["to_sink"],
                row["fold_cloth"],
                sum(row.values()),
            )
        )
    lines.append("")
    lines.append("SIMPLER has no cloth task; the common four-task subset is the only valid comparison.")
    lines.append("")
    lines.append("| Policy | open_drawer | close_drawer | to_basket | to_sink |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy, row in SIMPLER.items():
        lines.append(
            "| %s | %d | %d | %d | %d |"
            % (policy, row["open_drawer"], row["close_drawer"], row["to_basket"], row["to_sink"])
        )
    lines.append("")

    payload = reference_payload()
    limitations = payload.get("limitations") or []
    lines.append("## Reading rules")
    lines.append("")
    lines.append(
        "- **Indeterminacy.** Rows are sorted by estimate for readability only. An ordering is supported "
        "only when its one-sided bootstrap lower bound is positive at the family-adjusted alpha *and* the "
        "simultaneous exact binomial endpoint envelopes separate. Otherwise the ordering is indeterminate."
    )
    lines.append(
        "- **No published permutation claim.** MiniVLA leads Open pi-zero by one success in 250 in the "
        "reference, and four of five leave-one-task-out comparisons reverse that pair."
    )
    lines.append(
        "- **Exclusions are visible, not deleted.** Every excluded attempt stays in coverage and in the "
        "missing-outcome bounds. An unevaluable episode is not a known physical failure."
    )
    lines.append("- **Null means unknown.** A dash is an unmeasured quantity, never a zero.")
    for item in limitations:
        lines.append("- %s" % item)
    lines.append("")
    return "\n".join(lines) + "\n"


def _reason_suffix(reason: Any) -> str:
    return "" if not reason else " (%s)" % reason


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactStore",
    "ROOT_ARTIFACTS",
    "SECTION_8_ARTIFACTS",
    "WriteReceipt",
]
