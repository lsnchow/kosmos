"""Spec section 8 artifact writers.

These tests are mostly about refusals: an artifact must not drop a planned
episode, must not report best-of-three rehearsals, must not emit a non-portable
float, and must not turn an unknown value into a zero.
"""
from __future__ import annotations

import json

import pytest

from plumb.artifacts import (
    SECTION_8_ARTIFACTS,
    ArtifactError,
    ArtifactStore,
    _json_safe,
)
from plumb.gates import GateLedger
from plumb.reference import HUMAN, SIMPLER


def _cell(**overrides):
    cell = {
        "policy": "OpenVLA",
        "policy_variant": "OpenVLA",
        "task": "close_drawer",
        "n": 50,
        "valid": 40,
        "successes": 20,
        "missing": 10,
        "coverage": 0.8,
        "rate": 0.5,
        "positive_rate": 0.4,
        "positive_wilson": [0.27, 0.55],
        "missing_bounds": [0.4, 0.6],
        "missing_bounds_width": 0.2,
        "missingness_rate": 0.2,
        "status_counts": {"completed": 50},
        "validity_counts": {"valid": 40, "unknown": 10},
        "missing_reason_counts": {"judge_no_quorum": 10},
        "service_failures": 0,
        "reference_rate": 0.92,
        "simpler_rate": 0.04,
        "parity_status": "matched_provenance",
        "feedback_mode": "native_feedback",
    }
    cell.update(overrides)
    return cell


def _analysis(cells=None, status="accepted"):
    return {
        "analysis_identity": {"status": status, "reason": None},
        "ledger_state": {"status": "terminal"},
        "endpoint": {"name": "observed_positive_lower_bound"},
        "cells": cells if cells is not None else [_cell()],
        "summary": {"macro": 0.4},
        "reliability": {"status": "computed", "pearson": 0.8},
        "mdd": {"status": "computed", "conservative_mdd": 0.2},
        "mmrv": {"status": "computed"},
        "advanced_inference": {"status": "computed"},
        "lineage_leakage": {"status": "clean"},
    }


def _episodes(count=3):
    rows = []
    for index in range(count):
        rows.append(
            {
                "episode_id": "ep-%d" % index,
                "policy": "OpenVLA",
                "policy_variant": "OpenVLA",
                "task": "close_drawer",
                "start_id": "start-%03d" % index,
                "start_lineage_id": "lineage-%03d" % index,
                "cohort": "primary",
                "protocol_hash": "sha256:" + "a" * 64,
                "status": "completed" if index else "planned",
                "validity": "valid" if index else "unknown",
                "binary_success": True if index else None,
                "progress_score": 5 if index else None,
                "horizon_actions": 70,
                "mode": "qualification/pending-gate-review",
                "timing": {},
                "artifact_refs": {},
            }
        )
    return rows


def test_inventory_lists_every_section_8_artifact(tmp_path):
    store = ArtifactStore(tmp_path)
    inventory = store.inventory()
    for name in SECTION_8_ARTIFACTS:
        assert name in inventory["artifacts"]
    assert "protocol.json" in inventory["artifacts"]
    assert "assets.lock.json" in inventory["artifacts"]
    assert "scenarios.jsonl" in inventory["artifacts"]
    assert inventory["complete"] is False
    assert "leaderboard.md" in inventory["missing"]


def test_per_episode_keeps_planned_rows_and_labels_them_not_run(tmp_path):
    store = ArtifactStore(tmp_path)
    rows = _episodes()
    receipt = store.write_per_episode("run-1", rows)
    assert receipt.rows == 3, "every planned episode is finalized, including the unrun one"
    written = [json.loads(line) for line in receipt.path.read_text().splitlines()]
    planned = [row for row in written if row["status"] == "planned"][0]
    assert planned["missing_reason"] == "not_run"
    assert planned["binary_success"] is None
    assert planned["compute_gpu_seconds"] is None, "an unmeasured value stays null, not zero"


def test_per_episode_refuses_an_empty_run(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactError, match="empty per-episode"):
        store.write_per_episode("run-1", [])


def test_exclusions_asserts_the_bound_width_identity(tmp_path):
    store = ArtifactStore(tmp_path)
    store.write_exclusions(_analysis())
    payload = json.loads(store.exclusions_path.read_text())
    assert payload["convention"]["identity"] == "point-bound width == missing-outcome rate"
    assert payload["cells"][0]["missing_bounds_width"] == 0.2


def test_exclusions_rejects_bounds_that_disagree_with_missingness(tmp_path):
    store = ArtifactStore(tmp_path)
    broken = _analysis([_cell(missing_bounds_width=0.5, missingness_rate=0.2)])
    with pytest.raises(ArtifactError, match="does not equal missingness"):
        store.write_exclusions(broken)


def test_reliability_records_withheld_statistics_rather_than_omitting_them(tmp_path):
    store = ArtifactStore(tmp_path)
    analysis = _analysis()
    analysis["reliability"] = {"status": "not_computed", "reason": "no frozen protocol"}
    store.write_reliability(analysis)
    payload = json.loads(store.reliability_path.read_text())
    assert payload["split_half_reliability"]["status"] == "not_computed"
    assert payload["split_half_reliability"]["reason"] == "no frozen protocol"
    # Absent sweep artifacts are reported, not silently missing.
    assert payload["cost_fidelity"]["status"] == "not_computed"
    assert payload["drift"]["status"] == "not_computed"
    assert any("not establish validity" in note for note in payload["notes"])


def test_leaderboard_marks_a_non_primary_analysis_and_preserves_reference_tables(tmp_path):
    store = ArtifactStore(tmp_path)
    store.write_leaderboard(_analysis(status="rejected"), {"id": "run-1", "config": {"backend": "synthetic"}}, {})
    text = store.leaderboard_path.read_text()
    assert "Not a primary result" in text
    assert "Scientific fields are withheld" in text
    # Both published tables must be reproduced exactly.
    assert "| OpenVLA | 40 | 46 | 1 | 0 | 12 | 99 |" in text
    for policy, row in HUMAN.items():
        assert "| %s |" % policy in text
    assert "SIMPLER has no cloth task" in text
    assert str(SIMPLER["OpenVLA"]["close_drawer"]) in text
    assert "indeterminate" in text.lower()
    assert "never a zero" in text


def test_leaderboard_states_the_reversal_and_refuses_a_permutation_claim(tmp_path):
    store = ArtifactStore(tmp_path)
    store.write_leaderboard(_analysis(), {"id": "run-1"}, {})
    text = store.leaderboard_path.read_text()
    assert "one success in 250" in text
    assert "four of five" in text


def test_load_test_reports_every_rehearsal_and_refuses_best_of_three(tmp_path):
    store = ArtifactStore(tmp_path)
    rehearsals = [
        {"run_id": "a", "passed": True, "elapsed_seconds": 55.0},
        {"run_id": "b", "passed": False, "elapsed_seconds": 90.0},
        {"run_id": "c", "passed": True, "elapsed_seconds": 58.0},
    ]
    store.write_load_test(rehearsals, {"episodes": 1500, "seconds": 60, "usd": 11.25})
    payload = json.loads(store.load_test_path.read_text())
    assert payload["rehearsal_count"] == 3
    assert payload["all_rehearsals_reported"] is True
    assert payload["qualified"] is False, "one failure means the target was not met"
    assert len(payload["rehearsals"]) == 3, "a failed rehearsal is never dropped"


def test_load_test_requires_three_rehearsals_to_qualify(tmp_path):
    store = ArtifactStore(tmp_path)
    store.write_load_test([{"run_id": "a", "passed": True}], {"episodes": 1500})
    payload = json.loads(store.load_test_path.read_text())
    assert payload["qualified"] is False
    assert "fewer than three" in payload["reason"]


def test_economics_says_usd_is_unavailable_without_a_price_basis(tmp_path):
    store = ArtifactStore(tmp_path)
    store.write_economics({"price_basis": None, "allocations": []})
    payload = json.loads(store.economics_path.read_text())
    assert "no verified price basis" in payload["reason"]
    assert "counted once" in payload["views"]["total_demonstration_run_cost"]


def test_gates_writer_is_the_missing_call_site(tmp_path):
    store = ArtifactStore(tmp_path)
    ledger = GateLedger.new("sha256:" + "a" * 64)
    store.write_gates(ledger)
    payload = json.loads(store.gates_path.read_text())
    assert set(payload["gates"]) == {"A", "B", "C", "D", "E", "F"}
    assert all(entry["status"] == "not_run" for entry in payload["gates"].values())


def test_writers_reject_a_non_portable_float(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactError, match="non-finite"):
        store.write_economics({"price_basis": {"currency": "USD"}, "total": float("inf")})
    with pytest.raises(ArtifactError, match="non-finite"):
        _json_safe({"a": float("nan")})


def test_provenance_and_segment_paths_are_sanitised(tmp_path):
    store = ArtifactStore(tmp_path)
    receipt = store.record_provenance("../../escape", {"ok": True})
    assert store.provenance_dir in receipt.path.parents
    assert ".." not in receipt.path.name
    segments = store.segments_dir("../../run")
    assert store.results / "segments" in segments.parents
