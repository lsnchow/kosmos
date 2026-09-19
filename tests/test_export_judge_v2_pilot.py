"""Eligibility guards for fresh unqualified teacher-v2 pilot export."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_exporter():
    spec = importlib.util.spec_from_file_location("plumb_export_judge_v2_pilot", ROOT / "cluster" / "export_judge_v2_pilot.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accepted(label, count=3):
    semantic = [label["integrity"], label["collision"], label["progress"], label["completion_evidence"]]
    return {"status": "accepted_unique_modal_tuple", "modal_count": count, "selected": {"label": label, "semantic_tuple": semantic, "sample_index": 0}}


def excluded():
    return {"status": "excluded_modal_tie", "modal_count": 2}


def candidates():
    return {
        "clip-%02d" % index: {"clip_id": "clip-%02d" % index, "source_lineage_id": "lineage-%02d" % index, "cohort": "train" if index < 12 else "development_validation"}
        for index in range(16)
    }


def reports_for(selection_by_clip):
    return {clip_id: (Path("/v2") / (clip_id + ".json"), {}, selection) for clip_id, selection in selection_by_clip.items()}


def test_export_requires_full_raw_v2_coverage_minimum_splits_and_train_semantic_diversity():
    exporter = load_exporter()
    labels = {
        "clip-00": accepted({"integrity": "intact", "collision": "none_visible", "progress": 2, "completion_evidence": "not_met", "evidence_frame_indices": [0], "observable_reasons": "a"}),
        "clip-01": accepted({"integrity": "intact", "collision": "visible", "progress": 3, "completion_evidence": "not_met", "evidence_frame_indices": [1], "observable_reasons": "b"}),
        "clip-12": accepted({"integrity": "intact", "collision": "none_visible", "progress": 2, "completion_evidence": "not_met", "evidence_frame_indices": [0], "observable_reasons": "c"}),
    }
    selections = {clip_id: labels.get(clip_id, excluded()) for clip_id in candidates()}
    decision = exporter.eligibility(candidates(), reports_for(selections))
    assert decision["eligible"] is True
    assert decision["counts"]["accepted_train_distinct_semantic_tuples"] == 2


def test_homogeneous_or_incomplete_v2_results_create_only_an_ineligible_audit():
    exporter = load_exporter()
    label = {"integrity": "artifact", "collision": "visible", "progress": 5, "completion_evidence": "met", "evidence_frame_indices": [0], "observable_reasons": "same"}
    selections = {clip_id: accepted(label) for clip_id in candidates()}
    decision = exporter.eligibility(candidates(), reports_for(selections))
    assert decision["eligible"] is False
    assert "accepted_train_semantic_tuple_diversity_below_2" in decision["blockers"]
    partial = exporter.eligibility(candidates(), reports_for({"clip-00": accepted(label)}))
    assert "completed_v2_raw_reports_below_16" in partial["blockers"]
