"""V2 teacher profile stays isolated and rejects sub-quorum unique modes."""

from __future__ import annotations

import json
import pytest

from cluster import judge_teacher_v2 as v2


def _raw(label):
    return json.dumps({"integrity": label[0], "collision": label[1], "progress": label[2], "completion_evidence": label[3], "evidence_frame_indices": [0], "observable_reasons": "visible evidence"})


def _report(labels):
    samples = []
    for index, label in enumerate(labels):
        raw = _raw(label)
        samples.append({"sample_index": index, "attempts": [{"sample_index": index, "attempt_index": 0, "raw_output": raw, "parsed": json.loads(raw), "failure_reason": None}]})
    return {"raw_judge_samples": samples}


def test_v2_excludes_unique_two_of_five_mode_without_constructing_a_label():
    modal = ("intact", "none_visible", 4, "not_met")
    result = v2.select_v2(_report([modal, modal, ("artifact", "visible", 0, "not_met"), ("uncertain", "uncertain", None, "uncertain"), ("intact", "none_visible", 3, "not_met")]))
    assert result["status"] == "excluded_unique_mode_below_three"
    assert "selected" not in result
    assert result["profile_id"] == v2.PROFILE_ID


def test_v2_keeps_an_actual_lowest_modal_member_when_three_of_five_agree():
    modal = ("intact", "none_visible", 4, "not_met")
    result = v2.select_v2(_report([modal, ("artifact", "visible", 0, "not_met"), modal, modal, ("uncertain", "uncertain", None, "uncertain")]))
    assert result["status"] == "accepted_unique_modal_tuple"
    assert result["selected"]["sample_index"] == 0
    assert result["selected"]["label"]["progress"] == 4
    assert "integrity=artifact" in v2.v2_rubric()


def test_diagnostic_output_binds_input_and_profile_before_resume(tmp_path):
    output = tmp_path / "v2"
    freeze = {"profile": "v2", "input": "bound", "qualified": False}
    v2._bind_output(output, freeze)
    v2._bind_output(output, freeze)
    with pytest.raises(RuntimeError, match="different"):
        v2._bind_output(output, {**freeze, "input": "different"})
    assert json.loads((output / "freeze.json").read_text()) == freeze
