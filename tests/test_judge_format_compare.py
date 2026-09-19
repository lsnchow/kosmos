import json
import pytest
from cluster.judge_format_compare import bare_object, compare_outputs, _under


def label(integrity="intact"):
    return json.dumps({"integrity": integrity, "collision": "none_visible", "progress": 5,
                       "completion_evidence": "met", "evidence_frame_indices": [15], "observable_reasons": "Visible end state."})


def test_bare_json_is_distinct_from_schema_valid_fenced_json():
    compared = compare_outputs("```json\n" + label() + "\n```", label())
    assert not compared["base"]["bare_json"]
    assert compared["format_adapter"]["bare_json"]
    assert compared["base"]["status"] == "schema_valid"
    assert compared["semantic_drift"] is False


def test_semantic_drift_is_measured_not_assumed_absent():
    assert compare_outputs(label(), label("artifact"))["semantic_drift"] is True
    missing = compare_outputs("not JSON", label())
    assert missing["semantic_drift"] is None and not missing["semantic_comparable"]


def test_nonstandard_json_constants_and_arrays_are_not_bare_objects():
    assert not bare_object('{"x":NaN}')
    assert not bare_object("[]")


def test_adapter_path_refuses_escape_and_symlink(tmp_path):
    (tmp_path / "adapter").mkdir()
    assert _under(tmp_path, "adapter", directory=True) == tmp_path / "adapter"
    with pytest.raises(ValueError):
        _under(tmp_path, "../other", directory=True)
    (tmp_path / "alias").symlink_to(tmp_path / "adapter")
    with pytest.raises(ValueError, match="symlink"):
        _under(tmp_path, "alias", directory=True)
