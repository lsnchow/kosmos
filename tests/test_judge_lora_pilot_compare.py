"""Offline guards for saved-adapter development-validation comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_compare():
    spec = importlib.util.spec_from_file_location("plumb_judge_lora_pilot_compare", ROOT / "cluster" / "judge_lora_pilot_compare.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rubric_raw() -> str:
    return json.dumps(
        {
            "integrity": "intact",
            "collision": "none_visible",
            "progress": 4,
            "completion_evidence": "not_met",
            "evidence_frame_indices": [0, 15],
            "observable_reasons": "Visible partial drawer closure in the supplied frames.",
        }
    )


def test_compare_keeps_raw_schema_status_and_semantic_tuple_separate_from_accuracy():
    compare = load_compare()
    parsed = compare._parsed(rubric_raw())
    assert parsed["status"] == "schema_valid"
    assert parsed["semantic_tuple"] == ["intact", "none_visible", 4, "not_met"]
    invalid = compare._parsed("not rubric JSON")
    assert invalid["status"] == "schema_invalid"
    assert invalid["label"] is None


def test_selected_adapter_tree_must_be_regular_hash_bound_and_no_comparison_overwrite(tmp_path: Path):
    compare = load_compare()
    root = tmp_path / "pilot-output"
    adapter = root / "adapters" / "epoch-01"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    digest = compare._tree_hash(adapter)
    assert digest.startswith("sha256:")
    assert compare._safe_directory_under(root, str(adapter), "adapter") == adapter
    (adapter / "adapter_config.json").write_text('{"changed":true}', encoding="utf-8")
    assert compare._tree_hash(adapter) != digest
    report = tmp_path / "comparison.json"
    compare._write_exclusive(report, {"qualified": False, "status": "completed_unqualified_adapter_reload_diagnostic"})
    with pytest.raises(compare.CompareError, match="overwrite"):
        compare._write_exclusive(report, {"qualified": False})


def test_each_validation_clip_receives_its_own_no_clobber_raw_output_report(tmp_path: Path):
    compare = load_compare()
    output = tmp_path / "comparison.json"
    report = {
        "pilot_result": {"path": "pilot_result.json", "sha256": "sha256:" + "a" * 64},
        "selected_adapter": {"path": "adapter", "tree_sha256": "sha256:" + "b" * 64},
        "model_binding": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
        "generation": {"do_sample": False, "max_new_tokens": 512},
        "rows": [{"clip_id": "development-01", "base": {"raw_output": "{}"}, "adapter": {"raw_output": "{}"}}],
    }
    written = compare._write_per_clip_reports(output, report)
    assert len(written) == 1
    assert Path(written[0]["path"]).is_file()
    with pytest.raises(compare.CompareError, match="reuse"):
        compare._write_per_clip_reports(output, report)
