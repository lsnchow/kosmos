"""Offline guards for the real-row judge-LoRA preflight path."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "plumb_judge_lora_preflight_entrypoint", ROOT / "deploy" / "baseten" / "training" / "train_judge_lora.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def diagnostic_row() -> dict:
    return {
        "purpose": "framework_preflight_only",
        "qualified": False,
        "source_report_sha256": digest("existing-qwen-report"),
        "teacher_aggregation_status": "unknown",
        "instruction": "Put the pot to the left of the purple item.",
        "rubric": "Use visible evidence only and report the frozen rubric JSON.",
        "frames": ["frames/%02d.png" % index for index in range(16)],
        "frame_timestamps": [float(index) for index in range(16)],
        "reference_images": [
            {
                "path": "reference.png",
                "sha256": digest("reference"),
                "role": "scene",
                "provenance_uri": "artifact://reference.png",
            }
        ],
        "label": {
            "integrity": "uncertain",
            "collision": "visible",
            "progress": 3,
            "completion_evidence": "not_met",
            "evidence_frame_indices": list(range(16)),
            "observable_reasons": "Preserved parsed diagnostic vote; not an aggregate or calibration label.",
        },
    }


def test_preflight_row_requires_an_explicitly_unqualified_unknown_teacher_vote():
    entrypoint = load_entrypoint()
    row = entrypoint.validate_vlm_row(
        diagnostic_row(), split="preflight", index=0, allow_preflight_profile=True, require_dataset_identity=False
    )
    assert row["teacher_aggregation_status"] == "unknown"
    bad = diagnostic_row()
    bad["purpose"] = "training"
    with pytest.raises(entrypoint.TrainingConfigError, match="input_profile"):
        entrypoint.validate_vlm_row(
            bad, split="preflight", index=0, allow_preflight_profile=True, require_dataset_identity=False
        )


def test_training_rows_require_the_serving_video_reference_message_profile():
    entrypoint = load_entrypoint()
    row = {
        **diagnostic_row(),
        "clip_id": "development-clip",
        "source_lineage_id": "development-lineage",
        "cohort": "distillation_train",
        "task": "to_basket",
        "input_profile": entrypoint.SERVING_INPUT_PROFILE,
        "purpose": "training_dataset",
    }
    validated = entrypoint.validate_vlm_row(row, split="train", index=0)
    messages = entrypoint._serving_messages(validated)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"][0]["type"] == "video"
    assert messages[1]["content"][1]["type"] == "image"
    assert "nominal control timestamps" in messages[1]["content"][-1]["text"]


def test_batch_size_is_fixed_to_one_but_gradient_accumulation_is_explicitly_configurable():
    entrypoint = load_entrypoint()
    assert entrypoint.training_options({"training": {"gradient_accumulation_steps": 4}}) == {
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
    }
    with pytest.raises(entrypoint.TrainingConfigError, match="batch size 1"):
        entrypoint.training_options({"training": {"per_device_train_batch_size": 2}})


def test_collator_masks_attention_padding_and_media_not_the_assistant_eos_token():
    source = (ROOT / "deploy" / "baseten" / "training" / "train_judge_lora.py").read_text(encoding="utf-8")
    assert "labels[full_mask == 0] = -100" in source
    assert "all_special_ids" not in source
    assert "pad_token_id" not in source


def test_preflight_config_deliberately_needs_no_formal_dataset_or_preregistered_search():
    entrypoint = load_entrypoint()
    config = entrypoint.validate_preflight_config(
        {
            "purpose": "framework_preflight_only",
            "student_model": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": "a" * 40},
            "hyperparameters": {
                "lora_r": 64,
                "lora_alpha": 32,
                "learning_rate": 0.001,
                "num_train_epochs": 2,
                "lora_target_modules": ["q_proj"],
            },
            "training": {
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
            },
        }
    )
    assert "datasets" not in config


def test_local_model_manifest_must_bind_the_configured_model_and_revision(tmp_path: Path):
    entrypoint = load_entrypoint()
    root = tmp_path / "model"
    root.mkdir()
    config_file = root / "config.json"
    config_file.write_text("{}", encoding="utf-8")
    manifest = root / "PLUMB-TRAINING-MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "model_revision": "a" * 40,
                "file_records": [
                    {
                        "path": "config.json",
                        "bytes": config_file.stat().st_size,
                        "sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    binding = entrypoint._local_model_binding(
        str(root), str(manifest), {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": "a" * 40}
    )
    assert binding["model_manifest_sha256"].startswith("sha256:")
    config_file.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(entrypoint.TrainingConfigError, match="manifest"):
        entrypoint._local_model_binding(
            str(root), str(manifest), {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": "a" * 40}
        )
    with pytest.raises(entrypoint.TrainingConfigError, match="revision"):
        entrypoint._local_model_binding(
            str(root), str(manifest), {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": "c" * 40}
        )


def test_training_output_directory_is_fresh_disjoint_and_result_write_once(tmp_path: Path):
    entrypoint = load_entrypoint()
    dataset = tmp_path / "dataset"
    model = tmp_path / "model"
    dataset.mkdir()
    model.mkdir()
    output = entrypoint._claim_fresh_output_dir(str(tmp_path / "outputs" / "arm-a"), dataset_dir=str(dataset), model_root=str(model))
    assert output.is_dir()
    with pytest.raises(entrypoint.TrainingConfigError, match="already exists|concurrently"):
        entrypoint._claim_fresh_output_dir(str(output), dataset_dir=str(dataset), model_root=str(model))
    with pytest.raises(entrypoint.TrainingConfigError, match="overlap"):
        entrypoint._claim_fresh_output_dir(str(dataset / "bad-output"), dataset_dir=str(dataset), model_root=str(model))
    result_path = entrypoint.write_result(str(output), {"status": "completed", "metric_value": 0.5})
    assert Path(result_path).is_file()
    with pytest.raises(entrypoint.TrainingConfigError, match="overwriting"):
        entrypoint.write_result(str(output), {"status": "completed", "metric_value": 0.4})
