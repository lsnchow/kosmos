"""Offline safety checks for the explicitly unqualified judge-LoRA pilot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def load_pilot():
    spec = importlib.util.spec_from_file_location("plumb_judge_lora_pilot", ROOT / "cluster" / "judge_lora_pilot.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(pilot, clip_id: str, lineage: str, cohort: str, frame_hash: str, reference_hash: str) -> dict:
    return {
        "purpose": pilot.PILOT_PURPOSE,
        "qualified": False,
        "label_source": pilot.PILOT_LABEL_SOURCE,
        "input_profile": "qwen_rubric_serving_messages_v1",
        "clip_id": clip_id,
        "source_lineage_id": lineage,
        "cohort": cohort,
        "task": "close_drawer",
        "timestamp_semantics": "unverified_nominal_dataset_times",
        "instruction": "Close the drawer",
        "rubric": "Judge the supplied 16-frame video using visible evidence only.",
        "frames": ["media/frame-%02d.bin" % index for index in range(16)],
        "frame_sha256": [frame_hash] * 16,
        "frame_timestamps": [float(index) for index in range(16)],
        "reference_images": [{"path": "media/reference.bin", "sha256": reference_hash, "role": "scene", "provenance_uri": "artifact://scene"}],
        "label": {"integrity": "uncertain", "collision": "none_visible", "progress": 3, "completion_evidence": "not_met", "evidence_frame_indices": [0, 15], "observable_reasons": "Preserved uncalibrated teacher sample."},
        "raw_teacher_refs": [{"uri": "raw/teacher-report.json", "sha256": digest_bytes(b"raw-report"), "sample_index": 0}],
    }


def write_dataset(root: Path, pilot):
    media = root / "media"
    media.mkdir()
    raw_dir = root / "raw"
    raw_dir.mkdir()
    (raw_dir / "teacher-report.json").write_bytes(b"raw-report")
    frame = b"verified-frame"
    for index in range(16):
        (media / ("frame-%02d.bin" % index)).write_bytes(frame)
    reference = b"verified-reference"
    (media / "reference.bin").write_bytes(reference)
    frame_hash, reference_hash = digest_bytes(frame), digest_bytes(reference)
    train_rows = [row(pilot, "train-%d" % index, "lineage-train-%d" % index, pilot.TRAIN_COHORT, frame_hash, reference_hash) for index in range(2)]
    validation_rows = [row(pilot, "validation-0", "lineage-validation-0", pilot.VALIDATION_COHORT, frame_hash, reference_hash)]
    (root / pilot.TRAIN_FILE).write_text("".join(json.dumps(item) + "\n" for item in train_rows), encoding="utf-8")
    (root / pilot.VALIDATION_FILE).write_text("".join(json.dumps(item) + "\n" for item in validation_rows), encoding="utf-8")
    train_sha = digest_bytes((root / pilot.TRAIN_FILE).read_bytes())
    validation_sha = digest_bytes((root / pilot.VALIDATION_FILE).read_bytes())
    candidates = [
        {"clip_id": item["clip_id"], "source_lineage_id": item["source_lineage_id"], "split": "train", "status": "accepted"}
        for item in train_rows
    ] + [
        {"clip_id": validation_rows[0]["clip_id"], "source_lineage_id": validation_rows[0]["source_lineage_id"], "split": "development_validation", "status": "accepted"}
    ]
    for index in range(13):
        candidates.append({"clip_id": "excluded-%02d" % index, "source_lineage_id": "lineage-excluded-%02d" % index, "split": "train" if index < 10 else "development_validation", "status": "excluded", "exclusion_reason": "no unique modal teacher label"})
    source_hash = digest_bytes(b"source-report")
    (root / pilot.INPUTS_FILE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "schema": "plumb-judge-teacher-pilot-inputs-v1",
                "purpose": pilot.PILOT_PURPOSE,
                "qualified": False,
                "candidates": candidates,
                "raw_source_hashes": {
                    "candidate_manifest": {"uri": "candidate-inputs.json", "sha256": source_hash},
                    "model_manifest": {"uri": "model-manifest.json", "sha256": source_hash},
                    "runtime_lock": {"uri": "runtime-lock.json", "sha256": source_hash},
                    "collection_config_sha256": source_hash,
                    "teacher_reports": {candidate["clip_id"]: {"uri": "raw/teacher-report.json", "sha256": digest_bytes(b"raw-report")} for candidate in candidates},
                    "labels": {"uri": "teacher-pilot-labels.jsonl", "sha256": source_hash},
                    "exclusions": {"uri": "teacher-pilot-exclusions.jsonl", "sha256": source_hash},
                    "splits": {
                        "train": {"uri": pilot.TRAIN_FILE, "sha256": train_sha, "count": 2},
                        "development_validation": {"uri": pilot.VALIDATION_FILE, "sha256": validation_sha, "count": 1},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return candidates


def test_pilot_dataset_binds_all_candidates_but_accepts_only_unique_modal_rows(tmp_path: Path):
    pilot = load_pilot()
    candidates = write_dataset(tmp_path, pilot)
    splits, hashes, lineage_hash, bound_candidates = pilot.load_pilot_dataset(tmp_path)
    assert len(splits["train"]) == 2
    assert len(splits["development_validation"]) == 1
    assert pilot.INPUTS_FILE in hashes
    assert lineage_hash.startswith("sha256:")
    assert set(bound_candidates) == {candidate["clip_id"] for candidate in candidates}
    output = tmp_path / "output"
    output.mkdir()
    config = {"pilot_id": "pilot-v1"}
    pilot._freeze_output(output, config, digest_bytes(b"config"), hashes, lineage_hash, splits, bound_candidates)
    exclusions = json.loads((output / "future_formal_exclusion_lineages.json").read_text(encoding="utf-8"))
    assert len(exclusions["lineages"]) == 16


def test_pilot_rejects_any_jsonl_row_not_accepted_by_the_frozen_inputs_manifest(tmp_path: Path):
    pilot = load_pilot()
    write_dataset(tmp_path, pilot)
    train = tmp_path / pilot.TRAIN_FILE
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    rows[0]["clip_id"] = "excluded-00"
    train.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    try:
        pilot.load_pilot_dataset(tmp_path)
    except pilot.PilotError as error:
        assert "split hash" in str(error) or "accepted" in str(error)
    else:
        raise AssertionError("unaccepted frozen candidate was admitted to pilot JSONL")


def test_pilot_rows_are_rejected_by_the_formal_training_loader_cohort_guard(tmp_path: Path):
    pilot = load_pilot()
    write_dataset(tmp_path, pilot)
    from deploy.baseten.training import train_judge_lora as formal

    try:
        formal.load_split(str(tmp_path), "train")
    except formal.TrainingConfigError as error:
        assert "cohort" in str(error)
    else:
        raise AssertionError("uncalibrated pilot row entered the formal training loader")


def test_pilot_config_is_exactly_bounded_and_cannot_reuse_formal_search_settings(tmp_path: Path):
    pilot = load_pilot()
    payload = {
        "schema_version": 1,
        "purpose": pilot.PILOT_PURPOSE,
        "qualified": False,
        "pilot_id": "close-drawer-teacher-pilot-v1",
        "student_model": {"model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": "a" * 40},
        "lora": {"r": 64, "alpha": 32, "learning_rate": 1e-4, "target_modules": ["q_proj", "v_proj"], "epochs": 2},
        "training": {"batch_size": 1, "gradient_accumulation_steps": 1, "seed": 20260919},
    }
    path = tmp_path / "pilot-config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config, config_hash = pilot.load_pilot_config(path)
    assert config["pilot_id"] == payload["pilot_id"]
    assert config_hash.startswith("sha256:")
    payload["lora"]["learning_rate"] = 1e-3
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        pilot.load_pilot_config(path)
    except pilot.PilotError as error:
        assert "r64" in str(error)
    else:
        raise AssertionError("formal-search learning rate was admitted to the bounded pilot")
