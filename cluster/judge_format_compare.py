"""Reload a format-only adapter and measure JSON shape and semantic drift.

No human accuracy, calibration, teacher-agreement or production claim is made.
All four original development-validation inputs are paired under greedy decode.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from cluster import judge_format_pilot as pilot
from cluster import judge_lora_pilot_compare as comparison
from deploy.baseten.training import train_judge_lora as training


def _sha(path):
    return "sha256:" + pilot._sha(Path(path))


def _normalise(value):
    if not isinstance(value, str):
        raise ValueError("missing digest")
    value = value.removeprefix("sha256:")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid digest")
    return "sha256:" + value


def _under(root, relative, directory=False):
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("artifact must remain relative to immutable output root")
    current = root
    for name in part.parts:
        current = current / name
        if current.is_symlink():
            raise ValueError("artifact may not contain symlinks")
    current.resolve(strict=True).relative_to(root.resolve(strict=True))
    if not (current.is_dir() if directory else current.is_file()):
        raise ValueError("artifact is not a regular file/directory")
    return current


def bare_object(raw):
    def reject_constant(value):
        raise ValueError(value)
    try:
        return isinstance(json.loads(raw.strip(), parse_constant=reject_constant), dict)
    except (ValueError, TypeError, AttributeError):
        return False


def compare_outputs(base_raw, adapter_raw):
    base = comparison._parsed(base_raw)
    adapter = comparison._parsed(adapter_raw)
    comparable = base["status"] == adapter["status"] == "schema_valid"
    return {
        "base": {**base, "raw_output": base_raw, "bare_json": bare_object(base_raw)},
        "format_adapter": {**adapter, "raw_output": adapter_raw, "bare_json": bare_object(adapter_raw)},
        "semantic_comparable": comparable,
        "semantic_drift": base["semantic_tuple"] != adapter["semantic_tuple"] if comparable else None,
    }


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("allocated offline GPU required")
    destination = Path(args.output)
    if destination.exists() or destination.is_symlink():
        raise ValueError("comparison output already exists")
    result_path = Path(args.training_result)
    if result_path.is_symlink():
        raise ValueError("training result must be regular")
    result = json.loads(result_path.read_text())
    if result.get("purpose") != pilot.PURPOSE or result.get("qualified") is not False or result.get("status") != "completed_unqualified_structure_only_pilot":
        raise ValueError("requires completed unqualified format-only optimizer result")
    if result.get("optimizer_steps", 0) <= 0:
        raise ValueError("no recorded optimizer steps")
    root = result_path.parent.resolve(strict=True)
    adapter = _under(root, result["selected_checkpoint"]["adapter_path"], directory=True)
    if training._digest_directory(adapter) != _normalise(result["selected_checkpoint"]["adapter_tree_sha256"]):
        raise ValueError("saved format adapter hash mismatch")
    manifest = _under(root, result["input_manifest"]["path"])
    if _sha(manifest) != _normalise(result["input_manifest"]["sha256"]):
        raise ValueError("format input manifest hash mismatch")
    splits = pilot.load_input_manifest(manifest)
    if len(splits["train"]) != 12 or len(splits["development_validation"]) != 4:
        raise ValueError("original 12/4 assignment changed")
    if len({row["source_lineage_id"] for rows in splits.values() for row in rows}) != 16:
        raise ValueError("source lineages overlap")
    old_binding = result["model_binding"]
    binding = training._local_model_binding(args.model_root, args.model_manifest, old_binding)
    if any(binding[key] != old_binding[key] for key in ("model_id", "model_revision", "model_manifest_sha256")):
        raise ValueError("base model identity differs from training")
    modules = training._load_frameworks()
    torch, transformers, peft = (modules[name] for name in ("torch", "transformers", "peft"))
    if not torch.cuda.is_available():
        raise ValueError("allocated CUDA device required")
    torch.manual_seed(pilot.SEED)
    torch.cuda.manual_seed_all(pilot.SEED)
    processor = transformers.AutoProcessor.from_pretrained(binding["model_root"], local_files_only=True, trust_remote_code=False)
    base = transformers.AutoModelForVision2Seq.from_pretrained(binding["model_root"], local_files_only=True,
                                                             trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16)
    model = peft.PeftModel.from_pretrained(base, str(adapter), is_trainable=False, local_files_only=True).to("cuda")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rows = []
    for row in splits["development_validation"]:
        with model.disable_adapter():
            base_raw, base_timing = comparison._generate(model, processor, row, torch)
        adapter_raw, adapter_timing = comparison._generate(model, processor, row, torch)
        compared = compare_outputs(base_raw, adapter_raw)
        compared.update(clip_id=row["clip_id"], source_lineage_id=row["source_lineage_id"],
                        base_timing=base_timing, adapter_timing=adapter_timing)
        rows.append(compared)
    payload = {
        "kind": "plumb_judge_format_compare_v1", "purpose": pilot.PURPOSE, "qualified": False,
        "status": "completed_unqualified_format_adapter_comparison", "training_result_sha256": _sha(result_path),
        "input_manifest_sha256": _sha(manifest), "selected_checkpoint": result["selected_checkpoint"],
        "model_binding": binding, "rows": rows,
        "bare_json_counts": {name: sum(r[name]["bare_json"] for r in rows) for name in ("base", "format_adapter")},
        "schema_valid_counts": {name: sum(r[name]["status"] == "schema_valid" for r in rows) for name in ("base", "format_adapter")},
        "semantic_comparable_count": sum(r["semantic_comparable"] for r in rows),
        "semantic_drift_count": sum(r["semantic_drift"] is True for r in rows),
        "generation": {"do_sample": False, "max_new_tokens": 512, "seed": pilot.SEED},
        "wall_seconds": time.monotonic() - started, "timing_scope": "paired_greedy_generation_excludes_load",
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()), "gpu_identity": comparison._gpu_identity(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
        "limitations": ["Syntax-only loss does not guarantee semantic preservation; drift is measured only on four development inputs.",
                        "No human accuracy, semantic calibration, or production-scoring claim."], "filled_gates": [],
    }
    comparison._write_exclusive(destination, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("--training-result", "--model-root", "--model-manifest", "--output"):
        parser.add_argument(name, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps({key: result[key] for key in ("status", "bare_json_counts", "schema_valid_counts", "semantic_drift_count")}, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "qualified": False, "error": str(error), "traceback": traceback.format_exc()}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
