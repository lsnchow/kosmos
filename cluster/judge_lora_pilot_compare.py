#!/usr/bin/env python3
"""Reload and pairwise-compare an unqualified saved judge-LoRA pilot adapter.

The comparison is an engineering diagnostic only.  It proves that the selected
adapter can be reloaded from its immutable on-disk checkpoint and can generate
paired outputs over the accepted *development-validation* clips.  Agreement
with an uncalibrated teacher label is not accuracy, a human-quality claim,
Gate-D evidence, formal selection, or permission to score a burst.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PURPOSE = "uncalibrated_judge_distillation_pilot_adapter_compare"
SEED = 20260919
MAX_NEW_TOKENS = 512
_HEX = set("0123456789abcdef")


class CompareError(ValueError):
    """A saved pilot/result binding is unsafe for adapter reload comparison."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CompareError("%s must be a SHA-256 digest" % label)
    text = value.strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise CompareError("%s must be a SHA-256 digest" % label)
    return "sha256:" + text


def _load_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompareError("%s must be a regular file" % label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompareError("unable to read %s" % label) from error
    if not isinstance(value, Mapping):
        raise CompareError("%s must be a JSON object" % label)
    return dict(value)


def _safe_directory_under(root: Path, candidate: Any, label: str) -> Path:
    if not isinstance(candidate, str) or not candidate:
        raise CompareError("%s must be a path" % label)
    path = Path(candidate)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise CompareError("%s escapes the pilot output root" % label) from error
    current = root.resolve(strict=True)
    for part in resolved.relative_to(current).parts:
        current = current / part
        if current.is_symlink():
            raise CompareError("%s contains a symlink" % label)
    if not resolved.is_dir():
        raise CompareError("%s is not a regular directory" % label)
    return resolved


def _tree_hash(directory: Path) -> str:
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise CompareError("adapter checkpoint contains a symlink: %s" % path)
        if path.is_file():
            records.append({"path": str(path.relative_to(directory)), "sha256": _sha256_file(path), "bytes": path.stat().st_size})
    if not records:
        raise CompareError("adapter checkpoint has no regular files")
    return "sha256:" + hashlib.sha256(_canonical(records)).hexdigest()


def load_bindings(pilot_result_path: Path, dataset_root: Path, model_root: Path, model_manifest: Path) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    """Validate the immutable pilot result/checkpoint and local base binding."""

    from deploy.baseten.training import train_judge_lora as trainer
    from cluster import judge_lora_pilot as pilot

    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise CompareError("--dataset-root must be an existing non-symlink directory")
    result = _load_object(pilot_result_path, "pilot result")
    if result.get("purpose") != pilot.PILOT_PURPOSE or result.get("qualified") is not False:
        raise CompareError("pilot result is not explicitly unqualified pilot evidence")
    if result.get("status") != "completed_unqualified_exploratory_pilot":
        raise CompareError("pilot result did not complete the unqualified exploratory loop")
    pilot_root = pilot_result_path.parent.resolve(strict=True)
    checkpoint = result.get("selected_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise CompareError("pilot result lacks selected_checkpoint")
    adapter_dir = _safe_directory_under(pilot_root, checkpoint.get("adapter_path"), "selected_checkpoint.adapter_path")
    if _tree_hash(adapter_dir) != _digest(checkpoint.get("adapter_tree_sha256"), "selected_checkpoint.adapter_tree_sha256"):
        raise CompareError("selected adapter tree no longer matches its pilot result hash")
    file_hashes = result.get("dataset_files_sha256")
    if not isinstance(file_hashes, Mapping):
        raise CompareError("pilot result lacks frozen dataset file hashes")
    _, current_file_hashes, current_lineage_hash, _ = pilot.load_pilot_dataset(dataset_root)
    if dict(file_hashes) != current_file_hashes or result.get("source_lineages_sha256") != current_lineage_hash:
        raise CompareError("dataset files or reserved source lineages changed since the pilot")
    student = result.get("student_model")
    if not isinstance(student, Mapping):
        raise CompareError("pilot result lacks student model binding")
    binding = trainer._local_model_binding(str(model_root), str(model_manifest), student)
    recorded_binding = result.get("model_binding")
    if not isinstance(recorded_binding, Mapping):
        raise CompareError("pilot result lacks model binding")
    for field in ("model_manifest_sha256", "model_id", "model_revision"):
        if recorded_binding.get(field) != binding.get(field):
            raise CompareError("local base %s does not match the pilot result" % field)
    return result, adapter_dir, binding


def _move_to_cuda(inputs: Any) -> Any:
    if isinstance(inputs, Mapping):
        return {key: value.to("cuda") if callable(getattr(value, "to", None)) else value for key, value in inputs.items()}
    move = getattr(inputs, "to", None)
    return move("cuda") if callable(move) else inputs


def _encode_row(processor: Any, row: Mapping[str, Any]) -> Any:
    """Encode the exact serving diagnostic system/user video/reference layout."""

    from PIL import Image
    from deploy.baseten.training import train_judge_lora as trainer

    messages = trainer._serving_messages(row)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    frames = [Image.open(path).convert("RGB") for path in row["frames"]]
    references = [Image.open(reference["path"]).convert("RGB") for reference in row["reference_images"]]
    return processor(text=[prompt], images=[references], videos=[frames], padding=True, return_tensors="pt")


def _generate(model: Any, processor: Any, row: Mapping[str, Any], torch_module: Any) -> Tuple[str, Dict[str, Any]]:
    torch_module.manual_seed(SEED)
    torch_module.cuda.manual_seed_all(SEED)
    encoded = _move_to_cuda(_encode_row(processor, row))
    prompt_tokens = int(encoded["input_ids"].shape[1])
    torch_module.cuda.synchronize()
    started = time.monotonic()
    with torch_module.inference_mode():
        generated = model.generate(**encoded, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    torch_module.cuda.synchronize()
    continuation = generated[:, prompt_tokens:]
    raw = processor.batch_decode(continuation, skip_special_tokens=True)[0]
    return raw, {"wall_seconds": time.monotonic() - started, "prompt_tokens": prompt_tokens, "generated_tokens": int(continuation.shape[1])}


def _parsed(raw: str) -> Dict[str, Any]:
    from plumb.policies.judge import parse_rubric_json

    try:
        label = dict(parse_rubric_json(raw).as_dict())
    except Exception as error:
        return {"status": "schema_invalid", "error": type(error).__name__, "label": None, "semantic_tuple": None}
    return {
        "status": "schema_valid",
        "error": None,
        "label": label,
        "semantic_tuple": [label["integrity"], label["collision"], label["progress"], label["completion_evidence"]],
    }


def _gpu_identity() -> str:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) != 1:
        raise CompareError("comparison requires exactly one CUDA_VISIBLE_DEVICES allocation")
    try:
        result = subprocess.run(["nvidia-smi", "--id=" + visible[0], "--query-gpu=uuid,name", "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10)
        value = result.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        raise CompareError("unable to record comparison GPU UUID") from error
    if not value:
        raise CompareError("comparison GPU UUID is empty")
    return value


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise CompareError("refusing to overwrite comparison report") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _write_per_clip_reports(output: Path, report: Mapping[str, Any]) -> Sequence[Dict[str, str]]:
    """Persist each paired raw-output record independently without overwrite."""

    directory = output.parent / (output.stem + "-clips")
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError as error:
        raise CompareError("refusing to reuse per-clip comparison directory") from error
    shared = {
        "schema_version": 1,
        "kind": "plumb_unqualified_judge_lora_adapter_reload_clip",
        "purpose": PURPOSE,
        "qualified": False,
        "pilot_result": report["pilot_result"],
        "selected_adapter": report["selected_adapter"],
        "model_binding": report["model_binding"],
        "cohort": "development_validation_only",
        "seed": SEED,
        "generation": report["generation"],
        "human_quality_claim": False,
    }
    written = []
    for row in report["rows"]:
        clip_id = row.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id or "/" in clip_id or ".." in clip_id:
            raise CompareError("comparison row has an unsafe clip_id")
        path = directory / (clip_id + ".json")
        _write_exclusive(path, {**shared, "row": row})
        written.append({"clip_id": clip_id, "path": str(path), "sha256": _sha256_file(path)})
    return written


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise CompareError("adapter comparison requires an allocated Slurm GPU job")
    from deploy.baseten.training import train_judge_lora as trainer

    raw_result = Path(args.pilot_result)
    raw_dataset = Path(args.dataset_root)
    raw_model = Path(args.model_root)
    raw_manifest = Path(args.model_manifest)
    raw_output = Path(args.output)
    if any(path.is_symlink() for path in (raw_result, raw_dataset, raw_model, raw_manifest)):
        raise CompareError("comparison inputs may not be symlinks")
    if raw_output.exists() or raw_output.is_symlink() or (raw_output.parent / (raw_output.stem + "-clips")).exists():
        raise CompareError("comparison output or per-clip directory already exists; overwrite/reuse is forbidden")
    pilot_result_path = raw_result.resolve(strict=True)
    dataset_root = raw_dataset.resolve(strict=True)
    result, adapter_dir, binding = load_bindings(pilot_result_path, dataset_root, raw_model, raw_manifest)
    from cluster import judge_lora_pilot as pilot
    splits, _, _, _ = pilot.load_pilot_dataset(dataset_root)
    validation_rows = splits["development_validation"]
    if not validation_rows:
        raise CompareError("no accepted development-validation clips are available for comparison")
    modules = trainer._load_frameworks()
    torch_module, transformers, peft = modules["torch"], modules["transformers"], modules["peft"]
    if not bool(torch_module.cuda.is_available()):
        raise CompareError("comparison requires CUDA; no CPU substitute is permitted")
    student = result["student_model"]
    torch_module.manual_seed(SEED)
    torch_module.cuda.manual_seed_all(SEED)
    load_kwargs = {"revision": student["model_revision"], "local_files_only": True, "trust_remote_code": False, "torch_dtype": torch_module.bfloat16, "use_safetensors": True}
    processor = transformers.AutoProcessor.from_pretrained(binding["model_root"], **{key: value for key, value in load_kwargs.items() if key != "torch_dtype"})
    base_model = transformers.AutoModelForVision2Seq.from_pretrained(binding["model_root"], **load_kwargs)
    adapter_model = peft.PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    adapter_model = adapter_model.to("cuda")
    adapter_model.eval()
    disable_adapter = getattr(adapter_model, "disable_adapter", None)
    if not callable(disable_adapter):
        raise CompareError("reloaded PeftModel does not expose disable_adapter for paired base inference")
    torch_module.cuda.reset_peak_memory_stats()
    rows_report = []
    started = time.monotonic()
    for row in validation_rows:
        with disable_adapter():
            base_raw, base_timing = _generate(adapter_model, processor, row, torch_module)
        adapter_raw, adapter_timing = _generate(adapter_model, processor, row, torch_module)
        base = _parsed(base_raw)
        adapted = _parsed(adapter_raw)
        rows_report.append(
            {
                "clip_id": row["clip_id"],
                "source_lineage_id": row["source_lineage_id"],
                "cohort": row["cohort"],
                "timestamp_semantics": row.get("timestamp_semantics"),
                "input_profile": row["input_profile"],
                "frame_sha256": list(row["frame_sha256"]),
                "reference_images": [dict(reference) for reference in row["reference_images"]],
                "raw_teacher_refs": [dict(reference) for reference in row["raw_teacher_refs"]],
                "base": {"raw_output": base_raw, **base, "timing": base_timing},
                "adapter": {"raw_output": adapter_raw, **adapted, "timing": adapter_timing},
                "semantic_agree": base["status"] == "schema_valid" and adapted["status"] == "schema_valid" and base["semantic_tuple"] == adapted["semantic_tuple"],
                "uncalibrated_teacher_label": row["label"],
                "teacher_label_is_not_accuracy_or_human_qualification": True,
            }
        )
    both_valid = [row for row in rows_report if row["base"]["status"] == "schema_valid" and row["adapter"]["status"] == "schema_valid"]
    report = {
        "schema_version": 1,
        "kind": "plumb_unqualified_judge_lora_adapter_reload_comparison",
        "purpose": PURPOSE,
        "qualified": False,
        "status": "completed_unqualified_adapter_reload_diagnostic",
        "reason": "Reloaded saved adapter paired with disable_adapter baseline only on accepted development-validation clips; agreement is an uncalibrated fit diagnostic, not accuracy/calibration/formal selection/Gate evidence.",
        "pilot_result": {"path": str(pilot_result_path), "sha256": _sha256_file(pilot_result_path)},
        "selected_adapter": {"path": str(adapter_dir), "tree_sha256": _tree_hash(adapter_dir)},
        "model_binding": binding,
        "dataset_files_sha256": result["dataset_files_sha256"],
        "source_lineages_sha256": result["source_lineages_sha256"],
        "cohort": "development_validation_only",
        "seed": SEED,
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "serving_message_layout": True},
        "rows": rows_report,
        "counts": {"development_validation_clips": len(rows_report), "both_schema_valid": len(both_valid), "base_adapter_semantic_agree": sum(1 for row in rows_report if row["semantic_agree"])},
        "gpu_peak_memory_bytes": int(torch_module.cuda.max_memory_allocated()),
        "gpu_identity": _gpu_identity(),
        "wall_seconds": time.monotonic() - started,
        "timing_scope": "adapter reload plus paired greedy base/adapter generation over accepted development-validation clips only",
        "runtime_versions": {name: str(getattr(module, "__version__", "")) for name, module in modules.items() if name in trainer.EXPECTED_RUNTIME_VERSIONS},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
        "filled_gates": [],
        "human_quality_claim": False,
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-result", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args)
        output = Path(args.output)
        report["per_clip_reports"] = _write_per_clip_reports(output, report)
        _write_exclusive(output, report)
        print(json.dumps({"status": report["status"], "qualified": False, "output": args.output}, sort_keys=True))
        return 0
    except (CompareError, ValueError, OSError) as error:
        print(json.dumps({"status": "blocked", "qualified": False, "reason": str(error), "traceback": traceback.format_exc()}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
