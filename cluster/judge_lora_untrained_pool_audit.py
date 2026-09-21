#!/usr/bin/env python3
"""Paired base/LoRA output audit on clips excluded before pilot training.

This is a post-hoc engineering audit, not human-label accuracy, calibration,
or a held-out benchmark. Each source clip was in the original pilot collection
but never became a training or validation target because the v1 teacher tied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PURPOSE = "untrained_pilot_pool_base_adapter_output_audit_v1"
SEED = 20260919
MAX_NEW_TOKENS = 512


class AuditError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AuditError("%s must be a regular file" % name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError("cannot read %s" % name) from error
    if not isinstance(value, Mapping):
        raise AuditError("%s must be an object" % name)
    return value


def _under(root: Path, relative: str, name: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AuditError("%s has unsafe path" % name)
    try:
        path = (root / relative).resolve(strict=True)
        path.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise AuditError("%s escapes dataset root" % name) from error
    if path.is_symlink() or not path.is_file():
        raise AuditError("%s is not a regular file" % name)
    return path


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise AuditError("%s lacks a SHA-256" % name)
    text = value.removeprefix("sha256:").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise AuditError("%s has invalid SHA-256" % name)
    return "sha256:" + text


def _rows(media_root: Path, inputs_root: Path, maximum: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = _json(media_root / "candidate-inputs.json", "candidate manifest").get("clips")
    inputs = _json(inputs_root / "inputs.json", "frozen pilot inputs")
    selections = inputs.get("candidates")
    if not isinstance(candidates, list) or not isinstance(selections, list):
        raise AuditError("pilot candidate records are missing")
    excluded = {row.get("clip_id"): row for row in selections if isinstance(row, Mapping) and row.get("status") == "excluded"}
    if len(excluded) < 1:
        raise AuditError("no excluded untrained clips are available")
    from plumb.policies.tasks import BENCHMARK_TASK_REGISTRY
    task = BENCHMARK_TASK_REGISTRY.get("close_drawer")
    result: list[dict[str, Any]] = []
    for clip in candidates:
        if not isinstance(clip, Mapping) or clip.get("clip_id") not in excluded:
            continue
        frames = clip.get("frames")
        reference = clip.get("reference")
        if not isinstance(frames, list) or len(frames) != 16 or not isinstance(reference, Mapping):
            raise AuditError("candidate %s has invalid media binding" % clip.get("clip_id"))
        paths, timestamps, frame_hashes = [], [], []
        for index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise AuditError("candidate frame is invalid")
            path = _under(media_root, frame.get("path"), "frame")
            digest = _digest(frame.get("sha256"), "frame hash")
            if _sha256(path) != digest:
                raise AuditError("candidate frame hash changed")
            paths.append(str(path)); timestamps.append(float(frame.get("timestamp"))); frame_hashes.append(digest)
        reference_path = _under(media_root, reference.get("path"), "reference")
        reference_hash = _digest(reference.get("sha256"), "reference hash")
        if _sha256(reference_path) != reference_hash:
            raise AuditError("candidate reference hash changed")
        result.append({
            "clip_id": clip["clip_id"], "source_lineage_id": clip["source_lineage_id"],
            "frames": paths, "frame_timestamps": timestamps, "frame_sha256": frame_hashes,
            "reference_images": [{"path": str(reference_path), "sha256": reference_hash,
                "provenance_uri": reference.get("source_uri"), "role": reference.get("role")}],
            "instruction": task.instruction, "rubric": task.rubric,
            "exclusion_reason": excluded[clip["clip_id"]].get("exclusion_reason"),
        })
        if len(result) == maximum:
            break
    if len(result) != maximum:
        raise AuditError("expected %d excluded clips, found %d" % (maximum, len(result)))
    return result, {"candidate_manifest": _sha256(media_root / "candidate-inputs.json"), "inputs": _sha256(inputs_root / "inputs.json")}


def _bare_json(raw: str) -> bool:
    text = raw.strip()
    return text.startswith("{") and text.endswith("}") and not text.startswith("```")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise AuditError("an allocated GPU job is required")
    from cluster import judge_lora_pilot_compare as compare
    from deploy.baseten.training import train_judge_lora as trainer
    import torch
    import transformers
    import peft

    root = Path(args.dataset_root).resolve(strict=True)
    media_root = Path(args.media_root).resolve(strict=True) if args.media_root else root
    raw_output = Path(args.output)
    if raw_output.exists() or raw_output.is_symlink():
        raise AuditError("audit output already exists")
    pilot_result, adapter, binding = compare.load_bindings(
        Path(args.pilot_result), root, Path(args.model_root), Path(args.model_manifest)
    )
    rows, dataset_hashes = _rows(media_root, root, args.max_clips)
    if not torch.cuda.is_available():
        raise AuditError("CUDA is unavailable")
    student = pilot_result["student_model"]
    load = {"revision": student["model_revision"], "local_files_only": True, "trust_remote_code": False,
            "torch_dtype": torch.bfloat16, "use_safetensors": True}
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    processor = transformers.AutoProcessor.from_pretrained(binding["model_root"], **{key: value for key, value in load.items() if key != "torch_dtype"})
    base_model = transformers.AutoModelForVision2Seq.from_pretrained(binding["model_root"], **load)
    model = peft.PeftModel.from_pretrained(base_model, str(adapter), is_trainable=False).to("cuda")
    model.eval()
    disable = getattr(model, "disable_adapter", None)
    if not callable(disable):
        raise AuditError("PEFT adapter cannot be disabled for paired base inference")
    torch.cuda.reset_peak_memory_stats()
    observed = []
    started = time.monotonic()
    for row in rows:
        with disable():
            base_raw, base_timing = compare._generate(model, processor, row, torch)
        adapter_raw, adapter_timing = compare._generate(model, processor, row, torch)
        base = compare._parsed(base_raw)
        adapted = compare._parsed(adapter_raw)
        observed.append({
            "clip_id": row["clip_id"], "source_lineage_id": row["source_lineage_id"],
            "exclusion_reason": row["exclusion_reason"], "frame_sha256": row["frame_sha256"],
            "base": {**base, "raw_output": base_raw, "bare_json": _bare_json(base_raw), "timing": base_timing},
            "adapter": {**adapted, "raw_output": adapter_raw, "bare_json": _bare_json(adapter_raw), "timing": adapter_timing},
            "semantic_agree": base["status"] == adapted["status"] == "schema_valid" and base["semantic_tuple"] == adapted["semantic_tuple"],
        })
    def metrics(key: str) -> Mapping[str, Any]:
        values = [row[key] for row in observed]
        valid = [row for row in values if row["status"] == "schema_valid"]
        timing = [float(row["timing"]["wall_seconds"]) for row in values]
        tokens = [float(row["timing"]["generated_tokens"]) for row in values]
        return {"schema_valid": len(valid), "bare_json": sum(bool(row["bare_json"]) for row in values),
                "mean_wall_seconds": _mean(timing), "median_wall_seconds": _median(timing),
                "mean_generated_tokens": _mean(tokens), "median_generated_tokens": _median(tokens)}
    return {
        "schema_version": 1, "kind": "plumb_unqualified_lora_untrained_pool_output_audit",
        "purpose": PURPOSE, "qualified": False,
        "status": "completed_unqualified_engineering_audit",
        "reason": "Clips were excluded before pilot training because teacher outputs tied. This measures paired output behavior, not human agreement, semantic accuracy, calibration, or a held-out benchmark.",
        "cohort": "untrained_original_pilot_pool", "clip_count": len(observed),
        "pilot_result": {"path": str(Path(args.pilot_result)), "sha256": _sha256(Path(args.pilot_result))},
        "adapter": {"path": str(adapter), "tree_sha256": compare._tree_hash(adapter)},
        "model_binding": binding, "dataset_hashes": dataset_hashes,
        "generation": {"do_sample": False, "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS, "serving_message_layout": True},
        "metrics": {"base": metrics("base"), "adapter": metrics("adapter"),
                    "full_semantic_agreement": sum(bool(row["semantic_agree"]) for row in observed),
                    "full_semantic_disagreement": sum(not bool(row["semantic_agree"]) for row in observed)},
        "rows": observed, "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.monotonic() - started, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"), "human_quality_claim": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-result", required=True); parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-root", required=True); parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--media-root")
    parser.add_argument("--max-clips", type=int, default=9)
    args = parser.parse_args(argv)
    if not 1 <= args.max_clips <= 9: parser.error("--max-clips must be 1 through 9")
    try:
        report = run(args)
        encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({"status": report["status"], "output": args.output, "metrics": report["metrics"]}, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": {"type": type(error).__name__, "message": str(error)}}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
