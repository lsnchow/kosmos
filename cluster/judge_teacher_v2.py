"""Experimental, isolated teacher-v2 profile for unqualified pilot labels.

This module does not alter v1 collection records, the global primary judge, or
Gate-D rules.  It supplies a stricter diagnostic rubric and a target-selection
gate requiring one unique semantic mode with at least three of five actual,
schema-valid raw samples.  It never repairs a model response.
"""

from __future__ import annotations

import importlib.metadata
import argparse
import hashlib
import json
import os
import tempfile
import time
import traceback
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster import judge_teacher_pilot as v1
from plumb.policies.judge import JudgeInputProvenance, JudgeRequest, ReferenceImage
from plumb.policies.tasks import BENCHMARK_TASK_REGISTRY


PROFILE_ID = "qwen_teacher_v2_explicit_visual_definitions_unique_mode3_v1"
SELECTION_RULE_ID = "unique_modal_semantic_tuple_minimum_three_actual_members_v2"
SAMPLING = {"sample_count": 5, "temperature": 0.7, "top_p": 1.0, "max_new_tokens": 512}
VISUAL_DEFINITIONS = (
    "Definitions for this diagnostic only: integrity=artifact only when the supplied visual evidence itself is visibly "
    "corrupted, discontinuous, impossible, or unusable. Ordinary physical objects, the robot arm/gripper, shadows, "
    "occlusion, motion blur, and task-relevant clutter are not artifacts by themselves. collision=visible only for an "
    "unintended or unsafe collision. Intended grasping, touching a drawer handle, pushing/pulling the drawer, or other "
    "task-directed contact is not a collision by itself. Return exactly one JSON object with exactly integrity, collision, "
    "progress, completion_evidence, evidence_frame_indices, observable_reasons; use the enumerations and consistency "
    "rules in the system contract, with no prose or code fence. completion_evidence must be exactly met, not_met, or "
    "uncertain, never prose: met iff progress is exactly 5; not_met iff progress is an integer 0 through 4; uncertain "
    "requires progress null. observable_reasons must be one JSON string, not an array/object. evidence_frame_indices "
    "must be a nonempty JSON array of integers 0 through 15."
)


def v2_rubric() -> str:
    return BENCHMARK_TASK_REGISTRY.get("close_drawer").rubric + "\n\n" + v1.SCENE_CONTEXT_DISCLAIMER + "\n\n" + VISUAL_DEFINITIONS


def build_request(
    *,
    frames: Sequence[Any],
    timestamps: Sequence[float],
    reference: ReferenceImage,
    clip_id: str,
    video_sha256: str,
) -> JudgeRequest:
    """Build the diagnostic-only request; callers supply only blinded media."""

    task = BENCHMARK_TASK_REGISTRY.get("close_drawer")
    return JudgeRequest(
        frames=tuple(frames),
        frame_timestamps=tuple(timestamps),
        reference_images=(reference,),
        diagnostic_mode=True,
        task_instruction=task.instruction,
        task_rubric=v2_rubric(),
        provenance=JudgeInputProvenance(clip_id=clip_id, video_sha256=video_sha256, protocol_id=PROFILE_ID),
    )


def select_v2(raw_report: Mapping[str, Any]) -> Dict[str, Any]:
    """Select only a unique actual raw-sample mode with support >=3/5."""

    result = dict(v1._selection(raw_report))
    if result.get("status") != "accepted_unique_modal_tuple":
        result["profile_id"] = PROFILE_ID
        result["rule"] = SELECTION_RULE_ID
        return result
    if result.get("modal_count") < 3:
        result.pop("selected", None)
        result["status"] = "excluded_unique_mode_below_three"
    result["profile_id"] = PROFILE_ID
    result["rule"] = SELECTION_RULE_ID
    result["sampling"] = dict(SAMPLING)
    return result


def constrained_decoding_status() -> Dict[str, Any]:
    """Report installed support; this prototype never installs or enables it."""

    packages = {}
    for name in ("transformers", "lm-format-enforcer", "outlines", "guidance", "jsonformer"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "enabled": False,
        "reason": "No source/runtime-verified Qwen tokenizer JSON-grammar bridge is available; prompt-only v2 preserves raw output without repair.",
        "packages": packages,
    }


def profile_config() -> Dict[str, Any]:
    return {"profile_id": PROFILE_ID, "sampling": dict(SAMPLING), "rubric": v2_rubric(), "selection_rule": SELECTION_RULE_ID, "constrained_decoding": constrained_decoding_status()}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError("refusing to overwrite v2 evidence: %s" % path) from error
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def _existing(output: Path, clip: Mapping[str, Any], config_hash: str) -> Any:
    directory = output / "rawteacher-v2" / clip["clip_id"] / clip["input_binding_sha256"] / config_hash
    if not directory.is_dir(): return None
    for path in sorted(directory.glob("attempt-*.json"), reverse=True):
        try: payload = json.loads(path.read_text())
        except Exception: continue
        if payload.get("status") == "completed_unqualified" and payload.get("input_binding_sha256") == clip["input_binding_sha256"] and payload.get("profile_config_sha256") == config_hash:
            return path, payload
    return None


def _next(output: Path, clip: Mapping[str, Any], config_hash: str) -> Path:
    directory = output / "rawteacher-v2" / clip["clip_id"] / clip["input_binding_sha256"] / config_hash
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    number = 1
    while (directory / ("attempt-%04d.json" % number)).exists(): number += 1
    return directory / ("attempt-%04d.json" % number)


def _tuple_hist(selection: Mapping[str, Any]) -> Dict[str, int]:
    result = {}
    for item in selection.get("agreement", []):
        result[json.dumps(item["semantic_tuple"], separators=(",", ":"))] = int(item["count"])
    return result


def _bind_output(output: Path, frozen: Mapping[str, Any]) -> None:
    if output.is_symlink():
        raise RuntimeError("v2 output cannot be a symlink")
    if output.exists():
        path = output / "freeze.json"
        if path.is_symlink() or not path.is_file() or json.loads(path.read_text()) != dict(frozen):
            raise RuntimeError("existing v2 output has different or absent frozen bindings")
    else:
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        _write_new(output / "freeze.json", frozen)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("teacher v2 requires an allocated GPU job")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1": raise RuntimeError("offline flags are required")
    root = Path(args.dataset_root).resolve(strict=True)
    candidate_path, _ = v1._regular_relative(root, args.input_manifest, "candidate manifest")
    candidate = v1.validate_candidate_manifest(root, v1._json(candidate_path, "candidate manifest"))
    output_arg = Path(args.output_dir)
    if output_arg.is_symlink(): raise RuntimeError("v2 output-dir cannot be a symlink")
    output = output_arg.resolve()
    if output == root or output in root.parents or root in output.parents: raise RuntimeError("v2 output-dir must be separate from original input tree")
    model = v1._model_binding(Path(args.model_root), Path(args.model_manifest))
    runtime = v1._runtime_binding(Path(args.runtime_lock).resolve(strict=True))
    config = profile_config(); config_hash = _canonical_hash(config)
    _bind_output(output, {"purpose": "uncalibrated_teacher_v2_prompt_selection_diagnostic",
                          "qualified": False, "profile_config": config, "profile_config_sha256": config_hash,
                          "candidate_manifest_sha256": _sha(candidate_path), "model": model, "runtime": runtime})
    from plumb.policies.judge import QwenJudgeProfile, QwenRubricJudge
    judge = QwenRubricJudge(QwenJudgeProfile(profile_id=PROFILE_ID, local_model_path=model["root"], model_revision=v1.QWEN_REVISION, processor_revision=v1.QWEN_REVISION, transformers_version=v1.TRANSFORMERS_VERSION, local_files_only=True, trust_remote_code=False, torch_dtype="bfloat16", device_map="auto", asset_manifest_id="qwen-teacher-v2-model-manifest", asset_manifest_sha256=model["manifest_sha256"], runtime_lock_id="judge-training-runtime-tf449-v1", runtime_lock_sha256=runtime["sha256"]))
    if judge.capability().status.value != "ready_unqualified": raise RuntimeError("base Qwen teacher unavailable")
    pending = [clip for clip in candidate["clips"] if _existing(output, clip, config_hash) is None]
    if not 1 <= args.max_clips <= 16: raise ValueError("--max-clips must be 1..16")
    for clip in pending[:args.max_clips]:
        path = _next(output, clip, config_hash); started = time.perf_counter()
        try:
            frames = tuple(v1._load_rgb(root / item["path"], item["pixel_sha256"], "frame") for item in clip["frames"])
            ref = clip["reference"]
            reference = ReferenceImage(image=v1._load_rgb(root / ref["path"], ref["pixel_sha256"], "scene reference"), source_uri=ref["source_uri"], sha256=ref["sha256"])
            request = build_request(frames=frames, timestamps=[item["timestamp"] for item in clip["frames"]], reference=reference, clip_id=clip["clip_id"], video_sha256=clip["video"]["sha256"])
            raw = judge.evaluate(request, seeds=v1.TEACHER_SEEDS).as_dict(); selection = select_v2(raw)
            payload = {"schema":"plumb-judge-teacher-v2-clip-v1","qualified":False,"status":"completed_unqualified","profile_config":config,"profile_config_sha256":config_hash,"input_binding_sha256":clip["input_binding_sha256"],"clip":clip,"raw_judge_report":raw,"selection":selection,"source_release":os.environ.get("PLUMB_SOURCE_RELEASE"),"seconds":time.perf_counter()-started}
        except Exception as error:
            payload = {"schema":"plumb-judge-teacher-v2-clip-v1","qualified":False,"status":"failed","profile_config_sha256":config_hash,"input_binding_sha256":clip["input_binding_sha256"],"clip":clip,"error":{"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc(limit=8)},"seconds":time.perf_counter()-started}
        _write_new(path, payload)
    records = [_existing(output, clip, config_hash) for clip in candidate["clips"]]
    done = [record for record in records if record]
    selections = [record[1]["selection"] for record in done]
    accepted = sum(item.get("status") == "accepted_unique_modal_tuple" for item in selections)
    valid_slots = sum(len(item.get("valid_samples", [])) for item in selections)
    histogram = {}
    for item in selections:
        for key, count in _tuple_hist(item).items(): histogram[key] = histogram.get(key, 0) + count
    return {"schema":"plumb-judge-teacher-v2-summary-v1","qualified":False,"status":"partial" if len(done) < 16 else "completed_unqualified","profile_config":config,"profile_config_sha256":config_hash,"candidate_count":16,"completed_clip_count":len(done),"schema_valid_sample_slots":valid_slots,"accepted_unique_modal_at_least_three":accepted,"abstentions":len(done)-accepted,"tuple_histogram":histogram,"training_started":False,"filled_gates":[],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
            "producer_sha256": _sha(Path(__file__)), "model": model, "runtime": runtime,
            "raw_reports": [{"path": str(path.relative_to(output)), "sha256": _sha(path)} for path, _ in done]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-manifest", default="candidate-inputs.json"); parser.add_argument("--model-root", required=True); parser.add_argument("--model-manifest", required=True); parser.add_argument("--runtime-lock", required=True); parser.add_argument("--max-clips", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        result = run(args)
        _write_new(Path(args.output_dir) / ("summary-%s.json" % os.environ.get("SLURM_JOB_ID", "local")), result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error: print(json.dumps({"qualified":False,"status":"failed","error":{"type":type(error).__name__,"message":str(error)}}, sort_keys=True)); return 2


if __name__ == "__main__":
    raise SystemExit(main())
