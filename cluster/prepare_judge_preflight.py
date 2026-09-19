"""Reconstruct a real diagnostic teacher input for a training-framework check.

This is NOT a distillation dataset: an uncalibrated/disagreeing teacher sample
is retained solely to exercise multimodal loss/backpropagation. Never treat the
selected sample as a quorum result, human label, or qualified supervision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from cluster.policy_smoke import _load_video_frames
from plumb.policies.judge import parse_rubric_json
from plumb.policies.provenance import image_pixel_hash


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(report_path: Path, video: Path, reference: Path, output: Path) -> Dict[str, Any]:
    from PIL import Image

    if output.exists() or output.is_symlink():
        raise ValueError("preflight fixture output already exists; never overwrite evidence")
    report = json.loads(report_path.read_text())
    if report.get("kind") != "plumb_local_policy_smoke" or report.get("command") != "judge":
        raise ValueError("requires an actual persisted judge diagnostic")
    inputs = report["inputs"]
    if inputs.get("mode") != "diagnostic" or inputs.get("reference_role") != "scene":
        raise ValueError("only explicitly diagnostic scene-reference reports are accepted here")
    if digest(video) != inputs["generated_video"]["sha256"] or digest(reference) != inputs["reference"]["sha256"]:
        raise ValueError("video/reference bytes do not match the original teacher input")
    indices = inputs["selected_frame_indices"]
    if len(indices) != 16 or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices) or sorted(set(indices)) != indices:
        raise ValueError("source report must bind exactly16 unique ordered frame indices")
    all_frames = _load_video_frames(video)
    if len(all_frames) != inputs["source_video_frame_count"] or indices[-1] >= len(all_frames):
        raise ValueError("decoded video frame count differs from original teacher input")
    frames = [all_frames[index] for index in indices]
    votes = report["raw_judge_votes"]
    expected = votes["provenance"]["evidence_hashes"]["frame_pixel_hashes"]
    if [image_pixel_hash(frame)["sha256"] for frame in frames] != expected:
        raise ValueError("decoded video pixels differ from original teacher input")
    selected = None
    for sample in sorted(votes["raw_judge_samples"], key=lambda row: row["sample_index"]):
        for attempt in sorted(sample["attempts"], key=lambda row: row["attempt_index"]):
            if attempt.get("parsed") is not None:
                raw_parsed = parse_rubric_json(attempt["raw_output"])
                if raw_parsed.as_dict() != attempt["parsed"]:
                    raise ValueError("persisted parsed teacher target differs from raw output")
                selected = (sample, attempt)
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError("no schema-valid actual raw teacher sample is available")
    sample, attempt = selected
    output.mkdir(parents=True, exist_ok=False)
    frame_paths, frame_hashes = [], []
    for index, frame in enumerate(frames):
        path = output / ("frame-%02d.png" % index)
        Image.fromarray(frame).save(path, format="PNG")
        frame_paths.append(path.name)
        frame_hashes.append(digest(path))
    reference_path = output / "reference.png"
    with Image.open(reference) as source:
        source.convert("RGB").save(reference_path, format="PNG")
    row = {
        "schema": "plumb-judge-framework-preflight-input-v1", "purpose": "framework_preflight_only",
        "qualified": False, "eligible_for_training_dataset": False, "task": "diagnostic",
        "clip_id": "preflight-" + digest(video)[:16], "source_lineage_id": "source-video-sha256:" + digest(video),
        "source_report_sha256": digest(report_path), "source_video_sha256": digest(video),
        "teacher_aggregation_status": votes["judge_status"], "teacher_missing_reason": votes.get("missing_reason"),
        "target_selection": "first_schema_valid_actual_attempt_in_sample_order_not_a_quorum_label",
        "teacher_sample_index": sample["sample_index"], "teacher_attempt_index": attempt["attempt_index"],
        "teacher_model": report["model"], "teacher_raw_output": attempt["raw_output"],
        "instruction": inputs["task_instruction"], "rubric": inputs["task_rubric"],
        "label": attempt["parsed"], "prompt_format_retry": attempt.get("prompt_variant") != "base_v2",
        "frames": frame_paths, "frame_sha256": frame_hashes,
        "source_frame_indices": indices, "source_frame_pixel_hashes": expected,
        "frame_timestamps": inputs["frame_timestamps"],
        "reference_images": [{"path": reference_path.name, "sha256": digest(reference_path), "role": "scene",
                              "provenance_uri": "source-sha256:" + digest(reference)}],
        "limitations": ["Teacher is uncalibrated and aggregate outcome is preserved, not replaced by this one sample.",
                        "Loss/backpropagation checks only; no optimizer training, study result or judge-quality claim."]}
    with (output / "row.json").open("x", encoding="utf-8") as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    row = prepare(args.report, args.video, args.reference, args.output_dir)
    print(json.dumps({"row": str(args.output_dir / "row.json"), "purpose": row["purpose"], "qualified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
