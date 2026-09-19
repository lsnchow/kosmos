"""Select and prepare an explicitly unqualified, real-video judge LoRA pilot.

Only development data; selection is frozen before teacher calls. All selected
lineages, including later teacher abstentions, are reserved from future formal
study/calibration use. No success labels, robot-state conversion, or goals are
invented. Downloads use the existing cluster-only bounded acquisition helper.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from cluster.audit_bridge_metadata import audit, DATASET
from cluster.policy_smoke import _load_video_frames, uniform_video_indices
from plumb.policies.provenance import image_pixel_hash
from plumb.policies.tasks import BENCHMARK_TASK_REGISTRY, TASK_REGISTRY_HASH

REVISION = "0e9d76d07e9df3ea3eba257b2520d4913833fad2"
PURPOSE = "uncalibrated_judge_distillation_pilot"
CAMERA = "observation.images.image_0"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def make_plan(metadata_root, download_manifest, seed=20260919):
    metadata_root = Path(metadata_root)
    facts = audit(metadata_root, Path(download_manifest), expected_revision=REVISION)
    rows = [json.loads(line) for line in (metadata_root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    candidates = [row for row in rows if 16 <= row["length"] <= 100
                  and any(value.strip().casefold() == "close the drawer" for value in row["tasks"])]
    candidates.sort(key=lambda row: hashlib.sha256((str(seed) + ":" + str(row["episode_index"])).encode()).hexdigest())
    if len(candidates) < 16:
        raise ValueError("need16 distinct source episodes; do not duplicate frames/episodes")
    clips = []
    for offset, row in enumerate(candidates[:16]):
        episode = row["episode_index"]
        chunk = episode // 1000
        clips.append({"episode_index": episode, "source_length": row["length"],
                      "source_instructions": row["tasks"], "clip_id": "bridge-episode-%06d" % episode,
                      "source_lineage_id": "%s@%s:episode_%06d" % (DATASET, REVISION, episode),
                      "cohort": "train" if offset < 12 else "development_validation",
                      "parquet_path": "data/chunk-%03d/episode_%06d.parquet" % (chunk, episode),
                      "video_path": "videos/chunk-%03d/%s/episode_%06d.mp4" % (chunk, CAMERA, episode)})
    return {"schema": "plumb-judge-pilot-selection-v1", "purpose": PURPOSE, "qualified": False,
            "created_at": datetime.now(timezone.utc).isoformat(), "dataset": DATASET, "revision": REVISION,
            "seed": seed, "camera": CAMERA, "task": "close_drawer", "clips": clips,
            "metadata_input_files": facts["input_files"], "metadata_download_manifest_sha256": file_sha(download_manifest),
            "sampling_rule": "seeded SHA256 ordering of exact close-drawer instruction,16..100sourceframes; first12train,next4validation",
            "formal_excluded_lineages": [row["source_lineage_id"] for row in clips],
            "limitations": ["Experimental real-robot-video self-distillation, not world-rollout benchmark parity.",
                            "Scene camera/reset comparability is unqualified; no human labels or success labels are supplied.",
                            "All16 lineages remain development-only regardless of teacher acceptance or exclusion."]}


def validate_plan(plan):
    if plan.get("schema") != "plumb-judge-pilot-selection-v1" or plan.get("qualified") is not False or plan.get("purpose") != PURPOSE:
        raise ValueError("not an explicitly unqualified pilot plan")
    if plan.get("dataset") != DATASET or plan.get("revision") != REVISION:
        raise ValueError("pilot source identity mismatch")
    clips = plan.get("clips", [])
    if len(clips) != 16 or len({row["source_lineage_id"] for row in clips}) != 16:
        raise ValueError("pilot requires16 unique reserved lineages")
    if len({row["episode_index"] for row in clips}) != 16 or len({row["clip_id"] for row in clips}) != 16:
        raise ValueError("pilot cannot repeat an episode or clip identity")
    if [row["cohort"] for row in clips].count("train") != 12 or [row["cohort"] for row in clips].count("development_validation") != 4:
        raise ValueError("pilot split must remain12/4")
    if set(plan.get("formal_excluded_lineages", [])) != {row["source_lineage_id"] for row in clips}:
        raise ValueError("every selected lineage must be reserved from formal use")
    for row in clips:
        episode = row["episode_index"]
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
            raise ValueError("invalid source episode identity")
        if row["source_lineage_id"] != "%s@%s:episode_%06d" % (DATASET, REVISION, episode):
            raise ValueError("source lineage must bind exact episode and revision")
        for key in ("parquet_path", "video_path"):
            path = Path(row[key])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("source paths must be relative and contained")
    return clips


def download(plan_path, cluster_root, manifest_name=None):
    plan = json.loads(Path(plan_path).read_text())
    clips = validate_plan(plan)
    root = Path(cluster_root).resolve()
    if root.name != "plumb" or not str(root).startswith(("/scratch/", "/global/scratch/")):
        raise ValueError("pilot acquisition must remain in cluster scratch/plumb")
    receipt = root / "evidence" / (manifest_name or "IPEC-COMMUNITY--bridge_orig_lerobot-download.json")
    if receipt.exists():
        raise ValueError("download receipt already exists; preserve it rather than overwriting")
    command = [sys.executable, "-m", "cluster.download_assets", DATASET, "--repo-type", "dataset",
               "--revision", REVISION, "--root", str(root), "--max-gb", "0.05", "--execute"]
    if manifest_name is not None:
        command.extend(["--manifest-name", manifest_name])
    for clip in clips:
        for name in ("parquet_path", "video_path"):
            command.extend(["--allow", clip[name]])
    subprocess.run(command, check=True)


def _verified_source(root, relative, records):
    path = root / relative
    current = root
    if root.is_symlink():
        raise ValueError("source root may not be symlinked")
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("source path may not contain symlinks")
    if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
        raise ValueError("source path absent, symlinked, or escaping")
    record = records.get(relative)
    if not record or record["bytes"] != path.stat().st_size or record["sha256"] != file_sha(path):
        raise ValueError("source bytes disagree with downloaded manifest: " + relative)
    return path


def prepare(plan_path, source_root, receipt_path, output):
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    plan_path, source_root, receipt_path, output = map(Path, (plan_path, source_root, receipt_path, output))
    plan = json.loads(plan_path.read_text())
    clips = validate_plan(plan)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("repo") != DATASET or receipt.get("revision") != REVISION or receipt.get("repo_type") != "dataset":
        raise ValueError("pilot download receipt identity mismatch")
    records = {row["path"]: row for row in receipt["file_records"]}
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite a pilot dataset")
    output.mkdir(parents=True, exist_ok=False)
    frozen_clips = []
    transforms = []
    for clip in clips:
        video = _verified_source(source_root, clip["video_path"], records)
        parquet = _verified_source(source_root, clip["parquet_path"], records)
        table = pq.read_table(parquet, columns=["episode_index", "frame_index", "timestamp"]).to_pydict()
        decoded = _load_video_frames(video)
        if len(decoded) != clip["source_length"] or len(table["timestamp"]) != len(decoded):
            raise ValueError("video,metadata,Parquet counts disagree for " + clip["clip_id"])
        if table["frame_index"] != list(range(len(decoded))) or set(table["episode_index"]) != {clip["episode_index"]}:
            raise ValueError("source temporal/episode identity mismatch")
        times = table["timestamp"]
        if any(not math.isfinite(v) for v in times) or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("source timestamps must increase and be finite")
        selected = uniform_video_indices(len(decoded), 16)
        folder = output / "clips" / clip["clip_id"]
        folder.mkdir(parents=True)
        shutil.copy2(video, folder / "source.mp4")
        frame_records = []
        for index, source_index in enumerate(selected):
            frame = np.asarray(decoded[source_index])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError("source decoder must return RGB uint8 frames")
            path = folder / ("frame-%02d.png" % index)
            Image.fromarray(frame).save(path, format="PNG")
            frame_records.append({"path": path.relative_to(output).as_posix(), "sha256": file_sha(path),
                                  "pixel_sha256": image_pixel_hash(frame)["sha256"], "timestamp": float(times[source_index])})
        first = frame_records[0]
        frozen_clips.append({"clip_id": clip["clip_id"], "source_lineage_id": clip["source_lineage_id"], "cohort": clip["cohort"],
                             "timestamp_semantics": "source_parquet_nominal_timestamp",
                             "video": {"path": (folder / "source.mp4").relative_to(output).as_posix(), "sha256": file_sha(video)},
                             "frames": frame_records,
                             "reference": {"path": first["path"], "sha256": first["sha256"], "pixel_sha256": first["pixel_sha256"],
                                           "source_uri": "https://huggingface.co/datasets/%s/resolve/%s/%s#frame=0" % (DATASET, REVISION, clip["video_path"]),
                                           "role": "scene_context_not_goal"}})
        transforms.append({"clip_id": clip["clip_id"], "parquet_sha256": file_sha(parquet), "video_sha256": file_sha(video),
                           "frame_indices": list(selected), "source_frame_count": len(decoded), "resizing": "none",
                           "timestamp_source": "raw LeRobot Parquet timestamp; nominal dataset times, not asserted hardware-clock measurements"})
    task = BENCHMARK_TASK_REGISTRY.get("close_drawer")
    result = {"schema": "plumb-judge-teacher-pilot-input-v1", "purpose": PURPOSE, "qualified": False,
              "selection_sha256": file_sha(plan_path), "download_receipt_sha256": file_sha(receipt_path),
              "task": {"task_id": task.task_id, "task_registry_hash": TASK_REGISTRY_HASH, "rubric_hash": task.rubric_hash},
              "clips": frozen_clips}
    freeze_json(output / "selection.json", plan)
    freeze_json(output / "download.json", receipt)
    freeze_json(output / "transforms.json", transforms)
    freeze_json(output / "formal-excluded-lineages.json", plan["formal_excluded_lineages"])
    freeze_json(output / "candidate-inputs.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--metadata-root", type=Path, required=True)
    plan.add_argument("--metadata-manifest", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--plan", type=Path, required=True)
    fetch.add_argument("--root", type=Path, required=True)
    fetch.add_argument("--manifest-name")
    build = sub.add_parser("prepare")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--download-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        value = make_plan(args.metadata_root, args.metadata_manifest)
        freeze_json(args.output, value)
        print(json.dumps({"selection": str(args.output), "sha256": file_sha(args.output), "clips": 16, "qualified": False}))
    elif args.command == "download":
        download(args.plan, args.root, args.manifest_name)
    else:
        value = prepare(args.plan, args.source_root, args.download_manifest, args.output)
        print(json.dumps({"dataset_root": str(args.output), "clips": len(value["clips"]), "qualified": False}))


if __name__ == "__main__":
    main()
