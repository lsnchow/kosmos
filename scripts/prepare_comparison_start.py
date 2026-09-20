"""Prepare one source-bound, unqualified Bridge drawer start for KOSMOS.

This is deliberately narrower than a benchmark-start selector.  It freezes one
real 256x256 RGB frame and the *raw*, co-indexed LeRobot state/action records
that produced it.  It does not rescale the gripper, synthesize a zero state,
or claim that this source scene is a qualified match for the published robot
benchmark.

The selected lineage already belongs to the development-only judge pilot.  The
artifact carries that restriction forward: it is suitable for a frame-only
forward-dynamics smoke test, not a scored comparison or a formal calibration
cell.  The raw 8-D state is retained as evidence for a policy adapter only;
the world-model input must not use it until a source-bound normalizer has been
verified.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


DATASET = "IPEC-COMMUNITY/bridge_orig_lerobot"
REVISION = "0e9d76d07e9df3ea3eba257b2520d4913833fad2"
EPISODE = 31353
CAMERA = "observation.images.image_0"
SOURCE_INSTRUCTION = "close the drawer"
DEMO_INSTRUCTION = "Close the drawer"
PARQUET_RELATIVE_PATH = "data/chunk-031/episode_031353.parquet"
PARQUET_BYTES = 5187
PARQUET_SHA256 = "cf9dc5750321b0f5c769d256b6f70cebbb05677b3ab521d1bf64db073b469770"
STATE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper")
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
MAX_PARQUET_BYTES = 16 * 1024 * 1024


class StartPreparationError(ValueError):
    """Raised when an input cannot prove this exact source-bound start."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, maximum_bytes: int = 16 * 1024 * 1024) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise StartPreparationError("unsafe or missing JSON input: " + str(path))

    def reject(value: str) -> None:
        raise StartPreparationError("non-finite JSON constant in " + str(path) + ": " + value)

    try:
        return json.loads(path.read_bytes(), parse_constant=reject)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StartPreparationError("invalid JSON input %s: %s" % (path, error)) from error


def _contained(root: Path, relative: str) -> Path:
    candidate = root / relative
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise StartPreparationError("candidate asset path is not contained")
    if candidate.is_symlink() or not candidate.is_file() or root.resolve() not in candidate.resolve().parents:
        raise StartPreparationError("candidate asset is missing, linked, or escapes its input directory")
    return candidate


def _strict_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise StartPreparationError(label + " must be a finite number")
    return float(value)


def _vector(value: Any, *, width: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise StartPreparationError("%s must be a %d-D vector" % (label, width))
    return [_strict_number(item, label) for item in value]


def _read_lines(path: Path) -> list[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise StartPreparationError("unsafe or missing episode metadata: " + str(path))
    result = []
    for number, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise StartPreparationError("invalid episode metadata line %d" % number) from error
        if not isinstance(value, Mapping):
            raise StartPreparationError("episode metadata line %d is not an object" % number)
        result.append(value)
    return result


def _source_url() -> str:
    return "https://huggingface.co/datasets/%s/resolve/%s/%s" % (DATASET, REVISION, PARQUET_RELATIVE_PATH)


def _download_parquet() -> bytes:
    """Fetch only the pinned, tiny Parquet object and verify LFS content hash."""

    request = Request(_source_url(), headers={"User-Agent": "kosmos-comparison-start/1"})
    try:
        with urlopen(request, timeout=45) as response:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(min(1024 * 1024, MAX_PARQUET_BYTES + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_PARQUET_BYTES:
                    raise StartPreparationError("source Parquet exceeds the bounded selective-download limit")
                chunks.append(chunk)
    except StartPreparationError:
        raise
    except OSError as error:
        raise StartPreparationError("could not fetch pinned source Parquet: " + str(error)) from error
    raw = b"".join(chunks)
    if len(raw) != PARQUET_BYTES or _sha256_bytes(raw) != PARQUET_SHA256:
        raise StartPreparationError("pinned source Parquet byte/hash verification failed")
    return raw


def _candidate(candidate_inputs: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    document = _load_json(candidate_inputs)
    if not isinstance(document, Mapping) or document.get("schema") != "plumb-judge-teacher-pilot-input-v1":
        raise StartPreparationError("candidate input must be the frozen judge-pilot input document")
    if document.get("qualified") is not False:
        raise StartPreparationError("candidate input must remain explicitly unqualified")
    clips = document.get("clips")
    if not isinstance(clips, list):
        raise StartPreparationError("candidate input has no clips")
    clip_id = "bridge-episode-%06d" % EPISODE
    rows = [row for row in clips if isinstance(row, Mapping) and row.get("clip_id") == clip_id]
    if len(rows) != 1:
        raise StartPreparationError("expected exactly one selected source clip")
    clip = rows[0]
    lineage = "%s@%s:episode_%06d" % (DATASET, REVISION, EPISODE)
    if clip.get("source_lineage_id") != lineage or clip.get("cohort") != "development_validation":
        raise StartPreparationError("candidate is not the frozen development-only source lineage")
    reference = clip.get("reference")
    if not isinstance(reference, Mapping) or reference.get("role") != "scene_context_not_goal":
        raise StartPreparationError("source frame must remain scene context, never a substituted goal")
    frame_path = _contained(candidate_inputs.parent, reference.get("path", ""))
    expected_url = (
        "https://huggingface.co/datasets/%s/resolve/%s/videos/chunk-031/%s/episode_%06d.mp4#frame=0"
        % (DATASET, REVISION, CAMERA, EPISODE)
    )
    if reference.get("source_uri") != expected_url:
        raise StartPreparationError("candidate reference does not bind the pinned camera-0 frame")
    return document, clip, frame_path


def _verify_rgb_frame(path: Path, reference: Mapping[str, Any]) -> tuple[bytes, str, str]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != reference.get("sha256"):
        raise StartPreparationError("source frame PNG hash disagrees with frozen candidate input")
    try:
        with Image.open(BytesIO(raw)) as image:
            rgb = np.asarray(image.convert("RGB"))
    except (OSError, ValueError) as error:
        raise StartPreparationError("candidate frame is not decodable RGB PNG") from error
    if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3):
        raise StartPreparationError("candidate frame must be exactly RGB uint8 256x256")
    pixel_sha = "sha256:" + _sha256_bytes(rgb.tobytes(order="C"))
    if pixel_sha != reference.get("pixel_sha256"):
        raise StartPreparationError("source frame pixel hash disagrees with frozen candidate input")
    return raw, _sha256_bytes(raw), pixel_sha


def _verify_metadata(info_path: Path, episodes_path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    info = _load_json(info_path)
    if not isinstance(info, Mapping):
        raise StartPreparationError("Bridge info metadata must be an object")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise StartPreparationError("Bridge info metadata has no feature map")
    camera = features.get(CAMERA)
    state = features.get("observation.state")
    action = features.get("action")
    if not isinstance(camera, Mapping) or camera.get("dtype") != "video" or camera.get("shape") != [256, 256, 3]:
        raise StartPreparationError("metadata does not establish the required camera-0 RGB geometry")
    if not isinstance(state, Mapping) or state.get("dtype") != "float32" or state.get("shape") != [8]:
        raise StartPreparationError("metadata does not establish raw 8-D observation.state")
    if not isinstance(action, Mapping) or action.get("dtype") != "float32" or action.get("shape") != [7]:
        raise StartPreparationError("metadata does not establish raw 7-D action")
    if state.get("names", {}).get("motors") != list(STATE_NAMES):
        raise StartPreparationError("metadata state channel order differs from pinned Bridge convention")
    if action.get("names", {}).get("motors") != list(ACTION_NAMES):
        raise StartPreparationError("metadata action channel order differs from pinned Bridge convention")
    rows = [row for row in _read_lines(episodes_path) if row.get("episode_index") == EPISODE]
    if len(rows) != 1:
        raise StartPreparationError("exact source episode metadata is missing or duplicated")
    episode = rows[0]
    if episode.get("tasks") != [SOURCE_INSTRUCTION] or episode.get("length") != 19:
        raise StartPreparationError("source episode no longer has the exact close-drawer instruction/length")
    return info, episode


def _parse_parquet(raw: bytes) -> Mapping[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise StartPreparationError("pyarrow is required to decode the selected LeRobot Parquet source") from error
    try:
        table = pq.read_table(pa.BufferReader(raw))
    except Exception as error:  # pyarrow's exception hierarchy varies by version.
        raise StartPreparationError("could not read selected source Parquet") from error
    required = ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index")
    if tuple(table.column_names) != required or table.num_rows != 19:
        raise StartPreparationError("source Parquet schema/row count differs from pinned source")
    values = table.select(required).to_pydict()
    for key in required:
        if not isinstance(values.get(key), list) or len(values[key]) != 19:
            raise StartPreparationError("source Parquet column is absent or unaligned: " + key)
    if values["frame_index"] != list(range(19)) or set(values["episode_index"]) != {EPISODE}:
        raise StartPreparationError("source Parquet temporal or episode identity mismatch")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values["index"]):
        raise StartPreparationError("source Parquet global index is not integral")
    timestamps = [_strict_number(value, "timestamp") for value in values["timestamp"]]
    if timestamps[0] != 0.0 or any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise StartPreparationError("source Parquet timestamps are not an increasing frame-0 sequence")
    states = [_vector(value, width=8, label="observation.state") for value in values["observation.state"]]
    actions = [_vector(value, width=7, label="action") for value in values["action"]]
    return {
        "state": states[0],
        "action": actions[0],
        "timestamp": timestamps[0],
        "frame_index": values["frame_index"][0],
        "dataset_index": values["index"][0],
        "task_index": values["task_index"][0],
        "episode_frames": table.num_rows,
        "state_gripper_min": min(row[7] for row in states),
        "state_gripper_max": max(row[7] for row in states),
    }


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise StartPreparationError("refusing to overwrite immutable start artifact: " + str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def prepare(
    *,
    candidate_inputs: Path,
    metadata_info: Path,
    metadata_episodes: Path,
    output: Path,
    parquet_bytes: bytes,
) -> Mapping[str, Any]:
    """Verify inputs then atomically create the self-contained start artifact directory."""

    if output.name.startswith("start-") is False:
        raise StartPreparationError("output directory must be named start-* to keep evidence scoped")
    if output.exists() or output.is_symlink():
        raise StartPreparationError("refusing to overwrite start artifact directory")
    document, clip, frame_source = _candidate(candidate_inputs)
    reference = clip["reference"]
    frame_raw, frame_sha, pixel_sha = _verify_rgb_frame(frame_source, reference)
    info, episode = _verify_metadata(metadata_info, metadata_episodes)
    if len(parquet_bytes) != PARQUET_BYTES or _sha256_bytes(parquet_bytes) != PARQUET_SHA256:
        raise StartPreparationError("provided source Parquet does not match the pinned Hub LFS object")
    row = _parse_parquet(parquet_bytes)

    # All preflight validation happens before the output directory exists.  The
    # only following writes are to a new evidence directory and cannot replace
    # prior evidence.
    output.mkdir(parents=True, exist_ok=False)
    try:
        frame_output = output / "start.png"
        parquet_output = output / "source.parquet"
        frame_output.write_bytes(frame_raw)
        parquet_output.write_bytes(parquet_bytes)
        if _sha256_file(frame_output) != frame_sha or _sha256_file(parquet_output) != PARQUET_SHA256:
            raise StartPreparationError("new evidence copy changed while writing")
        raw_gripper = row["state"][7]
        artifact = {
            "schema": "kosmos-comparison-start-v1",
            "status": "prepared_unqualified",
            "qualified": False,
            "purpose": "frame_only_forward_dynamics_smoke_input",
            "task": {
                "task_id": "close_drawer",
                "task_instruction_verbatim": DEMO_INSTRUCTION,
                "source_instruction_verbatim": SOURCE_INSTRUCTION,
                "task_text_match": True,
                "qualified_scene_or_reset_match": False,
            },
            "source": {
                "dataset": DATASET,
                "revision": REVISION,
                "lineage_id": "%s@%s:episode_%06d" % (DATASET, REVISION, EPISODE),
                "episode_index": EPISODE,
                "camera": CAMERA,
                "camera_geometry": [256, 256, 3],
                "source_video_uri": reference["source_uri"],
                "source_frame_index": 0,
                "source_timestamp": row["timestamp"],
                "source_episode_frame_count": row["episode_frames"],
                "source_global_index": row["dataset_index"],
                "source_task_index": row["task_index"],
                "candidate_inputs": {
                    "path": str(candidate_inputs),
                    "sha256": _sha256_file(candidate_inputs),
                    "cohort": clip["cohort"],
                    "formal_exclusion_inherited": True,
                },
                "metadata": {
                    "info_path": str(metadata_info),
                    "info_sha256": _sha256_file(metadata_info),
                    "episodes_path": str(metadata_episodes),
                    "episodes_sha256": _sha256_file(metadata_episodes),
                    "fps": info.get("fps"),
                    "episode_declared_length": episode["length"],
                },
            },
            "frame": {
                "path": "start.png",
                "bytes": len(frame_raw),
                "sha256": frame_sha,
                "pixel_sha256": pixel_sha,
                "dtype": "uint8",
                "shape": [256, 256, 3],
                "color_order": "RGB",
                "resizing": "none",
            },
            "source_parquet": {
                "path": "source.parquet",
                "relative_path": PARQUET_RELATIVE_PATH,
                "url": _source_url(),
                "bytes": PARQUET_BYTES,
                "sha256": PARQUET_SHA256,
                "upstream_lfs_sha256": PARQUET_SHA256,
                "verified": True,
            },
            "proprioception": {
                "source_column": "observation.state",
                "frame_index": 0,
                "raw_float32_8d": row["state"],
                "channel_order": list(STATE_NAMES),
                "conversion_applied": False,
                "normalizer": None,
                "recorded_gripper_value": raw_gripper,
                "episode_gripper_range": [row["state_gripper_min"], row["state_gripper_max"]],
                "units": "not specified by the pinned LeRobot metadata; values are preserved raw",
            },
            "action": {
                "source_column": "action",
                "frame_index": 0,
                "raw_float32_7d": row["action"],
                "channel_order": list(ACTION_NAMES),
                "conversion_applied": False,
                "normalizer": None,
                "temporal_alignment": "co-indexed in the source Parquet; upstream metadata does not document actuation timing",
            },
            "use_constraints": {
                "manual_world_probe": {
                    "allowed": True,
                    "input": "start.png plus independently supplied 10-D test actions; no state field is sent to the world model",
                    "qualified": False,
                },
                "policy_input": {
                    "allowed": False,
                    "reason": "raw source state is available, but no source-bound policy normalizer/proprio convention has been verified",
                },
                "scored_comparison_or_calibration": {
                    "allowed": False,
                    "reason": "development-only lineage is formally excluded and reset/camera/task parity is unqualified",
                },
                "world_model_state_contract": {
                    "allowed": False,
                    "reason": "raw source gripper %.9g is not converted to the separately reported 0..0.39 profile; no conversion is inferred"
                    % raw_gripper,
                },
            },
            "limitations": [
                "This is one real Bridge source frame, not evidence that the generated rollout matches a real starting-state distribution.",
                "The frame is a prior development-only judge-pilot lineage and may never enter formal study or calibration counts.",
                "No success label, goal image, gripper conversion, action rescaling, or reset-state equivalence was invented.",
            ],
        }
        _write_new_json(output / "start.json", artifact)
    except Exception:
        # A failed new build is not evidence.  The output is only ever created
        # by this call and was preflighted above, so cleanup cannot delete user
        # or prior-run material.
        shutil.rmtree(output)
        raise
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-inputs", type=Path, required=True)
    parser.add_argument("--metadata-info", type=Path, required=True)
    parser.add_argument("--metadata-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="download only the pinned 5,187-byte source Parquet")
    source.add_argument("--parquet", type=Path, help="previously fetched pinned source Parquet")
    args = parser.parse_args()
    try:
        if args.download:
            parquet = _download_parquet()
        else:
            if args.parquet.is_symlink() or not args.parquet.is_file() or args.parquet.stat().st_size > MAX_PARQUET_BYTES:
                raise StartPreparationError("unsafe or missing provided source Parquet")
            parquet = args.parquet.read_bytes()
        artifact = prepare(
            candidate_inputs=args.candidate_inputs,
            metadata_info=args.metadata_info,
            metadata_episodes=args.metadata_episodes,
            output=args.output,
            parquet_bytes=parquet,
        )
    except (OSError, StartPreparationError, TypeError, KeyError) as error:
        print(json.dumps({"status": "blocked", "qualified": False, "reason": str(error)}))
        return 2
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "qualified": False,
                "output": str(args.output / "start.json"),
                "frame_sha256": artifact["frame"]["sha256"],
                "state_contract_allowed": artifact["use_constraints"]["world_model_state_contract"]["allowed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
