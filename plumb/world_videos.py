"""Hash-bound playback catalog of existing world-model outputs, not new jobs."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote


# Only output fields of these known diagnostics are eligible. In particular,
# source/conditioning videos in provenance are never treated as model output.
KINDS = {
    "cosmos3_nano_diffusers_smoke": (0, "Cosmos3-Nano", "Action-conditioned fixture; initial conditioning frame is included."),
    "plumb_irasim_openvla_closed_loop_diagnostic": (1, "OpenVLA → IRASim", "Recorded closed-loop diagnostic; visible distortion and physical fidelity remain unqualified."),
    "plumb_irasim_native_open_loop_reference_diagnostic": (2, "IRASim", "Open-loop reference using supplied actions; not live policy feedback."),
    "plumb_irasim_causal_history_replay_diagnostic": (3, "IRASim", "Saved-action history replay; the policy was not queried."),
    "plumb_irasim_state_representation_replay_intervention": (4, "IRASim", "Saved-action state-representation experiment; not a new policy rollout."),
    "cosmos3_action_probe": (5, "Cosmos3-Nano", "Action-intervention probe, not a policy evaluation."),
    "irasim_original_one_step_smoke": (6, "IRASim", "Short two-frame diagnostic, not a full trajectory."),
    "irasim_one_step_action_probe": (7, "IRASim", "Short one-step intervention probe, not a full trajectory."),
}


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _at(value: dict, *keys: str) -> dict:
    for key in keys:
        value = _mapping(value.get(key))
    return value


def _output_refs(report: dict, kind: str):
    if kind in {"cosmos3_nano_diffusers_smoke", "cosmos3_action_probe"}:
        yield "output", _at(report, "result", "output_artifacts", "video")
    elif kind == "irasim_original_one_step_smoke":
        yield "output", _at(report, "artifacts", "video")
    elif kind in {"irasim_one_step_action_probe", "plumb_irasim_native_open_loop_reference_diagnostic"}:
        yield "output", _at(report, "artifacts", "mp4")
    elif kind == "plumb_irasim_openvla_closed_loop_diagnostic":
        yield "closed loop", _at(report, "video")
    elif kind == "plumb_irasim_causal_history_replay_diagnostic":
        yield "history replay", _at(report, "outcome", "video")
        yield "same-seed repeat", _at(report, "same_seed_repeat", "video")
    elif kind == "plumb_irasim_state_representation_replay_intervention":
        for prefix, videos in (("replay", _at(report, "outcome", "videos")),
                               ("same-seed repeat", _at(report, "same_seed_repeat", "replay", "videos"))):
            for branch, data in sorted(videos.items()):
                yield prefix + " / " + branch, _at(_mapping(data), "mp4")


def _regular_under(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
        if root not in path.resolve().parents or not path.is_file():
            return False
        current = root
        if current.is_symlink():
            return False
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return False
        return True
    except (OSError, ValueError):
        return False


def _local_video(report_path: Path, ref: dict, root: Path):
    remote = ref.get("path")
    digest = ref.get("sha256")
    if not isinstance(remote, str) or "\\" in remote or not isinstance(digest, str):
        return None
    expected = digest.removeprefix("sha256:")
    parts = Path(remote).parts
    if not parts or ".." in parts or not remote.endswith(".mp4"):
        return None
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        return None
    # Remote absolute paths are provenance only. Resolve the longest existing
    # suffix inside this local report bundle (handles nested replay branches).
    for length in range(min(5, len(parts)), 0, -1):
        suffix = parts[-length:]
        if any(part in {"/", ""} for part in suffix):
            continue
        path = report_path.parent.joinpath(*suffix)
        if not _regular_under(path, root):
            continue
        if path.stat().st_size > 32 * 1024 * 1024:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return path, expected
    return None


def _positive(value: Any):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


@lru_cache(maxsize=256)
def _probe(path: str, digest: str) -> dict:
    """Cache measured container metadata by verified content identity."""
    binary = shutil.which("ffprobe")
    if not binary:
        return {}
    try:
        result = subprocess.run(
            [binary, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,nb_frames,avg_frame_rate:format=duration", "-of", "json", path],
            capture_output=True, timeout=5, check=True,
        )
        if len(result.stdout) > 65536:
            return {}
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        stream = _mapping(streams[0]) if streams else {}
        rate = stream.get("avg_frame_rate", "0/1")
        fps = _positive(Fraction(rate))
        return {"width": _positive(stream.get("width")), "height": _positive(stream.get("height")),
                "frame_count": _positive(stream.get("nb_frames")), "fps": fps,
                "duration_seconds": _positive(_at(payload, "format").get("duration"))}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, ZeroDivisionError, subprocess.SubprocessError):
        return {}


def _url(path: Path, root: Path) -> str:
    return "/api/artifacts/" + quote(path.relative_to(root).as_posix(), safe="/")


def world_videos_payload(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    videos = []
    seen = set()
    for evidence in (root / "cluster-evidence", root / "live-integrated" / "cluster-evidence"):
        if not evidence.is_dir() or evidence.is_symlink():
            continue
        reports = sorted([*evidence.glob("*.json"), *evidence.glob("*/*.json")])
        for report_path in reports:
            try:
                if not _regular_under(report_path, root) or report_path.stat().st_size > 4 * 1024 * 1024:
                    continue
                report = _mapping(json.loads(report_path.read_text()))
                kind = report.get("kind")
                if kind not in KINDS:
                    continue
                rank, model, caveat = KINDS[kind]
                for label, ref in _output_refs(report, kind):
                    match = _local_video(report_path, ref, root)
                    if match is None:
                        continue
                    path, digest = match
                    relative = path.relative_to(evidence).as_posix()
                    identity = (relative, digest)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    measured = _probe(str(path), digest)
                    frame_count = measured.get("frame_count") or _positive(ref.get("frame_count"))
                    clip_id = "world-video-" + hashlib.sha256(relative.encode()).hexdigest()[:16]
                    stem = path.stem.replace("_", " ")
                    title = "%s · %s" % (model, stem)
                    if label != "output":
                        title += " · " + label
                    videos.append({
                        "id": clip_id, "title": title, "model": model, "kind": kind,
                        "video_url": _url(path, root), "report_url": _url(report_path, root),
                        "sha256": "sha256:" + digest,
                        "report_sha256": "sha256:" + hashlib.sha256(report_path.read_bytes()).hexdigest(),
                        "download_name": "Kosmos-" + path.name,
                        "source_experiment": report_path.parent.name if report_path.parent != evidence else report_path.stem,
                        "frame_count": frame_count,
                        "duration_seconds": measured.get("duration_seconds"),
                        "width": measured.get("width"), "height": measured.get("height"),
                        "fps": measured.get("fps"), "status": report.get("status", "unknown"),
                        "qualified": False, "provenance": "recorded_model_output",
                        "notes": ["Previously generated recording; playback does not start a GPU job.", caveat,
                                  "Experimental footage; no task-success or physics-fidelity score is asserted."],
                        "_sort": (rank, relative),
                    })
            except (OSError, ValueError, TypeError):
                continue
    videos.sort(key=lambda item: item.pop("_sort"))
    return {"videos": videos, "total": len(videos), "qualified": False,
            "mode": "recorded_world_model_playback", "generation_jobs_submitted": 0}
