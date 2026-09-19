#!/usr/bin/env python3
"""Prepare opaque, development-only packets for later human review.

The input is the original 16-clip candidate manifest plus a local mirror of
its source video and sampled PNG files.  This is not a Gate-D exporter: it
creates no annotator registry, assignment, human ID, label, calibration
manifest, or claim of review completion.  It only prepares byte-preserving
media copies and blank worksheets for two reviewers to use later.

The public packet directory is intentionally blind.  Original clip IDs,
cohorts, source lineages, source paths, model votes, checkpoints, and file
hashes appear only in the separately supplied private resolver, which must be
outside ``data/live-integrated`` and is written mode 0600.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from plumb.policies.tasks import BENCHMARK_TASK_REGISTRY


SCHEMA = "plumb-development-review-packets-v1"
PURPOSE = "development_review_only"
EXPECTED_CLIP_COUNT = 16
EXPECTED_FRAME_COUNT = 16
DEFAULT_BASE_URL = "http://127.0.0.1:8787"
_SHA256_RE = re.compile(r"^(sha256:)?[0-9a-f]{64}$")


class ReviewPreparationError(ValueError):
    """A supplied candidate/media path cannot be prepared safely."""


REVIEW_GUIDANCE = {
    "integrity": {
        "intact": "No visible encoding, rendering, or camera defect materially prevents assessing the requested visible end state.",
        "artifact": "A visible encoding, rendering, or camera defect materially prevents reliable assessment; ordinary robot motion or uncertainty alone is not an artifact.",
        "uncertain": "Visible evidence is insufficient to decide whether the clip is intact or whether the end state can be assessed.",
    },
    "collision": {
        "none_visible": "No unintended task-relevant collision is visibly evidenced. Intended handle/gripper contact is not itself a collision.",
        "visible": "A task-relevant unintended collision is visibly evidenced; identify it in observable reasons and cite frames.",
        "uncertain": "The available frames do not establish whether an unintended collision occurred.",
    },
    "completion": "Use only the supplied media and task rubric. Do not infer invisible events. Cite visible frame indices for every judgement.",
}


@dataclass(frozen=True)
class SourceFrame:
    relative_path: str
    sha256: str
    pixel_sha256: str
    timestamp: float


@dataclass(frozen=True)
class SourceClip:
    source_clip_id: str
    source_cohort: str
    source_lineage_id: str
    task: str
    video_relative_path: str
    video_sha256: str
    frames: Tuple[SourceFrame, ...]


def _normalise_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.lower()):
        raise ReviewPreparationError("%s must be a SHA-256 digest" % field)
    return value.lower().removeprefix("sha256:")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        _absolute(path).relative_to(_absolute(parent))
        return True
    except ValueError:
        return False


def _resolved_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _reject_existing_symlink_components(path: Path, field: str) -> None:
    """Reject every existing symlink component before a private path is made."""

    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ReviewPreparationError("%s contains a symlink: %s" % (field, current))
        if not current.exists():
            break


def _validate_private_resolver_target(path: Path, served_root: Path) -> None:
    _reject_existing_symlink_components(path.parent, "private resolver parent")
    if _is_within(path, served_root) or _resolved_within(path, served_root):
        raise ReviewPreparationError("private resolver must be outside the served data/live-integrated root")
    if path.suffix.lower() != ".json":
        raise ReviewPreparationError("private resolver must end in .json")


def _safe_relative_path(value: Any, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ReviewPreparationError("%s must be a non-empty relative path" % field)
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise ReviewPreparationError("%s escapes the dataset root" % field)
    return path


def _safe_regular_file(root: Path, relative: str, field: str) -> Path:
    """Resolve one source media file without accepting a symlink or escape."""

    if root.is_symlink() or not root.is_dir():
        raise ReviewPreparationError("dataset root must be an existing non-symlink directory")
    parts = _safe_relative_path(relative, field).parts
    current = root
    for part in parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ReviewPreparationError("%s is missing: %s" % (field, current)) from error
        if stat.S_ISLNK(mode):
            raise ReviewPreparationError("%s contains a symlink: %s" % (field, current))
    if not current.is_file():
        raise ReviewPreparationError("%s is not a regular file: %s" % (field, current))
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ReviewPreparationError("%s resolves outside dataset root" % field) from error
    return current


def _read_candidate_manifest(path: Path) -> Tuple[List[SourceClip], str]:
    if path.is_symlink() or not path.is_file():
        raise ReviewPreparationError("candidate manifest must be a regular existing JSON file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReviewPreparationError("candidate manifest is not valid JSON") from error
    if not isinstance(raw, Mapping) or raw.get("qualified") is not False:
        raise ReviewPreparationError("candidate manifest must explicitly remain qualified=false")
    task_descriptor = raw.get("task")
    if isinstance(task_descriptor, Mapping):
        task = task_descriptor.get("task_id")
        if not isinstance(task, str) or not task:
            raise ReviewPreparationError("candidate manifest task descriptor lacks task_id")
        _normalise_digest(task_descriptor.get("rubric_hash"), "candidate manifest task.rubric_hash")
        _normalise_digest(task_descriptor.get("task_registry_hash"), "candidate manifest task.task_registry_hash")
    elif isinstance(task_descriptor, str) and task_descriptor:
        # Legacy pilot input shape; current candidate-inputs.json uses the
        # descriptor above and carries the task only once at manifest scope.
        task = task_descriptor
    else:
        raise ReviewPreparationError("candidate manifest task must be a task descriptor or task ID")
    candidates = raw.get("clips")
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_CLIP_COUNT:
        raise ReviewPreparationError("candidate manifest must contain exactly %d clips" % EXPECTED_CLIP_COUNT)
    clips: List[SourceClip] = []
    seen_ids = set()
    for item_index, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            raise ReviewPreparationError("candidate %d is not an object" % item_index)
        clip_id = item.get("clip_id")
        cohort = item.get("cohort")
        lineage = item.get("source_lineage_id")
        item_task = item.get("task")
        if isinstance(item_task, Mapping):
            item_task = item_task.get("task_id")
        task_for_clip = item_task if item_task is not None else task
        if not all(isinstance(value, str) and value for value in (clip_id, cohort, lineage, task_for_clip)):
            raise ReviewPreparationError("candidate %d lacks clip_id/cohort/source_lineage_id/task" % item_index)
        if clip_id in seen_ids:
            raise ReviewPreparationError("candidate manifest repeats source clip_id %s" % clip_id)
        seen_ids.add(clip_id)
        try:
            BENCHMARK_TASK_REGISTRY.get(task_for_clip)
        except KeyError as error:
            raise ReviewPreparationError("candidate %s names unknown task %r" % (clip_id, task_for_clip)) from error
        video = item.get("video")
        if not isinstance(video, Mapping):
            raise ReviewPreparationError("candidate %s lacks source video binding" % clip_id)
        video_path = video.get("path")
        video_hash = _normalise_digest(video.get("sha256"), "candidate %s video.sha256" % clip_id)
        _safe_relative_path(video_path, "candidate %s video.path" % clip_id)
        raw_frames = item.get("frames")
        if not isinstance(raw_frames, list) or len(raw_frames) != EXPECTED_FRAME_COUNT:
            raise ReviewPreparationError("candidate %s needs exactly %d sampled frames" % (clip_id, EXPECTED_FRAME_COUNT))
        frames: List[SourceFrame] = []
        frame_paths = set()
        for frame_index, frame in enumerate(raw_frames):
            if not isinstance(frame, Mapping):
                raise ReviewPreparationError("candidate %s frame %d is not an object" % (clip_id, frame_index))
            relative_path = frame.get("path")
            if relative_path in frame_paths:
                raise ReviewPreparationError("candidate %s repeats a sampled frame path" % clip_id)
            frame_paths.add(relative_path)
            _safe_relative_path(relative_path, "candidate %s frame.path" % clip_id)
            timestamp = frame.get("timestamp")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(float(timestamp))
            ):
                raise ReviewPreparationError("candidate %s frame %d timestamp is not finite" % (clip_id, frame_index))
            if frames and float(timestamp) <= frames[-1].timestamp:
                raise ReviewPreparationError("candidate %s frame timestamps must be strictly increasing" % clip_id)
            frames.append(
                SourceFrame(
                    relative_path=str(relative_path),
                    sha256=_normalise_digest(frame.get("sha256"), "candidate %s frame.sha256" % clip_id),
                    pixel_sha256=_normalise_digest(frame.get("pixel_sha256"), "candidate %s frame.pixel_sha256" % clip_id),
                    timestamp=float(timestamp),
                )
            )
        clips.append(
            SourceClip(
                source_clip_id=clip_id,
                source_cohort=cohort,
                source_lineage_id=lineage,
                task=task_for_clip,
                video_relative_path=str(video_path),
                video_sha256=video_hash,
                frames=tuple(frames),
            )
        )
    return clips, _sha256_file(path)


def _verify_source_media(dataset_root: Path, clips: Iterable[SourceClip]) -> Dict[Tuple[str, str], Path]:
    """Verify every original byte stream before the first public copy occurs."""

    verified: Dict[Tuple[str, str], Path] = {}
    for clip in clips:
        video = _safe_regular_file(dataset_root, clip.video_relative_path, "source video")
        actual = _sha256_file(video)
        if actual != clip.video_sha256:
            raise ReviewPreparationError("source video SHA-256 mismatch for %s" % clip.source_clip_id)
        verified[(clip.source_clip_id, "video")] = video
        for index, frame in enumerate(clip.frames):
            source = _safe_regular_file(dataset_root, frame.relative_path, "sampled PNG")
            actual = _sha256_file(source)
            if actual != frame.sha256:
                raise ReviewPreparationError(
                    "sampled PNG SHA-256 mismatch for %s frame %d" % (clip.source_clip_id, index)
                )
            if source.suffix.lower() != ".png":
                raise ReviewPreparationError("sampled frame is not a PNG for %s frame %d" % (clip.source_clip_id, index))
            verified[(clip.source_clip_id, "frame:%02d" % index)] = source
    return verified


def _copy_verified(source: Path, destination: Path, expected_digest: str) -> None:
    """Copy bytes without reencoding, then re-hash the new public copy."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ReviewPreparationError("refusing to overwrite public media: %s" % destination)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            for block in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                output_stream.write(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    if digest.hexdigest() != expected_digest or _sha256_file(destination) != expected_digest:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise ReviewPreparationError("copied media hash mismatch: %s" % destination)


def _write_new_text(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _write_private_resolver(path: Path, resolver: Mapping[str, Any], served_root: Path) -> None:
    _validate_private_resolver_target(path, served_root)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite existing private resolver")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = json.dumps(dict(resolver), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _normalise_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        raise ReviewPreparationError("base URL must start with http:// or https://")
    return value.rstrip("/")


def _assert_public_packet(packet: Mapping[str, Any], source_clip: SourceClip) -> None:
    """Defend against accidental source/model metadata in a public worksheet."""

    prohibited_fields = {
        "cohort", "source_lineage_id", "source_clip_id", "source_path", "sha256", "pixel_sha256",
        "checkpoint", "model", "teacher", "vote", "raw_teacher", "dataset_root",
    }
    leaked = prohibited_fields & set(packet)
    if leaked:
        raise ReviewPreparationError("public packet leaks private field names: %s" % ", ".join(sorted(leaked)))
    serialized = json.dumps(packet, sort_keys=True).lower()
    for prohibited_value in (source_clip.source_clip_id.lower(), source_clip.source_lineage_id.lower(), source_clip.source_cohort.lower()):
        if prohibited_value in serialized:
            raise ReviewPreparationError("public packet leaks source identity metadata")


def prepare_review_packets(
    *,
    candidate_manifest: Path,
    dataset_root: Path,
    public_out: Path,
    private_resolver: Path,
    base_url: str = DEFAULT_BASE_URL,
    served_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Prepare 16 public blank worksheets and a coordinator-only resolver.

    All source media hashes are checked before any packet/media is published.
    The function returns only public-facing summary paths and counts; private
    source identity stays in ``private_resolver``.
    """

    repository_root = Path(__file__).resolve().parents[1]
    served = _absolute(served_root or (repository_root / "data" / "live-integrated"))
    review_root = served / "review"
    public = _absolute(public_out)
    private = _absolute(private_resolver)
    source_root = _absolute(dataset_root)
    if not _is_within(public, review_root) or public == review_root:
        raise ReviewPreparationError("public output must be a fresh directory under data/live-integrated/review")
    if public.parent != review_root:
        raise ReviewPreparationError("public output must be directly inside data/live-integrated/review")
    if public.exists() or public.is_symlink():
        raise FileExistsError("refusing to overwrite existing public review directory")
    _validate_private_resolver_target(private, served)
    if private.exists() or private.is_symlink():
        raise FileExistsError("refusing to overwrite existing private resolver")
    base = _normalise_base_url(base_url)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ReviewPreparationError("dataset root must be an existing non-symlink directory")

    clips, candidate_manifest_sha256 = _read_candidate_manifest(_absolute(candidate_manifest))
    verified_sources = _verify_source_media(source_root, clips)

    review_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    if review_root.is_symlink() or not review_root.is_dir():
        raise ReviewPreparationError("public review root is not a regular directory")
    stage = review_root / (".%s-staging-%s" % (public.name, uuid.uuid4().hex))
    stage.mkdir(mode=0o700)
    worksheets: List[Dict[str, Any]] = []
    resolver_entries: Dict[str, Dict[str, Any]] = {}
    try:
        for position, source_clip in enumerate(clips, start=1):
            public_clip_id = "review-clip-" + str(uuid.uuid4())
            public_media_id = "review-media-" + str(uuid.uuid4())
            task = BENCHMARK_TASK_REGISTRY.get(source_clip.task)
            media_dir = stage / "media" / public_media_id
            copied_video = media_dir / "source.mp4"
            _copy_verified(
                verified_sources[(source_clip.source_clip_id, "video")], copied_video, source_clip.video_sha256
            )
            public_frames = []
            private_frames = []
            for index, frame in enumerate(source_clip.frames):
                destination = media_dir / "frames" / ("frame-%02d.png" % index)
                _copy_verified(
                    verified_sources[(source_clip.source_clip_id, "frame:%02d" % index)], destination, frame.sha256
                )
                frame_url = "%s/api/artifacts/review/%s/media/%s/frames/frame-%02d.png" % (
                    base,
                    public.name,
                    public_media_id,
                    index,
                )
                public_frames.append({"index": index, "timestamp": frame.timestamp, "url": frame_url})
                private_frames.append(
                    {
                        "index": index,
                        "timestamp": frame.timestamp,
                        "source_path": frame.relative_path,
                        "source_file_sha256": "sha256:" + frame.sha256,
                        "source_pixel_sha256": "sha256:" + frame.pixel_sha256,
                        "public_relative_path": "media/%s/frames/frame-%02d.png" % (public_media_id, index),
                    }
                )
            packet: Dict[str, Any] = {
                "schema": SCHEMA,
                "purpose": PURPOSE,
                "qualified": False,
                "gate_d_eligible": False,
                "export_eligible": False,
                "reservation": "reserved_development_review_only",
                "assignment_status": "unassigned",
                "reviewer_id": "",
                "clip_id": public_clip_id,
                "media_id": public_media_id,
                "position": position,
                "total_packets": EXPECTED_CLIP_COUNT,
                "task": source_clip.task,
                "task_instruction": task.instruction,
                "task_rubric": task.rubric,
                "review_guidance": REVIEW_GUIDANCE,
                "raw_video_url": "%s/api/artifacts/review/%s/media/%s/source.mp4" % (
                    base,
                    public.name,
                    public_media_id,
                ),
                "sampled_frames": public_frames,
                "integrity": "",
                "collision": "",
                "progress": "",
                "completion_evidence": "",
                "evidence_frame_indices": [],
                "observable_reasons": "",
            }
            _assert_public_packet(packet, source_clip)
            worksheets.append(packet)
            resolver_entries[public_media_id] = {
                "public_clip_id": public_clip_id,
                "source_clip_id": source_clip.source_clip_id,
                "original_cohort": source_clip.source_cohort,
                "source_lineage_id": source_clip.source_lineage_id,
                "task": source_clip.task,
                "input_manifest_sha256": "sha256:" + candidate_manifest_sha256,
                "source_video": {
                    "source_path": source_clip.video_relative_path,
                    "source_file_sha256": "sha256:" + source_clip.video_sha256,
                    "public_relative_path": "media/%s/source.mp4" % public_media_id,
                },
                "sampled_frames": private_frames,
                "reserved": True,
                "purpose": PURPOSE,
            }
        jsonl = "".join(json.dumps(packet, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for packet in worksheets)
        _write_new_text(stage / "blank-worksheets.jsonl", jsonl)
        index_lines = [
            "# Development review packets",
            "",
            "Purpose: `development_review_only`. These packets are unqualified, not Gate D eligible, and may not be exported as a primary result.",
            "",
            "All worksheets are unassigned and blank. Two actual reviewers may later provide independent reviews; this export does not create or name them.",
            "",
            "| Packet | Blank worksheet row | Raw video | Sampled frames |",
            "|---|---|---|---|",
        ]
        for packet in worksheets:
            frames = " ".join("[f%02d](%s)" % (item["index"], item["url"]) for item in packet["sampled_frames"])
            index_lines.append(
                "| `%s` | `blank-worksheets.jsonl` | [video](%s) | %s |"
                % (packet["clip_id"], packet["raw_video_url"], frames)
            )
        _write_new_text(stage / "INDEX.md", "\n".join(index_lines) + "\n")
        resolver = {
            "schema": "plumb-development-review-private-resolver-v1",
            "purpose": PURPOSE,
            "qualified": False,
            "gate_d_eligible": False,
            "candidate_manifest": {"path": str(_absolute(candidate_manifest)), "sha256": "sha256:" + candidate_manifest_sha256},
            "reserved_source_clip_count": len(resolver_entries),
            "media": resolver_entries,
        }
        _write_private_resolver(private, resolver, served)
        os.rename(stage, public)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        # If the resolver was created but public publication did not complete,
        # it is an orphaned coordinator artifact and no public packet references
        # it.  Leave it untouched for an explicit coordinator review.
        raise
    return {
        "public_out": str(public),
        "index": str(public / "INDEX.md"),
        "worksheets": str(public / "blank-worksheets.jsonl"),
        "private_resolver": str(private),
        "packets": len(worksheets),
        "purpose": PURPOSE,
        "qualified": False,
        "gate_d_eligible": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--public-out", type=Path, required=True)
    parser.add_argument("--private-resolver", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args(argv)
    try:
        result = prepare_review_packets(
            candidate_manifest=args.candidate_manifest,
            dataset_root=args.dataset_root,
            public_out=args.public_out,
            private_resolver=args.private_resolver,
            base_url=args.base_url,
        )
    except (ReviewPreparationError, FileExistsError, OSError) as error:
        print("review packet preparation failed: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
