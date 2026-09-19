#!/usr/bin/env python3
"""Local-only cluster smoke calls for the reviewed OpenVLA and Qwen adapters.

This command is deliberately narrower than an evaluation runner.  It records
one OpenVLA action from the first official Bridge fixture frame, or raw Qwen
rubric votes from a 16-frame Cosmos MP4 sample.  A successful process is not a
policy, judge, Gate-A, Gate-B, or Gate-D qualification result.

Run from a cluster checkout with the relevant *separate* virtual environment:

    PLUMB_ROOT=/scratch/<user>/plumb \
      $PLUMB_ROOT/venv-openvla-tf440/bin/python cluster/policy_smoke.py openvla ...

No Hub/model download is attempted.  ``HF_HUB_OFFLINE`` and
``TRANSFORMERS_OFFLINE`` are made true before any adapter is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import tempfile
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


DEFAULT_PLUMB_ROOT = Path("/scratch/lchow432/plumb")
REVIEWED_OPENVLA_REMOTE_CODE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
OPENVLA_MODEL_ID = "openvla/openvla-7b"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
VIDEO_SUFFIXES = frozenset((".mp4", ".mov", ".mkv", ".avi", ".webm"))
IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".bmp", ".webp"))


class SmokeInputError(ValueError):
    """A smoke request does not meet its intentionally narrow local contract."""


def plumb_root(value: Optional[object] = None) -> Path:
    """Resolve an explicit root or the portable cluster-root environment value."""

    chosen = value if value is not None else os.environ.get("PLUMB_ROOT", str(DEFAULT_PLUMB_ROOT))
    if not isinstance(chosen, (str, os.PathLike)):
        raise SmokeInputError("PLUMB_ROOT/--root must be a filesystem path.")
    return Path(chosen).expanduser()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError("Expected local file: %s" % candidate)
    return {
        "path": str(candidate.resolve()),
        "bytes": candidate.stat().st_size,
        "sha256": _sha256(candidate),
    }


def _is_immutable_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def model_metadata(manifest_path: Optional[Path], explicit_revision: Optional[str]) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
    """Resolve a full model revision, preferring the persisted download manifest."""

    manifest: Optional[Dict[str, Any]] = None
    manifest_revision: Optional[str] = None
    if manifest_path is not None and manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SmokeInputError("Model manifest is not valid JSON: %s" % manifest_path) from error
        if not isinstance(raw, dict):
            raise SmokeInputError("Model manifest must be a JSON object: %s" % manifest_path)
        manifest = dict(raw)
        candidate = raw.get("revision")
        if isinstance(candidate, str):
            manifest_revision = candidate
    if explicit_revision is not None and not _is_immutable_revision(explicit_revision):
        raise SmokeInputError("--model-revision must be a full 40-character hexadecimal commit.")
    if manifest_revision is not None and not _is_immutable_revision(manifest_revision):
        raise SmokeInputError("Model manifest revision must be a full 40-character hexadecimal commit.")
    if explicit_revision and manifest_revision and explicit_revision != manifest_revision:
        raise SmokeInputError("--model-revision disagrees with the local download manifest revision.")
    revision = manifest_revision or explicit_revision
    if revision is None:
        raise SmokeInputError("Supply a valid local model manifest or an explicit immutable --model-revision.")
    return revision, manifest_revision is not None, manifest


def uniform_video_indices(frame_count: int, sample_count: int = 16) -> Tuple[int, ...]:
    """Select exactly ``sample_count`` ordered frames including both endpoints."""

    if frame_count < sample_count:
        raise SmokeInputError(
            "Video contains %d frames; the frozen judge smoke needs at least %d and will not pad or repeat frames."
            % (frame_count, sample_count)
        )
    if sample_count < 2:
        raise SmokeInputError("sample_count must be at least two to retain both endpoints.")
    indices = tuple((position * (frame_count - 1) + (sample_count - 1) // 2) // (sample_count - 1) for position in range(sample_count))
    if indices[0] != 0 or indices[-1] != frame_count - 1 or len(set(indices)) != sample_count:
        raise SmokeInputError("Could not select unique uniformly spaced endpoint-inclusive video frames.")
    return indices


def _load_first_frame(path: Path) -> Any:
    """Lazily decode just the first image/video frame without model imports."""

    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        try:
            from PIL import Image  # type: ignore
        except ImportError as error:
            raise RuntimeError("Pillow is required to load an image fixture.") from error
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    if suffix in VIDEO_SUFFIXES:
        try:
            import imageio.v3 as iio  # type: ignore
        except ImportError as error:
            raise RuntimeError("imageio is required to decode an MP4 fixture.") from error
        return iio.imread(path, index=0)
    raise SmokeInputError("Unsupported image/video fixture suffix %r." % suffix)


def _load_video_frames(path: Path) -> Tuple[Any, ...]:
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise SmokeInputError("Judge smoke input must be a generated video (MP4/MOV/MKV/AVI/WEBM), not %s." % path)
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError as error:
        raise RuntimeError("imageio is required to decode the generated Cosmos MP4.") from error
    frames = tuple(iio.imiter(path))
    if not frames:
        raise SmokeInputError("Generated Cosmos video has no decodable frames.")
    return frames


def _reference_frame(path: Path, role: str, first_frame_loader: Callable[[Path], Any]) -> Any:
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES and role != "scene":
        raise SmokeInputError("A video first frame is context only; pass --reference-role scene, never goal.")
    return first_frame_loader(path)


def _package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    values: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_json_value(payload), stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _default_report_path(root: Path, command: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "evidence" / ("policy-smoke-%s-%s.json" % (command, stamp))


def _capability_json(adapter: Any) -> Optional[Dict[str, Any]]:
    capability = getattr(adapter, "capability", None)
    if not callable(capability):
        return None
    result = capability()
    status = getattr(result, "status", None)
    return {
        "status": getattr(status, "value", status),
        "reason": getattr(result, "reason", None),
        "source_verified": getattr(result, "source_verified", None),
        "evidence_uris": list(getattr(result, "evidence_uris", ()) or ()),
        "details": _json_value(getattr(result, "details", {})),
    }


def _openvla_dependencies() -> Tuple[Any, Any, Any, float]:
    started = time.perf_counter()
    module = importlib.import_module("plumb.policies.openvla")
    contracts = importlib.import_module("plumb.adapters.contracts")
    return module.OpenVLAPolicyProfile, module.OpenVLAPolicyAdapter, contracts.PolicyObservation, time.perf_counter() - started


def _judge_dependencies() -> Tuple[Any, Any, Any, Any, float]:
    started = time.perf_counter()
    module = importlib.import_module("plumb.policies.judge")
    return (
        module.QwenJudgeProfile,
        module.QwenRubricJudge,
        module.JudgeRequest,
        module.ReferenceImage,
        time.perf_counter() - started,
    )


def _manifest_record(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    return file_record(path) if path is not None and path.is_file() else None


def _openvla_smoke(
    args: argparse.Namespace,
    report: Dict[str, Any],
    *,
    first_frame_loader: Callable[[Path], Any] = _load_first_frame,
    dependency_loader: Callable[[], Tuple[Any, Any, Any, float]] = _openvla_dependencies,
    adapter_factory: Optional[Callable[[Any, Any], Any]] = None,
) -> None:
    model_path = Path(args.model_path)
    vision_path = Path(args.vision)
    manifest_path = Path(args.model_manifest) if args.model_manifest else None
    if not model_path.is_dir():
        raise FileNotFoundError("OpenVLA local model directory is absent: %s" % model_path)
    revision, from_metadata, metadata = model_metadata(manifest_path, args.model_revision)
    if revision != REVIEWED_OPENVLA_REMOTE_CODE_REVISION:
        raise SmokeInputError(
            "OpenVLA smoke only enables the reviewed remote-code revision %s, not %s."
            % (REVIEWED_OPENVLA_REMOTE_CODE_REVISION, revision)
        )
    if not isinstance(args.task, str) or not args.task.strip():
        raise SmokeInputError("--task must be the exact nonempty vendor task instruction.")
    report["inputs"] = {
        "vision": file_record(vision_path),
        "first_frame_index": 0,
        "task_instruction": args.task,
    }
    report["model"] = {
        "id": OPENVLA_MODEL_ID,
        "local_path": str(model_path.resolve()),
        "revision": revision,
        "revision_from_metadata": from_metadata,
        "manifest": _manifest_record(manifest_path),
        "metadata": metadata,
        "reviewed_remote_code_revision": REVIEWED_OPENVLA_REMOTE_CODE_REVISION,
        "local_files_only": True,
        "safetensors_required": True,
    }
    frame_started = time.perf_counter()
    first_frame = first_frame_loader(vision_path)
    report["timing"]["fixture_decode_seconds"] = time.perf_counter() - frame_started
    profile_cls, adapter_cls, observation_cls, import_seconds = dependency_loader()
    report["timing"]["adapter_import_seconds"] = import_seconds
    profile = profile_cls(
        profile_id="openvla-bridge-first-frame-smoke",
        local_model_path=str(model_path),
        checkpoint_revision=revision,
        remote_code_revision=REVIEWED_OPENVLA_REMOTE_CODE_REVISION,
        reviewed_remote_code_revision=REVIEWED_OPENVLA_REMOTE_CODE_REVISION,
        allow_trust_remote_code=True,
    )
    adapter = adapter_factory(profile, adapter_cls) if adapter_factory is not None else adapter_cls(profile)
    report["capability_before_call"] = _capability_json(adapter)
    call_started = time.perf_counter()
    action_report = adapter.predict_with_report(
        observation_cls(image_history=(first_frame,), prompt=args.task, timestamp=0.0)
    )
    report["timing"]["policy_call_seconds"] = time.perf_counter() - call_started
    action = tuple(getattr(action_report, "action"))
    if len(action) != 7:
        raise SmokeInputError("OpenVLA smoke expected exactly one native 7-D action, got %d values." % len(action))
    report["action"] = {
        "values": [float(value) for value in action],
        "dimension": 7,
        "normalization": getattr(action_report, "unnorm_key", None),
        "source_image_timestamp": getattr(action_report, "source_image_timestamp", None),
        "backend_calls": getattr(action_report, "backend_calls", None),
        "wall_seconds": getattr(action_report, "wall_seconds", None),
        "gpu_peak_memory_bytes": getattr(action_report, "gpu_peak_memory_bytes", None),
    }
    report["imports"] = _package_versions(("torch", "torchvision", "transformers", "tokenizers", "timm", "safetensors"))


def _judge_smoke(
    args: argparse.Namespace,
    report: Dict[str, Any],
    *,
    video_loader: Callable[[Path], Tuple[Any, ...]] = _load_video_frames,
    first_frame_loader: Callable[[Path], Any] = _load_first_frame,
    dependency_loader: Callable[[], Tuple[Any, Any, Any, Any, float]] = _judge_dependencies,
    judge_factory: Optional[Callable[[Any, Any], Any]] = None,
) -> None:
    model_path = Path(args.model_path)
    video_path = Path(args.video)
    reference_path = Path(args.reference)
    manifest_path = Path(args.model_manifest) if args.model_manifest else None
    if not model_path.is_dir():
        raise FileNotFoundError("Qwen local model directory is absent: %s" % model_path)
    revision, from_metadata, metadata = model_metadata(manifest_path, args.model_revision)
    if not isinstance(args.task, str) or not args.task.strip():
        raise SmokeInputError("--task must be an exact nonempty task instruction.")
    if not isinstance(args.rubric, str) or not args.rubric.strip():
        raise SmokeInputError("--rubric must be an explicit nonempty frozen task rubric.")
    if args.diagnostic is not True:
        raise SmokeInputError(
            "The free-text judge smoke is diagnostic only; pass --diagnostic explicitly. It cannot call primary scoring."
        )
    if args.reference_role not in ("goal", "scene"):
        raise SmokeInputError("--reference-role must be goal or scene.")
    seeds = tuple(args.seeds)
    if len(seeds) != 5 or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise SmokeInputError("--seeds must provide exactly five integer judge seeds.")
    if len(set(seeds)) != 5:
        raise SmokeInputError("--seeds must provide five distinct logged judge seeds.")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise SmokeInputError("--fps must be a finite positive nominal frame rate.")
    report["inputs"] = {
        "mode": "diagnostic",
        "generated_video": file_record(video_path),
        "reference": file_record(reference_path),
        "reference_role": args.reference_role,
        "task_instruction": args.task,
        "task_rubric": args.rubric,
        "judge_seeds": list(seeds),
        "nominal_video_fps": args.fps,
    }
    report["model"] = {
        "id": QWEN_MODEL_ID,
        "local_path": str(model_path.resolve()),
        "revision": revision,
        "processor_revision": args.processor_revision or revision,
        "revision_from_metadata": from_metadata,
        "manifest": _manifest_record(manifest_path),
        "metadata": metadata,
        "transformers_version": args.transformers_version,
        "local_files_only": True,
        "trust_remote_code": False,
        "safetensors_required": True,
    }
    decode_started = time.perf_counter()
    video_frames = video_loader(video_path)
    selected_indices = uniform_video_indices(len(video_frames), 16)
    selected_frames = tuple(video_frames[index] for index in selected_indices)
    reference_image = _reference_frame(reference_path, args.reference_role, first_frame_loader)
    report["timing"]["media_decode_seconds"] = time.perf_counter() - decode_started
    report["inputs"]["source_video_frame_count"] = len(video_frames)
    report["inputs"]["selected_frame_indices"] = list(selected_indices)
    report["inputs"]["frame_timestamps"] = [float(index) / float(args.fps) for index in selected_indices]
    profile_cls, judge_cls, request_cls, reference_cls, import_seconds = dependency_loader()
    report["timing"]["adapter_import_seconds"] = import_seconds
    profile = profile_cls(
        profile_id="qwen-cosmos-video-smoke",
        local_model_path=str(model_path),
        model_revision=revision,
        processor_revision=args.processor_revision or revision,
        transformers_version=args.transformers_version,
    )
    judge = judge_factory(profile, judge_cls) if judge_factory is not None else judge_cls(profile)
    report["capability_before_call"] = _capability_json(judge)
    reference = reference_cls(
        image=reference_image,
        source_uri="artifact://%s" % reference_path.name,
        sha256=report["inputs"]["reference"]["sha256"],
    )
    request = request_cls(
        frames=selected_frames,
        frame_timestamps=tuple(report["inputs"]["frame_timestamps"]),
        diagnostic_mode=True,
        task_instruction=args.task,
        task_rubric=args.rubric,
        reference_images=(reference,),
    )
    call_started = time.perf_counter()
    judge_report = judge.evaluate(request, seeds=seeds)
    report["timing"]["judge_call_seconds"] = time.perf_counter() - call_started
    raw = judge_report.as_dict() if callable(getattr(judge_report, "as_dict", None)) else _json_value(judge_report)
    report["raw_judge_votes"] = raw
    # This is a smoke fixture.  A context frame is expressly not a goal image,
    # and no VLM vote is promoted to an asserted task success in either role.
    report["outcome"] = {
        "asserted_success": None,
        "reason": (
            "reference_role_scene_context_only" if args.reference_role == "scene" else "unqualified_judge_smoke"
        ),
        "judge_status": getattr(judge_report, "judge_status", None),
        "raw_binary_success": getattr(judge_report, "binary_success", None),
        "raw_progress": getattr(judge_report, "progress", None),
    }
    report["imports"] = _package_versions(("torch", "transformers", "tokenizers", "safetensors", "pillow", "imageio"))


def run_smoke(
    args: argparse.Namespace,
    *,
    first_frame_loader: Callable[[Path], Any] = _load_first_frame,
    video_loader: Callable[[Path], Tuple[Any, ...]] = _load_video_frames,
    openvla_dependency_loader: Callable[[], Tuple[Any, Any, Any, float]] = _openvla_dependencies,
    judge_dependency_loader: Callable[[], Tuple[Any, Any, Any, Any, float]] = _judge_dependencies,
    adapter_factory: Optional[Callable[[Any, Any], Any]] = None,
    judge_factory: Optional[Callable[[Any, Any], Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Execute a bounded smoke subcommand and persist a report even on failure."""

    root = plumb_root(args.root)
    report_path = Path(args.report) if args.report else _default_report_path(root, args.command)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.perf_counter()
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "plumb_local_policy_smoke",
        "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
        "command": args.command,
        "started_at": _utc_now(),
        "plumb_root": str(root),
        "report_path": str(report_path),
        "qualified": False,
        "local_files_only": True,
        "offline_environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "timing": {},
        "status": "running",
    }
    try:
        if args.command == "openvla":
            _openvla_smoke(
                args,
                report,
                first_frame_loader=first_frame_loader,
                dependency_loader=openvla_dependency_loader,
                adapter_factory=adapter_factory,
            )
        elif args.command == "judge":
            _judge_smoke(
                args,
                report,
                video_loader=video_loader,
                first_frame_loader=first_frame_loader,
                dependency_loader=judge_dependency_loader,
                judge_factory=judge_factory,
            )
        else:
            raise SmokeInputError("Unsupported smoke command %r." % args.command)
        report["status"] = "completed_unqualified"
        code = 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(limit=8),
        }
        code = 2
    report["finished_at"] = _utc_now()
    report["timing"]["total_seconds"] = time.perf_counter() - started
    _atomic_json(report_path, report)
    return code, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        help="PLUMB scratch root. Defaults to $PLUMB_ROOT or /scratch/lchow432/plumb.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    openvla = commands.add_parser("openvla", help="Run one reviewed local OpenVLA Bridge first-frame action.")
    openvla.add_argument("--model-path", help="Local OpenVLA snapshot directory.")
    openvla.add_argument("--model-manifest", help="Local OpenVLA download manifest JSON.")
    openvla.add_argument("--model-revision", help="Fallback immutable 40-char checkpoint revision.")
    openvla.add_argument("--vision", required=True, help="Official vendor Bridge condition video/image; only frame 0 is used.")
    openvla.add_argument("--task", required=True, help="Exact vendor task instruction.")
    openvla.add_argument("--report", help="Output evidence JSON; persisted for success and failure.")

    judge = commands.add_parser("judge", help="Run five local Qwen rubric samples on a Cosmos MP4.")
    judge.add_argument("--model-path", help="Local Qwen snapshot directory.")
    judge.add_argument("--model-manifest", help="Local Qwen download manifest JSON.")
    judge.add_argument("--model-revision", help="Fallback immutable 40-char checkpoint revision.")
    judge.add_argument("--processor-revision", help="Pinned processor revision (default: model revision).")
    judge.add_argument("--transformers-version", default="4.49.0", help="Exact installed judge Transformers version.")
    judge.add_argument("--video", required=True, help="Generated Cosmos MP4 (must contain at least 16 decoded frames).")
    judge.add_argument("--fps", type=float, default=5.0, help="Declared nominal MP4/control frame rate for report timestamps.")
    judge.add_argument("--reference", required=True, help="Goal image or explicit scene-context image/video.")
    judge.add_argument("--reference-role", required=True, choices=("goal", "scene"), help="goal or scene; video first-frame context must be scene.")
    judge.add_argument("--task", required=True, help="Exact task instruction passed to the blinded judge.")
    judge.add_argument("--rubric", required=True, help="Frozen task-specific 0–5 rubric text (no success percentages).")
    judge.add_argument(
        "--diagnostic",
        action="store_true",
        help="Required acknowledgement: this free-text vendor smoke is diagnostic and never primary/Gate-D scoring.",
    )
    judge.add_argument("--seeds", type=int, nargs=5, required=True, metavar=("S1", "S2", "S3", "S4", "S5"), help="Exactly five logged judge sample seeds.")
    judge.add_argument("--report", help="Output evidence JSON; persisted for success and failure.")
    return parser


def _apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    root = plumb_root(args.root)
    if args.command == "openvla":
        if not args.model_path:
            args.model_path = str(root / "models" / "openvla--openvla-7b")
        if not args.model_manifest:
            args.model_manifest = str(root / "evidence" / "openvla--openvla-7b-download.json")
    elif args.command == "judge":
        if not args.model_path:
            args.model_path = str(root / "models" / "Qwen--Qwen2.5-VL-7B-Instruct")
        if not args.model_manifest:
            args.model_manifest = str(root / "evidence" / "Qwen--Qwen2.5-VL-7B-Instruct-download.json")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = _apply_defaults(parser.parse_args(argv))
    code, report = run_smoke(args)
    print(json.dumps(_json_value(report), sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
