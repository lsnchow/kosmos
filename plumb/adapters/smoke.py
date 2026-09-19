"""Run an actual local Cosmos3-Nano vendor/Bridge fixture smoke call.

This command performs no model or fixture downloads.  Put the immutable model
snapshot and the small official fixture files on the GPU worker first, then:

.. code-block:: bash

   python -m plumb.adapters.smoke cosmos \
     --model-path /models/Cosmos3-Nano \
     --actions /fixtures/bridge_20260501_0.json \
     --vision /fixtures/bridge_20260501_0.mp4 \
     --report /tmp/cosmos-smoke.json

The resulting report records actual adapter-call timing/memory/counts.  A
passing process is *not* a Gate-A pass: attach the report and fixture hashes to
the gates ledger, then evaluate its predefined thresholds independently.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import FeedbackMode, WorldRequest
from .worlds import (
    COSMOS3_BRIDGE_FIXTURE_ACTION_SHA256,
    COSMOS3_BRIDGE_FIXTURE_ACTION_URL,
    COSMOS3_BRIDGE_FIXTURE_VIDEO_URL,
    COSMOS3_NANO_MODEL_REVISION,
    DIFFUSERS_COSMOS3_PIN,
    Cosmos3NanoDiffusersAdapter,
    Cosmos3NanoDiffusersProfile,
)


OFFICIAL_BRIDGE_ACTION_COUNT = 16
OFFICIAL_BRIDGE_ACTION_WIDTH = 10


def _file_record(path: Path, *, source: Optional[str] = None) -> Dict[str, Any]:
    """Return evidence for an existing local file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    record: Dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": "sha256:" + digest.hexdigest(),
    }
    if source:
        record["source"] = source
    return record


def _read_action_rows(path: Path) -> Tuple[Tuple[float, ...], ...]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("actions", "action", "raw_actions"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("Action fixture must be a JSON list or an object containing actions/action/raw_actions.")
    rows = tuple(tuple(float(value) for value in row) for row in payload)
    if len(rows) != OFFICIAL_BRIDGE_ACTION_COUNT or any(len(row) != OFFICIAL_BRIDGE_ACTION_WIDTH for row in rows):
        raise ValueError(
            "Official Cosmos Bridge fixture must be exactly [%d,%d], got [%d,%s]."
            % (
                OFFICIAL_BRIDGE_ACTION_COUNT,
                OFFICIAL_BRIDGE_ACTION_WIDTH,
                len(rows),
                sorted({len(row) for row in rows}) if rows else "none",
            )
        )
    return rows


def _load_conditioning_first_frame(path: Path) -> Any:
    """Use optional media imports only after a real smoke was requested."""

    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        try:
            from PIL import Image  # type: ignore
        except ImportError as error:
            raise RuntimeError("Pillow is needed to load an image fixture.") from error
        return Image.open(path).convert("RGB")
    if suffix in {".mp4", ".mov", ".avi", ".webm"}:
        try:
            from diffusers.utils import load_video  # type: ignore
        except ImportError as error:
            raise RuntimeError("Pinned Diffusers media utilities are needed to decode a video fixture.") from error
        video = load_video(str(path))
        if len(video) < 1:
            raise ValueError("Conditioning fixture video has no frames.")
        return video[0]
    raise ValueError("Unsupported conditioning fixture suffix %r." % suffix)


def _detected_package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package in ("diffusers", "torch", "transformers", "accelerate", "huggingface_hub", "numpy", "imageio", "pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _runtime_context(adapter: Optional[Cosmos3NanoDiffusersAdapter] = None) -> Dict[str, Any]:
    packages = _detected_package_versions()
    context: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": packages,
        "version_tuple": {
            "diffusers": packages["diffusers"],
            "torch": packages["torch"],
            "transformers": packages["transformers"],
            "accelerate": packages["accelerate"],
            "huggingface_hub": packages["huggingface_hub"],
        },
    }
    runtime = getattr(adapter, "_runtime", None) if adapter is not None else None
    torch = getattr(runtime, "torch", None)
    if torch is not None:
        torch_context: Dict[str, Any] = {"version": getattr(torch, "__version__", None)}
        cuda = getattr(torch, "cuda", None)
        try:
            torch_context["cuda_available"] = bool(cuda is not None and cuda.is_available())
            torch_context["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
            if torch_context["cuda_available"]:
                torch_context["device_name"] = cuda.get_device_name(0)
                torch_context["device_count"] = cuda.device_count()
                torch_context["device_total_memory_bytes"] = int(cuda.get_device_properties(0).total_memory)
        except (AttributeError, RuntimeError):
            torch_context["cuda_observation_error"] = "unavailable"
        context["torch_runtime"] = torch_context
    return context


def _discover_model_manifest(model_path: Path, supplied: Optional[str]) -> Optional[Path]:
    if supplied:
        return Path(supplied)
    # Matches cluster/download_assets.py without making an external or recursive
    # discovery call. A missing manifest remains explicit in the report.
    try:
        root = model_path.parent.parent
        candidate = root / "evidence" / (model_path.name + "-download.json")
    except IndexError:
        return None
    return candidate if candidate.is_file() else None


def _provenance_context(action_path: Path, vision_path: Path, model_path: Path, model_manifest: Optional[str]) -> Dict[str, Any]:
    action_record = (
        _file_record(action_path, source=COSMOS3_BRIDGE_FIXTURE_ACTION_URL) if action_path.is_file() else None
    )
    video_record = (
        _file_record(vision_path, source=COSMOS3_BRIDGE_FIXTURE_VIDEO_URL) if vision_path.is_file() else None
    )
    result: Dict[str, Any] = {
        "action_fixture": action_record,
        "vision_fixture": video_record,
    }
    manifest = _discover_model_manifest(model_path, model_manifest)
    result["model_manifest"] = _file_record(manifest) if manifest is not None and manifest.is_file() else None
    result["model_manifest_missing"] = manifest is None or not manifest.is_file()
    if action_record is not None:
        expected = "sha256:" + COSMOS3_BRIDGE_FIXTURE_ACTION_SHA256
        action_record["expected_sha256"] = expected
        action_record["pinned_sha256_matches"] = action_record["sha256"] == expected
    return result


def _normalise_frame_to_uint8(frame: Any) -> Any:
    """Convert a Diffusers ``output_type='np'`` frame to HWC uint8 RGB.

    The current Cosmos action pipeline returns HWC RGB uint8, but accepting
    float [0,1], float [-1,1], and CHW representations makes artifact export a
    validating boundary rather than a silent source of black/washed-out video.
    """

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("numpy is required to export Cosmos smoke frame artifacts.") from error
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError("Expected one 3-D Cosmos frame, got shape %s." % (array.shape,))
    if array.shape[-1] not in (1, 3, 4) and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError("Expected RGB/RGBA/gray frame, got shape %s." % (array.shape,))
    if not np.all(np.isfinite(array)):
        raise ValueError("Cannot export a frame with non-finite values.")
    if np.issubdtype(array.dtype, np.floating):
        low, high = float(array.min()), float(array.max())
        if low >= -1.0001 and high <= 1.0001:
            if low < 0:
                array = (array + 1.0) * 127.5
            else:
                array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).round().astype(np.uint8))


def _write_visual_artifacts(
    frames: Sequence[Any],
    *,
    fps: float,
    output_video: Optional[str],
    output_frame_dir: Optional[str],
    frame_selection: str,
) -> Dict[str, Any]:
    """Write optional MP4/PNG artifacts and return their computed hashes."""

    if not output_video and not output_frame_dir:
        return {}
    normalised = tuple(_normalise_frame_to_uint8(frame) for frame in frames)
    artifacts: Dict[str, Any] = {}
    if output_video:
        destination = Path(output_video)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v3 as iio
            import numpy as np
        except ImportError as error:
            raise RuntimeError("imageio and numpy are required for --output-video.") from error
        # imageio-ffmpeg produces a real H.264 MP4 from explicit uint8 THWC.
        iio.imwrite(str(destination), np.stack(normalised, axis=0), fps=float(fps), codec="libx264")
        artifacts["video"] = _file_record(destination)
        artifacts["video"]["frame_count"] = len(normalised)
        artifacts["video"]["fps"] = float(fps)
        artifacts["video"]["pixel_format"] = "rgb24"
    if output_frame_dir:
        directory = Path(output_frame_dir)
        directory.mkdir(parents=True, exist_ok=True)
        selected = range(len(normalised)) if frame_selection == "all" else sorted({0, len(normalised) - 1})
        try:
            import imageio.v3 as iio
            write_png = lambda destination, frame: iio.imwrite(str(destination), frame, extension=".png")
        except ImportError:
            try:
                from PIL import Image
            except ImportError as error:
                raise RuntimeError("imageio or Pillow is required for --output-frame-dir.") from error
            write_png = lambda destination, frame: Image.fromarray(frame).save(destination, format="PNG")
        pngs = []
        for index in selected:
            destination = directory / ("frame-%03d.png" % index)
            write_png(destination, normalised[index])
            record = _file_record(destination)
            record["frame_index"] = index
            pngs.append(record)
        artifacts["frames"] = pngs
        artifacts["frame_selection"] = frame_selection
    return artifacts


def _request_context(request: Optional[WorldRequest], action_count: int, probe_length: Optional[int]) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "fixture_action_count": OFFICIAL_BRIDGE_ACTION_COUNT,
        "fixture_action_width": OFFICIAL_BRIDGE_ACTION_WIDTH,
        "selected_action_count": action_count,
        "probe_length": probe_length,
        "execution_kind": "gate_b_probe" if probe_length else "normal_fixture_smoke",
    }
    if request is not None:
        context.update({
            "profile_id": request.compatibility_profile_id,
            "prompt": request.prompt,
            "domain": request.domain,
            "action_count": len(request.compiled_actions),
            "action_width": len(request.compiled_actions[0]),
            "feedback_mode": request.feedback_mode.value,
            "seed": request.seed,
            "request_id": request.request_id,
        })
    return context


def _report(
    result: Any,
    adapter: Cosmos3NanoDiffusersAdapter,
    request: WorldRequest,
    *,
    provenance: Mapping[str, Any],
    output_artifacts: Mapping[str, Any],
    probe_length: Optional[int],
) -> dict:
    qualification = "gate_b_probe_not_normal_call" if probe_length else "not_qualified_by_smoke"
    profile = adapter.profile
    return {
        "schema_version": 1,
        "kind": "cosmos3_nano_diffusers_smoke",
        "status": "completed",
        "qualification": qualification,
        "capability_before_call": {
            "status": adapter.capability().status.value,
            "reason": adapter.capability().reason,
            "source_verified": adapter.capability().source_verified,
            "evidence_uris": list(adapter.capability().evidence_uris),
        },
        "request": _request_context(request, len(request.compiled_actions), probe_length),
        "provenance": dict(provenance),
        "runtime": _runtime_context(adapter),
        "profile": {
            "model_path": profile.local_model_path,
            "model_revision": profile.model_revision,
            "diffusers_revision": profile.diffusers_revision,
            "container_digest": profile.container_digest,
            "normalizer_revision": profile.normalizer_revision,
            "resolution_tier": profile.resolution_tier,
            "fps": profile.fps,
            "num_inference_steps": profile.num_inference_steps,
            "guidance_scale": profile.guidance_scale,
            "scheduler_flow_shift": profile.scheduler_flow_shift,
            "local_files_only": profile.local_files_only,
            "enable_safety_checker": profile.enable_safety_checker,
        },
        "result": {
            "backend": result.backend,
            "profile_id": result.profile_id,
            "returned_frame_count": len(result.frames),
            "future_frame_count": len(result.future_frames),
            "conditioning_frame_included": result.conditioning_frame_included,
            "nominal_frame_timestamps": list(result.nominal_frame_timestamps),
            "timing": result.timing.as_dict(),
            "metadata": dict(result.metadata),
            "artifact_hashes": dict(result.artifact_hashes),
            "output_artifacts": dict(output_artifacts),
        },
        "note": (
            "All timing/memory/count fields above came from this adapter call. Null is unknown, not zero. "
            "This smoke has not evaluated Gate A or Gate B. Probe output is only a controlled Gate-B experiment."
        ),
    }


def _failure_report(
    error: BaseException,
    *,
    args: argparse.Namespace,
    request: Optional[WorldRequest],
    action_count: int,
    adapter: Optional[Cosmos3NanoDiffusersAdapter],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "cosmos3_nano_diffusers_smoke",
        "status": "failed",
        "qualification": "not_qualified_by_smoke",
        "request": _request_context(request, action_count, args.probe_length),
        "provenance": dict(provenance),
        "runtime": _runtime_context(adapter),
        "profile": {
            "model_path": args.model_path,
            "model_revision": args.model_revision,
            "diffusers_revision": args.diffusers_revision,
            "container_digest": args.container_digest,
            "normalizer_revision": args.normalizer_revision,
            "resolution_tier": args.resolution_tier,
            "fps": args.fps,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "note": "Failure is durable evidence, not a pass or an inferred zero-performance result.",
    }


def _write_report(path: Optional[str], report: Mapping[str, Any]) -> Path:
    destination = Path(path) if path else Path.cwd() / "cosmos-smoke-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def cosmos_command(args: argparse.Namespace) -> int:
    action_path = Path(args.actions)
    vision_path = Path(args.vision)
    model_path = Path(args.model_path)
    action_count = OFFICIAL_BRIDGE_ACTION_COUNT
    adapter: Optional[Cosmos3NanoDiffusersAdapter] = None
    request: Optional[WorldRequest] = None
    provenance: Mapping[str, Any] = {
        "action_fixture": None,
        "vision_fixture": None,
        "model_manifest": None,
        "model_manifest_missing": True,
    }
    # Dry run deliberately performs no fixture decoding or ML import.  It can
    # therefore run on a control-plane host before files are synchronized.
    selected_count = args.probe_length or action_count
    try:
        provenance = _provenance_context(action_path, vision_path, model_path, args.model_manifest)
        if action_path.is_file():
            _read_action_rows(action_path)  # enforce exact [16,10] before setup.
            action_evidence = provenance.get("action_fixture")
            if isinstance(action_evidence, Mapping) and not action_evidence.get("pinned_sha256_matches"):
                raise ValueError(
                    "Action fixture SHA-256 does not match the pinned official Bridge fixture; refusing an unlabelled substitute."
                )
        profile = Cosmos3NanoDiffusersProfile(
            profile_id=args.profile_id,
            local_model_path=args.model_path,
            model_revision=args.model_revision,
            diffusers_revision=args.diffusers_revision,
            container_digest=args.container_digest,
            normalizer_revision=args.normalizer_revision,
            resolution_tier=args.resolution_tier,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            allowed_action_lengths=(OFFICIAL_BRIDGE_ACTION_COUNT,),
            probe_action_lengths=(args.probe_length,) if args.probe_length else (),
        )
        adapter = Cosmos3NanoDiffusersAdapter(profile)
        if args.dry_run:
            report = {
                "schema_version": 1,
                "kind": "cosmos3_nano_diffusers_smoke",
                "status": "not_run",
                "qualification": "not_qualified_by_smoke",
                "request": _request_context(None, selected_count, args.probe_length),
                "provenance": provenance,
                "runtime": _runtime_context(),
                "capability": {
                    "status": adapter.capability().status.value,
                    "reason": adapter.capability().reason,
                    "source_verified": adapter.capability().source_verified,
                    "evidence_uris": list(adapter.capability().evidence_uris),
                },
            }
            _write_report(args.report, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if not action_path.is_file() or not vision_path.is_file():
            raise FileNotFoundError("Both --actions and --vision must be local fixture files.")
        fixture_rows = _read_action_rows(action_path)
        rows = fixture_rows[: args.probe_length] if args.probe_length else fixture_rows
        frame = _load_conditioning_first_frame(vision_path)
        request = WorldRequest(
            conditioning_image=frame,
            prompt=args.prompt,
            domain="bridge_orig_lerobot",
            compiled_actions=rows,
            nominal_control_timestamps=tuple((index + 1) / args.fps for index in range(len(rows))),
            seed=args.seed,
            compatibility_profile_id=args.profile_id,
            feedback_mode=FeedbackMode.UNQUALIFIED,
            request_id=args.request_id,
        )
        result = adapter.generate(request)
        output_artifacts = _write_visual_artifacts(
            result.frames,
            fps=args.fps,
            output_video=args.output_video,
            output_frame_dir=args.output_frame_dir,
            frame_selection=args.frame_selection,
        )
        report = _report(
            result,
            adapter,
            request,
            provenance=provenance,
            output_artifacts=output_artifacts,
            probe_length=args.probe_length,
        )
        _write_report(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        report = _failure_report(
            error,
            args=args,
            request=request,
            action_count=selected_count,
            adapter=adapter,
            provenance=provenance,
        )
        destination = _write_report(args.report, report)
        error.smoke_report_path = str(destination)  # type: ignore[attr-defined]
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m plumb.adapters.smoke")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cosmos = subparsers.add_parser("cosmos", help="Run a local official Cosmos3-Nano FD fixture.")
    cosmos.add_argument("--model-path", required=True, help="Downloaded local nvidia/Cosmos3-Nano snapshot directory.")
    cosmos.add_argument("--actions", required=True, help="Local official [N,10] action JSON fixture.")
    cosmos.add_argument("--vision", required=True, help="Local official condition image or video fixture.")
    cosmos.add_argument("--report", help="Write success/failure evidence JSON here (default ./cosmos-smoke-report.json).")
    cosmos.add_argument("--model-manifest", help="Pinned local model download manifest; autodiscovered beside cluster evidence when omitted.")
    cosmos.add_argument("--output-video", help="Optional MP4 visual artifact path (condition plus all generated frames).")
    cosmos.add_argument("--output-frame-dir", help="Optional PNG visual artifact directory.")
    cosmos.add_argument("--frame-selection", choices=("first-last", "all"), default="first-last",
                        help="PNG selection when --output-frame-dir is set.")
    cosmos.add_argument("--profile-id", default="cosmos3-nano-diffusers-fd-smoke")
    cosmos.add_argument("--model-revision", default=COSMOS3_NANO_MODEL_REVISION)
    cosmos.add_argument("--diffusers-revision", default=DIFFUSERS_COSMOS3_PIN)
    cosmos.add_argument("--container-digest", default=None)
    cosmos.add_argument("--normalizer-revision", default=None)
    cosmos.add_argument("--prompt", default="Put the pot to the left of the purple item.")
    cosmos.add_argument("--seed", type=int, default=0)
    cosmos.add_argument("--request-id", default="vendor-bridge-smoke")
    cosmos.add_argument("--fps", type=float, default=5.0)
    cosmos.add_argument("--resolution-tier", type=int, choices=(256, 480, 704, 720), default=480)
    cosmos.add_argument("--steps", type=int, default=30)
    cosmos.add_argument("--guidance-scale", type=float, default=1.0)
    cosmos.add_argument("--probe-length", type=int, choices=(1, 4), default=None,
                        help="Controlled Gate-B probe: use only the 1- or 4-action fixture prefix; never a normal call.")
    cosmos.add_argument("--dry-run", action="store_true", help="Validate local profile availability without loading ML packages.")
    cosmos.set_defaults(handler=cosmos_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:  # CLI needs a concise machine-visible failure.
        report_path = getattr(error, "smoke_report_path", None)
        suffix = " (failure report: %s)" % report_path if report_path else ""
        print("smoke failed: %s%s" % (error, suffix), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
