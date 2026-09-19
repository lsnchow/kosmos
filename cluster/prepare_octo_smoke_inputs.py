#!/usr/bin/env python3
"""Prepare two provenance-bound canonical Octo observation PNGs on a login node.

AutoEval wraps its robot environment with ``ResizeObsImageWrapper`` targeting
``image_primary=(256, 256)``.  Its upstream implementation calls
``cv2.resize(image, (256, 256))`` with OpenCV's default INTER_LINEAR
interpolation.  VideoCapture exposes BGR while PLUMB's policy contract is RGB,
so this helper records the explicit BGR→RGB decoder conversion before applying
that resize to frames 0 and 1 of an existing immutable vendor video. It
preserves the original asset and exports a no-clobber sidecar with source,
raw-pixel and output hashes. The result is an *unqualified diagnostic input
profile*, not a claim of pixel identity with the unreleased robot camera
pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


AUTOEVAL_RESIZE_SOURCE_COMMIT = "3ea3ff44c6950433cfbcb4294a3deaa616533745"
MANIPULATOR_GYM_RESIZE_SOURCE_COMMIT = "c547d410b0738cfa94ebb0b9e64efa4084135f2e"
OUTPUT_SIZE = (256, 256)


class FixturePreparationError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cluster_root(path: Path) -> Path:
    root = path.resolve()
    if not str(root).startswith(("/scratch/", "/global/scratch/")) or root.name != "plumb":
        raise FixturePreparationError("Octo fixture preparation is allowed only in a cluster scratch/.../plumb directory")
    return root


def _write_new_bytes(path: Path, value: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError("Refusing to overwrite prepared Octo fixture file: %s" % path) from error
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    else:
        os.unlink(temporary_name)


def _frame_record(index: int, raw_bgr: Any, decoded_rgb: Any, resized_rgb: Any, output: Path) -> Dict[str, Any]:
    return {
        "source_frame_index": index,
        "source_bgr_shape": list(raw_bgr.shape),
        "source_bgr_dtype": str(raw_bgr.dtype),
        "source_bgr_pixel_sha256": _sha256_bytes(raw_bgr.tobytes()),
        "decoded_rgb_shape": list(decoded_rgb.shape),
        "decoded_rgb_dtype": str(decoded_rgb.dtype),
        "decoded_rgb_pixel_sha256": _sha256_bytes(decoded_rgb.tobytes()),
        "output_file": output.name,
        "output_file_sha256": _sha256_file(output),
        "output_rgb_shape": list(resized_rgb.shape),
        "output_rgb_dtype": str(resized_rgb.dtype),
        "output_rgb_pixel_sha256": _sha256_bytes(resized_rgb.tobytes()),
    }


def prepare(source: Path, output_dir: Path, *, cv2_module: Any, frame_indices: Tuple[int, int] = (0, 2)) -> Mapping[str, Any]:
    """Decode two ordered source frames and write the exact source-wrapper resize output."""

    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite existing prepared Octo fixture directory: %s" % output_dir)
    output_dir.mkdir(mode=0o700, parents=False)
    if len(frame_indices) != 2 or frame_indices[0] < 0 or frame_indices[1] <= frame_indices[0]:
        raise FixturePreparationError("frame_indices must be two increasing nonnegative source frame indices.")
    capture = cv2_module.VideoCapture(str(source))
    try:
        frames: Dict[int, Any] = {}
        for decoded_index in range(frame_indices[-1] + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise FixturePreparationError("Could not decode source frame %d with OpenCV." % decoded_index)
            if decoded_index not in frame_indices:
                continue
            if len(getattr(frame, "shape", ())) != 3 or frame.shape[2] != 3 or str(getattr(frame, "dtype", "")) != "uint8":
                raise FixturePreparationError("Source frame %d must be an OpenCV uint8 three-channel image." % decoded_index)
            frames[decoded_index] = frame
    finally:
        release = getattr(capture, "release", None)
        if callable(release):
            release()
    records = []
    for output_index, source_index in enumerate(frame_indices):
        frame_bgr = frames[source_index]
        # OpenCV's VideoCapture exposes BGR pixels. PLUMB's canonical policy
        # observation contract is RGB, so make that decoder conversion explicit
        # before applying AutoEval's source resize operation.
        frame_rgb = cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2RGB)
        # Exact call shape from ResizeObsImageWrapper: cv2.resize(obs[key], (256, 256)).
        resized_rgb = cv2_module.resize(frame_rgb, OUTPUT_SIZE)
        if tuple(getattr(resized_rgb, "shape", ())) != (256, 256, 3) or str(getattr(resized_rgb, "dtype", "")) != "uint8":
            raise FixturePreparationError("OpenCV resize did not produce canonical uint8 (256, 256, 3).")
        # cv2.imencode expects BGR. Convert back solely for codec I/O; the
        # bound policy pixels remain resized_rgb and smoke reloads them as RGB.
        encoded_ok, encoded = cv2_module.imencode(".png", cv2_module.cvtColor(resized_rgb, cv2_module.COLOR_RGB2BGR))
        if not encoded_ok:
            raise FixturePreparationError("OpenCV could not losslessly encode canonical frame %d as PNG." % source_index)
        output = output_dir / ("frame-%02d.png" % output_index)
        _write_new_bytes(output, bytes(encoded))
        records.append(_frame_record(source_index, frame_bgr, frame_rgb, resized_rgb, output))
    return {
        "schema": "plumb-octo-canonical-inputs-v1",
        "qualified": False,
        "source": {
            "path": str(source.resolve()),
            "sha256": _sha256_file(source),
            "frame_indices": list(frame_indices),
            "decoder": "OpenCV VideoCapture",
            "decoder_channel_conversion": "OpenCV VideoCapture BGR was converted to canonical RGB before resize; both BGR and RGB raw pixel hashes are retained per frame.",
        },
        "preprocessing": {
            "kind": "AutoEval ResizeObsImageWrapper-compatible cv2.resize",
            "target_size": [256, 256],
            "interpolation": "OpenCV default INTER_LINEAR",
            "color_conversion": "BGR→RGB before source-compatible resize; RGB→BGR only while encoding PNG for OpenCV codec semantics.",
            "autoeval_source": "https://github.com/zhouzypaul/auto_eval/blob/%s/run_eval.py#L300-L324" % AUTOEVAL_RESIZE_SOURCE_COMMIT,
            "resize_wrapper_source": "https://github.com/rail-berkeley/manipulator_gym/blob/%s/manipulator_gym/utils/gym_wrappers.py#L274-L312" % MANIPULATOR_GYM_RESIZE_SOURCE_COMMIT,
            "limitation": "AutoEval did not pin manipulator_gym at its source commit; this is a source-backed diagnostic input profile, not camera-pipeline pixel-equivalence evidence.",
        },
        "frames": records,
    }


def verify(output_dir: Path, *, cv2_module: Any) -> Mapping[str, Any]:
    manifest = output_dir / "manifest.json"
    if not manifest.is_file():
        raise FixturePreparationError("Missing prepared Octo input manifest: %s" % manifest)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if raw.get("schema") != "plumb-octo-canonical-inputs-v1" or raw.get("qualified") is not False:
        raise FixturePreparationError("Prepared Octo input manifest has an invalid qualification/schema claim.")
    frames = raw.get("frames")
    if not isinstance(frames, list) or len(frames) != 2:
        raise FixturePreparationError("Prepared Octo input manifest must bind exactly two frames.")
    previous_index = -1
    for record in frames:
        if not isinstance(record, Mapping) or not isinstance(record.get("source_frame_index"), int) or record["source_frame_index"] <= previous_index:
            raise FixturePreparationError("Prepared Octo input manifest frame ordering is invalid.")
        previous_index = record["source_frame_index"]
        path = output_dir / str(record.get("output_file", ""))
        image_bgr = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
        if image_bgr is None or tuple(image_bgr.shape) != (256, 256, 3) or str(image_bgr.dtype) != "uint8":
            raise FixturePreparationError("Prepared canonical PNG is missing or not uint8 256x256x3: %s" % path)
        image_rgb = cv2_module.cvtColor(image_bgr, cv2_module.COLOR_BGR2RGB)
        if record.get("output_file_sha256") != _sha256_file(path) or record.get("output_rgb_pixel_sha256") != _sha256_bytes(image_rgb.tobytes()):
            raise FixturePreparationError("Prepared canonical PNG hash mismatch: %s" % path)
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/scratch/lchow432/plumb"))
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frame-indices", type=int, nargs=2, default=(0, 2), metavar=("FIRST", "SECOND"))
    parser.add_argument("--execute", action="store_true", help="Decode/resize/write new PNGs and sidecar; otherwise print the plan only.")
    parser.add_argument("--verify", action="store_true", help="Verify an existing sidecar/PNGs only.")
    args = parser.parse_args(argv)
    if args.execute and args.verify:
        parser.error("--execute and --verify are mutually exclusive")
    root = _cluster_root(args.root)
    source = (args.source_video or root / "fixtures" / "bridge_video.mp4").resolve()
    output = (args.output_dir or root / "fixtures" / "octo-small-bridge-cv2-linear-rgb-256-v3").resolve()
    try:
        output.relative_to(root / "fixtures")
        source.relative_to(root / "fixtures")
    except ValueError as error:
        parser.error("source and output must remain within the cluster fixture directory")
    if not source.is_file():
        parser.error("source video is absent: %s" % source)
    plan = {
        "source_video": str(source),
        "source_sha256": _sha256_file(source),
        "output_dir": str(output),
        "output_frames": ["frame-00.png", "frame-01.png"],
        "source_frame_indices": list(args.frame_indices),
        "manifest": str(output / "manifest.json"),
        "qualified": False,
    }
    try:
        import cv2  # type: ignore

        if args.verify:
            print(json.dumps({"verified": verify(output, cv2_module=cv2)}, sort_keys=True, indent=2))
        elif args.execute:
            payload = prepare(source, output, cv2_module=cv2, frame_indices=tuple(args.frame_indices))
            _write_new_bytes(output / "manifest.json", (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"))
            print(json.dumps({"prepared": payload, "plan": plan}, sort_keys=True, indent=2))
        else:
            print(json.dumps(plan, sort_keys=True, indent=2))
    except Exception as error:
        print("Octo canonical input preparation failed: %s" % error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
