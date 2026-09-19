"""Stable local hashes used to bind judge evidence and implementation inputs.

These hashes establish byte/pixel identity inside a PLUMB report.  They do not
attest to the origin, privacy, or scientific validity of a clip, a model, or a
runtime.  Those facts remain explicit provenance/gate requirements.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(encoded)


def file_source_sha256(path: object) -> Optional[str]:
    """Hash local source bytes only; this is not a signed release attestation."""

    try:
        candidate = Path(str(path))
        if not candidate.is_file():
            return None
        return sha256_bytes(candidate.read_bytes())
    except OSError:
        return None


def image_pixel_hash(image: Any) -> Mapping[str, str]:
    """Hash the pixels/bytes presented to the processor with explicit semantics.

    Arrays/tensors bind dtype, shape, and contiguous bytes.  Pillow images are
    converted to RGB first because that is the representation supplied by the
    OpenVLA/Qwen preprocessing boundary.  Raw bytes are accepted for small
    fixture tests but marked as shape-unknown, so consumers can distinguish
    them from image arrays.
    """

    if isinstance(image, bytes):
        return {"sha256": sha256_bytes(image), "semantics": "raw_pixel_bytes_shape_unknown"}
    if isinstance(image, bytearray):
        return {"sha256": sha256_bytes(bytes(image)), "semantics": "raw_pixel_bytes_shape_unknown"}
    if isinstance(image, memoryview):
        return {"sha256": sha256_bytes(image.tobytes()), "semantics": "raw_pixel_bytes_shape_unknown"}

    candidate = image
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        contiguous = getattr(candidate, "contiguous", None)
        if callable(contiguous):
            candidate = contiguous()
        numpy = getattr(candidate, "numpy", None)
        if callable(numpy):
            candidate = numpy()

    mode = getattr(candidate, "mode", None)
    size = getattr(candidate, "size", None)
    convert = getattr(candidate, "convert", None)
    if isinstance(mode, str) and isinstance(size, tuple) and callable(convert):
        rgb = convert("RGB")
        raw = rgb.tobytes()
        return {
            "sha256": canonical_json_sha256(
                {"kind": "pillow_rgb", "size": list(rgb.size), "bytes_sha256": sha256_bytes(raw)}
            ),
            "semantics": "pillow_rgb_pixels",
        }

    tobytes = getattr(candidate, "tobytes", None)
    shape = getattr(candidate, "shape", None)
    dtype = getattr(candidate, "dtype", None)
    if callable(tobytes) and shape is not None:
        raw = tobytes()
        shape_values = [int(item) for item in tuple(shape)]
        return {
            "sha256": canonical_json_sha256(
                {
                    "kind": "array",
                    "shape": shape_values,
                    "dtype": str(dtype) if dtype is not None else "unknown",
                    "bytes_sha256": sha256_bytes(bytes(raw)),
                }
            ),
            "semantics": "array_pixels",
        }
    raise TypeError("Cannot produce a stable pixel hash for this frame/reference object.")
