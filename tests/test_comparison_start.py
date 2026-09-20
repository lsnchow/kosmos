"""Source-frame hashing uses the same semantics as the frozen pilot."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plumb.policies.provenance import image_pixel_hash

SPEC = importlib.util.spec_from_file_location("comparison_start", Path(__file__).parents[1] / "scripts/prepare_comparison_start.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_rgb_source_binds_frozen_shape_dtype_and_pixels(tmp_path):
    pixels = np.full((256, 256, 3), 80, dtype=np.uint8)
    frame = tmp_path / "start.png"
    Image.fromarray(pixels).save(frame)
    reference = {"sha256": hashlib.sha256(frame.read_bytes()).hexdigest(), "pixel_sha256": image_pixel_hash(pixels)["sha256"]}
    raw, file_sha, pixel_sha = module._verify_rgb_frame(frame, reference)
    assert raw == frame.read_bytes()
    assert pixel_sha == reference["pixel_sha256"]
    assert file_sha == reference["sha256"]
    reference["pixel_sha256"] = "sha256:" + hashlib.sha256(pixels.tobytes()).hexdigest()
    with pytest.raises(module.StartPreparationError, match="pixel hash"):
        module._verify_rgb_frame(frame, reference)


def test_rgb_source_refuses_implicit_color_conversion(tmp_path):
    frame = tmp_path / "gray.png"
    Image.new("L", (256, 256)).save(frame)
    with pytest.raises(module.StartPreparationError, match="already be RGB"):
        module._verify_rgb_frame(frame, {"sha256": hashlib.sha256(frame.read_bytes()).hexdigest()})
