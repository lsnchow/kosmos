"""Canonical Octo diagnostic fixture preparation tests without OpenCV/video codecs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cluster import prepare_octo_smoke_inputs as inputs


class _Capture:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


class _CV2:
    IMREAD_COLOR = 1
    COLOR_BGR2RGB = 2
    COLOR_RGB2BGR = 3

    def __init__(self, frames):
        self.frames = frames
        self.captures = []

    def VideoCapture(self, path):
        capture = _Capture(self.frames)
        self.captures.append(capture)
        return capture

    @staticmethod
    def resize(frame, size):
        assert size == (256, 256)
        return np.broadcast_to(frame[0, 0], (256, 256, 3)).copy()

    @staticmethod
    def cvtColor(frame, code):
        assert code in (_CV2.COLOR_BGR2RGB, _CV2.COLOR_RGB2BGR)
        return frame[..., ::-1].copy()

    @staticmethod
    def imencode(suffix, frame):
        assert suffix == ".png"
        return True, np.frombuffer(frame.tobytes(), dtype=np.uint8)

    @staticmethod
    def imread(path, flag):
        assert flag == _CV2.IMREAD_COLOR
        return np.frombuffer(open(path, "rb").read(), dtype=np.uint8).reshape(256, 256, 3)


def test_prepared_two_frame_manifest_binds_source_pixels_resize_and_png_bytes(tmp_path):
    source = tmp_path / "bridge.mp4"
    source.write_bytes(b"immutable-video")
    output = tmp_path / "canonical"
    cv2 = _CV2(
        [
            np.broadcast_to(np.asarray([7, 8, 9], dtype=np.uint8), (480, 640, 3)).copy(),
            np.broadcast_to(np.asarray([10, 11, 12], dtype=np.uint8), (480, 640, 3)).copy(),
            np.broadcast_to(np.asarray([13, 14, 15], dtype=np.uint8), (480, 640, 3)).copy(),
        ]
    )

    payload = inputs.prepare(source, output, cv2_module=cv2)
    inputs._write_new_bytes(output / "manifest.json", (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode())

    assert payload["qualified"] is False
    assert payload["source"]["frame_indices"] == [0, 2]
    assert payload["preprocessing"]["interpolation"] == "OpenCV default INTER_LINEAR"
    assert [item["source_bgr_shape"] for item in payload["frames"]] == [[480, 640, 3], [480, 640, 3]]
    assert [item["output_rgb_shape"] for item in payload["frames"]] == [[256, 256, 3], [256, 256, 3]]
    assert payload["frames"][0]["source_bgr_pixel_sha256"] != payload["frames"][0]["decoded_rgb_pixel_sha256"]
    assert cv2.captures[0].released is True
    verified = inputs.verify(output, cv2_module=cv2)
    assert verified["frames"][0]["output_rgb_pixel_sha256"] != verified["frames"][1]["output_rgb_pixel_sha256"]


def test_prepared_fixture_directory_is_no_clobber(tmp_path):
    source = tmp_path / "bridge.mp4"
    source.write_bytes(b"immutable-video")
    output = tmp_path / "canonical"
    output.mkdir()
    cv2 = _CV2([np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)])

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        inputs.prepare(source, output, cv2_module=cv2)
