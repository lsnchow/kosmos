import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cluster.prepare_susie_fixture import prepare


def test_fixture_binds_first_and_last_decoded_frames_without_certifying_goal(tmp_path, monkeypatch):
    frames = iter([np.full((8, 8, 3), value, dtype=np.uint8) for value in (1, 2, 3)])

    class Capture:
        def read(self):
            frame = next(frames, None)
            return frame is not None, frame

        def release(self):
            pass

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        __version__="fake-test-only", COLOR_BGR2RGB=1,
        VideoCapture=lambda path: Capture(), cvtColor=lambda frame, code: frame[:, :, ::-1],
        resize=lambda frame, size: np.full((size[1], size[0], 3), frame[0, 0, 0], dtype=np.uint8),
    ))
    video = tmp_path / "source.mp4"
    video.write_bytes(b"synthetic unit test video stand-in")
    output = tmp_path / "derived"
    report = prepare(video, output)
    assert report["qualified"] is False
    assert report["source"]["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert report["source"]["frame_indices"] == [0, 2]
    assert "not a certified task goal" in report["preprocessing"]["limitation"]
    assert report == json.loads((output / "manifest.json").read_text())
    for record in report["frames"]:
        assert record["output_file_sha256"] == hashlib.sha256((output / record["output_file"]).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="fresh"):
        prepare(video, output)
