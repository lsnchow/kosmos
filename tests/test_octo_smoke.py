"""Safety/provenance tests for the allocated-GPU Octo smoke entrypoint."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from PIL import Image

from cluster import octo_smoke


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pixel_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def test_smoke_png_loader_requires_manifest_bound_rgb_pixels_and_keeps_physical_timestamp_unknown(tmp_path):
    output = tmp_path / "inputs"
    output.mkdir()
    first = np.broadcast_to(np.asarray([1, 2, 3], dtype=np.uint8), (256, 256, 3)).copy()
    second = np.broadcast_to(np.asarray([7, 8, 9], dtype=np.uint8), (256, 256, 3)).copy()
    paths = (output / "frame-00.png", output / "frame-01.png")
    Image.fromarray(first, "RGB").save(paths[0])
    Image.fromarray(second, "RGB").save(paths[1])
    manifest = {
        "schema": "plumb-octo-canonical-inputs-v1",
        "qualified": False,
        "preprocessing": {"kind": "fixture"},
        "frames": [
            {"source_frame_index": 0, "output_file": paths[0].name, "output_file_sha256": _sha(paths[0]), "output_rgb_pixel_sha256": _pixel_sha(first)},
            {"source_frame_index": 2, "output_file": paths[1].name, "output_file_sha256": _sha(paths[1]), "output_rgb_pixel_sha256": _pixel_sha(second)},
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded_first, loaded_second, bound = octo_smoke._frames(*paths, manifest_path)

    np.testing.assert_array_equal(loaded_first, first)
    np.testing.assert_array_equal(loaded_second, second)
    assert [record["source_frame_index"] for record in bound["frames"]] == [0, 2]


def test_report_reservation_refuses_existing_evidence_before_any_gpu_work(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"previous":"evidence"}\n', encoding="utf-8")

    code = octo_smoke.main(
        [
            "--root",
            str(tmp_path),
            "--frame-0",
            "missing-0.png",
            "--frame-1",
            "missing-1.png",
            "--input-manifest",
            "missing.json",
            "--task-id",
            "close_drawer",
            "--report",
            str(report),
        ]
    )

    assert code == 2
    assert json.loads(report.read_text(encoding="utf-8")) == {"previous": "evidence"}


def test_gpu_metadata_does_not_call_bytes_in_use_a_peak():
    class Device:
        platform = "gpu"

        def memory_stats(self):
            return {"bytes_in_use": 123}

        def __str__(self):
            return "gpu:0"

    class Jax:
        @staticmethod
        def devices():
            return (Device(),)

    metadata = octo_smoke._gpu_metadata(Jax)

    assert metadata["gpu_peak_memory_bytes"] is None
    assert metadata["jax_memory_stats"]["bytes_in_use"] == 123
