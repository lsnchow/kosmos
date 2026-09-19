"""CPU fake-path contracts for the released-horizon IRASim reference control."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from cluster import irasim_native_reference as native
from cluster import irasim_state_replay as replay


def _source_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "plumb_irasim_openvla_closed_loop_diagnostic",
                "step_records": [
                    {"tick": index, "raw_native_action_7d": [0.1] * 7, "seed": index + 900}
                    for index in range(16)
                ],
                "source": {"fixture_video": {"sha256": "sha256:" + "a" * 64}},
            }
        )
    )
    return path


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    @property
    def shape(self):
        return self.value.shape

    def view(self, *shape):
        return _Tensor(self.value.reshape(shape))

    def __getitem__(self, key):
        return _Tensor(self.value[key])

    def clone(self):
        return _Tensor(self.value.copy())

    def detach(self):
        return self

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Bool:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class _Generator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class _Torch:
    float32 = "float32"
    Generator = _Generator

    @staticmethod
    def no_grad():
        return nullcontext()

    @staticmethod
    def tensor(value, **kwargs):
        return _Tensor(value)

    @staticmethod
    def device(value):
        return value

    @staticmethod
    def isfinite(tensor):
        return _Bool(bool(np.isfinite(tensor.value).all()))

    @staticmethod
    def equal(left, right):
        return bool(np.array_equal(left.value, right.value))


class _Pipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, actions, **kwargs):
        self.calls.append((actions, kwargs))
        mask = kwargs["mask_x"].value
        final = np.full((1, 15, 4, 32, 40), 0.25, dtype=np.float32)
        latents = _Tensor(np.concatenate((mask, final), axis=1))
        videos = np.full((1, 16, 3, 2, 2), -0.25, dtype=np.float32)
        return videos, latents


class _Adapter:
    class profile:
        num_frames = 16
        extras = 3
        mask_frame_num = 1
        device = "cpu"
        input_height = 256
        input_width = 320
        inference_steps = 50
        guidance_scale = 1.0

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _load(self):
        return _Torch, self.pipeline


def _config(tmp_path: Path, output: Path) -> native.NativeReferenceConfig:
    replay_config = replay.ReplayConfig(
        root=tmp_path,
        source_report=tmp_path / "source.json",
        output_dir=output,
        irasim_repo=tmp_path / "irasim-source",
        irasim_checkpoint=tmp_path / "checkpoint.safetensors",
        irasim_checkpoint_sha256=None,
        irasim_conversion_report=None,
        vae_path=tmp_path / "vae",
        scheduler_path=tmp_path / "scheduler",
        irasim_config=tmp_path / "frame.yaml",
        fixture_video=tmp_path / "fixture.mp4",
    )
    return native.NativeReferenceConfig(replay_config)


def test_full_fake_native_wrapper_calls_one_released_horizon_and_persists_16_frames(tmp_path, monkeypatch):
    _source_report(tmp_path / "source.json")
    output = tmp_path / "out"
    config = _config(tmp_path, output)
    pipeline = _Pipeline()

    def fake_encode(adapter, torch, pipeline, frame, seed):
        return _Tensor(np.zeros((1, 1, 4, 32, 40), dtype=np.float32)), {"initial_seed": seed}

    def fake_png(path, frame):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.asarray(frame).tobytes())

    def fake_mp4(path, frames):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"native-reference-%d" % len(frames))

    monkeypatch.setattr(replay, "_encode_mask", fake_encode)
    monkeypatch.setattr(native, "_write_png", fake_png)
    monkeypatch.setattr(native, "_write_mp4", fake_mp4)
    monkeypatch.setattr(native, "_video_export_frames", lambda frames: (tuple(frames), {"fake": True}))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    code, report = native.run_native_reference(
        config,
        adapter_loader=lambda unused: _Adapter(pipeline),
        frame_loader=lambda unused: np.zeros((2, 2, 3), dtype=np.uint8),
        provenance_loader=lambda unused: {"fixture": "bound"},
    )

    assert code == 0
    assert report["status"] == "completed_unqualified_reference"
    assert report["qualified"] is False
    assert report["policy_requery"] is False
    assert report["native_openvla_control_claim"] is False
    assert report["success_scoring"] is False
    assert len(report["shared_source_inputs"]["first_fifteen_raw_native_action_7d"]) == 15
    assert report["artifacts"]["mp4_frame_count"] == 16
    assert len(report["artifacts"]["frames"]) == 16
    assert (output / "irasim-native-reference.mp4").is_file()
    assert len(list((output / "frames").glob("*.png"))) == 16
    actions, kwargs = pipeline.calls[0]
    assert actions.shape == (1, 15, 7)
    assert kwargs["video_length"] == 16 and kwargs["output_type"] == "both"
    assert kwargs["mask_x"].shape == (1, 1, 4, 32, 40)


def test_static_guard_prevents_policy_requery_names_from_native_reference_source():
    text = Path(native.__file__).read_text(encoding="utf-8")
    assert "predict_action(" not in text
    assert "plumb.policies" not in text
    assert "OpenVLAPolicy" not in text
