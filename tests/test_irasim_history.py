import json

import numpy as np
import pytest

from cluster import irasim_history_replay as history
from cluster import irasim_state_replay as replay
from test_irasim_native_reference import _Adapter, _Tensor, _Torch, _config, _source_report


class HistoryPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, actions, **kwargs):
        self.calls.append((actions, kwargs))
        mask = kwargs["mask_x"].value
        nxt = mask[:, -1:] + 0.001
        latents = _Tensor(np.concatenate((mask, nxt), axis=1))
        video = np.zeros((1, kwargs["video_length"], 3, 2, 2), dtype=np.float32)
        video[:, -1] = nxt.mean()
        return video, latents


def test_full_history_run_repeats_and_rolls_window_without_future_actions(tmp_path, monkeypatch):
    _source_report(tmp_path / "source.json")
    config = _config(tmp_path, tmp_path / "out").replay
    object.__setattr__(config, "same_seed_repeat", True)
    pipe = HistoryPipeline()
    monkeypatch.setattr(replay, "_encode_mask", lambda *args: (_Tensor(np.zeros((1, 1, 4, 32, 40))), {}))
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.asarray(value).tobytes())
    monkeypatch.setattr(replay, "_write_png", write)
    monkeypatch.setattr(replay, "_write_mp4", write)
    monkeypatch.setattr(replay, "_video_export_frames", lambda frames: (frames, {}))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    code, report = history.run_history(config, adapter_loader=lambda _: _Adapter(pipe),
        provenance_loader=lambda _: {}, binding_loader=lambda _: {},
        frame_loader=lambda _: np.zeros((2, 2, 3), dtype=np.uint8))
    assert code == 0, report
    assert len(pipe.calls) == 32
    assert report["repeat_pixel_equal_by_tick"] == [True] * 16
    assert report["repeat_latent_equal_by_tick"] == [True] * 16
    assert [call[1]["video_length"] for call in pipe.calls[:16]] == list(range(2, 17)) + [16]
    rows = report["outcome"]["rows"]
    assert rows[-1]["action_ticks"] == list(range(1, 16))
    assert rows[-1]["known_frame_start"] == 1
    assert all(max(row["action_ticks"]) == row["tick"] for row in rows)
    assert all(call[0].shape[1] + 1 == call[1]["video_length"] for call in pipe.calls)
    assert report["qualified"] is False and report["policy_requery"] is False
    saved = json.loads((config.output_dir / "report.json").read_text())
    assert saved["outcome"]["frame_count"] == 17
    with pytest.raises(ValueError):
        history.run_history(config)
    assert json.loads((config.output_dir / "report.json").read_text()) == saved


def test_history_rejects_mutation_of_committed_latents():
    class Broken(HistoryPipeline):
        def __call__(self, *args, **kwargs):
            video, latent = super().__call__(*args, **kwargs)
            latent.value[:, 0] += 1
            return video, latent
    pipe = Broken()
    with pytest.raises(ValueError, match="altered committed"):
        history.history_step(_Torch, _Adapter(pipe), pipe, _Tensor(np.zeros((1, 1, 4, 32, 40))),
                             [replay.ReplayInput(0, (0.0,) * 7, 1)], 0)


def test_history_requires_exact_past_window():
    pipe = HistoryPipeline()
    inputs = [replay.ReplayInput(i, (0.0,) * 7, i) for i in range(16)]
    with pytest.raises(ValueError, match="causal window"):
        history.history_step(_Torch, _Adapter(pipe), pipe, _Tensor(np.zeros((1, 1, 4, 32, 40))), inputs, 15)
    assert pipe.calls == []
