"""CPU-only contracts for the IRASim state-representation replay diagnostic."""

from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext

import numpy as np

import pytest

from cluster import irasim_state_replay as replay
from cluster.irasim_state_replay import (
    REPLAY_TICKS,
    ReplayConfig,
    ReplayInputError,
    ReplayInput,
    _run_once,
    carry_last_latent,
    derive_stream_seed,
    generator_metadata,
    load_replay_inputs,
    run_replay,
    source_fixture_provenance,
    source_fixture_sha256,
    verify_irasim_source_checkout,
)


def _source_report(path, *, action=None, include_source=True):
    rows = []
    for tick in range(REPLAY_TICKS):
        rows.append(
            {
                "tick": tick,
                "raw_native_action_7d": list(action if action is not None else (0.0,) * 7),
                "seed": tick + 100,
            }
        )
    payload = {"kind": "plumb_irasim_openvla_closed_loop_diagnostic", "step_records": rows}
    if include_source:
        payload["source"] = {"fixture_video": {"sha256": "sha256:" + "a" * 64}}
    path.write_text(json.dumps(payload))
    return path


def _config(tmp_path, output_dir):
    return ReplayConfig(
        root=tmp_path,
        source_report=tmp_path / "source.json",
        output_dir=output_dir,
        irasim_repo=tmp_path / "irasim-source",
        irasim_checkpoint=tmp_path / "irasim.safetensors",
        irasim_checkpoint_sha256=None,
        irasim_conversion_report=None,
        vae_path=tmp_path / "vae",
        scheduler_path=tmp_path / "scheduler",
        irasim_config=tmp_path / "frame.yaml",
        fixture_video=tmp_path / "fixture.mp4",
    )


def test_load_replay_inputs_reads_only_ordered_raw_actions_and_seeds(tmp_path):
    report = _source_report(tmp_path / "report.json")

    inputs = load_replay_inputs(report)

    assert len(inputs) == 16
    assert inputs[0].tick == 0 and inputs[-1].tick == 15
    assert inputs[0].action == (0.0,) * 7
    assert inputs[-1].recorded_seed == 115
    assert source_fixture_sha256(report) == "a" * 64


def test_actual_closed_loop_shape_uses_sibling_provenance_sidecar_when_report_has_no_source(tmp_path):
    report = _source_report(tmp_path / "report.json", include_source=False)
    sidecar = tmp_path / "provenance.json"
    sidecar.write_text(
        json.dumps(
            {
                "kind": "plumb_irasim_openvla_closed_loop_diagnostic",
                "source": {"fixture_video": {"sha256": "sha256:" + "b" * 64}},
            }
        )
    )

    fixture, origin, sidecar_path = source_fixture_provenance(report)

    assert origin == "sibling_provenance_sidecar"
    assert sidecar_path == sidecar
    assert fixture["sha256"] == "sha256:" + "b" * 64
    assert source_fixture_sha256(report) == "b" * 64


def test_replay_input_rejects_missing_or_nonfinite_actions_without_policy_fallback(tmp_path):
    bad = _source_report(tmp_path / "bad.json", action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")))
    with pytest.raises(ReplayInputError, match="non-finite"):
        load_replay_inputs(bad)

    malformed = _source_report(tmp_path / "malformed.json")
    body = json.loads(malformed.read_text())
    body["step_records"] = body["step_records"][:-1]
    malformed.write_text(json.dumps(body))
    with pytest.raises(ReplayInputError, match="exactly 16"):
        load_replay_inputs(malformed)


def test_fake_branches_use_independent_vae_and_identical_paired_diffusion_streams():
    metadata = generator_metadata(12345)

    assert metadata["image_branch_diffusion_seed"] == metadata["latent_branch_diffusion_seed"]
    assert metadata["vae_seed"] != metadata["image_branch_diffusion_seed"]
    assert derive_stream_seed(12345, "initial_vae") != metadata["vae_seed"]
    with pytest.raises(ReplayInputError, match="Unsupported"):
        derive_stream_seed(12345, "shared")


class FakeFinalSlice:
    def __init__(self):
        self.cloned = False
        self.shape = (1, 1, 4, 32, 40)

    def clone(self):
        self.cloned = True
        return ("cloned-final-latent", self.shape)


class FakeLatents:
    shape = (1, 2, 4, 32, 40)

    def __init__(self):
        self.slice = FakeFinalSlice()
        self.key = None

    def __getitem__(self, key):
        self.key = key
        return self.slice


def test_latent_branch_carries_only_final_time_slice_and_clones_it():
    latents = FakeLatents()

    carried = carry_last_latent(latents)

    assert latents.key == (slice(None, None, None), slice(-1, None, None))
    assert latents.slice.cloned is True
    assert carried == ("cloned-final-latent", (1, 1, 4, 32, 40))


def test_preflight_rejects_existing_output_without_writing_or_overwriting(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(ReplayInputError, match="already exists"):
        _config(tmp_path, output)


def test_source_checkout_requires_pinned_head_and_clean_targeted_files(tmp_path, monkeypatch):
    repo = tmp_path / "irasim-source"
    repo.mkdir()
    calls = []

    def clean_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "HEAD":
            return subprocess.CompletedProcess(command, 0, stdout=replay.IRASIM_SOURCE_REVISION + "\n")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(replay.subprocess, "run", clean_run)
    verified = verify_irasim_source_checkout(repo)
    assert verified["git_head"] == replay.IRASIM_SOURCE_REVISION
    assert verified["targeted_git_diff_clean"] == {"unstaged": True, "staged": True}
    assert len(calls) == 3

    def dirty_run(command, **kwargs):
        if command[-1] == "HEAD":
            return subprocess.CompletedProcess(command, 0, stdout=replay.IRASIM_SOURCE_REVISION + "\n")
        return subprocess.CompletedProcess(command, 1, stdout="")

    monkeypatch.setattr(replay.subprocess, "run", dirty_run)
    with pytest.raises(ReplayInputError, match="targeted source files"):
        verify_irasim_source_checkout(repo)


def test_fresh_output_persists_unqualified_failure_record_before_heavy_adapter_load(tmp_path, monkeypatch):
    _source_report(tmp_path / "source.json")
    output = tmp_path / "out"
    config = _config(tmp_path, output)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    code, report = run_replay(config)

    assert code == 2
    assert report["qualified"] is False
    assert report["policy_requery"] is False
    assert report["success_scoring"] is False
    assert report["status"] == "failed"
    assert (output / "report.json").is_file()


def test_tensor_hash_contract_runs_only_when_local_torch_is_available():
    torch = pytest.importorskip("torch")
    from cluster.irasim_state_replay import tensor_hash

    tensor = torch.zeros((1, 2, 4, 32, 40), dtype=torch.float32)
    assert tensor_hash(tensor).startswith("sha256:")


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    @property
    def shape(self):
        return self.value.shape

    def clone(self):
        return _FakeTensor(self.value.copy())

    def __getitem__(self, key):
        return _FakeTensor(self.value[key])

    def detach(self):
        return self

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeTorch:
    @staticmethod
    def no_grad():
        return nullcontext()


class _FakeAdapter:
    class profile:
        num_frames = 16

    def _load(self):
        return _FakeTorch, object()


def test_full_fake_run_once_traverses_all_ticks_persists_branch_media_and_comparison(tmp_path, monkeypatch):
    encoded_frames = []

    def fake_encode(adapter, torch, pipeline, frame, seed):
        encoded_frames.append((np.asarray(frame).copy(), seed))
        # Keep tick zero shared; later re-encodes depend on the generated RGB
        # frame to exercise image-branch feedback without a real VAE.
        value = float(np.asarray(frame, dtype=np.float32).mean()) / 255.0
        return _FakeTensor(np.full((1, 1, 4, 32, 40), value, dtype=np.float32)), {"fake_vae_seed": seed}

    def fake_step(torch, adapter, pipeline, mask_x, action, diffusion_seed):
        mask = mask_x.value
        final_value = float(mask.mean()) + float(diffusion_seed % 7) / 20.0
        final = np.full((1, 1, 4, 32, 40), final_value, dtype=np.float32)
        latents = _FakeTensor(np.concatenate((mask, final), axis=1))
        # Decoded IRASim-form CHW frames in [-1,1]; paired tick zero is
        # identical because both branch mask and diffusion seed are identical.
        pixel = np.clip(final_value, -1.0, 1.0)
        video = np.full((1, 2, 3, 2, 2), pixel, dtype=np.float32)
        return video, latents

    def fake_mp4(path, frames):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-mp4-%d" % len(frames))

    def fake_png(path, frame):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.asarray(frame).tobytes())

    monkeypatch.setattr(replay, "_encode_mask", fake_encode)
    monkeypatch.setattr(replay, "_pipeline_step", fake_step)
    monkeypatch.setattr(replay, "_write_png", fake_png)
    monkeypatch.setattr(replay, "_write_mp4", fake_mp4)
    output = tmp_path / "out"
    config = _config(tmp_path, output)
    initial = np.zeros((2, 2, 3), dtype=np.uint8)
    inputs = tuple(ReplayInput(index, (0.0,) * 7, index + 700) for index in range(REPLAY_TICKS))

    outcome = _run_once(config, inputs, _FakeAdapter(), initial, output)

    assert len(outcome["branches"]["image_reencode"]) == REPLAY_TICKS
    assert len(outcome["branches"]["latent_carry"]) == REPLAY_TICKS
    assert len(outcome["paired_output_pixel_hash_equal_by_tick"]) == REPLAY_TICKS
    assert outcome["first_output_equality_assertion"] == {"pixel_equal": True, "final_latent_equal": True}
    assert len(encoded_frames) == REPLAY_TICKS  # shared initial encode + 15 image re-encodes
    for branch in ("image_reencode", "latent_carry"):
        assert outcome["videos"][branch]["frame_count"] == 17
        assert (output / branch / (branch + ".mp4")).is_file()
        records = outcome["branches"][branch]
        assert all("descriptive_output_pixel_mae" in record for record in records)
        assert all(record["output"]["final_latent_tensor_sha256"].startswith("sha256:") for record in records)
        assert len(list((output / branch / "frames").glob("*.png"))) == 17
