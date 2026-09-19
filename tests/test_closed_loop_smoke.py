"""Offline contracts for the cluster-only real-adapter closed-loop smoke."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

import cluster.closed_loop_smoke as closed_loop_smoke
from cluster.closed_loop_smoke import (
    ClosedLoopConfig,
    ClosedLoopInputError,
    WORLD_STATE_MODE,
    _irasim_frame_to_rgb,
    _irasim_float_to_uint8,
    _release_binding,
    _video_export_frames,
    run_cli,
    run_closed_loop,
)


class FakePolicy:
    def __init__(self):
        self.images = []

    def action_for(self, image, task, timestamp):
        self.images.append((image, task, timestamp))
        return (0.01, 0, 0, 0, 0, 0, 1), {"backend_calls": len(self.images)}


class FakeWorld:
    def __init__(self, fail_call=None):
        self.images = []
        self.actions = []
        self.fail_call = fail_call

    def generate_one_step(self, image, action, seed):
        self.images.append(image)
        self.actions.append((tuple(action), seed))
        if self.fail_call == len(self.images):
            raise RuntimeError("world unavailable")
        future = "generated-%d" % len(self.images)
        return SimpleNamespace(
            frames=("echo:%s" % image, future),
            native_action_scaled=tuple(action),
            timing={"backend_calls": 1, "wall_seconds": 0.01},
            condition_preprocessing={"seed": seed, "resize_applied": False},
        )


def config(tmp_path, ticks=3):
    return ClosedLoopConfig(
        root=tmp_path,
        model_path=tmp_path / "models" / "openvla",
        openvla_manifest=tmp_path / "evidence" / "openvla.json",
        irasim_repo=tmp_path / "irasim-source",
        irasim_checkpoint=tmp_path / "models" / "irasim.safetensors",
        irasim_checkpoint_sha256=None,
        irasim_conversion_report=None,
        vae_path=tmp_path / "models" / "vae",
        scheduler_path=tmp_path / "irasim-source" / "scheduler",
        irasim_config=tmp_path / "irasim-source" / "frame.yaml",
        fixture_video=tmp_path / "fixtures" / "bridge.mp4",
        ticks=ticks,
        output_dir=tmp_path / "out",
    )


def fake_png(path: Path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(frame))


def fake_mp4(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("|".join(frames)).encode("utf-8"))


def fake_video_exporter(frames):
    return tuple(frames), {"target_hw": None, "video_only_resizes": [], "raw_policy_world_frames_unchanged": True}


def test_fresh_generated_frame_drives_each_next_native_tick_and_condition_is_dropped(tmp_path):
    policy = FakePolicy()
    world = FakeWorld()
    report = run_closed_loop(
        config(tmp_path, ticks=3), policy, world, "vendor-frame-0",
        png_writer=fake_png, mp4_writer=fake_mp4, video_exporter=fake_video_exporter,
    )

    assert report["status"] == "completed_unqualified_diagnostic"
    assert report["ticks_completed"] == 3
    assert report["frames_persisted"] == 4
    assert report["binary_success"] is None and report["progress_score"] is None
    assert report["validity"] == "unknown"
    assert report["world_state_mode"] == WORLD_STATE_MODE
    assert [item[0] for item in policy.images] == ["vendor-frame-0", "generated-1", "generated-2"]
    assert world.images == ["vendor-frame-0", "generated-1", "generated-2"]
    assert all(len(action) == 7 for action, _ in world.actions)
    assert all(row["world_post_frame0_dropped"] is True for row in report["step_records"])
    assert (tmp_path / "out" / "provenance.json").is_file()
    assert (tmp_path / "out" / "state" / "step-002.json").is_file()
    assert (tmp_path / "out" / "closed-loop.mp4").is_file()
    provenance = json.loads((tmp_path / "out" / "provenance.json").read_text())
    first_step = json.loads((tmp_path / "out" / "state" / "step-000.json").read_text())
    first_png = tmp_path / "out" / "frames" / "frame-001.png"
    assert provenance["task_instruction"] == config(tmp_path).task
    assert provenance["initial_frame_artifact"]["sha256_semantics"] == "decoded_rgb_pixel_sha256"
    assert first_step["generated_frame_artifact"]["file_sha256"] == "sha256:" + hashlib.sha256(first_png.read_bytes()).hexdigest()
    assert first_step["generated_frame_artifact"]["decoded_pixel_sha256"] == first_step["generated_frame_sha256"]


def test_default_bounded_sixteen_ticks_writes_seventeen_frame_artifact_sequence(tmp_path):
    policy = FakePolicy()
    report = run_closed_loop(
        config(tmp_path, ticks=16), policy, FakeWorld(), "vendor-frame-0",
        png_writer=fake_png, mp4_writer=fake_mp4, video_exporter=fake_video_exporter,
    )
    assert report["ticks_requested"] == report["ticks_completed"] == 16
    assert report["frames_persisted"] == 17
    assert len(report["step_records"]) == 16
    assert len(policy.images) == 16


def test_early_world_failure_keeps_prior_step_and_emits_nullable_failure_record(tmp_path):
    report = run_closed_loop(
        config(tmp_path, ticks=3), FakePolicy(), FakeWorld(fail_call=2), "vendor-frame-0",
        png_writer=fake_png, mp4_writer=fake_mp4, video_exporter=fake_video_exporter,
    )
    assert report["status"] == "failed"
    assert report["ticks_completed"] == 1
    assert report["frames_persisted"] == 2
    assert report["binary_success"] is None
    assert report["step_records"][-1]["status"] == "failed"
    first = json.loads((tmp_path / "out" / "state" / "step-000.json").read_text())
    failed = json.loads((tmp_path / "out" / "state" / "step-001.json").read_text())
    assert first["status"] == "completed" and failed["status"] == "failed"


def test_config_keeps_loop_bounded_to_sixteen_real_ticks(tmp_path):
    with pytest.raises(ClosedLoopInputError):
        config(tmp_path, ticks=17)


def test_irasim_float_decode_always_uses_minus_one_to_one_mapping_with_overshoot():
    np = pytest.importorskip("numpy")
    converted = _irasim_float_to_uint8(np.array([-0.5, 0.0, 0.5, 1.2], dtype=np.float32))
    assert converted.tolist() == [63, 127, 191, 255]
    rgb = _irasim_frame_to_rgb(np.array([[[-0.5, 0.0, 0.5], [1.2, -0.5, 0.0]]], dtype=np.float32))
    assert rgb.tolist() == [[[63, 127, 191], [255, 63, 127]]]


def test_video_export_resizes_only_copy_of_mixed_size_vendor_condition(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    vendor = np.full((480, 640, 3), 17, dtype=np.uint8)
    generated = np.full((256, 320, 3), 44, dtype=np.uint8)
    exported, metadata = _video_export_frames((vendor, generated, generated.copy()))
    assert vendor.shape == (480, 640, 3)  # policy/world source was never changed
    assert [frame.shape for frame in exported] == [(256, 320, 3)] * 3
    assert exported[0] is not vendor
    assert metadata["target_hw"] == [256, 320]
    assert metadata["video_only_resizes"] == [{"frame_index": 0, "from_hw": [480, 640], "to_hw": [256, 320], "scope": "video_export_only"}]
    assert metadata["raw_policy_world_frames_unchanged"] is True


def test_frozen_release_binding_hashes_the_exact_release_json_bytes(tmp_path):
    manifest = {"files": [{"path": "cluster/closed_loop_smoke.py", "bytes": 1, "sha256": "a" * 64}]}
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    release = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "RELEASE.json"
    path.write_bytes(raw)
    bound_config = config(tmp_path)
    object.__setattr__(bound_config, "source_release", release)
    object.__setattr__(bound_config, "release_manifest", path)
    binding = _release_binding(bound_config)
    assert binding["status"] == "verified"
    assert binding["source_release"] == release
    assert binding["release_manifest"]["content"] == manifest
    assert binding["release_manifest"]["content_sha256"] == "sha256:" + release


def test_direct_loop_refuses_existing_artifact_bundle_before_policy_or_world_call(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "prior-evidence.json"
    sentinel.write_bytes(b"do not replace")
    policy = FakePolicy()
    world = FakeWorld()
    with pytest.raises(ClosedLoopInputError, match="existing closed-loop artifact directory"):
        run_closed_loop(
            config(tmp_path), policy, world, "vendor-frame-0",
            png_writer=fake_png, mp4_writer=fake_mp4, video_exporter=fake_video_exporter,
        )
    assert sentinel.read_bytes() == b"do not replace"
    assert policy.images == []
    assert world.images == []


def test_cli_refuses_existing_custom_report_before_setup_or_model_work(tmp_path, monkeypatch):
    old_report = tmp_path / "old-report.json"
    old_report.write_bytes(b"previous immutable evidence")
    cfg = config(tmp_path)
    object.__setattr__(cfg, "report_path", old_report)
    setup_called = False

    def forbidden_setup(_):
        nonlocal setup_called
        setup_called = True
        raise AssertionError("must not reach model/setup work")

    monkeypatch.setattr(closed_loop_smoke, "_verify_local_inputs", forbidden_setup)
    code, report = run_cli(cfg)
    assert code == 2
    assert report["status"] == "failed"
    assert setup_called is False
    assert old_report.read_bytes() == b"previous immutable evidence"
    assert not (tmp_path / "out").exists()


def test_atomic_artifact_claim_rejects_second_direct_runner_without_overwrite(tmp_path):
    started = threading.Event()
    release = threading.Event()
    first_policy = FakePolicy()

    def blocking_png(path, frame):
        path.parent.mkdir(parents=True, exist_ok=True)
        if frame == "vendor-frame-0":
            started.set()
            release.wait(timeout=2)
        path.write_text(str(frame))

    holder = {}

    def first_runner():
        try:
            holder["report"] = run_closed_loop(
                config(tmp_path, ticks=1), first_policy, FakeWorld(), "vendor-frame-0",
                png_writer=blocking_png, mp4_writer=fake_mp4, video_exporter=fake_video_exporter,
            )
        except Exception as exc:  # pragma: no cover - assertion below exposes it
            holder["error"] = exc

    thread = threading.Thread(target=first_runner)
    thread.start()
    assert started.wait(timeout=2)
    second_policy = FakePolicy()
    with pytest.raises(ClosedLoopInputError, match="existing closed-loop artifact directory"):
        run_closed_loop(
            config(tmp_path, ticks=1), second_policy, FakeWorld(), "vendor-frame-0",
            png_writer=fake_png, mp4_writer=fake_mp4, video_exporter=fake_video_exporter,
        )
    release.set()
    thread.join(timeout=3)
    assert "error" not in holder
    assert holder["report"]["status"] == "completed_unqualified_diagnostic"
    assert second_policy.images == []
    assert (tmp_path / "out" / "state" / "step-000.json").is_file()


def test_fresh_cli_claims_report_exclusively_then_writes_its_own_result(tmp_path, monkeypatch):
    cfg = config(tmp_path, ticks=1)
    calls = {}

    monkeypatch.setattr(closed_loop_smoke, "_release_binding", lambda _: {"status": "unavailable", "source_release": None})
    monkeypatch.setattr(closed_loop_smoke, "_verify_local_inputs", lambda _: {})
    monkeypatch.setattr(closed_loop_smoke, "_load_vendor_frame_zero", lambda _: "initial")
    monkeypatch.setattr(closed_loop_smoke, "_load_real_adapters", lambda _: (object(), object()))

    def owned_runner(config_value, policy, world, initial, reservation, **kwargs):
        calls["reservation"] = reservation
        assert reservation.report_claimed is True
        assert reservation.artifact_dir == cfg.output_dir
        return {
            "schema_version": 2,
            "status": "completed_unqualified_diagnostic",
            "qualified": False,
            "binary_success": None,
            "progress_score": None,
            "validity": "unknown",
            "missing_reason": "test",
        }

    monkeypatch.setattr(closed_loop_smoke, "_run_closed_loop_owned", owned_runner)
    code, report = run_cli(cfg)
    assert code == 0 and report["status"] == "completed_unqualified_diagnostic"
    assert calls["reservation"].report_claimed is True
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["status"] == "completed_unqualified_diagnostic"
