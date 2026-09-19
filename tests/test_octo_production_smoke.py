"""Offline guards for the separate production-Octo cluster smoke harness."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cluster import octo_production_smoke as smoke
from plumb.policies.octo import (
    OCTO_PRODUCTION_SOURCE_COMMIT,
    OCTO_V1_0_OBSERVATION_KEYS,
    OCTO_SMALL_CHECKPOINT_STEP,
    OCTO_SMALL_MODEL_ID,
    OctoPolicyProfile,
    _OctoRuntime,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pixel_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def test_production_profile_binds_241fb_and_its_timestep_mask_key():
    profile = OctoPolicyProfile(
        profile_id="fixture",
        local_model_path="/missing",
        model_id=OCTO_SMALL_MODEL_ID,
        checkpoint_revision="a" * 40,
        checkpoint_step=OCTO_SMALL_CHECKPOINT_STEP,
    )

    assert OCTO_PRODUCTION_SOURCE_COMMIT == smoke.PRODUCTION_SOURCE_COMMIT
    assert OCTO_V1_0_OBSERVATION_KEYS.pad_mask_key == "timestep_pad_mask"
    assert profile.review_error() is None
    assert OctoPolicyProfile(
        profile_id="fixture",
        local_model_path="/missing",
        model_id=OCTO_SMALL_MODEL_ID,
        checkpoint_revision="a" * 40,
        checkpoint_step=OCTO_SMALL_CHECKPOINT_STEP,
        source_revision="b" * 40,
    ).review_error() == "source_revision must equal the pinned production Octo source commit"


def test_production_smoke_png_loader_binds_the_existing_two_frame_manifest(tmp_path):
    first = np.broadcast_to(np.asarray([1, 2, 3], dtype=np.uint8), (256, 256, 3)).copy()
    second = np.broadcast_to(np.asarray([7, 8, 9], dtype=np.uint8), (256, 256, 3)).copy()
    frames = (tmp_path / "frame-00.png", tmp_path / "frame-01.png")
    Image.fromarray(first, "RGB").save(frames[0])
    Image.fromarray(second, "RGB").save(frames[1])
    manifest = {
        "schema": "plumb-octo-canonical-inputs-v1",
        "qualified": False,
        "preprocessing": {"kind": "fixture"},
        "frames": [
            {"source_frame_index": 0, "output_file": frames[0].name, "output_file_sha256": _sha(frames[0]), "output_rgb_pixel_sha256": _pixel_sha(first)},
            {"source_frame_index": 2, "output_file": frames[1].name, "output_file_sha256": _sha(frames[1]), "output_rgb_pixel_sha256": _pixel_sha(second)},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded_first, loaded_second, bound = smoke._frames(*frames, manifest_path)

    np.testing.assert_array_equal(loaded_first, first)
    np.testing.assert_array_equal(loaded_second, second)
    assert [record["source_frame_index"] for record in bound["frames"]] == [0, 2]


def test_production_mask_is_explicit_source_default_when_statistics_omit_a_mask(tmp_path):
    path = tmp_path / "dataset_statistics.json"
    path.write_text(json.dumps({"bridge_dataset": {"action": {"mean": [0.0] * 7, "std": [1.0] * 7}}}), encoding="utf-8")

    assert smoke._action_mask(path) == ((True,) * 7, "241fb_sample_actions_default_ones_like_mean")

    path.write_text(json.dumps({"bridge_dataset": {"action": {"mean": [0.0] * 7, "std": [1.0] * 7, "mask": [True] * 6}}}), encoding="utf-8")
    with pytest.raises(smoke.OctoProductionSmokeError, match="seven booleans"):
        smoke._action_mask(path)


def test_gpu_metadata_uses_explicit_jax_for_the_real_adapter_runtime_shape(monkeypatch):
    runtime = _OctoRuntime(
        jax_version="0.4.20",
        array_factory=lambda value, dtype: value,
        prng_key=lambda seed: ("prng", seed),
        fold_in=lambda key, step: (key, step),
        model_loader=lambda path, step: None,
    )

    class Device:
        platform = "gpu"
        device_kind = "H100"

        @staticmethod
        def memory_stats():
            return {"peak_bytes_in_use": 17}

        def __str__(self):
            return "gpu:0"

    class Jax:
        __version__ = "0.4.20"

        @staticmethod
        def devices():
            return (Device(),)

    monkeypatch.setattr(
        smoke,
        "_nvidia_smi_metadata",
        lambda: {"gpu_uuid": None, "gpu_model": None, "driver_version": None, "gpus": None, "returncode": None, "error": "unavailable"},
    )
    assert not hasattr(runtime, "jax")
    metadata = smoke._gpu_metadata_for_adapter_runtime(runtime, Jax)
    assert metadata["gpu_peak_memory_bytes"] == 17
    assert metadata["nvidia_smi"]["gpu_uuid"] is None

    Jax.__version__ = "0.4.21"
    with pytest.raises(smoke.OctoProductionSmokeError, match="does not match"):
        smoke._gpu_metadata_for_adapter_runtime(runtime, Jax)


def test_nvidia_metadata_records_physical_rows_without_guessing_cuda_mapping(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="GPU-uuid-0, NVIDIA H100 80GB HBM3, 550.54.15\nGPU-uuid-1, NVIDIA H100 80GB HBM3, 550.54.15\n",
            stderr="",
        ),
    )

    metadata = smoke._nvidia_smi_metadata()

    assert metadata["gpu_uuid"] is None
    assert metadata["gpus"] == [
        {"gpu_uuid": "GPU-uuid-0", "gpu_model": "NVIDIA H100 80GB HBM3", "driver_version": "550.54.15"},
        {"gpu_uuid": "GPU-uuid-1", "gpu_model": "NVIDIA H100 80GB HBM3", "driver_version": "550.54.15"},
    ]


def test_runtime_lock_requires_exact_suffix_inventory_dlimp_and_explicit_source_deviations(tmp_path, monkeypatch):
    root = tmp_path / "scratch" / "plumb"
    checkout = root / "source-octo-autoeval241fb"
    venv = root / "venv-octo-autoeval241fb"
    freeze = venv / "requirements-freeze-v1.txt"
    freeze.parent.mkdir(parents=True)
    freeze.write_text("octo @ git+https://github.com/octo-models/octo@241fb3514b7c40957a86d869fecb7c7fc353f540\n", encoding="utf-8")
    packages = {
        "jax": "0.4.20+computecanada",
        "jaxlib": "0.4.20+cuda12.cudnn89.computecanada",
        "flax": "0.7.5+computecanada",
        "optax": "0.1.5+computecanada",
        "chex": "0.1.85+computecanada",
        "orbax-checkpoint": "0.4.3+computecanada",
        "numpy": "1.26.4+computecanada",
        "tensorflow-probability": "0.23.0",
        "tensorflow": "2.15.1+computecanada",
        "transformers": "4.40.2+computecanada",
        "tokenizers": "0.19.1+computecanada",
        "huggingface-hub": "0.36.2+computecanada",
        "scipy": "1.11.2+computecanada",
        "tensorflow-io-gcs-filesystem": "0.32.0+computecanada",
        "imageio": "2.31.1+computecanada",
        "pillow": "11.1.0+computecanada",
        "dlimp": "0.0.0",
    }
    direct_url = {"url": "https://github.com/kvablack/dlimp", "vcs_info": {"vcs": "git", "commit_id": smoke.SOURCE_DLiMP_COMMIT}}
    lock = root / "evidence" / "octo-runtime-autoeval241fb-v1.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "schema": smoke.RUNTIME_SCHEMA,
                "runtime_profile": smoke.RUNTIME_PROFILE_ID,
                "source_checkout": str(checkout),
                "source_revision": smoke.PRODUCTION_SOURCE_COMMIT,
                "python_executable": str(venv / "bin" / "python"),
                "packages": packages,
                "pip_freeze": {"path": str(freeze), "sha256": _sha(freeze), "lines": freeze.read_text(encoding="utf-8").splitlines()},
                "dlimp": {"distribution": "dlimp", "source_revision": smoke.SOURCE_DLiMP_COMMIT, "direct_url": direct_url, "direct_url_sha256": smoke._canonical_json_sha256(direct_url)},
                "source_requirement_deviations": [
                    {"package": "numpy", "source_requirement": "1.24.3", "actual": packages["numpy"], "actual_base_version": "1.26.4", "reason": "recorded test deviation"},
                    {"package": "tensorflow", "source_requirement": "2.15.0", "actual": packages["tensorflow"], "actual_base_version": "2.15.1", "reason": "recorded test deviation"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(smoke, "_runtime_version", packages.__getitem__)
    monkeypatch.setattr(smoke, "_distribution_direct_url", lambda name: direct_url)
    monkeypatch.setattr(smoke.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=freeze.read_text(encoding="utf-8")))

    _, actual = smoke._verify_runtime_lock(root, lock, checkout)

    assert actual == packages
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["source_requirement_deviations"] = payload["source_requirement_deviations"][:1]
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(smoke.OctoProductionSmokeError, match="tensorflow deviation"):
        smoke._verify_runtime_lock(root, lock, checkout)


def test_production_report_reservation_never_replaces_existing_evidence(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"existing":true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        smoke._reserve_report(report)

    assert json.loads(report.read_text(encoding="utf-8")) == {"existing": True}
