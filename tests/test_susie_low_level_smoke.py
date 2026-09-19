"""CPU-only guards for the SuSIE_LL static-fixture diagnostic harness."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from cluster import susie_low_level_smoke as smoke
from plumb.policies.susie import (
    SUSIE_AUTOEVAL_RELEASED_ARM,
    SUSIE_AUTOEVAL_NORMALIZER_SHA256,
    SUSIE_BRIDGE_DATA_SOURCE_COMMIT,
    SUSIE_LOW_LEVEL_CHECKPOINT_FILE,
    SUSIE_LOW_LEVEL_CHECKPOINT_SHA256,
    SUSIE_LOW_LEVEL_MODEL_REVISION,
    SUSIE_LOW_LEVEL_README_SHA256,
    NativeWrapperEntryPoint,
    SuSIEPolicyProfile,
    SuSIEUnavailableError,
    _strict_param_shapes,
    _strict_state_tree,
    build_autoeval_gcbc_bridge_low_level,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pixel(array) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _inputs(tmp_path: Path, *, identical: bool = False):
    from PIL import Image

    tmp_path.mkdir(parents=True, exist_ok=True)
    current = np.zeros((256, 256, 3), dtype=np.uint8)
    goal = current.copy() if identical else np.full((256, 256, 3), 9, dtype=np.uint8)
    first, second = tmp_path / "current.png", tmp_path / "goal.png"
    Image.fromarray(current).save(first)
    Image.fromarray(goal).save(second)
    manifest = {
        "schema": "plumb-octo-canonical-inputs-v1",
        "qualified": False,
        "source": {"sha256": smoke.VENDOR_FIXTURE_SHA256, "frame_indices": [0, 16]},
        "preprocessing": {"limitation": "static source-video goal fixture only"},
        "frames": [
            {
                "source_frame_index": 0,
                "output_file": first.name,
                "output_file_sha256": _sha(first),
                "output_rgb_pixel_sha256": _pixel(current),
            },
            {
                "source_frame_index": 16,
                "output_file": second.name,
                "output_file_sha256": _sha(second),
                "output_rgb_pixel_sha256": _pixel(goal),
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, first, second, current, goal


def test_static_goal_fixture_requires_hash_bound_distinct_uint8_current_and_goal(tmp_path):
    manifest, current_path, goal_path, current, goal = _inputs(tmp_path)

    observed_current, observed_goal, payload = smoke._load_inputs(manifest, current_path, goal_path)

    assert np.array_equal(observed_current, current)
    assert np.array_equal(observed_goal, goal)
    assert payload["frames"][1]["source_frame_index"] == 16

    bad_manifest, bad_current, bad_goal, _, _ = _inputs(tmp_path / "identical", identical=True)
    with pytest.raises(smoke.SuSIELowLevelSmokeError, match="byte-identical"):
        smoke._load_inputs(bad_manifest, bad_current, bad_goal)


def test_harness_uses_direct_published_checkpoint_and_leaves_binary_gripper_unmasked():
    normalizer, _ = smoke._normalizer()

    assert SUSIE_LOW_LEVEL_CHECKPOINT_FILE == "checkpoint/checkpoint"
    assert len(SUSIE_LOW_LEVEL_CHECKPOINT_SHA256) == 64
    assert normalizer.statistics.statistics_sha256 == SUSIE_AUTOEVAL_NORMALIZER_SHA256
    assert normalizer.statistics.mask == (True, True, True, True, True, True, False)
    assert normalizer.denormalize((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))[-1] == 1.0


class _FakeJax:
    __version__ = "0.4.20+computecanada"

    class random:
        @staticmethod
        def PRNGKey(seed):
            return ("key", seed)

        @staticmethod
        def split(key):
            return (("left", key), ("right", key))

    class tree_util:
        @staticmethod
        def tree_leaves(value):
            return [value]

        @staticmethod
        def tree_flatten(value):
            return [value], "fixture-tree"

    @staticmethod
    def device_get(value):
        return value


class _FakeFlax:
    __version__ = "0.7.5+computecanada"


class _FakeAgent:
    def __init__(self, restored=False):
        self.restored = restored
        self.calls = []
        self.state = _FakeState(np.asarray([2 if restored else 1], dtype=np.float32), None)

    def tobytes(self):
        return b"restored" if self.restored else b"initial"

    def sample_actions(self, observations, goals, **kwargs):
        self.calls.append((observations, goals, kwargs))
        action = np.asarray([[1.0, -2.0, 0.0, 0.5, -0.5, 2.0, -0.1]], dtype=np.float32)
        return (action, action.copy()) if _FakeAgentClass.return_tuple else action

    def replace(self, *, state):
        copied = _FakeAgent(restored=self.restored)
        copied.state = state
        return copied


class _FakeState:
    def __init__(self, params, target_params):
        self.params = params
        self.target_params = target_params

    def replace(self, *, params, target_params):
        return _FakeState(params, target_params)


class _FakeAgentClass:
    created = []
    return_tuple = False

    @classmethod
    def create(cls, **kwargs):
        cls.created.append(kwargs)
        return _FakeAgent()


def _install_fake_bridge_runtime(monkeypatch, *, restored=True, raw_target=None):
    import plumb.policies.susie as susie

    agents_module = ModuleType("jaxrl_m.agents")
    agents_module.agents = {"gc_bc": _FakeAgentClass}
    vision_module = ModuleType("jaxrl_m.vision")
    vision_module.encoders = {"resnetv1-34": lambda **kwargs: ("encoder", kwargs)}
    training_module = ModuleType("flax.training")
    checkpoints_module = ModuleType("flax.training.checkpoints")
    def restore_checkpoint(path, target):
        if target is None:
            return {
                "state": {
                    "opt_states": {"training_only": "excluded"},
                    "params": np.asarray([2], dtype=np.float32),
                    "rng": np.asarray([0, 1], dtype=np.uint32),
                    "step": np.asarray(75000, dtype=np.int32),
                    "target_params": raw_target,
                }
            }
        return _FakeAgent(restored=restored)

    checkpoints_module.restore_checkpoint = restore_checkpoint
    serialization_module = ModuleType("flax.serialization")
    serialization_module.to_state_dict = lambda value: value
    serialization_module.from_state_dict = lambda target, state: state
    training_module.checkpoints = checkpoints_module
    flax_module = ModuleType("flax")
    flax_module.__version__ = _FakeFlax.__version__
    flax_module.serialization = serialization_module
    monkeypatch.setitem(sys.modules, "jax", _FakeJax)
    monkeypatch.setitem(sys.modules, "flax", flax_module)
    monkeypatch.setitem(sys.modules, "flax.training", training_module)
    monkeypatch.setitem(sys.modules, "flax.training.checkpoints", checkpoints_module)
    monkeypatch.setitem(sys.modules, "flax.serialization", serialization_module)
    monkeypatch.setitem(sys.modules, "jaxrl_m.agents", agents_module)
    monkeypatch.setitem(sys.modules, "jaxrl_m.vision", vision_module)
    monkeypatch.setattr(susie, "_verify_clean_source", lambda root: root / "model_training")
    monkeypatch.setattr(susie, "_under", lambda path, root: True)
    monkeypatch.setattr(
        susie,
        "_sha256_file",
        lambda path: SUSIE_LOW_LEVEL_README_SHA256 if path.name == "README.md" else SUSIE_LOW_LEVEL_CHECKPOINT_SHA256,
    )


def _direct_profile(tmp_path: Path, *, readme_text="---\nlicense: mit\n---\n"):
    checkpoint = tmp_path / "model" / "checkpoint" / "checkpoint"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture-checkpoint")
    readme = checkpoint.parents[1] / "README.md"
    readme.write_text(readme_text, encoding="utf-8")
    source = tmp_path / "bridge-source"
    source.mkdir()
    return SuSIEPolicyProfile(
        profile_id="direct-gcbc-fixture",
        low_level_model_path=str(checkpoint),
        low_level_revision=SUSIE_LOW_LEVEL_MODEL_REVISION,
        arm=SUSIE_AUTOEVAL_RELEASED_ARM,
        low_level_entry_point=NativeWrapperEntryPoint(
            module="plumb.policies.susie",
            attribute="build_autoeval_gcbc_bridge_low_level",
            loader_revision=SUSIE_BRIDGE_DATA_SOURCE_COMMIT,
            source_repo_path=str(source),
        ),
        goal_image_source="fixture-goal",
        jax_version="0.4.20+computecanada",
        low_level_license="MIT declared",
        low_level_license_path=str(readme),
    )


def test_direct_gc_bc_loader_uses_exact_autoeval_construction_and_thresholds_only_gripper(monkeypatch, tmp_path):
    _FakeAgentClass.created.clear()
    _FakeAgentClass.return_tuple = False
    _install_fake_bridge_runtime(monkeypatch, restored=True)
    profile = _direct_profile(tmp_path)

    model = build_autoeval_gcbc_bridge_low_level(profile)
    raw = model.sample_actions(
        image=np.zeros((256, 256, 3), dtype=np.uint8),
        goal_image=np.full((256, 256, 3), 2, dtype=np.uint8),
        rng=("unused", 9),
    )

    created = _FakeAgentClass.created[0]
    assert created["early_goal_concat"] is True
    assert created["shared_goal_encoder"] is True and created["use_proprio"] is False
    assert created["network_kwargs"] == {"hidden_dims": (256, 256, 256), "dropout_rate": 0.1}
    assert created["policy_kwargs"]["fixed_std"] == [1, 1, 1, 1, 1, 1, 0.1]
    assert raw == [[1.0, -2.0, 0.0, 0.5, -0.5, 2.0, 0.0]]
    assert model.last_model_normalized[-1] == pytest.approx(-0.1)
    assert model.last_wrapper_boundary[-1] == 0.0
    assert model.restore_digest["changed"] is True
    assert model._agent.calls[0][2] == {"temperature": 0.0, "argmax": True, "seed": None}


def test_direct_gc_bc_loader_rejects_silent_restore_and_bad_license(monkeypatch, tmp_path):
    _install_fake_bridge_runtime(monkeypatch, restored=False)
    with pytest.raises(SuSIEUnavailableError, match="byte-identical"):
        build_autoeval_gcbc_bridge_low_level(_direct_profile(tmp_path))

    _install_fake_bridge_runtime(monkeypatch, restored=True)
    with pytest.raises(SuSIEUnavailableError, match="does not declare"):
        build_autoeval_gcbc_bridge_low_level(_direct_profile(tmp_path / "bad", readme_text="no license notice"))


def test_direct_gc_bc_params_only_restore_rejects_optimizer_and_restores_strict_params(monkeypatch, tmp_path):
    _install_fake_bridge_runtime(monkeypatch, restored=False)
    profile = _direct_profile(tmp_path)
    profile = profile.__class__(**{**profile.__dict__, "inference_params_only": True})

    model = build_autoeval_gcbc_bridge_low_level(profile)

    assert model.restore_digest["method"] == "inference_params_only_source_checkpoint_no_optimizer_restore"
    assert model.restore_digest["optimizer_state_excluded"] is True
    assert model.restore_digest["raw_checkpoint_keys"] == {"top_level": ["state"], "state": ["opt_states", "params", "rng", "step", "target_params"]}
    assert model.restore_digest["inference_only_not_resumable"] is True


def test_strict_params_restore_rejects_missing_extra_shape_dtype_and_non_none_target(monkeypatch, tmp_path):
    with pytest.raises(SuSIEUnavailableError, match="keys differ"):
        _strict_state_tree({"actor": {"weight": 1}}, {"actor": {"extra": 1}})

    class ShapeJax:
        class tree_util:
            @staticmethod
            def tree_flatten(value):
                return [value], "same-tree"

        @staticmethod
        def device_get(value):
            return value

    with pytest.raises(SuSIEUnavailableError, match="shape differs"):
        _strict_param_shapes(ShapeJax, np.zeros((1,), dtype=np.float32), np.zeros((2,), dtype=np.float32))
    with pytest.raises(SuSIEUnavailableError, match="dtype differs"):
        _strict_param_shapes(ShapeJax, np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float16))

    _install_fake_bridge_runtime(monkeypatch, raw_target=np.asarray([9], dtype=np.float32))
    profile = _direct_profile(tmp_path)
    profile = profile.__class__(**{**profile.__dict__, "inference_params_only": True})
    with pytest.raises(SuSIEUnavailableError, match="target_params=None"):
        build_autoeval_gcbc_bridge_low_level(profile)


def test_direct_gc_bc_accepts_source_tuple_actions_action_mode(monkeypatch, tmp_path):
    _FakeAgentClass.return_tuple = True
    _install_fake_bridge_runtime(monkeypatch, restored=True)
    model = build_autoeval_gcbc_bridge_low_level(_direct_profile(tmp_path))
    row = model.sample_actions(
        image=np.zeros((256, 256, 3), dtype=np.uint8),
        goal_image=np.ones((256, 256, 3), dtype=np.uint8),
        rng=None,
    )
    assert row[0][-1] == 0.0
    _FakeAgentClass.return_tuple = False


def test_harness_full_cpu_fake_path_records_raw_transformed_repeat_and_runtime_deviation(monkeypatch, tmp_path):
    import plumb.policies.susie as susie

    manifest, current, goal, _, _ = _inputs(tmp_path / "fixture")
    checkpoint = tmp_path / "checkpoint" / "checkpoint"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    readme = checkpoint.parents[1] / "README.md"
    readme.write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    runtime_lock = tmp_path / "octo-runtime-cuda12-jax0420-v2.json"
    runtime_lock.write_text(json.dumps({"schema": "plumb-octo-runtime-lock-v1", "packages": {"jax": "0.4.20+computecanada"}}))

    class FakeJax:
        __version__ = "0.4.20+computecanada"

        @staticmethod
        def devices():
            return (SimpleNamespace(platform="gpu", device_kind="H100", __str__=lambda self: "fake-gpu"),)

    class FakeLowLevel:
        _jax = FakeJax()
        last_model_normalized = (1.0, -2.0, 0.0, 0.5, -0.5, 2.0, -0.1)
        source_runtime_deviation = {"core_jax_flax_match": True, "full_runtime_tuple_verified": False, "actual_jax_version": "0.4.20+computecanada", "source_required_versions": {"jax": "0.4.20", "numpy": "1.24.3"}}
        restore_digest = {"initialized": "a", "restored": "b", "changed": True}

    class FakeAdapter:
        def __init__(self, profile, *, normalizer, execution_mode):
            self._low_level = FakeLowLevel()
            self._runtime = SimpleNamespace(jax_version="0.4.20+computecanada")

        def reset(self, seed):
            self.seed = seed

        def propose(self, observation):
            return SimpleNamespace(actions=((0.00932144819, -0.0253021332234, -0.00014583133, 0.012849576956, -0.01468450583, 0.1564332274, 0.0),), backend_calls=1)

        def _load_runtime(self):
            return self._runtime

    monkeypatch.setattr(susie, "SuSIELowLevelPolicyAdapter", FakeAdapter)
    monkeypatch.setattr(smoke, "_disable_tensorflow_gpu", lambda: "2.15.1+computecanada")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-job")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    args = SimpleNamespace(
        root=tmp_path,
        checkpoint=checkpoint,
        checkpoint_readme=readme,
        bridge_source=bridge,
        input_manifest=manifest,
        frame_0=current,
        goal_frame=goal,
        runtime_lock=runtime_lock,
        source_release="a" * 64,
        profile_id="fixture-profile",
        jax_version="0.4.20+computecanada",
        seed=42,
        prompt="fixture prompt",
    )

    payload = smoke.run(args)

    assert payload["status"] == "completed_unqualified"
    assert payload["actions"]["native_model_normalized"][-1] == pytest.approx(-0.1)
    assert payload["actions"]["transformed_physical"][0][-1] == 0.0
    assert payload["restore"]["changed"] is True
    assert payload["runtime"]["tensorflow_gpu_visible"] is False
    assert payload["runtime"]["dependency_deviation"]["full_runtime_tuple_verified"] is False
    assert payload["timing"]["total_native_calls"] == 2
    assert payload["high_level_subgoal_assets_imported"] is False
