"""Source-behavior tests for the MiniVLA and SuSIE local-only loaders."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np
import pytest

from plumb.adapters.contracts import CapabilityStatus, PolicyObservation
from plumb.policies.diagnostics.minivla import (
    MINIVLA_BRIDGE_UNNORM_KEY,
    MINIVLA_AUXILIARY_ASSET_BLOCKER,
    MINIVLA_CONFIG_SHA256,
    MINIVLA_DATASET_STATISTICS_SHA256,
    MINIVLA_MODEL_REVISION,
    MINIVLA_SOURCE_COMMIT,
    MINIVLA_TORCHVISION_VERSION,
    MINIVLA_TORCH_VERSION,
    MINIVLA_TRANSFORMERS_VERSION,
    MINIVLA_VQBET_SOURCE_COMMIT,
    MINIVLA_VQ_CONFIG_SHA256,
    MINIVLA_VQ_LICENSE_BLOCKER,
    MINIVLA_VQ_MODEL_REVISION,
    MINIVLA_VQ_CHECKPOINT_SHA256,
    MINIVLA_TOKENIZERS_VERSION,
    MINIVLA_TIMM_VERSION,
    MINIVLA_CHECKPOINT_SHA256,
    MiniVLAPolicyAdapter,
    MiniVLAPolicyProfile,
    _MiniVLARuntime,
)
from plumb.policies.diagnostics.susie import (
    SUSIE_AUTOEVAL_ALGORITHM,
    SUSIE_CORRECTED_SENSITIVITY_ARM_ID,
    SUSIE_DIFFUSERS_VERSION,
    SUSIE_FLAX_VERSION,
    SUSIE_HIGH_LEVEL_MODEL_REVISION,
    SUSIE_HIGH_LEVEL_UNET_SHA256,
    SUSIE_HIGH_LEVEL_UNET_CONFIG_SHA256,
    SUSIE_JAX_VERSION,
    SUSIE_LL_REPLICATION_ARM_ID,
    SUSIE_LOW_LEVEL_GCBC_CHECKPOINT_SHA256,
    SUSIE_LOW_LEVEL_GCBC_MODEL_REVISION,
    SUSIE_REPLICATION_ARM_ID,
    SUSIE_SOURCE_COMMIT,
    SUSIE_BRIDGE_DATA_SOURCE_COMMIT,
    SUSIE_TRANSFORMERS_VERSION,
    SuSIELowLevelPolicyAdapter,
    SuSIEPolicyAdapter,
    SuSIEPolicyProfile,
    _SuSIERuntime,
)


REVISION = "a" * 40


class FakeImage:
    def __init__(self, source):
        self.source = source
        self.converted_to = None

    def convert(self, mode):
        self.converted_to = mode
        return self


class FakeMiniModel:
    def __init__(self):
        self.calls = []

    def predict_action(self, image, instruction, *, unnorm_key):
        self.calls.append((image, instruction, unnorm_key))
        return [0, 1, 2, 3, 4, 5, 6]


def mini_runtime():
    return _MiniVLARuntime(
        torch=object(),
        numpy=np,
        image_fromarray=FakeImage,
        model_config_cls=None,
        vision_factory=None,
        llm_factory=None,
        vla_cls=None,
        vqvae_cls=None,
        transformers_version=MINIVLA_TRANSFORMERS_VERSION,
        torch_version=MINIVLA_TORCH_VERSION,
        torchvision_version=MINIVLA_TORCHVISION_VERSION,
        tokenizers_version=MINIVLA_TOKENIZERS_VERSION,
        timm_version=MINIVLA_TIMM_VERSION,
    )


def mini_profile(**changes):
    value = dict(
        profile_id="mini-fixture",
        local_checkpoint_path="/unavailable/model.pt",
        local_config_path="/unavailable/config.json",
        local_dataset_statistics_path="/unavailable/dataset_statistics.json",
        local_vq_checkpoint_path="/unavailable/vq.pt",
        local_vq_config_path="/unavailable/vq-config.json",
        model_revision=MINIVLA_MODEL_REVISION,
        vq_revision=MINIVLA_VQ_MODEL_REVISION,
        code_revision=MINIVLA_SOURCE_COMMIT,
        vq_code_revision=MINIVLA_VQBET_SOURCE_COMMIT,
        checkpoint_sha256=MINIVLA_CHECKPOINT_SHA256,
        config_sha256=MINIVLA_CONFIG_SHA256,
        dataset_statistics_sha256=MINIVLA_DATASET_STATISTICS_SHA256,
        vq_checkpoint_sha256=MINIVLA_VQ_CHECKPOINT_SHA256,
        vq_config_sha256=MINIVLA_VQ_CONFIG_SHA256,
        vq_license_status="verified",
        vq_license_evidence_uri="artifact://licenses/minivla-vq-terms.txt",
    )
    value.update(changes)
    return MiniVLAPolicyProfile(**value)


def test_minivla_license_is_a_hard_block_before_any_fixture_factory_can_run():
    adapter = MiniVLAPolicyAdapter(
        mini_profile(vq_license_status="unresolved", vq_license_evidence_uri=None),
        runtime_factory=mini_runtime,
        model_factory=lambda profile, runtime: FakeMiniModel(),
    )

    capability = adapter.capability()

    assert capability.status is CapabilityStatus.BLOCKED
    assert capability.details["vq_license_blocker"] == MINIVLA_VQ_LICENSE_BLOCKER
    assert "Do not acquire, copy, convert, redistribute, or use" in capability.reason


def test_minivla_uses_source_prismatic_call_with_bridge_dataset_and_exposes_first_action_only():
    model = FakeMiniModel()
    adapter = MiniVLAPolicyAdapter(
        mini_profile(),
        runtime_factory=mini_runtime,
        model_factory=lambda profile, runtime: model,
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    report = adapter.predict_with_report(PolicyObservation(image_history=(image,), prompt="Close the drawer", timestamp=4.0))

    assert report.action == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert report.unnorm_key == MINIVLA_BRIDGE_UNNORM_KEY
    assert report.vq_input_horizon == 8
    assert report.vq_future_action_horizon == 7
    assert report.returned_action_count == 1
    source_image, instruction, unnorm_key = model.calls[0]
    assert source_image.source is image and source_image.converted_to == "RGB"
    assert instruction == "Close the drawer"
    assert unnorm_key == "bridge_dataset"
    assert adapter.capability().status is CapabilityStatus.READY_UNQUALIFIED

    with pytest.raises(ValueError, match="exactly one fresh"):
        adapter.predict_action(PolicyObservation(image_history=(image, image), prompt="Close the drawer"))
    with pytest.raises(ValueError, match="does not take proprioception"):
        adapter.predict_action(PolicyObservation(image_history=(image,), prompt="Close the drawer", proprio=(0.0,)))


def test_minivla_rejects_openvla_bridge_orig_substitution():
    adapter = MiniVLAPolicyAdapter(mini_profile(unnorm_key="bridge_orig"))
    assert adapter.capability().status is CapabilityStatus.BLOCKED
    assert "bridge_dataset" in adapter.capability().reason


def test_minivla_requires_reviewed_source_checkouts_for_a_real_runtime_not_just_revision_strings():
    capability = MiniVLAPolicyAdapter(mini_profile()).capability()

    assert capability.status is CapabilityStatus.UNAVAILABLE
    assert any("local_source_path" in value for value in capability.details["missing_source_paths"])
    assert any("local_vq_source_path" in value for value in capability.details["missing_source_paths"])


def test_minivla_real_construction_has_a_separate_auxiliary_timm_qwen_asset_blocker():
    assert MiniVLAPolicyAdapter(mini_profile()).profile.auxiliary_assets_error() == MINIVLA_AUXILIARY_ASSET_BLOCKER
    assert "timm.create_model(..., pretrained=True)" in MINIVLA_AUXILIARY_ASSET_BLOCKER


class FakeRandom:
    def __init__(self):
        self.split_calls = []

    def PRNGKey(self, seed):
        return ("key", seed)

    def split(self, key):
        self.split_calls.append(key)
        return (("next", key), ("sample", key))


class FakeJax:
    def __init__(self):
        self.random = FakeRandom()

    @staticmethod
    def device_get(value):
        return value


class FakeAgent:
    def __init__(self):
        self.calls = []

    def sample_actions(self, observations, goals, **kwargs):
        self.calls.append((observations, goals, kwargs))
        return np.asarray([[1.0, -1.0, 0.0, 0.5, -0.5, 2.0, -0.1]], dtype=np.float32)


class FakeAgentClass:
    created = []

    @classmethod
    def create(cls, **kwargs):
        cls.created.append(kwargs)
        return FakeAgent()


@dataclass
class FakeGoalFactory:
    calls: list

    def __call__(self, image, prompt):
        self.calls.append((image, prompt))
        return np.full((256, 256, 3), 19, dtype=np.uint8)


class FakeSamplerFactory:
    def __init__(self):
        self.args = []
        self.goal = FakeGoalFactory([])

    def __call__(self, *args):
        self.args.append(args)
        return self.goal


def susie_runtime(sampler_factory):
    return _SuSIERuntime(
        jax=FakeJax(),
        numpy=np,
        agent_registry={SUSIE_AUTOEVAL_ALGORITHM: FakeAgentClass},
        encoder_registry={"resnetv1-34": lambda **kwargs: ("encoder", kwargs)},
        restore_checkpoint=lambda checkpoint_dir, target: target,
        create_sample_fn=sampler_factory,
        jax_version=SUSIE_JAX_VERSION,
        flax_version=SUSIE_FLAX_VERSION,
        diffusers_version=SUSIE_DIFFUSERS_VERSION,
        transformers_version=SUSIE_TRANSFORMERS_VERSION,
    )


def susie_profile(arm_id=SUSIE_REPLICATION_ARM_ID, **changes):
    value = dict(
        profile_id="susie-fixture",
        arm_id=arm_id,
        local_low_level_checkpoint_dir="/unavailable/gcbc/checkpoint",
        low_level_checkpoint_sha256=SUSIE_LOW_LEVEL_GCBC_CHECKPOINT_SHA256,
        local_high_level_model_path="/unavailable/susie",
        local_stable_diffusion_path="/unavailable/sd15",
        stable_diffusion_revision=REVISION,
        stable_diffusion_asset_sha256="b" * 64,
        stable_diffusion_license_evidence_uri="artifact://licenses/sd15.txt",
        local_stable_diffusion_manifest_path="/unavailable/sd15-assets.lock.json",
        low_level_model_revision=SUSIE_LOW_LEVEL_GCBC_MODEL_REVISION,
        high_level_model_revision=SUSIE_HIGH_LEVEL_MODEL_REVISION,
        high_level_unet_sha256=SUSIE_HIGH_LEVEL_UNET_SHA256,
        high_level_unet_config_sha256=SUSIE_HIGH_LEVEL_UNET_CONFIG_SHA256,
        source_revision=SUSIE_SOURCE_COMMIT,
        bridge_data_source_revision=SUSIE_BRIDGE_DATA_SOURCE_COMMIT,
    )
    value.update(changes)
    return SuSIEPolicyProfile(**value)


def _fixture_adapter(adapter):
    # Source artifact hashes are deliberately immutable production gates.  The
    # fake runtime below has no 3.4 GB/258 MB assets, so retain the source call
    # path while avoiding an invented fixture artifact.
    adapter._validate_artifacts = lambda: None
    return adapter


def test_susie_replication_preserves_gc_bc_normalization_and_twenty_step_goal_cadence():
    FakeAgentClass.created.clear()
    sampler = FakeSamplerFactory()
    adapter = _fixture_adapter(SuSIEPolicyAdapter(susie_profile(), runtime_factory=lambda: susie_runtime(sampler)))
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    first = adapter.predict_with_report(PolicyObservation(image_history=(image,), prompt="Open the drawer", timestamp=1.0))
    second = adapter.predict_with_report(PolicyObservation(image_history=(image,), prompt="Open the drawer", timestamp=2.0))
    for _ in range(19):
        adapter.predict_action(PolicyObservation(image_history=(image,), prompt="Open the drawer"))

    assert first.arm_id == SUSIE_REPLICATION_ARM_ID
    assert first.algorithm == "gc_bc"
    assert first.goal_refreshed is True and second.goal_refreshed is False
    assert first.subgoal_cadence == 20
    assert len(sampler.goal.calls) == 2  # policy steps 0 and 20
    assert first.action[:6] == pytest.approx((
        0.00912848 + 1.9296819e-04,
        -0.0127196 + 1.3667766e-04,
        -1.4583133e-04,
        0.5 * 0.02606696 - 1.8390431e-04,
        -0.5 * 0.02875283 - 3.0808983e-04,
        2.0 * 0.07807977 + 2.7425270e-04,
    ))
    assert first.action[-1] == 0.0
    agent = adapter._agent
    assert all(call[2] == {"temperature": 0.0, "argmax": True, "seed": None} for call in agent.calls)
    assert sampler.args[0][1] is None  # never select source's W&B/Orbax branch
    assert sampler.args[0][2:6] == (50, 7.5, 4.0, 0.0)


def test_susie_ll_requires_a_scenario_goal_image_and_never_constructs_a_language_goal():
    FakeAgentClass.created.clear()
    sampler = FakeSamplerFactory()
    profile = susie_profile(
        arm_id=SUSIE_LL_REPLICATION_ARM_ID,
        local_high_level_model_path=None,
        local_stable_diffusion_path=None,
        stable_diffusion_revision=None,
        stable_diffusion_asset_sha256=None,
        stable_diffusion_license_evidence_uri=None,
    )
    adapter = _fixture_adapter(SuSIELowLevelPolicyAdapter(profile, runtime_factory=lambda: susie_runtime(sampler)))
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    goal = np.full((256, 256, 3), 7, dtype=np.uint8)

    with pytest.raises(ValueError, match="scenario-supplied goal"):
        adapter.predict_action(PolicyObservation(image_history=(image,), prompt="Open the drawer"))
    report = adapter.predict_with_report(PolicyObservation(image_history=(image,), goal_image=goal, prompt="Open the drawer"))

    assert report.arm_id == SUSIE_LL_REPLICATION_ARM_ID
    assert report.goal_refreshed is False and report.subgoal_cadence is None
    assert sampler.args == []
    assert np.array_equal(adapter._agent.calls[0][1]["image"][0], goal)


def test_susie_corrected_gc_ddpm_arm_is_a_distinct_id_and_cannot_reuse_gc_bc_weights():
    profile = susie_profile(arm_id=SUSIE_CORRECTED_SENSITIVITY_ARM_ID)
    capability = SuSIEPolicyAdapter(profile).capability()

    assert capability.status is CapabilityStatus.BLOCKED
    assert "gc_ddpm_bc" in capability.reason
    assert "cannot be loaded" in capability.reason


def test_susie_real_runtime_requires_verified_local_code_checkouts():
    capability = SuSIEPolicyAdapter(susie_profile()).capability()

    assert capability.status is CapabilityStatus.UNAVAILABLE
    assert any("local_susie_source_path" in value for value in capability.details["missing_source_paths"])
    assert any("local_bridge_data_source_path" in value for value in capability.details["missing_source_paths"])
    assert any("local_autoeval_source_path" in value for value in capability.details["missing_source_paths"])


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_diffusion_manifest(tmp_path):
    root = tmp_path / "sd15"
    records = []
    for directory, filename, body in (
        ("vae", "config.json", b"vae-config"),
        ("text_encoder", "config.json", b"text-config"),
        ("tokenizer", "tokenizer.json", b"tokenizer"),
    ):
        path = root / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        records.append({"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)})
    manifest = tmp_path / "sd15-assets.json"
    manifest.write_text(json.dumps({"model_id": "lodestones/stable-diffusion-v1-5-flax", "revision": REVISION, "file_records": records}))
    return root, manifest


def test_susie_stable_diffusion_manifest_binds_every_loaded_component_file(tmp_path):
    root, manifest = _stable_diffusion_manifest(tmp_path)
    profile = replace(
        susie_profile(),
        local_stable_diffusion_path=str(root),
        local_stable_diffusion_manifest_path=str(manifest),
        stable_diffusion_asset_sha256=_sha256(manifest),
    )
    adapter = SuSIEPolicyAdapter(profile, runtime_factory=lambda: susie_runtime(FakeSamplerFactory()))

    adapter._validate_stable_diffusion_manifest()
    (root / "vae" / "config.json").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="file SHA-256 mismatch"):
        adapter._validate_stable_diffusion_manifest()


def test_susie_stable_diffusion_manifest_rejects_path_escapes(tmp_path):
    root, manifest = _stable_diffusion_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["file_records"][0]["path"] = "../outside.bin"
    manifest.write_text(json.dumps(payload))
    profile = replace(
        susie_profile(),
        local_stable_diffusion_path=str(root),
        local_stable_diffusion_manifest_path=str(manifest),
        stable_diffusion_asset_sha256=_sha256(manifest),
    )
    adapter = SuSIEPolicyAdapter(profile, runtime_factory=lambda: susie_runtime(FakeSamplerFactory()))

    with pytest.raises(RuntimeError, match="may only record files"):
        adapter._validate_stable_diffusion_manifest()
