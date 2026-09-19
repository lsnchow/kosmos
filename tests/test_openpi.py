"""Offline source-contract tests for the local-only OpenPiZero adapter."""

from __future__ import annotations

from contextlib import nullcontext
import numpy as np
import pytest

from plumb.adapters.contracts import CapabilityStatus, PolicyObservation
from plumb.policies.diagnostics.openpi import (
    OPENPI_ACTION_HORIZON,
    OPENPI_BRIDGE_EEF_QUATERNION_WXYZ,
    OPENPI_SOURCE_COMMIT,
    OPENPI_TORCH_VERSION,
    OPENPI_TRANSFORMERS_VERSION,
    OpenPiUnavailableError,
    OpenPiZeroPolicyAdapter,
    OpenPiZeroPolicyProfile,
    _OpenPiRuntime,
)


REVISION = "a" * 40
CHECKSUM = "b" * 64


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def permute(self, *axes):
        return FakeTensor(np.transpose(self.value, axes))

    def __getitem__(self, item):
        return FakeTensor(self.value[item])

    def to(self, *args, **kwargs):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeTorch:
    uint8 = "uint8"
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(self):
        self.loads = []

    @staticmethod
    def as_tensor(value, dtype=None):
        return FakeTensor(value)

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    def load(self, path, *, map_location, weights_only):
        self.loads.append((path, map_location, weights_only))
        return {"model": {"_orig_mod.weight": FakeTensor([1.0])}}


class FakeCV2:
    INTER_LANCZOS4 = 4

    def __init__(self):
        self.calls = []

    def resize(self, image, size, interpolation):
        self.calls.append((image.shape, size, interpolation))
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, text, images):
        self.calls.append((text, images))
        return {
            "input_ids": FakeTensor([[10, 11]]),
            "pixel_values": FakeTensor(np.zeros((1, 3, 224, 224), dtype=np.float32)),
            "attention_mask": FakeTensor([[1, 1]]),
        }


class FakeModel:
    def __init__(self, proposal):
        self.proposal = proposal
        self.mask_calls = []
        self.calls = []
        self.evaluated = False

    def eval(self):
        self.evaluated = True
        return self

    def build_causal_mask_and_position_ids(self, attention, *, dtype):
        self.mask_calls.append((attention, dtype))
        token = FakeTensor([[0]])
        return token, token, token, token

    def split_full_mask_into_submasks(self, mask):
        return mask, mask

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeTensor(self.proposal[None])


def runtime(torch, cv2):
    return _OpenPiRuntime(
        torch=torch,
        numpy=np,
        cv2=cv2,
        omega_conf=None,
        model_cls=None,
        processor_cls=None,
        tokenizer_cls=None,
        quat2mat=lambda quaternion: np.eye(3),
        mat2euler=lambda matrix: np.asarray([0.25, -0.25, 0.5]),
        torch_version=OPENPI_TORCH_VERSION,
        transformers_version=OPENPI_TRANSFORMERS_VERSION,
        module_paths={"pizero": "fixture/pizero.py", "processing": "fixture/processing.py"},
    )


def profile(**changes):
    value = dict(
        profile_id="openpi-fixture",
        local_checkpoint_path="/unavailable/bridge_beta_step19296_2024-12-26_22-30_42.pt",
        checkpoint_sha256=CHECKSUM,
        checkpoint_revision=REVISION,
        source_checkout_path="/unavailable/openpi-source",
        source_revision=OPENPI_SOURCE_COMMIT,
        autoeval_checkout_path="/unavailable/autoeval-source",
        eval_config_path="/unavailable/openpi-source/config/eval/bridge.yaml",
        autoeval_statistics_path="/unavailable/autoeval-bounds/dataset_statistics.json",
        paligemma_path="/unavailable/paligemma",
        paligemma_revision=REVISION,
        paligemma_terms_evidence_uri="artifact://licenses/paligemma-terms.txt",
        asset_manifest_path="/unavailable/openpi-assets.json",
        asset_manifest_sha256=CHECKSUM,
    )
    value.update(changes)
    return OpenPiZeroPolicyProfile(**value)


def statistics():
    return {
        "action": {"q01": [10, 20, 30, 40, 50, 60, 0], "q99": [12, 24, 36, 48, 60, 72, 1]},
        "proprio": {"q01": [0] * 7, "q99": [1] * 7},
    }


def adapter_with_fakes(proposal):
    torch = FakeTorch()
    cv2 = FakeCV2()
    model = FakeModel(np.asarray(proposal, dtype=np.float32))
    processor = FakeProcessor()
    adapter = OpenPiZeroPolicyAdapter(
        profile(),
        runtime_factory=lambda: runtime(torch, cv2),
        model_factory=lambda profile, source_runtime: model,
        processor_factory=lambda profile, source_runtime: processor,
    )
    adapter.set_fixture_statistics(statistics())
    return adapter, model, processor, cv2, torch


def observation(*, image_history=None, proprio=None):
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    return PolicyObservation(
        image_history=(image,) if image_history is None else image_history,
        prompt="open the drawer",
        proprio=(0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0, 0.39) if proprio is None else proprio,
        timestamp=7.25,
        proprio_convention=OPENPI_BRIDGE_EEF_QUATERNION_WXYZ,
    )


def test_openpi_preserves_four_action_proposal_and_autoeval_bounds_gripper_rule():
    proposal = [
        [-1, 0, 1, -1, 0, 1, 0.2],
        [1, -1, 0, 1, -1, 0, -0.1],
        [0, 0, 0, 0, 0, 0, 0.0],
        [0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 1.0],
    ]
    adapter, model, processor, cv2, _ = adapter_with_fakes(proposal)

    report = adapter.predict_proposal_with_report(observation())

    assert report.native_proposal_horizon == OPENPI_ACTION_HORIZON
    assert report.certified_execute_prefix is None
    assert len(report.proposal) == 4
    assert report.proposal[0] == (10.0, 22.0, 36.0, 40.0, 55.0, 72.0, 1.0)
    assert report.proposal[1] == (12.0, 20.0, 33.0, 48.0, 50.0, 66.0, 0.0)
    assert report.proposal[2][-1] == 0.0  # strict source rule: only > 0 opens
    assert report.source_proprio[-1] == 0.39
    assert report.source_proprio_convention == OPENPI_BRIDGE_EEF_QUATERNION_WXYZ
    assert np.allclose(report.processed_proprio, (-0.6, -0.4, -0.2, -0.5, -1.0, 0.0, -0.22))
    assert len(model.calls) == 1
    assert processor.calls[0][0] == ["open the drawer"]
    assert cv2.calls == [((256, 256, 3), (224, 224), cv2.INTER_LANCZOS4)]
    assert adapter.capability().status is CapabilityStatus.READY_UNQUALIFIED


def test_openpi_requires_one_fresh_image_and_finite_eight_value_quaternion_proprio():
    adapter, model, _, _, _ = adapter_with_fakes(np.zeros((4, 7)))
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(Exception, match="exactly one fresh"):
        adapter.predict_proposal(observation(image_history=(image, image)))
    with pytest.raises(Exception, match="exactly 8 values"):
        adapter.predict_proposal(observation(proprio=(0.0,) * 7))
    with pytest.raises(Exception, match="finite"):
        adapter.predict_proposal(observation(proprio=(0.0,) * 7 + (float("nan"),)))
    with pytest.raises(Exception, match="unit norm"):
        adapter.predict_proposal(observation(proprio=(0.0,) * 8))
    with pytest.raises(Exception, match="unit norm"):
        adapter.predict_proposal(observation(proprio=(0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.39)))
    untagged = PolicyObservation(
        image_history=(image,), prompt="open the drawer", proprio=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.39)
    )
    with pytest.raises(Exception, match="proprio_convention"):
        adapter.predict_proposal(untagged)
    assert model.calls == []


def test_openpi_refuses_to_advertise_an_executed_prefix_and_reset_clears_only_reporting_state():
    adapter, model, _, _, _ = adapter_with_fakes(np.zeros((4, 7)))
    assert adapter.contract.certified_execute_prefix is None
    assert "TemporalEnsembleWrapper" in (adapter.contract.limitation or "")
    adapter.predict_proposal(observation())
    assert adapter.last_report is not None and adapter.last_report.backend_calls == 1
    first_proprio = model.calls[-1]["proprios"].value.copy()
    adapter.reset()
    assert adapter.last_report is None
    # A new call is a new current observation/proprio call, not a replay of a
    # hidden chunk/action history.
    adapter.predict_proposal(observation(proprio=(0.8, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0, 0.39)))
    assert len(model.calls) == 2
    assert not np.array_equal(first_proprio, model.calls[-1]["proprios"].value)
    with pytest.raises(OpenPiUnavailableError, match="execution prefix"):
        adapter.predict_action(observation())
    assert adapter.last_report.backend_calls == 1


def test_openpi_profile_rejects_normal_or_unreviewed_local_configuration_before_model_call():
    adapter = OpenPiZeroPolicyAdapter(profile(action_normalization_type="normal"))
    assert adapter.capability().status is CapabilityStatus.BLOCKED
    assert "bounds" in adapter.capability().reason
    with pytest.raises(OpenPiUnavailableError, match="bounds"):
        adapter.predict_proposal(observation())


def test_openpi_safe_loader_uses_weights_only_and_rejects_non_tensor_state_values(tmp_path):
    checkpoint = tmp_path / "bridge_beta_step19296_2024-12-26_22-30_42.pt"
    checkpoint.write_bytes(b"fixture")
    torch = FakeTorch()

    state = OpenPiZeroPolicyAdapter._safe_state_dict(torch, checkpoint)

    assert torch.loads == [(str(checkpoint), "cpu", True)]
    assert tuple(state) == ("weight",)

    class UnsafePayloadTorch(FakeTorch):
        def load(self, path, *, map_location, weights_only):
            return {"model": {"weight": object()}}

    with pytest.raises(OpenPiUnavailableError, match="tensors only"):
        OpenPiZeroPolicyAdapter._safe_state_dict(UnsafePayloadTorch(), checkpoint)
