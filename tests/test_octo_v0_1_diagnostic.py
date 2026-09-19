"""Tests for the pinned Octo 37951 native-v0.1 diagnostic profile.

This suite deliberately targets the separate profile that was GPU-tested.  It
must not change the production adapter's newer AutoEval sampler interface.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from plumb.adapters.contracts import CapabilityStatus, PolicyObservation
from plumb.policies.diagnostics.octo_v0_1 import (
    OCTO_AUTOEVAL_NEWER_API_PROFILE_ID,
    OCTO_BASE_CHECKPOINT_STEP,
    OCTO_BASE_MODEL_ID,
    OCTO_BASE_MODEL_REVISION,
    OCTO_JAX_VERSION,
    OCTO_NATIVE_V0_1_PROFILE_ID,
    OCTO_SMALL_MODEL_REVISION,
    OctoBaseV1Policy,
    OctoSmallV1Policy,
    OctoUnavailableError,
    OctoV1PolicyProfile,
    _OctoRuntime,
)

class FakeOctoRandom:
    def __init__(self):
        self.keys = []

    def PRNGKey(self, seed):
        self.keys.append(seed)
        return ("prng", seed)


class FakeOctoJax:
    __version__ = OCTO_JAX_VERSION

    def __init__(self):
        self.random = FakeOctoRandom()
        self.tree_map_calls = []

    def tree_map(self, fn, tree):
        self.tree_map_calls.append(tree)
        return {key: fn(value) for key, value in tree.items()}


class FakeOctoModel:
    def __init__(self, proposals, *, mean=None, std=None):
        self.proposals = list(proposals)
        self.dataset_statistics = {
            "bridge_dataset": {
                "action": {
                    "mean": list(mean) if mean is not None else [0.0] * 7,
                    "std": list(std) if std is not None else [1.0] * 7,
                }
            }
        }
        self.task_calls = []
        self.sample_calls = []

    def create_tasks(self, *, texts):
        self.task_calls.append(tuple(texts))
        return {"task": tuple(texts)}

    # The exact v0.1 source signature intentionally has no
    # unnormalization_statistics keyword; a newer AutoEval server API does.
    def sample_actions(self, observations, task, pad_mask=None, train=False, argmax=False, sample_shape=(), rng=None, temperature=1.0):
        self.sample_calls.append(
            {
                "observations": observations,
                "task": task,
                "pad_mask": pad_mask,
                "train": train,
                "argmax": argmax,
                "sample_shape": sample_shape,
                "rng": rng,
                "temperature": temperature,
            }
        )
        return np.asarray([self.proposals.pop(0)], dtype=np.float32)


class TrackedNativeRows:
    """JAX-like rows that fail if host conversion happens before source math."""

    def __init__(self, values, events):
        self.values = np.asarray(values, dtype=np.float32)
        self.events = events
        self.shape = self.values.shape

    def __mul__(self, std):
        self.events.append("multiply_on_native")
        return TrackedNativeRows(self.values * np.asarray(std, dtype=np.float32), self.events)

    def __add__(self, mean):
        self.events.append("add_on_native")
        return TrackedNativeRows(self.values + np.asarray(mean, dtype=np.float32), self.events)

    def __array__(self, dtype=None):
        self.events.append("host_collect")
        return np.asarray(self.values, dtype=dtype)


class TrackedNativeBatch:
    def __init__(self, values, events):
        self.rows = TrackedNativeRows(values, events)
        self.events = events
        self.shape = (1,) + self.rows.shape

    def __getitem__(self, index):
        assert index == 0
        self.events.append("remove_batch")
        return self.rows


class TrackedNativeOctoModel(FakeOctoModel):
    def __init__(self, proposal, *, mean, std, events):
        super().__init__([proposal], mean=mean, std=std)
        self.events = events

    def sample_actions(self, observations, task, pad_mask=None, train=False, argmax=False, sample_shape=(), rng=None, temperature=1.0):
        self.sample_calls.append({"observations": observations, "task": task, "rng": rng})
        return TrackedNativeBatch(self.proposals.pop(0), self.events)


def octo_profile(**changes):
    values = {
        "profile_id": "octo-small-fixture",
        "local_model_path": "/does/not/exist",
        "checkpoint_revision": OCTO_SMALL_MODEL_REVISION,
    }
    values.update(changes)
    return OctoV1PolicyProfile(**values)


def octo_runtime_with_history(history_calls):
    jax = FakeOctoJax()

    def stack_and_pad(history, num_obs):
        history_calls.append((tuple(history), num_obs))
        return {
            "image_primary": np.stack([item["image_primary"] for item in history]),
            "pad_mask": np.ones(len(history), dtype=np.float32),
        }

    return _OctoRuntime(
        jax=jax,
        numpy=np,
        model_cls=object,
        stack_and_pad=stack_and_pad,
        jax_version=OCTO_JAX_VERSION,
    )


def test_octo_small_replays_native_v0_1_history_normal_unnormalization_rng_and_temporal_ensemble():
    first = np.asarray([[column + row * 10 for column in range(7)] for row in range(4)], dtype=np.float32)
    second = np.asarray([[100 + column + row * 10 for column in range(7)] for row in range(4)], dtype=np.float32)
    mean = np.asarray([-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0], dtype=np.float32)
    std = np.asarray([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float32)
    model = FakeOctoModel([first, second], mean=mean, std=std)
    history_calls = []
    runtime = octo_runtime_with_history(history_calls)
    policy = OctoSmallV1Policy(
        octo_profile(),
        runtime_factory=lambda: runtime,
        model_factory=lambda profile, loaded_runtime: model,
    )
    image_one = np.zeros((2, 2, 3), dtype=np.uint8)
    image_two = np.ones((2, 2, 3), dtype=np.uint8)

    first_report = policy.predict_with_report(
        PolicyObservation(image_history=(image_one,), prompt="open the drawer", timestamp=1.0)
    )
    second_report = policy.predict_with_report(
        PolicyObservation(image_history=(image_two,), prompt="open the drawer", timestamp=2.0)
    )

    first_physical = first * std + mean
    second_physical = second * std + mean
    assert first_report.proposal == tuple(tuple(float(value) for value in row) for row in first_physical.tolist())
    assert first_report.action == tuple(float(value) for value in first_physical[0])
    # Normal unnormalization includes the gripper (dimension 6), rather than
    # thresholding it into a binary action.
    assert first_report.action[-1] == float(first[0, -1] * std[-1] + mean[-1])
    assert first_report.gripper_transformation.startswith("native-v0.1")
    assert first_report.normalization.startswith("Octo v0.1")
    # The released temporal-ensemble selection order at tick two is
    # proposal-one[1] plus proposal-two[0], with exp_weight=0 yielding equal
    # weights after v0.1 normal unnormalization.
    assert second_report.action == tuple(float(value) for value in (first_physical[1] + second_physical[0]) / 2.0)
    assert second_report.observation_count == 2
    assert len(model.task_calls) == 1
    assert runtime.jax.random.keys == [0, 0]
    assert history_calls[0][1] == 1
    assert history_calls[1][1] == 2
    # Preserve source stack_and_pad's current pad-mask behavior; this adapter
    # does not repair its max(num_obs, horizon) expression.
    np.testing.assert_array_equal(model.sample_calls[0]["observations"]["pad_mask"], np.ones((1, 2), dtype=np.float32))
    assert history_calls[0][0][0] is history_calls[0][0][1]
    np.testing.assert_array_equal(history_calls[1][0][0]["image_primary"], image_one)
    np.testing.assert_array_equal(history_calls[1][0][1]["image_primary"], image_two)
    assert model.sample_calls[0]["observations"]["image_primary"].shape == (1, 2, 2, 2, 3)
    assert model.sample_calls[0]["pad_mask"] is None
    assert model.sample_calls[0]["rng"] == ("prng", 0)


def test_octo_rejects_the_incompatible_autoeval_newer_api_profile_name():
    policy = OctoSmallV1Policy(
        octo_profile(native_api_profile_id=OCTO_AUTOEVAL_NEWER_API_PROFILE_ID),
        runtime_factory=lambda: octo_runtime_with_history([]),
        model_factory=lambda profile, loaded_runtime: FakeOctoModel([np.zeros((4, 7), dtype=np.float32)]),
    )

    result = policy.capability()

    assert result.status is CapabilityStatus.BLOCKED
    assert OCTO_NATIVE_V0_1_PROFILE_ID in result.reason


def test_octo_v0_1_unnormalizes_on_native_array_before_host_collection():
    normalized = np.asarray([[1.0] * 7] * 4, dtype=np.float32)
    mean = np.asarray([10.0] * 7, dtype=np.float32)
    std = np.asarray([0.25] * 7, dtype=np.float32)
    events = []
    model = TrackedNativeOctoModel(normalized, mean=mean, std=std, events=events)
    policy = OctoSmallV1Policy(
        octo_profile(),
        runtime_factory=lambda: octo_runtime_with_history([]),
        model_factory=lambda profile, loaded_runtime: model,
    )

    report = policy.predict_with_report(
        PolicyObservation(image_history=(np.zeros((2, 2, 3), dtype=np.uint8),), prompt="open the drawer")
    )

    assert events[:4] == ["remove_batch", "multiply_on_native", "add_on_native", "host_collect"]
    assert report.action == (10.25,) * 7


def test_octo_legacy_source_fixture_rejects_the_newer_unnormalization_keyword():
    model = FakeOctoModel([np.zeros((4, 7), dtype=np.float32)])

    with pytest.raises(TypeError, match="unnormalization_statistics"):
        model.sample_actions({}, {}, unnormalization_statistics={"mean": [0.0] * 7, "std": [1.0] * 7})


def test_octo_rejects_changing_the_cached_autoeval_instruction_until_reset():
    proposal = np.zeros((4, 7), dtype=np.float32)
    model = FakeOctoModel([proposal, proposal])
    history_calls = []
    runtime = octo_runtime_with_history(history_calls)
    policy = OctoSmallV1Policy(
        octo_profile(),
        runtime_factory=lambda: runtime,
        model_factory=lambda profile, loaded_runtime: model,
    )

    policy.predict(PolicyObservation(image_history=(np.zeros((2, 2, 3), dtype=np.uint8),), prompt="open the drawer"))
    with pytest.raises(Exception, match="call reset"):
        policy.predict(PolicyObservation(image_history=(np.zeros((2, 2, 3), dtype=np.uint8),), prompt="close the drawer"))
    policy.reset()
    policy.predict(PolicyObservation(image_history=(np.zeros((2, 2, 3), dtype=np.uint8),), prompt="close the drawer"))
    assert model.task_calls == [("open the drawer",), ("close the drawer",)]


def test_octo_base_is_separately_pinned_and_keeps_a_diagnostic_contract():
    proposal = np.zeros((4, 7), dtype=np.float32)
    model = FakeOctoModel([proposal])
    history_calls = []
    runtime = octo_runtime_with_history(history_calls)
    policy = OctoBaseV1Policy(
        octo_profile(
            profile_id="octo-base-fixture",
            model_id=OCTO_BASE_MODEL_ID,
            checkpoint_revision=OCTO_BASE_MODEL_REVISION,
            checkpoint_step=OCTO_BASE_CHECKPOINT_STEP,
        ),
        runtime_factory=lambda: runtime,
        model_factory=lambda profile, loaded_runtime: model,
    )

    assert policy.capability().status is CapabilityStatus.READY_UNQUALIFIED
    assert "diagnostic" in policy.contract.name.lower()
    assert policy.predict(PolicyObservation(image_history=(np.zeros((2, 2, 3), dtype=np.uint8),), prompt="open the drawer")) == (0.0,) * 7


def test_octo_records_an_explicit_compatibility_profile_for_a_nonreleased_jax_runtime():
    proposal = np.zeros((4, 7), dtype=np.float32)
    model = FakeOctoModel([proposal])
    history_calls = []
    runtime = replace(octo_runtime_with_history(history_calls), jax_version="0.4.31+computecanada")
    policy = OctoSmallV1Policy(
        octo_profile(
            jax_version="0.4.31",
            compatibility_profile_id="octo-jax-0.4.31-site-a",
            runtime_lock_id="octo-runtime-lock-a",
            runtime_lock_sha256="f" * 64,
        ),
        runtime_factory=lambda: runtime,
        model_factory=lambda profile, loaded_runtime: model,
    )

    result = policy.capability()
    assert result.status is CapabilityStatus.READY_UNQUALIFIED
    assert result.details["compatibility_profile_id"] == "octo-jax-0.4.31-site-a"
    assert result.details["requires_fixture_requalification"] is True
    assert policy.predict(PolicyObservation(image_history=(np.zeros((2, 2, 3), dtype=np.uint8),), prompt="open the drawer")) == (0.0,) * 7


def test_octo_requires_a_real_immutable_binding_before_it_can_consider_loading():
    history_calls = []
    runtime = octo_runtime_with_history(history_calls)

    class LocalModelClass:
        @staticmethod
        def load_pretrained(*args, **kwargs):
            raise AssertionError("missing immutable files must stop before a loader call")

    runtime = replace(runtime, model_cls=LocalModelClass)
    policy = OctoSmallV1Policy(
        octo_profile(local_model_path="/definitely/missing"),
        runtime_factory=lambda: runtime,
    )

    result = policy.capability()

    assert result.status is CapabilityStatus.BLOCKED
    assert result.details["local_files_only"] is True
    with pytest.raises(OctoUnavailableError, match="asset_manifest_path"):
        policy.predict(PolicyObservation(image_history=(object(),), prompt="open the drawer"))
