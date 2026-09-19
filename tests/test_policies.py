"""Policy/judge contract tests using fake local runtimes only."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest

from plumb.adapters.contracts import CapabilityStatus, PolicyObservation
from plumb.policies import (
    ExternalPolicyProfile,
    JudgeInputError,
    JudgeInputProvenance,
    JudgeRequest,
    JudgeSchemaError,
    OpenVLAPolicyAdapter,
    OpenVLAPolicyProfile,
    QwenJudgeProfile,
    QwenRubricJudge,
    ReferenceImage,
    parse_rubric_json,
)
from plumb.policies.tasks import BENCHMARK_TASK_REGISTRY, TASK_REGISTRY_HASH
from plumb.policies.native import OctoSmallV1Policy
from plumb.policies.octo import (
    OCTO_BASE_MODEL_ID,
    OCTO_BASE_MODEL_REVISION,
    OCTO_BASE_CHECKPOINT_STEP,
    OCTO_AUTOEVAL_NEWER_API_PROFILE_ID,
    OCTO_JAX_VERSION,
    OCTO_NATIVE_V0_1_PROFILE_ID,
    OCTO_SMALL_MODEL_REVISION,
    OctoBaseV1Policy,
    OctoUnavailableError,
    OctoV1PolicyProfile,
    _OctoRuntime,
)
from plumb.policies.openvla import OPENVLA_TRANSFORMERS_VERSION, OpenVLAUnavailableError, _OpenVLARuntime
from plumb.policies.judge import _QwenRuntime


REVISION = "a" * 40


class FakeCuda:
    def is_available(self):
        return False


class FakeDefaultGenerator:
    def __init__(self):
        self.seed = "initial"
        self.manual_seeds = []

    def manual_seed(self, seed):
        self.seed = seed
        self.manual_seeds.append(seed)
        return self


class FakeForkRng:
    def __init__(self, random, devices, enabled):
        self.random = random
        self.devices = devices
        self.enabled = enabled
        self.before = None

    def __enter__(self):
        self.before = self.random.default_generator.seed
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.random.default_generator.seed = self.before
        return False


class FakeRandom:
    def __init__(self):
        self.default_generator = FakeDefaultGenerator()
        self.fork_calls = []

    def fork_rng(self, *, devices, enabled):
        self.fork_calls.append((tuple(devices), enabled))
        return FakeForkRng(self, devices, enabled)

    def reset(self):
        self.default_generator.seed = "initial"
        self.default_generator.manual_seeds = []
        self.fork_calls = []


class FakeTorch:
    bfloat16 = "bf16"
    float16 = "f16"
    float32 = "f32"
    cuda = FakeCuda()
    random = FakeRandom()

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeOpenVLAModel:
    def __init__(self, action):
        self.action = action
        self.calls = []

    def eval(self):
        return self

    def predict_action(self, **kwargs):
        self.calls.append(kwargs)
        return self.action


class FakeImage:
    def __init__(self, source):
        self.source = source
        self.converted_to = None

    def convert(self, mode):
        self.converted_to = mode
        return self


class FakeOpenVLAProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"input_ids": [[42]], "pixel_values": "fixture-pixels"}


def openvla_profile(**changes):
    profile = {
        "profile_id": "openvla-fixture",
        "local_model_path": "/does/not/exist",
        "checkpoint_revision": REVISION,
        "remote_code_revision": REVISION,
        "reviewed_remote_code_revision": REVISION,
        "allow_trust_remote_code": True,
    }
    profile.update(changes)
    return OpenVLAPolicyProfile(**profile)


def fake_openvla_runtime():
    return _OpenVLARuntime(
        torch=FakeTorch,
        model_cls=object,
        processor_cls=object,
        image_fromarray=FakeImage,
        transformers_version=OPENVLA_TRANSFORMERS_VERSION,
    )


def test_openvla_invokes_native_predict_action_once_with_fresh_image_and_bridge_orig():
    model = FakeOpenVLAModel([0, 1, 2, 3, 4, 5, 6])
    processor = FakeOpenVLAProcessor()
    adapter = OpenVLAPolicyAdapter(
        openvla_profile(),
        runtime_factory=fake_openvla_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: processor,
    )
    image = object()

    report = adapter.predict_with_report(
        PolicyObservation(image_history=(image,), prompt="Close the drawer", timestamp=12.5)
    )

    assert report.action == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert report.unnorm_key == "bridge_orig"
    assert report.source_image_timestamp == 12.5
    assert len(model.calls) == 1
    assert len(processor.calls) == 1
    processor_args, processor_kwargs = processor.calls[0]
    assert processor_args[0] == "In: What action should the robot take to close the drawer?\nOut:"
    assert processor_args[1].source is image
    assert processor_args[1].converted_to == "RGB"
    assert processor_kwargs == {"return_tensors": "pt"}
    assert model.calls[0] == {
        "input_ids": [[42]],
        "pixel_values": "fixture-pixels",
        "unnorm_key": "bridge_orig",
        "do_sample": False,
    }
    assert adapter.capability().status is CapabilityStatus.READY_UNQUALIFIED


def test_openvla_refuses_stale_history_proprio_and_non_single_action():
    model = FakeOpenVLAModel([0] * 7)
    processor = FakeOpenVLAProcessor()
    adapter = OpenVLAPolicyAdapter(
        openvla_profile(),
        runtime_factory=fake_openvla_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: processor,
    )
    with pytest.raises(ValueError, match="exactly one fresh"):
        adapter.predict_action(PolicyObservation(image_history=(object(), object()), prompt="Open the drawer"))
    with pytest.raises(ValueError, match="must not receive proprioception"):
        adapter.predict_action(PolicyObservation(image_history=(object(),), prompt="Open the drawer", proprio=(0.0,)))

    too_long = FakeOpenVLAModel([0] * 14)
    bad_adapter = OpenVLAPolicyAdapter(
        openvla_profile(),
        runtime_factory=fake_openvla_runtime,
        model_factory=lambda profile, runtime: too_long,
        processor_factory=lambda profile, runtime: FakeOpenVLAProcessor(),
    )
    with pytest.raises(ValueError, match="exactly one 7-D action"):
        bad_adapter.predict_action(PolicyObservation(image_history=(object(),), prompt="Open the drawer"))
    assert len(too_long.calls) == 1


def test_openvla_never_trusts_remote_code_without_review_and_pinned_transformers_four():
    model = FakeOpenVLAModel([0] * 7)
    adapter = OpenVLAPolicyAdapter(
        openvla_profile(allow_trust_remote_code=False),
        runtime_factory=fake_openvla_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: FakeOpenVLAProcessor(),
    )
    assert adapter.capability().status is CapabilityStatus.BLOCKED
    with pytest.raises(OpenVLAUnavailableError, match="allow_trust_remote_code"):
        adapter.predict_action(PolicyObservation(image_history=(object(),), prompt="Open the drawer"))
    assert model.calls == []

    wrong_transformers = OpenVLAPolicyAdapter(
        openvla_profile(transformers_version="5.0.0"),
        runtime_factory=fake_openvla_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: FakeOpenVLAProcessor(),
    )
    with pytest.raises(OpenVLAUnavailableError, match="Transformers"):
        wrong_transformers.predict_action(PolicyObservation(image_history=(object(),), prompt="Open the drawer"))


class FakeQwenModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


class UnsupportedGenerationModel(FakeQwenModel):
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        raise ValueError("The following model_kwargs are not used by the model: ['generator']")


class FakeQwenProcessor:
    def __init__(self):
        self.messages = []
        self.encode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return "fixture-prompt"

    def __call__(self, **kwargs):
        self.encode_calls.append(kwargs)
        return {"input_ids": [[1, 2, 3]], "pixel_values": "pixels"}

    def batch_decode(self, generated, **kwargs):
        return [generated]


def sample_json(completion="met", progress=5):
    return json.dumps(
        {
            "integrity": "intact",
            "collision": "none_visible",
            "progress": progress,
            "completion_evidence": completion,
            "evidence_frame_indices": [0, 15],
            "observable_reasons": "The final frame visibly satisfies the stated completion condition.",
        }
    )


def qwen_profile(**changes):
    profile = {
        "profile_id": "qwen-fixture",
        "local_model_path": "/does/not/exist",
        "model_revision": REVISION,
        "processor_revision": REVISION,
    }
    profile.update(changes)
    return QwenJudgeProfile(**profile)


def qwen_runtime():
    return _QwenRuntime(torch=FakeTorch, model_cls=object, processor_cls=object, transformers_version="4.49.0")


def judge_request(**changes):
    request = {
        "frames": tuple("frame-%02d" % index for index in range(16)),
        "frame_timestamps": tuple(float(index) / 5.0 for index in range(16)),
        "task_instruction": "Close the drawer",
        "task_rubric": "Drawer milestones: completion requires the drawer visibly closed at the final frame.",
        "diagnostic_mode": True,
        "reference_images": (ReferenceImage("goal-image", "artifact://goals/close.png", "b" * 64),),
    }
    request.update(changes)
    return JudgeRequest(**request)


def test_qwen_judge_passes_only_allowed_blinded_inputs_and_returns_quorum_outcome():
    model = FakeQwenModel([sample_json()] * 5)
    processor = FakeQwenProcessor()
    FakeTorch.random.reset()
    judge = QwenRubricJudge(
        qwen_profile(),
        runtime_factory=qwen_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: processor,
    )

    report = judge.evaluate(judge_request(), seeds=[101, 102, 103, 104, 105])

    assert report.judge_status == "evaluable"
    assert report.binary_success is True
    assert report.progress == 5.0
    assert report.agreeing_samples == 5
    assert len(model.calls) == 5
    assert all(call["do_sample"] is True for call in model.calls)
    assert all(call["temperature"] == 0.7 and call["top_p"] == 1.0 for call in model.calls)
    assert all(call["max_new_tokens"] == 512 for call in model.calls)
    assert all("generator" not in call for call in model.calls)
    assert FakeTorch.random.default_generator.manual_seeds == [101, 102, 103, 104, 105]
    assert FakeTorch.random.default_generator.seed == "initial"
    assert FakeTorch.random.fork_calls == [((), True)] * 5
    assert all(call["videos"] == [["frame-%02d" % index for index in range(16)]] for call in processor.encode_calls)
    assert all(call["images"] == ["goal-image"] for call in processor.encode_calls)
    serialised_messages = "\n".join(repr(messages) for messages in processor.messages).lower()
    assert "policy" not in serialised_messages
    assert "action" not in serialised_messages
    assert "%" not in serialised_messages


def test_qwen_judge_retries_invalid_json_once_then_preserves_unknown_on_exhaustion():
    # Five samples × initial attempt plus one bounded retry.  All failures
    # remain visible and lead to null outcome rather than a forced label.
    model = FakeQwenModel(["not-json"] * 10)
    processor = FakeQwenProcessor()
    judge = QwenRubricJudge(
        qwen_profile(),
        runtime_factory=qwen_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: processor,
    )

    report = judge.evaluate(judge_request(), seeds=[1, 2, 3, 4, 5])

    assert report.judge_status == "unknown"
    assert report.binary_success is None
    assert report.progress is None
    assert report.missing_reason == "judge_sample_failure"
    assert [len(sample.attempts) for sample in report.sample_reports] == [2, 2, 2, 2, 2]
    assert all("schema_error" in sample.attempts[-1].failure_reason for sample in report.sample_reports)


def test_qwen_accepts_only_anchored_json_fence_and_records_transport_normalization():
    fenced = "```json\n" + sample_json() + "\n```"
    model = FakeQwenModel([fenced] * 5)
    processor = FakeQwenProcessor()
    judge = QwenRubricJudge(
        qwen_profile(),
        runtime_factory=qwen_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: processor,
    )

    report = judge.evaluate(judge_request(), seeds=[31, 32, 33, 34, 35])

    assert report.binary_success is True
    assert all(sample.attempts[0].transport_normalization == "json_code_fence_v1" for sample in report.sample_reports)
    system = processor.messages[0][0]["content"].lower()
    assert "raw json object" in system
    assert "no markdown code fence" in system
    assert "observable_reasons is one nonempty json string" in system
    with pytest.raises(JudgeSchemaError, match="observable_reasons"):
        payload = json.loads(sample_json())
        payload["observable_reasons"] = ["array is rejected"]
        parse_rubric_json(json.dumps(payload))
    with pytest.raises(JudgeSchemaError, match="anchored"):
        parse_rubric_json("prefix ```json\n" + sample_json() + "\n```")


def test_qwen_schema_retry_records_deterministic_format_only_prompt_variant():
    model = FakeQwenModel(sum((["not-json", sample_json()] for _ in range(5)), []))
    processor = FakeQwenProcessor()
    judge = QwenRubricJudge(
        qwen_profile(),
        runtime_factory=qwen_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: processor,
    )

    report = judge.evaluate(judge_request(), seeds=[41, 42, 43, 44, 45])

    assert report.binary_success is True
    for sample in report.sample_reports:
        assert [attempt.prompt_variant for attempt in sample.attempts] == ["base_v2", "format_retry_v1"]
        assert sample.attempts[0].prompt_source_hash != sample.attempts[1].prompt_source_hash
    system_messages = [messages[0]["content"] for messages in processor.messages]
    assert all("FORMAT_ONLY retry reminder" not in message for message in system_messages[::2])
    assert all("FORMAT_ONLY retry reminder" in message for message in system_messages[1::2])


def test_qwen_unsupported_generation_configuration_fails_fast_without_blind_retries():
    model = UnsupportedGenerationModel([])
    processor = FakeQwenProcessor()
    judge = QwenRubricJudge(
        qwen_profile(),
        runtime_factory=qwen_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: processor,
    )

    report = judge.evaluate(judge_request(), seeds=[21, 22, 23, 24, 25])

    assert report.judge_status == "unknown"
    assert report.binary_success is None
    assert len(model.calls) == 5
    assert [len(sample.attempts) for sample in report.sample_reports] == [1, 1, 1, 1, 1]
    assert all(
        attempt.failure_reason.startswith("unsupported_generation_configuration:")
        for sample in report.sample_reports
        for attempt in sample.attempts
    )


def test_qwen_judge_rejects_percentage_leakage_and_wrong_frame_count_before_model_use():
    model = FakeQwenModel([sample_json()] * 5)
    processor = FakeQwenProcessor()
    judge = QwenRubricJudge(
        qwen_profile(),
        runtime_factory=qwen_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: processor,
    )
    with pytest.raises(JudgeInputError, match="percentages"):
        judge.evaluate(judge_request(task_rubric="Human success was 92%."), seeds=[1, 2, 3, 4, 5])
    with pytest.raises(JudgeInputError, match="exactly 16"):
        judge.evaluate(judge_request(frames=("too-short",)), seeds=[1, 2, 3, 4, 5])
    assert model.calls == []


def test_primary_judge_rejects_free_text_and_uses_only_frozen_task_registry():
    with pytest.raises(JudgeInputError, match="canonical task_id only"):
        JudgeRequest(
            frames=(b"frame",) * 16,
            frame_timestamps=tuple(float(index) for index in range(16)),
            task_instruction="OpenVLA successfully used an action sequence.",
            task_rubric="Output met because the policy was successful.",
            reference_images=(ReferenceImage(b"goal", "artifact://goal", "c" * 64),),
        ).validate()

    task = BENCHMARK_TASK_REGISTRY.get("fold_cloth")
    assert task.instruction == "fold the cloth from top right to bottom left"
    assert TASK_REGISTRY_HASH.startswith("sha256:")


def test_diagnostic_text_is_retained_but_explicitly_gate_d_unavailable():
    model = FakeQwenModel([sample_json()] * 5)
    processor = FakeQwenProcessor()
    judge = QwenRubricJudge(
        qwen_profile(),
        runtime_factory=qwen_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: processor,
    )
    request = JudgeRequest(
        frames=(b"frame",) * 16,
        frame_timestamps=tuple(float(index) for index in range(16)),
        diagnostic_mode=True,
        task_instruction="Vendor pot fixture",
        task_rubric="Visible final state only.",
        reference_images=(ReferenceImage(b"scene", "artifact://scene", "d" * 64),),
    )

    report = judge.evaluate(request, seeds=[1, 2, 3, 4, 5])

    assert report.provenance["mode"] == "diagnostic"
    assert report.provenance["trust"]["test_mode"] is True
    assert report.provenance["trust"]["gate_d_eligible"] is False
    assert report.provenance["task_id"] is None
    with pytest.raises(JudgeInputError, match="Gate-D unavailable"):
        report.gate_d_payload()


def test_primary_report_binds_evidence_and_rejects_forged_aggregation_summary():
    model = FakeQwenModel([sample_json()] * 5)
    processor = FakeQwenProcessor()
    profile = qwen_profile(
        asset_manifest_id="assets-qwen-v1",
        asset_manifest_sha256="e" * 64,
        runtime_lock_id="runtime-qwen-v1",
        runtime_lock_sha256="f" * 64,
    )
    judge = QwenRubricJudge(
        profile,
        runtime_factory=qwen_runtime,
        model_factory=lambda profile, runtime: model,
        processor_factory=lambda profile, runtime: processor,
    )
    request = JudgeRequest(
        frames=tuple(b"frame-%02d" % index for index in range(16)),
        frame_timestamps=tuple(float(index) / 5.0 for index in range(16)),
        task_id="close_drawer",
        reference_images=(ReferenceImage(b"goal", "artifact://goal", "a" * 64),),
        provenance=JudgeInputProvenance(
            clip_id="clip-001",
            video_sha256="b" * 64,
            protocol_id="protocol-001",
            calibration_manifest_hash="c" * 64,
        ),
    )

    report = judge.evaluate(request, seeds=[11, 12, 13, 14, 15])

    provenance = report.provenance
    assert provenance["trust"]["gate_d_eligible"] is True
    assert provenance["clip_id"] == "clip-001"
    assert provenance["evidence_hashes"]["video_hash"] == "sha256:" + "b" * 64
    assert len(provenance["evidence_hashes"]["frame_pixel_hashes"]) == 16
    assert provenance["task_registry_hash"] == TASK_REGISTRY_HASH
    assert provenance["model"]["asset_manifest_id"] == "assets-qwen-v1"
    assert len(report.as_dict()["raw_judge_samples"]) == 5
    assert report.gate_d_payload()["binary_success"] is True

    forged = replace(report, binary_success=False)
    with pytest.raises(JudgeSchemaError, match="does not match"):
        forged.gate_d_payload()


def test_other_policy_contracts_are_explicitly_blocked_and_do_not_make_up_actions():
    policy = OctoSmallV1Policy(
        OctoV1PolicyProfile(profile_id="octo", local_model_path="/unused", checkpoint_revision="0" * 40)
    )
    assert policy.capability().status is CapabilityStatus.BLOCKED
    with pytest.raises(OctoUnavailableError, match="pinned"):
        policy.predict(PolicyObservation(image_history=(object(),), prompt="Open the drawer"))


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
