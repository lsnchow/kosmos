"""Policy/judge contract tests using fake local runtimes only."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace

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
    policy = OctoSmallV1Policy(ExternalPolicyProfile(profile_id="octo", local_model_path="/unused"))
    assert policy.capability().status is CapabilityStatus.BLOCKED
    with pytest.raises(RuntimeError, match="intentionally unimplemented"):
        policy.predict(PolicyObservation(image_history=(object(), object()), prompt="Open the drawer"))


def test_all_six_external_policy_hooks_stay_blocked_and_never_invent_an_action():
    from plumb.policies.native import (
        MiniVLAPolicy,
        OctoBaseV1Policy,
        OpenPiZeroPolicy,
        SuSIELowLevelPolicy,
        SuSIEPolicy,
    )

    hooks = (
        (OctoSmallV1Policy, 2),
        (OctoBaseV1Policy, 2),
        (MiniVLAPolicy, 1),
        (OpenPiZeroPolicy, 1),
        (SuSIEPolicy, 1),
        (SuSIELowLevelPolicy, 1),
    )
    for factory, history in hooks:
        policy = factory(ExternalPolicyProfile(profile_id="hook", local_model_path="/unused"))
        assert policy.capability().status is CapabilityStatus.BLOCKED
        assert policy.contract.certified_execute_prefix is None
        assert policy.contract.required_observation_history == history
        with pytest.raises(RuntimeError, match="intentionally unimplemented"):
            policy.predict(PolicyObservation(image_history=(object(),) * history, prompt="Open the drawer"))


def test_native_adapter_registry_matches_every_declared_contract():
    from plumb.policies.native import NATIVE_POLICY_CONTRACTS, native_adapter_types

    adapters = native_adapter_types()
    assert set(adapters) == set(NATIVE_POLICY_CONTRACTS)
    for name, adapter_class in adapters.items():
        contract = NATIVE_POLICY_CONTRACTS[name]
        assert adapter_class.base_contract is contract
        assert contract.certified_execute_prefix is None
        assert contract.implementation_status is CapabilityStatus.BLOCKED
        # Every native adapter satisfies the controller's plan/reset/resume surface.
        for attribute in ("plan_control", "reset", "snapshot_state", "restore_state", "propose", "certify"):
            assert callable(getattr(adapter_class, attribute, None)), (name, attribute)
        assert not hasattr(adapter_class, "predict_action"), name


def test_policies_package_exports_the_certification_surface_consistently():
    import plumb.policies as policies

    for name in policies.__all__:
        assert hasattr(policies, name), name
    for name in (
        "PolicyCertification",
        "PolicyCertificationError",
        "PolicyActionNormalizer",
        "PolicyActionNormalizerStatistics",
        "PolicyExecutionMode",
        "GripperStateValue",
        "GripperActionValue",
        "GripperPolarityConvention",
        "UNRESOLVED_BRIDGE_GRIPPER_CONVENTION",
        "mint_certification_payload",
        "OctoSmallV1PolicyAdapter",
        "MiniVLAPolicyAdapter",
        "OpenPiZeroPolicyAdapter",
        "SuSIEPolicyAdapter",
        "SuSIELowLevelPolicyAdapter",
    ):
        assert name in policies.__all__, name
    assert policies.UNRESOLVED_BRIDGE_GRIPPER_CONVENTION.resolved is False
    assert policies.BRIDGE_STATE_GRIPPER_CLOSED == 0.0
    assert policies.BRIDGE_STATE_GRIPPER_OPEN_LIMIT == 0.39
    assert policies.BRIDGE_ACTION_GRIPPER_HIGH == 0.996
    assert policies.OCTO_FORBIDDEN_MODEL_IDS == ("rail-berkeley/octo-small-1.5", "rail-berkeley/octo-base-1.5")
    assert policies.MINIVLA_VQ_LICENSE_STATUS == "absent_cardData_null"
    assert policies.MINIVLA_VQ_REDISTRIBUTION == "prohibited_pending_resolution"
