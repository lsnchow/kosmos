"""Adversarial end-to-end checks for the real qualification boundary.

These records are deliberately small, but retain the full 6 x 5 x 50 scope.
They prove that qualification needs bound assets and measured profiles, rather
than demonstrating that any claimed human identity is authentic.
"""
from __future__ import annotations

import copy
import hashlib

from plumb.adapters.contracts import CapabilityStatus, FeedbackMode
from plumb.gates import EXACT_TASK_PROMPTS, GateLedger, GateRecord, GateStatus, QualificationValidator, canonical_json_sha256
from plumb.protocol import AssetLock, AssetRecord, canonical_sha256
from plumb.reference import POLICY_IDS, TASK_IDS


def digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _asset(asset_id: str) -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id,
        repo_id="qualification/%s" % asset_id,
        repo_type="model",
        revision="rev-%s" % asset_id,
        sha256=digest("asset:" + asset_id),
        byte_length=1,
        retrieved_at="2026-09-19T00:00:00Z",
        license="MIT",
        access_status="resolved",
        loader_revision="loader-%s" % asset_id,
        container_digest=digest("container:" + asset_id),
        compatibility_profile_id="profile-%s" % asset_id,
    )


def _asset_lock() -> tuple[AssetLock, dict]:
    lock = AssetLock()
    for asset_id in ("world-model", "world-code", "judge-model", "judge-code", "scenario-source"):
        lock.add(_asset(asset_id))
    policies = {}
    for policy in POLICY_IDS:
        bound = ("policy-%s" % policy, "policy-code-%s" % policy)
        policies[policy] = list(bound)
        for asset_id in bound:
            lock.add(_asset(asset_id))
    return lock, {
        "world_model": ["world-model", "world-code"],
        "policies": policies,
        "judge": ["judge-model", "judge-code"],
        "scenario": ["scenario-source"],
    }


def _scenario() -> dict:
    tasks = {}
    for task in TASK_IDS:
        starts = []
        for index in range(50):
            token = "%s-%02d" % (task, index)
            starts.append(
                {
                    "start_id": "start-" + token,
                    "start_lineage_id": "lineage-" + token,
                    "source_dataset": "reviewed-real-source",
                    "source_revision": "source-revision",
                    "episode_id": "episode-" + token,
                    "frame_id": "0",
                    "image_hash": digest("image:" + token),
                    "state_hash": digest("state:" + token),
                    "instruction": EXACT_TASK_PROMPTS[task],
                    "state_convention": "bridge8",
                    "camera": "overhead",
                    "scene": "reviewed-scene",
                    "initial_state_stratum": "stratum-a",
                    "goal_reference_hashes": [digest("goal:" + token)],
                }
            )
        tasks[task] = {"starts": starts}
    result = {"schema_version": 1, "manifest_id": "qualification-panel-v1", "provenance_kind": "real_robot", "tasks": tasks}
    result["sha256"] = canonical_json_sha256(result)
    return result


def _profile(prefix: str, model_asset: str, code_asset: str, *, gpu: bool = False) -> dict:
    result = {
        "profile_id": prefix + "-profile",
        "model_asset_id": model_asset,
        "code_asset_id": code_asset,
        "model_asset_sha256": digest("asset:" + model_asset),
        "code_asset_sha256": digest("asset:" + code_asset),
        "model_revision": "model-revision-" + prefix,
        "code_revision": "code-revision-" + prefix,
        "normalizer_revision": "normalizer-revision-" + prefix,
        "container_digest": digest("runtime:" + prefix),
    }
    if gpu:
        result["gpu_profile"] = {
            "gpu_model": "H100",
            "device_uuid": "GPU-qualification-fixture",
            "driver_version": "570.0",
            "runtime_version": "cuda-12.8",
            "measured_at": "2026-09-19T00:01:00Z",
            "evidence_uri": "https://evidence.example/gpu-profile.json",
            "peak_memory_bytes": 123456,
        }
    return result


def _protocol(scenario: dict, lock: AssetLock, assets: dict) -> dict:
    protocol = {
        "protocol_id": "qualification-v1",
        "status": "frozen",
        "scenario_manifest_hash": scenario["sha256"],
        "asset_lock_hash": canonical_sha256(lock.to_mapping()),
        "qualification_assets": assets,
        "policy": {"kind": "released-wrappers"},
        "world_model": {"kind": "pinned-world"},
        "judge": {"kind": "pinned-judge"},
        "thresholds": {"frozen": True},
        "seeds": {"root": 1},
        "remote_record": "https://evidence.example/preregistration",
        "matrix": {
            "policies": list(POLICY_IDS),
            "tasks": list(TASK_IDS),
            "starts_per_task": 50,
            "total_planned_episodes": 1500,
        },
        "tolerances": {
            "gate_d": {
                "min_binary_kappa": 0.6,
                "min_weighted_progress_kappa": 0.6,
                "max_leniency_offset": 0.15,
                "min_consensus_coverage": 0.8,
            }
        },
        "judge_sampling": {"sample_count": 5},
    }
    protocol["sha256"] = canonical_json_sha256(protocol)
    return protocol


def _ledger(protocol: dict, scenario: dict) -> GateLedger:
    protocol_hash = protocol["sha256"]
    common = {
        "status": GateStatus.PASS,
        "protocol_hash": protocol_hash,
        "fixture_ids": ("real-fixture",),
        "evidence_uris": ("https://evidence.example/gate.json",),
        "thresholds": {"frozen": True},
        "evidence_kind": "real_cluster_execution",
    }
    backend = _profile("world", "world-model", "world-code", gpu=True)
    ledger = GateLedger.new(protocol_hash)
    ledger.record(
        GateRecord(
            gate_id="A",
            backend_revisions={"profile": backend["profile_id"]},
            measurements={
                "backend_calls": 1,
                "wall_seconds": 0.5,
                "gpu_peak_memory_bytes": 123456,
                "backend_profile": backend,
            },
            **common,
        )
    )
    prefixes = {policy: (1 if policy == "OpenVLA" else 4) for policy in POLICY_IDS}
    policy_profiles = {}
    policy_revisions = {}
    for policy in POLICY_IDS:
        profile = _profile("policy-" + policy, "policy-" + policy, "policy-code-" + policy)
        profile["executed_prefix_ticks"] = prefixes[policy]
        profile["certified_terminal_prefixes"] = {
            task: (1 if policy == "OpenVLA" else (2 if task in {"open_drawer", "close_drawer"} else 4))
            for task in TASK_IDS
        }
        policy_profiles[policy] = profile
        policy_revisions[policy] = {"asset": "policy-" + policy, "profile_id": profile["profile_id"]}
    ledger.record(
        GateRecord(
            gate_id="B",
            policy_revisions=policy_revisions,
            measurements={
                "interventions": {arm: {"status": "pass", "evidence_uri": "https://evidence.example/%s" % arm} for arm in ("original", "zero", "temporally_permuted", "sign_reversed", "cross_episode")},
                "feedback_mode": "native_feedback",
                "feedback_certification": {
                    "feedback_mode": "native_feedback",
                    "world_profile_id": backend["profile_id"],
                    "terminal_padding": False,
                    "evidence_hash": digest("feedback"),
                    "evidence_uri": "https://evidence.example/feedback.json",
                    "policy_executed_prefixes": prefixes,
                    "policy_profiles": policy_profiles,
                },
                "suffix_causality": {
                    "status": "pass",
                    "first_action_fixed_against_baseline": True,
                    "future_suffix_changed_first_generated_frame": False,
                    "evidence_hash": digest("suffix"),
                    "evidence_uri": "https://evidence.example/suffix.json",
                },
            },
            **common,
        )
    )
    start_ids = tuple("%s/%s" % (task, start["start_id"]) for task, group in scenario["tasks"].items() for start in group["starts"])
    ledger.record(
        GateRecord(
            gate_id="C",
            start_ids=start_ids,
            measurements={
                "starts_per_task": {task: 50 for task in TASK_IDS},
                "comparability": {task: {"status": "pass", "evidence_uri": "https://evidence.example/%s-panel" % task} for task in TASK_IDS},
            },
            **common,
        )
    )
    ledger.record(
        GateRecord(
            gate_id="D",
            judge_revisions={"profile": "judge-profile"},
            measurements={
                "binary_kappa": 0.7,
                "weighted_progress_kappa": 0.7,
                "leniency_offset": 0.1,
                "consensus_coverage": 0.9,
                "calibration_class": "human",
                "human_annotator_count": 2,
                "has_two_blinded_humans": True,
                "held_out_frozen_before_evaluation": True,
                "gate_d_decision_status": "pass",
                "thresholds_satisfied": True,
                "human_annotation_evidence": {
                    "annotator_ids": ["human:a", "human:b"],
                    "heldout_clip_count": 50,
                    "double_labeled_clip_count": 50,
                    "manifest_hash": digest("human-annotation-manifest"),
                    "evidence_uri": "https://evidence.example/human-annotations.json",
                },
                "judge_profile": {**_profile("judge", "judge-model", "judge-code"), "runtime_lock_hash": digest("judge-runtime")},
            },
            **common,
        )
    )
    return ledger


def _decision(ledger: GateLedger, scenario: dict, protocol: dict, lock: AssetLock):
    return QualificationValidator.validate(
        ledger,
        scenario,
        protocol,
        task="close_drawer",
        feedback_mode=FeedbackMode.NATIVE_FEEDBACK,
        asset_lock=lock,
    )


def test_structurally_bound_full_scope_evidence_can_qualify_without_authenticating_people():
    lock, assets = _asset_lock()
    scenario = _scenario()
    protocol = _protocol(scenario, lock, assets)
    decision = _decision(_ledger(protocol, scenario), scenario, protocol, lock)
    assert decision.capability.status is CapabilityStatus.QUALIFIED
    # The validator checks an auditable two-person declaration and artifacts;
    # it intentionally does not claim those fields prove a real-world identity.
    assert decision.capability.source_verified is True


def test_model_annotation_cannot_be_relabelled_as_human_qualification():
    lock, assets = _asset_lock()
    scenario = _scenario()
    protocol = _protocol(scenario, lock, assets)
    ledger = _ledger(protocol, scenario)
    record = ledger.records["D"]
    ledger.record(GateRecord.from_mapping({**record.to_mapping(), "measurements": {**record.measurements, "calibration_class": "model"}}))
    decision = _decision(ledger, scenario, protocol, lock)
    assert decision.capability.status is CapabilityStatus.BLOCKED
    assert any("model/external labels" in error for error in decision.errors)


def test_padding_and_suffix_sensitivity_cannot_be_promoted_to_native_feedback():
    lock, assets = _asset_lock()
    scenario = _scenario()
    protocol = _protocol(scenario, lock, assets)
    ledger = _ledger(protocol, scenario)
    record = ledger.records["B"]
    measurements = copy.deepcopy(record.measurements)
    measurements["feedback_certification"]["terminal_padding"] = True
    measurements["suffix_causality"]["future_suffix_changed_first_generated_frame"] = True
    ledger.record(GateRecord.from_mapping({**record.to_mapping(), "measurements": measurements}))
    decision = _decision(ledger, scenario, protocol, lock)
    assert decision.capability.status is CapabilityStatus.BLOCKED
    assert any("terminal padding" in error for error in decision.errors)
    assert any("suffix sensitivity" in error for error in decision.errors)


def test_unverified_runtime_asset_blocks_even_when_every_gate_status_is_pass():
    lock, assets = _asset_lock()
    scenario = _scenario()
    protocol = _protocol(scenario, lock, assets)
    broken = AssetLock.from_mapping(lock.to_mapping())
    original = broken.assets["world-code"]
    broken.add(
        AssetRecord(
            **{**original.__dict__, "container_digest": None}
        )
    )
    # Bind the broken lock honestly.  This isolates the row-level execution
    # check from the separately fail-closed protocol/lock hash check.
    protocol = {**protocol, "asset_lock_hash": canonical_sha256(broken.to_mapping())}
    protocol["sha256"] = canonical_json_sha256({key: value for key, value in protocol.items() if key != "sha256"})
    ledger = _ledger(protocol, scenario)
    decision = _decision(ledger, scenario, protocol, broken)
    assert decision.capability.status is CapabilityStatus.BLOCKED
    assert any("world-code is not execution-verified" in error for error in decision.errors)


def test_a_1499_episode_rehearsal_is_not_a_full_burst_gate_pass():
    record = GateRecord(
        gate_id="F",
        status=GateStatus.PASS,
        protocol_hash=digest("protocol"),
        evidence_uris=("https://evidence.example/gate-f.json",),
        thresholds={"max_seconds": 60},
        evidence_kind="real_platform_execution",
        measurements={
            "rehearsals": [
                {
                    "run_id": "short-%d" % index,
                    "planned_episodes": 1499,
                    "generated_episodes": 1499,
                    "terminal_service_failures": 0,
                    "elapsed_seconds": 59.0,
                    "total_demonstration_usd": 11.0,
                    "reused_cached_outputs": False,
                }
                for index in range(3)
            ],
            "displayed_capacity": {"replicas": 10},
            "cost_basis": {"source": "platform export"},
        },
    )
    assert any("1500" in error for error in record.pass_evidence_errors())
