"""Tests for frozen full-study planning and fail-closed burst evidence checks."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from plumb.adapters.contracts import FeedbackMode
from plumb.gates import GateLedger, GateRecord, GateStatus
from plumb.protocol import AssetLock, AssetRecord, canonical_sha256
from plumb.scenarios import scenario_manifest_hash
from plumb.study import (
    BURST_REHEARSAL_COHORT,
    PRIMARY_EPISODE_COUNT,
    PRIMARY_POLICIES,
    PRIMARY_TASKS,
    StudyValidationError,
    assess_burst_target_claim,
    assess_primary_readiness,
    build_cost_fidelity_design,
    build_drift_design,
    build_full_study_plan,
    build_transport_sensitivity_design,
    materialize_study_run,
    main as study_main,
    study_plan_from_mapping,
    validate_burst_rehearsals,
)


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def start(task: str, panel: str, index: int) -> dict:
    token = "%s-%s-%03d" % (task, panel, index)
    prompts = {
        "open_drawer": "Open the drawer",
        "close_drawer": "Close the drawer",
        "to_basket": "Put the eggplant in the yellow basket",
        "to_sink": "Put the eggplant in the blue sink",
        "fold_cloth": "fold the cloth from top right to bottom left",
    }
    return {
        "task": task,
        "start_id": "start-" + token,
        "start_lineage_id": "lineage-" + token,
        "source_dataset": "reviewed-real-robot-source",
        "source_revision": "source-revision-20260919",
        "episode_id": "source-episode-" + token,
        "frame_id": "frame-" + token,
        "image_hash": digest("image-" + token),
        "state_hash": digest("state-" + token),
        "goal_reference_hashes": [digest("goal-" + token)],
        "goal_reference_provenance": {"source": "reviewed-goal-panel", "id": token},
        "image_timestamp": "2026-09-19T01:00:00Z",
        "state_timestamp": "2026-09-19T01:00:00Z",
        "state_convention": "Bridge-8D",
        "camera": "bridge-overhead",
        "crop": "fixed-256-square",
        "camera_calibration": "camera-calibration-revision",
        "scene": "reviewed-task-scene",
        "scene_configuration": "recorded-object-layout",
        "initial_state_stratum": "stratum-a",
        "instruction": prompts[task],
    }


def full_scenario() -> dict:
    tasks = {}
    for task in PRIMARY_TASKS:
        tasks[task] = {
            "starts": [start(task, "primary", index) for index in range(50)],
            "panels": {
                "development": [start(task, "development", 0)],
                "heldout_calibration": [start(task, "heldout", 0)],
                "cost_confirmation": [start(task, "confirmation", 0)],
            },
        }
    scenario = {
        "schema_version": 1,
        "manifest_id": "real-panel-v1",
        "provenance_kind": "real_robot",
        "source_manifest_id": "source-panel-v1",
        "source_manifest_sha256": digest("source-manifest"),
        "shared_policy_ids": list(PRIMARY_POLICIES),
        "tasks": tasks,
        "status": "frozen",
        "frozen": True,
    }
    scenario["sha256"] = scenario_manifest_hash(scenario)
    return scenario


def qualification_asset(asset_id: str) -> AssetRecord:
    """A structurally complete lock row for qualification-bound test evidence."""

    return AssetRecord(
        asset_id=asset_id,
        repo_id="study-qualification/" + asset_id,
        repo_type="model",
        revision="revision-" + asset_id,
        sha256=digest("asset:" + asset_id),
        byte_length=1,
        retrieved_at="2026-09-19T00:00:00Z",
        license="MIT",
        access_status="resolved",
        loader_revision="loader-" + asset_id,
        container_digest=digest("container:" + asset_id),
        compatibility_profile_id="profile-" + asset_id,
    )


def qualification_asset_lock() -> tuple[AssetLock, dict]:
    """The exact full-scope lock shape QualificationValidator requires."""

    lock = AssetLock()
    for asset_id in ("world-model", "world-code", "judge-model", "judge-code", "scenario-source"):
        lock.add(qualification_asset(asset_id))
    policies = {}
    for policy in PRIMARY_POLICIES:
        bound = ("policy-" + policy, "policy-code-" + policy)
        policies[policy] = list(bound)
        for asset_id in bound:
            lock.add(qualification_asset(asset_id))
    return lock, {
        "world_model": ["world-model", "world-code"],
        "policies": policies,
        "judge": ["judge-model", "judge-code"],
        "scenario": ["scenario-source"],
    }


def protocol_metadata(lock: AssetLock, assets: dict) -> dict:
    policies = {
        policy: {
            "asset_manifest_id": "assets-" + policy.lower(),
            "adapter_hash": digest("adapter-" + policy),
            "profile_id": "policy-" + policy + "-profile",
        }
        for policy in PRIMARY_POLICIES
    }
    policies["Octo"]["display_name"] = "Octo-Small v1.0"
    return {
        "protocol_id": "full-study-v1",
        "status": "frozen",
        "remote_record": "https://records.example.test/plumb/full-study-v1",
        "policy": {"family": "released-wrappers"},
        "world_model": {"family": "pinned-world-backend"},
        "judge": {"family": "pinned-five-sample-judge"},
        "thresholds": {"minimum_scientific_coverage": 0.95},
        "tolerances": {
            "gate_d": {
                "min_binary_kappa": 0.6,
                "min_weighted_progress_kappa": 0.6,
                "max_leniency_offset": 0.15,
                "min_consensus_coverage": 0.8,
            }
        },
        "judge_sampling": {"sample_count": 5},
        "asset_lock_hash": canonical_sha256(lock.to_mapping()),
        "qualification_assets": assets,
        "feedback_mode": "native_feedback",
        "policy_revisions": policies,
        "world_model_revision": {
            "asset_manifest_id": "assets-world-v1",
            "backend_profile_hash": digest("backend-profile"),
            "model_revision": "world-model-revision",
            "code_revision": "world-code-revision",
            "container_digest": "sha256:world-container-digest",
        },
        "judge_revision": {
            "asset_manifest_id": "assets-judge-v1",
            "model_revision": "judge-model-revision",
            "processor_revision": "judge-processor-revision",
            "runtime_lock_hash": digest("judge-runtime-lock"),
            "rubric_hash": digest("judge-rubric"),
            "sampling_hash": digest("judge-sampling"),
        },
    }


@pytest.fixture(scope="module")
def scenario_and_plan():
    scenario = full_scenario()
    lock, assets = qualification_asset_lock()
    return scenario, build_full_study_plan(scenario, protocol_metadata(lock, assets), base_seed=20260919), lock, assets


def artifact(label: str, timestamp: str) -> dict:
    return {"uri": "artifact://burst/" + label, "sha256": digest(label), "created_at": timestamp}


def rehearsal(plan, run_id: str) -> dict:
    rows = materialize_study_run(plan, run_id, cohort=BURST_REHEARSAL_COHORT)
    for index, row in enumerate(rows):
        row.update(
            {
                "status": "completed",
                "validity": "valid",
                "executed_actions": row["horizon_actions"],
                "video_ref": artifact("%s-video-%04d" % (run_id, index), "2026-09-19T02:00:20Z"),
                "raw_judge_samples_ref": artifact("%s-judge-%04d" % (run_id, index), "2026-09-19T02:00:30Z"),
                "platform_request_ids": ["%s-request-%04d" % (run_id, index)],
                "binary_success": False,
                "judge_status": "evaluable",
                "judge_sampling": {"sample_count": 5, "quorum": 3},
                "agreeing_samples": 5,
                "missing_reason": None,
            }
        )
    return {
        "run_id": run_id,
        "plan_hash": plan.plan_hash,
        "protocol_hash": plan.protocol_hash,
        "fresh_execution": True,
        "evidence_kind": "platform_execution",
        "run_accepted_at": "2026-09-19T02:00:00Z",
        "finalized_at": "2026-09-19T02:00:59Z",
        "episodes": rows,
        "cost": {
            "kind": "total_demonstration",
            "status": "estimated",
            "usd": 11.0,
            "pricing_snapshot_ref": artifact("price-" + run_id, "2026-09-19T01:59:00Z"),
            "allocation_ledger_ref": artifact("allocation-" + run_id, "2026-09-19T02:00:59Z"),
        },
    }


def measured_profile(prefix: str, model_asset: str, code_asset: str, *, gpu: bool = False) -> dict:
    profile = {
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
        profile["gpu_profile"] = {
            "gpu_model": "H100",
            "device_uuid": "GPU-study-fixture",
            "driver_version": "570.0",
            "runtime_version": "cuda-12.8",
            "measured_at": "2026-09-19T00:01:00Z",
            "evidence_uri": "https://evidence.example/gpu-profile.json",
            "peak_memory_bytes": 123456,
        }
    return profile


def passing_gates(plan) -> GateLedger:
    """Structurally bound A--D evidence for readiness-path tests only."""

    bindings = plan.protocol["revision_bindings"]
    ledger = GateLedger.new(plan.protocol_hash)
    common = {
        "status": GateStatus.PASS,
        "protocol_hash": plan.protocol_hash,
        "fixture_ids": ("real-fixture",),
        "evidence_uris": ("https://evidence.example/gate.json",),
        "thresholds": {"frozen": True},
        "evidence_kind": "real_cluster_execution",
    }
    backend = measured_profile("world", "world-model", "world-code", gpu=True)
    ledger.record(
        GateRecord(
            gate_id="A",
            backend_revisions=bindings["world_model"],
            measurements={
                "backend_calls": 1,
                "wall_seconds": 0.5,
                "gpu_peak_memory_bytes": backend["gpu_profile"]["peak_memory_bytes"],
                "backend_profile": backend,
            },
            **common,
        )
    )
    prefixes = {policy: (1 if policy == "OpenVLA" else 4) for policy in PRIMARY_POLICIES}
    policy_profiles = {}
    for policy in PRIMARY_POLICIES:
        profile = measured_profile("policy-" + policy, "policy-" + policy, "policy-code-" + policy)
        profile["executed_prefix_ticks"] = prefixes[policy]
        profile["certified_terminal_prefixes"] = {
            task: (1 if policy == "OpenVLA" else (2 if task in {"open_drawer", "close_drawer"} else 4))
            for task in PRIMARY_TASKS
        }
        policy_profiles[policy] = profile
    ledger.record(
        GateRecord(
            gate_id="B",
            policy_revisions=bindings["policy"],
            measurements={
                "interventions": {
                    arm: {"status": "pass", "evidence_uri": "https://evidence.example/" + arm}
                    for arm in ("original", "zero", "temporally_permuted", "sign_reversed", "cross_episode")
                },
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
    ledger.record(
        GateRecord(
            gate_id="C",
            start_ids=tuple("%s/%s" % (episode.task, episode.start_id) for episode in plan.episodes),
            measurements={
                "starts_per_task": {task: 50 for task in PRIMARY_TASKS},
                "comparability": {
                    task: {"status": "pass", "evidence_uri": "https://evidence.example/" + task + "-panel"}
                    for task in PRIMARY_TASKS
                },
            },
            **common,
        )
    )
    judge_profile = measured_profile("judge", "judge-model", "judge-code")
    judge_profile["runtime_lock_hash"] = digest("judge-runtime")
    ledger.record(
        GateRecord(
            gate_id="D",
            judge_revisions=bindings["judge"],
            measurements={
                "calibration_report": {"gate_d": {"status": "pass", "passed": True}},
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
                "judge_profile": judge_profile,
            },
            **common,
        )
    )
    return ledger


def legacy_bare_pass_gates(plan) -> GateLedger:
    """Old status-only records remain explicitly insufficient for readiness."""

    ledger = GateLedger.new(plan.protocol_hash)
    common = {
        "status": GateStatus.PASS,
        "protocol_hash": plan.protocol_hash,
        "fixture_ids": ("legacy-fixture",),
        "evidence_uris": ("artifact://legacy-evidence.json",),
        "thresholds": {"frozen": True},
        "evidence_kind": "real_cluster_execution",
    }
    for gate_id in ("A", "B", "D"):
        ledger.record(GateRecord(gate_id=gate_id, **common))
    ledger.record(
        GateRecord(
            gate_id="C",
            start_ids=tuple("%s/%s" % (episode.task, episode.start_id) for episode in plan.episodes),
            **common,
        )
    )
    return ledger


def native_point(identifier: str, steps: int) -> dict:
    return {
        "operating_point_id": identifier,
        "resolution": [256, 256],
        "diffusion_steps": steps,
        "backend_profile_id": "world-profile-v1",
        "judge_profile_id": "judge-profile-v1",
        "feedback_mode": "native_feedback",
        "comparison_kind": "native_cost_fidelity",
        "backend_supported_action_lengths": [1],
        "policy_cadences": {
            policy: {"certified_execute_prefix": 1, "world_request_action_length": 1}
            for policy in PRIMARY_POLICIES
        },
    }


def transport_point(identifier: str, request_count: int) -> dict:
    return {
        "operating_point_id": identifier,
        "resolution": [256, 256],
        "diffusion_steps": 20,
        "request_count": request_count,
        "backend_profile_id": "world-profile-v1",
        "judge_profile_id": "judge-profile-v1",
        "feedback_mode": "unqualified",
        "comparison_kind": "transport_sensitivity",
    }


def test_full_plan_is_deterministic_immutable_and_run_materialization_is_unique(scenario_and_plan):
    scenario, plan, lock, assets = scenario_and_plan
    duplicate = build_full_study_plan(scenario, protocol_metadata(lock, assets), base_seed=20260919)
    assert plan.plan_hash == duplicate.plan_hash
    assert len(plan.episodes) == PRIMARY_EPISODE_COUNT
    open_start = [episode for episode in plan.episodes if episode.task == "open_drawer" and episode.start_id.endswith("primary-000")]
    assert len({episode.world_seed for episode in open_start}) == 1
    assert len({episode.policy_seed for episode in open_start}) == len(PRIMARY_POLICIES)
    first = materialize_study_run(plan, "run-one")
    second = materialize_study_run(plan, "run-two")
    assert {row["plan_slot_id"] for row in first} == {row["plan_slot_id"] for row in second}
    assert {row["episode_id"] for row in first}.isdisjoint({row["episode_id"] for row in second})
    restored = study_plan_from_mapping(plan.to_mapping())
    assert restored.plan_hash == plan.plan_hash


def test_readiness_requires_actual_bound_revisions_and_frozen_scenario(scenario_and_plan):
    scenario, plan, lock, _ = scenario_and_plan
    ledger = passing_gates(plan)
    missing_lock = assess_primary_readiness(plan, scenario, ledger)
    assert missing_lock["status"] == "blocked"
    assert any("asset lock" in error for error in missing_lock["errors"])
    report = assess_primary_readiness(plan, scenario, ledger, asset_lock=lock)
    assert report["status"] == "ready"
    changed = GateLedger.from_mapping(ledger.to_mapping())
    wrong_revisions = copy.deepcopy(changed.records["B"].policy_revisions)
    wrong_revisions["OpenVLA"]["adapter_hash"] = digest("wrong-openvla-adapter")
    changed.record(
        GateRecord.from_mapping(
            {
                **changed.records["B"].to_mapping(),
                "policy_revisions": wrong_revisions,
            }
        )
    )
    blocked = assess_primary_readiness(plan, scenario, changed, asset_lock=lock)
    assert blocked["status"] == "blocked"
    assert any("Gate B policy revisions" in error for error in blocked["errors"])
    stale_scenario = copy.deepcopy(scenario)
    stale_scenario["tasks"]["open_drawer"]["starts"][0]["scene"] = "changed-without-rehash"
    stale = assess_primary_readiness(plan, stale_scenario, ledger, asset_lock=lock)
    assert stale["status"] == "blocked"
    assert any("scenario manifest integrity" in error for error in stale["errors"])


def test_legacy_bare_pass_records_remain_blocked_even_with_a_bound_asset_lock(scenario_and_plan):
    scenario, plan, lock, _ = scenario_and_plan
    report = assess_primary_readiness(plan, scenario, legacy_bare_pass_gates(plan), asset_lock=lock)
    assert report["status"] == "blocked"
    assert any("backend revision" in error or "backend_profile" in error for error in report["errors"])


def test_burst_claim_threads_the_same_required_asset_lock(scenario_and_plan):
    scenario, plan, lock, _ = scenario_and_plan
    ledger = passing_gates(plan)
    rehearsal_report = {
        "plan_hash": plan.plan_hash,
        "protocol_hash": plan.protocol_hash,
        "status": "not_qualified",
        "report_hash": digest("incomplete-rehearsal-report"),
        "rehearsals": [],
        "errors": ["deliberately incomplete test fixture"],
    }
    missing = assess_burst_target_claim(plan, scenario, ledger, rehearsal_report)
    assert missing["status"] == "blocked"
    assert any("asset lock" in error for error in missing["errors"])
    bound = assess_burst_target_claim(plan, scenario, ledger, rehearsal_report, asset_lock=lock)
    assert bound["status"] == "blocked"
    assert not any("asset lock" in error for error in bound["errors"])


def test_readiness_cli_loads_the_explicit_asset_lock(scenario_and_plan, tmp_path, capsys):
    scenario, plan, lock, _ = scenario_and_plan
    paths = {
        "plan": tmp_path / "plan.json",
        "scenario": tmp_path / "scenario.json",
        "gates": tmp_path / "gates.json",
        "lock": tmp_path / "assets.lock.json",
    }
    paths["plan"].write_text(json.dumps(plan.to_mapping()), encoding="utf-8")
    paths["scenario"].write_text(json.dumps(scenario), encoding="utf-8")
    paths["gates"].write_text(json.dumps(passing_gates(plan).to_mapping()), encoding="utf-8")
    paths["lock"].write_text(json.dumps(lock.to_mapping()), encoding="utf-8")
    status = study_main(
        [
            "readiness",
            "--plan",
            str(paths["plan"]),
            "--scenario",
            str(paths["scenario"]),
            "--gates",
            str(paths["gates"]),
            "--asset-lock",
            str(paths["lock"]),
        ]
    )
    assert status == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


def test_cost_and_drift_design_hold_exact_horizons_and_disjoint_lineages(scenario_and_plan):
    scenario, plan, _, _ = scenario_and_plan
    development = [scenario["tasks"][task]["panels"]["development"][0] for task in PRIMARY_TASKS]
    confirmation = [scenario["tasks"][task]["panels"]["cost_confirmation"][0] for task in PRIMARY_TASKS]
    points = [native_point("op-256-20", 20), native_point("op-256-30", 30)]
    design = build_cost_fidelity_design(plan, development, confirmation, points)
    assert all(sum(condition["request_action_lengths"]) == condition["horizon_actions"] for condition in design.conditions)
    openvla = next(condition for condition in design.conditions if condition["policy"] == "OpenVLA")
    assert len(openvla["policy_feedback_boundaries"]) == openvla["horizon_actions"] == 70
    grouped = {}
    for condition in design.conditions:
        grouped.setdefault((condition["cohort"], condition["policy"], condition["task"], condition["start_id"]), set()).add(
            condition["seeds"]["world"]
        )
    assert all(len(seeds) == 1 for seeds in grouped.values())
    transport_points = [transport_point("transport-three", 3)]
    transport = build_transport_sensitivity_design(plan, development, transport_points)
    assert all(condition["policy_feedback_boundaries"] is None for condition in transport.conditions)
    drift = build_drift_design(
        plan,
        [
            {
                "task": task,
                "sequence_id": "heldout-" + task,
                "source_lineage_id": "drift-lineage-" + task,
                "actions_hash": digest("actions-" + task),
                "video_hash": digest("video-" + task),
                "recorded_control_ticks": 100,
            }
            for task in PRIMARY_TASKS
        ],
        transport_points,
    )
    assert all({arm["mode"] for arm in comparison["arms"]} == {"teacher_forced", "free_running"} for comparison in drift.comparisons)
    with pytest.raises(StudyValidationError):
        build_cost_fidelity_design(plan, development, development, points)
    forbidden_native_partition = native_point("bad-native", 20)
    forbidden_native_partition["request_count"] = 3
    with pytest.raises(StudyValidationError, match="may not set request_count"):
        build_cost_fidelity_design(plan, development, confirmation, [forbidden_native_partition])


def test_burst_validation_rejects_partial_mixed_stale_and_reused_runs(scenario_and_plan):
    _, plan, _, _ = scenario_and_plan
    good = [rehearsal(plan, "burst-%d" % index) for index in range(3)]
    # Deterministic generation may repeat bytes. Fresh URI/request/timestamp
    # identity still makes this a new execution rather than a reused artifact.
    good[1]["episodes"][0]["video_ref"]["sha256"] = good[0]["episodes"][0]["video_ref"]["sha256"]
    valid = validate_burst_rehearsals(plan, good)
    assert valid["status"] == "evidence_validated"

    removed = good[0]["episodes"].pop()
    original_protocol = good[1]["episodes"][0]["protocol_hash"]
    good[1]["episodes"][0]["protocol_hash"] = digest("other-protocol")
    original_created_at = good[2]["episodes"][0]["video_ref"]["created_at"]
    original_video = good[2]["episodes"][0]["video_ref"]
    good[2]["episodes"][0]["video_ref"] = copy.deepcopy(good[0]["episodes"][0]["video_ref"])
    good[2]["episodes"][0]["video_ref"].update(
        {"sha256": digest("attempted-overwrite-different-bytes"), "created_at": "2026-09-19T01:59:59Z"}
    )
    combined_bad = validate_burst_rehearsals(plan, good)
    assert combined_bad["status"] == "not_qualified"
    assert any(
        "records" in error or "missing" in error for error in combined_bad["rehearsals"][0]["errors"]
    )
    assert any("protocol_hash" in error for error in combined_bad["rehearsals"][1]["errors"])
    assert any("predates" in error for error in combined_bad["rehearsals"][2]["errors"])
    assert any("generation artifact URI/version identities are reused" in error for error in combined_bad["errors"])
    good[0]["episodes"].append(removed)
    good[1]["episodes"][0]["protocol_hash"] = original_protocol
    good[2]["episodes"][0]["video_ref"] = original_video
    good[2]["episodes"][0]["video_ref"]["created_at"] = original_created_at

    changed_rows = []
    for row in good[0]["episodes"][1:]:
        changed_rows.append((row, row["validity"], row["binary_success"], row["missing_reason"]))
        row.update({"validity": "invalid", "binary_success": None, "missing_reason": "validity_rejected"})
    unknown_rows = []
    for row in good[1]["episodes"]:
        unknown_rows.append((row, row["binary_success"], row["judge_status"], row["missing_reason"]))
        row.update({"binary_success": None, "judge_status": "unknown", "missing_reason": "judge_no_quorum"})
    coverage_report = validate_burst_rehearsals(plan, good)
    for row, validity, binary, reason in changed_rows:
        row.update({"validity": validity, "binary_success": binary, "missing_reason": reason})
    for row, binary, judge_status, reason in unknown_rows:
        row.update({"binary_success": binary, "judge_status": judge_status, "missing_reason": reason})
    assert coverage_report["status"] == "not_qualified"
    assert coverage_report["rehearsals"][0]["scientific_coverage"] == pytest.approx(1 / PRIMARY_EPISODE_COUNT)
    assert coverage_report["rehearsals"][1]["scientific_coverage"] == 0.0
