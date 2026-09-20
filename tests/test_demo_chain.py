"""CPU contracts for the KOSMOS low-level streaming demo Chain.

These tests use explicit fake policy/world workers only to exercise stream
framing and controller ordering. They neither emulate model inference nor mark
any policy/world output as qualified or scored.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deploy" / "baseten" / "demo_chain.py"
SPEC = importlib.util.spec_from_file_location("plumb_demo_chain_test", SOURCE)
assert SPEC and SPEC.loader
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def _frame(value: int = 0):
    return demo.PngFrame.from_array(np.full((3, 4, 3), value, dtype=np.uint8))


def _world(backend: str = "cosmos"):
    if backend == "cosmos":
        return demo.WorldProfile(
            backend="cosmos",
            profile_id="cosmos-demo-256",
            model_id="nvidia/Cosmos3-Nano",
            model_revision="e59a53c25979a090fa8706c9acc0c254a6e89b92",
            resolution_tier=256,
            inference_steps=30,
            asset_manifest_id="assets-demo",
        )
    return demo.WorldProfile(
        backend="irasim",
        profile_id="irasim-demo-50",
        model_id="bytedance/IRASim",
        source_revision="c72b6dade6fcd65971e0aa8ab49ea39b15108c90",
        resolution_tier=256,
        inference_steps=50,
        asset_manifest_id="assets-demo",
    )


def _rollout(**overrides: Any):
    payload = {
        "attempt_id": "attempt-1",
        "comparison_id": "comparison-1",
        "cell_id": "openvla-101",
        "policy_id": "OpenVLA",
        "world_seed": 101,
        "world": _world(),
        "source_frame": _frame(),
        "initial_source_state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.4],
        "bridge_control_profile_id": "bridge-demo",
        "held_gripper_action": 0.4,
        "source_bundle_id": "source-start-1",
        "source_bundle_sha256": "sha256:" + "a" * 64,
        "identity_hashes": {"configuration": "sha256:" + "b" * 64},
        "policy_revision": "openvla-source-revision",
        "source_state_lineage_id": "source-state-1",
        "proprio_lineage_id": "source-proprio-1",
        "proprio_converter_revision": "bridge-source-converter-1",
    }
    payload.update(overrides)
    return demo.RolloutInput(**payload)


class _Policy:
    def __init__(self):
        self.calls = []

    async def run_remote(self, request):
        self.calls.append(request)
        return demo.PolicyTurnResult(
            status="completed",
            raw_proposal_rows=[[0.01, 0, 0, 0, 0, 0, 0.4]],
            source_executed_rows=[[0.01, 0, 0, 0, 0, 0, 0.4]],
            policy_snapshot={"turn": request.action_index},
            identifiers={"worker": "fake-policy"},
            timing={"backend_calls": 1},
        )


class _World:
    def __init__(self):
        self.calls = []

    async def run_remote(self, request):
        self.calls.append(request)
        return demo.WorldTurnResult(
            status="completed",
            frames=[_frame((request.native_actions[0][0] != 0) * 80 + len(self.calls)) for _ in request.native_actions],
            expected_post_conditioning_frames=len(request.native_actions),
            conditioning_frame_removed=True,
            identifiers={"worker": "fake-world", "request_id": request.request_id},
            timing={"backend_calls": 1},
        )


def _collect(iterator):
    async def gather():
        return [value async for value in iterator]

    return asyncio.run(gather())


def test_rollout_stream_uses_receiver_envelope_and_exactly_seventy_fresh_turns():
    policy = _Policy()
    world = _World()
    core = demo.DemoControllerCore({"OpenVLA": policy}, world)

    records = [json.loads(value) for value in _collect(core.rollout_events(_rollout()))]
    assert all(record["schema"] == "kosmos-chain-stream-v1" for record in records)
    assert all(record["kind"] in {"header", "stage", "frame", "heartbeat", "terminal"} for record in records)
    assert len({record["event_id"] for record in records}) == len(records)
    assert all(record["comparison_id"] == "comparison-1" and record["cell_id"] == "openvla-101" for record in records)
    header = records[0]
    assert header["kind"] == "header"
    assert header["mode"] == "comparison"
    assert header["task"] == "close_drawer"
    assert header["task_instruction"] == "Close the drawer"
    assert header["policy"] == "OpenVLA"
    assert header["world_seed"] == 101
    assert header["identity_hashes"] == {"configuration": "sha256:" + "b" * 64}
    frames = [record for record in records if record["kind"] == "frame"]
    assert len(frames) == 70
    assert [record["frame_index"] for record in frames] == list(range(1, 71))
    assert all(len(record["action"]["executed"]) == 7 for record in frames)
    assert all(record["state"]["origin"] == "forecast_integrated" for record in frames)
    terminal = records[-1]
    assert terminal["kind"] == "terminal"
    assert terminal["status"] == "completed"
    assert terminal["action_count"] == 70
    assert terminal["complete"] is True
    assert len(policy.calls) == len(world.calls) == 70
    assert all(len(call.native_actions) == len(call.compiled_actions) == 1 for call in world.calls)
    assert [call.action_index for call in policy.calls] == list(range(70))
    assert policy.calls[0].state_kind == "source_measured"
    assert all(call.state_kind == "forecast_integrated" for call in policy.calls[1:])


def test_manual_cosmos_has_no_policy_call_and_keeps_only_sixteen_post_condition_frames():
    world = _World()
    core = demo.DemoControllerCore({}, world)
    condition = _frame()
    request = demo.ManualInput(
        attempt_id="manual-attempt",
        manual_session_id="manual-session",
        command_id="right-1",
        direction="right",
        world_seed=104,
        world=_world(),
        condition_frame=condition,
        state_snapshot=[0.1, 0.2, 0.3, 0, 0, 0, 0, 0.4],
        bridge_control_profile_id="bridge-demo",
        held_gripper_action=0.4,
        source_branch_id="branch-1",
        source_branch_sha256="sha256:" + "c" * 64,
        comparison_id="comparison-1",
        cell_id="openvla-101",
        identity_hashes={"configuration": "sha256:" + "b" * 64},
        source_event_id="evt-previous",
        source_png_sha256=condition.png_sha256,
    )
    records = [json.loads(value) for value in _collect(core.manual_events(request))]
    assert records[0]["mode"] == "manual"
    assert records[0]["source_event_id"] == "evt-previous"
    frames = [record for record in records if record["kind"] == "frame"]
    assert [record["frame_index"] for record in frames] == list(range(1, 17))
    assert records[-1]["status"] == "completed"
    assert records[-1]["action_count"] == 16
    assert len(world.calls) == 1
    assert world.calls[0].operation == "manual_segment"
    assert len(world.calls[0].native_actions) == len(world.calls[0].compiled_actions) == 16


def test_manual_irasim_is_explicitly_blocked_under_current_sixteen_post_frame_receiver_contract():
    world = _World()
    core = demo.DemoControllerCore({}, world)
    condition = _frame()
    request = demo.ManualInput(
        attempt_id="manual-irasim",
        manual_session_id="manual-session",
        command_id="right-1",
        direction="right",
        world_seed=104,
        world=_world("irasim"),
        condition_frame=condition,
        state_snapshot=[0.1, 0.2, 0.3, 0, 0, 0, 0, 0.4],
        bridge_control_profile_id="bridge-demo",
        held_gripper_action=0.4,
        source_branch_id="branch-1",
        source_branch_sha256="sha256:" + "c" * 64,
        comparison_id="comparison-1",
        cell_id="openvla-101",
        identity_hashes={"configuration": "sha256:" + "b" * 64},
        source_event_id="evt-previous",
        source_png_sha256=condition.png_sha256,
    )
    records = [json.loads(value) for value in _collect(core.manual_events(request))]
    assert records[-1]["kind"] == "terminal"
    assert records[-1]["status"] == "blocked"
    assert records[-1]["complete"] is False
    assert world.calls == []


def test_ndjson_parser_handles_split_and_joined_records_but_refuses_truncation():
    first = demo._event("heartbeat", {"attempt_id": "a", "comparison_id": "c", "cell_id": "x", "mode": "comparison"}).encode()
    second = demo._event("terminal", {"attempt_id": "a", "comparison_id": "c", "cell_id": "x", "mode": "comparison", "status": "failed"}).encode()
    parsed = demo.parse_ndjson_records([first[:9], first[9:] + second])
    assert [item["kind"] for item in parsed] == ["heartbeat", "terminal"]
    try:
        demo.parse_ndjson_records([first[:-1]])
    except demo.DemoContractError as error:
        assert "truncated" in str(error)
    else:  # pragma: no cover - protects a critical interruption invariant.
        raise AssertionError("truncated stream must be rejected")


def test_manual_direction_has_frozen_small_translation_and_source_bound_gripper_action():
    assert demo._manual_action("right", 0.0025, 0.4) == [0.0025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]
    assert demo._manual_action("up", 0.0025, 0.4) == [0.0, 0.0, 0.0025, 0.0, 0.0, 0.0, 0.4]


def test_cold_rpc_emits_heartbeats_without_retrying_the_model_call():
    calls = []

    async def delayed():
        calls.append(1)
        await asyncio.sleep(0.04)
        return "real-result"

    values = _collect(demo._await_progress(delayed(), interval=0.01))
    assert values[-1] == "real-result"
    assert None in values[:-1]
    assert calls == [1]


def test_raw_source_gripper_state_does_not_modify_held_world_gripper_action():
    first = [0.28, -0.006, 0.11, -0.01, -0.14, 0.002, 0.0, 1.000153]
    second = first[:-1] + [0.039]
    actions = [demo._manual_action("right", 0.0025, 1.0)] * 16
    raw, _, _ = demo._forecast_compile(_world(), "development", first, actions)
    alternate, _, _ = demo._forecast_compile(_world(), "development", second, actions)
    assert raw == alternate
    assert all(row[-1] == 1.0 for row in raw)


def test_cosmos_uint8_boundary_preserves_pixels_through_pil_not_unscaled_numpy():
    from PIL import Image
    from plumb.adapters.contracts import WorldResult, ServerTiming

    pixels = np.full((3, 4, 3), 128, dtype=np.uint8)

    class CapturingAdapter:
        def generate(self, request):
            assert isinstance(request.conditioning_image, Image.Image)
            assert request.conditioning_image.mode == "RGB"
            np.testing.assert_array_equal(np.asarray(request.conditioning_image), pixels)
            return WorldResult(backend="test-only", profile_id=request.compatibility_profile_id,
                               frames=tuple(pixels.copy() for _ in range(17)),
                               nominal_frame_timestamps=tuple(i / 5 for i in range(17)),
                               conditioning_frame_included=True, timing=ServerTiming(1, 0.0, False))

    runtime = demo.DemoWorldRuntime("cosmos")
    runtime._adapter = CapturingAdapter()
    request = demo.WorldTurnRequest(attempt_id="pixel-test", operation="manual_segment", world=_world(),
        condition=demo.PngFrame.from_array(pixels), native_actions=[[0.0] * 7] * 16,
        compiled_actions=[[0.0] * 10] * 16, nominal_control_timestamps=[i / 5 for i in range(16)],
        world_seed=101, state_before=[0.0] * 8, request_id="pixel-test")
    result = runtime._generate_cosmos(request, pixels, 0.0)
    assert result.status == "completed", result.reasons
    assert len(result.frames) == 16 and result.conditioning_frame_removed


def test_pinned_assets_are_hashed_once_per_worker_not_once_per_keypress(monkeypatch):
    calls = []

    def verify(asset_id):
        calls.append(asset_id)
        return {"path": "/immutable/test-cache", "verified": True}

    monkeypatch.setattr(demo, "_load_asset_manifest", verify)
    first = demo.DemoWorldRuntime("cosmos")
    assert first.readiness()["status"] == "ready_unqualified"
    assert first.readiness()["status"] == "ready_unqualified"
    assert len(calls) == 1
    assert demo.DemoWorldRuntime("cosmos").readiness()["status"] == "ready_unqualified"
    assert len(calls) == 2


def test_flat_service_dispatch_translates_only_when_full_execution_identity_is_present():
    source = _frame()
    state_payload = {"bridge_state": [0.1, 0.2, 0.3, 0, 0, 0, 0, 0.4]}
    state = {
        "payload": state_payload,
        "origin": "source-measured-bridge-v1",
        "conventions_sha256": "sha256:" + "d" * 64,
        "sha256": "sha256:" + __import__("hashlib").sha256(demo._canonical_json(state_payload).encode()).hexdigest(),
    }
    world_profile_sha = "sha256:" + "e" * 64
    payload = {
        "schema": "kosmos-chain-stream-v1",
        "kind": "comparison_rollout",
        "comparison_id": "comparison-1",
        "attempt_id": "attempt-1",
        "cell_id": "openvla-101",
        "task": "close_drawer",
        "task_instruction": "Close the drawer",
        "policy": "OpenVLA",
        "world_seed": 101,
        "horizon": 70,
        "start": {"id": "start-1", "png_base64": source.png_base64, "png_sha256": source.png_sha256, "state": state},
        "world": {
            "profile_sha256": world_profile_sha,
            "bridge_control_profile_id": "bridge-demo",
            "held_gripper_action": 0.4,
            "manual_seed": 104,
            "runtime_profile": _world().model_dump(mode="json"),
        },
        "policy_profile": {"profile": "sha256:" + "f" * 64},
        "identity_hashes": {"source_manifest": "sha256:" + "a" * 64, "world_profile": world_profile_sha},
        "claim_tier": "preview",
        "scored": False,
    }
    translated = demo.service_dispatch_to_stream_request(payload)
    assert translated.rollout is not None
    assert translated.rollout.source_frame.png_sha256 == source.png_sha256
    assert translated.rollout.initial_source_state == state_payload["bridge_state"]
    assert translated.rollout.world.profile_id == "cosmos-demo-256"
    incomplete = dict(payload)
    incomplete["world"] = {"profile_sha256": world_profile_sha}
    try:
        demo.service_dispatch_to_stream_request(incomplete)
    except demo.DemoContractError as error:
        assert "runtime_profile" in str(error)
    else:  # pragma: no cover
        raise AssertionError("display-only world config must not be executable")
