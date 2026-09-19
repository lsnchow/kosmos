"""Rehearsal transport and start resolution.

The rehearsal exists to prove the production path without a GPU. Its value
depends entirely on two properties, so both are tested hard here:

1. It speaks the **real** Chain contract, validated by the Chain's own models and
   its own ``_world_setup``. A rehearsal that accepted a looser envelope than the
   real Chain would manufacture false confidence.
2. Nothing it produces can be mistaken for a real result.
"""
from __future__ import annotations

import base64
import json
import pathlib

import pytest

from plumb.backends.baseten import BasetenBackendSettings, BasetenChainBackend
from plumb.engine import RunService
from plumb.platform import BasetenPlatformConfig, ChainAsyncEndpoint
from plumb.rehearsal import (
    REHEARSAL_CHAIN_ASYNC_URL,
    REHEARSAL_LABEL,
    SIMULATED_TRANSPORT,
    RehearsalChainTransport,
    RehearsalProfile,
    _validate_against_chain_contract,
    build_rehearsal_profile,
    chain_contract_available,
    rehearsal_enabled,
)
from plumb.starts import (
    BRIDGE_GRIPPER_STATE_MAX,
    RehearsalStartResolver,
    ResolvedStart,
    ScenarioStartResolver,
    StartResolutionError,
    _png_dimensions,
    _solid_png,
)


def _episode(**overrides):
    episode = {
        "run_id": "run-1",
        "episode_id": "ep-1",
        "policy": "OpenVLA",
        "policy_variant": "OpenVLA",
        "task": "close_drawer",
        "start_id": "start-000",
        "start_lineage_id": "lineage-000",
        "world_seed": 4242,
        "horizon_actions": 70,
        "protocol_hash": "sha256:" + "a" * 64,
        "mode": "qualification/pending-gate-review",
        "seeds": {"world": 4242, "policy": 7, "judge_samples": [1, 2, 3, 4, 5]},
    }
    episode.update(overrides)
    return episode


def _backend(tmp_path, **kwargs):
    return BasetenChainBackend(
        settings=BasetenBackendSettings(
            webhook_endpoint="https://plumb.example.com/api/callbacks",
            operating_point_id="op-256-30",
        ),
        data_dir=tmp_path,
        **kwargs,
    )


# --------------------------------------------------------------- start records


def test_a_start_must_carry_the_verbatim_prompt():
    with pytest.raises(StartResolutionError, match="verbatim prompt"):
        ResolvedStart(
            start_id="s",
            start_lineage_id="l",
            task="fold_cloth",
            prompt="Fold the cloth from top right to bottom left",  # capital F
            horizon_actions=80,
            frame_png_base64="x",
            frame_png_sha256="sha256:x",
            frame_height=256,
            frame_width=256,
            state=(0.0,) * 6 + (0.0, 0.0),
            provenance="test",
        )


def test_a_start_must_respect_the_bridge_state_convention():
    common = dict(
        start_id="s",
        start_lineage_id="l",
        task="close_drawer",
        prompt="Close the drawer",
        horizon_actions=70,
        frame_png_base64="x",
        frame_png_sha256="sha256:x",
        frame_height=256,
        frame_width=256,
        provenance="test",
    )
    with pytest.raises(StartResolutionError, match="8-D"):
        ResolvedStart(state=(0.0,) * 7, **common)
    with pytest.raises(StartResolutionError, match="channel 6"):
        ResolvedStart(state=(0.0,) * 6 + (0.5, 0.0), **common)
    with pytest.raises(StartResolutionError, match="gripper state"):
        ResolvedStart(state=(0.0,) * 6 + (0.0, 0.9), **common)
    ok = ResolvedStart(state=(0.0,) * 6 + (0.0, BRIDGE_GRIPPER_STATE_MAX), **common)
    assert ok.state[7] == BRIDGE_GRIPPER_STATE_MAX


def test_a_start_without_a_frame_is_refused():
    with pytest.raises(StartResolutionError, match="no conditioning frame"):
        ResolvedStart(
            start_id="s",
            start_lineage_id="l",
            task="close_drawer",
            prompt="Close the drawer",
            horizon_actions=70,
            frame_png_base64="",
            frame_png_sha256="sha256:x",
            frame_height=256,
            frame_width=256,
            state=(0.0,) * 8,
            provenance="test",
        )


def test_rehearsal_starts_are_deterministic_and_labelled():
    resolver = RehearsalStartResolver(256)
    first = resolver.resolve("close_drawer", "start-000", "lineage-000")
    second = resolver.resolve("close_drawer", "start-000", "lineage-000")
    assert first == second, "a rehearsal must be reproducible"
    assert first.provenance == "rehearsal_synthetic"
    assert first.horizon_actions == 70
    assert _png_dimensions(base64.b64decode(first.frame_png_base64)) == (256, 256)
    other = resolver.resolve("close_drawer", "start-001", "lineage-001")
    assert other.frame_png_sha256 != first.frame_png_sha256
    with pytest.raises(StartResolutionError, match="unknown task"):
        resolver.resolve("invented_task", "start-000", "lineage-000")


def test_the_real_resolver_refuses_a_missing_manifest_and_a_hash_mismatch(tmp_path):
    with pytest.raises(StartResolutionError, match="no scenario manifest"):
        ScenarioStartResolver(tmp_path / "scenarios.jsonl")

    image = tmp_path / "start.png"
    image.write_bytes(_solid_png(256, 256, 10, 20, 30))
    manifest = tmp_path / "scenarios.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task": "close_drawer",
                "start_id": "start-000",
                "image_uri": "start.png",
                "image_sha256": "sha256:" + "0" * 64,
                "state": [0.2, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.1],
            }
        )
        + "\n"
    )
    resolver = ScenarioStartResolver(manifest, artifact_root=tmp_path)
    with pytest.raises(StartResolutionError, match="hash does not match"):
        resolver.resolve("close_drawer", "start-000", "lineage-000")
    with pytest.raises(StartResolutionError, match="no start start-999"):
        resolver.resolve("close_drawer", "start-999", "lineage-999")


def test_the_real_resolver_returns_real_provenance(tmp_path):
    raw = _solid_png(256, 256, 1, 2, 3)
    image = tmp_path / "start.png"
    image.write_bytes(raw)
    import hashlib

    manifest = tmp_path / "scenarios.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task": "close_drawer",
                "start_id": "start-000",
                "image_uri": "start.png",
                "image_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "state": [0.2, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.1],
            }
        )
        + "\n"
    )
    start = ScenarioStartResolver(manifest, artifact_root=tmp_path).resolve(
        "close_drawer", "start-000", "lineage-000"
    )
    assert start.provenance == "scenario_manifest"
    assert start.prompt == "Close the drawer"
    assert start.source_uri.endswith("start.png")


# ------------------------------------------------------------ chain contract


def test_the_chain_contract_is_importable_for_validation():
    assert chain_contract_available() is True, (
        "without the Chain models the rehearsal cannot be a contract test"
    )


def test_an_empty_envelope_is_rejected_by_the_chain_contract():
    violation = _validate_against_chain_contract({})
    assert violation is not None and "RolloutRequest" in violation


def test_the_backend_payload_satisfies_the_real_chain_contract(tmp_path):
    backend = _backend(tmp_path, start_resolver=RehearsalStartResolver(256))
    payload = backend._build_entrypoint_input(_episode(), {}, "OpenVLA", "sha256:" + "a" * 64)
    assert _validate_against_chain_contract(payload) is None


def test_the_contract_check_catches_a_missing_world_seed(tmp_path):
    """The seed is never defaulted: same-seed repeatability is the evidence."""

    backend = _backend(tmp_path, start_resolver=RehearsalStartResolver(256))
    payload = backend._build_entrypoint_input(_episode(), {}, "OpenVLA", "sha256:" + "a" * 64)
    payload["world"]["payload"].pop("seed")
    violation = _validate_against_chain_contract(payload)
    assert violation is not None and "seed" in violation


def test_the_contract_check_catches_a_missing_episode_field(tmp_path):
    backend = _backend(tmp_path, start_resolver=RehearsalStartResolver(256))
    payload = backend._build_entrypoint_input(_episode(), {}, "OpenVLA", "sha256:" + "a" * 64)
    payload["policy"]["payload"].pop("initial_state")
    violation = _validate_against_chain_contract(payload)
    assert violation is not None and "EpisodeControlPayload" in violation


# ---------------------------------------------------------------- transport


def test_the_transport_requires_a_signing_secret(tmp_path):
    with pytest.raises(ValueError, match="webhook secret"):
        RehearsalChainTransport(webhook_secret="", artifacts_dir=tmp_path)


def test_the_transport_rejects_an_envelope_that_the_chain_would_reject(tmp_path):
    import asyncio

    from plumb.platform import HttpRequest

    transport = RehearsalChainTransport(webhook_secret="s", artifacts_dir=tmp_path)
    request = HttpRequest(
        method="POST",
        url=REHEARSAL_CHAIN_ASYNC_URL,
        headers={},
        body=json.dumps({"model_input": {"run_id": "r"}, "webhook_endpoint": "https://x/y"}).encode(),
    )
    response = asyncio.run(transport.request(request))
    assert response.status_code == 400
    assert b"Chain contract rejected" in response.body
    assert transport.stats.contract_rejections == 1


def test_the_transport_only_implements_the_routes_plumb_uses(tmp_path):
    import asyncio

    from plumb.platform import HttpRequest

    transport = RehearsalChainTransport(webhook_secret="s", artifacts_dir=tmp_path)
    response = asyncio.run(
        transport.request(HttpRequest(method="GET", url="https://chain-x.api.baseten.co/invented", headers={}))
    )
    assert response.status_code == 404, "an unexpected call must surface, not silently succeed"


def test_every_transport_report_says_it_is_not_a_measurement(tmp_path):
    transport = RehearsalChainTransport(webhook_secret="s", artifacts_dir=tmp_path)
    report = transport.report()
    assert report["transport"] == SIMULATED_TRANSPORT
    assert report["label"] == REHEARSAL_LABEL
    assert "measures nothing" in report["not_a_measurement"]


def test_a_rehearsal_result_never_claims_a_qualified_mode_or_a_world_model(tmp_path):
    transport = RehearsalChainTransport(webhook_secret="s", artifacts_dir=tmp_path)
    result = transport._build_result(
        "run-1", "ep-1", 5, transport.profile.stage_seconds(5), {"horizon_actions": 70}
    )
    assert result["transport"] == SIMULATED_TRANSPORT
    assert result["feedback_mode"] == "unqualified"
    assert result["parity_status"] == "unqualified"
    assert result["estimated_usd"] is None
    # The Chain's nested shape, deliberately with no flat aliases.
    assert "segments" in result and "gpu_seconds" in result and "timings" in result
    assert "n_segments" not in result and "compute_gpu_seconds" not in result


def test_a_rehearsal_short_final_segment_reports_the_exact_requested_horizon(tmp_path):
    transport = RehearsalChainTransport(webhook_secret="s", artifacts_dir=tmp_path)
    result = transport._build_result(
        "run-1",
        "ep-70",
        5,
        transport.profile.stage_seconds(5),
        {"horizon_actions": 70, "operating_point": {"action_chunk_size": 16}},
    )
    assert (result["horizon_actions"], result["executed_actions"]) == (70, 70)
    assert [segment["certified_frame_count"] for segment in result["segments"]] == [16, 16, 16, 16, 6]


def test_result_store_survives_a_dropped_webhook(tmp_path):
    transport = RehearsalChainTransport(
        webhook_secret="s",
        artifacts_dir=tmp_path,
        profile=RehearsalProfile(
            policy_seconds=0,
            world_seconds_per_chunk=0,
            validity_seconds=0,
            judge_seconds=0,
            dropped_webhook_rate=1.0,
        ),
    )
    request_id = "rehearsal-test-request"
    transport._complete_now(
        request_id,
        {
            "run_id": "run-1",
            "episode_id": "ep-store",
            "policy": {"payload": {"horizon_actions": 16}},
            "validity": {"payload": {"operating_point": {"action_chunk_size": 16}}},
        },
        "https://unused.example.test/callback",
    )
    result = transport.get_result(request_id)
    assert result is not None and result["status"] in {"completed", "failed"}
    assert transport.stats.webhooks_dropped == 1
    assert (tmp_path / "result-store" / (request_id + ".json")).is_file()


def test_backend_recovers_a_dropped_rehearsal_webhook_from_the_injected_result_store(tmp_path):
    transport = RehearsalChainTransport(
        webhook_secret="s",
        artifacts_dir=tmp_path / "rehearsal",
        profile=RehearsalProfile(
            policy_seconds=0,
            world_seconds_per_chunk=0,
            validity_seconds=0,
            judge_seconds=0,
            service_failure_rate=0,
            unevaluable_rate=0,
            dropped_webhook_rate=1.0,
        ),
    )
    backend = BasetenChainBackend(
        settings=BasetenBackendSettings(
            webhook_endpoint="https://plumb.example.test/api/callbacks",
            operating_point_id="op-256-30",
            poll_interval_seconds=0.001,
            request_deadline_seconds=2,
        ),
        data_dir=tmp_path / "data",
        config=BasetenPlatformConfig(
            api_key="rehearsal-key",
            webhook_secret="s",
            chain_endpoint=ChainAsyncEndpoint(REHEARSAL_CHAIN_ASYNC_URL),
        ),
        transport=transport,
        transport_kind="simulated",
        start_resolver=RehearsalStartResolver(256),
    )
    result = backend.execute(
        _episode(),
        {},
        tmp_path / "data" / "artifacts" / "run-1" / "ep-1" / "attempt-0001",
    )
    assert (result["horizon_actions"], result["executed_actions"]) == (70, 70)
    assert transport.stats.webhooks_dropped == 1
    assert backend.outbox.counts() == {"completed": 1}


def test_terminal_rehearsal_chain_failure_remains_failed_after_engine_persistence(tmp_path):
    transport = RehearsalChainTransport(
        webhook_secret="s",
        artifacts_dir=tmp_path / "rehearsal",
        profile=RehearsalProfile(
            policy_seconds=0,
            world_seconds_per_chunk=0,
            validity_seconds=0,
            judge_seconds=0,
            service_failure_rate=1.0,
            unevaluable_rate=0,
            dropped_webhook_rate=1.0,
        ),
    )
    data_dir = tmp_path / "data"
    backend = BasetenChainBackend(
        settings=BasetenBackendSettings(
            webhook_endpoint="https://plumb.example.test/api/callbacks",
            operating_point_id="op-256-30",
            poll_interval_seconds=0.001,
            request_deadline_seconds=2,
        ),
        data_dir=data_dir,
        config=BasetenPlatformConfig(
            api_key="rehearsal-key",
            webhook_secret="s",
            chain_endpoint=ChainAsyncEndpoint(REHEARSAL_CHAIN_ASYNC_URL),
        ),
        transport=transport,
        transport_kind="simulated",
        start_resolver=RehearsalStartResolver(256),
    )
    service = RunService(data_dir, backends={"baseten": backend})
    run = service.create_run(
        {
            "mode": "diagnostic",
            "backend": "baseten",
            "policies": ["OpenVLA"],
            "tasks": ["close_drawer"],
            "starts_per_task": 1,
            "max_workers": 1,
            "protocol_hash": "sha256:" + "a" * 64,
        }
    )
    service.execute_run(run["id"])
    episode = service.list_episodes(run["id"])[0]
    assert (episode["status"], episode["missing_reason"]) == ("failed", "service_failure")
    assert episode["world_identity"]["simulated_transport"] is True
    assert episode["world_identity"]["transport"] == SIMULATED_TRANSPORT
    assert len(episode["platform_request_ids"]) == 1
    assert episode["exclusion_reason"] == "simulated_terminal_service_failure"
    assert episode["error"]["backend_failure"]["missing_reason"] == "simulated_terminal_service_failure"
    assert "chain_terminal_failure" in episode["artifact_refs"]
    failure_path = data_dir / episode["artifact_refs"]["chain_terminal_failure"]["relative_path"]
    failure = json.loads(failure_path.read_text())
    assert failure["failure"]["reason"] == "remote_chain_terminal_failed"


def test_an_unscored_profile_never_returns_a_score(tmp_path):
    transport = RehearsalChainTransport(
        webhook_secret="s", artifacts_dir=tmp_path, profile=RehearsalProfile(scored=False)
    )
    for episode in range(20):
        result = transport._build_result(
            "run-1", "ep-%d" % episode, 1, transport.profile.stage_seconds(1), {"horizon_actions": 70}
        )
        assert result["binary_success"] is None
        assert result["progress_score"] is None


def test_a_simulated_backend_forces_unqualified_labels_and_marks_its_manifest(tmp_path):
    backend = _backend(tmp_path, transport_kind="simulated", start_resolver=RehearsalStartResolver(256))
    attempt = tmp_path / "artifacts" / "run-1" / "ep-1" / "attempt-0000"
    attempt.mkdir(parents=True)
    result = backend._result_from_chain(
        {
            "status": "completed",
            "validity": "valid",
            "binary_success": True,
            "progress_score": 5,
            "stages": {},
            # Even if the Chain claimed a qualified mode, a simulated transport
            # must not carry it.
            "feedback_mode": "native_feedback",
            "parity_status": "matched_provenance",
        },
        _episode(),
        attempt,
    )
    assert result["feedback_mode"] == "unqualified"
    assert result["parity_status"] == "unqualified"
    assert result["world_identity"]["simulated_transport"] is True
    manifest = json.loads((attempt / "baseten_attempt.json").read_text())
    assert manifest["claims"]["world_model"] is False, "no world model ran under a simulated transport"
    assert manifest["claims"]["qualified_measurement"] is False
    assert "simulated" in manifest["label"]


def test_rehearsal_is_opt_in_by_environment():
    assert rehearsal_enabled({}) is False
    assert rehearsal_enabled({"PLUMB_REHEARSAL_CHAIN": "0"}) is False
    assert rehearsal_enabled({"PLUMB_REHEARSAL_CHAIN": "1"}) is True
    profile = build_rehearsal_profile({"PLUMB_REHEARSAL_MAX_REPLICAS": "30", "PLUMB_REHEARSAL_SCORED": "0"})
    assert profile.max_replicas == 30 and profile.scored is False


# --- free play -------------------------------------------------------------
#
# Free-play skips the policy, the validity gate and the judge, so it is the one
# path whose envelope the scored contract does not describe.  It reached the
# browser returning HTTP 400 from the Chain, because the entrypoint required a
# protocol hash it was never given and an EpisodeControlPayload it never built.
# These lock both halves down.


def _freeplay_envelope(**overrides):
    import hashlib

    from plumb.starts import _solid_png

    png = _solid_png(64, 64, 10, 20, 30)
    frame = {
        "encoding": "png_base64",
        "data": base64.b64encode(png).decode("ascii"),
        "png_sha256": hashlib.sha256(png).hexdigest(),
        "height": 64,
        "width": 64,
    }
    protocol_hash = "sha256:" + "a" * 64
    ref = {"episode_id": "freeplay-s1", "protocol_hash": protocol_hash, "payload": {}}
    envelope = {
        "run_id": "freeplay",
        "episode_id": "freeplay-s1",
        "protocol_hash": protocol_hash,
        "policy": ref,
        "world": ref,
        "validity": ref,
        "judge": ref,
        "freeplay": {
            "task_id": "close_drawer",
            "prompt": "close the drawer",
            "compatibility_profile_id": "op-1",
            "domain": "bridge_orig_lerobot",
            "seed": 7,
            "conditioning_image": frame,
            "actions": [[0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 16,
        },
    }
    envelope["freeplay"].update(overrides)
    return envelope


def test_the_chain_accepts_a_freeplay_envelope():
    """The scored contract does not describe free-play, so it needs its own branch."""

    from plumb.rehearsal import _validate_against_chain_contract

    assert _validate_against_chain_contract(_freeplay_envelope()) is None


def test_a_freeplay_envelope_still_needs_a_protocol_hash():
    """An unscored frame still records which world configuration produced it."""

    from plumb.rehearsal import _validate_against_chain_contract

    envelope = _freeplay_envelope()
    envelope["protocol_hash"] = ""
    violation = _validate_against_chain_contract(envelope)
    assert violation is not None and "protocol_hash" in violation


def test_a_scored_envelope_is_still_held_to_the_scored_contract():
    """The free-play branch must not become a way to skip validation."""

    from plumb.rehearsal import _validate_against_chain_contract

    envelope = _freeplay_envelope()
    del envelope["freeplay"]
    violation = _validate_against_chain_contract(envelope)
    assert violation is not None, "without freeplay set, the policy payload must still be checked"
    assert "EpisodeControlPayload" in violation


def test_a_simulated_freeplay_result_scores_nothing(tmp_path):
    """No policy, no validity gate and no judge ran, so none may report a value."""

    transport = RehearsalChainTransport(webhook_secret="s", artifacts_dir=tmp_path)
    envelope = _freeplay_envelope()
    result = transport._build_freeplay_result("freeplay", "freeplay-s1", envelope["freeplay"])

    assert result["status"] == "completed"
    assert len(result["freeplay_frames"]) == 16, "one frame per commanded action row"
    assert result["validity"] is None, "no validity gate ran"
    assert result["binary_success"] is None, "no judge ran"
    assert result["judge_status"] is None
    assert result["qualified"] is False
    assert result["transport"] == SIMULATED_TRANSPORT


def test_freeplay_frames_that_fail_their_own_hash_are_dropped(tmp_path):
    """The point of the beat is that the pixels are what the world model produced."""

    import hashlib

    from plumb.starts import _solid_png

    png = _solid_png(8, 8, 1, 2, 3)
    encoded = base64.b64encode(png).decode("ascii")
    good = {"encoding": "png_base64", "data": encoded, "png_sha256": hashlib.sha256(png).hexdigest()}
    tampered = {"encoding": "png_base64", "data": encoded, "png_sha256": "0" * 64}

    backend = BasetenChainBackend(
        settings=BasetenBackendSettings(
            webhook_endpoint="https://example.invalid/api/callbacks",
            operating_point_id="op-1",
        ),
        data_dir=tmp_path,
        transport_kind="simulated",
    )
    urls = backend._persist_freeplay_frames("s", {"freeplay_frames": [good, tampered, good]})
    assert len(urls) == 2, "the frame whose bytes disagree with its declared hash is not shown"
