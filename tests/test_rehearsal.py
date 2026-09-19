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
