"""Baseten backend: outbox durability, honest refusals, and cost accounting.

The backend's job is to never invent an outcome.  An ambiguous POST, a lost
callback and a deadline are all explicit unevaluable results, and an unpriced
resource makes the whole cost view unknown rather than cheaper.
"""
from __future__ import annotations

import json

import pytest

from plumb.backends.baseten import (
    AllocationLedger,
    BackendNotConfigured,
    BasetenBackendSettings,
    BasetenChainBackend,
    SubmissionOutbox,
)
from plumb.platform import CallbackAssociation, PriceUnit, VerifiedPriceBasis
from plumb.records import normalise_backend_result


def _settings(**overrides):
    values = {
        "webhook_endpoint": "https://plumb.example.com/api/callbacks/run-1",
        "operating_point_id": "op-256-30",
    }
    values.update(overrides)
    return BasetenBackendSettings(**values)


def _episode(**overrides):
    episode = {
        "run_id": "run-1",
        "episode_id": "ep-1",
        "logical_key": "OpenVLA\x1fclose_drawer\x1fstart-000",
        "policy": "OpenVLA",
        "policy_variant": "OpenVLA",
        "task": "close_drawer",
        "start_id": "start-000",
        "world_seed": 4242,
        "horizon_actions": 70,
        "protocol_hash": "sha256:" + "a" * 64,
        "mode": "qualification/pending-gate-review",
    }
    episode.update(overrides)
    return episode


def _backend(tmp_path, **overrides):
    return BasetenChainBackend(settings=_settings(**overrides), data_dir=tmp_path)


def test_settings_reject_impossible_values():
    with pytest.raises(ValueError, match="action_chunk_size"):
        _settings(action_chunk_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        _settings(batch_size=0)
    with pytest.raises(ValueError, match="request_deadline_seconds"):
        _settings(request_deadline_seconds=0)


def test_unconfigured_backend_names_what_is_missing(tmp_path):
    backend = _backend(tmp_path)
    status = backend.configuration_status()
    assert status["configured"] is False
    assert any("BASETEN_API_KEY" in item for item in status["missing"])
    assert any("VerifiedChainQueueRoute" in item for item in status["missing"])
    assert any("VerifiedPriceBasis" in item for item in status["missing"])


def test_execute_refuses_without_configuration(tmp_path):
    backend = _backend(tmp_path)
    with pytest.raises(BackendNotConfigured, match="not configured"):
        backend.execute(_episode(), {}, tmp_path / "attempt")


def test_from_env_reports_missing_variables_rather_than_half_building(tmp_path):
    with pytest.raises(BackendNotConfigured, match="BASETEN_API_KEY"):
        BasetenChainBackend.from_env(_settings(), tmp_path, environ={})


def test_outbox_key_is_the_spec_logical_identity_tuple():
    episode = _episode()
    key = SubmissionOutbox.logical_key(episode, episode["protocol_hash"], "OpenVLA")
    parts = key.split("\x1f")
    assert parts == ["run-1", "OpenVLA", "close_drawer", "start-000", "4242", episode["protocol_hash"]]


def test_outbox_reserve_is_idempotent_for_one_logical_episode(tmp_path):
    outbox = SubmissionOutbox(tmp_path / "outbox.sqlite3")
    episode = _episode()
    key = SubmissionOutbox.logical_key(episode, episode["protocol_hash"], "OpenVLA")
    first = outbox.reserve(key, episode, "OpenVLA", episode["protocol_hash"], "sha256:x")
    second = outbox.reserve(key, episode, "OpenVLA", episode["protocol_hash"], "sha256:x")
    assert first["logical_key"] == second["logical_key"]
    assert outbox.counts() == {"planned": 1}, "a retry never creates a second statistical episode"


def test_outbox_rejects_a_changed_payload_for_the_same_cell(tmp_path):
    outbox = SubmissionOutbox(tmp_path / "outbox.sqlite3")
    episode = _episode()
    key = SubmissionOutbox.logical_key(episode, episode["protocol_hash"], "OpenVLA")
    outbox.reserve(key, episode, "OpenVLA", episode["protocol_hash"], "sha256:x")
    with pytest.raises(ValueError, match="different request payload"):
        outbox.reserve(key, episode, "OpenVLA", episode["protocol_hash"], "sha256:y")


def test_ambiguous_submission_is_not_terminal_and_is_never_resent(tmp_path):
    outbox = SubmissionOutbox(tmp_path / "outbox.sqlite3")
    episode = _episode()
    key = SubmissionOutbox.logical_key(episode, episode["protocol_hash"], "OpenVLA")
    outbox.reserve(key, episode, "OpenVLA", episode["protocol_hash"], "sha256:x")
    outbox.mark_ambiguous(key, {"reason": "ambiguous_submission"})
    assert outbox.counts() == {"awaiting_reconciliation": 1}
    assert len(outbox.unsettled()) == 1
    with pytest.raises(ValueError, match="terminal state"):
        outbox.settle(key, "awaiting_reconciliation", None, None)


def test_a_callback_that_beats_its_submission_is_parked_not_discarded(tmp_path):
    """The platform can deliver before the local submission commits.

    Discarding that callback would lose a real result the platform will not send
    again, so it is parked and bound once the submission lands. The HMAC has
    already been verified by the caller, so parking is not trusting an
    unauthenticated callback.
    """

    outbox = SubmissionOutbox(tmp_path / "outbox.sqlite3")

    class Early:
        request_id = "req-1"
        event_type = "predict.completed"
        raw = {"data": {"status": "completed"}}
        association = None

    assert outbox.record_callback(Early()) is True
    assert outbox.callback_for("req-1") is not None
    # Nothing to bind yet: the submission has not committed its request ID.
    assert outbox.bind_pending_callbacks() == 0

    episode = _episode()
    key = SubmissionOutbox.logical_key(episode, episode["protocol_hash"], "OpenVLA")
    outbox.reserve(key, episode, "OpenVLA", episode["protocol_hash"], "sha256:x")
    outbox.mark_submitted(key, "req-1")

    assert outbox.bind_pending_callbacks() == 1
    assert outbox.bind_pending_callbacks() == 0, "binding is idempotent"
    stored = outbox.callback_for("req-1")
    assert stored["run_id"] == "run-1" and stored["episode_id"] == "ep-1"
    assert int(stored["associated"]) == 1


def test_chain_response_shape_aliases_are_both_understood(tmp_path):
    """The Chain reports nested measurements; a flat alias is also accepted.

    Reading only one dialect silently nulled every spec section 8 measurement.
    """

    backend = _backend(tmp_path)
    episode = _episode()
    attempt = tmp_path / "artifacts" / "run-1" / "ep-1" / "attempt-0000"
    attempt.mkdir(parents=True)
    nested = backend._result_from_chain(
        {
            "status": "completed",
            "validity": "valid",
            "binary_success": True,
            "progress_score": 5,
            "stages": {},
            "segments": [{"index": 0}, {"index": 1}, {"index": 2}],
            "timings": [{"stage": "world", "seconds": 1.5}, {"stage": "judge", "seconds": 0.5}],
            "gpu_seconds": {"total_instrumented_gpu_seconds": 2.0, "allocated_gpu_seconds": 3.0},
        },
        episode,
        attempt,
    )
    assert nested["n_segments"] == 3
    assert nested["compute_gpu_seconds"] == pytest.approx(2.0)
    assert nested["allocated_gpu_seconds"] == pytest.approx(3.0)
    assert nested["timing"]["world_seconds"] == pytest.approx(1.5)

    flat = backend._result_from_chain(
        {
            "status": "completed",
            "validity": "valid",
            "binary_success": True,
            "progress_score": 5,
            "stages": {},
            "n_segments": 4,
            "timing": {"world_seconds": 9.0},
            "compute_gpu_seconds": 8.0,
            "allocated_gpu_seconds": 9.0,
        },
        episode,
        attempt,
    )
    assert flat["n_segments"] == 4
    assert flat["compute_gpu_seconds"] == pytest.approx(8.0)
    assert flat["timing"]["world_seconds"] == pytest.approx(9.0)


def test_absent_measurements_stay_none_and_are_never_zero(tmp_path):
    backend = _backend(tmp_path)
    episode = _episode()
    attempt = tmp_path / "artifacts" / "run-1" / "ep-1" / "attempt-0000"
    attempt.mkdir(parents=True)
    result = backend._result_from_chain(
        {"status": "completed", "validity": "valid", "binary_success": True, "progress_score": 5, "stages": {}},
        episode,
        attempt,
    )
    assert result["n_segments"] is None
    assert result["compute_gpu_seconds"] is None
    assert result["allocated_gpu_seconds"] is None
    assert result["estimated_usd"] is None


def test_a_submission_is_refused_without_a_resolvable_start(tmp_path):
    """The Chain needs the start's real frame and state; there is no fallback."""

    from plumb.starts import RehearsalStartResolver, StartResolutionError

    backend = _backend(tmp_path)
    assert backend.start_resolver is None
    # No resolver means no submission can be built at all. (``execute`` checks
    # for a configured client first, so this is asserted at the payload builder.)
    with pytest.raises(StartResolutionError, match="no start resolver"):
        backend._build_entrypoint_input(_episode(), {}, "OpenVLA", "sha256:" + "a" * 64)

    backend.start_resolver = RehearsalStartResolver(256)
    payload = backend._build_entrypoint_input(
        _episode(), {}, "OpenVLA", "sha256:" + "a" * 64
    )
    # The entrypoint payload must satisfy the Chain's own contract.
    from plumb.rehearsal import _validate_against_chain_contract

    assert _validate_against_chain_contract(payload) is None


def test_a_duplicate_callback_is_ignored_not_double_counted(tmp_path):
    outbox = SubmissionOutbox(tmp_path / "outbox.sqlite3")
    episode = _episode()
    key = SubmissionOutbox.logical_key(episode, episode["protocol_hash"], "OpenVLA")
    outbox.reserve(key, episode, "OpenVLA", episode["protocol_hash"], "sha256:x")
    outbox.mark_submitted(key, "req-1")

    class Fake:
        request_id = "req-1"
        event_type = "predict.completed"
        raw = {"data": {"status": "completed"}}
        association = CallbackAssociation(request_id="req-1", run_id="run-1", episode_id="ep-1")

    assert outbox.record_callback(Fake()) is True
    assert outbox.record_callback(Fake()) is False


def test_association_resolves_only_a_known_request_id(tmp_path):
    outbox = SubmissionOutbox(tmp_path / "outbox.sqlite3")
    episode = _episode()
    key = SubmissionOutbox.logical_key(episode, episode["protocol_hash"], "OpenVLA")
    outbox.reserve(key, episode, "OpenVLA", episode["protocol_hash"], "sha256:x")
    outbox.mark_submitted(key, "req-1")
    assert outbox.association("req-1").episode_id == "ep-1"
    assert outbox.association("req-unknown") is None


def test_reconcile_resolves_a_late_callback_and_keeps_the_rest_unknown(tmp_path):
    backend = _backend(tmp_path)
    episode = _episode()
    key = SubmissionOutbox.logical_key(episode, episode["protocol_hash"], "OpenVLA")
    backend.outbox.reserve(key, episode, "OpenVLA", episode["protocol_hash"], "sha256:x")
    backend.outbox.mark_submitted(key, "req-1")

    lost = _episode(episode_id="ep-2", start_id="start-001", world_seed=99)
    lost_key = SubmissionOutbox.logical_key(lost, lost["protocol_hash"], "OpenVLA")
    backend.outbox.reserve(lost_key, lost, "OpenVLA", lost["protocol_hash"], "sha256:y")
    backend.outbox.mark_submitted(lost_key, "req-2")

    class Fake:
        request_id = "req-1"
        event_type = "predict.completed"
        raw = {"data": {"status": "completed", "validity": "valid", "binary_success": True, "progress_score": 5}}
        association = CallbackAssociation(request_id="req-1", run_id="run-1", episode_id="ep-1")

    backend.outbox.record_callback(Fake())
    report = backend.reconcile("run-1")
    assert key in report["resolved"]
    assert [row["request_id"] for row in report["unresolved"]] == ["req-2"]
    assert "never replaced by a guess" in report["note"]


def test_a_blocked_chain_stage_becomes_an_unevaluable_outcome_naming_the_stage(tmp_path):
    backend = _backend(tmp_path)
    episode = _episode()
    attempt = tmp_path / "artifacts" / "run-1" / "ep-1" / "attempt-0000"
    attempt.mkdir(parents=True)
    result = backend._result_from_chain(
        {"status": "blocked", "stages": {"world": {"status": "blocked"}}, "reason": "no world contract"},
        episode,
        attempt,
    )
    assert result["validity"] == "unknown"
    assert result["binary_success"] is None
    assert result["progress_score"] is None
    # A structured reason, composed from the stage that blocked. Free-text prose
    # is kept as detail and must not become the machine-readable reason, because
    # exclusion accounting groups on this value.
    assert result["missing_reason"] == "stage_world_blocked"
    assert result["backend_metadata"]["detail"] == "no world contract"


def test_the_chains_own_structured_missing_reason_wins_over_the_composed_one(tmp_path):
    backend = _backend(tmp_path)
    episode = _episode()
    attempt = tmp_path / "artifacts" / "run-1" / "ep-1" / "attempt-0000"
    attempt.mkdir(parents=True)
    result = backend._result_from_chain(
        {
            "status": "failed",
            "stages": {"world": {"status": "failed"}},
            "missing_reason": "world_payload_missing_seed",
            "reason": "prose that must not be used as the reason code",
        },
        episode,
        attempt,
    )
    assert result["missing_reason"] == "world_payload_missing_seed"


def test_an_invalid_episode_never_carries_a_score(tmp_path):
    backend = _backend(tmp_path)
    episode = _episode()
    attempt = tmp_path / "artifacts" / "run-1" / "ep-1" / "attempt-0000"
    attempt.mkdir(parents=True)
    result = backend._result_from_chain(
        {
            "status": "completed",
            "validity": "invalid",
            # A backend that tries to report a score alongside invalid video must
            # not be believed.
            "binary_success": True,
            "progress_score": 5,
            "stages": {},
        },
        episode,
        attempt,
    )
    assert result["binary_success"] is None and result["progress_score"] is None
    # The engine's own validator must also accept the shape.
    normalised = normalise_backend_result(result, episode, attempt)
    assert normalised["validity"] == "invalid"


def test_the_attempt_manifest_never_claims_a_robot_or_a_qualified_measurement(tmp_path):
    backend = _backend(tmp_path)
    episode = _episode()
    attempt = tmp_path / "artifacts" / "run-1" / "ep-1" / "attempt-0000"
    attempt.mkdir(parents=True)
    refs = backend._persist_provenance(episode, attempt, {"status": "completed"}, ["req-1"])
    manifest = json.loads((attempt / "baseten_attempt.json").read_text())
    assert manifest["claims"] == {
        "real_robot": False,
        "physical_control": False,
        # A world model did run; claiming otherwise would be false.
        "world_model": True,
        "qualified_measurement": False,
    }
    assert "unqualified" in manifest["label"]
    assert refs["manifest"]["sha256"].startswith("sha256:")
    assert refs["manifest"]["url"].startswith("/api/artifacts/")


def test_real_model_artifact_refs_pass_the_engine_validator(tmp_path):
    backend = _backend(tmp_path)
    episode = _episode()
    attempt = tmp_path / "artifacts" / "run-1" / "ep-1" / "attempt-0000"
    attempt.mkdir(parents=True)
    result = backend._result_from_chain(
        {"status": "completed", "validity": "valid", "binary_success": True, "progress_score": 5, "stages": {}},
        episode,
        attempt,
    )
    normalised = normalise_backend_result(result, episode, attempt)
    assert normalised["artifact_refs"]["manifest"]["media_type"] == "application/json"


def test_allocations_are_counted_once_across_the_two_cost_views():
    ledger = AllocationLedger()
    ledger.allocate("h100", 0.0, 3600.0, count=1, view="marginal_execution")
    ledger.allocate("h100", 3600.0, 7200.0, count=1, view="prewarm")
    report = ledger.report()
    assert report["marginal_execution"]["resource_hours"]["h100"] == pytest.approx(1.0)
    assert report["total_demonstration_run"]["resource_hours"]["h100"] == pytest.approx(2.0)
    assert report["allocation_count"] == 2


def test_an_allocation_cannot_end_before_it_starts():
    ledger = AllocationLedger()
    with pytest.raises(ValueError, match="cannot end before"):
        ledger.allocate("h100", 10.0, 0.0)
    with pytest.raises(ValueError, match="unknown allocation view"):
        ledger.allocate("h100", 0.0, 1.0, view="invented")


def test_usd_is_none_without_a_price_basis_and_with_an_unpriced_resource():
    ledger = AllocationLedger()
    ledger.allocate("h100", 0.0, 3600.0)
    assert ledger.report()["marginal_execution"]["estimated_usd"] is None

    basis = VerifiedPriceBasis(
        currency="USD",
        unit=PriceUnit.PER_HOUR,
        evidence_uri="https://example.com/prices",
        retrieved_at="2026-09-19T04:00:00-04:00",
    )
    priced = AllocationLedger(price_basis=basis, rates_per_hour={"h100": 2.0})
    priced.allocate("h100", 0.0, 3600.0)
    assert priced.report()["marginal_execution"]["estimated_usd"] == pytest.approx(2.0)

    # One unpriced resource makes the whole view unknown, never silently cheaper.
    priced.allocate("cpu", 0.0, 3600.0)
    assert priced.report()["marginal_execution"]["estimated_usd"] is None


def test_telemetry_reports_unavailable_rather_than_zero(tmp_path):
    backend = _backend(tmp_path)
    payload = backend.telemetry("run-1")
    assert payload["platform_queue"]["status"] == "unavailable"
    assert payload["economics"]["marginal_execution"]["estimated_usd"] is None
    assert payload["economics"]["reconciliation_status"] == "estimated_pending_billing_reconciliation"
