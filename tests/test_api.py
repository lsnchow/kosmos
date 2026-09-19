"""API contract and qualification boundaries, not real-model performance tests."""
import time

from fastapi.testclient import TestClient

from plumb.api import create_app


def test_fixture_run_is_persisted_and_never_qualified(tmp_path):
    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        assert client.get("/api/health").json()["qualified"] is False
        config = {"mode": "synthetic", "backend": "synthetic", "policies": ["OpenVLA"],
                  "tasks": ["close_drawer"], "starts_per_task": 2, "idempotency_key": "api-test"}
        response = client.post("/api/runs", json=config)
        assert response.status_code == 201, response.text
        run_id = response.json()["id"]
        assert client.post("/api/runs", json=config).json()["id"] == run_id
        for _ in range(100):
            run = client.get("/api/runs/" + run_id).json()
            if run["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        assert run["status"] == "completed"
        assert run["total"] == run["completed"] == 2
        episodes = client.get("/api/runs/" + run_id + "/episodes").json()["episodes"]
        assert len(episodes) == 2
        assert all("synthetic" in row["mode"] for row in episodes)
        report = client.get("/api/runs/" + run_id + "/analysis")
        assert report.status_code == 200, report.text
        assert report.json()["qualified"] is False
        with client.stream("GET", "/api/runs/" + run_id + "/events") as stream:
            assert "event: snapshot" in stream.read().decode()


def test_unqualified_real_run_and_unknown_inputs_rejected(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as client:
        # "qualified" is not a settable mode at all: qualification is decided by
        # QualificationValidator from frozen manifests, never asserted by a caller.
        rejected = client.post("/api/runs", json={"mode": "qualified"})
        assert rejected.status_code == 422
        assert "QualificationValidator" in rejected.text
        assert client.post("/api/runs", json={"policies": ["invented"]}).status_code == 422
        assert client.post("/api/runs", json={"policies": []}).status_code == 422
        assert client.get("/api/runs/missing").status_code == 404
        assert client.get("/api/sweeps").json()["points"] == []
        assert client.get("/api/gates").json()["qualified"] is False


def test_real_model_backend_is_refused_with_named_gate_evidence(tmp_path):
    """A gated backend must be refused with its blockers, not silently downgraded."""

    with TestClient(create_app(tmp_path / "data")) as client:
        health = client.get("/api/health").json()
        assert "baseten" not in health["available_backends"]
        registered = {entry["name"]: entry for entry in health["registered_backends"]}
        assert registered["baseten"]["available"] is False
        assert registered["baseten"]["blocking_reasons"], "a blocked backend must say why"

        response = client.post(
            "/api/runs", json={"mode": "qualification", "backend": "baseten", "starts_per_task": 1}
        )
        assert response.status_code == 409, response.text
        detail = response.json()["detail"]
        assert detail["blocking_reasons"], "the refusal must name its blockers"
        # The gate ledger travels with the refusal so the console can show it.
        assert detail["gates"]["qualified"] is False
        assert {item["id"] for item in detail["gates"]["gates"]} == {"A", "B", "C", "D", "E", "F"}
        assert all(item["status"] == "not_run" for item in detail["gates"]["gates"])


def test_synthetic_backend_cannot_claim_a_real_mode(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as client:
        response = client.post("/api/runs", json={"mode": "qualification", "backend": "synthetic"})
        assert response.status_code == 422
        assert "synthetic" in response.text


def test_artifacts_cannot_escape_data_directory(tmp_path):
    root = tmp_path / "data"
    with TestClient(create_app(root)) as client:
        (tmp_path / "private.json").write_text('{"private": true}')
        (root / "escape.json").symlink_to(tmp_path / "private.json")
        assert client.get("/api/artifacts/escape.json").status_code == 404


def test_freeplay_refuses_to_fabricate_a_frame(tmp_path):
    """Free-play exists to prove the video is generated live.

    With no certified world backend it must fail loudly rather than return a
    placeholder frame, because a placeholder would defeat the entire point of
    the beat.
    """

    with TestClient(create_app(tmp_path / "data")) as client:
        response = client.post("/api/freeplay/step", json={"direction": "right"})
        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert detail["missing"], "the refusal must name what is missing"
        assert "placeholder" in detail["reason"]
        # The commanded chunk is still reported so the caller can see that one
        # keypress expands to a whole action chunk.
        assert detail["commanded_chunk"]["rows"] == 16
        assert detail["commanded_chunk"]["direction"] == "right"


def test_freeplay_takes_a_direction_not_an_arbitrary_trajectory(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as client:
        # A raw action vector is not an accepted input: expanding the chunk
        # server-side is what keeps a client from commanding anything it likes.
        assert client.post("/api/freeplay/step", json={"action": [999, -999, 999, 0, 0, 0, 8]}).status_code == 422
        assert client.post("/api/freeplay/step", json={"direction": "teleport"}).status_code == 422
        # Release-to-stop never reaches the world model and is always available.
        stopped = client.post("/api/freeplay/step", json={"direction": "stop"})
        assert stopped.status_code == 200
        body = stopped.json()
        assert body["commanded_rows"] == 0
        assert body["scored"] is False and body["qualified"] is False


def test_freeplay_chunk_is_constant_and_within_bridge_action_ranges():
    from plumb.api import _freeplay_chunk

    chunk = _freeplay_chunk("right", None)
    assert len(chunk) == 16, "one keypress commands a whole chunk"
    assert all(row == chunk[0] for row in chunk), "the chunk is constant in one direction"
    # Observed Bridge translation range is about +/-0.015; free-play must not be
    # able to command an implausible jump.
    assert all(abs(value) <= 0.015 for row in chunk for value in row)
    assert _freeplay_chunk("stop", None)[0] == [0.0] * 7


def test_imported_smoke_evidence_never_becomes_qualification(tmp_path):
    import json
    root = tmp_path / "data"
    evidence = root / "cluster-evidence"
    evidence.mkdir(parents=True)
    (evidence / "smoke.json").write_text(json.dumps({
        "kind": "cosmos3_nano_diffusers_smoke", "status": "completed",
        "result": {"returned_frame_count": 17, "timing": {"wall_seconds": 4.5}}}))
    with TestClient(create_app(root)) as client:
        response = client.get("/api/experiments").json()
        assert response["qualified"] is False
        assert len(response["experiments"]) == 1
        record = response["experiments"][0]
        assert record["qualification"] == "not_qualified_by_smoke"
        assert record["latency_seconds"] == 4.5
        assert record["model_load_seconds"] is None
        assert client.get(record["report_url"]).status_code == 200


def test_sse_segment_events_use_the_key_the_ledger_actually_projects(tmp_path):
    """The ledger projects the event kind as `type`, not `event_type`.

    Reading the wrong key matched nothing, so the console's rollout wall stayed
    empty while the ledger held over a thousand segment events. The mismatch was
    silent because an empty list is a legitimate state early in a run.
    """

    from plumb.ledger import Ledger

    ledger = Ledger(tmp_path / "probe.sqlite3")
    rows = ledger.list_events.__doc__ or ""
    # Assert the projection contract directly rather than trusting a comment.
    ledger.create_run(
        "run-x",
        {"mode": "synthetic", "backend": "synthetic", "policies": ["OpenVLA"], "tasks": ["close_drawer"]},
        [
            {
                "episode_id": "ep-1",
                "logical_key": "k",
                "policy": "OpenVLA",
                "task": "close_drawer",
                "start_id": "s",
                "start_lineage_id": "l",
                "world_seed": 1,
                "mode": "synthetic/unqualified",
                "horizon_actions": 70,
            }
        ],
    )
    events = ledger.list_events("run-x", after=0)
    assert events, "run_created must be recorded"
    assert "type" in events[0], "the projection key is `type`"
    assert "event_type" not in events[0], (
        "if this key ever appears, plumb/api.py's SSE filter must be revisited"
    )
    assert rows is not None


def test_episode_rows_carry_the_presentation_fields_the_console_reads(tmp_path):
    """A client that connects late or reloads rebuilds its wall from the row."""

    from plumb.ledger import Ledger

    ledger = Ledger(tmp_path / "rows.sqlite3")
    ledger.create_run(
        "run-y",
        {"mode": "synthetic", "backend": "synthetic", "policies": ["OpenVLA"], "tasks": ["close_drawer"]},
        [
            {
                "episode_id": "ep-1",
                "logical_key": "k",
                "policy": "OpenVLA",
                "task": "close_drawer",
                "start_id": "s",
                "start_lineage_id": "l",
                "world_seed": 1,
                "mode": "qualification/pending-gate-review",
                "horizon_actions": 70,
            }
        ],
    )
    row = ledger.list_episodes("run-y")[0]
    for field_name in ("frame_urls", "certified_frame_count", "provenance", "segments",
                       "presentation_track", "resolution"):
        assert field_name in row, field_name
    # A planned episode has reported nothing, and an unreported certified count
    # must read as unknown rather than as a measured zero.
    assert row["frame_urls"] == []
    assert row["certified_frame_count"] is None
    assert row["provenance"] is None
