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
        assert client.post("/api/runs", json={"mode": "qualified"}).status_code == 409
        assert client.post("/api/runs", json={"policies": ["invented"]}).status_code == 422
        assert client.post("/api/runs", json={"policies": []}).status_code == 422
        assert client.get("/api/runs/missing").status_code == 404
        assert client.get("/api/sweeps").json()["points"] == []
        assert client.get("/api/gates").json()["qualified"] is False


def test_artifacts_cannot_escape_data_directory(tmp_path):
    root = tmp_path / "data"
    with TestClient(create_app(root)) as client:
        (tmp_path / "private.json").write_text('{"private": true}')
        (root / "escape.json").symlink_to(tmp_path / "private.json")
        assert client.get("/api/artifacts/escape.json").status_code == 404


def test_freeplay_is_bounded_and_explicitly_synthetic(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as client:
        response = client.post("/api/freeplay/step", json={"action": [999, -999, 999, 0, 0, 0, 8]})
        assert response.status_code == 200
        body = response.json()
        assert body["qualified"] is False and body["mode"] == "synthetic"
        assert body["state"] == {"x": 0.53, "y": 0.47, "z": 0.53, "gripper": 1.0}
        assert client.post("/api/freeplay/step", json={"action": [0]}).status_code == 422


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
