from __future__ import annotations

import json
import stat

from fastapi import FastAPI
from fastapi.testclient import TestClient

from plumb.development_review import COOKIE_NAME, DevelopmentReviewStore, register_development_review_routes


def packet(clip_id: str = "review-clip-a") -> dict:
    return {
        "schema": "plumb-development-review-packets-v1",
        "purpose": "development_review_only",
        "gate_d_eligible": False,
        "clip_id": clip_id,
        "task": "close_drawer",
        "task_instruction": "Close the drawer",
        "task_rubric": "Visible completion only",
        "raw_video_url": "http://127.0.0.1:8787/api/artifacts/review/pilot/media/opaque.mp4",
        "review_guidance": {
            "integrity": {"intact": "Playable", "artifact": "Visible defect", "uncertain": "Cannot tell"},
            "collision": {"none_visible": "None visible", "visible": "Visible", "uncertain": "Cannot tell"},
            "completion": "Use visible evidence only.",
        },
        "sampled_frames": [
            {
                "index": index,
                "timestamp": index * 0.4,
                "url": "http://127.0.0.1:8787/api/artifacts/review/pilot/media/frame-%02d.png" % index,
            }
            for index in range(16)
        ],
        # This deliberately tempting field must never leave the private packet
        # reader because it is source metadata rather than review material.
        "teacher_outputs": {"hidden": True},
        "source_lineage": "never-expose",
    }


def store(tmp_path) -> DevelopmentReviewStore:
    repo = tmp_path / "repo"
    worksheet = repo / "data" / "live-integrated" / "review" / "pilot-review-v1" / "blank-worksheets.jsonl"
    worksheet.parent.mkdir(parents=True)
    worksheet.write_text(json.dumps(packet()) + "\n", encoding="utf-8")
    served = tmp_path / "served-data"
    served.mkdir()
    return DevelopmentReviewStore(
        repo_root=repo,
        review_root=repo / "data" / "live-integrated" / "review",
        served_root=served,
        private_root=tmp_path / "review-private",
    )


def app_with(store: DevelopmentReviewStore) -> FastAPI:
    app = FastAPI()
    register_development_review_routes(app, store)
    return app


def test_lists_only_opaque_review_material_and_keeps_db_private(tmp_path):
    review = store(tmp_path)
    with TestClient(app_with(review)) as client:
        listed = client.get("/api/development-review/sets")
        assert listed.json() == {
            "sets": [{"set_id": "pilot-review-v1", "title": "pilot-review-v1", "clip_count": 1, "status": "development_review_only"}]
        }
        clips = client.get("/api/development-review/sets/pilot-review-v1/clips").json()
        clip = clips["clips"][0]
        assert set(clip) == {"opaque_clip_id", "task", "media"}
        assert "teacher" not in json.dumps(clips) and "lineage" not in json.dumps(clips)
        assert clip["media"]["video_url"].endswith("opaque.mp4")
        assert clip["task"]["definitions"]["integrity"]["artifact"] == "Visible defect"
    assert not review.db_path.exists(), "GET packet discovery must not create a ratings database"


def test_explicit_human_post_creates_own_partial_draft_and_other_sessions_cannot_read_it(tmp_path):
    review = store(tmp_path)
    with TestClient(app_with(review)) as alice:
        created = alice.post(
            "/api/development-review/sessions",
            json={"set_id": "pilot-review-v1", "reviewer_id": "alice", "reviewer_kind": "human_self_reported"},
        )
        assert created.status_code == 201
        assert COOKIE_NAME in created.headers["set-cookie"].lower()
        assert "Max-Age=28800" in created.headers["set-cookie"]
        session_id = created.json()["session_id"]
        assert stat.S_IMODE(review.db_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(review.private_root.stat().st_mode) == 0o700
        draft = alice.put(
            "/api/development-review/sessions/%s/ratings/review-clip-a?set_id=pilot-review-v1" % session_id,
            json={"progress": 4, "evidence_frame_indices": [1], "observable_reason": "Drawer is nearly closed."},
        )
        assert draft.status_code == 200
        assert draft.json()["draft"] == {
            "progress": 4,
            "evidence_frame_indices": [1],
            "observable_reason": "Drawer is nearly closed.",
        }
        own = alice.get("/api/development-review/sets/pilot-review-v1/clips?session_id=" + session_id).json()
        assert own["clips"][0]["draft"]["progress"] == 4
        assert own["progress"] == {"saved_drafts": 1, "completed": 0, "total": 1}

    with TestClient(app_with(review)) as bob:
        created = bob.post(
            "/api/development-review/sessions",
            json={"set_id": "pilot-review-v1", "reviewer_id": "bob", "reviewer_kind": "model_assisted"},
        )
        bob_session = created.json()["session_id"]
        assert bob.get(
            "/api/development-review/sessions/%s/ratings/review-clip-a?set_id=pilot-review-v1" % bob_session
        ).json()["draft"] is None
        assert bob.get(
            "/api/development-review/sessions/%s/ratings/review-clip-a?set_id=pilot-review-v1" % session_id
        ).status_code == 403


def test_rejects_unsafe_origin_unknown_clip_path_traversal_and_invalid_entered_values(tmp_path):
    with TestClient(app_with(store(tmp_path))) as client:
        response = client.post(
            "/api/development-review/sessions",
            headers={"Origin": "https://evil.example"},
            json={"set_id": "pilot-review-v1", "reviewer_id": "alice", "reviewer_kind": "human_self_reported"},
        )
        assert response.status_code == 403
        cross_port = client.post(
            "/api/development-review/sessions",
            headers={"Origin": "http://127.0.0.1:9999", "Host": "127.0.0.1:8787"},
            json={"set_id": "pilot-review-v1", "reviewer_id": "alice", "reviewer_kind": "human_self_reported"},
        )
        assert cross_port.status_code == 403
        created = client.post(
            "/api/development-review/sessions",
            json={"set_id": "pilot-review-v1", "reviewer_id": "alice", "reviewer_kind": "human_self_reported"},
        ).json()
        session_id = created["session_id"]
        assert client.get("/api/development-review/sets/../clips").status_code in {404, 422}
        assert client.put(
            "/api/development-review/sessions/%s/ratings/not-a-clip?set_id=pilot-review-v1" % session_id,
            json={"progress": 2},
        ).status_code == 404
        assert client.put(
            "/api/development-review/sessions/%s/ratings/review-clip-a?set_id=pilot-review-v1" % session_id,
            json={"integrity": "invented"},
        ).status_code == 422
        assert client.put(
            "/api/development-review/sessions/%s/ratings/review-clip-a?set_id=pilot-review-v1" % session_id,
            json={"evidence_frame_indices": [99]},
        ).status_code == 422


def test_expired_session_and_private_symlink_are_rejected(tmp_path):
    review = store(tmp_path)
    with TestClient(app_with(review)) as client:
        session_id = client.post(
            "/api/development-review/sessions",
            json={"set_id": "pilot-review-v1", "reviewer_id": "alice", "reviewer_kind": "human_self_reported"},
        ).json()["session_id"]
        with review._connect() as connection:
            connection.execute("UPDATE sessions SET expires_at = '2000-01-01T00:00:00Z' WHERE session_id = ?", (session_id,))
        assert client.get(
            "/api/development-review/sessions/%s/ratings/review-clip-a?set_id=pilot-review-v1" % session_id
        ).status_code == 403

    private_link = tmp_path / "private-link"
    private_link.symlink_to(tmp_path / "served-data", target_is_directory=True)
    try:
        DevelopmentReviewStore(
            repo_root=tmp_path / "repo",
            review_root=tmp_path / "repo" / "data" / "live-integrated" / "review",
            served_root=tmp_path / "served-data",
            private_root=private_link,
        )
        assert False, "private symlink must be refused"
    except ValueError:
        pass


def test_explicit_null_clears_old_labels_and_merged_contradictions_are_rejected(tmp_path):
    with TestClient(app_with(store(tmp_path))) as client:
        session_id = client.post(
            "/api/development-review/sessions",
            json={"set_id": "pilot-review-v1", "reviewer_id": "alice", "reviewer_kind": "human_self_reported"},
        ).json()["session_id"]
        url = "/api/development-review/sessions/%s/ratings/review-clip-a?set_id=pilot-review-v1" % session_id
        complete = {
            "integrity": "intact",
            "collision": "none_visible",
            "progress": 5,
            "completion_evidence": "met",
            "evidence_frame_indices": [],
            "observable_reason": "",
        }
        assert client.put(url, json=complete).status_code == 200
        # A partial update is validated against the retained draft; this
        # contradiction cannot silently turn a completed rating into nonsense.
        assert client.put(url, json={"completion_evidence": "not_met"}).status_code == 422
        cleared = client.put(url, json={"integrity": None, "progress": None, "completion_evidence": None})
        assert cleared.status_code == 200
        assert "integrity" not in cleared.json()["draft"]
        assert "progress" not in cleared.json()["draft"]
        assert "completion_evidence" not in cleared.json()["draft"]


def test_packet_timestamps_must_increase_after_frame_index_sort_and_bad_sets_are_skipped(tmp_path):
    review = store(tmp_path)
    bad_timestamp = packet("review-clip-b")
    bad_timestamp["sampled_frames"][1]["timestamp"] = bad_timestamp["sampled_frames"][0]["timestamp"]
    worksheet = review.review_root / "bad-time" / "blank-worksheets.jsonl"
    worksheet.parent.mkdir()
    worksheet.write_text(json.dumps(bad_timestamp) + "\n", encoding="utf-8")
    missing = review.review_root / "missing-worksheet"
    missing.mkdir()
    assert review.list_sets() == [
        {"set_id": "pilot-review-v1", "title": "pilot-review-v1", "clip_count": 1, "status": "development_review_only"}
    ]


def test_malformed_json_origin_port_and_non_scalar_enum_values_are_client_errors(tmp_path):
    review = store(tmp_path)
    with TestClient(app_with(review)) as client:
        assert client.post(
            "/api/development-review/sessions",
            content="{not-json",
            headers={"content-type": "application/json"},
        ).status_code == 422
        assert client.post(
            "/api/development-review/sessions",
            headers={"Origin": "http://127.0.0.1:not-a-port"},
            json={"set_id": "pilot-review-v1", "reviewer_id": "alice", "reviewer_kind": "human_self_reported"},
        ).status_code == 403
        session_id = client.post(
            "/api/development-review/sessions",
            json={"set_id": "pilot-review-v1", "reviewer_id": "alice", "reviewer_kind": "human_self_reported"},
        ).json()["session_id"]
        url = "/api/development-review/sessions/%s/ratings/review-clip-a?set_id=pilot-review-v1" % session_id
        assert client.put(url, json={"integrity": ["intact"]}).status_code == 422
        assert client.put(url, content="not-json", headers={"content-type": "application/json"}).status_code == 422


def test_empty_reason_or_evidence_is_saved_as_a_draft_but_never_counted_complete(tmp_path):
    review = store(tmp_path)
    with TestClient(app_with(review)) as client:
        session_id = client.post(
            "/api/development-review/sessions",
            json={"set_id": "pilot-review-v1", "reviewer_id": "alice", "reviewer_kind": "human_self_reported"},
        ).json()["session_id"]
        url = "/api/development-review/sessions/%s/ratings/review-clip-a?set_id=pilot-review-v1" % session_id
        saved = {
            "integrity": "intact",
            "collision": "none_visible",
            "progress": 5,
            "completion_evidence": "met",
            "evidence_frame_indices": [],
            "observable_reason": "",
        }
        assert client.put(url, json=saved).status_code == 200
        assert client.get("/api/development-review/sets/pilot-review-v1/clips?session_id=" + session_id).json()["progress"]["completed"] == 0
        assert client.put(url, json={"evidence_frame_indices": [15], "observable_reason": "Visible final drawer state."}).status_code == 200
        assert client.get("/api/development-review/sets/pilot-review-v1/clips?session_id=" + session_id).json()["progress"]["completed"] == 1


def test_private_storage_is_revalidated_before_database_reads_after_a_path_swap(tmp_path):
    review = store(tmp_path)
    with TestClient(app_with(review)) as client:
        session_id = client.post(
            "/api/development-review/sessions",
            json={"set_id": "pilot-review-v1", "reviewer_id": "alice", "reviewer_kind": "human_self_reported"},
        ).json()["session_id"]
        moved = tmp_path / "moved-private"
        review.private_root.rename(moved)
        review.private_root.symlink_to(tmp_path / "served-data", target_is_directory=True)
        response = client.get(
            "/api/development-review/sessions/%s/ratings/review-clip-a?set_id=pilot-review-v1" % session_id
        )
        assert response.status_code == 422
