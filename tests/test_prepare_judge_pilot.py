import copy
import json

import pytest

from cluster import prepare_judge_pilot as pilot


def plan_fixture(tmp_path, monkeypatch):
    root = tmp_path / "metadata"
    (root / "meta").mkdir(parents=True)
    rows = [{"episode_index": i, "length": 20, "tasks": ["close the drawer"]} for i in range(24)]
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps(v) for v in rows))
    manifest = tmp_path / "download.json"
    manifest.write_text("{}")
    monkeypatch.setattr(pilot, "audit", lambda *a, **kw: {"input_files": {"test": "unit-only"}})
    return pilot.make_plan(root, manifest)


def test_selection_is_deterministic_disjoint_and_reserved_before_labels(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path, monkeypatch)
    clips = pilot.validate_plan(plan)
    assert len(clips) == 16 and len(set(plan["formal_excluded_lineages"])) == 16
    assert len([c for c in clips if c["cohort"] == "train"]) == 12
    assert len([c for c in clips if c["cohort"] == "development_validation"]) == 4
    assert all("label" not in c and "success" not in c for c in clips)
    again = pilot.make_plan(tmp_path / "metadata", tmp_path / "download.json")
    assert plan["clips"] == again["clips"]
    assert not plan["qualified"]


@pytest.mark.parametrize("change", ["repeat", "reservation", "escape", "cohort"])
def test_invalid_selection_is_not_silently_fixed(tmp_path, monkeypatch, change):
    plan = copy.deepcopy(plan_fixture(tmp_path, monkeypatch))
    if change == "repeat":
        plan["clips"][1] = dict(plan["clips"][0])
    elif change == "reservation":
        plan["formal_excluded_lineages"].pop()
    elif change == "escape":
        plan["clips"][0]["video_path"] = "../elsewhere.mp4"
    else:
        plan["clips"][0]["cohort"] = "primary"
    with pytest.raises(ValueError):
        pilot.validate_plan(plan)


def test_source_hash_and_containment_are_checked(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"raw-source")
    record = {"bytes": path.stat().st_size, "sha256": pilot.file_sha(path)}
    assert pilot._verified_source(tmp_path, "video.mp4", {"video.mp4": record}) == path
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="disagree"):
        pilot._verified_source(tmp_path, "video.mp4", {"video.mp4": record})


def test_frozen_plan_is_never_overwritten(tmp_path):
    path = tmp_path / "plan.json"
    pilot.freeze_json(path, {"source": 1})
    with pytest.raises(FileExistsError):
        pilot.freeze_json(path, {"source": 2})
    assert json.loads(path.read_text()) == {"source": 1}


def test_download_refuses_existing_receipt(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path, monkeypatch)
    path = tmp_path / "selection.json"
    pilot.freeze_json(path, plan)
    with pytest.raises(ValueError, match="cluster scratch"):
        pilot.download(path, tmp_path)
