import hashlib
import json

import pytest

from cluster.audit_bridge_metadata import DATASET, MetadataAuditError, audit

REVISION = "a" * 40


def fixture(tmp_path, *, gripper_max=1.112, duplicate=False, count=2):
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {"total_episodes": count, "total_frames": 4, "total_tasks": 2, "fps": 5,
            "features": {"observation.state": {"shape": [8]}, "action": {"shape": [7]}}}
    stats = {"observation.state": {"min": [0] * 8, "max": [1] * 7 + [gripper_max]}}
    tasks = [{"task_index": 0, "task": "Open the drawer"}, {"task_index": 1, "task": "put carrot in basket"}]
    episodes = [{"episode_index": 0, "tasks": ["Open the drawer"], "length": 2},
                {"episode_index": 0 if duplicate else 1, "tasks": ["put carrot in basket"], "length": 2}]
    files = {"info.json": json.dumps(info), "stats.json": json.dumps(stats),
             "tasks.jsonl": "\n".join(json.dumps(v) for v in tasks),
             "episodes.jsonl": "\n".join(json.dumps(v) for v in episodes)}
    records = []
    for name, text in files.items():
        (meta / name).write_text(text)
        records.append({"path": "meta/" + name, "bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest()})
    manifest = tmp_path / "download.json"
    manifest.write_text(json.dumps({"repo": DATASET, "repo_type": "dataset", "revision": REVISION, "file_records": records}))
    return root, manifest


def test_metadata_is_not_a_start_or_calibration_and_does_not_rescale(tmp_path):
    root, manifest = fixture(tmp_path)
    result = audit(root, manifest, expected_revision=REVISION)
    assert result["source_counts"] == {"episodes": 2, "tasks": 2, "frames": 4}
    assert not result["qualified"] and result["status"] == "needs_review"
    assert result["selected_starts"] == [] and not result["gripper_conversion_applied"]
    assert result["source_gripper_state_range"] == [0, 1.112]
    assert any("gripper_state_range_conflicts" in reason for reason in result["blockers"])
    assert result["task_coverage"]["open_drawer"]["exact_instruction_episode_count"] == 1
    assert result["task_coverage"]["to_basket"]["exact_instruction_episode_count"] == 0
    assert result["task_coverage"]["to_basket"]["discovery_episode_count"] == 1
    assert not result["task_coverage"]["to_basket"]["discovery_is_eligibility"]


def test_small_gripper_range_does_not_establish_parity(tmp_path):
    root, manifest = fixture(tmp_path, gripper_max=.39)
    result = audit(root, manifest, expected_revision=REVISION)
    assert not result["qualified"] and result["blockers"]
    assert "scene_reset_camera_and_object_parity_unreviewed" in result["blockers"]


def test_incomplete_index_is_reported_not_padded(tmp_path):
    root, manifest = fixture(tmp_path, count=100)
    result = audit(root, manifest, expected_revision=REVISION)
    assert "metadata_count_mismatch:total_episodes" in result["blockers"]
    assert result["source_counts"]["episodes"] == 2


def test_duplicate_episode_cannot_inflate_coverage(tmp_path):
    root, manifest = fixture(tmp_path, duplicate=True)
    with pytest.raises(MetadataAuditError, match="duplicate"):
        audit(root, manifest, expected_revision=REVISION)


def test_modified_metadata_refused(tmp_path):
    root, manifest = fixture(tmp_path)
    (root / "meta/info.json").write_text("{}")
    with pytest.raises(MetadataAuditError, match="does not match"):
        audit(root, manifest, expected_revision=REVISION)


def test_wrong_revision_refused(tmp_path):
    root, manifest = fixture(tmp_path)
    with pytest.raises(MetadataAuditError, match="revision mismatch"):
        audit(root, manifest, expected_revision="b" * 40)


def test_symlinked_metadata_refused(tmp_path):
    root, manifest = fixture(tmp_path)
    source = root / "meta/info.json"
    other = tmp_path / "original-info.json"
    source.rename(other)
    source.symlink_to(other)
    with pytest.raises(MetadataAuditError, match="symlink"):
        audit(root, manifest, expected_revision=REVISION)


def test_report_is_exclusive_write(tmp_path, monkeypatch):
    from cluster.audit_bridge_metadata import main
    root, manifest = fixture(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["audit", "--dataset-root", str(root), "--download-manifest", str(manifest),
                                   "--revision", REVISION, "--output", str(output)])
    assert main() == 0
    before = output.read_bytes()
    assert main() == 2 and output.read_bytes() == before


def test_unlabeled_episodes_are_counted_but_never_matched(tmp_path):
    root, manifest = fixture(tmp_path)
    for name in ("tasks.jsonl", "episodes.jsonl"):
        path = root / "meta" / name
        path.write_text(path.read_text().replace("Open the drawer", ""))
    payload = json.loads(manifest.read_text())
    for record in payload["file_records"]:
        raw = (root / record["path"]).read_bytes()
        record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    manifest.write_text(json.dumps(payload))
    result = audit(root, manifest, expected_revision=REVISION)
    assert result["unlabeled_episode_count"] == 1
    assert result["task_coverage"]["open_drawer"]["exact_instruction_episode_count"] == 0
    assert "source_contains_unlabeled_episodes" in result["blockers"]
