"""Contract tests for safe scenario import and immutable scenario freezes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plumb.scenarios import (
    COST_CONFIRMATION_PANEL,
    DEVELOPMENT_PANEL,
    HELDOUT_CALIBRATION_PANEL,
    PRIMARY_PANEL,
    PRIMARY_POLICY_IDS,
    TASKS,
    ScenarioError,
    freeze_scenario_manifest,
    import_scenario_manifest,
    load_explicit_source_manifest,
    load_frozen_scenario_manifest,
    scenario_readiness_report,
    sha256_file,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _start(task: str, ordinal: int, *, panel: str = PRIMARY_PANEL) -> dict:
    prefix = "{0}-{1}-{2:03d}".format(task, panel, ordinal)
    return {
        "task": task,
        "start_id": "start-" + prefix,
        "start_lineage_id": "lineage-" + prefix,
        "source_dataset": "bridge_orig_lerobot",
        "source_revision": "frozen-source-revision",
        "episode_id": "episode-" + prefix,
        "frame_id": "frame-000120",
        "image_timestamp": "2026-01-01T00:00:00Z",
        "state_timestamp": "2026-01-01T00:00:00Z",
        "image_hash": None,
        "state_hash": None,
        "state_convention": "Bridge state (x, y, z, rx, ry, rz, 0, gripper)",
        "camera": "wrist_rgb",
        "crop": {"x": 0, "y": 0, "width": 256, "height": 256},
        "camera_calibration": {"calibration_id": "wrist-cal-v1", "hash": "declared-in-source-record"},
        "scene": "Bridge fixture scene",
        "scene_configuration": {"objects": ["drawer", "eggplant", "basket", "sink", "cloth"]},
        "initial_state_stratum": "documented-source-stratum",
        "instruction": {
            "close_drawer": "Close the drawer",
            "open_drawer": "Open the drawer",
            "to_basket": "Put the eggplant in the yellow basket",
            "to_sink": "Put the eggplant in the blue sink",
            "fold_cloth": "fold the cloth from top right to bottom left",
        }[task],
        "goal_reference_hashes": [],
        "goal_reference_provenance": {"source": "explicit-goal-record"},
    }


def _source_and_scenario(root: Path) -> tuple[Path, Path]:
    # The importer only hashes these bytes.  They deliberately do not need a
    # decoder, which keeps source ingestion away from untrusted object formats.
    (root / "frame.png").write_bytes(b"not-decoded-image-bytes")
    (root / "goal.png").write_bytes(b"not-decoded-goal-bytes")
    _write_json(root / "state.json", {"bridge_state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.39]})
    frame_hash = sha256_file(root / "frame.png")
    goal_hash = sha256_file(root / "goal.png")
    state_hash = sha256_file(root / "state.json")

    task_groups = {}
    source_records = []
    for task in TASKS:
        panels = {
            DEVELOPMENT_PANEL: [_start(task, 0, panel=DEVELOPMENT_PANEL)],
            HELDOUT_CALIBRATION_PANEL: [_start(task, 0, panel=HELDOUT_CALIBRATION_PANEL)],
            COST_CONFIRMATION_PANEL: [_start(task, 0, panel=COST_CONFIRMATION_PANEL)],
        }
        starts = [_start(task, index) for index in range(50)]
        task_groups[task] = {"starts": starts, "panels": panels}
        for panel, rows in [(PRIMARY_PANEL, starts), *panels.items()]:
            for row in rows:
                row["image_hash"] = frame_hash
                row["state_hash"] = state_hash
                row["goal_reference_hashes"] = [goal_hash]
                source_records.append(
                    {
                        "source_dataset": row["source_dataset"],
                        "source_revision": row["source_revision"],
                        "episode_id": row["episode_id"],
                        "frame_id": row["frame_id"],
                        "start_lineage_id": row["start_lineage_id"],
                        "image_timestamp": row["image_timestamp"],
                        "state_timestamp": row["state_timestamp"],
                        "state_convention": row["state_convention"],
                        "image": {"path": "frame.png", "media_type": "image/png", "sha256": frame_hash},
                        "state": {"path": "state.json", "media_type": "application/json", "sha256": state_hash},
                        "goal_references": [
                            {"path": "goal.png", "media_type": "image/png", "sha256": goal_hash}
                        ],
                    }
                )
    source_path = root / "sources.json"
    _write_json(
        source_path,
        {
            "schema_version": 1,
            "source_manifest_id": "bridge-source-manifest-v1",
            "provenance_kind": "real_robot",
            "records": source_records,
        },
    )
    scenario_path = root / "scenario.json"
    _write_json(
        scenario_path,
        {
            "schema_version": 1,
            "manifest_id": "bridge-five-task-panel-v1",
            "provenance_kind": "real_robot",
            "source_manifest_sha256": sha256_file(source_path),
            "shared_policy_ids": list(PRIMARY_POLICY_IDS),
            "tasks": task_groups,
        },
    )
    return scenario_path, source_path


def test_import_freeze_and_readiness_preserve_provenance_without_qualifying(tmp_path: Path) -> None:
    scenario_path, source_path = _source_and_scenario(tmp_path)
    imported = import_scenario_manifest(scenario_path, source_path)
    assert imported["source_manifest_id"] == "bridge-source-manifest-v1"
    assert imported["source_manifest_sha256"] == sha256_file(source_path)

    frozen_path = tmp_path / "frozen-panel.json"
    frozen = freeze_scenario_manifest(imported, frozen_path)
    assert frozen_path.stat().st_mode & 0o222 == 0
    assert load_frozen_scenario_manifest(frozen_path)["sha256"] == frozen["sha256"]
    report = scenario_readiness_report(frozen)
    assert report["status"] == "ready_for_gate_c_evidence"
    assert report["gate_c"]["status"] == "not_run"
    assert report["gate_c"]["qualified"] is False
    assert all(row["panel_counts"][PRIMARY_PANEL] == 50 for row in report["tasks"].values())


def test_import_detects_tampered_source_artifacts(tmp_path: Path) -> None:
    scenario_path, source_path = _source_and_scenario(tmp_path)
    (tmp_path / "state.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ScenarioError, match="SHA-256"):
        import_scenario_manifest(scenario_path, source_path)


def test_source_manifest_rejects_pickle_like_artifacts_without_deserializing(tmp_path: Path) -> None:
    scenario_path, source_path = _source_and_scenario(tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    unsafe = tmp_path / "unsafe.pkl"
    unsafe.write_bytes(b"not ever unpickled")
    source["records"][0]["state"] = {
        "path": unsafe.name,
        "media_type": "application/json",
        "sha256": sha256_file(unsafe),
    }
    _write_json(source_path, source)
    with pytest.raises(ScenarioError, match="pickle/checkpoint"):
        load_explicit_source_manifest(source_path)
    # The scenario itself now has an old source-manifest hash; the failure above
    # proves unsafe content is rejected before any such object can be loaded.
    assert scenario_path.is_file()


def test_lineage_reuse_across_calibration_panel_is_rejected(tmp_path: Path) -> None:
    scenario_path, source_path = _source_and_scenario(tmp_path)
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    primary = scenario["tasks"]["close_drawer"]["starts"][0]
    development = scenario["tasks"]["close_drawer"]["panels"][DEVELOPMENT_PANEL][0]
    development["start_lineage_id"] = primary["start_lineage_id"]
    _write_json(scenario_path, scenario)
    with pytest.raises(ScenarioError, match="lineage"):
        import_scenario_manifest(scenario_path, source_path)


def test_readiness_reports_missing_disjoint_panels_instead_of_inventing_them(tmp_path: Path) -> None:
    scenario_path, source_path = _source_and_scenario(tmp_path)
    imported = import_scenario_manifest(scenario_path, source_path)
    imported["tasks"]["to_sink"]["panels"][COST_CONFIRMATION_PANEL] = []
    report = scenario_readiness_report(imported)
    assert report["status"] == "blocked"
    assert "cost_confirmation" in " ".join(report["blockers"])
    with pytest.raises(ScenarioError, match="cost_confirmation"):
        freeze_scenario_manifest(imported, tmp_path / "should-not-freeze.json")
