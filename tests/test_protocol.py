"""Protocol, preregistration, asset-lock and analysis-manifest behaviour.

The theme of these tests is that the module refuses to overclaim: a local hash is
not preregistration, an incomplete asset row is not verified, and a partially
identified run cannot be manifested.
"""
from __future__ import annotations

import json

import pytest

from plumb.protocol import (
    AssetLock,
    AssetRecord,
    PreregistrationRecord,
    ProtocolDocument,
    ProtocolError,
    amend,
    analysis_options_for_run,
    build_analysis_manifest,
    build_asset_lock,
    default_protocol,
    freeze,
    mdd_options,
)


def _episodes(protocol_hash, cohort="primary", count=2, policies=("OpenVLA",), tasks=("open_drawer",)):
    rows = []
    for policy in policies:
        for task in tasks:
            for index in range(count):
                rows.append(
                    {
                        "episode_id": "ep-%s-%s-%d" % (policy, task, index),
                        "policy": policy,
                        "policy_variant": policy,
                        "task": task,
                        "start_id": "start-%03d" % index,
                        "cohort": cohort,
                        "protocol_hash": protocol_hash,
                        "status": "completed",
                        "validity": "valid",
                        "binary_success": index % 2 == 0,
                    }
                )
    return rows


def test_document_hash_is_stable_and_roundtrips(tmp_path):
    document = default_protocol()
    path = tmp_path / "protocol.json"
    document.save(path)
    reloaded = ProtocolDocument.load(path)
    assert reloaded.sha256 == document.sha256
    assert reloaded.total_planned_episodes == 1500


def test_a_tampered_document_is_rejected_on_load(tmp_path):
    document = default_protocol()
    path = tmp_path / "protocol.json"
    document.save(path)
    payload = json.loads(path.read_text())
    payload["matrix"]["starts_per_task"] = 10
    path.write_text(json.dumps(payload))
    with pytest.raises(ProtocolError, match="hash mismatch"):
        ProtocolDocument.load(path)


def test_freezing_changes_the_hash_so_the_frozen_one_is_what_gets_registered():
    document = default_protocol()
    frozen = freeze(document)
    assert frozen.frozen is True
    assert frozen.sha256 != document.sha256


def test_an_amendment_gets_a_new_id_and_never_overwrites_its_parent():
    document = freeze(default_protocol())
    amended = amend(document, "protocol-v2")
    assert amended.amends == "protocol-v1"
    assert amended.frozen is False
    assert amended.sha256 != document.sha256
    with pytest.raises(ProtocolError, match="new protocol_id"):
        amend(document, "protocol-v1")


def test_a_local_hash_is_not_preregistration():
    document = default_protocol()
    record = document.preregistration_status()
    assert record.status == "unregistered"
    assert "no externally auditable timestamp" in record.blocking_reasons()


def test_preregistration_rejects_a_local_path_and_an_offsetless_timestamp():
    digest = "sha256:" + "a" * 64
    with pytest.raises(ProtocolError, match="remote artifact URI"):
        PreregistrationRecord(protocol_sha256=digest, uri="/tmp/prereg.json")
    with pytest.raises(ProtocolError, match="UTC offset"):
        PreregistrationRecord(protocol_sha256=digest, external_timestamp="2026-09-19T00:00:00")


def test_a_registered_record_requires_uri_commit_and_external_timestamp():
    digest = "sha256:" + "a" * 64
    partial = PreregistrationRecord(protocol_sha256=digest, uri="https://example.com#refs/tags/x", commit="abc")
    assert partial.status == "unregistered"
    full = PreregistrationRecord(
        protocol_sha256=digest,
        uri="https://example.com#refs/tags/x",
        commit="abc",
        signed=True,
        external_timestamp="2026-09-19T04:00:00-04:00",
    )
    assert full.status == "registered"
    unsigned = PreregistrationRecord(
        protocol_sha256=digest,
        uri="https://example.com#refs/tags/x",
        commit="abc",
        external_timestamp="2026-09-19T04:00:00-04:00",
    )
    assert unsigned.status == "registered_unsigned"


def test_an_incomplete_asset_row_is_never_verified():
    record = AssetRecord(asset_id="a", repo_id="r", repo_type="model")
    assert record.verified is False
    assert "sha256" in record.unresolved_fields()
    complete = AssetRecord(
        asset_id="a",
        repo_id="r",
        repo_type="model",
        revision="deadbeef",
        sha256="sha256:" + "b" * 64,
        byte_length=10,
        retrieved_at="2026-09-19T00:00:00Z",
        license="MIT",
        access_status="resolved",
    )
    assert complete.verified is True
    assert complete.unresolved_fields() == ()


def test_a_malformed_digest_does_not_count_as_verified():
    record = AssetRecord(
        asset_id="a",
        repo_id="r",
        repo_type="model",
        revision="x",
        sha256="not-a-digest",
        byte_length=1,
        retrieved_at="now",
        license="MIT",
        access_status="resolved",
    )
    assert record.verified is False
    assert "sha256_malformed" in record.unresolved_fields()


def test_the_discovery_plan_covers_every_spec_asset_and_starts_unresolved():
    lock = build_asset_lock()
    ids = set(lock.assets)
    # Every benchmark policy plus both world models, the judge, the feature
    # diagnostic and both datasets must have a row.
    for expected in (
        "policy.openvla",
        "policy.octo_small",
        "policy.minivla",
        "policy.minivla_vq",
        "policy.open_pi_zero",
        "policy.susie_subgoal",
        "policy.susie_lowlevel",
        "world.cosmos3_nano",
        "world.irasim_bridge",
        "judge.qwen2_5_vl_7b",
        "judge.vjepa2_features",
        "data.auto_eval",
        "data.bridge_orig_lerobot",
    ):
        assert expected in ids, expected
    assert lock.verified_asset_ids == (), "a discovery plan proves nothing is downloaded"
    assert lock.to_mapping()["summary"]["status"] == "incomplete"


def test_the_unlicensed_vq_asset_is_flagged_as_non_redistributable():
    lock = build_asset_lock()
    record = lock.assets["policy.minivla_vq"]
    assert record.license is None
    assert record.redistribution == "prohibited_pending_resolution"
    assert any("no license" in notice for notice in record.notices)


def test_a_retrieved_row_is_not_overwritten_by_the_plan():
    existing = AssetLock()
    existing.add(
        AssetRecord(
            asset_id="policy.openvla",
            repo_id="openvla/openvla-7b",
            repo_type="model",
            revision="47a0ec7f",
            sha256="sha256:" + "c" * 64,
            byte_length=1,
            retrieved_at="2026-09-19T00:00:00Z",
            license="MIT",
            access_status="resolved",
        )
    )
    merged = build_asset_lock(existing)
    assert merged.assets["policy.openvla"].verified is True


def test_asset_lock_roundtrips(tmp_path):
    lock = build_asset_lock()
    path = tmp_path / "assets.lock.json"
    lock.save(path)
    reloaded = AssetLock.load(path)
    assert set(reloaded.assets) == set(lock.assets)


def test_manifest_includes_every_planned_episode_including_failures():
    document = freeze(default_protocol())
    rows = _episodes(document.sha256, count=3)
    rows[0]["status"] = "failed"
    rows[0]["validity"] = "unknown"
    rows[0]["binary_success"] = None
    manifest = build_analysis_manifest(rows, protocol_hash=document.sha256)
    assert len(manifest["allowed_episode_ids"]) == 3, "a failed episode stays in the manifest"
    assert manifest["frozen"] is True
    assert manifest["manifest_sha256"].startswith("sha256:")


def test_manifest_refuses_a_partially_identified_run():
    document = freeze(default_protocol())
    rows = _episodes(document.sha256, count=2)
    rows[1]["protocol_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ProtocolError, match="partially-identified"):
        build_analysis_manifest(rows, protocol_hash=document.sha256)


def test_manifest_refuses_to_pool_two_variants_of_one_policy():
    document = freeze(default_protocol())
    rows = _episodes(document.sha256, count=2)
    rows[1]["policy_variant"] = "OpenVLA-corrected"
    with pytest.raises(ProtocolError, match="two variants"):
        build_analysis_manifest(rows, protocol_hash=document.sha256)


def test_manifest_requires_at_least_one_episode():
    with pytest.raises(ProtocolError, match="at least one"):
        build_analysis_manifest([], protocol_hash="sha256:" + "a" * 64)


def test_analysis_options_enable_the_headline_statistics():
    document = freeze(default_protocol())
    rows = _episodes(document.sha256, count=2, policies=document.policies, tasks=document.tasks)
    options = analysis_options_for_run(document, rows)
    assert options["analysis_mode"] == "primary"
    assert options["reliability"] is True
    assert options["mmrv"] is True
    assert options["advanced_inference"] is True
    assert options["analysis_manifest"]["frozen"] is True
    # MDD must be a mapping carrying explicit baseline rates; measurement
    # refuses to infer them.
    assert isinstance(options["mdd"], dict)
    assert options["mdd"]["baseline_rates"]
    assert options["mdd"]["simulations"] >= 10000


def test_mdd_is_retrospective_and_refuses_unequal_task_sizes():
    document = freeze(default_protocol())
    rows = _episodes(document.sha256, count=2, policies=("OpenVLA",), tasks=document.tasks)
    config = mdd_options(document, rows)
    assert config["enabled"] is True
    assert config["design"] == "retrospective"
    assert len(config["baseline_rates"]) == len(document.tasks)

    rows = [row for row in rows if not (row["task"] == "fold_cloth" and row["start_id"] == "start-001")]
    unequal = mdd_options(document, rows)
    assert unequal["enabled"] is False
    assert "equal task sample sizes" in unequal["reason"]


def test_mdd_reports_disabled_rather_than_guessing_when_a_task_is_empty():
    document = freeze(default_protocol())
    rows = _episodes(document.sha256, count=2, policies=("OpenVLA",), tasks=("open_drawer",))
    config = mdd_options(document, rows)
    assert config["enabled"] is False
    assert "no planned episodes" in config["reason"]


def test_tolerances_carry_the_spec_stated_burst_definition():
    document = default_protocol()
    assert document.tolerances.max_burst_seconds == 60.0
    assert document.tolerances.max_total_demonstration_usd == 11.25
    assert document.tolerances.required_rehearsals == 3
    assert document.tolerances.mdd_simulations_per_gap >= 10000
    assert document.tolerances.bootstrap_replicates >= 10000
    assert document.tolerances.split_half_repetitions >= 1000
