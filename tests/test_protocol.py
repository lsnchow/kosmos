"""Protocol, preregistration, asset-lock and analysis-manifest behaviour.

The theme of these tests is that the module refuses to overclaim: a local hash is
not preregistration, an incomplete asset row is not verified, and a partially
identified run cannot be manifested.
"""
from __future__ import annotations

import json

import pytest
import plumb.protocol as protocol_module

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
    discover_preregistration,
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
        remote_tag_verified=True,
        external_timestamp="2026-09-19T04:00:00-04:00",
    )
    assert full.status == "registered"
    unsigned = PreregistrationRecord(
        protocol_sha256=digest,
        uri="https://example.com#refs/tags/x",
        commit="abc",
        external_timestamp="2026-09-19T04:00:00-04:00",
    )
    assert unsigned.status == "unregistered"
    assert "signature was not cryptographically verified" in unsigned.blocking_reasons()[0]


def _prereg_git_responses(protocol_digest, *, remote_tag_oid="tag-oid", signature_looking_text=False):
    """A read-only git view for discovery tests; no key or remote is needed."""

    tag = "prereg/protocol-v1"
    tag_ref = "refs/tags/" + tag
    responses = {
        ("tag", "-l", tag, "--format=%(contents)"): (
            "protocol_sha256 "
            + protocol_digest
            + ("\n-----BEGIN PGP SIGNATURE-----\nforged text" if signature_looking_text else "")
        ),
        ("rev-parse", tag_ref): "tag-oid",
        ("cat-file", "-t", tag_ref): "tag",
        ("rev-list", "-n", "1", tag): "commit-oid",
        ("ls-remote", "--tags", "origin", tag_ref): "%s\t%s" % (remote_tag_oid, tag_ref),
        ("remote", "get-url", "origin"): "https://example.test/plumb.git",
        ("tag", "-l", tag, "--format=%(taggerdate:iso-strict)"): "2026-09-19T04:00:00+00:00",
    }

    def fake_git(args, cwd=None):
        return responses.get(tuple(args))

    return fake_git


def test_discovery_rejects_signature_looking_text_when_git_verify_tag_fails(monkeypatch, tmp_path):
    digest = "sha256:" + "a" * 64
    fake_git = _prereg_git_responses(digest, signature_looking_text=True)
    # This was the old bypass: the annotation could simply contain these words.
    monkeypatch.setattr(protocol_module, "_git", fake_git)
    monkeypatch.setattr(protocol_module, "_git_verify_tag", lambda tag, cwd=None: False)
    record = discover_preregistration(digest, "prereg/protocol-v1", repo_root=tmp_path)
    assert record.status == "unregistered"
    assert record.signed is False
    assert record.remote_tag_verified is False
    assert "verify-tag" in record.notes[0]


def test_discovery_rejects_a_remote_tag_with_a_different_tag_object_oid(monkeypatch, tmp_path):
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(protocol_module, "_git", _prereg_git_responses(digest, remote_tag_oid="other-tag-oid"))
    monkeypatch.setattr(protocol_module, "_git_verify_tag", lambda tag, cwd=None: True)
    record = discover_preregistration(digest, "prereg/protocol-v1", repo_root=tmp_path)
    assert record.status == "unregistered"
    assert record.signed is True
    assert record.remote_tag_verified is False
    assert record.tag_object_id == "tag-oid"
    assert record.remote_tag_object_id == "other-tag-oid"
    assert record.uri == "https://example.test/plumb.git#refs/tags/prereg/protocol-v1"
    assert "remote tag object" in record.blocking_reasons()[-1]
    assert "does not match" in record.notes[0]


def test_discovery_registers_only_a_cryptographically_verified_matching_tag(monkeypatch, tmp_path):
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(protocol_module, "_git", _prereg_git_responses(digest))
    monkeypatch.setattr(protocol_module, "_git_verify_tag", lambda tag, cwd=None: True)
    record = discover_preregistration(digest, "prereg/protocol-v1", repo_root=tmp_path)
    assert record.status == "registered"
    assert record.signed is True
    assert record.remote_tag_verified is True
    assert record.tag_object_id == record.remote_tag_object_id == "tag-oid"


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


def test_the_acquisition_plan_covers_every_spec_asset_and_starts_unresolved():
    """Assert on the artifact (repo id), not on an internal row key.

    Two plans contribute rows under different names, so keying the assertion on
    ``repo_id`` is what actually states the requirement: every asset spec section 1
    names must be planned for.
    """

    lock = build_asset_lock()
    repos = {record.repo_id.lower() for record in lock.assets.values()}
    for expected in (
        "openvla/openvla-7b",
        "rail-berkeley/octo-small",
        "stanford-iliad/minivla-vq-bridge-prismatic",
        "stanford-iliad/pretrain_vq",
        "allenzren/open-pi-zero",
        "kvablack/susie",
        "patreya/gcbc-bridge",
        "nvidia/cosmos3-nano",
        "qwen/qwen2.5-vl-7b-instruct",
        "facebook/vjepa2-vitl-fpc64-256",
        "zhouzypaul/auto_eval",
        "ipec-community/bridge_orig_lerobot",
    ):
        assert expected in repos, expected
    assert lock.verified_asset_ids == (), "a plan proves nothing is downloaded"
    assert lock.to_mapping()["summary"]["status"] == "incomplete"


def test_one_artifact_produces_exactly_one_lock_row():
    """The two plans overlap; a duplicated row would inflate the lock."""

    lock = build_asset_lock()
    seen = [(r.repo_id.lower(), (r.file_path or "").lower()) for r in lock.assets.values()]
    assert len(seen) == len(set(seen)), "an artifact appears under two keys"


def test_the_unlicensed_vq_asset_is_flagged_as_non_redistributable():
    """`Stanford-ILIAD/pretrain_vq` declares no license (spec section 1)."""

    lock = build_asset_lock()
    matches = [r for r in lock.assets.values() if r.repo_id.lower() == "stanford-iliad/pretrain_vq"]
    assert matches, "the VQ asset must be planned for; MiniVLA needs it"
    record = matches[0]
    assert record.license is None, "a license that was never verified must not be recorded"
    assert "prohibited" in record.redistribution
    assert record.verified is False


def test_an_advertised_but_unverified_license_never_counts_as_resolved():
    """A repo claiming a license is not the same as having checked it."""

    lock = build_asset_lock()
    advertised = [
        r
        for r in lock.assets.values()
        if any("advertised" in notice for notice in r.notices)
    ]
    assert advertised, "the plan records advertised-but-unverified licenses"
    for record in advertised:
        assert record.license is None
        assert record.access_status != "resolved"
        assert record.verified is False


def test_a_historical_size_estimate_is_never_recorded_as_a_measured_byte_length():
    lock = build_asset_lock()
    for record in lock.assets.values():
        if not record.verified:
            assert record.byte_length is None or record.sha256 is not None, (
                "byte_length means bytes actually retrieved, not a spec-table estimate"
            )


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
