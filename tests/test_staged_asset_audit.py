"""Fail-closed tests for the read-only cluster staging audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cluster.audit_staged_assets import AuditError, audit_manifests, write_immutable_report


REVISION = "a" * 40
CODE_REVISION = "b" * 40


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def download_manifest(root: Path, *, asset_plan_revision=None, path="weights.bin") -> tuple[Path, Path]:
    destination = root / "models" / "openvla--openvla-7b"
    data = b"pinned model bytes"
    staged = destination / path
    write(staged, data)
    record = {
        "path": path,
        "bytes": len(data),
        "sha256": digest(data),
        "upstream_sha256": digest(data),
        "verified": True,
    }
    manifest = {
        "repo": "openvla/openvla-7b",
        "repo_type": "model",
        "revision": REVISION,
        "destination": str(destination),
        "files": [path],
        "file_records": [record],
    }
    if asset_plan_revision is not None:
        manifest["asset_plan"] = {"revision": asset_plan_revision}
    manifest_path = root / "evidence" / "openvla--openvla-7b-download.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, staged


def test_download_manifest_reports_staged_integrity_but_not_full_asset_verification(tmp_path):
    root = tmp_path / "plumb"
    root.mkdir()
    manifest, _ = download_manifest(root)
    report = audit_manifests(root, [manifest])
    asset = report["assets"][0]
    assert asset["staged_integrity"]["status"] == "pass"
    assert asset["staged_integrity"]["publisher_digest_records"] == 1
    assert asset["fully_verified"] is False
    proposal = asset["partial_asset_lock_proposal"]
    assert proposal["asset_id"] == "openvla-7b"
    assert proposal["sha256"] is None
    assert proposal["container_digest"] is None
    assert proposal["compatibility_profile_id"] is None
    assert proposal["verified"] is False


def test_audit_report_is_new_no_clobber_evidence_only(tmp_path):
    root = tmp_path / "plumb"
    root.mkdir()
    manifest, _ = download_manifest(root)
    report = audit_manifests(root, [manifest])
    output = root / "evidence" / "staged-asset-audit.json"
    written = write_immutable_report(root, output, report)
    assert written == output
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["schema"] == "plumb-staged-asset-audit-v1"
    with pytest.raises(FileExistsError, match="overwrite"):
        write_immutable_report(root, output, report)
    assert not (root / "assets.lock.json").exists()
    assert not (root / "protocol.json").exists()


def test_digest_mismatch_is_reported_as_failed_staged_integrity(tmp_path):
    root = tmp_path / "plumb"
    root.mkdir()
    manifest, staged = download_manifest(root)
    staged.write_bytes(b"tampered")
    asset = audit_manifests(root, [manifest])["assets"][0]
    assert asset["staged_integrity"]["status"] == "fail"
    assert any("sha256 mismatch" in error for record in asset["files"] for error in record["errors"])


def test_path_escape_and_embedded_revision_mismatch_fail_closed(tmp_path):
    root = tmp_path / "plumb"
    root.mkdir()
    manifest, _ = download_manifest(root, asset_plan_revision="c" * 40, path="../outside.bin")
    asset = audit_manifests(root, [manifest])["assets"][0]
    assert asset["staged_integrity"]["status"] == "fail"
    assert any("escapes" in error or "revision does not match" in error for error in asset["staged_integrity"]["errors"] + asset["files"][0]["errors"])


def octo_manifest(root: Path, *, symlink_t5=False) -> Path:
    model_root = root / "models" / "rail-berkeley--octo-small"
    checkpoint = b"octo checkpoint"
    write(model_root / "270000" / "default" / "checkpoint", checkpoint)
    t5_root = root / "models" / ".hf-octo" / "hub" / "models--t5-base" / "snapshots" / REVISION
    token = b"tokenizer"
    t5_file = t5_root / "tokenizer.json"
    write(t5_file, token)
    if symlink_t5:
        external = root / "outside-tokenizer"
        external.write_bytes(token)
        t5_file.unlink()
        t5_file.symlink_to(external)
    manifest = {
        "schema": "plumb-octo-assets-v1",
        "model_id": "rail-berkeley/octo-small",
        "checkpoint_revision": REVISION,
        "code_revision": CODE_REVISION,
        "files": [{"relative_path": "270000/default/checkpoint", "bytes": len(checkpoint), "sha256": digest(checkpoint)}],
        "t5": {
            "model_id": "t5-base",
            "revision": REVISION,
            "tokenizer_path": str(t5_root),
            "files": [{"relative_path": "tokenizer.json", "bytes": len(token), "sha256": digest(token)}],
            "full_t5_model_weights_downloaded": False,
        },
    }
    path = model_root / "PLUMB-ASSET-MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_octo_core_stays_auditable_when_symlinked_t5_support_is_blocked(tmp_path):
    root = tmp_path / "plumb"
    root.mkdir()
    manifest = octo_manifest(root, symlink_t5=True)
    asset = audit_manifests(root, [manifest])["assets"][0]
    assert asset["staged_integrity"]["status"] == "pass"
    support = asset["support_components"]["t5_tokenizer"]
    assert support["status"] == "blocked"
    assert "symlink" in support["files"][0]["errors"][0]


def test_core_staged_symlink_is_never_followed(tmp_path):
    root = tmp_path / "plumb"
    root.mkdir()
    manifest, staged = download_manifest(root)
    external = root / "external-bytes"
    external.write_bytes(staged.read_bytes())
    staged.unlink()
    staged.symlink_to(external)
    asset = audit_manifests(root, [manifest])["assets"][0]
    assert asset["staged_integrity"]["status"] == "fail"
    assert "symlink" in asset["files"][0]["errors"][0]


def test_report_path_must_stay_directly_under_cluster_evidence(tmp_path):
    root = tmp_path / "plumb"
    root.mkdir()
    manifest, _ = download_manifest(root)
    report = audit_manifests(root, [manifest])
    with pytest.raises(AuditError, match="beneath"):
        write_immutable_report(root, tmp_path / "audit.json", report)
