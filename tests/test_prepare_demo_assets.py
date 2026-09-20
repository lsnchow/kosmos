"""Offline tests for the Cosmos demo asset content-lock workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deploy.baseten.prepare_demo_assets import (
    CONTENT_LOCK_SCHEMA,
    COSMOS_ASSET_ID,
    DEMO_ASSET_MANIFEST_SCHEMA,
    DemoAssetPreparationError,
    build_cosmos_content_lock,
    create_verified_demo_asset_manifest,
    load_cosmos_content_lock,
    verify_cosmos_runtime_files,
    write_json_no_clobber,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_CLUSTER_MANIFEST = ROOT / "data" / "baseten-comparison-evidence" / "cosmos-download.json"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source(records: dict[str, bytes]) -> dict:
    return {
        "repo": "nvidia/Cosmos3-Nano",
        "revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92",
        "files": list(records),
        "file_records": [
            {
                "path": path,
                "bytes": len(value),
                "sha256": _digest(value),
                "upstream_sha256": None,
                "verified": False,
            }
            for path, value in records.items()
        ],
    }


def _write_model(root: Path, records: dict[str, bytes]) -> None:
    for relative, value in records.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)


def test_real_cluster_manifest_builds_a_compact_runtime_lock_without_trusting_publisher_boolean():
    lock = build_cosmos_content_lock(REAL_CLUSTER_MANIFEST)

    assert lock["schema_version"] == CONTENT_LOCK_SCHEMA
    assert lock["asset_id"] == COSMOS_ASSET_ID
    assert lock["runtime_file_count"] == 36
    assert lock["runtime_bytes"] == 34922407982
    paths = {item["path"] for item in lock["file_records"]}
    assert "transformer/diffusion_pytorch_model-00001-of-00007.safetensors" in paths
    assert "sound_tokenizer/diffusion_pytorch_model.safetensors" in paths
    assert "README.md" not in paths
    assert not any(path.startswith(("assets/", "images/")) for path in paths)
    config = next(item for item in lock["file_records"] if item["path"] == "config.json")
    assert config["publisher_digest_verified_by_source_manifest"] is False
    # The lack of a publisher/LFS hash is preserved as provenance, but does
    # not remove the recorded cluster-computed digest from the runtime lock.
    assert config["sha256"] == "c32f2468a54542c21946bc8eab6172b911dcec9a7193a94c023ea2d4073bcda6"


def test_verifier_hashes_each_mounted_locked_file_then_mints_the_only_verified_asset_record(tmp_path):
    records = {"config.json": b"config", "transformer/model.safetensors": b"weights"}
    lock = build_cosmos_content_lock(_source(records))
    model_root = tmp_path / "model-cache" / "cosmos3-nano"
    _write_model(model_root, records)
    lock_path = tmp_path / "cosmos-lock.json"
    write_json_no_clobber(lock_path, lock)
    loaded, raw_sha = load_cosmos_content_lock(lock_path)

    report = verify_cosmos_runtime_files(loaded, model_root)
    manifest = create_verified_demo_asset_manifest(
        loaded,
        model_root=model_root,
        content_lock_path="/app/cosmos3-nano-runtime-lock.json",
        content_lock_sha256=raw_sha,
    )

    assert report["status"] == "verified"
    assert report["checked_all_locked_files"] is True
    asset = manifest["assets"][COSMOS_ASSET_ID]
    assert manifest["schema_version"] == DEMO_ASSET_MANIFEST_SCHEMA
    assert asset["verified"] is True
    assert asset["path"] == str(model_root)
    assert asset["content_lock_path"] == "/app/cosmos3-nano-runtime-lock.json"
    assert asset["content_lock_sha256"] == raw_sha
    assert asset["sha256"] == loaded["file_records_sha256"]


@pytest.mark.parametrize("mutation", ("missing", "bytes", "sha256", "escape"))
def test_verifier_rejects_missing_changed_and_escape_paths_instead_of_trusting_a_verified_flag(tmp_path, mutation):
    records = {"config.json": b"config", "transformer/model.safetensors": b"weights"}
    lock = build_cosmos_content_lock(_source(records))
    model_root = tmp_path / "model"
    _write_model(model_root, records)
    lock["verified"] = True  # Must have no effect on actual file verification.
    if mutation == "missing":
        (model_root / "config.json").unlink()
    elif mutation == "bytes":
        (model_root / "config.json").write_bytes(b"too-long")
    elif mutation == "sha256":
        (model_root / "transformer/model.safetensors").write_bytes(b"other!!")
    else:
        lock["file_records"][0]["path"] = "../outside.bin"

    with pytest.raises(DemoAssetPreparationError):
        verify_cosmos_runtime_files(lock, model_root)


def test_verifier_rejects_symlinked_runtime_file_and_no_clobber_writer_preserves_prior_lock(tmp_path):
    records = {"config.json": b"config"}
    lock = build_cosmos_content_lock(_source(records))
    model_root = tmp_path / "model"
    _write_model(model_root, records)
    external = tmp_path / "external-config"
    external.write_bytes(b"config")
    (model_root / "config.json").unlink()
    (model_root / "config.json").symlink_to(external)
    with pytest.raises(DemoAssetPreparationError, match="symlink"):
        verify_cosmos_runtime_files(lock, model_root)

    output = tmp_path / "lock.json"
    write_json_no_clobber(output, lock)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_json_no_clobber(output, lock)
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == CONTENT_LOCK_SCHEMA


def test_lock_rejects_source_manifest_path_escapes_and_records_must_cover_declared_source_files():
    escaped = _source({"config.json": b"config"})
    escaped["files"] = ["../escape"]
    with pytest.raises(DemoAssetPreparationError, match="unsafe"):
        build_cosmos_content_lock(escaped)

    incomplete = _source({"config.json": b"config"})
    incomplete["files"].append("transformer/missing.safetensors")
    with pytest.raises(DemoAssetPreparationError, match="disagree"):
        build_cosmos_content_lock(incomplete)

