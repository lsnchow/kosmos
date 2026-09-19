#!/usr/bin/env python3
"""Read-only audit of pinned cluster-staging manifests.

This tool runs *on the cluster* against an already staged ``.../plumb`` root.
It does not download, SSH, modify a model directory, edit ``assets.lock.json``,
or change ``protocol.json``.  Its only optional write is one no-clobber audit
report under ``<cluster-root>/evidence``.

Two evidence formats are intentionally supported, because they make different
claims:

* ``cluster.download_assets`` manifests bind Hub download file records.
* ``plumb-octo-assets-v1`` binds the Octo checkpoint and its tokenizer support
  files separately.

``staged_integrity`` means the current regular files exactly match the supplied
manifest's sizes/digests and revision bindings.  It is not a claim that a full
runtime/container/compatibility profile was verified.  Those missing facts keep
every proposed AssetLock row ``verified: false``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "plumb-staged-asset-audit-v1"
OCTO_SCHEMA = "plumb-octo-assets-v1"
_SHA256_HEX = set("0123456789abcdef")
_GIT_HEX = set("0123456789abcdef")

# These are proposal targets only.  They keep a real staging report easy to
# reconcile with the current lock without granting the report authority to
# mutate it.  Unknown repositories get a visibly provisional ``staged.*`` ID.
KNOWN_LOCK_ASSET_IDS = {
    "nvidia/Cosmos3-Nano": "cosmos3-nano",
    "openvla/openvla-7b": "openvla-7b",
    "rail-berkeley/octo-small": "octo-small",
    "rail-berkeley/octo-base": "octo-base",
}


class AuditError(ValueError):
    """A path or manifest cannot be audited safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= _SHA256_HEX


def _is_git_revision(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value.lower()) <= _GIT_HEX


def _regular_file_under(root: Path, candidate: Path, *, label: str) -> Path:
    """Return a regular non-symlink file beneath root, without following links."""

    if root.is_symlink():
        raise AuditError("cluster root may not be a symlink: %s" % root)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise AuditError("cluster root is not a regular directory: %s" % root)
    lexical = Path(os.path.abspath(str(candidate)))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise AuditError("%s escapes the cluster root: %s" % (label, candidate)) from error
    if not relative.parts:
        raise AuditError("%s names the cluster root, not a file" % label)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise AuditError("%s is missing: %s" % (label, current)) from error
        if stat.S_ISLNK(mode):
            raise AuditError("%s contains a symlink and is not audited: %s" % (label, current))
    if not current.is_file():
        raise AuditError("%s is not a regular file: %s" % (label, current))
    # This is redundant after walking lstat(), but protects against a raced
    # replacement between checks and makes the containment rule explicit.
    try:
        current.resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise AuditError("%s resolves outside the cluster root: %s" % (label, current)) from error
    return current


def _directory_under(root: Path, candidate: Path, *, label: str) -> Path:
    """Validate an existing non-symlink directory under root."""

    if root.is_symlink():
        raise AuditError("cluster root may not be a symlink: %s" % root)
    root = root.resolve(strict=True)
    lexical = Path(os.path.abspath(str(candidate)))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise AuditError("%s escapes the cluster root: %s" % (label, candidate)) from error
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise AuditError("%s is missing: %s" % (label, current)) from error
        if stat.S_ISLNK(mode):
            raise AuditError("%s contains a symlink and is not audited: %s" % (label, current))
    if not current.is_dir():
        raise AuditError("%s is not a directory: %s" % (label, current))
    return current


def _relative_file(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise AuditError("%s must be a non-empty relative path" % label)
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise AuditError("%s escapes its declared manifest directory: %r" % (label, value))
    return path


def _file_result(root: Path, base: Path, record: Mapping[str, Any], *, path_key: str) -> Dict[str, Any]:
    """Verify one manifest record and return errors instead of raising.

    The caller already selected a safe base.  A malformed relative path still
    becomes a fail-closed record, rather than a filesystem traversal.
    """

    errors: List[str] = []
    relative_value = record.get(path_key)
    try:
        relative = _relative_file(relative_value, label="file record.%s" % path_key)
        actual_path = base.joinpath(*relative.parts)
        path = _regular_file_under(root, actual_path, label="staged file %s" % relative_value)
    except AuditError as error:
        return {"path": relative_value, "status": "fail", "errors": [str(error)]}

    expected_bytes = record.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
        errors.append("manifest bytes must be a non-negative integer")
    elif path.stat().st_size != expected_bytes:
        errors.append("size mismatch: manifest=%d actual=%d" % (expected_bytes, path.stat().st_size))
    expected_digest = record.get("sha256")
    if not _is_sha256(expected_digest):
        errors.append("manifest sha256 is absent or malformed")
        actual_digest = None
    else:
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest.lower():
            errors.append("sha256 mismatch")
    return {
        "path": str(relative),
        "bytes": path.stat().st_size,
        "sha256": actual_digest,
        "status": "pass" if not errors else "fail",
        "errors": errors,
    }


def _partial_lock_proposal(
    *, asset_id: str, repo_id: str, repo_type: str, revision: Optional[str], total_bytes: Optional[int], manifest_path: Path,
    manifest_sha256: str, staged_integrity: Mapping[str, Any], source_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Produce a review-only AssetLock-shaped row without asserting full proof."""

    return {
        "asset_id": asset_id,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        # A multi-file manifest has no single canonical artifact digest.  Never
        # invent one merely to make the primary lock look complete.
        "file_path": None,
        "byte_length": total_bytes,
        "sha256": None,
        "source_url": source_url,
        "retrieved_at": None,
        "license": None,
        "access_status": "unresolved",
        "redistribution": "unresolved",
        "loader_revision": None,
        "container_digest": None,
        "compatibility_profile_id": None,
        "role": "unknown",
        "staged_integrity_manifest": {
            "path": str(manifest_path),
            "sha256": "sha256:" + manifest_sha256,
            "status": staged_integrity.get("status"),
        },
        "verified": False,
        "qualification_verified": False,
        "blocking_reasons": [
            "proposal only: do not merge automatically into assets.lock.json",
            "aggregate file digest intentionally unset",
            "container_digest is unknown",
            "compatibility_profile_id is unknown",
            "license/access review remains unresolved",
        ],
    }


def _audit_download_manifest(root: Path, manifest_path: Path, raw: Mapping[str, Any], manifest_digest: str) -> Dict[str, Any]:
    errors: List[str] = []
    repo = raw.get("repo")
    revision = raw.get("revision")
    if not isinstance(repo, str) or "/" not in repo:
        errors.append("download manifest repo is absent or malformed")
    if not _is_git_revision(revision):
        errors.append("download manifest revision is absent or not a 40-character immutable revision")
    plan_binding = raw.get("asset_plan")
    if isinstance(plan_binding, Mapping) and plan_binding.get("revision") is not None:
        if plan_binding.get("revision") != revision:
            errors.append("download manifest revision does not match its embedded asset_plan revision")
    destination = raw.get("destination")
    try:
        destination_path = _directory_under(root, Path(str(destination)), label="download destination")
    except AuditError as error:
        destination_path = None
        errors.append(str(error))
    if isinstance(repo, str) and destination_path is not None:
        expected_destination = root / "models" / repo.replace("/", "--")
        if destination_path != expected_destination:
            errors.append("download destination does not match the repository's pinned cluster path")
    planned_files = raw.get("files")
    file_records = raw.get("file_records")
    if not isinstance(planned_files, list) or not isinstance(file_records, list) or not file_records:
        errors.append("download manifest needs non-empty files and file_records arrays")
        records: List[Dict[str, Any]] = []
    else:
        expected_paths: List[str] = []
        for item in planned_files:
            try:
                expected_paths.append(str(_relative_file(item, label="download manifest.files")))
            except AuditError as error:
                errors.append(str(error))
        records = []
        seen = set()
        for item in file_records:
            if not isinstance(item, Mapping):
                errors.append("download file_records contains a non-object")
                continue
            path = item.get("path")
            if path in seen:
                errors.append("download file_records has a duplicate path %r" % path)
                continue
            seen.add(path)
            checked = (
                _file_result(root, destination_path, item, path_key="path")
                if destination_path is not None
                else {"path": path, "status": "fail", "errors": ["unsafe or missing destination"]}
            )
            upstream = item.get("upstream_sha256")
            if upstream is not None:
                if not _is_sha256(upstream):
                    checked.setdefault("errors", []).append("upstream_sha256 is malformed")
                elif item.get("sha256") != upstream:
                    checked.setdefault("errors", []).append("manifest sha256 differs from upstream_sha256")
                elif item.get("verified") is not True:
                    checked.setdefault("errors", []).append("publisher digest is present but manifest did not mark it verified")
            elif item.get("verified") is True:
                checked.setdefault("errors", []).append("manifest marks publisher verification without upstream_sha256")
            if checked.get("errors"):
                checked["status"] = "fail"
            records.append(checked)
        if set(expected_paths) != seen:
            errors.append("download files and file_records do not enumerate the same paths")
    status = "pass" if not errors and all(item.get("status") == "pass" for item in records) else "fail"
    publisher_verified = sum(1 for item in file_records or () if isinstance(item, Mapping) and item.get("upstream_sha256"))
    staged_integrity = {
        "status": status,
        "definition": "all declared regular staged files match the supplied download manifest locally",
        "files_checked": len(records),
        "publisher_digest_records": publisher_verified,
        "local_manifest_only_records": max(0, len(records) - publisher_verified),
        "errors": errors,
    }
    total_bytes = sum(item.get("bytes", 0) for item in records if isinstance(item.get("bytes"), int))
    asset_id = KNOWN_LOCK_ASSET_IDS.get(str(repo), "staged." + str(repo or "unknown").replace("/", "_"))
    return {
        "kind": "hub_download_manifest",
        "source_manifest": {"path": str(manifest_path), "sha256": "sha256:" + manifest_digest},
        "repo_id": repo,
        "repo_type": raw.get("repo_type") or "model",
        "revision": revision,
        "staged_integrity": staged_integrity,
        "files": records,
        "fully_verified": False,
        "fully_verified_blockers": [
            "container digest is not recorded by a download manifest",
            "runtime compatibility profile is not recorded by a download manifest",
            "asset lock license/access fields require separate review",
        ],
        "partial_asset_lock_proposal": _partial_lock_proposal(
            asset_id=asset_id,
            repo_id=str(repo or "unknown"),
            repo_type=str(raw.get("repo_type") or "model"),
            revision=revision if isinstance(revision, str) else None,
            total_bytes=total_bytes,
            manifest_path=manifest_path,
            manifest_sha256=manifest_digest,
            staged_integrity=staged_integrity,
            source_url=("https://huggingface.co/%s/tree/%s" % (repo, revision)) if isinstance(repo, str) and isinstance(revision, str) else None,
        ),
    }


def _audit_octo_manifest(root: Path, manifest_path: Path, raw: Mapping[str, Any], manifest_digest: str) -> Dict[str, Any]:
    errors: List[str] = []
    model_id = raw.get("model_id")
    checkpoint_revision = raw.get("checkpoint_revision")
    code_revision = raw.get("code_revision")
    if not isinstance(model_id, str) or model_id not in {"rail-berkeley/octo-small", "rail-berkeley/octo-base"}:
        errors.append("Octo manifest model_id is not a supported pinned Octo asset")
    if not _is_git_revision(checkpoint_revision):
        errors.append("Octo checkpoint_revision is absent or malformed")
    if not _is_git_revision(code_revision):
        errors.append("Octo code_revision is absent or malformed")
    expected_parent = root / "models" / str(model_id).replace("/", "--")
    if manifest_path.parent != expected_parent:
        errors.append("Octo manifest is not located in its model's pinned staging directory")
    files = raw.get("files")
    if not isinstance(files, list) or not files:
        errors.append("Octo manifest needs a non-empty files array")
        records: List[Dict[str, Any]] = []
    else:
        records = [_file_result(root, manifest_path.parent, item, path_key="relative_path") if isinstance(item, Mapping) else {"status": "fail", "errors": ["Octo files contains a non-object"]} for item in files]
    main_status = "pass" if not errors and all(item.get("status") == "pass" for item in records) else "fail"

    # Hugging Face cache snapshots commonly use symlinks to a content-addressed
    # blob store.  We deliberately do not follow them.  That support component
    # is blocked, while the separately regular Octo checkpoint remains auditable.
    t5_support: Dict[str, Any] = {"status": "blocked", "reason": "t5 support manifest is absent"}
    t5 = raw.get("t5")
    if isinstance(t5, Mapping):
        tokenizer_path = t5.get("tokenizer_path")
        try:
            tokenizer_root = _directory_under(root, Path(str(tokenizer_path)), label="Octo T5 tokenizer_path")
            t5_files = t5.get("files")
            if not isinstance(t5_files, list) or not t5_files:
                raise AuditError("Octo T5 support has no file records")
            support_records = [
                _file_result(root, tokenizer_root, item, path_key="relative_path")
                if isinstance(item, Mapping)
                else {"status": "fail", "errors": ["Octo T5 files contains a non-object"]}
                for item in t5_files
            ]
            support_has_symlink = any(
                "symlink" in error.lower()
                for item in support_records
                for error in item.get("errors", ())
                if isinstance(error, str)
            )
            t5_support = {
                "status": (
                    "pass"
                    if all(item.get("status") == "pass" for item in support_records)
                    else "blocked"
                    if support_has_symlink
                    else "fail"
                ),
                "model_id": t5.get("model_id"),
                "revision": t5.get("revision"),
                "files": support_records,
            }
        except AuditError as error:
            t5_support = {"status": "blocked", "reason": str(error)}
    staged_integrity = {
        "status": main_status,
        "definition": "regular Octo checkpoint/configuration files match the immutable Octo asset manifest locally",
        "files_checked": len(records),
        "errors": errors,
    }
    total_bytes = sum(item.get("bytes", 0) for item in records if isinstance(item.get("bytes"), int))
    return {
        "kind": "octo_asset_manifest",
        "source_manifest": {"path": str(manifest_path), "sha256": "sha256:" + manifest_digest},
        "repo_id": model_id,
        "repo_type": "model",
        "revision": checkpoint_revision,
        "code_revision": code_revision,
        "staged_integrity": staged_integrity,
        "files": records,
        "support_components": {"t5_tokenizer": t5_support},
        "fully_verified": False,
        "fully_verified_blockers": [
            "container digest is unknown",
            "runtime compatibility profile is unknown",
            "Octo T5 support is separately blocked when its cache uses symlinks",
            "asset lock license/access fields require separate review",
        ],
        "partial_asset_lock_proposal": _partial_lock_proposal(
            asset_id=KNOWN_LOCK_ASSET_IDS.get(str(model_id), "staged." + str(model_id or "octo").replace("/", "_")),
            repo_id=str(model_id or "unknown"),
            repo_type="model",
            revision=checkpoint_revision if isinstance(checkpoint_revision, str) else None,
            total_bytes=total_bytes,
            manifest_path=manifest_path,
            manifest_sha256=manifest_digest,
            staged_integrity=staged_integrity,
            source_url=("https://huggingface.co/%s/tree/%s" % (model_id, checkpoint_revision)) if isinstance(model_id, str) and isinstance(checkpoint_revision, str) else None,
        ),
    }


def audit_manifests(cluster_root: Path, manifest_paths: Iterable[Path]) -> Dict[str, Any]:
    """Read manifests and verify files without writing any asset or lock state."""

    if cluster_root.is_symlink():
        raise AuditError("cluster root may not be a symlink")
    root = cluster_root.resolve(strict=True)
    if not root.is_dir():
        raise AuditError("cluster root must be an existing non-symlink directory")
    assets: List[Dict[str, Any]] = []
    for requested in manifest_paths:
        manifest_path = _regular_file_under(root, requested, label="asset manifest")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise AuditError("asset manifest is not valid JSON: %s" % manifest_path) from error
        if not isinstance(raw, Mapping):
            raise AuditError("asset manifest must be a JSON object: %s" % manifest_path)
        digest = _sha256_file(manifest_path)
        if raw.get("schema") == OCTO_SCHEMA:
            assets.append(_audit_octo_manifest(root, manifest_path, raw, digest))
        elif "repo" in raw and "file_records" in raw:
            assets.append(_audit_download_manifest(root, manifest_path, raw, digest))
        else:
            assets.append(
                {
                    "kind": "unsupported_manifest",
                    "source_manifest": {"path": str(manifest_path), "sha256": "sha256:" + digest},
                    "staged_integrity": {"status": "fail", "errors": ["unsupported manifest schema"]},
                    "fully_verified": False,
                    "partial_asset_lock_proposal": None,
                }
            )
    passed = sum(1 for asset in assets if asset.get("staged_integrity", {}).get("status") == "pass")
    return {
        "schema": SCHEMA,
        "producer_sha256": _sha256_file(Path(__file__)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cluster_root": str(root),
        "assets": assets,
        "summary": {
            "assets": len(assets),
            "staged_integrity_pass": passed,
            "staged_integrity_not_pass": len(assets) - passed,
            "fully_verified": 0,
            "status": "partial_staged_evidence_only" if passed else "no_staged_integrity_passes",
        },
        "limitations": [
            "This report is an audit of supplied local manifests and regular staged files only.",
            "It does not edit assets.lock.json or protocol.json.",
            "staged_integrity does not imply a verified container, compatibility profile, license/access review, Gate A, or qualification.",
        ],
    }


def write_immutable_report(cluster_root: Path, output: Path, report: Mapping[str, Any]) -> Path:
    """Write exactly one no-clobber report under ``<root>/evidence``."""

    root = cluster_root.resolve(strict=True)
    evidence = root / "evidence"
    if evidence.exists():
        if evidence.is_symlink() or not evidence.is_dir():
            raise AuditError("audit report directory is not a regular evidence directory: %s" % evidence)
    else:
        evidence.mkdir(mode=0o700)
    destination = Path(os.path.abspath(str(output)))
    try:
        destination.relative_to(evidence)
    except ValueError as error:
        raise AuditError("audit report must be written beneath %s" % evidence) from error
    if destination.suffix != ".json":
        raise AuditError("audit report path must end in .json")
    # The parent must already be a regular non-symlink path under evidence;
    # only the final report is created by this tool.
    parent = destination.parent
    if parent != evidence:
        raise AuditError("audit report must be directly inside %s" % evidence)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite existing audit report: %s" % destination)
    encoded = json.dumps(dict(report), sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-root", type=Path, required=True, help="Existing cluster .../plumb root; never contacted remotely.")
    parser.add_argument("--manifest", type=Path, action="append", required=True, help="Existing download or Octo manifest under --cluster-root. Repeatable.")
    parser.add_argument("--output", type=Path, required=True, help="New report path directly under <cluster-root>/evidence.")
    args = parser.parse_args(argv)
    try:
        report = audit_manifests(args.cluster_root, args.manifest)
        output = write_immutable_report(args.cluster_root, args.output, report)
    except (AuditError, FileExistsError, OSError) as error:
        print("staged asset audit failed: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps({"report": str(output), "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
