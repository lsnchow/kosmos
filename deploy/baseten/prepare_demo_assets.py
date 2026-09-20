"""Prepare and verify the immutable Cosmos asset lock for the demo Chain.

This helper is deliberately offline.  It consumes the already-recorded
cluster download manifest, produces a small JSON lock for the files the
Diffusers Cosmos runtime actually needs, and verifies those exact bytes at the
Baseten mount before a ``kosmos-demo-assets-v1`` manifest can call the asset
verified.  It never invokes Hugging Face, Baseten, SSH, or a downloader.

The source download manifest's ``verified`` flag means a publisher digest was
available/compared for a particular source file.  It is provenance, not a
runtime trust decision: deployment verification hashes every selected mounted
file against the cluster-computed SHA-256 in this lock even when no publisher
LFS digest was supplied.

Typical operator sequence (the paths are illustrative and no command is run
by this module automatically)::

    python deploy/baseten/prepare_demo_assets.py lock \
      --source data/baseten-comparison-evidence/cosmos-download.json \
      --output /secure-build/cosmos3-nano-runtime-lock.json
    python deploy/baseten/prepare_demo_assets.py manifest \
      --lock /secure-build/cosmos3-nano-runtime-lock.json \
      --model-root /app/model_cache/cosmos3-nano \
      --content-lock-path /app/cosmos3-nano-runtime-lock.json \
      --output /app/kosmos-demo-assets.json

Both write operations are no-clobber.  The second command verifies all locked
runtime files before it emits the manifest consumed by ``demo_chain.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


COSMOS_ASSET_ID = "cosmos3-nano"
COSMOS_REPOSITORY = "nvidia/Cosmos3-Nano"
COSMOS_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
CONTENT_LOCK_SCHEMA = "kosmos-cosmos-runtime-lock-v1"
DEMO_ASSET_MANIFEST_SCHEMA = "kosmos-demo-assets-v1"

# These are documentation/example materials, not input to
# Cosmos3OmniPipeline.from_pretrained.  Everything else in the recorded
# snapshot is retained; that includes every tokenizer/config/model component,
# avoiding a fragile hand-curated list of shard names.
_EXCLUDED_EXACT = frozenset((".gitattributes", "BIAS.md", "EXPLAINABILITY.md", "PRIVACY.md", "README.md", "SAFETY.md"))
_EXCLUDED_PREFIXES = ("assets/", "images/")


class DemoAssetPreparationError(RuntimeError):
    """A lock or mounted model bytes are unsuitable for the demo worker."""


PathLike = Union[str, Path]


def _sha256_bytes(value: bytes, *, prefix: bool = False) -> str:
    digest = hashlib.sha256(value).hexdigest()
    return "sha256:" + digest if prefix else digest


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _relative_runtime_path(value: object) -> str:
    """Accept one safe POSIX-relative file path, never a directory/escape."""

    if not isinstance(value, str) or not value:
        raise DemoAssetPreparationError("content-lock file path must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/") or "\\" in value:
        raise DemoAssetPreparationError("content-lock path %r must be a safe POSIX-relative path" % value)
    if any(part in ("", ".", "..") for part in path.parts):
        raise DemoAssetPreparationError("content-lock path %r contains an unsafe path component" % value)
    if len(path.parts) == 0:
        raise DemoAssetPreparationError("content-lock path cannot name the model root")
    return path.as_posix()


def _runtime_file(path: str) -> bool:
    return path not in _EXCLUDED_EXACT and not path.startswith(_EXCLUDED_PREFIXES)


def _load_json_file(path_value: PathLike, *, label: str) -> Tuple[Mapping[str, Any], bytes]:
    path = Path(path_value)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise DemoAssetPreparationError("%s cannot be read as a JSON object: %s" % (label, path)) from error
    if not isinstance(value, Mapping):
        raise DemoAssetPreparationError("%s must be a JSON object" % label)
    return value, raw


def _source_records(source: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Tuple[str, ...]]:
    if source.get("repo") != COSMOS_REPOSITORY:
        raise DemoAssetPreparationError("source manifest repo must be %r" % COSMOS_REPOSITORY)
    if source.get("revision") != COSMOS_REVISION:
        raise DemoAssetPreparationError("source manifest revision must be the pinned Cosmos revision %s" % COSMOS_REVISION)
    declared = source.get("files")
    raw_records = source.get("file_records")
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        raise DemoAssetPreparationError("source manifest files must be a list")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise DemoAssetPreparationError("source manifest file_records must be a list")
    declared_paths = tuple(_relative_runtime_path(value) for value in declared)
    if len(set(declared_paths)) != len(declared_paths):
        raise DemoAssetPreparationError("source manifest files contains duplicate paths")

    records: Dict[str, Dict[str, Any]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise DemoAssetPreparationError("source file_records contains a non-object entry")
        path = _relative_runtime_path(raw.get("path"))
        if path in records:
            raise DemoAssetPreparationError("source file_records contains duplicate path %s" % path)
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise DemoAssetPreparationError("source file %s has invalid byte count" % path)
        if not _is_sha256(digest):
            raise DemoAssetPreparationError("source file %s has invalid cluster-computed sha256" % path)
        publisher = raw.get("upstream_sha256")
        if publisher is not None and not _is_sha256(publisher):
            raise DemoAssetPreparationError("source file %s has invalid publisher sha256" % path)
        verified = raw.get("verified")
        if not isinstance(verified, bool):
            raise DemoAssetPreparationError("source file %s must explicitly record publisher-digest verified true/false" % path)
        records[path] = {
            "path": path,
            "bytes": size,
            "sha256": digest,
            "publisher_sha256": publisher,
            # Preserve the source evidence. It is never accepted as a proxy
            # for hashing the later Baseten mount.
            "publisher_digest_verified_by_source_manifest": verified,
            "publisher_digest_matches_cluster_hash": bool(publisher is not None and publisher == digest),
        }
    if set(records) != set(declared_paths):
        missing = sorted(set(declared_paths) - set(records))
        extra = sorted(set(records) - set(declared_paths))
        raise DemoAssetPreparationError(
            "source manifest files/file_records disagree (missing=%s extra=%s)" % (missing, extra)
        )
    return records, declared_paths


def build_cosmos_content_lock(source_manifest: Union[PathLike, Mapping[str, Any]]) -> Dict[str, Any]:
    """Make a deterministic lock from recorded cluster hashes, without files/network.

    The resulting ``file_records`` intentionally use source manifest SHA-256
    values regardless of whether the source had a publisher LFS digest for a
    given file.  Those are the actual hashes that must later match the mounted
    Baseten bytes.
    """

    if isinstance(source_manifest, Mapping):
        source = source_manifest
        source_digest = _sha256_bytes(_canonical_json(source), prefix=True)
        source_label = "mapping:canonical-json"
    else:
        source, source_raw = _load_json_file(source_manifest, label="source download manifest")
        source_digest = _sha256_bytes(source_raw, prefix=True)
        source_label = str(Path(source_manifest))
    records, declared_paths = _source_records(source)
    selected = tuple(records[path] for path in declared_paths if _runtime_file(path))
    if not selected:
        raise DemoAssetPreparationError("source manifest selected no Cosmos runtime files")
    file_records = [dict(record) for record in selected]
    file_records_payload = [
        {"path": record["path"], "bytes": record["bytes"], "sha256": record["sha256"]} for record in file_records
    ]
    return {
        "schema_version": CONTENT_LOCK_SCHEMA,
        "asset_id": COSMOS_ASSET_ID,
        "repo": COSMOS_REPOSITORY,
        "revision": COSMOS_REVISION,
        "source_download_manifest": source_label,
        "source_download_manifest_sha256": source_digest,
        "selection": {
            "included": "all recorded source files except documentation, assets/, and images/",
            "excluded_exact": sorted(_EXCLUDED_EXACT),
            "excluded_prefixes": list(_EXCLUDED_PREFIXES),
        },
        "verification_basis": "cluster_computed_sha256_per_file",
        "publisher_digest_distinction": (
            "publisher_digest_verified_by_source_manifest is retained as source provenance only; "
            "Baseten verification rehashes every selected mounted file."
        ),
        "file_records": file_records,
        "file_records_sha256": _sha256_bytes(_canonical_json(file_records_payload), prefix=True),
        "runtime_file_count": len(file_records),
        "runtime_bytes": sum(int(record["bytes"]) for record in file_records),
        "publisher_digest_verified_file_count": sum(
            1 for record in file_records if record["publisher_digest_verified_by_source_manifest"] is True
        ),
        "publisher_digest_unverified_file_count": sum(
            1 for record in file_records if record["publisher_digest_verified_by_source_manifest"] is False
        ),
    }


def validate_cosmos_content_lock(lock: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a loaded lock and return its normalised core identity.

    A true ``verified`` field in any source/lock is intentionally ignored here.
    Verification success is established only by :func:`verify_cosmos_runtime_files`.
    """

    if not isinstance(lock, Mapping) or lock.get("schema_version") != CONTENT_LOCK_SCHEMA:
        raise DemoAssetPreparationError("content lock is not a %s record" % CONTENT_LOCK_SCHEMA)
    if lock.get("asset_id") != COSMOS_ASSET_ID or lock.get("repo") != COSMOS_REPOSITORY:
        raise DemoAssetPreparationError("content lock does not bind the Cosmos3-Nano asset identity")
    if lock.get("revision") != COSMOS_REVISION:
        raise DemoAssetPreparationError("content lock does not bind the pinned Cosmos revision")
    records = lock.get("file_records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise DemoAssetPreparationError("content lock must include nonempty file_records")
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise DemoAssetPreparationError("content lock contains a non-object file record")
        path = _relative_runtime_path(raw.get("path"))
        if path in seen:
            raise DemoAssetPreparationError("content lock contains duplicate path %s" % path)
        seen.add(path)
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not _is_sha256(digest):
            raise DemoAssetPreparationError("content lock record %s has invalid bytes/sha256" % path)
        normalized.append({"path": path, "bytes": size, "sha256": digest})
    if lock.get("runtime_file_count") != len(normalized):
        raise DemoAssetPreparationError("content lock runtime_file_count does not match file_records")
    if lock.get("runtime_bytes") != sum(item["bytes"] for item in normalized):
        raise DemoAssetPreparationError("content lock runtime_bytes does not match file_records")
    expected_digest = _sha256_bytes(_canonical_json(normalized), prefix=True)
    if lock.get("file_records_sha256") != expected_digest:
        raise DemoAssetPreparationError("content lock file_records_sha256 does not match its per-file records")
    return {
        "asset_id": COSMOS_ASSET_ID,
        "repo": COSMOS_REPOSITORY,
        "revision": COSMOS_REVISION,
        "file_records": normalized,
        "file_records_sha256": expected_digest,
        "runtime_file_count": len(normalized),
        "runtime_bytes": sum(item["bytes"] for item in normalized),
    }


def load_cosmos_content_lock(path_value: PathLike) -> Tuple[Dict[str, Any], str]:
    """Read a lock and return it with the raw-file SHA used by the worker."""

    lock, raw = _load_json_file(path_value, label="Cosmos content lock")
    normalised = validate_cosmos_content_lock(lock)
    # Return the original lock too, so publisher digest provenance remains
    # available to an artifact manifest, with validation already complete.
    result = dict(lock)
    result.update(normalised)
    return result, _sha256_bytes(raw, prefix=True)


def _mounted_file(root: Path, relative_path: str) -> Path:
    try:
        root_real = root.resolve(strict=True)
    except OSError as error:
        raise DemoAssetPreparationError("model root is not a readable mounted directory: %s" % root) from error
    if root.is_symlink() or not root_real.is_dir():
        raise DemoAssetPreparationError("model root must be a real directory, not a symlink: %s" % root)
    # ``relative_path`` was already parsed as strictly lexical POSIX segments,
    # so joining it cannot escape ``root``.  Do not call resolve() on the leaf:
    # that would follow a malicious symlink before the verifier gets to reject
    # it.  Existing intermediate symlinks are rejected for the same reason.
    candidate = root
    parts = PurePosixPath(relative_path).parts
    for index, component in enumerate(parts):
        candidate = candidate / component
        if index == len(parts) - 1:
            continue
        try:
            info = candidate.lstat()
        except OSError:
            # The final lstat in the verifier will report the first missing
            # locked file with its expected relative name.
            continue
        if stat.S_ISLNK(info.st_mode):
            raise DemoAssetPreparationError("locked runtime path contains a symlink: %s" % relative_path)
        if not stat.S_ISDIR(info.st_mode):
            raise DemoAssetPreparationError("locked runtime path has a non-directory parent: %s" % relative_path)
    return candidate


def verify_cosmos_runtime_files(lock: Mapping[str, Any], model_root: PathLike) -> Dict[str, Any]:
    """Hash every locked actual file under a mounted Cosmos model root.

    The returned report is only a report; callers must use
    :func:`create_verified_demo_asset_manifest` to mint a manifest record.
    Missing, symlinked, changed, or escaping paths raise instead of producing
    a partial ``verified`` result.
    """

    normalised = validate_cosmos_content_lock(lock)
    root = Path(model_root)
    checked: List[Dict[str, Any]] = []
    for record in normalised["file_records"]:
        relative_path = record["path"]
        candidate = _mounted_file(root, relative_path)
        try:
            info = candidate.lstat()
        except OSError as error:
            raise DemoAssetPreparationError("locked runtime file is missing: %s" % relative_path) from error
        if stat.S_ISLNK(info.st_mode):
            raise DemoAssetPreparationError("locked runtime file must not be a symlink: %s" % relative_path)
        if not stat.S_ISREG(info.st_mode):
            raise DemoAssetPreparationError("locked runtime path must be a regular file: %s" % relative_path)
        if info.st_size != record["bytes"]:
            raise DemoAssetPreparationError(
                "locked runtime file byte count mismatch for %s: expected %d got %d"
                % (relative_path, record["bytes"], info.st_size)
            )
        actual = _sha256_file(candidate)
        if actual != record["sha256"]:
            raise DemoAssetPreparationError("locked runtime file sha256 mismatch for %s" % relative_path)
        checked.append({"path": relative_path, "bytes": info.st_size, "sha256": actual})
    if len(checked) != normalised["runtime_file_count"]:
        raise DemoAssetPreparationError("runtime verification did not hash every locked file")
    return {
        "status": "verified",
        "asset_id": normalised["asset_id"],
        "repo": normalised["repo"],
        "revision": normalised["revision"],
        "model_root": str(root),
        "checked_all_locked_files": True,
        "runtime_file_count": len(checked),
        "runtime_bytes": sum(item["bytes"] for item in checked),
        "file_records_sha256": normalised["file_records_sha256"],
    }


def create_verified_demo_asset_manifest(
    lock: Mapping[str, Any],
    *,
    model_root: PathLike,
    content_lock_path: PathLike,
    content_lock_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify mounted files, then make the Cosmos record consumed by the worker.

    ``content_lock_path`` is the path that will be mounted in the worker.  The
    caller may supply a raw-file SHA when the content lock is staged elsewhere;
    otherwise a local readable lock path is hashed.  Either way it is carried
    into the record so the worker can rehash its mounted lock before trusting
    the per-file list.
    """

    normalised = validate_cosmos_content_lock(lock)
    verification = verify_cosmos_runtime_files(lock, model_root)
    if verification["status"] != "verified":  # Defensive: verifier normally raises on failure.
        raise DemoAssetPreparationError("Cosmos runtime verification did not complete")
    if content_lock_sha256 is None:
        candidate = Path(content_lock_path)
        try:
            content_lock_sha256 = _sha256_bytes(candidate.read_bytes(), prefix=True)
        except OSError as error:
            raise DemoAssetPreparationError("content lock cannot be hashed at %s" % candidate) from error
    if not isinstance(content_lock_sha256, str) or not content_lock_sha256.startswith("sha256:") or not _is_sha256(content_lock_sha256[7:]):
        raise DemoAssetPreparationError("content_lock_sha256 must be a sha256:<64-hex> digest")
    asset = {
        "asset_id": COSMOS_ASSET_ID,
        "path": str(Path(model_root)),
        "verified": True,
        # This names the deterministic file-record set, not an invented
        # aggregate model byte stream.
        "sha256": normalised["file_records_sha256"],
        "sha256_kind": "canonical_runtime_file_records_v1",
        "content_lock_path": str(content_lock_path),
        "content_lock_sha256": content_lock_sha256,
        "repo": COSMOS_REPOSITORY,
        "revision": COSMOS_REVISION,
        "runtime_file_count": normalised["runtime_file_count"],
        "runtime_bytes": normalised["runtime_bytes"],
        "verification": verification,
    }
    return {"schema_version": DEMO_ASSET_MANIFEST_SCHEMA, "assets": {COSMOS_ASSET_ID: asset}}


def write_json_no_clobber(path_value: PathLike, value: Mapping[str, Any]) -> Path:
    """Atomically create a JSON artifact without replacing prior evidence."""

    path = Path(path_value)
    parent = path.parent
    if not parent.is_dir():
        raise DemoAssetPreparationError("output parent directory does not exist: %s" % parent)
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except FileExistsError:
        raise FileExistsError("refusing to overwrite existing immutable asset artifact: %s" % path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


def _command_lock(arguments: argparse.Namespace) -> int:
    lock = build_cosmos_content_lock(arguments.source)
    output = write_json_no_clobber(arguments.output, lock)
    print(json.dumps({"status": "created", "content_lock": str(output), "file_records_sha256": lock["file_records_sha256"]}, sort_keys=True))
    return 0


def _command_verify(arguments: argparse.Namespace) -> int:
    lock, _ = load_cosmos_content_lock(arguments.lock)
    report = verify_cosmos_runtime_files(lock, arguments.model_root)
    print(json.dumps(report, sort_keys=True))
    return 0


def _command_manifest(arguments: argparse.Namespace) -> int:
    lock, raw_lock_sha = load_cosmos_content_lock(arguments.lock)
    manifest = create_verified_demo_asset_manifest(
        lock,
        model_root=arguments.model_root,
        content_lock_path=arguments.content_lock_path,
        content_lock_sha256=raw_lock_sha,
    )
    output = write_json_no_clobber(arguments.output, manifest)
    asset = manifest["assets"][COSMOS_ASSET_ID]
    print(json.dumps({"status": "created", "manifest": str(output), "asset_id": COSMOS_ASSET_ID, "sha256": asset["sha256"]}, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock_parser = subparsers.add_parser("lock", help="create a compact Cosmos runtime content lock from cluster evidence")
    lock_parser.add_argument("--source", required=True, type=Path, help="existing cosmos-download.json; no network access is attempted")
    lock_parser.add_argument("--output", required=True, type=Path, help="new no-clobber content lock path")
    lock_parser.set_defaults(handler=_command_lock)
    verify_parser = subparsers.add_parser("verify", help="hash all locked mounted Cosmos runtime files")
    verify_parser.add_argument("--lock", required=True, type=Path, help="content lock created by the lock command")
    verify_parser.add_argument("--model-root", required=True, type=Path, help="mounted Cosmos model root, e.g. /app/model_cache/cosmos3-nano")
    verify_parser.set_defaults(handler=_command_verify)
    manifest_parser = subparsers.add_parser("manifest", help="verify mounted files and create a verified demo asset manifest")
    manifest_parser.add_argument("--lock", required=True, type=Path, help="content lock created by the lock command")
    manifest_parser.add_argument("--model-root", required=True, type=Path, help="mounted Cosmos model root")
    manifest_parser.add_argument("--content-lock-path", required=True, type=Path, help="path where the same content lock will be mounted in the worker")
    manifest_parser.add_argument("--output", required=True, type=Path, help="new no-clobber kosmos-demo-assets-v1 path")
    manifest_parser.set_defaults(handler=_command_manifest)
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (DemoAssetPreparationError, FileExistsError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
