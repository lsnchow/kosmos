#!/usr/bin/env python3
"""Stage and verify immutable Octo v1 assets on a cluster login node.

The command is deliberately opt-in: without ``--execute`` it prints the exact
download plan and exits without a network request or filesystem mutation.  It
downloads only the released Octo snapshot and the small T5 *config/tokenizer*
files needed by Octo's source; it never downloads T5 model weights.

Do not run model loading or inference through this helper.  It is for a login
node with internet access.  A scheduler-allocated compute node must perform
the actual load/inference smoke after the isolated runtime is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


OCTO_CODE_REVISION = "37951e4e6d708fd76374f6e09e716763fe2673b1"
T5_MODEL_ID = "t5-base"
T5_REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"


@dataclass(frozen=True)
class FileSpec:
    relative_path: str
    bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class OctoAssetSpec:
    name: str
    model_id: str
    revision: str
    checkpoint_step: int
    checkpoint_bytes: int
    checkpoint_sha256: str
    license: str

    @property
    def octo_files(self) -> Tuple[FileSpec, ...]:
        step = str(self.checkpoint_step)
        return (
            FileSpec("config.json"),
            FileSpec("dataset_statistics.json"),
            FileSpec("example_batch.msgpack", 738368, "0ce74dd8e433ce4a8a1534c4ab9687d9fc3e444b3eece750047dcb22125d73ff"),
            FileSpec("%s/commit_success.txt" % step),
            FileSpec("%s/default/commit_success.txt" % step),
            FileSpec("%s/default/checkpoint" % step, self.checkpoint_bytes, self.checkpoint_sha256),
        )


ASSETS = {
    "small": OctoAssetSpec(
        name="small",
        model_id="rail-berkeley/octo-small",
        revision="03d88976c54a58e10480d2043a8c762b35bc2611",
        checkpoint_step=270000,
        checkpoint_bytes=546696551,
        checkpoint_sha256="590df097f8a37bbc1c3aac2a488c0fb08e72bae8abaedfb89f0685677d848962",
        license="MIT",
    ),
    "base": OctoAssetSpec(
        name="base",
        model_id="rail-berkeley/octo-base",
        revision="39d6c88fdbbcf6f841481a7d732f68c612d04609",
        checkpoint_step=300000,
        checkpoint_bytes=809848679,
        checkpoint_sha256="a16e66b6aac743afca4d9a830e32291362237212c50843d1ac0895270ea441fe",
        license="MIT",
    ),
}
T5_FILES = (
    FileSpec("config.json", 1208),
    FileSpec("spiece.model", 791656),
    FileSpec("tokenizer.json", 1389353),
)


class AssetError(RuntimeError):
    pass


def _require_cluster_root(root: Path) -> Path:
    resolved = root.resolve()
    if not str(resolved).startswith(("/scratch/", "/global/scratch/")) or resolved.name != "plumb":
        raise AssetError("Octo staging is permitted only in a cluster scratch/.../plumb directory")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, relative_path: str) -> Dict[str, Any]:
    if not path.is_file():
        raise AssetError("Expected staged file: %s" % path)
    return {"relative_path": relative_path, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _url(repo: str, revision: str, path: str) -> str:
    return "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, path)


def _validate_existing(path: Path, spec: FileSpec) -> None:
    if not path.is_file():
        raise AssetError("Expected file is not regular: %s" % path)
    if spec.bytes is not None and path.stat().st_size != spec.bytes:
        raise AssetError("Size mismatch for %s: expected %d, found %d" % (path, spec.bytes, path.stat().st_size))
    if spec.sha256 is not None and _sha256(path) != spec.sha256:
        raise AssetError("SHA-256 mismatch for %s" % path)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s-" % destination.name, suffix=".part", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream, urllib.request.urlopen(url) as response:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link is an atomic no-clobber publication on this shared
            # filesystem.  Two identical concurrent requests may race, but a
            # later writer never replaces the first staged immutable file.
            os.link(temporary_name, destination)
        except FileExistsError:
            _validate_existing(destination, FileSpec(destination.name))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    else:
        os.unlink(temporary_name)


def _stage_file(root: Path, repo: str, revision: str, spec: FileSpec, *, execute: bool) -> None:
    destination = root / spec.relative_path
    if destination.exists():
        _validate_existing(destination, spec)
        return
    if not execute:
        return
    _download(_url(repo, revision, spec.relative_path), destination)
    _validate_existing(destination, spec)


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if previous != encoded:
            raise AssetError("Refusing to overwrite a differing immutable asset manifest: %s" % path)
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise AssetError("Refusing to overwrite a concurrently created immutable asset manifest: %s" % path) from error
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    else:
        os.unlink(temporary_name)


def _write_new_text(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise AssetError("Refusing to overwrite immutable cache reference: %s" % path) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def paths(root: Path, spec: OctoAssetSpec) -> Tuple[Path, Path, Path]:
    model_root = root / "models" / ("rail-berkeley--octo-" + spec.name)
    hf_home = root / "models" / ".hf-octo"
    # Octo source asks for the historical identifier ``t5-base``.  The normal
    # Hugging Face cache lookup expects this snapshot/refs layout while offline.
    t5_snapshot = hf_home / "hub" / "models--t5-base" / "snapshots" / T5_REVISION
    return model_root, hf_home, t5_snapshot


def plan(root: Path, spec: OctoAssetSpec) -> Dict[str, Any]:
    model_root, hf_home, t5_snapshot = paths(root, spec)
    return {
        "model": {
            "id": spec.model_id,
            "revision": spec.revision,
            "license": spec.license,
            "checkpoint_step": spec.checkpoint_step,
            "checkpoint_bytes": spec.checkpoint_bytes,
            "checkpoint_sha256": spec.checkpoint_sha256,
            "target": str(model_root),
        },
        "octo_code": {"revision": OCTO_CODE_REVISION, "source": "https://github.com/octo-models/octo"},
        "t5_tokenizer_only": {
            "id": T5_MODEL_ID,
            "revision": T5_REVISION,
            "license": "Apache-2.0",
            "files": [{"path": item.relative_path, "bytes": item.bytes} for item in T5_FILES],
            "target": str(t5_snapshot),
            "full_t5_model_weights_downloaded": False,
        },
        "hf_home": str(hf_home),
        "total_known_bytes": spec.checkpoint_bytes + 738368 + sum(item.bytes or 0 for item in T5_FILES),
        "manifest": str(model_root / "PLUMB-ASSET-MANIFEST.json"),
    }


def execute(root: Path, spec: OctoAssetSpec) -> Path:
    root = _require_cluster_root(root)
    model_root, hf_home, t5_snapshot = paths(root, spec)
    for file_spec in spec.octo_files:
        _stage_file(model_root, spec.model_id, spec.revision, file_spec, execute=True)
    for file_spec in T5_FILES:
        _stage_file(t5_snapshot, T5_MODEL_ID, T5_REVISION, file_spec, execute=True)
    ref_path = hf_home / "hub" / "models--t5-base" / "refs" / "main"
    if ref_path.exists() and ref_path.read_text(encoding="utf-8") != T5_REVISION:
        raise AssetError("Refusing to replace a t5-base cache ref with another revision: %s" % ref_path)
    if not ref_path.exists():
        # huggingface_hub reads this value verbatim when resolving ``main``;
        # a trailing newline makes it look for a non-existent snapshot.
        _write_new_text(ref_path, T5_REVISION)
    manifest = {
        "schema": "plumb-octo-assets-v1",
        "model_id": spec.model_id,
        "checkpoint_revision": spec.revision,
        "code_revision": OCTO_CODE_REVISION,
        "files": [_record(model_root / item.relative_path, item.relative_path) for item in spec.octo_files],
        "t5": {
            "model_id": T5_MODEL_ID,
            "revision": T5_REVISION,
            "tokenizer_path": str(t5_snapshot),
            "files": [_record(t5_snapshot / item.relative_path, item.relative_path) for item in T5_FILES],
            "full_t5_model_weights_downloaded": False,
        },
    }
    manifest_path = model_root / "PLUMB-ASSET-MANIFEST.json"
    _write_manifest(manifest_path, manifest)
    return manifest_path


def verify(root: Path, spec: OctoAssetSpec) -> Path:
    model_root, hf_home, t5_snapshot = paths(root, spec)
    manifest_path = model_root / "PLUMB-ASSET-MANIFEST.json"
    if not manifest_path.is_file():
        raise AssetError("No immutable Octo asset manifest exists: %s" % manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema") != "plumb-octo-assets-v1" or raw.get("model_id") != spec.model_id or raw.get("checkpoint_revision") != spec.revision:
        raise AssetError("Asset manifest does not describe the requested immutable Octo model.")
    records = {item.get("relative_path"): item for item in raw.get("files", []) if isinstance(item, dict)}
    for file_spec in spec.octo_files:
        actual = _record(model_root / file_spec.relative_path, file_spec.relative_path)
        if records.get(file_spec.relative_path) != actual:
            raise AssetError("Asset manifest mismatch for %s" % file_spec.relative_path)
        _validate_existing(model_root / file_spec.relative_path, file_spec)
    t5 = raw.get("t5")
    if not isinstance(t5, dict) or t5.get("model_id") != T5_MODEL_ID or t5.get("revision") != T5_REVISION:
        raise AssetError("Asset manifest does not bind the required t5-base tokenizer snapshot.")
    t5_records = {item.get("relative_path"): item for item in t5.get("files", []) if isinstance(item, dict)}
    for file_spec in T5_FILES:
        actual = _record(t5_snapshot / file_spec.relative_path, file_spec.relative_path)
        if t5_records.get(file_spec.relative_path) != actual:
            raise AssetError("Asset manifest mismatch for t5-base %s" % file_spec.relative_path)
        _validate_existing(t5_snapshot / file_spec.relative_path, file_spec)
    ref = hf_home / "hub" / "models--t5-base" / "refs" / "main"
    if not ref.is_file() or ref.read_text(encoding="utf-8") != T5_REVISION:
        raise AssetError("Offline t5-base cache reference must contain exactly the pinned 40-character revision.")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/scratch/lchow432/plumb"), help="Existing cluster PLUMB root.")
    parser.add_argument("--model", choices=tuple(ASSETS), default="small")
    parser.add_argument("--execute", action="store_true", help="Download the planned immutable files. Omit for a no-write plan.")
    parser.add_argument("--verify", action="store_true", help="Verify an existing manifest/files only; makes no network request.")
    args = parser.parse_args(argv)
    if args.execute and args.verify:
        parser.error("--execute and --verify are mutually exclusive")
    spec = ASSETS[args.model]
    try:
        if args.execute or args.verify:
            args.root = _require_cluster_root(args.root)
        if args.verify:
            print(json.dumps({"verified_manifest": str(verify(args.root, spec))}, sort_keys=True, indent=2))
        elif args.execute:
            print(json.dumps({"staged_manifest": str(execute(args.root, spec)), "plan": plan(args.root, spec)}, sort_keys=True, indent=2))
        else:
            print(json.dumps(plan(args.root, spec), sort_keys=True, indent=2))
    except (AssetError, OSError, urllib.error.URLError, ValueError) as error:
        print("Octo asset preparation failed: %s" % error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
