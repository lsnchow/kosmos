"""Stage the exact private semantic epoch-02 adapter for the one-GPU demo worker.

This script is intentionally the only route by which the adapter enters the
Truss source directory. It rejects the formatting adapter, follows no symlinks,
checks the historical tree hash, and never copies weights into the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = ROOT / "deploy" / "baseten" / "judge_demo" / "model"
EXPECTED_TREE_SHA256 = "sha256:7a0b7796a0ac8b78fbb8ba8f4c4db01fe8d35c2edc3cb7a0fbeec9f4fb20523e"
EXPECTED_BASE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def tree_sha256(directory: Path) -> str:
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("adapter contains a symlink: %s" % path)
        if path.is_file():
            records.append({"path": path.relative_to(directory).as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    if not records:
        raise ValueError("adapter contains no regular files")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def copy_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source must be a regular directory")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("source contains a symlink: %s" % path)
    shutil.copytree(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True, help="Private semantic epoch-02 adapter directory.")
    parser.add_argument("--check", action="store_true", help="Only verify the already staged adapter/runtime.")
    args = parser.parse_args()
    adapter = args.adapter_dir.resolve(strict=True)
    actual = tree_sha256(adapter)
    if actual != EXPECTED_TREE_SHA256:
        raise SystemExit("refusing adapter tree %s; expected semantic epoch-02 %s" % (actual, EXPECTED_TREE_SHA256))
    staged_adapter = DEPLOYMENT / "adapter"
    staged_runtime = DEPLOYMENT / "plumb"
    if args.check:
        if not staged_adapter.is_dir() or tree_sha256(staged_adapter) != EXPECTED_TREE_SHA256:
            raise SystemExit("staged semantic adapter is missing or mismatched")
        for relative in ("__init__.py", "adapters/contracts.py", "policies/judge.py", "policies/tasks.py", "policies/provenance.py"):
            if not (staged_runtime / relative).is_file():
                raise SystemExit("staged judge runtime is missing %s" % relative)
        print("semantic judge staging is current")
        return 0
    for destination in (staged_adapter, staged_runtime):
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise SystemExit("refusing to replace non-directory staging path %s" % destination)
            shutil.rmtree(destination)
    copy_tree(adapter, staged_adapter)
    files = (
        "plumb/__init__.py",
        "plumb/adapters/contracts.py",
        "plumb/policies/judge.py",
        "plumb/policies/tasks.py",
        "plumb/policies/provenance.py",
    )
    for relative in files:
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise SystemExit("runtime source is unavailable: %s" % source)
        target = DEPLOYMENT / "plumb" / Path(relative).relative_to("plumb")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # The production package's namespace initializers expose every policy
    # adapter. This dedicated image intentionally contains only the judge's
    # dependency-light modules, so its namespace packages stay empty.
    (staged_runtime / "adapters" / "__init__.py").write_text("\"\"\"Judge-only adapter namespace.\"\"\"\n", encoding="utf-8")
    (staged_runtime / "policies" / "__init__.py").write_text("\"\"\"Judge-only policy namespace.\"\"\"\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "adapter_id": "semantic_pilot_epoch_02",
        "adapter_tree_sha256": tree_sha256(staged_adapter),
        "base_model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "base_model_revision": EXPECTED_BASE_REVISION,
        "source": "private staging only; not a public model source",
    }
    (DEPLOYMENT / "adapter-manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print("staged semantic epoch-02 adapter %s" % manifest["adapter_tree_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
