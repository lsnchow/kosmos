"""Cluster-only pinned downloads with byte/hash evidence and explicit size caps."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--root", type=Path, default=Path("/scratch/lchow432/plumb"))
    parser.add_argument("--max-gb", type=float, default=40)
    parser.add_argument("--allow", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not str(root).startswith(("/scratch/", "/global/scratch/")) or root.name != "plumb":
        parser.error("Large downloads are permitted only in a cluster scratch/.../plumb directory")
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    from huggingface_hub import HfApi, snapshot_download
    import fnmatch
    api = HfApi()
    info = api.repo_info(args.repo, repo_type=args.repo_type, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        parser.error("Revision must be the full resolved commit SHA")
    files = [f for f in info.siblings if not args.allow or any(fnmatch.fnmatch(f.rfilename, p) for p in args.allow)]
    size = sum(f.size or 0 for f in files)
    plan = {"repo": args.repo, "repo_type": args.repo_type, "revision": info.sha, "gated": getattr(info, "gated", None),
            "host": socket.gethostname(), "planned_bytes": size,
            "destination": str(root / "models" / args.repo.replace("/", "--")),
            "files": [f.rfilename for f in files]}
    print(json.dumps(plan, indent=2), flush=True)
    if size > args.max_gb * 1e9:
        parser.error("Download exceeds explicit size ceiling")
    if not args.execute:
        return
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    started = time.time()
    destination = snapshot_download(repo_id=args.repo, repo_type=args.repo_type, revision=args.revision,
                                    local_dir=plan["destination"],
                                    allow_patterns=args.allow or None, max_workers=4)
    records = []
    for f in files:
        path = Path(destination) / f.rfilename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = path.stat().st_size
        if f.size is not None and actual != f.size:
            raise RuntimeError("Unexpected file size: " + f.rfilename)
        lfs = getattr(f, "lfs", None)
        expected = getattr(lfs, "sha256", None) if lfs else None
        if isinstance(lfs, dict):
            expected = lfs.get("sha256")
        if expected and digest.hexdigest() != expected:
            raise RuntimeError("SHA-256 mismatch: " + f.rfilename)
        records.append({"path": f.rfilename, "bytes": actual, "sha256": digest.hexdigest(),
                        "upstream_sha256": expected, "verified": bool(expected)})
    plan.update({"started_at": started, "completed_at": time.time(), "downloaded_bytes": sum(r["bytes"] for r in records), "file_records": records})
    path = evidence / (args.repo.replace("/", "--") + "-download.json")
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print("MANIFEST " + str(path), flush=True)


if __name__ == "__main__":
    main()
