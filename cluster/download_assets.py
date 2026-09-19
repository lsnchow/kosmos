"""Cluster-only pinned downloads with byte/hash evidence and explicit size caps.

Every asset in the build spec has a named entry in :mod:`cluster.asset_plan`, so
the normal invocation is one command per asset::

    python -m cluster.download_assets --from-plan octo-small --execute

The plan supplies the repository, type, revision, file patterns, size ceiling,
and license/access status.  This tool refuses to run when the plan says the
revision is unresolved, when access is gated and terms have not been recorded as
accepted, or when the entry is marked ``fetch=False``.  Ad-hoc invocations with
an explicit ``repo``/``--revision`` still work for one-off probes.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cluster.asset_plan import (
    ASSET_PLAN,
    Access,
    AssetPlanEntry,
    RepoType,
    asset_plan_entries,
    get_plan,
    plan_names,
    validate_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("repo", nargs="?", help="Repository id; omit when --from-plan is used")
    parser.add_argument("--from-plan", dest="from_plan", help="Named entry in cluster.asset_plan")
    parser.add_argument("--list-plans", action="store_true", help="List planned assets and exit")
    parser.add_argument(
        "--resolve-revision",
        action="store_true",
        help="Look up the plan repository's current head commit, print it, and exit without downloading",
    )
    parser.add_argument("--repo-type", choices=("model", "dataset"), default=None)
    parser.add_argument("--revision", help="Full resolved commit SHA (required without --from-plan)")
    parser.add_argument("--root", type=Path, default=Path("/scratch/lchow432/plumb"))
    parser.add_argument("--max-gb", type=float, default=None)
    parser.add_argument("--allow", action="append", default=[], help="Extra allow pattern (extends the plan)")
    parser.add_argument("--ignore", action="append", default=[], help="Extra ignore pattern (extends the plan)")
    parser.add_argument(
        "--accept-terms",
        action="store_true",
        help="Record that the operator has accepted this asset's gated access terms",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def _proven_mirror_for(entry: AssetPlanEntry) -> Optional[AssetPlanEntry]:
    """The plan entry proven byte-identical to ``entry``, if any.

    Only ``ByteIdentity.PROVEN`` counts, and only from a publicly reachable row:
    a mirror that is itself gated moves the blocker rather than removing it.
    This reports a delivery channel, never a license -- the caller still refuses
    without ``--accept-terms``.
    """
    for candidate in ASSET_PLAN.values():
        evidence = candidate.mirror_evidence
        if evidence is None or evidence.official_repo_id != entry.repo_id:
            continue
        if evidence.byte_identity.value != "proven":
            continue
        if candidate.access is not Access.PUBLIC:
            continue
        return candidate
    return None


def _resolve_plan(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Optional[AssetPlanEntry]:
    if not args.from_plan:
        if not args.repo:
            parser.error("Supply a repository id or --from-plan <name>")
        if not args.revision:
            parser.error("--revision is required for an ad-hoc download")
        return None
    try:
        entry = get_plan(args.from_plan)
    except KeyError as error:
        parser.error(str(error))
        raise AssertionError("unreachable")  # pragma: no cover - parser.error exits
    problems = entry.errors()
    if problems:
        parser.error("Asset plan %r is inconsistent: %s" % (entry.name, "; ".join(problems)))
    if args.repo and args.repo != entry.repo_id:
        parser.error("%r disagrees with plan %r repo_id %r" % (args.repo, entry.name, entry.repo_id))
    if entry.revision is not None and args.revision and args.revision != entry.revision:
        parser.error("--revision disagrees with the plan's pinned revision")
    if not entry.fetch:
        parser.error(
            "Asset plan %r is marked fetch=False: %s"
            % (entry.name, "; ".join(entry.notes) or "it is deliberately not downloaded")
        )
    if entry.repo_type in (RepoType.URL, RepoType.GIT):
        parser.error(
            "Asset plan %r is a %s artifact, not a Hub snapshot; use its dedicated fetch/clone step at revision %r"
            % (entry.name, entry.repo_type.value, entry.revision)
        )
    if entry.access is Access.GATED_ACCEPT_TERMS and not args.accept_terms:
        # Checked before the unresolved-revision branch below: --resolve-revision
        # calls repo_info() against the gated host and comes back 401, which hides
        # the real blocker behind a transport error.
        hint = ""
        mirror = _proven_mirror_for(entry)
        if mirror is not None:
            hint = (
                " A public mirror (%s) was proven byte-identical to this repository at official revision %s "
                "per %s, so accepting the terms is the only remaining step -- the bytes need not come from the "
                "gated host. Byte identity changes the delivery path, not the license."
                % (mirror.repo_id, mirror.mirror_evidence.official_revision, mirror.mirror_evidence.evidence_path)
            )
        parser.error(
            "Asset plan %r is gated (%s). Re-run with --accept-terms once the terms are genuinely accepted.%s"
            % (entry.name, entry.license_status.value, hint)
        )
    if entry.revision is None and not args.revision and not args.resolve_revision:
        parser.error(
            "Asset plan %r has no immutable revision yet. Run --resolve-revision to look one up, then pass it "
            "with --revision and update the plan entry; unresolved revisions never reach a verified lock status."
            % entry.name
        )
    return entry


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_plans:
        problems = validate_plan()
        print(
            json.dumps(
                {
                    "plans": list(plan_names()),
                    "plan_problems": list(problems),
                    "entries": list(asset_plan_entries()) if not problems else [],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0 if not problems else 2

    entry = _resolve_plan(parser, args)
    repo = entry.repo_id if entry is not None else args.repo
    repo_type = entry.repo_type.value if entry is not None else (args.repo_type or "model")
    revision = (entry.revision if entry is not None else None) or args.revision
    plan_revision_unresolved = entry is not None and entry.revision is None
    allow = list(entry.allow_patterns()) if entry is not None else []
    allow.extend(args.allow)
    ignore = list(entry.ignore_patterns) if entry is not None else []
    ignore.extend(args.ignore)
    max_gb = args.max_gb if args.max_gb is not None else (entry.max_gb if entry is not None else 40.0)

    root = args.root.resolve()
    if not str(root).startswith(("/scratch/", "/global/scratch/")) or root.name != "plumb":
        parser.error("Large downloads are permitted only in a cluster scratch/.../plumb directory")
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    if args.resolve_revision:
        head = api.repo_info(repo, repo_type=repo_type, files_metadata=False)
        print(
            json.dumps(
                {
                    "asset_plan_name": getattr(entry, "name", None),
                    "repo": repo,
                    "repo_type": repo_type,
                    "resolved_head_sha": head.sha,
                    "gated": getattr(head, "gated", None),
                    "action_required": (
                        "Review this commit, record it as the plan entry's revision, and remove the "
                        "unresolved_immutable_revision blocker. This lookup is not a review."
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    info = api.repo_info(repo, repo_type=repo_type, revision=revision, files_metadata=True)
    if info.sha != revision:
        parser.error("Revision must be the full resolved commit SHA")
    files = [sibling for sibling in info.siblings if _selected(sibling.rfilename, allow, ignore)]
    if allow and not files:
        parser.error("No repository file matched the planned patterns: %s" % ", ".join(allow))
    size = sum(sibling.size or 0 for sibling in files)
    plan: Dict[str, Any] = {
        "repo": repo,
        "repo_type": repo_type,
        "revision": info.sha,
        "gated": getattr(info, "gated", None),
        "host": socket.gethostname(),
        "planned_bytes": size,
        "destination": str(root / "models" / repo.replace("/", "--")),
        "allow_patterns": allow,
        "ignore_patterns": ignore,
        "max_gb": float(max_gb),
        "files": [sibling.rfilename for sibling in files],
    }
    if entry is not None:
        plan["asset_plan"] = entry.lock_entry()
        plan["asset_plan_name"] = entry.name
        plan["terms_accepted"] = bool(args.accept_terms)
        plan["plan_revision_unresolved"] = bool(plan_revision_unresolved)
        if plan_revision_unresolved:
            print(
                "NOTICE asset plan %r still records no immutable revision. This download used --revision %s; "
                "update cluster/asset_plan.py before the lock entry may leave 'planned'." % (entry.name, revision),
                flush=True,
            )
        for trap in entry.traps:
            print("TRAP " + trap, flush=True)
        if entry.redistribution.value == "prohibited_pending_resolution":
            print(
                "NOTICE %s may not be redistributed or mirrored: license_status=%s, license=%r. "
                "Local use only until the terms are resolved."
                % (entry.repo_id, entry.license_status.value, entry.license),
                flush=True,
            )
    print(json.dumps(plan, indent=2), flush=True)
    if size > max_gb * 1e9:
        parser.error("Download exceeds explicit size ceiling")
    if not args.execute:
        return 0

    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    started = time.time()
    destination = snapshot_download(
        repo_id=repo,
        repo_type=repo_type,
        revision=revision,
        local_dir=plan["destination"],
        allow_patterns=allow or None,
        ignore_patterns=ignore or None,
        max_workers=4,
    )
    records: List[Dict[str, Any]] = []
    for sibling in files:
        path = Path(destination) / sibling.rfilename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = path.stat().st_size
        if sibling.size is not None and actual != sibling.size:
            raise RuntimeError("Unexpected file size: " + sibling.rfilename)
        expected_bytes = _planned_expected_bytes(entry, sibling.rfilename)
        if expected_bytes is not None and actual != expected_bytes:
            raise RuntimeError(
                "Planned exact byte length mismatch for %s: expected %d, got %d"
                % (sibling.rfilename, expected_bytes, actual)
            )
        lfs = getattr(sibling, "lfs", None)
        expected = getattr(lfs, "sha256", None) if lfs else None
        if isinstance(lfs, dict):
            expected = lfs.get("sha256")
        if expected and digest.hexdigest() != expected:
            raise RuntimeError("SHA-256 mismatch: " + sibling.rfilename)
        records.append(
            {
                "path": sibling.rfilename,
                "bytes": actual,
                "planned_expected_bytes": expected_bytes,
                "sha256": digest.hexdigest(),
                "upstream_sha256": expected,
                "verified": bool(expected),
            }
        )
    plan.update(
        {
            "started_at": started,
            "completed_at": time.time(),
            "downloaded_bytes": sum(record["bytes"] for record in records),
            "file_records": records,
            "all_files_upstream_verified": all(record["verified"] for record in records) if records else False,
        }
    )
    name = (entry.name if entry is not None else repo.replace("/", "--")) + "-download.json"
    path = evidence / name
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print("MANIFEST " + str(path), flush=True)
    return 0


def _selected(name: str, allow: List[str], ignore: List[str]) -> bool:
    if allow and not any(fnmatch.fnmatch(name, pattern) for pattern in allow):
        return False
    return not any(fnmatch.fnmatch(name, pattern) for pattern in ignore)


def _planned_expected_bytes(entry: Optional[AssetPlanEntry], name: str) -> Optional[int]:
    """Only an exactly known byte length; a historical estimate is never enforced."""

    if entry is None:
        return None
    for planned in entry.files:
        if planned.path == name and not planned.is_directory:
            return planned.expected_bytes
    return None


if __name__ == "__main__":
    raise SystemExit(main())
