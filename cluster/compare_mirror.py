"""Prove or disprove that an ungated mirror carries an official asset's bytes.

``google/paligemma-3b-pt-224`` is gated, and OpenPiZero needs seven support files
from it.  Spec section 1 allows ``leo009/paligemma-3b-pt-224`` as a candidate
mirror "only after provenance, hashes, and applicable terms are resolved" and
forbids claiming byte identity "without a saved comparison".  This tool produces
that saved comparison::

    python -m cluster.compare_mirror --asset paligemma \\
        --official-revision <40-hex> --mirror-revision <40-hex>            # dry run
    python -m cluster.compare_mirror --asset paligemma ... --execute       # fetch, hash, write

It fetches each file **from the mirror**, hashes the bytes itself, and compares
them against whatever the official repository exposes without accepting terms.
The Hub's metadata endpoint publishes a file size, a git blob SHA-1 for regular
files, and an LFS SHA-256 for large ones; for a gated repository it may publish
none of that.

``byte_identity`` is computed, never supplied, and reaches ``proven`` only when
**every** compared file matched a hash actually retrieved from the official
repository.  A size match is recorded and explicitly does not count: two
different files of equal length are trivial to produce.  When the official hashes
cannot be obtained the verdict is ``unproven`` and the report says exactly why.

Proven identity removes the gated *download*.  It does not touch the licence:
the Gemma Terms of Use govern these weights whichever host supplied the bytes,
and accepting them is a human act that this tool does not perform and does not
record as done.

Network access happens only under ``--execute``.  The fetcher is injected, so the
tests exercise every verdict without a socket.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from cluster.asset_plan import (
    UNPROVEN_BYTE_IDENTITY_BLOCKER,
    AssetPlanEntry,
    ByteIdentity,
    GEMMA_LICENSE_NOTE,
    LicenseStatus,
    LocalUse,
    MirrorEvidence,
    RepoType,
    get_plan,
    mirror_evidence_path,
)
from cluster.resolve_licenses import (
    FetchResponse,
    Fetcher,
    urlopen_fetcher,
)


SCHEMA_VERSION = 1
REPORT_KIND = "plumb_mirror_byte_comparison"

HUGGINGFACE_HOST = "https://huggingface.co"

#: Spec section 1's required PaliGemma support files, in the order it names them.
PALIGEMMA_SUPPORT_FILES: Tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "config.json",
)

#: 256 MB: these are tokenizer/config files, not weights.  A larger response is
#: refused rather than streamed, because this tool is not a weight downloader.
MAX_MIRROR_FILE_BYTES = 256 * 1024 * 1024

# -- per-file verdicts ----------------------------------------------------------

IDENTICAL = "identical"
DIFFERS = "differs"
OFFICIAL_HASH_UNOBTAINABLE = "official_hash_unobtainable"
OFFICIAL_FILE_ABSENT = "official_file_absent"
MIRROR_UNREACHABLE = "mirror_unreachable"

#: Which official hash a match was proved against.  Recorded per file so a reader
#: never has to guess which channel carried the proof.
CHANNEL_LFS_SHA256 = "official_lfs_sha256"
CHANNEL_GIT_BLOB_SHA1 = "official_git_blob_sha1"

SIZE_IS_NOT_IDENTITY_NOTE = (
    "A matching byte length is recorded but never counts as identity: two different files of equal length are "
    "trivial to produce. Only a match against a hash retrieved from the official repository counts."
)

GIT_BLOB_SHA1_NOTE = (
    "A git blob SHA-1 is a content hash the Hub publishes for regular files. SHA-1 is not collision-resistant "
    "against a determined adversary, so a match here is treated as identity of the retrieved content together with "
    "an exact byte-length match, and the channel is recorded per file rather than hidden."
)

UNPROVEN_NOTE = (
    "byte_identity is unproven. The mirror may not be substituted for the official repository on the strength of "
    "this report: spec section 1 forbids claiming byte identity without a saved comparison, and an unprovable "
    "comparison is reported as unproven rather than assumed."
)


class MirrorComparisonError(RuntimeError):
    """The comparison cannot be set up as asked."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_commit(value: Optional[str]) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def git_blob_sha1(data: bytes) -> str:
    """Git's own object hash for a blob: ``sha1("blob <len>\\0" + data)``."""

    header = ("blob %d\0" % len(data)).encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - compared against git's own object id


# -- targets --------------------------------------------------------------------


@dataclass(frozen=True)
class MirrorTarget:
    """One official/mirror pair, its file list, and the licence that governs it."""

    asset: str
    official_plan_name: str
    mirror_plan_name: str
    files: Tuple[str, ...]
    governing_license: str
    governing_license_note: str

    def report_path(self) -> str:
        return mirror_evidence_path(self.asset)


MIRROR_TARGETS: Tuple[MirrorTarget, ...] = (
    MirrorTarget(
        asset="paligemma",
        official_plan_name="paligemma-official",
        mirror_plan_name="paligemma-mirror-candidate",
        files=PALIGEMMA_SUPPORT_FILES,
        governing_license="Gemma Terms of Use",
        governing_license_note=GEMMA_LICENSE_NOTE,
    ),
)

TARGETS_BY_ASSET: Mapping[str, MirrorTarget] = {target.asset: target for target in MIRROR_TARGETS}


def target_names() -> Tuple[str, ...]:
    return tuple(TARGETS_BY_ASSET)


def get_target(asset: str) -> MirrorTarget:
    try:
        return TARGETS_BY_ASSET[asset]
    except KeyError:
        raise MirrorComparisonError(
            "Unknown mirror target %r; known targets: %s" % (asset, ", ".join(target_names()))
        ) from None


# -- official metadata ----------------------------------------------------------


@dataclass(frozen=True)
class OfficialFileMetadata:
    """What the official host published about one file, without a download."""

    path: str
    size: Optional[int] = None
    lfs_sha256: Optional[str] = None
    git_blob_sha1: Optional[str] = None

    @property
    def has_hash(self) -> bool:
        return bool(self.lfs_sha256 or self.git_blob_sha1)

    def payload(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "lfs_sha256": self.lfs_sha256,
            "git_blob_sha1": self.git_blob_sha1,
            "has_hash": self.has_hash,
        }


@dataclass(frozen=True)
class OfficialMetadata:
    """The official side of the comparison: reachable or not, and why not."""

    repo_id: str
    revision: Optional[str]
    reached: bool = False
    gated: Optional[bool] = None
    status: Optional[int] = None
    error: Optional[str] = None
    files: Mapping[str, OfficialFileMetadata] = field(default_factory=dict)

    def payload(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "reached": bool(self.reached),
            "gated": self.gated,
            "status": self.status,
            "error": self.error,
            "files": {name: item.payload() for name, item in sorted(self.files.items())},
        }


def metadata_url(repo_id: str, revision: str, repo_type: RepoType = RepoType.MODEL) -> str:
    """The Hub metadata endpoint, with blob metadata requested."""

    kind = "datasets" if repo_type is RepoType.DATASET else "models"
    return "%s/api/%s/%s/revision/%s?blobs=true" % (HUGGINGFACE_HOST, kind, repo_id, revision)


def file_url(repo_id: str, revision: str, path: str, repo_type: RepoType = RepoType.MODEL) -> str:
    prefix = "datasets/" if repo_type is RepoType.DATASET else ""
    return "%s/%s%s/resolve/%s/%s" % (HUGGINGFACE_HOST, prefix, repo_id, revision, path)


def _hex_or_none(value: Any, length: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if len(candidate) != length or any(character not in "0123456789abcdef" for character in candidate):
        return None
    return candidate


def read_official_metadata(
    repo_id: str,
    revision: Optional[str],
    response: Optional[FetchResponse],
    wanted: Sequence[str],
) -> OfficialMetadata:
    """Interpret the official metadata response, inventing nothing it omits."""

    if response is None:
        return OfficialMetadata(repo_id=repo_id, revision=revision, error="no request was made")
    if not response.ok:
        return OfficialMetadata(
            repo_id=repo_id,
            revision=revision,
            gated=True if response.gated else None,
            status=response.status,
            error=response.error or "HTTP %d" % response.status,
        )
    try:
        payload = json.loads((response.body or b"").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return OfficialMetadata(
            repo_id=repo_id,
            revision=revision,
            reached=True,
            status=response.status,
            error="response body was not JSON",
        )
    if not isinstance(payload, Mapping):
        return OfficialMetadata(
            repo_id=repo_id,
            revision=revision,
            reached=True,
            status=response.status,
            error="response body was not a JSON object",
        )
    siblings = payload.get("siblings")
    wanted_set = set(wanted)
    files: Dict[str, OfficialFileMetadata] = {}
    if isinstance(siblings, (list, tuple)):
        for sibling in siblings:
            if not isinstance(sibling, Mapping):
                continue
            name = sibling.get("rfilename")
            if not isinstance(name, str) or name not in wanted_set:
                continue
            lfs = sibling.get("lfs")
            lfs_sha = None
            if isinstance(lfs, Mapping):
                lfs_sha = _hex_or_none(lfs.get("sha256") or lfs.get("oid"), 64)
            size = sibling.get("size")
            files[name] = OfficialFileMetadata(
                path=name,
                size=int(size) if isinstance(size, int) else None,
                lfs_sha256=lfs_sha,
                git_blob_sha1=_hex_or_none(sibling.get("blobId") or sibling.get("oid"), 40),
            )
    return OfficialMetadata(
        repo_id=repo_id,
        revision=revision,
        reached=True,
        gated=bool(payload.get("gated")) if payload.get("gated") is not None else None,
        status=response.status,
        files=files,
    )


# -- per-file comparison --------------------------------------------------------


@dataclass(frozen=True)
class FileComparison:
    """One file's mirror bytes against the official host's published metadata."""

    path: str
    mirror_bytes: Optional[int] = None
    mirror_sha256: Optional[str] = None
    mirror_git_blob_sha1: Optional[str] = None
    official_bytes: Optional[int] = None
    official_sha256: Optional[str] = None
    official_git_blob_sha1: Optional[str] = None
    verdict: str = OFFICIAL_HASH_UNOBTAINABLE
    identity_channel: Optional[str] = None
    reason: Optional[str] = None

    @property
    def matched(self) -> bool:
        """Identity proved against a retrieved official hash -- nothing weaker."""

        return self.verdict == IDENTICAL and self.identity_channel is not None

    @property
    def size_matched(self) -> Optional[bool]:
        if self.mirror_bytes is None or self.official_bytes is None:
            return None
        return self.mirror_bytes == self.official_bytes

    def payload(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "mirror_bytes": self.mirror_bytes,
            "mirror_sha256": self.mirror_sha256,
            "mirror_git_blob_sha1": self.mirror_git_blob_sha1,
            "official_bytes": self.official_bytes,
            "official_sha256": self.official_sha256,
            "official_git_blob_sha1": self.official_git_blob_sha1,
            "size_matched": self.size_matched,
            "verdict": self.verdict,
            "identity_channel": self.identity_channel,
            "matched": self.matched,
            "reason": self.reason,
        }


def compare_file(
    path: str,
    mirror_body: Optional[bytes],
    official: Optional[OfficialFileMetadata],
    *,
    mirror_error: Optional[str] = None,
    official_reached: bool = True,
    official_error: Optional[str] = None,
) -> FileComparison:
    """One file's verdict.  A size match alone never yields ``identical``.

    An official repository that could not be read at all yields
    ``official_hash_unobtainable`` rather than ``official_file_absent``: not
    knowing whether a file exists is different from knowing it does not.
    """

    if mirror_body is None:
        return FileComparison(
            path=path,
            official_bytes=official.size if official else None,
            official_sha256=official.lfs_sha256 if official else None,
            official_git_blob_sha1=official.git_blob_sha1 if official else None,
            verdict=MIRROR_UNREACHABLE,
            reason=mirror_error or "the mirror did not return this file",
        )
    mirror_sha256 = hashlib.sha256(mirror_body).hexdigest()
    mirror_blob = git_blob_sha1(mirror_body)
    if official is None:
        return FileComparison(
            path=path,
            mirror_bytes=len(mirror_body),
            mirror_sha256=mirror_sha256,
            mirror_git_blob_sha1=mirror_blob,
            verdict=OFFICIAL_FILE_ABSENT if official_reached else OFFICIAL_HASH_UNOBTAINABLE,
            reason=(
                "the official metadata did not list this file, so there is nothing to compare the mirror's bytes "
                "against"
                if official_reached
                else "the official repository's metadata was not readable (%s), so no hash could be retrieved for "
                "this file. %s" % (official_error or "no response", SIZE_IS_NOT_IDENTITY_NOTE)
            ),
        )
    common = {
        "path": path,
        "mirror_bytes": len(mirror_body),
        "mirror_sha256": mirror_sha256,
        "mirror_git_blob_sha1": mirror_blob,
        "official_bytes": official.size,
        "official_sha256": official.lfs_sha256,
        "official_git_blob_sha1": official.git_blob_sha1,
    }
    if not official.has_hash:
        return FileComparison(
            verdict=OFFICIAL_HASH_UNOBTAINABLE,
            reason=(
                "the official repository published no content hash for this file, so identity cannot be proved. "
                + SIZE_IS_NOT_IDENTITY_NOTE
            ),
            **common,
        )
    size_disagrees = official.size is not None and official.size != len(mirror_body)
    if official.lfs_sha256 is not None:
        if official.lfs_sha256 == mirror_sha256 and not size_disagrees:
            return FileComparison(verdict=IDENTICAL, identity_channel=CHANNEL_LFS_SHA256, **common)
        return FileComparison(
            verdict=DIFFERS,
            reason="the mirror's SHA-256 does not match the official LFS SHA-256",
            **common,
        )
    if official.git_blob_sha1 == mirror_blob and not size_disagrees:
        return FileComparison(
            verdict=IDENTICAL,
            identity_channel=CHANNEL_GIT_BLOB_SHA1,
            reason=GIT_BLOB_SHA1_NOTE,
            **common,
        )
    return FileComparison(
        verdict=DIFFERS,
        reason="the mirror's git blob SHA-1 does not match the official blob id",
        **common,
    )


# -- report ---------------------------------------------------------------------


@dataclass(frozen=True)
class MirrorComparisonReport:
    """The saved comparison.  ``byte_identity`` is computed from the files."""

    asset: str
    official_repo_id: str
    official_revision: Optional[str]
    mirror_repo_id: str
    mirror_revision: Optional[str]
    files: Tuple[FileComparison, ...]
    official: OfficialMetadata
    governing_license: str
    governing_license_note: str
    requested_files: Tuple[str, ...] = ()
    executed: bool = False
    generated_at: Optional[str] = None
    host: Optional[str] = None
    report_path: Optional[str] = None

    @property
    def byte_identity(self) -> ByteIdentity:
        """``PROVEN`` only when every requested file matched an official hash."""

        requested = self.requested_files or tuple(item.path for item in self.files)
        if not requested or len(self.files) != len(requested):
            return ByteIdentity.UNPROVEN
        if not all(item.matched for item in self.files):
            return ByteIdentity.UNPROVEN
        return ByteIdentity.PROVEN

    @property
    def files_matched(self) -> int:
        return sum(1 for item in self.files if item.matched)

    def unproven_reasons(self) -> Tuple[str, ...]:
        """Exactly why identity is unproven, per distinct cause."""

        if self.byte_identity is ByteIdentity.PROVEN:
            return ()
        reasons: List[str] = []
        if not self.official.reached:
            reasons.append(
                "the official repository's metadata was not reachable without accepting terms (%s)"
                % (self.official.error or "no response")
            )
        missing = [name for name in self.requested_files if name not in {item.path for item in self.files}]
        if missing:
            reasons.append("no comparison was recorded for %s" % ", ".join(sorted(missing)))
        for item in self.files:
            if item.matched:
                continue
            reasons.append("%s: %s (%s)" % (item.path, item.verdict, item.reason or "no reason recorded"))
        if not reasons:
            reasons.append("no file matched a retrieved official hash")
        return tuple(dict.fromkeys(reasons))

    def evidence(self, report_path: Optional[str] = None) -> MirrorEvidence:
        """The :class:`MirrorEvidence` this report supports, and no more."""

        return MirrorEvidence(
            evidence_path=report_path or self.report_path or mirror_evidence_path(self.asset),
            official_repo_id=self.official_repo_id,
            byte_identity=self.byte_identity,
            files_compared=len(self.requested_files or self.files),
            files_matched=self.files_matched,
            governing_license=self.governing_license,
            unproven_reasons=self.unproven_reasons(),
            official_revision=self.official_revision,
            mirror_revision=self.mirror_revision,
        )

    def payload(self) -> Dict[str, Any]:
        identity = self.byte_identity
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "asset": self.asset,
            "generated_at": self.generated_at or _now(),
            "executed": bool(self.executed),
            "host": self.host,
            "report_path": self.report_path,
            "official_repo_id": self.official_repo_id,
            "official_revision": self.official_revision,
            "mirror_repo_id": self.mirror_repo_id,
            "mirror_revision": self.mirror_revision,
            "requested_files": list(self.requested_files),
            "byte_identity": identity.value,
            "files_compared": len(self.requested_files or self.files),
            "files_matched": self.files_matched,
            "unproven_reasons": list(self.unproven_reasons()),
            "files": [item.payload() for item in self.files],
            "official_metadata": self.official.payload(),
            "license": {
                "governing_license": self.governing_license,
                "governing_license_note": self.governing_license_note,
                "terms_accepted": False,
                "note": (
                    "This report says nothing about accepted terms. It compares bytes. Byte identity would remove "
                    "the gated download, never the licence obligation, and no tool here accepts a licence."
                ),
            },
            "discipline": [
                SIZE_IS_NOT_IDENTITY_NOTE,
                GIT_BLOB_SHA1_NOTE,
                "byte_identity is computed from the per-file verdicts and may only be 'proven' when every "
                "requested file matched a hash retrieved from the official repository.",
            ]
            + ([] if identity is ByteIdentity.PROVEN else [UNPROVEN_NOTE]),
            "plan_update": {
                "mirror_of": None,
                "mirror_evidence": self.evidence().payload(),
                "blocker": None if identity is ByteIdentity.PROVEN else UNPROVEN_BYTE_IDENTITY_BLOCKER,
            },
        }


def save_report(report: MirrorComparisonReport, path: Path) -> str:
    """Write the comparison and return the SHA-256 of the bytes written."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report.payload(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def update_mirror_entry(entry: AssetPlanEntry, report: MirrorComparisonReport, report_path: str) -> AssetPlanEntry:
    """Attach a saved comparison to its mirror row, refusing an inconsistent one.

    ``local_use`` deliberately stays ``REQUIRES_ACCEPTED_TERMS`` even for proven
    identity: the mirror changes which host served the bytes, not the licence that
    governs them.
    """

    if entry.mirror_of is None:
        raise MirrorComparisonError(
            "%s is not marked as a mirror; set mirror_of before attaching a comparison" % entry.name
        )
    evidence = report.evidence(report_path)
    notes = entry.notes + (report.governing_license_note,)
    if report.byte_identity is not ByteIdentity.PROVEN:
        notes = notes + (UNPROVEN_NOTE,)
    updated = replace(
        entry,
        mirror_evidence=evidence,
        local_use=LocalUse.REQUIRES_ACCEPTED_TERMS,
        notes=tuple(dict.fromkeys(notes)),
    )
    problems = updated.errors()
    if problems:
        raise MirrorComparisonError(
            "attaching the comparison to %s would produce an inconsistent row: %s"
            % (entry.name, "; ".join(problems))
        )
    return updated


# -- the comparison itself ------------------------------------------------------


def compare(
    target: MirrorTarget,
    fetcher: Optional[Fetcher],
    *,
    official_repo_id: str,
    official_revision: Optional[str],
    mirror_repo_id: str,
    mirror_revision: Optional[str],
    files: Optional[Sequence[str]] = None,
    report_path: Optional[str] = None,
    generated_at: Optional[str] = None,
    host: Optional[str] = None,
) -> MirrorComparisonReport:
    """Fetch the mirror's bytes, read the official metadata, and compare.

    ``fetcher=None`` performs no retrieval and returns an empty, unproven report,
    which is what the dry run reports.
    """

    wanted = tuple(files) if files else target.files
    if not wanted:
        raise MirrorComparisonError("a comparison needs at least one file path")
    if fetcher is None:
        return MirrorComparisonReport(
            asset=target.asset,
            official_repo_id=official_repo_id,
            official_revision=official_revision,
            mirror_repo_id=mirror_repo_id,
            mirror_revision=mirror_revision,
            files=(),
            official=OfficialMetadata(
                repo_id=official_repo_id,
                revision=official_revision,
                error="dry run: no request was made",
            ),
            governing_license=target.governing_license,
            governing_license_note=target.governing_license_note,
            requested_files=wanted,
            executed=False,
            generated_at=generated_at,
            host=host,
            report_path=report_path or target.report_path(),
        )
    if not _is_commit(official_revision) or not _is_commit(mirror_revision):
        raise MirrorComparisonError(
            "both revisions must be full 40-character commits; a comparison at a mutable branch proves nothing "
            "about a pinned one"
        )

    official_response = fetcher(metadata_url(official_repo_id, official_revision))
    official = read_official_metadata(official_repo_id, official_revision, official_response, wanted)

    comparisons: List[FileComparison] = []
    for path in wanted:
        response = fetcher(file_url(mirror_repo_id, mirror_revision, path))
        body = response.body if response.ok else None
        if body is not None and len(body) > MAX_MIRROR_FILE_BYTES:
            comparisons.append(
                compare_file(
                    path,
                    None,
                    official.files.get(path),
                    mirror_error="the mirror returned %d bytes, above this tool's %d-byte ceiling"
                    % (len(body), MAX_MIRROR_FILE_BYTES),
                    official_reached=official.reached,
                    official_error=official.error,
                )
            )
            continue
        comparisons.append(
            compare_file(
                path,
                body,
                official.files.get(path),
                mirror_error=None if body is not None else response.describe(),
                official_reached=official.reached,
                official_error=official.error,
            )
        )
    return MirrorComparisonReport(
        asset=target.asset,
        official_repo_id=official_repo_id,
        official_revision=official_revision,
        mirror_repo_id=mirror_repo_id,
        mirror_revision=mirror_revision,
        files=tuple(comparisons),
        official=official,
        governing_license=target.governing_license,
        governing_license_note=target.governing_license_note,
        requested_files=wanted,
        executed=True,
        generated_at=generated_at or _now(),
        host=host,
        report_path=report_path or target.report_path(),
    )


# -- command line ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--asset", default="paligemma", help="Named mirror target; default paligemma")
    parser.add_argument("--list-targets", action="store_true", help="List mirror targets and exit")
    parser.add_argument("--official-repo", help="Override the official repository id from the asset plan")
    parser.add_argument("--mirror-repo", help="Override the mirror repository id from the asset plan")
    parser.add_argument("--official-revision", help="Full 40-character commit of the official repository")
    parser.add_argument("--mirror-revision", help="Full 40-character commit of the mirror repository")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="File to compare; repeatable. Default: the seven PaliGemma support files from spec section 1.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Where to write the saved comparison")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable holding an optional Hugging Face token; unset means anonymous",
    )
    parser.add_argument("--execute", action="store_true", help="Actually fetch, and write the report")
    return parser


def _plan_repo(name: str) -> Tuple[str, Optional[str]]:
    entry = get_plan(name)
    return entry.repo_id, entry.revision


def main(argv: Optional[Sequence[str]] = None) -> int:
    import os

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_targets:
        print(
            json.dumps(
                {
                    "targets": [
                        {
                            "asset": target.asset,
                            "official": target.official_plan_name,
                            "mirror": target.mirror_plan_name,
                            "files": list(target.files),
                            "governing_license": target.governing_license,
                            "report_path": target.report_path(),
                        }
                        for target in MIRROR_TARGETS
                    ]
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    try:
        target = get_target(args.asset)
    except MirrorComparisonError as error:
        parser.error(str(error))
        raise AssertionError("unreachable")  # pragma: no cover - parser.error exits

    official_repo, official_pinned = _plan_repo(target.official_plan_name)
    mirror_repo, mirror_pinned = _plan_repo(target.mirror_plan_name)
    official_repo = args.official_repo or official_repo
    mirror_repo = args.mirror_repo or mirror_repo
    official_revision = args.official_revision or official_pinned
    mirror_revision = args.mirror_revision or mirror_pinned
    report_path = (args.report or Path(target.report_path())).as_posix()
    files = tuple(args.file) or target.files

    if not args.execute:
        print(
            json.dumps(
                {
                    "executed": False,
                    "asset": target.asset,
                    "report_path": report_path,
                    "official_repo_id": official_repo,
                    "official_revision": official_revision,
                    "mirror_repo_id": mirror_repo,
                    "mirror_revision": mirror_revision,
                    "revisions_pinned": bool(_is_commit(official_revision) and _is_commit(mirror_revision)),
                    "governing_license": target.governing_license,
                    "governing_license_note": target.governing_license_note,
                    "would_fetch": (
                        [metadata_url(official_repo, official_revision or "UNRESOLVED_REVISION")]
                        + [
                            file_url(mirror_repo, mirror_revision or "UNRESOLVED_REVISION", path)
                            for path in files
                        ]
                    ),
                    "notice": (
                        "Dry run: no network call was made and nothing was written. byte_identity stays unproven "
                        "until a saved comparison shows every file matching a retrieved official hash."
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    token = os.environ.get(args.token_env) or None
    # The support files include a 17 MB tokenizer, so the ceiling is this tool's,
    # not the license resolver's.  An oversize body is refused, never truncated.
    fetcher = urlopen_fetcher(timeout=args.timeout, token=token, max_bytes=MAX_MIRROR_FILE_BYTES)
    try:
        report = compare(
            target,
            fetcher,
            official_repo_id=official_repo,
            official_revision=official_revision,
            mirror_repo_id=mirror_repo,
            mirror_revision=mirror_revision,
            files=files,
            report_path=report_path,
            host=socket.gethostname(),
        )
    except MirrorComparisonError as error:
        parser.error(str(error))
        raise AssertionError("unreachable")  # pragma: no cover - parser.error exits

    digest = save_report(report, Path(report_path))
    print(json.dumps(report.payload(), indent=2), flush=True)
    print("REPORT %s sha256=%s" % (report_path, digest), flush=True)
    print("BYTE_IDENTITY " + report.byte_identity.value, flush=True)
    for reason in report.unproven_reasons():
        print("UNPROVEN " + reason, flush=True)
    print("LICENSE " + report.governing_license_note, flush=True)
    mirror_entry = get_plan(target.mirror_plan_name)
    try:
        updated = update_mirror_entry(mirror_entry, report, report_path)
    except MirrorComparisonError as error:
        print("REFUSED " + str(error), flush=True)
        return 0
    print(
        "ACTION record mirror_evidence=%s on %s in cluster/asset_plan.py; local_use stays %s because the licence "
        "travels with the weights."
        % (json.dumps(updated.mirror_evidence.payload()), mirror_entry.name, updated.local_use.value),
        flush=True,
    )
    if report.byte_identity is not ByteIdentity.PROVEN and mirror_entry.license_status is LicenseStatus.UNRESOLVED:
        print(
            "NOTICE %s stays fetch=False and its license stays unresolved. Run cluster/resolve_licenses.py against "
            "it separately; a comparison is not a licence." % mirror_entry.name,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
