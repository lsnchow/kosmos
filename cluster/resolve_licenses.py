"""Retrieve, hash, and record each pinned repository's actual license terms.

``cluster/asset_plan.py`` starts most rows at ``license_status="advertised_unverified"``:
a model card *claims* a license and nobody checked.  That is not a resolved
license, and :mod:`plumb.protocol` refuses to call such a row ``verified``.  This
tool closes that gap with work we can do ourselves -- no approvals, no requests
to third parties -- by reading the license where the publisher actually put it::

    python -m cluster.resolve_licenses                          # dry run, no network
    python -m cluster.resolve_licenses --asset code-openvla     # one row
    python -m cluster.resolve_licenses --execute                # fetch, hash, write

What it establishes and what it refuses to establish:

* a license file retrieved at the **pinned immutable revision** and hashed turns
  ``advertised_unverified`` into ``verified``, recording the identifier read from
  the text, the file path, its SHA-256, the revision, and the retrieval time;
* an unreachable, absent, or unrecognisable license stays unresolved with the
  reason recorded.  ``Stanford-ILIAD/pretrain_vq`` declares nothing at all
  (``cardData: null``), so its verdict is ``license=None`` with
  ``redistribution=prohibited_pending_resolution`` and a note -- never a guess;
* **local use and redistribution are answered separately.**  Most of these assets
  are publicly published and can be run locally while their redistribution terms
  stay open.  The plan used to conflate the two, which made runnable rows look
  blocked.  :class:`cluster.asset_plan.LocalUse` is the verdict the primary matrix
  actually depends on;
* a repository with no pinned revision is refused rather than read at a mutable
  branch: a license read at ``main`` is not evidence about a pinned commit.  Use
  ``download_assets.py --resolve-revision`` first.

Network access happens only under ``--execute``, matching
``cluster/download_assets.py``: the dry run prints the exact URLs it would fetch
and writes nothing.  The fetcher is injected, so tests exercise every verdict
without a socket.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import socket
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from cluster.asset_plan import (
    GATED_TERMS_BLOCKER,
    LICENSES_EVIDENCE_PATH,
    MISSING_LICENSE_EVIDENCE_BLOCKER,
    UNRESOLVED_LICENSE_BLOCKER,
    Access,
    AssetPlanEntry,
    LicenseEvidence,
    LicenseStatus,
    LocalUse,
    Redistribution,
    RepoType,
    asset_plan_entries,
    get_plan,
    plan_names,
    validate_plan,
)


SCHEMA_VERSION = 1
REPORT_KIND = "plumb_license_resolution"

#: Repository root, so a report can cite a repository-relative evidence path.
REPO_ROOT = Path(__file__).resolve().parents[1]

HUGGINGFACE_HOST = "https://huggingface.co"
GITHUB_API_HOST = "https://api.github.com"
GITHUB_RAW_HOST = "https://raw.githubusercontent.com"

#: Filenames that are license *files*.  ``README.md`` is deliberately absent: a
#: card's YAML front matter is metadata about a license, not the license.
LICENSE_FILE_NAMES: Tuple[str, ...] = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "LICENSE.rst",
    "LICENCE",
    "LICENCE.txt",
    "LICENSE-APACHE",
    "LICENSE-MIT",
    "COPYING",
    "COPYING.txt",
)

#: Retrieved alongside a license file when present, never instead of one.
NOTICE_FILE_NAMES: Tuple[str, ...] = ("NOTICE", "NOTICE.txt")

#: A license file above this size is refused rather than truncated.  A truncated
#: body would hash to something that was never published, which is exactly the
#: kind of fabricated evidence this module exists to prevent.
MAX_LICENSE_FILE_BYTES = 4_000_000

# -- reason codes ---------------------------------------------------------------

NO_PINNED_REVISION = "no_pinned_immutable_revision"
NO_LICENSE_ENDPOINT = "no_queryable_license_endpoint"
ENDPOINT_UNREACHABLE = "license_endpoint_unreachable"
GATED_METADATA = "license_metadata_requires_accepted_terms"
CARD_DATA_NULL = "card_data_null_no_license_declared"
CARD_DECLARES_OTHER = "card_declares_license_other_which_names_nothing"
CARD_DECLARES_OTHER_WITH_NAME = "card_declares_other_and_names_the_license_off_repo"
CARD_ONLY = "card_advertises_a_license_but_no_license_file_was_retrieved"
TEXT_UNIDENTIFIED = "license_file_retrieved_but_text_not_identified"
TEXT_AMBIGUOUS = "license_text_matched_more_than_one_signature"

CARD_NON_IDENTIFIERS = frozenset({"other", "unknown", "none", "unlicensed", "proprietary"})

LOCAL_USE_NOTE = (
    "local_use and redistribution are independent verdicts. PLUMB runs these bytes locally and redistributes "
    "nothing, so an open redistribution term does not block the primary matrix."
)


class LicenseResolutionError(RuntimeError):
    """A resolution would have written an internally inconsistent plan row."""


# -- license identification -----------------------------------------------------


@dataclass(frozen=True)
class LicenseSignature:
    """Phrases that must and must not appear for one identifiable license.

    Matching is conservative on purpose.  A text that matches nothing, or matches
    more than one signature, is reported unidentified rather than guessed at.
    """

    identifier: str
    required: Tuple[str, ...]
    forbidden: Tuple[str, ...] = ()
    spdx_listed: bool = True
    permits_redistribution: bool = True

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        if any(phrase in lowered for phrase in self.forbidden):
            return False
        return all(phrase in lowered for phrase in self.required)


LICENSE_SIGNATURES: Tuple[LicenseSignature, ...] = (
    LicenseSignature(
        identifier="Apache-2.0",
        required=("apache license", "version 2.0, january 2004", "http://www.apache.org/licenses/"),
    ),
    LicenseSignature(
        identifier="MIT",
        required=(
            "permission is hereby granted, free of charge",
            "the above copyright notice and this permission notice shall be included",
            "without warranty of any kind",
        ),
    ),
    LicenseSignature(
        identifier="BSD-3-Clause",
        required=(
            "redistribution and use in source and binary forms",
            "endorse or promote products derived",
        ),
    ),
    LicenseSignature(
        identifier="BSD-2-Clause",
        required=(
            "redistribution and use in source and binary forms",
            "redistributions in binary form must reproduce",
        ),
        forbidden=("endorse or promote products derived",),
    ),
    LicenseSignature(
        identifier="OpenMDW-1.1",
        required=("openmdw license", "version 1.1"),
        spdx_listed=False,
    ),
    LicenseSignature(
        identifier="CreativeML Open RAIL++-M",
        required=("creativeml open rail++-m license",),
        spdx_listed=False,
        # Use-based restrictions travel with any copy, so redistribution is not a
        # plain grant and stays a human decision.
        permits_redistribution=False,
    ),
    LicenseSignature(
        identifier="CC-BY-4.0",
        required=("creative commons attribution 4.0 international",),
        forbidden=("noncommercial", "sharealike", "no derivatives"),
    ),
    LicenseSignature(
        identifier="Gemma Terms of Use",
        required=("gemma terms of use",),
        spdx_listed=False,
        permits_redistribution=False,
    ),
)

SIGNATURES_BY_IDENTIFIER: Mapping[str, LicenseSignature] = {
    signature.identifier: signature for signature in LICENSE_SIGNATURES
}


def _folded(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def canonical_card_identifier(value: str) -> str:
    """A card string in the canonical casing of a known identifier, if it is one.

    This only normalises punctuation and case of a value that was read from the
    card -- ``"mit"`` becomes ``"MIT"``.  A card string that matches nothing known
    is returned verbatim rather than reshaped into something it did not say.
    """

    folded = _folded(value)
    for signature in LICENSE_SIGNATURES:
        if folded == _folded(signature.identifier):
            return signature.identifier
    return value


def is_repo_relative(link: str) -> bool:
    """Whether a ``license_link`` points inside the repository.

    An off-repo link names a page on somebody else's mutable website.  Fetching it
    would not be a license file at the pinned revision, so it is recorded and not
    followed.
    """

    candidate = link.strip()
    if not candidate or candidate.startswith(("/", "#", "?")):
        return False
    if "://" in candidate or candidate.lower().startswith(("http:", "https:", "mailto:")):
        return False
    return ".." not in candidate.split("/")


def identify_license(text: str) -> Tuple[Optional[LicenseSignature], Optional[str]]:
    """The one signature this text matches, or ``None`` plus the reason why not."""

    matched = [signature for signature in LICENSE_SIGNATURES if signature.matches(text)]
    if len(matched) == 1:
        return matched[0], None
    if not matched:
        return None, TEXT_UNIDENTIFIED
    return None, "%s: %s" % (TEXT_AMBIGUOUS, ", ".join(signature.identifier for signature in matched))


# -- retrieval ------------------------------------------------------------------


@dataclass(frozen=True)
class FetchResponse:
    """One retrieval result.  ``body`` is ``None`` when nothing was retrieved."""

    url: str
    status: int
    body: Optional[bytes] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.body is not None

    @property
    def gated(self) -> bool:
        return self.status in (401, 403)

    def describe(self) -> str:
        if self.error:
            return "%s (HTTP %d: %s)" % (self.url, self.status, self.error)
        return "%s (HTTP %d)" % (self.url, self.status)


#: A fetcher takes a URL and returns a :class:`FetchResponse`.  Injecting it is
#: what keeps every test off the network.
Fetcher = Callable[[str], FetchResponse]


def urlopen_fetcher(
    timeout: float = 30.0,
    token: Optional[str] = None,
    max_bytes: int = MAX_LICENSE_FILE_BYTES,
) -> Fetcher:
    """A real HTTP fetcher.  Built only under ``--execute``; never at import.

    A response larger than ``max_bytes`` is **refused**, not truncated.  Hashing a
    truncated body would produce a digest of bytes nobody published and turn a
    size limit into a false mismatch, so the limit surfaces as an error instead.
    """

    import urllib.error
    import urllib.request

    headers = {"User-Agent": "plumb-license-resolver/1"}
    if token:
        headers["Authorization"] = "Bearer " + token

    def fetch(url: str) -> FetchResponse:
        request = urllib.request.Request(url, headers=dict(headers))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    return FetchResponse(
                        url=url,
                        status=int(response.status),
                        error="response exceeds this tool's %d-byte ceiling; refused rather than truncated"
                        % max_bytes,
                    )
                return FetchResponse(url=url, status=int(response.status), body=body)
        except urllib.error.HTTPError as error:  # noqa: PERF203 - each branch records a distinct verdict
            return FetchResponse(url=url, status=int(error.code), error=str(error.reason))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            return FetchResponse(url=url, status=0, error=str(error))

    return fetch


@dataclass(frozen=True)
class Probe:
    """One URL this tool would fetch, and what it is for."""

    url: str
    purpose: str
    file_path: Optional[str] = None

    def payload(self) -> Dict[str, Any]:
        return {"url": self.url, "purpose": self.purpose, "file_path": self.file_path}


def _github_owner_repo(entry: AssetPlanEntry) -> Optional[Tuple[str, str]]:
    tail = entry.repo_id.split("github.com/", 1)[-1].strip("/")
    parts = [part for part in tail.split("/") if part]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def metadata_probe(entry: AssetPlanEntry) -> Optional[Probe]:
    """The repository-metadata URL at the pinned revision, if one exists."""

    if entry.revision is None:
        return None
    if entry.repo_type is RepoType.MODEL:
        return Probe(
            url="%s/api/models/%s/revision/%s" % (HUGGINGFACE_HOST, entry.repo_id, entry.revision),
            purpose="hf_repo_metadata",
        )
    if entry.repo_type is RepoType.DATASET:
        return Probe(
            url="%s/api/datasets/%s/revision/%s" % (HUGGINGFACE_HOST, entry.repo_id, entry.revision),
            purpose="hf_repo_metadata",
        )
    if entry.repo_type is RepoType.GIT:
        owner_repo = _github_owner_repo(entry)
        if owner_repo is None:
            return None
        return Probe(
            url="%s/repos/%s/%s/license?ref=%s" % (GITHUB_API_HOST, owner_repo[0], owner_repo[1], entry.revision),
            purpose="github_license_api",
        )
    return None


def license_file_probes(entry: AssetPlanEntry, names: Sequence[str]) -> Tuple[Probe, ...]:
    """Direct URLs for candidate license files at the pinned revision."""

    if entry.revision is None:
        return ()
    probes: List[Probe] = []
    for name in names:
        if entry.repo_type is RepoType.MODEL:
            url = "%s/%s/resolve/%s/%s" % (HUGGINGFACE_HOST, entry.repo_id, entry.revision, name)
        elif entry.repo_type is RepoType.DATASET:
            url = "%s/datasets/%s/resolve/%s/%s" % (HUGGINGFACE_HOST, entry.repo_id, entry.revision, name)
        elif entry.repo_type is RepoType.GIT:
            owner_repo = _github_owner_repo(entry)
            if owner_repo is None:
                continue
            url = "%s/%s/%s/%s/%s" % (GITHUB_RAW_HOST, owner_repo[0], owner_repo[1], entry.revision, name)
        else:
            continue
        probes.append(Probe(url=url, purpose="license_file", file_path=name))
    return tuple(probes)


def probes_for(entry: AssetPlanEntry) -> Tuple[Probe, ...]:
    """Everything this tool would fetch for one row, in order."""

    metadata = metadata_probe(entry)
    probes: List[Probe] = [] if metadata is None else [metadata]
    probes.extend(license_file_probes(entry, LICENSE_FILE_NAMES))
    return tuple(probes)


def refusal_reason(entry: AssetPlanEntry) -> Optional[str]:
    """Why this row cannot be resolved at all, or ``None`` if it can be tried."""

    if entry.revision is None:
        return NO_PINNED_REVISION
    if metadata_probe(entry) is None and not license_file_probes(entry, LICENSE_FILE_NAMES):
        return NO_LICENSE_ENDPOINT
    return None


# -- verdicts -------------------------------------------------------------------


@dataclass(frozen=True)
class RepoMetadata:
    """What a metadata probe established.  Absent values stay ``None``."""

    reached: bool = False
    gated: Optional[bool] = None
    card_license: Optional[str] = None
    #: Hugging Face's convention for a license it does not have an id for is
    #: ``license: other`` plus ``license_name`` and ``license_link``.  Reading only
    #: the first field would report "declares nothing" about a card that names one.
    card_license_name: Optional[str] = None
    card_license_link: Optional[str] = None
    card_data_present: Optional[bool] = None
    license_file_names: Tuple[str, ...] = ()
    siblings_listed: bool = False
    reported_spdx: Optional[str] = None
    inline_license_path: Optional[str] = None
    inline_license_bytes: Optional[bytes] = None
    status: Optional[int] = None
    error: Optional[str] = None

    def payload(self) -> Dict[str, Any]:
        return {
            "reached": bool(self.reached),
            "gated": self.gated,
            "card_license": self.card_license,
            "card_license_name": self.card_license_name,
            "card_license_link": self.card_license_link,
            "card_data_present": self.card_data_present,
            "license_file_names": list(self.license_file_names),
            "siblings_listed": bool(self.siblings_listed),
            "host_reported_spdx": self.reported_spdx,
            "status": self.status,
            "error": self.error,
        }


def _decode_json(body: Optional[bytes]) -> Optional[Mapping[str, Any]]:
    if not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _license_names_from_siblings(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    siblings = payload.get("siblings")
    if not isinstance(siblings, (list, tuple)):
        return ()
    known = {name.lower() for name in LICENSE_FILE_NAMES}
    found: List[str] = []
    for sibling in siblings:
        name = sibling.get("rfilename") if isinstance(sibling, Mapping) else sibling
        if isinstance(name, str) and name.lower() in known:
            found.append(name)
    return tuple(found)


def read_metadata(probe: Optional[Probe], response: Optional[FetchResponse]) -> RepoMetadata:
    """Interpret one metadata response without inventing anything absent."""

    if probe is None or response is None:
        return RepoMetadata()
    if not response.ok:
        return RepoMetadata(
            reached=False,
            gated=True if response.gated else None,
            status=response.status,
            error=response.error,
        )
    payload = _decode_json(response.body)
    if payload is None:
        return RepoMetadata(reached=True, status=response.status, error="response body was not JSON")
    if probe.purpose == "github_license_api":
        return _github_license_metadata(payload, response.status)
    card = payload.get("cardData")
    card_present = isinstance(card, Mapping)

    def card_string(key: str) -> Optional[str]:
        value = card.get(key) if card_present else None
        return str(value).strip() if isinstance(value, str) and value.strip() else None

    return RepoMetadata(
        reached=True,
        gated=bool(payload.get("gated")) if payload.get("gated") is not None else None,
        card_license=card_string("license"),
        card_license_name=card_string("license_name"),
        card_license_link=card_string("license_link"),
        card_data_present=card_present,
        license_file_names=_license_names_from_siblings(payload),
        siblings_listed=isinstance(payload.get("siblings"), (list, tuple)),
        status=response.status,
    )


def _github_license_metadata(payload: Mapping[str, Any], status: int) -> RepoMetadata:
    """GitHub's license endpoint returns the file itself, so read it in place."""

    license_block = payload.get("license")
    reported = None
    if isinstance(license_block, Mapping):
        candidate = license_block.get("spdx_id")
        if isinstance(candidate, str) and candidate.strip() and candidate != "NOASSERTION":
            reported = candidate
    path = payload.get("path")
    content = payload.get("content")
    decoded: Optional[bytes] = None
    if isinstance(content, str) and str(payload.get("encoding", "base64")) == "base64":
        try:
            decoded = base64.b64decode(content, validate=False)
        except (binascii.Error, ValueError):
            decoded = None
    return RepoMetadata(
        reached=True,
        card_license=reported,
        card_data_present=reported is not None,
        reported_spdx=reported,
        inline_license_path=str(path) if isinstance(path, str) and path else None,
        inline_license_bytes=decoded,
        status=status,
    )


@dataclass(frozen=True)
class LicenseResolution:
    """One row's verdict: what was retrieved, what it means, and what stays open.

    ``license_status`` reaches ``verified`` only through
    :attr:`evidence` naming a license file that was retrieved and hashed.  Every
    other outcome keeps the row's existing claim and records ``reason``.
    """

    name: str
    repo_id: str
    repo_type: str
    revision: Optional[str]
    license: Optional[str]
    license_status: LicenseStatus
    redistribution: Redistribution
    local_use: LocalUse
    evidence: Optional[LicenseEvidence] = None
    metadata: RepoMetadata = RepoMetadata()
    reason: Optional[str] = None
    attempted: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    spdx_listed: Optional[bool] = None
    #: True when the revision came from ``--revision NAME=SHA`` rather than from a
    #: pin in the plan.  A looked-up commit is not a reviewed pin.
    revision_overridden: bool = False
    #: Upstream NOTICE files found at the pinned revision, with their SHA-256, so
    #: an authorized archival copy can preserve them.
    notice_files: Tuple[Mapping[str, Any], ...] = ()

    @property
    def verified(self) -> bool:
        return self.license_status is LicenseStatus.VERIFIED

    @property
    def local_use_permitted(self) -> bool:
        return self.local_use in (LocalUse.PERMITTED_BY_LICENSE, LocalUse.PERMITTED_LOCAL_ONLY_NO_EXPRESS_TERMS)

    def payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "license": self.license,
            "license_status": self.license_status.value,
            "spdx_listed": self.spdx_listed,
            "redistribution": self.redistribution.value,
            "local_use": self.local_use.value,
            "local_use_permitted": self.local_use_permitted,
            "verified": self.verified,
            "reason": self.reason,
            "revision_overridden": bool(self.revision_overridden),
            "evidence": self.evidence.payload() if self.evidence is not None else None,
            "metadata": self.metadata.payload(),
            "attempted_urls": list(self.attempted),
            "notice_files": [dict(item) for item in self.notice_files],
            "notes": list(self.notes),
            "plan_update": {
                "license": self.license,
                "license_status": self.license_status.value,
                "redistribution": self.redistribution.value,
                "local_use": self.local_use.value,
                "license_evidence": self.evidence.payload() if self.evidence is not None else None,
            },
        }


def _base_local_use(entry: AssetPlanEntry, reachable: bool) -> LocalUse:
    """Local use derived from access alone, before any license is read."""

    if entry.access is Access.GATED_ACCEPT_TERMS or entry.mirror_of is not None:
        return LocalUse.REQUIRES_ACCEPTED_TERMS
    if reachable and entry.access in (Access.PUBLIC, Access.SOURCE_REPOSITORY):
        return LocalUse.PERMITTED_LOCAL_ONLY_NO_EXPRESS_TERMS
    return entry.local_use


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_entry(
    entry: AssetPlanEntry,
    fetcher: Optional[Fetcher] = None,
    *,
    evidence_path: str = LICENSES_EVIDENCE_PATH,
    retrieved_at: Optional[str] = None,
) -> LicenseResolution:
    """Resolve one row's license.  ``fetcher=None`` performs no retrieval at all."""

    def unchanged(reason: str, attempted: Tuple[str, ...] = (), notes: Tuple[str, ...] = ()) -> LicenseResolution:
        return LicenseResolution(
            name=entry.name,
            repo_id=entry.repo_id,
            repo_type=entry.repo_type.value,
            revision=entry.revision,
            license=entry.license,
            license_status=entry.license_status,
            redistribution=entry.redistribution,
            local_use=entry.local_use,
            evidence=LicenseEvidence(evidence_path=evidence_path, revision=entry.revision, reason=reason),
            reason=reason,
            attempted=attempted,
            notes=notes,
        )

    refusal = refusal_reason(entry)
    if refusal == NO_PINNED_REVISION:
        return unchanged(
            NO_PINNED_REVISION,
            notes=(
                "A license read at a mutable branch is not evidence about a pinned commit. Run "
                "'python -m cluster.download_assets --from-plan %s --resolve-revision', review the commit, record "
                "it as this entry's revision, then re-run this tool." % entry.name,
            ),
        )
    if refusal == NO_LICENSE_ENDPOINT:
        return unchanged(
            NO_LICENSE_ENDPOINT,
            notes=(
                "%s is fetched outside the Hub and exposes no license endpoint. Extract the archive, read the "
                "notices it ships, and record them here." % entry.repo_id,
            ),
        )
    if fetcher is None:
        return unchanged(
            "not_retrieved_dry_run",
            attempted=tuple(probe.url for probe in probes_for(entry)),
            notes=("Dry run: no network call was made and nothing was written.",),
        )

    attempted: List[str] = []
    metadata_request = metadata_probe(entry)
    metadata_response = None
    if metadata_request is not None:
        metadata_response = fetcher(metadata_request.url)
        attempted.append(metadata_request.url)
    metadata = read_metadata(metadata_request, metadata_response)

    # An in-repo license_link and the host's own file listing are better than
    # guessing.  Guess only when the host published no listing at all, so a
    # repository that demonstrably has no license file is not probed ten times.
    candidates: List[str] = []
    if metadata.card_license_link and is_repo_relative(metadata.card_license_link):
        candidates.append(metadata.card_license_link)
    candidates.extend(name for name in metadata.license_file_names if name not in candidates)
    if not metadata.siblings_listed:
        candidates.extend(name for name in LICENSE_FILE_NAMES if name not in candidates)
    retrieved_path: Optional[str] = None
    retrieved_bytes: Optional[bytes] = None
    retrieved_url: Optional[str] = None
    if metadata.inline_license_bytes is not None and metadata.inline_license_path:
        retrieved_path = metadata.inline_license_path
        retrieved_bytes = metadata.inline_license_bytes
        retrieved_url = metadata_request.url if metadata_request is not None else None
    else:
        for probe in license_file_probes(entry, candidates):
            response = fetcher(probe.url)
            attempted.append(probe.url)
            if response.ok:
                retrieved_path, retrieved_bytes, retrieved_url = probe.file_path, response.body, probe.url
                break
            if response.gated and metadata.gated is None:
                metadata = replace(metadata, gated=True)

    notices: List[Mapping[str, Any]] = []
    if retrieved_bytes is not None:
        for probe in license_file_probes(entry, NOTICE_FILE_NAMES):
            response = fetcher(probe.url)
            attempted.append(probe.url)
            if response.ok and response.body is not None:
                notices.append(
                    {
                        "path": probe.file_path,
                        "sha256": hashlib.sha256(response.body).hexdigest(),
                        "bytes": len(response.body),
                        "source_url": probe.url,
                    }
                )

    stamp = retrieved_at or _now()
    return _verdict(
        entry,
        metadata=metadata,
        retrieved_path=retrieved_path,
        retrieved_bytes=retrieved_bytes,
        retrieved_url=retrieved_url,
        attempted=tuple(attempted),
        evidence_path=evidence_path,
        retrieved_at=stamp,
        notice_files=tuple(notices),
    )


def _verdict(
    entry: AssetPlanEntry,
    *,
    metadata: RepoMetadata,
    retrieved_path: Optional[str],
    retrieved_bytes: Optional[bytes],
    retrieved_url: Optional[str],
    attempted: Tuple[str, ...],
    evidence_path: str,
    retrieved_at: str,
    notice_files: Tuple[Mapping[str, Any], ...] = (),
) -> LicenseResolution:
    """Turn one retrieval into a verdict.  Nothing here invents a license."""

    base_local_use = _base_local_use(entry, metadata.reached)
    notes: List[str] = []
    spdx_listed: Optional[bool] = None

    if retrieved_bytes is not None and retrieved_path is not None:
        digest = hashlib.sha256(retrieved_bytes).hexdigest()
        text = retrieved_bytes.decode("utf-8", errors="replace")
        signature, failure = identify_license(text)
        if signature is None:
            notes.append(
                "A license file was retrieved and hashed but its text was not identified. The bytes are recorded so "
                "a human can read the file this tool could not classify; no identifier is written."
            )
            return LicenseResolution(
                name=entry.name,
                repo_id=entry.repo_id,
                repo_type=entry.repo_type.value,
                revision=entry.revision,
                license=entry.license,
                license_status=entry.license_status,
                redistribution=entry.redistribution,
                local_use=base_local_use,
                evidence=LicenseEvidence(
                    evidence_path=evidence_path,
                    spdx=None,
                    license_file_path=retrieved_path,
                    license_file_sha256=digest,
                    revision=entry.revision,
                    retrieved_at=retrieved_at,
                    source_url=retrieved_url,
                    reason=failure,
                ),
                metadata=metadata,
                reason=failure,
                attempted=attempted,
                notes=tuple(notes),
                notice_files=notice_files,
            )
        spdx_listed = signature.spdx_listed
        if metadata.reported_spdx and metadata.reported_spdx != signature.identifier:
            notes.append(
                "The host reported %r while the retrieved text reads as %r. The text this tool hashed is the "
                "record; the host's label is not." % (metadata.reported_spdx, signature.identifier)
            )
        if metadata.card_license and metadata.card_license.lower() not in (
            signature.identifier.lower(),
            signature.identifier.lower().replace("-", ""),
        ):
            notes.append(
                "The card advertises %r while the retrieved license file reads as %r."
                % (metadata.card_license, signature.identifier)
            )
        local_use = (
            LocalUse.PERMITTED_BY_LICENSE
            if base_local_use is LocalUse.PERMITTED_LOCAL_ONLY_NO_EXPRESS_TERMS
            else base_local_use
        )
        redistribution = (
            Redistribution.PERMITTED_WITH_NOTICES
            if signature.permits_redistribution
            else Redistribution.PROHIBITED_PENDING_RESOLUTION
        )
        if signature.permits_redistribution:
            notes.append("Redistribution is permitted with notices; preserve the upstream license and NOTICE files.")
        else:
            notes.append(
                "%s carries obligations beyond attribution, so redistribution stays a human decision. Local use is "
                "unaffected." % signature.identifier
            )
        if not signature.spdx_listed:
            notes.append("%r is not an SPDX-listed identifier; it is the name the license text gives itself." % signature.identifier)
        return LicenseResolution(
            name=entry.name,
            repo_id=entry.repo_id,
            repo_type=entry.repo_type.value,
            revision=entry.revision,
            license=signature.identifier,
            license_status=LicenseStatus.VERIFIED,
            redistribution=redistribution,
            local_use=local_use,
            evidence=LicenseEvidence(
                evidence_path=evidence_path,
                spdx=signature.identifier,
                license_file_path=retrieved_path,
                license_file_sha256=digest,
                revision=entry.revision,
                retrieved_at=retrieved_at,
                source_url=retrieved_url,
            ),
            metadata=metadata,
            reason=None,
            attempted=attempted,
            notes=tuple(notes),
            spdx_listed=spdx_listed,
            notice_files=notice_files,
        )

    def failed(
        reason: str,
        *,
        license: Optional[str],
        license_status: LicenseStatus,
        redistribution: Redistribution,
        local_use: LocalUse,
        extra_notes: Sequence[str] = (),
    ) -> LicenseResolution:
        return LicenseResolution(
            name=entry.name,
            repo_id=entry.repo_id,
            repo_type=entry.repo_type.value,
            revision=entry.revision,
            license=license,
            license_status=license_status,
            redistribution=redistribution,
            local_use=local_use,
            evidence=LicenseEvidence(
                evidence_path=evidence_path,
                spdx=None,
                revision=entry.revision,
                retrieved_at=retrieved_at,
                reason=reason,
            ),
            metadata=metadata,
            reason=reason,
            attempted=attempted,
            notes=tuple(notes) + tuple(extra_notes),
            notice_files=notice_files,
        )

    if metadata.gated:
        return failed(
            GATED_METADATA,
            license=None,
            license_status=LicenseStatus.GATED_TERMS_REQUIRED,
            redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
            local_use=LocalUse.REQUIRES_ACCEPTED_TERMS,
            extra_notes=(
                "%s refused the license request without accepted terms. No tool here can accept them; a human must."
                % entry.repo_id,
            ),
        )
    if not metadata.reached:
        return failed(
            "%s: %s" % (ENDPOINT_UNREACHABLE, metadata.error or "no response"),
            license=entry.license,
            license_status=entry.license_status,
            redistribution=entry.redistribution,
            local_use=entry.local_use,
            extra_notes=(
                "Nothing was retrieved, so nothing changed. An unreachable license stays unresolved rather than "
                "being filled in from the model card or from memory.",
            ),
        )
    card = (metadata.card_license or "").strip()
    named = (metadata.card_license_name or "").strip()
    link = (metadata.card_license_link or "").strip()
    advertised_redistribution = (
        entry.redistribution
        if entry.redistribution is not Redistribution.UNRESOLVED
        else Redistribution.LOCAL_USE_ONLY
    )
    if not card:
        return failed(
            CARD_DATA_NULL,
            license=None,
            license_status=LicenseStatus.ABSENT_CARD_DATA_NULL,
            redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
            local_use=base_local_use,
            extra_notes=(
                "%s declares no license: the repository was reached at %s and neither a license file nor a card "
                "license was present. license stays null and redistribution stays prohibited_pending_resolution."
                % (entry.repo_id, entry.revision),
                LOCAL_USE_NOTE,
            ),
        )
    if card.lower() in CARD_NON_IDENTIFIERS:
        if not named:
            return failed(
                CARD_DECLARES_OTHER,
                license=None,
                license_status=LicenseStatus.UNRESOLVED,
                redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
                local_use=base_local_use,
                extra_notes=(
                    "%s declares license=%r, which names nothing, and supplies no license_name. The card data is "
                    "present, so this is unresolved rather than absent." % (entry.repo_id, card),
                    LOCAL_USE_NOTE,
                ),
            )
        return failed(
            CARD_DECLARES_OTHER_WITH_NAME,
            license=canonical_card_identifier(named),
            license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
            redistribution=advertised_redistribution,
            local_use=base_local_use,
            extra_notes=(
                "The card sets license=%r and names %r at %s. That name is recorded verbatim and stays "
                "advertised_unverified: a card is metadata about a license, not the license."
                % (card, named, entry.revision),
                (
                    "license_link %r is off-repo. An external page is mutable and is not a license file at the "
                    "pinned revision, so it was recorded and not followed; read it and record the result by hand "
                    "to finish this row." % link
                    if link and not is_repo_relative(link)
                    else "license_link %r is in-repo but was not retrievable at the pinned revision." % link
                ),
                LOCAL_USE_NOTE,
            ),
        )
    return failed(
        CARD_ONLY,
        license=canonical_card_identifier(card),
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=advertised_redistribution,
        local_use=base_local_use,
        extra_notes=(
            "The card advertises %r at %s but no license file was retrieved. A card is metadata about a license, "
            "not the license, so the status stays advertised_unverified." % (card, entry.revision),
            LOCAL_USE_NOTE,
        ),
    )


# -- applying resolutions to the plan -------------------------------------------


def update_entry(entry: AssetPlanEntry, resolution: LicenseResolution) -> AssetPlanEntry:
    """Apply one resolution to its plan row, refusing an inconsistent result.

    This is the only supported way to move a row to ``verified``: the updated
    entry is validated with :meth:`AssetPlanEntry.errors`, which requires the
    cited evidence, so a status can never be written without it.
    """

    if resolution.name != entry.name:
        raise LicenseResolutionError(
            "resolution for %r cannot be applied to plan row %r" % (resolution.name, entry.name)
        )
    blockers = [blocker for blocker in entry.blockers]
    if resolution.license_status is not LicenseStatus.UNRESOLVED and UNRESOLVED_LICENSE_BLOCKER in blockers:
        blockers.remove(UNRESOLVED_LICENSE_BLOCKER)
    if resolution.license_status is LicenseStatus.UNRESOLVED and UNRESOLVED_LICENSE_BLOCKER not in blockers:
        blockers.append(UNRESOLVED_LICENSE_BLOCKER)
    if resolution.local_use is LocalUse.REQUIRES_ACCEPTED_TERMS and GATED_TERMS_BLOCKER not in blockers:
        blockers.append(GATED_TERMS_BLOCKER)
    updated = replace(
        entry,
        license=resolution.license,
        license_status=resolution.license_status,
        redistribution=resolution.redistribution,
        local_use=resolution.local_use,
        license_evidence=resolution.evidence,
        blockers=tuple(dict.fromkeys(blockers)),
        notes=tuple(dict.fromkeys(entry.notes + resolution.notes)),
    )
    problems = updated.errors()
    if problems:
        raise LicenseResolutionError(
            "resolving %s would produce an inconsistent plan row: %s" % (entry.name, "; ".join(problems))
        )
    return updated


def apply_resolutions(
    resolutions: Sequence[LicenseResolution],
    entries: Optional[Mapping[str, AssetPlanEntry]] = None,
) -> Tuple[AssetPlanEntry, ...]:
    """Every resolved row, updated.  Raises rather than writing a bad row."""

    return tuple(
        update_entry(get_plan(resolution.name) if entries is None else entries[resolution.name], resolution)
        for resolution in resolutions
    )


def evidence_gaps(entries: Sequence[AssetPlanEntry], root: Optional[Path] = None) -> Tuple[str, ...]:
    """Verified rows whose cited evidence file is missing from disk.

    The plan's own validation checks the evidence *fields*, because
    ``validate_plan()`` must stay free of filesystem access.  This is the
    complementary on-disk check: a status of ``verified`` that cites a report
    nobody saved is not evidence.
    """

    base = REPO_ROOT if root is None else Path(root)
    gaps: List[str] = []
    for entry in entries:
        if entry.license_status is not LicenseStatus.VERIFIED:
            continue
        evidence = entry.license_evidence
        if evidence is None or not evidence.evidence_path:
            gaps.append("%s: verified without a cited evidence path" % entry.name)
            continue
        candidate = Path(evidence.evidence_path)
        if not candidate.is_absolute():
            candidate = base / candidate
        if not candidate.is_file():
            gaps.append("%s: cites %s, which does not exist" % (entry.name, evidence.evidence_path))
    return tuple(gaps)


# -- report ---------------------------------------------------------------------


def build_report(
    resolutions: Sequence[LicenseResolution],
    *,
    executed: bool,
    generated_at: Optional[str] = None,
    host: Optional[str] = None,
    evidence_path: str = LICENSES_EVIDENCE_PATH,
) -> Dict[str, Any]:
    """The ``results/provenance/licenses.json`` payload."""

    verified = [item.name for item in resolutions if item.verified]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "generated_at": generated_at or _now(),
        "executed": bool(executed),
        "host": host,
        "evidence_path": evidence_path,
        "summary": {
            "assets": len(resolutions),
            "verified": len(verified),
            "verified_assets": sorted(verified),
            "local_use_permitted": sorted(item.name for item in resolutions if item.local_use_permitted),
            "requires_accepted_terms": sorted(
                item.name for item in resolutions if item.local_use is LocalUse.REQUIRES_ACCEPTED_TERMS
            ),
            "redistribution_open": sorted(
                item.name
                for item in resolutions
                if item.redistribution
                in (Redistribution.PROHIBITED_PENDING_RESOLUTION, Redistribution.UNRESOLVED)
            ),
            "unresolved_reasons": {item.name: item.reason for item in resolutions if item.reason},
            "blocker_cleared_by_verification": MISSING_LICENSE_EVIDENCE_BLOCKER,
            "still_missing_license_evidence": sorted(item.name for item in resolutions if not item.verified),
        },
        "discipline": [
            "Only a license file that was retrieved and hashed at the pinned revision produces license_status="
            "'verified'. No identifier is written from memory, from a model card, or from a host's label.",
            "local_use and redistribution are independent verdicts; the primary matrix needs the first.",
            "A row with no pinned immutable revision is refused rather than read at a mutable branch.",
        ],
        "resolutions": [item.payload() for item in resolutions],
    }


def save_report(report: Mapping[str, Any], path: Path) -> str:
    """Write the report and return the SHA-256 of the bytes written."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


# -- command line ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Named entry in cluster.asset_plan; repeatable. Default: every entry.",
    )
    parser.add_argument("--list-assets", action="store_true", help="List planned assets and exit")
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Only rows marked required_for_primary_matrix",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(LICENSES_EVIDENCE_PATH),
        help="Where to write the provenance report under --execute",
    )
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="NAME=SHA",
        help=(
            "Resolve NAME's license at this 40-character commit without adopting it as a plan pin; repeatable. "
            "Use it for a commit looked up with download_assets.py --resolve-revision but not yet reviewed."
        ),
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable holding an optional Hugging Face token; unset means anonymous",
    )
    parser.add_argument("--execute", action="store_true", help="Actually fetch, and write the report")
    return parser


def parse_revision_overrides(values: Sequence[str]) -> Dict[str, str]:
    """``NAME=SHA`` pairs, refusing anything that is not a full commit."""

    overrides: Dict[str, str] = {}
    for value in values:
        name, separator, revision = value.partition("=")
        name, revision = name.strip(), revision.strip().lower()
        if not separator or not name or not revision:
            raise LicenseResolutionError("--revision expects NAME=SHA, got %r" % value)
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise LicenseResolutionError(
                "--revision %s must be a full 40-character commit; a branch name proves nothing about a pin" % name
            )
        entry = get_plan(name)
        if entry.revision is not None and entry.revision != revision:
            raise LicenseResolutionError(
                "--revision %s disagrees with the plan's pinned revision %s" % (name, entry.revision)
            )
        overrides[name] = revision
    return overrides


def apply_revision_overrides(
    entries: Sequence[AssetPlanEntry],
    overrides: Mapping[str, str],
) -> Tuple[Tuple[AssetPlanEntry, ...], Tuple[str, ...]]:
    """Entries with any override applied, plus the names that were overridden."""

    resolved: List[AssetPlanEntry] = []
    overridden: List[str] = []
    for entry in entries:
        revision = overrides.get(entry.name)
        if revision is None or entry.revision == revision:
            resolved.append(entry)
            continue
        resolved.append(replace(entry, revision=revision))
        overridden.append(entry.name)
    return tuple(resolved), tuple(overridden)


def selected_entries(asset: Sequence[str], required_only: bool) -> Tuple[AssetPlanEntry, ...]:
    names = list(asset) or list(plan_names())
    entries = tuple(get_plan(name) for name in names)
    if required_only:
        entries = tuple(entry for entry in entries if entry.required_for_primary_matrix)
    return entries


def main(argv: Optional[Sequence[str]] = None) -> int:
    import os

    parser = build_parser()
    args = parser.parse_args(argv)

    problems = validate_plan()
    if problems:
        print(json.dumps({"plan_problems": list(problems)}, indent=2), flush=True)
        return 2
    if args.list_assets:
        print(json.dumps({"assets": list(plan_names()), "entries": len(asset_plan_entries())}, indent=2), flush=True)
        return 0

    try:
        entries = selected_entries(args.asset, args.required_only)
        overrides = parse_revision_overrides(args.revision)
    except (KeyError, LicenseResolutionError) as error:
        parser.error(str(error))
        raise AssertionError("unreachable")  # pragma: no cover - parser.error exits
    entries, overridden = apply_revision_overrides(entries, overrides)

    evidence_path = args.report.as_posix()
    if not args.execute:
        plan = {
            "executed": False,
            "report_path": evidence_path,
            "assets": [
                {
                    "name": entry.name,
                    "repo_id": entry.repo_id,
                    "revision": entry.revision,
                    "license_status": entry.license_status.value,
                    "local_use": entry.local_use.value,
                    "revision_overridden": entry.name in overridden,
                    "refusal": refusal_reason(entry),
                    "would_fetch": [probe.payload() for probe in probes_for(entry)],
                }
                for entry in entries
            ],
            "notice": (
                "Dry run: no network call was made and nothing was written. Re-run with --execute to retrieve and "
                "hash the license files listed above."
            ),
        }
        print(json.dumps(plan, indent=2), flush=True)
        return 0

    token = os.environ.get(args.token_env) or None
    fetcher = urlopen_fetcher(timeout=args.timeout, token=token)
    resolutions = [
        replace(
            resolve_entry(entry, fetcher, evidence_path=evidence_path),
            revision_overridden=entry.name in overridden,
        )
        for entry in entries
    ]
    report = build_report(
        resolutions,
        executed=True,
        host=socket.gethostname(),
        evidence_path=evidence_path,
    )
    if overridden:
        report["revision_overrides"] = {
            name: overrides[name] for name in overridden
        }
        report["revision_override_note"] = (
            "These rows were resolved at a commit supplied on the command line, not at a revision the plan pins. "
            "Review the commit and record it in cluster/asset_plan.py before treating it as a pin."
        )
    report["authenticated"] = bool(token)
    if token:
        report["authentication_note"] = (
            "A Hugging Face token was supplied. A token belonging to an account that already accepted a gated "
            "repository's terms is not the same as an unauthenticated public read; gated rows resolved this way "
            "still require those accepted terms."
        )
    digest = save_report(report, args.report)
    print(json.dumps(report, indent=2), flush=True)
    print("REPORT %s sha256=%s" % (args.report, digest), flush=True)

    by_name = {entry.name: entry for entry in entries}
    updated: List[AssetPlanEntry] = []
    for resolution in resolutions:
        try:
            updated.append(update_entry(by_name[resolution.name], resolution))
        except LicenseResolutionError as error:
            print("REFUSED " + str(error), flush=True)
    for gap in evidence_gaps(updated):
        print("EVIDENCE GAP " + gap, flush=True)
    print(
        "ACTION apply the plan_update blocks above to cluster/asset_plan.py, citing %s as license_evidence. "
        "A status is not verified until the row carries that evidence." % evidence_path,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
