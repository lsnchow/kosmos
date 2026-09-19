"""License resolution and mirror byte-identity behaviour.

The theme is that neither tool may overclaim.  A license nobody retrieved is not a
license, a model card is not a license file, a matching byte length is not byte
identity, and a status of ``verified`` without a saved evidence file is a typo
rather than a fact.  Every fetch here is injected: no test touches the network.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace

import pytest

import cluster.compare_mirror as mirror
import cluster.resolve_licenses as licenses
from cluster.asset_plan import (
    GATED_TERMS_BLOCKER,
    LOCAL_USE_BLOCKER,
    MISSING_LICENSE_EVIDENCE_BLOCKER,
    UNPROVEN_BYTE_IDENTITY_BLOCKER,
    UNRESOLVED_REVISION_BLOCKER,
    Access,
    AssetPlanEntry,
    ByteIdentity,
    LicenseEvidence,
    LicenseStatus,
    LocalUse,
    MirrorEvidence,
    Redistribution,
    RepoType,
    asset_plan_entries,
    get_plan,
    plan_names,
    validate_plan,
)


REVISION = "a" * 40
MIRROR_REVISION = "b" * 40

APACHE_TEXT = b"""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
"""

MIT_TEXT = b"""MIT License

Copyright (c) 2024 Somebody

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction.

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED.
"""

GEMMA_TEXT = b"Gemma Terms of Use\n\nLast modified: some date. These terms govern use of the Gemma Services.\n"


class RecordingFetcher:
    """An injected fetcher.  It answers from a table and records every URL."""

    def __init__(self, responses=None, default_status=404, default_error="not found"):
        self.responses = dict(responses or {})
        self.default_status = default_status
        self.default_error = default_error
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url in self.responses:
            status, body = self.responses[url]
            return licenses.FetchResponse(url=url, status=status, body=body)
        return licenses.FetchResponse(url=url, status=self.default_status, error=self.default_error)


def pinned(name, **changes):
    """One plan row with a revision pinned, so it is eligible for resolution."""

    values = {"revision": REVISION}
    values.update(changes)
    return replace(get_plan(name), **values)


def github_license_response(text, spdx="MIT", path="LICENSE"):
    return json.dumps(
        {
            "license": {"spdx_id": spdx},
            "path": path,
            "encoding": "base64",
            "content": base64.b64encode(text).decode("ascii"),
        }
    ).encode("utf-8")


def hub_metadata(
    card_license=None,
    card_data=True,
    siblings=(),
    gated=False,
    license_name=None,
    license_link=None,
):
    payload = {"gated": gated, "siblings": [{"rfilename": name} for name in siblings]}
    card = {}
    if card_license:
        card["license"] = card_license
    if license_name:
        card["license_name"] = license_name
    if license_link:
        card["license_link"] = license_link
    payload["cardData"] = card if card_data else None
    return json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# license identification
# ---------------------------------------------------------------------------


def test_license_text_is_identified_only_when_it_is_unambiguous():
    apache, failure = licenses.identify_license(APACHE_TEXT.decode())
    assert apache is not None and apache.identifier == "Apache-2.0" and failure is None
    mit, failure = licenses.identify_license(MIT_TEXT.decode())
    assert mit is not None and mit.identifier == "MIT" and failure is None

    signature, failure = licenses.identify_license("This repository is free to use. Enjoy.")
    assert signature is None
    assert failure == licenses.TEXT_UNIDENTIFIED


def test_the_gemma_terms_are_identified_without_being_called_an_spdx_license():
    signature, failure = licenses.identify_license(GEMMA_TEXT.decode())
    assert failure is None
    assert signature.identifier == "Gemma Terms of Use"
    # Not on the SPDX list, and not a plain redistribution grant.
    assert signature.spdx_listed is False
    assert signature.permits_redistribution is False


def test_a_verified_license_with_obligations_keeps_redistribution_shut():
    entry = pinned("sdxl-vae")
    probe = licenses.license_file_probes(entry, ("LICENSE",))[0]
    text = b"CreativeML Open RAIL++-M License\ndated July 26, 2023\n"
    fetcher = RecordingFetcher({probe.url: (200, text)})

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.license == "CreativeML Open RAIL++-M"
    assert resolution.license_status is LicenseStatus.VERIFIED
    assert resolution.redistribution is Redistribution.PROHIBITED_PENDING_RESOLUTION
    # Local use is what the matrix needs, and a verified licence grants it.
    assert resolution.local_use is LocalUse.PERMITTED_BY_LICENSE
    assert resolution.spdx_listed is False
    assert licenses.update_entry(entry, resolution).errors() == ()


def test_a_license_file_that_cannot_be_classified_is_hashed_but_never_named():
    entry = pinned("code-susie")
    probe = licenses.metadata_probe(entry)
    body = b"You may use this however you like, but we name no license.\n"
    fetcher = RecordingFetcher({probe.url: (200, github_license_response(body, spdx=None))})

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.license_status is LicenseStatus.ADVERTISED_UNVERIFIED
    assert resolution.verified is False
    assert resolution.reason == licenses.TEXT_UNIDENTIFIED
    # The bytes are recorded so a human can read what the tool could not classify.
    assert resolution.evidence.license_file_sha256 == hashlib.sha256(body).hexdigest()
    assert resolution.evidence.spdx is None


# ---------------------------------------------------------------------------
# resolution verdicts
# ---------------------------------------------------------------------------


def test_a_retrieved_license_file_turns_advertised_unverified_into_verified():
    entry = pinned("code-octo")
    probe = licenses.metadata_probe(entry)
    fetcher = RecordingFetcher({probe.url: (200, github_license_response(APACHE_TEXT, spdx="Apache-2.0"))})

    resolution = licenses.resolve_entry(entry, fetcher)

    assert entry.license_status is LicenseStatus.ADVERTISED_UNVERIFIED
    assert resolution.license_status is LicenseStatus.VERIFIED
    assert resolution.license == "Apache-2.0"
    evidence = resolution.evidence
    assert evidence.spdx == "Apache-2.0"
    assert evidence.license_file_path == "LICENSE"
    assert evidence.license_file_sha256 == hashlib.sha256(APACHE_TEXT).hexdigest()
    assert evidence.revision == REVISION
    assert evidence.retrieved_at
    assert evidence.evidence_path == licenses.LICENSES_EVIDENCE_PATH

    updated = licenses.update_entry(entry, resolution)
    assert updated.license_status is LicenseStatus.VERIFIED
    assert MISSING_LICENSE_EVIDENCE_BLOCKER not in updated.verification_blockers()
    assert updated.errors() == ()


def test_an_unreachable_license_stays_unresolved_and_is_never_guessed():
    entry = pinned("code-openvla-mini")
    assert entry.license_status is LicenseStatus.UNRESOLVED
    fetcher = RecordingFetcher(default_status=0, default_error="connection reset")

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.license is None
    assert resolution.license_status is LicenseStatus.UNRESOLVED
    assert resolution.verified is False
    assert resolution.reason.startswith(licenses.ENDPOINT_UNREACHABLE)
    assert "connection reset" in resolution.reason
    assert resolution.evidence.spdx is None
    assert resolution.evidence.license_file_sha256 is None
    # It tried; it simply refuses to fill the gap from the model card or memory.
    assert fetcher.calls


def test_an_unreachable_license_never_promotes_an_advertised_claim():
    entry = pinned("code-susie")
    assert entry.license == "MIT" and entry.license_status is LicenseStatus.ADVERTISED_UNVERIFIED
    fetcher = RecordingFetcher(default_status=500, default_error="server error")

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.license_status is LicenseStatus.ADVERTISED_UNVERIFIED
    assert resolution.verified is False
    assert MISSING_LICENSE_EVIDENCE_BLOCKER in licenses.update_entry(entry, resolution).verification_blockers()


def test_a_model_card_is_metadata_about_a_license_not_the_license():
    entry = pinned("susie-subgoal")
    probe = licenses.metadata_probe(entry)
    fetcher = RecordingFetcher({probe.url: (200, hub_metadata(card_license="mit", siblings=("config.json",)))})

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.reason == licenses.CARD_ONLY
    assert resolution.license == "MIT"
    assert resolution.license_status is LicenseStatus.ADVERTISED_UNVERIFIED
    assert resolution.verified is False
    assert any("not the license" in note for note in resolution.notes)


def test_pretrain_vq_yields_a_null_license_and_prohibited_redistribution():
    entry = pinned("minivla-pretrain-vq")
    probe = licenses.metadata_probe(entry)
    fetcher = RecordingFetcher({probe.url: (200, hub_metadata(card_data=False, siblings=("config.json",)))})

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.repo_id == "Stanford-ILIAD/pretrain_vq"
    assert resolution.license is None
    assert resolution.license_status is LicenseStatus.ABSENT_CARD_DATA_NULL
    assert resolution.redistribution is Redistribution.PROHIBITED_PENDING_RESOLUTION
    assert resolution.reason == licenses.CARD_DATA_NULL
    assert any("declares no license" in note for note in resolution.notes)
    assert resolution.evidence.spdx is None

    updated = licenses.update_entry(entry, resolution)
    assert updated.license is None
    assert updated.redistribution is Redistribution.PROHIBITED_PENDING_RESOLUTION
    assert updated.errors() == ()


def test_a_card_license_of_other_names_nothing_and_is_not_recorded_as_a_license():
    entry = pinned("minivla-vq-bridge")
    probe = licenses.metadata_probe(entry)
    fetcher = RecordingFetcher({probe.url: (200, hub_metadata(card_license="other", siblings=("config.json",)))})

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.license is None
    assert resolution.reason == licenses.CARD_DECLARES_OTHER
    # Card data is present, so this is unresolved rather than absent.
    assert resolution.license_status is LicenseStatus.UNRESOLVED
    assert resolution.redistribution is Redistribution.PROHIBITED_PENDING_RESOLUTION


def test_the_license_other_plus_license_name_convention_is_read_not_ignored():
    """Cosmos3-Nano really uses ``license: other`` with a name and an off-repo link."""

    entry = pinned("cosmos3-nano")
    probe = licenses.metadata_probe(entry)
    fetcher = RecordingFetcher(
        {
            probe.url: (
                200,
                hub_metadata(
                    card_license="other",
                    license_name="openmdw1.1-license",
                    license_link="https://openmdw.ai/license/1-1/",
                    siblings=("README.md",),
                ),
            )
        }
    )

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.reason == licenses.CARD_DECLARES_OTHER_WITH_NAME
    # Recorded verbatim, because that is what the card says.
    assert resolution.license == "openmdw1.1-license"
    assert resolution.license_status is LicenseStatus.ADVERTISED_UNVERIFIED
    assert resolution.metadata.card_license_link == "https://openmdw.ai/license/1-1/"
    assert any("off-repo" in note for note in resolution.notes)
    assert any("not followed" in note for note in resolution.notes)


def test_an_in_repo_license_link_is_retrieved_and_hashed():
    entry = pinned("qwen2.5-vl-7b-instruct")
    probe = licenses.metadata_probe(entry)
    link_url = licenses.license_file_probes(entry, ("LICENSE.md",))[0].url
    fetcher = RecordingFetcher(
        {
            probe.url: (
                200,
                hub_metadata(card_license="other", license_name="qwen", license_link="LICENSE.md", siblings=()),
            ),
            link_url: (200, APACHE_TEXT),
        }
    )

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.license == "Apache-2.0"
    assert resolution.license_status is LicenseStatus.VERIFIED
    assert resolution.evidence.license_file_path == "LICENSE.md"
    assert link_url in fetcher.calls
    # The card's own name for it is superseded by the text that was hashed.
    assert any("card advertises" in note for note in resolution.notes)


def test_a_card_identifier_is_normalised_in_case_only():
    assert licenses.canonical_card_identifier("mit") == "MIT"
    assert licenses.canonical_card_identifier("apache-2.0") == "Apache-2.0"
    assert licenses.canonical_card_identifier("apache_2_0") == "Apache-2.0"
    # Nothing known: returned verbatim rather than reshaped.
    assert licenses.canonical_card_identifier("openmdw1.1-license") == "openmdw1.1-license"
    assert licenses.canonical_card_identifier("bespoke-terms") == "bespoke-terms"


def test_an_off_repo_license_link_is_never_followed():
    assert licenses.is_repo_relative("LICENSE") is True
    assert licenses.is_repo_relative("legal/LICENSE.md") is True
    assert licenses.is_repo_relative("https://openmdw.ai/license/1-1/") is False
    assert licenses.is_repo_relative("//example.invalid/LICENSE") is False
    assert licenses.is_repo_relative("../LICENSE") is False
    assert licenses.is_repo_relative("") is False


def test_a_listed_repository_without_a_license_file_is_not_probed_ten_times():
    entry = pinned("openvla-7b")
    probe = licenses.metadata_probe(entry)
    fetcher = RecordingFetcher(
        {probe.url: (200, hub_metadata(card_license="mit", siblings=("config.json", "README.md")))}
    )

    resolution = licenses.resolve_entry(entry, fetcher)

    # The host listed its files and none is a license, so guessing is pointless.
    assert fetcher.calls == [probe.url]
    assert resolution.reason == licenses.CARD_ONLY
    assert resolution.license == "MIT"


def test_a_gated_repository_records_that_a_human_must_accept_its_terms():
    entry = pinned("paligemma-official")
    fetcher = RecordingFetcher(default_status=401, default_error="Unauthorized")

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.license is None
    assert resolution.license_status is LicenseStatus.GATED_TERMS_REQUIRED
    assert resolution.local_use is LocalUse.REQUIRES_ACCEPTED_TERMS
    assert resolution.reason == licenses.GATED_METADATA
    assert any("a human must" in note for note in resolution.notes)
    assert GATED_TERMS_BLOCKER in licenses.update_entry(entry, resolution).local_use_blockers()


def test_a_row_without_a_pinned_revision_is_refused_rather_than_read_at_a_branch():
    entry = get_plan("octo-small")
    assert entry.revision is None
    fetcher = RecordingFetcher()

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.reason == licenses.NO_PINNED_REVISION
    assert resolution.license_status is entry.license_status
    assert fetcher.calls == []
    assert any("--resolve-revision" in note for note in resolution.notes)
    assert licenses.probes_for(entry) == ()
    assert UNRESOLVED_REVISION_BLOCKER in resolution_blockers(entry, resolution)


def resolution_blockers(entry, resolution):
    return licenses.update_entry(entry, resolution).local_use_blockers()


def test_an_external_archive_has_no_license_endpoint_and_says_so():
    entry = replace(get_plan("irasim-bridge-archive"), revision=REVISION)
    fetcher = RecordingFetcher()

    resolution = licenses.resolve_entry(entry, fetcher)

    assert entry.repo_type is RepoType.URL
    assert resolution.reason == licenses.NO_LICENSE_ENDPOINT
    assert fetcher.calls == []
    assert any("notices it ships" in note for note in resolution.notes)


def test_a_host_label_never_overrides_the_text_that_was_actually_hashed():
    entry = pinned("code-octo")
    probe = licenses.metadata_probe(entry)
    # The host says MIT; the retrieved file is Apache-2.0.
    fetcher = RecordingFetcher({probe.url: (200, github_license_response(APACHE_TEXT, spdx="MIT"))})

    resolution = licenses.resolve_entry(entry, fetcher)

    assert resolution.license == "Apache-2.0"
    assert resolution.metadata.reported_spdx == "MIT"
    assert any("the record" in note for note in resolution.notes)


def test_notice_files_are_recorded_with_their_hashes_for_an_archival_copy():
    entry = pinned("code-octo")
    probe = licenses.metadata_probe(entry)
    notice = b"IRASim\nCopyright 2024 ByteDance\n"
    notice_url = licenses.license_file_probes(entry, ("NOTICE",))[0].url
    fetcher = RecordingFetcher(
        {
            probe.url: (200, github_license_response(APACHE_TEXT, spdx="Apache-2.0")),
            notice_url: (200, notice),
        }
    )

    resolution = licenses.resolve_entry(entry, fetcher)

    assert [item["path"] for item in resolution.notice_files] == ["NOTICE"]
    assert resolution.notice_files[0]["sha256"] == hashlib.sha256(notice).hexdigest()


# ---------------------------------------------------------------------------
# local use versus redistribution
# ---------------------------------------------------------------------------


def test_local_use_and_redistribution_are_independent_verdicts():
    entry = pinned("minivla-pretrain-vq")
    probe = licenses.metadata_probe(entry)
    fetcher = RecordingFetcher({probe.url: (200, hub_metadata(card_data=False))})

    resolution = licenses.resolve_entry(entry, fetcher)

    # Redistribution stays shut; local use -- what the matrix needs -- is open.
    assert resolution.redistribution is Redistribution.PROHIBITED_PENDING_RESOLUTION
    assert resolution.local_use is LocalUse.PERMITTED_LOCAL_ONLY_NO_EXPRESS_TERMS
    assert resolution.local_use_permitted is True

    updated = licenses.update_entry(entry, resolution)
    # Nothing about redistribution may appear among the local-use blockers.
    assert updated.local_use_blockers() == ()
    assert "redistribution" not in " ".join(updated.local_use_blockers())
    assert updated.redistribution is Redistribution.PROHIBITED_PENDING_RESOLUTION


def test_the_primary_matrix_is_not_blocked_by_an_open_redistribution_term():
    prohibited_but_runnable = []
    for entry in (get_plan(name) for name in plan_names()):
        if not entry.required_for_primary_matrix:
            continue
        # Every primary-matrix row carries an explicit local-use verdict.
        assert LOCAL_USE_BLOCKER not in entry.local_use_blockers(), entry.name
        if (
            entry.redistribution is Redistribution.PROHIBITED_PENDING_RESOLUTION
            and entry.local_use is LocalUse.PERMITTED_LOCAL_ONLY_NO_EXPRESS_TERMS
        ):
            prohibited_but_runnable.append(entry.name)
    # The rows the audit called blocked are runnable: only their shipping right is open.
    assert "minivla-pretrain-vq" in prohibited_but_runnable
    assert "minivla-vq-bridge" in prohibited_but_runnable
    assert "susie-subgoal" in prohibited_but_runnable


def test_local_use_may_only_be_attributed_to_a_license_that_was_verified():
    entry = replace(get_plan("code-susie"), local_use=LocalUse.PERMITTED_BY_LICENSE)
    problems = entry.errors()
    assert any("actually verified" in problem for problem in problems)


def test_a_gated_row_can_never_report_permitted_local_use():
    entry = replace(get_plan("paligemma-official"), local_use=LocalUse.PERMITTED_LOCAL_ONLY_NO_EXPRESS_TERMS)
    assert any("until the terms are accepted" in problem for problem in entry.errors())


# ---------------------------------------------------------------------------
# evidence discipline
# ---------------------------------------------------------------------------


def test_a_status_of_verified_without_evidence_is_refused():
    entry = replace(get_plan("code-susie"), license_status=LicenseStatus.VERIFIED)
    problems = entry.errors()
    assert any("no license evidence is cited" in problem for problem in problems)
    assert any("resolve_licenses.py" in problem for problem in problems)


def test_verified_evidence_must_carry_a_hash_a_revision_and_a_timestamp():
    partial = LicenseEvidence(evidence_path=licenses.LICENSES_EVIDENCE_PATH, spdx=None, license_file_path="LICENSE")
    entry = replace(get_plan("code-susie"), license_status=LicenseStatus.VERIFIED, license_evidence=partial)
    problems = entry.errors()
    assert any("retrieved license file path and its SHA-256" in problem for problem in problems)
    assert any("when it was retrieved" in problem for problem in problems)
    assert any("revision it was read at" in problem for problem in problems)


def test_evidence_may_not_name_a_license_it_did_not_retrieve():
    fabricated = LicenseEvidence(evidence_path=licenses.LICENSES_EVIDENCE_PATH, spdx="MIT")
    assert any("retrieved and hashed" in problem for problem in fabricated.errors())


def test_an_identified_license_may_not_be_filed_under_an_unverified_status():
    evidence = LicenseEvidence(
        evidence_path=licenses.LICENSES_EVIDENCE_PATH,
        spdx="MIT",
        license_file_path="LICENSE",
        license_file_sha256=hashlib.sha256(MIT_TEXT).hexdigest(),
        revision=REVISION,
        retrieved_at="2026-09-19T00:00:00+00:00",
    )
    entry = replace(get_plan("code-susie"), revision=REVISION, license_evidence=evidence)
    assert any("must be verified rather than" in problem for problem in entry.errors())


def test_a_verified_license_must_be_the_identifier_that_was_read():
    evidence = LicenseEvidence(
        evidence_path=licenses.LICENSES_EVIDENCE_PATH,
        spdx="Apache-2.0",
        license_file_path="LICENSE",
        license_file_sha256=hashlib.sha256(APACHE_TEXT).hexdigest(),
        revision=REVISION,
        retrieved_at="2026-09-19T00:00:00+00:00",
    )
    entry = replace(
        get_plan("code-susie"),
        revision=REVISION,
        license="MIT",
        license_status=LicenseStatus.VERIFIED,
        license_evidence=evidence,
    )
    assert any("not 'MIT'" in problem for problem in entry.errors())


def test_evidence_read_at_another_revision_is_refused():
    evidence = LicenseEvidence(
        evidence_path=licenses.LICENSES_EVIDENCE_PATH,
        spdx="MIT",
        license_file_path="LICENSE",
        license_file_sha256=hashlib.sha256(MIT_TEXT).hexdigest(),
        revision="c" * 40,
        retrieved_at="2026-09-19T00:00:00+00:00",
    )
    entry = replace(
        get_plan("code-susie"),
        revision=REVISION,
        license="MIT",
        license_status=LicenseStatus.VERIFIED,
        license_evidence=evidence,
    )
    assert any("different revision" in problem for problem in entry.errors())


def test_every_verified_row_cites_an_evidence_file_that_exists(tmp_path):
    report_path = tmp_path / "provenance" / "licenses.json"
    relative = report_path.relative_to(tmp_path).as_posix()
    entry = pinned("code-susie")
    probe = licenses.metadata_probe(entry)
    fetcher = RecordingFetcher({probe.url: (200, github_license_response(MIT_TEXT, spdx="MIT"))})

    resolution = licenses.resolve_entry(entry, fetcher, evidence_path=relative)
    updated = licenses.apply_resolutions([resolution], entries={entry.name: entry})
    assert updated[0].license_status is LicenseStatus.VERIFIED

    # Before the report is saved the citation is a promise, not evidence.
    assert licenses.evidence_gaps(updated, root=tmp_path) == (
        "code-susie: cites %s, which does not exist" % relative,
    )

    digest = licenses.save_report(
        licenses.build_report([resolution], executed=True, evidence_path=relative), report_path
    )
    assert len(digest) == 64
    assert licenses.evidence_gaps(updated, root=tmp_path) == ()

    saved = json.loads(report_path.read_text())
    assert saved["kind"] == licenses.REPORT_KIND
    assert saved["summary"]["verified_assets"] == ["code-susie"]
    assert saved["resolutions"][0]["plan_update"]["license"] == "MIT"


def test_a_row_verified_without_a_cited_path_is_reported_as_a_gap():
    entry = replace(
        get_plan("code-susie"),
        license="MIT",
        license_status=LicenseStatus.VERIFIED,
        license_evidence=None,
    )
    assert licenses.evidence_gaps([entry]) == ("code-susie: verified without a cited evidence path",)


def test_every_verified_row_in_the_shipped_plan_is_backed_by_saved_evidence():
    assert validate_plan() == ()
    entries = [get_plan(name) for name in plan_names()]
    verified = [entry for entry in entries if entry.license_status is LicenseStatus.VERIFIED]
    assert verified, "the plan should carry the rows cluster/resolve_licenses.py resolved"

    for entry in verified:
        evidence = entry.license_evidence
        assert evidence is not None, entry.name
        assert evidence.spdx == entry.license, entry.name
        assert evidence.license_file_path, entry.name
        assert len(evidence.license_file_sha256) == 64, entry.name
        assert evidence.revision == entry.revision, entry.name
        assert evidence.retrieved_at, entry.name
        cited = licenses.REPO_ROOT / evidence.evidence_path
        assert cited.is_file(), "%s cites %s" % (entry.name, evidence.evidence_path)

    # The cited report agrees with the row: the plan never gets ahead of the file.
    saved = json.loads((licenses.REPO_ROOT / licenses.LICENSES_EVIDENCE_PATH).read_text())
    resolutions = {item["name"]: item for item in saved["resolutions"]}
    for entry in verified:
        recorded = resolutions[entry.name]
        assert recorded["license_status"] == "verified", entry.name
        assert recorded["license"] == entry.license, entry.name
        assert recorded["evidence"]["license_file_sha256"] == entry.license_evidence.license_file_sha256, entry.name
    assert licenses.evidence_gaps(entries) == ()


def test_every_unverified_row_in_the_shipped_plan_still_records_the_blocker():
    for entry in asset_plan_entries():
        if entry["license_status"] == "verified":
            assert MISSING_LICENSE_EVIDENCE_BLOCKER not in entry["verification_blockers"], entry["name"]
            continue
        assert MISSING_LICENSE_EVIDENCE_BLOCKER in entry["verification_blockers"], entry["name"]
        # Only the external archive, whose terms ship inside it, is left unresolved.
        assert entry["local_use"] != "unresolved" or entry["name"] == "irasim-bridge-archive", entry["name"]


def test_the_shipped_plan_records_the_proven_paligemma_comparison():
    mirror_row = get_plan("paligemma-mirror-candidate")
    evidence = mirror_row.mirror_evidence
    assert evidence is not None
    assert evidence.byte_identity is ByteIdentity.PROVEN
    assert (evidence.files_compared, evidence.files_matched) == (7, 7)
    cited = licenses.REPO_ROOT / evidence.evidence_path
    assert cited.is_file()

    saved = json.loads(cited.read_text())
    assert saved["byte_identity"] == "proven"
    assert saved["official_revision"] == evidence.official_revision
    assert saved["mirror_revision"] == evidence.mirror_revision
    assert [item["path"] for item in saved["files"]] == list(mirror.PALIGEMMA_SUPPORT_FILES)
    assert all(item["identity_channel"] for item in saved["files"])
    # Neither PaliGemma row claims a reviewed revision pin on the strength of a lookup.
    assert mirror_row.revision is None
    assert get_plan("paligemma-official").revision is None
    assert UNRESOLVED_REVISION_BLOCKER in mirror_row.local_use_blockers()
    # The download is no longer the blocker; accepted terms are.
    assert UNPROVEN_BYTE_IDENTITY_BLOCKER not in mirror_row.local_use_blockers()
    assert GATED_TERMS_BLOCKER in mirror_row.local_use_blockers()


def test_update_entry_refuses_a_resolution_for_a_different_row():
    resolution = licenses.resolve_entry(pinned("code-susie"), None)
    with pytest.raises(licenses.LicenseResolutionError, match="cannot be applied"):
        licenses.update_entry(get_plan("code-octo"), resolution)


def test_update_entry_refuses_a_resolution_that_would_contradict_the_row():
    entry = get_plan("openvla-7b")
    resolution = replace(
        licenses.resolve_entry(entry, None),
        license=None,
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.PERMITTED_WITH_NOTICES,
    )
    with pytest.raises(licenses.LicenseResolutionError, match="inconsistent plan row"):
        licenses.update_entry(entry, resolution)


# ---------------------------------------------------------------------------
# no network without --execute
# ---------------------------------------------------------------------------


def test_a_dry_run_fetches_nothing_and_writes_nothing(tmp_path, capsys):
    report = tmp_path / "licenses.json"
    code = licenses.main(["--asset", "code-openvla", "--report", str(report)])
    assert code == 0
    assert not report.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["assets"][0]["would_fetch"]
    assert "no network call was made" in payload["notice"]


def test_resolve_entry_without_a_fetcher_performs_no_retrieval():
    resolution = licenses.resolve_entry(pinned("code-susie"), None)
    assert resolution.reason == "not_retrieved_dry_run"
    assert resolution.verified is False
    assert resolution.attempted  # the URLs it *would* have fetched
    assert resolution.evidence.license_file_sha256 is None


def test_an_oversize_response_is_refused_rather_than_truncated(monkeypatch):
    """A truncated body would hash to bytes nobody published: a fabricated mismatch."""

    import urllib.request

    class FakeResponse:
        status = 200

        def __init__(self, payload):
            self._payload = payload

        def read(self, size=None):
            return self._payload if size is None else self._payload[:size]

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            return False

    payload = b"x" * 5000
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(payload))

    generous = licenses.urlopen_fetcher(max_bytes=10_000)("https://example.invalid/file")
    assert generous.ok is True
    assert generous.body == payload

    stingy = licenses.urlopen_fetcher(max_bytes=100)("https://example.invalid/file")
    assert stingy.ok is False
    assert stingy.body is None
    assert "refused rather than truncated" in stingy.error


def test_the_mirror_tool_reads_whole_support_files_not_license_sized_ones():
    # tokenizer.json is ~17 MB, far above the license-file ceiling.
    assert mirror.MAX_MIRROR_FILE_BYTES > licenses.MAX_LICENSE_FILE_BYTES
    assert mirror.MAX_MIRROR_FILE_BYTES >= 17_549_604


def test_neither_tool_opens_a_socket_at_import_time():
    # The real fetcher is a factory, so importing the module cannot fetch.
    assert callable(licenses.urlopen_fetcher)
    source = (licenses.REPO_ROOT / "cluster" / "resolve_licenses.py").read_text()
    top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not any("urllib" in line or "requests" in line or "http" in line for line in top_level)
    mirror_source = (licenses.REPO_ROOT / "cluster" / "compare_mirror.py").read_text()
    mirror_top = [line for line in mirror_source.splitlines() if line.startswith(("import ", "from "))]
    assert not any("urllib" in line or "requests" in line for line in mirror_top)


# ---------------------------------------------------------------------------
# mirror comparison
# ---------------------------------------------------------------------------


TARGET = mirror.get_target("paligemma")
MIRROR_BODIES = {name: ('{"file": "%s"}' % name).encode("utf-8") for name in TARGET.files}


def official_siblings(*, lfs=(), blob=(), sizes=True, omit=()):
    rows = []
    for name, body in MIRROR_BODIES.items():
        if name in omit:
            continue
        row = {"rfilename": name}
        if sizes:
            row["size"] = len(body)
        if name in lfs:
            row["lfs"] = {"sha256": hashlib.sha256(body).hexdigest()}
        if name in blob:
            row["blobId"] = mirror.git_blob_sha1(body)
        rows.append(row)
    return {"siblings": rows, "gated": True}


class MirrorFetcher:
    """Answers official metadata from a payload and mirror files from bytes."""

    def __init__(self, official_payload, mirror_status=200, mirror_bodies=None):
        self.official_payload = official_payload
        self.mirror_status = mirror_status
        self.mirror_bodies = MIRROR_BODIES if mirror_bodies is None else mirror_bodies
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if "/api/" in url:
            if self.official_payload is None:
                return licenses.FetchResponse(url=url, status=401, error="Unauthorized")
            return licenses.FetchResponse(
                url=url, status=200, body=json.dumps(self.official_payload).encode("utf-8")
            )
        name = url.rsplit("/", 1)[-1]
        if self.mirror_status != 200 or name not in self.mirror_bodies:
            return licenses.FetchResponse(url=url, status=self.mirror_status or 404, error="unavailable")
        return licenses.FetchResponse(url=url, status=200, body=self.mirror_bodies[name])


def run_comparison(official_payload, **kwargs):
    fetcher = MirrorFetcher(official_payload, **kwargs)
    report = mirror.compare(
        TARGET,
        fetcher,
        official_repo_id="google/paligemma-3b-pt-224",
        official_revision=REVISION,
        mirror_repo_id="leo009/paligemma-3b-pt-224",
        mirror_revision=MIRROR_REVISION,
    )
    return report, fetcher


def test_the_default_file_list_is_the_seven_support_files_from_the_spec():
    assert TARGET.files == (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "config.json",
    )
    assert mirror.PALIGEMMA_SUPPORT_FILES == TARGET.files


def test_byte_identity_cannot_be_proven_when_an_official_hash_was_unobtainable():
    report, _ = run_comparison(None)

    assert report.byte_identity is ByteIdentity.UNPROVEN
    assert report.files_matched == 0
    assert any("not reachable without accepting terms" in reason for reason in report.unproven_reasons())
    assert all(item.verdict == mirror.OFFICIAL_HASH_UNOBTAINABLE for item in report.files)
    payload = report.payload()
    assert payload["byte_identity"] == "unproven"
    assert mirror.UNPROVEN_NOTE in payload["discipline"]


def test_a_size_match_alone_never_proves_identity():
    report, _ = run_comparison(official_siblings())

    assert all(item.size_matched is True for item in report.files)
    assert report.files_matched == 0
    assert report.byte_identity is ByteIdentity.UNPROVEN
    assert all(item.verdict == mirror.OFFICIAL_HASH_UNOBTAINABLE for item in report.files)
    assert any(mirror.SIZE_IS_NOT_IDENTITY_NOTE in (item.reason or "") for item in report.files)


def test_one_file_without_an_official_hash_keeps_the_whole_report_unproven():
    rest = tuple(name for name in MIRROR_BODIES if name != "config.json")
    report, _ = run_comparison(official_siblings(blob=rest))

    assert report.files_matched == len(rest)
    assert report.byte_identity is ByteIdentity.UNPROVEN
    assert any("config.json" in reason for reason in report.unproven_reasons())


def test_identity_is_proven_only_against_a_retrieved_official_hash():
    report, _ = run_comparison(official_siblings(blob=tuple(MIRROR_BODIES)))

    assert report.byte_identity is ByteIdentity.PROVEN
    assert report.files_matched == len(TARGET.files)
    assert {item.identity_channel for item in report.files} == {mirror.CHANNEL_GIT_BLOB_SHA1}
    assert report.unproven_reasons() == ()

    lfs = ("tokenizer.model",)
    blob = tuple(name for name in MIRROR_BODIES if name not in lfs)
    mixed, _ = run_comparison(official_siblings(lfs=lfs, blob=blob))
    assert mixed.byte_identity is ByteIdentity.PROVEN
    assert {item.identity_channel for item in mixed.files} == {
        mirror.CHANNEL_LFS_SHA256,
        mirror.CHANNEL_GIT_BLOB_SHA1,
    }


def test_a_mismatching_hash_is_reported_as_differing_not_as_unobtainable():
    payload = official_siblings(blob=tuple(MIRROR_BODIES))
    payload["siblings"][0]["blobId"] = "0" * 40
    report, _ = run_comparison(payload)

    assert report.files[0].verdict == mirror.DIFFERS
    assert report.byte_identity is ByteIdentity.UNPROVEN
    assert any("does not match" in reason for reason in report.unproven_reasons())


def test_a_hash_that_matches_while_the_size_disagrees_is_not_identical():
    payload = official_siblings(blob=tuple(MIRROR_BODIES))
    payload["siblings"][0]["size"] = 1
    report, _ = run_comparison(payload)

    assert report.files[0].size_matched is False
    assert report.files[0].verdict == mirror.DIFFERS
    assert report.byte_identity is ByteIdentity.UNPROVEN


def test_an_unreachable_mirror_file_is_not_silently_skipped():
    report, _ = run_comparison(official_siblings(blob=tuple(MIRROR_BODIES)), mirror_status=404)

    assert all(item.verdict == mirror.MIRROR_UNREACHABLE for item in report.files)
    assert report.byte_identity is ByteIdentity.UNPROVEN
    assert len(report.files) == len(TARGET.files)


def test_a_file_the_official_repository_does_not_list_has_nothing_to_compare():
    report, _ = run_comparison(official_siblings(blob=tuple(MIRROR_BODIES), omit=("config.json",)))

    absent = next(item for item in report.files if item.path == "config.json")
    assert absent.verdict == mirror.OFFICIAL_FILE_ABSENT
    assert absent.mirror_sha256  # the mirror's bytes were still hashed and recorded
    assert report.byte_identity is ByteIdentity.UNPROVEN


def test_a_comparison_at_a_mutable_branch_is_refused():
    with pytest.raises(mirror.MirrorComparisonError, match="40-character commits"):
        mirror.compare(
            TARGET,
            MirrorFetcher(official_siblings()),
            official_repo_id="google/paligemma-3b-pt-224",
            official_revision="main",
            mirror_repo_id="leo009/paligemma-3b-pt-224",
            mirror_revision=MIRROR_REVISION,
        )


def test_the_gemma_license_note_survives_onto_the_mirror_row():
    entry = get_plan("paligemma-mirror-candidate")
    assert entry.mirror_of == "paligemma-official"
    assert any("Gemma Terms of Use" in note for note in entry.notes)
    assert any("Gemma Terms of Use" in note for note in get_plan("paligemma-official").notes)

    proven, _ = run_comparison(official_siblings(blob=tuple(MIRROR_BODIES)))
    path = mirror.mirror_evidence_path("paligemma")
    updated = mirror.update_mirror_entry(entry, proven, path)

    assert updated.mirror_evidence.byte_identity is ByteIdentity.PROVEN
    assert updated.mirror_evidence.governing_license == "Gemma Terms of Use"
    assert updated.mirror_evidence.evidence_path == path
    assert any("Gemma Terms of Use" in note for note in updated.notes)
    # Proven bytes remove the gated download, never the licence.
    assert updated.local_use is LocalUse.REQUIRES_ACCEPTED_TERMS
    assert GATED_TERMS_BLOCKER in updated.local_use_blockers()
    assert UNPROVEN_BYTE_IDENTITY_BLOCKER not in updated.local_use_blockers()
    assert updated.errors() == ()

    report_payload = proven.payload()
    assert report_payload["license"]["governing_license"] == "Gemma Terms of Use"
    assert report_payload["license"]["terms_accepted"] is False


def test_an_unproven_comparison_keeps_the_mirror_blocked():
    entry = get_plan("paligemma-mirror-candidate")
    unproven, _ = run_comparison(None)
    updated = mirror.update_mirror_entry(entry, unproven, mirror.mirror_evidence_path("paligemma"))

    assert updated.mirror_evidence.byte_identity is ByteIdentity.UNPROVEN
    assert updated.mirror_evidence.unproven_reasons
    assert UNPROVEN_BYTE_IDENTITY_BLOCKER in updated.local_use_blockers()
    assert UNPROVEN_BYTE_IDENTITY_BLOCKER in updated.verification_blockers()
    assert any(mirror.UNPROVEN_NOTE == note for note in updated.notes)


def test_mirror_evidence_refuses_a_proven_claim_it_cannot_support():
    assert any(
        "cannot be proven from zero compared files" in problem
        for problem in MirrorEvidence(
            evidence_path="results/provenance/mirror-comparison-paligemma.json",
            official_repo_id="google/paligemma-3b-pt-224",
            byte_identity=ByteIdentity.PROVEN,
        ).errors()
    )
    assert any(
        "did not match a retrieved official hash" in problem
        for problem in MirrorEvidence(
            evidence_path="results/provenance/mirror-comparison-paligemma.json",
            official_repo_id="google/paligemma-3b-pt-224",
            byte_identity=ByteIdentity.PROVEN,
            files_compared=7,
            files_matched=6,
        ).errors()
    )
    assert any(
        "must record why it is unproven" in problem
        for problem in MirrorEvidence(
            evidence_path="results/provenance/mirror-comparison-paligemma.json",
            official_repo_id="google/paligemma-3b-pt-224",
            byte_identity=ByteIdentity.UNPROVEN,
            files_compared=7,
        ).errors()
    )


def test_mirror_evidence_cannot_be_attached_to_a_row_that_is_not_a_mirror():
    report, _ = run_comparison(official_siblings(blob=tuple(MIRROR_BODIES)))
    with pytest.raises(mirror.MirrorComparisonError, match="not marked as a mirror"):
        mirror.update_mirror_entry(get_plan("paligemma-official"), report, "x.json")


def test_a_mirror_comparison_dry_run_fetches_nothing_and_writes_nothing(tmp_path, capsys):
    report = tmp_path / "mirror.json"
    code = mirror.main(["--asset", "paligemma", "--report", str(report)])
    assert code == 0
    assert not report.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["revisions_pinned"] is False
    assert len(payload["would_fetch"]) == len(TARGET.files) + 1
    assert "Gemma" in payload["governing_license_note"]


def test_a_saved_comparison_records_every_file_and_its_verdict(tmp_path):
    report, _ = run_comparison(official_siblings(blob=tuple(MIRROR_BODIES)))
    destination = tmp_path / "provenance" / "mirror-comparison-paligemma.json"
    digest = mirror.save_report(report, destination)

    assert len(digest) == 64
    saved = json.loads(destination.read_text())
    assert saved["kind"] == mirror.REPORT_KIND
    assert saved["byte_identity"] == "proven"
    assert [item["path"] for item in saved["files"]] == list(TARGET.files)
    for item in saved["files"]:
        assert item["mirror_sha256"]
        assert item["verdict"] == mirror.IDENTICAL
        assert item["identity_channel"]
    assert saved["plan_update"]["mirror_evidence"]["byte_identity"] == "proven"


def test_a_fabricated_plan_entry_cannot_smuggle_a_verified_status():
    fabricated = AssetPlanEntry(
        name="fabricated",
        repo_id="someone/thing",
        repo_type=RepoType.MODEL,
        purpose="test",
        source_url="https://example.invalid",
        license="MIT",
        license_status=LicenseStatus.VERIFIED,
        redistribution=Redistribution.PERMITTED_WITH_NOTICES,
        access=Access.PUBLIC,
        revision=REVISION,
        local_use=LocalUse.PERMITTED_BY_LICENSE,
    )
    assert any("no license evidence is cited" in problem for problem in fabricated.errors())
