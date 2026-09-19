"""Declarative, pinned plan for every asset PLUMB needs, with its traps encoded.

This module is data plus validation, not a downloader.  It exists so that:

* every asset in the build spec has exactly one named, reviewable entry rather
  than a hand-typed ``download_assets.py`` invocation;
* the discovery traps (Octo's non-root checkpoint path, Octo v1.0 versus 1.5,
  MiniVLA's ``=`` in a filename, ``pretrain_vq``'s absent license, PaliGemma's
  official-versus-mirror provenance, and ``auto_eval``'s pickles) live next to
  the asset they apply to instead of in prose someone has to remember;
* ``asset_plan_entries()`` can emit candidate ``assets.lock.json`` records whose
  unknown values stay ``null`` and explicitly block a ``verified`` status.

Sizes here are discovery aids.  ``expected_bytes`` is used only where the spec
states an exact byte count; everything else is
``historical_estimate_bytes`` and is never treated as verification.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import quote


SCHEMA_VERSION = 1
LOCK_ENTRY_KIND = "plumb_asset_lock_entry"

OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
COSMOS3_NANO_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
IRASIM_CODE_REVISION = "c72b6dade6fcd65971e0aa8ab49ea39b15108c90"
COSMOS_FRAMEWORK_REVISION = "c23e51f2f157ae3e51cfcd86ebfb5464850894f2"
AUTOEVAL_CODE_REVISION = "3ea3ff44c6950433cfbcb4294a3deaa616533745"
OPEN_PI_ZERO_CODE_REVISION = "c3df7fb062175c16f69d7ca4ce042958ea238fb7"
OPENVLA_CODE_REVISION = "c8f03f48af692657d3060c19588038c7220e9af9"

UNRESOLVED_REVISION_BLOCKER = "unresolved_immutable_revision"
UNRESOLVED_LICENSE_BLOCKER = "unresolved_license_or_access_terms"
MISSING_HASH_BLOCKER = "no_downloaded_file_hashes"
GATED_TERMS_BLOCKER = "gated_access_terms_not_recorded_as_accepted"


class RepoType(str, Enum):
    MODEL = "model"
    DATASET = "dataset"
    URL = "url"
    GIT = "git"


class LicenseStatus(str, Enum):
    """How well the license/access terms are actually established."""

    DECLARED = "declared"
    ADVERTISED_UNVERIFIED = "advertised_unverified"
    GATED_TERMS_REQUIRED = "gated_terms_required"
    ABSENT_CARD_DATA_NULL = "absent_cardData_null"
    UNRESOLVED = "unresolved"


class Redistribution(str, Enum):
    PERMITTED_WITH_NOTICES = "permitted_with_notices"
    LOCAL_USE_ONLY = "local_use_only"
    PROHIBITED_PENDING_RESOLUTION = "prohibited_pending_resolution"
    UNRESOLVED = "unresolved"


class Access(str, Enum):
    PUBLIC = "public"
    GATED_ACCEPT_TERMS = "gated_accept_terms"
    EXTERNAL_URL = "external_url"
    SOURCE_REPOSITORY = "source_repository"


@dataclass(frozen=True)
class PlannedFile:
    """One path inside a repository, with only the sizes we actually know."""

    path: str
    expected_bytes: Optional[int] = None
    historical_estimate_bytes: Optional[int] = None
    sha256: Optional[str] = None
    is_directory: bool = False
    notes: Optional[str] = None

    @property
    def url_path(self) -> str:
        """Percent-encoded path for a direct ``resolve/<rev>/`` URL.

        MiniVLA's checkpoint filename contains ``=``, which must be sent as
        ``%3D``.  Hub API clients take the literal path instead, so both forms
        are available and never confused for one another.
        """

        return quote(self.path, safe="/")

    def allow_pattern(self) -> str:
        return self.path + "/*" if self.is_directory else self.path

    def payload(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "url_path": self.url_path,
            "is_directory": bool(self.is_directory),
            "expected_bytes": self.expected_bytes,
            "historical_estimate_bytes": self.historical_estimate_bytes,
            "sha256": self.sha256,
            "verified": False,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AssetPlanEntry:
    """One pinned asset: what to fetch, under what terms, and what to watch for."""

    name: str
    repo_id: str
    repo_type: RepoType
    purpose: str
    source_url: str
    license: Optional[str]
    license_status: LicenseStatus
    redistribution: Redistribution
    access: Access
    revision: Optional[str] = None
    files: Tuple[PlannedFile, ...] = ()
    extra_allow_patterns: Tuple[str, ...] = ()
    ignore_patterns: Tuple[str, ...] = ()
    max_gb: float = 40.0
    historical_total_estimate_bytes: Optional[int] = None
    expected_file_count: Optional[int] = None
    required_for_primary_matrix: bool = False
    fetch: bool = True
    compatibility_profile_id: Optional[str] = None
    loader_revision: Optional[str] = None
    notes: Tuple[str, ...] = ()
    traps: Tuple[str, ...] = ()
    blockers: Tuple[str, ...] = field(default_factory=tuple)

    def allow_patterns(self) -> Tuple[str, ...]:
        """Literal Hub-API patterns; never the percent-encoded URL form."""

        patterns = [planned.allow_pattern() for planned in self.files]
        patterns.extend(self.extra_allow_patterns)
        return tuple(dict.fromkeys(patterns))

    def direct_download_url(self, planned: PlannedFile) -> str:
        """A ``resolve/<revision>/`` URL using the percent-encoded path."""

        if self.repo_type is RepoType.URL:
            return self.source_url
        if self.repo_type is RepoType.GIT:
            raise ValueError("%s is a source repository; clone it at its pinned commit." % self.name)
        revision = self.revision or "UNRESOLVED_REVISION"
        prefix = "datasets/" if self.repo_type is RepoType.DATASET else ""
        return "https://huggingface.co/%s%s/resolve/%s/%s" % (prefix, self.repo_id, revision, planned.url_path)

    def errors(self) -> Tuple[str, ...]:
        """Internal-consistency problems in the plan itself."""

        problems: List[str] = []
        if not self.name:
            problems.append("name is required")
        if not self.repo_id:
            problems.append("repo_id is required")
        if not self.purpose:
            problems.append("purpose is required")
        if not self.source_url:
            problems.append("source_url is required")
        if self.max_gb <= 0:
            problems.append("max_gb must be a positive explicit ceiling")
        if self.license is None:
            if self.license_status in (LicenseStatus.DECLARED, LicenseStatus.ADVERTISED_UNVERIFIED):
                problems.append("license is null but license_status claims a declared license")
            if self.redistribution is not Redistribution.PROHIBITED_PENDING_RESOLUTION:
                problems.append(
                    "license is null, so redistribution must be prohibited_pending_resolution, not %s"
                    % self.redistribution.value
                )
        else:
            if self.license_status is LicenseStatus.ABSENT_CARD_DATA_NULL:
                problems.append("license_status says the card data is null but a license string was recorded")
        if self.license_status is LicenseStatus.UNRESOLVED and UNRESOLVED_LICENSE_BLOCKER not in self.blockers:
            problems.append("an unresolved license must record the %r blocker" % UNRESOLVED_LICENSE_BLOCKER)
        if self.access is Access.GATED_ACCEPT_TERMS and GATED_TERMS_BLOCKER not in self.blockers:
            problems.append("gated access must record the %r blocker until terms are accepted" % GATED_TERMS_BLOCKER)
        if self.revision is None:
            if UNRESOLVED_REVISION_BLOCKER not in self.blockers:
                problems.append("a missing revision must record the %r blocker" % UNRESOLVED_REVISION_BLOCKER)
        elif not _is_commit(self.revision) and self.repo_type is not RepoType.URL:
            problems.append("revision must be a full 40-character hexadecimal commit")
        seen = set()
        for planned in self.files:
            if planned.path in seen:
                problems.append("duplicate planned file %r" % planned.path)
            seen.add(planned.path)
            if planned.expected_bytes is not None and planned.expected_bytes <= 0:
                problems.append("expected_bytes for %r must be positive when known" % planned.path)
        if self.repo_type in (RepoType.URL, RepoType.GIT) and self.files:
            problems.append("%s entries describe a whole artifact, not Hub file paths" % self.repo_type.value)
        return tuple(problems)

    def verification_blockers(self) -> Tuple[str, ...]:
        """Everything that must stop this entry from being marked ``verified``."""

        blockers: List[str] = list(self.blockers)
        if self.revision is None and UNRESOLVED_REVISION_BLOCKER not in blockers:
            blockers.append(UNRESOLVED_REVISION_BLOCKER)
        if any(planned.sha256 is None for planned in self.files) or not self.files:
            if MISSING_HASH_BLOCKER not in blockers:
                blockers.append(MISSING_HASH_BLOCKER)
        if self.license_status is LicenseStatus.UNRESOLVED and UNRESOLVED_LICENSE_BLOCKER not in blockers:
            blockers.append(UNRESOLVED_LICENSE_BLOCKER)
        return tuple(blockers)

    def lock_entry(self) -> Dict[str, Any]:
        """A candidate ``assets.lock.json`` record with unknowns left null."""

        return {
            "schema_version": SCHEMA_VERSION,
            "kind": LOCK_ENTRY_KIND,
            "name": self.name,
            "repo_id": self.repo_id,
            "repo_type": self.repo_type.value,
            "purpose": self.purpose,
            "source_url": self.source_url,
            "revision": self.revision,
            "revision_immutable": bool(self.revision is not None and _is_commit(self.revision)),
            "license": self.license,
            "license_status": self.license_status.value,
            "redistribution": self.redistribution.value,
            "license_notices_preserved": True,
            "access": self.access.value,
            "files": [planned.payload() for planned in self.files],
            "allow_patterns": list(self.allow_patterns()),
            "ignore_patterns": list(self.ignore_patterns),
            "expected_file_count": self.expected_file_count,
            "historical_total_estimate_bytes": self.historical_total_estimate_bytes,
            "max_gb": float(self.max_gb),
            "fetch": bool(self.fetch),
            "required_for_primary_matrix": bool(self.required_for_primary_matrix),
            "compatibility_profile_id": self.compatibility_profile_id,
            "loader_revision": self.loader_revision,
            "container_digest": None,
            "retrieved_at": None,
            "retrieval_host": None,
            "downloaded_bytes": None,
            "local_path": None,
            "notes": list(self.notes),
            "traps": list(self.traps),
            "status": "planned",
            "verified": False,
            "verification_blockers": list(self.verification_blockers()),
        }


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


_MIRRORING_NOTE = (
    "Automatic mirroring is removed. Private hosting is not a substitute for redistribution rights; "
    "preserve upstream notices for any authorized archival copy."
)
_SIZE_NOTE = "Historical sizes are discovery aids, not proof of a local download or of byte identity."


_ENTRIES: Tuple[AssetPlanEntry, ...] = (
    # ---------------------------------------------------------------- world model
    AssetPlanEntry(
        name="cosmos3-nano",
        repo_id="nvidia/Cosmos3-Nano",
        repo_type=RepoType.MODEL,
        purpose="Primary forward-dynamics world model for generated evaluation.",
        source_url="https://huggingface.co/nvidia/Cosmos3-Nano",
        license="OpenMDW-1.1",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=COSMOS3_NANO_REVISION,
        extra_allow_patterns=("*",),
        max_gb=40.0,
        historical_total_estimate_bytes=33_000_000_000,
        required_for_primary_matrix=True,
        compatibility_profile_id="cosmos3-nano-diffusers-fd",
        notes=(
            "Download size, parameter count, and peak inference memory are different quantities; record measured "
            "memory for the selected runtime.",
            _SIZE_NOTE,
        ),
    ),
    AssetPlanEntry(
        name="cosmos3-edge",
        repo_id="nvidia/Cosmos3-Edge",
        repo_type=RepoType.MODEL,
        purpose="Speed-arm world model for the cost-fidelity sweep.",
        source_url="https://huggingface.co/nvidia/Cosmos3-Edge",
        license="OpenMDW-1.1",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=None,
        extra_allow_patterns=("*",),
        max_gb=14.0,
        historical_total_estimate_bytes=8_500_000_000,
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
        notes=("Its model card lists 4B; Cosmos3-Super's 64B is excluded from the single-H100 design.", _SIZE_NOTE),
    ),
    AssetPlanEntry(
        name="cosmos-guardrail",
        repo_id="nvidia/Cosmos-1.0-Guardrail",
        repo_type=RepoType.MODEL,
        purpose="Optional gated safety checker; NOT fetched by the fixed robotics-fixture profile.",
        source_url="https://huggingface.co/nvidia/Cosmos-1.0-Guardrail",
        license=None,
        license_status=LicenseStatus.GATED_TERMS_REQUIRED,
        redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
        access=Access.GATED_ACCEPT_TERMS,
        revision=None,
        fetch=False,
        max_gb=4.0,
        blockers=(UNRESOLVED_REVISION_BLOCKER, GATED_TERMS_BLOCKER),
        notes=(
            "The fixed robotics-fixture profile uses the documented Diffusers enable_safety_checker=False and "
            "does not fetch this repository; that choice is recorded rather than implied.",
            "Do not transfer safety flags between backends.",
        ),
    ),
    AssetPlanEntry(
        name="irasim-bridge-archive",
        repo_id="lf-robot-opensource.bytetos.com/opensource_IRASim_v1/bridge_checkpoints_data.tar.gz",
        repo_type=RepoType.URL,
        purpose="Official IRASim Bridge checkpoints and data archive for the second world-model path.",
        source_url=(
            "https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/"
            "bridge_checkpoints_data.tar.gz"
        ),
        license=None,
        license_status=LicenseStatus.UNRESOLVED,
        redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
        access=Access.EXTERNAL_URL,
        max_gb=40.0,
        historical_total_estimate_bytes=32_000_000_000,
        loader_revision=IRASIM_CODE_REVISION,
        blockers=(UNRESOLVED_REVISION_BLOCKER, UNRESOLVED_LICENSE_BLOCKER),
        notes=(
            "IRASim's code is Apache-2.0, but archive-specific licenses and notices must be inspected separately.",
            "Hash the archive and the extracted weights, and run IRASim's own fixtures before marking it usable.",
            "Fetch with cluster/fetch_irasim_archive.sh; this is not a Hub snapshot.",
        ),
    ),
    AssetPlanEntry(
        name="sdxl-vae",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        repo_type=RepoType.MODEL,
        purpose="VAE and scheduler components required by the IRASim Bridge pipeline.",
        source_url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        license="CreativeML Open RAIL++-M",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=None,
        extra_allow_patterns=("vae/*", "scheduler/*", "model_index.json"),
        max_gb=2.0,
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
        notes=("Only the vae and scheduler subfolders are needed; the full SDXL UNet is not.",),
    ),
    # ------------------------------------------------------------------- policies
    AssetPlanEntry(
        name="openvla-7b",
        repo_id="openvla/openvla-7b",
        repo_type=RepoType.MODEL,
        purpose="OpenVLA anchor policy (one 7-D action per fresh image).",
        source_url="https://huggingface.co/openvla/openvla-7b",
        license="MIT",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=OPENVLA_REVISION,
        extra_allow_patterns=("*",),
        max_gb=20.0,
        historical_total_estimate_bytes=15_085_000_000,
        expected_file_count=18,
        required_for_primary_matrix=True,
        compatibility_profile_id="openvla-transformers-4.40.1",
        loader_revision=OPENVLA_CODE_REVISION,
        notes=(
            "The checkpoint revision and the reviewed remote-code revision are the same commit here and must "
            "both be pinned before trust_remote_code is enabled.",
            _SIZE_NOTE,
        ),
    ),
    AssetPlanEntry(
        name="octo-small",
        repo_id="rail-berkeley/octo-small",
        repo_type=RepoType.MODEL,
        purpose="Octo-Small v1.0 primary-matrix policy (two-image history, four-action proposal).",
        source_url="https://huggingface.co/rail-berkeley/octo-small",
        license="MIT",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=None,
        files=(
            PlannedFile(
                path="270000/default/checkpoint",
                historical_estimate_bytes=547_000_000,
                is_directory=True,
                notes="Octo checkpoints are NOT at the repository root.",
            ),
        ),
        extra_allow_patterns=("*.json", "*.msgpack"),
        max_gb=2.0,
        historical_total_estimate_bytes=547_000_000,
        required_for_primary_matrix=True,
        compatibility_profile_id="octo-v1.0-jax-0.4.20",
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
        traps=(
            "Octo checkpoints are NOT at the repository root: use 270000/default/checkpoint. "
            "resolve/main/checkpoint returns 404.",
            "Octo must be v1.0. Do NOT use rail-berkeley/octo-small-1.5; it is a different model.",
        ),
        notes=(
            "OctoModel.load_pretrained also needs the repository-root config/statistics/example-batch files; "
            "resolve the exact root file set at the pinned revision.",
            "The published reference table's Octo key maps to Octo-Small.",
        ),
    ),
    AssetPlanEntry(
        name="octo-base",
        repo_id="rail-berkeley/octo-base",
        repo_type=RepoType.MODEL,
        purpose="Octo-Base v1.0 separate diagnostic; not a seventh benchmark policy.",
        source_url="https://huggingface.co/rail-berkeley/octo-base",
        license="MIT",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=None,
        files=(
            PlannedFile(
                path="300000/default/checkpoint",
                historical_estimate_bytes=811_000_000,
                is_directory=True,
                notes="Octo checkpoints are NOT at the repository root.",
            ),
        ),
        extra_allow_patterns=("*.json", "*.msgpack"),
        max_gb=3.0,
        historical_total_estimate_bytes=811_000_000,
        compatibility_profile_id="octo-v1.0-jax-0.4.20",
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
        traps=(
            "Octo checkpoints are NOT at the repository root: use 300000/default/checkpoint.",
            "Octo must be v1.0. Do NOT use a 1.5 repository.",
        ),
    ),
    AssetPlanEntry(
        name="minivla-vq-bridge",
        repo_id="Stanford-ILIAD/minivla-vq-bridge-prismatic",
        repo_type=RepoType.MODEL,
        purpose="MiniVLA primary-matrix policy (seven-action chunk).",
        source_url="https://huggingface.co/Stanford-ILIAD/minivla-vq-bridge-prismatic",
        license=None,
        license_status=LicenseStatus.UNRESOLVED,
        redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
        access=Access.PUBLIC,
        revision=None,
        files=(
            PlannedFile(
                path="checkpoints/step-362500-epoch-21-loss=0.2259.pt",
                historical_estimate_bytes=5_550_000_000,
                notes="The '=' must be sent as %3D in a direct resolve URL; Hub API clients take the literal path.",
            ),
        ),
        ignore_patterns=("*.jsonl",),
        max_gb=8.0,
        historical_total_estimate_bytes=5_550_000_000,
        required_for_primary_matrix=True,
        compatibility_profile_id="minivla-transformers-4.40.1",
        blockers=(UNRESOLVED_REVISION_BLOCKER, UNRESOLVED_LICENSE_BLOCKER),
        traps=(
            "The filename contains '='. URL-encode it as %3D for direct downloads.",
            "Skip the 76 MB .jsonl file; it is a training log, not a weight.",
            "This is a legacy .pt pickle: convert it in an isolated, credential-free, network-free environment "
            "and keep the converted artifact hash. A pickle scan or strict=True alone does not establish safety.",
        ),
        notes=("MiniVLA remains required for the full matrix, with license resolution tracked as a dependency.",),
    ),
    AssetPlanEntry(
        name="minivla-pretrain-vq",
        repo_id="Stanford-ILIAD/pretrain_vq",
        repo_type=RepoType.MODEL,
        purpose="MiniVLA's VQ action tokenizer; required for its seven-action chunk.",
        source_url="https://huggingface.co/Stanford-ILIAD/pretrain_vq",
        license=None,
        license_status=LicenseStatus.ABSENT_CARD_DATA_NULL,
        redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
        access=Access.PUBLIC,
        revision=None,
        files=(
            PlannedFile(
                path=(
                    "pretrain_modvq+mx-bridge_dataset+fach-7+ng-7+nemb-256+nlatent-512/checkpoints/model.pt"
                ),
                historical_estimate_bytes=9_500_000,
                notes="The '+' characters must be percent-encoded in a direct resolve URL.",
            ),
        ),
        max_gb=0.1,
        historical_total_estimate_bytes=9_500_000,
        required_for_primary_matrix=True,
        blockers=(UNRESOLVED_REVISION_BLOCKER, UNRESOLVED_LICENSE_BLOCKER),
        traps=(
            "This repository declares NO license (cardData: null). Flag it to the user; do not redistribute.",
            "Legacy .pt pickle: isolate the conversion and retain the converted artifact hash.",
        ),
        notes=(_MIRRORING_NOTE,),
    ),
    AssetPlanEntry(
        name="open-pi-zero",
        repo_id="allenzren/open-pi-zero",
        repo_type=RepoType.MODEL,
        purpose="OpenPiZero primary-matrix policy (four-action chunk, refreshed proprioception).",
        source_url="https://huggingface.co/allenzren/open-pi-zero",
        license="MIT",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=None,
        files=(
            PlannedFile(
                path="bridge_beta_step19296_2024-12-26_22-30_42.pt",
                historical_estimate_bytes=11_773_000_000,
            ),
        ),
        max_gb=16.0,
        historical_total_estimate_bytes=11_773_000_000,
        required_for_primary_matrix=True,
        compatibility_profile_id="open-pi-zero-uv",
        loader_revision=OPEN_PI_ZERO_CODE_REVISION,
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
        traps=(
            "Verify full state-dict coverage in the pinned loader; strict=True checks keys, not provenance or "
            "deserialization safety.",
            "Legacy .pt pickle: isolate the conversion and retain the converted artifact hash.",
        ),
        notes=("action_normalization_type='bounds' is this adapter's setting, not a universal native-policy API.",),
    ),
    AssetPlanEntry(
        name="paligemma-official",
        repo_id="google/paligemma-3b-pt-224",
        repo_type=RepoType.MODEL,
        purpose="OpenPiZero's required PaliGemma tokenizer/preprocessor support files.",
        source_url="https://huggingface.co/google/paligemma-3b-pt-224",
        license=None,
        license_status=LicenseStatus.GATED_TERMS_REQUIRED,
        redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
        access=Access.GATED_ACCEPT_TERMS,
        revision=None,
        files=(
            PlannedFile(path="tokenizer.json"),
            PlannedFile(path="tokenizer.model"),
            PlannedFile(path="tokenizer_config.json"),
            PlannedFile(path="added_tokens.json"),
            PlannedFile(path="special_tokens_map.json"),
            PlannedFile(path="preprocessor_config.json"),
            PlannedFile(path="config.json"),
        ),
        max_gb=0.5,
        historical_total_estimate_bytes=21_900_000,
        required_for_primary_matrix=True,
        blockers=(UNRESOLVED_REVISION_BLOCKER, GATED_TERMS_BLOCKER),
        traps=("Prefer this official access path under accepted terms rather than any mirror.",),
        notes=("Configure an explicit application cache directory before fetching.",),
    ),
    AssetPlanEntry(
        name="paligemma-mirror-candidate",
        repo_id="leo009/paligemma-3b-pt-224",
        repo_type=RepoType.MODEL,
        purpose="Candidate PaliGemma mirror; usable only after provenance, hashes, and terms are resolved.",
        source_url="https://huggingface.co/leo009/paligemma-3b-pt-224",
        license=None,
        license_status=LicenseStatus.UNRESOLVED,
        redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
        access=Access.PUBLIC,
        revision=None,
        fetch=False,
        max_gb=0.5,
        blockers=(UNRESOLVED_REVISION_BLOCKER, UNRESOLVED_LICENSE_BLOCKER),
        traps=(
            "Do not claim byte identity with google/paligemma-3b-pt-224 without a saved file-by-file comparison.",
            "Applicable terms travel with the original model, not with the mirror's convenience.",
        ),
        notes=("fetch=False until a comparison report and the applicable terms are recorded.",),
    ),
    AssetPlanEntry(
        name="susie-subgoal",
        repo_id="kvablack/susie",
        repo_type=RepoType.MODEL,
        purpose="SuSIE subgoal diffusion component (goal-image conditioned).",
        source_url="https://huggingface.co/kvablack/susie",
        license=None,
        license_status=LicenseStatus.UNRESOLVED,
        redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
        access=Access.PUBLIC,
        revision=None,
        extra_allow_patterns=("*",),
        max_gb=5.0,
        historical_total_estimate_bytes=3_438_000_000,
        required_for_primary_matrix=True,
        blockers=(UNRESOLVED_REVISION_BLOCKER, UNRESOLVED_LICENSE_BLOCKER),
        notes=(
            "SuSIE needs separate subgoal and low-level components plus a pinned JAX/Flax Stable Diffusion stack.",
            "AutoEval's released configuration uses gc_bc while upstream specifies gc_ddpm_bc; keep the "
            "replication arm and the corrected arm as separate named policy identities.",
        ),
    ),
    AssetPlanEntry(
        name="susie-low-level",
        repo_id="patreya/gcbc-bridge",
        repo_type=RepoType.MODEL,
        purpose="Goal-conditioned low-level policy used by SuSIE and executed directly by SuSIE_LL.",
        source_url="https://huggingface.co/patreya/gcbc-bridge",
        license="MIT",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=None,
        files=(PlannedFile(path="checkpoint_75000", expected_bytes=258_718_956, is_directory=True),),
        max_gb=1.0,
        historical_total_estimate_bytes=258_718_956,
        required_for_primary_matrix=True,
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
        notes=("MIT is advertised on the card; confirm it at the pinned revision and preserve notices.",),
    ),
    # -------------------------------------------------------------- judge and data
    AssetPlanEntry(
        name="qwen2.5-vl-7b-instruct",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        repo_type=RepoType.MODEL,
        purpose="Rubric-judge candidate served with a pinned runtime and processor.",
        source_url="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
        license="Apache-2.0",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=None,
        extra_allow_patterns=("*",),
        max_gb=24.0,
        required_for_primary_matrix=True,
        compatibility_profile_id="qwen2.5-vl-judge",
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
        notes=("It supports image and video inputs; robot-scoring quality still requires Gate D.",),
    ),
    AssetPlanEntry(
        name="vjepa2-vitl-fpc64-256",
        repo_id="facebook/vjepa2-vitl-fpc64-256",
        repo_type=RepoType.MODEL,
        purpose="Frozen video-feature diagnostic; an encoder, not a text-following rubric judge.",
        source_url="https://huggingface.co/facebook/vjepa2-vitl-fpc64-256",
        license="Apache-2.0",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=None,
        files=(
            PlannedFile(path="config.json", expected_bytes=785),
            PlannedFile(path="model.safetensors", expected_bytes=1_303_947_864),
            PlannedFile(path="video_preprocessor_config.json", expected_bytes=1_298),
        ),
        ignore_patterns=("original/*",),
        max_gb=2.0,
        historical_total_estimate_bytes=1_303_949_947,
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
        traps=("Skip the duplicate original/model.pth.",),
        notes=(
            "Use its documented 64-frame preprocessing and explicit timestamp sampling, outside the "
            "deterministic validity gate.",
        ),
    ),
    AssetPlanEntry(
        name="bridge-orig-lerobot",
        repo_id="IPEC-COMMUNITY/bridge_orig_lerobot",
        repo_type=RepoType.DATASET,
        purpose="Bridge source episodes for start panels and normalizer/state fixtures.",
        source_url="https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot",
        license="Apache-2.0",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.LOCAL_USE_ONLY,
        access=Access.PUBLIC,
        revision=None,
        extra_allow_patterns=("meta/*",),
        max_gb=20.0,
        expected_file_count=99_673,
        required_for_primary_matrix=True,
        loader_revision=COSMOS_FRAMEWORK_REVISION,
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
        traps=(
            "Pull only manifest-selected episodes; the whole dataset is 99,673 files.",
            "It is not automatically the same scene distribution as AutoEval.",
        ),
        notes=("Extend allow_patterns with the exact manifest-selected episode paths at run time.",),
    ),
    AssetPlanEntry(
        name="auto-eval",
        repo_id="zhouzypaul/auto_eval",
        repo_type=RepoType.DATASET,
        purpose="Drawer-scene validation and human reannotation source.",
        source_url="https://huggingface.co/datasets/zhouzypaul/auto_eval",
        license=None,
        license_status=LicenseStatus.UNRESOLVED,
        redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
        access=Access.PUBLIC,
        revision=None,
        extra_allow_patterns=(),
        max_gb=40.0,
        historical_total_estimate_bytes=173_600_000_000,
        expected_file_count=None,
        loader_revision=AUTOEVAL_CODE_REVISION,
        blockers=(UNRESOLVED_REVISION_BLOCKER, UNRESOLVED_LICENSE_BLOCKER),
        traps=(
            "Some pickle objects import robot_eval_logger and wandb. A stub unpickler is NOT a security boundary: "
            "convert under restricted loading in an isolated, credential-free, network-free environment.",
            "The public index exposes the drawer scene only; it does not establish matched sink/cloth panels or "
            "policy identity, and it is not human-labelled five-task ground truth.",
            "173.6 GB and 12,054 episodes are handoff estimates; select episodes explicitly and never fetch the "
            "whole dataset under the default ceiling.",
        ),
        notes=("Supply explicit --allow patterns for the selected episodes; there is no whole-repo default here.",),
    ),
    # ------------------------------------------------------------- source repositories
    AssetPlanEntry(
        name="code-openvla",
        repo_id="github.com/openvla/openvla",
        repo_type=RepoType.GIT,
        purpose="Pinned OpenVLA Bridge control loop reviewed before trust_remote_code is enabled.",
        source_url="https://github.com/openvla/openvla/tree/" + OPENVLA_CODE_REVISION,
        license="MIT",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.PERMITTED_WITH_NOTICES,
        access=Access.SOURCE_REPOSITORY,
        revision=OPENVLA_CODE_REVISION,
        max_gb=0.5,
        loader_revision=OPENVLA_CODE_REVISION,
    ),
    AssetPlanEntry(
        name="code-auto-eval",
        repo_id="github.com/zhouzypaul/auto_eval",
        repo_type=RepoType.GIT,
        purpose="Released AutoEval policy wrappers and eval configuration for the replication arm.",
        source_url="https://github.com/zhouzypaul/auto_eval/tree/" + AUTOEVAL_CODE_REVISION,
        license="MIT",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.PERMITTED_WITH_NOTICES,
        access=Access.SOURCE_REPOSITORY,
        revision=AUTOEVAL_CODE_REVISION,
        max_gb=0.5,
        loader_revision=AUTOEVAL_CODE_REVISION,
        notes=("Its eval config uses gc_bc for SuSIE while upstream specifies gc_ddpm_bc.",),
    ),
    AssetPlanEntry(
        name="code-open-pi-zero",
        repo_id="github.com/allenzren/open-pi-zero",
        repo_type=RepoType.GIT,
        purpose="Pinned OpenPiZero loader and action normalization source.",
        source_url="https://github.com/allenzren/open-pi-zero/tree/" + OPEN_PI_ZERO_CODE_REVISION,
        license="MIT",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.PERMITTED_WITH_NOTICES,
        access=Access.SOURCE_REPOSITORY,
        revision=OPEN_PI_ZERO_CODE_REVISION,
        max_gb=0.5,
        loader_revision=OPEN_PI_ZERO_CODE_REVISION,
    ),
    AssetPlanEntry(
        name="code-irasim",
        repo_id="github.com/bytedance/IRASim",
        repo_type=RepoType.GIT,
        purpose="Pinned IRASim Bridge configuration and data adapter.",
        source_url="https://github.com/bytedance/IRASim/tree/" + IRASIM_CODE_REVISION,
        license="Apache-2.0",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.PERMITTED_WITH_NOTICES,
        access=Access.SOURCE_REPOSITORY,
        revision=IRASIM_CODE_REVISION,
        max_gb=0.5,
        loader_revision=IRASIM_CODE_REVISION,
    ),
    AssetPlanEntry(
        name="code-cosmos-framework",
        repo_id="github.com/NVIDIA/cosmos-framework",
        repo_type=RepoType.GIT,
        purpose="Pinned Bridge pose conventions and action normalization for the Cosmos compiler.",
        source_url="https://github.com/NVIDIA/cosmos-framework/tree/" + COSMOS_FRAMEWORK_REVISION,
        license="Apache-2.0",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.PERMITTED_WITH_NOTICES,
        access=Access.SOURCE_REPOSITORY,
        revision=COSMOS_FRAMEWORK_REVISION,
        max_gb=0.5,
        loader_revision=COSMOS_FRAMEWORK_REVISION,
    ),
    AssetPlanEntry(
        name="code-octo",
        repo_id="github.com/octo-models/octo",
        repo_type=RepoType.GIT,
        purpose="Octo v1.0 model API and the released temporal-ensembling wrapper.",
        source_url="https://github.com/octo-models/octo",
        license="MIT",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.PERMITTED_WITH_NOTICES,
        access=Access.SOURCE_REPOSITORY,
        revision=None,
        max_gb=0.5,
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
        traps=("Pin the v1.0 commit; the 1.5 line changes observation keys and checkpoints.",),
        notes=(
            "The released TemporalEnsembleWrapper's exp_weight and the exact observation key names must be read "
            "from this pinned commit and confirmed by a golden action fixture.",
        ),
    ),
    AssetPlanEntry(
        name="code-openvla-mini",
        repo_id="github.com/Stanford-ILIAD/openvla-mini",
        repo_type=RepoType.GIT,
        purpose="MiniVLA loader with use_extra=True and its seven-action chunk configuration.",
        source_url="https://github.com/Stanford-ILIAD/openvla-mini",
        license=None,
        license_status=LicenseStatus.UNRESOLVED,
        redistribution=Redistribution.PROHIBITED_PENDING_RESOLUTION,
        access=Access.SOURCE_REPOSITORY,
        revision=None,
        max_gb=0.5,
        blockers=(UNRESOLVED_REVISION_BLOCKER, UNRESOLVED_LICENSE_BLOCKER),
    ),
    AssetPlanEntry(
        name="code-susie",
        repo_id="github.com/kvablack/susie",
        repo_type=RepoType.GIT,
        purpose="Upstream SuSIE subgoal cadence and gc_ddpm_bc configuration.",
        source_url="https://github.com/kvablack/susie",
        license="MIT",
        license_status=LicenseStatus.ADVERTISED_UNVERIFIED,
        redistribution=Redistribution.PERMITTED_WITH_NOTICES,
        access=Access.SOURCE_REPOSITORY,
        revision=None,
        max_gb=0.5,
        blockers=(UNRESOLVED_REVISION_BLOCKER,),
    ),
)


ASSET_PLAN: Mapping[str, AssetPlanEntry] = {entry.name: entry for entry in _ENTRIES}


def plan_names() -> Tuple[str, ...]:
    return tuple(ASSET_PLAN)


def get_plan(name: str) -> AssetPlanEntry:
    try:
        return ASSET_PLAN[name]
    except KeyError:
        raise KeyError("Unknown asset plan %r; known plans: %s" % (name, ", ".join(plan_names()))) from None


def validate_plan() -> Tuple[str, ...]:
    """Every internal-consistency problem across the whole plan."""

    problems: List[str] = []
    for name, entry in ASSET_PLAN.items():
        if entry.name != name:
            problems.append("%s: entry name does not match its key" % name)
        problems.extend("%s: %s" % (name, error) for error in entry.errors())
    return tuple(problems)


def asset_plan_entries() -> Tuple[Mapping[str, Any], ...]:
    """Candidate ``assets.lock.json`` records for every planned asset.

    Unknown values stay ``null`` and appear in ``verification_blockers``, so an
    entry can never reach ``verified`` on the strength of this plan alone.
    """

    problems = validate_plan()
    if problems:
        raise ValueError("Asset plan is internally inconsistent:\n" + "\n".join(problems))
    return tuple(entry.lock_entry() for entry in _ENTRIES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the pinned PLUMB asset plan.")
    parser.add_argument("--name", help="Show one plan entry instead of all of them")
    parser.add_argument("--lock-entries", action="store_true", help="Emit assets.lock.json candidate records")
    parser.add_argument("--validate", action="store_true", help="Only report plan-consistency problems")
    args = parser.parse_args()
    problems = validate_plan()
    if args.validate:
        print(json.dumps({"problems": list(problems)}, indent=2))
        return 0 if not problems else 2
    if problems:
        print(json.dumps({"problems": list(problems)}, indent=2))
        return 2
    if args.name:
        print(json.dumps(get_plan(args.name).lock_entry(), indent=2))
        return 0
    if args.lock_entries:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "entries": list(asset_plan_entries())}, indent=2))
        return 0
    print(
        json.dumps(
            [
                {
                    "name": entry.name,
                    "repo_id": entry.repo_id,
                    "repo_type": entry.repo_type.value,
                    "revision": entry.revision,
                    "license": entry.license,
                    "license_status": entry.license_status.value,
                    "redistribution": entry.redistribution.value,
                    "access": entry.access.value,
                    "fetch": entry.fetch,
                    "required_for_primary_matrix": entry.required_for_primary_matrix,
                    "blockers": list(entry.verification_blockers()),
                }
                for entry in _ENTRIES
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
