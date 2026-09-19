"""Safely convert reviewed MiniVLA and OpenPiZero legacy pickles to safetensors.

MiniVLA and OpenPiZero are the two policies in the matrix whose released weights
ship as legacy PyTorch pickles, and their adapters refuse those pickles outright:
``plumb/policies/minivla.py`` and ``plumb/policies/openpizero.py`` consume only a
``.safetensors`` artifact that carries its own SHA-256, the source pickle's
SHA-256, and the SHA-256 of an *isolated conversion report*.  Nothing else in the
tree produces that third hash, so both policies stay blocked until this tool
runs.

It follows ``cluster/convert_irasim_checkpoint.py``:

* the pickle's globals are enumerated **before** anything is deserialized, and
  any global outside an explicit per-policy allowlist is named and refused;
* every approved non-builtin global is aliased to :class:`InertMetadata`, so no
  upstream ``__reduce__`` or ``__setstate__`` body ever runs;
* ``torch.load`` is called only with ``weights_only=True``.  There is no unsafe
  fallback in this file and none may be added;
* the output and report paths are reserved with ``O_EXCL`` and an existing
  artifact is never overwritten;
* a machine-readable report records what was, and was not, established.

Beyond that pattern it enforces the parts of the build spec the IRASim converter
left to the operator: it refuses to run while the environment still holds
credentials or root, it checks the source bytes against ``cluster/asset_plan.py``
instead of trusting the path it was handed, and it refuses to write a license
string for ``Stanford-ILIAD/pretrain_vq``, which declares none at all.

What this tool does *not* establish is that the source pickle is benign; see
:data:`SECURITY_NOTE`.  Allowlisted deserialization is a mitigation.  Unknown
values stay ``None`` and appear in ``verification_blockers``, so nothing here
invents a hash, a license, or a tensor.

Usage, from the repository root inside the disposable environment::

    python -m cluster.convert_policy_checkpoint --list-policies
    python -m cluster.convert_policy_checkpoint --policy openpizero \\
        --source /scratch/.../bridge_beta_step19296_2024-12-26_22-30_42.pt --inspect
    python -m cluster.convert_policy_checkpoint --policy openpizero \\
        --source /scratch/.../bridge_beta_step19296_2024-12-26_22-30_42.pt \\
        --output /scratch/.../open-pi-zero-bridge-beta.safetensors \\
        --source-revision <40-hex commit> --expected-source-sha256 <64-hex> --execute
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from cluster.asset_plan import (
    MISSING_HASH_BLOCKER,
    UNRESOLVED_LICENSE_BLOCKER,
    UNRESOLVED_REVISION_BLOCKER,
    AssetPlanEntry,
    LicenseStatus,
    PlannedFile,
    Redistribution,
    get_plan,
)


SCHEMA_VERSION = 1
REPORT_KIND = "plumb_policy_safe_checkpoint_conversion"

#: The only output container the policy adapters accept.
OUTPUT_SUFFIX = ".safetensors"

#: Nested state dicts are flattened with ``.`` so the emitted keys are real
#: ``torch.nn.Module`` parameter names: ``checkpoint["model"]["projector"]["fc.weight"]``
#: becomes ``projector.fc.weight``, which is what ``load_state_dict`` expects.
FLATTEN_SEPARATOR = "."

#: Mirrors ``plumb.policies.minivla.KNOWN_LICENSE_STATUSES``.  Duplicated rather
#: than imported to keep this cluster tool free of the inference package; the
#: test suite asserts the two stay equal.
ADAPTER_KNOWN_LICENSE_STATUSES = ("declared", "advertised_unverified", "absent_cardData_null", "unresolved")

#: Mirrors ``plumb.policies.minivla.MINIVLA_VQ_REDISTRIBUTION``.
ADAPTER_PROHIBITED_REDISTRIBUTION = "prohibited_pending_resolution"

STATE_DICT_COVERAGE_BLOCKER = "state_dict_coverage_unverified"
UNVERIFIED_SOURCE_SHA_BLOCKER = "source_sha256_not_verified"
NETWORK_ISOLATION_BLOCKER = "network_isolation_not_verifiable_from_inside"
SOURCE_NAME_BLOCKER = "source_filename_differs_from_asset_plan"
LICENSE_ACKNOWLEDGEMENT_BLOCKER = "unresolved_license_not_acknowledged"

SECURITY_NOTE = (
    "Allowlisted deserialization is a mitigation, not a proof of safety. This conversion enumerated the source "
    "pickle's globals before loading it, refused every global outside the reviewed per-policy allowlist, aliased "
    "each approved non-builtin global to an inert recorder so no upstream __reduce__ or __setstate__ body ran, and "
    "called torch.load only with weights_only=True. None of that proves the source pickle benign: a pickle scan or "
    "strict=True alone does not establish safety, and a stub unpickler is not a security boundary. Run this tool "
    "only in a disposable, unprivileged environment without credentials or network, discard that environment "
    "afterwards, and treat the recorded SHA-256 hashes as the only durable claim this report makes."
)

NETWORK_ISOLATION_NOTE = (
    "This tool performs no network access and cannot verify from inside the environment that the environment has "
    "none. Network isolation is the operator's to establish and is not claimed here."
)

STRICT_VERIFICATION_NOTE = (
    "Full state-dict coverage was not verified. strict=True checks keys, not provenance or deserialization safety, "
    "and it was not run at all here. Do not set the OpenPiZero profile's state_dict_coverage_verified on the "
    "strength of this report; re-run with --strict-verify against the pinned loader instead."
)


# -- credential and privilege isolation ----------------------------------------

#: Exact names checked first, including the two this project actually uses.
CREDENTIAL_ENVIRONMENT_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET",
    "BASETEN_API_KEY",
    "BASETEN_WEBHOOK_SECRET",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "OPENAI_API_KEY",
    "REPLICATE_API_TOKEN",
    "SSH_AUTH_SOCK",
    "WANDB_API_KEY",
)

#: Substring scan so an unlisted credential still trips the refusal.
CREDENTIAL_NAME_FRAGMENTS = (
    "ACCESS_KEY",
    "APIKEY",
    "API_KEY",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)

#: Credential material on disk, relative to the home directory.
CREDENTIAL_FILES = (
    ".aws/credentials",
    ".cache/huggingface/token",
    ".config/gh/hosts.yml",
    ".docker/config.json",
    ".git-credentials",
    ".huggingface/token",
    ".kaggle/kaggle.json",
    ".netrc",
    ".ssh/id_ed25519",
    ".ssh/id_rsa",
)


# -- pickle globals ------------------------------------------------------------

#: Globals that are never approvable, whatever the operator passes.  Aliasing
#: them to an inert recorder would in fact neutralise them, but a checkpoint that
#: references this machinery at all is not a checkpoint to convert: it is one to
#: report.  Parent modules match, so ``urllib.request.urlopen`` is covered by
#: ``urllib``.
NEVER_APPROVABLE_MODULES = frozenset(
    {
        "asyncio",
        "atexit",
        "bdb",
        "builtins",
        "cffi",
        "code",
        "codeop",
        "ctypes",
        "ftplib",
        "functools",
        "gc",
        "glob",
        "http",
        "httpx",
        "imp",
        "importlib",
        "marshal",
        "multiprocessing",
        "nt",
        "operator",
        "os",
        "paramiko",
        "pdb",
        "pickle",
        "pickletools",
        "platform",
        "posix",
        "pty",
        "requests",
        "runpy",
        "shutil",
        "signal",
        "smtplib",
        "socket",
        "ssl",
        "subprocess",
        "sys",
        "telnetlib",
        "tempfile",
        "threading",
        "torch.hub",
        "torch.jit",
        "torch.serialization",
        "types",
        "urllib",
        "webbrowser",
    }
)

#: Approvable globals that resolve to the real inert object instead of to
#: :class:`InertMetadata`, because their own behaviour is the point and they
#: carry no reachable code of their own.
INERT_BUILTIN_TARGETS: Mapping[str, Tuple[str, str]] = {
    "builtins.bool": ("builtins", "bool"),
    "builtins.bytes": ("builtins", "bytes"),
    "builtins.complex": ("builtins", "complex"),
    "builtins.dict": ("builtins", "dict"),
    "builtins.float": ("builtins", "float"),
    "builtins.frozenset": ("builtins", "frozenset"),
    "builtins.int": ("builtins", "int"),
    "builtins.list": ("builtins", "list"),
    "builtins.set": ("builtins", "set"),
    "builtins.str": ("builtins", "str"),
    "builtins.tuple": ("builtins", "tuple"),
    "collections.OrderedDict": ("collections", "OrderedDict"),
    "collections.defaultdict": ("collections", "defaultdict"),
}


class ConversionRefusal(RuntimeError):
    """This tool refused to proceed, with the reason stated in the message."""


class InertMetadata:
    """Inert pickle target: retain state without running the real class's code.

    Approved non-builtin globals are aliased to this, exactly as the IRASim
    converter aliases OmegaConf's metadata classes, so the unpickler can rebuild
    the object graph's shape without importing or executing anything upstream.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.state: Any = None

    def __setstate__(self, state: Any) -> None:
        self.state = state


@dataclass(frozen=True)
class PolicyTarget:
    """One convertible policy checkpoint and the review that applies to it."""

    policy: str
    asset_plan_name: str
    artifact_label: str
    adapter_fields_kind: str
    reviewed_globals: Tuple[str, ...]
    state_dict_keys: Tuple[str, ...]
    expectation: str

    def __post_init__(self) -> None:
        if self.adapter_fields_kind not in ("minivla_converted_artifact", "open_pi_zero_profile"):
            raise ValueError("Unknown adapter_fields_kind %r" % self.adapter_fields_kind)
        for name in self.reviewed_globals:
            reason = never_approvable_reason(name)
            if reason is not None:
                raise ValueError("Reviewed allowlist for %s may not contain %s: %s" % (self.policy, name, reason))

    def payload(self) -> Dict[str, Any]:
        entry = get_plan(self.asset_plan_name)
        planned = planned_file(entry)
        return {
            "policy": self.policy,
            "asset_plan_name": self.asset_plan_name,
            "repo_id": entry.repo_id,
            "repo_path": planned.path,
            "url_path": planned.url_path,
            "artifact_label": self.artifact_label,
            "adapter_fields_kind": self.adapter_fields_kind,
            "reviewed_globals": list(self.reviewed_globals),
            "state_dict_keys": list(self.state_dict_keys),
            "expectation": self.expectation,
            "license": entry.license,
            "license_status": entry.license_status.value,
            "redistribution": entry.redistribution.value,
            "license_acknowledgement_required": license_acknowledgement_required(entry),
            "historical_estimate_bytes": planned.historical_estimate_bytes,
            "revision_pinned_in_asset_plan": entry.revision is not None,
            "source_sha256_pinned_in_asset_plan": planned.sha256 is not None,
        }


POLICY_TARGETS: Tuple[PolicyTarget, ...] = (
    PolicyTarget(
        policy="minivla",
        asset_plan_name="minivla-vq-bridge",
        artifact_label="Stanford-ILIAD/minivla-vq-bridge-prismatic checkpoint",
        adapter_fields_kind="minivla_converted_artifact",
        reviewed_globals=(),
        state_dict_keys=("model",),
        expectation=(
            "The prismatic release stores per-submodule state dicts under 'model'. Ordered mappings and the tensor "
            "rebuild functions are already inside Torch's weights_only allowlist, so the reviewed allowlist here is "
            "empty: no additional global is expected. That is an expectation about a file this tool has not seen, "
            "not a claim about its bytes. Run --inspect first and review anything it reports."
        ),
    ),
    PolicyTarget(
        policy="minivla_vq",
        asset_plan_name="minivla-pretrain-vq",
        artifact_label="Stanford-ILIAD/pretrain_vq action tokenizer",
        adapter_fields_kind="minivla_converted_artifact",
        reviewed_globals=(),
        state_dict_keys=("model", "state_dict"),
        expectation=(
            "The VQ tokenizer checkpoint is a small state dict. The reviewed allowlist is empty: no global beyond "
            "Torch's weights_only defaults is expected. Confirm with --inspect before converting; this repository "
            "declares no license, so read the license block of the report too."
        ),
    ),
    PolicyTarget(
        policy="openpizero",
        asset_plan_name="open-pi-zero",
        artifact_label="allenzren/open-pi-zero bridge_beta checkpoint",
        adapter_fields_kind="open_pi_zero_profile",
        reviewed_globals=(),
        state_dict_keys=("model",),
        expectation=(
            "The pinned OpenPiZero loader reads this checkpoint's 'model' entry under weights_only=True, so the "
            "reviewed allowlist is empty and no additional global is expected. Confirm with --inspect. Coverage of "
            "the full state dict is a separate question from deserialization safety: use --strict-verify for it."
        ),
    ),
)

TARGETS_BY_POLICY: Mapping[str, PolicyTarget] = {target.policy: target for target in POLICY_TARGETS}


def policy_names() -> Tuple[str, ...]:
    return tuple(TARGETS_BY_POLICY)


def get_target(policy: str) -> PolicyTarget:
    try:
        return TARGETS_BY_POLICY[policy]
    except KeyError:
        raise ConversionRefusal(
            "Unknown policy %r; this tool converts exactly %s." % (policy, ", ".join(policy_names()))
        ) from None


def planned_file(entry: AssetPlanEntry) -> PlannedFile:
    """The single planned file of a convertible asset, from the pinned plan."""

    files = tuple(planned for planned in entry.files if not planned.is_directory)
    if len(files) != 1:
        raise ConversionRefusal(
            "Asset plan entry %r declares %d convertible files; this tool converts one named checkpoint per policy."
            % (entry.name, len(files))
        )
    return files[0]


def license_acknowledgement_required(entry: AssetPlanEntry) -> bool:
    """Whether the operator must acknowledge an unresolved license explicitly.

    ``Stanford-ILIAD/pretrain_vq`` declares none at all, and the MiniVLA
    checkpoint's own terms are unresolved in the plan, so both require it.
    """

    return (
        entry.license is None
        or entry.license_status
        in (LicenseStatus.UNRESOLVED, LicenseStatus.ABSENT_CARD_DATA_NULL, LicenseStatus.GATED_TERMS_REQUIRED)
        or entry.redistribution is Redistribution.PROHIBITED_PENDING_RESOLUTION
    )


def never_approvable_reason(name: str) -> Optional[str]:
    """Why ``name`` may never be allowlisted, or ``None`` if it may be."""

    if name in INERT_BUILTIN_TARGETS:
        return None
    if "." not in name:
        return "%r is not a qualified module.attribute pickle global" % name
    parts = name.rsplit(".", 1)[0].split(".")
    for index in range(len(parts), 0, -1):
        module = ".".join(parts[:index])
        if module in NEVER_APPROVABLE_MODULES:
            return (
                "module %r can execute code or reach the host; a checkpoint that references it is refused outright "
                "rather than converted" % module
            )
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_sha256(value: Optional[str]) -> Optional[str]:
    """A bare lowercase 64-hex digest, or ``None`` when the input is not one."""

    if not isinstance(value, str):
        return None
    candidate = value[len("sha256:"):] if value.startswith("sha256:") else value
    candidate = candidate.strip().lower()
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        return None
    return candidate


def is_immutable_revision(value: Optional[str]) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


# -- environment ---------------------------------------------------------------


@dataclass(frozen=True)
class Environment:
    """The process environment, injected so the refusals are unit-testable."""

    variables: Mapping[str, str]
    euid: Optional[int]
    home: Optional[str]

    @classmethod
    def from_process(cls) -> "Environment":
        getter = getattr(os, "geteuid", None)
        try:
            home: Optional[str] = str(Path.home())
        except (RuntimeError, OSError):
            home = None
        return cls(variables=dict(os.environ), euid=getter() if callable(getter) else None, home=home)


@dataclass(frozen=True)
class IsolationCheck:
    """What was actually established about the environment's isolation."""

    credential_variables: Tuple[str, ...]
    credential_files: Tuple[str, ...]
    privileged: bool
    euid: Optional[int]

    @property
    def refusals(self) -> Tuple[str, ...]:
        problems: List[str] = []
        if self.credential_variables:
            problems.append(
                "credentials are present in the environment: %s. The build spec requires a disposable environment "
                "without credentials; unset them (for example with 'env -u %s') rather than asking this tool to "
                "trust the operator."
                % (", ".join(self.credential_variables), " -u ".join(self.credential_variables))
            )
        if self.credential_files:
            problems.append(
                "credential material is readable in the home directory: %s. Convert in a disposable environment "
                "whose home directory holds no credentials." % ", ".join(self.credential_files)
            )
        if self.privileged:
            problems.append(
                "this process is running as root (euid 0). The build spec requires an unprivileged environment; "
                "re-run as a normal user."
            )
        return tuple(problems)

    def payload(self) -> Dict[str, Any]:
        return {
            "credential_environment_variables_present": list(self.credential_variables),
            "credential_files_present": list(self.credential_files),
            "credential_variable_names_checked": list(CREDENTIAL_ENVIRONMENT_VARIABLES),
            "credential_name_fragments_checked": list(CREDENTIAL_NAME_FRAGMENTS),
            "credential_files_checked": list(CREDENTIAL_FILES),
            "euid": self.euid,
            "unprivileged": not self.privileged,
            "network_isolation_verified": False,
            "network_isolation_note": NETWORK_ISOLATION_NOTE,
        }


def check_isolation(environment: Environment) -> IsolationCheck:
    """Name every credential and privilege the environment still carries."""

    exact = set(CREDENTIAL_ENVIRONMENT_VARIABLES)
    found: List[str] = []
    for name, value in environment.variables.items():
        if not value:
            continue
        upper = name.upper()
        if upper in exact or any(fragment in upper for fragment in CREDENTIAL_NAME_FRAGMENTS):
            found.append(name)
    files: List[str] = []
    if environment.home:
        root = Path(environment.home)
        for relative in CREDENTIAL_FILES:
            try:
                present = (root / relative).is_file()
            except OSError:
                present = False
            if present:
                files.append(relative)
    return IsolationCheck(
        credential_variables=tuple(sorted(found)),
        credential_files=tuple(files),
        privileged=environment.euid == 0,
        euid=environment.euid,
    )


# -- runtime seam --------------------------------------------------------------


@dataclass(frozen=True)
class ConverterRuntime:
    """The lazily imported Torch/safetensors surface this tool actually uses."""

    torch: Any
    save_file: Callable[..., Any]
    torch_version: Optional[str]
    safetensors_version: Optional[str]


def default_runtime() -> ConverterRuntime:
    """Import Torch and safetensors lazily; this module imports neither at top level."""

    try:
        import torch  # type: ignore
    except ImportError as error:
        raise ConversionRefusal(
            "Torch is unavailable. Install the policy's own CPU Torch build (2.5 or newer, whose weights_only "
            "unpickler can enumerate a checkpoint's globals) inside the disposable conversion environment; the "
            "import here is lazy."
        ) from error
    try:
        import safetensors.torch as safetensors_torch  # type: ignore
    except ImportError as error:
        raise ConversionRefusal(
            "safetensors is unavailable, and it is the only output container the policy adapters accept."
        ) from error
    serialization = getattr(torch, "serialization", None)
    for attribute in ("get_unsafe_globals_in_checkpoint", "safe_globals"):
        if not callable(getattr(serialization, attribute, None)):
            raise ConversionRefusal(
                "torch.serialization.%s is missing from Torch %r. This tool enumerates a checkpoint's globals "
                "before loading it and will not fall back to an unchecked load; use Torch 2.5 or newer."
                % (attribute, getattr(torch, "__version__", "unknown"))
            )
    safetensors_version: Optional[str]
    try:
        import safetensors  # type: ignore

        safetensors_version = str(getattr(safetensors, "__version__", "")) or None
    except ImportError:  # pragma: no cover - safetensors.torch imported above
        safetensors_version = None
    return ConverterRuntime(
        torch=torch,
        save_file=safetensors_torch.save_file,
        torch_version=str(getattr(torch, "__version__", "")) or None,
        safetensors_version=safetensors_version,
    )


def converter_source_revision(path: Path) -> Optional[str]:
    """This file's own git commit, or ``None`` when git cannot answer offline."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and is_immutable_revision(revision) else None


# -- request -------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionRequest:
    """One reviewed conversion, fully specified before anything is read."""

    target: PolicyTarget
    source: Path
    mode: str
    output: Optional[Path] = None
    report: Optional[Path] = None
    expected_source_sha256: Optional[str] = None
    source_revision: Optional[str] = None
    acknowledge_unresolved_license: bool = False
    operator_approved_globals: Tuple[str, ...] = ()
    reviewer: Optional[str] = None
    state_dict_key: Optional[str] = None
    strict_verify: Optional[str] = None
    strict_verify_repo: Optional[str] = None
    strict_verify_loader_revision: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in ("inspect", "dry_run", "execute"):
            raise ValueError("mode must be inspect, dry_run, or execute")

    @property
    def converts(self) -> bool:
        """Whether this request is a conversion, as opposed to an inspection."""

        return self.mode in ("dry_run", "execute")

    @property
    def report_path(self) -> Optional[Path]:
        if self.report is not None:
            return self.report
        if self.output is None:
            return None
        return self.output.with_suffix(".conversion-report.json")


# -- source and globals --------------------------------------------------------


@dataclass(frozen=True)
class SourceVerification:
    """The source pickle as measured, against what the plan and operator pinned."""

    local_path: str
    sha256: str
    bytes: int
    expected_sha256: Optional[str]
    expected_sha256_origin: Optional[str]
    basename_matches_plan: bool

    @property
    def verified(self) -> bool:
        return self.expected_sha256 is not None and self.expected_sha256 == self.sha256

    def payload(self) -> Dict[str, Any]:
        return {
            "local_path": self.local_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "expected_sha256": self.expected_sha256,
            "expected_sha256_origin": self.expected_sha256_origin,
            "sha256_verified": self.verified,
            "basename_matches_asset_plan": self.basename_matches_plan,
        }


def resolve_expected_sha256(planned: PlannedFile, operator: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """The digest to require, and where it came from, refusing a disagreement."""

    pinned = normalized_sha256(planned.sha256)
    if planned.sha256 is not None and pinned is None:
        raise ConversionRefusal(
            "Asset plan pins a malformed SHA-256 for %r; fix the plan rather than working around it." % planned.path
        )
    supplied = normalized_sha256(operator)
    if operator is not None and supplied is None:
        raise ConversionRefusal("--expected-source-sha256 must be a 64-character hexadecimal SHA-256.")
    if pinned is not None and supplied is not None and pinned != supplied:
        raise ConversionRefusal(
            "--expected-source-sha256 %s disagrees with the SHA-256 pinned in cluster/asset_plan.py for %r (%s). "
            "Resolve which one is right before converting anything." % (supplied, planned.path, pinned)
        )
    if pinned is not None:
        return pinned, "asset_plan"
    if supplied is not None:
        return supplied, "operator"
    return None, None


def verify_source(request: ConversionRequest, planned: PlannedFile) -> SourceVerification:
    """Measure the source file and compare it with the pinned expectation."""

    expected, origin = resolve_expected_sha256(planned, request.expected_source_sha256)
    source = request.source
    if not source.is_file():
        raise ConversionRefusal("Source checkpoint %s is not a file." % source)
    if expected is None and request.converts:
        raise ConversionRefusal(
            "cluster/asset_plan.py pins no SHA-256 for %r (sha256: null, blocker %r), so the source bytes cannot be "
            "verified. Pass --expected-source-sha256 with the digest recorded at download time. This tool will not "
            "fabricate one." % (planned.path, MISSING_HASH_BLOCKER)
        )
    size = source.stat().st_size
    if planned.expected_bytes is not None and size != planned.expected_bytes:
        raise ConversionRefusal(
            "Source %s is %d bytes; the plan states an exact %d for %r."
            % (source, size, planned.expected_bytes, planned.path)
        )
    verification = SourceVerification(
        local_path=str(source),
        sha256=sha256_of(source),
        bytes=size,
        expected_sha256=expected,
        expected_sha256_origin=origin,
        basename_matches_plan=source.name == Path(planned.path).name,
    )
    if expected is not None and not verification.verified:
        raise ConversionRefusal(
            "Source checksum mismatch for %s: measured %s, expected %s (%s). Refusing to convert bytes that are not "
            "the reviewed checkpoint."
            % (source, verification.sha256, expected, origin or "unknown origin")
        )
    return verification


@dataclass(frozen=True)
class GlobalsDecision:
    """Which pickle globals were observed, allowed, aliased, and refused."""

    observed: Tuple[str, ...]
    reviewed: Tuple[str, ...]
    operator_approved: Tuple[str, ...]
    reviewer: Optional[str]

    @property
    def approved(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.reviewed) | set(self.operator_approved)))

    @property
    def approved_used(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.observed) & set(self.approved)))

    @property
    def rejected(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.observed) - set(self.approved)))

    @property
    def never_approvable(self) -> Tuple[str, ...]:
        return tuple(sorted(name for name in self.observed if never_approvable_reason(name) is not None))

    @property
    def inert_aliased(self) -> Tuple[str, ...]:
        return tuple(name for name in self.approved if name not in INERT_BUILTIN_TARGETS)

    def payload(self) -> Dict[str, Any]:
        return {
            "method": "torch.load(weights_only=True) inside torch.serialization.safe_globals",
            "weights_only": True,
            "unsafe_fallback_available": False,
            "globals_enumerated_before_load": True,
            "observed_globals": list(self.observed),
            "reviewed_globals": list(self.reviewed),
            "operator_approved_globals": list(self.operator_approved),
            "operator_reviewer": self.reviewer,
            "approved_globals": list(self.approved),
            "approved_globals_used": list(self.approved_used),
            "rejected_globals": list(self.rejected),
            "never_approvable_globals_observed": list(self.never_approvable),
            "inert_aliased_globals": list(self.inert_aliased),
        }

    def refusal(self, policy: str) -> Optional[str]:
        if self.never_approvable:
            return (
                "Refusing %s outright: its pickle references %s, which can execute code or reach the host. "
                "--approve-global does not override this. Report the finding and obtain a safetensors export from "
                "upstream." % (policy, ", ".join(self.never_approvable))
            )
        if self.rejected:
            return (
                "Refusing to deserialize %s: the pickle references globals outside the reviewed allowlist: %s. "
                "Do not stub these modules into existence to make the load succeed, and do not widen the allowlist "
                "to make it pass; a stub unpickler is not a security boundary. Review each global by hand, then "
                "re-run with --approve-global NAME --reviewer NAME, or obtain a safetensors export from upstream."
                % (policy, ", ".join(self.rejected))
            )
        return None


def enumerate_globals(runtime: ConverterRuntime, request: ConversionRequest) -> GlobalsDecision:
    """List the checkpoint's unsafe globals before anything is deserialized."""

    for name in request.operator_approved_globals:
        reason = never_approvable_reason(name)
        if reason is not None:
            raise ConversionRefusal("Refusing to approve pickle global %s: %s." % (name, reason))
    if request.operator_approved_globals and not request.reviewer:
        raise ConversionRefusal(
            "--approve-global records a human review, so --reviewer is required and is written into the report."
        )
    observed = runtime.torch.serialization.get_unsafe_globals_in_checkpoint(str(request.source))
    names = tuple(sorted(str(name) for name in observed))
    return GlobalsDecision(
        observed=names,
        reviewed=tuple(request.target.reviewed_globals),
        operator_approved=tuple(sorted(set(request.operator_approved_globals))),
        reviewer=request.reviewer,
    )


def safe_alias_pairs(decision: GlobalsDecision) -> List[Tuple[Any, str]]:
    """``(object, name)`` pairs for ``torch.serialization.safe_globals``.

    Approved globals resolve to the real object only for the inert builtins;
    everything else resolves to :class:`InertMetadata`, so the unpickler never
    imports or runs the upstream class.
    """

    import importlib

    pairs: List[Tuple[Any, str]] = []
    for name in decision.approved:
        location = INERT_BUILTIN_TARGETS.get(name)
        if location is None:
            pairs.append((InertMetadata, name))
            continue
        module_name, attribute = location
        pairs.append((getattr(importlib.import_module(module_name), attribute), name))
    return pairs


# -- state dict ----------------------------------------------------------------


def select_state_dict(
    runtime: ConverterRuntime, loaded: Any, request: ConversionRequest
) -> Tuple[Any, Optional[str], Tuple[str, ...]]:
    """The subtree to convert, the key it came from, and the keys left behind."""

    target = request.target
    if not isinstance(loaded, Mapping):
        if request.state_dict_key:
            raise ConversionRefusal(
                "--state-dict-key %r was given but the checkpoint's top level is %s, not a mapping."
                % (request.state_dict_key, type(loaded).__name__)
            )
        raise ConversionRefusal(
            "Checkpoint top level is %s, not a mapping; this tool converts state dicts." % type(loaded).__name__
        )
    top_level = tuple(str(key) for key in loaded.keys())
    if request.state_dict_key:
        node: Any = loaded
        for part in request.state_dict_key.split(FLATTEN_SEPARATOR):
            if not isinstance(node, Mapping) or part not in node:
                raise ConversionRefusal(
                    "--state-dict-key %r does not resolve in this checkpoint; its top-level keys are %s."
                    % (request.state_dict_key, ", ".join(top_level) or "<none>")
                )
            node = node[part]
        ignored = tuple(sorted(key for key in top_level if key != request.state_dict_key.split(FLATTEN_SEPARATOR)[0]))
        return node, request.state_dict_key, ignored
    for candidate in target.state_dict_keys:
        if candidate in loaded:
            return loaded[candidate], candidate, tuple(sorted(key for key in top_level if key != candidate))
    if loaded and all(runtime.torch.is_tensor(value) for value in loaded.values()):
        return loaded, None, ()
    raise ConversionRefusal(
        "Cannot identify the state dict for %s: none of the expected keys (%s) is present and the top level is not a "
        "flat tensor mapping. Its top-level keys are %s. Pass --state-dict-key explicitly after reviewing them."
        % (
            request.target.policy,
            ", ".join(target.state_dict_keys) or "<none>",
            ", ".join("%s (%s)" % (key, type(loaded[key]).__name__) for key in top_level) or "<none>",
        )
    )


def flatten_tensors(runtime: ConverterRuntime, value: Any, prefix: str = "") -> Tuple[Dict[str, Any], List[str]]:
    """Flatten nested mappings of tensors, naming every non-tensor leaf found."""

    tensors: Dict[str, Any] = {}
    rejected: List[str] = []
    if runtime.torch.is_tensor(value):
        if prefix:
            tensors[prefix] = value
        else:
            rejected.append("<root> is a bare tensor with no parameter name")
        return tensors, rejected
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                rejected.append("%s[%r] has a non-string key" % (prefix or "<root>", key))
                continue
            child = key if not prefix else prefix + FLATTEN_SEPARATOR + key
            found, problems = flatten_tensors(runtime, item, child)
            rejected.extend(problems)
            for name, tensor in found.items():
                if name in tensors:
                    rejected.append("%s is produced by two different key paths" % name)
                    continue
                tensors[name] = tensor
        return tensors, rejected
    rejected.append("%s is a %s, not a tensor" % (prefix or "<root>", type(value).__name__))
    return tensors, rejected


def tensor_summary(tensors: Mapping[str, Any]) -> Tuple[int, Dict[str, int]]:
    """Total tensor bytes and a dtype histogram."""

    total = 0
    dtypes: Dict[str, int] = {}
    for tensor in tensors.values():
        total += int(tensor.numel()) * int(tensor.element_size())
        name = str(getattr(tensor, "dtype", "unknown"))
        dtypes[name] = dtypes.get(name, 0) + 1
    return total, dict(sorted(dtypes.items()))


def strict_state_dict_verification(
    request: ConversionRequest, tensors: Mapping[str, Any]
) -> Dict[str, Any]:
    """Load the flattened state dict into the pinned architecture with ``strict=True``.

    This imports third-party loader code, which is why it is opt-in: it is the
    same trust as normal inference, taken inside the disposable environment.
    """

    if not request.strict_verify:
        return {
            "performed": False,
            "entry_point": None,
            "loader_revision": None,
            "missing_keys": [],
            "unexpected_keys": [],
            "note": STRICT_VERIFICATION_NOTE,
        }
    module_name, separator, attribute = request.strict_verify.partition(":")
    if not module_name or not separator or not attribute:
        raise ConversionRefusal("--strict-verify must be module:attribute naming the reviewed model factory.")
    if not is_immutable_revision(request.strict_verify_loader_revision):
        raise ConversionRefusal(
            "--strict-verify-loader-revision must pin the reviewed loader as a 40-character hexadecimal revision."
        )
    import importlib

    if request.strict_verify_repo:
        repository = str(Path(request.strict_verify_repo))
        if not Path(repository).is_dir():
            raise ConversionRefusal("--strict-verify-repo %s is not a directory." % repository)
        if repository not in sys.path:
            sys.path.insert(0, repository)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ConversionRefusal(
            "Reviewed loader module %r is unavailable in this environment; install the policy's own image."
            % module_name
        ) from error
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ConversionRefusal("Reviewed loader %s:%s is not callable." % (module_name, attribute))
    model = factory()
    loader = getattr(model, "load_state_dict", None)
    if not callable(loader):
        raise ConversionRefusal("Reviewed loader %s:%s did not return a module with load_state_dict." % (module_name, attribute))
    try:
        result = loader(dict(tensors), strict=True)
    except RuntimeError as error:
        raise ConversionRefusal("strict=True state-dict load failed: %s" % error) from error
    missing = [str(key) for key in (getattr(result, "missing_keys", ()) or ())]
    unexpected = [str(key) for key in (getattr(result, "unexpected_keys", ()) or ())]
    if missing or unexpected:
        raise ConversionRefusal(
            "strict=True state-dict load reported missing %s and unexpected %s keys."
            % (", ".join(missing) or "<none>", ", ".join(unexpected) or "<none>")
        )
    return {
        "performed": True,
        "entry_point": "%s:%s" % (module_name, attribute),
        "loader_revision": request.strict_verify_loader_revision,
        "missing_keys": [],
        "unexpected_keys": [],
        "note": "strict=True reported full coverage. That checks keys, not provenance or deserialization safety.",
    }


# -- exclusive writes ----------------------------------------------------------


def reserve_exclusive(path: Path) -> None:
    """Claim ``path`` with ``O_EXCL``, refusing to overwrite anything."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ConversionRefusal(
            "Refusing to overwrite the existing artifact at %s. Inspect its provenance and remove it deliberately "
            "if it is genuinely stale." % path
        ) from None
    os.close(handle)


def release_reservation(path: Path) -> None:
    """Drop a reservation this run created, leaving real files untouched."""

    try:
        if path.is_file() and path.stat().st_size == 0:
            path.unlink()
    except OSError:  # pragma: no cover - best-effort cleanup
        pass


def write_bytes_exclusively(path: Path, payload: bytes) -> None:
    """Fill a reserved path atomically through a sibling temporary file."""

    handle, temporary_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".partial")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        release_temporary(temporary)
        raise


def release_temporary(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:  # pragma: no cover - best-effort cleanup
        pass


def flush_to_disk(path: Path) -> None:
    """``fsync`` a file another library wrote, so the measured hash is durable."""

    try:
        handle = os.open(str(path), os.O_RDONLY)
    except OSError:  # pragma: no cover - the caller measures the file next anyway
        return
    try:
        os.fsync(handle)
    except OSError:  # pragma: no cover - not every filesystem supports it
        pass
    finally:
        os.close(handle)


# -- artifact ------------------------------------------------------------------


@dataclass(frozen=True)
class ConvertedArtifact:
    """The safetensors artifact as written and measured."""

    path: str
    sha256: str
    bytes: int
    tensor_count: int
    total_tensor_bytes: int
    dtype_summary: Mapping[str, int]
    state_dict_key: Optional[str]
    ignored_top_level_keys: Tuple[str, ...]
    metadata: Mapping[str, str]

    def payload(self) -> Dict[str, Any]:
        return {
            "format": "safetensors",
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "tensor_count": self.tensor_count,
            "total_tensor_bytes": self.total_tensor_bytes,
            "dtype_summary": dict(self.dtype_summary),
            "state_dict_key": self.state_dict_key,
            "ignored_top_level_keys": list(self.ignored_top_level_keys),
            "flatten_separator": FLATTEN_SEPARATOR,
            "metadata": dict(self.metadata),
        }


def artifact_metadata(
    request: ConversionRequest,
    entry: AssetPlanEntry,
    planned: PlannedFile,
    verification: SourceVerification,
    revision: Optional[str],
    state_dict_key: Optional[str],
    tensor_count: int,
) -> Dict[str, str]:
    """String-only safetensors metadata; a license is never invented here."""

    metadata = {
        "plumb_conversion_schema_version": str(SCHEMA_VERSION),
        "plumb_conversion_kind": REPORT_KIND,
        "policy": request.target.policy,
        "asset_plan_name": entry.name,
        "source_repo_id": entry.repo_id,
        "source_repo_path": planned.path,
        "source_revision": revision or "unresolved",
        "source_sha256": verification.sha256,
        "state_dict_key": state_dict_key or "<root>",
        "flatten_separator": FLATTEN_SEPARATOR,
        "tensor_count": str(tensor_count),
        "license_status": entry.license_status.value,
        "redistribution": entry.redistribution.value,
        "security_note": SECURITY_NOTE,
    }
    if entry.license is not None:
        metadata["license"] = entry.license
    else:
        metadata["license_note"] = (
            "Upstream declares no license for %s. PLUMB records none rather than inventing one, and does not "
            "redistribute these bytes." % entry.repo_id
        )
    return metadata


# -- adapter handoff -----------------------------------------------------------


def validate_adapter_expectations(request: ConversionRequest, entry: AssetPlanEntry) -> None:
    """Refuse before converting if the plan would produce an unusable artifact.

    The MiniVLA adapter accepts a closed set of license statuses and insists that
    an unresolved license be paired with prohibited redistribution, so a plan that
    disagrees would waste a multi-gigabyte conversion.
    """

    if request.target.adapter_fields_kind != "minivla_converted_artifact":
        return
    status = entry.license_status.value
    if status not in ADAPTER_KNOWN_LICENSE_STATUSES:
        raise ConversionRefusal(
            "Asset plan records license_status %r for %s, which the MiniVLA adapter does not accept (it accepts %s). "
            "Resolve the plan before converting." % (status, entry.name, ", ".join(ADAPTER_KNOWN_LICENSE_STATUSES))
        )
    redistribution = entry.redistribution.value
    if (entry.license is None or status == "unresolved") and redistribution != ADAPTER_PROHIBITED_REDISTRIBUTION:
        raise ConversionRefusal(
            "Asset plan records redistribution %r for %s while its license is unresolved; the MiniVLA adapter "
            "requires %r. Resolve the plan before converting."
            % (redistribution, entry.name, ADAPTER_PROHIBITED_REDISTRIBUTION)
        )


def adapter_profile_fields(
    request: ConversionRequest,
    entry: AssetPlanEntry,
    artifact: ConvertedArtifact,
    verification: SourceVerification,
    report_path: Path,
    report_sha256: str,
    coverage_verified: bool,
) -> Dict[str, Any]:
    """Exactly the profile fields the consuming policy adapter requires.

    ``minivla``/``minivla_vq`` produce ``plumb.policies.minivla.ConvertedWeightArtifact``
    keyword arguments; ``openpizero`` produces the converted-checkpoint fields of
    ``plumb.policies.openpizero.OpenPiZeroPolicyProfile``.
    """

    if request.target.adapter_fields_kind == "minivla_converted_artifact":
        return {
            "label": request.target.artifact_label,
            "path": artifact.path,
            "sha256": artifact.sha256,
            "source_pickle_sha256": verification.sha256,
            "conversion_report_path": str(report_path),
            "conversion_report_sha256": report_sha256,
            "license": entry.license,
            "license_status": entry.license_status.value,
            "redistribution": entry.redistribution.value,
        }
    return {
        "converted_checkpoint_path": artifact.path,
        "converted_checkpoint_sha256": artifact.sha256,
        "source_pickle_sha256": verification.sha256,
        "conversion_report_sha256": report_sha256,
        "state_dict_coverage_verified": coverage_verified,
    }


# -- report --------------------------------------------------------------------


def base_report(request: ConversionRequest, runtime: Optional[ConverterRuntime]) -> Dict[str, Any]:
    """A complete report skeleton whose status starts at ``failed``."""

    entry = get_plan(request.target.asset_plan_name)
    planned = planned_file(entry)
    module = Path(__file__).resolve()
    revision = request.source_revision or entry.revision
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": "failed",
        "mode": request.mode,
        "policy": request.target.policy,
        "converter": {
            "path": "cluster/" + module.name,
            "source_revision": converter_source_revision(module),
            "source_sha256": sha256_of(module),
            "python_version": platform.python_version(),
            "torch_version": None if runtime is None else runtime.torch_version,
            "safetensors_version": None if runtime is None else runtime.safetensors_version,
        },
        "source": {
            "asset_plan_name": entry.name,
            "repo_id": entry.repo_id,
            "repo_type": entry.repo_type.value,
            "repo_path": planned.path,
            "url_path": planned.url_path,
            "source_url": entry.source_url,
            "download_url": entry.direct_download_url(planned) if entry.revision is not None else None,
            "revision": revision,
            "revision_origin": (
                "operator" if request.source_revision else ("asset_plan" if entry.revision else None)
            ),
            "revision_pinned_in_asset_plan": entry.revision is not None,
            "sha256_pinned_in_asset_plan": planned.sha256 is not None,
            "expected_bytes": planned.expected_bytes,
            "historical_estimate_bytes": planned.historical_estimate_bytes,
            "local_path": str(request.source),
            "sha256": None,
            "bytes": None,
            "expected_sha256": None,
            "expected_sha256_origin": None,
            "sha256_verified": False,
            "basename_matches_asset_plan": None,
        },
        "license": {
            "license": entry.license,
            "license_status": entry.license_status.value,
            "redistribution": entry.redistribution.value,
            "acknowledgement_required": license_acknowledgement_required(entry),
            "acknowledged_by_operator": bool(request.acknowledge_unresolved_license),
            "notices": list(entry.traps) + list(entry.notes),
        },
        "deserialization": {
            "method": "torch.load(weights_only=True) inside torch.serialization.safe_globals",
            "weights_only": True,
            "unsafe_fallback_available": False,
            "globals_enumerated_before_load": False,
            "expectation": request.target.expectation,
            "observed_globals": None,
            "reviewed_globals": list(request.target.reviewed_globals),
            "operator_approved_globals": list(request.operator_approved_globals),
            "operator_reviewer": request.reviewer,
            "approved_globals": None,
            "approved_globals_used": None,
            "rejected_globals": None,
            "never_approvable_globals_observed": None,
            "inert_aliased_globals": None,
        },
        "output": None,
        "strict_state_dict_verification": {
            "performed": False,
            "entry_point": None,
            "loader_revision": None,
            "missing_keys": [],
            "unexpected_keys": [],
            "note": STRICT_VERIFICATION_NOTE,
        },
        "isolation": None,
        "verification_blockers": [],
        "security_note": SECURITY_NOTE,
        "error": None,
    }


def verification_blockers(report: Mapping[str, Any]) -> List[str]:
    """Everything this report does not establish, named."""

    blockers: List[str] = [NETWORK_ISOLATION_BLOCKER]
    source = report["source"]
    if not source.get("revision"):
        blockers.append(UNRESOLVED_REVISION_BLOCKER)
    if not source.get("sha256_pinned_in_asset_plan"):
        blockers.append(MISSING_HASH_BLOCKER)
    if not source.get("sha256_verified"):
        blockers.append(UNVERIFIED_SOURCE_SHA_BLOCKER)
    if source.get("basename_matches_asset_plan") is False:
        blockers.append(SOURCE_NAME_BLOCKER)
    license_block = report["license"]
    if license_block.get("license") is None or license_block.get("license_status") in (
        "unresolved",
        "absent_cardData_null",
        "gated_terms_required",
    ):
        blockers.append(UNRESOLVED_LICENSE_BLOCKER)
    if license_block.get("acknowledgement_required") and not license_block.get("acknowledged_by_operator"):
        blockers.append(LICENSE_ACKNOWLEDGEMENT_BLOCKER)
    if not report["strict_state_dict_verification"].get("performed"):
        blockers.append(STATE_DICT_COVERAGE_BLOCKER)
    return sorted(dict.fromkeys(blockers))


# -- the conversion ------------------------------------------------------------


def run(
    request: ConversionRequest,
    *,
    runtime_factory: Callable[[], ConverterRuntime] = default_runtime,
    environment: Optional[Environment] = None,
) -> Dict[str, Any]:
    """Inspect or convert one checkpoint and return the operator summary.

    The returned mapping carries the report, the report's own SHA-256, and the
    policy-profile fields the consuming adapter needs.  The report file cannot
    contain its own hash, so the ordering is artifact, then report, then report
    hash.
    """

    environment = environment or Environment.from_process()
    entry = get_plan(request.target.asset_plan_name)
    planned = planned_file(entry)
    runtime: Optional[ConverterRuntime] = None
    report = base_report(request, None)
    reserved: List[Path] = []
    # Only a reserved report path receives refusal evidence; a dry run reserves
    # nothing and must therefore write nothing, even when it refuses.
    evidence: Optional[Path] = None
    summary: Dict[str, Any] = {
        "policy": request.target.policy,
        "report_path": None,
        "conversion_report_sha256": None,
        "policy_profile_fields": None,
    }
    try:
        isolation = check_isolation(environment)
        report["isolation"] = isolation.payload()
        if isolation.refusals:
            raise ConversionRefusal(
                "Refusing to run outside an isolated environment. " + " ".join(isolation.refusals)
            )
        revision = report["source"]["revision"]
        if request.source_revision is not None and not is_immutable_revision(request.source_revision):
            raise ConversionRefusal("--source-revision must be a 40-character hexadecimal commit.")
        if request.converts:
            validate_adapter_expectations(request, entry)
            if revision is None:
                raise ConversionRefusal(
                    "Neither cluster/asset_plan.py nor --source-revision pins an immutable revision for %r "
                    "(blocker %r). A converted artifact with no source revision is not traceable; supply the commit "
                    "you downloaded from." % (entry.name, UNRESOLVED_REVISION_BLOCKER)
                )
            if license_acknowledgement_required(entry) and not request.acknowledge_unresolved_license:
                raise ConversionRefusal(
                    "%s records license %r with status %r and redistribution %r. Converting it requires "
                    "--acknowledge-unresolved-license: local use may proceed, mirroring may not, and this tool will "
                    "not write a license string it cannot substantiate."
                    % (
                        entry.repo_id,
                        entry.license,
                        entry.license_status.value,
                        entry.redistribution.value,
                    )
                )
            if request.output is None:
                raise ConversionRefusal("--output is required unless --inspect is used.")
            if request.output.suffix.lower() != OUTPUT_SUFFIX:
                raise ConversionRefusal(
                    "--output must end in %s; the policy adapters refuse every other container, legacy pickles "
                    "most of all." % OUTPUT_SUFFIX
                )
            report_path = request.report_path
            if report_path is None or report_path.suffix.lower() != ".json":
                raise ConversionRefusal("--report must be a .json path.")
            summary["report_path"] = str(report_path)
            if request.mode == "execute":
                reserve_exclusive(request.output)
                reserved.append(request.output)
                reserve_exclusive(report_path)
                reserved.append(report_path)
                evidence = report_path
            else:
                for candidate in (request.output, report_path):
                    if candidate.exists():
                        raise ConversionRefusal(
                            "Refusing to overwrite the existing artifact at %s." % candidate
                        )
        elif request.report is not None:
            # An inspection is the security review, so keep its evidence when the
            # operator names a path for it.
            if request.report.suffix.lower() != ".json":
                raise ConversionRefusal("--report must be a .json path.")
            reserve_exclusive(request.report)
            reserved.append(request.report)
            summary["report_path"] = str(request.report)
            evidence = request.report

        runtime = runtime_factory()
        report["converter"]["torch_version"] = runtime.torch_version
        report["converter"]["safetensors_version"] = runtime.safetensors_version

        verification = verify_source(request, planned)
        report["source"].update(verification.payload())

        decision = enumerate_globals(runtime, request)
        report["deserialization"] = decision.payload()
        report["deserialization"]["expectation"] = request.target.expectation

        if request.mode == "inspect":
            report["status"] = "inspected"
            return _finish(report, summary, reserved, write_report=summary["report_path"] is not None)

        refusal = decision.refusal(request.target.policy)
        if refusal is not None:
            raise ConversionRefusal(refusal)

        if request.mode == "dry_run":
            report["status"] = "dry_run"
            return _finish(report, summary, reserved, write_report=False)

        with runtime.torch.serialization.safe_globals(safe_alias_pairs(decision)):
            loaded = runtime.torch.load(str(request.source), map_location="cpu", weights_only=True)
        subtree, state_dict_key, ignored = select_state_dict(runtime, loaded, request)
        tensors, problems = flatten_tensors(runtime, subtree)
        if problems:
            raise ConversionRefusal(
                "Refusing to write a partial artifact: the selected state dict has non-tensor leaves: %s. A "
                "safetensors file holds tensors only and this tool will not invent one." % "; ".join(sorted(problems))
            )
        if not tensors:
            raise ConversionRefusal("The selected state dict is empty; refusing to write an empty artifact.")
        report["strict_state_dict_verification"] = strict_state_dict_verification(request, tensors)

        total_tensor_bytes, dtypes = tensor_summary(tensors)
        metadata = artifact_metadata(
            request, entry, planned, verification, revision, state_dict_key, len(tensors)
        )
        output = request.output
        if output is None:  # pragma: no cover - guarded above for every converting mode
            raise ConversionRefusal("--output is required unless --inspect is used.")
        handle, temporary_name = tempfile.mkstemp(
            dir=str(output.parent), prefix=output.name + ".", suffix=".partial"
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            runtime.save_file(
                {name: tensor.detach().cpu().contiguous() for name, tensor in sorted(tensors.items())},
                str(temporary),
                metadata=metadata,
            )
            flush_to_disk(temporary)
            os.replace(str(temporary), str(output))
        except BaseException:
            release_temporary(temporary)
            raise
        artifact = ConvertedArtifact(
            path=str(output),
            sha256=sha256_of(output),
            bytes=output.stat().st_size,
            tensor_count=len(tensors),
            total_tensor_bytes=total_tensor_bytes,
            dtype_summary=dtypes,
            state_dict_key=state_dict_key,
            ignored_top_level_keys=ignored,
            metadata=metadata,
        )
        report["output"] = artifact.payload()
        report["status"] = "completed"
        _finish(report, summary, reserved, write_report=True)
        summary["policy_profile_fields"] = adapter_profile_fields(
            request,
            entry,
            artifact,
            verification,
            Path(str(summary["report_path"])),
            str(summary["conversion_report_sha256"]),
            bool(report["strict_state_dict_verification"]["performed"]),
        )
        return summary
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["verification_blockers"] = verification_blockers(report)
        for path in reserved:
            release_reservation(path)
        if evidence is not None:
            refused = evidence.with_name(evidence.stem + ".refused-" + uuid.uuid4().hex[:8] + evidence.suffix)
            try:
                write_bytes_exclusively(refused, _encode(report))
                summary["refusal_report_path"] = str(refused)
            except OSError:  # pragma: no cover - evidence is best-effort on a full disk
                summary["refusal_report_path"] = None
        summary["report"] = report
        return summary


def _encode(report: Mapping[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _finish(
    report: Dict[str, Any],
    summary: Dict[str, Any],
    reserved: Sequence[Path],
    *,
    write_report: bool,
) -> Dict[str, Any]:
    """Seal the report, write it when asked, and record its own SHA-256."""

    report["verification_blockers"] = verification_blockers(report)
    payload = _encode(report)
    if write_report:
        report_path = Path(str(summary["report_path"]))
        write_bytes_exclusively(report_path, payload)
        summary["conversion_report_sha256"] = sha256_of(report_path)
    else:
        for path in reserved:
            release_reservation(path)
        summary["conversion_report_sha256"] = hashlib.sha256(payload).hexdigest()
    summary["report"] = report
    return summary


# -- command line --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__ or "", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=policy_names(), help="Which reviewed policy checkpoint to convert")
    parser.add_argument("--source", type=Path, help="Local path to the downloaded legacy checkpoint")
    parser.add_argument("--output", type=Path, help="New %s path; never overwritten" % OUTPUT_SUFFIX)
    parser.add_argument("--report", type=Path, help="Conversion report path (default: <output>.conversion-report.json)")
    parser.add_argument("--execute", action="store_true", help="Actually convert; dry-run is the default")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Enumerate the pickle's globals and report, converting nothing",
    )
    parser.add_argument("--list-policies", action="store_true", help="Print the convertible policies and exit")
    parser.add_argument(
        "--acknowledge-unresolved-license",
        action="store_true",
        help="Required for assets whose license or access terms are unresolved upstream",
    )
    parser.add_argument("--expected-source-sha256", help="Required when the asset plan pins no SHA-256")
    parser.add_argument("--source-revision", help="The immutable 40-character commit the source was downloaded from")
    parser.add_argument(
        "--approve-global",
        action="append",
        default=[],
        metavar="MODULE.NAME",
        help="Add one reviewed pickle global to the allowlist; requires --reviewer",
    )
    parser.add_argument("--reviewer", help="Who reviewed the globals named by --approve-global")
    parser.add_argument("--state-dict-key", help="Explicit top-level key holding the state dict")
    parser.add_argument("--strict-verify", metavar="MODULE:ATTRIBUTE", help="Reviewed model factory for strict=True")
    parser.add_argument("--strict-verify-repo", help="Pinned loader repository to place on sys.path")
    parser.add_argument("--strict-verify-loader-revision", help="Reviewed loader commit, required with --strict-verify")
    return parser


def parse_request(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> ConversionRequest:
    """Validate the flag combination and freeze it into a request."""

    if arguments.inspect and arguments.execute:
        parser.error("--inspect converts nothing; do not combine it with --execute")
    if not arguments.policy:
        parser.error("--policy is required (or use --list-policies)")
    if not arguments.source:
        parser.error("--source is required")
    mode = "inspect" if arguments.inspect else ("execute" if arguments.execute else "dry_run")
    return ConversionRequest(
        target=get_target(arguments.policy),
        source=Path(arguments.source),
        mode=mode,
        output=Path(arguments.output) if arguments.output else None,
        report=Path(arguments.report) if arguments.report else None,
        expected_source_sha256=arguments.expected_source_sha256,
        source_revision=arguments.source_revision,
        acknowledge_unresolved_license=bool(arguments.acknowledge_unresolved_license),
        operator_approved_globals=tuple(arguments.approve_global or ()),
        reviewer=arguments.reviewer,
        state_dict_key=arguments.state_dict_key,
        strict_verify=arguments.strict_verify,
        strict_verify_repo=arguments.strict_verify_repo,
        strict_verify_loader_revision=arguments.strict_verify_loader_revision,
    )


def policy_catalogue() -> Dict[str, Any]:
    """The convertible policies, their pinned plan data and their reviews."""

    return {
        "schema_version": SCHEMA_VERSION,
        "output_suffix": OUTPUT_SUFFIX,
        "security_note": SECURITY_NOTE,
        "policies": [target.payload() for target in POLICY_TARGETS],
    }


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    runtime_factory: Callable[[], ConverterRuntime] = default_runtime,
    environment: Optional[Environment] = None,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.list_policies:
        print(json.dumps(policy_catalogue(), indent=2, sort_keys=True))
        return 0
    request = parse_request(parser, arguments)
    summary = run(request, runtime_factory=runtime_factory, environment=environment)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["report"]["status"] in ("completed", "dry_run", "inspected") else 2


if __name__ == "__main__":
    raise SystemExit(main())
