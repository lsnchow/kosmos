"""World-model adapters with explicit backend-specific action contracts."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .contracts import (
    BackendProfile,
    CapabilityResult,
    CapabilityStatus,
    COSMOS3_DIFFUSERS_SOURCE,
    IRASIM_COMMIT,
    ServerTiming,
    WorldRequest,
    WorldResult,
)


DIFFUSERS_COSMOS3_PIN = "a3e0b8ec235c27a6c17a21976daf7fd32d819d05"
COSMOS3_NANO_MODEL_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
COSMOS3_VENDOR_FIXTURE = (
    "https://raw.githubusercontent.com/NVIDIA/cosmos-framework/"
    "c23e51f2f157ae3e51cfcd86ebfb5464850894f2/inputs/omni/action_forward_dynamics_robot.json"
)
COSMOS3_BRIDGE_FIXTURE_ASSET_COMMIT = "2b17a2413bd86b2cf9b03823637108851e4ddf2d"
COSMOS3_BRIDGE_FIXTURE_ACTION_URL = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/"
    + COSMOS3_BRIDGE_FIXTURE_ASSET_COMMIT
    + "/inputs/action/bridge_20260501_0.json"
)
COSMOS3_BRIDGE_FIXTURE_VIDEO_URL = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/"
    + COSMOS3_BRIDGE_FIXTURE_ASSET_COMMIT
    + "/inputs/action/bridge_20260501_0.mp4"
)
# Hash of the small JSON action fixture, not the model checkpoint or MP4.
COSMOS3_BRIDGE_FIXTURE_ACTION_SHA256 = "5c26b3cb84799812a70b534ad939551d2ac308fdc870ea0e66163bb52c9d61da"


class BackendContractError(ValueError):
    """The request is not valid for the selected native backend."""


class BackendUnavailableError(RuntimeError):
    """A local model/dependency/loader required for a real call is missing."""


class CertificationError(ValueError):
    """A certification artifact is absent, malformed, or lacks real evidence."""


#: The only action length the reviewed source establishes for this Bridge FD
#: profile.  Spec section 12 records that the asserted universal
#: multiple-of-four restriction is *not* established by the reviewed source, so
#: longer or shorter lengths are neither assumed supported nor assumed
#: forbidden: they must be probed and certified.
UNCERTIFIED_COSMOS_ACTION_LENGTHS: Tuple[int, ...] = (16,)
UNCERTIFIED_DEFAULT_CLASS = "uncertified_default"
GATE_A_CERTIFIED_CLASS = "gate_a_certified"
ACTION_LENGTH_CERTIFICATION_KIND = "plumb_world_action_length_certification"

#: Terminal padding is the spec's sanctioned fallback *only* with measured
#: prefix-invariance evidence (section 2).  A padded run carries this protocol
#: identity and can never be reported as an exact horizon match.
PROTOCOL_EXACT_TERMINAL_HORIZON = "exact_terminal_horizon"
PROTOCOL_CERTIFIED_TERMINAL_PADDING = "certified_terminal_padding"

PADDING_POLICY_ZERO_DELTA_HOLD_GRIPPER = "zero_delta_hold_gripper"
PADDING_POLICY_REPEAT_LAST_ACTION = "repeat_last_action"
PADDING_POLICIES: Tuple[str, ...] = (
    PADDING_POLICY_ZERO_DELTA_HOLD_GRIPPER,
    PADDING_POLICY_REPEAT_LAST_ACTION,
)


def _normalised_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text.lower()):
        raise CertificationError("%s must be a SHA-256 digest, got %r" % (label, value))
    return "sha256:" + text.lower()


@dataclass(frozen=True)
class ProbedActionLength:
    """One probed action length plus the evidence that it was actually run.

    ``status="supported"`` additionally requires the observed frame count to
    satisfy the backend's own ``N actions -> N+1 frames`` contract.  A probe
    that returned only the conditioning frame is therefore recorded as
    ``unsupported`` with its real observation rather than rounded up.
    """

    action_length: int
    status: str
    evidence_uri: str
    evidence_sha256: str
    returned_frame_count: Optional[int] = None
    job_id: Optional[str] = None
    observed_at: Optional[str] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.action_length, bool) or not isinstance(self.action_length, int) or self.action_length < 1:
            raise CertificationError("probed action_length must be a positive integer")
        if self.status not in ("supported", "unsupported"):
            raise CertificationError("probed action length status must be supported or unsupported")
        if not self.evidence_uri:
            raise CertificationError(
                "action length %d needs an evidence URI; a certification without per-length evidence is a guess"
                % self.action_length
            )
        object.__setattr__(self, "evidence_sha256", _normalised_sha256(self.evidence_sha256, "evidence_sha256"))
        if self.status == "supported":
            if self.returned_frame_count is None:
                raise CertificationError(
                    "supported action length %d must record the observed returned frame count" % self.action_length
                )
            if int(self.returned_frame_count) != self.action_length + 1:
                raise CertificationError(
                    "action length %d returned %s frames, which does not satisfy the N -> N+1 frame contract; "
                    "record it as unsupported instead of certifying it"
                    % (self.action_length, self.returned_frame_count)
                )

    @property
    def expected_frame_count(self) -> int:
        return self.action_length + 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action_length": int(self.action_length),
            "status": self.status,
            "expected_frame_count": self.expected_frame_count,
            "returned_frame_count": None if self.returned_frame_count is None else int(self.returned_frame_count),
            "evidence_uri": self.evidence_uri,
            "evidence_sha256": self.evidence_sha256,
            "job_id": self.job_id,
            "observed_at": self.observed_at,
            "note": self.note,
        }


@dataclass(frozen=True)
class ActionLengthCertification:
    """Gate-A evidence for which action lengths a deployment actually runs.

    ``supported_lengths`` replaces a hardcoded guess.  The certification binds
    itself to one ``profile_id`` so evidence from a different resolution tier,
    scheduler, or container cannot be reused silently.
    """

    certification_id: str
    profile_id: str
    backend: str
    domain: str
    lengths: Tuple[ProbedActionLength, ...]
    source_uri: str
    source_sha256: str
    control_hz: float = 5.0
    certification_class: str = GATE_A_CERTIFIED_CLASS
    recorded_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.certification_id or not self.profile_id or not self.backend or not self.domain:
            raise CertificationError("certification needs certification_id, profile_id, backend, and domain")
        if not self.lengths:
            raise CertificationError("certification needs at least one probed action length")
        seen = [item.action_length for item in self.lengths]
        if len(seen) != len(set(seen)):
            raise CertificationError("certification lists a duplicate action length")
        if not self.supported_lengths:
            raise CertificationError(
                "certification records no supported action length; it cannot populate a world action profile"
            )
        if self.certification_class != GATE_A_CERTIFIED_CLASS:
            raise CertificationError("certification_class must be %r" % GATE_A_CERTIFIED_CLASS)
        object.__setattr__(self, "source_sha256", _normalised_sha256(self.source_sha256, "source_sha256"))
        if float(self.control_hz) <= 0:
            raise CertificationError("control_hz must be positive")

    @property
    def supported_lengths(self) -> Tuple[int, ...]:
        return tuple(sorted(item.action_length for item in self.lengths if item.status == "supported"))

    @property
    def unsupported_lengths(self) -> Tuple[int, ...]:
        return tuple(sorted(item.action_length for item in self.lengths if item.status == "unsupported"))

    @property
    def source_hash(self) -> str:
        return self.source_sha256

    def as_dict(self) -> Dict[str, Any]:
        return {
            "certification_id": self.certification_id,
            "kind": ACTION_LENGTH_CERTIFICATION_KIND,
            "profile_id": self.profile_id,
            "backend": self.backend,
            "domain": self.domain,
            "control_hz": float(self.control_hz),
            "certification_class": self.certification_class,
            "source_uri": self.source_uri,
            "source_sha256": self.source_sha256,
            "recorded_at": self.recorded_at,
            "supported_action_lengths": list(self.supported_lengths),
            "unsupported_action_lengths": list(self.unsupported_lengths),
            "lengths": [item.as_dict() for item in self.lengths],
        }


def certified_action_lengths(path: Any, *, profile_id: Optional[str] = None) -> ActionLengthCertification:
    """Load a Gate-A action-length certification artifact from disk.

    The artifact's own bytes are hashed here, so ``source_sha256`` identifies
    exactly the file that was read.  Any missing per-length evidence raises:
    there is no partial-credit path that yields an uncertified-but-trusted
    length set.
    """

    candidate = Path(str(path))
    try:
        raw = candidate.read_bytes()
    except OSError as error:
        raise CertificationError("cannot read action-length certification at %s: %s" % (candidate, error)) from error
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CertificationError("action-length certification at %s is not valid JSON: %s" % (candidate, error)) from error
    if not isinstance(payload, Mapping):
        raise CertificationError("action-length certification must be a JSON object")
    if payload.get("kind") != ACTION_LENGTH_CERTIFICATION_KIND:
        raise CertificationError(
            "expected kind=%r in %s, got %r" % (ACTION_LENGTH_CERTIFICATION_KIND, candidate, payload.get("kind"))
        )
    entries = payload.get("lengths")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise CertificationError("action-length certification needs a nonempty 'lengths' array")
    lengths: List[ProbedActionLength] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise CertificationError("every 'lengths' entry must be a JSON object")
        lengths.append(
            ProbedActionLength(
                action_length=int(entry["action_length"]),
                status=str(entry["status"]),
                evidence_uri=str(entry.get("evidence_uri") or ""),
                evidence_sha256=entry.get("evidence_sha256"),
                returned_frame_count=(
                    None if entry.get("returned_frame_count") is None else int(entry["returned_frame_count"])
                ),
                job_id=None if entry.get("job_id") is None else str(entry["job_id"]),
                observed_at=None if entry.get("observed_at") is None else str(entry["observed_at"]),
                note=None if entry.get("note") is None else str(entry["note"]),
            )
        )
    certification = ActionLengthCertification(
        certification_id=str(payload.get("certification_id") or ""),
        profile_id=str(payload.get("profile_id") or ""),
        backend=str(payload.get("backend") or ""),
        domain=str(payload.get("domain") or ""),
        lengths=tuple(lengths),
        source_uri=str(payload.get("source_uri") or candidate.as_posix()),
        source_sha256=digest,
        control_hz=float(payload.get("control_hz", 5.0)),
        recorded_at=None if payload.get("recorded_at") is None else str(payload["recorded_at"]),
    )
    if profile_id is not None and certification.profile_id != profile_id:
        raise CertificationError(
            "certification %r was recorded for profile %r, not %r; requalify instead of reusing it"
            % (certification.certification_id, certification.profile_id, profile_id)
        )
    return certification


@dataclass(frozen=True)
class TerminalPaddingCertificate:
    """Measured prefix-invariance evidence authorising terminal padding.

    Spec section 2: "For terminal padding, demonstrate prefix invariance to
    post-horizon actions; otherwise require an exact supported terminal
    length."  This type carries that measurement.  ``errors()`` is nonempty
    whenever the evidence does not actually support padding, and a controller
    must refuse to pad in that case.

    ``tolerance_frame_mae`` is supplied by the caller, normally from
    ``plumb.protocol.Tolerances.max_suffix_invariance_mae``; ``tolerance_source``
    records where the number came from so a relaxed threshold is visible.
    """

    certificate_id: str
    profile_id: str
    padded_action_length: int
    certified_prefix_lengths: Tuple[int, ...]
    prefix_invariance_trials: int
    max_observed_prefix_frame_mae: float
    tolerance_frame_mae: float
    evidence_uri: str
    evidence_sha256: str
    seed_matched: bool = True
    padding_action_policy: str = PADDING_POLICY_ZERO_DELTA_HOLD_GRIPPER
    tolerance_source: str = "caller_supplied"
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.padding_action_policy not in PADDING_POLICIES:
            raise CertificationError("padding_action_policy must be one of %s" % (PADDING_POLICIES,))
        if self.evidence_sha256:
            object.__setattr__(self, "evidence_sha256", _normalised_sha256(self.evidence_sha256, "evidence_sha256"))

    @property
    def protocol_identity(self) -> str:
        return PROTOCOL_CERTIFIED_TERMINAL_PADDING

    @property
    def certificate_hash(self) -> str:
        payload = {
            "certificate_id": self.certificate_id,
            "profile_id": self.profile_id,
            "padded_action_length": int(self.padded_action_length),
            "certified_prefix_lengths": list(self.certified_prefix_lengths),
            "prefix_invariance_trials": int(self.prefix_invariance_trials),
            "max_observed_prefix_frame_mae": float(self.max_observed_prefix_frame_mae),
            "tolerance_frame_mae": float(self.tolerance_frame_mae),
            "tolerance_source": self.tolerance_source,
            "seed_matched": bool(self.seed_matched),
            "padding_action_policy": self.padding_action_policy,
            "evidence_uri": self.evidence_uri,
            "evidence_sha256": self.evidence_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def errors(self) -> Tuple[str, ...]:
        """Why this certificate may not authorise padding. Empty means it may."""

        problems: List[str] = []
        if not self.certificate_id:
            problems.append("certificate_id is required")
        if not self.profile_id:
            problems.append("profile_id is required; padding evidence is backend-profile specific")
        if isinstance(self.padded_action_length, bool) or not isinstance(self.padded_action_length, int) or self.padded_action_length < 2:
            problems.append("padded_action_length must be an integer of at least 2")
        if not self.certified_prefix_lengths:
            problems.append("no consumed prefix length has demonstrated prefix invariance")
        for prefix in self.certified_prefix_lengths:
            if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 1:
                problems.append("certified prefix %r must be a positive integer" % (prefix,))
            elif isinstance(self.padded_action_length, int) and prefix >= self.padded_action_length:
                problems.append(
                    "certified prefix %d is not shorter than the padded length %d, so nothing was padded"
                    % (prefix, self.padded_action_length)
                )
        if int(self.prefix_invariance_trials or 0) < 1:
            problems.append("prefix invariance was never measured (zero trials)")
        if self.max_observed_prefix_frame_mae is None:
            problems.append("max_observed_prefix_frame_mae is null; an unmeasured residual is not invariance")
        if float(self.tolerance_frame_mae or 0.0) <= 0.0:
            problems.append("tolerance_frame_mae must be an explicit positive preregistered tolerance")
        elif float(self.max_observed_prefix_frame_mae) > float(self.tolerance_frame_mae):
            problems.append(
                "measured prefix residual %.6f exceeds the preregistered tolerance %.6f"
                % (float(self.max_observed_prefix_frame_mae), float(self.tolerance_frame_mae))
            )
        if not self.seed_matched:
            problems.append("prefix invariance must be measured at a matched seed and condition frame")
        if not self.evidence_uri or not self.evidence_sha256:
            problems.append("padding evidence needs a URI and SHA-256 digest")
        return tuple(problems)

    def certifies(self, *, padded_length: int, consumed_prefix: int, profile_id: Optional[str] = None) -> bool:
        if self.errors():
            return False
        if profile_id is not None and profile_id != self.profile_id:
            return False
        return int(padded_length) == int(self.padded_action_length) and int(consumed_prefix) in tuple(
            int(value) for value in self.certified_prefix_lengths
        )

    def padding_actions(self, consumed: Sequence[Sequence[float]], count: int) -> Tuple[Tuple[float, ...], ...]:
        """Build the declared post-horizon filler rows for a padded request.

        These rows are discarded output, not executed control.  The policy is
        whichever one the prefix-invariance evidence was measured under.
        """

        if count < 0:
            raise CertificationError("padding count cannot be negative")
        if not consumed:
            raise CertificationError("padding needs at least one consumed action to derive its declared filler")
        last = tuple(float(value) for value in consumed[-1])
        if self.padding_action_policy == PADDING_POLICY_REPEAT_LAST_ACTION:
            return tuple(last for _ in range(count))
        held = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, last[6] if len(last) >= 7 else 0.0)
        return tuple(held for _ in range(count))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "certificate_hash": self.certificate_hash,
            "profile_id": self.profile_id,
            "protocol_identity": self.protocol_identity,
            "padded_action_length": int(self.padded_action_length),
            "certified_prefix_lengths": [int(value) for value in self.certified_prefix_lengths],
            "prefix_invariance_trials": int(self.prefix_invariance_trials),
            "max_observed_prefix_frame_mae": (
                None if self.max_observed_prefix_frame_mae is None else float(self.max_observed_prefix_frame_mae)
            ),
            "tolerance_frame_mae": float(self.tolerance_frame_mae),
            "tolerance_source": self.tolerance_source,
            "seed_matched": bool(self.seed_matched),
            "padding_action_policy": self.padding_action_policy,
            "evidence_uri": self.evidence_uri,
            "evidence_sha256": self.evidence_sha256,
            "errors": list(self.errors()),
            "note": self.note,
        }


@dataclass(frozen=True)
class Cosmos3NanoDiffusersProfile:
    """Profile for one exact official Diffusers forward-dynamics serializer.

    It deliberately defaults to the local-only installation root.  The model
    identifier is metadata, not a signal to fetch 35 GB when a runtime starts.
    ``probe_action_lengths`` exists solely for Gate-B causal probes; it cannot
    turn a one-tick call into a qualified feedback implementation.

    ``allowed_action_lengths`` is *evidence*, not configuration.  Leaving
    ``action_length_certification`` unset keeps the single uncertified default
    length and reports ``action_length_certification_class="uncertified_default"``.
    Supplying a certification requires ``allowed_action_lengths`` to equal its
    certified supported lengths exactly, so a profile can never claim a length
    the deployment never demonstrated.
    """

    profile_id: str
    local_model_path: str
    model_revision: str = COSMOS3_NANO_MODEL_REVISION
    diffusers_revision: str = DIFFUSERS_COSMOS3_PIN
    container_digest: Optional[str] = None
    normalizer_revision: Optional[str] = None
    resolution_tier: int = 256
    fps: float = 5.0
    num_inference_steps: int = 30
    guidance_scale: float = 1.0
    scheduler_flow_shift: Optional[float] = 10.0
    device_map: str = "cuda"
    view_point: str = "ego_view"
    allowed_action_lengths: Tuple[int, ...] = UNCERTIFIED_COSMOS_ACTION_LENGTHS
    probe_action_lengths: Tuple[int, ...] = (1,)
    local_files_only: bool = True
    use_system_prompt: bool = False
    enable_safety_checker: bool = False
    action_length_certification: Optional[ActionLengthCertification] = None
    terminal_padding_certificate: Optional[TerminalPaddingCertificate] = None

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("Cosmos profile_id is required.")
        if not self.local_model_path:
            raise ValueError("Cosmos local_model_path is required; Hub fetching is intentionally disabled.")
        if self.resolution_tier not in (256, 480, 704, 720):
            raise ValueError("Cosmos action resolution_tier must be 256, 480, 704, or 720.")
        if self.fps <= 0 or self.num_inference_steps < 1:
            raise ValueError("Cosmos fps and num_inference_steps must be positive.")
        if not set(self.allowed_action_lengths).isdisjoint(set(self.probe_action_lengths)):
            raise ValueError("Normal and probe Cosmos action lengths must not overlap.")
        certification = self.action_length_certification
        if certification is not None:
            if certification.profile_id != self.profile_id:
                raise CertificationError(
                    "action-length certification %r belongs to profile %r, not %r"
                    % (certification.certification_id, certification.profile_id, self.profile_id)
                )
            if tuple(sorted(self.allowed_action_lengths)) != certification.supported_lengths:
                raise CertificationError(
                    "allowed_action_lengths %s does not match certified supported lengths %s; "
                    "use Cosmos3NanoDiffusersProfile.from_certification instead of asserting a length"
                    % (tuple(sorted(self.allowed_action_lengths)), certification.supported_lengths)
                )
        padding = self.terminal_padding_certificate
        if padding is not None and padding.profile_id != self.profile_id:
            raise CertificationError(
                "terminal padding certificate %r belongs to profile %r, not %r"
                % (padding.certificate_id, padding.profile_id, self.profile_id)
            )

    @classmethod
    def from_certification(
        cls, certification: ActionLengthCertification, **kwargs: Any
    ) -> "Cosmos3NanoDiffusersProfile":
        """Build a profile whose normal action lengths come from Gate-A evidence."""

        supported = certification.supported_lengths
        probe = tuple(
            length for length in kwargs.pop("probe_action_lengths", ()) if length not in supported
        )
        return cls(
            profile_id=kwargs.pop("profile_id", certification.profile_id),
            allowed_action_lengths=supported,
            probe_action_lengths=probe,
            action_length_certification=certification,
            fps=float(kwargs.pop("fps", certification.control_hz)),
            **kwargs,
        )

    @property
    def action_length_certification_class(self) -> str:
        return UNCERTIFIED_DEFAULT_CLASS if self.action_length_certification is None else GATE_A_CERTIFIED_CLASS

    @property
    def action_length_certification_source_hash(self) -> Optional[str]:
        """``None`` means uncertified, never an empty-but-trusted hash."""

        return None if self.action_length_certification is None else self.action_length_certification.source_hash

    def as_backend_profile(self) -> BackendProfile:
        return BackendProfile(
            profile_id=self.profile_id,
            backend="cosmos3_diffusers",
            model_id="nvidia/Cosmos3-Nano",
            model_revision=self.model_revision,
            code_revision=self.diffusers_revision,
            container_digest=self.container_digest,
            normalizer_revision=self.normalizer_revision,
            source_uri=COSMOS3_DIFFUSERS_SOURCE,
            local_model_path=self.local_model_path,
            local_files_only=self.local_files_only,
            metadata={
                "mode": "forward_dynamics",
                "domain_name": "bridge_orig_lerobot",
                "resolution_tier": self.resolution_tier,
                "fps": self.fps,
                "vendor_fixture": COSMOS3_VENDOR_FIXTURE,
                "vendor_fixture_action": {
                    "uri": COSMOS3_BRIDGE_FIXTURE_ACTION_URL,
                    "sha256": "sha256:" + COSMOS3_BRIDGE_FIXTURE_ACTION_SHA256,
                    "shape": [16, 10],
                },
                "vendor_fixture_video_uri": COSMOS3_BRIDGE_FIXTURE_VIDEO_URL,
                "frame_contract": "N actions -> N+1 returned frames, condition at index 0",
                "action_length_certification_class": self.action_length_certification_class,
                "action_length_certification_source_hash": self.action_length_certification_source_hash,
                "action_length_certification": (
                    None if self.action_length_certification is None else self.action_length_certification.as_dict()
                ),
                "terminal_padding_certificate": (
                    None if self.terminal_padding_certificate is None else self.terminal_padding_certificate.as_dict()
                ),
            },
        )


@dataclass(frozen=True)
class _CosmosRuntime:
    torch: Any
    pipeline_cls: Any
    action_condition_cls: Any
    scheduler_cls: Any = None


class Cosmos3NanoDiffusersAdapter:
    """Official ``Cosmos3OmniPipeline`` FD wrapper, imported only on use.

    A pipeline can be injected for fixture tests, but production code calls the
    public Diffusers API directly.  The adapter never downloads weights and
    never pads/repeats policy actions.  Diffusers itself can repeat a short
    sequence when a caller chooses a larger chunk size, so this wrapper sets
    ``chunk_size == len(compiled_actions)`` and records the action length.
    """

    def __init__(
        self,
        profile: Cosmos3NanoDiffusersProfile,
        *,
        pipeline_factory: Optional[Callable[[Cosmos3NanoDiffusersProfile, _CosmosRuntime], Any]] = None,
        runtime_factory: Optional[Callable[[], _CosmosRuntime]] = None,
    ) -> None:
        self.profile = profile
        self._pipeline_factory = pipeline_factory
        self._runtime_factory = runtime_factory
        self._runtime: Optional[_CosmosRuntime] = None
        self._pipeline: Any = None
        self._real_backend_calls = 0
        self._backend_call_attempts = 0

    @property
    def backend_call_attempts(self) -> int:
        """Actual pipeline invocation attempts, including calls that raise."""

        return self._backend_call_attempts

    @property
    def successful_backend_calls(self) -> int:
        """Calls that returned a frame result passing the adapter frame contract."""

        return self._real_backend_calls

    @property
    def backend_profile(self) -> BackendProfile:
        return self.profile.as_backend_profile()

    def capability(self) -> CapabilityResult:
        model_exists = Path(self.profile.local_model_path).is_dir()
        if not model_exists and self._pipeline_factory is None:
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="Cosmos3-Nano local model directory is not present; no Hub download was attempted.",
                source_verified=True,
                evidence_uris=(COSMOS3_DIFFUSERS_SOURCE, COSMOS3_VENDOR_FIXTURE),
                details={
                    "local_model_path": self.profile.local_model_path,
                    "local_files_only": True,
                    # Action-length certification status is independent of model
                    # availability, so it is reported in both branches.
                    "allowed_action_lengths": self.profile.allowed_action_lengths,
                    "action_length_certification_class": self.profile.action_length_certification_class,
                    "action_length_certification_source_hash": self.profile.action_length_certification_source_hash,
                },
            )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Official Diffusers forward-dynamics call is configured. Vendor and Bridge fixtures plus "
                "actual GPU shape/frame/latency evidence are still required for Gate A; this result is not qualification."
            ),
            source_verified=True,
            evidence_uris=(COSMOS3_DIFFUSERS_SOURCE, COSMOS3_VENDOR_FIXTURE),
            details={
                "model_revision": self.profile.model_revision,
                "diffusers_revision": self.profile.diffusers_revision,
                "allowed_action_lengths": self.profile.allowed_action_lengths,
                "probe_action_lengths": self.profile.probe_action_lengths,
                "expected_frame_rule": "N actions -> N+1 returned frames",
                "safety_checker": self.profile.enable_safety_checker,
                "action_length_certification_class": self.profile.action_length_certification_class,
                "action_length_certification_source_hash": self.profile.action_length_certification_source_hash,
                "terminal_padding_certificate_hash": (
                    None
                    if self.profile.terminal_padding_certificate is None
                    else self.profile.terminal_padding_certificate.certificate_hash
                ),
            },
        )

    def _load_runtime(self) -> _CosmosRuntime:
        if self._runtime is not None:
            return self._runtime
        if self._runtime_factory is not None:
            self._runtime = self._runtime_factory()
            return self._runtime
        try:
            import torch  # type: ignore
            from diffusers import Cosmos3OmniPipeline, UniPCMultistepScheduler  # type: ignore
            from diffusers import CosmosActionCondition  # type: ignore
        except ImportError as error:
            raise BackendUnavailableError(
                "Cosmos3 Diffusers runtime is unavailable. Install the pinned Diffusers source, torch, "
                "transformers and media dependencies on the GPU worker; imports remain lazy by design."
            ) from error
        self._runtime = _CosmosRuntime(
            torch=torch,
            pipeline_cls=Cosmos3OmniPipeline,
            action_condition_cls=CosmosActionCondition,
            scheduler_cls=UniPCMultistepScheduler,
        )
        return self._runtime

    def _ensure_pipeline(self, runtime: _CosmosRuntime) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_factory is not None:
            self._pipeline = self._pipeline_factory(self.profile, runtime)
            return self._pipeline
        path = Path(self.profile.local_model_path)
        if not path.is_dir():
            raise BackendUnavailableError(
                "Expected local Cosmos3-Nano checkpoint directory at %s; refusing a remote model download."
                % path
            )
        # This is the current official constructor.  ``dtype`` (not top-level
        # torch_dtype) is intentionally aligned with the pinned Diffusers docs.
        pipeline = runtime.pipeline_cls.from_pretrained(
            str(path),
            dtype=runtime.torch.bfloat16,
            device_map=self.profile.device_map,
            local_files_only=True,
            enable_safety_checker=self.profile.enable_safety_checker,
        )
        if self.profile.scheduler_flow_shift is not None and runtime.scheduler_cls is not None:
            pipeline.scheduler = runtime.scheduler_cls.from_config(
                pipeline.scheduler.config,
                flow_shift=self.profile.scheduler_flow_shift,
                use_karras_sigmas=False,
            )
        self._pipeline = pipeline
        return self._pipeline

    def _check_request(self, request: WorldRequest) -> str:
        request.validate_basic()
        if request.compatibility_profile_id != self.profile.profile_id:
            raise BackendContractError("WorldRequest profile ID does not match the Cosmos adapter profile.")
        if request.domain != "bridge_orig_lerobot":
            raise BackendContractError(
                "This Bridge FD profile only accepts domain='bridge_orig_lerobot', not %r." % request.domain
            )
        action_count = len(request.compiled_actions)
        if action_count not in self.profile.allowed_action_lengths + self.profile.probe_action_lengths:
            raise BackendContractError(
                "Cosmos request has %d actions; profile permits normal lengths %s and Gate-B probe lengths %s."
                % (action_count, self.profile.allowed_action_lengths, self.profile.probe_action_lengths)
            )
        for action in request.compiled_actions:
            if len(action) != 10:
                raise BackendContractError(
                    "Cosmos bridge_orig_lerobot forward dynamics requires 10-D rows; do not send IRASim 7-D actions."
                )
        return "probe_unqualified" if action_count in self.profile.probe_action_lengths else "configured_unqualified"

    @staticmethod
    def _frames_from_result(result: Any) -> Tuple[Any, ...]:
        video = getattr(result, "video", None)
        if video is None and isinstance(result, Mapping):
            video = result["video"] if "video" in result else result.get("videos")
        if video is None:
            raise BackendContractError("Cosmos pipeline result did not contain video frames.")
        return tuple(video)

    @staticmethod
    def _peak_memory(runtime: _CosmosRuntime) -> Optional[int]:
        cuda = getattr(runtime.torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                return int(cuda.max_memory_allocated())
        except (AttributeError, RuntimeError):
            return None
        return None

    def generate(self, request: WorldRequest) -> WorldResult:
        action_length_status = self._check_request(request)
        runtime = self._load_runtime()
        cold_start = self._pipeline is None
        load_start = time.perf_counter()
        pipeline = self._ensure_pipeline(runtime)
        model_load_seconds = time.perf_counter() - load_start

        # Peak memory is deliberately sampled only around a genuine backend
        # invocation.  It remains null if the runtime cannot report it.
        cuda = getattr(runtime.torch, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                cuda.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            pass

        raw_actions = runtime.torch.as_tensor(request.compiled_actions, dtype=runtime.torch.float32)
        action_condition = runtime.action_condition_cls(
            mode="forward_dynamics",
            chunk_size=len(request.compiled_actions),
            domain_name=request.domain,
            resolution_tier=self.profile.resolution_tier,
            raw_actions=raw_actions,
            image=request.conditioning_image,
            view_point=self.profile.view_point,
        )
        generator = None
        try:
            generator = runtime.torch.Generator(device="cuda").manual_seed(int(request.seed))
        except (AttributeError, RuntimeError, TypeError):
            # Some CPU fixture runtimes only support the device-less constructor.
            generator = runtime.torch.Generator().manual_seed(int(request.seed))

        started_unix = time.time()
        start = time.perf_counter()
        self._backend_call_attempts += 1
        try:
            # In action mode no top-level image/video/height/width/num_frames is
            # passed. Diffusers derives N+1 frames from CosmosActionCondition.
            result = pipeline(
                prompt=request.prompt,
                action=action_condition,
                fps=self.profile.fps,
                num_inference_steps=self.profile.num_inference_steps,
                guidance_scale=self.profile.guidance_scale,
                generator=generator,
                output_type="np",
                use_system_prompt=self.profile.use_system_prompt,
            )
        finally:
            finished_unix = time.time()
        elapsed = time.perf_counter() - start
        self._real_backend_calls += 1
        frames = self._frames_from_result(result)
        timestamps = tuple(index / self.profile.fps for index in range(len(frames)))
        timing = ServerTiming(
            backend_calls=1,
            wall_seconds=elapsed,
            cold_start=cold_start,
            gpu_peak_memory_bytes=self._peak_memory(runtime),
            model_load_seconds=model_load_seconds,
            started_at_unix=started_unix,
            finished_at_unix=finished_unix,
        )
        world_result = WorldResult(
            backend="cosmos3_diffusers",
            profile_id=self.profile.profile_id,
            frames=frames,
            nominal_frame_timestamps=timestamps,
            conditioning_frame_included=True,
            timing=timing,
            request_id=request.request_id,
            metadata={
                "model_id": "nvidia/Cosmos3-Nano",
                "model_revision": self.profile.model_revision,
                "diffusers_revision": self.profile.diffusers_revision,
                "mode": "forward_dynamics",
                "domain_name": request.domain,
                "action_length": len(request.compiled_actions),
                "action_length_status": action_length_status,
                "prompt_is_plain_task_string": True,
                "nominal_timestamps_only": True,
                "feedback_mode": request.feedback_mode.value,
            },
        )
        # Never trim, duplicate, or invent frames to make a test pass.
        world_result.validate(len(request.compiled_actions))
        return world_result


@dataclass(frozen=True)
class IRASimBridgeProfile:
    """Official IRASim Bridge contract; intentionally has no implicit loader."""

    profile_id: str
    archive_path: Optional[str] = None
    archive_sha256: Optional[str] = None
    code_revision: str = IRASIM_COMMIT
    loader_revision: Optional[str] = None
    container_digest: Optional[str] = None
    input_height: int = 256
    input_width: int = 320
    num_frames: int = 16
    action_count: int = 15
    action_scale: Tuple[float, ...] = (20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0)

    def __post_init__(self) -> None:
        if self.input_height != 256 or self.input_width != 320:
            raise ValueError("The pinned IRASim Bridge profile is 256x320.")
        if self.num_frames != 16 or self.action_count != 15 or len(self.action_scale) != 7:
            raise ValueError("The pinned IRASim Bridge contract is 16 frames, 15 7-D actions.")


@dataclass(frozen=True)
class IRASimPreparedRequest:
    profile_id: str
    scaled_actions: Tuple[Tuple[float, ...], ...]
    expected_frames: int
    metadata: Mapping[str, Any]


class IRASimBridgeAdapter:
    """Separate IRASim action preparation contract.

    No loader is provided because a known archive and a code checkout do not
    establish a safe/compatible runnable inference stack.  This adapter may
    prepare a fixture request but always reports unsupported execution until a
    reviewed native loader is supplied.
    """

    def __init__(self, profile: IRASimBridgeProfile) -> None:
        self.profile = profile

    def capability(self) -> CapabilityResult:
        return CapabilityResult(
            status=CapabilityStatus.UNSUPPORTED,
            reason=(
                "IRASim Bridge uses its own 256x320 / 16-frame / 15x7-D contract. "
                "No reviewed native loader is implemented, so it cannot be a Cosmos fallback or qualified backend."
            ),
            source_verified=True,
            evidence_uris=(
                "https://github.com/bytedance/IRASim/blob/%s/configs/evaluation/bridge/frame_ada.yaml" % IRASIM_COMMIT,
                "https://github.com/bytedance/IRASim/blob/%s/dataset/dataset_3D.py" % IRASIM_COMMIT,
            ),
            details={
                "input_size": [self.profile.input_height, self.profile.input_width],
                "num_frames": self.profile.num_frames,
                "action_shape": [self.profile.action_count, 7],
                "action_scale": list(self.profile.action_scale),
            },
        )

    def prepare(self, request: WorldRequest) -> IRASimPreparedRequest:
        request.validate_basic()
        if request.compatibility_profile_id != self.profile.profile_id:
            raise BackendContractError("WorldRequest profile ID does not match the IRASim profile.")
        if len(request.compiled_actions) != self.profile.action_count:
            raise BackendContractError(
                "IRASim needs exactly %d action rows, got %d." % (self.profile.action_count, len(request.compiled_actions))
            )
        scaled = []
        for row in request.compiled_actions:
            if len(row) != 7:
                raise BackendContractError("IRASim accepts native 7-D actions only; Cosmos 10-D rows are prohibited.")
            scaled.append(tuple(float(value) * self.profile.action_scale[index] for index, value in enumerate(row)))
        return IRASimPreparedRequest(
            profile_id=self.profile.profile_id,
            scaled_actions=tuple(scaled),
            expected_frames=self.profile.num_frames,
            metadata={
                "native_action_shape": [15, 7],
                "frame_contract": "condition frame + 15 action transitions = 16 frames",
                "source_revision": self.profile.code_revision,
                "execution_status": "unsupported_no_native_loader",
            },
        )

    def generate(self, request: WorldRequest) -> WorldResult:
        self.prepare(request)
        raise BackendUnavailableError(
            "IRASim request is structurally valid but execution is unsupported until a reviewed native loader, "
            "archive hashes/licenses, and independent fixtures are implemented."
        )
