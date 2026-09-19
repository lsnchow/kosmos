"""Frozen V-JEPA2 video-feature diagnostic, deliberately outside Stage A.

``facebook/vjepa2-vitl-fpc64-256`` is an **encoder**, not a text-following
rubric judge (spec 1).  This module therefore produces *features and distances*
and nothing else.  It never emits a success/failure label, a progress score, a
validity verdict, or a calibrated fidelity claim.

Separation from the deterministic validity gate is the point, so it is
structural rather than documentary (spec 5):

* This module does not import :mod:`plumb.validity`, and
  :mod:`plumb.validity` does not import this module.  Both directions are
  machine-checked by :func:`stage_a_separation_evidence`, which resolves each
  module's file with ``importlib.util.find_spec`` (locating without executing)
  and reads the real import edges out of its AST.
* :class:`~plumb.validity.StageAValidityGate` exposes ``vjepa_hook: None``.
  That is not an unfilled slot waiting for this file.  There is no hook object
  here for anything to install, and nothing in this module accepts, returns, or
  mutates a ``ValidityReport``.
* Every emitted record carries ``diagnostic_only=True``,
  ``alters_validity_gate=False``, ``is_a_task_score=False`` and an explicit
  limitation string.  ``as_dict`` self-checks that no forbidden verdict key
  (``validity``, ``reason_codes``, ``progress``, ``success``, ``score``, ...)
  ever appears in an emitted payload.

Preprocessing discipline mirrors the 16-frame judge contract in
:mod:`plumb.policies.judge`: the checkpoint is ``fpc64``, so a clip that cannot
supply the documented 64 frames under an explicit timestamp sampling rule is
**rejected**.  Nothing here pads, duplicates, interpolates, or silently
resamples a short clip.

Normalization constants are not hardcoded.  The shipped
``video_preprocessor_config.json`` is the authority for mean/std/rescale, so
this module pins the *file* and records the loaded processor's identity instead
of inventing numbers.

``torch``/``transformers`` imports are lazy, exactly as the policy and judge
adapters do, so ``import plumb.features`` works on a CPU laptop with neither
installed.  No GPU or asset was available when this module was written: the
processor call style and batch axis are declared explicitly and flagged as
requiring fixture certification before any real number is trusted.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from plumb.adapters.contracts import CapabilityResult, CapabilityStatus


FEATURE_DIAGNOSTIC_VERSION = "plumb-vjepa2-feature-diagnostic-v1"

VJEPA2_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_SOURCE_URI = "https://huggingface.co/facebook/vjepa2-vitl-fpc64-256"

#: The checkpoint name encodes its own contract: 64 frames per clip at 256 px.
VJEPA2_FRAMES_PER_CLIP = 64
VJEPA2_CROP_SIZE = 256

#: The repo has not resolved an immutable V-JEPA2 commit (the asset plan carries
#: an unresolved-revision blocker and ``assets.lock.json`` has no entry).  A
#: revision is therefore a required caller input, never a default here.
VJEPA2_REVISION_UNRESOLVED_NOTE = (
    "facebook/vjepa2-vitl-fpc64-256 has no resolved immutable commit in this repo's asset lock. "
    "A profile must supply one; this module will not invent or default a revision."
)

STAGE_A_MODULE = "plumb.validity"
FEATURE_MODULE = "plumb.features"


@dataclass(frozen=True)
class VJepaRequiredFile:
    """One spec-listed candidate file and its candidate byte length."""

    path: str
    candidate_bytes: int

    def as_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "candidate_bytes": int(self.candidate_bytes)}


#: Spec 1 candidate files.  ``original/model.pth`` is a duplicate of the
#: safetensors weights and is deliberately absent from this tuple.
VJEPA2_REQUIRED_FILES: Tuple[VJepaRequiredFile, ...] = (
    VJepaRequiredFile("config.json", 785),
    VJepaRequiredFile("model.safetensors", 1_303_947_864),
    VJepaRequiredFile("video_preprocessor_config.json", 1_298),
)

#: Never fetched, never loaded, never hashed.
VJEPA2_SKIPPED_FILES: Tuple[str, ...] = ("original/model.pth",)
VJEPA2_SKIPPED_FILE_REASON = (
    "original/model.pth duplicates model.safetensors at roughly 5.1 GB; the safetensors weights are loaded instead."
)

#: Explicit timestamp-sampling rules.  Both select 64 *distinct existing*
#: frames; neither fabricates a frame.
SAMPLING_UNIFORM_INCLUSIVE = "uniform_inclusive_endpoints_v1"
SAMPLING_CONTIGUOUS_PREFIX = "contiguous_prefix_v1"
SAMPLING_RULES: Tuple[str, ...] = (SAMPLING_UNIFORM_INCLUSIVE, SAMPLING_CONTIGUOUS_PREFIX)

SAMPLING_RULE_DESCRIPTIONS: Mapping[str, str] = {
    SAMPLING_UNIFORM_INCLUSIVE: (
        "64 distinct frame indexes spaced uniformly from the first source frame through the exact final "
        "control tick, including both endpoints; integer arithmetic, no interpolation."
    ),
    SAMPLING_CONTIGUOUS_PREFIX: (
        "the model card's literal arange(0, 64): the first 64 source frames in order, which discards the tail "
        "of a longer clip."
    ),
}

POOLING_MEAN_OVER_TOKENS = "mean_over_tokens_v1"
POOLING_IDENTITY_VECTOR = "identity_vector_v1"

FEATURE_API_VISION_FEATURES = "get_vision_features"
FEATURE_API_LAST_HIDDEN_STATE = "last_hidden_state"
FEATURE_API_PREFERENCE: Tuple[str, ...] = (FEATURE_API_VISION_FEATURES, FEATURE_API_LAST_HIDDEN_STATE)

PROCESSOR_SINGLE_VIDEO = "single_video_frame_sequence"
PROCESSOR_BATCHED_VIDEO_LIST = "batched_video_list"
PROCESSOR_CALL_STYLES: Tuple[str, ...] = (PROCESSOR_SINGLE_VIDEO, PROCESSOR_BATCHED_VIDEO_LIST)

#: The byte lengths in spec 1 are the only provenance pin available for this
#: asset, so verifying them is the default.  The development class exists for
#: fixtures and, like ``uncalibrated_development`` in the Stage A gate, labels
#: every record it produces so it can never be mistaken for a verified one.
ASSET_BYTES_VERIFIED = "candidate_bytes_verified"
ASSET_BYTES_DEVELOPMENT = "development_unverified_bytes"
ASSET_VERIFICATION_CLASSES: Tuple[str, ...] = (ASSET_BYTES_VERIFIED, ASSET_BYTES_DEVELOPMENT)

DISTANCE_COMPUTED = "computed"
DISTANCE_UNDEFINED_ZERO_NORM = "undefined_zero_norm"

FEATURE_DIAGNOSTIC_LIMITATION = (
    "An encoder feature distance is not a success measure and is not calibrated against human judgement. "
    "It cannot establish task success, failure, progress, physical fidelity, or episode validity, and it "
    "never participates in the deterministic Stage A validity gate."
)

DIVERGENCE_SCOPE_NOTE = (
    "visual divergence only; action alignment and integrity are separate measurements (spec 6.3)"
)
NO_TOLERANCE_NOTE = (
    "no divergence tolerance or threshold is applied here; 'no failure observed through the last tested "
    "control tick' is not an unlimited drift guarantee"
)
FIXTURE_VERIFICATION_NOTE = (
    "processor_call_style, batch axis and the shipped video_preprocessor_config.json must be certified "
    "against a golden fixture before any real embedding or distance is trusted; no asset or GPU was "
    "available when this module was written."
)

#: Keys that must never appear in an emitted diagnostic record.  This module is
#: an encoder: a verdict-shaped key here would be a category error.
FORBIDDEN_RESULT_KEYS: Tuple[str, ...] = (
    "binary_success",
    "collision",
    "completion_evidence",
    "fail",
    "failed",
    "failure",
    "integrity",
    "invalid",
    "label",
    "milestone",
    "pass",
    "passed",
    "progress",
    "progress_score",
    "reason_code",
    "reason_codes",
    "score",
    "success",
    "success_rate",
    "task_score",
    "valid",
    "validity",
    "verdict",
)


class VJepaConfigurationError(ValueError):
    """The diagnostic itself is misconfigured. This is never an episode verdict."""


class VJepaLoadError(RuntimeError):
    """The pinned frozen encoder cannot be loaded in this environment."""


class VJepaInputError(ValueError):
    """A clip or comparison input violates the documented 64-frame contract."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_hex_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _immutable_revision(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _is_dir(path: Any) -> bool:
    try:
        return os.path.isdir(str(path))
    except (OSError, TypeError, ValueError):
        return False


def _file_size(path: Any) -> Optional[int]:
    try:
        if not os.path.isfile(str(path)):
            return None
        return int(os.path.getsize(str(path)))
    except (OSError, TypeError, ValueError):
        return None


def _finite_float(value: Any, *, what: str) -> float:
    if isinstance(value, bool):
        raise VJepaInputError("%s must be a finite real number, not a bool." % what)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise VJepaInputError("%s must be a finite real number." % what) from error
    if not math.isfinite(numeric):
        raise VJepaInputError("%s must be finite; refusing to record a NaN or infinity." % what)
    return numeric


def _diagnostic_record(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Stamp the diagnostic labels on one record and self-check its keys."""

    record: Dict[str, Any] = dict(payload)
    record["diagnostic_only"] = True
    record["alters_validity_gate"] = False
    record["is_a_task_score"] = False
    record["limitation"] = FEATURE_DIAGNOSTIC_LIMITATION
    forbidden = sorted(key for key in record if key in FORBIDDEN_RESULT_KEYS)
    if forbidden:
        raise VJepaConfigurationError(
            "A V-JEPA2 feature record may not carry verdict-shaped keys %s; this diagnostic is an encoder, "
            "not a judge." % forbidden
        )
    return record


def _check_labels(
    *, diagnostic_only: bool, alters_validity_gate: bool, is_a_task_score: bool, limitation: str
) -> None:
    if diagnostic_only is not True:
        raise VJepaConfigurationError("V-JEPA2 feature records are diagnostic_only=True; there is no other mode.")
    if alters_validity_gate is not False:
        raise VJepaConfigurationError(
            "alters_validity_gate must be False: this diagnostic is structurally outside the Stage A gate."
        )
    if is_a_task_score is not False:
        raise VJepaConfigurationError("is_a_task_score must be False: an encoder distance is not a task score.")
    if limitation != FEATURE_DIAGNOSTIC_LIMITATION:
        raise VJepaConfigurationError("The limitation string is frozen and may not be weakened or replaced.")


# ---------------------------------------------------------------------------
# Frozen preprocessing and profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VJepaPreprocessing:
    """The documented 64-frame V-JEPA2 preprocessing contract, pinned.

    ``video_preprocessor_config_sha256`` pins the shipped preprocessor file
    when the asset lock has hashed it.  Mean/std/rescale values are read from
    that file at load time and are intentionally *not* duplicated here: the
    file is the authority, and transcribing guessed constants would be
    fabrication.
    """

    frames_per_clip: int = VJEPA2_FRAMES_PER_CLIP
    crop_size: int = VJEPA2_CROP_SIZE
    sampling_rule: str = SAMPLING_UNIFORM_INCLUSIVE
    require_distinct_frame_indices: bool = True
    pooling: str = POOLING_MEAN_OVER_TOKENS
    processor_call_style: str = PROCESSOR_SINGLE_VIDEO
    video_preprocessor_config_sha256: Optional[str] = None
    normalization_authority: str = "video_preprocessor_config.json"

    def configuration_error(self) -> Optional[str]:
        if self.frames_per_clip != VJEPA2_FRAMES_PER_CLIP:
            return (
                "frames_per_clip must be %d for %s; a different clip length is a different model contract"
                % (VJEPA2_FRAMES_PER_CLIP, VJEPA2_MODEL_ID)
            )
        if self.crop_size != VJEPA2_CROP_SIZE:
            return "crop_size must be %d for %s" % (VJEPA2_CROP_SIZE, VJEPA2_MODEL_ID)
        if self.sampling_rule not in SAMPLING_RULES:
            return "sampling_rule must be one of %s" % (SAMPLING_RULES,)
        if self.require_distinct_frame_indices is not True:
            return (
                "require_distinct_frame_indices must stay True; repeating a frame to reach 64 is the silent "
                "resampling this diagnostic refuses"
            )
        if self.pooling not in (POOLING_MEAN_OVER_TOKENS,):
            return "pooling must be %s" % POOLING_MEAN_OVER_TOKENS
        if self.processor_call_style not in PROCESSOR_CALL_STYLES:
            return "processor_call_style must be one of %s" % (PROCESSOR_CALL_STYLES,)
        if self.video_preprocessor_config_sha256 is not None and not _is_hex_sha256(
            self.video_preprocessor_config_sha256
        ):
            return "video_preprocessor_config_sha256 must be a 64-character hexadecimal SHA-256 when supplied"
        if not isinstance(self.normalization_authority, str) or not self.normalization_authority:
            return "normalization_authority must name the file that defines mean/std/rescale"
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frames_per_clip": int(self.frames_per_clip),
            "crop_size": int(self.crop_size),
            "sampling_rule": self.sampling_rule,
            "sampling_rule_description": SAMPLING_RULE_DESCRIPTIONS.get(self.sampling_rule),
            "require_distinct_frame_indices": bool(self.require_distinct_frame_indices),
            "pooling": self.pooling,
            "processor_call_style": self.processor_call_style,
            "video_preprocessor_config_sha256": self.video_preprocessor_config_sha256,
            "normalization_authority": self.normalization_authority,
        }

    @property
    def preprocessing_hash(self) -> str:
        return _canonical_hash(
            {
                "diagnostic_version": FEATURE_DIAGNOSTIC_VERSION,
                "preprocessing": self.as_dict(),
            }
        )


@dataclass(frozen=True)
class VJepaProfile:
    """One revision-pinned, local-only frozen V-JEPA2 encoder deployment.

    ``transformers_version`` has no default on purpose.  V-JEPA2 support is
    version-dependent and this repo has not certified a runtime for it, so the
    caller states the pin and the loader refuses a mismatch.
    """

    profile_id: str
    local_model_path: str
    model_revision: str
    transformers_version: str
    preprocessing: VJepaPreprocessing = VJepaPreprocessing()
    processor_revision: Optional[str] = None
    asset_verification: str = ASSET_BYTES_VERIFIED
    local_files_only: bool = True
    trust_remote_code: bool = False
    torch_dtype: str = "float32"
    device: Optional[str] = None
    device_map: Optional[str] = None
    container_digest: Optional[str] = None
    asset_manifest_id: Optional[str] = None
    asset_manifest_sha256: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None
    required_files: Tuple[VJepaRequiredFile, ...] = VJEPA2_REQUIRED_FILES
    skipped_files: Tuple[str, ...] = VJEPA2_SKIPPED_FILES
    model_id: str = VJEPA2_MODEL_ID

    def profile_error(self) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        if not self.local_model_path:
            return "local_model_path is required; Hub fetching is disabled"
        if self.model_id != VJEPA2_MODEL_ID:
            return "model_id must be %s for this diagnostic" % VJEPA2_MODEL_ID
        if not _immutable_revision(self.model_revision):
            return (
                "model_revision must be an immutable 40-character hexadecimal revision. "
                + VJEPA2_REVISION_UNRESOLVED_NOTE
            )
        if self.processor_revision is not None and not _immutable_revision(self.processor_revision):
            return "processor_revision must be an immutable 40-character hexadecimal revision when supplied"
        if not isinstance(self.transformers_version, str) or not self.transformers_version:
            return "transformers_version must be an explicit pinned runtime version"
        if not self.local_files_only:
            return "the frozen feature diagnostic is local-only; network retrieval is not permitted"
        if self.trust_remote_code:
            return "V-JEPA2 must load as a native pinned Transformers architecture, not via trust_remote_code"
        if self.torch_dtype not in ("float32", "bfloat16", "float16"):
            return "torch_dtype must be float32, bfloat16, or float16"
        if self.asset_verification not in ASSET_VERIFICATION_CLASSES:
            return "asset_verification must be one of %s" % (ASSET_VERIFICATION_CLASSES,)
        if not self.required_files:
            return "required_files must name the spec candidate files"
        for skipped in self.skipped_files:
            if any(required.path == skipped for required in self.required_files):
                return "%s is a skipped duplicate and must not appear in required_files" % skipped
        return self.preprocessing.configuration_error()

    @property
    def resolved_processor_revision(self) -> str:
        """The processor shares the model commit unless one is pinned separately."""

        return self.processor_revision or self.model_revision

    @property
    def verifies_candidate_bytes(self) -> bool:
        return self.asset_verification == ASSET_BYTES_VERIFIED

    def as_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "local_model_path": self.local_model_path,
            "model_revision": self.model_revision,
            "processor_revision": self.resolved_processor_revision,
            "transformers_version": self.transformers_version,
            "preprocessing": self.preprocessing.as_dict(),
            "asset_verification": self.asset_verification,
            "local_files_only": bool(self.local_files_only),
            "trust_remote_code": bool(self.trust_remote_code),
            "torch_dtype": self.torch_dtype,
            "device": self.device,
            "device_map": self.device_map,
            "container_digest": self.container_digest,
            "asset_manifest_id": self.asset_manifest_id,
            "asset_manifest_sha256": self.asset_manifest_sha256,
            "runtime_lock_id": self.runtime_lock_id,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "required_files": [item.as_dict() for item in self.required_files],
            "skipped_files": list(self.skipped_files),
            "skipped_file_reason": VJEPA2_SKIPPED_FILE_REASON,
            "feature_api_preference": list(FEATURE_API_PREFERENCE),
        }

    @property
    def parameters_hash(self) -> str:
        """Declared-parameter pin. Resolved values are hashed per embedding."""

        return _canonical_hash(
            {
                "diagnostic_version": FEATURE_DIAGNOSTIC_VERSION,
                "declared_profile": self.as_dict(),
            }
        )

    @property
    def measurement_parameters(self) -> Dict[str, Any]:
        """Only the declared fields that can change a measured feature value.

        The mount path, container digest and manifest identifiers are
        reproducibility metadata rather than measurement inputs: two mounts of
        the same revision produce the same numbers.  They stay in
        ``parameters_hash``.  ``device``/``device_map`` are included because
        placement can change floating-point results.
        """

        return {
            "measurement_contract": "plumb-vjepa2-measurement-v1",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "processor_revision": self.resolved_processor_revision,
            "transformers_version": self.transformers_version,
            "torch_dtype": self.torch_dtype,
            "device": self.device,
            "device_map": self.device_map,
            "preprocessing": self.preprocessing.as_dict(),
            "feature_api_preference": list(FEATURE_API_PREFERENCE),
        }

    @property
    def measurement_parameters_hash(self) -> str:
        """What two embeddings must share before a distance between them means anything."""

        return _canonical_hash(self.measurement_parameters)


@dataclass(frozen=True)
class _VJepaRuntime:
    """Lazily resolved third-party runtime. Never imported at module scope."""

    torch: Any
    model_cls: Any
    processor_cls: Any
    transformers_version: str


# ---------------------------------------------------------------------------
# Explicit timestamp sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VJepaSampling:
    """The explicit 64-index / 64-timestamp selection actually used."""

    rule: str
    frame_indices: Tuple[int, ...]
    frame_timestamps: Tuple[float, ...]
    source_frame_count: int
    source_first_timestamp: float
    source_last_timestamp: float

    def __post_init__(self) -> None:
        if self.rule not in SAMPLING_RULES:
            raise VJepaConfigurationError("sampling rule must be one of %s" % (SAMPLING_RULES,))
        if len(self.frame_indices) != VJEPA2_FRAMES_PER_CLIP:
            raise VJepaConfigurationError(
                "a V-JEPA2 clip is exactly %d frames; got %d indexes" % (VJEPA2_FRAMES_PER_CLIP, len(self.frame_indices))
            )
        if len(self.frame_timestamps) != VJEPA2_FRAMES_PER_CLIP:
            raise VJepaConfigurationError("one control timestamp is required for each of the 64 frames")
        if len(set(self.frame_indices)) != len(self.frame_indices):
            raise VJepaConfigurationError("frame indexes must be distinct; a repeated frame is silent resampling")

    @property
    def includes_first_source_frame(self) -> bool:
        return self.frame_indices[0] == 0

    @property
    def includes_final_control_tick(self) -> bool:
        return self.frame_indices[-1] == self.source_frame_count - 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "rule_description": SAMPLING_RULE_DESCRIPTIONS.get(self.rule),
            "frame_indices": list(self.frame_indices),
            "frame_timestamps": list(self.frame_timestamps),
            "frames_per_clip": VJEPA2_FRAMES_PER_CLIP,
            "source_frame_count": int(self.source_frame_count),
            "source_first_timestamp": self.source_first_timestamp,
            "source_last_timestamp": self.source_last_timestamp,
            "includes_first_source_frame": self.includes_first_source_frame,
            "includes_final_control_tick": self.includes_final_control_tick,
            "sampling_hash": self.sampling_hash,
        }

    @property
    def sampling_hash(self) -> str:
        return _canonical_hash(
            {
                "rule": self.rule,
                "frame_indices": list(self.frame_indices),
                "frame_timestamps": list(self.frame_timestamps),
            }
        )


def _as_list(value: Any, *, what: str) -> List[Any]:
    """Accept any non-text iterable, including numpy arrays, as a clip axis."""

    if isinstance(value, (str, bytes, bytearray)):
        raise VJepaInputError("%s must be a sequence, not text." % what)
    try:
        return list(value)
    except TypeError as error:
        raise VJepaInputError("%s must be an iterable sequence." % what) from error


def _validate_timestamps(timestamps: Any) -> Tuple[float, ...]:
    items = _as_list(timestamps, what="timestamps")
    if not items:
        raise VJepaInputError("timestamps must be a nonempty sequence of control timestamps.")
    values: List[float] = []
    previous: Optional[float] = None
    for position, value in enumerate(items):
        numeric = _finite_float(value, what="timestamps[%d]" % position)
        if previous is not None and numeric <= previous:
            raise VJepaInputError(
                "Control timestamps must be strictly increasing; timestamps[%d] is not greater than its "
                "predecessor. These are control timestamps, not display playback times." % position
            )
        values.append(numeric)
        previous = numeric
    return tuple(values)


def plan_sampling(timestamps: Any, *, rule: str = SAMPLING_UNIFORM_INCLUSIVE) -> VJepaSampling:
    """Resolve the explicit 64 frame indexes and timestamps for one clip.

    Raises :class:`VJepaInputError` when the clip cannot supply 64 distinct
    frames under the documented sampling.  Short clips are rejected, never
    padded, duplicated, interpolated, or resampled.
    """

    if rule not in SAMPLING_RULES:
        raise VJepaConfigurationError("sampling rule must be one of %s" % (SAMPLING_RULES,))
    values = _validate_timestamps(timestamps)
    available = len(values)
    if available < VJEPA2_FRAMES_PER_CLIP:
        raise VJepaInputError(
            "%s requires exactly %d frames per clip; this clip supplies %d. The feature diagnostic rejects a "
            "clip that cannot satisfy the documented 64-frame preprocessing rather than padding, duplicating, "
            "interpolating, or resampling it."
            % (VJEPA2_MODEL_ID, VJEPA2_FRAMES_PER_CLIP, available)
        )
    if rule == SAMPLING_CONTIGUOUS_PREFIX:
        indices = tuple(range(VJEPA2_FRAMES_PER_CLIP))
    else:
        span = available - 1
        last = VJEPA2_FRAMES_PER_CLIP - 1
        # Integer round-half-up of position * span / last: exact endpoints, no
        # floating-point drift, and monotone because span >= last here.
        indices = tuple(
            (2 * position * span + last) // (2 * last) for position in range(VJEPA2_FRAMES_PER_CLIP)
        )
    if len(set(indices)) != VJEPA2_FRAMES_PER_CLIP:
        raise VJepaInputError(
            "Sampling rule %s produced repeated frame indexes for a %d-frame clip; refusing to embed a clip "
            "with duplicated frames." % (rule, available)
        )
    return VJepaSampling(
        rule=rule,
        frame_indices=indices,
        frame_timestamps=tuple(values[index] for index in indices),
        source_frame_count=available,
        source_first_timestamp=values[0],
        source_last_timestamp=values[-1],
    )


# ---------------------------------------------------------------------------
# Emitted records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureVector:
    """One frozen-encoder embedding of one 64-frame clip. Not a score."""

    embedding: Tuple[float, ...]
    sampling: VJepaSampling
    model_revision: str
    preprocessing_hash: str
    parameters_hash: str
    measurement_hash: str
    feature_api_used: str
    pooling_applied: str
    asset_verification: str
    clip_id: Optional[str] = None
    model_id: str = VJEPA2_MODEL_ID
    diagnostic_version: str = FEATURE_DIAGNOSTIC_VERSION
    diagnostic_only: bool = True
    alters_validity_gate: bool = False
    is_a_task_score: bool = False
    limitation: str = FEATURE_DIAGNOSTIC_LIMITATION

    def __post_init__(self) -> None:
        _check_labels(
            diagnostic_only=self.diagnostic_only,
            alters_validity_gate=self.alters_validity_gate,
            is_a_task_score=self.is_a_task_score,
            limitation=self.limitation,
        )
        if not self.embedding:
            raise VJepaConfigurationError("An embedding cannot be empty; there is no default feature vector.")
        for value in self.embedding:
            if not isinstance(value, float) or not math.isfinite(value):
                raise VJepaConfigurationError("Embedding components must be finite floats.")
        if self.feature_api_used not in FEATURE_API_PREFERENCE:
            raise VJepaConfigurationError("feature_api_used must be one of %s" % (FEATURE_API_PREFERENCE,))
        if self.pooling_applied not in (POOLING_MEAN_OVER_TOKENS, POOLING_IDENTITY_VECTOR):
            raise VJepaConfigurationError("pooling_applied must name the pooling actually performed")
        if self.asset_verification not in ASSET_VERIFICATION_CLASSES:
            raise VJepaConfigurationError("asset_verification must be one of %s" % (ASSET_VERIFICATION_CLASSES,))

    @property
    def dimension(self) -> int:
        return len(self.embedding)

    @property
    def frame_indices(self) -> Tuple[int, ...]:
        return self.sampling.frame_indices

    @property
    def frame_timestamps(self) -> Tuple[float, ...]:
        return self.sampling.frame_timestamps

    @property
    def l2_norm(self) -> float:
        return math.sqrt(math.fsum(value * value for value in self.embedding))

    def as_dict(self) -> Dict[str, Any]:
        return _diagnostic_record(
            {
                "record_kind": "vjepa2_feature_vector",
                "clip_id": self.clip_id,
                "embedding": list(self.embedding),
                "dimension": self.dimension,
                "l2_norm": self.l2_norm,
                "frame_indices": list(self.frame_indices),
                "frame_timestamps": list(self.frame_timestamps),
                "sampling": self.sampling.as_dict(),
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "preprocessing_hash": self.preprocessing_hash,
                "parameters_hash": self.parameters_hash,
                "measurement_hash": self.measurement_hash,
                "feature_api_used": self.feature_api_used,
                "pooling_applied": self.pooling_applied,
                "asset_verification": self.asset_verification,
                "diagnostic_version": self.diagnostic_version,
            }
        )


@dataclass(frozen=True)
class ClipFeatureDistance:
    """A distance between two clips' embeddings. Not a success measure."""

    status: str
    cosine_distance: Optional[float]
    l2_distance: Optional[float]
    identical_embeddings: bool
    dimension: int
    model_revision: str
    preprocessing_hash: str
    measurement_hash: str
    asset_verification: str
    left_clip_id: Optional[str] = None
    right_clip_id: Optional[str] = None
    reason: Optional[str] = None
    diagnostic_version: str = FEATURE_DIAGNOSTIC_VERSION
    diagnostic_only: bool = True
    alters_validity_gate: bool = False
    is_a_task_score: bool = False
    limitation: str = FEATURE_DIAGNOSTIC_LIMITATION

    def __post_init__(self) -> None:
        _check_labels(
            diagnostic_only=self.diagnostic_only,
            alters_validity_gate=self.alters_validity_gate,
            is_a_task_score=self.is_a_task_score,
            limitation=self.limitation,
        )
        if self.status not in (DISTANCE_COMPUTED, DISTANCE_UNDEFINED_ZERO_NORM):
            raise VJepaConfigurationError("distance status must be computed or undefined_zero_norm")
        if self.status == DISTANCE_COMPUTED:
            if self.cosine_distance is None or self.l2_distance is None:
                raise VJepaConfigurationError("a computed distance must carry both distances, never a stand-in zero")
        elif self.cosine_distance is not None:
            raise VJepaConfigurationError("an undefined cosine distance stays None; it is never reported as zero")

    def as_dict(self) -> Dict[str, Any]:
        return _diagnostic_record(
            {
                "record_kind": "vjepa2_clip_distance",
                "status": self.status,
                "reason": self.reason,
                "cosine_distance": self.cosine_distance,
                "l2_distance": self.l2_distance,
                "identical_embeddings": bool(self.identical_embeddings),
                "dimension": int(self.dimension),
                "left_clip_id": self.left_clip_id,
                "right_clip_id": self.right_clip_id,
                "model_id": VJEPA2_MODEL_ID,
                "model_revision": self.model_revision,
                "preprocessing_hash": self.preprocessing_hash,
                "measurement_hash": self.measurement_hash,
                "asset_verification": self.asset_verification,
                "diagnostic_version": self.diagnostic_version,
            }
        )


@dataclass(frozen=True)
class FeatureDistancePoint:
    """One control tick and the clip-to-clip distance measured there."""

    control_tick: int
    distance: ClipFeatureDistance

    def as_dict(self) -> Dict[str, Any]:
        return _diagnostic_record(
            {
                "record_kind": "vjepa2_distance_point",
                "control_tick": int(self.control_tick),
                "distance": self.distance.as_dict(),
            }
        )


@dataclass(frozen=True)
class _DistanceSeries:
    """Shared honest summary of a sequence of clip-to-clip distances."""

    points: Tuple[FeatureDistancePoint, ...]

    @property
    def control_ticks(self) -> Tuple[int, ...]:
        return tuple(point.control_tick for point in self.points)

    @property
    def computed_point_count(self) -> int:
        return sum(1 for point in self.points if point.distance.status == DISTANCE_COMPUTED)

    @property
    def undefined_point_count(self) -> int:
        return len(self.points) - self.computed_point_count

    @property
    def last_control_tick_tested(self) -> Optional[int]:
        return self.points[-1].control_tick if self.points else None

    def _computed(self) -> Tuple[Tuple[int, float], ...]:
        return tuple(
            (point.control_tick, float(point.distance.cosine_distance))
            for point in self.points
            if point.distance.status == DISTANCE_COMPUTED and point.distance.cosine_distance is not None
        )

    @property
    def max_cosine_distance(self) -> Optional[float]:
        computed = self._computed()
        return max(value for _, value in computed) if computed else None

    @property
    def max_cosine_distance_control_tick(self) -> Optional[int]:
        computed = self._computed()
        if not computed:
            return None
        best = max(computed, key=lambda item: item[1])
        return best[0]

    @property
    def mean_cosine_distance(self) -> Optional[float]:
        computed = self._computed()
        if not computed:
            return None
        return math.fsum(value for _, value in computed) / len(computed)

    def _series_dict(self) -> Dict[str, Any]:
        return {
            "points": [point.as_dict() for point in self.points],
            "control_ticks": list(self.control_ticks),
            "point_count": len(self.points),
            "computed_point_count": self.computed_point_count,
            "undefined_point_count": self.undefined_point_count,
            "max_cosine_distance": self.max_cosine_distance,
            "max_cosine_distance_control_tick": self.max_cosine_distance_control_tick,
            "mean_cosine_distance": self.mean_cosine_distance,
            "last_control_tick_tested": self.last_control_tick_tested,
        }


@dataclass(frozen=True)
class TeacherForcedVsFreeRunningDivergence(_DistanceSeries):
    """Feature divergence between a teacher-forced and a free-running branch.

    Spec 6.3 asks for visual divergence at *equal control prefixes*, using the
    same actions and generation settings.  Both identifiers are required and
    recorded so a reader can see which branches were paired.  Nothing here
    declares a tolerance, a pass, or a drift limit.
    """

    # Both identifiers are required in practice.  They carry empty defaults only
    # because the inherited ``points`` field must stay positional; ``__post_init__``
    # rejects an empty value, so an unlabelled divergence cannot be constructed.
    action_sequence_id: str = ""
    generation_settings_id: str = ""
    diagnostic_version: str = FEATURE_DIAGNOSTIC_VERSION
    diagnostic_only: bool = True
    alters_validity_gate: bool = False
    is_a_task_score: bool = False
    limitation: str = FEATURE_DIAGNOSTIC_LIMITATION

    def __post_init__(self) -> None:
        _check_labels(
            diagnostic_only=self.diagnostic_only,
            alters_validity_gate=self.alters_validity_gate,
            is_a_task_score=self.is_a_task_score,
            limitation=self.limitation,
        )
        if not self.action_sequence_id:
            raise VJepaConfigurationError(
                "action_sequence_id is required: spec 6.3 compares branches driven by the same actions."
            )
        if not self.generation_settings_id:
            raise VJepaConfigurationError(
                "generation_settings_id is required: spec 6.3 compares branches at the same generation settings."
            )

    def as_dict(self) -> Dict[str, Any]:
        payload = self._series_dict()
        payload.update(
            {
                "record_kind": "vjepa2_teacher_forced_vs_free_running_divergence",
                "action_sequence_id": self.action_sequence_id,
                "generation_settings_id": self.generation_settings_id,
                "diagnostic_version": self.diagnostic_version,
                "scope_note": DIVERGENCE_SCOPE_NOTE,
                "tolerance_note": NO_TOLERANCE_NOTE,
            }
        )
        return _diagnostic_record(payload)


@dataclass(frozen=True)
class FeatureDriftCurve(_DistanceSeries):
    """Clip-to-clip feature distance from one reference clip over control ticks.

    ``monotone_nondecreasing`` is an observation about the measured points, not
    a verdict and not a drift bound.
    """

    reference_clip_id: Optional[str] = None
    x_axis: str = "control_ticks"
    diagnostic_version: str = FEATURE_DIAGNOSTIC_VERSION
    diagnostic_only: bool = True
    alters_validity_gate: bool = False
    is_a_task_score: bool = False
    limitation: str = FEATURE_DIAGNOSTIC_LIMITATION

    def __post_init__(self) -> None:
        _check_labels(
            diagnostic_only=self.diagnostic_only,
            alters_validity_gate=self.alters_validity_gate,
            is_a_task_score=self.is_a_task_score,
            limitation=self.limitation,
        )

    @property
    def monotone_nondecreasing(self) -> Optional[bool]:
        computed = self._computed()
        if len(computed) < 2:
            return None
        values = [value for _, value in computed]
        return all(later >= earlier for earlier, later in zip(values, values[1:]))

    def as_dict(self) -> Dict[str, Any]:
        payload = self._series_dict()
        payload.update(
            {
                "record_kind": "vjepa2_feature_drift_curve",
                "reference_clip_id": self.reference_clip_id,
                "x_axis": self.x_axis,
                "monotone_nondecreasing": self.monotone_nondecreasing,
                "diagnostic_version": self.diagnostic_version,
                "scope_note": DIVERGENCE_SCOPE_NOTE,
                "tolerance_note": NO_TOLERANCE_NOTE,
            }
        )
        return _diagnostic_record(payload)


# ---------------------------------------------------------------------------
# Comparison utilities (no model required; they read emitted FeatureVectors)
# ---------------------------------------------------------------------------


def _require_comparable(left: Any, right: Any) -> None:
    for name, candidate in (("left", left), ("right", right)):
        if not isinstance(candidate, FeatureVector):
            raise VJepaInputError("%s must be a FeatureVector emitted by this diagnostic." % name)
    mismatches: List[str] = []
    if left.model_id != right.model_id:
        mismatches.append("model_id")
    if left.model_revision != right.model_revision:
        mismatches.append("model_revision")
    if left.preprocessing_hash != right.preprocessing_hash:
        mismatches.append("preprocessing_hash")
    if left.measurement_hash != right.measurement_hash:
        mismatches.append("measurement_hash")
    if mismatches:
        raise VJepaInputError(
            "Refusing to report a distance across differing %s: embeddings from different encoders, revisions, "
            "preprocessing or feature APIs are not on a common scale." % ", ".join(mismatches)
        )
    if left.dimension != right.dimension:
        raise VJepaInputError(
            "Embedding dimensions differ (%d versus %d); there is no meaningful distance between them."
            % (left.dimension, right.dimension)
        )


def _weaker_asset_verification(left: FeatureVector, right: FeatureVector) -> str:
    if ASSET_BYTES_DEVELOPMENT in (left.asset_verification, right.asset_verification):
        return ASSET_BYTES_DEVELOPMENT
    return ASSET_BYTES_VERIFIED


def compare_clips(left: FeatureVector, right: FeatureVector) -> ClipFeatureDistance:
    """Cosine and L2 distance between two clips' frozen-encoder embeddings.

    Exactly equal embeddings report distance ``0.0`` and
    ``identical_embeddings=True``; reporting the floating-point residual of
    ``s / (sqrt(s) * sqrt(s))`` instead would be noise dressed as signal.
    A zero-norm embedding leaves the cosine distance ``None`` with a stated
    reason rather than defaulting it to zero.
    """

    _require_comparable(left, right)
    shared: Dict[str, Any] = {
        "dimension": left.dimension,
        "model_revision": left.model_revision,
        "preprocessing_hash": left.preprocessing_hash,
        "measurement_hash": left.measurement_hash,
        "asset_verification": _weaker_asset_verification(left, right),
        "left_clip_id": left.clip_id,
        "right_clip_id": right.clip_id,
    }
    if left.embedding == right.embedding:
        return ClipFeatureDistance(
            status=DISTANCE_COMPUTED,
            cosine_distance=0.0,
            l2_distance=0.0,
            identical_embeddings=True,
            **shared
        )
    left_norm = left.l2_norm
    right_norm = right.l2_norm
    squared = math.fsum((a - b) * (a - b) for a, b in zip(left.embedding, right.embedding))
    l2_distance = math.sqrt(squared)
    if left_norm == 0.0 or right_norm == 0.0:
        return ClipFeatureDistance(
            status=DISTANCE_UNDEFINED_ZERO_NORM,
            cosine_distance=None,
            l2_distance=l2_distance,
            identical_embeddings=False,
            reason=(
                "cosine distance is undefined when an embedding has zero norm; it is reported as unknown, "
                "not as zero"
            ),
            **shared
        )
    dot = math.fsum(a * b for a, b in zip(left.embedding, right.embedding))
    similarity = dot / (left_norm * right_norm)
    similarity = max(-1.0, min(1.0, similarity))
    return ClipFeatureDistance(
        status=DISTANCE_COMPUTED,
        cosine_distance=1.0 - similarity,
        l2_distance=l2_distance,
        identical_embeddings=False,
        **shared
    )


def _validate_control_ticks(control_ticks: Any, *, expected: int) -> Tuple[int, ...]:
    items = _as_list(control_ticks, what="control_ticks")
    if len(items) != expected:
        raise VJepaInputError(
            "control_ticks must label every compared clip: %d ticks for %d comparisons."
            % (len(items), expected)
        )
    values: List[int] = []
    previous: Optional[int] = None
    for position, value in enumerate(items):
        if isinstance(value, bool) or not isinstance(value, int):
            raise VJepaInputError("control_ticks[%d] must be an integer control tick." % position)
        if previous is not None and value <= previous:
            raise VJepaInputError(
                "control_ticks must be strictly increasing; control_ticks[%d] is not greater than its "
                "predecessor." % position
            )
        values.append(value)
        previous = value
    return tuple(values)


def teacher_forced_vs_free_running(
    teacher_forced: Sequence[FeatureVector],
    free_running: Sequence[FeatureVector],
    *,
    control_ticks: Sequence[int],
    action_sequence_id: str,
    generation_settings_id: str,
) -> TeacherForcedVsFreeRunningDivergence:
    """Feature divergence between paired teacher-forced and free-running clips.

    The two branches are paired positionally at equal control prefixes, as spec
    6.3 requires; unequal branch lengths are rejected rather than truncated.
    This measures *visual* divergence only.  Action alignment and integrity are
    separate measurements and are not inferred from these distances.
    """

    forced_clips = _as_list(teacher_forced, what="teacher_forced")
    free_clips = _as_list(free_running, what="free_running")
    if len(forced_clips) != len(free_clips):
        raise VJepaInputError(
            "Teacher-forced and free-running branches must pair one-to-one at equal control prefixes; got "
            "%d and %d clips. Refusing to truncate either branch."
            % (len(forced_clips), len(free_clips))
        )
    if not forced_clips:
        raise VJepaInputError("At least one control prefix is required to report a divergence.")
    ticks = _validate_control_ticks(control_ticks, expected=len(forced_clips))
    points = tuple(
        FeatureDistancePoint(control_tick=tick, distance=compare_clips(forced, free))
        for tick, forced, free in zip(ticks, forced_clips, free_clips)
    )
    return TeacherForcedVsFreeRunningDivergence(
        points=points,
        action_sequence_id=action_sequence_id,
        generation_settings_id=generation_settings_id,
    )


def clip_to_clip_drift_curve(
    reference: FeatureVector,
    clips: Sequence[FeatureVector],
    *,
    control_ticks: Sequence[int],
) -> FeatureDriftCurve:
    """Distance from one reference clip to later clips over control ticks.

    The x-axis is control ticks, as spec 6.3 requires.  A rising curve is an
    observation about encoder features, not a failure threshold and not a
    physical-fidelity claim.
    """

    compared = _as_list(clips, what="clips")
    if not compared:
        raise VJepaInputError("At least one compared clip is required to report a drift curve.")
    ticks = _validate_control_ticks(control_ticks, expected=len(compared))
    points = tuple(
        FeatureDistancePoint(control_tick=tick, distance=compare_clips(reference, clip))
        for tick, clip in zip(ticks, compared)
    )
    return FeatureDriftCurve(points=points, reference_clip_id=reference.clip_id)


# ---------------------------------------------------------------------------
# Structural separation evidence
# ---------------------------------------------------------------------------


def _module_source(name: str) -> Optional[str]:
    """Read a module's source without executing it."""

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return None
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or not str(origin).endswith(".py"):
        return None
    try:
        with open(str(origin), "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def _imported_module_names(source: str) -> Tuple[str, ...]:
    """Real import edges from the AST, so a docstring mention never counts."""

    names = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
                for alias in node.names:
                    names.add(node.module + "." + alias.name)
    return tuple(sorted(names))


def _imports_module(source: Optional[str], target: str) -> Optional[bool]:
    if source is None:
        return None
    imported = _imported_module_names(source)
    return any(name == target or name.startswith(target + ".") for name in imported)


def stage_a_separation_evidence() -> Dict[str, Any]:
    """Machine-checkable evidence that this diagnostic is outside Stage A.

    Both modules are located with ``importlib.util.find_spec``, which resolves
    a file path without executing the module, and their import edges are read
    from the AST.  An unresolvable source reports ``None``, never a convenient
    ``False``.
    """

    feature_source = _module_source(FEATURE_MODULE)
    stage_a_source = _module_source(STAGE_A_MODULE)
    return {
        "record_kind": "vjepa2_stage_a_separation_evidence",
        "feature_module": FEATURE_MODULE,
        "stage_a_module": STAGE_A_MODULE,
        "feature_module_source_resolved": feature_source is not None,
        "stage_a_source_resolved": stage_a_source is not None,
        "feature_module_imports_stage_a": _imports_module(feature_source, STAGE_A_MODULE),
        "stage_a_imports_feature_module": _imports_module(stage_a_source, FEATURE_MODULE),
        "provides_stage_a_hook": False,
        "accepts_or_returns_validity_report": False,
        "note": (
            "The Stage A gate's describe() reports vjepa_hook=None. That is not an unfilled slot: this module "
            "exposes no hook object, takes no ValidityReport, and returns no validity verdict or reason code, "
            "so a feature diagnostic cannot silently alter the deterministic gate (spec 5)."
        ),
    }


# ---------------------------------------------------------------------------
# The diagnostic
# ---------------------------------------------------------------------------


class VJepaFeatureDiagnostic:
    """Frozen ``facebook/vjepa2-vitl-fpc64-256`` encoder feature diagnostic.

    It embeds 64-frame clips and reports distances between them.  It produces
    no validity verdict, no success or failure label, and no progress score,
    and it has no path into the Stage A gate.
    """

    def __init__(
        self,
        profile: VJepaProfile,
        *,
        runtime_factory: Optional[Callable[[], _VJepaRuntime]] = None,
        model_factory: Optional[Callable[[VJepaProfile, _VJepaRuntime], Any]] = None,
        processor_factory: Optional[Callable[[VJepaProfile, _VJepaRuntime], Any]] = None,
    ) -> None:
        if not isinstance(profile, VJepaProfile):
            raise VJepaConfigurationError("profile must be a VJepaProfile.")
        self.profile = profile
        self._runtime_factory = runtime_factory
        self._model_factory = model_factory
        self._processor_factory = processor_factory
        self._runtime: Optional[_VJepaRuntime] = None
        self._model: Any = None
        self._processor: Any = None

    # -- identity ---------------------------------------------------------

    @property
    def parameters_hash(self) -> str:
        return self.profile.parameters_hash

    @property
    def preprocessing_hash(self) -> str:
        return self.profile.preprocessing.preprocessing_hash

    @property
    def measurement_parameters_hash(self) -> str:
        return self.profile.measurement_parameters_hash

    def measurement_hash(self, *, feature_api_used: str, pooling_applied: str) -> str:
        """Bind the measurement parameters to the feature API actually used."""

        return _canonical_hash(
            {
                "diagnostic_version": FEATURE_DIAGNOSTIC_VERSION,
                "measurement_parameters_hash": self.measurement_parameters_hash,
                "feature_api_used": feature_api_used,
                "pooling_applied": pooling_applied,
            }
        )

    def describe(self) -> Dict[str, Any]:
        """Deployment contract values for this diagnostic."""

        return _diagnostic_record(
            {
                "record_kind": "vjepa2_feature_diagnostic_contract",
                "diagnostic_version": FEATURE_DIAGNOSTIC_VERSION,
                "profile": self.profile.as_dict(),
                "parameters_hash": self.parameters_hash,
                "measurement_parameters_hash": self.measurement_parameters_hash,
                "preprocessing_hash": self.preprocessing_hash,
                "source_uri": VJEPA2_SOURCE_URI,
                "role": "frozen_video_feature_diagnostic",
                "is_a_rubric_judge": False,
                "separation": stage_a_separation_evidence(),
                "fixture_verification_required": FIXTURE_VERIFICATION_NOTE,
                "revision_note": VJEPA2_REVISION_UNRESOLVED_NOTE,
            }
        )

    # -- capability --------------------------------------------------------

    @property
    def _components_injected(self) -> bool:
        return self._model_factory is not None and self._processor_factory is not None

    def _asset_blocker(self) -> Optional[Dict[str, Any]]:
        """Named blocker describing an absent or unpinned asset, or ``None``."""

        root = self.profile.local_model_path
        if not _is_dir(root):
            return {
                "blocker": "asset_absent",
                "reason": (
                    "V-JEPA2 feature diagnostic is blocked: asset_absent. The pinned local directory %s does "
                    "not exist and Hub retrieval is disabled (local_files_only=True)." % root
                ),
                "missing_files": [item.path for item in self.profile.required_files],
                "size_mismatches": [],
            }
        missing: List[str] = []
        mismatches: List[Dict[str, Any]] = []
        for required in self.profile.required_files:
            observed = _file_size(os.path.join(root, required.path))
            if observed is None:
                missing.append(required.path)
            elif self.profile.verifies_candidate_bytes and observed != required.candidate_bytes:
                mismatches.append(
                    {
                        "path": required.path,
                        "observed_bytes": observed,
                        "candidate_bytes": required.candidate_bytes,
                    }
                )
        if missing:
            return {
                "blocker": "required_files_missing",
                "reason": (
                    "V-JEPA2 feature diagnostic is blocked: required_files_missing. %s is missing %s under %s; "
                    "no Hub download was attempted." % (VJEPA2_MODEL_ID, ", ".join(missing), root)
                ),
                "missing_files": missing,
                "size_mismatches": [],
            }
        if mismatches:
            listed = ", ".join(
                "%s is %d B where the spec candidate length is %d B"
                % (item["path"], item["observed_bytes"], item["candidate_bytes"])
                for item in mismatches
            )
            return {
                "blocker": "candidate_file_size_mismatch",
                "reason": (
                    "V-JEPA2 feature diagnostic is blocked: candidate_file_size_mismatch. %s. The spec "
                    "candidate byte lengths are the only provenance pin available for this asset." % listed
                ),
                "missing_files": [],
                "size_mismatches": mismatches,
            }
        return None

    def capability(self) -> CapabilityResult:
        """Report readiness. ``READY_UNQUALIFIED`` is not a fidelity claim."""

        details: Dict[str, Any] = {
            "profile_id": self.profile.profile_id,
            "model_id": self.profile.model_id,
            "role": "frozen_video_feature_diagnostic",
            "is_a_rubric_judge": False,
            "alters_validity_gate": False,
            "frames_per_clip": VJEPA2_FRAMES_PER_CLIP,
            "sampling_rule": self.profile.preprocessing.sampling_rule,
            "skipped_files": list(self.profile.skipped_files),
            "skipped_file_reason": VJEPA2_SKIPPED_FILE_REASON,
            "local_files_only": True,
            "asset_verification": self.profile.asset_verification,
            "limitation": FEATURE_DIAGNOSTIC_LIMITATION,
        }
        error = self.profile.profile_error()
        if error is not None:
            details["blocker"] = "profile_incomplete"
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="V-JEPA2 feature diagnostic is blocked: " + error + ".",
                source_verified=True,
                evidence_uris=(VJEPA2_SOURCE_URI,),
                details=details,
            )
        if self._components_injected:
            details["asset_presence_checked"] = False
            details["component_source"] = "injected_factories"
        else:
            details["asset_presence_checked"] = True
            details["component_source"] = "pinned_local_asset"
            blocker = self._asset_blocker()
            if blocker is not None:
                details.update(blocker)
                return CapabilityResult(
                    status=CapabilityStatus.BLOCKED,
                    reason=str(blocker["reason"]),
                    source_verified=True,
                    evidence_uris=(VJEPA2_SOURCE_URI,),
                    details=details,
                )
        details.update(
            {
                "model_revision": self.profile.model_revision,
                "processor_revision": self.profile.resolved_processor_revision,
                "transformers_version": self.profile.transformers_version,
                "torch_dtype": self.profile.torch_dtype,
                "parameters_hash": self.parameters_hash,
                "measurement_parameters_hash": self.measurement_parameters_hash,
                "preprocessing_hash": self.preprocessing_hash,
                "fixture_verification_required": FIXTURE_VERIFICATION_NOTE,
            }
        )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Pinned local V-JEPA2 feature extraction is configured with the documented 64-frame "
                "preprocessing and explicit timestamp sampling. It is an encoder diagnostic: it produces no "
                "validity verdict, no success label, and no progress score, and it is not calibrated against "
                "human judgement."
            ),
            source_verified=True,
            evidence_uris=(VJEPA2_SOURCE_URI,),
            details=details,
        )

    def _require_embeddable(self) -> None:
        result = self.capability()
        if result.status is not CapabilityStatus.READY_UNQUALIFIED:
            raise VJepaLoadError(
                "Refusing to embed a clip while the feature diagnostic is %s: %s"
                % (result.status.value, result.reason)
            )

    # -- lazy loading ------------------------------------------------------

    def _load_runtime(self) -> _VJepaRuntime:
        if self._runtime is not None:
            return self._runtime
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
            if runtime.transformers_version != self.profile.transformers_version:
                raise VJepaLoadError(
                    "Injected V-JEPA2 runtime version %r differs from profile-pinned %r."
                    % (runtime.transformers_version, self.profile.transformers_version)
                )
            self._runtime = runtime
            return runtime
        try:
            import torch  # type: ignore
            import transformers  # type: ignore
            from transformers import AutoModel, AutoVideoProcessor  # type: ignore
        except ImportError as error:
            raise VJepaLoadError(
                "V-JEPA2 feature dependencies are unavailable. Install the profile-pinned Transformers/Torch "
                "runtime in a feature-diagnostic image; imports here are lazy so this module stays importable "
                "on a CPU host."
            ) from error
        installed = str(getattr(transformers, "__version__", ""))
        if installed != self.profile.transformers_version:
            raise VJepaLoadError(
                "Installed Transformers %r differs from the V-JEPA2 profile pin %r; a different runtime is a "
                "different measurement." % (installed, self.profile.transformers_version)
            )
        self._runtime = _VJepaRuntime(
            torch=torch,
            model_cls=AutoModel,
            processor_cls=AutoVideoProcessor,
            transformers_version=installed,
        )
        return self._runtime

    def _ensure_components(self, runtime: _VJepaRuntime) -> Tuple[Any, Any]:
        error = self.profile.profile_error()
        if error is not None:
            raise VJepaLoadError("Refusing to load the V-JEPA2 encoder: " + error + ".")
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(self.profile, runtime)
            else:
                blocker = self._asset_blocker()
                if blocker is not None:
                    raise VJepaLoadError("Refusing to load the V-JEPA2 encoder: " + str(blocker["reason"]))
                dtype = getattr(runtime.torch, self.profile.torch_dtype, None)
                if dtype is None:
                    raise VJepaLoadError(
                        "The pinned Torch runtime lacks dtype %s." % self.profile.torch_dtype
                    )
                kwargs: Dict[str, Any] = {
                    "revision": self.profile.model_revision,
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "torch_dtype": dtype,
                }
                if self.profile.device_map is not None:
                    kwargs["device_map"] = self.profile.device_map
                self._model = runtime.model_cls.from_pretrained(self.profile.local_model_path, **kwargs)
            self._model = self._frozen_eval(self._model)
        if self._processor is None:
            if self._processor_factory is not None:
                self._processor = self._processor_factory(self.profile, runtime)
            else:
                self._processor = runtime.processor_cls.from_pretrained(
                    self.profile.local_model_path,
                    revision=self.profile.resolved_processor_revision,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            if not callable(self._processor):
                raise VJepaLoadError("The V-JEPA2 video processor is not callable for clip preprocessing.")
        return self._model, self._processor

    @staticmethod
    def _frozen_eval(model: Any) -> Any:
        """Put the encoder in eval mode; a frozen diagnostic has no dropout."""

        method = getattr(model, "eval", None)
        if callable(method):
            evaluated = method()
            if evaluated is not None:
                model = evaluated
        if not callable(getattr(model, "get_vision_features", None)) and not callable(model):
            raise VJepaLoadError(
                "The V-JEPA2 encoder exposes neither get_vision_features() nor a callable forward pass; "
                "refusing to invent a feature source."
            )
        return model

    @staticmethod
    def _inference_context(torch_module: Any) -> Any:
        function = getattr(torch_module, "inference_mode", None) or getattr(torch_module, "no_grad", None)
        return function() if callable(function) else contextlib.nullcontext()

    # -- embedding ---------------------------------------------------------

    def embed(
        self,
        frames: Sequence[Any],
        timestamps: Sequence[Any],
        *,
        clip_id: Optional[str] = None,
        sampling_rule: Optional[str] = None,
    ) -> FeatureVector:
        """Embed one clip under the documented 64-frame preprocessing.

        ``frames`` and ``timestamps`` describe the whole clip; the returned
        record states exactly which 64 indexes and control timestamps were
        used.  A clip that cannot supply 64 distinct frames is rejected.
        """

        clip_frames = _as_list(frames, what="frames")
        clip_timestamps = _as_list(timestamps, what="timestamps")
        if len(clip_frames) != len(clip_timestamps):
            raise VJepaInputError(
                "Each frame needs exactly one control timestamp; got %d frames and %d timestamps."
                % (len(clip_frames), len(clip_timestamps))
            )
        rule = sampling_rule or self.profile.preprocessing.sampling_rule
        sampling = plan_sampling(clip_timestamps, rule=rule)
        selected = [clip_frames[index] for index in sampling.frame_indices]
        if any(frame is None for frame in selected):
            raise VJepaInputError("A sampled frame is None; the diagnostic will not substitute a blank frame.")

        self._require_embeddable()
        runtime = self._load_runtime()
        model, processor = self._ensure_components(runtime)

        argument: Any = selected
        if self.profile.preprocessing.processor_call_style == PROCESSOR_BATCHED_VIDEO_LIST:
            argument = [selected]
        try:
            inputs = processor(argument, return_tensors="pt")
        except Exception as error:  # pragma: no cover - depends on the real processor
            raise VJepaLoadError(
                "The pinned V-JEPA2 video processor rejected a 64-frame clip under processor_call_style=%s. "
                "Certify the call style against a golden fixture rather than reshaping the clip here."
                % self.profile.preprocessing.processor_call_style
            ) from error
        if self.profile.device is not None:
            mover = getattr(inputs, "to", None)
            if callable(mover):
                inputs = mover(self.profile.device)

        raw, feature_api_used = self._vision_features(model, inputs, runtime)
        embedding, pooling_applied = _pool_features(raw)
        return FeatureVector(
            embedding=embedding,
            sampling=sampling,
            model_revision=self.profile.model_revision,
            preprocessing_hash=self.preprocessing_hash,
            parameters_hash=self.parameters_hash,
            measurement_hash=self.measurement_hash(
                feature_api_used=feature_api_used, pooling_applied=pooling_applied
            ),
            feature_api_used=feature_api_used,
            pooling_applied=pooling_applied,
            asset_verification=self.profile.asset_verification,
            clip_id=clip_id,
            model_id=self.profile.model_id,
        )

    def _vision_features(self, model: Any, inputs: Any, runtime: _VJepaRuntime) -> Tuple[Any, str]:
        """Call the documented feature API and record which one produced it."""

        getter = getattr(model, "get_vision_features", None)
        with self._inference_context(runtime.torch):
            if callable(getter):
                output = getter(**inputs) if isinstance(inputs, Mapping) else getter(inputs)
                return output, FEATURE_API_VISION_FEATURES
            output = model(**inputs) if isinstance(inputs, Mapping) else model(inputs)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, Mapping):
            hidden = output.get("last_hidden_state")
        if hidden is None:
            raise VJepaLoadError(
                "The V-JEPA2 encoder returned neither get_vision_features() output nor last_hidden_state; "
                "refusing to invent a feature source."
            )
        return hidden, FEATURE_API_LAST_HIDDEN_STATE

    # -- comparison utilities (delegate to the module-level functions) ----

    def compare_clips(self, left: FeatureVector, right: FeatureVector) -> ClipFeatureDistance:
        """Cosine and L2 distance between two clips. See :func:`compare_clips`."""

        return compare_clips(left, right)

    def teacher_forced_vs_free_running(
        self,
        teacher_forced: Sequence[FeatureVector],
        free_running: Sequence[FeatureVector],
        *,
        control_ticks: Sequence[int],
        action_sequence_id: str,
        generation_settings_id: str,
    ) -> TeacherForcedVsFreeRunningDivergence:
        """See :func:`teacher_forced_vs_free_running`."""

        return teacher_forced_vs_free_running(
            teacher_forced,
            free_running,
            control_ticks=control_ticks,
            action_sequence_id=action_sequence_id,
            generation_settings_id=generation_settings_id,
        )

    def clip_to_clip_drift_curve(
        self,
        reference: FeatureVector,
        clips: Sequence[FeatureVector],
        *,
        control_ticks: Sequence[int],
    ) -> FeatureDriftCurve:
        """See :func:`clip_to_clip_drift_curve`."""

        return clip_to_clip_drift_curve(reference, clips, control_ticks=control_ticks)

    def stage_a_separation_evidence(self) -> Dict[str, Any]:
        """See :func:`stage_a_separation_evidence`."""

        return stage_a_separation_evidence()


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def _materialise(value: Any) -> Any:
    """Detach/move/convert a tensor-like object into nested Python values."""

    for name in ("detach", "cpu", "tolist"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                value = method()
            except (TypeError, RuntimeError, ValueError):
                continue
    return value


def _nested_depth(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            raise VJepaLoadError("The V-JEPA2 encoder returned an empty feature axis; refusing to pool nothing.")
        depths = {_nested_depth(item) for item in value}
        if len(depths) != 1:
            raise VJepaLoadError("The V-JEPA2 feature tensor is ragged; refusing to pool an ambiguous shape.")
        return 1 + depths.pop()
    return 0


def _row_to_floats(row: Sequence[Any]) -> Tuple[float, ...]:
    values: List[float] = []
    for value in row:
        if isinstance(value, bool) or isinstance(value, (list, tuple)):
            raise VJepaLoadError("V-JEPA2 feature components must be real numbers.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise VJepaLoadError("V-JEPA2 feature components must be real numbers.") from error
        if not math.isfinite(numeric):
            raise VJepaLoadError(
                "The V-JEPA2 encoder produced a non-finite feature component; refusing to record it as a number."
            )
        values.append(numeric)
    if not values:
        raise VJepaLoadError("The V-JEPA2 encoder returned an empty feature vector.")
    return tuple(values)


def _pool_features(raw: Any) -> Tuple[Tuple[float, ...], str]:
    """Pool an encoder output into one clip vector, stating what was pooled.

    Accepted shapes are ``(1, tokens, dim)``, ``(1, dim)`` and ``(dim,)``.  A
    leading axis other than a single clip is rejected rather than guessed at:
    choosing an axis for the caller is how a feature number stops being
    traceable.
    """

    value = _materialise(raw)
    depth = _nested_depth(value)
    if depth == 1:
        return _row_to_floats(value), POOLING_IDENTITY_VECTOR
    if depth == 2:
        if len(value) != 1:
            raise VJepaLoadError(
                "Expected one clip in the leading axis of a 2-D V-JEPA2 feature tensor, got %d; refusing to "
                "guess which axis is tokens." % len(value)
            )
        return _row_to_floats(value[0]), POOLING_IDENTITY_VECTOR
    if depth == 3:
        if len(value) != 1:
            raise VJepaLoadError(
                "Expected one clip in the batch axis of a 3-D V-JEPA2 feature tensor, got %d." % len(value)
            )
        rows = [_row_to_floats(row) for row in value[0]]
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise VJepaLoadError("V-JEPA2 token features have inconsistent widths; refusing to pool them.")
        token_count = len(rows)
        pooled = tuple(math.fsum(row[column] for row in rows) / token_count for column in range(width))
        return pooled, POOLING_MEAN_OVER_TOKENS
    raise VJepaLoadError(
        "A %d-dimensional V-JEPA2 feature tensor has no declared pooling rule; refusing to invent one." % depth
    )


__all__ = [
    "ASSET_BYTES_DEVELOPMENT",
    "ASSET_BYTES_VERIFIED",
    "ClipFeatureDistance",
    "DISTANCE_COMPUTED",
    "DISTANCE_UNDEFINED_ZERO_NORM",
    "FEATURE_DIAGNOSTIC_LIMITATION",
    "FEATURE_DIAGNOSTIC_VERSION",
    "FeatureDistancePoint",
    "FeatureDriftCurve",
    "FeatureVector",
    "SAMPLING_CONTIGUOUS_PREFIX",
    "SAMPLING_UNIFORM_INCLUSIVE",
    "TeacherForcedVsFreeRunningDivergence",
    "VJEPA2_FRAMES_PER_CLIP",
    "VJEPA2_MODEL_ID",
    "VJEPA2_REQUIRED_FILES",
    "VJEPA2_SKIPPED_FILES",
    "VJepaConfigurationError",
    "VJepaFeatureDiagnostic",
    "VJepaInputError",
    "VJepaLoadError",
    "VJepaPreprocessing",
    "VJepaProfile",
    "VJepaRequiredFile",
    "VJepaSampling",
    "clip_to_clip_drift_curve",
    "compare_clips",
    "plan_sampling",
    "stage_a_separation_evidence",
    "teacher_forced_vs_free_running",
]
