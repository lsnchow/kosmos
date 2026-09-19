"""Local, blinded Qwen2.5-VL rubric judge with strict bounded aggregation.

The VLM receives only a fixed 16-frame video, control timestamps, the exact
task instruction, a frozen task rubric, and provenance-backed reference
images.  Its input object has no fields for policy identity, action commands,
world condition, gate status, or published reference percentages.  Invalid
outputs and disagreement produce a nullable outcome; they are never coerced
into a failure or success.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from plumb.adapters.contracts import CapabilityResult, CapabilityStatus

from .provenance import canonical_json_sha256, file_source_sha256, image_pixel_hash
from .tasks import BENCHMARK_TASK_REGISTRY, TASK_REGISTRY_HASH, BenchmarkTask


QWEN_JUDGE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_JUDGE_SOURCE = "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
_PERCENTAGE_PATTERN = re.compile(r"(?:\bpercent\b|\d+(?:\.\d+)?\s*%)", re.IGNORECASE)
_REQUIRED_OUTPUT_FIELDS = frozenset(
    {
        "integrity",
        "collision",
        "progress",
        "completion_evidence",
        "evidence_frame_indices",
        "observable_reasons",
    }
)
_GENERATION_RNG_LOCK = threading.RLock()
PROMPT_SYSTEM_CONTRACT_VERSION = "qwen-rubric-output-contract-v2"
_BASE_SYSTEM_CONTRACT = (
    "Judge only the supplied video, task instruction, rubric, and reference images. Use visible evidence only. "
    "Return exactly one raw JSON object and nothing else: no Markdown code fence, heading, explanation, or surrounding text. "
    "Its exact keys are integrity, collision, progress, completion_evidence, evidence_frame_indices, observable_reasons. "
    "integrity is intact, artifact, or uncertain. collision is none_visible, visible, or uncertain. "
    "progress is an integer from 0 through 5 or null. completion_evidence is met, not_met, or uncertain. "
    "met requires progress 5; not_met requires progress 0 through 4. evidence_frame_indices is a nonempty JSON array "
    "of integer indexes 0 through 15. observable_reasons is one nonempty JSON STRING of at most 1200 characters."
)
_FORMAT_ONLY_RETRY_REMINDER = (
    " FORMAT_ONLY retry reminder: emit the required raw JSON object directly; observable_reasons must be one JSON string, "
    "not an array or object."
)


class JudgeLoadError(RuntimeError):
    """The local pinned judge cannot be loaded safely in this environment."""


class JudgeInputError(ValueError):
    """The fixed judge protocol input is malformed or contains barred data."""


class JudgeSchemaError(ValueError):
    """A model response did not satisfy the frozen rubric JSON schema."""


class JudgeRefusalError(ValueError):
    """A textual model refusal is not a schema/transport retry candidate."""


@dataclass(frozen=True)
class QwenJudgeProfile:
    """One local, revision-pinned Qwen judge deployment profile."""

    profile_id: str
    local_model_path: str
    model_revision: str
    processor_revision: str
    transformers_version: str = "4.49.0"
    local_files_only: bool = True
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    device_map: Optional[str] = "auto"
    container_digest: Optional[str] = None
    asset_manifest_id: Optional[str] = None
    asset_manifest_sha256: Optional[str] = None
    runtime_lock_id: Optional[str] = None
    runtime_lock_sha256: Optional[str] = None

    def profile_error(self) -> Optional[str]:
        if not self.profile_id:
            return "profile_id is required"
        if not self.local_model_path:
            return "local_model_path is required; Hub fetching is disabled"
        if not _immutable_revision(self.model_revision):
            return "model_revision must be an immutable 40-character hexadecimal revision"
        if not _immutable_revision(self.processor_revision):
            return "processor_revision must be an immutable 40-character hexadecimal revision"
        if not isinstance(self.transformers_version, str) or not self.transformers_version:
            return "transformers_version must be an explicit pinned runtime version"
        if not self.local_files_only:
            return "judge loader is local-only; network retrieval is not permitted"
        if self.trust_remote_code:
            return "Qwen judge must use the native pinned Transformers architecture, not trust_remote_code"
        if self.torch_dtype not in ("bfloat16", "float16", "float32"):
            return "torch_dtype must be bfloat16, float16, or float32"
        return None


@dataclass(frozen=True)
class JudgeSamplingConfig:
    """Frozen current primary sampling operating point."""

    sample_count: int = 5
    quorum: int = 3
    temperature: float = 0.7
    top_p: float = 1.0
    max_new_tokens: int = 512
    retries_per_sample: int = 1

    def validate(self) -> None:
        if self.sample_count != 5 or self.quorum != 3:
            raise JudgeInputError("Primary judge protocol is fixed at five samples with a quorum of three.")
        if self.temperature != 0.7 or self.top_p != 1.0:
            raise JudgeInputError("Primary judge sampler is frozen at temperature=0.7 and top_p=1.0.")
        if self.max_new_tokens != 512:
            raise JudgeInputError("Primary judge output cap is frozen at 512 tokens.")
        if self.retries_per_sample != 1:
            raise JudgeInputError("Primary judge permits exactly one bounded retry per sample.")


@dataclass(frozen=True)
class ReferenceImage:
    """An opaque image plus source provenance required by the rubric protocol."""

    image: Any
    source_uri: str
    sha256: str

    def validate(self) -> None:
        if self.image is None:
            raise JudgeInputError("Reference image cannot be null.")
        if not isinstance(self.source_uri, str) or not self.source_uri:
            raise JudgeInputError("Reference image requires a provenance source_uri.")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise JudgeInputError("Reference image requires a 64-character SHA-256 digest.")


@dataclass(frozen=True)
class JudgeInputProvenance:
    """Caller-supplied artifact identity for a primary-scoring judge request.

    Values can be absent for an exploratory/local call.  Their absence does
    not make a video false, but it makes the resulting report Gate-D
    unavailable.  ``video_sha256`` identifies the original clip artifact;
    frame pixel hashes are recomputed locally by the judge and cannot be
    supplied by the caller.
    """

    clip_id: Optional[str] = None
    video_sha256: Optional[str] = None
    protocol_id: Optional[str] = None
    calibration_manifest_hash: Optional[str] = None

    def validate_present_values(self) -> None:
        if self.clip_id is not None and (not isinstance(self.clip_id, str) or not self.clip_id.strip()):
            raise JudgeInputError("clip_id must be a nonempty string when supplied.")
        if self.protocol_id is not None and (not isinstance(self.protocol_id, str) or not self.protocol_id.strip()):
            raise JudgeInputError("protocol_id must be a nonempty string when supplied.")
        if self.video_sha256 is not None and _normalise_sha256(self.video_sha256) is None:
            raise JudgeInputError("video_sha256 must be a SHA-256 digest when supplied.")
        if self.calibration_manifest_hash is not None and _normalise_sha256(self.calibration_manifest_hash) is None:
            raise JudgeInputError("calibration_manifest_hash must be a SHA-256 digest when supplied.")


@dataclass(frozen=True)
class JudgeRequest:
    """Exactly the blinded information permitted to reach the Qwen judge.

    This type intentionally excludes policy/action/reference-rate fields.
    Keeping reference image provenance separate from the VLM message lets the
    run ledger retain the evidence without exposing source labels or rates to
    the model.
    """

    frames: Tuple[Any, ...]
    frame_timestamps: Tuple[float, ...]
    reference_images: Tuple[ReferenceImage, ...]
    task_id: Optional[str] = None
    diagnostic_mode: bool = False
    task_instruction: Optional[str] = None
    task_rubric: Optional[str] = None
    provenance: Optional[JudgeInputProvenance] = None

    def validate(self) -> None:
        if len(self.frames) != 16:
            raise JudgeInputError(
                "Judge requires exactly 16 pre-sampled frames including the original start and final control tick; "
                "it will not pad, drop, or resample a clip."
            )
        if any(frame is None for frame in self.frames):
            raise JudgeInputError("Judge frames cannot contain null values.")
        if len(self.frame_timestamps) != 16:
            raise JudgeInputError("Judge needs one nominal control timestamp for every one of 16 frames.")
        previous: Optional[float] = None
        for timestamp in self.frame_timestamps:
            if isinstance(timestamp, bool):
                raise JudgeInputError("Frame timestamps must be finite numeric control times.")
            try:
                numeric = float(timestamp)
            except (TypeError, ValueError) as error:
                raise JudgeInputError("Frame timestamps must be finite numeric control times.") from error
            if not math.isfinite(numeric):
                raise JudgeInputError("Frame timestamps must be finite numeric control times.")
            if previous is not None and numeric <= previous:
                raise JudgeInputError("Frame timestamps must be strictly increasing control timestamps.")
            previous = numeric
        if self.diagnostic_mode:
            if self.task_id is not None:
                raise JudgeInputError("Diagnostic judge requests use explicit text only; omit task_id.")
            if not isinstance(self.task_instruction, str) or not self.task_instruction.strip():
                raise JudgeInputError("Diagnostic judge task_instruction must be a nonempty explicit string.")
            if not isinstance(self.task_rubric, str) or not self.task_rubric.strip():
                raise JudgeInputError("Diagnostic judge task_rubric must be a nonempty explicit string.")
            if _PERCENTAGE_PATTERN.search(self.task_instruction) or _PERCENTAGE_PATTERN.search(self.task_rubric):
                raise JudgeInputError("Reference success percentages must never be included in a judge instruction or rubric.")
        else:
            if self.task_instruction is not None or self.task_rubric is not None:
                raise JudgeInputError(
                    "Primary scoring accepts a canonical task_id only; free-text task instructions/rubrics are diagnostic-only."
                )
            if not isinstance(self.task_id, str) or not self.task_id:
                raise JudgeInputError("Primary scoring requires one canonical TaskRegistry task_id.")
            try:
                BENCHMARK_TASK_REGISTRY.get(self.task_id)
            except KeyError as error:
                raise JudgeInputError("Unknown primary TaskRegistry task_id %r." % self.task_id) from error
        if not self.reference_images:
            raise JudgeInputError("Judge requires at least one provenance-backed goal/reference image.")
        for reference in self.reference_images:
            if not isinstance(reference, ReferenceImage):
                raise JudgeInputError("reference_images must be ReferenceImage values with immutable provenance.")
            reference.validate()
        if self.provenance is not None:
            if not isinstance(self.provenance, JudgeInputProvenance):
                raise JudgeInputError("provenance must be a JudgeInputProvenance value when supplied.")
            self.provenance.validate_present_values()

    @property
    def is_primary(self) -> bool:
        return not self.diagnostic_mode

    def resolved_task(self) -> Optional[BenchmarkTask]:
        if not self.is_primary:
            return None
        assert self.task_id is not None
        return BENCHMARK_TASK_REGISTRY.get(self.task_id)

    def resolved_instruction_and_rubric(self) -> Tuple[str, str]:
        task = self.resolved_task()
        if task is not None:
            return task.instruction, task.rubric
        assert self.task_instruction is not None and self.task_rubric is not None
        return self.task_instruction, self.task_rubric


@dataclass(frozen=True)
class RubricSample:
    integrity: str
    collision: str
    progress: Optional[int]
    completion_evidence: str
    evidence_frame_indices: Tuple[int, ...]
    observable_reasons: str

    @property
    def decisive(self) -> bool:
        return (
            self.integrity == "intact"
            and self.progress is not None
            and self.completion_evidence in ("met", "not_met")
        )

    @property
    def binary_success(self) -> Optional[bool]:
        if not self.decisive:
            return None
        return self.completion_evidence == "met"

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "integrity": self.integrity,
            "collision": self.collision,
            "progress": self.progress,
            "completion_evidence": self.completion_evidence,
            "evidence_frame_indices": list(self.evidence_frame_indices),
            "observable_reasons": self.observable_reasons,
        }


@dataclass(frozen=True)
class JudgeAttempt:
    sample_index: int
    seed: int
    attempt_index: int
    raw_output: Optional[str]
    parsed: Optional[RubricSample]
    failure_reason: Optional[str]
    wall_seconds: Optional[float]
    gpu_peak_memory_bytes: Optional[int]
    prompt_variant: str = "base"
    prompt_source_hash: Optional[str] = None
    transport_normalization: str = "none"

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "sample_index": self.sample_index,
            "seed": self.seed,
            "attempt_index": self.attempt_index,
            "raw_output": self.raw_output,
            "parsed": None if self.parsed is None else self.parsed.as_dict(),
            "failure_reason": self.failure_reason,
            "wall_seconds": self.wall_seconds,
            "gpu_peak_memory_bytes": self.gpu_peak_memory_bytes,
            "prompt_variant": self.prompt_variant,
            "prompt_source_hash": self.prompt_source_hash,
            "transport_normalization": self.transport_normalization,
        }


@dataclass(frozen=True)
class JudgeSampleReport:
    sample_index: int
    seed: int
    attempts: Tuple[JudgeAttempt, ...]

    @property
    def final_sample(self) -> Optional[RubricSample]:
        if not self.attempts:
            return None
        return self.attempts[-1].parsed

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "sample_index": self.sample_index,
            "seed": self.seed,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class JudgeReport:
    """Episode-level bounded judge outcome.

    ``binary_success`` and ``progress`` are null when the VLM result is not
    evaluable.  ``judge_status`` is intentionally ``unknown`` rather than a
    fabricated negative outcome for invalid, refused, or disagreeing samples.
    """

    binary_success: Optional[bool]
    progress: Optional[int]
    judge_status: str
    missing_reason: Optional[str]
    agreeing_samples: int
    sample_reports: Tuple[JudgeSampleReport, ...]
    wall_seconds: Optional[float]
    gpu_peak_memory_bytes: Optional[int]
    sampling: JudgeSamplingConfig = JudgeSamplingConfig()
    provenance: Optional[Mapping[str, Any]] = None

    def validate_aggregation(self) -> None:
        """Recompute the outcome from raw samples; reject forged summary fields."""

        expected = _aggregate_sample_reports(self.sample_reports, self.sampling)
        actual = (
            self.binary_success,
            self.progress,
            self.judge_status,
            self.missing_reason,
            self.agreeing_samples,
        )
        if expected != actual:
            raise JudgeSchemaError(
                "JudgeReport aggregation does not match the raw sample reports; consumers must not trust summary fields."
            )

    def gate_d_payload(self) -> Mapping[str, Any]:
        """Return report JSON only after aggregation/provenance Gate-D checks.

        This verifies internal PLUMB report consistency.  It is not a signed
        attestation of data origin, model provenance, or calibration quality.
        """

        self.validate_aggregation()
        trust = self.provenance.get("trust") if isinstance(self.provenance, Mapping) else None
        if not isinstance(trust, Mapping) or trust.get("gate_d_eligible") is not True:
            raise JudgeInputError("Judge report is Gate-D unavailable because required provenance is missing or diagnostic.")
        return self.as_dict()

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "binary_success": self.binary_success,
            "progress": self.progress,
            "judge_status": self.judge_status,
            "missing_reason": self.missing_reason,
            "agreeing_samples": self.agreeing_samples,
            "samples": [sample.as_dict() for sample in self.sample_reports],
            "raw_judge_samples": [sample.as_dict() for sample in self.sample_reports],
            "sampling": {
                "sample_count": self.sampling.sample_count,
                "quorum": self.sampling.quorum,
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                "max_new_tokens": self.sampling.max_new_tokens,
                "retries_per_sample": self.sampling.retries_per_sample,
            },
            "provenance": _json_safe_mapping(self.provenance),
            "timing": {
                "wall_seconds": self.wall_seconds,
                "gpu_peak_memory_bytes": self.gpu_peak_memory_bytes,
            },
        }


@dataclass(frozen=True)
class _QwenRuntime:
    torch: Any
    model_cls: Any
    processor_cls: Any
    transformers_version: str


def _immutable_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _normalise_sha256(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        return None
    return "sha256:" + text


def _json_safe_mapping(value: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _prompt_source_hash(format_retry: bool) -> str:
    return canonical_json_sha256(
        {
            "version": PROMPT_SYSTEM_CONTRACT_VERSION,
            "base_system_contract": _BASE_SYSTEM_CONTRACT,
            "format_only_retry_reminder": _FORMAT_ONLY_RETRY_REMINDER if format_retry else None,
        }
    )


def _missing_reason_for_samples(samples: Sequence[JudgeSampleReport]) -> str:
    parsed = [sample.final_sample for sample in samples if sample.final_sample is not None]
    decisive = [sample for sample in parsed if sample is not None and sample.decisive]
    if not parsed:
        if any(
            attempt.failure_reason == "judge_refusal"
            for sample in samples
            for attempt in sample.attempts
        ):
            return "judge_refusal"
        return "judge_sample_failure"
    if not decisive:
        return "judge_uncertain"
    votes = {True: 0, False: 0}
    for sample in decisive:
        assert sample.binary_success is not None
        votes[sample.binary_success] += 1
    if votes[True] and votes[False]:
        return "judge_disagreement"
    return "judge_insufficient_quorum"


def _aggregate_sample_reports(
    reports: Sequence[JudgeSampleReport], sampling: JudgeSamplingConfig
) -> Tuple[Optional[bool], Optional[int], str, Optional[str], int]:
    """Pure, repeatable five-sample aggregation used by producer and consumer."""

    sampling.validate()
    if len(reports) != sampling.sample_count:
        raise JudgeSchemaError(
            "Judge report contains %d raw samples; frozen primary protocol requires %d."
            % (len(reports), sampling.sample_count)
        )
    expected_indices = tuple(range(sampling.sample_count))
    actual_indices = tuple(report.sample_index for report in reports)
    if actual_indices != expected_indices:
        raise JudgeSchemaError("Raw judge sample indexes must be exactly 0 through 4 in order.")
    seeds = tuple(report.seed for report in reports)
    if len(set(seeds)) != sampling.sample_count:
        raise JudgeSchemaError("Raw judge samples must carry five distinct logged seeds.")
    for report in reports:
        if not report.attempts or len(report.attempts) > sampling.retries_per_sample + 1:
            raise JudgeSchemaError("Each raw judge sample must retain one attempt plus at most one bounded retry.")
        if any(attempt.sample_index != report.sample_index or attempt.seed != report.seed for attempt in report.attempts):
            raise JudgeSchemaError("Raw judge attempt identity must match its sample identity.")
    decisive = []
    for report in reports:
        sample = report.final_sample
        if sample is not None and sample.decisive:
            decisive.append(sample)
    grouped = {True: [], False: []}
    for sample in decisive:
        assert sample.binary_success is not None and sample.progress is not None
        grouped[sample.binary_success].append(sample)
    winner: Optional[bool] = None
    winner_samples: Sequence[RubricSample] = ()
    for outcome in (True, False):
        if len(grouped[outcome]) >= sampling.quorum:
            winner = outcome
            winner_samples = grouped[outcome]
            break
    if winner is None:
        return (
            None,
            None,
            "unknown",
            _missing_reason_for_samples(reports),
            max(len(grouped[True]), len(grouped[False])),
        )
    median_progress = int(statistics.median_low([sample.progress for sample in winner_samples if sample.progress is not None]))
    return (winner, median_progress, "evaluable", None, len(winner_samples))


def _safe_pixel_hashes(images: Sequence[Any]) -> Tuple[Tuple[Optional[str], ...], Tuple[Optional[str], ...], Tuple[str, ...]]:
    hashes = []
    semantics = []
    problems = []
    for index, image in enumerate(images):
        try:
            record = image_pixel_hash(image)
            hashes.append(record["sha256"])
            semantics.append(record["semantics"])
        except (TypeError, ValueError) as error:
            hashes.append(None)
            semantics.append(None)
            problems.append("pixel_hash_unavailable_%d:%s" % (index, type(error).__name__))
    return tuple(hashes), tuple(semantics), tuple(problems)


def _normalise_json_transport(raw_output: str) -> Tuple[str, str]:
    """Accept only a bare object or one exactly anchored ``json`` code fence.

    The fence exception is lossless transport normalization for a known model
    wrapper.  It does not extract a substring, repair fields, permit prose, or
    accept multiple objects.  Its version is retained in each raw attempt.
    """

    if not isinstance(raw_output, str):
        raise JudgeSchemaError("Judge output must be text containing exactly one JSON object.")
    if raw_output.startswith("```json\n") and raw_output.endswith("\n```"):
        body = raw_output[len("```json\n") : -len("\n```")]
        if not body:
            raise JudgeSchemaError("JSON code fence was empty.")
        return body, "json_code_fence_v1"
    if raw_output.startswith("```json\r\n") and raw_output.endswith("\r\n```"):
        body = raw_output[len("```json\r\n") : -len("\r\n```")]
        if not body:
            raise JudgeSchemaError("JSON code fence was empty.")
        return body, "json_code_fence_v1"
    if raw_output.startswith("```") or raw_output.endswith("```"):
        raise JudgeSchemaError("Only one exactly anchored ```json code fence is accepted as transport normalization.")
    return raw_output, "none"


def _parse_rubric_json_with_transport(raw_output: str) -> Tuple[RubricSample, str]:
    """Parse frozen schema after lossless, versioned transport normalization."""

    normalized, transport_normalization = _normalise_json_transport(raw_output)
    try:
        payload = json.loads(normalized)
    except (TypeError, ValueError) as error:
        raise JudgeSchemaError("Judge output was not valid JSON.") from error
    if not isinstance(payload, dict):
        raise JudgeSchemaError("Judge output must be a JSON object.")
    fields = frozenset(payload)
    if fields != _REQUIRED_OUTPUT_FIELDS:
        missing = sorted(_REQUIRED_OUTPUT_FIELDS - fields)
        unexpected = sorted(fields - _REQUIRED_OUTPUT_FIELDS)
        raise JudgeSchemaError("Judge output fields differ from frozen schema (missing=%s, unexpected=%s)." % (missing, unexpected))
    integrity = payload["integrity"]
    collision = payload["collision"]
    completion = payload["completion_evidence"]
    if integrity not in ("intact", "artifact", "uncertain"):
        raise JudgeSchemaError("integrity must be intact, artifact, or uncertain.")
    if collision not in ("none_visible", "visible", "uncertain"):
        raise JudgeSchemaError("collision must be none_visible, visible, or uncertain.")
    if completion not in ("met", "not_met", "uncertain"):
        raise JudgeSchemaError("completion_evidence must be met, not_met, or uncertain.")
    progress_value = payload["progress"]
    if progress_value is not None and (isinstance(progress_value, bool) or not isinstance(progress_value, int)):
        raise JudgeSchemaError("progress must be an integer 0 through 5 or null.")
    if progress_value is not None and not 0 <= progress_value <= 5:
        raise JudgeSchemaError("progress must be an integer 0 through 5 or null.")
    if completion == "met" and progress_value != 5:
        raise JudgeSchemaError("completion_evidence='met' requires progress=5.")
    if completion == "not_met" and (progress_value is None or not 0 <= progress_value <= 4):
        raise JudgeSchemaError("completion_evidence='not_met' requires progress from 0 through 4.")
    frame_indices = payload["evidence_frame_indices"]
    if not isinstance(frame_indices, list) or not frame_indices:
        raise JudgeSchemaError("evidence_frame_indices must be a nonempty list of frame indexes.")
    if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 16 for index in frame_indices):
        raise JudgeSchemaError("evidence_frame_indices must contain only integer indexes 0 through 15.")
    reasons = payload["observable_reasons"]
    if not isinstance(reasons, str) or not reasons.strip() or len(reasons) > 1200:
        raise JudgeSchemaError("observable_reasons must be a concise nonempty string (at most 1200 characters).")
    return RubricSample(
        integrity=integrity,
        collision=collision,
        progress=progress_value,
        completion_evidence=completion,
        evidence_frame_indices=tuple(frame_indices),
        observable_reasons=reasons.strip(),
    ), transport_normalization


def parse_rubric_json(raw_output: str) -> RubricSample:
    """Public strict parser; transport metadata is available in ``JudgeAttempt``."""

    sample, _transport_normalization = _parse_rubric_json_with_transport(raw_output)
    return sample


class QwenRubricJudge:
    """A local-only Qwen2.5-VL judge using the fixed five-sample protocol."""

    def __init__(
        self,
        profile: QwenJudgeProfile,
        *,
        sampling: JudgeSamplingConfig = JudgeSamplingConfig(),
        runtime_factory: Optional[Callable[[], _QwenRuntime]] = None,
        model_factory: Optional[Callable[[QwenJudgeProfile, _QwenRuntime], Any]] = None,
        processor_factory: Optional[Callable[[QwenJudgeProfile, _QwenRuntime], Any]] = None,
    ) -> None:
        sampling.validate()
        self.profile = profile
        self.sampling = sampling
        self._runtime_factory = runtime_factory
        self._model_factory = model_factory
        self._processor_factory = processor_factory
        self._runtime: Optional[_QwenRuntime] = None
        self._model: Any = None
        self._processor: Any = None

    def capability(self) -> CapabilityResult:
        error = self.profile.profile_error()
        if error is not None:
            return CapabilityResult(
                status=CapabilityStatus.BLOCKED,
                reason="Qwen judge loading is blocked: " + error + ".",
                source_verified=True,
                evidence_uris=(QWEN_JUDGE_SOURCE,),
                details={"profile_id": self.profile.profile_id},
            )
        if not _is_dir(self.profile.local_model_path) and (self._model_factory is None or self._processor_factory is None):
            return CapabilityResult(
                status=CapabilityStatus.UNAVAILABLE,
                reason="Qwen judge local model directory is absent; no Hub download was attempted.",
                source_verified=True,
                evidence_uris=(QWEN_JUDGE_SOURCE,),
                details={"local_model_path": self.profile.local_model_path, "local_files_only": True},
            )
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Pinned local Qwen2.5-VL rubric inference is configured with the frozen 5-sample/3-quorum protocol. "
                "Gate D calibration is still required before primary scoring."
            ),
            source_verified=True,
            evidence_uris=(QWEN_JUDGE_SOURCE,),
            details={
                "model_revision": self.profile.model_revision,
                "processor_revision": self.profile.processor_revision,
                "transformers_version": self.profile.transformers_version,
                "sample_count": self.sampling.sample_count,
                "quorum": self.sampling.quorum,
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                "max_new_tokens": self.sampling.max_new_tokens,
                "local_files_only": True,
                "trust_remote_code": False,
                "asset_manifest_id": self.profile.asset_manifest_id,
                "asset_manifest_sha256": self.profile.asset_manifest_sha256,
                "runtime_lock_id": self.profile.runtime_lock_id,
                "runtime_lock_sha256": self.profile.runtime_lock_sha256,
            },
        )

    def _load_runtime(self) -> _QwenRuntime:
        if self._runtime is not None:
            return self._runtime
        if self._runtime_factory is not None:
            runtime = self._runtime_factory()
            if runtime.transformers_version != self.profile.transformers_version:
                raise JudgeLoadError(
                    "Injected Qwen runtime version %r differs from profile-pinned %r."
                    % (runtime.transformers_version, self.profile.transformers_version)
                )
            self._runtime = runtime
            return runtime
        try:
            import torch  # type: ignore
            import transformers  # type: ignore
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
        except ImportError as error:
            raise JudgeLoadError(
                "Qwen judge dependencies are unavailable. Install the profile-pinned Transformers/Torch runtime "
                "in a judge image; imports are lazy."
            ) from error
        installed = str(getattr(transformers, "__version__", ""))
        if installed != self.profile.transformers_version:
            raise JudgeLoadError(
                "Installed Transformers %r differs from Qwen profile-pinned %r."
                % (installed, self.profile.transformers_version)
            )
        self._runtime = _QwenRuntime(
            torch=torch,
            model_cls=Qwen2_5_VLForConditionalGeneration,
            processor_cls=AutoProcessor,
            transformers_version=installed,
        )
        return self._runtime

    def _ensure_components(self, runtime: _QwenRuntime) -> Tuple[Any, Any]:
        error = self.profile.profile_error()
        if error is not None:
            raise JudgeLoadError("Refusing to load Qwen judge: " + error + ".")
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(self.profile, runtime)
            else:
                if not _is_dir(self.profile.local_model_path):
                    raise JudgeLoadError(
                        "Expected local Qwen judge checkpoint at %s; refusing network retrieval."
                        % self.profile.local_model_path
                    )
                dtype = getattr(runtime.torch, self.profile.torch_dtype, None)
                if dtype is None:
                    raise JudgeLoadError("Pinned Qwen Torch runtime lacks dtype %s." % self.profile.torch_dtype)
                kwargs: Mapping[str, Any] = {
                    "revision": self.profile.model_revision,
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "use_safetensors": True,
                    "torch_dtype": dtype,
                }
                kwargs = dict(kwargs)
                if self.profile.device_map is not None:
                    kwargs["device_map"] = self.profile.device_map
                self._model = runtime.model_cls.from_pretrained(self.profile.local_model_path, **kwargs)
            eval_method = getattr(self._model, "eval", None)
            if callable(eval_method):
                evaluated = eval_method()
                if evaluated is not None:
                    self._model = evaluated
            if not callable(getattr(self._model, "generate", None)):
                raise JudgeLoadError("Qwen judge model does not expose generate().")
        if self._processor is None:
            if self._processor_factory is not None:
                self._processor = self._processor_factory(self.profile, runtime)
            else:
                self._processor = runtime.processor_cls.from_pretrained(
                    self.profile.local_model_path,
                    revision=self.profile.processor_revision,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            if not callable(getattr(self._processor, "apply_chat_template", None)):
                raise JudgeLoadError("Qwen judge processor does not expose apply_chat_template().")
            if not callable(self._processor):
                raise JudgeLoadError("Qwen judge processor is not callable for multimodal encoding.")
            if not callable(getattr(self._processor, "batch_decode", None)):
                raise JudgeLoadError("Qwen judge processor does not expose batch_decode().")
        return self._model, self._processor

    @staticmethod
    def _messages(request: JudgeRequest, *, format_retry: bool = False) -> Tuple[Mapping[str, Any], ...]:
        instruction, rubric = request.resolved_instruction_and_rubric()
        timestamp_text = ", ".join("%d:%.9g" % (index, float(value)) for index, value in enumerate(request.frame_timestamps))
        text = (
            "You are a blinded robot-rollout rubric judge. The video has exactly 16 ordered frames. "
            "Their nominal control timestamps (not display time) are: %s.\n"
            "Exact task instruction: %s\n"
            "Frozen task rubric: %s"
        ) % (timestamp_text, instruction, rubric)
        content = [{"type": "video", "video": "plumb-16-frame-video"}]
        # Mark reference positions in the text template while passing opaque
        # pixels separately to the processor.  URI/hash remain outside VLM input.
        for index, _reference in enumerate(request.reference_images):
            content.append({"type": "image", "image": "provenance-backed-reference-%d" % index})
        content.append({"type": "text", "text": text})
        return (
            {
                "role": "system",
                "content": _BASE_SYSTEM_CONTRACT + (_FORMAT_ONLY_RETRY_REMINDER if format_retry else ""),
            },
            {"role": "user", "content": content},
        )

    @staticmethod
    def _move_inputs(inputs: Any, model: Any) -> Any:
        device = getattr(model, "device", None)
        if device is None:
            return inputs
        move = getattr(inputs, "to", None)
        if callable(move):
            moved = move(device)
            return inputs if moved is None else moved
        if isinstance(inputs, Mapping):
            copied = {}
            for key, value in inputs.items():
                value_move = getattr(value, "to", None)
                copied[key] = value_move(device) if callable(value_move) else value
            return copied
        return inputs

    @staticmethod
    def _cuda_device_ids(torch_module: Any, model: Any) -> Tuple[int, ...]:
        """Find CUDA device indexes touched by this local model/device map."""

        candidates = []
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, Mapping):
            candidates.extend(device_map.values())
        candidates.append(getattr(model, "device", None))
        result = []
        for candidate in candidates:
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                if candidate not in result:
                    result.append(candidate)
                continue
            text = str(candidate)
            if not text.startswith("cuda"):
                continue
            try:
                index = int(text.split(":", 1)[1]) if ":" in text else 0
            except ValueError:
                continue
            if index not in result:
                result.append(index)
        cuda = getattr(torch_module, "cuda", None)
        try:
            if not result and cuda is not None and cuda.is_available():
                result.append(int(cuda.current_device()))
        except (AttributeError, RuntimeError):
            pass
        return tuple(result)

    @staticmethod
    def _set_seed(torch_module: Any, device_ids: Sequence[int], seed: int) -> None:
        """Set CPU plus explicitly selected CUDA generators inside fork_rng."""

        random_module = getattr(torch_module, "random", None)
        default_generator = getattr(random_module, "default_generator", None)
        manual_seed = getattr(default_generator, "manual_seed", None)
        if not callable(manual_seed):
            manual_seed = getattr(torch_module, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)
        cuda = getattr(torch_module, "cuda", None)
        cuda_seed = getattr(cuda, "manual_seed", None)
        cuda_device = getattr(cuda, "device", None)
        if callable(cuda_seed):
            for device_id in device_ids:
                if callable(cuda_device):
                    with cuda_device(device_id):
                        cuda_seed(seed)
                else:
                    cuda_seed(seed)

    @staticmethod
    def _seeded_generation_context(runtime: _QwenRuntime, model: Any, seed: int) -> Any:
        """Fork and restore Torch RNG state while serializing shared generation.

        Hugging Face ``generate`` does not consistently accept a ``generator``
        keyword across released versions.  Seeding the process generators under
        ``torch.random.fork_rng`` works across those versions while the shared
        lock prevents another judge thread from observing temporary state.
        """

        torch_module = runtime.torch
        device_ids = QwenRubricJudge._cuda_device_ids(torch_module, model)
        random_module = getattr(torch_module, "random", None)
        fork_rng = getattr(random_module, "fork_rng", None)

        class _SeededContext:
            @staticmethod
            def _snapshot_state() -> Tuple[Any, Tuple[Tuple[int, Any], ...]]:
                get_cpu = getattr(torch_module, "get_rng_state", None)
                cpu_state = get_cpu() if callable(get_cpu) else None
                cuda = getattr(torch_module, "cuda", None)
                get_cuda = getattr(cuda, "get_rng_state", None)
                cuda_device = getattr(cuda, "device", None)
                cuda_states = []
                if callable(get_cuda):
                    for device_id in device_ids:
                        try:
                            if callable(cuda_device):
                                with cuda_device(device_id):
                                    state = get_cuda()
                            else:
                                state = get_cuda(device_id)
                            cuda_states.append((device_id, state))
                        except (AttributeError, RuntimeError, TypeError):
                            continue
                return cpu_state, tuple(cuda_states)

            @staticmethod
            def _restore_state(saved: Tuple[Any, Tuple[Tuple[int, Any], ...]]) -> None:
                cpu_state, cuda_states = saved
                set_cpu = getattr(torch_module, "set_rng_state", None)
                if cpu_state is not None and callable(set_cpu):
                    set_cpu(cpu_state)
                cuda = getattr(torch_module, "cuda", None)
                set_cuda = getattr(cuda, "set_rng_state", None)
                cuda_device = getattr(cuda, "device", None)
                if callable(set_cuda):
                    for device_id, state in cuda_states:
                        if callable(cuda_device):
                            with cuda_device(device_id):
                                set_cuda(state)
                        else:
                            set_cuda(state, device_id)

            def __enter__(self_inner) -> None:
                _GENERATION_RNG_LOCK.acquire()
                try:
                    if callable(fork_rng):
                        self_inner._fork = fork_rng(devices=list(device_ids), enabled=True)
                        self_inner._fork.__enter__()
                        self_inner._saved_state = None
                    else:
                        self_inner._fork = None
                        self_inner._saved_state = self_inner._snapshot_state()
                    QwenRubricJudge._set_seed(torch_module, device_ids, seed)
                except BaseException:
                    if getattr(self_inner, "_fork", None) is not None:
                        self_inner._fork.__exit__(None, None, None)
                    elif getattr(self_inner, "_saved_state", None) is not None:
                        self_inner._restore_state(self_inner._saved_state)
                    _GENERATION_RNG_LOCK.release()
                    raise
                return None

            def __exit__(self_inner, exc_type: Any, exc: Any, traceback_value: Any) -> bool:
                try:
                    if getattr(self_inner, "_fork", None) is not None:
                        self_inner._fork.__exit__(exc_type, exc, traceback_value)
                    elif getattr(self_inner, "_saved_state", None) is not None:
                        self_inner._restore_state(self_inner._saved_state)
                finally:
                    _GENERATION_RNG_LOCK.release()
                return False

        return _SeededContext()

    @staticmethod
    def _inference_context(torch_module: Any) -> Any:
        function = getattr(torch_module, "inference_mode", None) or getattr(torch_module, "no_grad", None)
        return function() if callable(function) else contextlib.nullcontext()

    @staticmethod
    def _reset_peak(torch_module: Any) -> None:
        cuda = getattr(torch_module, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                cuda.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            return None

    @staticmethod
    def _peak_memory(torch_module: Any) -> Optional[int]:
        cuda = getattr(torch_module, "cuda", None)
        try:
            if cuda is not None and cuda.is_available():
                return int(cuda.max_memory_allocated())
        except (AttributeError, RuntimeError):
            return None
        return None

    @staticmethod
    def _trim_generated(generated: Any, input_ids: Any) -> Any:
        if input_ids is None:
            return generated
        try:
            return [output_ids[len(input_ids[index]) :] for index, output_ids in enumerate(generated)]
        except (TypeError, IndexError, KeyError):
            return generated

    @staticmethod
    def _decode(processor: Any, generated: Any, inputs: Any) -> str:
        if isinstance(generated, str):
            return generated
        if isinstance(generated, Sequence) and generated and isinstance(generated[0], str):
            return str(generated[0])
        input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else getattr(inputs, "input_ids", None)
        generated_only = QwenRubricJudge._trim_generated(generated, input_ids)
        decoded = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, Sequence) or not decoded or not isinstance(decoded[0], str):
            raise JudgeSchemaError("Qwen processor did not decode one textual judge output.")
        return decoded[0]

    @staticmethod
    def _looks_like_refusal(raw_output: Optional[str]) -> bool:
        if not isinstance(raw_output, str):
            return False
        text = raw_output.strip().lower()
        return any(
            text.startswith(prefix)
            for prefix in ("i cannot", "i can't", "i am unable", "i'm unable", "sorry", "i refuse", "unable to comply")
        )

    def _one_attempt(
        self, request: JudgeRequest, sample_index: int, seed: int, attempt_index: int, *, format_retry: bool = False
    ) -> JudgeAttempt:
        runtime = self._load_runtime()
        model, processor = self._ensure_components(runtime)
        self._reset_peak(runtime.torch)
        started = time.perf_counter()
        raw_output: Optional[str] = None
        transport_normalization = "none"
        prompt_variant = "format_retry_v1" if format_retry else "base_v2"
        try:
            messages = self._messages(request, format_retry=format_retry)
            prompt = processor.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
            encoded = processor(
                text=[prompt],
                images=[reference.image for reference in request.reference_images],
                videos=[list(request.frames)],
                padding=True,
                return_tensors="pt",
            )
            encoded = self._move_inputs(encoded, model)
            generation_kwargs = {
                "do_sample": True,
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                "max_new_tokens": self.sampling.max_new_tokens,
            }
            with self._inference_context(runtime.torch):
                with self._seeded_generation_context(runtime, model, seed):
                    if isinstance(encoded, Mapping):
                        generated = model.generate(**encoded, **generation_kwargs)
                    else:
                        generated = model.generate(encoded, **generation_kwargs)
            raw_output = self._decode(processor, generated, encoded)
            if self._looks_like_refusal(raw_output):
                raise JudgeRefusalError("judge_refusal")
            parsed, transport_normalization = _parse_rubric_json_with_transport(raw_output)
            failure_reason = None
        except JudgeRefusalError as error:
            parsed = None
            failure_reason = str(error)
        except JudgeSchemaError as error:
            parsed = None
            failure_reason = "schema_error: " + str(error)
        except (ValueError, TypeError) as error:
            parsed = None
            failure_reason = "unsupported_generation_configuration: %s: %s" % (type(error).__name__, error)
        except (TimeoutError, ConnectionError, OSError) as error:
            parsed = None
            failure_reason = "transient_transport_error: %s: %s" % (type(error).__name__, error)
        except Exception as error:
            parsed = None
            failure_reason = "runtime_error: %s: %s" % (type(error).__name__, error)
        return JudgeAttempt(
            sample_index=sample_index,
            seed=seed,
            attempt_index=attempt_index,
            raw_output=raw_output,
            parsed=parsed,
            failure_reason=failure_reason,
            wall_seconds=time.perf_counter() - started,
            gpu_peak_memory_bytes=self._peak_memory(runtime.torch),
            prompt_variant=prompt_variant,
            prompt_source_hash=_prompt_source_hash(format_retry),
            transport_normalization=transport_normalization,
        )

    def _sample(self, request: JudgeRequest, sample_index: int, seed: int) -> JudgeSampleReport:
        attempts = []
        for attempt_index in range(self.sampling.retries_per_sample + 1):
            attempt = self._one_attempt(
                request, sample_index, seed, attempt_index, format_retry=(attempt_index == 1)
            )
            attempts.append(attempt)
            if attempt.parsed is not None:
                break
            # Only JSON/schema and recognized transport failures receive the
            # one permitted retry. Unsupported generation options, OOMs, and
            # refusals are explicit unknowns, never blind repeat inference.
            if not (
                isinstance(attempt.failure_reason, str)
                and (
                    attempt.failure_reason.startswith("schema_error:")
                    or attempt.failure_reason.startswith("transient_transport_error:")
                )
            ):
                break
        return JudgeSampleReport(sample_index=sample_index, seed=seed, attempts=tuple(attempts))

    def _report_provenance(self, request: JudgeRequest) -> Mapping[str, Any]:
        """Build the binding record for this call without overstating trust.

        The source hash is the local bytes of this module at call time.  It
        binds a report to a local implementation snapshot but is neither a
        signed release nor evidence that the source/model/data are authentic.
        """

        frame_hashes, frame_semantics, frame_problems = _safe_pixel_hashes(request.frames)
        reference_hashes, reference_semantics, reference_problems = _safe_pixel_hashes(
            tuple(reference.image for reference in request.reference_images)
        )
        provided = request.provenance
        task = request.resolved_task()
        instruction, rubric = request.resolved_instruction_and_rubric()
        sampling_payload = {
            "sample_count": self.sampling.sample_count,
            "quorum": self.sampling.quorum,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "max_new_tokens": self.sampling.max_new_tokens,
            "retries_per_sample": self.sampling.retries_per_sample,
        }
        unavailable = list(frame_problems + reference_problems)
        video_hash = _normalise_sha256(provided.video_sha256) if provided is not None else None
        if request.diagnostic_mode:
            unavailable.append("diagnostic_mode")
        if provided is None or not provided.clip_id:
            unavailable.append("missing_clip_id")
        if video_hash is None:
            unavailable.append("missing_video_hash")
        if provided is None or not provided.protocol_id:
            unavailable.append("missing_protocol_id")
        calibration_manifest_hash = _normalise_sha256(
            provided.calibration_manifest_hash if provided is not None else None
        )
        if calibration_manifest_hash is None:
            unavailable.append("missing_calibration_manifest_hash")
        if not self.profile.asset_manifest_id:
            unavailable.append("missing_asset_manifest_id")
        if _normalise_sha256(self.profile.asset_manifest_sha256) is None:
            unavailable.append("missing_asset_manifest_sha256")
        if not self.profile.runtime_lock_id:
            unavailable.append("missing_runtime_lock_id")
        if _normalise_sha256(self.profile.runtime_lock_sha256) is None:
            unavailable.append("missing_runtime_lock_sha256")
        if any(value is None for value in frame_hashes):
            unavailable.append("missing_frame_pixel_hash")
        if any(value is None for value in reference_hashes):
            unavailable.append("missing_reference_pixel_hash")
        gate_d_eligible = request.is_primary and not unavailable
        if task is None:
            task_record: Mapping[str, Any] = {
                "task_id": None,
                "task_registry_id": None,
                "task_registry_hash": None,
                "rubric_hash": canonical_json_sha256({"diagnostic_rubric": rubric}),
                "instruction_hash": canonical_json_sha256({"diagnostic_instruction": instruction}),
            }
        else:
            task_record = {
                "task_id": task.task_id,
                "task_registry_id": BENCHMARK_TASK_REGISTRY.registry_id,
                "task_registry_hash": TASK_REGISTRY_HASH,
                "rubric_hash": task.rubric_hash,
                "instruction_hash": canonical_json_sha256({"task_id": task.task_id, "instruction": task.instruction}),
            }
        artifact_hashes: Dict[str, Optional[str]] = {"video": video_hash}
        for index, value in enumerate(frame_hashes):
            artifact_hashes["frame_%02d_pixels" % index] = value
        for index, reference in enumerate(request.reference_images):
            artifact_hashes["reference_%02d_source" % index] = "sha256:" + reference.sha256.lower()
            artifact_hashes["reference_%02d_pixels" % index] = reference_hashes[index]
        sampling_hash = canonical_json_sha256(sampling_payload)
        source_hash = file_source_sha256(__file__)
        return {
            "schema_version": 1,
            "mode": "primary" if request.is_primary else "diagnostic",
            "calibration_manifest_hash": calibration_manifest_hash,
            "clip_id": provided.clip_id if provided is not None else None,
            "protocol_id": provided.protocol_id if provided is not None else None,
            "task_id": task.task_id if task is not None else None,
            "task_registry_id": BENCHMARK_TASK_REGISTRY.registry_id if task is not None else None,
            "task_registry_hash": TASK_REGISTRY_HASH if task is not None else None,
            "rubric_hash": task.rubric_hash if task is not None else canonical_json_sha256({"diagnostic_rubric": rubric}),
            "sampling_hash": sampling_hash,
            "task": task_record,
            "evidence_hashes": {
                "video_hash": video_hash,
                "frame_pixel_hashes": list(frame_hashes),
                "frame_hash_semantics": list(frame_semantics),
                "timestamp_hash": canonical_json_sha256({"frame_timestamps": list(request.frame_timestamps)}),
                "reference_image_pixel_hashes": list(reference_hashes),
                "reference_hash_semantics": list(reference_semantics),
                "reference_source_hashes": ["sha256:" + reference.sha256.lower() for reference in request.reference_images],
            },
            "artifact_hashes": artifact_hashes,
            "model": {
                "id": QWEN_JUDGE_MODEL_ID,
                "profile_id": self.profile.profile_id,
                "model_revision": self.profile.model_revision,
                "processor_revision": self.profile.processor_revision,
                "transformers_version": self.profile.transformers_version,
                "container_digest": self.profile.container_digest,
                "asset_manifest_id": self.profile.asset_manifest_id,
                "asset_manifest_sha256": _normalise_sha256(self.profile.asset_manifest_sha256),
                "runtime_lock_id": self.profile.runtime_lock_id,
                "runtime_lock_sha256": _normalise_sha256(self.profile.runtime_lock_sha256),
                "local_files_only": self.profile.local_files_only,
                "trust_remote_code": self.profile.trust_remote_code,
            },
            "sampling": {"config": sampling_payload, "sampling_hash": sampling_hash},
            "prompt_contract": {
                "version": PROMPT_SYSTEM_CONTRACT_VERSION,
                "base_source_hash": _prompt_source_hash(False),
                "format_retry_source_hash": _prompt_source_hash(True),
                "transport_normalization": "json_code_fence_v1 accepts one exactly anchored ```json wrapper only.",
            },
            "producer": {
                "name": "plumb.policies.judge.QwenRubricJudge",
                "version": "plumb-judge-report-v1",
                "source_hash": source_hash,
                "source_hash_semantics": "source_hash is SHA-256 of local plumb/policies/judge.py bytes at call time; not an attestation.",
            },
            "trust": {
                "qualified": False,
                "gate_d_eligible": gate_d_eligible,
                "test_mode": request.diagnostic_mode,
                "gate_d_unavailable_reasons": sorted(set(unavailable)),
            },
        }

    def evaluate(self, request: JudgeRequest, *, seeds: Sequence[int]) -> JudgeReport:
        """Evaluate with exactly five logged seeds and at most one retry per sample."""

        request.validate()
        self.sampling.validate()
        if isinstance(seeds, (str, bytes)) or len(seeds) != self.sampling.sample_count:
            raise JudgeInputError("Judge requires exactly five explicitly logged integer sampling seeds.")
        normalised_seeds = []
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise JudgeInputError("Judge sampling seeds must be integers.")
            normalised_seeds.append(seed)
        if len(set(normalised_seeds)) != self.sampling.sample_count:
            raise JudgeInputError("Judge requires five distinct logged seeds for independent samples.")
        reports = tuple(self._sample(request, index, seed) for index, seed in enumerate(normalised_seeds))
        wall_values = [
            attempt.wall_seconds
            for report in reports
            for attempt in report.attempts
            if attempt.wall_seconds is not None
        ]
        memory_values = [
            attempt.gpu_peak_memory_bytes
            for report in reports
            for attempt in report.attempts
            if attempt.gpu_peak_memory_bytes is not None
        ]
        binary_success, progress, judge_status, missing_reason, agreeing_samples = _aggregate_sample_reports(
            reports, self.sampling
        )
        return JudgeReport(
            binary_success=binary_success,
            progress=progress,
            judge_status=judge_status,
            missing_reason=missing_reason,
            agreeing_samples=agreeing_samples,
            sample_reports=reports,
            wall_seconds=sum(wall_values) if wall_values else None,
            gpu_peak_memory_bytes=max(memory_values) if memory_values else None,
            sampling=self.sampling,
            provenance=self._report_provenance(request),
        )

    score = evaluate


def _is_dir(path: str) -> bool:
    # Kept tiny for monkeypatch-free tests and to avoid resolving symlinks into
    # a potential remote mount during capability inspection.
    try:
        from pathlib import Path

        return Path(path).is_dir()
    except (OSError, TypeError):
        return False
