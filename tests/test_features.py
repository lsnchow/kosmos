"""V-JEPA2 frozen feature-diagnostic tests. CPU-only, no torch, no network.

Every encoder here is an injected deterministic fake.  The transformers version
and model revision are fixture values, not resolved pins: this repo has no
resolved V-JEPA2 commit, and these tests must not imply one.

The load-bearing test is
``test_the_diagnostic_cannot_alter_a_stage_a_verdict``: it builds a real
:class:`~plumb.validity.StageAValidityGate` report, runs the entire feature
diagnostic against it, and asserts the verdict and its reason codes are
untouched and unreachable.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

from plumb.adapters.contracts import CapabilityStatus
from plumb.features import (
    ASSET_BYTES_DEVELOPMENT,
    ASSET_BYTES_VERIFIED,
    DISTANCE_COMPUTED,
    DISTANCE_UNDEFINED_ZERO_NORM,
    FEATURE_API_VISION_FEATURES,
    FEATURE_DIAGNOSTIC_LIMITATION,
    FORBIDDEN_RESULT_KEYS,
    POOLING_IDENTITY_VECTOR,
    POOLING_MEAN_OVER_TOKENS,
    SAMPLING_CONTIGUOUS_PREFIX,
    SAMPLING_UNIFORM_INCLUSIVE,
    VJEPA2_FRAMES_PER_CLIP,
    VJEPA2_MODEL_ID,
    VJEPA2_REQUIRED_FILES,
    VJEPA2_SKIPPED_FILES,
    FeatureVector,
    VJepaConfigurationError,
    VJepaFeatureDiagnostic,
    VJepaInputError,
    VJepaLoadError,
    VJepaPreprocessing,
    VJepaProfile,
    _VJepaRuntime,
    clip_to_clip_drift_curve,
    compare_clips,
    plan_sampling,
    stage_a_separation_evidence,
    teacher_forced_vs_free_running,
)
from plumb.validity import (
    ALL_REASON_CODES,
    CALIBRATION_DEVELOPMENT,
    FLOW_BLOCK_MATCHING,
    StageAParameters,
    StageAValidityGate,
)


# Fixture values only. A 40-hex string here is not a resolved V-JEPA2 commit.
FIXTURE_REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_FIXTURE_REVISION = "fedcba9876543210fedcba9876543210fedcba98"
FIXTURE_TRANSFORMERS_VERSION = "4.52.4"


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class FakeNoGrad:
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    def __enter__(self) -> "FakeNoGrad":
        self.entered += 1
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.exited += 1
        return False


class FakeTorch:
    """Only the two attributes the diagnostic actually touches."""

    float32 = "fixture-float32"

    def __init__(self) -> None:
        self.contexts: List[FakeNoGrad] = []

    def no_grad(self) -> FakeNoGrad:
        context = FakeNoGrad()
        self.contexts.append(context)
        return context


class FakeVideoProcessor:
    """Records its calls and passes the 64 selected frames straight through."""

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, Any]] = []

    def __call__(self, video: Any, return_tensors: Optional[str] = None) -> Dict[str, Any]:
        self.calls.append((video, return_tensors))
        return {"pixel_values_videos": [[float(frame) for frame in video]]}


class FakeVJepaEncoder:
    """Deterministic encoder stand-in: a fixed function of the 64 frames."""

    def __init__(
        self,
        *,
        tokens: int = 3,
        dimension: int = 4,
        batch: int = 1,
        output: Any = None,
        outputs: Optional[Sequence[Any]] = None,
    ) -> None:
        self.tokens = tokens
        self.dimension = dimension
        self.batch = batch
        self.output = output
        self.outputs = list(outputs) if outputs is not None else None
        self.calls: List[Mapping[str, Any]] = []
        self.eval_calls = 0

    def eval(self) -> "FakeVJepaEncoder":
        self.eval_calls += 1
        return self

    def get_vision_features(self, **inputs: Any) -> Any:
        self.calls.append(inputs)
        if self.outputs is not None:
            return self.outputs[(len(self.calls) - 1) % len(self.outputs)]
        if self.output is not None:
            return self.output
        frames = inputs["pixel_values_videos"][0]
        clip = [
            [
                math.fsum(frame * (token + 1) * (column + 2) for frame in frames) / len(frames)
                for column in range(self.dimension)
            ]
            for token in range(self.tokens)
        ]
        return [clip for _ in range(self.batch)]


def fake_runtime(version: str = FIXTURE_TRANSFORMERS_VERSION) -> _VJepaRuntime:
    return _VJepaRuntime(
        torch=FakeTorch(),
        model_cls=object,
        processor_cls=object,
        transformers_version=version,
    )


def write_asset_directory(root: Path, *, exact_candidate_bytes: bool = False) -> Path:
    """Create the three required files; ``original/model.pth`` is never created."""

    root.mkdir(parents=True, exist_ok=True)
    for required in VJEPA2_REQUIRED_FILES:
        path = root / required.path
        path.write_bytes(b"")
        if exact_candidate_bytes:
            # Sparse: st_size matches the spec candidate length without writing
            # 1.3 GB. The diagnostic only reads sizes during capability checks.
            os.truncate(str(path), required.candidate_bytes)
    return root


def build_profile(root: Path, **changes: Any) -> VJepaProfile:
    defaults: Dict[str, Any] = {
        "profile_id": "vjepa2-fixture",
        "local_model_path": str(root),
        "model_revision": FIXTURE_REVISION,
        "transformers_version": FIXTURE_TRANSFORMERS_VERSION,
        "asset_verification": ASSET_BYTES_DEVELOPMENT,
    }
    defaults.update(changes)
    return VJepaProfile(**defaults)


def build_diagnostic(
    tmp_path: Path,
    *,
    encoder: Optional[FakeVJepaEncoder] = None,
    processor: Optional[FakeVideoProcessor] = None,
    runtime: Optional[_VJepaRuntime] = None,
    profile: Optional[VJepaProfile] = None,
    create_asset: bool = True,
    name: str = "asset",
) -> Tuple[VJepaFeatureDiagnostic, FakeVJepaEncoder, FakeVideoProcessor, _VJepaRuntime]:
    root = tmp_path / name
    if create_asset:
        write_asset_directory(root)
    encoder = encoder if encoder is not None else FakeVJepaEncoder()
    processor = processor if processor is not None else FakeVideoProcessor()
    runtime = runtime if runtime is not None else fake_runtime()
    diagnostic = VJepaFeatureDiagnostic(
        profile if profile is not None else build_profile(root),
        runtime_factory=lambda: runtime,
        model_factory=lambda profile_, runtime_: encoder,
        processor_factory=lambda profile_, runtime_: processor,
    )
    return diagnostic, encoder, processor, runtime


def clip(count: int = VJEPA2_FRAMES_PER_CLIP, *, scale: float = 1.0, offset: float = 0.0) -> Tuple[List[float], List[float]]:
    """One synthetic clip: frame values and strictly increasing control times."""

    frames = [offset + scale * (index + 1) for index in range(count)]
    timestamps = [0.25 * index for index in range(count)]
    return frames, timestamps


def walk_keys(value: Any) -> List[str]:
    keys: List[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(walk_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.extend(walk_keys(item))
    return keys


# ---------------------------------------------------------------------------
# Profile, preprocessing and asset pinning
# ---------------------------------------------------------------------------


def test_profile_names_the_three_required_files_and_skips_the_duplicate_checkpoint() -> None:
    sizes = {item.path: item.candidate_bytes for item in VJEPA2_REQUIRED_FILES}
    assert sizes == {
        "config.json": 785,
        "model.safetensors": 1_303_947_864,
        "video_preprocessor_config.json": 1_298,
    }
    assert "original/model.pth" not in sizes
    assert VJEPA2_SKIPPED_FILES == ("original/model.pth",)

    described = build_profile(Path("/does/not/exist")).as_dict()
    assert [item["path"] for item in described["required_files"]] == sorted(sizes)
    assert described["skipped_files"] == ["original/model.pth"]
    assert "duplicates model.safetensors" in described["skipped_file_reason"]
    assert described["model_id"] == VJEPA2_MODEL_ID


def test_a_profile_may_not_list_the_skipped_duplicate_as_required() -> None:
    from plumb.features import VJepaRequiredFile

    profile = build_profile(
        Path("/does/not/exist"),
        required_files=VJEPA2_REQUIRED_FILES + (VJepaRequiredFile("original/model.pth", 5_100_000_000),),
    )

    assert "original/model.pth is a skipped duplicate" in str(profile.profile_error())


def test_profile_refuses_a_mutable_revision_and_states_the_unresolved_pin() -> None:
    error = build_profile(Path("/does/not/exist"), model_revision="main").profile_error()

    assert "immutable 40-character hexadecimal revision" in str(error)
    assert "no resolved immutable commit" in str(error)
    assert build_profile(Path("/does/not/exist")).profile_error() is None


def test_profile_refuses_remote_code_and_network_retrieval() -> None:
    assert "trust_remote_code" in str(
        build_profile(Path("/x"), trust_remote_code=True).profile_error()
    )
    assert "local-only" in str(build_profile(Path("/x"), local_files_only=False).profile_error())


def test_preprocessing_pins_the_documented_sixty_four_frame_contract() -> None:
    preprocessing = VJepaPreprocessing()

    assert preprocessing.frames_per_clip == 64 == VJEPA2_FRAMES_PER_CLIP
    assert preprocessing.crop_size == 256
    assert preprocessing.sampling_rule == SAMPLING_UNIFORM_INCLUSIVE
    assert preprocessing.configuration_error() is None
    # Mean/std/rescale are not transcribed here; the shipped file is the authority.
    assert preprocessing.normalization_authority == "video_preprocessor_config.json"
    assert preprocessing.video_preprocessor_config_sha256 is None

    assert "frames_per_clip must be 64" in str(
        replace(preprocessing, frames_per_clip=32).configuration_error()
    )
    assert "crop_size must be 256" in str(replace(preprocessing, crop_size=224).configuration_error())
    assert "silent resampling" in str(
        replace(preprocessing, require_distinct_frame_indices=False).configuration_error()
    )
    # The hash is stable across instances and sensitive to the sampling rule.
    assert preprocessing.preprocessing_hash == VJepaPreprocessing().preprocessing_hash
    assert (
        replace(preprocessing, sampling_rule=SAMPLING_CONTIGUOUS_PREFIX).preprocessing_hash
        != preprocessing.preprocessing_hash
    )


# ---------------------------------------------------------------------------
# The 64-frame contract: reject, never resample
# ---------------------------------------------------------------------------


def test_a_clip_that_cannot_supply_sixty_four_frames_is_rejected_not_resampled(tmp_path: Path) -> None:
    with pytest.raises(VJepaInputError) as short_clip:
        plan_sampling([0.1 * index for index in range(63)])
    message = str(short_clip.value)
    assert "requires exactly 64 frames per clip" in message
    assert "this clip supplies 63" in message
    assert "padding, duplicating, interpolating, or resampling" in message

    diagnostic, encoder, processor, _ = build_diagnostic(tmp_path)
    frames, timestamps = clip(10)
    with pytest.raises(VJepaInputError):
        diagnostic.embed(frames, timestamps)

    # The refusal happens before any encoder or processor work.
    assert encoder.calls == []
    assert processor.calls == []


def test_an_exactly_sixty_four_frame_clip_uses_every_frame_in_order() -> None:
    sampling = plan_sampling([0.25 * index for index in range(64)])

    assert sampling.frame_indices == tuple(range(64))
    assert sampling.includes_first_source_frame is True
    assert sampling.includes_final_control_tick is True


def test_uniform_sampling_is_explicit_distinct_and_keeps_both_endpoints() -> None:
    timestamps = [0.1 * index for index in range(200)]
    sampling = plan_sampling(timestamps, rule=SAMPLING_UNIFORM_INCLUSIVE)

    assert len(sampling.frame_indices) == 64
    assert len(set(sampling.frame_indices)) == 64
    assert list(sampling.frame_indices) == sorted(sampling.frame_indices)
    assert sampling.frame_indices[0] == 0
    assert sampling.frame_indices[-1] == 199
    # Timestamps are the real source timestamps at those indexes, not interpolated.
    assert sampling.frame_timestamps == tuple(timestamps[index] for index in sampling.frame_indices)
    assert sampling.source_frame_count == 200


def test_the_contiguous_prefix_rule_is_the_model_card_arange() -> None:
    sampling = plan_sampling([0.1 * index for index in range(200)], rule=SAMPLING_CONTIGUOUS_PREFIX)

    assert sampling.frame_indices == tuple(range(64))
    assert sampling.includes_final_control_tick is False
    assert "discards the tail" in str(sampling.as_dict()["rule_description"])


def test_nonmonotone_or_nonfinite_control_timestamps_are_rejected() -> None:
    timestamps = [0.25 * index for index in range(64)]
    repeated = list(timestamps)
    repeated[10] = repeated[9]
    with pytest.raises(VJepaInputError) as error:
        plan_sampling(repeated)
    assert "strictly increasing" in str(error.value)

    infinite = list(timestamps)
    infinite[3] = float("inf")
    with pytest.raises(VJepaInputError):
        plan_sampling(infinite)


def test_embed_requires_one_control_timestamp_per_frame(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    frames, timestamps = clip(70)

    with pytest.raises(VJepaInputError) as error:
        diagnostic.embed(frames, timestamps[:-1])

    assert "exactly one control timestamp" in str(error.value)


# ---------------------------------------------------------------------------
# Embedding: deterministic, recorded, never fabricated
# ---------------------------------------------------------------------------


def test_the_same_clip_embeds_identically_twice(tmp_path: Path) -> None:
    diagnostic, encoder, _, _ = build_diagnostic(tmp_path)
    frames, timestamps = clip(90)

    first = diagnostic.embed(frames, timestamps, clip_id="clip-a")
    second = diagnostic.embed(frames, timestamps, clip_id="clip-a")

    assert first.embedding == second.embedding
    assert first.frame_indices == second.frame_indices
    assert first.frame_timestamps == second.frame_timestamps
    assert first.model_revision == second.model_revision == FIXTURE_REVISION
    assert first.preprocessing_hash == second.preprocessing_hash
    assert first.parameters_hash == second.parameters_hash
    assert first.measurement_hash == second.measurement_hash
    assert first.as_dict() == second.as_dict()
    assert len(encoder.calls) == 2


def test_the_feature_record_states_the_indexes_revision_and_hashes(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    frames, timestamps = clip(80)

    record = diagnostic.embed(frames, timestamps, clip_id="clip-a").as_dict()

    assert record["frame_indices"][0] == 0
    assert record["frame_indices"][-1] == 79
    assert len(record["frame_indices"]) == 64
    assert record["frame_timestamps"] == [timestamps[index] for index in record["frame_indices"]]
    assert record["model_id"] == VJEPA2_MODEL_ID
    assert record["model_revision"] == FIXTURE_REVISION
    assert record["preprocessing_hash"] == diagnostic.preprocessing_hash
    assert record["parameters_hash"] == diagnostic.parameters_hash
    assert record["feature_api_used"] == FEATURE_API_VISION_FEATURES
    assert record["pooling_applied"] == POOLING_MEAN_OVER_TOKENS
    assert record["asset_verification"] == ASSET_BYTES_DEVELOPMENT
    assert record["dimension"] == len(record["embedding"]) == 4


def test_different_clips_produce_different_embeddings(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)

    left = diagnostic.embed(*clip(64, scale=1.0), clip_id="left")
    right = diagnostic.embed(*clip(64, scale=3.0), clip_id="right")

    assert left.embedding != right.embedding


def test_pooling_is_mean_over_tokens_and_is_recorded(tmp_path: Path) -> None:
    encoder = FakeVJepaEncoder(output=[[[1.0, 10.0], [3.0, 20.0]]])
    diagnostic, _, _, _ = build_diagnostic(tmp_path, encoder=encoder)

    pooled = diagnostic.embed(*clip(64))

    assert pooled.embedding == (2.0, 15.0)
    assert pooled.pooling_applied == POOLING_MEAN_OVER_TOKENS

    flat = FakeVJepaEncoder(output=[7.0, 8.0])
    identity, _, _, _ = build_diagnostic(tmp_path, encoder=flat, name="asset-flat")
    vector = identity.embed(*clip(64))
    assert vector.embedding == (7.0, 8.0)
    assert vector.pooling_applied == POOLING_IDENTITY_VECTOR


def test_an_ambiguous_batch_axis_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    encoder = FakeVJepaEncoder(batch=2)
    diagnostic, _, _, _ = build_diagnostic(tmp_path, encoder=encoder)

    with pytest.raises(VJepaLoadError) as error:
        diagnostic.embed(*clip(64))

    assert "batch axis" in str(error.value)


def test_a_nonfinite_encoder_output_is_refused_not_recorded(tmp_path: Path) -> None:
    encoder = FakeVJepaEncoder(output=[[[float("nan"), 1.0]]])
    diagnostic, _, _, _ = build_diagnostic(tmp_path, encoder=encoder)

    with pytest.raises(VJepaLoadError) as error:
        diagnostic.embed(*clip(64))

    assert "non-finite" in str(error.value)


def test_embedding_runs_under_no_grad_in_eval_mode(tmp_path: Path) -> None:
    runtime = fake_runtime()
    diagnostic, encoder, _, _ = build_diagnostic(tmp_path, runtime=runtime)

    diagnostic.embed(*clip(64))

    assert encoder.eval_calls == 1
    assert [context.entered for context in runtime.torch.contexts] == [1]


def test_a_runtime_version_mismatch_is_refused(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path, runtime=fake_runtime("4.40.1"))

    with pytest.raises(VJepaLoadError) as error:
        diagnostic.embed(*clip(64))

    assert "differs from profile-pinned" in str(error.value)


def test_the_processor_receives_exactly_the_sixty_four_selected_frames(tmp_path: Path) -> None:
    diagnostic, _, processor, _ = build_diagnostic(tmp_path)
    frames, timestamps = clip(128)

    vector = diagnostic.embed(frames, timestamps)

    (video, return_tensors), = processor.calls
    assert return_tensors == "pt"
    assert len(video) == 64
    assert video == [frames[index] for index in vector.frame_indices]


# ---------------------------------------------------------------------------
# Capability: BLOCKED with a named reason, and nothing embeddable
# ---------------------------------------------------------------------------


def test_capability_is_blocked_with_a_named_reason_when_the_asset_is_absent(tmp_path: Path) -> None:
    missing = tmp_path / "absent"
    diagnostic = VJepaFeatureDiagnostic(build_profile(missing))

    result = diagnostic.capability()

    assert result.status is CapabilityStatus.BLOCKED
    assert result.details["blocker"] == "asset_absent"
    assert "asset_absent" in result.reason
    assert str(missing) in result.reason
    assert "local_files_only=True" in result.reason
    assert result.details["alters_validity_gate"] is False

    # Nothing can be embedded in that state.
    with pytest.raises(VJepaLoadError) as error:
        diagnostic.embed(*clip(64))
    assert "Refusing to embed" in str(error.value)
    assert "blocked" in str(error.value)


def test_capability_names_the_missing_required_files(tmp_path: Path) -> None:
    root = tmp_path / "partial"
    root.mkdir()
    (root / "config.json").write_bytes(b"{}")
    diagnostic = VJepaFeatureDiagnostic(build_profile(root))

    result = diagnostic.capability()

    assert result.status is CapabilityStatus.BLOCKED
    assert result.details["blocker"] == "required_files_missing"
    assert result.details["missing_files"] == ["model.safetensors", "video_preprocessor_config.json"]
    assert "no Hub download was attempted" in result.reason


def test_capability_is_blocked_when_a_candidate_byte_length_disagrees(tmp_path: Path) -> None:
    root = write_asset_directory(tmp_path / "wrong-bytes")
    diagnostic = VJepaFeatureDiagnostic(build_profile(root, asset_verification=ASSET_BYTES_VERIFIED))

    result = diagnostic.capability()

    assert result.status is CapabilityStatus.BLOCKED
    assert result.details["blocker"] == "candidate_file_size_mismatch"
    assert "1303947864 B" in result.reason
    mismatched = {item["path"] for item in result.details["size_mismatches"]}
    assert mismatched == {"config.json", "model.safetensors", "video_preprocessor_config.json"}


def test_capability_is_ready_when_every_candidate_byte_length_matches(tmp_path: Path) -> None:
    root = tmp_path / "exact-bytes"
    try:
        write_asset_directory(root, exact_candidate_bytes=True)
    except OSError as error:  # pragma: no cover - filesystem dependent
        pytest.skip("sparse file creation unavailable: %s" % error)
    observed = os.path.getsize(str(root / "model.safetensors"))
    if observed != 1_303_947_864:  # pragma: no cover - filesystem dependent
        pytest.skip("sparse file size not reported exactly")

    diagnostic = VJepaFeatureDiagnostic(build_profile(root, asset_verification=ASSET_BYTES_VERIFIED))
    result = diagnostic.capability()

    assert result.status is CapabilityStatus.READY_UNQUALIFIED
    assert result.details["asset_verification"] == ASSET_BYTES_VERIFIED
    assert result.details["is_a_rubric_judge"] is False
    assert "encoder diagnostic" in result.reason
    assert "not calibrated against human judgement" in result.reason
    assert "certified against a golden fixture" in result.details["fixture_verification_required"]


def test_capability_is_blocked_on_an_incomplete_profile_before_touching_the_disk(tmp_path: Path) -> None:
    root = write_asset_directory(tmp_path / "asset")
    diagnostic = VJepaFeatureDiagnostic(build_profile(root, model_revision="main"))

    result = diagnostic.capability()

    assert result.status is CapabilityStatus.BLOCKED
    assert result.details["blocker"] == "profile_incomplete"
    assert "immutable 40-character hexadecimal revision" in result.reason


def test_development_byte_verification_labels_every_emitted_record(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)

    vector = diagnostic.embed(*clip(64), clip_id="dev")
    distance = diagnostic.compare_clips(vector, diagnostic.embed(*clip(64, scale=2.0)))

    assert vector.asset_verification == ASSET_BYTES_DEVELOPMENT
    assert vector.as_dict()["asset_verification"] == ASSET_BYTES_DEVELOPMENT
    assert distance.as_dict()["asset_verification"] == ASSET_BYTES_DEVELOPMENT


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def test_distances_are_zero_for_identical_clips(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    frames, timestamps = clip(64)
    left = diagnostic.embed(frames, timestamps, clip_id="left")
    right = diagnostic.embed(frames, timestamps, clip_id="right")

    distance = compare_clips(left, right)

    assert distance.status == DISTANCE_COMPUTED
    assert distance.cosine_distance == 0.0
    assert distance.l2_distance == 0.0
    assert distance.identical_embeddings is True
    assert distance.left_clip_id == "left"
    assert distance.right_clip_id == "right"


def test_distances_are_symmetric(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    left = diagnostic.embed(*clip(64, scale=1.0), clip_id="left")
    right = diagnostic.embed(*clip(70, scale=5.0, offset=2.0), clip_id="right")

    forward = compare_clips(left, right)
    backward = compare_clips(right, left)

    assert forward.status == DISTANCE_COMPUTED
    assert forward.cosine_distance == backward.cosine_distance
    assert forward.l2_distance == backward.l2_distance
    assert forward.l2_distance > 0.0
    assert forward.identical_embeddings is False


def test_a_zero_norm_embedding_leaves_cosine_undefined_rather_than_zero(tmp_path: Path) -> None:
    encoder = FakeVJepaEncoder(outputs=[[[[0.0, 0.0]]], [[[1.0, 2.0]]]])
    diagnostic, _, _, _ = build_diagnostic(tmp_path, encoder=encoder)

    distance = compare_clips(diagnostic.embed(*clip(64)), diagnostic.embed(*clip(64, scale=2.0)))

    assert distance.status == DISTANCE_UNDEFINED_ZERO_NORM
    assert distance.cosine_distance is None
    assert "not as zero" in str(distance.reason)
    assert distance.l2_distance == pytest.approx(math.sqrt(5.0))


def test_a_distance_across_different_revisions_or_preprocessing_is_refused(tmp_path: Path) -> None:
    first, _, _, _ = build_diagnostic(tmp_path, name="one")
    other_root = write_asset_directory(tmp_path / "two")
    second = VJepaFeatureDiagnostic(
        build_profile(other_root, model_revision=OTHER_FIXTURE_REVISION),
        runtime_factory=fake_runtime,
        model_factory=lambda profile_, runtime_: FakeVJepaEncoder(),
        processor_factory=lambda profile_, runtime_: FakeVideoProcessor(),
    )

    with pytest.raises(VJepaInputError) as error:
        compare_clips(first.embed(*clip(64)), second.embed(*clip(64)))

    assert "model_revision" in str(error.value)
    assert "not on a common scale" in str(error.value)


def test_the_mount_path_changes_the_declared_pin_but_not_the_measurement(tmp_path: Path) -> None:
    """Two mounts of the same revision produce comparable numbers."""

    first, _, _, _ = build_diagnostic(tmp_path, name="mount-a")
    second, _, _, _ = build_diagnostic(tmp_path, name="mount-b")

    assert first.parameters_hash != second.parameters_hash
    assert first.measurement_parameters_hash == second.measurement_parameters_hash

    distance = compare_clips(first.embed(*clip(64)), second.embed(*clip(64, scale=2.0)))
    assert distance.status == DISTANCE_COMPUTED

    # A different pinned revision is a different measurement, so it is refused.
    other_root = write_asset_directory(tmp_path / "mount-c")
    other_revision = VJepaFeatureDiagnostic(build_profile(other_root, model_revision=OTHER_FIXTURE_REVISION))
    assert other_revision.measurement_parameters_hash != first.measurement_parameters_hash


def test_a_distance_against_a_foreign_object_is_refused(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    vector = diagnostic.embed(*clip(64))

    with pytest.raises(VJepaInputError):
        compare_clips(vector, object())


# ---------------------------------------------------------------------------
# Teacher-forced versus free-running, and the drift curve
# ---------------------------------------------------------------------------


def teacher_and_free_branches(diagnostic: VJepaFeatureDiagnostic) -> Tuple[List[FeatureVector], List[FeatureVector]]:
    forced = [diagnostic.embed(*clip(64 + step * 8, scale=1.0), clip_id="tf-%d" % step) for step in range(3)]
    free = [diagnostic.embed(*clip(64 + step * 8, scale=1.0 + step), clip_id="fr-%d" % step) for step in range(3)]
    return forced, free


def test_teacher_forced_versus_free_running_reports_per_tick_visual_divergence(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    forced, free = teacher_and_free_branches(diagnostic)

    report = diagnostic.teacher_forced_vs_free_running(
        forced,
        free,
        control_ticks=[4, 8, 12],
        action_sequence_id="actions-fixture-1",
        generation_settings_id="generation-fixture-1",
    )

    assert report.control_ticks == (4, 8, 12)
    assert report.computed_point_count == 3
    assert report.undefined_point_count == 0
    assert report.last_control_tick_tested == 12
    assert report.max_cosine_distance_control_tick in (4, 8, 12)
    record = report.as_dict()
    assert record["action_sequence_id"] == "actions-fixture-1"
    assert record["generation_settings_id"] == "generation-fixture-1"
    assert "visual divergence only" in record["scope_note"]
    assert "not an unlimited drift guarantee" in record["tolerance_note"]
    assert len(record["points"]) == 3
    assert record["points"][0]["control_tick"] == 4


def test_identical_branches_diverge_by_zero(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    forced, _ = teacher_and_free_branches(diagnostic)

    report = teacher_forced_vs_free_running(
        forced,
        list(forced),
        control_ticks=[1, 2, 3],
        action_sequence_id="actions-fixture-1",
        generation_settings_id="generation-fixture-1",
    )

    assert report.max_cosine_distance == 0.0
    assert report.mean_cosine_distance == 0.0


def test_unequal_branch_lengths_are_rejected_not_truncated(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    forced, free = teacher_and_free_branches(diagnostic)

    with pytest.raises(VJepaInputError) as error:
        teacher_forced_vs_free_running(
            forced,
            free[:2],
            control_ticks=[1, 2, 3],
            action_sequence_id="actions-fixture-1",
            generation_settings_id="generation-fixture-1",
        )

    assert "equal control prefixes" in str(error.value)
    assert "Refusing to truncate" in str(error.value)


def test_a_divergence_without_matched_actions_or_settings_cannot_be_constructed(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    forced, free = teacher_and_free_branches(diagnostic)

    with pytest.raises(VJepaConfigurationError) as error:
        teacher_forced_vs_free_running(
            forced,
            free,
            control_ticks=[1, 2, 3],
            action_sequence_id="",
            generation_settings_id="generation-fixture-1",
        )

    assert "same actions" in str(error.value)


def test_the_drift_curve_is_indexed_by_control_ticks(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    reference = diagnostic.embed(*clip(64, scale=1.0), clip_id="reference")
    later = [diagnostic.embed(*clip(64, scale=1.0 + step), clip_id="tick-%d" % step) for step in range(1, 4)]

    curve = diagnostic.clip_to_clip_drift_curve(reference, later, control_ticks=[5, 10, 15])

    assert curve.control_ticks == (5, 10, 15)
    assert curve.reference_clip_id == "reference"
    record = curve.as_dict()
    assert record["x_axis"] == "control_ticks"
    assert record["last_control_tick_tested"] == 15
    assert record["monotone_nondecreasing"] in (True, False)
    assert record["computed_point_count"] == 3


def test_drift_curve_control_ticks_must_be_strictly_increasing(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    reference = diagnostic.embed(*clip(64), clip_id="reference")
    later = [diagnostic.embed(*clip(64, scale=2.0)), diagnostic.embed(*clip(64, scale=3.0))]

    with pytest.raises(VJepaInputError) as error:
        clip_to_clip_drift_curve(reference, later, control_ticks=[10, 10])

    assert "strictly increasing" in str(error.value)


# ---------------------------------------------------------------------------
# The separation guarantee
# ---------------------------------------------------------------------------


def build_stage_a_report(tmp_path: Path) -> Tuple[StageAValidityGate, Any]:
    """One real Stage A report from the real gate, with no feature input."""

    parameters = StageAParameters(
        parameter_set_id="plumb-features-separation-fixture",
        frame_height=32,
        frame_width=32,
        flow_downsample_factor=1,
        grid_rows=4,
        grid_cols=4,
        block_search_radius=2,
        flow_estimator=FLOW_BLOCK_MATCHING,
        min_reference_transitions=4,
        min_axis_map_transitions=4,
    )
    gate = StageAValidityGate(parameters=parameters, calibration_class=CALIBRATION_DEVELOPMENT)
    generator = np.random.default_rng(11)
    actions: Sequence[Sequence[float]] = [(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] * 5
    frames = [
        generator.integers(0, 256, size=(32, 32, 3), dtype=np.uint8) for _ in range(len(actions) + 1)
    ]
    report = gate.evaluate(
        frames=frames,
        actions=actions,
        nominal_timestamps=[0.2 * index for index in range(len(frames))],
        episode_id="separation-fixture",
    )
    return gate, report


def test_the_diagnostic_cannot_alter_a_stage_a_verdict(tmp_path: Path) -> None:
    gate, report = build_stage_a_report(tmp_path)
    verdict = report.validity
    codes = tuple(report.reason_codes)
    snapshot = json.loads(json.dumps(report.as_dict()))
    assert verdict in ("valid", "invalid", "unknown")
    # A real gate output with real reason codes, so this test cannot pass on an
    # empty report that had nothing to alter.
    assert codes
    assert set(codes) <= set(ALL_REASON_CODES)

    # Run the entire feature diagnostic surface against the same episode.
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    left = diagnostic.embed(*clip(64, scale=1.0), clip_id="left")
    right = diagnostic.embed(*clip(72, scale=4.0), clip_id="right")
    emitted = [
        left.as_dict(),
        right.as_dict(),
        diagnostic.compare_clips(left, right).as_dict(),
        diagnostic.clip_to_clip_drift_curve(left, [right], control_ticks=[7]).as_dict(),
        diagnostic.teacher_forced_vs_free_running(
            [left],
            [right],
            control_ticks=[7],
            action_sequence_id="actions-fixture-1",
            generation_settings_id="generation-fixture-1",
        ).as_dict(),
        diagnostic.describe(),
        dict(diagnostic.capability().details),
    ]

    # The verdict, its reason codes and the whole report are untouched.
    assert report.validity == verdict
    assert tuple(report.reason_codes) == codes
    assert json.loads(json.dumps(report.as_dict())) == snapshot

    # And structurally unreachable: the report is frozen.
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.validity = "valid"  # type: ignore[misc]

    # The gate still declares no learned-feature hook after the diagnostic ran.
    contract = gate.describe()
    assert contract["vjepa_hook"] is None
    assert contract["uses_learned_features"] is False

    # No emitted record carries a Stage A verdict, reason code or hook.
    dumped = json.dumps(emitted)
    for code in ALL_REASON_CODES:
        assert code not in dumped, code
    for record in emitted:
        assert "validity" not in record
        assert "reason_codes" not in record
        assert record["alters_validity_gate"] is False


def test_plumb_validity_does_not_import_plumb_features() -> None:
    evidence = stage_a_separation_evidence()

    assert evidence["stage_a_source_resolved"] is True
    assert evidence["feature_module_source_resolved"] is True
    assert evidence["stage_a_imports_feature_module"] is False
    assert evidence["feature_module_imports_stage_a"] is False
    assert evidence["provides_stage_a_hook"] is False
    assert evidence["accepts_or_returns_validity_report"] is False


def test_importing_the_feature_module_loads_neither_stage_a_nor_torch() -> None:
    script = (
        "import sys\n"
        "import plumb.features\n"
        "for banned in ('plumb.validity', 'torch', 'transformers', 'cv2'):\n"
        "    assert banned not in sys.modules, banned\n"
        "print('clean')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "clean" in completed.stdout


def feature_module_identifiers() -> set:
    """Every identifier the feature module's code actually uses."""

    source = Path(__file__).resolve().parents[1].joinpath("plumb", "features.py").read_text(encoding="utf-8")
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
            if node.asname:
                names.add(node.asname)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def test_no_stage_a_type_is_referenced_anywhere_in_the_feature_module() -> None:
    # AST identifiers only: a docstring mentioning Stage A must not count as a
    # reference, and a real annotation or argument must not escape one.
    identifiers = feature_module_identifiers()

    assert "plan_sampling" in identifiers, "the AST walk must actually be finding identifiers"
    for stage_a_type in ("ValidityReport", "ValidityReasonCode", "StageAValidityGate", "StageAParameters"):
        assert stage_a_type not in identifiers, stage_a_type


# ---------------------------------------------------------------------------
# Honest labelling on every emitted record
# ---------------------------------------------------------------------------


def all_records(tmp_path: Path) -> List[Dict[str, Any]]:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    left = diagnostic.embed(*clip(64, scale=1.0), clip_id="left")
    right = diagnostic.embed(*clip(96, scale=2.5), clip_id="right")
    distance = diagnostic.compare_clips(left, right)
    curve = diagnostic.clip_to_clip_drift_curve(left, [right], control_ticks=[3])
    divergence = diagnostic.teacher_forced_vs_free_running(
        [left],
        [right],
        control_ticks=[3],
        action_sequence_id="actions-fixture-1",
        generation_settings_id="generation-fixture-1",
    )
    return [
        left.as_dict(),
        right.as_dict(),
        distance.as_dict(),
        curve.as_dict(),
        curve.points[0].as_dict(),
        divergence.as_dict(),
        diagnostic.describe(),
    ]


def test_every_emitted_record_is_labelled_diagnostic_only_with_its_limitation(tmp_path: Path) -> None:
    records = all_records(tmp_path)
    assert len(records) == 7

    for record in records:
        assert record["diagnostic_only"] is True
        assert record["alters_validity_gate"] is False
        assert record["is_a_task_score"] is False
        assert record["limitation"] == FEATURE_DIAGNOSTIC_LIMITATION

    assert "not a success measure" in FEATURE_DIAGNOSTIC_LIMITATION
    assert "not calibrated against human judgement" in FEATURE_DIAGNOSTIC_LIMITATION
    assert "never participates in the deterministic Stage A validity gate" in FEATURE_DIAGNOSTIC_LIMITATION


def test_no_emitted_record_carries_a_success_failure_or_progress_key(tmp_path: Path) -> None:
    for record in all_records(tmp_path):
        offending = sorted(set(walk_keys(record)) & set(FORBIDDEN_RESULT_KEYS))
        assert offending == [], offending

    for banned in ("success", "progress", "validity", "score", "binary_success", "verdict"):
        assert banned in FORBIDDEN_RESULT_KEYS


def test_a_record_cannot_be_relabelled_as_a_score_or_a_gate_input(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)
    vector = diagnostic.embed(*clip(64), clip_id="left")

    with pytest.raises(VJepaConfigurationError) as score:
        replace(vector, is_a_task_score=True)
    assert "not a task score" in str(score.value)

    with pytest.raises(VJepaConfigurationError) as gate:
        replace(vector, alters_validity_gate=True)
    assert "outside the Stage A gate" in str(gate.value)

    with pytest.raises(VJepaConfigurationError) as weakened:
        replace(vector, limitation="calibrated fidelity measure")
    assert "frozen" in str(weakened.value)

    with pytest.raises(VJepaConfigurationError):
        replace(vector, diagnostic_only=False)


def test_the_diagnostic_contract_declares_it_is_not_a_rubric_judge(tmp_path: Path) -> None:
    diagnostic, _, _, _ = build_diagnostic(tmp_path)

    contract = diagnostic.describe()

    assert contract["role"] == "frozen_video_feature_diagnostic"
    assert contract["is_a_rubric_judge"] is False
    assert contract["separation"]["stage_a_imports_feature_module"] is False
    assert contract["profile"]["skipped_files"] == ["original/model.pth"]
    assert contract["profile"]["preprocessing"]["frames_per_clip"] == 64
    assert "no resolved immutable commit" in contract["revision_note"]


def test_the_module_exposes_no_validity_named_entry_point() -> None:
    import plumb.features as module

    public = [name for name in dir(module) if not name.startswith("_")]

    assert public
    assert [name for name in public if "validity" in name.lower()] == []
    assert [name for name in public if "success" in name.lower()] == []
    assert [name for name in public if "progress" in name.lower()] == []
