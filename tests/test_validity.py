"""Stage A deterministic validity gate tests.

Every fixture is synthetic and CPU-only.  The synthetic camera below declares a
commanded-metres to image-pixels mapping so the calibrated direction check has
something real to be fitted against; it is a test fixture, not a claim about
any physical camera.  No test asserts a qualified measurement.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, List, Sequence, Tuple

import numpy as np
import pytest

from plumb.validity import (
    ALL_REASON_CODES,
    CALIBRATION_DEVELOPMENT,
    CALIBRATION_PRIMARY,
    FLOW_BLOCK_MATCHING,
    FLOW_FARNEBACK,
    MotionReference,
    ReferenceTrajectory,
    SOURCE_DEVELOPMENT_SYNTHETIC,
    SOURCE_REAL,
    StageAConfigurationError,
    StageAParameters,
    StageAValidityGate,
    ValidityReasonCode,
    ValidityReport,
    _cv2_available,
    resolve_flow_estimator,
)


FRAME_SIZE = 96
PATCH_SIZE = 40
PARAMETERS = StageAParameters(
    parameter_set_id="plumb-stage-a-test-params",
    frame_height=FRAME_SIZE,
    frame_width=FRAME_SIZE,
    flow_downsample_factor=1,
    grid_rows=8,
    grid_cols=8,
    block_search_radius=4,
    min_axis_map_transitions=4,
    min_reference_transitions=4,
)
# Declared fixture camera: commanded (dx, dy, dz) metres -> image (u, v) pixels.
FIXTURE_CAMERA = np.asarray([[150.0, 0.0, 75.0], [0.0, 150.0, -75.0]])

MOVE_X = (0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
MOVE_BACK_X = (-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
PRESS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4)


def build_frames(
    actions: Sequence[Sequence[float]],
    *,
    seed: int = 0,
    jitter: float = 0.0,
    start: Tuple[float, float] = (28.0, 28.0),
    background_maximum: int = 200,
    frozen: bool = False,
    global_shift: Tuple[int, int] | None = None,
) -> List[np.ndarray]:
    """Static textured scene plus one textured patch moved by the command."""

    rng = np.random.default_rng(seed)
    background = rng.integers(0, background_maximum + 1, size=(FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    patch = rng.integers(0, 256, size=(PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    frames: List[np.ndarray] = []
    row, column = start
    for index in range(len(actions) + 1):
        if global_shift is None:
            frame = background.copy()
        else:
            frame = np.roll(
                background,
                (int(round(global_shift[1] * index)), int(round(global_shift[0] * index))),
                axis=(0, 1),
            ).copy()
        top, left = int(round(row)), int(round(column))
        assert 0 <= top <= FRAME_SIZE - PATCH_SIZE and 0 <= left <= FRAME_SIZE - PATCH_SIZE
        frame[top : top + PATCH_SIZE, left : left + PATCH_SIZE] = patch
        frames.append(frame)
        if index < len(actions) and not frozen:
            horizontal, vertical = FIXTURE_CAMERA @ np.asarray(actions[index][0:3], dtype=np.float64)
            if jitter:
                horizontal += rng.normal(0.0, jitter)
                vertical += rng.normal(0.0, jitter)
            column += horizontal
            row += vertical
    return frames


def timestamps(action_count: int) -> List[float]:
    return [index / 5.0 for index in range(action_count + 1)]


def state_rows(action_count: int) -> List[Tuple[float, ...]]:
    return [(0.0,) * 8 for _ in range(action_count + 1)]


def reference_trajectories(source_kind: str = SOURCE_REAL) -> List[ReferenceTrajectory]:
    plans = {
        "x-forward": [MOVE_X] * 4,
        "x-back": [MOVE_BACK_X] * 4,
        "y-down": [(0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0)] * 4,
        "y-up": [(0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0)] * 4,
        "z-forward": [(0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0)] * 4,
        "mixed": [(0.01, 0.01, 0.02, 0.0, 0.0, 0.0, 0.0)] * 4,
        "stationary-contact": [PRESS] * 4,
    }
    trajectories: List[ReferenceTrajectory] = []
    for index, (name, actions) in enumerate(plans.items()):
        frames = build_frames(actions, seed=10 + index, jitter=0.8)
        trajectories.append(
            ReferenceTrajectory(
                trajectory_id="bridge-" + name,
                source_uri="fixture://bridge/" + name,
                source_sha256="%064x" % (index + 1),
                frames=tuple(frames),
                actions=tuple(actions),
                source_kind=source_kind,
                source_dataset="synthetic-development-fixture",
            )
        )
    return trajectories


@pytest.fixture(scope="module")
def motion_reference() -> MotionReference:
    return MotionReference.fit_from_trajectories(
        reference_trajectories(),
        reference_id="stage-a-test-reference",
        source_description="Synthetic development fixture standing in for real Bridge trajectories.",
        parameters=PARAMETERS,
    )


@pytest.fixture(scope="module")
def calibrated_gate(motion_reference: MotionReference) -> StageAValidityGate:
    return StageAValidityGate(PARAMETERS, motion_reference=motion_reference, calibration_class=CALIBRATION_PRIMARY)


def evaluate(gate: StageAValidityGate, actions, frames, **kwargs) -> ValidityReport:
    return gate.evaluate(
        frames=frames,
        actions=actions,
        nominal_timestamps=kwargs.pop("nominal_timestamps", timestamps(len(actions))),
        conditioning_frame=kwargs.pop("conditioning_frame", frames[0] if frames else None),
        states=kwargs.pop("states", state_rows(len(actions))),
        **kwargs,
    )


# --- parameter set and identity -------------------------------------------


def test_parameter_set_is_frozen_and_hashes_split_measurement_from_thresholds() -> None:
    with pytest.raises(Exception):
        PARAMETERS.grid_rows = 4  # type: ignore[misc]
    threshold_edit = replace(PARAMETERS, max_inconsistent_transition_fraction=0.9)
    geometry_edit = replace(PARAMETERS, grid_rows=6)
    assert threshold_edit.measurement_hash == PARAMETERS.measurement_hash
    assert threshold_edit.declared_hash != PARAMETERS.declared_hash
    assert geometry_edit.measurement_hash != PARAMETERS.measurement_hash


def test_parameters_hash_pins_the_resolved_estimator_and_exposes_no_vjepa_hook() -> None:
    gate = StageAValidityGate(PARAMETERS, calibration_class=CALIBRATION_DEVELOPMENT)
    contract = gate.describe()
    assert contract["parameters_hash"] == gate.parameters_hash
    assert contract["parameters_hash"].startswith("sha256:")
    assert contract["calibration_reference_hash"] is None
    assert contract["uses_learned_features"] is False
    assert contract["vjepa_hook"] is None
    assert contract["flow_estimator"] in (FLOW_FARNEBACK, FLOW_BLOCK_MATCHING)
    assert set(contract["reason_codes"]) == set(ALL_REASON_CODES)
    # The fixture parameters leave the estimator on "auto", which is reproducible
    # only within one image; describe() says so instead of implying otherwise.
    assert contract["declared_flow_estimator"] == "auto"
    assert "must pin flow_estimator" in contract["deployment_note"]
    pinned = StageAValidityGate(
        replace(PARAMETERS, flow_estimator=FLOW_BLOCK_MATCHING), calibration_class=CALIBRATION_DEVELOPMENT
    )
    assert "pinned explicitly" in pinned.describe()["deployment_note"]
    assert pinned.parameters_hash != gate.parameters_hash or pinned.flow_estimator == gate.flow_estimator
    import plumb.validity as validity_module

    assert not [name for name in dir(validity_module) if "jepa" in name.lower()]


def test_flow_estimator_resolution_is_explicit_about_optional_cv2() -> None:
    assert resolve_flow_estimator(FLOW_BLOCK_MATCHING) == FLOW_BLOCK_MATCHING
    if _cv2_available():
        assert resolve_flow_estimator("auto") == FLOW_FARNEBACK
        assert resolve_flow_estimator(FLOW_FARNEBACK) == FLOW_FARNEBACK
    else:
        assert resolve_flow_estimator("auto") == FLOW_BLOCK_MATCHING
        with pytest.raises(StageAConfigurationError, match="cv2 is not importable"):
            resolve_flow_estimator(FLOW_FARNEBACK)
    with pytest.raises(StageAConfigurationError):
        resolve_flow_estimator("hand_tuned_per_episode")


def test_parameters_reject_a_grid_without_a_scene_reference_band() -> None:
    with pytest.raises(StageAConfigurationError, match="nonempty interior"):
        StageAParameters(grid_rows=4, grid_cols=4, scene_reference_border_blocks=2)
    with pytest.raises(StageAConfigurationError, match="at least 3 rows"):
        StageAParameters(grid_rows=2, grid_cols=2)


# --- calibration gating ----------------------------------------------------


def test_calibrated_mode_refuses_to_run_without_a_motion_reference() -> None:
    gate = StageAValidityGate(PARAMETERS, calibration_class=CALIBRATION_PRIMARY)
    actions = [MOVE_X] * 5
    report = evaluate(gate, actions, build_frames(actions, seed=99))
    assert report.validity == "unknown"
    assert ValidityReasonCode.NO_MOTION_REFERENCE.value in report.reason_codes
    assert report.calibration_reference_hash is None
    assert report.primary_scoring_eligible is False
    assert gate.primary_scoring_eligible is False


def test_development_mode_is_labelled_and_can_never_be_primary_evidence(motion_reference) -> None:
    gate = StageAValidityGate(PARAMETERS, motion_reference=motion_reference, calibration_class=CALIBRATION_DEVELOPMENT)
    actions = [MOVE_X] * 5
    report = evaluate(gate, actions, build_frames(actions, seed=99))
    assert report.calibration_class == CALIBRATION_DEVELOPMENT
    assert report.primary_scoring_eligible is False
    assert ValidityReasonCode.UNCALIBRATED_DEVELOPMENT_MODE.value in report.reason_codes
    # A development-mode marker is informational and does not veto a verdict.
    assert report.validity == "valid"


def test_unknown_calibration_class_is_a_configuration_error() -> None:
    with pytest.raises(StageAConfigurationError):
        StageAValidityGate(PARAMETERS, calibration_class="good_enough")


def test_reference_records_n_percentiles_provenance_and_camera_fit(motion_reference) -> None:
    assert motion_reference.trajectory_count == 7
    assert motion_reference.transition_count == 28
    assert motion_reference.moving_transition_count == 24
    assert motion_reference.static_transition_count == 4
    assert set(motion_reference.relative_motion_ratio_percentiles) == {"p01", "p05", "p50", "p95", "p99"}
    assert motion_reference.frame_delta_mae_percentiles["p99"] > 0
    assert motion_reference.camera_axis_map is not None
    assert motion_reference.camera_axis_map_residual_rms is not None
    assert motion_reference.provenance_hash.startswith("sha256:")
    assert motion_reference.reference_hash.startswith("sha256:")
    assert len(motion_reference.provenance) == 7
    assert all(item["source_sha256"] for item in motion_reference.provenance)
    assert motion_reference.contains_synthetic_sources is False
    assert motion_reference.measurement_hash == PARAMETERS.measurement_hash
    json.dumps(motion_reference.as_dict())


def test_reference_fitting_refuses_missing_provenance_and_thin_evidence() -> None:
    actions = [MOVE_X] * 4
    frames = build_frames(actions, seed=1)
    with pytest.raises(StageAConfigurationError, match="SHA-256"):
        ReferenceTrajectory("t", "fixture://t", "not-a-digest", tuple(frames), tuple(actions))
    with pytest.raises(StageAConfigurationError, match="action_count \\+ 1"):
        ReferenceTrajectory("t", "fixture://t", "0" * 64, tuple(frames[:-1]), tuple(actions))
    with pytest.raises(StageAConfigurationError, match="zero trajectories"):
        MotionReference.fit_from_trajectories([], reference_id="empty", source_description="none", parameters=PARAMETERS)
    thin = ReferenceTrajectory("thin", "fixture://thin", "0" * 64, tuple(frames[:2]), tuple(actions[:1]))
    with pytest.raises(StageAConfigurationError, match="at least 4 transitions"):
        MotionReference.fit_from_trajectories(
            [thin], reference_id="thin", source_description="thin", parameters=PARAMETERS
        )


def test_synthetic_sourced_reference_is_refused_for_primary_but_usable_in_development() -> None:
    synthetic = MotionReference.fit_from_trajectories(
        reference_trajectories(source_kind=SOURCE_DEVELOPMENT_SYNTHETIC),
        reference_id="synthetic-reference",
        source_description="explicitly synthetic",
        parameters=PARAMETERS,
    )
    assert synthetic.contains_synthetic_sources is True
    actions = [MOVE_X] * 5
    frames = build_frames(actions, seed=99)

    primary = StageAValidityGate(PARAMETERS, motion_reference=synthetic, calibration_class=CALIBRATION_PRIMARY)
    report = evaluate(primary, actions, frames)
    assert report.validity == "unknown"
    assert ValidityReasonCode.MOTION_REFERENCE_NOT_REAL_DATA.value in report.reason_codes
    assert primary.primary_scoring_eligible is False

    development = StageAValidityGate(PARAMETERS, motion_reference=synthetic, calibration_class=CALIBRATION_DEVELOPMENT)
    assert evaluate(development, actions, frames).validity == "valid"


def test_reference_from_different_measurement_parameters_or_estimator_is_refused(motion_reference) -> None:
    other_geometry = replace(PARAMETERS, grid_rows=6, grid_cols=6)
    gate = StageAValidityGate(other_geometry, motion_reference=motion_reference, calibration_class=CALIBRATION_PRIMARY)
    actions = [MOVE_X] * 5
    report = evaluate(gate, actions, build_frames(actions, seed=99))
    assert report.validity == "unknown"
    assert ValidityReasonCode.MOTION_REFERENCE_PARAMETERS_MISMATCH.value in report.reason_codes

    foreign_estimator = MotionReference(
        **{
            **motion_reference.__dict__,
            "flow_estimator": FLOW_FARNEBACK if motion_reference.flow_estimator != FLOW_FARNEBACK else FLOW_BLOCK_MATCHING,
        }
    )
    mismatch = StageAValidityGate(
        PARAMETERS, motion_reference=foreign_estimator, calibration_class=CALIBRATION_PRIMARY
    )
    mismatch_report = evaluate(mismatch, actions, build_frames(actions, seed=99))
    assert mismatch_report.validity == "unknown"
    assert ValidityReasonCode.MOTION_REFERENCE_ESTIMATOR_MISMATCH.value in mismatch_report.reason_codes


# --- structural checks -----------------------------------------------------


def test_frame_count_must_be_action_count_plus_one(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    frames = build_frames(actions, seed=99)
    report = calibrated_gate.evaluate(
        frames=frames[:-1],
        actions=actions,
        nominal_timestamps=timestamps(5)[:-1],
        conditioning_frame=frames[0],
        states=state_rows(5),
    )
    assert report.validity == "invalid"
    assert ValidityReasonCode.FRAME_COUNT_MISMATCH.value in report.reason_codes
    assert report.checks["frame_count"] == {
        "observed": 5,
        "expected": 6,
        "rule": "action_count + 1 frames, conditioning frame at index 0",
        "ok": False,
    }
    assert report.checks["motion_consistency"]["status"] == "not_run"


def test_nonmonotone_or_miscounted_timestamps_are_invalid(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    frames = build_frames(actions, seed=99)
    shuffled = timestamps(5)
    shuffled[2], shuffled[3] = shuffled[3], shuffled[2]
    report = evaluate(calibrated_gate, actions, frames, nominal_timestamps=shuffled)
    assert report.validity == "invalid"
    assert ValidityReasonCode.NONMONOTONE_TIMESTAMPS.value in report.reason_codes

    short = evaluate(calibrated_gate, actions, frames, nominal_timestamps=timestamps(5)[:-1])
    assert ValidityReasonCode.TIMESTAMP_COUNT_MISMATCH.value in short.reason_codes


def test_conditioning_frame_must_be_frame_zero(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    frames = build_frames(actions, seed=99)
    other = build_frames(actions, seed=5)[0]

    mismatch = evaluate(calibrated_gate, actions, frames, conditioning_frame=other)
    assert mismatch.validity == "invalid"
    assert ValidityReasonCode.CONDITIONING_FRAME_MISMATCH.value in mismatch.reason_codes

    unverified = evaluate(calibrated_gate, actions, frames, conditioning_frame=None)
    assert unverified.validity == "unknown"
    assert ValidityReasonCode.CONDITIONING_FRAME_UNVERIFIED.value in unverified.reason_codes
    assert unverified.checks["conditioning_frame"]["supplied"] is False

    verified = evaluate(calibrated_gate, actions, frames)
    assert verified.checks["conditioning_frame"]["verified"] is True


@pytest.mark.parametrize(
    "corrupt,expected",
    [
        ("string", ValidityReasonCode.DECODE_FAILURE),
        ("bytes", ValidityReasonCode.DECODE_FAILURE),
        ("shape", ValidityReasonCode.FRAME_SHAPE_MISMATCH),
        ("dtype", ValidityReasonCode.FRAME_DTYPE_MISMATCH),
        ("black", ValidityReasonCode.CONSTANT_FRAME),
        ("constant", ValidityReasonCode.CONSTANT_FRAME),
        ("nan", ValidityReasonCode.NONFINITE_FRAME),
        ("range", ValidityReasonCode.FRAME_VALUE_RANGE_INVALID),
    ],
)
def test_decode_integrity_failures_are_invalid_with_their_own_code(calibrated_gate, corrupt, expected) -> None:
    actions = [MOVE_X] * 5
    frames: List[Any] = list(build_frames(actions, seed=99))
    if corrupt == "string":
        frames[3] = "generated:3"
    elif corrupt == "bytes":
        frames[3] = b"\x89PNG not decoded here"
    elif corrupt == "shape":
        frames[3] = np.zeros((10, 10, 3), dtype=np.uint8)
    elif corrupt == "dtype":
        frames[3] = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.int32) + 40
    elif corrupt == "black":
        frames[3] = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    elif corrupt == "constant":
        frames[3] = np.full((FRAME_SIZE, FRAME_SIZE, 3), 128, dtype=np.uint8)
    elif corrupt == "nan":
        broken = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32) + 0.5
        broken[0, 0, 0] = np.nan
        frames[3] = broken
    else:
        frames[3] = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32) + 7.0
    report = evaluate(calibrated_gate, actions, frames)
    assert report.validity == "invalid"
    assert expected.value in report.reason_codes
    assert report.checks["decode"]["failures"]
    assert report.checks["decode"]["failures"][0]["frame_index"] == 3


def test_float01_and_channel_first_frames_decode_without_being_rescaled(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    frames = build_frames(actions, seed=99)
    converted: List[Any] = [np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0 for frame in frames]
    report = evaluate(calibrated_gate, actions, converted, conditioning_frame=converted[0])
    assert report.validity == "valid"
    assert report.checks["decode"]["observed_layouts"] == ["chw_transposed"]

    strict = StageAValidityGate(
        replace(PARAMETERS, accept_float01_frames=False),
        calibration_class=CALIBRATION_DEVELOPMENT,
    )
    strict_report = evaluate(strict, actions, converted, conditioning_frame=converted[0])
    assert ValidityReasonCode.FRAME_DTYPE_MISMATCH.value in strict_report.reason_codes


def test_nonfinite_actions_and_states_are_invalid(calibrated_gate) -> None:
    actions: List[Any] = [list(MOVE_X) for _ in range(5)]
    frames = build_frames([MOVE_X] * 5, seed=99)
    actions[2][1] = float("inf")
    report = evaluate(calibrated_gate, actions, frames)
    assert report.validity == "invalid"
    assert ValidityReasonCode.NONFINITE_ACTION.value in report.reason_codes
    assert report.checks["finite_values"]["nonfinite_action_rows"] == [2]

    states = [list(row) for row in state_rows(5)]
    states[1][0] = float("nan")
    state_report = evaluate(calibrated_gate, [MOVE_X] * 5, frames, states=states)
    assert state_report.validity == "invalid"
    assert ValidityReasonCode.NONFINITE_STATE.value in state_report.reason_codes


def test_wrong_action_width_is_invalid_not_a_crash(calibrated_gate) -> None:
    actions = [(0.02, 0.0, 0.0)] * 5
    report = evaluate(calibrated_gate, actions, build_frames([MOVE_X] * 5, seed=99))
    assert report.validity == "invalid"
    assert ValidityReasonCode.ACTION_WIDTH_MISMATCH.value in report.reason_codes


def test_absent_state_values_are_unknown_never_valid(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    report = evaluate(calibrated_gate, actions, build_frames(actions, seed=99), states=None)
    assert report.validity == "unknown"
    assert ValidityReasonCode.STATE_VALUES_NOT_SUPPLIED.value in report.reason_codes
    assert report.checks["finite_values"]["states_supplied"] is False


def test_state_rows_must_match_the_action_count_plus_one(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    report = evaluate(calibrated_gate, actions, build_frames(actions, seed=99), states=state_rows(5)[:-1])
    assert report.validity == "invalid"
    assert ValidityReasonCode.STATE_COUNT_MISMATCH.value in report.reason_codes
    assert report.checks["finite_values"]["state_rows_expected"] == 6
    assert report.checks["finite_values"]["state_row_count_ok"] is False


# --- dynamics --------------------------------------------------------------


def test_impossible_discontinuity_uses_the_calibrated_bound(calibrated_gate, motion_reference) -> None:
    actions = [MOVE_X] * 5
    frames = list(build_frames(actions, seed=99))
    frames[3] = np.asarray(255 - frames[3], dtype=np.uint8)
    report = evaluate(calibrated_gate, actions, frames)
    assert report.validity == "invalid"
    assert ValidityReasonCode.IMPOSSIBLE_DISCONTINUITY.value in report.reason_codes
    bound = report.checks["discontinuity"]["bound_0_255"]
    assert bound == pytest.approx(motion_reference.frame_delta_mae_percentiles["p99"] * 1.5, rel=1e-6)
    assert report.checks["discontinuity"]["bound_source"] == "reference_p99*1.500"
    assert report.checks["discontinuity"]["violating_transitions"] == [2, 3]


def test_command_consistent_motion_is_valid(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    report = evaluate(calibrated_gate, actions, build_frames(actions, seed=99))
    assert report.validity == "valid"
    assert report.reason_codes == ()
    assert report.primary_scoring_eligible is True
    assert report.checks["motion_consistency"]["verdict"] == "consistent"
    assert report.expected_static is False
    assert report.expected_static_source == "derived_from_commanded_magnitude"


def test_reversed_commanded_direction_is_command_motion_inconsistent(calibrated_gate) -> None:
    frames = build_frames([MOVE_X] * 5, seed=99)
    report = evaluate(calibrated_gate, [MOVE_BACK_X] * 5, frames)
    assert report.validity == "invalid"
    assert ValidityReasonCode.COMMAND_MOTION_INCONSISTENT.value in report.reason_codes
    assert all(item["direction_cosine"] < 0 for item in report.checks["transitions"])


def test_frozen_video_under_a_real_command_is_below_the_reference(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    report = evaluate(calibrated_gate, actions, build_frames(actions, seed=7, frozen=True))
    assert report.validity == "invalid"
    assert ValidityReasonCode.MOTION_BELOW_REFERENCE.value in report.reason_codes


def test_a_stationary_gripper_pressing_a_drawer_is_valid(calibrated_gate) -> None:
    """The spec's named false positive: near-static contact must not be invalid."""

    actions = [PRESS] * 5
    frames = build_frames(actions, seed=42)
    derived = evaluate(calibrated_gate, actions, frames)
    assert derived.validity == "valid"
    assert derived.reason_codes == ()
    assert derived.expected_static is True
    assert derived.checks["motion_consistency"]["static_transitions"] == [0, 1, 2, 3, 4]
    assert derived.checks["motion_consistency"]["static_transitions_are_admissible"] is True
    assert ValidityReasonCode.MOTION_BELOW_REFERENCE.value not in derived.reason_codes


def test_explicit_expected_static_hint_overrides_the_derived_value(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    frames = build_frames(actions, seed=7, frozen=True)
    hinted = evaluate(calibrated_gate, actions, frames, expected_static=True)
    assert hinted.validity == "valid"
    assert hinted.expected_static is True
    assert hinted.expected_static_source == "caller_hint"
    # Without the hint the same clip is a real motion failure.
    assert evaluate(calibrated_gate, actions, frames).validity == "invalid"


def test_gross_unexplained_motion_under_a_static_command_is_inconsistent(calibrated_gate) -> None:
    moving_frames = build_frames([MOVE_X] * 5, seed=99)
    report = evaluate(calibrated_gate, [PRESS] * 5, moving_frames)
    assert report.validity == "invalid"
    assert ValidityReasonCode.COMMAND_MOTION_INCONSISTENT.value in report.reason_codes
    assert report.expected_static is True


def test_global_flow_magnitude_alone_never_satisfies_the_check(calibrated_gate) -> None:
    """A whole-scene shift produces large flow but no arm motion relative to it."""

    actions = [MOVE_X] * 5
    frames = build_frames(actions, seed=5, frozen=True, global_shift=(3, 0))
    report = evaluate(calibrated_gate, actions, frames)
    first = report.checks["transitions"][0]
    assert first["active_interior_flow_magnitude"] >= 2.0
    assert first["scene_reference_flow_magnitude"] >= 2.0
    assert first["relative_motion_ratio"] == pytest.approx(1.0, abs=0.2)
    assert report.checks["motion_consistency"]["verdict"] != "consistent"
    assert ValidityReasonCode.MOTION_BELOW_REFERENCE.value in report.reason_codes


def test_low_texture_scene_is_unknown_not_invalid(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    frames = build_frames(actions, seed=3, background_maximum=1, frozen=True)
    report = evaluate(calibrated_gate, actions, frames)
    assert report.validity == "unknown"
    assert ValidityReasonCode.AMBIGUOUS_LOW_TEXTURE.value in report.reason_codes
    assert ValidityReasonCode.MOTION_BELOW_REFERENCE.value not in report.reason_codes


def test_missing_camera_direction_reference_is_unknown_not_valid() -> None:
    reference = MotionReference.fit_from_trajectories(
        reference_trajectories(),
        reference_id="no-axis-map",
        source_description="rank-deficient commanded axes",
        parameters=PARAMETERS,
    )
    stripped = MotionReference(**{**reference.__dict__, "camera_axis_map": None})
    gate = StageAValidityGate(PARAMETERS, motion_reference=stripped, calibration_class=CALIBRATION_PRIMARY)
    actions = [MOVE_X] * 5
    report = evaluate(gate, actions, build_frames(actions, seed=99))
    assert report.validity == "unknown"
    assert ValidityReasonCode.NO_DIRECTION_REFERENCE.value in report.reason_codes


def test_rank_deficient_commanded_axes_do_not_produce_a_camera_map() -> None:
    actions = [MOVE_X] * 5
    trajectories = [
        ReferenceTrajectory(
            "only-x-%d" % index,
            "fixture://only-x/%d" % index,
            "%064x" % (index + 40),
            tuple(build_frames(actions, seed=200 + index, jitter=0.5)),
            tuple(actions),
        )
        for index in range(3)
    ]
    reference = MotionReference.fit_from_trajectories(
        trajectories, reference_id="rank-deficient", source_description="single commanded axis", parameters=PARAMETERS
    )
    assert reference.camera_axis_map is None
    assert reference.direction_cosine_percentiles == {}


# --- report contract -------------------------------------------------------


def test_reason_codes_are_a_closed_enum_and_valid_carries_no_verdict_code() -> None:
    with pytest.raises(StageAConfigurationError, match="closed enum"):
        ValidityReport(
            validity="invalid",
            reason_codes=("looks_wrong_to_me",),
            calibration_class=CALIBRATION_PRIMARY,
            primary_scoring_eligible=True,
            parameters_hash="sha256:x",
            calibration_reference_hash=None,
            parameter_set_id="p",
            flow_estimator=FLOW_BLOCK_MATCHING,
            expected_static=None,
            expected_static_source="derived_from_commanded_magnitude",
        )
    with pytest.raises(StageAConfigurationError, match="cannot carry verdict reason codes"):
        ValidityReport(
            validity="valid",
            reason_codes=(ValidityReasonCode.MOTION_BELOW_REFERENCE.value,),
            calibration_class=CALIBRATION_PRIMARY,
            primary_scoring_eligible=True,
            parameters_hash="sha256:x",
            calibration_reference_hash=None,
            parameter_set_id="p",
            flow_estimator=FLOW_BLOCK_MATCHING,
            expected_static=None,
            expected_static_source="derived_from_commanded_magnitude",
        )
    with pytest.raises(StageAConfigurationError):
        ValidityReport(
            validity="probably_fine",
            reason_codes=(),
            calibration_class=CALIBRATION_PRIMARY,
            primary_scoring_eligible=True,
            parameters_hash="sha256:x",
            calibration_reference_hash=None,
            parameter_set_id="p",
            flow_estimator=FLOW_BLOCK_MATCHING,
            expected_static=None,
            expected_static_source="derived_from_commanded_magnitude",
        )


def test_report_is_json_serialisable_and_carries_both_deployment_hashes(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    report = evaluate(calibrated_gate, actions, build_frames(actions, seed=99))
    payload = report.as_dict()
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert '"validity": "valid"' in encoded
    assert payload["parameters_hash"] == calibrated_gate.parameters_hash
    assert payload["calibration_reference_hash"] == calibrated_gate.calibration_reference_hash
    assert payload["gate_version"] == "plumb-stage-a-gate-v1"
    assert payload["checks"]["bounds"]["source"].startswith("motion_reference:")


def test_zero_action_episode_has_no_transitions_and_is_unknown(calibrated_gate) -> None:
    frames = build_frames([MOVE_X], seed=99)[:1]
    report = calibrated_gate.evaluate(
        frames=frames,
        actions=[],
        nominal_timestamps=[0.0],
        conditioning_frame=frames[0],
        states=state_rows(0),
    )
    assert report.validity == "unknown"
    assert ValidityReasonCode.NO_TRANSITIONS.value in report.reason_codes


def test_evaluation_is_deterministic_for_the_same_inputs(calibrated_gate) -> None:
    actions = [MOVE_X] * 5
    frames = build_frames(actions, seed=99)
    first = evaluate(calibrated_gate, actions, frames)
    second = evaluate(calibrated_gate, actions, frames)
    assert first.as_dict() == second.as_dict()
