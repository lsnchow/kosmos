"""Stage A deterministic validity checks for generated PLUMB rollouts.

This gate is **Python, not a model call**.  It runs frame-count/order, decode
integrity, finite action/state, impossible-discontinuity and calibrated
command-motion-consistency checks with a frozen, hashable parameter set.  No
learned feature extractor participates: V-JEPA feature diagnostics are a
separate analysis and deliberately have no hook here, so they cannot silently
alter this deterministic gate (spec 5).

Stated limitations, all load-bearing:

* Whole-image optical-flow magnitude cannot identify arm motion, contact
  failure or physical collisions.  This module therefore never thresholds a
  global flow magnitude.  It compares an *interior* high-percentile block
  magnitude against a *scene/camera reference band* magnitude, and compares
  that relative ratio against a distribution fitted from real trajectories.
* A stationary gripper pressing a drawer can be valid.  Transitions whose
  commanded magnitude is below a declared static threshold are checked only
  for gross *unexplained* motion, never for missing motion.  Callers can also
  force the hint with ``expected_static``.
* Ambiguous observations are ``unknown``, not invalid.  Low scene texture, an
  unmeasurable flow direction, a missing motion reference and an unperformed
  check all produce ``unknown`` with an explicit reason code.
* Refusing to guess is the default.  In ``calibrated_primary`` mode the gate
  will not run without a :class:`MotionReference` fitted from real
  trajectories; it returns ``unknown`` / ``no_motion_reference`` instead.
* The direction check has low power against a real reference distribution,
  because real Bridge trajectories themselves contain ticks whose image-plane
  flow direction is unrelated to commanded translation.  That is reported
  honestly rather than compensated with a tighter invented bound.

``numpy`` is a declared project dependency and is imported at module scope;
``cv2`` is optional.  When OpenCV is absent a pure-numpy integer-offset block
matcher is used instead.  The two estimators are *different measurements*, so a
:class:`MotionReference` records which one fitted it and the gate refuses a
mismatched reference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np


STAGE_A_GATE_VERSION = "plumb-stage-a-gate-v1"

CALIBRATION_PRIMARY = "calibrated_primary"
CALIBRATION_DEVELOPMENT = "uncalibrated_development"
CALIBRATION_CLASSES = (CALIBRATION_PRIMARY, CALIBRATION_DEVELOPMENT)

FLOW_FARNEBACK = "farneback_cv2_fixed_v1"
FLOW_BLOCK_MATCHING = "numpy_block_matching_v1"
FLOW_ESTIMATORS = (FLOW_FARNEBACK, FLOW_BLOCK_MATCHING)

SOURCE_REAL = "real_bridge_trajectory"
SOURCE_DEVELOPMENT_SYNTHETIC = "development_synthetic"
SOURCE_KINDS = (SOURCE_REAL, SOURCE_DEVELOPMENT_SYNTHETIC)

VALIDITY_VALID = "valid"
VALIDITY_INVALID = "invalid"
VALIDITY_UNKNOWN = "unknown"


class StageAConfigurationError(ValueError):
    """The gate itself is misconfigured; this is never an episode verdict."""


class ValidityReasonCode(str, Enum):
    """Closed reason-code enum. Consumers must not invent new strings."""

    # Structural frame/order checks.
    FRAME_COUNT_MISMATCH = "frame_count_mismatch"
    TIMESTAMP_COUNT_MISMATCH = "timestamp_count_mismatch"
    NONMONOTONE_TIMESTAMPS = "nonmonotone_timestamps"
    CONDITIONING_FRAME_MISMATCH = "conditioning_frame_mismatch"
    CONDITIONING_FRAME_UNVERIFIED = "conditioning_frame_unverified"
    # Decode integrity.
    DECODE_FAILURE = "decode_failure"
    FRAME_SHAPE_MISMATCH = "frame_shape_mismatch"
    FRAME_DTYPE_MISMATCH = "frame_dtype_mismatch"
    FRAME_VALUE_RANGE_INVALID = "frame_value_range_invalid"
    NONFINITE_FRAME = "nonfinite_frame"
    CONSTANT_FRAME = "constant_frame"
    # Finite numeric inputs.
    ACTION_WIDTH_MISMATCH = "action_width_mismatch"
    NONFINITE_ACTION = "nonfinite_action"
    NONFINITE_STATE = "nonfinite_state"
    STATE_VALUES_NOT_SUPPLIED = "state_values_not_supplied"
    STATE_COUNT_MISMATCH = "state_count_mismatch"
    # Dynamics.
    IMPOSSIBLE_DISCONTINUITY = "impossible_discontinuity"
    COMMAND_MOTION_INCONSISTENT = "command_motion_inconsistent"
    MOTION_BELOW_REFERENCE = "motion_below_reference"
    # Ambiguity and refusal to guess.
    AMBIGUOUS_LOW_TEXTURE = "ambiguous_low_texture"
    AMBIGUOUS_MOTION_MAGNITUDE = "ambiguous_motion_magnitude"
    AMBIGUOUS_MOTION_DIRECTION = "ambiguous_motion_direction"
    NO_TRANSITIONS = "no_transitions"
    NO_MOTION_REFERENCE = "no_motion_reference"
    NO_DIRECTION_REFERENCE = "no_direction_reference"
    MOTION_REFERENCE_ESTIMATOR_MISMATCH = "motion_reference_estimator_mismatch"
    MOTION_REFERENCE_PARAMETERS_MISMATCH = "motion_reference_parameters_mismatch"
    MOTION_REFERENCE_NOT_REAL_DATA = "motion_reference_not_real_data"
    # Mode marker; informational, never a verdict on its own.
    UNCALIBRATED_DEVELOPMENT_MODE = "uncalibrated_development_mode"


INVALID_REASON_CODES: frozenset = frozenset(
    {
        ValidityReasonCode.FRAME_COUNT_MISMATCH,
        ValidityReasonCode.TIMESTAMP_COUNT_MISMATCH,
        ValidityReasonCode.NONMONOTONE_TIMESTAMPS,
        ValidityReasonCode.CONDITIONING_FRAME_MISMATCH,
        ValidityReasonCode.DECODE_FAILURE,
        ValidityReasonCode.FRAME_SHAPE_MISMATCH,
        ValidityReasonCode.FRAME_DTYPE_MISMATCH,
        ValidityReasonCode.FRAME_VALUE_RANGE_INVALID,
        ValidityReasonCode.NONFINITE_FRAME,
        ValidityReasonCode.CONSTANT_FRAME,
        ValidityReasonCode.ACTION_WIDTH_MISMATCH,
        ValidityReasonCode.NONFINITE_ACTION,
        ValidityReasonCode.NONFINITE_STATE,
        ValidityReasonCode.STATE_COUNT_MISMATCH,
        ValidityReasonCode.IMPOSSIBLE_DISCONTINUITY,
        ValidityReasonCode.COMMAND_MOTION_INCONSISTENT,
        ValidityReasonCode.MOTION_BELOW_REFERENCE,
    }
)

UNKNOWN_REASON_CODES: frozenset = frozenset(
    {
        ValidityReasonCode.CONDITIONING_FRAME_UNVERIFIED,
        ValidityReasonCode.STATE_VALUES_NOT_SUPPLIED,
        ValidityReasonCode.AMBIGUOUS_LOW_TEXTURE,
        ValidityReasonCode.AMBIGUOUS_MOTION_MAGNITUDE,
        ValidityReasonCode.AMBIGUOUS_MOTION_DIRECTION,
        ValidityReasonCode.NO_TRANSITIONS,
        ValidityReasonCode.NO_MOTION_REFERENCE,
        ValidityReasonCode.NO_DIRECTION_REFERENCE,
        ValidityReasonCode.MOTION_REFERENCE_ESTIMATOR_MISMATCH,
        ValidityReasonCode.MOTION_REFERENCE_PARAMETERS_MISMATCH,
        ValidityReasonCode.MOTION_REFERENCE_NOT_REAL_DATA,
    }
)

INFORMATIONAL_REASON_CODES: frozenset = frozenset({ValidityReasonCode.UNCALIBRATED_DEVELOPMENT_MODE})

ALL_REASON_CODES: Tuple[str, ...] = tuple(sorted(code.value for code in ValidityReasonCode))

# Parameters that change a *measured* value.  A MotionReference is bound to
# these only, so a decision-threshold edit does not force a refit while a
# geometry/decode/flow edit does.
_MEASUREMENT_FIELDS: Tuple[str, ...] = (
    "frame_height",
    "frame_width",
    "frame_channels",
    "accept_float01_frames",
    "accept_channel_first_frames",
    "flow_downsample_factor",
    "grid_rows",
    "grid_cols",
    "scene_reference_border_blocks",
    "active_block_percentile",
    "scene_reference_floor",
    "flow_estimator",
    "block_search_radius",
    "farneback_pyr_scale",
    "farneback_levels",
    "farneback_winsize",
    "farneback_iterations",
    "farneback_poly_n",
    "farneback_poly_sigma",
    "rotation_lever_arm_metres",
    "static_command_magnitude",
)

_PERCENTILE_KEYS: Tuple[Tuple[str, float], ...] = (
    ("p01", 1.0),
    ("p05", 5.0),
    ("p50", 50.0),
    ("p95", 95.0),
    ("p99", 99.0),
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _round(value: Optional[float], places: int = 6) -> Optional[float]:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return round(numeric, places)


def _is_hex_sha256(value: Any) -> bool:
    text = str(value or "")
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


@dataclass(frozen=True)
class StageAParameters:
    """Frozen, hashable Stage A parameter set.

    ``declared_hash`` covers every field.  ``measurement_hash`` covers only the
    fields that change a measured quantity and is what binds a fitted
    :class:`MotionReference` to this pipeline.
    """

    parameter_set_id: str = "plumb-stage-a-params-v1"
    # Decode integrity and expected frame contract.
    frame_height: int = 256
    frame_width: int = 256
    frame_channels: int = 3
    accept_float01_frames: bool = True
    accept_channel_first_frames: bool = True
    min_frame_pixel_std: float = 0.5
    min_frame_max_value: float = 1.0
    # Flow geometry and the camera/scene reference band.
    flow_downsample_factor: int = 2
    grid_rows: int = 8
    grid_cols: int = 8
    scene_reference_border_blocks: int = 1
    active_block_percentile: float = 90.0
    scene_reference_floor: float = 0.05
    # Flow estimator, fixed parameters only.
    flow_estimator: str = "auto"
    block_search_radius: int = 4
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_winsize: int = 15
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2
    # Commanded-magnitude model.  The lever arm is a declared scale used to put
    # commanded rotation and translation on one axis; it is not a measurement.
    rotation_lever_arm_metres: float = 0.05
    static_command_magnitude: float = 0.004
    # Decision thresholds.  Values prefixed ``uncalibrated_`` are used only in
    # uncalibrated_development mode and can never gate primary scoring.
    static_max_relative_motion: float = 8.0
    static_bound_multiplier: float = 1.5
    discontinuity_bound_multiplier: float = 1.5
    min_direction_flow_magnitude: float = 0.35
    max_inconsistent_transition_fraction: float = 0.25
    max_ambiguous_transition_fraction: float = 0.5
    uncalibrated_frame_delta_mae_bound: float = 96.0
    uncalibrated_min_relative_motion: float = 1.15
    uncalibrated_ambiguous_relative_motion: float = 1.35
    uncalibrated_min_texture_gradient: float = 1.0
    uncalibrated_direction_cosine_min: float = -0.35
    uncalibrated_direction_cosine_ambiguous: float = 0.1
    # Reference fitting guards.
    min_axis_map_transitions: int = 6
    axis_map_singular_value_floor: float = 1e-6
    min_reference_transitions: int = 8

    def __post_init__(self) -> None:
        if not self.parameter_set_id:
            raise StageAConfigurationError("parameter_set_id is required")
        if self.frame_channels != 3:
            raise StageAConfigurationError("Stage A decodes 3-channel RGB frames only")
        if self.frame_height < 4 or self.frame_width < 4:
            raise StageAConfigurationError("frame_height/frame_width must be at least 4")
        if self.flow_downsample_factor < 1:
            raise StageAConfigurationError("flow_downsample_factor must be >= 1")
        if self.grid_rows < 3 or self.grid_cols < 3:
            raise StageAConfigurationError("the block grid needs at least 3 rows and columns for a reference band")
        border = self.scene_reference_border_blocks
        if border < 1 or self.grid_rows <= 2 * border or self.grid_cols <= 2 * border:
            raise StageAConfigurationError(
                "scene_reference_border_blocks must leave a nonempty interior region"
            )
        reduced_height = self.frame_height // self.flow_downsample_factor
        reduced_width = self.frame_width // self.flow_downsample_factor
        if reduced_height < self.grid_rows or reduced_width < self.grid_cols:
            raise StageAConfigurationError("downsampled frames are too small for the requested block grid")
        if not 50.0 <= self.active_block_percentile <= 100.0:
            raise StageAConfigurationError("active_block_percentile must be between 50 and 100")
        if self.flow_estimator not in ("auto",) + FLOW_ESTIMATORS:
            raise StageAConfigurationError("flow_estimator must be auto, %s, or %s" % FLOW_ESTIMATORS)
        if self.block_search_radius < 1:
            raise StageAConfigurationError("block_search_radius must be >= 1")
        if self.scene_reference_floor <= 0:
            raise StageAConfigurationError("scene_reference_floor must be positive to bound the motion ratio")
        if self.static_command_magnitude < 0:
            raise StageAConfigurationError("static_command_magnitude cannot be negative")
        if not 0.0 < self.max_inconsistent_transition_fraction <= 1.0:
            raise StageAConfigurationError("max_inconsistent_transition_fraction must be in (0, 1]")
        if not 0.0 < self.max_ambiguous_transition_fraction <= 1.0:
            raise StageAConfigurationError("max_ambiguous_transition_fraction must be in (0, 1]")
        if self.min_reference_transitions < 4:
            raise StageAConfigurationError("a motion reference needs at least four transitions")

    def as_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in sorted(asdict(self).items())}

    @property
    def declared_hash(self) -> str:
        return _canonical_hash(self.as_dict())

    @property
    def measurement_hash(self) -> str:
        payload = {key: getattr(self, key) for key in _MEASUREMENT_FIELDS}
        payload["measurement_contract"] = "plumb-stage-a-measurement-v1"
        return _canonical_hash(payload)


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------


def _decode_frame(frame: Any, parameters: StageAParameters) -> Tuple[Optional[np.ndarray], Optional[ValidityReasonCode], Dict[str, Any]]:
    """Decode one frame to ``uint8`` HWC, or report why it cannot be decoded.

    ``bytes`` are refused: Stage A does not guess an image codec.  Callers must
    decode with their pinned decoder before this gate runs.
    """

    detail: Dict[str, Any] = {}
    if isinstance(frame, (bytes, bytearray, memoryview)):
        detail["reason"] = "raw bytes require an explicit pinned decoder; Stage A does not guess a codec"
        return None, ValidityReasonCode.DECODE_FAILURE, detail
    try:
        array = np.asarray(frame)
    except (TypeError, ValueError) as error:
        detail["reason"] = "%s: %s" % (type(error).__name__, error)
        return None, ValidityReasonCode.DECODE_FAILURE, detail
    if array.dtype == np.dtype(object) or array.dtype.kind in ("U", "S", "V", "O"):
        detail["reason"] = "array dtype %s is not decodable pixel data" % array.dtype
        detail["dtype"] = str(array.dtype)
        return None, ValidityReasonCode.DECODE_FAILURE, detail

    layout = "hwc"
    if array.ndim == 3 and array.shape[0] == parameters.frame_channels and array.shape[-1] != parameters.frame_channels:
        if not parameters.accept_channel_first_frames:
            detail["reason"] = "channel-first frames are refused by this parameter set"
            detail["shape"] = [int(value) for value in array.shape]
            return None, ValidityReasonCode.FRAME_SHAPE_MISMATCH, detail
        array = np.transpose(array, (1, 2, 0))
        layout = "chw_transposed"
    detail["layout"] = layout
    detail["shape"] = [int(value) for value in array.shape]
    detail["dtype"] = str(array.dtype)

    expected = (parameters.frame_height, parameters.frame_width, parameters.frame_channels)
    if array.ndim != 3 or tuple(int(value) for value in array.shape) != expected:
        detail["expected_shape"] = list(expected)
        return None, ValidityReasonCode.FRAME_SHAPE_MISMATCH, detail

    if array.dtype.kind == "f":
        finite = np.isfinite(array)
        if not bool(finite.all()):
            detail["nonfinite_count"] = int(np.size(array) - int(finite.sum()))
            return None, ValidityReasonCode.NONFINITE_FRAME, detail
        if not parameters.accept_float01_frames:
            return None, ValidityReasonCode.FRAME_DTYPE_MISMATCH, detail
        minimum = float(array.min())
        maximum = float(array.max())
        detail["float_range"] = [_round(minimum), _round(maximum)]
        if minimum < -1e-4 or maximum > 1.0 + 1e-4:
            return None, ValidityReasonCode.FRAME_VALUE_RANGE_INVALID, detail
        decoded = np.clip(array.astype(np.float64) * 255.0, 0.0, 255.0).round().astype(np.uint8)
        detail["converted_from"] = "float01"
        return decoded, None, detail
    if array.dtype == np.uint8:
        return array, None, detail
    return None, ValidityReasonCode.FRAME_DTYPE_MISMATCH, detail


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Fixed BT.601 luma; no adaptive normalization is applied anywhere."""

    values = frame.astype(np.float64)
    return 0.299 * values[..., 0] + 0.587 * values[..., 1] + 0.114 * values[..., 2]


def _downsample(gray: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return gray
    return gray[::factor, ::factor]


def _block_mean(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    height, width = values.shape
    block_height = height // rows
    block_width = width // cols
    cropped = values[: block_height * rows, : block_width * cols]
    return cropped.reshape(rows, block_height, cols, block_width).mean(axis=(1, 3))


def _shift(values: np.ndarray, row_offset: int, column_offset: int) -> np.ndarray:
    radius = max(abs(row_offset), abs(column_offset))
    if radius == 0:
        return values
    padded = np.pad(values, radius, mode="edge")
    height, width = values.shape
    top = radius + row_offset
    left = radius + column_offset
    return padded[top : top + height, left : left + width]


def _texture_energy(gray: np.ndarray) -> float:
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    horizontal = float(np.abs(np.diff(gray, axis=1)).mean())
    vertical = float(np.abs(np.diff(gray, axis=0)).mean())
    return 0.5 * (horizontal + vertical)


def resolve_flow_estimator(declared: str) -> str:
    """Resolve ``auto`` against the actually importable estimator.

    OpenCV is an optional import.  The resolved name is recorded in every
    report and enforced against the motion reference, because Farneback and the
    numpy block matcher are different measurements.
    """

    if declared in FLOW_ESTIMATORS:
        if declared == FLOW_FARNEBACK and not _cv2_available():
            raise StageAConfigurationError(
                "flow_estimator=%s was requested but cv2 is not importable; install OpenCV or use %s"
                % (FLOW_FARNEBACK, FLOW_BLOCK_MATCHING)
            )
        return declared
    if declared != "auto":
        raise StageAConfigurationError("unknown flow_estimator %r" % declared)
    return FLOW_FARNEBACK if _cv2_available() else FLOW_BLOCK_MATCHING


def _cv2_available() -> bool:
    """Whether OpenCV is importable. It is an optional dependency by design."""

    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover - depends on the host environment.
        return False
    return callable(getattr(cv2, "calcOpticalFlowFarneback", None))


def _farneback_block_flow(previous: np.ndarray, current: np.ndarray, parameters: StageAParameters) -> np.ndarray:
    import cv2  # type: ignore

    flow = cv2.calcOpticalFlowFarneback(
        previous.astype(np.float32),
        current.astype(np.float32),
        None,
        parameters.farneback_pyr_scale,
        parameters.farneback_levels,
        parameters.farneback_winsize,
        parameters.farneback_iterations,
        parameters.farneback_poly_n,
        parameters.farneback_poly_sigma,
        0,
    )
    horizontal = _block_mean(np.asarray(flow[..., 0], dtype=np.float64), parameters.grid_rows, parameters.grid_cols)
    vertical = _block_mean(np.asarray(flow[..., 1], dtype=np.float64), parameters.grid_rows, parameters.grid_cols)
    return np.stack([horizontal, vertical], axis=-1)


def _block_matching_flow(previous: np.ndarray, current: np.ndarray, parameters: StageAParameters) -> np.ndarray:
    """Integer-offset block matching with deterministic tie-breaking.

    Candidate offsets are evaluated in a fixed order and a strictly lower SAD
    is required to replace the incumbent, so the result does not depend on
    numpy reduction order.
    """

    radius = parameters.block_search_radius
    rows = parameters.grid_rows
    cols = parameters.grid_cols
    best_cost: Optional[np.ndarray] = None
    best_horizontal = np.zeros((rows, cols), dtype=np.float64)
    best_vertical = np.zeros((rows, cols), dtype=np.float64)
    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            shifted = _shift(current, row_offset, column_offset)
            cost = _block_mean(np.abs(previous - shifted), rows, cols)
            if best_cost is None:
                best_cost = cost
                best_horizontal = np.full((rows, cols), float(column_offset))
                best_vertical = np.full((rows, cols), float(row_offset))
                continue
            better = cost < best_cost
            best_horizontal = np.where(better, float(column_offset), best_horizontal)
            best_vertical = np.where(better, float(row_offset), best_vertical)
            best_cost = np.minimum(cost, best_cost)
    return np.stack([best_horizontal, best_vertical], axis=-1)


def _block_flow(previous: np.ndarray, current: np.ndarray, parameters: StageAParameters, estimator: str) -> np.ndarray:
    if estimator == FLOW_FARNEBACK:
        return _farneback_block_flow(previous, current, parameters)
    if estimator == FLOW_BLOCK_MATCHING:
        return _block_matching_flow(previous, current, parameters)
    raise StageAConfigurationError("unknown resolved flow estimator %r" % estimator)


def _interior_mask(parameters: StageAParameters) -> np.ndarray:
    mask = np.zeros((parameters.grid_rows, parameters.grid_cols), dtype=bool)
    border = parameters.scene_reference_border_blocks
    mask[border : parameters.grid_rows - border, border : parameters.grid_cols - border] = True
    return mask


# ---------------------------------------------------------------------------
# Per-transition measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TransitionMetrics:
    index: int
    frame_delta_mae: float
    scene_reference_magnitude: float
    active_magnitude: float
    relative_motion_ratio: float
    observed_vector: Tuple[float, float]
    observed_magnitude: float
    texture_energy: float
    commanded_translation: float
    commanded_rotation: float
    commanded_magnitude: float
    expected_static: bool
    direction_cosine: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "frame_delta_mae_0_255": _round(self.frame_delta_mae),
            "scene_reference_flow_magnitude": _round(self.scene_reference_magnitude),
            "active_interior_flow_magnitude": _round(self.active_magnitude),
            "relative_motion_ratio": _round(self.relative_motion_ratio),
            "observed_flow_vector": [_round(self.observed_vector[0]), _round(self.observed_vector[1])],
            "observed_flow_magnitude": _round(self.observed_magnitude),
            "texture_energy": _round(self.texture_energy),
            "commanded_translation_l2": _round(self.commanded_translation),
            "commanded_rotation_l2": _round(self.commanded_rotation),
            "commanded_magnitude": _round(self.commanded_magnitude),
            "expected_static": bool(self.expected_static),
            "direction_cosine": _round(self.direction_cosine),
        }


def _commanded_components(action: Sequence[float], parameters: StageAParameters) -> Tuple[float, float, float]:
    translation = float(math.sqrt(sum(float(value) ** 2 for value in action[0:3])))
    rotation = float(math.sqrt(sum(float(value) ** 2 for value in action[3:6])))
    magnitude = float(math.sqrt(translation ** 2 + (parameters.rotation_lever_arm_metres * rotation) ** 2))
    return translation, rotation, magnitude


def _measure_transition(
    index: int,
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    action: Sequence[float],
    parameters: StageAParameters,
    estimator: str,
    interior: np.ndarray,
    camera_axis_map: Optional[np.ndarray],
    expected_static_override: Optional[bool],
) -> _TransitionMetrics:
    previous_gray = _to_gray(previous_frame)
    current_gray = _to_gray(current_frame)
    frame_delta_mae = float(np.abs(current_gray - previous_gray).mean())
    reduced_previous = _downsample(previous_gray, parameters.flow_downsample_factor)
    reduced_current = _downsample(current_gray, parameters.flow_downsample_factor)
    flow = _block_flow(reduced_previous, reduced_current, parameters, estimator)
    magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    border_values = magnitudes[~interior]
    scene_reference = float(np.median(border_values)) if border_values.size else 0.0
    interior_values = magnitudes[interior]
    active = float(np.percentile(interior_values, parameters.active_block_percentile, method="linear"))
    ratio = active / max(scene_reference, parameters.scene_reference_floor)

    selected = interior & (magnitudes >= active)
    if not bool(selected.any()):
        selected = interior
    observed = (
        float(flow[..., 0][selected].mean()),
        float(flow[..., 1][selected].mean()),
    )
    observed_magnitude = float(math.sqrt(observed[0] ** 2 + observed[1] ** 2))

    translation, rotation, magnitude = _commanded_components(action, parameters)
    expected_static = (
        bool(expected_static_override)
        if expected_static_override is not None
        else magnitude <= parameters.static_command_magnitude
    )

    direction_cosine: Optional[float] = None
    if camera_axis_map is not None and translation > 0.0 and observed_magnitude >= parameters.min_direction_flow_magnitude:
        commanded_vector = camera_axis_map @ np.asarray([float(value) for value in action[0:3]], dtype=np.float64)
        commanded_norm = float(np.linalg.norm(commanded_vector))
        if commanded_norm > 0.0:
            direction_cosine = float(
                (observed[0] * commanded_vector[0] + observed[1] * commanded_vector[1])
                / (observed_magnitude * commanded_norm)
            )

    return _TransitionMetrics(
        index=index,
        frame_delta_mae=frame_delta_mae,
        scene_reference_magnitude=scene_reference,
        active_magnitude=active,
        relative_motion_ratio=ratio,
        observed_vector=observed,
        observed_magnitude=observed_magnitude,
        texture_energy=_texture_energy(reduced_previous),
        commanded_translation=translation,
        commanded_rotation=rotation,
        commanded_magnitude=magnitude,
        expected_static=expected_static,
        direction_cosine=direction_cosine,
    )


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        key: float(np.percentile(array, quantile, method="linear")) for key, quantile in _PERCENTILE_KEYS
    }


# ---------------------------------------------------------------------------
# Motion reference, fitted from real trajectories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceTrajectory:
    """One provenance-backed trajectory used to fit a motion reference.

    ``source_kind`` is caller-declared metadata.  This module cannot verify
    that pixels came from a real robot, so it records the declaration and
    refuses to use a synthetic-sourced reference for primary scoring.
    """

    trajectory_id: str
    source_uri: str
    source_sha256: str
    frames: Tuple[Any, ...]
    actions: Tuple[Tuple[float, ...], ...]
    source_kind: str = SOURCE_REAL
    source_dataset: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.trajectory_id or not isinstance(self.trajectory_id, str):
            raise StageAConfigurationError("reference trajectory needs a nonempty trajectory_id")
        if not self.source_uri or not isinstance(self.source_uri, str):
            raise StageAConfigurationError("reference trajectory %r needs a source_uri" % self.trajectory_id)
        if not _is_hex_sha256(self.source_sha256):
            raise StageAConfigurationError(
                "reference trajectory %r needs a SHA-256 source digest" % self.trajectory_id
            )
        if self.source_kind not in SOURCE_KINDS:
            raise StageAConfigurationError("source_kind must be one of %s" % (SOURCE_KINDS,))
        if len(self.frames) != len(self.actions) + 1:
            raise StageAConfigurationError(
                "reference trajectory %r must have action_count + 1 frames" % self.trajectory_id
            )
        if not self.actions:
            raise StageAConfigurationError("reference trajectory %r has no actions" % self.trajectory_id)

    def provenance(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "source_uri": self.source_uri,
            "source_sha256": str(self.source_sha256),
            "source_kind": self.source_kind,
            "source_dataset": self.source_dataset,
            "action_count": len(self.actions),
        }


@dataclass(frozen=True)
class MotionReference:
    """Command-motion distribution fitted from real Bridge trajectories.

    The gate compares relative motion against these percentiles instead of
    equating low optical flow with invalidity.  Every field is a recorded
    measurement over the declared sources; nothing here is a default.
    """

    reference_id: str
    flow_estimator: str
    measurement_hash: str
    trajectory_count: int
    transition_count: int
    moving_transition_count: int
    static_transition_count: int
    frame_delta_mae_percentiles: Mapping[str, float]
    relative_motion_ratio_percentiles: Mapping[str, float]
    static_relative_motion_ratio_percentiles: Mapping[str, float]
    direction_cosine_percentiles: Mapping[str, float]
    texture_gradient_percentiles: Mapping[str, float]
    provenance: Tuple[Mapping[str, Any], ...]
    provenance_hash: str
    source_description: str
    camera_axis_map: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    camera_axis_map_residual_rms: Optional[float] = None
    camera_axis_map_transitions: int = 0
    contains_synthetic_sources: bool = False
    fitted_parameter_set_id: str = ""

    def __post_init__(self) -> None:
        if not self.reference_id:
            raise StageAConfigurationError("motion reference needs a reference_id")
        if self.flow_estimator not in FLOW_ESTIMATORS:
            raise StageAConfigurationError("motion reference flow_estimator must be a resolved estimator name")
        if self.trajectory_count < 1 or self.transition_count < 1:
            raise StageAConfigurationError("motion reference must record at least one trajectory and transition")
        if not self.provenance:
            raise StageAConfigurationError("motion reference must record per-trajectory provenance")
        if not self.frame_delta_mae_percentiles:
            raise StageAConfigurationError("motion reference must record frame-delta percentiles")

    @property
    def reference_hash(self) -> str:
        return _canonical_hash(self.as_dict())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "flow_estimator": self.flow_estimator,
            "measurement_hash": self.measurement_hash,
            "fitted_parameter_set_id": self.fitted_parameter_set_id,
            "trajectory_count": int(self.trajectory_count),
            "transition_count": int(self.transition_count),
            "moving_transition_count": int(self.moving_transition_count),
            "static_transition_count": int(self.static_transition_count),
            "frame_delta_mae_percentiles": {key: _round(value) for key, value in sorted(self.frame_delta_mae_percentiles.items())},
            "relative_motion_ratio_percentiles": {
                key: _round(value) for key, value in sorted(self.relative_motion_ratio_percentiles.items())
            },
            "static_relative_motion_ratio_percentiles": {
                key: _round(value) for key, value in sorted(self.static_relative_motion_ratio_percentiles.items())
            },
            "direction_cosine_percentiles": {
                key: _round(value) for key, value in sorted(self.direction_cosine_percentiles.items())
            },
            "texture_gradient_percentiles": {
                key: _round(value) for key, value in sorted(self.texture_gradient_percentiles.items())
            },
            "camera_axis_map": (
                None
                if self.camera_axis_map is None
                else [[_round(value) for value in row] for row in self.camera_axis_map]
            ),
            "camera_axis_map_residual_rms": _round(self.camera_axis_map_residual_rms),
            "camera_axis_map_transitions": int(self.camera_axis_map_transitions),
            "contains_synthetic_sources": bool(self.contains_synthetic_sources),
            "provenance": [dict(sorted(item.items())) for item in self.provenance],
            "provenance_hash": self.provenance_hash,
            "source_description": self.source_description,
        }

    @classmethod
    def fit_from_trajectories(
        cls,
        trajectories: Sequence[ReferenceTrajectory],
        *,
        reference_id: str,
        source_description: str,
        parameters: Optional[StageAParameters] = None,
    ) -> "MotionReference":
        """Fit the reference distribution from decoded real trajectories.

        Raises instead of returning a partially fitted reference: an
        unfittable reference must block calibrated scoring, not silently widen
        its own bounds.
        """

        parameters = parameters or StageAParameters()
        estimator = resolve_flow_estimator(parameters.flow_estimator)
        if not trajectories:
            raise StageAConfigurationError("cannot fit a motion reference from zero trajectories")
        interior = _interior_mask(parameters)

        frame_deltas: List[float] = []
        moving_ratios: List[float] = []
        static_ratios: List[float] = []
        cosines: List[float] = []
        textures: List[float] = []
        design: List[List[float]] = []
        target: List[List[float]] = []
        transition_count = 0
        for trajectory in trajectories:
            decoded: List[np.ndarray] = []
            for position, frame in enumerate(trajectory.frames):
                array, failure, detail = _decode_frame(frame, parameters)
                if array is None:
                    raise StageAConfigurationError(
                        "reference trajectory %r frame %d does not decode (%s: %s)"
                        % (trajectory.trajectory_id, position, failure.value if failure else "unknown", detail)
                    )
                decoded.append(array)
            for index, action in enumerate(trajectory.actions):
                if len(action) < 6 or any(not math.isfinite(float(value)) for value in action):
                    raise StageAConfigurationError(
                        "reference trajectory %r action %d is not a finite 7-D Bridge action"
                        % (trajectory.trajectory_id, index)
                    )
                metrics = _measure_transition(
                    index,
                    decoded[index],
                    decoded[index + 1],
                    action,
                    parameters,
                    estimator,
                    interior,
                    None,
                    None,
                )
                transition_count += 1
                frame_deltas.append(metrics.frame_delta_mae)
                textures.append(metrics.texture_energy)
                if metrics.expected_static:
                    static_ratios.append(metrics.relative_motion_ratio)
                    continue
                moving_ratios.append(metrics.relative_motion_ratio)
                if metrics.observed_magnitude >= parameters.min_direction_flow_magnitude and metrics.commanded_translation > 0:
                    design.append([float(value) for value in action[0:3]])
                    target.append([metrics.observed_vector[0], metrics.observed_vector[1]])

        if transition_count < parameters.min_reference_transitions:
            raise StageAConfigurationError(
                "motion reference needs at least %d transitions, got %d"
                % (parameters.min_reference_transitions, transition_count)
            )
        if not moving_ratios:
            raise StageAConfigurationError(
                "motion reference has no non-static transitions; it cannot calibrate commanded motion"
            )

        axis_map: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
        residual_rms: Optional[float] = None
        if len(design) >= parameters.min_axis_map_transitions:
            matrix = np.asarray(design, dtype=np.float64)
            observations = np.asarray(target, dtype=np.float64)
            singular = np.linalg.svd(matrix, compute_uv=False)
            if float(singular.min()) >= parameters.axis_map_singular_value_floor:
                solution, _residuals, _rank, _sv = np.linalg.lstsq(matrix, observations, rcond=None)
                fitted = matrix @ solution
                residual_rms = float(np.sqrt(np.mean((observations - fitted) ** 2)))
                transposed = solution.T
                axis_map = (
                    (float(transposed[0][0]), float(transposed[0][1]), float(transposed[0][2])),
                    (float(transposed[1][0]), float(transposed[1][1]), float(transposed[1][2])),
                )
                # The direction-cosine distribution can only be measured once
                # the camera axis map exists, so it is computed from the stored
                # observed/commanded pairs rather than by re-running flow.
                for commanded, observed in zip(matrix, observations):
                    predicted = np.asarray(axis_map, dtype=np.float64) @ commanded
                    predicted_norm = float(np.linalg.norm(predicted))
                    observed_norm = float(np.linalg.norm(observed))
                    if predicted_norm > 0.0 and observed_norm > 0.0:
                        cosines.append(float(np.dot(observed, predicted) / (observed_norm * predicted_norm)))

        provenance = tuple(trajectory.provenance() for trajectory in trajectories)
        provenance_hash = _canonical_hash(
            {
                "reference_id": reference_id,
                "flow_estimator": estimator,
                "measurement_hash": parameters.measurement_hash,
                "trajectories": [dict(sorted(item.items())) for item in provenance],
            }
        )
        return cls(
            reference_id=reference_id,
            flow_estimator=estimator,
            measurement_hash=parameters.measurement_hash,
            trajectory_count=len(trajectories),
            transition_count=transition_count,
            moving_transition_count=len(moving_ratios),
            static_transition_count=len(static_ratios),
            frame_delta_mae_percentiles=_percentiles(frame_deltas),
            relative_motion_ratio_percentiles=_percentiles(moving_ratios),
            static_relative_motion_ratio_percentiles=_percentiles(static_ratios),
            direction_cosine_percentiles=_percentiles(cosines),
            texture_gradient_percentiles=_percentiles(textures),
            provenance=provenance,
            provenance_hash=provenance_hash,
            source_description=source_description,
            camera_axis_map=axis_map,
            camera_axis_map_residual_rms=residual_rms,
            camera_axis_map_transitions=len(design),
            contains_synthetic_sources=any(
                trajectory.source_kind != SOURCE_REAL for trajectory in trajectories
            ),
            fitted_parameter_set_id=parameters.parameter_set_id,
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidityReport:
    """Stage A outcome. ``unknown`` means unknown: never ``valid``, never zero."""

    validity: Literal["valid", "invalid", "unknown"]
    reason_codes: Tuple[str, ...]
    calibration_class: str
    primary_scoring_eligible: bool
    parameters_hash: str
    calibration_reference_hash: Optional[str]
    parameter_set_id: str
    flow_estimator: str
    expected_static: Optional[bool]
    expected_static_source: str
    checks: Mapping[str, Any] = field(default_factory=dict)
    gate_version: str = STAGE_A_GATE_VERSION

    def __post_init__(self) -> None:
        if self.validity not in (VALIDITY_VALID, VALIDITY_INVALID, VALIDITY_UNKNOWN):
            raise StageAConfigurationError("validity must be valid, invalid, or unknown")
        if self.calibration_class not in CALIBRATION_CLASSES:
            raise StageAConfigurationError("calibration_class must be one of %s" % (CALIBRATION_CLASSES,))
        unknown_codes = [code for code in self.reason_codes if code not in ALL_REASON_CODES]
        if unknown_codes:
            raise StageAConfigurationError("reason codes are a closed enum; rejected %s" % unknown_codes)
        if self.validity == VALIDITY_VALID:
            offending = [code for code in self.reason_codes if code not in
                         {item.value for item in INFORMATIONAL_REASON_CODES}]
            if offending:
                raise StageAConfigurationError("a valid verdict cannot carry verdict reason codes %s" % offending)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "validity": self.validity,
            "reason_codes": list(self.reason_codes),
            "calibration_class": self.calibration_class,
            "primary_scoring_eligible": bool(self.primary_scoring_eligible),
            "parameters_hash": self.parameters_hash,
            "calibration_reference_hash": self.calibration_reference_hash,
            "parameter_set_id": self.parameter_set_id,
            "flow_estimator": self.flow_estimator,
            "expected_static": self.expected_static,
            "expected_static_source": self.expected_static_source,
            "gate_version": self.gate_version,
            "checks": dict(self.checks),
        }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class StageAValidityGate:
    """Deterministic Stage A validity gate. Pure Python; no model is called.

    In ``calibrated_primary`` mode a :class:`MotionReference` fitted from real
    trajectories is mandatory.  Without one, :meth:`evaluate` returns
    ``unknown`` with ``no_motion_reference`` rather than guessing a bound.

    ``uncalibrated_development`` mode exists for development fixtures only.
    Its reports carry ``calibration_class="uncalibrated_development"``,
    ``primary_scoring_eligible=False`` and the
    ``uncalibrated_development_mode`` reason code, so a development verdict can
    never be mistaken for primary-scoring evidence.
    """

    def __init__(
        self,
        parameters: Optional[StageAParameters] = None,
        *,
        motion_reference: Optional[MotionReference] = None,
        calibration_class: str = CALIBRATION_PRIMARY,
    ) -> None:
        if calibration_class not in CALIBRATION_CLASSES:
            raise StageAConfigurationError("calibration_class must be one of %s" % (CALIBRATION_CLASSES,))
        self.parameters = parameters or StageAParameters()
        self.motion_reference = motion_reference
        self.calibration_class = calibration_class
        self.flow_estimator = resolve_flow_estimator(self.parameters.flow_estimator)
        self._interior = _interior_mask(self.parameters)

    # -- identity ---------------------------------------------------------

    @property
    def parameters_hash(self) -> str:
        """Deployable parameter pin: declared parameters plus resolved estimator."""

        return _canonical_hash(
            {
                "gate_version": STAGE_A_GATE_VERSION,
                "declared_parameters": self.parameters.as_dict(),
                "resolved_flow_estimator": self.flow_estimator,
            }
        )

    @property
    def calibration_reference_hash(self) -> Optional[str]:
        """``None`` means no calibrated reference is loaded, never zero."""

        return None if self.motion_reference is None else self.motion_reference.reference_hash

    @property
    def primary_scoring_eligible(self) -> bool:
        return self.calibration_class == CALIBRATION_PRIMARY and not self._reference_blockers()

    def describe(self) -> Dict[str, Any]:
        """Deployment contract values, e.g. for the Baseten ValidityWorker."""

        return {
            "gate_version": STAGE_A_GATE_VERSION,
            "parameter_set_id": self.parameters.parameter_set_id,
            "parameters_hash": self.parameters_hash,
            "declared_parameters_hash": self.parameters.declared_hash,
            "measurement_hash": self.parameters.measurement_hash,
            "calibration_reference_hash": self.calibration_reference_hash,
            "calibration_class": self.calibration_class,
            "flow_estimator": self.flow_estimator,
            "primary_scoring_eligible": self.primary_scoring_eligible,
            "reason_codes": list(ALL_REASON_CODES),
            "uses_learned_features": False,
            "vjepa_hook": None,
            "declared_flow_estimator": self.parameters.flow_estimator,
            "deployment_note": (
                "parameters_hash includes the resolved flow estimator, so a deployment must pin flow_estimator "
                "explicitly rather than leaving it 'auto': the same declared parameters resolve to "
                "farneback_cv2_fixed_v1 where OpenCV is installed and numpy_block_matching_v1 where it is not, "
                "which are different measurements and different hashes."
                if self.parameters.flow_estimator == "auto"
                else "flow_estimator is pinned explicitly, so parameters_hash is reproducible across images."
            ),
        }

    # -- reference gating --------------------------------------------------

    def _reference_blockers(self) -> Tuple[ValidityReasonCode, ...]:
        if self.calibration_class != CALIBRATION_PRIMARY:
            return ()
        reference = self.motion_reference
        if reference is None:
            return (ValidityReasonCode.NO_MOTION_REFERENCE,)
        blockers: List[ValidityReasonCode] = []
        if reference.flow_estimator != self.flow_estimator:
            blockers.append(ValidityReasonCode.MOTION_REFERENCE_ESTIMATOR_MISMATCH)
        if reference.measurement_hash != self.parameters.measurement_hash:
            blockers.append(ValidityReasonCode.MOTION_REFERENCE_PARAMETERS_MISMATCH)
        if reference.contains_synthetic_sources:
            blockers.append(ValidityReasonCode.MOTION_REFERENCE_NOT_REAL_DATA)
        return tuple(blockers)

    def _bounds(self) -> Dict[str, Any]:
        """Resolve the decision bounds actually used, with their source."""

        parameters = self.parameters
        reference = self.motion_reference
        if self.calibration_class == CALIBRATION_PRIMARY and reference is not None:
            moving = reference.relative_motion_ratio_percentiles
            static = reference.static_relative_motion_ratio_percentiles
            cosines = reference.direction_cosine_percentiles
            texture = reference.texture_gradient_percentiles
            return {
                "source": "motion_reference:" + reference.reference_id,
                "direction_reference_available": reference.camera_axis_map is not None,
                "frame_delta_mae_bound": float(
                    reference.frame_delta_mae_percentiles.get("p99", 0.0) * parameters.discontinuity_bound_multiplier
                ),
                "frame_delta_mae_bound_source": "reference_p99*%.3f" % parameters.discontinuity_bound_multiplier,
                "min_relative_motion": float(moving.get("p01", 0.0)),
                "ambiguous_relative_motion": float(moving.get("p05", 0.0)),
                "static_max_relative_motion": (
                    float(static["p99"] * parameters.static_bound_multiplier)
                    if static
                    else float(parameters.static_max_relative_motion)
                ),
                "static_bound_source": "reference_static_p99" if static else "declared_parameter",
                "direction_cosine_min": (
                    float(cosines["p01"]) if cosines else None
                ),
                "direction_cosine_ambiguous": (
                    float(cosines["p05"]) if cosines else None
                ),
                "min_texture_gradient": float(texture.get("p01", 0.0)) if texture else None,
            }
        return {
            "source": "uncalibrated_development_parameters",
            "direction_reference_available": reference is not None and reference.camera_axis_map is not None,
            "frame_delta_mae_bound": float(parameters.uncalibrated_frame_delta_mae_bound),
            "frame_delta_mae_bound_source": "declared_parameter",
            "min_relative_motion": float(parameters.uncalibrated_min_relative_motion),
            "ambiguous_relative_motion": float(parameters.uncalibrated_ambiguous_relative_motion),
            "static_max_relative_motion": float(parameters.static_max_relative_motion),
            "static_bound_source": "declared_parameter",
            "direction_cosine_min": float(parameters.uncalibrated_direction_cosine_min),
            "direction_cosine_ambiguous": float(parameters.uncalibrated_direction_cosine_ambiguous),
            "min_texture_gradient": float(parameters.uncalibrated_min_texture_gradient),
        }

    # -- evaluation -------------------------------------------------------

    def evaluate(
        self,
        *,
        frames: Sequence[Any],
        actions: Sequence[Sequence[float]],
        nominal_timestamps: Sequence[float],
        conditioning_frame: Any = None,
        states: Optional[Sequence[Sequence[float]]] = None,
        expected_static: Optional[bool] = None,
        episode_id: Optional[str] = None,
    ) -> ValidityReport:
        """Run every Stage A check and return one explicit :class:`ValidityReport`.

        ``frames`` must already be decoded frame objects (arrays or array-like);
        ``actions`` are the physical 7-D Bridge command rows; ``states`` are the
        per-tick state vectors.  Omitting ``states`` or ``conditioning_frame``
        leaves a required check unperformed, which produces ``unknown`` rather
        than an optimistic ``valid``.
        """

        parameters = self.parameters
        codes: List[ValidityReasonCode] = []
        checks: Dict[str, Any] = {
            "episode_id": episode_id,
            "gate_version": STAGE_A_GATE_VERSION,
            "flow_estimator": self.flow_estimator,
            "calibration_class": self.calibration_class,
        }
        if self.calibration_class == CALIBRATION_DEVELOPMENT:
            codes.append(ValidityReasonCode.UNCALIBRATED_DEVELOPMENT_MODE)

        reference_blockers = self._reference_blockers()
        codes.extend(reference_blockers)
        checks["motion_reference"] = {
            "reference_id": None if self.motion_reference is None else self.motion_reference.reference_id,
            "reference_hash": self.calibration_reference_hash,
            "blockers": [code.value for code in reference_blockers],
        }

        # 1. Frame count, conditioning frame position, and nominal order.
        expected_frames = len(actions) + 1
        frame_count_ok = len(frames) == expected_frames
        if not frame_count_ok:
            codes.append(ValidityReasonCode.FRAME_COUNT_MISMATCH)
        checks["frame_count"] = {
            "observed": len(frames),
            "expected": expected_frames,
            "rule": "action_count + 1 frames, conditioning frame at index 0",
            "ok": bool(frame_count_ok),
        }

        timestamp_ok = len(nominal_timestamps) == len(frames)
        if not timestamp_ok:
            codes.append(ValidityReasonCode.TIMESTAMP_COUNT_MISMATCH)
        monotone = True
        previous: Optional[float] = None
        for value in nominal_timestamps:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                monotone = False
                break
            if not math.isfinite(numeric) or (previous is not None and numeric <= previous):
                monotone = False
                break
            previous = numeric
        if not monotone:
            codes.append(ValidityReasonCode.NONMONOTONE_TIMESTAMPS)
        checks["timestamps"] = {
            "count_ok": bool(timestamp_ok),
            "strictly_increasing": bool(monotone),
            "first": _round(float(nominal_timestamps[0])) if nominal_timestamps and monotone else None,
            "last": _round(float(nominal_timestamps[-1])) if nominal_timestamps and monotone else None,
            "semantics": "wrapper-assigned nominal control timestamps, not measured physical timing",
        }

        # 2. Decode integrity for every frame.
        decoded: List[Optional[np.ndarray]] = []
        decode_details: List[Dict[str, Any]] = []
        decode_codes: List[ValidityReasonCode] = []
        constant_frames: List[int] = []
        for position, frame in enumerate(frames):
            array, failure, detail = _decode_frame(frame, parameters)
            detail = dict(detail)
            detail["frame_index"] = position
            if array is None and failure is not None:
                detail["failure"] = failure.value
                decode_codes.append(failure)
                decoded.append(None)
                decode_details.append(detail)
                continue
            assert array is not None
            values = array.astype(np.float64)
            standard_deviation = float(values.std())
            maximum = float(values.max())
            detail["pixel_std"] = _round(standard_deviation)
            detail["pixel_max"] = _round(maximum)
            if standard_deviation < parameters.min_frame_pixel_std or maximum < parameters.min_frame_max_value:
                detail["failure"] = ValidityReasonCode.CONSTANT_FRAME.value
                detail["constant_kind"] = "all_black" if maximum < parameters.min_frame_max_value else "all_constant"
                constant_frames.append(position)
                decode_codes.append(ValidityReasonCode.CONSTANT_FRAME)
                decoded.append(None)
                decode_details.append(detail)
                continue
            decoded.append(array)
            decode_details.append(detail)
        codes.extend(decode_codes)
        checks["decode"] = {
            "frames_submitted": len(frames),
            "frames_decoded": sum(1 for item in decoded if item is not None),
            "expected_shape": [parameters.frame_height, parameters.frame_width, parameters.frame_channels],
            "expected_dtype": "uint8" + (" or float in [0,1]" if parameters.accept_float01_frames else ""),
            "observed_dtypes": sorted({str(item.get("dtype")) for item in decode_details if item.get("dtype")}),
            "observed_layouts": sorted({str(item.get("layout")) for item in decode_details if item.get("layout")}),
            "constant_frames": constant_frames,
            # Only failing frames keep their full detail; a per-frame record for
            # every tick would dominate the episode artifact without adding
            # evidence about what went wrong.
            "failures": [item for item in decode_details if "failure" in item],
        }

        # 3. Conditioning frame must be frame 0.
        conditioning = {"supplied": conditioning_frame is not None, "verified": False, "method": None}
        if conditioning_frame is None:
            codes.append(ValidityReasonCode.CONDITIONING_FRAME_UNVERIFIED)
            conditioning["method"] = "not_supplied"
        else:
            reference_array, failure, _detail = _decode_frame(conditioning_frame, parameters)
            first = decoded[0] if decoded else None
            conditioning["method"] = "decoded_pixel_equality"
            if reference_array is None:
                codes.append(ValidityReasonCode.CONDITIONING_FRAME_UNVERIFIED)
                conditioning["method"] = "conditioning_frame_undecodable"
                conditioning["failure"] = failure.value if failure else None
            elif first is None:
                conditioning["method"] = "frame_0_undecodable"
            elif bool(np.array_equal(reference_array, first)):
                conditioning["verified"] = True
            else:
                codes.append(ValidityReasonCode.CONDITIONING_FRAME_MISMATCH)
        checks["conditioning_frame"] = conditioning

        # 4. Finite action and state values.
        action_width_ok = True
        nonfinite_actions: List[int] = []
        for index, action in enumerate(actions):
            row = list(action)
            if len(row) != 7:
                action_width_ok = False
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in row):
                nonfinite_actions.append(index)
        if not action_width_ok:
            codes.append(ValidityReasonCode.ACTION_WIDTH_MISMATCH)
        if nonfinite_actions:
            codes.append(ValidityReasonCode.NONFINITE_ACTION)
        nonfinite_states: List[int] = []
        state_count_ok = True
        if states is None:
            codes.append(ValidityReasonCode.STATE_VALUES_NOT_SUPPLIED)
        else:
            for index, state in enumerate(states):
                if any(
                    not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))
                    for value in state
                ):
                    nonfinite_states.append(index)
            if nonfinite_states:
                codes.append(ValidityReasonCode.NONFINITE_STATE)
            # One state per control tick plus the start state, mirroring the
            # frame rule. A mismatch means the caller's own records disagree.
            state_count_ok = len(states) == len(actions) + 1
            if not state_count_ok:
                codes.append(ValidityReasonCode.STATE_COUNT_MISMATCH)
        checks["finite_values"] = {
            "action_rows": len(actions),
            "action_width_expected": 7,
            "action_width_ok": bool(action_width_ok),
            "nonfinite_action_rows": nonfinite_actions,
            "state_rows": None if states is None else len(states),
            "state_rows_expected": len(actions) + 1,
            "state_row_count_ok": bool(state_count_ok),
            "nonfinite_state_rows": nonfinite_states,
            "states_supplied": states is not None,
        }

        structural_failure = any(code in INVALID_REASON_CODES for code in codes)
        bounds = self._bounds()
        checks["bounds"] = {key: (_round(value) if isinstance(value, float) else value) for key, value in bounds.items()}

        if structural_failure or any(item is None for item in decoded) or not frame_count_ok:
            checks["motion_consistency"] = {
                "status": "not_run",
                "reason": "structural or decode checks failed; motion metrics would not be interpretable",
            }
            checks["discontinuity"] = {"status": "not_run"}
            return self._report(codes, checks, expected_static, "not_evaluated")

        # 5. Impossible discontinuities and calibrated command-motion consistency.
        expected_static_source = "caller_hint" if expected_static is not None else "derived_from_commanded_magnitude"
        axis_map: Optional[np.ndarray] = None
        if self.motion_reference is not None and self.motion_reference.camera_axis_map is not None:
            axis_map = np.asarray(self.motion_reference.camera_axis_map, dtype=np.float64)
        metrics: List[_TransitionMetrics] = []
        for index, action in enumerate(actions):
            previous_frame = decoded[index]
            current_frame = decoded[index + 1]
            assert previous_frame is not None and current_frame is not None
            metrics.append(
                _measure_transition(
                    index,
                    previous_frame,
                    current_frame,
                    action,
                    parameters,
                    self.flow_estimator,
                    self._interior,
                    axis_map,
                    expected_static,
                )
            )

        discontinuity_bound = float(bounds["frame_delta_mae_bound"])
        violating = [item.index for item in metrics if item.frame_delta_mae > discontinuity_bound]
        if violating:
            codes.append(ValidityReasonCode.IMPOSSIBLE_DISCONTINUITY)
        checks["discontinuity"] = {
            "status": "run",
            "bound_0_255": _round(discontinuity_bound),
            "bound_source": bounds["frame_delta_mae_bound_source"],
            "max_observed": _round(max((item.frame_delta_mae for item in metrics), default=0.0)),
            "violating_transitions": violating,
        }

        motion = self._motion_verdict(metrics, bounds, reference_blockers)
        codes.extend(motion["codes"])
        checks["motion_consistency"] = motion["detail"]
        checks["transitions"] = [item.as_dict() for item in metrics]

        episode_static: Optional[bool]
        if expected_static is not None:
            episode_static = bool(expected_static)
        elif metrics:
            episode_static = all(item.expected_static for item in metrics)
        else:
            episode_static = None
        return self._report(codes, checks, episode_static, expected_static_source)

    def _motion_verdict(
        self,
        metrics: Sequence[_TransitionMetrics],
        bounds: Mapping[str, Any],
        reference_blockers: Sequence[ValidityReasonCode],
    ) -> Dict[str, Any]:
        """Classify each transition, then apply the fraction thresholds.

        Nothing here thresholds a global flow magnitude.  Every magnitude
        decision uses the interior/reference-band ratio against the calibrated
        distribution, and a legitimately static transition is never failed for
        missing motion.
        """

        parameters = self.parameters
        inconsistent: Dict[str, List[int]] = {}
        ambiguous: Dict[str, List[int]] = {}
        consistent: List[int] = []
        static_transitions: List[int] = []
        min_texture = bounds.get("min_texture_gradient")
        cosine_min = bounds.get("direction_cosine_min")
        cosine_ambiguous = bounds.get("direction_cosine_ambiguous")

        def add(bucket: Dict[str, List[int]], code: ValidityReasonCode, index: int) -> None:
            bucket.setdefault(code.value, []).append(index)

        for item in metrics:
            if min_texture is not None and item.texture_energy < float(min_texture):
                add(ambiguous, ValidityReasonCode.AMBIGUOUS_LOW_TEXTURE, item.index)
                continue
            if item.expected_static:
                static_transitions.append(item.index)
                if item.relative_motion_ratio > float(bounds["static_max_relative_motion"]):
                    add(inconsistent, ValidityReasonCode.COMMAND_MOTION_INCONSISTENT, item.index)
                else:
                    consistent.append(item.index)
                continue
            if item.relative_motion_ratio < float(bounds["min_relative_motion"]):
                add(inconsistent, ValidityReasonCode.MOTION_BELOW_REFERENCE, item.index)
                continue
            if item.relative_motion_ratio < float(bounds["ambiguous_relative_motion"]):
                add(ambiguous, ValidityReasonCode.AMBIGUOUS_MOTION_MAGNITUDE, item.index)
                continue
            if not bounds.get("direction_reference_available") or cosine_min is None or cosine_ambiguous is None:
                # No camera/scene direction reference exists, so direction is
                # unmeasured rather than satisfied.
                add(ambiguous, ValidityReasonCode.NO_DIRECTION_REFERENCE, item.index)
                continue
            if item.direction_cosine is None:
                add(ambiguous, ValidityReasonCode.AMBIGUOUS_MOTION_DIRECTION, item.index)
                continue
            if item.direction_cosine < float(cosine_min):
                add(inconsistent, ValidityReasonCode.COMMAND_MOTION_INCONSISTENT, item.index)
                continue
            if item.direction_cosine < float(cosine_ambiguous):
                add(ambiguous, ValidityReasonCode.AMBIGUOUS_MOTION_DIRECTION, item.index)
                continue
            consistent.append(item.index)

        total = len(metrics)
        inconsistent_count = sum(len(values) for values in inconsistent.values())
        ambiguous_count = sum(len(values) for values in ambiguous.values())
        inconsistent_fraction = inconsistent_count / total if total else 0.0
        ambiguous_fraction = ambiguous_count / total if total else 0.0

        codes: List[ValidityReasonCode] = []
        if total == 0:
            codes.append(ValidityReasonCode.NO_TRANSITIONS)
            verdict = "no_transitions"
        elif inconsistent_fraction > parameters.max_inconsistent_transition_fraction:
            codes.extend(ValidityReasonCode(value) for value in sorted(inconsistent))
            verdict = "inconsistent"
        elif ambiguous_fraction > parameters.max_ambiguous_transition_fraction:
            codes.extend(ValidityReasonCode(value) for value in sorted(ambiguous))
            verdict = "ambiguous"
        elif reference_blockers:
            verdict = "blocked_by_missing_reference"
        else:
            verdict = "consistent"

        return {
            "codes": codes,
            "detail": {
                "status": "run",
                "verdict": verdict,
                "criterion": (
                    "interior/reference-band relative motion and direction against a calibrated distribution; "
                    "global optical-flow magnitude is never thresholded on its own"
                ),
                "transition_count": total,
                "consistent_transitions": consistent,
                "static_transitions": static_transitions,
                "static_transitions_are_admissible": True,
                "inconsistent_transitions": {key: values for key, values in sorted(inconsistent.items())},
                "ambiguous_transitions": {key: values for key, values in sorted(ambiguous.items())},
                "inconsistent_fraction": _round(inconsistent_fraction),
                "ambiguous_fraction": _round(ambiguous_fraction),
                "max_inconsistent_transition_fraction": parameters.max_inconsistent_transition_fraction,
                "max_ambiguous_transition_fraction": parameters.max_ambiguous_transition_fraction,
            },
        }

    def _report(
        self,
        codes: Sequence[ValidityReasonCode],
        checks: Mapping[str, Any],
        expected_static: Optional[bool],
        expected_static_source: str,
    ) -> ValidityReport:
        unique = sorted({code.value for code in codes})
        if any(ValidityReasonCode(code) in INVALID_REASON_CODES for code in unique):
            validity: Literal["valid", "invalid", "unknown"] = VALIDITY_INVALID
        elif any(ValidityReasonCode(code) in UNKNOWN_REASON_CODES for code in unique):
            validity = VALIDITY_UNKNOWN
        else:
            validity = VALIDITY_VALID
        return ValidityReport(
            validity=validity,
            reason_codes=tuple(unique),
            calibration_class=self.calibration_class,
            primary_scoring_eligible=self.calibration_class == CALIBRATION_PRIMARY and not self._reference_blockers(),
            parameters_hash=self.parameters_hash,
            calibration_reference_hash=self.calibration_reference_hash,
            parameter_set_id=self.parameters.parameter_set_id,
            flow_estimator=self.flow_estimator,
            expected_static=expected_static,
            expected_static_source=expected_static_source,
            checks=dict(checks),
        )


__all__ = [
    "ALL_REASON_CODES",
    "CALIBRATION_CLASSES",
    "CALIBRATION_DEVELOPMENT",
    "CALIBRATION_PRIMARY",
    "FLOW_BLOCK_MATCHING",
    "FLOW_ESTIMATORS",
    "FLOW_FARNEBACK",
    "INVALID_REASON_CODES",
    "MotionReference",
    "REFERENCE_SOURCE_KINDS",
    "ReferenceTrajectory",
    "SOURCE_DEVELOPMENT_SYNTHETIC",
    "SOURCE_REAL",
    "STAGE_A_GATE_VERSION",
    "StageAConfigurationError",
    "StageAParameters",
    "StageAValidityGate",
    "UNKNOWN_REASON_CODES",
    "VALIDITY_INVALID",
    "VALIDITY_UNKNOWN",
    "VALIDITY_VALID",
    "ValidityReasonCode",
    "ValidityReport",
    "resolve_flow_estimator",
]

REFERENCE_SOURCE_KINDS = SOURCE_KINDS
