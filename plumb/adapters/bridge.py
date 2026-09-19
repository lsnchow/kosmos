"""Stateful Bridge control integration and the pinned Cosmos 10-D compiler.

This is intentionally a small, pure-Python transcription of the relevant
released transforms.  It is not a claim that simulated state is measured robot
state: :class:`BridgeControlIntegrator` always produces ``forecast`` state.

Pinned sources:

* Bridge controller ``_next_qpos`` and ``action2transform_local`` at
  ``rail-berkeley/bridge_data_robot@b841131...``;
* Cosmos Bridge dataset at ``NVIDIA/cosmos-framework@c23e51...``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from .contracts import CapabilityResult, CapabilityStatus, COSMOS3_FRAMEWORK_COMMIT, FeedbackMode


Matrix3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
Matrix4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]
Vector3 = Tuple[float, float, float]
Action7 = Tuple[float, float, float, float, float, float, float]
Action10 = Tuple[float, float, float, float, float, float, float, float, float, float]


# Exact constants from cosmos_framework/.../bridge_orig_lerobot_dataset.py
DEFAULT_ROTATION: Matrix3 = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
BRIDGE_TO_OPENCV: Matrix3 = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
TCP_TO_FLANGE: Matrix4 = (
    (1.0, 0.0, 0.0, -0.093575),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % label)
    return value


def _matmul3(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _matvec3(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(sum(matrix[row][k] * vector[k] for k in range(3)) for row in range(3))  # type: ignore[return-value]


def _transpose3(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def _add3(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _sub3(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def euler_xyz_matrix(euler: Sequence[float]) -> Matrix3:
    """Match released Bridge ``eulerAnglesToRotationMatrix``: ``Rz @ Ry @ Rx``."""

    if len(euler) != 3:
        raise ValueError("Euler rotation must have three values.")
    x, y, z = (_finite(value, "euler") for value in euler)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx: Matrix3 = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    ry: Matrix3 = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    rz: Matrix3 = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    return _matmul3(rz, _matmul3(ry, rx))


def matrix_to_euler_xyz(rotation: Matrix3) -> Vector3:
    """Match released Bridge ``rotationMatrixToEulerAngles`` including gimbal branch."""

    sy = math.sqrt(rotation[0][0] ** 2 + rotation[1][0] ** 2)
    if sy >= 1e-6:
        return (
            math.atan2(rotation[2][1], rotation[2][2]),
            math.atan2(-rotation[2][0], sy),
            math.atan2(rotation[1][0], rotation[0][0]),
        )
    return (
        math.atan2(-rotation[1][2], rotation[1][1]),
        math.atan2(-rotation[2][0], sy),
        0.0,
    )


def pose(rotation: Matrix3, translation: Vector3) -> Matrix4:
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], translation[0]),
        (rotation[1][0], rotation[1][1], rotation[1][2], translation[1]),
        (rotation[2][0], rotation[2][1], rotation[2][2], translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def pose_parts(value: Matrix4) -> Tuple[Matrix3, Vector3]:
    return (
        ((value[0][0], value[0][1], value[0][2]), (value[1][0], value[1][1], value[1][2]), (value[2][0], value[2][1], value[2][2])),
        (value[0][3], value[1][3], value[2][3]),
    )


def pose_mul(left: Matrix4, right: Matrix4) -> Matrix4:
    l_rotation, l_translation = pose_parts(left)
    r_rotation, r_translation = pose_parts(right)
    return pose(_matmul3(l_rotation, r_rotation), _add3(_matvec3(l_rotation, r_translation), l_translation))


def pose_inverse(value: Matrix4) -> Matrix4:
    rotation, translation = pose_parts(value)
    rotation_transposed = _transpose3(rotation)
    inverse_translation = tuple(-element for element in _matvec3(rotation_transposed, translation))
    return pose(rotation_transposed, inverse_translation)  # type: ignore[arg-type]


def _pose_with_rotation(value: Matrix4, rotation: Matrix3) -> Matrix4:
    _, translation = pose_parts(value)
    return pose(rotation, translation)


@dataclass(frozen=True)
class BridgeState:
    """Bridge state in the controller's Euler convention.

    ``gripper`` is the continuous desired position, not necessarily the raw
    action's final gripper channel.  That distinction is preserved by the
    compiler because Cosmos training data uses the action gripper channel.
    """

    xyz: Vector3
    euler_xyz: Vector3
    gripper: float
    source: str = "forecast"

    @classmethod
    def from_values(cls, values: Sequence[float], source: str = "forecast") -> "BridgeState":
        # Source records sometimes include an unused seventh state field before
        # continuous gripper.  Support 7-D and stated 8-D layout explicitly.
        if len(values) not in (7, 8):
            raise ValueError("Bridge state must be 7-D or 8-D, got %d." % len(values))
        xyz = tuple(_finite(values[index], "state") for index in range(3))
        euler = tuple(_finite(values[index], "state") for index in range(3, 6))
        gripper = _finite(values[-1], "state gripper")
        return cls(xyz=xyz, euler_xyz=euler, gripper=gripper, source=source)  # type: ignore[arg-type]

    def values8(self) -> Tuple[float, ...]:
        return self.xyz + self.euler_xyz + (0.0, self.gripper)


class GripperMode(str, Enum):
    ABSOLUTE = "absolute"
    DELTA = "delta"


@dataclass(frozen=True)
class BridgeControlProfile:
    """Values must be copied from a pinned controller/fixture before qualification."""

    profile_id: str
    source_revision: str = "b841131ecd512bafb303075bd8f8b677e0bf9f1f"
    gripper_mode: GripperMode = GripperMode.ABSOLUTE
    action_low: Action7 = (-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi, 0.0)
    action_high: Action7 = (1.0, 1.0, 1.0, math.pi, math.pi, math.pi, 1.0)
    workspace_low: Optional[Vector3] = None
    workspace_high: Optional[Vector3] = None
    gripper_low: float = 0.0
    gripper_high: float = 1.0
    pose_origin: str = "last_commanded_pose"

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("BridgeControlProfile.profile_id is required.")
        if len(self.action_low) != 7 or len(self.action_high) != 7:
            raise ValueError("Bridge action limits must be 7-D.")
        if (self.workspace_low is None) != (self.workspace_high is None):
            raise ValueError("workspace_low and workspace_high must be provided together.")
        if self.gripper_low > self.gripper_high:
            raise ValueError("Invalid gripper range.")


@dataclass(frozen=True)
class ForecastState:
    state: BridgeState
    feedback_mode: FeedbackMode = FeedbackMode.FORECAST_STATE
    integration_source: str = "last_commanded_pose"


class BridgeControlIntegrator:
    """Released controller local-action transform with explicit forecast labeling."""

    def __init__(self, profile: BridgeControlProfile) -> None:
        self.profile = profile

    def capability(self) -> CapabilityResult:
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Implements pinned local action transform and controller update, but forecast state "
                "requires held-out trajectory validation before it can support a qualified cell."
            ),
            source_verified=True,
            evidence_uris=(
                "https://github.com/rail-berkeley/bridge_data_robot/blob/"
                "b841131ecd512bafb303075bd8f8b677e0bf9f1f/"
                "widowx_envs/widowx_envs/base/robot_base_env.py#L182-L246",
            ),
            details={"feedback_mode": FeedbackMode.FORECAST_STATE.value},
        )

    def clip_action(self, action: Sequence[float]) -> Action7:
        if len(action) != 7:
            raise ValueError("Bridge control action must be exactly 7-D.")
        return tuple(
            min(max(_finite(action[index], "action"), self.profile.action_low[index]), self.profile.action_high[index])
            for index in range(7)
        )  # type: ignore[return-value]

    def advance(self, previous: BridgeState, action: Sequence[float]) -> ForecastState:
        # ``RobotBaseEnv.step`` clips the complete physical action before
        # calling ``_next_qpos``. Preserve that action for both state evolution
        # and the Cosmos action gripper channel.
        clipped = self.clip_action(action)

        # Exact released action2transform_local: T_eef @ T_delta @ inv(T_eef),
        # then controller's delta_transform.dot(previous_target_transform).
        eef = pose(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), previous.xyz)
        local_delta = pose(euler_xyz_matrix(clipped[3:6]), clipped[0:3])
        delta = pose_mul(pose_mul(eef, local_delta), pose_inverse(eef))
        previous_transform = pose(_matmul3(euler_xyz_matrix(previous.euler_xyz), DEFAULT_ROTATION), previous.xyz)
        next_transform = pose_mul(delta, previous_transform)
        rotation, translation = pose_parts(next_transform)

        if self.profile.workspace_low is not None and self.profile.workspace_high is not None:
            translation = tuple(
                min(max(translation[index], self.profile.workspace_low[index]), self.profile.workspace_high[index])
                for index in range(3)
            )  # type: ignore[assignment]
            next_transform = pose(rotation, translation)  # type: ignore[arg-type]

        if self.profile.gripper_mode is GripperMode.ABSOLUTE:
            gripper = clipped[6]
        else:
            gripper = previous.gripper + clipped[6]
        gripper = min(max(gripper, self.profile.gripper_low), self.profile.gripper_high)

        # Released transform2state removes DEFAULT_ROTATION before Euler extraction.
        target_euler = matrix_to_euler_xyz(_matmul3(rotation, _transpose3(DEFAULT_ROTATION)))
        return ForecastState(
            state=BridgeState(xyz=translation, euler_xyz=target_euler, gripper=gripper, source="forecast"),  # type: ignore[arg-type]
            integration_source=self.profile.pose_origin,
        )

    def integrate(self, initial: BridgeState, actions: Sequence[Sequence[float]]) -> Tuple[BridgeState, ...]:
        states = [initial]
        current = initial
        for action in actions:
            current = self.advance(current, action).state
            states.append(current)
        return tuple(states)


class NormalizationMethod(str, Enum):
    QUANTILE = "quantile"
    MEANSTD = "meanstd"
    MINMAX = "minmax"


@dataclass(frozen=True)
class ActionNormalizer:
    """Portable exact implementation of Framework action_normalization.py."""

    revision: str
    method: NormalizationMethod
    low: Tuple[float, ...]
    high: Tuple[float, ...]
    source_path: Optional[str] = None

    @classmethod
    def from_json(cls, path: str, revision: str, method: str = "quantile") -> "ActionNormalizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        parsed = NormalizationMethod(method)
        key_low, key_high = {
            NormalizationMethod.QUANTILE: ("q01", "q99"),
            NormalizationMethod.MEANSTD: ("mean", "std"),
            NormalizationMethod.MINMAX: ("min", "max"),
        }[parsed]
        if key_low not in data or key_high not in data:
            raise ValueError("Normalizer file lacks %s/%s." % (key_low, key_high))
        return cls(
            revision=revision,
            method=parsed,
            low=tuple(float(value) for value in data[key_low]),
            high=tuple(float(value) for value in data[key_high]),
            source_path=str(path),
        )

    def _validate(self, values: Sequence[float]) -> None:
        if len(values) != len(self.low) or len(self.low) != len(self.high):
            raise ValueError("Normalizer width does not match the action width.")

    def normalize(self, values: Sequence[float]) -> Tuple[float, ...]:
        self._validate(values)
        output: List[float] = []
        for value, low, high in zip(values, self.low, self.high):
            if self.method is NormalizationMethod.MEANSTD:
                denominator = max(high, 1e-8)
                output.append((float(value) - low) / denominator)
            else:
                denominator = max(high - low, 1e-8)
                output.append(2.0 * (float(value) - low) / denominator - 1.0)
        return tuple(output)

    def denormalize(self, values: Sequence[float]) -> Tuple[float, ...]:
        self._validate(values)
        output: List[float] = []
        for value, low, high in zip(values, self.low, self.high):
            if self.method is NormalizationMethod.MEANSTD:
                output.append(float(value) * high + low)
            else:
                output.append(0.5 * (float(value) + 1.0) * (high - low) + low)
        return tuple(output)


class NormalizationBoundary(str, Enum):
    EXTERNAL = "external"
    BACKEND = "backend"


@dataclass(frozen=True)
class CompiledBridgeActions:
    raw_actions: Tuple[Action10, ...]
    backend_actions: Tuple[Action10, ...]
    absolute_cosmos_poses: Tuple[Matrix4, ...]
    forecast_states: Tuple[BridgeState, ...]
    normalizer_revision: Optional[str]
    normalization_boundary: NormalizationBoundary
    normalization_applications: int
    feedback_mode: FeedbackMode = FeedbackMode.FORECAST_STATE


class BridgeToCosmosCompiler:
    """Compile a stateful Bridge control trajectory into Cosmos 10-D actions.

    The compiler copies the NVIDIA dataset's state -> FK/TCP/OpenCV transforms,
    then uses ``T_i^-1 @ T_(i+1)`` and column-major rot6d.  The caller chooses
    whether a pinned normalizer is applied externally or inside the backend;
    this makes applying it twice mechanically visible in artifacts.
    """

    def __init__(
        self,
        integrator: BridgeControlIntegrator,
        normalizer: Optional[ActionNormalizer],
        normalization_boundary: NormalizationBoundary = NormalizationBoundary.BACKEND,
    ) -> None:
        self.integrator = integrator
        self.normalizer = normalizer
        self.normalization_boundary = normalization_boundary

    @staticmethod
    def _cosmos_absolute_pose(state: BridgeState) -> Matrix4:
        state_pose = pose(_matmul3(euler_xyz_matrix(state.euler_xyz), DEFAULT_ROTATION), state.xyz)
        tcp_pose = pose_mul(state_pose, TCP_TO_FLANGE)
        rotation, translation = pose_parts(tcp_pose)
        return pose(_matmul3(rotation, BRIDGE_TO_OPENCV), translation)

    @staticmethod
    def _relative_to_rot6d(relative: Matrix4, gripper_action: float) -> Action10:
        rotation, translation = pose_parts(relative)
        # Framework uses matrix[:, :, :2].transpose(0, 2, 1).reshape(-1, 6).
        return (
            translation[0],
            translation[1],
            translation[2],
            rotation[0][0],
            rotation[1][0],
            rotation[2][0],
            rotation[0][1],
            rotation[1][1],
            rotation[2][1],
            _finite(gripper_action, "gripper action"),
        )

    def compile(self, initial_state: BridgeState, bridge_actions: Sequence[Sequence[float]]) -> CompiledBridgeActions:
        if not bridge_actions:
            raise ValueError("Cannot compile an empty Bridge trajectory.")
        canonical_actions = tuple(self.integrator.clip_action(action) for action in bridge_actions)
        states = self.integrator.integrate(initial_state, canonical_actions)
        absolute = tuple(self._cosmos_absolute_pose(state) for state in states)
        raw: List[Action10] = []
        for index, action in enumerate(canonical_actions):
            relative = pose_mul(pose_inverse(absolute[index]), absolute[index + 1])
            raw.append(self._relative_to_rot6d(relative, action[6]))

        if self.normalization_boundary is NormalizationBoundary.EXTERNAL:
            if self.normalizer is None:
                raise ValueError("External normalization requires a pinned ActionNormalizer.")
            backend = tuple(self.normalizer.normalize(action) for action in raw)
            applications = 1
        else:
            backend = tuple(raw)
            applications = 0

        return CompiledBridgeActions(
            raw_actions=tuple(raw),
            backend_actions=backend,  # type: ignore[arg-type]
            absolute_cosmos_poses=absolute,
            forecast_states=states,
            normalizer_revision=self.normalizer.revision if self.normalizer else None,
            normalization_boundary=self.normalization_boundary,
            normalization_applications=applications,
        )

    def capability(self) -> CapabilityResult:
        return CapabilityResult(
            status=CapabilityStatus.READY_UNQUALIFIED,
            reason=(
                "Pinned matrix conversion is implemented; checkpoint-specific normalizer boundary and held-out "
                "Bridge fixture still need Gate A/B evidence."
            ),
            source_verified=True,
            evidence_uris=(
                "https://github.com/NVIDIA/cosmos-framework/blob/%s/"
                "cosmos_framework/data/generator/action/datasets/bridge_orig_lerobot_dataset.py" % COSMOS3_FRAMEWORK_COMMIT,
            ),
            details={
                "normalization_boundary": self.normalization_boundary.value,
                "normalizer_revision": self.normalizer.revision if self.normalizer else None,
                "feedback_mode": FeedbackMode.FORECAST_STATE.value,
            },
        )
