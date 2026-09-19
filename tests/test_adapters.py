"""Pure-Python contract tests for PLUMB adapters.

These tests use fixture-backed fake backends only to test serialization and
never mark a gate as passed.  Real GPU calls belong to ``adapters.smoke``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from plumb.adapters import (
    ActionNormalizer,
    BridgeControlIntegrator,
    BridgeControlProfile,
    BridgeState,
    BridgeToCosmosCompiler,
    CapabilityStatus,
    Cosmos3NanoDiffusersAdapter,
    Cosmos3NanoDiffusersProfile,
    FeedbackMode,
    GripperMode,
    IRASimBridgeAdapter,
    IRASimBridgeProfile,
    NormalizationBoundary,
    WorldRequest,
)
from plumb.adapters.worlds import BackendContractError, _CosmosRuntime
from plumb.adapters import smoke
from plumb.adapters import probe_suite
from plumb.gates import (
    GateLedger,
    GateRecord,
    GateStatus,
    QualificationValidator,
    canonical_json_sha256,
)


class BridgeCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial = BridgeState.from_values((0.31, -0.11, 0.24, 0.2, -0.3, 0.4, 0.0, 0.39), source="measured")
        self.actions = (
            (0.01, -0.02, 0.03, 0.04, -0.05, 0.06, 0.0),
            (-0.02, 0.01, -0.01, -0.03, 0.02, 0.01, 0.39),
        )

    def test_pinned_stateful_matrix_golden(self) -> None:
        profile = BridgeControlProfile(profile_id="golden", gripper_mode=GripperMode.ABSOLUTE)
        compiler = BridgeToCosmosCompiler(BridgeControlIntegrator(profile), None)
        result = compiler.compile(self.initial, self.actions)
        # Golden values are the source-convention calculation for a nonidentity
        # start pose: local controller transform -> state/FK -> TCP -> OpenCV ->
        # backward framewise relative pose -> column-order rot6d.
        expected = (
            (0.0198064442, -0.0057143526, -0.0319342192, 0.9973135932, -0.0644811177,
             0.0347531635, 0.0626848322, 0.9967543812, 0.0505105473, 0.0),
            (-0.0176379972, 0.0130735643, 0.0083590463, 0.9998686007, -0.0092219384,
             -0.0133318132, 0.0087659924, 0.9993880422, -0.0338629374, 0.39),
        )
        for actual_row, expected_row in zip(result.raw_actions, expected):
            for actual, wanted in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, wanted, places=7)
        self.assertEqual(result.feedback_mode, FeedbackMode.FORECAST_STATE)
        self.assertEqual(result.forecast_states[0].source, "measured")
        self.assertEqual(result.forecast_states[1].source, "forecast")

    def test_quantile_normalizer_is_applied_once_and_invertible(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "normalizer.json"
            path.write_text(json.dumps({"q01": [0.0] * 10, "q99": [2.0] * 10}), encoding="utf-8")
            normalizer = ActionNormalizer.from_json(str(path), revision="fixture-q", method="quantile")
            compiler = BridgeToCosmosCompiler(
                BridgeControlIntegrator(BridgeControlProfile(profile_id="normalizer")),
                normalizer,
                normalization_boundary=NormalizationBoundary.EXTERNAL,
            )
            result = compiler.compile(self.initial, self.actions)
            self.assertEqual(result.normalization_applications, 1)
            self.assertNotEqual(result.raw_actions, result.backend_actions)
            for raw, transformed in zip(result.raw_actions, result.backend_actions):
                restored = normalizer.denormalize(transformed)
                for actual, wanted in zip(restored, raw):
                    self.assertAlmostEqual(actual, wanted, places=10)

    def test_integrator_records_forecast_not_measured_state(self) -> None:
        integrator = BridgeControlIntegrator(BridgeControlProfile(profile_id="integrator"))
        next_state = integrator.advance(self.initial, self.actions[0])
        self.assertEqual(next_state.feedback_mode, FeedbackMode.FORECAST_STATE)
        self.assertEqual(next_state.state.source, "forecast")

    def test_controller_clipping_is_reflected_in_compiled_gripper(self) -> None:
        compiler = BridgeToCosmosCompiler(
            BridgeControlIntegrator(BridgeControlProfile(profile_id="clipping")), None
        )
        compiled = compiler.compile(self.initial, ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0),))
        self.assertEqual(compiled.raw_actions[0][-1], 1.0)
        self.assertEqual(compiled.forecast_states[-1].gripper, 1.0)


class Rot6dCertificationTests(unittest.TestCase):
    """Spec section 2: a standalone Euler round trip cannot validate rot6d.

    These tests reconstruct rotation matrices from the compiled 6-D columns,
    recompute the pinned stateful transform chain independently in numpy, and
    exercise nonidentity/near-gimbal rotations, gripper endpoints, and
    normalizer inversion.  They complement, and do not replace, the hardcoded
    golden in :class:`BridgeCompilerTests`.
    """

    def setUp(self) -> None:
        self.initial = BridgeState.from_values((0.31, -0.11, 0.24, 0.2, -0.3, 0.4, 0.0, 0.39), source="measured")
        self.actions = (
            (0.01, -0.02, 0.03, 0.04, -0.05, 0.06, 0.0),
            (-0.02, 0.01, -0.01, -0.03, 0.02, 0.01, 0.39),
        )

    @staticmethod
    def _compiler(profile_id="rot6d", **kwargs):
        return BridgeToCosmosCompiler(
            BridgeControlIntegrator(BridgeControlProfile(profile_id=profile_id, **kwargs)), None
        )

    @staticmethod
    def _rotation_from_rot6d(row):
        """Rebuild the full rotation from the two encoded columns."""

        import numpy as np

        first = np.asarray(row[3:6], dtype=np.float64)
        second = np.asarray(row[6:9], dtype=np.float64)
        third = np.cross(first, second)
        return np.column_stack([first, second, third])

    @staticmethod
    def _independent_absolute_pose(state):
        """Recompute the pinned state -> FK -> TCP -> OpenCV chain in numpy."""

        import numpy as np

        from plumb.adapters.bridge import BRIDGE_TO_OPENCV, DEFAULT_ROTATION, TCP_TO_FLANGE

        x, y, z = state.euler_xyz
        rotation_x = np.asarray(
            [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]], dtype=np.float64
        )
        rotation_y = np.asarray(
            [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]], dtype=np.float64
        )
        rotation_z = np.asarray(
            [[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]], dtype=np.float64
        )
        euler = rotation_z @ rotation_y @ rotation_x
        state_pose = np.eye(4)
        state_pose[:3, :3] = euler @ np.asarray(DEFAULT_ROTATION, dtype=np.float64)
        state_pose[:3, 3] = np.asarray(state.xyz, dtype=np.float64)
        tcp = state_pose @ np.asarray(TCP_TO_FLANGE, dtype=np.float64)
        result = np.eye(4)
        result[:3, :3] = tcp[:3, :3] @ np.asarray(BRIDGE_TO_OPENCV, dtype=np.float64)
        result[:3, 3] = tcp[:3, 3]
        return result

    def test_rot6d_columns_reconstruct_an_orthonormal_right_handed_rotation(self) -> None:
        import numpy as np

        compiled = self._compiler().compile(self.initial, self.actions)
        for row in compiled.raw_actions:
            rotation = self._rotation_from_rot6d(row)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)
            self.assertAlmostEqual(float(np.linalg.norm(rotation[:, 0])), 1.0, places=9)
            self.assertAlmostEqual(float(np.linalg.norm(rotation[:, 1])), 1.0, places=9)
            self.assertAlmostEqual(float(rotation[:, 0] @ rotation[:, 1]), 0.0, places=9)

    def test_absolute_poses_match_an_independent_numpy_transform_chain(self) -> None:
        import numpy as np

        compiled = self._compiler().compile(self.initial, self.actions)
        for state, pose in zip(compiled.forecast_states, compiled.absolute_cosmos_poses):
            np.testing.assert_allclose(
                np.asarray(pose, dtype=np.float64), self._independent_absolute_pose(state), atol=1e-12
            )

    def test_compiled_rows_equal_an_independent_relative_transform_computation(self) -> None:
        import numpy as np

        compiled = self._compiler().compile(self.initial, self.actions)
        for index, row in enumerate(compiled.raw_actions):
            previous = self._independent_absolute_pose(compiled.forecast_states[index])
            following = self._independent_absolute_pose(compiled.forecast_states[index + 1])
            relative = np.linalg.inv(previous) @ following
            np.testing.assert_allclose(np.asarray(row[0:3]), relative[:3, 3], atol=1e-12)
            # Column-major rot6d: first column then second column.
            np.testing.assert_allclose(np.asarray(row[3:6]), relative[:3, 0], atol=1e-12)
            np.testing.assert_allclose(np.asarray(row[6:9]), relative[:3, 1], atol=1e-12)
            np.testing.assert_allclose(self._rotation_from_rot6d(row), relative[:3, :3], atol=1e-9)

    def test_nonidentity_rotations_including_near_gimbal_stay_orthonormal(self) -> None:
        import math

        import numpy as np

        starts = (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.1, 0.2, 0.3, math.pi / 3, -math.pi / 4, 2.9, 0.0, 0.2),
            (0.1, 0.2, 0.3, 0.3, math.pi / 2, 0.4, 0.0, 0.2),
            (0.1, 0.2, 0.3, 0.3, -math.pi / 2, 0.4, 0.0, 0.2),
            (0.1, 0.2, 0.3, 3.1, 1.5707, -3.1, 0.0, 0.2),
        )
        rotating = ((0.005, -0.004, 0.003, 0.5, -0.4, 0.3, 0.1),)
        for values in starts:
            state = BridgeState.from_values(values, source="measured")
            compiled = self._compiler().compile(state, rotating)
            row = compiled.raw_actions[0]
            self.assertTrue(all(math.isfinite(value) for value in row))
            rotation = self._rotation_from_rot6d(row)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)
            independent = np.linalg.inv(
                self._independent_absolute_pose(compiled.forecast_states[0])
            ) @ self._independent_absolute_pose(compiled.forecast_states[1])
            np.testing.assert_allclose(rotation, independent[:3, :3], atol=1e-9)

    def test_gimbal_branch_is_actually_exercised_by_a_pitch_of_ninety_degrees(self) -> None:
        import math

        integrator = BridgeControlIntegrator(BridgeControlProfile(profile_id="gimbal"))
        state = BridgeState.from_values((0.1, 0.2, 0.3, 0.3, math.pi / 2, 0.4, 0.0, 0.2), source="measured")
        advanced = integrator.advance(state, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2))
        # The released ``rotationMatrixToEulerAngles`` gimbal branch sets yaw to
        # exactly zero, so this asserts the branch ran rather than assuming it.
        self.assertEqual(advanced.state.euler_xyz[2], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in advanced.state.euler_xyz))

    def test_gripper_endpoints_travel_from_the_action_channel(self) -> None:
        compiler = self._compiler("gripper-endpoints")
        for gripper in (0.0, 1.0):
            compiled = compiler.compile(self.initial, ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper),))
            self.assertEqual(compiled.raw_actions[0][9], gripper)
            self.assertEqual(compiled.forecast_states[-1].gripper, gripper)
        below = compiler.compile(self.initial, ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0),))
        self.assertEqual(below.raw_actions[0][9], 0.0)

    def test_action_gripper_units_are_not_state_gripper_units(self) -> None:
        """Delta mode: the 10-D row carries the action channel, not the state."""

        compiler = self._compiler("gripper-delta", gripper_mode=GripperMode.DELTA)
        compiled = compiler.compile(self.initial, ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2),))
        self.assertAlmostEqual(compiled.raw_actions[0][9], 0.2, places=12)
        self.assertAlmostEqual(compiled.forecast_states[-1].gripper, 0.59, places=12)
        self.assertNotAlmostEqual(compiled.raw_actions[0][9], compiled.forecast_states[-1].gripper, places=3)

    def test_backend_boundary_applies_the_normalizer_zero_times(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "normalizer.json"
            path.write_text(json.dumps({"q01": [-1.0] * 10, "q99": [3.0] * 10}), encoding="utf-8")
            normalizer = ActionNormalizer.from_json(str(path), revision="fixture-q", method="quantile")
            compiled = BridgeToCosmosCompiler(
                BridgeControlIntegrator(BridgeControlProfile(profile_id="backend-boundary")),
                normalizer,
                normalization_boundary=NormalizationBoundary.BACKEND,
            ).compile(self.initial, self.actions)
            self.assertEqual(compiled.normalization_applications, 0)
            self.assertEqual(compiled.raw_actions, compiled.backend_actions)
            self.assertEqual(compiled.normalizer_revision, "fixture-q")

    def test_a_double_normalizer_application_is_mechanically_detectable(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "normalizer.json"
            path.write_text(json.dumps({"q01": [-1.0] * 10, "q99": [3.0] * 10}), encoding="utf-8")
            normalizer = ActionNormalizer.from_json(str(path), revision="fixture-q", method="quantile")
            compiled = BridgeToCosmosCompiler(
                BridgeControlIntegrator(BridgeControlProfile(profile_id="external-boundary")),
                normalizer,
                normalization_boundary=NormalizationBoundary.EXTERNAL,
            ).compile(self.initial, self.actions)
            self.assertEqual(compiled.normalization_applications, 1)
            once = compiled.backend_actions[0]
            twice = normalizer.normalize(once)
            self.assertNotEqual(once, twice)
            for restored, expected in zip(normalizer.denormalize(once), compiled.raw_actions[0]):
                self.assertAlmostEqual(restored, expected, places=12)
            # Inverting once-normalized rows must not recover a twice-normalized row.
            self.assertNotEqual(tuple(normalizer.denormalize(twice)), tuple(compiled.raw_actions[0]))

    def test_rot6d_is_not_validated_by_a_standalone_euler_round_trip(self) -> None:
        """A pure Euler round trip passes even when the pose chain is dropped.

        This is the spec's stated reason for matrix-equivalence tests: the
        rotation reaching rot6d is the *relative OpenCV-frame* transform, not
        the raw commanded Euler delta, so an Euler-only check cannot catch a
        missing DEFAULT_ROTATION/TCP/OpenCV stage.
        """

        import numpy as np

        from plumb.adapters.bridge import euler_xyz_matrix, matrix_to_euler_xyz

        commanded = (0.5, -0.4, 0.3)
        round_tripped = matrix_to_euler_xyz(euler_xyz_matrix(commanded))
        for actual, expected in zip(round_tripped, commanded):
            self.assertAlmostEqual(actual, expected, places=12)

        compiled = self._compiler("euler-vs-matrix").compile(
            BridgeState.from_values((0.0,) * 8, source="measured"),
            ((0.0, 0.0, 0.0) + commanded + (0.0,),),
        )
        naive = np.asarray(euler_xyz_matrix(commanded), dtype=np.float64)
        actual_rotation = self._rotation_from_rot6d(compiled.raw_actions[0])
        self.assertGreater(float(np.abs(actual_rotation - naive).max()), 1e-6)


class ActionLengthCertificationTests(unittest.TestCase):
    """Action lengths must come from probed evidence, not a hardcoded guess."""

    @staticmethod
    def _payload(**overrides):
        payload = {
            "schema_version": 1,
            "kind": "plumb_world_action_length_certification",
            "certification_id": "cosmos3-nano-fd-lengths",
            "profile_id": "cosmos-fd",
            "backend": "cosmos3_diffusers",
            "domain": "bridge_orig_lerobot",
            "control_hz": 5.0,
            "source_uri": "fixture://evidence/action-lengths.json",
            "lengths": [
                {
                    "action_length": 16,
                    "status": "supported",
                    "returned_frame_count": 17,
                    "evidence_uri": "fixture://evidence/n16.json",
                    "evidence_sha256": "a" * 64,
                    "job_id": "937398",
                },
                {
                    "action_length": 4,
                    "status": "supported",
                    "returned_frame_count": 5,
                    "evidence_uri": "fixture://evidence/n4.json",
                    "evidence_sha256": "b" * 64,
                    "job_id": "937398",
                },
                {
                    "action_length": 1,
                    "status": "unsupported",
                    "returned_frame_count": 1,
                    "evidence_uri": "fixture://evidence/n1.json",
                    "evidence_sha256": "c" * 64,
                    "note": "returned only the conditioning frame",
                },
            ],
        }
        payload.update(overrides)
        return payload

    def _write(self, tempdir, payload):
        path = Path(tempdir) / "certification.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loader_hashes_the_artifact_and_separates_supported_from_probed(self) -> None:
        from plumb.adapters.worlds import certified_action_lengths

        with tempfile.TemporaryDirectory() as tempdir:
            path = self._write(tempdir, self._payload())
            certification = certified_action_lengths(path, profile_id="cosmos-fd")
            self.assertEqual(certification.supported_lengths, (4, 16))
            self.assertEqual(certification.unsupported_lengths, (1,))
            self.assertTrue(certification.source_hash.startswith("sha256:"))
            self.assertEqual(certification.source_hash, certification.source_sha256)
            self.assertEqual(certification.certification_class, "gate_a_certified")
            self.assertEqual(
                certification.source_hash,
                "sha256:" + __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            )

    def test_a_profile_built_from_certification_cannot_claim_an_extra_length(self) -> None:
        from plumb.adapters.worlds import CertificationError, certified_action_lengths

        with tempfile.TemporaryDirectory() as tempdir:
            certification = certified_action_lengths(self._write(tempdir, self._payload()))
            profile = Cosmos3NanoDiffusersProfile.from_certification(
                certification, local_model_path="/injected", probe_action_lengths=(1,)
            )
            self.assertEqual(profile.allowed_action_lengths, (4, 16))
            self.assertEqual(profile.probe_action_lengths, (1,))
            self.assertEqual(profile.action_length_certification_class, "gate_a_certified")
            self.assertEqual(profile.action_length_certification_source_hash, certification.source_hash)
            with self.assertRaises(CertificationError):
                Cosmos3NanoDiffusersProfile(
                    profile_id="cosmos-fd",
                    local_model_path="/injected",
                    allowed_action_lengths=(4, 16, 70),
                    action_length_certification=certification,
                )

    def test_the_uncertified_default_is_the_single_reviewed_length(self) -> None:
        from plumb.adapters.worlds import UNCERTIFIED_COSMOS_ACTION_LENGTHS

        profile = Cosmos3NanoDiffusersProfile(profile_id="cosmos-fd", local_model_path="/injected")
        self.assertEqual(UNCERTIFIED_COSMOS_ACTION_LENGTHS, (16,))
        self.assertEqual(profile.allowed_action_lengths, (16,))
        self.assertEqual(profile.action_length_certification_class, "uncertified_default")
        self.assertIsNone(profile.action_length_certification_source_hash)
        details = Cosmos3NanoDiffusersAdapter(profile).capability().details
        self.assertEqual(details["action_length_certification_class"], "uncertified_default")
        self.assertIsNone(details["action_length_certification_source_hash"])

    def test_a_probe_that_broke_the_frame_contract_cannot_be_certified(self) -> None:
        from plumb.adapters.worlds import CertificationError, certified_action_lengths

        payload = self._payload()
        payload["lengths"] = [
            {
                "action_length": 1,
                "status": "supported",
                "returned_frame_count": 1,
                "evidence_uri": "fixture://evidence/n1.json",
                "evidence_sha256": "c" * 64,
            }
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(CertificationError) as raised:
                certified_action_lengths(self._write(tempdir, payload))
            self.assertIn("N -> N+1 frame contract", str(raised.exception))

    def test_evidence_free_or_foreign_certifications_are_refused(self) -> None:
        from plumb.adapters.worlds import CertificationError, certified_action_lengths

        missing_evidence = self._payload()
        missing_evidence["lengths"] = [
            {"action_length": 16, "status": "supported", "returned_frame_count": 17, "evidence_sha256": "a" * 64}
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(CertificationError):
                certified_action_lengths(self._write(tempdir, missing_evidence))
            with self.assertRaises(CertificationError):
                certified_action_lengths(self._write(tempdir, self._payload()), profile_id="other-profile")
            with self.assertRaises(CertificationError):
                certified_action_lengths(self._write(tempdir, self._payload(kind="something_else")))
            with self.assertRaises(CertificationError):
                certified_action_lengths(Path(tempdir) / "absent.json")
            no_supported = self._payload()
            no_supported["lengths"] = [
                {
                    "action_length": 1,
                    "status": "unsupported",
                    "evidence_uri": "fixture://evidence/n1.json",
                    "evidence_sha256": "c" * 64,
                }
            ]
            with self.assertRaises(CertificationError):
                certified_action_lengths(self._write(tempdir, no_supported))

    def test_padding_certificate_evidence_is_checked_not_trusted(self) -> None:
        from plumb.adapters.worlds import PROTOCOL_CERTIFIED_TERMINAL_PADDING, TerminalPaddingCertificate

        good = TerminalPaddingCertificate(
            certificate_id="tpc-1",
            profile_id="cosmos-fd",
            padded_action_length=4,
            certified_prefix_lengths=(2,),
            prefix_invariance_trials=12,
            max_observed_prefix_frame_mae=0.02,
            tolerance_frame_mae=0.25,
            evidence_uri="fixture://evidence/prefix-invariance.json",
            evidence_sha256="d" * 64,
            tolerance_source="protocol.Tolerances.max_suffix_invariance_mae",
        )
        self.assertEqual(good.errors(), ())
        self.assertEqual(good.protocol_identity, PROTOCOL_CERTIFIED_TERMINAL_PADDING)
        self.assertTrue(good.certifies(padded_length=4, consumed_prefix=2, profile_id="cosmos-fd"))
        self.assertFalse(good.certifies(padded_length=4, consumed_prefix=3, profile_id="cosmos-fd"))
        self.assertFalse(good.certifies(padded_length=16, consumed_prefix=2, profile_id="cosmos-fd"))
        self.assertFalse(good.certifies(padded_length=4, consumed_prefix=2, profile_id="another"))
        self.assertEqual(
            good.padding_actions([(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7)], 2),
            ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7),) * 2,
        )

        for overrides, expected in (
            ({"prefix_invariance_trials": 0}, "never measured"),
            ({"max_observed_prefix_frame_mae": 9.0}, "exceeds the preregistered tolerance"),
            ({"tolerance_frame_mae": 0.0}, "explicit positive preregistered tolerance"),
            ({"seed_matched": False}, "matched seed"),
            ({"evidence_uri": ""}, "URI and SHA-256"),
            ({"certified_prefix_lengths": ()}, "no consumed prefix"),
            ({"certified_prefix_lengths": (4,)}, "nothing was padded"),
        ):
            candidate = TerminalPaddingCertificate(
                **{
                    **{
                        "certificate_id": "tpc-1",
                        "profile_id": "cosmos-fd",
                        "padded_action_length": 4,
                        "certified_prefix_lengths": (2,),
                        "prefix_invariance_trials": 12,
                        "max_observed_prefix_frame_mae": 0.02,
                        "tolerance_frame_mae": 0.25,
                        "evidence_uri": "fixture://evidence/prefix-invariance.json",
                        "evidence_sha256": "d" * 64,
                    },
                    **overrides,
                }
            )
            self.assertTrue(
                any(expected in problem for problem in candidate.errors()),
                msg="%r did not report %r (got %s)" % (overrides, expected, candidate.errors()),
            )
            self.assertFalse(candidate.certifies(padded_length=4, consumed_prefix=2))


class _FakeTensor(tuple):
    pass


class _FakeGenerator:
    def manual_seed(self, seed):
        self.seed = seed
        return self


class _FakeTorch:
    float32 = "float32"
    bfloat16 = "bfloat16"
    cuda = None

    @staticmethod
    def as_tensor(value, dtype=None):
        return _FakeTensor(tuple(value))

    @staticmethod
    def Generator(device=None):
        return _FakeGenerator()


class _FakeActionCondition:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakePipeline:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(video=self.frames)


class _DynamicFakePipeline(_FakePipeline):
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        count = kwargs["action"].chunk_size
        return SimpleNamespace(video=list(range(count + 1)))


class CosmosAdapterTests(unittest.TestCase):
    def _adapter(self, frames):
        profile = Cosmos3NanoDiffusersProfile(
            profile_id="fixture-cosmos",
            local_model_path="/not-needed-for-injected-fixture",
            allowed_action_lengths=(16,),
            probe_action_lengths=(1,),
        )
        runtime = _CosmosRuntime(_FakeTorch, None, _FakeActionCondition)
        pipe = _FakePipeline(frames)
        adapter = Cosmos3NanoDiffusersAdapter(
            profile,
            runtime_factory=lambda: runtime,
            pipeline_factory=lambda _profile, _runtime: pipe,
        )
        return adapter, pipe

    @staticmethod
    def _request(count, profile_id="fixture-cosmos"):
        return WorldRequest(
            conditioning_image=object(),
            prompt="Close the drawer",
            domain="bridge_orig_lerobot",
            compiled_actions=tuple(tuple(float(column) for column in range(10)) for _ in range(count)),
            nominal_control_timestamps=tuple((index + 1) / 5.0 for index in range(count)),
            seed=7,
            compatibility_profile_id=profile_id,
        )

    def test_real_serializer_shape_prompt_and_model_specific_frame_count(self) -> None:
        adapter, pipe = self._adapter(list(range(17)))
        result = adapter.generate(self._request(16))
        self.assertEqual(len(result.frames), 17)
        self.assertEqual(len(result.future_frames), 16)
        self.assertEqual(result.timing.backend_calls, 1)
        request = pipe.calls[0]
        self.assertNotIn("num_frames", request)
        self.assertNotIn("image", request)
        self.assertEqual(request["prompt"], "Close the drawer")
        self.assertEqual(request["action"].mode, "forward_dynamics")
        self.assertEqual(request["action"].chunk_size, 16)
        self.assertEqual(request["action"].domain_name, "bridge_orig_lerobot")
        self.assertEqual(len(request["action"].raw_actions[0]), 10)
        self.assertEqual(result.metadata["action_length_status"], "configured_unqualified")

    def test_one_tick_is_explicit_probe_not_qualified(self) -> None:
        adapter, _ = self._adapter(["condition", "future"])
        result = adapter.generate(self._request(1))
        self.assertEqual(result.metadata["action_length_status"], "probe_unqualified")
        self.assertEqual(len(result.future_frames), 1)

    def test_bad_frame_count_is_never_fabricated(self) -> None:
        adapter, _ = self._adapter(list(range(16)))
        with self.assertRaises(ValueError):
            adapter.generate(self._request(16))


class SmokeArtifactTests(unittest.TestCase):
    def test_np_frame_normalization_handles_hwc_float_and_chw_minus_one_to_one(self) -> None:
        import numpy as np

        hwc = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        self.assertEqual(smoke._normalise_frame_to_uint8(hwc).tolist(), [[[0, 128, 255]]])
        chw = np.stack(
            [np.full((2, 5), -1.0), np.zeros((2, 5)), np.full((2, 5), 1.0)], axis=0
        ).astype(np.float32)
        self.assertEqual(smoke._normalise_frame_to_uint8(chw)[0, 0].tolist(), [0, 128, 255])

    def test_fixture_reader_rejects_nonofficial_action_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            action_path = Path(tempdir) / "actions.json"
            action_path.write_text(json.dumps([[0.0] * 10 for _ in range(15)]), encoding="utf-8")
            with self.assertRaises(ValueError):
                smoke._read_action_rows(action_path)

    def test_cli_persists_failure_report_without_ml_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            folder = Path(tempdir)
            action_path = folder / "bad-actions.json"
            action_path.write_text(json.dumps([[0.0] * 10 for _ in range(15)]), encoding="utf-8")
            report_path = folder / "report.json"
            code = smoke.main([
                "cosmos", "--model-path", str(folder / "model"), "--actions", str(action_path),
                "--vision", str(folder / "missing.mp4"), "--report", str(report_path),
            ])
            self.assertEqual(code, 2)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"]["type"], "ValueError")
            self.assertIn("traceback", report["error"])
            self.assertEqual(report["request"]["fixture_action_count"], 16)

    def test_png_artifact_helper_writes_first_and_last_without_model_runtime(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tempdir:
            records = smoke._write_visual_artifacts(
                [np.zeros((2, 3, 3), dtype=np.uint8), np.full((2, 3, 3), 255, dtype=np.uint8)],
                fps=5.0,
                output_video=None,
                output_frame_dir=str(Path(tempdir) / "frames"),
                frame_selection="first-last",
            )
            self.assertEqual([item["frame_index"] for item in records["frames"]], [0, 1])
            self.assertTrue(all(item["sha256"].startswith("sha256:") for item in records["frames"]))


class ProbeSuiteTests(unittest.TestCase):
    def _adapter(self):
        profile = Cosmos3NanoDiffusersProfile(
            profile_id="fixture-probes",
            local_model_path="/injected",
            allowed_action_lengths=(16,),
            probe_action_lengths=(1, 4),
        )
        runtime = _CosmosRuntime(_FakeTorch, None, _FakeActionCondition)
        pipe = _DynamicFakePipeline([])
        adapter = Cosmos3NanoDiffusersAdapter(
            profile,
            runtime_factory=lambda: runtime,
            pipeline_factory=lambda _profile, _runtime: pipe,
        )
        return adapter, pipe

    @staticmethod
    def _rows(offset: float = 0.0):
        return tuple(tuple(float(column) + offset for column in range(10)) for _ in range(16))

    def test_bounded_cases_reuse_one_injected_pipeline_and_continue(self) -> None:
        adapter, pipeline = self._adapter()
        rows = self._rows()
        cases = probe_suite._case_definitions(rows, include_suffix_perturbation=True)
        artifact_calls = []

        def fake_artifact_writer(frames, **kwargs):
            artifact_calls.append((len(frames), kwargs))
            return {"fake": {"frame_count": len(frames)}}

        with tempfile.TemporaryDirectory() as tempdir:
            reports = probe_suite.run_cases(
                adapter, cases, condition_frame=object(), prompt="Close the drawer", seed=9,
                output_dir=Path(tempdir), provenance={"fixture": "real"}, fps=5.0,
                frame_selection="first-last", artifact_writer=fake_artifact_writer,
            )
            self.assertEqual([report["status"] for report in reports], ["completed"] * 9)
            self.assertEqual(
                [call["action"].chunk_size for call in pipeline.calls], [16, 16, 1, 4, 16, 16, 16, 16, 16]
            )
            self.assertEqual(
                [report["result"]["returned_frame_count"] for report in reports],
                [17, 17, 2, 5, 17, 17, 17, 17, 17],
            )
            self.assertEqual(reports[2]["qualification"], "gate_b_probe_only_not_qualified")
            self.assertTrue(reports[-1]["case"]["first_action_fixed_against_baseline"])
            self.assertEqual(len(artifact_calls), 10)  # one shared condition artifact + each case
            self.assertTrue((Path(tempdir) / "future-suffix-permuted-16.json").is_file())
            # Every non-baseline arm carries an intervention-effect comparison
            # and the explicit distributional criterion.
            for report in reports[1:]:
                self.assertIn("raw_frame_comparison_to_baseline", report)
                self.assertIn("shuffled trajectory may still succeed", report["intervention_effect_criterion"])

    def test_gate_b_arm_names_match_the_gate_ledger(self) -> None:
        from plumb.gates import REQUIRED_GATE_B_ARMS

        self.assertEqual(probe_suite.REQUIRED_GATE_B_ARMS, REQUIRED_GATE_B_ARMS)

    def test_all_five_paired_arms_and_both_extra_controls_are_constructed(self) -> None:
        cases = probe_suite._case_definitions(
            self._rows(),
            include_suffix_perturbation=True,
            cross_episode_actions=self._rows(offset=1.0),
            condition_frame_keys=("already_successful", "stationary_contact"),
        )
        coverage = probe_suite.gate_b_arm_coverage(cases)
        self.assertEqual(coverage["missing_arms"], [])
        self.assertEqual(coverage["missing_additional_controls"], [])
        self.assertEqual(
            coverage["present_arms"],
            {
                "cross_episode": "cross-episode",
                "original": "baseline-16",
                "sign_reversed": "sign-reversed",
                "temporally_permuted": "temporally-permuted-prefix",
                "zero": "zero-encoded-vectors-16",
            },
        )
        self.assertEqual(coverage["additional_controls"]["legitimate_stationary"], ["stationary-contact-16"])
        self.assertEqual(coverage["additional_controls"]["already_successful"], ["already-successful-start"])

    def test_absent_cross_episode_and_start_inputs_are_reported_not_fabricated(self) -> None:
        cases = probe_suite._case_definitions(self._rows(), include_suffix_perturbation=True)
        identifiers = [case.case_id for case in cases]
        self.assertNotIn("cross-episode", identifiers)
        self.assertNotIn("already-successful-start", identifiers)
        coverage = probe_suite.gate_b_arm_coverage(cases)
        self.assertEqual(coverage["missing_arms"], ["cross_episode"])
        self.assertEqual(coverage["missing_additional_controls"], ["already_successful"])
        with self.assertRaises(ValueError):
            probe_suite._case_definitions(
                self._rows(), False, cross_episode_actions=self._rows()  # identical to the baseline
            )
        with self.assertRaises(ValueError):
            probe_suite._case_definitions(self._rows(), False, cross_episode_actions=self._rows(1.0)[:8])

    def test_temporal_permutation_reorders_the_consumed_prefix_without_editing_rows(self) -> None:
        rows = self._rows()
        case = next(
            item
            for item in probe_suite._case_definitions(rows, True)
            if item.case_id == "temporally-permuted-prefix"
        )
        self.assertEqual(sorted(probe_suite.TEMPORAL_PERMUTATION), list(range(16)))
        self.assertNotEqual(probe_suite.TEMPORAL_PERMUTATION, tuple(reversed(range(16))))
        self.assertEqual(sorted(case.actions), sorted(rows))
        self.assertEqual(case.gate_b_arm, "temporally_permuted")
        # Distinct from the suffix probe: row 0 is not held fixed.
        self.assertFalse(case.first_action_fixed_against_baseline)

    def test_sign_reversal_negates_translation_only_and_states_its_limitation(self) -> None:
        rows = self._rows(offset=1.0)
        case = next(item for item in probe_suite._case_definitions(rows, False) if item.case_id == "sign-reversed")
        for original, reversed_row in zip(rows, case.actions):
            self.assertEqual(reversed_row[0:3], tuple(-value for value in original[0:3]))
            self.assertEqual(reversed_row[3:], original[3:])
        self.assertIn("off-manifold", case.limitation)
        self.assertEqual(case.gate_b_arm, "sign_reversed")

    def test_stationary_control_is_an_on_manifold_identity_transform(self) -> None:
        rows = self._rows()
        case = next(
            item for item in probe_suite._case_definitions(rows, False) if item.case_id == "stationary-contact-16"
        )
        for original, row in zip(rows, case.actions):
            self.assertEqual(row[0:3], (0.0, 0.0, 0.0))
            self.assertEqual(row[3:9], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))  # identity rot6d columns
            self.assertEqual(row[9], original[9])  # source gripper is held
        self.assertIn("A legitimately stationary or contact *scene*", case.limitation)
        self.assertIsNone(case.gate_b_arm)

    def test_alternate_condition_frames_are_used_only_where_supplied(self) -> None:
        adapter, _pipeline = self._adapter()
        cases = probe_suite._case_definitions(
            self._rows(),
            include_suffix_perturbation=False,
            condition_frame_keys=("already_successful", "stationary_contact"),
        )
        baseline_frame = object()
        successful_frame = object()
        contact_frame = object()

        def fake_artifact_writer(frames, **kwargs):
            return {"fake": {"frame_count": len(frames)}}

        with tempfile.TemporaryDirectory() as tempdir:
            reports = probe_suite.run_cases(
                adapter, cases, condition_frame=baseline_frame, prompt="Close the drawer", seed=3,
                output_dir=Path(tempdir), provenance={}, fps=5.0, frame_selection="first-last",
                artifact_writer=fake_artifact_writer,
                condition_frames={"already_successful": successful_frame, "stationary_contact": contact_frame},
            )
        by_id = {report["case"]["id"]: report for report in reports}
        self.assertEqual(by_id["baseline-16"]["condition_artifact"]["condition_frame_key"], "baseline")
        self.assertEqual(
            by_id["already-successful-start"]["condition_artifact"]["condition_frame_key"], "already_successful"
        )
        self.assertEqual(
            by_id["stationary-contact-16"]["condition_artifact"]["condition_frame_key"], "stationary_contact"
        )
        self.assertTrue(by_id["stationary-contact-16"]["condition_artifact"]["requested_condition_frame_available"])

    def test_gate_b_evidence_uses_canonical_keys_and_never_claims_a_pass(self) -> None:
        adapter, _pipeline = self._adapter()
        cases = probe_suite._case_definitions(
            self._rows(),
            include_suffix_perturbation=True,
            cross_episode_actions=self._rows(offset=1.0),
            condition_frame_keys=("already_successful", "stationary_contact"),
        )

        def fake_artifact_writer(frames, **kwargs):
            return {"fake": {"frame_count": len(frames)}}

        with tempfile.TemporaryDirectory() as tempdir:
            reports = probe_suite.run_cases(
                adapter, cases, condition_frame=object(), prompt="Close the drawer", seed=1,
                output_dir=Path(tempdir), provenance={}, fps=5.0, frame_selection="first-last",
                artifact_writer=fake_artifact_writer,
                condition_frames={"already_successful": object(), "stationary_contact": object()},
            )
        evidence = probe_suite.gate_b_evidence(reports, cases)
        self.assertEqual(set(evidence["interventions"]), set(probe_suite.REQUIRED_GATE_B_ARMS))
        self.assertIsNotNone(evidence["suffix_causality"])
        self.assertEqual(evidence["suffix_causality"]["case_id"], "future-suffix-permuted-16")
        self.assertEqual(evidence["feedback_mode"], "unqualified")
        self.assertFalse(evidence["qualifies_gate_b"])
        self.assertTrue(evidence["qualification_blockers"])
        self.assertIn("never receives actions", evidence["criterion"])
        # A suite without the suffix probe leaves suffix_causality null so the
        # gate refuses a Gate-B pass rather than inferring one.
        without_suffix = probe_suite._case_definitions(self._rows(), include_suffix_perturbation=False)
        self.assertIsNone(probe_suite.gate_b_evidence([], without_suffix)["suffix_causality"])

    def test_probe_case_rejects_an_unknown_gate_b_arm_name(self) -> None:
        with self.assertRaises(ValueError):
            probe_suite.ProbeCase("x", self._rows(), "kind", "description", gate_b_arm="permuted")

    def test_raw_metrics_compare_frames_before_video_encoding(self) -> None:
        import numpy as np

        baseline = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(17)]
        candidate = [np.full((2, 3, 3), index + 2, dtype=np.uint8) for index in range(17)]
        metrics = probe_suite._raw_frame_metrics(baseline, candidate)
        self.assertEqual(metrics["status"], "ok")
        self.assertEqual(metrics["comparison_space"], "raw_output_type_np_uint8_before_mp4_encode")
        self.assertEqual(metrics["pixel_mae_0_to_255"]["first_future"], 2.0)


class IRASimTests(unittest.TestCase):
    def test_irasim_is_separate_15_by_7_scaled_contract(self) -> None:
        adapter = IRASimBridgeAdapter(IRASimBridgeProfile(profile_id="irasim"))
        request = WorldRequest(
            conditioning_image=object(),
            prompt="Close the drawer",
            domain="bridge",
            compiled_actions=tuple((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7) for _ in range(15)),
            nominal_control_timestamps=tuple(index / 5.0 for index in range(15)),
            seed=1,
            compatibility_profile_id="irasim",
        )
        prepared = adapter.prepare(request)
        self.assertEqual(prepared.expected_frames, 16)
        self.assertEqual(prepared.scaled_actions[0], (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0.7))
        self.assertEqual(adapter.capability().status, CapabilityStatus.UNSUPPORTED)
        invalid = WorldRequest(
            conditioning_image=object(), prompt="Close the drawer", domain="bridge",
            compiled_actions=tuple((0.0,) * 10 for _ in range(15)),
            nominal_control_timestamps=tuple(index / 5.0 for index in range(15)), seed=1,
            compatibility_profile_id="irasim",
        )
        with self.assertRaises(BackendContractError):
            adapter.prepare(invalid)


class GateQualificationTests(unittest.TestCase):
    @staticmethod
    def _scenario():
        start = {
            "start_id": "s-1", "start_lineage_id": "lineage-1", "source_dataset": "bridge",
            "source_revision": "rev", "episode_id": "ep", "frame_id": "0", "image_hash": "sha256:image",
            "state_hash": "sha256:state", "instruction": "Close the drawer", "state_convention": "bridge8",
            "camera": "wrist", "scene": "drawer", "initial_state_stratum": "closed",
            "goal_reference_hashes": ["sha256:goal"],
        }
        manifest = {"schema_version": 1, "manifest_id": "scenario", "provenance_kind": "real_robot", "tasks": {"close_drawer": {"starts": [start]}}}
        manifest["sha256"] = canonical_json_sha256(manifest)
        return manifest

    @staticmethod
    def _protocol(scenario):
        protocol = {
            "protocol_id": "p", "status": "frozen", "scenario_manifest_hash": scenario["sha256"],
            "policy": {}, "world_model": {}, "judge": {}, "thresholds": {"v": 1}, "seeds": {"x": 1},
            "remote_record": "s3://immutable/p",
        }
        protocol["sha256"] = canonical_json_sha256(protocol)
        return protocol

    @staticmethod
    def _pass_record(gate_id, protocol_hash):
        measurements = {"backend_calls": 1, "wall_seconds": 0.1, "gpu_peak_memory_bytes": 1}
        if gate_id == "B":
            # Spec section 0 names five paired control arms plus a suffix-causality
            # result; the gate refuses a pass that is missing any of them.
            measurements["interventions"] = {
                "original": 1,
                "zero": 1,
                "temporally_permuted": 1,
                "sign_reversed": 1,
                "cross_episode": 1,
            }
            measurements["suffix_causality"] = {"max_frame1_mae": 0.0, "invariant": True}
            measurements["feedback_mode"] = "native_feedback"
        if gate_id == "C":
            measurements["starts_per_task"] = {
                "open_drawer": 50,
                "close_drawer": 50,
                "to_basket": 50,
                "to_sink": 50,
                "fold_cloth": 50,
            }
            measurements["comparability"] = {"close_drawer": "matched_provenance"}
        if gate_id == "D":
            measurements.update(
                {
                    "binary_kappa": 0.7,
                    "weighted_progress_kappa": 0.65,
                    "leniency_offset": 0.05,
                    "consensus_coverage": 0.9,
                    "calibration_class": "human",
                    "held_out_frozen_before_evaluation": True,
                }
            )
        return GateRecord(
            gate_id=gate_id, status=GateStatus.PASS, protocol_hash=protocol_hash,
            fixture_ids=("fixture",), start_ids=("s-1",) if gate_id == "C" else (),
            evidence_uris=("https://evidence.example/%s" % gate_id,), measurements=measurements,
            thresholds={"frozen": True}, evidence_kind="real_backend",
        )

    def test_synthetic_evidence_cannot_qualify(self) -> None:
        scenario = self._scenario()
        protocol = self._protocol(scenario)
        ledger = GateLedger.new(protocol["sha256"])
        for gate_id in ("A", "B", "C", "D"):
            record = self._pass_record(gate_id, protocol["sha256"])
            if gate_id == "A":
                record = GateRecord.from_mapping({**record.to_mapping(), "evidence_kind": "synthetic"}, gate_id)
            ledger.record(record)
        decision = QualificationValidator.validate(
            ledger, scenario, protocol, task="close_drawer", feedback_mode=FeedbackMode.NATIVE_FEEDBACK
        )
        self.assertEqual(decision.capability.status, CapabilityStatus.BLOCKED)
        self.assertIn("synthetic", decision.capability.reason)

    def test_actual_evidence_and_frozen_manifest_can_qualify(self) -> None:
        scenario = self._scenario()
        protocol = self._protocol(scenario)
        ledger = GateLedger.new(protocol["sha256"])
        for gate_id in ("A", "B", "C", "D"):
            ledger.record(self._pass_record(gate_id, protocol["sha256"]))
        decision = QualificationValidator.validate(
            ledger, scenario, protocol, task="close_drawer", feedback_mode=FeedbackMode.NATIVE_FEEDBACK
        )
        self.assertEqual(decision.capability.status, CapabilityStatus.QUALIFIED)

    def test_forecast_state_never_becomes_qualified_cell(self) -> None:
        scenario = self._scenario()
        protocol = self._protocol(scenario)
        ledger = GateLedger.new(protocol["sha256"])
        for gate_id in ("A", "B", "C", "D"):
            ledger.record(self._pass_record(gate_id, protocol["sha256"]))
        decision = QualificationValidator.validate(
            ledger, scenario, protocol, task="close_drawer", feedback_mode=FeedbackMode.FORECAST_STATE
        )
        self.assertEqual(decision.capability.status, CapabilityStatus.BLOCKED)


if __name__ == "__main__":
    unittest.main()
