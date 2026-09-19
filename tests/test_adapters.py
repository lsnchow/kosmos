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

    def test_bounded_cases_reuse_one_injected_pipeline_and_continue(self) -> None:
        adapter, pipeline = self._adapter()
        rows = tuple(tuple(float(column) for column in range(10)) for _ in range(16))
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
            self.assertEqual([report["status"] for report in reports], ["completed"] * 6)
            self.assertEqual([call["action"].chunk_size for call in pipeline.calls], [16, 16, 1, 4, 16, 16])
            self.assertEqual([report["result"]["returned_frame_count"] for report in reports], [17, 17, 2, 5, 17, 17])
            self.assertEqual(reports[2]["qualification"], "gate_b_probe_only_not_qualified")
            self.assertTrue(reports[-1]["case"]["first_action_fixed_against_baseline"])
            self.assertEqual(len(artifact_calls), 7)  # one shared condition artifact + each case
            self.assertTrue((Path(tempdir) / "future-suffix-permuted-16.json").is_file())

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
            measurements["interventions"] = {"original": 1, "zero": 1, "permuted": 1, "sign_reversed": 1, "cross_episode": 1}
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
