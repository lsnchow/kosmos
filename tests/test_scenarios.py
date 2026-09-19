"""Offline adversarial tests for the Gate C scenario panels and their driver."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cluster import build_scenarios as driver
from plumb.gates import EXACT_TASK_PROMPTS, GateRecord, ScenarioManifestValidator
from plumb.scenarios import (
    AUTO_EVAL_DATASET,
    BRIDGE_LEROBOT_DATASET,
    BRIDGE_STATE_CONVENTION,
    CALIBRATION_COHORT,
    COST_CONFIRMATION_COHORT,
    DEVELOPMENT_COHORT,
    MATCHED_DISTRIBUTION,
    MATCHED_PROVENANCE,
    PRIMARY_COHORT,
    PRIMARY_STARTS_PER_TASK,
    TASK_SCENE_FAMILY,
    TASKS,
    ComparisonAxis,
    IndependencePolicy,
    SceneClaim,
    ScenarioError,
    ScenarioManifest,
    SourceDataset,
    StartRecord,
    build_manifest,
    default_scene_claims,
    gate_c_evidence,
    independence_violations,
    read_manifest_jsonl,
    verbatim_prompt,
    verbatim_prompt_consistency,
)


REVISION = "a" * 40
COHORT_OFFSETS = {
    PRIMARY_COHORT: 0,
    DEVELOPMENT_COHORT: 1000,
    CALIBRATION_COHORT: 2000,
    COST_CONFIRMATION_COHORT: 3000,
}


def digest(label):
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def start(
    task,
    cohort,
    index,
    *,
    scene_family=None,
    episode_id=None,
    frame_index=None,
    world_seed=None,
    dataset=None,
    lineage=None,
    calibration=None,
    image_hash=None,
    state_hash=None,
):
    family = scene_family or TASK_SCENE_FAMILY[task]
    source = dataset or (AUTO_EVAL_DATASET if family == "drawer" else BRIDGE_LEROBOT_DATASET)
    episode = episode_id or "%s-%s-ep-%04d" % (family, task, index)
    frame = 0 if frame_index is None else frame_index
    key = "%s:%s:%s:%s" % (task, cohort, episode, frame)
    return StartRecord(
        start_id="start-%s-%s-%04d" % (task, cohort, index),
        task=task,
        cohort=cohort,
        start_lineage_id=lineage or "%s@%s:%s:frame-%06d" % (source, REVISION, episode, frame),
        source_dataset=source,
        source_revision=REVISION,
        episode_id=episode,
        frame_id="frame-%06d" % frame,
        frame_index=frame,
        image_hash=image_hash or digest("image:" + key),
        state_hash=state_hash or digest("state:" + key),
        image_timestamp="2026-09-19T00:00:00.000Z",
        state_timestamp="2026-09-19T00:00:00.000Z",
        state_convention=BRIDGE_STATE_CONVENTION,
        camera={
            "camera_id": "widowx_over_shoulder",
            "crop": {"kind": "identity", "width": 256, "height": 256},
            "calibration": calibration or {"status": "unresolved", "reason": "no scene fiducial recovered"},
        },
        scene={
            "scene_family": family,
            "objects": [family, "table"],
            "layout": "source_episode_only",
            "reset_randomization": {"status": "unresolved"},
        },
        initial_state_stratum="stratum-%d" % (index % 5),
        instruction=verbatim_prompt(task),
        goal_reference_hashes={"goal_image": digest("goal:" + key)},
        goal_reference_provenance={
            "source_uri": "hf://datasets/%s@%s/%s" % (source, REVISION, episode),
            "source_dataset": source,
        },
        source_scene_family=family,
        world_seed=world_seed,
    )


def all_starts(tasks=TASKS, primary=PRIMARY_STARTS_PER_TASK, development=3, calibration=3, cost=2):
    counts = (
        (PRIMARY_COHORT, primary),
        (DEVELOPMENT_COHORT, development),
        (CALIBRATION_COHORT, calibration),
        (COST_CONFIRMATION_COHORT, cost),
    )
    records = []
    for task in tasks:
        for cohort, count in counts:
            for index in range(count):
                records.append(start(task, cohort, index + COHORT_OFFSETS[cohort]))
    return records


def sources():
    return (
        SourceDataset(
            dataset_id=AUTO_EVAL_DATASET,
            revision=REVISION,
            role="scenario_and_label_source",
            scene_families=("drawer",),
            selection_rule="drawer scene index",
        ),
        SourceDataset(
            dataset_id=BRIDGE_LEROBOT_DATASET,
            revision=REVISION,
            role="scenario_source",
            scene_families=("basket", "sink", "cloth"),
            selection_rule="language instruction match",
        ),
    )


def manifest(**kwargs):
    return build_manifest("scenarios-test-v1", all_starts(**kwargs), sources())


class VerbatimPromptTests(unittest.TestCase):
    def test_prompts_match_gates_including_the_lowercase_fold(self):
        result = verbatim_prompt_consistency()
        self.assertEqual("pass", result["status"], result["mismatches"])
        self.assertTrue(result["lowercase_fold_preserved"])
        self.assertEqual("fold the cloth from top right to bottom left", verbatim_prompt("fold_cloth"))
        self.assertTrue(verbatim_prompt("fold_cloth").startswith("fold "))
        self.assertNotEqual("Fold", verbatim_prompt("fold_cloth")[:4])
        for task, prompt in EXACT_TASK_PROMPTS.items():
            self.assertEqual(prompt, verbatim_prompt(task))

    def test_a_paraphrased_instruction_is_refused_on_the_start_record(self):
        with self.assertRaises(ScenarioError):
            StartRecord(
                **{
                    **{
                        field: getattr(start("fold_cloth", PRIMARY_COHORT, 0), field)
                        for field in (
                            "start_id",
                            "task",
                            "cohort",
                            "start_lineage_id",
                            "source_dataset",
                            "source_revision",
                            "episode_id",
                            "frame_id",
                            "frame_index",
                            "image_hash",
                            "state_hash",
                            "image_timestamp",
                            "state_timestamp",
                            "state_convention",
                            "camera",
                            "scene",
                            "initial_state_stratum",
                            "goal_reference_hashes",
                            "goal_reference_provenance",
                            "source_scene_family",
                        )
                    },
                    "instruction": "Fold the cloth from top right to bottom left",
                }
            )


class SceneSubstitutionTests(unittest.TestCase):
    def test_a_drawer_start_cannot_populate_a_sink_task(self):
        with self.assertRaises(ScenarioError) as raised:
            start("to_sink", PRIMARY_COHORT, 0, scene_family="drawer", dataset=AUTO_EVAL_DATASET)
        message = str(raised.exception)
        self.assertIn("cannot populate task to_sink", message)
        self.assertIn("filled with drawer starts", message)
        self.assertIn("autoeval_sink_cloth_basket_scene_assets", message)

    def test_a_drawer_start_cannot_populate_the_cloth_or_basket_tasks_either(self):
        for task in ("fold_cloth", "to_basket"):
            with self.assertRaises(ScenarioError):
                start(task, PRIMARY_COHORT, 0, scene_family="drawer", dataset=AUTO_EVAL_DATASET)

    def test_auto_eval_cannot_be_declared_as_a_sink_scene_source(self):
        with self.assertRaises(ScenarioError):
            SourceDataset(
                dataset_id=AUTO_EVAL_DATASET,
                revision=REVISION,
                role="scenario_source",
                scene_families=("sink",),
                selection_rule="unsupported",
            )
        with self.assertRaises(ScenarioError):
            start("to_sink", PRIMARY_COHORT, 0, dataset=AUTO_EVAL_DATASET)


class CohortSeparationTests(unittest.TestCase):
    def test_a_lineage_crossing_two_cohorts_raises(self):
        records = all_starts(tasks=("open_drawer",))
        primary = next(item for item in records if item.cohort == PRIMARY_COHORT)
        index = next(
            position
            for position, item in enumerate(records)
            if item.cohort == CALIBRATION_COHORT
        )
        leaked = start(
            "open_drawer",
            CALIBRATION_COHORT,
            9999,
            lineage=primary.start_lineage_id,
            episode_id="distinct-episode",
        )
        records[index] = leaked
        with self.assertRaises(ScenarioError) as raised:
            build_manifest("leak", records, sources())
        self.assertIn("may appear once", str(raised.exception))

    def test_cohort_leakage_is_detected_when_two_distinct_starts_share_a_lineage_label(self):
        first = start("open_drawer", PRIMARY_COHORT, 0)
        second = start(
            "open_drawer",
            CALIBRATION_COHORT,
            1,
            episode_id="other-episode",
            lineage=first.start_lineage_id,
        )
        from plumb.measurement import validate_source_lineage_leakage

        result = validate_source_lineage_leakage(
            [
                {"start_lineage_id": first.start_lineage_id, "cohort": first.cohort},
                {"start_lineage_id": second.start_lineage_id, "cohort": second.cohort},
            ]
        )
        self.assertEqual("fail", result["status"])
        self.assertEqual(
            [PRIMARY_COHORT, CALIBRATION_COHORT].sort(), result["violations"][0]["cohorts"].sort()
        )

    def test_every_required_cohort_is_reported_when_missing(self):
        records = [start("open_drawer", PRIMARY_COHORT, index) for index in range(3)]
        built = build_manifest("partial", records, sources())
        self.assertEqual(
            [DEVELOPMENT_COHORT, CALIBRATION_COHORT, COST_CONFIRMATION_COHORT],
            [name for name in built.missing_cohorts()],
        )
        self.assertTrue(
            any(item.startswith("missing_cohort_panel:") for item in built.resolved_open_dependencies())
        )


class IndependenceTests(unittest.TestCase):
    def test_neighbouring_frames_from_one_episode_are_not_independent_starts(self):
        first = start("open_drawer", PRIMARY_COHORT, 0, episode_id="shared", frame_index=0)
        second = start("open_drawer", PRIMARY_COHORT, 1, episode_id="shared", frame_index=3)
        violations = independence_violations([first, second], IndependencePolicy(min_source_frame_gap=10))
        reasons = [item["reason"] for item in violations]
        self.assertIn("neighbouring_source_frames_are_not_independent_starts", reasons)
        with self.assertRaises(ScenarioError) as raised:
            build_manifest("neighbours", [first, second], sources())
        self.assertIn("not independent physical starts", str(raised.exception))

    def test_a_frame_gap_at_or_above_the_policy_is_accepted(self):
        first = start("open_drawer", PRIMARY_COHORT, 0, episode_id="shared", frame_index=0)
        second = start("open_drawer", PRIMARY_COHORT, 1, episode_id="shared", frame_index=10)
        self.assertEqual(
            (), independence_violations([first, second], IndependencePolicy(min_source_frame_gap=10))
        )

    def test_a_repeated_world_seed_is_not_a_new_start(self):
        first = start("open_drawer", PRIMARY_COHORT, 0, episode_id="ep-a", world_seed=7)
        second = start("open_drawer", PRIMARY_COHORT, 1, episode_id="ep-b", world_seed=7)
        reasons = [item["reason"] for item in independence_violations([first, second])]
        self.assertIn("repeated_world_seed", reasons)

    def test_reseeding_one_physical_start_is_not_a_new_start(self):
        first = start("open_drawer", PRIMARY_COHORT, 0, episode_id="ep-a", frame_index=4, world_seed=1)
        second = start(
            "open_drawer",
            PRIMARY_COHORT,
            1,
            episode_id="ep-a",
            frame_index=4,
            world_seed=2,
            lineage="distinct-lineage-label",
        )
        reasons = [item["reason"] for item in independence_violations([first, second])]
        self.assertIn("repeated_world_seed_is_not_a_new_start", reasons)

    def test_a_repeated_source_image_or_state_is_not_a_new_start(self):
        shared_image = digest("identical-pixels")
        shared_state = digest("identical-state")
        first = start(
            "open_drawer", PRIMARY_COHORT, 0, episode_id="ep-a", image_hash=shared_image, state_hash=shared_state
        )
        second = start(
            "open_drawer",
            PRIMARY_COHORT,
            1,
            episode_id="ep-b",
            image_hash=shared_image,
            state_hash=shared_state,
        )
        reasons = {item["reason"] for item in independence_violations([first, second])}
        self.assertIn("repeated_source_image", reasons)
        self.assertIn("repeated_source_state", reasons)


class SceneClaimTests(unittest.TestCase):
    def test_handoff_scene_claims_default_to_unverified(self):
        built = manifest()
        for task in TASKS:
            panel = built.panel(task)
            for claim in panel.scene_claims:
                self.assertFalse(claim.verified)
                self.assertIsNone(claim.source_record_uri)
        drawer = {claim.claim_id for claim in default_scene_claims("drawer")}
        cloth = {claim.claim_id for claim in default_scene_claims("cloth")}
        sink = {claim.claim_id for claim in default_scene_claims("sink")}
        self.assertEqual({"drawer_taped_to_table"}, drawer)
        self.assertEqual({"cloth_taped_to_table"}, cloth)
        self.assertEqual({"sink_thin_plastic_wrap"}, sink)
        # The handoff records no basket fixture detail; none is invented.
        self.assertEqual((), default_scene_claims("basket"))

    def test_a_claim_cannot_be_verified_without_naming_a_source_record(self):
        with self.assertRaises(ScenarioError):
            SceneClaim(
                claim_id="drawer_taped_to_table",
                statement="The drawer unit is taped to the table.",
                scene_family="drawer",
                verified=True,
            )
        verified = SceneClaim(
            claim_id="drawer_taped_to_table",
            statement="The drawer unit is taped to the table.",
            scene_family="drawer",
            verified=True,
            source_record_uri="hf://datasets/zhouzypaul/auto_eval/scene_notes.md",
            verification_method="read the source scene record and confirmed the tape",
        )
        self.assertTrue(verified.verified)

    def test_unverified_claims_are_reported_in_the_measurements(self):
        built = manifest()
        unverified = built.measurements()["unverified_scene_claims"]
        self.assertEqual(["drawer_taped_to_table"], unverified["open_drawer"])
        self.assertEqual(["sink_thin_plastic_wrap"], unverified["to_sink"])
        self.assertEqual(["cloth_taped_to_table"], unverified["fold_cloth"])
        self.assertEqual([], unverified["to_basket"])


class SuccessCriterionTests(unittest.TestCase):
    def test_the_drawer_metric_threshold_needs_an_unresolved_calibrated_reference(self):
        built = manifest()
        criterion = built.panel("open_drawer").success_criterion
        self.assertEqual("opened >= 1.5 cm", criterion.criterion)
        self.assertEqual("metric_threshold", criterion.measurement_kind)
        self.assertTrue(criterion.calibration_reference_required)
        self.assertEqual("unresolved", criterion.calibration_reference_status)
        self.assertIsNone(criterion.calibration_reference_uri)
        self.assertFalse(criterion.resolved)
        self.assertTrue(
            any("calibration_reference" in item for item in built.resolved_open_dependencies())
        )

    def test_a_visible_criterion_records_that_no_calibration_is_required(self):
        built = manifest()
        for task in ("close_drawer", "to_basket", "to_sink"):
            criterion = built.panel(task).success_criterion
            self.assertFalse(criterion.calibration_reference_required)
            self.assertEqual("not_required", criterion.calibration_reference_status)
            self.assertTrue(criterion.resolved)

    def test_every_camera_records_an_explicit_calibration_status(self):
        with self.assertRaises(ScenarioError):
            start("open_drawer", PRIMARY_COHORT, 0, calibration={"mm_per_pixel": 0.5})
        with self.assertRaises(ScenarioError):
            start("open_drawer", PRIMARY_COHORT, 0, calibration={"status": "resolved"})


class ComparabilityTests(unittest.TestCase):
    def test_drawer_panels_are_matched_provenance_and_others_are_matched_distribution(self):
        built = manifest()
        self.assertEqual(MATCHED_PROVENANCE, built.panel("open_drawer").comparability)
        self.assertEqual(MATCHED_PROVENANCE, built.panel("close_drawer").comparability)
        for task in ("to_basket", "to_sink", "fold_cloth"):
            self.assertEqual(MATCHED_DISTRIBUTION, built.panel(task).comparability)

    def test_a_matched_distribution_panel_must_enumerate_its_limitations(self):
        built = manifest()
        panel = built.panel("to_sink")
        self.assertGreaterEqual(len(panel.limitations), 2)
        joined = " ".join(panel.limitations)
        self.assertIn("not automatically the same scene distribution", joined)
        self.assertIn("trial-level starting states", joined)
        self.assertIn(
            "Matched-distribution comparison", panel.as_mapping()["comparability_meaning"]
        )

    def test_a_non_drawer_task_cannot_claim_matched_provenance(self):
        records = [start("to_sink", PRIMARY_COHORT, index) for index in range(2)]
        from plumb.scenarios import TaskPanel, UNPAIRED_REFERENCE_LIMITATION

        with self.assertRaises(ScenarioError):
            TaskPanel(
                task="to_sink",
                comparability=MATCHED_PROVENANCE,
                starts_by_cohort={PRIMARY_COHORT: tuple(records)},
                limitations=(UNPAIRED_REFERENCE_LIMITATION,),
            )

    def test_all_five_gate_c_axes_default_to_unresolved(self):
        built = manifest()
        for task in TASKS:
            self.assertEqual(5, len(built.panel(task).unresolved_axes()))
        self.assertEqual(
            "pass_with_limitations", built.panel("close_drawer").task_status()
        )

    def test_a_matched_axis_must_name_its_evidence(self):
        with self.assertRaises(ScenarioError):
            ComparisonAxis(axis="objects", status="matched")
        self.assertEqual(
            "matched", ComparisonAxis(axis="objects", status="matched", evidence_uri="file://x").status
        )


class ManifestValidationTests(unittest.TestCase):
    def test_the_manifest_validates_against_the_gate_validator(self):
        built = manifest()
        self.assertEqual((), built.validate())
        mapping = built.to_manifest_mapping()
        self.assertEqual((), ScenarioManifestValidator.validate(mapping, require_primary_panel=True))
        for task in TASKS:
            self.assertEqual(PRIMARY_STARTS_PER_TASK, len(mapping["tasks"][task]["starts"]))
        self.assertTrue(mapping["sha256"].startswith("sha256:"))

    def test_a_short_primary_panel_fails_the_fifty_start_requirement(self):
        built = manifest(primary=PRIMARY_STARTS_PER_TASK - 1)
        errors = built.validate(require_primary_panel=True)
        self.assertTrue(any("exactly 50 distinct primary starts" in item for item in errors))
        self.assertEqual("incomplete", built.manifest_status())

    def test_synthetic_provenance_cannot_qualify_a_real_reference_cell(self):
        with self.assertRaises(ScenarioError):
            ScenarioManifest(
                manifest_id="synthetic-attempt",
                provenance_kind="synthetic",
                panels=manifest().panels,
                sources=sources(),
            )

    def test_cohort_starts_are_excluded_from_the_primary_panel_count(self):
        built = manifest(development=4, calibration=4, cost=4)
        mapping = built.to_manifest_mapping()
        self.assertEqual(PRIMARY_STARTS_PER_TASK, len(mapping["tasks"]["open_drawer"]["starts"]))
        self.assertEqual(4, len(mapping["cohorts"][DEVELOPMENT_COHORT]["tasks"]["open_drawer"]["starts"]))
        self.assertEqual(
            PRIMARY_STARTS_PER_TASK + 12, sum(built.cohort_counts()[c]["open_drawer"] for c in built.cohort_counts())
        )

    def test_an_incomplete_manifest_names_its_unacquired_panels(self):
        built = build_manifest("drawer-only", all_starts(tasks=("open_drawer", "close_drawer")), sources())
        self.assertEqual(("to_basket", "to_sink", "fold_cloth"), built.absent_tasks())
        self.assertEqual("incomplete", built.manifest_status())
        split = built.partition_errors(built.validate())
        self.assertEqual((), split["blocking"])
        self.assertTrue(split["unacquired_task_panels"])


class SerializationTests(unittest.TestCase):
    def test_jsonl_round_trip_binds_every_start_to_the_manifest_hash(self):
        built = manifest()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "scenarios.jsonl"
            written = built.write_jsonl(path)
            self.assertEqual(built.sha256, written["sha256"])
            back = read_manifest_jsonl(path)
            self.assertEqual("pass", back["status"])
            self.assertEqual(len(built.all_starts()), back["start_record_count"])
            self.assertEqual(built.sha256, back["declared_sha256"])
            self.assertEqual({built.sha256}, {row["manifest_sha256"] for row in back["starts"]})
            self.assertEqual("complete", back["header"]["manifest_status"])

    def test_write_refuses_to_overwrite_an_existing_manifest(self):
        built = manifest()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "scenarios.jsonl"
            built.write_jsonl(path)
            with self.assertRaises(ScenarioError):
                built.write_jsonl(path)
            built.write_jsonl(path, overwrite=True)

    def test_an_invalid_manifest_is_never_written(self):
        built = manifest(primary=2)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "scenarios.jsonl"
            with self.assertRaises(ScenarioError):
                built.write_jsonl(path)
            self.assertFalse(path.exists())

    def test_the_written_json_is_byte_identical_to_the_hashed_mapping(self):
        built = manifest()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "scenarios.json"
            built.write_json(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(built.to_manifest_mapping(), payload)


class GateCEvidenceTests(unittest.TestCase):
    def test_evidence_is_blocked_while_dependencies_are_open_and_loads_as_a_gate_record(self):
        built = manifest()
        bundle = gate_c_evidence(
            built, evidence_uris=["file:///evidence/scenarios.json"], protocol_hash="sha256:" + "b" * 64
        )
        self.assertEqual("blocked", bundle["status"])
        self.assertEqual(
            {"pass_with_limitations"}, set(bundle["measurements"]["per_task_gate_status"].values())
        )
        self.assertTrue(bundle["open_dependencies"])
        record = GateRecord.from_mapping(bundle)
        self.assertEqual("C", record.gate_id)
        self.assertEqual("blocked", record.status.value)
        self.assertEqual(len(built.all_starts()), len(record.start_ids))
        self.assertEqual(PRIMARY_STARTS_PER_TASK, record.thresholds["primary_starts_per_task"])
        self.assertEqual("real_source_frames", record.evidence_kind)

    def test_evidence_without_a_protocol_hash_or_uri_records_that_reason(self):
        bundle = gate_c_evidence(manifest(), evidence_uris=[])
        self.assertIn("Gate C evidence bundle needs at least one evidence URI", bundle["reasons"])
        self.assertIn("Gate C evidence bundle needs the frozen protocol hash", bundle["reasons"])

    def test_a_passed_gate_c_record_would_still_need_real_evidence_fields(self):
        built = manifest()
        bundle = gate_c_evidence(
            built, evidence_uris=["file:///x"], protocol_hash="sha256:" + "b" * 64
        )
        forced = dict(bundle)
        forced["status"] = "pass"
        record = GateRecord.from_mapping(forced)
        # Gate C's own evidence requirements in plumb.gates must already be
        # satisfied by this bundle's shape; only the open dependencies block it.
        self.assertEqual((), record.pass_evidence_errors())
        self.assertEqual(
            PRIMARY_STARTS_PER_TASK, record.measurements["starts_per_task"]["open_drawer"]
        )
        self.assertIn("limitations", record.measurements["comparability"]["to_sink"])


class AcquisitionDriverTests(unittest.TestCase):
    def test_plan_is_a_dry_run_and_never_contacts_a_network(self):
        code, plan = driver.run(
            [
                "plan",
                "--manifest-id",
                "scen-v1",
                "--revision",
                "%s=%s" % (AUTO_EVAL_DATASET, REVISION),
                "--revision",
                "%s=%s" % (BRIDGE_LEROBOT_DATASET, "b" * 40),
            ]
        )
        self.assertEqual(0, code)
        self.assertEqual(5, len(plan["tasks"]))
        self.assertEqual(
            {AUTO_EVAL_DATASET, BRIDGE_LEROBOT_DATASET},
            {item["source_dataset"] for item in plan["tasks"]},
        )
        drawer = next(item for item in plan["tasks"] if item["task"] == "open_drawer")
        self.assertEqual(AUTO_EVAL_DATASET, drawer["source_dataset"])
        self.assertEqual("matched_provenance", drawer["comparability"])
        sink = next(item for item in plan["tasks"] if item["task"] == "to_sink")
        self.assertEqual(BRIDGE_LEROBOT_DATASET, sink["source_dataset"])
        self.assertEqual("matched_distribution_with_limitations", sink["comparability"])
        self.assertIn("not automatically the same scene distribution", sink["use_limit"])
        self.assertTrue(all("--execute" in item["command"] for item in plan["download_commands"]))

    def test_the_plan_refuses_a_download_root_outside_cluster_scratch(self):
        with self.assertRaises(driver.AcquisitionError) as raised:
            driver.build_plan(manifest_id="scen-v1", revisions={}, root="/tmp/plumb")
        self.assertIn("cluster scratch", str(raised.exception))
        with self.assertRaises(driver.AcquisitionError):
            driver.build_plan(manifest_id="scen-v1", revisions={}, root="/scratch/me/elsewhere")
        plan = driver.build_plan(manifest_id="scen-v1", revisions={}, root="/scratch/me/plumb")
        self.assertEqual("/scratch/me/plumb", plan["root"])
        # The documented placeholder still renders a plan without touching disk.
        self.assertEqual(
            "/scratch/<user>/plumb", driver.build_plan(manifest_id="s", revisions={})["root"]
        )

    def test_the_plan_refuses_a_stub_unpickler_and_names_the_refused_modules(self):
        code, plan = driver.run(["plan", "--manifest-id", "scen-v1"])
        self.assertEqual(0, code)
        conversion = plan["safe_conversion"]
        self.assertIn("robot_eval_logger", conversion["refused_module_prefixes"])
        self.assertIn("wandb", conversion["refused_module_prefixes"])
        self.assertIn("not a security boundary", conversion["reason"])

    def test_refused_and_unreviewed_pickle_globals_are_never_stubbed(self):
        with self.assertRaises(driver.AcquisitionError) as raised:
            driver.validate_pickle_globals(["numpy.ndarray", "robot_eval_logger.logger.EvalLogger"])
        self.assertIn("robot_eval_logger", str(raised.exception))
        self.assertIn("not a security boundary", str(raised.exception))
        with self.assertRaises(driver.AcquisitionError):
            driver.validate_pickle_globals(["wandb.sdk.wandb_run.Run"])
        with self.assertRaises(driver.AcquisitionError):
            driver.validate_pickle_globals(["some.unreviewed.Thing"])
        self.assertEqual(
            ["numpy.dtype", "numpy.ndarray"],
            driver.validate_pickle_globals(["numpy.ndarray", "numpy.dtype"]),
        )

    def test_conversion_refuses_a_checksum_mismatch_before_deserializing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode.pkl"
            source.write_bytes(b"not-really-a-pickle")
            report = driver.convert_auto_eval_pickle(
                source, root / "episode.json", expected_sha256="sha256:" + "0" * 64, execute=True
            )
        self.assertEqual("failed", report["status"])
        self.assertEqual("source checksum mismatch", report["error"]["message"])

    def test_conversion_dry_run_attempts_no_deserialization(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode.pkl"
            source.write_bytes(b"payload")
            report = driver.convert_auto_eval_pickle(
                source,
                root / "episode.json",
                expected_sha256=driver.sha256_file(source),
                execute=False,
            )
        self.assertEqual("planned", report["status"])
        self.assertIn("Dry run", report["note"])
        self.assertIn("isolation_requirement", report)

    def test_select_reports_a_shortfall_instead_of_padding_the_panel(self):
        rows = [
            {"episode_id": "sink-ep-%04d" % index, "language_instruction": verbatim_prompt("to_sink")}
            for index in range(5)
        ]
        result = driver.select_starts(
            task="to_sink",
            index_rows=rows,
            source_dataset=BRIDGE_LEROBOT_DATASET,
            source_revision=REVISION,
            counts=driver.cohort_counts(),
        )
        self.assertEqual("insufficient_source_episodes", result["status"])
        self.assertEqual(5, result["selected_starts"])
        self.assertIn("do not pad this cell", result["reason"])
        self.assertTrue(result["shortfall"])

    def test_select_ignores_episodes_whose_instruction_does_not_match(self):
        rows = [
            {"episode_id": "ep-1", "language_instruction": "Put the carrot in the bowl"},
            {"episode_id": "ep-2", "language_instruction": verbatim_prompt("to_basket")},
        ]
        result = driver.select_starts(
            task="to_basket",
            index_rows=rows,
            source_dataset=BRIDGE_LEROBOT_DATASET,
            source_revision=REVISION,
            counts={PRIMARY_COHORT: 1},
        )
        self.assertEqual(1, result["matched_episodes"])
        self.assertEqual("ep-2", result["starts"][0].episode_id)

    def test_select_and_assemble_produce_a_validated_manifest(self):
        start_rows = []
        for task in TASKS:
            rows = []
            for index in range(70):
                key = "%s-%d" % (task, index)
                rows.append(
                    {
                        "episode_id": "%s-ep-%04d" % (task, index),
                        "language_instruction": verbatim_prompt(task),
                        "image_sha256": digest("image:" + key),
                        "state_sha256": digest("state:" + key),
                        "goal_reference_sha256": digest("goal:" + key),
                        "timestamp": "2026-09-19T00:00:00Z",
                        "objects": [TASK_SCENE_FAMILY[task], "table"],
                    }
                )
            result = driver.select_starts(
                task=task,
                index_rows=rows,
                source_dataset=driver.TASK_SOURCE_DATASET[task],
                source_revision=REVISION,
                counts=driver.cohort_counts(
                    development=5, calibration=10, cost_confirmation=5
                ),
            )
            self.assertEqual("pass", result["status"], result["reason"])
            self.assertEqual([], result["independence_violations"])
            start_rows.extend(item.as_mapping() for item in result["starts"])
            self.assertTrue(all(item["content_hashes_resolved"] for item in result["provenance"]))
        built = driver.assemble(
            manifest_id="scen-v1",
            start_rows=start_rows,
            revisions={AUTO_EVAL_DATASET: REVISION, BRIDGE_LEROBOT_DATASET: REVISION},
            evidence_uris=["file:///evidence/provenance.jsonl"],
            protocol_hash="sha256:" + "b" * 64,
        )
        self.assertEqual((), built["manifest"].validate())
        self.assertEqual("complete", built["manifest"].manifest_status())
        self.assertEqual(350, len(built["gate_c_evidence"]["start_ids"]))
        self.assertEqual("blocked", built["gate_c_evidence"]["status"])

    def test_placeholder_hashes_are_flagged_in_the_start_notes(self):
        result = driver.select_starts(
            task="to_basket",
            index_rows=[{"episode_id": "ep-1", "language_instruction": verbatim_prompt("to_basket")}],
            source_dataset=BRIDGE_LEROBOT_DATASET,
            source_revision=REVISION,
            counts={PRIMARY_COHORT: 1},
        )
        record = result["starts"][0]
        self.assertTrue(any("placeholder content hashes" in note for note in record.notes))
        self.assertFalse(result["provenance"][0]["content_hashes_resolved"])
        self.assertFalse(record.goal_reference_provenance["resolved"])


if __name__ == "__main__":
    unittest.main()
