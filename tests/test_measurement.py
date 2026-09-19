import json
import unittest
from importlib.util import find_spec

from plumb.measurement import (
    analyze,
    bonferroni_endpoint_envelopes,
    exact_mcnemar_pvalue,
    leave_one_task_out_macro,
    mean_maximum_rank_violation,
    outer_lineage_bootstrap_reliability,
    paired_macro_sign_test,
    reference_uncertainty_parametric_bootstrap,
    split_half_reliability,
    task_stratified_lineage_bundle_bootstrap,
    validate_source_lineage_leakage,
)
from plumb.reference import HUMAN, SIMPLER, reference_arithmetic, reference_payload


def episode(policy, task, start_id, lineage, success, validity="valid", status="completed", **extra):
    record = {
        "policy": policy,
        "task": task,
        "start_id": start_id,
        "start_lineage_id": lineage,
        "status": status,
        "binary_success": success,
        "validity": validity,
        "progress_score": 5 if success is True else None,
    }
    record.update(extra)
    return record


class ReferenceTests(unittest.TestCase):
    def test_reference_tables_and_arithmetic_are_preserved(self):
        self.assertEqual(HUMAN["OpenVLA"]["close_drawer"], 46)
        self.assertEqual(SIMPLER["OpenVLA"]["close_drawer"], 2)
        self.assertNotIn("fold_cloth", SIMPLER["OpenVLA"])
        arithmetic = reference_arithmetic()
        self.assertEqual(arithmetic["primary_trials"], 1500)
        self.assertEqual(arithmetic["octo_total"], 2)
        self.assertEqual(arithmetic["minivla_total"], 127)
        self.assertEqual(arithmetic["openpizero_total"], 126)
        self.assertEqual(arithmetic["minivla_openpizero_reversal_count"], 4)
        payload = reference_payload()
        self.assertEqual(payload["policy_aliases"]["OctoSmall"], "Octo")
        json.dumps(payload)


class CellMeasurementTests(unittest.TestCase):
    def test_all_missing_and_missingness_extremes_never_divide_by_zero(self):
        records = [
            episode("OpenVLA", "open_drawer", "a", "la", None, validity="unknown"),
            episode("OpenVLA", "open_drawer", "b", "lb", None, validity="invalid"),
        ]
        result = analyze(
            records,
            {"policies": ["OpenVLA"], "tasks": ["open_drawer"], "planned_per_cell": 2},
        )
        cell = result["cells"][0]
        self.assertEqual(cell["n"], 2)
        self.assertEqual(cell["valid"], 0)
        self.assertEqual(cell["successes"], 0)
        self.assertIsNone(cell["rate"])
        self.assertIsNone(cell["wilson"])
        self.assertEqual(cell["missing_bounds"], [0.0, 1.0])
        self.assertEqual(cell["missing_bounds_detail"], {
            "lower": 0.0,
            "upper": 1.0,
            "width": 1.0,
            "method": "horowitz_manski_no_assumption",
        })

        partial = analyze(
            [
                episode("OpenVLA", "open_drawer", "a", "la", True),
                episode("OpenVLA", "open_drawer", "b", "lb", None, validity="unknown"),
            ],
            {"policies": ["OpenVLA"], "tasks": ["open_drawer"], "planned_per_cell": 3},
        )["cells"][0]
        self.assertEqual((partial["n"], partial["valid"], partial["successes"]), (3, 1, 1))
        self.assertAlmostEqual(partial["coverage"], 1 / 3)
        self.assertEqual(partial["rate"], 1.0)
        self.assertAlmostEqual(partial["positive_rate"], 1 / 3)
        self.assertAlmostEqual(partial["missing_bounds"][0], 1 / 3)
        self.assertEqual(partial["missing_bounds"][1], 1.0)
        self.assertAlmostEqual(partial["missing_bounds_detail"]["width"], 2 / 3)
        self.assertEqual(partial["missing_reason_counts"]["unrecorded_planned"], 1)

    def test_terminal_service_failures_remain_in_denominator(self):
        result = analyze(
            [
                episode("OpenVLA", "open_drawer", "ok", "one", True),
                episode("OpenVLA", "open_drawer", "failed", "two", None, validity="unknown", status="failed"),
            ],
            {"policies": ["OpenVLA"], "tasks": ["open_drawer"]},
        )
        cell = result["cells"][0]
        self.assertEqual((cell["n"], cell["valid"], cell["successes"]), (2, 1, 1))
        self.assertEqual(cell["service_failures"], 1)
        self.assertEqual(cell["missing_reason_counts"]["service_failure"], 1)
        self.assertEqual(cell["positive_rate"], 0.5)


class PairingAndLeakageTests(unittest.TestCase):
    def test_paired_exact_mcnemar_uses_matched_starts(self):
        records = []
        for index in range(3):
            records.append(episode("A", "t", "s%d" % index, "l%d" % index, True))
            records.append(episode("B", "t", "s%d" % index, "l%d" % index, False))
        result = analyze(records, {"policies": ["A", "B"], "tasks": ["t"]})
        pair = next(row for row in result["pairwise"] if row["scope"] == "t")
        self.assertEqual(pair["paired"]["a_only"], 3)
        self.assertEqual(pair["paired"]["b_only"], 0)
        self.assertEqual(pair["paired"]["paired_n"], 3)
        self.assertEqual(pair["paired"]["p_value"], 0.25)
        self.assertEqual(exact_mcnemar_pvalue(3, 0), 0.25)
        self.assertEqual(pair["ordering"], "indeterminate")
        self.assertFalse(pair["supported"])

    def test_source_lineage_cross_cohort_is_rejected(self):
        records = [
            episode("A", "t", "a", "shared", True, cohort="development"),
            episode("A", "t", "b", "shared", False, cohort="primary"),
        ]
        leakage = validate_source_lineage_leakage(records)
        self.assertEqual(leakage["status"], "fail")
        self.assertEqual(leakage["violations"][0]["cohorts"], ["development", "primary"])


class ReliabilityAndMMRVTests(unittest.TestCase):
    def test_constant_split_vectors_are_undefined_not_perfect(self):
        records = []
        for index in range(4):
            for policy in ("A", "B"):
                records.append(episode(policy, "t", "s%d" % index, "l%d" % index, False))
        first = split_half_reliability(records, policies=["A", "B"], tasks=["t"], repetitions=8, seed=7)
        second = split_half_reliability(records, policies=["A", "B"], tasks=["t"], repetitions=8, seed=7)
        self.assertEqual(first["status"], "undefined")
        self.assertEqual(first["undefined_splits"], 8)
        self.assertIsNone(first["mean_correlation"])
        self.assertEqual(first, second)

    def test_mmrv_argument_order_and_ties_match_simpler_definition(self):
        self.assertAlmostEqual(mean_maximum_rank_violation([0.9, 0.5], [0.1, 0.9]), 0.8)
        self.assertAlmostEqual(mean_maximum_rank_violation([0.1, 0.9], [0.9, 0.5]), 0.4)
        # A simulator tie disagrees with only the higher real policy, which
        # contributes its real 0.6 margin; mean over two policies is 0.3.
        self.assertAlmostEqual(mean_maximum_rank_violation([0.5, 0.5], [0.2, 0.8]), 0.3)

    def test_basic_mmrv_is_lightweight_and_permutation_baseline_is_opt_in(self):
        policies = ("OpenVLA", "OpenPiZero", "Octo", "MiniVLA", "SuSIE", "SuSIE_LL")
        tasks = ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")
        records = [
            episode(policy, task, "%s-%s" % (policy, task), "lineage-%s-%s" % (policy, task), True)
            for policy in policies
            for task in tasks
        ]
        basic = analyze(records)
        self.assertEqual(basic["mmrv"]["status"], "computed")
        self.assertEqual(basic["mmrv"]["random_order_baseline"]["status"], "not_computed")
        expanded = analyze(records, {"mmrv": {"random_order_baseline": True}})
        self.assertEqual(expanded["mmrv"]["random_order_baseline"]["macro"]["permutations"], 720)


class AdvancedInferenceTests(unittest.TestCase):
    policies = ("A", "B")
    tasks = ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")

    def _matched_records(self, n=4, clustered=False, all_missing=False):
        records = []
        for task_index, task in enumerate(self.tasks):
            for index in range(n):
                lineage = "%s-lineage-%d" % (task, index // 2 if clustered else index)
                for policy in self.policies:
                    if all_missing:
                        records.append(episode(policy, task, "start-%d" % index, lineage, None, validity="unknown"))
                    else:
                        # Vary complete outcomes so split vectors are not
                        # accidentally constant in the outer-bootstrap test.
                        success = (index + task_index) % 2 == 0 if policy == "A" else (index + task_index) % 3 == 0
                        records.append(episode(policy, task, "start-%d" % index, lineage, success))
        return records

    def test_bootstrap_all_missing_has_no_supported_direction_and_is_explicit(self):
        records = self._matched_records(all_missing=True)
        report = task_stratified_lineage_bundle_bootstrap(
            records, self.policies, self.tasks, bootstrap_replicates=25, seed=3, min_lineage_clusters=2
        )
        self.assertEqual(report["status"], "computed")
        task_pair = next(row for row in report["pairwise"] if row["scope"] == "open_drawer")
        self.assertLessEqual(task_pair["a_over_b"]["lower_confidence_bound"], 0.0)
        self.assertLessEqual(task_pair["b_over_a"]["lower_confidence_bound"], 0.0)

        result = analyze(
            records,
            {
                "policies": self.policies,
                "tasks": self.tasks,
                "advanced_inference": {"enabled": True, "bootstrap_replicates": 25, "min_lineage_clusters": 2},
            },
        )
        pair = next(row for row in result["pairwise"] if row["scope"] == "open_drawer")
        self.assertEqual(result["advanced_inference"]["status"].split("_")[0], "computed")
        self.assertEqual(pair["ordering"], "indeterminate")
        self.assertFalse(pair["supported"])

    def test_shared_start_with_conflicting_lineage_rejects_bootstrap(self):
        records = self._matched_records()
        records[1]["start_lineage_id"] = "conflicting-lineage"
        report = task_stratified_lineage_bundle_bootstrap(
            records, self.policies, self.tasks, bootstrap_replicates=10, min_lineage_clusters=2
        )
        self.assertEqual(report["status"], "unavailable")
        reasons = [error["reason"] for error in report["errors"]]
        self.assertIn("inconsistent_start_lineage", reasons)

    def test_clustered_lineages_stay_indeterminate(self):
        records = self._matched_records(clustered=True)
        report = task_stratified_lineage_bundle_bootstrap(
            records, self.policies, self.tasks, bootstrap_replicates=25, seed=4, min_lineage_clusters=2
        )
        self.assertIn("open_drawer", report["design"]["clustered_tasks"])
        result = analyze(
            records,
            {
                "policies": self.policies,
                "tasks": self.tasks,
                "advanced_inference": {"enabled": True, "bootstrap_replicates": 25, "min_lineage_clusters": 2},
            },
        )
        pair = next(row for row in result["pairwise"] if row["scope"] == "open_drawer")
        self.assertEqual(pair["ordering"], "indeterminate")
        self.assertIn("Clustered", pair["ordering_reason"])

    def test_macro_sign_test_and_optional_exact_envelopes(self):
        records = []
        for task in self.tasks:
            for index in range(4):
                records.append(episode("A", task, "start-%d" % index, "%s-%d" % (task, index), True))
                records.append(episode("B", task, "start-%d" % index, "%s-%d" % (task, index), False))
        sign = paired_macro_sign_test(records, "A", "B")
        self.assertEqual(sign["status"], "computed")
        self.assertEqual((sign["a_only"], sign["b_only"], sign["paired_n"]), (20, 0, 20))
        self.assertTrue(sign["passes_bonferroni_exact_test"])
        envelopes = bonferroni_endpoint_envelopes(records, self.policies, self.tasks, min_lineage_clusters=2)
        if envelopes["status"] == "computed":
            self.assertEqual(envelopes["family_tail_alpha"], 0.05 / 60.0)
            self.assertEqual(len(envelopes["cells"]), 10)
        else:
            self.assertIn("SciPy", envelopes["reason"])

    def test_outer_reliability_bootstrap_is_seeded_and_constant_case_undefined(self):
        records = self._matched_records()
        first = outer_lineage_bootstrap_reliability(
            records, self.policies, self.tasks, outer_replicates=4, inner_repetitions=4, seed=17
        )
        second = outer_lineage_bootstrap_reliability(
            records, self.policies, self.tasks, outer_replicates=4, inner_repetitions=4, seed=17
        )
        self.assertEqual(first, second)
        self.assertIn(first["status"], ("exploratory_low_replicate_count", "computed_with_undefined_outer_replicates"))

        constant = []
        for task in self.tasks:
            for index in range(4):
                for policy in self.policies:
                    constant.append(episode(policy, task, "start-%d" % index, "%s-%d" % (task, index), False))
        undefined = outer_lineage_bootstrap_reliability(
            constant, self.policies, self.tasks, outer_replicates=3, inner_repetitions=3, seed=1
        )
        self.assertEqual(undefined["status"], "undefined")


class AnalysisIdentityAndLiveLedgerTests(unittest.TestCase):
    def test_primary_manifest_rejects_mixed_susie_variants_without_pooling(self):
        records = []
        for variant in ("released-config", "corrected-upstream"):
            for index in range(50):
                records.append(
                    episode(
                        {"name": "SuSIE", "variant": variant},
                        "open_drawer",
                        "%s-start-%d" % (variant, index),
                        "%s-lineage-%d" % (variant, index),
                        variant == "released-config",
                        cohort="primary",
                        protocol_hash="sha256:frozen",
                        episode_id="%s-%d" % (variant, index),
                    )
                )
        result = analyze(
            records,
            {
                "analysis_mode": "primary",
                "policies": ["SuSIE"],
                "tasks": ["open_drawer"],
                "analysis_manifest": {
                    "allowed_episode_ids": [row["episode_id"] for row in records],
                    "identity": {"cohort": "primary", "protocol_hash": "sha256:frozen"},
                    "policy_variants": {"SuSIE": "released-config"},
                },
            },
        )
        self.assertEqual(result["analysis_identity"]["status"], "rejected")
        self.assertEqual(result["summary"]["status"], "provisional_operational_only")
        variant_cells = [cell for cell in result["cells"] if cell["reported_records"]]
        self.assertEqual(sorted(cell["n"] for cell in variant_cells), [50, 50])
        self.assertTrue(all(cell["n"] != 100 for cell in variant_cells))
        self.assertTrue(all(cell["rate"] is None for cell in variant_cells))
        self.assertEqual(result["pairwise"], [])

    def test_primary_manifest_accepts_exact_identity_and_retains_variant_label(self):
        records = [
            episode(
                {"name": "SuSIE", "variant": "released-config"},
                "open_drawer",
                "start-%d" % index,
                "lineage-%d" % index,
                index % 2 == 0,
                cohort="primary",
                protocol_hash="sha256:frozen",
                episode_id="episode-%d" % index,
            )
            for index in range(4)
        ]
        result = analyze(
            records,
            {
                "analysis_mode": "primary",
                "policies": ["SuSIE"],
                "tasks": ["open_drawer"],
                "analysis_manifest": {
                    "allowed_episode_ids": [row["episode_id"] for row in records],
                    "identity": {"cohort": "primary", "protocol_hash": "sha256:frozen"},
                    "policy_variants": {"SuSIE": "released-config"},
                },
            },
        )
        self.assertEqual(result["analysis_identity"]["status"], "accepted")
        cell = result["cells"][0]
        self.assertEqual((cell["policy"], cell["policy_variant"], cell["n"]), ("SuSIE", "released-config", 4))
        self.assertEqual(cell["positive_rate"], 0.5)

    def test_live_ledger_is_operational_only_until_every_row_terminal(self):
        live_records = [
            episode("OpenVLA", "open_drawer", "a", "la", True, episode_id="a"),
            episode("OpenVLA", "open_drawer", "b", "lb", None, validity="unknown", status="planned", episode_id="b"),
        ]
        live = analyze(live_records, {"policies": ["OpenVLA"], "tasks": ["open_drawer"]})
        cell = live["cells"][0]
        self.assertTrue(live["ledger_state"]["active"])
        self.assertEqual(live["summary"]["status"], "provisional_operational_only")
        self.assertEqual(cell["n"], 2)
        self.assertIsNone(cell["rate"])
        self.assertIsNone(cell["positive_rate"])
        self.assertIsNone(cell["wilson"])
        self.assertIsNone(cell["missing_bounds"])
        self.assertEqual(live["pairwise"], [])
        self.assertEqual(live["mmrv"]["status"], "not_computed")

        live_records[1]["status"] = "failed"
        terminal = analyze(live_records, {"policies": ["OpenVLA"], "tasks": ["open_drawer"]})["cells"][0]
        self.assertEqual((terminal["n"], terminal["valid"], terminal["successes"]), (2, 1, 1))
        self.assertEqual(terminal["positive_rate"], 0.5)
        self.assertEqual(terminal["missing_bounds"], [0.5, 1.0])

    def test_cross_task_lineage_is_unavailable_for_split_and_bootstrap(self):
        records = []
        for task in ("t1", "t2"):
            for index in range(2):
                lineage = "shared-across-tasks" if index == 0 else "%s-lineage" % task
                for policy in ("A", "B"):
                    records.append(episode(policy, task, "%s-start-%d" % (task, index), lineage, policy == "A"))
        split = split_half_reliability(records, policies=["A", "B"], tasks=["t1", "t2"], repetitions=3)
        self.assertEqual(split["status"], "unavailable")
        self.assertIn("spans selected tasks", split["reason"])
        bootstrap = task_stratified_lineage_bundle_bootstrap(
            records, ["A", "B"], ["t1", "t2"], bootstrap_replicates=5, min_lineage_clusters=2
        )
        self.assertEqual(bootstrap["status"], "unavailable")
        self.assertIn("source_lineage_crosses_tasks", [error["reason"] for error in bootstrap["errors"]])
        outer = outer_lineage_bootstrap_reliability(
            records, ["A", "B"], ["t1", "t2"], outer_replicates=3, inner_repetitions=3
        )
        self.assertEqual(outer["status"], "unavailable")


class PartialBoundWidthTests(unittest.TestCase):
    """Acceptance check: expected partial-bound widths must match missingness."""

    def test_partial_bound_width_equals_missingness_in_every_cell(self):
        records = []
        # Four cells with deliberately unequal missingness, including V=0 and
        # complete coverage, so one shared width cannot pass by coincidence.
        plans = {
            "complete": [True, True, False, False],
            "one_missing": [True, False, None, False],
            "three_missing": [True, None, None, None],
            "all_missing": [None, None, None, None],
        }
        for task, outcomes in plans.items():
            for index, outcome in enumerate(outcomes):
                records.append(
                    episode(
                        "OpenVLA",
                        task,
                        "%s-start-%d" % (task, index),
                        "%s-lineage-%d" % (task, index),
                        outcome,
                        validity="valid" if outcome is not None else "unknown",
                    )
                )
        result = analyze(records, {"policies": ["OpenVLA"], "tasks": sorted(plans)})
        expected = {"all_missing": 1.0, "complete": 0.0, "one_missing": 0.25, "three_missing": 0.75}
        for cell in result["cells"]:
            bounds = cell["missing_bounds_detail"]
            missing_rate = (cell["n"] - cell["valid"]) / float(cell["n"])
            self.assertAlmostEqual(bounds["width"], expected[cell["task"]], places=12)
            self.assertAlmostEqual(bounds["width"], missing_rate, places=12)
            self.assertAlmostEqual(bounds["width"], cell["missingness_rate"], places=12)
            self.assertAlmostEqual(bounds["upper"] - bounds["lower"], missing_rate, places=12)
            self.assertAlmostEqual(cell["missing_bounds_width"], missing_rate, places=12)
            self.assertEqual(bounds["method"], "horowitz_manski_no_assumption")

    def test_unrecorded_planned_episodes_widen_the_bound_by_exactly_their_share(self):
        cell = analyze(
            [episode("OpenVLA", "open_drawer", "a", "la", True)],
            {"policies": ["OpenVLA"], "tasks": ["open_drawer"], "planned_per_cell": 4},
        )["cells"][0]
        self.assertEqual((cell["n"], cell["valid"]), (4, 1))
        self.assertAlmostEqual(cell["missing_bounds_detail"]["width"], 0.75, places=12)
        self.assertAlmostEqual(
            cell["missing_bounds_detail"]["upper"] - cell["missing_bounds_detail"]["lower"], 0.75, places=12
        )

    def test_summary_partial_bound_width_matches_pooled_missingness(self):
        records = [
            episode("A", "t", "s0", "l0", True),
            episode("A", "t", "s1", "l1", None, validity="invalid"),
            episode("B", "t", "s0", "l0", False),
            episode("B", "t", "s1", "l1", None, validity="unknown"),
        ]
        summary = analyze(records, {"policies": ["A", "B"], "tasks": ["t"]})["summary"]
        bounds = summary["missing_bounds"]
        missing_rate = (summary["n"] - summary["valid"]) / float(summary["n"])
        self.assertAlmostEqual(bounds["width"], 0.5, places=12)
        self.assertAlmostEqual(bounds["width"], missing_rate, places=12)
        self.assertAlmostEqual(bounds["upper"] - bounds["lower"], missing_rate, places=12)


class EndpointEnvelopeTests(unittest.TestCase):
    tasks = ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")

    def _records(self, starts, clustered=False):
        records = []
        for task in self.tasks:
            for index in range(starts):
                lineage = "%s-lineage-%d" % (task, index // 2 if clustered else index)
                records.append(episode("A", task, "%s-start-%d" % (task, index), lineage, True))
                records.append(episode("B", task, "%s-start-%d" % (task, index), lineage, False))
        return records

    @unittest.skipUnless(find_spec("scipy") is not None, "SciPy is required for exact endpoint envelopes")
    def test_independent_start_envelopes_compute_and_match_the_closed_form(self):
        envelopes = bonferroni_endpoint_envelopes(
            self._records(6), ["A", "B"], self.tasks, min_lineage_clusters=2
        )
        self.assertEqual(envelopes["status"], "computed")
        self.assertEqual(envelopes["family_tail_alpha"], 0.05 / 60.0)
        self.assertEqual(len(envelopes["cells"]), 10)
        self.assertTrue(envelopes["design"]["independent_starts"])
        alpha = 0.05 / 60.0
        for cell in envelopes["cells"]:
            self.assertEqual(cell["n"], 6)
            lower = cell["lower_y"]["value"]
            upper = cell["upper_z"]["value"]
            self.assertEqual(cell["lower_y"]["method"], "one_sided_clopper_pearson")
            self.assertTrue(0.0 <= lower <= upper <= 1.0)
            if cell["policy"] == "A":
                # Y=6/6, so Beta(6,1) inverts in closed form to alpha**(1/6).
                self.assertAlmostEqual(lower, alpha ** (1.0 / 6.0), places=10)
                self.assertEqual(upper, 1.0)
            else:
                # Y=0 and Z=0: the lower endpoint is exactly zero and the upper
                # endpoint inverts Beta(1,6) to 1-alpha**(1/6).
                self.assertEqual(lower, 0.0)
                self.assertAlmostEqual(upper, 1.0 - alpha ** (1.0 / 6.0), places=10)

    @unittest.skipUnless(find_spec("scipy") is not None, "SciPy is required for exact endpoint envelopes")
    def test_six_starts_cannot_separate_the_family_but_fifty_starts_can(self):
        small = bonferroni_endpoint_envelopes(self._records(6), ["A", "B"], self.tasks, min_lineage_clusters=2)
        cells = {(row["policy"], row["task"]): row for row in small["cells"]}
        self.assertLess(
            cells[("A", "open_drawer")]["lower_y"]["value"], cells[("B", "open_drawer")]["upper_z"]["value"]
        )
        large = bonferroni_endpoint_envelopes(self._records(50), ["A", "B"], self.tasks, min_lineage_clusters=2)
        self.assertEqual(large["status"], "computed")
        big_cells = {(row["policy"], row["task"]): row for row in large["cells"]}
        self.assertGreater(
            big_cells[("A", "open_drawer")]["lower_y"]["value"], big_cells[("B", "open_drawer")]["upper_z"]["value"]
        )
        result = analyze(
            self._records(50),
            {
                "policies": ["A", "B"],
                "tasks": self.tasks,
                "advanced_inference": {"enabled": True, "bootstrap_replicates": 10000, "min_lineage_clusters": 2},
            },
        )
        pair = next(row for row in result["pairwise"] if row["scope"] == "open_drawer")
        self.assertTrue(pair["advanced_inference"]["endpoint_envelope_guard"]["a_over_b"]["separated"])

    @unittest.skipUnless(find_spec("scipy") is not None, "SciPy is required for exact endpoint envelopes")
    def test_the_sampling_envelope_is_not_the_partial_bound_width(self):
        # The Horowitz-Manski point-bound width is exactly the missing-outcome
        # rate, but a sampling confidence region need not have that width; the
        # two must never be conflated.
        records = self._records(6)
        result = analyze(records, {"policies": ["A", "B"], "tasks": self.tasks})
        cell = next(row for row in result["cells"] if row["policy"] == "A" and row["task"] == "open_drawer")
        self.assertEqual(cell["missingness_rate"], 0.0)
        self.assertEqual(cell["missing_bounds_detail"]["width"], 0.0)
        envelope = next(
            row
            for row in bonferroni_endpoint_envelopes(records, ["A", "B"], self.tasks, min_lineage_clusters=2)["cells"]
            if row["policy"] == "A" and row["task"] == "open_drawer"
        )
        envelope_width = envelope["upper_z"]["value"] - envelope["lower_y"]["value"]
        self.assertGreater(envelope_width, cell["missing_bounds_detail"]["width"])

    def test_clustered_lineage_envelopes_stay_indeterminate_rather_than_nominal(self):
        envelopes = bonferroni_endpoint_envelopes(
            self._records(6, clustered=True), ["A", "B"], self.tasks, min_lineage_clusters=2
        )
        self.assertEqual(envelopes["status"], "unavailable")
        self.assertIn("independent starts", envelopes["reason"])
        self.assertIn("indeterminate", envelopes["reason"])
        self.assertNotIn("cells", envelopes)
        result = analyze(
            self._records(6, clustered=True),
            {
                "policies": ["A", "B"],
                "tasks": self.tasks,
                "advanced_inference": {"enabled": True, "bootstrap_replicates": 10000, "min_lineage_clusters": 2},
            },
        )
        self.assertEqual(result["advanced_inference"]["endpoint_envelopes"]["status"], "unavailable")
        for row in result["pairwise"]:
            self.assertEqual(row["ordering"], "indeterminate")
            self.assertFalse(row["supported"])
        pair = next(row for row in result["pairwise"] if row["scope"] == "open_drawer")
        self.assertEqual(
            pair["advanced_inference"]["endpoint_envelope_guard"]["status"], "unavailable"
        )


class ReferenceUncertaintyTests(unittest.TestCase):
    policies = ("OpenVLA", "OpenPiZero", "Octo", "MiniVLA", "SuSIE", "SuSIE_LL")
    tasks = ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")

    def _records(self):
        return [
            episode(
                policy,
                task,
                "%s-%s-%d" % (policy, task, index),
                "lineage-%s-%s-%d" % (policy, task, index),
                (index + len(policy)) % 3 == 0,
            )
            for policy in self.policies
            for task in self.tasks
            for index in range(4)
        ]

    def _bootstrap(self, replicates=120):
        return analyze(
            self._records(),
            {"reference_uncertainty": {"enabled": True, "replicates": replicates, "seed": 11}},
        )["reference_uncertainty"]

    def test_the_bootstrap_discloses_its_model_and_never_pairs_to_human_trials(self):
        report = self._bootstrap()
        self.assertEqual(report["status"], "computed")
        disclosure = report["model_disclosure"]
        self.assertEqual(disclosure["reference_n_per_cell"], 50)
        self.assertEqual(disclosure["draw"], "S*_cell ~ Binomial(n=50, p_hat_cell)")
        self.assertIn("binomial", disclosure["model"])
        self.assertFalse(disclosure["paired_with_human_trials"])
        self.assertIn("never paired with human trials", disclosure["pairing_note"])
        self.assertIn("trial-level pairing between generated episodes and human trials", disclosure["not_modelled"])
        self.assertEqual(report["method"], "disclosed_binomial_model_parametric_bootstrap_of_published_reference")

    def test_conditional_and_reference_uncertainty_results_stay_distinct(self):
        report = self._bootstrap()
        conditional = report["conditional_on_reference"]
        sensitivity = report["reference_uncertainty_sensitivity"]
        self.assertNotIn("interval", conditional)
        published_macro = sum(HUMAN["MiniVLA"].values()) / (5.0 * 50.0)
        self.assertAlmostEqual(conditional["macro_reference_rate"]["MiniVLA"], published_macro, places=12)
        interval = sensitivity["macro_reference_rate/MiniVLA"]["interval"]
        self.assertEqual(interval["method"], "outer_lineage_bootstrap_percentile")
        self.assertLessEqual(interval["lower"], published_macro)
        self.assertGreaterEqual(interval["upper"], published_macro)
        self.assertEqual(sensitivity["macro_reference_rate/MiniVLA"]["replicates"], 120)
        self.assertIn("mmrv_macro", conditional)
        self.assertIn("mmrv_macro", sensitivity)

    def test_degenerate_plug_in_cells_are_disclosed_not_hidden(self):
        report = self._bootstrap()
        # Octo and SuSIE_LL have published zeros, so their plug-in p is exactly 0.
        self.assertIn("Octo/open_drawer", report["degenerate_plug_in_cells"])
        self.assertIn("SuSIE_LL/fold_cloth", report["degenerate_plug_in_cells"])
        self.assertIn("understates reference uncertainty", report["degenerate_plug_in_note"])
        zeros = report["reference_uncertainty_sensitivity"]["macro_reference_rate/SuSIE_LL"]
        self.assertEqual((zeros["minimum"], zeros["maximum"]), (0.0, 0.0))

    def test_the_bootstrap_is_deterministic_and_opt_in(self):
        self.assertEqual(self._bootstrap(), self._bootstrap())
        default = analyze(self._records())["reference_uncertainty"]
        self.assertEqual(default["status"], "not_computed")
        self.assertIn("reference_uncertainty.enabled=true", default["reason"])
        method = next(
            row
            for row in analyze(self._records())["unimplemented_advanced_inference"]["methods"]
            if row["name"] == "reference_uncertainty_parametric_bootstrap"
        )
        self.assertEqual(method["status"], "not_computed")
        self.assertIn("unavailable", method["trial_level_pairing"])
        enabled = next(
            row
            for row in analyze(
                self._records(), {"reference_uncertainty": {"enabled": True, "replicates": 20}}
            )["unimplemented_advanced_inference"]["methods"]
            if row["name"] == "reference_uncertainty_parametric_bootstrap"
        )
        self.assertEqual(enabled["status"], "computed")

    def test_a_missing_published_cell_makes_the_bootstrap_unavailable(self):
        records = [
            episode("UnknownPolicy", "open_drawer", "s%d" % index, "l%d" % index, True) for index in range(3)
        ]
        report = analyze(
            records,
            {
                "policies": ["UnknownPolicy"],
                "tasks": ["open_drawer"],
                "reference_uncertainty": {"enabled": True, "replicates": 20},
            },
        )["reference_uncertainty"]
        self.assertEqual(report["status"], "unavailable")
        self.assertIn("published reference numerator", report["reason"])
        self.assertFalse(report["model_disclosure"]["paired_with_human_trials"])

    def test_the_bootstrap_rejects_impossible_arguments(self):
        with self.assertRaises(ValueError):
            reference_uncertainty_parametric_bootstrap({}, ["OpenVLA"], ["open_drawer"], replicates=0)
        with self.assertRaises(ValueError):
            reference_uncertainty_parametric_bootstrap({}, ["OpenVLA"], ["open_drawer"], confidence=1.0)


class LeaveOneTaskOutTests(unittest.TestCase):
    def _records(self):
        # A wins overall only because of t1; dropping t1 reverses the ordering.
        successes = {"A": {"t1": 4, "t2": 0, "t3": 0}, "B": {"t1": 1, "t2": 1, "t3": 1}}
        records = []
        for policy, by_task in successes.items():
            for task, wins in by_task.items():
                for index in range(4):
                    records.append(
                        episode(policy, task, "%s-start-%d" % (task, index), "%s-lineage-%d" % (task, index), index < wins)
                    )
        return records

    def test_leave_one_task_out_is_reported_separately_for_plumb_rates(self):
        result = analyze(self._records(), {"policies": ["A", "B"], "tasks": ["t1", "t2", "t3"]})
        loto = result["leave_one_task_out"]
        self.assertEqual(loto["status"], "computed")
        self.assertEqual(loto["method"], "equal_weight_task_macro_leave_one_task_out")
        self.assertEqual(loto["endpoint"], "observed_positive_lower_bound")
        self.assertAlmostEqual(loto["full"]["macro_observed_positive_rate"]["A"], 1.0 / 3.0)
        self.assertAlmostEqual(loto["full"]["macro_observed_positive_rate"]["B"], 0.25)
        self.assertEqual(loto["full"]["visual_order"], ["A", "B"])
        # Every number carries its n.
        self.assertEqual(loto["full"]["n"]["A"], 12)
        self.assertEqual(loto["full"]["task_count"], 3)
        self.assertEqual(len(loto["leave_one_out"]), 3)
        self.assertIn("not a random sample", loto["note"])

    def test_a_dropped_task_that_reverses_an_order_is_surfaced(self):
        loto = analyze(self._records(), {"policies": ["A", "B"], "tasks": ["t1", "t2", "t3"]})["leave_one_task_out"]
        rows = {row["dropped_task"]: row for row in loto["leave_one_out"]}
        dropped_t1 = rows["t1"]
        self.assertEqual(dropped_t1["task_count"], 2)
        self.assertEqual(dropped_t1["n"]["A"], 8)
        self.assertEqual(dropped_t1["macro_observed_positive_rate"]["A"], 0.0)
        self.assertEqual(dropped_t1["macro_observed_positive_rate"]["B"], 0.25)
        self.assertEqual(dropped_t1["pairwise_order_change_count"], 1)
        self.assertEqual(dropped_t1["pairwise_order_changes"][0]["full_relation"], 1)
        self.assertEqual(dropped_t1["pairwise_order_changes"][0]["leave_one_out_relation"], -1)
        self.assertTrue(dropped_t1["visual_order_changed"])
        self.assertAlmostEqual(dropped_t1["maximum_absolute_macro_shift"], 1.0 / 3.0)
        self.assertEqual(rows["t2"]["pairwise_order_change_count"], 0)
        self.assertFalse(rows["t2"]["visual_order_changed"])
        self.assertEqual(loto["total_pairwise_order_changes"], 1)

    def test_leave_one_task_out_needs_two_tasks_and_complete_cells(self):
        single = analyze(
            [episode("A", "t1", "s0", "l0", True)], {"policies": ["A"], "tasks": ["t1"]}
        )["leave_one_task_out"]
        self.assertEqual(single["status"], "unavailable")
        self.assertIn("at least two tasks", single["reason"])
        missing = leave_one_task_out_macro({}, ["A"], ["t1", "t2"])
        self.assertEqual(missing["status"], "unavailable")
        self.assertEqual(missing["missing_cells"], ["A/t1", "A/t2"])

    def test_a_cell_with_no_conditional_rate_leaves_the_conditional_macro_null(self):
        records = [
            episode("A", "t1", "s0", "l0", True),
            episode("A", "t2", "s1", "l1", None, validity="unknown"),
        ]
        loto = analyze(records, {"policies": ["A"], "tasks": ["t1", "t2"]})["leave_one_task_out"]
        self.assertIsNone(loto["full"]["macro_conditional_rate"]["A"])
        self.assertAlmostEqual(loto["full"]["macro_observed_positive_rate"]["A"], 0.5)
        self.assertEqual(loto["full"]["evaluable"]["A"], 1)


class SweepAndDriftWiringTests(unittest.TestCase):
    horizons = {"open_drawer": 70, "close_drawer": 70, "to_basket": 100, "to_sink": 100, "fold_cloth": 80}

    def _records(self):
        return [episode("OpenVLA", "open_drawer", "s%d" % index, "l%d" % index, index % 2 == 0) for index in range(4)]

    def _cost_artifacts(self):
        policies = ("OpenVLA", "OpenPiZero", "Octo", "MiniVLA", "SuSIE", "SuSIE_LL")
        successes = {"OpenVLA": 8, "OpenPiZero": 6, "Octo": 1, "MiniVLA": 7, "SuSIE": 3, "SuSIE_LL": 0}
        outcomes = {
            "%s/%s" % (policy, task): {
                "start-%03d" % index: index < successes[policy] for index in range(10)
            }
            for policy in policies
            for task in self.horizons
        }
        operating_point = {
            "resolution_tier": 480,
            "denoising_steps": 20,
            "chunk_action_length": 16,
            "batch_size": 1,
            "task_horizon_actions": self.horizons,
            "resolution_validation_ref": "preflight-1",
            "seed_set_id": "seeds-1",
            "judge_revision_id": "judge-1",
        }
        return {
            "tolerances": {
                "protocol_id": "cf-1",
                "maximum_absolute_cell_error": 0.1,
                "minimum_coverage": 0.9,
                "minimum_pairwise_order_agreement": 1.0,
                "required_task_horizon_actions": self.horizons,
            },
            "full_fidelity_reference": {
                "operating_point": dict(operating_point, operating_point_id="full", resolution_tier=704, denoising_steps=30),
                "outcomes": outcomes,
            },
            "points": [
                {
                    "operating_point": dict(
                        operating_point,
                        operating_point_id="cheap",
                        estimated_usd=9.0,
                        cost_basis_ref="snapshot-1",
                        measured_gpu_seconds=900.0,
                    ),
                    "outcomes": outcomes,
                }
            ],
        }

    def _drift_artifacts(self):
        return {
            "backend_profile_id": "cosmos-256-profile",
            "tolerances": {
                "protocol_id": "drift-1",
                "maximum_mean_action_error": 0.1,
                "action_error_metric": "mean_per_tick_l2",
            },
            "points": [
                {
                    "backend_profile_id": "cosmos-256-profile",
                    "control_ticks": ticks,
                    "fd_request_action_length": 16,
                    "autoregressive_boundaries": list(range(16, ticks + 1, 16)),
                    "policy_feedback_boundaries": list(range(16, ticks + 1, 16)),
                    "mean_action_error": error,
                    "action_error_metric": "mean_per_tick_l2",
                }
                for ticks, error in ((32, 0.01), (64, 0.9))
            ],
        }

    def test_sweep_keys_are_not_computed_without_supplied_artifacts(self):
        result = analyze(self._records(), {"policies": ["OpenVLA"], "tasks": ["open_drawer"]})
        for key in ("cost_fidelity", "drift"):
            self.assertEqual(result[key]["status"], "not_computed", key)
            self.assertIn("protocol.%s" % key, result[key]["reason"])

    def test_an_enabled_flag_without_artifacts_is_unavailable_not_an_empty_measurement(self):
        result = analyze(
            self._records(),
            {"policies": ["OpenVLA"], "tasks": ["open_drawer"], "cost_fidelity": {"enabled": True}, "drift": {"enabled": True}},
        )
        for key in ("cost_fidelity", "drift"):
            self.assertEqual(result[key]["status"], "unavailable", key)
            self.assertIn("empty sweep is not a measurement", result[key]["reason"])
        disabled = analyze(
            self._records(), {"policies": ["OpenVLA"], "tasks": ["open_drawer"], "drift": {"enabled": False}}
        )
        self.assertEqual(disabled["drift"]["status"], "not_computed")

    def test_real_artifacts_produce_gated_unconfirmed_results(self):
        result = analyze(
            self._records(),
            {
                "policies": ["OpenVLA"],
                "tasks": ["open_drawer"],
                "cost_fidelity": self._cost_artifacts(),
                "drift": self._drift_artifacts(),
            },
        )
        cost = result["cost_fidelity"]
        self.assertEqual(cost["status"], "computed")
        self.assertEqual(cost["selection"]["operating_point_id"], "cheap")
        self.assertEqual(cost["selection"]["confirmation_status"], "unconfirmed")
        self.assertFalse(cost["selection"]["qualified_claim_allowed"])
        self.assertEqual(cost["qualified_claim"]["status"], "blocked")
        drift = result["drift"]
        self.assertEqual(drift["status"], "computed")
        self.assertEqual(drift["axis"]["x_axis"], "control_ticks")
        self.assertEqual(drift["horizon_tolerance"]["last_horizon_meeting_tolerance"], 32)
        self.assertFalse(drift["horizon_tolerance"]["unlimited_guarantee"])
        json.dumps(result, allow_nan=False)

    def test_malformed_sweep_artifacts_are_rejected_without_raising(self):
        result = analyze(
            self._records(),
            {
                "policies": ["OpenVLA"],
                "tasks": ["open_drawer"],
                "cost_fidelity": {"points": [{"operating_point": {}, "outcomes": {}}]},
                "drift": {"points": [{"backend_profile_id": "b"}]},
            },
        )
        self.assertEqual(result["cost_fidelity"]["status"], "rejected")
        self.assertEqual(result["drift"]["status"], "rejected")
        self.assertIsInstance(result["drift"]["reason"], str)

    def test_a_nonterminal_ledger_withholds_every_new_measurement(self):
        records = self._records()
        records.append(
            episode("OpenVLA", "open_drawer", "live", "live-lineage", None, validity="unknown", status="running")
        )
        result = analyze(
            records,
            {
                "policies": ["OpenVLA"],
                "tasks": ["open_drawer"],
                "cost_fidelity": self._cost_artifacts(),
                "drift": self._drift_artifacts(),
                "reference_uncertainty": {"enabled": True, "replicates": 20},
            },
        )
        for key in ("cost_fidelity", "drift", "leave_one_task_out", "reference_uncertainty"):
            self.assertEqual(result[key]["status"], "not_computed", key)
            self.assertIn("nonterminal", result[key]["reason"])


if __name__ == "__main__":
    unittest.main()
