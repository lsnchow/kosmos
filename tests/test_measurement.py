import json
import unittest

from plumb.measurement import (
    analyze,
    bonferroni_endpoint_envelopes,
    exact_mcnemar_pvalue,
    mean_maximum_rank_violation,
    outer_lineage_bootstrap_reliability,
    paired_macro_sign_test,
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


if __name__ == "__main__":
    unittest.main()
