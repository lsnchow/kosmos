import json
import unittest

from plumb.sweeps import (
    CONFIRMATION_PANEL,
    AllocationInterval,
    CostFidelitySweep,
    DriftPoint,
    DriftStudy,
    DriftTolerances,
    GpuSecondModel,
    OtherCharge,
    PriceBasis,
    ResourceRate,
    SweepError,
    SweepPoint,
    SweepTolerances,
    build_cost_fidelity_report,
    build_drift_report,
)


HORIZONS = {"open_drawer": 70, "close_drawer": 70, "to_basket": 100, "to_sink": 100, "fold_cloth": 80}
POLICIES = ("OpenVLA", "OpenPiZero", "Octo", "MiniVLA", "SuSIE", "SuSIE_LL")
TASKS = tuple(HORIZONS)
BASE_SUCCESSES = {"OpenVLA": 8, "OpenPiZero": 6, "Octo": 1, "MiniVLA": 7, "SuSIE": 3, "SuSIE_LL": 0}
STARTS = 10


def outcome_vectors(successes_by_policy, missing=0, starts=STARTS, prefix="start"):
    """Per-cell start-keyed vectors; ``None`` is an unevaluable planned start."""

    vectors = {}
    for policy in POLICIES:
        for task in TASKS:
            cell = {}
            for index in range(starts):
                key = "%s-%03d" % (prefix, index)
                cell[key] = None if index < missing else index < successes_by_policy[policy]
            vectors["%s/%s" % (policy, task)] = cell
    return vectors


def lineages(prefix="start", lineage_prefix="lineage", starts=STARTS):
    return {"%s-%03d" % (prefix, index): "%s-%03d" % (lineage_prefix, index) for index in range(starts)}


def tolerances(**overrides):
    values = {
        "protocol_id": "cost-fidelity-protocol-1",
        "maximum_absolute_cell_error": 0.1,
        "minimum_coverage": 0.9,
        "minimum_pairwise_order_agreement": 1.0,
        "required_task_horizon_actions": HORIZONS,
    }
    values.update(overrides)
    return SweepTolerances(**values)


def point(operating_point_id, usd=None, gpu_seconds=None, **overrides):
    values = {
        "operating_point_id": operating_point_id,
        "resolution_tier": 480,
        "denoising_steps": 20,
        "chunk_action_length": 16,
        "batch_size": 1,
        "task_horizon_actions": HORIZONS,
        "resolution_validation_ref": "worldadapter-preflight-1",
        "seed_set_id": "seeds-frozen-1",
        "judge_revision_id": "judge-rev-1",
        "estimated_usd": usd,
        "cost_basis_ref": "price-snapshot-1" if usd is not None else None,
        "measured_gpu_seconds": gpu_seconds,
    }
    values.update(overrides)
    return SweepPoint(**values)


def verified_basis(**overrides):
    values = {
        "basis_id": "price-snapshot-1",
        "currency": "USD",
        "billing_unit": "per_hour",
        "applicable_resource_types": ("H100",),
        "tax_treatment": "excluded from the estimate",
        "egress_treatment": "billed separately",
        "storage_treatment": "billed separately",
        "retrieved_at": "2026-09-19T00:00:00Z",
        "reconciliation_method": "monthly invoice reconciliation",
        "units_confirmed": True,
    }
    values.update(overrides)
    return PriceBasis(**values)


def run_intervals():
    return [
        AllocationInterval(
            interval_id="prewarm-1", resource_id="replica-1", resource_type="H100", stage="prewarm",
            start_seconds=0.0, end_seconds=3600.0,
        ),
        AllocationInterval(
            interval_id="execution-1", resource_id="replica-1", resource_type="H100", stage="execution",
            start_seconds=3600.0, end_seconds=7200.0, attributed_to_marginal_execution=True,
        ),
        AllocationInterval(
            interval_id="cooldown-1", resource_id="replica-1", resource_type="H100", stage="cooldown",
            start_seconds=7200.0, end_seconds=10800.0,
        ),
    ]


class PriceBasisAndUsdTests(unittest.TestCase):
    def test_usd_is_unavailable_until_the_price_basis_confirms_its_units(self):
        model = GpuSecondModel(
            verified_basis(units_confirmed=False),
            [ResourceRate("H100", 2.18, rate_unit="per_hour", currency="USD")],
            run_intervals(),
            gpu_resource_types=("H100",),
        )
        view = model.total_demonstration_run_cost()
        self.assertIsNone(view["estimated_usd"])
        self.assertEqual(view["usd_status"], "unavailable")
        self.assertIn("price_units_unconfirmed", [row["reason"] for row in view["usd_unavailable_reasons"]])
        # The measurable allocation is still reported; only the money is withheld.
        self.assertAlmostEqual(view["allocated_resource_hours"], 3.0)
        self.assertAlmostEqual(view["allocated_gpu_seconds"], 10800.0)

    def test_a_field_named_price_is_not_dollars_per_hour(self):
        for unit in ("price", "per_instance_hour", "credits"):
            model = GpuSecondModel(
                verified_basis(),
                [ResourceRate("H100", 2.18, rate_unit=unit, currency="USD")],
                run_intervals(),
            )
            view = model.total_demonstration_run_cost()
            self.assertIsNone(view["estimated_usd"], unit)
            self.assertIn("unknown_billing_unit", [row["reason"] for row in view["usd_unavailable_reasons"]])

    def test_foreign_currency_and_uncovered_resources_never_become_usd(self):
        foreign = GpuSecondModel(
            verified_basis(currency="EUR"),
            [ResourceRate("H100", 2.0, rate_unit="per_hour", currency="EUR")],
            run_intervals(),
        )
        self.assertIsNone(foreign.total_demonstration_run_cost()["estimated_usd"])
        self.assertIn(
            "non_usd_price_basis_without_confirmed_conversion",
            [row["reason"] for row in foreign.total_demonstration_run_cost()["usd_unavailable_reasons"]],
        )
        uncovered = GpuSecondModel(
            verified_basis(applicable_resource_types=("A100",)),
            [ResourceRate("H100", 2.0, rate_unit="per_hour", currency="USD")],
            run_intervals(),
        )
        self.assertIn(
            "resource_not_covered_by_price_basis",
            [row["reason"] for row in uncovered.total_demonstration_run_cost()["usd_unavailable_reasons"]],
        )
        mismatched = GpuSecondModel(
            verified_basis(),
            [ResourceRate("H100", 2.0, rate_unit="per_hour", currency="unknown")],
            run_intervals(),
        )
        self.assertIn(
            "rate_currency_mismatch",
            [row["reason"] for row in mismatched.total_demonstration_run_cost()["usd_unavailable_reasons"]],
        )

    def test_missing_rate_and_unknown_charge_block_usd_separately(self):
        no_rate = GpuSecondModel(verified_basis(), [], run_intervals())
        self.assertIn(
            "missing_resource_rate", [row["reason"] for row in no_rate.total_demonstration_run_cost()["usd_unavailable_reasons"]]
        )
        unknown_charge = GpuSecondModel(
            verified_basis(),
            [ResourceRate("H100", 1.0, rate_unit="per_hour", currency="USD")],
            run_intervals(),
            other_charges=[OtherCharge("egress-1", "network_egress", None)],
        )
        view = unknown_charge.total_demonstration_run_cost()
        self.assertIsNone(view["estimated_usd"])
        self.assertIn("other_charge_amount_unknown", [row["reason"] for row in view["usd_unavailable_reasons"]])
        # A known charge is added to the rate-based component, not substituted.
        known = GpuSecondModel(
            verified_basis(),
            [ResourceRate("H100", 1.0, rate_unit="per_hour", currency="USD")],
            run_intervals(),
            other_charges=[OtherCharge("storage-1", "storage", 0.25)],
        )
        self.assertAlmostEqual(known.total_demonstration_run_cost()["estimated_usd"], 3.25)


class AllocationLedgerTests(unittest.TestCase):
    def test_overlapping_intervals_on_one_lane_are_rejected_not_billed_twice(self):
        intervals = run_intervals() + [
            AllocationInterval(
                interval_id="retry-1", resource_id="replica-1", resource_type="H100", stage="execution",
                start_seconds=4000.0, end_seconds=5000.0, attributed_to_marginal_execution=True,
            )
        ]
        model = GpuSecondModel(
            verified_basis(), [ResourceRate("H100", 1.0, rate_unit="per_hour", currency="USD")], intervals
        )
        validation = model.validate()
        self.assertEqual(validation["status"], "rejected")
        self.assertEqual([error["reason"] for error in validation["errors"]], ["overlapping_allocation_intervals"])
        self.assertIsNone(model.total_demonstration_run_cost()["estimated_usd"])

    def test_duplicate_interval_ids_are_rejected(self):
        intervals = run_intervals()
        intervals.append(
            AllocationInterval(
                interval_id="prewarm-1", resource_id="replica-2", resource_type="H100", stage="prewarm",
                start_seconds=0.0, end_seconds=3600.0,
            )
        )
        model = GpuSecondModel(verified_basis(), [], intervals)
        self.assertIn("duplicate_allocation_interval_ids", [error["reason"] for error in model.validate()["errors"]])

    def test_concurrent_distinct_lanes_are_counted_once_each(self):
        intervals = [
            AllocationInterval(
                interval_id="replica-%d" % index, resource_id="replica-%d" % index, resource_type="H100",
                stage="execution", start_seconds=0.0, end_seconds=3600.0, attributed_to_marginal_execution=True,
            )
            for index in range(3)
        ]
        model = GpuSecondModel(
            verified_basis(), [ResourceRate("H100", 2.0, rate_unit="per_hour", currency="USD")], intervals,
            gpu_resource_types=("H100",),
        )
        view = model.marginal_execution_estimate()
        self.assertEqual(model.validate()["status"], "accepted")
        self.assertAlmostEqual(view["allocated_resource_hours"], 3.0)
        self.assertAlmostEqual(view["estimated_usd"], 6.0)

    def test_stage_hours_reconcile_to_the_view_total_across_prewarm_and_cooldown(self):
        model = GpuSecondModel(
            verified_basis(), [ResourceRate("H100", 1.0, rate_unit="per_hour", currency="USD")], run_intervals()
        )
        total = model.total_demonstration_run_cost()
        self.assertEqual(total["stage_coverage_status"], "complete")
        self.assertEqual(sorted(total["allocated_resource_hours_by_stage"]), ["cooldown", "execution", "prewarm"])
        self.assertAlmostEqual(
            sum(total["allocated_resource_hours_by_stage"].values()), total["allocated_resource_hours"]
        )
        execution_only = GpuSecondModel(
            verified_basis(),
            [ResourceRate("H100", 1.0, rate_unit="per_hour", currency="USD")],
            [run_intervals()[1]],
        )
        incomplete = execution_only.total_demonstration_run_cost()
        self.assertEqual(incomplete["stage_coverage_status"], "incomplete")
        self.assertEqual(incomplete["missing_required_stages"], ["prewarm", "cooldown"])

    def test_the_two_views_stay_distinct_and_the_marginal_one_omits_idle_capacity(self):
        intervals = run_intervals() + [
            AllocationInterval(
                interval_id="idle-1", resource_id="replica-2", resource_type="H100", stage="idle",
                start_seconds=0.0, end_seconds=10800.0,
            )
        ]
        model = GpuSecondModel(
            verified_basis(), [ResourceRate("H100", 1.0, rate_unit="per_hour", currency="USD")], intervals
        )
        marginal = model.marginal_execution_estimate()
        total = model.total_demonstration_run_cost()
        self.assertAlmostEqual(marginal["estimated_usd"], 1.0)
        self.assertAlmostEqual(total["estimated_usd"], 6.0)
        self.assertIn("may omit idle capacity", marginal["label"])
        self.assertNotIn("idle", marginal["allocated_resource_hours_by_stage"])
        self.assertIn("idle", total["allocated_resource_hours_by_stage"])
        self.assertEqual(marginal["interval_ids"], ["execution-1"])

    def test_shared_capacity_attribution_scales_hours_but_never_duplicates_them(self):
        shared = AllocationInterval(
            interval_id="shared-1", resource_id="replica-9", resource_type="H100", stage="execution",
            start_seconds=0.0, end_seconds=3600.0, shared_capacity_attribution_fraction=0.25,
            attribution_method="even split across four runs", attributed_to_marginal_execution=True,
        )
        model = GpuSecondModel(
            verified_basis(), [ResourceRate("H100", 4.0, rate_unit="per_hour", currency="USD")], [shared]
        )
        self.assertAlmostEqual(model.marginal_execution_estimate()["allocated_resource_hours"], 0.25)
        self.assertAlmostEqual(model.marginal_execution_estimate()["estimated_usd"], 1.0)

    def test_degenerate_and_negative_allocations_are_refused_at_construction(self):
        with self.assertRaises(SweepError):
            AllocationInterval(
                interval_id="zero", resource_id="r", resource_type="H100", stage="execution",
                start_seconds=10.0, end_seconds=10.0,
            )
        with self.assertRaises(SweepError):
            AllocationInterval(
                interval_id="backwards", resource_id="r", resource_type="H100", stage="execution",
                start_seconds=20.0, end_seconds=10.0,
            )
        with self.assertRaises(SweepError):
            AllocationInterval(
                interval_id="negative", resource_id="r", resource_type="H100", stage="execution",
                start_seconds=-1.0, end_seconds=10.0,
            )
        with self.assertRaises(SweepError):
            AllocationInterval(
                interval_id="zero-share", resource_id="r", resource_type="H100", stage="execution",
                start_seconds=0.0, end_seconds=10.0, shared_capacity_attribution_fraction=0.0,
            )

    def test_gpu_seconds_and_measured_seconds_are_labelled_separately(self):
        undeclared = GpuSecondModel(verified_basis(), [], run_intervals())
        view = undeclared.total_demonstration_run_cost()
        self.assertIsNone(view["allocated_gpu_seconds"])
        self.assertEqual(view["allocated_gpu_seconds_status"], "unavailable")
        self.assertIsNone(view["measured_compute_gpu_seconds"])
        instrumented = GpuSecondModel(
            verified_basis(), [], run_intervals(), gpu_resource_types=("H100",), measured_compute_gpu_seconds=9000.0
        )
        instrumented_view = instrumented.total_demonstration_run_cost()
        self.assertAlmostEqual(instrumented_view["allocated_gpu_seconds"], 10800.0)
        self.assertAlmostEqual(instrumented_view["measured_compute_gpu_seconds"], 9000.0)
        self.assertNotEqual(
            instrumented_view["telemetry_labels"]["allocated_gpu_seconds"],
            instrumented_view["telemetry_labels"]["measured_compute_gpu_seconds"],
        )

    def test_prior_and_settled_costs_are_separate_and_never_zero(self):
        model = GpuSecondModel(
            verified_basis(),
            [ResourceRate("H100", 1.0, rate_unit="per_hour", currency="USD")],
            run_intervals(),
            prior_cost_reports={"judge_distillation": None, "development_sweeps": 12.5},
        )
        prior = model.prior_costs()
        self.assertEqual(prior["status"], "reported")
        self.assertIsNone(prior["entries"]["judge_distillation"])
        self.assertEqual(model.settled_cost()["status"], "unavailable")
        self.assertIsNone(model.settled_cost()["settled_usd"])
        with self.assertRaises(SweepError):
            GpuSecondModel(verified_basis(), [], run_intervals(), settled_cost_usd=11.0)
        json.dumps(model.as_dict(), allow_nan=False)


class SweepPointContractTests(unittest.TestCase):
    def test_a_usd_amount_requires_a_recorded_pricing_snapshot(self):
        with self.assertRaises(SweepError):
            point("no-basis", usd=3.0, cost_basis_ref=None)

    def test_a_point_cannot_claim_success_while_naming_failing_tolerances(self):
        with self.assertRaises(SweepError):
            point("forged", tolerances_met=True, failing_tolerances=("coverage",))

    def test_horizon_and_coverage_travel_with_every_cost_point(self):
        row = point("p", usd=5.0, coverage=0.98).as_dict()
        # The five benchmark horizons differ, so the slider-facing horizon field
        # must expose the whole mapping rather than collapse to one number.
        self.assertEqual(row["horizon"], dict(sorted(HORIZONS.items())))
        self.assertEqual(row["task_horizon_actions"], dict(sorted(HORIZONS.items())))
        single = point("single-task", usd=5.0, task_horizon_actions={"fold_cloth": 80}).as_dict()
        self.assertEqual(single["horizon"], 80)
        self.assertEqual(row["coverage"], 0.98)
        self.assertFalse(row["qualified"])
        self.assertEqual(row["chunk_partition"]["to_basket"], 7)
        self.assertEqual(row["num_inference_steps"], row["denoising_steps"])
        self.assertIn("shorter task is not a cheaper equivalent evaluation", row["horizon_note"])

    def test_unevaluated_points_report_unknown_rather_than_passing(self):
        candidate = point("unevaluated")
        self.assertIsNone(candidate.tolerances_met)
        self.assertEqual(candidate.usd_status, "unavailable")
        self.assertIsNone(candidate.coverage)


class CostFidelitySelectionTests(unittest.TestCase):
    def _sweep(self, **overrides):
        return CostFidelitySweep(
            point("full-fidelity-704-30", usd=40.0, gpu_seconds=4000.0, resolution_tier=704, denoising_steps=30),
            outcome_vectors(BASE_SUCCESSES),
            tolerances(**overrides),
            lineages=lineages(),
        )

    def test_a_cheaper_point_that_fails_a_tolerance_is_not_selected(self):
        sweep = self._sweep()
        sweep.add_point(point("faithful-480-20", usd=12.0, gpu_seconds=1200.0), outcome_vectors(BASE_SUCCESSES), lineages())
        reordered = dict(BASE_SUCCESSES)
        reordered["Octo"] = 9
        sweep.add_point(point("cheap-256-10", usd=3.0, gpu_seconds=300.0, resolution_tier=256, denoising_steps=10),
                        outcome_vectors(reordered), lineages())
        selection = sweep.select()
        self.assertEqual(selection["status"], "selected")
        self.assertEqual(selection["operating_point_id"], "faithful-480-20")
        self.assertEqual(selection["cost"], 12.0)
        rejected = {row["operating_point_id"]: row for row in selection["rejected"]}
        self.assertIn("cheap-256-10", rejected)
        self.assertIn("pairwise_order_agreement", rejected["cheap-256-10"]["failing_tolerances"])
        self.assertIn("per_cell_absolute_error", rejected["cheap-256-10"]["failing_tolerances"])
        self.assertEqual(rejected["cheap-256-10"]["estimated_usd"], 3.0)

    def test_a_shorter_horizon_point_cannot_undercut_a_longer_one(self):
        sweep = self._sweep()
        sweep.add_point(point("faithful-480-20", usd=12.0, gpu_seconds=1200.0), outcome_vectors(BASE_SUCCESSES), lineages())
        short = dict(HORIZONS)
        short["to_basket"] = 40
        sweep.add_point(
            point("short-horizon-1", usd=1.0, gpu_seconds=100.0, task_horizon_actions=short),
            outcome_vectors(BASE_SUCCESSES),
            lineages(),
        )
        selection = sweep.select()
        self.assertEqual(selection["operating_point_id"], "faithful-480-20")
        rejected = {row["operating_point_id"]: row for row in selection["rejected"]}
        self.assertEqual(rejected["short-horizon-1"]["failing_tolerances"], ["fixed_task_horizon"])
        self.assertEqual(rejected["short-horizon-1"]["status"], "disqualified")
        # The shorter horizon is still visible next to its cost, not deleted.
        self.assertEqual(rejected["short-horizon-1"]["task_horizon_actions"]["to_basket"], 40)
        detail = next(row for row in sweep.report()["point_detail"] if row["operating_point_id"] == "short-horizon-1")
        self.assertEqual(detail["horizon_mismatch"]["to_basket"], {"declared": 40, "required": 100})

    def test_every_failed_setting_and_its_cost_survives_in_the_report(self):
        sweep = self._sweep()
        reordered = dict(BASE_SUCCESSES)
        reordered["Octo"] = 9
        sweep.add_point(point("faithful-480-20", usd=12.0, gpu_seconds=1200.0), outcome_vectors(BASE_SUCCESSES), lineages())
        sweep.add_point(point("failed-256-10", usd=3.0, gpu_seconds=300.0, resolution_tier=256), outcome_vectors(reordered), lineages())
        sweep.add_point(
            point("failed-unpriced", gpu_seconds=50.0, resolution_tier=256, denoising_steps=4),
            outcome_vectors(reordered),
            lineages(),
        )
        report = sweep.report()
        self.assertEqual([row["id"] for row in report["points"]], ["faithful-480-20", "failed-256-10", "failed-unpriced"])
        ledger = report["cost_ledger"]
        self.assertEqual(ledger["sweep_gpu_seconds"]["known_sum"], 1550.0)
        self.assertEqual(ledger["sweep_estimated_usd"]["status"], "partial")
        self.assertEqual(ledger["sweep_estimated_usd"]["operating_points_without_usd"], ["failed-unpriced"])
        self.assertEqual(ledger["sweep_estimated_usd"]["known_sum"], 15.0)

    def test_an_unconfirmed_selection_cannot_support_a_qualified_claim(self):
        sweep = self._sweep()
        sweep.add_point(point("faithful-480-20", usd=12.0, gpu_seconds=1200.0), outcome_vectors(BASE_SUCCESSES), lineages())
        selection = sweep.select()
        self.assertEqual(selection["confirmation_status"], "unconfirmed")
        self.assertFalse(selection["qualified_claim_allowed"])
        claim = sweep.qualified_claim()
        self.assertEqual(claim["status"], "blocked")
        self.assertIsNone(claim["operating_point"])
        self.assertIn("unconfirmed", claim["reason"])
        self.assertEqual(sweep.report()["confirmation"]["status"], "not_computed")

    def test_confirmation_needs_a_lineage_disjoint_panel(self):
        sweep = self._sweep()
        sweep.add_point(point("faithful-480-20", usd=12.0, gpu_seconds=1200.0), outcome_vectors(BASE_SUCCESSES), lineages())
        candidate = point("faithful-480-20-confirm", usd=12.0, gpu_seconds=1200.0, panel=CONFIRMATION_PANEL)
        reference = point(
            "full-fidelity-confirm", usd=40.0, gpu_seconds=4000.0, resolution_tier=704, denoising_steps=30,
            panel=CONFIRMATION_PANEL,
        )
        shared = sweep.confirm_on_disjoint_panel(
            candidate, outcome_vectors(BASE_SUCCESSES), reference, outcome_vectors(BASE_SUCCESSES), lineages=lineages()
        )
        self.assertEqual(shared["status"], "unavailable")
        self.assertEqual(shared["panel_disjointness"]["status"], "fail")
        self.assertEqual(sweep.qualified_claim()["status"], "blocked")

        unverifiable = sweep.confirm_on_disjoint_panel(
            candidate, outcome_vectors(BASE_SUCCESSES), reference, outcome_vectors(BASE_SUCCESSES), lineages=None
        )
        self.assertEqual(unverifiable["status"], "unavailable")
        self.assertEqual(unverifiable["panel_disjointness"]["status"], "unverifiable")

        confirmed = sweep.confirm_on_disjoint_panel(
            candidate,
            outcome_vectors(BASE_SUCCESSES),
            reference,
            outcome_vectors(BASE_SUCCESSES),
            lineages=lineages(lineage_prefix="confirmation-lineage"),
        )
        self.assertEqual(confirmed["status"], "confirmed")
        self.assertEqual(confirmed["panel_disjointness"]["status"], "pass")
        claim = sweep.qualified_claim()
        self.assertEqual(claim["status"], "allowed")
        self.assertTrue(claim["operating_point"]["qualified"])

    def test_a_disjoint_panel_can_reject_the_development_selection(self):
        sweep = self._sweep()
        sweep.add_point(point("faithful-480-20", usd=12.0, gpu_seconds=1200.0), outcome_vectors(BASE_SUCCESSES), lineages())
        drifted = dict(BASE_SUCCESSES)
        drifted["MiniVLA"] = 1
        rejected = sweep.confirm_on_disjoint_panel(
            point("faithful-480-20-confirm", usd=12.0, panel=CONFIRMATION_PANEL),
            outcome_vectors(drifted),
            point("full-fidelity-confirm", usd=40.0, resolution_tier=704, denoising_steps=30, panel=CONFIRMATION_PANEL),
            outcome_vectors(BASE_SUCCESSES),
            lineages=lineages(lineage_prefix="confirmation-lineage"),
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("per_cell_absolute_error", rejected["failing_tolerances"])
        claim = sweep.qualified_claim()
        self.assertEqual(claim["status"], "blocked")
        self.assertEqual(claim["confirmation_status"], "rejected")

    def test_a_confirmation_arm_must_use_the_selected_setting(self):
        sweep = self._sweep()
        sweep.add_point(point("faithful-480-20", usd=12.0, gpu_seconds=1200.0), outcome_vectors(BASE_SUCCESSES), lineages())
        wrong = sweep.confirm_on_disjoint_panel(
            point("other-setting", usd=9.0, denoising_steps=8, panel=CONFIRMATION_PANEL),
            outcome_vectors(BASE_SUCCESSES),
            point("full-fidelity-confirm", usd=40.0, resolution_tier=704, denoising_steps=30, panel=CONFIRMATION_PANEL),
            outcome_vectors(BASE_SUCCESSES),
            lineages=lineages(lineage_prefix="confirmation-lineage"),
        )
        self.assertEqual(wrong["status"], "rejected")
        self.assertEqual(wrong["failing_tolerances"], ["configuration_matches_selection"])

    def test_no_arm_is_called_cheapest_when_a_cost_is_unknown(self):
        sweep = self._sweep()
        sweep.add_point(point("priced", usd=12.0, gpu_seconds=1200.0), outcome_vectors(BASE_SUCCESSES), lineages())
        sweep.add_point(
            point("unpriced", resolution_tier=256), outcome_vectors(BASE_SUCCESSES), lineages()
        )
        selection = sweep.select()
        self.assertEqual(selection["status"], "unavailable")
        self.assertIn("USD is never guessed", selection["reason"])
        self.assertEqual(selection["operating_points_without_comparable_cost"], ["unpriced"])
        self.assertEqual(sweep.qualified_claim()["status"], "blocked")

    def test_gpu_seconds_rank_only_when_no_arm_has_usd(self):
        sweep = self._sweep()
        sweep.add_point(point("cheap-gpu", gpu_seconds=900.0), outcome_vectors(BASE_SUCCESSES), lineages())
        sweep.add_point(
            point("dear-gpu", gpu_seconds=1800.0, resolution_tier=704), outcome_vectors(BASE_SUCCESSES), lineages()
        )
        selection = sweep.select()
        self.assertEqual(selection["cost_metric"], "measured_gpu_seconds")
        self.assertEqual(selection["operating_point_id"], "cheap-gpu")
        self.assertFalse(selection["usd_available"])

    def test_mismatched_starts_seeds_or_judge_disqualify_an_arm(self):
        sweep = self._sweep()
        sweep.add_point(
            point("other-starts", usd=2.0), outcome_vectors(BASE_SUCCESSES, prefix="different"), lineages()
        )
        sweep.add_point(point("other-seeds", usd=2.0, seed_set_id="seeds-other"), outcome_vectors(BASE_SUCCESSES), lineages())
        sweep.add_point(point("other-judge", usd=2.0, judge_revision_id="judge-rev-2"), outcome_vectors(BASE_SUCCESSES), lineages())
        sweep.add_point(point("unvalidated", usd=2.0, resolution_validation_ref=None), outcome_vectors(BASE_SUCCESSES), lineages())
        failures = {
            row["operating_point_id"]: row["failing_tolerances"] for row in sweep.select()["rejected"]
        }
        self.assertIn("matched_starts", failures["other-starts"])
        self.assertIn("same_seeds", failures["other-seeds"])
        self.assertIn("same_judge", failures["other-judge"])
        self.assertIn("validated_resolution", failures["unvalidated"])
        self.assertEqual(sweep.select()["status"], "unavailable")

    def test_one_noisy_ranking_match_is_not_enough_evidence(self):
        sweep = self._sweep(minimum_compared_pairs=200, minimum_evaluated_cells=60)
        sweep.add_point(point("faithful", usd=12.0), outcome_vectors(BASE_SUCCESSES), lineages())
        failures = sweep.select()["rejected"][0]["failing_tolerances"]
        self.assertIn("sufficient_compared_pairs", failures)
        self.assertIn("sufficient_evaluated_cells", failures)
        self.assertEqual(sweep.select()["status"], "unavailable")

    def test_coverage_below_tolerance_fails_per_cell(self):
        sweep = self._sweep()
        sweep.add_point(point("lossy", usd=4.0), outcome_vectors(BASE_SUCCESSES, missing=3), lineages())
        rejected = sweep.select()["rejected"][0]
        self.assertIn("coverage", rejected["failing_tolerances"])
        self.assertAlmostEqual(rejected["coverage"], 0.7)
        detail = sweep.report()["point_detail"][0]
        self.assertEqual(len(detail["cells_below_minimum_coverage"]), 30)

    def test_a_declared_coverage_must_match_the_point_s_own_outcome_vectors(self):
        sweep = self._sweep()
        sweep.add_point(point("overclaimed", usd=4.0, coverage=1.0), outcome_vectors(BASE_SUCCESSES, missing=3), lineages())
        failures = sweep.select()["rejected"][0]["failing_tolerances"]
        self.assertIn("declared_coverage_matches_outcomes", failures)

    def test_integer_outcomes_are_refused_so_missingness_cannot_become_failure(self):
        sweep = self._sweep()
        vectors = outcome_vectors(BASE_SUCCESSES)
        vectors["OpenVLA/open_drawer"]["start-000"] = 1
        with self.assertRaises(SweepError):
            sweep.add_point(point("malformed", usd=1.0), vectors, lineages())

    def test_duplicate_operating_point_ids_are_refused(self):
        sweep = self._sweep()
        sweep.add_point(point("dup", usd=1.0), outcome_vectors(BASE_SUCCESSES), lineages())
        with self.assertRaises(SweepError):
            sweep.add_point(point("dup", usd=2.0), outcome_vectors(BASE_SUCCESSES), lineages())

    def test_a_cost_model_supplies_the_point_cost_and_stays_unavailable_when_unpriced(self):
        sweep = self._sweep()
        priced = GpuSecondModel(
            verified_basis(), [ResourceRate("H100", 2.0, rate_unit="per_hour", currency="USD")], run_intervals(),
            gpu_resource_types=("H100",),
        )
        evaluated = sweep.add_point(
            point("from-model"), outcome_vectors(BASE_SUCCESSES), lineages(), cost_model=priced
        )
        self.assertAlmostEqual(evaluated.estimated_usd, 6.0)
        self.assertEqual(evaluated.cost_basis_ref, "price-snapshot-1")
        # Allocated GPU-seconds are never relabelled as instrumented measurement.
        self.assertIsNone(evaluated.measured_gpu_seconds)
        views = sweep.report()["cost_ledger"]["rows"][0]["cost_views"]
        self.assertAlmostEqual(views["total_demonstration_run_cost"]["allocated_gpu_seconds"], 10800.0)
        self.assertIsNone(views["total_demonstration_run_cost"]["measured_compute_gpu_seconds"])
        instrumented = GpuSecondModel(
            verified_basis(), [ResourceRate("H100", 2.0, rate_unit="per_hour", currency="USD")], run_intervals(),
            gpu_resource_types=("H100",), measured_compute_gpu_seconds=8400.0,
        )
        timed = sweep.add_point(
            point("from-instrumented-model", denoising_steps=12), outcome_vectors(BASE_SUCCESSES), lineages(),
            cost_model=instrumented,
        )
        self.assertAlmostEqual(timed.measured_gpu_seconds, 8400.0)
        unpriced = GpuSecondModel(
            verified_basis(units_confirmed=False),
            [ResourceRate("H100", 2.0, rate_unit="per_hour", currency="USD")],
            run_intervals(),
        )
        second = sweep.add_point(
            point("from-unpriced-model", resolution_tier=256), outcome_vectors(BASE_SUCCESSES), lineages(),
            cost_model=unpriced,
        )
        self.assertIsNone(second.estimated_usd)
        self.assertIn("price_units_unconfirmed", second.usd_unavailable_reason)

    def test_the_report_is_json_safe_and_deterministic(self):
        first = self._sweep()
        first.add_point(point("faithful", usd=12.0, gpu_seconds=1200.0), outcome_vectors(BASE_SUCCESSES), lineages())
        second = self._sweep()
        second.add_point(point("faithful", usd=12.0, gpu_seconds=1200.0), outcome_vectors(BASE_SUCCESSES), lineages())
        report = first.report()
        self.assertEqual(report, second.report())
        json.dumps(report, allow_nan=False)
        self.assertIn("not the published human reference table", report["comparison_target"])
        self.assertEqual(report["selection"]["gate_invalidation_key"], "operating_point")

    def test_malformed_artifacts_are_rejected_without_raising(self):
        rejected = build_cost_fidelity_report({"tolerances": {}, "points": []})
        self.assertEqual(rejected["status"], "rejected")
        self.assertIsInstance(rejected["reason"], str)
        self.assertEqual(build_cost_fidelity_report("not-a-mapping")["status"], "rejected")
        self.assertEqual(build_cost_fidelity_report({"tolerances": {"protocol_id": "p"}})["status"], "rejected")

    def test_artifacts_round_trip_through_the_json_builder(self):
        artifacts = {
            "tolerances": {
                "protocol_id": "p1",
                "maximum_absolute_cell_error": 0.1,
                "minimum_coverage": 0.9,
                "minimum_pairwise_order_agreement": 1.0,
                "required_task_horizon_actions": HORIZONS,
            },
            "full_fidelity_reference": {
                "operating_point": {
                    "operating_point_id": "full",
                    "resolution_tier": 704,
                    "denoising_steps": 30,
                    "chunk_action_length": 16,
                    "batch_size": 1,
                    "task_horizon_actions": HORIZONS,
                    "resolution_validation_ref": "preflight",
                    "seed_set_id": "seeds",
                    "judge_revision_id": "judge",
                },
                "outcomes": outcome_vectors(BASE_SUCCESSES),
                "lineages": lineages(),
            },
            "points": [
                {
                    "operating_point": {
                        "operating_point_id": "cheap",
                        "resolution_tier": 480,
                        "denoising_steps": 20,
                        "chunk_action_length": 16,
                        "batch_size": 1,
                        "task_horizon_actions": HORIZONS,
                        "resolution_validation_ref": "preflight",
                        "seed_set_id": "seeds",
                        "judge_revision_id": "judge",
                        "estimated_usd": 9.0,
                        "cost_basis_ref": "snapshot",
                    },
                    "outcomes": outcome_vectors(BASE_SUCCESSES),
                    "lineages": lineages(),
                }
            ],
        }
        report = build_cost_fidelity_report(artifacts)
        self.assertEqual(report["status"], "computed")
        self.assertEqual(report["selection"]["operating_point_id"], "cheap")
        self.assertFalse(report["selection"]["qualified_claim_allowed"])
        json.dumps(report, allow_nan=False)


def drift_point(control_ticks, chunk=16, backend="cosmos-256-profile", **overrides):
    values = {
        "backend_profile_id": backend,
        "control_ticks": control_ticks,
        "fd_request_action_length": chunk,
        "autoregressive_boundaries": tuple(range(chunk, control_ticks + 1, chunk)),
        "policy_feedback_boundaries": tuple(range(chunk, control_ticks + 1, chunk)),
        "fd_requests": control_ticks // chunk,
        "control_hz": 5.0,
        "nominal_elapsed_seconds": control_ticks / 5.0,
        "feedback_mode": "native_feedback",
    }
    values.update(overrides)
    return DriftPoint(**values)


def held_out_sequence(prefix=2, ticks=6, free_offset=0.2, **overrides):
    sequence = {
        "sequence_id": "bridge-holdout-1",
        "held_out": True,
        "control_prefix_ticks": prefix,
        "generation_settings_hash": "sha256:generation-settings",
        "real_actions": [[0.0] * 7 for _ in range(ticks)],
        "teacher_forced_actions": [[0.01] * 7 for _ in range(ticks)],
        "free_running_actions": [[free_offset] * 7 for _ in range(ticks)],
        "teacher_forced_integrity": "intact",
        "free_running_integrity": "artifact",
        "visual_divergence": {
            "metric": "lpips_vgg",
            "units": "unitless",
            "teacher_forced": [0.01] * (ticks - prefix),
            "free_running": [0.3] * (ticks - prefix),
        },
    }
    sequence.update(overrides)
    return sequence


class DriftPointContractTests(unittest.TestCase):
    def test_a_drift_point_cannot_be_built_without_its_four_tuple(self):
        complete = drift_point(64)
        self.assertEqual(complete.four_tuple, (64, 16, (16, 32, 48, 64), (16, 32, 48, 64)))
        required = {
            "backend_profile_id": "cosmos-256-profile",
            "control_ticks": 64,
            "fd_request_action_length": 16,
            "autoregressive_boundaries": (16, 32, 48, 64),
            "policy_feedback_boundaries": (16, 32, 48, 64),
        }
        for field_name in required:
            values = dict(required)
            values[field_name] = None
            with self.assertRaises(SweepError, msg=field_name):
                DriftPoint(**values)
        with self.assertRaises(SweepError):
            drift_point(64, autoregressive_boundaries=())
        with self.assertRaises(SweepError):
            drift_point(64, policy_feedback_boundaries=(32, 16))
        with self.assertRaises(SweepError):
            drift_point(64, autoregressive_boundaries=(16, 128))
        self.assertEqual(
            sorted(DriftPoint.from_mapping(complete.as_dict()).four_tuple[2]), [16, 32, 48, 64]
        )
        with self.assertRaises(SweepError):
            DriftPoint.from_mapping({"backend_profile_id": "b", "control_ticks": 32})

    def test_the_x_axis_is_control_ticks_and_never_a_request_count(self):
        study = DriftStudy("cosmos-256-profile", [drift_point(32), drift_point(64)])
        axis = study.axis()
        self.assertEqual(axis["x_axis"], "control_ticks")
        self.assertTrue(axis["request_count_is_not_an_axis"])
        self.assertEqual(axis["tested_horizons"], [32, 64])
        self.assertEqual([row["control_ticks"] for row in axis["rows"]], [32, 64])
        self.assertTrue(all(row["x_axis"] == "control_ticks" for row in axis["rows"]))
        self.assertEqual(axis["x_axis_alternative"], "nominal_elapsed_seconds")
        bare = DriftStudy("cosmos-256-profile", [drift_point(32, nominal_elapsed_seconds=None)])
        self.assertIsNone(bare.axis()["x_axis_alternative"])

    def test_an_unnamed_error_metric_is_refused(self):
        with self.assertRaises(SweepError):
            drift_point(64, mean_action_error=0.1)
        with self.assertRaises(SweepError):
            drift_point(64, mean_visual_divergence=0.1)
        self.assertIsNotNone(drift_point(64, mean_action_error=0.1, action_error_metric="l2").mean_action_error)

    def test_a_study_cannot_mix_backend_profiles_or_repeat_a_horizon(self):
        with self.assertRaises(SweepError):
            DriftStudy("cosmos-256-profile", [drift_point(32), drift_point(64, backend="irasim-profile")])
        with self.assertRaises(SweepError):
            DriftStudy("cosmos-256-profile", [drift_point(32), drift_point(32)])


class DriftRequestDisplayTests(unittest.TestCase):
    def test_the_request_display_needs_backend_action_length_and_cadence(self):
        study = DriftStudy("cosmos-256-profile", [drift_point(16), drift_point(96)])
        display = study.request_display()
        self.assertEqual(display["status"], "available")
        self.assertEqual(display["backend_profile_id"], "cosmos-256-profile")
        self.assertEqual([row["fd_requests"] for row in display["rows"]], [1, 6])
        self.assertTrue(all(row["fd_request_action_length"] == 16 for row in display["rows"]))
        self.assertTrue(all(row["policy_feedback_cadence_ticks"] == 16 for row in display["rows"]))
        self.assertFalse(display["is_common_physical_duration_across_policies"])
        self.assertIn("not the drift x-axis", display["caveat"])

        missing = DriftStudy("cosmos-256-profile", [drift_point(16, fd_requests=None)])
        self.assertEqual(missing.request_display()["status"], "unavailable")
        self.assertEqual(missing.request_display()["incomplete_points"][0]["missing"], ["fd_requests"])
        self.assertEqual(missing.request_display()["rows"], [])

    def test_irregular_feedback_cadence_blocks_the_request_display(self):
        irregular = DriftStudy(
            "cosmos-256-profile",
            [drift_point(64, policy_feedback_boundaries=(16, 48, 64))],
        )
        self.assertIsNone(irregular.points[0].policy_feedback_cadence_ticks)
        self.assertEqual(irregular.request_display()["status"], "unavailable")

    def test_a_request_count_disagreeing_with_boundaries_is_flagged(self):
        study = DriftStudy("cosmos-256-profile", [drift_point(64, fd_requests=2)])
        display = study.request_display()
        self.assertEqual(display["status"], "available")
        self.assertEqual(
            display["fd_request_count_disagrees_with_autoregressive_boundaries"][0]["autoregressive_boundaries"], 4
        )

    def test_requests_outside_one_to_six_are_reported(self):
        study = DriftStudy("cosmos-256-profile", [drift_point(160, fd_requests=10)])
        self.assertEqual(study.request_display()["requests_outside_one_to_six"], [10])


class DriftPartitionGuardTests(unittest.TestCase):
    def test_a_chunk_length_varying_partition_sweep_is_chunking_sensitivity(self):
        study = DriftStudy("cosmos-256-profile", [drift_point(64, chunk=8), drift_point(96, chunk=16)])
        classification = study.partition_sweep_classification()
        self.assertEqual(classification["classification"], "chunking_sensitivity")
        self.assertFalse(classification["isolated_drift"])
        self.assertIn("not isolated drift", classification["reason"])
        self.assertEqual(classification["chunk_action_lengths"], [8, 16])

    def test_a_fixed_horizon_partition_sweep_is_chunking_sensitivity(self):
        # Same horizon, different partitions: the chunk length must change, so
        # this is chunk sensitivity by construction, never isolated drift.
        study = DriftStudy(
            "cosmos-256-profile",
            [
                drift_point(64, chunk=16),
                DriftPoint(
                    backend_profile_id="cosmos-256-profile",
                    control_ticks=63,
                    fd_request_action_length=21,
                    autoregressive_boundaries=(21, 42, 63),
                    policy_feedback_boundaries=(21, 42, 63),
                ),
            ],
        )
        self.assertEqual(study.partition_sweep_classification()["classification"], "chunking_sensitivity")

    def test_isolated_drift_requires_a_constant_chunk_length(self):
        study = DriftStudy("cosmos-256-profile", [drift_point(32), drift_point(64), drift_point(96)])
        classification = study.partition_sweep_classification()
        self.assertEqual(classification["classification"], "isolated_drift")
        self.assertTrue(classification["isolated_drift"])
        self.assertEqual(classification["chunk_action_lengths"], [16])
        self.assertEqual(classification["partition_counts"], [2, 4, 6])

    def test_a_single_horizon_cannot_be_classified(self):
        single = DriftStudy("cosmos-256-profile", [drift_point(64)])
        self.assertEqual(single.partition_sweep_classification()["classification"], "insufficient_points")
        self.assertFalse(single.partition_sweep_classification()["isolated_drift"])
        constant = DriftStudy("cosmos-256-profile", [])
        self.assertFalse(constant.partition_sweep_classification()["isolated_drift"])


class DriftHorizonToleranceTests(unittest.TestCase):
    tolerances = DriftTolerances(
        protocol_id="drift-protocol-1",
        maximum_mean_action_error=0.1,
        action_error_metric="mean_per_tick_l2",
        minimum_integrity_intact_fraction=0.9,
    )

    def _study(self, rows):
        points = [
            drift_point(
                ticks,
                mean_action_error=error,
                action_error_metric="mean_per_tick_l2",
                integrity_intact_fraction=intact,
                n_sequences=5,
            )
            for ticks, error, intact in rows
        ]
        return DriftStudy("cosmos-256-profile", points, tolerances=self.tolerances)

    def test_the_reported_horizon_is_the_last_tested_one_and_guarantees_nothing_beyond(self):
        study = self._study([(16, 0.01, 1.0), (32, 0.02, 1.0), (48, 0.09, 0.95), (64, 0.30, 0.6)])
        report = study.last_horizon_meeting_tolerance()
        self.assertEqual(report["status"], "computed")
        self.assertEqual(report["last_horizon_meeting_tolerance"], 48)
        self.assertEqual(report["last_tested_horizon"], 64)
        self.assertFalse(report["unlimited_guarantee"])
        self.assertIn("not an unlimited drift guarantee", report["note"])
        self.assertIn("last *tested*", report["note"])
        self.assertEqual(report["failing_horizons"], [64])
        self.assertEqual(report["x_axis"], "control_ticks")

    def test_a_later_pass_after_an_earlier_failure_does_not_extend_the_horizon(self):
        study = self._study([(16, 0.01, 1.0), (32, 0.50, 1.0), (48, 0.01, 1.0)])
        report = study.last_horizon_meeting_tolerance()
        self.assertEqual(report["last_horizon_meeting_tolerance"], 16)
        self.assertEqual(report["failing_horizons"], [32])
        self.assertIn("no failing or unmeasured horizon below it", report["contiguity_rule"])

    def test_an_unmeasured_horizon_never_counts_as_passing(self):
        points = [
            drift_point(16, mean_action_error=0.01, action_error_metric="mean_per_tick_l2", integrity_intact_fraction=1.0),
            drift_point(32, integrity_intact_fraction=1.0),
        ]
        study = DriftStudy("cosmos-256-profile", points, tolerances=self.tolerances)
        report = study.last_horizon_meeting_tolerance()
        self.assertEqual(report["last_horizon_meeting_tolerance"], 16)
        self.assertEqual(report["unmeasured_horizons"], [32])
        self.assertEqual(report["per_horizon"][1]["unmeasured"], ["mean_action_error"])

    def test_a_metric_mismatch_fails_rather_than_silently_comparing(self):
        points = [
            drift_point(16, mean_action_error=0.01, action_error_metric="some_other_metric", integrity_intact_fraction=1.0)
        ]
        study = DriftStudy("cosmos-256-profile", points, tolerances=self.tolerances)
        report = study.last_horizon_meeting_tolerance()
        self.assertIsNone(report["last_horizon_meeting_tolerance"])
        self.assertEqual(report["per_horizon"][0]["failing"], ["action_error_metric_mismatch"])

    def test_no_tolerances_means_no_horizon_claim(self):
        study = DriftStudy("cosmos-256-profile", [drift_point(64)])
        report = study.last_horizon_meeting_tolerance()
        self.assertEqual(report["status"], "unavailable")
        self.assertIsNone(report["last_horizon_meeting_tolerance"])
        self.assertFalse(report["unlimited_guarantee"])
        with self.assertRaises(SweepError):
            DriftTolerances(protocol_id="empty")
        with self.assertRaises(SweepError):
            DriftTolerances(protocol_id="unnamed", maximum_mean_action_error=0.1)


class TeacherForcedVersusFreeRunningTests(unittest.TestCase):
    def _study(self):
        return DriftStudy("cosmos-256-profile", [drift_point(64)])

    def test_the_two_arms_are_compared_at_equal_control_prefixes(self):
        report = self._study().teacher_forced_vs_free_running([held_out_sequence()])
        self.assertEqual(report["status"], "computed")
        self.assertEqual(report["x_axis"], "control_ticks")
        row = report["sequences"][0]
        self.assertEqual(row["control_prefix_ticks"], 2)
        self.assertEqual(row["evaluated_ticks"], 4)
        self.assertLess(row["teacher_forced"]["mean_per_tick_l2"], row["free_running"]["mean_per_tick_l2"])
        self.assertGreater(row["free_running_minus_teacher_forced"]["mean_per_tick_l2"], 0.0)
        self.assertEqual(row["integrity"]["teacher_forced"], "intact")
        self.assertEqual(row["integrity"]["free_running"], "artifact")
        self.assertEqual(row["visual_divergence"]["metric"], "lpips_vgg")
        self.assertGreater(row["visual_divergence"]["free_running_minus_teacher_forced_mean"], 0.0)
        self.assertEqual(len(report["per_prefix"]), 1)
        self.assertEqual(report["per_prefix"][0]["integrity"]["free_running"]["intact_fraction"], 0.0)

    def test_different_prefixes_are_never_averaged_into_one_headline(self):
        report = self._study().teacher_forced_vs_free_running(
            [held_out_sequence(prefix=2), held_out_sequence(prefix=3, sequence_id="holdout-2")]
        )
        self.assertEqual([row["control_prefix_ticks"] for row in report["per_prefix"]], [2, 3])
        self.assertEqual([row["n_sequences"] for row in report["per_prefix"]], [1, 1])

    def test_mismatched_generation_settings_or_arms_exclude_a_sequence(self):
        cases = {
            "generation settings": held_out_sequence(
                teacher_forced_generation_settings_hash="sha256:a",
                free_running_generation_settings_hash="sha256:b",
            ),
            "not held out": held_out_sequence(held_out=False),
            "no prefix": held_out_sequence(control_prefix_ticks=0),
            "prefix covers everything": held_out_sequence(prefix=6, ticks=6),
        }
        for label, sequence in cases.items():
            report = self._study().teacher_forced_vs_free_running([sequence])
            self.assertEqual(report["status"], "unavailable", label)
            self.assertEqual(report["n_excluded"], 1, label)
            self.assertEqual(report["sequences"][0]["status"], "excluded", label)

    def test_unequal_arm_lengths_are_excluded_rather_than_truncated(self):
        sequence = held_out_sequence()
        sequence["free_running_actions"] = sequence["free_running_actions"][:-1]
        report = self._study().teacher_forced_vs_free_running([sequence])
        self.assertEqual(report["sequences"][0]["status"], "excluded")
        self.assertIn("exactly the real sequence's control ticks", report["sequences"][0]["reason"])

    def test_visual_divergence_without_a_named_metric_is_unavailable_not_zero(self):
        without = held_out_sequence(visual_divergence=None)
        report = self._study().teacher_forced_vs_free_running([without])
        divergence = report["sequences"][0]["visual_divergence"]
        self.assertEqual(divergence["status"], "unavailable")
        self.assertIn("not fabricated", divergence["reason"])
        self.assertEqual(report["per_prefix"][0]["visual_divergence"]["status"], "unavailable")

        unnamed = held_out_sequence()
        unnamed["visual_divergence"] = dict(unnamed["visual_divergence"])
        unnamed["visual_divergence"]["metric"] = "  "
        unnamed_report = self._study().teacher_forced_vs_free_running([unnamed])
        self.assertEqual(unnamed_report["sequences"][0]["visual_divergence"]["status"], "unavailable")

        wrong_length = held_out_sequence()
        wrong_length["visual_divergence"] = dict(wrong_length["visual_divergence"])
        wrong_length["visual_divergence"]["free_running"] = [0.3]
        mismatched = self._study().teacher_forced_vs_free_running([wrong_length])
        self.assertEqual(mismatched["sequences"][0]["visual_divergence"]["status"], "unavailable")

    def test_divergent_metrics_across_sequences_are_not_aggregated(self):
        second = held_out_sequence(sequence_id="holdout-2")
        second["visual_divergence"] = dict(second["visual_divergence"])
        second["visual_divergence"]["metric"] = "psnr"
        report = self._study().teacher_forced_vs_free_running([held_out_sequence(), second])
        self.assertEqual(report["per_prefix"][0]["visual_divergence"]["status"], "unavailable")
        self.assertIn("different metrics", report["per_prefix"][0]["visual_divergence"]["reason"])

    def test_an_unknown_integrity_label_is_refused(self):
        with self.assertRaises(SweepError):
            self._study().teacher_forced_vs_free_running([held_out_sequence(free_running_integrity="broken")])

    def test_no_sequences_is_not_computed_rather_than_zero_drift(self):
        report = self._study().teacher_forced_vs_free_running([])
        self.assertEqual(report["status"], "not_computed")
        self.assertEqual(report["sequences"], [])


class DriftReportTests(unittest.TestCase):
    def test_the_report_is_json_safe_and_keeps_every_guard(self):
        study = DriftStudy(
            "cosmos-256-profile",
            [
                drift_point(32, mean_action_error=0.01, action_error_metric="mean_per_tick_l2", integrity_intact_fraction=1.0),
                drift_point(64, mean_action_error=0.40, action_error_metric="mean_per_tick_l2", integrity_intact_fraction=0.5),
            ],
            tolerances=DriftTolerances(
                protocol_id="drift-protocol-1",
                maximum_mean_action_error=0.1,
                action_error_metric="mean_per_tick_l2",
                minimum_integrity_intact_fraction=0.9,
            ),
            sequences=[held_out_sequence()],
        )
        report = study.report()
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["status"], "computed")
        self.assertEqual(report["axis"]["x_axis"], "control_ticks")
        self.assertEqual(report["partition_sweep"]["classification"], "isolated_drift")
        self.assertEqual(report["horizon_tolerance"]["last_horizon_meeting_tolerance"], 32)
        self.assertFalse(report["horizon_tolerance"]["unlimited_guarantee"])
        self.assertEqual(report["teacher_forced_vs_free_running"]["status"], "computed")
        self.assertFalse(report["request_display"]["is_common_physical_duration_across_policies"])

    def test_malformed_drift_artifacts_are_rejected_without_raising(self):
        self.assertEqual(build_drift_report({"points": []})["status"], "rejected")
        self.assertEqual(build_drift_report("nope")["status"], "rejected")
        self.assertEqual(
            build_drift_report({"points": [{"backend_profile_id": "b", "control_ticks": 16}]})["status"], "rejected"
        )

    def test_drift_artifacts_round_trip_through_the_json_builder(self):
        artifacts = {
            "backend_profile_id": "cosmos-256-profile",
            "tolerances": {
                "protocol_id": "drift-protocol-1",
                "maximum_mean_action_error": 0.1,
                "action_error_metric": "mean_per_tick_l2",
            },
            "points": [
                drift_point(
                    ticks, mean_action_error=error, action_error_metric="mean_per_tick_l2"
                ).as_dict()
                for ticks, error in ((32, 0.01), (64, 0.5))
            ],
            "teacher_forced_vs_free_running": [held_out_sequence()],
        }
        report = build_drift_report(artifacts)
        self.assertEqual(report["status"], "computed")
        self.assertEqual(report["horizon_tolerance"]["last_horizon_meeting_tolerance"], 32)
        self.assertEqual(report["horizon_tolerance"]["failing_horizons"], [64])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
