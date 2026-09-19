"""Cost-fidelity and drift sweeps for PLUMB.

Two of the four headline measurements live here.

``CostFidelitySweep`` selects a cheaper validated operating point only when it
reproduces the *full-fidelity operating point's own cells* inside preregistered
tolerances, and only after a disjoint confirmation panel agrees.  A shorter task
horizon is not a cheaper equivalent evaluation, so the fixed task horizon and
the coverage travel with every cost point and a horizon mismatch disqualifies a
setting outright.  Every failed setting and its cost is preserved in the output.

``DriftStudy`` keeps the drift x-axis in control ticks.  The 1--6-request display
survives only as backend-specific transport/horizon sensitivity, with its backend
profile, FD-request action length and feedback cadence stated.  A partition sweep
whose chunk length also changes is labelled ``chunking_sensitivity``, never
``isolated_drift``.

Neither class fabricates a measurement.  Unknown GPU-seconds, USD, visual
divergence or integrity labels stay ``None`` with an explicit reason, and USD is
unavailable until a vendor/account price basis with confirmed units exists: an
API field named ``price`` is not evidence of dollars per hour.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .reference import canonical_policy, canonical_task


SCHEMA_VERSION = 1

DEVELOPMENT_PANEL = "development"
CONFIRMATION_PANEL = "cost_confirmation"

MARGINAL_VIEW = "marginal_execution_estimate"
TOTAL_VIEW = "total_demonstration_run_cost"
COST_VIEWS = (MARGINAL_VIEW, TOTAL_VIEW)

# The selected operating point is the one whose *total demonstration-run cost*
# is lowest.  The marginal view may omit idle capacity, so it is reported but
# never used to claim one setting is cheaper than another.
SELECTION_COST_VIEW = TOTAL_VIEW

# The total view claims "prewarm start through cooldown"; those two stages must
# actually be present before that claim is made.
REQUIRED_TOTAL_VIEW_STAGES = ("prewarm", "cooldown")
IDLE_STAGES = ("idle", "cold_start")

# Mirrors the judge's integrity vocabulary.  It is duplicated rather than
# imported so this module stays independent of the calibration workflow.
INTEGRITY_LABELS = ("intact", "artifact", "uncertain")

# Confirmed time units only.  A bare "price" or an ambiguous "per_instance_hour"
# is deliberately absent: allocated resource hours already carry the resource
# count, so a per-instance figure cannot be multiplied safely.
BILLING_UNIT_HOURS = {
    "per_hour": 1.0,
    "usd_per_hour": 1.0,
    "per_resource_hour": 1.0,
    "per_gpu_hour": 1.0,
    "per_minute": 1.0 / 60.0,
    "usd_per_minute": 1.0 / 60.0,
    "per_second": 1.0 / 3600.0,
    "usd_per_second": 1.0 / 3600.0,
}

FIDELITY_TOLERANCE_NAMES = (
    "per_cell_absolute_error",
    "mean_absolute_cell_error",
    "coverage",
    "pairwise_order_agreement",
    "sufficient_compared_pairs",
    "sufficient_evaluated_cells",
)
DISQUALIFYING_TOLERANCE_NAMES = (
    "fixed_task_horizon",
    "validated_resolution",
    "matched_cells",
    "matched_starts",
    "same_seeds",
    "same_judge",
    "declared_coverage_matches_outcomes",
)


class SweepError(ValueError):
    """Raised when a sweep object cannot be constructed from its inputs."""


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SweepError("{0} must be a non-empty string".format(field_name))
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SweepError("{0} must be a positive integer".format(field_name))
    return value


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SweepError("{0} must be a finite number".format(field_name))
    return float(value)


def _nonnegative(value: Any, field_name: str) -> float:
    number = _finite(value, field_name)
    if number < 0.0:
        raise SweepError("{0} must not be negative".format(field_name))
    return number


def _probability(value: Any, field_name: str) -> float:
    number = _finite(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise SweepError("{0} must be a proportion from 0 through 1".format(field_name))
    return number


def _optional_probability(value: Any, field_name: str) -> Optional[float]:
    return None if value is None else _probability(value, field_name)


def _optional_nonnegative(value: Any, field_name: str) -> Optional[float]:
    return None if value is None else _nonnegative(value, field_name)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SweepError("{0} must be a mapping".format(field_name))
    return value


def _safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(present) / len(present) if present else None


def _boundaries(value: Any, field_name: str, control_ticks: int) -> Tuple[int, ...]:
    if value is None or isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        raise SweepError("{0} must be a list of control-tick indices".format(field_name))
    if not value:
        raise SweepError("{0} must not be empty".format(field_name))
    boundaries: List[int] = []
    for item in value:
        tick = _positive_int(item, "{0} entries".format(field_name))
        if tick > control_ticks:
            raise SweepError("{0} entries must not exceed control_ticks".format(field_name))
        boundaries.append(tick)
    if sorted(set(boundaries)) != boundaries:
        raise SweepError("{0} must be strictly increasing and unique".format(field_name))
    return tuple(boundaries)


def _order_relation(left: float, right: float) -> int:
    """Strict three-way relation; a tie must remain a tie in both vectors."""

    if left == right:
        return 0
    return 1 if left > right else -1


# ---------------------------------------------------------------------------
# Cost attribution (spec section 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceBasis:
    """The vendor/account pricing basis that must exist before USD is shown.

    ``units_confirmed`` is the explicit statement that currency, billing unit
    and resource applicability were read from the contract or account price
    endpoint and converted, rather than assumed from a field name.
    """

    basis_id: str
    currency: str
    billing_unit: str
    applicable_resource_types: Tuple[str, ...]
    tax_treatment: str
    egress_treatment: str
    storage_treatment: str
    retrieved_at: str
    reconciliation_method: str
    units_confirmed: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "basis_id",
            "currency",
            "billing_unit",
            "tax_treatment",
            "egress_treatment",
            "storage_treatment",
            "retrieved_at",
            "reconciliation_method",
        ):
            _nonempty_string(getattr(self, field_name), "price basis {0}".format(field_name))
        types = self.applicable_resource_types
        if isinstance(types, (str, bytes)) or not isinstance(types, (list, tuple)) or not types:
            raise SweepError("price basis applicable_resource_types must be a non-empty sequence")
        object.__setattr__(
            self,
            "applicable_resource_types",
            tuple(_nonempty_string(item, "applicable_resource_types entry") for item in types),
        )
        if not isinstance(self.units_confirmed, bool):
            raise SweepError("price basis units_confirmed must be a boolean")

    def hourly_usd(self, rate: "ResourceRate") -> Tuple[Optional[float], Optional[str]]:
        """Convert one declared rate to USD per resource-hour, or explain why not."""

        if not self.units_confirmed:
            return None, "price_units_unconfirmed"
        if self.currency.strip().upper() != "USD":
            return None, "non_usd_price_basis_without_confirmed_conversion"
        if rate.currency.strip().upper() != self.currency.strip().upper():
            return None, "rate_currency_mismatch"
        if rate.resource_type not in self.applicable_resource_types:
            return None, "resource_not_covered_by_price_basis"
        unit = rate.rate_unit if rate.rate_unit != "inherit_billing_unit" else self.billing_unit
        unit_hours = BILLING_UNIT_HOURS.get(unit.strip().lower())
        if unit_hours is None:
            return None, "unknown_billing_unit"
        if rate.rate_value is None:
            return None, "missing_rate_value"
        return rate.rate_value / unit_hours, None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "basis_id": self.basis_id,
            "currency": self.currency,
            "billing_unit": self.billing_unit,
            "applicable_resource_types": list(self.applicable_resource_types),
            "tax_treatment": self.tax_treatment,
            "egress_treatment": self.egress_treatment,
            "storage_treatment": self.storage_treatment,
            "retrieved_at": self.retrieved_at,
            "reconciliation_method": self.reconciliation_method,
            "units_confirmed": self.units_confirmed,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PriceBasis":
        data = dict(_mapping(payload, "price_basis"))
        types = data.get("applicable_resource_types", ())
        return cls(
            basis_id=data.get("basis_id", ""),
            currency=data.get("currency", ""),
            billing_unit=data.get("billing_unit", ""),
            applicable_resource_types=tuple(types) if isinstance(types, (list, tuple)) else types,
            tax_treatment=data.get("tax_treatment", ""),
            egress_treatment=data.get("egress_treatment", ""),
            storage_treatment=data.get("storage_treatment", ""),
            retrieved_at=data.get("retrieved_at", ""),
            reconciliation_method=data.get("reconciliation_method", ""),
            units_confirmed=bool(data.get("units_confirmed", False)),
        )


@dataclass(frozen=True)
class ResourceRate:
    """One timestamped account price for one resource type.

    ``rate_value`` is the raw number from the contract or price endpoint and
    ``rate_unit`` is its *declared* unit.  ``rate_unit="price"`` therefore
    resolves to an unknown billing unit rather than dollars per hour.
    """

    resource_type: str
    rate_value: Optional[float]
    rate_unit: str = "inherit_billing_unit"
    currency: str = "unknown"
    source_ref: Optional[str] = None

    def __post_init__(self) -> None:
        _nonempty_string(self.resource_type, "resource rate resource_type")
        _nonempty_string(self.rate_unit, "resource rate rate_unit")
        _nonempty_string(self.currency, "resource rate currency")
        if self.rate_value is not None:
            object.__setattr__(self, "rate_value", _nonnegative(self.rate_value, "resource rate rate_value"))
        if self.source_ref is not None:
            _nonempty_string(self.source_ref, "resource rate source_ref")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type,
            "rate_value": self.rate_value,
            "rate_unit": self.rate_unit,
            "currency": self.currency,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class AllocationInterval:
    """One allocation window on one billable resource lane.

    Overlap is checked per ``resource_id``: distinct replicas legitimately run
    at the same wall-clock time, while the same replica cannot be billed twice
    for one second.  ``allocated_resource_hours`` already multiplies the
    resource count, so a rate must be per single resource unit per hour.
    """

    interval_id: str
    resource_id: str
    resource_type: str
    stage: str
    start_seconds: float
    end_seconds: float
    resource_count: int = 1
    attributed_to_marginal_execution: bool = False
    shared_capacity_attribution_fraction: float = 1.0
    attribution_method: str = "exclusive_lane"

    def __post_init__(self) -> None:
        for field_name in ("interval_id", "resource_id", "resource_type", "stage", "attribution_method"):
            _nonempty_string(getattr(self, field_name), "allocation {0}".format(field_name))
        object.__setattr__(self, "start_seconds", _nonnegative(self.start_seconds, "allocation start_seconds"))
        object.__setattr__(self, "end_seconds", _nonnegative(self.end_seconds, "allocation end_seconds"))
        if self.end_seconds <= self.start_seconds:
            raise SweepError("allocation end_seconds must be strictly after start_seconds")
        _positive_int(self.resource_count, "allocation resource_count")
        fraction = _probability(self.shared_capacity_attribution_fraction, "shared_capacity_attribution_fraction")
        if fraction <= 0.0:
            raise SweepError("shared_capacity_attribution_fraction must be greater than zero")
        if not isinstance(self.attributed_to_marginal_execution, bool):
            raise SweepError("allocation attributed_to_marginal_execution must be a boolean")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def allocated_resource_seconds(self) -> float:
        return self.duration_seconds * self.resource_count * self.shared_capacity_attribution_fraction

    @property
    def allocated_resource_hours(self) -> float:
        return self.allocated_resource_seconds / 3600.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "interval_id": self.interval_id,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "stage": self.stage,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "resource_count": self.resource_count,
            "attributed_to_marginal_execution": self.attributed_to_marginal_execution,
            "shared_capacity_attribution_fraction": self.shared_capacity_attribution_fraction,
            "attribution_method": self.attribution_method,
            "allocated_resource_hours": self.allocated_resource_hours,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AllocationInterval":
        data = dict(_mapping(payload, "allocation interval"))
        return cls(
            interval_id=data.get("interval_id", ""),
            resource_id=data.get("resource_id", ""),
            resource_type=data.get("resource_type", ""),
            stage=data.get("stage", ""),
            start_seconds=data.get("start_seconds", 0.0),
            end_seconds=data.get("end_seconds", 0.0),
            resource_count=data.get("resource_count", 1),
            attributed_to_marginal_execution=bool(data.get("attributed_to_marginal_execution", False)),
            shared_capacity_attribution_fraction=data.get("shared_capacity_attribution_fraction", 1.0),
            attribution_method=data.get("attribution_method", "exclusive_lane"),
        )


@dataclass(frozen=True)
class OtherCharge:
    """A storage/network/service/tax charge added outside resource-hour rates."""

    charge_id: str
    category: str
    amount_usd: Optional[float]
    views: Tuple[str, ...] = (TOTAL_VIEW,)
    basis_ref: Optional[str] = None

    def __post_init__(self) -> None:
        _nonempty_string(self.charge_id, "charge_id")
        _nonempty_string(self.category, "charge category")
        if self.amount_usd is not None:
            object.__setattr__(self, "amount_usd", _nonnegative(self.amount_usd, "charge amount_usd"))
        views = self.views
        if isinstance(views, (str, bytes)) or not isinstance(views, (list, tuple)) or not views:
            raise SweepError("charge views must be a non-empty sequence")
        normalised = tuple(_nonempty_string(view, "charge view") for view in views)
        unknown = [view for view in normalised if view not in COST_VIEWS]
        if unknown:
            raise SweepError("charge views must name known cost views: {0}".format(sorted(unknown)))
        object.__setattr__(self, "views", normalised)
        if self.basis_ref is not None:
            _nonempty_string(self.basis_ref, "charge basis_ref")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "charge_id": self.charge_id,
            "category": self.category,
            "amount_usd": self.amount_usd,
            "views": list(self.views),
            "basis_ref": self.basis_ref,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OtherCharge":
        data = dict(_mapping(payload, "other charge"))
        views = data.get("views", (TOTAL_VIEW,))
        return cls(
            charge_id=data.get("charge_id", ""),
            category=data.get("category", ""),
            amount_usd=data.get("amount_usd"),
            views=tuple(views) if isinstance(views, (list, tuple)) else views,
            basis_ref=data.get("basis_ref"),
        )


class GpuSecondModel:
    """Attribute non-overlapping allocation intervals to the two cost views.

    ``estimated_usd = sum(resource_rate_per_hour * allocated_resource_hours) +
    other_charges`` for each view.  Every interval is counted exactly once, a
    repeated interval ID or an overlapping window on one resource lane is an
    error rather than extra spend, and USD stays ``None`` whenever the price
    basis, units, applicability or a charge amount is unknown.
    """

    def __init__(
        self,
        price_basis: Optional[PriceBasis] = None,
        rates: Sequence[ResourceRate] = (),
        intervals: Sequence[AllocationInterval] = (),
        other_charges: Sequence[OtherCharge] = (),
        gpu_resource_types: Sequence[str] = (),
        measured_compute_gpu_seconds: Optional[float] = None,
        prior_cost_reports: Optional[Mapping[str, Optional[float]]] = None,
        settled_cost_usd: Optional[float] = None,
        settled_reconciliation_ref: Optional[str] = None,
    ) -> None:
        if price_basis is not None and not isinstance(price_basis, PriceBasis):
            raise SweepError("price_basis must be a PriceBasis")
        self.price_basis = price_basis
        self.rates: Dict[str, ResourceRate] = {}
        for rate in rates:
            if not isinstance(rate, ResourceRate):
                raise SweepError("rates must contain ResourceRate objects")
            if rate.resource_type in self.rates:
                raise SweepError("duplicate resource rate for {0}".format(rate.resource_type))
            self.rates[rate.resource_type] = rate
        self.intervals: Tuple[AllocationInterval, ...] = tuple(intervals)
        for interval in self.intervals:
            if not isinstance(interval, AllocationInterval):
                raise SweepError("intervals must contain AllocationInterval objects")
        self.other_charges: Tuple[OtherCharge, ...] = tuple(other_charges)
        for charge in self.other_charges:
            if not isinstance(charge, OtherCharge):
                raise SweepError("other_charges must contain OtherCharge objects")
        if isinstance(gpu_resource_types, (str, bytes)):
            raise SweepError("gpu_resource_types must be a sequence of resource type names")
        self.gpu_resource_types = tuple(
            _nonempty_string(item, "gpu_resource_types entry") for item in gpu_resource_types
        )
        self.measured_compute_gpu_seconds = _optional_nonnegative(
            measured_compute_gpu_seconds, "measured_compute_gpu_seconds"
        )
        self.prior_cost_reports: Dict[str, Optional[float]] = {}
        if prior_cost_reports is not None:
            for key, value in _mapping(prior_cost_reports, "prior_cost_reports").items():
                self.prior_cost_reports[_nonempty_string(key, "prior cost key")] = _optional_nonnegative(
                    value, "prior cost {0}".format(key)
                )
        self.settled_cost_usd = _optional_nonnegative(settled_cost_usd, "settled_cost_usd")
        self.settled_reconciliation_ref = (
            None if settled_reconciliation_ref is None else _nonempty_string(settled_reconciliation_ref, "settled_reconciliation_ref")
        )
        if self.settled_cost_usd is not None and self.settled_reconciliation_ref is None:
            raise SweepError("a settled cost requires a billing reconciliation reference")

    # -- integrity -----------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        """Report duplicate IDs, overlapping lanes and missing rate coverage."""

        duplicate_ids = sorted(
            interval_id for interval_id, count in Counter(item.interval_id for item in self.intervals).items() if count > 1
        )
        overlaps: List[Dict[str, Any]] = []
        lanes: Dict[str, List[AllocationInterval]] = {}
        for interval in self.intervals:
            lanes.setdefault(interval.resource_id, []).append(interval)
        for resource_id, lane in sorted(lanes.items()):
            ordered = sorted(lane, key=lambda item: (item.start_seconds, item.end_seconds, item.interval_id))
            for previous, current in zip(ordered, ordered[1:]):
                if current.start_seconds < previous.end_seconds:
                    overlaps.append(
                        {
                            "resource_id": resource_id,
                            "interval_ids": [previous.interval_id, current.interval_id],
                            "overlap_seconds": previous.end_seconds - current.start_seconds,
                        }
                    )
        missing_rates = sorted(
            {interval.resource_type for interval in self.intervals if interval.resource_type not in self.rates}
        )
        marginal_idle = sorted(
            interval.interval_id
            for interval in self.intervals
            if interval.attributed_to_marginal_execution and interval.stage.strip().lower() in IDLE_STAGES
        )
        errors: List[Dict[str, Any]] = []
        if duplicate_ids:
            errors.append({"reason": "duplicate_allocation_interval_ids", "interval_ids": duplicate_ids})
        if overlaps:
            errors.append({"reason": "overlapping_allocation_intervals", "overlaps": overlaps})
        return {
            "status": "rejected" if errors else "accepted",
            "interval_count": len(self.intervals),
            "resource_lane_count": len(lanes),
            "errors": errors,
            "resource_types_without_rate": missing_rates,
            "marginal_intervals_on_idle_stages": marginal_idle,
            "note": (
                "Allocation intervals are counted once per resource lane; a repeated ID or an "
                "overlapping window on one lane is an error, not additional spend."
            ),
        }

    # -- views ---------------------------------------------------------------

    def _view_intervals(self, view: str) -> Tuple[AllocationInterval, ...]:
        if view == MARGINAL_VIEW:
            return tuple(item for item in self.intervals if item.attributed_to_marginal_execution)
        return self.intervals

    def view(self, view: str) -> Dict[str, Any]:
        """Return one labelled cost view without guessing an unknown price."""

        if view not in COST_VIEWS:
            raise SweepError("unknown cost view: {0}".format(view))
        validation = self.validate()
        intervals = self._view_intervals(view)
        hours_by_type: Dict[str, float] = {}
        hours_by_stage: Dict[str, float] = {}
        for interval in intervals:
            hours_by_type[interval.resource_type] = (
                hours_by_type.get(interval.resource_type, 0.0) + interval.allocated_resource_hours
            )
            hours_by_stage[interval.stage] = hours_by_stage.get(interval.stage, 0.0) + interval.allocated_resource_hours
        total_hours = sum(hours_by_type.values())
        unavailable: List[Dict[str, Any]] = []
        if validation["status"] != "accepted":
            unavailable.append({"reason": "allocation_ledger_rejected"})
        if self.price_basis is None:
            unavailable.append({"reason": "missing_price_basis"})
        components: List[Dict[str, Any]] = []
        resource_usd = 0.0
        for resource_type in sorted(hours_by_type):
            rate = self.rates.get(resource_type)
            hourly: Optional[float] = None
            reason: Optional[str] = None
            if rate is None:
                reason = "missing_resource_rate"
            elif self.price_basis is None:
                reason = "missing_price_basis"
            else:
                hourly, reason = self.price_basis.hourly_usd(rate)
            if hourly is None:
                unavailable.append({"reason": reason, "resource_type": resource_type})
            else:
                resource_usd += hourly * hours_by_type[resource_type]
            components.append(
                {
                    "resource_type": resource_type,
                    "allocated_resource_hours": hours_by_type[resource_type],
                    "rate_per_hour_usd": hourly,
                    "rate": None if rate is None else rate.as_dict(),
                    "unavailable_reason": reason,
                }
            )
        charges = [charge for charge in self.other_charges if view in charge.views]
        charges_usd = 0.0
        for charge in charges:
            if charge.amount_usd is None:
                unavailable.append({"reason": "other_charge_amount_unknown", "charge_id": charge.charge_id})
            else:
                charges_usd += charge.amount_usd
        estimated_usd = None if unavailable else resource_usd + charges_usd
        gpu_hours = (
            sum(hours_by_type.get(resource_type, 0.0) for resource_type in self.gpu_resource_types)
            if self.gpu_resource_types
            else None
        )
        stages_present = sorted(hours_by_stage)
        missing_stages = (
            [stage for stage in REQUIRED_TOTAL_VIEW_STAGES if stage not in hours_by_stage]
            if view == TOTAL_VIEW
            else []
        )
        return {
            "view": view,
            "label": (
                "Marginal execution estimate: attributed policy/world/validity/judge work, retries and "
                "transfers. It may omit idle capacity."
                if view == MARGINAL_VIEW
                else "Total demonstration-run cost: every allocated resource from prewarm start through "
                "cooldown, including idle/cold capacity, all stages, retries, CPU, storage and network."
            ),
            "formula": "estimated_usd = sum(resource_rate_per_hour * allocated_resource_hours) + other_charges",
            "interval_count": len(intervals),
            "interval_ids": sorted(item.interval_id for item in intervals),
            "window_seconds": (
                {
                    "start": min(item.start_seconds for item in intervals),
                    "end": max(item.end_seconds for item in intervals),
                }
                if intervals
                else None
            ),
            "allocated_resource_hours": total_hours,
            "allocated_resource_hours_by_type": hours_by_type,
            "allocated_resource_hours_by_stage": hours_by_stage,
            "stages": stages_present,
            "stage_coverage_status": "complete" if not missing_stages else "incomplete",
            "missing_required_stages": missing_stages,
            "allocated_gpu_seconds": None if gpu_hours is None else gpu_hours * 3600.0,
            "allocated_gpu_seconds_status": "computed" if gpu_hours is not None else "unavailable",
            "allocated_gpu_seconds_reason": (
                None if gpu_hours is not None else "gpu_resource_types were not declared; no SKU is guessed to be a GPU."
            ),
            "measured_compute_gpu_seconds": self.measured_compute_gpu_seconds,
            "telemetry_labels": {
                "allocated_gpu_seconds": "separate resource-allocation ledger",
                "measured_compute_gpu_seconds": "instrumented per-stage timings",
            },
            "components": components,
            "other_charges": [charge.as_dict() for charge in charges],
            "estimated_usd": estimated_usd,
            "usd_status": "computed" if estimated_usd is not None else "unavailable",
            "usd_unavailable_reasons": unavailable,
            "price_basis": None if self.price_basis is None else self.price_basis.as_dict(),
            "double_counting_check": validation,
        }

    def marginal_execution_estimate(self) -> Dict[str, Any]:
        return self.view(MARGINAL_VIEW)

    def total_demonstration_run_cost(self) -> Dict[str, Any]:
        return self.view(TOTAL_VIEW)

    def prior_costs(self) -> Dict[str, Any]:
        return {
            "status": "reported" if self.prior_cost_reports else "unavailable",
            "entries": dict(sorted(self.prior_cost_reports.items())),
            "note": "Prior development/training/calibration spend is reported separately and never recorded as zero.",
        }

    def settled_cost(self) -> Dict[str, Any]:
        if self.settled_cost_usd is None:
            return {
                "status": "unavailable",
                "reason": "Settled cost requires billing reconciliation after usage posts; an estimate is not a settlement.",
                "settled_usd": None,
            }
        return {
            "status": "settled",
            "settled_usd": self.settled_cost_usd,
            "reconciliation_ref": self.settled_reconciliation_ref,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "allocation_ledger": {
                "intervals": [interval.as_dict() for interval in self.intervals],
                "validation": self.validate(),
            },
            "views": {view: self.view(view) for view in COST_VIEWS},
            "prior_costs": self.prior_costs(),
            "settled_cost": self.settled_cost(),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GpuSecondModel":
        data = dict(_mapping(payload, "cost_model"))
        basis = data.get("price_basis")
        rates = data.get("rates", ())
        intervals = data.get("intervals", ())
        charges = data.get("other_charges", ())
        for name, value in (("rates", rates), ("intervals", intervals), ("other_charges", charges)):
            if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
                raise SweepError("cost_model {0} must be a list".format(name))
        gpu_types = data.get("gpu_resource_types", ())
        return cls(
            price_basis=None if basis is None else PriceBasis.from_mapping(basis),
            rates=[ResourceRate(**dict(_mapping(item, "rate"))) for item in rates],
            intervals=[AllocationInterval.from_mapping(item) for item in intervals],
            other_charges=[OtherCharge.from_mapping(item) for item in charges],
            gpu_resource_types=tuple(gpu_types) if isinstance(gpu_types, (list, tuple)) else gpu_types,
            measured_compute_gpu_seconds=data.get("measured_compute_gpu_seconds"),
            prior_cost_reports=data.get("prior_cost_reports"),
            settled_cost_usd=data.get("settled_cost_usd"),
            settled_reconciliation_ref=data.get("settled_reconciliation_ref"),
        )


# ---------------------------------------------------------------------------
# Cost-fidelity sweep (spec section 6.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepTolerances:
    """Tolerances chosen before any sweep arm is scored.

    ``required_task_horizon_actions`` has no default on purpose: the frozen
    per-task horizon is part of the preregistration, and a sweep that does not
    state it cannot detect a shorter task masquerading as a saving.
    """

    protocol_id: str
    maximum_absolute_cell_error: float
    minimum_coverage: float
    minimum_pairwise_order_agreement: float
    required_task_horizon_actions: Mapping[str, int]
    maximum_mean_absolute_cell_error: Optional[float] = None
    minimum_compared_pairs: int = 15
    minimum_evaluated_cells: int = 5
    require_validated_resolution: bool = True

    def __post_init__(self) -> None:
        _nonempty_string(self.protocol_id, "tolerances protocol_id")
        _probability(self.maximum_absolute_cell_error, "maximum_absolute_cell_error")
        _probability(self.minimum_coverage, "minimum_coverage")
        _probability(self.minimum_pairwise_order_agreement, "minimum_pairwise_order_agreement")
        if self.maximum_mean_absolute_cell_error is not None:
            _probability(self.maximum_mean_absolute_cell_error, "maximum_mean_absolute_cell_error")
        _positive_int(self.minimum_compared_pairs, "minimum_compared_pairs")
        _positive_int(self.minimum_evaluated_cells, "minimum_evaluated_cells")
        if not isinstance(self.require_validated_resolution, bool):
            raise SweepError("require_validated_resolution must be a boolean")
        object.__setattr__(self, "required_task_horizon_actions", _horizon_mapping(self.required_task_horizon_actions))
        if not self.required_task_horizon_actions:
            raise SweepError("required_task_horizon_actions must name at least one task horizon")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "maximum_absolute_cell_error": self.maximum_absolute_cell_error,
            "minimum_coverage": self.minimum_coverage,
            "minimum_pairwise_order_agreement": self.minimum_pairwise_order_agreement,
            "required_task_horizon_actions": dict(sorted(self.required_task_horizon_actions.items())),
            "maximum_mean_absolute_cell_error": self.maximum_mean_absolute_cell_error,
            "minimum_compared_pairs": self.minimum_compared_pairs,
            "minimum_evaluated_cells": self.minimum_evaluated_cells,
            "require_validated_resolution": self.require_validated_resolution,
            "note": (
                "A single noisy ranking match does not prove fidelity, so a minimum number of compared "
                "pairs and evaluated cells is preregistered alongside the error tolerances."
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SweepTolerances":
        data = dict(_mapping(payload, "tolerances"))
        allowed = {
            "protocol_id",
            "maximum_absolute_cell_error",
            "minimum_coverage",
            "minimum_pairwise_order_agreement",
            "required_task_horizon_actions",
            "maximum_mean_absolute_cell_error",
            "minimum_compared_pairs",
            "minimum_evaluated_cells",
            "require_validated_resolution",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise SweepError("unknown tolerance fields: {0}".format(unknown))
        try:
            return cls(**data)
        except TypeError as error:
            raise SweepError("tolerances are missing a required field: {0}".format(error)) from error


def _horizon_mapping(value: Any) -> Dict[str, int]:
    mapping = _mapping(value, "task horizon mapping")
    horizons: Dict[str, int] = {}
    for task, horizon in mapping.items():
        task_id = canonical_task(task)
        if task_id is None:
            raise SweepError("task horizon keys must be task identifiers")
        horizons[task_id] = _positive_int(horizon, "task horizon for {0}".format(task_id))
    return horizons


@dataclass(frozen=True)
class SweepPoint:
    """One validated operating point, its fixed horizon, coverage and cost.

    The fixed task horizon and the coverage travel with the point so a shorter
    task can never be displayed as a cheaper equivalent evaluation.  The chunk
    partition is the FD-request action length together with that fixed horizon;
    the partition count is derived, never an independent parameter.

    The fidelity fields are filled in by :class:`CostFidelitySweep` from the
    supplied per-cell outcome vectors.  Values passed in by a caller are
    recomputed, never trusted, and ``tolerances_met=None`` means "not evaluated
    on any panel" -- it never satisfies a selection gate.
    """

    operating_point_id: str
    resolution_tier: int
    denoising_steps: int
    chunk_action_length: int
    batch_size: int
    task_horizon_actions: Mapping[str, int]
    coverage: Optional[float] = None
    measured_gpu_seconds: Optional[float] = None
    estimated_usd: Optional[float] = None
    cost_basis_ref: Optional[str] = None
    usd_unavailable_reason: Optional[str] = None
    panel: str = DEVELOPMENT_PANEL
    label: Optional[str] = None
    resolution_validation_ref: Optional[str] = None
    seed_set_id: Optional[str] = None
    judge_revision_id: Optional[str] = None
    guidance_scale: Optional[float] = None
    per_cell_absolute_error: Mapping[str, float] = field(default_factory=dict)
    max_absolute_cell_error: Optional[float] = None
    mean_absolute_cell_error: Optional[float] = None
    pairwise_order_agreement: Optional[float] = None
    compared_pairs: Optional[int] = None
    evaluated_cells: Optional[int] = None
    tolerances_met: Optional[bool] = None
    failing_tolerances: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_string(self.operating_point_id, "operating_point_id")
        _nonempty_string(self.panel, "sweep point panel")
        _positive_int(self.resolution_tier, "resolution_tier")
        _positive_int(self.denoising_steps, "denoising_steps")
        _positive_int(self.chunk_action_length, "chunk_action_length")
        _positive_int(self.batch_size, "batch_size")
        object.__setattr__(self, "task_horizon_actions", _horizon_mapping(self.task_horizon_actions))
        if not self.task_horizon_actions:
            raise SweepError("a sweep point must state its fixed task horizon")
        object.__setattr__(self, "coverage", _optional_probability(self.coverage, "sweep point coverage"))
        object.__setattr__(
            self, "measured_gpu_seconds", _optional_nonnegative(self.measured_gpu_seconds, "measured_gpu_seconds")
        )
        object.__setattr__(self, "estimated_usd", _optional_nonnegative(self.estimated_usd, "estimated_usd"))
        if self.estimated_usd is not None and self.cost_basis_ref is None:
            raise SweepError("a USD amount requires a recorded cost_basis_ref pricing snapshot")
        for field_name in ("cost_basis_ref", "usd_unavailable_reason", "label", "resolution_validation_ref", "seed_set_id", "judge_revision_id"):
            value = getattr(self, field_name)
            if value is not None:
                _nonempty_string(value, "sweep point {0}".format(field_name))
        if self.guidance_scale is not None:
            object.__setattr__(self, "guidance_scale", _nonnegative(self.guidance_scale, "guidance_scale"))
        errors = _mapping(self.per_cell_absolute_error, "per_cell_absolute_error")
        object.__setattr__(
            self,
            "per_cell_absolute_error",
            {
                _nonempty_string(key, "per_cell_absolute_error key"): _probability(
                    value, "per_cell_absolute_error[{0}]".format(key)
                )
                for key, value in errors.items()
            },
        )
        object.__setattr__(
            self, "max_absolute_cell_error", _optional_probability(self.max_absolute_cell_error, "max_absolute_cell_error")
        )
        object.__setattr__(
            self,
            "mean_absolute_cell_error",
            _optional_probability(self.mean_absolute_cell_error, "mean_absolute_cell_error"),
        )
        object.__setattr__(
            self,
            "pairwise_order_agreement",
            _optional_probability(self.pairwise_order_agreement, "pairwise_order_agreement"),
        )
        for field_name in ("compared_pairs", "evaluated_cells"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise SweepError("sweep point {0} must be a non-negative integer".format(field_name))
        if self.tolerances_met is not None and not isinstance(self.tolerances_met, bool):
            raise SweepError("tolerances_met must be True, False or None")
        failing = self.failing_tolerances
        if isinstance(failing, (str, bytes)) or not isinstance(failing, (list, tuple)):
            raise SweepError("failing_tolerances must be a sequence of tolerance names")
        object.__setattr__(
            self, "failing_tolerances", tuple(_nonempty_string(name, "failing tolerance name") for name in failing)
        )
        if self.tolerances_met is True and self.failing_tolerances:
            raise SweepError("a point cannot meet its tolerances while naming failing tolerances")

    @property
    def usd_status(self) -> str:
        return "computed" if self.estimated_usd is not None else "unavailable"

    @property
    def partition_counts(self) -> Dict[str, int]:
        """Derived FD-request partitions per task at this chunk action length."""

        return {
            task: int(math.ceil(horizon / float(self.chunk_action_length)))
            for task, horizon in sorted(self.task_horizon_actions.items())
        }

    @property
    def configuration_key(self) -> Tuple[Any, ...]:
        """Identity of the *setting*, excluding panel, cost and fidelity fields."""

        return (
            self.resolution_tier,
            self.denoising_steps,
            self.chunk_action_length,
            self.batch_size,
            tuple(sorted(self.task_horizon_actions.items())),
            self.guidance_scale,
        )

    def as_dict(self) -> Dict[str, Any]:
        horizons = dict(sorted(self.task_horizon_actions.items()))
        distinct = sorted(set(horizons.values()))
        return {
            # Keys the dashboard cost slider reads directly.
            "id": self.operating_point_id,
            "label": self.label or self.operating_point_id,
            "estimated_usd": self.estimated_usd,
            "coverage": self.coverage,
            "horizon": distinct[0] if len(distinct) == 1 else horizons,
            "qualified": False,
            # Full record.
            "operating_point_id": self.operating_point_id,
            "panel": self.panel,
            "resolution_tier": self.resolution_tier,
            "denoising_steps": self.denoising_steps,
            "num_inference_steps": self.denoising_steps,
            "guidance_scale": self.guidance_scale,
            "chunk_action_length": self.chunk_action_length,
            "chunk_partition": self.partition_counts,
            "batch_size": self.batch_size,
            "task_horizon_actions": horizons,
            "measured_gpu_seconds": self.measured_gpu_seconds,
            "usd_status": self.usd_status,
            "usd_unavailable_reason": self.usd_unavailable_reason,
            "cost_basis_ref": self.cost_basis_ref,
            "resolution_validation_ref": self.resolution_validation_ref,
            "seed_set_id": self.seed_set_id,
            "judge_revision_id": self.judge_revision_id,
            "per_cell_absolute_error": dict(sorted(self.per_cell_absolute_error.items())),
            "max_absolute_cell_error": self.max_absolute_cell_error,
            "mean_absolute_cell_error": self.mean_absolute_cell_error,
            "pairwise_order_agreement": self.pairwise_order_agreement,
            "compared_pairs": self.compared_pairs,
            "evaluated_cells": self.evaluated_cells,
            "tolerances_met": self.tolerances_met,
            "failing_tolerances": list(self.failing_tolerances),
            "horizon_note": (
                "The fixed task horizon and coverage travel with this cost point; a shorter task is not a "
                "cheaper equivalent evaluation."
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SweepPoint":
        data = dict(_mapping(payload, "operating_point"))
        data.pop("num_inference_steps", None)
        data.pop("chunk_partition", None)
        for derived in ("id", "qualified", "horizon", "usd_status", "horizon_note"):
            data.pop(derived, None)
        allowed = {
            "operating_point_id",
            "resolution_tier",
            "denoising_steps",
            "chunk_action_length",
            "batch_size",
            "task_horizon_actions",
            "coverage",
            "measured_gpu_seconds",
            "estimated_usd",
            "cost_basis_ref",
            "usd_unavailable_reason",
            "panel",
            "label",
            "resolution_validation_ref",
            "seed_set_id",
            "judge_revision_id",
            "guidance_scale",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise SweepError("unknown operating point fields: {0}".format(unknown))
        try:
            return cls(**data)
        except TypeError as error:
            raise SweepError("operating point is missing a required field: {0}".format(error)) from error


def _cell_key(value: Any) -> Tuple[str, str]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise SweepError("a cell key tuple must be (policy, task)")
        policy_raw, task_raw = value
    elif isinstance(value, str):
        if value.count("/") != 1:
            raise SweepError("a cell key string must be 'policy/task'")
        policy_raw, task_raw = value.split("/", 1)
    else:
        raise SweepError("a cell key must be a 'policy/task' string or a (policy, task) tuple")
    policy = canonical_policy(policy_raw)
    task = canonical_task(task_raw)
    if policy is None or task is None:
        raise SweepError("a cell key must name a policy and a task")
    return policy, task


def _parse_outcome_vectors(raw: Any, field_name: str) -> Dict[Tuple[str, str], Dict[str, Optional[bool]]]:
    """Read per-cell start-keyed outcome vectors without coercing missingness.

    ``True``/``False`` are evaluable outcomes and ``None`` is an unevaluable or
    unrecorded planned start.  Integers are rejected so a malformed export
    cannot become a failure by accident.
    """

    mapping = _mapping(raw, field_name)
    if not mapping:
        raise SweepError("{0} must contain at least one cell".format(field_name))
    vectors: Dict[Tuple[str, str], Dict[str, Optional[bool]]] = {}
    for key, value in mapping.items():
        cell = _cell_key(key)
        if cell in vectors:
            raise SweepError("duplicate cell key in {0}: {1}".format(field_name, cell))
        starts = _mapping(value, "{0}[{1}]".format(field_name, key))
        if not starts:
            raise SweepError("{0}[{1}] must contain at least one planned start".format(field_name, key))
        outcomes: Dict[str, Optional[bool]] = {}
        for start_id, outcome in starts.items():
            start = _nonempty_string(start_id, "start identifier")
            if start in outcomes:
                raise SweepError("duplicate start {0} in {1}[{2}]".format(start, field_name, key))
            if outcome is not None and not isinstance(outcome, bool):
                raise SweepError("outcomes must be True, False or None; {0}[{1}][{2}] is not".format(field_name, key, start))
            outcomes[start] = outcome
        vectors[cell] = outcomes
    return vectors


def _cell_stats(vector: Mapping[str, Optional[bool]]) -> Dict[str, Any]:
    """Observed-positive endpoint statistics for one cell, matching section 6."""

    n = len(vector)
    valid = sum(1 for outcome in vector.values() if outcome is not None)
    successes = sum(1 for outcome in vector.values() if outcome is True)
    return {
        "n": n,
        "valid": valid,
        "successes": successes,
        "observed_positive_rate": successes / float(n) if n else None,
        "coverage": valid / float(n) if n else None,
        "conditional_rate": successes / float(valid) if valid else None,
    }


def _macro_rates(
    stats: Mapping[Tuple[str, str], Mapping[str, Any]], policies: Sequence[str], tasks: Sequence[str]
) -> Dict[str, Optional[float]]:
    macro: Dict[str, Optional[float]] = {}
    for policy in policies:
        values = [stats[(policy, task)]["observed_positive_rate"] for task in tasks if (policy, task) in stats]
        macro[policy] = _safe_mean(values) if len(values) == len(tasks) else None
    return macro


def _order_agreement(
    point_stats: Mapping[Tuple[str, str], Mapping[str, Any]],
    reference_stats: Mapping[Tuple[str, str], Mapping[str, Any]],
    policies: Sequence[str],
    tasks: Sequence[str],
) -> Dict[str, Any]:
    """Compare strict pairwise order between a sweep arm and full fidelity.

    Ties are strict: a tie in one vector and an ordering in the other is a
    disagreement, matching the pinned SIMPLER-style comparison used elsewhere in
    PLUMB.  Macro pairs use equal-task means.
    """

    rows: List[Dict[str, Any]] = []
    agreements = 0
    compared = 0
    point_macro = _macro_rates(point_stats, policies, tasks)
    reference_macro = _macro_rates(reference_stats, policies, tasks)
    scopes: List[Tuple[str, Dict[str, Optional[float]], Dict[str, Optional[float]]]] = [
        (
            task,
            {policy: point_stats[(policy, task)]["observed_positive_rate"] for policy in policies if (policy, task) in point_stats},
            {
                policy: reference_stats[(policy, task)]["observed_positive_rate"]
                for policy in policies
                if (policy, task) in reference_stats
            },
        )
        for task in tasks
    ]
    if len(policies) > 1:
        scopes.append(("macro", point_macro, reference_macro))
    for scope, point_values, reference_values in scopes:
        for policy_a, policy_b in combinations(policies, 2):
            point_a = point_values.get(policy_a)
            point_b = point_values.get(policy_b)
            reference_a = reference_values.get(policy_a)
            reference_b = reference_values.get(policy_b)
            if None in (point_a, point_b, reference_a, reference_b):
                rows.append(
                    {
                        "scope": scope,
                        "policy_a": policy_a,
                        "policy_b": policy_b,
                        "status": "not_comparable",
                        "reason": "A rate is unavailable for this pair.",
                    }
                )
                continue
            point_relation = _order_relation(float(point_a), float(point_b))
            reference_relation = _order_relation(float(reference_a), float(reference_b))
            agrees = point_relation == reference_relation
            compared += 1
            agreements += 1 if agrees else 0
            rows.append(
                {
                    "scope": scope,
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "status": "compared",
                    "sweep_relation": point_relation,
                    "full_fidelity_relation": reference_relation,
                    "agrees": agrees,
                }
            )
    return {
        "agreement": agreements / float(compared) if compared else None,
        "compared_pairs": compared,
        "agreeing_pairs": agreements,
        "rows": rows,
        "tie_convention": "strict: a tie must be a tie in both vectors",
    }


class CostFidelitySweep:
    """Vary validated resolution, denoising steps and chunk partitions honestly.

    The comparison target is the full-fidelity *operating point's own measured
    cells*, not the published human table.  Every arm must share the reference's
    cells, starts, seeds, judge revision and fixed task horizon; an arm that does
    not is disqualified rather than scored, because it is not the same
    evaluation.  Failed and disqualified arms and all of their costs stay in the
    output.
    """

    def __init__(
        self,
        full_fidelity_point: SweepPoint,
        full_fidelity_outcomes: Any,
        tolerances: SweepTolerances,
        lineages: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not isinstance(full_fidelity_point, SweepPoint):
            raise SweepError("full_fidelity_point must be a SweepPoint")
        if not isinstance(tolerances, SweepTolerances):
            raise SweepError("tolerances must be a SweepTolerances preregistration")
        self.tolerances = tolerances
        self.reference_outcomes = _parse_outcome_vectors(full_fidelity_outcomes, "full_fidelity_outcomes")
        self.reference_stats = {cell: _cell_stats(vector) for cell, vector in self.reference_outcomes.items()}
        self.policies = sorted({policy for policy, _ in self.reference_outcomes})
        self.tasks = sorted({task for _, task in self.reference_outcomes})
        reference_n = sum(stats["n"] for stats in self.reference_stats.values())
        reference_valid = sum(stats["valid"] for stats in self.reference_stats.values())
        reference_coverage = reference_valid / float(reference_n) if reference_n else None
        if (
            full_fidelity_point.coverage is not None
            and reference_coverage is not None
            and abs(full_fidelity_point.coverage - reference_coverage) > 1e-9
        ):
            raise SweepError("the full-fidelity point's declared coverage disagrees with its own outcome vectors")
        # The full-fidelity arm is the comparison target, never a candidate, so
        # its fidelity fields stay unevaluated rather than a self-comparison.
        self.full_fidelity_point = replace(
            full_fidelity_point,
            panel=DEVELOPMENT_PANEL,
            coverage=full_fidelity_point.coverage if full_fidelity_point.coverage is not None else reference_coverage,
        )
        self.reference_lineages = _lineage_mapping(lineages, "full_fidelity lineages")
        self._points: List[SweepPoint] = []
        self._details: Dict[str, Dict[str, Any]] = {}
        self._cost_views: Dict[str, Dict[str, Any]] = {}
        self._lineages: Dict[str, Dict[str, str]] = {}
        self._confirmation: Optional[Dict[str, Any]] = None
        self._confirmation_reference: Dict[Tuple[str, str], Dict[str, Optional[bool]]] = {}
        self._confirmation_reference_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._confirmation_reference_point: SweepPoint = self.full_fidelity_point

    # -- evaluation ----------------------------------------------------------

    def _evaluate(
        self, point: SweepPoint, outcomes: Mapping[Tuple[str, str], Mapping[str, Optional[bool]]], panel: str
    ) -> Tuple[SweepPoint, Dict[str, Any]]:
        reference_outcomes = self.reference_outcomes if panel == DEVELOPMENT_PANEL else self._confirmation_reference
        reference_stats = self.reference_stats if panel == DEVELOPMENT_PANEL else self._confirmation_reference_stats
        failing: List[str] = []
        disqualified: List[str] = []

        if self.tolerances.require_validated_resolution and point.resolution_validation_ref is None:
            disqualified.append("validated_resolution")
        required = self.tolerances.required_task_horizon_actions
        declared = dict(point.task_horizon_actions)
        horizon_mismatch = {
            task: {"declared": declared.get(task), "required": horizon}
            for task, horizon in required.items()
            if declared.get(task) != horizon
        }
        extra_tasks = sorted(set(declared) - set(required))
        if horizon_mismatch or extra_tasks:
            disqualified.append("fixed_task_horizon")
        if set(outcomes) != set(reference_outcomes):
            disqualified.append("matched_cells")
        mismatched_starts = sorted(
            "{0}/{1}".format(*cell)
            for cell in set(outcomes) & set(reference_outcomes)
            if set(outcomes[cell]) != set(reference_outcomes[cell])
        )
        if mismatched_starts:
            disqualified.append("matched_starts")
        reference_point = self.full_fidelity_point if panel == DEVELOPMENT_PANEL else self._confirmation_reference_point
        if point.seed_set_id is None or point.seed_set_id != reference_point.seed_set_id:
            disqualified.append("same_seeds")
        if point.judge_revision_id is None or point.judge_revision_id != reference_point.judge_revision_id:
            disqualified.append("same_judge")

        stats = {cell: _cell_stats(vector) for cell, vector in outcomes.items()}
        shared = sorted(set(stats) & set(reference_stats))
        per_cell_error: Dict[str, float] = {}
        cell_rows: List[Dict[str, Any]] = []
        coverage_failures: List[str] = []
        for cell in shared:
            sweep_cell = stats[cell]
            reference_cell = reference_stats[cell]
            error = None
            if sweep_cell["observed_positive_rate"] is not None and reference_cell["observed_positive_rate"] is not None:
                error = abs(sweep_cell["observed_positive_rate"] - reference_cell["observed_positive_rate"])
                per_cell_error["{0}/{1}".format(*cell)] = error
            coverage = sweep_cell["coverage"]
            if coverage is None or coverage < self.tolerances.minimum_coverage:
                coverage_failures.append("{0}/{1}".format(*cell))
            cell_rows.append(
                {
                    "cell": "{0}/{1}".format(*cell),
                    "policy": cell[0],
                    "task": cell[1],
                    "n": sweep_cell["n"],
                    "valid": sweep_cell["valid"],
                    "successes": sweep_cell["successes"],
                    "observed_positive_rate": sweep_cell["observed_positive_rate"],
                    "coverage": coverage,
                    "full_fidelity_n": reference_cell["n"],
                    "full_fidelity_observed_positive_rate": reference_cell["observed_positive_rate"],
                    "absolute_error": error,
                }
            )
        max_error = max(per_cell_error.values()) if per_cell_error else None
        mean_error = _safe_mean(list(per_cell_error.values()))
        panel_n = sum(row["n"] for row in cell_rows)
        panel_valid = sum(row["valid"] for row in cell_rows)
        panel_coverage = panel_valid / float(panel_n) if panel_n else None
        if point.coverage is not None and panel_coverage is not None and abs(point.coverage - panel_coverage) > 1e-9:
            disqualified.append("declared_coverage_matches_outcomes")

        order = _order_agreement(stats, reference_stats, self.policies, self.tasks)
        if coverage_failures:
            failing.append("coverage")
        if max_error is None or max_error > self.tolerances.maximum_absolute_cell_error:
            failing.append("per_cell_absolute_error")
        if self.tolerances.maximum_mean_absolute_cell_error is not None and (
            mean_error is None or mean_error > self.tolerances.maximum_mean_absolute_cell_error
        ):
            failing.append("mean_absolute_cell_error")
        if order["agreement"] is None or order["agreement"] < self.tolerances.minimum_pairwise_order_agreement:
            failing.append("pairwise_order_agreement")
        if order["compared_pairs"] < self.tolerances.minimum_compared_pairs:
            failing.append("sufficient_compared_pairs")
        if len(cell_rows) < self.tolerances.minimum_evaluated_cells:
            failing.append("sufficient_evaluated_cells")

        all_failing = tuple(dict.fromkeys(disqualified + failing))
        evaluated = replace(
            point,
            panel=panel,
            coverage=point.coverage if point.coverage is not None else panel_coverage,
            per_cell_absolute_error=per_cell_error,
            max_absolute_cell_error=max_error,
            mean_absolute_cell_error=mean_error,
            pairwise_order_agreement=order["agreement"],
            compared_pairs=order["compared_pairs"],
            evaluated_cells=len(cell_rows),
            tolerances_met=not all_failing,
            failing_tolerances=() if not all_failing else all_failing,
        )
        detail = {
            "operating_point_id": point.operating_point_id,
            "panel": panel,
            "status": "disqualified" if disqualified else "meets_tolerances" if not failing else "fails_tolerances",
            "disqualifying_violations": disqualified,
            "failing_tolerances": list(all_failing),
            "panel_n": panel_n,
            "panel_valid": panel_valid,
            "panel_coverage": panel_coverage,
            "cells_below_minimum_coverage": coverage_failures,
            "horizon_mismatch": horizon_mismatch,
            "unexpected_horizon_tasks": extra_tasks,
            "cells_with_mismatched_starts": mismatched_starts,
            "cells": cell_rows,
            "pairwise_order": order,
        }
        return evaluated, detail

    # -- arms ----------------------------------------------------------------

    def add_point(
        self,
        point: SweepPoint,
        outcomes: Any,
        lineages: Optional[Mapping[str, str]] = None,
        cost_model: Optional[GpuSecondModel] = None,
    ) -> SweepPoint:
        """Score one sweep arm on the development panel and retain it forever."""

        if not isinstance(point, SweepPoint):
            raise SweepError("point must be a SweepPoint")
        if any(existing.operating_point_id == point.operating_point_id for existing in self._points):
            raise SweepError("duplicate operating_point_id: {0}".format(point.operating_point_id))
        if cost_model is not None:
            point = _apply_cost_model(point, cost_model)
            self._cost_views[point.operating_point_id] = {
                view: cost_model.view(view) for view in COST_VIEWS
            }
        parsed = _parse_outcome_vectors(outcomes, "outcomes for {0}".format(point.operating_point_id))
        evaluated, detail = self._evaluate(point, parsed, DEVELOPMENT_PANEL)
        self._points.append(evaluated)
        self._details[evaluated.operating_point_id] = detail
        self._lineages[evaluated.operating_point_id] = _lineage_mapping(
            lineages, "lineages for {0}".format(point.operating_point_id)
        )
        return evaluated

    @property
    def points(self) -> Tuple[SweepPoint, ...]:
        return tuple(self._points)

    # -- selection -----------------------------------------------------------

    def select(self) -> Dict[str, Any]:
        """Cheapest development arm meeting every preregistered tolerance."""

        development = [point for point in self._points if point.panel == DEVELOPMENT_PANEL]
        eligible = [point for point in development if point.tolerances_met is True]
        rejected = [
            {
                "operating_point_id": point.operating_point_id,
                "failing_tolerances": list(point.failing_tolerances),
                "status": self._details[point.operating_point_id]["status"],
                "estimated_usd": point.estimated_usd,
                "measured_gpu_seconds": point.measured_gpu_seconds,
                "task_horizon_actions": dict(sorted(point.task_horizon_actions.items())),
                "coverage": point.coverage,
            }
            for point in development
            if point.tolerances_met is not True
        ]
        base = {
            "panel": DEVELOPMENT_PANEL,
            "cost_view": SELECTION_COST_VIEW,
            "evaluated_operating_point_ids": [point.operating_point_id for point in development],
            "eligible_operating_point_ids": [point.operating_point_id for point in eligible],
            "rejected": rejected,
            "confirmation_status": "unconfirmed",
            "qualified_claim_allowed": False,
            "qualified_claim_block_reason": (
                "A development-panel selection is unconfirmed until the same setting is reproduced inside "
                "tolerance on a disjoint confirmation panel."
            ),
            "gate_invalidation_key": "operating_point",
            "note": (
                "Selection compares only the total demonstration-run cost view; the marginal estimate may omit "
                "idle capacity and never establishes that one setting is cheaper."
            ),
        }
        if not eligible:
            return {
                **base,
                "status": "unavailable",
                "reason": "No development arm met every preregistered tolerance.",
                "operating_point_id": None,
                "cost_metric": None,
                "cost": None,
            }
        if all(point.estimated_usd is not None for point in eligible):
            metric = "estimated_usd"
        elif all(point.measured_gpu_seconds is not None for point in eligible):
            metric = "measured_gpu_seconds"
        else:
            return {
                **base,
                "status": "unavailable",
                "reason": (
                    "At least one eligible arm has no comparable cost, so no arm can be called cheapest. "
                    "USD is never guessed to complete a ranking."
                ),
                "operating_point_id": None,
                "cost_metric": None,
                "cost": None,
                "operating_points_without_comparable_cost": [
                    point.operating_point_id
                    for point in eligible
                    if point.estimated_usd is None and point.measured_gpu_seconds is None
                ],
            }
        costs = {point.operating_point_id: float(getattr(point, metric)) for point in eligible}
        cheapest = min(costs.values())
        tied = sorted(identifier for identifier, cost in costs.items() if cost == cheapest)
        selected_id = tied[0]
        selected = next(point for point in eligible if point.operating_point_id == selected_id)
        return {
            **base,
            "status": "selected",
            "reason": None,
            "operating_point_id": selected_id,
            "cost_metric": metric,
            "cost": costs[selected_id],
            "cost_ranking": dict(sorted(costs.items(), key=lambda item: (item[1], item[0]))),
            "tied_operating_point_ids": tied,
            "tie_break": "lowest operating_point_id; a tie is not a measured difference" if len(tied) > 1 else None,
            "usd_available": metric == "estimated_usd",
            "operating_point": selected.as_dict(),
        }

    # -- confirmation --------------------------------------------------------

    def confirm_on_disjoint_panel(
        self,
        candidate_point: SweepPoint,
        candidate_outcomes: Any,
        reference_point: SweepPoint,
        reference_outcomes: Any,
        lineages: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        """Confirm or reject the development selection on a disjoint panel."""

        selection = self.select()
        if selection["status"] != "selected":
            return {
                "status": "unavailable",
                "reason": "There is no confirmed-eligible development selection to confirm.",
                "selection_status": selection["status"],
                "panel": CONFIRMATION_PANEL,
            }
        if not isinstance(candidate_point, SweepPoint) or not isinstance(reference_point, SweepPoint):
            raise SweepError("confirmation points must be SweepPoint objects")
        selected = next(
            point for point in self._points if point.operating_point_id == selection["operating_point_id"]
        )
        if candidate_point.configuration_key != selected.configuration_key:
            return {
                "status": "rejected",
                "reason": "The confirmation arm does not use the selected setting.",
                "failing_tolerances": ["configuration_matches_selection"],
                "panel": CONFIRMATION_PANEL,
                "selected_operating_point_id": selected.operating_point_id,
            }
        confirmation_lineages = _lineage_mapping(lineages, "confirmation lineages")
        development_lineages = {
            lineage for mapping in self._lineages.values() for lineage in mapping.values()
        } | set(self.reference_lineages.values())
        disjointness: Dict[str, Any]
        if not confirmation_lineages or not development_lineages:
            disjointness = {
                "status": "unverifiable",
                "reason": (
                    "Source lineage identifiers are required on both panels before a confirmation panel can be "
                    "called disjoint."
                ),
                "shared_lineages": [],
            }
        else:
            shared = sorted(set(confirmation_lineages.values()) & development_lineages)
            disjointness = {
                "status": "fail" if shared else "pass",
                "shared_lineages": shared,
                "development_lineage_count": len(development_lineages),
                "confirmation_lineage_count": len(set(confirmation_lineages.values())),
            }
        if disjointness["status"] != "pass":
            return {
                "status": "unavailable",
                "reason": (
                    "A cost-confirmation panel must not share a source lineage with the development sweep."
                    if disjointness["status"] == "fail"
                    else disjointness["reason"]
                ),
                "panel_disjointness": disjointness,
                "panel": CONFIRMATION_PANEL,
                "selected_operating_point_id": selected.operating_point_id,
            }
        if reference_point.configuration_key != self.full_fidelity_point.configuration_key:
            return {
                "status": "rejected",
                "reason": "The confirmation panel's reference arm is not the full-fidelity setting.",
                "failing_tolerances": ["configuration_matches_full_fidelity"],
                "panel": CONFIRMATION_PANEL,
                "selected_operating_point_id": selected.operating_point_id,
            }
        self._confirmation_reference = _parse_outcome_vectors(reference_outcomes, "confirmation reference outcomes")
        self._confirmation_reference_stats = {
            cell: _cell_stats(vector) for cell, vector in self._confirmation_reference.items()
        }
        self._confirmation_reference_point = replace(reference_point, panel=CONFIRMATION_PANEL)
        parsed = _parse_outcome_vectors(candidate_outcomes, "confirmation candidate outcomes")
        evaluated, detail = self._evaluate(candidate_point, parsed, CONFIRMATION_PANEL)
        confirmed = evaluated.tolerances_met is True
        result = {
            "status": "confirmed" if confirmed else "rejected",
            "reason": None if confirmed else "The selected setting did not hold inside tolerance on the disjoint panel.",
            "panel": CONFIRMATION_PANEL,
            "selected_operating_point_id": selected.operating_point_id,
            "confirmation_operating_point": evaluated.as_dict(),
            "failing_tolerances": list(evaluated.failing_tolerances),
            "detail": detail,
            "panel_disjointness": disjointness,
            "reference_operating_point_id": reference_point.operating_point_id,
            "note": (
                "Confirmation repeats the preregistered tolerances against a full-fidelity reference measured on "
                "the same disjoint panel; one matching ranking is not itself evidence of fidelity."
            ),
        }
        self._confirmation = result
        return result

    def qualified_claim(self) -> Dict[str, Any]:
        """Return the operating point usable for a qualified claim, if any."""

        selection = self.select()
        if selection["status"] != "selected":
            return {
                "status": "blocked",
                "reason": selection.get("reason", "No development selection is available."),
                "operating_point": None,
            }
        if self._confirmation is None:
            return {
                "status": "blocked",
                "reason": (
                    "The selected operating point is unconfirmed: a disjoint confirmation panel has not been "
                    "evaluated, so it must not be used for a qualified claim."
                ),
                "operating_point": None,
                "operating_point_id": selection["operating_point_id"],
                "confirmation_status": "unconfirmed",
            }
        if self._confirmation["status"] != "confirmed":
            return {
                "status": "blocked",
                "reason": "The disjoint confirmation panel rejected the selected operating point.",
                "operating_point": None,
                "operating_point_id": selection["operating_point_id"],
                "confirmation_status": self._confirmation["status"],
                "failing_tolerances": self._confirmation.get("failing_tolerances", []),
            }
        payload = dict(selection["operating_point"])
        payload["qualified"] = True
        return {
            "status": "allowed",
            "reason": None,
            "operating_point": payload,
            "operating_point_id": selection["operating_point_id"],
            "confirmation_status": "confirmed",
        }

    # -- reporting -----------------------------------------------------------

    def cost_ledger(self) -> Dict[str, Any]:
        """Preserve every arm's cost, including the arms that failed."""

        rows = [
            {
                "operating_point_id": point.operating_point_id,
                "panel": point.panel,
                "status": self._details[point.operating_point_id]["status"],
                "measured_gpu_seconds": point.measured_gpu_seconds,
                "estimated_usd": point.estimated_usd,
                "usd_status": point.usd_status,
                "cost_basis_ref": point.cost_basis_ref,
                "cost_views": self._cost_views.get(point.operating_point_id),
            }
            for point in self._points
        ]
        known_gpu = [row["measured_gpu_seconds"] for row in rows if row["measured_gpu_seconds"] is not None]
        known_usd = [row["estimated_usd"] for row in rows if row["estimated_usd"] is not None]
        unknown_gpu = [row["operating_point_id"] for row in rows if row["measured_gpu_seconds"] is None]
        unknown_usd = [row["operating_point_id"] for row in rows if row["estimated_usd"] is None]
        return {
            "rows": rows,
            "sweep_gpu_seconds": {
                "known_sum": sum(known_gpu) if known_gpu else None,
                "status": "complete" if not unknown_gpu else "partial",
                "operating_points_without_gpu_seconds": unknown_gpu,
            },
            "sweep_estimated_usd": {
                "known_sum": sum(known_usd) if known_usd else None,
                "status": "complete" if not unknown_usd else "partial",
                "operating_points_without_usd": unknown_usd,
            },
            "note": "Failed and disqualified settings keep their costs; the sweep's own spend is not discarded.",
        }

    def report(self) -> Dict[str, Any]:
        selection = self.select()
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "computed",
            "measurement": "cost_fidelity",
            "comparison_target": (
                "the full-fidelity operating point's own measured cells, not the published human reference table"
            ),
            "endpoint": "observed_positive_lower_bound",
            "tolerances": self.tolerances.as_dict(),
            "full_fidelity_reference": {
                "operating_point": self.full_fidelity_point.as_dict(),
                "cells": [
                    {
                        "cell": "{0}/{1}".format(*cell),
                        "n": stats["n"],
                        "valid": stats["valid"],
                        "successes": stats["successes"],
                        "observed_positive_rate": stats["observed_positive_rate"],
                        "coverage": stats["coverage"],
                    }
                    for cell, stats in sorted(self.reference_stats.items())
                ],
            },
            "matrix": {"policies": list(self.policies), "tasks": list(self.tasks)},
            "points": [point.as_dict() for point in self._points],
            "point_detail": [self._details[point.operating_point_id] for point in self._points],
            "cost_ledger": self.cost_ledger(),
            "selection": selection,
            "confirmation": self._confirmation
            or {
                "status": "not_computed",
                "reason": "No disjoint confirmation panel has been supplied.",
                "panel": CONFIRMATION_PANEL,
            },
            "qualified_claim": self.qualified_claim(),
        }


def _apply_cost_model(point: SweepPoint, cost_model: GpuSecondModel) -> SweepPoint:
    if not isinstance(cost_model, GpuSecondModel):
        raise SweepError("cost_model must be a GpuSecondModel")
    view = cost_model.view(SELECTION_COST_VIEW)
    usd = view["estimated_usd"]
    basis = None
    if usd is not None:
        basis = point.cost_basis_ref or (view["price_basis"] or {}).get("basis_id")
    reason = None
    if usd is None:
        reasons = sorted({str(item.get("reason")) for item in view["usd_unavailable_reasons"]})
        reason = "; ".join(reasons) if reasons else "The price basis is unavailable."
    # Instrumented compute seconds and the allocation ledger's GPU-seconds are
    # separate telemetry: the allocated figure is never relabelled as measured.
    gpu_seconds = view["measured_compute_gpu_seconds"]
    return replace(
        point,
        estimated_usd=usd,
        cost_basis_ref=basis,
        usd_unavailable_reason=reason,
        measured_gpu_seconds=point.measured_gpu_seconds if point.measured_gpu_seconds is not None else gpu_seconds,
    )


def _lineage_mapping(value: Any, field_name: str) -> Dict[str, str]:
    if value is None:
        return {}
    mapping = _mapping(value, field_name)
    return {
        _nonempty_string(start, "{0} start identifier".format(field_name)): _nonempty_string(
            lineage, "{0} lineage identifier".format(field_name)
        )
        for start, lineage in mapping.items()
    }


def build_cost_fidelity_report(artifacts: Any) -> Dict[str, Any]:
    """Build a cost-fidelity report from JSON artifacts, or explain the refusal."""

    try:
        data = dict(_mapping(artifacts, "cost_fidelity artifacts"))
        tolerances = SweepTolerances.from_mapping(data.get("tolerances", {}))
        reference = dict(_mapping(data.get("full_fidelity_reference", {}), "full_fidelity_reference"))
        sweep = CostFidelitySweep(
            SweepPoint.from_mapping(reference.get("operating_point", {})),
            reference.get("outcomes", {}),
            tolerances,
            lineages=reference.get("lineages"),
        )
        raw_points = data.get("points", ())
        if isinstance(raw_points, (str, bytes)) or not isinstance(raw_points, (list, tuple)):
            raise SweepError("cost_fidelity points must be a list")
        if not raw_points:
            raise SweepError("cost_fidelity requires at least one swept operating point")
        for entry in raw_points:
            row = dict(_mapping(entry, "sweep point entry"))
            cost_model = row.get("cost_model")
            sweep.add_point(
                SweepPoint.from_mapping(row.get("operating_point", {})),
                row.get("outcomes", {}),
                lineages=row.get("lineages"),
                cost_model=None if cost_model is None else GpuSecondModel.from_mapping(cost_model),
            )
        confirmation = data.get("confirmation_panel")
        if confirmation is not None:
            panel = dict(_mapping(confirmation, "confirmation_panel"))
            sweep.confirm_on_disjoint_panel(
                SweepPoint.from_mapping(panel.get("candidate_operating_point", {})),
                panel.get("candidate_outcomes", {}),
                SweepPoint.from_mapping(panel.get("reference_operating_point", {})),
                panel.get("reference_outcomes", {}),
                lineages=panel.get("lineages"),
            )
        return sweep.report()
    except SweepError as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "rejected",
            "measurement": "cost_fidelity",
            "reason": str(error),
        }


# ---------------------------------------------------------------------------
# Drift (spec section 6.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftPoint:
    """One drift horizon on the control-tick axis.

    The four-tuple ``(control ticks, FD-request action length, autoregressive
    boundaries, policy feedback boundaries)`` and the backend profile identity
    are mandatory: without them a horizon number cannot be interpreted, and the
    1--6-request display cannot be labelled with the backend and cadence it
    belongs to.  ``control_ticks`` -- never a request count -- is the x-axis.
    """

    backend_profile_id: str
    control_ticks: int
    fd_request_action_length: int
    autoregressive_boundaries: Tuple[int, ...]
    policy_feedback_boundaries: Tuple[int, ...]
    nominal_elapsed_seconds: Optional[float] = None
    control_hz: Optional[float] = None
    fd_requests: Optional[int] = None
    task: Optional[str] = None
    feedback_mode: Optional[str] = None
    mean_action_error: Optional[float] = None
    action_error_metric: Optional[str] = None
    integrity_intact_fraction: Optional[float] = None
    mean_visual_divergence: Optional[float] = None
    visual_divergence_metric: Optional[str] = None
    n_sequences: Optional[int] = None

    def __post_init__(self) -> None:
        if self.backend_profile_id is None:
            raise SweepError("a DriftPoint requires its backend_profile_id")
        _nonempty_string(self.backend_profile_id, "backend_profile_id")
        for field_name in ("control_ticks", "fd_request_action_length"):
            if getattr(self, field_name) is None:
                raise SweepError("a DriftPoint requires {0}".format(field_name))
            _positive_int(getattr(self, field_name), field_name)
        if self.fd_request_action_length > self.control_ticks:
            raise SweepError("fd_request_action_length must not exceed control_ticks")
        for field_name in ("autoregressive_boundaries", "policy_feedback_boundaries"):
            if getattr(self, field_name) is None:
                raise SweepError("a DriftPoint requires {0}".format(field_name))
            object.__setattr__(
                self, field_name, _boundaries(getattr(self, field_name), field_name, self.control_ticks)
            )
        object.__setattr__(
            self, "nominal_elapsed_seconds", _optional_nonnegative(self.nominal_elapsed_seconds, "nominal_elapsed_seconds")
        )
        if self.control_hz is not None:
            value = _nonnegative(self.control_hz, "control_hz")
            if value <= 0.0:
                raise SweepError("control_hz must be greater than zero")
        if self.fd_requests is not None:
            _positive_int(self.fd_requests, "fd_requests")
        if self.task is not None:
            object.__setattr__(self, "task", canonical_task(self.task))
        for field_name in ("feedback_mode", "action_error_metric", "visual_divergence_metric"):
            value = getattr(self, field_name)
            if value is not None:
                _nonempty_string(value, "drift point {0}".format(field_name))
        object.__setattr__(
            self, "mean_action_error", _optional_nonnegative(self.mean_action_error, "mean_action_error")
        )
        object.__setattr__(
            self,
            "integrity_intact_fraction",
            _optional_probability(self.integrity_intact_fraction, "integrity_intact_fraction"),
        )
        object.__setattr__(
            self, "mean_visual_divergence", _optional_nonnegative(self.mean_visual_divergence, "mean_visual_divergence")
        )
        if self.mean_action_error is not None and self.action_error_metric is None:
            raise SweepError("mean_action_error requires a named action_error_metric")
        if self.mean_visual_divergence is not None and self.visual_divergence_metric is None:
            raise SweepError("mean_visual_divergence requires a named visual_divergence_metric")
        if self.n_sequences is not None:
            _positive_int(self.n_sequences, "n_sequences")

    @property
    def four_tuple(self) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]:
        return (
            self.control_ticks,
            self.fd_request_action_length,
            self.autoregressive_boundaries,
            self.policy_feedback_boundaries,
        )

    @property
    def partition_count(self) -> int:
        return len(self.autoregressive_boundaries)

    @property
    def policy_feedback_cadence_ticks(self) -> Optional[int]:
        """The uniform feedback spacing in control ticks, or ``None`` if it varies."""

        edges = (0,) + self.policy_feedback_boundaries
        spacings = {second - first for first, second in zip(edges, edges[1:])}
        return spacings.pop() if len(spacings) == 1 else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend_profile_id": self.backend_profile_id,
            "x_axis": "control_ticks",
            "control_ticks": self.control_ticks,
            "fd_request_action_length": self.fd_request_action_length,
            "autoregressive_boundaries": list(self.autoregressive_boundaries),
            "policy_feedback_boundaries": list(self.policy_feedback_boundaries),
            "policy_feedback_cadence_ticks": self.policy_feedback_cadence_ticks,
            "partition_count": self.partition_count,
            "nominal_elapsed_seconds": self.nominal_elapsed_seconds,
            "control_hz": self.control_hz,
            "fd_requests": self.fd_requests,
            "task": self.task,
            "feedback_mode": self.feedback_mode,
            "mean_action_error": self.mean_action_error,
            "action_error_metric": self.action_error_metric,
            "integrity_intact_fraction": self.integrity_intact_fraction,
            "mean_visual_divergence": self.mean_visual_divergence,
            "visual_divergence_metric": self.visual_divergence_metric,
            "n_sequences": self.n_sequences,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DriftPoint":
        data = dict(_mapping(payload, "drift point"))
        for derived in ("x_axis", "partition_count", "policy_feedback_cadence_ticks"):
            data.pop(derived, None)
        allowed = {
            "backend_profile_id",
            "control_ticks",
            "fd_request_action_length",
            "autoregressive_boundaries",
            "policy_feedback_boundaries",
            "nominal_elapsed_seconds",
            "control_hz",
            "fd_requests",
            "task",
            "feedback_mode",
            "mean_action_error",
            "action_error_metric",
            "integrity_intact_fraction",
            "mean_visual_divergence",
            "visual_divergence_metric",
            "n_sequences",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise SweepError("unknown drift point fields: {0}".format(unknown))
        for required in ("backend_profile_id", "control_ticks", "fd_request_action_length", "autoregressive_boundaries", "policy_feedback_boundaries"):
            if required not in data:
                raise SweepError("a DriftPoint requires {0}".format(required))
        return cls(**data)


@dataclass(frozen=True)
class DriftTolerances:
    """Drift tolerances chosen before any horizon is tested."""

    protocol_id: str
    maximum_mean_action_error: Optional[float] = None
    action_error_metric: Optional[str] = None
    minimum_integrity_intact_fraction: Optional[float] = None
    maximum_mean_visual_divergence: Optional[float] = None
    visual_divergence_metric: Optional[str] = None

    def __post_init__(self) -> None:
        _nonempty_string(self.protocol_id, "drift tolerances protocol_id")
        object.__setattr__(
            self,
            "maximum_mean_action_error",
            _optional_nonnegative(self.maximum_mean_action_error, "maximum_mean_action_error"),
        )
        object.__setattr__(
            self,
            "minimum_integrity_intact_fraction",
            _optional_probability(self.minimum_integrity_intact_fraction, "minimum_integrity_intact_fraction"),
        )
        object.__setattr__(
            self,
            "maximum_mean_visual_divergence",
            _optional_nonnegative(self.maximum_mean_visual_divergence, "maximum_mean_visual_divergence"),
        )
        for field_name in ("action_error_metric", "visual_divergence_metric"):
            value = getattr(self, field_name)
            if value is not None:
                _nonempty_string(value, "drift tolerances {0}".format(field_name))
        if self.maximum_mean_action_error is not None and self.action_error_metric is None:
            raise SweepError("maximum_mean_action_error requires a named action_error_metric")
        if self.maximum_mean_visual_divergence is not None and self.visual_divergence_metric is None:
            raise SweepError("maximum_mean_visual_divergence requires a named visual_divergence_metric")
        if (
            self.maximum_mean_action_error is None
            and self.minimum_integrity_intact_fraction is None
            and self.maximum_mean_visual_divergence is None
        ):
            raise SweepError("drift tolerances must constrain at least one measured quantity")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "maximum_mean_action_error": self.maximum_mean_action_error,
            "action_error_metric": self.action_error_metric,
            "minimum_integrity_intact_fraction": self.minimum_integrity_intact_fraction,
            "maximum_mean_visual_divergence": self.maximum_mean_visual_divergence,
            "visual_divergence_metric": self.visual_divergence_metric,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DriftTolerances":
        data = dict(_mapping(payload, "drift tolerances"))
        allowed = {
            "protocol_id",
            "maximum_mean_action_error",
            "action_error_metric",
            "minimum_integrity_intact_fraction",
            "maximum_mean_visual_divergence",
            "visual_divergence_metric",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise SweepError("unknown drift tolerance fields: {0}".format(unknown))
        try:
            return cls(**data)
        except TypeError as error:
            raise SweepError("drift tolerances are missing a required field: {0}".format(error)) from error


def _action_matrix(value: Any, field_name: str) -> Tuple[Tuple[float, ...], ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)) or not value:
        raise SweepError("{0} must be a non-empty list of action vectors".format(field_name))
    rows: List[Tuple[float, ...]] = []
    for index, row in enumerate(value):
        if isinstance(row, (str, bytes, Mapping)) or not isinstance(row, (list, tuple)) or not row:
            raise SweepError("{0}[{1}] must be a non-empty numeric action vector".format(field_name, index))
        rows.append(tuple(_finite(item, "{0}[{1}] component".format(field_name, index)) for item in row))
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise SweepError("{0} rows must all have the same action dimension".format(field_name))
    return tuple(rows)


def _action_error(
    real: Sequence[Sequence[float]], predicted: Sequence[Sequence[float]]
) -> Dict[str, Any]:
    per_tick = [math.dist(actual, forecast) for actual, forecast in zip(real, predicted)]
    absolute = [
        abs(actual_value - forecast_value)
        for actual, forecast in zip(real, predicted)
        for actual_value, forecast_value in zip(actual, forecast)
    ]
    return {
        "mean_per_tick_l2": _safe_mean(per_tick),
        "max_per_tick_l2": max(per_tick) if per_tick else None,
        "mean_absolute_error": _safe_mean(absolute),
        "n_ticks": len(per_tick),
        "action_dim": len(real[0]) if real else None,
        "metric": "euclidean_distance_per_control_tick_on_raw_7d_actions",
    }


def _divergence_series(value: Any, expected_ticks: int) -> Dict[str, Any]:
    """Read a supplied visual-divergence series; never invent one."""

    if value is None:
        return {
            "status": "unavailable",
            "reason": "No visual-divergence series was supplied; a frame distance is not fabricated.",
        }
    data = dict(_mapping(value, "visual_divergence"))
    metric = data.get("metric")
    if metric is None or not str(metric).strip():
        return {
            "status": "unavailable",
            "reason": "A visual-divergence series without a named metric cannot be interpreted.",
        }
    units = data.get("units")
    arms: Dict[str, Any] = {}
    for arm in ("teacher_forced", "free_running"):
        series = data.get(arm)
        if series is None:
            return {"status": "unavailable", "reason": "visual_divergence is missing the {0} arm.".format(arm)}
        if isinstance(series, (str, bytes, Mapping)) or not isinstance(series, (list, tuple)) or not series:
            raise SweepError("visual_divergence[{0}] must be a non-empty numeric series".format(arm))
        values = [_nonnegative(item, "visual_divergence[{0}] value".format(arm)) for item in series]
        if len(values) != expected_ticks:
            return {
                "status": "unavailable",
                "reason": "visual_divergence[{0}] covers {1} ticks, not the {2} evaluated ticks.".format(
                    arm, len(values), expected_ticks
                ),
            }
        arms[arm] = {"mean": _safe_mean(values), "max": max(values), "n_ticks": len(values)}
    gap = arms["free_running"]["mean"] - arms["teacher_forced"]["mean"]
    return {
        "status": "computed",
        "metric": str(metric).strip(),
        "units": None if units is None else str(units),
        "teacher_forced": arms["teacher_forced"],
        "free_running": arms["free_running"],
        "free_running_minus_teacher_forced_mean": gap,
    }


def _integrity_label(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    label = _nonempty_string(value, field_name).lower()
    if label not in INTEGRITY_LABELS:
        raise SweepError("{0} must be one of {1}".format(field_name, list(INTEGRITY_LABELS)))
    return label


class DriftStudy:
    """Drift on the control-tick axis, with the request display kept in its place."""

    def __init__(
        self,
        backend_profile_id: str,
        points: Sequence[DriftPoint] = (),
        tolerances: Optional[DriftTolerances] = None,
        sequences: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.backend_profile_id = _nonempty_string(backend_profile_id, "backend_profile_id")
        self.points: Tuple[DriftPoint, ...] = tuple(points)
        for point in self.points:
            if not isinstance(point, DriftPoint):
                raise SweepError("points must contain DriftPoint objects")
            if point.backend_profile_id != self.backend_profile_id:
                raise SweepError(
                    "a drift study cannot mix backend profiles: {0} != {1}".format(
                        point.backend_profile_id, self.backend_profile_id
                    )
                )
        horizons = [point.control_ticks for point in self.points]
        if len(set(horizons)) != len(horizons):
            raise SweepError("each drift point must test a distinct control-tick horizon")
        if tolerances is not None and not isinstance(tolerances, DriftTolerances):
            raise SweepError("tolerances must be a DriftTolerances preregistration")
        self.tolerances = tolerances
        self.sequences: Tuple[Mapping[str, Any], ...] = tuple(sequences)

    # -- axis ----------------------------------------------------------------

    def axis(self) -> Dict[str, Any]:
        rows = [point.as_dict() for point in sorted(self.points, key=lambda item: item.control_ticks)]
        elapsed = [point.nominal_elapsed_seconds for point in self.points]
        return {
            "x_axis": "control_ticks",
            "x_axis_alternative": "nominal_elapsed_seconds" if self.points and all(value is not None for value in elapsed) else None,
            "request_count_is_not_an_axis": True,
            "backend_profile_id": self.backend_profile_id,
            "tested_horizons": sorted(point.control_ticks for point in self.points),
            "rows": rows,
            "note": (
                "Nominal elapsed time is derived from the control clock, not measured physical duration; a request "
                "count is never the drift x-axis."
            ),
        }

    def request_display(self) -> Dict[str, Any]:
        """The 1--6-request display, only with backend, action length and cadence."""

        if not self.points:
            return {"status": "unavailable", "reason": "No drift points were supplied.", "rows": []}
        missing: List[Dict[str, Any]] = []
        for point in self.points:
            gaps = []
            if point.fd_requests is None:
                gaps.append("fd_requests")
            if point.policy_feedback_cadence_ticks is None:
                gaps.append("policy_feedback_cadence_ticks")
            if gaps:
                missing.append({"control_ticks": point.control_ticks, "missing": gaps})
        if missing:
            return {
                "status": "unavailable",
                "reason": (
                    "The 1-6-request display may only be shown with its backend, FD-request action length and "
                    "feedback cadence stated."
                ),
                "incomplete_points": missing,
                "rows": [],
            }
        out_of_range = sorted({point.fd_requests for point in self.points if not 1 <= int(point.fd_requests) <= 6})
        inconsistent = [
            {
                "control_ticks": point.control_ticks,
                "fd_requests": point.fd_requests,
                "autoregressive_boundaries": len(point.autoregressive_boundaries),
            }
            for point in self.points
            if point.fd_requests != point.partition_count
        ]
        return {
            "status": "available",
            "backend_profile_id": self.backend_profile_id,
            "rows": [
                {
                    "fd_requests": point.fd_requests,
                    "control_ticks": point.control_ticks,
                    "fd_request_action_length": point.fd_request_action_length,
                    "policy_feedback_cadence_ticks": point.policy_feedback_cadence_ticks,
                    "feedback_mode": point.feedback_mode,
                }
                for point in sorted(self.points, key=lambda item: item.control_ticks)
            ],
            "requests_outside_one_to_six": list(out_of_range),
            "fd_request_count_disagrees_with_autoregressive_boundaries": inconsistent,
            "is_common_physical_duration_across_policies": False,
            "interpretation": "backend_specific_transport_and_horizon_sensitivity",
            "caveat": (
                "This display is backend-specific transport/horizon sensitivity, not a common physical duration "
                "across policies, and it is not the drift x-axis."
            ),
        }

    # -- chunking guard ------------------------------------------------------

    def partition_sweep_classification(self) -> Dict[str, Any]:
        """Label a partition sweep chunking-sensitivity unless chunk length is held."""

        if len(self.points) < 2:
            return {
                "classification": "insufficient_points",
                "reason": "At least two horizons are required before a sweep can be classified.",
                "isolated_drift": False,
            }
        chunk_lengths = sorted({point.fd_request_action_length for point in self.points})
        horizons = sorted({point.control_ticks for point in self.points})
        partitions = sorted({point.partition_count for point in self.points})
        chunk_varies = len(chunk_lengths) > 1
        horizon_varies = len(horizons) > 1
        if chunk_varies:
            classification = "chunking_sensitivity"
            reason = (
                "The FD-request chunk length changes across the swept points, so this measures sensitivity to "
                "chunking, not isolated drift."
            )
        elif horizon_varies:
            classification = "isolated_drift"
            reason = "Chunk length is held constant while the control-tick horizon varies."
        else:
            classification = "no_variation"
            reason = "Neither the chunk length nor the control-tick horizon varies."
        return {
            "classification": classification,
            "isolated_drift": classification == "isolated_drift",
            "reason": reason,
            "chunk_action_lengths": chunk_lengths,
            "control_tick_horizons": horizons,
            "partition_counts": partitions,
            "chunk_length_varies": chunk_varies,
            "horizon_varies": horizon_varies,
        }

    # -- teacher forcing -----------------------------------------------------

    def teacher_forced_vs_free_running(
        self, sequences: Optional[Sequence[Mapping[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Compare the two arms at equal control prefixes on held-out real data."""

        candidates = tuple(sequences if sequences is not None else self.sequences)
        if not candidates:
            return {
                "status": "not_computed",
                "reason": "No held-out real action/video sequences were supplied.",
                "sequences": [],
            }
        rows: List[Dict[str, Any]] = []
        for index, raw in enumerate(candidates):
            rows.append(self._compare_sequence(raw, index))
        usable = [row for row in rows if row["status"] == "computed"]
        by_prefix: Dict[int, List[Dict[str, Any]]] = {}
        for row in usable:
            by_prefix.setdefault(int(row["control_prefix_ticks"]), []).append(row)
        prefixes = [
            {
                "control_prefix_ticks": prefix,
                "n_sequences": len(group),
                "evaluated_ticks": sorted({row["evaluated_ticks"] for row in group}),
                "teacher_forced": {
                    "mean_per_tick_l2": _safe_mean([row["teacher_forced"]["mean_per_tick_l2"] for row in group]),
                    "mean_absolute_error": _safe_mean([row["teacher_forced"]["mean_absolute_error"] for row in group]),
                },
                "free_running": {
                    "mean_per_tick_l2": _safe_mean([row["free_running"]["mean_per_tick_l2"] for row in group]),
                    "mean_absolute_error": _safe_mean([row["free_running"]["mean_absolute_error"] for row in group]),
                },
                "free_running_minus_teacher_forced_mean_per_tick_l2": _safe_mean(
                    [row["free_running_minus_teacher_forced"]["mean_per_tick_l2"] for row in group]
                ),
                "integrity": _integrity_summary(group),
                "visual_divergence": _visual_summary(group),
            }
            for prefix, group in sorted(by_prefix.items())
        ]
        return {
            "status": "computed" if usable else "unavailable",
            "reason": None if usable else "No supplied sequence satisfied the equal-prefix held-out requirements.",
            "comparison": "teacher_forced_vs_free_running",
            "x_axis": "control_ticks",
            "n_supplied": len(rows),
            "n_used": len(usable),
            "n_excluded": len(rows) - len(usable),
            "per_prefix": prefixes,
            "sequences": rows,
            "note": (
                "Aggregates are grouped by control prefix: results from different prefixes are never averaged into "
                "one headline, and the same actions and generation settings are required on both arms."
            ),
        }

    def _compare_sequence(self, raw: Any, index: int) -> Dict[str, Any]:
        data = dict(_mapping(raw, "held-out sequence {0}".format(index)))
        sequence_id = data.get("sequence_id")
        identity = str(sequence_id).strip() if sequence_id is not None else "index:{0}".format(index)
        if data.get("held_out") is not True:
            return {
                "sequence_id": identity,
                "status": "excluded",
                "reason": "Only sequences explicitly labelled held_out=true may be compared.",
            }
        prefix = data.get("control_prefix_ticks")
        if prefix is None or isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 1:
            return {
                "sequence_id": identity,
                "status": "excluded",
                "reason": "A comparison needs a positive integer control_prefix_ticks.",
            }
        settings = data.get("generation_settings_hash")
        teacher_settings = data.get("teacher_forced_generation_settings_hash", settings)
        free_settings = data.get("free_running_generation_settings_hash", settings)
        if teacher_settings is None or free_settings is None:
            return {
                "sequence_id": identity,
                "status": "excluded",
                "reason": "Both arms must state their generation settings hash.",
            }
        if str(teacher_settings) != str(free_settings):
            return {
                "sequence_id": identity,
                "status": "excluded",
                "reason": "The two arms used different generation settings, so they are not comparable.",
            }
        real = _action_matrix(data.get("real_actions"), "real_actions")
        teacher = _action_matrix(data.get("teacher_forced_actions"), "teacher_forced_actions")
        free = _action_matrix(data.get("free_running_actions"), "free_running_actions")
        if not len(real) == len(teacher) == len(free):
            return {
                "sequence_id": identity,
                "status": "excluded",
                "reason": "Both arms must cover exactly the real sequence's control ticks.",
            }
        if not len(real[0]) == len(teacher[0]) == len(free[0]):
            return {
                "sequence_id": identity,
                "status": "excluded",
                "reason": "Both arms must use the real sequence's action dimension.",
            }
        if prefix >= len(real):
            return {
                "sequence_id": identity,
                "status": "excluded",
                "reason": "The control prefix leaves no evaluated continuation ticks.",
            }
        teacher_error = _action_error(real[prefix:], teacher[prefix:])
        free_error = _action_error(real[prefix:], free[prefix:])
        return {
            "sequence_id": identity,
            "status": "computed",
            "source": "held_out_real",
            "control_prefix_ticks": prefix,
            "control_ticks": len(real),
            "evaluated_ticks": len(real) - prefix,
            "generation_settings_hash": str(teacher_settings),
            "teacher_forced": teacher_error,
            "free_running": free_error,
            "free_running_minus_teacher_forced": {
                "mean_per_tick_l2": free_error["mean_per_tick_l2"] - teacher_error["mean_per_tick_l2"],
                "mean_absolute_error": free_error["mean_absolute_error"] - teacher_error["mean_absolute_error"],
            },
            "integrity": {
                "teacher_forced": _integrity_label(data.get("teacher_forced_integrity"), "teacher_forced_integrity"),
                "free_running": _integrity_label(data.get("free_running_integrity"), "free_running_integrity"),
                "labels": list(INTEGRITY_LABELS),
            },
            "visual_divergence": _divergence_series(data.get("visual_divergence"), len(real) - prefix),
        }

    # -- horizon tolerance ---------------------------------------------------

    def last_horizon_meeting_tolerance(
        self, tolerances: Optional[DriftTolerances] = None
    ) -> Dict[str, Any]:
        """Report the last *tested* horizon meeting every preregistered tolerance."""

        active = tolerances if tolerances is not None else self.tolerances
        base = {
            "x_axis": "control_ticks",
            "unlimited_guarantee": False,
            "note": (
                "This is the last *tested* control-tick horizon meeting the preregistered tolerances. "
                "'No failure observed through H' is not an unlimited drift guarantee."
            ),
        }
        if active is None:
            return {
                **base,
                "status": "unavailable",
                "reason": "Preregistered drift tolerances are required before a horizon can be reported.",
                "last_horizon_meeting_tolerance": None,
            }
        if not self.points:
            return {
                **base,
                "status": "unavailable",
                "reason": "No drift horizons were tested.",
                "last_horizon_meeting_tolerance": None,
                "tolerances": active.as_dict(),
            }
        rows: List[Dict[str, Any]] = []
        for point in sorted(self.points, key=lambda item: item.control_ticks):
            failing: List[str] = []
            unknown: List[str] = []
            if active.maximum_mean_action_error is not None:
                if point.mean_action_error is None:
                    unknown.append("mean_action_error")
                elif point.action_error_metric != active.action_error_metric:
                    failing.append("action_error_metric_mismatch")
                elif point.mean_action_error > active.maximum_mean_action_error:
                    failing.append("mean_action_error")
            if active.minimum_integrity_intact_fraction is not None:
                if point.integrity_intact_fraction is None:
                    unknown.append("integrity_intact_fraction")
                elif point.integrity_intact_fraction < active.minimum_integrity_intact_fraction:
                    failing.append("integrity_intact_fraction")
            if active.maximum_mean_visual_divergence is not None:
                if point.mean_visual_divergence is None:
                    unknown.append("mean_visual_divergence")
                elif point.visual_divergence_metric != active.visual_divergence_metric:
                    failing.append("visual_divergence_metric_mismatch")
                elif point.mean_visual_divergence > active.maximum_mean_visual_divergence:
                    failing.append("mean_visual_divergence")
            status = "fails" if failing else "unknown" if unknown else "meets"
            rows.append(
                {
                    "control_ticks": point.control_ticks,
                    "status": status,
                    "failing": failing,
                    "unmeasured": unknown,
                    "n_sequences": point.n_sequences,
                    "mean_action_error": point.mean_action_error,
                    "integrity_intact_fraction": point.integrity_intact_fraction,
                    "mean_visual_divergence": point.mean_visual_divergence,
                }
            )
        last: Optional[int] = None
        for row in rows:
            if row["status"] != "meets":
                break
            last = int(row["control_ticks"])
        return {
            **base,
            "status": "computed",
            "reason": None,
            "last_horizon_meeting_tolerance": last,
            "last_tested_horizon": rows[-1]["control_ticks"],
            "tested_horizons": [row["control_ticks"] for row in rows],
            "failing_horizons": [row["control_ticks"] for row in rows if row["status"] == "fails"],
            "unmeasured_horizons": [row["control_ticks"] for row in rows if row["status"] == "unknown"],
            "per_horizon": rows,
            "tolerances": active.as_dict(),
            "contiguity_rule": (
                "The reported horizon is the largest tested horizon with no failing or unmeasured horizon below it."
            ),
        }

    def report(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "computed",
            "measurement": "drift",
            "backend_profile_id": self.backend_profile_id,
            "axis": self.axis(),
            "request_display": self.request_display(),
            "partition_sweep": self.partition_sweep_classification(),
            "teacher_forced_vs_free_running": self.teacher_forced_vs_free_running(),
            "horizon_tolerance": self.last_horizon_meeting_tolerance(),
        }


def _integrity_summary(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"labels": list(INTEGRITY_LABELS)}
    for arm in ("teacher_forced", "free_running"):
        labels = [row["integrity"][arm] for row in group]
        present = [label for label in labels if label is not None]
        summary[arm] = {
            "n_labelled": len(present),
            "n_unlabelled": len(labels) - len(present),
            "counts": dict(sorted(Counter(present).items())),
            "intact_fraction": (
                sum(1 for label in present if label == "intact") / float(len(present)) if present else None
            ),
        }
    return summary


def _visual_summary(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    computed = [row["visual_divergence"] for row in group if row["visual_divergence"]["status"] == "computed"]
    if not computed:
        return {
            "status": "unavailable",
            "reason": "No sequence supplied a named visual-divergence series.",
            "n_sequences": 0,
        }
    metrics = sorted({row["metric"] for row in computed})
    if len(metrics) > 1:
        return {
            "status": "unavailable",
            "reason": "Visual-divergence series use different metrics and are not aggregated: {0}".format(metrics),
            "n_sequences": len(computed),
        }
    return {
        "status": "computed",
        "metric": metrics[0],
        "n_sequences": len(computed),
        "teacher_forced_mean": _safe_mean([row["teacher_forced"]["mean"] for row in computed]),
        "free_running_mean": _safe_mean([row["free_running"]["mean"] for row in computed]),
        "free_running_minus_teacher_forced_mean": _safe_mean(
            [row["free_running_minus_teacher_forced_mean"] for row in computed]
        ),
    }


def build_drift_report(artifacts: Any) -> Dict[str, Any]:
    """Build a drift report from JSON artifacts, or explain the refusal."""

    try:
        data = dict(_mapping(artifacts, "drift artifacts"))
        raw_points = data.get("points", ())
        if isinstance(raw_points, (str, bytes)) or not isinstance(raw_points, (list, tuple)):
            raise SweepError("drift points must be a list")
        if not raw_points:
            raise SweepError("drift requires at least one control-tick horizon")
        points = [DriftPoint.from_mapping(entry) for entry in raw_points]
        backend = data.get("backend_profile_id") or points[0].backend_profile_id
        tolerances = data.get("tolerances")
        sequences = data.get("teacher_forced_vs_free_running", ())
        if isinstance(sequences, (str, bytes)) or not isinstance(sequences, (list, tuple)):
            raise SweepError("teacher_forced_vs_free_running must be a list of held-out sequences")
        study = DriftStudy(
            backend,
            points,
            tolerances=None if tolerances is None else DriftTolerances.from_mapping(tolerances),
            sequences=[dict(_mapping(entry, "held-out sequence")) for entry in sequences],
        )
        return study.report()
    except SweepError as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "rejected",
            "measurement": "drift",
            "reason": str(error),
        }


__all__ = [
    "SCHEMA_VERSION",
    "DEVELOPMENT_PANEL",
    "CONFIRMATION_PANEL",
    "MARGINAL_VIEW",
    "TOTAL_VIEW",
    "COST_VIEWS",
    "SELECTION_COST_VIEW",
    "INTEGRITY_LABELS",
    "SweepError",
    "PriceBasis",
    "ResourceRate",
    "AllocationInterval",
    "OtherCharge",
    "GpuSecondModel",
    "SweepTolerances",
    "SweepPoint",
    "CostFidelitySweep",
    "build_cost_fidelity_report",
    "DriftPoint",
    "DriftTolerances",
    "DriftStudy",
    "build_drift_report",
]
