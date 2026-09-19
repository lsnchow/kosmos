"""Published comparator data used by PLUMB measurement reports.

The values in this module are transcription data, not measurements made by
PLUMB.  Keeping them in a small dependency-free module makes it harder for a
live rollout result to be confused with the AutoEval paper's real-robot table.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional


# Keep this order frozen anywhere a deterministic policy tie break is needed.
POLICY_IDS = (
    "OpenVLA",
    "OpenPiZero",
    "Octo",
    "MiniVLA",
    "SuSIE",
    "SuSIE_LL",
)

TASK_IDS = (
    "open_drawer",
    "close_drawer",
    "to_basket",
    "to_sink",
    "fold_cloth",
)

REFERENCE_TRIALS_PER_CELL = 50


# AutoEval Table 2.  These are successes out of 50 real-robot trials.
HUMAN = {
    "OpenVLA":     {"open_drawer": 40, "close_drawer": 46, "to_basket":  1, "to_sink":  0, "fold_cloth": 12},
    "OpenPiZero":  {"open_drawer": 24, "close_drawer": 45, "to_basket":  7, "to_sink": 47, "fold_cloth":  3},
    "Octo":        {"open_drawer":  0, "close_drawer":  0, "to_basket":  0, "to_sink":  0, "fold_cloth":  2},
    "MiniVLA":     {"open_drawer": 32, "close_drawer": 49, "to_basket": 38, "to_sink":  0, "fold_cloth":  8},
    "SuSIE":       {"open_drawer":  2, "close_drawer": 13, "to_basket":  0, "to_sink":  0, "fold_cloth": 10},
    "SuSIE_LL":    {"open_drawer":  0, "close_drawer":  0, "to_basket":  0, "to_sink":  0, "fold_cloth":  0},
}


# AutoEval Table 3.  It intentionally has no fold_cloth column.
SIMPLER = {
    "OpenVLA":     {"open_drawer": 32, "close_drawer":  2, "to_basket":  1, "to_sink": 0},
    "OpenPiZero":  {"open_drawer": 34, "close_drawer": 24, "to_basket": 45, "to_sink": 6},
    "Octo":        {"open_drawer":  3, "close_drawer":  0, "to_basket":  6, "to_sink": 3},
    "MiniVLA":     {"open_drawer": 30, "close_drawer": 23, "to_basket": 10, "to_sink": 2},
    "SuSIE":       {"open_drawer":  0, "close_drawer": 41, "to_basket":  7, "to_sink": 0},
    "SuSIE_LL":    {"open_drawer":  1, "close_drawer":  0, "to_basket":  0, "to_sink": 0},
}


POLICY_MANIFEST = {
    "OpenVLA": {"display_name": "OpenVLA", "reference_key": "OpenVLA"},
    "OpenPiZero": {"display_name": "OpenPiZero", "reference_key": "OpenPiZero"},
    # The paper's table key is Octo.  The required benchmark asset is
    # Octo-Small v1.0, so its runtime-facing alias resolves to this key.
    "Octo": {"display_name": "Octo-Small v1.0", "reference_key": "Octo", "runtime_name": "OctoSmall"},
    "MiniVLA": {"display_name": "MiniVLA", "reference_key": "MiniVLA"},
    "SuSIE": {"display_name": "SuSIE", "reference_key": "SuSIE"},
    "SuSIE_LL": {"display_name": "SuSIE_LL", "reference_key": "SuSIE_LL"},
}


_POLICY_ALIASES = {
    "openvla": "OpenVLA",
    "openpizero": "OpenPiZero",
    "open pi zero": "OpenPiZero",
    "open-pi-zero": "OpenPiZero",
    "octo": "Octo",
    "octosmall": "Octo",
    "octo-small": "Octo",
    "octo-small v1.0": "Octo",
    "minivla": "MiniVLA",
    "susie": "SuSIE",
    "susie_ll": "SuSIE_LL",
    "susie-ll": "SuSIE_LL",
}

_TASK_ALIASES = {
    "open_drawer": "open_drawer",
    "close_drawer": "close_drawer",
    "to_basket": "to_basket",
    "to_sink": "to_sink",
    "fold_cloth": "fold_cloth",
}


def _name_from_value(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        value = value.get("name")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def canonical_policy(value: Any) -> Optional[str]:
    """Return the frozen reference policy ID when *value* is a known alias."""

    name = _name_from_value(value)
    if name is None:
        return None
    return _POLICY_ALIASES.get(name.lower(), name)


def canonical_task(value: Any) -> Optional[str]:
    """Return a canonical task ID without inventing a mapping for unknown work."""

    name = _name_from_value(value)
    if name is None:
        return None
    return _TASK_ALIASES.get(name.lower(), name)


def reference_rate(policy: Any, task: Any, table: str = "human") -> Optional[float]:
    """Return a published cell rate, or ``None`` when that cell is absent."""

    policy_id = canonical_policy(policy)
    task_id = canonical_task(task)
    source = HUMAN if table.lower() == "human" else SIMPLER if table.lower() == "simpler" else None
    if source is None or policy_id not in source or task_id not in source[policy_id]:
        return None
    return source[policy_id][task_id] / float(REFERENCE_TRIALS_PER_CELL)


def reference_successes(policy: Any, task: Any, table: str = "human") -> Optional[int]:
    """Return the published numerator, preserving SIMPLER's absent cloth cell."""

    policy_id = canonical_policy(policy)
    task_id = canonical_task(task)
    source = HUMAN if table.lower() == "human" else SIMPLER if table.lower() == "simpler" else None
    if source is None or policy_id not in source or task_id not in source[policy_id]:
        return None
    return source[policy_id][task_id]


def reference_arithmetic() -> Dict[str, Any]:
    """Return auditable arithmetic checks rather than silently relying on prose."""

    totals = {policy: sum(HUMAN[policy].values()) for policy in POLICY_IDS}
    leave_one_out = {}
    for task in TASK_IDS:
        leave_one_out[task] = {
            "MiniVLA": totals["MiniVLA"] - HUMAN["MiniVLA"][task],
            "OpenPiZero": totals["OpenPiZero"] - HUMAN["OpenPiZero"][task],
        }
    reversals = [
        task
        for task, values in leave_one_out.items()
        if values["OpenPiZero"] > values["MiniVLA"]
    ]
    return {
        "human_totals": totals,
        "primary_cells": len(POLICY_IDS) * len(TASK_IDS),
        "primary_trials": len(POLICY_IDS) * len(TASK_IDS) * REFERENCE_TRIALS_PER_CELL,
        "octo_total": totals["Octo"],
        "minivla_total": totals["MiniVLA"],
        "openpizero_total": totals["OpenPiZero"],
        "leave_one_task_out": leave_one_out,
        "minivla_openpizero_reversal_tasks": reversals,
        "minivla_openpizero_reversal_count": len(reversals),
        "headline_openvla_close_drawer": {
            "human_successes": HUMAN["OpenVLA"]["close_drawer"],
            "human_rate": reference_rate("OpenVLA", "close_drawer"),
            "simpler_successes": SIMPLER["OpenVLA"]["close_drawer"],
            "simpler_rate": reference_rate("OpenVLA", "close_drawer", "simpler"),
            "percentage_point_gap": 88.0,
        },
    }


def _validate_constants() -> None:
    assert tuple(HUMAN) == POLICY_IDS
    assert tuple(SIMPLER) == POLICY_IDS
    assert all(tuple(HUMAN[policy]) == TASK_IDS for policy in POLICY_IDS)
    assert all("fold_cloth" not in SIMPLER[policy] for policy in POLICY_IDS)
    arithmetic = reference_arithmetic()
    assert arithmetic["primary_trials"] == 1500
    assert arithmetic["octo_total"] == 2
    assert arithmetic["minivla_total"] == 127
    assert arithmetic["openpizero_total"] == 126
    assert arithmetic["minivla_openpizero_reversal_count"] == 4


_validate_constants()


def reference_payload() -> Dict[str, Any]:
    """Return JSON-serializable published-reference data for clients and reports."""

    return {
        "schema_version": 1,
        "source": {
            "name": "AutoEval",
            "human_table": "Table 2",
            "simpler_table": "Table 3",
            "note": "Published comparator data; not PLUMB outcomes.",
        },
        "trials_per_cell": REFERENCE_TRIALS_PER_CELL,
        "policy_ids": list(POLICY_IDS),
        "task_ids": list(TASK_IDS),
        "policy_manifest": deepcopy(POLICY_MANIFEST),
        "policy_aliases": {"OctoSmall": "Octo", "Octo-Small": "Octo", "Octo-Small v1.0": "Octo"},
        "human": deepcopy(HUMAN),
        "simpler": deepcopy(SIMPLER),
        "human_rates": {
            policy: {task: successes / float(REFERENCE_TRIALS_PER_CELL) for task, successes in row.items()}
            for policy, row in HUMAN.items()
        },
        "simpler_rates": {
            policy: {task: successes / float(REFERENCE_TRIALS_PER_CELL) for task, successes in row.items()}
            for policy, row in SIMPLER.items()
        },
        "arithmetic": reference_arithmetic(),
        "limitations": [
            "SIMPLER has no fold_cloth column; absence is not a zero.",
            "Human trial identities are unavailable, so generated episodes cannot be paired to them.",
            "Octo is the paper key and maps to the Octo-Small v1.0 benchmark policy.",
        ],
    }


__all__ = [
    "HUMAN",
    "SIMPLER",
    "POLICY_IDS",
    "TASK_IDS",
    "REFERENCE_TRIALS_PER_CELL",
    "POLICY_MANIFEST",
    "canonical_policy",
    "canonical_task",
    "reference_rate",
    "reference_successes",
    "reference_arithmetic",
    "reference_payload",
]
