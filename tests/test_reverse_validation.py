"""Reverse validation: the loss correlation is exactly blind to judge validity."""
from __future__ import annotations

import json
import random

import pytest

from reverse_validation import (
    Checkpoint,
    ReverseValidationError,
    build_demonstration,
    chance_agreement,
    item_agreement,
    load_checkpoints,
    main,
    pearson,
    run_sweep,
    scramble,
)


def test_pearson_is_undefined_not_perfect_for_a_constant_vector():
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert pearson([1.0], [2.0]) is None
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    with pytest.raises(ReverseValidationError, match="equal-length"):
        pearson([1.0, 2.0], [1.0])


def test_scramble_preserves_the_multiset_exactly():
    rng = random.Random(0)
    values = [0.1, 0.4, 0.4, 0.9, 0.2]
    for fraction in (0.0, 0.25, 0.5, 1.0):
        result = scramble(values, fraction, rng)
        assert sorted(result) == sorted(values), "the aggregate cannot change if the multiset does not"
    with pytest.raises(ReverseValidationError, match=r"\[0, 1\]"):
        scramble(values, 1.5, rng)


def test_chance_agreement_is_the_majority_rate_not_always_one_half():
    assert chance_agreement([0, 1, 0, 1]) == pytest.approx(0.5)
    # With an unbalanced set, guessing the majority beats a coin flip, so the
    # honest floor is higher than 0.5.
    assert chance_agreement([1, 1, 1, 0]) == pytest.approx(0.75)


def test_a_checkpoint_rejects_mismatched_or_nonfinite_inputs():
    with pytest.raises(ReverseValidationError, match="scores for"):
        Checkpoint("c", 1.0, (0.5, 0.5), (1,))
    with pytest.raises(ReverseValidationError, match="non-finite judge score"):
        Checkpoint("c", 1.0, (float("nan"),), (1,))
    with pytest.raises(ReverseValidationError, match="must be finite"):
        Checkpoint("c", float("inf"), (0.5,), (1,))
    with pytest.raises(ReverseValidationError, match="labels must be 0/1"):
        Checkpoint("c", 1.0, (0.5,), (2,))
    with pytest.raises(ReverseValidationError, match="no items"):
        Checkpoint("c", 1.0, (), ())


def test_a_loss_correlation_needs_at_least_two_checkpoints():
    with pytest.raises(ReverseValidationError, match="at least two checkpoints"):
        run_sweep([Checkpoint("c", 1.0, (0.5,), (1,))])


def test_the_mean_aggregate_loss_correlation_is_exactly_invariant():
    """The headline finding, asserted as an exact identity rather than a trend."""

    checkpoints, _ = build_demonstration(checkpoints=8, items=60, seed=7)
    report = run_sweep(checkpoints, repetitions=25, seed=7)
    baseline = report["baseline_loss_correlation"]["mean"]
    for row in report["sweep"]:
        # Every repetition at every scramble fraction must give the identical
        # correlation: min == max == mean == baseline.
        assert row["loss_correlation_mean_min"] == pytest.approx(baseline, abs=1e-12)
        assert row["loss_correlation_mean_max"] == pytest.approx(baseline, abs=1e-12)
        assert row["loss_correlation_mean_shift"] == pytest.approx(0.0, abs=1e-12)
    assert report["conclusion"]["mean_aggregate_loss_correlation_invariant"] is True


def test_item_agreement_collapses_toward_chance_while_the_correlation_does_not():
    checkpoints, _ = build_demonstration(checkpoints=8, items=120, seed=11)
    report = run_sweep(checkpoints, repetitions=25, seed=11)
    unscrambled = report["sweep"][0]["item_agreement_mean"]
    scrambled = report["sweep"][-1]["item_agreement_mean"]
    assert unscrambled > 0.9, "the constructed judge starts out informative"
    assert scrambled < unscrambled - 0.2, "scrambling must actually destroy item-level validity"
    assert scrambled == pytest.approx(report["chance_agreement"], abs=0.12)
    assert report["conclusion"]["collapsed_to_chance"] is True


def test_a_non_permutation_invariant_aggregate_does_move():
    """The contrast that keeps the finding from being overstated."""

    checkpoints, _ = build_demonstration(checkpoints=8, items=120, seed=13)
    report = run_sweep(checkpoints, repetitions=25, seed=13)
    shift = report["sweep"][-1]["loss_correlation_item_paired_agreement_shift"]
    assert shift is not None and abs(shift) > 0.05


def test_the_report_states_the_question_the_answer_and_the_limits():
    checkpoints, provenance = build_demonstration(checkpoints=4, items=40, seed=3)
    report = run_sweep(checkpoints, repetitions=10, seed=3)
    assert report["answer"] == "No."
    assert "permutation-invariant" in report["mechanism"]
    assert "no evidence" in report["conclusion"]["reading"]
    assert report["limitations"], "a negative result still states its scope"
    assert provenance["kind"] == "analytic_demonstration"
    assert provenance["not_an_empirical_claim"] is True
    # The magnitude is disclosed as construction-dependent.
    assert "property of the construction" in provenance["magnitude_note"]
    # The whole report must be JSON-portable.
    json.dumps(report, allow_nan=False)


def test_real_data_is_labelled_empirical_and_requires_aligned_items(tmp_path):
    scores = tmp_path / "scores.jsonl"
    losses = tmp_path / "losses.json"
    rows = []
    for checkpoint in ("ckpt-0", "ckpt-1"):
        for index in range(4):
            rows.append(
                {
                    "checkpoint_id": checkpoint,
                    "item_id": "item-%d" % index,
                    "judge_score": 0.9 if index % 2 else 0.1,
                    "label": index % 2,
                }
            )
    scores.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    losses.write_text(json.dumps({"ckpt-0": 2.0, "ckpt-1": 1.0}))
    checkpoints, provenance = load_checkpoints(scores, losses)
    assert provenance["kind"] == "empirical"
    assert provenance["cohort"] == "reverse_validation"
    assert [c.checkpoint_id for c in checkpoints] == ["ckpt-0", "ckpt-1"]
    assert item_agreement(checkpoints[0].judge_scores, checkpoints[0].labels) == pytest.approx(1.0)


def test_real_data_rejects_a_missing_loss_and_a_ragged_item_set(tmp_path):
    scores = tmp_path / "scores.jsonl"
    losses = tmp_path / "losses.json"
    scores.write_text(json.dumps({"checkpoint_id": "a", "item_id": "i", "judge_score": 0.5, "label": 1}) + "\n")
    losses.write_text(json.dumps({}))
    with pytest.raises(ReverseValidationError, match="no training loss"):
        load_checkpoints(scores, losses)

    scores.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"checkpoint_id": "a", "item_id": "i1", "judge_score": 0.5, "label": 1},
                {"checkpoint_id": "b", "item_id": "i1", "judge_score": 0.5, "label": 1},
                {"checkpoint_id": "b", "item_id": "i2", "judge_score": 0.5, "label": 0},
            )
        )
        + "\n"
    )
    losses.write_text(json.dumps({"a": 1.0, "b": 0.5}))
    with pytest.raises(ReverseValidationError, match="same item set"):
        load_checkpoints(scores, losses)

    scores.write_text(json.dumps({"checkpoint_id": "a", "judge_score": 0.5, "label": 1}) + "\n")
    losses.write_text(json.dumps({"a": 1.0}))
    with pytest.raises(ReverseValidationError, match="item_id"):
        load_checkpoints(scores, losses)


def test_cli_writes_a_portable_report(tmp_path):
    output = tmp_path / "reverse_validation.json"
    assert main(["demonstrate", "--checkpoints", "5", "--items", "40", "--repetitions", "10", "--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["conclusion"]["mean_aggregate_loss_correlation_invariant"] is True
    assert payload["provenance"]["kind"] == "analytic_demonstration"
