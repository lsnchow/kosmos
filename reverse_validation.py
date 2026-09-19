#!/usr/bin/env python3
"""Reverse validation: can training loss validate an LLM judge?

The question comes from ``SCRIPT.md``:

    "We also ran the reverse-validation experiment -- whether training loss can
    validate an LLM judge. The answer is no, and we can show why: scramble a
    judge's per-item scores until it's barely better than a coin flip and its
    loss correlation doesn't move at all. It stays at minus nought point nine
    two."

The answer is no, and the mechanism is not subtle once stated.

A "loss correlation" is computed between a training loss and a judge's
**aggregate** score, across checkpoints.  The usual aggregate is a mean.  A mean
is invariant under permutation of its inputs.  So scrambling which item received
which score leaves the aggregate bit-identical, and therefore leaves the loss
correlation bit-identical -- while item-level agreement with ground truth decays
to chance.

That is the whole finding: **a permutation-invariant aggregate cannot see
per-item validity at all.**  A loss correlation is not weak evidence about judge
quality; it is exactly zero evidence.  A judge that has been reduced to a coin
flip keeps the same loss correlation as a perfect one.

Two modes.

``demonstrate``
    An analytic demonstration on constructed data.  It proves the mechanism and
    is labelled ``analytic_demonstration`` -- it is not an empirical claim about
    any particular judge.

``analyse``
    The same sweep over real judge reports and real checkpoint losses.  Results
    are labelled ``empirical`` and carry their cohort and n.

Usage:

    python reverse_validation.py demonstrate --output results/reverse_validation.json
    python reverse_validation.py analyse --judge-scores scores.jsonl \\
        --checkpoint-losses losses.json --output results/reverse_validation.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

#: Scramble fractions swept.  0.0 is the unscrambled judge; 1.0 is a full
#: within-checkpoint permutation.
DEFAULT_SCRAMBLE_GRID: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

#: Aggregates to test.  ``mean`` is the one practitioners use and the one that is
#: exactly permutation-invariant.  The paired aggregate is included to show what
#: a *non*-invariant statistic looks like, so the finding is not mistaken for a
#: claim that all aggregates are blind.
AGGREGATES: Tuple[str, ...] = ("mean", "item_paired_agreement")


class ReverseValidationError(ValueError):
    """Inputs cannot support the experiment."""


@dataclass(frozen=True)
class Checkpoint:
    """One training checkpoint: its loss, its per-item judge scores, its labels."""

    checkpoint_id: str
    training_loss: float
    judge_scores: Tuple[float, ...]
    labels: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.training_loss):
            raise ReverseValidationError("training_loss must be finite for %s" % self.checkpoint_id)
        if len(self.judge_scores) != len(self.labels):
            raise ReverseValidationError(
                "checkpoint %s has %d scores for %d labels"
                % (self.checkpoint_id, len(self.judge_scores), len(self.labels))
            )
        if not self.judge_scores:
            raise ReverseValidationError("checkpoint %s has no items" % self.checkpoint_id)
        if any(not math.isfinite(value) for value in self.judge_scores):
            raise ReverseValidationError("checkpoint %s has a non-finite judge score" % self.checkpoint_id)
        if any(label not in (0, 1) for label in self.labels):
            raise ReverseValidationError("checkpoint %s labels must be 0/1" % self.checkpoint_id)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson correlation, or ``None`` when it is undefined.

    A constant vector has no correlation.  Returning ``None`` rather than 0.0 or
    1.0 keeps an undefined statistic from being read as a measured one.
    """

    if len(xs) != len(ys):
        raise ReverseValidationError("correlation needs equal-length vectors")
    if len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(value * value for value in dx)) * math.sqrt(sum(value * value for value in dy))
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def scramble(values: Sequence[float], fraction: float, rng: random.Random) -> List[float]:
    """Permute a *fraction* of ``values`` among themselves.

    The multiset is preserved exactly, which is the point: the aggregate cannot
    change, only the item-to-score assignment can.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ReverseValidationError("scramble fraction must be in [0, 1]")
    result = list(values)
    count = int(round(fraction * len(result)))
    if count < 2:
        return result
    indices = rng.sample(range(len(result)), count)
    shuffled = [result[index] for index in indices]
    rng.shuffle(shuffled)
    for index, value in zip(indices, shuffled):
        result[index] = value
    return result


def item_agreement(scores: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> float:
    """Fraction of items where the thresholded judge score matches the label."""

    matches = sum(1 for score, label in zip(scores, labels) if int(score >= threshold) == label)
    return matches / float(len(labels))


def chance_agreement(labels: Sequence[int]) -> float:
    """Agreement a coin flip would reach on this label distribution.

    For a balanced set this is 0.5; for an unbalanced one, majority-class
    prediction does better, so the honest floor is the majority rate.
    """

    positives = sum(labels)
    return max(positives, len(labels) - positives) / float(len(labels))


def aggregate_value(name: str, scores: Sequence[float], labels: Sequence[int]) -> float:
    if name == "mean":
        return statistics.fmean(scores)
    if name == "item_paired_agreement":
        return item_agreement(scores, labels)
    raise ReverseValidationError("unknown aggregate %r" % name)


@dataclass
class SweepResult:
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, row: Mapping[str, Any]) -> None:
        self.rows.append(dict(row))


def run_sweep(
    checkpoints: Sequence[Checkpoint],
    scramble_grid: Sequence[float] = DEFAULT_SCRAMBLE_GRID,
    repetitions: int = 200,
    seed: int = 20260919,
) -> Dict[str, Any]:
    """Sweep the scramble fraction and report what moves and what does not.

    At each fraction the judge's per-item scores are permuted *within each
    checkpoint*, so every checkpoint keeps its own score multiset.  Then:

    * item-level agreement with the labels is recomputed, and
    * the loss/aggregate correlation is recomputed for each aggregate.

    ``repetitions`` independent permutations give a distribution rather than one
    lucky draw.
    """

    if len(checkpoints) < 2:
        raise ReverseValidationError("a loss correlation needs at least two checkpoints")
    losses = [checkpoint.training_loss for checkpoint in checkpoints]
    all_labels = [label for checkpoint in checkpoints for label in checkpoint.labels]

    baseline_correlations: Dict[str, Optional[float]] = {}
    for name in AGGREGATES:
        baseline_correlations[name] = pearson(
            losses, [aggregate_value(name, c.judge_scores, c.labels) for c in checkpoints]
        )

    sweep = SweepResult()
    for fraction in scramble_grid:
        agreements: List[float] = []
        correlations: Dict[str, List[float]] = {name: [] for name in AGGREGATES}
        for repetition in range(repetitions):
            rng = random.Random((seed, fraction, repetition).__hash__())
            scrambled = [
                Checkpoint(
                    checkpoint_id=checkpoint.checkpoint_id,
                    training_loss=checkpoint.training_loss,
                    judge_scores=tuple(scramble(checkpoint.judge_scores, fraction, rng)),
                    labels=checkpoint.labels,
                )
                for checkpoint in checkpoints
            ]
            pooled_scores = [score for c in scrambled for score in c.judge_scores]
            agreements.append(item_agreement(pooled_scores, all_labels))
            for name in AGGREGATES:
                value = pearson(losses, [aggregate_value(name, c.judge_scores, c.labels) for c in scrambled])
                if value is not None:
                    correlations[name].append(value)
        row: Dict[str, Any] = {
            "scramble_fraction": fraction,
            "repetitions": repetitions,
            "item_agreement_mean": statistics.fmean(agreements),
            "item_agreement_min": min(agreements),
            "item_agreement_max": max(agreements),
        }
        for name in AGGREGATES:
            values = correlations[name]
            row["loss_correlation_%s_mean" % name] = statistics.fmean(values) if values else None
            row["loss_correlation_%s_min" % name] = min(values) if values else None
            row["loss_correlation_%s_max" % name] = max(values) if values else None
            baseline = baseline_correlations[name]
            row["loss_correlation_%s_shift" % name] = (
                None if baseline is None or not values else statistics.fmean(values) - baseline
            )
        sweep.add(row)

    chance = chance_agreement(all_labels)
    final = sweep.rows[-1]
    mean_shift = final.get("loss_correlation_mean_shift")
    invariant = mean_shift is not None and abs(mean_shift) < 1e-12

    return {
        "schema_version": SCHEMA_VERSION,
        "question": (
            "Can the correlation between training loss and an LLM judge's aggregate score "
            "validate that judge?"
        ),
        "answer": "No.",
        "mechanism": (
            "A loss correlation is computed against a permutation-invariant aggregate. Scrambling "
            "which item received which score preserves the aggregate exactly, so the correlation "
            "does not move, while item-level agreement decays toward chance. The statistic cannot "
            "see per-item validity at all."
        ),
        "checkpoints": len(checkpoints),
        "items_per_checkpoint": len(checkpoints[0].judge_scores),
        "total_items": len(all_labels),
        "chance_agreement": chance,
        "baseline_loss_correlation": baseline_correlations,
        "sweep": sweep.rows,
        "conclusion": {
            "mean_aggregate_loss_correlation_invariant": invariant,
            "mean_aggregate_correlation_shift_at_full_scramble": mean_shift,
            "item_agreement_at_full_scramble": final["item_agreement_mean"],
            "item_agreement_unscrambled": sweep.rows[0]["item_agreement_mean"],
            "collapsed_to_chance": abs(final["item_agreement_mean"] - chance) < 0.1,
            "reading": (
                "A judge scrambled to chance keeps the same loss correlation as the unscrambled one. "
                "Loss correlation is therefore not weak evidence of judge quality; it is no evidence. "
                "Judge validity has to be established against labels, which is what Gate D does."
            ),
        },
        "limitations": [
            "This isolates one specific failure of one specific validation shortcut.",
            "It does not show that training loss is uninformative about the trained model.",
            "A non-permutation-invariant aggregate can move; the item_paired_agreement column is "
            "included precisely to show that contrast.",
        ],
    }


def build_demonstration(
    checkpoints: int = 12,
    items: int = 200,
    seed: int = 20260919,
    loss_noise: float = 0.16,
) -> Tuple[List[Checkpoint], Dict[str, Any]]:
    """Construct checkpoints with a strong negative loss/mean-score correlation.

    This is a *constructed* demonstration of a mathematical fact, not a
    measurement of any real judge, and the output is labelled
    ``analytic_demonstration``.  The construction is transparent: training loss
    falls over checkpoints, the true success rate rises with it, and the judge's
    mean score tracks the true rate with noise -- exactly the situation in which
    a practitioner is tempted to read the loss correlation as validation.

    A note on the magnitude.  ``SCRIPT.md`` quotes a loss correlation of about
    -0.92 from the original experiment, whose data is not in this workspace.  The
    correlation this construction produces depends on ``loss_noise`` and is
    reported as observed; it is deliberately not tuned to reproduce a specific
    figure, because the *magnitude* is a property of the construction while the
    finding is the **invariance** -- the correlation does not move at all when the
    judge is scrambled to chance.  Point ``loss_noise`` at a real training curve's
    noise level if you want the demonstration to resemble one.
    """

    rng = random.Random(seed)
    built: List[Checkpoint] = []
    for index in range(checkpoints):
        progress = index / float(max(1, checkpoints - 1))
        # Loss decays; the true success rate rises. This is the correlation a
        # practitioner sees and misreads as evidence about the judge.
        loss = 2.0 * math.exp(-1.8 * progress) + rng.gauss(0.0, loss_noise)
        true_rate = 0.1 + 0.75 * progress
        labels = tuple(1 if rng.random() < true_rate else 0 for _ in range(items))
        # A judge that is genuinely informative: it mostly agrees with the label.
        scores = tuple(
            min(1.0, max(0.0, (0.8 if label else 0.2) + rng.gauss(0.0, 0.12))) for label in labels
        )
        built.append(
            Checkpoint(checkpoint_id="ckpt-%02d" % index, training_loss=loss, judge_scores=scores, labels=labels)
        )
    observed = pearson(
        [c.training_loss for c in built], [statistics.fmean(c.judge_scores) for c in built]
    )
    provenance = {
        "kind": "analytic_demonstration",
        "not_an_empirical_claim": True,
        "construction": (
            "Training loss decays over checkpoints; the true success rate rises; the judge's mean "
            "score tracks the true rate with noise. No real judge or real training run is involved."
        ),
        "observed_loss_correlation": observed,
        "loss_noise": loss_noise,
        "magnitude_note": (
            "SCRIPT.md quotes about -0.92 from the original experiment, whose data is not in this "
            "workspace. This magnitude is a property of the construction and is reported as observed; "
            "the finding is the invariance, not the number."
        ),
        "seed": seed,
    }
    return built, provenance


def load_checkpoints(scores_path: Path, losses_path: Path) -> Tuple[List[Checkpoint], Dict[str, Any]]:
    """Load real per-item judge scores and real checkpoint losses.

    ``scores_path`` is JSONL with ``{checkpoint_id, item_id, judge_score, label}``.
    ``losses_path`` is JSON mapping ``checkpoint_id -> training_loss``.
    Item ordering is made deterministic by sorting on ``item_id`` so a rerun
    reproduces the same pairing.
    """

    losses = json.loads(losses_path.read_text())
    if not isinstance(losses, Mapping):
        raise ReverseValidationError("checkpoint losses must be a JSON object")
    grouped: Dict[str, List[Tuple[str, float, int]]] = {}
    with scores_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ReverseValidationError("line %d of %s is not JSON" % (number, scores_path)) from exc
            for required in ("checkpoint_id", "item_id", "judge_score", "label"):
                if required not in row:
                    raise ReverseValidationError("line %d is missing %r" % (number, required))
            grouped.setdefault(str(row["checkpoint_id"]), []).append(
                (str(row["item_id"]), float(row["judge_score"]), int(row["label"]))
            )
    checkpoints: List[Checkpoint] = []
    for checkpoint_id in sorted(grouped):
        if checkpoint_id not in losses:
            raise ReverseValidationError("no training loss recorded for checkpoint %s" % checkpoint_id)
        rows = sorted(grouped[checkpoint_id], key=lambda item: item[0])
        checkpoints.append(
            Checkpoint(
                checkpoint_id=checkpoint_id,
                training_loss=float(losses[checkpoint_id]),
                judge_scores=tuple(row[1] for row in rows),
                labels=tuple(row[2] for row in rows),
            )
        )
    sizes = {len(c.judge_scores) for c in checkpoints}
    if len(sizes) != 1:
        raise ReverseValidationError(
            "every checkpoint must score the same item set; saw sizes %s" % sorted(sizes)
        )
    provenance = {
        "kind": "empirical",
        "not_an_empirical_claim": False,
        "judge_scores_source": str(scores_path),
        "checkpoint_losses_source": str(losses_path),
        "cohort": "reverse_validation",
    }
    return checkpoints, provenance


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demonstrate", help="Run the analytic demonstration on constructed data")
    demo.add_argument("--checkpoints", type=int, default=12)
    demo.add_argument("--items", type=int, default=200)
    demo.add_argument("--repetitions", type=int, default=200)
    demo.add_argument("--seed", type=int, default=20260919)
    demo.add_argument(
        "--loss-noise",
        type=float,
        default=0.16,
        help="Training-loss noise in the construction. Sets the correlation magnitude, not the finding.",
    )
    demo.add_argument("--output", type=Path, default=None)

    real = commands.add_parser("analyse", help="Run the sweep on real judge scores and checkpoint losses")
    real.add_argument("--judge-scores", type=Path, required=True)
    real.add_argument("--checkpoint-losses", type=Path, required=True)
    real.add_argument("--repetitions", type=int, default=200)
    real.add_argument("--seed", type=int, default=20260919)
    real.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.command == "demonstrate":
        checkpoints, provenance = build_demonstration(
            checkpoints=args.checkpoints, items=args.items, seed=args.seed, loss_noise=args.loss_noise
        )
    else:
        checkpoints, provenance = load_checkpoints(args.judge_scores, args.checkpoint_losses)

    report = run_sweep(checkpoints, repetitions=args.repetitions, seed=args.seed)
    report["provenance"] = provenance

    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
        print(str(args.output))
    else:
        print(encoded)

    conclusion = report["conclusion"]
    print(
        "\nitem agreement: %.3f unscrambled -> %.3f fully scrambled (chance %.3f)"
        % (
            conclusion["item_agreement_unscrambled"],
            conclusion["item_agreement_at_full_scramble"],
            report["chance_agreement"],
        )
    )
    print(
        "loss correlation (mean aggregate): %.4f, shift under full scramble: %s"
        % (report["baseline_loss_correlation"]["mean"], conclusion["mean_aggregate_correlation_shift_at_full_scramble"])
    )
    print("invariant: %s" % conclusion["mean_aggregate_loss_correlation_invariant"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
