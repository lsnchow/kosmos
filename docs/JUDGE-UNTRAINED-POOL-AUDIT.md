# Base vs. post-trained judge: untrained-pool output audit

This post-hoc engineering audit compared the frozen Qwen2.5-VL-7B base model
with the saved pilot LoRA (`epoch-02`) on nine clips that were explicitly
excluded before pilot training because the v1 teacher had tied semantic
outputs. They were never training or validation targets. They were still part
of the original pilot collection, so this is not a fresh held-out benchmark,
human-label evaluation, calibration result, or quality claim.

## Result

| Metric | Base | Pilot LoRA |
| --- | ---: | ---: |
| Clips | 9 | 9 |
| Schema-valid outputs | 8/9 | 8/9 |
| Bare JSON outputs | 0/9 | 9/9 |
| Mean generated tokens | 86.8 | 64.2 |
| Mean greedy-generation time | 1.851 s | 1.589 s |
| Full semantic agreement with the paired base output | — | 0/9 |

The adapter reduced generated tokens by 26% and mean generation time by 14%.
It did not increase the schema-valid rate. Across the eight clips where both
outputs parsed, it changed integrity from `intact` to `artifact` in every
case; it changed collision on two clips; progress and completion stayed at
`5` / `met`. That is a strong pattern of semantic drift toward the pilot target
tuple, not evidence of improved visual judgment.

## Interpretation

The pilot produced a real, measurable formatting/latency effect: direct JSON,
shorter answers, and lower greedy decode time. It did not produce evidence of
semantic improvement. This is consistent with the training-set collapse:
every accepted target was `[artifact, visible, 5, met]`.

## Reproducibility

- Slurm job: `974866`, one H100, 1 minute 28 seconds, exit `0`.
- Source release: `1f6b969e32a947d5214c76c997e440527561754e138263c77da402b1e6760bd2`.
- Immutable remote result:
  `/scratch/lchow432/plumb/evidence/judge-untrained-pool-audit-1f6b969e32a947d5214c76c997e440527561754e138263c77da402b1e6760bd2.json`.
- Script: `cluster/judge_lora_untrained_pool_audit.py`.

The next quality evaluation needs fresh, independently human-labelled clips.
