# Evaluation evidence

Kosmos records model outputs and evaluation evidence separately. The distinction
matters: a model can become easier to serve without becoming a better evaluator.

## Completed paired audit

The base Qwen2.5-VL-7B judge and its saved LoRA adapter were run with identical
frames, scene reference, rubric, deterministic decoding, and model revision on
nine clips excluded before the pilot training set was constructed. Those clips
were never LoRA training or validation targets, but they were part of the
original pilot collection; this is an untrained-pool engineering audit, not an
independent held-out benchmark.

| Metric | Base | LoRA |
| --- | ---: | ---: |
| Schema-valid output | 8 / 9 | 8 / 9 |
| Bare JSON output | 0 / 9 | 9 / 9 |
| Mean generated tokens | 86.8 | 64.2 |
| Mean greedy decode time | 1.851 s | 1.589 s |
| Full semantic agreement between paired outputs | — | 0 / 9 |

The adapter made direct JSON reliable and reduced output length and greedy decode
time. It also shifted integrity from `intact` to `artifact` on every parseable
paired clip. It therefore demonstrates an output-contract/latency effect, not a
semantic-quality gain.

## What a quality claim requires

The next post-training study should use a fresh, stratified evaluation panel that
is not used for training, with independently produced human labels. Evaluate the
base and candidate adapter under the same frame-selection, prompt, seed,
temperature, and quorum protocol. Report coverage, abstention, schema-valid
rate, completion agreement, integrity agreement, latency, and paired confidence
intervals.

Kosmos does not currently include a verified RL training or reward-based policy
improvement result. The repository intentionally does not present VLM output
formatting gains as RL or policy-quality gains.
