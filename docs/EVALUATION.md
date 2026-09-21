# Evaluation evidence

Kosmos records model outputs and evaluation evidence separately. The distinction
matters: a model can become easier to serve without becoming a better evaluator.

## GPU-backed inference profile

Slurm accounting from Sep. 18–21 recorded 46.17 allocated GPU-hours: 42.17 on
H100s and 4.01 on H200s. This spans 44 primary allocations across diagnostics,
conversion, replay, live workers, and pilot work. Twenty diagnostic jobs
completed; the remainder includes deliberate failed probes, cancelled
replacements, and time-bounded presentation workers. Allocation-hours measure
engineering work, not policy-quality or evaluation throughput.

| Path | Measured unit | Inference time | Peak GPU memory |
| --- | --- | ---: | ---: |
| OpenVLA | One native 7-D action | 0.836 s | 15.5 GiB |
| IRASim | One image-conditioned action → two frames | 2.55 s | 3.52 GiB |
| Cosmos | 16 actions → 17 generated frames | 4.52 s | 36.5 GiB |
| Qwen VLM judge | Five-sample structured judgment | 29.3 s | 16.9 GiB |

The judge data-collection pass covered 16 real Bridge clips, requested 80 judge
sample slots, and retained 141 raw attempts. The LoRA pilot used rank-64 Q/V
adapters for two epochs and eight optimizer steps, with 20.6 GiB peak CUDA
allocation. These are inference/training engineering measurements, not RL
results or evidence of policy improvement.

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
