# SuSIE_LL `gc_bc` static-goal diagnostic

This is a one-action engineering diagnostic for the released AutoEval
replication component. It is not a PLUMB policy evaluation, success result,
benchmark cell, Gate A/B result, or primary-study record.

## Immutable inputs

- Low-level weights: [`patreya/gcbc-bridge` revision
  `1a4c15dd9ad780a257e9494f0fac79cbe8e64793`](https://huggingface.co/patreya/gcbc-bridge/tree/1a4c15dd9ad780a257e9494f0fac79cbe8e64793).
  The direct Flax restore file is `checkpoint/checkpoint`, size 258,718,956 B,
  SHA-256 `80b354db7a05d514d6df5b5a4395469902b0ff362d383edeeb4abb8c1c9d9e33`.
  The historical source run name `checkpoint_75000` appears only in
  `checkpoint/commit_success.txt`; it is not the published file path.
- License disposition: immutable `README.md` has SHA-256
  `d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17`
  and declares `license: mit`. This is a publisher-declared card/README
  notice; preserve it with the artifact. The snapshot has no separate license
  file, so do not inflate this into an independent redistribution audit.
- Code: [AutoEval `GCPolicy` and normalizer
  `3ea3ff44c6950433cfbcb4294a3deaa616533745`](https://github.com/zhouzypaul/auto_eval/blob/3ea3ff44c6950433cfbcb4294a3deaa616533745/auto_eval/robot/policy.py#L454-L550),
  [AutoEval `gc_bc` configuration](https://github.com/zhouzypaul/auto_eval/blob/3ea3ff44c6950433cfbcb4294a3deaa616533745/scripts/configs/eval_config.py#L5-L111),
  and [SOAR `model_training/jaxrl_m` `gc_bc` agent
  `eabd5f16a856e484884a22e257a941bb358cea08`](https://github.com/rail-berkeley/soar/blob/eabd5f16a856e484884a22e257a941bb358cea08/model_training/jaxrl_m/agents/continuous/gc_bc.py).
  AutoEval's pinned README routes optional `jaxrl_m` installation to
  `rail-berkeley/soar/tree/main/model_training`; BridgeData V2's similarly
  named implementation rejects AutoEval's `std_parameterization` field and is
  not substituted here.

## What the harness checks

`cluster/susie_low_level_smoke.py` requires a Slurm GPU allocation and an
immutable PLUMB source release. It verifies the release manifest byte hash and
every source-file hash before importing policy code. It then verifies:

1. a clean SOAR source checkout at `eabd5f16…`, importing only its
   `model_training/` subdirectory;
2. the direct checkpoint digest and publisher README digest/license declaration;
3. a source-compatible agent construction: `resnetv1-34`, 256px goal/current
   image batches, `gc_bc`, `early_goal_concat=True`, fixed gripper standard
   deviation `0.1`, and deterministic `argmax=True` sampling;
4. that restore changes the initialized tensor-tree digest—preventing Flax's
   no-checkpoint fallback from looking like inference;
5. a hash-bound, explicitly unqualified two-frame fixture. It contains source
   video frame 0 as current image and the source video's actual final decoded
   frame as a static goal image. They must be distinct RGB uint8 256×256 PNGs;
6. finite one-row normalized and transformed actions and an exact reset/repeat
   match.

The source wrapper affine-transforms its first six dimensions using AutoEval's
`ACT_MEAN`/`ACT_STD` and thresholds gripper at zero. PLUMB therefore receives
the first six normalized values plus the source-thresholded binary gripper, and
uses a normalizer mask of `(true, true, true, true, true, true, false)` so that
the affine transform happens exactly once.

## Runtime deviation

SOAR's model-training requirements use JAX 0.4.20, Flax 0.7.5, Distrax 0.1.5,
NumPy 1.24.3, TensorFlow 2.15.0, Orbax 0.3.5, and SciPy 1.12.0. The existing
Octo runtime matches the JAX/Flax/Distrax core but has NumPy 1.26.4,
TensorFlow 2.15.1, Orbax 0.4.3, SciPy 1.11.2, and site build suffixes. The
report records module versions separately from immutable runtime-lock package
metadata, enumerates these mismatches, and keeps
`full_runtime_tuple_verified=false`. Passing the static action/repeat check
does not validate the complete source runtime tuple, robot behavior, task
achievement, generated-image feedback, or the full SuSIE diffusion stack.

TensorFlow GPU visibility is disabled before JAX imports and XLA preallocation
is disabled. No SuSIE high-level diffusion, Stable Diffusion, model download,
world model, or judge is imported by this diagnostic.

## Measured static-fixture result — 2026-09-19

Trillium job `940190` completed with exit `0:0` on one NVIDIA H100 80GB GPU.
It used immutable PLUMB release
`fdc52dcb09cebae2bdcd3d213a4865aa28c43f87be425f483a456a01afca44ea`;
the report is preserved locally at
`data/live-integrated/cluster-evidence/susie-ll-gcbc-940190/report.json` and
cluster-side as `evidence/susie-ll-gcbc-static-goal-v5.json`.

- The source fixture was the immutable vendor video SHA-256
  `a86cfc81633b216891ca26dc58c72193a979c10ad72f123175fa8d61a67cdaec`,
  decoded with OpenCV 4.10.0. Current/goal were its distinct first and actual
  final decoded frames (indices 0 and 16 of 17), transformed into hash-bound
  256px RGB PNGs. The final frame remains demonstrated conditioning only—not a
  task goal, success label, or feedback observation.
- Strict inference-only restore accepted raw keys `{state}` and state keys
  `{opt_states, params, rng, step, target_params}`, required
  `target_params=null`, excluded optimizer state, and recorded unequal
  initialized/restored parameter digests. It is not a training checkpoint
  resume path.
- The one native output was finite `(1, 7)` and exactly repeated after reset:
  normalized action `[-0.136262, 0.020200, -0.467370, -0.122081,
  0.590357, -0.008607, 0.540946]`; transformed physical action
  `[-0.001051, 0.000394, -0.005892, -0.003366, 0.016666, -0.000398, 1.0]`.
  The first `8.5735383 s` includes source import, model construction, restore,
  and first invocation; the loaded repeat was `0.00886015 s`. Neither number
  is world-model, platform, or evaluation throughput.
- The runtime core JAX/Flax versions matched SOAR's requirements. Full runtime
  compatibility remains unverified: NumPy, TensorFlow, Orbax, and SciPy differ
  from the SOAR pins. The condensed runtime-lock package map omitted Distrax,
  while its full pip freeze records `distrax==0.1.5+computecanada`, matching the
  source requirement; that provenance discrepancy remains visible rather than
  being silently repaired.

Four preceding failed diagnostic reports were retained separately. No new GPU
run is implied by this documentation update. This successful static action does
not qualify SuSIE_LL, SuSIE, a five-task policy comparison, or any gate.
