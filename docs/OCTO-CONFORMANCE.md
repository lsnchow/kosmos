# Octo 241fb / AutoEval source conformance

This note distinguishes source conformance from policy qualification. The
production Octo-Small adapter is pinned to Octo `241fb3514b7c40957a86d869fecb7c7fc353f540`; the comparison wrapper is pinned to AutoEval `3ea3ff44c6950433cfbcb4294a3deaa616533745`. Neither the successful 938946 production smoke nor a source-conformance report certifies an executed prefix, a physical rollout, generated-image feedback, or a benchmark outcome.

## Source-backed behavior

The pinned AutoEval rollout wraps Octo with `HistoryWrapper(env, horizon=2)` followed by `TemporalEnsembleWrapper(env, 4)`. On reset, the source history wrapper repeats the first observed state and marks the older slot as padding. Its temporal wrapper has `exp_weight=0` by default; it selects the same overlapping chunk rows and uniform weights that PLUMB's `OctoTemporalEnsembler` implements.

The source `OctoPolicy` caches the first text task until its policy is reset, calls `jax.random.PRNGKey(0)` on every sampler call, and requests native normal unnormalization from `model.dataset_statistics["bridge_dataset"]["action"]`. Production PLUMB now mirrors the source task cache and static sampler key. It instead asks Octo for normalized rows using `unnormalization_statistics=None`, then applies the same source statistics once in its revisioned normalizer. This is algebraically comparable only when the source mask/statistics are bound to the same checkpoint; it is not an execution-prefix certification.

The two profiles remain separate:

| Profile | Source API | RNG | Normalization |
| --- | --- | --- | --- |
| Production 241fb | `timestep_pad_mask`, accepts `unnormalization_statistics` | AutoEval `PRNGKey(0)` per call | PLUMB applies source normal stats once |
| Diagnostic 37951 | `pad_mask`, rejects the newer keyword | native diagnostic profile | native JAX all-seven `mean + std * action` |

## What blocks an action comparison

The canonical 938946 PNG manifest binds RGB inputs and transform provenance only. AutoEval's `OctoPolicy` requires a `proprio` key and forwards it with the full two-step observation history. The source rollout configures `ManipulatorEnv` as `StateEncoding.POS_EULER`, then applies `ConvertState2Proprio` before history wrapping. No manifest-bound output of that exact conversion, or post-action third observation, is currently staged.

Do not replace those missing values with zero vectors, a canonical Euler state, repeated images, or inferred timestamps. A native LeRobot raw state cannot be handed to Octo as an AutoEval `proprio` value until the exact pinned `ConvertState2Proprio` implementation and input field mapping have been inspected, revision-bound, and replayed. In particular, the observed source gripper range `0.046..1.112` is provenance about source data, not authority to clamp/renormalize it to the old provisional `0..0.39` range. Doing either would create an unreviewable synthetic conformance fixture.

## Local-only source harness

`python -m plumb.policies.octo_autoeval_conformance` reads two pre-existing
clean local checkouts, hashes the exact policy/rollout/wrapper/model files, and
checks the source-level behavior above. It imports no model framework, loads no
weights, contacts no service, and writes no evidence.

```bash
.venv/bin/python -m plumb.policies.octo_autoeval_conformance \
  --autoeval-source /path/to/auto_eval \
  --octo-source /path/to/source-octo-autoeval241fb
```

Its `action_comparison` result is deliberately `blocked_missing_manifest_bound_real_proprio_history` until a separately reviewed fixture can bind, at minimum:

- the exact ordered raw observations needed for reset plus two real steps;
- RGB file/pixel hashes and source transform provenance for each observation;
- finite, source-convention proprio histories with their own hash/provenance;
- the exact Octo checkpoint statistics/mask and runtime/source locks; and
- a predeclared tolerance and no-clobber raw output path.

The bounded real-action comparison path is therefore: hash-bind one real episode parquet/video plus the exact raw state field; hash-bind the external `manipulator_gym` source revision that implements `ConvertState2Proprio`; record and validate its finite conversion output for reset plus two history ticks; and only then replay the actual pinned AutoEval policy call and PLUMB production adapter on those same converted observations, checkpoint statistics, source/runtime locks, and `PRNGKey(0)`. The comparison report must retain raw proposals and selected temporal-ensemble rows, remain `qualified=false`, and never mint a certification fixture.

Even then, a match would be a source-conformance diagnostic. A golden action
fixture, source-wrapper execution-prefix evidence, and Gate-A/B evidence would
still be required before PLUMB can expose a certified prefix.
