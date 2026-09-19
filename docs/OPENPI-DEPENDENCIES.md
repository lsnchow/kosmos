# OpenPiZero dependency and provenance boundary

`plumb.policies.openpi.OpenPiZeroPolicyAdapter` is a lazy, local-only adapter
for the Bridge beta OpenPiZero path. It is source-backed by
[`allenzren/open-pi-zero` commit `c3df7fb062175c16f69d7ca4ce042958ea238fb7`](https://github.com/allenzren/open-pi-zero/tree/c3df7fb062175c16f69d7ca4ce042958ea238fb7)
and the AutoEval replication wrapper at
[`zhouzypaul/auto_eval` commit `3ea3ff44c6950433cfbcb4294a3deaa616533745`](https://github.com/zhouzypaul/auto_eval/blob/3ea3ff44c6950433cfbcb4294a3deaa616533745/auto_eval/robot/policy.py#L264-L451).
It does not download a checkpoint, PaliGemma file, or model code.

## Source behavior implemented

The source model's Bridge config declares a 4 × 7 normalized proposal. The
adapter accepts one 256 × 256 RGB `uint8` image and a current eight-value Bridge
EEF pose in the source wrapper's `(x, y, z, qw, qx, qy, qz, gripper)` input
form. Callers must mark it with
`proprio_convention="bridge_eef_quaternion_wxyz"`; an untagged or canonical
Euler state is rejected before a model call. It uses the pinned source
quaternion-to-matrix/euler conversion, then
creates seven model proprio values from position, converted Euler orientation,
and gripper openness. The WXYZ quaternion must be finite and unit-normalized
within `1e-3`; zero or malformed quaternions are rejected instead of being
coerced to an orientation. The input is refreshed for every proposal; no image,
proprio, or action chunk is replayed from a previous tick.

The AutoEval wrapper's `action_normalization_type="bounds"` is also preserved:
it denormalizes action dimensions 0–5 from `q01`/`q99`, while the seventh
normalized action becomes exactly `1` when it is strictly positive and `0`
otherwise. The source OpenPi config's separate `p01`/`p99` `bound` adapter is
not silently substituted for this AutoEval replication behavior.

The native proposal is retained as four rows. AutoEval's run loop passes PiZero
chunks through an external `TemporalEnsembleWrapper(env, 4)`; this repository
does not pin that implementation or infer its effective executed prefix. The
adapter therefore exposes `predict_proposal()` and declares
`certified_execute_prefix=None`. It cannot be used as a one-action policy or
as a qualified feedback controller until the exact execution wrapper and a
golden action/state fixture are reviewed.

## Required staged inputs before a real load

The profile must point to an immutable `plumb-openpi-assets-v1` manifest whose
SHA-256 is itself recorded in the profile. The manifest binds the following:

- clean OpenPiZero and AutoEval checkout HEADs at their pinned commits;
- the original `bridge_beta_step19296_2024-12-26_22-30_42.pt` checkpoint,
  its immutable host revision and full SHA-256;
- the exact source `config/eval/bridge.yaml` checksum;
- the separately staged AutoEval `bridge_orig` bounds-statistics file with
  `q01`/`q99` action and proprio vectors and its full SHA-256;
- the official `google/paligemma-3b-pt-224` model ID, immutable revision,
  accepted-terms evidence URI, and SHA-256 for `tokenizer.json`,
  `tokenizer.model`, `tokenizer_config.json`, `added_tokens.json`,
  `special_tokens_map.json`, `preprocessor_config.json`, and `config.json`.

The implementation requires Python 3.10, Torch 2.5.0, and Transformers 4.47.1,
matching the pinned source project's dependency declaration. PaliGemma is
opened only from its provided local path with `local_files_only=True` and
`trust_remote_code=False`.

The source config is also checked to retain `cond_steps=1`,
`horizon_steps=4`, `action_dim=7`, `proprio_dim=7`, and `flow_sampling=beta`
for the named Bridge beta checkpoint.

The legacy `.pt` checkpoint is loaded only by
`torch.load(..., map_location="cpu", weights_only=True)`. Payloads without a
tensor-only `model` state dict are rejected, compiled `_orig_mod.` prefixes are
checked for collisions, and model coverage is loaded with `strict=True`. The
adapter never uses `weights_only=False`, a generic pickle loader, or a
deserialization fallback.

## Remaining blockers

No OpenPiZero checkpoint, approved PaliGemma support snapshot, AutoEval bounds
statistics snapshot, staged manifest, or matching Python 3.10 policy runtime is
present in this repository. The published checkpoint is historically about
11.77 GB; it has not been downloaded. PaliGemma access/terms and checksum-bound
file provenance must be recorded before use. The source wrapper accepts a
quaternion pose while PLUMB's canonical state record is separately declared in
Euler form, so a provenance-bound conversion fixture is required. Gate A/B,
native generated-image feedback, the execution-prefix fixture, and scientific
qualification remain unresolved.
