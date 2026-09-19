# Parallel-branch integration handoff

The tested num2 continuation is on **`num2-verified-continuation`**, based on
`425a6dd`. Its implementation commit is `f5eddee`. The 212-pass test result and
cluster evidence in `HANDOFF.md` apply to this branch.

While num2 ran, Kevin independently advanced `origin/main` through `bdd7efd`
and `6625b17`. That line adds a substantially larger end-to-end backend,
artifact/protocol/scenario/rehearsal/sweep stack and a reorganized frontend.
It also adds `BUILD-SPEC.md`, `SCRIPT.md`, and `reverse_validation.py`.
Those changes are preserved on main; they have not been merged or runtime
validated by this continuation. Do not overwrite either line or force-push.

## Concrete merge boundary

A read-only `git merge-tree --write-tree --name-only` check against
`6625b17` reported conflicts in these files:

```text
plumb/annotation.py
plumb/api.py
plumb/calibration.py
plumb/cli.py
plumb/ledger.py
plumb/policies/__init__.py
plumb/policies/minivla.py
plumb/policies/native.py
plumb/policies/octo.py
plumb/policies/susie.py
plumb/scenarios.py
tests/test_calibration.py
tests/test_policies.py
tests/test_scenarios.py
```

Resolving text conflicts alone is insufficient: the two branches introduced
different public interfaces and persistence paths. Use a separate integration
branch/worktree, preserve both source histories, and verify the combined result.

## Evidence and behavior to carry forward

- Preserve main's additional backend/UI work while reconciling its interfaces
  with this branch's durable outbox and validation workflows. This branch's
  outbox tests cover callback-before-ack, ambiguous POSTs, exact signed raw-byte
  retention, cancellation races, artifact-before-terminal ordering, strict
  label/identity/horizon checks, and fixture-versus-cloud ownership isolation.
- **Do not transfer the Octo API fix without its code profile.** The actual
  successful GPU tests here use source `37951e4…` (v0.1), whose sampler returns
  normalized actions and rejects `unnormalization_statistics`. Main's
  requirement file instead pins `241fb3514b7c40957a86d869fecb7c7fc353f540` and
  calls the newer API. Test that different code/checkpoint/runtime tuple on an
  allocated GPU; neither the older success nor its API failure certifies it.
- Preserve MiniVLA's VQ-license and auxiliary-asset blockers, the exposed-one-
  action versus seven-future-action configuration distinction, strict source/
  artifact checks, and separate SuSIE replication/sensitivity identities.
- Main adds `openpizero.py`; this branch adds `openpi.py`. Reconcile their
  interfaces deliberately. This branch requires tagged, finite unit WXYZ
  quaternion input and does not silently interpret canonical Euler8 as a
  quaternion. Its external execution-prefix certification remains missing.
- Keep the full 6×5×50 matrix, native feedback boundaries independent of video
  request partitions, scientific coverage as evaluable `V/1500`, immutable
  source-lineage splits, raw judge evidence, and genuinely fresh rehearsals.
- Keep all local and cluster diagnostics separate from fixture/rehearsal
  outputs. The actual history replay still distorts; native policy calls and
  eight-worker reproducibility do not qualify physical fidelity or task scores.
- Review the provenance and claims of the newly present original-material
  filenames before treating the prior missing inputs as resolved. The reviewed
  `AGENT-BUILD-SPEC.md` qualification and no-unsupported-claim requirements
  remain the reference for this continuation.

All raw GPU evidence remains in ignored local `data/cluster-evidence/` and
cluster scratch. Models remain cluster-only. The active allocation and local
API state are documented in the current first sections of `HANDOFF.md` and
`CLUSTER-OPERATIONS.md`; inspect live state before using their IDs.
