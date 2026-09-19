# Go live — the sequence once GPU and Baseten access exist

This document separates implemented software from the runtime and human
evidence still required. Re-run checks against the current integrated revision;
earlier successful fixtures do not certify changed adapters.

Run this first, any time, to see exactly where you are:

```bash
.venv/bin/python scripts/e2e_smoke.py
```

It reports every check as **pass**, **pending** (correct but waiting on hardware,
credentials or people — with the blocker named) or **fail** (something is
broken). It exits non-zero only on a `fail`. When the steps below are done, run
`--require-qualified`, which turns every remaining `pending` into a failure.

The current check output is authoritative; do not reuse an old pass/pending
count. Actual account access, matched source data, licenses, and two human
annotators remain external inputs. Missing inputs remain blockers.

---

## What is already verified without a GPU

The integration path is not "written but untested". `scripts/rehearse.py` runs the
**entire production code path** against a simulated Chain that speaks the real
protocol — the real `/async_run_remote` envelope, real HMAC-signed webhooks, real
queue-status and deployment shapes:

```bash
.venv/bin/python scripts/rehearse.py --starts 50     # the full 1,500-episode matrix
```

The full-matrix rehearsal completes 1,500/1,500 episodes and exercises:

- submission envelope construction, validated against the Chain's **own** pydantic
  models and its own `_world_setup` (so a field-name drift fails loudly instead of
  silently nulling every measurement — this caught two real bugs),
- the durable outbox, with one logical row per
  `(run_id, policy_variant, task, start_id, world_seed, protocol_hash)` and
  idempotent resubmission,
- HMAC callback authentication over the raw body, including callbacks that arrive
  **before** the local submission commits (parked, then bound),
- ~25 deliberately dropped and ~30 deliberately duplicated webhooks per run, with
  the reconciler resolving the former and ignoring the latter,
- telemetry, the two cost views, the ledger, measurement, all artifact writers,
  and the dashboard feed.

Timings measured locally (all with a simulated Chain, so none is a model or
platform measurement): 1,500 episodes in ~150 s, fast analysis 0.19 s, full
statistics on publish ~43 s.

**What the rehearsal cannot tell you:** anything about a model, a real latency or
a real cost. That is what the steps below are for.

---

## Step 1 — credentials (5 minutes)

```bash
cp .env.example .env
```

Fill in `BASETEN_API_KEY`, `BASETEN_WEBHOOK_SECRET`, and `PLUMB_WEBHOOK_ENDPOINT`
(a publicly reachable URL that forwards to this app's `/api/callbacks/{run_id}`).
`BASETEN_CHAIN_ASYNC_URL` comes from step 2.

`.env.example` documents every variable, including the exact accepted shape of the
Chain async URL. Note that `/async_predict` is a **model** route and is rejected —
Chains use `/async_run_remote`.

## Step 2 — deploy the Chain (30–60 minutes)

Follow `deploy/baseten/DEPLOY.md`, which is 11 ordered steps and names the
`model-contracts.json` nulls each one fills. In short:

```bash
python -m venv .venv-deploy && .venv-deploy/bin/pip install truss
.venv-deploy/bin/truss chains push ./deploy/baseten/chain.py --environment production
```

Nine chainlets: a CPU controller, five per-stack GPU policy workers, a
micro-batched H100 world worker, a CPU validity worker, and a GPU judge worker.
The Chain has already been validated against the real `truss==0.18.30` framework
validator — zero errors, all nine construct — so a push should not surprise you.

Then record the Chain ID into `BASETEN_CHAIN_ASYNC_URL`. That is the whole step.

### You do not need a raised replica cap

`BUILD-SPEC.md` scripts a booth ask for 100 replicas. The arithmetic says that
was the wrong knob, and `plumb/capacity.py` works it out from measured inputs:

- **Per-episode latency is irreducible by replica count.** Each episode's
  policy/world loop is sequential (spec section 7); only episodes parallelise. At
  OpenVLA's required native cadence a 100-tick task is 100 sequential rounds, so
  with the recorded latencies one episode needs ~536 s. `smallest_sufficient_cap`
  returns `None` there — not a large number, *impossible*.
- **Batch fit requires measurement.** The recorded36.5GB single-call peak does
  not determine multi-batch memory, which also depends on shared weights,
  activations and workspaces. The planner uses batch1 unless a matching measured
  batch profile establishes another setting. Linear extrapolation is a projection.
- **Per-call latency beats replicas.** Going from 4.52 s to 0.30 s per call buys
  more than going from 1 to 100 replicas — and that is a resolution and
  denoising-steps decision entirely in your control, which is what the
  cost/fidelity sweep is for.

`plan_burst()` can report a projection or an explicitly labelled subset when
capacity is insufficient. A subset does not satisfy the requested full burst:
the target remains1,500 complete episodes in60s for total demonstration cost
at most$11.25, confirmed in three fresh rehearsals. Report an unmet target when
those conditions are not achieved. Unmeasured batching cannot support measured
fit or throughput claims.

## Step 3 — stage the assets (1–3 hours, mostly download)

```bash
python -m cluster.download_assets --list-plans
python -m cluster.download_assets --from-plan policy.openvla --execute
# ... and so on; 27 entries in cluster/asset_plan.py
.venv/bin/plumb assets-lock            # re-run to fold in the real hashes
```

Every row stays `verified: false` until an actual retrieval records a hash, byte
length and license. Three traps are already encoded so you cannot hit them: Octo's
checkpoint is not at the repo root and must be v1.0, MiniVLA's filename needs
`%3D`, and `Stanford-ILIAD/pretrain_vq` **declares no license** — it is flagged
`redistribution: prohibited_pending_resolution` and requires an explicit
acknowledgement to load.

MiniVLA and OpenPiZero ship legacy `.pt` pickles which the adapters refuse. They
need a safe conversion following the `cluster/convert_irasim_checkpoint.py`
pattern (`safe_globals` allowlist, `strict=True`, refuse-to-overwrite).

## Step 4 — Gate A, and the probe that unblocks four of five tasks

```bash
python -m plumb.adapters.smoke cosmos --report results/gate-a.json
```

Run the pinned vendor fixture through the **actual deployed payload** and record
shapes, frame order, normalization, measured GPU memory, and warm **and** cold
latency.

Then the probe that matters most: **sweep `action_chunk_size` over
{1, 2, 3, 4, 6, 8, 12, 16}** and certify which lengths the deployment supports.
`allowed_action_lengths` is currently `(16,)`, and with only 16:

| task | horizon | representable as sums of 16? |
|---|---|---|
| open_drawer | 70 | **no** |
| close_drawer | 70 | **no** |
| to_basket | 100 | **no** |
| to_sink | 100 | **no** |
| fold_cloth | 80 | yes (5×16) |

Certifying N=4 (already observed working) makes basket, sink and cloth exact.
**70 mod 4 == 2**, so the two drawer tasks additionally need a certified
non-multiple-of-four length — certify any of 1, 2, 3, 5 or 6 — or a
`TerminalPaddingCertificate` carrying measured prefix-invariance evidence. The
controller refuses to pad without one, and a padded run carries a distinct
protocol identity.

Also sweep the micro-batch size and record the GPU-seconds-per-tick curve. That
curve is the burst budget (see "the arithmetic" below).

## Step 5 — Gate B, the decision that sizes the study

```bash
python -m plumb.adapters.probe_suite --report results/gate-b.json
```

Five paired control arms (original, zero, temporally permuted, sign reversed,
cross episode) plus stationary-contact and already-successful controls, then the
**suffix-causality** test. That test decides whether OpenVLA runs at 84 ticks per
episode or at a chunked-prefix approximation — roughly a 6× cost difference.

Gate B already *failed* on the cluster for the pinned Cosmos/OpenVLA one-step
profile. The Chain reports that rather than padding around it. Both outcomes are
publishable; only the label differs.

## Step 6 — Gate C scenario panels

`docs/GATE_C_RUNBOOK.md`. Drawer starts come from `zhouzypaul/auto_eval` (real
WidowX video) and get `matched_provenance`; basket/sink/cloth come from
`IPEC-COMMUNITY/bridge_orig_lerobot` and get
`matched_distribution_with_limitations`, because AutoEval's trial-level starts are
not published. **Sink and cloth assets are still an acquisition dependency** — the
code raises rather than substituting drawer starts.

Until `scenarios.jsonl` exists, a real submission is refused with
`start_unresolved`: the Chain needs each start's actual conditioning frame and 8-D
Bridge state, and there is no synthetic fallback on the production path.

## Step 7 — Gate D judge calibration

`docs/GATE_D_RUNBOOK.md`. 150 clips (100 development / 50 held-out, 20/10 per
task), frozen before annotation.

**This is the one step that needs people.** Gate D wants two blinded human
annotators. The annotation surface is built (`/api/annotate/*`, `plumb annotate`),
`annotator_type` is a closed enum, and a model annotator can never be recorded as
`human`. With no human labels, Gate D is blocked. A development report named
`pass_with_limitations` is not a Gate D pass and cannot unlock qualified
scoring. The annotation workflow is documented in the runbook; its completion
time depends on the clips and annotators.

## Step 8 — freeze and preregister

```bash
.venv/bin/plumb freeze-protocol
git add protocol.json assets.lock.json && git commit -m 'Freeze protocol-v1'
git tag -s prereg/protocol-v1 -m "protocol_sha256 <hash>"
git push origin prereg/protocol-v1
.venv/bin/plumb prereg-status
```

A local hash is **not** preregistration — spec §6 is explicit. The signed tag
pushed to the remote is what supplies the external timestamp, and
`prereg-status` will keep reporting `unregistered` with its blockers listed until
it exists. That tag is also the artifact `BUILD-SPEC.md` wants quoted on stage
("goalposts we cannot move until we cross them").

## Step 9 — the operating point, then the primary run

Run the cost/fidelity sweep **before** the primary study, select the cheapest
setting meeting the preregistered tolerances on the development panel, then
confirm it on the disjoint panel. `plumb/sweeps.py` refuses to call an
unconfirmed selection qualified.

Then the primary run: 6 policies × 5 tasks × 50 matched starts.

```bash
.venv/bin/plumb publish <run_id>      # writes all 14 spec section 8 artifacts
.venv/bin/plumb gates                 # every gate and its blocking evidence
```

## Step 10 — Gate F rehearsals, then the demo

Three fresh 1,500-episode rehearsals. Prewarmed weights are allowed; cached
actions, segments, videos and judge outputs are not. Gate F's evidence check
already enforces all 1,500 completing, no terminal service failures, ≤60 s and
≤$11.25, and it **refuses best-of-three** — all three are reported, failures
included.

```bash
.venv/bin/python scripts/e2e_smoke.py --require-qualified
```

---

## The arithmetic you should know before the burst

The claim is 1,500 episodes / 60 s / ≈$11. Per spec §7 each episode's
policy↔world feedback is sequential; only episodes parallelise. Horizons are
70/70/100/100/80 ticks, mean 84.

At OpenVLA's native one-action-per-frame cadence that is 1,500 × 84 = **126,000
forward-dynamics calls**. Sixty seconds at 100 replicas with `concurrency_target=1`
is 6,000 replica-seconds, requiring **≤0.048 replica-seconds per action** on
average, including overhead. This is a target bound, not measured feasibility.

Batch packing may improve aggregate throughput, but its speedup and memory cost
must be measured. It does not remove each episode's sequential feedback path:
a 100-tick episode must also finish its dependent policy/world calls, scoring,
and overhead within 60 seconds. A measured 16-action forward latency is not a
measurement of native one-action feedback. The micro-batcher is implemented;
neither its tests nor single-call memory estimates establish a feasible batch
size or the burst target. Gate A needs the actual supported native profile and
measured batch/latency evidence.

If the target is missed, report the measured number. Spec §7 forbids silently
shrinking to 400 episodes or replaying cached video as live, and `SCRIPT.md`
already frames the $11 as a *promise* — if the receipt says 180 s, the stage says
180 s.

---

## Things that stay true no matter what

- A gated backend will not execute until gates A/B/C carry passing evidence, and
  the refusal names its blockers.
- `qualified` is not a settable mode. Only
  `plumb.gates.QualificationValidator` decides, from frozen manifests.
- Free-play returns **503 with a named reason** rather than a placeholder frame.
  The whole point of that beat is that the video is really being generated.
- Telemetry that is unavailable is `null` with a status string, never `0`.
- Model-annotator calibration is never called human calibration.
- A rehearsal episode carries `transport: "simulated"` on its ledger row and its
  artifact manifest, and its manifest records `world_model: false` — because no
  world model ran.
