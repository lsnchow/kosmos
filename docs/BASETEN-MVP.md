# Baseten policy diagnostic MVP

The real ML path is **localhost → Baseten H100 → validated action → durable
local report → browser**. Open <http://127.0.0.1:8787/live#cloud-diagnostic> and
use **Run cloud model**. The separate task-prompt/burst controls are explicitly
synthetic rehearsals, not cloud rollouts.

## Deployed identity

- Team: **33 / q8grpdw**; CLI profile: **plumb-api**.
- Model: **3mzlenow**, `plumb-susie-ll-gcbc-static-goal-mvp`.
- Verified deployment: **q929yoj**, `mvp-gcbc-v6-isolated`, promoted to the
  model's Baseten `production` environment. That environment name is not a
  scientific qualification or a calibrated-judge claim.
- One H100 maximum; minimum zero; idle scale-down delay 60 seconds.
- The key remains in the CLI's credential store. Neither the browser nor this
  repository needs its value. The local runner explicitly selects the profile
  and removes ambient API-key/token overrides from its subprocess environment.

Start/restart the local API and built frontend from the repository:

```bash
npm --prefix web run build
.venv/bin/python scripts/serve_baseten_mvp.py \
  --model-id 3mzlenow --deployment-id q929yoj \
  --profile plumb-api --timeout-seconds 600
```

The running instance is kept in LOCAL tmux `plumb-live`. Do not start a second
process on the same port/data directory. This MVP journal is single-process;
it is not a multi-worker distributed scheduler.

## What runs

The public `patreya/gcbc-bridge` checkpoint is pinned to
`1a4c15dd9ad780a257e9494f0fac79cbe8e64793`, its bytes and publisher MIT README
are hash-checked, and the SOAR inference source is pinned to
`eabd5f16a856e484884a22e257a941bb358cea08`.

Truss's HTTP process and the ML process use separate Python environments.
The persistent child owns JAX 0.4.20, CUDA 12.2/cuDNN 8.9, Flax 0.7.5,
TensorFlow 2.15.0 and protobuf 4.25.8. It requires a JAX-visible GPU and uses
strict, full-tree parameter restore, **not** optimizer/training resume.
There is no CPU fallback. Requests are serialized, bounded and hash-bound;
the localhost validator independently recomputes the physical action from
the returned normalized action. No model/path/URL override is accepted.

Inputs are two distinct 256px frames from the pinned vendor video. The final
video frame is demonstrated goal conditioning, not a certified task goal.
The Mac decode/resize profile has its own pixel hashes; it must not be called
pixel-identical to the previous cluster fixture. No weights are on the laptop.

## Observed verification

- Browser request `cloud-e2dd4f395307440bae93300840adc632` completed with a
  finite 7-D action, matching input/model hashes and a verified changed
  parameter digest. The response identifies an NVIDIA H100 80GB.
- `cloud-repeat-20260919-01` returned exactly the same normalized and physical
  actions. Its model invocation was 0.0368s; the CLI/cloud round-trip was
  0.5648s. The first invocation included compilation: 13.369s model time,
  15.381s request time. These are individual diagnostics, not throughput claims.
- An intentionally invalid PNG request was rejected by the cloud model;
  `cloud-recovery-20260919-01` then succeeded on the same loaded worker.
- Browser reload, actual API process restart, saved-report readback,
  duplicate-ID replay without inference, and local invalid-input rejection
  passed. Desktop and 390px mobile browser checks found no JavaScript errors
  or horizontal overflow after the integration fixes.
- After observing **SCALED_TO_ZERO / zero replicas**, request
  `cloud-cold-20260919-01` woke the model and completed: 31.900s client request,
  14.869s model load, 2.769s inference. Its physical action differed from the
  first worker by at most approximately 0.00000056. Warm repeats were exact;
  bitwise identity across cold workers is not claimed. Raw values are retained.

## Persistence and failure semantics

Data lives under `data/live-integrated/`:

- `cloud-diagnostics.sqlite3`: request journal.
- `cloud-diagnostics/<request-id>/`: original input envelope, raw CLI response,
  and validated result or failure. Endpoint/revision identity is included in
  idempotency; a request ID cannot silently switch models or deployments.
- `cloud-diagnostic-fixtures/susie-v1/`: PNGs and source/transform manifest.

Cloud build/runtime evidence and the intentional malformed-request artifacts
are under `data/baseten-mvp-evidence/`. Prior failed deployments are retained
but inactive; their failures were not rewritten into successes.

Timeouts, lost responses and interrupted in-flight work never create a success
or trigger an automatic inference retry. A reviewer must explicitly acknowledge
an unresolved result before creating a new request, which may add compute.
Local persistence is verified; this is **not** an exactly-once remote result
store that can recover an output lost before it reached localhost.

## Boundaries

This MVP produces a real **policy action**, not a generated video rollout or
success score. All A–F qualification gates remain unchanged. The biased judge
adapters remain disabled. Reliable world feedback, remaining policies/scenarios,
independent judge labels, the full study and measured scaling/cost claims are
still separate work. Billing credits were not established by training quota;
do not treat available H100 capacity or zero reported usage as free compute.
Organization SSH is disabled; it was not needed for this managed deployment.
