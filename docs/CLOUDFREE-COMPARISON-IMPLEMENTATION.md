# Cloudflare-free comparison implementation

Execution record, 2026-09-20. This is not a claim that the complete product is
ready. The current slice is explicitly unscored; the scientific judge, gates,
reliability report, and large burst remain separate work.

## Architecture that is now implemented

The browser requests a readiness/cost quote, then explicitly creates a matched
comparison. `plumb/comparisons.py` freezes the source, three policies, four seeds,
world profile, and budget reservation in SQLite before dispatch. Its background
receiver sends one authenticated streaming Baseten Chain request per cell.
The browser never receives the provider credential.

`deploy/baseten/demo_chain.py` contains a CPU controller, three isolated policy
workers, and one selected world worker. The controller owns the 70-step loop.
Each policy turn consumes the previous generated observation; Octo serializes
its source history/temporal ensemble, and MiniVLA retains its actual one-row
source execution contract. Unexecuted proposal rows are not padding.

The Chain streams identity, stage, heartbeat, PNG, and terminal records. Local
PNG bytes are hash-checked and committed before SSE announces them. SQLite is
the replay/reload authority. A missing terminal record stays incomplete or
ambiguous, with no automatic remote retry. There is no Cloudflare, public
callback, R2 bucket, or remotely durable result-store claim in this path.

The React wall uses policy columns and seed rows, stable cell IDs, and actual
persisted frames only. Only a complete, explicitly promoted 12-cell set is the
precomputed presentation. The legacy gallery remains separate. Manual branches
bind their source frame/state and advance only after a complete segment. Key
repeat, keyup, hidden tabs, and same-render double presses do not enqueue work.

## Four priorities and real acceptance status

| Priority | Implemented | Remaining acceptance |
| --- | --- | --- |
| Baseten execution | Real CPU streaming Chain; deployed H100 Cosmos worker; real returned PNGs/MP4 | Full three-policy Chain GPU rehearsal, profile/readiness evidence |
| Coherent action-responsive video | Pinned assets; actual right/left probes; input-type defect diagnosed and corrected | Visually acceptable corrected output, paired directional controls, multi-step drift test |
| Three policies × four seeds | Native policy router; fixed 12-cell/70-action journal; full CPU end-to-end test | MiniVLA assets/rights, Octo source-proprio replay, world native feedback cadence, real matrix |
| Fullscreen steering | UI, durable branch flow, 16-frame receiver, key handling, actual world-only Chain probe | Browser-to-real-GPU steering on an admitted source/profile, second-command chaining, mobile rehearsal |

The current admitted wall/manual manifest is Cosmos-specific. IRASim's actual
15-action/16-structural-frame contract is implemented in its adapter but not
yet admitted by the comparison service/UI. Do not pad it to 16 actions or
describe that remaining integration as finished.

## Deployed evidence

Use team **33 / q8grpdw**, CLI profile **plumb-api**.

- CPU Chain `5qekn2qo`, deployment `rwnpr943`: actual cross-Chainlet receipt
  stream passed. Cold completion 56.40 s; warm completion 1.72 s. These are
  transport measurements, not model inference latency.
- World probe Chain `yqv10xq8`: the first deployment `owpjdv8w` lost its outer
  stream after 60 s of silence. Provider logs show the GPU call later ended.
  Its output was not recovered or called complete.
- Deployment `7wl1e2eq` added heartbeats and delivered 16 post-conditioning
  frames plus an MP4. Cold end-to-end ~136 s; warm opposite-direction ~36 s.
  The images were unacceptable and remain negative evidence.
- Root cause: the wrapper sent RGB uint8 NumPy directly into a Diffusers path
  that treats NumPy as normalized floats. The pinned existing smoke used PIL.
  Deployment `03yd07l3` contains the PIL fix; its probe must be inspected before
  accepting visual quality. Do not infer a pass from deployment status.
- Old world deployments are deactivated. Current settings are min 0 / max 1,
  concurrency target 1, scale-down delay 60 s. Recheck before continuing work.
- The original SuSIE_LL diagnostic `3mzlenow/q929yoj` is unchanged. It is not
  this Chain and does not generate these videos.

Evidence lives under `data/baseten-comparison-evidence/`: immutable stream
receipts, source manifest, per-probe requests/events/PNGs/MP4/reports, and browser
traces. It is intentionally not primary scientific evidence.

## Source and asset boundaries

The drawer start is Bridge episode 31353 at immutable dataset revision
`0e9d76d07e9df3ea3eba257b2520d4913833fad2`. Its frame is genuinely 256×256 RGB
and its state/action are from the co-indexed, hash-verified Parquet row.
It is an existing development-only lineage, not a formal study/calibration
start. Raw gripper state is about 1.00015, unlike the spec's 0..0.39 convention.
No guessed normalization is applied. The mechanical pose probe holds the
recorded gripper **action** constant and does not feed the state into a policy.

Cosmos uses the pinned public publisher cache and a packaged per-file SHA lock.
Weights are verified in the worker, not copied to the laptop. Baseten BDN volume
listing is disabled for this organization, but standard ModelRepo caching
worked. No organization setting or other team's resources were changed.

MiniVLA VQ rights/asset checks remain blocked. Octo source execution is not its
raw four-row proposal. Cosmos accepting 16 actions does not prove that it can
deliver the one-action causal feedback needed by these native policies.
Do not remove those checks merely to populate a wall.

## Verification commands

```sh
.venv/bin/python -m pytest -q
cd web && npm test && npm run build
```

`tests/test_comparisons.py` exercises the actual service request translator,
controller, stream, and SQLite receiver with only policy/world inference faked.
It covers all 840 cell actions, a manual segment, branch rebinding, missing
terminal records, idempotency, and manual budget exhaustion. This is CPU
contract evidence, not a GPU fidelity result.

The real-browser smoke does not mock API routes or submit inference on entry:

```sh
.venv-deploy/bin/python scripts/smoke_comparison_browser.py \
  --output-dir data/baseten-comparison-evidence/browser-check
```

Once an actual complete set exists, add `--require-comparison`. Only an explicit
`--steer --require-comparison` sends one real ArrowRight command. Preserve its
trace, screenshots, command receipt, 16 frames, and terminal record. Follow with
a separately recorded second command to verify source chaining. A blocked-state
browser pass does not satisfy that positive GPU acceptance test.

## Budget and continuation

Incremental cap is $100, with no assumed shared credit. $5 is reserved for CPU
deployment/stream verification and $15 for bounded single-H100 probes, including
load and idle residency. These are reservations, not reconciled charges.
Set the localhost comparison admission cap to $80 while that $20 is reserved
outside its journal. Manual commands also reserve controller/world residency;
failed or ambiguous work is not silently refunded.

Next: inspect the corrected probe; deploy/measure the once-per-worker asset
verification optimization; obtain opposite-direction and reset controls; then
resolve native feedback and the remaining policy prerequisites before any
full-wall collection. Human calibration, qualified rankings/reliability,
distillation, and the elastic burst are not delivered by this preview work.
