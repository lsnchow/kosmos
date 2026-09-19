# Kosmos architecture

Updated: 2026-09-19, after the media-first console implementation.

This document describes the code and the locally verified UI, not a claim that
every ML backend is deployed. “Kosmos” is the product name across the landing
page and console; “Nightshift” is its former name, and `plumb` remains the Python
package. These are parts of the same repository, not different services.

## 1. The product in plain English

The intended product lets you give several robot controllers the same task and
starting scene, watch how each behaves inside a learned world model, and—once
the evaluation is validated—compare their performance.

Three different components have different jobs:

| Component                               | Input                                                                         | Output                                                 | What it does not establish                      |
| --------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------- |
| Policy: the robot's controller          | Camera observations, task or goal image, and any required robot state/history | Arm/gripper actions                                    | Whether those actions will succeed              |
| World model: a learned visual predictor | Starting observation, compatible actions, model-specific conditioning         | Predicted subsequent observations/video                | That the predicted physics match a real robot   |
| Judge: a separate vision-language model | Sampled raw frames, task, rubric and allowed references                       | Structured judgments, subject to validation and quorum | A trustworthy success label without calibration |

A closed-loop rollout repeats observation → policy action → predicted next
observation. The next policy call must receive the correct new observation and
state. Showing a video once, looping its playback, and repeatedly querying a
policy are three different operations.

## 2. What works today versus what remains

| Capability                                   | Current boundary                                                                    |
| -------------------------------------------- | ----------------------------------------------------------------------------------- |
| Local console and API                        | Working; frontend build served by the localhost API                                 |
| Video gallery                                | Working; up to six existing multi-frame recordings, clearly labelled experiments    |
| Recording archive                            | Working; full hash-checked catalog, including small probes and replays              |
| Baseten policy diagnostic                    | Previously verified real SuSIE_LL inference; one action, not a video                |
| Synthetic run engine                         | Working engineering path for persistence, events, cancellation and analysis         |
| Real policy/world adapters                   | Implemented to different readiness levels; contracts can explicitly block execution |
| Matched six-policy video comparison          | Not complete; the gallery must not imply it exists                                  |
| Live world-video generation from the gallery | Not implemented by this UI change                                                   |
| Calibrated automatic success scores          | Not established; rejected judge adapters stay disabled                              |
| Qualified study and ranking                  | Not established; qualification gates are unchanged                                  |

The existing saved clips are genuine world-model outputs, but include supplied
action fixtures and replay interventions. They are not six different policy
results. The full comparison plan is in [DEMO-RESTRUCTURE-PLAN.md](DEMO-RESTRUCTURE-PLAN.md).

## 3. Navigation: where to go and why

Existing URLs are retained so old links and bookmarks still work.

| Visible page      | URL         | Purpose                                                                         | When you need it                                         |
| ----------------- | ----------- | ------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Video gallery     | `/console`  | Watch a small selection of saved model-generated videos                         | Default demo entry                                       |
| Saved runs        | `/results`  | Select a persisted run, then inspect its measurements                           | Investigating a specific run                             |
| Recording archive | `/clips`    | Browse every saved world-model recording; legacy clip exercises are collapsed   | Finding a particular experiment                          |
| Developer tools   | `/live`     | Cloud policy check, synthetic task/run tools, telemetry and run frames          | Testing integrations, not presenting a policy comparison |
| Validation        | `/evidence` | Qualification gates, imported diagnostic reports, collapsed research references | Asking “what can we actually trust?”                     |
| Compute & cost    | `/cost`     | Recorded resource/quality experiments and engineering controls                  | Investigating measured speed/cost, not claiming a bill   |
| Review clips      | `/review`   | Separate manual development-review workflow                                     | Entering explicit review drafts                          |

Only Video gallery and Saved runs are primary navigation. The other destinations
live under **Tools & validation**, which opens automatically on one of those
routes. The archive also has a direct link from the gallery. Mobile uses the
existing **Sections** disclosure.

The old console's 4% simulator / 92% real-robot cards are now under Validation →
Research background. They are cited reference-study values, not app-generated
results. “Pending” means no frozen estimate has been recorded.

## 4. Frontend ownership and data flow

The frontend uses React, TypeScript, React Router, Vite, existing CSS tokens and
Radix dialog primitives. Important files:

| File                                                             | Responsibility                                                                  |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [App.tsx](../web/src/App.tsx)                                    | Routes and shared providers                                                     |
| [AppShell.tsx](../web/src/AppShell.tsx)                          | Header, mobile navigation, sidebar, errors and shared dialogs                   |
| [Sidebar.tsx](../web/src/components/Sidebar.tsx)                 | Human-readable navigation labels and descriptions                               |
| [AppData.tsx](../web/src/AppData.tsx)                            | Shared protocol/run/evidence state, polling, run selection and explicit actions |
| [useRunStream.ts](../web/src/hooks/useRunStream.ts)              | Run event-stream lifecycle and reconnection                                     |
| [OverviewPage.tsx](../web/src/pages/OverviewPage.tsx)            | Gallery entry page and collapsed policy explanation                             |
| [WorldVideoGrid.tsx](../web/src/components/WorldVideoGrid.tsx)   | Small saved-video grid; independent catalog loading                             |
| [WorldVideoQueue.tsx](../web/src/components/WorldVideoQueue.tsx) | Full archive player, queue, downloads and source reports                        |
| [RolloutWall.tsx](../web/src/components/RolloutWall.tsx)         | Developer viewport: saved-video queue or persisted run-frame wall               |
| [lib/api.ts](../web/src/lib/api.ts)                              | API client types and artifact URL handling                                      |

`AppDataProvider` remains outside the router. Navigation does not destroy the
selected run or restart the event stream. Background read-only polling remains;
“no GPU request on page open” does not mean “no HTTP requests.”

The gallery independently performs `GET /api/world-videos`. It filters unsafe
media paths, clips with at most two frames, and duplicate media hashes/URLs,
then takes at most six in server order. This is a deterministic playback
selection, not quality curation, random sampling, or policy matching.

Cards expose the recorded model name, title, duration, native FPS, size,
experimental label, download and source report. Exact task text is not yet a
first-class catalog field: it remains in source reports. The UI does not invent
an instruction when one is missing.

Playback uses native HTML video: muted, inline and looping. IntersectionObserver
pauses offscreen media; page visibility pauses background playback; reduced
motion suppresses automatic playback. **Pause all** disables automatic loops,
while native per-video controls remain usable. At most six grid players mount.
Loading, catalog failure/retry, missing media and decode failure have explicit
states. Playback never invokes a model.

There is no new generated-poster pipeline or interpolated derivative in this
change. The browser displays decoded video frames. The startup animation was
removed so it no longer briefly hides the console contents.

## 5. Local API and the three currently separate execution paths

[plumb/api.py](../plumb/api.py) creates the FastAPI application, API routes,
artifact-serving boundary and static SPA fallback. A built `web/dist` is served
from the same localhost origin. Vite development mode proxies `/api` to port
8787 instead.

### A. Saved world-model playback: read-only

1. The browser requests `/api/world-videos`.
2. [world_videos.py](../plumb/world_videos.py) scans allowed cluster-evidence
   report locations under the data root.
3. Only output fields from recognized diagnostic kinds are eligible. Source
   conditioning videos are not treated as generated outputs.
4. Local MP4 bytes must match the report's SHA-256. Paths, symlinks and size
   limits are checked. Optional `ffprobe` supplies measured container metadata.
5. The catalog returns artifact/report URLs, metadata and limitations.
6. The browser reads `/api/artifacts/...` and plays the existing bytes.

No policy, world model or judge runs in this path. Catalog inclusion proves
which saved file is being displayed, not the validity of its depicted physics.

### B. Cloud diagnostic: one real policy action

1. Developer tools reads `/api/cloud-diagnostics/status` and saved records.
2. Only an explicit **Run cloud model** action POSTs a diagnostic request.
3. [cloud_diagnostic.py](../plumb/cloud_diagnostic.py) binds the request to fixed,
   hash-checked current/goal PNGs and pinned model/deployment identities.
4. The server invokes the configured Baseten CLI profile. Credentials stay
   server-side; they are not sent to the browser.
5. The deployed low-level goal-conditioned policy produces a finite 7-D action.
6. The local service validates the response and stores inputs, raw response,
   identity, timings and result/failure in its separate journal/artifact tree.

The recorded prompt is metadata for this diagnostic; it is not a language input
to the native SuSIE_LL goal-conditioned action sampler. A successful action
response is not a successful robot task or a generated rollout.

Previously verified deployment: model `3mzlenow`, deployment `q929yoj`, profile
`plumb-api`, team `q8grpdw`. These are historical verified identities, not a fresh
cloud health or billing check in this UI task. Runtime/deployment details live in
[BASETEN-MVP.md](BASETEN-MVP.md) and [mvp_policy](../deploy/baseten/mvp_policy).

### C. RunService: synthetic engineering or separately gated real execution

1. An explicit run action POSTs `/api/runs` with its configuration/idempotency.
2. [engine.py](../plumb/engine.py) resolves the backend and checks required gates
   before accepting gated real-model work.
3. [ledger.py](../plumb/ledger.py) persists the run, logical episodes, attempts,
   events and ownership leases. Execution is bounded and cancellable.
4. Workers write attempt artifacts and terminal results. A retry is another
   attempt, not another statistical episode.
5. The UI reads run/episode/analysis/telemetry endpoints and subscribes to
   `/api/runs/{id}/events` for updates.

The current prompt and burst engineering controls use synthetic fixtures.
Their images and labels are not learned-model results. They are useful tests of
the control plane, but cannot become real evaluations by changing their labels.

## 6. The real ML pipeline and why integration is hard

[rollout.py](../plumb/rollout.py) owns the contract-aware rollout controller.
[adapters](../plumb/adapters) and [policies](../plumb/policies) separate model
implementations from that orchestration.

### Policy layer

The target policies are OpenVLA, OpenPiZero, Octo-Small, MiniVLA, SuSIE and
SuSIE_LL. Their observation histories, action normalization, goal formats,
proposal horizons, executed prefixes and state requirements differ. They are
not interchangeable HTTP endpoints accepting an arbitrary image and string.

Source-backed contracts live in `policies/contracts.py`, `policies/native.py`
and `adapters/contracts.py`. Loader modules live beside them. A declared
contract remains blocked until the corresponding pinned-wrapper fixtures and
runtime evidence support it. An adapter file existing does not prove deployed
inference or valid closed-loop feedback.

For example, a policy needing robot proprioception cannot safely be fed an
invented vector extracted from an attractive video. Full SuSIE and SuSIE_LL
are also different paths: a high-level image-goal mechanism plus a controller
is not the same as supplying the low-level controller an existing goal image.

See [POLICY-DEPENDENCIES.md](POLICY-DEPENDENCIES.md) and
[OPENPI-DEPENDENCIES.md](OPENPI-DEPENDENCIES.md) for the loader-specific records.
Those records and fresh runtime checks take precedence over broad readiness
summaries; old blockers must be rechecked before new work.

### World-model layer

`adapters/worlds.py` contains the Cosmos forward-dynamics integration; the
repository also contains IRASim contracts/runtime work. The action compiler
must map source actions into the target world's actual representation, not
silently rename dimensions or units.

The saved Cosmos example takes a source image, 16 recorded actions and the
prompt “Put the pot to the left of the purple item.” It yields 17 frames at
5 FPS, 640 × 480, encoded as a 3.4-second video. The extra frame is the initial
conditioning frame. These are configured nominal times, not measured physical
robot timing.

Changing the MP4's FPS alone speeds up the same frames. Interpolation would be
a separate presentation derivative, not new predicted evidence. Higher
resolution alone cannot fix object warping, contacts or causal feedback.

### Validity and judge layer

[validity.py](../plumb/validity.py) performs deterministic frame/action/state
checks and calibrated motion-consistency checks. Ambiguous or unsupported
checks remain `unknown`; missing calibration is not permission to guess.

[policies/judge.py](../plumb/policies/judge.py) accepts a fixed 16-frame clip,
including required endpoints, with task/rubric/reference context. Each judgment
sees those sampled frames together. The frozen primary protocol calls for five
judgments with a three-vote quorum. It is not one judge call per playback frame.

Per-rollout **Stage A validity / Stage B judge** are distinct from project-wide
**qualification gates A–F**. The reused letters do not mean the same thing.
The UI cleanup does not calibrate the judge, pass a gate or enable old adapters.

## 7. Durability, cloud topology and trust boundaries

| Storage / module                                     | Responsibility                                                                  | Boundary                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Data-root `plumb.sqlite3`; `ledger.py`               | Logical runs, episodes, attempts, events and leases                             | Logical intent differs from transport attempts                   |
| Run/attempt artifacts; `artifacts.py`                | Persisted outputs and metadata                                                  | Never make terminal records point at unwritten output            |
| `cloud-diagnostics.sqlite3` and `cloud-diagnostics/` | Isolated policy diagnostic journal and evidence                                 | Separate from RunService and qualification                       |
| `cluster-evidence/`                                  | Imported model diagnostic reports and output files                              | Recordings, not live requests                                    |
| `outbox.py`, `platform.py`, `backends/baseten.py`    | Durable async submission, authenticated callback association and reconciliation | An ambiguous POST must not be blindly reissued                   |
| `result_store.py`                                    | Explicitly bound, integrity-checked remote result storage                       | A callback is a notification, not durable output storage         |
| Development-review private SQLite                    | Reviewer-entered drafts                                                         | Outside served artifacts; not automatically calibration evidence |

The small deployed policy MVP uses an isolated ML worker process to separate
its JAX/Flax/TensorFlow/CUDA dependencies from the HTTP/Truss process. It does
not represent deployment of the entire [Chain topology](../deploy/baseten/chain.py).

That larger topology describes a controller, policy workers, world worker,
deterministic validity worker and judge worker. Workers perform real adapter
calls or return explicit blocks. Its code and local rehearsals are not proof
that all workers are running on Baseten today.

Diagnostic startup marks interrupted running work ambiguous and pending work
failed. Local persistence cannot recover a remote output that never reached
the laptop. The broader outbox/result-store path has distinct lifecycle
requirements; do not assume its guarantees automatically cover the diagnostic.

The app is localhost-oriented. Review writes have their own loopback-origin,
cookie and private-storage rules. Do not expose this console publicly on the
assumption it has production multi-user authentication or tenant isolation.
See [DEVELOPMENT-REVIEW.md](DEVELOPMENT-REVIEW.md).

## 8. Measurements, qualification and latency

[measurement.py](../plumb/measurement.py) derives summaries from persisted
results, including coverage and missing-outcome bounds. It distinguishes
descriptive quantities from supported statistical inference. A missing result
does not become a success, and a displayed ordering need not be a supported
ranking. [gates.py](../plumb/gates.py), protocol records and calibration evidence
control stronger claims.

The broader study remains six policies × five tasks × 50 starts. The demo
gallery neither completes that study nor removes its requirements.

Historical observations—not new measurements or a live-demo SLA:

| Path                                | Recorded timing                         | Exclusions / interpretation                                  |
| ----------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| Cosmos sample on cluster            | 4.107 s inference; 169.793 s model load | Not Baseten end-to-end or a policy feedback loop             |
| 16-tick OpenVLA → IRASim diagnostic | 142.214 s total                         | Includes loading/artifacts; visible drift, not qualified     |
| Baseten policy-only warm requests   | Approximately 0.56–1.28 s client time   | One action, not a world video                                |
| Baseten policy-only cold requests   | Approximately 31.9–54.8 s client time   | Cold-start overhead varies                                   |
| Gallery playback                    | Local HTTP/media decoding only          | No inference; exact first-paint latency not benchmarked here |

Do not multiply a single-action or one-shot-render timing into a claimed
six-policy comparison duration. Measure queue wait, warm-up, policy calls,
world inference, encoding, persistence and browser readiness separately, then
measure time to first tile and time to the whole set.

## 9. Remaining work for the intended product

In dependency order:

1. **Quality pilot:** verify a compatible world profile with acceptable temporal
   consistency, action response and native frame rate. Preserve failures/raw
   output; measure memory, load and inference costs.
2. **Authentic policy loops:** finish missing assets/loaders and verify each
   policy's native observation/action/feedback contract. Do not fabricate state.
3. **Matched comparison manifest:** bind shared task/start/goal/horizon/world
   profile and each actual policy identity to immutable results. Preserve
   failures and disclose cadence differences. Avoid best-seed-per-policy curation.
4. **Curated media pipeline:** explicit posters, raw masters, lightweight
   previews and any labelled interpolation. Expose exact conditioning in the
   API and inspector rather than requiring a report download.
5. **Live generation jobs:** a distinct bounded, durable world-video workflow
   with queued/warming/generating/encoding/ready/failed states, stored result
   recovery and explicit reruns. Keep saved results visible while new jobs run.
6. **Scientific scoring:** independent calibration data, frozen judge/validity
   settings, completed qualification evidence and appropriate uncertainty.
7. **Production hardening:** authentication/authorization, durable hosted media
   and result storage, quotas/budgets, observability, reconciliation operations,
   cold-start handling and measured concurrency. The large frontend bundle also
   remains a code-splitting follow-up.

For the visual MVP, steps 1–5 matter most. A calibrated ranking is not required
to honestly demonstrate videos, but it is required before selling those videos
as validated policy performance. More judge fine-tuning is not a substitute for
correct world feedback and independent labels.

## 10. Local operation and verification of this UI change

With the existing API on port 8787, rebuild the frontend and reload Console:

```bash
npm --prefix web run test
npm --prefix web run build
```

The existing API serves the rebuilt files; this UI change did not require a
backend restart or cloud deployment. Open <http://127.0.0.1:8787/console>.
For startup from scratch, use the root README or the pinned diagnostic command
in [BASETEN-MVP.md](BASETEN-MVP.md); do not start a second listener on that port.

Verified after implementation:

- 263 frontend tests passed, including gallery filtering, read-only requests,
  pause/resume, offscreen playback, reduced motion, errors/retry and axe checks.
- Typecheck and production build passed. Vite still warns about a >500 kB chunk.
- Headless Chromium loaded all six gallery MP4s, advanced playback, and paused
  all six through the visible control.
- Desktop navigation to Saved runs, Validation, Compute & cost, Developer tools
  and Recording archive worked; the mobile Sections menu navigated correctly.
- A 390 px viewport had no horizontal overflow; reduced-motion videos remained
  paused; screenshots were visually inspected.
- Those browser checks saw no JavaScript errors and no non-GET requests.

No model generation, GPU job, judge training, gate mutation or deployment was
performed for this UI cleanup. Python backend code was unchanged; its complete
test suite was not rerun for a frontend-only change.
