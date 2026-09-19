# Demo restructure plan — video grid first

Status: planning only. No render, deployment, autoscaling, or application change
is authorized/executed by this document itself.

## Confirmed product decision

Lucas chose: **pre-generated video grid immediately; optional live generation**.
Keep the existing visual design. Reorganize the experience around visible video
results, not low-level diagnostics. The public landing page can remain; entering
the dashboard should open the gallery rather than the current diagnostics page.

Lucas also chose: **different policies on the same task and starting scene**.
The main grid is therefore a matched policy comparison, initially six tiles
(OpenVLA, OpenPiZero, Octo-Small, MiniVLA, SuSIE, SuSIE_LL), not six unrelated
scenes or six seeds of one model. Task/scene/seed selection changes the whole
comparison set. Do not relabel the existing37 probes as policy results.

“Images” is provisionally interpreted as the posters/first frames of actual
generated video clips. Use compatible, provenance-backed robot observations for
evaluation starts. Newly invented AI scenes would be a separately labelled visual
demo input, not quietly substituted for real robot observations.

## What the current system actually does

1. Baseten SuSIE_LL: current + goal images → one7-D action. Its text field is
   logged but is not a language input to this goal-conditioned policy.
2. Recorded world videos: existing Cosmos/IRASim outputs, including short probes
   and repeats. Playback never invokes a model.
3. TaskPrompt currently launches synthetic engineering fixtures, not live video.
4. The judge is a separate VLM path:16 sampled frames jointly per judgment;
   the frozen primary protocol specifies five judgments and3-vote quorum.
   It is not the video generator and is not calibrated/deployed for MVP scores.

The exported Cosmos sample has17 frames at5fps (3.4s encoded duration),640×480,
30 denoising steps. Its actual world prompt is:
**“Put the pot to the left of the purple item.”** It also received a source
conditioning image and16 recorded vendor actions. These facts are in
data/live-integrated/cluster-evidence/cosmos-smoke-937575.json and the prompt is
passed to the pipeline in plumb/adapters/worlds.py. It is not a text-only render.

FPS is configured in both the pipeline and exported video; nominal timing is
not a measurement of robot timing. Raising only export FPS speeds up the same
17 frames. Increasing resolution does not fix missing motion or object warping.

## Target user flow

1. Enter dashboard: cached posters are immediately visible in a responsive grid.
2. Visible tiles become muted looping clips as their media loads. No GPU request
   is made by opening the page, scrolling, looping, or inspecting a saved result.
3. Each tile shows its actual policy identity, shared task text, clip length,
   and a compact “pre-generated” badge. Never label a saved clip live.
4. Click a tile: inspector shows the start image, optional real goal reference,
   exact task prompt, action-source/policy identity, raw/display versions,
   generation settings, timestamps/latency scopes, limitations, and download.
5. Optional **Run comparison**: submit one matched set of independent policy
   jobs for the selected compatible task/scene/seed, or explicitly rerun one
   selected policy. Show queued/warming/generating/encoding/ready or
   failed states. The old clip remains clearly labelled as the previous version.
6. Publish the new result to its tile only after bytes, metadata and manifest are
   durable. Optional judging runs afterwards; video availability never waits on it.

Preserve normal causal ordering inside a real closed-loop rollout:
policy observation → action → world-generated observation → next policy action.
Independent jobs can run concurrently within measured capacity; a gallery of
twelve clips does not require twelve GPUs or twelve simultaneous live rollouts.

## Phase1 — establish a quality profile before a bulk render

Use Cosmos as the initial candidate because there is a verified640×480 output.
Do not assume the accepted resolution_tier values prove every profile works on
Baseten. Inspect the pinned runtime and test the actual payload on one H100.

- Begin with one known compatible start/action fixture and a small comparison
  (at most four initial candidates), preserving seeds and all failed outputs.
- Compare the existing480 tier with a supported higher tier (candidate720),
  measuring memory, render time, object identity, gripper geometry, temporal
  stability and visible alignment with the requested motion.
- Denoising settings are experiments, not a promise that more steps improve
  quality. Freeze the chosen profile before launching the gallery batch.
- Prefer a genuinely higher native temporal rate only if the model/action
  timing contract supports it. Never increase control rate or repeat actions
  simply to manufacture a frame count.
- If needed for the demo, produce a separately labelled24/30fps interpolated
  presentation copy with unchanged duration. Inspect it for invented contacts
  and motion artifacts. It is never fed back to the policy, judge, or metrics.
- Preserve the raw MP4/frames and timing. Do not upscale and claim new detail.

Exit condition: one visually acceptable raw clip plus an acceptable browser
preview, with an explicit native/display FPS distinction. If fidelity remains
poor, change the profile/backend or narrow the demo; do not publish a full grid
of weak outputs merely to reach a tile count.

## Phase2 — produce the curated gallery assets

- Freeze one comparison manifest: task instruction, real starting image/state,
  goal reference if needed, nominal time horizon, world-model revision/profile,
  seed configuration, and six exact policy identities. Validate policy-specific
  observation requirements; never invent proprioception for an incompatible
  model. Derived forecast state must be disclosed rather than called measured.
- Complete policy loaders/assets and native feedback compatibility. OpenVLA,
  Octo and SuSIE_LL have diagnostics, not blanket closed-loop qualification;
  remaining policies still need actual runtime evidence. An internal milestone
  may use two working policies, but the six-tile target is not complete until all
  six have genuine outputs or visibly reported failures/unavailability.
- Hold task/start/world profile/horizon matched across policies and preserve
  each policy's native query cadence. Same integer seed alone does not establish
  matched diffusion noise across different horizon/noise-shape profiles.
- Choose any qualitative demo scene/profile using a separate pilot. Publish a
  complete matched comparison set, not the best independent seed for each
  policy. Retain failures and selection rationale; do not imply benchmark ranking.
- Initial public deliverable: one six-policy comparison set. Add further
  task/scene sets behind a selector after quality/input coverage supports them.
  Keep all37 existing experiments in Archive.
- Target short4–6s examples where a supported horizon permits it; retain a
  shorter honest clip rather than pad/reverse/repeat frames to fake a rollout.
- High-quality master: candidate native720-tier output if verified. Dashboard
  derivative: lighter480-tier-sized encode, same duration/content, H.264/yuv420p
  with fast-start metadata. The exact choice follows Phase1 measurements.
- Export one small poster per video, original and presentation hashes, full
  prompt, action-source identity, seed, model/source revisions, dimensions,
  raw/display FPS, duration and transformations in a gallery manifest.
- Do not generate unrelated still images and animate them as if they were
  action-conditioned world-model predictions.

Generation must be queued before the demo. Estimate its budget from the pilot,
start with one GPU worker, and never infer inference entitlement/cost from the
training quota. Warn before additional large asset downloads. Do not train the
world model or restart the rejected judge fine-tuning experiments.

## Phase3 — make the grid the main dashboard

Retain the current styling, primitives, sidebar and inspector conventions.

- Main dashboard: new WorldVideoGrid using the matched comparison manifest, not a single
  playlist and not the synthetic RunService's most recent rows.
- Keep WorldVideoQueue as the recording archive and preserve Run frames for
  actual persisted episode events.
- Move CloudDiagnosticPanel, synthetic TaskPrompt and raw engineering controls
  into an explicit Developer/Evidence area. They must not lead the user flow.
- Add exact task text and conditioning type to the catalog/UI; currently the
  prompt is buried in source reports.
- Shared task/start/goal/seed controls above the grid; each policy card maps to
  its own immutable job/result. Unsupported policies show an honest state rather
  than a borrowed clip. Keep one visual timeline/horizon for comparisons.
- Separate raw and presentation-video URLs and transformation metadata.
- Native muted inline looping, Pause all and per-tile controls. Pause offscreen
  and background-tab videos; respect reduced-motion with posters/play controls.
- Limit active decoders/preloads to visible tiles; render every poster first.
- Replay loops should visibly reset. Avoid reverse/ping-pong motion that could
  be mistaken for a predicted robot recovery.

Likely files: web/src/pages/OverviewPage.tsx, LivePage.tsx,
components/WorldVideoGrid.tsx (new), WorldVideoQueue.tsx, GalleryModal.tsx,
plumb/world_videos.py and a versioned curated gallery manifest.

## Phase4 — one honest live world-video route

Port/verify a bounded world-model worker using the existing Cosmos adapter and
the proven isolated-runtime deployment pattern. Do not push the nine-Chainlet
full-study topology unchanged.

Add a durable world-generation job route/service distinct from both
/api/cloud-diagnostics (one policy action) and synthetic /api/runs.
Use the existing journal/artifact/outbox patterns; every job binds scene, prompt,
action source, seed, model/runtime and requested profile. Persist actual stage
timings and output hashes. No blind retries after ambiguous submission.

Bootstrap mode: **recorded-action world preview**, with the action source visible.
Task presets bind compatible inputs. Text editing must reach the world model's
actual prompt argument. A changed prompt with unchanged recorded actions must
not be described as a policy following a new instruction. Bootstrap previews
belong in developer tools, not in the matched policy grid as final policy results.

Target grid mode: **closed-loop policy rollout**, only for a policy/world pair with
supported native feedback boundaries. SuSIE_LL requires goal-image conditioning;
language-only control needs a genuinely language-conditioned policy. No stale
frames, invented proprioception, or padded future actions to conceal a mismatch.

The VLM judge stays downstream, optional and explicitly experimental until
calibrated. Current protocol uses16 raw sampled frames together per judgment,
not one expensive VLM call for every displayed video frame.

## Latency and acceptance criteria

Observed, not promises:
- Existing640×480 Cosmos sample:4.107s generation on the cluster, excluding
  169.793s load. This is not Baseten or full-rollout latency.
- Existing16-tick OpenVLA→IRASim diagnostic:142.214s including load/artifacts,
  with substantial distortion. The fast one-shot sample is not that workflow.
- Baseten policy-only calls: about0.56–1.28s warm request time; observed cold
  requests31.9s and54.8s. These are not world-video timings.

Product targets to verify:
- Poster grid visible within1s on the demo machine; visible loops start within
  about2s from cached assets. No model cold-start is on this critical path.
- Live generation acknowledges immediately and shows durable progress; a
  sub30s short warm preview is an initial target, not a committed SLA.
- Profile actual world-video cold/warm generation, encoding, persistence and
  browser-ready time separately before quoting a live-demo duration.
- Profile the complete six-policy set separately: time to first finished tile,
  time to all terminal tiles, failure count and peak/billed resource use. Do not
  multiply single-policy timings into a claimed concurrent-grid latency.
- Keep a worker warm only for an explicit demo window, issue a warm-up render,
  then restore idle scale-to-zero. Warm replicas incur cost.

Tests: no inference on page open/loop; prompt round-trip to actual model input;
raw/display FPS and duration correctness; missing/tampered media fail closed;
jobs survive app restart; failed jobs do not replace prior results; all selected
clips decode/play/loop; keyboard/Pause-all/reduced-motion/mobile checks; live
browser generation-to-saved-video test on the actual deployed worker.

## Out of scope for this restructuring

No invented automatic success labels; no claim that prettier motion proves
physics; no scientific policy ranking from curated examples; no full-study
scope deletion. The six-policy/five-task study and A–F evidence gates remain
separate and unchanged. This plan changes the demo's front door and adds an
explicit real-video workflow rather than mixing engineering paths together.
