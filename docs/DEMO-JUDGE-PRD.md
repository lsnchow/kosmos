# KOSMOS — demo-first judge integration PRD

Date: 2026-09-20. Requested by Lucas in a side conversation.

This document proposes the new demo priority. Creating it does not cancel any
running job, change the application, or communicate instructions to the main
thread. Preserve all existing code, checkpoints, failed results, and receipts.

## Latest directive — fast endgame implementation

Lucas explicitly requests FAST, targeted fixes. This section supersedes any
earlier suggestion to expand testing, infrastructure, or the comparison wall.

- Make the smallest direct changes that put the trained judge and usable video
  flow on screen. Reuse existing endpoints, loaders and components; a thin
  adapter is preferable to another service framework or generalized abstraction.
- Do not write new automated test suites, run repeated full-suite passes, launch
  more planning/audit agents, retrain models, or broaden the task. Do not wait
  for the 12-cell wall, all policy loaders, or scientific qualification.
- Keep verification lean: build/typecheck the changed frontend and perform one
  real browser click-through proving clip → loaded trained adapter → visible
  result, followed by reload. Fix concrete failures found there. Do not call a
  mocked or hardcoded result working inference.
- Retain essential credential, artifact-identity, budget and no-silent-retry
  protections. Speed does not authorize fabricated outputs or hidden costs.
- If the previous agent is stopped, first inspect its existing job/deployment
  state read-only. Do not duplicate requests or leave superseded GPUs running.

### Immediate console cleanup — exact removal list

Remove these blocks/copy from the primary console, including equivalent current
wording. Do not replace them with another large explanation of missing features:

> Start a bounded OpenVLA or manual experimental evaluation, then inspect its persisted frames alongside the saved recording archive.
>
> Browse all recordings →
>
> Matched control wall
>
> comparison release blocked
>
> No explicitly promoted completed matched comparison is saved locally.
>
> No promoted 12-cell set is available. The legacy live evaluation and recording archive remain below; they are not substituted into this controlled wall.
>
> Fresh comparison
>
> A new run is deliberate. First request a server-side cost/readiness quote; only an approved quote can admit work.
>
> No comparison-source manifest is present at comparison-source-manifest.json
>
> Prepare fresh matched-set quote
>
> Legacy experimental controls
>
> This earlier bounded demo remains available for engineering checks. It is not a cell in the controlled wall and is not a policy comparison.
>
> Run the existing OpenVLA controller or steer a bounded manual session. These are live experimental outputs, not a policy comparison or a scored result.
>
> live demo worker connection outcome is ambiguous

Keep saved clips accessible directly on the page. Move technical diagnostics
and legacy tooling out of the primary journey; preserve their code and records.
Removing a raw error banner does not mean claiming its backend is healthy. If
the user explicitly requests an unavailable operation, show one short, contextual
message and a working alternative, such as **Live generation is offline. Assess
a saved clip instead.** Keep technical details collapsed and do not replay an
ambiguous request automatically.

### Fix the confusing OpenVLA-only action

The existing **Run OpenVLA evaluation — unavailable** action must not remain the
primary dead-end. Use **Generate & assess** only when the real generation and
judge path is connected, and **Assess with our trained judge** for saved clips.
If live generation remains unavailable, prioritize the functioning saved-clip
assessment rather than merely renaming or enabling a broken button.

OpenVLA is the existing robot controller: image/task → arm actions. It appears
alone because that was the previously implemented policy demo, not because the
product or the judge can only evaluate OpenVLA. The post-trained Qwen VLM is a
different component: video/task → assessment. Show controller identity in a
small details field, and show the actual Qwen/LoRA identity on the result panel.
Do not invent a multi-policy selector or spend endgame implementing extra
policies. The judge can assess supported, properly sourced clips independently
of whether this app generated them through OpenVLA.

The visible screen should be: **task + video + one useful action + trained-judge
assessment**. Keep one concise experimental-judge disclosure; remove the wall
of qualification/legacy prose. Patch the current components directly.

## 1. Product decision

Kosmos is a virtual test bench for robot controllers: inspect a robot rollout,
run our post-trained visual judge on it, and see its assessment beside the video.

The hackathon demo must visibly connect **video → our trained judge → result**.
Do not spend the remaining integration time completing a twelve-tile matrix
before this basic product loop works.

The matched control wall means three policies × four seeds on one starting
scene. It is useful for eventual controlled comparisons, but is **not required
for this submission's demonstration**. Remove it from the primary demo journey;
keep its code and APIs behind developer tools. Do not delete them or populate
them with videos from unrelated policies/scenes.

## 2. Scope and priorities

### P0 — ship first: score an existing real clip with our actual trained adapter

1. Load the exact Qwen base and selected saved LoRA; verify their identities.
2. Run inference on one existing real, task-bound robot clip.
3. Persist the request, selected input frames, raw model response, parsed fields,
   timing, model/adapter identity, and explicit experimental status.
4. Expose the operation through localhost and display its real output beside
   the clip. This path must work without launching a world-model job.

### P0 — connect the visible end-to-end loop

1. Use one supported task and one compatible starting scene.
2. Show generation stages and the resulting video.
3. Queue the same experimental judge once the full video is committed.
4. Show the judge's assessment without changing routes or opening logs.

Generation failure must not prevent scoring another already-saved real clip.
Judge failure must not hide a successfully generated video.

### P1 — rehearse the live demonstration

Prewarm the actual selected deployment within the existing budget. Measure
generation and judging separately. Prepare an explicitly labelled completed
example. Run desktop and projector-sized browser rehearsals.

### Defer from the stage-critical path

- Completing the 12-cell/70-action policy comparison.
- MiniVLA/Octo rollout collection required only for that comparison.
- The 1,500-rollout burst, 100-replica expansion, and queue-depth spectacle.
- New training, new calibration collection, cost/drift sweeps, or a larger study.
- Reliability/error-bar claims that lack measured evidence.
- A chatbot, arbitrary new tasks, policy creation/training, and broad UI redesign.
- Keyboard steering if it competes with getting the trained judge onto the screen.

Do not interrupt an active request or abandon billable infrastructure blindly.
Finish or safely stop scoped work, retain evidence, then redirect effort.

## 3. What currently exists

The latest local handoff reports real Baseten Cosmos video delivery, but the
first delivered images failed visual inspection. A NumPy/PIL intensity-boundary
fix was deployed and was being tested. Continue the minimum visual/action
checks needed to get a usable clip; deployment success alone is not visual QA.

Recorded timings before that fix were approximately 136 seconds cold and
36 seconds warm for the world probe. These are not validated timings for the
corrected profile or for the judge. A two-second live loop is not established.

The saved post-trained judge is **Qwen2.5-VL-7B plus the semantic pilot LoRA**:

- Base revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Adapter: `/scratch/lchow432/plumb/experiments/judge-lora-pilot-v1-trained/adapters/epoch-02`.
- Recorded adapter tree SHA-256: `7a0b7796a0ac8b78fbb8ba8f4c4db01fe8d35c2edc3cb7a0fbeec9f4fb20523e`.
- Training job: 939451. Saved-adapter reload: 939458.
- Training happened on Trillium, **not Baseten Training Jobs**.

Verify current checkpoint availability and the recorded tree-hash algorithm
before packaging. Use the private deployment boundary; do not publish weights
or credentials to the browser or a public repository.

This pilot learned homogeneous, weak teacher targets. Its lower loss did not
establish better judgment. That limits the claim, not the ability to expose an
explicitly experimental inference feature.

Do **not** silently substitute the later formatting-only adapter, the base
model, a frontier API, or stored example responses. If the selected adapter
cannot load, report that exact failure. Optional base-model comparison must be
separately labelled and retain its own output.

References: `docs/JUDGE-PILOT-V1.md`, `docs/JUDGE-FORMAT-PILOT.md`, and
`docs/CLOUDFREE-COMPARISON-IMPLEMENTATION.md`.

## 4. Primary screen and user flow

One page, three visible areas:

1. **Experiment:** supported task, starting scene, actual action source, and
   either Generate & assess or Assess saved clip.
2. **Video:** source before generation, actual committed video afterward, with
   saved/live provenance and playback controls.
3. **Our post-trained judge:** task progress, visual integrity, collision
   assessment, completion assessment, and a concise explanation if the model
   actually supplies one. Link cited evidence frames when available.

Visible label: **Experimental judge — not calibrated against human ratings.**
Keep this concise. Detailed limitations and exact revisions belong in an
expandable evidence panel, not a wall of blocking qualification text.

Default the page to a real completed example, explicitly marked **Saved example**.
Do not dispatch inference merely by opening or refreshing the page.

Use a constrained task selector, not a chatbot. The initial task should be
`Close the drawer` when its clip actually depicts that task. Never relabel the
pot/purple-object fixture as a drawer task. Other saved clips retain their own
actual task and may be assessed only under a supported rubric.

If the action source is a keyboard command or recorded actions, say so. Do not
call a world-only probe an OpenVLA policy rollout. An evaluation selects an
existing controller; it does not create or train a policy.

## 5. Judge runtime and evidence contract

### Loading and execution

- Extend the existing Qwen/PEFT loading and saved-adapter replay path; do not
  invent a new model stack. Inspect `plumb/policies/judge.py` and the recorded
  pilot/reload scripts before implementing the adapter-enabled runtime.
- Readiness verifies the base revision, actual adapter files/hash, loaded active
  adapter, processor/runtime identity, and ability to perform real inference.
- A label saying "fine-tuned" is insufficient. Capture a server-side receipt
  that establishes the selected adapter was loaded and enabled.
- Prefer a dedicated Baseten judge Chainlet, independently scaled from the
  world worker. Preserve the existing Cloudflare-free localhost result journal.
- Run orchestration in the CPU step so the world GPU is not held waiting for
  a network call to the judge. No new object-store or public callback dependency.
- Keep secrets server-side, limit replicas and concurrency, and account for
  judge cold-load/residency in the shared $100 incremental budget.

### Inputs

- Exactly 16 real, ordered sampled frames from the complete committed clip,
  with a deterministic sampling manifest and source video SHA-256.
- No repeated frames to satisfy the count, no interpolation, and no unrelated
  reference/goal image. Clips with insufficient distinct frame indexes show
  an actionable unsupported-input message.
- The exact task instruction and versioned existing rubric.
- Preserve the trained prompt/reference protocol for the first implementation.
  If a genuine goal image is unavailable, do not relabel the starting image as
  a goal. Record the actual scene-reference role.
- No policy identity, expected success, desired verdict, or published answer
  in the judge's visual assessment prompt.

### Outputs and claim boundary

- Persist raw responses and structured integrity/collision/progress/completion
  fields. Derive the displayed completion category in Python, not by asking
  the model to emit the final application label.
- Default to the existing nonzero-temperature, five-sample/three-vote protocol
  where applicable. Preserve abstention, malformed-output and no-quorum states.
- If displaying an early sample for responsiveness, label it **Sample 1 of 5 —
  provisional**. Do not present it as consensus or quietly lower the quorum.
- Display **Unable to assess** for missing/invalid/abstained results, not zero
  success, and not a fabricated positive result.
- No changes to formal qualification gates, primary-study scoring eligibility,
  or held-out data membership. Experimental judge output stays in its own store.
- No claims of improvement, calibrated accuracy, human equivalence, or policy
  ranking unless separate evidence actually supports those claims.

## 6. Local API and persistence

Suggested additive API, reusing existing journal/background-worker patterns:

- `GET /api/demo/judge/readiness`: adapter/base identity, availability, reasons.
- `POST /api/demo/judgments`: existing server-resolved clip ID, explicit
  supported profile, and idempotency key; return durable judgment ID.
- `GET /api/demo/judgments/{id}`: status, committed result, provenance and timing.
- `GET /api/demo/judgments/{id}/events`: durable stage/sample/result updates.

The browser submits an existing artifact ID, not an arbitrary filesystem path
or URL to fetch. The server resolves and verifies the video. Bind each judgment
to the exact video hash, task, sample indexes, rubric hash, base/adapter revision,
sampling configuration, and inference attempt ID.

States: queued → warming → sampling/assessing → completed | abstained | failed |
interrupted. Use elapsed time and observed stages, not fabricated percentages.
Send heartbeat updates during long loads. Reconnection replays committed state
without resubmitting inference. Same idempotency key/different payload rejects.
Ambiguous remote completion is not automatically retried.

The Generate & assess action is one explicit user admission covering generation
and judging. Commit the video first, then submit its judgment once. Selecting or
playing a saved clip never implicitly launches another paid request.

## 7. Demo timing and fallback

Total presentation: one minute of explanation, two minutes in the application.

- 0–20 seconds of demo: select the real saved clip and click Assess with our judge.
- 20–50 seconds: play the video while actual judging stages update.
- 50–90 seconds: show returned rubric fields and evidence. If not finished,
  use an explicitly labelled saved assessment from an actual previous request.
- 90–120 seconds: optionally show a prewarmed fresh generation and explain how
  its committed video enters the same judge path. End on video plus assessment.

Judge integration is the primary live interaction. Full fresh generation is
optional if its measured latency threatens the two-minute slot.

Measure at least one cold and three warm end-to-end judgments on the exact
deployment. A warm assessment within roughly 30 seconds is a demo target, not
a claim until measured. If slower, start it before the pitch or narrate a saved
real result while its live status remains visible. Never disguise playback as
new inference. Do not begin a new optimization/training project to save the pitch.

## 8. Minimal E2E verification and definition of done

No new automated-test project or broad regression campaign in this endgame pass.
Use existing protections, build/typecheck changed frontend code, and do this
short real-browser smoke:

1. Load localhost: the requested wall/legacy text is gone and a real clip plays.
2. Click Assess with our trained judge. Confirm one real inference attempt with
   the specified LoRA loaded and enabled—not the base model alone.
3. See actual status followed by the persisted assessment beside the clip;
   displayed values must match the real response, including abstention/error.
4. Reload: the same clip and result remain without another paid request.
5. If Generate & assess is exposed as working, click it once and confirm the
   actual generated video feeds the same judge path. Otherwise keep this action
   unavailable with the short saved-clip alternative, not an OpenVLA dead-end.

Record the request ID, adapter identity, elapsed time and a screenshot. That is
the immediate verification scope; expand coverage after the demo, not before
the first working product loop.

Do not mark this PRD complete because an adapter file exists, a CLI test passes,
a Chain deploys, or a result card is hardcoded. Completion means **a real clip
was assessed by our loaded trained adapter and the result appeared in the app**.

## 9. Main-thread handoff

Finish the currently active, bounded world-image correction/inspection safely.
Then make the saved-clip trained-judge path the next implementation task, ahead
of completing the matched wall. After that, connect the fresh video path and
rehearse. Report named blockers with evidence; do not replace this visible
product work with another research milestone or another adapter-training run.

Likely integration surfaces: `plumb/policies/judge.py`, an additive experimental
judge service/runtime, `plumb/api.py`, `deploy/baseten/`, and the console's existing
video/session components. Preserve legacy API behavior and use existing accessible
UI primitives. Read relevant UI skills before changing components.
