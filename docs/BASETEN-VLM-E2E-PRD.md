# Baseten VLM end-to-end demo PRD

## Outcome

Ship one real demo path:

```text
Saved Close the drawer video → Baseten Qwen + semantic LoRA → persisted browser assessment
```

This is an experimental visual-judge demo, not a calibrated score, policy
ranking, or qualification result.

## Baseten deployment

- Deploy one dedicated **private Baseten H100** judge worker, independently
  scaled from any policy/world deployment.
- Configure `min_replica=0`, `max_replica=1`, and request concurrency `1`.
  Record the actual model and deployment IDs plus autoscaling read-back. Do not
  assume capacity is free or infer billing from quota.
- Keep Baseten credentials in the named CLI/server profile. The browser never
  receives an API key, deploy token, model URL, or model path.
- Keep incremental VLM work within the shared $100 cap. No automatic retry may
  turn an ambiguous remote outcome into additional spend.

## Exact model contract

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Base/processor revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Required adapter: semantic pilot `epoch-02`
- Required adapter tree hash:
  `sha256:7a0b7796a0ac8b78fbb8ba8f4c4db01fe8d35c2edc3cb7a0fbeec9f4fb20523e`
- Do not substitute the base model, the formatting-only adapter, a cached
  response, or another model if loading fails.
- Use a dedicated Baseten runtime compatible with the semantic adapter reload:
  Transformers 4.49 and PEFT 0.14.
- Package the adapter only inside Baseten's private deployment boundary,
  verify its tree hash at worker load, and never expose or copy the weights to
  the browser or laptop.

## Minimal architecture

1. The localhost app resolves one supported saved `Close the drawer` clip by
   ID; the browser never submits a path or arbitrary URL.
2. The app verifies the source MP4 SHA-256, decodes the complete clip, and
   chooses exactly 16 ordered, distinct source-frame indexes, including both
   endpoints. It does not duplicate or interpolate frames.
3. The app persists the frame manifest and scene-reference image before remote
   dispatch. The reference is recorded as scene context, never as a goal.
4. The private Baseten worker loads Qwen plus the semantic adapter through PEFT,
   verifies the active enabled adapter, then runs the existing `QwenRubricJudge`
   contract: temperature 0.7, five samples, three-vote quorum, and only one
   bounded retry for schema/recognized transport failures.
5. The worker returns raw outputs, parsed samples, adapter receipt, per-sample
   timings, total VLM timing, and peak GPU memory.
6. The localhost app validates and persists the response in SQLite. Ambiguous,
   malformed, refused, or no-quorum results become `Unable to assess`; they are
   never coerced into success/failure and are never automatically retried.
7. The browser polls only persisted state and restores it after reload without
   submitting another inference request.

## Shipped screen and controls

The primary screen is only:

- task and saved video;
- one action: **Assess with our trained judge**;
- current observed status;
- persisted assessment beside the video;
- one disclosure: **Experimental judge — not calibrated against human ratings.**

Every remaining visible control on this screen must work against real state:

- video playback;
- source/details disclosure;
- assessment submit and status updates;
- result/receipt disclosure;
- browser reload restores the same request and result.

Remove primary-screen dead ends instead of leaving disabled legacy controls.

## Selective post-inference breakdown

Persist and show one expandable **Inference breakdown** after a request. It
must contain the values that change per inference or prove that the requested
model was actually used:

### Input and identity

- durable request ID and terminal state;
- source video SHA-256;
- selected frame indexes, timestamps, and decoded-frame hashes;
- scene-reference hash and role;
- task ID and rubric hash;
- base model ID/revision;
- adapter ID, adapter tree hash, active-adapter identity, and enabled status.

### Inference execution

- Baseten model/deployment ID, configured replica bound, and observed peak VRAM;
- cold model-load time when it occurred;
- per-sample VLM inference time and total VLM inference time;
- five seeds, raw output, parsed result, retry count, and final status per
  sample;
- quorum/result computation, including the reason for `Unable to assess`.

### End-to-end timing

- source decode/frame-preparation time;
- remote request/queue time outside the model call when observed;
- local validation and durable-persistence time;
- total user-action-to-persisted-result wall time.

Do **not** clutter this breakdown with static environment metadata that is not
needed to understand a particular inference, such as CUDA version, generic OS
details, package inventories, or unchanged cluster setup notes.

## Explicit exclusions

- No full OpenVLA, Cosmos, or IRASim migration in this task.
- No new model training, LoRA experiments, calibration, distillation, policy
  ranking, or qualification-gate changes.
- No Cloudflare, object-store, or public endpoint.
- No broad refactors, linting, full backend/frontend test suites, or research
  expansion.

## Lean acceptance check

1. Frontend build/typecheck succeeds.
2. One real browser click reaches Baseten and returns a response from the verified
   semantic adapter.
3. Displayed fields match the persisted raw report and derived quorum result.
4. Reload shows the exact persisted assessment with no second inference call.
5. Save the request ID, adapter receipt, inference breakdown, and screenshot.

If private packaging, adapter-hash verification, Baseten runtime compatibility,
autoscaling read-back, or remote inference fails, stop and report the exact
blocker. Never replace the requested semantic adapter with another model or a
fabricated response.
