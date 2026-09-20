# Live policy → world → judge demonstration

Verified 2026-09-20. The primary console now exposes one explicit live path:
OpenVLA (VLA policy) → IRASim (world model) → Qwen semantic adapter (VLM judge).
The six saved gallery videos and previous sessions remain below it. Judge
history is selectable; hashes, seeds, raw outputs and frame metadata are in
collapsed Debug / Reproducibility details. An abstention is shown once.

## Actual fresh generation

- Trillium job `942822`, H100 on `trig0025`, bounded 30 minutes;
  scheduler end time `2026-09-20T02:32:58` Toronto.
- Existing immutable runtime release
  `55d88899e254968dbbb1a5046fb338be7d4e568093a9b842c5ebd39355834626`.
- Private tunnel in tmux `plumb-live-gpu-942822`: localhost 8918 → trig0025:8917.
- Local API remains on 8787, with existing judge flags and the restored live
  worker flags (`--live-demo-url http://127.0.0.1:8918` and the existing private
  token file). No token contents were printed or sent to the browser.
- Fresh session: `0972b7a6a3094259b3b179435771c4ec`.
- Browser observed every completed step **0, 1, 2, …, 16** through SSE snapshots
  of persisted GPU outputs. The progress bar advances only on completed steps.
- Sixteen real OpenVLA calls, each followed by an IRASim prediction, took
  about **36.07 s** from browser admission through visible completion.
- The completed MP4 has 17 distinct frame indexes, 5 FPS, hash
  `sha256:1b06027787413f8574655080724fa6ff33276b65bb10bdf884ee1d4f6c1fe1c6`.
- The source is the verified original drawer fixture. The generated RGB frame
  feeds the next policy/world call; this is an experimental RGB feedback path.

The UI reports a policy/world *cycle* in progress, and the last completed
policy/world timings. It does not claim visibility into individual denoising
iterations or the precise policy-to-world transition inside an in-flight RPC.
The VLM is explicitly invoked after the video has committed. Reload and SSE
reconnection only read persisted state.

## Same fresh video reached the VLM

Judgment `judge-c61aaaa1d19e4d2cafaa2e4fd6462a3f` used the completed live session
via server-resolved clip ID `live-demo:0972b7a6a3094259b3b179435771c4ec`.
Baseten Team 33 model `qjjoyl2q`, deployment `w556dmj`, verified semantic
epoch-02 and 56 active LoRA layers. It abstained on visual artifacts.

Cold/first-call measurements:

- VLM inference: **16.909 s**
- Remote request including wake: **92.542 s**
- Model load: **37.108 s**
- Overhead outside inference: **75.633 s**, including load/transport/queue;
  not independently measured queue time
- Preprocessing: **0.249 s**
- Local validation and durable commit: **0.00645 s**
- Admission to durable assessment: **92.830 s**
- Peak allocated GPU memory: **18,391,195,648 bytes**

Earlier warm saved-clip inference remains available in history at 13.625 s
VLM / 14.645 s remote request. The performance view labels observational
history separately from a controlled concurrency benchmark.

## Additional metrics and external blocker

The instrumented worker source now contains token callback timings, input and
output counts, first-forward CUDA timing, subsequent decode CUDA timing,
decode tokens/sec, GPU identity, batch size and a post-call device HBM snapshot.
It has **not been deployed or GPU-verified**. The deployment API rejected
`semantic-epoch02-instrumented-v2` with HTTP 400:
`You must add a payment method to deploy models.` No new deployment was created.
The existing v1 deployment still woke and served the real fresh-video request.

Consequently TTFT/prefill/decode/token throughput/HBM are not asserted for
existing results. GPU utilization, isolated queue latency and concurrent-load
throughput remain unmeasured. The UI uses real sample latency median/p95 and
the observed management replica count; it does not invent missing metrics.

## Evidence

`data/private/semantic-judge/live-flow-proof/` contains the complete generated
session, observed step sequence, intermediate browser screenshots, judge result,
reload verification, final desktop and mobile screenshots. Browser verification
observed one generation POST, one explicit assessment POST, and no additional
POSTs on reload. No page errors or mobile horizontal overflow. Build/typecheck
passed. Private data and weights remain ignored by git.

Open the completed run with:
`http://127.0.0.1:8787/console?run=0972b7a6a3094259b3b179435771c4ec`.
It is labelled as a generated saved result. Press **Generate live rollout**
to admit a new run while the bounded worker allocation is active.
