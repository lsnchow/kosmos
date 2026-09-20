# Fresh Cosmos demo

The primary console starts empty on every load/refresh, including old `?run=`
URLs. It no longer restores the previous experiment from localStorage. A new
explicit **Generate** click creates a new run; previous and in-flight runs
remain in the persisted Past runs list. Refresh never cancels a server job or
starts another one. Past-run dialogs play the completed MP4.

## Real runtime

Baseten authentication still returned HTTP 403. Cosmos therefore runs on the
existing pinned Trillium runtime and weights, independently of the Baseten
judge:

- Slurm job `942950`, one H100, node `trig0022`.
- Bounded allocation ends `2026-09-20T03:20:32` Toronto.
- Source release `6903e97b15bb1f1df909de77927d12d67e3be0718c755f9cece4f1392df90049`.
- Private worker port 8919, local authenticated SSH tunnel port 8920,
  tmux `plumb-cosmos-gpu-942950`.
- `/api/demo/status` reads back `world_model=cosmos`.
- Model `nvidia/Cosmos3-Nano`, revision
  `e59a53c25979a090fa8706c9acc0c254a6e89b92`, 256 resolution, 30 denoising steps,
  guidance 1.0, 5 FPS. Existing `venv-model-std2023`, no new weight download.

This path uses supplied manual right actions compiled from the verified drawer
starting pose, with the source gripper action held. **No VLA policy runs in this
Cosmos probe.** The UI labels it Actions → Cosmos → VLM rather than claiming
OpenVLA feedback. Each Generate call uses a new request seed and invokes the
actual model. It does not play a cached result.

The worker streams actual Diffusers step-end callbacks, followed by the newly
decoded frames. The UI does not claim intermediate video frames are available
before decoding. Pixel conversion retains the established float [0,1] → uint8
boundary. There is no frame duplication/interpolation to meet the count.

## Verification

Real browser run `f1c0ac8a3df04824b458ec36512ccb81` completed with sixteen new
predicted frames plus the original scene. Model compute measured **2.961 s**;
preloading the worker before the request took **147.377 s**. Peak allocated
GPU memory was **33,637,465,600 bytes**. These timings apply to this 256-tier
configuration, not to earlier Baseten or 480-tier measurements.

Chromium observed actual diffusion progress (0, 13, 30 at its polling cadence),
completed media, and one generation POST. Reload showed an empty experiment
with Generate enabled and made no inference POST. Past-run video playback,
mobile layout, and omission of missing inference-stat placeholders passed.
The frontend build/typecheck passed.

The automatic VLM attempt is recorded separately. Baseten authentication is
still a blocker for that step; a successful Cosmos run is retained regardless.
No judge response is fabricated or retried automatically.

Evidence: `data/private/semantic-judge/cosmos-live-proof/` and the existing
`data/live-integrated/live-demo/` SQLite/media journal.

Local API startup keeps the existing Baseten judge arguments and uses
`--live-demo-url http://127.0.0.1:8920` with the existing private token-file
argument. Runtime credentials are never browser-delivered.
