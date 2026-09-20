# Longer Cosmos rollout and live pot assessment

The default pot demo now generates **32 future frames plus the starting image**,
giving **6.6 seconds at normal 5 FPS playback**. The 3.4-second option remains.
This is two real 16-action Cosmos calls: the second is conditioned on the exact
last PNG of the first. The original fixture has only 16 actions, so the second
chunk explicitly replays that action plan. The UI discloses this. It is not a
repeated video, an interpolation, or a new recorded action continuation.

The seed controls stochastic sampling with otherwise unchanged inputs. The
second chunk uses the selected seed + 1. The selected task and seed are kept
in the session journal and refresh still starts a fresh empty experiment.

## Real completed check

- Session `6112d45a886240a18dd9560ef15a1f03`
- Judge `judge-2d78603847724dc893f93ee40e9b5e4c`
- 33 persisted frames; second-chunk input hash equals frame 16's PNG hash.
- Browser measured MP4 duration 6.6 seconds at normal playback speed.
- Cosmos compute: 5.635 seconds across two calls.
- Judge compute: 41.768 seconds; remote stream: 44.326 seconds.
- Browser observed actual completed-sample counts 0, 1, 2, 3, 4, 5.
- Trained adapter and all 56 enabled LoRA layers verified. Judge abstained.
- One browser generation POST; judging was dispatched by the server.
- Evidence: `data/private/semantic-judge/long-pot-judge-proof/`.

The pot judge uses an explicitly diagnostic task/rubric in
`plumb/policies/demo_tasks.py`. The benchmark registry is unchanged. It uses
the same verified semantic epoch-02 adapter, pinned Qwen base, temperature
0.7, five samples, three-vote quorum, and bounded schema retry contract.

## Layout

Video is capped at 22 rem high and sits beside the judge. Setup collapses once
a run starts. The overall tracker counts observed diffusion steps and observed
judge samples, not an estimated elapsed-time percentage. The judgement panel
shows sample progress while the model is running.

## Runtime

Baseten still rejected authentication, so the exact semantic adapter is now
served through an authenticated private Trillium tunnel. No model weights
were downloaded or substituted. Local base files were checked using the
recorded training manifest; the adapter tree hash is unchanged.

The scheduler refused extending the prior allocation. After both queues were
idle, it was replaced through normal `sbatch` admission by **943050**, one H100
on `trig0015`, bounded to one hour, ending **2026-09-20 04:40:55 Toronto**.
`cluster/cosmos_judge_demo.sbatch` runs Cosmos and Qwen in their separate pinned
environments on that allocation. Source release:
`1c803cabb6889570bf4de1706a15eb06861931bf0b81a2a1bbc91738ebc934a9`.

Local tunnel `plumb-full-tunnel-943050`:
- localhost 8924 → trig0015:8919 (Cosmos)
- localhost 8925 → trig0015:8921 (trained judge)

The API uses the existing private token file, with
`PLUMB_DEMO_JUDGE_WORKER_URL=http://127.0.0.1:8925`,
`PLUMB_DEMO_JUDGE_WORKER_TOKEN_FILE=<existing private token-file path>`,
`PLUMB_DEMO_JUDGE_ALLOCATION_ID=943050`, and
`--live-demo-url http://127.0.0.1:8924`. Model URLs and credentials are not
sent to the browser. Earlier results remain in their original stores.
