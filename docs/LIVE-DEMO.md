# Live demo — working evaluation and manual steering

Verified 2026-09-19 on a real NVIDIA H100. This supersedes the earlier note that
all main-gallery controls were unavailable. Exact restoration of an old hidden
world state is still unsupported; a **new image-conditioned branch** now works.

## Normal user flow

Open <http://127.0.0.1:8787/console> and click **New evaluation**.

- **Run the existing OpenVLA controller:** enter the task instruction, choose
  1–8 steps (default 4), and start. Watch newly generated frames appear. The
  instruction is passed to the real policy on every step.
- **Steer manually:** start a session, then click a labelled direction. Each
  click requests 1–4 steps and updates the displayed image. The task note only
  labels this mode; directional commands drive its predictions.
- **New interactive branch** on a gallery tile starts a manual session from
  the recording's verified final image. It does not alter the original clip or
  restore that clip's original hidden simulator state.

The main page lists saved live evaluations. **Open evaluation** reopens their
persisted state; **Download recording** downloads the generated MP4. A new
evaluation uses an existing controller—it does not train a new policy.

## What actually runs

The browser submits to the localhost API. The API journals the job, then sends
each current image to an authenticated GPU worker over a local SSH tunnel.

1. Policy mode calls the real pinned OpenVLA model for one native 7-D action.
   Manual mode supplies a bounded directional 7-D action instead.
2. The original IRASim one-step runtime predicts a new RGB observation.
3. The API validates and persists that frame, action, model identity, timing,
   input hash and returned conditioning hash.
4. The **newly persisted image** becomes the next request's input. No action
   or canned response is substituted for the policy/world calls.
5. The browser polls persisted progress and displays the newest PNG. The MP4
   is a separate playback/download artifact, not an extra frame or the live view.

State mode is `experimental_reencoded_rgb_stateless`: each world step encodes
the supplied RGB image afresh. This is an experimental image-feedback protocol,
not native latent-state restoration or validated robot physics. Visual drift
remains possible. No judge or automatic success score is run by this demo.

## Verified results and latency

| Check                      | Actual result                                                                          |
| -------------------------- | -------------------------------------------------------------------------------------- |
| Worker startup             | OpenVLA loaded in 86.45 s; IRASim in 6.68 s; 93.12 s combined model load               |
| Browser manual commands    | Two consecutive commands displayed new frames in 3.12 s and 3.23 s                     |
| Manual world inference     | Approximately 1.68 s per step, plus encoding, HTTP, persistence and UI polling         |
| Browser OpenVLA evaluation | Four real policy/world steps completed and displayed in 10.06 s                        |
| OpenVLA per-step inference | Approximately 145 ms policy + 1.69 s world                                             |
| Mobile gallery branch      | Verified source image, one real command, new displayed frame; no overflow or JS errors |
| Video encoding             | Five-frame policy recording decodes at 320 × 256, 5 FPS, 1.0 s                         |

Five FPS is nominal playback timing, not throughput or measured physical robot
time. Frames are not interpolated or repeated to claim smoother model output.
The GUI updates at each completed prediction, not as a continuous 60 FPS game.

Recorded sessions under `data/live-integrated/live-demo/`:

- `b310ffb9b6984117b181dcb11df30a73`: browser manual steering, 3 generated
  frames; consecutive input/output hashes match.
- `269c23a9c6d64fd8b70a44028af0a3ca`: browser OpenVLA evaluation, 4 generated
  frames; UI observed progress 0, 1, 2, 3, 4; all feedback hashes match.
- `722b5b6e98f74897b495f0c4b95747ef`: browser/mobile manual branch from the
  saved Cosmos recording's final image; source SHA and feedback hashes match.
- `09663358d4204110b7b4d500dc7d6dd5` and
  `a4de8a93c3de4e8fa34e4788397e483b`: actual one-step service verification.
- `4fbba6c9330d424794df481e9e3ccda8`: preserved failed extraction attempt.
  Seeking 1 ms before EOF skipped the last low-FPS frame; extraction now decodes
  through the bounded clip and retains the last actual frame, with a regression test.

These are experimental outputs, not qualified policy comparisons. No gates,
calibration records, published reference percentages or score labels changed.

## Runtime and operating window

Live generation currently uses **Trillium**, not the separate Baseten policy
diagnostic. The existing Baseten deployment remains available through Developer
tools; it was neither replaced nor relabelled as a world-model endpoint.

- Slurm job: `941263`, one H100 / 24 CPUs on `trig0001`, account `def-ikarlin`.
- Allocation: 2026-09-19 19:40:51–21:40:51, Toronto time. It ends automatically.
- Source release: `55d88899e254968dbbb1a5046fb338be7d4e568093a9b842c5ebd39355834626`.
- Runtime: existing `/scratch/lchow432/plumb/venv-irasim`; no new model download.
- Modules: `StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17 opencv/4.11.0`.
- Worker: compute-node private interface, port `8917`, Bearer authentication.
- LOCAL tmux `plumb-live-gpu`: SSH forward from localhost `8917` through
  `trillium-gpu` to `trig0001:8917`. Do not terminate the shared SSH master.
- LOCAL tmux `plumb-live`: localhost API on `8787` with live-demo flags enabled.

The private token is in `data/private/live-demo-t1xfYe/worker.token` locally
and `/scratch/lchow432/plumb/private/live-demo-20260919-1936/worker.token`
remotely. Contents are never logged, served, committed or sent to the browser.
The worker is not internet-facing; the API accepts a loopback tunnel URL.

After the allocation expires, saved PNGs/MP4s and session records remain usable,
but new inference needs a fresh allocation/tunnel. This is a bounded local demo
deployment, not yet an always-on hosted product.

## Restart / renew safely

First inspect `squeue`, the local listeners, and the tmux panes. Do not duplicate
an active GPU job or interrupt an in-flight run. Root owns cluster operations.

If the existing allocation still runs, reconnect the tunnel to its actual node.
If it has ended, submit a new bounded job using the immutable release:

```bash
ssh trillium-gpu 'sbatch --parsable --account=def-ikarlin --time=02:00:00 /scratch/lchow432/plumb/releases/55d88899e254968dbbb1a5046fb338be7d4e568093a9b842c5ebd39355834626/cluster/live_demo_server.sbatch /scratch/lchow432/plumb/releases/55d88899e254968dbbb1a5046fb338be7d4e568093a9b842c5ebd39355834626 /scratch/lchow432/plumb/private/live-demo-20260919-1936/worker.token 8917'
```

Use that new job's actual allocated node in the SSH forward. Wait for the
authenticated `/health` to report ready; model loading is not instantaneous.
Do not cancel other jobs or change quotas to make this one start.

Restart the API with the existing diagnostic pins and new loopback worker flags:

```bash
.venv/bin/python scripts/serve_baseten_mvp.py \
  --model-id 3mzlenow --deployment-id q929yoj --profile plumb-api \
  --timeout-seconds 600 \
  --live-demo-url http://127.0.0.1:8917 \
  --live-demo-token-file data/private/live-demo-t1xfYe/worker.token
```

Model/runtime upgrades require a new tested immutable source release, not edits
inside the running release. The live session API/data do not require regenerating
the old gallery clips.

## Code and safeguards

| File                              | Responsibility                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `web/src/components/LiveDemo.tsx` | New-evaluation form, saved-session list, live image/controls and polling                                     |
| `plumb/live_demo.py`              | SQLite sessions/jobs/commands, bounded background queue, HTTP worker client, integrity checks and MP4 export |
| `cluster/live_demo_server.py`     | Pinned model loading, authenticated health/fixture/step endpoints and single-GPU inference lock              |
| `cluster/live_demo_server.sbatch` | Allocation-owned module/runtime startup                                                                      |
| `scripts/serve_baseten_mvp.py`    | Existing API startup plus optional live-worker URL/token-file arguments                                      |

Commands have idempotency IDs; reusing an ID with different content is rejected.
One local worker queue serializes GPU requests. Steps are bounded to 1–8 per
evaluation/session, and 1–4 per manual command. A manual session is initialized
without inference; movement requires an explicit click. A failed/ambiguous
remote request is not blindly retried. Restart recovery preserves evidence and
does not pretend interrupted work succeeded. Original source footage is never
overwritten, and unsupported exact-checkpoint restore remains a separate path.

Verification: 1,379 Python tests passed / 6 skipped; 272 frontend tests passed;
production build and real desktop/mobile GPU E2E checks passed. Existing Pillow
deprecation and Vite >500 kB bundle warnings remain. The next product work is
better world fidelity, longer/validated feedback, the other policy runtimes,
calibrated judging, and durable managed deployment—not additional disabled UI.
