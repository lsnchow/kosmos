# Automatic judge handoff and image-quality diagnosis

New primary-console submissions now admit **Generate & assess live**. The
server persists `auto_assess` on the session and calls the judge once after
the MP4 and its hash manifest are committed. The judgment is bound back to
the session using `auto-judge:<session-id>` as the idempotency key. Browser
reloads only read state; they cannot trigger this handoff or a retry.

The progress bar counts 16 completed generation steps plus one assessment
stage. It holds during judging and reaches 17/17 only when an actual judge
response is completed/abstained. A failed judge stops at 16/17 and retains the
generated video. A final summary displays generation and total wall time,
summed VLA/world calls, VLM compute/remote/cold-load time and peak VLM memory.

Actual verification: session `ddc27ebe085342038a816c70547b58a0` generated all
16 steps and automatically submitted `judge-70dddcd2b6c74b7fbdf8485e3f5ee237`.
Baseten rejected the request with HTTP 403 Authentication failed; the
management deployment read independently returned 403 Authorization error.
No inference retry occurred. One judgment exists for this clip, and browser
reloads created no additional POSTs. Failure summary and stopped progress
were verified in Chromium; build/typecheck passed. Evidence is under
`data/private/semantic-judge/automatic-proof/`.

## Why this live footage differs from the cleaner drawer recording

The cleaner saved `baseten-world-right-v3` clip used Cosmos3-Nano with manual
actions, 30 inference steps at 256 resolution. The live path instead uses the
existing OpenVLA/IRASim worker from immutable release
`55d88899e254968dbbb1a5046fb338be7d4e568093a9b842c5ebd39355834626`.
Its IRASim profile remains 256×320, 50 inference steps, guidance 1.0. No
runtime/checkpoint/sampler change was made to that deployed release here.

The changed experimental setup uses the drawer start and 16 feedback steps;
the earlier short live policy demonstration used four. Each step decodes a
predicted RGB image, uses it for the next policy action, then re-encodes it
for another one-frame world prediction. The checkpoint retains its 16-frame
architecture while this experimental call asks for condition + one future
frame. Compounding feedback/re-encoding drift is a plausible explanation for
the visual collapse. Its exact contribution versus scene/actions/conditioning
has not been isolated by a controlled comparison in this task.

The distortion exists in the saved generated PNGs themselves, including
`live-demo/frames/0972b7a6a3094259b3b179435771c4ec/0016-predicted.png`; it is not
introduced by browser playback. This turn diagnoses the quality problem; it
does not claim to have fixed IRASim fidelity or replaced generation with a
saved clip.
