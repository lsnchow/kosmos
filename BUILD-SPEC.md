# NIGHTSHIFT — build spec

Everything else is reference. This is what you build.

---

## Read the track name literally

The prize is **"Best Use of Baseten."** Not best research. Platform use is the primary axis; rigor is how you win among the projects that all use it well. Two consequences that change the build:

- **Your pipeline must be a Chain.** Policy, world model, validity gate and judge are four steps with four different hardware profiles that each want to scale independently. That is precisely what Chains exists for, and hand-rolling the orchestration throws away their flagship primitive. It also implements your own best argument — an H100 at eleven cents a minute should never block on a network call to a scoring model — in their vocabulary instead of yours. **This was missing from my earlier spec and it is the single biggest gap in platform coverage.**
- **Frame it as a product with published reliability specs, not a research instrument.** See below.

---

## The framing that lands hardest

**The evaluation-as-trustworthy-product thesis has an exact analogue in robotics, and nobody has built it.**

Parsed took production data, built evaluators, validated them against expert judgment, and turned the resulting signal into specialized models that beat frontier models on specific jobs. You are doing the structurally identical thing one domain over: take a policy, roll it out, score it with a validated judge, and produce a signal a robotics team can actually act on.

Every piece of your design is a move from their own published work:

| Your component | Their published position |
|---|---|
| Deterministic validity gate instead of a VLM call | They replaced a model call with Python and reported 100% accuracy |
| Judge distillation into a small model | "An ensemble of weaker, faster, cheaper models" beats one frontier call |
| Decomposed fields, label computed in code | "Stop with your monolithic eval prompts" — rubric fan-out |
| The four reliability numbers | "Searching for a measurement tool that doesn't wobble when you hold it" |
| Timestamped pre-registration | "Goalposts that we cannot move until we cross them" |

**Do not say "we built your company for robots."** Say the domain claim: evaluation is the bottleneck in robotics for the same reason it was in language, the same fix applies, and nobody has shipped it. Then let them notice the parallel themselves. They will.

**So the product one-liner is:** *point us at a policy endpoint, get a ranked report in twenty minutes for a few dollars — and here are the four numbers that say how much you should trust it.*

The four reliability numbers stop being research findings and become **the spec sheet of a product**. That is a better story for this track and it costs you nothing to build.

---

## The version you build

**Three groups have built world-model policy evaluators. All three reported accuracy. None reported precision. You do.**

That is the whole pitch. You are not claiming to have invented evaluating robot policies in a dream — that is published, three times, one of them on your exact robot. You are claiming that nobody put error bars on it, and that an evaluation instrument without error bars is not an instrument.

Both judges founded an evaluation company. Their own essay is about "searching for a measurement tool that doesn't wobble when you hold it." You are handing them the precision.

---

## What ships

Four things, in priority order. If you only get the first two, you still have a submission.

1. **A rollout engine** — policy in, dreamed video out, score out. Runs on Baseten.
2. **Four numbers about the engine itself**, with intervals.
3. **A live dashboard** showing rollouts landing and queue depth.
4. **A published environment** so someone else can drop in a fourth policy.

---

## The stack, exact

| Layer | Choice | Size | Note |
|---|---|---|---|
| World model | `nvidia/Cosmos3-Nano`, forward dynamics, domain `bridge_orig_lerobot` | 33 GB | Ungated. Take `Cosmos3-Edge` (8.5 GB) if latency bites |
| Policy 1 | `openvla/openvla-7b` | 15 GB | Trivial to stand up, pre-made server exists |
| Policy 2 | `Stanford-ILIAD/minivla-vq-bridge-prismatic` + `pretrain_vq` | 5.6 GB | Two-part download |
| Policy 3 | `rail-berkeley/octo-small` v1.0 | 0.55 GB | Path is `270000/default/checkpoint`, not root |
| Judge | VLM API call + `facebook/vjepa2-vitl-fpc64-256` for the distilled version | 1.3 GB | |
| Ground truth | AutoEval Table 2 — 6 policies, 5 tasks, 250 human-scored trials each | — | Printed table, no download |
| Training | Baseten **Training Jobs** (Axolotl/TRL), not the RL SDK | — | RL SDK is form-gated and text-only |
| Orchestration | **Baseten Chains** — four steps, four hardware profiles, independent autoscaling | — | Do not hand-roll this |

Everything ungated. Nothing waits on a human. **Mirror `pretrain_vq` to your own account in the first ten minutes** — it has no institutional owner and could vanish.

Your three policies give you **15 paired cells** against human ground truth, with a clean three-tier spread: MiniVLA 50.8%, OpenVLA 39.6%, Octo 0.8%.

---

## The four numbers

All four come out of **one rollout pool plus two sweeps.** That is why this fits in a weekend.

| # | Number | How | Status in the literature |
|---|---|---|---|
| 1 | **Split-half reliability** | Split your rollouts per cell in half, correlate the two halves | **Zero prior hits. This is your headline.** |
| 2 | **Minimum detectable difference** | From observed between-rollout variance, the gap you'd need for 80% power | Unpublished |
| 3 | **Cost vs ranking fidelity** | Sweep resolution, denoising steps, chunk count; plot agreement against GPU-seconds | Unpublished |
| 4 | **Drift vs ranking fidelity** | Chain 1, 2, 3, 4, 5, 6 generation steps; plot where the ranking breaks | **Nobody has this. Falls straight out of single-frame conditioning.** |

Report agreement with real robots too — but as **validation, not headline**, against the published bar of Pearson 0.78 on this robot.

---

## End to end

```
  ┌─ one rollout ────────────────────────────────────────────┐
  │                                                           │
  │  frame 0  ──►  POLICY (Baseten endpoint)                 │
  │                images + state  ──►  16 actions (7-D)      │
  │                        │                                  │
  │                        ▼  convert to 10-D                 │
  │                WORLD MODEL (Baseten, async_predict)       │
  │                frame 0 + [16,10]  ──►  17 frames          │
  │                        │                                  │
  │                        ▼  drop frame 0, keep 16           │
  │                take frame 16 as new frame 0 ──┐           │
  │                chain 5x  ◄────────────────────┘           │
  │                        │                                  │
  │                        ▼  80 frames                       │
  │            ┌───────────────────────────┐                  │
  │            │ STAGE A — gate (Python)   │                  │
  │            │ recover motion from frames│                  │
  │            │ vs commanded; optical flow│                  │
  │            │ invalid → EXCLUDE, count  │                  │
  │            └───────────┬───────────────┘                  │
  │                        ▼                                  │
  │            ┌───────────────────────────┐                  │
  │            │ STAGE B — judge (VLM)     │                  │
  │            │ 16 frames + goal + motion │                  │
  │            │ summary. integrity /      │                  │
  │            │ collision / progress as   │                  │
  │            │ separate fields. Label    │                  │
  │            │ computed in Python.       │                  │
  │            └───────────┬───────────────┘                  │
  └────────────────────────┼──────────────────────────────────┘
                           ▼
              one JSONL line per rollout
       (seed, policy rev, WM rev, judge rev, prompt hash,
        score, steps, termination, wallclock, video path)
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   success rate      split-half r      cost & drift
   ± Wilson CI       MDD               curves
         │
         ▼
   compare to AutoEval Table 2 (15 cells)
```

**Critical mechanics, each of which silently ruins a run:**

- **16 actions in, 17 frames out.** Never send a frame count — it's derived as `chunk + 1`. Chunk must be a multiple of 4.
- **Frame 0 is your input echoed back.** Drop it from every chunk before concatenating, or you feed duplicate frames to the judge.
- **Conditioning is one frame.** Do not try to re-anchor on five; the field is rejected in action mode.
- **Rotation is the first two columns, column-major.** Raw metres.
- **Bootstrap clusters on tasks, not episodes.** Wilson intervals for the rates themselves.
- **Judge at non-zero temperature** with self-consistency. A judge published that temperature-zero determinism is a fallacy.

---

## Build order

| Hour | Do | Kills the project if it fails |
|---|---|---|
| 0–0.5 | **Forward-dynamics smoke test.** One call, Bridge domain, chunk 16, real frame. | Yes — no published proof this works on Bridge |
| 0.5–1.5 | **Action-permutation null.** Shuffle the action chunk; the judge must stop scoring success. | Yes — if it passes, nothing downstream means anything |
| 1.5–2 | Four integration tests: frame order, gripper polarity, rotation round-trip, step magnitude | Silently, yes |
| 2–6 | Rollout harness, async fan-out, one policy end to end | |
| 6–10 | Three policies × 3 tasks × 100 rollouts | |
| 10–12 | **Split-half + MDD.** Your headline. | |
| 12–14 | Judge calibration: 150 clips, two raters, 50 overlapping | |
| 14–17 | Cost sweep and drift sweep | |
| 17–20 | Judge distillation on Training Jobs | |
| 20–22 | Dashboard | |
| 22–23 | **Reverse validation** — 35 min, answers a judge's published open question | |
| 23–24 | Publish as an environment, write the submission | |

Two hard checkpoints: **if the smoke test fails, switch world model immediately** to the Apache-2.0 alternative with a native Bridge checkpoint. **If the null test fails, stop and fix it** — do not build on top.

---

## Cut list — do not build these

- **The checkpoint ladder.** A published paper already did checkpoint-sweep ranking preservation on this robot. Your MDD comes from observed variance and doesn't need it. Stretch goal only.
- **SuSIE and SuSIE-LL.** Two servers, pinned JAX/Flax, and the inversion they buy is now a secondary claim.
- **Running SIMPLER yourself.** Use its published table.
- **Open pi-zero.** Reachable, but a fourth policy adds 5 cells and costs hours.
- **Anything touching the world model's weights.** Eight H100s in every shipped recipe.

---

## Demo — three shocks in 180 seconds

A wall of twelve tiles was modest. This is built around three moments that should actually land.

| Clock | Beat | Sec |
|---|---|---|
| 0:00 | **Shock 1 — the room gets it wrong.** Six clips, show of hands, only one is real. | 20 |
| 0:20 | Problem and prior work. All three reported accuracy; none reported precision. | 12 |
| 0:32 | **Shock 2 — the called shot.** Simulator 4%. You state yours *before looking*. Then 92%. | 20 |
| 0:52 | The four numbers, framed as their own three criteria. | 18 |
| 1:10 | Economics and limits. The eleven dollars is a **promise**, not yet a receipt. | 25 |
| 1:35 | **Shock 3 — LIVE, THE FINALE. 85 seconds: the wall (12s), arrow keys (20s), the burst (40s), the dial (13s).** | 85 |

**The live app closes the pitch and gets 47% of the slot.** Two earlier versions of this plan got it wrong: one gave the app 60 seconds and spent them all on a counter climbing, so the product was never shown; the other put the app in the middle, so the pitch ended on slides. The wall is back, it goes first inside the live block, and the live block goes last.

**The structural win of demo-last:** the eleven dollars becomes a promise the burst keeps ninety seconds later, instead of a number you describe after the fact.

### Why shock 3 is the centrepiece

The study you validate against took a Berkeley lab **several days and 1,500 real robot trials**, with a person scoring every episode and twenty-minute cooldowns because the motors overheat. You reproduce its scale in **one minute for about eleven dollars**, in front of them, from zero.

**And it is only possible because of the cost curve.** At the cheapest setting on your fidelity sweep — the one you proved still ranks correctly — a rollout is roughly four GPU-seconds. 1,500 of those is about 1.7 GPU-hours, which one hundred replicas chew through in a minute. Say that in one clause: *this runs at the cheap end of our curve, which is the end we showed still ranks correctly.* The two results hold each other up.

**It is also the only beat in the entire demo that could not exist on another platform.** Elastic burst to a hundred replicas and back is the product. Make the replica graph big.

### The booth ask, day zero

**Your concurrency cap defaults to one replica.** Ask Baseten at the booth, in these words: *"can you raise our inference concurrency to a hundred replicas for a sixty-second burst on Sunday morning — we want to reproduce a 1,500-trial robot study live on stage."* That is a specific, flattering, technically literate ask, and it gets a human at Baseten personally invested in your demo working. Do it in the first hour.

**If the cap is not raised:** run 400 rollouts at 30 replicas and change the number on the slide. Still dramatic, still the same argument.

### Pre-test shock 1, or cut it

Saturday night, show the six clips to strangers at the venue. If people reliably pick the real one, **cut that slide** and open on the called shot. The beat only works if the room genuinely fails.

### Arrow keys — in the pitch, and it is proof rather than decoration

**Why it earns 20 of the 85 seconds.** The wall provokes a question a sharp judge will ask silently: *are those twelve videos pre-rendered?* Arrow keys answer it in four seconds. You press right, wait two seconds, and the world invents an arm moving right. Nothing pre-recorded responds to a keypress. Put it immediately after the wall, where the doubt forms.

**The mechanics.** One keypress commands a whole 16-action chunk, so the loop is: press, ~2 second wait, 16 frames of motion. Four or five presses fit in 20 seconds. No policy, no language model, no scoring — the keypress becomes an action vector directly, which also makes it the most robust thing in the demo, because you have removed language grounding entirely.

**You drive, not them.** Handing over a keyboard on stage costs ten awkward seconds and can fail. Drive it yourself and narrate. **Save the handover for the booth**, where you have ten minutes and a lingering judge — that is where "you are the policy now" lands.

**Narrate the latency, do not apologise for it.** "I press right — and it is inventing the next second and a half of video right now." The wait is the interesting part.

## Things to say, verbatim

- "We used your published LoRA optimum — learning rate 1e-3, r=64, alpha 32, two epochs — rather than the repository default."
- "Concurrency target of one is correct here; a diffusion model saturates the GPU. Our levers are batch packing and replicas."
- "The validity gate is Python, not a model call. We took the work away from the model."
- "Goalposts we cannot move until we cross them" — quote their manifesto when you show the timestamped pre-registration commit.

## Never say

SOTA · solves · matches human performance · first ever · time to first token · tokens per second · "we solved continual learning" · rollout volume as a headline.

Say "we could not find a published version" instead of "first."
