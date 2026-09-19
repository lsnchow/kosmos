# PLUMB — pass-off

Quick orientation for anyone joining the project. Design revised after adversarial review, 2026-09-18; implementation evidence updated 2026-09-19. The local control plane, dashboard, and cluster diagnostics now run. No qualified PLUMB scientific result or Baseten deployment is established. See [HANDOFF.md](HANDOFF.md) for resumable execution state.

---

## What it is

**A ruler for robot policies — with measurements of how much that ruler wobbles.**

The full build evaluates six robot policies across five tasks inside a generative video world model on Baseten: **50 matched starts per task, 1,500 attempted virtual episodes**. It compares those results with the published AutoEval human reference, then measures repeatability, minimum detectable difference, drift, and the cost-fidelity curve. Comparison requires compatible scenes, policy feedback, action conventions, and task horizons; generating 1,500 videos alone does not establish validity.

**Twelve concurrent virtual rollouts on screen, with a scoreboard that tells you how much to trust it.**

## Why this is compelling

The product puts policy scores, uncertainty, failure coverage, and inference economics on the same screen. World-model evaluation already has substantial prior work; PLUMB's proposed contribution is an auditable measurement protocol and live execution of it. A dated literature comparison is required before making any “first published” claim. Sources and implementation detail are in [the build spec](AGENT-BUILD-SPEC.md#12-evidence-and-open-dependencies).

## The claim, precisely

Report per-cell virtual success, its difference from the published reference, and supported policy ordering with explicit uncertainty and exclusions. MiniVLA leads Open π0 by one success in 250 in the reference; **four of five** leave-one-task-out comparisons reverse that pair. The four headline outputs remain **split-half reliability, minimum detectable difference, drift, and cost-fidelity**. None is yet a PLUMB result. [AutoEval](https://arxiv.org/html/2503.24278v2)

## What the demo is

Three beats in 180 seconds; rehearse using the qualified configuration.

1. **Six clips. Which is real?** One real clip, five generated clips, randomized positions. Reveal provenance afterward. Report audience accuracy only if actually collected.
2. **The called shot, with a locked protocol.** Show PLUMB's frozen-run estimate next to the published OpenVLA/close-drawer results: 92% human versus 4% SIMPLER. The published gap is 88 percentage points; PLUMB's error remains to be measured. Since the reference is already known, this is a preregistered comparison, not a blind prediction.
3. **The full matrix, live.** One button launches all 1,500 virtual episodes. **Qualification target: complete scoring and persistence in 60 seconds for approximately $11 total demonstration-run cost.** Start with a 100-replica capacity hypothesis, then size from measurements. Show measured scale-up separately from the prewarmed execution timer; warm-up and other stages count toward the total cost.

The burst showcases Baseten Chains, independently scaled inference stages, and live observability. Its cheap configuration must pass held-out fidelity checks and three full load rehearsals. The target remains in scope if qualification fails; display actual results and the unmet target rather than substitute a smaller study or simulated counters.

## The business case

A compatible-policy screen could reduce demand for engineer and robot time. Quantify that value against a documented customer workflow after measuring PLUMB's accuracy, runtime, and cost. Physical confirmation, initial calibration, and continuing human audits remain part of the workflow.

**Show the rollout counter and estimated spend together**, with separate marginal execution and fully loaded demonstration costs. Reconcile estimates against billed usage afterward.

For Baseten, evaluation is a recurring inference workload: each compatible fine-tune can trigger another run. Episodes parallelize across replicas; each episode's policy/world feedback remains sequential. This business hypothesis depends on passing the measurement gates.

## Where the risk is

| Risk | Status |
|---|---|
| Action and feedback fidelity | **Blocked for Cosmos native one-step; experimental for IRASim.** Cosmos returned no next frame for one action. IRASim completed a real 16-tick OpenVLA feedback loop, but severe visual drift appeared; this is an engineering trace, not fidelity qualification. |
| Five-task benchmark comparability | **Unresolved.** Matched drawer, sink, and cloth starts, goal references, and control horizons must be established. |
| 1,500 episodes / 60 seconds / ~$11 | **Target, not evidence.** Capacity, cold starts, policy inference, judge sampling, persistence, and billing all matter. |
| Judge reliability | **Uncalibrated.** Actual five-sample Qwen diagnostics ran; the closed-loop clip returned unknown with insufficient quorum. Blinded human calibration is still required before scoring, and again after distillation. |
| Assets and missing artifacts | **Selected assets cluster-side and hash-verified.** Cosmos Nano, OpenVLA, Qwen, original IRASim safe tensors, SDXL VAE, and Octo-Small are staged; remaining policy assets, scenario panels, and original handoff material remain dependencies. No model weights were copied to the laptop. |

Current demoable engineering evidence: the dashboard at `http://127.0.0.1:8787`
separates the 1,500-row **synthetic fixture** from actual model diagnostics and
their playable clips. Trillium job `937704` completed 16 fresh OpenVLA→IRASim
ticks and persisted 17 frames in 142.21 seconds including model load and artifacts.
The exact trace is auditable, but the final gripper is visibly distorted; neither
task success nor physical fidelity is asserted. This does not satisfy the real
1,500-episode study or its performance target.

The continuation adds durable submission/callback recovery, scenario and blinded
calibration workflows, full-study planning, and distillation-data preparation.
Octo-Small completed real native-v0.1 GPU calls and an eight-worker consistency
check. That source profile differs from the unverified newer AutoEval API, so
it is diagnostic evidence only. A new causal-history IRASim replay still showed
substantial visual distortion. See the current first section of `HANDOFF.md`.

## What we deliberately are not doing

The design excludes a checkpoint ladder, world-model adaptation, policy fine-tuning for improvement, a local SIMPLER installation, and an environment hub. These are scope choices. All six benchmark policies, the four measurement goals, judge distillation, free-play, the 480p hero, and reverse validation stay in scope.

## Honest limitations, stated on the page

Virtual success is not automatically real-world success. Show indeterminate orderings when uncertainty or missing outcomes prevents separation. Audit possible training-data overlap; shared model lineage is not established for every component. Octo's low published score and the SuSIE configuration mismatch warrant sensitivity analyses, not unsupported accusations of bugs. Measure judge leniency rather than assume it. Forecast proprioception and unverified feedback approximations must be disclosed.

## Documents

- [AGENT-BUILD-SPEC.md](AGENT-BUILD-SPEC.md) — implementation requirements, evidence, gates, and known missing dependencies.
- [README.md](README.md) — run the implemented app; [HANDOFF.md](HANDOFF.md) — cluster jobs, source releases, diagnostics, and continuation instructions.
- **Missing from this workspace:** `BUILD-SPEC.md`, `SCRIPT.md`, `htn2026-field-guide.md`, and `reverse_validation.py`. Recover their original sources before treating them as evidence. Reverse validation remains planned; its original question and script have not been supplied.
