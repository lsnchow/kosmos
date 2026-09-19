# PLUMB judge distillation — Baseten Training Jobs

Three files, one job per preregistered LoRA arm.

| File | What it is |
|---|---|
| `requirements.txt` | Pinned lock *input* for the training image. Nothing built from it has been resolved by pip. |
| `job_config.py` | The concrete job configuration. Renders the Training Jobs payloads; has no submit mode. |
| `train_judge_lora.py` | The entrypoint that runs inside the job. Importable without torch; `--check` validates on CPU. |

**It has never been submitted.** This machine has no GPU, no Baseten
credentials, and torch is not installed. What *was* verified: both Python
modules import with torch absent, `job_config.py` renders three payloads whose
lineage disjointness reads `pass`, and `train_judge_lora.py --check` accepts a
valid config and refuses a leaky one — all under `tests/test_distillation.py`.
That checks this directory's own logic. It is not a job, not an image build, and
not evidence about training behaviour, convergence, GPU memory, or cost.

`plumb/distillation.py` owns every refusal. This directory only supplies the
container identity and the values. `docs/DISTILLATION_RUNBOOK.md` is the ordered
runbook for the moment credentials go live.

## Why Training Jobs and not the Chain

`AGENT-BUILD-SPEC.md` §7: judge distillation uses **Training Jobs**, with
framework/model compatibility checked in its own container. It is not the RL SDK
(form-gated and text-only, per `BUILD-SPEC.md`), not a Chainlet, and Loops'
access and model catalogue are not a dependency. The serving Chain in
`../chain.py` is untouched by distillation: a distilled judge is a *new judge
revision*, which invalidates Gates D, E and F rather than editing them.

## The one thing that is not a knob

The dataset URI, the Gate-D held-out lineages, and the primary-study lineages
are required command-line inputs to `job_config.py`. They have no defaults on
purpose. Guessing them would defeat the disjointness check that is the entire
reason this job is gated.
