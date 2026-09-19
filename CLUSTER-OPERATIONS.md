# PLUMB cluster operations — handoff for num2

## Current Baseten managed inference

Model3mzlenow / deploymentq929yoj, team33/q8grpdw, profileplumb-api is the
verified diagnostic MVP. Use docs/BASETEN-MVP.md. Actual cloud H100 calls and
cold wake passed; isolated child runtime avoids Truss/ML dependency collisions.
Latest GitHub frontend is merged. LOCAL plumb-live serves localhost8787; b10
retains operator CLI context. Max1/min0 replicas,60s idle scale-down. Older
failed/superseded versions are retained inactive; do not reactivate them.
Organization SSH is disabled (provider400); no setting was changed to bypass
that. GPU/credit quotas were not modified. Managed model inference works
without SSH. This deployment's production label does not qualify any gate.

## NEW Baseten access — LOCAL tmux `b10`

Baseten API-key profile `plumb-api` is configured and verified. Use it explicitly
outside `b10` (`--profile plumb-api`); the shell in `b10` has
`BASETEN_PROFILE=plumb-api`. Global OAuth/default configuration is unchanged.
The parent workspace is Hack the North, but use **team 33 / `q8grpdw`** for
resources: `--team q8grpdw`. The default team named Hack the North has no H100
capacity. Last verified Team 33 training allowance: **4 H100s, 0 in use**;
recheck with `baseten train capacity describe --profile plumb-api`.

`baseten ssh setup --profile plumb-api` completed and installed its managed
SSH configuration. A future owned RUNNING SSH-enabled training job connects as
`ssh training-job-JOB_ID-0.ssh.baseten.co`; replace JOB_ID with the actual job ID.
No job was created and no live SSH session tested in the side conversation.
Workspace SSH enablement and billing/product entitlement still need checking.
Do not confuse Baseten setup with restored Trillium authentication below.
Do not inspect/print credentials, change shared team limits, interrupt another
pane's running work, or treat closing SSH as stopping billable compute.
See HANDOFF.md's new first section and docs/BASETEN-TEAM33-MVP-TODO.md.

Final authentication recheck: Trillium BatchMode returned permission denied
after job940190 completed and its evidence was mirrored. The last successful
scheduler check was empty; no later job was submitted. No Duo retry was sent.
Do not treat the earlier authenticated snapshots below as current access.

## Current judge format-only pilot — do not deploy the adapter

Training job 939998 and reload comparison 940000 completed from immutable release
`f6b30d952be1b2e9cccf4c5915a4e85a636b22b620634e485dfde0ab77c08fc5`.
The pilot used 12 development-train rows, 4 development rows, 24 steps, and a
structure-only mask with zero semantic supervised tokens. Syntax loss changed
1.71370849→0.02586903 and bare JSON 0/4→4/4, but judgments drifted 4/4 while
both base and adapter remained schema-valid 4/4. Keep the adapter disabled.
Evidence: local `data/live-integrated/cluster-evidence/judge-format-{pilot-939998,compare-940000}/report.json`; remote
`/scratch/lchow432/plumb/experiments/judge-format-only-v1` and
`evidence/judge-format-only-v1-comparison.json`. No new weights were downloaded;
adapters stay cluster-only, gates are unchanged, and no human labels exist.
Root verified live `/review` in isolated Chromium/temporary-DB QA: 16 clips and
decoded images loaded; draft save/reload/resume, no JS errors, and no 390px
horizontal overflow. Production review DB is absent and isolated
`model_assisted` test drafts are not human labels. App tmux `plumb-live`
PID19103 exposes24 diagnostics and all24 report URLs return200; health is
`qualified=false`, all gates `not_run`, synthetic-only. Do not claim a live
human-review outcome or full-scope completion. See `docs/JUDGE-FORMAT-PILOT.md`.

SuSIE_LL job940190 COMPLETED0:0 in20 seconds on one H100, status
`completed_unqualified`. It made two finite `[1,7]` gc_bc calls with exact
reset/repeat on a static vendor first/final-frame fixture—not rollout, task
success, Gate A/B, or primary evidence. Checkpoint:
`/scratch/lchow432/plumb/models/patreya--gcbc-bridge/checkpoint/checkpoint`, SHA`80b354...`; publisher
README declares MIT. Correct SOAR source is
`/scratch/lchow432/plumb/source-soar-gcbc@eabd5f16a856e484884a22e257a941bb358cea08/model_training`,
not wrong `bc60...`. Strict params-only inference restore used
`target_params=None` and excluded optimizer state; it is not training resume.
Full restore940180 and failures940172/940173/940174 are preserved locally.
Evidence local `data/live-integrated/cluster-evidence/susie-ll-gcbc-940190/report.json`,
remote `evidence/susie-ll-gcbc-static-goal-v5.json`, release `fdc52d...`. No
adapter promotion; last successful scheduler check empty. Latest checks: Python1267 passed/6 skipped,
frontend144 passed/build passed.

## Access restored — latest continuation

User reauthenticated; Trillium BatchMode SSH now works. Scheduler check showed
no user jobs running. Completed model jobs were not repeated. Source review
media23MB and the full v2 raw report bundle were mirrored locally; no model or
adapter weights were copied. Actual development-review packets are ready; see
HANDOFF.md's newest section. Earlier expired-session notes below are historical.

## Latest — actual pilot and v2 diagnostic, SSH expired

Teacher939423 COMPLETED0:0 (5m34s), optimizer training939451 COMPLETED0:0
(46s), saved-adapter comparison939458 COMPLETED0:0 (1m42s). Actual adapter and
metrics are in docs/JUDGE-PILOT-V1.md; never replay these jobs unchanged.
Teacher-v2 diagnostic939753 reported a completed16-clip summary with75/80 valid
sample slots butzero3-of-5 unique modes. Its raw bundle remains in
`experiments/judge-teacher-v2-diagnostic`, not yet fully mirrored locally.

Afterward Trillium's local control socket disappeared and BatchMode SSH failed
with permission denied. LOCAL drac:0.0 is a laptop zsh prompt, no active SSH
shell. One normal re-login sent a Duo push; it timed out and the agent-owned
PTY was cancelled. User needs `ssh trillium-gpu` and Duo approval locally.
Never repeatedly push, delete sockets, or disturb other cluster sessions.
No new GPU task was launched after the completedv2 diagnostic. Recheck live
Slurm state on reconnect. See docs/JUDGE-REVIEW-NEXT.md for transfer/audit steps.

## Judge LoRA preflight — latest, 2026-09-19 afternoon

Job939218 COMPLETED0:0 on one Trillium H100,57s Slurm elapsed. It checked
Qwen LoRA forward/backward only: **no optimizer step, training run or adapter**.
No jobs remained in `squeue` afterward. See docs/DISTILLATION_RUNBOOK.md's first
section for actual metrics and pending user choice on an uncalibrated pilot.

- Python: `/scratch/lchow432/plumb/venv-judge-lora-tf449/bin/python`.
- Modules: `StdEnv/2023 python/3.11 arrow/19.0.1`; batch additionally loads
  `cuda/12.2 cudnn/8.9.5.29`. Arrow module is required by Datasets/PyArrow.
- Runtime lock: `evidence/judge-training-runtime-tf449-v1.json`.
- Fixture: `fixtures/judge-lora-preflight-v1/row.json` with16 hash-bound PNGs
  decoded from the original teacher MP4 and a separately bound scene reference.
- Model view: `models/qwen-judge-training-view-cc594-v1`, a **hard-linked** view
  of the existing Qwen files, excluding only downloader `.cache` metadata.
  Every actual model/support file is independently hash-checked before load.
  Do NOT edit weights in either view: their inodes are shared. No new weights
  were downloaded. Only LoRA adapters may be saved to a fresh output directory.
- Raw report/log: `evidence/judge-lora-preflight-939218.json` and
  `logs/judge-lora-preflight-939218.log`; local copy is under
  `data/live-integrated/cluster-evidence/judge-lora-preflight-939218/`.
- Source release:82eb4ab6e401eea765857d7e568444d1dd9cc893ea774a35621d02f53c51fdf2.

Do not relaunch this completed check unchanged. New training requires real
dataset work and an explicit protocol/profile distinction; do not fabricate
Gate D, preregistration or cohort proofs to satisfy the formal config parser.

## Latest integration operations — 2026-09-19 late morning

Allocation937277 expired TIMEOUT at09:11 EDT; the older running/pending notes
are historical. Trillium BatchMode SSH works. One-H100 debug job938918 tested
the newer production241fb Octo harness and FAILED before inference on a missing
harness runtime field. Preserve its raw report/log. Retry938946 COMPLETED0:0
with three actual production241fb native calls and exact reset/repeat. The new
release and report locations are in HANDOFF.md. Do not repeat this successful
check or mutate either release; neither check is a qualified policy evaluation.
Both jobs are terminal; no new allocation is intentionally left running. Recheck
live scheduler state before any future submission. The live local app is now
LOCAL tmux `plumb-live`, port8787, using the merged main source; older plumb-api
and PID65815 references below are historical. See HANDOFF.md for restart/data.

Production241fb source and environment are separate from the old tested37951
ones: `/scratch/lchow432/plumb/source-octo-autoeval241fb` and
`/scratch/lchow432/plumb/venv-octo-autoeval241fb`. Runtime lock is
`evidence/octo-runtime-autoeval241fb-v1.json` (SHA in HANDOFF.md). All weights and
canonical input PNGs are reused, not re-downloaded. Its dlimp pinned dependency
needs TF2.15.0 but the runtime uses reviewed2.15.1+computecanada; pip-check output
and this explicit exception are in the lock. Do not call the environment exact
upstream-equivalent or silently amend old runtime locks.

Submit `cluster/octo_production_smoke.sbatch` from login with the immutable
release directory, frame-00.png, frame-01.png, manifest.json, close_drawer.
It requests one H100,24CPUs,15minutes, no explicit --mem. Imports/package locking
on login use CUDA_VISIBLE_DEVICES=""; all actual inference runs in the job.
Root exclusively owns cluster operations; coding agents must not SSH/tmux/srun.

## Continuation update — 2026-09-19 morning

This section supersedes the older capacity/allocation snapshot below.

- All five SSH routes were successfully rechecked. Keep using the existing
  multiplex connections; never delete sockets to recover an apparent error.
- **937277 started at05:11 EDT**, with trig0010/trig0030 and eight H100s. Its
  scheduled end is09:11 EDT. The new bounded Octo checks have completed; no
  model step is still running. Root asked asynchronously whether to release
  the now-idle parent allocation, because cancellation was previously withheld.
  Without an explicit reply it remains intact; recheck `squeue` before acting.
- Remote `drac:0.0` now contains the allocated compute shell on trig0010,
  rather than a pending SALLOC request. The site rejects `srun --jobid=937277`
  from a login node. Root launched steps from the **inspected idle compute
  prompt** in that pane. Never send keys while a step or allocation wait is
  active, and give cluster-control ownership to only one worker at a time.
- Completed steps:937277.0 failed on the old Octo API keyword;937277.1 succeeded
  with the corrected native-v0.1 API;937277.2 succeeded with eight workers.
  Reports and logs are preserved. Source releases and artifact paths are in the
  new first section of `HANDOFF.md`. These are policy diagnostics, not episodes.
- Job937822 completed the new IRASim causal past-history replay on one H100.
  It remains visibly distorted despite exact repeatability. Do not rerun it.
- The LOCAL app now runs in tmux `plumb-api` on127.0.0.1:8787. This is unrelated
  to all SSH/tmux allocation sessions; inspect its actual listener before restart.

### Octo environment (do not mix with the older Python3.11 stacks)

Use `/scratch/lchow432/plumb/venv-octo/bin/python` with
`StdEnv/2023 python/3.10 cuda/12.2 cudnn/8.9.5.29`. It uses Alliance
JAX/JAXlib0.4.20, Flax0.7.5, Orbax0.4.3, TF2.15.1 and exact TFP0.23.
TF2.15.1 is a disclosed compatibility variation, not a source-equivalence claim.
Runtime lockv2 and an inline package inventory live under `evidence/`; its SHA
is `5c10b8881e8ab97830ee84f565aeebe1cb689f1d161359a64a2ca450a62e74b9`.

`HF_HOME` must be `/scratch/lchow432/plumb/models/.hf-octo`, with both offline
flags set. T5 config/tokenizer support is present; full T5 model weights were
not needed. TensorFlow GPU visibility is disabled before model imports so JAX
owns the allocated device. Preserve module `PYTHONPATH` when adding a release.

`cluster/octo_smoke.sbatch` accepts a frozen source directory, two canonical
PNG paths, their manifest, and a task ID. Reports now include job/step/rank to
avoid overwriting a failure during retries inside the same allocation. Current
verified RGB inputs are under
`fixtures/octo-small-bridge-cv2-linear-rgb-256-v3/`: files `frame-00.png`,
`frame-01.png`, and `manifest.json`, sourced from original frame indices0/2.
Do not substitute the earlier0/1 fixture or the original480p MP4 silently.

For a future *new hypothesis*, after checking an idle allocated shell:

```bash
srun --jobid=937277 --overlap --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --cpus-per-task=24 --time=00:15:00 \
  --output=/scratch/lchow432/plumb/logs/octo-new-%J.log \
  bash "$PLUMB_RELEASE/cluster/octo_smoke.sbatch" "$PLUMB_RELEASE" \
  "$OCTO_INPUTS/frame-00.png" "$OCTO_INPUTS/frame-01.png" \
  "$OCTO_INPUTS/manifest.json" close_drawer
```

Set those variables to inspected exact paths in that compute shell. If937277
has expired, submit a bounded one-H100 batch job from login instead. Do not
resubmit the completed conformance checks merely to reproduce their success.

---

Read with HANDOFF.md. Last capacity/auth check: 2026-09-19 at about04:11 EDT.
Capacity is a snapshot: recheck before submitting. All five SSH connections worked.

FINAL RECHECK CAVEAT: a later Trillium BatchMode call returned
"mux_client_request_session: read from master failed: Broken pipe" followed by
"Permission denied (keyboard-interactive,hostbased)". Thus the earlier all-five
auth success is historical, not a guarantee for the new context. Inspect the
existing local drac SSH pane/control connection before requesting reauth.
Do not assume remote batch jobs/evidence disappeared. If MFA is truly required
while the user sleeps, do independent local work and report the access blocker.

## GPU requirement and current jobs

ONE H100 is sufficient for the next experiments. Every completed real diagnostic
used one allocated GPU; OpenVLA and IRASim together also fit one H100.
No long training run or active GPU experiment is waiting to finish.

Only job937277 is pending: Trillium, two whole nodes x four H100, four hours,
compute_full_node, reason Priority, no start estimate. It is not an auth failure.
The user now says "use what you can get"; do NOT wait for this large request or
create duplicates. Cancellation was previously asked about but not clearly
confirmed, so it was left unchanged. Check before cancelling anything; never
cancel unrelated user work. If it starts, do not leave allocated GPUs idle.

| Cluster | Actual latest observation |
|---|---|
| Trillium | One-H100 debug and one-H200 compute_h200 test-only requests both estimated immediate starts. Node trig0063 had four idle H200s. All model assets already staged here. |
| Fir | Mostly occupied full H100s. Two unallocated GPUs on fc10519 were MIXED+PLANNED, not guaranteed free. |
| Rorqual | Mostly occupied full H100s. One unallocated GPU on rg31902 was MIXED+PLANNED. |
| Nibi | Full H100s occupied; g29 is DOWN+DRAIN, unusable. A100 node o1 had six unallocated GPUs, subject to scheduler eligibility. |
| Queen's | Idle A100 nodes exist. One-A100 dry-run estimated04:31:42 on frnt107, not a guaranteed allocation. Qwen is already staged here. |

Dry-run IDs937782/937783 and12240152 were sbatch --test-only outputs, NOT
submitted GPU jobs. No jobs were queued for this user on Fir/Rorqual/Nibi/Queen's.
Do not misread MIXED node state as free GPUs: CPU slots can remain free while all
GPUs are occupied. Inspect CfgTRES versus AllocTRES with scontrol show node.

## Two separate tmux layers

The laptop has local tmux sessions containing SSH clients. The clusters also
have their own remote tmux servers. Same name on different hosts != same session.
Being back at a login prompt does not prove a batch job died; Slurm is authoritative.

| LOCAL session/pane | Destination | User |
|---|---|---|
| drac:0.0 | SSH alias trillium-gpu -> trillium-gpu.alliancecan.ca | lchow432 |
| drac2:0.0 | fir.alliancecan.ca | lchow432 |
| drac3:0.0 | rorqual.alliancecan.ca | lchow432 |
| drac4:0.0 | nibi.alliancecan.ca | lchow432 |
| cac:0.0 | SSH alias cac -> login.cac.queensu.ca | hpc6308 |

Uppercase LOCAL sessions DRAC and CAC are unrelated zsh shells: leave them alone.
Pane IDs change; inspect names/current commands rather than relying on old IDs.

REMOTE Trillium: login trig-login01, tmux session drac, pane drac:0.0 is SALLOC
waiting for937277. Do not send Ctrl-C, exit, or arbitrary text to this pane.
REMOTE Queen's: login login1.frontenac.local, session plumb, four bash windows
plumb:0.0 through plumb:3.0. Its old A100 allocation12238810 has expired.

### Preferred read-only access from the laptop

~~~bash
tmux list-sessions
tmux list-panes -a -F '#S:#I.#P #{pane_id} #{pane_current_command}'
ssh -O check trillium-gpu
ssh -o BatchMode=yes -o ConnectTimeout=10 trillium-gpu 'hostname; squeue -u lchow432'
ssh -o BatchMode=yes lchow432@fir.alliancecan.ca 'hostname; squeue -u lchow432'
ssh -o BatchMode=yes lchow432@rorqual.alliancecan.ca 'hostname; squeue -u lchow432'
ssh -o BatchMode=yes lchow432@nibi.alliancecan.ca 'hostname; squeue -u lchow432'
ssh -o BatchMode=yes cac 'hostname; squeue -u hpc6308'
ssh trillium-gpu 'tmux list-sessions; tmux list-panes -a'
ssh cac 'tmux list-sessions; tmux list-panes -a'
~~~

Direct SSH commands reuse authenticated ControlMaster connections. Current config:
ControlMaster=auto, ControlPersist=28800 seconds, sockets under
/Users/lucas/.ssh/sockets/. Do not delete sockets or kill working SSH sessions.

To inspect the remote allocation without interfering:

~~~bash
ssh trillium-gpu 'tmux capture-pane -p -t drac:0.0 -S -20'
ssh trillium-gpu 'scontrol show job 937277; squeue --start -j 937277'
~~~

For a human terminal, local "tmux attach -t drac" shows the laptop SSH session.
"ssh -t trillium-gpu 'tmux attach -t drac'" attaches the REMOTE tmux session.
Detach with Ctrl-b then d. Do not exit an allocation shell just to detach.
Nested tmux has two prefix layers; direct SSH commands are simpler for automation.

For NEW interactive work use a separate remote window:

~~~bash
ssh trillium-gpu 'tmux new-window -t drac -n plumb-next'
~~~

Inspect hostname and an idle shell prompt before sending keys. Never paste into
a running installer, Python REPL, allocation request or authentication prompt.
If BatchMode fails, existing authenticated local panes may still work: inspect
them without killing them. If access really expired, MFA requires the user.
Do local coding while waiting; do not invent credentials or bypass authentication.

## Trillium: use the already-installed single-GPU route

Root: /scratch/lchow432/plumb. Models/, environments, fixtures, evidence, logs,
and immutable releases already exist (actual directory name is lowercase models).
No PLUMB assets were staged on Fir/Rorqual/Nibi. Avoid redistributing80+GB
unnecessarily. H200 is a possible alternative, but PLUMB inference on H200 has
NOT yet been validated; label the hardware change and recheck conformance.

Non-submitting eligibility check:

~~~bash
ssh trillium-gpu 'sbatch --test-only --account=def-ikarlin --partition=debug --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=24 --time=00:10:00 --output=/scratch/lchow432/plumb/logs/eligibility-only-%j.log --wrap=true'
~~~

Replace debug with compute_h200 to test H200. Trillium requires scratch output:
home is read-only on compute nodes. Do NOT supply --mem; the site rejects it
and assigns memory per GPU. Use24CPUs/GPU and bounded wall time.

Batch jobs survive loss of the SSH connection without interactive tmux.
If using salloc, launch it inside REMOTE tmux, not merely laptop tmux.
GPU work must run on an allocated compute node. Never use login-node GPUs or
devices outside CUDA_VISIBLE_DEVICES. Do not infer ownership from nvidia-smi alone.

## Immutable source release / submission workflow

From the laptop, after NEW code changes:

~~~bash
cd /Users/lucas/Desktop/tax_stuff/htn26
.venv/bin/python -m pytest -q
.venv/bin/python cluster/make_release.py
~~~

The final command prints archive path,64-hex release SHA, byte count and file count.
Copy that exact small archive to /scratch/lchow432/plumb/releases/ using scp.
Extract into a NEW releases/RELEASE_SHA directory; never modify an old release.
The tar contains source/tests and RELEASE.json, not models/run data.

Inside the remote shell, set PLUMB_RELEASE to the actual new frozen directory:

~~~bash
sbatch "$PLUMB_RELEASE/cluster/irasim_probe.sbatch" "$PLUMB_RELEASE"
~~~

All current Trillium sbatch scripts accept the source directory as argument1.
Pick the script appropriate to the NEW hypothesis; do not repeat finished runs.
An intentional H200 test can override the script's partition with
--partition=compute_h200 on the sbatch command line. Record hardware separately.

Check final squeue/sacct status AND the persisted experiment JSON. Some site
log footers showed provisional exit0 before the final failed state appeared.
An MP4 or successful scheduler footer is not scientific qualification.

~~~bash
ssh trillium-gpu 'sacct -j 937775 --format=JobID,State,ExitCode,Elapsed -n'
~~~

937775 is a completed example; replace it with your actual NEW job ID.
Copy only small evidence bundles back locally, e.g.:

~~~bash
scp -r trillium-gpu:/scratch/lchow432/plumb/evidence/irasim-native-reference-937775 data/cluster-evidence/
~~~

That example bundle is already copied. It illustrates recovery on a fresh clone.

## Queen's Qwen lane: exact paths and request

Real root: /global/scratch/hpc6308/plumb.
Exported SCRATCH=/scratch/hpc6308 is stale/wrong; do not use it.
Python: /global/scratch/hpc6308/plumb/venv-qwen-tf517/bin/python.
The handoff's older shorthand "venv-qwen" means this exact directory.

Inside a NEW window in REMOTE tmux plumb, a request that worked:

~~~bash
salloc --account=def-hpcg6234_gpu --qos=normal --nodes=1 --ntasks=1 --gres=gpu:a100:1 --cpus-per-gpu=8 --mem=64G --time=2:00:00 --job-name=plumb-qwen-judge
~~~

Omit --partition: the site's plugin routes the job. Check for an existing
allocation first. After grant, verify hostname, SLURM_JOB_ID and
CUDA_VISIBLE_DEVICES. If still on login, enter a compute step using srun under
that allocation; an salloc environment on login is not itself a GPU shell.

Required modules: StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17.
Set HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 and PYTHONNOUSERSITE=1.
Prepend the frozen release to PYTHONPATH while preserving its module value.

Historical successful format-v2 release:
c09228efefb25cc5edbfe2639079dd66c9b3bced62830e0feb427e2687b96f54.
Use its cluster/policy_smoke.py directly, or freeze NEW source after edits.
The helper cluster/cac/run_qwen_smoke.sh defaults to mutable source/; do not
mistake that for a frozen release. Put --root before the judge subcommand.

For vendor clips use --diagnostic, --reference-role scene, actual task/rubric,
five logged seeds (previously101..105), and a NEW --report filename. CLI --help
lists all arguments. Never relabel a scene reference as a goal image or force
the judge to produce agreement. Preserve raw unknown/invalid samples.

## Environment and data safety

- Downloads/installations run on LOGIN nodes into shared scratch; compute has
  no internet. All inference uses local-only/offline assets.
- Warn before additional large downloads. Do not copy model weights to/from
  the laptop; use the existing cluster assets.
- Do not overwrite module PYTHONPATH. IRASim specifically requires
  opencv/4.11.0 from the module system, not a public pip OpenCV wheel.
- Cosmos needs its Transformers5/Diffusers0.41.dev environment. OpenVLA needs
  Transformers4.40.1. IRASim uses a different Diffusers0.24 environment.
  Full pins and model hashes are in HANDOFF.md.
- Never torch.load(weights_only=False). The IRASim safe checkpoint is already
  converted and strict-verified; use it.
- The private Git repo excludes local data/, weights, environments and secrets.
  Kevin's GitHub write access does not imply cluster access; never share keys.
- Avoid shell variables named HOME, PATH or path (path is special in zsh).
  Use task-specific names and the existing PLUMB_ROOT convention.
