# PLUMB execution handoff — implementation in progress

## REAL live steering and New evaluation verified — 2026-09-19 evening

The earlier disabled-controls state is superseded for the main demo. Open
`http://127.0.0.1:8787/console` → **New evaluation** → OpenVLA or **Steer
manually**. Each gallery tile also starts a **New interactive branch** from
its verified final image. This is a fresh IRASim RGB-reconditioned branch, not
exact restoration of the recording's former hidden world state. Original media
is preserved. Saved live evaluations appear on Console and Saved runs; older
synthetic engineering results are collapsed separately on Saved runs.

**Actual H100 backend is running:** Slurm `941263`, `trig0001`, 1 H100/24 CPUs,
bounded 2 hours, expires **2026-09-19 21:40:51 Toronto time**. Worker startup
loaded OpenVLA in 86.45 s and IRASim in 6.68 s. No model downloads or new training.
This is the Trillium live world runtime, not the separate Baseten policy-only
diagnostic; that deployment remains unchanged. Do not silently extend GPU time.

Worker immutable release:
`55d88899e254968dbbb1a5046fb338be7d4e568093a9b842c5ebd39355834626`.
LOCAL tmux `plumb-live-gpu` forwards `127.0.0.1:8917` through the existing
Trillium SSH master to `trig0001:8917`; never kill that shared master.
LOCAL `plumb-live` now runs API PID 57490 at verification time (recheck) with
the old pinned Baseten arguments plus `--live-demo-url http://127.0.0.1:8917`
and `--live-demo-token-file data/private/live-demo-t1xfYe/worker.token`.
Token is private/gitignored and never browser-delivered or logged.

Actual browser proof, persisted in `data/live-integrated/live-demo/`:

- Manual `b310ffb9b6984117b181dcb11df30a73`: 3 generated frames; next input is
  prior output by exact PNG hash. Consecutive displayed commands took 3.12/3.23 s.
- Policy `269c23a9c6d64fd8b70a44028af0a3ca`: 4 real OpenVLA + IRASim steps,
  10.06 s to displayed completion; UI observed 0→1→2→3→4; saved 5-frame MP4.
- Gallery/mobile `722b5b6e98f74897b495f0c4b95747ef`: source final-image branch,
  one real manual step, matched source/input/output provenance; no overflow or
  JS errors; close restored focus. First failed extraction `4fbba6...` retained.

New `plumb/live_demo.py` journals sessions/jobs/commands in SQLite and saves
immutable PNGs/actions/model/timing/hash lineage. HTTP worker is stateless and
Bearer-authenticated. Each policy tick obtains a fresh action from its new
observation. Manual timing `policy:null` is preserved; no policy call is made.
Commands serialize and use idempotency keys; ambiguous failures are not retried.
The current live preview displays the newest PNG; MP4 is a separate download,
not appended to the frame list. Final-frame extraction fully decodes the short
clip instead of seeking past its final 5 FPS presentation timestamp.

Full validation: **1,379 Python passed / 6 skipped; 272 frontend passed; build
passed**, plus actual desktop/mobile GPU browser flows. Saved records survive
the API restart. Existing Pillow/Vite warnings remain. No judge scores or gate
passes are claimed. This is experimental image feedback; world fidelity and
the other policy runtimes remain unfinished. Do not call it a full comparison.

Read [docs/LIVE-DEMO.md](docs/LIVE-DEMO.md) for actual timings, startup/renewal,
source/asset identity, limits, and the current simple product flow. Current
changes are local; no push was performed in this feature turn.

## Gallery “Take the controls” integration — 2026-09-19

Every gallery tile now opens a source-specific control dialog from `/console`.
Opening it pauses the grid, previews the unchanged original recording, and
checks `GET /api/freeplay/status?video_id=...` without invoking inference.
Unavailable controls explain why; technical details are collapsed. Keyboard
focus returns on close, video-seeking keys do not steer, and closing/failure
drops queued commands. Stop acknowledgements do not count as generated chunks.

**Actual live steering remains unavailable for current recordings.** They lack
resumable world-state checkpoints, and the deployed cloud MVP is policy-only.
Do not describe the new UI/contract as a working interactive GPU rollout.

The API requires exact video ID/hash, verifies optional checkpoint manifests,
and only dispatches through an explicitly configured source-branch adapter.
Sessions lock source/mode/checkpoint/adapter identity; overlapping commands
reject and ambiguous source failures quarantine the session. Unsupported
recordings never fall back to generic task-start free-play. The old developer
path is retained and explicitly distinguished.

Verification: 1,368 Python tests passed / 6 skipped; 270 frontend tests passed;
typecheck/build passed. Browser desktop/mobile checks verified source preview,
disabled controls, gallery pause, focus restoration, no JS errors, and no
inference POSTs on entry/keys/close. A separate localhost negative API check
returned 409 before generation for the current unsupported source. Existing
Pillow deprecation and Vite bundle-size warnings remain.

API restarted with the same pinned Baseten diagnostic command in LOCAL tmux
`plumb-live`; PID 50994 at verification time (recheck before any restart).
No GPU job, cloud deployment, checkpoint fabrication or qualification change.
Read [docs/INTERACTIVE-CONTROLS.md](docs/INTERACTIVE-CONTROLS.md) for the contract
and the remaining concrete runtime/checkpoint work. Changes are local; no push
was performed for this feature turn.

## Media-first console implemented — 2026-09-19

Open `http://127.0.0.1:8787/console` and refresh. The built frontend now opens
on a responsive six-recording video gallery, not the qualification wall.
Only **Video gallery** and **Saved runs** are primary navigation; **Tools &
validation** contains Recording archive, Developer tools, Validation,
Compute & cost, and Review clips. Existing URLs remain valid. Published
reference percentages moved into a collapsed Validation section, explicitly
identified as reference-study results. Run selection now precedes the scoreboard.

The gallery reads the existing hash-checked catalog, excludes two-frame probes
and duplicate media, and loops visible clips muted. Pause all, offscreen and
background pausing, reduced-motion behavior, failure/retry states and downloads
are implemented. It shows saved experiments, NOT six authentic policy results.
Exact conditioning is still in source reports. No new render, poster pipeline,
live world-video route, interpolation, judge training or gate changes occurred.

Verification: 263 frontend tests passed, typecheck/build passed. Real Chromium
loaded/played all six clips and paused them; desktop and mobile navigation
passed; 390 px had no horizontal overflow; reduced-motion stayed paused;
zero JS errors or non-GET requests during checks. Vite's existing >500 kB chunk
warning remains. The existing API serves the rebuilt assets without restart.
Python backend code was unchanged and its full suite was not rerun this turn.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the deep, current breakdown
of UI, control plane, ML paths, persistence, deployment and remaining work.
[docs/DEMO-RESTRUCTURE-PLAN.md](docs/DEMO-RESTRUCTURE-PLAN.md) remains the broader
plan; the matched-policy generation and quality phases are still outstanding.

## World-model videos delivered to Downloads and rollout viewport

The unchanged Cosmos3 MP4 is in Downloads as
`Kosmos-Cosmos3-generated-937575.mp4` with a provenance sidecar. It is a real
previously generated3.4s/640×480 clip, not a new render or a synthetic fixture.
All37 existing generated MP4s now appear in the default World-model videos tab
of `/live#world-model-videos`; Play all is a63.8s playback queue. The previous
synthetic/episode grid is retained under Run frames, not relabelled or deleted.
Catalog `/api/world-videos` verifies output hashes and excludes source footage.
All37 fully decoded and played through in a real browser;70 media/report range
reads passed, zero JS errors/POSTs, mobile390px no overflow. No GPU job launched.
Tests:1360 Python passed/6skipped;256 frontend passed/build after the latest
landing-page merge (abd70ec preserves origin/main06bbf55 and the video queue).
See docs/WORLD-VIDEO-PLAYBACK.md. Model generation/scoring gates remain unchanged.

## Baseten diagnostic MVP verified — localhost E2E is live

Open http://127.0.0.1:8787/live#cloud-diagnostic and use **Run cloud model**.
The Kosmos landing/frontend changes were merged through origin/main2d41a65;
merge01ee13e preserves both the latest frontend and this ML/backend work.

Actual Baseten model **3mzlenow**, team **33/q8grpdw**, profile **plumb-api**:
deployment **q929yoj** (`mvp-gcbc-v6-isolated`) is the tested production
environment target. Its HTTP/Truss process is isolated from a persistent ML
venv child (JAX0.4.20/Flax0.7.5/TF2.15.0/protobuf4.25.8/CUDA12.2/cuDNN8.9).
The cloud response verifies NVIDIA H10080GB, pinned checkpoint/source hashes,
and strict changed parameter restore. No CPU fallback or judge adapter is used.

Verified real requests, with raw evidence under data/live-integrated/cloud-diagnostics/:
- cloud-e2dd4f395307440bae93300840adc632: real browser request, finite7-D action,
  15.381s client request /13.369s first model invocation (includes compilation).
- cloud-repeat-20260919-01: exact warm-repeat action,0.565s client /0.0368s model.
- cloud-recovery-20260919-01: succeeds after intentionally malformed cloud PNG
  request was rejected; same loaded worker.0.00883s model invocation.
- cloud-cold-20260919-01: verified wake from SCALED_TO_ZERO/0replicas;
  31.900s client /14.869s load /2.769s invocation. Cold-worker physical action
  differs by at most5.59e-7 from the first worker; no bitwise cold-repeat claim.

Actual API process restart, browser reload, saved-report readback, duplicate-ID
idempotency, invalid local POST rejection, and desktop/390px mobile checks all
passed; no JS errors or horizontal overflow. Current API PID34177 in LOCAL
tmuxplumb-live (recheck PID before any restart). Startup:
`.venv/bin/python scripts/serve_baseten_mvp.py --model-id 3mzlenow --deployment-id q929yoj --profile plumb-api --timeout-seconds 600`.
Data root is data/live-integrated. Full verification:1356 Python passed/6skipped;
255 frontend passed, typecheck/build passed. Existing npm moderate advisories
and >500kB bundle warning remain follow-up work, not hidden test failures.

Deployment max1/min0,60s idle scale-down; all earlier attempts are inactive
or failed, not deleted. Evidence/config/failure receipts live in
data/baseten-mvp-evidence/. Billing credits/cost are not established by quota.
Organization SSH is disabled; managed inference did not require enabling it.
Never print/read CLI credential values or change shared quotas/defaults.
Read docs/BASETEN-MVP.md for operation, measurements and failure semantics.

This is a **real cloud policy-action diagnostic**, not a generated video rollout,
success score, calibrated judge, or completed scientific study. Synthetic task
controls now say so explicitly. Gates A–F remain not_run; biased judge adapters
remain disabled. Local durable storage is verified, not exactly-once recovery
of a remote output lost before localhost received it. Preserve this boundary.

## NEW: Baseten Team 33 access configured — MVP deployment priority

Lucas's side conversation explicitly requests getting the full MVP running on
Baseten. This access update supersedes older "no Baseten credentials" notes;
it does not change the existing scientific gates or make a deployment complete.

- Baseten CLI **1.0.0** is installed. API-key profile **`plumb-api`** was
  successfully authenticated. Its credential is stored by the CLI, never in
  repository files. No new key is needed; do not print credential/config contents.
- The authenticated parent workspace is **Hack the North**. The intended
  resource-owning team is **`33`, ID `q8grpdw`**. The separate default team
  named Hack the North (`wljp62q`) has zero H100 quota. This was the apparent
  workspace-lock problem: use the correct team, not another API key.
- Last verified training capacity: Team 33 **H100 limit 4, baseline 4, in use 0**.
  Recheck before launch. Capacity is not billing credit or inference entitlement;
  neither cost coverage nor account-side SSH enablement has been confirmed.
- LOCAL tmux **`b10`** is configured with `BASETEN_PROFILE=plumb-api`.
  Informational shell variables `PLUMB_BASETEN_TEAM_ID=q8grpdw` and
  `PLUMB_BASETEN_TEAM_NAME=33` do NOT automatically select a Baseten team.
  Explicitly use **`--team q8grpdw`** on project/workstation creation.
  Outside that shell, explicitly use **`--profile plumb-api`**; the previous
  global OAuth/default profile was intentionally retained.
- `baseten ssh setup --profile plumb-api` generated the Baseten SSH keypair and
  managed config block, pinned to this profile. Other SSH routes were preserved.
  No Baseten GPU job/deployment was created and no live SSH connection verified.
  Inspect the `b10` pane before sending commands; do not interrupt user commands.

Start with read-only `baseten whoami --profile plumb-api`,
`baseten org team describe --profile plumb-api --team-id q8grpdw`, and
`baseten train capacity describe --profile plumb-api`. Confirm existing
Team 33 resources and product/credit limits before creating anything; do not
modify shared quotas or unrelated teams' jobs.

Then follow `docs/BASETEN-TEAM33-MVP-TODO.md` and
`deploy/baseten/DEPLOY.md`: adapt the existing cloud scaffold into a bounded
MVP deployment, establish a real model request, durable results, and dashboard
integration. A GPU shell alone is not the requested deliverable. Start small;
the reported capacity does not justify pushing every GPU Chainlet at once.
Revalidate the cloud runtime rather than copying Alliance wheel assumptions.
Keep the biased judge adapters disabled and label diagnostic/world-fidelity
limitations honestly. Do not wait for more weak-label training to wire the MVP.

Final access note: after successful job940190 and an empty scheduler check,
the final Trillium BatchMode check returned permission denied. Authentication
has expired again; no new job was submitted afterward. Do not retry Duo while
the user is away. Completed evidence is already mirrored. SuSIE implementation
and source/results documentation are pushed in `892e221`.
The final artifact-route hardening denies reserved private directories, even
through symlink aliases, when a broader data root is served. The review API also
rejects null/non-string reviewer identities instead of coercing them into names.

## Current judge format-only pilot — adapter disabled

Job 939998 trained a **structure-only** JSON-format adapter on 12 development
train rows for 24 optimizer steps; all semantic target tokens were masked. Its
4-row development syntax loss changed 1.71370849→0.02586903. Reload compare
job 940000 changed bare JSON 0/4→4/4 while schema validity remained 4/4 for both
base and adapter, but semantic tuples drifted on 4/4 rows. The adapter is
therefore disabled: this is no semantic, human-quality, calibration, Gate D,
or production-scoring result. Reports are local under
`data/live-integrated/cluster-evidence/judge-format-{pilot-939998,compare-940000}/report.json`;
cluster artifacts are `/scratch/lchow432/plumb/experiments/judge-format-only-v1`
and `evidence/judge-format-only-v1-comparison.json`, source release
`f6b30d952be1b2e9cccf4c5915a4e85a636b22b620634e485dfde0ab77c08fc5`.
No new weights were downloaded; adapters remain cluster-only. Root verified
live `/review` at `http://127.0.0.1:8787/review` in isolated Chromium QA with a
temporary DB: 16 clips/images decoded, drafts saved/reloaded/resumed, no JS
errors, and no 390px horizontal overflow. Production review DB is absent and no
human labels exist; the isolated `model_assisted` test draft is not human data.
App tmux `plumb-live` PID19103 now exposes24 diagnostic cards; all24 report
URLs return200. Health remains `qualified=false`, all gates are
`not_run`, and execution remains synthetic-only. See `docs/JUDGE-FORMAT-PILOT.md`.

SuSIE_LL job940190 **COMPLETED0:0** in20 seconds on one H100, status
`completed_unqualified`. It made two finite `[1,7]` gc_bc calls with exact
reset/repeat on a static vendor-video first/final-frame conditioning fixture;
this is not rollout, task success, Gate A/B, or a primary row. Checkpoint is
cluster-only at `/scratch/lchow432/plumb/models/patreya--gcbc-bridge/checkpoint/checkpoint`, SHA
`80b354...`; publisher README declares MIT (not independently re-licensed).
The correct clean staged source is SOAR
`/scratch/lchow432/plumb/source-soar-gcbc@eabd5f16a856e484884a22e257a941bb358cea08/model_training`,
not the wrong `bc60...` upstream source. Restore was strict inference-only
params-only (`target_params=None`; optimizer state excluded), not training
resume; failed full restore940180 and failures940172/940173/940174 are preserved
in local report directories. Evidence: local
`data/live-integrated/cluster-evidence/susie-ll-gcbc-940190/report.json`, remote
`evidence/susie-ll-gcbc-static-goal-v5.json`, release
`fdc52dcb09cebae2bdcd3d213a4865aa28c43f87be425f483a456a01afca44ea`.
No adapter promotion follows from this smoke. Latest checks: Python1267 passed,
6 skipped; frontend144 passed and build passed. Scheduler is empty.

## Access restored; actual development-review packets ready

User reauthenticated Trillium. BatchMode SSH works again; latest scheduler
check showed no user jobs running. No completed GPU experiment was repeated.
The full v2 raw bundle is now mirrored at
`data/judge-teacher-v2-evidence/judge-teacher-v2-diagnostic/`.
The exporter was run against actual mirrored evidence and wrote
`data/judge-teacher-v2-evidence/export-audit-939753.json`:0 accepted train,
0 accepted validation, no dataset created. Its expected blocked exit2 is not
a tooling failure; do not relax the label rule.

The23MB source review media were mirrored without model weights. Actual16
opaque, blank, unassigned development-review packets are now published:
`data/live-integrated/review/pilot-review-v1/INDEX.md` and
`data/live-integrated/review/pilot-review-v1/blank-worksheets.jsonl`.
All272 localhost video/frame URLs returned200. Every label field remains blank.
Coordinator-only resolver: `data/private/judge-pilot-review-v1-resolver.json`,
mode0600, outside the served root; never give it to reviewers. Byte hashes were
verified before publication. Source mirror: `data/judge-pilot-v1-review-source`.

Next dependency is actual human development review, not SSH/setup. Obtain real
reviewer identities/independence before recording labels; no names or labels
have been invented. These16 already-used clips are not fresh held-out Gate-D
data. The existing trained adapter remains experimental and disabled for
production scoring. Older SSH-blocker/packets-pending notes below are history.

## Latest execution boundary — v2 completed, SSH reauthentication needed

The actual v1 adapter is trained and its saved reload is verified. Teacher-v2
job939753 subsequently reported **75/80 schema-valid samples,0/16 accepted
three-vote modes,16 abstentions**, status completed_unqualified. No v2 training
was started and the acceptance rule was not weakened. Source release948d59...
and raw reports remain on Trillium; the complete v2 report bundle is not yet
mirrored locally (summary was observed in SSH stdout).

**Trillium SSH expired**: control socket is absent and LOCAL drac:0.0 is back
at a laptop zsh prompt. Other cluster sessions were left untouched. One normal
SSH re-login/Duo push was attempted; it timed out, then that owned PTY was
cancelled. Do not repeatedly send pushes or delete sockets. User must run
`ssh trillium-gpu` and complete Duo locally to restore transfers/remote work.
An asynchronous request for reauthentication is outstanding.

Local work continued: v2 exporter recomputes raw selection and refuses weak,
homogeneous or insufficient data; actual current v2 export should produce only
a blocked audit. The development-review packet builder verifies original
video/PNG hashes, uses opaque IDs and existing artifact URLs, leaves every
worksheet blank/unassigned, and keeps a0600 coordinator resolver outside the
served root. **No real review packets exist yet**: full source media still
needs mirroring after reauth. No human identities/labels were fabricated.

See `docs/JUDGE-REVIEW-NEXT.md` for exact next commands. Latest local full suite:
**1,229 passed,6skipped**. App8787 exposes17 diagnostics including the actual
trained pilot and saved-adapter comparison; both report links were checked200.
All qualification gates remain unchanged; no Baseten submission/deployment.

## Actual judge pilot trained and reloaded — 2026-09-19 afternoon

**Actual optimizer training is now complete.** User approved the unqualified
pilot; do not revert to approval-pending/preflight-only instructions below.
Teacher939423, training939451 and adapter-reload939458 all COMPLETED0:0.
Two epochs/8 optimizer steps on4 accepted training clips;3 development clips.
Dev loss0.7981337→0.6012993→0.3868534. Selected saved adapter epoch02 tree SHA
7a0b7796a0ac8b78fbb8ba8f4c4db01fe8d35c2edc3cb7a0fbeec9f4fb20523e.

**Do NOT scale or deploy this v1 judge.** All7 accepted labels came from a
single valid vote (sample2), all artifact/visible/5/met; no3/5 teacher quorum.
The reloaded adapter repeats that tuple onall3 dev clips, while the base says
intact. This is observed label/selection bias, not improved human accuracy.
See `docs/JUDGE-PILOT-V1.md` for complete measured results, artifact paths,
source releases, preserved failure and raw-summary counter caveat.

Work continues on a separate experimental teacher-v2 **prompt/selection
diagnostic**, not more v1 training: explicit integrity/collision definitions,
exact enum consistency reminder, require unique semantic mode≥3/5. It must
not alter the global primary judge, repair old responses, or fill gates.
The user asked to keep going; next is running the bounded v2 diagnostic on
the existing16 developmental inputs and reviewing actual schema/agreement.

Teacher-v2 diagnostic job939753 was submitted from immutable source release
948d59d53df2e7ce644ba2177702d00fe89854a33b258e99acecd99cb46d1151.
Output namespace: `/scratch/lchow432/plumb/experiments/judge-teacher-v2-diagnostic`.
Same base model/runtime/16 source clips; it is NOT the v1 trained adapter.
It produces raw reports and summary only, never training JSONL or gate records.
Recheck scheduler and per-clip reports before resuming; all v1 artifacts remain
untouched. Latest local full suite:1,221passed/6skipped.

## Approved experimental judge pilot — acquisition/execution history

The user explicitly approved the unqualified teacher-labelled development
pilot and instructed continued execution. The earlier approval-pending text
below is superseded. Do not ask the same permission again.

Frozen selection:16 distinct real Bridge close-drawer episodes,12 train and4
development validation, from dataset revision0e9d76d07e9df3ea3eba257b2520d4913833fad2.
Selection SHA18717375576b9fb3e08606a7e09b30d73f2dd731b167b3e99c0866f122cf5bf8;
local `data/judge-pilot-v1-selection.json`, remote `fixtures/judge-pilot-v1-selection.json`.
All16 source lineages, including later abstentions, remain excluded from future
formal study/calibration. This pilot does NOT replace the full6×5×50 scope.

Remote prepared data:
`/scratch/lchow432/plumb/experiments/judge-lora-pilot-v1-prepared`.
Each clip has16 distinct lossless RGB PNGs sampled from its actual video,
Parquet nominal timestamps, and initial scene context (NOT a fabricated goal).
No robot-state/gripper conversion occurs. Original source receipts and media
are preserved. Only2,666,878 bytes of selected video/Parquet were acquired;
no model weights were downloaded.

Acquisition caught a real large-Hub-listing bug: repo_info.siblings omitted12
selected videos. `cluster/download_assets.py` now queries exact explicit paths
or a paginated tree, checks the full byte ceiling, and writes no-clobber named
receipts. Old20-file receipt `evidence/IPEC-COMMUNITY--bridge_orig_lerobot-download.json`
and partial prepared directory `experiments/judge-lora-pilot-v1` are preserved.
Complete32-file receipt: `evidence/judge-pilot-v1-source-complete-download.json`.

Teacher job939398 failed before model inference because a new collector expected
model_id rather than the existing download manifest's repo field. It is fixed
by reusing the same full file/hash/revision validator as the training path;
the failed job/log and immutable release80a737fb... are preserved.
Teacher retry939423 later COMPLETED; the earlier running snapshot is historical. Its immutable release is
`cacf20c6120b4511c6bb5a2fe76c249dea2842b61d538bcfec26a87b59f1832d`.
It uses existing Qwen7B weights and venv-judge-lora-tf449, one H100 debug15min.
Check actual `squeue`/`sacct` and rawteacher per-clip files before any retry.

Next actions: finish teacher collection (resumable completed hash-matched clips),
validate finalinputs/splits, then actually run `cluster/judge_lora_pilot.sbatch`
with `cluster/judge_lora_pilot_config.json`, a NEW outputdir and the pinned model
view. Two manual AdamW epochs, r64/alpha32/lr1e-4/batch1/seed20260919; every real
optimizer step/checkpoint/loss is recorded. Then run the saved-adapter reload
comparison on the originally assigned dev-validation clips only. No Gate-D/E,
human accuracy, formal search, primary score, or calibrated-judge claim.

## Judge fine-tuning priority — 2026-09-19 afternoon

User explicitly requested judge fine-tuning ASAP. **Actual optimizer training
has NOT started; no adapter is saved.** One-H100 Trillium preflight939218
COMPLETED0:0 with real Qwen forward/backward,20,185,088 trainable LoRA parameters,
finite loss0.6149674654,56 nonzero-gradient tensors and31,381,548,544B peak CUDA
allocation. This is not teacher calibration or a trained-judge claim.
See the new first section of `docs/DISTILLATION_RUNBOOK.md` for source release,
runtime lock, exact module/env/model paths, timing scope and fixture limitations.

Next meaningful choice: user was asked asynchronously whether to start an
explicitly unqualified development pilot using newly teacher-labelled clips,
or wait for calibrated training data. No answer yet. Do not fabricate a Gate-D
pass, formal preregistration, human labels, training splits, or qualified base
judge to start formal training. The existing single diagnostic has no quorum;
it is framework-preflight-only. Preserve this explicit boundary.

Final verification this continuation: **1,191 passed,6skipped**. Live API8787
was restarted in LOCAL tmux `plumb-live` (PID95884 at restart), still using
`data/live-integrated`; no active run was interrupted. It exposes15 diagnostic
cards, including the new preflight report, whose HTTP link returns200.
Real submission stays disabled and qualified=false. Trillium `squeue` is empty;
939218 is complete and no training/monitor process is left running.
Video-curation safeguards bound the reconstructed pixels and retained original
raw media/reports; no generated human labels or substituted goal images.

Other implemented work this continuation:

- Real opt-in S3-compatible result persistence/readback for the Chain, conditional
  immutable writes, secret-free pre-POST result keys, deployment-context secrets,
  request/result identity/digest checks and bounded reads. Ambiguous POSTs can
  recover by result key without a request ID. No store was provisioned or called;
  no Baseten deployment/spend occurred. Missing config blocks before GPU work.
  Actual SDK structure validates. A fresh60-episode simulated rehearsal passed:
 59completed/1injectedfailure/2droppedcallbacks, zero unresolved, gates NOT_RUN;
  `data/rehearsal-run-aakd64k1`, run `run-e070853a3ebf465fa5ac9d8bcc4fb266`.
- Source audit corrected production Octo to AutoEval's static PRNGKey(0) and
  cached task with safe reset/restore. New RNG0v2 profile is distinct from the
  earlier GPU smoke; no v2 GPU result is claimed. Raw LeRobot proprio is not
  silently substituted for the source wrapper's converted history.
- Rehashed staged Cosmos, OpenVLA, Octo and Bridge metadata on the cluster:
 4core staged-integrity passes,0fully-verified deployment profiles. See
  `docs/STAGED-ASSET-AUDIT.md`; no weights copied to laptop.
- Acquired only6MB Bridge metadata, audited53,192 source episodes and exact
  instruction coverage. No basket/sink exact matches; no study starts selected.
  Source gripper extrema0.046–1.112 conflict with the provisional0–0.39 profile;
  no guessed rescaling. See `docs/GATE_C_RUNBOOK.md`.

## Integrated continuation — 2026-09-19 late morning

This section supersedes the branch, process, test and allocation snapshots below.
**Delivered:** merge commit `3832de3` is pushed to origin/main. It preserves both
parents: newer main `634a796` and num2 `dd8a12f`. The normal workspace
`/Users/lucas/Desktop/tax_stuff/htn26` is now on main at this integration.
The integration worktree `/Users/lucas/Desktop/tax_stuff/htn26-integration`
retains the full rehearsal data and separate deploy-validation environment.
The older num2 branch and all diagnostic evidence are preserved.

The merged production backend, protocol/artifact/annotation UI and Training Jobs
path remain intact. Older source-profile adapters are deliberately isolated in
`plumb/policies/diagnostics`; offline distillation preparation is now
`plumb.distillation_preparation`. The two Baseten outboxes remain distinct paths,
not two dispatchers for one episode. Both fail closed on ambiguous POSTs.

Integration fixes include terminal remote failure classification and raw failure
artifact retention; simulated independent result-store recovery for lost webhooks;
exact short terminal action segments; failed-row transport/request provenance;
pre-POST dispatch persistence; strict protocol/asset/profile/human qualification;
and cryptographically verified signed preregistration tags bound to exact remote
tag object IDs. A remote tag does not establish an independent receipt timestamp.
The existing prereg/protocol-v1 tag is unsigned and is not accepted.

Verification: **1,149 Python tests passed, six optional tests skipped**;
**136 frontend tests passed**, TypeScript/build passed; Truss 0.18.30's actual
Chain SDK validator passed. Readiness check: **16 pass, seven pending, zero fail**.
SDK definition validation is not deployment validation.

Full fresh simulated production-path rehearsal:
`data/rehearsal-run-e5wxygzy`, run `run-153c1964ba9849f5bc2638a9d67e14e4`.
All 1,500 POSTs reached terminal state: **1,463 completed / 37 injected failures**;
26 dropped and 37 duplicate webhooks; zero unresolved submissions or unexpected
completion errors. All 1,500 rows preserve simulated provenance and request IDs;
all 1,463 generated results preserve segment/timing fields; 30 analysis cells.
All gates stayed NOT_RUN. Its 38.601s execution and 194.738s publication are
simulation/harness timings, not model/platform throughput or cost measurements.
Run-derived artifact writers passed; real calibration/economics/load-test/scenario
inputs remain absent. The previous failing 1,500-row rehearsal is preserved at
`data/rehearsal-run-nlsttg56` and must not be relabelled a success.

Separate Octo production runtime: source
`241fb3514b7c40957a86d869fecb7c7fc353f540`, new cluster checkout
`source-octo-autoeval241fb`, copied environment `venv-octo-autoeval241fb`.
Old 37951 checkout/environment/evidence remain untouched. New dlimp is pinned to
`5edaa4691567873d495633f2708982b42edf1972`. Runtime lock:
`/scratch/lchow432/plumb/evidence/octo-runtime-autoeval241fb-v1.json`, SHA
`6f156bfa1e220e4573ecad68c993183ee84f1588cd9689ff1882892ad8a8f0bb`.
It records exact package/freeze provenance and the known NumPy1.24.3→1.26.4 /
TF2.15.0→2.15.1 Alliance deviations. Pip check is NOT clean; only that exact
reviewed dlimp TF mismatch is accepted and recorded, unknown mismatches block.
One-H100 job938918 failed before inference due to a harness runtime-field bug;
raw report/log/lock are preserved in `data/live-integrated/cluster-evidence/
octo-production-938918/`. Failure source release is
`ba220f4c214b313d3b828ae82561278485cfbf8d318cc533e3a1eca82f9293c7`.

**Production241fb retry938946 COMPLETED (0:0)**, source release
`5788a3adaa49a6cae88bc7066b514f2a81f19c53c2860a2f7f04e051fb0cc7ba`.
It exercised the actual production adapter, three native calls and exact reset/
repeat with static canonical frame inputs. First call9.1691s includes JIT;
second0.03036s and repeat0.03016s; full diagnostic29.4249s. JAX allocator peak
577,332,736B is not total device memory. H100 UUID and driver are recorded.
Report/log are in `data/live-integrated/cluster-evidence/octo-production-938946/`.
This closes the production API runtime mismatch, not policy qualification:
no certified execute prefix, generated-image feedback, physics or task outcome.
The failed938918 harness is preserved, and its missing-runtime-field bug has
a regression test. No job remains intentionally running after these checks.

Original eight-H100 allocation937277 expired TIMEOUT at09:11 EDT. It is not
running or awaiting cancellation. Model inference remains scheduler-only, with
bounded one-H100 jobs; all weights remain on clusters.

**Live app:** http://127.0.0.1:8787 in LOCAL tmux `plumb-live`, PID89195 at
cutover. It runs the merged source from the normal workspace and uses
`/Users/lucas/Desktop/tax_stuff/htn26/data/live-integrated`, a copied/migrated
data directory. The original data directory's database and evidence remain
untouched; the old idle API65815 was gracefully stopped after checking no active
runs. Health, all19 report/media URLs (14 diagnostics/5 videos), and blocked real
backend were verified. Both938918failure and938946success are visible, unscored.
No live Baseten submission or cloud spending was enabled. Frontend production
dependency audit reports zero vulnerabilities; three moderate dev-dependency
advisories remain, with no forced dependency upgrades applied.

To restart this exact local service after inspecting the current listener:

```bash
cd /Users/lucas/Desktop/tax_stuff/htn26
PLUMB_DATA_DIR=/Users/lucas/Desktop/tax_stuff/htn26/data/live-integrated \
  .venv/bin/python -m uvicorn plumb.api:create_app --factory --host 127.0.0.1 --port 8787
```

Tests were run in the integration worktree with the normal workspace's .venv.
The Chain SDK validation venv is `.venv-deploy` in the integration worktree;
it is excluded from git. Source staging is refreshed with
`python deploy/baseten/stage_packages.py`. Never upload model/data directories
as deployment source. No automatic long-running monitor remains after delivery.

Still not done: acceptable causal world-model fidelity; remaining source-native
policy/state/gripper/execution-prefix certification; real matched panels and
independent humans; signed/reviewed preregistration; actual Baseten credentials,
object store, deployment images/account lifecycle/capacity/pricing; qualified
full study, fresh human recalibration after judge distillation, and three real
1,500-episode cost/latency rehearsals. No fabricated labels or successful gates.
WorldGym research found a causal single-action candidate but no pinned/license-
verified weight package or approved gated SD3 support; no download/adoption or
world-model training was performed.

## Num2 continuation — 2026-09-19 06:09 EDT (supersedes the older snapshot below)

**Morning entrypoint:** source implementation and bounded GPU checks are done
for this continuation. Read this section first, then the workflow documents as
needed. The full scientific project is not complete. Local API runs in tmux
`plumb-api`; source code is separate from ignored local/cluster evidence.

**Parallel-main update:** the tested continuation is preserved on branch
`num2-verified-continuation` (implementation commit `f5eddee`). During execution
Kevin advanced `origin/main` to `6625b17` with another substantial build.
No force-push or automatic merge was performed. Read
[INTEGRATION-HANDOFF.md](INTEGRATION-HANDOFF.md) before combining them:14 paths
have conflicts, and their interfaces need deliberate integration. Test results
here apply to the continuation branch, not the unreviewed combined system.

The continuation implements additional control-plane and study tooling. The
full scientific study is still blocked; no primary result, physical-fidelity,
Baseten deployment, or speed/cost claim is established. All prior raw evidence
is preserved. Source at the start of this continuation was commit `425a6dd`.

Completed code tracks:

- Durable Baseten outbox, one-POST ambiguity handling, exact raw signed callback
  storage, callback-before-ack association, leased finalization, cancellation,
  and local reconciliation. Fixture workers cannot claim/reclaim cloud-owned
  episodes. The API exposes optional signed callback ingress and a sanitized
  outbox view; it never submits cloud work automatically. See
  `deploy/baseten/README.md`. Actual deployed serializers, independent object
  storage and account-tested Chain lifecycle routes remain prerequisites.
- Safe scenario import checks explicit real-robot provenance, local image/state/
  goal hashes, finite 8-D states and disjoint source lineages. Calibration
  selection is frozen before CLI export/report; packets use opaque identifiers
  and a private resolver. Two distinct human owner attestations are required.
  See `SCENARIO-CALIBRATION-WORKFLOW.md`; use `plumb scenarios --help` and
  `plumb annotation --help`. No actual panels or annotations were invented.
- Full-study planning and run-specific episode materialization retain all
  6 × 5 × 50 slots and exact task horizons. Readiness binds actual gate and
  component revisions. Native cost comparisons preserve each policy's feedback
  boundaries; 1–6-request partition experiments are a separate unqualified
  sensitivity. Rehearsal checks distinguish judging attempts from scientific
  V/1,500 coverage. See `docs/STUDY-WORKFLOW.md` and `plumb study --help`.
- Distillation preparation freezes an 80/20 development-only training/
  validation split with teacher, rubric, sampling, media and raw-output
  bindings. It rejects held-out/primary/cost lineage leakage and prepares a
  no-submit Training Jobs readiness record. A distilled revision still needs
  fresh held-out humans and paired-video evidence. See
  `docs/DISTILLATION-WORKFLOW.md` and `plumb distillation --help`; no training
  job, dataset upload or cloud spend occurred.
- Source-backed policy candidates now cover Octo, MiniVLA, SuSIE/SuSIE_LL,
  and OpenPiZero.
  MiniVLA remains blocked by the unlicensed VQ dependency and unbound auxiliary
  vision/language assets. Its pinned tokenizer exposes one action, despite
  seven-future-action latent configuration; do not advertise a seven-action
  proposal. SuSIE `gc_bc` replication and `gc_ddpm_bc` sensitivity have distinct
  identities; corrected-arm weights remain missing. See
  `docs/POLICY-DEPENDENCIES.md`. Unit fixtures do not certify live loaders.
  OpenPiZero retains its four-row proposal and requires an explicitly tagged,
  unit WXYZ quaternion pose; canonical Euler state is not silently accepted.
  Its external execution wrapper/prefix, state conversion, checksum-bound
  PaliGemma support assets and checkpoint remain unqualified/unavailable. See
  `docs/OPENPI-DEPENDENCIES.md`.

New actual GPU evidence:

- Trillium H100 job **937822**, immutable source release
  `a763859e977ce3a6baf6fe0f83f63137fa2cf6229921e11d021e0785b8f7ba83`,
  completed successfully. It replays the saved 16 actions with a growing window
  of up to 15 committed past latents, producing one new frame per request with
  no future actions. This changes inference conditioning from training mask=1;
  it is not a policy re-query run or qualification.
- Runtime **187.265 seconds** includes loading, 16 ticks, their repeat, and
  artifacts. All 16 pixel and final-latent hashes repeat exactly. Tick zero
  also matches the earlier 937751 latent-carry replay. The inspected final
  frame still has substantial blur and gripper/object distortion, so history
  conditioning did **not** resolve fidelity. Different horizon/noise shapes
  prevent treating cross-profile differences as a history-only causal effect.
- The whole small bundle is mirrored at
  `data/cluster-evidence/irasim-history-937822/`; original cluster location is
  `/scratch/lchow432/plumb/evidence/irasim-history-937822/`. The API now exposes
  its digest-checked video as an unscored diagnostic. Do not resubmit this run.

Octo staging: Trillium now has the pinned MIT Octo-Small snapshot at
`models/rail-berkeley--octo-small`, revision
`03d88976c54a58e10480d2043a8c762b35bc2611`, checkpoint SHA
`590df097f8a37bbc1c3aac2a488c0fb08e72bae8abaedfb89f0685677d848962`.
Only the Apache-2.0 T5 config/tokenizer files were needed, not T5 model weights:
the Octo checkpoint already restores encoder parameters. They are cached under
`models/.hf-octo`, T5 revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`.
The clean `source-octo` checkout pins
`37951e4e6d708fd76374f6e09e716763fe2673b1`. See
`cluster/prepare_octo_assets.py` and `cluster/setup_octo_runtime.sh`.

### Octo runtime and actual GPU results

- Isolated Trillium `venv-octo`: Python3.10, Alliance JAX0.4.20 / CUDA12 JAXlib,
  Flax0.7.5, Orbax0.4.3, TF2.15.1, exact pure-Python TFP0.23. TF2.15.1 is a
  recorded compatibility deviation from the source lead2.15.0. Load modules
  `StdEnv/2023 python/3.10 cuda/12.2 cudnn/8.9.5.29`; do not mix it with the
  Python3.11 IRASim/OpenVLA environments. `pip check` and offline T5 imports pass.
- Immutable runtime lock:
  `/scratch/lchow432/plumb/evidence/octo-runtime-cuda12-jax0420-v2.json`, SHA
  `5c10b8881e8ab97830ee84f565aeebe1cb689f1d161359a64a2ca450a62e74b9`.
  Version1 is preserved separately. The v2 lock includes the actual package
  inventory; no public CUDA wheel was installed. Assets total about550MB;
  no full T5 weight download was needed.
- Canonical lossless PNGs and provenance are in
  `fixtures/octo-small-bridge-cv2-linear-rgb-256-v3/` on Trillium. The helper
  records BGR→RGB decoding, source-compatible OpenCV linear resize, original
  MP4 hash, source RGB/BGR hashes, and output file/pixel hashes. Source frame
  indices are **0 and2** because0/1 decoded identically. Physical control
  timestamps remain unknown. Manifest SHA:
  `996bdb5294d27b5d0245aa93f61272364f0f1300dd386abb7eb5aa42ae5e34c1`.
  This is a declared diagnostic input transform, not camera-pipeline parity.
- Allocation937277 step0 failed after checkpoint loading: the pinned Octo
  v0.1 API does not accept AutoEval's newer `unnormalization_statistics`
  keyword. Preserve `data/cluster-evidence/octo-smoke-937277-step0/` and release
  `7342b517935f7054db57b84b8e975a9bda816dc54c5537267405b7a3da1965ff`.
  Failure report SHA is
  `d91bd5a05357a865a40d52366652071c99426056c67706bce61deedc02a4857b`.
- Corrected native-v0.1 profile calls the actual source API and unnormalizes
  `actions * std + mean` on the native JAX array before host collection, as
  the pinned source notebook does. This is explicitly **not** a verified
  substitute for the newer AutoEval server profile. Its source history-mask
  behavior is preserved; the code does not silently repair upstream semantics.
- **937277.1 completed** from release
  `b90de29b065e81586c3479301780acbeee63bc118cb08ab6bcfefd6c2c0d3645`:
  three real native calls, each with a finite4×7 proposal; reset/repeat exact.
  First call9.4596s includes JIT, subsequent calls0.03633/0.03674s;
  total38.6858s includes loading/setup. JAX-reported peak577,332,736B is an
  allocator measurement, not total device memory. Bundle:
  `data/cluster-evidence/octo-smoke-937277-step1/`.
- **937277.2 completed**: eight independent worker processes across
  trig0010/trig0030, one visible allocated H100 each,24 native calls total.
  Every worker reset/repeat was exact; both observations' proposals and
  selected actions matched across all workers. Scheduler step elapsed28s;
  worker totals23.52–24.85s. This is **not** a rollout/burst/cost result.
  Immutable release:
  `46c353e2786fd1b2dd86fc72223d26af716d2691df515082d5cfa0658c1b8392`.
  All eight raw reports/logs and a hash-bound comparison summary are in
  `data/cluster-evidence/octo-repro8-937277-step2/`. Physical GPU UUIDs were
  not recorded; logical CUDA ordinal0 is per worker, not a global GPU identity.
- All these diagnostics remain unqualified: no generated-image feedback,
  primary task outcome, real-robot parity, or full AutoEval replication claim.

All five SSH routes were rechecked successfully during this continuation.
The historical eight-H100 request **937277 started at05:11 EDT** on two nodes
and is scheduled to end at09:11 EDT. Its bounded GPU checks are complete.
Cancellation permission was requested asynchronously after the checks because
the prior handoff explicitly withheld it; absent a reply, the parent allocation
is preserved. Recheck live state before touching it. No duplicate allocation
was submitted. The site rejects `srun` from a login node even with `--jobid`:
run a new bounded step only from the inspected idle allocated compute shell
inside REMOTE `drac:0.0`. Never paste while that pane is running a step.

Root verification: **212 passed, 5 skipped** in the lightweight
local environment; frontend production build and TypeScript checks passed.
The four original-IRASim tensor checks also passed inside the H100 batch job.
Fresh synthetic integration `run-08d9b32ef23144d086bccc8d89ce5eee` completed
1,500/1,500, exported 1,500 JSONL rows, and returned all30 API analysis cells
with `qualified=false`. Its scratch test data is
`/tmp/plumb-num2-integration.jjbXlK`; its timing is not model performance.

The actual local API was restarted with current source in **LOCAL tmux
`plumb-api:0.0`**, listening at `http://127.0.0.1:8787`. PID at final restart65815;
inspect the pane/listener before any future restart. It exposes12 diagnostic
cards and5 playable digest-checked clips, whose report/media HTTP links were
all checked. Baseten outbox is empty; callback ingress and automatic cloud
submission are disabled. No existing run was active during restart.

### What remains (do not shrink the requested scope)

1. A fidelity-acceptable causal world profile and exact remaining native
   wrapper/state certification. Current IRASim profiles visibly distort; the
   Octo native-v0.1 diagnostic is not certified AutoEval replication. OpenPi's
   executed prefix/quaternion conversion and MiniVLA's support assets/license/
   exposed-action contract remain unresolved.
2. Real matched five-task panels plus disjoint development/calibration/cost
   cohorts,150 selected generated calibration clips, and two actual human
   annotators. The new CLI workflows are usable once those inputs exist.
3. Real Baseten credentials, immutable deployment images/serializers, independent
   object storage, account-tested lifecycle APIs, capacity and pricing; then
   actual full-matrix execution, drift/cost sweeps, held-out confirmation,
   distillation training/fresh calibration, and three complete rehearsals.
4. Newer `origin/main` now contains `BUILD-SPEC.md`, `SCRIPT.md`, and
   `reverse_validation.py`. They are not merged into this tested branch;
   verify provenance and reconcile them with the reviewed spec before treating
   the prior missing-original-material requirement as resolved.

Model weights remain only on clusters. The private source repo deliberately
excludes `data/`, environments and model artifacts. Preserve all failed and
successful reports. Curation guidance was applied to the new fixture path:
reuse existing media processing, preserve originals, and hash-bind every
derived RGB input and transform.

The following older sections retain useful asset/runtime details and historical
evidence. Their process IDs, test counts and live scheduler descriptions are
snapshots, not current guarantees.

---

**For the new context named num2:** also read [CLUSTER-OPERATIONS.md](CLUSTER-OPERATIONS.md)
in full. It contains the verified local/remote tmux layout, exact SSH/Slurm
commands, latest five-cluster capacity check and submission/recovery procedures.
[NUM2-START-HERE.md](NUM2-START-HERE.md) is the paste-ready continuation prompt.

Latest update: all five SSH connections work. ONE H100 is enough for next
experiments; no GPU experiment is currently running. Do not wait for eight H100s.
Single-H100 and single-H200 Trillium scheduler dry-runs estimated immediate
starts at04:11 EDT; recheck before submitting. Old job937277 remains pending
Priority, unchanged. The user now says to use whatever resources are available.

GitHub is private at https://github.com/lsnchow/htn26, main; baseline commit88c1b0d.
kevinvalenciaa's write access is ACTIVE, not merely an invitation.
The exact Queen's environment is venv-qwen-tf517, not the older shorthand below.
No automatic switch into another chat has been performed.

Final access caveat: a later direct Trillium SSH check returned a broken
multiplex pipe/permission denied. The earlier all-five success is a timestamped
snapshot, not a guarantee. See CLUSTER-OPERATIONS.md for recovery; do not infer
that remote jobs or saved evidence were lost.

Updated 2026-09-19, approximately 03:58 EDT. Preserve the full specification.
User authorized Terra xhigh coding agents, cluster execution through drac/drac2/
drac3/drac4/cac, and cluster-side downloads. Warn before any further large download.
Do not copy weights to the laptop. No Baseten deployment/spend has occurred.

## Current state

- Workspace: /Users/lucas/Desktop/tax_stuff/htn26; initially only the two original
  specs existed. Git is initialized on main; private remote is
  https://github.com/lsnchow/htn26. Source/docs/tests only: model weights, run data,
  environments, frontend dependencies/build output, and local smoke databases
  remain excluded. Cluster evidence files referenced here are not in the repo.
- Both AGENT-BUILD-SPEC.md and PASS-OFF.md now distinguish implemented diagnostics
  from qualification targets. README.md has local run instructions.
- Local API: http://127.0.0.1:8787, PID 54550 / tool session 57328 at this update.
  Restart after source edits; do not start duplicate listeners.
- Latest full local suite: 141 passed, 5 skipped (no laptop Torch).
  A Pyflakes undefined-name/syntax test now covers all production Python paths.
  Frontend TypeScript/build passed. baseline-ui skill governs frontend edits.
  Accessibility skill also applied: named evidence links/native video controls,
  visible focus, improved evidence label contrast. No autoplay or animation.
- Real evidence is separate from synthetic episodes in /api/experiments, with
  digest-checked media links. Never attach real smoke clips to fixture episodes.
- Fixture run run-7511654d13c840a996f1d1044832cf98 completed 1,500/1,500 rows / 30
  cells; data/exports/full-matrix.jsonl. Its 4.251s is harness-only fixture timing.
- Fresh post-audit HTTP integration run run-cd12504f64264144ba9b70a361bc559e
  also completed1,500/1,500 with export1,500rows; all real evidence links passed.
- No qualified six-policy study, calibration, physical fidelity, or cost claim.

## Cluster access and allocations

Local tmux sessions contain SSH login shells; returning to a login node does not
mean a batch job stopped. Check scheduler state. Do not disturb unrelated work.

- drac: SSH alias trillium-gpu -> lchow432@trillium-gpu.alliancecan.ca.
  Root /scratch/lchow432/plumb, login trig-login01. Multiplex authentication works.
- drac2: Fir; drac3: Rorqual; drac4: Nibi, all user lchow432.
  Inspected capacity; no PLUMB GPU work started there.
- cac: Queen's, user hpc6308, root /global/scratch/hpc6308/plumb.
  The exported /scratch/hpc6308 value is invalid. Agent cluster_cac owns remote CAC.
- Original Trillium 8-H100 job 937177 ended on laptop disconnect.
  Replacement 937277 remains PENDING Priority, 2 nodes × 4 H100, four hours.
  It is inside remote tmux drac and survives laptop disconnect. Do NOT duplicate it.
- Queen's A100 job 12238810 on frnt191 ended around 03:23:36 EDT.
  Only its CUDA-visible GPU0 was ours. Last Qwen run finished 03:23:15.
  No replacement requested.
- Trillium debug single-H100 jobs launch quickly. Use partition=debug, nodes=1,
  ntasks=1, gpus-per-node=1, cpus-per-task=24, bounded time.
  Do NOT pass --mem (site rejects it; automatic 187.5 GiB per GPU).
- Job 937739 completed the corrected two-seed IRASim action probe,18/18cases.
  Preceding 937728 failed preflight on bare-vs-prefixed SHA serialization, before
  GPU inference. A new release fixes this; old evidence remains unchanged.

GPU nodes have no internet. Downloads/installations use login nodes and shared
scratch; model inference only on scheduler-allocated compute. Preserve module
PYTHONPATH; overwriting it breaks CVMFS packages such as OpenCV.

## Real diagnostic results

All raw reports/small clips are in data/cluster-evidence locally and evidence/
on the owning cluster. These are not scored study episodes.

| Job | Observation | Limitation |
|---|---|---|
| 937372 | Cosmos H100: 16 actions → 17 frames, 4.524s inference, 167.684s load, 36.52GB peak allocation | Runtime only |
| 937398 | Cosmos N16/N4 work; N1 returns only conditioning frame | Native OpenVLA one-step blocked |
| 937423 | Raw exact-repeat MAE=0; permuted future suffix changes first future frame MAE0.7997/255, frame4 MAE20.5919 | Same-seed noncausal suffix effect; not Gate B |
| 937500 | OpenVLA emits real native7D action, 0.8359s inference, 15.50GB peak | One action, not evaluation |
| 937575 | Cosmos480 video captured, 17frames, 4.1075s inference | Qualitative clip only |
| 937666 | Original IRASim one action → two frames, 2.5467s inference, 23.114s load, 3.517GB peak | Experimental short horizon |
| 937704 | 16 fresh OpenVLA→IRASim ticks, 17frameMP4, 142.214s total inclload/artifacts | Severe visible gripper/blob drift |
| 937739 | 18action-control cases; same-seed repeats both MAE0; directed changes1.197–3.960/255 MAE vs original | Action sensitivity, not physical direction/fidelity |
| 937751 | Paired image-reencode/latent-carry replay plus repeat,124.802s total; bothrepeatall16pixels+latentsidentically | Bothstillshowseveregripperdrift; carryaloneisnotafix |
| 937775 | Released-horizon15action/16frame open-loop reference,34.469s total | More coherent inspected final frame, but firstpredictionsees14futureactions; notnativefeedback |

937704 has an independently audited chain: each policy image hash equals the
previous generated image hash, 16finite7D actions, original IRASim scaling, correct
frame0 removal, and identical state/report rows. It explicitly re-encodes the image
each tick, not upstream latent carry. Its schema-v1 bundle omits exact task and
source-release identity and labels pixel hashes as artifact SHA. Do not rewrite it.
Future closed-loop schema v2 adds exact task/hash, release manifest binding, separate
decoded-pixel and PNG-file hashes.

Qwen actual diagnostics:
- First attempt rejected generate(generator=...) under Transformers5.17; preserved.
- RNG fix used isolated/forked Torch RNG state instead, preserving all raw outputs.
- Format-v2 allows only an anchored JSON code fence as logged transport normalization;
  observable_reasons remains a string, no semantic coercion. Four of five parsed
  on the Cosmos clip; insufficient decisive quorum, unknown.
- Latest data/cluster-evidence/closed-loop-937704/qwen-closed-loop-937704-12238810.json:
  judge call29.292s, peak16.949GB, one decisive success out of5; others uncertain
  or artifact. Unknown/no asserted grade. Scene reference is NOT a goal label.
- Processor check confirms all16selected frames reach Qwen; temporal patch2,
  video_grid_thw [8,18,22], do_sample_frames=false.

## Resume here — current blockers and safe next actions

All bounded diagnostics above have completed. Their artifacts are mirrored locally;
do not resubmit them. All coding agents have handed off their changes. The pending
eight-H100 allocation937277 is the only outstanding cluster request. Root asked the
user whether to cancel it; without a reply it remains queued. Check live scheduler
state before acting; do not infer authorization to cancel from this handoff.

1. Preserve the actual conclusion: Cosmos's pinned one-action path returns no
   future frame; experimental IRASim one-step paths run but visibly drift. Latent
   carry is source-backed and repeatable but did not fix the observed artifact.
   The full-horizon reference sees future actions and cannot replace native
   OpenVLA feedback. Do not enable scored study dispatch from these diagnostics.
2. Paired replay937751 uses independent VAE/diffusion RNG streams. Both branches
   reproduce all16pixel/final-latent hashes on repeat. They first match then diverge
   to final paired MAE10.705/255. Different noise shapes mean native reference937775
   cannot causally attribute the visual difference to horizon alone.
3. Further qualified work needs an acceptable causal feedback backend/profile,
   matched five-task scenario/start/goal panels, and two human annotators with
   held-out labels. WorldGym was researched but NOT adopted/downloaded: explicit
   project/checkpoint licensing and gated VAE access were unresolved. Don't silently
   expand scope into world-model training/adaptation (explicitly excluded by spec).
4. Other policy loaders, full cost/drift studies, judge distillation, and original
   reverse-validation material remain in the full plan. Do not substitute the
   synthetic fixture or the open-loop reference for those requirements.
5. Baseten deployment remains disabled. deploy/baseten/cluster-runtime-evidence.json
   binds selected cluster report hashes; model-contracts.json records those local
   profiles separately from still-null cloud images/serializers/prices/capacity.
   No account credential was available and no cloud request was submitted.
6. Local console now exposes8diagnostic cards/4digest-checked videos separately
   from fixture rows. Full-width layout verified in /tmp/plumb-evidence-layout.png;
   latest build includes the open-loop card and accessible named links. Runtime
   completion never means task success. Restart only when needed after source edits.

Closed-loop CLI output is now atomically write-once: private directory/report
reservations prevent races, and setup failures cannot overwrite prior evidence.
Full-path fake replay tests and production undefined-name checks guard the runtime
failures encountered during this implementation. GPU runtime checks ran on cluster.

## Source releases and evidence immutability

cluster/make_release.py creates a small source-only tar plus RELEASE.json manifest.
Extract to a NEW releases/<SHA>, pass that path to sbatch. Never patch old releases.
Keep report failures and retries separate.

- 534baac1b25e975b7fd635b474169bc7a322969c9537ca4ae7c34d894ef5d026:
  successful closedloop937704.
- 6b13622d40106fbc63ffba3a61053e798123e5a343e211144f22dc5dd61e2327:
  probe937728 preflight failure.
- 1bdc0f11b7f87a0a3d66eae5d1f36afeac2b4c281578d5c469795dcb7e6063fc:
  corrected probe937739.
- d52e328780bcb1a6303faafb4acf7a07676e25936974237adf23b413410a479e:
  paired replay937748 failed on missing _pixel_mae after its firstpair; preserved.
  Four real-runtime CPU-tensor unit tests passed before inference.
- d9edd8ab133e5a89b4fa25f20140c94a27b9bb36910fe852399251f262005e4c:
  corrected replay, full16tick CPUfake regression, staticundefinedname guard.
  Job937751 completed; allartifacts copiedlocally. Do not resubmit it.
- 5bf5213cf992af4c2aaae4525617bce60d25de595830d4465ee2f2fc5d6ffb01:
  native open-loop reference937775 completed; no causal-feedback claim.
- Qwen formatv2 CAC frozen release c092…96f54 is recorded in its report; use full
  report source_release value, not this abbreviation, for any rerun.

## Offline runtimes and downloaded models

Trillium modules: StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17;
IRASim also opencv/4.11.0. Use Alliance Torch2.6.0+computecanada and
torchvision0.21.0+computecanada (module version may omit the metadata suffix).
Do not install big laptop CUDA/Torch wheels or use unreviewed pickle loaders.

- Cosmos: venv-model-std2023, Transformers5.17, Diffusers0.41.dev source
  a3e0b8ec235c27a6c17a21976daf7fd32d819d05, hub1.32.
  models/nvidia--Cosmos3-Nano, 34.99GB,
  revision e59a53c25979a090fa8706c9acc0c254a6e89b92.
- OpenVLA: venv-openvla-tf440; TF4.40.1/tokenizers0.19.1/timm0.9.10.
  models/openvla--openvla-7b,15.09GB,
  revision47a0ec7fc4ec123775a391911046cf33cf9ed83f.
  Reviewed HF remote-code opt-in is mandatory.
- Qwen: models/Qwen--Qwen2.5-VL-7B-Instruct,16.60GB on Trillium and CAC,
  revisioncc594898137f460bfe9f0759e9844b3ce807cfb5.
  CAC venv-qwen uses TF5.17/tokenizers0.23.2. See cluster/cac/README.md.
- IRASim: venv-irasim; TF4.40.1, Diffusers0.24.0, hub0.25.2,
  accelerate0.24.1, timm0.9.10. irasim-source pinned
  c72b6dade6fcd65971e0aa8ab49ea39b15108c90.
  Config model16/extras3/mask1 remains unchanged. One-step probes use video_length2;
  the separate open-loop reference explicitly uses video_length16 with15actions.
  Native action7D is scaled [20,20,20,20,20,20,1], never Cosmos10D.
  Safe checkpoint models/irasim/frame_ada_0300000.safetensors SHA
  d0ea8e8bec50818e414a278fe58fb8d187f577cfe8b260cadc880cfa6107778d.
  Legacy source pt SHAff740a0faf7cfeabfdd598562276afb88e9423dd3cb0c10eecfe005c621094ef.
  Conversion report evidence/irasim-safe-conversion-937657.json.
- IRASim archive is uncompressed TAR despite .tar.gz suffix. Slow ByteDance URL
  stopped; six2GiB verified parts from datasets/fangqi/IRASim revision
  dfcbf85c27b2c5d041dbf5df3df272111e758b48 contain the target checkpoint.
  No full33GB archive needed. Safe extraction uses fixed member offset1536,
  length10867774258. Do not use torch.load(weights_only=False).
- VAE: models/stabilityai--stable-diffusion-xl-base-1.0 (only VAE ~335MB),
  revision462165984030d82259a11f4367a4eed129e94a7b, separate OpenRAIL++ license.
  IRASim scheduler from its pinned source pretrained_models/scheduler.
- Vendor fixture: fixtures/bridge_video.mp4 and bridge_actions.json from
  cosmos-dependencies2b17a2413bd86b2cf9b03823637108851e4ddf2d.
  Action SHA5c26b3cb84799812a70b534ad939551d2ac308fdc870ea0e66163bb52c9d61da.
  IRASim fixture PNG/raw7D action/provenance derived from OpenVLA937500.

## Implementation boundaries and remaining scope

Engine ledger has leases/CAS, stale-owner recovery, cancellation, immutable own-
attempt media, logical denominators, and progress/success invariants.
Measurement refuses active-run statistics, mixed primary manifests/variants, and
cross-task lineage inference. GateD requires raw five-sample provenance-bound
judge reports plus held-out humans; copied labels cannot qualify it.
Baseten submission retries are hard-disabled after ambiguous POST outcomes;
durable callback association still needs integration before enabling deployment.

Other policy families currently have contracts/hooks, not six validated live
loaders. Primary task registry prevents arbitrary prompt/action/success leakage.
The live API enables only synthetic engine runs; cluster diagnostics are separate.
No human annotations, matched five-task panels, calibrated judge/distillation,
qualified cost/drift sweeps, external Baseten credentials/capacity, or original
reverse-validation question/script exist. Preserve these requirements in scope.
