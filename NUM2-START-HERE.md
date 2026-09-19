# PLUMB continuation — current entrypoint

Continue in `/Users/lucas/Desktop/tax_stuff/htn26`, branch `main`.
Read the first sections of HANDOFF.md and CLUSTER-OPERATIONS.md, then the
relevant runbooks below. Older handoff sections are historical snapshots, not
current authentication, process, job, or approval state.

The merge is delivered. Review/format implementation is pushed in `89f1b99`,
operator documentation in `f736b52`, and actual SuSIE inference in `892e221`.
Check git log/status before editing. Preserve
other contributors' changes; never force-push or overwrite evidence.

## Actual completed work

- Judge semantic pilot: training 939451 and reload 939458 completed.
  Do not scale/deploy: singleton teacher labels were homogeneous and the
  adapter copied their bias. See docs/JUDGE-PILOT-V1.md.
- Teacher-v2 939753: 75/80 schema-valid samples, zero 3/5 unique modes;
  all 16 clips abstained. Export guard correctly created no training dataset.
- Structure-only judge training 939998 and reload 940000: 12 train/4 dev,
  24 optimizer steps; semantic target tokens masked. Bare JSON 0/4→4/4,
  schema validity 4/4 for both, but all four judgment tuples drifted.
  Adapter stays disabled. See docs/JUDGE-FORMAT-PILOT.md.
- SuSIE_LL 940190 COMPLETED 0:0 on one H100: strict parameter restore,
  finite 1×7 native action, exact reset/repeat. Static first/final vendor-video
  conditioning only—not a rollout, task result, or Gate pass.
  Correct source: SOAR eabd5f16a856e484884a22e257a941bb358cea08,
  model_training/; not BridgeData's incompatible gc_bc.
  Parameters-only inference excludes optimizer state and cannot resume training.
  Four preceding failures are preserved. See docs/SUSIE-LL-DIAGNOSTIC.md.
- Private development review works at http://127.0.0.1:8787/review.
  Real browser QA covered 16 decoded frames, draft save/reload, and mobile
  layout using an isolated temporary model-assisted test store. Production
  review data remains blank; no human labels were fabricated.
  See docs/DEVELOPMENT-REVIEW.md.

## State and boundaries

The app uses data/live-integrated, local tmux plumb-live, port 8787.
It exposes 24 diagnostics. Recheck its current PID before restarting.
All A–F gates remain not_run; real submission is disabled. Final verification:
1,267 Python tests passed, 6 skipped;
144 frontend tests passed and production build passed.

Trillium's scheduler was empty after completed job940190. A later final
BatchMode check returned permission denied; authentication has expired again.
No new job was submitted afterward. Do not spam Duo or delete sockets. Root alone owns
cluster operations. Use one allocated H100 for bounded NEW experiments; never
infer on login nodes or repeat completed jobs unchanged. Models/adapters stay
on clusters. Warn before additional large downloads. Do not train the world model.

SuSIE checkpoint: /scratch/lchow432/plumb/models/patreya--gcbc-bridge/checkpoint/checkpoint.
Successful report: data/live-integrated/cluster-evidence/susie-ll-gcbc-940190/report.json.
Review resolver: data/private/judge-pilot-review-v1-resolver.json; keep private.

The complete six-policy × five-task × 50-start scientific study is NOT done.
Remaining requirements include reliable native world feedback, remaining policy
assets/conformance, matched real starts, independent human calibration, and
cloud/deployment/cost evidence. Never fabricate them or promote diagnostics.
Continue independent implementation without repeated permission/continuation
questions. Use Terra xhigh subagents only for useful bounded parallel tasks,
with explicit ownership.
