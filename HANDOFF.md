# PLUMB execution handoff — implementation in progress

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
