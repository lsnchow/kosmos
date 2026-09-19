# Next: real development review, not more weak-label training

The experimental adapter exists and reloads; see [the v1 report](JUDGE-PILOT-V1.md).
The v2 prompt/selection diagnostic then reported75/80 schema-valid samples but
**0/16 clips with a unique three-vote semantic mode**. All16 abstained. No v2
training dataset or adapter was created, and the threshold was not relaxed.

Trillium job939753 retained its raw reports and
`/scratch/lchow432/plumb/experiments/judge-teacher-v2-diagnostic/summary-939753.json`.
The summary was observed through SSH stdout. SSH expired before the complete
v2 bundle could be mirrored; do not claim a local copy already exists.

## Restore access and audit

Run `ssh trillium-gpu` and complete Duo locally. The previous control socket
expired; one agent-initiated Duo request timed out and was cancelled. No
password/passcode needs to be shared in chat. Recheck actual job state after
login; do not rerun completed teacher/training/comparison jobs.

Stage a fresh source release before using the new exporter on the cluster.
Run its CPU-only audit with CUDA_VISIBLE_DEVICES empty and the existing
Python3.11/Arrow training environment:

```bash
python -m cluster.export_judge_v2_pilot \
  --source-root /scratch/lchow432/plumb/experiments/judge-lora-pilot-v1-prepared \
  --v2-root /scratch/lchow432/plumb/experiments/judge-teacher-v2-diagnostic \
  --output-root /scratch/lchow432/plumb/experiments/judge-v2-pilot-dataset \
  --audit-output /scratch/lchow432/plumb/evidence/judge-v2-pilot-export-audit-939753.json
```

It recomputes selection from raw teacher outputs. With current evidence it
should emit a **blocked audit only**, return2, and create no dataset. Future
export requires≥2 train rows,≥1 validation row, and≥2 distinct training semantic
tuples, each supported by a unique mode of at least3/5 actual samples. No
cohort reassignment or label repair is allowed.

## Prepare blank review materials

`cluster.prepare_judge_review` is implemented/tested but has **not yet been run
on the actual16 clips**, because their video/PNG mirror needs restored SSH.
Mirror only `candidate-inputs.json` and `clips/` from the prepared dataset into
local `data/judge-pilot-v1-review-source/`; inspect transfer size first. Never
copy the model view or trained adapter to the laptop.

```bash
.venv/bin/python -m cluster.prepare_judge_review \
  --candidate-manifest data/judge-pilot-v1-review-source/candidate-inputs.json \
  --dataset-root data/judge-pilot-v1-review-source \
  --public-out data/live-integrated/review/pilot-review-v1 \
  --private-resolver data/private/judge-pilot-review-v1-resolver.json
```

This verifies every source MP4/PNG hash before publishing anything. It creates
opaque-ID media copies, a Markdown index with working localhost artifact links,
and16 blank, unassigned JSONL worksheets. It creates no humans, annotations,
registry or calibration claim. The coordinator resolver retains source
identity/cohort/lineage/hashes, is mode0600, and stays **outside** the served
`data/live-integrated` directory. Do not give it to reviewers.

Two real reviewers can later independently inspect the clips, definitions and
visible evidence. These already-used developmental clips remain reserved;
they cannot become fresh held-out calibration data. A qualified judge still
needs the separate, stratified, genuinely human-labelled formal panel.
