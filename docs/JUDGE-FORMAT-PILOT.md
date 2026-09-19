# Judge JSON format-only pilot

This is an unqualified formatting experiment, not a semantic judge, calibration
result, Gate D result, or production adapter.

## Recorded outcome

- Training job **939998** completed a two-epoch, 24-optimizer-step pilot on 12
  development-train rows. The paired development set has 4 rows.
- The target mask was structure-only: semantic value tokens supervised = **0**.
  The recorded development syntax loss changed from **1.71370849** to
  **0.02586903**.
- Compare job **940000** reloaded the saved epoch-2 adapter and generated
  greedily on the 4 development rows. Bare JSON changed from **0/4** base to
  **4/4** adapter; both were schema-valid **4/4**. Semantic tuples nevertheless
  drifted on **4/4** comparable rows.

The adapter therefore remains disabled. Better JSON framing did not preserve
judgments on this small development comparison, and neither report makes a
human-quality, calibration, or scoring claim.

## Evidence and locations

- Local training report:
  `data/live-integrated/cluster-evidence/judge-format-pilot-939998/report.json`
- Local compare report:
  `data/live-integrated/cluster-evidence/judge-format-compare-940000/report.json`
- Cluster experiment directory:
  `/scratch/lchow432/plumb/experiments/judge-format-only-v1`
- Cluster compare evidence:
  `/scratch/lchow432/plumb/evidence/judge-format-only-v1-comparison.json`
- Immutable source release:
  `f6b30d952be1b2e9cccf4c5915a4e85a636b22b620634e485dfde0ab77c08fc5`

No new base-model weights were downloaded. The adapter remains cluster-only.
All qualification gates remain unchanged, and the full six-policy, five-task,
1,500-episode scope is still incomplete.

## Review status

Root verified the live review route at `http://127.0.0.1:8787/review` with an
isolated Chromium QA database: all 16 clips and decoded images loaded, drafts
saved/reloaded/resumed, no JavaScript errors occurred, and a 390px viewport had
no horizontal overflow. This is UI/temporary-DB verification only. The
production review DB is absent, no human labels have been recorded or inferred,
and an isolated `model_assisted` test draft is not a human annotation.
