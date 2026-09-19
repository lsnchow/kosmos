# Distilled-judge preparation workflow

This is a preparation workflow for a **new judge revision**. It freezes local
training/validation inputs and a no-submit Training Job readiness record. It
does not upload data, invoke Baseten, spend money, start a GPU, or mark a
distilled judge calibrated.

## Preconditions

Start from a verified frozen calibration selection created by `plumb annotation
freeze`. It contains 100 development clips and 50 held-out clips, immutable
media hashes, and a passing lineage-partition record against the primary and
cost-confirmation panels.

Only the 100 `development` clips can be exported. The tool rejects:

- held-out labels or teacher reports;
- primary or cost-confirmation lineage overlap from the frozen selection;
- missing development labels or teacher raw outputs;
- duplicate or shared train/validation source lineages;
- a copied held-out calibration as evidence for a distilled revision.

## Frozen protocol input

Prepare a JSON object like this schema fragment. Replace every bracketed value
with a reviewed, immutable value; the fragment is not an actual training plan
or a claim that a configuration is optimal.

```json
{
  "protocol_id": "<new-distillation-protocol-id>",
  "teacher": {
    "model_id": "<teacher-model-id>",
    "model_revision": "<reviewed-model-revision>",
    "processor_revision": "<reviewed-processor-revision>",
    "runtime_lock_id": "<frozen-runtime-lock-id>",
    "asset_manifest_id": "<frozen-asset-manifest-id>",
    "producer_source_hash": "sha256:<reviewed-producer-source>"
  },
  "rubric_hash": "sha256:<frozen-rubric>",
  "sampling_hash": "sha256:<frozen-teacher-sampling>",
  "development_split_seed": <integer>,
  "candidate_lora": {
    "learning_rate": 0.001,
    "rank": 64,
    "alpha": 32,
    "status": "candidate_not_optimized"
  },
  "search_plan": {
    "preregistered": true,
    "candidates": ["<at-least-two-unselected-candidate-configurations>"],
    "early_stopping": {"enabled": true, "metric": "<development-validation-metric>", "patience": <positive-integer>}
  }
}
```

The `lr=1e-3`, `r=64`, `alpha=32` LoRA configuration is retained as one
candidate. The workflow requires it to appear in a preregistered small search
with at least one other candidate and early stopping on the development
validation split. It never selects a winner or calls it portable/optimal.

## Teacher report input

Provide raw teacher output for every development clip. Each report keeps the
full raw output (either `raw_outputs` or retained sample attempts) and binds it
to the teacher, rubric, sampling, frozen calibration manifest, and clip media
hash.

```json
{
  "raw_outputs": ["<unmodified-teacher-text-output>"],
  "provenance": {
    "clip_id": "<internal-development-clip-id>",
    "calibration_manifest_hash": "sha256:<selection-manifest-hash>",
    "media_hash": "sha256:<development-media-hash>",
    "rubric_hash": "sha256:<frozen-rubric>",
    "sampling_hash": "sha256:<frozen-sampling>",
    "teacher": {"<exact-fields-from-protocol-teacher>": "<exact-values>"}
  }
}
```

Use the completed development human annotations returned from the blinded
annotation workflow. Do not include held-out annotations in the command.

```bash
plumb distillation freeze-dataset \
  --calibration-selection freezes/calibration-selection-v1.json \
  --annotations returned/development-annotations-a.jsonl \
  --annotations returned/development-annotations-b.jsonl \
  --annotators '<annotator-a-id>' '<annotator-b-id>' \
  --teacher-reports teacher-development-raw.jsonl \
  --protocol distillation-protocol.json \
  --output freezes/distillation-dataset-v1.json
```

The artifact contains 80 deterministic training and 20 deterministic validation
records, stratified as 16/4 within each task. It refers to source media by its
frozen media reference and SHA-256; it does not copy media or labels to a cloud
service.

## Training Job readiness, without submission

Supply only non-secret reviewed job inputs. Account, container image, and
runtime lock remain explicit requirements; unknown values remain null/missing
and become blockers. There is intentionally no Baseten endpoint or request
payload in this repository because the exact current Training Jobs API contract
has not been verified for this deployment.

```json
{
  "job_name": "<candidate-name>",
  "account_id": "<reviewed-account-id-or-omit>",
  "runtime_image": "<reviewed-image-digest-or-omit>",
  "runtime_lock_id": "<reviewed-runtime-lock-or-omit>"
}
```

```bash
plumb distillation job-readiness \
  --dataset freezes/distillation-dataset-v1.json \
  --job-inputs training-job-inputs.json \
  --output freezes/distillation-job-readiness-v1.json
```

The result always says `submission_performed: false`, has no endpoint or API
payload, and remains blocked until the exact account/runtime/image and official
request contract are reviewed. Credentials such as API keys, tokens, secrets,
passwords, and authorization values are rejected from the readiness record.

## Post-training evidence for every distilled revision

No distilled model can reuse the original held-out calibration labels. For each
new revision, retain a separate evidence declaration and run:

```bash
plumb distillation revision-readiness \
  --dataset freezes/distillation-dataset-v1.json \
  --revision-id '<new-distilled-revision-id>' \
  --evidence fresh-distilled-judge-evidence.json \
  --output freezes/distilled-revision-readiness-v1.json
```

The declaration must show a fresh, distinct held-out calibration manifest with
fresh human annotations, no copied labels, lineage disjointness from development,
provenance-bound raw judge reports, and a passing paired frozen-video comparison.
Even a structurally complete result still has `primary_scoring_allowed: false`:
Gate D must be recorded from real evidence, and original-teacher results remain
preserved alongside the distilled revision.
