# Gate D runbook — judge calibration

Gate D blocks **primary scoring**. It needs a frozen rubric/model/sampling/label
mapping, blinded human calibration, a held-out report, and validity plus
action-text leakage checks.

The panel is fixed at **150 generated clips: 100 development and 50 held-out**,
stratified across all five tasks (20 development and 10 held-out per task) and
covering policies, apparent successes/failures, and generation defects.
Selection is frozen before annotating. Each development clip receives one
annotation; both annotators independently label all 50 held-out clips, giving the
50-clip overlap and 200 total ratings. Source-state lineages stay disjoint from
the primary study and cost confirmation.

`plumb/calibration.py` enforces all of those counts. This runbook drives it.

**Status right now:** `not_run`. No clip pool exists, the judge has not been run
over a frozen panel, and there is no human annotation.

---

## The honest part: who annotates

Gate D requires **two blinded human annotators**. Three explicitly typed arms
exist and none of them is ever recorded as another:

| `annotator_type` | What it is | Satisfies Gate D |
|---|---|---|
| `human` | A person. `annotator_id` must start `human:` | Yes, with two of them |
| `model` | A blinded model pass; records `model_id` and `model_revision`. `annotator_id` must start `model:` | No |
| `external_label` | Third-party labels this project did not produce, e.g. AutoEval's own classifier labels on real drawer video. `annotator_id` must start `external:` | No |

Whenever two blinded humans are absent:

- `results/judge_calibration.json` carries `calibration_class` naming the actual
  arms, for example `external_label + model_reference`.
- Gate D reads **`pass_with_limitations`**, `passed` stays `false`,
  `thresholds_satisfied` records whether the numbers were met, and
  `human_annotation` is listed in `open_dependencies`.
- `plumb/gates.py` receives status `blocked`, since `GateStatus` has no
  `pass_with_limitations` member. The richer value is in
  `measurements.gate_d_decision_status`.

Unresolved disagreement between the two annotators stays unresolved and remains
in the coverage and missingness counts. **No project-team tie-break after seeing
VLM output is permitted**, and no code path offers one.

---

## Prerequisites

| Requirement | Why |
|---|---|
`results/scenarios.jsonl` from Gate C | Supplies the calibration-cohort lineages and the reserved primary / cost-confirmation lineages
A pool of generated clips over calibration-cohort starts | The 150 clips are generated video, not source video
Local Qwen2.5-VL-7B-Instruct snapshot at a pinned 40-hex revision | The judge loader is local-only
One goal/reference image per task with a recorded SHA-256 | `JudgeRequest` requires provenance-backed references
Frozen `protocol.json` and its SHA-256 | Gate D evidence binds to a protocol hash
`imageio` and `Pillow` in the judge environment | Clip decode and reference load

```bash
export PLUMB_ROOT=/scratch/$USER/plumb
export QWEN_REV=cc594898137f460bfe9f0759e9844b3ce807cfb5
export PROTOCOL_ID=protocol-v1
export PROTOCOL_HASH=$(python -c "import json;print(json.load(open('protocol.json'))['sha256'])")
export BASE_SEED=20260919
```

---

## Step 1 — Assemble the candidate clip pool

One JSONL row per generated clip:

```json
{"task":"open_drawer","policy":"OpenVLA","apparent_outcome":"apparent_failure",
 "generation_defect":false,"source_lineage_id":"<calibration-cohort lineage>",
 "media_ref":"/scratch/.../generated/OpenVLA/open_drawer-0007.mp4",
 "video_sha256":"sha256:..."}
```

`apparent_outcome` is one of `apparent_success`, `apparent_failure`, `unknown`.
It is the operator's cheap pre-label used only for stratification; it is never a
ground-truth label and never reaches an annotator.

Every `source_lineage_id` must come from the Gate C **`calibration`** cohort. To
fill 30 clips per task with disjoint development and held-out lineages you need
at least 30 distinct calibration lineages per task, which is why
`build_scenarios.py` defaults `--calibration-starts 30`.

Aim for well above 30 candidates per task so the stratified fill can actually
cover policies, apparent outcomes, and defects.

## Step 2 — Freeze the panel before any annotation

```bash
python cluster/build_calibration_panel.py select \
  --pool results/clip_pool.jsonl \
  --exclude-lineages results/scenarios.jsonl \
  --manifest-out results/calibration_manifest.jsonl \
  --freeze-out results/calibration_panel_freeze.json \
  --media-root $PLUMB_ROOT/calibration/clips \
  --stage-media \
  --execute
```

**Produces**

- `results/calibration_manifest.jsonl` — 150 rows validated by
  `plumb.calibration.validate_manifest`.
- `results/calibration_panel_freeze.json` — the selection salt, per-task strata
  coverage, rejected candidates with reasons, and the
  `calibration_manifest_hash`.
- `$PLUMB_ROOT/calibration/clips/<clip_id>.mp4` — each clip hard-linked (or
  copied) to a **blinded** path, with source and staged digests compared.

Selection is a pure function of the pool: re-running on the same pool reproduces
the same panel byte for byte. Clip IDs are `clip-<12 hex>` derived from a salted
hash of the source identity, and media references are rewritten to
`artifact://calibration/<clip_id>.mp4`, so a generated path containing a policy
directory name cannot reach the blinded view.

**Exit code 2 means stop.** `insufficient_candidate_pool` means a stratum cell
cannot be filled. Generate more clips. Never reuse a lineage to reach a count.

`--exclude-lineages results/scenarios.jsonl` is what keeps the panel disjoint
from the primary study and cost confirmation. Do not omit it.

Check the freeze record before continuing:

- `status` is `pass`
- `selected_clips` is 150 and `shortfall` is empty
- `strata_coverage.<task>.strata` shows more than one policy and more than one
  apparent outcome actually drawn
- `staging.status` is `pass` with 150 staged

## Step 3 — Pin the pre-run judge evidence

```bash
python cluster/build_calibration_panel.py freeze \
  --manifest results/calibration_manifest.jsonl \
  --media-root $PLUMB_ROOT/calibration/clips \
  --protocol-id $PROTOCOL_ID \
  --model-revision $QWEN_REV \
  --processor-revision $QWEN_REV \
  --transformers-version 4.49.0 \
  --runtime-lock-id <runtime lock id> \
  --asset-manifest-id <assets.lock.json id> \
  --output results/frozen_judge_evidence.prerun.json \
  --execute
```

**Produces** the calibration manifest hash, the 50 held-out video SHA-256
digests, the task registry ID and hash, the model identity, the protocol ID, and
the producer identity (SHA-256 of the local `plumb/policies/judge.py` bytes).

`heldout_artifact_hashes` is deliberately empty here. The judge computes frame
pixel hashes at call time, so they cannot be preregistered; step 6 fills them and
labels that as post-hoc consistency checking, not preregistration.

**Record the file's `created_at`.** Gate D requires the judge to be frozen before
the held-out split was evaluated, and that claim is checked by comparing this
timestamp against the judge run's.

`status: blocked` with `missing_heldout_media` means staging did not complete.
Fix step 2 first.

## Step 4 — Choose the rubric and sampling on development data only

Development is the only place a protocol decision may be made. Run the judge
over `--split development` and inspect the results:

```bash
python cluster/build_calibration_panel.py judge \
  --manifest results/calibration_manifest.jsonl \
  --frozen results/frozen_judge_evidence.prerun.json \
  --media-root $PLUMB_ROOT/calibration/clips \
  --model-path $PLUMB_ROOT/models/Qwen--Qwen2.5-VL-7B-Instruct \
  --reference open_drawer=$PLUMB_ROOT/goal/open_drawer.png \
  --reference close_drawer=$PLUMB_ROOT/goal/close_drawer.png \
  --reference to_basket=$PLUMB_ROOT/goal/to_basket.png \
  --reference to_sink=$PLUMB_ROOT/goal/to_sink.png \
  --reference fold_cloth=$PLUMB_ROOT/goal/fold_cloth.png \
  --base-seed $BASE_SEED --split development \
  --reports-out results/judge_reports.development.jsonl \
  --run-report results/judge_run.development.json \
  --execute
```

Any revision to the protocol must freeze its sample count, quorum, schema, and
cost model together, and must happen **before** step 5. The current protocol's
count stays five and quorum stays three.

## Step 5 — Run the frozen judge over the 50 held-out clips

Drop `--execute` first to see the plan: which clips, which seeds, which frozen
video hash.

```bash
python cluster/build_calibration_panel.py judge \
  --manifest results/calibration_manifest.jsonl \
  --frozen results/frozen_judge_evidence.prerun.json \
  --media-root $PLUMB_ROOT/calibration/clips \
  --model-path $PLUMB_ROOT/models/Qwen--Qwen2.5-VL-7B-Instruct \
  --reference open_drawer=$PLUMB_ROOT/goal/open_drawer.png \
  --reference close_drawer=$PLUMB_ROOT/goal/close_drawer.png \
  --reference to_basket=$PLUMB_ROOT/goal/to_basket.png \
  --reference to_sink=$PLUMB_ROOT/goal/to_sink.png \
  --reference fold_cloth=$PLUMB_ROOT/goal/fold_cloth.png \
  --base-seed $BASE_SEED --split heldout \
  --reports-out results/judge_reports.heldout.jsonl \
  --run-report results/judge_run.heldout.json \
  --execute
```

**Produces** one raw report per clip in
`results/judge_reports.heldout.jsonl`: all five samples, each sample's attempts,
raw outputs, logged seeds, and the full provenance block that
`plumb.calibration.validate_primary_judge_evidence` re-aggregates.

Fixed operating point, enforced by `JudgeSamplingConfig.validate` and re-checked
by `plumb.calibration._validate_primary_sampling`:

| Setting | Value |
|---|---|
| samples | 5 |
| quorum | 3 |
| temperature | 0.7 |
| top_p | 1.0 |
| `max_new_tokens` | 512 |
| retries per sample | 1, for schema/transport failure only |

Sixteen frames are sampled uniformly from the original start through the exact
final control tick, both endpoints included. A clip that cannot supply 16 frames
is **rejected**, never padded or resampled. Each clip's five seeds are derived
deterministically from the base seed and clip ID and are logged in the run
report, so the run is reproducible.

The runner is append-only and resumable: re-running skips clips already in
`--reports-out`. One clip's failure is recorded in `failures` and does not lose
the batch.

The judge never receives a policy name, action text, command, reference success
percentage, condition label, or gate outcome. `JudgeRequest.validate` refuses a
free-text instruction in primary mode and refuses any percentage in a diagnostic
rubric.

**Record the run report's `created_at`** as `judge_started_at`.

## Step 6 — Bind the produced artifact hashes and emit the Gate D bundle

```bash
python cluster/build_calibration_panel.py bind \
  --manifest results/calibration_manifest.jsonl \
  --frozen results/frozen_judge_evidence.prerun.json \
  --reports results/judge_reports.heldout.jsonl \
  --run-report results/judge_run.heldout.json \
  --output results/frozen_judge_evidence.json \
  --gate-evidence results/gate_d_evidence.judge.json \
  --evidence-uri file://$PWD/results/judge_reports.heldout.jsonl \
  --evidence-uri file://$PWD/results/calibration_panel_freeze.json \
  --protocol-hash $PROTOCOL_HASH \
  --execute
```

`status: blocked` with `heldout_clips_without_bound_artifact_hashes` means the
reports do not cover all 50 held-out clips. `null_artifact_hashes` means the
judge could not hash a frame, which makes those reports Gate-D unavailable.

The binding checks immutable identifiers and a declared producer. It is **not** a
cryptographic attestation that a model ran, and the emitted record says so.

## Step 7 — Register the annotators, honestly

Two blinded humans (the real Gate D path):

```bash
python -m plumb.annotation register \
  --output results/annotators.json \
  --annotator '{"annotator_id":"human:alex","annotator_type":"human"}' \
  --annotator '{"annotator_id":"human:blair","annotator_type":"human"}'
```

Surrogate arms when no second human is available before the deadline:

```bash
python -m plumb.annotation register \
  --output results/annotators.json \
  --annotator '{"annotator_id":"external:autoeval-classifier","annotator_type":"external_label",
    "label_source":"zhouzypaul/auto_eval drawer classifier labels",
    "source_uri":"hf://datasets/zhouzypaul/auto_eval"}' \
  --annotator '{"annotator_id":"model:qwen-blind-pass-b","annotator_type":"model",
    "model_id":"Qwen/Qwen2.5-VL-7B-Instruct","model_revision":"'$QWEN_REV'"}'
```

A `model` or `external_label` annotator cannot use a `human:` identifier, cannot
omit its model/source identity, and cannot be serialized with
`annotator_type: "human"`. The external-label arm is scoped to drawer clips,
because that is the only scene AutoEval's classifier labels cover.

## Step 8 — Annotate, blinded

Every annotation subcommand takes the same three paths. Spell them out (an
unquoted shell variable does not word-split in zsh):

```bash
python -m plumb.annotation session \
  --manifest results/calibration_manifest.jsonl \
  --annotators results/annotators.json \
  --store results/labels.jsonl \
  --annotator-id human:alex
```

`queue` lists every blinded clip assigned to that annotator; `show` renders one:

```bash
python -m plumb.annotation queue \
  --manifest results/calibration_manifest.jsonl \
  --annotators results/annotators.json \
  --store results/labels.jsonl \
  --annotator-id human:alex

python -m plumb.annotation show \
  --manifest results/calibration_manifest.jsonl \
  --annotators results/annotators.json \
  --store results/labels.jsonl \
  --annotator-id human:alex --clip-id <clip_id>
```

Record one label:

```bash
python -m plumb.annotation submit \
  --manifest results/calibration_manifest.jsonl \
  --annotators results/annotators.json \
  --store results/labels.jsonl \
  --annotator-id human:alex --clip-id <clip_id> \
  --integrity intact --collision none_visible \
  --progress 4 --completion-evidence not_met \
  --frames '[3,11,15]' \
  --reason "Drawer moves but does not reach the closed position by the final frame."
```

Leave a field off when the video does not show it. A blank field stays unknown,
the capture is recorded as incomplete, and it is **not** counted as a label or as
a failure. `progress` may be null; `integrity`, `collision`,
`completion_evidence`, evidence frame indices, and an observable reason are the
other capture fields.

Coverage at any time:

```bash
python -m plumb.annotation status \
  --manifest results/calibration_manifest.jsonl \
  --annotators results/annotators.json \
  --store results/labels.jsonl
```

The view shows the clip, the task instruction, the frozen 0–5 milestone rubric,
and nothing else. No policy, backend, command or action text, source condition,
gate status, reference rate, world seed, cohort, or split. Enforcement uses
`plumb.calibration.BLINDED_EXPORT_FIELDS` as the allowlist and
`plumb.calibration._FORBIDDEN_BLIND_FIELDS` as the denylist, plus a value scan
that rejects any policy, backend, or split label appearing inside a string.

### The ~30-minute human upgrade path

This is the step that turns Gate D from `pass_with_limitations` into a real human
calibration. It needs **two people and about 30 minutes each**.

1. Register two `human:` annotators (step 7, first form). Keep any earlier
   surrogate store file; start a fresh `--store results/labels.human.jsonl`.
2. Each person opens the blinded `/annotate` view (or runs
   `python -m plumb.annotation queue`) and labels **all 50 held-out clips**
   independently. At roughly 30 seconds per clip that is about 25 minutes.
   They must not discuss clips, and neither may see the VLM output.
3. Development clips can follow later; the held-out 50 are what Gate D needs for
   the overlap and the judge comparison.
4. Re-run step 9 with the human store. `calibration_class` becomes `human`,
   `has_two_blinded_humans` becomes true, and `human_annotation` disappears from
   `open_dependencies`.
5. Whatever the two disagree on **stays unresolved**. Do not adjudicate it, and
   never adjudicate after seeing the judge's answer.

Full coverage needs 200 ratings (150 development assignments plus the 50
held-out clips labelled twice). The held-out 50 × 2 is the Gate D minimum.

## Step 9 — Write `results/judge_calibration.json`

```bash
python -m plumb.annotation report \
  --manifest results/calibration_manifest.jsonl \
  --annotators results/annotators.json \
  --store results/labels.jsonl \
  --judge-reports results/judge_reports.heldout.jsonl \
  --protocol results/gate_d_protocol.json \
  --tolerances results/gate_d_tolerances.json \
  --manifest-id calibration-panel-v1 \
  --scenario-manifest-sha256 <scenario manifest sha256> \
  --output results/judge_calibration.json
```

`results/gate_d_protocol.json` is a `FrozenGateDProtocol` whose
`evidence_manifest` is the bound record from step 6:

```json
{"protocol_hash":"sha256:...","rubric_hash":"sha256:...","sampling_hash":"sha256:...",
 "label_schema_version":1,"evidence_manifest":{ "...": "contents of results/frozen_judge_evidence.json" }}
```

`results/gate_d_tolerances.json` is a `GateDTolerances`, written **before**
held-out labels are exposed:

```json
{"minimum_heldout_overlap":50,
 "minimum_human_annotation_coverage":1.0,
 "minimum_human_consensus_coverage":0.8,
 "minimum_judge_comparison_coverage":0.8,
 "minimum_sensitivity":0.8,
 "minimum_specificity":0.8,
 "maximum_absolute_leniency_offset":0.1}
```

Those numbers are the shape the gate requires. Choose the actual values on
development fixtures and freeze them in `protocol.json` before exposing held-out
labels. Passing the pooled gate does not establish tight task-specific accuracy;
each task's 10-clip result is exploratory and carries intervals.

`results/judge_calibration.json` contains:

- `calibration_class` and the full annotator registry with each declared type
- `split_ids` (manifest ID, calibration manifest hash, development and held-out
  clip IDs) and `annotation_ids` (clip, annotator, annotator type)
- `human_human_agreement` with raw agreement and its stated interpretation:
  a reproducibility reference for this two-annotator protocol, **not** a
  mathematical ceiling on judge accuracy
- `cohens_kappa` for `binary_success`, `integrity`, `collision`, and
  `completion_evidence`
- `progress_quadratic_weighted_kappa` over the frozen 0–5 scale
- `confusion_matrix`, `sensitivity`, `specificity`
- `uncertainty`: Wilson 95% intervals for sensitivity and specificity, a paired
  Wald interval for the leniency offset, and `null_coverage` for missingness
- `leniency_offset` — judge-positive minus human-positive proportion on the same
  consensus-labelled cases — with a check that it equals
  (false positives − false negatives) / comparable cases
- `stratification_disclosure`: prevalence-sensitive statistics from an enriched
  calibration sample are not population estimates without sampling weights
- `gate_d` with `status`, `passed`, `thresholds_satisfied`, and
  `open_dependencies`

The file is written atomically and never contains NaN or infinity; an undefined
statistic is `null`. A degenerate kappa denominator yields `null`, not zero.

`plumb.calibration` can write the same artifact directly:

```bash
python -m plumb.calibration report --manifest ... --annotations ... \
  --annotators human:alex human:blair \
  --annotator-registry results/annotators.json \
  --output results/judge_calibration.json
```

`--output` refuses to run without `--annotator-registry`: a missing annotator
type is never treated as human.

## Step 10 — Record the gate

```python
import json
from plumb.annotation import AnnotatorRegistry, gate_d_evidence
from plumb.gates import GateLedger, GateRecord

payload = json.load(open("results/judge_calibration.json"))
registry = AnnotatorRegistry.load("results/annotators.json")
bundle = gate_d_evidence(
    payload,
    registry,
    judge_evidence=json.load(open("results/gate_d_evidence.judge.json")),
    evidence_uris=["file:///.../judge_reports.heldout.jsonl"],
    protocol_hash="sha256:...",
    freeze_created_at=json.load(open("results/frozen_judge_evidence.prerun.json"))["created_at"],
    judge_started_at=json.load(open("results/judge_run.heldout.json"))["created_at"],
)
ledger = GateLedger.load("results/gates.json")
ledger.record(GateRecord.from_mapping(bundle))
ledger.save("results/gates.json")
```

---

## Which `results/gates.json` fields this fills

| Field | Source |
|---|---|
| `gates.D.status` | `pass` only with two blinded humans **and** every tolerance met; otherwise `blocked` |
| `gates.D.protocol_hash` | `--protocol-hash` |
| `gates.D.fixture_ids` | The 50 held-out clip IDs |
| `gates.D.judge_revisions` | Model ID, model and processor revision, transformers version, task registry ID and hash, producer source hash |
| `gates.D.evidence_uris` | Judge report JSONL, panel freeze record, bound evidence manifest |
| `gates.D.evidence_kind` | `real_vlm_judge_reports_over_generated_clips` |
| `gates.D.measurements.binary_kappa` | Cohen's kappa on the two annotators' visible-completion labels where both were decisive |
| `gates.D.measurements.weighted_progress_kappa` | Quadratic-weighted kappa over 0–5 |
| `gates.D.measurements.leniency_offset` | Judge-positive minus human-positive on consensus-labelled cases |
| `gates.D.measurements.consensus_coverage` | Held-out binary-success consensus coverage |
| `gates.D.measurements.calibration_class` | `human`, `human + model_reference`, `external_label + model_reference`, … |
| `gates.D.measurements.held_out_frozen_before_evaluation` | Freeze timestamp ≤ judge-run timestamp; `null` when either is absent |
| `gates.D.measurements.gate_d_decision_status` | `pass` / `pass_with_limitations` / `fail` / `blocked` / `not_evaluable` |
| `gates.D.measurements.judge_status_counts` | Counted `evaluable` vs `unknown` per-clip outcomes |
| `gates.D.thresholds` | The frozen `GateDTolerances` plus the sampling operating point |
| `gates.D.reasons` | Every threshold reason code and every open dependency |

`plumb.gates.GateRecord.pass_evidence_errors` independently requires
`binary_kappa`, `weighted_progress_kappa`, `leniency_offset`,
`consensus_coverage`, a non-empty `calibration_class`, and
`held_out_frozen_before_evaluation`. A null blocks the pass, which is correct
when the quantity was not measured.

## Acceptance thresholds

| Threshold | Value |
|---|---|
| Panel | exactly 150 clips: 100 development, 50 held-out |
| Per task | exactly 20 development and 10 held-out |
| Held-out overlap | all 50 clips labelled independently by both annotators |
| Total ratings at full coverage | 200 |
| Split lineage separation | development and held-out lineages disjoint, and both disjoint from primary and cost confirmation |
| Blinded human annotators | 2 |
| Sampling | 5 samples, quorum 3, T=0.7, top_p 1.0, 512-token cap, 1 bounded retry |
| Frames per clip | exactly 16, endpoints included, never padded |
| Sensitivity / specificity / leniency offset / coverages | per the frozen `GateDTolerances` |
| Judge frozen before the held-out split was evaluated | required |

If the gate fails, revise and obtain a **fresh** held-out set. Do not tune
against the same test set and then present it as untouched evidence.

## What Gate D cannot establish

- Five samples are not five independent robot episodes.
- Refusals, exhausted retries, and disagreement are explicit missing outcomes,
  never forced labels. Stage-A invalid or unknown episodes stay unevaluable for
  primary rates.
- The evidence binding checks declared identifiers and a declared producer. It
  does not prove a model ran and does not establish scientific validity.
- Human-human agreement is a reproducibility reference, not a ceiling.
- Statistics come from a deliberately enriched sample. Without sampling weights
  they are not population estimates.
- Distillation is a separate judge revision with its own fresh held-out
  calibration and paired frozen-video comparison; training labels and compute
  are an additional acquisition and budget item, not supplied by these 50 clips.

## Remaining external dependencies

1. **`human_annotation`** — two blinded human annotators labelling the 50
   held-out clips. This is the only dependency that converts
   `pass_with_limitations` into `pass`.
2. A pool of generated clips over Gate C calibration-cohort starts, which needs
   a fidelity-qualified world-model path (Gates A and B).
3. A local Qwen2.5-VL snapshot at a pinned revision, on a GPU.
4. One provenance-backed goal/reference image per task with a recorded SHA-256.
5. Preregistered numerical tolerances chosen on development fixtures and frozen
   in `protocol.json` before held-out labels are exposed.
6. An externally auditable timestamp for the frozen protocol; a local hash alone
   is not preregistration.
