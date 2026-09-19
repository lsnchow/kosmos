# Distillation runbook — a second judge revision, on Training Jobs

Judge distillation produces a **new judge revision**. It never edits the
original judge's results, and it cannot score a burst until it has passed a
**fresh** held-out calibration and a **paired frozen-video comparison** of its
own. Both of those are Gate E evidence.

The spec paragraph this implements is the last one of `AGENT-BUILD-SPEC.md` §5:
distil on Training Jobs as a separate revision; keep development/training,
held-out calibration and primary evaluation lineages disjoint; treat LoRA
`lr=1e-3, r=64, alpha=32` as one candidate configuration rather than a portable
optimum and compare a preregistered small search with early stopping on
development validation; treat training labels and compute as an additional
acquisition/budget item that the 50 held-out clips do not supply; preserve the
original judge's results.

`plumb/distillation.py` enforces every one of those as a refusal. This runbook
drives it. `deploy/baseten/training/` is the job.

**Status right now:** `not_run`. No Training Job has ever been submitted from
this repository, no adapter exists, no distillation dataset has been acquired,
and `results/gates.json` does not exist. Gate D has not passed either, so the
base judge this would distil from is not yet frozen.

---

## Prerequisites that are not commands

| Blocker | Why it blocks | Who resolves it |
|---|---|---|
| Gate D `pass` | Distillation needs a frozen base judge revision to distil *from* and to compare *against*. A distilled copy of an unqualified judge qualifies nothing. | `docs/GATE_D_RUNBOOK.md` |
| A **second** 150-clip panel | The 50 clips that qualified the original judge cannot re-qualify the distilled one. §5: obtain a fresh held-out set; do not tune against and reuse the same test set. | Acquisition |
| Training labels | An additional acquisition item. They are **not** supplied by the 50 held-out clips, and their cost is never recorded as zero. | Acquisition / budget |
| `BASETEN_API_KEY` | No job can be submitted without it. | Account |
| GPU capacity for training | Separate from the burst's world-model capacity, and a separate line in the cost report. | Account |
| The Training Jobs route | The exact route and accepted body are unconfirmed against <https://docs.baseten.co/training/overview>. Every rendered payload says so. | Step 5 |

Two decisions the project owner must make before step 1, because the code
refuses to guess them:

1. **Which student model.** `BUILD-SPEC.md` names
   `facebook/vjepa2-vitl-fpc64-256` (1.3 GB) for the distilled judge. That is a
   video encoder: LoRA on it yields a scalar/feature head, not the frozen rubric
   JSON schema. `plumb.distillation.StudentModel` requires an explicit
   `output_contract`, and a `scalar_head` revision is permanently barred from
   scoring a burst because the schema, enums and label mapping are part of the
   frozen judge protocol (§5). To *replace* the primary judge, the student must
   emit `rubric_json_schema` — a LoRA adapter over the pinned Qwen2.5-VL base.
   Pick one; a V-JEPA head remains a legitimate separate analysis.
2. **Where the training labels come from.** `original_judge_aggregate`,
   `original_judge_raw_samples`, `human`, or `external_label`. The first two are
   self-distillation from the teacher's own development-clip outputs; the third
   is a new annotation acquisition.

---

## Step 1 — Preregister the search, before any job runs

The two named arms are mandatory. `LoRASearchSpace` refuses a search that omits
either of them, refuses a selection split that is not development validation,
and refuses an early-stopping rule that watches a held-out split.

```bash
.venv/bin/python - <<'PY'
import json
from plumb.distillation import default_search_space

search = default_search_space(
    search_id="lora-search-v1",
    preregistered_at="2026-09-19T00:00:00+00:00",
)
print(json.dumps(search.as_mapping(), indent=2, sort_keys=True))
print("preregistration hash:", search.preregistration_hash())
PY
```

Three arms are rendered: the published `lr=1e-3, r=64, alpha=32, 2 epochs`
configuration, the policy-repository default `lr=5e-4, r=32`, and one
project-chosen interpolation. Each arm records its `provenance` and its
`unsourced_fields` — the hyperparameters this project chose rather than read from
a source. All three carry
`"portability": "candidate_configuration_not_a_portable_optimum"`.

Commit the hash to the same external record as `protocol.json`. The hash
`LoRASearchSpace.preregistration_hash()` returns is a **local content hash**, not
an independently timestamped preregistration; §6 requires the remote record
separately, and `as_mapping()` labels it
`preregistration_status: local_content_hash_only_no_external_timestamp`.

**Fills:** nothing in `results/gates.json` yet. This step makes the search
auditable before it can be influenced by a result.

---

## Step 2 — Build the dataset, and let it refuse a leak

`DistillationDataset` cannot be constructed if a training or
development-validation source lineage also appears in the Gate-D held-out set, a
fresh held-out set, or the primary study. The decision is made by
`plumb.measurement.validate_source_lineage_leakage` — the same checker the
primary study uses — and only `pass` is accepted: an `unverifiable` result (a row
with no lineage ID) is an inability to establish the guarantee, not permission to
proceed.

```bash
.venv/bin/python - <<'PY'
import json
from plumb.distillation import (
    LineageExclusions, TrainingBudgetItem, dataset_from_calibration_manifest,
)

with open("data/calibration/manifest.json", encoding="utf-8") as handle:
    manifest = json.load(handle)
with open("data/calibration/heldout_lineages.json", encoding="utf-8") as handle:
    gate_d_heldout = json.load(handle)
with open("data/primary/lineages.json", encoding="utf-8") as handle:
    primary = json.load(handle)

dataset = dataset_from_calibration_manifest(
    manifest,
    dataset_id="distill-dataset-v1",
    exclusions=LineageExclusions(
        gate_d_heldout=tuple(gate_d_heldout),
        primary_study=tuple(primary),
    ),
    label_source="original_judge_aggregate",
    budget=TrainingBudgetItem(
        label_count=100,
        label_source="original_judge_aggregate",
        usd_unavailable_reason="no verified price basis for the training accelerator yet",
    ),
    storage_uri="s3://plumb-distillation/dataset-v1/",
    frozen_at="2026-09-19T00:00:00+00:00",
)
print("dataset hash:", dataset.content_hash())
print(json.dumps(dataset.as_mapping(), indent=2, sort_keys=True))
PY
```

Only the **development** clips become training data; the held-out rows are
dropped *and* their lineages must still be declared in `exclusions`. Dropping
them from the training file is not the same as proving they never leaked, so an
undeclared held-out lineage is refused by name.

The split rule is deterministic: sorted development clip IDs, every fifth clip
validates. Early stopping must watch data the trainer never fits on.

**Fills:** `distilled_judge.training_data` (clip and lineage counts, the dataset
content hash, and the `lineage_disjointness` verdict).

---

## Step 3 — Check framework/model compatibility in the training container

§7 requires this in its own container. Nothing about the judge's serving image
transfers: the pins differ, the task differs, and `requirements.txt`'s header
lists exactly what is unverified.

```bash
# Build the training image from the pinned lock input and record its digest.
.venv/bin/python deploy/baseten/training/job_config.py --digest

# Inside the built container, load the base at its pinned revision and confirm
# TRL + peft accept the Qwen2.5-VL processor. This is the check whose result
# sets framework_compatibility_checked; it is not a training run.
python -c "import torch, transformers, peft, trl; print(torch.__version__, transformers.__version__, peft.__version__, trl.__version__)"
python -c "
import transformers
p = transformers.AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', revision='<40-hex>', local_files_only=True)
m = transformers.AutoModelForVision2Seq.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', revision='<40-hex>', local_files_only=True, torch_dtype='bfloat16')
print(type(p).__name__, type(m).__name__)
"
```

Until this succeeds, leave `--framework-compatibility-checked` off. Every
rendered payload then carries
`framework_compatibility: unchecked_until_the_training_container_builds` and lists
it as a submission blocker. Do not set the flag from a successful *import*; it
requires the model and processor to actually load.

**Fills:** `distilled_judge.revision` prerequisites; removes one entry from every
payload's `submission_blockers`.

---

## Step 4 — Render the payloads and validate them on CPU

```bash
.venv/bin/python deploy/baseten/training/job_config.py \
  --calibration-manifest data/calibration/manifest.json \
  --heldout-lineages data/calibration/heldout_lineages.json \
  --primary-lineages data/primary/lineages.json \
  --dataset-uri s3://plumb-distillation/dataset-v1/ \
  --dataset-id distill-dataset-v1 \
  --frozen-at 2026-09-19T00:00:00+00:00 \
  --preregistered-at 2026-09-19T00:00:00+00:00 \
  --search-id lora-search-v1 \
  --base-judge data/distillation/base_judge.json \
  --student data/distillation/student.json \
  --label-source original_judge_aggregate \
  --label-count 100 \
  --output results/distillation/payloads.json \
  --render
```

Split the file into one config per arm and check each on CPU. `--check` imports
no framework at all:

```bash
.venv/bin/python - <<'PY'
import json, pathlib
payloads = json.loads(pathlib.Path("results/distillation/payloads.json").read_text())
out = pathlib.Path("results/distillation"); out.mkdir(parents=True, exist_ok=True)
for payload in payloads:
    (out / "{0}.json".format(payload["search"]["arm_id"])).write_text(
        json.dumps(payload, indent=2, sort_keys=True))
    print(payload["search"]["arm_id"], payload["status"], payload["submission_blockers"])
PY

for arm in results/distillation/*-*.json; do
  .venv/bin/python deploy/baseten/training/train_judge_lora.py \
    --config "$arm" \
    --dataset-dir data/distillation/dataset-v1 \
    --excluded-lineages data/distillation/excluded_lineages.json \
    --check
done
```

`--check` refuses: a config whose recorded `lineage_disjointness` is not `pass`,
a config that selects on anything but development validation, an early-stopping
rule that monitors a held-out split, a dataset row carrying an excluded lineage,
a row without exactly 16 frame references, and a validation split that shares a
lineage with training.

**Fills:** nothing. It prevents three wasted GPU jobs.

---

## Step 5 — Submit one job per arm

Confirm the route and body against <https://docs.baseten.co/training/overview>
first. The payloads supply the values; they do not claim to know the route, and
nothing in this repository submits them.

```bash
export BASETEN_API_KEY='<from the Baseten dashboard>'

# Confirm the route and accepted body, then submit each arm separately.
# Each arm is its own job: they must not share an output directory, because the
# adapter digest and the development-validation metric are per-arm evidence.
for arm in results/distillation/*-*.json; do
  echo "submit $arm with the confirmed Training Jobs route"
done
```

Record the returned job ID for every arm, including the ones that fail. A failed
arm stays in the search record with its reason; an arm that vanishes turns the
preregistered search into a different, smaller search.

**Fills:** `distilled_judge.lora_search.results[].training_job_id`.

---

## Step 6 — Read back every arm, then select on development validation

Each finished job writes `arm_result.json` into its output directory, with the
metric, its source, the adapter URI and the adapter digest. Copy those numbers
in; do not retype them.

```bash
.venv/bin/python - <<'PY'
import json, pathlib
from plumb.distillation import ArmResult, default_search_space, select_arm, spoken_claim

search = default_search_space(
    search_id="lora-search-v1", preregistered_at="2026-09-19T00:00:00+00:00"
)
results = []
for path in sorted(pathlib.Path("results/distillation/arms").glob("*/arm_result.json")):
    raw = json.loads(path.read_text())
    results.append(ArmResult(
        arm_id=raw["arm_id"],
        status=raw["status"],
        selection_metric=raw["selection_metric"],
        metric_value=raw["metric_value"],
        metric_source=raw["metric_source"],
        training_job_id=raw["training_job_id"],
        adapter_uri=raw["adapter_uri"],
        adapter_sha256=raw["adapter_sha256"],
        epochs_completed=raw["epochs_completed"],
        early_stopped_at_epoch=raw["early_stopped_at_epoch"],
        reported_gpu_seconds=raw["reported_gpu_seconds"],
    ))

selection = select_arm(search, results)
print(json.dumps(selection.as_mapping(), indent=2, sort_keys=True))
print(json.dumps(spoken_claim(search, selection), indent=2))
PY
```

`ArmResult` refuses a result scored on any held-out split, a metric with no
source, and a `completed` arm with no adapter digest. `select_arm` refuses to
name a winner on a tie, reports `not_selectable` when no arm completed, and flags
`selected_with_gaps` when an arm is missing from the record.

`spoken_claim` is the demo guard. `BUILD-SPEC.md` lists "We used your published
LoRA optimum — learning rate 1e-3, r=64, alpha 32, two epochs — rather than the
repository default" as a verbatim line. That sentence is only sayable if the
published arm actually won here; if it lost, `spoken_claim` returns the honest
wording instead, which names the arm that won and says why the published numbers
are not presented as a portable optimum. Say what it returns.

**Fills:** `distilled_judge.lora_search` — every arm including losers and
failures, the ranking, the winner, and `spoken_claim`.

---

## Step 7 — Register the distilled revision

A new revision with its own identity hash. It refuses to reuse the base judge's
revision ID or its results file, and it refuses a different rubric hash: the
rubric is frozen, so a distilled judge scoring a different rubric is a different
measurement.

```bash
.venv/bin/python - <<'PY'
import json
from plumb.distillation import (
    BaseJudgeRevision, DistilledJudgeRevision, StudentModel,
    frozen_sampling_for_distilled_judge,
)

revision = DistilledJudgeRevision(
    revision_id="qwen2.5-vl-7b-distilled-rev-b",
    base_judge=BaseJudgeRevision(**json.load(open("data/distillation/base_judge.json"))),
    student=StudentModel(**json.load(open("data/distillation/student.json"))),
    adapter_uri="s3://plumb-distillation/adapters/<winning-arm>/",
    adapter_sha256="sha256:<from arm_result.json>",
    training_dataset_hash="sha256:<from step 2>",
    search_arm_id="<winning arm>",
    search_preregistration_hash="sha256:<from step 1>",
    training_job_id="<job id>",
    rubric_hash="sha256:<the frozen rubric hash, unchanged>",
    sampling=frozen_sampling_for_distilled_judge(),
    results_ref="results/judge_calibration_distilled_rev_b.json",
    created_at="2026-09-19T00:00:00+00:00",
)
print("identity hash:", revision.identity_hash())
print(json.dumps(revision.as_mapping(), indent=2, sort_keys=True))
PY
```

If the distilled judge needs a different sample count, pass it through
`frozen_sampling_for_distilled_judge(sample_count=..., quorum=...)`. Changing the
count without refreezing the quorum is refused: §5 requires the count, quorum,
schema and cost model to be frozen together.

This revision **invalidates Gates D, E and F** (`GateLedger.invalidate_for_change("judge", ...)`).
It does not inherit the base judge's Gate D.

**Fills:** `distilled_judge.revision`, `.original_judge_results_ref`,
`.distilled_judge_results_ref`.

---

## Step 8 — Fresh held-out calibration

A second full 150-clip panel, annotated by two blinded humans, with the distilled
judge's five-sample reports bound to a frozen evidence manifest of its own. All
of the statistics come from `plumb.calibration.build_calibration_report` and the
threshold decision from `plumb.calibration.evaluate_gate_d`; nothing is
recomputed.

```bash
.venv/bin/python - <<'PY'
import json
from plumb.calibration import FrozenGateDProtocol, GateDTolerances
from plumb.distillation import BaseJudgeRevision, run_fresh_heldout_calibration

result, report = run_fresh_heldout_calibration(
    calibration_id="fresh-heldout-v1",
    fresh_manifest=json.load(open("data/calibration/manifest_fresh.json")),
    original_manifest=json.load(open("data/calibration/manifest.json")),
    human_annotations=json.load(open("data/calibration/fresh_human_labels.json")),
    annotator_ids=("annotator-alex", "annotator-blair"),
    distilled_judge_annotations=None,
    distilled_judge_reports=json.load(open("results/distillation/fresh_judge_reports.json")),
    protocol=FrozenGateDProtocol(
        protocol_hash="sha256:<the DISTILLED judge's own frozen protocol hash>",
        rubric_hash="sha256:<unchanged rubric hash>",
        sampling_hash="sha256:<distilled sampling hash>",
        evidence_manifest=...,   # FrozenJudgeEvidenceManifest over the fresh clips
    ),
    tolerances=GateDTolerances(50, 1.0, 0.9, 0.9, 0.9, 0.9, 0.1),
    base_judge=BaseJudgeRevision(**json.load(open("data/distillation/base_judge.json"))),
    report_ref="results/judge_calibration_distilled_rev_b.json",
)
print(json.dumps(result.as_mapping(), indent=2, sort_keys=True))
PY
```

Freeze those tolerances before the fresh held-out labels are exposed. They are
not required to equal Gate D's.

What this refuses, and why each refusal exists:

| Refusal | Reason |
|---|---|
| The fresh panel *is* the original panel (same manifest hash) | Reusing the set that qualified the original judge is tuning against the test set. |
| A relabelled panel with the original **lineages** | New clip IDs over the same source states is the same test set wearing a hat. `validate_source_lineage_leakage` catches it. |
| A panel reusing original held-out **clip IDs** | Same. |
| The distilled judge calibrated under the original judge's `gate_d_protocol_hash` | A new judge revision needs its own frozen protocol, not the one it replaces. |

If it fails, revise and obtain **another** fresh panel. Do not re-expose this one
and present it as untouched evidence.

**Fills:** `distilled_judge.fresh_heldout_calibration` — status, `passed`, both
manifest hashes, the disjointness report, and the Gate-D metrics
(`sensitivity`, `specificity`, `leniency_offset`, coverage). Also
`results/judge_calibration_distilled_rev_b.json`, a new file; the original
`results/judge_calibration.json` is never touched.

---

## Step 9 — Paired frozen-video comparison

Both judges score the identical clip set with the identical frames, and the
comparison is paired clip by clip. The interval comes from
`plumb.annotation.paired_difference_interval`.

```bash
.venv/bin/python - <<'PY'
import json
from plumb.distillation import (
    FrozenVideoScore, PairedComparisonTolerances, compare_frozen_videos,
)

def scores(path):
    return tuple(FrozenVideoScore(**row) for row in json.load(open(path)))

comparison = compare_frozen_videos(
    comparison_id="paired-frozen-video-v1",
    original=scores("results/distillation/original_judge_scores.json"),
    distilled=scores("results/distillation/distilled_judge_scores.json"),
    tolerances=PairedComparisonTolerances(
        minimum_paired_clips=40,
        maximum_absolute_binary_difference=0.1,
        maximum_progress_disagreement_rate=0.2,
        minimum_paired_coverage=0.9,
    ),
)
print(json.dumps(comparison.as_mapping(), indent=2, sort_keys=True))
PY
```

Freeze those tolerances before computing the comparison; the result records their
hash. Re-scoring the original judge here is a *new* run of the original judge on
a frozen clip set — it produces a new artifact and does not overwrite its
calibration results.

Refused: a clip present for one revision and absent for the other; a clip whose
video digest or 16 frame digests differ between the revisions (then they did not
see the same video); and a revision compared against itself.

An unevaluable clip on either side is an unevaluable *pair*, counted in coverage
rather than dropped. Thin coverage fails the gate instead of being averaged away.

**Fills:** `distilled_judge.paired_frozen_video_comparison` — paired counts,
`binary_agreement_rate`, `positive_only_*`, the paired Wald interval, progress
disagreement rate, the clip-set hash and the tolerances hash.

---

## Step 10 — Assemble the Gate E evidence

```bash
.venv/bin/python - <<'PY'
import json
from plumb.distillation import DistillationGate, assert_scoring_authorised
from plumb.gates import GateLedger, GateRecord

gate = DistillationGate(
    revision=...,            # step 7
    search=...,              # step 1
    arm_results=...,         # step 6, every arm
    selection=...,           # step 6
    dataset=...,             # step 2
    fresh_calibration=...,   # step 8
    paired_comparison=...,   # step 9
    requested_for_scoring=True,
)
print(json.dumps(gate.authorisation(), indent=2))
assert_scoring_authorised(gate)     # raises, listing every blocker

ledger = GateLedger.load("results/gates.json")
existing = ledger.records["E"]
measurements = dict(existing.measurements)
measurements.update(gate.gate_e_measurements_patch())
ledger.record(GateRecord.from_mapping({**existing.to_mapping(), "measurements": measurements}, "E"))
ledger.save("results/gates.json")

record = ledger.records["E"]
print("Gate E blocking errors:", record.pass_evidence_errors())
PY
```

`gate_e_measurements_patch()` returns exactly `{"distilled_judge": {...}}` and
touches nothing else in Gate E. Merge it into whatever Gate E already holds.

`DistillationGate` is the only sanctioned producer of that mapping.
`used_for_scoring` is never emitted as `true` while a blocker stands, so setting
`requested_for_scoring=True` on an unvalidated revision produces
`used_for_scoring: false` with `scoring_request_refused: true` and the blockers
listed — not a lie that Gate E has to catch later. Call
`assert_scoring_authorised` immediately before any burst that names a distilled
judge: Gate E catches a dishonest ledger after the fact, but that stops the run.

When no distilled judge exists, record
`plumb.distillation.unvalidated_distilled_judge_evidence()` rather than leaving
the field absent. Gate E's rule is conditional — a distilled judge that is not
used for scoring passes trivially — and that is the correct state today.

**Fills:** `results/gates.json` → `gates.E.measurements.distilled_judge`.

---

## Which `results/gates.json` fields this fills

Everything below lives under `gates.E.measurements.distilled_judge`.

| Field | Source |
|---|---|
| `used_for_scoring` | `DistillationGate.evidence()`. The key `plumb/gates.py` Gate E reads. False unless requested **and** unblocked. |
| `separately_validated` | The other key Gate E reads. True only when the fresh calibration and the paired comparison both exist and both pass. |
| `scoring_request_refused`, `blockers` | Why a requested revision was not authorised. |
| `revision` | Step 7 — identity hash, base judge, student, adapter digest, dataset hash, arm, sampling hash, `invalidates_gates: [D, E, F]`. |
| `original_judge_results_preserved`, `original_judge_results_ref`, `original_judge_results_sha256` | Step 7. The original artifact and its digest, unmodified. |
| `distilled_judge_results_ref` | Step 7. A different file, enforced at construction. |
| `training_data` | Step 2 — clip/lineage counts, dataset content hash, exclusion counts, `lineage_disjointness`. |
| `lora_search` | Steps 1 and 6 — arms, preregistration hash, early stopping, every recorded result, the selection, `spoken_claim`. |
| `fresh_heldout_calibration` | Step 8 — status, `passed`, both manifest hashes, disjointness, Gate-D metrics. |
| `paired_frozen_video_comparison` | Step 9 — paired counts, intervals, clip-set and tolerances hashes. |
| `training_budget_item` | Step 2 — `budget_item: judge_distillation`, label count and source, GPU seconds, `usd_status`. |

Adjacent fields this runbook does **not** fill: `analysis_protocol_frozen`,
`planned_episodes`, `terminal_episodes`, `exclusion_sensitivity` and
`cost_setting_confirmation` are Gate E's other requirements and come from the
measurement and cost-sweep work.

The cost line also belongs in `results/economics.json`, as a *prior* cost:

```python
from plumb.sweeps import GpuSecondModel
model = GpuSecondModel(prior_cost_reports=budget.as_prior_cost_report())
model.prior_costs()   # {"status": "reported", "entries": {"judge_distillation": <usd or None>}}
```

`TrainingBudgetItem` refuses `estimated_usd=0.0`, refuses an amount without a
`cost_basis_ref`, and refuses an absent amount without a
`usd_unavailable_reason`. Distillation is never free and never folded into the
burst's marginal execution estimate.

---

## Acceptance thresholds (frozen before evaluation)

| Threshold | Where it lives | Set before |
|---|---|---|
| Search selection metric, direction, split | `LoRASearchSpace` | Any job runs (step 1) |
| Early stopping patience, `min_delta`, max epochs | `EarlyStoppingRule` | Any job runs (step 1) |
| Fresh held-out sensitivity / specificity / leniency / coverage | `GateDTolerances` | The fresh held-out labels are exposed (step 8) |
| `minimum_paired_clips`, `maximum_absolute_binary_difference`, `maximum_progress_disagreement_rate`, `minimum_paired_coverage` | `PairedComparisonTolerances` | The comparison is computed (step 9) |

No numeric default is supplied for any of them. Every value above is a project
decision that must be recorded before it can be influenced by a result.

---

## What distillation cannot establish

- **That the published LoRA configuration is optimal.** The search compares three
  arms on one dataset with one student. A winner here is a winner here.
- **That the distilled judge is cheaper.** No GPU-second or dollar figure exists
  until a job runs and its allocation ledger is read. `reported_gpu_seconds` is
  `None` until then, and `None` means unknown.
- **That the distilled judge may replace the original.** Passing a fresh
  calibration and a paired comparison authorises it to score a burst. It does not
  retroactively validate the original judge's results, which stay exactly where
  they are, nor does it transfer Gate D.
- **Task-specific accuracy.** The fresh panel is 10 clips per task. Pooled
  metrics are the gate; per-task numbers are exploratory, with intervals.
- **Anything about the training container.** Until step 3 succeeds,
  `framework_compatibility` reads
  `unchecked_until_the_training_container_builds`, and that is not pessimism —
  nothing has been built.

---

## Remaining external dependencies

1. Gate D must pass, giving a frozen base judge revision to distil from.
2. A second, genuinely independent 150-clip panel must be acquired, with
   lineages disjoint from the first panel and from the primary study.
3. Training labels must be acquired and their cost recorded as a separate budget
   line.
4. Baseten credentials and training GPU capacity.
5. The Training Jobs route and accepted request body must be confirmed against
   the vendor reference; every rendered payload currently says
   `route_status: unconfirmed_against_training_jobs_reference`.
6. The student-model decision in "Prerequisites" — a `scalar_head` revision can
   never score a burst under the frozen rubric schema.

None of these is a scope cut.
