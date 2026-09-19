# Gate C runbook — scenario parity

Gate C blocks **five-task reference validation**. It needs five task-specific
start panels with real image/state provenance, goal references, an initial-state
distribution, and documented scene/control comparability against AutoEval.

Nothing in this runbook fabricates a start, a scene fact, or a comparability
claim. Every step either produces real evidence or fails and names what is
missing. Run the steps in order; each one is safe to re-run.

## Actual metadata audit — 2026-09-19

Only the README and four metadata files (6,007,767 bytes) were acquired on
Trillium from Bridge LeRobot revision
`0e9d76d07e9df3ea3eba257b2520d4913833fad2`; no episode/video was selected for the
study. The hash-bound local audit is
`data/bridge-metadata-0e9d76d/audit-v1.json`. The pinned index contains53,192
episodes,19,974 task strings and1,893,026 frames;14,532 episodes have no usable
language label. Exact case-insensitive benchmark-instruction matches:

| Task | Episodes | Qualification |
| --- | ---: | --- |
| open_drawer | 473 | Not reviewed for scene/reset parity |
| close_drawer | 416 | Not an AutoEval drawer-panel replacement |
| to_basket | 0 | No exact instruction match |
| to_sink | 0 | No exact instruction match |
| fold_cloth | 100 | Direction/scene/metric references still unreviewed |

Keyword searches are discovery only: for example, basket matches include
moving objects **out of** a basket, not the required task. Do not widen them
into eligibility rules merely to fill the matrix.

Published eight-dimensional state statistics also conflict with the provisional
0–0.39 gripper profile: source gripper extrema are0.0463782921–1.1121242046.
No scale, clamp, action/state interchange, or state-unit conversion is justified
by this aggregate range. A source-backed conversion and trajectory fixture are
still needed. All metadata counts are raw source facts, not successful or
independent eligible benchmark starts.

Reproduce this no-network/no-model audit against the downloaded metadata:

```bash
python -m cluster.audit_bridge_metadata \
  --dataset-root data/bridge-metadata-0e9d76d \
  --download-manifest data/bridge-metadata-0e9d76d/download.json \
  --revision 0e9d76d07e9df3ea3eba257b2520d4913833fad2 \
  --output data/bridge-metadata-0e9d76d/audit-new.json
```

The original planning steps below remain prerequisites for actual selection;
the primary asset lock/protocol are intentionally unchanged by this audit.

**Gate status:** `not_run`. The Bridge metadata revision above is pinned for
audit only; no episode has been downloaded and no scene record verified. Both source
datasets carry `unresolved_immutable_revision` in `cluster/asset_plan.py`, and
`zhouzypaul/auto_eval` additionally carries `unresolved_license_or_access_terms`.

---

## Prerequisites

| Requirement | Why | How to confirm |
|---|---|---|
| Cluster scratch directory `/scratch/<user>/plumb` | Source pulls are permitted only there | `cluster/download_assets.py` refuses any other root |
| Resolved commit SHA for `IPEC-COMMUNITY/bridge_orig_lerobot` | Revision pinning | `huggingface-cli` or the Hub API; record it in `cluster/asset_plan.py` |
| Resolved commit SHA **and** license/access terms for `zhouzypaul/auto_eval` | Its dataset card declares no license | Read the card, record the decision, then set the plan entry |
| A disposable, credential-free, network-free environment for pickle conversion | `auto_eval` pickles reference `robot_eval_logger` and `wandb` | A separate container or an unprivileged user with no tokens mounted |
| Frozen `protocol.json` and its SHA-256 | Gate C evidence binds to a protocol hash | `plumb freeze-protocol` then `plumb prereg-status` |

Set once per shell:

```bash
export PLUMB_ROOT=/scratch/$USER/plumb
export AUTOEVAL_REV=<40-hex commit for zhouzypaul/auto_eval>
export BRIDGE_REV=<40-hex commit for IPEC-COMMUNITY/bridge_orig_lerobot>
export PROTOCOL_HASH=$(python -c "import json;print(json.load(open('protocol.json'))['sha256'])")
```

---

## Step 1 — Print the acquisition plan (no network, no writes)

```bash
python cluster/build_scenarios.py plan \
  --manifest-id scenarios-v1 \
  --revision zhouzypaul/auto_eval=$AUTOEVAL_REV \
  --revision IPEC-COMMUNITY/bridge_orig_lerobot=$BRIDGE_REV \
  --primary-starts 50 --development-starts 10 \
  --calibration-starts 30 --cost-confirmation-starts 10 \
  --starts-per-episode 1 --min-source-frame-gap 10 \
  --root $PLUMB_ROOT --max-gb 40 \
  --output results/scenario_acquisition_plan.json --execute
```

**Produces** `results/scenario_acquisition_plan.json`: source dataset and pinned
revision per task, the language-instruction selection rule, cohort counts, the
distinct-episode requirement (100 per task with these counts), the exact
download commands, the pickle allowlist, and `open_dependencies`.

**Check before continuing:** every task's `revision_immutable` is `true` and
`open_dependencies` is empty. If a revision is unresolved, stop and resolve it.
The plan's `comparability` column must read `matched_provenance` for the two
drawer tasks and `matched_distribution_with_limitations` for basket, sink, and
cloth. Any other assignment is a bug, not a result.

## Step 2 — Pull only the metadata indexes

```bash
python -m cluster.download_assets --from-plan bridge-orig-lerobot \
  --root $PLUMB_ROOT --allow 'meta/*' --execute
python -m cluster.download_assets --from-plan auto-eval \
  --root $PLUMB_ROOT --allow 'meta/*' --allow '*.json' --execute
```

**Produces** `$PLUMB_ROOT/evidence/<repo>-download.json` with per-file byte
length and SHA-256.

**Do not** drop the `--allow` patterns. `bridge_orig_lerobot` is 99,673 files
and `auto_eval` is a ~173 GB historical estimate; the ceiling exists to stop an
accidental whole-dataset pull.

## Step 3 — Select independent starts, one task at a time

For each of the five tasks:

```bash
python cluster/build_scenarios.py select \
  --task open_drawer \
  --index $PLUMB_ROOT/models/zhouzypaul--auto_eval/meta/episodes.jsonl \
  --revision zhouzypaul/auto_eval=$AUTOEVAL_REV \
  --revision IPEC-COMMUNITY/bridge_orig_lerobot=$BRIDGE_REV \
  --primary-starts 50 --development-starts 10 \
  --calibration-starts 30 --cost-confirmation-starts 10 \
  --starts-per-episode 1 --min-source-frame-gap 10 \
  --output results/starts-open_drawer.jsonl \
  --provenance results/provenance-open_drawer.jsonl \
  --execute
```

Repeat with `--task close_drawer` (same index), and with
`--task to_basket|to_sink|fold_cloth` against
`$PLUMB_ROOT/models/IPEC-COMMUNITY--bridge_orig_lerobot/meta/episodes.jsonl`.

**Produces** one start record and one provenance record per selected start.

**Exit code 2 means stop.** Two failure modes are expected and must not be
worked around:

- `insufficient_source_episodes` — fewer instruction-matched episodes than the
  cohort plan needs. Widen the pattern with `--pattern TASK=<regex> --regex`
  **and record the widened pattern**, or reduce the cohort counts. Never raise
  `--starts-per-episode` to manufacture starts from neighbouring frames.
- `fail` with `independence_violations` — the selection produced two starts that
  are the same physical start. Fix the index or the gap, do not relax the guard.

`content_hashes_resolved: false` in the provenance means the index did not carry
image/state/goal digests, so the start records hold placeholder hashes and a
`placeholder content hashes` note. **Those cannot support a Gate C pass.** Pull
the selected episode files and re-run `select` against an index that carries the
real digests.

## Step 4 — Convert `auto_eval` pickles under restricted loading

In the isolated environment only, once per pickle:

```bash
python cluster/build_scenarios.py convert-auto-eval \
  --pickle $PLUMB_ROOT/models/zhouzypaul--auto_eval/<episode>.pkl \
  --output $PLUMB_ROOT/converted/<episode>.json \
  --source-sha256 <sha256 from the download manifest> \
  --report $PLUMB_ROOT/evidence/convert-<episode>.json \
  --execute
```

**Produces** a schema-validated JSON episode (7-D actions, 8-D proprio, frame
count, AutoEval's own classifier label) plus a report with the source and output
digests.

The tool **refuses** rather than stubbing when the checkpoint unpickles
`robot_eval_logger`, `wandb`, or anything outside its reviewed allowlist. A stub
unpickler is not a security boundary. If it refuses, obtain a safe export from
upstream; do not extend the allowlist to make it pass.

AutoEval's classifier label is recorded as a **third-party external label** on
real drawer video. It is not a PLUMB human annotation and not five-task ground
truth.

## Step 5 — Verify the scene claims against source scene records

Three handoff claims are carried as `scene_claims` with `verified: false`:

| Claim | Scene |
|---|---|
| `drawer_taped_to_table` | drawer |
| `cloth_taped_to_table` | cloth |
| `sink_thin_plastic_wrap` | sink |

To mark one verified you must supply both a `source_record_uri` and a
`verification_method`; the dataclass refuses a bare `verified=True`. Pass
verified claims into `plumb.scenarios.build_manifest(scene_claims={...})`. Until
then Gate C reports `pass_with_limitations` per task and lists
`source_scene_record_verification` as an open dependency.

## Step 6 — Resolve the metric calibration reference (or leave it unresolved)

`Open the drawer` succeeds at `opened >= 1.5 cm`. That is a metric threshold, so
it needs a scene-specific calibrated reference — a known-size fiducial in frame
or measured drawer travel. **Do not estimate it from pixel counts.**

Until a reference exists, every start's `camera.calibration.status` stays
`unresolved`, `open_drawer`'s `calibration_reference_status` stays `unresolved`,
and `scene_metric_calibration_reference` is an open dependency. `fold_cloth`'s
quarter-diagonal rule likewise needs a scene record fixing the cloth extent and
the designated top-right corner.

To resolve: set `camera.calibration` to
`{"status": "resolved", "reference_uri": "...", "mm_per_pixel": ...}` on each
start and construct `SuccessCriterion` with
`calibration_reference_status="resolved"` and a `calibration_reference_uri`.

## Step 7 — Assemble and validate the manifest

```bash
python cluster/build_scenarios.py assemble \
  --manifest-id scenarios-v1 \
  --starts results/starts-open_drawer.jsonl \
  --starts results/starts-close_drawer.jsonl \
  --starts results/starts-to_basket.jsonl \
  --starts results/starts-to_sink.jsonl \
  --starts results/starts-fold_cloth.jsonl \
  --revision zhouzypaul/auto_eval=$AUTOEVAL_REV \
  --revision IPEC-COMMUNITY/bridge_orig_lerobot=$BRIDGE_REV \
  --evidence-uri file://$PWD/results/provenance-open_drawer.jsonl \
  --evidence-uri file://$PWD/results/scenario_acquisition_plan.json \
  --protocol-hash $PROTOCOL_HASH \
  --output results/scenarios.jsonl \
  --gate-evidence results/gate_c_evidence.json \
  --execute
```

**Produces**

- `results/scenarios.jsonl` — a header record plus one line per start, every
  start bound to the manifest SHA-256.
- `results/gate_c_evidence.json` — a bundle shaped for
  `plumb.gates.GateRecord.from_mapping`.

The manifest is validated by `plumb.gates.ScenarioManifestValidator` **before**
anything is written. `assemble` refuses to write on any validation error and
refuses to overwrite an existing manifest without `--overwrite`.

If only the drawer panels exist, add `--allow-incomplete`: the manifest is
written, `manifest_status` reads `incomplete`, and `unacquired_task_panels`
lists the missing tasks. That is the honest state, not a pass.

Re-verify the written file at any time:

```python
from plumb.scenarios import read_manifest_jsonl
print(read_manifest_jsonl("results/scenarios.jsonl")["status"])   # "pass"
```

## Step 8 — Record the gate

Merge `results/gate_c_evidence.json` into the ledger:

```python
import json
from plumb.gates import GateLedger, GateRecord
ledger = GateLedger.load("results/gates.json")
ledger.record(GateRecord.from_mapping(json.load(open("results/gate_c_evidence.json"))))
ledger.save("results/gates.json")
```

Then `plumb gates` prints every blocking evidence error.

---

## Which `results/gates.json` fields this fills

| Field | Source |
|---|---|
| `gates.C.status` | `blocked` while any dependency is open; `pass` only when every per-task status is `pass` |
| `gates.C.protocol_hash` | `--protocol-hash` |
| `gates.C.start_ids` | Every start ID in every cohort |
| `gates.C.evidence_uris` | `--evidence-uri` (plan, provenance, download manifests, conversion reports) |
| `gates.C.evidence_kind` | `real_source_frames` |
| `gates.C.measurements.starts_per_task` | Primary-cohort count per task (read by `GateRecord.pass_evidence_errors`) |
| `gates.C.measurements.comparability` | Per task: comparability, limitations, task status, unresolved axes, unverified claims |
| `gates.C.measurements.cohort_counts` | Per cohort per task |
| `gates.C.measurements.distinct_source_episodes`, `max_starts_per_source_episode` | Concentration of starts across episodes |
| `gates.C.measurements.lineage_separation_status` | `plumb.measurement.validate_source_lineage_leakage` result |
| `gates.C.measurements.camera_calibration_status` | Per start: `unresolved` or `resolved` |
| `gates.C.measurements.per_task_gate_status` | `pass` / `pass_with_limitations` / `blocked` per task |
| `gates.C.measurements.scenario_manifest_sha256` | Manifest hash; `protocol.json.scenario_manifest_hash` must match |
| `gates.C.thresholds` | The frozen thresholds below |
| `gates.C.reasons` | Validator errors plus every open dependency |

`GateStatus` has no `pass_with_limitations` member, so the richer per-task
statuses live in `measurements.per_task_gate_status`. The ledger status stays
`blocked` while any dependency is open.

## Acceptance thresholds (frozen before evaluation)

| Threshold | Value |
|---|---|
| `primary_starts_per_task` | exactly 50, shared across all six policies |
| `required_cohorts` | `primary`, `development`, `calibration`, `cost_confirmation` all non-empty |
| `lineage_cohort_crossings_allowed` | 0 |
| `verbatim_instruction_match_required` | true, including the lowercase `fold` in task 5 |
| `min_source_frame_gap` | 10 source frames between two starts from one episode |
| `max_starts_per_episode` | 1 with the recommended settings |
| repeated world seed / source image / source state | rejected as a new start |
| `scene_claims_must_be_verified_for_pass` | true |
| `calibration_reference_must_be_resolved_for_pass` | true |
| all five AutoEval comparison axes | `matched` with an `evidence_uri` |

A per-task `pass` needs all of: 50 primary starts, `matched_provenance`, every
comparison axis matched, every scene claim verified, and the success criterion's
calibration reference resolved.

## What Gate C cannot establish

- **No paired reconstruction.** AutoEval publishes aggregate per-cell tables.
  The original trial-level starting states are not released, so no start here
  reconstructs a specific original trial. Every panel carries that limitation.
- **Basket, sink, and cloth are matched-distribution only.** No AutoEval scene
  asset or scene record for those tasks has been acquired, so scene layout,
  object set, and reset/randomization distribution are matched by source-dataset
  language instruction alone. A per-cell discrepancy cannot be attributed to the
  world model until the five axes are resolved.
- **Do not fill those cells with drawer starts.** `plumb.scenarios.StartRecord`
  raises when a start's source scene family does not match its task, and
  `zhouzypaul/auto_eval` cannot be declared as a sink/cloth/basket source. The
  absence is an acquisition dependency.

## Remaining external dependencies

1. Resolved immutable commit SHA for both source datasets.
2. License and access terms for `zhouzypaul/auto_eval`.
3. Downloaded episode files, so start records carry real image/state/goal
   digests instead of placeholders.
4. Source scene records for the taped drawer, taped cloth, and sink plastic wrap.
5. A scene-specific metric calibration reference for the 1.5 cm drawer threshold
   and the cloth diagonal.
6. Evidence for the five AutoEval comparison axes: scene layout, objects,
   reset/randomization distribution, policy checkpoint/wrapper, and control
   timing / task horizon.
