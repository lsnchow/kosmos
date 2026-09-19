# Full-study workflow

`plumb.study` is an offline contract for the requested primary study. It does
not create scenarios, invoke a model, estimate a cost, change gate status, or
write cloud records. A local plan hash binds JSON content; it is not an
independent preregistration timestamp.

## Freeze a primary plan

First create a scenario manifest with `plumb.scenarios`. The plan builder
requires its self-validating, `real_robot`, `status: "frozen"` manifest:

- five task groups;
- exactly 50 primary starts per task, shared by the six policies;
- nonempty disjoint `development`, `heldout_calibration`, and
  `cost_confirmation` panels for every task;
- the canonical policy key `Octo`, whose immutable policy identity explicitly
  says `Octo-Small v1.0`.

The minimal protocol metadata shape is:

```json
{
  "protocol_id": "plumb-primary-v1",
  "status": "frozen",
  "remote_record": "https://immutable-record.example/protocol-v1",
  "feedback_mode": "native_feedback",
  "policy": {"wrapper_family": "released"},
  "world_model": {"name": "pinned backend"},
  "judge": {"name": "pinned five-sample judge"},
  "thresholds": {"minimum_scientific_coverage": 0.95},
  "policy_revisions": {
    "OpenVLA": {"asset_manifest_id": "...", "adapter_hash": "sha256:..."},
    "OpenPiZero": {"asset_manifest_id": "...", "adapter_hash": "sha256:..."},
    "Octo": {"display_name": "Octo-Small v1.0", "asset_manifest_id": "...", "adapter_hash": "sha256:..."},
    "MiniVLA": {"asset_manifest_id": "...", "adapter_hash": "sha256:..."},
    "SuSIE": {"asset_manifest_id": "...", "adapter_hash": "sha256:..."},
    "SuSIE_LL": {"asset_manifest_id": "...", "adapter_hash": "sha256:..."}
  },
  "world_model_revision": {
    "asset_manifest_id": "...", "backend_profile_hash": "sha256:...",
    "model_revision": "...", "code_revision": "...", "container_digest": "..."
  },
  "judge_revision": {
    "asset_manifest_id": "...", "model_revision": "...", "processor_revision": "...",
    "runtime_lock_hash": "sha256:...", "rubric_hash": "sha256:...", "sampling_hash": "sha256:..."
  }
}
```

Run:

```sh
plumb study plan --scenario frozen-scenarios.json --protocol protocol.json --base-seed 20260919 > plan.json
```

The output contains exactly 1,500 stable plan slots. Policy and judge streams
are policy-specific; the world seed is deliberately identical for the six
policies at a matched task/start. `materialize` gives every actual run a new,
run-namespaced ledger `episode_id` while retaining `plan_slot_id`:

```sh
plumb study materialize --plan plan.json --run-id run-unique-id > rows.json
plumb study materialize --plan plan.json --run-id rehearsal-1 --cohort burst_rehearsal > rehearsal-rows.json
```

## Readiness

```sh
plumb study readiness --plan plan.json --scenario frozen-scenarios.json --gates results/gates.json
```

`ready` requires actual, non-synthetic A--D records whose protocol hashes and
revision bindings exactly equal the plan. It checks all 250 Gate-C task/start
identities, all six Gate-B wrapper bindings, the world binding in Gate A, and a
passed provenance-bound held-out calibration report plus judge binding in Gate
D. The scenario content hash is recomputed, not trusted from its declared
field. Gate E is deliberately not an execution prerequisite because it depends
on the completed primary matrix; Gate F is a later demonstration claim gate.

Additional cohort rows may be supplied as a JSON object to `--cohorts`. Every
row needs `start_lineage_id` (or `source_lineage_id`). Any lineage crossing a
development, calibration, primary, cost-confirmation, or supplied cohort
blocks readiness.

## Cost/fidelity and drift schedules

An operating-point row is:

```json
{
  "operating_point_id": "256-steps20-native",
  "resolution": [256, 256], "diffusion_steps": 20,
  "backend_profile_id": "same-pinned-world-profile",
  "judge_profile_id": "same-frozen-judge-profile",
  "feedback_mode": "native_feedback",
  "comparison_kind": "native_cost_fidelity",
  "backend_supported_action_lengths": [1],
  "policy_cadences": {
    "OpenVLA": {"certified_execute_prefix": 1, "world_request_action_length": 1},
    "OpenPiZero": {"certified_execute_prefix": 1, "world_request_action_length": 1},
    "Octo": {"certified_execute_prefix": 1, "world_request_action_length": 1},
    "MiniVLA": {"certified_execute_prefix": 1, "world_request_action_length": 1},
    "SuSIE": {"certified_execute_prefix": 1, "world_request_action_length": 1},
    "SuSIE_LL": {"certified_execute_prefix": 1, "world_request_action_length": 1}
  }
}
```

All candidates in a cost design must keep backend profile, judge profile, and
feedback mode equal. Every request ends at an independently certified native
execute/requery boundary; the backend must explicitly support every resulting
length, including a terminal remainder. OpenVLA is therefore 70 separate
fresh-observation/world calls for the 70-tick tasks unless independent evidence
certifies a different native boundary. No 23-action or old five-chunk request
can be labelled native feedback. Resolution and diffusion steps may vary only
with this cadence held fixed. Development and confirmation starts must include
all five tasks and be disjoint from each other and primary source lineages.

```sh
plumb study cost-design --plan plan.json --development-starts dev.json \
  --confirmation-starts confirmation.json --operating-points points.json
```

The 1--6 request display is a distinct, unqualified transport/horizon arm:

```sh
plumb study transport-design --plan plan.json --starts transport-starts.json \
  --operating-points transport-points.json
plumb study drift-design --plan plan.json --heldout-sequences drift.json \
  --operating-points transport-points.json
```

Transport points use `comparison_kind: "transport_sensitivity"`, an
unqualified feedback mode, and `request_count` from 1 through 6. Their output
does not invent policy query boundaries and cannot select a native cost/fidelity
setting. `drift.json` rows need `task`, `sequence_id`, `source_lineage_id`,
`actions_hash`, `video_hash`, and `recorded_control_ticks` (at least the task
horizon); `control_timestamps_hash` is optional. The emitted teacher-forced
and free-running recorded-action replay arms have zero policy queries and no
inferred policy feedback boundaries. They share source actions, generation
seed, full horizon, and operating point; they schedule measurements only.

## Burst evidence

Validate three fresh records only after each has a complete final ledger:

```sh
plumb study validate-rehearsals --plan plan.json --rehearsals three-runs.json > rehearsal-report.json
plumb study claim --plan plan.json --scenario frozen-scenarios.json \
  --gates results/gates.json --rehearsal-report rehearsal-report.json
```

Each rehearsal object requires a distinct `run_id`, matching plan/protocol
hashes, `fresh_execution: true`, accepted/finalized timestamps, all 1,500
completed burst rows, unique platform request IDs, and fresh generation/judge
artifact references with SHA-256 and timestamps inside the run window. Valid
videos need a judge artifact. An evaluable outcome additionally needs a strict
boolean `binary_success`, `judge_status: "evaluable"`, five-sample/three-quorum
`judge_sampling`, and at least three agreeing samples. Non-valid or
unevaluable rows need an explicit reason. Each row must retain exact frozen
horizon/executed-action counts, seeds, and policy/world/judge revision bindings.
The cost object must bind a pricing snapshot and allocation ledger and state a
total demonstration-run USD amount as `estimated` or `settled`.

The target is only permitted when all three have <=60 seconds, <=$11.25 total
demonstration cost, frozen scientific coverage, no incomplete/service rows,
no cross-run reuse, Gates A--F passed with matching revisions, and Gate F binds
the exact rehearsal report hash and three run IDs. Estimated costs remain
estimated until billing reconciliation.

Scientific coverage is `V / 1500` (valid, quorum-backed evaluable outcomes),
not judge-attempt coverage. Every eligible valid video must have a judge
attempt as a separate requirement; a run with one judged valid output and
1,499 invalid or unknown records does not meet the target.

Missing panels, remote preregistration, real gate evidence, full execution,
human labels, and billing data are therefore blockers rather than defaults.
