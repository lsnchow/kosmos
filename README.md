# PLUMB

A working control plane and measurement console for world-model robot-policy
evaluation. The full scientific scope remains in [AGENT-BUILD-SPEC.md](AGENT-BUILD-SPEC.md).
Engineering fixtures, real model smoke tests, and qualified policy evaluations
are distinct modes; none is silently substituted for another.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test,analytics]'
npm --prefix web install
npm --prefix web run build
.venv/bin/plumb serve --port 8787
```

Open <http://127.0.0.1:8787>. API documentation is at `/docs`. The default
listener is localhost, not a public service. Mutable run data lives in `data/`
(or `PLUMB_DATA_DIR`) and is excluded from source control.

The orange button executes the full 6 × 5 × 50 **synthetic engineering fixture**
matrix. It exercises real persistence, cancellation, event delivery, artifact
serving, and analysis, but its images/labels are not learned-model or robot results.
The console does not fabricate Baseten queue depth, cost, or qualification.

```bash
.venv/bin/plumb run --starts 50 --data-dir data
.venv/bin/plumb reference
.venv/bin/python -m pytest -q
npm --prefix web run test
```

The Python suite includes an undefined-name/syntax scan of production sources;
GPU tensor checks skip on the lightweight laptop environment and run in the
cluster ML environment. Frontend checks include the baseline-UI and accessibility
constraints used for the real-evidence controls and report links.

## Implemented components

- Transactional SQLite run/episode/attempt ledger, idempotency, bounded execution,
  cancellation, owner leases and orphan recovery; immutable attempt artifacts.
- JSON-safe measurement reports: exact published reference, Wilson intervals,
  coverage and missing-outcome bounds, paired tests, MMRV, lineage checks, and
  explicit opt-in reliability/power calculations. Unsupported inference remains
  unavailable or indeterminate rather than being approximated under a false name.
- React dashboard with 12 rollout tiles, uncertainty/reference table, gate and
  telemetry views, full-matrix fixture launch, cancellation, and bounded unscored
  free-play. The cost slider remains empty until actual sweep evidence exists.
- Lazy local-only Cosmos forward-dynamics wrapper and smoke CLI; stateful Bridge
  action compiler, separate IRASim contract, and protocol/scenario gate validation.
- Actual local-only OpenVLA and Qwen loaders, original IRASim one-step runtime,
  and a 16-tick cluster closed-loop diagnostic with auditable feedback hashes.
  Real clips are displayed separately from synthetic fixture episodes. The first
  loop shows severe visual drift; it is not a qualified policy evaluation.
- Source-backed Octo, MiniVLA, SuSIE/SuSIE_LL, and OpenPiZero adapter paths.
  Artifact, license, state-conversion and execution-wrapper blockers remain
  explicit; these are not six validated live policies. See the
  [MiniVLA/SuSIE](docs/POLICY-DEPENDENCIES.md) and
  [OpenPiZero](docs/OPENPI-DEPENDENCIES.md) dependency records.
- Baseten async Chain client with verified-only queue routing, callback checking,
  resource-price unit validation, and a deployment topology requiring real model
  contracts before it can execute.
- Durable Baseten submission outbox, signed callback storage, request association,
  crash recovery, and cancellation tracking. Callback ingestion is opt-in;
  the local application never automatically submits cloud work.
- Provenance-checked scenario import, immutable calibration selection, blinded
  annotation packets, and full-study planning with run-specific episode IDs.

## Scenario, calibration, and study workflows

```bash
.venv/bin/plumb scenarios --help
.venv/bin/plumb annotation --help
.venv/bin/plumb study --help
.venv/bin/plumb distillation --help
```

The [scenario/calibration workflow](SCENARIO-CALIBRATION-WORKFLOW.md)
documents local source manifests, finite Bridge states, split isolation,
opaque annotation packets, and independent human ratings. The
[study workflow](docs/STUDY-WORKFLOW.md) freezes all 1,500 planned episodes,
checks readiness, prepares cost/drift comparisons, and validates three fresh
full-matrix rehearsals. These tools require supplied evidence; they do not
create the missing panels, human labels, or scientific results.

The [distillation workflow](docs/DISTILLATION-WORKFLOW.md) prepares immutable
development-only training/validation inputs and checks launch and fresh
calibration prerequisites. It does not submit a training job or claim a trained
judge exists.

The [Baseten delivery guide](deploy/baseten/README.md#durable-application-delivery)
describes the durable state machine and optional signed callback endpoint.
`GET /api/baseten/outbox` shows delivery state without raw payloads or secrets.
Callback receipt alone does not finalize or score an episode.

Real GPU validation is tracked in [HANDOFF.md](HANDOFF.md). A successful model
download/import or fixture call does not certify feedback fidelity, the five-task
reference comparison, the human judge calibration, or the one-minute cost target.

## Cluster use

See [cluster/README.md](cluster/README.md). All large assets and ML runtime
dependencies are downloaded directly to `/scratch/lchow432/plumb` on Trillium.
The laptop holds source, lightweight CPU dependencies, the frontend, and small
diagnostic report/media copies only. No model weights are stored on the laptop.

- `cluster/download_assets.py`: pinned size plan, explicit execution and size cap,
  file SHA-256 verification and provenance manifest.
- `cluster/setup_runtime.sh`: Alliance modules/wheels and pinned Diffusers runtime.
- `cluster/fetch_fixtures.py`: the small immutable official Bridge smoke fixture.
- `cluster/smoke.sbatch`: single-H100 real-inference smoke, offline on compute.

Do not execute GPU jobs on the login node. Keep SLURM allocations inside the
**cluster-side** `drac` tmux session so laptop network changes cannot end them.
Do not copy models back to the laptop or infer successful qualification from the
synthetic UI. Baseten credentials are not bundled or printed.

## Remaining external evidence

The real study still needs matched five-task scenario panels, certified native
feedback/state handling, two human annotators and held-out labels, actual cost
and drift sweeps, Baseten capacity/account configuration, and the missing original
reverse-validation question/script. Their absence does not remove them from scope.
