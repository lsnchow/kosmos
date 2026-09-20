# Kosmos

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

Go to [Video gallery](http://127.0.0.1:8787/console) to watch saved model outputs.
**Saved runs** contains run history and measurements. The remaining pages are
grouped under **Tools & validation**, each with a plain-language description.
The gallery is experimental footage, not a completed six-policy comparison.
**New evaluation** now runs real OpenVLA/IRASim steps or opens manual steering.
Gallery **New interactive branch** starts a fresh prediction from a clip's last
image. See the [working live-demo flow and operating window](docs/LIVE-DEMO.md).
This is experimental image feedback, not exact hidden-state restoration or a
qualified success score.
Read the [architecture breakdown](docs/ARCHITECTURE.md) for the page map, model
roles, current execution paths, storage boundaries, and remaining product work.

The dashboard's **Review clips** link opens `/review`: opaque review
media and private, explicitly entered drafts. See the
[operator guide](docs/DEVELOPMENT-REVIEW.md). These development clips and
self-reported reviewers do not constitute held-out calibration or a Gate D pass.

The orange button executes the full 6 × 5 × 50 **synthetic engineering fixture**
matrix. It exercises real persistence, cancellation, event delivery, artifact
serving, and analysis, but its images/labels are not learned-model or robot results.
The console does not fabricate Baseten queue depth, cost, or qualification.

The real Baseten policy-action MVP is a separate, verified flow at
[/live#cloud-diagnostic](http://127.0.0.1:8787/live#cloud-diagnostic).
See [Baseten MVP operation and evidence](docs/BASETEN-MVP.md) for the deployed
identity and startup command. It does not turn synthetic runs into real rollouts
or enable uncalibrated judge scoring.

Previously generated world-model MP4s can also be watched in the
[recording archive](http://127.0.0.1:8787/clips).
See [video playback and provenance](docs/WORLD-VIDEO-PLAYBACK.md). Playback is
separate from submitting new generation jobs.

```bash
.venv/bin/plumb run --starts 50 --data-dir data
.venv/bin/plumb reference
.venv/bin/python -m pytest -q
npm --prefix web run test
```

The Python suite includes an undefined-name/syntax scan of production sources;
GPU tensor checks skip on the lightweight laptop environment and run in the
cluster ML environment.

`npm --prefix web run test` type-checks the dashboard and then runs its Vitest
suite under jsdom: tile frame accumulation and provenance labelling, burst
double-submit protection, the cost slider issuing no request on drag, scoreboard
intervals and indeterminacy, telemetry staleness, free-play generating and
no-backend states, called-shot pending state, event-stream reconnection, and an
axe-core pass over the rendered console. The axe pass disables the two rules
jsdom cannot evaluate or that do not apply (computed colour contrast, which has
no stylesheet or layout in jsdom, and video captions for silent generated
clips); palette contrast is instead fixed by measured ratios in `styles.css`.

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

## Testing the production path without a GPU

`scripts/rehearse.py` runs the **entire** production code path against a simulated
Chain that speaks the documented Baseten protocol: the real `/async_run_remote`
envelope, real HMAC-signed webhooks, real queue-status and deployment shapes.

```bash
.venv/bin/python scripts/rehearse.py --starts 2 --policies OpenVLA,MiniVLA
.venv/bin/python scripts/rehearse.py --starts 50      # the full 1,500-episode matrix
```

Submissions are validated against the Chain's own pydantic models and its own
`_world_setup`, so a field-name drift between the app and the Chain fails loudly
instead of silently nulling every measurement. Each run deliberately drops and
duplicates a fraction of its webhooks so the reconciler and the idempotency path
are exercised rather than assumed.

The Chain is **simulated**: no model runs, and no latency or cost figure from a
rehearsal is a measurement. Every episode it produces carries
`transport: "simulated"` on its ledger row and its artifact manifest, and that
manifest records `world_model: false`.

For overall readiness, including what is still waiting on hardware:

```bash
.venv/bin/python scripts/e2e_smoke.py                     # pass / pending / fail
.venv/bin/python scripts/e2e_smoke.py --require-qualified  # pending becomes failure
```

`docs/GO_LIVE.md` is the ordered sequence for the moment GPU and Baseten access
exist.

## Remaining external evidence

The real study still needs matched five-task scenario panels, certified native
feedback/state handling, two human annotators and held-out labels, actual cost
and drift sweeps, Baseten capacity/account configuration, and the missing original
reverse-validation question/script. Their absence does not remove them from scope.
