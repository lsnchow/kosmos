# Architecture

Kosmos is a local evaluation control plane around four distinct model roles:

| Role | Input | Output | Boundary |
| --- | --- | --- | --- |
| Policy | Observation, task, policy-specific state | Action proposal | An action is not a task result. |
| World model | Scene and compatible actions | Predicted frames or video | A visual prediction is not real-world physics. |
| Validation | Stored media and deterministic checks | Validity/provenance record | Invalid or missing media is retained, not silently discarded. |
| VLM judge | Sampled frames, task, rubric, allowed reference | Structured judgment | A model judgment needs external calibration before it becomes a success label. |

The FastAPI application owns run admission, durable SQLite state, event streams,
idempotency, cancellation, artifact serving, and result recovery. The React
console reads persisted state; opening a page does not create an inference job.

## Execution paths

- **Local engineering path:** the `kosmos` CLI and FastAPI service exercise the
  durable ledger with synthetic fixtures and saved artifacts.
- **Cluster diagnostics:** pinned GPU scripts execute individual policy,
  world-model, and judge probes. Their reports bind source releases, model
  revisions, media hashes, and measured timings.
- **Baseten topology:** `deploy/baseten/chain.py` describes a typed,
  separately-scaled deployment graph. It is deployment code and remains distinct
  from local/cluster diagnostic evidence.

## Repository layout

```text
core runtime        control plane, adapters, evaluation, and statistics
web/               React/Vite console
cluster/           immutable-release GPU diagnostics and batch scripts
deploy/baseten/    typed Baseten Chains deployment topology
tests/             unit, contract, and integration coverage
```

See [Reproducibility](REPRODUCIBILITY.md) for execution commands and
[Evaluation](EVALUATION.md) for the current evidence boundary.
