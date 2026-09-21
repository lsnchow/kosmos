# Baseten topology

This directory defines a typed Baseten Chains topology for Kosmos. It separates
policy inference, world-model generation, deterministic validation, VLM judging,
and durable result delivery so incompatible model runtimes do not share one
container.

```text
Rollout controller
  ├── policy worker
  ├── world-model worker
  ├── validation worker
  └── VLM judge worker
          ↓
   signed callback + durable local outbox
```

`chain.py` contains the deployment graph, pydantic wire contracts, batching
planner, and autoscaling hypotheses. `model-contracts.json` identifies the
evidence required before an individual stage can execute. The deployment design
has been validated against the SDK surface, but it is not evidence of a pushed
Baseten deployment, active replicas, measured throughput, or policy quality.

Use a dedicated deployment environment only after filling the required account,
model, and runtime evidence. Credentials and deployment secrets belong in
environment or provider secret storage, never in this repository.
