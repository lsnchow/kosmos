# PLUMB Baseten deployment

`chain.py` is a complete, pushable Baseten Chains topology with real adapter
calls: a CPU `RolloutController` entrypoint, entrypoint-side `PolicyRouter`
dispatch, five per-stack policy Chainlets, a micro-batched H100 world Chainlet, a
CPU deterministic-validity Chainlet, and a GPU rubric-judge Chainlet. It deploys
nothing on import, contains no weights or credentials, and never fabricates a
frame, action, validity label, judge vote, container digest, or price.

**It has never been pushed.** This machine has no GPU and no Baseten credentials.
What *was* verified: `truss==0.18.30` was installed in a throwaway venv, `chain.py`
was imported with the SDK present, and
`truss_chains.framework.raise_validation_errors()` collected **no errors** — all
nine Chainlets construct, the entrypoint resolves its eight dependencies at
`retries=0`, and both endpoints validate as async with pydantic I/O. That checks
the SDK surface and this module's definitions. It is not a push, not an image
build, and not evidence about GPU behaviour, capacity, latency, or cost.
`DEPLOY.md` is the ordered runbook for the moment credentials go live.

## Files

| File | What it is |
| --- | --- |
| `chain.py` | The Chain. Nine Chainlets, typed pydantic wire contracts, the pure `MicroBatchPlanner`, and the autoscaling hypotheses. |
| `DEPLOY.md` | Ordered, copy-pasteable deploy runbook. Each step names the `model-contracts.json` nulls it fills. |
| `model-contracts.json` | Required cloud evidence. Every remaining `null` carries a `resolved_by` naming the exact command or step that supplies it. |
| `requirements/*.txt` | One installable lock input per incompatible dependency stack, re-resolved to public PyPI wheels and immutable git revisions. |
| `requirements.lock-inputs.txt` | Superseded; now an index into `requirements/` that preserves the original review gate. |
| `stage_packages.py` | Copies `plumb/**` into `../_chain_packages` for `DockerImage(external_package_dirs=...)`. Mandatory before a push. |
| `cluster-runtime-evidence.json` | Recorded ComputeCanada cluster diagnostics. Deliberately **not** a Baseten configuration file. |
| `../../tests/test_chain.py` | 160 tests, no GPU required: import guard, wire contracts, frame codec, routing table, autoscaling declarations, micro-batcher, controller helpers, requirements files. 159 run with `truss_chains` absent; the last runs the real SDK validator in the deploy venv. |

## Topology

```
RolloutController  (CPU entrypoint, external /async_run_remote queue)
  |- PolicyRouter  (entrypoint-side table lookup, not a Chainlet)
  |    |- OpenVLAWorker      H100  transformers 4.40.1 + torch 2.6.0
  |    |- OctoWorker         H100  JAX 0.4.20 (py3.10)        serves Octo-Small, Octo-Base
  |    |- MiniVLAWorker      H100  torch 2.2.0 + flash-attn 2.5.5
  |    |- OpenPiZeroWorker   H100  torch 2.6.0 + PaliGemma
  |    '- SusieWorker        H100  JAX/Flax SD                serves SuSIE, SuSIE_LL
  |- WorldWorker             H100  Cosmos3 Diffusers FD, micro-batched
  |- ValidityWorker          CPU   deterministic Stage A
  '- JudgeWorker             H100  Qwen2.5-VL, frozen 5-sample protocol
```

Every dependency uses `chains.depends(..., retries=0)`. A transport retry creates
an attempt, not a new statistical episode, and attempt accounting belongs to the
application ledger. Only the entrypoint uses the external async queue; internal
Chainlet calls are awaited RPCs.

## The three load-bearing design decisions

**Micro-batching is the throughput lever.** `AGENT-BUILD-SPEC.md` §7 says
throughput comes from batch packing and replica count. `MicroBatchQueue` collects
up to `BATCH_MAX` (default 16) requests arriving within `BATCH_WINDOW_MS`
(default 15 ms), groups them by a key covering everything that changes the
forward's tensor shapes or schedule, executes one group, and scatters results
back to the individual awaiting callers. `predict_concurrency` follows `BATCH_MAX`
so the collector can actually see concurrent callers. The grouping and scatter
logic is a pure, GPU-free class and is unit-tested directly.

One honest limitation: `Cosmos3NanoDiffusersAdapter` exposes only
`generate(request)`. A genuinely *fused* diffusion forward needs a batched
`CosmosActionCondition` path, so the default execution mode runs the collected
group as consecutive adapter calls and labels itself
`batch_execution_mode="sequential_adapter_calls"`. If `plumb` gains a reviewed
`generate_batch`, the worker uses it once `PLUMB_WORLD_FUSED_BATCH=1` and reports
`fused_adapter_generate_batch`. It never claims one forward when it made several.

**Weight caching, not warm hope.** `chains.Assets(cached=[...])` declares the
three models with immutable revisions (Cosmos3-Nano `e59a53c2…`, OpenVLA
`47a0ec7f…`, Qwen2.5-VL `cc594898…`) so replicas do not pay the recorded
167.68 s Cosmos cold load, which would be fatal to a 60-second burst. Policy
checkpoints without a resolved immutable commit are deliberately **absent** from
every cached list: `truss_config.ModelRepo` requires a revision, and inventing
one would forge provenance.

**Autoscaling is applied out of band.** `chains.RemoteConfig` in truss 0.18.30
has no autoscaling field — the only `min_replica` occurrence in the SDK is a
docstring. The values live in `chain.py::AUTOSCALING_HYPOTHESES`, are mirrored
into each Chainlet's `ChainletOptions.metadata`, and are emitted as management-API
bodies by `autoscaling_patch_payloads()`. That is the side
`plumb/platform.py` already reads back from `chainlet.autoscaling_settings`.

## What is deliberately blocked

| Stage | Result today | Why |
| --- | --- | --- |
| OpenVLA policy | `blocked` until `PLUMB_OPENVLA_REVIEWED_REMOTE_CODE_ACK` matches the reviewed revision | `trust_remote_code` must record a human review, not infer consent from a successful download. |
| Octo, MiniVLA, OpenPiZero, SuSIE, SuSIE_LL | `blocked`, naming the certified adapter and its missing evidence | `plumb.policies` ships real `*PolicyAdapter`s, but each needs provenance the Chain must not invent (converted-artifact digests, conversion-report hashes, immutable revisions, a pinned action normalizer). The Chain loads the non-fabricating hook and lists the exact missing profile fields. |
| World (Cosmos Edge arm) | `blocked` without `PLUMB_WORLD_EDGE_REVISION` | No immutable Cosmos3-Edge revision has been recorded. |
| Validity | **runs**, and reports `no_motion_reference` | `plumb.validity.StageAValidityGate` has landed and the Chain binds and calls it. With no `MotionReference` fitted from real Bridge trajectories the calibrated command-motion check cannot run, so the gate names that instead of equating low optical flow with invalidity. |
| Judge | `blocked` without provenance-backed reference panels and five distinct logged seeds | The worker will not invent a sampling lineage or a goal panel. |
| Controller loop | `blocked` when `certified_execute_prefix` is null | Only OpenVLA declares one. The loop never assumes five chunks, never truncates, never pads. |
| `RolloutResult.qualified` | hard-coded `false` | Gates A–D are external reviews in `plumb.gates`, never inferred from a call that returned. |

## Recorded local diagnostics, not cloud qualification

`cluster-runtime-evidence.json` records completed ComputeCanada/Queen's GPU
diagnostics. **They do not transfer to any image defined here.** HANDOFF.md pins
Alliance local wheel builds (`torch 2.6.0+computecanada`,
`tokenizers 0.19.1+computecanada`, `transformers 5.17.0+computecanada`) whose
version strings do not exist on PyPI, so every Baseten image installs a different
binary build of the same upstream version: different kernels, CUDA minor
versions, attention backends, and allocator behaviour. Each file under
`requirements/` states this in its header, and every number below must be
re-measured on Baseten before it is quoted.

| Local diagnostic | What was observed | Still blocked |
| --- | --- | --- |
| Cosmos3-Nano / Diffusers, job 937372 | 16 actions → 17 frames; 4.524269459 s inference; 167.683719175 s model load; 36,521,475,584 B peak | Baseten image/serializer, Gate A/B, cloud capacity and cost |
| OpenVLA, job 937500 | one native 7-D `bridge_orig` action; 0.835902354 s; 15,500,827,648 B peak | deployed policy profile and native feedback qualification |
| IRASim Frame-Ada, job 937666 | one 7-D scaled action → two frames; 2.546707168 s; 23.114047749 s load; 3,516,738,560 B peak | action/frame fidelity and all cloud configuration |
| Qwen judge, job 12238810 | five raw samples; 29.291511048 s; 16,948,912,640 B peak; unknown/insufficient quorum | human calibration, Gate D, judge deployment profile |

These are cluster diagnostics only. The IRASim 18-case action probe and 16-tick
closed-loop trace remain experimental and unqualified. The completed 937751
state-representation replay is qualitative: its pair and repeat hashes are
reproducible, but both branches show gripper drift, it never requeried a policy,
and it scored no task success. IRASim is **not** wired as a Chainlet: it has its
own 256×320 / 16-frame / 15×7-D contract, `IRASimBridgeAdapter` reports
`UNSUPPORTED` with no reviewed native loader, and an IRASim smoke pass cannot
clear Cosmos's causal-feedback failure.

### Evidence consistency check

Read-only; contacts no cluster and no Baseten.

```bash
jq empty deploy/baseten/model-contracts.json
jq empty deploy/baseten/cluster-runtime-evidence.json
jq -r '.local_reports[] | [.report.sha256, .report.path] | @tsv' deploy/baseten/cluster-runtime-evidence.json | while IFS=$'\t' read -r expected report_path; do
  test "sha256:$(shasum -a 256 "$report_path" | awk '{print $1}')" = "$expected" || exit 1
done
```

### Requirements lock-input digest check

Compare against `model-contracts.json` → `chain_topology.requirements_lock_digests`.

```bash
cd deploy/baseten && for f in requirements/*.txt; do \
  printf '%s  sha256:%s\n' "$f" "$(shasum -a 256 "$f" | awk '{print $1}')"; done
```

## Operator hand-off

1. Read `DEPLOY.md` top to bottom. It is ordered and each step names the
   `model-contracts.json` nulls it fills.
2. `python deploy/baseten/stage_packages.py` — mandatory. Without it `import
   plumb` fails inside every image and all nine Chainlets return `blocked`.
3. `pip install 'truss==0.18.30'` in a dedicated venv, then
   `truss chains push ./deploy/baseten/chain.py --environment production`. A
   *development* deployment is capped at one replica and cannot demonstrate
   autoscaling.
4. Apply the autoscaling hypotheses through the management API (step 7), then
   re-tune them from Gate A measurements. `AGENT-BUILD-SPEC.md:285` is explicit
   that there is no assumed universal throughput preset.
5. Submit complete logical episodes to the deployment's `/async_run_remote` URL.
   The application control plane owns the ledger, outbox, artifact persistence,
   request-ID association, retry-attempt records, signed callback verification,
   and reconciliation.

`plumb.platform.BasetenChainClient` exposes a Chain queue metric only after an
account-tested exact route is supplied, exposes active replicas from chain
deployment management, and keeps prices, currency, and hourly rates unavailable
until the account billing unit and currency are evidenced.

Source contracts: [Chain invocation](https://docs.baseten.co/development/chain/invocation),
[async inference and webhook signatures](https://docs.baseten.co/inference/async),
[Chains SDK reference](https://docs.baseten.co/reference/sdk/chains),
[Chain deployment management](https://docs.baseten.co/reference/management-api/deployments/gets-a-chain-deployment-by-id),
[autoscaling](https://docs.baseten.co/deployment/autoscaling/overview),
[cold starts](https://docs.baseten.co/deployment/autoscaling/cold-starts).
