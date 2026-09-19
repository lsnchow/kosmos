# Team 33 Baseten access — MVP priority

## Verified configuration update — 2026-09-19

Lucas explicitly requested configuration in LOCAL tmux `b10`.
API-key login succeeded under named CLI profile `plumb-api`; the preexisting
OAuth/default profile was retained. The key authenticates to the **Hack the
North organization/workspace**. Within it, the default team `Hack the North`
(`wljp62q`) has zero H100 quota, while the intended team is **33** (`q8grpdw`).
The live capacity response showed Team 33 H100 **limit 4, baseline 4, in use 0**.
This supersedes the earlier expectation of two GPUs, but does not authorize
launching four or establish remaining billing credits.

`b10` now has `BASETEN_PROFILE=plumb-api`, plus informational
`PLUMB_BASETEN_TEAM_ID=q8grpdw` and `PLUMB_BASETEN_TEAM_NAME=33` variables.
Those PLUMB variables do not automatically select a team in the Baseten CLI:
pass **`--team q8grpdw`** explicitly when creating a project/workstation.
Global organization/team defaults were not changed.

`baseten ssh setup --profile plumb-api` generated the Baseten SSH keypair and
its managed `~/.ssh/config` block. Existing non-Baseten SSH configuration was
left intact. No running Team 33 GPU job was established, no live SSH connection
was verified, and no GPU job/deployment was created. Account-side SSH enablement
still needs verification with an owned running workload. Credential value is
not recorded here; the CLI handled credential storage.

## Original planning note (superseded where noted above)

Added from Lucas's side conversation on 2026-09-19. This is an additive TODO,
not a deployment, capacity verification, or change to the scientific protocol.
Do not interrupt work already running in the main conversation.

## New access information

- Lucas reports Baseten GPU access for **Team 33**, expecting **two H100s**.
- Resolve the team's actual account identifier; do not assume its API ID is `33`.
- A credential was supplied in the side conversation. Its value is deliberately
  not copied into this file, source control, logs, or shell examples. This side
  conversation did not configure or test it. Use secure runtime credential
  configuration when executing the main-thread deployment work.
- Verify whether the allocation/credits cover Training Jobs, Dedicated
  Inference, or both; a training quota does not establish inference entitlement.
- Two GPUs are a reported expectation, not yet a confirmed reservation or
  permission to provision the entire existing multi-GPU topology.

## Ordered MVP TODO

- [x] Authenticate API-key profile `plumb-api`, resolve Team 33 (`q8grpdw`),
  inspect training capacity (4 H100 limit/baseline, 0 in use at verification),
  and configure local SSH in `b10`. No GPU job or deployment created.
- [ ] Finish account preflight: confirm required key permissions, product
  access, credits, billing and expiry; recheck live GPU usage. Record no
  credential values. Official current docs
  describe `baseten train capacity describe`; inspect the installed CLI version
  first because this repo previously validated Truss 0.18.30, not every newer
  CLI surface. Do not change organization/team quotas.
- [ ] Choose the compute route from actual entitlements: managed inference for
  a persistent model endpoint if available; Training Jobs with SSH for bounded
  runtime setup/debugging otherwise. Verify networking before treating an SSH
  training container as an externally callable application server.
- [ ] Scope an explicit **unqualified cloud diagnostic MVP**, separate from the
  full six-policy/five-task scientific study. One supported policy, one task,
  one actual request, persistent artifacts, and a visible dashboard result are
  the first integration milestone. That alone is not a full evaluated rollout.
- [ ] Reuse `deploy/baseten/DEPLOY.md`, `chain.py`, existing outbox/callback/result
  store code and the current dashboard. First establish one bounded cloud-native
  model smoke with pinned public-runtime dependencies and hash-bound assets.
  Alliance wheel builds and timings do not automatically transfer to Baseten.
- [ ] Plan a topology within the verified two-H100 limit. Do not push all
  existing GPU Chainlets unchanged. Start with one GPU and account for peak
  memory; use the second only for a measured need. Sequential stages and
  separate incompatible runtimes are options, not an assumed working layout.
- [ ] Complete cloud wiring: endpoint/auth, externally reachable authenticated
  callback if used, private durable result storage/readback, timeout/error
  behavior, and dashboard connection. Keep credentials server-side.
- [ ] Demonstrate actual request -> model output -> saved artifact -> UI,
  including one failed-request case and restart/readback. If world-model
  feedback is not yet credible, show the real policy diagnostic and clearly
  labelled recorded clips; do not present them as a successful closed-loop run.
- [ ] Run only a bounded smoke before larger use; record Team 33 attribution,
  actual GPU count, runtime, latency scope and billing observations. Configure
  lifecycle/idle shutdown for newly created resources. Do not stop unrelated jobs.
- [ ] Continue judge development separately. Existing semantic and format-only
  adapters remain disabled; new GPU access does not cure biased targets or
  replace independent labels. Never weaken formal gates to make the demo pass.

## Baseten products worth using

- **Training Jobs:** existing scripts on managed GPUs, SSH debugging, job logs
  and checkpoint handling. Useful for reproducing the cluster runtime and
  subsequent well-founded fine-tuning. [Training products](https://www.baseten.co/products/training/),
  [job and capacity management](https://docs.baseten.co/training/management).
- **Custom model serving / Dedicated Inference:** deploy model code behind an
  API instead of building the application around an interactive shell.
  [Model development](https://docs.baseten.co/development/model/overview).
- **Chains:** Python orchestration across models with separate dependencies and
  resource choices. Strong match for PLUMB later; the current full topology is
  not a two-GPU MVP deployment. [Chains](https://docs.baseten.co/development/chain/overview).
- **Model APIs:** hosted models may be useful for a separately labelled
  exploratory comparison or application feature without hosting those weights.
  Check supported model/modality, data terms, credits and access first; do not
  silently substitute a different model for the frozen judge.
  [Model APIs](https://docs.baseten.co/inference/model-apis/overview).

Access verification and deployment remain TODO. No paid job, endpoint, account
setting, credential configuration, or existing main-thread work was changed by
this note. The only workspace addition from this side conversation is this file.
