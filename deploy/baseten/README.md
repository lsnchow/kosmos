# PLUMB Baseten template

`chain.py` is a custom Baseten Chains topology: CPU `RolloutController`, custom
policy/world/validity/judge Chainlets, typed JSON request/result models, and a
fail-closed stage sequence. It does not deploy on import and does not include
weights, credentials, model downloads, fake dependency pins, or a claim that
any policy/world/judge contract is certified.

`cluster-runtime-evidence.json` records the completed local allocated-GPU
diagnostics separately. It is deliberately not a Baseten configuration file.
The observed local model/runtime/revision, memory, and timing values are useful
compatibility inputs; they neither identify a Baseten container nor establish
deployment serialization, capacity, pricing, scientific fidelity, or any
qualification gate.

The initial `WorldWorker` resource shape is one H100 with
`predict_concurrency=1`, matching the build specification's benchmarking
starting point. It is not a capacity or performance result. The local Cosmos
diagnostic did run on a cluster H100; that does not reserve, select, or measure
an H100 in a Baseten account. Policy and judge accelerator/image choices remain
unresolved for deployment in `model-contracts.json`, rather than being guessed.

## Recorded local diagnostics, not cloud qualification

The evidence file contains the exact report paths and digests. Its selected
records are:

| Local diagnostic | What was observed | Still blocked |
| --- | --- | --- |
| Cosmos3-Nano / Diffusers, job 937372 | 16 actions → 17 frames; 4.524269459 s inference; 167.683719175 s model load; 36,521,475,584 B peak allocation | Baseten image/serializer, Gate A/B, cloud capacity and cost |
| OpenVLA, job 937500 | one native 7-D `bridge_orig` action; 0.835902354 s; 15,500,827,648 B peak allocation | deployed policy profile and native feedback qualification |
| IRASim Frame-Ada, job 937666 | one 7-D scaled action → two frames; 2.546707168 s; 23.114047749 s model load; 3,516,738,560 B peak allocation | action/frame fidelity and all cloud configuration |
| Qwen judge, job 12238810 | five raw samples; 29.291511048 s judge call; 16,948,912,640 B peak; unknown/insufficient quorum | human calibration, Gate D, and judge deployment profile |

These are cluster diagnostics only. The IRASim 18-case action probe and 16-tick
closed-loop trace remain experimental, unqualified diagnostics. The separate
completed 937751 state-representation replay is qualitative only: its pair and
repeat 16-tick hashes are reproducible under the intervention protocol, but
both branches show gripper drift. It did not requery a policy or score task
success, and supplies no cloud or fidelity qualification evidence. The
preserved earlier state-replay failure also supplies no deployment evidence.

### Evidence consistency check

Run this read-only check from the repository root after moving or updating a
local diagnostic report. It validates JSON syntax and verifies every selected
local-report digest; it does not contact a cluster or Baseten.

```bash
jq empty deploy/baseten/model-contracts.json
jq empty deploy/baseten/cluster-runtime-evidence.json
jq -r '.local_reports[] | [.report.sha256, .report.path] | @tsv' deploy/baseten/cluster-runtime-evidence.json | while IFS=$'\t' read -r expected report_path; do
  test "sha256:$(shasum -a 256 "$report_path" | awk '{print $1}')" = "$expected" || exit 1
done
```

## Operator hand-off

1. Read `cluster-runtime-evidence.json` as local diagnostic provenance, then
   resolve every *deployment* `null` and `unresolved` item in
   `model-contracts.json`: Baseten asset/container records, exact serializers,
   Gate A/B/D evidence, account capacity, and the pricing contract. Do not copy
   local cluster timing or GPU labels into a cloud configuration.
2. Write reviewed immutable lock inputs in `requirements.lock-inputs.txt` and
   update each Chainlet to use only its own compatible environment.
3. Test with vendor and Bridge fixtures, then push explicitly using the official
   Chains CLI command shown in `chain.py`. A development deployment is limited
   to one replica; use a reviewed environment deployment for autoscaling.
4. Submit a complete logical episode to the deployment's
   `/async_run_remote` URL. The application control plane owns ledger/outbox,
   artifact persistence, request-ID association, retry-attempt records, signed
   callback verification, and reconciliation. Internal Chainlet RPCs are
   awaited; they do not use the external async queue.

The companion `plumb.platform.BasetenChainClient` only exposes a Chain queue
metric after an account-tested exact route is supplied. It exposes active
replicas from chain deployment management, and keeps prices/currency/hourly
rates unavailable until the account billing unit and currency are evidenced.

## Durable application delivery

`plumb.outbox.BasetenOutbox` and the SQLite `baseten_outbox`,
`baseten_callbacks`, and `baseten_outbox_events` tables now provide the local
control-plane side of Chain delivery. A complete, identity-bound
`request: RolloutRequest` payload is committed against one planned logical
episode before the one allowed `/async_run_remote` POST. The payload's run ID,
episode ID, protocol hash, and all stage identities must equal the authoritative
ledger row and frozen run configuration.

The persisted state machine records the pre-POST boundary, returned request ID,
immutable request-attempt lineage, signed raw callback bytes plus their digest,
and callback finalization leases. It handles all of the following without
creating another logical episode or incrementing a terminal counter twice:

- a process crash after the durable pre-POST record becomes `ambiguous` on
  recovery and is never POSTed again;
- a callback that arrives before the POST response is committed is retained
  unassociated and atomically joined when the request ID is saved;
- an identical webhook is idempotent; changed signed bytes for the same request
  ID are a conflict and stop processing;
- an accepted callback is only terminalized after an artifact writer commits a
  final artifact reference, using a callback CAS lease;
- cancellation is recorded before any remote cancellation attempt. A late
  callback remains durable and must still be artifact-finalized, including work
  already allocated by the platform.

The artifact writer must supply named references containing `uri`, full
`sha256`, and `media_type`, after verifying and committing those bytes in its
storage system. Completed normalized results must bind `run_id`, `episode_id`,
and `protocol_hash`, preserve the planned `horizon_actions`, report that exact
`executed_actions` count, and use consistent boolean/null outcomes and integer
progress. Strings such as `"false"`, shortened horizons, and conflicting callback
identities are rejected before any terminal counter changes. The receiving API
does not infer this normalization or artifact verification from a signed callback.

Only a proved HTTP rejection can be manually requeued on the same outbox record
with an operator reason. Transport failures, timeouts, 5xx responses,
unreadable 2xx responses, interrupted client calls, and uncertain remote DELETE
outcomes remain reconciliation records; they are not automatic retries.

The local API can be configured for signed callback ingress with
`PLUMB_ENABLE_BASETEN_CALLBACKS=1`, `BASETEN_API_KEY`,
`BASETEN_CHAIN_ASYNC_URL`, and `BASETEN_WEBHOOK_SECRET`. It accepts at most one
MiB of raw bytes and stores no callback content in its read-only outbox response.
This switch only enables receiving and recovering records; it does not enable
deployment, submission, artifact scoring, or spend.

The durable SQLite state machine, raw-callback preservation, and offline fault
tests are implemented. The remaining deployment prerequisites are still
unresolved: an actual Chain object-storage writer and immutable artifact
manifest, account-tested Chain request status/cancel route, credentials,
deployment profile, capacity, pricing basis, and all scientific gates. No
Baseten POST, remote cancellation, deployment, or billable request has been
made by this repository.

The reconciler can recover expired local dispatch/finalization/cancellation
leases with no network traffic. A remote lifecycle/cancel controller is injected
only after the account has tested and recorded the exact Chain request-management
route. Current official docs show a model-ID status/cancel route, while the
Chain invocation docs establish only Chain submission routes, so this repository
does not infer a Chain status or cancellation URL. Lifecycle status is never
treated as output: Baseten documents that status endpoints do not return model
output and failed webhook delivery can lose it. The deployed Chain must persist
the result independently in object storage, and its artifact manifest is what
the callback finalizer records.

Source contracts: [Chain invocation](https://docs.baseten.co/development/chain/invocation),
[async inference and webhook signatures](https://docs.baseten.co/inference/async),
[Chains SDK reference](https://docs.baseten.co/reference/sdk/chains), and
[Chain deployment management](https://docs.baseten.co/reference/management-api/deployments/gets-a-chain-deployment-by-id).
