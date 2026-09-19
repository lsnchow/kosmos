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

Source contracts: [Chain invocation](https://docs.baseten.co/development/chain/invocation),
[async inference and webhook signatures](https://docs.baseten.co/inference/async),
[Chains SDK reference](https://docs.baseten.co/reference/sdk/chains), and
[Chain deployment management](https://docs.baseten.co/reference/management-api/deployments/gets-a-chain-deployment-by-id).
