# Baseten semantic judge: real browser verification

Verified 2026-09-20 against Team **33**, ID `q8grpdw`, using the stored
`plumb-api` profile. The parent organization is Hack the North; the model's
actual owning team reads back as `33`.

## Deployment

- Private authenticated model: `qjjoyl2q`
- Deployment: `w556dmj`, `semantic-epoch02-e2e-v1`
- One H100; request concurrency 1
- Autoscaling read-back: min 0, max 1, concurrency target 1, idle delay 60 s
- Archive download disabled at creation
- Base and processor: `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Semantic adapter: `semantic_pilot_epoch_02`
- Verified tree: `sha256:7a0b7796a0ac8b78fbb8ba8f4c4db01fe8d35c2edc3cb7a0fbeec9f4fb20523e`
- Enabled active adapter verified on all 56 LoRA layers
- Transformers 4.49.0 and PEFT 0.14.0

The adapter copies were already on the laptop from the earlier recovery turn.
This deployment used those existing ignored staging files; no additional
cluster-to-laptop weight transfer occurred. Weights and credentials remain
untracked and outside browser artifact routes. The public model/processor
snapshot was fetched by the private worker at the pinned revision.

## Actual request and result

`judge-636e04d1b04b4147839f7ecfe0ebc551` completed with `status=abstained`.
The UI displays **Unable to assess**. Four final samples parsed with visual
integrity `artifact`; the fifth exhausted its schema retry. No decisive
completion vote survived. Collision also lacked three agreeing valid votes.
Raw completion/progress assertions stay in sample details, not in the
application's completion assessment.

Measured warm request:

- Source decode/frame preparation: 0.187 s
- VLM inference, five samples including one schema retry: 13.625 s
- Remote request: 14.645 s
- Remote overhead outside VLM inference: 1.020 s (not separate queue telemetry)
- Local validation plus first durable commit: 0.0068 s
- Server admission through first persisted result: 14.902 s
- Peak allocated VRAM: 17,870,431,744 bytes
- Initial worker load, measured on its first call: 73.641 s

One earlier completed GPU call, `judge-1573511dbd6047a6bf0204f0d64abf75`,
was rejected locally because CLI JSON serialization rounded 63-bit seeds.
Its raw response and failure are preserved. Seed derivation now uses five
distinct 31-bit integers, exactly representable in the transport. The second
request was an explicit browser verification after that fix, not an automatic
retry of an ambiguous request. There were two actual GPU requests total.

## Acceptance evidence

Evidence is local under `data/private/semantic-judge/`:

- `push.json` and `autoscaling-readback.json`
- `browser-acceptance-int31/verification.json`: receipt, timing, one browser
  POST in the successful verification, zero reload submissions, no page errors
- `browser-acceptance-int31/judgment.json`: complete persisted response
- `browser-acceptance-int31/assessment.png` and `reload.png`

The original inputs, scene context, selected indexes, decoded pixel hashes,
raw/parsed attempts and final output are also in the existing SQLite journal
and `data/live-integrated/demo-judgments/<request-id>/`.

Frontend build/typecheck and real Chromium submit/readback/reload passed.
Displayed fields were compared to the local quorum computation over the
retained raw report. No full test suite, new training, policy rollout, or
qualification change was performed.

The dedicated VLM work reserves $15 within the shared $100 cap. This is a
reservation, not settled billing. Min zero/max one and the 60-second idle
delay bound residency between explicit requests; they do not impose an
account-wide monetary quota.

## Local startup

```bash
env PLUMB_COMPARISON_BUDGET_USD=80 .venv/bin/python scripts/serve_baseten_mvp.py \
  --model-id 3mzlenow --deployment-id q929yoj --profile plumb-api \
  --timeout-seconds 600 --judge-model-id qjjoyl2q \
  --judge-deployment-id w556dmj --judge-profile plumb-api --judge-timeout-seconds 600
```

Open `http://127.0.0.1:8787/console`. Reload reads the persisted assessment;
only the explicit assessment button admits another GPU request.
