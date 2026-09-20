# Private semantic Qwen judge worker

This is a single-H100, private model deployment for the saved-clip demo. It is
not the older full PLUMB Chain and it is not a public weights endpoint. The
worker loads only the semantic pilot adapter recorded as
`semantic_pilot_epoch_02`:

- Qwen base: `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`
- adapter tree: `sha256:7a0b7796a0ac8b78fbb8ba8f4c4db01fe8d35c2edc3cb7a0fbeec9f4fb20523e`
- sampling: five samples, `temperature=0.7`, three-vote quorum

The adapter is not present in this repository. Stage it only from a restored,
authenticated copy of the recorded semantic `epoch-02` directory:

```bash
python deploy/baseten/stage_demo_judge.py \
  --adapter-dir /scratch/lchow432/plumb/experiments/judge-lora-pilot-v1-trained/adapters/epoch-02
python deploy/baseten/stage_demo_judge.py \
  --adapter-dir /scratch/lchow432/plumb/experiments/judge-lora-pilot-v1-trained/adapters/epoch-02 --check
baseten model push --dir deploy/baseten/judge_demo --team q8grpdw --profile plumb-api \
  --deployment-name plumb-qwen-semantic-judge-demo --wait --output json
baseten model deployment update-autoscaling --profile plumb-api \
  --model-id <model-id> --deployment-id <deployment-id> \
  --min-replica 0 --max-replica 1 --concurrency-target 1 --scale-down-delay 60
baseten model deployment describe --profile plumb-api \
  --model-id <model-id> --deployment-id <deployment-id> --output json
```

Do not push if the staging hash check fails. Do not replace the adapter with the
format-only adapter, the base model, or a Hub adapter. Configure the localhost
app only after the deployment reports one H100 maximum:

```bash
.venv/bin/python scripts/serve_baseten_mvp.py \
  --model-id 3mzlenow --deployment-id q929yoj --profile plumb-api \
  --judge-model-id <model-id> --judge-deployment-id <deployment-id> --judge-profile plumb-api
```

The first explicit browser assessment creates the durable request/adapter/timing
receipt. The worker's output remains experimental and is not human-calibrated.
