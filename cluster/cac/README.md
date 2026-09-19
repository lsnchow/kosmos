# Frontenac Qwen judge lane

This is a CAC-only lane for the local Qwen2.5-VL-7B judge.  Frontenac exports
`SCRATCH=/scratch/$USER`, but that path is absent on both login1 and its compute
nodes.  Its mounted shared scratch path is `/global/scratch/$USER`; this lane
uses `/global/scratch/$USER/plumb` and rejects other paths.

The normal scheduler request must omit `--partition`, because the site submit
plugin routes the request itself:

```bash
salloc --account=def-hpcg6234_gpu --qos=normal --nodes=1 --ntasks=1 \
  --gres=gpu:a100:1 --cpus-per-gpu=8 --mem=64G --time=2:00:00 \
  --job-name=plumb-qwen-judge
```

With an allocation active, install/download on `login1` in a separate shell:

```bash
PLUMB_ROOT=/global/scratch/$USER/plumb bash source/cluster/cac/setup_qwen_runtime.sh
PLUMB_ROOT=/global/scratch/$USER/plumb bash source/cluster/cac/download_qwen_assets.sh
```

Run the local-only smoke through the allocated Slurm shell.  Supply the exact
video, reference, frozen task/rubric, and five logged seeds; it pins profile
`transformers_version` to the installed module version `5.17.0`.

```bash
PLUMB_ROOT=/global/scratch/$USER/plumb bash source/cluster/cac/run_qwen_smoke.sh \
  --video /path/to/cosmos.mp4 --reference /path/to/context.png --reference-role scene \
  --task '...' --rubric '...' --seeds 101 102 103 104 105
```

The manifest and runtime freeze belong in `evidence/`; the checkpoint remains
on shared scratch and is never copied through a workstation.
