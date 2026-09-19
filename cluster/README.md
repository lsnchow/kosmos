# Cluster execution

Use the existing `drac` SSH session and cluster-side `drac` tmux allocation.
Original job `937177` ended after a laptop disconnect; replacement `937277` was
queued inside remote tmux. The debug queue has successfully run real one-H100
smoke/probe jobs. Check [the handoff](../HANDOFF.md) and live `squeue` before every
resumed session; job/node identifiers are not permanent.

All model/runtime downloads belong under `/scratch/lchow432/plumb`. Local source
code can be synchronized to `/scratch/lchow432/plumb/source`; never copy the model
directory back to the laptop. Cluster manifests/logs go in `evidence/` and `logs/`.
Queen's has a different real shared scratch path: `/global/scratch/hpc6308/plumb`.
Use its cluster-specific scripts under `cac/`; its stale `$SCRATCH` value is wrong.

`bootstrap.sh` creates the small download environment. `download_assets.py` first
prints a pinned size plan and downloads only with `--execute`, verifies file sizes
and upstream SHA-256s where supplied, and writes a provenance manifest. A model
download or successful fixture call is not scientific qualification.

Do not run GPU workloads on login nodes, cancel the user's allocation, or replace
the existing `drac:0.0` shell. Use additional `drac` windows and allocated compute
nodes for project-owned processes. Record actual per-stage GPU usage and failures.

## Reproducible diagnostics

Use `cluster/make_release.py` to bundle source plus a SHA-addressed `RELEASE.json`.
Upload that small archive to a new `releases/<hash>` directory, then pass its full
path as the first argument to the desired sbatch script. Do not modify an old
release or overwrite evidence from a failed run. The scripts have offline compute
flags and use already-downloaded local models.

- `smoke.sbatch` / `probe.sbatch`: actual Cosmos runtime and horizon/suffix controls.
- `openvla_smoke.sbatch`: native OpenVLA action in its isolated Transformers4.40 env.
- `irasim_smoke.sbatch`: safe original IRASim checkpoint and one-action/two-frame call.
- `closed_loop_smoke.sbatch`: 16fresh OpenVLA actions, each based on a newly generated
  IRASim image. Re-encoded-image state mode; explicitly unqualified.
- `irasim_probe.sbatch`: same-start/same-seed repeats and held-gripper ±axis controls.
- `irasim_state_replay.sbatch`: replays saved actions, comparing source-backed latent
  carry with image re-encoding using independent VAE/diffusion RNG streams. No policy
  requery, task score, or implied fidelity improvement.
- `irasim_native_reference.sbatch`: one released-horizon15action/16frame open-loop
  call. Its first prediction sees14future action rows; it is not native OpenVLA
  feedback and cannot stand in for the one-step protocol.

Trillium's debug partition uses one GPU/24CPUs and supplies memory automatically;
omit `--mem`. Preserve existing module `PYTHONPATH` when adding the source release.
OpenCV comes from `module load opencv/4.11.0`, not a PyPI wheel. Read the handoff for
the exact environments, successful/failed job IDs, asset pins, and raw reports.
