# Presentation runtime — 2026-09-20 (Toronto)

- Demo: 12:30 p.m.
- GPU allocation: `943555`, node `trig0063`, one H200, `compute_h200`.
- Allocation window: 09:50:01–13:50:01 EDT.
- Console/tunnel supervision: through 13:45 EDT, in local tmux `plumb-live`.
- Pinned worker source: `/scratch/lchow432/plumb/releases/e841443b2a7c43924a482ab8e2eeb6247708a686`.
- Console: `http://127.0.0.1:8787`; private tunnel ports 8924 (Cosmos) and 8925 (judge).
- Baseten is not the active execution path; configured credentials returned 403.

`scripts/presentation_watchdog.py` owns the console and SSH tunnel subprocesses,
restarts an exited process on its five-second check, and runs `caffeinate -i`
for the presentation window. SSH keepalives detect broken connections. This
does not bypass Duo when a fresh login requires approval. Keep the Mac plugged
in, its lid open, and its network connected.

`cluster/cosmos_judge_presentation.sbatch` supervises both model processes
within the one four-hour allocation, with up to four startup attempts. It
does not allocate more GPUs. Model restarts need time to reload weights;
in-flight inference is not silently replayed.

Old H100 allocation `943488` was cancelled after the H200 passed the real
generation/judge check. Temporary warm-up tunnels on 8926/8927 were removed.
The serving/telemetry PRD remains rolled back in the Git stash.

## Verified run

- Session: `86819c55242f40e988a9b84dfa85d596`.
- Judge: `judge-301c8342c3e5488fa86e63824c4414f2`.
- Cosmos: 16 generated frames, 3.594 seconds of model execution.
- Judge: all five samples returned; enabled semantic epoch-02 adapter verified.
- Judge compute: 29.344 seconds; remote request: 39.373 seconds.
- Judge abstained. This verifies execution and persistence, not task success.
- Screenshot: `data/private/semantic-judge/presentation-h200-ready.png`.

The runtime is intentionally bounded; a later presentation needs a new
allocation/window. Do not restart the earlier 10:00-expiry configuration.
