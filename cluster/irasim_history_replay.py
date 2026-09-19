"""Causal growing-history IRASim replay; exploratory conditioning, never scoring.

The original 16-frame / extras=3 / training mask=1 model is unchanged. At tick t,
condition on up to 15 already-generated latents and supply the matching past
actions plus the current saved action. Generate exactly one unseen final frame.
The released pipeline accepts mask_x with multiple frames, but this differs
from its training conditioning distribution. No future action is invented.
Saved actions are replayed; this is NOT a newly queried policy rollout.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

from cluster import irasim_state_replay as replay


def window_for_tick(tick: int) -> tuple[int, int]:
    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
        raise ValueError("tick must be a nonnegative integer")
    return max(0, tick - 14), tick + 1


def history_step(torch: Any, adapter: Any, pipeline: Any, history: Any,
                 inputs: Sequence[replay.ReplayInput], tick: int) -> tuple[Any, Any, dict]:
    start, stop = window_for_tick(tick)
    if len(inputs) <= tick or any(item.tick != i for i, item in enumerate(inputs)):
        raise ValueError("Saved actions must be contiguous and include the current tick")
    known = stop - start
    if tuple(history.shape) != (1, known, 4, 32, 40):
        raise ValueError("History must contain exactly the causal window's known latents")
    replay._require_finite(torch, history, "history condition")
    selected = inputs[start:stop]
    scaled = [[float(v) * (20.0 if i < 6 else 1.0) for i, v in enumerate(row.action)] for row in selected]
    actions = torch.tensor(scaled, device=adapter.profile.device, dtype=torch.float32).view(1, known, 7)
    replay._require_finite(torch, actions, "history actions")
    seed = replay.derive_stream_seed(inputs[tick].recorded_seed, "diffusion")
    videos, latents = pipeline(
        actions, mask_x=history, video_length=known + 1,
        height=adapter.profile.input_height, width=adapter.profile.input_width,
        num_inference_steps=adapter.profile.inference_steps,
        guidance_scale=adapter.profile.guidance_scale,
        generator=replay._torch_generator(torch, adapter.profile.device, seed),
        output_type="both", return_dict=False, device=torch.device(adapter.profile.device),
    )
    if latents is None or tuple(latents.shape) != (1, known + 1, 4, 32, 40):
        raise ValueError("History pipeline returned an incorrect latent shape")
    replay._require_finite(torch, latents, "history output")
    if not bool(torch.equal(latents[:, :known], history)):
        raise ValueError("Pipeline altered committed conditioning history")
    if videos is None or int(videos.shape[1]) != known + 1:
        raise ValueError("History pipeline returned an incorrect frame count")
    metadata = {"tick": tick, "known_frame_start": start, "known_frame_end": tick,
                "action_ticks": list(range(start, stop)), "condition_frames": known,
                "video_length": known + 1, "future_action_rows": 0,
                "action_padding": False, "diffusion_seed": seed,
                "raw_native_actions_7d": [list(row.action) for row in selected],
                "condition_latent_sha256": replay.tensor_hash(history)}
    return videos, latents, metadata


def release_binding(config: replay.ReplayConfig) -> Mapping[str, Any]:
    digest = replay._normalised_sha256(config.source_release, "source release")
    root = config.root / "releases" / digest
    manifest = root / "RELEASE.json"
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Source release manifest hash mismatch")
    for entry in json.loads(raw)["files"]:
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(root.resolve()) or replay._sha256(target) != entry["sha256"]:
            raise ValueError("Source release file hash mismatch")
    if Path(__file__).resolve() != root / "cluster" / "irasim_history_replay.py":
        raise ValueError("Diagnostic must execute from the verified frozen release")
    # The older replay verifies pipeline files; bind the model implementation too.
    for options in ([], ["--cached"]):
        subprocess.run(["git", "-C", str(config.irasim_repo), "diff", *options,
                        "--exit-code", "--", "models/irasim.py"], check=True, capture_output=True)
    return {"source_release": digest, "manifest": replay._file_record(manifest),
            "all_release_files_verified": True,
            "model_source": replay._file_record(config.irasim_repo / "models/irasim.py")}


def run_once(config: replay.ReplayConfig, inputs: Sequence[replay.ReplayInput],
             adapter: Any, initial_frame: Any, run_dir: Path) -> dict:
    torch, pipeline = adapter._load()
    if (adapter.profile.num_frames, adapter.profile.extras, adapter.profile.mask_frame_num) != (16, 3, 1):
        raise ValueError("Original model architecture and training config must remain 16/3/1")
    seed = replay.derive_stream_seed(inputs[0].recorded_seed, "initial_vae")
    history, preprocessing = replay._encode_mask(adapter, torch, pipeline, initial_frame, seed)
    frames = [initial_frame]
    initial = replay._branch_artifact(run_dir, "history", 0, initial_frame, history, history)
    rows = []
    for tick in range(len(inputs)):
        with torch.no_grad():
            videos, latents, metadata = history_step(torch, adapter, pipeline, history, inputs, tick)
        frame = replay._irasim_frame_to_rgb(videos[0, -1])
        final = latents[:, -1:].clone()
        metadata["output"] = replay._branch_artifact(run_dir, "history", tick + 1, frame, latents, final)
        rows.append(metadata)
        frames.append(frame)
        history = latents[:, -15:].clone()
        # Commit progress at each completed boundary; partial failure stays recoverable evidence.
        replay._atomic_json(run_dir / "progress.json", {"completed_ticks": len(rows), "rows": rows})
    path = run_dir / "history" / "history.mp4"
    encoded, export = replay._video_export_frames(frames)
    replay._write_mp4(path, encoded)
    return {"initial": initial, "initial_vae_seed": seed, "preprocessing": preprocessing,
            "rows": rows, "video": replay._file_record(path), "export": export,
            "frame_count": len(frames)}


def run_history(config: replay.ReplayConfig, *, adapter_loader=replay._load_adapter,
                provenance_loader=replay._source_provenance,
                binding_loader=release_binding, frame_loader=replay._load_vendor_frame_zero) -> tuple[int, dict]:
    config.__post_init__()
    config.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"schema_version": 1, "kind": "plumb_irasim_causal_history_replay_diagnostic",
              "qualified": False, "success_scoring": False, "policy_requery": False,
              "binary_success": None, "progress_score": None, "validity": "unknown",
              "started_at": replay._utc_now(),
              "limitation": "Growing history differs from training mask=1; saved-action replay is not a policy evaluation. Noise shapes vary with horizon; cross-profile differences are not history-only causal effects.",
              "hardware": {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                           "node": os.environ.get("SLURMD_NODENAME")}}
    try:
        if any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
            raise ValueError("Both offline environment flags must equal 1")
        inputs = replay.load_replay_inputs(config.source_report)
        report["release_binding"] = binding_loader(config)
        report["provenance"] = provenance_loader(config)
        frame = frame_loader(config.fixture_video)
        adapter = adapter_loader(config)
        report["outcome"] = run_once(config, inputs, adapter, frame, config.output_dir)
        if config.same_seed_repeat:
            repeat = run_once(config, inputs, adapter, frame, config.output_dir / "repeat")
            report["same_seed_repeat"] = repeat
            report["repeat_pixel_equal_by_tick"] = [
                a["output"]["raw_pixel_sha256"] == b["output"]["raw_pixel_sha256"]
                for a, b in zip(report["outcome"]["rows"], repeat["rows"])]
            report["repeat_latent_equal_by_tick"] = [
                a["output"]["final_latent_tensor_sha256"] == b["output"]["final_latent_tensor_sha256"]
                for a, b in zip(report["outcome"]["rows"], repeat["rows"])]
        report["status"] = "completed_unqualified_diagnostic"
        code = 0
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                                               "traceback": traceback.format_exc(limit=10)})
        code = 2
    report["finished_at"] = replay._utc_now()
    report["total_seconds"] = time.perf_counter() - started
    replay._atomic_json(config.output_dir / "report.json", report)
    return code, report


def main() -> int:
    parser = replay.build_parser()
    parser.description = __doc__
    config = replay.config_from_args(parser.parse_args())
    code, report = run_history(config)
    print(json.dumps(replay._jsonable(report), sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
