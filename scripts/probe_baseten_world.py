"""One bounded, unscored source-pose/manual Cosmos probe on a real Chain.

No policy, judge, qualifying gate, automatic retry, or presentation promotion.
The source dataset's raw gripper state is retained, not converted to 0..0.39.
It does not enter the world action: the recorded source action's gripper is
held constant. Pose integration remains a disclosed development hypothesis.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plumb.comparisons import BasetenCliStreamTransport, ComparisonService
from plumb.records import canonical_json
from deploy.baseten.demo_chain import service_dispatch_to_stream_request


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", choices=("right", "left", "up", "down"), default="right")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--resolution", type=int, choices=(256, 480), default=256)
    args = parser.parse_args()
    source_raw = args.start.read_bytes()
    source = json.loads(source_raw)
    png = (args.start.parent / source["frame"]["path"]).read_bytes()
    if digest(png) != "sha256:" + source["frame"]["sha256"]:
        raise ValueError("Source frame changed")
    state_values = source["proprioception"]["raw_float32_8d"]
    gripper_action = source["action"]["raw_float32_7d"][-1]
    state_payload = {"values": state_values}
    lock_path = Path(__file__).resolve().parents[1] / "deploy/baseten/assets/cosmos3-nano-runtime-lock.json"
    profile = {
        "backend": "cosmos", "profile_id": f"cosmos-source-pose-probe-{args.resolution}-30",
        "domain": "bridge_orig_lerobot", "model_id": "nvidia/Cosmos3-Nano",
        "model_revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92",
        "resolution_tier": args.resolution, "inference_steps": 30,
        "asset_manifest_id": digest(lock_path.read_bytes()),
    }
    profile_sha = digest(canonical_json(profile).encode())
    attempt = uuid.uuid4().hex
    world = {"runtime_profile": profile, "profile_sha256": profile_sha,
             "bridge_control_profile_id": "bridge-source-pose-development-probe",
             "held_gripper_action": gripper_action, "manual_seed": args.seed}
    dispatch = {
        "schema": "kosmos-chain-stream-v1", "kind": "manual_world_segment", "mode": "manual",
        "comparison_id": "world-probe", "cell_id": "not-a-comparison-cell", "attempt_id": attempt,
        "session_id": "probe-" + attempt, "command_id": "command-" + attempt,
        "direction": args.direction, "task": "close_drawer", "task_instruction": "Close the drawer",
        "world": world, "identity_hashes": {"world_profile": profile_sha, "source_manifest": digest(source_raw)},
        "source": {"event_id": "source-frame-0", "png_base64": base64.b64encode(png).decode(),
                   "png_sha256": digest(png), "state": {"payload": state_payload,
                       "origin": "development_raw_source_pose_unqualified",
                       "conventions_sha256": digest(canonical_json(source["proprioception"]).encode()),
                       "sha256": digest(canonical_json(state_payload).encode())}},
        "claim_tier": "preview", "scored": False,
    }
    typed = service_dispatch_to_stream_request(dispatch)
    assert typed.manual is not None  # Validate exact deployed request before billing.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_text(json.dumps({"request": dispatch}, indent=2) + "\n")
    started = time.monotonic()
    report = {"schema": "kosmos-world-probe-evidence-v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "url": args.url, "attempt_id": attempt, "scored": False, "qualified": False,
              "source_state_limitations": source["use_constraints"],
              "probe_scope": "raw-source-pose hypothesis; recorded gripper ACTION held; no policy input", "frames": []}
    try:
        chunks = BasetenCliStreamTransport().stream(run_remote_url=args.url, cli_profile="plumb-api",
                                                   payload={"request": dispatch}, timeout_seconds=900)
        with (args.output / "events.jsonl").open("x") as events:
            terminal = None
            for record in ComparisonService._decode_ndjson(chunks):
                if record.get("attempt_id") != attempt:
                    raise ValueError("Stream attempt mismatch")
                events.write(json.dumps(record) + "\n")
                events.flush()
                kind = record.get("kind")
                print(json.dumps({"kind": kind, "stage": record.get("stage"), "status": record.get("status"),
                                  "elapsed": time.monotonic() - started, "reasons": record.get("reasons")}), flush=True)
                if kind == "frame":
                    raw = base64.b64decode(record["png_base64"], validate=True)
                    if digest(raw) != record["png_sha256"]:
                        raise ValueError("Frame digest mismatch")
                    index = len(report["frames"]) + 1
                    if record["frame_index"] != index:
                        raise ValueError("Frame order mismatch")
                    path = args.output / (f"frame-{index:03d}.png")
                    path.write_bytes(raw)
                    report["frames"].append({"path": path.name, "sha256": digest(raw), "arrival_seconds": time.monotonic() - started})
                elif kind == "terminal":
                    terminal = record
            report["terminal"] = terminal
            if terminal is None or terminal.get("status") != "completed" or len(report["frames"]) != 16:
                raise ValueError("World probe did not complete exactly sixteen post-conditioning frames")
        subprocess.run(["ffmpeg", "-v", "error", "-framerate", "5", "-start_number", "1", "-i",
                        str(args.output / "frame-%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(args.output / "probe.mp4")], check=True)
        report["status"] = "completed_unqualified"
    except Exception as error:
        report["status"] = "failed_or_incomplete"
        report["error_type"] = type(error).__name__
    report["wall_seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "completed_unqualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
