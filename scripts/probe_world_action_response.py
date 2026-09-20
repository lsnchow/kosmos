"""Bounded direct-worker action intervention; never a policy rollout or gate pass.

Uses the exact deployed Chain world worker to isolate model response from the
controller. Three same-start/same-seed calls: +x, -x and zero translation.
No inference retry. Original per-call requests, responses and PNGs are retained.
"""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.baseten.demo_chain import WorldTurnRequest, _forecast_compile, service_dispatch_to_stream_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-request", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manual = service_dispatch_to_stream_request(json.loads(args.source_request.read_text())["request"]).manual
    if manual is None:
        raise ValueError("Expected existing bound manual source request")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for label, dx in (("positive_x", 0.015), ("negative_x", -0.015), ("zero", 0.0)):
        folder = args.output / label
        folder.mkdir()
        native = [[dx, 0, 0, 0, 0, 0, manual.held_gripper_action] for _ in range(16)]
        compiled, final_state, detail = _forecast_compile(manual.world, manual.bridge_control_profile_id, manual.state_snapshot, native)
        identity = uuid.uuid4().hex
        request = WorldTurnRequest(attempt_id=identity, operation="manual_segment", world=manual.world,
            condition=manual.condition_frame, native_actions=native, compiled_actions=compiled,
            nominal_control_timestamps=[i / 5 for i in range(16)], world_seed=manual.world_seed,
            state_before=manual.state_snapshot, state_after=final_state, request_id=identity)
        request_path = folder / "request.json"
        request_path.write_text(json.dumps({"request": request.model_dump(mode="json")}) + "\n")
        started = time.monotonic()
        with (folder / "response.json").open("xb") as response, (folder / "cli.stderr").open("xb") as stderr:
            result = subprocess.run(["baseten", "model", "predict", "--model-id", args.model_id,
                "--deployment-id", args.deployment_id, "--profile", "plumb-api", "--file", str(request_path)],
                stdin=subprocess.DEVNULL, stdout=response, stderr=stderr, timeout=900)
        summary = {"label": label, "dx": dx, "world_seed": manual.world_seed,
            "wall_seconds": time.monotonic() - started, "model_id": args.model_id,
            "deployment_id": args.deployment_id, "scored": False, "qualified": False, "retries": 0,
            "compiler": detail, "status": "failed_or_incomplete"}
        summaries.append(summary)
        if result.returncode:
            (args.output / "report.json").write_text(json.dumps(summaries, indent=2) + "\n")
            return 1
        value = json.loads((folder / "response.json").read_text())
        summary["status"] = value.get("status")
        summary["timing"] = value.get("timing")
        summary["reasons"] = value.get("reasons")
        if value.get("status") != "completed" or len(value.get("frames", [])) != 16:
            (args.output / "report.json").write_text(json.dumps(summaries, indent=2) + "\n")
            return 1
        for index, frame in enumerate(value["frames"], start=1):
            raw = base64.b64decode(frame["png_base64"], validate=True)
            if "sha256:" + hashlib.sha256(raw).hexdigest() != frame["png_sha256"]:
                raise ValueError("Output hash mismatch")
            (folder / (f"frame-{index:03d}.png")).write_bytes(raw)
        print(json.dumps(summary), flush=True)
        (args.output / "report.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
