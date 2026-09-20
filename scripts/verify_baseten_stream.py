"""Run the CPU-only Chain canary and retain measured incremental receipts.

This makes one explicit real request; it never retries or claims GPU inference.
Authentication remains inside the Baseten CLI wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plumb.comparisons import BasetenCliStreamTransport


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--profile", default="plumb-api")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    nonce = uuid.uuid4().hex
    output = args.output_dir / ("stream-canary-" + nonce + ".json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    pending = b""
    receipts = []
    error = None
    try:
        for block in BasetenCliStreamTransport().stream(
            run_remote_url=args.url, cli_profile=args.profile,
            payload={"nonce": nonce}, timeout_seconds=120,
        ):
            pending += block
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("nonce") != nonce or record.get("schema") != "kosmos-stream-canary-v1":
                    raise ValueError("Canary response identity mismatch")
                if record.get("model_calls") != 0:
                    raise ValueError("Canary unexpectedly reported model work")
                elapsed = time.monotonic() - started
                receipts.append({"arrival_seconds": elapsed, "record": record})
                print(json.dumps({"type": record["type"], "arrival_seconds": elapsed}), flush=True)
        if pending.strip():
            raise ValueError("Unterminated canary record")
        records = [item["record"] for item in receipts]
        if [r["type"] for r in records] != ["header", "receipt", "receipt", "receipt", "terminal"]:
            raise ValueError("Missing or extra canary records")
        if [r["index"] for r in records[1:4]] != [0, 1, 2]:
            raise ValueError("Receipt order mismatch")
        if records[-1]["status"] != "completed_transport_only":
            raise ValueError("Canary missing terminal completion")
        spread = receipts[-1]["arrival_seconds"] - receipts[1]["arrival_seconds"]
        if spread < 0.4:
            raise ValueError("Receipts were buffered instead of arriving incrementally")
    except Exception as exc:
        # Do not persist a potentially credential-bearing provider exception.
        error = type(exc).__name__
    report = {
        "schema": "kosmos-stream-canary-evidence-v1", "started_at": started_at,
        "nonce": nonce, "url": args.url, "wall_seconds": time.monotonic() - started,
        "status": "passed_transport_only" if error is None else "failed_transport_only",
        "error_type": error, "receipts": receipts, "model_calls": 0, "scored": False,
        "gpu_inference_verified": False, "retries": 0,
        "source_sha256": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(str(output))
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
