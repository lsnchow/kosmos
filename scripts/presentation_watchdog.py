"""Keep the existing console and SSH tunnel alive during a bounded demo window.

Does not submit GPU jobs or inference requests. Worker recovery stays inside
the explicitly allocated Slurm job. Runtime logs stay in the owning terminal.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--until", required=True, help="ISO timestamp with timezone")
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"trig\d+", args.node) or not args.allocation.isdigit():
        parser.error("Expected a Trillium node and numeric allocation")
    until = datetime.fromisoformat(args.until)
    if until.tzinfo is None or not 0 < until.timestamp()-time.time() <= 6*3600:
        parser.error("Demo window must be timezone-aware and at most six hours from now")
    if not args.token_file.is_file(): parser.error("Worker token file is missing")
    root = Path(__file__).resolve().parents[1]
    for port in (8787, 8924, 8925):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                parser.error("Port %s is already occupied; stop its old owner before handing over" % port)
    env = dict(os.environ, PLUMB_COMPARISON_BUDGET_USD="65",
               PLUMB_DEMO_JUDGE_ALLOCATION_ID=args.allocation,
               PLUMB_DEMO_JUDGE_WORKER_URL="http://127.0.0.1:8925",
               PLUMB_DEMO_JUDGE_WORKER_TOKEN_FILE=str(args.token_file.resolve()))
    commands = {
        "tunnel": ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                   "-o", "ServerAliveCountMax=3", "-L", "127.0.0.1:8924:%s:8919" % args.node,
                   "-L", "127.0.0.1:8925:%s:8921" % args.node, "trillium-gpu"],
        "console": [str(root / ".venv/bin/python"), "scripts/serve_baseten_mvp.py",
                    "--model-id", "3mzlenow", "--deployment-id", "q929yoj", "--profile", "plumb-api",
                    "--timeout-seconds", "600", "--judge-model-id", "qjjoyl2q",
                    "--judge-deployment-id", "w556dmj", "--judge-profile", "plumb-api",
                    "--judge-timeout-seconds", "600", "--live-demo-url", "http://127.0.0.1:8924",
                    "--live-demo-token-file", str(args.token_file.resolve())],
    }
    processes = {}
    stop = False
    def stopping(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    awake = subprocess.Popen(["caffeinate", "-i", "-t", str(int(until.timestamp()-time.time()))])
    print("Presentation supervision through %s; allocation %s on %s" % (args.until, args.allocation, args.node), flush=True)
    try:
        while not stop and time.time() < until.timestamp():
            for name, command in commands.items():
                previous = processes.get(name)
                if previous is None or previous.poll() is not None:
                    print("Starting %s%s" % (name, " after exit %s" % previous.returncode if previous else ""), flush=True)
                    processes[name] = subprocess.Popen(command, cwd=root, env=env, start_new_session=True)
            time.sleep(5)
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        awake.terminate(); awake.wait()


if __name__ == "__main__": main()
