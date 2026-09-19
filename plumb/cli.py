"""Small command interface for serving, exercising, and analyzing PLUMB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="plumb")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--data-dir", type=Path, default=Path("data"))
    run = commands.add_parser("run", help="Run an explicitly synthetic integration fixture")
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--starts", type=int, default=50)
    run.add_argument("--seed", type=int, default=20260919)
    commands.add_parser("reference")
    report = commands.add_parser("analyze")
    report.add_argument("episodes", type=Path)
    export = commands.add_parser("export")
    export.add_argument("run_id")
    export.add_argument("--data-dir", type=Path, default=Path("data"))
    export.add_argument("--output", type=Path, required=True)
    recover = commands.add_parser("recover", help="Recover an orphaned fixture run after its owner lease expires")
    recover.add_argument("run_id")
    recover.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from plumb.api import create_app
        uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port)
    elif args.command == "run":
        from plumb.engine import RunService
        service = RunService(args.data_dir)
        result = service.create_run({"mode": "synthetic", "backend": "synthetic", "starts_per_task": args.starts, "seed": args.seed})
        run_id = result.get("id", result.get("run_id"))
        service.execute_run(run_id)
        print(json.dumps(service.get_run(run_id), indent=2, allow_nan=False))
    elif args.command == "reference":
        from plumb.reference import reference_payload
        print(json.dumps(reference_payload(), indent=2))
    elif args.command == "analyze":
        from plumb.measurement import analyze
        with args.episodes.open() as source:
            rows = [json.loads(line) for line in source if line.strip()]
        print(json.dumps(analyze(rows), indent=2, allow_nan=False))
    elif args.command == "export":
        from plumb.engine import RunService
        service = RunService(args.data_dir)
        service.get_run(args.run_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as destination:
            for row in service.list_episodes(args.run_id, limit=10000):
                destination.write(json.dumps(row, allow_nan=False) + "\n")
        print(str(args.output))
    elif args.command == "recover":
        from plumb.engine import RunService
        service = RunService(args.data_dir)
        service.recover_run(args.run_id, force=False)
        service.execute_run(args.run_id)
        print(json.dumps(service.get_run(args.run_id), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
