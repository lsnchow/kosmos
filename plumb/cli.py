"""Small command interface for serving, exercising, and analyzing Kosmos."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> Optional[int]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    workflows = {
        "scenarios": ("plumb.scenarios", "Import, freeze, and inspect provenance-backed scenario panels"),
        "annotation": ("plumb.annotation", "Prepare blinded human calibration and import independent ratings"),
        "study": ("plumb.study", "Freeze the full study and validate designs and rehearsal evidence"),
        "distillation": ("plumb.distillation_preparation", "Prepare development-only judge training data and check launch prerequisites"),
    }
    if arguments and arguments[0] in workflows:
        module = importlib.import_module(workflows[arguments[0]][0])
        return module.main(arguments[1:])
    parser = argparse.ArgumentParser(prog="kosmos")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in workflows.items():
        commands.add_parser(name, help=description, add_help=False)
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

    lock = commands.add_parser("assets-lock", help="Write assets.lock.json from the pinned discovery plan")
    lock.add_argument("--output", type=Path, default=Path("assets.lock.json"))

    freeze_cmd = commands.add_parser(
        "freeze-protocol", help="Write and freeze protocol.json, then print the tag command to preregister it"
    )
    freeze_cmd.add_argument("--output", type=Path, default=Path("protocol.json"))
    freeze_cmd.add_argument("--protocol-id", default="protocol-v1")
    freeze_cmd.add_argument("--seed", type=int, default=20260919)
    freeze_cmd.add_argument("--starts", type=int, default=50)
    freeze_cmd.add_argument("--scenario-manifest-hash", default=None)
    freeze_cmd.add_argument("--asset-lock", type=Path, default=Path("assets.lock.json"))
    freeze_cmd.add_argument("--operating-point", default=None)

    prereg = commands.add_parser("prereg-status", help="Report whether the frozen protocol is externally timestamped")
    prereg.add_argument("--protocol", type=Path, default=Path("protocol.json"))

    gates_cmd = commands.add_parser("gates", help="Print the gate ledger and every blocking evidence error")
    gates_cmd.add_argument("--results-dir", type=Path, default=Path("."))

    publish = commands.add_parser("publish", help="Write the spec section 8 artifacts for a run")
    publish.add_argument("run_id")
    publish.add_argument("--data-dir", type=Path, default=Path("data"))
    publish.add_argument("--results-dir", type=Path, default=Path("."))
    publish.add_argument("--protocol", type=Path, default=Path("protocol.json"))

    inventory = commands.add_parser("artifacts", help="Report which spec section 8 artifacts exist")
    inventory.add_argument("--results-dir", type=Path, default=Path("."))

    args = parser.parse_args(arguments)
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
    elif args.command == "assets-lock":
        from plumb.protocol import AssetLock, build_asset_lock
        existing = AssetLock.load(args.output) if args.output.exists() else None
        lock = build_asset_lock(existing)
        digest = lock.save(args.output)
        summary = lock.to_mapping()["summary"]
        print(json.dumps({"path": str(args.output), "sha256": digest, "summary": summary}, indent=2))
        if summary["unresolved"]:
            print(
                "\n%d of %d assets are unresolved. They stay unverified until an actual retrieval\n"
                "records a hash, byte length and license. Gate A is blocked until then."
                % (summary["unresolved"], summary["total"])
            )
    elif args.command == "freeze-protocol":
        from plumb.protocol import AssetLock, default_protocol, freeze
        lock_hash = None
        if args.asset_lock.exists():
            lock = AssetLock.load(args.asset_lock)
            lock_hash = lock.to_mapping()["summary"]["status"]
            from plumb.protocol import canonical_sha256
            lock_hash = canonical_sha256(lock.to_mapping())
        document = freeze(
            default_protocol(protocol_id=args.protocol_id, seed=args.seed, starts_per_task=args.starts),
            scenario_manifest_hash=args.scenario_manifest_hash,
            asset_lock_hash=lock_hash,
            operating_point_id=args.operating_point,
        )
        document.save(args.output)
        bare = document.sha256.split(":", 1)[-1]
        print(json.dumps({"path": str(args.output), "protocol_hash": document.sha256, "frozen": True}, indent=2))
        print(
            "\nThis is a local hash, which is NOT preregistration. To create the externally\n"
            "timestamped record spec section 6 requires, run:\n\n"
            "  git add %s && git commit -m 'Freeze %s'\n"
            "  git tag -s %s -m 'protocol_sha256 %s'\n"
            "  git push origin %s\n\n"
            "Then check it with: kosmos prereg-status"
            % (args.output, args.protocol_id, document.preregistration_tag, bare, document.preregistration_tag)
        )
    elif args.command == "prereg-status":
        from plumb.protocol import ProtocolDocument
        document = ProtocolDocument.load(args.protocol)
        record = document.preregistration_status()
        print(json.dumps({"protocol_hash": document.sha256, **record.to_mapping()}, indent=2))
    elif args.command == "gates":
        from plumb.artifacts import ArtifactStore
        from plumb.gates import GATE_DESCRIPTIONS, GateLedger
        store = ArtifactStore(args.results_dir)
        ledger = GateLedger.load(str(store.gates_path))
        rows = []
        for gate_id, description in GATE_DESCRIPTIONS.items():
            record = ledger.records[gate_id]
            rows.append(
                {
                    "gate": gate_id,
                    "description": description,
                    "status": record.status.value,
                    "evidence_errors": list(record.pass_evidence_errors()),
                }
            )
        print(json.dumps({"source": store.gates_path.as_posix(), "gates": rows}, indent=2))
    elif args.command == "publish":
        from plumb.artifacts import ArtifactStore
        from plumb.engine import RunService
        from plumb.gates import GateLedger
        from plumb.measurement import analyze
        from plumb.protocol import ProtocolDocument, analysis_options_for_run, protocol_payload
        store = ArtifactStore(args.results_dir)
        service = RunService(args.data_dir, gates_path=store.gates_path)
        run = service.get_run(args.run_id)
        rows = service.list_episodes(args.run_id, limit=10000)
        options = run.get("config")
        proto_payload = {}
        if args.protocol.exists():
            document = ProtocolDocument.load(args.protocol)
            proto_payload = protocol_payload(document)
            if document.frozen:
                options = analysis_options_for_run(
                    document, rows, cohort=str((run.get("config") or {}).get("cohort") or "primary")
                )
        result = analyze(rows, options)
        receipts = {
            "per_episode": store.write_per_episode(args.run_id, rows, run).to_mapping(),
            "run_ledger": store.write_run_ledger(service.list_runs()).to_mapping(),
            "leaderboard": store.write_leaderboard(result, run, proto_payload).to_mapping(),
            "reliability": store.write_reliability(result).to_mapping(),
            "exclusions": store.write_exclusions(result).to_mapping(),
            "gates": store.write_gates(GateLedger.load(str(store.gates_path))).to_mapping(),
        }
        print(json.dumps({"run_id": args.run_id, "receipts": receipts}, indent=2))
    elif args.command == "artifacts":
        from plumb.artifacts import ArtifactStore
        print(json.dumps(ArtifactStore(args.results_dir).inventory(), indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
