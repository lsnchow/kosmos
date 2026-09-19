#!/bin/sh
# Start PLUMB in rehearsal mode with gates seeded, for browser verification.
#
# The Chain is SIMULATED. Nothing here measures a model, a latency or a cost --
# every episode it produces carries transport="simulated" on its ledger row and
# its artifact manifest. This exists to verify the console end to end.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PLUMB_PORT:-8788}"
WORK="${PLUMB_DEMO_DIR:-/tmp/plumb-demo}"

rm -rf "$WORK"
mkdir -p "$WORK/data" "$WORK/results-root"

"$ROOT/.venv/bin/python" - "$WORK" "$ROOT" <<'PY'
import pathlib, shutil, sys
sys.path.insert(0, ".")
from plumb.protocol import ProtocolDocument, default_protocol, freeze
from scripts.rehearse import seed_rehearsal_gates

work, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
repo_protocol = root / "protocol.json"

# Prefer the repository's frozen, preregistered protocol. Minting a fresh one
# here would give it a hash no tag covers, and the called-shot panel would then
# correctly report "no preregistration record" -- which reads as a defect on
# stage when in fact the protocol simply was not the registered one.
if repo_protocol.is_file():
    document = ProtocolDocument.load(repo_protocol)
    shutil.copy2(repo_protocol, work / "data" / "protocol.json")
    record = document.preregistration_status(repo_root=root)
    print("using the repository protocol %s (%s)" % (document.protocol_id, record.status))
else:
    document = freeze(default_protocol(starts_per_task=50, protocol_id="protocol-demo"))
    document.save(work / "data" / "protocol.json")
    print("no repository protocol; minted an unregistered one")

seed_rehearsal_gates(work / "results-root", document.sha256)
print("rehearsal gates seeded for", document.sha256[:28])
PY

PLUMB_DATA_DIR="$WORK/data" \
PLUMB_RESULTS_DIR="$WORK/results-root" \
PLUMB_REHEARSAL_CHAIN=1 \
BASETEN_WEBHOOK_SECRET=rehearsal-secret-do-not-use-in-production \
PLUMB_WEBHOOK_ENDPOINT="http://127.0.0.1:$PORT/api/callbacks" \
PLUMB_PORT="$PORT" PLUMB_MAX_WORKERS=16 PYTHONPATH="$ROOT" \
exec "$ROOT/.venv/bin/python" -m uvicorn --factory plumb.api:create_app \
  --host 127.0.0.1 --port "$PORT" --log-level warning
