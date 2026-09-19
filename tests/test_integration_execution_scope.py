from types import SimpleNamespace

from plumb.engine import RunService
from plumb.gates import GateLedger, GateStatus
from scripts.rehearse import seed_rehearsal_gates


def test_rehearsal_can_execute_without_promoting_gates_or_unlocking_real_transport(tmp_path):
    gates_path = seed_rehearsal_gates(tmp_path / "results", "sha256:" + "1" * 64)
    gates = GateLedger.load(str(gates_path))
    assert all(gates.records[g].status is GateStatus.NOT_RUN for g in ("A", "B", "C"))
    adapter = SimpleNamespace(transport_kind="simulated")
    service = RunService(tmp_path / "data", backends={"baseten": adapter}, gates_path=gates_path)
    assert service.backend_blocking_reasons("baseten") == []
    adapter.transport_kind = "real"
    reasons = service.backend_blocking_reasons("baseten")
    assert all(any("gate " + gate + " is not_run" in reason for reason in reasons) for gate in ("A", "B", "C"))
