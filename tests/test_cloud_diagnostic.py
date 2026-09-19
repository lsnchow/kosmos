from __future__ import annotations

import base64
import hashlib
import json
import pytest

from plumb.cloud_diagnostic import CliInvocationResult, CloudDiagnosticConfig, CloudDiagnosticService


PNG = b"\x89PNG\r\n\x1a\nfixture"


def config(**changes):
    values = {
        "model_id": "model-1",
        "deployment_id": "deployment-1",
        "cli_profile": "plumb-api",
        "current_png_base64": base64.b64encode(PNG).decode(),
        "goal_png_base64": base64.b64encode(PNG + b"goal").decode(),
        "current_png_sha256": "sha256:" + hashlib.sha256(PNG).hexdigest(),
        "goal_png_sha256": "sha256:" + hashlib.sha256(PNG + b"goal").hexdigest(),
        "prompt": "Fixed static current/goal diagnostic; do not infer task success.",
        "model_revision": "revision-1",
        "checkpoint_sha256": "c" * 64,
        "soar_revision": "d" * 40,
        "timeout_seconds": 5,
    }
    values.update(changes)
    return CloudDiagnosticConfig(**values)


def model_response(request_id: str):
    return {
        "schema_version": 1,
        "request_id": request_id,
        "status": "completed_unqualified",
        "qualified": False,
        "model": {"repo_id": "patreya/gcbc-bridge", "revision": "revision-1", "checkpoint_sha256": "c" * 64, "soar_revision": "d" * 40, "restore": {"changed": True, "optimizer_state_excluded": True, "method": "inference_params_only_source_checkpoint_no_optimizer_restore"}},
        "inputs": {
            "current_png_sha256": hashlib.sha256(PNG).hexdigest(),
            "goal_png_sha256": hashlib.sha256(PNG + b"goal").hexdigest(),
            "prompt_sha256": hashlib.sha256("Fixed static current/goal diagnostic; do not infer task success.".encode()).hexdigest(),
        },
        "actions": {"native_model_normalized": [0.1] * 7, "transformed_physical": [[0.00110581619, 0.00140863766, 0.00108366567, 0.00242279169, 0.00256719317, 0.0080822297, 1.0]], "shape": [1, 7], "finite": True},
        "timing": {"inference_seconds": 0.5},
    }


def test_durable_fixed_fixture_diagnostic_persists_input_raw_response_and_unqualified_result(tmp_path):
    calls = []

    def runner(argv, input_bytes, timeout):
        calls.append((argv, input_bytes, timeout))
        payload = json.loads(input_bytes)
        return CliInvocationResult(0, json.dumps(model_response(payload["request_id"])).encode(), b"")

    service = CloudDiagnosticService(tmp_path / "data", config(), runner)
    pending = service.submit("cloud-test-1")
    assert pending["status"] == "pending"
    completed = service.execute("cloud-test-1")
    assert completed["status"] == "completed"
    assert completed["qualified"] is False and completed["result"]["diagnostic_only"] is True
    assert completed["result"]["physical_action"][-1] == 1.0
    assert completed["result"]["timing"]["client_request_seconds"] >= 0
    assert completed["result"]["target"]["deployment_id"] == "deployment-1"
    assert calls[0][0][:4] == ["baseten", "model", "predict", "--profile"]
    assert "--file" in calls[0][0] and "--output" in calls[0][0]
    artifacts = tmp_path / "data" / "cloud-diagnostics" / "cloud-test-1"
    assert {path.name for path in artifacts.iterdir()} == {"input.json", "raw-response.json", "result.json"}
    assert service.submit("cloud-test-1")["status"] == "completed"
    assert len(calls) == 1


def test_cli_uncertainty_fails_manual_reconciliation_without_a_retry_or_fake_label(tmp_path):
    def runner(_argv, _input, _timeout):
        return CliInvocationResult(1, b"not-json", b"CLI error")

    service = CloudDiagnosticService(tmp_path / "data", config(), runner)
    service.submit("cloud-fail-1")
    failed = service.execute("cloud-fail-1")
    assert failed["status"] == "failed"
    assert failed["result"] is None
    assert failed["error"]["manual_reconciliation_required"] is True
    assert failed["error"]["automatic_retry_allowed"] is False
    assert service.execute("cloud-fail-1")["status"] == "failed"


def test_request_id_cannot_rebind_a_different_fixed_input(tmp_path):
    service = CloudDiagnosticService(tmp_path / "data", config(), lambda *_: CliInvocationResult(0, b"{}", b""))
    service.submit("cloud-idempotent")
    service.config = config(prompt="another fixed prompt")
    try:
        service.submit("cloud-idempotent")
        assert False, "an idempotency key must not bind a different input"
    except ValueError:
        pass


def test_idempotency_binds_exact_endpoint_and_restart_never_resubmits(tmp_path):
    calls = []
    runner = lambda *args: calls.append(args)
    service = CloudDiagnosticService(tmp_path / "data", config(), runner)
    service.submit("queued")
    service.config = config(deployment_id="another-deployment")
    with pytest.raises(ValueError, match="different diagnostic input"):
        service.submit("queued")
    reopened = CloudDiagnosticService(tmp_path / "data", config(), runner)
    assert reopened.get("queued")["status"] == "failed"
    assert reopened.execute("queued")["error"]["kind"] == "InterruptedBeforeSubmission"
    assert calls == []


@pytest.mark.parametrize("mutation", ["hash", "revision", "physical", "schema", "repo"])
def test_response_validation_refuses_mismatched_evidence(tmp_path, mutation):
    service = CloudDiagnosticService(tmp_path / "data", config())
    response = model_response("test")
    if mutation == "hash":
        response["inputs"]["current_png_sha256"] = "0" * 64
    elif mutation == "revision":
        response["model"]["revision"] = "wrong"
    elif mutation == "physical":
        response["actions"]["transformed_physical"][0][0] = 99
    elif mutation == "schema":
        response["schema_version"] = True
    else:
        response["model"]["repo_id"] = "another/model"
    with pytest.raises(ValueError):
        service._normalise_response(response, "test")
