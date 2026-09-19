import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from plumb.api import create_app
from plumb.platform import BasetenChainClient, BasetenPlatformConfig, ChainAsyncEndpoint


class NoNetwork:
    async def request(self, request):
        raise AssertionError("Callback ingestion and local reconciliation must not send HTTP requests")


def client_config():
    return BasetenChainClient(BasetenPlatformConfig(
        api_key="test-not-a-credential", webhook_secret="test-secret",
        chain_endpoint=ChainAsyncEndpoint.for_target("test-chain", "development")), transport=NoNetwork())


def signed(body):
    return {"x-baseten-request-id": "request-1",
            "x-baseten-signature": "v1=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest(),
            "content-type": "application/json"}


def test_callback_ingress_disabled_without_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("PLUMB_ENABLE_BASETEN_CALLBACKS", raising=False)
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/baseten/callback", content=b"{}").status_code == 503
        assert client.post("/api/baseten/reconcile").status_code == 503
        outbox = client.get("/api/baseten/outbox").json()
        assert outbox["submissions"] == []
        assert outbox["automatic_submission_enabled"] is False


def test_signed_callback_is_durable_before_association_and_after_restart(tmp_path):
    body = json.dumps({"request_id": "request-1", "type": "async_predict.completed",
                       "data": {"untrusted_score": True}}).encode()
    with TestClient(create_app(tmp_path, baseten_client=client_config())) as client:
        assert client.post("/api/baseten/callback", content=body).status_code == 401
        response = client.post("/api/baseten/callback", content=body, headers=signed(body))
        assert response.status_code == 202, response.text
        assert response.json() == {"request_id": "request-1", "persisted": True, "duplicate": False,
                                   "associated": False, "logical_result_finalized": False}
    with TestClient(create_app(tmp_path, baseten_client=client_config())) as client:
        repeated = client.post("/api/baseten/callback", content=body, headers=signed(body))
        assert repeated.status_code == 202 and repeated.json()["duplicate"] is True
        recovery = client.post("/api/baseten/reconcile")
        assert recovery.status_code == 200 and recovery.json()["submission_posts"] == 0
        assert client.get("/api/runs").json()["runs"] == []


def test_callback_size_limit_and_authenticated_malformed_input(tmp_path):
    with TestClient(create_app(tmp_path, baseten_client=client_config())) as client:
        assert client.post("/api/baseten/callback", content=b"x" * (1024 * 1024 + 1)).status_code == 413
        bad = b"{}"
        assert client.post("/api/baseten/callback", content=bad, headers=signed(bad)).status_code == 422
