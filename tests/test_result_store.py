from __future__ import annotations

import json

import pytest

from plumb.result_store import (
    ResultStoreBinding,
    ResultStoreConfigurationError,
    ResultStoreIntegrityError,
    S3ResultStore,
    S3ResultStoreConfig,
    request_payload_digest,
)


class ClientError(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}


class Body:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self.data if size < 0 else self.data[:size]

    def close(self) -> None:
        self.closed = True


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.puts: list[dict] = []
        self.heads = 0
        self.last_body = None

    def put_object(self, **kwargs):
        self.puts.append(dict(kwargs))
        identity = (kwargs["Bucket"], kwargs["Key"])
        if kwargs.get("IfNoneMatch") == "*" and identity in self.objects:
            raise ClientError("PreconditionFailed")
        self.objects[identity] = bytes(kwargs["Body"])
        return {"ResponseMetadata": {"RequestId": "put-1"}}

    def get_object(self, **kwargs):
        identity = (kwargs["Bucket"], kwargs["Key"])
        if identity not in self.objects:
            raise ClientError("NoSuchKey")
        value = self.objects[identity]
        self.last_body = Body(value)
        return {"ContentLength": len(value), "Body": self.last_body, "ResponseMetadata": {"RequestId": "get-1"}}

    def head_bucket(self, **kwargs):
        self.heads += 1
        return {"ResponseMetadata": {"RequestId": "head-1"}}


def config(**changes):
    values = {
        "bucket": "plumb-results-bucket",
        "region": "us-east-1",
        "endpoint_url": "https://s3.example.test",
        "prefix": "plumb/terminal-results",
        "evidence_uri": "artifact://operations/result-store-route.json",
        "verified_at": "2026-09-19T00:00:00Z",
    }
    values.update(changes)
    return S3ResultStoreConfig(**values)


def binding(**changes):
    values = {
        "result_key": "a" * 32,
        "run_id": "run-1",
        "episode_id": "episode-1",
        "protocol_hash": "sha256:" + "b" * 64,
        "request_payload_sha256": "sha256:" + "c" * 64,
    }
    values.update(changes)
    return ResultStoreBinding(**values)


def result():
    return {
        "run_id": "run-1",
        "episode_id": "episode-1",
        "protocol_hash": "sha256:" + "b" * 64,
        "status": "completed",
        "horizon_actions": 70,
        "executed_actions": 70,
    }


def test_result_store_uses_conditional_create_and_verifies_the_readback():
    client = FakeS3()
    store = S3ResultStore(config(), client=client)
    receipt = store.put_terminal_result(binding(), result())

    assert receipt.operation == "put_terminal_result"
    assert client.puts[0]["IfNoneMatch"] == "*"
    assert client.puts[0]["Key"] == "plumb/terminal-results/" + "a" * 32 + ".json"
    assert "https://" not in json.dumps(client.puts[0], default=str)
    assert store.get_result(binding()) == result()
    assert client.last_body.closed is True
    # A retried identical terminal write is idempotent; different content is not.
    assert store.put_terminal_result(binding(), result()).operation == "put_terminal_result_existing"


def test_result_store_rejects_tampering_and_never_accepts_payload_selected_destinations():
    client = FakeS3()
    store = S3ResultStore(config(), client=client)
    store.put_terminal_result(binding(), result())
    location = (config().bucket, store.object_key(binding().result_key))
    payload = json.loads(client.objects[location])
    payload["binding"]["episode_id"] = "other-episode"
    client.objects[location] = json.dumps(payload).encode()
    with pytest.raises(ResultStoreIntegrityError):
        store.get_result(binding())
    with pytest.raises(ResultStoreConfigurationError):
        config(endpoint_url="http://s3.example.test")
    with pytest.raises(ResultStoreConfigurationError):
        config(prefix="../outside")


def test_manifest_normalization_rejects_unexpected_binding_fields_and_invalid_prewrite_result_identity():
    client = FakeS3()
    store = S3ResultStore(config(), client=client)
    broken = dict(result())
    broken["episode_id"] = "other-episode"
    with pytest.raises(ResultStoreIntegrityError):
        store.put_terminal_result(binding(), broken)
    assert client.puts == []

    store.put_terminal_result(binding(), result())
    location = (config().bucket, store.object_key(binding().result_key))
    payload = json.loads(client.objects[location])
    payload["binding"]["surprise"] = "not allowed"
    client.objects[location] = json.dumps(payload).encode()
    with pytest.raises(ResultStoreIntegrityError, match="invalid binding"):
        store.get_result(binding())


def test_configuration_and_preflight_are_explicit_and_do_not_contact_the_store_at_construction():
    client = FakeS3()
    store = S3ResultStore(config(), client=client)
    assert client.heads == 0
    assert store.preflight().operation == "head_bucket"
    assert client.heads == 1
    with pytest.raises(ResultStoreConfigurationError, match="PLUMB_RESULT_STORE_BUCKET"):
        S3ResultStoreConfig.from_env({})


def test_request_payload_digest_excludes_only_the_circular_result_store_binding():
    unsigned = {"run_id": "run-1", "episode_id": "episode-1", "result_store": {"result_key": "a" * 32}}
    same = {"run_id": "run-1", "episode_id": "episode-1"}
    assert request_payload_digest(unsigned) == request_payload_digest(same)
