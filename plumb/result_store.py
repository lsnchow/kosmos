"""Fail-closed, S3-compatible storage for immutable Chain terminal results.

This module deliberately separates result *identity* from result *location*.
The application persists only an opaque ``result_key`` in its durable outbox;
the bucket, endpoint, credentials, and object prefix are deployment
configuration.  No presigned URL or credential can therefore leak through an
async Chain payload, the local ledger, callback history, or application logs.

The implementation imports :mod:`boto3` only when an operator actually reads,
writes, or explicitly preflights the configured store.  Constructing a config
or registering the backend never performs network I/O.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Optional, Protocol
from urllib.parse import urlsplit

from .records import canonical_json


RESULT_STORE_SCHEMA_VERSION = 1
DEFAULT_MAX_RESULT_BYTES = 8 * 1024 * 1024
_KEY_RE = re.compile(r"^[a-f0-9]{32}$")
_DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")


class ResultStoreError(RuntimeError):
    """Base class for a result-store operation that must not be guessed."""


class ResultStoreConfigurationError(ResultStoreError, ValueError):
    """The opt-in result-store configuration is incomplete or unsafe."""


class ResultStoreConflictError(ResultStoreError):
    """An immutable result key already contains different durable content."""


class ResultStoreIntegrityError(ResultStoreError):
    """A downloaded manifest is malformed, oversized, or identity-mismatched."""


@dataclass(frozen=True, repr=False)
class S3Credentials:
    """Runtime-only AWS credential values resolved from deployment context."""

    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.access_key_id or not self.secret_access_key:
            raise ResultStoreConfigurationError("result-store AWS access and secret key values are required")


class S3CompatibleClient(Protocol):
    """Minimal boto3 client surface, intentionally small for fake-client tests."""

    def put_object(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def get_object(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def head_bucket(self, **kwargs: Any) -> Mapping[str, Any]: ...


def _required(values: Mapping[str, str], name: str) -> str:
    value = str(values.get(name, "")).strip()
    if not value:
        raise ResultStoreConfigurationError("%s is required for the immutable result store" % name)
    return value


def _normalise_prefix(value: str) -> str:
    prefix = value.strip().strip("/")
    if not prefix or "//" in prefix or any(part in ("", ".", "..") for part in prefix.split("/")):
        raise ResultStoreConfigurationError("PLUMB_RESULT_STORE_PREFIX must be a non-empty relative object prefix")
    if not all(re.fullmatch(r"[A-Za-z0-9._=-]+", part) for part in prefix.split("/")):
        raise ResultStoreConfigurationError("PLUMB_RESULT_STORE_PREFIX contains unsupported object-key characters")
    return prefix


@dataclass(frozen=True)
class S3ResultStoreConfig:
    """Explicit, secret-free connection details for a pre-provisioned store."""

    bucket: str
    region: str
    endpoint_url: str
    prefix: str
    evidence_uri: str
    verified_at: str
    expected_bucket_owner: Optional[str] = None
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES
    connect_timeout_seconds: float = 3.0
    read_timeout_seconds: float = 10.0
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", self.bucket):
            raise ResultStoreConfigurationError("result-store bucket must be an explicit DNS-compatible bucket name")
        if not re.fullmatch(r"[a-z0-9-]{2,64}", self.region):
            raise ResultStoreConfigurationError("result-store region is invalid")
        parsed = urlsplit(self.endpoint_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ResultStoreConfigurationError("result-store endpoint must be an absolute HTTPS URL without credentials/query")
        if parsed.path not in ("", "/"):
            raise ResultStoreConfigurationError("result-store endpoint must not include an object path")
        _normalise_prefix(self.prefix)
        if not self.evidence_uri.strip():
            raise ResultStoreConfigurationError("result-store evidence_uri is required")
        try:
            value = self.verified_at.replace("Z", "+00:00")
            parsed_time = datetime.fromisoformat(value)
        except ValueError as error:
            raise ResultStoreConfigurationError("result-store verified_at must be ISO-8601") from error
        if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
            raise ResultStoreConfigurationError("result-store verified_at must include a timezone")
        if self.expected_bucket_owner is not None and not re.fullmatch(r"\d{12}", self.expected_bucket_owner):
            raise ResultStoreConfigurationError("result-store expected_bucket_owner must be a 12-digit account ID")
        if isinstance(self.max_result_bytes, bool) or not isinstance(self.max_result_bytes, int) or not 1 <= self.max_result_bytes <= 64 * 1024 * 1024:
            raise ResultStoreConfigurationError("result-store max_result_bytes must be an integer from 1 through 67108864")
        for name, value in (("connect_timeout_seconds", self.connect_timeout_seconds), ("read_timeout_seconds", self.read_timeout_seconds)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) <= 60:
                raise ResultStoreConfigurationError("result-store %s must be a number from 0 through 60" % name)
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or not 1 <= self.max_attempts <= 5:
            raise ResultStoreConfigurationError("result-store max_attempts must be an integer from 1 through 5")

    @property
    def normalised_prefix(self) -> str:
        return _normalise_prefix(self.prefix)

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "S3ResultStoreConfig":
        import os

        values = os.environ if environ is None else environ
        maximum = str(values.get("PLUMB_RESULT_STORE_MAX_BYTES", DEFAULT_MAX_RESULT_BYTES)).strip()
        try:
            max_bytes = int(maximum)
            connect_timeout = float(str(values.get("PLUMB_RESULT_STORE_CONNECT_TIMEOUT_SECONDS", "3")).strip())
            read_timeout = float(str(values.get("PLUMB_RESULT_STORE_READ_TIMEOUT_SECONDS", "10")).strip())
            max_attempts = int(str(values.get("PLUMB_RESULT_STORE_MAX_ATTEMPTS", "2")).strip())
        except ValueError as error:
            raise ResultStoreConfigurationError("result-store timeout/size configuration is invalid") from error
        owner = str(values.get("PLUMB_RESULT_STORE_EXPECTED_BUCKET_OWNER", "")).strip() or None
        return cls(
            bucket=_required(values, "PLUMB_RESULT_STORE_BUCKET"),
            region=_required(values, "PLUMB_RESULT_STORE_REGION"),
            endpoint_url=_required(values, "PLUMB_RESULT_STORE_ENDPOINT_URL"),
            prefix=_required(values, "PLUMB_RESULT_STORE_PREFIX"),
            evidence_uri=_required(values, "PLUMB_RESULT_STORE_EVIDENCE_URI"),
            verified_at=_required(values, "PLUMB_RESULT_STORE_VERIFIED_AT"),
            expected_bucket_owner=owner,
            max_result_bytes=max_bytes,
            connect_timeout_seconds=connect_timeout,
            read_timeout_seconds=read_timeout,
            max_attempts=max_attempts,
        )


@dataclass(frozen=True)
class ResultStoreBinding:
    """Secret-free pre-POST binding carried by the request and durable outbox."""

    result_key: str
    run_id: str
    episode_id: str
    protocol_hash: str
    request_payload_sha256: str

    def __post_init__(self) -> None:
        if not _KEY_RE.fullmatch(self.result_key):
            raise ResultStoreIntegrityError("result_key must be an opaque lowercase 32-hex token")
        for name, value in (
            ("run_id", self.run_id),
            ("episode_id", self.episode_id),
            ("protocol_hash", self.protocol_hash),
        ):
            if not _IDENTITY_RE.fullmatch(value):
                raise ResultStoreIntegrityError("result-store %s is invalid" % name)
        if not _DIGEST_RE.fullmatch(self.request_payload_sha256):
            raise ResultStoreIntegrityError("result-store request_payload_sha256 must be a SHA-256 digest")

    @classmethod
    def new(
        cls, run_id: str, episode_id: str, protocol_hash: str, request_payload_sha256: str
    ) -> "ResultStoreBinding":
        return cls(uuid.uuid4().hex, run_id, episode_id, protocol_hash, request_payload_sha256)

    def as_mapping(self) -> dict[str, str]:
        return {
            "result_key": self.result_key,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "protocol_hash": self.protocol_hash,
            "request_payload_sha256": self.request_payload_sha256,
        }


def request_payload_digest(request_without_result_store: Mapping[str, Any]) -> str:
    """Hash request inputs excluding the binding that would otherwise be circular."""

    value = dict(request_without_result_store)
    value.pop("result_store", None)
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResultManifest:
    """Full terminal output plus the identity that authorizes reading it."""

    binding: ResultStoreBinding
    result: Mapping[str, Any]
    created_at: str
    result_sha256: str

    @classmethod
    def create(cls, binding: ResultStoreBinding, result: Mapping[str, Any]) -> "ResultManifest":
        copied = dict(result)
        _validate_terminal_result(copied, binding)
        encoded = canonical_json(copied).encode("utf-8")
        return cls(
            binding=binding,
            result=copied,
            created_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            result_sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_STORE_SCHEMA_VERSION,
            "binding": self.binding.as_mapping(),
            "result": dict(self.result),
            "result_sha256": self.result_sha256,
            "created_at": self.created_at,
        }

    def encoded(self) -> bytes:
        return canonical_json(self.as_mapping()).encode("utf-8")

    @classmethod
    def decode(cls, body: bytes, expected: ResultStoreBinding) -> "ResultManifest":
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ResultStoreIntegrityError("stored result manifest is not UTF-8 JSON") from error
        if not isinstance(value, Mapping) or value.get("schema_version") != RESULT_STORE_SCHEMA_VERSION:
            raise ResultStoreIntegrityError("stored result manifest has an unsupported schema")
        if set(value) != {"schema_version", "binding", "result", "result_sha256", "created_at"}:
            raise ResultStoreIntegrityError("stored result manifest has unexpected fields")
        raw_binding = value.get("binding")
        result = value.get("result")
        digest = value.get("result_sha256")
        created_at = value.get("created_at")
        if not isinstance(raw_binding, Mapping) or not isinstance(result, Mapping) or not isinstance(digest, str) or not isinstance(created_at, str):
            raise ResultStoreIntegrityError("stored result manifest is missing required fields")
        try:
            binding = ResultStoreBinding(**dict(raw_binding))
        except (TypeError, ResultStoreError, ValueError) as error:
            raise ResultStoreIntegrityError("stored result manifest has an invalid binding") from error
        if binding != expected:
            raise ResultStoreIntegrityError("stored result manifest does not match the durable result binding")
        actual = "sha256:" + hashlib.sha256(canonical_json(dict(result)).encode("utf-8")).hexdigest()
        if digest != actual:
            raise ResultStoreIntegrityError("stored result manifest result_sha256 does not match result bytes")
        _validate_terminal_result(dict(result), expected)
        return cls(binding=binding, result=dict(result), created_at=created_at, result_sha256=digest)


@dataclass(frozen=True)
class ResultStoreReceipt:
    """Evidence returned only after an operator explicitly invokes a store call."""

    operation: str
    bucket: str
    endpoint_url: str
    timestamp: str
    request_id: Optional[str]
    result_key: Optional[str] = None

    def as_mapping(self) -> dict[str, Optional[str]]:
        return {
            "operation": self.operation,
            "bucket": self.bucket,
            "endpoint_url": self.endpoint_url,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "result_key": self.result_key,
        }


def _validate_terminal_result(result: Mapping[str, Any], binding: ResultStoreBinding) -> None:
    if result.get("status") not in {"completed", "blocked", "failed"}:
        raise ResultStoreIntegrityError("stored result does not declare a terminal status")
    for field, expected_value in (
        ("run_id", binding.run_id),
        ("episode_id", binding.episode_id),
        ("protocol_hash", binding.protocol_hash),
    ):
        if result.get(field) != expected_value:
            raise ResultStoreIntegrityError("stored result does not match durable %s" % field)


class S3ResultStore:
    """Conditional-write S3-compatible result store with verified reads."""

    def __init__(
        self,
        config: S3ResultStoreConfig,
        client: Optional[S3CompatibleClient] = None,
        credentials: Optional[S3Credentials] = None,
    ) -> None:
        self.config = config
        self._client = client
        self._credentials = credentials

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "S3ResultStore":
        return cls(S3ResultStoreConfig.from_env(environ))

    def sdk_available(self) -> bool:
        """Whether the optional boto3 extra is installed; this makes no I/O."""

        return self._client is not None or importlib.util.find_spec("boto3") is not None

    def _sdk_client(self) -> S3CompatibleClient:
        if self._client is not None:
            return self._client
        if not self.sdk_available():
            raise ResultStoreConfigurationError("boto3 is required only when using the configured result store")
        import boto3
        from botocore.config import Config

        kwargs: dict[str, Any] = {
            "region_name": self.config.region,
            "endpoint_url": self.config.endpoint_url,
            "config": Config(
                connect_timeout=float(self.config.connect_timeout_seconds),
                read_timeout=float(self.config.read_timeout_seconds),
                retries={"max_attempts": self.config.max_attempts, "mode": "standard"},
            ),
        }
        if self._credentials is not None:
            kwargs.update(
                {
                    "aws_access_key_id": self._credentials.access_key_id,
                    "aws_secret_access_key": self._credentials.secret_access_key,
                    "aws_session_token": self._credentials.session_token,
                }
            )
        self._client = boto3.client("s3", **kwargs)
        return self._client

    def object_key(self, result_key: str) -> str:
        if not _KEY_RE.fullmatch(result_key):
            raise ResultStoreIntegrityError("result_key must be an opaque lowercase 32-hex token")
        return "%s/%s.json" % (self.config.normalised_prefix, result_key)

    def _base_arguments(self) -> MutableMapping[str, Any]:
        values: MutableMapping[str, Any] = {"Bucket": self.config.bucket}
        if self.config.expected_bucket_owner:
            values["ExpectedBucketOwner"] = self.config.expected_bucket_owner
        return values

    @staticmethod
    def _request_id(response: Mapping[str, Any]) -> Optional[str]:
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, Mapping):
            value = metadata.get("RequestId")
            return str(value) if value else None
        return None

    @staticmethod
    def _error_code(error: BaseException) -> Optional[str]:
        response = getattr(error, "response", None)
        if isinstance(response, Mapping):
            detail = response.get("Error")
            if isinstance(detail, Mapping) and detail.get("Code") is not None:
                return str(detail["Code"])
        return None

    def put_terminal_result(self, binding: ResultStoreBinding, result: Mapping[str, Any]) -> ResultStoreReceipt:
        manifest = ResultManifest.create(binding, result)
        encoded = manifest.encoded()
        if len(encoded) > self.config.max_result_bytes:
            raise ResultStoreIntegrityError("terminal result manifest exceeds configured maximum size")
        arguments = self._base_arguments()
        arguments.update(
            {
                "Key": self.object_key(binding.result_key),
                "Body": encoded,
                "ContentType": "application/json",
                "IfNoneMatch": "*",
                "Metadata": {"result-sha256": manifest.result_sha256[7:]},
            }
        )
        try:
            response = self._sdk_client().put_object(**arguments)
        except Exception as error:  # boto3 client errors are intentionally not a hard dependency
            if self._error_code(error) not in {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}:
                raise ResultStoreError("conditional result-store write failed: %s" % type(error).__name__) from error
            existing = self.get_manifest(binding)
            if existing is None or existing.result_sha256 != manifest.result_sha256:
                raise ResultStoreConflictError("immutable result key already holds different content") from error
            return ResultStoreReceipt(
                operation="put_terminal_result_existing",
                bucket=self.config.bucket,
                endpoint_url=self.config.endpoint_url,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                request_id=None,
                result_key=binding.result_key,
            )
        return ResultStoreReceipt(
            operation="put_terminal_result",
            bucket=self.config.bucket,
            endpoint_url=self.config.endpoint_url,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            request_id=self._request_id(response),
            result_key=binding.result_key,
        )

    def get_manifest(self, binding: ResultStoreBinding) -> Optional[ResultManifest]:
        arguments = self._base_arguments()
        arguments["Key"] = self.object_key(binding.result_key)
        try:
            response = self._sdk_client().get_object(**arguments)
        except Exception as error:
            if self._error_code(error) in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise ResultStoreError("result-store read failed: %s" % type(error).__name__) from error
        length = response.get("ContentLength")
        if isinstance(length, int) and length > self.config.max_result_bytes:
            raise ResultStoreIntegrityError("stored result manifest exceeds configured maximum size")
        body = response.get("Body")
        if body is None or not callable(getattr(body, "read", None)):
            raise ResultStoreIntegrityError("result-store response has no readable body")
        try:
            raw = body.read(self.config.max_result_bytes + 1)
            if not isinstance(raw, bytes) or len(raw) > self.config.max_result_bytes:
                raise ResultStoreIntegrityError("stored result manifest exceeds configured maximum size")
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        return ResultManifest.decode(raw, binding)

    def get_result(self, binding: ResultStoreBinding) -> Optional[Mapping[str, Any]]:
        manifest = self.get_manifest(binding)
        return None if manifest is None else dict(manifest.result)

    def preflight(self) -> ResultStoreReceipt:
        """Explicitly test bucket reachability; never called during registration."""

        response = self._sdk_client().head_bucket(**self._base_arguments())
        return ResultStoreReceipt(
            operation="head_bucket",
            bucket=self.config.bucket,
            endpoint_url=self.config.endpoint_url,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            request_id=self._request_id(response),
        )


__all__ = [
    "DEFAULT_MAX_RESULT_BYTES",
    "RESULT_STORE_SCHEMA_VERSION",
    "ResultManifest",
    "ResultStoreBinding",
    "ResultStoreConfigurationError",
    "ResultStoreConflictError",
    "ResultStoreError",
    "ResultStoreIntegrityError",
    "ResultStoreReceipt",
    "S3CompatibleClient",
    "S3Credentials",
    "S3ResultStore",
    "S3ResultStoreConfig",
    "request_payload_digest",
]
