"""Shared record contracts for the local PLUMB control plane.

This module intentionally describes *run bookkeeping*, not a qualified robot
evaluation.  The only backend implemented by the local engine is an explicitly
labelled synthetic engineering fixture.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_POLICIES = (
    "OpenVLA",
    "OpenPiZero",
    # The public/reference key maps to the required Octo-Small v1.0 wrapper.
    "Octo",
    "MiniVLA",
    "SuSIE",
    "SuSIE_LL",
)
DEFAULT_TASKS = (
    "open_drawer",
    "close_drawer",
    "to_basket",
    "to_sink",
    "fold_cloth",
)

# These are declared control horizons from the build specification, retained as
# fixture metadata only.  They are not evidence that any policy was executed.
SYNTHETIC_HORIZONS = {
    "open_drawer": 70,
    "close_drawer": 70,
    "to_basket": 100,
    "to_sink": 100,
    "fold_cloth": 80,
}

SYNTHETIC_LABEL = "synthetic engineering test artifact; unqualified; not real robot physics"


class ConfigurationError(ValueError):
    """Raised when a caller requests an unsupported or unsafe run config."""


def utc_now() -> str:
    """Return an unambiguous UTC timestamp suitable for persisted records."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def lease_deadline(seconds: float) -> str:
    """Return a UTC lease deadline; callers validate the positive duration."""

    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _string_list(value: Any, field: str, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigurationError("%s must be a non-empty list of strings" % field)
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ConfigurationError("%s must contain only non-empty strings" % field)
    items = [item.strip() for item in value]
    if len(set(items)) != len(items):
        raise ConfigurationError("%s must not contain duplicates" % field)
    return items


def normalise_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and fill a local-run configuration.

    The public control plane must not present synthetic fixtures as qualified
    model results.  Consequently, qualified modes are rejected until the
    external gates have actual evidence; arbitrary backend names are rejected
    rather than silently falling back to fixture output.
    """

    if not isinstance(config, Mapping):
        raise ConfigurationError("config must be a mapping")
    try:
        # This also catches non-serializable values before a run is committed.
        supplied = json.loads(canonical_json(dict(config)))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("config must be JSON serializable") from exc

    mode = supplied.get("mode", "synthetic")
    if mode == "qualified":
        raise ConfigurationError(
            "qualified mode is unavailable: PLUMB qualification gates A-F have not passed"
        )
    if mode != "synthetic":
        raise ConfigurationError("only mode='synthetic' is available in this local control engine")

    backend = supplied.get("backend", "synthetic")
    if backend != "synthetic":
        raise ConfigurationError("only backend='synthetic' is configured locally")

    starts = supplied.get("starts_per_task", 50)
    if isinstance(starts, bool) or not isinstance(starts, int) or starts < 1 or starts > 10_000:
        raise ConfigurationError("starts_per_task must be an integer from 1 through 10000")

    seed = supplied.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("seed must be an integer")

    workers = supplied.get("max_workers", 12)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1 or workers > 32:
        raise ConfigurationError("max_workers must be an integer from 1 through 32")

    idempotency_key = supplied.get("idempotency_key")
    if idempotency_key is not None and (
        not isinstance(idempotency_key, str) or not idempotency_key.strip() or len(idempotency_key) > 256
    ):
        raise ConfigurationError("idempotency_key must be a non-empty string of at most 256 characters")

    normalised = dict(supplied)
    normalised.update(
        {
            "mode": mode,
            "backend": backend,
            "policies": _string_list(supplied.get("policies"), "policies", DEFAULT_POLICIES),
            "tasks": _string_list(supplied.get("tasks"), "tasks", DEFAULT_TASKS),
            "starts_per_task": starts,
            "seed": seed,
            "max_workers": workers,
            "qualification": "unqualified",
            "fixture_label": SYNTHETIC_LABEL,
        }
    )
    if idempotency_key is None:
        normalised.pop("idempotency_key", None)
    else:
        normalised["idempotency_key"] = idempotency_key.strip()
    total = len(normalised["policies"]) * len(normalised["tasks"]) * starts
    if total > 10_000:
        raise ConfigurationError("the logical episode matrix exceeds the 10000-episode local limit")
    return normalised


def deterministic_seed(base_seed: int, policy: str, task: str, start_id: str) -> int:
    """Create a reproducible per-cell seed without relying on Python hash()."""

    payload = "%d\x1f%s\x1f%s\x1f%s" % (base_seed, policy, task, start_id)
    # SQLite INTEGER is signed 64-bit. Keep the deterministic stream inside
    # that domain instead of making storage depend on a backend-specific text
    # conversion.
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def start_id_for(index: int) -> str:
    return "start-%03d" % index


def start_lineage_for(task: str, index: int) -> str:
    return "synthetic-fixture:%s:%03d" % (task, index)


def horizon_for(task: str) -> int:
    return SYNTHETIC_HORIZONS.get(task, 70)


def decode_backend_output(value: Any) -> Dict[str, Any]:
    """Accept JSON emitted by an offline worker adapter.

    A future process-backed adapter can return a mapping, JSON bytes/string, a
    JSON file path, or ``{"offline_output_path": ...}``.  The engine still
    persists and validates the terminal record in its own durable ledger.
    """

    if isinstance(value, Mapping):
        offline_path = value.get("offline_output_path")
        if offline_path is not None:
            return decode_backend_output(Path(str(offline_path)))
        return dict(value)
    if isinstance(value, Path):
        try:
            with value.open("r", encoding="utf-8") as handle:
                parsed = json.load(handle)
        except OSError as exc:
            raise ValueError("unable to read offline worker output") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("offline worker output must be a JSON object")
        return dict(parsed)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_file():
            return decode_backend_output(candidate)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("backend string output must be JSON or an existing JSON path") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("backend JSON output must be an object")
        return dict(parsed)
    raise ValueError("backend output must be an object or JSON offline-worker output")


def normalise_backend_result(
    result: Mapping[str, Any], episode: Mapping[str, Any], artifact_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Give adapter output a safe, explicit measurement shape.

    Null is retained as null for unknown/unevaluable outputs.  In particular,
    no missing label is converted to a failure or a success.
    """

    raw = dict(result)
    validity = raw.get("validity", "unknown")
    if validity not in ("valid", "invalid", "unknown"):
        raise ValueError("backend validity must be valid, invalid, or unknown")
    binary_success = raw.get("binary_success")
    if binary_success not in (None, True, False, 0, 1):
        raise ValueError("backend binary_success must be true, false, or null")
    progress_score = raw.get("progress_score")
    if progress_score is not None and (
        isinstance(progress_score, bool) or not isinstance(progress_score, int) or not 0 <= progress_score <= 5
    ):
        raise ValueError("backend progress_score must be an integer from 0 through 5 or null")
    if validity != "valid":
        if binary_success is not None or progress_score is not None:
            raise ValueError("invalid/unknown backend output must keep binary_success and progress_score null")
    elif binary_success is None:
        if progress_score is not None:
            raise ValueError("nullable binary_success requires nullable progress_score")
    else:
        binary_success = bool(binary_success)
        if binary_success and progress_score != 5:
            raise ValueError("binary_success=true requires progress_score=5")
        if not binary_success and progress_score not in (0, 1, 2, 3, 4):
            raise ValueError("binary_success=false requires progress_score from 0 through 4")

    missing_reason = raw.get("missing_reason")
    if binary_success is None:
        missing_reason = str(missing_reason or "unevaluable_backend_output")
    else:
        missing_reason = None

    artifact_refs = raw.get("artifact_refs", {})
    if not isinstance(artifact_refs, Mapping):
        raise ValueError("backend artifact_refs must be an object")
    artifact_refs = _validate_synthetic_artifact_refs(dict(artifact_refs), episode, artifact_root)
    timing = raw.get("timing", {})
    if not isinstance(timing, Mapping):
        raise ValueError("backend timing must be an object")

    return {
        "validity": validity,
        "binary_success": binary_success,
        "progress_score": progress_score,
        "missing_reason": missing_reason,
        "horizon_actions": int(raw.get("horizon_actions", episode["horizon_actions"])),
        "artifact_refs": artifact_refs,
        "timing": dict(timing),
        "backend_metadata": dict(raw.get("backend_metadata", {}))
        if isinstance(raw.get("backend_metadata", {}), Mapping)
        else {},
    }


def _validate_synthetic_artifact_refs(
    refs: Dict[str, Any], episode: Mapping[str, Any], artifact_root: Optional[Path]
) -> Dict[str, Dict[str, str]]:
    """Accept only hash-verified, own-attempt synthetic artifacts.

    The local control plane may exercise an injected backend for tests, but it
    is still a *synthetic* run.  A result cannot smuggle an arbitrary URL (or a
    real-looking clip outside its own attempt directory) into the dashboard.
    Empty refs remain valid for narrow fake-backend failure/recovery tests.
    """

    if not refs:
        return {}
    if artifact_root is None:
        raise ValueError("nonempty artifact_refs require an own-attempt artifact directory")
    root = artifact_root.resolve()
    if not root.is_dir():
        raise ValueError("own-attempt artifact directory does not exist")
    data_root = root.parents[3]
    allowed_media = {"image/svg+xml": ".svg", "application/json": ".json"}
    validated: Dict[str, Dict[str, str]] = {}
    for name, ref in refs.items():
        if not isinstance(name, str) or not isinstance(ref, Mapping):
            raise ValueError("artifact_refs must contain named typed artifact mappings; top-level URLs are forbidden")
        relative = ref.get("relative_path")
        media_type = ref.get("media_type")
        digest = ref.get("sha256")
        if not isinstance(relative, str) or not isinstance(media_type, str) or not isinstance(digest, str):
            raise ValueError("artifact refs require relative_path, media_type, and sha256")
        if media_type not in allowed_media or not digest.startswith("sha256:"):
            raise ValueError("synthetic artifact refs require an allowed typed media value and SHA-256")
        candidate = (data_root / relative).resolve()
        if root not in candidate.parents or candidate.suffix.lower() != allowed_media[media_type] or not candidate.is_file():
            raise ValueError("synthetic artifact must resolve under this episode attempt directory")
        if file_digest(candidate) != digest:
            raise ValueError("synthetic artifact SHA-256 does not match persisted bytes")
        expected_uri = "artifact://%s" % relative
        expected_url = "/api/artifacts/%s" % relative
        if ref.get("uri", expected_uri) != expected_uri or ref.get("url", expected_url) != expected_url:
            raise ValueError("synthetic artifact references may not use arbitrary URLs")
        if ref.get("artifact_path", relative) != relative:
            raise ValueError("synthetic artifact_path must equal data-root-relative path")
        label = str(ref.get("label", "")).lower()
        if "synthetic" not in label or "unqualified" not in label:
            raise ValueError("synthetic artifacts must retain synthetic/unqualified labels")
        validated[name] = {
            "uri": expected_uri,
            "relative_path": relative,
            "artifact_path": relative,
            "url": expected_url,
            "sha256": digest,
            "media_type": media_type,
            "label": str(ref["label"]),
        }
    manifest = validated.get("manifest")
    frame = validated.get("fixture_frame")
    if manifest is None or frame is None:
        raise ValueError("nonempty synthetic output requires fixture_frame and manifest artifacts")
    try:
        with (data_root / manifest["relative_path"]).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError("synthetic artifact manifest is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("synthetic artifact manifest must be an object")
    claims = payload.get("claims")
    if (
        payload.get("kind") != "synthetic_engineering_fixture"
        or "synthetic" not in str(payload.get("label", "")).lower()
        or payload.get("run_id") != episode.get("run_id")
        or payload.get("episode_id") != episode.get("episode_id")
        or not isinstance(claims, Mapping)
        or any(claims.get(name) is not False for name in ("real_robot", "physical_control", "world_model", "qualified_measurement"))
    ):
        raise ValueError("artifact manifest is not an own unqualified synthetic fixture")
    return validated
