"""Shared record contracts for the PLUMB control plane.

This module describes *run bookkeeping*, not a qualified robot evaluation.  Two
kinds of backend are registered: an explicitly labelled synthetic engineering
fixture, and real-model execution on a deployed Baseten Chain.  Selecting the
latter is necessary but not sufficient -- ``plumb.engine`` additionally requires
passing gate evidence, and ``plumb.gates.QualificationValidator`` remains the
only thing that can call a resulting cell qualified.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


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

#: Execution modes a caller may request.  ``qualified`` is deliberately absent:
#: qualification is decided by ``plumb.gates.QualificationValidator``, never
#: asserted by a run configuration.
#:
#: ``synthetic``    engineering fixture; no learned model touches it.
#: ``diagnostic``   real models, explicitly not a primary-study cell.
#: ``qualification``real models against a frozen protocol; the gate ledger still
#:                  decides whether the resulting cells are qualified.
RUN_MODES: Dict[str, str] = {
    "synthetic": "synthetic/unqualified",
    "diagnostic": "diagnostic/unqualified",
    "qualification": "qualification/pending-gate-review",
}

#: Registered execution backends.  ``requires_gates`` backends execute real
#: models and must not start until the caller has shown passing gate evidence;
#: the engine enforces that separately because only it can see the gate ledger.
BACKENDS: Dict[str, Dict[str, Any]] = {
    "synthetic": {
        "requires_gates": False,
        "label": SYNTHETIC_LABEL,
        "description": "Local deterministic fixture. No learned model, no robot physics.",
    },
    "baseten": {
        "requires_gates": True,
        "label": "real-model execution on a deployed Baseten Chain",
        "description": "Submits logical episodes to the Chain entrypoint /async_run_remote.",
        "required_gates": ("A", "B", "C"),
    },
}

#: Gates that must carry passing evidence before a gated backend may execute.
#: Gate D blocks *primary scoring*, not execution, so a diagnostic cohort can
#: run before calibration exists; ``plumb.measurement`` refuses to publish
#: primary statistics from it.
GATED_BACKEND_REQUIRED_GATES: Tuple[str, ...] = ("A", "B", "C")

#: Cohorts whose source-state lineages must stay disjoint (spec section 6).
#: ``plumb.measurement.validate_source_lineage_leakage`` enforces the disjointness;
#: this is the closed set of names it may see.
COHORTS: Tuple[str, ...] = (
    "primary",
    "development",
    "calibration",
    "cost_confirmation",
    "reverse_validation",
    "load_rehearsal",
)


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
    """Validate and fill a run configuration.

    The control plane must not present synthetic fixtures as qualified model
    results, so ``qualified`` is not a settable mode and arbitrary backend names
    are rejected rather than silently falling back to fixture output.  A gated
    backend is accepted here as *well-formed*; ``plumb.engine`` separately
    refuses to execute it without passing gate evidence, because only the engine
    can see the gate ledger.
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
        # ``qualified`` is never a caller-supplied value.  Qualification is a
        # separate, inspectable decision made by ``plumb.gates`` from frozen
        # manifests and passed gates; a run configuration cannot assert it.
        raise ConfigurationError(
            "qualified mode is not settable by a caller: qualification is decided by "
            "plumb.gates.QualificationValidator from frozen manifests and passed gates"
        )
    if mode not in RUN_MODES:
        raise ConfigurationError(
            "mode must be one of %s" % ", ".join("'%s'" % item for item in sorted(RUN_MODES))
        )

    backend = supplied.get("backend", "synthetic")
    if backend not in BACKENDS:
        raise ConfigurationError(
            "backend must be one of %s" % ", ".join("'%s'" % item for item in sorted(BACKENDS))
        )
    if BACKENDS[backend]["requires_gates"] and mode == "synthetic":
        raise ConfigurationError(
            "backend '%s' executes real models and cannot run in synthetic mode" % backend
        )
    if not BACKENDS[backend]["requires_gates"] and mode != "synthetic":
        raise ConfigurationError("backend 'synthetic' can only run in synthetic mode")

    starts = supplied.get("starts_per_task", 50)
    if isinstance(starts, bool) or not isinstance(starts, int) or starts < 1 or starts > 10_000:
        raise ConfigurationError("starts_per_task must be an integer from 1 through 10000")

    seed = supplied.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("seed must be an integer")

    workers = supplied.get("max_workers", 12)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1 or workers > 32:
        raise ConfigurationError("max_workers must be an integer from 1 through 32")

    cohort = supplied.get("cohort", "primary")
    if cohort not in COHORTS:
        raise ConfigurationError(
            "cohort must be one of %s" % ", ".join("'%s'" % item for item in sorted(COHORTS))
        )

    protocol_hash = supplied.get("protocol_hash")
    if protocol_hash is not None and (
        not isinstance(protocol_hash, str) or not protocol_hash.startswith("sha256:")
    ):
        raise ConfigurationError("protocol_hash must be a 'sha256:...' string when supplied")
    if BACKENDS[backend]["requires_gates"] and cohort == "primary" and not protocol_hash:
        raise ConfigurationError(
            "a primary-cohort real-model run requires a frozen protocol_hash"
        )

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
            # Only a synthetic run carries the fixture label.  Attaching it to a
            # real-model run would mislabel real output as a fixture, and
            # omitting it from a fixture run would hide what it is.
            "fixture_label": SYNTHETIC_LABEL if backend == "synthetic" else None,
            "backend_label": BACKENDS[backend]["label"],
            "episode_mode": RUN_MODES[mode],
            "cohort": cohort,
            "protocol_hash": protocol_hash,
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
    mode = str(episode.get("mode") or "")
    profile = "synthetic" if "synthetic" in mode else "real_model"
    artifact_refs = _validate_attempt_artifact_refs(dict(artifact_refs), episode, artifact_root, profile=profile)
    timing = raw.get("timing", {})
    if not isinstance(timing, Mapping):
        raise ValueError("backend timing must be an object")

    normalised: Dict[str, Any] = {
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
    # Spec section 8 measurement, cost and identity fields are carried through
    # only when the backend actually supplied them, so an absent measurement
    # stays absent rather than being written as a zero.  Numeric fields are
    # validated here because the ledger stores them directly.
    for name in ("executed_actions", "n_segments"):
        if name in raw and raw[name] is not None:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("backend %s must be a non-negative integer or null" % name)
            normalised[name] = value
    for name in ("compute_gpu_seconds", "allocated_gpu_seconds", "estimated_usd"):
        if name in raw and raw[name] is not None:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("backend %s must be a number or null" % name)
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("backend %s must be a finite non-negative number" % name)
            normalised[name] = float(value)
    for name in (
        "raw_judge_samples_ref",
        "segments_manifest_ref",
        "video_ref",
        "timing_ref",
        "cost_basis_ref",
        "exclusion_reason",
        "feedback_mode",
        "parity_status",
        "presentation_track",
    ):
        if name in raw and raw[name] is not None:
            normalised[name] = str(raw[name])
    if "segments" in raw and raw["segments"] is not None:
        values = raw["segments"]
        if not isinstance(values, (list, tuple)):
            raise ValueError("backend segments must be a list")
        normalised["segments"] = [dict(item) for item in values if isinstance(item, Mapping)]
    for name in ("platform_request_ids", "attempt_ids"):
        if name in raw and raw[name] is not None:
            values = raw[name]
            if not isinstance(values, (list, tuple)):
                raise ValueError("backend %s must be a list" % name)
            normalised[name] = [str(item) for item in values]
    for name in ("policy_identity", "world_identity", "judge_identity", "seeds"):
        if name in raw and raw[name] is not None:
            if not isinstance(raw[name], Mapping):
                raise ValueError("backend %s must be an object" % name)
            normalised[name] = dict(raw[name])
    return normalised


#: Media types a real-model attempt may persist, with their required suffix.
#: Everything must still resolve inside the attempt's own directory and match a
#: recorded SHA-256; widening the type list does not widen the containment rule.
REAL_MODEL_MEDIA = {
    "application/json": ".json",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "video/mp4": ".mp4",
}

SYNTHETIC_MEDIA = {"image/svg+xml": ".svg", "application/json": ".json"}


def _validate_synthetic_artifact_refs(
    refs: Dict[str, Any], episode: Mapping[str, Any], artifact_root: Optional[Path]
) -> Dict[str, Dict[str, str]]:
    """Back-compatible entry point for the synthetic fixture profile."""

    return _validate_attempt_artifact_refs(refs, episode, artifact_root, profile="synthetic")


def _validate_attempt_artifact_refs(
    refs: Dict[str, Any],
    episode: Mapping[str, Any],
    artifact_root: Optional[Path],
    profile: str = "synthetic",
) -> Dict[str, Dict[str, str]]:
    """Accept only hash-verified, own-attempt artifacts.

    Two profiles share one containment rule.  ``synthetic`` additionally
    requires the fixture manifest and its four false claims, so a fixture can
    never be dressed up as a model result.  ``real_model`` allows PNG/JPEG/MP4
    media and requires a provenance manifest whose ``qualified_measurement`` is
    false -- a backend records what it produced, and only ``plumb.gates`` may
    later call a cell qualified.

    Neither profile permits an arbitrary URL: a result cannot smuggle an
    external clip into the dashboard.  Empty refs stay valid for narrow
    fake-backend failure/recovery tests.
    """

    if profile not in ("synthetic", "real_model"):
        raise ValueError("unknown artifact profile %r" % profile)
    if not refs:
        return {}
    if artifact_root is None:
        raise ValueError("nonempty artifact_refs require an own-attempt artifact directory")
    root = artifact_root.resolve()
    if not root.is_dir():
        raise ValueError("own-attempt artifact directory does not exist")
    data_root = root.parents[3]
    allowed_media = SYNTHETIC_MEDIA if profile == "synthetic" else REAL_MODEL_MEDIA
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
            raise ValueError("artifact refs require an allowed typed media value and SHA-256")
        candidate = (data_root / relative).resolve()
        if root not in candidate.parents or candidate.suffix.lower() != allowed_media[media_type] or not candidate.is_file():
            raise ValueError("artifact must resolve under this episode attempt directory")
        if file_digest(candidate) != digest:
            raise ValueError("artifact SHA-256 does not match persisted bytes")
        expected_uri = "artifact://%s" % relative
        expected_url = "/api/artifacts/%s" % relative
        if ref.get("uri", expected_uri) != expected_uri or ref.get("url", expected_url) != expected_url:
            raise ValueError("artifact references may not use arbitrary URLs")
        if ref.get("artifact_path", relative) != relative:
            raise ValueError("artifact_path must equal data-root-relative path")
        label = str(ref.get("label", "")).lower()
        if profile == "synthetic":
            if "synthetic" not in label or "unqualified" not in label:
                raise ValueError("synthetic artifacts must retain synthetic/unqualified labels")
        else:
            if not label.strip():
                raise ValueError("real-model artifacts require a descriptive label")
            if "synthetic" in label:
                raise ValueError("a real-model artifact must not be labelled synthetic")
            if "qualified" in label and "unqualified" not in label:
                raise ValueError("a backend artifact may not label itself qualified")
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
    if profile == "synthetic":
        frame = validated.get("fixture_frame")
        if manifest is None or frame is None:
            raise ValueError("nonempty synthetic output requires fixture_frame and manifest artifacts")
        required_false = ("real_robot", "physical_control", "world_model", "qualified_measurement")
        expected_kind = "synthetic_engineering_fixture"
    else:
        if manifest is None:
            raise ValueError("nonempty real-model output requires a provenance manifest artifact")
        # A world model *did* run, so ``world_model`` is legitimately true here.
        # What a backend may never assert is a physical robot or a qualified
        # measurement; those two remain false in every persisted manifest.
        required_false = ("real_robot", "physical_control", "qualified_measurement")
        expected_kind = "plumb_real_model_attempt"
    try:
        with (data_root / manifest["relative_path"]).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError("artifact manifest is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("artifact manifest must be an object")
    claims = payload.get("claims")
    if (
        payload.get("kind") != expected_kind
        or payload.get("run_id") != episode.get("run_id")
        or payload.get("episode_id") != episode.get("episode_id")
        or not isinstance(claims, Mapping)
        or any(claims.get(name) is not False for name in required_false)
    ):
        raise ValueError("artifact manifest is not an own %s manifest with the required false claims" % expected_kind)
    if profile == "synthetic" and "synthetic" not in str(payload.get("label", "")).lower():
        raise ValueError("synthetic artifact manifest must retain its synthetic label")
    return validated
