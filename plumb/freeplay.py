"""Source-bound free-play capability contracts.

A gallery MP4 is evidence of a past generation, not a restorable world state.
This module keeps that distinction at the API boundary: source-bound controls
are available only when a hash-bound checkpoint manifest *and* an explicitly
configured adapter both say that the exact source branch can be restored.

The protocol below is deliberately small.  Implementations must make
``source_branch_status`` a local capability check; the gallery's GET endpoint
must never wake a model, submit a job, or probe a remote deployment.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, Sequence


SOURCE_BRANCH_SCHEMA = "plumb-source-branch-v1"


class SourceBranchAdapter(Protocol):
    """Adapter required to continue one recorded world-state branch exactly.

    ``source`` is a server-derived, hash-bound identity.  It is never composed
    from a client supplied path, timestamp, or arbitrary recording URL.  A
    positive status must echo the manifest's ``adapter`` and checkpoint SHA-256
    so the capability is attested to this exact restorable branch.
    """

    def source_branch_status(self, source: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a local, side-effect-free exact-branch capability report."""

    def source_branch_step(
        self,
        *,
        session_id: str,
        source: Mapping[str, Any],
        actions: Sequence[Sequence[float]],
        protocol_hash: str,
        seed: int,
        resolution: int,
    ) -> Mapping[str, Any]:
        """Generate from the manifest-bound state, not a generic task start."""


def source_identity(video: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only the public, server-derived identity a branch adapter needs."""

    branch = video.get("source_branch")
    return {
        "video_id": str(video["id"]),
        "sha256": str(video["sha256"]),
        "title": str(video.get("title") or ""),
        "model": str(video.get("model") or ""),
        "kind": str(video.get("kind") or ""),
        "video_url": str(video.get("video_url") or ""),
        "report_url": str(video.get("report_url") or ""),
        # This is emitted only by world_videos after it has verified the
        # manifest's source hash and local checkpoint artifact.
        "source_branch": dict(branch) if isinstance(branch, Mapping) else None,
    }


def task_start_status(
    backend: Any,
    missing: Sequence[str],
    protocol_hash: Optional[str],
    protocol_reason: Optional[str],
) -> Dict[str, Any]:
    """Describe legacy free-play without calling it a continuation."""

    if backend is None or not callable(getattr(backend, "freeplay_step", None)):
        return {
            "available": False,
            "mode": "task_start",
            "reason": (
                "Task-start free-play is unavailable because no certified world-model free-play "
                "backend is configured. It is not a continuation of a saved recording."
            ),
            "source": None,
            "binding": _binding(False, None, None, requested=False),
            "missing": list(missing) or ["a world-model free-play path on the configured backend"],
            "scored": False,
            "qualified": False,
        }
    if not protocol_hash:
        return {
            "available": False,
            "mode": "task_start",
            "reason": (
                "Task-start free-play needs a frozen protocol before it can generate frames. "
                "It is not a continuation of a saved recording."
            ),
            "source": None,
            "binding": _binding(False, None, None, requested=False),
            "missing": [protocol_reason or "no frozen protocol.json is loaded"],
            "scored": False,
            "qualified": False,
        }
    return {
        "available": True,
        "mode": "task_start",
        "reason": (
            "An unscored task-start free-play path is configured. Each request resolves a task start; "
            "it does not restore or continue a saved recording."
        ),
        "source": None,
        "binding": _binding(False, None, None, requested=False),
        "missing": [],
        "scored": False,
        "qualified": False,
    }


def recording_branch_status(backend: Any, source: Mapping[str, Any]) -> Dict[str, Any]:
    """Return an honest source-branch capability report without backend work."""

    identity = source_identity(source)
    branch = identity.get("source_branch")
    if not isinstance(branch, Mapping):
        status = getattr(backend, "source_branch_status", None)
        step = getattr(backend, "source_branch_step", None)
        backend_missing = not callable(status) or not callable(step)
        backend_reason = None
        if not backend_missing:
            # The adapter contract requires this to be local and side-effect
            # free. It distinguishes an installed source-branch interface from
            # a policy-only/generic backend that merely has the method shape.
            try:
                declared = status(identity)
            except Exception as exc:
                backend_missing = True
                backend_reason = "The source-branch backend could not confirm restore capability: %s" % str(exc)
            else:
                supported = (
                    isinstance(declared, Mapping)
                    and declared.get("available") is True
                    and declared.get("exact_branch_supported") is True
                )
                if not supported:
                    backend_missing = True
                    if isinstance(declared, Mapping) and isinstance(declared.get("reason"), str):
                        backend_reason = declared["reason"]
        backend_note = (
            " %s Generic task-start free-play is not a continuation."
            % (backend_reason or "No source-branch world-model backend is configured either.")
            if backend_missing
            else " The configured backend is not allowed to infer a checkpoint from MP4 pixels; generic task-start free-play is not a continuation."
        )
        return _recording_unavailable(
            identity,
            "This saved MP4 has no hash-bound resumable world-state checkpoint manifest, so it cannot be continued exactly."
            + backend_note,
            missing=(
                ("a hash-bound resumable world-state checkpoint manifest", "a source-branch world-model backend")
                if backend_missing
                else ("a hash-bound resumable world-state checkpoint manifest",)
            ),
        )
    adapter_name = branch.get("adapter") if isinstance(branch.get("adapter"), str) else None
    checkpoint = branch.get("checkpoint") if isinstance(branch.get("checkpoint"), Mapping) else None
    if not adapter_name or checkpoint is None:
        return _recording_unavailable(
            identity,
            "This saved recording has an incomplete source-branch manifest; a named adapter and verified checkpoint are required.",
            checkpoint=checkpoint,
            adapter=adapter_name,
        )
    status = getattr(backend, "source_branch_status", None)
    step = getattr(backend, "source_branch_step", None)
    if not callable(status) or not callable(step):
        return _recording_unavailable(
            identity,
            "No source-branch adapter is configured for this verified checkpoint. Generic task-start free-play is not a continuation.",
            checkpoint=checkpoint,
            adapter=adapter_name,
        )
    try:
        declared = status(identity)
    except Exception as exc:
        return _recording_unavailable(
            identity,
            "The source-branch adapter could not confirm local restore capability: %s" % str(exc),
            checkpoint=checkpoint,
            adapter=adapter_name,
        )
    if not isinstance(declared, Mapping):
        return _recording_unavailable(
            identity,
            "The source-branch adapter returned an invalid capability report; refusing to treat the recording as resumable.",
            checkpoint=checkpoint,
            adapter=adapter_name,
        )
    exact = declared.get("exact_branch_supported") is True
    attested = (
        declared.get("adapter") == adapter_name
        and declared.get("checkpoint_sha256") == checkpoint.get("sha256")
    )
    available = declared.get("available") is True and exact and attested
    reason = declared.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = (
            "The configured source-branch adapter can restore this hash-bound checkpoint exactly."
            if available
            else "The source-branch adapter did not attest to this exact adapter and checkpoint binding."
        )
    return {
        "available": available,
        "mode": "recording_branch",
        "reason": reason,
        "source": identity,
        "binding": _binding(exact and available, checkpoint, adapter_name),
        "missing": [] if available else list(_strings(declared.get("missing"))),
        "scored": False,
        "qualified": False,
    }


def source_branch_step(
    backend: Any,
    *,
    session_id: str,
    source: Mapping[str, Any],
    actions: Sequence[Sequence[float]],
    protocol_hash: str,
    seed: int,
    resolution: int,
) -> Mapping[str, Any]:
    """Call only the explicit source-bound adapter after capability validation."""

    step = getattr(backend, "source_branch_step", None)
    if not callable(step):
        raise RuntimeError("No source-branch adapter is configured")
    return step(
        session_id=session_id,
        source=source_identity(source),
        actions=actions,
        protocol_hash=protocol_hash,
        seed=seed,
        resolution=resolution,
    )


def _binding(
    exact: bool,
    checkpoint: Optional[Mapping[str, Any]],
    adapter: Optional[str],
    *,
    requested: bool = True,
) -> Dict[str, Any]:
    return {
        "requested": requested,
        "exact_branch_supported": exact,
        "checkpoint": dict(checkpoint) if isinstance(checkpoint, Mapping) else None,
        "adapter": adapter,
    }


def _recording_unavailable(
    source: Mapping[str, Any],
    reason: str,
    *,
    checkpoint: Optional[Mapping[str, Any]] = None,
    adapter: Optional[str] = None,
    missing: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "available": False,
        "mode": "recording_branch",
        "reason": reason,
        "source": dict(source),
        "binding": _binding(False, checkpoint, adapter),
        "missing": list(missing),
        "scored": False,
        "qualified": False,
    }


def _strings(value: Any) -> Sequence[str]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


__all__ = [
    "SOURCE_BRANCH_SCHEMA",
    "SourceBranchAdapter",
    "recording_branch_status",
    "source_branch_step",
    "source_identity",
    "task_start_status",
]
