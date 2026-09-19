"""Resolve a planned episode's start into the payload the Chain requires.

``deploy/baseten/chain.py``'s ``EpisodeControlPayload`` requires the things a
rollout genuinely cannot begin without: the exact task prompt, the conditioning
frame's pixels, and the 8-D Bridge state.  Those come from the ScenarioManifest
(spec section 3), which is a Gate C dependency.

So there are exactly two honest outcomes when a submission is built:

* a manifest start resolves, and the episode carries real provenance; or
* it does not, and the submission is refused with the missing start named.

There is deliberately no third branch that invents a frame.  A rehearsal start
exists for exercising the transport, and it is a *separate, explicitly labelled*
resolver whose output records ``provenance="rehearsal_synthetic"`` — the Chain
and the ledger can both see what they were given.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from .gates import EXACT_TASK_PROMPTS
from .protocol import TASK_HORIZONS

#: The Bridge state convention (spec section 2): (x, y, z, rx, ry, rz, 0, gripper).
#: Channel 6 is structurally zero and the gripper channel runs 0..0.39 where 0
#: means fully closed.
BRIDGE_STATE_WIDTH = 8
BRIDGE_GRIPPER_STATE_MAX = 0.39


class StartResolutionError(RuntimeError):
    """A start could not be resolved, so no submission may be built."""


@dataclass(frozen=True)
class ResolvedStart:
    """Everything a Chain submission needs about one starting state."""

    start_id: str
    start_lineage_id: str
    task: str
    prompt: str
    horizon_actions: int
    frame_png_base64: str
    frame_png_sha256: str
    frame_height: int
    frame_width: int
    state: Tuple[float, ...]
    provenance: str
    goal_png_base64: Optional[str] = None
    goal_png_sha256: Optional[str] = None
    source_uri: Optional[str] = None

    def __post_init__(self) -> None:
        if self.prompt != EXACT_TASK_PROMPTS.get(self.task):
            raise StartResolutionError(
                "start %s carries prompt %r, which is not the verbatim prompt for %s"
                % (self.start_id, self.prompt, self.task)
            )
        if len(self.state) != BRIDGE_STATE_WIDTH:
            raise StartResolutionError(
                "start %s has a %d-wide state; Bridge state is %d-D"
                % (self.start_id, len(self.state), BRIDGE_STATE_WIDTH)
            )
        if self.state[6] != 0.0:
            raise StartResolutionError(
                "start %s sets the structural channel 6 to %r; it must be zero"
                % (self.start_id, self.state[6])
            )
        if not 0.0 <= self.state[7] <= BRIDGE_GRIPPER_STATE_MAX:
            raise StartResolutionError(
                "start %s gripper state %r is outside the observed 0..%s range"
                % (self.start_id, self.state[7], BRIDGE_GRIPPER_STATE_MAX)
            )
        if not self.frame_png_base64:
            raise StartResolutionError("start %s has no conditioning frame" % self.start_id)

    def frame_payload(self) -> Dict[str, Any]:
        return {
            "encoding": "png_base64",
            "data": self.frame_png_base64,
            "png_sha256": self.frame_png_sha256,
            "nominal_timestamp": 0.0,
            "height": self.frame_height,
            "width": self.frame_width,
        }

    def goal_payload(self) -> Optional[Dict[str, Any]]:
        if not self.goal_png_base64:
            return None
        return {
            "encoding": "png_base64",
            "data": self.goal_png_base64,
            "png_sha256": self.goal_png_sha256,
            "height": self.frame_height,
            "width": self.frame_width,
        }


class StartResolver(Protocol):
    """Resolves a planned episode's ``start_id`` into a ``ResolvedStart``."""

    provenance: str

    def resolve(self, task: str, start_id: str, start_lineage_id: str) -> ResolvedStart:
        """Return the start, or raise ``StartResolutionError``."""


class ScenarioStartResolver:
    """Reads real starts from a ``scenarios.jsonl`` ScenarioManifest."""

    provenance = "scenario_manifest"

    def __init__(self, manifest_path: Path, artifact_root: Optional[Path] = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else self.manifest_path.parent
        if not self.manifest_path.is_file():
            raise StartResolutionError("no scenario manifest at %s" % self.manifest_path)
        self._by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise StartResolutionError(
                        "line %d of %s is not JSON" % (number, self.manifest_path)
                    ) from exc
                if not isinstance(row, Mapping):
                    continue
                task = str(row.get("task") or "")
                start_id = str(row.get("start_id") or "")
                if task and start_id:
                    self._by_key[(task, start_id)] = row

    def resolve(self, task: str, start_id: str, start_lineage_id: str) -> ResolvedStart:
        row = self._by_key.get((task, start_id))
        if row is None:
            raise StartResolutionError(
                "scenario manifest has no start %s for task %s; Gate C must supply it"
                % (start_id, task)
            )
        image_ref = row.get("image_uri") or row.get("image_path") or row.get("frame_uri")
        if not image_ref:
            raise StartResolutionError("start %s records no image reference" % start_id)
        path = (self.artifact_root / str(image_ref)).resolve()
        if not path.is_file():
            raise StartResolutionError("start %s image is not present at %s" % (start_id, path))
        raw = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        recorded = row.get("image_sha256")
        if recorded and str(recorded) not in (digest, digest.split(":", 1)[-1]):
            raise StartResolutionError(
                "start %s image hash does not match the manifest; refusing to use it" % start_id
            )
        state = row.get("state") or row.get("proprio")
        if not isinstance(state, Sequence):
            raise StartResolutionError("start %s records no 8-D Bridge state" % start_id)
        height, width = _png_dimensions(raw)
        goal_ref = row.get("goal_image_uri")
        goal_b64 = goal_sha = None
        if goal_ref:
            goal_path = (self.artifact_root / str(goal_ref)).resolve()
            if not goal_path.is_file():
                raise StartResolutionError("start %s goal image is missing at %s" % (start_id, goal_path))
            goal_raw = goal_path.read_bytes()
            goal_b64 = base64.b64encode(goal_raw).decode("ascii")
            goal_sha = "sha256:" + hashlib.sha256(goal_raw).hexdigest()
        return ResolvedStart(
            start_id=start_id,
            start_lineage_id=start_lineage_id,
            task=task,
            prompt=EXACT_TASK_PROMPTS[task],
            horizon_actions=TASK_HORIZONS[task],
            frame_png_base64=base64.b64encode(raw).decode("ascii"),
            frame_png_sha256=digest,
            frame_height=height,
            frame_width=width,
            state=tuple(float(value) for value in state),
            provenance=self.provenance,
            goal_png_base64=goal_b64,
            goal_png_sha256=goal_sha,
            source_uri=path.as_posix(),
        )


class RehearsalStartResolver:
    """Deterministic, explicitly labelled synthetic starts for rehearsal only.

    This exists so the transport, ledger, reconciler and dashboard can be
    exercised before Gate C delivers real panels.  Its output is labelled
    ``rehearsal_synthetic`` everywhere it travels, and it is a separate class
    from the real resolver so a production path cannot fall back to it by
    accident.
    """

    provenance = "rehearsal_synthetic"

    def __init__(self, resolution: int = 256) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.resolution = int(resolution)

    def resolve(self, task: str, start_id: str, start_lineage_id: str) -> ResolvedStart:
        if task not in EXACT_TASK_PROMPTS:
            raise StartResolutionError("unknown task %r" % task)
        seed = hashlib.sha256(("%s\x1f%s" % (task, start_id)).encode("utf-8")).digest()
        raw = _solid_png(self.resolution, self.resolution, seed[0], seed[1], seed[2])
        # A plausible Bridge pose: the exact values do not matter because nothing
        # scientific is derived from a rehearsal, but they must satisfy the
        # convention so the contract checks are genuinely exercised.
        state = (
            0.20 + (seed[3] / 255.0) * 0.10,
            -0.05 + (seed[4] / 255.0) * 0.10,
            0.06 + (seed[5] / 255.0) * 0.05,
            (seed[6] / 255.0) * 0.05,
            (seed[7] / 255.0) * 0.05,
            (seed[8] / 255.0) * 0.05,
            0.0,
            (seed[9] / 255.0) * BRIDGE_GRIPPER_STATE_MAX,
        )
        return ResolvedStart(
            start_id=start_id,
            start_lineage_id=start_lineage_id,
            task=task,
            prompt=EXACT_TASK_PROMPTS[task],
            horizon_actions=TASK_HORIZONS[task],
            frame_png_base64=base64.b64encode(raw).decode("ascii"),
            frame_png_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
            frame_height=self.resolution,
            frame_width=self.resolution,
            state=state,
            provenance=self.provenance,
            source_uri="plumb://rehearsal-synthetic-start/%s/%s" % (task, start_id),
        )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _solid_png(height: int, width: int, red: int, green: int, blue: int) -> bytes:
    """Build a minimal valid RGB PNG without requiring Pillow."""

    header = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    row = bytes([red, green, blue]) * width
    scanlines = b"".join(b"\x00" + row for _ in range(height))
    body = _png_chunk(b"IDAT", zlib.compress(scanlines, 6))
    return b"\x89PNG\r\n\x1a\n" + header + body + _png_chunk(b"IEND", b"")


def _png_dimensions(raw: bytes) -> Tuple[int, int]:
    """Read height and width from a PNG's IHDR without a decoder."""

    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise StartResolutionError("conditioning frame is not a PNG")
    width, height = struct.unpack(">II", raw[16:24])
    return int(height), int(width)


__all__ = [
    "BRIDGE_GRIPPER_STATE_MAX",
    "BRIDGE_STATE_WIDTH",
    "RehearsalStartResolver",
    "ResolvedStart",
    "ScenarioStartResolver",
    "StartResolutionError",
    "StartResolver",
]
