"""Explicitly synthetic fixture backend for exercising the PLUMB control plane."""

from __future__ import annotations

import hashlib
import html
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from .records import SYNTHETIC_LABEL, file_digest, horizon_for, utc_now


class SyntheticExecutionError(RuntimeError):
    """An intentional synthetic service error used to test failure accounting."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _relative_ref(artifact_root: Path, path: Path, media_type: str) -> Dict[str, str]:
    # ``artifact_root`` is ``<data>/artifacts/<run>/<episode>/attempt-XXXX``.
    # Preserve a data-root-relative path so the application's artifact route
    # can resolve every immutable attempt artifact uniformly.
    relative = path.relative_to(artifact_root.parents[3]).as_posix()
    return {
        "uri": "artifact://%s" % relative,
        "relative_path": relative,
        "artifact_path": relative,
        "url": "/api/artifacts/%s" % relative,
        "sha256": file_digest(path),
        "media_type": media_type,
        "label": SYNTHETIC_LABEL,
    }


class SyntheticBackend:
    """Deterministic test fixture generator; it never claims real-world fidelity."""

    label = SYNTHETIC_LABEL

    def _matches(self, configured: Any, episode: Mapping[str, Any]) -> bool:
        if not isinstance(configured, (list, tuple, set)):
            return False
        identifiers = {str(episode["episode_id"]), str(episode["start_id"]), str(episode["logical_key"])}
        return any(str(item) in identifiers for item in configured)

    def execute(self, episode: Mapping[str, Any], config: Mapping[str, Any], artifact_dir: Path) -> Dict[str, Any]:
        """Write a labelled SVG and manifest, then return its score-shaped fixture."""

        started = time.perf_counter()
        if self._matches(config.get("synthetic_failure_start_ids"), episode):
            raise SyntheticExecutionError("intentional synthetic fixture service failure")

        fixture_seed = int(episode["world_seed"])
        digest = hashlib.sha256(str(fixture_seed).encode("ascii")).digest()
        success = bool(digest[0] & 1)
        missing = self._matches(config.get("synthetic_missing_start_ids"), episode)
        progress = 5 if success else int(digest[1] % 5)
        horizontal = 35 + int(digest[2] % 180)
        vertical = 100 + int(digest[3] % 80)
        colour = "#36d399" if success else "#f87171"
        svg = """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"320\" height=\"180\" viewBox=\"0 0 320 180\" role=\"img\" aria-label=\"Synthetic engineering test fixture\">
  <rect width=\"320\" height=\"180\" fill=\"#102033\"/>
  <text x=\"12\" y=\"20\" font-family=\"monospace\" font-size=\"10\" fill=\"#fbbf24\">SYNTHETIC — UNQUALIFIED ENGINEERING TEST ARTIFACT</text>
  <text x=\"12\" y=\"35\" font-family=\"monospace\" font-size=\"9\" fill=\"#cbd5e1\">NOT REAL ROBOT / NOT PHYSICS SIMULATION</text>
  <rect x=\"24\" y=\"145\" width=\"272\" height=\"12\" rx=\"3\" fill=\"#334155\"/>
  <line x1=\"80\" y1=\"150\" x2=\"%d\" y2=\"%d\" stroke=\"#93c5fd\" stroke-width=\"10\" stroke-linecap=\"round\"/>
  <circle cx=\"%d\" cy=\"%d\" r=\"12\" fill=\"%s\"/>
  <text x=\"12\" y=\"174\" font-family=\"monospace\" font-size=\"9\" fill=\"#cbd5e1\">%s · %s · %s</text>
</svg>\n""" % (
            horizontal,
            vertical,
            horizontal,
            vertical,
            colour,
            html.escape(str(episode["policy"])),
            html.escape(str(episode["task"])),
            html.escape(str(episode["start_id"])),
        )
        frame_path = artifact_dir / "synthetic_fixture.svg"
        _atomic_text(frame_path, svg)
        manifest = {
            "schema_version": 1,
            "kind": "synthetic_engineering_fixture",
            "label": SYNTHETIC_LABEL,
            "created_at": utc_now(),
            "run_id": episode["run_id"],
            "episode_id": episode["episode_id"],
            "policy": episode["policy"],
            "task": episode["task"],
            "world_seed": fixture_seed,
            "claims": {
                "real_robot": False,
                "physical_control": False,
                "world_model": False,
                "qualified_measurement": False,
            },
            "frame": "synthetic_fixture.svg",
        }
        manifest_path = artifact_dir / "synthetic_manifest.json"
        _atomic_json(manifest_path, manifest)

        refs = {
            "fixture_frame": _relative_ref(artifact_dir, frame_path, "image/svg+xml"),
            "manifest": _relative_ref(artifact_dir, manifest_path, "application/json"),
        }
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if missing:
            return {
                "validity": "unknown",
                "binary_success": None,
                "progress_score": None,
                "missing_reason": "synthetic_fixture_missing_label",
                "horizon_actions": horizon_for(str(episode["task"])),
                "artifact_refs": refs,
                "timing": {"fixture_generation_ms": elapsed_ms, "source": "synthetic"},
                "backend_metadata": {"label": SYNTHETIC_LABEL, "fixture_seed": fixture_seed},
            }
        return {
            "validity": "valid",
            "binary_success": success,
            "progress_score": progress,
            "missing_reason": None,
            "horizon_actions": horizon_for(str(episode["task"])),
            "artifact_refs": refs,
            "timing": {"fixture_generation_ms": elapsed_ms, "source": "synthetic"},
            "backend_metadata": {"label": SYNTHETIC_LABEL, "fixture_seed": fixture_seed},
        }
