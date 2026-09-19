"""Execution backends for the PLUMB engine.

A backend implements ``execute(episode, config, artifact_dir) -> mapping`` and is
selected by name from ``plumb.records.BACKENDS``.  ``plumb.engine.RunService``
refuses to run a gated backend until its declared gates carry passing evidence.
"""

from __future__ import annotations

__all__ = ["baseten"]
