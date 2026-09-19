"""Compatibility access to the source-specific OpenPiZero diagnostic.

Production controller code must use :mod:`plumb.policies.openpizero`.
The source-bound AutoEval reproduction is unqualified and lives explicitly in
:mod:`plumb.policies.diagnostics.openpi`; this module retains the initial
continuation import path without creating a second implementation.
"""

from .diagnostics import openpi as _diagnostic

__all__ = tuple(name for name in dir(_diagnostic) if not name.startswith("_"))


def __getattr__(name: str):
    return getattr(_diagnostic, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
