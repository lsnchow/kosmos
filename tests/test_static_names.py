"""Catch unexecuted-path NameErrors without importing GPU runtimes locally."""
import io
from pathlib import Path

from pyflakes.api import checkPath
from pyflakes.reporter import Reporter


def test_production_sources_have_no_undefined_names_or_syntax_errors():
    root = Path(__file__).resolve().parents[1]
    warnings, errors = io.StringIO(), io.StringIO()
    reporter = Reporter(warnings, errors)
    for directory in ("plumb", "cluster", "deploy"):
        for source in sorted((root / directory).rglob("*.py")):
            checkPath(str(source), reporter)
    critical = [line for line in warnings.getvalue().splitlines()
                if "undefined name" in line or "may be undefined" in line]
    assert not errors.getvalue(), errors.getvalue()
    assert not critical, "\n".join(critical)
