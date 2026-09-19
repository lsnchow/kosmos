"""Offline safety tests for the dedicated Octo 241fb runtime-lock writer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cluster import lock_octo_production_runtime as lock


PACKAGES = {
    "tensorflow": "2.15.1+computecanada",
    "numpy": "1.26.4+computecanada",
}


def test_pip_check_keeps_exact_output_and_accepts_only_reviewed_tensorflow_and_numpy_lines(monkeypatch):
    stdout = (
        "dlimp 0.1.0 has requirement tensorflow==2.15.0, but you have tensorflow 2.15.1+computecanada.\n"
        "dlimp 0.1.0 has requirement numpy==1.24.3, but you have numpy 1.26.4+computecanada.\n"
    )
    monkeypatch.setattr(
        lock.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=stdout, stderr=""),
    )

    report = lock._checked_pip_check(PACKAGES)

    assert report["returncode"] == 1
    assert report["stdout"] == stdout
    assert report["stderr"] == ""
    assert [item["package"] for item in report["accepted_deviations"]] == ["tensorflow", "numpy"]


def test_pip_check_blocks_an_unknown_broken_dependency(monkeypatch):
    stdout = "other-package 1.0 has requirement unsafe-lib==9, but you have unsafe-lib 8.\n"
    monkeypatch.setattr(
        lock.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=stdout, stderr=""),
    )

    with pytest.raises(RuntimeError, match="unreviewed dependency break"):
        lock._checked_pip_check(PACKAGES)


def test_pip_check_blocks_a_known_package_with_an_unreviewed_version(monkeypatch):
    stdout = "dlimp 0.1.0 has requirement tensorflow==2.14.0, but you have tensorflow 2.15.1+computecanada.\n"
    monkeypatch.setattr(
        lock.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=stdout, stderr=""),
    )

    with pytest.raises(RuntimeError, match="unreviewed dependency break"):
        lock._checked_pip_check(PACKAGES)


def test_freeze_path_is_created_exclusively_and_must_match_live_inventory(tmp_path):
    root = tmp_path / "scratch" / "plumb"
    venv = root / "venv-octo-autoeval241fb"
    venv.mkdir(parents=True)
    freeze = venv / "requirements-freeze-v1.txt"
    lines = ["dlimp @ git+https://github.com/kvablack/dlimp@" + lock.DLIMP_SOURCE_COMMIT, "numpy==1.26.4+computecanada"]

    result = lock._checked_freeze(root, freeze, lines)

    assert result["lines"] == lines
    assert freeze.read_text(encoding="utf-8").splitlines() == lines
    assert lock._checked_freeze(root, freeze, lines)["sha256"] == result["sha256"]
    with pytest.raises(RuntimeError, match="differs"):
        lock._checked_freeze(root, freeze, lines + ["tensorflow==2.15.1+computecanada"])
