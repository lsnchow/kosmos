"""Stage the ``plumb`` package for a Chains push.  Read-only against the repo.

Why this exists: ``truss chains push`` treats the directory holding the
entrypoint file as the chain workspace, so ``deploy/baseten/`` is copied into
every Chainlet image, but ``plumb/`` sits outside it.  The SDK's answer is
``DockerImage(external_package_dirs=[...])``, and ``truss_chains`` copies the
*contents* of each external directory into the image's ``packages/`` dir.

Pointing that at the repository root would work and is tempting, but
``gather_chain`` honours only truss's own built-in ``.truss_ignore`` — which
excludes ``.venv`` and ``.git`` but not ``web/node_modules`` — so a push would
upload roughly 117 MB of JavaScript to every one of nine Chainlet images.  This
script therefore copies just the Python package into ``deploy/_chain_packages/``,
which is what ``chain.py`` declares as its external package dir.

Usage (from the repository root, before ``truss chains push``):

    python deploy/baseten/stage_packages.py
    python deploy/baseten/stage_packages.py --check   # verify without writing

It copies source files only: no weights, no data, no caches, no credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = REPO_ROOT / "deploy" / "_chain_packages"
STAGED_PACKAGES: Tuple[str, ...] = ("plumb", "cluster")

# Everything the images must never receive.  Model weights and datasets reach a
# Chainlet through ``chains.Assets(cached=[...])`` at an immutable revision, never
# by being baked into an image from a developer's working tree.
EXCLUDED_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "data",
        "models",
        "fixtures",
        "artifacts",
    }
)
EXCLUDED_SUFFIXES = frozenset(
    {".pyc", ".pyo", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".npz", ".tar", ".gz", ".mp4", ".png", ".jpg"}
)


def _iter_source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        yield path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(check_only: bool) -> int:
    missing: List[str] = [name for name in STAGED_PACKAGES if not (REPO_ROOT / name).is_dir()]
    if missing:
        print("ERROR: package(s) not found under %s: %s" % (REPO_ROOT, ", ".join(missing)), file=sys.stderr)
        return 2

    problems: List[str] = []
    copied = 0
    total_bytes = 0
    for name in STAGED_PACKAGES:
        source_root = REPO_ROOT / name
        target_root = STAGING_DIR / name
        if not check_only and target_root.exists():
            shutil.rmtree(target_root)
        for source in _iter_source_files(source_root):
            relative = source.relative_to(source_root)
            target = target_root / relative
            if check_only:
                if not target.is_file():
                    problems.append("missing: %s" % (target.relative_to(REPO_ROOT)))
                elif _digest(target) != _digest(source):
                    problems.append("stale: %s" % (target.relative_to(REPO_ROOT)))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            copied += 1
            total_bytes += source.stat().st_size

    if check_only:
        if problems:
            print("Staging is out of date (%d problem(s)):" % len(problems), file=sys.stderr)
            for problem in problems[:20]:
                print("  %s" % problem, file=sys.stderr)
            print("Run: python deploy/baseten/stage_packages.py", file=sys.stderr)
            return 1
        print("Staging is current: %d file(s), %.1f KiB under %s" % (copied, total_bytes / 1024.0, STAGING_DIR))
        return 0

    print("Staged %d file(s), %.1f KiB into %s" % (copied, total_bytes / 1024.0, STAGING_DIR))
    print("chain.py declares this directory as external_package_dirs; run the push from the repository root.")
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the staged copy matches the working tree without writing anything",
    )
    arguments = parser.parse_args(argv)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    return stage(arguments.check)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
