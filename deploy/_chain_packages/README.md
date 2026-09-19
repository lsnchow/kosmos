# Chain external-package staging

This directory must **exist** at `truss chains push` time and is otherwise empty
in version control. `truss_chains.deployment.chain_gatherer.gather_chain` raises
`ValueError: External packages directory ... is not a directory` if an
`external_package_dirs` entry is missing, so the directory (not its contents) is
tracked via `.gitignore`.

Fill it with:

```bash
python deploy/baseten/stage_packages.py
```

That copies `plumb/**` (source files only — no weights, data, caches, or
credentials) here. `deploy/baseten/chain.py` declares
`external_package_dirs=[make_abs_path_here("../_chain_packages")]`, and truss
copies the *contents* of this directory into each image's `packages/` dir, which
makes `import plumb` work inside every Chainlet.

Why not point `external_package_dirs` at the repository root: `gather_chain`
applies only truss's built-in `.truss_ignore`, which excludes `.venv` and `.git`
but not `web/node_modules` (~117 MB). Staging keeps nine Chainlet images small.

If this directory is empty at push time the push still succeeds, and every
Chainlet returns `status="blocked"` with an explicit `plumb ... is not importable
inside this image` reason. That is the intended fail-closed behaviour — it is
never a silent fallback to synthetic output.
