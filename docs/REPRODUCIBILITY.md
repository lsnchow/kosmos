# Reproducibility

## Local application

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test,analytics]'
npm --prefix web install
npm --prefix web run build
.venv/bin/plumb serve --port 8787
```

The default app binds to localhost and stores mutable state under `data/`, which
is excluded from Git. Copy `.env.example` only when configuring an external
backend; never commit credentials, model weights, runtime caches, or run data.

## Verification

```bash
.venv/bin/python -m pytest -q
npm --prefix web run test
```

The test suite covers the ledger, contracts, artifact boundaries, API paths, and
frontend behavior. GPU diagnostics are cluster jobs rather than laptop tests.

## Cluster diagnostics

Cluster scripts use immutable source bundles made with:

```bash
python3 cluster/make_release.py
```

Upload the resulting archive to a new SHA-named release directory and submit the
appropriate `cluster/*.sbatch` script from an allocated GPU node. Keep weights,
credentials, runtime caches, and raw run evidence outside the Git worktree.
Every diagnostic should persist model identity, source-release hash, media hashes,
and measured timings before it is discussed as evidence.
