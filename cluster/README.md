# Cluster diagnostics

`cluster/` contains GPU-only diagnostics for policy adapters, world models, and
the VLM judge. These scripts run on allocated compute nodes and write immutable
reports; they do not run inference on login nodes.

Create a source bundle before submitting a job:

```bash
python3 cluster/make_release.py
```

The bundle contains source and a SHA-addressed `RELEASE.json`, never model
weights or credentials. Upload it to a new release directory on the cluster,
then pass that directory to the selected `.sbatch` script.

Common diagnostics include Cosmos forward dynamics, OpenVLA action probes,
IRASim image-conditioned replay, adapter checks, and paired judge audits. Their
reports must retain source/media hashes, model revision, allocation identity, and
measured timings. A successful GPU call does not by itself establish policy
quality or benchmark validity.
