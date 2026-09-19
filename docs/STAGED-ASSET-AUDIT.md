# Staged asset audit

`assets.lock.json` is deliberately conservative: it records an asset as
`verified` only after the complete asset/code/runtime/access record is known.
That is different from having already downloaded pinned bytes on the cluster.

`cluster.audit_staged_assets` supplies a read-only bridge between those facts.
It audits pre-existing cluster staging manifests and produces a new immutable
report. It never downloads, contacts a remote service, runs inference, edits
`assets.lock.json`, or changes `protocol.json`.

Run it on the cluster login node, using the existing manifests and a new report
name directly under the cluster evidence directory:

```bash
python -m cluster.audit_staged_assets \
  --cluster-root /scratch/lchow432/plumb \
  --manifest /scratch/lchow432/plumb/evidence/openvla--openvla-7b-download.json \
  --manifest /scratch/lchow432/plumb/evidence/nvidia--Cosmos3-Nano-download.json \
  --manifest /scratch/lchow432/plumb/models/rail-berkeley--octo-small/PLUMB-ASSET-MANIFEST.json \
  --output /scratch/lchow432/plumb/evidence/staged-asset-audit-2026-09-19.json
```

The output path is mandatory, must end in `.json`, must be directly inside
`<cluster-root>/evidence`, and is created with no-clobber semantics. Re-running
with the same path fails rather than replacing the prior report.

## What it verifies

The audit understands the two manifest shapes emitted by the repository:

- `cluster.download_assets` manifests bind a repository, immutable revision,
  destination, and individual `file_records` containing byte lengths and local
  SHA-256 values. When a publisher digest is recorded, the report checks that
  it agrees as well.
- `plumb-octo-assets-v1` binds the Octo model/checkpoint and code revisions plus
  per-file byte lengths and SHA-256 values. Its T5 tokenizer support is reported
  separately.

`staged_integrity: pass` means that all of the audited, regular staged files
match that manifest's local byte/hash/revision bindings. It does not mean that
every file has a publisher-provided digest: the report separately counts
`publisher_digest_records` and `local_manifest_only_records`.

The audit refuses to follow symlinks, relative-path escapes, malformed digest
fields, revision mismatches, size mismatches, and digest mismatches. A symlink
in a model's core files fails that model's staged-integrity result. Hugging Face
T5 cache files often point to blob storage with symlinks; those are reported as
a separately blocked T5 support component, without treating the independently
regular Octo checkpoint as verified support.

## What it does not verify

Every report intentionally says `fully_verified: false`. A staging manifest
does not establish any of the following:

- runtime container digest;
- compatibility profile or validated loader stack;
- license/access and redistribution review;
- GPU execution, Gate A, Gate B, or a qualified study cell.

The report includes `partial_asset_lock_proposal` objects for human review. They
are AssetLock-shaped but keep aggregate `sha256`, `container_digest`, and
`compatibility_profile_id` null and `verified: false`. They are not read by the
application and must not be copied into the primary lock automatically.

## Actual execution

Trillium report `evidence/staged-asset-audit-20260919-v1.json` re-hashed the
staged OpenVLA, Cosmos3-Nano, Octo-Small and Bridge metadata packages. All four
core staged-integrity checks passed; **zero** fully verified deployment profiles
were claimed. A small report copy is in local `data/staged-asset-audits/`.
The report binds producer SHA-256
`1f70cbea85204286d327bd652e342e8b328600fb597a4ccbdabe11610d430af0`.
No weights were downloaded or copied to the laptop for this audit.
