# Policy checkpoint conversion — MiniVLA and OpenPiZero

MiniVLA and OpenPiZero are the only two policies in the matrix whose released
weights are legacy PyTorch pickles. Their adapters refuse those pickles outright,
so both policies stay blocked until a reviewed `.safetensors` artifact exists
alongside **three** hashes:

| Hash | Why the adapter insists on it |
|---|---|
| the converted artifact's SHA-256 | the bytes inference actually loads |
| the source pickle's SHA-256 | so the conversion is traceable to a reviewed file |
| the conversion report's SHA-256 | because "a pickle scan or `strict=True` alone does not establish safety" |

`cluster/convert_policy_checkpoint.py` is the only thing in this repository that
produces the third one.

**Status right now:** `not_run`. `cluster/asset_plan.py` pins neither an immutable
revision nor a SHA-256 for any of the three source files
(`unresolved_immutable_revision`, `no_downloaded_file_hashes`), so no conversion
can be executed until someone downloads the files and records what they got. The
converter refuses to invent either value.

---

## What it refuses, and why each refusal exists

The build spec's rule is the whole design:

> Pin and review remote code before enabling `trust_remote_code`. Construct known
> architectures and prefer safetensors or supported weights-only loaders. If
> legacy `.pt`/`.pkl` conversion is necessary, isolate it in a disposable
> unprivileged environment without credentials or network, validate output
> schemas, and retain converted artifact hashes. A pickle scan or `strict=True`
> alone does not establish safety.

| Refusal | Trigger |
|---|---|
| credentials present | any of 18 named variables, or any variable whose name contains `API_KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `ACCESS_KEY`, `APIKEY`, `PRIVATE_KEY`, with a non-empty value |
| credential files present | `~/.netrc`, `~/.aws/credentials`, `~/.huggingface/token`, `~/.ssh/id_rsa` and six more |
| privileged | `euid == 0` |
| unreviewed pickle global | any global outside the per-policy allowlist, **named** in the refusal |
| executable machinery in the pickle | a global under `os`, `posix`, `subprocess`, `builtins`, `urllib`, … — refused outright, and `--approve-global` cannot override it |
| unverified source bytes | no expected SHA-256, or a measured digest that disagrees |
| untraceable source | no immutable 40-hex revision from the plan or `--source-revision` |
| unresolved license not acknowledged | `minivla` and `minivla_vq` without `--acknowledge-unresolved-license` |
| existing artifact | the output or report path already exists (`O_EXCL`) |
| wrong container | `--output` not ending in `.safetensors` |
| non-tensor leaf in the state dict | a safetensors file holds tensors only, and the converter will not invent one |
| unidentifiable state dict | no expected top-level key and not a flat tensor mapping; the refusal lists the keys it did see |

Two of those deserve emphasis. **The allowlist is never widened to make a load
succeed**, and unknown modules are never stubbed into existence — that is exactly
the "a stub unpickler is not a security boundary" failure. And every approved
non-builtin global is aliased to an inert recorder, so the upstream class's
`__reduce__` and `__setstate__` never run even when the name is allowed.

---

## Prerequisites

| Requirement | Why | How to confirm |
|---|---|---|
| A disposable, unprivileged container | The spec requires it; the converter enforces the parts it can observe | the report's `isolation` block |
| No credentials in the environment | The conversion needs none | `env -u HF_TOKEN -u BASETEN_API_KEY …`, or `env -i` |
| No network | Nothing here fetches anything | the operator's to establish — see below |
| CPU Torch 2.5 or newer | Older releases cannot enumerate a checkpoint's globals before loading it | the converter refuses and names the missing function |
| `safetensors` | the only container the adapters accept | as above |
| The downloaded source pickle, plus the SHA-256 recorded at download time | The plan pins no hash | `cluster/download_assets.py` evidence |
| The immutable commit the file came from | The adapters require a 40-hex `checkpoint_revision` | as above |

**Network isolation is not verified and is not claimed.** The tool performs no
network access, but it cannot establish from inside the environment that the
environment has none. The report says so in
`isolation.network_isolation_note`, and
`network_isolation_not_verifiable_from_inside` is always in
`verification_blockers`.

---

## The three conversions

```bash
python -m cluster.convert_policy_checkpoint --list-policies
```

| `--policy` | Asset plan entry | Source file | License status in the plan |
|---|---|---|---|
| `minivla` | `minivla-vq-bridge` | `checkpoints/step-362500-epoch-21-loss=0.2259.pt` (~5.55 GB) | `unresolved`, redistribution prohibited |
| `minivla_vq` | `minivla-pretrain-vq` | `pretrain_modvq+…+nlatent-512/checkpoints/model.pt` (~9.5 MB) | **`absent_cardData_null`** — no license at all |
| `openpizero` | `open-pi-zero` | `bridge_beta_step19296_2024-12-26_22-30_42.pt` (~11.77 GB) | `advertised_unverified` MIT, local use only |

Repository IDs, paths, sizes and license status are read from
`cluster/asset_plan.py` at run time, not duplicated here or in the converter.

### Step 1 — inspect, converting nothing

```bash
python -m cluster.convert_policy_checkpoint --policy openpizero \
  --source /scratch/$USER/plumb/models/open-pi-zero/bridge_beta_step19296_2024-12-26_22-30_42.pt \
  --inspect
```

This enumerates the pickle's globals **before** anything is deserialized, measures
the file's SHA-256, and writes nothing. It reports rejected globals instead of
refusing, because finding out what is in the file is the point. The reviewed
allowlist for all three policies is currently **empty**: the expectation is that
these checkpoints carry nothing beyond Torch's own `weights_only` defaults. That
is an expectation about files this repository has never opened, not a claim about
their bytes — which is why this step comes first.

Add `--report evidence/<policy>-inspection.json` to keep the inspection itself as
evidence; it is written with the same refuse-to-overwrite rule as a conversion
report. Without `--report`, `--inspect` prints and writes nothing.

If `--inspect` reports a global, review the actual class by hand, then either
obtain a safetensors export from upstream or re-run with
`--approve-global MODULE.NAME --reviewer "<name>"`. The reviewer's name goes into
the report. Do not extend an allowlist to make a refusal go away.

### Step 2 — dry run

```bash
python -m cluster.convert_policy_checkpoint --policy openpizero \
  --source .../bridge_beta_step19296_2024-12-26_22-30_42.pt \
  --output /scratch/$USER/plumb/models/open-pi-zero/bridge-beta-step19296.safetensors \
  --source-revision <40-hex commit> \
  --expected-source-sha256 <64-hex digest>
```

Dry run is the default. It applies every precondition that can be checked without
deserializing anything — isolation, source hash, revision, license
acknowledgement, output availability, output container, and the globals allowlist
— and then stops. What it cannot check in advance is what the object graph turns
out to contain: a non-tensor leaf, an unidentifiable state dict or an incomplete
`--strict-verify` can still refuse during `--execute`.

### Step 3 — execute

Add `--execute`. For `minivla` and `minivla_vq`, also add
`--acknowledge-unresolved-license`.

```bash
python -m cluster.convert_policy_checkpoint --policy minivla_vq \
  --source .../pretrain_vq/.../checkpoints/model.pt \
  --output /scratch/$USER/plumb/models/minivla/pretrain-vq.safetensors \
  --source-revision <40-hex commit> --expected-source-sha256 <64-hex digest> \
  --acknowledge-unresolved-license --execute
```

Optionally verify full state-dict coverage against the pinned loader:

```bash
  --strict-verify open_pi_zero.model:build_bridge_wrapper \
  --strict-verify-repo /scratch/$USER/plumb/code/open-pi-zero \
  --strict-verify-loader-revision c3df7fb062175c16f69d7ca4ce042958ea238fb7
```

This is opt-in because it **imports third-party loader code**, which executes it.
That is the same trust as normal inference, taken inside the disposable
environment. Without it, `strict_state_dict_verification.performed` is `false`,
`state_dict_coverage_unverified` is a blocker, and the OpenPiZero profile's
`state_dict_coverage_verified` must stay `false` — the adapter will refuse, which
is the correct outcome.

---

## What gets written, in what order

A report cannot contain its own hash, so the order is fixed:

1. the output and report paths are reserved with `O_EXCL`, before the source is
   even hashed — an existing artifact stops the run immediately;
2. tensors are written to a sibling temporary file, `fsync`ed, then moved onto the
   reserved output path;
3. the artifact's SHA-256 is measured from the file on disk;
4. the report is written to `<output>.conversion-report.json`;
5. the report's own SHA-256 is measured from the file on disk and printed.

On a refused `--execute`, both reservations are released and the failure report is
written to a `<output>.conversion-report.refused-<8 hex>.json` sibling. Evidence is
kept and the canonical names stay free for a corrected re-run. Nothing is ever
overwritten. A dry run reserves nothing, so it writes nothing at all — including
when it refuses.

Keys are flattened with `.`, so `checkpoint["model"]["projector"]["fc.weight"]`
becomes `projector.fc.weight` — a real `torch.nn.Module` parameter name, which is
what makes `--strict-verify` and any reconstruction shim mechanical rather than
guesswork. `output.state_dict_key`, `output.flatten_separator` and
`output.ignored_top_level_keys` record the mapping exactly.

---

## The report

`schema_version: 1`, `kind: plumb_policy_safe_checkpoint_conversion`,
`status` one of `completed`, `dry_run`, `inspected`, `failed`.

| Section | Fields |
|---|---|
| `converter` | `path`, `source_revision` (git commit, or `null` offline), `source_sha256` (this file's own digest, always present), `python_version`, `torch_version`, `safetensors_version` |
| `source` | `asset_plan_name`, `repo_id`, `repo_type`, `repo_path`, `url_path` (percent-encoded — MiniVLA's `%3D`), `source_url`, `download_url`, `revision`, `revision_origin`, `revision_pinned_in_asset_plan`, `sha256_pinned_in_asset_plan`, `local_path`, `sha256`, `bytes`, `expected_sha256`, `expected_sha256_origin`, `sha256_verified`, `expected_bytes`, `historical_estimate_bytes`, `basename_matches_asset_plan` |
| `license` | `license`, `license_status`, `redistribution`, `acknowledgement_required`, `acknowledged_by_operator`, `notices` (the plan's own traps and notes) |
| `deserialization` | `method`, `weights_only`, `unsafe_fallback_available`, `globals_enumerated_before_load`, `expectation`, `observed_globals`, `reviewed_globals`, `operator_approved_globals`, `operator_reviewer`, `approved_globals`, **`approved_globals_used`**, **`rejected_globals`**, `never_approvable_globals_observed`, `inert_aliased_globals` |
| `output` | `format`, `path`, `sha256`, `bytes`, `tensor_count`, `total_tensor_bytes`, `dtype_summary`, `state_dict_key`, `ignored_top_level_keys`, `flatten_separator`, `metadata` |
| `strict_state_dict_verification` | `performed`, `entry_point`, `loader_revision`, `missing_keys`, `unexpected_keys`, `note` |
| `isolation` | `credential_environment_variables_present`, `credential_files_present`, the three lists of what was *checked*, `euid`, `unprivileged`, `network_isolation_verified`, `network_isolation_note` |
| top level | `status`, `mode`, `policy`, `verification_blockers`, `security_note`, `error` |

`security_note` states plainly that allowlisted deserialization is a mitigation
and not a proof of safety. It is also embedded in the safetensors metadata, so it
travels with the artifact.

`verification_blockers` is the honest part. A completed MiniVLA conversion today
still reports:

```json
["network_isolation_not_verifiable_from_inside",
 "no_downloaded_file_hashes",
 "state_dict_coverage_unverified",
 "unresolved_license_or_access_terms"]
```

### Licenses are never invented

`Stanford-ILIAD/pretrain_vq` declares no license (`cardData: null`). The report
records `license: null`, the safetensors metadata **omits** any `license` key and
carries a `license_note` saying PLUMB records none rather than inventing one, and
the emitted adapter fields carry `license: null` with
`redistribution: prohibited_pending_resolution`. Local use may proceed;
mirroring may not, and private hosting is not a substitute for redistribution
rights.

---

## Wiring the result into the adapters

`--execute` prints a summary whose `policy_profile_fields` are *exactly* the
keyword arguments the consuming adapter needs.

For `minivla` and `minivla_vq` they construct
`plumb.policies.minivla.ConvertedWeightArtifact` directly:

```python
artifact = ConvertedWeightArtifact(**summary["policy_profile_fields"])
assert artifact.review_error() is None
```

`MiniVLAPolicyProfile` then needs, beyond the two artifacts: `checkpoint_revision`
and `vq_revision` (the same commits passed as `--source-revision`),
`loader_revision` for the reviewed `Stanford-ILIAD/openvla-mini` commit, and
`vq_license_acknowledged=True`.

For `openpizero` the fields are the converted-checkpoint fields of
`OpenPiZeroPolicyProfile`: `converted_checkpoint_path`,
`converted_checkpoint_sha256`, `source_pickle_sha256`, `conversion_report_sha256`
and `state_dict_coverage_verified`. The profile additionally needs its PaliGemma
support files, proprio layout, entry point and application cache directory, none
of which this tool produces.

`tests/test_policy_converter.py` asserts against those real
`review_error()` implementations rather than a restatement of them, so a change to
either adapter's requirements fails here first.

---

## Known gaps, in files this tool does not own

1. **The pinned loaders read `.pt`, not `.safetensors`.** `plumb/policies/minivla.py`
   hands `converted_checkpoint.path` to `prismatic.load_vla`, and the OpenPiZero
   entry point takes the converted path too. Upstream `load_vla` calls
   `torch.load`. A small shim that reads the safetensors file and reassembles the
   nested structure — which `output.state_dict_key` and
   `output.flatten_separator` describe exactly — is still needed in the policy
   image. The converter deliberately records enough to make that shim mechanical
   instead of guessing at the wrapper's API here.
2. **`cluster/asset_plan.py` pins no revision and no SHA-256** for these three
   files. Until a download records them, every conversion needs
   `--source-revision` and `--expected-source-sha256` from the operator, and
   `no_downloaded_file_hashes` stays in the blockers. Adding the resolved values
   to the plan is the fix; the converter then prefers the plan and refuses an
   operator value that disagrees with it.
3. **The reviewed allowlists are empty by design.** They should be filled in from
   an actual `--inspect` of the real files, with the reviewer named, rather than
   from an assumption about what is inside them.
