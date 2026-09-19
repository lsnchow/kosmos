# Asset provenance — licences and mirror byte identity

Two tools turn licence and access questions in `cluster/asset_plan.py` from claims
into retrieved, hashed evidence. Both are dry-run by default and write nothing
without `--execute`, matching `cluster/download_assets.py`.

| Tool | Answers | Writes |
|---|---|---|
| `cluster/resolve_licenses.py` | what licence a pinned repository actually ships, and whether PLUMB may run it locally | `results/provenance/licenses.json` |
| `cluster/compare_mirror.py` | whether an ungated mirror's bytes are provably the official host's bytes | `results/provenance/mirror-comparison-<asset>.json` |

Neither tool fabricates anything. `null` means unknown, an unreachable licence stays
unresolved with its reason recorded, and `byte_identity` reaches `proven` only when
every compared file matched a hash retrieved from the official repository.

---

## Three questions the plan used to answer as one

`AssetPlanEntry` now carries three separate verdicts, because collapsing them made
runnable assets look blocked.

| Field | Question | Who needs it |
|---|---|---|
| `license_status` | How well are the terms *established*? | `assets.lock.json`'s `verified` status |
| `local_use` | May PLUMB **run** these bytes? | the primary matrix |
| `redistribution` | May PLUMB **ship** these bytes? | nothing PLUMB does |

`LicenseStatus.VERIFIED` is the only value that asserts somebody read the licence,
and `AssetPlanEntry.errors()` refuses it without a `LicenseEvidence` record naming
the retrieved file, its SHA-256, the revision it was read at, and when. A status
cannot be hand-edited to `verified`.

`LocalUse.PERMITTED_LOCAL_ONLY_NO_EXPRESS_TERMS` is a recorded decision with its
basis, not a licence claim: the publisher distributes the bytes publicly, we read
no express grant, we run them locally, and we redistribute nothing. Use
`AssetPlanEntry.local_use_blockers()` to ask what stops the matrix;
`verification_blockers()` remains the stricter question the lock asks.

`Stanford-ILIAD/pretrain_vq` is the case that motivated the split. Its `cardData`
is null, so its licence is `null` and its redistribution stays
`prohibited_pending_resolution` — and its local use is permitted. MiniVLA is not
blocked by a redistribution right it never needed.

---

## `cluster/resolve_licenses.py`

```sh
.venv/bin/python -m cluster.resolve_licenses                      # dry run: prints every URL, fetches nothing
.venv/bin/python -m cluster.resolve_licenses --asset code-octo    # one row
.venv/bin/python -m cluster.resolve_licenses --execute            # fetch, hash, write the report
```

It reads the repository's metadata at the pinned revision, then the licence file
itself: GitHub's `repos/<owner>/<repo>/license?ref=<commit>` endpoint returns the
file inline, and Hugging Face's `api/models/<repo>/revision/<commit>` endpoint
lists the repository's files so the licence is fetched by name rather than guessed
at. The retrieved bytes are hashed and the text is matched against a conservative
signature table; a text that matches nothing, or more than one signature, is
reported unidentified with its hash recorded rather than labelled.

What it refuses to do:

- **Read a mutable branch.** A row with no pinned revision is refused with
  `no_pinned_immutable_revision`, because a licence read at `main` is not evidence
  about a pinned commit. Run `download_assets.py --from-plan <name> --resolve-revision`,
  review the commit, record it, then re-run.
- **Believe a model card.** A card's `license` field is metadata *about* a licence.
  It is recorded, normalised in case only (`mit` becomes `MIT`), and left at
  `advertised_unverified`. Hugging Face's `license: other` plus `license_name` and
  `license_link` convention is read too; an off-repo `license_link` is recorded and
  **not** followed, because an external page is mutable and is not a file at the
  pinned revision.
- **Believe a host's label.** When GitHub reports one SPDX id and the retrieved
  text reads as another, the hashed text is the record and the disagreement is
  noted. `NVIDIA/cosmos-framework` is a live example, below.
- **Truncate.** A response above the tool's ceiling is refused, not cut short.
  Hashing a truncated body would produce a digest of bytes nobody published.

`update_entry(entry, resolution)` applies one verdict to its plan row and validates
the result, so an inconsistent row raises instead of being written.

### What it established, 2026-09-19

Five rows moved from `advertised_unverified` to `verified`, each citing
`results/provenance/licenses.json`:

| Asset | Licence read | File | Revision |
|---|---|---|---|
| `code-openvla` | MIT | `LICENSE` | `c8f03f48…` |
| `code-auto-eval` | MIT | `LICENSE` | `3ea3ff44…` |
| `code-open-pi-zero` | MIT | `LICENSE` | `c3df7fb0…` |
| `code-irasim` | Apache-2.0 | `LICENSE` | `c72b6dad…` |
| `code-cosmos-framework` | **OpenMDW-1.1** | `LICENSE` | `c23e51f2…` |

`code-cosmos-framework` is a correction. The plan claimed Apache-2.0; the `LICENSE`
file at the pinned commit is the OpenMDW License Agreement version 1.1, and
GitHub's own API reports `spdx_id: NOASSERTION` for it. The hashed text is the
record.

Eighteen rows returned `no_pinned_immutable_revision`. That blocker is real work,
but it is not this tool's: resolving a head commit is a lookup, and adopting it as a
pin is a reviewed act.

`--revision NAME=SHA` resolves a licence at a commit that was looked up but not yet
adopted as a plan pin — that is how both PaliGemma rows were read. The flag refuses
a branch name and refuses a commit that disagrees with an existing pin, and the
report marks the row `revision_overridden: true` so a looked-up commit is never
mistaken for a reviewed one.

---

## `cluster/compare_mirror.py`

Spec section 1 permits `leo009/paligemma-3b-pt-224` as a candidate mirror "only
after provenance, hashes, and applicable terms are resolved" and forbids claiming
byte identity "without a saved comparison". This tool produces that comparison.

```sh
.venv/bin/python -m cluster.compare_mirror --asset paligemma \
    --official-revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
    --mirror-revision 39996beb6fb17c5d16a50d3ef8f7a96ad9d03986 --execute
```

It fetches each file **from the mirror**, hashes the bytes itself, and compares
them against what the official repository publishes without accepting terms. Two
hash channels count, and each match records which one carried it:

| Channel | Where it comes from |
|---|---|
| `official_lfs_sha256` | the Hub's `lfs.sha256` for large files |
| `official_git_blob_sha1` | the Hub's `blobId`, which is git's own object hash |

Sizes are recorded and never count. Two different files of equal length are
trivial to produce, so a size match alone leaves the verdict
`official_hash_unobtainable`. A SHA-1 match is accepted only together with an exact
byte-length match, and the report states that SHA-1 is not collision-resistant
against a determined adversary.

`byte_identity` is a computed property, not a field anyone sets. It is `proven` only
when every requested file matched a retrieved official hash; anything else is
`unproven` with a per-cause reason list. `MirrorEvidence.errors()` refuses a
`proven` claim that its own file counts do not support.

### What it established, 2026-09-19

`results/provenance/mirror-comparison-paligemma.json`: **`byte_identity: proven`,
7 of 7 files matched.**

| File | Bytes | Channel |
|---|---|---|
| `tokenizer.json` | 17,549,604 | `official_lfs_sha256` |
| `tokenizer.model` | 4,264,023 | `official_lfs_sha256` |
| `tokenizer_config.json` | 39,968 | `official_git_blob_sha1` |
| `added_tokens.json` | 24 | `official_git_blob_sha1` |
| `special_tokens_map.json` | 607 | `official_git_blob_sha1` |
| `preprocessor_config.json` | 699 | `official_git_blob_sha1` |
| `config.json` | 1,027 | `official_git_blob_sha1` |

The comparison was possible because Hugging Face publishes a gated repository's
file sizes, git blob ids and LFS SHA-256s anonymously. No gated download was
performed and no terms were accepted to obtain them. Re-running the tool at the
same two commits reproduces every hash.

### The licence did not move

The Gemma Terms of Use govern these weights whichever host supplied the bytes.
Proven identity removes the gated **download**; it does not touch the licence, and
no tool here accepts a licence or records one as accepted. Accordingly:

- the mirror row keeps `local_use=requires_accepted_terms` and `fetch=False`;
- `update_mirror_entry` refuses to relax `local_use` even for a proven comparison;
- every report carries `license.terms_accepted: false` and the governing-licence
  note, and both PaliGemma rows carry that note in the plan;
- the mirror's own model card advertises `license: gemma`, which is the same
  obligation the official repository carries. The licence travelled with the bytes.

`plumb/policies/openpizero.py` already accepts the mirror once
`PaliGemmaSupportFiles.mirror_comparison_report_sha256` is set — use the
`REPORT <path> sha256=<digest>` line the tool prints for the exact saved file. It
also requires `terms_accepted=True` regardless of which repository served the
bytes, which is the correct remaining gate.

---

## What is still gated

| Blocker | Cleared by | Self-serve? |
|---|---|---|
| `no_retrieved_license_file_evidence` | `resolve_licenses.py --execute` | yes, for any pinned row |
| `mirror_byte_identity_unproven` | `compare_mirror.py --execute` | yes; done for PaliGemma |
| `local_use_terms_unresolved` | an explicit `local_use` verdict | yes; only `irasim-bridge-archive` remains, and its terms ship inside the archive |
| `unresolved_immutable_revision` | `download_assets.py --resolve-revision` plus a human reviewing the commit | yes, but it is a review, not a lookup |
| `no_downloaded_file_hashes` | `download_assets.py --execute` on a cluster scratch path | yes, on the cluster |
| `gated_access_terms_not_recorded_as_accepted` | **a person accepting the Gemma terms** | **no** |

One genuine approval remains in the whole plan: somebody with a Hugging Face
account must accept the Gemma terms for `google/paligemma-3b-pt-224` before
OpenPiZero's support files may be *used*. Byte identity has removed the need to
download from the gated host; it cannot remove that. `nvidia/Cosmos-1.0-Guardrail`
is also gated and stays `fetch=False` by design, so it needs no approval at all.

## Reproducing the evidence

The two commands below regenerate `results/provenance/licenses.json` and
`results/provenance/mirror-comparison-paligemma.json` exactly as the plan cites
them. The `--revision` overrides are the commits the PaliGemma rows were read at;
neither row pins them.

```sh
.venv/bin/python -m cluster.resolve_licenses --execute \
    --revision paligemma-official=35e4f46485b4d07967e7e9935bc3786aad50687c \
    --revision paligemma-mirror-candidate=39996beb6fb17c5d16a50d3ef8f7a96ad9d03986
.venv/bin/python -m cluster.compare_mirror --asset paligemma \
    --official-revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
    --mirror-revision 39996beb6fb17c5d16a50d3ef8f7a96ad9d03986 --execute
.venv/bin/python -m pytest tests/test_license_resolution.py -q
```

Each licence file's SHA-256 and each mirror file's SHA-256 reproduce exactly; only
`generated_at`, `retrieved_at` and the report's own digest move between runs.

The tests inject a fake fetcher and never touch the network. They cover the cases
that matter: an unreachable licence stays unresolved, `pretrain_vq` yields a null
licence with prohibited redistribution, local use and redistribution move
independently, `byte_identity` cannot be `proven` when an official hash was
unobtainable, a size match alone never proves identity, an oversize body is refused
rather than truncated, every verified row cites a file that exists, and the
Gemma-licence note survives onto the mirror row.
