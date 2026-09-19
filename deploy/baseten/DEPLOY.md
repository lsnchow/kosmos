# PLUMB Baseten deploy runbook

Exact ordered sequence for the moment GPU capacity and credentials are live.
Every step names the `model-contracts.json` nulls it fills.

None of the commands below has been executed against Baseten. `chain.py` *was*
loaded with `truss==0.18.30` installed, and `truss_chains`' own
`raise_validation_errors()` collected **no errors**: all nine Chainlets construct,
the entrypoint resolves eight dependencies at `retries=0`, every
`make_abs_path_here` path resolves, and the cached `ModelRepo`s build. Step 4
reproduces that check. It is not a push: this machine has no GPU and no Baseten
credentials, so nothing here is evidence about GPU behaviour, capacity, latency,
or cost.

Run everything from the repository root.

---

## 0. Preconditions that are not commands

| Precondition | Why it blocks |
| --- | --- |
| Hugging Face access accepted for `nvidia/Cosmos3-Nano`, `openvla/openvla-7b`, `Qwen/Qwen2.5-VL-7B-Instruct` | The three cached volumes download at container start using the `hf_access_token` secret. |
| `hf_access_token` saved in Baseten org secrets | `chains.Assets(secret_keys=["hf_access_token"])` only grants access to a secret that exists. Manage at `app.baseten.co/settings/secrets`. |
| OpenVLA remote-code review actually performed | `openvla-7b` requires `trust_remote_code=True`. Step 2 records the review; do not set the acknowledgement variable without having read revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`. |
| Concurrency cap raised on the account | `max_replica: 100` on the world worker is a *request*. A configured maximum is neither an active replica count nor reserved capacity. |

---

## 1. Environment variables

Two distinct sets. The **push-time** set is baked into the images by
`ChainletOptions.env_variables` when `truss chains push` imports `chain.py`; the
**client-side** set is read later by `plumb.platform.BasetenPlatformConfig.from_env`.

### 1a. Client-side (read by `plumb/platform.py`)

```bash
export BASETEN_API_KEY='<account api key>'                 # required, never logged
export BASETEN_CHAIN_ASYNC_URL='<filled in step 6>'        # required
export BASETEN_WEBHOOK_SECRET='<async callback signing secret>'   # optional, must be non-empty if set
export BASETEN_MANAGEMENT_BASE_URL='https://api.baseten.co/v1'    # optional, this is the default
```

`BASETEN_CHAIN_ASYNC_URL` cannot be filled until after the push: it contains the
Chain ID. Export a placeholder now and the config constructor will reject it.

### 1b. Push-time knobs (baked into the images)

```bash
# World-model micro-batching. predict_concurrency follows BATCH_MAX automatically.
export PLUMB_WORLD_BATCH_MAX=16                # default 16, clamped to 1..256
export PLUMB_WORLD_BATCH_WINDOW_MS=15          # default 15.0 ms, clamped to 0..5000

# Replica ceilings. Baseten's own max_replica default is 1, so these matter.
export PLUMB_WORLD_MAX_REPLICA=100             # default 100
export PLUMB_JUDGE_MAX_REPLICA=100             # default 100
export PLUMB_POLICY_MAX_REPLICA=25             # default 25
export PLUMB_VALIDITY_MAX_REPLICA=10           # default 10
export PLUMB_CONTROLLER_MAX_REPLICA=10         # default 10

# World-model arm.
export PLUMB_WORLD_VARIANT=cosmos3_nano        # cosmos3_nano | cosmos3_edge
export PLUMB_WORLD_RESOLUTION_TIER=256         # 256 | 480 | 704 | 720
# Required only for the Edge speed arm; must be a 40-hex immutable revision.
# export PLUMB_WORLD_EDGE_REVISION='<resolve from the HF API, record retrieval time>'

# Off unless plumb.adapters.worlds gains a reviewed generate_batch.
export PLUMB_WORLD_FUSED_BATCH=0

# OpenVLA remote-code review acknowledgement. Must equal the reviewed revision
# exactly, or OpenVLAWorker returns blocked.
export PLUMB_OPENVLA_REVIEWED_REMOTE_CODE_ACK='47a0ec7fc4ec123775a391911046cf33cf9ed83f'

# Optional: path to assets.lock.json inside the image, once it exists.
# export PLUMB_ASSET_MANIFEST_PATH=/app/assets.lock.json

# Optional: override the cached-volume mount root (default /app/model_cache).
# export PLUMB_MODEL_CACHE_ROOT=/app/model_cache

# Stage-A calibration class. Default is the primary class. The development class
# is safe to expose because plumb.validity stamps those reports
# primary_scoring_eligible=False with an uncalibrated_development_mode code.
export PLUMB_VALIDITY_CALIBRATION_CLASS=calibrated_primary   # | uncalibrated_development
```

Verify what will actually be baked before pushing:

```bash
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("plumb_chain", "deploy/baseten/chain.py")
chain = importlib.util.module_from_spec(spec); spec.loader.exec_module(chain)
print("BATCH_MAX        ", chain.BATCH_MAX)
print("BATCH_WINDOW_MS  ", chain.BATCH_WINDOW_MS)
print("world variant    ", chain.selected_world_variant().variant_id,
      chain.selected_world_variant().revision or "<NO REVISION - arm will block>")
for item in chain.AUTOSCALING_HYPOTHESES:
    print("  %-28s %s" % (item.chainlet_name, item.as_settings()))
PY
```

**Fills:** nothing yet. This step only makes the baked configuration auditable.

---

## 2. A dedicated push venv

`truss` must not share the project venv: the project pins `numpy<3` and test
tooling, and a resolver conflict during a deploy window is avoidable pain.

```bash
python3 -m venv /tmp/plumb-truss
/tmp/plumb-truss/bin/pip install --upgrade pip
/tmp/plumb-truss/bin/pip install 'truss==0.18.30'
/tmp/plumb-truss/bin/truss --version
```

Pin the exact version. `chain.py` was validated against 0.18.30; a newer SDK may
rename `pip_requirements_file` (already deprecated in favour of
`requirements_file`) or change `ChainletOptions`. If you must use a different
version, first confirm the surface still matches:

```bash
/tmp/plumb-truss/bin/python - <<'PY'
import inspect, truss_chains as chains
from truss.base import truss_config
print("Compute      ", inspect.signature(chains.Compute.__init__))
print("Assets       ", inspect.signature(chains.Assets.__init__))
print("RemoteConfig ", sorted(chains.RemoteConfig.model_fields))
print("Options      ", sorted(chains.ChainletOptions.model_fields))
print("DockerImage  ", sorted(chains.DockerImage.model_fields))
print("ModelRepo    ", sorted(truss_config.ModelRepo.model_fields))
print("BasetenImage ", [member.name for member in chains.BasetenImage])
PY
```

Expect `RemoteConfig` to contain `assets`, `build_commands`, `compute`,
`docker_image`, `name`, `options` — and **no** autoscaling field. That absence is
why step 7 exists. `chain.py` feature-detects `requirements_file`
versus the deprecated `pip_requirements_file` from `DockerImage.model_fields`, so
either name works without a code change.

Then log in:

```bash
/tmp/plumb-truss/bin/truss login   # or: export BASETEN_API_KEY=... (truss reads it too)
```

**Fills:** `sdk.verified_against_version` if you pin a different version.

---

## 3. Stage the `plumb` package

```bash
python deploy/baseten/stage_packages.py
python deploy/baseten/stage_packages.py --check    # must print "Staging is current"
```

`chain.py` declares `external_package_dirs=[make_abs_path_here("../_chain_packages")]`.
`truss_chains` copies the *contents* of that directory into each image's
`packages/` dir, which is what makes `import plumb` work inside a Chainlet.

Do **not** replace this with the repository root: `gather_chain` applies only
truss's built-in `.truss_ignore`, which excludes `.venv` and `.git` but not
`web/node_modules` (~117 MB), and the chain has nine Chainlets.

If you skip this step the push still succeeds and every Chainlet returns
`status="blocked"` with `plumb ... is not importable inside this image`. That is
the intended fail-closed behaviour, not a silent fallback.

**Fills:** nothing. Prevents nine broken images.

---

## 4. Pre-push validation (no network, no GPU, no push)

```bash
.venv/bin/python -m pyflakes deploy/
.venv/bin/python -m pytest tests/test_chain.py -q
cd deploy/baseten && for f in requirements/*.txt; do \
  printf '%s  sha256:%s\n' "$f" "$(shasum -a 256 "$f" | awk '{print $1}')"; done; cd -
```

The digests must match `chain_topology.requirements_lock_digests` in
`model-contracts.json`. If they do not, a requirements file changed and the
recorded lock input is stale.

Then run the framework's **own** validator from the push venv. This constructs all
nine `RemoteConfig`s, resolves every `make_abs_path_here` path, builds the cached
`ModelRepo`s, and runs `truss_chains`' endpoint and I/O checks — everything a push
does except contacting Baseten:

```bash
/tmp/plumb-truss/bin/pip install pytest
/tmp/plumb-truss/bin/python -m pytest tests/test_chain.py -k framework_validator -v
```

Or directly, which also prints the resolved compute and asset specs:

```bash
/tmp/plumb-truss/bin/python - <<'PYEOF'
import sys; sys.path.insert(0, "deploy/baseten")
import chain
from truss_chains import framework
framework.raise_validation_errors()          # collected during class creation
for name in chain.WORKER_REQUIREMENTS_FILES:
    config = getattr(chain, name).remote_config
    compute, assets = config.get_compute_spec(), config.get_asset_spec()
    print("%-20s %-26s gpu=%-22s conc=%-3s cached=%d secrets=%s" % (
        name, config.name, compute.accelerator.accelerator or "-",
        compute.predict_concurrency, len(assets.cached), sorted(assets.secrets) or "-"))
print("entrypoint deps:", sorted(framework.get_descriptor(chain.RolloutController).dependencies))
PYEOF
```

`raise_validation_errors()` prints nothing and raises nothing when the Chain is
valid, so success looks like the table alone. Expect `plumb-world-worker` at
`conc=16` (or your `BATCH_MAX`), exactly one cached repo on the world, OpenVLA,
and judge workers, and eight entrypoint dependencies. This was run against
`truss==0.18.30` and passed.

**Fills:** nothing. Catches a stale lock and an SDK-surface mismatch before a paid
build.

---

## 5. Push

```bash
/tmp/plumb-truss/bin/truss chains push ./deploy/baseten/chain.py --environment production
```

Notes that matter:

- `@chains.mark_entrypoint("PLUMB Rollout Controller")` means the file alone is
  enough; no class name argument is needed.
- Nine images build: one per policy stack (five), world, judge, validity, and the
  CPU controller. The MiniVLA image additionally compiles `flash-attn==2.5.5`
  from source in `build_commands`, which needs `nvcc` in the base image. If that
  build fails, drop the `build_commands` entry and re-push: `MiniVLAWorker`
  returns `blocked` either way, because no reviewed loader exists.
- `--environment production` is deliberate. A **development** deployment is
  capped at one replica and cannot demonstrate autoscaling.
- Do not push from a dirty `deploy/_chain_packages` (re-run step 3 after any
  `plumb/` edit).

**Fills:** the build-log image digests feed every
`workers.*.deployment_container_digest` in step 9.

---

## 6. Read back the Chain ID and build the async URL

`truss chains push` prints the Chain and deployment IDs. Confirm them against the
management API rather than copying from terminal scrollback:

```bash
curl -s -H "Authorization: Api-Key ${BASETEN_API_KEY}" \
  https://api.baseten.co/v1/chains | python -m json.tool | tee /tmp/plumb-chains.json
```

Take the `id` of the chain whose name is `PLUMB Rollout Controller`. Then build
the submission URL — do not hand-assemble it, because
`plumb.platform.ChainAsyncEndpoint` validates the shape strictly (HTTPS only,
`chain-*.api.baseten.co` host, port 443 or default, no credentials, query, or
fragment, and a path of exactly `{production|development}/async_run_remote` or
`{deployment|environments}/<id>/async_run_remote`):

```bash
CHAIN_ID='<chain id from above>'
.venv/bin/python - <<PY
from plumb.platform import ChainAsyncEndpoint
endpoint = ChainAsyncEndpoint.for_target("${CHAIN_ID}", "production")
print(endpoint.url)
PY
export BASETEN_CHAIN_ASYNC_URL='https://chain-<CHAIN_ID>.api.baseten.co/production/async_run_remote'
.venv/bin/python -c "from plumb.platform import BasetenPlatformConfig as C; print('config ok:', C.from_env().chain_endpoint.url)"
```

Other accepted targets: `development`, `deployment/<deployment_id>`,
`environments/<environment_name>`.

`/async_predict` is for **model** deployments and is rejected. Only this
entrypoint URL uses the external async queue; internal Chainlet calls are awaited
RPCs.

**Fills:** the client-side `BASETEN_CHAIN_ASYNC_URL`. Records the Chain ID that
every later step needs.

---

## 7. Apply the autoscaling hypotheses

`chains.RemoteConfig` in truss 0.18.30 cannot declare replica counts, so this is
a post-push management-API step. Print the exact values:

```bash
.venv/bin/python -c "
import json, sys; sys.path.insert(0, 'deploy/baseten')
import chain; print(json.dumps(chain.autoscaling_patch_payloads(), indent=2))"
```

Then, for each Chainlet:

1. Confirm the chain-deployment autoscaling route and the accepted field names
   against the Baseten management-API reference for **chain** deployments (the
   model-scoped route is not evidence of a chain route).
2. `PATCH` one Chainlet's autoscaling settings with `min_replica`,
   `max_replica`, `concurrency_target`, `autoscaling_window`.
3. `GET` the chain deployment and diff the echoed `autoscaling_settings` against
   what you sent. Save that response as evidence.

Verify the read-back path PLUMB already implements:

```bash
.venv/bin/python - <<'PY'
import asyncio
from plumb.platform import BasetenChainClient, BasetenPlatformConfig
async def main():
    client = BasetenChainClient(BasetenPlatformConfig.from_env())
    status = await client.chain_deployment_status(chain_id="<CHAIN_ID>", deployment_id="<DEPLOYMENT_ID>")
    for chainlet in status.chainlets:
        print(chainlet.name, chainlet.active_replica_count,
              chainlet.configured_min_replica, chainlet.configured_max_replica)
asyncio.run(main())
PY
```

Every value you just applied is an **initial hypothesis**, not a measurement.
`AGENT-BUILD-SPEC.md:285` is explicit that there is no assumed universal
throughput preset: re-derive `batch_max`, `predict_concurrency`, and
`max_replica` from Gate A's measured memory and full-chain throughput, then
re-push and re-PATCH.

**Fills:** `autoscaling.exact_management_api_body`.

---

## 8. Test and record the Chain queue-status route

`plumb.platform` reports the platform queue as **unavailable** until an
account-tested exact route exists. Test the candidate, keep the raw response as
evidence, and only then construct the verified route:

```bash
curl -s -D /tmp/plumb-queue-headers.txt \
  -H "Authorization: Api-Key ${BASETEN_API_KEY}" \
  'https://chain-<CHAIN_ID>.api.baseten.co/production/async_queue_status' \
  | tee /tmp/plumb-queue-status.json
```

If that 404s, try the documented alternatives in order and record which one
answered, with its exact response schema. Then upload the saved response to the
project artifact store and build the value:

```bash
.venv/bin/python - <<'PY'
from plumb.platform import VerifiedChainQueueRoute
route = VerifiedChainQueueRoute(
    url="https://chain-<CHAIN_ID>.api.baseten.co/production/async_queue_status",
    evidence_uri="<artifact store URI of the saved raw response>",
    # ISO-8601 WITH an offset. `date -u +%Y-%m-%dT%H:%M:%S+00:00` produces one.
    verified_at="2026-09-19T00:00:00+00:00",
)
print(route)
PY
```

All three fields are mandatory and validated: the URL must be HTTPS on a
`chain-*.api.baseten.co` host with no query or fragment, and `verified_at` must
carry an explicit UTC offset. A route without saved evidence is not verified, and
queue depth stays unavailable rather than being guessed.

Also confirm the field names in the response actually carry queued and
in-progress **request** counts. Request counts differ from logical episode counts
during retries, and the two must never be relabelled as each other.

**Fills:** `async_submission.queue_status_route`.

---

## 9. Turn the lock inputs into locks, and record digests

For each of the nine images, from a running replica:

```bash
# Per Chainlet, in a shell on the replica:
python -m pip freeze > /tmp/<chainlet>-freeze.txt      # or: uv pip freeze
ls -la /app/model_cache/                               # confirm the cached volumes mounted
cat /app/model_cache/cosmos3-nano/.git* 2>/dev/null    # or the HF snapshot ref metadata
```

Then:

1. Store each `freeze` output next to its lock input as
   `requirements/<name>.resolved.txt`.
2. Read each Chainlet's image digest from the chain-deployment management
   response (or the final image line of its build log) — never from a local
   `docker build`.
3. For the three cached models, read the revision actually materialised on the
   volume and compare it to the declared revision
   (`e59a53c25979a090fa8706c9acc0c254a6e89b92`,
   `47a0ec7fc4ec123775a391911046cf33cf9ed83f`,
   `cc594898137f460bfe9f0759e9844b3ce807cfb5`). A mismatch is a hard stop.
4. Time a cold replica start separately from steady-state inference. The recorded
   167.683719175 s Cosmos load came from a ComputeCanada cluster with
   `+computecanada` wheels and does not transfer; the whole point of the cached
   volume is to avoid paying it, and the residual is unmeasured.

**Fills:** every `workers.*.deployment_container_digest`,
`workers.*.deployment_model_revision`,
`workers.JudgeWorker.deployment_processor_revision`,
`workers.WorldWorker.deployment_backend_profile_hash`,
`workers.WorldWorker.assets.residual_cold_start_seconds`,
`chain_topology.lock_status`, and
`performance_target.measured_cold_start_seconds`.

---

## 10. Establish the pricing basis

USD stays unavailable until the account's billing semantics are evidenced. Do not
assume an API field named `price` is already hourly USD.

```bash
curl -s -H "Authorization: Api-Key ${BASETEN_API_KEY}" \
  https://api.baseten.co/v1/instance_types/prices \
  | tee /tmp/plumb-instance-prices.json
```

Then read the vendor/account contract for: currency, billing unit, which
resources the rate applies to, and tax/egress/storage treatment. Save both the
API response and the contract excerpt to the artifact store. Only then:

```bash
.venv/bin/python - <<'PY'
from plumb.platform import PriceUnit, VerifiedPriceBasis
basis = VerifiedPriceBasis(
    currency="USD",                       # exactly three uppercase letters
    unit=PriceUnit.PER_HOUR,              # PER_HOUR | PER_MINUTE | PER_SECOND
    evidence_uri="<artifact store URI of the saved price response + contract excerpt>",
    retrieved_at="2026-09-19T00:00:00+00:00",   # ISO-8601 with an explicit offset
)
print(basis)
PY
```

Cost then follows `estimated_usd = sum(rate_per_hour * allocated_resource_hours)
+ other_charges`, over nonoverlapping allocations, reported as two separate
views: a **marginal execution estimate** and a **total demonstration-run cost**
covering prewarm through cooldown including idle and cold capacity. Both stay
labelled estimates until billing reconciliation posts.

**Fills:** `telemetry_and_economics.pricing_currency`, `.pricing_unit`,
`.estimated_usd`, and later `.settled_cost`.

---

## 11. First real submission (one episode, not a burst)

Submit one complete logical episode. The application control plane must create
its ledger row *before* submitting and persist the returned platform request ID.

The `RolloutRequest` payload shape is defined by the pydantic models in
`chain.py`: `EpisodeControlPayload` in `policy.payload`, `WorldStagePayload`
fields in `world.payload`, `ValidityStagePayload` hints in `validity.payload`, and
`JudgeStagePayload` fields in `judge.payload`. All four `StageRequest`s must carry
the same `episode_id` and `protocol_hash` as the envelope, or the entrypoint
returns `failed: stage_request_identity_mismatch` without doing any work.

Expect one of these, and treat each as information rather than a problem:

- `blocked` with `certified_execute_prefix_unresolved` — the policy arm has no
  Gate-B-certified executed prefix. Correct behaviour; only OpenVLA declares one.
- `completed` validity with `validity: "unknown"` and a `no_motion_reference`
  reason code — `plumb.validity.StageAValidityGate` runs, but no `MotionReference`
  fitted from real Bridge trajectories is mounted, so the calibrated
  command-motion check cannot run. That is the gate working correctly.
- `blocked` on a policy arm with `A certified loader exists as
  plumb.policies.<Adapter>` — the block reason lists the exact profile fields the
  certified adapter still needs.
- `blocked` with `judge_reference_images_missing` or `judge_seeds_missing` — the
  frozen protocol has to supply provenance-backed reference panels and five
  distinct logged seeds. There are no matched five-task panels yet.
- `completed` with `judge_status: "unknown"` — a real result. Insufficient quorum
  is an explicit missing outcome, never a forced label.

**Fills:** nothing in `model-contracts.json`. Confirms the wiring end to end.

---

## 12. What is still not deployable after all of this

- **Gate A** has not run against the actual deployed payload.
- **Gate B** *failed* on the cluster for the pinned Cosmos/OpenVLA one-step
  profile (native one-action inference returned only the condition frame, jobs
  `937372`/`937423`). It must not be patched over with hidden future actions or
  padding.
- **Gate D** needs 150 blinded clips with human labels. None exist.
- Four of the six policy arms have no reviewed native loader and no immutable
  checkpoint revision, so they return `blocked` by design.
- `Stanford-ILIAD/pretrain_vq` declares no license, and PaliGemma access terms
  are unresolved.
- Durable callback association is unintegrated; HANDOFF.md records that Baseten
  submission retries are hard-disabled after ambiguous POST outcomes.

None of these is a scope cut. They are the dependencies the full build still
needs, and the chain reports every one of them as `blocked` rather than emitting
a number it cannot support.
