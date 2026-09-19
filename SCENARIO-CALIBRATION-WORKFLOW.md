# Scenario and calibration workflow

This workflow prepares Gate C and Gate D evidence. It does not mark either gate
as passed. A frozen manifest proves only its local structure, selected source
artifacts, hashes, and declared split boundaries. Scene/reset/control parity,
human labels, and judge outputs remain separate evidence requirements.

## 1. Source records

Create one explicit JSON source manifest. It is the only file the scenario
importer follows. Every artifact path is relative to this file; allowed source
types are PNG/JPEG/WebP images and JSON state. Pickle, torch, numpy, checkpoint,
remote URL, and arbitrary object formats are rejected and never deserialized.

```json
{
  "schema_version": 1,
  "source_manifest_id": "<immutable-source-manifest-id>",
  "provenance_kind": "real_robot",
  "records": [{
    "source_dataset": "<dataset-id>",
    "source_revision": "<immutable-revision>",
    "episode_id": "<episode-id>",
    "frame_id": "<frame-id>",
    "start_lineage_id": "<physical-start-lineage-id>",
    "image_timestamp": "<source-image-timestamp>",
    "state_timestamp": "<source-state-timestamp>",
    "state_convention": "<declared-Bridge-8D-convention>",
    "image": {"path": "images/<frame>.png", "media_type": "image/png", "sha256": "sha256:<digest>"},
    "state": {"path": "states/<frame>.json", "media_type": "application/json", "sha256": "sha256:<digest>"},
    "goal_references": [{"path": "goals/<goal>.png", "media_type": "image/png", "sha256": "sha256:<digest>"}]
  }]
}
```

The state JSON itself contains either a JSON array or one of `bridge_state`,
`state`, or `proprio`, with exactly eight finite numeric values. The importer
checks this schema, but it does not assert that the vector has physical fidelity.

## 2. Scenario proposal and freeze

Hash the source manifest bytes and put the prefixed digest in
`source_manifest_sha256`. The scenario manifest must explicitly say
`provenance_kind: "real_robot"`; omitted provenance is rejected.

```json
{
  "schema_version": 1,
  "manifest_id": "<new-scenario-panel-id>",
  "provenance_kind": "real_robot",
  "source_manifest_sha256": "sha256:<source-manifest-byte-digest>",
  "shared_policy_ids": ["OpenVLA", "OpenPiZero", "Octo", "MiniVLA", "SuSIE", "SuSIE_LL"],
  "tasks": {
    "close_drawer": {
      "starts": ["<50 distinct primary start records>"],
      "panels": {
        "development": ["<disjoint real starts>"],
        "heldout_calibration": ["<disjoint real starts>"],
        "cost_confirmation": ["<disjoint real starts>"]
      }
    }
  }
}
```

Repeat the task group for all five exact benchmark task IDs. `Octo` is the
canonical measurement/reference key and is bound to the required **Octo-Small
v1.0** policy identity; it is not permission to substitute a different Octo
checkpoint. Each start record has all of these fields:

```json
{
  "start_id": "<unique-start-id>",
  "start_lineage_id": "<unique-physical-source-lineage>",
  "source_dataset": "<dataset-id>", "source_revision": "<revision>",
  "episode_id": "<episode>", "frame_id": "<frame>",
  "image_timestamp": "<timestamp>", "state_timestamp": "<timestamp>",
  "image_hash": "sha256:<digest>", "state_hash": "sha256:<digest>",
  "state_convention": "<same-as-source-record>",
  "camera": "<camera-id>", "crop": {"<declared-crop>": "<value>"},
  "camera_calibration": {"<declared-calibration>": "<value>"},
  "scene": "<scene-id>", "scene_configuration": {"<objects/reset>": "<value>"},
  "initial_state_stratum": "<stratum>",
  "instruction": "Close the drawer",
  "goal_reference_hashes": ["sha256:<digest>"],
  "goal_reference_provenance": {"<declared-goal-source>": "<value>"}
}
```

The instruction has to be the exact registry string for its task. A physical
source lineage and a start ID may occur once across all tasks and panels:
neighboring frames and different world seeds do not create independent starts.

```bash
# Both input files must already exist locally; no remote source is fetched.
plumb scenarios import --scenario scenario-proposal.json --sources source-records.json --output scenario-imported.json
plumb scenarios freeze --scenario scenario-imported.json --output freezes/scenario-panel-v1.json
plumb scenarios readiness --scenario freezes/scenario-panel-v1.json
```

`freeze` creates its output once with a content SHA-256 and read-only mode. A
different amendment needs a new path and manifest ID. A structurally complete
readiness result is `ready_for_gate_c_evidence`, never a Gate C pass.

## 3. Calibration selection and owners

Create a separate 150-clip JSON/JSONL/CSV manifest: 20 development and 10
held-out clips for each task. Each row contains `clip_id`, `media_ref`,
`media_hash`, `task`, `split`, and `source_lineage_id`. `media_hash` is required
to freeze. Development and held-out lineages cannot overlap. The following
coordinator-only file supplies primary/cost lineage IDs extracted from the
frozen study panels; it prevents calibration selection from reusing them.

```json
{
  "primary": ["<every-primary-source-lineage>"],
  "cost_confirmation": ["<every-cost-confirmation-source-lineage>"]
}
```

```bash
plumb annotation freeze \
  --manifest calibration-clips.json \
  --reserved-lineages frozen-study-lineages.json \
  --output freezes/calibration-selection-v1.json
```

Declare two distinct human owners before emitting packets. This is an
accountability attestation, not identity proof, and must be retained by the
coordinator rather than put in a packet.

```json
{
  "<annotator-a-id>": {"owner_id": "<human-a-id>", "independent_attestation": true, "attested_at": "<timestamp>"},
  "<annotator-b-id>": {"owner_id": "<human-b-id>", "independent_attestation": true, "attested_at": "<timestamp>"}
}
```

The owner IDs must differ. Each held-out clip is assigned to both people;
development clips receive one deterministic assignment.

```bash
plumb annotation export \
  --manifest freezes/calibration-selection-v1.json \
  --annotator '<annotator-a-id>' --annotators '<annotator-a-id>' '<annotator-b-id>' \
  --ownership annotator-ownership.json --output packets/annotator-a.jsonl \
  --private-media-resolver private/annotation-media-resolver.json

plumb annotation export \
  --manifest freezes/calibration-selection-v1.json \
  --annotator '<annotator-b-id>' --annotators '<annotator-a-id>' '<annotator-b-id>' \
  --ownership annotator-ownership.json --output packets/annotator-b.jsonl
```

Packets use opaque clip and media tokens. They omit source lineage, split,
policy, backend, commands/actions, conditions, seeds, gate status, and filenames.
The optional media resolver contains the real media locations, is written with
owner-only permissions, and must stay with the trusted coordinator/viewer that
serves opaque tokens. Do not send it to either annotator.

## 4. Import and report

Annotators independently complete only their packet fields. Imports reject
unknown fields, non-assigned labels, duplicate labels, leaked metadata, wrong
opaque tokens, malformed milestones, and inconsistent completion/progress.

```bash
plumb annotation report \
  --manifest freezes/calibration-selection-v1.json \
  --annotations returned/annotator-a.jsonl --annotations returned/annotator-b.jsonl \
  --annotators '<annotator-a-id>' '<annotator-b-id>' \
  --ownership annotator-ownership.json \
  --judge-reports actual-heldout-raw-five-sample-reports.jsonl \
  --protocol frozen-judge-protocol.json --tolerances preregistered-tolerances.json
```

The report retains disagreements as unresolved and Gate D cannot pass without
the frozen selection, independent ownership declaration, complete provenance-
bound raw five-sample judge reports, and the supplied preregistered tolerances.
Actual human annotation, scene-parity review, judge inference, and external
Gate C/D evidence remain required work; this workflow never creates them.
