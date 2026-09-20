# Take the controls: gallery integration

## Current default: real image-conditioned live steering

**The working main flow is now [LIVE-DEMO.md](LIVE-DEMO.md).** Console → New
evaluation runs OpenVLA or manual steering against a real H100 IRASim worker.
Gallery → New interactive branch uses the selected clip's verified final image.
Actual browser policy/manual/source-branch flows passed. It deliberately starts
a fresh image-conditioned branch rather than pretending an MP4 contains a hidden
world-state checkpoint.

The sections below document the **earlier exact-checkpoint capability path**,
which remains separate. Its unsupported state must not be confused with the new
working `/api/demo/...` flow. The new main-gallery action no longer routes to
that unavailable exact-restoration dialog.

## What is implemented

The earlier integration opened a saved video through **Take the controls**.

The gallery pauses behind an accessible control dialog. The selected original
recording remains playable and clearly labelled as playback. The dialog checks
whether that exact saved source can start an interactive branch. Opening,
seeking, closing, or checking availability does not submit model inference.

For a supported source, the intended command path replaces the robot policy
with directional user commands. It generates an unscored branch; the original
recording and policy comparison remain unchanged. Commands are bounded chunks,
not a real-time robot connection. Releasing sends stop; closing drops queued
commands but does not cancel a chunk already submitted.

## Current limitation — actual steering is not enabled

The existing catalog contains MP4 output recordings, not resumable world-state
checkpoints. The deployed Baseten MVP produces a policy action, not interactive
world-model frames. The new UI therefore reports **Interactive generation
unavailable** and disables its directional controls for current recordings.

This is a completed main-flow entry point and source-binding/capability layer,
not a completed live steering backend. Successful unit tests using an explicit
test adapter are not evidence of GPU inference or checkpoint restoration.

The old `/live` → Run frames → free-play path is retained for developer use. It
starts from a task fixture; it does not restore the selected recording and must
not be used as a fallback for the gallery's source-bound control request.

## API contract

`GET /api/freeplay/status?video_id=<catalog-id>` is read-only. Its response
includes:

- `available`, `reason`, and `missing` explaining capability or blockers.
- `mode: "recording_branch"` for a selected recording.
- `source.video_id` and `source.sha256`, resolved from the current verified
  server catalog rather than client-supplied paths.
- `binding.exact_branch_supported`, checkpoint identity, and adapter name.
- `scored: false` and `qualified: false`.

Without `video_id`, the endpoint describes the legacy task-start capability;
that is a different mode, not permission to continue a recording.

Only an explicit user command may POST `/api/freeplay/step` with:

```json
{
  "direction": "right",
  "video_id": "<current catalog ID>",
  "source_sha256": "sha256:<current recording digest>"
}
```

Subsequent requests include the returned `session_id`. Both recording identity
fields must be supplied together. Unsupported or stale sources are refused
before a generation call, including attempts to bypass the check using `stop`.
Sessions bind their mode, recording identity, checkpoint hash and adapter;
they cannot silently switch to a different source. Concurrent commands for one
session are rejected.

The frontend independently requires a matching source ID/hash, recording-branch
mode and exact-branch support. It also verifies these fields on returned command
responses before displaying generated frames. A mismatched response is an error,
not a video to show under the chosen source's name.

## Checkpoints and adapter boundary

`plumb/world_videos.py` recognizes an optional source-report manifest:

```json
{
  "source_branch": {
    "schema": "plumb-source-branch-v1",
    "source_video_sha256": "sha256:<output video digest>",
    "adapter": "<checkpoint-compatible adapter ID>",
    "checkpoint": {
      "path": "<report-relative or recorded artifact path>",
      "sha256": "sha256:<checkpoint digest>"
    }
  }
}
```

This declaration alone does not enable controls. The catalog must resolve and
hash-check the checkpoint artifact within its allowed data root, and the
configured adapter must confirm it can restore that state. Checkpoint files are
not loaded or deserialized by the catalog. No MP4 frame is promoted to a hidden
world state simply because it can be decoded.

`plumb/freeplay.py` defines the source-bound adapter interface:

- `source_branch_status(source)`: local, side-effect-free capability check.
- `source_branch_step(...)`: actual restoration/continuation and generation.

The adapter's capability result must attest the manifest's adapter ID and
checkpoint SHA as well as exact-branch support. A generic `available: true`
without those identity bindings is rejected.

`BasetenChainBackend` can delegate to an explicitly injected adapter. It does
not provide a default live implementation. A supported adapter must preserve
session-specific evolving world state rather than reset to the same initial
checkpoint for every keypress. The source checkpoint is an initial branch
anchor; the manifest is not an arbitrary playback-timestamp selector.

Catalog IDs now incorporate evidence-root identity and content hash, so changed
bytes or identical relative names from different roots cannot select the wrong
recording. Refresh existing pages after upgrading the backend.

## Remaining work to make this genuinely interactive

1. Choose a world runtime with a verified restoration/continuation contract.
2. Export real compatible state checkpoints alongside selected recordings,
   including necessary action-compiler state and model/profile provenance.
3. Implement the concrete source adapter with bounded inference, isolated
   per-session evolving state, explicit failure/ambiguity handling and cleanup.
4. Persist generated branches and command history if they must survive a
   server restart. The present free-play session map is in memory.
5. Verify two or more successive commands use the previous result's correct
   state, then profile cold/warm latency and demonstrate it in a real browser.
6. Add user-selectable checkpoint positions only where actual saved checkpoints
   exist. Keep interpolated presentation frames out of conditioning/evaluation.

Do not enable controls by weakening qualification checks, declaring a fixture
restorable without evidence, or routing the selection to a generic task start.
