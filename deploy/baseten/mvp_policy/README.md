# PLUMB SuSIE_LL static-goal Truss MVP

This package deploys one stateless public `patreya/gcbc-bridge` action model.
It is explicitly unqualified: no world model, rollout, success label, judge,
or benchmark result is exposed. Deployment is explicit; merely starting the
local API does not provision Baseten resources. See
[`docs/BASETEN-MVP.md`](../../../docs/BASETEN-MVP.md) for the verified deployment,
localhost workflow, evidence and lifecycle controls.

Request JSON has exactly `schema_version`, `request_id`, `current_png_base64`,
`goal_png_base64`, and `prompt`. Both inputs must be distinct RGB 256×256 PNGs.
The isolated ML child downloads only the pinned checkpoint, README, and commit
marker from Hugging Face in the cloud, validates hashes, and restores parameters
only. Its framework/protobuf/CUDA dependencies cannot be overridden by the
Truss HTTP process. The fixed child communicates through bounded JSON frames;
timeouts/protocol failures never replay a request automatically.
