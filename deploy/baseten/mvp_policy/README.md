# PLUMB SuSIE_LL static-goal Truss MVP

This package deploys one stateless public `patreya/gcbc-bridge` action model.
It is explicitly unqualified: no world model, rollout, success label, judge,
or benchmark result is exposed. Operators deploy it through Truss; this repo
does not perform Baseten writes.

Request JSON has exactly `schema_version`, `request_id`, `current_png_base64`,
`goal_png_base64`, and `prompt`. Both inputs must be distinct RGB 256×256 PNGs.
The model downloads only the pinned checkpoint, README, and commit marker from
Hugging Face in the cloud, validates hashes, and restores parameters only.
