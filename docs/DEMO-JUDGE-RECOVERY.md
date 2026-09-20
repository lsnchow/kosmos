# Demo judge recovery

The scrolling six-video gallery is restored beneath the judge panel on
`/console`. Frontend build and Chromium reload checks passed.

The earlier adapter availability conclusion was incorrect: it checked CAC,
but the semantic pilot was trained on Trillium. BatchMode SSH to
`trillium-gpu` succeeded and the entire `epoch-02` directory was copied to
`data/private/semantic-judge/epoch-02` and staged with
`deploy/baseten/stage_demo_judge.py`. Its tree hash matches the recorded
`sha256:7a0b7796a0ac8b78fbb8ba8f4c4db01fe8d35c2edc3cb7a0fbeec9f4fb20523e`.
Both copies are gitignored. No training or GPU allocation was launched.

The original drawer conditioning image is now copied to
`data/live-integrated/demo-judge-fixtures/scene-reference.png`; the service
verifies it against the recording's `source_png_sha256`. The gallery poster
was a generated frame and is no longer used as the initial scene reference.
FFmpeg frame extraction now uses `-fps_mode passthrough`; the installed
version rejects `-vsync`. All 16 selected frames decoded successfully.

Current external blocker: both saved Baseten profiles (`plumb-api` and
`lucas.sn.chow@gmail.com`) return HTTP 403 PERMISSION_DENIED for `whoami` and
model listing. The configured policy deployment read also returns 403.
Baseten access must be restored before deployment and actual inference QA.
No real adapter inference, inference receipt, or result persistence proof has
yet been obtained. Do not describe the feature as end-to-end verified.

Once access returns, continue with the private deployment runbook in
`deploy/baseten/judge_demo/README.md`, verify replica/budget limits, connect
the localhost judge deployment arguments, then perform one browser assessment
and reload. Inspect the remaining worker/runtime integration during that real
call; only local compilation/staging and input extraction have been verified.
