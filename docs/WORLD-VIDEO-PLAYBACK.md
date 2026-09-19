# Recorded world-model video playback

Open http://127.0.0.1:8787/live#world-model-videos.
The **Rollout viewport → World-model videos** tab is the default view.
**Play all** starts at the first recording and advances through the catalog;
Pause, Previous, Next, individual selection, MP4 download and raw-report links
are available. **Run frames** preserves the existing episode/synthetic grid.

The catalog currently contains **37 existing generated MP4s**, totaling **63.8
seconds** at their encoded playback rates. They include Cosmos3 output,
OpenVLA→IRASim closed-loop diagnostics, native-horizon references, state/history
replays and two-frame intervention probes. Repeats remain separately labelled.
These are not 37 newly submitted jobs or one continuous 64-second rollout.

`GET /api/world-videos` includes only output artifact fields from recognized
diagnostic reports. It verifies SHA-256 against the report, refuses symlink or
path-escape media, and never treats source videos in provenance as generated
outputs. Optional ffprobe metadata reports actual encoded duration/dimensions.
No ledger rows, success scores, qualification gates or GPU jobs are created.

Downloads copy:

- `/Users/lucas/Downloads/Kosmos-Cosmos3-generated-937575.mp4`
- Adjacent `.md` records provenance and limitations.
- Original, unchanged H.264 MP4: 640×480,17 frames at5fps,3.4seconds.
- SHA-256: `3b7e719ae870872e20b71b427474fddd0072449c0ac1d9a86471c71e0c93b603`.

Verification: all37 MP4s decoded with FFmpeg and were played end-to-end in
Chromium's real native player, automatically advancing through the complete
queue. No JavaScript errors or POST/model requests occurred; the 390px mobile
view had no horizontal overflow. All70 unique MP4/report links returned206 for
byte-range reads. Tests:1360 Python passed/6skipped. After merging the latest
landing-page changes,256 frontend tests passed and the production build passed.

Recordings retain “recorded model output / not live / unqualified” labels.
Some have visible distortion. Their presence is not proof of physical fidelity,
task success, a calibrated judge or a completed scientific study.
