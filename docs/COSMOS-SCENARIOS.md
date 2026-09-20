# Cosmos scenario presets

The console defaults to **Move bowl / pot · recorded trajectory**, seed 0.
The exact instruction remains `Put the pot to the left of the purple item.`
It uses the matching official Bridge scene/action fixture rather than the
drawer's manual action compilation. The second selectable scenario is the
existing drawer manual-action probe. A numeric seed lets users sample fresh
variations; every click still invokes Cosmos. Refresh resets the selection to
pot/seed 0 and preserves runs in history.

## Inputs and execution

- Source video SHA-256:
  `a86cfc81633b216891ca26dc58c72193a979c10ad72f123175fa8d61a67cdaec`
- Decoded starting PNG SHA-256:
  `65db577ed68303af3931781428da7d017337b3b7be8c2eaac4fa192b2cb9dd69`
- Matching 16×10-D action fixture SHA-256:
  `5c26b3cb84799812a70b534ad939551d2ac308fdc870ea0e66163bb52c9d61da`
- Pot preset: resolution tier 480, 30 steps, guidance 1, 5 FPS.
- Drawer preset retains tier 256. Both use the existing pinned Cosmos weights.
- MP4 export now preserves the Cosmos output dimensions instead of always
  shrinking them to the legacy IRASim size.
- No live VLA is invoked in either Cosmos preset.

The pot task does not reuse the drawer-only judge profile. It currently
finishes as a generation-only run with its video, summary and actual timings.
Drawer automatic assessment remains separately wired.

## Real verification

- Seed 0 run: `c643f94635a0455799612dc1b673e22d`, 3.6875 s Cosmos inference,
  36,531,743,232-byte peak GPU allocation.
- Seed 1 run: `357243a997004b7ab413f1c8491a2f7e`, 2.8199 s inference. Its
  generated image hashes differ from seed 0.
- Both completed 16 future frames. Browser scenario switching, seed input,
  fresh refresh defaults and mobile layout passed. Build/typecheck passed.
- Visual inspection of seed 0's middle/final PNGs shows the gripper reaching
  the metal bowl and moving it, while the surrounding scene stays coherent.
  This is a useful generated-motion example, not a calibrated success label.
- Evidence: `data/private/semantic-judge/pot-scenario-proof/`.

## Recommended and unsuitable demo setups

- Best existing visual-motion example: pot fixture, matching recorded actions,
  seed 0. Do not claim this proves full task completion or policy quality.
- Observed poor case: IRASim 16-step RGB feedback on the drawer scene suffered
  severe visible distortion from repeated predictions/re-encoding.
- Weak success demo: Cosmos drawer + fixed right action produced coherent
  footage but did not demonstrate closing the drawer in the inspected run.
- Fabric folding: no verified matching scene/action setup has been prepared
  here. It is untested, not a measured failure or a ready demo preset.

## Worker

Replaced idle job 942950 with bounded job **942991**, H100 on `trig0022`.
End time: `2026-09-20T03:42:30` Toronto. Source release:
`78f82e546e99f5a559e50d5578e4e05d6495132f4a612573b5f442b551fdbd15`.
The existing localhost 8920 → trig0022:8919 private tunnel remains valid.
