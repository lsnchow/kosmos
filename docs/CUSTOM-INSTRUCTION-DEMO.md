# Custom instruction rollout and assessment

The current selector uses Cosmos text-conditioning demo profiles, with OpenVLA
selected by default. These names do not identify executed VLA checkpoints or
measured relative policy performance. The task remains in `session.prompt`;
the separate motion-style addition is persisted in
`session.source.conditioning_prompt`. The judge receives the task alone.

Custom saved clips resolve server-side to `demo_custom_instruction_v1`. A
deterministic task-specific rubric is persisted and hashed, then recomputed by
the private judge worker before evaluating the same 16 sampled frames. The
semantic epoch-02 LoRA, five samples, quorum, and raw records are retained.

The default is one 16-action chunk. The optional 32-action mode repeats the
recorded action sequence; it is not a newly planned continuation and can undo
earlier motion. Text conditioning does not guarantee a new destination is
reached with this supplied action sequence.

## Real verification, 2026-09-20

- Session: `9dce3f259fd94be78c08e1871b05c538`.
- Task: `Pick up the pot and put it on the green towel.`
- Judge: `judge-bff21950ac6d44619ce0f4e76455f298`.
- 16 generated action frames; automatic custom-instruction judging ran.
- Worker inference: 23.3647 seconds. Remote request: 24.0522 seconds.
- Result: abstained; two completion votes and three uncertain votes.
- The inspected final frame does not show successful placement on the towel.
- This verifies dispatch and persistence, not task success or judge calibration.
- Private worker allocation `943172`, one H100, bounded to 30 minutes;
  source revision `e841443`.
- Screenshot: `data/private/semantic-judge/custom-automatic-judge.png`.

Frontend typecheck/build and targeted request/rubric checks passed. Existing
historical clips and strict judge results remain intact.
