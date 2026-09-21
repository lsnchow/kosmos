import { describe, expect, it } from "vitest";
import { burstIdempotencyKey, burstIdentity, burstRequestBody } from "./burst";

const identity = burstIdentity({
  policies: ["OpenVLA", "Octo"],
  tasks: ["close_drawer", "fold_cloth"],
  startsPerTask: 50,
  seed: 20260919,
});

describe("burst idempotency key", () => {
  it("is stable for one attempt, so a duplicate submission collapses", () => {
    // The protection that matters on stage: two clicks racing past the
    // in-flight guard must reach the ledger as the same key.
    expect(burstIdempotencyKey(identity, 3)).toBe(burstIdempotencyKey(identity, 3));
  });

  it("does not depend on the order policies and tasks arrive in", () => {
    const reversed = burstIdentity({
      policies: ["Octo", "OpenVLA"],
      tasks: ["fold_cloth", "close_drawer"],
      startsPerTask: 50,
      seed: 20260919,
    });
    expect(burstIdempotencyKey(reversed, 0)).toBe(burstIdempotencyKey(identity, 0));
  });

  it("changes between attempts, so a deliberate re-run is a new run", () => {
    // Without this the key was constant forever: every input to the identity is
    // a constant, so the pre-roll burst started before the pitch and the burst
    // pressed on stage derived the same key. The server would return the
    // finished pre-roll run, no tile would ignite, and the beat would be silent.
    const keys = [0, 1, 2].map((attempt) => burstIdempotencyKey(identity, attempt));
    expect(new Set(keys).size).toBe(3);
  });

  it("keeps the attempt out of the request body", () => {
    // The ledger refuses a key it has seen against a *different* configuration.
    // Two attempts must therefore differ in key and in nothing else.
    const first = burstRequestBody(identity, 0);
    const second = burstRequestBody(identity, 1);
    expect(first.idempotency_key).not.toBe(second.idempotency_key);

    const { idempotency_key: _a, ...firstConfig } = first;
    const { idempotency_key: _b, ...secondConfig } = second;
    expect(firstConfig).toEqual(secondConfig);
    expect(Object.keys(first)).not.toContain("attempt");
  });

  it("keeps the shape the panel prints and the server accepts", () => {
    expect(burstIdempotencyKey(identity, 7)).toMatch(/^kosmos-burst-[0-9a-f]{8}$/);
  });
});
