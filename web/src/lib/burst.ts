import type { CreateRunBody } from "./api";
import { canonicalJson, stableHashHex } from "./format";

export type BurstIdentity = {
  mode: "synthetic";
  backend: "synthetic";
  policies: string[];
  tasks: string[];
  starts_per_task: number;
  seed: number;
};

/**
 * The burst identity is everything that decides *which run this is*. Policies
 * and tasks are sorted so that a different iteration order cannot produce a
 * different key for the same run.
 */
export function burstIdentity(input: {
  policies: string[];
  tasks: string[];
  startsPerTask: number;
  seed: number;
}): BurstIdentity {
  return {
    mode: "synthetic",
    backend: "synthetic",
    policies: [...input.policies].sort(),
    tasks: [...input.tasks].sort(),
    starts_per_task: input.startsPerTask,
    seed: input.seed,
  };
}

/**
 * Derive the idempotency key from run identity.
 *
 * The previous build minted `crypto.randomUUID()` on every click. The ledger's
 * `UNIQUE` constraint on `idempotency_key` can only collapse a duplicate
 * submission if the duplicate arrives with the *same* key, so a fresh UUID
 * bypassed the protection entirely: two clicks on stage started two concurrent
 * 1,500-episode runs and doubled the cost.
 *
 * With a derived key the second POST matches the first row, the server returns
 * the existing run, and no second execution is scheduled.
 *
 * That alone was too strong. Every input to the identity is a constant, so the
 * key was constant *forever*: the pre-roll burst started before the pitch and
 * the burst pressed on stage derived the same key, the server returned the
 * already-finished pre-roll run, no tile ignited, and the forty-second beat was
 * dead air.
 *
 * `attempt` restores the distinction the key is actually meant to draw. It is
 * held while a run is alive, so a duplicate submission still collapses into the
 * run already going; it advances once that run is terminal, so a deliberate
 * second run gets a key of its own.
 */
export function burstIdempotencyKey(identity: BurstIdentity, attempt = 0): string {
  // `attempt` enters the hash but never the request body, so every attempt
  // submits an identical run configuration under a different key. That matters:
  // the ledger refuses a key it has already seen against a *different* config,
  // and accepts a new key for the same one.
  return `kosmos-burst-${stableHashHex(canonicalJson({ ...identity, attempt }))}`;
}

export function burstRequestBody(identity: BurstIdentity, attempt = 0): CreateRunBody {
  return { ...identity, idempotency_key: burstIdempotencyKey(identity, attempt) };
}
