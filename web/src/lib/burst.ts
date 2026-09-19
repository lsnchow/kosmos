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
 */
export function burstIdempotencyKey(identity: BurstIdentity): string {
  return `plumb-burst-${stableHashHex(canonicalJson(identity))}`;
}

export function burstRequestBody(identity: BurstIdentity): CreateRunBody {
  return { ...identity, idempotency_key: burstIdempotencyKey(identity) };
}
