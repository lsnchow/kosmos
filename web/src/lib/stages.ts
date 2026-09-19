/**
 * The four-stage Chain, derived from records the console already holds.
 *
 * PLUMB's pipeline is four Baseten Chainlets with four hardware profiles —
 * policy, world model, validity gate, judge — and until now that structure had
 * no representation on screen at all. This module turns the events the run
 * stream already delivers into a per-stage state, and it invents nothing:
 *
 *   - the **world** stage is read from `segment_completed` events, via the wall
 *     state those events accumulate into;
 *   - the other three are read from the persisted episode records that arrive in
 *     the same stream's `snapshot` payload.
 *
 * A stage whose field is absent is `unreported`, and its detail line says "not
 * reported". It is never `pending` (which would imply the work is still coming)
 * and never silently `reported` from an adjacent signal. In particular, frames
 * arriving does not prove a policy emitted an action horizon, and a judge score
 * is not inferred from an episode merely reaching a terminal status.
 */
import type { Episode } from "./api";
import { pickNumber, pickString } from "./format";
import type { WallState } from "./wall";

export type StageKey = "policy" | "world" | "validity" | "judge";

/**
 * `pending`   nothing for this stage has arrived yet
 * `active`    evidence is arriving and the run has not reached a terminal status
 * `reported`  the stage reported, and the run is finished
 * `unreported` records exist but this stage's field is absent from all of them
 * `failed`    the records carry explicit failure evidence for this stage
 */
export type StageState = "pending" | "active" | "reported" | "unreported" | "failed";

export type Stage = {
  key: StageKey;
  label: string;
  /** What this Chainlet is, in one clause. Static copy, not data. */
  role: string;
  state: StageState;
  /** The counts that were actually reported. */
  detail: string;
  /** Which record produced `detail`. */
  source: string;
};

export const STAGE_ORDER: StageKey[] = ["policy", "world", "validity", "judge"];

const VALIDITY_VALUES = ["valid", "invalid", "unknown"] as const;

function countValidity(episodes: Episode[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const episode of episodes) {
    const value = pickString(episode.validity);
    if (value === undefined) continue;
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return counts;
}

function plural(count: number, one: string, many = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`;
}

export function stageLadder({
  wall,
  episodes,
  runTerminal,
}: {
  wall: WallState;
  episodes: Episode[];
  runTerminal?: boolean;
}): Stage[] {
  const assigned = wall.slots.filter((slot) => slot !== undefined);
  const segmentCount = assigned.reduce((sum, slot) => sum + (slot?.segmentCount ?? 0), 0);
  const frameCount = assigned.reduce((sum, slot) => sum + (slot?.frames.length ?? 0), 0);
  const certified = assigned.reduce<number | undefined>((sum, slot) => {
    if (slot?.certifiedFrameCount === undefined) return sum;
    return (sum ?? 0) + slot.certifiedFrameCount;
  }, undefined);
  const uncertifiedSegments = assigned.reduce(
    (sum, slot) => sum + (slot?.segmentsMissingCertifiedCount ?? 0),
    0,
  );

  // `active` only ever narrows `reported`; it never manufactures evidence.
  const settled = (): StageState => (runTerminal ? "reported" : "active");

  const episodeCount = episodes.length;
  const horizons = episodes.flatMap((episode) => {
    const value = pickNumber(episode.horizon_actions);
    return value === undefined ? [] : [value];
  });

  const policy: Stage = {
    key: "policy",
    label: "Policy",
    role: "Policy endpoint · proposes an action chunk from one observation",
    state: "pending",
    detail: "No episode record for this run yet.",
    source: "Application ledger · episode records",
  };
  if (episodeCount > 0) {
    if (horizons.length === 0) {
      policy.state = "unreported";
      policy.detail = `${plural(episodeCount, "episode record")} · action horizon not reported`;
    } else {
      policy.state = settled();
      policy.detail = `${horizons.length} of ${episodeCount} records report an action horizon · longest ${Math.max(
        ...horizons,
      )} actions`;
    }
  }

  const world: Stage = {
    key: "world",
    label: "World model",
    role: "World Chainlet · turns a compiled action chunk into frames",
    state: "pending",
    detail: "No segment_completed event has arrived yet.",
    source: "Run event stream · segment_completed",
  };
  if (segmentCount > 0) {
    world.state = settled();
    // The event count and the frame count have different subjects and are kept
    // apart on purpose: a reloaded episode row contributes accumulated frames
    // without a segment event behind it, so folding the two into one number
    // would attribute frames to events that never arrived.
    const framePart =
      certified === undefined
        ? `${frameCount} frames accumulated · certified count not reported`
        : `${certified} certified frames accumulated`;
    const uncertifiedPart =
      certified !== undefined && uncertifiedSegments > 0
        ? ` · ${uncertifiedSegments} without a certified count`
        : "";
    world.detail = `${plural(segmentCount, "segment event")} · ${framePart}${uncertifiedPart}`;
  }

  const validityCounts = countValidity(episodes);
  const validityTotal = Object.values(validityCounts).reduce((sum, count) => sum + count, 0);
  const validity: Stage = {
    key: "validity",
    label: "Validity gate",
    role: "Deterministic Python, not a model call · invalid rollouts are excluded and counted",
    state: "pending",
    detail: "No episode record for this run yet.",
    source: "Application ledger · episode validity field",
  };
  if (episodeCount > 0) {
    if (validityTotal === 0) {
      validity.state = "unreported";
      validity.detail = `${plural(episodeCount, "episode record")} · validity not reported`;
    } else {
      validity.state = settled();
      const known = VALIDITY_VALUES.filter((value) => validityCounts[value] !== undefined).map(
        (value) => `${validityCounts[value]} ${value}`,
      );
      const other = Object.keys(validityCounts)
        .filter((value) => !VALIDITY_VALUES.includes(value as (typeof VALIDITY_VALUES)[number]))
        .map((value) => `${validityCounts[value]} ${value}`);
      const unreported = episodeCount - validityTotal;
      validity.detail = [
        ...known,
        ...other,
        unreported > 0 ? `${unreported} not reported` : undefined,
      ]
        .filter((part): part is string => part !== undefined)
        .join(" · ");
    }
  }

  const scored = episodes.filter((episode) => typeof episode.binary_success === "boolean").length;
  const missingReasons = episodes.filter((episode) => pickString(episode.missing_reason)).length;
  const judge: Stage = {
    key: "judge",
    label: "Judge",
    role: "Rubric VLM · decomposed fields, label computed in Python",
    state: "pending",
    detail: "No episode record for this run yet.",
    source: "Application ledger · binary outcome and missing reason",
  };
  if (episodeCount > 0) {
    if (scored === 0) {
      judge.state = "unreported";
      judge.detail =
        missingReasons > 0
          ? `No binary outcome reported · ${missingReasons} record${missingReasons === 1 ? "" : "s"} carry a missing reason`
          : `${plural(episodeCount, "episode record")} · no binary outcome reported`;
    } else {
      judge.state = settled();
      judge.detail = `${scored} of ${episodeCount} records carry a binary outcome${
        missingReasons > 0 ? ` · ${missingReasons} with a missing reason` : ""
      }`;
    }
  }

  return [policy, world, validity, judge];
}
