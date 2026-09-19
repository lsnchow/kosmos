/**
 * The 12-tile wall is a *viewport*, not a census.
 *
 * Slots are stable and keyed by `(policy, task)`. Once an identity claims a
 * slot it keeps it for the lifetime of the run, and every `segment_completed`
 * event appends its frames to that slot. This is deliberately not a rolling
 * "last 12 episodes" window: a rolling window makes tiles flicker between
 * unrelated identities and can never show a clip growing.
 *
 * Nothing here invents a frame count. `certifiedFrameCount` is the sum of the
 * counts the server actually certified; segments that omit one are counted
 * separately so the UI can say so instead of guessing 16 per chunk.
 */
import {
  artifactUrl,
  artifactUrls,
  readProvenance,
  type Episode,
  type Provenance,
  type SegmentCompletedEvent,
} from "./api";
import { pickNumber, pickString, pickBoolean} from "./format";

export const WALL_SLOT_COUNT = 12;

/**
 * Presentation timing only: each slot's first frame appears 50 ms after the
 * previous slot's, inside the spec's 40–60 ms band. It changes nothing about
 * when the segment was generated or persisted.
 */
export const IGNITION_STEP_MS = 50;

export function ignitionDelayMs(slot: number): number {
  return slot * IGNITION_STEP_MS;
}

export type TileSlot = {
  slot: number;
  key: string;
  policy: string;
  task: string;
  runId?: string;
  episodeId?: string;
  /** Distinct episode ids that have contributed frames to this slot. */
  episodeIds: string[];
  status?: string;
  resolution?: string;
  /** What was actually asked for. A free-text task has no registry entry. */
  instruction?: string;
  /** Whether a published human score exists for this task. */
  benchmark?: boolean;
  /** Accumulated frame URLs, in arrival order, already resolved to fetchable URLs. */
  frames: string[];
  /** Sum of server-certified frame counts. Undefined when no segment reported one. */
  certifiedFrameCount?: number;
  /** Segments that arrived without a certified count, so the UI can disclose it. */
  segmentsMissingCertifiedCount: number;
  segmentCount: number;
  /** Most recently reported provenance for this slot. */
  provenance?: Provenance;
  /** Every provenance seen, so a mixed-source tile can be labelled mixed. */
  provenanceCounts: Partial<Record<Provenance, number>>;
  /** Dedupe set of `${episode_id}#${segment_index}`; replays must not double-count. */
  seenSegments: string[];
};

export type WallState = {
  slots: (TileSlot | undefined)[];
  byKey: Record<string, number>;
  byEpisode: Record<string, number>;
  /**
   * Identities observed after all 12 slots were claimed. The count proves the
   * wall is a viewport rather than the total number of policies or robots.
   */
  overflowKeys: string[];
};

export function identityKey(policy: unknown, task: unknown): string {
  return `${String(policy ?? "unknown-policy")}::${String(task ?? "unknown-task")}`;
}

export function createWall(size: number = WALL_SLOT_COUNT): WallState {
  return {
    slots: Array.from({ length: size }, () => undefined),
    byKey: {},
    byEpisode: {},
    overflowKeys: [],
  };
}

function cloneWall(state: WallState): WallState {
  return {
    slots: [...state.slots],
    byKey: { ...state.byKey },
    byEpisode: { ...state.byEpisode },
    overflowKeys: [...state.overflowKeys],
  };
}

/**
 * Claim the lowest free slot for an identity, or return the slot it already
 * holds. Returns `undefined` when the wall is full, recording the overflow.
 */
function claimSlot(
  next: WallState,
  key: string,
  policy: string,
  task: string,
): number | undefined {
  const existing = next.byKey[key];
  if (existing !== undefined) return existing;
  const free = next.slots.findIndex((slot) => slot === undefined);
  if (free < 0) {
    if (!next.overflowKeys.includes(key)) next.overflowKeys.push(key);
    return undefined;
  }
  next.byKey[key] = free;
  next.slots[free] = {
    slot: free,
    key,
    policy,
    task,
    episodeIds: [],
    frames: [],
    segmentsMissingCertifiedCount: 0,
    segmentCount: 0,
    provenanceCounts: {},
    seenSegments: [],
  };
  return free;
}

/**
 * Seed slot identities from a snapshot's episode list. Episodes may also carry
 * already-persisted `frame_urls`, which are merged without duplicating frames
 * an SSE segment already delivered.
 */
export function applyEpisodes(state: WallState, episodes: Episode[]): WallState {
  if (episodes.length === 0) return state;
  const next = cloneWall(state);
  let changed = false;

  for (const episode of episodes) {
    const policy = pickString(episode.policy, episode.policy_variant) ?? "unknown-policy";
    const task = pickString(episode.task) ?? "unknown-task";
    const key = identityKey(policy, task);
    const slotIndex = claimSlot(next, key, policy, task);
    if (slotIndex === undefined) {
      changed = true;
      continue;
    }
    const current = next.slots[slotIndex];
    if (!current) continue;
    changed = true;
    const episodeId = pickString(episode.episode_id, episode.id);
    if (episodeId) next.byEpisode[episodeId] = slotIndex;

    const merged: TileSlot = {
      ...current,
      frames: [...current.frames],
      seenSegments: [...current.seenSegments],
      provenanceCounts: { ...current.provenanceCounts },
      runId: pickString(episode.run_id, current.runId),
      episodeId: episodeId ?? current.episodeId,
      status: pickString(episode.status, current.status),
      resolution: pickString(episode.resolution, current.resolution),
      instruction: pickString(episode.task_instruction, current.instruction),
      // `undefined` is not `false`: a record that never reported the flag must
      // not be shown as off-benchmark, which is a claim about the task.
      benchmark: pickBoolean(episode.benchmark_task) ?? current.benchmark,
      episodeIds: [...current.episodeIds],
    };
    if (episodeId && !merged.episodeIds.includes(episodeId)) merged.episodeIds.push(episodeId);

    // An episode row may carry its own persisted frames (for example after a
    // page reload mid-run). Treat the whole row as one synthetic segment id so
    // reloading cannot duplicate frames already appended from SSE.
    const rowSegmentId = episodeId ? `${episodeId}#row` : undefined;
    const rowFrames = artifactUrls(episode.frame_urls);
    const singleFrame = artifactUrl(pickString(episode.frame_url, episode.video_url, episode.artifact_path));
    const candidateFrames = rowFrames.length > 0 ? rowFrames : singleFrame ? [singleFrame] : [];
    if (rowSegmentId && candidateFrames.length > 0 && !merged.seenSegments.includes(rowSegmentId)) {
      const unseen = candidateFrames.filter((url) => !merged.frames.includes(url));
      if (unseen.length > 0) {
        merged.seenSegments.push(rowSegmentId);
        merged.frames.push(...unseen);
        // The server reported how many segments produced these frames. Reading
        // it here keeps a polled row consistent with a streamed one, which
        // otherwise showed frames against a segment count of zero.
        const reported = pickNumber(episode.n_segments);
        if (reported !== undefined) merged.segmentCount = Math.max(merged.segmentCount, reported);
        const certified = pickNumber(episode.certified_frame_count);
        if (certified === undefined) {
          merged.segmentsMissingCertifiedCount += 1;
        } else {
          merged.certifiedFrameCount = (merged.certifiedFrameCount ?? 0) + certified;
        }
        const provenance = readProvenance(episode.provenance);
        if (provenance) {
          merged.provenance = provenance;
          merged.provenanceCounts[provenance] = (merged.provenanceCounts[provenance] ?? 0) + 1;
        }
      }
    }
    next.slots[slotIndex] = merged;
  }

  return changed ? next : state;
}

/**
 * Claim a slot for a rollout that has been requested but has not yet persisted
 * anything.
 *
 * Without this the tile would not appear until its first segment landed, which
 * for a five-chunk rollout is seconds of a stage beat with nothing on screen.
 * The slot is real, keyed the same way as every other, and carries no frames
 * and no counts -- it claims a position, not a result. `applyEpisodes` merges
 * into it by the same `(policy, task)` key when the records arrive.
 */
export function reserveSlot(
  state: WallState,
  claim: { policy: string; task: string; instruction?: string; benchmark?: boolean; runId?: string },
): WallState {
  const key = identityKey(claim.policy, claim.task);
  if (state.byKey[key] !== undefined) return state;

  // The viewport is twelve slots and is usually full, so a requested rollout
  // takes the front and pushes the rest back. It is the one the reader just
  // asked for; making them hunt for it, or silently dropping it into the
  // overflow count, would both be wrong. Whatever falls off the end is recorded
  // as overflow exactly as it would have been otherwise.
  const next = cloneWall(state);
  const reserved: TileSlot = {
    slot: 0,
    key,
    policy: claim.policy,
    task: claim.task,
    runId: claim.runId,
    instruction: claim.instruction,
    benchmark: claim.benchmark,
    status: "generating",
    episodeIds: [],
    frames: [],
    segmentsMissingCertifiedCount: 0,
    segmentCount: 0,
    provenanceCounts: {},
    seenSegments: [],
  };
  const kept = next.slots.filter((slot): slot is TileSlot => slot !== undefined);
  const displaced = kept.length >= next.slots.length ? kept[kept.length - 1] : undefined;
  const ordered = [reserved, ...kept].slice(0, next.slots.length);

  next.slots = next.slots.map((_, index) => ordered[index]);
  next.byKey = {};
  next.byEpisode = {};
  ordered.forEach((slot, index) => {
    if (!slot) return;
    slot.slot = index;
    next.byKey[slot.key] = index;
    for (const episodeId of slot.episodeIds) next.byEpisode[episodeId] = index;
  });
  if (displaced && !next.overflowKeys.includes(displaced.key)) next.overflowKeys.push(displaced.key);
  return next;
}

/**
 * Re-key a reserved slot onto the task id the server actually assigned.
 *
 * The optimistic tile is claimed before the request is sent, so its task id is
 * the browser's guess. The server derives its own from a SHA-256 of the prompt,
 * which a browser cannot compute synchronously — so the two ids differ, and
 * without this the arriving episodes would claim a *second* slot (or land in
 * overflow on a full wall) and the rollout would appear twice or not at all.
 */
export function adoptTaskId(state: WallState, policy: string, fromTask: string, toTask: string): WallState {
  if (fromTask === toTask) return state;
  const fromKey = identityKey(policy, fromTask);
  const index = state.byKey[fromKey];
  if (index === undefined) return state;
  const current = state.slots[index];
  if (!current) return state;

  const next = cloneWall(state);
  const toKey = identityKey(policy, toTask);
  delete next.byKey[fromKey];
  next.byKey[toKey] = index;
  next.slots[index] = { ...current, key: toKey, task: toTask };
  return next;
}

/** Append one persisted segment's frames to its slot. Idempotent per segment id. */
export function applySegment(state: WallState, event: SegmentCompletedEvent): WallState {
  const episodeId = pickString(event.episode_id);
  const segmentIndex = pickNumber(event.segment_index);
  const frames = artifactUrls(event.frame_urls);
  const next = cloneWall(state);

  let slotIndex: number | undefined;
  const policy = pickString(event.policy);
  const task = pickString(event.task);
  if (policy || task) {
    slotIndex = claimSlot(next, identityKey(policy, task), policy ?? "unknown-policy", task ?? "unknown-task");
  } else if (episodeId !== undefined) {
    slotIndex = next.byEpisode[episodeId];
  }
  if (slotIndex === undefined) return next.overflowKeys.length === state.overflowKeys.length ? state : next;

  const current = next.slots[slotIndex];
  if (!current) return state;

  const segmentId =
    episodeId !== undefined && segmentIndex !== undefined
      ? `${episodeId}#${segmentIndex}`
      : episodeId !== undefined
        ? `${episodeId}#${current.segmentCount}`
        : undefined;
  if (segmentId !== undefined && current.seenSegments.includes(segmentId)) return state;

  const merged: TileSlot = {
    ...current,
    frames: [...current.frames, ...frames],
    seenSegments: segmentId === undefined ? [...current.seenSegments] : [...current.seenSegments, segmentId],
    provenanceCounts: { ...current.provenanceCounts },
    episodeIds: [...current.episodeIds],
    segmentCount: current.segmentCount + 1,
    runId: pickString(event.run_id, current.runId),
    status: pickString(event.status, current.status),
    resolution: pickString(event.resolution, current.resolution),
  };

  if (episodeId !== undefined) {
    next.byEpisode[episodeId] = slotIndex;
    merged.episodeId = episodeId;
    if (!merged.episodeIds.includes(episodeId)) merged.episodeIds.push(episodeId);
  }

  const certified = pickNumber(event.certified_frame_count);
  if (certified === undefined) {
    merged.segmentsMissingCertifiedCount = current.segmentsMissingCertifiedCount + 1;
  } else {
    merged.certifiedFrameCount = (current.certifiedFrameCount ?? 0) + certified;
  }

  const provenance = readProvenance(event.provenance);
  if (provenance) {
    merged.provenance = provenance;
    merged.provenanceCounts[provenance] = (merged.provenanceCounts[provenance] ?? 0) + 1;
  }

  next.slots[slotIndex] = merged;
  return next;
}

export function provenanceIsMixed(slot: TileSlot): boolean {
  return Object.keys(slot.provenanceCounts).length > 1;
}

export const PROVENANCE_LABELS: Record<Provenance, string> = {
  live: "live",
  cached: "cached",
  replayed: "replayed",
  qualitative: "qualitative",
};

export const PROVENANCE_DESCRIPTIONS: Record<Provenance, string> = {
  live: "Generated during this run.",
  cached: "Reused from an earlier generation; not generated during this run.",
  replayed: "Replayed from a persisted artifact; not generated during this run.",
  qualitative: "Illustrative only. Not part of any scored comparison.",
};
