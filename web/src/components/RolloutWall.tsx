import { Glyph } from "./Terminal";
import { MetaList } from "./MetaList";
import { formatCount, formatFrameCaption} from "../lib/format";
import { cn } from "../lib/utils";
import {
  PROVENANCE_DESCRIPTIONS,
  PROVENANCE_LABELS,
  ignitionDelayMs,
  provenanceIsMixed,
  type TileSlot,
  type WallState,
} from "../lib/wall";
import { FramePlayer } from "./MediaFrame";
import { EmptyState, Note, Panel, SourceChip, StatusPill } from "./Primitives";

/**
 * Per-tile provenance. Without it a replayed clip and a live one are the same
 * pixels, which is the single easiest way for this demo to mislead somebody.
 */
/** Ids at or below this length are already readable and are shown whole. */
const SHORT_ID_MAX = 16;
/** How much of a hash is kept. Eight hex characters, as `git --oneline` does. */
const HASH_PREFIX = 8;

/**
 * An identifier abbreviated the way a shell abbreviates one: the semantic
 * prefix in full, then the first eight characters of the hash.
 * `episode-8d1f0deeb7434058bc1bc10b585f8b67` becomes `episode-8d1f0dee`.
 *
 * Short ids are returned untouched — abbreviating `ep-77` to `77` would throw
 * away the half that says what kind of thing it is, which is the opposite of
 * the point. Nothing is invented for an id that was never reported.
 */
function shortId(id: string | undefined): string {
  if (!id) return "episode id unreported";
  if (id.length <= SHORT_ID_MAX) return id;
  const cut = id.lastIndexOf("-");
  if (cut <= 0) return id.slice(0, HASH_PREFIX);
  return `${id.slice(0, cut + 1)}${id.slice(cut + 1, cut + 1 + HASH_PREFIX)}`;
}

export function ProvenanceBadge({ slot }: { slot: TileSlot }) {
  const mixed = provenanceIsMixed(slot);
  if (!slot.provenance) {
    return (
      <span
        className="provenance-badge provenance-unreported"
        title="No segment reported a provenance label, so this tile's source is unknown."
      >
        no provenance
      </span>
    );
  }
  const entries = Object.entries(slot.provenanceCounts)
    .map(([name, count]) => `${name}: ${count}`)
    .join(", ");
  return (
    <span
      className={cn("provenance-badge", `provenance-${slot.provenance}`)}
      title={mixed ? `Mixed sources — ${entries}` : PROVENANCE_DESCRIPTIONS[slot.provenance]}
    >
      {mixed ? `mixed · ${PROVENANCE_LABELS[slot.provenance]}` : PROVENANCE_LABELS[slot.provenance]}
    </span>
  );
}

function frameCountLabel(slot: TileSlot): string {
  // The `Frames` label carries the noun, so the value is the number and the one
  // caveat that changes its meaning. "uncertified" is not cosmetic: it marks a
  // count the server never certified, and it must not read as a certified one.
  if (slot.certifiedFrameCount === undefined) return `${formatCount(slot.frames.length)} · uncertified`;
  const extra = slot.segmentsMissingCertifiedCount;
  return extra > 0
    ? `${formatCount(slot.certifiedFrameCount)} · ${extra} uncertified`
    : formatCount(slot.certifiedFrameCount);
}

/**
 * Whether this rollout has a human score to be checked against.
 *
 * Required, not decorative. The project's claim is that it is explicit about
 * where it can and cannot be verified, and a tile from a typed prompt looks
 * exactly like a benchmark tile unless it says otherwise. `undefined` prints
 * nothing: a record that never reported the flag is not evidence either way.
 */
export function GroundTruthBadge({ benchmark }: { benchmark?: boolean }) {
  if (benchmark === undefined) return null;
  return benchmark ? (
    <p className="truth-badge truth-badge-benchmark">
      <Glyph name="badge" className="shrink-0" />
      ground truth available
    </p>
  ) : (
    <p className="truth-badge truth-badge-off">
      <Glyph name="slash" className="shrink-0" />
      off-benchmark — no human score to compare against
    </p>
  );
}

function RolloutTile({
  slot,
  index,
  onExpand,
  onDrive,
  outOfScope,
}: {
  slot?: TileSlot;
  index: number;
  onExpand?: (slot: TileSlot) => void;
  onDrive?: (slot: TileSlot) => void;
  outOfScope?: boolean;
}) {
  const label = `#${String(index + 1).padStart(2, "0")}`;

  if (!slot) {
    return (
      <article className="rollout-tile rollout-tile-idle" aria-label={`Viewport slot ${label}, unassigned`}>
        <FramePlayer
          frames={[]}
          alt={`Viewport slot ${label} has no assigned episode`}
          className="tile-media"
          emptyReason="No episode has claimed this slot"
        />
        <div className="tile-topline">
          <span>{label}</span>
          <StatusPill status="waiting">waiting</StatusPill>
        </div>
        <div className="tile-caption">
          <div>Unassigned slot</div>
          <span>Awaiting a persisted episode event</span>
        </div>
      </article>
    );
  }

  const igniting = slot.frames.length > 0;
  return (
    <article
      className={cn("rollout-tile", igniting && "rollout-tile-lit", outOfScope && "rollout-tile-unscoped")}
      // A tile claimed by a request that has not persisted a frame yet. It says
      // "generating" rather than showing an empty slot, because the rollout was
      // genuinely asked for -- the absence is latency, not a missing record.
      data-status={slot.frames.length === 0 && slot.status === "generating" ? "generating" : undefined}
      style={{ animationDelay: `${ignitionDelayMs(index)}ms` }}
      aria-label={`Viewport slot ${label}: ${slot.policy} on ${slot.task}${
        outOfScope ? ", outside the selected task scope" : ""
      }`}
    >
      <FramePlayer
        frames={slot.frames}
        alt={`Accumulated generated frames for ${slot.policy} on ${slot.task}`}
        className="tile-media"
        offset={index}
        emptyReason={
          slot.status === "generating"
            ? "Generating — the first chunk lands in about two seconds"
            : "No persisted segment event yet"
        }
      />
      <div className="tile-topline">
        <span>{label}</span>
        <div className="tile-topline-right">
          <ProvenanceBadge slot={slot} />
          {onExpand && (
            <button
              type="button"
              className="tile-expand"
              onClick={() => onExpand(slot)}
              aria-label={`Enlarge the persisted clip for ${slot.policy} on ${slot.task}`}
            >
              <Glyph name="expand" />
            </button>
          )}
          {onDrive && (
            <button
              type="button"
              className="tile-expand"
              onClick={() => onDrive(slot)}
              aria-label={`Open free-play control for ${slot.policy} on ${slot.task}`}
            >
              <Glyph name="drive" />
            </button>
          )}
        </div>
      </div>
      <div className="tile-caption">
        <div>{slot.policy}</div>
        {/* What was asked for, in the words it was asked in. A free-text task
            has no registry entry, so the task id would say nothing. */}
        <span>{slot.instruction ?? slot.task}</span>
      </div>
      <GroundTruthBadge benchmark={slot.benchmark} />
      <MetaList
        className="tile-meta"
        items={[
          { label: "Frames", value: frameCountLabel(slot), valueClassName: "tabular-nums" },
          { label: "Segments", value: formatCount(slot.segmentCount), valueClassName: "tabular-nums" },
        ]}
      />
      {/*
        * A short id, the way a shell prints one. This was the full 40-character
        * episode hash under each of twelve tiles — 480 characters of identifier
        * on one screen, none of it readable at that size and none of it
        * typed by anyone. The `title` still carries the full run and episode,
        * and the viewer prints them in full, so nothing is lost but the noise.
        */}
      <p className="tile-id truncate" title={`run ${slot.runId ?? "unreported"} · episode ${slot.episodeId ?? "unreported"}`}>
        {shortId(slot.episodeId)}
        {slot.episodeIds.length > 1 ? ` · ${slot.episodeIds.length} episodes` : ""}
      </p>
    </article>
  );
}

export function RolloutWall({
  wall,
  runId,
  onExpand,
  onDrive,
  scopedTask,
  scopedInstruction,
}: {
  wall: WallState;
  runId?: string;
  onExpand?: (slot: TileSlot) => void;
  onDrive?: (slot: TileSlot) => void;
  /** A task id from the frozen registry, chosen on the landing page. */
  scopedTask?: string;
  /** The verbatim instruction for `scopedTask`, for the disclosure line. */
  scopedInstruction?: string;
}) {
  const assigned = wall.slots.filter((slot): slot is TileSlot => slot !== undefined);
  const withFrames = assigned.filter((slot) => slot.frames.length > 0);
  const totalFrames = assigned.reduce((sum, slot) => sum + slot.frames.length, 0);
  const inScope = scopedTask ? assigned.filter((slot) => slot.task === scopedTask).length : undefined;

  return (
    <Panel
      title="Rollout viewport"
      className="rollouts-panel"
      action={<SourceChip>{runId ? `run ${runId}` : "no run selected"}</SourceChip>}
    >
      <div className="wall-summary" role="status">
        <span>
          <b className="tabular-nums">{formatCount(withFrames.length)}</b> of {wall.slots.length} slots have
          frames
        </span>
        <span>
          <b className="tabular-nums">{formatCount(totalFrames)}</b> accumulated frames
        </span>
        {wall.overflowKeys.length > 0 && (
          <span className="wall-overflow">
            <Glyph name="layers" />
            {formatCount(wall.overflowKeys.length)} further policy/task identities are running outside this
            viewport
          </span>
        )}
        {scopedTask !== undefined && (
          <span className="wall-scope">
            <Glyph name="target" />
            Scoped to {scopedInstruction ? `“${scopedInstruction}”` : scopedTask} ·{" "}
            <b className="tabular-nums">{formatCount(inScope)}</b> matching slots, the rest dimmed rather
            than dropped
          </span>
        )}
      </div>
      {assigned.length === 0 ? (
        <EmptyState className="wall-empty" icon={<Glyph name="layers" />}>
          No episode events have arrived for this run, so no slot has an identity yet. Tiles extend only when
          the application persists a segment; nothing here is pre-rendered to fill the grid.
        </EmptyState>
      ) : (
        <div className="rollout-grid">
          {wall.slots.map((slot, index) => (
            <RolloutTile
              key={slot?.key ?? `slot-${index}`}
              slot={slot}
              index={index}
              onExpand={onExpand}
              onDrive={onDrive}
              outOfScope={scopedTask !== undefined && slot !== undefined && slot.task !== scopedTask}
            />
          ))}
        </div>
      )}
      <Note summary="What the twelve tiles are, and are not">
        The tile count is a viewport choice, not the total robot or policy count — twelve tiles do not mean
        twelve robots. Each slot is keyed by policy and task and accumulates frames from persisted
        <code> segment_completed</code> events; frame counts are the counts the server certified, never a
        hardcoded chunk size. Tile ignition is staggered {ignitionDelayMs(1)} ms per slot as presentation
        timing only. Playback speed is a display choice and is not the control timing of the rollout. Cached,
        replayed and qualitative media keep their own badge.
      </Note>
    </Panel>
  );
}
