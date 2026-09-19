import { Expand, Gamepad2, Layers, Target } from "lucide-react";
import { formatCount } from "../lib/format";
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
  if (slot.certifiedFrameCount !== undefined) {
    const suffix =
      slot.segmentsMissingCertifiedCount > 0
        ? ` (+${slot.segmentsMissingCertifiedCount} seg. uncertified)`
        : "";
    return `${formatCount(slot.certifiedFrameCount)} certified frames${suffix}`;
  }
  if (slot.frames.length > 0) {
    return `${formatCount(slot.frames.length)} frames · certified count not reported`;
  }
  return "no frames yet";
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
          <div className="truncate">Unassigned slot</div>
          <span className="truncate">Awaiting a persisted episode event</span>
        </div>
      </article>
    );
  }

  const igniting = slot.frames.length > 0;
  return (
    <article
      className={cn("rollout-tile", igniting && "rollout-tile-lit", outOfScope && "rollout-tile-unscoped")}
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
        emptyReason="No persisted segment event yet"
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
              <Expand aria-hidden="true" className="size-3.5" />
            </button>
          )}
          {onDrive && (
            <button
              type="button"
              className="tile-expand"
              onClick={() => onDrive(slot)}
              aria-label={`Open free-play control for ${slot.policy} on ${slot.task}`}
            >
              <Gamepad2 aria-hidden="true" className="size-3.5" />
            </button>
          )}
        </div>
      </div>
      <div className="tile-caption">
        <div className="truncate">{slot.policy}</div>
        <span className="truncate">{slot.task}</span>
      </div>
      <dl className="tile-meta">
        <div>
          <dt>Frames</dt>
          <dd className="tabular-nums">{frameCountLabel(slot)}</dd>
        </div>
        <div>
          <dt>Segments</dt>
          <dd className="tabular-nums">{formatCount(slot.segmentCount)}</dd>
        </div>
      </dl>
      <p className="tile-id truncate" title={`run ${slot.runId ?? "unreported"} · episode ${slot.episodeId ?? "unreported"}`}>
        {slot.runId ? `${slot.runId} · ` : ""}
        {slot.episodeId ?? "episode id unreported"}
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
  streamNote,
  scopedTask,
  scopedInstruction,
}: {
  wall: WallState;
  runId?: string;
  onExpand?: (slot: TileSlot) => void;
  onDrive?: (slot: TileSlot) => void;
  streamNote?: React.ReactNode;
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
      eyebrow="12 viewport slots · persisted segment events only"
      className="rollouts-panel"
      action={<SourceChip>{runId ? `run ${runId}` : "no run selected"}</SourceChip>}
    >
      {streamNote}
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
            <Layers aria-hidden="true" className="size-3.5" />
            {formatCount(wall.overflowKeys.length)} further policy/task identities are running outside this
            viewport
          </span>
        )}
        {scopedTask !== undefined && (
          <span className="wall-scope">
            <Target aria-hidden="true" className="size-3.5" />
            Scoped to {scopedInstruction ? `“${scopedInstruction}”` : scopedTask} ·{" "}
            <b className="tabular-nums">{formatCount(inScope)}</b> matching slots, the rest dimmed rather
            than dropped
          </span>
        )}
      </div>
      {assigned.length === 0 ? (
        <EmptyState className="wall-empty" icon={<Layers aria-hidden="true" className="size-5" />}>
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
      <Note>
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
