/**
 * FIG.4 — the console's tile wall and its queue.
 *
 * Twelve rollout tiles accumulating frames, with the queue depth draining beside
 * them. Both are schematics of the console's live view rather than a recording
 * of one: the console draws the same twelve tiles from real event-stream frames,
 * and this figure is labelled illustrative wherever it appears.
 *
 * Twelve is not an arbitrary count. It is what the console shows, and past about
 * a dozen a viewer stops perceiving individual tiles at all.
 */
import { BURST_TARGET, CHAIN_STEPS } from "../content";
import { stagger, useProgress } from "./useProgress";

const TILES = 12;
/** Five chunks a rollout, sixteen frames a chunk. The tile fills in steps. */
const CHUNKS = 5;

export function ConsoleFigure({ play }: { play: boolean }) {
  const progress = useProgress(play, 2600);

  // Queue depth drains as tiles complete: starts at the full replica count and
  // falls to zero, which is the shape the telemetry strip actually shows.
  const depth = Math.max(0, Math.round(BURST_TARGET.replicas * (1 - progress)));

  return (
    <div className="console-figure">
      <div className="console-wall" aria-hidden="true">
        {Array.from({ length: TILES }, (_, index) => {
          const local = stagger(progress, index, TILES, 0.55);
          // Held flat between chunks: frames land in a burst when a chunk
          // completes rather than trickling in one at a time.
          const chunks = Math.floor(local * CHUNKS);
          return (
            <div key={index} className="console-tile" data-active={local > 0 ? "true" : "false"}>
              <span className="console-tile-fill" style={{ transform: `scaleX(${chunks / CHUNKS})` }} />
              <span className="console-tile-index font-mono">{String(index + 1).padStart(2, "0")}</span>
            </div>
          );
        })}
      </div>

      {/* The four Chain steps, lit in order as the packet travels them. */}
      <ol className="console-chain">
        {CHAIN_STEPS.map((step, index) => {
          const local = stagger(progress, index, CHAIN_STEPS.length, 0.5);
          return (
            <li key={step.id} data-on={local > 0.4 ? "true" : "false"}>
              <span className="console-chain-name">{step.name}</span>
              <span className="console-chain-hw font-mono">{step.hardware}</span>
            </li>
          );
        })}
      </ol>

      <dl className="console-readout">
        <div>
          <dt>Queue depth</dt>
          <dd className="text-fg-strong">{depth}</dd>
        </div>
        <div>
          <dt>Replicas</dt>
          <dd>{BURST_TARGET.replicas}</dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd className="text-fg-dim">{BURST_TARGET.status}</dd>
        </div>
      </dl>
      <p className="sr-only">
        A schematic of the console: twelve rollout tiles filling a chunk at a time while the queue
        drains across {BURST_TARGET.replicas} replicas.
      </p>
    </div>
  );
}
