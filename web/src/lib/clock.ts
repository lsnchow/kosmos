/**
 * One shared display clock drives every accumulated-frame player on the page.
 *
 * Twelve tiles each running their own timer is twelve wakeups per frame and
 * twelve independent React renders; on a projector at 1 Hz telemetry updates
 * that is where frames get dropped. A single interval publishing a monotonic
 * tick lets each player derive its own frame index arithmetically.
 *
 * The clock counts *display* frames. It is not a control timestamp and carries
 * no claim about how fast the robot or the world model actually ran.
 */

export const DISPLAY_FPS = 8;

type Listener = () => void;

let tick = 0;
let handle: ReturnType<typeof setInterval> | undefined;
const listeners = new Set<Listener>();

function publish() {
  tick += 1;
  for (const listener of listeners) listener();
}

function start() {
  if (handle !== undefined || typeof setInterval !== "function") return;
  handle = setInterval(publish, Math.round(1000 / DISPLAY_FPS));
}

function stop() {
  if (handle === undefined) return;
  clearInterval(handle);
  handle = undefined;
}

export function subscribeDisplayClock(listener: Listener): () => void {
  listeners.add(listener);
  start();
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) stop();
  };
}

export function getDisplayTick(): number {
  return tick;
}

/** Test seam: advance the clock deterministically without waiting on timers. */
export function advanceDisplayClock(steps = 1): void {
  for (let index = 0; index < steps; index += 1) publish();
}

/** Test seam: reset module state between tests. */
export function resetDisplayClock(): void {
  tick = 0;
  listeners.clear();
  stop();
}
