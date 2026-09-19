/**
 * One animation driver for all four figures.
 *
 * Every figure on this page animates the same way: a single scalar runs 0 → 1
 * once, and each element derives its own state from that scalar with a per-index
 * offset. One `requestAnimationFrame` loop and one state update per frame drives
 * thirty matrix cells or six error bars alike — the alternative, a timer per
 * element, is thirty times the work to show the same thing.
 *
 * When `play` is false the hook returns 1, not 0. That is the reduced-motion
 * contract for this page: a reader who has asked for no animation gets the
 * finished figure, because the figure *is* the evidence. Returning 0 would hide
 * the one thing the panel exists to show.
 */
import { useEffect, useState } from "react";

/** Cubic ease-out. The same curve `Reveal` uses, so entrances agree. */
function easeOut(t: number) {
  return 1 - Math.pow(1 - t, 3);
}

export function useProgress(play: boolean, durationMs = 1800) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!play) return;

    let frame = 0;
    let start: number | null = null;

    const step = (now: number) => {
      start ??= now;
      const elapsed = now - start;
      const linear = Math.min(1, elapsed / durationMs);
      setProgress(easeOut(linear));
      if (linear < 1) frame = requestAnimationFrame(step);
    };

    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [play, durationMs]);

  return play ? progress : 1;
}

/**
 * The share of `progress` belonging to item `index` of `count`, as its own
 * 0 → 1 ramp.
 *
 * `overlap` is how much of the sequence each item occupies: at 1 every item
 * spans the whole window and they move together; at 0.2 each gets a fifth of it
 * and the group reads as a sweep. The default leaves a clear leading edge while
 * still finishing every item by the end of the run.
 */
export function stagger(progress: number, index: number, count: number, overlap = 0.45) {
  if (count <= 1) return progress;
  const span = overlap;
  const start = (index / (count - 1)) * (1 - span);
  return Math.max(0, Math.min(1, (progress - start) / span));
}
