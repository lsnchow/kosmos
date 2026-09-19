import { useSyncExternalStore } from "react";
import { getDisplayTick, subscribeDisplayClock } from "../lib/clock";

/**
 * Index into an accumulated frame list, driven by the shared display clock.
 *
 * `offset` staggers players so twelve tiles do not march in lockstep. When the
 * frame list grows mid-playback the modulo simply widens; nothing resets, so a
 * tile visibly gets longer as segments land instead of restarting.
 */
export function useFlipbook(frameCount: number, options?: { offset?: number; paused?: boolean }) {
  const tick = useSyncExternalStore(subscribeDisplayClock, getDisplayTick, () => 0);
  if (frameCount <= 0) return 0;
  if (options?.paused) return frameCount - 1;
  const offset = options?.offset ?? 0;
  return (tick + offset) % frameCount;
}
