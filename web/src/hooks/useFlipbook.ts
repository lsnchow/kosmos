import { useSyncExternalStore } from "react";
import { getDisplayTick, subscribeDisplayClock } from "../lib/clock";
import { prefersReducedMotion } from "../lib/motion";

/**
 * Index into an accumulated frame list, driven by the shared display clock.
 *
 * `offset` staggers players so twelve tiles do not march in lockstep. When the
 * frame list grows mid-playback the modulo simply widens; nothing resets, so a
 * tile visibly gets longer as segments land instead of restarting.
 *
 * Under `prefers-reduced-motion` the player holds its last frame rather than
 * playing. That is the same thing `paused` already does, and it is the right
 * reading of the request: the evidence is the frames, and the newest one is
 * still shown — what stops is twelve tiles flickering at 8 fps.
 */
export function useFlipbook(frameCount: number, options?: { offset?: number; paused?: boolean }) {
  const tick = useSyncExternalStore(subscribeDisplayClock, getDisplayTick, () => 0);
  if (frameCount <= 0) return 0;
  if (options?.paused || prefersReducedMotion()) return frameCount - 1;
  const offset = options?.offset ?? 0;
  return (tick + offset) % frameCount;
}
