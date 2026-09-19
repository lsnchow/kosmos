/**
 * One answer to "does this viewer want motion", for the motion that CSS cannot
 * reach.
 *
 * `@media (prefers-reduced-motion: reduce)` in `styles.css` and `landing.css`
 * covers every declarative animation. It cannot cover two things, because both
 * are driven from JavaScript: the landing's `requestAnimationFrame` crossfade
 * and the 8 fps flipbook that plays accumulated rollout frames. Those two read
 * this instead.
 *
 * Deliberately a plain function rather than a hook with a change listener. A
 * viewer flipping the OS setting mid-session is not a case worth a subscription
 * in every player on the page; the next mount picks it up.
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}
