/**
 * The first viewport.
 *
 * The masthead used to live here. It is `Nav` now, a sibling mounted by
 * `Landing`, because it has to outlast the hero: sticky inside this element it
 * scrolled away the moment the hero did. It is also hidden over the hero, which
 * is why it no longer belongs to it.
 *
 * Nothing here fetches, and nothing here is remote. Every string is a constant
 * in `content.ts` and the backdrop is a locally compiled shader, so the hero
 * renders identically with
 * no network — a landing page that goes blank when the API is the thing being
 * demoed has failed at the one moment it matters.
 */
import { CRTWarp } from "./CRTWarp";
import { PRODUCT } from "./content";

export function Hero() {
  return (
    <div className="hero">
      {/*
        The backdrop is a CRT phosphor field on the GPU, not a video.

        It carried a remote MP4 once, which is the thing this page must never do
        again: a landing page that goes blank when the network is the thing
        being demoed has failed at the one moment it matters. This is `three` as
        a dependency and a shader compiled locally, so it renders with no
        network. It freezes under `prefers-reduced-motion`, stops when scrolled
        past, and renders nothing at all if WebGL is unavailable.
      */}
      <CRTWarp className="hero-backdrop" />

      <div className="hero-body">
        <h1 className="hero-title">
          <span className="text-fg-strong">Compare robot AI models</span>{" "}
          <span className="text-fg-muted">without a robot.</span>
        </h1>

        <p className="hero-promise">{PRODUCT.promise}</p>
      </div>
    </div>
  );
}
