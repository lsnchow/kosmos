/**
 * FIG.3 — artifacts converging on one addressed cell.
 *
 * A supplied asset, self-hosted for the same reason as FIG.1: no CDN, and the
 * page renders identically with no network. `mix-blend-mode: screen` in the
 * stylesheet drops the artwork's black field so the diagram sits on the page's
 * own surface rather than inside a black rectangle.
 *
 * The alt text describes what the diagram asserts — several artifact kinds
 * resolving to a single addressed record — rather than listing its shapes,
 * because the claim is what a reader who cannot see it needs.
 */
const ALT =
  "Several artifact kinds — a document, a chart, an image — converging through one gate onto a single " +
  "addressed cell in a grid, which resolves downward to one stored record.";

export function ProvenanceFunnel() {
  return (
    <img
      className="pipe-figure"
      src="/figures/provenance-funnel.png"
      alt={ALT}
      width={1364}
      height={1153}
      loading="lazy"
      decoding="async"
    />
  );
}
