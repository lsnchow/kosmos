/**
 * FIG.4 — the console's event plane over the run matrix.
 *
 * A supplied asset, self-hosted like the others. The artwork carries its own
 * labels, which are the same three commitments the pillar lists beside it; the
 * alt text is built from those items in `content.ts` rather than transcribed,
 * so what a screen reader hears cannot drift from what the pillar claims.
 */
import { PILLARS } from "../content";

const CONSOLE_PILLAR = PILLARS.find((pillar) => pillar.number === "04");

const ALT =
  "The console's event plane above the run matrix, resolving to a labelled synthetic mode below it" +
  (CONSOLE_PILLAR ? `: ${CONSOLE_PILLAR.items.map((item) => item.title).join(", ")}.` : ".");

export function ConsoleMatrix() {
  return (
    <img
      className="pipe-figure"
      src="/figures/console-matrix.png"
      alt={ALT}
      width={1254}
      height={1254}
      loading="lazy"
      decoding="async"
    />
  );
}
