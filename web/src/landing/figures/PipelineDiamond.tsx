/**
 * FIG.1 — the four Chain steps, arranged as a cycle.
 *
 * This is a supplied asset rather than a drawing: the PNG in `public/figures`
 * is the artwork as given, not an SVG approximation of it. It is self-hosted,
 * which is the rule the rest of the project holds to — no CDN, and the page
 * renders identically with no network.
 *
 * `mix-blend-mode: screen` is doing real work in the stylesheet. The artwork is
 * white line work on a black field, and the page's field is the console's
 * blue-purple charcoal; screening drops the black to nothing so the diagram
 * sits on the page's own surface rather than inside a black rectangle.
 *
 * The alt text is built from `PIPELINE_STAGES` rather than written out beside
 * it, so the text a screen reader hears cannot drift from the labels the
 * artwork carries.
 */
import { PIPELINE_STAGES } from "../content";

const ALT = `The four Chain steps as a cycle, each on its own field around a shared run: ${PIPELINE_STAGES.map(
  (stage) => `${stage.label} (${stage.note})`,
).join(", ")}.`;

export function PipelineDiamond() {
  return (
    <img
      className="pipe-figure"
      src="/figures/pipeline-cycle.png"
      alt={ALT}
      width={1200}
      height={824}
      loading="lazy"
      decoding="async"
    />
  );
}
