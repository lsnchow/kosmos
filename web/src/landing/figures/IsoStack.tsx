/**
 * The pillar graphic: four planes stacked in isometric projection.
 *
 * Each plane carries one of the pillar's commitments, and they are stacked in
 * the order they depend on each other — the topmost is the one a reader meets
 * first in the list beside it, the lowest is the property the other three exist
 * to produce. The dashed verticals tie the top plane's corners to the bottom's,
 * so the group reads as one object seen from above rather than four cards.
 *
 * Drawn rather than rendered: this is SVG on the page's own tokens, so it costs
 * no request, scales without an asset, and repaints with the palette.
 *
 * Deliberately static. The first version faded each plane in off a rAF driver,
 * which meant the graphic's *content* depended on an animation completing — and
 * a browser that throttles rAF (a background tab, a reduced-power mode) left
 * four planes at zero opacity and the labels invisible. The entrance now
 * belongs to the `Reveal` wrapping the whole cell; the planes themselves are
 * always drawn.
 */

/** Isometric: a plane twice as wide as tall, text along atan(0.5) ≈ 26.57°. */
const ISO_ANGLE = 26.57;

const W = 440;
const H = 560;
const CX = 220;
/** Plane half-width and half-height. The 2:1 ratio is what makes it read iso. */
const HW = 196;
const HH = 98;
/** Vertical distance between plane centres. */
const STEP = 104;
const TOP_Y = 118;

type Layer = { label: string; sub?: string };

function planePoints(cy: number) {
  return [
    `${CX},${cy - HH}`,
    `${CX + HW},${cy}`,
    `${CX},${cy + HH}`,
    `${CX - HW},${cy}`,
  ].join(" ");
}

export function IsoStack({ number, layers }: { number: string; layers: readonly Layer[] }) {
  const bottomY = TOP_Y + (layers.length - 1) * STEP;

  return (
    <svg
      className="iso-stack"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={layers.map((layer) => layer.label).join(", ")}
    >
      {/* The verticals that bind the stack into one solid. */}
      <g className="iso-guide">
        <line x1={CX - HW} x2={CX - HW} y1={TOP_Y} y2={bottomY} />
        <line x1={CX + HW} x2={CX + HW} y1={TOP_Y} y2={bottomY} />
      </g>

      {/*
        Painted bottom-up so a higher plane overlaps the one beneath it, which
        is what gives the stack its depth. The array is read top-down, so the
        index is flipped rather than the data reversed.
      */}
      {[...layers].reverse().map((layer, reversedIndex) => {
        const index = layers.length - 1 - reversedIndex;
        const cy = TOP_Y + index * STEP;
        const isTop = index === 0;

        return (
          <g
            key={layer.label}
            className="iso-plane"
            data-top={isTop ? "true" : "false"}
          >
            <polygon points={planePoints(cy)} />

            <g transform={`rotate(${ISO_ANGLE} ${CX} ${cy})`}>
              <text x={CX} y={cy} className="iso-label" textAnchor="middle">
                {layer.label}
              </text>
              {layer.sub && (
                <text x={CX} y={cy + 17} className="iso-sub" textAnchor="middle">
                  {layer.sub}
                </text>
              )}
            </g>

            {/* The pillar number sits on the top plane, and the marker with it. */}
            {isTop && (
              <>
                <text x={CX} y={cy - 26} className="iso-number" textAnchor="middle">
                  {number}
                </text>
                <rect x={CX + HW - 34} y={cy - 12} width={9} height={9} className="iso-marker" />
              </>
            )}
          </g>
        );
      })}
    </svg>
  );
}
