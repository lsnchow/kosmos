/**
 * FIG.2 — success rates with the interval drawn around them.
 *
 * This is the page's whole argument rendered once: a point estimate is a dot, a
 * measurement is a dot with a bar through it, and the two published cells we can
 * cite disagree by more than either interval is wide. The gap between them is
 * drawn as a span rather than left for the reader to subtract, because that
 * eighty-eight points *is* the finding.
 *
 * The four headline reliability numbers are not plotted, because we have not
 * measured them. The panel says so in words rather than leaving an empty axis
 * that a reader could mistake for a zero.
 *
 * The SVG is rendered at its natural size (`max-width` matches the viewBox) so
 * that 10px type is 10px on screen. Letting it stretch to the panel width
 * scaled every glyph and hairline with it.
 */
import { CALLED_SHOT, FOUR_NUMBERS, OTHER_GAPS, PENDING } from "../content";
import { stagger, useProgress } from "./useProgress";

/**
 * Wilson score interval — the same estimator the backend reports with, so the
 * bar a visitor sees here is the bar the console would draw for the same cell.
 * z = 1.96 for a 95% interval.
 */
function wilson(successes: number, trials: number, z = 1.96) {
  if (trials === 0) return { low: 0, high: 1 };
  const p = successes / trials;
  const denominator = 1 + (z * z) / trials;
  const centre = p + (z * z) / (2 * trials);
  const spread = z * Math.sqrt((p * (1 - p)) / trials + (z * z) / (4 * trials * trials));
  return {
    low: Math.max(0, (centre - spread) / denominator),
    high: Math.min(1, (centre + spread) / denominator),
  };
}

const TRIALS = CALLED_SHOT.human.trials;

/** The two published cells the gap is measured between. */
const PRIMARY = [
  {
    key: "human",
    label: "Real arm",
    sub: "human-scored",
    successes: CALLED_SHOT.human.successes,
    tone: "accent" as const,
  },
  {
    key: "sim",
    label: "SIMPLER",
    sub: "physics sim",
    successes: CALLED_SHOT.simulator.successes,
    tone: "strong" as const,
  },
];

/**
 * The next-largest disagreements, so the headline cell is not cherry-picked.
 *
 * The label column is 116 units wide and the sub-label sets at 8.5px mono, which
 * is about 20 characters. These task strings run to 38, so they are cut to the
 * part that distinguishes them — both rows are the same policy on the same
 * object, and only the destination differs.
 */
const SUB_MAX = 20;

const SECONDARY = OTHER_GAPS.map((gap, index) => ({
  key: `other-${index}`,
  label: gap.policy,
  sub: gap.task.length > SUB_MAX ? `…${gap.task.slice(-(SUB_MAX - 1))}` : gap.task,
  successes: Math.round(gap.human * TRIALS),
  tone: "dim" as const,
}));

/* Geometry. One column of labels, one plot, one column of values. */
const W = 560;
const PLOT_L = 132;
const PLOT_R = 486;
const TICK_Y = 14;
const ROW_Y = [46, 96, 148, 178];
const GAP_Y = 71;
const RULE_Y = 126;
const H = 198;

const x = (rate: number) => PLOT_L + rate * (PLOT_R - PLOT_L);

function Row({
  row,
  y,
  progress,
  index,
  count,
}: {
  row: (typeof PRIMARY)[number] | (typeof SECONDARY)[number];
  y: number;
  progress: number;
  index: number;
  count: number;
}) {
  const local = stagger(progress, index, count, 0.5);
  const rate = row.successes / TRIALS;
  const { low, high } = wilson(row.successes, TRIALS);
  // The bar grows outward from the estimate, so the dot lands first and the
  // uncertainty arrives around it.
  const drawnLow = rate - (rate - low) * local;
  const drawnHigh = rate + (high - rate) * local;
  const big = row.tone !== "dim";

  return (
    <g data-tone={row.tone} opacity={local === 0 ? 0 : 1}>
      <text x={PLOT_L - 16} y={y + 1} className="rl-label" textAnchor="end">
        {row.label}
      </text>
      <text x={PLOT_L - 16} y={y + 13} className="rl-sub" textAnchor="end">
        {row.sub}
      </text>

      <line x1={x(drawnLow)} x2={x(drawnHigh)} y1={y} y2={y} className="rl-bar" />
      <line x1={x(drawnLow)} x2={x(drawnLow)} y1={y - 4} y2={y + 4} className="rl-cap" />
      <line x1={x(drawnHigh)} x2={x(drawnHigh)} y1={y - 4} y2={y + 4} className="rl-cap" />
      <circle cx={x(rate)} cy={y} r={big ? 4 : 3} className="rl-dot" />

      <text x={PLOT_R + 18} y={y + 4} className={big ? "rl-value rl-value-big" : "rl-value"}>
        {Math.round(rate * 100)}%
      </text>
    </g>
  );
}

export function ReliabilityFigure({ play }: { play: boolean }) {
  const progress = useProgress(play, 2000);
  // The span is the conclusion, so it arrives after both rows it connects.
  const gap = stagger(progress, 3, 4, 0.4);
  const rows = [...PRIMARY, ...SECONDARY];

  return (
    <div className="reliability-figure">
      <svg
        className="reliability-plot"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Published success rates with 95% Wilson score intervals. The real arm reads ${Math.round(
          CALLED_SHOT.human.rate * 100,
        )} per cent and the physics simulator ${Math.round(
          CALLED_SHOT.simulator.rate * 100,
        )} per cent on the same cell — a gap of ${CALLED_SHOT.gapPoints} points.`}
      >
        {/* Gridlines every 25 points, labelled at the quarters only. */}
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={TICK_Y + 6} y2={H - 8} className="rl-grid" />
            <text x={x(tick)} y={TICK_Y} className="rl-tick" textAnchor="middle">
              {Math.round(tick * 100)}
            </text>
          </g>
        ))}

        {rows.map((row, index) => (
          <Row key={row.key} row={row} y={ROW_Y[index]} progress={progress} index={index} count={rows.length} />
        ))}

        {/* The gap, drawn between the two published cells it separates. */}
        <g className="rl-gap" opacity={gap}>
          <line x1={x(CALLED_SHOT.simulator.rate)} x2={x(CALLED_SHOT.simulator.rate)} y1={ROW_Y[1] - 10} y2={GAP_Y} />
          <line x1={x(CALLED_SHOT.human.rate)} x2={x(CALLED_SHOT.human.rate)} y1={ROW_Y[0] + 10} y2={GAP_Y} />
          <line
            x1={x(CALLED_SHOT.simulator.rate)}
            x2={x(CALLED_SHOT.human.rate)}
            y1={GAP_Y}
            y2={GAP_Y}
            strokeDasharray="2 3"
          />
          <text
            x={(x(CALLED_SHOT.simulator.rate) + x(CALLED_SHOT.human.rate)) / 2}
            y={GAP_Y - 7}
            className="rl-gap-label"
            textAnchor="middle"
          >
            {CALLED_SHOT.gapPoints} points apart
          </text>
        </g>

        {/* Separates the cited headline cell from the corroborating rows. */}
        <line x1={0} x2={W} y1={RULE_Y} y2={RULE_Y} className="rl-rule" />
      </svg>

      <p className="rl-source font-mono">{CALLED_SHOT.source}</p>

      {/* The four numbers this figure cannot plot yet. Listed, not omitted. */}
      <div className="reliability-pending">
        <p className="rl-pending-head font-mono">The four, unplotted</p>
        <ul>
          {FOUR_NUMBERS.map((number) => (
            <li key={number.title}>
              <span className="text-fg-muted">{number.title}</span>
              <span className="font-mono text-fg-dim">{PENDING}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
