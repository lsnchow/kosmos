import { formatCount } from "../lib/format";

/**
 * One 1 Hz telemetry sample. A value is `undefined` when the platform did not
 * report it — never 0. A zero would read as "scaled to nothing", which is a
 * different and much more interesting claim than "not reported".
 */
export type ReplicaSample = {
  /** Epoch milliseconds from the *server's* telemetry timestamp. */
  t: number;
  active?: number;
  desired?: number;
};

/**
 * Ring-buffer cap. 1 Hz for ~3 minutes is ~180 samples; 600 leaves headroom for
 * a long rehearsal without unbounded growth or a re-layout per sample.
 */
export const MAX_SAMPLES = 600;

export function pushSample(samples: ReplicaSample[], next: ReplicaSample): ReplicaSample[] {
  const last = samples[samples.length - 1];
  if (last && last.t === next.t) {
    // The stream repeats the same server timestamp between ticks; do not plot
    // the same instant twice, and do not invent motion by nudging the clock.
    const merged = [...samples];
    merged[merged.length - 1] = next;
    return merged;
  }
  const appended = [...samples, next];
  return appended.length > MAX_SAMPLES ? appended.slice(appended.length - MAX_SAMPLES) : appended;
}

const VIEW_WIDTH = 1200;
const VIEW_HEIGHT = 420;
const PAD_LEFT = 78;
const PAD_RIGHT = 132;
const PAD_TOP = 28;
const PAD_BOTTOM = 56;

/**
 * Series identity is carried by hue *and* dash *and* a direct end label, so it
 * never depends on color alone.
 *
 * The hues are the palette's own `--series-*` tokens rather than literals, so
 * they cannot drift away from the stylesheet the way the previous pair did.
 * Both clear 3:1 against the panel surface (lime 12.6:1, blue 9.9:1), which is
 * asserted in styles.contrast.test.ts.
 */
const SERIES = [
  { key: "active" as const, label: "Active", color: "var(--series-active)", dash: undefined },
  { key: "desired" as const, label: "Desired", color: "var(--series-desired)", dash: "10 8" },
];

function niceMax(values: number[]): number {
  const max = values.length > 0 ? Math.max(...values) : 0;
  if (max <= 4) return 4;
  const padded = max * 1.15;
  const magnitude = 10 ** Math.floor(Math.log10(padded));
  return Math.ceil(padded / magnitude) * magnitude;
}

type Point = { index: number; value: number };

/** Split a series into contiguous runs so a gap renders as a gap. */
export function segmentsFor(samples: ReplicaSample[], key: "active" | "desired"): Point[][] {
  const runs: Point[][] = [];
  let current: Point[] = [];
  samples.forEach((sample, index) => {
    const value = sample[key];
    if (value === undefined || !Number.isFinite(value)) {
      if (current.length > 0) runs.push(current);
      current = [];
      return;
    }
    current.push({ index, value });
  });
  if (current.length > 0) runs.push(current);
  return runs;
}

export function ReplicaChart({
  samples,
  unavailableReason,
}: {
  samples: ReplicaSample[];
  unavailableReason?: string;
}) {
  const reported = samples.filter(
    (sample) => sample.active !== undefined || sample.desired !== undefined,
  );

  if (reported.length === 0) {
    return (
      <div className="chart-empty" role="status">
        <strong>No replica counts have been reported</strong>
        <p className="text-pretty">
          {unavailableReason ??
            "The deployment metrics export has not supplied active or desired replica counts for this run. A configured maximum is not an active replica count, so nothing is plotted."}
        </p>
      </div>
    );
  }

  const values = reported.flatMap((sample) =>
    [sample.active, sample.desired].filter((value): value is number => value !== undefined),
  );
  const top = niceMax(values);
  const span = Math.max(samples.length - 1, 1);
  const plotWidth = VIEW_WIDTH - PAD_LEFT - PAD_RIGHT;
  const plotHeight = VIEW_HEIGHT - PAD_TOP - PAD_BOTTOM;
  const xFor = (index: number) => PAD_LEFT + (index / span) * plotWidth;
  const yFor = (value: number) => PAD_TOP + plotHeight - (value / top) * plotHeight;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => Math.round(top * fraction));
  const uniqueTicks = Array.from(new Set(ticks));
  const elapsedSeconds =
    samples.length > 1 ? Math.round((samples[samples.length - 1].t - samples[0].t) / 1000) : 0;

  const gapCount = SERIES.reduce(
    (sum, series) => sum + Math.max(0, segmentsFor(samples, series.key).length - 1),
    0,
  );

  return (
    <figure className="chart-figure">
      <svg
        className="replica-chart"
        viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        role="img"
        aria-label={`Active and desired replicas over ${elapsedSeconds} seconds of this run`}
      >
        {uniqueTicks.map((tick) => (
          <g key={tick}>
            <line
              x1={PAD_LEFT}
              x2={VIEW_WIDTH - PAD_RIGHT}
              y1={yFor(tick)}
              y2={yFor(tick)}
              className="chart-grid"
            />
            <text x={PAD_LEFT - 14} y={yFor(tick) + 8} className="chart-tick" textAnchor="end">
              {tick}
            </text>
          </g>
        ))}

        <line
          x1={PAD_LEFT}
          x2={VIEW_WIDTH - PAD_RIGHT}
          y1={PAD_TOP + plotHeight}
          y2={PAD_TOP + plotHeight}
          className="chart-axis"
        />

        {SERIES.map((series) => {
          const runs = segmentsFor(samples, series.key);
          const lastRun = runs[runs.length - 1];
          const lastPoint: Point | undefined = lastRun?.[lastRun.length - 1];
          return (
            <g key={series.key}>
              {runs.map((run, runIndex) => {
                const points = run.map((point) => `${xFor(point.index)},${yFor(point.value)}`).join(" ");
                return run.length === 1 ? (
                  <circle
                    key={runIndex}
                    cx={xFor(run[0].index)}
                    cy={yFor(run[0].value)}
                    r={6}
                    style={{ fill: series.color }}
                  />
                ) : (
                  <polyline
                    key={runIndex}
                    points={points}
                    fill="none"
                    style={{ stroke: series.color }}
                    strokeWidth={4}
                    strokeDasharray={series.dash}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    vectorEffect="non-scaling-stroke"
                  />
                );
              })}
              {lastPoint && (
                <>
                  <circle
                    cx={xFor(lastPoint.index)}
                    cy={yFor(lastPoint.value)}
                    r={7}
                    style={{ fill: series.color, stroke: "var(--surface-1)" }}
                    strokeWidth={2}
                  />
                  <text
                    x={xFor(lastPoint.index) + 18}
                    y={yFor(lastPoint.value) + 9}
                    className="chart-series-label"
                  >
                    {series.label} {lastPoint.value}
                  </text>
                </>
              )}
            </g>
          );
        })}

        <text x={PAD_LEFT} y={VIEW_HEIGHT - 16} className="chart-tick">
          0 s
        </text>
        <text x={VIEW_WIDTH - PAD_RIGHT} y={VIEW_HEIGHT - 16} className="chart-tick" textAnchor="end">
          {elapsedSeconds} s
        </text>
      </svg>

      <div className="chart-legend">
        {SERIES.map((series) => (
          <span key={series.key}>
            <svg aria-hidden="true" width="26" height="12" viewBox="0 0 26 12">
              <line
                x1="1"
                x2="25"
                y1="6"
                y2="6"
                style={{ stroke: series.color }}
                strokeWidth="4"
                strokeDasharray={series.dash}
                strokeLinecap="round"
              />
            </svg>
            {series.label} replicas
          </span>
        ))}
        <span className="chart-legend-note">
          {formatCount(reported.length)} reported samples
          {gapCount > 0 ? ` · ${formatCount(gapCount)} reporting gap${gapCount === 1 ? "" : "s"}` : ""}
        </span>
      </div>

      <figcaption>
        Active replicas come from the Chain deployment management API and desired replicas from the metrics
        export. A gap in a line is a gap in the platform's reporting, not a scale-down to zero.
      </figcaption>

      <details className="chart-table">
        <summary>Replica samples as a table</summary>
        <div className="table-wrap">
          <table>
            <caption className="sr-only">Active and desired replica counts per reported sample</caption>
            <thead>
              <tr>
                <th scope="col">Elapsed</th>
                <th scope="col">Active</th>
                <th scope="col">Desired</th>
              </tr>
            </thead>
            <tbody>
              {reported.map((sample) => (
                <tr key={sample.t}>
                  <td className="tabular-nums">
                    {samples.length > 0 ? `${Math.round((sample.t - samples[0].t) / 1000)} s` : "—"}
                  </td>
                  <td className="tabular-nums">{formatCount(sample.active, "not reported")}</td>
                  <td className="tabular-nums">{formatCount(sample.desired, "not reported")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
