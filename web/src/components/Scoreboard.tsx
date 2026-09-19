import { Gauge, Scale } from "lucide-react";
import {
  isOperationalSummary,
  type AnalysisCell,
  type AnalysisResponse,
  type JsonRecord,
  type PairwiseVerdict,
} from "../lib/api";
import {
  formatCount,
  formatRateAsPercent,
  formatRateInterval,
  pickNumber,
  pickString,
  readInterval,
} from "../lib/format";
import { cn } from "../lib/utils";
import { EmptyState, Note, Panel, SourceChip, StatusPill } from "./Primitives";

const BAR_WIDTH = 240;
const BAR_HEIGHT = 34;

/**
 * A Wilson interval drawn, not just printed.
 *
 * Two marks share one 0–100% axis: the PLUMB estimate as a circle and the
 * published human reference as a diamond, each with its own interval whiskers.
 * Shape distinguishes them, so the pair does not rely on color, and the numeric
 * values remain in the adjacent cells for anyone who cannot use the graphic.
 */
function ErrorBar({
  estimate,
  interval,
  reference,
  referenceInterval,
  label,
}: {
  estimate?: number;
  interval?: [number, number];
  reference?: number;
  referenceInterval?: [number, number];
  label: string;
}) {
  const x = (value: number) => Math.max(0, Math.min(1, value)) * BAR_WIDTH;
  const estimateY = 12;
  const referenceY = 25;

  if (estimate === undefined && reference === undefined) {
    return <span className="bar-absent">no estimate</span>;
  }

  return (
    <svg
      className="error-bar"
      viewBox={`0 0 ${BAR_WIDTH} ${BAR_HEIGHT}`}
      role="img"
      aria-label={label}
      width={BAR_WIDTH}
      height={BAR_HEIGHT}
    >
      <line x1={0} x2={BAR_WIDTH} y1={estimateY} y2={estimateY} className="bar-track" />
      <line x1={0} x2={BAR_WIDTH} y1={referenceY} y2={referenceY} className="bar-track" />

      {interval && (
        <g className="bar-interval bar-interval-estimate">
          <line x1={x(interval[0])} x2={x(interval[1])} y1={estimateY} y2={estimateY} />
          <line x1={x(interval[0])} x2={x(interval[0])} y1={estimateY - 5} y2={estimateY + 5} />
          <line x1={x(interval[1])} x2={x(interval[1])} y1={estimateY - 5} y2={estimateY + 5} />
        </g>
      )}
      {estimate !== undefined && <circle cx={x(estimate)} cy={estimateY} r={5} className="bar-point-estimate" />}

      {referenceInterval && (
        <g className="bar-interval bar-interval-reference">
          <line x1={x(referenceInterval[0])} x2={x(referenceInterval[1])} y1={referenceY} y2={referenceY} />
          <line
            x1={x(referenceInterval[0])}
            x2={x(referenceInterval[0])}
            y1={referenceY - 5}
            y2={referenceY + 5}
          />
          <line
            x1={x(referenceInterval[1])}
            x2={x(referenceInterval[1])}
            y1={referenceY - 5}
            y2={referenceY + 5}
          />
        </g>
      )}
      {reference !== undefined && (
        <rect
          x={x(reference) - 4.5}
          y={referenceY - 4.5}
          width={9}
          height={9}
          transform={`rotate(45 ${x(reference)} ${referenceY})`}
          className="bar-point-reference"
        />
      )}
    </svg>
  );
}

function cellKey(cell: AnalysisCell) {
  return `${String(cell.policy ?? "?")}::${String(cell.task ?? "?")}`;
}

/**
 * Order rows by the server's own `visual_order` where it exists, falling back to
 * a descending sort on the estimate. Either way the order is presentational: the
 * server also reports `ordering: "indeterminate"`, and that verdict is what the
 * table states.
 */
function rankPolicies(analysis: AnalysisResponse, cells: AnalysisCell[]): string[] {
  const macro = analysis.rankings?.find((ranking) => ranking.scope === "macro");
  if (macro?.visual_order && macro.visual_order.length > 0) return macro.visual_order;
  const totals = new Map<string, number>();
  for (const cell of cells) {
    const policy = String(cell.policy ?? "?");
    const value = pickNumber(cell.positive_rate, cell.rate) ?? -1;
    totals.set(policy, Math.max(totals.get(policy) ?? -1, value));
  }
  return [...totals.entries()].sort((a, b) => b[1] - a[1]).map(([policy]) => policy);
}

function pairSummaryFor(policy: string, pairwise: PairwiseVerdict[]) {
  const involved = pairwise.filter(
    (pair) => pair.policy_a === policy || pair.policy_b === policy,
  );
  const supported = involved.filter((pair) => pair.supported === true).length;
  return { total: involved.length, supported };
}

/** Provisional operational counts. Deliberately not rendered as study statistics. */
function ProvisionalBand({ summary }: { summary: JsonRecord }) {
  return (
    <div className="provisional-band" role="status">
      <div className="provisional-title">
        <StatusPill status="pending">provisional</StatusPill>
        <strong>Live operational counts only</strong>
      </div>
      <p className="text-pretty">
        {pickString(summary.reason) ??
          "The ledger still has nonterminal records, so only operational counts are available."}{" "}
        These are not study statistics and no rate is computed from them.
      </p>
      <dl className="provisional-counts tabular-nums">
        <div>
          <dt>Reported records</dt>
          <dd>{formatCount(summary.reported_records)}</dd>
        </div>
        <div>
          <dt>Planned records</dt>
          <dd>{formatCount(summary.planned_records)}</dd>
        </div>
        <div>
          <dt>Terminal</dt>
          <dd>{formatCount(summary.terminal_records)}</dd>
        </div>
        <div>
          <dt>Nonterminal</dt>
          <dd>{formatCount(summary.nonterminal_records)}</dd>
        </div>
      </dl>
    </div>
  );
}

export function Scoreboard({
  analysis,
  protocol,
}: {
  analysis: AnalysisResponse;
  protocol?: JsonRecord;
}) {
  const cells = analysis.cells ?? [];
  const pairwise = analysis.pairwise ?? [];
  const summary = analysis.summary;
  const provisional = isOperationalSummary(summary);

  const backend = pickString(protocol?.backend_revision, protocol?.backend) ?? "not recorded";
  const judge = pickString(protocol?.judge_revision, protocol?.judge) ?? "not recorded";
  const parity = pickString(protocol?.parity, protocol?.scenario_parity) ?? "not recorded";

  const policyOrder = rankPolicies(analysis, cells);
  const rankOf = new Map(policyOrder.map((policy, index) => [policy, index + 1]));
  const sortedCells = [...cells].sort((a, b) => {
    const rankA = rankOf.get(String(a.policy ?? "?")) ?? Number.MAX_SAFE_INTEGER;
    const rankB = rankOf.get(String(b.policy ?? "?")) ?? Number.MAX_SAFE_INTEGER;
    if (rankA !== rankB) return rankA - rankB;
    return String(a.task ?? "").localeCompare(String(b.task ?? ""));
  });

  const macroRanking = analysis.rankings?.find((ranking) => ranking.scope === "macro");
  const indeterminatePairs = pairwise.filter((pair) => pair.supported !== true);

  return (
    <Panel
      title="Scoreboard"
      eyebrow={provisional ? "Provisional · study statistics unavailable" : "Persisted analysis · unqualified"}
      className="score-panel"
      action={<SourceChip>published reference is a separate column</SourceChip>}
      id="scoreboard"
    >
      <div className="provenance-row">
        <span>
          Parity: <b>{parity}</b>
        </span>
        <span>
          Backend: <b>{backend}</b>
        </span>
        <span>
          Judge: <b>{judge}</b>
        </span>
        {analysis.endpoint?.name && (
          <span>
            Endpoint: <b>{analysis.endpoint.name}</b>
          </span>
        )}
        {analysis.lineage_leakage?.status && (
          <span>
            Lineage check: <b>{String(analysis.lineage_leakage.status)}</b>
          </span>
        )}
      </div>

      {provisional && summary && <ProvisionalBand summary={summary} />}

      {sortedCells.length === 0 ? (
        <EmptyState className="scoreboard-empty" icon={<Gauge aria-hidden="true" className="size-5" />}>
          No persisted analysis exists for the selected run. Provisional rollout counts are not presented as
          study statistics, and no cell is filled with a zero.
        </EmptyState>
      ) : (
        <>
          <div className="table-wrap">
            <table className="score-table">
              <caption className="sr-only">
                Per-cell generated rates against the published human reference, with intervals, coverage
                denominators, exclusions and pairwise separation
              </caption>
              <thead>
                <tr>
                  <th scope="col">Rank</th>
                  <th scope="col">Policy / task</th>
                  <th scope="col">Estimate vs reference</th>
                  <th scope="col">
                    Generated rate <span className="th-sub">S/V</span>
                  </th>
                  <th scope="col">
                    Wilson 95% <span className="th-sub">generated</span>
                  </th>
                  <th scope="col">
                    Human reference <span className="th-sub">AutoEval T2</span>
                  </th>
                  <th scope="col">
                    Wilson 95% <span className="th-sub">reference</span>
                  </th>
                  <th scope="col">
                    SIMPLER <span className="th-sub">AutoEval T3</span>
                  </th>
                  <th scope="col">
                    Coverage <span className="th-sub">V / n</span>
                  </th>
                  <th scope="col">
                    Excluded &amp; invalid <span className="th-sub">n − V</span>
                  </th>
                  <th scope="col">Missingness bounds</th>
                  <th scope="col">Pair separation</th>
                </tr>
              </thead>
              <tbody>
                {sortedCells.map((cell) => {
                  const policy = String(cell.policy ?? "—");
                  // `policy` is an arm id that can encode variant and cohort;
                  // the readable base name goes in the cell and the full arm id
                  // stays available as a tooltip rather than being discarded.
                  const policyLabel = pickString(cell.policy_base) ?? policy;
                  const task = String(cell.task ?? "—");
                  const n = pickNumber(cell.n);
                  const valid = pickNumber(cell.valid);
                  const estimate = pickNumber(cell.positive_rate, cell.rate);
                  const wilson = readInterval(cell.positive_wilson) ?? readInterval(cell.wilson);
                  const reference = pickNumber(cell.reference_rate);
                  const referenceWilson = readInterval(cell.reference_wilson);
                  const simpler = pickNumber(cell.simpler_rate);
                  const excluded = n !== undefined && valid !== undefined ? n - valid : undefined;
                  const reasons = Object.entries(cell.missing_reason_counts ?? {});
                  const pairs = pairSummaryFor(policy, pairwise);
                  const blocked = cell.scientific_status === "blocked";
                  return (
                    <tr key={cellKey(cell)} className={cn(blocked && "row-blocked")}>
                      <td className="tabular-nums rank-cell">
                        {rankOf.get(policy) ?? "—"}
                        <span className="rank-note">visual</span>
                      </td>
                      <td title={policy}>
                        <strong>{policyLabel}</strong>
                        <span>{task}</span>
                        {cell.policy_variant && <span>variant {String(cell.policy_variant)}</span>}
                      </td>
                      <td>
                        <ErrorBar
                          estimate={estimate}
                          interval={wilson}
                          reference={reference}
                          referenceInterval={referenceWilson}
                          label={`${policy} on ${task}: generated ${formatRateAsPercent(estimate)} with Wilson interval ${formatRateInterval(wilson)}; human reference ${formatRateAsPercent(reference)} with interval ${formatRateInterval(referenceWilson)}`}
                        />
                      </td>
                      <td className="tabular-nums">
                        {blocked ? (
                          <span className="cell-blocked" title={pickString(cell.scientific_reason)}>
                            blocked
                          </span>
                        ) : (
                          formatRateAsPercent(estimate)
                        )}
                      </td>
                      <td className="tabular-nums">{formatRateInterval(cell.positive_wilson ?? cell.wilson)}</td>
                      <td className="tabular-nums">
                        {formatRateAsPercent(reference)}
                        {cell.reference_successes !== null && cell.reference_successes !== undefined && (
                          <span>
                            {formatCount(cell.reference_successes)}/{formatCount(cell.reference_n)}
                          </span>
                        )}
                      </td>
                      <td className="tabular-nums">{formatRateInterval(cell.reference_wilson)}</td>
                      <td className="tabular-nums">
                        {simpler === undefined ? (
                          <span className="cell-absent" title="SIMPLER has no column for this task; absence is not a zero.">
                            no published cell
                          </span>
                        ) : (
                          formatRateAsPercent(simpler)
                        )}
                      </td>
                      <td className="tabular-nums">
                        {formatRateAsPercent(cell.coverage)}
                        <span>
                          {formatCount(valid)} / {formatCount(n)}
                        </span>
                      </td>
                      <td className="tabular-nums">
                        {formatCount(excluded)}
                        {reasons.length > 0 && (
                          <span title={reasons.map(([reason, count]) => `${reason}: ${count}`).join(", ")}>
                            {reasons.map(([reason, count]) => `${reason} ${count}`).join(" · ")}
                          </span>
                        )}
                        {pickNumber(cell.service_failures) ? (
                          <span>service failures {formatCount(cell.service_failures)}</span>
                        ) : null}
                      </td>
                      <td className="tabular-nums">{formatRateInterval(cell.missing_bounds)}</td>
                      <td>
                        {pairs.total === 0 ? (
                          <span className="cell-absent">no pair tested</span>
                        ) : (
                          <StatusPill status={pairs.supported > 0 ? "pass" : "unknown"}>
                            {`${pairs.supported}/${pairs.total} separated`}
                          </StatusPill>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="bar-key" aria-hidden="true">
            <span>
              <svg viewBox="0 0 16 16" width="14" height="14">
                <circle cx="8" cy="8" r="5" className="bar-point-estimate" />
              </svg>
              generated estimate
            </span>
            <span>
              <svg viewBox="0 0 16 16" width="14" height="14">
                <rect x="3.5" y="3.5" width="9" height="9" transform="rotate(45 8 8)" className="bar-point-reference" />
              </svg>
              published human reference
            </span>
            <span>whiskers are Wilson 95% intervals</span>
          </div>
        </>
      )}

      {macroRanking && (
        <div className="ranking-verdict">
          <Scale aria-hidden="true" className="size-4" />
          <div>
            <strong>
              Macro ordering: {String(macroRanking.ordering ?? "indeterminate")}
              {macroRanking.supported === true ? "" : " — not supported"}
            </strong>
            <p className="text-pretty">
              {pickString(macroRanking.reason) ??
                "The rank column is a visual sort of the estimates. It is not a claim that the order is resolved."}
            </p>
          </div>
        </div>
      )}

      {pairwise.length > 0 && (
        <details className="pairwise-detail">
          <summary>
            Pairwise separation — {formatCount(indeterminatePairs.length)} of {formatCount(pairwise.length)}{" "}
            comparisons indeterminate
          </summary>
          <ul className="pairwise-list">
            {pairwise.map((pair, index) => {
              const supported = pair.supported === true;
              const paired = pair.paired ?? {};
              return (
                <li key={`${pair.scope}-${pair.policy_a}-${pair.policy_b}-${index}`}>
                  <div className="pairwise-head">
                    <StatusPill status={supported ? "pass" : "unknown"}>
                      {supported ? "separated" : "indeterminate"}
                    </StatusPill>
                    <b>
                      {String(pair.policy_a ?? "?")} vs {String(pair.policy_b ?? "?")}
                    </b>
                    <span>{String(pair.scope ?? "—")}</span>
                  </div>
                  <p className="text-pretty">
                    {pickString(pair.ordering_reason) ??
                      "No separation reason was reported for this pair."}
                  </p>
                  <span className="pairwise-stats tabular-nums">
                    {pickNumber(paired.paired_n) !== undefined && `paired n ${formatCount(paired.paired_n)} · `}
                    {pickNumber(paired.p_value) !== undefined &&
                      `exact McNemar p ${Number(paired.p_value).toExponential(2)} · `}
                    {pickNumber(paired.family_alpha) !== undefined &&
                      `family alpha ${Number(paired.family_alpha).toExponential(2)}`}
                    {pickString(paired.reason) ?? ""}
                  </span>
                </li>
              );
            })}
          </ul>
        </details>
      )}

      <Note>
        The rank column sorts estimates visually and nothing more. An ordering is claimed only where the
        pairwise test reports it as supported under its declared family alpha; every other pair stays
        indeterminate. Coverage carries both denominators, so <code>V</code> and <code>n</code> are never
        collapsed into a single rate. Wilson intervals are descriptive where starts are correlated or
        repeated. Published reference cells are AutoEval's real-robot results, not PLUMB outcomes, and an
        absent SIMPLER cell is an absence rather than a zero.
      </Note>
    </Panel>
  );
}
