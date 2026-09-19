import { Glyph } from "./Terminal";
import { isRecord, type CalledShot, type JsonRecord } from "../lib/api";
import { formatCount, formatRateAsPercent, pickNumber, pickString } from "../lib/format";
import { Note, Panel, SourceChip, StatusPill } from "./Primitives";

/**
 * Recover the headline cell from `protocol.reference` when the dedicated
 * endpoint is unavailable. These two numbers already ship inside `/api/protocol`
 * and the old client fetched and discarded them.
 */
export function calledShotFromProtocol(protocol: JsonRecord | undefined): CalledShot | undefined {
  const reference = protocol?.reference;
  if (!isRecord(reference)) return undefined;
  const arithmetic = reference.arithmetic;
  const headline = isRecord(arithmetic) ? arithmetic.headline_openvla_close_drawer : undefined;
  if (!isRecord(headline)) return undefined;
  const trials = pickNumber(reference.trials_per_cell);
  return {
    cell: "OpenVLA / close_drawer",
    human_rate: pickNumber(headline.human_rate) ?? null,
    human_n: trials ?? null,
    simpler_rate: pickNumber(headline.simpler_rate) ?? null,
    simpler_n: trials ?? null,
    plumb_estimate: null,
    plumb_status: "pending",
    prereg_uri: null,
    prereg_sha256: null,
  };
}

/**
 * The called shot.
 *
 * Kosmos's estimate for this cell is pre-registered and frozen *before* the
 * published human number is looked at. When there is no estimate yet the panel
 * shows it as pending — never as a number, and never as a comparison Kosmos has
 * already won.
 */
export function CalledShotPanel({ calledShot }: { calledShot?: CalledShot }) {
  const humanRate = pickNumber(calledShot?.human_rate);
  const simplerRate = pickNumber(calledShot?.simpler_rate);
  const estimate = pickNumber(calledShot?.plumb_estimate);
  const status = pickString(calledShot?.plumb_status) ?? (estimate === undefined ? "pending" : "recorded");
  // The control plane nests these under `preregistration`; accept either.
  const prereg = pickString(calledShot?.prereg_uri, calledShot?.preregistration?.uri);
  const sha = pickString(calledShot?.prereg_sha256, calledShot?.preregistration?.sha256);

  // Prefer the server's own arithmetic over recomputing it here.
  const gapPoints =
    pickNumber(calledShot?.published_gap_points) ??
    (humanRate !== undefined && simplerRate !== undefined
      ? Math.abs(humanRate - simplerRate) * 100
      : undefined);

  return (
    <Panel
      title="The called shot"
      className="calledshot-panel"
      action={<SourceChip>{pickString(calledShot?.cell) ?? "OpenVLA / close_drawer"}</SourceChip>}
      id="calledshot"
    >
      <div className="calledshot-grid">
        <div className="calledshot-cell">
          <span>Physics simulator</span>
          <strong className="tabular-nums">{formatRateAsPercent(simplerRate)}</strong>
          <span className="calledshot-source">
            SIMPLER · AutoEval Table 3 · n={formatCount(calledShot?.simpler_n)}
          </span>
        </div>
        <div className="calledshot-cell calledshot-cell-plumb">
          <span>Kosmos pre-registered estimate</span>
          {estimate === undefined ? (
            <strong className="calledshot-pending">
              <StatusPill status="pending" title={status}>
                {status.split(/[_\s]/)[0].toLowerCase()}
              </StatusPill>
            </strong>
          ) : (
            <strong className="tabular-nums">{formatRateAsPercent(estimate)}</strong>
          )}
          <span className="calledshot-source">
            {estimate === undefined
              ? "No frozen estimate has been recorded for this cell yet."
              : "Frozen in the preregistration below before the reference was read."}
          </span>
        </div>
        <div className="calledshot-cell">
          <span>Real robot, human-scored</span>
          <strong className="tabular-nums">{formatRateAsPercent(humanRate)}</strong>
          <span className="calledshot-source">
            AutoEval Table 2 · n={formatCount(calledShot?.human_n)}
          </span>
        </div>
      </div>

      {gapPoints !== undefined && (
        <p className="calledshot-gap">
          <Glyph name="target" />
          {/* One span, not three bare nodes: .calledshot-gap is a flex row, so
              every top-level child became a flex item and the `gap` meant to
              separate the glyph from the sentence was opening columns inside
              the sentence itself. */}
          <span>
            The simulator and the real robot disagree by{" "}
            <b className="tabular-nums">{gapPoints.toFixed(0)} percentage points</b> on this one cell.
          </span>
        </p>
      )}

      <div className="prereg-row">
        <Glyph name="lock" />
        {prereg ? (
          <div>
            <a href={prereg} target="_blank" rel="noreferrer">
              {prereg}
            </a>
            <code>{sha ?? "digest not reported"}</code>
          </div>
        ) : (
          <div>
            <strong>No preregistration record was returned</strong>
            <p className="text-pretty">
              Without a signed tag URI and digest there is no independent timing evidence, so this estimate is
              not presented as a pre-registered call.
            </p>
          </div>
        )}
      </div>

      <Note summary="Where these reference numbers come from">
        Both reference numbers are AutoEval's published results, not Kosmos outcomes. The already-published
        human value is not treated as a blind target; what is frozen is Kosmos's own estimate and the time it
        was recorded.
      </Note>
    </Panel>
  );
}
