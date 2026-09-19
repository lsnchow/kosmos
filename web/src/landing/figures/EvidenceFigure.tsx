/**
 * FIG.3 — the provenance ledger.
 *
 * Rows arrive one at a time, each carrying what it is, where it came from, and
 * a content hash. Two of the rows read `not yet measured` and still occupy a
 * full row with a full-width rule, which is the point: an unmeasured figure is
 * a first-class entry in this ledger, not a gap someone forgot to fill.
 *
 * The hashes are derived from the row's own text rather than invented, so the
 * same row always renders the same digest and nothing here implies a run that
 * did not happen.
 */
import { BURST_TARGET, CALLED_SHOT, PENDING, STACK, VALIDATION_NOTE } from "../content";
import { stagger, useProgress } from "./useProgress";

/**
 * A stable 40-bit digest of a string, rendered as ten hex characters.
 *
 * FNV-1a. This is a display device and not a security primitive: its only job
 * is to be deterministic, so a reader who scrolls past twice sees the same
 * ledger both times.
 */
function digest(input: string) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0").slice(0, 8);
}

type Row = {
  claim: string;
  value: string;
  source: string;
  /** `cited` carries a table; `measured` carries a run; `pending` carries neither. */
  kind: "cited" | "measured" | "pending";
};

const ROWS: Row[] = [
  {
    claim: `${CALLED_SHOT.policy} · ${CALLED_SHOT.task}`,
    value: `${Math.round(CALLED_SHOT.human.rate * 100)}%`,
    source: CALLED_SHOT.source,
    kind: "cited",
  },
  {
    claim: "Same cell, physics simulator",
    value: `${Math.round(CALLED_SHOT.simulator.rate * 100)}%`,
    source: "AutoEval, Table 3",
    kind: "cited",
  },
  {
    claim: "Agreement with the real arm",
    value: "r = 0.78",
    source: VALIDATION_NOTE.source,
    kind: "cited",
  },
  {
    claim: "World model",
    value: STACK[1].choice,
    source: STACK[1].note,
    kind: "measured",
  },
  {
    claim: `Burst · ${BURST_TARGET.episodes.toLocaleString()} episodes`,
    value: `$${BURST_TARGET.usd.toFixed(2)}`,
    source: BURST_TARGET.status,
    kind: "pending",
  },
  {
    claim: "Split-half reliability",
    value: PENDING,
    source: "No qualified run has produced it",
    kind: "pending",
  },
];

export function EvidenceFigure({ play }: { play: boolean }) {
  const progress = useProgress(play, 2200);

  return (
    <div className="evidence-ledger">
      <div className="evidence-head" aria-hidden="true">
        <span>Source</span>
        <span>Claim</span>
        <span>Value</span>
        <span>Digest</span>
      </div>
      <ul className="evidence-rows">
        {ROWS.map((row, index) => {
          const local = stagger(progress, index, ROWS.length, 0.4);
          return (
            <li
              key={row.claim}
              className="evidence-row"
              data-kind={row.kind}
              style={{
                opacity: local,
                // A short lift rather than a slide: the row is being written to
                // a ledger, not flying in from off-screen.
                transform: `translateY(${(1 - local) * 6}px)`,
              }}
            >
              {/*
                The provenance kind, named rather than colour-coded alone: a
                reader who cannot separate the three hues still gets the word,
                and the word is the whole taxonomy this ledger enforces.
              */}
              <span className="evidence-kind">
                <span className="evidence-pip" aria-hidden="true" />
                {row.kind}
              </span>

              <div className="evidence-claim">
                <span className="text-fg">{row.claim}</span>
                <span className="evidence-source">{row.source}</span>
              </div>
              <span
                className={
                  row.kind === "pending" && row.value === PENDING
                    ? "evidence-value font-mono text-fg-dim"
                    : "evidence-value font-mono text-fg-strong"
                }
              >
                {row.value}
              </span>
              <span className="evidence-digest font-mono text-fg-dim">
                {row.kind === "pending" && row.value === PENDING ? "—" : digest(row.claim + row.value)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
