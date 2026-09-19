import { Glyph } from "./Terminal";
import type { Gate } from "../lib/api";
import { pickString } from "../lib/format";
import { EmptyState, Panel, StatusPill } from "./Primitives";

export function gateBlockers(gates: Gate[]): string[] {
  if (gates.length === 0) return ["The gate ledger has not been returned."];
  return gates
    .filter((gate) => String(gate.status).toLowerCase() !== "pass")
    .map(
      (gate) =>
        `${pickString(gate.name, gate.id) ?? "Gate"}: ${
          pickString(gate.reason, gate.summary, gate.blocker) ?? String(gate.status ?? "not_run")
        }`,
    );
}

export function GatePanel({ gates }: { gates: Gate[] }) {
  const ordered = [...gates].sort((a, b) =>
    (pickString(a.id, a.name) ?? "").localeCompare(pickString(b.id, b.name) ?? ""),
  );
  return (
    <Panel title="Qualification gates" className="gates-panel" id="gates">
      {ordered.length === 0 ? (
        <EmptyState icon={<Glyph name="alert" />}>
          No gate ledger was returned. Qualified claims remain unavailable.
        </EmptyState>
      ) : (
        <ol className="gate-list">
          {ordered.map((gate, index) => {
            const marker = pickString(gate.id) ?? index + 1;
            const name = pickString(gate.name, gate.id) ?? `Gate ${index + 1}`;
            const reason = pickString(
              gate.reason,
              gate.summary,
              gate.blocker,
              gate.details,
              gate.description,
            );
            return (
              <li key={`${name}-${index}`} className="gate-row">
                <div>
                  <div className="gate-name">
                    <span className="gate-index">{marker}</span>
                    {/* The API returns no separate name for these gates, so
                        `name` falls back to the id and the row read "A  A".
                        The badge already carries the id; print the name only
                        when it says something the badge does not. */}
                    {name === String(marker) ? null : name}
                  </div>
                  <p className="text-pretty">{reason ?? "No evidence reason recorded."}</p>
                </div>
                <StatusPill status={gate.status} />
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}
