import type { Run } from "../lib/api";
import { formatClockTime, formatCount } from "../lib/format";
import { DataValue, Note, Panel } from "./Primitives";

export function RunLedgerPanel({
  runs,
  activeRun,
  onSelect,
}: {
  runs: Run[];
  activeRun?: Run;
  onSelect: (run: Run) => void;
}) {
  return (
    <Panel title="Run history">
      <label className="select-label" htmlFor="run-select">
        Choose a saved run
      </label>
      <select
        id="run-select"
        value={activeRun?.id ?? ""}
        onChange={(event) => {
          const run = runs.find((item) => item.id === event.target.value);
          if (run) onSelect(run);
        }}
      >
        <option value="">Select a run</option>
        {runs.map((run) => (
          <option key={run.id} value={run.id}>
            {run.id ?? "unnamed run"} · {run.status ?? "unknown"}
          </option>
        ))}
      </select>
      <div className="run-metrics">
        <DataValue
          label="Status"
          value={String(activeRun?.status ?? "—")}
          source="Logical run ledger"
        />
        <DataValue
          label="Created"
          value={formatClockTime(activeRun?.created_at)}
          source="Logical run ledger"
        />
        <DataValue
          label="Planned"
          value={formatCount(activeRun?.total)}
          source="Planned logical episodes"
        />
        <DataValue
          label="Completed"
          value={formatCount(activeRun?.completed)}
          source="Terminal logical episodes"
        />
        <DataValue
          label="Evaluable"
          value={formatCount(activeRun?.evaluable)}
          source="Non-null binary outcome"
        />
        <DataValue
          label="Failed"
          value={formatCount(activeRun?.failed)}
          source="Terminal service failures"
        />
        <DataValue
          label="Cancelled"
          value={formatCount(activeRun?.cancelled)}
          source="Explicit terminal records"
        />
        <DataValue
          label="Successes"
          value={formatCount(activeRun?.successes)}
          source="Evaluable positives"
        />
      </div>
      <Note summary="What counts as an episode">
        These are logical episodes, not request attempts: a transport retry
        creates an attempt, never a new statistical episode. The ledger exposes
        no separate submitted or excluded counter, so neither is derived by
        subtraction here.
      </Note>
    </Panel>
  );
}
