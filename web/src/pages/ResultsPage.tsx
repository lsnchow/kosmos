/**
 * The persisted analysis, and which run produced it.
 *
 * The scoreboard is the same component the Live page ends on — one instance of
 * one component rendered by two pages, never a copy.
 */
import { useAppData } from "../AppData";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { HeroRollout } from "../components/HeroRollout";
import { RunLedgerPanel } from "../components/RunLedgerPanel";
import { Scoreboard } from "../components/Scoreboard";
import { PageHeader } from "./PageHeader";

export function ResultsPage() {
  const { analysis, protocol, runs, activeRun, loadRun, episodes } =
    useAppData();

  return (
    <div className="page-stack">
      <PageHeader
        title="Saved runs"
        lede="Select a past run to inspect its recorded measurements. Synthetic runs test the software; their scores are not robot performance."
      />

      <div className="split-grid">
        <ErrorBoundary region="Run ledger">
          <RunLedgerPanel
            runs={runs}
            activeRun={activeRun}
            onSelect={(run) => void loadRun(run)}
          />
        </ErrorBoundary>
        <ErrorBoundary region="Presentation rollout">
          <HeroRollout episodes={episodes} />
        </ErrorBoundary>
      </div>
      <ErrorBoundary region="Scoreboard">
        <Scoreboard analysis={analysis} protocol={protocol} />
      </ErrorBoundary>
    </div>
  );
}
