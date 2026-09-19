/**
 * The persisted analysis, and which run produced it.
 *
 * The scoreboard is the same component the Live page ends on — one instance of
 * one component rendered by two pages, never a copy.
 */
import { useAppData } from "../AppData";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { HeroRollout } from "../components/HeroRollout";
import { Note } from "../components/Primitives";
import { RunLedgerPanel } from "../components/RunLedgerPanel";
import { Scoreboard } from "../components/Scoreboard";
import { BASE_REFRESH_MS } from "../AppData";
import { PageHeader } from "./PageHeader";

export function ResultsPage() {
  const { analysis, protocol, runs, activeRun, loadRun, episodes } = useAppData();

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Persisted analysis · unqualified"
        title="Results"
        lede="Per-cell generated rates beside the published real-robot reference, with coverage, exclusions and missingness bounds carried alongside every estimate."
      />

      <ErrorBoundary region="Scoreboard">
        <Scoreboard analysis={analysis} protocol={protocol} />
      </ErrorBoundary>

      <div className="split-grid">
        <ErrorBoundary region="Run ledger">
          <RunLedgerPanel runs={runs} activeRun={activeRun} onSelect={(run) => void loadRun(run)} />
        </ErrorBoundary>
        <ErrorBoundary region="Presentation rollout">
          <HeroRollout episodes={episodes} />
        </ErrorBoundary>
      </div>

      <Note>
        Panels update on a {BASE_REFRESH_MS / 1000}-second poll and, for the selected run, on its event
        stream. Every number carries the source that produced it; where a source reported nothing, the panel
        says so rather than showing a zero.
      </Note>
    </div>
  );
}
