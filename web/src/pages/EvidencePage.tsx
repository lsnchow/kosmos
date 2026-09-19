/**
 * What has been checked, by whom, and what has not.
 */
import { useAppData } from "../AppData";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { GatePanel } from "../components/GatePanel";
import { SmokeEvidencePanel } from "../components/SmokeEvidencePanel";
import { PageHeader } from "./PageHeader";

export function EvidencePage() {
  const { gates, experiments } = useAppData();

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Evidence ledger"
        title="Evidence"
        lede="Gates A–F with the evidence each one rests on, and any smoke report imported from a real-model run. A gate that has not run says so rather than defaulting to a pass."
      />

      <div className="split-grid">
        <ErrorBoundary region="Smoke evidence">
          <SmokeEvidencePanel experiments={experiments} />
        </ErrorBoundary>
        <ErrorBoundary region="Qualification gates">
          <GatePanel gates={gates} />
        </ErrorBoundary>
      </div>
    </div>
  );
}
