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
        title="Evidence"
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
