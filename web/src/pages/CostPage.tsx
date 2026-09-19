/**
 * What a rollout costs, and what accuracy that buys.
 */
import { useAppData } from "../AppData";
import { BurstPanel } from "../components/BurstPanel";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { SweepPanel } from "../components/SweepPanel";
import { PageHeader } from "./PageHeader";

export function CostPage() {
  const {
    sweeps,
    activeRun,
    telemetry,
    samples,
    submitting,
    policies,
    tasks,
    plannedEpisodes,
    idempotencyKey,
    launchBurst,
    setCancelOpen,
  } = useAppData();

  return (
    <div className="page-stack">
      <PageHeader
        title="Compute & cost"
        lede="Inspect saved speed, resource, and quality experiments. Estimates are not an invoice; missing measurements stay unavailable."
      />

      <ErrorBoundary region="Cost–fidelity slider">
        <SweepPanel sweeps={sweeps} />
      </ErrorBoundary>

      <ErrorBoundary region="Burst panel">
        <BurstPanel
          run={activeRun}
          telemetry={telemetry}
          samples={samples}
          submitting={submitting}
          policyCount={policies.length}
          taskCount={tasks.length}
          plannedEpisodes={plannedEpisodes}
          idempotencyKey={idempotencyKey}
          onLaunch={() => void launchBurst()}
          onCancel={() => setCancelOpen(true)}
        />
      </ErrorBoundary>
    </div>
  );
}
