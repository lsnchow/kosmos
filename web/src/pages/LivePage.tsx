/**
 * Advanced developer tools, preserved at the original /live URL. The old
 * engineering panel sequence remains here; the demo now opens at /console.
 */
import { useAppData } from "../AppData";
import { BurstPanel } from "../components/BurstPanel";
import { CloudDiagnosticPanel } from "../components/CloudDiagnosticPanel";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { RolloutWall } from "../components/RolloutWall";
import { Scoreboard } from "../components/Scoreboard";
import { StageLadder } from "../components/StageLadder";
import { SweepPanel } from "../components/SweepPanel";
import { TaskPrompt } from "../components/TaskPrompt";
import { TelemetryStrip } from "../components/TelemetryStrip";
import { taskInstruction } from "../lib/tasks";
import { PageHeader } from "./PageHeader";

export function LivePage() {
  const {
    wall,
    activeRun,
    episodes,
    runIsTerminal,
    scopedTask,
    setViewerSlot,
    launchPrompt,
    promptBusy,
    openFreeplay,
    telemetry,
    stream,
    analysis,
    now,
    samples,
    submitting,
    policies,
    tasks,
    plannedEpisodes,
    idempotencyKey,
    launchBurst,
    setCancelOpen,
    sweeps,
    protocol,
  } = useAppData();

  return (
    <div className="page-stack">
      <PageHeader
        title="Developer tools"
        lede="Run an isolated cloud policy check or a synthetic rehearsal. These tools do not generate the gallery’s matched policy-comparison videos."
      />

      <ErrorBoundary region="Cloud diagnostic">
        <CloudDiagnosticPanel />
      </ErrorBoundary>

      <TaskPrompt
        onSubmit={(text) => void launchPrompt(text)}
        busy={promptBusy}
      />

      <ErrorBoundary region="Rollout viewport">
        <RolloutWall
          wall={wall}
          runId={activeRun?.id}
          onExpand={setViewerSlot}
          onDrive={openFreeplay}
          scopedTask={scopedTask}
          scopedInstruction={taskInstruction(scopedTask)}
        />
      </ErrorBoundary>

      <ErrorBoundary region="Chain stages">
        <StageLadder
          wall={wall}
          episodes={episodes}
          runId={activeRun?.id}
          runTerminal={runIsTerminal}
        />
      </ErrorBoundary>

      <ErrorBoundary region="Live telemetry">
        <TelemetryStrip
          run={activeRun}
          telemetry={telemetry}
          streamStatus={stream.status}
          retryCount={stream.retryCount}
          ledgerState={analysis.ledger_state}
          now={now}
        />
      </ErrorBoundary>

      <div className="split-grid">
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
        <ErrorBoundary region="Cost–fidelity slider">
          <SweepPanel sweeps={sweeps} />
        </ErrorBoundary>
      </div>

      {/* Beat four lands here, so the closing line has the error bars behind it. */}
      <ErrorBoundary region="Scoreboard">
        <Scoreboard analysis={analysis} protocol={protocol} compact />
      </ErrorBoundary>
    </div>
  );
}
