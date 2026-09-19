import { CellBar, Glyph } from "./Terminal";
import type { Run, Telemetry } from "../lib/api";
import { formatCount, formatUsd, isTerminalStatus, pickNumber, formatCountPair} from "../lib/format";
import { InfoTip, DataValue, Note, Panel, SourceChip, StatusPill } from "./Primitives";
import { ReplicaChart, type ReplicaSample } from "./ReplicaChart";

/**
 * Whether the burst button may be pressed.
 *
 * `submitting` alone is not enough: the old build cleared its flag as soon as
 * the POST *returned*, which is long before the run executes, so a second stage
 * click landed on an enabled button. The button stays disabled for as long as a
 * run is alive, and the derived idempotency key means even a race cannot start a
 * second execution.
 */
export function burstDisabledReason(input: {
  submitting: boolean;
  run?: Run;
  policyCount: number;
  taskCount: number;
}): string | undefined {
  if (input.submitting) return "Submitting this run";
  if (input.policyCount === 0 || input.taskCount === 0)
    return "The protocol did not return policies and tasks";
  if (input.run && !isTerminalStatus(input.run.status))
    return `Run ${input.run.id ?? ""} is ${String(input.run.status ?? "active")}`.trim();
  return undefined;
}

export function BurstPanel({
  run,
  telemetry,
  samples,
  submitting,
  policyCount,
  taskCount,
  plannedEpisodes,
  idempotencyKey,
  onLaunch,
  onCancel,
}: {
  run?: Run;
  telemetry?: Telemetry;
  samples: ReplicaSample[];
  submitting: boolean;
  policyCount: number;
  taskCount: number;
  plannedEpisodes?: number;
  idempotencyKey: string;
  onLaunch: () => void;
  onCancel: () => void;
}) {
  const disabledReason = burstDisabledReason({ submitting, run, policyCount, taskCount });
  const active = Boolean(run && !isTerminalStatus(run.status));
  const completed = pickNumber(run?.completed);
  const total = pickNumber(run?.total, plannedEpisodes);
  const marginal = pickNumber(telemetry?.marginal_estimated_usd);
  const totalUsd = pickNumber(telemetry?.total_estimated_usd);
  const queueStatus = typeof telemetry?.platform_queue_status === "string"
    ? telemetry.platform_queue_status
    : undefined;

  return (
    <Panel
      title="Full-matrix burst"
      className="burst-panel"
      action={<SourceChip>key {idempotencyKey}</SourceChip>}
      id="burst"
    >
      <div className="burst-actions">
        <button
          type="button"
          className="button button-primary button-stage"
          disabled={Boolean(disabledReason)}
          aria-describedby={disabledReason ? "burst-disabled-reason" : undefined}
          onClick={onLaunch}
        >
          <Glyph name="play" />
          {submitting ? "Submitting…" : `Run ${formatCount(plannedEpisodes, "the full matrix of")} episodes`}
        </button>
        {active && (
          <button type="button" className="button button-danger" onClick={onCancel}>
            <Glyph name="stop" />
            Cancel run
          </button>
        )}
        <StatusPill status={run?.status ?? "no run"}>{String(run?.status ?? "no run")}</StatusPill>
      </div>
      {disabledReason && (
        <p className="burst-disabled" id="burst-disabled-reason">
          Disabled: {disabledReason}.
          <InfoTip label="why the button is disabled">
            Repeat submissions carry the same derived idempotency key, so the ledger returns the existing run
            instead of starting a second one.
          </InfoTip>
        </p>
      )}

      {/*
        * The donor's footer bar. It is aria-hidden and carries no number of its
        * own: the count it draws is already stated, exactly once, by the
        * "Rollouts completed" metric below it. A bar that also printed the
        * figure would be a second place for the same fact to go stale.
        */}
      {completed !== undefined && total !== undefined && total > 0 && (
        <p className="burst-progress" aria-hidden="true">
          <span className="burst-progress-label">tasks</span>
          <CellBar value={completed / total} width={40} tone="good" />
        </p>
      )}

      <div className="burst-metrics">
        <DataValue
          label="Rollouts completed"
          value={
            completed === undefined
              ? "—"
              : formatCountPair(completed, total)
          }
          source="Application ledger · logical episodes"
          size="large"
        />
        <DataValue
          label="Marginal estimate"
          value={formatUsd(marginal)}
          source="Allocation ledger × account prices"
          size="large"
        />
        <DataValue
          label="Total run estimate"
          value={formatUsd(totalUsd)}
          source="Prewarm through cooldown, all stages"
          size="large"
        />
      </div>

      <ReplicaChart
        samples={samples}
        unavailableReason={
          queueStatus === "unavailable"
            ? "Platform replica and queue telemetry is reported unavailable for this deployment, so nothing is plotted."
            : undefined
        }
      />

      <Note summary="How cost and counts are derived">
        Estimated USD is an allocation-ledger estimate against timestamped account prices, not settled
        billing. Completed counts are logical episodes from the application ledger and do not grow with
        transport retries. The scale-up is timed separately from the run so a cold start is not hidden inside
        it.
      </Note>
    </Panel>
  );
}
