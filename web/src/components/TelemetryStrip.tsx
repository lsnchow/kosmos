import { Glyph } from "./Terminal";
import type { Run, Telemetry } from "../lib/api";
import {
  ageSeconds,
  formatClockTime,
  formatCount,
  formatGpuSeconds,
  formatUsd,
  pickNumber,
  pickString,
  formatCountPair,
} from "../lib/format";
import type { StreamStatus } from "../hooks/useRunStream";
import { DataValue, StatusPill } from "./Primitives";

/** Beyond this, a 1 Hz feed is not current and the strip says so. */
export const STALE_AFTER_SECONDS = 5;

/**
 * Freshness comes from the *server's* timestamp.
 *
 * The old strip used `eventReceivedAt`, a browser clock reading taken when the
 * message arrived. That value is fresh by construction — it says the browser
 * received something recently, not that the metric behind it is current. It
 * cannot detect a server replaying a cached sample.
 */
export function freshnessOf(
  telemetry: Telemetry | undefined,
  now: number,
): { stale: boolean; ageSeconds?: number; reason: string; source?: unknown } {
  if (!telemetry) return { stale: false, reason: "No telemetry event has arrived yet." };
  const source = telemetry.fresh_at ?? telemetry.timestamp;
  const age = ageSeconds(source, now);
  if (telemetry.stale === true) {
    return { stale: true, ageSeconds: age, reason: "The server marked this sample stale.", source };
  }
  if (age === undefined) {
    return {
      stale: true,
      reason: "The telemetry sample carries no server timestamp, so its age cannot be established.",
      source,
    };
  }
  if (age > STALE_AFTER_SECONDS) {
    return {
      stale: true,
      ageSeconds: age,
      reason: "Older than the 1 s cadence this feed declares.",
      source,
    };
  }
  return { stale: false, ageSeconds: age, reason: "Current against the declared 1 s cadence.", source };
}

export function TelemetryStrip({
  run,
  telemetry,
  streamStatus,
  retryCount,
  ledgerState,
  now,
}: {
  run?: Run;
  telemetry?: Telemetry;
  streamStatus: StreamStatus;
  retryCount: number;
  ledgerState?: { nonterminal_status_counts?: Record<string, number> };
  now: number;
}) {
  const freshness = freshnessOf(telemetry, now);
  const stale = freshness.stale;
  const tone = stale ? ("warn" as const) : undefined;
  const ledgerSource = "Application ledger · logical episodes";

  const queue = pickNumber(telemetry?.platform_queue);
  const queueStatus = pickString(telemetry?.platform_queue_status);
  const activeReplicas = pickNumber(telemetry?.active_replicas);
  const desiredReplicas = pickNumber(telemetry?.desired_replicas);
  const gpuSeconds = pickNumber(telemetry?.gpu_seconds, telemetry?.compute_seconds);
  const marginal = pickNumber(telemetry?.marginal_estimated_usd);
  const total = pickNumber(telemetry?.total_estimated_usd);
  const reconciliation = pickString(telemetry?.reconciliation_status);
  const nonterminal = ledgerState?.nonterminal_status_counts ?? {};
  const inFlight = Object.entries(nonterminal)
    .map(([status, count]) => `${status} ${count}`)
    .join(" · ");

  const degraded = streamStatus === "degraded";

  return (
    <section className="telemetry">
      <h2 className="telemetry-title">
        {degraded ? (
          <Glyph name="offline" />
        ) : (
          <Glyph name="live" />
        )}
        <span>Live telemetry</span>
        <StatusPill status={degraded ? "degraded" : streamStatus}>{streamStatus}</StatusPill>
        {stale && <StatusPill status="stale">stale</StatusPill>}
      </h2>

      <DataValue
        label="Completed"
        value={formatCountPair(run?.completed, run?.total)}
        source={ledgerSource}
      />
      <DataValue
        label="Platform queue"
        value={queue === undefined ? (queueStatus ?? "unavailable") : formatCount(queue)}
        source={
          queue === undefined
            ? "Chain queue route unverified · never ledger counts"
            : "Baseten async queue status"
        }
        tone={queue === undefined ? undefined : tone}
      />
      <DataValue
        label="Active / desired replicas"
        value={
          activeReplicas === undefined && desiredReplicas === undefined
            ? "unavailable"
            : formatCountPair(activeReplicas, desiredReplicas, "n/r")
        }
        source="Chain deployment API / metrics export"
        tone={tone}
      />
      <DataValue
        label="Compute seconds"
        value={formatGpuSeconds(gpuSeconds)}
        source="Per-stage instrumentation"
        tone={tone}
      />
      <DataValue
        label="Marginal cost"
        value={formatUsd(marginal)}
        source="Allocation ledger estimate"
        tone={tone}
      />
      <DataValue
        label="Total run cost"
        value={formatUsd(total)}
        source="Prewarm→cooldown estimate"
        tone={tone}
      />
      <DataValue
        label="Reconciliation"
        value={reconciliation ?? "not reported"}
        source="Callback / timeout reconciler"
      />

      <div className={stale || degraded ? "freshness freshness-stale" : "freshness"}>
        <Glyph name="clock" />
        <div>
          <span>
            {freshness.source === undefined
              ? "No server timestamp"
              : `Server sample ${formatClockTime(freshness.source)}`}
            {freshness.ageSeconds !== undefined && ` · ${freshness.ageSeconds.toFixed(0)} s old`}
          </span>
          <span className="freshness-reason">{freshness.reason}</span>
          {degraded && (
            <span className="freshness-reason" role="alert">
              Event stream disconnected — reconnect attempt {formatCount(retryCount)}. Values below are the
              last received, not the current state.
            </span>
          )}
          {inFlight && <span className="freshness-reason">In flight: {inFlight}</span>}
        </div>
      </div>
    </section>
  );
}
