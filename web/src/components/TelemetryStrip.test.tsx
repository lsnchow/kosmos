import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Telemetry } from "../lib/api";
import { STALE_AFTER_SECONDS, TelemetryStrip, freshnessOf } from "./TelemetryStrip";

const NOW = Date.UTC(2026, 8, 21, 9, 0, 0);

function telemetry(overrides: Partial<Telemetry> = {}): Telemetry {
  return {
    source: "application_ledger",
    timestamp: NOW / 1000,
    fresh_at: NOW / 1000,
    stale: false,
    reconciliation_status: "reconciled",
    platform_queue: 320,
    active_replicas: 87,
    desired_replicas: 100,
    gpu_seconds: 1_842.5,
    marginal_estimated_usd: 6.12,
    total_estimated_usd: 10.84,
    ...overrides,
  };
}

function renderStrip(overrides: Partial<Parameters<typeof TelemetryStrip>[0]> = {}) {
  render(
    <TelemetryStrip
      streamStatus="open"
      retryCount={0}
      now={NOW}
      telemetry={telemetry()}
      {...overrides}
    />,
  );
}

describe("telemetry freshness", () => {
  it("reads the server's own timestamp, not the browser clock", () => {
    const verdict = freshnessOf(telemetry({ fresh_at: (NOW - 30_000) / 1000 }), NOW);
    expect(verdict.stale).toBe(true);
    expect(verdict.ageSeconds).toBe(30);
    expect(verdict.reason).toMatch(/Older than the 1 s cadence/);
  });

  it("is fresh inside the declared cadence window", () => {
    const verdict = freshnessOf(telemetry({ fresh_at: (NOW - 2_000) / 1000 }), NOW);
    expect(verdict.stale).toBe(false);
    expect(verdict.ageSeconds).toBe(2);
  });

  it("goes stale just past the threshold", () => {
    const verdict = freshnessOf(
      telemetry({ fresh_at: (NOW - (STALE_AFTER_SECONDS + 1) * 1000) / 1000 }),
      NOW,
    );
    expect(verdict.stale).toBe(true);
  });

  it("honours the server's own stale flag even for a recent sample", () => {
    const verdict = freshnessOf(telemetry({ stale: true }), NOW);
    expect(verdict.stale).toBe(true);
    expect(verdict.reason).toMatch(/server marked this sample stale/i);
  });

  it("treats a sample with no server timestamp as stale rather than current", () => {
    const verdict = freshnessOf(telemetry({ timestamp: undefined, fresh_at: undefined }), NOW);
    expect(verdict.stale).toBe(true);
    expect(verdict.reason).toMatch(/carries no server timestamp/i);
  });

  it("does not call a missing telemetry object stale", () => {
    const verdict = freshnessOf(undefined, NOW);
    expect(verdict.stale).toBe(false);
    expect(verdict.reason).toMatch(/No telemetry event has arrived yet/i);
  });
});

describe("<TelemetryStrip />", () => {
  it("renders real platform values with their source attribution", () => {
    renderStrip();
    expect(screen.getByText("320")).toBeInTheDocument();
    expect(screen.getByText("87 / 100")).toBeInTheDocument();
    expect(screen.getByText("1,842.5 GPU-s")).toBeInTheDocument();
    expect(screen.getByText("$6.12")).toBeInTheDocument();
    expect(screen.getByText("$10.84")).toBeInTheDocument();
    expect(screen.getByText("Baseten async queue status")).toBeInTheDocument();
    expect(screen.getByText("Chain deployment API / metrics export")).toBeInTheDocument();
  });

  it("shows a stale badge and reason when the server sample is old", () => {
    renderStrip({ telemetry: telemetry({ fresh_at: (NOW - 42_000) / 1000 }) });
    expect(screen.getByText("stale")).toBeInTheDocument();
    expect(screen.getByText(/42 s old/)).toBeInTheDocument();
    expect(screen.getByText(/Older than the 1 s cadence this feed declares\./)).toBeInTheDocument();
  });

  it("shows a stale badge when the server itself flags the sample", () => {
    renderStrip({ telemetry: telemetry({ stale: true }) });
    expect(screen.getByText("stale")).toBeInTheDocument();
  });

  it("never fabricates a zero for an unreported metric", () => {
    renderStrip({
      telemetry: telemetry({
        platform_queue: null,
        active_replicas: null,
        desired_replicas: null,
        gpu_seconds: null,
        marginal_estimated_usd: null,
        total_estimated_usd: null,
        platform_queue_status: "unavailable",
      }),
    });
    expect(screen.queryByText("0")).not.toBeInTheDocument();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    expect(screen.queryByText("0 / 0")).not.toBeInTheDocument();
    expect(screen.getAllByText("unavailable").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Unavailable").length).toBe(2);
  });

  it("never relabels ledger counts as measured platform queue depth", () => {
    renderStrip({
      run: { id: "run-1", completed: 412, total: 1500 },
      telemetry: telemetry({ platform_queue: null, platform_queue_status: "unavailable" }),
    });
    expect(screen.getByText("412 / 1,500")).toBeInTheDocument();
    expect(
      screen.getByText("Chain queue route unverified · never ledger counts"),
    ).toBeInTheDocument();
  });

  it("shows an em dash rather than 0 when the ledger reported no counts", () => {
    renderStrip({ run: {} });
    expect(screen.getByText("— / —")).toBeInTheDocument();
  });

  it("reports the reconciliation status", () => {
    renderStrip();
    expect(screen.getByText("Reconciliation")).toBeInTheDocument();
    expect(screen.getByText("reconciled")).toBeInTheDocument();
  });

  it("says reconciliation was not reported instead of implying success", () => {
    renderStrip({ telemetry: telemetry({ reconciliation_status: undefined }) });
    expect(screen.getByText("not reported")).toBeInTheDocument();
  });

  it("makes a dropped event stream visible instead of looking like a slow run", () => {
    renderStrip({ streamStatus: "degraded", retryCount: 3 });
    expect(screen.getByText("degraded")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(
      /Event stream disconnected — reconnect attempt 3/i,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/last received, not the current state/i);
  });

  it("shows in-flight ledger counts separately from platform telemetry", () => {
    renderStrip({ ledgerState: { nonterminal_status_counts: { running: 88, planned: 1000 } } });
    expect(screen.getByText(/In flight: running 88 · planned 1000/)).toBeInTheDocument();
  });
});
