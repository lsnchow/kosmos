import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Episode, SegmentCompletedEvent, Telemetry } from "../lib/api";
import { FakeEventSource } from "../test/harness";
import { reconnectDelayMs, useRunStream, type EventSourceFactory } from "./useRunStream";

function Probe({
  runId,
  onSegment,
  onTelemetry,
  onEpisodes,
  onMalformed,
  runTerminal,
}: {
  runId?: string;
  onSegment?: (segment: SegmentCompletedEvent) => void;
  onTelemetry?: (telemetry: Telemetry) => void;
  onEpisodes?: (episodes: Episode[]) => void;
  onMalformed?: (message: string) => void;
  runTerminal?: boolean;
}) {
  const factory: EventSourceFactory = (url) => new FakeEventSource(url) as unknown as EventSource;
  const state = useRunStream(
    runId,
    { onSegment, onTelemetry, onEpisodes, onMalformed },
    { factory, runTerminal },
  );
  return (
    <div>
      <span data-testid="status">{state.status}</span>
      <span data-testid="retries">{state.retryCount}</span>
    </div>
  );
}

describe("useRunStream", () => {
  it("is idle with no run selected", () => {
    FakeEventSource.reset();
    render(<Probe />);
    expect(screen.getByTestId("status")).toHaveTextContent("idle");
    expect(FakeEventSource.instances).toHaveLength(0);
  });

  it("connects to the run's event endpoint", () => {
    FakeEventSource.reset();
    render(<Probe runId="run-1" />);
    expect(screen.getByTestId("status")).toHaveTextContent("connecting");
    expect(FakeEventSource.latest()?.url).toBe("/api/runs/run-1/events");
  });

  it("goes open on the first snapshot", () => {
    FakeEventSource.reset();
    const onEpisodes = vi.fn();
    render(<Probe runId="run-1" onEpisodes={onEpisodes} />);
    act(() => FakeEventSource.latest()?.emit("snapshot", { episodes: [{ episode_id: "ep-1" }] }));
    expect(screen.getByTestId("status")).toHaveTextContent("open");
    expect(onEpisodes).toHaveBeenCalledWith([{ episode_id: "ep-1" }]);
  });

  it("surfaces a dropped stream as degraded instead of silently doing nothing", () => {
    // The old handler was `() => setEventReceivedAt(prior => prior)`, a
    // deliberate no-op: a dead stream looked exactly like a slow run.
    FakeEventSource.reset();
    render(<Probe runId="run-1" />);
    act(() => FakeEventSource.latest()?.fail());
    expect(screen.getByTestId("status")).toHaveTextContent("degraded");
    expect(screen.getByTestId("retries")).toHaveTextContent("1");
  });

  it("counts consecutive failures and resets the count on a good message", () => {
    FakeEventSource.reset();
    render(<Probe runId="run-1" />);
    act(() => FakeEventSource.latest()?.fail());
    expect(screen.getByTestId("retries")).toHaveTextContent("1");
    act(() => FakeEventSource.latest()?.emit("snapshot", {}));
    expect(screen.getByTestId("status")).toHaveTextContent("open");
    expect(screen.getByTestId("retries")).toHaveTextContent("0");
  });

  it("backs off between reconnection attempts and caps the delay", () => {
    expect(reconnectDelayMs(0)).toBe(1000);
    expect(reconnectDelayMs(1)).toBe(2000);
    expect(reconnectDelayMs(2)).toBe(4000);
    expect(reconnectDelayMs(99)).toBe(15000);
    expect(reconnectDelayMs(99)).toBe(reconnectDelayMs(4));
  });

  it("schedules a fresh connection after a closed source", () => {
    vi.useFakeTimers();
    try {
      FakeEventSource.reset();
      render(<Probe runId="run-1" />);
      act(() => FakeEventSource.latest()?.fail());
      expect(FakeEventSource.instances).toHaveLength(1);
      // One failure has been recorded, so the scheduled delay is the second step.
      act(() => void vi.advanceTimersByTime(reconnectDelayMs(1)));
      expect(FakeEventSource.instances).toHaveLength(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("treats the end of a finished run's stream as a clean close, not a drop", () => {
    // A server that ends the response leaves EventSource in CONNECTING, so this
    // case is invisible to a readyState check. Showing "degraded" here would put
    // a disconnection warning on screen the moment the burst succeeds.
    FakeEventSource.reset();
    render(<Probe runId="run-1" runTerminal />);
    const source = FakeEventSource.latest();
    act(() => {
      source!.readyState = 0;
      source!.onerror?.(new Event("error"));
    });
    expect(screen.getByTestId("status")).toHaveTextContent("closed");
    expect(screen.getByTestId("retries")).toHaveTextContent("0");
  });

  it("does not reconnect in a loop to a completed run", () => {
    vi.useFakeTimers();
    try {
      FakeEventSource.reset();
      render(<Probe runId="run-1" runTerminal />);
      act(() => FakeEventSource.latest()?.fail());
      act(() => void vi.advanceTimersByTime(60_000));
      expect(FakeEventSource.instances).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("delivers segment_completed events for the wall", () => {
    FakeEventSource.reset();
    const onSegment = vi.fn();
    render(<Probe runId="run-1" onSegment={onSegment} />);
    const payload = {
      episode_id: "ep-1",
      segment_index: 3,
      frame_urls: ["a/0.png"],
      certified_frame_count: 16,
      provenance: "live",
    };
    act(() => FakeEventSource.latest()?.emit("segment_completed", payload));
    expect(onSegment).toHaveBeenCalledWith(payload);
  });

  it("keeps receiving telemetry after the run reaches a terminal status", () => {
    FakeEventSource.reset();
    const onTelemetry = vi.fn();
    render(<Probe runId="run-1" onTelemetry={onTelemetry} />);
    const source = FakeEventSource.latest();
    act(() => source?.emit("snapshot", { run: { id: "run-1", status: "completed" } }));
    // Cost reconciliation and late segments arrive after the run is terminal;
    // closing here is what froze the old dashboard the instant a run finished.
    act(() => source?.emit("snapshot", { telemetry: { total_estimated_usd: 10.9 } }));
    expect(onTelemetry).toHaveBeenLastCalledWith({ total_estimated_usd: 10.9 });
    expect(screen.getByTestId("status")).toHaveTextContent("open");
  });

  it("reports malformed payloads rather than failing silently", () => {
    FakeEventSource.reset();
    const onMalformed = vi.fn();
    render(<Probe runId="run-1" onMalformed={onMalformed} />);
    const source = FakeEventSource.latest();
    act(() => {
      for (const listener of ["snapshot"]) {
        source?.addEventListener(listener, () => undefined);
      }
      source?.emit("snapshot", undefined);
    });
    expect(onMalformed).toHaveBeenCalledWith(
      "The run event stream returned a non-object snapshot.",
    );
  });
});
