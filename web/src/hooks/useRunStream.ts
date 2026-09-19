import { useCallback, useEffect, useRef, useState } from "react";
import {
  isRecord,
  type AnalysisCell,
  type Episode,
  type Run,
  type SegmentCompletedEvent,
  type Telemetry,
} from "../lib/api";

/**
 * Connection state for the run event stream.
 *
 * A dropped SSE stream used to be invisible: `onerror` was an identity no-op, so
 * a disconnected stream looked exactly like a slow run. `degraded` exists so the
 * UI can say which of the two it is.
 */
export type StreamStatus = "idle" | "connecting" | "open" | "degraded" | "closed";

export type StreamState = {
  status: StreamStatus;
  /** Consecutive failed connection attempts; resets on a successful message. */
  retryCount: number;
  lastMessageAt?: number;
  lastErrorAt?: number;
};

type Handlers = {
  onRun?: (run: Run) => void;
  onEpisodes?: (episodes: Episode[]) => void;
  onAnalysis?: (cells: AnalysisCell[]) => void;
  onTelemetry?: (telemetry: Telemetry) => void;
  onSegment?: (segment: SegmentCompletedEvent) => void;
  onMalformed?: (message: string) => void;
};

/**
 * Backoff for reconnection. `EventSource` reconnects on its own for a clean
 * disconnect, but not after the browser closes the source, so an explicit
 * schedule is needed for the demo case where the API restarts mid-run.
 */
const RECONNECT_DELAYS_MS = [1000, 2000, 4000, 8000, 15000];

export function reconnectDelayMs(retryCount: number): number {
  return RECONNECT_DELAYS_MS[Math.min(retryCount, RECONNECT_DELAYS_MS.length - 1)];
}

export type EventSourceFactory = (url: string) => EventSource;

/**
 * Subscribe to `/api/runs/{id}/events`.
 *
 * The stream is kept open after the run reaches a terminal status: telemetry,
 * cost reconciliation and late segment events all continue to arrive, and
 * closing early is what made the old dashboard freeze the moment a run
 * finished. The caller decides when to stop by clearing `runId`.
 */
export function useRunStream(
  runId: string | undefined,
  handlers: Handlers,
  options?: {
    factory?: EventSourceFactory;
    enabled?: boolean;
    /**
     * True when the run already has a terminal status. The contract is that the
     * stream keeps running past terminal, but a server that closes it instead is
     * finishing cleanly, not dropping. Reporting that as "degraded" would put a
     * red disconnection warning on screen the moment the burst succeeds.
     */
    runTerminal?: boolean;
  },
) {
  const [state, setState] = useState<StreamState>({ status: "idle", retryCount: 0 });
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;
  // The factory is held in a ref rather than being a dependency: an inline
  // `factory={(url) => ...}` prop is a new function on every render, and as a
  // dependency that would tear down and reopen the stream on every render.
  const factoryRef = useRef(options?.factory);
  factoryRef.current = options?.factory;
  const enabled = options?.enabled ?? true;
  const terminalRef = useRef(false);
  terminalRef.current = options?.runTerminal ?? false;
  const retryRef = useRef(0);

  const bumpRetry = useCallback(() => {
    retryRef.current += 1;
    setState({ status: "degraded", retryCount: retryRef.current, lastErrorAt: Date.now() });
  }, []);

  useEffect(() => {
    if (!runId || !enabled) {
      setState({ status: "idle", retryCount: 0 });
      return;
    }
    const create = factoryRef.current ?? ((url: string) => new EventSource(url));
    let source: EventSource | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      setState((prior) => ({ ...prior, status: prior.retryCount > 0 ? "degraded" : "connecting" }));
      source = create(`/api/runs/${encodeURIComponent(runId)}/events`);

      source.addEventListener("snapshot", (event) => {
        retryRef.current = 0;
        setState({ status: "open", retryCount: 0, lastMessageAt: Date.now() });
        let snapshot: unknown;
        try {
          snapshot = JSON.parse((event as MessageEvent<string>).data);
        } catch {
          handlersRef.current.onMalformed?.("The run event stream returned malformed snapshot data.");
          return;
        }
        if (!isRecord(snapshot)) {
          handlersRef.current.onMalformed?.("The run event stream returned a non-object snapshot.");
          return;
        }
        if (isRecord(snapshot.run)) handlersRef.current.onRun?.(snapshot.run as Run);
        if (Array.isArray(snapshot.episodes)) handlersRef.current.onEpisodes?.(snapshot.episodes as Episode[]);
        if (isRecord(snapshot.analysis) && Array.isArray(snapshot.analysis.cells)) {
          handlersRef.current.onAnalysis?.(snapshot.analysis.cells as AnalysisCell[]);
        }
        if (isRecord(snapshot.telemetry)) handlersRef.current.onTelemetry?.(snapshot.telemetry as Telemetry);
      });

      source.addEventListener("segment_completed", (event) => {
        retryRef.current = 0;
        setState({ status: "open", retryCount: 0, lastMessageAt: Date.now() });
        try {
          const parsed = JSON.parse((event as MessageEvent<string>).data);
          if (isRecord(parsed)) handlersRef.current.onSegment?.(parsed as SegmentCompletedEvent);
        } catch {
          handlersRef.current.onMalformed?.("The run event stream returned a malformed segment event.");
        }
      });

      source.onerror = () => {
        if (disposed) return;
        if (terminalRef.current) {
          // Expected end of stream for a finished run. A server that ends the
          // response leaves `EventSource` in CONNECTING, not CLOSED, so this
          // cannot be detected from `readyState` — and left alone the browser
          // would reconnect to a completed run forever.
          source?.close();
          setState({ status: "closed", retryCount: 0 });
          return;
        }
        // `EventSource` retries CONNECTING itself; only a CLOSED source needs us.
        bumpRetry();
        if (source?.readyState === 2) {
          source.close();
          timer = setTimeout(connect, reconnectDelayMs(retryRef.current));
        }
      };
    };

    connect();
    return () => {
      disposed = true;
      if (timer !== undefined) clearTimeout(timer);
      source?.close();
      setState({ status: "closed", retryCount: retryRef.current });
    };
  }, [bumpRetry, enabled, runId]);

  return state;
}
