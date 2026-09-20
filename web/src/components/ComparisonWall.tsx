import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { artifactUrl } from "../lib/api";
import {
  COMPARISON_HORIZON,
  COMPARISON_INSTRUCTION,
  COMPARISON_POLICIES,
  COMPARISON_SEEDS,
  COMPARISON_TASK,
  comparisonApi,
  comparisonEventUrl,
  comparisonReady,
  newIdempotencyKey,
  strictManualContractReason,
  type ComparisonCell,
  type ComparisonDetail,
  type ComparisonEvent,
  type ComparisonFrame,
  type ComparisonQuote,
  type ComparisonReadiness,
  type ManualCommand,
  type ManualDirection,
  type ManualFrame,
  type ManualSession,
} from "../lib/comparisons";
import { cn } from "../lib/utils";
import { FramePlayer, MediaFrame } from "./MediaFrame";
import { EmptyState, Panel, SourceChip, StatusPill } from "./Primitives";
import { ComparisonManualControlDialog } from "./ComparisonManualControlDialog";

type ComparisonLoad = "loading" | "ready" | "unavailable" | "error";

type ControlState = {
  cell: ComparisonCell;
  trigger: HTMLButtonElement;
  session?: ManualSession;
  preparing: boolean;
  error?: string;
};

function errorMessage(reason: unknown, fallback: string): string {
  if (reason instanceof Error) return reason.message;
  if (typeof reason === "string") return reason;
  if (reason && typeof reason === "object") {
    try { return JSON.stringify(reason); } catch { return fallback; }
  }
  return fallback;
}

/** The promoted wall is strict: no partial or reshaped matrix is rendered as a saved comparison. */
export function comparisonProblem(comparison: ComparisonDetail, presentation: boolean): string | undefined {
  if (
    comparison.schema !== "kosmos-comparison-v1" ||
    comparison.scored !== false ||
    comparison.claim_tier !== "preview" ||
    comparison.task !== COMPARISON_TASK ||
    comparison.task_instruction !== COMPARISON_INSTRUCTION ||
    comparison.horizon !== COMPARISON_HORIZON
  )
    return "The server response does not match the unscored comparison-v1 contract.";
  if (comparison.policies.join("|") !== COMPARISON_POLICIES.join("|"))
    return "The server response did not contain the required OpenVLA, MiniVLA, and Octo-Small column order.";
  if (comparison.seeds.join("|") !== COMPARISON_SEEDS.join("|"))
    return "The server response did not contain the four matched world seeds 101–104.";
  if (!comparison.start?.id || !comparison.start.png_url || !comparison.start.sha256)
    return "The server did not provide the immutable shared start bundle.";
  if (!comparison.world?.id || !comparison.world.profile_sha256)
    return "The server did not provide the immutable world profile identity.";
  if (comparison.cells.length !== 12) return "The server response did not contain exactly twelve fixed cells.";

  const expected = new Set(COMPARISON_POLICIES.flatMap((policy) => COMPARISON_SEEDS.map((seed) => `${policy}|${seed}`)));
  const seen = new Set<string>();
  for (const cell of comparison.cells) {
    const key = `${cell.policy}|${cell.seed}`;
    if (!cell.id || !expected.has(key) || seen.has(key))
      return "The server supplied missing or duplicate policy/seed cells.";
    seen.add(key);
  }
  if (seen.size !== expected.size) return "The server supplied an incomplete matched policy/seed matrix.";
  if (presentation) {
    if (!comparison.presentation || comparison.status !== "completed")
      return "Only a server-promoted completed comparison may be presented as precomputed media.";
    const incomplete = comparison.cells.find(
      (cell) =>
        cell.status !== "completed" ||
        !cell.terminal_received ||
        cell.action_count !== COMPARISON_HORIZON ||
        cell.frame_count !== COMPARISON_HORIZON ||
        cell.frames.length !== COMPARISON_HORIZON,
    );
    if (incomplete)
      return `Promoted cell ${incomplete.id} is not a completed 70-action / 70-frame result.`;
  }
  return undefined;
}

function usePlaybackPaused(element: React.RefObject<HTMLElement>) {
  const [paused, setPaused] = useState(true);
  useEffect(() => {
    const node = element.current;
    if (!node) return;
    const updateHidden = () => setPaused(document.hidden);
    if (typeof IntersectionObserver === "undefined") {
      setPaused(document.hidden);
      document.addEventListener("visibilitychange", updateHidden);
      return () => document.removeEventListener("visibilitychange", updateHidden);
    }
    let intersecting = false;
    const observer = new IntersectionObserver(([entry]) => {
      intersecting = entry.isIntersecting;
      setPaused(document.hidden || !intersecting);
    }, { threshold: 0.15 });
    const visibility = () => setPaused(document.hidden || !intersecting);
    observer.observe(node);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      observer.disconnect();
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [element]);
  return paused;
}

function policyIdentity(readiness: ComparisonReadiness | undefined, policy: string): string {
  const hashes = readiness?.policies.find((item) => item.id === policy)?.identity_hashes;
  if (!hashes) return "not included in the presentation response";
  return Object.entries(hashes).map(([name, hash]) => `${name}: ${hash}`).join(" · ");
}

function frameUrls(cell: ComparisonCell) {
  // Frame IDs—not pixels—are the identity. Two real generated frames can have
  // the same pixels, so deliberately do not image-hash deduplicate here.
  return cell.frames.map((frame) => artifactUrl(frame.url)).filter((url): url is string => Boolean(url));
}

function ComparisonCellTile({
  cell,
  readiness,
  origin,
  onTakeControl,
}: {
  cell: ComparisonCell;
  readiness?: ComparisonReadiness;
  origin: "precomputed" | "prewarmed" | "fresh";
  onTakeControl: (cell: ComparisonCell, trigger: HTMLButtonElement) => void;
}) {
  const ref = useRef<HTMLElement>(null);
  const paused = usePlaybackPaused(ref);
  const urls = frameUrls(cell);
  const manualUnavailableReason = strictManualContractReason(readiness?.world);
  const strictManual = manualUnavailableReason === undefined;
  const failed = ["failed", "interrupted", "ambiguous"].includes(cell.status);

  return (
    <article ref={ref} className="min-w-72 overflow-hidden border border-[var(--line-2)]" aria-label={`${cell.policy}, seed ${cell.seed}, cell ${cell.id}`}>
      <FramePlayer
        frames={urls}
        alt={`Persisted frames for ${cell.policy}, matched world seed ${cell.seed}`}
        className="tile-media w-full"
        paused={paused}
        emptyReason={
          cell.status === "queued" || cell.status === "running"
            ? "Generating — no committed frame has arrived yet"
            : "No committed generated frame was recorded for this cell"
        }
      />
      <div className="tile-caption">
        <div>{cell.policy}</div>
        <span className="tabular-nums">Seed {cell.seed} · {cell.id}</span>
        <StatusPill status={failed ? "failed" : cell.status}>{cell.status}</StatusPill>
        <span>Outcome: not scored</span>
      </div>
      <dl className="tile-meta tabular-nums">
        <div><dt>Native actions</dt><dd>{cell.action_count} / {COMPARISON_HORIZON}</dd></div>
        <div><dt>Committed frames</dt><dd>{cell.frames.length} / {cell.frame_count}</dd></div>
        <div><dt>Origin</dt><dd>{cell.origin ?? origin}</dd></div>
      </dl>
      {Boolean(cell.error) && <p className="inline-error" role="alert">{errorMessage(cell.error, "This cell reported an error.")}</p>}
      <div className="flex flex-wrap gap-2 p-2">
        <button
          type="button"
          className="button button-secondary"
          disabled={!strictManual || !cell.latest_frame}
          title={
            !strictManual
              ? manualUnavailableReason
              : !cell.latest_frame
                ? "Steering needs a committed source frame from this cell."
                : undefined
          }
          onClick={(event) => onTakeControl(cell, event.currentTarget)}
        >
          Take control
        </button>
      </div>
      <details className="p-2 text-xs text-fg-muted">
        <summary>Immutable lineage</summary>
        <p className="mt-2 break-words">Policy identity: {policyIdentity(readiness, cell.policy)}</p>
        <p className="break-words">Attempt: {cell.attempt_id ?? "not reported"}</p>
        <p className="break-words">Latest frame SHA: {cell.latest_frame?.sha256 ?? "not reported"}</p>
      </details>
    </article>
  );
}

function cellFor(comparison: ComparisonDetail, policy: ComparisonCell["policy"], seed: number): ComparisonCell | undefined {
  // This lookup is only for fixed table placement. State and deduplication are
  // always keyed by server-assigned cell id, never by policy/task text.
  return comparison.cells.find((cell) => cell.policy === policy && cell.seed === seed);
}

function compareFrame(existing: ComparisonFrame, incoming: ComparisonFrame) {
  return existing.event_id === incoming.event_id;
}

/** Apply one durable controller event without merging visual duplicates by pixel hash. */
export function applyComparisonEvent(prior: ComparisonDetail, event: ComparisonEvent): ComparisonDetail {
  if (event.comparison_id !== prior.id || event.sequence <= 0) return prior;
  const payload = event.payload;
  const payloadText = (key: string) => typeof payload[key] === "string" ? payload[key] : undefined;
  const payloadNumber = (key: string) => typeof payload[key] === "number" ? payload[key] : undefined;
  const patchCell = (id: string | undefined, patch: (cell: ComparisonCell) => ComparisonCell) => {
    if (!id || !prior.cells.some((cell) => cell.id === id)) return prior;
    return { ...prior, cells: prior.cells.map((cell) => cell.id === id ? patch(cell) : cell), updated_at: new Date().toISOString() };
  };

  if (event.type === "frame") {
    const url = payloadText("url");
    const sha256 = payloadText("sha256");
    const frameIndex = payloadNumber("frame_index");
    if (!url || !sha256 || frameIndex === undefined) return prior;
    const frame: ComparisonFrame = {
      event_id: payloadText("event_id") ?? `stream-${event.sequence}`,
      segment_id: payloadText("segment_id") ?? event.attempt_id ?? "stream",
      frame_index: frameIndex,
      url,
      sha256,
      action: payload.action,
      state: payload.state && typeof payload.state === "object" ? payload.state : undefined,
    };
    return patchCell(event.cell_id, (cell) => {
      if (cell.frames.some((current) => compareFrame(current, frame))) return cell;
      const frames = [...cell.frames, frame].sort((left, right) => left.frame_index - right.frame_index);
      return {
        ...cell,
        frames,
        latest_frame: frames.at(-1) ?? cell.latest_frame,
        frame_count: Math.max(cell.frame_count, frames.length),
        action_count: Math.max(cell.action_count, payloadNumber("action_count") ?? cell.action_count),
      };
    });
  }

  if (event.type === "cell_status" || event.type === "terminal") {
    const status = payloadText("status");
    return patchCell(event.cell_id, (cell) => ({
      ...cell,
      status: status && ["queued", "running", "completed", "failed", "interrupted", "ambiguous"].includes(status)
        ? status as ComparisonCell["status"]
        : cell.status,
      action_count: payloadNumber("action_count") ?? cell.action_count,
      frame_count: payloadNumber("frame_count") ?? cell.frame_count,
      error: payloadText("error") ?? cell.error,
      terminal_received: event.type === "terminal" ? true : cell.terminal_received,
    }));
  }
  if (event.type === "comparison_terminal") {
    const status = payloadText("status");
    if (status && ["completed", "failed", "interrupted", "ambiguous"].includes(status)) {
      return { ...prior, status: status as ComparisonDetail["status"], updated_at: new Date().toISOString() };
    }
  }
  return prior;
}

function appendManualFrame(command: ManualCommand, frame: ManualFrame) {
  const priorFrames = command.frames ?? [];
  if (priorFrames.some((current) => current.frame_index === frame.frame_index && current.sha256 === frame.sha256)) return command;
  const frames = [...priorFrames, frame].sort((left, right) => left.frame_index - right.frame_index);
  return { ...command, frames, frame_count: Math.max(command.frame_count, frames.length) };
}

function applyManualEvent(session: ManualSession, event: ComparisonEvent): ManualSession {
  const payload = event.payload;
  const sessionId = typeof payload.session_id === "string" ? payload.session_id : undefined;
  if (sessionId !== session.id) return session;
  const commandId = typeof payload.command_id === "string" ? payload.command_id : undefined;
  if (!commandId) return session;
  const commands = [...(session.commands ?? [])];
  const index = commands.findIndex((command) => command.id === commandId);
  if (event.type === "manual_command_started" && index >= 0) {
    commands[index] = { ...commands[index], status: "running" };
    return { ...session, active_command: commandId, status: "running", commands };
  }
  if (event.type === "manual_frame" && index >= 0) {
    const url = typeof payload.url === "string" ? payload.url : undefined;
    const sha256 = typeof payload.sha256 === "string" ? payload.sha256 : undefined;
    const frameIndex = typeof payload.frame_index === "number" ? payload.frame_index : undefined;
    if (!url || !sha256 || frameIndex === undefined) return session;
    commands[index] = appendManualFrame(commands[index], { frame_index: frameIndex, url, sha256 });
    return { ...session, commands };
  }
  if (event.type === "manual_terminal" && index >= 0) {
    const status = typeof payload.status === "string" ? payload.status : "ambiguous";
    commands[index] = {
      ...commands[index],
      status: ["completed", "failed", "interrupted", "ambiguous"].includes(status)
        ? status as ManualCommand["status"]
        : "ambiguous",
      frame_count: typeof payload.frame_count === "number" ? payload.frame_count : commands[index].frame_count,
      error: payload.error ?? commands[index].error,
    };
    const nextSource = payload.next_source;
    const source = nextSource && typeof nextSource === "object" &&
      typeof (nextSource as Record<string, unknown>).event_id === "string" &&
      typeof (nextSource as Record<string, unknown>).url === "string" &&
      typeof (nextSource as Record<string, unknown>).sha256 === "string"
      ? {
          event_id: (nextSource as Record<string, unknown>).event_id as string,
          url: (nextSource as Record<string, unknown>).url as string,
          sha256: (nextSource as Record<string, unknown>).sha256 as string,
          state: (nextSource as Record<string, unknown>).state as ManualSession["source"]["state"],
        }
      : session.source;
    return {
      ...session,
      active_command: null,
      status: commands[index].status === "completed"
        ? "ready"
        : commands[index].status === "queued" || commands[index].status === "running"
          ? "ambiguous"
          : commands[index].status,
      commands,
      source,
    };
  }
  return session;
}

function useComparisonEvents(
  comparisonId: string | undefined,
  sequence: number,
  onEvent: (event: ComparisonEvent) => void,
  onReload: () => Promise<void>,
) {
  const sequenceRef = useRef(sequence);
  const eventRef = useRef(onEvent);
  const reloadRef = useRef(onReload);
  const [state, setState] = useState<"idle" | "connecting" | "open" | "reconnecting" | "closed">("idle");
  sequenceRef.current = sequence;
  eventRef.current = onEvent;
  reloadRef.current = onReload;

  useEffect(() => {
    if (!comparisonId) {
      setState("idle");
      return;
    }
    let source: EventSource | undefined;
    let retry: number | undefined;
    let disposed = false;
    const handle = (message: Event) => {
      try {
        const event = JSON.parse((message as MessageEvent<string>).data) as ComparisonEvent;
        if (!event || typeof event.sequence !== "number" || typeof event.type !== "string") return;
        eventRef.current(event);
        setState("open");
      } catch {
        // A malformed stream record does not become a result; reload from the
        // durable endpoint rather than guessing a missing frame.
        void reloadRef.current();
      }
    };
    const connect = () => {
      if (disposed) return;
      setState("connecting");
      source = new EventSource(comparisonEventUrl(comparisonId, sequenceRef.current));
      ["comparison", "stage", "frame", "heartbeat", "terminal", "cell_status", "comparison_terminal", "manual_session_created", "manual_command_started", "manual_frame", "manual_terminal"]
        .forEach((type) => source?.addEventListener(type, handle));
      source.addEventListener("complete", () => {
        source?.close();
        setState("closed");
      });
      source.onerror = () => {
        if (disposed) return;
        source?.close();
        setState("reconnecting");
        retry = window.setTimeout(() => {
          void reloadRef.current().finally(connect);
        }, 1_000);
      };
    };
    connect();
    return () => {
      disposed = true;
      source?.close();
      if (retry) window.clearTimeout(retry);
    };
  }, [comparisonId]);
  return state;
}

export function ComparisonWall() {
  const [load, setLoad] = useState<ComparisonLoad>("loading");
  const [readiness, setReadiness] = useState<ComparisonReadiness>();
  const [comparison, setComparison] = useState<ComparisonDetail>();
  const [message, setMessage] = useState<string>();
  const [quote, setQuote] = useState<ComparisonQuote>();
  const [quoteBusy, setQuoteBusy] = useState(false);
  const [launchBusy, setLaunchBusy] = useState(false);
  const [control, setControl] = useState<ControlState>();
  const [sessions, setSessions] = useState<Record<string, ManualSession>>({});
  const sequence = useRef(0);
  const launchKey = useRef<string>();

  const loadPresentation = useCallback(async () => {
    setLoad("loading");
    setMessage(undefined);
    const [readinessResult, presentationResult] = await Promise.allSettled([
      comparisonApi.readiness(),
      comparisonApi.presentation(),
    ]);
    if (readinessResult.status === "fulfilled") setReadiness(readinessResult.value);
    if (presentationResult.status === "fulfilled") {
      const response = presentationResult.value;
      if (response.presentation) {
        const problem = comparisonProblem(response.presentation, true);
        if (problem) {
          setLoad("unavailable");
          setMessage(problem);
          return;
        }
        setComparison(response.presentation);
        setLoad("ready");
        return;
      }
      setLoad("unavailable");
      setMessage(response.reason ?? "No promoted precomputed matched set is available.");
      return;
    }
    setLoad("error");
    setMessage(
      readinessResult.status === "rejected"
        ? errorMessage(readinessResult.reason, "The comparison readiness endpoint could not be loaded.")
        : errorMessage(presentationResult.reason, "The promoted comparison could not be loaded."),
    );
  }, []);

  useEffect(() => { void loadPresentation(); }, [loadPresentation]);

  const reloadComparison = useCallback(async () => {
    if (!comparison) return;
    const next = await comparisonApi.detail(comparison.id);
    const problem = comparisonProblem(next, false);
    if (problem) throw new Error(problem);
    setComparison(next);
  }, [comparison]);

  const handleEvent = useCallback((event: ComparisonEvent) => {
    if (event.sequence <= sequence.current) return;
    sequence.current = event.sequence;
    if (["manual_command_started", "manual_frame", "manual_terminal"].includes(event.type)) {
      setSessions((prior) => {
        const next = Object.fromEntries(
          Object.entries(prior).map(([cellId, session]) => [cellId, applyManualEvent(session, event)]),
        );
        return next;
      });
      setControl((prior) => prior?.session
        ? { ...prior, session: applyManualEvent(prior.session, event) }
        : prior);
      return;
    }
    setComparison((prior) => prior ? applyComparisonEvent(prior, event) : prior);
  }, []);

  const stream = useComparisonEvents(comparison?.id, sequence.current, handleEvent, reloadComparison);

  const prepareQuote = async () => {
    setQuoteBusy(true);
    setMessage(undefined);
    try {
      const response = await comparisonApi.quote();
      setReadiness(response.readiness);
      setQuote(response.quote);
    } catch (reason) {
      setMessage(errorMessage(reason, "A fresh comparison quote could not be prepared."));
    } finally {
      setQuoteBusy(false);
    }
  };

  const launch = async () => {
    if (!quote) return;
    setLaunchBusy(true);
    setMessage(undefined);
    launchKey.current ??= newIdempotencyKey("comparison");
    try {
      const response = await comparisonApi.create({ quote_id: quote.id, idempotency_key: launchKey.current });
      const problem = comparisonProblem(response.comparison, false);
      if (problem) throw new Error(problem);
      sequence.current = 0;
      setComparison(response.comparison);
      setQuote(undefined);
      setLoad("ready");
    } catch (reason) {
      setMessage(errorMessage(reason, "The fresh matched comparison was not admitted."));
    } finally {
      setLaunchBusy(false);
    }
  };

  const takeControl = async (cell: ComparisonCell, trigger: HTMLButtonElement) => {
    if (!comparison) return;
    const cached = sessions[cell.id];
    // Once a branch has an accepted manual command, reopening the same selected
    // cell continues that durable branch. Before any command it is safe to
    // reuse only when the selected policy frame is still the frozen source.
    if (cached && ((cached.commands?.length ?? 0) > 0 || cached.source.event_id === cell.latest_frame?.event_id)) {
      setControl({ cell, trigger, session: cached, preparing: false });
      return;
    }
    // This POST happens only in the explicit button handler. The server merely
    // freezes an image/state source; it does not run policy or world inference.
    setControl({ cell, trigger, preparing: true });
    try {
      const session = await comparisonApi.createManualSession(comparison.id, { cell_id: cell.id });
      setSessions((prior) => ({ ...prior, [cell.id]: session }));
      setControl((prior) => prior?.cell.id === cell.id ? { ...prior, session, preparing: false, error: undefined } : prior);
    } catch (reason) {
      setControl((prior) => prior?.cell.id === cell.id
        ? { ...prior, preparing: false, error: errorMessage(reason, "The branch source could not be frozen.") }
        : prior);
    }
  };

  const command = async (direction: ManualDirection) => {
    if (!control?.session) throw new Error("The branch source is not ready.");
    const response = await comparisonApi.command(control.session.id, {
      direction,
      idempotency_key: newIdempotencyKey("manual"),
    });
    const session = control.session;
    const existing = session.commands ?? [];
    const commands = existing.some((item) => item.id === response.command.id)
      ? existing.map((item) => item.id === response.command.id ? response.command : item)
      : [...existing, response.command];
    const next = { ...session, status: "running" as const, active_command: response.command.id, commands: commands.map((item) => ({ ...item, frames: item.frames ?? [] })) };
    setSessions((prior) => ({ ...prior, [control.cell.id]: next }));
    setControl((prior) => prior ? { ...prior, session: next } : prior);
  };

  const currentProblem = comparison ? comparisonProblem(comparison, comparison.presentation) : undefined;
  const readyForFresh = comparisonReady(readiness);
  const identity = comparison
    ? comparison.origin ?? (comparison.presentation ? "precomputed matched set" : "fresh matched set")
    : "comparison release blocked";
  const manualUnavailableReason = comparison ? strictManualContractReason(comparison.world) : undefined;

  return (
    <Panel
      title="Matched control wall"
      id="matched-control-wall"
      action={<SourceChip>{identity}</SourceChip>}
    >
      {load === "loading" && <div className="empty-state" role="status">Loading the promoted comparison record…</div>}
      {message && <p className="inline-error" role="alert">{message}</p>}

      {comparison && !currentProblem && (
        <>
          <div className="status-strip" role="status">
            <p><strong>{comparison.presentation ? "Precomputed matched Baseten set" : "Fresh matched set"}</strong> · 3 policies × 4 matched seeds · unscored demo</p>
            <span className="tabular-nums">stream: {stream}</span>
          </div>
          {manualUnavailableReason && (
            <p className="inline-error" role="status">{manualUnavailableReason}</p>
          )}
          <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,1fr)_12rem]">
            <div>
              <p className="eyebrow">Shared task and starting bundle</p>
              <h3 className="text-balance text-fg-strong">{comparison.task_instruction}</h3>
              <p className="text-pretty text-sm text-fg-muted">
                One immutable source state for every cell. The source image is separate from generated output frames.
              </p>
              <dl className="state-readout tabular-nums">
                <div><dt>Start ID</dt><dd>{comparison.start.id}</dd></div>
                <div><dt>Start SHA</dt><dd>{comparison.start.sha256}</dd></div>
                <div><dt>World profile</dt><dd>{comparison.world.id}</dd></div>
                <div><dt>Profile SHA</dt><dd>{comparison.world.profile_sha256}</dd></div>
              </dl>
            </div>
            <MediaFrame
              src={artifactUrl(comparison.start.png_url)}
              alt="The one shared source frame for this matched control wall"
              className="w-full aspect-square"
              emptyReason="The source image URI was not available."
            />
          </div>
          <div className="table-wrap mt-4">
            <table aria-label="Three policies across four matched world seeds">
              <thead>
                <tr>
                  <th scope="col">World seed</th>
                  {COMPARISON_POLICIES.map((policy) => <th scope="col" key={policy}>{policy}</th>)}
                </tr>
              </thead>
              <tbody>
                {COMPARISON_SEEDS.map((seed) => (
                  <tr key={seed}>
                    <th scope="row">Seed {seed}</th>
                    {COMPARISON_POLICIES.map((policy) => {
                      const cell = cellFor(comparison, policy, seed);
                      return (
                        <td key={policy} className="align-top">
                          {cell ? <ComparisonCellTile cell={cell} readiness={readiness} origin={comparison.origin ?? (comparison.presentation ? "precomputed" : "fresh")} onTakeControl={(candidate, trigger) => void takeControl(candidate, trigger)} /> : (
                            <EmptyState>Missing server-assigned cell descriptor.</EmptyState>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-pretty text-sm text-fg-muted">
            Generated-only playback uses committed frames. It does not measure physical robot duration, and it does not assign a score, ranking, success count, or judge result.
          </p>
        </>
      )}

      {load !== "loading" && !comparison && (
        <EmptyState>
          No promoted 12-cell set is available. The legacy live evaluation and recording archive remain below; they are not substituted into this controlled wall.
        </EmptyState>
      )}

      <div className="mt-4 border-t border-[var(--line-1)] pt-4">
        <h3 className="text-balance text-fg-strong">Fresh comparison</h3>
        <p className="text-pretty text-sm text-fg-muted">
          A new run is deliberate. First request a server-side cost/readiness quote; only an approved quote can admit work.
        </p>
        {!readyForFresh && readiness && (
          <ul className="mt-3 list-disc pl-5 text-sm text-fg-muted">
            {[...readiness.blockers, ...readiness.mechanical.reasons, ...readiness.demo_quality.reasons]
              .filter((reason, index, all) => all.indexOf(reason) === index)
              .map((reason) => <li key={reason}>{reason}</li>)}
          </ul>
        )}
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button type="button" className="button button-secondary" disabled={!readyForFresh || quoteBusy || launchBusy} onClick={() => void prepareQuote()}>
            {quoteBusy ? "Preparing quote…" : "Prepare fresh matched-set quote"}
          </button>
          {quote && (
            <>
              <span className="tabular-nums text-sm text-fg-muted">Reserved: ${quote.reservation_usd.toFixed(2)} · expires {quote.expires_at}</span>
              <button type="button" className="button button-primary" disabled={launchBusy} onClick={() => void launch()}>
                {launchBusy ? "Admitting run…" : "Run fresh matched set"}
              </button>
            </>
          )}
        </div>
      </div>

      {control && comparison && (
        <ComparisonManualControlDialog
          open
          onOpenChange={(open) => { if (!open) setControl(undefined); }}
          comparison={comparison}
          cell={control.cell}
          session={control.session}
          preparing={control.preparing}
          error={control.error}
          returnFocusTo={control.trigger}
          onCommand={command}
        />
      )}
    </Panel>
  );
}
