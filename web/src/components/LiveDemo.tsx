import * as Dialog from "@radix-ui/react-dialog";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  api,
  artifactUrl,
  type DemoDirection,
  type DemoMode,
  type DemoSession,
  type DemoStatus,
} from "../lib/api";
import { cn } from "../lib/utils";
import { MediaFrame } from "./MediaFrame";
import { EmptyState, Panel, StatusPill } from "./Primitives";
import { InferenceBreakdown } from "./InferenceBreakdown";
import { Assessment } from "./DemoJudgePanel";
import { JudgeLogs } from "./JudgeLogs";

const DEFAULT_PROMPT = "Put the pot to the left of the purple item.";
const ACTIVE_STATES = new Set(["queued", "initializing", "running"]);

export type DemoSourceVideo = {
  id: string;
  title: string;
};

function errorMessage(reason: unknown, fallback: string) {
  if (reason instanceof ApiError) return reason.reason ?? reason.message;
  return reason instanceof Error ? reason.message : fallback;
}

function isLiveState(state: string | undefined) {
  return ACTIVE_STATES.has(state ?? "");
}

function sessionFrames(session: DemoSession) {
  const urls = [
    ...(session.frames?.map((frame) => frame.url) ?? []),
    ...(session.frame_urls ?? []),
  ]
    .map(artifactUrl)
    .filter((value): value is string => Boolean(value));
  return [...new Set(urls)];
}

function requestId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function")
    return crypto.randomUUID();
  return `demo-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function timingEntries(session: DemoSession) {
  const latest = session.world_model === "cosmos" ? session.frames?.find((frame) => typeof frame.timings_ms?.world === "number") : session.frames?.at(-1);
  return Object.entries(latest?.timings_ms ?? {}).filter(
    (entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1]),
  );
}

function metadataLabel(value: unknown) {
  if (typeof value === "string") return value;
  if (!value || typeof value !== "object" || Array.isArray(value))
    return "not reported";
  try {
    const text = JSON.stringify(value);
    return text.length > 120 ? `${text.slice(0, 117)}…` : text;
  } catch {
    return "not reported";
  }
}

function restoreFocusOnClose(open: boolean, returnFocusTo?: HTMLElement) {
  const wasOpen = useRef(false);
  useEffect(() => {
    if (!open) return;
    wasOpen.current = true;
    return () => {
      if (
        !wasOpen.current ||
        !returnFocusTo ||
        !document.contains(returnFocusTo)
      )
        return;
      wasOpen.current = false;
      queueMicrotask(() => returnFocusTo.focus());
    };
  }, [open, returnFocusTo]);
}

function DialogShell({
  open,
  onOpenChange,
  children,
  returnFocusTo,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: React.ReactNode;
  returnFocusTo?: HTMLElement;
}) {
  const closing = useRef(false);
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(next) => {
        closing.current = !next;
        onOpenChange(next);
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content
          className="freeplay-dialog live-demo-dialog max-w-6xl mx-auto"
          aria-describedby="demo-dialog-description"
          onCloseAutoFocus={(event) => {
            // Replacing the creation form with the session dialog must not
            // pull focus back behind the new modal during its mount.
            if (open && !closing.current) {
              event.preventDefault();
              return;
            }
            if (returnFocusTo && document.contains(returnFocusTo)) {
              event.preventDefault();
              returnFocusTo.focus();
            }
          }}
        >
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

const COMMANDS: { direction: DemoDirection; label: string; symbol: string }[] =
  [
    { direction: "up", label: "Move up", symbol: "↑" },
    { direction: "forward", label: "Move forward", symbol: "↑↑" },
    { direction: "left", label: "Move left", symbol: "←" },
    { direction: "right", label: "Move right", symbol: "→" },
    { direction: "down", label: "Move down", symbol: "↓" },
    { direction: "back", label: "Move back", symbol: "↓↓" },
  ];

function DemoSessionSurface({
  initialSession,
}: {
  initialSession: DemoSession;
}) {
  const [session, setSession] = useState(initialSession);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [commandBusy, setCommandBusy] = useState(false);
  const [commandSteps, setCommandSteps] = useState(1);
  const commandInFlight = useRef(false);
  const open = true;

  useEffect(() => {
    setSession(initialSession);
    setLoading(true);
    setError(undefined);
    let mounted = true;
    const refresh = async () => {
      try {
        const next = await api.demoSession(initialSession.id);
        if (mounted) {
          setSession(next);
          setError(undefined);
        }
      } catch (reason) {
        if (mounted)
          setError(
            errorMessage(reason, "The live evaluation could not be refreshed."),
          );
      } finally {
        if (mounted) setLoading(false);
      }
    };
    void refresh();
    return () => {
      mounted = false;
    };
  }, [initialSession]);

  // The stream replays committed frames; reconnecting never dispatches work.
  useEffect(() => {
    if (["completed", "blocked", "error"].includes(session.state)) return;
    const stream = new EventSource(`/api/demo/sessions/${encodeURIComponent(session.id)}/events`);
    stream.addEventListener("snapshot", (event) => {
      try { setSession(JSON.parse((event as MessageEvent).data)); setError(undefined); }
      catch { setError("Could not read generation update."); }
    });
    stream.onerror = () => setError("Generation updates disconnected; reconnecting to saved progress…");
    return () => stream.close();
  }, [session.id, session.state]);

  const frames = useMemo(() => sessionFrames(session), [session]);
  const latest = session.frames?.at(-1);
  const timings = timingEntries(session);
  const canCommand =
    session.mode === "manual" &&
    session.state === "ready" &&
    session.remaining_steps !== 0 &&
    !commandBusy;

  const command = async (direction: DemoDirection) => {
    if (!canCommand || commandInFlight.current) return;
    commandInFlight.current = true;
    setCommandBusy(true);
    setError(undefined);
    try {
      const next = await api.demoCommand(session.id, {
        direction,
        steps: commandSteps,
        request_id: requestId(),
      });
      setSession(next);
      announceDemoSessionsChanged();
    } catch (reason) {
      setError(errorMessage(reason, "The manual command was not accepted."));
    } finally {
      commandInFlight.current = false;
      setCommandBusy(false);
    }
  };

  return (
    <>
      <header className="dialog-header">
        <div>
          <p className="eyebrow">Live experimental evaluation · unqualified</p>
          <Dialog.Title className="text-balance">
            {session.title || "Live evaluation"}
          </Dialog.Title>
          <Dialog.Description
            id="demo-dialog-description"
            className="dialog-description text-pretty"
          >
            {session.world_model === "cosmos" ? "Cosmos generated this video from the supplied manual action sequence." : session.mode === "policy"
              ? "OpenVLA chooses each action; IRASim predicts the next view. No task-success score is assigned."
              : "Each button sends one bounded directional action. Buttons never repeat from a held key."}
          </Dialog.Description>
        </div>
        <Dialog.Close asChild>
          <button type="button" className="button button-secondary">
            Close evaluation
          </button>
        </Dialog.Close>
      </header>

      <div className="freeplay-layout" aria-busy={loading || commandBusy}>
        <section
          className="freeplay-stage"
          aria-label="Persisted live evaluation frame"
        >
          <MediaFrame
            src={session.state === "completed" && session.latest_video_url ? artifactUrl(session.latest_video_url) : frames.at(-1)}
            alt={`Latest persisted frame for ${session.title || session.id}`}
            className="freeplay-media"
            emptyReason={
              loading
                ? "Loading persisted evaluation frames…"
                : "No persisted frame has arrived for this evaluation yet."
            }
          />
          <div className="stage-label">
            <StatusPill status={session.state}>{session.state}</StatusPill>
            <span className="tabular-nums">
              {frames.length} persisted frame{frames.length === 1 ? "" : "s"}
            </span>
          </div>
          <div className="generation-progress">
            <div className="policy-progress" role="progressbar" aria-label="Generated world-model steps" aria-valuemin={0} aria-valuemax={session.mode === "policy" ? (session.requested_steps ?? 4) : (session.max_steps ?? 8)} aria-valuenow={session.completed_steps ?? 0}>
              <span style={{ transform: `scaleX(${Math.min(1, (session.completed_steps ?? 0) / (session.mode === "policy" ? (session.requested_steps ?? 4) : (session.max_steps ?? 8)))})` }} />
            </div>
            <p className="judge-muted tabular-nums" role="status">{isLiveState(session.state) ? `Generating step ${(session.completed_steps ?? 0) + 1}` : session.state === "completed" ? "Generation complete" : "Ready"} · {session.completed_steps ?? 0} frames generated</p>
            <p className="judge-muted text-pretty">Each update is a newly generated frame. Progress advances only when a step finishes.</p>
          </div>
        </section>

        <aside className="freeplay-controls">
          <p className="control-note text-pretty">
            Task: {session.prompt || "Not recorded"}
          </p>
          {artifactUrl(session.latest_video_url) && (
            <a
              className="button button-secondary"
              href={artifactUrl(session.latest_video_url)}
              download
            >
              Download recording
            </a>
          )}
          {session.source?.mode ===
            "fresh_image_reconditioned_not_exact_checkpoint_restore" && (
            <p className="control-note text-pretty">
              New branch from the source’s verified final image. It is
              image-reconditioned, not an exact checkpoint resume.
            </p>
          )}
          {session.error && (
            <p className="inline-error" role="alert">
              {session.error}
            </p>
          )}
          {error && (
            <p className="inline-error" role="alert">
              {error}
            </p>
          )}

          {session.mode === "manual" && (
            <fieldset
              className="mt-4"
              disabled={!canCommand}
              aria-describedby="manual-command-help"
            >
              <legend className="text-sm text-[var(--text-strong)]">
                Steer manually
              </legend>
              <label
                className="block mt-2 text-sm text-[var(--text-muted)]"
                htmlFor={`command-steps-${session.id}`}
              >
                Steps per command
              </label>
              <select
                id={`command-steps-${session.id}`}
                className="task-prompt-input w-full mt-1"
                value={commandSteps}
                onChange={(event) =>
                  setCommandSteps(Number(event.target.value))
                }
              >
                {[1, 2, 3, 4].map((steps) => (
                  <option key={steps} value={steps}>
                    {steps}
                  </option>
                ))}
              </select>
              <p id="manual-command-help" className="control-note mt-2">
                Commands are limited to 1–4 steps. Controls are unavailable
                while a command is running.
              </p>
              <div
                className="grid grid-cols-3 gap-2"
                aria-label="Manual directional controls"
              >
                {COMMANDS.map((item) => (
                  <button
                    key={item.direction}
                    type="button"
                    className="button button-secondary min-h-12"
                    onClick={() => void command(item.direction)}
                    aria-label={item.label}
                  >
                    <span aria-hidden="true">{item.symbol}</span>
                    <span>{item.label.replace("Move ", "")}</span>
                  </button>
                ))}
              </div>
            </fieldset>
          )}

          <dl className="state-readout mt-4">
            <div>
              <dt>State</dt>
              <dd>
                <StatusPill status={session.state}>{session.state}</StatusPill>
              </dd>
            </div>
            <div>
              <dt>Mode</dt>
              <dd>
                {session.policy_label ||
                  (session.mode === "policy"
                    ? "OpenVLA"
                    : "Manual directional action")}
              </dd>
            </div>
            <div>
              <dt>Frames</dt>
              <dd className="tabular-nums">{frames.length}</dd>
            </div>
            <div>
              <dt>Live steps</dt>
              <dd className="tabular-nums">
                {session.completed_steps ?? 0} /{" "}
                {session.mode === "policy"
                  ? (session.requested_steps ?? "?")
                  : (session.max_steps ?? "?")}
              </dd>
            </div>
            {typeof latest?.timings_ms?.total === "number" && (
              <div>
                <dt>Last model step</dt>
                <dd className="tabular-nums">
                  {(latest.timings_ms.total / 1000).toFixed(2)} s
                </dd>
              </div>
            )}
          </dl>
          <details className="mt-3 text-sm">
            <summary>Model details & timings</summary>
            <dl className="state-readout mt-3">
              <div>
                <dt>Model</dt>
                <dd title={metadataLabel(latest?.model)}>
                  {metadataLabel(latest?.model)}
                </dd>
              </div>
              <div>
                <dt>Revision</dt>
                <dd title={metadataLabel(latest?.revision)}>
                  {metadataLabel(latest?.revision)}
                </dd>
              </div>
              {timings.map(([name, value]) => (
                <div key={name}>
                  <dt>{name}</dt>
                  <dd className="tabular-nums">{Math.round(value)} ms</dd>
                </div>
              ))}
            </dl>
          </details>
        </aside>
      </div>
      {session.judgment && <section aria-label="Saved judge assessment"><p className="eyebrow">LoRA-post-trained rollout judge</p><JudgeLogs judgment={session.judgment} /><Assessment judgment={session.judgment} /><InferenceBreakdown judgment={session.judgment} /></section>}
    </>
  );
}

export function DemoSessionDialog({
  session,
  open,
  onOpenChange,
  returnFocusTo,
}: {
  session: DemoSession;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  returnFocusTo?: HTMLElement;
}) {
  restoreFocusOnClose(open, returnFocusTo);
  return (
    <DialogShell
      open={open}
      onOpenChange={onOpenChange}
      returnFocusTo={returnFocusTo}
    >
      <DemoSessionSurface initialSession={session} />
    </DialogShell>
  );
}

/** Dispatches a small, local event so the saved-session list refreshes after a mutation. */
export function announceDemoSessionsChanged() {
  window.dispatchEvent(new Event("demo-sessions-changed"));
}

export function NewEvaluationDialog({
  open,
  onOpenChange,
  sourceVideo,
  returnFocusTo,
  onSessionCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceVideo?: DemoSourceVideo;
  returnFocusTo?: HTMLElement;
  onSessionCreated?: (session: DemoSession) => void;
}) {
  const [mode, setMode] = useState<DemoMode>(sourceVideo ? "manual" : "policy");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [steps, setSteps] = useState(4);
  const [status, setStatus] = useState<DemoStatus>();
  const [statusError, setStatusError] = useState<string>();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string>();
  const [created, setCreated] = useState<DemoSession>();
  const creating = useRef(false);

  restoreFocusOnClose(open, returnFocusTo);

  useEffect(() => {
    if (!open || created) return;
    let mounted = true;
    setMode(sourceVideo ? "manual" : "policy");
    setPrompt(DEFAULT_PROMPT);
    setSteps(sourceVideo ? 1 : 4);
    setStatus(undefined);
    setStatusError(undefined);
    setError(undefined);
    void api
      .demoStatus()
      .then((next) => {
        if (mounted) setStatus(next);
      })
      .catch((reason) => {
        if (mounted)
          setStatusError(
            errorMessage(
              reason,
              "The live demo availability could not be checked.",
            ),
          );
      });
    return () => {
      mounted = false;
    };
  }, [open, sourceVideo?.id, created]);

  useEffect(() => {
    if (mode === "manual") setSteps(1);
  }, [mode]);

  useEffect(() => {
    if (!open) setCreated(undefined);
  }, [open]);

  const create = async () => {
    const task = prompt.trim();
    if (!task || submitting || creating.current || !status?.available) return;
    creating.current = true;
    setSubmitting(true);
    setError(undefined);
    try {
      const next = await api.createDemoSession({
        prompt: task,
        mode: sourceVideo ? "manual" : mode,
        steps: sourceVideo || mode === "manual" ? 1 : steps,
        ...(sourceVideo ? { source_video_id: sourceVideo.id } : {}),
      });
      setCreated(next);
      onSessionCreated?.(next);
      announceDemoSessionsChanged();
    } catch (reason) {
      setError(
        errorMessage(reason, "The new evaluation could not be started."),
      );
    } finally {
      creating.current = false;
      setSubmitting(false);
    }
  };

  if (created && open) {
    return (
      <DemoSessionDialog
        session={created}
        open
        returnFocusTo={returnFocusTo}
        onOpenChange={onOpenChange}
      />
    );
  }

  const unavailableReason = statusError ?? status?.reason;
  const disabledReason = !prompt.trim()
    ? "Enter a task instruction to start this evaluation."
    : !status && !statusError
      ? "Checking whether the live runtime is available…"
      : !status?.available
        ? (unavailableReason ?? "The live runtime is unavailable.")
        : undefined;
  const canStart =
    Boolean(status?.available) && prompt.trim().length > 0 && !submitting;
  return (
    <DialogShell
      open={open}
      onOpenChange={onOpenChange}
      returnFocusTo={returnFocusTo}
    >
      <header className="dialog-header">
        <div>
          <p className="eyebrow">Interactive experimental demo · unqualified</p>
          <Dialog.Title className="text-balance">New evaluation</Dialog.Title>
          <Dialog.Description
            id="demo-dialog-description"
            className="dialog-description text-pretty"
          >
            {sourceVideo
              ? `Start a new manual branch from “${sourceVideo.title}”. The server verifies and reconditions its final image; this is not an exact checkpoint resume.`
              : "Choose the existing OpenVLA controller or steer one bounded manual experiment."}
          </Dialog.Description>
        </div>
        <Dialog.Close asChild>
          <button type="button" className="button button-secondary">
            Close
          </button>
        </Dialog.Close>
      </header>

      <form
        className="grid gap-4 max-w-3xl"
        onSubmit={(event) => {
          event.preventDefault();
          void create();
        }}
        aria-busy={submitting}
      >
        <fieldset disabled={submitting || Boolean(sourceVideo)}>
          <legend className="text-sm text-[var(--text-strong)]">
            Evaluation mode
          </legend>
          <div className="grid gap-2 mt-2">
            <label className="flex items-start gap-2 text-sm text-[var(--text)]">
              <input
                type="radio"
                name="demo-mode"
                value="policy"
                checked={mode === "policy"}
                onChange={() => setMode("policy")}
              />
              <span>
                <b>Run the existing OpenVLA controller</b>
                <br />
                <span className="text-[var(--text-muted)]">
                  Runs the selected bounded number of policy steps.
                </span>
              </span>
            </label>
            <label className="flex items-start gap-2 text-sm text-[var(--text)]">
              <input
                type="radio"
                name="demo-mode"
                value="manual"
                checked={mode === "manual"}
                onChange={() => setMode("manual")}
              />
              <span>
                <b>Steer manually</b>
                <br />
                <span className="text-[var(--text-muted)]">
                  Starts a session; directional commands are sent only when you
                  click a control.
                </span>
              </span>
            </label>
          </div>
        </fieldset>

        <div>
          <label
            className="block text-sm text-[var(--text-strong)]"
            htmlFor="demo-prompt"
          >
            {mode === "manual" ? "Goal / note" : "Task instruction"}
          </label>
          <input
            id="demo-prompt"
            className="task-prompt-input w-full mt-1"
            value={prompt}
            maxLength={200}
            required
            aria-invalid={!prompt.trim()}
            aria-describedby={!prompt.trim() ? "demo-availability" : undefined}
            onChange={(event) => setPrompt(event.target.value)}
          />
          {mode === "manual" && (
            <p className="control-note mt-2">
              This note labels the session. Your directional commands—not this
              text—control the prediction.
            </p>
          )}
        </div>
        {mode === "policy" && !sourceVideo ? (
          <div>
            <label
              className="block text-sm text-[var(--text-strong)]"
              htmlFor="demo-steps"
            >
              Bounded controller steps
            </label>
            <select
              id="demo-steps"
              className="task-prompt-input w-full mt-1"
              value={steps}
              onChange={(event) => setSteps(Number(event.target.value))}
            >
              {[1, 2, 3, 4, 5, 6, 7, 8].map((count) => (
                <option key={count} value={count}>
                  {count} step{count === 1 ? "" : "s"}
                </option>
              ))}
            </select>
            <p className="control-note mt-2">
              The existing controller is limited to 1–8 steps.
            </p>
          </div>
        ) : (
          <p className="control-note">
            Manual sessions do not issue a movement until you click a
            directional control. Each command is limited to 1–4 steps.
          </p>
        )}
        {disabledReason && (
          <p id="demo-availability" className="control-note" role="status">
            {disabledReason}
          </p>
        )}
        {error && (
          <p className="inline-error" role="alert">
            {error}
          </p>
        )}
        <div className="flex flex-wrap gap-3">
          <button
            type="submit"
            className="button button-primary"
            disabled={!canStart}
            aria-describedby={!canStart ? "demo-availability" : undefined}
          >
            {submitting
              ? "Starting evaluation…"
              : sourceVideo
                ? "Start manual image branch"
                : mode === "policy"
                  ? "Run OpenVLA evaluation"
                  : "Start manual session"}
          </button>
          {status && (
            <StatusPill status={status.health}>
              {status.available ? "runtime available" : status.health}
            </StatusPill>
          )}
        </div>
      </form>
    </DialogShell>
  );
}

export function LiveDemoPanel({ limit = 3, historyOnly = false }: { limit?: number; historyOnly?: boolean }) {
  const [status, setStatus] = useState<DemoStatus>();
  const [sessions, setSessions] = useState<DemoSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [newOpen, setNewOpen] = useState(false);
  const [selected, setSelected] = useState<DemoSession>();
  const newButton = useRef<HTMLButtonElement>(null);
  const selectedButton = useRef<HTMLButtonElement>();

  const load = useCallback(async () => {
    try {
      const [nextStatus, response] = await Promise.all([
        api.demoStatus(),
        api.demoSessions(),
      ]);
      setStatus(nextStatus);
      setSessions(Array.isArray(response.sessions) ? response.sessions : []);
      setError(undefined);
    } catch (reason) {
      setError(
        errorMessage(reason, "The live evaluation list could not be loaded."),
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const refresh = () => {
      void load();
    };
    window.addEventListener("demo-sessions-changed", refresh);
    return () => window.removeEventListener("demo-sessions-changed", refresh);
  }, [load]);

  const jobsRunning = sessions.some((session) => isLiveState(session.state));
  useEffect(() => {
    if (!jobsRunning) return;
    const timer = window.setInterval(() => void load(), 1_000);
    return () => window.clearInterval(timer);
  }, [jobsRunning, load]);

  return (
    <Panel
      title={historyOnly ? "Past runs" : "Live evaluations"}
      action={
        !historyOnly && <button
          ref={newButton}
          type="button"
          className="button button-primary"
          onClick={() => setNewOpen(true)}
        >
          New evaluation
        </button>
      }
    >
      {!historyOnly && <p className="text-pretty text-sm text-[var(--text-muted)] mt-3">Generate a fresh rollout and watch each world-model step arrive.</p>}
      {!historyOnly && status && !status.available && <p className="control-note mt-3">Live generation is currently unavailable. Saved runs remain accessible.</p>}
      {error && (
        <p className="inline-error" role="alert">
          {error}
        </p>
      )}
      {loading ? (
        <p className="control-note mt-4" role="status">
          Loading saved live evaluations…
        </p>
      ) : sessions.length === 0 ? (
        <EmptyState className="mt-4">
          No saved live evaluations yet. Start a bounded experiment to see its
          persisted frames here.
        </EmptyState>
      ) : (
        <ul
          className="mt-4 grid gap-3 md:grid-cols-3"
          aria-label="Saved live evaluations"
        >
          {sessions.slice(0, limit).map((session) => (
            <li
              key={session.id}
              className="flex min-w-0 flex-col gap-3 border border-[var(--line-2)] p-3"
            >
              {sessionFrames(session).at(-1) ? (
                <img
                  className="aspect-[4/3] w-full object-contain bg-black"
                  loading="lazy"
                  src={sessionFrames(session).at(-1)}
                  alt={`Latest saved frame: ${session.title || session.id}`}
                />
              ) : (
                <p className="control-note">No frame saved for this session.</p>
              )}
              <div className="min-w-0">
                <p className="text-pretty text-sm text-[var(--text-strong)]">
                  {session.title || session.prompt || session.id}
                </p>
                <p className="text-pretty text-sm text-[var(--text-muted)]">
                  {session.world_model === "cosmos" ? "Cosmos · supplied actions" : session.mode === "policy"
                    ? "Existing OpenVLA controller"
                    : "Manual steering"}{" "}
                  · {sessionFrames(session).length} persisted frames
                </p>
              </div>
              <div className="mt-auto flex flex-wrap items-center gap-3">
                <StatusPill status={session.state}>{session.state}</StatusPill>
                <button
                  type="button"
                  className="button button-secondary"
                  onClick={(event) => {
                    selectedButton.current = event.currentTarget;
                    setSelected(session);
                  }}
                >
                  Open evaluation
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {sessions.length > limit && (
        <p className="mt-3 text-sm">
          <a href="/results" className="underline">
            View all {sessions.length} saved evaluations →
          </a>
        </p>
      )}
      <NewEvaluationDialog
        open={newOpen}
        onOpenChange={setNewOpen}
        returnFocusTo={newButton.current ?? undefined}
      />
      {selected && (
        <DemoSessionDialog
          session={selected}
          open
          returnFocusTo={selectedButton.current}
          onOpenChange={(next) => {
            if (!next) setSelected(undefined);
          }}
        />
      )}
    </Panel>
  );
}
