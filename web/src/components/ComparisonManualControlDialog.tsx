import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useMemo, useRef, useState } from "react";
import { artifactUrl } from "../lib/api";
import {
  type ComparisonCell,
  type ComparisonDetail,
  type ManualDirection,
  type ManualSession,
  strictManualContractReason,
} from "../lib/comparisons";
import { cn } from "../lib/utils";
import { MediaFrame } from "./MediaFrame";
import { StatusPill } from "./Primitives";

const DIRECTIONS: { direction: ManualDirection; label: string; symbol: string }[] = [
  { direction: "up", label: "Move up", symbol: "↑" },
  { direction: "forward", label: "Move forward", symbol: "⇡" },
  { direction: "left", label: "Move left", symbol: "←" },
  { direction: "right", label: "Move right", symbol: "→" },
  { direction: "down", label: "Move down", symbol: "↓" },
  { direction: "back", label: "Move back", symbol: "⇣" },
];

const KEY_DIRECTIONS: Record<string, ManualDirection> = {
  ArrowUp: "up",
  ArrowDown: "down",
  ArrowLeft: "left",
  ArrowRight: "right",
};

function isEditableTarget(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    Boolean(target.closest("input, textarea, select, button, a, video, [contenteditable='true']"))
  );
}

function commandIsActive(session: ManualSession | undefined): boolean {
  const command = session?.commands?.at(-1);
  return Boolean(session?.active_command || command?.status === "queued" || command?.status === "running");
}

function useDocumentHidden() {
  const [hidden, setHidden] = useState(() => document.hidden);
  useEffect(() => {
    const update = () => setHidden(document.hidden);
    document.addEventListener("visibilitychange", update);
    return () => document.removeEventListener("visibilitychange", update);
  }, []);
  return hidden;
}

/** Play a returned de-echoed segment once, then retain its final real frame. */
function ManualSegmentPlayer({
  sourceUrl,
  sourceAlt,
  frames,
  active,
  hidden,
}: {
  sourceUrl?: string;
  sourceAlt: string;
  frames: { frame_index: number; url: string }[];
  active: boolean;
  hidden: boolean;
}) {
  const signature = frames.map((frame) => `${frame.frame_index}:${frame.url}`).join("|");
  const [index, setIndex] = useState(-1);

  useEffect(() => {
    if (frames.length === 0) {
      setIndex(-1);
      return;
    }
    setIndex(0);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || hidden) {
      setIndex(frames.length - 1);
      return;
    }
    let current = 0;
    const timer = window.setInterval(() => {
      current += 1;
      if (current >= frames.length) {
        window.clearInterval(timer);
        setIndex(frames.length - 1);
        return;
      }
      setIndex(current);
    }, 200);
    return () => window.clearInterval(timer);
  }, [frames.length, hidden, signature]);

  const frame = index >= 0 ? frames[Math.min(index, frames.length - 1)] : undefined;
  return (
    <div className="freeplay-stage" aria-busy={active} aria-label="Manual branch frame">
      <MediaFrame
        src={artifactUrl(frame?.url ?? sourceUrl)}
        alt={frame ? `Generated manual branch frame ${frame.frame_index}` : sourceAlt}
        className="freeplay-media"
        emptyReason="No committed source frame was provided for this branch."
      />
      {active && (
        <div className="generating-overlay" role="status" aria-live="polite">
          <strong>Generating segment</strong>
          <span>One command is active. Further movement is disabled until it reaches a terminal state.</span>
        </div>
      )}
      <div className="stage-label">
        <span className="tabular-nums">
          {frames.length > 0 ? `${frames.length} committed post-conditioning frames` : "source frame"}
        </span>
      </div>
    </div>
  );
}

export function ComparisonManualControlDialog({
  open,
  onOpenChange,
  comparison,
  cell,
  session,
  preparing,
  error,
  returnFocusTo,
  onCommand,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  comparison: ComparisonDetail;
  cell: ComparisonCell;
  session?: ManualSession;
  preparing?: boolean;
  error?: string;
  returnFocusTo?: HTMLElement;
  onCommand: (direction: ManualDirection) => Promise<void>;
}) {
  const surfaceRef = useRef<HTMLDivElement>(null);
  const heldKeys = useRef(new Set<string>());
  const commandLatch = useRef(false);
  const [dispatching, setDispatching] = useState(false);
  const [commandError, setCommandError] = useState<string>();
  const [elapsedMs, setElapsedMs] = useState(0);
  const active = commandIsActive(session);
  const pageHidden = useDocumentHidden();
  const lastCommand = session?.commands?.at(-1);
  const frames = useMemo(
    () => [...(lastCommand?.frames ?? [])].sort((a, b) => a.frame_index - b.frame_index),
    [lastCommand?.frames],
  );
  const profile = comparison.world.manual;
  const manualUnavailableReason = strictManualContractReason(comparison.world);
  const strictManualProfile = manualUnavailableReason === undefined;
  const controlsEnabled = Boolean(session && !preparing && !active && !dispatching && !pageHidden && strictManualProfile);

  useEffect(() => {
    // An accepted command retains the latch until its durable state becomes
    // terminal. This closes the gap where two distinct keydowns arrive before
    // React has painted the disabled controls.
    if (!active && !dispatching) commandLatch.current = false;
  }, [active, dispatching]);

  useEffect(() => {
    if (!active) {
      setElapsedMs(0);
      return;
    }
    const started = Date.now();
    const timer = window.setInterval(() => setElapsedMs(Date.now() - started), 100);
    return () => window.clearInterval(timer);
  }, [active]);

  useEffect(() => {
    if (!open) heldKeys.current.clear();
  }, [open]);

  const requestCommand = async (direction: ManualDirection) => {
    if (!controlsEnabled || document.hidden || commandLatch.current) return;
    commandLatch.current = true;
    setDispatching(true);
    setCommandError(undefined);
    try {
      await onCommand(direction);
    } catch (reason) {
      setCommandError(reason instanceof Error ? reason.message : "The command was not accepted.");
      commandLatch.current = false;
    } finally {
      setDispatching(false);
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content
          className="freeplay-dialog comparison-control-dialog"
          aria-describedby="comparison-control-description"
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            surfaceRef.current?.focus();
          }}
          onCloseAutoFocus={(event) => {
            if (returnFocusTo && document.contains(returnFocusTo)) {
              event.preventDefault();
              returnFocusTo.focus();
            }
          }}
        >
          <header className="dialog-header">
            <div>
              <p className="eyebrow">Unscored manual branch</p>
              <Dialog.Title className="text-balance">Keyboard steering</Dialog.Title>
              <Dialog.Description id="comparison-control-description" className="dialog-description text-pretty">
                A new image-conditioned branch from this cell’s latest committed image and associated state snapshot.
                It is not a hidden-state restore or a continuation of the policy rollout.
              </Dialog.Description>
            </div>
            <Dialog.Close asChild>
              <button type="button" className="button button-secondary">
                Close control
              </button>
            </Dialog.Close>
          </header>

          <div className="freeplay-layout">
            <div
              ref={surfaceRef}
              tabIndex={0}
              className="min-w-0 outline-none focus-visible:ring-2 focus-visible:ring-accent"
              aria-label="Keyboard steering surface. Use arrow keys for one command at a time."
              onBlur={() => heldKeys.current.clear()}
              onKeyDown={(event) => {
                const direction = KEY_DIRECTIONS[event.key];
                if (!direction || event.repeat || isEditableTarget(event.target) || heldKeys.current.has(event.key)) return;
                if (!controlsEnabled) return;
                event.preventDefault();
                heldKeys.current.add(event.key);
                void requestCommand(direction);
              }}
              onKeyUp={(event) => {
                // Key release deliberately sends no request. It only permits a
                // later distinct keydown after the current command is terminal.
                heldKeys.current.delete(event.key);
              }}
            >
              <ManualSegmentPlayer
                sourceUrl={session?.source.url}
                sourceAlt="Latest committed source frame for this manual branch"
                frames={frames}
                active={active || dispatching}
                hidden={pageHidden}
              />
            </div>

            <aside className="freeplay-controls" aria-busy={preparing || active || dispatching}>
              <h2 className="text-balance">Branch details</h2>
              <dl className="state-readout tabular-nums">
                <div><dt>Selected cell</dt><dd>{cell.id}</dd></div>
                <div><dt>Policy · seed</dt><dd>{cell.policy} · {cell.seed}</dd></div>
                <div><dt>Source frame SHA</dt><dd>{session?.source.sha256 ?? "preparing"}</dd></div>
                <div><dt>Target profile</dt><dd>{comparison.world.id}</dd></div>
                <div><dt>Contract</dt><dd>{profile.action_rows} actions → {profile.structural_frames} returned → {profile.post_conditioning_frames} displayed</dd></div>
              </dl>

              {preparing && <p className="request-state" role="status">Freezing the branch source. This does not run inference.</p>}
              {!strictManualProfile && (
                <p className="inline-error" role="alert">
                  {manualUnavailableReason}
                </p>
              )}
              {pageHidden && <p className="request-state" role="status">Commands are unavailable while this page is hidden.</p>}
              {error && <p className="inline-error" role="alert">{error}</p>}
              {commandError && <p className="inline-error" role="alert">{commandError}</p>}
              {lastCommand && lastCommand.error != null ? (
                <p className="inline-error" role="alert">
                  {typeof lastCommand.error === "string" ? lastCommand.error : "The manual command ended without a usable segment."}
                </p>
              ) : null}
              {active && (
                <p className="request-state" role="status">
                  Generating {lastCommand?.direction ?? "command"} · {(elapsedMs / 1000).toFixed(1)} s client wall time
                </p>
              )}

              <p id="comparison-command-help" className="control-note text-pretty">
                Use this focused surface’s arrow keys, or one directional button. Key repeat and keyup submit nothing; there is no command queue.
              </p>
              <div className="dpad" aria-label="Manual world-model commands">
                {DIRECTIONS.map(({ direction, label, symbol }) => (
                  <button
                    key={direction}
                    type="button"
                    disabled={!controlsEnabled}
                    onClick={() => void requestCommand(direction)}
                    aria-label={label}
                    aria-describedby="comparison-command-help"
                  >
                    <span aria-hidden="true">{symbol}</span>
                  </button>
                ))}
              </div>

              <fieldset disabled className="mt-4" aria-label="Graded labels unavailable">
                <legend>Graded labels</legend>
                <p className="control-note text-pretty">
                  Integrity, collision, progress, and binary pass/fail: Unavailable — judge calibration not run.
                </p>
              </fieldset>
              <p className="control-note text-pretty">Outcome: not scored.</p>
              <StatusPill status={session?.status ?? (preparing ? "queued" : "blocked")}>
                {session?.status ?? (preparing ? "preparing" : "unavailable")}
              </StatusPill>
            </aside>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
