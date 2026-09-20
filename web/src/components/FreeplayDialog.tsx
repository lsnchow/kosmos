import * as Dialog from "@radix-ui/react-dialog";
import { AsciiSpinner, Glyph } from "./Terminal";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  ApiError,
  api,
  artifactUrls,
  type FreeplayDirection,
  type FreeplayResponse,
} from "../lib/api";
import {
  formatCount,
  formatMilliseconds,
  pickNumber,
  pickString,
} from "../lib/format";
import { FramePlayer } from "./MediaFrame";
import { StatusPill } from "./Primitives";
import type { WorldVideo } from "./WorldVideoQueue";
import { artifactUrl } from "../lib/api";
import { cn } from "../lib/utils";

type RecordingReadiness = {
  available: boolean;
  reason: string;
  mode?: string;
  binding?: { exact_branch_supported?: boolean };
  source?: { video_id: string; sha256: string } | null;
};

const KEY_DIRECTIONS: Record<string, FreeplayDirection> = {
  ArrowUp: "up",
  ArrowDown: "down",
  ArrowLeft: "left",
  ArrowRight: "right",
};

const DIRECTION_LABELS: Record<FreeplayDirection, string> = {
  up: "up",
  down: "down",
  left: "left",
  right: "right",
  stop: "stop",
};

type Chunk = {
  direction: FreeplayDirection;
  frames: string[];
  frameCount?: number;
  latencyMs?: number;
  clientElapsedMs: number;
  backend?: string;
  resolution?: string;
  reason?: string;
  /** Server-declared per-axis clamp. The client never substitutes its own. */
  actionClamp?: number;
  chunkSize?: number;
};

/**
 * Arrow-key free-play.
 *
 * One keypress is one direction. The server expands it into a constant
 * 16-action chunk and calls the world model, so the client never asserts an
 * action magnitude of its own — the old build sent ±0.08 per axis while the
 * server clamped to ±0.03, i.e. 2.7× what was honoured, and displayed the
 * client's number. Here the clamp is whatever the server echoes back, and if it
 * echoes nothing the UI says so rather than printing a number.
 *
 * The generating wait is real and is shown, not hidden: it is the interesting
 * part. Nothing pre-recorded answers a keypress.
 */
/**
 * The frames' measured size, never the size that was requested.
 *
 * The server reports `requested_resolution` and the frames' own dimensions
 * separately, because they can differ -- a 64px rehearsal frame answering a
 * 480p request is exactly the case that must not be captioned "480p". When the
 * Chain reports no dimensions this returns undefined and the panel says so.
 */
function measuredResolution(
  response: Record<string, unknown>,
): string | undefined {
  const height = pickNumber(response.frame_height);
  const width = pickNumber(response.frame_width);
  if (height === undefined || width === undefined) return undefined;
  return `${width}x${height}`;
}

export function FreeplayDialog({
  open,
  onOpenChange,
  subjectLabel,
  recording,
  returnFocusTo,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  subjectLabel?: string;
  recording?: WorldVideo;
  returnFocusTo?: HTMLButtonElement;
}) {
  const [sessionId, setSessionId] = useState<string>();
  const [chunk, setChunk] = useState<Chunk>();
  const [chunkCount, setChunkCount] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [generating, setGenerating] = useState<FreeplayDirection>();
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState<string>();
  const [unavailable, setUnavailable] = useState<string>();
  const [ignoredPress, setIgnoredPress] = useState(false);
  const [stopAcknowledged, setStopAcknowledged] = useState(false);
  const [readiness, setReadiness] = useState<RecordingReadiness>();
  const [readinessError, setReadinessError] = useState<string>();
  const [readinessAttempt, setReadinessAttempt] = useState(0);
  const active = useRef(open);
  active.current = open;
  const recordingReady =
    !recording ||
    (Boolean(recording.sha256) &&
      readiness?.available === true &&
      readiness.mode === "recording_branch" &&
      readiness.binding?.exact_branch_supported === true &&
      readiness.source?.video_id === recording.id &&
      readiness.source?.sha256 === recording.sha256);
  const canDispatch = useRef(recordingReady);
  canDispatch.current = recordingReady;

  useEffect(() => {
    if (!open || !recording) return;
    const abort = new AbortController();
    setReadiness(undefined);
    setReadinessError(undefined);
    void fetch(
      `/api/freeplay/status?video_id=${encodeURIComponent(recording.id)}`,
      { signal: abort.signal },
    )
      .then(async (response) => {
        if (!response.ok)
          throw new Error(
            "Could not verify interactive availability. No generation was started.",
          );
        const payload = (await response.json()) as RecordingReadiness;
        if (
          typeof payload.available !== "boolean" ||
          typeof payload.reason !== "string"
        ) {
          throw new Error(
            "The server did not return a valid interactive capability. Controls remain disabled.",
          );
        }
        if (!abort.signal.aborted) setReadiness(payload);
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted)
          setReadinessError(
            reason instanceof Error
              ? reason.message
              : "Interactive availability could not be checked.",
          );
      });
    return () => abort.abort();
  }, [open, recording, readinessAttempt]);

  const heldKeys = useRef(new Set<string>());
  const inFlight = useRef(false);
  const queued = useRef<FreeplayDirection>();
  const sessionRef = useRef<string>();
  sessionRef.current = sessionId;

  useEffect(() => {
    active.current = open;
    return () => {
      active.current = false;
      queued.current = undefined;
      heldKeys.current.clear();
    };
  }, [open]);

  const dispatch = useCallback(
    async (direction: FreeplayDirection) => {
      if (!active.current || !canDispatch.current) return;
      if (inFlight.current) {
        // One pending slot, latest wins. Release-to-stop still lands, and a held
        // key cannot pile up an unbounded queue of generation requests.
        queued.current = direction;
        setIgnoredPress(true);
        return;
      }
      inFlight.current = true;
      setGenerating(direction);
      setIgnoredPress(false);
      setError(undefined);
      const startedAt = Date.now();
      let commandCompleted = false;
      try {
        const response: FreeplayResponse = await api.freeplayStep({
          session_id: sessionRef.current,
          direction,
          ...(recording
            ? { video_id: recording.id, source_sha256: recording.sha256 }
            : {}),
        });
        if (!active.current) return;
        if (
          recording &&
          (response.mode !== "recording_branch" ||
            response.source?.video_id !== recording.id ||
            response.source.sha256 !== recording.sha256 ||
            response.binding?.exact_branch_supported !== true)
        ) {
          throw new Error(
            "The response was not bound to this recording. No returned frames were displayed.",
          );
        }
        const frames = artifactUrls(response.frame_urls);
        if (response.session_id) setSessionId(response.session_id);

        // Release-to-stop commands nothing, so it is an acknowledgement rather
        // than a clip. Letting it through here would blank the stage after every
        // keypress and — worse — clear a "no certified backend" message, because
        // the stop call succeeds even when the world model is unreachable.
        if (direction === "stop") {
          commandCompleted = true;
          setStopAcknowledged(true);
          return;
        }

        setUnavailable(undefined);
        setStopAcknowledged(false);
        setChunk({
          direction,
          frames,
          frameCount: pickNumber(response.frame_count),
          latencyMs: pickNumber(response.latency_ms),
          clientElapsedMs: Date.now() - startedAt,
          backend: pickString(response.backend),
          resolution: measuredResolution(response),
          reason: pickString(response.reason),
          actionClamp: pickNumber(response.action_clamp),
          chunkSize: pickNumber(response.chunk_size),
        });
        setChunkCount((prior) => prior + 1);
        setTotalFrames((prior) => prior + frames.length);
        commandCompleted = frames.length > 0;
        if (frames.length === 0) {
          setError(
            pickString(response.reason) ??
              "The world backend returned no frames for this command and gave no reason.",
          );
        }
      } catch (requestError) {
        if (recording) {
          canDispatch.current = false;
          setReadiness({
            available: false,
            reason:
              "The last command failed or could not be verified. No queued movement was sent; check availability before trying again.",
          });
        }
        if (requestError instanceof ApiError && requestError.status === 503) {
          if (recording)
            setReadiness({
              available: false,
              reason:
                requestError.reason ??
                requestError.message ??
                "The recording's interactive backend is unavailable.",
            });
          // No certified world backend. Say that, and show no frame at all.
          setUnavailable(
            requestError.reason ??
              requestError.message ??
              "No certified world backend is configured for free-play.",
          );
          setChunk(undefined);
        } else {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "The free-play request failed.",
          );
        }
      } finally {
        inFlight.current = false;
        setGenerating(undefined);
        const next = queued.current;
        queued.current = undefined;
        if (commandCompleted && next && active.current && canDispatch.current)
          void dispatch(next);
      }
    },
    [recording],
  );

  // A real elapsed counter during the real wait. It is client wall time and is
  // labelled as such; the server's own latency is reported separately.
  useEffect(() => {
    if (!generating) {
      setElapsedMs(0);
      return;
    }
    const startedAt = Date.now();
    const handle = setInterval(() => setElapsedMs(Date.now() - startedAt), 100);
    return () => clearInterval(handle);
  }, [generating]);

  useEffect(() => {
    if (!open || !recordingReady) return;
    const keyDown = (event: KeyboardEvent) => {
      const direction = KEY_DIRECTIONS[event.key];
      if (
        recording &&
        event.target instanceof Element &&
        event.target.closest(
          "video, input, textarea, select, [contenteditable]",
        )
      )
        return;
      if (!direction || event.repeat || heldKeys.current.has(event.key)) return;
      event.preventDefault();
      heldKeys.current.add(event.key);
      void dispatch(direction);
    };
    const keyUp = (event: KeyboardEvent) => {
      if (!heldKeys.current.delete(event.key)) return;
      event.preventDefault();
      void dispatch("stop");
    };
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      heldKeys.current.clear();
    };
  }, [dispatch, open, recordingReady, recording]);

  const buttonHandlers = (direction: FreeplayDirection) => ({
    disabled: !recordingReady,
    onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      void dispatch(direction);
    },
    onPointerUp: () => void dispatch("stop"),
    onPointerLeave: () => void dispatch("stop"),
    onKeyDown: (event: ReactKeyboardEvent<HTMLButtonElement>) => {
      if ((event.key === " " || event.key === "Enter") && !event.repeat)
        void dispatch(direction);
    },
    onKeyUp: (event: ReactKeyboardEvent<HTMLButtonElement>) => {
      if (event.key === " " || event.key === "Enter") void dispatch("stop");
    },
  });

  const frames = chunk?.frames ?? [];

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content
          className={cn("freeplay-dialog", recording && "recording-control-dialog")}
          aria-describedby="freeplay-description"
          onCloseAutoFocus={(event) => {
            if (returnFocusTo) {
              event.preventDefault();
              returnFocusTo.focus();
            }
          }}
        >
          <header className="dialog-header">
            <div>
              <p className="eyebrow">
                Unscored · contributes nothing to the scoreboard
              </p>
              <Dialog.Title className="text-balance">
                {recording ? "Take the controls" : "Free-play control"}
                {!recording && subjectLabel ? ` — ${subjectLabel}` : ""}
              </Dialog.Title>
            </div>
            <Dialog.Close className="icon-button" aria-label="Close free-play">
              <Glyph name="close" />
            </Dialog.Close>
          </header>
          <Dialog.Description
            id="freeplay-description"
            className="dialog-description text-pretty"
          >
            {recording ? (
              "When available, steer a separate branch with arrow keys. Opening this view starts no generation. Your original recording stays unchanged."
            ) : (
              <>
                One keypress is one direction. The server expands it into a
                fixed 16-action chunk and calls the world model, so there is no
                policy and no language model in this loop. Fixed instruction,
                clamped actions, release to stop. Nothing here enters a scored
                comparison.
              </>
            )}
          </Dialog.Description>

          {recording && (
            <div className="recording-readiness" aria-live="polite">
              {!readiness && !readinessError ? (
                <p role="status">
                  Checking this recording’s interactive availability…
                </p>
              ) : (
                <>
                  <strong>
                    {recordingReady
                      ? "Interactive branch available"
                      : "Interactive generation unavailable"}
                  </strong>
                  <p className="text-pretty">
                    {readinessError ??
                      (recordingReady
                        ? "Ready to branch from this recording’s saved checkpoint."
                        : "This recording is not ready to resume. Live controls need a saved world state and a compatible generation backend.")}
                  </p>
                  {readiness && (
                    <details>
                      <summary>Technical details</summary>
                      <p className="text-pretty">{readiness.reason}</p>
                    </details>
                  )}
                  {readiness?.available && !recordingReady && (
                    <p className="text-pretty">
                      The server did not confirm this recording’s exact source
                      binding. Reload the gallery before trying again.
                    </p>
                  )}
                  <button
                    type="button"
                    disabled={Boolean(generating)}
                    className="button button-secondary"
                    onClick={() => setReadinessAttempt(readinessAttempt + 1)}
                  >
                    Check availability again
                  </button>
                </>
              )}
            </div>
          )}

          <div className="freeplay-layout">
            <div className="freeplay-stage">
              {recording && frames.length === 0 ? (
                <div className="recording-source-preview">
                  <p className="eyebrow">
                    {recording.model} · original recording · playback only
                  </p>
                  <video
                    controls
                    muted
                    playsInline
                    preload="metadata"
                    src={
                      recording.video_url.startsWith("/api/artifacts/")
                        ? artifactUrl(recording.video_url)
                        : undefined
                    }
                    aria-label={`Original recording: ${recording.title}`}
                  />
                  <p className="text-pretty">
                    Seeking this video changes playback only. A video frame is
                    not a restorable world-model checkpoint.
                  </p>
                </div>
              ) : unavailable ? (
                <div className="freeplay-unavailable" role="alert">
                  <Glyph name="alert" />
                  <strong>No certified world backend for free-play</strong>
                  <p className="text-pretty">{unavailable}</p>
                  <p className="text-pretty freeplay-unavailable-note">
                    No frame is shown, because there is no frame. A placeholder
                    here would be a claim that a world model answered a keypress
                    when none did.
                  </p>
                </div>
              ) : (
                <FramePlayer
                  frames={frames}
                  alt={
                    chunk
                      ? `Generated frames for the ${DIRECTION_LABELS[chunk.direction]} command`
                      : "No generated free-play frames yet"
                  }
                  className="freeplay-media"
                  emptyReason="Press an arrow key to command a 16-action chunk"
                />
              )}

              {generating && (
                <div
                  className="generating-overlay"
                  role="status"
                  aria-live="polite"
                >
                  <AsciiSpinner className="freeplay-spinner" />
                  <strong>Generating</strong>
                  <span>
                    Inventing the next chunk of video from the{" "}
                    {DIRECTION_LABELS[generating]} command
                  </span>
                  <span className="tabular-nums generating-elapsed">
                    {(elapsedMs / 1000).toFixed(1)} s elapsed
                  </span>
                  <span className="generating-source">client wall time</span>
                </div>
              )}

              {(!recording || chunk) && (
                <div className="stage-label">
                  <span>{chunk?.resolution ?? "resolution not reported"}</span>
                  <span aria-hidden="true">·</span>
                  <span>{chunk?.backend ?? "backend not reported"}</span>
                  <span aria-hidden="true">·</span>
                  <StatusPill status="unqualified">unqualified</StatusPill>
                </div>
              )}
            </div>

            <aside className="freeplay-controls">
              {recording && (
                <h2 className="text-balance">Interactive branch</h2>
              )}
              {recording && !recordingReady && (
                <p className="control-note text-pretty">
                  Directional controls unlock only when the server confirms a
                  supported starting checkpoint.{" "}
                  {frames.length === 0
                    ? "No new frames have been generated."
                    : "Previously returned frames are retained."}
                </p>
              )}
              {!recording && (
                <p className="control-note text-pretty">
                  Arrow keys, or press and hold a button. Releasing sends a stop
                  command. The per-axis clamp is applied by the server; this
                  panel prints only the clamp the server reports.
                </p>
              )}
              {recording && (
                <p className="control-note text-pretty">
                  Each press requests one bounded chunk, not real-time robot
                  motion. Closing this view drops queued commands but does not
                  cancel a chunk already submitted.
                </p>
              )}
              <div
                className="dpad"
                aria-label="Directional world-model commands"
              >
                <span />
                <button
                  type="button"
                  aria-label="Command the arm up"
                  {...buttonHandlers("up")}
                >
                  <Glyph name="arrowUp" />
                </button>
                <span />
                <button
                  type="button"
                  aria-label="Command the arm left"
                  {...buttonHandlers("left")}
                >
                  <Glyph name="arrowLeft" />
                </button>
                <button
                  type="button"
                  disabled={!recordingReady}
                  aria-label="Send a stop command"
                  onClick={() => void dispatch("stop")}
                >
                  <Glyph name="stop" />
                </button>
                <button
                  type="button"
                  aria-label="Command the arm right"
                  {...buttonHandlers("right")}
                >
                  <Glyph name="arrowRight" />
                </button>
                <span />
                <button
                  type="button"
                  aria-label="Command the arm down"
                  {...buttonHandlers("down")}
                >
                  <Glyph name="arrowDown" />
                </button>
                <span />
              </div>

              {(!recording || chunk) && (
                <dl className="state-readout tabular-nums">
                  <div>
                    <dt>Last command</dt>
                    <dd>
                      {stopAcknowledged
                        ? "stop"
                        : chunk
                          ? DIRECTION_LABELS[chunk.direction]
                          : "—"}
                    </dd>
                  </div>
                  <div>
                    <dt>Frames returned</dt>
                    <dd>
                      {formatCount(
                        chunk?.frameCount ?? (frames.length || undefined),
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>Server latency</dt>
                    <dd>{formatMilliseconds(chunk?.latencyMs)}</dd>
                  </div>
                  <div>
                    <dt>Round trip</dt>
                    <dd>{formatMilliseconds(chunk?.clientElapsedMs)}</dd>
                  </div>
                  <div>
                    <dt>Actions per chunk</dt>
                    <dd>{formatCount(chunk?.chunkSize, "not reported")}</dd>
                  </div>
                  <div>
                    <dt>Server clamp / axis</dt>
                    <dd>
                      {chunk?.actionClamp === undefined
                        ? "not reported"
                        : `±${chunk.actionClamp}`}
                    </dd>
                  </div>
                  <div>
                    <dt>Chunks this session</dt>
                    <dd>{formatCount(chunkCount)}</dd>
                  </div>
                  <div>
                    <dt>Frames this session</dt>
                    <dd>{formatCount(totalFrames)}</dd>
                  </div>
                </dl>
              )}

              {ignoredPress && (
                <p className="request-state" role="status">
                  A command arrived mid-generation. The most recent one is
                  queued; the rest were dropped rather than stacked.
                </p>
              )}
              {chunk?.reason && (
                <p className="control-note text-pretty">{chunk.reason}</p>
              )}
              {error && (
                <p className="inline-error" role="alert">
                  {error}
                </p>
              )}
            </aside>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
