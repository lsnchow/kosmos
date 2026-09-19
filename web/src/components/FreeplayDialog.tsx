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
import { ApiError, api, artifactUrls, type FreeplayDirection, type FreeplayResponse } from "../lib/api";
import { formatCount, formatMilliseconds, pickNumber, pickString } from "../lib/format";
import { FramePlayer } from "./MediaFrame";
import { StatusPill } from "./Primitives";

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
function measuredResolution(response: Record<string, unknown>): string | undefined {
  const height = pickNumber(response.frame_height);
  const width = pickNumber(response.frame_width);
  if (height === undefined || width === undefined) return undefined;
  return `${width}x${height}`;
}

export function FreeplayDialog({
  open,
  onOpenChange,
  subjectLabel,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  subjectLabel?: string;
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

  const heldKeys = useRef(new Set<string>());
  const inFlight = useRef(false);
  const queued = useRef<FreeplayDirection>();
  const sessionRef = useRef<string>();
  sessionRef.current = sessionId;

  const dispatch = useCallback(async (direction: FreeplayDirection) => {
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
    try {
      const response: FreeplayResponse = await api.freeplayStep({
        session_id: sessionRef.current,
        direction,
      });
      const frames = artifactUrls(response.frame_urls);
      if (response.session_id) setSessionId(response.session_id);

      // Release-to-stop commands nothing, so it is an acknowledgement rather
      // than a clip. Letting it through here would blank the stage after every
      // keypress and — worse — clear a "no certified backend" message, because
      // the stop call succeeds even when the world model is unreachable.
      if (direction === "stop" && frames.length === 0) {
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
      if (frames.length === 0) {
        setError(
          pickString(response.reason) ??
            "The world backend returned no frames for this command and gave no reason.",
        );
      }
    } catch (requestError) {
      if (requestError instanceof ApiError && requestError.status === 503) {
        // No certified world backend. Say that, and show no frame at all.
        setUnavailable(
          requestError.reason ??
            requestError.message ??
            "No certified world backend is configured for free-play.",
        );
        setChunk(undefined);
      } else {
        setError(
          requestError instanceof Error ? requestError.message : "The free-play request failed.",
        );
      }
    } finally {
      inFlight.current = false;
      setGenerating(undefined);
      const next = queued.current;
      queued.current = undefined;
      if (next) void dispatch(next);
    }
  }, []);

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
    if (!open) return;
    const keyDown = (event: KeyboardEvent) => {
      const direction = KEY_DIRECTIONS[event.key];
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
  }, [dispatch, open]);

  const buttonHandlers = (direction: FreeplayDirection) => ({
    onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      void dispatch(direction);
    },
    onPointerUp: () => void dispatch("stop"),
    onPointerLeave: () => void dispatch("stop"),
    onKeyDown: (event: ReactKeyboardEvent<HTMLButtonElement>) => {
      if ((event.key === " " || event.key === "Enter") && !event.repeat) void dispatch(direction);
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
        <Dialog.Content className="freeplay-dialog" aria-describedby="freeplay-description">
          <header className="dialog-header">
            <div>
              <p className="eyebrow">Unscored · contributes nothing to the scoreboard</p>
              <Dialog.Title className="text-balance">
                Free-play control{subjectLabel ? ` — ${subjectLabel}` : ""}
              </Dialog.Title>
            </div>
            <Dialog.Close className="icon-button" aria-label="Close free-play">
              <Glyph name="close" />
            </Dialog.Close>
          </header>
          <Dialog.Description id="freeplay-description" className="dialog-description text-pretty">
            One keypress is one direction. The server expands it into a fixed 16-action chunk and calls the
            world model, so there is no policy and no language model in this loop. Fixed instruction, clamped
            actions, release to stop. Nothing here enters a scored comparison.
          </Dialog.Description>

          <div className="freeplay-layout">
            <div className="freeplay-stage">
              {unavailable ? (
                <div className="freeplay-unavailable" role="alert">
                  <Glyph name="alert" />
                  <strong>No certified world backend for free-play</strong>
                  <p className="text-pretty">{unavailable}</p>
                  <p className="text-pretty freeplay-unavailable-note">
                    No frame is shown, because there is no frame. A placeholder here would be a claim that a
                    world model answered a keypress when none did.
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
                <div className="generating-overlay" role="status" aria-live="polite">
                  <AsciiSpinner className="freeplay-spinner" />
                  <strong>Generating</strong>
                  <span>
                    Inventing the next chunk of video from the {DIRECTION_LABELS[generating]} command
                  </span>
                  <span className="tabular-nums generating-elapsed">
                    {(elapsedMs / 1000).toFixed(1)} s elapsed
                  </span>
                  <span className="generating-source">client wall time</span>
                </div>
              )}

              <div className="stage-label">
                <span>{chunk?.resolution ?? "resolution not reported"}</span>
                <span aria-hidden="true">·</span>
                <span>{chunk?.backend ?? "backend not reported"}</span>
                <span aria-hidden="true">·</span>
                <StatusPill status="unqualified">unqualified</StatusPill>
              </div>
            </div>

            <aside className="freeplay-controls">
              <p className="control-note text-pretty">
                Arrow keys, or press and hold a button. Releasing sends a stop command. The per-axis clamp is
                applied by the server; this panel prints only the clamp the server reports.
              </p>
              <div className="dpad" aria-label="Directional world-model commands">
                <span />
                <button type="button" aria-label="Command the arm up" {...buttonHandlers("up")}>
                  <Glyph name="arrowUp" />
                </button>
                <span />
                <button type="button" aria-label="Command the arm left" {...buttonHandlers("left")}>
                  <Glyph name="arrowLeft" />
                </button>
                <button type="button" aria-label="Send a stop command" onClick={() => void dispatch("stop")}>
                  <Glyph name="stop" />
                </button>
                <button type="button" aria-label="Command the arm right" {...buttonHandlers("right")}>
                  <Glyph name="arrowRight" />
                </button>
                <span />
                <button type="button" aria-label="Command the arm down" {...buttonHandlers("down")}>
                  <Glyph name="arrowDown" />
                </button>
                <span />
              </div>

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
                  <dd>{formatCount(chunk?.frameCount ?? (frames.length || undefined))}</dd>
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

              {ignoredPress && (
                <p className="request-state" role="status">
                  A command arrived mid-generation. The most recent one is queued; the rest were dropped
                  rather than stacked.
                </p>
              )}
              {chunk?.reason && <p className="control-note text-pretty">{chunk.reason}</p>}
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
