import { Eye, Film, Pause, Play } from "lucide-react";
import { useState } from "react";
import { api, artifactUrl, type SixClipResponse } from "../lib/api";
import { formatCount, formatRateAsPercent, pickNumber, pickString } from "../lib/format";
import { cn } from "../lib/utils";
import { EmptyState, Note, Panel, SourceChip, StatusPill } from "./Primitives";

/**
 * Six clips, one real. Positions are randomized by the server and the answer is
 * sealed until reveal, so the panel cannot leak it through ordering or markup.
 *
 * It never claims the audience was fooled. An accuracy figure appears only when
 * the server reports answers that were actually recorded; with no recorded
 * answers the panel says exactly that.
 */
export function SixClipPanel({
  data,
  onRevealed,
}: {
  data?: SixClipResponse;
  onRevealed?: (revealed: SixClipResponse) => void;
}) {
  const [revealing, setRevealing] = useState(false);
  const [error, setError] = useState<string>();
  const [playing, setPlaying] = useState(true);

  const clips = data?.clips ?? [];
  const revealed = data?.revealed === true;
  // `audience_accuracy` is the name the control plane currently ships.
  const answers = data?.recorded_answers ?? data?.audience_accuracy;
  const recordedN = pickNumber(answers?.n);
  const recordedCorrect = pickNumber(answers?.correct);

  const reveal = async () => {
    setRevealing(true);
    setError(undefined);
    try {
      const response = await api.revealSixClip();
      onRevealed?.(response);
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "The reveal request failed.",
      );
    } finally {
      setRevealing(false);
    }
  };

  return (
    <Panel
      title="Real or generated?"
      className="sixclip-panel"
      action={<SourceChip>{revealed ? "revealed" : "provenance sealed"}</SourceChip>}
      id="sixclip"
    >
      {clips.length === 0 ? (
        <EmptyState icon={<Film aria-hidden="true" className="size-5" />}>
          {pickString(data?.reason) ??
            "No six-clip panel has been published by the API."}{" "}
          No stand-in clips are shown, because a generated clip labelled as the test would make the test
          meaningless.
        </EmptyState>
      ) : (
        <>
          <div className="sixclip-controls">
            <button
              type="button"
              className="button button-secondary"
              onClick={() => setPlaying((prior) => !prior)}
            >
              {playing ? (
                <Pause aria-hidden="true" className="size-4" />
              ) : (
                <Play aria-hidden="true" className="size-4" />
              )}
              {playing ? "Pause clips" : "Play clips"}
            </button>
            <button
              type="button"
              className="button button-primary"
              onClick={() => void reveal()}
              disabled={revealing || revealed}
            >
              <Eye aria-hidden="true" className="size-4" />
              {revealed ? "Revealed" : revealing ? "Revealing…" : "Reveal which is real"}
            </button>
          </div>

          <ol className="sixclip-grid">
            {clips.map((clip, index) => {
              const url = artifactUrl(clip.url);
              const provenance = pickString(clip.provenance);
              const isReal = provenance === "real";
              return (
                <li
                  key={pickString(clip.id) ?? `clip-${index}`}
                  className={cn("sixclip-cell", revealed && isReal && "sixclip-cell-real")}
                >
                  <span className="sixclip-number" aria-hidden="true">
                    {index + 1}
                  </span>
                  {url ? (
                    <video
                      className="sixclip-video"
                      src={url}
                      muted
                      loop
                      playsInline
                      autoPlay={playing}
                      controls={!playing}
                      aria-label={`Clip ${index + 1}${revealed && provenance ? `, ${provenance}` : ", provenance sealed"}`}
                      ref={(element) => {
                        if (!element) return;
                        if (playing) void element.play().catch(() => undefined);
                        else element.pause();
                      }}
                    />
                  ) : (
                    <div className="sixclip-missing">Clip {index + 1} has no persisted media</div>
                  )}
                  <div className="sixclip-caption">
                    <span>Clip {index + 1}</span>
                    {revealed ? (
                      <StatusPill status={isReal ? "pass" : "unknown"}>
                        {provenance ?? "not reported"}
                      </StatusPill>
                    ) : (
                      <StatusPill status="unknown">sealed</StatusPill>
                    )}
                  </div>
                  {revealed && clip.source && <p className="sixclip-source truncate">{clip.source}</p>}
                </li>
              );
            })}
          </ol>

          {error && (
            <p className="inline-error" role="alert">
              {error}
            </p>
          )}

          <div className="sixclip-result" role="status">
            {recordedN !== undefined && recordedCorrect !== undefined && recordedN > 0 ? (
              <span>
                Recorded audience answers: <b className="tabular-nums">{formatCount(recordedCorrect)}</b> of{" "}
                <b className="tabular-nums">{formatCount(recordedN)}</b> identified the real clip (
                {formatRateAsPercent(recordedCorrect / recordedN)})
                {answers?.method ? ` · ${answers.method}` : ""}.
              </span>
            ) : (
              <span>
                No audience answers were recorded, so no accuracy figure is shown and no claim is made about
                whether the room could tell.
              </span>
            )}
          </div>
        </>
      )}
      <Note summary="How this test is run">
        Clip order is randomized by the server and the real clip's identity stays sealed until reveal. A
        result here is an audience observation, not a measurement of the world model.
      </Note>
    </Panel>
  );
}
