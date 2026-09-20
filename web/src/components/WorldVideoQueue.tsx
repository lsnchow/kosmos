import { useEffect, useRef, useState } from "react";
import { artifactUrl } from "../lib/api";
import { formatCount, formatSeconds } from "../lib/format";
import { Glyph } from "./Terminal";
import { EmptyState, Note, SourceChip, StatusPill } from "./Primitives";

export type WorldVideo = {
  id: string;
  title: string;
  model: string;
  kind: string;
  video_url: string;
  report_url?: string;
  sha256?: string;
  report_sha256?: string;
  download_name?: string;
  source_experiment?: string;
  frame_count?: number;
  duration_seconds?: number;
  fps?: number;
  width?: number;
  height?: number;
  notes?: string[];
  qualified: false;
  provenance: "recorded_model_output";
};

type Catalog = { videos?: WorldVideo[]; total?: number; qualified?: false };

function safeArtifactUrl(value: unknown): string | undefined {
  if (typeof value === "string" && value.startsWith("//")) return undefined;
  return artifactUrl(value);
}

function downloadName(video: WorldVideo): string {
  const stem = video.id.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "recording";
  return stem.endsWith(".mp4") ? stem : `${stem}.mp4`;
}

/**
 * Recorded files only. This component neither submits work nor constructs a
 * rollout: it queues the catalog's already-persisted world-model videos for
 * native browser playback.
 */
export function WorldVideoQueue() {
  const [videos, setVideos] = useState<WorldVideo[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [playingAll, setPlayingAll] = useState(false);
  const [failedIds, setFailedIds] = useState<Set<string>>(() => new Set());
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let active = true;
    void fetch("/api/world-videos")
      .then(async (response) => {
        const payload: unknown = await response.json().catch(() => undefined);
        if (!response.ok) throw new Error(
          payload && typeof payload === "object" && typeof (payload as Record<string, unknown>).detail === "string"
            ? (payload as Record<string, string>).detail
            : "Recorded world-video catalog is unavailable.",
        );
        return payload as Catalog;
      })
      .then((catalog) => {
        if (!active) return;
        const records = Array.isArray(catalog.videos) ? catalog.videos : [];
        setVideos(records);
        setSelectedIndex(0);
      })
      .catch((requestError: unknown) => {
        if (active) setError(requestError instanceof Error ? requestError.message : "Recorded world-video catalog is unavailable.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const selected = videos[selectedIndex];
  const selectedUrl = safeArtifactUrl(selected?.video_url);
  const selectedFailed = selected ? failedIds.has(selected.id) : false;

  useEffect(() => {
    if (!playingAll || !selectedUrl || !videoRef.current) return;
    void videoRef.current.play().catch(() => {
      setPlayingAll(false);
      setError("The selected recording could not start. It remains visible; choose another recording or use its download link.");
    });
  }, [playingAll, selectedUrl, selectedIndex]);

  const choose = (index: number) => {
    setPlayingAll(false);
    videoRef.current?.pause();
    setSelectedIndex(index);
    setError(undefined);
  };

  const playAll = () => {
    if (!videos[0] || !safeArtifactUrl(videos[0].video_url)) return;
    setError(undefined);
    setFailedIds(new Set());
    setSelectedIndex(0);
    setPlayingAll(true);
  };

  const pause = () => {
    videoRef.current?.pause();
    setPlayingAll(false);
  };

  const next = () => {
    if (selectedIndex >= videos.length - 1) return;
    choose(selectedIndex + 1);
  };

  const previous = () => {
    if (selectedIndex <= 0) return;
    choose(selectedIndex - 1);
  };

  const ended = () => {
    if (!playingAll) return;
    const nextIndex = videos.findIndex((video, index) => index > selectedIndex && !failedIds.has(video.id));
    if (nextIndex < 0) {
      setPlayingAll(false);
      return;
    }
    setSelectedIndex(nextIndex);
  };

  const failed = () => {
    if (!selected) return;
    setFailedIds((prior) => new Set([...prior, selected.id]));
    setPlayingAll(false);
    setError("This recording could not play. It remains selected and visible in the catalog; playback stopped without retrying it.");
  };

  if (loading) return <p className="text-pretty text-sm text-[var(--text-muted)]" role="status">Loading recorded world-model videos…</p>;
  if (error && videos.length === 0) {
    return <EmptyState>Recorded world-model videos are unavailable from this catalog. Select Run frames to inspect persisted episode frames instead.</EmptyState>;
  }
  if (!selected || !selectedUrl) {
    return <EmptyState>No recorded world-model video is available in this catalog. Select Run frames to inspect persisted episode frames instead.</EmptyState>;
  }

  const reportUrl = safeArtifactUrl(selected.report_url);
  return (
    <section aria-labelledby="world-video-queue-heading">
      <div className="wall-summary">
        <span><b>Recorded model output</b> · not live · unqualified</span>
        <span>Playback queue only — it does not generate, submit, score, or invoke models.</span>
        <span><b className="tabular-nums">{selectedIndex + 1}</b> of {formatCount(videos.length)} catalog recordings</span>
      </div>
      <h3 className="text-balance text-base font-medium" id="world-video-queue-heading">{selected.title}</h3>
      <p className="text-pretty text-sm text-[var(--text-muted)]">{selected.model} · {selected.kind}</p>
      <video
        ref={videoRef}
        className="mt-3 max-h-96 w-full rounded border border-[var(--line-3)] bg-black object-contain"
        controls
        playsInline
        preload="metadata"
        src={selectedUrl}
        onEnded={ended}
        onError={failed}
        aria-label={`Recorded model output: ${selected.title}`}
      />
      <p className="mt-2 text-pretty text-sm text-[var(--text-muted)]">No caption track is supplied by this catalog.</p>
      {error && <p className="inline-error" role="alert">{error}</p>}

      <div className="burst-actions mt-3">
        <button type="button" className="button button-primary" onClick={playAll} disabled={playingAll}>
          <Glyph name="play" /> Play all
        </button>
        <button type="button" className="button button-secondary" onClick={pause} disabled={!playingAll}>
          <Glyph name="pause" /> Pause
        </button>
        <button type="button" className="button button-secondary" onClick={previous} disabled={selectedIndex === 0}>Previous</button>
        <button type="button" className="button button-secondary" onClick={next} disabled={selectedIndex >= videos.length - 1}>{selectedFailed ? "Skip failed recording" : "Next"}</button>
        <StatusPill status="unqualified">unqualified</StatusPill>
      </div>
      {selectedFailed && <p className="burst-disabled">Playback is stopped for this failed recording. Select the next recording manually, or explicitly start Play all from the first recording; no automatic retry is attempted.</p>}

      <div className="run-metrics">
        <div className="metric"><p>Frames</p><strong>{formatCount(selected.frame_count)}</strong></div>
        <div className="metric"><p>Duration</p><strong>{formatSeconds(selected.duration_seconds)}</strong></div>
        <div className="metric"><p>Dimensions</p><strong>{selected.width && selected.height ? `${selected.width} × ${selected.height}` : "not reported"}</strong></div>
        <div className="metric"><p>SHA-256</p><strong className="truncate" title={selected.sha256}>{selected.sha256 ?? "not reported"}</strong></div>
      </div>
      <div className="mt-3 flex flex-wrap gap-3 text-sm">
        <a className="button button-secondary" href={selectedUrl} download={selected.download_name || downloadName(selected)}>Download MP4 <Glyph name="arrowDown" /></a>
        {reportUrl && <a className="button button-secondary" href={reportUrl} target="_blank" rel="noreferrer">Open report <Glyph name="arrowUpRight" /></a>}
      </div>
      {selected.source_experiment && <p className="mt-3 break-words text-pretty text-sm text-[var(--text-muted)]">Source experiment: {selected.source_experiment}</p>}
      {(selected.notes?.length ?? 0) > 0 && <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-[var(--text-muted)]">{selected.notes?.map((note, index) => <li key={`${note}-${index}`} className="text-pretty">{note}</li>)}</ul>}

      <ol className="mt-4 grid min-w-0 gap-2" aria-label="Recorded world-model video queue">
        {videos.map((video, index) => (
          <li className="min-w-0" key={video.id}>
            <button
              type="button"
              className="button button-secondary min-w-0 w-full justify-between overflow-hidden text-left"
              aria-pressed={index === selectedIndex}
              aria-label={`Select recorded model output ${video.title}, ${video.model}${video.source_experiment ? `, source experiment ${video.source_experiment}` : ""}`}
              onClick={() => choose(index)}
            >
              <span className="min-w-0 truncate" title={video.source_experiment ? `${video.title} · ${video.source_experiment}` : video.title}>{video.title}</span>
              <SourceChip>{video.provenance === "recorded_model_output" ? "recorded model output" : "recording"}</SourceChip>
            </button>
          </li>
        ))}
      </ol>
      <Note summary="What these recordings are">
        Every item is an existing saved world-model output from the server catalog, including short probes when
        supplied. Selecting or playing a file does not generate a frame, submit a request, or establish a score.
      </Note>
    </section>
  );
}
