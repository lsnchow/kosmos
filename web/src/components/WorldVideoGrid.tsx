import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { artifactUrl } from "../lib/api";
import { formatSeconds } from "../lib/format";
import type { WorldVideo } from "./WorldVideoQueue";
import { NewEvaluationDialog } from "./LiveDemo";

function mediaUrl(value: unknown) {
  return typeof value === "string" && value.startsWith("/api/artifacts/")
    ? artifactUrl(value)
    : undefined;
}

/** A small playback selection, never interpreted as matched policy results. */
export function gallerySelection(videos: WorldVideo[]) {
  const seen = new Set<string>();
  return videos
    .filter((video) => {
      if (!video || typeof video !== "object") return false;
      const identity = video.sha256 || video.video_url;
      if (
        !mediaUrl(video.video_url) ||
        (video.frame_count ?? 0) <= 2 ||
        seen.has(identity)
      )
        return false;
      seen.add(identity);
      return true;
    })
    .slice(0, 6);
}

function VideoCard({
  video,
  paused,
  onNewBranch,
}: {
  video: WorldVideo;
  paused: boolean;
  onNewBranch: (video: WorldVideo, trigger: HTMLButtonElement) => void;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const [failed, setFailed] = useState(false);
  const src = mediaUrl(video.video_url);
  const report = mediaUrl(video.report_url);

  useEffect(() => {
    const player = ref.current;
    if (!player || failed) return;
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let visible = false;
    const sync = () => {
      if (!visible || document.hidden || paused || motion.matches)
        player.pause();
      else
        void player.play().catch(() => {
          /* Native controls remain available if autoplay is blocked. */
        });
    };
    const observer = new IntersectionObserver(
      ([entry]) => {
        visible = entry.isIntersecting;
        sync();
      },
      { threshold: 0.15 },
    );
    observer.observe(player);
    document.addEventListener("visibilitychange", sync);
    motion.addEventListener("change", sync);
    sync();
    return () => {
      observer.disconnect();
      document.removeEventListener("visibilitychange", sync);
      motion.removeEventListener("change", sync);
      player.pause();
    };
  }, [paused, failed]);

  return (
    <article className="gallery-card">
      <video
        ref={ref}
        src={src}
        controls
        muted
        playsInline
        loop
        preload="metadata"
        aria-label={`Saved experiment: ${video.title}`}
        onError={() => setFailed(true)}
      />
      <div className="gallery-card-body">
        <p className="eyebrow">Pre-generated · experimental</p>
        <h3 className="text-balance">{video.model}</h3>
        <p className="gallery-caption text-pretty">{video.title}</p>
        <p className="gallery-caption tabular-nums">
          {formatSeconds(video.duration_seconds)} · {video.fps ?? "Unknown"} FPS
          ·{" "}
          {video.width && video.height
            ? `${video.width} × ${video.height}`
            : "Size not reported"}
        </p>
        {failed && (
          <p className="inline-error" role="alert">
            This clip could not play. Download it or choose another recording.
          </p>
        )}
        <button
          type="button"
          className="button button-secondary"
          aria-label={`Start a new interactive branch from ${video.title}`}
          onClick={(event) => onNewBranch(video, event.currentTarget)}
        >
          New interactive branch →
        </button>
        <details>
          <summary>Clip details & download</summary>
          <p className="text-pretty">
            {video.notes?.[1] ??
              "Saved world-model experiment; not a matched policy comparison."}
          </p>
          <p className="text-pretty">
            Task instruction and conditioning, when recorded, are in the source
            report. No task text is inferred from the video.
          </p>
          <div className="gallery-links">
            <a href={src} download>
              Download MP4
            </a>
            {report && (
              <a href={report} target="_blank" rel="noreferrer">
                Source report ↗
              </a>
            )}
          </div>
        </details>
      </div>
    </article>
  );
}

export function WorldVideoGrid() {
  const [videos, setVideos] = useState<WorldVideo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [attempt, setAttempt] = useState(0);
  const [paused, setPaused] = useState(false);
  const [branch, setBranch] = useState<{
    video: WorldVideo;
    trigger: HTMLButtonElement;
  }>();

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(undefined);
    void fetch("/api/world-videos", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok)
          throw new Error("The recording catalog could not be loaded.");
        const payload = await response.json();
        if (!Array.isArray(payload.videos))
          throw new Error("The recording catalog has an unexpected format.");
        if (!controller.signal.aborted) setVideos(payload.videos);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setError(
            reason instanceof Error
              ? reason.message
              : "The recording catalog could not be loaded.",
          );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [attempt]);

  const selected = gallerySelection(videos);
  return (
    <section aria-labelledby="gallery-heading" aria-busy={loading}>
      <div className="gallery-toolbar">
        <div>
          <h2 id="gallery-heading" className="text-balance">
            Saved experiments
          </h2>
          <p className="text-pretty">
            Real model outputs, not yet a matched six-policy comparison.
          </p>
        </div>
        {selected.length > 0 && (
          <button
            type="button"
            className="button button-secondary"
            onClick={() => setPaused(!paused)}
          >
            {paused ? "Resume loops" : "Pause all"}
          </button>
        )}
      </div>
      {loading ? (
        <div
          className="gallery-grid"
          role="status"
          aria-label="Loading saved videos"
        >
          {[0, 1, 2].map((index) => (
            <div className="gallery-placeholder" key={index}>
              Loading saved video…
            </div>
          ))}
        </div>
      ) : error ? (
        <div>
          <p role="alert">{error}</p>
          <button
            type="button"
            className="button button-secondary"
            onClick={() => setAttempt(attempt + 1)}
          >
            Retry catalog
          </button>
        </div>
      ) : selected.length ? (
        <div className="gallery-grid">
          {selected.map((video) => (
            <VideoCard
              key={video.id}
              video={video}
              paused={paused || Boolean(branch)}
              onNewBranch={(video, trigger) => setBranch({ video, trigger })}
            />
          ))}
        </div>
      ) : (
        <p className="text-pretty">
          No multi-frame recordings are available yet.{" "}
          <Link to="/clips">Check the recording archive →</Link>
        </p>
      )}
      <p className="gallery-footnote text-pretty">
        Watching a clip never starts generation or scoring. A new interactive
        branch starts from this clip’s last verified image and uses experimental
        IRASim prediction; it is not an exact checkpoint resume. Loops replay
        the saved clip; they do not extend the robot’s trajectory. Short probes
        and repeats remain in the archive.
      </p>
      {branch && (
        <NewEvaluationDialog
          key={branch.video.id}
          open
          onOpenChange={(open) => {
            if (!open) setBranch(undefined);
          }}
          sourceVideo={branch.video}
          returnFocusTo={branch.trigger}
        />
      )}
    </section>
  );
}
