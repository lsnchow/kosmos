import { Activity } from "lucide-react";
import type { ReactNode } from "react";
import { useFlipbook } from "../hooks/useFlipbook";
import { cn } from "../lib/utils";

/**
 * A single persisted artifact. Videos keep native controls and do not autoplay,
 * because an evidence clip that plays itself invites the reader to treat it as
 * live. The accumulated-frame players below are the presentation surfaces.
 */
export function isVideoSource(src: string | undefined): boolean {
  return typeof src === "string" && /\.(mp4|webm|mov)(\?|$)/i.test(src);
}

/**
 * The "nothing to show here" state, written once.
 *
 * It appeared twice in this file, identically, for the single-frame and the
 * flipbook paths. The reason string differs because the two absences differ:
 * no media at all, versus no segment event yet.
 */
function MediaEmpty({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn("media-empty", className)}>
      <Activity aria-hidden="true" className="size-5" />
      <span>{children}</span>
    </div>
  );
}

export function MediaFrame({
  src,
  alt,
  className,
  emptyReason,
}: {
  src?: string;
  alt: string;
  className?: string;
  emptyReason?: ReactNode;
}) {
  if (!src) {
    return <MediaEmpty className={className}>{emptyReason ?? "No persisted media for this slot yet"}</MediaEmpty>;
  }
  const isVideo = isVideoSource(src);
  return isVideo ? (
    <video className={cn("media", className)} controls muted playsInline src={src} aria-label={alt} />
  ) : (
    <img className={cn("media", className)} src={src} alt={alt} />
  );
}

/**
 * Plays an accumulated frame list on the shared display clock.
 *
 * Frames are individual images, so playback is a flipbook rather than a video
 * element. It loops, and it widens as segments land — the clip visibly gets
 * longer instead of restarting. The rendered `<img>` keeps every frame's URL in
 * the DOM order it arrived so nothing about ordering is inferred at paint time.
 *
 * The playback rate is a display choice. Frame timestamps in the protocol are
 * control timestamps and are not the same thing, which the caption states.
 */
export function FramePlayer({
  frames,
  alt,
  className,
  offset,
  paused,
  emptyReason,
}: {
  frames: string[];
  alt: string;
  className?: string;
  offset?: number;
  paused?: boolean;
  emptyReason?: ReactNode;
}) {
  const index = useFlipbook(frames.length, { offset, paused });
  if (frames.length === 0) {
    return <MediaEmpty className={className}>{emptyReason ?? "No persisted segment event yet"}</MediaEmpty>;
  }
  const src = frames[Math.min(index, frames.length - 1)];
  return (
    <img
      className={cn("media", className)}
      src={src}
      alt={alt}
      data-frame-index={index}
      data-frame-count={frames.length}
      decoding="async"
    />
  );
}
