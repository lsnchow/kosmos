/**
 * The two video players the landing page uses.
 *
 * Both fade themselves in on `canplay` rather than rendering opaque from the
 * start, which does double duty: it hides the first-frame pop, and it means a
 * video that never loads stays invisible over whatever is painted behind it
 * instead of leaving a black rectangle. That is the only reason the page can
 * still be shown on a dead network.
 *
 * The fade is written with `requestAnimationFrame` against the element's own
 * `style.opacity` rather than a CSS transition, because `HeroVideo` needs to
 * interrupt a fade mid-flight when a loop boundary arrives early.
 */
import { useEffect, useRef } from "react";
import { cn } from "../lib/utils";

const FADE_MS = 500;
/** How far before the end to start fading out. */
const TAIL_S = 0.55;
/** Black held between loops, so the restart is a cut to black, not a jump. */
const GAP_MS = 100;

/**
 * Ramp an element's opacity over `FADE_MS`, cancelling any ramp already
 * running on it. The returned handle lets an unmount stop the loop.
 */
function makeFader(element: HTMLElement) {
  let frame = 0;

  function cancel() {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
  }

  function to(target: number, done?: () => void) {
    cancel();
    const from = Number(element.style.opacity || "0");
    const started = performance.now();
    const step = (now: number) => {
      const progress = Math.min(1, (now - started) / FADE_MS);
      element.style.opacity = String(from + (target - from) * progress);
      if (progress < 1) {
        frame = requestAnimationFrame(step);
        return;
      }
      frame = 0;
      done?.();
    };
    frame = requestAnimationFrame(step);
  }

  return { to, cancel };
}

/**
 * The hero's background loop, with a crossfade to black at each boundary.
 *
 * `loop` is deliberately not set. A native loop cuts straight from the last
 * frame back to the first, and on a slow pan that reads as a stutter; fading
 * out over the last half second and back in from black does not.
 */
export function HeroVideo({ src, className }: { src: string; className?: string }) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    const fade = makeFader(video);
    let restart: ReturnType<typeof setTimeout> | undefined;

    const onCanPlay = () => {
      void video.play().catch(() => undefined);
      fade.to(1);
    };

    const onTimeUpdate = () => {
      if (!Number.isFinite(video.duration)) return;
      if (video.duration - video.currentTime <= TAIL_S) fade.to(0);
    };

    const onEnded = () => {
      video.style.opacity = "0";
      restart = setTimeout(() => {
        video.currentTime = 0;
        void video.play().catch(() => undefined);
        fade.to(1);
      }, GAP_MS);
    };

    video.addEventListener("canplay", onCanPlay);
    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("ended", onEnded);
    // `canplay` may already have fired before this effect ran.
    if (video.readyState >= 3) onCanPlay();

    return () => {
      video.removeEventListener("canplay", onCanPlay);
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("ended", onEnded);
      if (restart) clearTimeout(restart);
      fade.cancel();
    };
  }, [src]);

  return (
    <video
      ref={ref}
      className={className}
      style={{ opacity: 0 }}
      src={src}
      muted
      autoPlay
      playsInline
      preload="auto"
      // Decorative: the page's argument is in its text, and the console is
      // where frames that mean something carry their provenance.
      aria-hidden="true"
      tabIndex={-1}
    />
  );
}

/** A section video: native loop, same fade-in on first play. */
export function LoopVideo({ src, className }: { src: string; className?: string }) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    const fade = makeFader(video);
    const onCanPlay = () => {
      void video.play().catch(() => undefined);
      fade.to(1);
    };
    video.addEventListener("canplay", onCanPlay);
    if (video.readyState >= 3) onCanPlay();
    return () => {
      video.removeEventListener("canplay", onCanPlay);
      fade.cancel();
    };
  }, [src]);

  return (
    <video
      ref={ref}
      className={cn("h-full w-full object-cover", className)}
      style={{ opacity: 0 }}
      src={src}
      muted
      autoPlay
      loop
      playsInline
      preload="auto"
      aria-hidden="true"
      tabIndex={-1}
    />
  );
}
