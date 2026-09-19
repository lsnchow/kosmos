import { useEffect, useRef, useState } from "react";

/**
 * Re-run `task` on an interval, plus once on mount.
 *
 * Gates, sweeps, experiments and the run list previously updated only when
 * somebody pressed Refresh, which on stage means a stale panel nobody notices.
 * Polling pauses while the tab is hidden so a demo laptop left open overnight
 * does not hammer the control plane.
 */
export function usePolling(task: () => void | Promise<void>, intervalMs: number, enabled = true) {
  const taskRef = useRef(task);
  taskRef.current = task;

  useEffect(() => {
    if (!enabled || intervalMs <= 0) return;
    let disposed = false;

    const run = () => {
      if (disposed) return;
      void taskRef.current();
    };

    // The first load always runs. Gating it on visibility means a console opened
    // in a background tab — which is exactly how a demo machine is set up — shows
    // empty panels until somebody happens to focus it.
    run();

    const tick = () => {
      if (typeof document !== "undefined" && document.visibilityState === "hidden") return;
      run();
    };
    const handle = setInterval(tick, intervalMs);

    // Catch up immediately when the tab comes back, rather than waiting out the
    // remainder of an interval in front of an audience.
    const onVisibility = () => {
      if (document.visibilityState === "visible") run();
    };
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", onVisibility);
    }

    return () => {
      disposed = true;
      clearInterval(handle);
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", onVisibility);
      }
    };
  }, [enabled, intervalMs]);
}

/** A once-per-second wall clock, used to age server-supplied timestamps. */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const handle = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(handle);
  }, [intervalMs]);
  return now;
}

const PRESENTATION_STORAGE_KEY = "plumb.presentation";

/**
 * Presentation mode raises the whole type scale for a projector. It is a
 * display setting: it changes no number, label, or qualification.
 */
export function usePresentationMode(): [boolean, (next: boolean) => void] {
  const [enabled, setEnabled] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    try {
      return window.localStorage.getItem(PRESENTATION_STORAGE_KEY) === "on";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.dataset.presentation = enabled ? "on" : "off";
    try {
      window.localStorage.setItem(PRESENTATION_STORAGE_KEY, enabled ? "on" : "off");
    } catch {
      // A blocked storage quota must not stop the mode from applying.
    }
  }, [enabled]);

  return [enabled, setEnabled];
}
