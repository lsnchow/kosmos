import { Glyph } from "./Terminal";
import { MetaList } from "./MetaList";
import { useCallback, useEffect, useId, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { FramePlayer } from "./MediaFrame";

const FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

export type ViewerMeta = { label: string; value: string };

/**
 * A fullscreen viewer for one wall tile or one example card.
 *
 * Ported from Mirage's `GalleryModal`, and it keeps the fix their commit db98e0f
 * made: the dialog is portalled to `document.body`. Rendered inside the page it
 * was trapped in the landing's stacking context, so the backdrop painted over it
 * and ate its clicks. That is a real bug, not a style preference.
 *
 * Added here, because Mirage's version had none of it: a focus trap, focus
 * restored to whatever opened the dialog, `aria-modal` with a real accessible
 * name, and a backdrop that only closes on its own mousedown — clicking inside
 * and releasing over the backdrop no longer dismisses the dialog mid-pitch.
 *
 * The viewer shows persisted frames and nothing else. With no frames it renders
 * the reason, never a stand-in image.
 */
export function GalleryModal({
  open,
  onClose,
  title,
  subtitle,
  frames,
  emptyReason,
  badge,
  meta,
  footer,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  frames: string[];
  emptyReason?: ReactNode;
  badge?: ReactNode;
  meta?: ViewerMeta[];
  footer?: ReactNode;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const openerRef = useRef<Element | null>(null);
  const backdropDown = useRef(false);
  const headingId = useId();
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  // Remember who opened this, then give them the focus back on close. Losing
  // the caret to <body> is how a keyboard user loses their place entirely.
  useEffect(() => {
    if (!open) return;
    openerRef.current = document.activeElement;
    closeRef.current?.focus();
    return () => {
      const opener = openerRef.current;
      if (opener instanceof HTMLElement && document.contains(opener)) opener.focus();
    };
  }, [open]);

  const onKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      onCloseRef.current();
      return;
    }
    if (event.key !== "Tab") return;
    const root = dialogRef.current;
    if (!root) return;
    // No visibility filter: `offsetParent` is always null without a layout
    // engine, which would empty this list under test and disable the trap.
    // Everything matching FOCUSABLE inside the dialog is on screen by
    // construction; `hidden` and `aria-hidden` are the only exclusions.
    const focusable = Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
      (element) => !element.hidden && element.getAttribute("aria-hidden") !== "true",
    );
    if (focusable.length === 0) {
      event.preventDefault();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    // Wrap at both ends, and pull focus back in if it has already escaped.
    if (event.shiftKey && (active === first || !root.contains(active))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (active === last || !root.contains(active))) {
      event.preventDefault();
      first.focus();
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [onKeyDown, open]);

  if (!open) return null;

  return createPortal(
    <div
      className="viewer-overlay"
      onMouseDown={(event) => {
        backdropDown.current = event.target === event.currentTarget;
      }}
      onMouseUp={(event) => {
        if (backdropDown.current && event.target === event.currentTarget) onClose();
        backdropDown.current = false;
      }}
    >
      <div
        ref={dialogRef}
        className="viewer"
        role="dialog"
        aria-modal="true"
        aria-labelledby={headingId}
      >
        <header className="viewer-header">
          <div>
            <h2 id={headingId} className="text-balance">
              {title}
            </h2>
            {subtitle && <p title={subtitle}>{subtitle}</p>}
          </div>
          <div className="viewer-header-right">
            {badge}
            <button
              ref={closeRef}
              type="button"
              className="icon-button"
              onClick={onClose}
              aria-label="Close viewer"
            >
              <Glyph name="close" />
            </button>
          </div>
        </header>

        <div className="viewer-stage">
          <FramePlayer
            frames={frames}
            alt={`Accumulated persisted frames for ${title}`}
            className="viewer-media"
            emptyReason={emptyReason ?? "No persisted frames for this record"}
          />
        </div>

        {meta && meta.length > 0 && (
          <MetaList className="viewer-meta" items={meta} />
        )}

        <div className="viewer-foot">
          <span className="viewer-hint">Escape closes</span>
          {footer}
        </div>
      </div>
    </div>,
    document.body,
  );
}
