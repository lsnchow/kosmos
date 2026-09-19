import { useId, useState, type ReactNode } from "react";
import { cn } from "../lib/utils";

export function statusClass(status: unknown) {
  const value = String(status ?? "unknown").toLowerCase();
  if (["pass", "passed", "completed", "complete", "healthy", "ready", "open", "live"].includes(value))
    return "status-good";
  if (["fail", "failed", "blocked", "error", "degraded", "stale", "unavailable"].includes(value))
    return "status-bad";
  if (["running", "queued", "in_progress", "in-progress", "generating", "connecting", "pending"].includes(value))
    return "status-active";
  return "status-muted";
}

export function StatusPill({
  status,
  children,
  title,
}: {
  status?: unknown;
  children?: ReactNode;
  title?: string;
}) {
  return (
    <span className={cn("status-pill", statusClass(status))} title={title}>
      {children ?? String(status ?? "unknown")}
    </span>
  );
}

export function Panel({
  title,
  children,
  className,
  action,
  id,
}: {
  title: string;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
  id?: string;
}) {
  return (
    <section className={cn("panel", className)} id={id}>
      <div className="panel-heading">
        <h2 className="text-balance">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

/**
 * A source attribution, on demand.
 *
 * Every figure in this console says where it came from — that attribution is
 * the product, not decoration, and it is the difference between "the platform
 * reported this" and "we computed this". But printing it as a permanent third
 * line under all thirty-odd metrics meant two thirds of the ink on a panel was
 * provenance, and a reader scanning for the number had to skip past it every
 * time.
 *
 * So it moves behind an `i`. The text is still in the DOM, still associated
 * with its control by `aria-describedby`, and still reachable by keyboard —
 * this hides it from the eye, never from the accessibility tree or from
 * anything reading the page. Hover or focus brings it back.
 */
export function InfoTip({ label, children }: { label: string; children: ReactNode }) {
  const id = useId();
  // WCAG 1.4.13 asks that content revealed on hover or focus can be dismissed
  // without moving the pointer or the focus. Escape does that; the next hover
  // or focus clears the flag, so dismissing one tooltip does not disable it.
  const [dismissed, setDismissed] = useState(false);
  return (
    <span
      className={cn("infotip", dismissed && "infotip-dismissed")}
      onKeyDown={(event) => {
        if (event.key === "Escape") setDismissed(true);
      }}
      onPointerEnter={() => setDismissed(false)}
      onFocus={() => setDismissed(false)}
    >
      <button type="button" className="infotip-trigger" aria-label={`Source for ${label}`} aria-describedby={id}>
        <span aria-hidden="true">i</span>
      </button>
      <span id={id} role="tooltip" className="infotip-body">
        {children}
      </span>
    </span>
  );
}

/** A single metric: a label, the figure, and its source one `i` away. */
export function DataValue({
  label,
  value,
  source,
  tone,
  size,
}: {
  label: string;
  value: string;
  source: string;
  tone?: "default" | "warn" | "bad";
  size?: "default" | "large";
}) {
  return (
    <div className={cn("metric", size === "large" && "metric-large", tone && `metric-${tone}`)}>
      <p>
        {label}
        <InfoTip label={label}>{source}</InfoTip>
      </p>
      <strong className="tabular-nums">{value}</strong>
    </div>
  );
}

/**
 * An empty state always says *why* it is empty. It never stands in for a zero.
 */
export function EmptyState({
  icon,
  children,
  className,
}: {
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("empty-state", className)}>
      {icon}
      <p className="text-pretty">{children}</p>
    </div>
  );
}

export function SourceChip({ children }: { children: ReactNode }) {
  // The only surviving truncation in the sheet: run ids are values, not labels,
  // so abbreviating one loses nothing a hover cannot restore. `title` is what
  // makes that true -- without it the id would be gone.
  const full = typeof children === "string" ? children : undefined;
  return (
    <span className="source-chip" title={full}>
      {children}
    </span>
  );
}

/**
 * A caveat, available but not shouting.
 *
 * Ten panels each carried a paragraph of this prose permanently on screen, and
 * together they made the live page 1,947 words. The content is not decoration —
 * it is where this console says what a number does *not* mean — so it is kept
 * and kept exact. What changes is that the reader opens it when the question
 * occurs to them, instead of reading it every time.
 *
 * `summary` is required and should name the specific question, not say "note":
 * a disclosure nobody can predict the contents of is one nobody opens.
 */
export function Note({ summary, children }: { summary: string; children: ReactNode }) {
  return (
    <details className="note">
      <summary>{summary}</summary>
      <div className="note-body text-pretty">{children}</div>
    </details>
  );
}
