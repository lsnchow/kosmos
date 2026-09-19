import type { ReactNode } from "react";
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
 * A single metric with its source attribution. The `source` line is not
 * decoration: it is the difference between "the platform reported this" and "we
 * computed this", and it stays attached to every number on the page.
 */
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
      <p>{label}</p>
      <strong className="tabular-nums">{value}</strong>
      <span>{source}</span>
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
