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
  eyebrow,
  children,
  className,
  action,
  id,
}: {
  title: string;
  eyebrow?: string;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
  id?: string;
}) {
  return (
    <section className={cn("panel", className)} id={id}>
      <div className="panel-heading">
        <div>
          {eyebrow && <p className="eyebrow">{eyebrow}</p>}
          <h2 className="text-balance">{title}</h2>
        </div>
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
  return <span className="source-chip">{children}</span>;
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="table-note text-pretty">{children}</p>;
}
