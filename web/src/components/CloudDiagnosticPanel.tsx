import { useCallback, useEffect, useMemo, useState } from "react";
import { usePolling } from "../hooks/usePolling";
import { formatSeconds } from "../lib/format";
import { Glyph } from "./Terminal";
import { DataValue, EmptyState, Note, Panel, SourceChip, StatusPill } from "./Primitives";

const ROOT = "/api/cloud-diagnostics";
const POLL_MS = 3_000;

type RecordMap = Record<string, unknown>;
type RequestState = "pending" | "running" | "completed" | "failed" | "ambiguous";

type CloudStatus = {
  configured?: boolean;
  available?: boolean;
  reason?: string;
  policy?: string;
  qualified?: false;
  model_id?: string;
  deployment_id?: string;
  fixture?: { current_url?: string; goal_url?: string; description?: string };
};

type CloudRequest = {
  request_id?: string;
  status?: RequestState;
  qualified?: false;
  diagnostic_only?: true;
  physical_action?: number[];
  modelrevision?: string;
  timing?: { load_seconds_once?: number; inference_seconds?: number };
  report_url?: string;
  raw_response_url?: string;
  error?: { kind?: string; message?: string; manual_reconciliation_required?: boolean; automatic_retry_allowed?: boolean };
};

class CloudDiagnosticError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = "CloudDiagnosticError";
  }
}

function record(value: unknown): RecordMap | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? value as RecordMap : undefined;
}

function text(...values: unknown[]): string | undefined {
  return values.find((value): value is string => typeof value === "string" && value.trim().length > 0);
}

function finite(...values: unknown[]): number | undefined {
  return values.find((value): value is number => typeof value === "number" && Number.isFinite(value));
}

function safeUrl(value: unknown): string | undefined {
  const url = text(value);
  if (!url) return undefined;
  if (url.startsWith("//")) return undefined;
  return url.startsWith("/") || /^https?:\/\//i.test(url) ? url : undefined;
}

function errorText(payload: unknown, fallback: string): string {
  const value = record(payload);
  return text(value?.detail, value?.reason, value?.error) ?? fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  const payload: unknown = await response.json().catch(() => undefined);
  if (!response.ok) throw new CloudDiagnosticError(response.status, errorText(payload, `Request failed (${response.status}).`));
  return payload as T;
}

function statusLabel(value: unknown): RequestState | "not requested" {
  return value === "pending" || value === "running" || value === "completed" || value === "failed" || value === "ambiguous"
    ? value
    : "not requested";
}

function pillStatus(value: RequestState | "not requested"): string {
  return value === "ambiguous" ? "failed" : value;
}

function actionValues(value: unknown): number[] | undefined {
  if (!Array.isArray(value) || value.length !== 7) return undefined;
  return value.every((item) => typeof item === "number" && Number.isFinite(item)) ? value : undefined;
}

function timingValues(value: unknown): { loadSeconds?: number; inferenceSeconds?: number } {
  const timing = record(value);
  return {
    loadSeconds: finite(timing?.load_seconds_once),
    inferenceSeconds: finite(timing?.inference_seconds),
  };
}

function actionText(values: number[] | undefined): string {
  return values ? `[${values.map((value) => value.toFixed(5)).join(", ")}]` : "not reported";
}

/**
 * A one-fixture cloud smoke request, intentionally separate from study and
 * synthetic task controls. It has no completion score or browser-side retry.
 */
export function CloudDiagnosticPanel() {
  const [status, setStatus] = useState<CloudStatus>();
  const [requests, setRequests] = useState<CloudRequest[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [detail, setDetail] = useState<CloudRequest>();
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string>();
  const [manualAcknowledged, setManualAcknowledged] = useState(false);

  const refresh = useCallback(async () => {
    const outcomes = await Promise.allSettled([
      request<CloudStatus>(`${ROOT}/status`),
      request<{ requests?: CloudRequest[] }>(ROOT),
    ]);
    const [statusResult, listResult] = outcomes;
    if (statusResult.status === "fulfilled") setStatus(statusResult.value);
    if (listResult.status === "fulfilled") {
      const next = listResult.value.requests ?? [];
      // A POST can resolve before the next list read has observed the new
      // persisted row. Preserve that returned pending record until the server
      // list includes it; otherwise the panel would erase its own queued state
      // and visually regress to an older history snapshot.
      setRequests((prior) => [
        ...next,
        ...prior.filter((existing) => existing.request_id && !next.some((item) => item.request_id === existing.request_id)),
      ]);
      setSelectedId((prior) => prior ?? next[0]?.request_id);
    }
    const failure = outcomes.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") {
      setError(failure.reason instanceof Error ? failure.reason.message : "Cloud diagnostic status could not be loaded.");
    } else {
      setError(undefined);
    }
    setLoading(false);
  }, []);

  const selectedSummary = requests.find((item) => item.request_id === selectedId);
  const selected = selectedSummary ? { ...detail, ...selectedSummary } : detail;
  const selectedState = statusLabel(selected?.status);
  const activeRequest = requests.some((item) => item.status === "pending" || item.status === "running");

  useEffect(() => {
    void refresh();
  }, [refresh]);
  usePolling(refresh, POLL_MS, activeRequest);

  useEffect(() => {
    if (!selectedId) {
      setDetail(undefined);
      return;
    }
    let active = true;
    void request<CloudRequest>(`${ROOT}/${encodeURIComponent(selectedId)}`)
      .then((payload) => {
        if (active) setDetail(payload);
      })
      .catch((requestError: unknown) => {
        if (active) setError(requestError instanceof Error ? requestError.message : "Cloud diagnostic detail could not be loaded.");
      });
    return () => {
      active = false;
    };
  }, [selectedId, selectedSummary?.status]);

  // An acknowledgement belongs to the selected historical record, never the
  // browser session. Reopening another request always requires a fresh review.
  useEffect(() => {
    setManualAcknowledged(false);
  }, [selectedId]);

  const selectedNeedsManualReconciliation = selectedState === "failed" || selectedState === "ambiguous";
  const disabledReason = !status?.configured
    ? text(status?.reason, "Cloud diagnostic is not configured.")
    : !status.available
      ? text(status.reason, "Cloud diagnostic is not available.")
      : submitting
        ? "Submitting the cloud diagnostic."
        : activeRequest
          ? "A cloud diagnostic is already pending or running."
          : selectedNeedsManualReconciliation && !manualAcknowledged
            ? "Review and acknowledge the prior record before creating a new diagnostic request."
            : undefined;

  const submit = async () => {
    if (disabledReason || submitting) return;
    setSubmitting(true);
    setError(undefined);
    try {
      const created = await request<CloudRequest>(ROOT, {
        method: "POST",
        // The server owns the fixed fixture prompt and request payload. An
        // empty body asks it to create a new persisted diagnostic record only.
        body: JSON.stringify({}),
      });
      if (!created.request_id) throw new Error("The service did not return a cloud diagnostic request record.");
      setRequests((prior) => [created, ...prior.filter((item) => item.request_id !== created.request_id)]);
      setSelectedId(created.request_id);
      setDetail(created);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The cloud diagnostic could not be queued.");
    } finally {
      setSubmitting(false);
    }
  };

  const fixture = status?.fixture;
  const currentUrl = safeUrl(fixture?.current_url);
  const goalUrl = safeUrl(fixture?.goal_url);
  const modelId = text(status?.model_id);
  const deploymentId = text(status?.deployment_id);
  const action = actionValues(selected?.physical_action);
  const timing = timingValues(selected?.timing);
  const readiness = status?.available === true
    ? "available"
    : status?.configured
      ? "configured"
      : "not configured";

  return (
    <Panel
      title="Cloud diagnostic"
      id="cloud-diagnostic"
      action={<SourceChip>{status?.policy ?? "policy not reported"} · diagnostic only</SourceChip>}
    >
      <p className="task-prompt-verdict text-pretty">
        <b>Unqualified cloud diagnostic.</b> It requests one goal-conditioned native action for pinned current and goal frames.
        It is not a generated rollout, a judge score, or a study result.
      </p>

      <div className="grid gap-3 sm:grid-cols-2" aria-label="Pinned diagnostic fixture">
        {currentUrl ? (
          <figure className="rounded border border-[var(--line-3)] p-2">
            <img className="aspect-video w-full object-contain" src={currentUrl} alt="Pinned diagnostic current observation" />
            <figcaption className="mt-2 text-pretty text-sm text-[var(--text-muted)]">Current observation</figcaption>
          </figure>
        ) : <EmptyState>Current fixture frame is not available from the server.</EmptyState>}
        {goalUrl ? (
          <figure className="rounded border border-[var(--line-3)] p-2">
            <img className="aspect-video w-full object-contain" src={goalUrl} alt="Pinned diagnostic goal observation" />
            <figcaption className="mt-2 text-pretty text-sm text-[var(--text-muted)]">Goal condition</figcaption>
          </figure>
        ) : <EmptyState>Goal fixture frame is not available from the server.</EmptyState>}
      </div>
      {fixture?.description && <p className="mt-3 text-pretty text-sm text-[var(--text-muted)]">{fixture.description}</p>}

      <div className="run-metrics">
        <DataValue label="Cloud readiness" value={readiness} source="Server configuration state; it does not probe GPU or remote reachability" />
        <DataValue label="Model" value={modelId ?? "not reported"} source="Server-reported Baseten model identity" />
        <DataValue label="Deployment" value={deploymentId ?? "not reported"} source="Server-reported Baseten deployment identity" />
      </div>
      {status?.reason && <p className="text-pretty text-sm text-[var(--text-muted)]">Status note: {status.reason}</p>}

      <div className="burst-actions">
        <button
          type="button"
          className="button button-primary"
          disabled={Boolean(disabledReason)}
          aria-describedby={disabledReason ? "cloud-disabled-reason" : undefined}
          onClick={() => void submit()}
        >
          <Glyph name="play" />
          {submitting ? "Queueing cloud model…" : selectedNeedsManualReconciliation ? "Run new diagnostic" : "Run cloud model"}
        </button>
        <StatusPill status={pillStatus(selectedState)}>{selectedState}</StatusPill>
      </div>
      {disabledReason && <p className="burst-disabled" id="cloud-disabled-reason">Disabled: {disabledReason}</p>}
      {loading && <p className="mt-3 text-sm text-[var(--text-muted)]" role="status">Loading cloud diagnostic records…</p>}
      {error && <p className="inline-error" role="alert">{error}</p>}

      {selected && (
        <section className="mt-4" aria-labelledby="cloud-result-heading">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-balance text-base font-medium" id="cloud-result-heading">Selected server record</h3>
            <SourceChip>{selected.request_id ?? "request id not reported"}</SourceChip>
          </div>
          <div className="run-metrics">
            <DataValue label="Server state" value={selectedState} source="Persisted cloud diagnostic record" />
            <DataValue label="Reported 7-D physical action" value={actionText(action)} source="Server response in physical units; never client-generated" />
            <DataValue label="Inference timing" value={formatSeconds(timing.inferenceSeconds)} source="Server timing record" />
            <DataValue label="One-time load timing" value={formatSeconds(timing.loadSeconds)} source="Server timing record" />
          </div>
          {selectedNeedsManualReconciliation && <p className="inline-error" role="alert">{text(selected.error?.message, selected.error?.kind, selectedState === "ambiguous" ? "The cloud diagnostic state is ambiguous after restart." : "This cloud diagnostic failed.")} Use manual reconciliation; this page does not retry it.</p>}
          {selectedNeedsManualReconciliation && (
            <div className="mt-3 rounded border border-[var(--line-3)] p-3">
              <label className="flex items-start gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={manualAcknowledged}
                  onChange={(event) => setManualAcknowledged(event.target.checked)}
                />
                <span>I reviewed the previous failure; create a new request</span>
              </label>
              <p className="mt-2 text-pretty text-sm text-[var(--text-muted)]">
                The prior {selectedState} record may already have executed. This creates a new server request and may incur additional compute; it does not resubmit the old request.
              </p>
            </div>
          )}
          <div className="mt-3 flex flex-wrap gap-3 text-sm">
            {safeUrl(selected.report_url) && <a className="button button-secondary" href={safeUrl(selected.report_url)} target="_blank" rel="noreferrer">Open saved report <Glyph name="arrowUpRight" /></a>}
            {safeUrl(selected.raw_response_url) && <a className="button button-secondary" href={safeUrl(selected.raw_response_url)} target="_blank" rel="noreferrer">Open raw response <Glyph name="arrowUpRight" /></a>}
          </div>
        </section>
      )}

      <section className="mt-4" aria-labelledby="cloud-history-heading">
        <h3 className="text-balance text-base font-medium" id="cloud-history-heading">Saved diagnostic history</h3>
        {requests.length === 0 ? <EmptyState>No cloud diagnostic server record exists yet.</EmptyState> : (
          <ol className="mt-3 grid gap-2">
            {requests.map((item) => (
              <li key={item.request_id ?? `${item.status}-${requests.indexOf(item)}`}>
                <button
                  type="button"
                  className="button button-secondary w-full justify-between text-left"
                  aria-pressed={item.request_id === selectedId}
                  onClick={() => setSelectedId(item.request_id)}
                >
                  <span className="truncate">{item.request_id ?? "request id not reported"}</span>
                  <StatusPill status={pillStatus(statusLabel(item.status))}>{statusLabel(item.status)}</StatusPill>
                </button>
              </li>
            ))}
          </ol>
        )}
      </section>

      <Note summary="What this diagnostic can establish">
        This panel exposes only persisted server readiness and one fixed-fixture native-action request. A completed
        record shows that the server reported completion; it does not establish robot success, world fidelity,
        policy quality, calibration, or any qualified Baseten deployment claim.
      </Note>
    </Panel>
  );
}
