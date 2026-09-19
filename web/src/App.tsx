import * as AlertDialog from "@radix-ui/react-alert-dialog";
import * as Dialog from "@radix-ui/react-dialog";
import {
  Activity,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ChevronRight,
  CircleAlert,
  Clock3,
  Expand,
  ExternalLink,
  Gauge,
  Play,
  Radio,
  RotateCcw,
  ShieldAlert,
  SlidersHorizontal,
  Square,
  X,
} from "lucide-react";
import {
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  api,
  artifactUrl,
  isRecord,
  type AnalysisCell,
  type Episode,
  type Experiment,
  type Gate,
  type JsonRecord,
  type Run,
  type SweepPoint,
} from "./lib/api";
import { cn } from "./lib/utils";

type ProtocolOption = { id: string; name: string };
type Telemetry = JsonRecord | undefined;

const EMPTY_TILES = Array.from({ length: 12 }, (_, index) => index);
const ZERO_ACTION = [0, 0, 0, 0, 0, 0, 0];
const ACTION_STEP = 0.08;

function asOptions(value: unknown): ProtocolOption[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (typeof item === "string") return [{ id: item, name: item }];
    if (!isRecord(item)) return [];
    const id = pickString(item.id, item.name, item.policy, item.task, item.slug);
    const name = pickString(item.display_name, item.label, item.name, item.id) ?? id;
    return id && name ? [{ id, name }] : [];
  });
}

function pickString(...values: unknown[]): string | undefined {
  return values.find((value): value is string => typeof value === "string" && value.length > 0);
}

function pickNumber(...values: unknown[]): number | undefined {
  return values.find((value): value is number => typeof value === "number" && Number.isFinite(value));
}

function isTerminal(status: unknown) {
  return ["completed", "complete", "failed", "cancelled", "canceled"].includes(
    String(status).toLowerCase(),
  );
}

function formatNumber(value: unknown) {
  const numeric = pickNumber(value);
  return numeric === undefined ? "—" : new Intl.NumberFormat("en-US").format(numeric);
}

function formatPercent(value: unknown) {
  const numeric = pickNumber(value);
  if (numeric === undefined) return "—";
  const normalized = numeric <= 1 ? numeric * 100 : numeric;
  return `${normalized.toFixed(normalized % 1 === 0 ? 0 : 1)}%`;
}

function formatUsd(value: unknown) {
  const numeric = pickNumber(value);
  return numeric === undefined
    ? "Unavailable"
    : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(numeric);
}

function formatSeconds(value: unknown) {
  const numeric = pickNumber(value);
  if (numeric === undefined) return "—";
  return `${numeric.toLocaleString("en-US", { maximumFractionDigits: 3 })} s`;
}

function formatBytes(value: unknown) {
  const numeric = pickNumber(value);
  if (numeric === undefined) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = numeric;
  let unitIndex = 0;
  while (amount >= 1024 && unitIndex < units.length - 1) {
    amount /= 1024;
    unitIndex += 1;
  }
  return `${amount.toLocaleString("en-US", { maximumFractionDigits: 2 })} ${units[unitIndex]}`;
}

function formatInterval(value: unknown) {
  if (Array.isArray(value)) return value.map(formatPercent).join(" – ");
  if (typeof value === "string") return value;
  if (isRecord(value)) {
    const lower = pickNumber(value.lower, value.min);
    const upper = pickNumber(value.upper, value.max);
    if (lower !== undefined && upper !== undefined) return `${formatPercent(lower)} – ${formatPercent(upper)}`;
  }
  return "—";
}

function experimentNotes(value: unknown) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.filter((item): item is string => typeof item === "string").join(" · ");
  return undefined;
}

function experimentModels(value: unknown) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.filter((item): item is string => typeof item === "string").join(" · ");
  return undefined;
}

function experimentProfile(experiment: Experiment) {
  const kind = String(experiment.kind ?? "").toLowerCase();
  const model = String(experiment.model ?? "").toLowerCase();
  if (kind === "cosmos3_nano_diffusers_smoke") {
    return {
      title: "Cosmos Nano forward-dynamics smoke",
      context: "Single forward-dynamics runtime smoke; it is not a closed-loop task evaluation.",
      needsIntegrityWarning: false,
    };
  }
  if (kind === "plumb_irasim_openvla_closed_loop_diagnostic") {
    return {
      title: "Closed-loop real-model diagnostic",
      context: "16-tick diagnostic; video integrity is not a task score.",
      needsIntegrityWarning: true,
    };
  }
  if (kind === "irasim_original_one_step_smoke") {
    return {
      title: "IRASim native one-step smoke",
      context: "Native IRASim request shape; unqualified smoke evidence.",
      needsIntegrityWarning: false,
    };
  }
  if (kind === "plumb_irasim_native_open_loop_reference_diagnostic") {
    return {
      title: "IRASim full-horizon open-loop reference",
      timingLabel: "Open-loop total",
      context: "15 supplied actions include future actions; this is not fresh policy feedback or a qualified evaluation.",
      needsIntegrityWarning: true,
    };
  }
  if (kind.includes("qwen") || model.includes("qwen") || String(experiment.stage ?? "").toLowerCase() === "judge") {
    return {
      title: "Qwen judge smoke",
      context: "Runtime evidence only; it is not a calibrated scoring result.",
      needsIntegrityWarning: false,
    };
  }
  if (kind === "plumb_local_policy_smoke") {
    return {
      title: "Local policy command smoke",
      context: "Policy/runtime exercise; not a world-fidelity or task-success result.",
      needsIntegrityWarning: false,
    };
  }
  return {
    title: "Imported real-model smoke",
    context: "Imported runtime evidence; qualification remains unknown.",
    needsIntegrityWarning: false,
  };
}

function formatDate(value: unknown) {
  if (typeof value !== "string") return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? value
    : new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit" }).format(date);
}

function statusClass(status: unknown) {
  const value = String(status ?? "unknown").toLowerCase();
  if (["pass", "completed", "complete", "healthy", "ready"].includes(value)) return "status-good";
  if (["fail", "failed", "blocked", "error"].includes(value)) return "status-bad";
  if (["running", "queued", "in_progress", "in-progress"].includes(value)) return "status-active";
  return "status-muted";
}

function StatusPill({ status, children }: { status?: unknown; children?: string }) {
  return <span className={cn("status-pill", statusClass(status))}>{children ?? String(status ?? "unknown")}</span>;
}

function Panel({
  title,
  eyebrow,
  children,
  className,
  action,
}: {
  title: string;
  eyebrow?: string;
  children: React.ReactNode;
  className?: string;
  action?: React.ReactNode;
}) {
  return (
    <section className={cn("panel", className)}>
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

function DataValue({ label, value, source }: { label: string; value: string; source: string }) {
  return (
    <div className="metric">
      <p>{label}</p>
      <strong className="tabular-nums">{value}</strong>
      <span>{source}</span>
    </div>
  );
}

function getEpisodeMedia(episode: Episode | undefined) {
  if (!episode) return undefined;
  return artifactUrl(
    pickString(episode.video_url, episode.frame_url, episode.artifact_path, episode.media_url, episode.artifact),
  );
}

function MediaFrame({ src, alt, className }: { src?: string; alt: string; className?: string }) {
  if (!src) {
    return (
      <div className={cn("media-empty", className)}>
        <Activity aria-hidden="true" className="size-5" />
        <span>Awaiting persisted media</span>
      </div>
    );
  }
  const isVideo = /\.(mp4|webm|mov)(\?|$)/i.test(src);
  return isVideo ? (
    <video className={cn("media", className)} controls muted playsInline src={src} aria-label={alt} />
  ) : (
    <img className={cn("media", className)} src={src} alt={alt} />
  );
}

function GatePanel({ gates }: { gates: Gate[] }) {
  const orderedGates = [...gates].sort((a, b) => (pickString(a.id, a.name) ?? "").localeCompare(pickString(b.id, b.name) ?? ""));
  return (
    <Panel title="Qualification gates" eyebrow="Evidence ledger" className="gates-panel">
      {orderedGates.length === 0 ? (
        <div className="empty-state">
          <ShieldAlert aria-hidden="true" className="size-5" />
          <p className="text-pretty">No gate ledger was returned. Qualified claims remain unavailable.</p>
        </div>
      ) : (
        <ol className="gate-list">
          {orderedGates.map((gate, index) => {
            const name = pickString(gate.name, gate.id) ?? `Gate ${index + 1}`;
            const reason = pickString(gate.reason, gate.summary, gate.blocker, gate.details, gate.description);
            return (
              <li key={`${name}-${index}`} className="gate-row">
                <div>
                  <div className="gate-name"><span className="gate-index">{index + 1}</span>{name}</div>
                  <p className="text-pretty">{reason ?? "No evidence reason recorded."}</p>
                </div>
                <StatusPill status={gate.status} />
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}

function RolloutTile({ episode, slot }: { episode?: Episode; slot: number }) {
  const episodeId = pickString(episode?.episode_id, episode?.id);
  const media = getEpisodeMedia(episode);
  return (
    <article className="rollout-tile">
      <MediaFrame src={media} alt={episodeId ? `Episode ${episodeId}` : "Unassigned rollout slot"} />
      <div className="tile-topline">
        <span>#{String(slot + 1).padStart(2, "0")}</span>
        <StatusPill status={episode?.status}>{episode ? String(episode.status ?? "persisted") : "waiting"}</StatusPill>
      </div>
      <div className="tile-caption">
        <div className="truncate">{pickString(episode?.policy, episode?.policy_name) ?? "No policy event"}</div>
        <span className="truncate">{pickString(episode?.task, episode?.task_id) ?? "Awaiting run"}</span>
      </div>
      {episodeId && <p className="tile-id truncate">{episodeId}</p>}
    </article>
  );
}

function Scoreboard({ cells, protocol }: { cells: AnalysisCell[]; protocol: JsonRecord | undefined }) {
  const sortedCells = [...cells].sort((a, b) => {
    const policy = String(a.policy ?? "").localeCompare(String(b.policy ?? ""));
    return policy || String(a.task ?? "").localeCompare(String(b.task ?? ""));
  });
  const backend = pickString(protocol?.backend_revision, protocol?.backend, protocol?.world_backend) ?? "not recorded";
  const judge = pickString(protocol?.judge_revision, protocol?.judge) ?? "not recorded";
  const parity = pickString(protocol?.parity, protocol?.scenario_parity) ?? "not recorded";

  return (
    <Panel
      title="Scoreboard"
      eyebrow="Synthetic fixture analysis · unqualified"
      className="score-panel"
      action={<span className="source-chip">published reference is separate</span>}
    >
      <div className="provenance-row">
        <span>Parity: <b>{parity}</b></span>
        <span>Backend: <b>{backend}</b></span>
        <span>Judge: <b>{judge}</b></span>
      </div>
      {sortedCells.length === 0 ? (
        <div className="empty-state scoreboard-empty">
          <Gauge aria-hidden="true" className="size-5" />
          <p className="text-pretty">No persisted analysis exists for the selected run. Provisional rollout counts are not presented as study statistics.</p>
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Policy / task</th>
                <th>Virtual rate</th>
                <th>Published reference</th>
                <th>Wilson interval</th>
                <th>Coverage</th>
                <th>Missingness bounds</th>
              </tr>
            </thead>
            <tbody>
              {sortedCells.map((cell, index) => {
                const bounds = formatInterval(cell.missing_bounds);
                const wilson = formatInterval(cell.wilson);
                return (
                  <tr key={`${String(cell.policy)}-${String(cell.task)}-${index}`}>
                    <td><strong>{String(cell.policy ?? "—")}</strong><span>{String(cell.task ?? "—")}</span></td>
                    <td className="tabular-nums">{formatPercent(cell.rate ?? cell.positive_rate)}</td>
                    <td className="tabular-nums">{formatPercent(cell.reference_rate)}</td>
                    <td className="tabular-nums">{wilson}</td>
                    <td className="tabular-nums">{formatPercent(cell.coverage)}</td>
                    <td className="tabular-nums">{bounds}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="table-note text-pretty">This local fixture analysis is not a real-policy result. Ordering stays indeterminate unless a qualified persisted analysis supports separation under its declared missingness and uncertainty protocol.</p>
    </Panel>
  );
}

function TelemetryStrip({ run, telemetry, eventReceivedAt }: { run?: Run; telemetry: Telemetry; eventReceivedAt?: Date }) {
  const ledgerSource = "Application ledger · logical episodes";
  const queueValue = pickNumber(telemetry?.platform_queue, telemetry?.queue_depth, telemetry?.queued);
  const activeReplicas = pickNumber(telemetry?.active_replicas, telemetry?.replicas_active);
  const desiredReplicas = pickNumber(telemetry?.desired_replicas, telemetry?.replicas_desired);
  const gpuSeconds = pickNumber(telemetry?.gpu_seconds, telemetry?.compute_seconds);
  const marginal = pickNumber(telemetry?.marginal_estimated_usd, telemetry?.marginal_usd);
  const total = pickNumber(telemetry?.total_estimated_usd, telemetry?.estimated_usd);
  const freshAt = pickString(telemetry?.fresh_at, telemetry?.updated_at) ?? (eventReceivedAt ? eventReceivedAt.toISOString() : undefined);

  return (
    <section className="telemetry" aria-label="Live telemetry">
      <div className="telemetry-title"><Radio aria-hidden="true" className="size-4" /><span>Live telemetry</span></div>
      <DataValue label="Completed" value={`${formatNumber(run?.completed)} / ${formatNumber(run?.total)}`} source={ledgerSource} />
      <DataValue label="Platform queue" value={formatNumber(queueValue)} source="Platform queue API" />
      <DataValue label="Active / desired" value={`${formatNumber(activeReplicas)} / ${formatNumber(desiredReplicas)}`} source="Deployment metrics" />
      <DataValue label="Compute seconds" value={formatNumber(gpuSeconds)} source="Stage instrumentation" />
      <DataValue label="Marginal cost" value={formatUsd(marginal)} source="Allocation ledger estimate" />
      <DataValue label="Total run cost" value={formatUsd(total)} source="Allocation ledger estimate" />
      <div className="freshness"><Clock3 aria-hidden="true" className="size-3.5" />{freshAt ? `Last event ${formatDate(freshAt)}` : "No telemetry event yet"}</div>
    </section>
  );
}

function SweepPanel({ sweeps }: { sweeps: { points: SweepPoint[]; status?: string } }) {
  const [selected, setSelected] = useState(0);
  const points = sweeps.points;
  const point = points[selected];
  useEffect(() => setSelected((current) => Math.min(current, Math.max(points.length - 1, 0))), [points.length]);

  return (
    <Panel title="Cost–fidelity operating point" eyebrow="Precomputed sweep only" className="sweep-panel">
      {points.length === 0 ? (
        <div className="empty-state">
          <SlidersHorizontal aria-hidden="true" className="size-5" />
          <p className="text-pretty">No persisted sweep points are available. The slider will not generate, estimate, or imply a cheaper setting.</p>
        </div>
      ) : (
        <>
          <div className="range-row">
            <label htmlFor="cost-sweep">Sweep point <span className="tabular-nums">{selected + 1} of {points.length}</span></label>
            <input id="cost-sweep" type="range" min="0" max={points.length - 1} value={selected} onChange={(event) => setSelected(Number(event.target.value))} />
          </div>
          <div className="sweep-readout">
            <div><span>Setting</span><strong>{pickString(point?.label, point?.id) ?? "Unnamed point"}</strong></div>
            <div><span>Estimated cost</span><strong className="tabular-nums">{formatUsd(point?.estimated_usd)}</strong></div>
            <div><span>Fixed task horizon</span><strong className="tabular-nums">{point?.horizon ?? "—"}</strong></div>
            <div><span>Coverage</span><strong className="tabular-nums">{formatPercent(point?.coverage)}</strong></div>
            <div><span>Qualification</span><StatusPill status={point?.qualified ? "pass" : "not_run"}>{point?.qualified ? "confirmed" : "not confirmed"}</StatusPill></div>
          </div>
        </>
      )}
      <p className="table-note text-pretty">Cost is an allocation-ledger estimate until billing reconciliation; short horizons are never treated as comparable savings.</p>
    </Panel>
  );
}

function SmokeEvidencePanel({ experiments }: { experiments: Experiment[] }) {
  const realExperiments = experiments.filter((experiment) => !String(experiment.kind ?? "").toLowerCase().includes("synthetic"));
  return (
    <Panel title="Real model smoke evidence" eyebrow="Imported cluster report" className="smoke-panel">
      {realExperiments.length === 0 ? (
        <div className="empty-state smoke-empty">
          <Activity aria-hidden="true" className="size-5" />
          <p className="text-pretty">No imported real-model smoke report has been returned by the API.</p>
        </div>
      ) : (
        <div className="smoke-list">
          {realExperiments.map((experiment, index) => {
            const reportUrl = artifactUrl(experiment.report_url);
            const videoUrl = artifactUrl(experiment.video_url);
            const profile = experimentProfile(experiment);
            const label = pickString(experiment.model, experiment.id) ?? `Smoke report ${index + 1}`;
            const notes = experimentNotes(experiment.notes);
            const models = experimentModels(experiment.models);
            const outcome = pickString(experiment.outcome) ?? "unknown";
            const totalSeconds = pickNumber(experiment.total_seconds);
            const latencySeconds = pickNumber(experiment.latency_seconds);
            const modelLoadSeconds = pickNumber(experiment.model_load_seconds);
            const ticksCompleted = pickNumber(experiment.ticks_completed);
            const ticksRequested = pickNumber(experiment.ticks_requested);
            const actionDimensions = pickNumber(experiment.action_dimensions);
            const timingScope = pickString(experiment.timing_scope);
            const inferenceExcludesLoad = timingScope?.endsWith("_excludes_model_load") ?? false;
            const callLabel = inferenceExcludesLoad ? "Inference (excl. load)" : "Measured call";
            const totalLabel = "Total (incl. load)";
            return (
              <article className="smoke-card" key={pickString(experiment.id, experiment.report_url) ?? index}>
                <div className="smoke-card-topline">
                  <div className="truncate"><strong className="truncate">{profile.title}</strong><span className="truncate">{label}</span></div>
                  <StatusPill status={experiment.qualification ?? "unknown"}>{experiment.qualification ?? "unknown"}</StatusPill>
                </div>
                <div className="smoke-context">
                  <span>Stage: {pickString(experiment.stage, experiment.kind) ?? "not recorded"}</span>
                  <span>Report state: {experiment.status ?? "unknown"}</span>
                  <span>Timing scope: {timingScope ?? "not declared"}</span>
                  <span>Task outcome: <b>{outcome}</b> · no task score</span>
                </div>
                {videoUrl && (
                  <div className="smoke-video-wrap">
                    <video className="smoke-video" controls preload="metadata" playsInline src={videoUrl} aria-label={`Persisted diagnostic video for ${profile.title}: ${label}`} />
                    <span>Persisted clip · {formatNumber(experiment.frame_count)} returned frames</span>
                  </div>
                )}
                <dl className="smoke-metrics tabular-nums">
                  <div><dt>Frames</dt><dd>{formatNumber(experiment.frame_count)}</dd></div>
                  <div><dt>Ticks</dt><dd>{ticksCompleted === undefined && ticksRequested === undefined ? "—" : `${formatNumber(ticksCompleted)} / ${formatNumber(ticksRequested)}`}</dd></div>
                  {actionDimensions !== undefined && <div><dt>Action shape</dt><dd>{formatNumber(actionDimensions)}D</dd></div>}
                  {totalSeconds !== undefined && <div><dt>{totalLabel}</dt><dd>{formatSeconds(totalSeconds)}</dd></div>}
                  {latencySeconds !== undefined && <div><dt>{callLabel}</dt><dd>{formatSeconds(latencySeconds)}</dd></div>}
                  {modelLoadSeconds !== undefined && <div><dt>Model load</dt><dd>{formatSeconds(modelLoadSeconds)}</dd></div>}
                  <div><dt>Peak GPU memory</dt><dd>{formatBytes(experiment.gpu_peak_memory_bytes)}</dd></div>
                </dl>
                {models && <p className="smoke-models truncate">Models: {models}</p>}
                <div className="smoke-links">
                  {reportUrl && <a href={reportUrl} target="_blank" rel="noreferrer" aria-label={`Open report for ${label}, ${experiment.id ?? profile.title} (new tab)`}>Report <ExternalLink aria-hidden="true" className="size-3" /></a>}
                  {videoUrl && <a href={videoUrl} target="_blank" rel="noreferrer" aria-label={`Open video for ${label}, ${experiment.id ?? profile.title} (new tab)`}>Open video <ExternalLink aria-hidden="true" className="size-3" /></a>}
                </div>
                {profile.needsIntegrityWarning && <p className="smoke-warning text-pretty"><ShieldAlert aria-hidden="true" className="size-3.5" />World-video integrity: unknown. Inspecting a distorted frame does not establish fidelity.</p>}
                {notes && <p className="smoke-notes text-pretty">{notes}</p>}
                <p className="table-note text-pretty">{profile.context} Runtime completion does not mean task success; timings are stage-scoped, not cross-kind throughput comparisons.</p>
              </article>
            );
          })}
        </div>
      )}
    </Panel>
  );
}

function FreeplayDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const [backend, setBackend] = useState<"synthetic" | "real">("synthetic");
  const [sessionId, setSessionId] = useState<string>();
  const [result, setResult] = useState<{
    mode?: string;
    frame_url?: string;
    state?: { x?: number; y?: number; z?: number; gripper?: number };
    latency_ms?: number;
    qualified?: boolean;
  }>();
  const [error, setError] = useState<string>();
  const [pending, setPending] = useState(false);
  const heldKeys = useRef(new Set<string>());

  const dispatch = useCallback(async (action: number[]) => {
    if (backend !== "synthetic") {
      setError("A qualified real-world control backend has not passed the required gates.");
      return;
    }
    setPending(true);
    setError(undefined);
    try {
      const response = await api.freeplayStep({ session_id: sessionId, action });
      setSessionId(response.session_id);
      setResult(response);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Free-play request failed.");
    } finally {
      setPending(false);
    }
  }, [backend, sessionId]);

  useEffect(() => {
    if (!open) return;
    const actionForKey = (key: string) => {
      if (key === "ArrowUp") return [0, ACTION_STEP, 0, 0, 0, 0, 0];
      if (key === "ArrowDown") return [0, -ACTION_STEP, 0, 0, 0, 0, 0];
      if (key === "ArrowLeft") return [-ACTION_STEP, 0, 0, 0, 0, 0, 0];
      if (key === "ArrowRight") return [ACTION_STEP, 0, 0, 0, 0, 0, 0];
      return undefined;
    };
    const keyDown = (event: KeyboardEvent) => {
      const action = actionForKey(event.key);
      if (!action || event.repeat || heldKeys.current.has(event.key)) return;
      event.preventDefault();
      heldKeys.current.add(event.key);
      void dispatch(action);
    };
    const keyUp = (event: KeyboardEvent) => {
      if (!heldKeys.current.delete(event.key)) return;
      event.preventDefault();
      void dispatch(ZERO_ACTION);
    };
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      if (heldKeys.current.size > 0) void dispatch(ZERO_ACTION);
      heldKeys.current.clear();
    };
  }, [dispatch, open]);

  const buttonHandlers = (action: number[]) => ({
    onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      void dispatch(action);
    },
    onPointerUp: () => void dispatch(ZERO_ACTION),
    onPointerLeave: () => void dispatch(ZERO_ACTION),
    onKeyDown: (event: ReactKeyboardEvent<HTMLButtonElement>) => {
      if ((event.key === " " || event.key === "Enter") && !event.repeat) void dispatch(action);
    },
    onKeyUp: (event: ReactKeyboardEvent<HTMLButtonElement>) => {
      if (event.key === " " || event.key === "Enter") void dispatch(ZERO_ACTION);
    },
  });
  const frame = artifactUrl(result?.frame_url);

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="freeplay-dialog" aria-describedby="freeplay-description">
          <header className="dialog-header">
            <div>
              <p className="eyebrow">Unscored simulator</p>
              <Dialog.Title className="text-balance">Free-play control</Dialog.Title>
            </div>
            <Dialog.Close className="icon-button" aria-label="Close free-play"><X aria-hidden="true" className="size-5" /></Dialog.Close>
          </header>
          <Dialog.Description id="freeplay-description" className="dialog-description text-pretty">
            Fixed simulated task input. This surface does not contribute to the scoreboard or make a real-world claim.
          </Dialog.Description>
          <div className="freeplay-layout">
            <div className="freeplay-stage">
              <MediaFrame src={frame} alt="Latest synthetic free-play frame" />
              <div className="stage-label">{result?.mode ?? "synthetic"} · {result?.qualified ? "qualified" : "unqualified"}</div>
            </div>
            <aside className="freeplay-controls">
              <label htmlFor="backend-select">Control backend</label>
              <select id="backend-select" value={backend} onChange={(event) => setBackend(event.target.value as "synthetic" | "real") }>
                <option value="synthetic">Synthetic simulator</option>
                <option value="real" disabled>Real backend — unavailable pending gates</option>
              </select>
              <p className="control-note text-pretty">Arrow keys or press-and-hold controls send bounded 7-D actions. Releasing sends a stop action.</p>
              <div className="dpad" aria-label="Directional robot controls">
                <span />
                <button type="button" aria-label="Move simulated arm up" {...buttonHandlers([0, ACTION_STEP, 0, 0, 0, 0, 0])}><ArrowUp aria-hidden="true" /></button>
                <span />
                <button type="button" aria-label="Move simulated arm left" {...buttonHandlers([-ACTION_STEP, 0, 0, 0, 0, 0, 0])}><ArrowLeft aria-hidden="true" /></button>
                <button type="button" aria-label="Stop simulated arm" onClick={() => void dispatch(ZERO_ACTION)}><Square aria-hidden="true" className="size-4" /></button>
                <button type="button" aria-label="Move simulated arm right" {...buttonHandlers([ACTION_STEP, 0, 0, 0, 0, 0, 0])}><ArrowRight aria-hidden="true" /></button>
                <span />
                <button type="button" aria-label="Move simulated arm down" {...buttonHandlers([0, -ACTION_STEP, 0, 0, 0, 0, 0])}><ArrowDown aria-hidden="true" /></button>
                <span />
              </div>
              <dl className="state-readout tabular-nums">
                <div><dt>x</dt><dd>{result?.state?.x ?? "—"}</dd></div>
                <div><dt>y</dt><dd>{result?.state?.y ?? "—"}</dd></div>
                <div><dt>z</dt><dd>{result?.state?.z ?? "—"}</dd></div>
                <div><dt>gripper</dt><dd>{result?.state?.gripper ?? "—"}</dd></div>
                <div><dt>server latency</dt><dd>{result?.latency_ms === undefined ? "—" : `${result.latency_ms} ms`}</dd></div>
              </dl>
              {pending && <p className="request-state" role="status">Sending bounded action…</p>}
              {error && <p className="inline-error" role="alert">{error}</p>}
            </aside>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export default function App() {
  const [health, setHealth] = useState<JsonRecord>();
  const [protocol, setProtocol] = useState<JsonRecord>();
  const [gates, setGates] = useState<Gate[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [sweeps, setSweeps] = useState<{ points: SweepPoint[]; status?: string }>({ points: [] });
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [activeRun, setActiveRun] = useState<Run>();
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisCell[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry>();
  const [eventReceivedAt, setEventReceivedAt] = useState<Date>();
  const [loadError, setLoadError] = useState<string>();
  const [actionError, setActionError] = useState<string>();
  const [launching, setLaunching] = useState(false);
  const [freeplayOpen, setFreeplayOpen] = useState(false);
  const [cancelOpen, setCancelOpen] = useState(false);

  const refreshBase = useCallback(async () => {
    setLoadError(undefined);
    const outcomes = await Promise.allSettled([api.health(), api.protocol(), api.gates(), api.runs(), api.sweeps(), api.experiments()]);
    // Imported smoke evidence is additive: an older control-plane process may
    // not expose it yet, without making the core ledger/protocol unavailable.
    const errors = outcomes.slice(0, 5).flatMap((outcome) => outcome.status === "rejected" ? [outcome.reason instanceof Error ? outcome.reason.message : "Request failed"] : []);
    const [healthResult, protocolResult, gatesResult, runsResult, sweepsResult, experimentsResult] = outcomes;
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    if (protocolResult.status === "fulfilled") setProtocol(protocolResult.value);
    if (gatesResult.status === "fulfilled") setGates(gatesResult.value.gates ?? []);
    if (runsResult.status === "fulfilled") setRuns(runsResult.value.runs ?? []);
    if (sweepsResult.status === "fulfilled") setSweeps({ points: sweepsResult.value.points ?? [], status: sweepsResult.value.status });
    if (experimentsResult.status === "fulfilled") setExperiments(experimentsResult.value.experiments ?? []);
    if (errors.length > 0) setLoadError(errors.join(" · "));
  }, []);

  useEffect(() => { void refreshBase(); }, [refreshBase]);

  const loadRun = useCallback(async (run: Run) => {
    const runId = run.id;
    if (!runId) return;
    setActiveRun(run);
    setEpisodes([]);
    setAnalysis([]);
    setActionError(undefined);
    const results = await Promise.allSettled([api.run(runId), api.episodes(runId), api.analysis(runId)]);
    if (results[0].status === "fulfilled") setActiveRun(results[0].value);
    if (results[1].status === "fulfilled") setEpisodes(results[1].value.episodes ?? []);
    if (results[2].status === "fulfilled") setAnalysis(results[2].value.cells ?? []);
    const failures = results.flatMap((result) => result.status === "rejected" ? [result.reason instanceof Error ? result.reason.message : "Run data request failed"] : []);
    if (failures.length) setActionError(failures.join(" · "));
  }, []);

  useEffect(() => {
    if (!activeRun?.id || isTerminal(activeRun.status)) return;
    const eventSource = new EventSource(`/api/runs/${encodeURIComponent(activeRun.id)}/events`);
    eventSource.addEventListener("snapshot", (event) => {
      try {
        const snapshot = JSON.parse((event as MessageEvent<string>).data) as JsonRecord;
        if (isRecord(snapshot.run)) {
          const nextRun = snapshot.run as Run;
          setActiveRun(nextRun);
          if (nextRun.id && isTerminal(nextRun.status)) {
            void api.analysis(nextRun.id).then((response) => setAnalysis(response.cells ?? [])).catch(() => undefined);
          }
        }
        if (Array.isArray(snapshot.episodes)) setEpisodes(snapshot.episodes as Episode[]);
        if (isRecord(snapshot.analysis) && Array.isArray(snapshot.analysis.cells)) setAnalysis(snapshot.analysis.cells as AnalysisCell[]);
        if (isRecord(snapshot.telemetry)) setTelemetry(snapshot.telemetry);
        setEventReceivedAt(new Date());
      } catch {
        setActionError("The run event stream returned malformed snapshot data.");
      }
    });
    eventSource.onerror = () => setEventReceivedAt((prior) => prior);
    return () => eventSource.close();
  }, [activeRun?.id, activeRun?.status]);

  const policies = useMemo(() => asOptions(protocol?.policies), [protocol]);
  const tasks = useMemo(() => asOptions(protocol?.tasks), [protocol]);
  const gateReasons = useMemo(() => {
    if (gates.length === 0) return ["Gate ledger has not been returned."];
    return gates
      .filter((gate) => String(gate.status).toLowerCase() !== "pass")
      .map((gate) => `${pickString(gate.name, gate.id) ?? "Gate"}: ${pickString(gate.reason, gate.summary, gate.blocker) ?? String(gate.status ?? "not_run")}`);
  }, [gates]);
  const canLaunchSynthetic = policies.length > 0 && tasks.length > 0 && !launching;
  const heroEpisode = episodes.find((episode) => pickString(episode.presentation, episode.track) === "hero") ?? episodes[0];

  const launchSynthetic = async () => {
    if (!canLaunchSynthetic) return;
    setLaunching(true);
    setActionError(undefined);
    try {
      const run = await api.createSyntheticRun({
        mode: "synthetic",
        backend: "synthetic",
        policies: policies.map((policy) => policy.id),
        tasks: tasks.map((task) => task.id),
        starts_per_task: 50,
        seed: 20260919,
        idempotency_key: crypto.randomUUID(),
      });
      setRuns((prior) => [run, ...prior.filter((item) => item.id !== run.id)]);
      await loadRun(run);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Could not submit the synthetic burst.");
    } finally {
      setLaunching(false);
    }
  };

  const cancelRun = async () => {
    if (!activeRun?.id) return;
    setActionError(undefined);
    try {
      const updated = await api.cancelRun(activeRun.id);
      setActiveRun(updated);
      setRuns((prior) => prior.map((run) => run.id === updated.id ? updated : run));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Could not cancel this run.");
    } finally {
      setCancelOpen(false);
    }
  };

  return (
    <div className="app-shell">
      <a className="skip-link" href="#console">Skip to console</a>
      <header className="topbar">
        <a className="brand" href="#console" aria-label="PLUMB measurement console home"><span className="brand-mark">P</span><span>PLUMB</span></a>
        <nav aria-label="Console sections"><a href="#rollouts">Rollouts</a><a href="#scoreboard">Scores</a><a href="#gates">Gates</a></nav>
        <div className="topbar-right">
          <span className="api-health"><span className={cn("health-dot", health ? "health-known" : "health-pending")} />API {pickString(health?.status) ?? "checking"}</span>
          <button type="button" className="button button-quiet" onClick={() => void refreshBase()}><RotateCcw aria-hidden="true" className="size-4" />Refresh</button>
        </div>
      </header>

      <main id="console" className="console">
        <section className="masthead">
          <div>
            <p className="eyebrow"><span className="fixture-dot" />Synthetic fixture mode</p>
            <h1 className="text-balance">Measure the ruler before trusting the ranking.</h1>
            <p className="masthead-copy text-pretty">PLUMB keeps generated-rollout evidence, published real-robot reference, missingness, and runtime accounting in one console. No qualified real-world conclusion is available until its gates pass.</p>
          </div>
          <div className="masthead-actions">
            <button type="button" className="button button-primary" disabled={!canLaunchSynthetic} onClick={() => void launchSynthetic()}>
              <Play aria-hidden="true" className="size-4" />{launching ? "Submitting…" : "Run 1,500 synthetic episodes"}
            </button>
            <button type="button" className="button button-secondary" onClick={() => setFreeplayOpen(true)}><Expand aria-hidden="true" className="size-4" />Open free-play</button>
            <p>6 policies × 5 tasks × 50 starts. Fixture scores are engineering tests, not robot results.</p>
          </div>
        </section>

        {(loadError || actionError) && <div className="error-banner" role="alert"><CircleAlert aria-hidden="true" className="size-5" /><span>{actionError ?? loadError}</span></div>}

        <section className="qualified-blocker" aria-label="Real burst qualification status">
          <div><ShieldAlert aria-hidden="true" className="size-5" /><div><strong>Qualified real burst unavailable</strong><p className="text-pretty">Real backends cannot be dispatched through this console until evidence gates qualify the chosen protocol.</p></div></div>
          <ul>{gateReasons.slice(0, 3).map((reason) => <li key={reason}>{reason}</li>)}</ul>
        </section>

        <section id="rollouts" className="workspace-grid">
          <Panel
            title="Rollout viewport"
            eyebrow="12 concurrent slots · persisted events only"
            className="rollouts-panel"
            action={<span className="source-chip">{activeRun?.id ? `run ${activeRun.id}` : "no run selected"}</span>}
          >
            <div className="rollout-grid">
              {EMPTY_TILES.map((slot) => <RolloutTile key={slot} slot={slot} episode={episodes[slot]} />)}
            </div>
            <p className="table-note text-pretty">Slots are a viewport, not a total count. Frames appear only after the application persists an episode event; replayed and cached media retain their source labels.</p>
          </Panel>

          <aside className="side-stack">
            <Panel title="Run ledger" eyebrow="Application ledger">
              <label className="select-label" htmlFor="run-select">Inspect persisted run</label>
              <select id="run-select" value={activeRun?.id ?? ""} onChange={(event) => {
                const run = runs.find((item) => item.id === event.target.value);
                if (run) void loadRun(run);
              }}>
                <option value="">Select a run</option>
                {runs.map((run) => <option key={run.id} value={run.id}>{run.id ?? "unnamed run"} · {run.status ?? "unknown"}</option>)}
              </select>
              <div className="run-metrics">
                <DataValue label="Status" value={String(activeRun?.status ?? "—")} source="Logical run ledger" />
                <DataValue label="Created" value={formatDate(activeRun?.created_at)} source="Logical run ledger" />
                <DataValue label="Evaluable" value={formatNumber(activeRun?.evaluable)} source="Logical run ledger" />
                <DataValue label="Failed" value={formatNumber(activeRun?.failed)} source="Logical run ledger" />
              </div>
              {activeRun && !isTerminal(activeRun.status) && <button type="button" className="button button-danger" onClick={() => setCancelOpen(true)}><Square aria-hidden="true" className="size-3.5" />Cancel run</button>}
            </Panel>
            <Panel title="480p presentation track" eyebrow="Separate, unscored">
              <div className="hero-frame"><MediaFrame src={getEpisodeMedia(heroEpisode)} alt="480p presentation rollout" /></div>
              <p className="table-note text-pretty">The presentation hero is distinct from the 256p primary scoring protocol. No 480p result is included in a qualified comparison without its own operating-point evidence.</p>
            </Panel>
          </aside>
        </section>

        <TelemetryStrip run={activeRun} telemetry={telemetry} eventReceivedAt={eventReceivedAt} />

        <section className="evidence-section" aria-label="Imported real-model evidence">
          <SmokeEvidencePanel experiments={experiments} />
        </section>

        <section id="scoreboard" className="lower-grid">
          <Scoreboard cells={analysis} protocol={protocol} />
          <div className="right-lower"><SweepPanel sweeps={sweeps} /><div id="gates"><GatePanel gates={gates} /></div></div>
        </section>

        <footer className="footer-note">
          <span>PLUMB measurement console</span>
          <span>Sources: application ledger, persisted artifacts, declared platform telemetry, and published reference.</span>
          <a href="/api/protocol" target="_blank" rel="noreferrer">Protocol record <ExternalLink aria-hidden="true" className="size-3" /></a>
        </footer>
      </main>

      <FreeplayDialog open={freeplayOpen} onOpenChange={setFreeplayOpen} />

      <AlertDialog.Root open={cancelOpen} onOpenChange={setCancelOpen}>
        <AlertDialog.Portal>
          <AlertDialog.Overlay className="dialog-overlay" />
          <AlertDialog.Content className="alert-dialog">
            <AlertDialog.Title>Cancel this run?</AlertDialog.Title>
            <AlertDialog.Description className="text-pretty">Cancellation leaves explicit terminal records and retains already allocated work and cost.</AlertDialog.Description>
            <div className="alert-actions"><AlertDialog.Cancel className="button button-secondary">Keep running</AlertDialog.Cancel><AlertDialog.Action className="button button-danger" onClick={() => void cancelRun()}>Cancel run</AlertDialog.Action></div>
          </AlertDialog.Content>
        </AlertDialog.Portal>
      </AlertDialog.Root>
    </div>
  );
}
