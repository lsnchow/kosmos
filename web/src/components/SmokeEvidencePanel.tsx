import { Activity, ExternalLink, ShieldAlert } from "lucide-react";
import { artifactUrl, type Experiment } from "../lib/api";
import {
  formatBytes,
  formatCount,
  formatSeconds,
  joinStrings,
  pickNumber,
  pickString,
  formatCountPair,
} from "../lib/format";
import { EmptyState, Note, Panel, StatusPill } from "./Primitives";

type Profile = {
  title: string;
  context: string;
  timingLabel?: string;
  needsIntegrityWarning: boolean;
};

export function experimentProfile(experiment: Experiment): Profile {
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
      context:
        "15 supplied actions include future actions; this is not fresh policy feedback or a qualified evaluation.",
      needsIntegrityWarning: true,
    };
  }
  if (
    kind.includes("qwen") ||
    model.includes("qwen") ||
    String(experiment.stage ?? "").toLowerCase() === "judge"
  ) {
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

/**
 * Imported cluster smoke reports.
 *
 * Every record the endpoint returns is rendered. The previous build filtered out
 * kinds containing "synthetic", which was dead code: `experiments_payload` only
 * emits from a fixed allow-list of real-model kinds and skips anything else, so
 * the filter could never match and only obscured that fact.
 */
export function SmokeEvidencePanel({ experiments }: { experiments: Experiment[] }) {
  return (
    <Panel title="Real model smoke evidence" eyebrow="Imported cluster report" className="smoke-panel">
      {experiments.length === 0 ? (
        <EmptyState className="smoke-empty" icon={<Activity aria-hidden="true" className="size-5" />}>
          No imported real-model smoke report has been returned by the API.
        </EmptyState>
      ) : (
        <div className="smoke-list">
          {experiments.map((experiment, index) => {
            const reportUrl = artifactUrl(experiment.report_url);
            const videoUrl = artifactUrl(experiment.video_url);
            const profile = experimentProfile(experiment);
            const label = pickString(experiment.model, experiment.id) ?? `Smoke report ${index + 1}`;
            const notes = joinStrings(experiment.notes);
            const models = joinStrings(experiment.models);
            const outcome = pickString(experiment.outcome) ?? "unknown";
            const totalSeconds = pickNumber(experiment.total_seconds);
            const latencySeconds = pickNumber(experiment.latency_seconds);
            const modelLoadSeconds = pickNumber(experiment.model_load_seconds);
            const ticksCompleted = pickNumber(experiment.ticks_completed);
            const ticksRequested = pickNumber(experiment.ticks_requested);
            const actionDimensions = pickNumber(experiment.action_dimensions);
            const timingScope = pickString(experiment.timing_scope);
            const excludesLoad = timingScope?.endsWith("_excludes_model_load") ?? false;
            const callLabel = excludesLoad ? "Inference (excl. load)" : "Measured call";
            return (
              <article className="smoke-card" key={pickString(experiment.id, experiment.report_url) ?? index}>
                <div className="smoke-card-topline">
                  <div>
                    <strong>{profile.title}</strong>
                    <span>{label}</span>
                  </div>
                  <StatusPill status={experiment.qualification ?? "unknown"}>
                    {String(experiment.qualification ?? "unknown")}
                  </StatusPill>
                </div>
                <div className="smoke-context">
                  <span>Stage: {pickString(experiment.stage, experiment.kind) ?? "not recorded"}</span>
                  <span>Report state: {String(experiment.status ?? "unknown")}</span>
                  <span>Timing scope: {timingScope ?? "not declared"}</span>
                  <span>
                    Task outcome: <b>{outcome}</b> · no task score
                  </span>
                </div>
                {videoUrl && (
                  <div className="smoke-video-wrap">
                    <video
                      className="smoke-video"
                      controls
                      preload="metadata"
                      playsInline
                      src={videoUrl}
                      aria-label={`Persisted diagnostic video for ${profile.title}: ${label}`}
                    />
                    <span>Persisted clip · {formatCount(experiment.frame_count)} returned frames</span>
                  </div>
                )}
                <dl className="smoke-metrics tabular-nums">
                  <div>
                    <dt>Frames</dt>
                    <dd>{formatCount(experiment.frame_count)}</dd>
                  </div>
                  <div>
                    <dt>Ticks</dt>
                    <dd>
                      {ticksCompleted === undefined && ticksRequested === undefined
                        ? "—"
                        : formatCountPair(ticksCompleted, ticksRequested)}
                    </dd>
                  </div>
                  {actionDimensions !== undefined && (
                    <div>
                      <dt>Action shape</dt>
                      <dd>{formatCount(actionDimensions)}D</dd>
                    </div>
                  )}
                  {totalSeconds !== undefined && (
                    <div>
                      <dt>{profile.timingLabel ?? "Total (incl. load)"}</dt>
                      <dd>{formatSeconds(totalSeconds)}</dd>
                    </div>
                  )}
                  {latencySeconds !== undefined && (
                    <div>
                      <dt>{callLabel}</dt>
                      <dd>{formatSeconds(latencySeconds)}</dd>
                    </div>
                  )}
                  {modelLoadSeconds !== undefined && (
                    <div>
                      <dt>Model load</dt>
                      <dd>{formatSeconds(modelLoadSeconds)}</dd>
                    </div>
                  )}
                  <div>
                    <dt>Peak GPU memory</dt>
                    <dd>{formatBytes(experiment.gpu_peak_memory_bytes)}</dd>
                  </div>
                </dl>
                {models && <p className="smoke-models truncate">Models: {models}</p>}
                <div className="smoke-links">
                  {reportUrl && (
                    <a
                      href={reportUrl}
                      target="_blank"
                      rel="noreferrer"
                      aria-label={`Open report for ${label}, ${experiment.id ?? profile.title} (new tab)`}
                    >
                      Report <ExternalLink aria-hidden="true" className="size-3" />
                    </a>
                  )}
                  {videoUrl && (
                    <a
                      href={videoUrl}
                      target="_blank"
                      rel="noreferrer"
                      aria-label={`Open video for ${label}, ${experiment.id ?? profile.title} (new tab)`}
                    >
                      Open video <ExternalLink aria-hidden="true" className="size-3" />
                    </a>
                  )}
                </div>
                {profile.needsIntegrityWarning && (
                  <p className="smoke-warning text-pretty">
                    <ShieldAlert aria-hidden="true" className="size-4" />
                    World-video integrity: unknown. Inspecting a distorted frame does not establish fidelity.
                  </p>
                )}
                {notes && <p className="smoke-notes text-pretty">{notes}</p>}
                <Note>
                  {profile.context} Runtime completion does not mean task success; timings are stage-scoped,
                  not cross-kind throughput comparisons.
                </Note>
              </article>
            );
          })}
        </div>
      )}
    </Panel>
  );
}
