import { isRecord, type DemoJudgment } from "../lib/api";
import { useEffect, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { demoHeuristic } from "../lib/demoHeuristic";

function value(item: unknown): string {
  if (item === null || item === undefined) return "Not observed";
  if (typeof item === "number") return Number.isInteger(item) ? String(item) : item.toFixed(3);
  if (Array.isArray(item) && item.every((entry) => typeof entry !== "object")) return item.join(", ");
  if (typeof item === "object") return JSON.stringify(item, null, 2);
  return String(item);
}

function seconds(item: unknown) { return typeof item === "number" ? `${item.toFixed(3)} s` : "Not measured"; }
function gib(item: unknown) { return typeof item === "number" ? `${(item / 2 ** 30).toFixed(2)} GiB` : "Not measured"; }
function present(item: unknown) { return item !== null && item !== undefined && !["Not observed", "Not measured", "Not incurred in this request"].includes(String(item)) && !String(item).includes("—"); }
function quantile(values: number[], q: number) {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return undefined;
  return sorted[Math.max(0, Math.ceil(q * sorted.length) - 1)];
}

/** Only request-specific input, model identity, outputs and measured timings. */
export function InferenceBreakdown({ judgment, history = [] }: { judgment: DemoJudgment; history?: DemoJudgment[] }) {
  return <Dialog.Root>
    <Dialog.Trigger asChild><button type="button" className="button button-secondary">Telemetry</button></Dialog.Trigger>
    <Dialog.Portal>
      <Dialog.Overlay className="dialog-overlay" />
      <Dialog.Content className="freeplay-dialog judge-log-dialog">
        <header className="dialog-header"><div><Dialog.Title className="text-balance">Inference telemetry</Dialog.Title><Dialog.Description>Measured timings, memory, model identity, and saved responses.</Dialog.Description></div><Dialog.Close asChild><button type="button" className="button button-secondary">Close telemetry</button></Dialog.Close></header>
        <div className="judge-log-body"><TelemetryDetails judgment={judgment} history={history} /></div>
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>;
}

function TelemetryDetails({ judgment, history }: { judgment: DemoJudgment; history: DemoJudgment[] }) {
  const [observation, setObservation] = useState<Record<string, unknown>>({});
  const [allRuns, setAllRuns] = useState<DemoJudgment[]>();
  useEffect(() => {
    let current = true;
    void fetch("/api/demo/judge/performance").then((r) => r.ok ? r.json() : {}).then((v) => { if (current) setObservation(v); }).catch(() => undefined);
    void fetch("/api/demo/judgments").then((r) => r.ok ? r.json() : {}).then((v) => { if (current && isRecord(v) && Array.isArray(v.judgments)) setAllRuns(v.judgments as DemoJudgment[]); }).catch(() => undefined);
    return () => { current = false; };
  }, [judgment.status]);
  const input = judgment.input ?? {};
  const result = judgment.result ?? {};
  const report = isRecord(result.report) ? result.report : {};
  const receipt = isRecord(result.adapter_receipt) ? result.adapter_receipt : {};
  const timing = isRecord(result.timing) ? result.timing : {};
  const frames = Array.isArray(input.frames) ? input.frames.filter(isRecord) : [];
  const samples = Array.isArray(report.raw_judge_samples) ? report.raw_judge_samples.filter(isRecord) : [];
  const sampleTimes = samples.flatMap((sample) => {
    const attempts = Array.isArray(sample.attempts) ? sample.attempts.filter(isRecord) : [];
    return attempts.length && attempts.every((attempt) => typeof attempt.wall_seconds === "number") ? [attempts.reduce((sum, attempt) => sum + (attempt.wall_seconds as number), 0)] : [];
  });
  const generation = Array.isArray(result.generation_metrics) ? result.generation_metrics.filter(isRecord) : [];
  const deployment = isRecord(result.deployment) ? result.deployment : isRecord(input.deployment) ? input.deployment : {};
  const config = isRecord(input.sampling) ? input.sampling : {};
  const observedRuns = allRuns ?? history;
  const requestTimes = observedRuns.flatMap((entry) => typeof entry.result?.timing?.remote_request_seconds === "number" ? [entry.result.timing.remote_request_seconds] : []);
  const byTemperature = (cold: boolean) => observedRuns.flatMap((entry) => entry.result?.timing?.first_call_on_replica === cold && typeof entry.result.timing.remote_request_seconds === "number" ? [entry.result.timing.remote_request_seconds] : []);
  const retries = samples.reduce((sum, sample) => sum + (Array.isArray(sample.attempts) ? Math.max(0, sample.attempts.length - 1) : 0), 0);
  const performance: [string, unknown][] = [
    ["Model", receipt.base_model_id], ["Revision", receipt.base_model_revision],
    ["Active adapter", receipt.active_adapter], ["Adapter enabled", receipt.adapter_enabled],
    ["Request concurrency", deployment.concurrency], ["Replica bound", `${deployment.min_replicas ?? "—"}–${deployment.max_replicas ?? "—"}`],
    ["Active replicas · latest observation", observation.active_replicas],
    ["Replica observation time", typeof observation.active_replicas === "number" ? observation.observed_at : undefined],
    ["GPU", generation[0]?.gpu_type ?? observation.gpu_configuration],
    ["Peak allocated GPU memory", gib(timing.gpu_peak_memory_bytes)],
    ["Cold / warm request", timing.first_call_on_replica === true ? "First call after load" : timing.first_call_on_replica === false ? "Warm replica" : "Not observed"],
    ["Cold model load", timing.worker_load_seconds == null && timing.first_call_on_replica === false ? "Not incurred in this request" : seconds(timing.worker_load_seconds)],
    ["VLM inference", seconds(timing.worker_inference_seconds)],
    ["End-to-end remote latency", seconds(timing.remote_request_seconds)],
    ["Overhead outside inference", seconds(timing.outside_inference_seconds)],
    ["Preprocessing / decode", seconds(timing.preparation_seconds ?? input.preparation_seconds)],
    ["Admission to saved result", seconds(timing.action_to_persisted_seconds)],
    ["Validation / persistence", seconds(timing.validation_and_persistence_seconds)],
    ["Inference share of remote latency", typeof timing.worker_inference_seconds === "number" && typeof timing.remote_request_seconds === "number" && timing.remote_request_seconds > 0 ? `${(100 * timing.worker_inference_seconds / timing.remote_request_seconds).toFixed(1)}%` : undefined],
    ["Max new tokens", config.max_new_tokens], ["Samples / quorum", `${config.sample_count ?? "—"} / ${config.quorum ?? "—"}`],
    ["Retries used / allowed per sample", samples.length ? `${retries} total / ${config.retries_per_sample ?? "—"} per sample` : undefined],
    ["Sample median latency", seconds(quantile(sampleTimes, 0.5))],
    ["Sample p95 latency", seconds(quantile(sampleTimes, 0.95))],
    ["Fastest sample", seconds(quantile(sampleTimes, 0))],
    ["Slowest sample", seconds(quantile(sampleTimes, 1))],
    ["Model calls including retries", samples.length ? samples.length + retries : undefined],
    ["Valid final samples", samples.length ? `${samples.filter((sample) => { const attempts = Array.isArray(sample.attempts) ? sample.attempts.filter(isRecord) : []; const final = attempts.at(-1); return final && !final.failure_reason && isRecord(final.parsed); }).length} / ${samples.length}` : undefined],
  ];
  if (generation.length) {
    const numbers = (key: string) => generation.flatMap((entry) => typeof entry[key] === "number" ? [entry[key] as number] : []);
    performance.push(["Batch size", generation[0].batch_size],
      ["Median TTFT", seconds(quantile(numbers("ttft_seconds"), 0.5))],
      ["Input tokens (all attempts)", numbers("input_tokens").length === generation.length ? numbers("input_tokens").reduce((a,b)=>a+b,0) : undefined],
      ["Output tokens (all attempts)", numbers("output_tokens").length === generation.length ? numbers("output_tokens").reduce((a,b)=>a+b,0) : undefined],
      ["Median prefill GPU time", seconds(quantile(numbers("prefill_gpu_seconds"),0.5))],
      ["Median decode GPU time", seconds(quantile(numbers("decode_gpu_seconds"),0.5))],
      ["Median decode tokens/sec", quantile(numbers("decode_tokens_per_second"),0.5)],
      ["Device HBM used · post-call snapshot", gib(generation.at(-1)?.device_hbm_used_bytes_after)]);
  }
  const rows: [string, unknown][] = [
    ["Request ID", judgment.id], ["State", judgment.status],
    ["Video SHA-256", input.video_sha256], ["Frame indexes", input.frame_indexes],
    ["Frame timestamps (seconds)", input.frame_timestamps], ["Scene reference", input.scene_reference],
    ["Task", input.task_id], ["Rubric hash", input.rubric_hash],
    ["Base model", receipt.base_model_id], ["Base revision", receipt.base_model_revision],
    ["Processor revision", receipt.processor_revision],
    ["Adapter", receipt.adapter_id], ["Adapter tree hash", receipt.adapter_tree_sha256],
    ["Active adapter", receipt.active_adapter], ["Adapter enabled", receipt.adapter_enabled],
    ["Verified adapter layers", receipt.verified_enabled_layer_count],
    ["Deployment", result.deployment ?? input.deployment],
    ["Peak GPU memory (bytes)", timing.gpu_peak_memory_bytes],
    ["First call on replica", timing.first_call_on_replica],
    ["Cold load (seconds)", timing.worker_load_seconds],
    ["Total VLM inference (seconds)", timing.worker_inference_seconds],
    ["Decode and frame preparation (seconds)", timing.preparation_seconds ?? input.preparation_seconds],
    ["Remote request (seconds)", timing.remote_request_seconds],
    ["Remote overhead outside inference (seconds)", timing.outside_inference_seconds],
    ["Overhead scope", timing.outside_inference_scope],
    ["Validation and durable persistence (seconds)", timing.validation_and_persistence_seconds],
    ["Admission to persisted result (seconds)", timing.action_to_persisted_seconds],
    ["Total timing scope", timing.total_scope],
    ["Sampling", input.sampling], ["Seeds", input.seeds],
    ["Derived assessment", result.assessment], ["Unable-to-assess reason", report.missing_reason ?? judgment.error?.message ?? "None"],
    ["Demo completion heuristic · v1", demoHeuristic(judgment)],
  ];
  return <section className="inference-breakdown" aria-label="Inference performance">
    <h3 className="text-balance">Inference performance</h3>
    <dl className="performance-grid tabular-nums">{performance.filter(([, item]) => present(item)).map(([label, item]) => <div key={label}><dt>{label}</dt><dd>{value(item)}</dd></div>)}</dl>
    {sampleTimes.length > 0 && <p className="judge-muted tabular-nums">Per-sample latency, including retries: {sampleTimes.map((time, index) => `#${index + 1} ${time.toFixed(2)} s`).join(" · ")}. Summary over {sampleTimes.length} samples.</p>}
    {requestTimes.length > 0 && <details className="judge-details">
      <summary>Recorded request timings</summary>
      <p className="text-pretty">{requestTimes.length} recorded responses. Median remote latency: {seconds(quantile(requestTimes,0.5))}.</p>
      {byTemperature(false).length > 0 && <p>Warm median: {seconds(quantile(byTemperature(false), 0.5))} · {byTemperature(false).length} requests.</p>}
      {byTemperature(true).length > 0 && <p>First-call median: {seconds(quantile(byTemperature(true), 0.5))} · {byTemperature(true).length} requests.</p>}
      {requestTimes.length >= 20 && <p>p95: {seconds(quantile(requestTimes,0.95))}</p>}
      {requestTimes.length >= 100 && <p>p99: {seconds(quantile(requestTimes,0.99))}</p>}
    </details>}
    <details className="judge-details">
    <summary>Debug / Reproducibility</summary>
    <dl className="judge-receipt tabular-nums">
      {rows.filter(([,item]) => present(item)).map(([label, item]) => <div key={label}><dt>{label}</dt><dd><pre className="inference-value">{value(item)}</pre></dd></div>)}
    </dl>
    <h3 className="text-balance">Decoded frame hashes</h3>
    {frames.map((frame, index) => <p key={index} className="tabular-nums">Frame {String(frame.index)}: {String(frame.pixel_sha256)}</p>)}
    <h3 className="text-balance">Five-sample record</h3>
    <p className="text-pretty">Completion needs at least three decisive votes with intact visual integrity. Progress is the lower median among the agreeing votes.</p>
    {samples.map((sample, index) => {
      const attempts = Array.isArray(sample.attempts) ? sample.attempts.filter(isRecord) : [];
      const final = attempts[attempts.length - 1];
      const seconds = attempts.reduce((sum, attempt) => sum + (typeof attempt.wall_seconds === "number" ? attempt.wall_seconds : 0), 0);
      return <details key={index} className="inference-sample">
        <summary>Sample {index + 1} · {final?.parsed ? "Parsed" : "Unable to assess"} · {seconds.toFixed(2)} s · {Math.max(0, attempts.length - 1)} retries</summary>
        <p className="tabular-nums">Seed: {String(sample.seed)}</p>
        {attempts.map((attempt, attemptIndex) => <div key={attemptIndex}>
          <p className="tabular-nums">Attempt {attemptIndex + 1} · {value(attempt.wall_seconds)} s</p>
          <p>Raw output</p><pre className="inference-value">{value(attempt.raw_output)}</pre>
          <p>Parsed result</p><pre className="inference-value">{value(attempt.parsed)}</pre>
          <p>Failure: {value(attempt.failure_reason ?? "None")}</p>
        </div>)}
      </details>;
    })}
    </details>
  </section>;
}
