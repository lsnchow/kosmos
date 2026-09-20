import { useEffect, useRef, useState } from "react";
import { api, artifactUrl, isRecord, type DemoJudgeReadiness, type DemoJudgment, type DemoJudgeClip } from "../lib/api";
import { StatusPill } from "./Primitives";
import { InferenceBreakdown } from "./InferenceBreakdown";
import { JudgeLogs } from "./JudgeLogs";
import { demoHeuristic } from "../lib/demoHeuristic";

const TERMINAL = new Set(["completed", "abstained", "failed", "interrupted"]);

function judgmentKey() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `judge-${crypto.randomUUID()}`;
  }
  return `judge-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function statusText(status: string | undefined) {
  switch (status) {
    case "queued":
      return "Queued";
    case "warming":
      return "Waiting for trained judge";
    case "assessing":
      return "Assessing five samples";
    case "completed":
      return "Assessment complete";
    case "abstained":
      return "Unable to assess";
    case "interrupted":
      return "Assessment interrupted";
    case "failed":
      return "Assessment failed";
    default:
      return "Not assessed";
  }
}

function detailValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return `${value.toFixed(2)} s`;
  return typeof value === "string" && value ? value : "Not recorded";
}

export function Assessment({ judgment }: { judgment?: DemoJudgment }) {
  const assessment = judgment?.result?.assessment;
  const heuristic = demoHeuristic(judgment);
  if (!judgment) {
    return <p className="judge-muted text-pretty">Run the trained judge to save an assessment beside this clip.</p>;
  }
  if (!TERMINAL.has(judgment.status)) {
    return (
      <div className="judge-progress" role="status" aria-live="polite">
        <StatusPill status={judgment.status}>{statusText(judgment.status)}</StatusPill>
        {judgment.progress && <p className="tabular-nums">{judgment.progress.completed_samples ?? 0} / 5 samples complete</p>}
      </div>
    );
  }
  if (judgment.status !== "completed" || !assessment || assessment.status !== "evaluable") {
    if (heuristic) return <div className="judge-result" role="status">
      <p className="eyebrow">Demo heuristic</p>
      <h3 className="text-balance">{heuristic.label}</h3>
      <p className="judge-muted text-pretty">{heuristic.votes}/{heuristic.total} completion votes agree. Completion-only estimate; visual-integrity and collision votes are excluded. Original judge result and all votes are retained in telemetry.</p>
    </div>;
    const report = judgment.result?.report;
    const samples = Array.isArray(report?.raw_judge_samples) ? report.raw_judge_samples.filter(isRecord) : [];
    const observations = samples.flatMap((sample) => {
      const attempts = Array.isArray(sample.attempts) ? sample.attempts.filter(isRecord) : [];
      const final = attempts.at(-1)?.parsed;
      return isRecord(final) ? [final] : [];
    });
    const counts = (field: string) => {
      const tallies = new Map<string, number>();
      for (const observation of observations) {
        const value = observation[field];
        if (value !== undefined && value !== null) tallies.set(String(value), (tallies.get(String(value)) ?? 0) + 1);
      }
      return [...tallies].map(([value, count]) => `${value.replaceAll("_", " ")}: ${count}/${samples.length} samples`).join(" · ");
    };
    return (
      <div className="judge-unable" role="status">
        <p className="text-pretty">{observations.length ? "Model observations · final verdict withheld" : "No valid assessment returned"}</p>
        {observations.length > 0 && <>
          <dl className="judge-metrics tabular-nums">
            <div><dt>Reported completion</dt><dd>{counts("completion_evidence")}</dd></div>
            <div><dt>Reported progress · 0–5</dt><dd>{counts("progress")}</dd></div>
            <div><dt>Visual integrity</dt><dd>{counts("integrity")}</dd></div>
            <div><dt>Collision observations</dt><dd>{counts("collision")}</dd></div>
          </dl>
          <p className="judge-muted text-pretty">These are the model’s claims. Only {assessment?.quorum ?? 0}/{samples.length} samples support a decisive verdict with intact visual integrity; three are required.</p>
          {observations.length < samples.length && <p className="judge-muted">{samples.length - observations.length} sample(s) produced invalid or missing output.</p>}
        </>}
        {!observations.length && assessment && <p className="judge-muted text-pretty">The model did not return usable structured observations. Raw attempts are retained in Debug / Reproducibility.</p>}
        {judgment.error?.message && <p className="judge-muted text-pretty">{judgment.error.message}</p>}
      </div>
    );
  }
  return (
    <div className="judge-result" aria-live="polite">
      <StatusPill status="completed">Assessment complete</StatusPill>
      <dl className="judge-metrics tabular-nums">
        <div><dt>Task progress</dt><dd>{assessment.progress ?? "Not recorded"} / 5</dd></div>
        <div><dt>Visual integrity</dt><dd>{detailValue(assessment.visual_integrity)}</dd></div>
        <div><dt>Collision</dt><dd>{detailValue(assessment.collision)}</dd></div>
        <div><dt>Completion</dt><dd>{detailValue(assessment.completion)}</dd></div>
      </dl>
      {assessment.explanation && <p className="judge-explanation text-pretty">{assessment.explanation}</p>}
      {assessment.evidence_frame_indices?.length ? (
        <p className="judge-muted tabular-nums">Evidence frames: {assessment.evidence_frame_indices.join(", ")}</p>
      ) : null}
    </div>
  );
}

export function DemoJudgePanel({ clipOverride, onStatusChange, automatic = false, providedJudgment }: { clipOverride?: DemoJudgeClip; onStatusChange?: (status: string) => void; automatic?: boolean; providedJudgment?: DemoJudgment | null }) {
  const [readiness, setReadiness] = useState<DemoJudgeReadiness>();
  const [judgment, setJudgment] = useState<DemoJudgment>();
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string>();
  const [history, setHistory] = useState<DemoJudgment[]>([]);
  useEffect(() => { if (providedJudgment) { setJudgment(providedJudgment); setHistory((old) => [providedJudgment, ...old.filter((item) => item.id !== providedJudgment.id)]); } }, [providedJudgment]);
  useEffect(() => { if (judgment?.status) onStatusChange?.(judgment.status); }, [judgment?.status, onStatusChange]);
  const requestKey = useRef(judgmentKey());
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!judgment || TERMINAL.has(judgment.status)) return;
    const start = Date.parse(judgment.created_at ?? "");
    const timer = window.setInterval(() => setElapsed(Math.max(0, (Date.now() - start) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [judgment?.id, judgment?.status]);

  useEffect(() => {
    let active = true;
    void api
      .demoJudgeReadiness()
      .then(async (next) => {
        if (!active) return;
        setReadiness(next);
        const clipId = clipOverride?.id ?? next.default_clip?.id;
        if (!clipId) return;
        const saved = await api.demoJudgments(clipId);
        if (active) { setHistory(saved.judgments ?? []); setJudgment(saved.judgments?.[0]); }
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "The trained judge could not be reached.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [clipOverride?.id]);

  useEffect(() => {
    if (!judgment?.id || TERMINAL.has(judgment.status)) return;
    let active = true;
    const poll = async () => {
      try {
        const next = await api.demoJudgment(judgment.id);
        if (active) { setJudgment(next); setHistory((old) => [next, ...old.filter((item) => item.id !== next.id)]); }
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : "The assessment status could not be refreshed.");
      }
    };
    const timer = window.setInterval(() => void poll(), 1500);
    void poll();
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [judgment?.id, judgment?.status]);

  const clip = clipOverride ?? readiness?.default_clip;
  const video = artifactUrl(clip?.video_url);
  const canSubmit = Boolean(clip?.id && readiness?.available && !submitting && (!judgment || TERMINAL.has(judgment.status)));
  const assess = async () => {
    if (!clip?.id || !canSubmit) return;
    setSubmitting(true);
    setError(undefined);
    try {
      const next = await api.createDemoJudgment({
        clip_id: clip.id,
        profile: "semantic_pilot_epoch_02",
        idempotency_key: requestKey.current,
      });
      setJudgment(next);
      requestKey.current = judgmentKey();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The assessment request could not be created.");
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return <section className="judge-panel" aria-busy="true"><p role="status">Loading saved assessment example…</p></section>;
  }
  if (!clip || !video) {
    return <section className="judge-panel"><p className="text-pretty" role="status">No assessment-ready saved drawer clip is available.</p></section>;
  }

  return (
    <section className="judge-panel" aria-labelledby="trained-judge-heading">
      <div className="judge-layout">
        <div className="judge-clip">
          <p className="eyebrow">{clipOverride ? "Generated in this run" : "Saved playback"}</p>
          <h2 className="text-balance">{clip.task_label ?? "Close the drawer"}</h2>
          <p className="judge-muted text-pretty">{clip.action_source ?? "Recorded action source"} · {clip.controller_identity ?? "Controller identity not recorded"}</p>
          <video className="judge-video" controls muted playsInline preload="metadata" src={video} aria-label={`Generated recording for ${clip.task_label ?? "Close the drawer"}`} />
          <details className="judge-details">
            <summary>Recording details</summary>
            <p className="text-pretty">Scene reference role: {clip.scene_reference_role ?? "not recorded"}. It is not labelled as a goal image.</p>
            <p className="tabular-nums">Video hash: {clip.video_sha256 ?? "not recorded"}</p>
          </details>
        </div>
        <div className="judge-assessment">
          <p className="eyebrow">LoRA-post-trained rollout judge</p>
          <p className="judge-muted">Qwen2.5-VL · semantic epoch-02</p>
          <h2 id="trained-judge-heading" className="text-balance">Assessment</h2>
          {judgment && <JudgeLogs judgment={judgment} />}
          {!automatic && <button type="button" className="button button-primary" onClick={() => void assess()} disabled={!canSubmit}>
            {submitting ? "Saving assessment request…" : "Assess with our trained judge"}
          </button>}
          {!readiness?.available && <p className="judge-muted text-pretty">{readiness?.reason ?? "The trained judge is unavailable."}</p>}
          {error && <p className="inline-error text-pretty" role="alert">{error}</p>}
          <Assessment judgment={judgment} />
          {history.length > 1 && <div className="policy-field"><label htmlFor={`judge-history-${clip.id}`}>Past assessments</label><select id={`judge-history-${clip.id}`} value={judgment?.id ?? ""} onChange={(event) => setJudgment(history.find((item) => item.id === event.target.value))}>
            {history.map((item) => <option key={item.id} value={item.id}>{item.created_at ? new Date(item.created_at).toLocaleString() : item.id} · {item.status}</option>)}
          </select></div>}
          {judgment && !TERMINAL.has(judgment.status) && <p className="judge-muted tabular-nums">Elapsed: {Math.floor(elapsed)} s</p>}
        </div>
      </div>
      {judgment && <InferenceBreakdown judgment={judgment} history={history} />}
    </section>
  );
}
