import { useEffect, useState } from "react";
import { api, artifactUrl, type DemoJudgeReadiness, type DemoJudgment } from "../lib/api";
import { StatusPill } from "./Primitives";

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
      return "Warming trained judge";
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

function Assessment({ judgment }: { judgment?: DemoJudgment }) {
  const assessment = judgment?.result?.assessment;
  if (!judgment) {
    return <p className="judge-muted text-pretty">Run the trained judge to save an assessment beside this clip.</p>;
  }
  if (!TERMINAL.has(judgment.status)) {
    return (
      <div className="judge-progress" role="status" aria-live="polite">
        <StatusPill status={judgment.status}>{statusText(judgment.status)}</StatusPill>
        <p className="text-pretty">This request is persisted. Reloading will reconnect to it without submitting another inference request.</p>
      </div>
    );
  }
  if (judgment.status !== "completed" || !assessment || assessment.status !== "evaluable") {
    return (
      <div className="judge-unable" role="status">
        <StatusPill status={judgment.status}>{statusText(judgment.status)}</StatusPill>
        <p className="text-pretty">Unable to assess</p>
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

export function DemoJudgePanel() {
  const [readiness, setReadiness] = useState<DemoJudgeReadiness>();
  const [judgment, setJudgment] = useState<DemoJudgment>();
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string>();

  useEffect(() => {
    let active = true;
    void api
      .demoJudgeReadiness()
      .then(async (next) => {
        if (!active) return;
        setReadiness(next);
        const clipId = next.default_clip?.id;
        if (!clipId) return;
        const saved = await api.demoJudgments(clipId);
        if (active) setJudgment(saved.judgments?.[0]);
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
  }, []);

  useEffect(() => {
    if (!judgment?.id || TERMINAL.has(judgment.status)) return;
    let active = true;
    const poll = async () => {
      try {
        const next = await api.demoJudgment(judgment.id);
        if (active) setJudgment(next);
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

  const clip = readiness?.default_clip;
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
        idempotency_key: judgmentKey(),
      });
      setJudgment(next);
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
          <p className="eyebrow">Saved example</p>
          <h2 className="text-balance">{clip.task_label ?? "Close the drawer"}</h2>
          <p className="judge-muted text-pretty">{clip.action_source ?? "Recorded action source"} · {clip.controller_identity ?? "Controller identity not recorded"}</p>
          <video className="judge-video" controls playsInline preload="metadata" src={video} aria-label={`Saved robot recording for ${clip.task_label ?? "Close the drawer"}`} />
          <details className="judge-details">
            <summary>Recording details</summary>
            <p className="text-pretty">Scene reference role: {clip.scene_reference_role ?? "not recorded"}. It is not labelled as a goal image.</p>
            <p className="tabular-nums">Video hash: {clip.video_sha256 ?? "not recorded"}</p>
          </details>
        </div>
        <div className="judge-assessment">
          <p className="eyebrow">Our post-trained judge</p>
          <h2 id="trained-judge-heading" className="text-balance">Assessment</h2>
          <p className="judge-disclosure text-pretty">Experimental judge — not calibrated against human ratings.</p>
          <button type="button" className="button button-primary" onClick={() => void assess()} disabled={!canSubmit}>
            {submitting ? "Saving assessment request…" : "Assess with our trained judge"}
          </button>
          {!readiness?.available && <p className="judge-muted text-pretty">{readiness?.reason ?? "The trained judge is unavailable."}</p>}
          {error && <p className="inline-error text-pretty" role="alert">{error}</p>}
          <Assessment judgment={judgment} />
          {judgment?.result?.adapter_receipt && (
            <details className="judge-details">
              <summary>Adapter and timing receipt</summary>
              <dl className="judge-receipt tabular-nums">
                <div><dt>Adapter</dt><dd>{detailValue(judgment.result.adapter_receipt.adapter_id)}</dd></div>
                <div><dt>Adapter hash</dt><dd>{detailValue(judgment.result.adapter_receipt.adapter_tree_sha256)}</dd></div>
                <div><dt>Base model</dt><dd>{detailValue(judgment.result.adapter_receipt.base_model_revision)}</dd></div>
                <div><dt>Request time</dt><dd>{detailValue(judgment.result.timing?.client_request_seconds)}</dd></div>
              </dl>
            </details>
          )}
        </div>
      </div>
    </section>
  );
}
