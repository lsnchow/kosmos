import { useState, type FormEvent } from "react";
import { cn } from "../lib/utils";

const STORAGE_KEY = "kosmos-policy-draft-v1";
const stages = [
  { title: "Verify inputs", detail: "Check the policy configuration, task and rollout length." },
  { title: "Generate rollout", detail: "Run the controller through the world model and record the video." },
  { title: "Visual consistency", detail: "Python checks for frame artifacts, discontinuities and suspicious motion. These checks cannot prove physical accuracy." },
  { title: "Judge task", detail: "The trained visual judge assesses progress and task completion." },
  { title: "Save result", detail: "Keep the video, checks and assessment together." },
];
type Draft = { name: string; source: string; endpoint: string; task: string; steps: string };
const initial: Draft = { name: "My policy experiment", source: "OpenVLA", endpoint: "", task: "close_drawer", steps: "70" };
function readDraft(): Draft {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (stored && Object.keys(initial).every((key) => typeof stored[key] === "string")) return stored;
  } catch { /* Browser storage is optional. */ }
  return initial;
}

/** Local configuration only until the complete evaluation service is available. */
export function PolicyExperiment() {
  const [draft, setDraft] = useState<Draft>(readDraft);
  const [open, setOpen] = useState(false);
  const [verified, setVerified] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState("");
  const [inspected, setInspected] = useState(0);
  const update = (key: keyof Draft, value: string) => {
    setDraft((prior) => ({ ...prior, [key]: value }));
    setVerified(false);
    setNotice("");
    setErrors({});
  };
  const verify = (event: FormEvent) => {
    event.preventDefault();
    const next: Record<string, string> = {};
    if (!draft.name.trim()) next.name = "Give this experiment a name.";
    if (!["OpenVLA", "custom"].includes(draft.source)) next.source = "Choose a policy source.";
    if (draft.task !== "close_drawer") next.task = "Choose the supported drawer task.";
    if (!/^\d+$/.test(draft.steps) || Number(draft.steps) < 1 || Number(draft.steps) > 70) next.steps = "Use a whole number from 1 to 70.";
    if (draft.source === "custom") {
      try {
        const url = new URL(draft.endpoint);
        if (!["https:", "http:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error();
      } catch { next.endpoint = "Enter an HTTP(S) endpoint without credentials, query parameters or fragments."; }
    }
    setErrors(next);
    if (Object.keys(next).length) return;
    setVerified(true);
    setInspected(1);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(draft));
      setNotice("Configuration checked and draft saved in this browser. Endpoint connectivity and model compatibility await compute.");
    } catch {
      setNotice("Configuration checked. Browser storage is unavailable; keep this page open to retain your draft.");
    }
  };
  return (
    <section className="policy-experiment" aria-labelledby="policy-experiment-title">
      <div className="policy-experiment-header">
        <div>
          <p className="eyebrow">New experiment</p>
          <h2 id="policy-experiment-title" className="text-balance">Test a policy</h2>
        </div>
        <button className="button button-secondary" type="button" aria-expanded={open} aria-controls="policy-configuration" onClick={() => setOpen(!open)}>
          {open ? "Hide setup" : "Configure experiment"} <span aria-hidden="true">{open ? "−" : "+"}</span>
        </button>
      </div>

      {open && <form id="policy-configuration" className="policy-configuration" onSubmit={verify} noValidate>
        <div className="policy-fields">
          <div className="policy-field">
            <label htmlFor="experiment-name">Experiment name</label>
            <input id="experiment-name" required maxLength={120} value={draft.name} onChange={(e) => update("name", e.target.value)} aria-invalid={Boolean(errors.name)} aria-describedby={errors.name ? "experiment-name-error" : undefined} />
            {errors.name && <p id="experiment-name-error" className="inline-error">{errors.name}</p>}
          </div>
          <div className="policy-field">
            <label htmlFor="policy-source">Policy source</label>
            <select id="policy-source" value={draft.source} onChange={(e) => update("source", e.target.value)} aria-describedby="policy-source-help">
              <option value="OpenVLA">OpenVLA · existing controller</option>
              <option value="custom">Custom policy endpoint · draft</option>
            </select>
            <p id="policy-source-help" className="judge-muted">Custom policies need runtime integration before execution.</p>
          </div>
          {draft.source === "custom" && <div className="policy-field policy-field-wide">
            <label htmlFor="policy-endpoint">Policy endpoint</label>
            <input id="policy-endpoint" type="url" required placeholder="https://your-policy.example/predict" value={draft.endpoint} onChange={(e) => update("endpoint", e.target.value)} aria-invalid={Boolean(errors.endpoint)} aria-describedby="policy-endpoint-help policy-endpoint-error" />
            <p id="policy-endpoint-help" className="judge-muted">Address only. No API keys. This local check does not contact the endpoint.</p>
            <p id="policy-endpoint-error" className="inline-error">{errors.endpoint}</p>
          </div>}
          <div className="policy-field">
            <label htmlFor="policy-task">Task</label>
            <select id="policy-task" value={draft.task} onChange={(e) => update("task", e.target.value)}><option value="close_drawer">Close the drawer</option></select>
          </div>
          <div className="policy-field">
            <label htmlFor="policy-scene">Starting scene</label>
            <select id="policy-scene" defaultValue="drawer"><option value="drawer">Saved drawer scene · Bridge</option></select>
          </div>
          <div className="policy-field">
            <label htmlFor="policy-steps">Rollout length · actions</label>
            <input id="policy-steps" type="number" min={1} max={70} step={1} required value={draft.steps} onChange={(e) => update("steps", e.target.value)} aria-invalid={Boolean(errors.steps)} aria-describedby={errors.steps ? "policy-steps-error" : undefined} />
            {errors.steps && <p id="policy-steps-error" className="inline-error">{errors.steps}</p>}
          </div>
        </div>
        <div className="policy-actions">
          <button type="submit" className="button button-primary">Verify configuration</button>
          <button type="button" className="button button-secondary" disabled aria-describedby="policy-offline">Run evaluation</button>
        </div>
        <p role="status" className="judge-muted text-pretty">{notice}</p>
        {Object.keys(errors).length > 0 && <p role="alert" className="inline-error">Check the highlighted configuration fields.</p>}
      </form>}

      <div className="policy-pipeline">
        <div className="policy-pipeline-caption"><span>{verified ? "Ready for compute" : "Evaluation pipeline"}</span><span id="policy-offline" className="judge-muted">Compute offline · setup available</span></div>
        <div className="policy-progress" role="progressbar" aria-label="Evaluation stages completed" aria-valuemin={0} aria-valuemax={5} aria-valuenow={verified ? 1 : 0} aria-valuetext={verified ? "Local configuration checked. Generation and assessment have not started." : "No stages completed. Configure an experiment to begin."}>
          <span style={{ transform: `scaleX(${verified ? 0.2 : 0})` }} />
        </div>
        <ol className="policy-stages">
          {stages.map((stage, index) => (
            <li key={stage.title}>
              <button type="button" className={cn("policy-stage", inspected === index && "policy-stage-selected", verified && index === 0 && "policy-stage-complete")} onClick={() => setInspected(index)} aria-pressed={inspected === index} aria-controls="policy-stage-detail">
                <span className="policy-stage-number tabular-nums" aria-hidden="true">{verified && index === 0 ? "✓" : `0${index + 1}`}</span>
                <span>{stage.title}<small>{index === 0 ? (verified ? "Checked locally" : "Needs configuration") : (verified && index === 1 ? "Waiting for compute" : "Not started")}</small></span>
              </button>
            </li>
          ))}
        </ol>
        <p id="policy-stage-detail" className="judge-muted text-pretty" aria-live="polite">{stages[inspected].detail}</p>
      </div>
    </section>
  );
}
