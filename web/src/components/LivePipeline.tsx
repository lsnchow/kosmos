import { useEffect, useState } from "react";
import { api, type DemoSession, type DemoStatus } from "../lib/api";
import { cn } from "../lib/utils";
import { DemoJudgePanel } from "./DemoJudgePanel";
import { demoHeuristic } from "../lib/demoHeuristic";
import { TextConditioningChat } from "./TextConditioningChat";

const active = (state?: string) => ["queued", "initializing", "running"].includes(state ?? "");
const terminal = (state?: string) => ["completed", "abstained", "failed", "interrupted"].includes(state ?? "");
const seconds = (number: unknown) => typeof number === "number" ? `${number.toFixed(2)} s` : undefined;
const behaviors = [
  { id: "openvla", label: "OpenVLA" },
  { id: "pi0", label: "MiniVLA" },
  { id: "octo", label: "Octo" },
  { id: "baseline", label: "Supplied-action demo" },
];

export function LivePipeline() {
  // A refresh deliberately starts with a blank experiment, never a previous run.
  const [session, setSession] = useState<DemoSession>();
  const [status, setStatus] = useState<DemoStatus>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [connected, setConnected] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [scene, setScene] = useState<"pot" | "drawer">("pot");
  const [seed, setSeed] = useState("0");
  const [length, setLength] = useState(16);
  const [behaviorId, setBehaviorId] = useState<"openvla" | "pi0" | "octo" | "baseline">("openvla");
  const behavior = behaviors.find((entry) => entry.id === behaviorId) ?? behaviors[0];
  const pot = scene === "pot";
  const defaultTask = pot ? "Put the pot to the left of the purple item." : "Close the drawer";
  const [task, setTask] = useState(defaultTask);
  const conditioningPrompt = task.trim();
  const running = active(session?.state);
  const awaitingJudge = Boolean(session?.auto_assess && !["blocked", "error"].includes(session.state) && !session.assessment_error && !terminal(session.judgment?.status));
  const complete = session?.state === "completed";
  const judgment = session?.judgment;
  const done = (complete && !session?.auto_assess) || terminal(judgment?.status) || Boolean(session?.assessment_error) || ["blocked", "error"].includes(session?.state ?? "");
  const judging = complete && awaitingJudge;

  useEffect(() => {
    let mounted = true;
    window.history.replaceState(null, "", "/console");
    try { localStorage.removeItem("kosmos-current-live-rollout"); } catch { /* optional storage */ }
    const check = () => api.demoStatus().then((value) => { if (mounted) setStatus(value); }).catch(() => undefined);
    void check();
    const timer = window.setInterval(() => void check(), 5000);
    return () => { mounted = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    if (!session || (!running && !awaitingJudge)) return;
    const stream = new EventSource(`/api/demo/sessions/${encodeURIComponent(session.id)}/events`);
    stream.onopen = () => setConnected(true);
    stream.addEventListener("snapshot", (event) => {
      try { setSession(JSON.parse((event as MessageEvent).data)); setConnected(true); }
      catch { setError("Could not read generation progress."); }
    });
    stream.onerror = () => setConnected(false);
    const timer = window.setInterval(() => setNow(Date.now()), 250);
    return () => { stream.close(); window.clearInterval(timer); };
  }, [session?.id, running, awaitingJudge]);

  const start = async () => {
    if (busy || running || awaitingJudge) return;
    if (!task.trim() || conditioningPrompt.length > 1000) { setError("Enter an instruction of up to 1,000 characters."); return; }
    if (!/^\d+$/.test(seed) || Number(seed) > 2147483647) { setError("Use an integer seed from 0 to 2147483647."); return; }
    setBusy(true); setError(undefined);
    try {
      const next = await api.createDemoSession({ title: `${behavior.label} · seed ${seed}`, prompt: conditioningPrompt, demo_policy: behaviorId, mode: "policy", steps: length, starting_scene: scene, world_model: "cosmos", auto_assess: true, seed: Number(seed) });
      setSession(next);
      window.dispatchEvent(new Event("demo-sessions-changed"));
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not start Cosmos generation."); }
    finally { setBusy(false); }
  };

  const frameCount = session?.completed_steps ?? 0;
  const progress = session?.world_progress;
  const diffusionTotal = ((session?.requested_steps ?? length) / 16) * 30;
  const generationSteps = Math.min(diffusionTotal, progress?.step ?? (complete ? diffusionTotal : 0));
  const judgeComplete = ["completed", "abstained"].includes(judgment?.status ?? "");
  const judgedSamples = judgeComplete ? 5 : judgment?.progress?.completed_samples ?? 0;
  const totalStages = diffusionTotal + (session?.auto_assess ? 5 : 0);
  const latest = session?.frames?.at(-1);
  const end = judgment?.finished_at ? Date.parse(judgment.finished_at) : done && session?.updated_at ? session.updated_at * 1000 : now;
  const elapsed = session?.created_at ? Math.max(0, (end - session.created_at * 1000) / 1000) : 0;
  const timing = judgment?.result?.timing;
  const worldTime = session?.frames?.flatMap((frame) => typeof frame.timings_ms?.world === "number" ? [frame.timings_ms.world / 1000] : []);
  const summary = [
    ["Total run wall time", seconds(elapsed)],
    ["Generation wall time", session?.generation_completed_at && session.created_at ? seconds(session.generation_completed_at - session.created_at) : undefined],
    ["Cosmos inference", worldTime?.length ? seconds(worldTime.reduce((a,b) => a+b,0)) : undefined],
    ["Peak Cosmos GPU memory", typeof session?.world_timing?.peak_gpu_bytes === "number" ? `${(session.world_timing.peak_gpu_bytes / 2 ** 30).toFixed(2)} GiB` : undefined],
    ["VLM inference", seconds(timing?.worker_inference_seconds)],
    ["VLM remote latency", seconds(timing?.remote_request_seconds)],
    ["VLM cold load", seconds(timing?.worker_load_seconds)],
    ["Peak VLM GPU memory", typeof timing?.gpu_peak_memory_bytes === "number" ? `${(timing.gpu_peak_memory_bytes / 2 ** 30).toFixed(2)} GiB` : undefined],
  ].filter(([,value]) => value !== undefined);

  return <section className="live-pipeline" aria-labelledby="live-pipeline-title">
    <div className="policy-experiment-header">
      <div><p className="eyebrow">{running ? "LIVE COSMOS GENERATION" : judging ? "JUDGING GENERATED VIDEO" : done ? "RUN FINISHED" : "New Cosmos experiment"}</p>
        <h2 id="live-pipeline-title" className="text-balance">Generate with Cosmos</h2>
        <p className="judge-muted text-pretty">{length} actions · {diffusionTotal} diffusion steps</p>
      </div>
      <div className="header-actions"><TextConditioningChat instruction={task} defaultInstruction={defaultTask} scene={scene} disabled={busy || running || awaitingJudge} onInstructionChange={(value) => { setTask(value); setSession(undefined); }} /><button className="button button-primary" onClick={() => void start()} disabled={busy || running || awaitingJudge || !status?.available || status.world_model !== "cosmos"}>Generate</button></div>
    </div>
    <details className="judge-details" open={!session}><summary>Run setup</summary><div className="policy-fields">
      <div className="policy-field"><label htmlFor="cosmos-scenario">Scenario</label><select id="cosmos-scenario" value={scene} disabled={busy || running || awaitingJudge} onChange={(event) => { const nextScene = event.target.value as "pot" | "drawer"; setScene(nextScene); setTask(nextScene === "pot" ? "Put the pot to the left of the purple item." : "Close the drawer"); setLength(16); setSession(undefined); }}>
        <option value="pot">Move bowl / pot · recorded trajectory</option>
        <option value="drawer">Drawer · manual action probe</option>
      </select></div>
      <div className="policy-field"><select id="cosmos-behavior" aria-label="Generation profile" value={behaviorId} disabled={busy || running || awaitingJudge} onChange={(event) => { setBehaviorId(event.target.value as typeof behaviorId); setSession(undefined); setError(undefined); }}>{behaviors.map((entry) => <option key={entry.id} value={entry.id}>{entry.label}</option>)}</select></div>
      <div className="policy-field policy-field-wide"><label htmlFor="cosmos-main-prompt">Main prompt</label><textarea id="cosmos-main-prompt" value={task} disabled={busy || running || awaitingJudge} maxLength={1000} rows={3} onChange={(event) => { setTask(event.target.value); setSession(undefined); }} /></div>
      <div className="policy-field"><label htmlFor="cosmos-seed">Variation seed</label><input id="cosmos-seed" type="number" min={0} max={2147483647} step={1} value={seed} disabled={busy || running || awaitingJudge} onChange={(event) => { setSeed(event.target.value); setSession(undefined); }} /><p className="judge-muted">Seed 0 matches the previous setup. Change it for a new sampled variation.</p></div>
      {pot && <div className="policy-field"><label htmlFor="cosmos-length">Rollout length</label><select id="cosmos-length" value={length} disabled={busy || running || awaitingJudge} onChange={(event) => { setLength(Number(event.target.value)); setSession(undefined); }}><option value={32}>6.6 seconds · two generated chunks</option><option value={16}>3.4 seconds · one generated chunk</option></select><p className="judge-muted">The longer mode replays the action plan from the first chunk’s final predicted frame.</p></div>}
    </div></details>
    <ol className="pipeline-chain" aria-label="Actions to Cosmos to judge">
      <li className="pipeline-node"><h3 className="text-balance">{behavior.label}</h3><p>{task}</p></li>
      <li className={cn("pipeline-node", running && "pipeline-active")}><span className="eyebrow">02 · WORLD MODEL</span><h3 className="text-balance">Cosmos3-Nano</h3><p>Scene + text + actions → new video</p><strong>{running ? `${progress?.stage ?? "Preparing"} · ${generationSteps}/${diffusionTotal}` : complete ? "New video saved" : "Ready for a fresh generation"}</strong></li>
      <li className={cn("pipeline-node", judging && "pipeline-active")}><span className="eyebrow">03 · VLM JUDGE</span><h3 className="text-balance">Qwen + semantic LoRA</h3><p>Completed video → assessment</p><strong>{judging ? `Assessing · ${judgedSamples}/5 samples complete` : judgment ? `Assessment ${judgment.status}` : "Starts after the video is saved"}</strong></li>
    </ol>
    {error && <p role="alert" className="inline-error">{error}</p>}
    {(session?.error || session?.assessment_error) && <p role="alert" className="inline-error">{session.error ?? session.assessment_error}</p>}
    {session && <div className="generation-progress">
      <div className="policy-pipeline-caption"><span className="tabular-nums">{judging ? "VLM assessment" : done ? "Run finished" : progress?.stage ?? "Preparing Cosmos"} · {elapsed.toFixed(1)} s</span><span>{running || awaitingJudge ? connected ? "Live updates connected" : "Reconnecting…" : "Saved to past runs"}</span></div>
      <div className="policy-progress" role="progressbar" aria-label="Generation and assessment progress" aria-valuemin={0} aria-valuemax={totalStages} aria-valuenow={generationSteps + judgedSamples} aria-valuetext={`${generationSteps} of ${diffusionTotal} diffusion steps; ${judgedSamples} of 5 judge samples`}><span style={{transform:`scaleX(${(generationSteps + judgedSamples)/totalStages})`}} /></div>
      <p className="judge-muted">Progress follows actual diffusion steps. Video frames arrive after decoding.</p>
    </div>}
    {!complete && <div className="live-frame-stage">{latest?.url ? <img src={latest.url} alt={frameCount ? `New Cosmos frame ${frameCount}` : "Starting scene"} /> : <div className="live-frame-empty">Press Generate to create a new Cosmos video.</div>}<p>{frameCount ? `${frameCount} newly generated frames received` : "Starting scene only · no generated video yet"}</p></div>}
    {done && <section className="run-summary" aria-label="Run summary"><h3 className="text-balance">Run summary</h3><p className="text-pretty">{judgment?.error?.message ?? session?.assessment_error ?? session?.error ?? (demoHeuristic(judgment) ? `${demoHeuristic(judgment)?.label} · Completion-vote estimate` : judgment?.result?.assessment?.completion) ?? "Generation finished"}</p><dl className="performance-grid tabular-nums">{summary.map(([label,value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl></section>}
    {complete && session?.latest_video_url && (session.auto_assess ? <DemoJudgePanel automatic providedJudgment={judgment} clipOverride={{id:`live-demo:${session.id}`,title:session.title,task_id:pot ? "demo_pot_left_v1" : "close_drawer",task_label:session.prompt,video_url:session.latest_video_url,action_source:"Supplied action trajectory",controller_identity:"Cosmos3-Nano world model",scene_reference_role:"Initial scene"}} /> : <section className="judge-panel" aria-label="Custom rollout and assessment"><div className="judge-layout"><div className="custom-rollout-media"><video className="judge-video" src={session.latest_video_url} controls playsInline preload="metadata" aria-label="Generated custom-condition rollout" /></div><div><p className="eyebrow">VLM assessment</p><h3 className="text-balance">Not assessed</h3><p className="judge-muted text-pretty">Custom instructions and simulated behaviors do not have an automatic scoring rubric.</p><details className="judge-details"><summary>Run instruction</summary><p className="text-pretty">{session.prompt}</p></details></div></div></section>)}
  </section>;
}
