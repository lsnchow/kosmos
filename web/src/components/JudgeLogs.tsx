import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";
import { isRecord, type DemoJudgment, type JsonRecord } from "../lib/api";

export function JudgeLogs({ judgment }: { judgment: DemoJudgment }) {
  const [open, setOpen] = useState(false);
  const [events, setEvents] = useState<JsonRecord[]>([]);
  const [error, setError] = useState<string>();
  useEffect(() => {
    if (!open) return;
    let mounted = true;
    const read = async () => {
      try {
        const r = await fetch(`/api/demo/judgments/${encodeURIComponent(judgment.id)}/events`);
        if (!r.ok) throw new Error("Could not read persisted logs.");
        const payload = await r.json();
        if (mounted) { setEvents(Array.isArray(payload.events) ? payload.events.filter(isRecord) : []); setError(undefined); }
      } catch (e) { if (mounted) setError(e instanceof Error ? e.message : "Log read failed."); }
    };
    void read();
    const timer = ["queued", "warming", "assessing"].includes(judgment.status) ? window.setInterval(() => void read(), 1000) : undefined;
    return () => { mounted = false; window.clearInterval(timer); };
  }, [open, judgment.id, judgment.status]);
  const samples = judgment.result?.report?.raw_judge_samples;
  return <Dialog.Root open={open} onOpenChange={setOpen}>
    <Dialog.Trigger asChild><button type="button" className="button button-secondary">Pop out logs</button></Dialog.Trigger>
    <Dialog.Portal>
      <Dialog.Overlay className="dialog-overlay" />
      <Dialog.Content className="freeplay-dialog judge-log-dialog">
        <header className="dialog-header"><div><Dialog.Title>Judge logs</Dialog.Title><Dialog.Description>Persisted events and raw model responses for this request.</Dialog.Description></div><Dialog.Close asChild><button className="button button-secondary">Close logs</button></Dialog.Close></header>
        <div className="judge-log-body">
          <p className="tabular-nums">{judgment.id} · {judgment.status}</p>
          {error && <p role="alert" className="inline-error">{error}</p>}
          <h3 className="text-balance">Execution events</h3>
          <pre className="inference-value">{events.map((event) => `${event.created_at}  ${event.stage}  ${event.status}\n${JSON.stringify(event.detail)}`).join("\n\n") || "Waiting for persisted events…"}</pre>
          {judgment.error && <><h3 className="text-balance">Request error</h3><pre className="inference-value">{JSON.stringify(judgment.error, null, 2)}</pre></>}
          {Array.isArray(samples) && samples.filter(isRecord).map((sample, index) => <section key={index}>
            <h3 className="text-balance">Sample {index + 1}</h3>
            {Array.isArray(sample.attempts) && sample.attempts.filter(isRecord).map((attempt, attemptIndex) => <div key={attemptIndex}>
              <p className="tabular-nums">Attempt {attemptIndex + 1} · {typeof attempt.wall_seconds === "number" ? attempt.wall_seconds.toFixed(3) + " s" : ""}</p>
              <pre className="inference-value">{typeof attempt.raw_output === "string" ? attempt.raw_output : "No output returned"}</pre>
              {typeof attempt.failure_reason === "string" && <p className="inline-error">{attempt.failure_reason}</p>}
            </div>)}
          </section>)}
        </div>
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>;
}
