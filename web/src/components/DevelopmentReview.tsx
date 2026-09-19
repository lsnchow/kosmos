import { useEffect, useMemo, useRef, useState } from "react";
import { EmptyState, Note, Panel, StatusPill } from "./Primitives";

const REVIEW_ROOT = "/api/development-review";
const REVIEW_SESSION_STORAGE_KEY = "plumb:development-review:session-v1";

const INTEGRITY_VALUES = ["intact", "artifact", "uncertain"] as const;
const COLLISION_VALUES = ["none_visible", "visible", "uncertain"] as const;
const COMPLETION_VALUES = ["met", "not_met", "uncertain"] as const;

type ReviewerKind = "human_self_reported" | "model_assisted";
type Integrity = (typeof INTEGRITY_VALUES)[number];
type Collision = (typeof COLLISION_VALUES)[number];
type CompletionEvidence = (typeof COMPLETION_VALUES)[number];

type ReviewSet = {
  set_id: string;
  title?: string;
  clip_count?: number;
  status?: "development_review_only";
};

type Frame = { index: number; url: string; timestamp?: number | string };

type RatingDraft = {
  integrity?: Integrity | null;
  collision?: Collision | null;
  progress?: number | null;
  completion_evidence?: CompletionEvidence | null;
  evidence_frame_indices?: number[];
  observable_reason?: string;
};

type ReviewClip = {
  opaque_clip_id: string;
  task: {
    id?: string;
    instruction?: string;
    rubric?: string;
    definitions?: {
      integrity?: Partial<Record<Integrity, string>>;
      collision?: Partial<Record<Collision, string>>;
      completion?: string;
    };
    progress_definitions?: string[];
  };
  media: { video_url?: string; frames?: Frame[] };
  draft?: RatingDraft;
};

type ClipResponse = {
  set_id: string;
  clips: ReviewClip[];
  progress?: { saved_drafts?: number; completed?: number; total?: number };
};

type SessionResponse = { session_id: string; reviewer_id: string; reviewer_kind: ReviewerKind };
type StoredReviewSession = SessionResponse & { set_id: string };

type DraftForm = {
  integrity: "" | Integrity;
  collision: "" | Collision;
  progress: "" | `${number}`;
  completion_evidence: "" | CompletionEvidence;
  evidence_frame_indices: number[];
  observable_reason: string;
};

function emptyDraft(): DraftForm {
  return {
    integrity: "",
    collision: "",
    progress: "",
    completion_evidence: "",
    evidence_frame_indices: [],
    observable_reason: "",
  };
}

function toDraftForm(value?: RatingDraft): DraftForm {
  return {
    integrity: value?.integrity ?? "",
    collision: value?.collision ?? "",
    progress: value?.progress === undefined || value.progress === null ? "" : String(value.progress) as `${number}`,
    completion_evidence: value?.completion_evidence ?? "",
    evidence_frame_indices: value?.evidence_frame_indices ?? [],
    observable_reason: value?.observable_reason ?? "",
  };
}

function requestPayload(draft: DraftForm): RatingDraft {
  return {
    // Explicit null clears stale enum/progress values on the partial-draft API.
    integrity: draft.integrity || null,
    collision: draft.collision || null,
    progress: draft.progress === "" ? null : Number(draft.progress),
    completion_evidence: draft.completion_evidence || null,
    evidence_frame_indices: draft.evidence_frame_indices,
    observable_reason: draft.observable_reason,
  };
}

function errorMessage(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object") {
    const value = payload as Record<string, unknown>;
    if (typeof value.detail === "string") return value.detail;
    if (typeof value.reason === "string") return value.reason;
  }
  return fallback;
}

class ReviewRequestError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = "ReviewRequestError";
  }
}

function readStoredSession(): StoredReviewSession | undefined {
  try {
    const raw = window.sessionStorage.getItem(REVIEW_SESSION_STORAGE_KEY);
    if (!raw) return undefined;
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== "object") return undefined;
    const record = value as Record<string, unknown>;
    if (
      typeof record.session_id !== "string" || !record.session_id ||
      typeof record.set_id !== "string" || !record.set_id ||
      typeof record.reviewer_id !== "string" || !record.reviewer_id ||
      (record.reviewer_kind !== "human_self_reported" && record.reviewer_kind !== "model_assisted")
    ) return undefined;
    return record as StoredReviewSession;
  } catch {
    return undefined;
  }
}

function storeSession(session: StoredReviewSession): void {
  try {
    window.sessionStorage.setItem(REVIEW_SESSION_STORAGE_KEY, JSON.stringify(session));
  } catch {
    // Storage can be disabled; cookie-authenticated review still works now.
  }
}

function clearStoredSession(): void {
  try {
    window.sessionStorage.removeItem(REVIEW_SESSION_STORAGE_KEY);
  } catch {
    // A storage failure must not turn an expired session into a render failure.
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const payload: unknown = await response.json().catch(() => undefined);
  if (!response.ok) throw new ReviewRequestError(response.status, errorMessage(payload, `Request failed (${response.status}).`));
  return payload as T;
}

function reviewSetName(item: ReviewSet): string {
  return item.title?.trim() || item.set_id;
}

function clipName(clip: ReviewClip, position: number): string {
  return `Review clip ${position + 1} of ${clip.opaque_clip_id}`;
}

function timestampText(timestamp: Frame["timestamp"]): string {
  if (timestamp === undefined || timestamp === null) return "Timestamp not reported";
  return typeof timestamp === "number" && Number.isFinite(timestamp)
    ? `${timestamp.toFixed(3)} s`
    : String(timestamp);
}

function rawTimestamp(timestamp: Frame["timestamp"]): string | undefined {
  return timestamp === undefined || timestamp === null ? undefined : String(timestamp);
}

function hasSavedValue(draft: RatingDraft | undefined): boolean {
  if (!draft) return false;
  return Boolean(
    draft.integrity ||
    draft.collision ||
    (draft.progress !== undefined && draft.progress !== null) ||
    draft.completion_evidence ||
    (draft.evidence_frame_indices?.length ?? 0) > 0 ||
    draft.observable_reason?.trim(),
  );
}

function taskDefinitions(task: ReviewClip["task"]): string[] {
  const definitions = task.definitions;
  return [
    ...Object.entries(definitions?.integrity ?? {}).map(([key, value]) => `Integrity — ${key}: ${value}`),
    ...Object.entries(definitions?.collision ?? {}).map(([key, value]) => `Collision — ${key}: ${value}`),
    ...(definitions?.completion ? [`Completion — ${definitions.completion}`] : []),
    ...(task.progress_definitions ?? []),
  ].filter((value) => value.trim().length > 0);
}

function FormError({ id, children }: { id: string; children?: string }) {
  if (!children) return null;
  return <p className="mt-1 text-sm text-red-300" id={id} role="alert">{children}</p>;
}

/**
 * Development-only independent review. The server owns all selection,
 * assignment, cookie authentication, and draft isolation; this client only
 * displays opaque review media and submits the current reviewer's own fields.
 */
export function DevelopmentReview() {
  const [sets, setSets] = useState<ReviewSet[]>([]);
  const [selectedSetId, setSelectedSetId] = useState("");
  const [reviewerId, setReviewerId] = useState("");
  const [reviewerKind, setReviewerKind] = useState<ReviewerKind>("human_self_reported");
  const [setsLoading, setSetsLoading] = useState(true);
  const [setsLoaded, setSetsLoaded] = useState(false);
  const [resuming, setResuming] = useState(false);
  const resumeStarted = useRef(false);
  const [setupError, setSetupError] = useState<string>();
  const [session, setSession] = useState<SessionResponse>();
  const [clips, setClips] = useState<ReviewClip[]>([]);
  const [serverProgress, setServerProgress] = useState<ClipResponse["progress"]>();
  const [savedClipIds, setSavedClipIds] = useState<Set<string>>(() => new Set());
  const [clipIndex, setClipIndex] = useState(0);
  const [drafts, setDrafts] = useState<Record<string, DraftForm>>({});
  const [saveError, setSaveError] = useState<string>();
  const [saveStatus, setSaveStatus] = useState<string>();
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let active = true;
    void requestJson<{ sets?: ReviewSet[] }>(`${REVIEW_ROOT}/sets`)
      .then((payload) => {
        if (!active) return;
        setSets(payload.sets ?? []);
        setSetsLoaded(true);
      })
      .catch((error: unknown) => {
        if (active) setSetupError(error instanceof Error ? error.message : "Review sets could not be loaded.");
      })
      .finally(() => {
        if (active) setSetsLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const currentClip = clips[clipIndex];
  const currentDraft = currentClip ? drafts[currentClip.opaque_clip_id] ?? emptyDraft() : emptyDraft();
  const setupBusy = setsLoading || resuming;
  const savedDraftCount = useMemo(
    () => serverProgress?.saved_drafts ?? savedClipIds.size,
    [savedClipIds, serverProgress?.saved_drafts],
  );
  const total = serverProgress?.total ?? clips.length;

  const setDraft = (next: Partial<DraftForm>) => {
    if (!currentClip) return;
    setDrafts((prior) => ({
      ...prior,
      [currentClip.opaque_clip_id]: { ...currentDraft, ...next },
    }));
    setSaveError(undefined);
    setSaveStatus(undefined);
  };

  const loadClips = async (setId: string, sessionId: string) => {
    const payload = await requestJson<ClipResponse>(
      `${REVIEW_ROOT}/sets/${encodeURIComponent(setId)}/clips?session_id=${encodeURIComponent(sessionId)}`,
    );
    const nextClips = payload.clips ?? [];
    setClips(nextClips);
    setServerProgress(payload.progress);
    setClipIndex(0);
    setDrafts(Object.fromEntries(nextClips.flatMap((clip) => clip.draft ? [[clip.opaque_clip_id, toDraftForm(clip.draft)]] : [])));
    setSavedClipIds(new Set(nextClips.filter((clip) => hasSavedValue(clip.draft)).map((clip) => clip.opaque_clip_id)));
  };

  useEffect(() => {
    if (!setsLoaded || resumeStarted.current) return;
    // A state update here would rerun this effect and invoke its cleanup,
    // marking the active request stale before its deferred HTTP response can
    // establish the session. A ref makes the single-attempt guard durable
    // without becoming an effect dependency.
    resumeStarted.current = true;
    const stored = readStoredSession();
    if (!stored) return;
    let active = true;
    setResuming(true);
    void loadClips(stored.set_id, stored.session_id)
      .then(() => {
        if (!active) return;
        setSelectedSetId(stored.set_id);
        setReviewerId(stored.reviewer_id);
        setReviewerKind(stored.reviewer_kind);
        setSession({
          session_id: stored.session_id,
          reviewer_id: stored.reviewer_id,
          reviewer_kind: stored.reviewer_kind,
        });
      })
      .catch((error: unknown) => {
        if (!active) return;
        if (error instanceof ReviewRequestError && (error.status === 401 || error.status === 403)) {
          clearStoredSession();
          setSetupError("Your saved development review session has expired. Start a new review session to continue.");
          return;
        }
        setSetupError(
          error instanceof Error
            ? `Your saved review session could not be resumed: ${error.message}`
            : "Your saved review session could not be resumed. You can start a new review session.",
        );
      })
      .finally(() => {
        if (active) setResuming(false);
      });
    return () => {
      active = false;
    };
  }, [setsLoaded]);

  const startSession = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSetupError(undefined);
    const trimmedReviewerId = reviewerId.trim();
    if (!selectedSetId) {
      setSetupError("Choose a development review set before starting.");
      return;
    }
    if (!trimmedReviewerId) {
      setSetupError("Enter your reviewer ID before starting.");
      return;
    }
    setSetsLoading(true);
    try {
      const nextSession = await requestJson<SessionResponse>(`${REVIEW_ROOT}/sessions`, {
        method: "POST",
        body: JSON.stringify({ set_id: selectedSetId, reviewer_id: trimmedReviewerId, reviewer_kind: reviewerKind }),
      });
      if (!nextSession.session_id) throw new Error("The review service did not create a session.");
      await loadClips(selectedSetId, nextSession.session_id);
      setSession(nextSession);
      storeSession({
        session_id: nextSession.session_id,
        set_id: selectedSetId,
        reviewer_id: nextSession.reviewer_id,
        reviewer_kind: nextSession.reviewer_kind,
      });
    } catch (error) {
      setSetupError(error instanceof Error ? error.message : "The review session could not be started.");
    } finally {
      setSetsLoading(false);
    }
  };

  const saveDraft = async (advance: boolean) => {
    if (!session || !currentClip || saving) return;
    setSaving(true);
    setSaveError(undefined);
    setSaveStatus(undefined);
    try {
      const saved = await requestJson<{ draft?: RatingDraft }>(
        `${REVIEW_ROOT}/sessions/${encodeURIComponent(session.session_id)}/ratings/${encodeURIComponent(currentClip.opaque_clip_id)}?set_id=${encodeURIComponent(selectedSetId)}`,
        { method: "PUT", body: JSON.stringify(requestPayload(currentDraft)) },
      );
      const savedDraft = saved.draft ?? requestPayload(currentDraft);
      const hasSavedDraft = hasSavedValue(savedDraft);
      const wasSavedDraft = savedClipIds.has(currentClip.opaque_clip_id);
      setDrafts((prior) => ({ ...prior, [currentClip.opaque_clip_id]: toDraftForm(savedDraft) }));
      setClips((prior) => prior.map((clip) => clip.opaque_clip_id === currentClip.opaque_clip_id ? { ...clip, draft: savedDraft } : clip));
      setSavedClipIds((prior) => {
        const next = new Set(prior);
        if (hasSavedDraft) next.add(currentClip.opaque_clip_id);
        else next.delete(currentClip.opaque_clip_id);
        return next;
      });
      setServerProgress((progress) => progress ? {
        ...progress,
        saved_drafts: Math.max(
          0,
          (progress.saved_drafts ?? savedClipIds.size) + (hasSavedDraft && !wasSavedDraft ? 1 : !hasSavedDraft && wasSavedDraft ? -1 : 0),
        ),
      } : progress);
      setSaveStatus("Your draft was saved.");
      if (advance && clipIndex < clips.length - 1) setClipIndex((prior) => prior + 1);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "The draft could not be saved.");
    } finally {
      setSaving(false);
    }
  };

  const navigate = (offset: -1 | 1) => {
    setClipIndex((prior) => Math.max(0, Math.min(clips.length - 1, prior + offset)));
    setSaveError(undefined);
    setSaveStatus(undefined);
  };

  return (
    <main className="min-h-dvh bg-[var(--surface-0)]">
      <div className="console max-w-7xl">
        <a className="button button-secondary mb-6" href="/">Return to evaluation console</a>
        <header className="mb-6 max-w-3xl">
          <p className="eyebrow">Independent development review</p>
          <h1 className="text-balance">Review visible evidence, one opaque clip at a time.</h1>
          <p className="masthead-copy text-pretty">
            This is development-review-only. It is not Gate D calibration, does not expose source or model metadata,
            and does not establish a policy or task result.
          </p>
        </header>

        {!session ? (
          <Panel title="Start a review session" eyebrow="Your own saved drafts only">
            {setupBusy && <p className="text-pretty text-sm text-[var(--text-muted)]" id="sets-loading" role="status">{resuming ? "Resuming your saved development review session…" : "Loading development review sets…"}</p>}
            {!setupBusy && sets.length === 0 && !setupError && (
              <EmptyState>No development review sets are available yet. Return when the coordinator has assigned one.</EmptyState>
            )}
            {sets.length > 0 && (
              <form className="grid max-w-2xl gap-4" onSubmit={(event) => void startSession(event)} noValidate>
                <div>
                  <label className="select-label" htmlFor="review-set">Development review set</label>
                  <select id="review-set" value={selectedSetId} onChange={(event) => setSelectedSetId(event.target.value)} aria-invalid={Boolean(setupError && !selectedSetId)} aria-describedby={setupError && !selectedSetId ? "session-error" : undefined}>
                    <option value="">Choose a set</option>
                    {sets.map((item) => <option key={item.set_id} value={item.set_id}>{reviewSetName(item)}{item.clip_count === undefined ? "" : ` (${item.clip_count} clips)`}</option>)}
                  </select>
                </div>
                <div>
                  <label className="select-label" htmlFor="reviewer-id">Reviewer ID</label>
                  <input id="reviewer-id" className="w-full rounded border border-slate-600 bg-slate-900 p-2 text-sm text-slate-100" value={reviewerId} onChange={(event) => setReviewerId(event.target.value)} autoComplete="username" required aria-invalid={Boolean(setupError && !reviewerId.trim())} aria-describedby={setupError && !reviewerId.trim() ? "session-error" : undefined} />
                </div>
                <div>
                  <label className="select-label" htmlFor="reviewer-kind">Reviewer kind</label>
                  <select id="reviewer-kind" value={reviewerKind} onChange={(event) => setReviewerKind(event.target.value as ReviewerKind)}>
                    <option value="human_self_reported">Human — self-reported</option>
                    <option value="model_assisted">Model-assisted review</option>
                  </select>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <button className="button button-primary" type="submit" disabled={setupBusy} aria-describedby={setupBusy ? "sets-loading" : undefined}>Start review</button>
                  <p className="text-pretty text-sm text-[var(--text-muted)]">The service sets an HttpOnly local session cookie. This page cannot read it.</p>
                </div>
              </form>
            )}
            <FormError id="session-error">{setupError}</FormError>
            <Note>No placeholder reviewer, rating, or calibration status is created when no session has been started.</Note>
          </Panel>
        ) : clips.length === 0 ? (
          <Panel title="Review session" eyebrow="Development-only">
            <EmptyState>Your session has no assigned opaque clips. No rating form is shown.</EmptyState>
          </Panel>
        ) : currentClip ? (
          <section className="grid gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(22rem,0.8fr)]">
            <Panel title={`Clip ${clipIndex + 1} of ${clips.length}`} eyebrow="Development review only" action={<StatusPill status="pending">saved drafts {savedDraftCount} / {total}</StatusPill>}>
              <p className="mb-4 break-all text-sm text-[var(--text-muted)]">{clipName(currentClip, clipIndex)}</p>
              <section aria-labelledby="review-task-heading" className="mb-5">
                <h3 className="mb-2 text-balance text-lg font-semibold text-[var(--text-strong)]" id="review-task-heading">Task and rubric</h3>
                <p className="text-pretty">{currentClip.task.instruction ?? "Task instruction was not reported."}</p>
                <p className="mt-2 text-pretty text-sm text-[var(--text-muted)]">{currentClip.task.rubric ?? "Canonical task rubric was not reported."}</p>
                {taskDefinitions(currentClip.task).length > 0 ? (
                  <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-[var(--text-muted)]">
                    {taskDefinitions(currentClip.task).map((definition, index) => <li key={`${definition}-${index}`} className="text-pretty">{definition}</li>)}
                  </ul>
                ) : <p className="mt-3 text-sm text-[var(--text-muted)]">Canonical definitions were not reported.</p>}
              </section>
              {currentClip.media.video_url ? <video className="max-h-96 w-full rounded border border-slate-700 bg-slate-950 object-contain" controls playsInline preload="metadata" src={currentClip.media.video_url} poster={currentClip.media.frames?.[0]?.url} aria-label={`Review video for clip ${clipIndex + 1}`} /> : <EmptyState>No review video was reported for this opaque clip.</EmptyState>}
              <section className="mt-5" aria-labelledby="frame-heading">
                <h3 className="mb-3 text-balance text-lg font-semibold text-[var(--text-strong)]" id="frame-heading">Indexed evidence frames</h3>
                {(currentClip.media.frames?.length ?? 0) === 0 ? <EmptyState>No indexed PNG frames were reported for this opaque clip.</EmptyState> : (
                  <ol className="grid grid-cols-2 gap-3 sm:grid-cols-4" aria-label="Indexed evidence frames">
                    {currentClip.media.frames?.map((frame) => <li key={frame.index} className="rounded border border-slate-700 bg-slate-900 p-2">
                      <img className="aspect-square w-full object-contain" src={frame.url} alt={`Evidence frame ${frame.index} at ${timestampText(frame.timestamp)}`} />
                      <p className="mt-2 text-sm font-semibold tabular-nums">Frame {frame.index}</p>
                      <p className="break-words text-xs text-[var(--text-muted)]" title={rawTimestamp(frame.timestamp)}>{timestampText(frame.timestamp)}</p>
                    </li>)}
                  </ol>
                )}
                {(currentClip.media.frames?.length ?? 0) !== 16 && <p className="mt-3 text-sm text-amber-200" role="status">The service reported {currentClip.media.frames?.length ?? 0} indexed frames, not 16. No missing frames are invented.</p>}
              </section>
            </Panel>

            <Panel title="Your draft" eyebrow="Only visible evidence" action={<StatusPill status="pending">{saving ? "saving" : "unsent or saved"}</StatusPill>}>
              <form onSubmit={(event) => { event.preventDefault(); void saveDraft(false); }} className="grid gap-4" noValidate>
                <fieldset className="grid gap-4">
                  <legend className="mb-2 text-balance font-semibold">Observed labels</legend>
                  <div><label className="select-label" htmlFor="integrity">Integrity</label><select id="integrity" value={currentDraft.integrity} onChange={(event) => setDraft({ integrity: event.target.value as DraftForm["integrity"] })} aria-invalid={Boolean(saveError)} aria-describedby={saveError ? "save-error" : undefined}><option value="">Not observed</option>{INTEGRITY_VALUES.map((value) => <option key={value} value={value}>{value.replace("_", " ")}</option>)}</select></div>
                  <div><label className="select-label" htmlFor="collision">Collision</label><select id="collision" value={currentDraft.collision} onChange={(event) => setDraft({ collision: event.target.value as DraftForm["collision"] })} aria-invalid={Boolean(saveError)} aria-describedby={saveError ? "save-error" : undefined}><option value="">Not observed</option>{COLLISION_VALUES.map((value) => <option key={value} value={value}>{value.replace("_", " ")}</option>)}</select></div>
                  <div><label className="select-label" htmlFor="progress">Progress</label><select id="progress" value={currentDraft.progress} onChange={(event) => setDraft({ progress: event.target.value as DraftForm["progress"] })} aria-invalid={Boolean(saveError)} aria-describedby={saveError ? "save-error" : undefined}><option value="">Not observed</option>{[0, 1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value}</option>)}</select></div>
                  <div><label className="select-label" htmlFor="completion">Completion evidence</label><select id="completion" value={currentDraft.completion_evidence} onChange={(event) => setDraft({ completion_evidence: event.target.value as DraftForm["completion_evidence"] })} aria-invalid={Boolean(saveError)} aria-describedby={saveError ? "save-error" : undefined}><option value="">Not observed</option>{COMPLETION_VALUES.map((value) => <option key={value} value={value}>{value.replace("_", " ")}</option>)}</select></div>
                </fieldset>
                <fieldset>
                  <legend className="mb-2 text-balance font-semibold">Evidence frame indices</legend>
                  <p className="mb-3 text-pretty text-sm text-[var(--text-muted)]">Select only frames supporting this draft. Leave all unchecked when no visible frame supports it.</p>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {Array.from({ length: 16 }, (_, index) => {
                      const frame = currentClip.media.frames?.find((item) => item.index === index);
                      const checked = currentDraft.evidence_frame_indices.includes(index);
                      return <label key={index} className="flex items-center gap-2 rounded border border-slate-700 p-2 text-sm"><input type="checkbox" checked={checked} onChange={() => setDraft({ evidence_frame_indices: checked ? currentDraft.evidence_frame_indices.filter((value) => value !== index) : [...currentDraft.evidence_frame_indices, index].sort((a, b) => a - b) })} aria-invalid={Boolean(saveError)} aria-describedby={saveError ? "save-error" : undefined} />Frame {index}{frame?.timestamp !== undefined && frame?.timestamp !== null ? <span className="sr-only">, {frame.timestamp}</span> : null}</label>;
                    })}
                  </div>
                </fieldset>
                <div><label className="select-label" htmlFor="observable-reason">Observable reason</label><textarea id="observable-reason" className="min-h-28 w-full rounded border border-slate-600 bg-slate-900 p-2 text-sm text-slate-100" value={currentDraft.observable_reason} onChange={(event) => setDraft({ observable_reason: event.target.value })} aria-invalid={Boolean(saveError)} aria-describedby={saveError ? "reason-help save-error" : "reason-help"} /><p className="mt-1 text-sm text-[var(--text-muted)]" id="reason-help">Describe only what is visible. Do not infer source, model, cohort, or outcome metadata.</p></div>
                <FormError id="save-error">{saveError}</FormError>
                {saveStatus && <p className="text-sm text-emerald-200" role="status">{saveStatus}</p>}
                <div className="flex flex-wrap gap-3"><button className="button button-quiet" type="button" disabled={clipIndex === 0} aria-describedby={clipIndex === 0 ? "previous-help" : undefined} onClick={() => navigate(-1)}>Previous</button><button className="button button-quiet" type="button" disabled={clipIndex === clips.length - 1} aria-describedby={clipIndex === clips.length - 1 ? "next-help" : undefined} onClick={() => navigate(1)}>Next</button><button className="button button-secondary" type="submit" disabled={saving}>{saving ? "Saving…" : "Save draft"}</button><button className="button button-primary" type="button" disabled={saving || clipIndex === clips.length - 1} aria-describedby={clipIndex === clips.length - 1 ? "next-help" : undefined} onClick={() => void saveDraft(true)}>Save and next</button>{clipIndex === 0 && <p className="self-center text-sm text-[var(--text-muted)]" id="previous-help">This is the first assigned clip.</p>}{clipIndex === clips.length - 1 && <p className="self-center text-sm text-[var(--text-muted)]" id="next-help">This is the final assigned clip.</p>}</div>
              </form>
              <Note>Nothing on this page is a calibration decision or Gate D evidence. The server returns only this reviewer's own drafts.</Note>
            </Panel>
          </section>
        ) : null}
      </div>
    </main>
  );
}
