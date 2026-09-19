export type JsonRecord = Record<string, unknown>;

export type Run = JsonRecord & {
  id?: string;
  status?: string;
  mode?: string;
  created_at?: string;
  completed?: number;
  failed?: number;
  total?: number;
  evaluable?: number;
  successes?: number;
};

export type Episode = JsonRecord & {
  id?: string;
  episode_id?: string;
  policy?: string;
  task?: string;
  status?: string;
  frame_url?: string;
  video_url?: string;
  artifact_path?: string;
};

export type AnalysisCell = JsonRecord & {
  policy?: string;
  task?: string;
  n?: number;
  valid?: number;
  successes?: number;
  coverage?: number;
  rate?: number | null;
  positive_rate?: number | null;
  wilson?: unknown;
  missing_bounds?: unknown;
  reference_rate?: number | null;
};

export type Gate = JsonRecord & {
  id?: string;
  name?: string;
  status?: string;
  reason?: string;
  summary?: string;
};

export type SweepPoint = JsonRecord & {
  id?: string;
  label?: string;
  estimated_usd?: number | null;
  coverage?: number | null;
  horizon?: number | string | null;
  qualified?: boolean;
};

export type Experiment = JsonRecord & {
  id?: string;
  kind?: string;
  status?: string;
  model?: string;
  frame_count?: number;
  action_dimensions?: number;
  latency_seconds?: number;
  model_load_seconds?: number;
  gpu_peak_memory_bytes?: number;
  qualification?: string;
  report_url?: string;
  video_url?: string | null;
  stage?: string;
  ticks_completed?: number;
  ticks_requested?: number;
  total_seconds?: number;
  outcome?: string | null;
  notes?: string | string[];
  timing_scope?: string;
  models?: string[] | string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as JsonRecord;
      if (typeof payload.detail === "string") message = payload.detail;
    } catch {
      // An empty or non-JSON response is still reported using its HTTP status.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<JsonRecord>("/api/health"),
  protocol: () => request<JsonRecord>("/api/protocol"),
  gates: () => request<{ gates?: Gate[] }>("/api/gates"),
  runs: () => request<{ runs?: Run[] }>("/api/runs"),
  run: (runId: string) => request<Run>(`/api/runs/${encodeURIComponent(runId)}`),
  episodes: (runId: string) =>
    request<{ episodes?: Episode[] }>(`/api/runs/${encodeURIComponent(runId)}/episodes`),
  analysis: (runId: string) =>
    request<{ cells?: AnalysisCell[]; summary?: JsonRecord }>(
      `/api/runs/${encodeURIComponent(runId)}/analysis`,
    ),
  sweeps: () => request<{ points?: SweepPoint[]; status?: string }>("/api/sweeps"),
  experiments: () => request<{ experiments?: Experiment[] }>("/api/experiments"),
  createSyntheticRun: (body: {
    mode: "synthetic";
    backend: "synthetic";
    policies: string[];
    tasks: string[];
    starts_per_task: 50;
    seed: number;
    idempotency_key: string;
  }) =>
    request<Run>("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  cancelRun: (runId: string) =>
    request<Run>(`/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" }),
  freeplayStep: (body: { session_id?: string; action: number[] }) =>
    request<{
      session_id?: string;
      mode?: string;
      frame_url?: string;
      state?: { x?: number; y?: number; z?: number; gripper?: number };
      latency_ms?: number;
      qualified?: boolean;
    }>("/api/freeplay/step", { method: "POST", body: JSON.stringify(body) }),
};

export function isRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function artifactUrl(value: unknown): string | undefined {
  if (typeof value !== "string" || value.length === 0) return undefined;
  if (/^https?:\/\//i.test(value) || value.startsWith("data:") || value.startsWith("/")) return value;
  return `/api/artifacts/${value.split("/").map(encodeURIComponent).join("/")}`;
}
