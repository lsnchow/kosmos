export type JsonRecord = Record<string, unknown>;

/**
 * A run record. The ledger exposes `total` (planned logical episodes),
 * `completed`, `failed`, `cancelled`, `evaluable` and `successes` — and nothing
 * else. There is no `planned`, `submitted`, `running` or `excluded` column, so
 * those counts are read from the analysis ledger state instead of being faked
 * from a subtraction.
 */
export type Run = JsonRecord & {
  id?: string;
  status?: string;
  mode?: string;
  config?: JsonRecord;
  created_at?: string;
  updated_at?: string;
  completed_at?: string | null;
  completed?: number;
  failed?: number;
  cancelled?: number;
  total?: number;
  evaluable?: number;
  successes?: number;
};

/** Provenance of generated media. Never inferred — only read from the server. */
export type Provenance = "live" | "cached" | "replayed" | "qualitative";

export const PROVENANCE_VALUES: Provenance[] = ["live", "cached", "replayed", "qualitative"];

export function readProvenance(value: unknown): Provenance | undefined {
  return PROVENANCE_VALUES.find((candidate) => candidate === value);
}

export type ArtifactRef = {
  uri?: string;
  relative_path?: string;
  artifact_path?: string;
  url?: string;
  sha256?: string;
  media_type?: string;
  label?: string;
};

export type Episode = JsonRecord & {
  /** The instruction the rollout was given; free-text tasks have no registry entry. */
  task_instruction?: string | null;
  /** True for the five frozen benchmark tasks. Absent when the record did not say. */
  benchmark_task?: boolean | null;
  id?: string;
  episode_id?: string;
  run_id?: string;
  logical_key?: string;
  policy?: string;
  policy_variant?: string;
  task?: string;
  status?: string;
  start_id?: string;
  start_lineage_id?: string;
  world_seed?: number;
  binary_success?: boolean | null;
  validity?: "valid" | "invalid" | "unknown" | string;
  progress_score?: number | null;
  missing_reason?: string | null;
  horizon_actions?: number;
  mode?: string;
  artifact_refs?: Record<string, ArtifactRef>;
  frame_url?: string | null;
  video_url?: string | null;
  artifact_path?: string;
  timing?: JsonRecord;
  attempt_count?: number;
  created_at?: string;
  completed_at?: string;
  error?: { type?: string; message?: string } | null;
  // --- Fields required for the wall and the hero track. Absent today. ---
  frame_urls?: string[];
  certified_frame_count?: number;
  provenance?: Provenance;
  resolution?: string;
  /**
   * Explicit presentation selector. `hero_480p` marks the single 480p
   * presentation rollout; the 256p scoring protocol never sets it. Selecting a
   * hero by array position instead silently relabels tile #01 as 480p.
   */
  presentation_track?: string;
};

export type AnalysisCell = JsonRecord & {
  policy?: string;
  policy_base?: string;
  policy_variant?: string | null;
  task?: string;
  n?: number;
  valid?: number;
  successes?: number;
  missing?: number;
  coverage?: number | null;
  rate?: number | null;
  positive_rate?: number | null;
  wilson?: unknown;
  positive_wilson?: unknown;
  missing_bounds?: unknown;
  missingness_rate?: number | null;
  reference_rate?: number | null;
  reference_successes?: number | null;
  reference_n?: number | null;
  reference_wilson?: unknown;
  simpler_rate?: number | null;
  simpler_successes?: number | null;
  service_failures?: number;
  validity_counts?: Record<string, number>;
  missing_reason_counts?: Record<string, number>;
  status_counts?: Record<string, number>;
  ledger_status?: string;
  /** "primary_manifest_bound" | "diagnostic" | "blocked". */
  scientific_status?: string;
  scientific_reason?: string;
  interval_caveat?: string;
};

/**
 * One pairwise ordering verdict, as the measurement module already emits it.
 * `ordering` is either "indeterminate" or "<a>_over_<b>"; `supported` is true
 * only when the declared separation test passed at its family alpha.
 */
export type PairwiseVerdict = JsonRecord & {
  scope?: string;
  policy_a?: string;
  policy_b?: string;
  observed_positive_difference?: number | null;
  status?: string;
  ordering?: string;
  supported?: boolean;
  ordering_reason?: string;
  paired?: JsonRecord & {
    paired_n?: number;
    a_only?: number;
    b_only?: number;
    p_value?: number;
    family_alpha?: number;
    passes_bonferroni_exact_test?: boolean;
    method?: string;
    status?: string;
    reason?: string;
  };
};

/**
 * A visual ranking. The server emits `ordering: "indeterminate"` and
 * `supported: false` alongside the sort, which is exactly the rule a scoreboard
 * has to keep: it may sort estimates, it may not claim the order.
 */
export type RankingScope = JsonRecord & {
  scope?: string;
  visual_order?: string[];
  estimates?: { policy?: string; estimate?: number | null; reference_rate?: number | null }[];
  ordering?: string;
  supported?: boolean;
  reason?: string;
};

/** Provisional operational counts, kept separate from study statistics. */
export type OperationalSummary = {
  status?: string;
  reason?: string;
  reported_records?: number;
  planned_records?: number;
  terminal_records?: number;
  nonterminal_records?: number;
  status_counts?: Record<string, number>;
  scientific_rates?: null;
};

export type StudySummary = {
  status?: undefined;
  n?: number;
  valid?: number;
  successes?: number;
  missing?: number;
  coverage?: number | null;
  rate?: number | null;
  positive_rate?: number | null;
  missing_bounds?: unknown;
  status_counts?: Record<string, number>;
  missing_reason_counts?: Record<string, number>;
  service_failures?: number;
  macro?: JsonRecord[];
  per_task?: Record<string, JsonRecord>;
};

export type AnalysisSummary = JsonRecord & (OperationalSummary | StudySummary);

export type AnalysisResponse = {
  schema_version?: number;
  endpoint?: { name?: string; definition?: string };
  analysis_identity?: JsonRecord & { status?: string; reason?: string };
  ledger_state?: {
    active?: boolean;
    nonterminal_status_counts?: Record<string, number>;
    run_ids?: string[];
  };
  matrix?: { policies?: string[]; tasks?: string[] };
  cells?: AnalysisCell[];
  summary?: AnalysisSummary;
  rankings?: RankingScope[];
  pairwise?: PairwiseVerdict[];
  advanced_inference?: JsonRecord & { status?: string; reason?: string };
  lineage_leakage?: JsonRecord & { status?: string; reason?: string | null };
  reliability?: JsonRecord & { status?: string; reason?: string };
  mmrv?: JsonRecord & { status?: string; reason?: string };
  mdd?: JsonRecord & { status?: string; reason?: string };
  mode?: string;
  qualified?: boolean;
};

/** `summary` is a union; this is the only safe way to read it. */
export function isOperationalSummary(
  summary: AnalysisSummary | undefined,
): summary is OperationalSummary & JsonRecord {
  return summary?.status === "provisional_operational_only";
}

export type Gate = JsonRecord & {
  id?: string;
  name?: string;
  description?: string;
  status?: string;
  reason?: string;
  summary?: string;
  blocker?: string;
  details?: string;
};

/**
 * One precomputed cost–fidelity operating point. Every field is read from a
 * completed sweep; the slider never estimates a point that was not measured.
 */
export type SweepPoint = JsonRecord & {
  operating_point_id?: string;
  id?: string;
  label?: string;
  resolution?: string | number;
  denoise_steps?: number;
  chunk_partition?: string | number;
  batch_size?: number;
  fixed_task_horizon?: number | string;
  coverage?: number | null;
  gpu_seconds?: number | null;
  estimated_usd?: number | null;
  pairwise_order_agreement?: number | null;
  tolerances_met?: boolean;
  qualification?: string;
};

export type SweepResponse = { points?: SweepPoint[]; status?: string; reason?: string };

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

/** Live telemetry, §7 source mapping. Every numeric field may be absent. */
export type Telemetry = JsonRecord & {
  source?: string;
  timestamp?: number | string;
  fresh_at?: number | string;
  stale?: boolean;
  reconciliation_status?: string;
  platform_queue?: number | null;
  platform_queue_status?: string;
  active_replicas?: number | null;
  desired_replicas?: number | null;
  gpu_seconds?: number | null;
  compute_seconds?: number | null;
  marginal_estimated_usd?: number | null;
  total_estimated_usd?: number | null;
  mode?: string;
};

export type RunSnapshot = {
  run?: Run;
  episodes?: Episode[];
  analysis?: AnalysisResponse;
  telemetry?: Telemetry;
};

/** SSE `segment_completed` payload. Tiles accumulate from these. */
export type SegmentCompletedEvent = {
  episode_id?: string;
  run_id?: string;
  policy?: string;
  task?: string;
  segment_index?: number;
  frame_urls?: string[];
  certified_frame_count?: number;
  provenance?: Provenance;
  resolution?: string;
  status?: string;
};

export type SixClip = {
  id?: string;
  url?: string;
  provenance_sealed?: boolean;
  provenance?: "real" | "generated";
  source?: string;
};

/** Present only when audience answers were actually recorded. */
export type AudienceAnswers = { n?: number; correct?: number; method?: string } | null;

export type SixClipResponse = {
  clips?: SixClip[];
  revealed?: boolean;
  status?: string;
  reason?: string;
  recorded_answers?: AudienceAnswers;
  /** The name the control plane currently ships for the same field. */
  audience_accuracy?: AudienceAnswers;
};

export type CalledShot = {
  cell?: string;
  policy?: string;
  task?: string;
  human_rate?: number | null;
  human_successes?: number | null;
  human_n?: number | null;
  simpler_rate?: number | null;
  simpler_successes?: number | null;
  simpler_n?: number | null;
  published_gap_points?: number | null;
  plumb_estimate?: number | null;
  plumb_status?: string;
  plumb_run_id?: string | null;
  /** Flat form. */
  prereg_uri?: string | null;
  prereg_sha256?: string | null;
  /** Nested form currently shipped by the control plane. */
  preregistration?: { status?: string; uri?: string | null; sha256?: string | null } | null;
  status?: string;
  reason?: string;
  note?: string;
};

export type FreeplayDirection = "up" | "down" | "left" | "right" | "stop";

/**
 * The interactive demo is deliberately separate from the historical free-play
 * developer tool. A demo session has persisted media and an explicit
 * experimental state mode; it is not a benchmark result or checkpoint resume.
 */
export type DemoMode = "policy" | "manual";
export type DemoDirection =
  | "up"
  | "down"
  | "left"
  | "right"
  | "forward"
  | "back";
export type DemoState =
  | "queued"
  | "initializing"
  | "ready"
  | "running"
  | "completed"
  | "blocked"
  | "error";

export type DemoFrame = {
  index?: number;
  url?: string;
  sha256?: string;
  role?: "source" | "fixture" | "predicted" | string;
  input_sha256?: string;
  conditioning_sha256?: string;
  action7?: number[];
  timings_ms?: Record<string, number | null>;
  model?: string | JsonRecord;
  revision?: string | JsonRecord;
  provenance?: string;
};

export type DemoSession = {
  world_model?: string;
  world_progress?: { stage: string; step: number; total: number };
  world_timing?: { world_seconds?: number; model_load_seconds?: number | null; peak_gpu_bytes?: number };
  auto_assess?: boolean;
  judgment?: DemoJudgment | null;
  assessment_error?: string | null;
  generation_completed_at?: number;
  created_at?: number;
  updated_at?: number;
  id: string;
  title?: string;
  prompt?: string;
  mode: DemoMode;
  steps?: number;
  requested_steps?: number;
  completed_steps?: number;
  max_steps?: number;
  remaining_steps?: number;
  current_frame_index?: number;
  policy_label?: "OpenVLA" | "Manual directional action" | string;
  state: DemoState;
  frames?: DemoFrame[];
  frame_urls?: string[];
  latest_video_url?: string;
  error?: string;
  warnings?: string[];
  state_mode?: "experimental_reencoded_rgb_stateless" | string;
  qualified?: false;
  source?: {
    video_id?: string;
    sha256?: string;
    mode?: "fresh_image_reconditioned_not_exact_checkpoint_restore" | string;
  };
};

export type DemoStatus = {
  world_model?: string;
  available: boolean;
  configured: boolean;
  health: "ready" | "available" | "unavailable" | "error" | "not_configured" | string;
  reason?: string;
  state_mode?: "experimental_reencoded_rgb_stateless" | string;
  manual_directions?: DemoDirection[];
  qualified?: false;
};

export type DemoJudgeClip = {
  id: string;
  title?: string;
  video_url?: string;
  poster_url?: string;
  report_url?: string;
  video_sha256?: string;
  task_id?: string;
  task_label?: string;
  action_source?: string;
  controller_identity?: string;
  scene_reference_role?: string;
};

export type DemoJudgeReadiness = {
  profile?: string;
  configured?: boolean;
  available?: boolean;
  reason?: string;
  loaded_adapter_verified?: boolean;
  base_model?: { id?: string; revision?: string };
  adapter?: { id?: string; tree_sha256?: string; recorded_path?: string };
  default_clip?: DemoJudgeClip | null;
};

export type DemoJudgeAssessment = {
  status?: "evaluable" | "unable_to_assess" | string;
  label?: string;
  progress?: number | null;
  visual_integrity?: string;
  collision?: string;
  completion?: string;
  explanation?: string | null;
  evidence_frame_indices?: number[];
  quorum?: number;
};

export type DemoJudgment = {
  progress?: { stage?: string; completed_samples?: number; sample_count?: number } | null;
  id: string;
  status: "queued" | "warming" | "assessing" | "completed" | "abstained" | "failed" | "interrupted" | string;
  clip_id?: string;
  profile?: string;
  created_at?: string;
  updated_at?: string;
  finished_at?: string | null;
  input?: JsonRecord & { frame_indexes?: number[]; video_sha256?: string; clip?: DemoJudgeClip };
  result?: JsonRecord & {
    assessment?: DemoJudgeAssessment;
    adapter_receipt?: JsonRecord;
    timing?: JsonRecord;
    report?: JsonRecord;
  };
  error?: { kind?: string; message?: string; automatic_retry_allowed?: boolean } | null;
  result_url?: string | null;
  raw_response_url?: string | null;
  failure_url?: string | null;
  experimental?: true;
  qualified?: false;
};

export type CreateDemoSessionBody = {
  demo_policy?: "openvla" | "pi0" | "octo" | "baseline";
  world_model?: "cosmos";
  auto_assess?: boolean;
  starting_scene?: "drawer" | "pot";
  seed?: number;
  title?: string;
  prompt?: string;
  mode: DemoMode;
  steps?: number;
  source_video_id?: string;
};

export type FreeplayResponse = {
  mode?: string;
  source?: { video_id?: string; sha256?: string };
  binding?: { exact_branch_supported?: boolean };
  session_id?: string;
  frame_urls?: string[];
  frame_count?: number;
  latency_ms?: number;
  generating?: boolean;
  qualified?: boolean;
  backend?: string;
  reason?: string;
  resolution?: string;
  /** Server-declared per-axis clamp, echoed so the UI never asserts its own. */
  action_clamp?: number;
  chunk_size?: number;
};

export class ApiError extends Error {
  readonly status: number;
  readonly payload: JsonRecord | undefined;
  readonly reason: string | undefined;

  constructor(message: string, status: number, payload?: JsonRecord) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
    this.reason = typeof payload?.reason === "string" ? payload.reason : undefined;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`.trim();
    let payload: JsonRecord | undefined;
    try {
      const parsed = (await response.json()) as unknown;
      if (isRecord(parsed)) {
        payload = parsed;
        const detail = parsed.detail;
        if (typeof detail === "string") message = detail;
        else if (isRecord(detail)) {
          payload = { ...parsed, ...detail };
          if (typeof detail.reason === "string") message = detail.reason;
        } else if (typeof parsed.reason === "string") message = parsed.reason;
      }
    } catch {
      // An empty or non-JSON error body is still reported using its HTTP status.
    }
    throw new ApiError(message, response.status, payload);
  }
  return response.json() as Promise<T>;
}

export type CreateRunBody = {
  mode: "synthetic";
  backend: "synthetic";
  policies: string[];
  tasks: string[];
  starts_per_task: number;
  seed: number;
  idempotency_key: string;
  /** Free-text task strings. The server forces these into the exploration cohort. */
  prompts?: string[];
};

export const api = {
  health: () => request<JsonRecord>("/api/health"),
  protocol: () => request<JsonRecord>("/api/protocol"),
  gates: () => request<{ gates?: Gate[]; qualified?: boolean; reason?: string }>("/api/gates"),
  runs: () => request<{ runs?: Run[] }>("/api/runs"),
  run: (runId: string) => request<Run>(`/api/runs/${encodeURIComponent(runId)}`),
  episodes: (runId: string) =>
    request<{ episodes?: Episode[] }>(`/api/runs/${encodeURIComponent(runId)}/episodes`),
  analysis: (runId: string) =>
    request<AnalysisResponse>(`/api/runs/${encodeURIComponent(runId)}/analysis`),
  sweeps: () => request<SweepResponse>("/api/sweeps"),
  experiments: () => request<{ experiments?: Experiment[] }>("/api/experiments"),
  createRun: (body: CreateRunBody) =>
    request<Run>("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  cancelRun: (runId: string) =>
    request<Run>(`/api/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" }),
  freeplayStep: (body: { session_id?: string; direction: FreeplayDirection; video_id?: string; source_sha256?: string }) =>
    request<FreeplayResponse>("/api/freeplay/step", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  demoStatus: () => request<DemoStatus>("/api/demo/status"),
  demoSessions: () => request<{ sessions?: DemoSession[] }>("/api/demo/sessions"),
  demoSession: (sessionId: string) =>
    request<DemoSession>(`/api/demo/sessions/${encodeURIComponent(sessionId)}`),
  createDemoSession: (body: CreateDemoSessionBody) =>
    request<DemoSession>("/api/demo/sessions", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  demoCommand: (
    sessionId: string,
    body: { direction: DemoDirection; steps?: number; request_id?: string },
  ) =>
    request<DemoSession>(`/api/demo/sessions/${encodeURIComponent(sessionId)}/commands`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  demoJudgeReadiness: () => request<DemoJudgeReadiness>("/api/demo/judge/readiness"),
  demoJudgments: (clipId?: string) =>
    request<{ judgments?: DemoJudgment[] }>(
      `/api/demo/judgments${clipId ? `?clip_id=${encodeURIComponent(clipId)}` : ""}`,
    ),
  demoJudgment: (judgmentId: string) =>
    request<DemoJudgment>(`/api/demo/judgments/${encodeURIComponent(judgmentId)}`),
  createDemoJudgment: (body: { clip_id: string; profile: "semantic_pilot_epoch_02"; idempotency_key: string }) =>
    request<DemoJudgment>("/api/demo/judgments", { method: "POST", body: JSON.stringify(body) }),
  sixClip: () => request<SixClipResponse>("/api/clips/sixclip"),
  revealSixClip: () =>
    request<SixClipResponse>("/api/clips/sixclip/reveal", { method: "POST" }),
  calledShot: () => request<CalledShot>("/api/calledshot"),
};

export function isRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function artifactUrl(value: unknown): string | undefined {
  if (typeof value !== "string" || value.length === 0) return undefined;
  if (/^https?:\/\//i.test(value) || value.startsWith("data:") || value.startsWith("/")) return value;
  return `/api/artifacts/${value.split("/").map(encodeURIComponent).join("/")}`;
}

export function artifactUrls(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const url = artifactUrl(item);
    return url ? [url] : [];
  });
}
