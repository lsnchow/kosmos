/**
 * Client contract for the isolated, unscored comparison demo.
 *
 * This intentionally does not reuse the study-run client types.  A comparison
 * is a fixed 3 × 4 matched presentation with durable cell ids and no judge
 * fields; mixing it with the historical run ledger would make it far too easy
 * for a UI to imply a score that this release does not have.
 */

export const COMPARISON_SCHEMA = "kosmos-comparison-v1" as const;
export const COMPARISON_TASK = "close_drawer" as const;
export const COMPARISON_INSTRUCTION = "Close the drawer" as const;
export const COMPARISON_POLICIES = ["OpenVLA", "MiniVLA", "Octo-Small"] as const;
export const COMPARISON_SEEDS = [101, 102, 103, 104] as const;
export const COMPARISON_HORIZON = 70;

export type ComparisonPolicy = {
  id: string;
  identity_hashes?: Record<string, string>;
};

export type ComparisonState = {
  payload?: unknown;
  origin?: string;
  conventions_sha256?: string;
  sha256?: string;
};

export type ComparisonStart = {
  id: string;
  png_url: string;
  sha256: string;
  state: ComparisonState;
};

export type ManualProfile = {
  action_rows: number;
  structural_frames: number;
  post_conditioning_frames: number;
};

export type ComparisonWorld = {
  id: string;
  profile_sha256: string;
  manual: ManualProfile;
};

export type ComparisonBudget = {
  cap_usd: number;
  reserved_usd: number;
  available_usd: number;
};

export type ComparisonReadiness = {
  schema: typeof COMPARISON_SCHEMA;
  scored: false;
  claim_tier: "preview";
  task: typeof COMPARISON_TASK;
  task_instruction: typeof COMPARISON_INSTRUCTION;
  available: boolean;
  mechanical: { status: "ready" | "blocked"; reasons: string[] };
  demo_quality: { status: "approved" | "blocked"; reasons: string[] };
  blockers: string[];
  manifest_sha256: string | null;
  start: ComparisonStart | null;
  policies: ComparisonPolicy[];
  seeds: number[];
  horizon: number;
  world: ComparisonWorld | null;
  budget: ComparisonBudget;
  quote: null;
};

export type ComparisonFrame = {
  event_id: string;
  segment_id: string;
  frame_index: number;
  url: string;
  sha256: string;
  action?: unknown;
  state?: ComparisonState;
  created_at?: string;
};

export type ComparisonCellStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "interrupted"
  | "ambiguous";

export type ComparisonCell = {
  id: string;
  policy: (typeof COMPARISON_POLICIES)[number];
  seed: number;
  status: ComparisonCellStatus;
  attempt_id: string | null;
  action_count: number;
  frame_count: number;
  terminal_received: boolean;
  /** Ordered committed output only; the shared input is never included here. */
  frames: ComparisonFrame[];
  latest_frame: ComparisonFrame | null;
  /** Server reason objects are rendered as text by the UI; never a score. */
  error: unknown | null;
  /** Explicit server origin when the journal has it. */
  origin?: "precomputed" | "prewarmed" | "fresh";
};

export type ComparisonReservation = {
  reserved_usd: number;
  breakdown: QuoteBreakdown[];
};

export type ComparisonDetail = {
  schema: typeof COMPARISON_SCHEMA;
  scored: false;
  claim_tier: "preview";
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "interrupted" | "ambiguous";
  created_at: string;
  updated_at: string;
  task: typeof COMPARISON_TASK;
  task_instruction: typeof COMPARISON_INSTRUCTION;
  horizon: number;
  seeds: number[];
  policies: (typeof COMPARISON_POLICIES)[number][];
  start: ComparisonStart;
  world: ComparisonWorld;
  reservation: ComparisonReservation;
  cells: ComparisonCell[];
  scores: null;
  presentation: boolean;
  origin?: "precomputed" | "prewarmed" | "fresh";
};

export type QuoteBreakdown = {
  worker: string;
  cell_count: number;
  bound_seconds: number;
  rate_usd_per_hour: number;
  residency_margin: number;
  reserved_usd: number;
};

export type ComparisonQuote = {
  id: string;
  expires_at: string;
  manifest_sha256: string;
  reservation_usd: number;
  breakdown: QuoteBreakdown[];
};

export type PresentationResponse = {
  schema: typeof COMPARISON_SCHEMA;
  scored: false;
  claim_tier: "preview";
  task: typeof COMPARISON_TASK;
  task_instruction: typeof COMPARISON_INSTRUCTION;
  presentation: ComparisonDetail | null;
  status: "unavailable" | "available";
  reason?: string;
};

export type QuoteResponse = {
  schema: typeof COMPARISON_SCHEMA;
  scored: false;
  claim_tier: "preview";
  task: typeof COMPARISON_TASK;
  task_instruction: typeof COMPARISON_INSTRUCTION;
  quote: ComparisonQuote;
  readiness: ComparisonReadiness;
};

export type CreateComparisonResponse = {
  schema: typeof COMPARISON_SCHEMA;
  scored: false;
  claim_tier: "preview";
  task: typeof COMPARISON_TASK;
  task_instruction: typeof COMPARISON_INSTRUCTION;
  comparison: ComparisonDetail;
  idempotent: boolean;
};

export type ManualDirection = "up" | "down" | "left" | "right" | "forward" | "back";

export type ManualCommandStatus = "queued" | "running" | "completed" | "failed" | "interrupted" | "ambiguous";

export type ManualFrame = {
  frame_index: number;
  url: string;
  sha256: string;
  state?: ComparisonState;
};

export type ManualCommand = {
  id: string;
  status: ManualCommandStatus;
  direction: ManualDirection;
  action_rows: number;
  structural_frames: number;
  post_conditioning_frames: number;
  frame_count: number;
  /** GET session supplies these; command admission supplies an empty `segment`. */
  frames?: ManualFrame[];
  segment?: ManualFrame[];
  error?: unknown | null;
};

export type ManualSession = {
  id: string;
  status: "ready" | "running" | "failed" | "interrupted" | "ambiguous";
  comparison_id: string;
  cell_id: string;
  source: {
    event_id: string;
    url: string;
    sha256: string;
    state: ComparisonState;
  };
  active_command: string | null;
  commands?: ManualCommand[];
};

export type ManualCommandResponse = {
  schema: typeof COMPARISON_SCHEMA;
  scored: false;
  claim_tier: "preview";
  task: typeof COMPARISON_TASK;
  task_instruction: typeof COMPARISON_INSTRUCTION;
  command: ManualCommand;
};

export type ComparisonEvent = {
  sequence: number;
  type:
    | "stage"
    | "frame"
    | "heartbeat"
    | "terminal"
    | "cell_status"
    | "comparison_terminal"
    | "manual_session_created"
    | "manual_command_started"
    | "manual_frame"
    | "manual_terminal";
  comparison_id: string;
  cell_id?: string;
  attempt_id?: string;
  origin?: "precomputed" | "prewarmed" | "fresh";
  payload: Record<string, unknown>;
};

export class ComparisonApiError extends Error {
  readonly status: number;
  readonly reason?: string;

  constructor(message: string, status: number, reason?: string) {
    super(message);
    this.name = "ComparisonApiError";
    this.status = status;
    this.reason = reason;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    let reason: string | undefined;
    try {
      const value: unknown = await response.json();
      if (value && typeof value === "object") {
        const record = value as Record<string, unknown>;
        if (typeof record.reason === "string") reason = record.reason;
        else if (typeof record.detail === "string") reason = record.detail;
      }
    } catch {
      // The status remains useful when a proxy returned a non-JSON error.
    }
    throw new ComparisonApiError(reason ?? `${response.status} ${response.statusText}`.trim(), response.status, reason);
  }
  return response.json() as Promise<T>;
}

export const comparisonApi = {
  readiness: () => request<ComparisonReadiness>("/api/comparisons/readiness"),
  presentation: () => request<PresentationResponse>("/api/comparisons/presentation"),
  quote: () => request<QuoteResponse>("/api/comparisons/quote", { method: "POST", body: "{}" }),
  create: (body: { quote_id: string; idempotency_key: string }) =>
    request<CreateComparisonResponse>("/api/comparisons", { method: "POST", body: JSON.stringify(body) }),
  detail: (comparisonId: string) =>
    request<ComparisonDetail>(`/api/comparisons/${encodeURIComponent(comparisonId)}`),
  createManualSession: (comparisonId: string, body: { cell_id: string }) =>
    request<ManualSession>(`/api/comparisons/${encodeURIComponent(comparisonId)}/manual-sessions`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  manualSession: (sessionId: string) =>
    request<ManualSession>(`/api/manual-sessions/${encodeURIComponent(sessionId)}`),
  command: (sessionId: string, body: { direction: ManualDirection; idempotency_key: string }) =>
    request<ManualCommandResponse>(`/api/manual-sessions/${encodeURIComponent(sessionId)}/commands`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

export function comparisonEventUrl(comparisonId: string, after?: number): string {
  const query = after === undefined ? "" : `?after=${encodeURIComponent(String(after))}`;
  return `/api/comparisons/${encodeURIComponent(comparisonId)}/events${query}`;
}

export function comparisonReady(readiness: ComparisonReadiness | undefined): boolean {
  return Boolean(
    readiness?.available &&
      readiness.mechanical.status === "ready" &&
      readiness.demo_quality.status === "approved" &&
      readiness.start &&
      readiness.world &&
      readiness.horizon === COMPARISON_HORIZON &&
      readiness.policies.map((policy) => policy.id).join("|") === COMPARISON_POLICIES.join("|") &&
      readiness.seeds.join("|") === COMPARISON_SEEDS.join("|") &&
      readiness.world.manual.action_rows === 16 &&
      readiness.world.manual.structural_frames === 17 &&
      readiness.world.manual.post_conditioning_frames === 16,
  );
}

/**
 * Manual steering is intentionally stricter than general world availability.
 * The presentation admits only the certified Cosmos-style 16 → 17 contract;
 * an IRASim native 15 → 16 profile is named as unavailable rather than quietly
 * adapted into a different interaction.
 */
export function strictManualContractReason(world: ComparisonWorld | null | undefined): string | undefined {
  if (
    world?.manual.action_rows === 16 &&
    world.manual.structural_frames === 17 &&
    world.manual.post_conditioning_frames === 16
  )
    return undefined;
  if (
    world?.id.toLowerCase().includes("irasim") &&
    world.manual.action_rows === 15 &&
    world.manual.structural_frames === 16
  )
    return "Steering is unavailable: IRASim’s native 15 actions → 16 structural frames contract is not the admitted 16 actions → 17 returned frames contract.";
  return "Steering is unavailable: the selected world profile does not attest the required 16 actions → 17 returned frames contract.";
}

export function newIdempotencyKey(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return `${prefix}-${crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
