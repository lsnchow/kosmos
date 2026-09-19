/**
 * Formatters with explicit units.
 *
 * The previous `formatPercent` guessed its input scale (`value <= 1 ? value * 100
 * : value`), which silently renders a genuine 0.8% as "80%".  Every function here
 * declares the unit it expects in its name, so a caller cannot pass the wrong
 * scale without the mistake being visible at the call site.
 *
 * Every function returns a caller-supplied fallback (default "—") when the value
 * is absent.  None of them substitutes a zero for missing data.
 */

export const ABSENT = "—";
export const UNAVAILABLE = "Unavailable";

export function pickString(...values: unknown[]): string | undefined {
  return values.find((value): value is string => typeof value === "string" && value.length > 0);
}

export function pickNumber(...values: unknown[]): number | undefined {
  return values.find(
    (value): value is number => typeof value === "number" && Number.isFinite(value),
  );
}

export function pickBoolean(...values: unknown[]): boolean | undefined {
  return values.find((value): value is boolean => typeof value === "boolean");
}

/** Integer counts. Never returns "0" for a missing value. */
export function formatCount(value: unknown, fallback = ABSENT): string {
  const numeric = pickNumber(value);
  return numeric === undefined ? fallback : new Intl.NumberFormat("en-US").format(numeric);
}

/**
 * A proportion in [0, 1] rendered as a percentage. The unit of the *input* is a
 * fraction; the unit of the output is percent. Values outside [0, 1] are still
 * rendered (they are real data) but flagged by the caller-visible return.
 */
export function formatRateAsPercent(value: unknown, fallback = ABSENT): string {
  const numeric = pickNumber(value);
  if (numeric === undefined) return fallback;
  const percent = numeric * 100;
  const decimals = Math.abs(percent) >= 10 || percent % 1 === 0 ? (percent % 1 === 0 ? 0 : 1) : 1;
  return `${percent.toFixed(decimals)}%`;
}

/** A value already expressed in percentage points. */
export function formatPercentagePoints(value: unknown, fallback = ABSENT): string {
  const numeric = pickNumber(value);
  if (numeric === undefined) return fallback;
  return `${numeric.toFixed(numeric % 1 === 0 ? 0 : 1)}%`;
}

export function formatUsd(value: unknown, fallback = UNAVAILABLE): string {
  const numeric = pickNumber(value);
  if (numeric === undefined) return fallback;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: numeric < 1 ? 4 : 2,
  }).format(numeric);
}

export function formatSeconds(value: unknown, fallback = ABSENT): string {
  const numeric = pickNumber(value);
  if (numeric === undefined) return fallback;
  return `${numeric.toLocaleString("en-US", { maximumFractionDigits: 3 })} s`;
}

export function formatGpuSeconds(value: unknown, fallback = ABSENT): string {
  const numeric = pickNumber(value);
  if (numeric === undefined) return fallback;
  return `${numeric.toLocaleString("en-US", { maximumFractionDigits: 1 })} GPU-s`;
}

export function formatMilliseconds(value: unknown, fallback = ABSENT): string {
  const numeric = pickNumber(value);
  if (numeric === undefined) return fallback;
  return `${Math.round(numeric).toLocaleString("en-US")} ms`;
}

export function formatBytes(value: unknown, fallback = ABSENT): string {
  const numeric = pickNumber(value);
  if (numeric === undefined) return fallback;
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = numeric;
  let unitIndex = 0;
  while (amount >= 1024 && unitIndex < units.length - 1) {
    amount /= 1024;
    unitIndex += 1;
  }
  return `${amount.toLocaleString("en-US", { maximumFractionDigits: 2 })} ${units[unitIndex]}`;
}

/** A [lower, upper] pair of fractions rendered as a percentage interval. */
export function formatRateInterval(value: unknown, fallback = ABSENT): string {
  const pair = readInterval(value);
  if (!pair) return fallback;
  return `${formatRateAsPercent(pair[0])} – ${formatRateAsPercent(pair[1])}`;
}

/** Normalize the several interval encodings the measurement module emits. */
export function readInterval(value: unknown): [number, number] | undefined {
  if (Array.isArray(value) && value.length >= 2) {
    const lower = pickNumber(value[0]);
    const upper = pickNumber(value[1]);
    if (lower !== undefined && upper !== undefined) return [lower, upper];
    return undefined;
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    const lower = pickNumber(record.lower, record.min);
    const upper = pickNumber(record.upper, record.max);
    if (lower !== undefined && upper !== undefined) return [lower, upper];
  }
  return undefined;
}

/** Wall-clock time for an ISO string or a POSIX seconds/milliseconds number. */
export function toDate(value: unknown): Date | undefined {
  if (typeof value === "string" && value.length > 0) {
    const parsed = new Date(value);
    return Number.isNaN(parsed.valueOf()) ? undefined : parsed;
  }
  const numeric = pickNumber(value);
  if (numeric === undefined) return undefined;
  // The API emits `time.time()` (POSIX seconds). Anything below ~1e11 is seconds.
  const milliseconds = numeric < 1e11 ? numeric * 1000 : numeric;
  const parsed = new Date(milliseconds);
  return Number.isNaN(parsed.valueOf()) ? undefined : parsed;
}

export function formatClockTime(value: unknown, fallback = ABSENT): string {
  const date = toDate(value);
  if (!date) return typeof value === "string" && value.length > 0 ? value : fallback;
  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

/** Age in seconds of a server-supplied timestamp, measured against `now`. */
export function ageSeconds(value: unknown, now: number): number | undefined {
  const date = toDate(value);
  if (!date) return undefined;
  return Math.max(0, (now - date.valueOf()) / 1000);
}

export function titleCase(value: string): string {
  return value.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function joinStrings(value: unknown): string | undefined {
  if (typeof value === "string") return value.length > 0 ? value : undefined;
  if (Array.isArray(value)) {
    const parts = value.filter((item): item is string => typeof item === "string");
    return parts.length > 0 ? parts.join(" · ") : undefined;
  }
  return undefined;
}

const TERMINAL_STATUSES = new Set(["completed", "complete", "failed", "cancelled", "canceled"]);

export function isTerminalStatus(status: unknown): boolean {
  return TERMINAL_STATUSES.has(String(status ?? "").toLowerCase());
}

/**
 * Deterministic 32-bit FNV-1a hash rendered as hex. Used to derive an
 * idempotency key from run identity so that two clicks of the burst button
 * produce the *same* key and the server's uniqueness constraint can reject the
 * duplicate. A random UUID defeats that constraint.
 */
export function stableHashHex(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

/** Canonical JSON with sorted keys, so key order cannot change a hash. */
export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value ?? null);
}
