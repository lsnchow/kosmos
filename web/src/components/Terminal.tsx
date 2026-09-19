import { useSyncExternalStore } from "react";
import { getDisplayTick, subscribeDisplayClock } from "../lib/clock";
import { cn } from "../lib/utils";

/**
 * Character-cell widgets, ported from the Rich TUI in `longshot/dashboard.py`.
 *
 * The point of these is quantisation. A CSS width of `62.4%` is a claim this
 * console cannot support: it implies a precision the underlying count does not
 * have, and it animates through values that were never measured. A terminal bar
 * is N cells wide and fills a whole cell at a time, so the drawing can only ever
 * say something the data actually said. Every widget here rounds the same way
 * the donor does — `int()` for bars, `round()` for the four-cell meter — and
 * renders literal glyphs rather than sized boxes.
 *
 * All of them are `aria-hidden` and paired with real text by their callers. A
 * screen reader gets "87 of 100 replicas", never "block block block light-shade".
 */

/** The donor's ramps, verbatim. */
const BAR_FULL = "█"; // █
const BAR_TRACK = "░"; // ░
const METER_FULL = "■"; // ■
const METER_EMPTY = "□"; // □
const SPARK_RAMP = " ▁▂▃▄▅▆▇█"; // ▁▂▃▄▅▆▇█

export type CellTone = "accent" | "good" | "caution" | "bad" | "info" | "muted";

function toneClass(tone: CellTone | undefined) {
  return `cell-${tone ?? "accent"}`;
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/**
 * A progress bar drawn in cells: `████████░░░░░░░░` .
 *
 * `width` is a cell count, not a length. The donor uses 20 for the merge-rate
 * bar and 50 for the footer; both are exposed as tokens rather than magic
 * numbers so a caller picks a register instead of a pixel size.
 */
export function CellBar({
  value,
  width = 20,
  tone,
  className,
}: {
  /** Fraction in [0, 1]. Values outside the range are clamped, never wrapped. */
  value: number;
  width?: number;
  tone?: CellTone;
  className?: string;
}) {
  // int(), not round(): a bar must never show a cell of progress that has not
  // happened. 19.9 cells of 20 is nineteen cells and an unfinished twentieth.
  const filled = Math.min(width, Math.floor(clamp01(value) * width));
  return (
    <span aria-hidden="true" className={cn("cell-bar", toneClass(tone), className)}>
      <span className="cell-bar-fill">{BAR_FULL.repeat(filled)}</span>
      <span className="cell-bar-track">{BAR_TRACK.repeat(Math.max(0, width - filled))}</span>
    </span>
  );
}

/**
 * The four-cell node meter from the planner tree: `■■□□`.
 *
 * This one rounds rather than truncates, matching the donor — at four cells a
 * floor would show an empty meter for anything under 25%, which reads as "not
 * started" for work that is a fifth done.
 */
export function Meter({
  progress,
  width = 4,
  tone,
  className,
}: {
  progress: number;
  width?: number;
  tone?: CellTone;
  className?: string;
}) {
  const fill = Math.min(width, Math.max(0, Math.round(clamp01(progress) * width)));
  return (
    <span aria-hidden="true" className={cn("cell-meter", toneClass(tone), className)}>
      <span className="cell-bar-fill">{METER_FULL.repeat(fill)}</span>
      <span className="cell-bar-track">{METER_EMPTY.repeat(width - fill)}</span>
    </span>
  );
}

/**
 * A sparkline over the eight block heights, normalised to the tallest bucket.
 *
 * Normalising to the local maximum is the donor's choice and it is the right
 * one for a shape-at-a-glance widget, but it means the height of a bar carries
 * no absolute magnitude. Callers print the current value beside it — the
 * sparkline says "rising", the number says how much.
 */
export function Sparkline({
  buckets,
  width = 10,
  tone,
  className,
}: {
  buckets: readonly number[];
  width?: number;
  tone?: CellTone;
  className?: string;
}) {
  const padded =
    buckets.length >= width
      ? buckets.slice(buckets.length - width)
      : [...new Array<number>(width - buckets.length).fill(0), ...buckets];
  const peak = Math.max(1, ...padded.map((bucket) => (Number.isFinite(bucket) ? bucket : 0)));
  const last = SPARK_RAMP.length - 1;
  const glyphs = padded
    .map((bucket) => {
      const level = Math.floor((Number.isFinite(bucket) ? Math.max(0, bucket) : 0) / peak * last);
      return SPARK_RAMP[Math.min(last, level)];
    })
    .join("");
  return (
    <span aria-hidden="true" className={cn("cell-spark", toneClass(tone), className)}>
      {glyphs}
    </span>
  );
}

/**
 * Tree connectors: `├─ ` for a sibling, `└─ ` for the last child.
 *
 * Rendered as generated-content-free real glyphs so they survive copy-paste,
 * but `aria-hidden`, because the tree structure is already carried by the list
 * markup the caller wraps this in.
 */
export function TreeBranch({ last = false, className }: { last?: boolean; className?: string }) {
  return (
    <span aria-hidden="true" className={cn("tree-branch", className)}>
      {last ? "└─ " : "├─ "}
    </span>
  );
}

/** The planner's thinking indicator: ● ●● ●●● at 2 Hz. CSS drives the cycle. */
export function Dots({ className }: { className?: string }) {
  return <span aria-hidden="true" className={cn("cell-dots", className)} />;
}

/**
 * The spinner, as a terminal draws one. Four frames on the shared display
 * clock rather than a CSS animation, so it is deterministic under test —
 * `advanceDisplayClock()` steps it exactly as it steps the flipbooks.
 */
const SPINNER_FRAMES = ["|", "/", "-", "\\"] as const;

export function AsciiSpinner({ className }: { className?: string }) {
  const tick = useSyncExternalStore(subscribeDisplayClock, getDisplayTick, () => 0);
  // The display clock runs at 8 Hz; halving it puts the spinner at 4 Hz, which
  // is the donor's whole-dashboard refresh rate.
  const frame = SPINNER_FRAMES[Math.floor(tick / 2) % SPINNER_FRAMES.length];
  return (
    <span aria-hidden="true" className={cn("ascii-spinner", className)}>
      {frame}
    </span>
  );
}

/** A blinking block cursor. Purely decorative; the input beside it is the control. */
export function Caret({ className }: { className?: string }) {
  return <span aria-hidden="true" className={cn("caret", className)} />;
}

/**
 * Text glyphs in place of drawn icons.
 *
 * Every entry is a character a terminal can print. They are always
 * `aria-hidden`, exactly as the lucide icons they replace were, so no
 * accessible name changes — an icon-only control still gets its name from
 * `aria-label`, and a labelled one still reads as its label alone.
 */
const GLYPHS = {
  play: "▸", // ▸
  stop: "■", // ■
  pause: "‖", // ‖
  expand: "⤡", // ⤡
  drive: "»", // »
  close: "×", // ×
  check: "✓", // ✓
  cross: "✗", // ✗
  warn: "!",
  alert: "⚠", // ⚠
  arrowRight: "→", // →
  arrowUpRight: "↗", // ↗
  arrowUp: "↑", // ↑
  arrowDown: "↓", // ↓
  arrowLeft: "←", // ←
  enter: "↵", // ↵
  clock: "@",
  live: "●", // ●
  offline: "×", // ×
  target: "⌖", // ⌖
  badge: "+",
  slash: "⊘", // ⊘
  link: "──", // ──
  eye: "?",
  lock: "#",
  sparkle: "*",
  minus: "−", // −
  refresh: "⟳", // ⟳
  menu: "≡", // ≡
  present: "▭", // ▭
  gauge: "⌗", // ⌗
  table: "≡", // ≡
  shield: "§", // §
  flask: "⚗", // ⚗
  dash: "∷", // ∷
  film: "▷", // ▷
  monitor: "▭", // ▭
  sliders: "─○─", // ─○─
  layers: "≣", // ≣
  scale: "⚖", // ⚖
  activity: "∿", // ∿
} as const;

export type GlyphName = keyof typeof GLYPHS;

export function Glyph({ name, className }: { name: GlyphName; className?: string }) {
  return (
    <span aria-hidden="true" className={cn("glyph", className)}>
      {GLYPHS[name]}
    </span>
  );
}
