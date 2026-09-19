import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * The contrast ratios in the header of `styles.css` are a claim about the
 * palette, so they are checked the way every other number on this project is
 * checked: computed, not eyeballed.
 *
 * Two things are asserted. First, each ink clears the ratio the comment block
 * documents, against every surface text sits on. Second, the comment block
 * actually contains those numbers, so it cannot quietly drift away from the
 * tokens beneath it — which is exactly how the previous sheet ended up quoting
 * ratios measured against a background it no longer used.
 */
/*
 * The sheet is read off disk rather than imported. Vitest resolves a CSS import
 * to an empty string by default (`css: false`), and jsdom has no cascade to
 * inspect, so the declarations themselves are the only thing to assert on.
 */
const css = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "styles.css"), "utf8");

function rootTokens(source: string): Record<string, string> {
  const block = /:root\s*\{([\s\S]*?)\n\}/.exec(source.replace(/\/\*[\s\S]*?\*\//g, ""));
  if (!block) throw new Error("styles.css has no :root block");
  const tokens: Record<string, string> = {};
  for (const match of block[1].matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/g)) {
    tokens[match[1]] = match[2].trim();
  }
  return tokens;
}

const tokens = rootTokens(css);

function channel(value: number): number {
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

export function relativeLuminance(hex: string): number {
  const cleaned = hex.trim().replace("#", "");
  const full =
    cleaned.length === 3
      ? cleaned
          .split("")
          .map((part) => part + part)
          .join("")
      : cleaned;
  if (!/^[0-9a-f]{6}$/i.test(full)) throw new Error(`Not a hex colour: ${hex}`);
  const [r, g, b] = [0, 2, 4].map((offset) => channel(parseInt(full.slice(offset, offset + 2), 16) / 255));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrastRatio(a: string, b: string): number {
  const first = relativeLuminance(a);
  const second = relativeLuminance(b);
  const lighter = Math.max(first, second);
  const darker = Math.min(first, second);
  return (lighter + 0.05) / (darker + 0.05);
}

function colour(token: string): string {
  const value = tokens[token];
  if (value === undefined) throw new Error(`styles.css declares no ${token}`);
  return value;
}

/** Every surface a text node actually sits on, darkest to lightest. */
const TEXT_SURFACES = ["--surface-0", "--surface-1", "--surface-2", "--surface-3"] as const;

/**
 * The documented floor for each ink, measured against `--surface-3` — the
 * lightest of the surfaces above, so every figure is a worst case.
 */
const DOCUMENTED: Record<string, number> = {
  "--text-strong": 15.9,
  "--text": 14.8,
  "--text-muted": 9.3,
  "--text-dim": 7.3,
  "--accent": 11.0,
  "--accent-bright": 12.6,
  "--good": 11.0,
  "--bad": 5.9,
  "--caution-text": 12.0,
  "--caution-strong": 15.6,
  "--caution": 8.5,
  "--info": 8.6,
};

/**
 * Tokens that are legitimately used as a `color:` without meeting the 4.5:1 ink
 * bar, each with the reason and the pair it *is* measured against.
 */
const INK_EXCEPTIONS: Record<string, { against: string; minimum: number; why: string }> = {
  "--accent-ink": {
    against: "--accent",
    minimum: 4.5,
    why: "black ink on the lime fill of .button-primary and .skip-link",
  },
};

describe("palette contrast", () => {
  it("declares the surfaces and inks the comment block measures", () => {
    for (const token of [...TEXT_SURFACES, ...Object.keys(DOCUMENTED)]) {
      expect(colour(token)).toMatch(/^#[0-9a-f]{3,6}$/i);
    }
  });

  it.each(Object.entries(DOCUMENTED))(
    "%s clears its documented ratio on every text-bearing surface",
    (token, documented) => {
      const floor = contrastRatio(colour(token), colour("--surface-3"));
      expect(floor).toBeGreaterThanOrEqual(documented);
      for (const surface of TEXT_SURFACES) {
        expect(contrastRatio(colour(token), colour(surface))).toBeGreaterThanOrEqual(documented);
      }
    },
  );

  it("keeps every documented ratio in the comment block, so the comment cannot drift", () => {
    const normalized = css.replace(/\s+/g, " ");
    // The header must name the surface it measured against; the previous sheet's
    // numbers were void precisely because that surface changed underneath them.
    expect(normalized).toContain("--surface-3");
    for (const [token, documented] of Object.entries(DOCUMENTED)) {
      expect(normalized).toContain(`${token} ${documented.toFixed(1)}:1`);
    }
  });

  it("holds WCAG AA for body text on every surface", () => {
    for (const token of ["--text-strong", "--text", "--text-muted", "--text-dim"]) {
      for (const surface of TEXT_SURFACES) {
        expect(contrastRatio(colour(token), colour(surface))).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("holds the 3:1 non-text minimum for the one line that is a control's only affordance", () => {
    for (const surface of TEXT_SURFACES) {
      expect(contrastRatio(colour("--line-interactive"), colour(surface))).toBeGreaterThanOrEqual(3);
    }
    // The focus ring has to be visible against the page it outlines.
    expect(contrastRatio(colour("--accent-bright"), colour("--surface-0"))).toBeGreaterThanOrEqual(3);
    // Danger and caution hairlines group content; they are not sole affordances,
    // but a red hairline that cannot be seen is not a warning.
    expect(contrastRatio(colour("--danger-line"), colour("--surface-0"))).toBeGreaterThanOrEqual(3);
  });

  it("keeps ink legible on every accent fill", () => {
    expect(contrastRatio(colour("--accent-ink"), colour("--accent"))).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(colour("--accent-ink"), colour("--accent-bright"))).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps lime and amber apart, so brand and caution are never the same colour", () => {
    // A status-heavy console cannot use one hue for "this is PLUMB" and "this
    // number is not qualified". Distinct hues, and distinct enough in luminance
    // that the two are not one ramp.
    expect(colour("--accent")).not.toBe(colour("--caution"));
    expect(colour("--accent")).not.toBe(colour("--caution-text"));
    expect(contrastRatio(colour("--accent"), colour("--caution"))).toBeGreaterThan(1.1);
  });

  it("uses no ink anywhere in the sheet that fails 4.5:1 on --surface-3", () => {
    const failures: string[] = [];
    const withoutComments = css.replace(/\/\*[\s\S]*?\*\//g, "");

    for (const match of withoutComments.matchAll(/(?<![a-z-])color\s*:\s*var\((--[a-z0-9-]+)\)/g)) {
      const token = match[1];
      const exception = INK_EXCEPTIONS[token];
      if (exception) {
        const ratio = contrastRatio(colour(token), colour(exception.against));
        if (ratio < exception.minimum) {
          failures.push(`${token} vs ${exception.against} is ${ratio.toFixed(2)}:1 (${exception.why})`);
        }
        continue;
      }
      const ratio = contrastRatio(colour(token), colour("--surface-3"));
      if (ratio < 4.5) failures.push(`color: var(${token}) is ${ratio.toFixed(2)}:1 on --surface-3`);
    }

    // Literal hex inks bypass the token system entirely, so there must be none.
    for (const match of withoutComments.matchAll(/(?<![a-z-])color\s*:\s*(#[0-9a-f]{3,8})/gi)) {
      failures.push(`hard-coded ink ${match[1]} — inks belong to the token system`);
    }

    expect(failures).toEqual([]);
  });
});

describe("type scale and presentation mode", () => {
  it("keeps every step of the scale", () => {
    for (const step of ["3xs", "2xs", "xs", "sm", "md", "lg", "xl", "2xl", "3xl"]) {
      expect(tokens[`--fs-${step}`]).toBeDefined();
    }
  });

  it("re-declares every step in presentation mode at 20 px or above", () => {
    const block = /html\[data-presentation="on"\]\s*\{([\s\S]*?)\n\}/.exec(css);
    expect(block).not.toBeNull();
    const declared = [...(block?.[1].matchAll(/(--fs-[a-z0-9]+)\s*:\s*([\d.]+)rem/g) ?? [])];
    expect(declared.length).toBe(9);
    for (const [, , rem] of declared) {
      expect(Number(rem) * 16).toBeGreaterThanOrEqual(20);
    }
  });

  it("sets every compared figure in the mono face with tabular figures", () => {
    // The scoreboard, the telemetry strip and the ledger are read down a column.
    for (const selector of [
      ".mono, .tabular-nums {",
      "\ntable {",
      ".ladder-detail {",
      ".metric strong {",
      ".tile-meta dd {",
      ".sweep-readout strong {",
      ".provisional-counts dd {",
      ".viewer-meta dd {",
    ]) {
      const index = css.indexOf(selector);
      expect(index, `${selector} is missing`).toBeGreaterThan(-1);
      const rule = css.slice(index, css.indexOf("}", index));
      expect(rule, selector).toContain("var(--font-mono)");
      expect(rule, selector).toContain("tabular-nums");
    }
  });

  it("loads every font from a local path, never a CDN", () => {
    const faces = [...css.matchAll(/@font-face\s*\{([\s\S]*?)\}/g)].map((match) => match[1]);
    expect(faces.length).toBe(6);
    for (const face of faces) {
      expect(face).toMatch(/url\("\/fonts\/[A-Za-z-]+\.woff2"\)/);
      expect(face).not.toMatch(/https?:/);
    }
  });
});
