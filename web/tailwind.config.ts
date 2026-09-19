import type { Config } from "tailwindcss";

/**
 * The console is styled by `src/styles.css` and its token system; Tailwind is
 * here for the landing page, which is utility-driven.
 *
 * These extensions exist so the two cannot drift. That was the stated goal
 * before and it was not met: the landing painted its text in `text-white/70`
 * and friends — a six-step opacity ramp with no relationship to the console's
 * inks — and reached for the three shared colours a total of seven times.
 *
 * The ramp is now the ink scale. `text-fg-muted` on the landing is the same
 * #bebebe the scoreboard sets a source line in, read from the same custom
 * property rather than approximated with an alpha. Opacity is left for
 * chrome — rules, fills, scrims — where it is a material and not a voice.
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: "var(--accent)",
        "accent-bright": "var(--accent-bright)",
        caution: "var(--caution-text)",
        "accent-ink": "var(--accent-ink)",
        fg: {
          DEFAULT: "var(--text)",
          strong: "var(--text-strong)",
          muted: "var(--text-muted)",
          dim: "var(--text-dim)",
        },
      },
      /*
       * One scale, shared. The landing was picking from Tailwind's eleven
       * default steps — xs through 7xl, with `text-sm` alone used thirty times
       * — while the console ran on six role tokens. Two type systems on two
       * routes of one product is exactly the drift these extensions exist to
       * stop, so `text-*` now resolves to the console's `--fs-*`.
       *
       * The consequence is deliberate: presentation mode, which re-declares
       * every --fs-* step, now scales the landing too. It did not before.
       *
       * Display sizes above the console's largest role keep a clamp, because a
       * hero has to hold a viewport the console never sees.
       */
      fontSize: {
        xs: ["var(--fs-3xs)", { lineHeight: "1.4" }],
        sm: ["var(--fs-2xs)", { lineHeight: "1.55" }],
        base: ["var(--fs-xs)", { lineHeight: "1.6" }],
        lg: ["var(--fs-lg)", { lineHeight: "1.35" }],
        xl: ["var(--fs-xl)", { lineHeight: "1.25" }],
        "2xl": ["var(--fs-2xl)", { lineHeight: "1.2" }],
        "3xl": ["var(--fs-3xl)", { lineHeight: "1.15" }],
        "4xl": ["clamp(2rem, 3.4vw, 2.75rem)", { lineHeight: "1.12" }],
        "5xl": ["clamp(2.5rem, 4.4vw, 3.5rem)", { lineHeight: "1.08" }],
        "6xl": ["clamp(2.75rem, 5.2vw, 4rem)", { lineHeight: "1.05" }],
        "7xl": ["clamp(3rem, 6vw, 4.75rem)", { lineHeight: "1.02" }],
      },
      fontFamily: {
        /* There is no proportional register any more: `font-sans` resolves to
           the same mono stack as `font-mono`, so a stray `font-sans` on the
           landing cannot reintroduce one. */
        sans: ["Geist Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        mono: ["Geist Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        display: ["Bitcount Grid Double", "Geist Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
