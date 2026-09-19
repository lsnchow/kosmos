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
