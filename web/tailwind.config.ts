import type { Config } from "tailwindcss";

/**
 * The console is styled by `src/styles.css` and its token system; Tailwind is
 * here for the landing page, which is utility-driven.
 *
 * The extensions below exist so the two cannot drift: `font-mono` on the
 * landing is the same Geist Mono the scoreboard sets its figures in, and
 * `text-accent` is the same lime the console uses for brand and pass, read from
 * the same custom property rather than re-typed as a hex.
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: "var(--accent)",
        "accent-bright": "var(--accent-bright)",
        caution: "var(--caution-text)",
      },
      fontFamily: {
        sans: ["Geist Sans", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["Geist Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        serif: ["Instrument Serif", "ui-serif", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
} satisfies Config;
