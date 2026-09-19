import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vitest/config";

const here = dirname(fileURLToPath(import.meta.url));
const NOTICES = "THIRD-PARTY-NOTICES.md";

/**
 * Serve and ship `web/THIRD-PARTY-NOTICES.md` at `/THIRD-PARTY-NOTICES.md`.
 *
 * The attribution the MIT licence requires lives in exactly one file. Copying it
 * into `public/` would leave two copies to drift apart, so the file stays where
 * it is and this plugin exposes it in dev and emits it into the build.
 */
function thirdPartyNotices(): Plugin {
  const read = () => readFileSync(resolve(here, NOTICES), "utf8");
  return {
    name: "plumb-third-party-notices",
    configureServer(server) {
      server.middlewares.use(`/${NOTICES}`, (_request, response) => {
        response.setHeader("Content-Type", "text/markdown; charset=utf-8");
        response.end(read());
      });
    },
    generateBundle() {
      this.emitFile({ type: "asset", fileName: NOTICES, source: read() });
    },
  };
}

export default defineConfig({
  plugins: [react(), thirdPartyNotices()],
  server: {
    proxy: {
      "/api": "http://localhost:8787",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    restoreMocks: true,
  },
});
