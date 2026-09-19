import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import axe from "axe-core";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { AppDataProvider } from "./AppData";
import { AppRoutes } from "./App";
import { FakeEventSource, http, installEventSource, mockFetch } from "./test/harness";

const PROTOCOL = {
  policies: ["OpenVLA", "OpenPiZero", "Octo", "MiniVLA", "SuSIE", "SuSIE_LL"],
  tasks: ["open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth"],
  starts_per_task: 50,
  total: 1500,
  mode: "synthetic",
  qualified: false,
  backend_revision: "irasim@c72b6da",
  judge_revision: "qwen2.5vl@cc59489",
  parity: "unverified",
  target: { episodes: 1500, seconds: 60, usd: 11.25, status: "unmeasured" },
  reference: {
    trials_per_cell: 50,
    arithmetic: {
      headline_openvla_close_drawer: {
        human_rate: 0.92,
        simpler_rate: 0.04,
        percentage_point_gap: 88,
      },
    },
  },
};

const GATES = {
  gates: [
    { id: "A", description: "Pinned model/backend conformance", status: "not_run", summary: "No measured GPU inference." },
    { id: "B", description: "Action grounding", status: "fail", summary: "Cosmos one-step profile fails causal feedback." },
    { id: "C", description: "Matched real starting states", status: "not_run", summary: "Sink and cloth assets unacquired." },
  ],
  qualified: false,
  reason: "Fixture runs cannot qualify.",
};

const SWEEPS = {
  status: "available",
  reason: "Development sweep only.",
  points: [
    {
      operating_point_id: "op-a",
      resolution: "256x256",
      denoise_steps: 30,
      chunk_partition: "4x16",
      batch_size: 4,
      fixed_task_horizon: 70,
      coverage: 0.96,
      gpu_seconds: 12.4,
      estimated_usd: 1.02,
      pairwise_order_agreement: 0.94,
      tolerances_met: true,
      qualification: "development_only",
    },
    {
      operating_point_id: "op-b",
      resolution: "128x128",
      denoise_steps: 8,
      chunk_partition: "1x16",
      batch_size: 16,
      fixed_task_horizon: 70,
      coverage: 0.71,
      gpu_seconds: 3.9,
      estimated_usd: 0.04,
      pairwise_order_agreement: 0.58,
      tolerances_met: false,
      qualification: "failed_tolerances",
    },
  ],
};

const ANALYSIS = {
  endpoint: { name: "observed_positive_lower_bound" },
  lineage_leakage: { status: "pass", reason: null },
  ledger_state: { active: false, nonterminal_status_counts: {}, run_ids: ["run-1"] },
  matrix: { policies: PROTOCOL.policies, tasks: PROTOCOL.tasks },
  cells: [
    {
      policy: "OpenVLA",
      task: "close_drawer",
      n: 50,
      valid: 44,
      successes: 31,
      coverage: 0.88,
      rate: 0.7045,
      positive_rate: 0.62,
      positive_wilson: [0.481, 0.744],
      missing_bounds: [0.62, 0.74],
      reference_rate: 0.92,
      reference_successes: 46,
      reference_n: 50,
      reference_wilson: [0.808, 0.972],
      simpler_rate: 0.04,
      missing_reason_counts: { invalid_world_video: 6 },
      scientific_status: "diagnostic",
    },
  ],
  rankings: [
    {
      scope: "macro",
      visual_order: ["OpenVLA"],
      ordering: "indeterminate",
      supported: false,
      reason: "Clustered starts lack a validated boundary-aware method.",
    },
  ],
  pairwise: [
    {
      scope: "close_drawer",
      policy_a: "OpenVLA",
      policy_b: "Octo",
      status: "computed",
      ordering: "indeterminate",
      supported: false,
      ordering_reason: "Envelopes overlap at the family alpha.",
      paired: { paired_n: 50, p_value: 0.04, family_alpha: 0.05 / 75 },
    },
  ],
  qualified: false,
};

const EPISODES = {
  episodes: [
    {
      episode_id: "ep-1",
      run_id: "run-1",
      policy: "OpenVLA",
      task: "close_drawer",
      status: "completed",
      frame_urls: ["run-1/ep-1/0.png", "run-1/ep-1/1.png"],
      certified_frame_count: 16,
      provenance: "live",
      resolution: "256x256",
    },
    {
      episode_id: "ep-hero",
      run_id: "run-1",
      policy: "OpenVLA",
      task: "open_drawer",
      status: "completed",
      frame_urls: ["run-1/ep-hero/0.png"],
      certified_frame_count: 16,
      provenance: "live",
      resolution: "480x480",
      presentation_track: "hero_480p",
    },
  ],
};

const SIXCLIP = {
  revealed: false,
  clips: Array.from({ length: 6 }, (_, index) => ({
    id: `clip-${index}`,
    url: `clips/${index}.mp4`,
    provenance_sealed: true,
  })),
  recorded_answers: null,
};

function installRoutes() {
  return mockFetch({
    "/api/health": { status: "ok", version: "0.1.0", qualified: false, available_backends: ["synthetic", "baseten"] },
    "/api/cloud-diagnostics/status": {
      configured: false,
      available: false,
      reason: "No cloud diagnostic is configured in this test control plane.",
      policy: "SuSIE_LL",
      qualified: false,
      fixture: {},
    },
    "/api/cloud-diagnostics": { requests: [] },
    "/api/protocol": PROTOCOL,
    "/api/gates": GATES,
    "/api/runs/run-1/episodes": EPISODES,
    "/api/runs/run-1/analysis": ANALYSIS,
    "/api/runs/run-1": { id: "run-1", status: "completed", total: 1500, completed: 1500, failed: 0, cancelled: 0, evaluable: 1320, successes: 640 },
    "/api/runs": { runs: [{ id: "run-1", status: "completed", total: 1500, completed: 1500, evaluable: 1320, successes: 640, created_at: "2026-09-21T09:00:00Z" }] },
    "/api/sweeps": SWEEPS,
    "/api/experiments": {
      source: "persisted_cluster_evidence",
      qualified: false,
      experiments: [
        {
          id: "cosmos/smoke",
          kind: "cosmos3_nano_diffusers_smoke",
          status: "completed",
          model: "nvidia/Cosmos3-Nano",
          stage: "world",
          frame_count: 17,
          latency_seconds: 4.52,
          model_load_seconds: 167.68,
          total_seconds: 172.2,
          gpu_peak_memory_bytes: 41_000_000_000,
          qualification: "not_qualified_by_smoke",
          timing_scope: "inference_excludes_model_load",
          outcome: "unknown",
          notes: ["16 actions produced 17 frames."],
          report_url: "cluster-evidence/cosmos.json",
          video_url: null,
        },
      ],
    },
    "/api/clips/sixclip": SIXCLIP,
    "/api/calledshot": {
      cell: "OpenVLA/close_drawer",
      human_rate: 0.92,
      human_n: 50,
      simpler_rate: 0.04,
      simpler_n: 50,
      plumb_estimate: null,
      plumb_status: "pending",
      prereg_uri: null,
      prereg_sha256: null,
    },
  });
}


/**
 * Pages mount under a router, so a test renders one route rather than the whole
 * document. `MemoryRouter` keeps navigation in memory; `AppDataProvider` is the
 * store every page reads from, and it holds the poll and the event stream, so
 * it wraps the router rather than sitting inside it.
 */
async function renderAt(route: string, heading: RegExp) {
  installEventSource();
  const routes = installRoutes();
  const view = render(
    <AppDataProvider>
      <MemoryRouter initialEntries={[route]}>
        <AppRoutes />
      </MemoryRouter>
    </AppDataProvider>,
  );
  await screen.findByRole("heading", { level: 1, name: heading });
  return { ...view, ...routes };
}

const overview = () => renderAt("/console", /Measure the ruler before trusting the ranking/i);
const live = async () => {
  const view = await renderAt("/live", /^Live run$/i);
  // Wait on something the *episodes* produce, not on protocol data. The wall's
  // tiles only exist once /api/runs/run-1/episodes has landed; waiting on a
  // protocol field instead let the test proceed with an empty wall whenever the
  // two responses resolved in the other order.
  await waitFor(() => expect(document.querySelectorAll(".rollout-tile").length).toBe(12));
  return view;
};

describe("Nightshift shell", () => {
  it("gives every page exactly one h1 and marks the current one", async () => {
    // Two h1s on one scrolling page read as two document titles. One page, one
    // title, and the sidebar says which page you are on without relying on the
    // lime rail to carry it.
    await overview();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    const current = document.querySelector('[aria-current="page"]');
    expect(current?.textContent).toMatch(/Overview/);
  });

  it("routes every sidebar link to a page that renders", async () => {
    for (const [route, title] of [
      ["/live", /^Live run$/i],
      ["/results", /^Results$/i],
      ["/evidence", /^Evidence$/i],
      ["/cost", /^Cost$/i],
      ["/clips", /^Clips$/i],
    ] as const) {
      const { unmount } = await renderAt(route, title);
      expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
      unmount();
    }
  });

  it("polls the control plane without a manual refresh", async () => {
    const { calls } = await overview();
    const paths = calls.map((call) => call.url);
    for (const path of ["/api/gates", "/api/sweeps", "/api/experiments", "/api/runs"]) {
      expect(paths.some((url) => url.startsWith(path))).toBe(true);
    }
  });

  it("links the third-party notices from the footer", async () => {
    await overview();
    const link = screen.getByRole("link", { name: /Third-party notices/i });
    expect(link).toHaveAttribute("href", "/THIRD-PARTY-NOTICES.md");
  });

  it("has no free-text input anywhere, including the surface that calls the world model", async () => {
    await overview();
    const assertNoFreeText = (where: string) => {
      expect(document.querySelectorAll("textarea"), where).toHaveLength(0);
      expect(document.querySelectorAll("[contenteditable]"), where).toHaveLength(0);
      const typed = [...document.querySelectorAll("input")].map((input) => input.type);
      expect(typed.filter((type) => type !== "range"), where).toEqual([]);
    };
    assertNoFreeText("overview");

    // Free-play is the only surface that dispatches to the world model directly.
    fireEvent.click(screen.getByRole("button", { name: /Drive the world model/i }));
    expect(await screen.findByRole("dialog", { name: /Free-play control/i })).toBeInTheDocument();
    assertNoFreeText("free-play dialog");
    expect(screen.getByText(/Fixed instruction, clamped actions, release to stop/i)).toBeInTheDocument();
  });
});

describe("Nightshift pages render their beats", () => {
  it("puts the claim, the called shot and the gates on the overview", async () => {
    await overview();
    expect(screen.getByRole("heading", { name: /The called shot/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Qualification gates/i })).toBeInTheDocument();
    expect(screen.getByText(/Not qualified/i)).toBeInTheDocument();
  });

  it("keeps the live page in the script's beat order", async () => {
    // SCRIPT.md runs wall -> controls -> burst -> dial and lands on the
    // scoreboard. The order is the demo's, so it is asserted, not assumed.
    await live();
    const headings = screen
      .getAllByRole("heading", { level: 2 })
      .map((node) => node.textContent ?? "");
    const indexOf = (pattern: RegExp) => headings.findIndex((text) => pattern.test(text));
    const wall = indexOf(/Rollout viewport/i);
    const burst = indexOf(/Full-matrix burst/i);
    const dial = indexOf(/Cost–fidelity operating point/i);
    const scores = indexOf(/^Scoreboard$/i);
    expect(wall).toBeGreaterThanOrEqual(0);
    expect(burst).toBeGreaterThan(wall);
    expect(dial).toBeGreaterThan(burst);
    expect(scores).toBeGreaterThan(dial);
  });

  it("keeps the synthetic task prompt separate from the disabled cloud diagnostic", async () => {
    await live();
    const prompt = screen.getByRole("region", { name: "Run a synthetic rehearsal task" });
    const cloud = screen.getByRole("heading", { name: "Cloud diagnostic" }).closest("section") as HTMLElement;
    expect(screen.getByLabelText(/Task instruction/i)).toBeInTheDocument();
    expect(cloud.compareDocumentPosition(prompt) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByRole("button", { name: "Run cloud model" })).toBeDisabled();
    expect(screen.getAllByText(/No cloud diagnostic is configured/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Synthetic rehearsal · no ML or robot inference/i)).toBeInTheDocument();
  });

  it("puts the scoreboard and the run ledger on results", async () => {
    await renderAt("/results", /^Results$/i);
    expect(await screen.findByRole("heading", { name: /^Scoreboard$/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Run ledger/i })).toBeInTheDocument();
  });

  it("puts the gates and the imported smoke report on evidence", async () => {
    await renderAt("/evidence", /^Evidence$/i);
    expect(screen.getByRole("heading", { name: /Qualification gates/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Real model smoke evidence/i })).toBeInTheDocument();
  });

  it("puts the dial on cost", async () => {
    await renderAt("/cost", /^Cost$/i);
    expect(screen.getByRole("heading", { name: /Cost–fidelity operating point/i })).toBeInTheDocument();
  });

  it("puts the six-clip test and the 480p track on clips", async () => {
    await renderAt("/clips", /^Clips$/i);
    expect(screen.getByRole("heading", { name: /Real or generated\?/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /480p presentation rollout/i })).toBeInTheDocument();
  });

  it("selects the hero episode from its explicit flag, not array position", async () => {
    await renderAt("/results", /^Results$/i);
    // ep-1 is episodes[0]; the hero is ep-hero, chosen by presentation_track.
    expect(await screen.findByText("ep-hero")).toBeInTheDocument();
    expect(screen.getByText("480x480")).toBeInTheDocument();
  });

  it("never uses a phrase from the never-say list", async () => {
    for (const [route, title] of [
      ["/", /Evaluate a robot policy/i],
      ["/console", /Measure the ruler/i],
      ["/live", /^Live run$/i],
      ["/results", /^Results$/i],
      ["/evidence", /^Evidence$/i],
      ["/cost", /^Cost$/i],
      ["/clips", /^Clips$/i],
    ] as const) {
      const { unmount } = await renderAt(route, title);
      const text = (document.body.textContent ?? "").toLowerCase();
      for (const phrase of [
        "SOTA",
        "solves",
        "matches human performance",
        "first ever",
        "time to first token",
        "tokens per second",
      ]) {
        expect(text, `${route} says "${phrase}"`).not.toContain(phrase.toLowerCase());
      }
      unmount();
    }
  });

  it("routes /review to the development review tool", async () => {
    // It arrived on main behind a `window.location.pathname` check in
    // main.tsx, which a client-side navigation never re-evaluates. As a route
    // it has to render from a link click and from a cold load alike.
    await renderAt("/review", /Review visible evidence, one opaque clip at a time/i);
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  });

  it("has no axe violations on any page", { timeout: 120_000 }, async () => {
    for (const [route, title] of [
      ["/", /Evaluate a robot policy/i],
      ["/console", /Measure the ruler/i],
      ["/live", /^Live run$/i],
      ["/results", /^Results$/i],
      ["/evidence", /^Evidence$/i],
      ["/cost", /^Cost$/i],
      ["/clips", /^Clips$/i],
    ] as const) {
      const { container, unmount } = await renderAt(route, title);
      let results!: axe.AxeResults;
      // Ported from main: axe drives its own async passes, and the display
      // clock ticks under it, so any state that lands mid-scan has to be
      // flushed inside act() or React warns and the run races the tick.
      await act(async () => {
        results = await axe.run(container, {
          // jsdom has no asset server; don't wait for image/font preloads. All
          // the semantic rules below still run.
          preload: false,
          rules: {
            // jsdom has no layout or stylesheet, so computed-colour checks
            // cannot run here. Contrast is verified against the palette in
            // styles.css instead.
            "color-contrast": { enabled: false },
            // The six clips are silent generated robot video with no speech
            // track; a captions file would have nothing to transcribe.
            "video-caption": { enabled: false },
          },
        });
      });
      expect(
        results.violations.map((v) => `${route} ${v.id}: ${v.help} (${v.nodes.length} node(s))`),
      ).toEqual([]);
      unmount();
    }
  });
});

describe("Nightshift live page interactions", () => {
  it("advances the Chain ladder when a real segment event lands", async () => {
    await live();
    const stages = screen.getByRole("heading", { name: /Chain stages/i }).closest(".panel") as HTMLElement;

    // Before any segment event the world stage waits; it does not assume.
    expect(within(stages).getByText(/No segment_completed event has arrived yet/)).toBeInTheDocument();
    // The other three stages have records but not their own fields.
    expect(within(stages).getAllByText("not reported")).toHaveLength(3);
    expect(within(stages).getByText(/action horizon not reported/)).toBeInTheDocument();
    expect(within(stages).getByText(/validity not reported/)).toBeInTheDocument();
    expect(within(stages).getByText(/no binary outcome reported/)).toBeInTheDocument();

    const source = FakeEventSource.latest();
    expect(source?.url).toBe("/api/runs/run-1/events");
    act(() =>
      source?.emit("segment_completed", {
        episode_id: "ep-1",
        run_id: "run-1",
        policy: "OpenVLA",
        task: "close_drawer",
        segment_index: 0,
        frame_urls: ["run-1/ep-1/2.png"],
        certified_frame_count: 14,
        provenance: "live",
      }),
    );

    // Both episode rows already carried 16 certified frames each before any event
    // arrived; this event adds its own 14. One event, forty-six frames — the two
    // counts keep separate subjects instead of being folded into one number.
    expect(
      within(stages).getByText("1 segment event · 46 certified frames accumulated"),
    ).toBeInTheDocument();
    expect(within(stages).getByRole("status")).toHaveTextContent("1 of 4 stages have reported");
  });

  it("reports the certified count as absent when a segment omits it", async () => {
    await live();
    const stages = screen.getByRole("heading", { name: /Chain stages/i }).closest(".panel") as HTMLElement;
    act(() =>
      FakeEventSource.latest()?.emit("segment_completed", {
        episode_id: "ep-1",
        run_id: "run-1",
        policy: "OpenVLA",
        task: "close_drawer",
        segment_index: 0,
        frame_urls: ["run-1/ep-1/2.png", "run-1/ep-1/3.png"],
      }),
    );
    expect(
      within(stages).getByText(
        "1 segment event · 32 certified frames accumulated · 1 without a certified count",
      ),
    ).toBeInTheDocument();
  });

  it("launches a benchmark task from a preset and claims a tile for it", async () => {
    const { calls } = await live();
    // The preset fills the input; submitting starts the rollout.
    fireEvent.click(screen.getByRole("button", { name: "Close the drawer" }));
    fireEvent.click(screen.getByRole("button", { name: "Run synthetic rehearsal" }));
    await waitFor(() => expect(calls.some((call) => call.method === "POST")).toBe(true));
    const body = JSON.parse(calls.filter((call) => call.method === "POST").at(-1)?.body ?? "{}");
    // A benchmark task goes through as a registry task id, not as free text.
    expect(body.tasks).toEqual(["close_drawer"]);
    expect(body.prompts ?? []).toEqual([]);
  });

  it("sends a typed task as a free-text prompt, not as a registry task", async () => {
    const { calls } = await live();
    fireEvent.change(screen.getByLabelText(/Task instruction/i), {
      target: { value: "Put the spoon in the drawer" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run synthetic rehearsal" }));
    await waitFor(() => expect(calls.some((call) => call.method === "POST")).toBe(true));
    const body = JSON.parse(calls.filter((call) => call.method === "POST").at(-1)?.body ?? "{}");
    expect(body.prompts).toEqual(["Put the spoon in the drawer"]);
    expect(body.tasks).toEqual([]);
  });

  it("says before submitting that canonical matching is not a synthetic measurement", async () => {
    await live();
    const input = screen.getByLabelText(/Task instruction/i);

    fireEvent.change(input, { target: { value: "Close the drawer" } });
    expect(screen.getByText(/Canonical instruction match\./)).toBeInTheDocument();
    expect(screen.getByText(/synthetic rehearsal is not a measurement/i)).toBeInTheDocument();

    // A near-miss is not a canonical registry instruction.
    fireEvent.change(input, { target: { value: "close the drawer" } });
    expect(screen.getByText(/Custom instruction\./)).toBeInTheDocument();
  });

  it("blows a wall tile up to full screen and gives the focus back on close", async () => {
    await live();
    const expand = screen.getByRole("button", {
      name: /Enlarge the persisted clip for OpenVLA on close_drawer/i,
    });
    expand.focus();
    fireEvent.click(expand);

    const dialog = screen.getByRole("dialog", { name: "OpenVLA · close_drawer" });
    expect(within(dialog).getByText("run-1")).toBeInTheDocument();
    expect(within(dialog).getByText("16 certified")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /Take the controls/i })).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "OpenVLA · close_drawer" })).not.toBeInTheDocument();
    expect(document.activeElement).toBe(expand);
  });

  it("hands a tile straight to free-play, which is the stage path", async () => {
    await live();
    fireEvent.click(
      screen.getByRole("button", { name: /Open free-play control for OpenVLA on close_drawer/i }),
    );
    const dialog = await screen.findByRole("dialog", { name: /Free-play control/i });
    expect(dialog).toHaveAccessibleName(/OpenVLA · close_drawer/);
  });
});

/**
 * Today's API sends no `frame_urls`, no `provenance`, no `presentation_track`, an
 * empty sweep list, null telemetry and no clip or called-shot route. The console
 * has to degrade by saying so, not by filling the gaps with zeros.
 */
describe("Nightshift against the current backend responses", () => {
  async function renderTodayAt(route: string, heading: RegExp) {
    installEventSource();
    const routes = mockFetch({
      "/api/health": { status: "ok", version: "0.1.0", qualified: false, available_backends: ["synthetic"] },
      "/api/cloud-diagnostics/status": {
        configured: false,
        available: false,
        reason: "No cloud diagnostic is configured.",
        policy: "SuSIE_LL",
        qualified: false,
        fixture: {},
      },
      "/api/cloud-diagnostics": { requests: [] },
      "/api/protocol": {
        policies: PROTOCOL.policies,
        tasks: PROTOCOL.tasks,
        starts_per_task: 50,
        total: 1500,
        reference: PROTOCOL.reference,
        target: { episodes: 1500, seconds: 60, usd: 11.25, status: "unmeasured" },
        mode: "synthetic",
        qualified: false,
      },
      "/api/gates": {
        gates: [{ id: "A", description: "Pinned model/backend conformance", status: "not_run", summary: "Pinned model/backend conformance" }],
        qualified: false,
        reason: "Fixture runs cannot qualify.",
      },
      "/api/runs": { runs: [] },
      "/api/sweeps": {
        points: [],
        status: "unavailable",
        reason: "No independent real-model cost-fidelity sweep has completed.",
      },
      "/api/experiments": { experiments: [], source: "persisted_cluster_evidence", qualified: false },
      "/api/clips/sixclip": http(404, { detail: "Not Found" }),
      "/api/calledshot": http(404, { detail: "Not Found" }),
    });
    const view = render(
      <AppDataProvider>
        <MemoryRouter initialEntries={[route]}>
          <AppRoutes />
        </MemoryRouter>
      </AppDataProvider>,
    );
    await screen.findByRole("heading", { level: 1, name: heading });
    return { ...view, ...routes };
  }

  it("does not report a load error when only the additive routes are missing", async () => {
    // The clip and called-shot routes 404 here; the core control plane did not
    // fail, so the console must not display a load error over the whole page.
    await renderTodayAt("/results", /^Results$/i);
    await waitFor(() => expect(screen.getByRole("heading", { name: /^Scoreboard$/i })).toBeInTheDocument());
    expect(document.querySelector(".error-banner")).toBeNull();
  });

  it("recovers the called shot from protocol.reference and marks it pending", async () => {
    await renderTodayAt("/console", /Measure the ruler/i);
    expect(await screen.findByText("92%")).toBeInTheDocument();
    expect(screen.getByText("4%")).toBeInTheDocument();
    expect(screen.getByText("pending")).toBeInTheDocument();
    expect(screen.getByText("No preregistration record was returned")).toBeInTheDocument();
  });

  it("explains the empty sweep instead of rendering an unreachable slider", async () => {
    await renderTodayAt("/cost", /^Cost$/i);
    expect(
      await screen.findByText(/No independent real-model cost-fidelity sweep has completed\./),
    ).toBeInTheDocument();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
  });

  it("shows no hero rollout rather than relabelling a 256p episode as 480p", async () => {
    await renderTodayAt("/clips", /^Clips$/i);
    expect(await screen.findByText(/No episode declares/)).toBeInTheDocument();
    expect(screen.getByText(/is not promoted here and relabelled 480p/)).toBeInTheDocument();
  });

  it("explains the missing six-clip panel rather than showing stand-in clips", async () => {
    await renderTodayAt("/clips", /^Clips$/i);
    expect(
      await screen.findByText(/No six-clip panel has been published by the API\./),
    ).toBeInTheDocument();
    expect(document.querySelectorAll("video")).toHaveLength(0);
  });

  it("invents no cost, queue depth or replica count", async () => {
    await renderTodayAt("/live", /^Live run$/i);
    await screen.findByRole("heading", { name: /Full-matrix burst/i });
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    expect(screen.getByText(/No replica counts have been reported/)).toBeInTheDocument();
    expect(screen.getByText(/No telemetry event has arrived yet\./)).toBeInTheDocument();
  });
});
