import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import axe from "axe-core";
import { describe, expect, it } from "vitest";
import App from "./App";
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

async function renderConsole() {
  installEventSource();
  const routes = installRoutes();
  const view = render(<App />);
  await screen.findByRole("heading", { name: /Measure the ruler before trusting the ranking/i });
  await waitFor(() => expect(screen.getByText("irasim@c72b6da")).toBeInTheDocument());
  return { ...view, ...routes };
}

describe("PLUMB console", () => {
  it("renders every demo beat", async () => {
    await renderConsole();
    expect(screen.getByRole("heading", { name: /Real or generated\?/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /The called shot/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Rollout viewport/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Full-matrix burst/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Cost–fidelity operating point/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /^Scoreboard$/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /480p presentation rollout/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Qualification gates/i })).toBeInTheDocument();
  });

  it("reports the baseten backend when health advertises it", async () => {
    await renderConsole();
    await waitFor(() =>
      expect(document.querySelector(".api-health")?.textContent ?? "").toMatch(
        /synthetic, baseten/,
      ),
    );
  });

  it("has no axe violations on the main view", { timeout: 60_000 }, async () => {
    const { container } = await renderConsole();
    const results = await axe.run(container, {
      rules: {
        // jsdom has no layout or stylesheet, so computed-colour checks cannot run
        // here. Contrast is verified against the palette in styles.css instead.
        "color-contrast": { enabled: false },
        // The six clips are silent generated robot video with no speech track;
        // a captions file would have nothing to transcribe.
        "video-caption": { enabled: false },
      },
    });
    const summary = results.violations.map(
      (violation) => `${violation.id}: ${violation.help} (${violation.nodes.length} node(s))`,
    );
    expect(summary).toEqual([]);
  });

  it("never uses a phrase from the never-say list", async () => {
    await renderConsole();
    const text = document.body.textContent ?? "";
    for (const phrase of [
      "SOTA",
      "solves",
      "matches human performance",
      "first ever",
      "time to first token",
      "tokens per second",
    ]) {
      expect(text.toLowerCase()).not.toContain(phrase.toLowerCase());
    }
  });

  it("selects the hero episode from its explicit flag, not array position", async () => {
    await renderConsole();
    // ep-1 is episodes[0]; the hero is ep-hero, chosen by presentation_track.
    expect(await screen.findByText("ep-hero")).toBeInTheDocument();
    expect(screen.getByText("480x480")).toBeInTheDocument();
  });

  it("polls the control plane without a manual refresh", async () => {
    const { calls } = await renderConsole();
    const paths = calls.map((call) => call.url);
    for (const path of ["/api/gates", "/api/sweeps", "/api/experiments", "/api/runs"]) {
      expect(paths.some((url) => url.startsWith(path))).toBe(true);
    }
  });
});

/**
 * Today's API sends no `frame_urls`, no `provenance`, no `presentation_track`, an
 * empty sweep list, null telemetry and no clip or called-shot route. The console
 * has to degrade by saying so, not by filling the gaps with zeros.
 */
describe("PLUMB console against the current backend responses", () => {
  async function renderAgainstToday() {
    installEventSource();
    const routes = mockFetch({
      "/api/health": { status: "ok", version: "0.1.0", qualified: false, available_backends: ["synthetic"] },
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
    const view = render(<App />);
    await screen.findByRole("heading", { name: /Measure the ruler/i });
    return { ...view, ...routes };
  }

  it("does not report a load error when only the additive routes are missing", async () => {
    await renderAgainstToday();
    // The clip and called-shot routes 404 here; the core control plane did not
    // fail, so the console must not display a load error over the whole page.
    await waitFor(() => expect(screen.getByRole("heading", { name: /^Scoreboard$/i })).toBeInTheDocument());
    expect(document.querySelector(".error-banner")).toBeNull();
    expect(screen.queryByText(/No route double/)).not.toBeInTheDocument();
  });

  it("recovers the called shot from protocol.reference and marks it pending", async () => {
    await renderAgainstToday();
    expect(await screen.findByText("92%")).toBeInTheDocument();
    expect(screen.getByText("4%")).toBeInTheDocument();
    expect(screen.getByText("pending")).toBeInTheDocument();
    expect(screen.getByText("No preregistration record was returned")).toBeInTheDocument();
  });

  it("explains the empty sweep instead of rendering an unreachable slider", async () => {
    await renderAgainstToday();
    expect(
      await screen.findByText(/No independent real-model cost-fidelity sweep has completed\./),
    ).toBeInTheDocument();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
  });

  it("shows no hero rollout rather than relabelling a 256p episode as 480p", async () => {
    await renderAgainstToday();
    expect(
      await screen.findByText(/No episode declares/),
    ).toBeInTheDocument();
    expect(screen.getByText(/is not promoted here and relabelled 480p/)).toBeInTheDocument();
  });

  it("explains the missing six-clip panel rather than showing stand-in clips", async () => {
    await renderAgainstToday();
    expect(
      await screen.findByText(/No six-clip panel has been published by the API\./),
    ).toBeInTheDocument();
    expect(document.querySelectorAll("video")).toHaveLength(0);
  });

  it("invents no cost, queue depth or replica count", async () => {
    await renderAgainstToday();
    await screen.findByRole("heading", { name: /Full-matrix burst/i });
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    expect(screen.getByText(/No replica counts have been reported/)).toBeInTheDocument();
    expect(screen.getByText(/No telemetry event has arrived yet\./)).toBeInTheDocument();
  });
});

/**
 * The landing page and the console share one document: the landing is the first
 * screen, the console sits below it, and every in-page anchor the stage sequence
 * uses still resolves. Nothing routes, so nothing can unmount mid-pitch.
 */
describe("PLUMB landing page over the console", () => {
  it("opens on the wordmark with the console still in the document", async () => {
    await renderConsole();
    expect(screen.getByRole("heading", { level: 1, name: "PLUMB" })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /Measure the ruler before trusting the ranking/i }),
    ).toBeInTheDocument();
    // One main landmark, with both screens inside it.
    expect(screen.getAllByRole("main")).toHaveLength(1);
    expect(document.getElementById("console")).not.toBeNull();
  });

  it("resolves every landing destination to a console region that exists", async () => {
    await renderConsole();
    for (const anchor of ["console", "rollouts", "gates"]) {
      expect(document.getElementById(anchor), anchor).not.toBeNull();
    }
    for (const region of ["#stages", "#sixclip", "#calledshot", "#rollouts", "#burst", "#sweep", "#scoreboard", "#gates"]) {
      const link = document.querySelector(`.topbar nav a[href="${region}"]`);
      expect(link, region).not.toBeNull();
      expect(document.getElementById(region.slice(1)), region).not.toBeNull();
    }
  });

  it("has no free-text input anywhere, including the surface that calls the world model", async () => {
    await renderConsole();
    const assertNoFreeText = (where: string) => {
      expect(document.querySelectorAll("textarea"), where).toHaveLength(0);
      expect(document.querySelectorAll("[contenteditable]"), where).toHaveLength(0);
      const typed = [...document.querySelectorAll("input")].map((input) => input.type);
      expect(typed.filter((type) => type !== "range"), where).toEqual([]);
    };
    assertNoFreeText("landing and console");

    // Free-play is the only surface that dispatches to the world model directly.
    fireEvent.click(screen.getByRole("button", { name: /Drive the world model/i }));
    expect(await screen.findByRole("dialog", { name: /Free-play control/i })).toBeInTheDocument();
    assertNoFreeText("free-play dialog");
    expect(screen.getByText(/Fixed instruction, clamped actions, release to stop/i)).toBeInTheDocument();
  });

  it("advances the Chain ladder when a real segment event lands", async () => {
    await renderConsole();
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
    await renderConsole();
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

  it("scopes the viewport from a task chip without dropping a slot", async () => {
    await renderConsole();
    fireEvent.click(screen.getByRole("button", { name: "Close the drawer" }));
    expect(screen.getByText(/the rest dimmed rather than dropped/i)).toBeInTheDocument();
    expect(screen.getByText(/Scoped to “Close the drawer”/)).toBeInTheDocument();
    expect(document.querySelectorAll(".rollout-tile")).toHaveLength(12);
    expect(
      screen.getByLabelText(/Viewport slot #02: OpenVLA on open_drawer, outside the selected task scope/i),
    ).toBeInTheDocument();
  });

  it("blows a wall tile up to full screen and gives the focus back on close", async () => {
    await renderConsole();
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
    await renderConsole();
    fireEvent.click(
      screen.getByRole("button", { name: /Open free-play control for OpenVLA on close_drawer/i }),
    );
    const dialog = await screen.findByRole("dialog", { name: /Free-play control/i });
    expect(dialog).toHaveAccessibleName(/OpenVLA · close_drawer/);
  });

  it("links the third-party notices from both footers", async () => {
    await renderConsole();
    const links = screen.getAllByRole("link", { name: /Third-party notices/i });
    expect(links.length).toBeGreaterThanOrEqual(2);
    for (const link of links) expect(link).toHaveAttribute("href", "/THIRD-PARTY-NOTICES.md");
  });
});
