import { act, render, screen, waitFor } from "@testing-library/react";
import axe from "axe-core";
import { describe, expect, it } from "vitest";
import App from "./App";
import { http, installEventSource, mockFetch } from "./test/harness";

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
    expect(screen.getByRole("link", { name: "Development review" })).toHaveAttribute("href", "/review");
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
    let results!: axe.AxeResults;
    await act(async () => {
      results = await axe.run(container, {
      // jsdom has no asset server; don't wait for image/font preloads. All
      // semantic rules below still run, with clock updates inside React act.
      preload: false,
      rules: {
        // jsdom has no layout or stylesheet, so computed-colour checks cannot run
        // here. Contrast is verified against the palette in styles.css instead.
        "color-contrast": { enabled: false },
        // The six clips are silent generated robot video with no speech track;
        // a captions file would have nothing to transcribe.
        "video-caption": { enabled: false },
      },
      });
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
