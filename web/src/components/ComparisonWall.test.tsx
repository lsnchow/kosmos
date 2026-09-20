import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import axe from "axe-core";
import { describe, expect, it, vi } from "vitest";
import { comparisonProblem, applyComparisonEvent, ComparisonWall } from "./ComparisonWall";
import { FakeEventSource, installEventSource, mockFetch } from "../test/harness";
import type { ComparisonDetail, ComparisonEvent, ComparisonReadiness, ManualSession } from "../lib/comparisons";

function frame(cellId: string, index: number, sha256 = `sha-${cellId}-${index}`) {
  return {
    event_id: `${cellId}-event-${index}`,
    segment_id: `${cellId}-segment-${index}`,
    frame_index: index,
    url: `comparisons/${cellId}/${index}.png`,
    sha256,
    state: { origin: "forecast", sha256: `state-${cellId}-${index}` },
  };
}

function detail(overrides: Partial<ComparisonDetail> = {}): ComparisonDetail {
  const policies: ComparisonDetail["policies"] = ["OpenVLA", "MiniVLA", "Octo-Small"];
  const seeds = [101, 102, 103, 104];
  const cells = policies.flatMap((policy) => seeds.map((seed) => {
    const id = `${policy.toLowerCase().replaceAll("-", "")}-${seed}`;
    const frames = Array.from({ length: 70 }, (_, index) => frame(id, index + 1));
    return {
      id,
      policy,
      seed,
      status: "completed" as const,
      attempt_id: `${id}-attempt`,
      action_count: 70,
      frame_count: 70,
      terminal_received: true,
      frames,
      latest_frame: frames.at(-1) ?? null,
      error: null,
    };
  }));
  return {
    schema: "kosmos-comparison-v1",
    scored: false,
    claim_tier: "preview",
    id: "comparison-1",
    status: "completed",
    created_at: "2026-09-19T00:00:00Z",
    updated_at: "2026-09-19T00:00:00Z",
    task: "close_drawer",
    task_instruction: "Close the drawer",
    horizon: 70,
    seeds,
    policies,
    start: {
      id: "bridge-close-drawer-start-1",
      png_url: "comparisons/start.png",
      sha256: "start-sha",
      state: { origin: "source_measured", sha256: "source-state-sha" },
    },
    world: {
      id: "cosmos-nano-480",
      profile_sha256: "world-profile-sha",
      manual: { action_rows: 16, structural_frames: 17, post_conditioning_frames: 16 },
    },
    reservation: { reserved_usd: 25, breakdown: [] },
    cells,
    scores: null,
    presentation: true,
    ...overrides,
  };
}

function readiness(overrides: Partial<ComparisonReadiness> = {}): ComparisonReadiness {
  return {
    schema: "kosmos-comparison-v1",
    scored: false,
    claim_tier: "preview",
    task: "close_drawer",
    task_instruction: "Close the drawer",
    available: true,
    mechanical: { status: "ready", reasons: [] },
    demo_quality: { status: "approved", reasons: [] },
    blockers: [],
    manifest_sha256: "manifest-sha",
    start: {
      id: "bridge-close-drawer-start-1",
      png_url: "comparisons/start.png",
      sha256: "start-sha",
      state: { origin: "source_measured", sha256: "source-state-sha" },
    },
    policies: [
      { id: "OpenVLA", identity_hashes: { adapter: "openvla-sha" } },
      { id: "MiniVLA", identity_hashes: { adapter: "minivla-sha" } },
      { id: "Octo-Small", identity_hashes: { adapter: "octo-sha" } },
    ],
    seeds: [101, 102, 103, 104],
    horizon: 70,
    world: {
      id: "cosmos-nano-480",
      profile_sha256: "world-profile-sha",
      manual: { action_rows: 16, structural_frames: 17, post_conditioning_frames: 16 },
    },
    budget: { cap_usd: 100, reserved_usd: 0, available_usd: 100 },
    quote: null,
    ...overrides,
  };
}

function manualSession(cellId: string): ManualSession {
  return {
    id: "manual-1",
    status: "ready",
    comparison_id: "comparison-1",
    cell_id: cellId,
    source: {
      event_id: `${cellId}-event-70`,
      url: `comparisons/${cellId}/70.png`,
      sha256: `sha-${cellId}-70`,
      state: { origin: "forecast", sha256: "manual-source-state" },
    },
    active_command: null,
    commands: [],
  };
}

function setDocumentHidden(hidden: boolean) {
  Object.defineProperty(document, "hidden", { configurable: true, value: hidden });
  act(() => document.dispatchEvent(new Event("visibilitychange")));
}

function installRoutes(options?: { presentation?: ComparisonDetail | null; readiness?: ComparisonReadiness }) {
  const promoted = options?.presentation === undefined ? detail() : options.presentation;
  const currentReadiness = options?.readiness ?? readiness();
  const cellId = promoted?.cells[0]?.id ?? "openvla-101";
  return mockFetch({
    "/api/comparisons/readiness": currentReadiness,
    "/api/comparisons/presentation": {
      schema: "kosmos-comparison-v1",
      scored: false,
      claim_tier: "preview",
      task: "close_drawer",
      task_instruction: "Close the drawer",
      presentation: promoted,
      status: promoted ? "available" : "unavailable",
      reason: promoted ? undefined : "No exact three-policy wall has been promoted.",
    },
    "/api/comparisons/comparison-1/manual-sessions": () => manualSession(cellId),
    "/api/manual-sessions/manual-1/commands": {
      schema: "kosmos-comparison-v1",
      scored: false,
      claim_tier: "preview",
      task: "close_drawer",
      task_instruction: "Close the drawer",
      command: {
        id: "command-1",
        status: "running",
        direction: "right",
        action_rows: 16,
        structural_frames: 17,
        post_conditioning_frames: 16,
        frame_count: 0,
        frames: [],
      },
    },
    "/api/comparisons/quote": {
      schema: "kosmos-comparison-v1",
      scored: false,
      claim_tier: "preview",
      task: "close_drawer",
      task_instruction: "Close the drawer",
      quote: { id: "quote-1", expires_at: "2026-09-20T00:00:00Z", manifest_sha256: "manifest-sha", reservation_usd: 25, breakdown: [] },
      readiness: currentReadiness,
    },
  });
}

describe("<ComparisonWall />", () => {
  it("uses only a promoted complete three-by-four wall and never starts work on load", async () => {
    installEventSource();
    const { calls } = installRoutes();
    render(<ComparisonWall />);

    await screen.findByText("Precomputed matched Baseten set");
    expect(screen.getByText("Close the drawer")).toBeInTheDocument();
    expect(screen.getAllByText("Outcome: not scored")).toHaveLength(12);
    expect(screen.getAllByRole("columnheader")).toHaveLength(4);
    expect(screen.getAllByRole("rowheader")).toHaveLength(4);
    expect(screen.queryByText(/winner|success rate|scoreboard/i)).not.toBeInTheDocument();
    expect(calls.every((call) => call.method === "GET")).toBe(true);
    await waitFor(() =>
      expect(FakeEventSource.latest()?.url).toBe("/api/comparisons/comparison-1/events?after=0"),
    );
  });

  it("keeps the control branch inert until the explicit tile click and submits one key command", async () => {
    installEventSource();
    const { calls } = installRoutes();
    render(<ComparisonWall />);
    await screen.findByText("Precomputed matched Baseten set");

    const firstTile = screen.getByLabelText(/OpenVLA, seed 101/i);
    fireEvent.click(within(firstTile).getByRole("button", { name: "Take control" }));
    await screen.findByRole("dialog", { name: "Keyboard steering" });
    expect(calls.filter((call) => call.url.includes("/commands") && call.method === "POST")).toHaveLength(0);

    const surface = screen.getByLabelText(/Keyboard steering surface/i);
    fireEvent.keyDown(surface, { key: "ArrowRight", repeat: false });
    fireEvent.keyDown(surface, { key: "ArrowRight", repeat: true });
    fireEvent.keyUp(surface, { key: "ArrowRight" });
    await waitFor(() => expect(calls.filter((call) => call.url.includes("/commands") && call.method === "POST")).toHaveLength(1));
    const command = calls.find((call) => call.url.includes("/commands") && call.method === "POST");
    expect(JSON.parse(command?.body ?? "{}")).toMatchObject({ direction: "right" });
    expect(JSON.parse(command?.body ?? "{}")).not.toHaveProperty("action");
  });

  it("adopts the terminal segment’s final committed frame as the next branch source", async () => {
    installEventSource();
    installRoutes();
    render(<ComparisonWall />);
    await screen.findByText("Precomputed matched Baseten set");
    fireEvent.click(within(screen.getByLabelText(/OpenVLA, seed 101/i)).getByRole("button", { name: "Take control" }));
    await screen.findByRole("dialog", { name: "Keyboard steering" });
    fireEvent.click(screen.getByRole("button", { name: "Move right" }));
    await screen.findByText(/Generating right/i);
    act(() => FakeEventSource.latest()?.emit("comparison", {
      sequence: 1,
      type: "manual_terminal",
      comparison_id: "comparison-1",
      cell_id: "openvla-101",
      payload: {
        session_id: "manual-1",
        command_id: "command-1",
        status: "completed",
        frame_count: 16,
        next_source: {
          event_id: "manual-final-event",
          url: "comparisons/manual/final.png",
          sha256: "manual-final-sha",
          state: { origin: "forecast", sha256: "manual-final-state" },
        },
      },
    }));
    expect(await screen.findByText("manual-final-sha")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Move up" })).toBeEnabled();
  });

  it("holds the last committed manual frame and admits no input while the page is hidden", async () => {
    installEventSource();
    const { calls } = installRoutes();
    render(<ComparisonWall />);
    await screen.findByText("Precomputed matched Baseten set");
    fireEvent.click(within(screen.getByLabelText(/OpenVLA, seed 101/i)).getByRole("button", { name: "Take control" }));
    await screen.findByRole("dialog", { name: "Keyboard steering" });
    fireEvent.click(screen.getByRole("button", { name: "Move right" }));
    await waitFor(() => expect(calls.filter((call) => call.url.includes("/commands") && call.method === "POST")).toHaveLength(1));
    act(() => {
      FakeEventSource.latest()?.emit("comparison", {
        sequence: 1,
        type: "manual_frame",
        comparison_id: "comparison-1",
        cell_id: "openvla-101",
        payload: { session_id: "manual-1", command_id: "command-1", frame_index: 1, url: "manual/1.png", sha256: "manual-1" },
      });
      FakeEventSource.latest()?.emit("comparison", {
        sequence: 2,
        type: "manual_frame",
        comparison_id: "comparison-1",
        cell_id: "openvla-101",
        payload: { session_id: "manual-1", command_id: "command-1", frame_index: 2, url: "manual/2.png", sha256: "manual-2" },
      });
    });
    try {
      setDocumentHidden(true);
      expect(await screen.findByAltText("Generated manual branch frame 2")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Move up" })).toBeDisabled();
      fireEvent.keyDown(screen.getByLabelText(/Keyboard steering surface/i), { key: "ArrowUp" });
      expect(calls.filter((call) => call.url.includes("/commands") && call.method === "POST")).toHaveLength(1);
    } finally {
      setDocumentHidden(false);
    }
  });

  it("latches two distinct controls from the same React turn to one command", async () => {
    installEventSource();
    const { calls } = installRoutes();
    render(<ComparisonWall />);
    await screen.findByText("Precomputed matched Baseten set");
    fireEvent.click(within(screen.getByLabelText(/OpenVLA, seed 101/i)).getByRole("button", { name: "Take control" }));
    await screen.findByRole("dialog", { name: "Keyboard steering" });
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Move up" }));
      fireEvent.click(screen.getByRole("button", { name: "Move right" }));
    });
    await waitFor(() => expect(calls.filter((call) => call.url.includes("/commands") && call.method === "POST")).toHaveLength(1));
  });

  it("shows blockers and does not construct twelve substitute clips when no promoted set exists", async () => {
    installEventSource();
    const blocked = readiness({
      available: false,
      mechanical: { status: "blocked", reasons: ["openvla_source_conformance_missing"] },
      demo_quality: { status: "blocked", reasons: ["full_length_visual_review_missing"] },
      blockers: ["three_policy_world_profile_not_ready"],
      start: null,
      world: null,
    });
    const { calls } = installRoutes({ presentation: null, readiness: blocked });
    render(<ComparisonWall />);
    expect(await screen.findByText(/No promoted 12-cell set is available/i)).toBeInTheDocument();
    expect(screen.getByText("three_policy_world_profile_not_ready")).toBeInTheDocument();
    expect(screen.queryByText("Outcome: not scored")).not.toBeInTheDocument();
    expect(calls.every((call) => call.method === "GET")).toBe(true);
  });

  it("names an IRASim 15-to-16 fallback as unavailable instead of adapting it into steering", async () => {
    installEventSource();
    const irasimWorld = {
      id: "irasim-native",
      profile_sha256: "irasim-profile-sha",
      manual: { action_rows: 15, structural_frames: 16, post_conditioning_frames: 15 },
    };
    installRoutes({ presentation: detail({ world: irasimWorld }), readiness: readiness({ world: irasimWorld }) });
    render(<ComparisonWall />);
    expect(await screen.findByText(/IRASim’s native 15 actions → 16 structural frames contract/i)).toBeInTheDocument();
    expect(within(screen.getByLabelText(/OpenVLA, seed 101/i)).getByRole("button", { name: "Take control" })).toBeDisabled();
  });

  it("keeps two same-pixel committed frames when their durable event ids differ", () => {
    const source = detail({ presentation: false, status: "running" });
    source.cells = source.cells.map((cell, index) => index === 0
      ? { ...cell, status: "running", terminal_received: false, action_count: 1, frame_count: 0, frames: [], latest_frame: null }
      : cell);
    const first: ComparisonEvent = {
      sequence: 1,
      type: "frame",
      comparison_id: source.id,
      cell_id: source.cells[0].id,
      payload: { event_id: "frame-a", segment_id: "segment-a", frame_index: 1, url: "same.png", sha256: "same-pixels" },
    };
    const second: ComparisonEvent = {
      ...first,
      sequence: 2,
      payload: { event_id: "frame-b", segment_id: "segment-b", frame_index: 2, url: "same.png", sha256: "same-pixels" },
    };
    const applied = applyComparisonEvent(applyComparisonEvent(source, first), second);
    expect(applied.cells[0].frames.map((item) => item.event_id)).toEqual(["frame-a", "frame-b"]);
  });

  it("rejects a promoted partial wall before it can be labelled precomputed", () => {
    const partial = detail();
    partial.cells[0] = { ...partial.cells[0], action_count: 69 };
    expect(comparisonProblem(partial, true)).toMatch(/70-action/i);
  });

  it("keeps the controlled wall and its steering dialog free of basic accessibility violations", async () => {
    document.title = "Kosmos comparison test";
    vi.stubGlobal("matchMedia", () => ({
      matches: true,
      media: "(prefers-reduced-motion: reduce)",
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    }));
    installEventSource();
    installRoutes();
    render(<ComparisonWall />);
    await screen.findByText("Precomputed matched Baseten set");
    fireEvent.click(within(screen.getByLabelText(/OpenVLA, seed 101/i)).getByRole("button", { name: "Take control" }));
    await screen.findByRole("dialog", { name: "Keyboard steering" });
    let result!: axe.AxeResults;
    await act(async () => {
      result = await axe.run(document, { rules: { "color-contrast": { enabled: false } } });
    });
    expect(result.violations).toEqual([]);
  });
});
