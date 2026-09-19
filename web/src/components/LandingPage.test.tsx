import { fireEvent, render, screen, within } from "@testing-library/react";
import axe from "axe-core";
import { describe, expect, it, vi } from "vitest";
import type { Episode, Gate, Run } from "../lib/api";
import { BENCHMARK_TASKS } from "../lib/tasks";
import { LandingPage, railCards, resolveBackdrop } from "./LandingPage";

/**
 * The five strings the protocol registry freezes, spelled out here rather than
 * imported, so a paraphrase in `lib/tasks.ts` fails this file instead of passing
 * it. The lowercase `fold` is deliberate: the spec calls it out twice.
 */
const VERBATIM_TASKS = [
  "Close the drawer",
  "Open the drawer",
  "Put the eggplant in the yellow basket",
  "Put the eggplant in the blue sink",
  "fold the cloth from top right to bottom left",
];

const GATES: Gate[] = [
  { id: "A", name: "Gate A", status: "not_run", reason: "No measured GPU inference." },
  { id: "B", name: "Gate B", status: "fail", reason: "Cosmos one-step profile fails causal feedback." },
];

const RUNS: Run[] = [{ id: "run-1", status: "completed", total: 1500 }];

const EPISODES: Episode[] = [
  {
    episode_id: "ep-1",
    run_id: "run-1",
    policy: "OpenVLA",
    task: "close_drawer",
    status: "completed",
    frame_urls: ["run-1/ep-1/0.png", "run-1/ep-1/1.png"],
    certified_frame_count: 14,
    provenance: "live",
  },
  {
    episode_id: "ep-2",
    run_id: "run-1",
    policy: "Octo",
    task: "open_drawer",
    status: "completed",
    frame_urls: ["run-1/ep-2/0.png"],
    provenance: "cached",
  },
];

const HERO: Episode = {
  episode_id: "ep-hero",
  run_id: "run-1",
  policy: "OpenVLA",
  task: "open_drawer",
  presentation_track: "hero_480p",
  frame_urls: ["run-1/ep-hero/0.png", "run-1/ep-hero/1.png"],
  resolution: "480x480",
};

function stubMotionPreference(reduce: boolean) {
  vi.stubGlobal(
    "matchMedia",
    ((query: string) => ({
      matches: reduce && query.includes("prefers-reduced-motion"),
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    })) as typeof window.matchMedia,
  );
}

function renderLanding(
  overrides: Partial<Parameters<typeof LandingPage>[0]> = {},
  options: { reducedMotion?: boolean } = {},
) {
  stubMotionPreference(options.reducedMotion ?? false);
  const onScopeTask = vi.fn();
  const onEnterConsole = vi.fn();
  const onOpenFreeplay = vi.fn();
  const view = render(
    <LandingPage
      runs={RUNS}
      episodes={EPISODES}
      gates={GATES}
      protocol={{ mode: "synthetic" }}
      onScopeTask={onScopeTask}
      onEnterConsole={onEnterConsole}
      onOpenFreeplay={onOpenFreeplay}
      {...overrides}
    />,
  );
  return { ...view, onScopeTask, onEnterConsole, onOpenFreeplay };
}

describe("<LandingPage /> wordmark and positioning", () => {
  it("leads with the wordmark and one positioning line", () => {
    renderLanding();
    expect(screen.getByRole("heading", { level: 1, name: "Nightshift" })).toBeInTheDocument();
    expect(
      screen.getByText(
        /Point us at a policy endpoint and get a ranked report in twenty minutes for a few dollars/i,
      ),
    ).toBeInTheDocument();
  });

  it("marks the positioning line as a claim rather than a receipt", () => {
    renderLanding();
    expect(screen.getByText(/That is the product claim, not a receipt/i)).toBeInTheDocument();
    expect(screen.getByText(/no result here is qualified yet/i)).toBeInTheDocument();
  });

  it("links the third-party notices from the footer", () => {
    renderLanding();
    const link = screen.getByRole("link", { name: /Third-party notices/i });
    expect(link).toHaveAttribute("href", "/THIRD-PARTY-NOTICES.md");
  });

  it("uses no phrase from the never-say list", () => {
    renderLanding();
    const text = (document.body.textContent ?? "").toLowerCase();
    for (const phrase of [
      "sota",
      "solves",
      "matches human performance",
      "first ever",
      "time to first token",
      "tokens per second",
    ]) {
      expect(text).not.toContain(phrase);
    }
  });
});

describe("<LandingPage /> task chips", () => {
  it("carries the five registry strings verbatim, including the lowercase fold", () => {
    renderLanding();
    for (const instruction of VERBATIM_TASKS) {
      expect(screen.getByRole("button", { name: instruction })).toBeInTheDocument();
    }
    const fold = screen.getByRole("button", { name: /fold the cloth/ });
    expect(fold.textContent).toBe("fold the cloth from top right to bottom left");
    expect(fold.textContent).not.toBe("Fold the cloth from top right to bottom left");
  });

  it("keeps the chip list to exactly the five registry tasks", () => {
    renderLanding();
    const chips = screen
      .getAllByRole("button")
      .filter((button) => button.classList.contains("chip"))
      .map((button) => button.textContent);
    expect(chips).toEqual(VERBATIM_TASKS);
    expect(BENCHMARK_TASKS.map((task) => task.instruction)).toEqual(VERBATIM_TASKS);
  });

  it("offers no text field anywhere, so nothing typed can reach a model", () => {
    renderLanding();
    expect(document.querySelectorAll("textarea")).toHaveLength(0);
    expect(document.querySelectorAll("[contenteditable]")).toHaveLength(0);
    expect(document.querySelectorAll('input:not([type="range"])')).toHaveLength(0);
  });

  it("scopes rather than submits: selecting a chip issues no request", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const { onScopeTask } = renderLanding();
    fireEvent.click(screen.getByRole("button", { name: "Close the drawer" }));
    expect(onScopeTask).toHaveBeenCalledWith("close_drawer");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("marks the scoped chip pressed and offers to clear it", () => {
    const { onScopeTask } = renderLanding({ scopedTask: "fold_cloth" });
    expect(screen.getByRole("button", { name: /fold the cloth/ })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "Close the drawer" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(screen.getByText("fold_cloth")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear scope" }));
    expect(onScopeTask).toHaveBeenCalledWith(undefined);
  });

  it("says what the viewport shows when nothing is scoped", () => {
    renderLanding();
    expect(screen.getByText(/No task selected/i)).toBeInTheDocument();
  });
});

describe("<LandingPage /> example rail", () => {
  it("builds three cards at most, from persisted episodes only", () => {
    const many = Array.from({ length: 5 }, (_, index) => ({
      ...EPISODES[0],
      episode_id: `ep-${index}`,
    }));
    expect(railCards(many)).toHaveLength(3);
    expect(railCards([{ episode_id: "no-media", policy: "OpenVLA", task: "to_sink" }])).toEqual([]);
  });

  it("shows one card per persisted episode, each with its own provenance badge", () => {
    renderLanding();
    fireEvent.click(screen.getByRole("tab", { name: "Gallery" }));
    const cards = screen.getAllByRole("listitem");
    expect(cards).toHaveLength(2);
    expect(within(cards[0]).getByText("live")).toBeInTheDocument();
    expect(within(cards[1]).getByText("cached")).toBeInTheDocument();
  });

  it("labels a card whose provenance was never reported rather than assuming live", () => {
    renderLanding({ episodes: [{ ...EPISODES[0], provenance: undefined }] });
    fireEvent.click(screen.getByRole("tab", { name: "Gallery" }));
    expect(screen.getByText("no provenance")).toBeInTheDocument();
    expect(screen.queryByText("live")).not.toBeInTheDocument();
  });

  it("explains an empty rail with no runs, and shows no imagery at all", () => {
    renderLanding({ runs: [], episodes: [] });
    fireEvent.click(screen.getByRole("tab", { name: "Gallery" }));
    expect(screen.getByText(/No run has been persisted, so there is nothing to show here/i)).toBeInTheDocument();
    expect(screen.getByText(/never stands in illustrative footage/i)).toBeInTheDocument();
    expect(document.querySelectorAll("img")).toHaveLength(0);
    expect(document.querySelectorAll("video")).toHaveLength(0);
  });

  it("distinguishes 'no runs' from 'runs with no persisted frames'", () => {
    renderLanding({ runs: RUNS, episodes: [{ episode_id: "bare", policy: "Octo", task: "to_sink" }] });
    fireEvent.click(screen.getByRole("tab", { name: "Gallery" }));
    expect(
      screen.getByText(/1 persisted run, but no episode of the selected run has persisted frames yet/i),
    ).toBeInTheDocument();
  });

  it("opens a card full screen in a modal dialog", () => {
    renderLanding();
    fireEvent.click(screen.getByRole("tab", { name: "Gallery" }));
    fireEvent.click(screen.getAllByRole("button", { name: /OpenVLA/ })[0]);
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAccessibleName("OpenVLA · close_drawer");
    expect(within(dialog).getByText("14 certified frames")).toBeInTheDocument();
    expect(within(dialog).getByText("ep-1")).toBeInTheDocument();
  });

  it("does not print a frame count it was not given", () => {
    renderLanding();
    fireEvent.click(screen.getByRole("tab", { name: "Gallery" }));
    fireEvent.click(screen.getAllByRole("button", { name: /Octo/ })[0]);
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("1 frames · certified count not reported")).toBeInTheDocument();
  });
});

describe("<LandingPage /> background", () => {
  it("falls back to a flat gradient when no episode declares the hero track", () => {
    expect(resolveBackdrop([], { reducedMotion: false })).toEqual({
      kind: "gradient",
      label: 'no episode declares presentation_track "hero_480p" · flat background',
    });
  });

  it("plays the hero rollout's frames when there is one", () => {
    const backdrop = resolveBackdrop([HERO], { reducedMotion: false });
    expect(backdrop.kind).toBe("frames");
    expect(backdrop.label).toBe("480p hero rollout · 2 persisted frames");
  });

  it("prefers a persisted video over the frame flipbook", () => {
    const backdrop = resolveBackdrop([{ ...HERO, video_url: "run-1/ep-hero/clip.mp4" }], {
      reducedMotion: false,
    });
    expect(backdrop).toMatchObject({ kind: "video", src: "/api/artifacts/run-1/ep-hero/clip.mp4" });
  });

  it("never treats a non-video artifact as a video source", () => {
    const backdrop = resolveBackdrop([{ ...HERO, frame_urls: [], video_url: "run-1/ep-hero/0.png" }], {
      reducedMotion: false,
    });
    expect(backdrop.kind).toBe("still");
  });

  it("falls back to a still when reduced motion is requested", () => {
    const backdrop = resolveBackdrop([{ ...HERO, video_url: "run-1/ep-hero/clip.mp4" }], {
      reducedMotion: true,
    });
    expect(backdrop).toMatchObject({
      kind: "still",
      label: "480p hero rollout · still frame, reduced motion",
    });
  });

  it("renders no video element at all under prefers-reduced-motion", () => {
    renderLanding({ episodes: [{ ...HERO, video_url: "run-1/ep-hero/clip.mp4" }] }, { reducedMotion: true });
    expect(document.querySelectorAll("video")).toHaveLength(0);
    expect(document.querySelector(".landing-backdrop img")).not.toBeNull();
    expect(screen.getByText(/still frame, reduced motion/i)).toBeInTheDocument();
  });

  it("plays the video when motion is allowed", () => {
    renderLanding({ episodes: [{ ...HERO, video_url: "run-1/ep-hero/clip.mp4" }] });
    const video = document.querySelector("video");
    expect(video).not.toBeNull();
    expect(video).toHaveAttribute("src", "/api/artifacts/run-1/ep-hero/clip.mp4");
  });

  it("replaces media that failed to load with the gradient, never a broken frame", () => {
    renderLanding({ episodes: [{ ...HERO, video_url: "run-1/ep-hero/clip.mp4" }] });
    fireEvent.error(document.querySelector("video") as HTMLVideoElement);
    expect(document.querySelectorAll("video")).toHaveLength(0);
    expect(document.querySelector(".landing-backdrop-gradient")).not.toBeNull();
    expect(screen.getByText(/media did not load, flat background/i)).toBeInTheDocument();
  });

  it("always attributes whatever is behind the text", () => {
    renderLanding();
    expect(screen.getByText(/^Background:/)).toBeInTheDocument();
  });
});

describe("<LandingPage /> navigation", () => {
  it("centres a three-tab nav over the console, the gallery and the gates", () => {
    renderLanding();
    const tablist = screen.getByRole("tablist", { name: /Landing sections/i });
    expect(tablist.className).toContain("tablist-center");
    expect(within(tablist).getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "Console",
      "Gallery",
      "Gates",
    ]);
  });

  it("routes each panel into the matching console region", () => {
    const { onEnterConsole } = renderLanding();
    fireEvent.click(screen.getByRole("button", { name: /Open the console/i }));
    expect(onEnterConsole).toHaveBeenCalledWith("console");

    fireEvent.click(screen.getByRole("tab", { name: "Gallery" }));
    fireEvent.click(screen.getByRole("button", { name: /Open the rollout viewport/i }));
    expect(onEnterConsole).toHaveBeenCalledWith("rollouts");

    fireEvent.click(screen.getByRole("tab", { name: "Gates" }));
    fireEvent.click(screen.getByRole("button", { name: /Open the gate ledger/i }));
    expect(onEnterConsole).toHaveBeenCalledWith("gates");
  });

  it("opens free-play from the landing without entering the console", () => {
    const { onOpenFreeplay, onEnterConsole } = renderLanding();
    fireEvent.click(screen.getByRole("button", { name: /Drive the world model/i }));
    expect(onOpenFreeplay).toHaveBeenCalled();
    expect(onEnterConsole).not.toHaveBeenCalled();
  });

  it("lists the outstanding gate blockers on the gates tab", () => {
    renderLanding();
    fireEvent.click(screen.getByRole("tab", { name: "Gates" }));
    expect(screen.getByText(/Gate B: Cosmos one-step profile fails causal feedback/)).toBeInTheDocument();
    expect(screen.getByText(/Real backends cannot be dispatched/i)).toBeInTheDocument();
  });

  it("says the gate ledger was not returned rather than implying it passed", () => {
    renderLanding({ gates: [] });
    fireEvent.click(screen.getByRole("tab", { name: "Gates" }));
    expect(screen.getByText(/The gate ledger has not been returned/i)).toBeInTheDocument();
  });
});

describe("<LandingPage /> accessibility", () => {
  it("has no axe violations", { timeout: 60_000 }, async () => {
    const { container } = renderLanding();
    const results = await axe.run(container, {
      rules: {
        // jsdom has no layout or stylesheet, so computed-colour checks cannot run
        // here. Contrast is verified against the palette in styles.contrast.test.ts.
        "color-contrast": { enabled: false },
      },
    });
    expect(
      results.violations.map((violation) => `${violation.id}: ${violation.help}`),
    ).toEqual([]);
  });

  it("has no axe violations with the gallery open and empty", { timeout: 60_000 }, async () => {
    const { container } = renderLanding({ runs: [], episodes: [] });
    fireEvent.click(screen.getByRole("tab", { name: "Gallery" }));
    const results = await axe.run(container, { rules: { "color-contrast": { enabled: false } } });
    expect(
      results.violations.map((violation) => `${violation.id}: ${violation.help}`),
    ).toEqual([]);
  });
});
