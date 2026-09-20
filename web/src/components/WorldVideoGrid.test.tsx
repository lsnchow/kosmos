import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import axe from "axe-core";
import { http, mockFetch } from "../test/harness";
import { gallerySelection, WorldVideoGrid } from "./WorldVideoGrid";
import type { WorldVideo } from "./WorldVideoQueue";

const video: WorldVideo = {
  id: "saved-cosmos",
  title: "Cosmos saved fixture",
  model: "Cosmos3-Nano",
  kind: "fixture",
  video_url: "/api/artifacts/cosmos.mp4",
  report_url: "/api/artifacts/cosmos.json",
  sha256: "sha256:first",
  frame_count: 17,
  fps: 5,
  duration_seconds: 3.4,
  width: 640,
  height: 480,
  qualified: false,
  provenance: "recorded_model_output",
};
const mount = () =>
  render(
    <MemoryRouter>
      <WorldVideoGrid />
    </MemoryRouter>,
  );

describe("saved video gallery", () => {
  it("opens a new image-reconditioned branch without inference, pauses the gallery, and restores focus on close", async () => {
    const { calls } = mockFetch({
      "/api/world-videos": { videos: [video] },
      "/api/demo/status": {
        available: true,
        configured: true,
        health: "ready",
      },
    });
    mount();
    const trigger = await screen.findByRole("button", {
      name: "Start a new interactive branch from Cosmos saved fixture",
    });
    trigger.focus();
    fireEvent.click(trigger);
    const dialog = await screen.findByRole("dialog", { name: "New evaluation" });
    expect(within(dialog).getByText(/not an exact checkpoint resume/i)).toBeInTheDocument();
    expect(within(dialog).getByDisplayValue("Put the pot to the left of the purple item.")).toBeInTheDocument();
    expect(calls.every((call) => call.method === "GET")).toBe(true);
    expect(
      calls.some(
        (call) => call.url === "/api/demo/status",
      ),
    ).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it("excludes tiny probes, unsafe paths and duplicate bytes, and caps visible decoders at six", () => {
    expect(
      gallerySelection([
        video,
        { ...video, id: "duplicate" },
        { ...video, frame_count: 2 },
        { ...video, video_url: "https://other.example/a.mp4" },
      ]),
    ).toEqual([video]);
    expect(
      gallerySelection(
        Array.from({ length: 9 }, (_, i) => ({
          ...video,
          id: String(i),
          sha256: String(i),
        })),
      ),
    ).toHaveLength(6);
  });

  it("shows actual metadata with muted loops and makes no inference requests", async () => {
    const { calls } = mockFetch({ "/api/world-videos": { videos: [video] } });
    const play = vi
      .spyOn(HTMLMediaElement.prototype, "play")
      .mockResolvedValue();
    const pause = vi.spyOn(HTMLMediaElement.prototype, "pause");
    mount();
    const player = (await screen.findByLabelText(
      "Saved experiment: Cosmos saved fixture",
    )) as HTMLVideoElement;
    expect(player.muted).toBe(true);
    expect(player).toHaveAttribute("loop");
    expect(player).toHaveAttribute("controls");
    expect(screen.getByText(/5 FPS/)).toBeInTheDocument();
    await waitFor(() => expect(play).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: "Pause all" }));
    expect(pause).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Resume loops" }));
    expect(calls).toEqual([
      { url: "/api/world-videos", method: "GET", body: undefined },
    ]);
  });

  it("pauses offscreen and hidden-page media and resumes only visible media", async () => {
    let visibility!: IntersectionObserverCallback;
    vi.stubGlobal(
      "IntersectionObserver",
      class {
        constructor(callback: IntersectionObserverCallback) {
          visibility = callback;
        }
        observe() {}
        disconnect() {}
      },
    );
    mockFetch({ "/api/world-videos": { videos: [video] } });
    const play = vi
      .spyOn(HTMLMediaElement.prototype, "play")
      .mockResolvedValue();
    const pause = vi.spyOn(HTMLMediaElement.prototype, "pause");
    mount();
    await screen.findByLabelText("Saved experiment: Cosmos saved fixture");
    expect(play).not.toHaveBeenCalled();
    act(() =>
      visibility(
        [{ isIntersecting: true } as IntersectionObserverEntry],
        {} as IntersectionObserver,
      ),
    );
    expect(play).toHaveBeenCalledTimes(1);
    act(() =>
      visibility(
        [{ isIntersecting: false } as IntersectionObserverEntry],
        {} as IntersectionObserver,
      ),
    );
    const count = pause.mock.calls.length;
    fireEvent(document, new Event("visibilitychange"));
    expect(pause.mock.calls.length).toBeGreaterThan(count);
    expect(play).toHaveBeenCalledTimes(1);
  });

  it("does not start playback for reduced motion", async () => {
    vi.spyOn(window, "matchMedia").mockReturnValue({
      matches: true,
      addEventListener() {},
      removeEventListener() {},
    } as unknown as MediaQueryList);
    mockFetch({ "/api/world-videos": { videos: [video] } });
    const play = vi
      .spyOn(HTMLMediaElement.prototype, "play")
      .mockResolvedValue();
    mount();
    await screen.findByLabelText("Saved experiment: Cosmos saved fixture");
    expect(play).not.toHaveBeenCalled();
  });

  it("reports playback errors without a silent retry or replacement", async () => {
    mockFetch({ "/api/world-videos": { videos: [video] } });
    mount();
    fireEvent.error(
      await screen.findByLabelText("Saved experiment: Cosmos saved fixture"),
    );
    expect(screen.getByRole("alert")).toHaveTextContent("could not play");
    expect(
      screen.getByLabelText("Saved experiment: Cosmos saved fixture"),
    ).toHaveAttribute("src", video.video_url);
  });

  it("has an actionable empty state and retryable catalog failure", async () => {
    let fail = true;
    mockFetch({
      "/api/world-videos": () => (fail ? http(503, {}) : { videos: [] }),
    });
    mount();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "could not be loaded",
    );
    fail = false;
    fireEvent.click(screen.getByRole("button", { name: "Retry catalog" }));
    expect(
      await screen.findByRole("link", { name: /Check the recording archive/ }),
    ).toHaveAttribute("href", "/clips");
  });

  it("keeps the image-branch dialog accessible when the runtime is unavailable", async () => {
    mockFetch({
      "/api/world-videos": { videos: [video] },
      "/api/demo/status": {
        available: false,
        configured: false,
        health: "not_configured",
        reason: "The interactive runtime is not configured.",
      },
    });
    mount();
    fireEvent.click(
      await screen.findByRole("button", {
        name: "Start a new interactive branch from Cosmos saved fixture",
      }),
    );
    await screen.findByText("The interactive runtime is not configured.");
    expect(screen.getByRole("button", { name: "Start manual image branch" })).toBeDisabled();
    let result!: axe.AxeResults;
    await act(async () => {
      result = await axe.run(document.body, {
        preload: false,
        rules: {
          "color-contrast": { enabled: false },
          "video-caption": { enabled: false },
        },
      });
    });
    expect(result.violations).toEqual([]);
  });

  it("passes semantic accessibility checks with real gallery cards", async () => {
    mockFetch({ "/api/world-videos": { videos: [video] } });
    const { container } = mount();
    await screen.findByLabelText("Saved experiment: Cosmos saved fixture");
    let result!: axe.AxeResults;
    await act(async () => {
      result = await axe.run(container, {
        preload: false,
        rules: {
          "color-contrast": { enabled: false },
          "video-caption": { enabled: false },
        },
      });
    });
    expect(result.violations).toEqual([]);
  });
});
