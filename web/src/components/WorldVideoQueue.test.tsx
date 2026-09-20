import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import axe from "axe-core";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { mockFetch } from "../test/harness";
import { WorldVideoQueue } from "./WorldVideoQueue";

const VIDEOS = [
  {
    id: "cosmos-full-3p4s",
    title: "Cosmos full recording · 3.4 s",
    model: "Cosmos3-Nano",
    kind: "full horizon recorded output",
    video_url: "/api/artifacts/cosmos-full.mp4",
    report_url: "/api/artifacts/cosmos-full.json",
    sha256: "sha256:abc123",
    download_name: "cosmos-full-original.mp4",
    source_experiment: "cosmos-forward-dynamics-full",
    frame_count: 17,
    duration_seconds: 3.4,
    width: 480,
    height: 480,
    notes: ["Recorded output; no live request was made by this page."],
    qualified: false,
    provenance: "recorded_model_output" as const,
  },
  {
    id: "irasim-two-frame-probe",
    title: "IRASim two-frame probe",
    model: "IRASim",
    kind: "short recorded probe",
    video_url: "/api/artifacts/irasim-probe.mp4",
    frame_count: 2,
    duration_seconds: 0.4,
    qualified: false,
    provenance: "recorded_model_output" as const,
  },
];

function install(videos: unknown = VIDEOS) {
  return mockFetch({ "/api/world-videos": { videos, total: Array.isArray(videos) ? videos.length : 0, qualified: false } });
}

describe("<WorldVideoQueue />", () => {
  it("defaults to the first server-recorded video and does not autoplay it", async () => {
    install();
    render(<WorldVideoQueue />);
    const video = await screen.findByLabelText("Recorded model output: Cosmos full recording · 3.4 s");
    expect(video).toHaveAttribute("controls");
    expect(video).not.toHaveAttribute("autoplay");
    expect(screen.getByText("Recorded model output", { selector: "b" }).parentElement).toHaveTextContent("not live · unqualified");
    expect(screen.getByText(/Playback queue only — it does not generate/i)).toBeInTheDocument();
    expect(screen.getByText("1", { selector: "b" }).parentElement).toHaveTextContent("1 of 2 catalog recordings");
    expect(screen.getByText("17")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Download MP4/i })).toHaveAttribute("download", "cosmos-full-original.mp4");
    expect(screen.getByText(/Source experiment: cosmos-forward-dynamics-full/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /source experiment cosmos-forward-dynamics-full/i })).toBeInTheDocument();
  });

  it("starts ordered playback only after Play all, then advances on ended", async () => {
    install();
    const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
    render(<WorldVideoQueue />);
    const user = userEvent.setup();
    await screen.findByLabelText("Recorded model output: Cosmos full recording · 3.4 s");
    expect(play).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByLabelText("Recorded model output: IRASim two-frame probe");
    await user.click(screen.getByRole("button", { name: "Play all" }));
    const first = await screen.findByLabelText("Recorded model output: Cosmos full recording · 3.4 s");
    await waitFor(() => expect(play).toHaveBeenCalledTimes(1));
    fireEvent.ended(first);
    await screen.findByLabelText("Recorded model output: IRASim two-frame probe");
    await waitFor(() => expect(play).toHaveBeenCalledTimes(2));
    play.mockRestore();
  });

  it("keeps a failed recording selected, stops automatic playback, and exposes manual skip", async () => {
    install();
    render(<WorldVideoQueue />);
    const user = userEvent.setup();
    const first = await screen.findByLabelText("Recorded model output: Cosmos full recording · 3.4 s");
    fireEvent.error(first);
    expect(await screen.findByRole("alert")).toHaveTextContent(/remains selected/i);
    expect(screen.getByLabelText("Recorded model output: Cosmos full recording · 3.4 s")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Skip failed recording" }));
    expect(await screen.findByLabelText("Recorded model output: IRASim two-frame probe")).toBeInTheDocument();
  });

  it("shows an honest empty catalog with Run frames as the next action", async () => {
    install([]);
    render(<WorldVideoQueue />);
    expect(await screen.findByText(/No recorded world-model video is available/i)).toBeInTheDocument();
    expect(screen.getByText(/Select Run frames/i)).toBeInTheDocument();
  });

  it("has no axe violations", async () => {
    install();
    const { container } = render(<WorldVideoQueue />);
    await screen.findByRole("heading", { name: "Cosmos full recording · 3.4 s" });
    let result!: axe.AxeResults;
    await act(async () => {
      result = await axe.run(container, { preload: false, rules: { "color-contrast": { enabled: false }, "video-caption": { enabled: false } } });
    });
    expect(result.violations).toEqual([]);
  });
});
