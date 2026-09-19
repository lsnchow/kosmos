import { act, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { advanceDisplayClock } from "../lib/clock";
import { applySegment, createWall, type WallState } from "../lib/wall";
import { RolloutWall } from "./RolloutWall";

function wallWith(
  segments: Parameters<typeof applySegment>[1][],
): WallState {
  return segments.reduce((state, event) => applySegment(state, event), createWall());
}

describe("<RolloutWall />", () => {
  it("renders twelve viewport slots once any identity is assigned", () => {
    const wall = wallWith([
      { episode_id: "a", policy: "OpenVLA", task: "close_drawer", segment_index: 0, frame_urls: ["f/0.png"], certified_frame_count: 16, provenance: "live" },
    ]);
    render(<RolloutWall wall={wall} />);
    expect(screen.getAllByRole("article")).toHaveLength(12);
    expect(screen.getAllByText(/No episode has claimed this slot/i)).toHaveLength(11);
    expect(
      screen.getByText(/tile count is a viewport choice, not the total robot or policy count/i),
    ).toBeInTheDocument();
  });

  it("explains why the wall is empty instead of showing placeholder frames", () => {
    render(<RolloutWall wall={createWall()} />);
    expect(screen.queryAllByRole("article")).toHaveLength(0);
    expect(
      screen.getByText(/No episode events have arrived for this run, so no slot has an identity yet/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/nothing here is pre-rendered to fill the grid/i)).toBeInTheDocument();
  });

  it("shows a per-tile provenance badge for every tile that has one", () => {
    const wall = wallWith([
      { episode_id: "a", policy: "OpenVLA", task: "close_drawer", segment_index: 0, frame_urls: ["f/0.png"], certified_frame_count: 16, provenance: "live" },
      { episode_id: "b", policy: "Octo", task: "open_drawer", segment_index: 0, frame_urls: ["g/0.png"], certified_frame_count: 16, provenance: "cached" },
      { episode_id: "c", policy: "MiniVLA", task: "to_sink", segment_index: 0, frame_urls: ["h/0.png"], certified_frame_count: 16, provenance: "replayed" },
      { episode_id: "d", policy: "SuSIE", task: "fold_cloth", segment_index: 0, frame_urls: ["i/0.png"], certified_frame_count: 16, provenance: "qualitative" },
    ]);
    render(<RolloutWall wall={wall} />);
    expect(screen.getByText("live")).toBeInTheDocument();
    expect(screen.getByText("cached")).toBeInTheDocument();
    expect(screen.getByText("replayed")).toBeInTheDocument();
    expect(screen.getByText("qualitative")).toBeInTheDocument();
  });

  it("labels a tile whose provenance was never reported rather than assuming live", () => {
    const wall = wallWith([
      { episode_id: "a", policy: "OpenVLA", task: "close_drawer", segment_index: 0, frame_urls: ["f/0.png"] },
    ]);
    render(<RolloutWall wall={wall} />);
    expect(screen.getByText("no provenance")).toBeInTheDocument();
    expect(screen.queryByText("live")).not.toBeInTheDocument();
  });

  it("appends the certified frame count the server reported", () => {
    const wall = wallWith([
      { episode_id: "a", policy: "OpenVLA", task: "close_drawer", segment_index: 0, frame_urls: ["f/0.png", "f/1.png"], certified_frame_count: 14, provenance: "live" },
      { episode_id: "a", policy: "OpenVLA", task: "close_drawer", segment_index: 1, frame_urls: ["f/2.png"], certified_frame_count: 9, provenance: "live" },
    ]);
    render(<RolloutWall wall={wall} />);
    expect(screen.getByText("23")).toBeInTheDocument();
    expect(screen.queryByText(/16 certified/)).not.toBeInTheDocument();
  });

  it("discloses an uncertified frame count rather than printing a chunk size", () => {
    const wall = wallWith([
      { episode_id: "a", policy: "OpenVLA", task: "close_drawer", segment_index: 0, frame_urls: ["f/0.png", "f/1.png"] },
    ]);
    render(<RolloutWall wall={wall} />);
    expect(screen.getByText("2 · uncertified")).toBeInTheDocument();
  });

  it("shows run, episode, task and policy identity on a live tile", () => {
    const wall = wallWith([
      { episode_id: "ep-77", run_id: "run-5", policy: "OpenVLA", task: "close_drawer", segment_index: 0, frame_urls: ["f/0.png"], certified_frame_count: 16, provenance: "live" },
    ]);
    render(<RolloutWall wall={wall} runId="run-5" />);
    const tile = screen.getByLabelText(/Viewport slot #01: OpenVLA on close_drawer/i);
    expect(within(tile).getByText("OpenVLA")).toBeInTheDocument();
    expect(within(tile).getByText("close_drawer")).toBeInTheDocument();
    // The episode id differs per tile, so it stays on the tile. The run id is
    // the same for all twelve and is stated once, in the panel header.
    expect(within(tile).getByText(/ep-77/)).toBeInTheDocument();
    expect(within(tile).getByTitle(/run run-5 · episode ep-77/)).toBeInTheDocument();
    expect(screen.getByText("run run-5")).toBeInTheDocument();
  });

  it("advances the accumulated clip on the shared display clock and loops it", () => {
    const wall = wallWith([
      { episode_id: "a", policy: "OpenVLA", task: "close_drawer", segment_index: 0, frame_urls: ["f/0.png", "f/1.png"], certified_frame_count: 2, provenance: "live" },
    ]);
    render(<RolloutWall wall={wall} />);
    const image = screen.getByAltText(/Accumulated generated frames for OpenVLA on close_drawer/i);
    const first = image.getAttribute("src");
    act(() => advanceDisplayClock(1));
    expect(image.getAttribute("src")).not.toBe(first);
    // Two frames, so one more tick wraps back around: the clip loops.
    act(() => advanceDisplayClock(1));
    expect(image.getAttribute("src")).toBe(first);
  });

  it("reports how many identities are running outside the viewport", () => {
    const segments = Array.from({ length: 14 }, (_, index) => ({
      episode_id: `ep-${index}`,
      policy: `P${index}`,
      task: "t",
      segment_index: 0,
      frame_urls: ["f/0.png"],
      certified_frame_count: 16,
      provenance: "live" as const,
    }));
    render(<RolloutWall wall={wallWith(segments)} />);
    expect(
      screen.getByText(/2 further policy\/task identities are running outside this viewport/i),
    ).toBeInTheDocument();
  });
});
