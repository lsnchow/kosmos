import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Episode } from "../lib/api";
import { stageLadder } from "../lib/stages";
import { applyEpisodes, applySegment, createWall, type WallState } from "../lib/wall";
import { StageLadder } from "./StageLadder";

function wallFrom(segments: Parameters<typeof applySegment>[1][]): WallState {
  return segments.reduce((state, event) => applySegment(state, event), createWall());
}

const SEGMENT = {
  episode_id: "ep-1",
  run_id: "run-1",
  policy: "OpenVLA",
  task: "close_drawer",
  segment_index: 0,
  frame_urls: ["run-1/ep-1/0.png", "run-1/ep-1/1.png"],
  certified_frame_count: 14,
  provenance: "live" as const,
};

/** An episode row exactly as the API ships it today: none of the stage fields. */
const BARE_EPISODE: Episode = {
  episode_id: "ep-1",
  run_id: "run-1",
  policy: "OpenVLA",
  task: "close_drawer",
  status: "completed",
};

function step(label: string) {
  return screen.getByText(label).closest("li") as HTMLElement;
}

describe("stageLadder()", () => {
  it("waits rather than assuming, when no record has arrived", () => {
    const stages = stageLadder({ wall: createWall(), episodes: [] });
    expect(stages.map((stage) => stage.key)).toEqual(["policy", "world", "validity", "judge"]);
    expect(stages.every((stage) => stage.state === "pending")).toBe(true);
    expect(stages[1].detail).toMatch(/No segment_completed event has arrived yet/);
  });

  it("advances the world stage from a real segment_completed event", () => {
    const stages = stageLadder({ wall: wallFrom([SEGMENT]), episodes: [], runTerminal: false });
    const world = stages.find((stage) => stage.key === "world")!;
    expect(world.state).toBe("active");
    expect(world.detail).toBe("1 segment event · 14 certified frames accumulated");
    expect(world.source).toBe("Run event stream · segment_completed");
  });

  it("accumulates further segment events into the same stage", () => {
    const wall = wallFrom([SEGMENT, { ...SEGMENT, segment_index: 1, certified_frame_count: 9 }]);
    const world = stageLadder({ wall, episodes: [], runTerminal: true }).find(
      (stage) => stage.key === "world",
    )!;
    expect(world.state).toBe("reported");
    expect(world.detail).toBe("2 segment events · 23 certified frames accumulated");
  });

  it("says the certified count was not reported rather than assuming a chunk size", () => {
    const wall = wallFrom([{ ...SEGMENT, certified_frame_count: undefined }]);
    const world = stageLadder({ wall, episodes: [] }).find((stage) => stage.key === "world")!;
    expect(world.detail).toBe("1 segment event · 2 frames accumulated · certified count not reported");
    expect(world.detail).not.toMatch(/16/);
  });

  it("discloses segments that arrived without a certified count alongside those that did", () => {
    const wall = wallFrom([SEGMENT, { ...SEGMENT, segment_index: 1, certified_frame_count: undefined }]);
    const world = stageLadder({ wall, episodes: [] }).find((stage) => stage.key === "world")!;
    expect(world.detail).toBe("2 segment events · 14 certified frames accumulated · 1 without a certified count");
  });

  it("marks policy, validity and judge not reported when the records omit their fields", () => {
    const stages = stageLadder({
      wall: wallFrom([SEGMENT]),
      episodes: [BARE_EPISODE],
      runTerminal: true,
    });
    const byKey = Object.fromEntries(stages.map((stage) => [stage.key, stage]));
    expect(byKey.policy.state).toBe("unreported");
    expect(byKey.policy.detail).toBe("1 episode record · action horizon not reported");
    expect(byKey.validity.state).toBe("unreported");
    expect(byKey.validity.detail).toBe("1 episode record · validity not reported");
    expect(byKey.judge.state).toBe("unreported");
    expect(byKey.judge.detail).toBe("1 episode record · no binary outcome reported");
    // Frames landing is not evidence that a policy reported an action horizon.
    expect(byKey.world.state).toBe("reported");
  });

  it("reads each of the other three stages from its own field", () => {
    const stages = stageLadder({
      wall: wallFrom([SEGMENT]),
      episodes: [
        { ...BARE_EPISODE, horizon_actions: 70, validity: "valid", binary_success: true },
        {
          ...BARE_EPISODE,
          episode_id: "ep-2",
          horizon_actions: 70,
          validity: "invalid",
          binary_success: null,
          missing_reason: "invalid_world_video",
        },
      ],
      runTerminal: true,
    });
    const byKey = Object.fromEntries(stages.map((stage) => [stage.key, stage]));
    expect(byKey.policy.detail).toBe("2 of 2 records report an action horizon · longest 70 actions");
    expect(byKey.validity.detail).toBe("1 valid · 1 invalid");
    expect(byKey.judge.detail).toBe("1 of 2 records carry a binary outcome · 1 with a missing reason");
  });

  it("counts records whose validity was reported separately from those that were not", () => {
    const validity = stageLadder({
      wall: createWall(),
      episodes: [{ ...BARE_EPISODE, validity: "valid" }, { ...BARE_EPISODE, episode_id: "ep-2" }],
    }).find((stage) => stage.key === "validity")!;
    expect(validity.detail).toBe("1 valid · 1 not reported");
  });

  it("never counts a replayed segment twice", () => {
    const wall = wallFrom([SEGMENT, SEGMENT]);
    const world = stageLadder({ wall, episodes: [] }).find((stage) => stage.key === "world")!;
    expect(world.detail).toBe("1 segment event · 14 certified frames accumulated");
  });
});

describe("<StageLadder />", () => {
  it("names all four Chain stages in order", () => {
    render(<StageLadder wall={createWall()} episodes={[]} />);
    expect(screen.getByRole("heading", { name: /Chain stages/i })).toBeInTheDocument();
    const labels = screen.getAllByRole("listitem").map((item) => item.textContent ?? "");
    expect(labels).toHaveLength(4);
    expect(labels[0]).toMatch(/Policy/);
    expect(labels[1]).toMatch(/World model/);
    expect(labels[2]).toMatch(/Validity gate/);
    expect(labels[3]).toMatch(/Judge/);
  });

  it("shows 'not reported' on a stage whose field is absent", () => {
    render(
      <StageLadder
        wall={applyEpisodes(createWall(), [BARE_EPISODE])}
        episodes={[BARE_EPISODE]}
        runId="run-1"
        runTerminal
      />,
    );
    expect(within(step("Judge")).getByText("not reported")).toBeInTheDocument();
    expect(within(step("Judge")).getByText(/no binary outcome reported/)).toBeInTheDocument();
  });

  it("marks the world stage reported once a segment event has landed", () => {
    render(<StageLadder wall={wallFrom([SEGMENT])} episodes={[]} runId="run-1" runTerminal />);
    expect(within(step("World model")).getByText("reported")).toBeInTheDocument();
    expect(
      within(step("World model")).getByText(/1 segment event · 14 certified frames accumulated/),
    ).toBeInTheDocument();
    expect(within(step("Policy")).getByText("waiting")).toBeInTheDocument();
  });

  it("counts only the stages that actually reported", () => {
    render(<StageLadder wall={wallFrom([SEGMENT])} episodes={[BARE_EPISODE]} runTerminal />);
    expect(screen.getByRole("status")).toHaveTextContent("1 of 4 stages have reported");
  });

  it("states that the world stage is read from segment events", () => {
    render(<StageLadder wall={createWall()} episodes={[]} />);
    expect(screen.getByText(/the world stage from/i)).toBeInTheDocument();
    expect(screen.getByText("segment_completed")).toBeInTheDocument();
  });
});
