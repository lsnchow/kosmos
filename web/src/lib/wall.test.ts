import { describe, expect, it } from "vitest";
import type { Episode, SegmentCompletedEvent } from "./api";
import {
  applyEpisodes,
  applySegment,
  createWall,
  identityKey,
  ignitionDelayMs,
  provenanceIsMixed,
  WALL_SLOT_COUNT,
} from "./wall";

function segment(overrides: Partial<SegmentCompletedEvent> = {}): SegmentCompletedEvent {
  return {
    episode_id: "ep-1",
    run_id: "run-1",
    policy: "OpenVLA",
    task: "close_drawer",
    segment_index: 0,
    frame_urls: ["a/0.png", "a/1.png"],
    certified_frame_count: 2,
    provenance: "live",
    ...overrides,
  };
}

describe("the 12-tile wall", () => {
  it("starts with twelve empty slots and no fabricated identities", () => {
    const wall = createWall();
    expect(wall.slots).toHaveLength(WALL_SLOT_COUNT);
    expect(wall.slots.every((slot) => slot === undefined)).toBe(true);
    expect(wall.overflowKeys).toEqual([]);
  });

  it("accumulates frames across successive segments instead of replacing them", () => {
    let wall = createWall();
    wall = applySegment(wall, segment({ segment_index: 0, frame_urls: ["a/0.png", "a/1.png"] }));
    expect(wall.slots[0]?.frames).toHaveLength(2);

    wall = applySegment(wall, segment({ segment_index: 1, frame_urls: ["a/2.png", "a/3.png"] }));
    expect(wall.slots[0]?.frames).toHaveLength(4);
    expect(wall.slots[0]?.frames.at(-1)).toBe("/api/artifacts/a/3.png");
    expect(wall.slots[0]?.segmentCount).toBe(2);
  });

  it("sums the certified frame counts the server reported, never a hardcoded 16", () => {
    let wall = createWall();
    wall = applySegment(wall, segment({ segment_index: 0, certified_frame_count: 14 }));
    wall = applySegment(wall, segment({ segment_index: 1, certified_frame_count: 9 }));
    expect(wall.slots[0]?.certifiedFrameCount).toBe(23);
    expect(wall.slots[0]?.segmentsMissingCertifiedCount).toBe(0);
  });

  it("counts segments that reported no certified frame count rather than guessing one", () => {
    let wall = createWall();
    wall = applySegment(wall, segment({ certified_frame_count: undefined }));
    expect(wall.slots[0]?.certifiedFrameCount).toBeUndefined();
    expect(wall.slots[0]?.segmentsMissingCertifiedCount).toBe(1);
    expect(wall.slots[0]?.frames).toHaveLength(2);
  });

  it("ignores a replayed duplicate segment so frames are not double-counted", () => {
    let wall = createWall();
    wall = applySegment(wall, segment({ segment_index: 0 }));
    const afterFirst = wall;
    wall = applySegment(wall, segment({ segment_index: 0 }));
    expect(wall).toBe(afterFirst);
    expect(wall.slots[0]?.frames).toHaveLength(2);
    expect(wall.slots[0]?.certifiedFrameCount).toBe(2);
  });

  it("keys slots by policy and task so an identity keeps its slot for the whole run", () => {
    let wall = createWall();
    wall = applySegment(wall, segment({ policy: "OpenVLA", task: "close_drawer" }));
    wall = applySegment(wall, segment({ policy: "Octo", task: "open_drawer", episode_id: "ep-2" }));
    wall = applySegment(
      wall,
      segment({ policy: "OpenVLA", task: "close_drawer", episode_id: "ep-9", segment_index: 7 }),
    );

    expect(wall.slots[0]?.key).toBe(identityKey("OpenVLA", "close_drawer"));
    expect(wall.slots[1]?.key).toBe(identityKey("Octo", "open_drawer"));
    // The third event belongs to slot 0's identity, and a *different* episode.
    expect(wall.slots[0]?.frames).toHaveLength(4);
    expect(wall.slots[0]?.episodeIds).toEqual(["ep-1", "ep-9"]);
    expect(wall.slots[2]).toBeUndefined();
  });

  it("records overflow identities rather than evicting a tile", () => {
    let wall = createWall();
    for (let index = 0; index < WALL_SLOT_COUNT + 3; index += 1) {
      wall = applySegment(
        wall,
        segment({ policy: `P${index}`, task: "t", episode_id: `ep-${index}` }),
      );
    }
    expect(wall.slots.filter(Boolean)).toHaveLength(WALL_SLOT_COUNT);
    expect(wall.overflowKeys).toHaveLength(3);
    expect(wall.slots[0]?.policy).toBe("P0");
  });

  it("tracks every provenance it was told and flags a mixed-source tile", () => {
    let wall = createWall();
    wall = applySegment(wall, segment({ segment_index: 0, provenance: "live" }));
    expect(provenanceIsMixed(wall.slots[0]!)).toBe(false);

    wall = applySegment(wall, segment({ segment_index: 1, provenance: "replayed" }));
    const slot = wall.slots[0]!;
    expect(slot.provenance).toBe("replayed");
    expect(slot.provenanceCounts).toEqual({ live: 1, replayed: 1 });
    expect(provenanceIsMixed(slot)).toBe(true);
  });

  it("leaves provenance undefined when the server reported none", () => {
    const wall = applySegment(createWall(), segment({ provenance: undefined }));
    expect(wall.slots[0]?.provenance).toBeUndefined();
    expect(wall.slots[0]?.provenanceCounts).toEqual({});
  });

  it("rejects an unrecognised provenance string instead of displaying it", () => {
    const wall = applySegment(
      createWall(),
      segment({ provenance: "definitely-real" as never }),
    );
    expect(wall.slots[0]?.provenance).toBeUndefined();
  });

  it("seeds identities from snapshot episodes without duplicating SSE frames", () => {
    const episode: Episode = {
      episode_id: "ep-1",
      run_id: "run-1",
      policy: "OpenVLA",
      task: "close_drawer",
      status: "running",
      frame_urls: ["a/0.png", "a/1.png"],
      certified_frame_count: 2,
      provenance: "live",
    };
    let wall = applySegment(createWall(), segment({ segment_index: 0 }));
    wall = applyEpisodes(wall, [episode]);
    expect(wall.slots[0]?.frames).toHaveLength(2);
  });

  it("assigns an identity from a snapshot even before any segment arrives", () => {
    const wall = applyEpisodes(createWall(), [
      { episode_id: "ep-1", policy: "MiniVLA", task: "to_basket", status: "running" },
    ]);
    expect(wall.slots[0]?.policy).toBe("MiniVLA");
    expect(wall.slots[0]?.frames).toEqual([]);
  });

  it("staggers ignition inside the 40-60 ms presentation band", () => {
    expect(ignitionDelayMs(0)).toBe(0);
    const step = ignitionDelayMs(1) - ignitionDelayMs(0);
    expect(step).toBeGreaterThanOrEqual(40);
    expect(step).toBeLessThanOrEqual(60);
    expect(ignitionDelayMs(11)).toBe(step * 11);
  });
});
