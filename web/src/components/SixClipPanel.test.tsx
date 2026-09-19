import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SixClipPanel } from "./SixClipPanel";

const CLIPS = Array.from({ length: 6 }, (_, index) => ({
  id: `clip-${index}`,
  url: `clips/${index}.mp4`,
  provenance_sealed: true,
}));

describe("<SixClipPanel />", () => {
  it("explains an unpublished panel rather than showing stand-in clips", () => {
    render(
      <SixClipPanel
        data={{
          clips: [],
          revealed: false,
          status: "unavailable",
          reason: "No six-clip manifest exists.",
        }}
      />,
    );
    expect(screen.getByText(/No six-clip manifest exists\./)).toBeInTheDocument();
    expect(
      screen.getByText(/a generated clip labelled as the test would make the test meaningless/i),
    ).toBeInTheDocument();
    expect(document.querySelectorAll("video")).toHaveLength(0);
  });

  it("keeps provenance sealed until reveal", () => {
    render(<SixClipPanel data={{ clips: CLIPS, revealed: false }} />);
    expect(screen.getAllByText("sealed")).toHaveLength(6);
    expect(screen.queryByText("real")).not.toBeInTheDocument();
    expect(screen.queryByText("generated")).not.toBeInTheDocument();
  });

  it("labels each clip's provenance once revealed", () => {
    render(
      <SixClipPanel
        data={{
          revealed: true,
          clips: CLIPS.map((clip, index) => ({
            ...clip,
            provenance: index === 3 ? ("real" as const) : ("generated" as const),
            source: index === 3 ? "bridge_orig episode 41" : "irasim@c72b6da",
          })),
        }}
      />,
    );
    expect(screen.getByText("real")).toBeInTheDocument();
    expect(screen.getAllByText("generated")).toHaveLength(5);
  });

  it("makes no accuracy claim when no answers were recorded", () => {
    render(<SixClipPanel data={{ clips: CLIPS, revealed: true, recorded_answers: null }} />);
    expect(
      screen.getByText(/No audience answers were recorded, so no accuracy figure is shown/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/no claim is made about whether the room could tell/i)).toBeInTheDocument();
  });

  it("does not claim the audience was fooled", () => {
    render(<SixClipPanel data={{ clips: CLIPS, revealed: true }} />);
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/fooled/i);
    expect(text).not.toMatch(/nobody gets that right/i);
  });

  it("reports an accuracy figure only from answers that were actually recorded", () => {
    render(
      <SixClipPanel
        data={{ clips: CLIPS, revealed: true, recorded_answers: { n: 40, correct: 9, method: "show of hands" } }}
      />,
    );
    expect(screen.getByText(/identified the real clip/)).toHaveTextContent("9");
    expect(screen.getByText(/identified the real clip/)).toHaveTextContent("40");
    expect(screen.getByText(/22.5%/)).toBeInTheDocument();
    expect(screen.getByText(/show of hands/)).toBeInTheDocument();
  });

  it("accepts the audience_accuracy field the control plane currently ships", () => {
    render(
      <SixClipPanel data={{ clips: CLIPS, revealed: true, audience_accuracy: { n: 20, correct: 4 } }} />,
    );
    expect(screen.getByText(/20%/)).toBeInTheDocument();
  });
});
