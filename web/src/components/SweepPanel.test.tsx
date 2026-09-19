import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { SweepPoint, SweepResponse } from "../lib/api";
import { mockFetch } from "../test/harness";
import { SweepPanel } from "./SweepPanel";

function point(overrides: Partial<SweepPoint> = {}): SweepPoint {
  return {
    operating_point_id: "op-256-30-4",
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
    qualification: "held_out_confirmed",
    ...overrides,
  };
}

const sweeps: SweepResponse = {
  status: "available",
  reason: "Development sweep only; the held-out panel has not been run.",
  points: [
    point({ operating_point_id: "op-a", estimated_usd: 1.02, tolerances_met: true }),
    point({ operating_point_id: "op-b", estimated_usd: 0.41, tolerances_met: true }),
    point({
      operating_point_id: "op-c",
      estimated_usd: 0.04,
      tolerances_met: false,
      pairwise_order_agreement: 0.61,
      fixed_task_horizon: 70,
      qualification: "failed_tolerances",
    }),
  ],
};

describe("<SweepPanel />", () => {
  it("issues no network request while the slider is dragged", () => {
    const { calls } = mockFetch({});
    render(<SweepPanel sweeps={sweeps} />);
    const slider = screen.getByLabelText(/Operating point/i);

    fireEvent.change(slider, { target: { value: "1" } });
    fireEvent.change(slider, { target: { value: "2" } });
    fireEvent.change(slider, { target: { value: "0" } });
    fireEvent.change(slider, { target: { value: "2" } });

    expect(calls).toHaveLength(0);
  });

  it("reads a different precomputed point on each drag without regenerating", () => {
    mockFetch({});
    render(<SweepPanel sweeps={sweeps} />);
    const slider = screen.getByLabelText(/Operating point/i);
    expect(screen.getByText("op-a")).toBeInTheDocument();

    fireEvent.change(slider, { target: { value: "2" } });
    expect(screen.getByText("op-c")).toBeInTheDocument();
    expect(screen.getByText("$0.04")).toBeInTheDocument();
  });

  it("shows the fixed task horizon and coverage beside every cost point", () => {
    render(<SweepPanel sweeps={sweeps} />);
    expect(screen.getByText("Fixed task horizon")).toBeInTheDocument();
    expect(screen.getByText("70 control ticks")).toBeInTheDocument();
    expect(screen.getByText("Coverage")).toBeInTheDocument();
    expect(screen.getByText("96%")).toBeInTheDocument();
    expect(
      screen.getByText(/so a shorter task cannot read as a saving/i),
    ).toBeInTheDocument();
  });

  it("marks where preregistered tolerances break", () => {
    render(<SweepPanel sweeps={sweeps} />);
    expect(
      screen.getByText(/Preregistered tolerances first fail at point 3 of 3 \(op-c\)/i),
    ).toBeInTheDocument();
  });

  it("flags the selected point when it failed tolerances", () => {
    render(<SweepPanel sweeps={sweeps} />);
    fireEvent.change(screen.getByLabelText(/Operating point/i), { target: { value: "2" } });
    expect(screen.getByText("tolerances not met")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/failed its preregistered/i);
  });

  it("surfaces the server's reason string rather than discarding it", () => {
    render(<SweepPanel sweeps={sweeps} />);
    expect(
      screen.getByText(/Development sweep only; the held-out panel has not been run\./),
    ).toBeInTheDocument();
  });

  it("explains an empty sweep with the server's reason and offers no slider", () => {
    render(
      <SweepPanel
        sweeps={{
          points: [],
          status: "unavailable",
          reason: "No independent real-model cost-fidelity sweep has completed.",
        }}
      />,
    );
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
    expect(
      screen.getByText(/No independent real-model cost-fidelity sweep has completed\./),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText(/will not generate, estimate, or imply a cheaper setting/i).length,
    ).toBeGreaterThan(0);
  });

  it("says a field was not reported instead of printing a placeholder number", () => {
    render(
      <SweepPanel
        sweeps={{
          points: [{ operating_point_id: "op-sparse" }],
          status: "available",
        }}
      />,
    );
    expect(screen.getAllByText("not reported").length).toBeGreaterThanOrEqual(4);
    expect(screen.getByText("tolerance result not reported")).toBeInTheDocument();
  });
});
