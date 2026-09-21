import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { burstIdempotencyKey, burstIdentity, burstRequestBody } from "../lib/burst";
import { BurstPanel, burstDisabledReason } from "./BurstPanel";

const identity = burstIdentity({
  policies: ["OpenVLA", "Octo"],
  tasks: ["close_drawer", "open_drawer"],
  startsPerTask: 50,
  seed: 20260919,
});

function renderPanel(overrides: Partial<Parameters<typeof BurstPanel>[0]> = {}) {
  const onLaunch = vi.fn();
  const props: Parameters<typeof BurstPanel>[0] = {
    samples: [],
    submitting: false,
    policyCount: 6,
    taskCount: 5,
    plannedEpisodes: 1500,
    idempotencyKey: burstIdempotencyKey(identity),
    onLaunch,
    onCancel: vi.fn(),
    ...overrides,
  };
  render(<BurstPanel {...props} />);
  return { onLaunch };
}

describe("burst idempotency", () => {
  it("derives the same key from the same run identity, so a duplicate click collides", () => {
    const again = burstIdentity({
      policies: ["OpenVLA", "Octo"],
      tasks: ["close_drawer", "open_drawer"],
      startsPerTask: 50,
      seed: 20260919,
    });
    expect(burstIdempotencyKey(again)).toBe(burstIdempotencyKey(identity));
  });

  it("is insensitive to policy and task ordering", () => {
    const reordered = burstIdentity({
      policies: ["Octo", "OpenVLA"],
      tasks: ["open_drawer", "close_drawer"],
      startsPerTask: 50,
      seed: 20260919,
    });
    expect(burstIdempotencyKey(reordered)).toBe(burstIdempotencyKey(identity));
  });

  it("changes when the run identity changes", () => {
    const differentSeed = burstIdentity({
      policies: ["OpenVLA", "Octo"],
      tasks: ["close_drawer", "open_drawer"],
      startsPerTask: 50,
      seed: 1,
    });
    expect(burstIdempotencyKey(differentSeed)).not.toBe(burstIdempotencyKey(identity));
  });

  it("puts the derived key in the request body and never a random value", () => {
    const body = burstRequestBody(identity);
    expect(body.idempotency_key).toBe(burstIdempotencyKey(identity));
    expect(body.idempotency_key).toMatch(/^kosmos-burst-[0-9a-f]{8}$/);
    expect(burstRequestBody(identity)).toEqual(body);
  });
});

describe("<BurstPanel /> double-fire protection", () => {
  it("stays enabled when there is no run", () => {
    expect(burstDisabledReason({ submitting: false, policyCount: 6, taskCount: 5 })).toBeUndefined();
  });

  it("is disabled while submitting", () => {
    expect(burstDisabledReason({ submitting: true, policyCount: 6, taskCount: 5 })).toBe(
      "Submitting this run",
    );
  });

  it("stays disabled after the POST returns, for as long as the run is not terminal", () => {
    // The old build cleared its flag here and re-enabled the button, which is
    // how two stage clicks became two concurrent 1,500-episode runs.
    expect(
      burstDisabledReason({
        submitting: false,
        run: { id: "run-1", status: "running" },
        policyCount: 6,
        taskCount: 5,
      }),
    ).toContain("running");
  });

  it("re-enables only once the run reaches a terminal status", () => {
    for (const status of ["completed", "failed", "cancelled"]) {
      expect(
        burstDisabledReason({
          submitting: false,
          run: { id: "run-1", status },
          policyCount: 6,
          taskCount: 5,
        }),
      ).toBeUndefined();
    }
  });

  it("is disabled when the protocol returned no matrix", () => {
    expect(burstDisabledReason({ submitting: false, policyCount: 0, taskCount: 0 })).toBe(
      "The protocol did not return policies and tasks",
    );
  });

  it("cannot be fired twice while a run is alive", async () => {
    const user = userEvent.setup();
    const { onLaunch } = renderPanel({ run: { id: "run-1", status: "running" } });
    const button = screen.getByRole("button", { name: /Run 1,500 episodes/i });
    expect(button).toBeDisabled();
    await user.click(button);
    await user.click(button);
    expect(onLaunch).not.toHaveBeenCalled();
  });

  it("fires exactly once per click when the button is live", async () => {
    const user = userEvent.setup();
    const { onLaunch } = renderPanel();
    await user.click(screen.getByRole("button", { name: /Run 1,500 episodes/i }));
    expect(onLaunch).toHaveBeenCalledTimes(1);
  });

  it("explains the disabled state and names the idempotency protection", () => {
    renderPanel({ run: { id: "run-1", status: "running" } });
    expect(screen.getByText(/same derived idempotency key/i)).toBeInTheDocument();
  });
});

describe("<BurstPanel /> honesty", () => {
  it("shows the replica chart's reason instead of plotting a fabricated climb", () => {
    renderPanel();
    expect(screen.getByText(/No replica counts have been reported/i)).toBeInTheDocument();
    expect(
      screen.getByText(/A configured maximum is not an active replica count/i),
    ).toBeInTheDocument();
  });

  it("shows cost as unavailable rather than as zero dollars", () => {
    renderPanel();
    expect(screen.getAllByText("Unavailable").length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
  });

  it("shows an em dash rather than 0 completed when the ledger reported nothing", () => {
    renderPanel();
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("plots active and desired replicas once the platform reports them", () => {
    renderPanel({
      samples: [
        { t: 1_000, active: 1, desired: 100 },
        { t: 2_000, active: 40, desired: 100 },
        { t: 3_000, active: 97, desired: 100 },
      ],
    });
    expect(screen.getByRole("img", { name: /Active and desired replicas over 2 seconds/i })).toBeInTheDocument();
    expect(screen.getByText(/Active 97/)).toBeInTheDocument();
    expect(screen.getByText(/Desired 100/)).toBeInTheDocument();
    expect(screen.getByText(/3 reported samples/)).toBeInTheDocument();
  });

  it("reports a reporting gap rather than drawing through it", () => {
    renderPanel({
      samples: [
        { t: 1_000, active: 1, desired: 100 },
        { t: 2_000 },
        { t: 3_000, active: 97, desired: 100 },
      ],
    });
    expect(screen.getByText(/2 reported samples · 2 reporting gaps/)).toBeInTheDocument();
    expect(screen.getByText(/A gap in a line is a gap in the platform's reporting/i)).toBeInTheDocument();
  });
});
