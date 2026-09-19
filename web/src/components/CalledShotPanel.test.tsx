import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { CalledShot } from "../lib/api";
import { CalledShotPanel, calledShotFromProtocol } from "./CalledShotPanel";

const PENDING: CalledShot = {
  cell: "OpenVLA / close_drawer",
  human_rate: 0.92,
  human_n: 50,
  simpler_rate: 0.04,
  simpler_n: 50,
  plumb_estimate: null,
  plumb_status: "pending",
  prereg_uri: null,
  prereg_sha256: null,
};

describe("<CalledShotPanel />", () => {
  it("renders the published human and simulator cells that already ship in /api/protocol", () => {
    render(<CalledShotPanel calledShot={PENDING} />);
    expect(screen.getByText("92%")).toBeInTheDocument();
    expect(screen.getByText("4%")).toBeInTheDocument();
    expect(screen.getByText(/AutoEval Table 2 · n=50/)).toBeInTheDocument();
    expect(screen.getByText(/AutoEval Table 3 · n=50/)).toBeInTheDocument();
  });

  it("shows the estimate as pending when it is null, never as a number", () => {
    render(<CalledShotPanel calledShot={PENDING} />);
    expect(screen.getByText("pending")).toBeInTheDocument();
    expect(
      screen.getByText(/No frozen estimate has been recorded for this cell yet\./),
    ).toBeInTheDocument();
    // The only percentages on the panel are the two published ones.
    expect(screen.getAllByText(/^\d+(\.\d+)?%$/)).toHaveLength(2);
  });

  it("states the 88-point disagreement from the two published cells", () => {
    render(<CalledShotPanel calledShot={PENDING} />);
    expect(screen.getByText(/88 percentage points/)).toBeInTheDocument();
  });

  it("says no preregistration was returned rather than implying one exists", () => {
    render(<CalledShotPanel calledShot={PENDING} />);
    expect(screen.getByText("No preregistration record was returned")).toBeInTheDocument();
    expect(screen.getByText(/there is no independent timing evidence/i)).toBeInTheDocument();
  });

  it("renders the estimate with its signed preregistration tag and digest once recorded", () => {
    render(
      <CalledShotPanel
        calledShot={{
          ...PENDING,
          plumb_estimate: 0.78,
          plumb_status: "frozen",
          prereg_uri: "https://example.invalid/plumb/releases/tag/prereg-2026-09-19",
          prereg_sha256: "sha256:4f1c0d9e2b",
        }}
      />,
    );
    expect(screen.getByText("78%")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /prereg-2026-09-19/ })).toBeInTheDocument();
    expect(screen.getByText("sha256:4f1c0d9e2b")).toBeInTheDocument();
    expect(screen.getByText(/Frozen in the preregistration below/i)).toBeInTheDocument();
  });

  it("does not claim PLUMB beat the comparator", () => {
    render(<CalledShotPanel calledShot={{ ...PENDING, plumb_estimate: 0.78 }} />);
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/\bbeat(s|en)?\b/i);
    expect(text).not.toMatch(/\bSOTA\b/);
    expect(text).not.toMatch(/\bsolves\b/i);
    expect(text).not.toMatch(/matches human performance/i);
    expect(text).not.toMatch(/first ever/i);
  });

  it("recovers the headline cell from protocol.reference when the endpoint is absent", () => {
    const recovered = calledShotFromProtocol({
      reference: {
        trials_per_cell: 50,
        arithmetic: {
          headline_openvla_close_drawer: {
            human_rate: 0.92,
            simpler_rate: 0.04,
            percentage_point_gap: 88,
          },
        },
      },
    });
    expect(recovered?.human_rate).toBe(0.92);
    expect(recovered?.simpler_rate).toBe(0.04);
    expect(recovered?.human_n).toBe(50);
    // Recovery supplies the published reference only; it never invents an estimate.
    expect(recovered?.plumb_estimate).toBeNull();
    expect(recovered?.plumb_status).toBe("pending");
    expect(recovered?.prereg_uri).toBeNull();
  });

  it("returns nothing to recover when the protocol carries no reference", () => {
    expect(calledShotFromProtocol(undefined)).toBeUndefined();
    expect(calledShotFromProtocol({})).toBeUndefined();
    expect(calledShotFromProtocol({ reference: {} })).toBeUndefined();
  });

  it("reads the nested preregistration block the control plane ships", () => {
    render(
      <CalledShotPanel
        calledShot={{
          ...PENDING,
          preregistration: {
            status: "registered",
            uri: "https://example.invalid/tag/prereg-1",
            sha256: "sha256:abc123",
          },
        }}
      />,
    );
    expect(screen.getByRole("link", { name: /prereg-1/ })).toBeInTheDocument();
    expect(screen.getByText("sha256:abc123")).toBeInTheDocument();
  });

  it("says nothing is registered when the nested block is unregistered", () => {
    render(
      <CalledShotPanel
        calledShot={{ ...PENDING, preregistration: { status: "unregistered", uri: null, sha256: null } }}
      />,
    );
    expect(screen.getByText("No preregistration record was returned")).toBeInTheDocument();
  });

  it("prefers the server's own published gap over recomputing it", () => {
    render(<CalledShotPanel calledShot={{ ...PENDING, published_gap_points: 88 }} />);
    expect(screen.getByText(/88 percentage points/)).toBeInTheDocument();
  });

  it("shows an em dash rather than zero when a published cell is missing", () => {
    render(<CalledShotPanel calledShot={{ cell: "OpenVLA / close_drawer" }} />);
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
    expect(screen.queryByText(/percentage points/)).not.toBeInTheDocument();
  });
});
