import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AnalysisCell, AnalysisResponse } from "../lib/api";
import { Scoreboard } from "./Scoreboard";

function cell(overrides: Partial<AnalysisCell> = {}): AnalysisCell {
  return {
    policy: "OpenVLA",
    task: "close_drawer",
    n: 50,
    valid: 44,
    successes: 31,
    missing: 6,
    coverage: 0.88,
    rate: 0.7045,
    positive_rate: 0.62,
    wilson: [0.556, 0.825],
    positive_wilson: [0.481, 0.744],
    missing_bounds: [0.62, 0.74],
    reference_rate: 0.92,
    reference_successes: 46,
    reference_n: 50,
    reference_wilson: [0.808, 0.972],
    simpler_rate: 0.04,
    simpler_successes: 2,
    service_failures: 1,
    missing_reason_counts: { invalid_world_video: 4, judge_no_quorum: 2 },
    scientific_status: "primary_manifest_bound",
    ...overrides,
  };
}

const analysis: AnalysisResponse = {
  endpoint: { name: "observed_positive_lower_bound" },
  lineage_leakage: { status: "pass" },
  cells: [
    cell(),
    cell({ policy: "Octo", task: "close_drawer", positive_rate: 0.02, reference_rate: 0.0, successes: 1, reference_successes: 0, simpler_rate: 0.0 }),
    cell({ policy: "MiniVLA", task: "close_drawer", positive_rate: 0.8, reference_rate: 0.98, successes: 40, reference_successes: 49, simpler_rate: 0.46 }),
  ],
  rankings: [
    {
      scope: "macro",
      visual_order: ["MiniVLA", "OpenVLA", "Octo"],
      ordering: "indeterminate",
      supported: false,
      reason: "Clustered starts lack a validated boundary-aware confidence method.",
    },
  ],
  pairwise: [
    {
      scope: "close_drawer",
      policy_a: "MiniVLA",
      policy_b: "Octo",
      status: "computed",
      ordering: "MiniVLA_over_Octo",
      supported: true,
      ordering_reason: "Conservative contrast lower bound is positive.",
      paired: { paired_n: 50, a_only: 39, b_only: 0, p_value: 1.8e-11, family_alpha: 0.05 / 75, passes_bonferroni_exact_test: true },
    },
    {
      scope: "close_drawer",
      policy_a: "MiniVLA",
      policy_b: "OpenVLA",
      status: "computed",
      ordering: "indeterminate",
      supported: false,
      ordering_reason: "Endpoint envelopes overlap at the family alpha.",
      paired: { paired_n: 50, a_only: 12, b_only: 3, p_value: 0.035, family_alpha: 0.05 / 75, passes_bonferroni_exact_test: false },
    },
  ],
};

describe("<Scoreboard />", () => {
  it("renders a rank column and marks it as a visual sort only", () => {
    render(<Scoreboard analysis={analysis} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("MiniVLA")).toBeInTheDocument();
    expect(within(rows[2]).getByText("Octo")).toBeInTheDocument();
    expect(screen.getAllByText("visual").length).toBe(3);
  });

  it("follows the server's own visual_order rather than sorting alphabetically", () => {
    render(<Scoreboard analysis={analysis} />);
    const rows = screen.getAllByRole("row").slice(1);
    const policies = rows.map((row) => within(row).getAllByRole("cell")[1].textContent ?? "");
    expect(policies[0]).toContain("MiniVLA");
    expect(policies[1]).toContain("OpenVLA");
    expect(policies[2]).toContain("Octo");
  });

  it("draws a visual error bar carrying both Wilson intervals", () => {
    render(<Scoreboard analysis={analysis} />);
    const bar = screen.getByRole("img", {
      name: /OpenVLA on close_drawer: generated 62% with Wilson interval 48.1% – 74.4%; human reference 92% with interval 80.8% – 97.2%/i,
    });
    expect(bar).toBeInTheDocument();
    expect(screen.getByText(/whiskers are Wilson 95% intervals/i)).toBeInTheDocument();
  });

  it("renders reference_wilson and simpler_rate instead of dropping them", () => {
    render(<Scoreboard analysis={analysis} />);
    // All three fixture cells carry the same reference interval.
    expect(screen.getAllByText("80.8% – 97.2%")).toHaveLength(3);
    expect(screen.getByText("4%")).toBeInTheDocument();
    expect(screen.getByText("46%")).toBeInTheDocument();
    expect(screen.getByText(/AutoEval T3/)).toBeInTheDocument();
  });

  it("shows an explicit exclusions column with both n and V denominators", () => {
    render(<Scoreboard analysis={analysis} />);
    expect(screen.getByText(/Excluded & invalid/)).toBeInTheDocument();
    expect(screen.getByText("n − V")).toBeInTheDocument();
    expect(screen.getByText("V / n")).toBeInTheDocument();
    expect(screen.getAllByText("44 / 50").length).toBe(3);
    expect(screen.getAllByText(/invalid_world_video 4 · judge_no_quorum 2/).length).toBe(3);
  });

  it("marks a pair as indeterminate and reports the supported one separately", () => {
    render(<Scoreboard analysis={analysis} />);
    expect(screen.getByText(/1 of 2 comparisons indeterminate/i)).toBeInTheDocument();
    expect(screen.getByText("separated")).toBeInTheDocument();
    expect(screen.getByText("indeterminate")).toBeInTheDocument();
    expect(screen.getByText(/Endpoint envelopes overlap at the family alpha\./)).toBeInTheDocument();
  });

  it("keeps the indeterminacy verdict next to the ranking", () => {
    render(<Scoreboard analysis={analysis} />);
    expect(screen.getByText(/Macro ordering: indeterminate — not supported/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Clustered starts lack a validated boundary-aware confidence method\./),
    ).toBeInTheDocument();
  });

  it("counts separated pairs per policy row", () => {
    render(<Scoreboard analysis={analysis} />);
    // MiniVLA is in both pairs but only one of them separated.
    expect(screen.getByText("1/2 separated")).toBeInTheDocument();
    // Octo's single pair separated; OpenVLA's single pair did not.
    expect(screen.getByText("1/1 separated")).toBeInTheDocument();
    expect(screen.getByText("0/1 separated")).toBeInTheDocument();
  });

  it("separates provisional operational counts from study statistics", () => {
    render(
      <Scoreboard
        analysis={{
          ...analysis,
          summary: {
            status: "provisional_operational_only",
            reason: "The ledger has nonterminal running records.",
            reported_records: 412,
            planned_records: 1500,
            terminal_records: 400,
            nonterminal_records: 1100,
          },
        }}
      />,
    );
    expect(screen.getByText("provisional")).toBeInTheDocument();
    expect(screen.getByText("Live operational counts only")).toBeInTheDocument();
    expect(screen.getByText(/These are not study statistics and no rate is computed from them\./)).toBeInTheDocument();
    expect(screen.getByText("412")).toBeInTheDocument();
  });

  it("says an absent SIMPLER cell is an absence, not a zero", () => {
    render(
      <Scoreboard
        analysis={{ ...analysis, cells: [cell({ task: "fold_cloth", simpler_rate: null, simpler_successes: null })] }}
      />,
    );
    expect(screen.getByText("no published cell")).toBeInTheDocument();
  });

  it("explains an empty analysis without inventing a zero", () => {
    render(<Scoreboard analysis={{}} />);
    expect(screen.getByText(/No persisted analysis exists for the selected run/i)).toBeInTheDocument();
    expect(screen.getByText(/no cell is filled with a zero/i)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("marks a blocked cell rather than printing its suppressed rate", () => {
    render(
      <Scoreboard
        analysis={{
          ...analysis,
          cells: [cell({ scientific_status: "blocked", scientific_reason: "Nonterminal ledger records." })],
        }}
      />,
    );
    expect(screen.getByText("blocked")).toBeInTheDocument();
  });

  it("exposes parity, backend and judge revisions", () => {
    render(
      <Scoreboard
        analysis={analysis}
        protocol={{ parity: "unverified", backend_revision: "irasim@c72b6da", judge_revision: "qwen2.5vl@cc59489" }}
      />,
    );
    expect(screen.getByText("unverified")).toBeInTheDocument();
    expect(screen.getByText("irasim@c72b6da")).toBeInTheDocument();
    expect(screen.getByText("qwen2.5vl@cc59489")).toBeInTheDocument();
  });
});
