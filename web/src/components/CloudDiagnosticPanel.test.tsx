import { act, render, screen, waitFor } from "@testing-library/react";
import axe from "axe-core";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { mockFetch } from "../test/harness";
import { CloudDiagnosticPanel } from "./CloudDiagnosticPanel";

const STATUS = {
  configured: true,
  available: true,
  policy: "SuSIE_LL",
  qualified: false,
  model_id: "susie-ll-diagnostic-v1",
  deployment_id: "dep-diagnostic-1",
  fixture: {
    current_url: "/api/artifacts/fixture-current.png",
    goal_url: "/api/artifacts/fixture-goal.png",
    description: "Pinned fixed-fixture diagnostic only.",
  },
};

const COMPLETED = {
  request_id: "cloud-1",
  status: "completed",
  qualified: false,
  diagnostic_only: true,
  physical_action: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 1],
  timing: { load_seconds_once: 4.25, inference_seconds: 0.83 },
  report_url: "/api/artifacts/cloud-1/report.json",
  raw_response_url: "/api/artifacts/cloud-1/raw.json",
};

function install(status: unknown = STATUS, records: unknown[] = [COMPLETED]) {
  const created = { request_id: "cloud-new", status: "pending", qualified: false, diagnostic_only: true };
  return mockFetch({
    "/api/cloud-diagnostics/status": status,
    "/api/cloud-diagnostics/cloud-1": COMPLETED,
    "/api/cloud-diagnostics": (init: RequestInit | undefined) =>
      (init?.method ?? "GET").toUpperCase() === "POST" ? created : { requests: records },
  });
}

describe("<CloudDiagnosticPanel />", () => {
  it("shows the server-pinned frames and queues one unqualified diagnostic without inventing a prompt or score", async () => {
    const { calls } = install();
    render(<CloudDiagnosticPanel />);
    const user = userEvent.setup();
    await screen.findByRole("heading", { name: "Cloud diagnostic" });

    expect(screen.getByAltText("Pinned diagnostic current observation")).toBeInTheDocument();
    expect(screen.getByAltText("Pinned diagnostic goal observation")).toBeInTheDocument();
    expect(screen.getByText(/not a generated rollout, a judge score, or a study result/i)).toBeInTheDocument();
    expect(screen.getByText("susie-ll-diagnostic-v1")).toBeInTheDocument();
    expect(screen.getByText("Reported 7-D physical action")).toBeInTheDocument();
    expect(screen.getByText(/\[0\.01000, 0\.02000/)).toBeInTheDocument();
    expect(screen.queryByText(/success rate|passed gate/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Run cloud model" }));
    await waitFor(() => expect(calls.some((call) => call.method === "POST")).toBe(true));
    const post = calls.find((call) => call.method === "POST");
    expect(JSON.parse(post?.body ?? "{}")).toEqual({});
    expect(screen.getAllByText("cloud-new")).toHaveLength(2);
    expect(screen.getAllByText("pending").length).toBeGreaterThan(0);
  });

  it("explains an unavailable diagnostic next to its disabled button", async () => {
    install({ ...STATUS, configured: false, available: false, reason: "Baseten deployment is not configured." }, []);
    render(<CloudDiagnosticPanel />);
    const button = await screen.findByRole("button", { name: "Run cloud model" });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-describedby", "cloud-disabled-reason");
    expect(screen.getAllByText(/Baseten deployment is not configured/i)).toHaveLength(2);
  });

  it("labels configured readiness without claiming a remote GPU is reachable, and rejects protocol-relative fixture URLs", async () => {
    install({
      ...STATUS,
      available: undefined,
      reason: "A configured deployment has no live runtime probe.",
      fixture: { current_url: "//untrusted.example/current.png", goal_url: "//untrusted.example/goal.png" },
    }, []);
    render(<CloudDiagnosticPanel />);
    await screen.findByRole("heading", { name: "Cloud diagnostic" });
    expect(screen.getByText("configured")).toBeInTheDocument();
    expect(screen.getByText(/does not probe GPU or remote reachability/i)).toBeInTheDocument();
    expect(screen.getAllByText(/A configured deployment has no live runtime probe/i)).toHaveLength(2);
    expect(screen.getByText(/Current fixture frame is not available/i)).toBeInTheDocument();
    expect(screen.getByText(/Goal fixture frame is not available/i)).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("keeps a failed record disabled until a reviewer explicitly acknowledges it", async () => {
    const failed = {
      request_id: "cloud-failed",
      status: "failed",
      qualified: false,
      diagnostic_only: true,
      error: { kind: "transport", message: "Platform callback was not reconciled.", manual_reconciliation_required: true, automatic_retry_allowed: false },
    };
    const { calls } = install(STATUS, [failed]);
    render(<CloudDiagnosticPanel />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/manual reconciliation/i);
    expect(screen.getByRole("button", { name: "Run new diagnostic" })).toBeDisabled();
    expect(screen.getByLabelText("I reviewed the previous failure; create a new request")).not.toBeChecked();
    expect(calls.some((call) => call.method === "POST")).toBe(false);
  });

  it("creates a new request only after manual acknowledgement, preserving the failed record", async () => {
    const failed = {
      request_id: "cloud-failed",
      status: "failed",
      qualified: false,
      diagnostic_only: true,
      error: { kind: "transport", message: "Platform callback was not reconciled.", manual_reconciliation_required: true, automatic_retry_allowed: false },
    };
    const { calls } = install(STATUS, [failed]);
    render(<CloudDiagnosticPanel />);
    await screen.findByRole("alert");
    const user = userEvent.setup();
    await user.click(screen.getByLabelText("I reviewed the previous failure; create a new request"));
    const button = screen.getByRole("button", { name: "Run new diagnostic" });
    expect(button).toBeEnabled();
    expect(screen.getByText(/may incur additional compute/i)).toBeInTheDocument();
    await user.click(button);
    await waitFor(() => expect(calls.some((call) => call.method === "POST")).toBe(true));
    const post = calls.find((call) => call.method === "POST");
    expect(JSON.parse(post?.body ?? "{}")).toEqual({});
    expect(screen.getAllByText("cloud-failed").length).toBeGreaterThan(0);
    expect(screen.getAllByText("cloud-new").length).toBeGreaterThan(0);
  });

  it("resets acknowledgement when another historical record is selected", async () => {
    const first = { request_id: "cloud-failed-a", status: "failed", qualified: false, diagnostic_only: true };
    const second = { request_id: "cloud-failed-b", status: "ambiguous", qualified: false, diagnostic_only: true };
    install(STATUS, [first, second]);
    render(<CloudDiagnosticPanel />);
    await screen.findByRole("alert");
    const user = userEvent.setup();
    const acknowledgement = screen.getByLabelText("I reviewed the previous failure; create a new request");
    await user.click(acknowledgement);
    expect(acknowledgement).toBeChecked();
    await user.click(screen.getByRole("button", { name: /cloud-failed-b/i }));
    expect(screen.getByLabelText("I reviewed the previous failure; create a new request")).not.toBeChecked();
  });

  it("treats an ambiguous post-restart record as manual reconciliation, not a runnable state", async () => {
    const ambiguous = {
      request_id: "cloud-ambiguous",
      status: "ambiguous",
      qualified: false,
      diagnostic_only: true,
    };
    const { calls } = install(STATUS, [ambiguous]);
    render(<CloudDiagnosticPanel />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/state is ambiguous after restart/i);
    expect(screen.getAllByText("ambiguous").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Run new diagnostic" })).toBeDisabled();
    expect(calls.some((call) => call.method === "POST")).toBe(false);
  });

  it("truncates only the visible history identifier while retaining its full accessible name and title", async () => {
    const requestId = "cloud-e2dd4f395307440bae93300840adc632";
    install(STATUS, [{ ...COMPLETED, request_id: requestId }]);
    render(<CloudDiagnosticPanel />);
    const row = await screen.findByRole("button", { name: `Open cloud diagnostic ${requestId}, completed` });
    const visible = row.querySelector(`[title="${requestId}"]`);
    expect(visible).toHaveTextContent("cloud-e2dd…40adc632");
  });

  it("has no axe violations", async () => {
    install();
    const { container } = render(<CloudDiagnosticPanel />);
    await screen.findByRole("heading", { name: "Cloud diagnostic" });
    let result!: axe.AxeResults;
    await act(async () => {
      result = await axe.run(container, { preload: false, rules: { "color-contrast": { enabled: false } } });
    });
    expect(result.violations).toEqual([]);
  });
});
