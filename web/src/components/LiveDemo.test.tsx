import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { LiveDemoPanel } from "./LiveDemo";
import { mockFetch } from "../test/harness";

const status = {
  available: true,
  configured: true,
  health: "ready",
  qualified: false,
  state_mode: "experimental_reencoded_rgb_stateless",
};

const manual = {
  id: "manual-1",
  title: "Manual test",
  prompt: "Put the pot to the left of the purple item.",
  mode: "manual" as const,
  steps: 1,
  policy_label: "Manual directional action",
  state: "ready" as const,
  frames: [{ index: 0, url: "demo/manual/0.png", role: "source" as const }],
  latest_video_url: "demo/manual/recording.mp4",
  qualified: false as const,
  state_mode: "experimental_reencoded_rgb_stateless",
};

describe("<LiveDemoPanel />", () => {
  it("starts a bounded manual session and sends one click-bounded command", async () => {
    const { calls } = mockFetch({
      "/api/demo/status": status,
      "/api/demo/sessions/manual-1/commands": {
        ...manual,
        state: "running",
        frames: [
          ...manual.frames,
          {
            index: 1,
            url: "demo/manual/1.png",
            role: "predicted",
            timings_ms: { worker_step: 1732 },
            model: "IRASim",
            revision: "test-revision",
          },
        ],
      },
      "/api/demo/sessions/manual-1": manual,
      "/api/demo/sessions": (init: RequestInit | undefined) => {
        if ((init?.method ?? "GET") === "POST") return manual;
        return { sessions: [] };
      },
    });

    render(<LiveDemoPanel />);
    await screen.findByText(/No saved live evaluations yet\./);
    fireEvent.click(screen.getByRole("button", { name: "New evaluation" }));
    await screen.findByRole("dialog", { name: "New evaluation" });
    await userEvent.click(
      screen.getByRole("radio", { name: /Steer manually/i }),
    );
    expect(
      screen.getByText(/do not issue a movement until you click/i),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Start manual session" }),
    );

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Move up" })).toBeEnabled(),
    );
    const sessionPost = calls.find(
      (call) => call.url === "/api/demo/sessions" && call.method === "POST",
    );
    expect(JSON.parse(sessionPost?.body ?? "{}")).toEqual({
      prompt: "Put the pot to the left of the purple item.",
      mode: "manual",
      steps: 1,
    });
    expect(calls.filter((call) => call.method === "POST")).toHaveLength(1);

    expect(
      await screen.findByAltText("Latest persisted frame for Manual test"),
    ).toHaveAttribute("src", "/api/artifacts/demo/manual/0.png");
    expect(
      screen.getByRole("link", { name: "Download recording" }),
    ).toHaveAttribute("href", "/api/artifacts/demo/manual/recording.mp4");

    fireEvent.click(screen.getByRole("button", { name: "Move up" }));
    await waitFor(() =>
      expect(calls.filter((call) => call.method === "POST")).toHaveLength(2),
    );
    const command = calls.find((call) => call.url.endsWith("/commands"));
    expect(JSON.parse(command?.body ?? "{}")).toMatchObject({
      direction: "up",
      steps: 1,
    });
  });

  it("uses four bounded steps by default for the existing OpenVLA controller", async () => {
    const policy = {
      ...manual,
      id: "policy-1",
      mode: "policy" as const,
      steps: 4,
      policy_label: "OpenVLA",
      state: "queued" as const,
    };
    const { calls } = mockFetch({
      "/api/demo/status": status,
      "/api/demo/sessions/policy-1": policy,
      "/api/demo/sessions": (init: RequestInit | undefined) => {
        if ((init?.method ?? "GET") === "POST") return policy;
        return { sessions: [] };
      },
    });
    render(<LiveDemoPanel />);
    await screen.findByText(/No saved live evaluations yet\./);
    fireEvent.click(screen.getByRole("button", { name: "New evaluation" }));
    await screen.findByRole("dialog", { name: "New evaluation" });
    fireEvent.click(
      screen.getByRole("button", { name: "Run OpenVLA evaluation" }),
    );
    await waitFor(() =>
      expect(calls.filter((call) => call.method === "POST")).toHaveLength(1),
    );
    const request = calls.find((call) => call.method === "POST");
    expect(JSON.parse(request?.body ?? "{}")).toMatchObject({
      mode: "policy",
      steps: 4,
      prompt: "Put the pot to the left of the purple item.",
    });
  });
});
