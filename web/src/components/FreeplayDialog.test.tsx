import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { deferred, http, mockFetch } from "../test/harness";
import { FreeplayDialog } from "./FreeplayDialog";

const OK_CHUNK = {
  session_id: "sess-1",
  frame_urls: ["free/0.png", "free/1.png", "free/2.png"],
  frame_count: 16,
  latency_ms: 2_040,
  generating: false,
  qualified: false,
  backend: "irasim@c72b6da",
  requested_resolution: 480,
  frame_height: 480,
  frame_width: 480,
  action_clamp: 0.03,
  chunk_size: 16,
  reason: "Unscored free-play; no qualified world fidelity is claimed.",
};

function open() {
  render(<FreeplayDialog open onOpenChange={() => undefined} />);
}

// `{Key>}` presses without releasing. A bare `{Key}` would also fire keyup,
// which is release-to-stop and would send a second, "stop" command.

describe("<FreeplayDialog />", () => {
  const recording = {
    id: "clip-1",
    title: "Saved experiment",
    model: "Cosmos",
    kind: "smoke",
    video_url: "/api/artifacts/saved.mp4",
    sha256: "sha256:source",
    qualified: false as const,
    provenance: "recorded_model_output" as const,
  };

  it("binds an explicitly supported recording to every command and does not generate on open", async () => {
    const { calls } = mockFetch({
      "/api/freeplay/status": {
        available: true,
        mode: "recording_branch",
        binding: { exact_branch_supported: true },
        reason: "Test backend with explicit checkpoint binding.",
        source: { video_id: recording.id, sha256: recording.sha256 },
      },
      "/api/freeplay/step": {
        ...OK_CHUNK,
        mode: "recording_branch",
        source: { video_id: recording.id, sha256: recording.sha256 },
        binding: { exact_branch_supported: true },
      },
    });
    render(
      <FreeplayDialog
        open
        recording={recording}
        onOpenChange={() => undefined}
      />,
    );
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Command the arm up" }),
      ).toBeEnabled(),
    );
    expect(calls.every((call) => call.method === "GET")).toBe(true);
    await userEvent.keyboard("{ArrowUp>}");
    await waitFor(() =>
      expect(calls.filter((call) => call.method === "POST")).toHaveLength(1),
    );
    const submitted = JSON.parse(
      calls.find((call) => call.method === "POST")?.body ?? "{}",
    );
    expect(submitted).toEqual({
      direction: "up",
      video_id: recording.id,
      source_sha256: recording.sha256,
    });
  });

  it("does not treat a different source identity as permission to generate", async () => {
    const { calls } = mockFetch({
      "/api/freeplay/status": {
        available: true,
        mode: "recording_branch",
        binding: { exact_branch_supported: true },
        reason: "Ready",
        source: { video_id: "other-video", sha256: recording.sha256 },
      },
    });
    render(
      <FreeplayDialog
        open
        recording={recording}
        onOpenChange={() => undefined}
      />,
    );
    expect(
      await screen.findByText(/did not confirm this recording’s exact source/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Command the arm up" }),
    ).toBeDisabled();
    await userEvent.keyboard("{ArrowUp}");
    expect(calls.every((call) => call.method === "GET")).toBe(true);
  });

  it("keeps video-seeking arrow keys out of generation commands", async () => {
    const { calls } = mockFetch({
      "/api/freeplay/status": {
        available: true,
        mode: "recording_branch",
        binding: { exact_branch_supported: true },
        reason: "Ready",
        source: { video_id: recording.id, sha256: recording.sha256 },
      },
    });
    render(
      <FreeplayDialog
        open
        recording={recording}
        onOpenChange={() => undefined}
      />,
    );
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Command the arm up" }),
      ).toBeEnabled(),
    );
    fireEvent.keyDown(
      screen.getByLabelText("Original recording: Saved experiment"),
      { key: "ArrowRight" },
    );
    expect(calls.every((call) => call.method === "GET")).toBe(true);
  });

  it("keeps controls disabled on a failed readiness check and supports a read-only retry", async () => {
    const { calls } = mockFetch({ "/api/freeplay/status": http(503, {}) });
    render(
      <FreeplayDialog
        open
        recording={recording}
        onOpenChange={() => undefined}
      />,
    );
    expect(
      await screen.findByText(/Could not verify interactive availability/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Command the arm up" }),
    ).toBeDisabled();
    fireEvent.click(
      screen.getByRole("button", { name: "Check availability again" }),
    );
    await waitFor(() => expect(calls).toHaveLength(2));
    expect(calls.every((call) => call.method === "GET")).toBe(true);
  });

  it("drops queued commands when the controls are closed during generation", async () => {
    const response = deferred<unknown>();
    const { calls } = mockFetch({
      "/api/freeplay/step": () => response.promise,
    });
    const view = render(<FreeplayDialog open onOpenChange={() => undefined} />);
    await userEvent.keyboard("{ArrowRight>}{ArrowLeft>}");
    await waitFor(() => expect(calls).toHaveLength(1));
    view.unmount();
    await act(async () => {
      response.resolve(OK_CHUNK);
    });
    expect(calls).toHaveLength(1);
  });

  it("sends one direction per keypress, not a seven-element action vector", async () => {
    const { calls } = mockFetch({ "/api/freeplay/step": OK_CHUNK });
    open();
    await userEvent.keyboard("{ArrowRight>}");
    await waitFor(() => expect(calls).toHaveLength(1));
    expect(JSON.parse(calls[0].body ?? "{}")).toEqual({ direction: "right" });
  });

  it("sends a stop command when the key is released", async () => {
    const { calls } = mockFetch({ "/api/freeplay/step": OK_CHUNK });
    open();
    await userEvent.keyboard("{ArrowUp>}{/ArrowUp}");
    await waitFor(() => expect(calls.length).toBeGreaterThanOrEqual(2));
    const directions = calls.map(
      (call) => JSON.parse(call.body ?? "{}").direction,
    );
    expect(directions).toContain("up");
    expect(directions).toContain("stop");
  });

  it("reuses the session the server returned", async () => {
    const { calls } = mockFetch({ "/api/freeplay/step": OK_CHUNK });
    open();
    await userEvent.keyboard("{ArrowRight>}");
    await waitFor(() => expect(calls).toHaveLength(1));
    await userEvent.keyboard("{ArrowLeft>}");
    await waitFor(() => expect(calls).toHaveLength(2));
    expect(JSON.parse(calls[1].body ?? "{}")).toEqual({
      session_id: "sess-1",
      direction: "left",
    });
  });

  it("shows a real generating state across the wait rather than hiding it", async () => {
    const gate = deferred<unknown>();
    mockFetch({ "/api/freeplay/step": () => gate.promise as never });
    open();
    await userEvent.keyboard("{ArrowRight>}");

    const status = await screen.findByRole("status");
    expect(status).toHaveTextContent(/Generating/);
    expect(status).toHaveTextContent(
      /Inventing the next chunk of video from the right command/i,
    );
    expect(status).toHaveTextContent(/s elapsed/);
    expect(status).toHaveTextContent(/client wall time/);

    gate.resolve(OK_CHUNK);
    await waitFor(() =>
      expect(screen.queryByText("Generating")).not.toBeInTheDocument(),
    );
  });

  it("renders the returned frames and the server's own latency", async () => {
    mockFetch({ "/api/freeplay/step": OK_CHUNK });
    open();
    await userEvent.keyboard("{ArrowRight>}");
    expect(
      await screen.findByAltText(/Generated frames for the right command/i),
    ).toBeInTheDocument();
    expect(screen.getByText("2,040 ms")).toBeInTheDocument();
    expect(screen.getByText("Frames returned")).toBeInTheDocument();
    expect(screen.getByText("Actions per chunk")).toBeInTheDocument();
    // frame_count and chunk_size are both 16 in this fixture.
    expect(screen.getAllByText("16")).toHaveLength(2);
    expect(screen.getByText("480x480")).toBeInTheDocument();
    expect(screen.getByText("irasim@c72b6da")).toBeInTheDocument();
  });

  it("prints only the clamp the server declared", async () => {
    mockFetch({ "/api/freeplay/step": OK_CHUNK });
    open();
    await userEvent.keyboard("{ArrowRight>}");
    expect(await screen.findByText("±0.03")).toBeInTheDocument();
  });

  it("says the clamp was not reported rather than asserting a client constant", async () => {
    mockFetch({
      "/api/freeplay/step": {
        ...OK_CHUNK,
        action_clamp: undefined,
        chunk_size: undefined,
      },
    });
    open();
    await userEvent.keyboard("{ArrowRight>}");
    await waitFor(() =>
      expect(screen.getAllByText("not reported").length).toBe(2),
    );
    expect(screen.queryByText(/0\.08/)).not.toBeInTheDocument();
  });

  it("handles a 503 with the server's named reason and shows no frame", async () => {
    mockFetch({
      "/api/freeplay/step": http(503, {
        reason: "no_certified_world_backend_configured",
      }),
    });
    open();
    await userEvent.keyboard("{ArrowRight>}");

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      /No certified world backend for free-play/i,
    );
    expect(alert).toHaveTextContent(/no_certified_world_backend_configured/);
    expect(alert).toHaveTextContent(/A placeholder here would be a claim/i);
    expect(screen.queryByAltText(/Generated frames/i)).not.toBeInTheDocument();
  });

  it("reports an empty frame list instead of pretending a chunk arrived", async () => {
    mockFetch({
      "/api/freeplay/step": {
        ...OK_CHUNK,
        frame_urls: [],
        reason: "world_call_returned_no_frames",
      },
    });
    open();
    await userEvent.keyboard("{ArrowRight>}");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /world_call_returned_no_frames/,
    );
  });

  it("never claims the free-play surface is scored", () => {
    mockFetch({ "/api/freeplay/step": OK_CHUNK });
    open();
    expect(
      screen.getByText(/contributes nothing to the scoreboard/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Nothing here enters a scored comparison/i),
    ).toBeInTheDocument();
    expect(screen.getByText("unqualified")).toBeInTheDocument();
  });

  it("keeps the no-backend message when the trailing stop call succeeds", async () => {
    // `stop` commands nothing, so the control plane answers 200 even with no
    // world backend. Pressing and releasing an arrow must not clear the
    // explanation and leave a blank stage.
    mockFetch({
      "/api/freeplay/step": (init: RequestInit | undefined) => {
        const body = JSON.parse(String(init?.body ?? "{}"));
        return body.direction === "stop"
          ? {
              session_id: "s",
              direction: "stop",
              frame_urls: [],
              frame_count: 0,
              commanded_rows: 0,
              reason: "Release-to-stop: no action was commanded.",
            }
          : http(503, { reason: "no_certified_world_backend_configured" });
      },
    });
    open();
    await userEvent.keyboard("{ArrowRight>}{/ArrowRight}");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      /No certified world backend for free-play/i,
    );
    expect(alert).toHaveTextContent(/no_certified_world_backend_configured/);
  });

  it("does not blank a rendered clip when the key is released", async () => {
    mockFetch({
      "/api/freeplay/step": (init: RequestInit | undefined) => {
        const body = JSON.parse(String(init?.body ?? "{}"));
        return body.direction === "stop"
          ? {
              session_id: "sess-1",
              direction: "stop",
              frame_urls: [],
              frame_count: 0,
              commanded_rows: 0,
            }
          : OK_CHUNK;
      },
    });
    open();
    await userEvent.keyboard("{ArrowRight>}{/ArrowRight}");
    expect(
      await screen.findByAltText(/Generated frames for the right command/i),
    ).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("queues at most the newest command while one is generating", async () => {
    const gate = deferred<unknown>();
    let served = 0;
    const { calls } = mockFetch({
      "/api/freeplay/step": () => {
        served += 1;
        return served === 1 ? (gate.promise as never) : OK_CHUNK;
      },
    });
    open();
    await userEvent.keyboard("{ArrowRight>}");
    await waitFor(() => expect(calls).toHaveLength(1));

    await userEvent.keyboard("{ArrowUp>}");
    await userEvent.keyboard("{ArrowDown>}");
    await userEvent.keyboard("{ArrowLeft>}");
    // Still one in flight: the extra presses did not each open a request.
    expect(calls).toHaveLength(1);
    expect(
      await screen.findByText(
        /most recent one is queued; the rest were dropped/i,
      ),
    ).toBeInTheDocument();

    gate.resolve(OK_CHUNK);
    await waitFor(() => expect(calls).toHaveLength(2));
    expect(JSON.parse(calls[1].body ?? "{}").direction).toBe("left");
  });

  it("captions the frames' measured size, not the size that was requested", async () => {
    // The 480p request is answered with 64px frames. Printing "480x480" here
    // would caption a rehearsal frame with a resolution nothing produced, which
    // is the one claim this panel makes on stage.
    mockFetch({
      "/api/freeplay/step": {
        ...OK_CHUNK,
        requested_resolution: 480,
        frame_height: 64,
        frame_width: 64,
      },
    });
    open();
    await userEvent.keyboard("{ArrowRight>}");
    expect(await screen.findByText("64x64")).toBeInTheDocument();
    expect(screen.queryByText("480x480")).not.toBeInTheDocument();
  });

  it("says so when the Chain reports no frame dimensions", async () => {
    mockFetch({
      "/api/freeplay/step": {
        ...OK_CHUNK,
        frame_height: null,
        frame_width: null,
      },
    });
    open();
    await userEvent.keyboard("{ArrowRight>}");
    expect(
      await screen.findByText("resolution not reported"),
    ).toBeInTheDocument();
  });
});
