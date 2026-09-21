import { StrictMode } from "react";
import { act, render, screen, waitFor } from "@testing-library/react";
import axe from "axe-core";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { deferred, http, mockFetch } from "../test/harness";
import { DevelopmentReview } from "./DevelopmentReview";

const clips = {
  set_id: "development-a",
  progress: { saved_drafts: 0, total: 2 },
  clips: [{
    opaque_clip_id: "review://clip/opaque-a",
    task: {
      id: "open_drawer",
      instruction: "Open the drawer",
      rubric: "Mark only visibly supported progress.",
      definitions: {
        integrity: { intact: "Visible media is intact.", artifact: "Visible artifact.", uncertain: "Cannot determine." },
        collision: { none_visible: "No collision is visible.", visible: "Collision is visible.", uncertain: "Cannot determine." },
        completion: "Use only visible completion evidence.",
      },
      progress_definitions: ["0: no directed approach."],
    },
    media: {
      video_url: "/review/opaque-a.mp4",
      frames: Array.from({ length: 16 }, (_, index) => ({ index, url: `/review/opaque-a-${index}.png`, timestamp: index === 0 ? 0 : index === 1 ? 0.400000005 : `00:00:${String(index).padStart(2, "0")}` })),
    },
  }, {
    opaque_clip_id: "review://clip/opaque-b",
    task: { id: "close_drawer", instruction: "Close the drawer", rubric: "Mark only visibly supported progress." },
    media: {
      video_url: "/review/opaque-b.mp4",
      frames: Array.from({ length: 16 }, (_, index) => ({ index, url: `/review/opaque-b-${index}.png`, timestamp: index })),
    },
  }],
};

function installRoutes(
  ratingResponse: unknown | ((init: RequestInit | undefined) => unknown) = (init: RequestInit | undefined) => ({ draft: JSON.parse(String(init?.body ?? "{}")) }),
  clipResponse: unknown = clips,
) {
  return mockFetch({
    "/api/development-review/sets/development-a/clips": clipResponse,
    "/api/development-review/sessions/session-a/ratings/review%3A%2F%2Fclip%2Fopaque-a": ratingResponse,
    "/api/development-review/sessions": { session_id: "session-a", reviewer_id: "reviewer-a", reviewer_kind: "human_self_reported" },
    "/api/development-review/sets": { sets: [{ set_id: "development-a", title: "Drawer review", clip_count: 1, status: "development_review_only" }] },
  });
}

async function openReview(ratingResponse?: unknown | ((init: RequestInit | undefined) => unknown)) {
  const routes = installRoutes(ratingResponse);
  const view = render(<DevelopmentReview />);
  const user = userEvent.setup();
  await user.selectOptions(await screen.findByLabelText("Development review set"), "development-a");
  await user.type(screen.getByLabelText("Reviewer ID"), "reviewer-a");
  await user.click(screen.getByRole("button", { name: "Start review" }));
  await screen.findByRole("heading", { name: "Clip 1 of 2" });
  return { user, ...routes, ...view };
}

describe("<DevelopmentReview />", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("starts blank, submits only the reviewer-selected development draft, and keeps source labels absent", async () => {
    const { calls, user } = await openReview();
    expect(screen.getByText(/development-review-only/i)).toBeInTheDocument();
    expect(screen.queryByText(/No placeholder reviewer, rating, or calibration status/i)).not.toBeInTheDocument();
    expect(screen.getAllByRole("img")).toHaveLength(16);
    expect(screen.getByAltText("Evidence frame 0 at 0.000 s")).toBeInTheDocument();
    expect(screen.getByText("0.400 s")).toHaveAttribute("title", "0.400000005");
    expect(screen.getByLabelText("Review video for clip 1")).toHaveAttribute("poster", "/review/opaque-a-0.png");
    expect(screen.getByText("Integrity — intact: Visible media is intact.")).toBeInTheDocument();
    expect(screen.queryByText("source-episode-0001")).not.toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Integrity"), "intact");
    await user.selectOptions(screen.getByLabelText("Progress"), "2");
    await user.click(screen.getByLabelText(/Frame 3/i));
    await user.type(screen.getByLabelText("Observable reason"), "Visible contact at the drawer handle.");
    await user.click(screen.getByRole("button", { name: "Save draft" }));

    await waitFor(() => expect(calls.some((call) => call.method === "PUT")).toBe(true));
    const save = calls.find((call) => call.method === "PUT");
    expect(JSON.parse(save?.body ?? "{}")).toEqual({ integrity: "intact", collision: null, progress: 2, completion_evidence: null, evidence_frame_indices: [3], observable_reason: "Visible contact at the drawer handle." });
    expect(screen.getByRole("status")).toHaveTextContent(/Your draft was saved/i);
    expect(screen.getByText("saved drafts 1 / 2")).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Integrity"), "");
    await user.selectOptions(screen.getByLabelText("Progress"), "");
    await user.click(screen.getByLabelText(/Frame 3/i));
    await user.clear(screen.getByLabelText("Observable reason"));
    await user.click(screen.getByRole("button", { name: "Save draft" }));
    await waitFor(() => expect(calls.filter((call) => call.method === "PUT")).toHaveLength(2));
    expect(JSON.parse(calls.filter((call) => call.method === "PUT")[1].body ?? "{}")).toEqual({ integrity: null, collision: null, progress: null, completion_evidence: null, evidence_frame_indices: [], observable_reason: "" });
    expect(screen.getByText("saved drafts 0 / 2")).toBeInTheDocument();
  });

  it("keeps a local unsaved draft when reviewers navigate backward and clears the prior saved status", async () => {
    const { user } = await openReview();
    await user.type(screen.getByLabelText("Observable reason"), "Visible at frame zero.");
    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByRole("heading", { name: "Clip 2 of 2" });
    await user.click(screen.getByRole("button", { name: "Previous" }));
    await screen.findByRole("heading", { name: "Clip 1 of 2" });
    expect(screen.getByLabelText("Observable reason")).toHaveValue("Visible at frame zero.");
    expect(screen.queryByText("Your draft was saved.")).not.toBeInTheDocument();
  });

  it("links validation errors to the blank required setup fields", async () => {
    installRoutes();
    render(<DevelopmentReview />);
    const user = userEvent.setup();
    await screen.findByLabelText("Development review set");
    await user.click(screen.getByRole("button", { name: "Start review" }));
    const error = await screen.findByRole("alert");
    expect(error).toHaveAttribute("id", "session-error");
    expect(screen.getByLabelText("Development review set")).toHaveAttribute("aria-describedby", "session-error");
    expect(screen.getByLabelText("Reviewer ID")).toHaveAttribute("aria-describedby", "session-error");
  });

  it("links a server validation error to every rating field without fabricating a saved draft", async () => {
    const { user } = await openReview(http(422, { detail: "Progress is inconsistent with completion evidence." }));
    await user.click(screen.getByRole("button", { name: "Save draft" }));
    const error = await screen.findByRole("alert");
    expect(error).toHaveAttribute("id", "save-error");
    expect(screen.getByLabelText("Progress")).toHaveAttribute("aria-describedby", "save-error");
    expect(screen.getByLabelText("Observable reason")).toHaveAttribute("aria-describedby", "reason-help save-error");
    expect(screen.getByText("saved drafts 0 / 2")).toBeInTheDocument();
  });

  it("resumes an authorized local review after a browser refresh and restores only its own saved draft", async () => {
    const first = await openReview();
    expect(JSON.parse(window.sessionStorage.getItem("kosmos:development-review:session-v1") ?? "{}")).toEqual({
      session_id: "session-a",
      set_id: "development-a",
      reviewer_id: "reviewer-a",
      reviewer_kind: "human_self_reported",
    });
    first.unmount();

    const resumedClips = {
      ...clips,
      clips: [{ ...clips.clips[0], draft: { progress: 0 } }, clips.clips[1]],
    };
    const resumed = installRoutes(undefined, resumedClips);
    render(<DevelopmentReview />);
    await screen.findByRole("heading", { name: "Clip 1 of 2" });
    expect(screen.getByLabelText("Progress")).toHaveValue("0");
    expect(resumed.calls.some((call) => call.url.includes("/sets/development-a/clips?session_id=session-a"))).toBe(true);
    expect(resumed.calls.some((call) => call.method === "POST")).toBe(false);
  });

  it("does not cancel a deferred resume request when the sets-loaded state changes in StrictMode", async () => {
    window.sessionStorage.setItem("kosmos:development-review:session-v1", JSON.stringify({
      session_id: "session-a",
      set_id: "development-a",
      reviewer_id: "reviewer-a",
      reviewer_kind: "human_self_reported",
    }));
    const gate = deferred<unknown>();
    const routes = installRoutes(undefined, () => gate.promise);
    render(<StrictMode><DevelopmentReview /></StrictMode>);
    await screen.findByText(/Resuming your saved development review session/i);
    gate.resolve(clips);
    await screen.findByRole("heading", { name: "Clip 1 of 2" });
    expect(routes.calls.filter((call) => call.url.includes("/sets/development-a/clips?session_id=session-a"))).toHaveLength(1);
    expect(routes.calls.some((call) => call.method === "POST")).toBe(false);
  });

  it("clears expired session metadata and returns to the setup form with a useful error", async () => {
    window.sessionStorage.setItem("kosmos:development-review:session-v1", JSON.stringify({
      session_id: "expired-session",
      set_id: "development-a",
      reviewer_id: "reviewer-a",
      reviewer_kind: "human_self_reported",
    }));
    installRoutes(undefined, http(403, { detail: "review_session_expired" }));
    render(<DevelopmentReview />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/saved development review session has expired/i);
    expect(window.sessionStorage.getItem("kosmos:development-review:session-v1")).toBeNull();
    expect(screen.getByRole("heading", { name: "Start a review session" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /Clip 1 of/i })).not.toBeInTheDocument();
  });

  it("uses native form controls that are keyboard reachable and has no axe violations", async () => {
    const { user } = await openReview();
    await user.tab();
    expect(screen.getByRole("link", { name: "Return to evaluation console" })).toHaveFocus();
    await user.tab();
    expect(screen.getByLabelText("Integrity")).toHaveFocus();

    let result!: axe.AxeResults;
    await act(async () => {
      result = await axe.run(document.body, { preload: false, rules: { "color-contrast": { enabled: false }, "video-caption": { enabled: false } } });
    });
    expect(result.violations).toEqual([]);
  });
});
