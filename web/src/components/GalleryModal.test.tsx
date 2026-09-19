import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { GalleryModal } from "./GalleryModal";

function Harness({ frames = ["a/0.png", "a/1.png"] }: { frames?: string[] }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Enlarge tile
      </button>
      <button type="button">Somewhere else</button>
      <GalleryModal
        open={open}
        onClose={() => setOpen(false)}
        title="OpenVLA · close_drawer"
        subtitle="ep-1"
        frames={frames}
        emptyReason="No persisted segment event has reached this slot"
        meta={[
          { label: "Run", value: "run-1" },
          { label: "Frames", value: "14 certified" },
        ]}
        footer={
          <button type="button" className="button">
            Take the controls
          </button>
        }
      />
    </>
  );
}

function open() {
  fireEvent.click(screen.getByRole("button", { name: "Enlarge tile" }));
  return screen.getByRole("dialog");
}

describe("<GalleryModal />", () => {
  it("renders nothing until it is opened", () => {
    render(<Harness />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("mounts to document.body rather than inside the page", () => {
    render(<Harness />);
    open();
    const overlay = document.querySelector(".viewer-overlay");
    // Mirage's commit db98e0f fixed exactly this: rendered in place, the landing's
    // stacking context painted over the dialog and swallowed its clicks.
    expect(overlay?.parentElement).toBe(document.body);
  });

  it("is a modal dialog with an accessible name taken from its heading", () => {
    render(<Harness />);
    const dialog = open();
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAccessibleName("OpenVLA · close_drawer");
  });

  it("moves focus into the dialog on open", () => {
    render(<Harness />);
    open();
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Close viewer" }));
  });

  it("traps Tab inside the dialog at both ends", () => {
    render(<Harness />);
    open();
    const close = screen.getByRole("button", { name: "Close viewer" });
    const last = screen.getByRole("button", { name: "Take the controls" });

    // Shift+Tab off the first control wraps to the last, not out to the page.
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(last);

    // Tab off the last control wraps back to the first.
    fireEvent.keyDown(document, { key: "Tab" });
    expect(document.activeElement).toBe(close);
  });

  it("pulls focus back in if it has escaped to the page behind", () => {
    render(<Harness />);
    open();
    screen.getByRole("button", { name: "Somewhere else" }).focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Close viewer" }));
  });

  it("closes on Escape", () => {
    render(<Harness />);
    open();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("restores focus to whatever opened it", () => {
    render(<Harness />);
    const trigger = screen.getByRole("button", { name: "Enlarge tile" });
    trigger.focus();
    open();
    expect(document.activeElement).not.toBe(trigger);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(document.activeElement).toBe(trigger);
  });

  it("closes on a backdrop press that both starts and ends on the backdrop", () => {
    render(<Harness />);
    open();
    const overlay = document.querySelector(".viewer-overlay") as HTMLElement;
    fireEvent.mouseDown(overlay);
    fireEvent.mouseUp(overlay);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("does not close when a drag begins inside the dialog and ends on the backdrop", () => {
    render(<Harness />);
    const dialog = open();
    const overlay = document.querySelector(".viewer-overlay") as HTMLElement;
    fireEvent.mouseDown(dialog);
    fireEvent.mouseUp(overlay);
    expect(screen.getByRole("dialog")).toBe(dialog);
  });

  it("shows the persisted frames and the record's own metadata", () => {
    render(<Harness />);
    open();
    const image = screen.getByAltText(/Accumulated persisted frames for OpenVLA · close_drawer/i);
    expect(image).toHaveAttribute("src", "a/0.png");
    expect(screen.getByText("run-1")).toBeInTheDocument();
    expect(screen.getByText("14 certified")).toBeInTheDocument();
    expect(screen.getByText("ep-1")).toBeInTheDocument();
  });

  it("states why it is empty rather than showing a stand-in image", () => {
    render(<Harness frames={[]} />);
    open();
    expect(screen.getByText(/No persisted segment event has reached this slot/)).toBeInTheDocument();
    expect(document.querySelectorAll(".viewer-stage img")).toHaveLength(0);
  });

  it("says how to close itself", () => {
    render(<Harness />);
    open();
    expect(screen.getByText("Escape closes")).toBeInTheDocument();
  });
});
