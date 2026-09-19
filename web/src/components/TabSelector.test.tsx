import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { TabSelector, type TabItem } from "./TabSelector";

const TABS: TabItem<"console" | "gallery" | "gates">[] = [
  { id: "console", label: "Console" },
  { id: "gallery", label: "Gallery" },
  { id: "gates", label: "Gates" },
];

function Harness({ initial = "console" as "console" | "gallery" | "gates" }) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <TabSelector tabs={TABS} value={value} onChange={setValue} label="Landing sections" idPrefix="t" />
      <div id={`t-panel-${value}`} role="tabpanel" aria-labelledby={`t-tab-${value}`}>
        {value} panel
      </div>
    </>
  );
}

describe("<TabSelector />", () => {
  it("exposes a tablist with one selected tab", () => {
    render(<Harness />);
    expect(screen.getByRole("tablist", { name: "Landing sections" })).toBeInTheDocument();
    expect(screen.getAllByRole("tab")).toHaveLength(3);
    expect(screen.getByRole("tab", { name: "Console" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Gallery" })).toHaveAttribute("aria-selected", "false");
  });

  it("keeps the whole widget to one tab stop", () => {
    render(<Harness />);
    expect(screen.getByRole("tab", { name: "Console" })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("tab", { name: "Gallery" })).toHaveAttribute("tabindex", "-1");
    expect(screen.getByRole("tab", { name: "Gates" })).toHaveAttribute("tabindex", "-1");
  });

  it("points aria-controls only at a panel that exists", () => {
    render(<Harness />);
    const selected = screen.getByRole("tab", { name: "Console" });
    const controls = selected.getAttribute("aria-controls");
    expect(controls).toBe("t-panel-console");
    expect(document.getElementById(controls!)).not.toBeNull();
    expect(screen.getByRole("tab", { name: "Gallery" })).not.toHaveAttribute("aria-controls");
  });

  it("moves selection with the arrow keys and wraps at both ends", () => {
    render(<Harness />);
    const list = screen.getByRole("tablist");
    fireEvent.keyDown(list, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Gallery" })).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(list, { key: "ArrowRight" });
    fireEvent.keyDown(list, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Console" })).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(list, { key: "ArrowLeft" });
    expect(screen.getByRole("tab", { name: "Gates" })).toHaveAttribute("aria-selected", "true");
  });

  it("jumps to the ends with Home and End", () => {
    render(<Harness />);
    const list = screen.getByRole("tablist");
    fireEvent.keyDown(list, { key: "End" });
    expect(screen.getByRole("tab", { name: "Gates" })).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(list, { key: "Home" });
    expect(screen.getByRole("tab", { name: "Console" })).toHaveAttribute("aria-selected", "true");
  });

  it("moves the caret with the selection, so the keyboard does not lose its place", () => {
    render(<Harness />);
    const list = screen.getByRole("tablist");
    screen.getByRole("tab", { name: "Console" }).focus();
    fireEvent.keyDown(list, { key: "ArrowRight" });
    expect(document.activeElement).toBe(screen.getByRole("tab", { name: "Gallery" }));
  });

  it("selects on click", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("tab", { name: "Gates" }));
    expect(screen.getByRole("tab", { name: "Gates" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel")).toHaveTextContent("gates panel");
  });

  it("ignores keys that are not tablist navigation", () => {
    render(<Harness />);
    fireEvent.keyDown(screen.getByRole("tablist"), { key: "a" });
    expect(screen.getByRole("tab", { name: "Console" })).toHaveAttribute("aria-selected", "true");
  });
});
