import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "./ErrorBoundary";

function Boom({ fail }: { fail: boolean }): JSX.Element {
  if (fail) throw new Error("wilson interval was not an array");
  return <p>panel contents</p>;
}

describe("<ErrorBoundary />", () => {
  it("renders its children when nothing throws", () => {
    render(
      <ErrorBoundary region="Scoreboard">
        <Boom fail={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByText("panel contents")).toBeInTheDocument();
  });

  it("keeps the page readable when a panel throws mid-render", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    render(
      <div>
        <p>the rest of the console</p>
        <ErrorBoundary region="Scoreboard">
          <Boom fail />
        </ErrorBoundary>
      </div>,
    );
    // A render throw used to blank the entire page, mid-pitch.
    expect(screen.getByText("the rest of the console")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/Scoreboard could not render/i);
    consoleError.mockRestore();
  });

  it("names what failed and does not substitute placeholder content", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    render(
      <ErrorBoundary region="Scoreboard">
        <Boom fail />
      </ErrorBoundary>,
    );
    expect(screen.getByText("wilson interval was not an array")).toBeInTheDocument();
    expect(
      screen.getByText(/contents are unavailable rather than approximated/i),
    ).toBeInTheDocument();
    expect(screen.queryByText("panel contents")).not.toBeInTheDocument();
    consoleError.mockRestore();
  });

  it("offers a retry that remounts the region", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const onReset = vi.fn();
    const { rerender } = render(
      <ErrorBoundary region="Scoreboard" onReset={onReset}>
        <Boom fail />
      </ErrorBoundary>,
    );
    rerender(
      <ErrorBoundary region="Scoreboard" onReset={onReset}>
        <Boom fail={false} />
      </ErrorBoundary>,
    );
    await userEvent.click(screen.getByRole("button", { name: /Retry this panel/i }));
    expect(onReset).toHaveBeenCalledTimes(1);
    expect(screen.getByText("panel contents")).toBeInTheDocument();
    consoleError.mockRestore();
  });
});
