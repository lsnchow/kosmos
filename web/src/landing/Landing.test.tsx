import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { Landing } from "./Landing";
import { BURST_TARGET, CALLED_SHOT, FOUR_NUMBERS, PENDING } from "./content";

function renderLanding() {
  return render(
    <MemoryRouter>
      <Landing />
    </MemoryRouter>,
  );
}

describe("landing page", () => {
  it("renders with no network at all", () => {
    // No fetch double is installed. A landing page that needs the API is a
    // landing page that goes blank when the API is the thing being demoed.
    renderLanding();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      /Evaluate a robot policy without a robot/i,
    );
  });

  it("points every call to action at the console", () => {
    renderLanding();
    const toConsole = screen
      .getAllByRole("link", { name: /Open the console/i })
      .map((link) => link.getAttribute("href"));
    expect(toConsole.length).toBeGreaterThanOrEqual(2);
    for (const href of toConsole) expect(href).toBe("/console");

    expect(screen.getByRole("link", { name: /Watch a live run/i })).toHaveAttribute("href", "/live");
    expect(screen.getByRole("link", { name: /Start a live run/i })).toHaveAttribute("href", "/live");
  });

  it("prints the called shot with both published rates and its citation", () => {
    renderLanding();
    const section = document.querySelector("#called-shot") as HTMLElement;
    expect(within(section).getByText("92%")).toBeInTheDocument();
    expect(within(section).getByText("4%")).toBeInTheDocument();
    expect(
      within(section).getByText(new RegExp(CALLED_SHOT.source.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))),
    ).toBeInTheDocument();
  });

  it("shows no value for a number the instrument has not produced", () => {
    // The whole pitch is that an evaluator without error bars is not an
    // instrument. A landing page that invented a split-half correlation to fill
    // the card would be making the exact mistake it accuses three prior systems
    // of, so every unmeasured figure renders as the pending string.
    renderLanding();
    const section = document.querySelector("#numbers") as HTMLElement;
    expect(within(section).getAllByText(PENDING)).toHaveLength(FOUR_NUMBERS.length);
    for (const number of FOUR_NUMBERS) {
      expect(within(section).getByRole("heading", { name: number.title })).toBeInTheDocument();
    }
  });

  it("labels the burst figures a target rather than a receipt", () => {
    renderLanding();
    const section = document.querySelector("#stack") as HTMLElement;
    expect(within(section).getByText(/1,500 rollouts in 60 seconds for \$11\.25/)).toBeInTheDocument();
    expect(within(section).getByText(BURST_TARGET.status)).toBeInTheDocument();
  });

  it("states its limits on the page rather than in a footnote", () => {
    renderLanding();
    const section = document.querySelector("#limits") as HTMLElement;
    expect(within(section).getByRole("heading", { name: /Not qualified yet/i })).toBeInTheDocument();
    expect(within(section).getByText(/share a backbone lineage/i)).toBeInTheDocument();
  });

  it("keeps one h1 and no heading level skipped", () => {
    renderLanding();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    const levels = screen
      .getAllByRole("heading")
      .map((heading) => Number(heading.tagName.slice(1)));
    let previous = 0;
    for (const level of levels) {
      expect(level, `heading jumps from h${previous} to h${level}`).toBeLessThanOrEqual(previous + 1);
      previous = level;
    }
  });

  it("marks its imagery as illustrative rather than as model output", () => {
    // The atmosphere on this page is stock footage. The console is where
    // generated frames carry provenance, and a product whose pitch is honesty
    // about measurement cannot let a visitor mistake one for the other.
    renderLanding();
    const footer = screen.getByRole("contentinfo");
    expect(within(footer).getByText(/Page imagery is illustrative/i)).toBeInTheDocument();
    expect(within(footer).getByText(/carry provenance in the console/i)).toBeInTheDocument();
  });

  it("keeps every video decorative, silent and out of the tab order", () => {
    renderLanding();
    const videos = [...document.querySelectorAll("video")];
    expect(videos.length).toBeGreaterThan(0);
    for (const video of videos) {
      expect(video).toHaveAttribute("aria-hidden", "true");
      expect(video.muted).toBe(true);
      expect(video.tabIndex).toBe(-1);
      // Opacity 0 until `canplay`: a blocked CDN leaves the CSS field showing
      // rather than a black rectangle, which is what lets the page render with
      // no network at all.
      expect(video.style.opacity).toBe("0");
    }
  });

  it("paints a CSS field behind every video, including the twelve-tile hero wall", () => {
    renderLanding();
    const tiles = [...document.querySelectorAll(".dream-tile")];
    const heroWall = tiles.filter((tile) => tile.querySelector(".dream-chunk"));
    expect(heroWall).toHaveLength(12);
    for (const tile of heroWall) expect(tile.closest("[aria-hidden='true']")).not.toBeNull();
    // The rest are fallback surfaces, one per section video.
    expect(tiles.length - heroWall.length).toBe(document.querySelectorAll("video").length - 1);
  });
});
