import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { Landing } from "./Landing";
import { BURST_TARGET, CALLED_SHOT, FOUR_NUMBERS, PENDING, PILLARS, PRODUCT } from "./content";

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

  it("carries the product name", () => {
    renderLanding();
    expect(screen.getAllByText(PRODUCT.name).length).toBeGreaterThanOrEqual(2);
  });

  it("points every call to action at a real route", () => {
    renderLanding();
    const toConsole = screen
      .getAllByRole("link", { name: /Open the console/i })
      .map((link) => link.getAttribute("href"));
    expect(toConsole.length).toBeGreaterThanOrEqual(2);
    for (const href of toConsole) expect(href).toBe("/console");

    // "Start a live run" appears twice on purpose — once as the fourth
    // pillar's action and once in the closing block — so this asserts every
    // one of them lands on /live rather than that there is only one.
    const toLive = screen
      .getAllByRole("link", { name: /(Watch|Start) a live run/i })
      .map((link) => link.getAttribute("href"));
    expect(toLive.length).toBeGreaterThanOrEqual(3);
    for (const href of toLive) expect(href).toBe("/live");
  });

  it("renders all four pillars, numbered, each with its figure", () => {
    renderLanding();
    for (const pillar of PILLARS) {
      const section = document.querySelector(`#${pillar.id}`) as HTMLElement;
      expect(section).toBeTruthy();
      expect(within(section).getByRole("heading", { level: 2 })).toHaveTextContent(pillar.headline);
      // Three sub-items, and the figure panel that belongs to the pillar.
      for (const item of pillar.items) {
        expect(within(section).getByText(item.title)).toBeInTheDocument();
      }
      expect(section.querySelector(".fig-panel")).toBeTruthy();
    }
  });

  it("numbers the figure panels FIG.1 through FIG.4", () => {
    renderLanding();
    for (let index = 1; index <= PILLARS.length; index += 1) {
      expect(screen.getByText(`FIG.${index}`)).toBeInTheDocument();
    }
  });

  it("shows no value for a number the instrument has not produced", () => {
    // The whole pitch is that an evaluator without error bars is not an
    // instrument. A landing page that invented a split-half correlation to fill
    // the panel would be making the exact mistake it accuses three prior systems
    // of, so every unmeasured figure renders as the pending string.
    renderLanding();
    const section = document.querySelector("#reliability") as HTMLElement;
    for (const number of FOUR_NUMBERS) {
      expect(within(section).getByText(number.title)).toBeInTheDocument();
    }
    // One per unmeasured number in FIG.2's pending list.
    expect(within(section).getAllByText(PENDING).length).toBeGreaterThanOrEqual(
      FOUR_NUMBERS.length,
    );
  });

  it("prints the called shot's published rates inside the evidence ledger", () => {
    renderLanding();
    const section = document.querySelector("#evidence") as HTMLElement;
    expect(within(section).getByText("92%")).toBeInTheDocument();
    expect(within(section).getByText("4%")).toBeInTheDocument();
    expect(within(section).getByText(CALLED_SHOT.source)).toBeInTheDocument();
  });

  it("labels the burst figures a target rather than a receipt", () => {
    renderLanding();
    expect(screen.getAllByText(BURST_TARGET.status).length).toBeGreaterThanOrEqual(1);
  });

  it("keeps the limits on the page at full weight", () => {
    renderLanding();
    const section = document.querySelector("#limits") as HTMLElement;
    expect(within(section).getByText("Not qualified yet")).toBeInTheDocument();
    expect(within(section).getByText(/share a backbone lineage/)).toBeInTheDocument();
  });

  it("has exactly one h1 and skips no heading level", () => {
    renderLanding();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    const levels = screen
      .getAllByRole("heading")
      .map((heading) => Number(heading.tagName.slice(1)));
    for (let index = 1; index < levels.length; index += 1) {
      expect(levels[index] - levels[index - 1]).toBeLessThanOrEqual(1);
    }
  });

  it("says the page imagery is not model output", () => {
    renderLanding();
    expect(screen.getByText(/Page imagery is illustrative/)).toBeInTheDocument();
    expect(screen.getByText(/carry provenance in the console/)).toBeInTheDocument();
  });

  it("loads no remote media at all", () => {
    // The hero backdrop was a CDN-hosted MP4. It is CSS now, which is the rule
    // the rest of the project holds to: nothing on this page can fail to load
    // and leave a broken element behind, and the page has to render identically
    // with no network — the one moment that matters is when the API is the
    // thing being demoed.
    renderLanding();
    expect(document.querySelectorAll("video")).toHaveLength(0);
    expect(document.querySelectorAll("img")).toHaveLength(0);
    for (const element of document.querySelectorAll("[src]")) {
      expect(element.getAttribute("src")).not.toMatch(/^https?:/);
    }
  });

  it("draws a static hero backdrop with nothing animating behind the headline", () => {
    // The backdrop was a wall of drifting gradients that read as a video still
    // buffering. It is one static lattice now: motion behind a headline is
    // something the eye keeps checking on instead of reading past.
    renderLanding();
    expect(document.querySelectorAll(".hero-lattice")).toHaveLength(1);
    expect(document.querySelectorAll(".dream-tile")).toHaveLength(0);
    // Nothing in the hero carries an inline animation delay any more.
    const hero = document.querySelector(".hero") as HTMLElement;
    for (const element of hero.querySelectorAll<HTMLElement>("[style]")) {
      expect(element.style.animationDelay).toBe("");
    }
  });

  it("draws the full matrix in FIG.1", () => {
    renderLanding();
    const section = document.querySelector("#evaluate") as HTMLElement;
    // Six policies against five tasks.
    expect(section.querySelectorAll(".matrix-cell")).toHaveLength(30);
    // Fifty rollouts a cell.
    expect(section.querySelectorAll(".matrix-cell")[0].querySelectorAll(".matrix-dot")).toHaveLength(
      50,
    );
  });

  it("draws twelve console tiles in FIG.4", () => {
    renderLanding();
    const section = document.querySelector("#console") as HTMLElement;
    expect(section.querySelectorAll(".console-tile")).toHaveLength(12);
  });
});
