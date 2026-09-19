import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { Landing } from "./Landing";
import { LIMITS, PILLARS, PRODUCT } from "./content";

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

  it("renders all four pillars with their number, claim and sub-items", () => {
    renderLanding();
    for (const pillar of PILLARS) {
      const section = document.querySelector(`#${pillar.id}`) as HTMLElement;
      expect(section).toBeTruthy();
      expect(within(section).getByRole("heading", { level: 2 })).toHaveTextContent(pillar.headline);
      expect(within(section).getByText(pillar.number)).toBeInTheDocument();
      for (const item of pillar.items) {
        expect(within(section).getByText(item.title)).toBeInTheDocument();
        expect(within(section).getByText(item.key)).toBeInTheDocument();
      }
      expect(within(section).getByRole("link", { name: new RegExp(pillar.cta.label, "i") })).toHaveAttribute(
        "href",
        pillar.cta.href,
      );
    }
  });

  it("carries no figure panels", () => {
    // The four FIG.N data panels were removed. This asserts they are gone
    // rather than merely unstyled — an orphaned panel with no CSS would still
    // render its markup and would not show up in a visual check.
    renderLanding();
    expect(document.querySelectorAll(".fig-panel")).toHaveLength(0);
    expect(document.querySelectorAll("svg.reliability-plot")).toHaveLength(0);
    expect(document.querySelectorAll(".matrix-cell, .console-tile, .evidence-row")).toHaveLength(0);
    expect(screen.queryByText(/^FIG\.\d$/)).not.toBeInTheDocument();
  });

  it("keeps every limit on the page at full weight", () => {
    // With the figures gone this block is the only place the page states a
    // caveat about itself, so it is checked in full rather than sampled.
    renderLanding();
    const section = document.querySelector("#limits") as HTMLElement;
    for (const limit of LIMITS) {
      expect(within(section).getByText(limit.title)).toBeInTheDocument();
      expect(within(section).getByText(limit.body)).toBeInTheDocument();
    }
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

});
