import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { Landing } from "./Landing";
import { LIMITS, PILLARS, PIPELINE_STAGES, PRODUCT } from "./content";

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

  it("renders all four pillars as `NN Title`, claim, action and commitments", () => {
    renderLanding();
    for (const pillar of PILLARS) {
      const section = document.querySelector(`#${pillar.id}`) as HTMLElement;
      expect(section).toBeTruthy();

      // The heading is the number and the name on one line — the claim moved
      // out of it and into the paragraph below.
      const heading = within(section).getByRole("heading", { level: 2 });
      expect(heading).toHaveTextContent(pillar.number);
      expect(heading).toHaveTextContent(pillar.name);
      expect(within(section).getByText(pillar.headline)).toBeInTheDocument();

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

  it("lays the pillars out as a four-quadrant grid", () => {
    renderLanding();
    const grid = document.querySelector(".pillar-grid") as HTMLElement;
    expect(grid).toBeTruthy();
    expect(grid.querySelectorAll(":scope > .pillar-cell")).toHaveLength(PILLARS.length);
  });

  it("gives every pillar a numbered figure with its own graphic", () => {
    renderLanding();
    expect(document.querySelectorAll(".fig-panel")).toHaveLength(PILLARS.length);
    for (let index = 1; index <= PILLARS.length; index += 1) {
      expect(screen.getByText(`FIG.${index}`)).toBeInTheDocument();
    }

    // Two graphics, and each pillar names which it carries. The labels are the
    // figure's content, so they are asserted rather than the shape alone.
    for (const pillar of PILLARS) {
      const section = document.querySelector(`#${pillar.id}`) as HTMLElement;

      if (pillar.figure !== "stack") {
        // Supplied artwork, so the labels live in the pixels; the accessible
        // name is what can be asserted. An empty alt would make the figure
        // invisible to a screen reader rather than merely undescribed, so the
        // name is required to be non-trivial.
        const artwork = within(section).getByRole("img");
        expect(artwork).toHaveClass("pipe-figure");
        expect(artwork.getAttribute("alt")?.length ?? 0).toBeGreaterThan(20);
        continue;
      }

      const stack = section.querySelector("svg.iso-stack") as SVGElement;
      expect(stack).toBeTruthy();
      expect(stack.querySelectorAll(".iso-plane")).toHaveLength(pillar.layers.length);
      for (const layer of pillar.layers) {
        expect(within(section).getByText(layer.label)).toBeInTheDocument();
      }
    }
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


  it("serves every asset from this origin and none from a CDN", () => {
    // The hero backdrop was once a CDN-hosted MP4. The rule that replaced it is
    // not "no media" but "no *remote* media": the page has to render
    // identically with no network, which a same-origin file satisfies and a
    // third-party URL does not. FIG.1's artwork is self-hosted under /figures.
    renderLanding();
    expect(document.querySelectorAll("video")).toHaveLength(0);
    for (const element of document.querySelectorAll("[src]")) {
      const src = element.getAttribute("src") ?? "";
      expect(src).not.toMatch(/^https?:/);
      expect(src.startsWith("/")).toBe(true);
    }
  });

  it("gives the supplied FIG.1 artwork alt text built from its own labels", () => {
    // The artwork carries the stage names as pixels, so the alt text is
    // generated from the same constants rather than typed out beside them —
    // otherwise the two drift and the screen-reader copy goes stale silently.
    renderLanding();
    const image = document.querySelector("img.pipe-figure") as HTMLImageElement;
    expect(image).toBeTruthy();
    expect(image.getAttribute("src")).toBe("/figures/pipeline-cycle.png");
    for (const stage of PIPELINE_STAGES) {
      expect(image.alt).toContain(stage.label);
      expect(image.alt).toContain(stage.note);
    }
  });

  it("gives the hero a decorative backdrop that announces nothing", () => {
    // The hero has carried a CDN video, a wall of drifting gradients, and a
    // hairline lattice. It carries a locally compiled shader now. Two things
    // are asserted: that the dead layers did not survive, and that the live one
    // is hidden from assistive technology — a canvas of moving phosphor holds
    // no information, and announcing it is noise.
    renderLanding();
    const hero = document.querySelector(".hero") as HTMLElement;
    expect(hero).toBeTruthy();
    for (const selector of [".hero-lattice", ".hero-scrim", ".dream-tile"]) {
      expect(document.querySelectorAll(selector)).toHaveLength(0);
    }
    const backdrop = hero.querySelector(".hero-backdrop") as HTMLElement;
    expect(backdrop).toBeTruthy();
    expect(backdrop).toHaveAttribute("aria-hidden", "true");
    expect(backdrop.textContent).toBe("");

    // Nothing in the hero carries an inline animation delay any more.
    for (const element of hero.querySelectorAll<HTMLElement>("[style]")) {
      expect(element.style.animationDelay).toBe("");
    }
  });

  it("renders the page without WebGL rather than failing", () => {
    // jsdom has no WebGL, so constructing the renderer throws and the component
    // takes its fallback path. That is the case this asserts: the heading is
    // still on screen and the backdrop is empty — which is also what a
    // locked-down browser or a refusing driver produces.
    renderLanding();
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
    expect(document.querySelector(".hero-backdrop")?.querySelector("canvas")).toBeNull();
  });

});
