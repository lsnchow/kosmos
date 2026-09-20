/**
 * The four pillar sections, the limits block, and the footer.
 *
 * Every pillar has the same shape — heading, body, action, three numbered
 * sub-items — so there is one component rather than four. The differences
 * between them live in `PILLARS`, which means a fifth pillar is a data edit and
 * not a new file, and it means no pillar can quietly acquire a layout the
 * others do not have.
 */
import type { ReactNode } from "react";
import {
  FOOTER_COLUMNS,
  FOOTER_DISPLAY,
  LIMITS,
  PILLARS,
  PRODUCT,
  CONSOLE_PATH,
  LIVE_PATH,
} from "./content";
import { ConsoleMatrix } from "./figures/ConsoleMatrix";
import { IsoStack } from "./figures/IsoStack";
import { ProvenanceFunnel } from "./figures/ProvenanceFunnel";
import { PipelineDiamond } from "./figures/PipelineDiamond";
import { FigPanel, PillButton, Reveal, SectionCard } from "./Pieces";

type Pillar = (typeof PILLARS)[number];

/**
 * One pillar: its graphic above, its claim and its commitments below.
 *
 * The cell is a quadrant of a 2 x 2 grid rather than a full-width band, so the
 * graphic and the copy share a column and a reader takes the pillar as one
 * object. The heading is `NN Title` on a single line — the number set dim
 * beside it rather than stacked above — and the three commitments sit opposite
 * as single lines, numbered `N.1` through `N.3`.
 */
function PillarSection({ pillar, index }: { pillar: Pillar; index: number }) {
  return (
    <section id={pillar.id} className="pillar-cell">
      <FigPanel index={index}>
        {pillar.figure === "pipeline" ? (
          <PipelineDiamond />
        ) : pillar.figure === "provenance" ? (
          <ProvenanceFunnel />
        ) : pillar.figure === "console" ? (
          <ConsoleMatrix />
        ) : (
          <IsoStack number={pillar.number} layers={pillar.layers} />
        )}
      </FigPanel>

      <div className="pillar-foot">
        <Reveal className="pillar-copy">
          <h2 className="pillar-heading">
            <span className="pillar-number font-mono">{pillar.number}</span>
            <span className="text-fg-strong">{pillar.name}</span>
          </h2>
          <p className="pillar-body">{pillar.headline}</p>
          <PillButton href={pillar.cta.href}>{pillar.cta.label}</PillButton>
        </Reveal>

        <Reveal className="pillar-items" delay={0.1}>
          <ul>
            {pillar.items.map((item) => (
              <li key={item.key}>
                <span className="pillar-item-key font-mono">{item.key}</span>
                <span className="pillar-item-text">{item.title}</span>
              </li>
            ))}
          </ul>
        </Reveal>
      </div>
    </section>
  );
}

export function Pillars() {
  // The grid draws the hairlines between quadrants, so the cells themselves
  // carry no border and cannot double one up along a shared edge.
  return (
    <div className="pillar-grid">
      {PILLARS.map((pillar, index) => (
        <PillarSection key={pillar.id} pillar={pillar} index={index + 1} />
      ))}
    </div>
  );
}

/**
 * The limits, at the same weight as the pillars and above the closing action.
 *
 * An evaluation product that hides its own caveats has already lost the
 * argument it is making, so this block is not a footnote and is not collapsed.
 */
export function LimitsSection() {
  return (
    <SectionCard id="limits">
      <Reveal className="limits-head">
        <p className="pillar-number font-mono" aria-hidden="true">
          05
        </p>
        <h2 className="pillar-heading">
          <span className="text-fg-strong">Limits.</span>{" "}
          <span className="text-fg-muted">Goalposts we cannot move until we cross them.</span>
        </h2>
      </Reveal>
      {/* One `Reveal` around the list rather than one per card: a <div> between
          <ul> and <li> is invalid markup, and the axe pass reads it as one. */}
      <Reveal delay={0.1}>
        <ul className="limits-grid">
          {LIMITS.map((limit) => (
            <li key={limit.title} className="limit-card">
              <h3 className="text-fg">{limit.title}</h3>
              <p className="text-fg-muted">{limit.body}</p>
            </li>
          ))}
        </ul>
      </Reveal>
    </SectionCard>
  );
}

export function CtaSection() {
  return (
    <SectionCard className="cta-section">
      <Reveal className="cta-inner">
        <h2 className="pillar-heading">
          <span className="text-fg-strong">We measured the ruler</span>{" "}
          <span className="text-fg-muted">before we trusted it.</span>
        </h2>
        <p className="pillar-body">{PRODUCT.tagline}</p>
        <div className="hero-actions">
          <PillButton href={CONSOLE_PATH} variant="primary">
            Open the console
          </PillButton>
          <PillButton href={LIVE_PATH}>Start a live run</PillButton>
        </div>
      </Reveal>
    </SectionCard>
  );
}

function FooterLink({ href, external, children }: { href: string; external?: boolean; children: ReactNode }) {
  // Both footer link groups leave the SPA or jump within it; the external ones
  // are real documents (raw protocol JSON, the notices file) and must not be
  // routed, or the shell swallows them into a 404.
  return (
    <a href={href} {...(external ? { target: "_blank", rel: "noreferrer" } : {})}>
      {children}
    </a>
  );
}

export function LandingFooter() {
  return (
    <footer className="landing-footer">
      <div className="footer-top">
        <div className="footer-brand">
          <span className="wordmark-mark" aria-hidden="true">
            K
          </span>
          <span className="wordmark-text">{PRODUCT.name}</span>
        </div>

        <div className="footer-columns">
          {FOOTER_COLUMNS.map((column) => (
            <div key={column.heading}>
              <p className="footer-heading font-mono">{column.heading}</p>
              <ul>
                {column.links.map((link) => (
                  <li key={link.label}>
                    <FooterLink href={link.href} external={"external" in link && link.external}>
                      {link.label}
                    </FooterLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* The tagline set large in the dot-matrix face, where its grid reads as
          character rather than as a blob. */}
      <p className="footer-display" aria-hidden="true">
        {FOOTER_DISPLAY}
      </p>

      <div className="footer-bottom">
        <p className="footer-copy font-mono">
          © {new Date().getFullYear()} {PRODUCT.name}
        </p>
      </div>
    </footer>
  );
}
