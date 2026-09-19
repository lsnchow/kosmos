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
  FOOTER_NOTE,
  LIMITS,
  PILLARS,
  PRODUCT,
  CONSOLE_PATH,
  LIVE_PATH,
} from "./content";
import { PillButton, Reveal, SectionCard } from "./Pieces";

type Pillar = (typeof PILLARS)[number];

function PillarSection({ pillar }: { pillar: Pillar }) {
  return (
    <SectionCard id={pillar.id}>
      <div className="pillar-head">
        <Reveal className="pillar-copy">
          <p className="pillar-number font-mono" aria-hidden="true">
            {pillar.number}
          </p>
          <h2 className="pillar-heading">
            <span className="text-fg-strong">{pillar.lead}</span>{" "}
            <span className="text-fg-muted">{pillar.headline}</span>
          </h2>
          <p className="pillar-body">{pillar.body}</p>
          <PillButton href={pillar.cta.href}>{pillar.cta.label}</PillButton>
        </Reveal>

        <Reveal className="pillar-items" delay={0.12}>
          <ul>
            {pillar.items.map((item) => (
              <li key={item.key}>
                <span className="pillar-item-key font-mono" aria-hidden="true">
                  {item.key}
                </span>
                <span className="pillar-item-text">
                  <span className="text-fg">{item.title}</span>
                  <span className="text-fg-muted">{item.body}</span>
                </span>
              </li>
            ))}
          </ul>
        </Reveal>
      </div>
    </SectionCard>
  );
}

export function Pillars() {
  return (
    <>
      {PILLARS.map((pillar) => (
        <PillarSection key={pillar.id} pillar={pillar} />
      ))}
    </>
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
        <p className="footer-note">{FOOTER_NOTE}</p>
        <p className="footer-copy font-mono">
          © {new Date().getFullYear()} {PRODUCT.name}
        </p>
      </div>
    </footer>
  );
}
