/**
 * The masthead and the first viewport.
 *
 * The nav is sticky and numbered: the same `01 … 04` that label the pillars
 * below, so a reader who scrolls back up finds the spine they just walked. It
 * is the only navigation on the page — there is no mobile drawer, because four
 * anchors and two buttons collapse to a scrollable row without one.
 *
 * Nothing here fetches, and nothing here is remote. Every string is a constant
 * in `content.ts` and the backdrop is CSS, so the hero renders identically with
 * no network — a landing page that goes blank when the API is the thing being
 * demoed has failed at the one moment it matters.
 */
import { Link } from "react-router-dom";
import { Glyph } from "../components/Terminal";
import { BUILT_ON, CONSOLE_PATH, LIVE_PATH, NAV_LINKS, PRODUCT, PROTOCOL_PATH } from "./content";
import { PillButton } from "./Pieces";

/** The wordmark. Set in the display face at a size where its grid still reads. */
function Wordmark() {
  return (
    <Link to="/" className="wordmark" aria-label={`${PRODUCT.name} home`}>
      <span aria-hidden="true" className="wordmark-mark">
        K
      </span>
      <span className="wordmark-text">{PRODUCT.name}</span>
    </Link>
  );
}

function Nav() {
  return (
    <header className="landing-nav">
      <nav className="landing-nav-inner" aria-label="Primary">
        <Wordmark />

        <ul className="nav-tabs">
          {NAV_LINKS.map((link) => (
            <li key={link.href}>
              <a href={link.href}>
                <span>{link.label}</span>
                <span className="nav-tab-number" aria-hidden="true">
                  {link.number}
                </span>
              </a>
            </li>
          ))}
        </ul>

        <div className="nav-actions">
          <a className="nav-link" href={PROTOCOL_PATH} target="_blank" rel="noreferrer">
            Protocol record
          </a>
          <PillButton href={CONSOLE_PATH} variant="primary">
            Open the console
          </PillButton>
        </div>
      </nav>
    </header>
  );
}

export function Hero() {
  return (
    <div className="hero">
      {/*
        The backdrop is static.

        It was a remote MP4, then a wall of drifting gradients that read as a
        video still buffering — motion behind a headline that the eye keeps
        checking on instead of reading past. What is left is the console's own
        idiom: a hairline lattice on the field colour, drawn once and never
        moved, with a scrim handing off to the first section.

        Nothing here loads. The hero renders identically with no network, which
        is the one moment that matters if the API is the thing being demoed.
      */}
      <div className="hero-backdrop" aria-hidden="true">
        <div className="hero-lattice" />
        <div className="hero-scrim" />
      </div>

      <Nav />

      <div className="hero-body">
        <h1 className="hero-title">
          <span className="text-fg-strong">Evaluate a robot policy</span>{" "}
          <span className="text-fg-muted">without a robot.</span>
        </h1>

        <p className="hero-promise">{PRODUCT.promise}</p>

        <div className="hero-actions">
          <PillButton href={CONSOLE_PATH} variant="primary">
            Open the console
          </PillButton>
          <PillButton href={LIVE_PATH}>Watch a live run</PillButton>
        </div>

        <div className="hero-built-on">
          <span className="font-mono text-xs uppercase tracking-[0.18em] text-fg-dim">Built on</span>
          <ul>
            {BUILT_ON.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </div>

      <a className="hero-scroll" href={`#${NAV_LINKS[0].href.slice(1)}`} aria-label="Skip to the first section">
        <Glyph name="arrowRight" />
      </a>
    </div>
  );
}
