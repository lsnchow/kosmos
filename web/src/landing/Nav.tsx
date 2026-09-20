/**
 * The masthead.
 *
 * It lives beside the hero rather than inside it, and is fixed rather than
 * sticky. Sticky put it inside `.hero`'s bounds, which meant it scrolled away
 * with the hero instead of following the reader down the page; fixed and
 * hoisted to `Landing` it persists over every section.
 *
 * It is hidden over the hero and arrives on scroll. The hero already carries
 * the wordmark's job — it says what the product is at full size — and both of
 * the nav's actions, so a bar over it is a second copy of what the reader is
 * already looking at. The numbered tabs only start meaning something once the
 * pillars they point at exist above the fold.
 *
 * Hidden means `visibility: hidden`, not just transparent: a bar a keyboard
 * user can tab into but nobody can see is worse than no bar.
 */
import { Link } from "react-router-dom";
import { CONSOLE_PATH, NAV_LINKS, PRODUCT, PROTOCOL_PATH } from "./content";
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

export function Nav({ visible }: { visible: boolean }) {
  return (
    <header className="landing-nav" data-visible={visible ? "true" : "false"}>
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
