/**
 * The console's primary navigation.
 *
 * Six pages grouped by what a person is trying to do, not by which component
 * happens to render the data. `Live run` is deliberately first among the
 * working pages: it is the one the 180-second demo is driven from, and it keeps
 * the wall, the burst and the dial in script order on a single scroll.
 *
 * Ordered as the product is used, not alphabetically: Overview states the
 * claim, Live run produces the evidence, Results reads it. The last three are
 * supporting detail and sit below a divider so the spine is obvious.
 */
import { NavLink } from "react-router-dom";
import { Glyph, type GlyphName } from "./Terminal";

export type NavItem = {
  to: string;
  label: string;
  /** A printable glyph, not a drawn icon: this console is set in one cell grid. */
  icon: GlyphName;
  /** First of the supporting pages; a divider is drawn above it. */
  startsSecondary?: boolean;
};

export const NAV_ITEMS: NavItem[] = [
  { to: "/console", label: "Overview", icon: "dash" },
  { to: "/live", label: "Live run", icon: "activity" },
  { to: "/results", label: "Results", icon: "table" },
  { to: "/evidence", label: "Evidence", icon: "shield", startsSecondary: true },
  { to: "/cost", label: "Cost", icon: "gauge" },
  { to: "/clips", label: "Clips", icon: "flask" },
];

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="sidebar" aria-label="Console sections">
      <ul>
        {NAV_ITEMS.map((item) => {
          return (
            <li key={item.to} className={item.startsSecondary ? "sidebar-divider" : undefined}>
              <NavLink
                to={item.to}
                end={item.to === "/console"}
                onClick={onNavigate}
                // `aria-current="page"` is what a screen reader announces;
                // reverse video is only its visual echo, so the state is never
                // carried by colour alone.
                className={({ isActive }) => (isActive ? "sidebar-link sidebar-link-active" : "sidebar-link")}
              >
                <Glyph name={item.icon} className="shrink-0" />
                <span className="sidebar-text">{item.label}</span>
              </NavLink>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
