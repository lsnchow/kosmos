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
import {
  Activity,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  ShieldCheck,
  Table2,
  type LucideIcon,
} from "lucide-react";
import { NavLink } from "react-router-dom";

export type NavItem = {
  to: string;
  label: string;
  icon: LucideIcon;
  /** First of the supporting pages; a divider is drawn above it. */
  startsSecondary?: boolean;
};

export const NAV_ITEMS: NavItem[] = [
  { to: "/console", label: "Overview", icon: LayoutDashboard },
  { to: "/live", label: "Live run", icon: Activity },
  { to: "/results", label: "Results", icon: Table2 },
  { to: "/evidence", label: "Evidence", icon: ShieldCheck, startsSecondary: true },
  { to: "/cost", label: "Cost", icon: Gauge },
  { to: "/clips", label: "Clips", icon: FlaskConical },
];

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="sidebar" aria-label="Console sections">
      <ul>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <li key={item.to} className={item.startsSecondary ? "sidebar-divider" : undefined}>
              <NavLink
                to={item.to}
                end={item.to === "/console"}
                onClick={onNavigate}
                // `aria-current="page"` is what a screen reader announces; the
                // lime rail is only its visual echo, so the state is never
                // carried by colour alone.
                className={({ isActive }) => (isActive ? "sidebar-link sidebar-link-active" : "sidebar-link")}
              >
                <Icon aria-hidden="true" className="size-4 shrink-0" />
                <span className="sidebar-text">{item.label}</span>
              </NavLink>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
