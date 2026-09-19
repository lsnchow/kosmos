/**
 * The console's primary navigation.
 *
 * Six pages grouped by what a person is trying to do, not by which component
 * happens to render the data. `Live run` is deliberately first among the
 * working pages: it is the one the 180-second demo is driven from, and it keeps
 * the wall, the burst and the dial in script order on a single scroll.
 *
 * Each item states what it holds rather than only naming itself, because a
 * one-word label ("Evidence") does not tell a first-time viewer whether the
 * gates live there or on "Results".
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
  hint: string;
  icon: LucideIcon;
};

export const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Overview", hint: "The claim, the called shot, what is not qualified", icon: LayoutDashboard },
  { to: "/live", label: "Live run", hint: "Wall, chain, telemetry, burst, dial", icon: Activity },
  { to: "/results", label: "Results", hint: "Scoreboard and the run ledger", icon: Table2 },
  { to: "/evidence", label: "Evidence", hint: "Gates A–F, imported smoke reports", icon: ShieldCheck },
  { to: "/cost", label: "Cost", hint: "Cost–fidelity dial and replica accounting", icon: Gauge },
  { to: "/clips", label: "Clips", hint: "Six-clip test, gallery, 480p track", icon: FlaskConical },
];

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="sidebar" aria-label="Console sections">
      <ul>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.to === "/"}
                onClick={onNavigate}
                // `aria-current="page"` is what a screen reader announces; the
                // lime rail is only its visual echo, so the state is never
                // carried by colour alone.
                className={({ isActive }) => (isActive ? "sidebar-link sidebar-link-active" : "sidebar-link")}
              >
                <Icon aria-hidden="true" className="size-4 shrink-0" />
                <span className="sidebar-text">
                  <strong>{item.label}</strong>
                  <span className="sidebar-hint">{item.hint}</span>
                </span>
              </NavLink>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
