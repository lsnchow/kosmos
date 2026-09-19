import { NavLink, useLocation } from "react-router-dom";
import { Glyph, type GlyphName } from "./Terminal";

export type NavItem = {
  to: string;
  label: string;
  hint: string;
  icon: GlyphName;
};

export const NAV_ITEMS: NavItem[] = [
  {
    to: "/console",
    label: "Video gallery",
    hint: "Watch saved model outputs",
    icon: "play",
  },
  {
    to: "/results",
    label: "Saved runs",
    hint: "Past runs and their measurements",
    icon: "table",
  },
  {
    to: "/clips",
    label: "Recording archive",
    hint: "All clips, including short probes",
    icon: "flask",
  },
  {
    to: "/live",
    label: "Developer tools",
    hint: "Cloud checks and synthetic tests",
    icon: "activity",
  },
  {
    to: "/evidence",
    label: "Validation",
    hint: "What is verified—and what is not",
    icon: "shield",
  },
  {
    to: "/cost",
    label: "Compute & cost",
    hint: "Recorded resource measurements",
    icon: "gauge",
  },
  {
    to: "/review",
    label: "Review clips",
    hint: "Manually annotate development clips",
    icon: "shield",
  },
];

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { pathname } = useLocation();
  const links = (items: NavItem[]) => (
    <ul>
      {items.map((item) => (
        <li key={item.to}>
          <NavLink
            to={item.to}
            end
            onClick={onNavigate}
            className={({ isActive }) =>
              isActive ? "sidebar-link sidebar-link-active" : "sidebar-link"
            }
          >
            <Glyph name={item.icon} className="shrink-0" />
            <span className="sidebar-text">
              <span>{item.label}</span>
              <small className="sidebar-hint">{item.hint}</small>
            </span>
          </NavLink>
        </li>
      ))}
    </ul>
  );
  return (
    <nav className="sidebar" aria-label="Console sections">
      {links(NAV_ITEMS.slice(0, 2))}
      <details
        className="sidebar-tools"
        key={pathname}
        open={NAV_ITEMS.slice(2).some((item) => item.to === pathname)}
      >
        <summary>Tools & validation</summary>
        {links(NAV_ITEMS.slice(2))}
      </details>
    </nav>
  );
}
