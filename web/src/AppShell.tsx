/**
 * Persistent chrome: topbar, sidebar, and the three overlays.
 *
 * This component never unmounts, so the poll, the event stream and every open
 * dialog survive navigation. The page itself is the only thing that swaps.
 *
 * The overlays live here rather than on a page for the same reason. Beat two of
 * the demo blows a wall tile up to full screen and drives it with the arrow
 * keys; if free-play were a route, taking the controls would navigate away from
 * the wall it came from.
 */
import * as AlertDialog from "@radix-ui/react-alert-dialog";
import { CircleAlert, Expand, ExternalLink, Menu, Presentation, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { useAppData } from "./AppData";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { FreeplayDialog } from "./components/FreeplayDialog";
import { GalleryModal } from "./components/GalleryModal";
import { ProvenanceBadge } from "./components/RolloutWall";
import { Sidebar } from "./components/Sidebar";
import { formatCount, formatFrameCaption, pickString } from "./lib/format";
import { cn } from "./lib/utils";

export function AppShell() {
  const {
    health,
    backends,
    presentation,
    setPresentation,
    refresh,
    loadError,
    actionError,
    viewerSlot,
    setViewerSlot,
    openFreeplay,
    freeplayOpen,
    setFreeplayOpen,
    freeplaySubject,
    cancelOpen,
    setCancelOpen,
    cancelRun,
  } = useAppData();
  const [navOpen, setNavOpen] = useState(false);
  const location = useLocation();

  // Below the sidebar breakpoint the nav is a disclosure. Leaving it open after
  // a route change would cover the page the reader just asked for.
  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#page">
        Skip to content
      </a>

      <header className="topbar">
        <button
          type="button"
          className="button button-quiet nav-toggle"
          aria-expanded={navOpen}
          aria-controls="sidebar-region"
          onClick={() => setNavOpen((open) => !open)}
        >
          <Menu aria-hidden="true" className="size-4" />
          Sections
        </button>
        <Link className="brand" to="/console" aria-label="Nightshift console home">
          <span className="brand-mark">N</span>
          <span>Nightshift</span>
        </Link>
        <div className="topbar-right">
          <span className="api-health">
            <span className={cn("health-dot", health ? "health-known" : "health-pending")} />
            API {pickString(health?.status) ?? "checking"}
            {backends.length > 0 && <span className="backend-list">· {backends.join(", ")}</span>}
          </span>
          <button
            type="button"
            className={cn("button", presentation ? "button-primary" : "button-quiet")}
            onClick={() => setPresentation(!presentation)}
            aria-pressed={presentation}
          >
            <Presentation aria-hidden="true" className="size-4" />
            Presentation mode
          </button>
          <button type="button" className="button button-quiet" onClick={() => void refresh()}>
            <RotateCcw aria-hidden="true" className="size-4" />
            Refresh
          </button>
        </div>
      </header>

      <div className={cn("shell-body", navOpen && "shell-body-nav-open")}>
        <div className="sidebar-region" id="sidebar-region">
          <Sidebar onNavigate={() => setNavOpen(false)} />
        </div>

        <main id="page" className="page" tabIndex={-1}>
          {(loadError ?? actionError) && (
            <div className="error-banner" role="alert">
              <CircleAlert aria-hidden="true" className="size-5" />
              <span>{actionError ?? loadError}</span>
            </div>
          )}
          <ErrorBoundary region="Page">
            <Outlet />
          </ErrorBoundary>

          <footer className="footer-note">
            <span>Nightshift</span>
            <a href="/THIRD-PARTY-NOTICES.md" target="_blank" rel="noreferrer">
              Third-party notices <ExternalLink aria-hidden="true" className="size-3" />
            </a>
            <a href="/api/protocol" target="_blank" rel="noreferrer">
              Protocol record <ExternalLink aria-hidden="true" className="size-3" />
            </a>
          </footer>
        </main>
      </div>

      {/* One wall tile at full screen. Provenance travels with the clip: a
          replayed frame must not lose its label by being made bigger. */}
      <GalleryModal
        open={viewerSlot !== undefined}
        onClose={() => setViewerSlot(undefined)}
        title={viewerSlot ? `${viewerSlot.policy} · ${viewerSlot.task}` : ""}
        subtitle={viewerSlot?.episodeId ?? "episode id not reported"}
        frames={viewerSlot?.frames ?? []}
        emptyReason="No persisted segment event has reached this slot, so there is no frame to enlarge."
        badge={viewerSlot ? <ProvenanceBadge slot={viewerSlot} /> : undefined}
        meta={
          viewerSlot
            ? [
                { label: "Run", value: viewerSlot.runId ?? "not reported" },
                { label: "Segments", value: formatCount(viewerSlot.segmentCount) },
                {
                  label: "Frames",
                  value: formatFrameCaption(
                    viewerSlot.frames.length,
                    viewerSlot.certifiedFrameCount,
                    undefined,
                  ),
                },
                { label: "Resolution", value: viewerSlot.resolution ?? "not reported" },
              ]
            : []
        }
        footer={
          <button
            type="button"
            className="button button-primary"
            onClick={() => {
              const slot = viewerSlot;
              setViewerSlot(undefined);
              openFreeplay(slot);
            }}
          >
            <Expand aria-hidden="true" className="size-4" />
            Take the controls
          </button>
        }
      />

      <FreeplayDialog open={freeplayOpen} onOpenChange={setFreeplayOpen} subjectLabel={freeplaySubject} />

      <AlertDialog.Root open={cancelOpen} onOpenChange={setCancelOpen}>
        <AlertDialog.Portal>
          <AlertDialog.Overlay className="dialog-overlay" />
          <AlertDialog.Content className="alert-dialog">
            <AlertDialog.Title>Cancel this run?</AlertDialog.Title>
            <AlertDialog.Description className="text-pretty">
              Cancellation leaves explicit terminal records and retains already allocated work and its cost.
            </AlertDialog.Description>
            <div className="alert-actions">
              <AlertDialog.Cancel className="button button-secondary">Keep running</AlertDialog.Cancel>
              <AlertDialog.Action className="button button-danger" onClick={() => void cancelRun()}>
                Cancel run
              </AlertDialog.Action>
            </div>
          </AlertDialog.Content>
        </AlertDialog.Portal>
      </AlertDialog.Root>
    </div>
  );
}
