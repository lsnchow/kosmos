/**
 * The entry point, and the start of the path through the product.
 *
 * The claim, then the one thing blocking it, then the called shot and the
 * gates. The primary action goes to the run, because the run is what the rest
 * of the console is about — free-play is a demonstration, not the flow.
 */
import { Glyph } from "../components/Terminal";
import { Link } from "react-router-dom";
import { STARTS_PER_TASK, useAppData } from "../AppData";
import { CalledShotPanel } from "../components/CalledShotPanel";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { GatePanel } from "../components/GatePanel";
import { StatusPill } from "../components/Primitives";
import { pickString } from "../lib/format";
import { PageHeader } from "./PageHeader";

export function OverviewPage() {
  const { protocol, policies, tasks, blockers, calledShot, gates, openFreeplay } = useAppData();
  const matrix =
    policies.length > 0 && tasks.length > 0
      ? `${policies.length} policies × ${tasks.length} tasks × ${STARTS_PER_TASK} starts`
      : undefined;

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={
          <StatusPill status="unqualified" title={`mode: ${pickString(protocol?.mode) ?? "not reported"}`}>
            unqualified
          </StatusPill>
        }
        title="Measure the ruler before trusting the ranking."
        lede="Six robot policies scored inside a generative world model, with four numbers saying how far to trust the ranking."
        actions={
          <>
            <Link className="button button-primary" to="/live">
              Start a run
              <Glyph name="arrowRight" />
            </Link>
            <button type="button" className="button button-quiet" onClick={() => openFreeplay()}>
              <Glyph name="expand" />
              Drive the world model
            </button>
          </>
        }
      />

      {/* One line, not a paragraph. The blockers themselves are the content. */}
      <section className="status-strip" aria-label="Qualification status">
        <p>
          <strong>Not qualified.</strong> Real backends stay blocked until these pass
          {matrix ? ` · ${matrix}` : ""}
        </p>
        <ul>
          {blockers.slice(0, 3).map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      </section>

      <div className="split-grid">
        <ErrorBoundary region="Called shot">
          <CalledShotPanel calledShot={calledShot} />
        </ErrorBoundary>
        <ErrorBoundary region="Qualification gates">
          <GatePanel gates={gates} />
        </ErrorBoundary>
      </div>
    </div>
  );
}
