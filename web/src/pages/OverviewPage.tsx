import { Link } from "react-router-dom";
import { ComparisonWall } from "../components/ComparisonWall";
import { LiveDemoPanel } from "../components/LiveDemo";
import { WorldVideoGrid } from "../components/WorldVideoGrid";
import { PageHeader } from "./PageHeader";

/** The demo opens on saved media. Playback never submits inference work. */
export function OverviewPage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Interactive demo and saved media"
        title="Evaluate or watch"
        lede="Start a bounded OpenVLA or manual experimental evaluation, then inspect its persisted frames alongside the saved recording archive."
        actions={
          <Link className="button button-secondary" to="/clips">
            Browse all recordings →
          </Link>
        }
      />
      <ComparisonWall />
      <section aria-labelledby="legacy-demo-heading">
        <h2 id="legacy-demo-heading" className="text-balance">
          Legacy experimental controls
        </h2>
        <p className="text-pretty text-sm text-fg-muted">
          This earlier bounded demo remains available for engineering checks. It is not a cell in the controlled wall and is not a policy comparison.
        </p>
        <LiveDemoPanel />
      </section>
      <WorldVideoGrid />
      <details className="gallery-guide">
        <summary>What are policies, and what will we compare?</summary>
        <p className="text-pretty">
          A policy is the robot’s controller: it looks at the scene and goal,
          then chooses arm and gripper movements. The world model predicts the
          next view; a separate judge can assess the result.
        </p>
        <p className="text-pretty">
          The controlled preview fixes OpenVLA, MiniVLA, and Octo-Small across
          four matched world seeds from one Close the drawer starting bundle.
        </p>
        <p className="text-pretty">
          The wall remains unscored: it reports operational progress and
          persisted media, never a success score, rank, or policy winner.
        </p>
        <Link to="/evidence">See what has been verified →</Link>
      </details>
    </div>
  );
}
