import { Link } from "react-router-dom";
import { WorldVideoGrid } from "../components/WorldVideoGrid";
import { PageHeader } from "./PageHeader";

/** The demo opens on saved media. Playback never submits inference work. */
export function OverviewPage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Watch first · no GPU request"
        title="Video gallery"
        lede="Watch saved robot world-model videos. A world model predicts what the camera would see after a robot moves."
        actions={
          <Link className="button button-secondary" to="/clips">
            Browse all recordings →
          </Link>
        }
      />
      <WorldVideoGrid />
      <details className="gallery-guide">
        <summary>What are policies, and what will we compare?</summary>
        <p className="text-pretty">
          A policy is the robot’s controller: it looks at the scene and goal,
          then chooses arm and gripper movements. The world model predicts the
          next view; a separate judge can assess the result.
        </p>
        <p className="text-pretty">
          The planned comparison gives OpenVLA, OpenPiZero, Octo-Small, MiniVLA,
          SuSIE, and SuSIE_LL the same task and starting scene. SuSIE_LL is the
          low-level, goal-image-conditioned controller.
        </p>
        <p className="text-pretty">
          That matched six-policy set is not ready. The recordings above are
          saved experiments, not a policy ranking. No success score is claimed.
        </p>
        <Link to="/evidence">See what has been verified →</Link>
      </details>
    </div>
  );
}
