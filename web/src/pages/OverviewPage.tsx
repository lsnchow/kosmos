import { DemoJudgePanel } from "../components/DemoJudgePanel";
import { WorldVideoGrid } from "../components/WorldVideoGrid";
import { PolicyExperiment } from "../components/PolicyExperiment";
import { PageHeader } from "./PageHeader";

/** The primary console is one saved clip and one explicit trained-judge action. */
export function OverviewPage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Robot experiments"
        title="Watch. Test. Assess."
        lede="Set up a policy experiment or explore the saved recordings below."
      />
      <PolicyExperiment />
      <WorldVideoGrid />
      <details className="gallery-guide">
        <summary>Assess a saved clip with our trained judge</summary>
        <DemoJudgePanel />
      </details>
    </div>
  );
}
