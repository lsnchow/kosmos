import { DemoJudgePanel } from "../components/DemoJudgePanel";
import { PageHeader } from "./PageHeader";
import { WorldVideoGrid } from "../components/WorldVideoGrid";
import { LiveDemoPanel } from "../components/LiveDemo";
import { LivePipeline } from "../components/LivePipeline";

/** The primary console is one saved clip and one explicit trained-judge action. */
export function OverviewPage() {
  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Robot experiments"
        title="Generate. Inspect. Assess."
        lede="Start a fresh rollout or reopen a past run."
      />
      <LivePipeline />
      <LiveDemoPanel historyOnly />
      <WorldVideoGrid />
      <details className="gallery-guide"><summary>Past drawer assessments</summary><DemoJudgePanel /></details>
    </div>
  );
}
