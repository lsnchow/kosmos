/**
 * Generated media, and the test that asks whether you can tell.
 */
import { useAppData } from "../AppData";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { HeroRollout } from "../components/HeroRollout";
import { SixClipPanel } from "../components/SixClipPanel";
import { PageHeader } from "./PageHeader";
import { WorldVideoQueue } from "../components/WorldVideoQueue";

export function ClipsPage() {
  const { sixClip, setSixClip, episodes } = useAppData();

  return (
    <div className="page-stack">
      <PageHeader
        title="Recording archive"
        lede="All saved world-model recordings, including short probes and repeated experiments. Watching a recording never starts a GPU job."
      />
      <ErrorBoundary region="Recording archive">
        <WorldVideoQueue />
      </ErrorBoundary>
      <details className="gallery-guide">
        <summary>Development clip exercise & presentation track</summary>
        <div className="split-grid">
          <ErrorBoundary region="Six-clip panel">
            <SixClipPanel data={sixClip} onRevealed={setSixClip} />
          </ErrorBoundary>
          <ErrorBoundary region="Presentation rollout">
            <HeroRollout episodes={episodes} />
          </ErrorBoundary>
        </div>
      </details>
    </div>
  );
}
