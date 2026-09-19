/**
 * Generated media, and the test that asks whether you can tell.
 */
import { useAppData } from "../AppData";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { HeroRollout } from "../components/HeroRollout";
import { SixClipPanel } from "../components/SixClipPanel";
import { PageHeader } from "./PageHeader";

export function ClipsPage() {
  const { sixClip, setSixClip, episodes } = useAppData();

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Audience test · one of six is a real robot"
        title="Clips"
        lede="Clip order is randomised by the server and the real clip's identity stays sealed until reveal. A result here is an audience observation, not a measurement of the world model."
      />

      <div className="split-grid">
        <ErrorBoundary region="Six-clip panel">
          <SixClipPanel data={sixClip} onRevealed={setSixClip} />
        </ErrorBoundary>
        <ErrorBoundary region="Presentation rollout">
          <HeroRollout episodes={episodes} />
        </ErrorBoundary>
      </div>
    </div>
  );
}
