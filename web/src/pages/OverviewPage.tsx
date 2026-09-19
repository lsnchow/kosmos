/**
 * What the project claims, and what is not yet qualified.
 *
 * Pre-live beats of the 180 seconds: the positioning line, the called shot
 * frozen before the reference is read, and the gate blockers stated plainly.
 */
import { Expand, ShieldAlert } from "lucide-react";
import { useAppData, STARTS_PER_TASK } from "../AppData";
import { CalledShotPanel } from "../components/CalledShotPanel";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { GatePanel } from "../components/GatePanel";
import { pickString } from "../lib/format";
import { PageHeader } from "./PageHeader";

export function OverviewPage() {
  const { protocol, policies, tasks, blockers, calledShot, gates, openFreeplay } = useAppData();

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow={
          <>
            <span className="fixture-dot" />
            {pickString(protocol?.mode) ?? "mode not reported"} · unqualified
          </>
        }
        title="Measure the ruler before trusting the ranking."
        lede="Nightshift keeps generated-rollout evidence, the published real-robot reference, missingness and runtime accounting in one console. No qualified real-world conclusion is available until its gates pass."
        actions={
          <>
            <button type="button" className="button button-secondary" onClick={() => openFreeplay()}>
              <Expand aria-hidden="true" className="size-4" />
              Open free-play
            </button>
            <p>
              {policies.length > 0 && tasks.length > 0
                ? `${policies.length} policies × ${tasks.length} tasks × ${STARTS_PER_TASK} starts.`
                : "The protocol has not returned its policy and task matrix."}{" "}
              Fixture scores are engineering tests, not robot results.
            </p>
          </>
        }
      />

      <section className="qualified-blocker" aria-label="Real burst qualification status">
        <div>
          <ShieldAlert aria-hidden="true" className="size-5" />
          <div>
            <strong>Qualified real burst unavailable</strong>
            <p className="text-pretty">
              Real backends cannot be dispatched through this console until evidence gates qualify the chosen
              protocol.
            </p>
          </div>
        </div>
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
