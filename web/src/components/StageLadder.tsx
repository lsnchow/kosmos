import { Glyph } from "./Terminal";
import type { Episode } from "../lib/api";
import { STAGE_ORDER, stageLadder, type StageState } from "../lib/stages";
import type { WallState } from "../lib/wall";
import { Note, Panel, SourceChip, StatusPill } from "./Primitives";

/**
 * The Chain, drawn.
 *
 * Adapted from Mirage's `ProgressSteps`: a left rail whose colour carries the
 * step state. Theirs had three steps and only pending/active/done/error, which
 * cannot express the state PLUMB is usually in — the stage ran, and the field
 * this panel wants was not reported. That state gets its own rail colour, its
 * own word, and the literal phrase "not reported" in the detail line.
 */
const PILL_WORDS: Record<StageState, string> = {
  pending: "waiting",
  active: "running",
  reported: "reported",
  unreported: "not reported",
  failed: "failed",
};

const PILL_STATUS: Record<StageState, string> = {
  pending: "waiting",
  active: "running",
  reported: "pass",
  unreported: "stale",
  failed: "failed",
};

export function StageLadder({
  wall,
  episodes,
  runId,
  runTerminal,
}: {
  wall: WallState;
  episodes: Episode[];
  runId?: string;
  runTerminal?: boolean;
}) {
  const stages = stageLadder({ wall, episodes, runTerminal });
  const reported = stages.filter((stage) => stage.state === "active" || stage.state === "reported").length;

  return (
    <Panel
      title="Chain stages"
      id="stages"
      action={<SourceChip>{runId ? `run ${runId}` : "no run selected"}</SourceChip>}
    >
      <p className="ladder-summary" role="status">
        <Glyph name="link" className="shrink-0" />
        <span>
          <b>{reported}</b> of {STAGE_ORDER.length} stages have reported for this run
        </span>
      </p>
      <ol className="ladder">
        {stages.map((stage, index) => (
          <li key={stage.key} className={`ladder-step ladder-step-${stage.state}`}>
            <span className="ladder-index" aria-hidden="true">
              {index + 1}
            </span>
            <div className="ladder-body">
              <div className="ladder-head">
                <span className="ladder-label">{stage.label}</span>
                <StatusPill status={PILL_STATUS[stage.state]}>{PILL_WORDS[stage.state]}</StatusPill>
              </div>
              <p className="ladder-detail">{stage.detail}</p>
              <p className="ladder-source">{stage.source}</p>
            </div>
          </li>
        ))}
      </ol>
      <Note summary="Where each stage's status is read from">
        Four stages, four hardware profiles, each scaling on its own — that is why the pipeline is a Chain
        and not one process. Every line above is read from a persisted record: the world stage from{" "}
        <code>segment_completed</code> events, the other three from episode records. A stage that reported
        nothing says so; frames arriving is not evidence that a policy reported an action horizon, and a
        terminal episode is not evidence that a judge scored it.
      </Note>
    </Panel>
  );
}
