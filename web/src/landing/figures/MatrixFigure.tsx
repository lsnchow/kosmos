/**
 * FIG.1 — the evaluation matrix filling in.
 *
 * Six policies down, five tasks across, fifty rollouts a cell. The figure draws
 * the real shape of a run: thirty cells sweeping to completion, with the episode
 * counter reaching exactly 1,500 and stopping there.
 *
 * The cells fill left-to-right, top-to-bottom rather than at random because that
 * is the order the ledger actually writes them in, and a figure that implied a
 * different execution order would be decoration rather than a diagram.
 */
import { POLICIES, TASKS } from "../content";
import { stagger, useProgress } from "./useProgress";

const ROLLOUTS_PER_CELL = 50;
/** Fifty dots a cell, laid out ten across and five down. */
const CELL_COLUMNS = 10;

export function MatrixFigure({ play }: { play: boolean }) {
  const progress = useProgress(play, 2400);
  const cellCount = POLICIES.length * TASKS.length;

  // Completed episodes across the whole matrix, derived from one scalar so the
  // readout can never disagree with the cells above it.
  let filled = 0;
  const cells = POLICIES.flatMap((policy, row) =>
    TASKS.map((task, column) => {
      const index = row * TASKS.length + column;
      const local = stagger(progress, index, cellCount, 0.35);
      const done = Math.round(local * ROLLOUTS_PER_CELL);
      filled += done;
      return { policy, task: task.prompt, row, column, local, done };
    }),
  );

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="matrix-grid">
        {cells.map((cell) => (
          <div
            key={`${cell.policy}-${cell.task}`}
            className="matrix-cell"
            title={`${cell.policy} · ${cell.task}`}
          >
            <div className="matrix-dots" aria-hidden="true">
              {Array.from({ length: ROLLOUTS_PER_CELL }, (_, dot) => {
                const column = dot % CELL_COLUMNS;
                const dotRow = Math.floor(dot / CELL_COLUMNS);
                return (
                  <span
                    key={dot}
                    className="matrix-dot"
                    data-on={dot < cell.done ? "true" : "false"}
                    style={{
                      gridColumn: column + 1,
                      gridRow: dotRow + 1,
                    }}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <dl className="matrix-readout">
        <div>
          <dt>Episodes</dt>
          <dd className="text-fg-strong">{filled.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Cells</dt>
          <dd>
            {cells.filter((cell) => cell.local >= 1).length} / {cellCount}
          </dd>
        </div>
        <div>
          <dt>Shape</dt>
          <dd>
            {POLICIES.length} × {TASKS.length} × {ROLLOUTS_PER_CELL}
          </dd>
        </div>
      </dl>
      <p className="sr-only">
        An evaluation matrix of {POLICIES.length} policies against {TASKS.length} tasks, fifty
        rollouts each, totalling {(cellCount * ROLLOUTS_PER_CELL).toLocaleString()} episodes.
      </p>
    </div>
  );
}
