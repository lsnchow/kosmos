/**
 * The five benchmark tasks, as a filter on the rollout viewport.
 *
 * Lifted out of the old landing page and placed next to the wall it scopes,
 * which is where it was always acting. The strings are the frozen registry's,
 * verbatim — including the lowercase "fold" — and selecting one sends nothing:
 * it filters what is already on screen. A world model here only ever receives a
 * registry task string, which is why this surface has no text field.
 */
import { BENCHMARK_TASKS, TASK_REGISTRY_ID } from "../lib/tasks";

export type TaskScopeProps = {
  scopedTask?: string;
  onScopeTask: (task: string | undefined) => void;
};

export function TaskScope({ scopedTask, onScopeTask }: TaskScopeProps) {
  const scoped = BENCHMARK_TASKS.find((task) => task.id === scopedTask);

  return (
    <section className="task-scope" aria-label="Scope the rollout viewport">
      <p className="task-scope-lede text-pretty">
        Five fixed strings, exactly as registry <code>{TASK_REGISTRY_ID}</code> freezes them. Selecting one
        scopes what the viewport shows; non-matching tiles are dimmed, never dropped.
      </p>
      <ul className="chip-row">
        {BENCHMARK_TASKS.map((task) => {
          const active = task.id === scopedTask;
          return (
            <li key={task.id}>
              <button
                type="button"
                className="chip"
                aria-pressed={active}
                onClick={() => onScopeTask(active ? undefined : task.id)}
              >
                {task.instruction}
              </button>
            </li>
          );
        })}
      </ul>
      <p className="chip-scope">
        {scoped ? (
          <>
            <span>Viewport scoped to</span> <b>{scoped.id}</b>
            <span>· {scoped.maxSteps} max steps</span>
            <button type="button" className="button button-quiet" onClick={() => onScopeTask(undefined)}>
              Clear scope
            </button>
          </>
        ) : (
          <span>No task selected, so the viewport shows every identity it has been sent.</span>
        )}
      </p>
    </section>
  );
}
