/**
 * The five benchmark tasks, as a filter on the rollout viewport.
 *
 * Lifted out of the old landing page and placed next to the wall it scopes,
 * which is where it was always acting. The strings are the frozen registry's,
 * verbatim — including the lowercase "fold" — and selecting one sends nothing:
 * it filters what is already on screen. A world model here only ever receives a
 * registry task string, which is why this surface has no text field.
 */
import { BENCHMARK_TASKS } from "../lib/tasks";

export type TaskScopeProps = {
  scopedTask?: string;
  onScopeTask: (task: string | undefined) => void;
};

export function TaskScope({ scopedTask, onScopeTask }: TaskScopeProps) {
  const scoped = BENCHMARK_TASKS.find((task) => task.id === scopedTask);

  return (
    <section className="task-scope" aria-label="Scope the rollout viewport">
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
      {/* Nothing is said when nothing is selected: five unpressed chips already
          say that, and a sentence repeating it is read on every visit. */}
      {scoped && (
        <p className="chip-scope">
          <b>{scoped.id}</b>
          <span>· {scoped.maxSteps} max steps · others dimmed</span>
          <button type="button" className="button button-quiet" onClick={() => onScopeTask(undefined)}>
            Clear
          </button>
        </p>
      )}
    </section>
  );
}
