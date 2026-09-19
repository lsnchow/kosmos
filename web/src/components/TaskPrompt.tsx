/**
 * The system's primary input: a task string.
 *
 * Every policy here is language-conditioned, so a rollout has always taken an
 * instruction — this exposes it rather than adding a capability. Submitting
 * launches a rollout through the same endpoint, planner and backend as any
 * other; there is no second code path.
 *
 * The one distinction the UI must make is whether a human score exists to
 * compare the result against. Five frozen strings have one; anything typed does
 * not. That is stated before submission, not only after: the button itself says
 * which kind of run is about to start, so nobody has to be told.
 */
import { CornerDownLeft, Sparkles } from "lucide-react";
import { useState } from "react";
import { BENCHMARK_TASKS, SUGGESTED_PROMPTS } from "../lib/tasks";
import { cn } from "../lib/utils";

export type TaskPromptProps = {
  /** Launches a rollout for this instruction. Non-blocking: the tile fills in. */
  onSubmit: (instruction: string) => void;
  busy?: boolean;
};

export function TaskPrompt({ onSubmit, busy }: TaskPromptProps) {
  const [value, setValue] = useState("");
  const trimmed = value.trim();
  // Matched character for character, including the lowercase `fold`. A prompt
  // that merely resembles a benchmark task is not one, because the reference
  // cell was measured against the exact string.
  const benchmark = BENCHMARK_TASKS.find((task) => task.instruction === trimmed);

  const submit = () => {
    if (!trimmed || busy) return;
    onSubmit(trimmed);
    setValue("");
  };

  return (
    <section className="task-prompt" aria-label="Run a task">
      <form
        className="task-prompt-bar"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <label className="sr-only" htmlFor="task-prompt-input">
          Task instruction
        </label>
        <input
          id="task-prompt-input"
          className="task-prompt-input"
          type="text"
          autoComplete="off"
          spellCheck={false}
          placeholder={SUGGESTED_PROMPTS[0]}
          value={value}
          maxLength={200}
          onChange={(event) => setValue(event.target.value)}
        />
        <button type="submit" className="button button-primary" disabled={!trimmed || busy}>
          {busy ? "Starting…" : "Run"}
          <CornerDownLeft aria-hidden="true" className="size-4" />
        </button>
      </form>

      {/* Says which kind of run this is *before* it starts. */}
      <p className={cn("task-prompt-verdict", trimmed && !benchmark && "task-prompt-verdict-off")}>
        {!trimmed ? (
          <span>Pick a benchmark task, or describe one in your own words.</span>
        ) : benchmark ? (
          <span>
            <b>Benchmark task.</b> Scored against the published human result for {benchmark.id}.
          </span>
        ) : (
          <span>
            <b>Off-benchmark.</b> Runs the same pipeline; there is no human score to compare it against.
          </span>
        )}
      </p>

      <div className="task-prompt-lists">
        <div>
          <p className="task-prompt-legend">Benchmark tasks · ground truth available</p>
          <ul className="chip-row">
            {BENCHMARK_TASKS.map((task) => (
              <li key={task.id}>
                <button
                  type="button"
                  className="chip"
                  aria-pressed={task.instruction === trimmed}
                  onClick={() => setValue(task.instruction)}
                >
                  {task.instruction}
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="task-prompt-legend">
            <Sparkles aria-hidden="true" className="size-3" />
            Off-benchmark · within the policies&rsquo; training distribution
          </p>
          <ul className="chip-row">
            {SUGGESTED_PROMPTS.map((prompt) => (
              <li key={prompt}>
                <button
                  type="button"
                  className="chip chip-quiet"
                  aria-pressed={prompt === trimmed}
                  onClick={() => setValue(prompt)}
                >
                  {prompt}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
