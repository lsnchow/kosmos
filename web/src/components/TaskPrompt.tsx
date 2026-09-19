/**
 * Synthetic rehearsal input: a task string.
 *
 * This control intentionally reaches only ``AppData.launchPrompt``, which is
 * hard-bound to the synthetic backend. It does not call an ML model, a robot,
 * or the cloud diagnostic endpoint.
 *
 * Five strings have canonical registry identities. Matching one identifies the
 * instruction exactly; it does not create a measurement or comparison.
 */
import { Glyph } from "./Terminal";
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
    <section className="task-prompt" aria-label="Run a synthetic rehearsal task">
      <p className="task-prompt-legend">
        <Glyph name="warn" /> Synthetic rehearsal · no ML or robot inference
      </p>
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
          {busy ? "Starting rehearsal…" : "Run synthetic rehearsal"}
          <Glyph name="enter" />
        </button>
      </form>

      {/* Says which kind of run this is *before* it starts. */}
      <p className={cn("task-prompt-verdict", trimmed && !benchmark && "task-prompt-verdict-off")}>
        {!trimmed ? (
          <span>Pick a canonical instruction, or describe one in your own words. Every submission is unscored synthetic rehearsal.</span>
        ) : benchmark ? (
          <span>
            <b>Canonical instruction match.</b> This exactly matches registry task {benchmark.id}; the synthetic rehearsal is not a measurement.
          </span>
        ) : (
          <span>
            <b>Custom instruction.</b> This synthetic rehearsal is unscored and has no ML or robot inference.
          </span>
        )}
      </p>

      <div className="task-prompt-lists">
        <div>
          <p className="task-prompt-legend">Canonical registry instructions · exact string matching only</p>
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
            <Glyph name="sparkle" />
            Custom synthetic rehearsal prompts · unscored
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
