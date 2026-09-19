/**
 * The frozen benchmark task registry, mirrored for the UI.
 *
 * These five strings are copied character for character from
 * `plumb/policies/tasks.py` (registry `plumb-benchmark-tasks-v1`), which the
 * implementation spec marks "verbatim, do not paraphrase" — including the
 * lowercase `fold` in the cloth task, which the spec calls out twice.
 *
 * The UI only ever *selects* one of these. It never composes a prompt: free text
 * typed in a browser must not reach the world model or a primary judge, so this
 * console has no text input at all. That is why the landing page offers chips
 * rather than the prompt box its visual language was borrowed from.
 */
export type BenchmarkTask = {
  /** `task_id` in the Python registry. */
  id: string;
  /** `instruction` in the Python registry. Verbatim. */
  instruction: string;
  /** `max_steps` in the Python registry. */
  maxSteps: number;
};

export const BENCHMARK_TASKS: BenchmarkTask[] = [
  { id: "close_drawer", instruction: "Close the drawer", maxSteps: 70 },
  { id: "open_drawer", instruction: "Open the drawer", maxSteps: 70 },
  { id: "to_basket", instruction: "Put the eggplant in the yellow basket", maxSteps: 100 },
  { id: "to_sink", instruction: "Put the eggplant in the blue sink", maxSteps: 100 },
  { id: "fold_cloth", instruction: "fold the cloth from top right to bottom left", maxSteps: 80 },
];

export const TASK_REGISTRY_ID = "plumb-benchmark-tasks-v1";

export function taskInstruction(taskId: string | undefined): string | undefined {
  return BENCHMARK_TASKS.find((task) => task.id === taskId)?.instruction;
}
