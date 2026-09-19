/**
 * The frozen benchmark task registry, mirrored for the UI.
 *
 * These five strings are copied character for character from
 * `plumb/policies/tasks.py` (registry `plumb-benchmark-tasks-v1`), which the
 * implementation spec marks "verbatim, do not paraphrase" — including the
 * lowercase `fold` in the cloth task, which the spec calls out twice.
 *
 * These five are the only tasks with a published human score to compare
 * against, which is the entire reason the distinction is drawn in the UI. A
 * typed prompt runs through exactly the same pipeline; what it does not have is
 * a reference cell, and the tile says so rather than leaving the reader to
 * assume one exists.
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

/**
 * Off-benchmark prompts offered as a starting point.
 *
 * The policies were trained on BridgeData V2 — a tabletop and kitchen
 * manipulation set — so a prompt using its objects and verbs produces coherent
 * actions, while one far outside it produces incoherent actions and visibly
 * broken video. These five stay inside that distribution: same scene furniture
 * (drawer, sink, cloth, containers), same imperative "verb the object" shape,
 * one object moved per instruction.
 *
 * NOT YET VERIFIED BY TRIAL. They are chosen from the training distribution,
 * not measured against a running world model — no GPU or Baseten deployment
 * exists in this workspace to try them on. Gate A is where they get confirmed
 * or replaced; until then this list is a reasoned starting point and nothing
 * stronger, which is why nothing in the UI describes them as tested.
 */
export const SUGGESTED_PROMPTS: string[] = [
  "Pick up the cloth and put it in the sink",
  "Move the eggplant onto the towel",
  "Put the spoon in the drawer",
  "Take the lid off the pot",
  "Push the bowl to the left",
];

/** Tasks outside the frozen registry have no reference cell to compare against. */
export function isBenchmarkTask(taskId: string | undefined): boolean {
  return BENCHMARK_TASKS.some((task) => task.id === taskId);
}

/**
 * The task id the server will derive for a free-text prompt.
 *
 * Mirrors `custom_task_id` in `plumb/api.py`. It has to match: the wall keys
 * slots by `(policy, task)`, so an optimistic tile keyed differently from the
 * record that arrives two seconds later would occupy a second slot and the
 * rollout would appear twice.
 *
 * FNV-1a over the trimmed prompt is not the server's SHA-256, so this computes
 * the same *shape* rather than the same digest — which is why `launchPrompt`
 * reconciles on the run's own episode records rather than trusting this id to
 * collide. Kept deliberately simple: a browser does not need a hash here, it
 * needs a stable local key.
 */
export function customTaskId(prompt: string): string {
  let hash = 0x811c9dc5;
  for (const char of prompt.trim()) {
    hash ^= char.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return `custom:${hash.toString(16).padStart(8, "0")}`;
}
