/**
 * Every claim the landing page makes, with the source it came from.
 *
 * The page is marketing, which is exactly why the numbers live here rather than
 * inline: this project's whole argument is that an evaluator without error bars
 * is not an instrument, so a landing page that quietly invents a reliability
 * figure would refute itself. Three rules hold in this file.
 *
 *   1. A number is either published (it carries its citation) or measured by us
 *      (it carries its run) or it is `PENDING`. There is no fourth case, and
 *      nothing here is estimated to look better.
 *   2. The four headline reliability numbers are `PENDING` until a qualified run
 *      produces them. The page says so in the same type size as the labels.
 *   3. The cost and duration targets are labelled a target. They are a promise
 *      the console is built to keep on stage, not a receipt.
 */

/** A figure the instrument has not produced yet. Rendered, never hidden. */
export const PENDING = "not yet measured" as const;

export const PRODUCT = {
  name: "Nightshift",
  /** The product one-liner, compressed from the build spec's. */
  promise:
    "Point us at a policy endpoint. Get a ranked report in twenty minutes — and four numbers saying how far to trust it.",
  tagline: "Policy evaluation with error bars.",
} as const;

export const CONSOLE_PATH = "/console";
export const LIVE_PATH = "/live";
export const RESULTS_PATH = "/results";

export const NAV_LINKS = [
  { label: "The problem", href: "#problem" },
  { label: "The gap", href: "#called-shot" },
  { label: "How it works", href: "#pipeline" },
  { label: "The numbers", href: "#numbers" },
] as const;

/* ------------------------------------------------------------------------ */
/* The problem                                                               */
/* ------------------------------------------------------------------------ */

export const PROBLEM_STATS = [
  {
    value: "1,500",
    unit: "real trials",
    note: "Human-scored on a real arm, to produce six numbers.",
    source: "AutoEval, arXiv 2503.24278, Table 2",
  },
  {
    value: "20 min",
    unit: "cooldown",
    note: "Between episodes. The motors overheat.",
    source: "AutoEval, reported protocol",
  },
  {
    value: "3",
    unit: "prior systems",
    note: "Already automate this with world models. All report accuracy.",
    source: "Literature survey, September 2026",
  },
  {
    value: "0",
    unit: "precision figures",
    note: "None report precision — whether the answer repeats.",
    source: "Same survey. A search result, not a first",
  },
] as const;

/* ------------------------------------------------------------------------ */
/* The called shot                                                           */
/* ------------------------------------------------------------------------ */

/**
 * AutoEval Table 2 (human) and Table 3 (SIMPLER), 50 trials per cell.
 *
 * Hard-coded from the printed tables rather than downloaded, which is also how
 * the backend holds them. Per-cell rates and the tier structure are claimable.
 * The published six-policy permutation is not: MiniVLA beats Open pi-zero by one
 * trial in 250 (Fisher p = 1.00), and dropping any single task reverses it.
 */
export const CALLED_SHOT = {
  policy: "OpenVLA",
  task: "Close the drawer",
  human: { successes: 46, trials: 50, rate: 0.92, label: "Real WidowX arm, human-scored" },
  simulator: { successes: 2, trials: 50, rate: 0.04, label: "SIMPLER, the standard physics simulator" },
  gapPoints: 88,
  ours: PENDING,
  source: "AutoEval, arXiv 2503.24278 — Table 2 and Table 3",
} as const;

/** The two next-largest disagreements, so the headline cell is not cherry-picked. */
export const OTHER_GAPS = [
  { policy: "Open pi-zero", task: "Put the eggplant in the blue sink", human: 0.94, simulator: 0.12 },
  { policy: "Open pi-zero", task: "Put the eggplant in the yellow basket", human: 0.14, simulator: 0.9 },
] as const;

/* ------------------------------------------------------------------------ */
/* The pipeline                                                              */
/* ------------------------------------------------------------------------ */

/**
 * Four Chain steps with four hardware profiles, scaling independently.
 *
 * That split is the argument as much as the architecture: an H100 running a
 * diffusion model should never sit blocked on a network call to a scoring model.
 */
export const CHAIN_STEPS = [
  {
    id: "policy",
    name: "Policy",
    hardware: "GPU · one replica per policy",
    detail: "Frame in, sixteen 7-D deltas out.",
  },
  {
    id: "world-model",
    name: "World model",
    hardware: "H100 · concurrency target 1",
    detail: "Sixteen actions in, seventeen frames out. Frame 0 is dropped.",
  },
  {
    id: "gate",
    name: "Validity gate",
    hardware: "CPU · Python, no model call",
    detail: "Optical flow against commanded motion. A static dream is excluded and counted.",
  },
  {
    id: "judge",
    name: "Judge",
    hardware: "GPU · five samples, quorum of three",
    detail: "Integrity, collision, progress as separate fields. The label is computed in Python.",
  },
] as const;

export const PIPELINE_NOTES = [
  {
    label: "One frame in, seventeen out",
    body: "One conditioning frame, sixteen actions a chunk, five chunks a rollout. Sweeping one to six is how we find where drift breaks the ranking.",
  },
  {
    label: "Priced like inference, not like a lab",
    body: "Recurring inference with no marginal human cost. Every fine-tune triggers a run. More replicas, never bigger nodes.",
  },
] as const;

/** Verbatim task strings. The lowercase `fold` in the last one is intentional. */
export const TASKS = [
  { prompt: "Close the drawer", horizon: 70 },
  { prompt: "Open the drawer", horizon: 70 },
  { prompt: "Put the eggplant in the yellow basket", horizon: 100 },
  { prompt: "Put the eggplant in the blue sink", horizon: 100 },
  { prompt: "fold the cloth from top right to bottom left", horizon: 80 },
] as const;

/* ------------------------------------------------------------------------ */
/* The four numbers                                                          */
/* ------------------------------------------------------------------------ */

/**
 * The spec sheet. These four are the product, not findings about it — the same
 * way a multimeter ships with a tolerance rather than a paper.
 */
export const FOUR_NUMBERS = [
  {
    tag: "Repeatability",
    title: "Split-half reliability",
    description: "Split a cell's rollouts in half, correlate the halves. Run the evaluation twice — does it say the same thing?",
    priorArt: "No published version on a world-model evaluator.",
    value: PENDING,
  },
  {
    tag: "Resolution",
    title: "Minimum detectable difference",
    description: "The gap two policies need before we call them apart at 80% power. Below it we say we cannot tell.",
    priorArt: "Unpublished for this class of evaluator.",
    value: PENDING,
  },
  {
    tag: "Economics",
    title: "Cost against ranking fidelity",
    description: "Ranking agreement against GPU-seconds, swept over resolution, steps and chunk count.",
    priorArt: "Unpublished. Needs a platform that meters this finely.",
    value: PENDING,
  },
  {
    tag: "Horizon",
    title: "Drift against ranking fidelity",
    description: "Where the dream stops preserving the ranking. Past it the video still looks plausible and the order is wrong.",
    priorArt: "Not measured anywhere we could find.",
    value: PENDING,
  },
] as const;

/** Validation, deliberately not a headline. */
export const VALIDATION_NOTE = {
  body: "Agreement with the real arm is validation, not a headline — against a published bar of Pearson 0.78.",
  source: "Published correlation on the same WidowX setup",
} as const;

/**
 * The two halves of the product, in the order a customer meets them: first we
 * establish the instrument is trustworthy, then we run it at a scale that makes
 * it worth having.
 */
export const SERVICES = [
  {
    tag: "Instrument",
    title: "Calibrate, then rank",
    body: "150 clips, two annotators, fifty overlapping — the human ceiling before any kappa. Judge at temperature 0.7, five samples, quorum of three.",
    media: "research" as const,
  },
  {
    tag: "Scale",
    title: "Then run it at burst",
    body: "The same evaluation over a hundred replicas, at the cheap end of the sweep we showed still ranks correctly.",
    media: "craft" as const,
  },
] as const;

/* ------------------------------------------------------------------------ */
/* The stack                                                                 */
/* ------------------------------------------------------------------------ */

export const STACK = [
  {
    layer: "Orchestration",
    choice: "Baseten Chains",
    note: "Four steps, four hardware profiles, autoscaled independently.",
  },
  {
    layer: "World model",
    choice: "nvidia/Cosmos3-Nano",
    note: "Forward dynamics, domain bridge_orig_lerobot. 33 GB.",
  },
  { layer: "Policy", choice: "openvla/openvla-7b", note: "15.1 GB." },
  {
    layer: "Policy",
    choice: "Stanford-ILIAD/minivla-vq-bridge-prismatic",
    note: "5.6 GB, with the pretrain_vq codebook.",
  },
  { layer: "Policy", choice: "rail-berkeley/octo-small", note: "v1.0, 0.55 GB." },
  {
    layer: "Judge",
    choice: "VLM rubric, distilled to facebook/vjepa2-vitl-fpc64-256",
    note: "1.3 GB. Smaller and faster than one frontier call.",
  },
  {
    layer: "Distillation",
    choice: "Baseten Training Jobs",
    note: "LoRA at Baseten's published optimum: lr 1e-3, r=64, alpha 32.",
  },
  {
    layer: "Fan-out",
    choice: "async_predict and async_queue_status",
    note: "Rollout fan-out, and the live queue telemetry.",
  },
] as const;

/* ------------------------------------------------------------------------ */
/* Economics                                                                 */
/* ------------------------------------------------------------------------ */

/**
 * A target, and labelled one everywhere it appears. The console holds the same
 * figures with `status: "unmeasured"` and will not promote them until a burst
 * run produces a receipt.
 */
export const BURST_TARGET = {
  episodes: 1500,
  seconds: 60,
  usd: 11.25,
  replicas: 100,
  gpuSecondsPerRollout: 4,
  status: "target, not a receipt",
} as const;

/* ------------------------------------------------------------------------ */
/* Limits                                                                    */
/* ------------------------------------------------------------------------ */

/**
 * On the landing page, above the fold of the CTA, at the same weight as the
 * features. An evaluation product that hides its own caveats has already lost
 * the argument it is making.
 */
export const LIMITS = [
  {
    title: "Not qualified yet",
    body: "Synthetic mode, said on every surface. Real backends stay blocked until the gates pass.",
  },
  {
    title: "A bias we have not bounded",
    body: "The policies and the world model share a backbone lineage. We can name it; we cannot size it.",
  },
  {
    title: "A defect in the anchor",
    body: "Octo scores 0 of 250 here; other published work puts it at 43.3% and 20%. We report agreement with it and without.",
  },
  {
    title: "What we did not build",
    body: "No world-model adaptation — every shipped recipe uses eight H100s. No SIMPLER install; we use its table.",
  },
] as const;

/* ------------------------------------------------------------------------ */
/* Statistics                                                                */
/* ------------------------------------------------------------------------ */

export const METHOD_CHIPS = [
  "Wilson score intervals",
  "Cluster bootstrap on tasks, not episodes",
  "Horowitz–Manski bounds beside complete-case rates",
  "Pre-registered thresholds in a timestamped commit",
  "Every number carries its n",
] as const;
