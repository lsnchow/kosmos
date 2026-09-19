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
  name: "Kosmos",
  /** The product one-liner, compressed from the build spec's. */
  promise:
    "Point us at a policy endpoint. Get a ranked report in twenty minutes — and four numbers saying how far to trust it.",
  tagline: "Policy evaluation with error bars.",
} as const;

export const CONSOLE_PATH = "/console";
export const LIVE_PATH = "/live";
export const RESULTS_PATH = "/results";
export const EVIDENCE_PATH = "/evidence";
export const PROTOCOL_PATH = "/api/protocol";

/* ------------------------------------------------------------------------ */
/* The pillar spine                                                          */
/* ------------------------------------------------------------------------ */

/**
 * Four pillars, numbered, in the order a reader has to accept them.
 *
 * The order is an argument and not a menu: the matrix is worthless if it does
 * not repeat (02), a repeatable number is worthless if you cannot trace it
 * (03), and none of it matters if you cannot watch it happen (04).
 *
 * Every `items` entry is a claim already made elsewhere in this file or in the
 * build spec. Nothing here is new marketing.
 */
export const PILLARS = [
  {
    id: "evaluate",
    number: "01",
    name: "Evaluate",
    /** Rendered as a white lead-in followed by the muted remainder. */
    lead: "Evaluate.",
    headline: "Six policies, five tasks, fifty rollouts — and not one robot arm.",
    body: "The full 6 × 5 × 50 matrix runs as recurring inference. Every episode is a row in a transactional ledger with its own artifacts, so a run that dies halfway resumes instead of restarting.",
    cta: { label: "Open the console", href: CONSOLE_PATH },
    items: [
      { key: "1.1", title: "1,500 episodes a run", body: "The same matrix the published work scored by hand." },
      { key: "1.2", title: "Four steps, four hardware profiles", body: "Policy, world model, validity gate, judge — autoscaled independently." },
      { key: "1.3", title: "Bounded and cancellable", body: "Owner leases, orphan recovery, immutable attempt artifacts." },
    ],
  },
  {
    id: "reliability",
    number: "02",
    name: "Reliability",
    lead: "Reliability.",
    headline: "Four numbers about the instrument, published before any finding.",
    body: "Three prior systems already automate this with world models. All three report accuracy. None reports precision — whether the answer repeats. These four are the product, the way a multimeter ships with a tolerance rather than a paper.",
    cta: { label: "See the report", href: RESULTS_PATH },
    items: [
      { key: "2.1", title: "Wilson intervals on every rate", body: "Never a bare percentage. Every number carries its n." },
      { key: "2.2", title: "Cluster bootstrap on tasks", body: "On tasks, not episodes — the unit that actually varies." },
      { key: "2.3", title: "Pre-registered thresholds", body: "In a timestamped commit, before the run that tests them." },
    ],
  },
  {
    id: "evidence",
    number: "03",
    name: "Evidence",
    lead: "Evidence.",
    headline: "Every figure carries its citation, or it reads “not yet measured”.",
    body: "Published numbers arrive with their table. Numbers we measured arrive with their run id and artifact hash. There is no third category, and nothing is estimated to fill a gap in the layout.",
    cta: { label: "Read the record", href: EVIDENCE_PATH },
    items: [
      { key: "3.1", title: "Provenance on every artifact", body: "Immutable, content-addressed, served from the run that made it." },
      { key: "3.2", title: "Coverage and missing-outcome bounds", body: "Horowitz–Manski bounds beside complete-case rates." },
      { key: "3.3", title: "Unsupported inference stays unavailable", body: "Indeterminate rather than approximated under a false name." },
    ],
  },
  {
    id: "console",
    number: "04",
    name: "Console",
    lead: "Console.",
    headline: "Watch fifteen hundred rollouts land in a minute.",
    body: "Twelve live tiles, a gate view, queue telemetry, and a cost sweep that stays empty until real sweep evidence exists. Nothing on this surface is fabricated to look busy.",
    cta: { label: "Start a live run", href: LIVE_PATH },
    items: [
      { key: "4.1", title: "Server-sent events, not polling", body: "Reconnects on drop; the tile wall accumulates frames in order." },
      { key: "4.2", title: "Cancel mid-matrix", body: "Cancellation is tracked through the outbox and the ledger alike." },
      { key: "4.3", title: "Synthetic mode is labelled", body: "Fixture episodes are shown separately from real clips. Always." },
    ],
  },
] as const;

export const NAV_LINKS = PILLARS.map((pillar) => ({
  label: pillar.name,
  number: pillar.number,
  href: `#${pillar.id}`,
}));

/* ------------------------------------------------------------------------ */
/* The trust row                                                             */
/* ------------------------------------------------------------------------ */

/**
 * The hero's supporting row. These are components of the stack we actually run,
 * not customers and not investors — this product has neither, and a logo wall
 * implying otherwise would be the first lie on the page.
 */
export const BUILT_ON = [
  "Baseten Chains",
  "NVIDIA Cosmos",
  "OpenVLA",
  "Octo",
  "MiniVLA",
  "V-JEPA 2",
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

/** Verbatim task strings. The lowercase `fold` in the last one is intentional. */
export const TASKS = [
  { prompt: "Close the drawer", horizon: 70 },
  { prompt: "Open the drawer", horizon: 70 },
  { prompt: "Put the eggplant in the yellow basket", horizon: 100 },
  { prompt: "Put the eggplant in the blue sink", horizon: 100 },
  { prompt: "fold the cloth from top right to bottom left", horizon: 80 },
] as const;

/** The six policies the matrix ranks. Six rows against five task columns. */
export const POLICIES = [
  "OpenVLA",
  "Octo",
  "MiniVLA",
  "Open pi-zero",
  "SuSIE",
  "SuSIE_LL",
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
 * Kept on the page, at the same weight as the pillars. An evaluation product
 * that hides its own caveats has already lost the argument it is making.
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
/* Footer                                                                    */
/* ------------------------------------------------------------------------ */

export const FOOTER_COLUMNS = [
  {
    heading: "Platform",
    links: [
      { label: "Console", href: CONSOLE_PATH },
      { label: "Live run", href: LIVE_PATH },
      { label: "Results", href: RESULTS_PATH },
      { label: "Evidence", href: EVIDENCE_PATH },
    ],
  },
  {
    heading: "Record",
    links: [
      { label: "Protocol", href: PROTOCOL_PATH, external: true },
      { label: "Third-party notices", href: "/THIRD-PARTY-NOTICES.md", external: true },
    ],
  },
] as const;

/** The footer's closing line. It is the tagline, set large. */
export const FOOTER_DISPLAY = "Measure the ruler." as const;

export const FOOTER_NOTE =
  "Page imagery is illustrative and is not model output. Generated frames carry provenance in the console." as const;
