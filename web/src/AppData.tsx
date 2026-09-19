/**
 * One store for the whole console.
 *
 * Every page reads from here, which is not a stylistic preference: pages mount
 * under a router `<Outlet/>`, and you cannot drill props through an `<Outlet/>`.
 * Before the split this state lived as sixteen `useState` hooks in `App.tsx`
 * and reached components as up to eleven props each.
 *
 * Polling and the event stream live here too, so navigating between pages does
 * not refetch and does not drop the SSE connection mid-run. That matters on
 * stage: the burst is started before the pitch begins, and clicking to another
 * page must not restart it or lose the tiles it has already filled.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { calledShotFromProtocol } from "./components/CalledShotPanel";
import { gateBlockers } from "./components/GatePanel";
import { pushSample, type ReplicaSample } from "./components/ReplicaChart";
import { useNow, usePolling } from "./hooks/usePolling";
import { useRunStream, type StreamStatus } from "./hooks/useRunStream";
import {
  api,
  isRecord,
  type AnalysisResponse,
  type CalledShot,
  type Episode,
  type Experiment,
  type Gate,
  type JsonRecord,
  type Run,
  type SixClipResponse,
  type SweepResponse,
  type Telemetry,
} from "./lib/api";
import { burstIdempotencyKey, burstIdentity, burstRequestBody, type BurstIdentity } from "./lib/burst";
import { isTerminalStatus, pickNumber, pickString, toDate } from "./lib/format";
import { BENCHMARK_TASKS, customTaskId } from "./lib/tasks";
import { applyEpisodes, applySegment, adoptTaskId, createWall, reserveSlot, type TileSlot, type WallState } from "./lib/wall";

export const BASE_REFRESH_MS = 10_000;
export const STARTS_PER_TASK = 50;
const RUN_SEED = 20260919;

export type ProtocolOption = { id: string; name: string };

export function asOptions(value: unknown): ProtocolOption[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (typeof item === "string") return [{ id: item, name: item }];
    if (!isRecord(item)) return [];
    const id = pickString(item.id, item.name, item.policy, item.task, item.slug);
    const name = pickString(item.display_name, item.label, item.name, item.id) ?? id;
    return id && name ? [{ id, name }] : [];
  });
}

/** A run started from the prompt bar, which the console shows but never adopts. */
export function isExplorationRun(run: Run): boolean {
  const config = run.config;
  return isRecord(config) && config.cohort === "exploration";
}

export type AppDataValue = {
  /* control plane */
  health?: JsonRecord;
  protocol?: JsonRecord;
  gates: Gate[];
  runs: Run[];
  sweeps: SweepResponse;
  experiments: Experiment[];
  sixClip?: SixClipResponse;
  calledShot?: CalledShot;
  /* selected run */
  activeRun?: Run;
  episodes: Episode[];
  analysis: AnalysisResponse;
  telemetry?: Telemetry;
  samples: ReplicaSample[];
  wall: WallState;
  runIsTerminal: boolean;
  stream: { status: StreamStatus; retryCount: number };
  /* derived */
  policies: ProtocolOption[];
  tasks: ProtocolOption[];
  blockers: string[];
  plannedEpisodes?: number;
  identity: BurstIdentity;
  idempotencyKey: string;
  now: number;
  /* errors and flags */
  loadError?: string;
  actionError?: string;
  submitting: boolean;
  /* ui selection */
  scopedTask?: string;
  setScopedTask: (task: string | undefined) => void;
  viewerSlot?: TileSlot;
  setViewerSlot: (slot: TileSlot | undefined) => void;
  freeplayOpen: boolean;
  freeplaySubject?: string;
  openFreeplay: (slot?: TileSlot) => void;
  setFreeplayOpen: (open: boolean) => void;
  cancelOpen: boolean;
  setCancelOpen: (open: boolean) => void;
  /* actions */
  refresh: () => Promise<void>;
  loadRun: (run: Run) => Promise<void>;
  launchBurst: () => Promise<void>;
  /** Launch one rollout for a task string. Additive: never resets the wall. */
  launchPrompt: (instruction: string) => Promise<void>;
  promptBusy: boolean;
  cancelRun: () => Promise<void>;
  setSixClip: (next: SixClipResponse) => void;
};

const AppDataContext = createContext<AppDataValue | undefined>(undefined);

export function useAppData(): AppDataValue {
  const value = useContext(AppDataContext);
  if (!value) throw new Error("useAppData must be used inside <AppDataProvider>");
  return value;
}

export function AppDataProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<JsonRecord>();
  const [protocol, setProtocol] = useState<JsonRecord>();
  const [gates, setGates] = useState<Gate[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [sweeps, setSweeps] = useState<SweepResponse>({ points: [] });
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [sixClip, setSixClip] = useState<SixClipResponse>();
  const [calledShot, setCalledShot] = useState<CalledShot>();
  const [activeRun, setActiveRun] = useState<Run>();
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResponse>({});
  const [telemetry, setTelemetry] = useState<Telemetry>();
  const [samples, setSamples] = useState<ReplicaSample[]>([]);
  const [wall, setWall] = useState(() => createWall());
  const [loadError, setLoadError] = useState<string>();
  const [actionError, setActionError] = useState<string>();
  const [submitting, setSubmitting] = useState(false);
  const [burstAttempt, setBurstAttempt] = useState(0);
  const [promptBusy, setPromptBusy] = useState(false);
  /** Runs started from the prompt bar, merged into the wall alongside the main run. */
  const [promptRuns, setPromptRuns] = useState<string[]>([]);
  const [freeplayOpen, setFreeplayOpen] = useState(false);
  const [freeplaySubject, setFreeplaySubject] = useState<string>();
  const [cancelOpen, setCancelOpen] = useState(false);
  const [scopedTask, setScopedTask] = useState<string>();
  const [viewerSlot, setViewerSlot] = useState<TileSlot>();
  const now = useNow();
  const lastSampleAt = useRef<number>();

  const refresh = useCallback(async () => {
    const outcomes = await Promise.allSettled([
      api.health(),
      api.protocol(),
      api.gates(),
      api.runs(),
      api.sweeps(),
      api.experiments(),
      api.sixClip(),
      api.calledShot(),
    ]);
    const [healthResult, protocolResult, gatesResult, runsResult, sweepsResult, experimentsResult, clipsResult, calledShotResult] =
      outcomes;
    // The first five are the core control plane. Imported evidence, the clip
    // panel and the called shot are additive: an older API process can lack them
    // without making the ledger or protocol unavailable.
    const errors = outcomes
      .slice(0, 5)
      .flatMap((outcome) =>
        outcome.status === "rejected"
          ? [outcome.reason instanceof Error ? outcome.reason.message : "Request failed"]
          : [],
      );
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    if (protocolResult.status === "fulfilled") setProtocol(protocolResult.value);
    if (gatesResult.status === "fulfilled") setGates(gatesResult.value.gates ?? []);
    if (runsResult.status === "fulfilled") setRuns(runsResult.value.runs ?? []);
    if (sweepsResult.status === "fulfilled") setSweeps(sweepsResult.value);
    if (experimentsResult.status === "fulfilled") setExperiments(experimentsResult.value.experiments ?? []);
    if (clipsResult.status === "fulfilled") setSixClip(clipsResult.value);
    if (calledShotResult.status === "fulfilled") setCalledShot(calledShotResult.value);
    setLoadError(errors.length > 0 ? errors.join(" · ") : undefined);
  }, []);

  usePolling(refresh, BASE_REFRESH_MS);

  const loadRun = useCallback(async (run: Run) => {
    const runId = run.id;
    if (!runId) return;
    setActiveRun(run);
    setEpisodes([]);
    setAnalysis({});
    setWall(createWall());
    setSamples([]);
    setTelemetry(undefined);
    lastSampleAt.current = undefined;
    setActionError(undefined);
    const results = await Promise.allSettled([api.run(runId), api.episodes(runId), api.analysis(runId)]);
    if (results[0].status === "fulfilled") setActiveRun(results[0].value);
    if (results[1].status === "fulfilled") {
      const rows = results[1].value.episodes ?? [];
      setEpisodes(rows);
      setWall((prior) => applyEpisodes(prior, rows));
    }
    if (results[2].status === "fulfilled") setAnalysis(results[2].value);
    const failures = results.flatMap((result) =>
      result.status === "rejected"
        ? [result.reason instanceof Error ? result.reason.message : "Run data request failed"]
        : [],
    );
    if (failures.length > 0) setActionError(failures.join(" · "));
  }, []);

  // On stage the batch is started before the app is on screen, so the console
  // adopts the newest run by itself. Without this the wall is empty until
  // somebody picks a run from a dropdown mid-pitch.
  const autoSelected = useRef(false);
  useEffect(() => {
    if (autoSelected.current || activeRun?.id || runs.length === 0) return;
    // Skip exploration runs. A typed prompt is one episode and is meant to
    // appear *beside* the study on the wall, not to become the run the console
    // is about -- selecting it would reset the wall to a single tile and drop
    // the twelve the presenter is talking over.
    const study = runs.find((run) => !isExplorationRun(run)) ?? runs[0];
    autoSelected.current = true;
    void loadRun(study);
  }, [activeRun?.id, loadRun, runs]);

  const runIsTerminal = isTerminalStatus(activeRun?.status);
  const stream = useRunStream(
    activeRun?.id,
    {
      onRun: (run) => setActiveRun(run),
      onEpisodes: (rows) => {
        setEpisodes(rows);
        setWall((prior) => applyEpisodes(prior, rows));
      },
      onAnalysis: (cells) => setAnalysis((prior) => ({ ...prior, cells })),
      onTelemetry: (next) => {
        setTelemetry(next);
        // Plot only against the server's own timestamp. Without one there is no
        // position on the time axis, so the sample is dropped rather than placed
        // at the browser's clock.
        const stamp = toDate(next.fresh_at ?? next.timestamp);
        if (!stamp) return;
        const t = stamp.valueOf();
        if (lastSampleAt.current === t) return;
        lastSampleAt.current = t;
        setSamples((prior) =>
          pushSample(prior, { t, active: pickNumber(next.active_replicas), desired: pickNumber(next.desired_replicas) }),
        );
      },
      onSegment: (segment) => setWall((prior) => applySegment(prior, segment)),
      onMalformed: (message) => setActionError(message),
    },
    { runTerminal: runIsTerminal },
  );

  // A terminal run still needs its finalized analysis; the stream stays open for
  // late telemetry and cost reconciliation, so this fires on the transition.
  useEffect(() => {
    if (!activeRun?.id || !runIsTerminal) return;
    void api
      .analysis(activeRun.id)
      .then((response) => setAnalysis(response))
      .catch(() => undefined);
  }, [activeRun?.id, runIsTerminal]);

  const policies = useMemo(() => asOptions(protocol?.policies), [protocol]);
  const tasks = useMemo(() => asOptions(protocol?.tasks), [protocol]);
  const blockers = useMemo(() => gateBlockers(gates), [gates]);
  const plannedEpisodes = useMemo(() => {
    const declared = pickNumber(protocol?.total);
    if (declared !== undefined) return declared;
    if (policies.length === 0 || tasks.length === 0) return undefined;
    return policies.length * tasks.length * STARTS_PER_TASK;
  }, [policies.length, protocol?.total, tasks.length]);

  const identity = useMemo(
    () =>
      burstIdentity({
        policies: policies.map((policy) => policy.id),
        tasks: tasks.map((task) => task.id),
        startsPerTask: STARTS_PER_TASK,
        seed: RUN_SEED,
      }),
    [policies, tasks],
  );
  // The attempt the *next* press will use. While a run is alive it is the
  // current one, so a duplicate submission collapses into that run. Once the run
  // is terminal the next press is a deliberate new run and gets its own key --
  // without this the stage burst would return the pre-roll run finished before
  // the pitch began.
  const nextAttempt = activeRun && runIsTerminal ? burstAttempt + 1 : burstAttempt;
  const idempotencyKey = useMemo(
    () => burstIdempotencyKey(identity, nextAttempt),
    [identity, nextAttempt],
  );
  const resolvedCalledShot = calledShot ?? calledShotFromProtocol(protocol);

  const launchBurst = useCallback(async () => {
    if (submitting) return;
    setSubmitting(true);
    setActionError(undefined);
    setBurstAttempt(nextAttempt);
    try {
      const run = await api.createRun(burstRequestBody(identity, nextAttempt));
      setRuns((prior) => [run, ...prior.filter((item) => item.id !== run.id)]);
      await loadRun(run);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Could not submit the burst.");
    } finally {
      setSubmitting(false);
    }
  }, [identity, loadRun, nextAttempt, submitting]);

  const launchPrompt = useCallback(
    async (instruction: string) => {
      const text = instruction.trim();
      if (!text || promptBusy) return;
      setPromptBusy(true);
      setActionError(undefined);
      // The tile is claimed before the request is sent, so it is on screen for
      // the whole round trip. A five-chunk rollout takes seconds; a stage beat
      // cannot open with an empty grid waiting for a POST to return.
      const policy = policies[0]?.id ?? "OpenVLA";
      const benchmark = BENCHMARK_TASKS.some((task) => task.instruction === text);
      const taskId = benchmark
        ? (BENCHMARK_TASKS.find((task) => task.instruction === text)?.id ?? text)
        : customTaskId(text);
      setWall((prior) => reserveSlot(prior, { policy, task: taskId, instruction: text, benchmark }));
      try {
        const run = await api.createRun({
          mode: "synthetic",
          backend: "synthetic",
          policies: [policy],
          tasks: benchmark ? [taskId] : [],
          prompts: benchmark ? [] : [text],
          starts_per_task: 1,
          seed: RUN_SEED,
          // Same principle as the burst key: stable within one submission so a
          // double-fire collapses, distinct across submissions so running the
          // same task twice on purpose is two runs.
          idempotency_key: `prompt-${customTaskId(text).slice(7)}-${promptRuns.length}`,
        });
        setRuns((prior) => [run, ...prior.filter((item) => item.id !== run.id)]);
        // The server assigned the real task id; move the reserved tile onto it
        // so the episodes that follow merge into this slot rather than claiming
        // another one.
        const assigned = isRecord(run.config) && Array.isArray(run.config.tasks) ? run.config.tasks : [];
        const serverTask = assigned.find((value): value is string => typeof value === "string");
        if (serverTask) setWall((prior) => adoptTaskId(prior, policy, taskId, serverTask));
        // Registered, not selected. Selecting it would call loadRun, which
        // resets the wall -- destroying the twelve tiles this rollout is meant
        // to appear beside.
        if (run.id) setPromptRuns((prior) => (prior.includes(run.id!) ? prior : [...prior, run.id!]));
      } catch (error) {
        setActionError(error instanceof Error ? error.message : "Could not start that task.");
      } finally {
        setPromptBusy(false);
      }
    },
    [policies, promptBusy, promptRuns.length],
  );

  // Prompt rollouts are merged into the existing wall on the same cadence the
  // rest of the console polls at, through the same `applyEpisodes` reducer the
  // main run uses. No second rendering path, and no new endpoint.
  useEffect(() => {
    if (promptRuns.length === 0) return;
    let cancelled = false;
    const merge = async () => {
      for (const id of promptRuns) {
        try {
          const response = await api.episodes(id);
          const rows = response.episodes ?? [];
          if (!cancelled && rows.length > 0) setWall((prior) => applyEpisodes(prior, rows));
        } catch {
          // A prompt rollout that cannot be read is not a console-wide failure;
          // its tile keeps saying "generating" rather than claiming a result.
        }
      }
    };
    void merge();
    const timer = setInterval(() => void merge(), 2_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [promptRuns]);

  const cancelRun = useCallback(async () => {
    if (!activeRun?.id) return;
    setActionError(undefined);
    try {
      const updated = await api.cancelRun(activeRun.id);
      setActiveRun(updated);
      setRuns((prior) => prior.map((run) => (run.id === updated.id ? updated : run)));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Could not cancel this run.");
    } finally {
      setCancelOpen(false);
    }
  }, [activeRun?.id]);

  const openFreeplay = useCallback((slot?: TileSlot) => {
    setFreeplaySubject(slot ? `${slot.policy} · ${slot.task}` : undefined);
    setFreeplayOpen(true);
  }, []);

  const value: AppDataValue = {
    health,
    protocol,
    gates,
    runs,
    sweeps,
    experiments,
    sixClip,
    calledShot: resolvedCalledShot,
    activeRun,
    episodes,
    analysis,
    telemetry,
    samples,
    wall,
    runIsTerminal,
    stream: { status: stream.status, retryCount: stream.retryCount },
    policies,
    tasks,
    blockers,
    plannedEpisodes,
    identity,
    idempotencyKey,
    now,
    loadError,
    actionError,
    submitting,
    scopedTask,
    setScopedTask,
    viewerSlot,
    setViewerSlot,
    freeplayOpen,
    freeplaySubject,
    openFreeplay,
    setFreeplayOpen,
    cancelOpen,
    setCancelOpen,
    refresh,
    loadRun,
    launchBurst,
    launchPrompt,
    promptBusy,
    cancelRun,
    setSixClip,
  };

  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>;
}
