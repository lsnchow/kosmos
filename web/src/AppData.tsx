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
import { useNow, usePolling, usePresentationMode } from "./hooks/usePolling";
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
import { applyEpisodes, applySegment, createWall, type TileSlot, type WallState } from "./lib/wall";

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
  backends: string[];
  now: number;
  /* errors and flags */
  loadError?: string;
  actionError?: string;
  submitting: boolean;
  presentation: boolean;
  setPresentation: (on: boolean) => void;
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
  const [freeplayOpen, setFreeplayOpen] = useState(false);
  const [freeplaySubject, setFreeplaySubject] = useState<string>();
  const [cancelOpen, setCancelOpen] = useState(false);
  const [scopedTask, setScopedTask] = useState<string>();
  const [viewerSlot, setViewerSlot] = useState<TileSlot>();
  const [presentation, setPresentation] = usePresentationMode();
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
    autoSelected.current = true;
    void loadRun(runs[0]);
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
  const idempotencyKey = useMemo(() => burstIdempotencyKey(identity), [identity]);
  const resolvedCalledShot = calledShot ?? calledShotFromProtocol(protocol);

  const launchBurst = useCallback(async () => {
    if (submitting) return;
    setSubmitting(true);
    setActionError(undefined);
    try {
      const run = await api.createRun(burstRequestBody(identity));
      setRuns((prior) => [run, ...prior.filter((item) => item.id !== run.id)]);
      await loadRun(run);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Could not submit the burst.");
    } finally {
      setSubmitting(false);
    }
  }, [identity, loadRun, submitting]);

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

  const backends = Array.isArray(health?.available_backends)
    ? health.available_backends.filter((item): item is string => typeof item === "string")
    : [];

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
    backends,
    now,
    loadError,
    actionError,
    submitting,
    presentation,
    setPresentation,
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
    cancelRun,
    setSixClip,
  };

  return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>;
}
