import * as AlertDialog from "@radix-ui/react-alert-dialog";
import { CircleAlert, Expand, ExternalLink, Presentation, RotateCcw, ShieldAlert } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BurstPanel } from "./components/BurstPanel";
import { CalledShotPanel, calledShotFromProtocol } from "./components/CalledShotPanel";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { FreeplayDialog } from "./components/FreeplayDialog";
import { GalleryModal } from "./components/GalleryModal";
import { GatePanel, gateBlockers } from "./components/GatePanel";
import { HeroRollout } from "./components/HeroRollout";
import { LandingPage } from "./components/LandingPage";
import { Note } from "./components/Primitives";
import { pushSample, type ReplicaSample } from "./components/ReplicaChart";
import { ProvenanceBadge, RolloutWall } from "./components/RolloutWall";
import { RunLedgerPanel } from "./components/RunLedgerPanel";
import { Scoreboard } from "./components/Scoreboard";
import { SixClipPanel } from "./components/SixClipPanel";
import { SmokeEvidencePanel } from "./components/SmokeEvidencePanel";
import { StageLadder } from "./components/StageLadder";
import { SweepPanel } from "./components/SweepPanel";
import { TelemetryStrip } from "./components/TelemetryStrip";
import { useNow, usePolling, usePresentationMode } from "./hooks/usePolling";
import { useRunStream } from "./hooks/useRunStream";
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
import { burstIdempotencyKey, burstIdentity, burstRequestBody } from "./lib/burst";
import { formatCount, isTerminalStatus, pickNumber, pickString, toDate } from "./lib/format";
import { taskInstruction } from "./lib/tasks";
import { cn } from "./lib/utils";
import { applyEpisodes, applySegment, createWall, type TileSlot } from "./lib/wall";

const BASE_REFRESH_MS = 10_000;
const STARTS_PER_TASK = 50;
const RUN_SEED = 20260919;

type ProtocolOption = { id: string; name: string };

function asOptions(value: unknown): ProtocolOption[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (typeof item === "string") return [{ id: item, name: item }];
    if (!isRecord(item)) return [];
    const id = pickString(item.id, item.name, item.policy, item.task, item.slug);
    const name = pickString(item.display_name, item.label, item.name, item.id) ?? id;
    return id && name ? [{ id, name }] : [];
  });
}

export default function App() {
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
  /** A task id from the frozen registry, chosen on the landing page. */
  const [scopedTask, setScopedTask] = useState<string>();
  /** The wall tile currently blown up to full screen. */
  const [viewerSlot, setViewerSlot] = useState<TileSlot>();
  const [presentation, setPresentation] = usePresentationMode();
  const now = useNow();
  const lastSampleAt = useRef<number>();

  const refreshBase = useCallback(async () => {
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

  usePolling(refreshBase, BASE_REFRESH_MS);

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
  const stream = useRunStream(activeRun?.id, {
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
        pushSample(prior, {
          t,
          active: pickNumber(next.active_replicas),
          desired: pickNumber(next.desired_replicas),
        }),
      );
    },
    onSegment: (segment) => setWall((prior) => applySegment(prior, segment)),
    onMalformed: (message) => setActionError(message),
  }, { runTerminal: runIsTerminal });

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

  const launchBurst = async () => {
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
  };

  const cancelRun = async () => {
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
  };

  const openFreeplay = (slot?: TileSlot) => {
    setFreeplaySubject(slot ? `${slot.policy} · ${slot.task}` : undefined);
    setFreeplayOpen(true);
  };

  /**
   * Move from the landing page to a console region. The console is below the
   * landing in the same document, so this is a scroll plus a focus move — there
   * is no route to break and nothing unmounts mid-pitch.
   */
  const enterConsole = (anchor: string) => {
    const target = typeof document === "undefined" ? null : document.getElementById(anchor);
    target?.scrollIntoView({ block: "start" });
    if (target instanceof HTMLElement) target.focus({ preventScroll: true });
  };

  const backends = Array.isArray(health?.available_backends)
    ? health.available_backends.filter((item): item is string => typeof item === "string")
    : [];

  return (
    <div className="app-shell">
      <a className="skip-link" href="#console">
        Skip to console
      </a>
      <header className="topbar">
        <a className="brand" href="#main" aria-label="Nightshift landing page">
          <span className="brand-mark">N</span>
          <span>Nightshift</span>
        </a>
        <nav aria-label="Console sections">
          <a href="#stages">Chain</a>
          <a href="#sixclip">Clips</a>
          <a href="#calledshot">Called shot</a>
          <a href="#rollouts">Wall</a>
          <a href="#burst">Burst</a>
          <a href="#sweep">Dial</a>
          <a href="#scoreboard">Scores</a>
          <a href="#gates">Gates</a>
        </nav>
        <div className="topbar-right">
          <span className="api-health">
            <span className={cn("health-dot", health ? "health-known" : "health-pending")} />
            API {pickString(health?.status) ?? "checking"}
            {backends.length > 0 && <span className="backend-list">· {backends.join(", ")}</span>}
          </span>
          <button
            type="button"
            className={cn("button", presentation ? "button-primary" : "button-quiet")}
            onClick={() => setPresentation(!presentation)}
            aria-pressed={presentation}
          >
            <Presentation aria-hidden="true" className="size-4" />
            Presentation mode
          </button>
          <button type="button" className="button button-quiet" onClick={() => void refreshBase()}>
            <RotateCcw aria-hidden="true" className="size-4" />
            Refresh
          </button>
        </div>
      </header>

      <main id="main">
        <ErrorBoundary region="Landing page">
          <LandingPage
            runs={runs}
            episodes={episodes}
            gates={gates}
            protocol={protocol}
            scopedTask={scopedTask}
            onScopeTask={setScopedTask}
            onEnterConsole={enterConsole}
            onOpenFreeplay={() => openFreeplay()}
          />
        </ErrorBoundary>

        {/*
          The console keeps its own id and stays in the document below the
          landing, so the skip link, every in-page anchor and the whole stage
          sequence work exactly as before.
        */}
        <div id="console" className="console" tabIndex={-1}>
        <section className="masthead">
          <div>
            <p className="eyebrow">
              <span className="fixture-dot" />
              {pickString(protocol?.mode) ?? "mode not reported"} · unqualified
            </p>
            <h1 className="text-balance">Measure the ruler before trusting the ranking.</h1>
            <p className="masthead-copy text-pretty">
              Nightshift keeps generated-rollout evidence, the published real-robot reference, missingness and
              runtime accounting in one console. No qualified real-world conclusion is available until its
              gates pass.
            </p>
          </div>
          <div className="masthead-actions">
            <button type="button" className="button button-secondary" onClick={() => openFreeplay()}>
              <Expand aria-hidden="true" className="size-4" />
              Open free-play
            </button>
            <p>
              {policies.length > 0 && tasks.length > 0
                ? `${policies.length} policies × ${tasks.length} tasks × ${STARTS_PER_TASK} starts.`
                : "The protocol has not returned its policy and task matrix."}{" "}
              Fixture scores are engineering tests, not robot results.
            </p>
          </div>
        </section>

        {(loadError ?? actionError) && (
          <div className="error-banner" role="alert">
            <CircleAlert aria-hidden="true" className="size-5" />
            <span>{actionError ?? loadError}</span>
          </div>
        )}

        <section className="qualified-blocker" aria-label="Real burst qualification status">
          <div>
            <ShieldAlert aria-hidden="true" className="size-5" />
            <div>
              <strong>Qualified real burst unavailable</strong>
              <p className="text-pretty">
                Real backends cannot be dispatched through this console until evidence gates qualify the
                chosen protocol.
              </p>
            </div>
          </div>
          <ul>
            {blockers.slice(0, 3).map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        </section>

        <section className="talk-grid">
          <ErrorBoundary region="Six-clip panel">
            <SixClipPanel data={sixClip} onRevealed={setSixClip} />
          </ErrorBoundary>
          <ErrorBoundary region="Called shot">
            <CalledShotPanel calledShot={resolvedCalledShot} />
          </ErrorBoundary>
        </section>

        <section id="rollouts" className="workspace-grid" tabIndex={-1}>
          {/* The Chain ladder sits directly under the wall it describes: the
              tiles show frames arriving, the ladder shows which of the four
              stages actually reported them. */}
          <div className="workspace-main">
            <ErrorBoundary region="Rollout viewport">
              <RolloutWall
                wall={wall}
                runId={activeRun?.id}
                onExpand={setViewerSlot}
                onDrive={openFreeplay}
                scopedTask={scopedTask}
                scopedInstruction={taskInstruction(scopedTask)}
              />
            </ErrorBoundary>
            <ErrorBoundary region="Chain stages">
              <StageLadder
                wall={wall}
                episodes={episodes}
                runId={activeRun?.id}
                runTerminal={runIsTerminal}
              />
            </ErrorBoundary>
          </div>
          <aside className="side-stack">
            <ErrorBoundary region="Run ledger">
              <RunLedgerPanel runs={runs} activeRun={activeRun} onSelect={(run) => void loadRun(run)} />
            </ErrorBoundary>
            <ErrorBoundary region="Presentation rollout">
              <HeroRollout episodes={episodes} />
            </ErrorBoundary>
          </aside>
        </section>

        <ErrorBoundary region="Live telemetry">
          <TelemetryStrip
            run={activeRun}
            telemetry={telemetry}
            streamStatus={stream.status}
            retryCount={stream.retryCount}
            ledgerState={analysis.ledger_state}
            now={now}
          />
        </ErrorBoundary>

        <section className="burst-grid">
          <ErrorBoundary region="Burst panel">
            <BurstPanel
              run={activeRun}
              telemetry={telemetry}
              samples={samples}
              submitting={submitting}
              policyCount={policies.length}
              taskCount={tasks.length}
              plannedEpisodes={plannedEpisodes}
              idempotencyKey={idempotencyKey}
              onLaunch={() => void launchBurst()}
              onCancel={() => setCancelOpen(true)}
            />
          </ErrorBoundary>
          <div id="sweep">
            <ErrorBoundary region="Cost–fidelity slider">
              <SweepPanel sweeps={sweeps} />
            </ErrorBoundary>
          </div>
        </section>

        <section className="score-section">
          <ErrorBoundary region="Scoreboard">
            <Scoreboard analysis={analysis} protocol={protocol} />
          </ErrorBoundary>
        </section>

        <section className="lower-grid" aria-label="Qualification and imported evidence">
          <ErrorBoundary region="Smoke evidence">
            <SmokeEvidencePanel experiments={experiments} />
          </ErrorBoundary>
          <div className="right-lower">
            <ErrorBoundary region="Qualification gates">
              <GatePanel gates={gates} />
            </ErrorBoundary>
          </div>
        </section>

        <footer className="footer-note">
          <span>Nightshift measurement console</span>
          <span>
            Sources: application ledger, persisted artifacts, declared platform telemetry, and published
            reference.
          </span>
          <a href="/THIRD-PARTY-NOTICES.md" target="_blank" rel="noreferrer">
            Third-party notices <ExternalLink aria-hidden="true" className="size-3" />
          </a>
          <a href="/api/protocol" target="_blank" rel="noreferrer">
            Protocol record <ExternalLink aria-hidden="true" className="size-3" />
          </a>
        </footer>
        <Note>
          Panels update on a {BASE_REFRESH_MS / 1000}-second poll and, for the selected run, on its event
          stream. Every number carries the source that produced it; where a source reported nothing, the panel
          says so rather than showing a zero.
        </Note>
        </div>
      </main>

      {/* One wall tile at full screen. Provenance travels with the clip: a
          replayed frame must not lose its label by being made bigger. */}
      <GalleryModal
        open={viewerSlot !== undefined}
        onClose={() => setViewerSlot(undefined)}
        title={viewerSlot ? `${viewerSlot.policy} · ${viewerSlot.task}` : ""}
        subtitle={viewerSlot?.episodeId ?? "episode id not reported"}
        frames={viewerSlot?.frames ?? []}
        emptyReason="No persisted segment event has reached this slot, so there is no frame to enlarge."
        badge={viewerSlot ? <ProvenanceBadge slot={viewerSlot} /> : undefined}
        meta={
          viewerSlot
            ? [
                { label: "Run", value: viewerSlot.runId ?? "not reported" },
                { label: "Segments", value: formatCount(viewerSlot.segmentCount) },
                {
                  label: "Frames",
                  value:
                    viewerSlot.certifiedFrameCount === undefined
                      ? `${formatCount(viewerSlot.frames.length)} · certified count not reported`
                      : `${formatCount(viewerSlot.certifiedFrameCount)} certified`,
                },
                { label: "Resolution", value: viewerSlot.resolution ?? "not reported" },
              ]
            : []
        }
        footer={
          <button
            type="button"
            className="button button-primary"
            onClick={() => {
              const slot = viewerSlot;
              setViewerSlot(undefined);
              openFreeplay(slot);
            }}
          >
            <Expand aria-hidden="true" className="size-4" />
            Take the controls
          </button>
        }
      />

      <FreeplayDialog open={freeplayOpen} onOpenChange={setFreeplayOpen} subjectLabel={freeplaySubject} />

      <AlertDialog.Root open={cancelOpen} onOpenChange={setCancelOpen}>
        <AlertDialog.Portal>
          <AlertDialog.Overlay className="dialog-overlay" />
          <AlertDialog.Content className="alert-dialog">
            <AlertDialog.Title>Cancel this run?</AlertDialog.Title>
            <AlertDialog.Description className="text-pretty">
              Cancellation leaves explicit terminal records and retains already allocated work and its cost.
            </AlertDialog.Description>
            <div className="alert-actions">
              <AlertDialog.Cancel className="button button-secondary">Keep running</AlertDialog.Cancel>
              <AlertDialog.Action className="button button-danger" onClick={() => void cancelRun()}>
                Cancel run
              </AlertDialog.Action>
            </div>
          </AlertDialog.Content>
        </AlertDialog.Portal>
      </AlertDialog.Root>
    </div>
  );
}
