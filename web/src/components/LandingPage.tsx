import { ExternalLink, Film, Images, Layers, ShieldAlert, Terminal, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";
import { useReducedMotion } from "../hooks/usePolling";
import {
  artifactUrl,
  artifactUrls,
  readProvenance,
  type Episode,
  type Gate,
  type JsonRecord,
  type Provenance,
  type Run,
} from "../lib/api";
import { formatCount, pickNumber, pickString } from "../lib/format";
import { BENCHMARK_TASKS, TASK_REGISTRY_ID } from "../lib/tasks";
import { cn } from "../lib/utils";
import { PROVENANCE_DESCRIPTIONS, PROVENANCE_LABELS } from "../lib/wall";
import { GalleryModal } from "./GalleryModal";
import { gateBlockers } from "./GatePanel";
import { HERO_TRACK, selectHeroEpisode } from "./HeroRollout";
import { FramePlayer, isVideoSource } from "./MediaFrame";
import { EmptyState } from "./Primitives";
import { TabSelector, type TabItem } from "./TabSelector";

export type LandingTab = "console" | "gallery" | "gates";

const TABS: TabItem<LandingTab>[] = [
  { id: "console", label: "Console" },
  { id: "gallery", label: "Gallery" },
  { id: "gates", label: "Gates" },
];

/**
 * What the landing page can honestly put behind its text.
 *
 * Order of preference is the 480p presentation rollout, then a single still from
 * it, then a CSS gradient. The gradient is the floor deliberately: it cannot 404,
 * so there is no state in which this page shows a broken media element. A still
 * always wins when the viewer has asked for reduced motion.
 */
export type Backdrop =
  | { kind: "video"; src: string; label: string }
  | { kind: "frames"; frames: string[]; label: string }
  | { kind: "still"; src: string; label: string }
  | { kind: "gradient"; label: string };

export function resolveBackdrop(
  episodes: Episode[],
  options: { reducedMotion: boolean },
): Backdrop {
  const hero = selectHeroEpisode(episodes);
  if (!hero) {
    return {
      kind: "gradient",
      label: `no episode declares presentation_track "${HERO_TRACK}" · flat background`,
    };
  }
  const frames = artifactUrls(hero.frame_urls);
  const video = artifactUrl(pickString(hero.video_url));
  // A `video_url` that does not name a video file is used as a still — which is
  // what the hero panel already does — rather than handed to a <video> element
  // that would then fail to load.
  const still =
    frames[0] ??
    artifactUrl(pickString(hero.frame_url)) ??
    (video !== undefined && !isVideoSource(video) ? video : undefined);

  if (options.reducedMotion) {
    return still
      ? { kind: "still", src: still, label: "480p hero rollout · still frame, reduced motion" }
      : {
          kind: "gradient",
          label: "reduced motion requested and the hero rollout has no still frame · flat background",
        };
  }
  if (video && isVideoSource(video)) {
    return { kind: "video", src: video, label: "480p hero rollout · persisted video" };
  }
  if (frames.length > 1) {
    return {
      kind: "frames",
      frames,
      label: `480p hero rollout · ${frames.length} persisted frames`,
    };
  }
  if (still) {
    return { kind: "still", src: still, label: "480p hero rollout · one persisted frame" };
  }
  return {
    kind: "gradient",
    label: "the hero episode has no persisted media · flat background",
  };
}

export type RailCard = {
  key: string;
  policy: string;
  task: string;
  runId?: string;
  episodeId?: string;
  status?: string;
  frames: string[];
  provenance?: Provenance;
  certifiedFrameCount?: number;
};

/**
 * Cards for the example rail, built only from episodes that actually persisted
 * frames. An episode with no media is not padded out with a placeholder image;
 * it is simply not a card, and the rail says how many runs it had to work with.
 */
export function railCards(episodes: Episode[], limit = 3): RailCard[] {
  const cards: RailCard[] = [];
  for (const episode of episodes) {
    const frames = artifactUrls(episode.frame_urls);
    const single = artifactUrl(pickString(episode.frame_url, episode.video_url));
    const resolved = frames.length > 0 ? frames : single ? [single] : [];
    if (resolved.length === 0) continue;
    cards.push({
      key: pickString(episode.episode_id, episode.id) ?? `card-${cards.length}`,
      policy: pickString(episode.policy, episode.policy_variant) ?? "policy not reported",
      task: pickString(episode.task) ?? "task not reported",
      runId: pickString(episode.run_id),
      episodeId: pickString(episode.episode_id, episode.id),
      status: pickString(episode.status),
      frames: resolved,
      provenance: readProvenance(episode.provenance),
      certifiedFrameCount: pickNumber(episode.certified_frame_count),
    });
    if (cards.length >= limit) break;
  }
  return cards;
}

function ProvenanceBadge({ provenance }: { provenance?: Provenance }) {
  if (!provenance) {
    return (
      <span
        className="provenance-badge provenance-unreported"
        title="No record reported a provenance label, so this clip's source is unknown."
      >
        no provenance
      </span>
    );
  }
  return (
    <span
      className={cn("provenance-badge", `provenance-${provenance}`)}
      title={PROVENANCE_DESCRIPTIONS[provenance]}
    >
      {PROVENANCE_LABELS[provenance]}
    </span>
  );
}

function frameLine(card: RailCard): string {
  if (card.certifiedFrameCount !== undefined) {
    return `${formatCount(card.certifiedFrameCount)} certified frames`;
  }
  return `${formatCount(card.frames.length)} frames · certified count not reported`;
}

export function LandingPage({
  runs,
  episodes,
  gates,
  protocol,
  scopedTask,
  onScopeTask,
  onEnterConsole,
  onOpenFreeplay,
}: {
  runs: Run[];
  episodes: Episode[];
  gates: Gate[];
  protocol?: JsonRecord;
  scopedTask?: string;
  onScopeTask: (taskId: string | undefined) => void;
  onEnterConsole: (anchor: string) => void;
  onOpenFreeplay: () => void;
}) {
  const [tab, setTab] = useState<LandingTab>("console");
  const [openCard, setOpenCard] = useState<RailCard>();
  const [mediaFailed, setMediaFailed] = useState(false);
  const reducedMotion = useReducedMotion();

  const resolved = useMemo(
    () => resolveBackdrop(episodes, { reducedMotion }),
    [episodes, reducedMotion],
  );
  // A media element that failed to load is replaced by the gradient, not left
  // on the page as a broken frame with an X in it.
  const backdrop: Backdrop = mediaFailed
    ? { kind: "gradient", label: `${resolved.label} · media did not load, flat background` }
    : resolved;

  const cards = useMemo(() => railCards(episodes), [episodes]);
  const blockers = useMemo(() => gateBlockers(gates), [gates]);
  const mode = pickString(protocol?.mode);
  const scoped = BENCHMARK_TASKS.find((task) => task.id === scopedTask);

  return (
    <section className="landing" aria-labelledby="landing-wordmark">
      <div className="landing-backdrop" aria-hidden="true">
        {backdrop.kind === "video" && (
          <video
            className="landing-backdrop-media"
            src={backdrop.src}
            autoPlay
            loop
            muted
            playsInline
            tabIndex={-1}
            onError={() => setMediaFailed(true)}
          />
        )}
        {backdrop.kind === "still" && (
          <img
            className="landing-backdrop-media"
            src={backdrop.src}
            alt=""
            onError={() => setMediaFailed(true)}
          />
        )}
        {backdrop.kind === "frames" && (
          <FramePlayer frames={backdrop.frames} alt="" className="landing-backdrop-media" />
        )}
        {backdrop.kind === "gradient" && <div className="landing-backdrop-gradient" />}
      </div>
      <div className="landing-wash" aria-hidden="true" />

      <div className="landing-nav">
        <TabSelector
          tabs={TABS}
          value={tab}
          onChange={setTab}
          label="Landing sections"
          idPrefix="landing"
          centered
          display
        />
      </div>

      <div className="landing-body">
        <div className="landing-main">
          <h1 id="landing-wordmark" className="landing-wordmark">
            PLUMB
          </h1>
          <p className="landing-line text-pretty">
            Point us at a policy endpoint and get a ranked report in twenty minutes for a few dollars — with
            four numbers that say how much to trust it.
          </p>
          <p className="landing-qualifier">
            <TriangleAlert aria-hidden="true" className="size-4" />
            <span>
              That is the product claim, not a receipt. The twenty minutes and the dollars are targets this
              console has not reconciled, and no result here is qualified yet — the gate ledger says what is
              missing.
            </span>
          </p>

          {tab === "console" && (
            <div
              id="landing-panel-console"
              role="tabpanel"
              aria-labelledby="landing-tab-console"
              tabIndex={0}
              className="landing-section"
            >
              <h2>Pick a task, then open the console</h2>
              <p className="text-pretty">
                Five fixed strings, exactly as registry <code>{TASK_REGISTRY_ID}</code> freezes them.
                Selecting one scopes what the rollout viewport shows. It sends nothing: a world model here
                only ever receives a registry task string, and this page has no text field for that reason.
              </p>
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
              <p className="chip-scope">
                {scoped ? (
                  <>
                    <span>Viewport scoped to</span> <b>{scoped.id}</b>
                    <span>· {scoped.maxSteps} max steps</span>
                    <button type="button" className="button button-quiet" onClick={() => onScopeTask(undefined)}>
                      Clear scope
                    </button>
                  </>
                ) : (
                  <span>No task selected, so the viewport shows every identity it has been sent.</span>
                )}
              </p>
              <div className="landing-actions">
                <button
                  type="button"
                  className="button button-primary"
                  onClick={() => onEnterConsole("console")}
                >
                  <Terminal aria-hidden="true" className="size-4" />
                  Open the console
                </button>
                <button type="button" className="button button-secondary" onClick={onOpenFreeplay}>
                  <Film aria-hidden="true" className="size-4" />
                  Drive the world model
                </button>
              </div>
            </div>
          )}

          {tab === "gallery" && (
            <div
              id="landing-panel-gallery"
              role="tabpanel"
              aria-labelledby="landing-tab-gallery"
              tabIndex={0}
              className="landing-section"
            >
              <h2>Rollouts this console has persisted</h2>
              <p className="text-pretty">
                Every card is one persisted episode with its own provenance label. Cached, replayed and
                qualitative clips keep their badge, so a reused clip never passes for a live one.
              </p>
              {cards.length === 0 ? (
                <EmptyState icon={<Images aria-hidden="true" className="size-5" />}>
                  {runs.length === 0
                    ? "No run has been persisted, so there is nothing to show here. This rail is built from persisted episodes and never stands in illustrative footage for them."
                    : `${formatCount(runs.length)} persisted run${runs.length === 1 ? "" : "s"}, but no episode of the selected run has persisted frames yet. Nothing is shown in their place.`}
                </EmptyState>
              ) : (
                <ul className="rail">
                  {cards.map((card) => (
                    <li key={card.key}>
                      <button type="button" className="rail-card" onClick={() => setOpenCard(card)}>
                        <span className="rail-frame">
                          <FramePlayer
                            frames={card.frames}
                            alt={`Persisted frames for ${card.policy} on ${card.task}`}
                            className="rail-media"
                          />
                          <span className="rail-badge">
                            <ProvenanceBadge provenance={card.provenance} />
                          </span>
                        </span>
                        <span className="rail-meta">
                          <strong>{card.policy}</strong>
                          <span>{card.task}</span>
                          <span className="rail-id">{card.runId ?? "run not reported"}</span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              <div className="landing-actions">
                <button
                  type="button"
                  className="button button-primary"
                  onClick={() => onEnterConsole("rollouts")}
                >
                  <Layers aria-hidden="true" className="size-4" />
                  Open the rollout viewport
                </button>
              </div>
            </div>
          )}

          {tab === "gates" && (
            <div
              id="landing-panel-gates"
              role="tabpanel"
              aria-labelledby="landing-tab-gates"
              tabIndex={0}
              className="landing-section"
            >
              <h2>What is not qualified yet</h2>
              <p className="text-pretty">
                Real backends cannot be dispatched from this console until the evidence gates qualify the
                chosen protocol. Until then every number on the console is labelled{" "}
                {mode ? <code>{mode}</code> : "with the mode the protocol reported"} and unqualified.
              </p>
              <ul className="chip-row">
                {blockers.slice(0, 4).map((reason) => (
                  <li key={reason} className="chip">
                    <ShieldAlert aria-hidden="true" className="size-3.5" />
                    {reason}
                  </li>
                ))}
              </ul>
              <div className="landing-actions">
                <button
                  type="button"
                  className="button button-primary"
                  onClick={() => onEnterConsole("gates")}
                >
                  <ShieldAlert aria-hidden="true" className="size-4" />
                  Open the gate ledger
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      <footer className="landing-footer">
        <span>PLUMB measurement console</span>
        <span className="landing-backdrop-note">
          <Film aria-hidden="true" className="size-3.5" />
          Background: {backdrop.label}
        </span>
        <a href="/THIRD-PARTY-NOTICES.md" target="_blank" rel="noreferrer">
          Third-party notices <ExternalLink aria-hidden="true" className="size-3" />
        </a>
      </footer>

      <GalleryModal
        open={openCard !== undefined}
        onClose={() => setOpenCard(undefined)}
        title={openCard ? `${openCard.policy} · ${openCard.task}` : ""}
        subtitle={openCard?.episodeId ?? "episode id not reported"}
        frames={openCard?.frames ?? []}
        badge={<ProvenanceBadge provenance={openCard?.provenance} />}
        meta={
          openCard
            ? [
                { label: "Run", value: openCard.runId ?? "not reported" },
                { label: "Status", value: openCard.status ?? "not reported" },
                { label: "Frames", value: frameLine(openCard) },
              ]
            : []
        }
        footer={
          <button
            type="button"
            className="button button-secondary"
            onClick={() => {
              setOpenCard(undefined);
              onEnterConsole("rollouts");
            }}
          >
            <Layers aria-hidden="true" className="size-4" />
            Open the rollout viewport
          </button>
        }
      />
    </section>
  );
}
