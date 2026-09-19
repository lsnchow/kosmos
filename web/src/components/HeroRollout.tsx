import { Glyph } from "./Terminal";
import { artifactUrl, artifactUrls, readProvenance, type Episode } from "../lib/api";
import { formatCount, pickNumber, pickString } from "../lib/format";
import { FramePlayer, MediaFrame } from "./MediaFrame";
import { EmptyState, Note, Panel, SourceChip, StatusPill } from "./Primitives";

/** The one explicit flag that marks the 480p presentation rollout. */
export const HERO_TRACK = "hero_480p";

/**
 * Select the hero episode from an explicit flag only.
 *
 * The old selector read `presentation` or `track` — keys no backend writes — so
 * it always fell through to `episodes[0]` and relabelled tile #01 as "480p".
 * Returning `undefined` is the correct answer when no episode declares itself
 * the hero.
 */
export function selectHeroEpisode(episodes: Episode[]): Episode | undefined {
  return episodes.find((episode) => pickString(episode.presentation_track) === HERO_TRACK);
}

export function HeroRollout({ episodes }: { episodes: Episode[] }) {
  const hero = selectHeroEpisode(episodes);
  const frames = hero ? artifactUrls(hero.frame_urls) : [];
  const single = hero ? artifactUrl(pickString(hero.frame_url, hero.video_url)) : undefined;
  const provenance = hero ? readProvenance(hero.provenance) : undefined;
  const resolution = hero ? pickString(hero.resolution) : undefined;

  return (
    <Panel
      title="480p presentation rollout"
      className="hero-panel"
      action={<SourceChip>{resolution ?? "resolution not reported"}</SourceChip>}
    >
      {!hero ? (
        <EmptyState icon={<Glyph name="monitor" />}>
          No episode declares <code>presentation_track = &quot;{HERO_TRACK}&quot;</code>, so there is no hero
          rollout to show. A 256p scoring episode is not promoted here and relabelled 480p.
        </EmptyState>
      ) : (
        <>
          <div className="hero-frame">
            {frames.length > 0 ? (
              <FramePlayer
                frames={frames}
                alt="480p presentation rollout, accumulated generated frames"
                className="hero-media"
              />
            ) : (
              <MediaFrame
                src={single}
                alt="480p presentation rollout"
                className="hero-media"
                emptyReason="The hero episode has no persisted frames yet"
              />
            )}
          </div>
          <div className="hero-meta">
            <StatusPill status={hero.status ?? "unknown"}>{String(hero.status ?? "unknown")}</StatusPill>
            {provenance && <StatusPill status={provenance === "live" ? "live" : "unknown"}>{provenance}</StatusPill>}
            <span>{pickString(hero.policy) ?? "policy not reported"}</span>
            <span>{pickString(hero.task) ?? "task not reported"}</span>
            <span className="tabular-nums">
              {hero.certified_frame_count === undefined
                ? `${formatCount(frames.length || undefined)} frames · certified count not reported`
                : `${formatCount(pickNumber(hero.certified_frame_count))} certified frames`}
            </span>
          </div>
          <p className="hero-id truncate" title={pickString(hero.episode_id, hero.id) ?? "episode id not reported"}>
            {pickString(hero.episode_id, hero.id) ?? "episode id not reported"}
          </p>
        </>
      )}
      <Note summary="Why 480p is a separate track">
        The presentation track is a separate operating point from the 256p primary scoring protocol. If a 480p
        setting ever enters a scored comparison it is qualified as its own operating point, with its own sweep
        and its own confirmation panel.
      </Note>
    </Panel>
  );
}
