/**
 * The first screen: navigation, the claim, the two ways in.
 *
 * The heading names the differentiator rather than the category, because the
 * category already has three entrants. What none of them published is the
 * second line.
 */
import { Glyph } from "../components/Terminal";
import { Link } from "react-router-dom";
import { CONSOLE_PATH, LIVE_PATH, NAV_LINKS, PRODUCT } from "./content";
import { VIDEO } from "./media";
import { DreamWall, Glass } from "./Pieces";
import { HeroVideo } from "./Video";

function Nav() {
  return (
    <nav className="relative z-20 px-6 py-6" aria-label="Landing sections">
      <Glass className="mx-auto flex max-w-5xl items-center justify-between rounded-full px-6 py-3">
        <div className="flex items-center">
          <Link to="/" className="flex items-center gap-2.5 no-underline" aria-label={`${PRODUCT.name} home`}>
            <span
              aria-hidden="true"
              className="grid size-6 place-items-center rounded-sm bg-accent font-mono text-sm font-semibold text-black"
            >
              N
            </span>
            <span className="text-lg font-semibold text-white">{PRODUCT.name}</span>
          </Link>
          <ul className="m-0 ml-8 hidden list-none gap-8 p-0 md:flex">
            {NAV_LINKS.map((link) => (
              <li key={link.href}>
                <a className="text-sm font-medium text-white/80 no-underline hover:text-white" href={link.href}>
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
        <div className="flex items-center gap-4">
          <a
            className="hidden text-sm font-medium text-white/80 no-underline hover:text-white sm:inline"
            href="/api/protocol"
            target="_blank"
            rel="noreferrer"
          >
            Protocol record
          </a>
          <Glass
            as="div"
            className="rounded-full"
          >
            <Link
              to={CONSOLE_PATH}
              className="block px-6 py-2 text-sm font-medium text-white no-underline"
            >
              Open the console
            </Link>
          </Glass>
        </div>
      </Glass>
    </nav>
  );
}

export function Hero() {
  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden">
      {/* Three layers, bottom to top: the CSS wall, the video, the washes.
          The wall is not a placeholder that gets replaced — it is what remains
          when the CDN is unreachable, and the video fades in over it only once
          the browser says it can play. The washes hold the headline at full
          contrast over whichever of the two is showing. */}
      <DreamWall className="absolute inset-0 h-full w-full" />
      <HeroVideo
        src={VIDEO.hero}
        className="absolute inset-0 h-full w-full object-cover object-bottom"
      />
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_rgba(0,0,0,0.08)_0%,_rgba(0,0,0,0.2)_60%,_rgba(0,0,0,0.5)_100%)]"
      />
      <div aria-hidden="true" className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-black via-black/60 to-transparent" />

      <Nav />

      <div className="relative z-10 flex flex-1 flex-col items-center justify-center gap-8 px-6 py-12 text-center">
        <Glass className="flex items-center gap-2.5 rounded-full px-4 py-1.5">
          <span aria-hidden="true" className="size-1.5 rounded-full bg-caution" />
          <span className="font-mono text-xs text-white/80">
            Unqualified · synthetic mode · every number carries its status
          </span>
        </Glass>

        <h1 className="font-display-serif m-0 max-w-5xl text-5xl leading-[1.05] tracking-tight text-white sm:text-6xl md:text-7xl lg:text-8xl">
          Evaluate a robot policy <em className="italic text-white/60">without a robot.</em>
        </h1>

        <p className="m-0 max-w-2xl text-base leading-relaxed text-white/80 md:text-lg">
          {PRODUCT.promise}
        </p>

        <div className="flex flex-col items-center gap-3 sm:flex-row">
          <Glass className="rounded-full liquid-glass-accent">
            <Link
              to={CONSOLE_PATH}
              className="flex items-center gap-2 px-8 py-3 text-sm font-medium text-white no-underline"
            >
              Open the console
              <Glyph name="arrowRight" />
            </Link>
          </Glass>
          <Glass className="rounded-full">
            <Link
              to={LIVE_PATH}
              className="flex items-center gap-2 px-8 py-3 text-sm font-medium text-white no-underline"
            >
              <Glyph name="play" />
              Watch a live run
            </Link>
          </Glass>
        </div>

        {/* The one sentence the whole project turns on, placed where a
            newsletter field would be on a page selling something else. */}
        <p className="m-0 max-w-xl text-sm leading-relaxed text-white/70">
          Three groups have already automated this with world models. All three reported accuracy.
          <span className="text-white"> None reported precision.</span>
        </p>
      </div>

    </div>
  );
}
