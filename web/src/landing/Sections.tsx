/**
 * The five scrolling sections under the hero.
 *
 * Order is the argument: the cost of the status quo, the size of the error the
 * status quo's cheap substitute makes, how we do it instead, what we measure
 * about ourselves, and what we still cannot claim. The limits sit above the
 * final call to action rather than in a footnote, because a product whose pitch
 * is "we put error bars on it" cannot bury its own.
 */
import { Glyph } from "../components/Terminal";
import { Link } from "react-router-dom";
import { cn } from "../lib/utils";
import {
  BURST_TARGET,
  CALLED_SHOT,
  CHAIN_STEPS,
  CONSOLE_PATH,
  FOUR_NUMBERS,
  LIMITS,
  LIVE_PATH,
  METHOD_CHIPS,
  OTHER_GAPS,
  PIPELINE_NOTES,
  PROBLEM_STATS,
  PRODUCT,
  RESULTS_PATH,
  SERVICES,
  STACK,
  TASKS,
  VALIDATION_NOTE,
} from "./content";
import { VIDEO } from "./media";
import { Figure, Glass, Reveal, SectionLabel } from "./Pieces";
import { LoopVideo } from "./Video";

/**
 * A video panel.
 *
 * The `dream-tile` underneath is not decoration on top of decoration: the video
 * fades in over it once it can play, so an unreachable CDN leaves a drifting
 * latent field rather than an empty well.
 */
function VideoPanel({ src, className }: { src: string; className?: string }) {
  return (
    <div className={cn("dream-tile", className)}>
      <LoopVideo src={src} className="absolute inset-0" />
    </div>
  );
}

const percent = (rate: number) => `${Math.round(rate * 100)}%`;

/* ------------------------------------------------------------------------ */

export function ProblemSection() {
  return (
    <section
      id="problem"
      className="relative overflow-hidden bg-black px-6 pb-16 pt-32 md:pb-24 md:pt-44"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(255,255,255,0.07)_0%,_transparent_70%)]"
      />
      <div className="relative mx-auto max-w-6xl">
        <Reveal from={{ y: 20 }} duration={0.6}>
          <SectionLabel>The problem</SectionLabel>
        </Reveal>

        <Reveal from={{ y: 40 }} delay={0.1} duration={0.8}>
          <h2 className="font-display-serif m-0 mt-6 max-w-4xl text-4xl leading-[1.1] tracking-tight text-fg-strong md:text-6xl lg:text-7xl">
            Comparing two policies costs a person{" "}
            <em className="not-italic text-fg-dim underline decoration-1 underline-offset-[0.14em] decoration-accent/60">a week standing next to an arm.</em>
          </h2>
        </Reveal>

        <ul className="m-0 mt-14 grid list-none grid-cols-1 gap-px overflow-hidden rounded-none bg-white/10 p-0 sm:grid-cols-2 lg:grid-cols-4">
          {PROBLEM_STATS.map((stat, index) => (
            <li key={stat.unit} className="bg-black">
              <Reveal from={{ y: 40 }} delay={index * 0.08} duration={0.7} className="h-full">
                <div className="h-full p-6 md:p-8">
                  <Figure value={stat.value} unit={stat.unit} note={stat.note} source={stat.source} />
                </div>
              </Reveal>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------------ */

/** A horizontal bar, sized by its rate. */
function RateBar({
  label,
  sublabel,
  rate,
  tone,
}: {
  label: string;
  sublabel: string;
  rate: number | null;
  tone: "real" | "sim" | "ours";
}) {
  const fill =
    tone === "real" ? "bg-accent" : tone === "sim" ? "bg-white/25" : "bg-white/10";
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-sm text-fg">{label}</span>
        <span className="font-mono text-sm tabular-nums text-fg-strong">
          {rate === null ? "—" : percent(rate)}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-none bg-white/10">
        <div
          className={`h-full rounded-none ${fill}`}
          style={{ width: rate === null ? "0%" : `${rate * 100}%` }}
        />
      </div>
      <span className="font-mono text-[0.6875rem] leading-relaxed text-fg-dim">{sublabel}</span>
    </div>
  );
}

export function CalledShotSection() {
  return (
    <section id="called-shot" className="relative overflow-hidden bg-black px-6 py-24 md:py-32">
      <div className="relative mx-auto max-w-6xl">
        <Reveal from={{ y: 20 }} duration={0.6}>
          <SectionLabel>The gap</SectionLabel>
        </Reveal>

        <Reveal from={{ y: 40 }} delay={0.1} duration={0.8}>
          <h2 className="font-display-serif m-0 mt-6 max-w-4xl text-4xl leading-[1.1] tracking-tight text-fg-strong md:text-6xl">
            The cheap substitute is{" "}
            <em className="not-italic text-fg-dim underline decoration-1 underline-offset-[0.14em] decoration-accent/60">eighty-eight points wrong.</em>
          </h2>
        </Reveal>

        <Reveal from={{ y: 60 }} delay={0.12} duration={0.9} className="mt-12">
          <div className="relative aspect-video overflow-hidden rounded-none">
            <VideoPanel src={VIDEO.calledShot} className="absolute inset-0 h-full w-full" />
            <div
              aria-hidden="true"
              className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent"
            />
            <div className="absolute inset-x-0 bottom-0 flex flex-col gap-6 p-6 md:flex-row md:items-end md:justify-between md:p-10">
              <Glass className="max-w-md rounded-none p-6 md:p-8">
                <p className="m-0 text-xs uppercase tracking-[0.18em] text-fg-dim">Our approach</p>
                <p className="m-0 mt-3 text-sm leading-relaxed text-fg-strong md:text-base">
                  The validity gate is Python, not a model call. We took the work away from the
                  model.
                </p>
              </Glass>
              <Glass className="self-start rounded-none md:self-auto">
                <Link
                  to={RESULTS_PATH}
                  className="flex items-center gap-2 px-8 py-3 text-sm font-medium text-fg-strong no-underline"
                >
                  See the full table
                  <Glyph name="arrowUpRight" />
                </Link>
              </Glass>
            </div>
          </div>
        </Reveal>

        <Reveal from={{ y: 40 }} delay={0.15} duration={0.8} className="mt-8">
          <Glass className="rounded-none p-6 md:p-10">
            <div className="flex flex-wrap items-baseline justify-between gap-4">
              <p className="m-0 font-mono text-sm text-fg">
                {CALLED_SHOT.policy} · “{CALLED_SHOT.task}”
              </p>
              <p className="m-0 font-mono text-xs uppercase tracking-[0.18em] text-fg-dim">
                50 trials per cell
              </p>
            </div>

            <div className="mt-8 grid grid-cols-1 gap-8 md:grid-cols-3">
              <RateBar
                label="Real arm, human-scored"
                sublabel={`${CALLED_SHOT.human.successes} of ${CALLED_SHOT.human.trials} · ${CALLED_SHOT.source}`}
                rate={CALLED_SHOT.human.rate}
                tone="real"
              />
              <RateBar
                label="Physics simulator"
                sublabel={`${CALLED_SHOT.simulator.successes} of ${CALLED_SHOT.simulator.trials} · SIMPLER, same paper`}
                rate={CALLED_SHOT.simulator.rate}
                tone="sim"
              />
              <RateBar
                label={`${PRODUCT.name}`}
                sublabel={`${CALLED_SHOT.ours} — the console states its estimate before the reveal`}
                rate={null}
                tone="ours"
              />
            </div>

            <p className="m-0 mt-8 text-sm leading-relaxed text-fg-muted md:text-base">
              Not an outlier: two more cells in the same table disagree by{" "}
              {Math.round((OTHER_GAPS[0].human - OTHER_GAPS[0].simulator) * 100)} and{" "}
              {Math.round((OTHER_GAPS[1].simulator - OTHER_GAPS[1].human) * 100)} points, in
              opposite directions.
            </p>
          </Glass>
        </Reveal>

      </div>
    </section>
  );
}

/* ------------------------------------------------------------------------ */

export function PipelineSection() {
  return (
    <section id="pipeline" className="relative overflow-hidden bg-black px-6 py-24 md:py-36">
      <div className="relative mx-auto max-w-6xl">
        <Reveal from={{ y: 40 }} duration={0.8}>
          <h2 className="font-display-serif m-0 text-4xl leading-[1.05] tracking-tight text-fg-strong md:text-6xl lg:text-7xl">
            Four steps, <em className="not-italic text-fg-dim underline decoration-1 underline-offset-[0.14em] decoration-accent/60">four</em> hardware profiles.
          </h2>
        </Reveal>

        <Reveal from={{ y: 40 }} delay={0.1} duration={0.8} className="mt-12">
          <Glass className="rounded-none p-6 md:p-8">
              <ol className="relative m-0 flex list-none flex-col gap-6 p-0">
                {/* The rail the packet travels. Decorative; the list is the
                    content and reads in order without it. */}
                <span
                  aria-hidden="true"
                  className="absolute bottom-6 left-[0.6875rem] top-6 w-px overflow-hidden bg-white/10"
                >
                  <span className="chain-flow absolute inset-x-0 h-1/4" />
                </span>
                {CHAIN_STEPS.map((step, index) => (
                  <li key={step.id} className="relative flex gap-5 pl-0">
                    <span
                      aria-hidden="true"
                      className="relative z-10 mt-1 grid size-6 shrink-0 place-items-center rounded-none bg-black font-mono text-[0.6875rem] text-fg ring-1 ring-white/20"
                    >
                      {index + 1}
                    </span>
                    <div>
                      <p className="m-0 text-base font-medium text-fg-strong">{step.name}</p>
                      <p className="m-0 mt-1 font-mono text-[0.6875rem] uppercase tracking-[0.14em] text-accent-bright">
                        {step.hardware}
                      </p>
                      <p className="m-0 mt-2 text-sm leading-relaxed text-fg-muted">{step.detail}</p>
                    </div>
                  </li>
                ))}
              </ol>
          </Glass>
        </Reveal>

        <div className="mt-8 grid grid-cols-1 gap-10 md:grid-cols-2 md:gap-12">
          <Reveal from={{ x: -40 }} duration={0.8}>
            <VideoPanel src={VIDEO.pipeline} className="aspect-[4/3] w-full rounded-none" />
          </Reveal>

          <Reveal from={{ x: 40 }} duration={0.8} className="flex flex-col justify-center">
            <div className="flex flex-col gap-10">
              {PIPELINE_NOTES.map((note, index) => (
                <div key={note.label} className="flex flex-col gap-4">
                  {index > 0 && <span aria-hidden="true" className="h-px w-full bg-white/10" />}
                  <p className="m-0 text-xs uppercase tracking-[0.18em] text-fg-dim">
                    {note.label}
                  </p>
                  <p className="m-0 text-base leading-relaxed text-fg-muted md:text-lg">
                    {note.body}
                  </p>
                </div>
              ))}

              <div className="flex flex-col gap-4">
                <span aria-hidden="true" className="h-px w-full bg-white/10" />
                <p className="m-0 text-xs uppercase tracking-[0.18em] text-fg-dim">
                  Five fixed tasks
                </p>
                <ul className="m-0 flex list-none flex-wrap gap-2 p-0">
                  {TASKS.map((task) => (
                    <li key={task.prompt}>
                      <span className="inline-flex items-center gap-2 rounded-none border border-white/15 px-3.5 py-1.5 font-mono text-[0.6875rem] text-fg">
                        {task.prompt}
                        <span className="text-fg-dim">{task.horizon} steps</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------------ */

export function NumbersSection() {
  return (
    <section id="numbers" className="relative overflow-hidden bg-black px-6 py-24 md:py-36">
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_rgba(255,255,255,0.05)_0%,_transparent_60%)]"
      />
      <div className="relative mx-auto max-w-6xl">
        <Reveal from={{ y: 30 }} duration={0.7}>
          <div className="flex flex-wrap items-end justify-between gap-4">
            <h2 className="font-display-serif m-0 text-3xl tracking-tight text-fg-strong md:text-5xl">
              The four numbers we publish about ourselves
            </h2>
            <p className="m-0 hidden text-sm text-fg-dim md:block">The spec sheet</p>
          </div>
        </Reveal>

        <Reveal from={{ y: 20 }} delay={0.08} duration={0.6}>
          <p className="m-0 mt-6 max-w-2xl text-base leading-relaxed text-fg-muted">
            An evaluator without error bars is not an instrument. Each of these is blank until a
            qualified run fills it in.
          </p>
        </Reveal>

        <ul className="m-0 mt-14 grid list-none grid-cols-1 gap-6 p-0 md:grid-cols-2 md:gap-8">
          {FOUR_NUMBERS.map((item, index) => (
            <li key={item.title}>
              <Reveal from={{ y: 50 }} delay={index * 0.15} duration={0.8} className="h-full">
                <Glass className="group flex h-full flex-col rounded-none p-6 md:p-8">
                  <div className="flex items-start justify-between gap-4">
                    <span className="font-mono text-xs uppercase tracking-[0.18em] text-fg-dim">
                      {item.tag}
                    </span>
                    <span className="grid size-8 shrink-0 place-items-center rounded-none bg-white/5 text-fg-muted transition-colors group-hover:bg-white/10 group-hover:text-fg-strong">
                      <Glyph name="arrowUpRight" />
                    </span>
                  </div>
                  <h3 className="m-0 mt-4 text-xl tracking-tight text-fg-strong md:text-2xl">
                    {item.title}
                  </h3>
                  <p className="m-0 mt-3 flex-1 text-sm leading-relaxed text-fg-muted">
                    {item.description}
                  </p>
                  <div className="mt-6 border-t border-white/10 pt-5">
                    <dl className="m-0 flex items-baseline justify-between gap-4">
                      <dt className="m-0 text-xs uppercase tracking-[0.14em] text-fg-dim">
                        Value
                      </dt>
                      <dd className="m-0 flex items-center gap-2 font-mono text-sm text-caution">
                        <Glyph name="minus" />
                        {item.value}
                      </dd>
                    </dl>
                    <p className="m-0 mt-2 text-[0.6875rem] leading-relaxed text-fg-dim">
                      {item.priorArt}
                    </p>
                  </div>
                </Glass>
              </Reveal>
            </li>
          ))}
        </ul>

        <Reveal from={{ y: 30 }} duration={0.7} className="mt-10">
          <div className="flex flex-col gap-6 border-t border-white/10 pt-8 md:flex-row md:items-start md:justify-between">
            <p className="m-0 max-w-xl text-sm leading-relaxed text-fg-muted">
              {VALIDATION_NOTE.body}
            </p>
            <ul className="m-0 flex list-none flex-wrap gap-2 p-0 md:justify-end">
              {METHOD_CHIPS.map((chip) => (
                <li key={chip}>
                  <span className="inline-flex items-center gap-2 rounded-none border border-white/15 px-3.5 py-1.5 text-[0.6875rem] text-fg">
                    <Glyph name="check" className="text-accent" />
                    {chip}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------------ */

export function ServicesSection() {
  return (
    <section id="services" className="relative overflow-hidden bg-black px-6 py-24 md:py-32">
      <div className="relative mx-auto max-w-6xl">
        <Reveal from={{ y: 30 }} duration={0.7}>
          <div className="flex flex-wrap items-end justify-between gap-4">
            <h2 className="font-display-serif m-0 text-3xl tracking-tight text-fg-strong md:text-5xl">
              What we do
            </h2>
            <p className="m-0 hidden text-sm text-fg-dim md:block">Two halves, in order</p>
          </div>
        </Reveal>

        <ul className="m-0 mt-12 grid list-none grid-cols-1 gap-6 p-0 md:grid-cols-2 md:gap-8">
          {SERVICES.map((service, index) => (
            <li key={service.title}>
              <Reveal from={{ y: 50 }} delay={index * 0.15} duration={0.8} className="h-full">
                <Glass className="group flex h-full flex-col overflow-hidden rounded-none">
                  <div className="relative aspect-video overflow-hidden">
                    <VideoPanel
                      src={VIDEO[service.media]}
                      className="absolute inset-0 h-full w-full transition-transform duration-700 group-hover:scale-105"
                    />
                    <div
                      aria-hidden="true"
                      className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent"
                    />
                  </div>
                  <div className="flex flex-1 flex-col p-6 md:p-8">
                    <div className="flex items-start justify-between gap-4">
                      <span className="font-mono text-xs uppercase tracking-[0.18em] text-fg-dim">
                        {service.tag}
                      </span>
                      <span className="grid size-8 shrink-0 place-items-center rounded-none bg-white/5 text-fg-muted transition-colors group-hover:bg-white/10 group-hover:text-fg-strong">
                        <Glyph name="arrowUpRight" />
                      </span>
                    </div>
                    <h3 className="m-0 mt-4 text-xl tracking-tight text-fg-strong md:text-2xl">
                      {service.title}
                    </h3>
                    <p className="m-0 mt-3 text-sm leading-relaxed text-fg-muted">{service.body}</p>
                  </div>
                </Glass>
              </Reveal>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------------ */

export function StackSection() {
  return (
    <section id="stack" className="relative overflow-hidden bg-black px-6 py-24 md:py-32">
      <div className="relative mx-auto max-w-6xl">
        <Reveal from={{ y: 20 }} duration={0.6}>
          <SectionLabel>The stack</SectionLabel>
        </Reveal>

        <div className="mt-6 grid grid-cols-1 gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] lg:gap-16">
          <Reveal from={{ y: 40 }} delay={0.1} duration={0.8}>
            <h2 className="font-display-serif m-0 text-4xl leading-[1.1] tracking-tight text-fg-strong md:text-5xl">
              Everything ungated. <em className="not-italic text-fg-dim underline decoration-1 underline-offset-[0.14em] decoration-accent/60">Nothing waits on a human.</em>
            </h2>
            <p className="m-0 mt-6 text-base leading-relaxed text-fg-muted">
              A thousand rollouts is an inference bill, not a lab booking. GPU-seconds, cost per
              thousand, queue depth and live replica count all read from{" "}
              <code className="font-mono text-fg">async_queue_status</code>.
            </p>

            <Glass className="mt-8 rounded-none p-6 md:p-8">
              <p className="m-0 text-xs uppercase tracking-[0.18em] text-fg-dim">
                The burst, on stage
              </p>
              <p className="m-0 mt-4 font-display-serif text-3xl leading-tight text-fg-strong md:text-4xl">
                {BURST_TARGET.episodes.toLocaleString("en-US")} rollouts in{" "}
                {BURST_TARGET.seconds} seconds for ${BURST_TARGET.usd.toFixed(2)}
              </p>
              <p className="m-0 mt-4 text-sm leading-relaxed text-fg-muted">
                About {BURST_TARGET.gpuSecondsPerRollout} GPU-seconds a rollout, across{" "}
                {BURST_TARGET.replicas} replicas.
              </p>
              <p className="m-0 mt-4 font-mono text-[0.6875rem] uppercase tracking-[0.14em] text-caution">
                {BURST_TARGET.status}
              </p>
            </Glass>
          </Reveal>

          <Reveal from={{ y: 40 }} delay={0.2} duration={0.8}>
            <ul className="m-0 flex list-none flex-col p-0">
              {STACK.map((row) => (
                <li
                  key={`${row.layer}-${row.choice}`}
                  className="grid grid-cols-1 gap-1 border-t border-white/10 py-4 first:border-t-0 first:pt-0 sm:grid-cols-[8rem_minmax(0,1fr)] sm:gap-4"
                >
                  <span className="font-mono text-[0.6875rem] uppercase tracking-[0.14em] text-fg-dim">
                    {row.layer}
                  </span>
                  <span>
                    <span className="block font-mono text-sm text-fg-strong">{row.choice}</span>
                    <span className="mt-1 block text-sm leading-relaxed text-fg-muted">
                      {row.note}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------------ */

export function LimitsSection() {
  return (
    <section id="limits" className="relative overflow-hidden bg-black px-6 py-24 md:py-32">
      <div className="relative mx-auto max-w-6xl">
        <Reveal from={{ y: 20 }} duration={0.6}>
          <SectionLabel>What we cannot claim</SectionLabel>
        </Reveal>

        <Reveal from={{ y: 40 }} delay={0.1} duration={0.8}>
          <h2 className="font-display-serif m-0 mt-6 max-w-3xl text-4xl leading-[1.1] tracking-tight text-fg-strong md:text-5xl">
            Goalposts we cannot move <em className="not-italic text-fg-dim underline decoration-1 underline-offset-[0.14em] decoration-accent/60">until we cross them.</em>
          </h2>
        </Reveal>

        <ul className="m-0 mt-12 grid list-none grid-cols-1 gap-px overflow-hidden rounded-none bg-white/10 p-0 md:grid-cols-2">
          {LIMITS.map((limit, index) => (
            <li key={limit.title} className="bg-black">
              <Reveal from={{ y: 30 }} delay={index * 0.08} duration={0.7} className="h-full">
                <div className="h-full p-6 md:p-8">
                  <h3 className="m-0 text-lg tracking-tight text-fg-strong">{limit.title}</h3>
                  <p className="m-0 mt-3 text-sm leading-relaxed text-fg-muted">{limit.body}</p>
                </div>
              </Reveal>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------------ */

export function CtaSection() {
  return (
    <section className="relative overflow-hidden bg-black px-6 pb-24 pt-16 md:pb-32 md:pt-24">
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom,_rgba(123,230,96,0.13)_0%,_transparent_65%)]"
      />
      <div className="relative mx-auto max-w-4xl text-center">
        <Reveal from={{ y: 40 }} duration={0.8}>
          <h2 className="font-display-serif m-0 text-4xl leading-[1.05] tracking-tight text-fg-strong md:text-6xl lg:text-7xl">
            We measured the ruler <em className="not-italic text-fg-dim underline decoration-1 underline-offset-[0.14em] decoration-accent/60">before we trusted it.</em>
          </h2>
          <p className="m-0 mt-6 text-base leading-relaxed text-fg-muted md:text-lg">
            Everything above is a claim. The console is where it runs.
          </p>
        </Reveal>

        <Reveal from={{ y: 30 }} delay={0.12} duration={0.7}>
          <div className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Glass className="rounded-none liquid-glass-accent">
              <Link
                to={CONSOLE_PATH}
                className="flex items-center gap-2 px-8 py-3.5 text-sm font-medium text-accent-ink no-underline"
              >
                Open the console
                <Glyph name="arrowRight" />
              </Link>
            </Glass>
            <Glass className="rounded-none">
              <Link
                to={LIVE_PATH}
                className="flex items-center gap-2 px-8 py-3.5 text-sm font-medium text-fg-strong no-underline"
              >
                Start a live run
              </Link>
            </Glass>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

export function LandingFooter() {
  return (
    <footer className="border-t border-white/10 bg-black px-6 py-10">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4">
        <div>
          <p className="m-0 text-sm text-fg-muted">
            {PRODUCT.name} — {PRODUCT.tagline}
          </p>
          <p className="m-0 mt-1 text-[0.6875rem] text-fg-dim">
            Page imagery is illustrative. Generated rollouts carry provenance in the console.
          </p>
        </div>
        <ul className="m-0 flex list-none flex-wrap gap-6 p-0 text-sm">
          <li>
            <Link className="text-fg-muted no-underline hover:text-fg-strong" to={CONSOLE_PATH}>
              Console
            </Link>
          </li>
          <li>
            <Link className="text-fg-muted no-underline hover:text-fg-strong" to={LIVE_PATH}>
              Live run
            </Link>
          </li>
          <li>
            <a
              className="text-fg-muted no-underline hover:text-fg-strong"
              href="/api/protocol"
              target="_blank"
              rel="noreferrer"
            >
              Protocol record
            </a>
          </li>
          <li>
            <a
              className="text-fg-muted no-underline hover:text-fg-strong"
              href="/THIRD-PARTY-NOTICES.md"
              target="_blank"
              rel="noreferrer"
            >
              Third-party notices
            </a>
          </li>
        </ul>
      </div>
    </footer>
  );
}
