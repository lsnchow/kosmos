/**
 * The parts every landing section reuses: the scroll reveal, the section frame,
 * the button and the two-tone heading.
 *
 * `Reveal` is one component rather than a `motion.div` per section because the
 * reduced-motion decision has to be made in exactly one place. Under that
 * preference the content is rendered at its final position with no transform at
 * all — not a faster animation, none.
 */
import { motion, useInView, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";
import { useRef } from "react";
import { Link } from "react-router-dom";
import { Glyph } from "../components/Terminal";
import { cn } from "../lib/utils";

type RevealProps = {
  children: ReactNode;
  className?: string;
  /** Distance travelled on entry. A negative x comes from the left. */
  from?: { y?: number; x?: number };
  delay?: number;
  duration?: number;
};

export function Reveal({ children, className, from = { y: 40 }, delay = 0, duration = 0.8 }: RevealProps) {
  const ref = useRef<HTMLDivElement>(null);
  // `once` because a section that re-animates every time it scrolls back into
  // view reads as a glitch rather than an entrance.
  const inView = useInView(ref, { once: true, margin: "-100px" });
  const reduced = useReducedMotion();

  if (reduced) {
    return (
      <div ref={ref} className={className}>
        {children}
      </div>
    );
  }

  return (
    <motion.div
      ref={ref}
      className={className}
      initial={{ opacity: 0, y: from.y ?? 0, x: from.x ?? 0 }}
      animate={inView ? { opacity: 1, y: 0, x: 0 } : undefined}
      transition={{ duration, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}

/**
 * A labelled figure panel.
 *
 * `FIG.n` sits in the corner in mono at the dimmest ink on the page, the way a
 * plate is numbered in a paper. The graphic fills the panel above the pillar's
 * copy; the panel itself draws no border, because the cell it lives in already
 * has one and a second hairline inside it would read as a frame.
 */
export function FigPanel({
  index,
  children,
  className,
}: {
  index: number;
  children: ReactNode;
  className?: string;
}) {
  return (
    <figure className={cn("fig-panel", className)}>
      <figcaption className="fig-panel-label font-mono">FIG.{index}</figcaption>
      <div className="fig-panel-body">{children}</div>
    </figure>
  );
}

/** The small mono label that opens a section. */
export function SectionLabel({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p className={cn("m-0 font-mono text-xs uppercase tracking-[0.18em] text-fg-dim", className)}>
      {children}
    </p>
  );
}

/**
 * A framed surface.
 *
 * One hairline, square, over a flat fill. Depth on this page comes from exactly
 * one thing — a rule drawn or not drawn — so there is no blur, no shadow and no
 * gradient edge anywhere in the frame system.
 */
export function Glass({
  children,
  className,
  as: Component = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "aside" | "li" | "article";
}) {
  return <Component className={cn("liquid-glass", className)}>{children}</Component>;
}

/**
 * A full-bleed section card: the bordered box each pillar lives inside.
 *
 * The border is the page's structural grammar. Sections butt against each other
 * with a single shared hairline rather than floating on a margin, which is what
 * makes a long page read as one instrument panel instead of eight slides.
 */
export function SectionCard({
  id,
  children,
  className,
}: {
  id?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={cn("section-card", className)}>
      {children}
    </section>
  );
}

/**
 * The page's one button shape: a square-cornered pill with a trailing glyph.
 *
 * `primary` is reverse video — the single loudest element in any viewport — and
 * there is never more than one of them on screen at a time.
 */
export function PillButton({
  href,
  children,
  variant = "secondary",
  external = false,
  className,
}: {
  href: string;
  children: ReactNode;
  variant?: "primary" | "secondary";
  external?: boolean;
  className?: string;
}) {
  const classes = cn("pill-button", variant === "primary" ? "pill-primary" : "pill-secondary", className);
  const content = (
    <>
      <span>{children}</span>
      <Glyph name={external ? "arrowUpRight" : "arrowRight"} />
    </>
  );

  // An absolute path that leaves the SPA (the raw protocol JSON, the notices
  // file) has to be a real navigation; a router <Link> would 404 into the shell.
  if (external) {
    return (
      <a className={classes} href={href} target="_blank" rel="noreferrer">
        {content}
      </a>
    );
  }

  return (
    <Link className={classes} to={href}>
      {content}
    </Link>
  );
}

/**
 * A heading in two inks: a white lead-in, then the sentence in muted grey.
 *
 * The split carries the hierarchy that a single weight cannot at this size —
 * the lead-in names the pillar, the remainder makes the claim.
 */
export function TwoToneHeading({
  lead,
  children,
  className,
  as: Component = "h2",
}: {
  lead: string;
  children: ReactNode;
  className?: string;
  as?: "h1" | "h2";
}) {
  return (
    <Component className={cn("two-tone-heading", className)}>
      <span className="text-fg-strong">{lead}</span> <span className="text-fg-muted">{children}</span>
    </Component>
  );
}

/** A figure with its provenance underneath. Used wherever a number appears. */
export function Figure({
  value,
  unit,
  note,
  source,
  className,
}: {
  value: string;
  unit?: string;
  note?: string;
  source?: string;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <p className="m-0 flex items-baseline gap-2">
        <span className="font-sans text-4xl leading-none text-fg-strong md:text-5xl">{value}</span>
        {unit && <span className="text-sm text-fg-muted">{unit}</span>}
      </p>
      {note && <p className="m-0 text-sm leading-relaxed text-fg-muted">{note}</p>}
      {source && (
        <p className="m-0 font-mono text-[0.6875rem] leading-relaxed text-fg-dim">{source}</p>
      )}
    </div>
  );
}
