/**
 * The parts every landing section reuses: the scroll reveal, the glass surfaces
 * and the hero's dream wall.
 *
 * `Reveal` is one component rather than a `motion.div` per section because the
 * reduced-motion decision has to be made in exactly one place. Under that
 * preference the content is rendered at its final position with no transform at
 * all — not a faster animation, none.
 */
import { motion, useInView, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";
import { useRef } from "react";
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

/** The small label that opens every section. */
export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <p className="m-0 text-xs text-fg-dim">{children}</p>
  );
}

/**
 * A glass panel.
 *
 * `liquid-glass` paints a masked gradient border through a `::before`, so the
 * element needs its own border radius passed down for the mask to follow.
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
 * The hero background: twelve tiles that ignite in sequence and then extend in
 * five chunks, on a loop.
 *
 * It is a schematic and the page labels it one. Drawing it in CSS rather than
 * shipping a video is the same rule the console follows — nothing on the page
 * can fail to load and leave a broken element behind — and it means the one
 * thing the illustration shows is the one thing a still frame cannot: a rollout
 * that grows a chunk at a time.
 */
export function DreamWall({ className }: { className?: string }) {
  return (
    <div
      className={cn("grid grid-cols-4 grid-rows-3 gap-px bg-white/10", className)}
      aria-hidden="true"
    >
      {Array.from({ length: 12 }, (_, index) => (
        <div
          key={index}
          className="dream-tile"
          // 40–60 ms between tiles: fast enough to read as one wall igniting,
          // slow enough that the eye still resolves individual tiles.
          style={{ animationDelay: `${index * 52}ms` }}
        >
          <span
            className="dream-chunk"
            style={{ animationDelay: `${(index % 5) * 420}ms` }}
          />
        </div>
      ))}
    </div>
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
        <span className="font-display-serif text-4xl leading-none text-fg-strong md:text-5xl">
          {value}
        </span>
        {unit && <span className="text-sm text-fg-muted">{unit}</span>}
      </p>
      {note && <p className="m-0 text-sm leading-relaxed text-fg-muted">{note}</p>}
      {source && (
        <p className="m-0 font-mono text-[0.6875rem] leading-relaxed text-fg-dim">{source}</p>
      )}
    </div>
  );
}
