/**
 * The landing page.
 *
 * Deliberately outside `AppShell`: it has no topbar, no sidebar, no poll and no
 * event stream, so a visitor who never reaches the console costs the backend
 * nothing and a backend that is down costs the page nothing. Every figure it
 * shows is a constant in `content.ts`, so the page renders identically with no
 * network at all.
 *
 * `.kosmos-landing` is load-bearing. It redeclares the ink and surface tokens
 * for this subtree only, which is what lets the landing run a neutral near-black
 * palette while the console keeps its own without either sheet knowing about the
 * other. Remove the class and the page inherits the console's charcoal.
 */
import { useEffect, useRef, useState } from "react";
import { Hero } from "./Hero";
import "./landing.css";
import { PILLARS, PRODUCT } from "./content";
import { Nav } from "./Nav";
import { CtaSection, LandingFooter, Pillars } from "./Sections";

/**
 * True once the reader has scrolled past the element the ref is on.
 *
 * An observer rather than a scroll listener: the question is only ever "is this
 * one element still on screen", which is what `IntersectionObserver` answers
 * natively, off the main thread, without a handler firing on every frame of a
 * scroll it does not care about.
 *
 * `boundingClientRect.top < 0` is the half that matters. A sentinel is also
 * "not intersecting" while it is still *below* the fold, and without that check
 * the bar would be visible on first paint — the one moment it must not be.
 */
function useScrolledPast() {
  const ref = useRef<HTMLDivElement>(null);
  const [past, setPast] = useState(false);

  useEffect(() => {
    const sentinel = ref.current;
    if (!sentinel || typeof IntersectionObserver === "undefined") return;

    const observer = new IntersectionObserver(([entry]) => {
      setPast(!entry.isIntersecting && entry.boundingClientRect.top < 0);
    });
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, []);

  return { ref, past };
}

export function Landing() {
  const { ref: sentinelRef, past } = useScrolledPast();

  // The console sets a document title for the whole app; the landing is a
  // different document to a reader and to a link preview.
  useEffect(() => {
    const previous = document.title;
    document.title = `${PRODUCT.name} — ${PRODUCT.tagline.toLowerCase().replace(/\.$/, "")}`;
    return () => {
      document.title = previous;
    };
  }, []);

  return (
    <div className="kosmos-landing">
      <a className="skip-link" href={`#${PILLARS[0].id}`}>
        Skip to content
      </a>
      <Nav visible={past} />
      <Hero />
      {/*
        Sits on the hero's bottom edge. The bar appears exactly when this
        crosses the top of the viewport, so the reveal is tied to the hero
        ending rather than to a pixel count that a shorter screen would hit at
        a different point in the page.
      */}
      <div ref={sentinelRef} className="nav-sentinel" aria-hidden="true" />
      <main>
        <Pillars />
        <CtaSection />
      </main>
      <LandingFooter />
    </div>
  );
}
