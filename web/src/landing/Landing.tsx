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
import { useEffect } from "react";
import { Hero } from "./Hero";
import "./landing.css";
import { PILLARS, PRODUCT } from "./content";
import { CtaSection, LandingFooter, LimitsSection, Pillars } from "./Sections";

export function Landing() {
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
      <Hero />
      <main>
        <Pillars />
        <LimitsSection />
        <CtaSection />
      </main>
      <LandingFooter />
    </div>
  );
}
