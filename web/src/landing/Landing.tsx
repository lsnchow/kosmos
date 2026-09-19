/**
 * The landing page.
 *
 * Deliberately outside `AppShell`: it has no topbar, no sidebar, no poll and no
 * event stream, so a visitor who never reaches the console costs the backend
 * nothing and a backend that is down costs the page nothing. Every figure it
 * shows is a constant in `content.ts`, so the page renders identically with no
 * network at all.
 */
import { useEffect } from "react";
import { Hero } from "./Hero";
import "./landing.css";
import {
  CalledShotSection,
  CtaSection,
  LandingFooter,
  LimitsSection,
  NumbersSection,
  PipelineSection,
  ProblemSection,
  ServicesSection,
  StackSection,
} from "./Sections";

export function Landing() {
  // The console sets a document title for the whole app; the landing is a
  // different document to a reader and to a link preview.
  useEffect(() => {
    const previous = document.title;
    document.title = "Nightshift — policy evaluation with error bars";
    return () => {
      document.title = previous;
    };
  }, []);

  return (
    <div className="bg-black">
      <a className="skip-link" href="#problem">
        Skip to content
      </a>
      <Hero />
      <main>
        <ProblemSection />
        <CalledShotSection />
        <PipelineSection />
        <NumbersSection />
        <ServicesSection />
        <StackSection />
        <LimitsSection />
        <CtaSection />
      </main>
      <LandingFooter />
    </div>
  );
}
