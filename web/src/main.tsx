import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { DevelopmentReview } from "./components/DevelopmentReview";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./styles.css";

const page = window.location.pathname === "/review" ? <DevelopmentReview /> : <App />;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* Regions have their own boundaries; this one is the last line of defence
        so a throw outside every panel still leaves a readable page. */}
    <ErrorBoundary region="PLUMB console">
      {page}
    </ErrorBoundary>
  </StrictMode>,
);
