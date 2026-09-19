/**
 * Routing and providers. Nothing else.
 *
 * This file was 567 lines holding sixteen `useState` hooks, every fetch, and
 * ten stacked sections. The state moved to `AppData`, the chrome to `AppShell`,
 * and each section to the page it belongs on.
 *
 * `/` is the landing page and `/console` is the console's overview. The five
 * working pages keep the flat paths they have always had — `/live` in
 * particular is typed from memory under stage lights, and moving it to buy URL
 * symmetry would be a bad trade.
 */
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AppDataProvider } from "./AppData";
import { AppShell } from "./AppShell";
import { Landing } from "./landing/Landing";
import { DevelopmentReview } from "./components/DevelopmentReview";
import { ClipsPage } from "./pages/ClipsPage";
import { CostPage } from "./pages/CostPage";
import { EvidencePage } from "./pages/EvidencePage";
import { LivePage } from "./pages/LivePage";
import { OverviewPage } from "./pages/OverviewPage";
import { ResultsPage } from "./pages/ResultsPage";

export function AppRoutes() {
  return (
    <Routes>
      <Route index element={<Landing />} />
      {/*
       * The review tool sits outside <AppShell/> deliberately. It is an
       * internal annotation surface with its own masthead and its own
       * `.console` wrapper, and a reviewer scoring clips has no use for the
       * sidebar, the run poller or the event stream. It arrived on main as a
       * `window.location.pathname === "/review"` check in main.tsx, which
       * predates the router; as a route it survives a client-side navigation
       * and a hard reload alike.
       */}
      <Route path="review" element={<DevelopmentReview />} />
      <Route element={<AppShell />}>
        <Route path="console" element={<OverviewPage />} />
        <Route path="live" element={<LivePage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="evidence" element={<EvidencePage />} />
        <Route path="cost" element={<CostPage />} />
        <Route path="clips" element={<ClipsPage />} />
        {/* An unknown path shows the overview rather than a dead end. */}
        <Route path="*" element={<OverviewPage />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  return (
<AppDataProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AppDataProvider>
  );
}
