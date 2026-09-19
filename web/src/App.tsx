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
