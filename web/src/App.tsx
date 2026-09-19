/**
 * Routing and providers. Nothing else.
 *
 * This file was 567 lines holding sixteen `useState` hooks, every fetch, and
 * ten stacked sections. The state moved to `AppData`, the chrome to `AppShell`,
 * and each section to the page it belongs on.
 */
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AppDataProvider } from "./AppData";
import { AppShell } from "./AppShell";
import { ClipsPage } from "./pages/ClipsPage";
import { CostPage } from "./pages/CostPage";
import { EvidencePage } from "./pages/EvidencePage";
import { LivePage } from "./pages/LivePage";
import { OverviewPage } from "./pages/OverviewPage";
import { ResultsPage } from "./pages/ResultsPage";

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<OverviewPage />} />
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
