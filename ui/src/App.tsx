import { Routes, Route } from "react-router-dom"
import { Providers } from "@/app/providers"
import { AppSidebar } from "@/components/app-sidebar"
import HomePage from "@/app/page"
import MetricsPage from "@/app/metrics/page"
import TimelinePage from "@/app/timeline/page"
import RolloutsPage from "@/app/rollouts/page"
import RolloutsDiscardedPage from "@/app/rollouts-discarded/page"
import TopologyPage from "@/app/topology/page"
import InfraPage from "@/app/infra/page"
import EvalsPage from "@/app/evals/page"
import AboutPage from "@/app/about/page"

export default function App() {
  return (
    <Providers>
      <AppSidebar />
      <main className="ml-56 h-screen overflow-hidden bg-background">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/metrics" element={<MetricsPage />} />
          <Route path="/timeline" element={<TimelinePage />} />
          <Route path="/rollouts" element={<RolloutsPage />} />
          <Route path="/rollouts-discarded" element={<RolloutsDiscardedPage />} />
          <Route path="/topology" element={<TopologyPage />} />
          <Route path="/infra" element={<InfraPage />} />
          <Route path="/evals" element={<EvalsPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </main>
    </Providers>
  )
}
