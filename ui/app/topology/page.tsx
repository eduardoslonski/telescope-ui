
import { useMemo } from "react"
import { useAtomValue } from "jotai"
import { selectedRunPathAtom } from "@/lib/atoms"
import { useRunSummary } from "@/hooks/use-run-data"
import { NoRunSelectedState } from "@/components/no-run-selected-state"
import { TopologyViewer, parseTopology } from "@/components/topology-viewer"

export default function TopologyPage() {
  const selectedRunPath = useAtomValue(selectedRunPathAtom)

  const { data: summaryData, error } = useRunSummary(
    selectedRunPath || "",
    !!selectedRunPath,
    false // no polling needed for static topology
  )

  const topology = useMemo(
    () => parseTopology(summaryData?.summary),
    [summaryData?.summary]
  )

  if (!selectedRunPath) {
    return (
      <NoRunSelectedState description="Select a run from the sidebar to view cluster topology." />
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-sm text-destructive mb-2">
            Failed to load topology data
          </div>
          <div className="text-xs text-muted-foreground">
            {error instanceof Error ? error.message : "Unknown error"}
          </div>
        </div>
      </div>
    )
  }

  if (!summaryData) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-sm text-muted-foreground animate-pulse">
          Loading topology…
        </div>
      </div>
    )
  }

  if (!topology || topology.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-sm text-muted-foreground mb-2">
            No topology data available
          </div>
          <div className="text-xs text-muted-foreground/70">
            The selected run does not have setup information in its W&B summary.
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full w-full">
      <TopologyViewer topology={topology} />
    </div>
  )
}

