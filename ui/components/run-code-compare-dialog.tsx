
import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Spinner } from "@/components/ui/spinner"
import { useRuns } from "@/hooks/use-run-data"
import { formatTimeAgo } from "@/lib/format"
import type { Run } from "@/lib/types"
import { cn } from "@/lib/utils"
import { MiddleEllipsisLabel } from "@/components/middle-ellipsis-label"

function RunSelector({
  runs,
  currentRunId,
  selectedCompareRunId,
  onSelect,
}: {
  runs: Run[]
  currentRunId: string
  selectedCompareRunId: string | null
  onSelect: (runId: string) => void
}) {
  return (
    <div className="flex flex-col gap-1">
      {runs.map((run) => {
        const isCurrent = run.run_id === currentRunId
        const isSelected = run.run_id === selectedCompareRunId
        const displayName = run.name || run.run_id.split("/").pop() || run.run_id
        return (
          <button
            key={run.run_id}
            onClick={() => !isCurrent && onSelect(run.run_id)}
            disabled={isCurrent}
            className={cn(
              "flex w-full items-center gap-2 rounded-md px-3 py-2 text-left transition-colors",
              isCurrent
                ? "cursor-not-allowed opacity-40"
                : cn(
                  "cursor-pointer hover:bg-muted",
                  isSelected && "bg-muted"
                )
            )}
          >
            <div
              className="h-2.5 w-2.5 shrink-0 rounded-full"
              style={{ backgroundColor: run.color }}
            />
            <div className="min-w-0 flex-1">
              <div className="min-w-0">
                <MiddleEllipsisLabel text={displayName} className="text-sm font-medium" />
              </div>
              <div className="truncate text-xs text-muted-foreground">{run.run_id}</div>
            </div>
            {isCurrent && (
              <Badge variant="secondary" className="h-4 px-1.5 text-[10px] shrink-0">
                Current
              </Badge>
            )}
            {!isCurrent && isSelected && (
              <Badge variant="secondary" className="h-4 px-1.5 text-[10px] shrink-0">
                Comparing
              </Badge>
            )}
            {run.is_tracking && (
              <span
                className="h-2 w-2 rounded-full bg-green-500 shrink-0"
                title="Live"
              />
            )}
            {run.is_syncing && (
              <span
                className="h-2 w-2 rounded-full bg-blue-500 shrink-0"
                title="Syncing"
              />
            )}
            {run.created_at && (
              <span className="shrink-0 text-[10px] text-muted-foreground">
                {formatTimeAgo(run.created_at)}
              </span>
            )}
          </button>
        )
      })}
      {runs.length === 0 && (
        <div className="py-6 text-center text-sm text-muted-foreground">
          No other runs available.
        </div>
      )}
    </div>
  )
}

export function RunCodeCompareDialog({
  currentRunId,
  selectedCompareRunId,
  onSelectCompareRun,
  disabled = false,
}: {
  currentRunId: string
  selectedCompareRunId: string | null
  onSelectCompareRun: (runId: string | null) => void
  disabled?: boolean
}) {
  const [isOpen, setIsOpen] = useState(false)

  const { data: runsData, isLoading: isLoadingRuns } = useRuns()
  const runs = runsData?.runs ?? []
  const currentRun = runs.find((run) => run.run_id === currentRunId) ?? null
  const currentRunName = currentRun?.name || currentRunId.split("/").pop() || currentRunId
  const selectedCompareRun = runs.find((run) => run.run_id === selectedCompareRunId) ?? null
  const selectedCompareRunName =
    selectedCompareRun?.name || selectedCompareRun?.run_id.split("/").pop() || null

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        setIsOpen(open)
      }}
    >
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="h-7 px-2 text-xs" disabled={disabled}>
          Compare
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-7xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle>Select Run To Compare</DialogTitle>
          <DialogDescription>
            Choose another run to compare against{" "}
            <span className="font-sans text-foreground">{currentRunName}</span>.
          </DialogDescription>
        </DialogHeader>
        {selectedCompareRunId && (
          <div className="flex items-center justify-between rounded-md border px-3 py-2 text-xs">
            <span className="text-muted-foreground">
              Currently comparing with{" "}
              <span className="font-medium text-foreground">
                {selectedCompareRunName || selectedCompareRunId}
              </span>
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => {
                onSelectCompareRun(null)
                setIsOpen(false)
              }}
            >
              Stop comparing
            </Button>
          </div>
        )}
        <ScrollArea className="h-[480px] pr-4">
          {isLoadingRuns ? (
            <div className="flex items-center justify-center py-8">
              <Spinner className="h-5 w-5" />
            </div>
          ) : (
            <RunSelector
              runs={runs}
              currentRunId={currentRunId}
              selectedCompareRunId={selectedCompareRunId}
              onSelect={(runId) => {
                onSelectCompareRun(runId)
                setIsOpen(false)
              }}
            />
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}

