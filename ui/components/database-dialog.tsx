
import { useCallback, useEffect, useRef, useState } from "react"
import { CheckCircle2, Database, XCircle } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { API_BASE } from "@/lib/constants"
import { cn } from "@/lib/utils"

interface CompactionStatus {
  status: "idle" | "pausing_syncs" | "exporting" | "importing" | "finalizing" | "done" | "error"
  error: string | null
  size_before: number | null
  size_after: number | null
}

const STEPS = [
  { key: "pausing_syncs", label: "Pausing syncs" },
  { key: "exporting", label: "Exporting database" },
  { key: "importing", label: "Compressing data" },
  { key: "finalizing", label: "Finalizing" },
] as const

type StepKey = (typeof STEPS)[number]["key"]

const STEP_ORDER: StepKey[] = STEPS.map((s) => s.key)

function stepIndex(status: string): number {
  return STEP_ORDER.indexOf(status as StepKey)
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${bytes} B`
}

export function DatabaseDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const [dbSize, setDbSize] = useState<number | null>(null)
  const [loadingSize, setLoadingSize] = useState(false)
  const [compaction, setCompaction] = useState<CompactionStatus | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const isCompacting =
    compaction != null &&
    compaction.status !== "idle" &&
    compaction.status !== "done" &&
    compaction.status !== "error"

  const fetchSize = useCallback(async () => {
    setLoadingSize(true)
    try {
      const res = await fetch(`${API_BASE}/database-info`)
      if (res.ok) {
        const data = await res.json()
        setDbSize(data.size_bytes)
      }
    } catch {
      // ignore
    } finally {
      setLoadingSize(false)
    }
  }, [])

  useEffect(() => {
    if (open) {
      fetchSize()
      setCompaction(null)
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [open, fetchSize])

  const startCompaction = async () => {
    try {
      const res = await fetch(`${API_BASE}/compact-database`, { method: "POST" })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        setCompaction({
          status: "error",
          error: data.detail || "Failed to start compaction",
          size_before: null,
          size_after: null,
        })
        return
      }

      setCompaction({ status: "pausing_syncs", error: null, size_before: null, size_after: null })

      pollRef.current = setInterval(async () => {
        try {
          const r = await fetch(`${API_BASE}/compact-database/status`)
          if (r.ok) {
            const status: CompactionStatus = await r.json()
            setCompaction(status)
            if (status.status === "done" || status.status === "error") {
              if (pollRef.current) clearInterval(pollRef.current)
              pollRef.current = null
              if (status.status === "done" && status.size_after != null) {
                setDbSize(status.size_after)
              }
            }
          }
        } catch {
          // keep polling
        }
      }, 500)
    } catch (e) {
      setCompaction({
        status: "error",
        error: e instanceof Error ? e.message : "Unknown error",
        size_before: null,
        size_after: null,
      })
    }
  }

  const handleOpenChange = (v: boolean) => {
    if (isCompacting && !v) return
    onOpenChange(v)
  }

  const currentStepIdx = compaction ? stepIndex(compaction.status) : -1
  const isDone = compaction?.status === "done"
  const isError = compaction?.status === "error"
  const showProgress = compaction != null && compaction.status !== "idle"

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        className={cn("sm:max-w-md", isCompacting && "[&>[data-slot=dialog-close]]:hidden")}
        onInteractOutside={(e) => { if (isCompacting) e.preventDefault() }}
        onEscapeKeyDown={(e) => { if (isCompacting) e.preventDefault() }}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Database
          </DialogTitle>
          {!showProgress && (
            <DialogDescription>
              Manage your local database storage.
            </DialogDescription>
          )}
        </DialogHeader>

        {!showProgress && (
          <div className="space-y-4 py-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Database size</span>
              <span className="text-sm font-medium">
                {loadingSize ? (
                  <Spinner className="h-3.5 w-3.5" />
                ) : dbSize != null ? (
                  formatBytes(dbSize)
                ) : (
                  "—"
                )}
              </span>
            </div>
            <div className="rounded-md border border-border bg-muted/30 p-3">
              <p className="text-xs text-muted-foreground">
                Compressing rewrites the database to reclaim unused space and
                apply optimal compression. This can reduce the file size
                by 20–30% depending on the data. Syncs will be paused during compression.
              </p>
            </div>
          </div>
        )}

        {showProgress && (
          <div className="space-y-3 py-2">
            {STEPS.map((step, i) => {
              const isCurrent = step.key === compaction.status
              const isCompleted = isDone || currentStepIdx > i || (isError && currentStepIdx > i)
              const isFailed = isError && step.key === compaction.status

              return (
                <div key={step.key} className="flex items-center gap-2.5">
                  {isCompleted ? (
                    <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0" />
                  ) : isFailed ? (
                    <XCircle className="h-4 w-4 text-destructive shrink-0" />
                  ) : isCurrent ? (
                    <Spinner className="h-4 w-4 text-primary shrink-0" />
                  ) : (
                    <div className="h-4 w-4 rounded-full border border-border shrink-0" />
                  )}
                  <span
                    className={cn(
                      "text-sm",
                      isCompleted && "text-muted-foreground",
                      isCurrent && !isError && "text-foreground font-medium",
                      isFailed && "text-destructive font-medium",
                      !isCompleted && !isCurrent && !isFailed && "text-muted-foreground/60"
                    )}
                  >
                    {step.label}
                    {isCurrent && !isDone && !isError && "…"}
                  </span>
                </div>
              )
            })}

            {isCompacting && (
              <p className="text-xs text-muted-foreground pt-1">
                Do not close the application while compressing.
              </p>
            )}

            {isDone && compaction.size_before != null && compaction.size_after != null && (
              <div className="rounded-md border border-green-500/20 bg-green-500/5 p-3 mt-2">
                <p className="text-sm text-foreground">
                  <span className="font-medium">
                    {formatBytes(compaction.size_before)}
                  </span>
                  {" → "}
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {formatBytes(compaction.size_after)}
                  </span>
                  <span className="text-muted-foreground ml-1.5">
                    ({Math.round((1 - compaction.size_after / compaction.size_before) * 100)}% reduction)
                  </span>
                </p>
              </div>
            )}

            {isError && compaction.error && (
              <div className="rounded-md border border-destructive/20 bg-destructive/5 p-3 mt-2">
                <p className="text-sm text-destructive">
                  {compaction.error}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  The original database has been restored.
                </p>
              </div>
            )}
          </div>
        )}

        <DialogFooter>
          {!showProgress && (
            <Button onClick={startCompaction} disabled={loadingSize || dbSize == null}>
              Compress Database
            </Button>
          )}
          {(isDone || isError) && (
            <Button onClick={() => onOpenChange(false)}>
              Close
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
