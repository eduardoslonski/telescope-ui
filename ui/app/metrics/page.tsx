
import { useState, useMemo, useEffect, useRef, useCallback } from "react"
import { formatDurationHms } from "@/lib/format"
import { useAtom, useAtomValue } from "jotai"
import { useQueryClient } from "@tanstack/react-query"
import { Plus } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { NoRunSelectedState } from "@/components/no-run-selected-state"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { ToggleWithInput } from "@/components/ui/toggle-with-input"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { API_BASE } from "@/lib/constants"
import { StepMetricsCharts } from "@/components/step-metrics-charts"
import { CustomMetricsView } from "@/components/custom-metrics-view"
import {
  selectedRunPathAtom,
  visibleRunsAtom,
  metricsShowEmaAtom,
  metricsEmaSpanAtom,
  metricsXAxisModeAtom,
  metricsMaxLimitEnabledAtom,
  metricsMaxStepAtom,
  metricsMaxTimeAtom,
  metricsScrollTopAtom,
  metricsViewModeAtom,
  metricsActiveTemplateIdAtom,
  metricsActiveTemplateNameAtom,
  hoveredRunIdAtom,
} from "@/lib/atoms"
import {
  useRunSummaries,
  useRuns,
  useCustomMetricsLayout,
  useCustomMetricsTemplates,
} from "@/hooks/use-run-data"
import type { CustomMetricsTemplateSummary } from "@/lib/types"

const formatDuration = formatDurationHms

/**
 * Parse a duration string into seconds.
 * Accepts: "30s", "10m", "2h", "1h30m", "1m30s", or a plain number (defaults to seconds).
 * Returns null if the input is invalid.
 */
function parseDuration(input: string): number | null {
  const trimmed = input.trim()
  if (!trimmed) return null

  // Try compound pattern like "1h30m", "2m30s", "1h30m15s", etc.
  const compoundRe = /^(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?$/i
  const compoundMatch = trimmed.match(compoundRe)
  if (
    compoundMatch &&
    (compoundMatch[1] || compoundMatch[2] || compoundMatch[3])
  ) {
    const h = parseInt(compoundMatch[1] || "0", 10)
    const m = parseInt(compoundMatch[2] || "0", 10)
    const s = parseInt(compoundMatch[3] || "0", 10)
    const total = h * 3600 + m * 60 + s
    return total > 0 ? total : null
  }

  // Plain number -> default to seconds
  const num = parseFloat(trimmed)
  if (Number.isFinite(num) && num > 0) return Math.round(num)

  return null
}

export default function MetricsPage() {
  const [scrollRoot, setScrollRoot] = useState<HTMLDivElement | null>(null)
  const scrollTopRef = useRef(0)
  const [savedScrollTop, setSavedScrollTop] = useAtom(metricsScrollTopAtom)
  const selectedRunPath = useAtomValue(selectedRunPathAtom)
  const visibleRuns = useAtomValue(visibleRunsAtom)
  const hoveredRunId = useAtomValue(hoveredRunIdAtom)
  const [viewMode, setViewMode] = useAtom(metricsViewModeAtom)

  // Ref to the CustomMetricsView to trigger "New Section" from header
  const [newSectionTrigger, setNewSectionTrigger] = useState(0)

  // Template state
  const queryClient = useQueryClient()
  const [activeTemplateId, setActiveTemplateId] = useAtom(metricsActiveTemplateIdAtom)
  const [activeTemplateName, setActiveTemplateName] = useAtom(metricsActiveTemplateNameAtom)
  const { data: templatesData } = useCustomMetricsTemplates()
  const templates = templatesData?.templates ?? []
  const [loadTemplateOpen, setLoadTemplateOpen] = useState(false)
  const [nameDialogOpen, setNameDialogOpen] = useState(false)
  const [nameDialogMode, setNameDialogMode] = useState<"new" | "duplicate" | "rename">("new")
  const [nameDialogValue, setNameDialogValue] = useState("")
  const [deleteTemplateOpen, setDeleteTemplateOpen] = useState(false)
  const [layoutSnapshotTrigger, setLayoutSnapshotTrigger] = useState(0)
  const layoutSnapshotCallbackRef = useRef<((layout: unknown) => void) | null>(null)
  const pendingSaveLayoutRef = useRef<unknown>(null)

  const openNameDialog = useCallback((mode: "new" | "duplicate" | "rename") => {
    setNameDialogMode(mode)
    setNameDialogValue(mode === "rename" ? (activeTemplateName ?? "") : "")
    if (mode === "new") {
      pendingSaveLayoutRef.current = { sections: [] }
      setNameDialogOpen(true)
    } else if (mode === "duplicate") {
      layoutSnapshotCallbackRef.current = (currentLayout: unknown) => {
        layoutSnapshotCallbackRef.current = null
        pendingSaveLayoutRef.current = currentLayout
        setNameDialogOpen(true)
      }
      setLayoutSnapshotTrigger((n) => n + 1)
    } else {
      setNameDialogOpen(true)
    }
  }, [activeTemplateName])

  const handleNameDialogSubmit = useCallback(async () => {
    const name = nameDialogValue.trim()
    if (!name) return

    if (nameDialogMode === "new" || nameDialogMode === "duplicate") {
      const layout = pendingSaveLayoutRef.current ?? { sections: [] }
      const res = await fetch(`${API_BASE}/custom-metrics-templates`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, layout }),
      })
      if (res.ok) {
        const data = await res.json()
        setActiveTemplateId(data.id)
        setActiveTemplateName(data.name)
        queryClient.invalidateQueries({ queryKey: ["custom-metrics-templates"] })
      }
      pendingSaveLayoutRef.current = null
    } else if (nameDialogMode === "rename" && activeTemplateId) {
      const res = await fetch(`${API_BASE}/custom-metrics-templates/${activeTemplateId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      })
      if (res.ok) {
        setActiveTemplateName(name)
        queryClient.invalidateQueries({ queryKey: ["custom-metrics-templates"] })
      }
    }
    setNameDialogOpen(false)
  }, [nameDialogValue, nameDialogMode, activeTemplateId, setActiveTemplateId, setActiveTemplateName, queryClient])

  const handleLoadTemplate = useCallback(
    (template: CustomMetricsTemplateSummary | null) => {
      if (template === null) {
        setActiveTemplateId(null)
        setActiveTemplateName(null)
      } else {
        setActiveTemplateId(template.id)
        setActiveTemplateName(template.name)
      }
      setLoadTemplateOpen(false)
    },
    [setActiveTemplateId, setActiveTemplateName],
  )

  const handleDeleteTemplate = useCallback(async () => {
    if (!activeTemplateId) return
    await fetch(`${API_BASE}/custom-metrics-templates/${activeTemplateId}`, {
      method: "DELETE",
    })
    setActiveTemplateId(null)
    setActiveTemplateName(null)
    queryClient.invalidateQueries({ queryKey: ["custom-metrics-templates"] })
    setDeleteTemplateOpen(false)
  }, [activeTemplateId, setActiveTemplateId, setActiveTemplateName, queryClient])

  // EMA settings (persisted in atoms)
  const [showEma, setShowEma] = useAtom(metricsShowEmaAtom)
  const [emaSpan, setEmaSpan] = useAtom(metricsEmaSpanAtom)
  const [emaSpanInput, setEmaSpanInput] = useState(emaSpan.toString())
  const [xAxisMode, setXAxisMode] = useAtom(metricsXAxisModeAtom)

  // Max limit settings (persisted in atoms)
  const [maxLimitEnabled, setMaxLimitEnabled] = useAtom(
    metricsMaxLimitEnabledAtom,
  )
  const [maxStep, setMaxStep] = useAtom(metricsMaxStepAtom)
  const [maxTime, setMaxTime] = useAtom(metricsMaxTimeAtom)
  const [maxStepInput, setMaxStepInput] = useState(maxStep.toString())
  const [maxTimeInput, setMaxTimeInput] = useState(formatDuration(maxTime))

  useEffect(() => {
    setMaxStepInput(maxStep.toString())
  }, [maxStep])

  useEffect(() => {
    setMaxTimeInput(formatDuration(maxTime))
  }, [maxTime])

  // Prefetch custom metrics layout so it's cached when switching to Custom tab
  useCustomMetricsLayout()

  // Always poll when on this page - ensures data stays fresh
  const shouldPoll = true

  // Restore scroll position when the scroll container mounts
  useEffect(() => {
    if (!scrollRoot) return
    scrollRoot.scrollTop = savedScrollTop
    scrollTopRef.current = savedScrollTop
    // Only run when scrollRoot is first set (not when savedScrollTop changes)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scrollRoot])

  // Track scroll position and persist it on unmount
  useEffect(() => {
    if (!scrollRoot) return
    const handleScroll = () => {
      scrollTopRef.current = scrollRoot.scrollTop
    }
    scrollRoot.addEventListener("scroll", handleScroll, { passive: true })
    return () => {
      scrollRoot.removeEventListener("scroll", handleScroll)
      setSavedScrollTop(scrollTopRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scrollRoot])

  // Get all runs to get their persistent colors
  const { data: runsData } = useRuns()

  // Compute runs to display with their colors
  // Selected run is always included, plus any toggled visible runs
  const runsToDisplay = useMemo(() => {
    const allRuns = runsData?.runs || []
    const runsToShow = new Set<string>(visibleRuns)
    if (selectedRunPath) runsToShow.add(selectedRunPath)

    return allRuns
      .filter((run) => runsToShow.has(run.run_id))
      .map((run) => ({
        runPath: run.run_id,
        runName: run.name,
        color: run.color,
        isSelected: run.run_id === selectedRunPath,
      }))
  }, [selectedRunPath, visibleRuns, runsData?.runs])

  // Get summaries from all visible runs to build the full metric catalog
  const allRunPaths = useMemo(() => {
    const paths = new Set<string>(visibleRuns)
    if (selectedRunPath) paths.add(selectedRunPath)
    return Array.from(paths)
  }, [selectedRunPath, visibleRuns])

  const {
    customMetricSections,
    availableRewardNames,
    evalsList,
    availableSampleTags,
    availableEnvs,
    totalSteps,
    error,
  } = useRunSummaries(allRunPaths, shouldPoll)

  // No run selected
  if (!selectedRunPath) {
    return (
      <NoRunSelectedState description="Select a run from the sidebar to view training metrics." />
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="py-2.5 px-4">
          <div className="flex items-center justify-start gap-2 h-7">
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={viewMode}
              onValueChange={(value) => {
                if (value) setViewMode(value as "all" | "custom")
              }}
            >
              <ToggleGroupItem value="all" className="text-xs px-2 py-1 h-7">
                All
              </ToggleGroupItem>
              <ToggleGroupItem value="custom" className="text-xs px-2 py-1 h-7">
                Custom
              </ToggleGroupItem>
            </ToggleGroup>
            {viewMode === "custom" && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs px-2"
                  onClick={() => setNewSectionTrigger((n) => n + 1)}
                >
                  <Plus className="h-3 w-3 mr-1" />
                  New Section
                </Button>
                <div className="w-px h-4 bg-border mx-0.5" />
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm" className="h-7 text-xs px-2">
                      Templates
                      {activeTemplateName && (
                        <span className="ml-1 text-muted-foreground">
                          ({activeTemplateName})
                        </span>
                      )}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start">
                    <DropdownMenuItem onClick={() => setLoadTemplateOpen(true)}>
                      Load
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => openNameDialog("new")}>
                      New
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => openNameDialog("duplicate")}>
                      Duplicate
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem
                      onClick={() => openNameDialog("rename")}
                      disabled={!activeTemplateId}
                    >
                      Rename
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => setDeleteTemplateOpen(true)}
                      disabled={!activeTemplateId}
                      className="text-destructive focus:text-destructive"
                    >
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </>
            )}
            <div className="w-px h-4 bg-border mx-0.5" />
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={xAxisMode}
              onValueChange={(value) => {
                if (value) setXAxisMode(value as "step" | "time")
              }}
            >
              <ToggleGroupItem value="step" className="text-xs px-2 py-1 h-7">
                Step
              </ToggleGroupItem>
              <ToggleGroupItem value="time" className="text-xs px-2 py-1 h-7">
                Time
              </ToggleGroupItem>
            </ToggleGroup>
            <ToggleWithInput
              label="EMA"
              variant="selecting"
              size="sm"
              enabled={showEma}
              onEnabledChange={setShowEma}
              value={emaSpanInput}
              onValueChange={setEmaSpanInput}
              onValueCommit={(value) => {
                const parsed = parseInt(value, 10)
                if (!isNaN(parsed) && parsed >= 2 && parsed <= 100) {
                  setEmaSpan(parsed)
                  setEmaSpanInput(parsed.toString())
                } else {
                  setEmaSpanInput(emaSpan.toString())
                }
              }}
              inputMin={2}
              inputMax={100}
              inputWidth="w-10"
            />
            <ToggleWithInput
              label={xAxisMode === "step" ? "Max Step" : "Max Time"}
              variant="selecting"
              size="sm"
              enabled={maxLimitEnabled}
              onEnabledChange={setMaxLimitEnabled}
              value={xAxisMode === "step" ? maxStepInput : maxTimeInput}
              inputType={xAxisMode === "step" ? "number" : "text"}
              onValueChange={(value) => {
                if (xAxisMode === "step") {
                  setMaxStepInput(value)
                } else {
                  setMaxTimeInput(value)
                }
              }}
              onValueCommit={(value) => {
                if (xAxisMode === "step") {
                  const parsed = parseInt(value, 10)
                  if (!isNaN(parsed) && parsed >= 1) {
                    setMaxStep(parsed)
                    setMaxStepInput(parsed.toString())
                  } else {
                    setMaxStepInput(maxStep.toString())
                  }
                } else {
                  const parsed = parseDuration(value)
                  if (parsed !== null) {
                    setMaxTime(parsed)
                    setMaxTimeInput(formatDuration(parsed))
                  } else {
                    setMaxTimeInput(formatDuration(maxTime))
                  }
                }
              }}
              inputMin={1}
              inputWidth="w-16"
            />
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6" ref={setScrollRoot}>
        <div className="max-w-7xl mx-auto">
          {error && (
            <Card className="border-destructive mb-6">
              <CardContent className="pt-6">
                <p className="text-destructive">
                  Error:{" "}
                  {error instanceof Error ? error.message : "Unknown error"}
                </p>
              </CardContent>
            </Card>
          )}

          {viewMode === "all" ? (
            <StepMetricsCharts
              runs={runsToDisplay}
              shouldPoll={shouldPoll}
              totalSteps={totalSteps}
              showEma={showEma}
              emaSpan={emaSpan}
              hoveredRunId={hoveredRunId}
              availableRewardNames={availableRewardNames}
              customMetricSections={customMetricSections}
              xAxisMode={xAxisMode}
              scrollRoot={scrollRoot}
              maxStep={
                maxLimitEnabled && xAxisMode === "step" ? maxStep : undefined
              }
              maxTime={
                maxLimitEnabled && xAxisMode === "time" ? maxTime : undefined
              }
              evalsList={evalsList}
              availableSampleTags={availableSampleTags}
              availableEnvs={availableEnvs}
            />
          ) : (
            <CustomMetricsView
              runs={runsToDisplay}
              shouldPoll={shouldPoll}
              showEma={showEma}
              emaSpan={emaSpan}
              hoveredRunId={hoveredRunId}
              availableRewardNames={availableRewardNames}
              customMetricSections={customMetricSections}
              xAxisMode={xAxisMode}
              scrollRoot={scrollRoot}
              maxStep={
                maxLimitEnabled && xAxisMode === "step" ? maxStep : undefined
              }
              maxTime={
                maxLimitEnabled && xAxisMode === "time" ? maxTime : undefined
              }
              evalsList={evalsList}
              availableSampleTags={availableSampleTags}
              availableEnvs={availableEnvs}
              newSectionTrigger={newSectionTrigger}
              activeTemplateId={activeTemplateId}
              layoutSnapshotTrigger={layoutSnapshotTrigger}
              onLayoutSnapshot={(layout) => {
                if (layoutSnapshotCallbackRef.current) {
                  layoutSnapshotCallbackRef.current(layout)
                }
              }}
            />
          )}
        </div>
      </div>

      {/* Load Template Dialog */}
      <Dialog open={loadTemplateOpen} onOpenChange={setLoadTemplateOpen}>
        <DialogContent className="sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle>Load Template</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-1 max-h-[300px] overflow-y-auto">
            <button
              className={cn(
                "text-left px-3 py-2 rounded-md text-sm hover:bg-accent transition-colors",
                activeTemplateId === null && "bg-accent font-medium",
              )}
              onClick={() => handleLoadTemplate(null)}
            >
              Default
            </button>
            {templates.map((t) => (
              <button
                key={t.id}
                className={cn(
                  "text-left px-3 py-2 rounded-md text-sm hover:bg-accent transition-colors",
                  activeTemplateId === t.id && "bg-accent font-medium",
                )}
                onClick={() => handleLoadTemplate(t)}
              >
                {t.name}
              </button>
            ))}
            {templates.length === 0 && (
              <p className="text-xs text-muted-foreground px-3 py-2">
                No saved templates yet.
              </p>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Name Dialog (New / Duplicate / Rename) */}
      <Dialog open={nameDialogOpen} onOpenChange={setNameDialogOpen}>
        <DialogContent className="sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle>
              {nameDialogMode === "new"
                ? "New Template"
                : nameDialogMode === "duplicate"
                  ? "Duplicate Template"
                  : "Rename Template"}
            </DialogTitle>
          </DialogHeader>
          <Input
            placeholder="Template name"
            value={nameDialogValue}
            onChange={(e) => setNameDialogValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && nameDialogValue.trim()) {
                handleNameDialogSubmit()
              }
            }}
            autoFocus
          />
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setNameDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleNameDialogSubmit}
              disabled={!nameDialogValue.trim()}
            >
              {nameDialogMode === "rename" ? "Rename" : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Template Confirmation */}
      <AlertDialog open={deleteTemplateOpen} onOpenChange={setDeleteTemplateOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Template</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &ldquo;{activeTemplateName}&rdquo;? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteTemplate}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
