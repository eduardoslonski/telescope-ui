
import { useState, useMemo, useEffect, useRef } from "react"
import { formatDurationHms } from "@/lib/format"
import { useAtom, useAtomValue } from "jotai"
import { Plus } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { NoRunSelectedState } from "@/components/no-run-selected-state"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { ToggleWithInput } from "@/components/ui/toggle-with-input"
import { Button } from "@/components/ui/button"
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
  hoveredRunIdAtom,
} from "@/lib/atoms"
import { useRunSummary, useRuns, useCustomMetricsLayout } from "@/hooks/use-run-data"

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

  // Get summary to know if we have step metrics data
  const { data: summaryData, error } = useRunSummary(
    selectedRunPath || "",
    !!selectedRunPath,
    shouldPoll,
  )

  const stepMetricsInfo = summaryData?.step_metrics_info
  const totalSteps = stepMetricsInfo?.local_steps ?? 0
  const availableRewardNames = summaryData?.available_rollout_metric_names ?? []
  const customMetricSections = stepMetricsInfo?.custom_metric_sections ?? {}
  const evalsList = summaryData?.eval_info?.evals ?? []
  const availableSampleTags = summaryData?.available_sample_tags ?? {}

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
              <Button
                variant="outline"
                size="sm"
                className="h-7 text-xs px-2"
                onClick={() => setNewSectionTrigger((n) => n + 1)}
              >
                <Plus className="h-3 w-3 mr-1" />
                New Section
              </Button>
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
              newSectionTrigger={newSectionTrigger}
            />
          )}
        </div>
      </div>
    </div>
  )
}
