
import { useCallback, useEffect, useMemo, useState } from "react"
import { useAtom, useAtomValue, useSetAtom } from "jotai"
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core"
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  rectSortingStrategy,
} from "@dnd-kit/sortable"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Toggle } from "@/components/ui/toggle"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { ToggleWithInput } from "@/components/ui/toggle-with-input"
import { RunConfigPanel } from "@/components/run-config-panel"
import { RunConfigCompareDialog } from "@/components/run-config-compare-dialog"
import { RunCodeVisualizer } from "@/components/run-code-visualizer"
import { LogsViewer } from "@/components/logs-viewer"
import { NoRunSelectedState } from "@/components/no-run-selected-state"
import { RunInfo } from "@/components/step-metrics-charts"
import {
  PlotSelectPopover,
  SortablePlotCard,
  buildPlotCatalog,
} from "@/components/custom-metrics-view"
import {
  selectedRunPathAtom,
  visibleRunsAtom,
  isTrackingAtom,
  isSyncingAtom,
  hoveredRunIdAtom,
  overviewShowCodeViewAtom,
  overviewShowLogsViewAtom,
  overviewPlotsAtom,
  overviewShowEmaAtom,
  overviewEmaSpanAtom,
  overviewShowAllRunsAtom,
} from "@/lib/atoms"
import {
  useRunSummary,
  useRuns,
  useStepMetricSingle,
} from "@/hooks/use-run-data"

const NUM_BARS = 100

function ProgressBars({
  current,
  total,
  fillClassName,
  activeBorderClassName,
}: {
  current: number
  total: number
  fillClassName: string
  activeBorderClassName?: string
}) {
  const stepsPerBar = total / NUM_BARS

  return (
    <div className="flex gap-[2px]">
      {Array.from({ length: NUM_BARS }).map((_, i) => {
        const barStart = i * stepsPerBar
        const barEnd = (i + 1) * stepsPerBar
        let fillFraction = 0
        if (current >= barEnd) {
          fillFraction = 1
        } else if (current > barStart) {
          fillFraction = (current - barStart) / stepsPerBar
        }

        const borderClass =
          fillFraction > 0 && activeBorderClassName
            ? activeBorderClassName
            : "border-muted-foreground/50"

        return (
          <div
            key={i}
            className={`relative flex-1 h-8 border ${borderClass} rounded-[2px] overflow-hidden bg-background`}
          >
            {fillFraction > 0 && (
              <div
                className={`absolute top-0 left-0 bottom-0 transition-all duration-200 ${fillClassName}`}
                style={{ width: `${fillFraction * 100}%` }}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

function formatDuration(totalSeconds: number): string {
  const days = Math.floor(totalSeconds / 86400)
  const hours = Math.floor((totalSeconds % 86400) / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = Math.floor(totalSeconds % 60)

  if (days > 0) {
    return `${days}d ${hours}h ${minutes}m`
  }
  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`
  }
  return `${seconds}s`
}

function useElapsedTime(
  createdAt: string | null,
  isRunning: boolean,
): number | null {
  const [nowMs, setNowMs] = useState<number | null>(null)

  useEffect(() => {
    if (!createdAt || !isRunning) return

    const updateNow = () => setNowMs(Date.now())
    const kickoffTimeout = setTimeout(updateNow, 0)
    const interval = setInterval(() => {
      setNowMs(Date.now())
    }, 1000)
    return () => {
      clearTimeout(kickoffTimeout)
      clearInterval(interval)
    }
  }, [createdAt, isRunning])

  if (!createdAt || !isRunning || nowMs === null) return null

  const startTime = new Date(createdAt).getTime()
  if (isNaN(startTime)) return null

  return (nowMs - startTime) / 1000
}

function formatCreatedAt(dateStr: string): string {
  const d = new Date(dateStr)
  if (isNaN(d.getTime())) return dateStr
  const mm = String(d.getMonth() + 1).padStart(2, "0")
  const dd = String(d.getDate()).padStart(2, "0")
  const yyyy = d.getFullYear()
  const hh = String(d.getHours()).padStart(2, "0")
  const min = String(d.getMinutes()).padStart(2, "0")
  const ss = String(d.getSeconds()).padStart(2, "0")
  return `${mm}/${dd}/${yyyy} ${hh}:${min}:${ss}`
}

function extractConfigSummary(config: Record<string, unknown>): {
  model: string | null
  envNames: string[]
} {
  // Extract raw value (handles wandb {value: ...} wrapper)
  const unwrap = (v: unknown): unknown =>
    v && typeof v === "object" && "value" in v
      ? (v as { value: unknown }).value
      : v

  const rawModel = unwrap(config.model)
  const model = typeof rawModel === "string" ? rawModel : null

  const rawEnvs = unwrap(config.environments)
  let envNames: string[] = []
  let parsed = rawEnvs
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed)
    } catch {
      /* keep as-is */
    }
  }
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    envNames = Object.keys(parsed as Record<string, unknown>)
  } else if (Array.isArray(parsed)) {
    envNames = parsed.map((e) =>
      typeof e === "string"
        ? e
        : e && typeof e === "object" && "name" in e
          ? String((e as { name: unknown }).name)
          : String(e)
    )
  }

  return { model, envNames }
}

function parseNumeric(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string") {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

/**
 * Estimate remaining training time using exponentially weighted average
 * of recent step durations. Recent steps are weighted more heavily since
 * they better predict future step times (as batch composition, model state, etc. evolve).
 *
 * Uses a half-life of 10 steps: a step 10 ago gets half the weight of the most recent.
 */
function estimateTimeToFinish(
  stepTimings: { step: number; value: number }[],
  currentStep: number,
  totalSteps: number,
): number | null {
  if (stepTimings.length === 0 || currentStep >= totalSteps) return null

  // Sort by step ascending
  const sorted = [...stepTimings].sort((a, b) => a.step - b.step)

  // Use up to the last 50 steps for estimation
  const recentSteps = sorted.slice(-50)
  if (recentSteps.length === 0) return null

  const halfLife = 10
  const decay = Math.pow(0.5, 1 / halfLife)
  const n = recentSteps.length

  let weightedSum = 0
  let weightSum = 0
  for (let i = 0; i < n; i++) {
    // i=0 is the oldest in the window, i=n-1 is the most recent
    const weight = Math.pow(decay, n - 1 - i)
    weightedSum += recentSteps[i].value * weight
    weightSum += weight
  }

  const avgTimePerStep = weightedSum / weightSum
  const remainingSteps = totalSteps - currentStep
  return avgTimePerStep * remainingSteps
}

function TrainingProgressSection({
  currentStep,
  totalSteps,
  runtimeSeconds,
  estimatedSecondsToFinish,
}: {
  currentStep: number
  totalSteps: number
  runtimeSeconds: number | null
  estimatedSecondsToFinish: number | null
}) {
  // Steps are 0-indexed, so completing step 0 means 1 step done
  const completedSteps = currentStep >= 0 ? currentStep + 1 : 0
  const percentage = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0
  const clampedPercentage = Math.min(percentage, 100)

  return (
    <div className="space-y-1.5">
      <div className="text-xs font-medium text-muted-foreground">
        Training Progress
      </div>
      <div className="flex items-baseline gap-1.5 flex-wrap">
        <span className="text-sm font-bold tabular-nums">
          {clampedPercentage.toFixed(2)}%
        </span>
        <span className="text-sm text-muted-foreground tabular-nums">
          {completedSteps} / {totalSteps} Steps
        </span>
        {runtimeSeconds !== null && (
          <>
            <span className="text-sm text-muted-foreground">·</span>
            <span className="text-sm text-muted-foreground tabular-nums">
              {formatDuration(runtimeSeconds)}
            </span>
          </>
        )}
        {estimatedSecondsToFinish !== null && estimatedSecondsToFinish > 0 && (
          <>
            <span className="text-sm text-muted-foreground">·</span>
            <span className="text-sm text-muted-foreground tabular-nums">
              ETA: {formatDuration(estimatedSecondsToFinish)}
            </span>
          </>
        )}
      </div>
      <ProgressBars
        current={completedSteps}
        total={totalSteps}
        fillClassName="bg-foreground"
        activeBorderClassName="border-foreground"
      />
    </div>
  )
}

function TrainerBucketSection({
  groupsDone,
  totalGroups,
  step,
  waitingBuckets,
}: {
  groupsDone: number
  totalGroups: number
  step: number | null
  waitingBuckets?: number[]
}) {
  const percentage = totalGroups > 0 ? (groupsDone / totalGroups) * 100 : 0
  const clampedPercentage = Math.min(percentage, 100)
  const waitingCount = waitingBuckets?.length ?? 0

  return (
    <div className="space-y-1.5">
      <div className="text-xs font-medium text-muted-foreground">
        Trainer Bucket{step !== null ? ` (Step ${step})` : ""}
        {waitingCount > 0 && (
          <span className="ml-2 font-normal">
            · {waitingCount} batch{waitingCount !== 1 ? "es" : ""} waiting for
            trainer
            {" · "}
            <span className="tabular-nums">
              steps {waitingBuckets!.join(", ")}
            </span>
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="text-sm font-bold tabular-nums">
          {clampedPercentage.toFixed(2)}%
        </span>
        <span className="text-sm text-muted-foreground tabular-nums">
          {groupsDone} / {totalGroups} Groups
        </span>
      </div>
      <ProgressBars
        current={groupsDone}
        total={totalGroups}
        fillClassName="bg-muted-foreground/40"
      />
    </div>
  )
}

export default function HomePage() {
  const selectedRunPath = useAtomValue(selectedRunPathAtom)
  const visibleRuns = useAtomValue(visibleRunsAtom)
  const hoveredRunIdRaw = useAtomValue(hoveredRunIdAtom)
  const setIsTracking = useSetAtom(isTrackingAtom)
  const setIsSyncingLocal = useSetAtom(isSyncingAtom)

  // Plots & config state
  const [plotsOpen, setPlotsOpen] = useState(true)
  const [configsOpen, setConfigsOpen] = useState(true)
  const [configViewMode, setConfigViewMode] = useState<"custom" | "all">("custom")
  const [collapseAllSignal, setCollapseAllSignal] = useState(0)
  const [expandAllSignal, setExpandAllSignal] = useState(0)
  const [showEma, setShowEma] = useAtom(overviewShowEmaAtom)
  const [emaSpan, setEmaSpan] = useAtom(overviewEmaSpanAtom)
  const [emaSpanInput, setEmaSpanInput] = useState(String(emaSpan))
  const [showAllRuns, setShowAllRuns] = useAtom(overviewShowAllRunsAtom)
  const [showCodeView, setShowCodeView] = useAtom(overviewShowCodeViewAtom)
  const [showLogsView, setShowLogsView] = useAtom(overviewShowLogsViewAtom)
  const [overviewPlots, setOverviewPlots] = useAtom(overviewPlotsAtom)

  // Only highlight hovered run when all selected runs are shown
  const hoveredRunId = showAllRuns ? hoveredRunIdRaw : null

  const {
    data: summaryData,
    isLoading: isLoadingSummary,
    error: summaryError,
  } = useRunSummary(
    selectedRunPath || "",
    !!selectedRunPath,
    true,
  )

  // Get all runs for colors
  const { data: runsData } = useRuns()

  // Build runs to display (same logic as Metrics)
  const runsToDisplay: RunInfo[] = useMemo(() => {
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

  // Update local state from server
  useEffect(() => {
    if (summaryData) {
      setIsTracking(summaryData.is_tracking)
      setIsSyncingLocal(summaryData.is_syncing)
    }
  }, [summaryData, setIsTracking, setIsSyncingLocal])

  // Extract config values
  const numberOfSteps =
    typeof summaryData?.config?.number_of_steps === "number"
      ? summaryData.config.number_of_steps
      : null
  const batchSizeForTrainer =
    typeof summaryData?.config?.prompts_batch_size_for_trainer === "number"
      ? summaryData.config.prompts_batch_size_for_trainer
      : typeof summaryData?.config?.batch_size_for_trainer === "number"
        ? summaryData.config.batch_size_for_trainer
        : null
  const summarySteps =
    summaryData?.summary?.steps && typeof summaryData.summary.steps === "object"
      ? (summaryData.summary.steps as Record<string, unknown>)
      : null
  const lastTrainingStep =
    parseNumeric(summaryData?.summary?.["steps/last_training_step"]) ??
    parseNumeric(summarySteps?.last_training_step) ??
    parseNumeric(summaryData?.trainer_info?.last_training_step) ??
    parseNumeric(summaryData?.rollout_info?.last_training_step) ??
    parseNumeric(summaryData?.last_rollout_step) ??
    -1
  const trainerBucket = summaryData?.trainer_bucket_info
  const totalSteps = summaryData?.step_metrics_info?.local_steps ?? 0
  const customMetricSections =
    summaryData?.step_metrics_info?.custom_metric_sections ?? {}
  const availableRewardNames = summaryData?.available_rollout_metric_names ?? []
  const evalsList = summaryData?.eval_info?.evals ?? []

  const plotCatalog = useMemo(
    () => buildPlotCatalog(customMetricSections, availableRewardNames, evalsList),
    [customMetricSections, availableRewardNames, evalsList],
  )

  // Runs for plots: either just the selected run or all visible runs
  const plotRuns: RunInfo[] = useMemo(() => {
    if (showAllRuns) return runsToDisplay
    const selected = runsToDisplay.find((r) => r.isSelected)
    return selected ? [selected] : runsToDisplay.slice(0, 1)
  }, [showAllRuns, runsToDisplay])

  const plotSensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  )

  const handlePlotDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over } = event
      if (!over || active.id === over.id) return
      setOverviewPlots((prev) => {
        const oldIdx = prev.findIndex((p) => p.id === active.id)
        const newIdx = prev.findIndex((p) => p.id === over.id)
        if (oldIdx === -1 || newIdx === -1) return prev
        return arrayMove(prev, oldIdx, newIdx)
      })
    },
    [setOverviewPlots],
  )

  const availableSampleTags = summaryData?.available_sample_tags ?? {}
  const chartProps = useMemo(
    () => ({
      runs: plotRuns,
      shouldPoll: true,
      showEma,
      emaSpan,
      hoveredRunId,
      xAxisMode: "step" as const,
      scrollRoot: null,
      availableSampleTags,
    }),
    [plotRuns, showEma, emaSpan, hoveredRunId, availableSampleTags],
  )

  // Find current run info for created_at and state
  const currentRun = useMemo(() => {
    if (!selectedRunPath || !runsData?.runs) return null
    return runsData.runs.find((r) => r.run_id === selectedRunPath) ?? null
  }, [selectedRunPath, runsData])

  const isRunRunning = currentRun?.state?.toLowerCase() === "running"

  // For running runs, compute elapsed time from created_at
  const elapsedTime = useElapsedTime(
    currentRun?.created_at ?? null,
    isRunRunning,
  )

  // For finished runs, use _runtime from wandb summary
  const summaryRuntime = useMemo(() => {
    if (isRunRunning) return null
    const rt = summaryData?.summary?._runtime
    if (typeof rt === "number") return rt
    const wandbRt = (summaryData?.summary?._wandb as Record<string, unknown>)
      ?.runtime
    if (typeof wandbRt === "number") return wandbRt
    return null
  }, [isRunRunning, summaryData?.summary])

  const runtimeSeconds = isRunRunning ? elapsedTime : summaryRuntime

  // Fetch timing_step_total for ETA estimation
  const { data: timingData } = useStepMetricSingle(
    selectedRunPath || "",
    "timing_step_total",
    !!selectedRunPath && !!numberOfSteps && lastTrainingStep > 0,
    true,
  )

  // Compute estimated time to finish (exclude first step — it's typically
  // much slower due to compilation / warmup and would skew the ETA)
  const estimatedSecondsToFinish = useMemo(() => {
    if (!timingData?.metrics || !numberOfSteps || lastTrainingStep <= 0)
      return null
    const allTimings = timingData.metrics
      .filter((m) => m.metric_name === "timing_step_total")
      .map((m) => ({ step: m.step, value: m.value }))
    if (allTimings.length === 0) return null
    const firstStep = Math.min(...allTimings.map((t) => t.step))
    const stepTimings = allTimings.filter((t) => t.step !== firstStep)
    return estimateTimeToFinish(stepTimings, lastTrainingStep, numberOfSteps)
  }, [timingData, numberOfSteps, lastTrainingStep])

  // No run selected
  if (!selectedRunPath) {
    return (
      <NoRunSelectedState description="Select a run from the sidebar or add a new one to get started." />
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="py-2.5 px-6">
          <div className="max-w-7xl mx-auto flex items-center gap-2.5 h-7">
            {currentRun && (
              <>
                <div
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: currentRun.color }}
                />
                <span className="text-sm font-semibold truncate">
                  {currentRun.name || currentRun.run_id.split("/").pop()}
                </span>
                <span className="text-sm text-muted-foreground truncate">
                  {currentRun.run_id}
                </span>
                {currentRun.created_at && (
                  <>
                    <span className="text-sm text-muted-foreground">·</span>
                    <span className="text-sm text-muted-foreground tabular-nums whitespace-nowrap">
                      Created at {formatCreatedAt(currentRun.created_at)}
                    </span>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </header>

      {/* Content */}
      <div
        className={
          showCodeView || showLogsView
            ? "flex-1 min-h-0 overflow-hidden"
            : "flex-1 overflow-auto p-6"
        }
      >
        {showCodeView ? (
          <RunCodeVisualizer
            key={selectedRunPath}
            runPath={selectedRunPath}
            onBack={() => setShowCodeView(false)}
          />
        ) : showLogsView ? (
          <LogsViewer
            key={`logs-${selectedRunPath}`}
            runPath={selectedRunPath}
            onBack={() => setShowLogsView(false)}
          />
        ) : (
          <div className="max-w-7xl mx-auto space-y-6">
            {/* Progress Sections */}
            {isLoadingSummary ? (
              <div className="space-y-3">
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-8 w-64" />
                <Skeleton className="h-8 w-full" />
              </div>
            ) : summaryError ? (
              <Card className="border-destructive">
                <CardContent className="pt-6">
                  <p className="text-destructive">
                    Failed to load run summary. Make sure the run has been
                    synced.
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-5">
                {numberOfSteps !== null ? (
                  <TrainingProgressSection
                    currentStep={lastTrainingStep}
                    totalSteps={numberOfSteps}
                    runtimeSeconds={runtimeSeconds ?? null}
                    estimatedSecondsToFinish={estimatedSecondsToFinish}
                  />
                ) : (
                  <div className="space-y-1.5">
                    <div className="text-xs font-medium text-muted-foreground">
                      Training Progress
                    </div>
                    <div className="flex items-baseline gap-1.5">
                      <span className="text-sm text-muted-foreground">
                        Waiting for data…
                      </span>
                    </div>
                  </div>
                )}
                {batchSizeForTrainer !== null && trainerBucket && (
                  <TrainerBucketSection
                    groupsDone={trainerBucket.groups_done}
                    totalGroups={batchSizeForTrainer}
                    step={trainerBucket.step}
                    waitingBuckets={summaryData?.waiting_buckets}
                  />
                )}
              </div>
            )}

            {/* Plots Section */}
            {totalSteps > 0 && runsToDisplay.length > 0 && (
              <div>
                {plotsOpen ? (
                  <>
                    {/* Expanded: Hide button + EMA toggle */}
                    <div className="flex items-center gap-2 mb-3">
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => setPlotsOpen(false)}
                      >
                        Hide Plots
                      </Button>
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
                      <Toggle
                        variant="selecting"
                        size="sm"
                        pressed={showAllRuns}
                        onPressedChange={setShowAllRuns}
                        className="text-xs px-2"
                      >
                        All selected runs
                      </Toggle>
                      <PlotSelectPopover
                        catalog={plotCatalog}
                        existingPlots={overviewPlots}
                        onSelect={(item) => {
                          const id = Math.random().toString(36).slice(2, 10)
                          setOverviewPlots((prev) => [
                            ...prev,
                            {
                              id,
                              metricKey: item.metricKey,
                              label: item.label,
                              plotType: item.plotType,
                              evalName: item.evalName,
                              distMetricType: item.distMetricType,
                              inferenceMetricType: item.inferenceMetricType,
                            },
                          ])
                        }}
                      >
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2 text-xs"
                        >
                          Plots
                        </Button>
                      </PlotSelectPopover>
                    </div>
                    <DndContext
                      sensors={plotSensors}
                      collisionDetection={closestCenter}
                      onDragEnd={handlePlotDragEnd}
                    >
                      <SortableContext
                        items={overviewPlots.map((p) => p.id)}
                        strategy={rectSortingStrategy}
                      >
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {overviewPlots.map((plot) => (
                            <SortablePlotCard
                              key={plot.id}
                              plot={plot}
                              onRemove={() =>
                                setOverviewPlots((prev) =>
                                  prev.filter((p) => p.id !== plot.id)
                                )
                              }
                              chartProps={chartProps}
                            />
                          ))}
                        </div>
                      </SortableContext>
                    </DndContext>
                  </>
                ) : (
                  /* Collapsed: just the header */
                  <button
                    onClick={() => setPlotsOpen(true)}
                    className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors flex items-center gap-1.5"
                  >
                    <svg
                      className="h-4 w-4 text-muted-foreground -rotate-90"
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="m6 9 6 6 6-6" />
                    </svg>
                    <span className="text-sm font-semibold">Plots</span>
                  </button>
                )}
              </div>
            )}

            {/* Config Summary (model + environments) */}
            {summaryData?.config && (() => {
              const { model, envNames } = extractConfigSummary(summaryData.config)
              if (!model && envNames.length === 0) return null
              return (
                <div className="flex items-center gap-3 text-sm mb-4">
                  {model && (
                    <span className="font-semibold">{model}</span>
                  )}
                  {envNames.map((name) => (
                    <span
                      key={name}
                      className="text-muted-foreground"
                    >
                      {name}
                    </span>
                  ))}
                </div>
              )
            })()}

            {/* Config Info */}
            {summaryData?.config &&
              Object.keys(summaryData.config).length > 0 && (
                <div>
                  {configsOpen ? (
                    <>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-semibold">
                          Run Configuration
                        </span>
                        {summaryData.custom_config && (
                          <ToggleGroup
                            type="single"
                            value={configViewMode}
                            onValueChange={(value) => {
                              if (value) setConfigViewMode(value as "custom" | "all")
                            }}
                            variant="outline"
                            size="sm"
                          >
                            <ToggleGroupItem value="custom" className="text-xs px-3 h-7">
                              Custom
                            </ToggleGroupItem>
                            <ToggleGroupItem value="all" className="text-xs px-3 h-7">
                              All
                            </ToggleGroupItem>
                          </ToggleGroup>
                        )}
                        <RunConfigCompareDialog
                          currentRunId={selectedRunPath}
                          currentConfig={summaryData.config}
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2 text-xs"
                          onClick={() => {
                            setCollapseAllSignal((s) => s + 1)
                          }}
                        >
                          Collapse All
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2 text-xs"
                          onClick={() => {
                            setExpandAllSignal((s) => s + 1)
                          }}
                        >
                          Expand All
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2 text-xs"
                          onClick={() => setShowCodeView(true)}
                        >
                          Code
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2 text-xs"
                          onClick={() => setShowLogsView(true)}
                        >
                          Logs
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2 text-xs"
                          onClick={() => setConfigsOpen(false)}
                        >
                          Hide Configs
                        </Button>
                      </div>
                      <RunConfigPanel
                        config={
                          configViewMode === "custom" && summaryData.custom_config
                            ? summaryData.custom_config
                            : summaryData.config
                        }
                        collapseAllSignal={collapseAllSignal}
                        expandAllSignal={expandAllSignal}
                      />
                    </>
                  ) : (
                    <button
                      onClick={() => setConfigsOpen(true)}
                      className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors flex items-center gap-1.5"
                    >
                      <svg
                        className="h-4 w-4 text-muted-foreground -rotate-90"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="m6 9 6 6 6-6" />
                      </svg>
                      <span className="text-sm font-semibold">
                        Run Configuration
                      </span>
                    </button>
                  )}
                </div>
              )}
          </div>
        )}
      </div>
    </div>
  )
}
