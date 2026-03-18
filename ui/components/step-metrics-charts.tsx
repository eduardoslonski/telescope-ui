
import {
  useMemo,
  useState,
  useRef,
  useEffect,
  useCallback,
  RefObject,
} from "react"
import { useAtom, useAtomValue } from "jotai"
import uPlot from "uplot"
import { darkModeAtom, metricsChartFiltersAtom, syncedCursorAtom, type MetricsChartFilterState } from "@/lib/atoms"
import {
  getSidebarRunNameParts,
  SIDEBAR_MAX_RUN_NAME_CHARS,
} from "@/lib/run-name"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ChevronDown, X, SlidersHorizontal, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  useStepMetricsMultiRun,
  useStepTimes,
  useStepDistributionOverTime,
  useEvalStepMetricsMultiRun,
  useInferencePerformance,
} from "@/hooks/use-run-data"
import {
  formatSecondsTooltipHtml,
  formatSecondsCompact,
  formatSecondsHuman,
  formatValueSmart,
  formatDurationHms,
} from "@/lib/format"

// Metrics that should have "ignore first step" enabled by default
export const DEFAULT_IGNORE_FIRST_STEP_METRICS = new Set([
  // Full Step (Total Time)
  "timing_step_total",
  "timing_step_active",
  "timing_save_batch_total",
  "timing_microbatch_count",
  "timing_forward_total",
  "timing_backward_total",
  "timing_loss_computation_total",
  "timing_compute_kl_total",
  "timing_compute_entropy_total",
  "timing_data_to_device_total",
  "timing_prepare_tensors_total",
  "timing_waiting_for_data",
  "timing_weight_sync_trainer_total",
  "timing_weight_sync_inference_total",
  // Timeline Inference
  "timing_avg_inference_time",
  "timing_avg_compute_reward_time",
  "timing_generation_normal_pct",
  "timing_generation_discarded_pct",
  "timing_generation_canceled_pct",
  "timing_generation_all_pct",
  "timing_compute_reward_normal_pct",
  "timing_compute_reward_discarded_pct",
  "timing_compute_reward_canceled_pct",
  "timing_compute_reward_all_pct",
  "timing_idle_pct",
  // Microbatch (Mean Time per Microbatch)
  "timing_forward_microbatch_mean",
  "timing_backward_microbatch_mean",
  "timing_loss_computation_microbatch_mean",
  "timing_compute_kl_microbatch_mean",
  "timing_compute_entropy_microbatch_mean",
  "timing_data_to_device_microbatch_mean",
  "timing_prepare_tensors_microbatch_mean",
])

// Filter badge component that shows X on hover
export function FilterBadge({
  label,
  onRemove,
}: {
  label: string
  onRemove: () => void
}) {
  return (
    <span
      className="group inline-flex items-center gap-0.5 px-1.5 py-0.5 text-[9px] bg-background border border-border text-muted-foreground rounded-full cursor-pointer hover:bg-muted transition-colors whitespace-nowrap shrink-0"
      onClick={(e) => {
        e.stopPropagation()
        onRemove()
      }}
    >
      {label}
      <X className="h-2.5 w-2.5 opacity-0 group-hover:opacity-100 transition-opacity" />
    </span>
  )
}

// Helper to compute IQR bounds for outlier detection
export function computeIQRBounds(
  values: number[],
): { lower: number; upper: number } | null {
  if (values.length < 4) return null

  const sorted = [...values].sort((a, b) => a - b)
  const n = sorted.length

  // Calculate Q1 and Q3 using linear interpolation
  const q1Index = (n - 1) * 0.25
  const q3Index = (n - 1) * 0.75

  const q1 =
    sorted[Math.floor(q1Index)] +
    (q1Index % 1) * (sorted[Math.ceil(q1Index)] - sorted[Math.floor(q1Index)])
  const q3 =
    sorted[Math.floor(q3Index)] +
    (q3Index % 1) * (sorted[Math.ceil(q3Index)] - sorted[Math.floor(q3Index)])

  const iqr = q3 - q1
  const multiplier = 6.0 // More conservative than standard 1.5 to only filter extreme outliers

  return {
    lower: q1 - multiplier * iqr,
    upper: q3 + multiplier * iqr,
  }
}

// Run info type for multi-run support
export interface RunInfo {
  runPath: string
  runName?: string | null
  color: string
  isSelected: boolean
}

export function getRunIdFromPath(runPath: string): string {
  return runPath.split("/").pop() || runPath
}

export function getRunDisplayName(run: Pick<RunInfo, "runPath" | "runName">): string {
  const name = run.runName?.trim()
  return name || getRunIdFromPath(run.runPath)
}

export const TOOLTIP_RUN_NAME_MAX_CHARS = SIDEBAR_MAX_RUN_NAME_CHARS - 1
export const TOOLTIP_RUN_NAME_MAX_WIDTH_CLASS = "max-w-[190px]"

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
}

export function formatTooltipRunNameHtml(name: string): string {
  const fullNameHtml = escapeHtml(name)
  const { isTruncated, prefix, suffix } = getSidebarRunNameParts(
    name,
    TOOLTIP_RUN_NAME_MAX_CHARS,
  )

  if (!isTruncated) {
    return `<span class="inline-block ${TOOLTIP_RUN_NAME_MAX_WIDTH_CLASS} truncate align-bottom" title="${fullNameHtml}">${escapeHtml(
      prefix,
    )}</span>`
  }

  return `<span class="inline-flex ${TOOLTIP_RUN_NAME_MAX_WIDTH_CLASS} min-w-0 items-center align-bottom" title="${fullNameHtml}"><span class="shrink-0">${escapeHtml(
    prefix,
  )}</span><span class="shrink-0 px-0.5 text-muted-foreground/70">...</span><span class="shrink-0">${escapeHtml(
    suffix,
  )}</span></span>`
}

export function formatRunLabelHtml(run: Pick<RunInfo, "runPath" | "runName">): string {
  const runNameHtml = formatTooltipRunNameHtml(getRunDisplayName(run))
  const runIdHtml = escapeHtml(getRunIdFromPath(run.runPath))
  return `${runNameHtml} <span class="text-muted-foreground/70">(${runIdHtml})</span>`
}

function formatTooltipElapsedTime(seconds: number): string {
  const absSeconds = Math.abs(seconds)

  if (absSeconds >= 60) {
    return formatSecondsHuman(seconds)
  }

  if (absSeconds > 0 && absSeconds < 1) {
    const ms = seconds * 1000
    const absMs = Math.abs(ms)
    if (absMs < 1) return `${parseFloat(ms.toFixed(2))}ms`
    if (absMs < 10) return `${parseFloat(ms.toFixed(1))}ms`
    return `${Math.round(ms)}ms`
  }

  return formatSecondsCompact(seconds)
}

const TIME_TOOLTIP_MAX_DISTANCE_PX = 28

function findNearestDefinedIndex(
  series: ArrayLike<number | null | undefined> | undefined,
  xValues: ArrayLike<number>,
  targetIdx: number,
  maxDistanceX?: number | null,
): number | null {
  if (
    !series ||
    series.length === 0 ||
    targetIdx < 0 ||
    targetIdx >= series.length
  ) {
    return null
  }

  const targetX = Number(xValues[targetIdx])
  const isWithinDistance = (candidateIdx: number): boolean => {
    if (maxDistanceX === null || maxDistanceX === undefined) return true
    const distance = Math.abs(Number(xValues[candidateIdx]) - targetX)
    return distance <= maxDistanceX
  }

  const targetValue = series[targetIdx]
  if (targetValue !== null && targetValue !== undefined) {
    return isWithinDistance(targetIdx) ? targetIdx : null
  }

  let left = targetIdx - 1
  let right = targetIdx + 1

  while (left >= 0 || right < series.length) {
    const leftValue = left >= 0 ? series[left] : undefined
    const rightValue = right < series.length ? series[right] : undefined

    const hasLeft = leftValue !== null && leftValue !== undefined
    const hasRight = rightValue !== null && rightValue !== undefined

    if (hasLeft && hasRight) {
      const leftDist = Math.abs(Number(xValues[left]) - targetX)
      const rightDist = Math.abs(Number(xValues[right]) - targetX)
      const nearestIdx = rightDist < leftDist ? right : left
      return isWithinDistance(nearestIdx) ? nearestIdx : null
    }
    if (hasLeft) return isWithinDistance(left) ? left : null
    if (hasRight) return isWithinDistance(right) ? right : null

    left -= 1
    right += 1
  }

  return null
}

function formatMetricLabel(metricName: string): string {
  return metricName
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ")
}

// Base metric groups configuration (static metrics)
const BASE_METRIC_GROUPS = {
  reward: {
    title: "Reward",
    prefixes: [{ prefix: "reward_sum", label: "Reward" }],
  },
  advantage: {
    title: "Advantage",
    prefixes: [{ prefix: "advantage", label: "Advantage" }],
    showZeroLine: true,
  },
  rollouts: {
    title: "Rollouts",
    prefixes: [
      { prefix: "length_prompt", label: "Tokens (Prompt)" },
      { prefix: "length_completion", label: "Tokens (Completion)" },
      { prefix: "length_sum", label: "Tokens (Total)" },
    ],
    unit: "tokens",
  },
}

// Helper to build metric groups with dynamic reward names
function buildMetricGroups(rewardNames: string[]) {
  const groups: Record<
    string,
    {
      title: string
      prefixes: Array<{ prefix: string; label: string }>
      showZeroLine?: boolean
      unit?: string
    }
  > = { ...BASE_METRIC_GROUPS }

  // Add dynamic reward prefixes as a separate "Samples Metrics" group
  if (rewardNames.length > 0) {
    const dynamicRewardPrefixes = rewardNames.map((name) => ({
      prefix: `reward_${name}`,
      label: formatRewardNameForLabel(name),
    }))

    groups.samples_metrics = {
      title: "Samples Metrics",
      prefixes: dynamicRewardPrefixes,
    }
  }

  return groups
}

function formatRewardNameForLabel(name: string): string {
  // Convert snake_case to Title Case
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ")
}

const STAT_SUFFIXES = ["mean", "std", "min", "max"] as const
const SUFFIX_LABELS: Record<string, string> = {
  mean: "Mean",
  std: "Std",
  min: "Min",
  max: "Max",
}

// Hook to detect if element is on screen
// Returns { isVisible, isVisibleSticky }
// - isVisible: true when element is currently in viewport (for immediate fetch on scroll)
// - isVisibleSticky: stays true for a short period after becoming invisible (for stable polling)
function useOnScreen<T extends Element>(
  ref: RefObject<T | null>,
  options?: IntersectionObserverInit,
) {
  const noIntersectionObserver = typeof IntersectionObserver === "undefined"
  const [isVisible, setIsVisible] = useState(noIntersectionObserver)
  const [isVisibleSticky, setIsVisibleSticky] = useState(noIntersectionObserver)
  const stickyTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const root = options?.root ?? null
  const rootMargin = options?.rootMargin ?? "0px"
  const threshold = options?.threshold ?? 0

  useEffect(() => {
    const element = ref.current
    if (!element) return
    if (typeof IntersectionObserver === "undefined") return

    const observer = new IntersectionObserver(
      ([entry]) => {
        const nowVisible = entry.isIntersecting
        setIsVisible(nowVisible)

        if (nowVisible) {
          // Immediately set sticky to true when becoming visible
          if (stickyTimeoutRef.current) {
            clearTimeout(stickyTimeoutRef.current)
            stickyTimeoutRef.current = null
          }
          setIsVisibleSticky(true)
        } else {
          // When becoming invisible, keep sticky true for 2 seconds
          // This allows polling to continue during scroll
          if (stickyTimeoutRef.current) {
            clearTimeout(stickyTimeoutRef.current)
          }
          stickyTimeoutRef.current = setTimeout(() => {
            setIsVisibleSticky(false)
            stickyTimeoutRef.current = null
          }, 2000)
        }
      },
      { root, rootMargin, threshold },
    )

    observer.observe(element)

    return () => {
      observer.disconnect()
      if (stickyTimeoutRef.current) {
        clearTimeout(stickyTimeoutRef.current)
      }
    }
  }, [ref, root, rootMargin, threshold])

  // Return combined state: use sticky for polling stability
  // isVisible is for immediate detection, isVisibleSticky keeps queries alive during scroll
  return isVisible || isVisibleSticky
}

interface StepMetricsChartsProps {
  runs: RunInfo[]
  shouldPoll: boolean
  minStep?: number | null
  maxStep?: number | null
  maxTime?: number | null
  totalSteps?: number | null
  showEma?: boolean
  emaSpan?: number
  hoveredRunId?: string | null
  availableRewardNames?: string[]
  availableSampleTags?: Record<string, string[]>
  availableEnvs?: string[]
  customMetricSections?: Record<string, Record<string, string[]>>
  xAxisMode?: "step" | "time"
  scrollRoot?: Element | null
  evalsList?: Array<{
    eval_name: string
    available_rollout_metric_names: string[]
  }>
}

export function StepMetricsCharts({
  runs,
  shouldPoll,
  maxStep,
  maxTime,
  showEma = false,
  emaSpan = 10,
  hoveredRunId = null,
  availableRewardNames = [],
  availableSampleTags,
  availableEnvs,
  customMetricSections = {},
  xAxisMode = "step",
  scrollRoot = null,
  evalsList = [],
}: StepMetricsChartsProps) {
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({})
  const [rewardOpen, setRewardOpen] = useState(true)
  const [samplesMetricsOpen, setSamplesMetricsOpen] = useState(true)
  const [advantageOpen, setAdvantageOpen] = useState(true)
  const [evalsOpen, setEvalsOpen] = useState(true)
  const [openEvalGroups, setOpenEvalGroups] = useState<Record<string, boolean>>(
    {},
  )
  const [rolloutsOpen, setRolloutsOpen] = useState(true)
  const [discardedOpen, setDiscardedOpen] = useState(true)
  const [timelineTrainerOpen, setTimelineTrainerOpen] = useState(true)
  const [timelineInferenceOpen, setTimelineInferenceOpen] = useState(true)
  const [inferencePerformanceOpen, setInferencePerformanceOpen] = useState(true)

  // Build metric groups with dynamic reward names
  const METRIC_GROUPS = useMemo(
    () => buildMetricGroups(availableRewardNames),
    [availableRewardNames],
  )

  const shouldFetchStepTimes = xAxisMode === "time"

  const stepTimes0 = useStepTimes(
    runs[0]?.runPath || "",
    shouldFetchStepTimes && !!runs[0],
    shouldPoll,
  )
  const stepTimes1 = useStepTimes(
    runs[1]?.runPath || "",
    shouldFetchStepTimes && !!runs[1],
    shouldPoll,
  )
  const stepTimes2 = useStepTimes(
    runs[2]?.runPath || "",
    shouldFetchStepTimes && !!runs[2],
    shouldPoll,
  )
  const stepTimes3 = useStepTimes(
    runs[3]?.runPath || "",
    shouldFetchStepTimes && !!runs[3],
    shouldPoll,
  )
  const stepTimes4 = useStepTimes(
    runs[4]?.runPath || "",
    shouldFetchStepTimes && !!runs[4],
    shouldPoll,
  )
  const stepTimes5 = useStepTimes(
    runs[5]?.runPath || "",
    shouldFetchStepTimes && !!runs[5],
    shouldPoll,
  )
  const stepTimes6 = useStepTimes(
    runs[6]?.runPath || "",
    shouldFetchStepTimes && !!runs[6],
    shouldPoll,
  )
  const stepTimes7 = useStepTimes(
    runs[7]?.runPath || "",
    shouldFetchStepTimes && !!runs[7],
    shouldPoll,
  )
  const stepTimes8 = useStepTimes(
    runs[8]?.runPath || "",
    shouldFetchStepTimes && !!runs[8],
    shouldPoll,
  )
  const stepTimes9 = useStepTimes(
    runs[9]?.runPath || "",
    shouldFetchStepTimes && !!runs[9],
    shouldPoll,
  )
  const stepTimes10 = useStepTimes(
    runs[10]?.runPath || "",
    shouldFetchStepTimes && !!runs[10],
    shouldPoll,
  )
  const stepTimes11 = useStepTimes(
    runs[11]?.runPath || "",
    shouldFetchStepTimes && !!runs[11],
    shouldPoll,
  )
  const stepTimes12 = useStepTimes(
    runs[12]?.runPath || "",
    shouldFetchStepTimes && !!runs[12],
    shouldPoll,
  )
  const stepTimes13 = useStepTimes(
    runs[13]?.runPath || "",
    shouldFetchStepTimes && !!runs[13],
    shouldPoll,
  )
  const stepTimes14 = useStepTimes(
    runs[14]?.runPath || "",
    shouldFetchStepTimes && !!runs[14],
    shouldPoll,
  )
  const stepTimes15 = useStepTimes(
    runs[15]?.runPath || "",
    shouldFetchStepTimes && !!runs[15],
    shouldPoll,
  )
  const stepTimes16 = useStepTimes(
    runs[16]?.runPath || "",
    shouldFetchStepTimes && !!runs[16],
    shouldPoll,
  )
  const stepTimes17 = useStepTimes(
    runs[17]?.runPath || "",
    shouldFetchStepTimes && !!runs[17],
    shouldPoll,
  )
  const stepTimes18 = useStepTimes(
    runs[18]?.runPath || "",
    shouldFetchStepTimes && !!runs[18],
    shouldPoll,
  )
  const stepTimes19 = useStepTimes(
    runs[19]?.runPath || "",
    shouldFetchStepTimes && !!runs[19],
    shouldPoll,
  )

  const { isStepTimesFetching, isStepTimesRefetching } = useMemo(() => {
    if (!shouldFetchStepTimes)
      return { isStepTimesFetching: false, isStepTimesRefetching: false }
    const queries = [
      stepTimes0,
      stepTimes1,
      stepTimes2,
      stepTimes3,
      stepTimes4,
      stepTimes5,
      stepTimes6,
      stepTimes7,
      stepTimes8,
      stepTimes9,
      stepTimes10,
      stepTimes11,
      stepTimes12,
      stepTimes13,
      stepTimes14,
      stepTimes15,
      stepTimes16,
      stepTimes17,
      stepTimes18,
      stepTimes19,
    ]
    const fetching = queries.some((query) => query.isFetching)
    // Refetching = fetching queries all have data already
    const fetchingQueries = queries.filter((query) => query.isFetching)
    const refetching =
      fetching &&
      fetchingQueries.length > 0 &&
      fetchingQueries.every((query) => !!query.data)
    return { isStepTimesFetching: fetching, isStepTimesRefetching: refetching }
  }, [
    shouldFetchStepTimes,
    stepTimes0,
    stepTimes1,
    stepTimes2,
    stepTimes3,
    stepTimes4,
    stepTimes5,
    stepTimes6,
    stepTimes7,
    stepTimes8,
    stepTimes9,
    stepTimes10,
    stepTimes11,
    stepTimes12,
    stepTimes13,
    stepTimes14,
    stepTimes15,
    stepTimes16,
    stepTimes17,
    stepTimes18,
    stepTimes19,
  ])

  const { stepTimesByRun, firstStepTimesByRun } = useMemo(() => {
    if (!shouldFetchStepTimes) {
      return {
        stepTimesByRun: new Map<string, Map<number, number>>(),
        firstStepTimesByRun: new Map<string, number>(),
      }
    }

    const runQueries = [
      stepTimes0,
      stepTimes1,
      stepTimes2,
      stepTimes3,
      stepTimes4,
      stepTimes5,
      stepTimes6,
      stepTimes7,
      stepTimes8,
      stepTimes9,
      stepTimes10,
      stepTimes11,
      stepTimes12,
      stepTimes13,
      stepTimes14,
      stepTimes15,
      stepTimes16,
      stepTimes17,
      stepTimes18,
      stepTimes19,
    ]

    const timesByRun = new Map<string, Map<number, number>>()
    const firstTimesByRun = new Map<string, number>()

    runs.forEach((run, index) => {
      if (index >= 20) return
      const query = runQueries[index]
      const data = query.data
      if (
        !data ||
        data.first_step_time === null ||
        data.first_step_time === undefined
      ) {
        return
      }
      if (!data.step_times || data.step_times.length === 0) return

      const stepTimeMap = new Map<number, number>()
      data.step_times.forEach((entry) => {
        if (typeof entry.time === "number") {
          stepTimeMap.set(entry.step, entry.time)
        }
      })

      if (stepTimeMap.size > 0) {
        timesByRun.set(run.runPath, stepTimeMap)
        firstTimesByRun.set(run.runPath, data.first_step_time)
      }
    })

    return { stepTimesByRun: timesByRun, firstStepTimesByRun: firstTimesByRun }
  }, [
    runs,
    shouldFetchStepTimes,
    stepTimes0,
    stepTimes1,
    stepTimes2,
    stepTimes3,
    stepTimes4,
    stepTimes5,
    stepTimes6,
    stepTimes7,
    stepTimes8,
    stepTimes9,
    stepTimes10,
    stepTimes11,
    stepTimes12,
    stepTimes13,
    stepTimes14,
    stepTimes15,
    stepTimes16,
    stepTimes17,
    stepTimes18,
    stepTimes19,
  ])

  const axisProps = {
    xAxisMode,
    stepTimesByRun,
    firstStepTimesByRun,
    isStepTimesFetching,
    isStepTimesRefetching,
    scrollRoot,
    maxStepLimit: maxStep,
    maxTimeLimit: maxTime,
  }

  const sectionNames = useMemo(
    () => Object.keys(customMetricSections),
    [customMetricSections],
  )

  const toggleGroup = useCallback((name: string) => {
    setOpenGroups((prev) => ({ ...prev, [name]: !(prev[name] ?? true) }))
  }, [])

  const toggleEvalGroup = useCallback((name: string) => {
    setOpenEvalGroups((prev) => ({ ...prev, [name]: !(prev[name] ?? true) }))
  }, [])

  const hasEvals = evalsList.length > 0

  return (
    <div className="-mt-3">
      {/* Dynamic General Metric Sections from step_metrics table */}
      {sectionNames.map((sectionName, idx) => {
        const isOpen = openGroups[sectionName] ?? true
        const groups = customMetricSections[sectionName] ?? {}
        const groupEntries = Object.entries(groups)
        return (
          <div key={sectionName}>
            {idx > 0 && <div className="border-t border-border my-3" />}
            <Collapsible
              open={isOpen}
              onOpenChange={() => toggleGroup(sectionName)}
            >
              <CollapsibleTrigger asChild>
                <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
                  <div className="flex items-center gap-1.5">
                    <ChevronDown
                      className={cn(
                        "h-4 w-4 text-muted-foreground transition-transform",
                        !isOpen && "-rotate-90",
                      )}
                    />
                    <h3 className="text-sm font-semibold">
                      {formatMetricLabel(sectionName)}
                    </h3>
                  </div>
                </div>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="space-y-4 mt-3 mb-4">
                  {groupEntries.map(([groupName, metricNames]) => (
                    <div key={groupName}>
                      {groupName && (
                        <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                          {formatMetricLabel(groupName)}
                        </h4>
                      )}
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {metricNames.map((metricName) => (
                          <MetricChart
                            key={metricName}
                            runs={runs}
                            shouldPoll={shouldPoll}
                            metricName={metricName}
                            label={formatMetricLabel(metricName)}
                            showEma={showEma}
                            emaSpan={emaSpan}
                            hoveredRunId={hoveredRunId}
                            availableSampleTags={availableSampleTags}
                            availableEnvs={availableEnvs}
                            {...axisProps}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>
        )
      })}

      {sectionNames.length > 0 && (
        <div className="border-t border-border my-3" />
      )}

      {/* Reward Metrics Section */}
      <Collapsible open={rewardOpen} onOpenChange={setRewardOpen}>
        <CollapsibleTrigger asChild>
          <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <div className="flex items-center gap-1.5">
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  !rewardOpen && "-rotate-90",
                )}
              />
              <h3 className="text-sm font-semibold">
                {METRIC_GROUPS.reward.title}
              </h3>
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="space-y-6 mt-3 mb-4">
            {METRIC_GROUPS.reward.prefixes.map((prefixInfo) => (
              <MetricPrefixSection
                key={prefixInfo.prefix}
                runs={runs}
                shouldPoll={shouldPoll}
                prefix={prefixInfo.prefix}
                label={prefixInfo.label}
                hideGroupLabel
                showEma={showEma}
                emaSpan={emaSpan}
                hoveredRunId={hoveredRunId}
                availableSampleTags={availableSampleTags}
                availableEnvs={availableEnvs}
                {...axisProps}
                extraCharts={
                  prefixInfo.prefix === "reward_sum" ? (
                    <MetricChart
                      runs={runs}
                      shouldPoll={shouldPoll}
                      metricName="reward_gini_mean"
                      label={`${prefixInfo.label} Sparsity (Gini avg across groups)`}
                      showEma={showEma}
                      emaSpan={emaSpan}
                      hoveredRunId={hoveredRunId}
                      availableSampleTags={availableSampleTags}
                      availableEnvs={availableEnvs}
                      {...axisProps}
                    />
                  ) : undefined
                }
              />
            ))}
          </div>
        </CollapsibleContent>
      </Collapsible>

      {METRIC_GROUPS.samples_metrics && (
        <>
          <div className="border-t border-border my-3" />

          {/* Samples Metrics Section */}
          <Collapsible
            open={samplesMetricsOpen}
            onOpenChange={setSamplesMetricsOpen}
          >
            <CollapsibleTrigger asChild>
              <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
                <div className="flex items-center gap-1.5">
                  <ChevronDown
                    className={cn(
                      "h-4 w-4 text-muted-foreground transition-transform",
                      !samplesMetricsOpen && "-rotate-90",
                    )}
                  />
                  <h3 className="text-sm font-semibold">
                    {METRIC_GROUPS.samples_metrics.title}
                  </h3>
                </div>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="space-y-6 mt-3 mb-4">
                {METRIC_GROUPS.samples_metrics.prefixes.map((prefixInfo) => (
                  <MetricPrefixSection
                    key={prefixInfo.prefix}
                    runs={runs}
                    shouldPoll={shouldPoll}
                    prefix={prefixInfo.prefix}
                    label={prefixInfo.label}
                    showEma={showEma}
                    emaSpan={emaSpan}
                    hoveredRunId={hoveredRunId}
                    availableSampleTags={availableSampleTags}
                    availableEnvs={availableEnvs}
                    {...axisProps}
                    extraCharts={
                      <MetricChart
                        runs={runs}
                        shouldPoll={shouldPoll}
                        metricName={`${prefixInfo.prefix}_gini_mean`}
                        label={`${prefixInfo.label} Sparsity (Gini avg across groups)`}
                        showEma={showEma}
                        emaSpan={emaSpan}
                        hoveredRunId={hoveredRunId}
                        availableSampleTags={availableSampleTags}
                        availableEnvs={availableEnvs}
                        {...axisProps}
                      />
                    }
                  />
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        </>
      )}

      <div className="border-t border-border my-3" />

      {/* Advantage Metrics Section */}
      <Collapsible open={advantageOpen} onOpenChange={setAdvantageOpen}>
        <CollapsibleTrigger asChild>
          <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <div className="flex items-center gap-1.5">
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  !advantageOpen && "-rotate-90",
                )}
              />
              <h3 className="text-sm font-semibold">
                {METRIC_GROUPS.advantage.title}
              </h3>
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="space-y-6 mt-3 mb-4">
            {METRIC_GROUPS.advantage.prefixes.map((prefixInfo) => (
              <MetricPrefixSection
                key={prefixInfo.prefix}
                runs={runs}
                shouldPoll={shouldPoll}
                prefix={prefixInfo.prefix}
                label={prefixInfo.label}
                hideGroupLabel
                showZeroLine
                showEma={showEma}
                emaSpan={emaSpan}
                hoveredRunId={hoveredRunId}
                availableSampleTags={availableSampleTags}
                availableEnvs={availableEnvs}
                {...axisProps}
              />
            ))}
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Evals Metrics Section */}
      {hasEvals && (
        <>
          <div className="border-t border-border my-3" />

          <Collapsible open={evalsOpen} onOpenChange={setEvalsOpen}>
            <CollapsibleTrigger asChild>
              <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
                <div className="flex items-center gap-1.5">
                  <ChevronDown
                    className={cn(
                      "h-4 w-4 text-muted-foreground transition-transform",
                      !evalsOpen && "-rotate-90",
                    )}
                  />
                  <h3 className="text-sm font-semibold">Evals</h3>
                </div>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mt-3 mb-4 space-y-4">
                {evalsList.map((evalEntry, evalIdx) => {
                  const evalName = evalEntry.eval_name
                  const evalMetricNames =
                    evalEntry.available_rollout_metric_names
                  const isEvalGroupOpen = openEvalGroups[evalName] ?? true
                  return (
                    <div key={evalName}>
                      {evalIdx > 0 && (
                        <div className="border-t border-gray-100 my-3" />
                      )}
                      <Collapsible
                        open={isEvalGroupOpen}
                        onOpenChange={() => toggleEvalGroup(evalName)}
                      >
                        <CollapsibleTrigger asChild>
                          <div className="py-1 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
                            <div className="flex items-center gap-1.5">
                              <ChevronDown
                                className={cn(
                                  "h-3.5 w-3.5 text-muted-foreground transition-transform",
                                  !isEvalGroupOpen && "-rotate-90",
                                )}
                              />
                              <h4 className="text-sm font-medium text-muted-foreground">
                                {evalName}
                              </h4>
                            </div>
                          </div>
                        </CollapsibleTrigger>
                        <CollapsibleContent>
                          <div className="space-y-6 mt-3">
                            {evalMetricNames.map((metricName) => (
                              <EvalMetricPrefixSection
                                key={`${evalName}-reward_${metricName}`}
                                runs={runs}
                                shouldPoll={shouldPoll}
                                evalName={evalName}
                                prefix={`reward_${metricName}`}
                                label={formatMetricLabel(metricName)}
                                showEma={showEma}
                                emaSpan={emaSpan}
                                hoveredRunId={hoveredRunId}
                                scrollRoot={scrollRoot}
                                xAxisMode={xAxisMode}
                                stepTimesByRun={axisProps.stepTimesByRun}
                                firstStepTimesByRun={
                                  axisProps.firstStepTimesByRun
                                }
                                isStepTimesFetching={
                                  axisProps.isStepTimesFetching
                                }
                                isStepTimesRefetching={
                                  axisProps.isStepTimesRefetching
                                }
                                maxStepLimit={axisProps.maxStepLimit}
                                maxTimeLimit={axisProps.maxTimeLimit}
                                availableEnvs={availableEnvs}
                              />
                            ))}
                            <EvalMetricPrefixSection
                              runs={runs}
                              shouldPoll={shouldPoll}
                              evalName={evalName}
                              prefix="length_completion"
                              label="Tokens (Completion)"
                              unit="tokens"
                              isTokenMetric
                              showEma={showEma}
                              emaSpan={emaSpan}
                              hoveredRunId={hoveredRunId}
                              scrollRoot={scrollRoot}
                              xAxisMode={xAxisMode}
                              stepTimesByRun={axisProps.stepTimesByRun}
                              firstStepTimesByRun={
                                axisProps.firstStepTimesByRun
                              }
                              isStepTimesFetching={
                                axisProps.isStepTimesFetching
                              }
                              isStepTimesRefetching={
                                axisProps.isStepTimesRefetching
                              }
                              maxStepLimit={axisProps.maxStepLimit}
                              maxTimeLimit={axisProps.maxTimeLimit}
                            />
                            <EvalMetricPrefixSection
                              runs={runs}
                              shouldPoll={shouldPoll}
                              evalName={evalName}
                              prefix="length_prompt"
                              label="Tokens (Prompt)"
                              unit="tokens"
                              isTokenMetric
                              showEma={showEma}
                              emaSpan={emaSpan}
                              hoveredRunId={hoveredRunId}
                              scrollRoot={scrollRoot}
                              xAxisMode={xAxisMode}
                              stepTimesByRun={axisProps.stepTimesByRun}
                              firstStepTimesByRun={
                                axisProps.firstStepTimesByRun
                              }
                              isStepTimesFetching={
                                axisProps.isStepTimesFetching
                              }
                              isStepTimesRefetching={
                                axisProps.isStepTimesRefetching
                              }
                              maxStepLimit={axisProps.maxStepLimit}
                              maxTimeLimit={axisProps.maxTimeLimit}
                            />
                            <EvalMetricPrefixSection
                              runs={runs}
                              shouldPoll={shouldPoll}
                              evalName={evalName}
                              prefix="length_sum"
                              label="Tokens (Total)"
                              unit="tokens"
                              isTokenMetric
                              showEma={showEma}
                              emaSpan={emaSpan}
                              hoveredRunId={hoveredRunId}
                              scrollRoot={scrollRoot}
                              xAxisMode={xAxisMode}
                              stepTimesByRun={axisProps.stepTimesByRun}
                              firstStepTimesByRun={
                                axisProps.firstStepTimesByRun
                              }
                              isStepTimesFetching={
                                axisProps.isStepTimesFetching
                              }
                              isStepTimesRefetching={
                                axisProps.isStepTimesRefetching
                              }
                              maxStepLimit={axisProps.maxStepLimit}
                              maxTimeLimit={axisProps.maxTimeLimit}
                            />
                          </div>
                        </CollapsibleContent>
                      </Collapsible>
                    </div>
                  )
                })}
              </div>
            </CollapsibleContent>
          </Collapsible>
        </>
      )}

      <div className="border-t border-border my-3" />

      {/* Rollouts Metrics Section */}
      <Collapsible open={rolloutsOpen} onOpenChange={setRolloutsOpen}>
        <CollapsibleTrigger asChild>
          <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <div className="flex items-center gap-1.5">
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  !rolloutsOpen && "-rotate-90",
                )}
              />
              <h3 className="text-sm font-semibold">
                {METRIC_GROUPS.rollouts.title}
              </h3>
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="space-y-6 mt-3 mb-4">
            {METRIC_GROUPS.rollouts.prefixes.map((prefixInfo) => (
              <MetricPrefixSection
                key={prefixInfo.prefix}
                runs={runs}
                shouldPoll={shouldPoll}
                prefix={prefixInfo.prefix}
                label={prefixInfo.label}
                unit="tokens"
                isTokenMetric
                showEma={showEma}
                emaSpan={emaSpan}
                hoveredRunId={hoveredRunId}
                availableSampleTags={availableSampleTags}
                availableEnvs={availableEnvs}
                {...axisProps}
              />
            ))}
            {/* General Metrics */}
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                General
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="stop_reason_length_pct"
                  label="% Stop Reason = Length"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  availableSampleTags={availableSampleTags}
                  availableEnvs={availableEnvs}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="group_length_gini_mean"
                  label="Group Completion Length Gini"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  availableSampleTags={availableSampleTags}
                  availableEnvs={availableEnvs}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="group_length_max_median_ratio_mean"
                  label="Group Completion Length Max/Median"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  availableSampleTags={availableSampleTags}
                  availableEnvs={availableEnvs}
                  {...axisProps}
                />
              </div>
            </div>
            {/* Off-Policy Steps */}
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                Off-Policy Steps
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="off_policy_steps_mean"
                  label="Off-Policy Steps (Mean)"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="off_policy_steps_std"
                  label="Off-Policy Steps (Std)"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                {(() => {
                  const selectedRun = runs.find((r) => r.isSelected) ?? runs[0]
                  return selectedRun ? (
                    <DistributionOverTimeChart
                      runPath={selectedRun.runPath}
                      metricType="off_policy_steps"
                      label="Off-Policy Steps (Dist. Over Time)"
                      shouldPoll={shouldPoll}
                      scrollRoot={scrollRoot}
                    />
                  ) : null
                })()}
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>

      <div className="border-t border-border my-3" />

      {/* Discarded Rollouts Metrics Section */}
      <Collapsible open={discardedOpen} onOpenChange={setDiscardedOpen}>
        <CollapsibleTrigger asChild>
          <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <div className="flex items-center gap-1.5">
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  !discardedOpen && "-rotate-90",
                )}
              />
              <h3 className="text-sm font-semibold">Discarded Rollouts</h3>
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="space-y-6 mt-3 mb-4">
            {/* Discarded Rollouts General Stats */}
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                Discarded Rollouts General
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_count"
                  label="Discarded Count"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_zero_advantage_pct"
                  label="Zero Advantage %"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_max_async_pct"
                  label="Max Async %"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_stop_reason_length_pct"
                  label="% Stop Reason = Length"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_group_length_gini_mean"
                  label="Group Completion Length Gini"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_group_length_max_median_ratio_mean"
                  label="Group Completion Length Max/Median"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
              </div>
            </div>

            {/* Zero Advantage Breakdown */}
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                Zero Advantage Breakdown
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_zero_advantage_all_zero_pct"
                  label="Zero Advantage (All Reward = 0) %"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_zero_advantage_all_positive_pct"
                  label="Zero Advantage (All Reward > 0) %"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_zero_advantage_mean_reward"
                  label="Zero Advantage Mean Reward"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
              </div>
            </div>

            {/* Canceled */}
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                Canceled
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="canceled_count"
                  label="Canceled Count"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
              </div>
            </div>

            {/* Discarded Off-Policy Steps */}
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                Off-Policy Steps
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_off_policy_steps_mean"
                  label="Off-Policy Steps (Mean)"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="discarded_off_policy_steps_std"
                  label="Off-Policy Steps (Std)"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                {(() => {
                  const selectedRun = runs.find((r) => r.isSelected) ?? runs[0]
                  return selectedRun ? (
                    <DistributionOverTimeChart
                      runPath={selectedRun.runPath}
                      metricType="discarded_off_policy_steps"
                      label="Off-Policy Steps (Dist. Over Time)"
                      shouldPoll={shouldPoll}
                      scrollRoot={scrollRoot}
                    />
                  ) : null
                })()}
              </div>
            </div>

            {/* Discarded Token Metrics */}
            <MetricPrefixSection
              runs={runs}
              shouldPoll={shouldPoll}
              prefix="discarded_length_prompt"
              label="Discarded Tokens (Prompt)"
              unit="tokens"
              isTokenMetric
              showEma={showEma}
              emaSpan={emaSpan}
              hoveredRunId={hoveredRunId}
              {...axisProps}
            />
            <MetricPrefixSection
              runs={runs}
              shouldPoll={shouldPoll}
              prefix="discarded_length_completion"
              label="Discarded Tokens (Completion)"
              unit="tokens"
              isTokenMetric
              showEma={showEma}
              emaSpan={emaSpan}
              hoveredRunId={hoveredRunId}
              {...axisProps}
            />
            <MetricPrefixSection
              runs={runs}
              shouldPoll={shouldPoll}
              prefix="discarded_length_sum"
              label="Discarded Tokens (Total)"
              unit="tokens"
              isTokenMetric
              showEma={showEma}
              emaSpan={emaSpan}
              hoveredRunId={hoveredRunId}
              {...axisProps}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>

      <div className="border-t border-border my-3" />

      {/* Timeline Trainer Section */}
      <Collapsible open={timelineTrainerOpen} onOpenChange={setTimelineTrainerOpen}>
        <CollapsibleTrigger asChild>
          <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <div className="flex items-center gap-1.5">
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  !timelineTrainerOpen && "-rotate-90",
                )}
              />
              <h3 className="text-sm font-semibold">Timeline Trainer</h3>
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="space-y-6 mt-3 mb-4">
            {/* Full Step - Total time per operation */}
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                Full Step (Total Time)
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_step_total"
                  label="Time per Step"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_step_active"
                  label="Time per Step Active (Excl. Wait)"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_microbatch_count"
                  label="Microbatches per Step"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_forward_total"
                  label="Forward"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_backward_total"
                  label="Backward"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_loss_computation_total"
                  label="Loss Computation"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_compute_kl_total"
                  label="Compute KL"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_compute_entropy_total"
                  label="Compute Entropy"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_data_to_device_total"
                  label="Data to Device"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_prepare_tensors_total"
                  label="Prepare Tensors"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_waiting_for_data"
                  label="Waiting for Data"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_weight_sync_trainer_total"
                  label="Weight Broadcast (Trainer)"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_weight_sync_inference_total"
                  label="Weight Broadcast (Inference)"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
              </div>
            </div>

            {/* Microbatch - Mean time per microbatch */}
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                Microbatch (Mean Time per Microbatch)
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_forward_microbatch_mean"
                  label="Forward"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_backward_microbatch_mean"
                  label="Backward"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_loss_computation_microbatch_mean"
                  label="Loss Computation"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_compute_kl_microbatch_mean"
                  label="Compute KL"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_compute_entropy_microbatch_mean"
                  label="Compute Entropy"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_data_to_device_microbatch_mean"
                  label="Data to Device"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_prepare_tensors_microbatch_mean"
                  label="Prepare Tensors"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>

      <div className="border-t border-border my-3" />

      {/* Timeline Inference Section */}
      <Collapsible open={timelineInferenceOpen} onOpenChange={setTimelineInferenceOpen}>
        <CollapsibleTrigger asChild>
          <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <div className="flex items-center gap-1.5">
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  !timelineInferenceOpen && "-rotate-90",
                )}
              />
              <h3 className="text-sm font-semibold">Timeline Inference</h3>
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="space-y-6 mt-3 mb-4">
            {/* Batch & Averages */}
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                Batch & Averages
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_save_batch_total"
                  label="Batch Completion (Save Batch)"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_avg_inference_time"
                  label="Avg Generation Time"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_avg_compute_reward_time"
                  label="Avg Compute Reward Time"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
              </div>
            </div>

            {/* Time Breakdown (% of Step Time) */}
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground mb-3">
                Time Breakdown (% of Step Time)
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_generation_normal_pct"
                  label="Generation Normal"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_generation_discarded_pct"
                  label="Generation Discarded"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_generation_canceled_pct"
                  label="Generation Canceled"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_generation_all_pct"
                  label="Generation All"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_compute_reward_normal_pct"
                  label="Compute Reward Normal"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_compute_reward_discarded_pct"
                  label="Compute Reward Discarded"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_compute_reward_canceled_pct"
                  label="Compute Reward Canceled"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_compute_reward_all_pct"
                  label="Compute Reward All"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
                <MetricChart
                  runs={runs}
                  shouldPoll={shouldPoll}
                  metricName="timing_idle_pct"
                  label="Idle Time"
                  showEma={showEma}
                  emaSpan={emaSpan}
                  hoveredRunId={hoveredRunId}
                  {...axisProps}
                />
              </div>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>

      <div className="border-t border-border my-3" />

      {/* Inference Performance Section */}
      <Collapsible open={inferencePerformanceOpen} onOpenChange={setInferencePerformanceOpen}>
        <CollapsibleTrigger asChild>
          <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <div className="flex items-center gap-1.5">
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  !inferencePerformanceOpen && "-rotate-90",
                )}
              />
              <h3 className="text-sm font-semibold">Inference Performance</h3>
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-3 mb-4">
            {runs[0] && (
              <InferencePerformanceSection
                runs={runs}
                shouldPoll={shouldPoll}
                hoveredRunId={hoveredRunId}
                scrollRoot={scrollRoot}
              />
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}

// ============================================================================
// Distribution Over Time Chart (heatmap) - single run
// ============================================================================

const DIST_TARGET_COMPACT_BINS = 100

function aggregateDistributionData(
  steps: number[],
  counts: number[][],
  binEdges: number[],
): { steps: number[]; counts: number[][] } {
  const numSteps = steps.length
  if (numSteps <= DIST_TARGET_COMPACT_BINS) {
    return { steps, counts }
  }

  const stepsPerBin = Math.ceil(numSteps / DIST_TARGET_COMPACT_BINS)
  const newSteps: number[] = []
  const newCounts: number[][] = []
  const numValueBins = binEdges.length - 1

  for (let i = 0; i < numSteps; i += stepsPerBin) {
    const endIdx = Math.min(i + stepsPerBin, numSteps)
    const midIdx = Math.floor((i + endIdx - 1) / 2)
    newSteps.push(steps[midIdx])

    const aggregatedCounts = new Array(numValueBins).fill(0)
    for (let j = i; j < endIdx; j++) {
      const stepCounts = counts[j]
      for (let k = 0; k < numValueBins; k++) {
        aggregatedCounts[k] += stepCounts[k] || 0
      }
    }
    newCounts.push(aggregatedCounts)
  }

  return { steps: newSteps, counts: newCounts }
}

interface DistributionOverTimeChartProps {
  runPath: string
  metricType: string
  label: string
  showZeroLine?: boolean
  isTokenMetric?: boolean
  shouldPoll: boolean
  scrollRoot?: Element | null
  headerPrefix?: React.ReactNode
  headerSuffix?: React.ReactNode
}

export function DistributionOverTimeChart({
  runPath,
  metricType,
  label,
  showZeroLine,
  isTokenMetric,
  shouldPoll,
  scrollRoot = null,
  headerPrefix,
  headerSuffix,
}: DistributionOverTimeChartProps) {
  const darkMode = useAtomValue(darkModeAtom)
  const visibilityRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const hoverLineRef = useRef<HTMLDivElement>(null)

  const isVisible = useOnScreen(visibilityRef, {
    root: scrollRoot,
    threshold: 0,
  })

  const { data, isFetching, isRefetching } = useStepDistributionOverTime(
    runPath,
    metricType,
    isVisible && !!runPath && !!metricType,
    shouldPoll,
  )

  const hasData = data && data.steps.length > 0 && data.bin_edges.length > 0

  const isCompact = hasData && data.steps.length > DIST_TARGET_COMPACT_BINS

  const displayData = useMemo(() => {
    if (!hasData || !data) return null
    if (!isCompact) {
      return { steps: data.steps, counts: data.counts }
    }
    return aggregateDistributionData(data.steps, data.counts, data.bin_edges)
  }, [hasData, data, isCompact])

  const formatValue = useCallback(
    (v: number | null | undefined): string => {
      if (v === undefined || v === null) return "N/A"
      if (isTokenMetric) {
        return Math.round(v).toLocaleString()
      }
      return formatValueSmart(v)
    },
    [isTokenMetric],
  )

  const formatYAxisLabel = useCallback(
    (value: number) => {
      if (Math.abs(value) >= 1000)
        return `${parseFloat((value / 1000).toFixed(1))}k`
      if (isTokenMetric) return Math.round(value).toString()
      if (Number.isInteger(value)) return value.toLocaleString()
      if (Math.abs(value) < 0.01 && value !== 0) return value.toExponential(1)
      return String(parseFloat(value.toFixed(2)))
    },
    [isTokenMetric],
  )

  const getDistributionPadding = useCallback(() => {
    const defaultPadding = { top: 8, right: 8, bottom: 24, left: 40 }
    if (!canvasRef.current || !data) return defaultPadding
    const ctx = canvasRef.current.getContext("2d")
    if (!ctx) return defaultPadding

    const font = "10px system-ui, sans-serif"
    const yLabelCount = 5
    let maxLabelWidth = 0

    ctx.save()
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.font = font

    for (let i = 0; i < yLabelCount; i++) {
      const value =
        (data.global_min ?? 0) +
        (i / (yLabelCount - 1)) *
          ((data.global_max ?? 0) - (data.global_min ?? 0))
      const labelText = formatYAxisLabel(value)
      maxLabelWidth = Math.max(maxLabelWidth, ctx.measureText(labelText).width)
    }

    ctx.restore()

    return {
      top: 8,
      right: 8,
      bottom: 24,
      left: Math.max(36, Math.ceil(maxLabelWidth) + 12),
    }
  }, [data, formatYAxisLabel])

  // Draw the heatmap
  useEffect(() => {
    if (
      !containerRef.current ||
      !canvasRef.current ||
      !hasData ||
      !data ||
      !displayData
    )
      return

    const container = containerRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const width = container.clientWidth
    const height = 200
    const tickLabelColor = darkMode ? "rgba(255, 255, 255, 0.65)" : "rgba(100, 100, 100, 0.9)"
    const gridColor = darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(128, 128, 128, 0.15)"
    const axisFont = "10px system-ui, sans-serif"

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    ctx.scale(dpr, dpr)
    ctx.font = axisFont

    ctx.clearRect(0, 0, width, height)

    const padding = getDistributionPadding()
    const chartWidth = width - padding.left - padding.right
    const chartHeight = height - padding.top - padding.bottom

    const steps = displayData.steps
    const binEdges = data.bin_edges
    const counts = displayData.counts
    const numSteps = steps.length
    const numBins = binEdges.length - 1

    if (numSteps === 0 || numBins === 0) return

    let maxCount = 0
    for (const stepCounts of counts) {
      for (const count of stepCounts) {
        if (count > maxCount) maxCount = count
      }
    }

    const xLabelCount = Math.min(5, numSteps)
    const yLabelCount = 5

    const bandWidth = chartWidth / Math.max(numSteps, 1)
    const xScale = (stepIdx: number) =>
      padding.left + (stepIdx + 0.5) * bandWidth
    const yScale = (value: number) => {
      const range = (data.global_max ?? 1) - (data.global_min ?? 0)
      const normalized = (value - (data.global_min ?? 0)) / (range || 1)
      return padding.top + chartHeight - normalized * chartHeight
    }

    const getColor = (count: number) => {
      if (count === 0) return "transparent"
      const intensity = Math.sqrt(count / maxCount)
      const alpha = 0.15 + intensity * 0.75
      return `rgba(59, 130, 246, ${alpha})`
    }

    // Draw grid lines
    ctx.strokeStyle = gridColor
    ctx.lineWidth = 1

    for (let i = 0; i < xLabelCount; i++) {
      const stepIdx = Math.floor(
        (i / Math.max(xLabelCount - 1, 1)) * (numSteps - 1),
      )
      const x = xScale(stepIdx)
      ctx.beginPath()
      ctx.moveTo(x, padding.top)
      ctx.lineTo(x, height - padding.bottom)
      ctx.stroke()
    }

    for (let i = 0; i < yLabelCount; i++) {
      const value =
        (data.global_min ?? 0) +
        (i / (yLabelCount - 1)) *
          ((data.global_max ?? 0) - (data.global_min ?? 0))
      const y = yScale(value)
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(width - padding.right, y)
      ctx.stroke()
    }

    // Draw bars
    const barWidth = bandWidth

    for (let stepIdx = 0; stepIdx < numSteps; stepIdx++) {
      const x = padding.left + stepIdx * bandWidth
      const stepCounts = counts[stepIdx]

      for (let binIdx = 0; binIdx < numBins; binIdx++) {
        const count = stepCounts[binIdx]
        if (count === 0) continue

        const binTop = binEdges[binIdx + 1]
        const binBottom = binEdges[binIdx]
        const y1 = yScale(binTop)
        const y2 = yScale(binBottom)
        const barHeight = Math.max(1, y2 - y1)

        ctx.fillStyle = getColor(count)
        ctx.fillRect(x, y1, barWidth, barHeight)
      }
    }

    // Draw zero line if needed
    if (
      showZeroLine &&
      (data.global_min ?? 0) < 0 &&
      (data.global_max ?? 0) > 0
    ) {
      const zeroY = yScale(0)
      ctx.strokeStyle = darkMode ? "rgba(255, 255, 255, 0.3)" : "rgba(128, 128, 128, 0.5)"
      ctx.lineWidth = 1
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(padding.left, zeroY)
      ctx.lineTo(width - padding.right, zeroY)
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Draw axes
    ctx.strokeStyle = tickLabelColor
    ctx.lineWidth = 1

    ctx.beginPath()
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(width - padding.right, height - padding.bottom)
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(padding.left, padding.top)
    ctx.lineTo(padding.left, height - padding.bottom)
    ctx.stroke()

    // X axis labels
    ctx.fillStyle = tickLabelColor
    ctx.font = axisFont
    ctx.textAlign = "center"
    for (let i = 0; i < xLabelCount; i++) {
      const stepIdx = Math.floor(
        (i / Math.max(xLabelCount - 1, 1)) * (numSteps - 1),
      )
      const step = steps[stepIdx]
      const x = xScale(stepIdx)
      const labelText =
        step >= 1000 ? `${(step / 1000).toFixed(0)}k` : step.toString()
      ctx.fillText(labelText, x, height - 6)
    }

    // Y axis labels
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"

    for (let i = 0; i < yLabelCount; i++) {
      const value =
        (data.global_min ?? 0) +
        (i / (yLabelCount - 1)) *
          ((data.global_max ?? 0) - (data.global_min ?? 0))
      const y = yScale(value)
      ctx.fillText(formatYAxisLabel(value), padding.left - 4, y)
    }
  }, [
    data,
    displayData,
    hasData,
    showZeroLine,
    formatYAxisLabel,
    getDistributionPadding,
    darkMode,
  ])

  // Handle resize
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    const resizeObserver = new ResizeObserver(() => {
      if (canvasRef.current && containerRef.current) {
        const width = containerRef.current.clientWidth
        canvasRef.current.style.width = `${width}px`
      }
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
    }
  }, [])

  // Handle mouse move for tooltip
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (
        !data ||
        !displayData ||
        !tooltipRef.current ||
        !canvasRef.current ||
        !containerRef.current
      )
        return

      const rect = canvasRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const width = containerRef.current.clientWidth
      const height = 200
      const padding = getDistributionPadding()
      const chartWidth = width - padding.left - padding.right
      const chartHeight = height - padding.top - padding.bottom
      if (chartWidth <= 0 || chartHeight <= 0) {
        if (hoverLineRef.current) hoverLineRef.current.style.display = "none"
        return
      }

      if (
        x < padding.left ||
        x > width - padding.right ||
        y < padding.top ||
        y > height - padding.bottom
      ) {
        tooltipRef.current.style.display = "none"
        if (hoverLineRef.current) hoverLineRef.current.style.display = "none"
        return
      }

      const relativeX = (x - padding.left) / chartWidth
      const stepIdx = Math.min(
        displayData.steps.length - 1,
        Math.max(0, Math.round(relativeX * (displayData.steps.length - 1))),
      )
      const step = displayData.steps[stepIdx]

      if (hoverLineRef.current) {
        const bandWidth = chartWidth / Math.max(displayData.steps.length, 1)
        const hoverLineX = padding.left + (stepIdx + 0.5) * bandWidth
        hoverLineRef.current.style.display = "block"
        hoverLineRef.current.style.left = `${hoverLineX}px`
        hoverLineRef.current.style.top = `${padding.top}px`
        hoverLineRef.current.style.bottom = `${padding.bottom}px`
      }

      const relativeY = 1 - (y - padding.top) / chartHeight
      const numBins = data.bin_edges.length - 1
      const binIdx = Math.min(
        numBins - 1,
        Math.max(0, Math.floor(relativeY * numBins)),
      )

      const binMin = data.bin_edges[binIdx]
      const binMax = data.bin_edges[binIdx + 1]
      const count = displayData.counts[stepIdx]?.[binIdx] ?? 0
      const stepCounts = displayData.counts[stepIdx] ?? []
      const totalCount = stepCounts.reduce(
        (sum: number, c: number) => sum + c,
        0,
      )
      const percentage =
        totalCount > 0 ? ((count / totalCount) * 100).toFixed(1) : "0.0"

      const isAggregated =
        isCompact && data.steps.length > DIST_TARGET_COMPACT_BINS
      const aggregationNote = isAggregated ? " (aggregated)" : ""

      tooltipRef.current.innerHTML = `
        <div class="font-medium mb-1">Step ${step.toLocaleString()}${aggregationNote}</div>
        <div class="text-muted-foreground">Range: ${formatValue(
          binMin,
        )} to ${formatValue(binMax)}</div>
        <div class="text-muted-foreground">Count: ${count.toLocaleString()} (${percentage}%)</div>
      `
      tooltipRef.current.style.display = "block"

      tooltipRef.current.style.left = "0px"
      tooltipRef.current.style.top = "0px"
      const fixedOrigin = tooltipRef.current.getBoundingClientRect()

      const cursorViewportX = e.clientX
      let tooltipX = cursorViewportX
      if (tooltipX + fixedOrigin.width + 20 > window.innerWidth) {
        tooltipX = cursorViewportX - fixedOrigin.width
      }
      const maxX = Math.max(4, window.innerWidth - fixedOrigin.width - 4)
      tooltipX = Math.min(Math.max(4, tooltipX), maxX)

      const containerRect = containerRef.current!.getBoundingClientRect()

      let tooltipY = containerRect.top - fixedOrigin.height + 20
      if (tooltipY < 4) {
        tooltipY = containerRect.bottom - 20
      }

      tooltipRef.current.style.left = `${tooltipX - fixedOrigin.left}px`
      tooltipRef.current.style.top = `${tooltipY - fixedOrigin.top}px`
    },
    [data, displayData, formatValue, getDistributionPadding, isCompact],
  )

  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) {
      tooltipRef.current.style.display = "none"
    }
    if (hoverLineRef.current) {
      hoverLineRef.current.style.display = "none"
    }
  }, [])

  const showLoadingOpacity = isFetching && !isRefetching

  return (
    <div
      ref={visibilityRef}
      className={cn(
        "group/chart rounded-lg border border-border p-3 transition-opacity bg-background",
        showLoadingOpacity && "opacity-50",
      )}
    >
      <div className="flex items-center justify-between mb-3">
        {headerPrefix}
        <div className="flex-1 min-w-0">
          <h4
            className="text-xs font-medium leading-snug line-clamp-2 break-words"
            title={label}
          >
            {label}
          </h4>
        </div>
        {headerSuffix && (
          <div className="flex items-center gap-2 shrink-0">{headerSuffix}</div>
        )}
      </div>
      {hasData ? (
        <div className="h-[200px] relative bg-background rounded" ref={containerRef}>
          <canvas
            className="block w-full h-full max-w-full"
            ref={canvasRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />
          <div
            ref={hoverLineRef}
            className="absolute z-[2] border-l border-dashed border-gray-500/70 pointer-events-none"
            style={{ display: "none" }}
          />
          <div
            ref={tooltipRef}
            className="fixed z-[9999] max-w-[360px] bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl border border-border pointer-events-none"
            style={{ display: "none" }}
          />
          {isRefetching && (
            <Loader2 className="absolute bottom-0.5 left-0.5 h-3 w-3 animate-spin text-muted-foreground" />
          )}
        </div>
      ) : (
        <div className="h-[200px] flex items-center justify-center text-muted-foreground text-xs rounded">
          {isFetching ? "Loading..." : "No data"}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Eval Metric Charts — uses eval-step-metrics API for multi-run eval data
// ============================================================================

function useEvalMetricsByRunPath(
  runs: RunInfo[],
  evalName: string,
  metricName: string,
  shouldPoll: boolean,
  enabled: boolean,
  envFilters?: string[],
) {
  const runPaths = useMemo(
    () =>
      runs
        .slice(0, 20)
        .map((run) => run.runPath)
        .filter(Boolean),
    [runs],
  )

  const query = useEvalStepMetricsMultiRun(
    runPaths,
    evalName,
    metricName ? [metricName] : [],
    enabled && runPaths.length > 0 && !!metricName,
    shouldPoll,
    envFilters,
  )

  const metricsByRunPath = useMemo(() => {
    const metrics = new Map<string, { step: number; value: number }[]>()
    query.data?.runs?.forEach((run) => {
      metrics.set(run.run_path, run.metrics ?? [])
    })
    return metrics
  }, [query.data])

  return {
    metricsByRunPath,
    isFetching: query.isFetching,
    isRefetching: query.isFetching && !!query.data,
  }
}

interface EvalMetricChartProps {
  runs: RunInfo[]
  shouldPoll: boolean
  evalName: string
  metricName: string
  label?: string
  unit?: string
  showZeroLine?: boolean
  statType?: string
  isTokenMetric?: boolean
  showEma?: boolean
  emaSpan?: number
  hoveredRunId?: string | null
  xAxisMode?: "step" | "time"
  stepTimesByRun?: Map<string, Map<number, number>>
  firstStepTimesByRun?: Map<string, number>
  isStepTimesFetching?: boolean
  isStepTimesRefetching?: boolean
  availableEnvs?: string[]
  scrollRoot?: Element | null
  maxStepLimit?: number | null
  maxTimeLimit?: number | null
  headerPrefix?: React.ReactNode
  headerSuffix?: React.ReactNode
  filterKey?: string
}

export function EvalMetricChart({
  runs,
  shouldPoll,
  evalName,
  metricName,
  label,
  unit,
  showZeroLine,
  statType,
  isTokenMetric,
  showEma = false,
  emaSpan = 10,
  hoveredRunId = null,
  xAxisMode = "step",
  stepTimesByRun,
  firstStepTimesByRun,
  isStepTimesFetching = false,
  isStepTimesRefetching = false,
  scrollRoot = null,
  maxStepLimit,
  maxTimeLimit,
  headerPrefix,
  headerSuffix,
  availableEnvs,
  filterKey: filterKeyProp,
}: EvalMetricChartProps) {
  const darkMode = useAtomValue(darkModeAtom)
  const visibilityRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<uPlot | null>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  const [syncedCursor, setSyncedCursor] = useAtom(syncedCursorAtom)
  const chartId = useMemo(
    () => `eval-metric-${evalName}-${metricName}`,
    [evalName, metricName],
  )
  const isSyncingRef = useRef(false)
  const isCtrlPressedRef = useRef(false)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Control" || e.key === "Meta")
        isCtrlPressedRef.current = true
    }
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Control" || e.key === "Meta") {
        isCtrlPressedRef.current = false
        setSyncedCursor(null)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    window.addEventListener("keyup", handleKeyUp)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("keyup", handleKeyUp)
    }
  }, [setSyncedCursor])

  // --- Chart filter state (shared atom, same as MetricChart) ---
  const filterKey = filterKeyProp ?? `eval/${evalName}/${metricName}`
  const [metricsChartFilters, setMetricsChartFilters] = useAtom(
    metricsChartFiltersAtom,
  )
  const shouldDefaultIgnoreFirstStep =
    DEFAULT_IGNORE_FIRST_STEP_METRICS.has(metricName)
  const defaultIgnoreFirstStepValue = shouldDefaultIgnoreFirstStep
  const metricFilters = metricsChartFilters[filterKey]
  const ignoreOutliers = metricFilters?.ignoreOutliers ?? false
  const ignoreFirstStep =
    metricFilters?.ignoreFirstStep ?? defaultIgnoreFirstStepValue
  const minY = metricFilters?.minY ?? null
  const maxY = metricFilters?.maxY ?? null
  const tagFilters = metricFilters?.tagFilters ?? {}
  const envFilters = metricFilters?.envFilters ?? []
  const hasActiveTagFilters = Object.keys(tagFilters).length > 0
  const hasActiveEnvFilters = envFilters.length > 0

  const updateMetricFilters = useCallback(
    (
      updater: (current: MetricsChartFilterState) => MetricsChartFilterState,
    ) => {
      setMetricsChartFilters((prev) => {
        const defaults: MetricsChartFilterState = {
          ignoreOutliers: false,
          ignoreFirstStep: defaultIgnoreFirstStepValue,
          minY: null,
          maxY: null,
          tagFilters: {},
          envFilters: [],
        }
        const current = prev[filterKey] ?? defaults
        const next = updater(current)

        const isAtDefaultValues =
          next.ignoreOutliers === defaults.ignoreOutliers &&
          next.ignoreFirstStep === defaults.ignoreFirstStep &&
          next.minY === defaults.minY &&
          next.maxY === defaults.maxY &&
          Object.keys(next.tagFilters ?? {}).length === 0 &&
          (next.envFilters ?? []).length === 0

        if (isAtDefaultValues) {
          if (!Object.prototype.hasOwnProperty.call(prev, filterKey)) {
            return prev
          }
          const rest = { ...prev }
          delete rest[filterKey]
          return rest
        }

        return { ...prev, [filterKey]: next }
      })
    },
    [filterKey, setMetricsChartFilters, defaultIgnoreFirstStepValue],
  )

  const setIgnoreOutliers = useCallback(
    (checked: boolean) => {
      updateMetricFilters((current) => ({
        ...current,
        ignoreOutliers: checked,
      }))
    },
    [updateMetricFilters],
  )

  const setIgnoreFirstStep = useCallback(
    (checked: boolean) => {
      updateMetricFilters((current) => ({
        ...current,
        ignoreFirstStep: checked,
      }))
    },
    [updateMetricFilters],
  )

  const setMinY = useCallback(
    (value: number | null) => {
      updateMetricFilters((current) => ({
        ...current,
        minY: value,
      }))
    },
    [updateMetricFilters],
  )

  const setMaxY = useCallback(
    (value: number | null) => {
      updateMetricFilters((current) => ({
        ...current,
        maxY: value,
      }))
    },
    [updateMetricFilters],
  )

  const toggleTagFilter = useCallback(
    (tagName: string, tagValue: string) => {
      updateMetricFilters((current) => {
        const currentTags = { ...(current.tagFilters ?? {}) }
        const currentValues = currentTags[tagName] ?? []
        if (currentValues.includes(tagValue)) {
          const newValues = currentValues.filter((v) => v !== tagValue)
          if (newValues.length === 0) {
            delete currentTags[tagName]
          } else {
            currentTags[tagName] = newValues
          }
        } else {
          currentTags[tagName] = [...currentValues, tagValue]
        }
        return { ...current, tagFilters: currentTags }
      })
    },
    [updateMetricFilters],
  )

  const toggleEnvFilter = useCallback(
    (envName: string) => {
      updateMetricFilters((current) => {
        const currentEnvs = [...(current.envFilters ?? [])]
        const idx = currentEnvs.indexOf(envName)
        if (idx >= 0) {
          currentEnvs.splice(idx, 1)
        } else {
          currentEnvs.push(envName)
        }
        return { ...current, envFilters: currentEnvs }
      })
    },
    [updateMetricFilters],
  )

  const isVisible = useOnScreen(visibilityRef, {
    root: scrollRoot,
    threshold: 0,
  })

  const sortedRuns = useMemo(() => {
    const nonSelected = runs.filter((r) => !r.isSelected)
    const selected = runs.filter((r) => r.isSelected)
    return [...nonSelected, ...selected]
  }, [runs])
  const plottedRuns = useMemo(() => sortedRuns.slice(0, 20), [sortedRuns])
  const runDataIndexByRunPath = useMemo(() => {
    const indexByPath = new Map<string, number>()
    plottedRuns.forEach((run, idx) => {
      indexByPath.set(run.runPath, idx)
    })
    return indexByPath
  }, [plottedRuns])

  const {
    metricsByRunPath,
    isFetching: isMetricsFetching,
    isRefetching: isMetricsRefetching,
  } = useEvalMetricsByRunPath(runs, evalName, metricName, shouldPoll, isVisible, envFilters)

  const {
    uplotData,
    seriesConfig,
    isFetching,
    isRefetching,
    hasData,
    timeToStepByRun,
    outlierBounds,
  } = useMemo(() => {
    const stepSet = new Set<number>()
    const runData: Map<number, Map<number, number>> = new Map()
    const runStepTimes: Map<number, Map<number, number>> = new Map()
    const timeToStepByRun: Map<number, Map<number, number>> = new Map()
    const useTimeAxis =
      xAxisMode === "time" && !!stepTimesByRun && !!firstStepTimesByRun
    const fetching = isMetricsFetching

    sortedRuns.forEach((run, index) => {
      if (index >= 20) return
      const metrics = metricsByRunPath.get(run.runPath) ?? []
      const stepMap = new Map<number, number>()
      metrics.forEach((m) => {
        stepSet.add(m.step)
        stepMap.set(m.step, m.value)
      })
      runData.set(index, stepMap)

      if (useTimeAxis) {
        const stTimes = stepTimesByRun?.get(run.runPath)
        const firstStepTime = firstStepTimesByRun?.get(run.runPath)
        if (stTimes && firstStepTime !== undefined) {
          const relativeTimes = new Map<number, number>()
          const timeToStep = new Map<number, number>()
          stTimes.forEach((time, step) => {
            const relativeTime = time - firstStepTime
            relativeTimes.set(step, relativeTime)
            timeToStep.set(relativeTime, step)
          })
          runStepTimes.set(index, relativeTimes)
          timeToStepByRun.set(index, timeToStep)
        }
      }
    })

    const computeIsRefetching = () => {
      if (!fetching && !isStepTimesFetching) return false
      if (fetching && !isMetricsRefetching) return false
      if (isStepTimesFetching && !isStepTimesRefetching) return false
      return true
    }

    if (stepSet.size === 0) {
      return {
        uplotData: null,
        seriesConfig: [],
        isFetching: fetching || isStepTimesFetching,
        isRefetching: computeIsRefetching(),
        hasData: false,
        timeToStepByRun,
        outlierBounds: null,
      }
    }

    let sortedSteps = Array.from(stepSet).sort((a, b) => a - b)
    if (maxStepLimit != null && !useTimeAxis) {
      sortedSteps = sortedSteps.filter((step) => step <= maxStepLimit)
    }

    let xValues: number[] = sortedSteps
    if (useTimeAxis) {
      const timeSet = new Set<number>()
      runData.forEach((stepMap, runIndex) => {
        const timeMap = runStepTimes.get(runIndex)
        if (!timeMap) return
        stepMap.forEach((_value, step) => {
          const time = timeMap.get(step)
          if (time !== undefined) timeSet.add(time)
        })
      })
      if (timeSet.size === 0) {
        return {
          uplotData: null,
          seriesConfig: [],
          isFetching: fetching || isStepTimesFetching,
          isRefetching: computeIsRefetching(),
          hasData: false,
          timeToStepByRun,
          outlierBounds: null,
        }
      }
      let sortedTimes = Array.from(timeSet).sort((a, b) => a - b)
      if (maxTimeLimit != null) {
        sortedTimes = sortedTimes.filter((time) => time <= maxTimeLimit)
      }
      xValues = sortedTimes
    }

    // Compute outlier bounds across all runs if ignoreOutliers is enabled
    let outlierBounds: { lower: number; upper: number } | null = null
    if (ignoreOutliers) {
      const allValues: number[] = []
      runData.forEach((stepMap) => {
        stepMap.forEach((value) => {
          allValues.push(value)
        })
      })
      outlierBounds = computeIQRBounds(allValues)
    }

    const xData = new Float64Array(xValues)
    const series: uPlot.Series[] = [
      { label: useTimeAxis ? "Time (s)" : "Step" },
    ]
    const dataArrays: (Float64Array | number[])[] = [xData]

    sortedRuns.forEach((run, index) => {
      if (index >= 20) return
      const stepMap = runData.get(index) || new Map()
      let values: (number | null)[]
      if (useTimeAxis) {
        const timeMap = runStepTimes.get(index)
        const timeToValue = new Map<number, number>()
        if (timeMap) {
          stepMap.forEach((value, step) => {
            const time = timeMap.get(step)
            if (time !== undefined) timeToValue.set(time, value)
          })
        }
        values = xValues.map((time) => timeToValue.get(time) ?? null)
      } else {
        values = sortedSteps.map((step) => stepMap.get(step) ?? null)
      }
      const isHovered = hoveredRunId === run.runPath
      const someRunIsHovered = hoveredRunId !== null
      const valueAlpha =
        someRunIsHovered && !isHovered ? 0.1 : showEma ? 0.3 : 1
      dataArrays.push(values as number[])
      series.push({
        label: getRunDisplayName(run),
        stroke: run.color,
        width: 1,
        alpha: valueAlpha,
        spanGaps: true,
        points: { show: false },
      })
    })

    if (showEma) {
      sortedRuns.forEach((run, index) => {
        if (index >= 20) return
        const stepMap = runData.get(index) || new Map()
        const alpha = 2 / (emaSpan + 1)
        let ema: number | null = null
        let emaValues: (number | null)[]
        if (useTimeAxis) {
          const timeMap = runStepTimes.get(index)
          const emaByTime = new Map<number, number>()
          sortedSteps.forEach((step) => {
            const value = stepMap.get(step)
            if (value !== undefined) {
              ema = ema === null ? value : alpha * value + (1 - alpha) * ema
              const time = timeMap?.get(step)
              if (time !== undefined && ema !== null) emaByTime.set(time, ema)
            }
          })
          emaValues = xValues.map((time) => emaByTime.get(time) ?? null)
        } else {
          const values: (number | null)[] = []
          sortedSteps.forEach((step) => {
            const value = stepMap.get(step)
            if (value !== undefined) {
              ema = ema === null ? value : alpha * value + (1 - alpha) * ema
              values.push(ema)
            } else {
              values.push(null)
            }
          })
          emaValues = values
        }
        const isHovered = hoveredRunId === run.runPath
        const someRunIsHovered = hoveredRunId !== null
        const emaAlpha = someRunIsHovered && !isHovered ? 0.1 : 1
        dataArrays.push(emaValues as number[])
        series.push({
          label: `${getRunDisplayName(run)} (EMA)`,
          stroke: run.color,
          width: 1.5,
          alpha: emaAlpha,
          spanGaps: true,
          points: { show: false },
        })
      })
    }

    return {
      uplotData: dataArrays as uPlot.AlignedData,
      seriesConfig: series,
      isFetching: fetching || isStepTimesFetching,
      isRefetching: computeIsRefetching(),
      hasData: true,
      timeToStepByRun,
      outlierBounds,
    }
  }, [
    sortedRuns,
    metricsByRunPath,
    isMetricsFetching,
    isMetricsRefetching,
    showEma,
    emaSpan,
    hoveredRunId,
    xAxisMode,
    stepTimesByRun,
    firstStepTimesByRun,
    isStepTimesFetching,
    isStepTimesRefetching,
    maxStepLimit,
    maxTimeLimit,
    ignoreOutliers,
  ])

  const displayStatType = statType || metricName.split("_").pop() || ""
  const isTimeAxis = xAxisMode === "time"

  const formatValue = useCallback(
    (v: number | null | undefined): string => {
      if (v === undefined || v === null) return "N/A"
      if (isTokenMetric) {
        if (displayStatType === "min" || displayStatType === "max")
          return Math.round(v).toLocaleString()
        return formatValueSmart(v)
      }
      return formatValueSmart(v)
    },
    [isTokenMetric, displayStatType],
  )

  const formatYAxisTick = useCallback(
    (v: number): string => {
      if (Math.abs(v) >= 1000) return `${parseFloat((v / 1000).toFixed(1))}k`
      if (isTokenMetric) {
        if (displayStatType === "min" || displayStatType === "max")
          return Math.round(v).toString()
        return String(parseFloat(v.toFixed(1)))
      }
      if (Number.isInteger(v)) return v.toLocaleString()
      if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1)
      return String(parseFloat(v.toFixed(2)))
    },
    [isTokenMetric, displayStatType],
  )

  const formatXAxisTick = useCallback(
    (v: number): string => {
      if (isTimeAxis) return formatSecondsCompact(v)
      return v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toString()
    },
    [isTimeAxis],
  )

  useEffect(() => {
    if (!containerRef.current || !uplotData || !hasData) return
    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight || 192

    let minVal = Infinity
    let maxVal = -Infinity
    for (let i = 1; i < uplotData.length; i++) {
      const arr = uplotData[i]
      for (let j = 0; j < arr.length; j++) {
        if (ignoreFirstStep && j === 0) continue
        const v = arr[j]
        if (v !== null && v !== undefined) {
          if (ignoreOutliers && outlierBounds) {
            if (v >= outlierBounds.lower && v <= outlierBounds.upper) {
              if (v < minVal) minVal = v
              if (v > maxVal) maxVal = v
            }
          } else {
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
          }
        }
      }
    }

    if (minVal === Infinity && outlierBounds) {
      minVal = outlierBounds.lower
      maxVal = outlierBounds.upper
    }

    const range = maxVal - minVal
    const absMax = Math.max(Math.abs(maxVal), Math.abs(minVal))
    const minPadding = Math.max(absMax * 0.05, 1e-10)
    const padding = Math.max(range * 0.1, minPadding)
    let yMin: number, yMax: number
    if (showZeroLine) {
      yMin = Math.min(minVal - padding, 0)
      yMax = Math.max(maxVal + padding, 0)
    } else {
      yMin = minVal - padding
      yMax = maxVal + padding
    }

    if (minY !== null) yMin = minY
    if (maxY !== null) yMax = maxY

    const gridColor = darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(128, 128, 128, 0.15)"
    const tickLabelColor = darkMode ? "rgba(255, 255, 255, 0.65)" : "rgba(100, 100, 100, 0.9)"

    const calcYAxisSize = (u: uPlot, values: string[]) => {
      if (!values || values.length === 0) return 40
      const maxLen = Math.max(...values.map((v) => v.length))
      return Math.max(40, maxLen * 7 + 14)
    }

    const opts: uPlot.Options = {
      width,
      height,
      padding: [4, 8, 0, 0],
      cursor: { show: true, x: true, y: false, points: { show: false } },
      legend: { show: false },
      scales: { x: { time: false }, y: { range: [yMin, yMax] } },
      axes: [
        {
          stroke: tickLabelColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: 24,
          values: (u, vals) => vals.map(formatXAxisTick),
        },
        {
          stroke: tickLabelColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: calcYAxisSize,
          values: (u, vals) => vals.map(formatYAxisTick),
        },
      ],
      series: seriesConfig,
      hooks: {
        draw: [
          (u) => {
            if (!uplotData) return
            const xSeriesData = uplotData[0] as ArrayLike<number>
            const runCount = Math.min(sortedRuns.length, 20)
            const seriesCount = u.series.length
            const ctx = u.ctx
            ctx.save()
            for (let seriesIdx = 1; seriesIdx < seriesCount; seriesIdx++) {
              const yData = uplotData[seriesIdx] as
                | (number | null | undefined)[]
                | undefined
              if (!yData) continue
              let lastIndex = -1
              for (let i = yData.length - 1; i >= 0; i--) {
                if (yData[i] !== null && yData[i] !== undefined) {
                  lastIndex = i
                  break
                }
              }
              if (lastIndex === -1) continue
              const xVal = xSeriesData[lastIndex]
              const yVal = yData[lastIndex]
              if (xVal === undefined || yVal === undefined || yVal === null)
                continue
              const x = u.valToPos(Number(xVal), "x", true)
              const y = u.valToPos(Number(yVal), "y", true)
              if (!Number.isFinite(x) || !Number.isFinite(y)) continue
              const s = u.series[seriesIdx]
              const seriesAlpha = typeof s?.alpha === "number" ? s.alpha : 1
              let stroke = typeof s?.stroke === "string" ? s.stroke : undefined
              if (!stroke) {
                const baseIdx = seriesIdx - 1
                if (baseIdx < runCount) stroke = sortedRuns[baseIdx]?.color
                else if (showEma) {
                  const emaIdx = baseIdx - runCount
                  if (emaIdx >= 0 && emaIdx < runCount)
                    stroke = sortedRuns[emaIdx]?.color
                }
              }
              if (!stroke) continue
              ctx.globalAlpha = seriesAlpha
              ctx.fillStyle = stroke
              ctx.beginPath()
              ctx.arc(x, y, 4, 0, Math.PI * 2)
              ctx.fill()
            }
            ctx.restore()
          },
        ],
        setCursor: [
          (u) => {
            if (!tooltipRef.current) return
            const { left, top, idx } = u.cursor
            if (
              idx === null ||
              idx === undefined ||
              left === undefined ||
              top === undefined ||
              left < 0
            ) {
              tooltipRef.current.style.display = "none"
              return
            }
            const xValue = uplotData[0][idx]
            if (xValue === undefined) {
              tooltipRef.current.style.display = "none"
              return
            }
            if (isCtrlPressedRef.current && !isSyncingRef.current) {
              setSyncedCursor({
                xValue: Number(xValue),
                sourceChartId: chartId,
              })
            }
            const headerLabel = isTimeAxis
              ? `Time ${formatTooltipElapsedTime(Number(xValue))}`
              : `Step ${Number(xValue).toLocaleString()}`
            let html = `<div class="font-medium mb-1">${headerLabel}</div>`
            const numRuns = plottedRuns.length
            runs.forEach((run) => {
              const runIdx = runDataIndexByRunPath.get(run.runPath)
              if (runIdx === undefined) return
              const valueIdx = runIdx + 1
              const valueSeries = uplotData[valueIdx] as
                | ArrayLike<number | null | undefined>
                | undefined
              if (!valueSeries) return
              const value = valueSeries[idx]
              let emaValue: number | null | undefined = undefined
              if (showEma) {
                const emaIdx = numRuns + runIdx + 1
                emaValue = (
                  uplotData[emaIdx] as
                    | ArrayLike<number | null | undefined>
                    | undefined
                )?.[idx]
              }
              if (value === null || value === undefined) return
              html += `
                <div class="mt-1">
                  <div class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full shrink-0" style="background-color: ${run.color}"></div>
                    <span>${formatRunLabelHtml(run)}</span>
                  </div>
                  <div class="ml-4 flex items-center gap-2">
                    <span class="font-medium" style="opacity: ${showEma ? 0.7 : 1}">${formatValue(value)}${unit ? ` ${unit}` : ""}</span>
                    ${
                      showEma && emaValue !== undefined && emaValue !== null
                        ? `<span class="text-muted-foreground">(EMA: ${formatValue(emaValue)})</span>`
                        : ""
                    }
                  </div>
                </div>
              `
            })
            tooltipRef.current.innerHTML = html
            tooltipRef.current.style.display = "block"
            const containerRect = containerRef.current!.getBoundingClientRect()
            const tooltipRect = tooltipRef.current.getBoundingClientRect()
            let tooltipX = containerRect.left + left + 15
            if (tooltipX + tooltipRect.width + 20 > window.innerWidth) {
              tooltipX =
                containerRect.left + left - tooltipRect.width - 15 + u.bbox.left
            }
            tooltipX = Math.max(4, tooltipX)
            let tooltipY = containerRect.top - tooltipRect.height + 20
            if (tooltipY < 4) tooltipY = containerRect.bottom - 20
            tooltipRef.current.style.left = `${tooltipX}px`
            tooltipRef.current.style.top = `${tooltipY}px`
          },
        ],
      },
    }

    if (chartRef.current) chartRef.current.destroy()
    const chart = new uPlot(opts, uplotData, container)
    chartRef.current = chart

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: newWidth, height: newHeight } = entry.contentRect
        if (chart && newWidth > 0 && newHeight > 0)
          chart.setSize({ width: newWidth, height: newHeight })
      }
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
      chart.destroy()
      chartRef.current = null
    }
  }, [
    uplotData,
    seriesConfig,
    hasData,
    formatXAxisTick,
    formatYAxisTick,
    formatValue,
    isTimeAxis,
    timeToStepByRun,
    sortedRuns,
    hoveredRunId,
    showEma,
    showZeroLine,
    unit,
    runs,
    plottedRuns,
    runDataIndexByRunPath,
    chartId,
    setSyncedCursor,
    ignoreOutliers,
    ignoreFirstStep,
    outlierBounds,
    minY,
    maxY,
    darkMode,
  ])

  useEffect(() => {
    if (!syncedCursor) {
      if (tooltipRef.current) tooltipRef.current.style.display = "none"
      return
    }
    if (
      syncedCursor.sourceChartId === chartId ||
      !chartRef.current ||
      !uplotData
    )
      return
    const chart = chartRef.current
    const xData = uplotData[0]
    if (!xData || xData.length === 0) return
    const targetX = syncedCursor.xValue
    let closestIdx = 0
    let closestDist = Infinity
    for (let i = 0; i < xData.length; i++) {
      const dist = Math.abs(Number(xData[i]) - targetX)
      if (dist < closestDist) {
        closestDist = dist
        closestIdx = i
      }
    }
    const left = chart.valToPos(Number(xData[closestIdx]), "x")
    if (left >= 0 && left <= chart.width) {
      isSyncingRef.current = true
      chart.setCursor({ left, top: 0 })
      isSyncingRef.current = false
    }
  }, [syncedCursor, chartId, uplotData])

  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) tooltipRef.current.style.display = "none"
    if (syncedCursor?.sourceChartId === chartId) setSyncedCursor(null)
  }, [syncedCursor, chartId, setSyncedCursor])

  const showLoadingOpacity = isFetching && !isRefetching
  const titleText =
    label ||
    (displayStatType
      ? `${displayStatType.charAt(0).toUpperCase()}${displayStatType.slice(1)}`
      : metricName)

  return (
    <div
      ref={visibilityRef}
      className={cn(
        "group/chart rounded-lg border border-border p-3 transition-opacity bg-background h-[246px] flex flex-col",
        showLoadingOpacity && "opacity-50",
      )}
    >
      <div className="shrink-0 mb-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-0.5 min-w-0">
            {headerPrefix}
            <h4
              className="text-xs font-medium leading-snug line-clamp-2 break-words"
              title={metricName}
            >
              {titleText}
            </h4>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="h-5 px-1.5 text-[10px] rounded border border-border hover:bg-muted flex items-center gap-1 transition-all opacity-0 group-hover/chart:opacity-100">
                  <SlidersHorizontal className="h-3 w-3" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-[160px]">
                <DropdownMenuCheckboxItem
                  checked={ignoreOutliers}
                  onCheckedChange={setIgnoreOutliers}
                >
                  Ignore Outliers
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem
                  checked={ignoreFirstStep}
                  onCheckedChange={setIgnoreFirstStep}
                >
                  Ignore First Step
                </DropdownMenuCheckboxItem>
                <DropdownMenuSeparator />
                <div className="flex items-center gap-2 px-2 py-1.5">
                  <span className="text-xs text-muted-foreground font-medium w-10">Min Y</span>
                  <input
                    type="number"
                    className="w-20 h-6 px-2 text-xs rounded border border-border bg-background focus:outline-none focus:ring-1 focus:ring-ring"
                    placeholder="Auto"
                    value={minY ?? ""}
                    onChange={(e) => {
                      const val = e.target.value
                      setMinY(val === "" ? null : Number(val))
                    }}
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                  />
                </div>
                <div className="flex items-center gap-2 px-2 py-1.5">
                  <span className="text-xs text-muted-foreground font-medium w-10">Max Y</span>
                  <input
                    type="number"
                    className="w-20 h-6 px-2 text-xs rounded border border-border bg-background focus:outline-none focus:ring-1 focus:ring-ring"
                    placeholder="Auto"
                    value={maxY ?? ""}
                    onChange={(e) => {
                      const val = e.target.value
                      setMaxY(val === "" ? null : Number(val))
                    }}
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                  />
                </div>
                {availableEnvs && availableEnvs.length > 1 && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuLabel className="text-[10px] text-muted-foreground">
                      Filter by Environment
                    </DropdownMenuLabel>
                    {availableEnvs.map((envName) => (
                      <DropdownMenuCheckboxItem
                        key={envName}
                        checked={envFilters.includes(envName)}
                        onCheckedChange={() => toggleEnvFilter(envName)}
                      >
                        {envName}
                      </DropdownMenuCheckboxItem>
                    ))}
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
            {headerSuffix}
          </div>
        </div>
        {(ignoreOutliers || ignoreFirstStep || minY !== null || maxY !== null || hasActiveTagFilters || hasActiveEnvFilters) && (
          <div className="flex items-center gap-1 mt-1 flex-wrap">
            {ignoreOutliers && (
              <FilterBadge
                label="Ignore Outliers"
                onRemove={() => setIgnoreOutliers(false)}
              />
            )}
            {ignoreFirstStep && (
              <FilterBadge
                label="Ignore First Step"
                onRemove={() => setIgnoreFirstStep(false)}
              />
            )}
            {minY !== null && (
              <FilterBadge
                label={`Min Y: ${minY}`}
                onRemove={() => setMinY(null)}
              />
            )}
            {maxY !== null && (
              <FilterBadge
                label={`Max Y: ${maxY}`}
                onRemove={() => setMaxY(null)}
              />
            )}
            {envFilters.map((envName) => (
              <FilterBadge
                key={`env:${envName}`}
                label={`env: ${envName}`}
                onRemove={() => toggleEnvFilter(envName)}
              />
            ))}
            {Object.entries(tagFilters).map(([tagName, tagValues]) =>
              tagValues.map((tagValue) => (
                <FilterBadge
                  key={`${tagName}:${tagValue}`}
                  label={`${tagName}: ${tagValue}`}
                  onRemove={() => toggleTagFilter(tagName, tagValue)}
                />
              ))
            )}
          </div>
        )}
      </div>
      {hasData ? (
        <div
          className="flex-1 min-h-0 relative bg-background rounded"
          ref={containerRef}
          onMouseLeave={handleMouseLeave}
        >
          <div
            ref={tooltipRef}
            className="fixed z-[9999] max-w-[360px] bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl border border-border pointer-events-none"
            style={{ display: "none" }}
          />
          {isRefetching && (
            <Loader2 className="absolute bottom-0.5 left-0.5 h-3 w-3 animate-spin text-muted-foreground" />
          )}
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex items-center justify-center text-muted-foreground text-xs rounded">
          {isFetching ? "Loading..." : "No data"}
        </div>
      )}
    </div>
  )
}

interface EvalMetricPrefixSectionProps {
  runs: RunInfo[]
  shouldPoll: boolean
  evalName: string
  prefix: string
  label: string
  unit?: string
  showZeroLine?: boolean
  isTokenMetric?: boolean
  showEma?: boolean
  emaSpan?: number
  hoveredRunId?: string | null
  xAxisMode?: "step" | "time"
  stepTimesByRun?: Map<string, Map<number, number>>
  firstStepTimesByRun?: Map<string, number>
  isStepTimesFetching?: boolean
  isStepTimesRefetching?: boolean
  scrollRoot?: Element | null
  maxStepLimit?: number | null
  maxTimeLimit?: number | null
  availableEnvs?: string[]
}

function EvalMetricPrefixSection({
  runs,
  shouldPoll,
  evalName,
  prefix,
  label,
  unit,
  showZeroLine,
  isTokenMetric,
  showEma,
  emaSpan,
  hoveredRunId,
  xAxisMode,
  stepTimesByRun,
  firstStepTimesByRun,
  isStepTimesFetching,
  isStepTimesRefetching,
  scrollRoot,
  maxStepLimit,
  maxTimeLimit,
  availableEnvs,
}: EvalMetricPrefixSectionProps) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-muted-foreground mb-3">
        {label}
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {STAT_SUFFIXES.map((suffix) => (
          <EvalMetricChart
            key={`${evalName}-${prefix}_${suffix}`}
            runs={runs}
            shouldPoll={shouldPoll}
            evalName={evalName}
            metricName={`${prefix}_${suffix}`}
            label={`${label} ${SUFFIX_LABELS[suffix]}`}
            unit={unit}
            showZeroLine={showZeroLine}
            statType={suffix}
            isTokenMetric={isTokenMetric}
            showEma={showEma}
            emaSpan={emaSpan}
            hoveredRunId={hoveredRunId}
            xAxisMode={xAxisMode}
            stepTimesByRun={stepTimesByRun}
            firstStepTimesByRun={firstStepTimesByRun}
            isStepTimesFetching={isStepTimesFetching}
            isStepTimesRefetching={isStepTimesRefetching}
            scrollRoot={scrollRoot}
            maxStepLimit={maxStepLimit}
            maxTimeLimit={maxTimeLimit}
            availableEnvs={availableEnvs}
          />
        ))}
      </div>
    </div>
  )
}

// ============================================================================
// Section for a single metric prefix (e.g., "reward_format") with 4 charts
// ============================================================================

interface MetricPrefixSectionProps {
  runs: RunInfo[]
  shouldPoll: boolean
  prefix: string
  label: string
  unit?: string
  showZeroLine?: boolean
  isTokenMetric?: boolean
  showEma?: boolean
  emaSpan?: number
  hoveredRunId?: string | null
  xAxisMode?: "step" | "time"
  stepTimesByRun?: Map<string, Map<number, number>>
  firstStepTimesByRun?: Map<string, number>
  isStepTimesFetching?: boolean
  isStepTimesRefetching?: boolean
  scrollRoot?: Element | null
  maxStepLimit?: number | null
  maxTimeLimit?: number | null
  extraCharts?: React.ReactNode
  hideGroupLabel?: boolean
  availableSampleTags?: Record<string, string[]>
  availableEnvs?: string[]
}

function MetricPrefixSection({
  runs,
  shouldPoll,
  prefix,
  label,
  unit,
  showZeroLine,
  isTokenMetric,
  showEma,
  emaSpan,
  hoveredRunId,
  xAxisMode,
  stepTimesByRun,
  firstStepTimesByRun,
  isStepTimesFetching,
  isStepTimesRefetching,
  scrollRoot,
  maxStepLimit,
  maxTimeLimit,
  extraCharts,
  hideGroupLabel,
  availableSampleTags,
  availableEnvs,
}: MetricPrefixSectionProps) {
  // Use the first selected run for distribution over time (single-run chart)
  const selectedRun = runs.find((r) => r.isSelected) ?? runs[0]

  return (
    <div>
      {!hideGroupLabel && (
        <h3 className="text-sm font-semibold text-muted-foreground mb-3">
          {label}
        </h3>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {STAT_SUFFIXES.map((suffix) => (
          <MetricChart
            key={`${prefix}_${suffix}`}
            runs={runs}
            shouldPoll={shouldPoll}
            metricName={`${prefix}_${suffix}`}
            label={`${label} ${SUFFIX_LABELS[suffix]}`}
            unit={unit}
            showZeroLine={showZeroLine}
            statType={suffix}
            isTokenMetric={isTokenMetric}
            showEma={showEma}
            emaSpan={emaSpan}
            hoveredRunId={hoveredRunId}
            xAxisMode={xAxisMode}
            stepTimesByRun={stepTimesByRun}
            firstStepTimesByRun={firstStepTimesByRun}
            isStepTimesFetching={isStepTimesFetching}
            isStepTimesRefetching={isStepTimesRefetching}
            scrollRoot={scrollRoot}
            maxStepLimit={maxStepLimit}
            maxTimeLimit={maxTimeLimit}
            availableSampleTags={availableSampleTags}
            availableEnvs={availableEnvs}
          />
        ))}
        {extraCharts}
        {selectedRun && (
          <DistributionOverTimeChart
            runPath={selectedRun.runPath}
            metricType={prefix}
            label={`${label} (Dist. Over Time)`}
            showZeroLine={showZeroLine}
            isTokenMetric={isTokenMetric}
            shouldPoll={shouldPoll}
            scrollRoot={scrollRoot}
          />
        )}
      </div>
    </div>
  )
}

// Hook to fetch metric data for multiple runs in one request
function useMetricsByRunPath(
  runs: RunInfo[],
  metricName: string,
  shouldPoll: boolean,
  enabled: boolean,
  tagFilters?: Record<string, string[]>,
  envFilters?: string[],
) {
  const runPaths = useMemo(
    () =>
      runs
        .slice(0, 20)
        .map((run) => run.runPath)
        .filter(Boolean),
    [runs],
  )
  const metricNames = useMemo(() => {
    if (!metricName) return []
    return [metricName]
  }, [metricName])

  const query = useStepMetricsMultiRun(
    runPaths,
    metricNames,
    enabled && runPaths.length > 0 && metricNames.length > 0,
    shouldPoll,
    tagFilters,
    envFilters,
  )

  const metricsByRunPath = useMemo(() => {
    const metrics = new Map<string, { step: number; value: number }[]>()
    query.data?.runs?.forEach((run) => {
      metrics.set(run.run_path, run.metrics ?? [])
    })
    return metrics
  }, [query.data])

  // isRefetching is true when refetching with existing data (background polling)
  // isFetching && !isRefetching means initial load or user-triggered fetch without cache
  return {
    metricsByRunPath,
    isFetching: query.isFetching,
    isRefetching: query.isFetching && !!query.data,
  }
}

// ============================================================================
// Metric Chart - Shared chart for general + prefix metrics
// ============================================================================

interface MetricChartProps {
  runs: RunInfo[]
  shouldPoll: boolean
  metricName: string
  label?: string
  unit?: string
  showZeroLine?: boolean
  statType?: string
  isTokenMetric?: boolean
  showEma?: boolean
  emaSpan?: number
  hoveredRunId?: string | null
  xAxisMode?: "step" | "time"
  stepTimesByRun?: Map<string, Map<number, number>>
  firstStepTimesByRun?: Map<string, number>
  isStepTimesFetching?: boolean
  isStepTimesRefetching?: boolean
  scrollRoot?: Element | null
  defaultIgnoreOutliers?: boolean
  maxStepLimit?: number | null
  maxTimeLimit?: number | null
  headerPrefix?: React.ReactNode
  headerSuffix?: React.ReactNode
  availableSampleTags?: Record<string, string[]>
  availableEnvs?: string[]
  filterKey?: string
}

export function MetricChart({
  runs,
  shouldPoll,
  metricName,
  label,
  unit,
  showZeroLine,
  statType,
  isTokenMetric,
  showEma = false,
  emaSpan = 10,
  hoveredRunId = null,
  xAxisMode = "step",
  stepTimesByRun,
  firstStepTimesByRun,
  isStepTimesFetching = false,
  isStepTimesRefetching = false,
  scrollRoot = null,
  defaultIgnoreOutliers,
  maxStepLimit,
  maxTimeLimit,
  headerPrefix,
  headerSuffix,
  availableSampleTags,
  availableEnvs,
  filterKey: filterKeyProp,
}: MetricChartProps) {
  const darkMode = useAtomValue(darkModeAtom)
  const visibilityRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<uPlot | null>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  // Synced cursor for CTRL+hover
  const [syncedCursor, setSyncedCursor] = useAtom(syncedCursorAtom)
  const [metricsChartFilters, setMetricsChartFilters] = useAtom(
    metricsChartFiltersAtom,
  )
  const chartId = useMemo(() => `metric-${metricName}`, [metricName])
  const isSyncingRef = useRef(false)
  const isCtrlPressedRef = useRef(false)

  // Track CTRL key for synced tooltips
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Control" || e.key === "Meta") {
        isCtrlPressedRef.current = true
      }
    }
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Control" || e.key === "Meta") {
        isCtrlPressedRef.current = false
        setSyncedCursor(null)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    window.addEventListener("keyup", handleKeyUp)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("keyup", handleKeyUp)
    }
  }, [setSyncedCursor])

  // Determine default values based on metric name
  const filterKey = filterKeyProp ?? metricName
  const defaultIgnoreOutliersValue = defaultIgnoreOutliers ?? false
  const shouldDefaultIgnoreFirstStep =
    DEFAULT_IGNORE_FIRST_STEP_METRICS.has(metricName)
  const defaultIgnoreFirstStepValue = shouldDefaultIgnoreFirstStep
  const metricFilters = metricsChartFilters[filterKey]
  const ignoreOutliers =
    metricFilters?.ignoreOutliers ?? defaultIgnoreOutliersValue
  const ignoreFirstStep =
    metricFilters?.ignoreFirstStep ?? defaultIgnoreFirstStepValue
  const minY = metricFilters?.minY ?? null
  const maxY = metricFilters?.maxY ?? null

  const tagFilters = metricFilters?.tagFilters ?? {}
  const envFilters = metricFilters?.envFilters ?? []

  const updateMetricFilters = useCallback(
    (
      updater: (current: MetricsChartFilterState) => MetricsChartFilterState,
    ) => {
      setMetricsChartFilters((prev) => {
        const defaults: MetricsChartFilterState = {
          ignoreOutliers: defaultIgnoreOutliersValue,
          ignoreFirstStep: defaultIgnoreFirstStepValue,
          minY: null,
          maxY: null,
          tagFilters: {},
          envFilters: [],
        }
        const current = prev[filterKey] ?? defaults
        const next = updater(current)

        const isAtDefaultValues =
          next.ignoreOutliers === defaults.ignoreOutliers &&
          next.ignoreFirstStep === defaults.ignoreFirstStep &&
          next.minY === defaults.minY &&
          next.maxY === defaults.maxY &&
          Object.keys(next.tagFilters ?? {}).length === 0 &&
          (next.envFilters ?? []).length === 0

        if (isAtDefaultValues) {
          if (!Object.prototype.hasOwnProperty.call(prev, filterKey)) {
            return prev
          }
          const rest = { ...prev }
          delete rest[filterKey]
          return rest
        }

        return { ...prev, [filterKey]: next }
      })
    },
    [
      filterKey,
      setMetricsChartFilters,
      defaultIgnoreOutliersValue,
      defaultIgnoreFirstStepValue,
    ],
  )

  const setIgnoreOutliers = useCallback(
    (checked: boolean) => {
      updateMetricFilters((current) => ({
        ...current,
        ignoreOutliers: checked,
      }))
    },
    [updateMetricFilters],
  )

  const setIgnoreFirstStep = useCallback(
    (checked: boolean) => {
      updateMetricFilters((current) => ({
        ...current,
        ignoreFirstStep: checked,
      }))
    },
    [updateMetricFilters],
  )

  const setMinY = useCallback(
    (value: number | null) => {
      updateMetricFilters((current) => ({
        ...current,
        minY: value,
      }))
    },
    [updateMetricFilters],
  )

  const setMaxY = useCallback(
    (value: number | null) => {
      updateMetricFilters((current) => ({
        ...current,
        maxY: value,
      }))
    },
    [updateMetricFilters],
  )

  const toggleTagFilter = useCallback(
    (tagName: string, tagValue: string) => {
      updateMetricFilters((current) => {
        const currentTags = { ...(current.tagFilters ?? {}) }
        const currentValues = currentTags[tagName] ?? []
        if (currentValues.includes(tagValue)) {
          const newValues = currentValues.filter((v) => v !== tagValue)
          if (newValues.length === 0) {
            delete currentTags[tagName]
          } else {
            currentTags[tagName] = newValues
          }
        } else {
          currentTags[tagName] = [...currentValues, tagValue]
        }
        return { ...current, tagFilters: currentTags }
      })
    },
    [updateMetricFilters],
  )

  const toggleEnvFilter = useCallback(
    (envName: string) => {
      updateMetricFilters((current) => {
        const currentEnvs = [...(current.envFilters ?? [])]
        const idx = currentEnvs.indexOf(envName)
        if (idx >= 0) {
          currentEnvs.splice(idx, 1)
        } else {
          currentEnvs.push(envName)
        }
        return { ...current, envFilters: currentEnvs }
      })
    },
    [updateMetricFilters],
  )

  const hasActiveTagFilters = Object.keys(tagFilters).length > 0
  const hasActiveEnvFilters = envFilters.length > 0

  const isVisible = useOnScreen(visibilityRef, {
    root: scrollRoot,
    threshold: 0,
  })

  // Sort runs so the selected run is last (drawn on top in uPlot)
  const sortedRuns = useMemo(() => {
    const nonSelected = runs.filter((r) => !r.isSelected)
    const selected = runs.filter((r) => r.isSelected)
    return [...nonSelected, ...selected]
  }, [runs])
  const plottedRuns = useMemo(() => sortedRuns.slice(0, 20), [sortedRuns])
  const runDataIndexByRunPath = useMemo(() => {
    const indexByPath = new Map<string, number>()
    plottedRuns.forEach((run, idx) => {
      indexByPath.set(run.runPath, idx)
    })
    return indexByPath
  }, [plottedRuns])

  const {
    metricsByRunPath,
    isFetching: isMetricsFetching,
    isRefetching: isMetricsRefetching,
  } = useMetricsByRunPath(runs, metricName, shouldPoll, isVisible, tagFilters, envFilters)

  // Combine data from all runs into uPlot format
  const {
    uplotData,
    seriesConfig,
    isFetching,
    isRefetching,
    hasData,
    timeToStepByRun,
    outlierBounds,
  } = useMemo(() => {
    const stepSet = new Set<number>()
    const runData: Map<number, Map<number, number>> = new Map()
    const runStepTimes: Map<number, Map<number, number>> = new Map()
    const timeToStepByRun: Map<number, Map<number, number>> = new Map()
    const useTimeAxis =
      xAxisMode === "time" && !!stepTimesByRun && !!firstStepTimesByRun
    const fetching = isMetricsFetching

    sortedRuns.forEach((run, index) => {
      if (index >= 20) return
      const metrics = metricsByRunPath.get(run.runPath) ?? []
      const stepMap = new Map<number, number>()

      metrics.forEach((m) => {
        stepSet.add(m.step)
        stepMap.set(m.step, m.value)
      })

      runData.set(index, stepMap)

      if (useTimeAxis) {
        const stepTimes = stepTimesByRun?.get(run.runPath)
        const firstStepTime = firstStepTimesByRun?.get(run.runPath)
        if (stepTimes && firstStepTime !== undefined) {
          const relativeTimes = new Map<number, number>()
          const timeToStep = new Map<number, number>()
          stepTimes.forEach((time, step) => {
            const relativeTime = time - firstStepTime
            relativeTimes.set(step, relativeTime)
            timeToStep.set(relativeTime, step)
          })
          runStepTimes.set(index, relativeTimes)
          timeToStepByRun.set(index, timeToStep)
        }
      }
    })

    // Combined refetching: true only if all fetching sources are refetching (have data)
    const computeIsRefetching = () => {
      if (!fetching && !isStepTimesFetching) return false
      if (fetching && !isMetricsRefetching) return false
      if (isStepTimesFetching && !isStepTimesRefetching) return false
      return true
    }

    if (stepSet.size === 0) {
      return {
        uplotData: null,
        seriesConfig: [],
        isFetching: fetching || isStepTimesFetching,
        isRefetching: computeIsRefetching(),
        hasData: false,
        timeToStepByRun,
        outlierBounds: null,
        minStep: null,
      }
    }

    // Find the minimum step across all runs (the "first step")
    const minStep = Math.min(...Array.from(stepSet))

    let sortedSteps = Array.from(stepSet).sort((a, b) => a - b)

    // Apply maxStepLimit filter
    if (maxStepLimit != null && !useTimeAxis) {
      sortedSteps = sortedSteps.filter((step) => step <= maxStepLimit)
    }

    let xValues: number[] = sortedSteps
    if (useTimeAxis) {
      const timeSet = new Set<number>()
      runData.forEach((stepMap, runIndex) => {
        const timeMap = runStepTimes.get(runIndex)
        if (!timeMap) return
        stepMap.forEach((_value, step) => {
          const time = timeMap.get(step)
          if (time !== undefined) timeSet.add(time)
        })
      })

      if (timeSet.size === 0) {
        return {
          uplotData: null,
          seriesConfig: [],
          isFetching: fetching || isStepTimesFetching,
          isRefetching: computeIsRefetching(),
          hasData: false,
          timeToStepByRun,
          outlierBounds: null,
          minStep: null,
        }
      }

      let sortedTimes = Array.from(timeSet).sort((a, b) => a - b)

      // Apply maxTimeLimit filter
      if (maxTimeLimit != null) {
        sortedTimes = sortedTimes.filter((time) => time <= maxTimeLimit)
      }

      xValues = sortedTimes
    }

    // Compute outlier bounds across all runs if ignoreOutliers is enabled
    let outlierBounds: { lower: number; upper: number } | null = null
    if (ignoreOutliers) {
      const allValues: number[] = []
      runData.forEach((stepMap) => {
        stepMap.forEach((value) => {
          allValues.push(value)
        })
      })
      outlierBounds = computeIQRBounds(allValues)
    }

    const xData = new Float64Array(xValues)
    const series: uPlot.Series[] = [
      { label: useTimeAxis ? "Time (s)" : "Step" },
    ]
    const dataArrays: (Float64Array | number[])[] = [xData]

    sortedRuns.forEach((run, index) => {
      if (index >= 20) return
      const stepMap = runData.get(index) || new Map()

      let values: (number | null)[]
      if (useTimeAxis) {
        const timeMap = runStepTimes.get(index)
        const timeToValue = new Map<number, number>()
        if (timeMap) {
          stepMap.forEach((value, step) => {
            const time = timeMap.get(step)
            if (time !== undefined) timeToValue.set(time, value)
          })
        }
        values = xValues.map((time) => timeToValue.get(time) ?? null)
      } else {
        values = sortedSteps.map((step) => stepMap.get(step) ?? null)
      }

      const isHovered = hoveredRunId === run.runPath
      const someRunIsHovered = hoveredRunId !== null
      let valueAlpha: number
      if (someRunIsHovered && !isHovered) {
        valueAlpha = 0.1
      } else {
        valueAlpha = showEma ? 0.3 : 1
      }

      dataArrays.push(values as number[])
      series.push({
        label: getRunDisplayName(run),
        stroke: run.color,
        width: 1,
        alpha: valueAlpha,
        spanGaps: true,
        points: { show: false },
      })
    })

    if (showEma) {
      sortedRuns.forEach((run, index) => {
        if (index >= 20) return
        const stepMap = runData.get(index) || new Map()
        const alpha = 2 / (emaSpan + 1)
        let ema: number | null = null

        let emaValues: (number | null)[]
        if (useTimeAxis) {
          const timeMap = runStepTimes.get(index)
          const emaByTime = new Map<number, number>()
          sortedSteps.forEach((step) => {
            const value = stepMap.get(step)
            if (value !== undefined) {
              // Skip outliers when computing EMA if ignoreOutliers is enabled
              const isOutlier =
                outlierBounds &&
                (value < outlierBounds.lower || value > outlierBounds.upper)
              // Skip first step when computing EMA if ignoreFirstStep is enabled
              const isFirstStep = ignoreFirstStep && step === minStep
              if (!isOutlier && !isFirstStep) {
                const nextEma =
                  ema === null ? value : alpha * value + (1 - alpha) * ema
                ema = nextEma
              }
              // Always record the current EMA value for this time point
              const time = timeMap?.get(step)
              if (time !== undefined && ema !== null) {
                emaByTime.set(time, ema)
              }
            }
          })
          emaValues = xValues.map((time) => emaByTime.get(time) ?? null)
        } else {
          const values: (number | null)[] = []
          sortedSteps.forEach((step) => {
            const value = stepMap.get(step)
            if (value !== undefined) {
              // Skip outliers when computing EMA if ignoreOutliers is enabled
              const isOutlier =
                outlierBounds &&
                (value < outlierBounds.lower || value > outlierBounds.upper)
              // Skip first step when computing EMA if ignoreFirstStep is enabled
              const isFirstStep = ignoreFirstStep && step === minStep
              if (!isOutlier && !isFirstStep) {
                if (ema === null) {
                  ema = value
                } else {
                  ema = alpha * value + (1 - alpha) * ema
                }
              }
              // Always push the current EMA (or null if not yet initialized)
              values.push(ema)
            } else {
              values.push(null)
            }
          })
          emaValues = values
        }

        const isHovered = hoveredRunId === run.runPath
        const someRunIsHovered = hoveredRunId !== null
        let emaAlpha: number
        if (someRunIsHovered && !isHovered) {
          emaAlpha = 0.1
        } else {
          emaAlpha = 1
        }

        dataArrays.push(emaValues as number[])
        series.push({
          label: `${getRunDisplayName(run)} (EMA)`,
          stroke: run.color,
          width: 1.5,
          alpha: emaAlpha,
          spanGaps: true,
          points: { show: false },
        })
      })
    }

    return {
      uplotData: dataArrays as uPlot.AlignedData,
      seriesConfig: series,
      isFetching: fetching || isStepTimesFetching,
      isRefetching: computeIsRefetching(),
      hasData: true,
      timeToStepByRun,
      outlierBounds,
      minStep,
    }
  }, [
    sortedRuns,
    metricsByRunPath,
    isMetricsFetching,
    isMetricsRefetching,
    showEma,
    emaSpan,
    hoveredRunId,
    xAxisMode,
    stepTimesByRun,
    firstStepTimesByRun,
    isStepTimesFetching,
    isStepTimesRefetching,
    ignoreOutliers,
    ignoreFirstStep,
    maxStepLimit,
    maxTimeLimit,
  ])

  const displayStatType = statType || metricName.split("_").pop() || ""
  const isTimeAxis = xAxisMode === "time"
  const isTimingMetric =
    metricName.startsWith("timing_") &&
    metricName !== "timing_microbatch_count" &&
    !metricName.endsWith("_pct")

  const formatValue = useCallback(
    (v: number | null | undefined): string => {
      if (v === undefined || v === null) return "N/A"
      if (isTokenMetric) {
        if (displayStatType === "min" || displayStatType === "max") {
          return Math.round(v).toLocaleString()
        }
        return formatValueSmart(v)
      }
      return formatValueSmart(v)
    },
    [isTokenMetric, displayStatType],
  )

  const formatYAxisTick = useCallback(
    (v: number): string => {
      if (isTimingMetric) return formatSecondsCompact(v)
      if (Math.abs(v) >= 1000) return `${parseFloat((v / 1000).toFixed(1))}k`
      if (isTokenMetric) {
        if (displayStatType === "min" || displayStatType === "max") {
          return Math.round(v).toString()
        }
        return String(parseFloat(v.toFixed(1)))
      }
      if (Number.isInteger(v)) return v.toLocaleString()
      if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1)
      return String(parseFloat(v.toFixed(2)))
    },
    [isTokenMetric, isTimingMetric, displayStatType],
  )

  const formatXAxisTick = useCallback(
    (v: number): string => {
      if (isTimeAxis) return formatSecondsCompact(v)
      return v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toString()
    },
    [isTimeAxis],
  )

  useEffect(() => {
    if (!containerRef.current || !uplotData || !hasData) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight || 192

    let minVal = Infinity
    let maxVal = -Infinity
    for (let i = 1; i < uplotData.length; i++) {
      const arr = uplotData[i]
      for (let j = 0; j < arr.length; j++) {
        // Skip first data point (first step) for y-range when ignoreFirstStep is enabled
        if (ignoreFirstStep && j === 0) continue
        const v = arr[j]
        if (v !== null && v !== undefined) {
          // When ignoring outliers, clamp to bounds for y-range calculation
          if (ignoreOutliers && outlierBounds) {
            if (v >= outlierBounds.lower && v <= outlierBounds.upper) {
              if (v < minVal) minVal = v
              if (v > maxVal) maxVal = v
            }
          } else {
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
          }
        }
      }
    }

    // If all values were outliers, fall back to using the bounds
    if (minVal === Infinity && outlierBounds) {
      minVal = outlierBounds.lower
      maxVal = outlierBounds.upper
    }

    const range = maxVal - minVal
    // Ensure minimum padding to avoid uPlot axis calculation errors when range is tiny
    // Scale by data magnitude so small values still show detail, with absolute floor for numerical safety
    const absMax = Math.max(Math.abs(maxVal), Math.abs(minVal))
    const minPadding = Math.max(absMax * 0.05, 1e-10)
    const padding = Math.max(range * 0.1, minPadding)
    let yMin: number, yMax: number
    if (showZeroLine) {
      yMin = Math.min(minVal - padding, 0)
      yMax = Math.max(maxVal + padding, 0)
    } else {
      yMin = minVal - padding
      yMax = maxVal + padding
    }

    // Apply user-specified Y bounds
    if (minY !== null) {
      yMin = minY
    }
    if (maxY !== null) {
      yMax = maxY
    }

    const gridColor = darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(128, 128, 128, 0.15)"
    const tickLabelColor = darkMode ? "rgba(255, 255, 255, 0.65)" : "rgba(100, 100, 100, 0.9)"

    // Dynamic Y axis size calculation based on tick label width
    const calcYAxisSize = (u: uPlot, values: string[]) => {
      if (!values || values.length === 0) return 40
      const maxLen = Math.max(...values.map((v) => v.length))
      // ~7px per character for 10px font + 14px padding for ticks
      return Math.max(40, maxLen * 7 + 14)
    }

    const opts: uPlot.Options = {
      width,
      height,
      padding: [4, 8, 0, 0],
      cursor: { show: true, x: true, y: false, points: { show: false } },
      legend: { show: false },
      scales: { x: { time: false }, y: { range: [yMin, yMax] } },
      axes: [
        {
          stroke: tickLabelColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: 24,
          values: (u, vals) => vals.map(formatXAxisTick),
        },
        {
          stroke: tickLabelColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: calcYAxisSize,
          values: (u, vals) => vals.map(formatYAxisTick),
        },
      ],
      series: seriesConfig,
      hooks: {
        draw: [
          (u) => {
            if (!uplotData) return
            const xData = uplotData[0] as ArrayLike<number>
            const runCount = Math.min(sortedRuns.length, 20)
            const seriesCount = u.series.length
            const ctx = u.ctx

            ctx.save()
            for (let seriesIdx = 1; seriesIdx < seriesCount; seriesIdx++) {
              const yData = uplotData[seriesIdx] as
                | (number | null | undefined)[]
                | undefined
              if (!yData) continue

              let lastIndex = -1
              for (let i = yData.length - 1; i >= 0; i--) {
                const v = yData[i]
                if (v !== null && v !== undefined) {
                  lastIndex = i
                  break
                }
              }
              if (lastIndex === -1) continue

              const xVal = xData[lastIndex]
              const yVal = yData[lastIndex]
              if (xVal === undefined || yVal === undefined || yVal === null)
                continue

              const x = u.valToPos(Number(xVal), "x", true)
              const y = u.valToPos(Number(yVal), "y", true)
              if (!Number.isFinite(x) || !Number.isFinite(y)) continue

              const series = u.series[seriesIdx]
              const alpha = typeof series?.alpha === "number" ? series.alpha : 1
              let stroke =
                typeof series?.stroke === "string" ? series.stroke : undefined
              if (!stroke) {
                const baseIdx = seriesIdx - 1
                if (baseIdx < runCount) {
                  stroke = sortedRuns[baseIdx]?.color
                } else if (showEma) {
                  const emaIdx = baseIdx - runCount
                  if (emaIdx >= 0 && emaIdx < runCount) {
                    stroke = sortedRuns[emaIdx]?.color
                  }
                }
              }
              if (!stroke) continue

              ctx.globalAlpha = alpha
              ctx.fillStyle = stroke
              ctx.beginPath()
              ctx.arc(x, y, 4, 0, Math.PI * 2)
              ctx.fill()
            }
            ctx.restore()
          },
        ],
        setCursor: [
          (u) => {
            if (!tooltipRef.current) return
            const { left, top, idx } = u.cursor

            if (
              idx === null ||
              idx === undefined ||
              left === undefined ||
              top === undefined ||
              left < 0
            ) {
              tooltipRef.current.style.display = "none"
              return
            }

            const xValue = uplotData[0][idx]
            if (xValue === undefined) {
              tooltipRef.current.style.display = "none"
              return
            }

            // Update synced cursor when CTRL is held and this is user interaction
            if (isCtrlPressedRef.current && !isSyncingRef.current) {
              setSyncedCursor({
                xValue: Number(xValue),
                sourceChartId: chartId,
              })
            }

            const headerStep = (() => {
              if (!isTimeAxis || !timeToStepByRun) return null
              const time = Number(xValue)
              if (hoveredRunId) {
                const hoveredIndex = sortedRuns.findIndex(
                  (run) => run.runPath === hoveredRunId,
                )
                if (hoveredIndex >= 0) {
                  const step = timeToStepByRun.get(hoveredIndex)?.get(time)
                  if (step !== undefined) return step
                }
              }
              const selectedIndex = sortedRuns.findIndex(
                (run) => run.isSelected,
              )
              if (selectedIndex >= 0) {
                const step = timeToStepByRun.get(selectedIndex)?.get(time)
                if (step !== undefined) return step
              }
              let firstStep: number | null = null
              const uniqueSteps = new Set<number>()
              sortedRuns.forEach((_run, runIdx) => {
                const step = timeToStepByRun.get(runIdx)?.get(time)
                if (step !== undefined) {
                  uniqueSteps.add(step)
                  if (firstStep === null) firstStep = step
                }
              })
              if (uniqueSteps.size === 1) return firstStep
              return firstStep
            })()

            const headerLabel = isTimeAxis
              ? `Time ${formatTooltipElapsedTime(Number(xValue))}${
                  headerStep !== null
                    ? ` <span class="text-muted-foreground">(Step ${Number(
                        headerStep,
                      ).toLocaleString()})</span>`
                    : ""
                }`
              : `Step ${Number(xValue).toLocaleString()}`
            let html = `<div class="font-medium mb-1">${headerLabel}</div>`
            const numRuns = plottedRuns.length
            const xSeries = uplotData[0] as ArrayLike<number>
            const maxDistanceX = (() => {
              if (!isTimeAxis) return null
              const xScale = u.scales.x
              const minX = Number(xScale.min)
              const maxX = Number(xScale.max)
              if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return null
              const xRange = Math.abs(maxX - minX)
              if (xRange <= 0) return null
              const plotWidthPx = Math.max(1, u.bbox.width)
              return (xRange * TIME_TOOLTIP_MAX_DISTANCE_PX) / plotWidthPx
            })()

            runs.forEach((run) => {
              const runIdx = runDataIndexByRunPath.get(run.runPath)
              if (runIdx === undefined) return
              const valueIdx = runIdx + 1
              const valueSeries = uplotData[valueIdx] as
                | ArrayLike<number | null | undefined>
                | undefined
              if (!valueSeries) return

              const valuePointIdx = isTimeAxis
                ? findNearestDefinedIndex(
                    valueSeries,
                    xSeries,
                    idx,
                    maxDistanceX,
                  )
                : idx
              if (valuePointIdx === null) return

              const value = valueSeries[valuePointIdx]

              let emaValue: number | null | undefined = undefined
              if (showEma) {
                const emaIdx = numRuns + runIdx + 1
                const emaSeries = uplotData[emaIdx] as
                  | ArrayLike<number | null | undefined>
                  | undefined
                emaValue = emaSeries?.[valuePointIdx]
              }

              if (value === null || value === undefined) return

              html += `
                <div class="mt-1">
                  <div class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full shrink-0" style="background-color: ${
                      run.color
                    }"></div>
                    <span>${formatRunLabelHtml(run)}</span>
                  </div>
                  <div class="ml-4 flex items-center gap-2">
                    <span class="font-medium" style="opacity: ${
                      showEma ? 0.7 : 1
                    }">${isTimingMetric ? formatSecondsTooltipHtml(value) : `${formatValue(value)}${unit ? ` ${unit}` : ""}`}</span>
                    ${
                      showEma && emaValue !== undefined && emaValue !== null
                        ? `<span class="text-muted-foreground">(EMA: ${isTimingMetric ? formatSecondsTooltipHtml(emaValue) : formatValue(emaValue)})</span>`
                        : ""
                    }
                  </div>
                </div>
              `
            })

            tooltipRef.current.innerHTML = html
            tooltipRef.current.style.display = "block"

            const containerRect = containerRef.current!.getBoundingClientRect()
            const tooltipRect = tooltipRef.current.getBoundingClientRect()

            // Horizontal: prefer right of cursor, flip left if overflowing
            let tooltipX = containerRect.left + left + 15
            if (tooltipX + tooltipRect.width + 20 > window.innerWidth) {
              // Keep right-side alignment unchanged; when flipped left, compensate by plot offset.
              tooltipX =
                containerRect.left + left - tooltipRect.width - 15 + u.bbox.left
            }
            tooltipX = Math.max(4, tooltipX)

            // Vertical: prefer overlapping top of chart, flip to bottom if overflowing viewport
            let tooltipY = containerRect.top - tooltipRect.height + 20
            if (tooltipY < 4) {
              tooltipY = containerRect.bottom - 20
            }

            tooltipRef.current.style.left = `${tooltipX}px`
            tooltipRef.current.style.top = `${tooltipY}px`
          },
        ],
      },
    }

    if (chartRef.current) chartRef.current.destroy()
    const chart = new uPlot(opts, uplotData, container)
    chartRef.current = chart

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: newWidth, height: newHeight } = entry.contentRect
        if (chart && newWidth > 0 && newHeight > 0) {
          chart.setSize({ width: newWidth, height: newHeight })
        }
      }
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
      chart.destroy()
      chartRef.current = null
    }
  }, [
    uplotData,
    seriesConfig,
    hasData,
    formatXAxisTick,
    formatYAxisTick,
    formatValue,
    isTimeAxis,
    timeToStepByRun,
    sortedRuns,
    hoveredRunId,
    showEma,
    showZeroLine,
    unit,
    ignoreOutliers,
    outlierBounds,
    ignoreFirstStep,
    minY,
    maxY,
    runs,
    plottedRuns,
    runDataIndexByRunPath,
    chartId,
    isTimingMetric,
    setSyncedCursor,
    darkMode,
  ])

  // Sync cursor from other charts when CTRL is held
  useEffect(() => {
    // If synced cursor was cleared, hide tooltip on non-source charts
    if (!syncedCursor) {
      if (tooltipRef.current) {
        tooltipRef.current.style.display = "none"
      }
      return
    }

    if (
      syncedCursor.sourceChartId === chartId ||
      !chartRef.current ||
      !uplotData
    )
      return

    const chart = chartRef.current
    const xData = uplotData[0]
    if (!xData || xData.length === 0) return

    // Find closest index for the synced xValue
    const targetX = syncedCursor.xValue
    let closestIdx = 0
    let closestDist = Infinity
    for (let i = 0; i < xData.length; i++) {
      const dist = Math.abs(Number(xData[i]) - targetX)
      if (dist < closestDist) {
        closestDist = dist
        closestIdx = i
      }
    }

    // Convert to pixel position and set cursor
    const left = chart.valToPos(Number(xData[closestIdx]), "x")
    if (left >= 0 && left <= chart.width) {
      isSyncingRef.current = true
      chart.setCursor({ left, top: 0 })
      isSyncingRef.current = false
    }
  }, [syncedCursor, chartId, uplotData])

  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) tooltipRef.current.style.display = "none"
    // Clear synced cursor if this was the source
    if (syncedCursor?.sourceChartId === chartId) {
      setSyncedCursor(null)
    }
  }, [syncedCursor, chartId, setSyncedCursor])

  // Show opacity only for user-triggered fetches (not background polling)
  const showLoadingOpacity = isFetching && !isRefetching
  const titleText =
    label ||
    (displayStatType
      ? `${displayStatType.charAt(0).toUpperCase()}${displayStatType.slice(1)}`
      : metricName)

  return (
    <div
      ref={visibilityRef}
      className={cn(
        "group/chart rounded-lg border border-border p-3 transition-opacity bg-background h-[246px] flex flex-col",
        showLoadingOpacity && "opacity-50",
      )}
    >
      <div className="shrink-0 mb-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-0.5 min-w-0">
            {headerPrefix}
            <h4
              className="text-xs font-medium leading-snug line-clamp-2 break-words"
              title={metricName}
            >
              {titleText}
            </h4>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="h-5 px-1.5 text-[10px] rounded border border-border hover:bg-muted flex items-center gap-1 transition-all opacity-0 group-hover/chart:opacity-100">
                  <SlidersHorizontal className="h-3 w-3" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-[160px]">
                <DropdownMenuCheckboxItem
                  checked={ignoreOutliers}
                  onCheckedChange={setIgnoreOutliers}
                >
                  Ignore Outliers
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem
                  checked={ignoreFirstStep}
                  onCheckedChange={setIgnoreFirstStep}
                >
                  Ignore First Step
                </DropdownMenuCheckboxItem>
                <DropdownMenuSeparator />
                <div className="flex items-center gap-2 px-2 py-1.5">
                  <span className="text-xs text-muted-foreground font-medium w-10">Min Y</span>
                  <input
                    type="number"
                    className="w-20 h-6 px-2 text-xs rounded border border-border bg-background focus:outline-none focus:ring-1 focus:ring-ring"
                    placeholder="Auto"
                    value={minY ?? ""}
                    onChange={(e) => {
                      const val = e.target.value
                      setMinY(val === "" ? null : Number(val))
                    }}
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                  />
                </div>
                <div className="flex items-center gap-2 px-2 py-1.5">
                  <span className="text-xs text-muted-foreground font-medium w-10">Max Y</span>
                  <input
                    type="number"
                    className="w-20 h-6 px-2 text-xs rounded border border-border bg-background focus:outline-none focus:ring-1 focus:ring-ring"
                    placeholder="Auto"
                    value={maxY ?? ""}
                    onChange={(e) => {
                      const val = e.target.value
                      setMaxY(val === "" ? null : Number(val))
                    }}
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                  />
                </div>
                {availableEnvs && availableEnvs.length > 1 && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuLabel className="text-[10px] text-muted-foreground">
                      Filter by Environment
                    </DropdownMenuLabel>
                    {availableEnvs.map((envName) => (
                      <DropdownMenuCheckboxItem
                        key={envName}
                        checked={envFilters.includes(envName)}
                        onCheckedChange={() => toggleEnvFilter(envName)}
                      >
                        {envName}
                      </DropdownMenuCheckboxItem>
                    ))}
                  </>
                )}
                {availableSampleTags && Object.keys(availableSampleTags).length > 0 && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuLabel className="text-[10px] text-muted-foreground">
                      Filter by Tag
                    </DropdownMenuLabel>
                    {Object.entries(availableSampleTags).map(([tagName, tagValues]) => (
                      <DropdownMenuSub key={tagName}>
                        <DropdownMenuSubTrigger className="text-xs">
                          {tagName}
                          {tagFilters[tagName]?.length ? (
                            <span className="ml-1 text-[10px] text-muted-foreground">
                              ({tagFilters[tagName].length})
                            </span>
                          ) : null}
                        </DropdownMenuSubTrigger>
                        <DropdownMenuSubContent className="max-h-[300px] overflow-y-auto">
                          {tagValues.map((tagValue) => (
                            <DropdownMenuCheckboxItem
                              key={tagValue}
                              checked={(tagFilters[tagName] ?? []).includes(tagValue)}
                              onCheckedChange={() => toggleTagFilter(tagName, tagValue)}
                            >
                              {tagValue}
                            </DropdownMenuCheckboxItem>
                          ))}
                        </DropdownMenuSubContent>
                      </DropdownMenuSub>
                    ))}
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
            {headerSuffix}
          </div>
        </div>
        {(ignoreOutliers || ignoreFirstStep || minY !== null || maxY !== null || hasActiveTagFilters || hasActiveEnvFilters) && (
          <div className="flex items-center gap-1 mt-1 flex-wrap">
            {ignoreOutliers && (
              <FilterBadge
                label="Ignore Outliers"
                onRemove={() => setIgnoreOutliers(false)}
              />
            )}
            {ignoreFirstStep && (
              <FilterBadge
                label="Ignore First Step"
                onRemove={() => setIgnoreFirstStep(false)}
              />
            )}
            {minY !== null && (
              <FilterBadge
                label={`Min Y: ${minY}`}
                onRemove={() => setMinY(null)}
              />
            )}
            {maxY !== null && (
              <FilterBadge
                label={`Max Y: ${maxY}`}
                onRemove={() => setMaxY(null)}
              />
            )}
            {envFilters.map((envName) => (
              <FilterBadge
                key={`env:${envName}`}
                label={`env: ${envName}`}
                onRemove={() => toggleEnvFilter(envName)}
              />
            ))}
            {Object.entries(tagFilters).map(([tagName, tagValues]) =>
              tagValues.map((tagValue) => (
                <FilterBadge
                  key={`${tagName}:${tagValue}`}
                  label={`${tagName}: ${tagValue}`}
                  onRemove={() => toggleTagFilter(tagName, tagValue)}
                />
              ))
            )}
          </div>
        )}
      </div>
      {hasData ? (
        <div
          className="flex-1 min-h-0 relative bg-background rounded"
          ref={containerRef}
          onMouseLeave={handleMouseLeave}
        >
          <div
            ref={tooltipRef}
            className="fixed z-[9999] max-w-[360px] bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl border border-border pointer-events-none"
            style={{ display: "none" }}
          />
          {/* Loading indicator for background polling */}
          {isRefetching && (
            <Loader2 className="absolute bottom-0.5 left-0.5 h-3 w-3 animate-spin text-muted-foreground" />
          )}
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex items-center justify-center text-muted-foreground text-xs rounded">
          {isFetching ? "Loading..." : "No data"}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Inference Performance Metric Chart - multi-run line chart matching MetricChart style
// ============================================================================

/**
 * Parse a duration string into seconds.
 * Accepts: "30s", "10m", "2h", "1h30m", "1m30s", or a plain number (defaults to seconds).
 */
function parseDuration(input: string): number | null {
  const trimmed = input.trim()
  if (!trimmed) return null
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
  const num = parseFloat(trimmed)
  if (Number.isFinite(num) && num > 0) return Math.round(num)
  return null
}

const INFERENCE_AVG_METRICS = new Set([
  "avg_time_prefill",
  "avg_time_decode",
  "avg_time_compute_reward",
  "avg_time_queue",
  "avg_time_ttft",
  "avg_time_inference",
  "avg_time_e2e",
  "avg_time_generation",
])

interface InferencePerformanceMetricChartProps {
  runs: RunInfo[]
  shouldPoll: boolean
  inferenceMetricType: string
  label: string
  hoveredRunId?: string | null
  scrollRoot?: Element | null
  headerPrefix?: React.ReactNode
  headerSuffix?: React.ReactNode
}

function InferencePerformanceMetricChart({
  runs,
  shouldPoll,
  inferenceMetricType,
  label,
  hoveredRunId = null,
  scrollRoot = null,
  headerPrefix,
  headerSuffix,
}: InferencePerformanceMetricChartProps) {
  const darkMode = useAtomValue(darkModeAtom)
  const visibilityRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<uPlot | null>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  const [ignoreOutliers, setIgnoreOutliers] = useState(false)
  const [ignoreFirstStep, setIgnoreFirstStep] = useState(true)

  const [bucketSeconds, setBucketSeconds] = useState(60)
  const [intervalInput, setIntervalInput] = useState(formatDurationHms(60))

  const handleIntervalCommit = useCallback(
    (value: string) => {
      const parsed = parseDuration(value)
      if (parsed !== null && parsed >= 1) {
        setBucketSeconds(parsed)
        setIntervalInput(formatDurationHms(parsed))
      } else {
        setIntervalInput(formatDurationHms(bucketSeconds))
      }
    },
    [bucketSeconds],
  )

  const isVisible = useOnScreen(visibilityRef, {
    root: scrollRoot,
    threshold: 0,
  })

  const isAvgMetric = INFERENCE_AVG_METRICS.has(inferenceMetricType)

  // Sort runs: non-selected first, selected last (drawn on top)
  const sortedRuns = useMemo(() => {
    const nonSelected = runs.filter((r) => !r.isSelected)
    const selected = runs.filter((r) => r.isSelected)
    return [...nonSelected, ...selected]
  }, [runs])
  const plottedRuns = useMemo(() => sortedRuns.slice(0, 20), [sortedRuns])

  // Fetch data for each run individually (each keyed by runPath+bucketSeconds so react-query caches them)
  const infPerf0 = useInferencePerformance(plottedRuns[0]?.runPath ?? "", isVisible && !!plottedRuns[0], shouldPoll, bucketSeconds)
  const infPerf1 = useInferencePerformance(plottedRuns[1]?.runPath ?? "", isVisible && !!plottedRuns[1], shouldPoll, bucketSeconds)
  const infPerf2 = useInferencePerformance(plottedRuns[2]?.runPath ?? "", isVisible && !!plottedRuns[2], shouldPoll, bucketSeconds)
  const infPerf3 = useInferencePerformance(plottedRuns[3]?.runPath ?? "", isVisible && !!plottedRuns[3], shouldPoll, bucketSeconds)
  const infPerf4 = useInferencePerformance(plottedRuns[4]?.runPath ?? "", isVisible && !!plottedRuns[4], shouldPoll, bucketSeconds)
  const infPerf5 = useInferencePerformance(plottedRuns[5]?.runPath ?? "", isVisible && !!plottedRuns[5], shouldPoll, bucketSeconds)
  const infPerf6 = useInferencePerformance(plottedRuns[6]?.runPath ?? "", isVisible && !!plottedRuns[6], shouldPoll, bucketSeconds)
  const infPerf7 = useInferencePerformance(plottedRuns[7]?.runPath ?? "", isVisible && !!plottedRuns[7], shouldPoll, bucketSeconds)
  const infPerf8 = useInferencePerformance(plottedRuns[8]?.runPath ?? "", isVisible && !!plottedRuns[8], shouldPoll, bucketSeconds)
  const infPerf9 = useInferencePerformance(plottedRuns[9]?.runPath ?? "", isVisible && !!plottedRuns[9], shouldPoll, bucketSeconds)

  const queries = useMemo(
    () => [infPerf0, infPerf1, infPerf2, infPerf3, infPerf4, infPerf5, infPerf6, infPerf7, infPerf8, infPerf9],
    [infPerf0, infPerf1, infPerf2, infPerf3, infPerf4, infPerf5, infPerf6, infPerf7, infPerf8, infPerf9],
  )

  const isFetching = queries.some((q, i) => i < plottedRuns.length && q.isFetching)
  const isRefetching = isFetching && queries.filter((q, i) => i < plottedRuns.length && q.isFetching).every((q) => !!q.data)

  // Build uPlot data with one series per run
  // Each run's times are made relative to that run's own first_time so runs align by elapsed time
  const { uplotData, seriesConfig, hasData, outlierBounds } = useMemo(() => {
    // For each run, compute relative-time -> value map using the run's own first_time
    const runRelData: Map<number, Map<number, number>> = new Map()
    const relTimeSet = new Set<number>()
    let anyData = false

    plottedRuns.forEach((_, idx) => {
      if (idx >= 10) return
      const data = queries[idx].data
      if (!data || data.first_time === null) return
      const runFirstTime = data.first_time
      const buckets = (data as unknown as Record<string, unknown>)[inferenceMetricType] as
        | Array<{ time: number; count?: number; value?: number }>
        | undefined
      if (!buckets || buckets.length === 0) return

      // Compute the first step completion time (relative) for this run
      let firstStepRelTime: number | null = null
      if (ignoreFirstStep) {
        const stepTimes = data.step_times ?? []
        if (stepTimes.length > 0) {
          // First step_time entry is the earliest step completion
          firstStepRelTime = stepTimes[0].time - runFirstTime
        }
      }

      anyData = true
      const valueByRelTime = new Map<number, number>()
      buckets.forEach((b) => {
        const val = isAvgMetric ? b.value : b.count
        if (val !== undefined && val !== null) {
          const relTime = b.time - runFirstTime
          // Skip buckets before first step completes
          if (ignoreFirstStep && firstStepRelTime !== null && relTime < firstStepRelTime) return
          valueByRelTime.set(relTime, val)
          relTimeSet.add(relTime)
        }
      })
      runRelData.set(idx, valueByRelTime)
    })

    if (!anyData || relTimeSet.size === 0) {
      return { uplotData: null, seriesConfig: [], hasData: false, outlierBounds: null }
    }

    // Compute outlier bounds across all runs if enabled
    let bounds: { lower: number; upper: number } | null = null
    if (ignoreOutliers) {
      const allValues: number[] = []
      runRelData.forEach((valueMap) => {
        valueMap.forEach((v) => allValues.push(v))
      })
      bounds = computeIQRBounds(allValues)
    }

    const sortedRelTimes = Array.from(relTimeSet).sort((a, b) => a - b)
    const xData = new Float64Array(sortedRelTimes)

    const series: uPlot.Series[] = [{ label: "Time (s)" }]
    const dataArrays: (Float64Array | (number | null)[])[] = [xData]

    plottedRuns.forEach((run, idx) => {
      if (idx >= 10) return
      const valueByRelTime = runRelData.get(idx)

      const values: (number | null)[] = sortedRelTimes.map((t) =>
        valueByRelTime?.get(t) ?? null,
      )

      const isHovered = hoveredRunId === run.runPath
      const someRunIsHovered = hoveredRunId !== null
      const valueAlpha = someRunIsHovered && !isHovered ? 0.1 : 1

      dataArrays.push(values)
      series.push({
        label: getRunDisplayName(run),
        stroke: run.color,
        width: 1,
        alpha: valueAlpha,
        spanGaps: true,
        points: { show: false },
      })
    })

    return {
      uplotData: dataArrays as uPlot.AlignedData,
      seriesConfig: series,
      hasData: true,
      outlierBounds: bounds,
    }
  }, [plottedRuns, queries, inferenceMetricType, hoveredRunId, isAvgMetric, ignoreOutliers, ignoreFirstStep])

  const formatValue = useCallback(
    (v: number | null | undefined): string => {
      if (v === undefined || v === null) return "N/A"
      if (isAvgMetric) return formatSecondsTooltipHtml(v)
      return formatValueSmart(v)
    },
    [isAvgMetric],
  )

  const formatYAxisTick = useCallback(
    (v: number): string => {
      if (isAvgMetric) return formatSecondsCompact(v)
      if (Math.abs(v) >= 1000) return `${parseFloat((v / 1000).toFixed(1))}k`
      if (Number.isInteger(v)) return v.toLocaleString()
      return String(parseFloat(v.toFixed(2)))
    },
    [isAvgMetric],
  )

  useEffect(() => {
    if (!containerRef.current || !uplotData || !hasData || !isVisible) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight || 192

    let minVal = Infinity
    let maxVal = -Infinity
    for (let i = 1; i < uplotData.length; i++) {
      const arr = uplotData[i]
      for (let j = 0; j < arr.length; j++) {
        const v = arr[j]
        if (v !== null && v !== undefined && Number.isFinite(v)) {
          if (ignoreOutliers && outlierBounds) {
            if (v >= outlierBounds.lower && v <= outlierBounds.upper) {
              if (v < minVal) minVal = v
              if (v > maxVal) maxVal = v
            }
          } else {
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
          }
        }
      }
    }
    if (minVal === Infinity && outlierBounds) {
      minVal = outlierBounds.lower
      maxVal = outlierBounds.upper
    }
    if (minVal === Infinity) { minVal = 0; maxVal = 1 }

    const range = maxVal - minVal
    const absMax = Math.max(Math.abs(maxVal), Math.abs(minVal))
    const minPadding = Math.max(absMax * 0.05, 1e-10)
    const padding = Math.max(range * 0.1, minPadding)
    const yMin = minVal - padding
    const yMax = maxVal + padding

    const gridColor = darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(128, 128, 128, 0.15)"
    const tickLabelColor = darkMode ? "rgba(255, 255, 255, 0.65)" : "rgba(100, 100, 100, 0.9)"

    const calcYAxisSize = (_u: uPlot, values: string[]) => {
      if (!values || values.length === 0) return 40
      const maxLen = Math.max(...values.map((v) => v.length))
      return Math.max(40, maxLen * 7 + 14)
    }

    const opts: uPlot.Options = {
      width,
      height,
      padding: [4, 8, 0, 0],
      cursor: { show: true, x: true, y: false, points: { show: false } },
      legend: { show: false },
      scales: { x: { time: false }, y: { range: [yMin, yMax] } },
      axes: [
        {
          stroke: tickLabelColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: 24,
          values: (_u, vals) => vals.map((v) => formatSecondsCompact(v)),
        },
        {
          stroke: tickLabelColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: calcYAxisSize,
          values: (_u, vals) => vals.map(formatYAxisTick),
        },
      ],
      series: seriesConfig,
      hooks: {
        draw: [
          (u) => {
            if (!uplotData) return
            const ctx = u.ctx
            ctx.save()
            for (let seriesIdx = 1; seriesIdx < u.series.length; seriesIdx++) {
              const yData = uplotData[seriesIdx] as (number | null | undefined)[] | undefined
              if (!yData) continue
              let lastIndex = -1
              for (let i = yData.length - 1; i >= 0; i--) {
                if (yData[i] !== null && yData[i] !== undefined) { lastIndex = i; break }
              }
              if (lastIndex === -1) continue
              const xVal = uplotData[0][lastIndex]
              const yVal = yData[lastIndex]
              if (xVal === undefined || yVal === undefined || yVal === null) continue
              const x = u.valToPos(Number(xVal), "x", true)
              const y = u.valToPos(Number(yVal), "y", true)
              if (!Number.isFinite(x) || !Number.isFinite(y)) continue
              const series = u.series[seriesIdx]
              const alpha = typeof series?.alpha === "number" ? series.alpha : 1
              const stroke = typeof series?.stroke === "string" ? series.stroke : sortedRuns[seriesIdx - 1]?.color
              if (!stroke) continue
              ctx.globalAlpha = alpha
              ctx.fillStyle = stroke
              ctx.beginPath()
              ctx.arc(x, y, 4, 0, Math.PI * 2)
              ctx.fill()
            }
            ctx.restore()
          },
        ],
        setCursor: [
          (u) => {
            if (!tooltipRef.current || !uplotData) return
            const { left, top, idx } = u.cursor
            if (idx === null || idx === undefined || left === undefined || top === undefined || left < 0) {
              tooltipRef.current.style.display = "none"
              return
            }
            const xValue = uplotData[0][idx]
            if (xValue === undefined) { tooltipRef.current.style.display = "none"; return }

            const headerLabel = `Time ${formatDurationHms(Number(xValue))}`
            let html = `<div class="font-medium mb-1">${headerLabel}</div>`

            runs.forEach((run) => {
              const runIdx = plottedRuns.findIndex((r) => r.runPath === run.runPath)
              if (runIdx === -1) return
              const valueIdx = runIdx + 1
              const valueSeries = uplotData[valueIdx] as (number | null | undefined)[] | undefined
              if (!valueSeries) return
              const value = valueSeries[idx]
              if (value === null || value === undefined) return

              html += `
                <div class="mt-1">
                  <div class="flex items-center gap-2">
                    <div class="w-2 h-2 rounded-full shrink-0" style="background-color: ${run.color}"></div>
                    <span>${formatRunLabelHtml(run)}</span>
                  </div>
                  <div class="ml-4 flex items-center gap-2">
                    <span class="font-medium">${formatValue(value)}</span>
                  </div>
                </div>
              `
            })

            tooltipRef.current.innerHTML = html
            tooltipRef.current.style.display = "block"

            const containerRect = containerRef.current!.getBoundingClientRect()
            const tooltipRect = tooltipRef.current.getBoundingClientRect()
            let tooltipX = containerRect.left + left + 15
            if (tooltipX + tooltipRect.width + 20 > window.innerWidth) {
              tooltipX = containerRect.left + left - tooltipRect.width - 15 + u.bbox.left
            }
            tooltipX = Math.max(4, tooltipX)
            let tooltipY = containerRect.top - tooltipRect.height + 20
            if (tooltipY < 4) { tooltipY = containerRect.bottom - 20 }
            tooltipRef.current.style.left = `${tooltipX}px`
            tooltipRef.current.style.top = `${tooltipY}px`
          },
        ],
      },
    }

    if (chartRef.current) chartRef.current.destroy()
    const chart = new uPlot(opts, uplotData, container)
    chartRef.current = chart

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: newWidth, height: newHeight } = entry.contentRect
        if (chart && newWidth > 0 && newHeight > 0) {
          chart.setSize({ width: newWidth, height: newHeight })
        }
      }
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
      chart.destroy()
      chartRef.current = null
    }
  }, [
    uplotData,
    seriesConfig,
    hasData,
    isVisible,
    ignoreOutliers,
    outlierBounds,
    darkMode,
    sortedRuns,
    runs,
    plottedRuns,
    hoveredRunId,
    formatValue,
    formatYAxisTick,
    bucketSeconds,
  ])

  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) tooltipRef.current.style.display = "none"
  }, [])

  const showLoadingOpacity = isFetching && !isRefetching

  return (
    <div
      ref={visibilityRef}
      className={cn(
        "group/chart rounded-lg border border-border p-3 transition-opacity bg-background h-[246px] flex flex-col",
        showLoadingOpacity && "opacity-50",
      )}
    >
      <div className="shrink-0 mb-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-0.5 min-w-0">
            {headerPrefix}
            <h4
              className="text-xs font-medium leading-snug line-clamp-2 break-words"
              title={label}
            >
              {label}
            </h4>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="h-5 px-1.5 text-[10px] rounded border border-border hover:bg-muted flex items-center gap-1 transition-all opacity-0 group-hover/chart:opacity-100">
                  <SlidersHorizontal className="h-3 w-3" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-[160px]">
                <DropdownMenuCheckboxItem
                  checked={ignoreOutliers}
                  onCheckedChange={setIgnoreOutliers}
                >
                  Ignore Outliers
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem
                  checked={ignoreFirstStep}
                  onCheckedChange={setIgnoreFirstStep}
                >
                  Ignore First Step
                </DropdownMenuCheckboxItem>
                <DropdownMenuSeparator />
                <div className="flex items-center gap-2 px-2 py-1.5">
                  <span className="text-xs text-muted-foreground font-medium w-14">Interval</span>
                  <input
                    type="text"
                    className="w-20 h-6 px-2 text-xs rounded border border-border bg-background focus:outline-none focus:ring-1 focus:ring-ring"
                    placeholder="1m"
                    value={intervalInput}
                    onChange={(e) => setIntervalInput(e.target.value)}
                    onBlur={(e) => handleIntervalCommit(e.target.value)}
                    onKeyDown={(e) => {
                      e.stopPropagation()
                      if (e.key === "Enter") handleIntervalCommit(e.currentTarget.value)
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                </div>
              </DropdownMenuContent>
            </DropdownMenu>
            {headerSuffix}
          </div>
        </div>
        {(ignoreOutliers || ignoreFirstStep) && (
          <div className="flex items-center gap-1 mt-1 flex-wrap">
            {ignoreOutliers && (
              <FilterBadge
                label="Ignoring Outliers"
                onRemove={() => setIgnoreOutliers(false)}
              />
            )}
            {ignoreFirstStep && (
              <FilterBadge
                label="Ignoring First Step"
                onRemove={() => setIgnoreFirstStep(false)}
              />
            )}
          </div>
        )}
      </div>
      {hasData ? (
        <div
          className="flex-1 min-h-0 relative bg-background rounded"
          ref={containerRef}
          onMouseLeave={handleMouseLeave}
        >
          <div
            ref={tooltipRef}
            className="fixed z-[9999] max-w-[360px] bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl border border-border pointer-events-none"
            style={{ display: "none" }}
          />
          {isRefetching && (
            <Loader2 className="absolute bottom-0.5 left-0.5 h-3 w-3 animate-spin text-muted-foreground" />
          )}
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex items-center justify-center text-muted-foreground text-xs rounded">
          {isFetching ? "Loading..." : "No data"}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Inference Performance Section
// ============================================================================

interface InferencePerformanceSectionProps {
  runs: RunInfo[]
  shouldPoll: boolean
  hoveredRunId?: string | null
  scrollRoot?: Element | null
}

export function InferencePerformanceSection({
  runs,
  shouldPoll,
  hoveredRunId = null,
  scrollRoot = null,
}: InferencePerformanceSectionProps) {
  const sharedProps = { runs, shouldPoll, hoveredRunId, scrollRoot }

  return (
    <div>
      <h4 className="text-xs font-medium text-muted-foreground mb-2 mt-1">Requests Count</h4>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <InferencePerformanceMetricChart inferenceMetricType="inference_calls" label="Inference Calls" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="requests_done" label="Requests Done" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="rollouts_group_done" label="Rollouts Group Done" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="rollouts_group_done_kept" label="Rollouts Group Done Kept" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="rollouts_group_done_discarded" label="Rollouts Group Done Discarded" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="rollouts_group_done_canceled" label="Rollouts Group Done Canceled" {...sharedProps} />
      </div>
      <h4 className="text-xs font-medium text-muted-foreground mb-2 mt-4">Latency</h4>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <InferencePerformanceMetricChart inferenceMetricType="avg_time_queue" label="Avg Queue Time" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="avg_time_ttft" label="Avg Time to First Token" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="avg_time_prefill" label="Avg Prefill Time" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="avg_time_decode" label="Avg Decode Time" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="avg_time_inference" label="Avg Inference Time" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="avg_time_e2e" label="Avg E2E Latency" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="avg_time_generation" label="Avg Generation Duration" {...sharedProps} />
        <InferencePerformanceMetricChart inferenceMetricType="avg_time_compute_reward" label="Avg Compute Reward Time" {...sharedProps} />
      </div>
    </div>
  )
}

// ============================================================================
// Inference Performance Chart Card (standalone, for custom metrics view)
// ============================================================================

interface InferencePerformanceChartCardProps {
  runs: RunInfo[]
  shouldPoll: boolean
  hoveredRunId?: string | null
  scrollRoot?: Element | null
  inferenceMetricType: string
  label: string
  headerPrefix?: React.ReactNode
  headerSuffix?: React.ReactNode
}

export function InferencePerformanceChartCard({
  runs,
  shouldPoll,
  hoveredRunId = null,
  scrollRoot = null,
  inferenceMetricType,
  label,
  headerPrefix,
  headerSuffix,
}: InferencePerformanceChartCardProps) {
  return (
    <InferencePerformanceMetricChart
      runs={runs}
      shouldPoll={shouldPoll}
      hoveredRunId={hoveredRunId}
      scrollRoot={scrollRoot}
      inferenceMetricType={inferenceMetricType}
      label={label}
      headerPrefix={headerPrefix}
      headerSuffix={headerSuffix}
    />
  )
}
