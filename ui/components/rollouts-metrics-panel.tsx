
import {
  useState,
  useMemo,
  useRef,
  useEffect,
  useCallback,
  type RefObject,
} from "react"
import { createPortal } from "react-dom"
import { useAtom, useAtomValue, useSetAtom } from "jotai"
import type { SetStateAction, WritableAtom } from "jotai"
import uPlot from "uplot"
import { X, ChevronDown, PanelRightClose, Loader2, SlidersHorizontal, GripVertical, Maximize2, Minimize2 } from "lucide-react"
import {
  DndContext,
  closestCenter,
  PointerSensor,
  KeyboardSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core"
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ToggleWithInput } from "@/components/ui/toggle-with-input"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"
import {
  rolloutsSelectedMetricsAtom,
  rolloutsMetricsSidebarOpenAtom,
  rolloutsSelectedStepAtom,
  selectedRunPathAtom,
  visibleRunsAtom,
  hoveredRunIdAtom,
  darkModeAtom,
  metricsChartFiltersAtom,
  type MetricsChartFilterState,
} from "@/lib/atoms"
import {
  useStepMetricSingle,
  useEvalStepMetricSingle,
  useStepHistogram,
  useStepDistributionOverTime,
  useRuns,
  useRunSummary,
} from "@/hooks/use-run-data"
import { formatValueSmart } from "@/lib/format"
import {
  PlotSelectPopover,
  type PlotCatalogItem,
} from "@/components/custom-metrics-view"
import {
  type RunInfo,
  getRunIdFromPath,
  getRunDisplayName,
  TOOLTIP_RUN_NAME_MAX_CHARS,
  TOOLTIP_RUN_NAME_MAX_WIDTH_CLASS,
  escapeHtml,
  formatTooltipRunNameHtml,
  formatRunLabelHtml,
  InferencePerformanceChartCard,
  InferencePerformanceAreaChartCard,
  INFERENCE_PERF_AREA_VARIANTS,
  InferenceUtilizationAreaChartCard,
  INFERENCE_UTIL_AREA_VARIANTS,
  TrainerPerformanceChartCard,
  TrainerPerformanceAreaChartCard,
  TRAINER_PERF_AREA_VARIANTS,
  FilterBadge,
  DEFAULT_IGNORE_FIRST_STEP_METRICS,
  computeIQRBounds,
} from "@/components/step-metrics-charts"

// Hook to detect if element is on screen
// - isVisible: true when element is currently in viewport
// - isVisibleSticky: stays true briefly after scrolling away (for stable polling)
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
          if (stickyTimeoutRef.current) {
            clearTimeout(stickyTimeoutRef.current)
            stickyTimeoutRef.current = null
          }
          setIsVisibleSticky(true)
        } else {
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

  return isVisible || isVisibleSticky
}

// Fullscreen hook + button for chart cards
function useChartFullscreen() {
  const [isFullscreen, setIsFullscreen] = useState(false)

  useEffect(() => {
    if (!isFullscreen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsFullscreen(false)
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [isFullscreen])

  const toggleFullscreen = useCallback(() => setIsFullscreen((f) => !f), [])

  const fullscreenPortal = useCallback(
    (content: React.ReactElement) =>
      isFullscreen ? createPortal(content, document.body) : content,
    [isFullscreen],
  )

  return { isFullscreen, toggleFullscreen, fullscreenPortal }
}

function FullscreenButton({
  isFullscreen,
  onClick,
}: {
  isFullscreen: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "h-5 px-1.5 text-[10px] rounded border border-border hover:bg-muted flex items-center gap-1 transition-all",
        isFullscreen
          ? "opacity-100"
          : "opacity-0 group-hover/card:opacity-100",
      )}
      title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
    >
      {isFullscreen ? (
        <Minimize2 className="h-3 w-3" />
      ) : (
        <Maximize2 className="h-3 w-3" />
      )}
    </button>
  )
}

// Base metric configuration (static metrics)
const BASE_METRICS_CONFIG = {
  rewards: {
    label: "Reward",
    metrics: [
      { prefix: "reward_sum", label: "Reward" },
      {
        prefix: "reward_gini_mean",
        label: "Reward Sparsity (Gini)",
        simple: true,
      },
    ],
  },
  advantage: {
    label: "Advantage",
    metrics: [{ prefix: "advantage", label: "Advantage" }],
    showZeroLine: true,
  },
  tokens: {
    label: "Rollouts",
    metrics: [
      { prefix: "length_prompt", label: "Tokens Prompt" },
      { prefix: "length_completion", label: "Tokens Completion" },
      { prefix: "length_sum", label: "Tokens Total" },
    ],
    unit: "tokens",
  },
  rollouts_general: {
    label: "Rollouts",
    metrics: [
      {
        prefix: "stop_reason_length_pct",
        label: "% Stop Reason = Length",
        simple: true,
        group: "General",
      },
      {
        prefix: "group_length_gini_mean",
        label: "Group Completion Length Gini",
        simple: true,
        group: "General",
      },
      {
        prefix: "group_length_max_median_ratio_mean",
        label: "Group Completion Length Max/Median",
        simple: true,
        group: "General",
      },
    ],
  },
}

// Discarded-only metrics (these already have discarded_ prefix in their metric key)
const DISCARDED_STATS_CONFIG = {
  discarded_stats: {
    label: "Discarded Rollouts",
    metrics: [
      { prefix: "discarded_count", label: "Discarded Count", simple: true, group: "General" },
      {
        prefix: "discarded_zero_advantage_pct",
        label: "Discarded Zero Advantage %",
        simple: true,
        group: "General",
      },
      { prefix: "discarded_max_async_pct", label: "Discarded Max Async %", simple: true, group: "General" },
      {
        prefix: "discarded_stop_reason_length_pct",
        label: "Discarded % Stop Reason = Length",
        simple: true,
        group: "General",
      },
      {
        prefix: "discarded_group_length_gini_mean",
        label: "Discarded Group Completion Length Gini",
        simple: true,
        group: "General",
      },
      {
        prefix: "discarded_group_length_max_median_ratio_mean",
        label: "Discarded Group Completion Length Max/Median",
        simple: true,
        group: "General",
      },
      {
        prefix: "discarded_zero_advantage_all_zero_pct",
        label: "Discarded Zero Adv (All Reward = 0) %",
        simple: true,
        group: "General",
      },
      {
        prefix: "discarded_zero_advantage_all_positive_pct",
        label: "Discarded Zero Adv (All Reward > 0) %",
        simple: true,
        group: "General",
      },
      {
        prefix: "discarded_zero_advantage_mean_reward",
        label: "Discarded Zero Adv Mean Reward",
        simple: true,
        group: "General",
      },
      { prefix: "canceled_count", label: "Canceled Count", simple: true, group: "Canceled" },
    ],
  },
  discarded_tokens: {
    label: "Discarded Rollouts",
    metrics: [
      { prefix: "discarded_length_prompt", label: "Discarded Tokens Prompt" },
      { prefix: "discarded_length_completion", label: "Discarded Tokens Completion" },
      { prefix: "discarded_length_sum", label: "Discarded Tokens Total" },
    ],
    unit: "tokens",
  },
}

const TIMELINE_CONFIG = {
  timeline_trainer_full_step: {
    label: "Timeline Trainer",
    metrics: [
      { prefix: "timing_step_total", label: "Time per Step", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_step_active", label: "Time per Step Active", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_microbatch_count", label: "Microbatches per Step", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_forward_total", label: "Timing Forward", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_backward_total", label: "Timing Backward", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_loss_computation_total", label: "Timing Loss Computation", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_compute_kl_total", label: "Timing Compute KL", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_compute_entropy_total", label: "Timing Compute Entropy", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_data_to_device_total", label: "Timing Data to Device", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_prepare_tensors_total", label: "Timing Prepare Tensors", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_waiting_for_data", label: "Timing Waiting for Data", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_weight_sync_trainer_total", label: "Timing Weight Broadcast (Trainer)", simple: true, group: "Full Step (Total Time)" },
      { prefix: "timing_weight_sync_inference_total", label: "Timing Weight Broadcast (Inference)", simple: true, group: "Full Step (Total Time)" },
    ],
  },
  timeline_trainer_microbatch: {
    label: "Timeline Trainer",
    metrics: [
      { prefix: "timing_forward_microbatch_mean", label: "Microbatch Timing Forward", simple: true, group: "Microbatch (Mean Time)" },
      { prefix: "timing_backward_microbatch_mean", label: "Microbatch Timing Backward", simple: true, group: "Microbatch (Mean Time)" },
      { prefix: "timing_loss_computation_microbatch_mean", label: "Microbatch Timing Loss Computation", simple: true, group: "Microbatch (Mean Time)" },
      { prefix: "timing_compute_kl_microbatch_mean", label: "Microbatch Timing Compute KL", simple: true, group: "Microbatch (Mean Time)" },
      { prefix: "timing_compute_entropy_microbatch_mean", label: "Microbatch Timing Compute Entropy", simple: true, group: "Microbatch (Mean Time)" },
      { prefix: "timing_data_to_device_microbatch_mean", label: "Microbatch Timing Data to Device", simple: true, group: "Microbatch (Mean Time)" },
      { prefix: "timing_prepare_tensors_microbatch_mean", label: "Microbatch Timing Prepare Tensors", simple: true, group: "Microbatch (Mean Time)" },
    ],
  },
  timeline_inference: {
    label: "Timeline Inference",
    metrics: [
      { prefix: "timing_save_batch_total", label: "Batch Completion (Save Batch)", simple: true, group: "Batch & Averages" },
      { prefix: "timing_avg_inference_time", label: "Avg Generation Time", simple: true, group: "Batch & Averages" },
      { prefix: "timing_avg_compute_reward_time", label: "Avg Compute Reward Time", simple: true, group: "Batch & Averages" },
      { prefix: "timing_generation_normal_pct", label: "Inference % Generation (Normal)", simple: true, group: "Time Breakdown (% of Step Time)" },
      { prefix: "timing_generation_discarded_pct", label: "Inference % Generation (Discarded)", simple: true, group: "Time Breakdown (% of Step Time)" },
      { prefix: "timing_generation_canceled_pct", label: "Inference % Generation (Canceled)", simple: true, group: "Time Breakdown (% of Step Time)" },
      { prefix: "timing_generation_all_pct", label: "Inference % Generation (All)", simple: true, group: "Time Breakdown (% of Step Time)" },
      { prefix: "timing_compute_reward_normal_pct", label: "Inference % Compute Reward (Normal)", simple: true, group: "Time Breakdown (% of Step Time)" },
      { prefix: "timing_compute_reward_discarded_pct", label: "Inference % Compute Reward (Discarded)", simple: true, group: "Time Breakdown (% of Step Time)" },
      { prefix: "timing_compute_reward_canceled_pct", label: "Inference % Compute Reward (Canceled)", simple: true, group: "Time Breakdown (% of Step Time)" },
      { prefix: "timing_compute_reward_all_pct", label: "Inference % Compute Reward (All)", simple: true, group: "Time Breakdown (% of Step Time)" },
      { prefix: "timing_idle_pct", label: "Inference % Idle Time", simple: true, group: "Time Breakdown (% of Step Time)" },
    ],
  },
}

const STAT_SUFFIXES = ["mean", "std", "min", "max"] as const
const HISTOGRAM_SUFFIX = "histogram" as const
const DISTRIBUTION_OVER_TIME_SUFFIX = "distribution_over_time" as const

type MetricOption = {
  category: string
  categoryLabel: string
  group?: string
  metricKey: string
  label: string
  prefix: string
  suffix: string
  showZeroLine?: boolean
  unit?: string
  isHistogram?: boolean
  isDistributionOverTime?: boolean
  histogramMetricType?: string // e.g., "reward_sum", "advantage", "length_prompt", etc.
  isEvalMetric?: boolean
  evalName?: string
  isInferencePerformance?: boolean
  inferenceMetricType?: string
  isInferencePerformanceArea?: boolean
  isInferenceUtilizationArea?: boolean
  inferenceAreaCategories?: string[]
  isTrainerPerformance?: boolean
  trainerMetricType?: string
  isTrainerPerformanceArea?: boolean
  trainerAreaCategories?: string[]
}

// Map metric prefix to histogram API metric_type
function getHistogramMetricType(prefix: string): string {
  // Direct mappings for base metrics
  if (prefix === "reward_sum") return "reward_sum"
  if (prefix === "advantage") return "advantage"
  if (prefix === "length_prompt") return "length_prompt"
  if (prefix === "length_completion") return "length_completion"
  if (prefix === "length_sum") return "length_sum"
  // Dynamic rewards: reward_accuracy -> reward_accuracy (API expects reward_<name>)
  if (prefix.startsWith("reward_")) return prefix
  return prefix
}

type MetricConfigEntry = {
  label: string
  metrics: Array<{ prefix: string; label: string; simple?: boolean; group?: string }>
  showZeroLine?: boolean
  unit?: string
}

function pushMetricOptions(
  options: MetricOption[],
  category: string,
  categoryConfig: MetricConfigEntry,
) {
  for (const metric of categoryConfig.metrics) {
    const isSimple = "simple" in metric && metric.simple
    const metricGroup = "group" in metric ? (metric as { group?: string }).group : undefined
    const showZeroLine =
      "showZeroLine" in categoryConfig ? categoryConfig.showZeroLine : false
    const unit = "unit" in categoryConfig ? categoryConfig.unit : undefined

    if (isSimple) {
      options.push({
        category,
        categoryLabel: categoryConfig.label,
        group: metricGroup,
        metricKey: metric.prefix,
        label: metric.label,
        prefix: metric.prefix,
        suffix: "",
        showZeroLine,
        unit,
      })
      continue
    }

    for (const suffix of STAT_SUFFIXES) {
      const metricKey = `${metric.prefix}_${suffix}`
      const suffixLabel = suffix.charAt(0).toUpperCase() + suffix.slice(1)
      options.push({
        category,
        categoryLabel: categoryConfig.label,
        group: metricGroup,
        metricKey,
        label: `${metric.label} - ${suffixLabel}`,
        prefix: metric.prefix,
        suffix,
        showZeroLine,
        unit,
      })
    }

    const histogramKey = `${metric.prefix}_${HISTOGRAM_SUFFIX}`
    options.push({
      category,
      categoryLabel: categoryConfig.label,
      group: metricGroup,
      metricKey: histogramKey,
      label: `${metric.label} - Histogram`,
      prefix: metric.prefix,
      suffix: HISTOGRAM_SUFFIX,
      showZeroLine,
      unit,
      isHistogram: true,
      histogramMetricType: getHistogramMetricType(metric.prefix),
    })

    const distOverTimeKey = `${metric.prefix}_${DISTRIBUTION_OVER_TIME_SUFFIX}`
    options.push({
        category,
        categoryLabel: categoryConfig.label,
        group: metricGroup,
        metricKey: distOverTimeKey,
        label: `${metric.label} - Distribution Over Time`,
        prefix: metric.prefix,
        suffix: DISTRIBUTION_OVER_TIME_SUFFIX,
        showZeroLine,
        unit,
        isDistributionOverTime: true,
        histogramMetricType: getHistogramMetricType(metric.prefix),
      })
  }
}

// Build flattened list of all available metrics with descriptive names
// Order matches buildPlotCatalog: Custom → Reward → Samples → Advantage → Evals → Rollouts → Discarded → Timeline
function buildMetricOptions(
  rewardNames: string[],
  customMetricSections: Record<string, Record<string, string[]>> = {},
  evalsList: Array<{ eval_name: string; available_rollout_metric_names: string[] }> = [],
): MetricOption[] {
  const options: MetricOption[] = []

  // 1. Custom sections
  for (const [sectionName, groups] of Object.entries(customMetricSections)) {
    for (const [groupName, metricNames] of Object.entries(groups)) {
      const category = groupName
        ? `custom_${sectionName}_${groupName}`
        : `custom_${sectionName}`
      pushMetricOptions(options, category, {
        label: formatRewardNameForLabel(sectionName),
        metrics: metricNames.map((name) => ({
          prefix: name,
          label: formatRewardNameForLabel(name),
          simple: true,
          group: groupName ? formatRewardNameForLabel(groupName) : undefined,
        })),
      })
    }
  }

  // 2. Reward
  pushMetricOptions(options, "rewards", BASE_METRICS_CONFIG.rewards)

  // 3. Samples Metrics
  if (rewardNames.length > 0) {
    pushMetricOptions(options, "rollout_metrics", {
      label: "Samples Metrics",
      metrics: rewardNames.flatMap((name) => [
        {
          prefix: `reward_${name}`,
          label: formatRewardNameForLabel(name),
        },
        {
          prefix: `reward_${name}_gini_mean`,
          label: `${formatRewardNameForLabel(name)} Sparsity (Gini)`,
          simple: true,
        },
      ]),
    })
  }

  // 4. Advantage
  pushMetricOptions(options, "advantage", BASE_METRICS_CONFIG.advantage)

  // 5. Evals
  options.push(...buildEvalSectionOptions(evalsList))

  // 6. Rollouts (tokens + general)
  pushMetricOptions(options, "tokens", BASE_METRICS_CONFIG.tokens)
  pushMetricOptions(options, "rollouts_general", BASE_METRICS_CONFIG.rollouts_general)

  // 7. Discarded Rollouts
  for (const [key, config] of Object.entries(DISCARDED_STATS_CONFIG)) {
    pushMetricOptions(options, key, config)
  }

  // 8. Timeline Trainer & Inference
  for (const [key, config] of Object.entries(TIMELINE_CONFIG)) {
    pushMetricOptions(options, key, config)
  }

  // 9. Inference Performance
  for (const m of [
    { key: "inference_calls", label: "Inference Calls", inferenceMetricType: "inference_calls" },
    { key: "requests_done", label: "Requests Done", inferenceMetricType: "requests_done" },
    { key: "rollouts_group_done", label: "Rollouts Group Done", inferenceMetricType: "rollouts_group_done" },
    { key: "rollouts_group_done_kept", label: "Rollouts Group Done Kept", inferenceMetricType: "rollouts_group_done_kept" },
    { key: "rollouts_group_done_discarded", label: "Rollouts Group Done Discarded", inferenceMetricType: "rollouts_group_done_discarded" },
    { key: "rollouts_group_done_canceled", label: "Rollouts Group Done Canceled", inferenceMetricType: "rollouts_group_done_canceled" },
  ]) {
    options.push({
      category: "inference_performance",
      categoryLabel: "Inference Performance",
      metricKey: m.key,
      label: m.label,
      prefix: m.key,
      suffix: "",
      isInferencePerformance: true,
      inferenceMetricType: m.inferenceMetricType,
    })
  }

  // Inference Performance - Utilization (Area)
  for (const variant of INFERENCE_UTIL_AREA_VARIANTS) {
    options.push({
      category: "inference_utilization_area",
      categoryLabel: "Inference Performance",
      group: "Breakdown (Area)",
      metricKey: variant.key,
      label: variant.label,
      prefix: variant.key,
      suffix: "",
      isInferenceUtilizationArea: true,
      inferenceAreaCategories: [...variant.categories],
    })
  }

  // Inference Performance - Breakdown (Area)
  for (const variant of INFERENCE_PERF_AREA_VARIANTS) {
    options.push({
      category: "inference_performance_area",
      categoryLabel: "Inference Performance",
      group: "Breakdown (Area)",
      metricKey: variant.key,
      label: variant.label,
      prefix: variant.key,
      suffix: "",
      isInferencePerformanceArea: true,
      inferenceAreaCategories: [...variant.categories],
    })
  }

  // Trainer Performance - Area Charts (first, to match Metrics page order)
  for (const variant of TRAINER_PERF_AREA_VARIANTS) {
    options.push({
      category: "trainer_performance_area",
      categoryLabel: "Trainer Performance",
      group: "Time Breakdown (Area)",
      metricKey: variant.key,
      label: variant.label,
      prefix: variant.key,
      suffix: "",
      isTrainerPerformanceArea: true,
      trainerAreaCategories: [...variant.categories],
    })
  }

  // Trainer Performance - % Time per Category (line charts)
  for (const m of [
    { key: "trainer_perf_idle", label: "Trainer % Idle", trainerMetricType: "idle" },
    { key: "trainer_perf_working", label: "Trainer % Working", trainerMetricType: "working" },
    { key: "trainer_perf_working_except_weight_sync", label: "Trainer % Working (excl. Weight Sync)", trainerMetricType: "working_except_weight_sync" },
    { key: "trainer_perf_forward", label: "Trainer % Forward", trainerMetricType: "forward" },
    { key: "trainer_perf_backward", label: "Trainer % Backward", trainerMetricType: "backward" },
    { key: "trainer_perf_optimizer", label: "Trainer % Optimizer", trainerMetricType: "optimizer" },
    { key: "trainer_perf_loss_computation", label: "Trainer % Loss Computation", trainerMetricType: "loss_computation" },
    { key: "trainer_perf_weight_broadcast", label: "Trainer % Weight Sync", trainerMetricType: "weight_broadcast" },
    { key: "trainer_perf_data_wait", label: "Trainer % Data Wait", trainerMetricType: "data_wait" },
    { key: "trainer_perf_grad_clip", label: "Trainer % Grad Clip", trainerMetricType: "grad_clip" },
    { key: "trainer_perf_grad_norm", label: "Trainer % Grad Norm", trainerMetricType: "grad_norm" },
    { key: "trainer_perf_checkpoint", label: "Trainer % Checkpoint", trainerMetricType: "checkpoint" },
  ]) {
    options.push({
      category: "trainer_performance",
      categoryLabel: "Trainer Performance",
      group: "% Time per Category",
      metricKey: m.key,
      label: m.label,
      prefix: m.key,
      suffix: "",
      isTrainerPerformance: true,
      trainerMetricType: m.trainerMetricType,
    })
  }

  return options
}

// Build eval section options matching buildPlotCatalog's Evals section + histogram + distribution_over_time
function buildEvalSectionOptions(
  evalsList: Array<{ eval_name: string; available_rollout_metric_names: string[] }>,
): MetricOption[] {
  const options: MetricOption[] = []

  for (const evalEntry of evalsList) {
    const evalName = evalEntry.eval_name
    const evalPrefix = `eval/${evalName}/`

    // Reward metrics per eval
    for (const metricName of evalEntry.available_rollout_metric_names) {
      const metricLabel = formatRewardNameForLabel(metricName)
      const groupLabel = `${evalName} / ${metricLabel}`
      const basePrefix = `${evalPrefix}reward_${metricName}`
      const category = "evals"

      for (const suffix of STAT_SUFFIXES) {
        const suffixLabel = suffix.charAt(0).toUpperCase() + suffix.slice(1)
        options.push({
          category,
          categoryLabel: "Evals",
          group: groupLabel,
          metricKey: `${basePrefix}_${suffix}`,
          label: `${evalName} ${metricLabel} - ${suffixLabel}`,
          prefix: basePrefix,
          suffix,
          isEvalMetric: true,
          evalName,
        })
      }

      options.push({
        category,
        categoryLabel: "Evals",
        group: groupLabel,
        metricKey: `${basePrefix}_${HISTOGRAM_SUFFIX}`,
        label: `${evalName} ${metricLabel} - Histogram`,
        prefix: basePrefix,
        suffix: HISTOGRAM_SUFFIX,
        isHistogram: true,
        histogramMetricType: basePrefix,
        isEvalMetric: true,
        evalName,
      })

      options.push({
        category,
        categoryLabel: "Evals",
        group: groupLabel,
        metricKey: `${basePrefix}_${DISTRIBUTION_OVER_TIME_SUFFIX}`,
        label: `${evalName} ${metricLabel} - Distribution Over Time`,
        prefix: basePrefix,
        suffix: DISTRIBUTION_OVER_TIME_SUFFIX,
        isDistributionOverTime: true,
        histogramMetricType: basePrefix,
        isEvalMetric: true,
        evalName,
      })
    }

    // Token metrics per eval
    for (const tokenPrefix of ["length_completion", "length_prompt", "length_sum"]) {
      const tokenLabel = tokenPrefix === "length_completion"
        ? "Tokens Completion"
        : tokenPrefix === "length_prompt"
          ? "Tokens Prompt"
          : "Tokens Total"
      const groupLabel = `${evalName} / ${tokenLabel}`
      const basePrefix = `${evalPrefix}${tokenPrefix}`
      const category = "evals"

      for (const suffix of STAT_SUFFIXES) {
        const suffixLabel = suffix.charAt(0).toUpperCase() + suffix.slice(1)
        options.push({
          category,
          categoryLabel: "Evals",
          group: groupLabel,
          metricKey: `${basePrefix}_${suffix}`,
          label: `${evalName} ${tokenLabel} - ${suffixLabel}`,
          prefix: basePrefix,
          suffix,
          unit: "tokens",
          isEvalMetric: true,
          evalName,
        })
      }

      options.push({
        category,
        categoryLabel: "Evals",
        group: groupLabel,
        metricKey: `${basePrefix}_${HISTOGRAM_SUFFIX}`,
        label: `${evalName} ${tokenLabel} - Histogram`,
        prefix: basePrefix,
        suffix: HISTOGRAM_SUFFIX,
        unit: "tokens",
        isHistogram: true,
        histogramMetricType: basePrefix,
        isEvalMetric: true,
        evalName,
      })

      options.push({
        category,
        categoryLabel: "Evals",
        group: groupLabel,
        metricKey: `${basePrefix}_${DISTRIBUTION_OVER_TIME_SUFFIX}`,
        label: `${evalName} ${tokenLabel} - Distribution Over Time`,
        prefix: basePrefix,
        suffix: DISTRIBUTION_OVER_TIME_SUFFIX,
        unit: "tokens",
        isDistributionOverTime: true,
        histogramMetricType: basePrefix,
        isEvalMetric: true,
        evalName,
      })
    }
  }

  return options
}

function formatRewardNameForLabel(name: string): string {
  // Convert snake_case to Title Case
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ")
}

// Group metrics by category for the select dropdown
function getMetricsByCategory(options: MetricOption[]) {
  const grouped: Record<string, { label: string; metrics: MetricOption[] }> = {}

  for (const option of options) {
    if (!grouped[option.category]) {
      grouped[option.category] = {
        label: option.categoryLabel,
        metrics: [],
      }
    }
    grouped[option.category].metrics.push(option)
  }

  return grouped
}


type AtomWithState<T> = WritableAtom<T, [SetStateAction<T>], void>

export interface EvalConfig {
  prefix: string
  rewardNames: string[]
  selectedSampleIdx?: number | null
}

interface RolloutsMetricsPanelProps {
  currentStep?: number | null
  maxSelectableStep?: number | null
  openAtom?: AtomWithState<boolean>
  selectedMetricsAtom?: AtomWithState<string[]>
  selectedStepAtom?: AtomWithState<number | null>
  /** When true, includes discarded stats in the default metric selection */
  includeDiscardedDefaults?: boolean
  /** When provided, show eval-specific metrics (same structure as training but with prefix, no reward/advantage/training categories) */
  evalConfig?: EvalConfig
}

// Convert MetricOption[] to PlotCatalogItem[] for PlotSelectPopover
function metricOptionsToCatalog(options: MetricOption[]): PlotCatalogItem[] {
  // Build group labels from prefixes (strip suffix from label)
  const prefixGroupLabels = new Map<string, string>()
  for (const opt of options) {
    if (prefixGroupLabels.has(opt.prefix)) continue
    if (opt.suffix) {
      const sepIdx = opt.label.lastIndexOf(" - ")
      prefixGroupLabels.set(
        opt.prefix,
        sepIdx > 0 ? opt.label.substring(0, sepIdx) : opt.label,
      )
    } else {
      prefixGroupLabels.set(opt.prefix, opt.label)
    }
  }

  return options.map((opt) => ({
    section: opt.categoryLabel,
    group: opt.group || (opt.suffix ? (() => {
      const g = prefixGroupLabels.get(opt.prefix)
      return g && g !== opt.categoryLabel ? g : undefined
    })() : undefined),
    metricKey: opt.metricKey,
    label: opt.label,
    plotType: opt.isInferencePerformance
      ? ("inference_performance" as const)
      : opt.isInferencePerformanceArea
        ? ("inference_performance_area" as const)
        : opt.isInferenceUtilizationArea
          ? ("inference_utilization_area" as const)
          : opt.isTrainerPerformance
            ? ("trainer_performance" as const)
            : opt.isTrainerPerformanceArea
              ? ("trainer_performance_area" as const)
              : opt.isHistogram
                ? ("histogram" as const)
                : opt.isDistributionOverTime
                  ? ("distribution_over_time" as const)
                  : opt.isEvalMetric
                    ? ("eval_metric" as const)
                    : ("step_metric" as const),
    evalName: opt.evalName,
    histogramMetricType: opt.histogramMetricType,
    inferenceMetricType: opt.inferenceMetricType,
    inferenceAreaCategories: opt.inferenceAreaCategories,
    trainerMetricType: opt.trainerMetricType,
    trainerAreaCategories: opt.trainerAreaCategories,
    simple: !opt.suffix,
  }))
}

// Toggle button component to open the sidebar
export function RolloutsMetricsSidebarToggle({
  openAtom = rolloutsMetricsSidebarOpenAtom,
}: {
  openAtom?: AtomWithState<boolean>
} = {}) {
  const [isOpen, setIsOpen] = useAtom(openAtom)
  const selectedRunPath = useAtomValue(selectedRunPathAtom)

  if (!selectedRunPath || isOpen) return null

  return (
    <Button variant="outline" size="sm" onClick={() => setIsOpen(true)}>
      Metrics
    </Button>
  )
}

function SortableChartWrapper({
  id,
  children,
}: {
  id: string
  children: (dragHandle: React.ReactNode) => React.ReactNode
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id })

  const style: React.CSSProperties = {
    transform: transform
      ? `translate3d(${Math.round(transform.x)}px, ${Math.round(transform.y)}px, 0)`
      : undefined,
    transition,
    opacity: isDragging ? 0.4 : 1,
    position: "relative" as const,
    zIndex: isDragging ? 50 : undefined,
  }

  const dragHandle = (
    <button
      {...attributes}
      {...listeners}
      className="-ml-1.5 cursor-grab active:cursor-grabbing p-0.5 rounded hover:bg-muted transition-all shrink-0 opacity-0 group-hover/card:opacity-100 group-hover/chart:opacity-100"
    >
      <GripVertical className="h-3 w-3 text-muted-foreground" />
    </button>
  )

  return (
    <div ref={setNodeRef} style={style}>
      {children(dragHandle)}
    </div>
  )
}

export function RolloutsMetricsPanel({
  currentStep,
  maxSelectableStep,
  openAtom = rolloutsMetricsSidebarOpenAtom,
  selectedMetricsAtom = rolloutsSelectedMetricsAtom,
  selectedStepAtom = rolloutsSelectedStepAtom,
  includeDiscardedDefaults = false,
  evalConfig,
}: RolloutsMetricsPanelProps) {
  const [isOpen, setIsOpen] = useAtom(openAtom)
  const [selectedMetrics, setSelectedMetrics] = useAtom(selectedMetricsAtom)

  // Selected vs All mode (only used in eval mode)
  const [showSelectedSample, setShowSelectedSample] = useState(false)
  const effectiveShowSelected =
    showSelectedSample && evalConfig?.selectedSampleIdx != null

  // EMA settings (local state for this panel)
  const [showEma, setShowEma] = useState(false)
  const [emaSpan, setEmaSpan] = useState(10)
  const [emaSpanInput, setEmaSpanInput] = useState("10")
  const selectedRunPath = useAtomValue(selectedRunPathAtom)
  const visibleRuns = useAtomValue(visibleRunsAtom)
  const hoveredRunId = useAtomValue(hoveredRunIdAtom)
  // Keep metrics updated like /metrics
  const shouldPoll = true

  const isEvalMode = !!evalConfig

  const { data: runsData } = useRuns()
  const { data: summaryData } = useRunSummary(
    selectedRunPath || "",
    !!selectedRunPath,
    shouldPoll,
  )

  const availableRewardNames = useMemo(
    () => summaryData?.available_rollout_metric_names ?? [],
    [summaryData?.available_rollout_metric_names],
  )
  const customMetricSections = useMemo(
    () => summaryData?.step_metrics_info?.custom_metric_sections ?? {},
    [summaryData?.step_metrics_info?.custom_metric_sections],
  )
  const evalsList = useMemo(
    () => summaryData?.eval_info?.evals ?? [],
    [summaryData?.eval_info?.evals],
  )

  // Apply default metrics on first load if none are selected (training mode only)
  const defaultsAppliedRef = useRef(false)
  useEffect(() => {
    if (isEvalMode) return
    if (defaultsAppliedRef.current) return
    if (selectedMetrics.length > 0) {
      defaultsAppliedRef.current = true
      return
    }
    if (!summaryData) return
    defaultsAppliedRef.current = true

    const defaults = [
      "reward_sum_mean",
      "reward_sum_distribution_over_time",
      "advantage_distribution_over_time",
      "length_completion_mean",
      "length_completion_std",
      "length_completion_min",
      "length_completion_max",
      "length_completion_distribution_over_time",
      ...availableRewardNames.flatMap((name) => [
        `reward_${name}_mean`,
        `reward_${name}_distribution_over_time`,
      ]),
      // Include discarded stats by default when requested
      ...(includeDiscardedDefaults
        ? [
            "discarded_count",
            "discarded_zero_advantage_pct",
            "discarded_max_async_pct",
            "discarded_zero_advantage_all_zero_pct",
            "discarded_zero_advantage_all_positive_pct",
          ]
        : []),
    ]
    setSelectedMetrics(defaults)
  }, [summaryData, availableRewardNames, isEvalMode]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-add rollout metric means + distribution over time when new ones become available (training mode only)
  const knownGenMetricsRef = useRef<Set<string>>(new Set())
  useEffect(() => {
    if (isEvalMode) return
    if (availableRewardNames.length === 0) return
    const newKeys: string[] = []
    for (const name of availableRewardNames) {
      if (!knownGenMetricsRef.current.has(name)) {
        const meanKey = `reward_${name}_mean`
        const distKey = `reward_${name}_distribution_over_time`
        if (!selectedMetrics.includes(meanKey)) newKeys.push(meanKey)
        if (!selectedMetrics.includes(distKey)) newKeys.push(distKey)
      }
      knownGenMetricsRef.current.add(name)
    }
    if (newKeys.length > 0) {
      setSelectedMetrics((prev) => [...prev, ...newKeys])
    }
  }, [availableRewardNames, isEvalMode]) // eslint-disable-line react-hooks/exhaustive-deps

  // Eval mode: auto-select defaults when eval config changes (new eval name or new reward names)
  const lastEvalConfigKeyRef = useRef<string>("")
  useEffect(() => {
    if (!isEvalMode || !evalConfig) return
    const configKey = `${evalConfig.prefix}|${evalConfig.rewardNames.join(",")}`
    if (configKey === lastEvalConfigKeyRef.current) return
    lastEvalConfigKeyRef.current = configKey
    const p = evalConfig.prefix
    const defaults = [
      `${p}length_completion_mean`,
      `${p}length_completion_std`,
      `${p}length_completion_min`,
      `${p}length_completion_max`,
      `${p}length_completion_distribution_over_time`,
      ...evalConfig.rewardNames.flatMap((name) => [
        `${p}reward_${name}_mean`,
        `${p}reward_${name}_distribution_over_time`,
      ]),
    ]
    setSelectedMetrics(defaults)
  }, [evalConfig, isEvalMode, setSelectedMetrics])

  // Compute runs to display with their persistent colors
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

  // Build metric options — always include all training metrics + eval section for all evals
  const allMetricOptions = useMemo(
    () => buildMetricOptions(availableRewardNames, customMetricSections, evalsList),
    [availableRewardNames, customMetricSections, evalsList],
  )

  const metricsByCategory = useMemo(
    () => getMetricsByCategory(allMetricOptions),
    [allMetricOptions],
  )

  // Filter selected metrics to only include valid ones that exist in current options,
  // preserving the original index in selectedMetrics for stable removal of duplicates.
  // Also assign stable sortable IDs using occurrence-counting so dnd-kit can track
  // items correctly across reorders (positional indices break drag animations).
  const validSelectedEntries = useMemo(() => {
    const validKeys = new Set(allMetricOptions.map((m) => m.metricKey))
    const counts: Record<string, number> = {}
    return selectedMetrics
      .map((key, idx) => {
        const count = counts[key] || 0
        counts[key] = count + 1
        return { metricKey: key, originalIndex: idx, sortableId: `${key}::${count}` }
      })
      .filter(({ metricKey }) => validKeys.has(metricKey))
  }, [selectedMetrics, allMetricOptions])

  const validSelectedMetrics = useMemo(
    () => validSelectedEntries.map((e) => e.metricKey),
    [validSelectedEntries],
  )

  // Build catalog for PlotSelectPopover (same data as allMetricOptions, in catalog format)
  const sidebarCatalog = useMemo(
    () => metricOptionsToCatalog(allMetricOptions),
    [allMetricOptions],
  )

  // Build existingPlots for PlotSelectPopover "already added" detection
  const existingPlotsForPopover = useMemo(() => {
    return validSelectedMetrics.map((key) => {
      const config = allMetricOptions.find((m) => m.metricKey === key)
      return {
        id: key,
        metricKey: key,
        label: config?.label || key,
        plotType: (config?.isInferencePerformance
          ? "inference_performance"
          : config?.isInferencePerformanceArea
            ? "inference_performance_area"
            : config?.isInferenceUtilizationArea
              ? "inference_utilization_area"
              : config?.isTrainerPerformance
                ? "trainer_performance"
                : config?.isTrainerPerformanceArea
                  ? "trainer_performance_area"
                  : config?.isHistogram
                    ? "histogram"
                    : config?.isDistributionOverTime
                      ? "distribution_over_time"
                      : config?.isEvalMetric
                        ? "eval_metric"
                        : "step_metric") as "step_metric" | "eval_metric" | "histogram" | "distribution_over_time" | "inference_performance" | "inference_performance_area" | "inference_utilization_area" | "trainer_performance" | "trainer_performance_area",
      }
    })
  }, [validSelectedMetrics, allMetricOptions])

  const handleAddMetric = (metricKey: string) => {
    if (!metricKey) return
    setSelectedMetrics((prev) => [...prev, metricKey])
  }

  const handleRemoveMetricAt = (index: number) => {
    setSelectedMetrics((prev) => prev.filter((_, i) => i !== index))
  }

  // Drag-and-drop sensors and handler for plot reordering
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
      const activeEntry = validSelectedEntries.find((e) => e.sortableId === active.id)
      const overEntry = validSelectedEntries.find((e) => e.sortableId === over.id)
      if (!activeEntry || !overEntry) return
      setSelectedMetrics((prev) =>
        arrayMove(prev, activeEntry.originalIndex, overEntry.originalIndex),
      )
    },
    [setSelectedMetrics, validSelectedEntries],
  )

  const getMetricConfig = (metricKey: string) => {
    return allMetricOptions.find((m) => m.metricKey === metricKey)
  }

  // Scroll root for IntersectionObserver (lazy-fetch only visible charts)
  const [scrollRoot, setScrollRoot] = useState<Element | null>(null)
  const scrollAreaRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (!node || !isOpen) {
        setScrollRoot(null)
        return
      }
      const viewport = node.querySelector(
        '[data-slot="scroll-area-viewport"]',
      )
      if (viewport) setScrollRoot(viewport as Element)
    },
    [isOpen],
  )

  // Don't render if no run selected or sidebar is closed
  if (!selectedRunPath || !isOpen) {
    return null
  }

  return (
    <div className="w-[320px] border-l border-border bg-card/50 backdrop-blur flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-border flex items-center justify-between shrink-0">
        <h2 className="font-semibold text-sm">
          {isEvalMode ? "Eval Metrics" : "Training Metrics"}
        </h2>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setIsOpen(false)}
        >
          <PanelRightClose className="h-4 w-4" />
        </Button>
      </div>

      {/* Controls: metrics select + EMA */}
      <div className="px-4 py-2 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          {/* Metrics select popover */}
          <PlotSelectPopover
            catalog={sidebarCatalog}
            onSelect={(item) => handleAddMetric(item.metricKey)}
            existingPlots={existingPlotsForPopover}
            allowDuplicates
          >
            <button
              type="button"
              className={cn(
                "flex flex-1 items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs transition-colors select-none min-h-[28px]",
                "border-input bg-transparent hover:bg-accent/50",
                validSelectedMetrics.length > 0
                  ? "text-foreground"
                  : "text-muted-foreground",
              )}
            >
              <span className="font-medium">Charts</span>
              {validSelectedMetrics.length > 0 && (
                <span className="text-[10px] bg-accent text-accent-foreground rounded px-1.5 py-0 font-medium">
                  {validSelectedMetrics.length}
                </span>
              )}
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground shrink-0 ml-auto" />
            </button>
          </PlotSelectPopover>

          {/* EMA toggle */}
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
        </div>
      </div>

      {/* Charts */}
      <ScrollArea ref={scrollAreaRef} className="flex-1 min-h-0">
        <div className="p-4 space-y-4">
          {/* All/Selected toggle (eval mode only) */}
          {isEvalMode && (
            <div className="flex items-center gap-1 mb-3">
              <Button
                variant={showSelectedSample ? "outline" : "default"}
                size="sm"
                className="h-6 px-2.5 text-xs"
                onClick={() => setShowSelectedSample(false)}
              >
                All
              </Button>
              <Button
                variant={showSelectedSample ? "default" : "outline"}
                size="sm"
                className="h-6 px-2.5 text-xs"
                onClick={() => setShowSelectedSample(true)}
                disabled={evalConfig?.selectedSampleIdx == null}
              >
                Selected
              </Button>
            </div>
          )}

          {validSelectedMetrics.length === 0 ? (
            <div className="h-32 flex items-center justify-center text-muted-foreground text-sm text-center px-4">
              Click on a metric above to add charts
            </div>
          ) : (
            <DndContext
              sensors={plotSensors}
              collisionDetection={closestCenter}
              onDragEnd={handlePlotDragEnd}
            >
              <SortableContext
                items={validSelectedEntries.map(({ sortableId }) => sortableId)}
                strategy={verticalListSortingStrategy}
              >
                <div className="space-y-4">
                  {validSelectedEntries.map(({ metricKey, originalIndex, sortableId }) => {
                    const config = getMetricConfig(metricKey)

                    // In eval "Selected" mode: skip histograms, distributions, and
                    // non-mean stat suffixes — only show one raw-value chart per prefix
                    if (effectiveShowSelected) {
                      if (
                        config?.isHistogram ||
                        config?.isDistributionOverTime
                      )
                        return null
                      if (
                        config?.suffix &&
                        config.suffix !== "mean" &&
                        config.suffix !== ""
                      )
                        return null
                    }

                    // In eval "Selected" mode, inject /s/{sampleIdx}/ into eval metric keys
                    const evalSamplePrefix =
                      isEvalMode && effectiveShowSelected
                        ? `/s/${evalConfig!.selectedSampleIdx}/`
                        : "/"
                    const sampleFilteredKey =
                      isEvalMode && metricKey.startsWith("eval/")
                        ? metricKey.replace(
                            /^(eval\/[^/]+)\//,
                            `$1${evalSamplePrefix}`,
                          )
                        : metricKey

                    const fetchKey = sampleFilteredKey

                    const rawMetricType = config?.histogramMetricType || ""
                    const fetchMetricType =
                      isEvalMode && rawMetricType.startsWith("eval/")
                        ? rawMetricType.replace(
                            /^(eval\/[^/]+)\//,
                            `$1${evalSamplePrefix}`,
                          )
                        : rawMetricType

                    // In "Selected" mode, strip " - Mean" from labels
                    const chartLabel =
                      effectiveShowSelected && config?.suffix === "mean"
                        ? (config.label || metricKey).replace(
                            / - Mean$/,
                            "",
                          )
                        : config?.label || metricKey

                    // Render inference performance chart
                    if (config?.isInferencePerformance && config.inferenceMetricType) {
                      const removeBtn = (
                        <button
                          onClick={() => handleRemoveMetricAt(originalIndex)}
                          className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-muted/50 opacity-0 group-hover/card:opacity-100 group-hover/chart:opacity-100 transition-opacity"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      )
                      return (
                        <SortableChartWrapper key={sortableId} id={sortableId}>
                          {(dragHandle) => (
                            <InferencePerformanceChartCard
                              runs={runsToDisplay}
                              shouldPoll={shouldPoll}
                              hoveredRunId={hoveredRunId}
                              scrollRoot={scrollRoot}
                              inferenceMetricType={config.inferenceMetricType!}
                              label={chartLabel}
                              headerPrefix={dragHandle}
                              headerSuffix={removeBtn}
                            />
                          )}
                        </SortableChartWrapper>
                      )
                    }

                    // Render inference performance area chart
                    if (config?.isInferencePerformanceArea && config.inferenceAreaCategories) {
                      const removeBtn = (
                        <button
                          onClick={() => handleRemoveMetricAt(originalIndex)}
                          className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-muted/50 opacity-0 group-hover/card:opacity-100 group-hover/chart:opacity-100 transition-opacity"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      )
                      return (
                        <SortableChartWrapper key={sortableId} id={sortableId}>
                          {(dragHandle) => (
                            <InferencePerformanceAreaChartCard
                              runs={runsToDisplay}
                              shouldPoll={shouldPoll}
                              scrollRoot={scrollRoot}
                              label={chartLabel}
                              categories={config.inferenceAreaCategories!}
                              headerPrefix={dragHandle}
                              headerSuffix={removeBtn}
                            />
                          )}
                        </SortableChartWrapper>
                      )
                    }

                    // Render inference utilization area chart
                    if (config?.isInferenceUtilizationArea && config.inferenceAreaCategories) {
                      const removeBtn = (
                        <button
                          onClick={() => handleRemoveMetricAt(originalIndex)}
                          className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-muted/50 opacity-0 group-hover/card:opacity-100 group-hover/chart:opacity-100 transition-opacity"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      )
                      return (
                        <SortableChartWrapper key={sortableId} id={sortableId}>
                          {(dragHandle) => (
                            <InferenceUtilizationAreaChartCard
                              runs={runsToDisplay}
                              shouldPoll={shouldPoll}
                              scrollRoot={scrollRoot}
                              label={chartLabel}
                              categories={config.inferenceAreaCategories!}
                              headerPrefix={dragHandle}
                              headerSuffix={removeBtn}
                            />
                          )}
                        </SortableChartWrapper>
                      )
                    }

                    // Render trainer performance chart
                    if (config?.isTrainerPerformance && config.trainerMetricType) {
                      const removeBtn = (
                        <button
                          onClick={() => handleRemoveMetricAt(originalIndex)}
                          className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-muted/50 opacity-0 group-hover/card:opacity-100 group-hover/chart:opacity-100 transition-opacity"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      )
                      return (
                        <SortableChartWrapper key={sortableId} id={sortableId}>
                          {(dragHandle) => (
                            <TrainerPerformanceChartCard
                              runs={runsToDisplay}
                              shouldPoll={shouldPoll}
                              hoveredRunId={hoveredRunId}
                              scrollRoot={scrollRoot}
                              trainerMetricType={config.trainerMetricType!}
                              label={chartLabel}
                              headerPrefix={dragHandle}
                              headerSuffix={removeBtn}
                            />
                          )}
                        </SortableChartWrapper>
                      )
                    }

                    // Render trainer performance area chart
                    if (config?.isTrainerPerformanceArea && config.trainerAreaCategories) {
                      const removeBtn = (
                        <button
                          onClick={() => handleRemoveMetricAt(originalIndex)}
                          className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-muted/50 opacity-0 group-hover/card:opacity-100 group-hover/chart:opacity-100 transition-opacity"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      )
                      return (
                        <SortableChartWrapper key={sortableId} id={sortableId}>
                          {(dragHandle) => (
                            <TrainerPerformanceAreaChartCard
                              runs={runsToDisplay}
                              shouldPoll={shouldPoll}
                              scrollRoot={scrollRoot}
                              label={chartLabel}
                              categories={config.trainerAreaCategories!}
                              headerPrefix={dragHandle}
                              headerSuffix={removeBtn}
                            />
                          )}
                        </SortableChartWrapper>
                      )
                    }

                    // Render histogram chart for histogram metrics
                    if (config?.isHistogram) {
                      return (
                        <SortableChartWrapper key={sortableId} id={sortableId}>
                          {(dragHandle) => (
                            <HistogramChart
                              runPath={selectedRunPath}
                              step={currentStep}
                              metricType={fetchMetricType}
                              label={chartLabel}
                              showZeroLine={config?.showZeroLine}
                              unit={config?.unit}
                              isTokenMetric={config?.category === "tokens"}
                              onRemove={() => handleRemoveMetricAt(originalIndex)}
                              scrollRoot={scrollRoot}
                              headerPrefix={dragHandle}
                            />
                          )}
                        </SortableChartWrapper>
                      )
                    }

                    // Render distribution over time chart
                    if (config?.isDistributionOverTime) {
                      return (
                        <SortableChartWrapper key={sortableId} id={sortableId}>
                          {(dragHandle) => (
                            <DistributionOverTimeChart
                              runPath={selectedRunPath}
                              metricType={fetchMetricType}
                              label={chartLabel}
                              showZeroLine={config?.showZeroLine}
                              unit={config?.unit}
                              isTokenMetric={config?.category === "tokens"}
                              currentStep={currentStep}
                              shouldPoll={shouldPoll}
                              onRemove={() => handleRemoveMetricAt(originalIndex)}
                              selectedStepAtom={selectedStepAtom}
                              maxSelectableStep={maxSelectableStep}
                              scrollRoot={scrollRoot}
                              headerPrefix={dragHandle}
                            />
                          )}
                        </SortableChartWrapper>
                      )
                    }

                    return (
                      <SortableChartWrapper key={sortableId} id={sortableId}>
                        {(dragHandle) => (
                          <MetricChart
                            runs={runsToDisplay}
                            shouldPoll={shouldPoll}
                            metricKey={fetchKey}
                            label={chartLabel}
                            showZeroLine={config?.showZeroLine}
                            unit={config?.unit}
                            isTokenMetric={config?.category === "tokens"}
                            showEma={showEma}
                            emaSpan={emaSpan}
                            currentStep={currentStep}
                            onRemove={() => handleRemoveMetricAt(originalIndex)}
                            hoveredRunId={hoveredRunId}
                            selectedStepAtom={selectedStepAtom}
                            maxSelectableStep={maxSelectableStep}
                            scrollRoot={scrollRoot}
                            filterKey={`${metricKey}-${originalIndex}`}
                            headerPrefix={dragHandle}
                          />
                        )}
                      </SortableChartWrapper>
                    )
                  })}
                </div>
              </SortableContext>
            </DndContext>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}

// Individual Metric Chart Component using uPlot
interface MetricChartProps {
  runs: RunInfo[]
  shouldPoll: boolean
  metricKey: string
  label: string
  showZeroLine?: boolean
  unit?: string
  isTokenMetric?: boolean
  showEma?: boolean
  emaSpan?: number
  currentStep?: number | null
  onRemove: () => void
  hoveredRunId?: string | null
  selectedStepAtom?: AtomWithState<number | null>
  maxSelectableStep?: number | null
  scrollRoot?: Element | null
  filterKey?: string
  headerPrefix?: React.ReactNode
}

function parseEvalMetricKey(metricName: string): {
  isEval: boolean
  evalName: string
  rawMetricName: string
  sampleIdx?: number
} {
  const sampleMatch = metricName.match(/^eval\/([^/]+)\/s\/(\d+)\/(.+)$/)
  if (sampleMatch) {
    return {
      isEval: true,
      evalName: sampleMatch[1],
      rawMetricName: sampleMatch[3],
      sampleIdx: parseInt(sampleMatch[2], 10),
    }
  }
  const match = metricName.match(/^eval\/([^/]+)\/(.+)$/)
  if (match) {
    return { isEval: true, evalName: match[1], rawMetricName: match[2] }
  }
  return { isEval: false, evalName: "", rawMetricName: metricName }
}

function useRunMetricData(
  runPath: string,
  metricName: string,
  enabled: boolean,
  shouldPoll: boolean,
) {
  const { isEval, evalName, rawMetricName, sampleIdx } =
    parseEvalMetricKey(metricName)

  const trainResult = useStepMetricSingle(
    runPath,
    metricName,
    enabled && !isEval,
    shouldPoll,
  )
  const evalResult = useEvalStepMetricSingle(
    runPath,
    evalName,
    rawMetricName,
    enabled && isEval,
    shouldPoll,
    sampleIdx,
  )

  return isEval ? evalResult : trainResult
}

function MetricChart({
  runs,
  shouldPoll,
  metricKey,
  label,
  showZeroLine,
  unit,
  isTokenMetric,
  showEma = false,
  emaSpan = 10,
  currentStep,
  onRemove,
  hoveredRunId = null,
  selectedStepAtom = rolloutsSelectedStepAtom,
  maxSelectableStep,
  scrollRoot = null,
  filterKey: filterKeyProp,
  headerPrefix,
}: MetricChartProps) {
  const setSelectedStep = useSetAtom(selectedStepAtom)
  const darkMode = useAtomValue(darkModeAtom)
  const { isFullscreen, toggleFullscreen, fullscreenPortal } = useChartFullscreen()
  const visibilityRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<uPlot | null>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const cursorIdxRef = useRef<number | null>(null)

  // --- Chart filter state ---
  const filterKey = filterKeyProp ?? metricKey
  const [metricsChartFilters, setMetricsChartFilters] = useAtom(metricsChartFiltersAtom)
  const shouldDefaultIgnoreFirstStep = DEFAULT_IGNORE_FIRST_STEP_METRICS.has(metricKey)
  const metricFilters = metricsChartFilters[filterKey]
  const ignoreOutliers = metricFilters?.ignoreOutliers ?? false
  const ignoreFirstStep = metricFilters?.ignoreFirstStep ?? shouldDefaultIgnoreFirstStep
  const minY = metricFilters?.minY ?? null
  const maxY = metricFilters?.maxY ?? null

  const updateMetricFilters = useCallback(
    (updater: (current: MetricsChartFilterState) => MetricsChartFilterState) => {
      setMetricsChartFilters((prev) => {
        const defaults: MetricsChartFilterState = {
          ignoreOutliers: false,
          ignoreFirstStep: shouldDefaultIgnoreFirstStep,
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
          Object.keys(next.tagFilters ?? {}).length === 0
        if (isAtDefaultValues) {
          if (!Object.prototype.hasOwnProperty.call(prev, filterKey)) return prev
          const rest = { ...prev }
          delete rest[filterKey]
          return rest
        }
        return { ...prev, [filterKey]: next }
      })
    },
    [filterKey, setMetricsChartFilters, shouldDefaultIgnoreFirstStep],
  )

  const setIgnoreOutliers = useCallback(
    (checked: boolean) => updateMetricFilters((c) => ({ ...c, ignoreOutliers: checked })),
    [updateMetricFilters],
  )
  const setIgnoreFirstStep = useCallback(
    (checked: boolean) => updateMetricFilters((c) => ({ ...c, ignoreFirstStep: checked })),
    [updateMetricFilters],
  )
  const setMinYFilter = useCallback(
    (v: number | null) => updateMetricFilters((c) => ({ ...c, minY: v })),
    [updateMetricFilters],
  )
  const setMaxYFilter = useCallback(
    (v: number | null) => updateMetricFilters((c) => ({ ...c, maxY: v })),
    [updateMetricFilters],
  )

  const isOnScreen = useOnScreen(visibilityRef, {
    root: scrollRoot,
    threshold: 0,
  })
  const isVisible = isOnScreen || isFullscreen

  // Fetch data for each run (only when visible)
  const run0 = useRunMetricData(
    runs[0]?.runPath || "",
    metricKey,
    isVisible && !!runs[0],
    shouldPoll,
  )
  const run1 = useRunMetricData(
    runs[1]?.runPath || "",
    metricKey,
    isVisible && !!runs[1],
    shouldPoll,
  )
  const run2 = useRunMetricData(
    runs[2]?.runPath || "",
    metricKey,
    isVisible && !!runs[2],
    shouldPoll,
  )
  const run3 = useRunMetricData(
    runs[3]?.runPath || "",
    metricKey,
    isVisible && !!runs[3],
    shouldPoll,
  )
  const run4 = useRunMetricData(
    runs[4]?.runPath || "",
    metricKey,
    isVisible && !!runs[4],
    shouldPoll,
  )
  const run5 = useRunMetricData(
    runs[5]?.runPath || "",
    metricKey,
    isVisible && !!runs[5],
    shouldPoll,
  )
  const run6 = useRunMetricData(
    runs[6]?.runPath || "",
    metricKey,
    isVisible && !!runs[6],
    shouldPoll,
  )
  const run7 = useRunMetricData(
    runs[7]?.runPath || "",
    metricKey,
    isVisible && !!runs[7],
    shouldPoll,
  )
  const run8 = useRunMetricData(
    runs[8]?.runPath || "",
    metricKey,
    isVisible && !!runs[8],
    shouldPoll,
  )
  const run9 = useRunMetricData(
    runs[9]?.runPath || "",
    metricKey,
    isVisible && !!runs[9],
    shouldPoll,
  )

  // Sort runs so the selected run is last (drawn on top in uPlot)
  const sortedRuns = useMemo(() => {
    const nonSelected = runs.filter((r) => !r.isSelected)
    const selected = runs.filter((r) => r.isSelected)
    return [...nonSelected, ...selected]
  }, [runs])
  const plottedRuns = useMemo(() => sortedRuns.slice(0, 10), [sortedRuns])
  const runDataIndexByRunPath = useMemo(() => {
    const indexByPath = new Map<string, number>()
    plottedRuns.forEach((run, idx) => {
      indexByPath.set(run.runPath, idx)
    })
    return indexByPath
  }, [plottedRuns])

  // Combine data from all runs into uPlot format
  const {
    uplotData,
    seriesConfig,
    isFetching,
    isRefetching,
    isPlaceholderData,
    hasData,
    minStep,
    maxStep,
  } = useMemo(() => {
    const runQueries = [
      run0,
      run1,
      run2,
      run3,
      run4,
      run5,
      run6,
      run7,
      run8,
      run9,
    ]

    // Build a map from runPath to query result for sorted iteration
    const queryByRunPath = new Map<string, typeof run0>()
    runs.forEach((run, i) => {
      if (i < 10) queryByRunPath.set(run.runPath, runQueries[i])
    })

    // Collect all steps and values per run (using sortedRuns order)
    const stepSet = new Set<number>()
    const runData: Map<number, Map<number, number>> = new Map()
    let fetching = false
    let refetching = false
    let placeholder = false

    sortedRuns.forEach((run, index) => {
      if (index >= 10) return
      const query = queryByRunPath.get(run.runPath)
      if (!query) return
      if (query.isFetching) fetching = true
      if (query.isRefetching) refetching = true
      if (query.isPlaceholderData) placeholder = true

      const metrics = query.data?.metrics || []
      const stepMap = new Map<number, number>()

      metrics.forEach((m) => {
        stepSet.add(m.step)
        stepMap.set(m.step, m.value)
      })

      runData.set(index, stepMap)
    })

    if (stepSet.size === 0) {
      return {
        uplotData: null,
        seriesConfig: [],
        isFetching: fetching,
        isRefetching: refetching,
        isPlaceholderData: placeholder,
        hasData: false,
        minStep: 0,
        maxStep: 0,
      }
    }

    // Sort steps
    const sortedSteps = Array.from(stepSet).sort((a, b) => a - b)

    // Build columnar data
    const xData = new Float64Array(sortedSteps)
    const series: uPlot.Series[] = [{ label: "Step" }]
    const dataArrays: (Float64Array | number[])[] = [xData]

    // Add value series for each run (sorted: selected last = drawn on top)
    sortedRuns.forEach((run, index) => {
      if (index >= 10) return
      const stepMap = runData.get(index) || new Map()
      const values: (number | null)[] = sortedSteps.map(
        (step) => stepMap.get(step) ?? null,
      )

      // Calculate opacity based on hover state
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

    // Add EMA series if enabled
    if (showEma) {
      sortedRuns.forEach((run, index) => {
        if (index >= 10) return
        const stepMap = runData.get(index) || new Map()
        const alpha = 2 / (emaSpan + 1)
        let ema: number | null = null
        const emaValues: (number | null)[] = []

        sortedSteps.forEach((step) => {
          const value = stepMap.get(step)
          if (value !== undefined) {
            if (ema === null) {
              ema = value
            } else {
              ema = alpha * value + (1 - alpha) * ema
            }
            emaValues.push(ema)
          } else {
            emaValues.push(null)
          }
        })

        // Calculate opacity based on hover state
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
      isFetching: fetching,
      isRefetching: refetching,
      isPlaceholderData: placeholder,
      hasData: true,
      minStep: sortedSteps[0],
      maxStep: sortedSteps[sortedSteps.length - 1],
    }
  }, [
    runs,
    sortedRuns,
    run0,
    run1,
    run2,
    run3,
    run4,
    run5,
    run6,
    run7,
    run8,
    run9,
    showEma,
    emaSpan,
    hoveredRunId,
  ])

  // Extract stat type for formatting
  const statType = metricKey.split("_").pop() || ""

  const formatValue = useCallback(
    (v: number | null | undefined): string => {
      if (v === undefined || v === null) return "N/A"
      if (isTokenMetric) {
        if (statType === "min" || statType === "max") {
          return Math.round(v).toLocaleString()
        }
        return formatValueSmart(v)
      }
      return formatValueSmart(v)
    },
    [isTokenMetric, statType],
  )

  const formatYAxisTick = useCallback(
    (v: number): string => {
      if (Math.abs(v) >= 1000) return `${parseFloat((v / 1000).toFixed(1))}k`
      if (isTokenMetric) {
        if (statType === "min" || statType === "max") {
          return Math.round(v).toString()
        } else {
          return String(parseFloat(v.toFixed(1)))
        }
      }
      if (Number.isInteger(v)) return v.toLocaleString()
      if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1)
      return String(parseFloat(v.toFixed(2)))
    },
    [isTokenMetric, statType],
  )

  const formatXAxisTick = useCallback((v: number): string => {
    if (v >= 1000) return `${(v / 1000).toFixed(0)}k`
    return v.toString()
  }, [])

  // Create/update chart
  useEffect(() => {
    if (!containerRef.current || !uplotData || !hasData) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight || 160

    // Compute outlier bounds if needed
    let outlierBounds: { lower: number; upper: number } | null = null
    if (ignoreOutliers) {
      const allValues: number[] = []
      for (let i = 1; i < uplotData.length; i++) {
        const arr = uplotData[i]
        const startJ = ignoreFirstStep ? 1 : 0
        for (let j = startJ; j < arr.length; j++) {
          const v = arr[j]
          if (v !== null && v !== undefined) allValues.push(v as number)
        }
      }
      outlierBounds = computeIQRBounds(allValues)
    }

    // Calculate Y domain
    let minVal = Infinity
    let maxVal = -Infinity
    for (let i = 1; i < uplotData.length; i++) {
      const arr = uplotData[i]
      const startJ = ignoreFirstStep ? 1 : 0
      for (let j = startJ; j < arr.length; j++) {
        const v = arr[j]
        if (v !== null && v !== undefined) {
          if (ignoreOutliers && outlierBounds && ((v as number) < outlierBounds.lower || (v as number) > outlierBounds.upper)) continue
          if (v < minVal) minVal = v as number
          if (v > maxVal) maxVal = v as number
        }
      }
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

    // Apply manual min/max Y overrides
    if (minY !== null) yMin = minY
    if (maxY !== null) yMax = maxY

    const gridColor = darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(128, 128, 128, 0.15)"
    const tickLabelColor = darkMode ? "rgba(255, 255, 255, 0.65)" : "rgba(100, 100, 100, 0.9)"

    // Dynamic Y axis size calculation based on tick label width
    const calcYAxisSize = (u: uPlot, values: string[]) => {
      if (!values || values.length === 0) return 36
      const maxLen = Math.max(...values.map((v) => v.length))
      // ~6px per character for 10px font + 12px padding for ticks
      return Math.max(36, maxLen * 6 + 12)
    }

    // Check if currentStep is in range for highlight
    const showStepHighlight =
      currentStep != null &&
      Number.isFinite(minStep) &&
      Number.isFinite(maxStep) &&
      currentStep >= minStep &&
      currentStep <= maxStep

    const opts: uPlot.Options = {
      width,
      height,
      padding: [4, 4, 0, 0],
      cursor: {
        show: true,
        x: true,
        y: false,
        points: {
          show: false,
        },
      },
      legend: {
        show: false,
      },
      scales: {
        x: {
          time: false,
        },
        y: {
          range: [yMin, yMax],
        },
      },
      axes: [
        {
          stroke: tickLabelColor,
          grid: {
            stroke: gridColor,
            width: 1,
          },
          ticks: {
            stroke: gridColor,
            width: 1,
          },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: 24,
          values: (u, vals) => vals.map(formatXAxisTick),
        },
        {
          stroke: tickLabelColor,
          grid: {
            stroke: gridColor,
            width: 1,
          },
          ticks: {
            stroke: gridColor,
            width: 1,
          },
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
            // Draw current step highlight line
            if (showStepHighlight && currentStep != null) {
              const ctx = u.ctx
              const x = u.valToPos(currentStep, "x", true)
              const y0 = u.valToPos(yMax, "y", true)
              const y1 = u.valToPos(yMin, "y", true)

              ctx.save()
              ctx.strokeStyle = darkMode ? "rgba(255, 255, 255, 0.2)" : "rgba(128, 128, 128, 0.25)"
              ctx.lineWidth = 2
              ctx.beginPath()
              ctx.moveTo(x, y0)
              ctx.lineTo(x, y1)
              ctx.stroke()
              ctx.restore()
            }

            // Draw zero line if needed
            if (showZeroLine && yMin < 0 && yMax > 0) {
              const ctx = u.ctx
              const y = u.valToPos(0, "y", true)
              const x0 = u.valToPos(u.scales.x.min!, "x", true)
              const x1 = u.valToPos(u.scales.x.max!, "x", true)

              ctx.save()
              ctx.strokeStyle = darkMode ? "rgba(255, 255, 255, 0.35)" : "rgba(128, 128, 128, 0.5)"
              ctx.lineWidth = 1
              ctx.setLineDash([5, 5])
              ctx.beginPath()
              ctx.moveTo(x0, y)
              ctx.lineTo(x1, y)
              ctx.stroke()
              ctx.restore()
            }
          },
        ],
        setCursor: [
          (u) => {
            if (!tooltipRef.current) return
            const { left, idx } = u.cursor

            // Store cursor index for click handling
            cursorIdxRef.current = idx ?? null

            if (
              idx === null ||
              idx === undefined ||
              left === undefined ||
              left < 0
            ) {
              tooltipRef.current.style.display = "none"
              return
            }

            const step = uplotData[0][idx]
            if (step === undefined) {
              tooltipRef.current.style.display = "none"
              return
            }

            // Build tooltip content
            let html = `<div class="font-medium mb-1">Step ${Number(
              step,
            ).toLocaleString()}</div>`

            const numRuns = plottedRuns.length
            runs.forEach((run) => {
              const runIdx = runDataIndexByRunPath.get(run.runPath)
              if (runIdx === undefined) return

              const valueIdx = runIdx + 1
              const value = uplotData[valueIdx]?.[idx]

              let emaValue: number | null | undefined = undefined
              if (showEma) {
                const emaIdx = numRuns + runIdx + 1
                emaValue = uplotData[emaIdx]?.[idx]
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
                    }">${formatValue(value)}${unit ? ` ${unit}` : ""}</span>
                    ${
                      showEma && emaValue !== undefined && emaValue !== null
                        ? `<span class="text-muted-foreground">(EMA: ${formatValue(
                            emaValue,
                          )})</span>`
                        : ""
                    }
                  </div>
                </div>
              `
            })

            tooltipRef.current.innerHTML = html
            tooltipRef.current.style.display = "block"

            // Detect fixed positioning offset (backdrop-filter ancestors shift the coordinate system)
            tooltipRef.current.style.left = "0px"
            tooltipRef.current.style.top = "0px"
            const fixedOrigin = tooltipRef.current.getBoundingClientRect()

            const containerRect = containerRef.current!.getBoundingClientRect()

            // Horizontal: prefer right of cursor, flip left if overflowing
            let tooltipX = containerRect.left + left + 15
            if (tooltipX + fixedOrigin.width + 20 > window.innerWidth) {
              // Keep right-side alignment unchanged; when flipped left, compensate by plot offset.
              tooltipX =
                containerRect.left + left - fixedOrigin.width - 15 + u.bbox.left
            }
            tooltipX = Math.max(4, tooltipX)

            // Vertical: prefer overlapping top of chart, flip to bottom if overflowing viewport
            let tooltipY = containerRect.top - fixedOrigin.height + 20
            if (tooltipY < 4) {
              tooltipY = containerRect.top + 8
            }

            tooltipRef.current.style.left = `${tooltipX - fixedOrigin.left}px`
            tooltipRef.current.style.top = `${tooltipY - fixedOrigin.top}px`
          },
        ],
      },
    }

    // Destroy existing chart
    if (chartRef.current) {
      chartRef.current.destroy()
    }

    // Create new chart
    const chart = new uPlot(opts, uplotData, container)
    chartRef.current = chart

    // Handle click for step selection - use tracked cursor index
    const handleClick = () => {
      const idx = cursorIdxRef.current
      if (idx !== null && idx !== undefined) {
        let step = uplotData[0][idx]
        if (step !== undefined && !isNaN(step)) {
          // Clamp to max selectable step if the clicked step exceeds the run's max
          if (maxSelectableStep != null && step > maxSelectableStep) {
            step = maxSelectableStep
          }
          setSelectedStep(step)
        }
      }
    }
    container.addEventListener("click", handleClick)

    // Handle resize
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
      container.removeEventListener("click", handleClick)
      resizeObserver.disconnect()
      chart.destroy()
      chartRef.current = null
    }
  }, [
    uplotData,
    seriesConfig,
    hasData,
    showZeroLine,
    formatYAxisTick,
    formatXAxisTick,
    formatValue,
    sortedRuns,
    showEma,
    unit,
    currentStep,
    minStep,
    maxStep,
    setSelectedStep,
    maxSelectableStep,
    runs,
    plottedRuns,
    runDataIndexByRunPath,
    darkMode,
    ignoreOutliers,
    ignoreFirstStep,
    minY,
    maxY,
    isFullscreen,
  ])

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) {
      tooltipRef.current.style.display = "none"
    }
  }, [])

  const showLoadingOpacity = isFetching && (!isRefetching || isPlaceholderData)
  const hasActiveFilters = ignoreOutliers || ignoreFirstStep || minY !== null || maxY !== null

  return fullscreenPortal(
    <div
      ref={visibilityRef}
      className={cn(
        "group/card bg-background",
        isFullscreen
          ? "fixed inset-0 left-56 z-50 p-6 flex flex-col"
          : "rounded-lg border border-border p-3 transition-opacity",
        !isFullscreen && showLoadingOpacity && "opacity-50",
      )}
    >
      <div className="shrink-0 mb-2">
        <div className="flex items-center justify-between gap-2">
          {!isFullscreen && headerPrefix}
          <div className="flex-1 min-w-0">
            <h4
              className={cn("font-medium leading-snug line-clamp-2 break-words", isFullscreen ? "text-sm" : "text-xs")}
              title={label}
            >
              {label}
            </h4>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="h-5 px-1.5 text-[10px] rounded border border-border hover:bg-muted flex items-center gap-1 transition-all opacity-0 group-hover/card:opacity-100">
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
                      setMinYFilter(val === "" ? null : Number(val))
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
                      setMaxYFilter(val === "" ? null : Number(val))
                    }}
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                  />
                </div>
              </DropdownMenuContent>
            </DropdownMenu>
            <FullscreenButton isFullscreen={isFullscreen} onClick={toggleFullscreen} />
            {!isFullscreen && (
              <button
                onClick={onRemove}
                className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-muted/50 opacity-0 group-hover/card:opacity-100 transition-opacity"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>
        {hasActiveFilters && (
          <div className="flex items-center gap-1 mt-1 flex-wrap">
            {ignoreOutliers && (
              <FilterBadge label="Ignore Outliers" onRemove={() => setIgnoreOutliers(false)} />
            )}
            {ignoreFirstStep && (
              <FilterBadge label="Ignore First Step" onRemove={() => setIgnoreFirstStep(false)} />
            )}
            {minY !== null && (
              <FilterBadge label={`Min Y: ${minY}`} onRemove={() => setMinYFilter(null)} />
            )}
            {maxY !== null && (
              <FilterBadge label={`Max Y: ${maxY}`} onRemove={() => setMaxYFilter(null)} />
            )}
          </div>
        )}
      </div>
      {hasData ? (
        <div
          className={cn("cursor-pointer relative bg-background rounded", isFullscreen ? "flex-1 min-h-0" : "h-40")}
          ref={containerRef}
          onMouseLeave={handleMouseLeave}
        >
          {/* Custom tooltip */}
          <div
            ref={tooltipRef}
            className="fixed z-[9999] max-w-[360px] bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl border border-border pointer-events-none"
            style={{ display: "none" }}
          />
          {isRefetching && !isPlaceholderData && (
            <Loader2 className="absolute bottom-0.5 left-0.5 h-3 w-3 animate-spin text-muted-foreground" />
          )}
        </div>
      ) : (
        <div className={cn("flex items-center justify-center text-muted-foreground text-xs rounded", isFullscreen ? "flex-1 min-h-0" : "h-40")}>
          {isFetching ? "Loading..." : "No data"}
        </div>
      )}
    </div>
  )
}

// Histogram Chart Component
interface HistogramChartProps {
  runPath: string
  step: number | null | undefined
  metricType: string
  label: string
  showZeroLine?: boolean
  unit?: string
  isTokenMetric?: boolean
  onRemove: () => void
  scrollRoot?: Element | null
  headerPrefix?: React.ReactNode
}

// Compute histogram bins from values
function computeHistogramBins(
  values: number[],
  numBins: number = 20,
): { binEdges: number[]; counts: number[] } {
  if (values.length === 0) {
    return { binEdges: [], counts: [] }
  }

  const min = Math.min(...values)
  const max = Math.max(...values)

  // Handle case where all values are the same
  if (min === max) {
    return { binEdges: [min - 0.5, min + 0.5], counts: [values.length] }
  }

  const binWidth = (max - min) / numBins
  const binEdges: number[] = []
  const counts: number[] = new Array(numBins).fill(0)

  // Create bin edges
  for (let i = 0; i <= numBins; i++) {
    binEdges.push(min + i * binWidth)
  }

  // Count values in each bin
  for (const value of values) {
    let binIndex = Math.floor((value - min) / binWidth)
    // Handle edge case for max value
    if (binIndex >= numBins) binIndex = numBins - 1
    counts[binIndex]++
  }

  return { binEdges, counts }
}

function HistogramChart({
  runPath,
  step,
  metricType,
  label,
  showZeroLine,
  isTokenMetric,
  onRemove,
  scrollRoot = null,
  headerPrefix,
}: HistogramChartProps) {
  const darkMode = useAtomValue(darkModeAtom)
  const visibilityRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<uPlot | null>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)

  const isVisible = useOnScreen(visibilityRef, {
    root: scrollRoot,
    threshold: 0,
  })

  // Fetch histogram data (only when visible)
  const { data, isFetching, isRefetching, isPlaceholderData } =
    useStepHistogram(
      runPath,
      step ?? null,
      metricType,
      isVisible &&
        !!runPath &&
        step !== null &&
        step !== undefined &&
        !!metricType,
    )

  // Compute histogram bins
  const { histogramData, hasData } = useMemo(() => {
    const values = data?.values ?? []
    if (values.length === 0) {
      return { histogramData: null, hasData: false }
    }

    const { binEdges, counts } = computeHistogramBins(values, 25)

    // Create uPlot-compatible data: x values are bin centers, y values are counts
    const binCenters = binEdges
      .slice(0, -1)
      .map((edge, i) => (edge + binEdges[i + 1]) / 2)

    return {
      histogramData: {
        binEdges,
        binCenters,
        counts,
        totalCount: values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        mean: values.reduce((a, b) => a + b, 0) / values.length,
      },
      hasData: true,
    }
  }, [data?.values])

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

  // Create/update chart
  useEffect(() => {
    if (!containerRef.current || !histogramData || !hasData) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight || 160

    // Prepare data for uPlot
    const xData = new Float64Array(histogramData.binCenters)
    const yData = new Float64Array(histogramData.counts)

    const uplotData: uPlot.AlignedData = [xData, yData]

    // Calculate Y domain
    const maxCount = Math.max(...histogramData.counts)
    const yMax = maxCount * 1.1

    const gridColor = darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(128, 128, 128, 0.15)"
    const tickLabelColor = darkMode ? "rgba(255, 255, 255, 0.65)" : "rgba(100, 100, 100, 0.9)"
    const barColor = "rgba(59, 130, 246, 0.7)" // Blue bars

    // Calculate bar width based on bin width
    const binWidth = histogramData.binEdges[1] - histogramData.binEdges[0]

    // Dynamic Y axis size calculation based on tick label width
    const calcYAxisSize = (u: uPlot, values: string[]) => {
      if (!values || values.length === 0) return 36
      const maxLen = Math.max(...values.map((v) => v.length))
      // ~6px per character for 10px font + 12px padding for ticks
      return Math.max(36, maxLen * 6 + 12)
    }

    const opts: uPlot.Options = {
      width,
      height,
      padding: [4, 4, 0, 0],
      cursor: {
        show: true,
        x: true,
        y: false,
        points: {
          show: false,
        },
      },
      legend: {
        show: false,
      },
      scales: {
        x: {
          time: false,
        },
        y: {
          range: [0, yMax],
        },
      },
      axes: [
        {
          stroke: tickLabelColor,
          grid: {
            stroke: gridColor,
            width: 1,
          },
          ticks: {
            stroke: gridColor,
            width: 1,
          },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: 24,
          values: (u, vals) =>
            vals.map((v) => {
              if (Math.abs(v) >= 1000)
                return `${parseFloat((v / 1000).toFixed(1))}k`
              if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1)
              return String(parseFloat(v.toFixed(2)))
            }),
        },
        {
          stroke: tickLabelColor,
          grid: {
            stroke: gridColor,
            width: 1,
          },
          ticks: {
            stroke: gridColor,
            width: 1,
          },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: calcYAxisSize,
          values: (u, vals) =>
            vals.map((v) => {
              if (v >= 1000) return `${(v / 1000).toFixed(0)}k`
              return Math.round(v).toString()
            }),
        },
      ],
      series: [
        { label: "Value" },
        {
          label: "Count",
          stroke: barColor,
          fill: barColor,
          width: 0,
          paths: (u, seriesIdx) => {
            // Custom bar rendering
            const bars: Path2D = new Path2D()
            const { data } = u
            const xValues = data[0]
            const yValues = data[seriesIdx]

            for (let i = 0; i < xValues.length; i++) {
              const x = xValues[i]
              const y = yValues[i]

              if (y === null || y === undefined || y === 0) continue

              // Calculate pixel positions
              const x0 = u.valToPos(x - binWidth / 2, "x", true)
              const x1 = u.valToPos(x + binWidth / 2, "x", true)
              const y0 = u.valToPos(0, "y", true)
              const y1 = u.valToPos(y, "y", true)

              const barWidth = Math.max(1, x1 - x0 - 1) // Small gap between bars

              bars.rect(x0 + 0.5, y1, barWidth, y0 - y1)
            }

            return { stroke: bars, fill: bars }
          },
        },
      ],
      hooks: {
        draw: [
          (u) => {
            // Draw zero line if needed
            if (
              showZeroLine &&
              histogramData.min < 0 &&
              histogramData.max > 0
            ) {
              const ctx = u.ctx
              const x = u.valToPos(0, "x", true)
              const y0 = u.valToPos(0, "y", true)
              const y1 = u.valToPos(yMax, "y", true)

              ctx.save()
              ctx.strokeStyle = darkMode ? "rgba(255, 255, 255, 0.35)" : "rgba(128, 128, 128, 0.5)"
              ctx.lineWidth = 1
              ctx.setLineDash([5, 5])
              ctx.beginPath()
              ctx.moveTo(x, y0)
              ctx.lineTo(x, y1)
              ctx.stroke()
              ctx.restore()
            }
          },
        ],
        setCursor: [
          (u) => {
            if (!tooltipRef.current) return
            const { left, idx } = u.cursor

            if (
              idx === null ||
              idx === undefined ||
              left === undefined ||
              left < 0
            ) {
              tooltipRef.current.style.display = "none"
              return
            }

            const binCenter = histogramData.binCenters[idx]
            const count = histogramData.counts[idx]
            const binMin = histogramData.binEdges[idx]
            const binMax = histogramData.binEdges[idx + 1]

            if (binCenter === undefined || count === undefined) {
              tooltipRef.current.style.display = "none"
              return
            }

            const percentage = (
              (count / histogramData.totalCount) *
              100
            ).toFixed(1)

            tooltipRef.current.innerHTML = `
              <div class="font-medium mb-1">Range: ${formatValue(
                binMin,
              )} - ${formatValue(binMax)}</div>
              <div class="text-muted-foreground">Count: ${count.toLocaleString()} (${percentage}%)</div>
            `
            tooltipRef.current.style.display = "block"

            // Detect fixed positioning offset (backdrop-filter ancestors shift the coordinate system)
            tooltipRef.current.style.left = "0px"
            tooltipRef.current.style.top = "0px"
            const fixedOrigin = tooltipRef.current.getBoundingClientRect()

            const containerRect = containerRef.current!.getBoundingClientRect()

            // Horizontal: prefer right of cursor, flip left if overflowing
            let tooltipX = containerRect.left + left + 15
            if (tooltipX + fixedOrigin.width + 20 > window.innerWidth) {
              // Keep right-side alignment unchanged; when flipped left, compensate by plot offset.
              tooltipX =
                containerRect.left + left - fixedOrigin.width - 15 + u.bbox.left
            }
            tooltipX = Math.max(4, tooltipX)

            // Vertical: prefer overlapping top of chart, flip to bottom if overflowing viewport
            let tooltipY = containerRect.top - fixedOrigin.height + 20
            if (tooltipY < 4) {
              tooltipY = containerRect.top + 8
            }

            tooltipRef.current.style.left = `${tooltipX - fixedOrigin.left}px`
            tooltipRef.current.style.top = `${tooltipY - fixedOrigin.top}px`
          },
        ],
      },
    }

    // Destroy existing chart
    if (chartRef.current) {
      chartRef.current.destroy()
    }

    // Create new chart
    const chart = new uPlot(opts, uplotData, container)
    chartRef.current = chart

    // Handle resize
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
  }, [histogramData, hasData, showZeroLine, formatValue, darkMode])

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) {
      tooltipRef.current.style.display = "none"
    }
  }, [])

  const showLoadingOpacity = isFetching && (!isRefetching || isPlaceholderData)

  return (
    <div
      ref={visibilityRef}
      className={cn(
        "group/card rounded-lg border border-border p-3 transition-opacity",
        "bg-background",
        showLoadingOpacity && "opacity-50",
      )}
    >
      <div className="flex items-center justify-between mb-2">
        {headerPrefix}
        <div className="flex-1 min-w-0">
          <h4
            className="text-xs font-medium leading-snug line-clamp-2 break-words"
            title={label}
          >
            {label}
          </h4>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onRemove}
            className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-muted/50 opacity-0 group-hover/card:opacity-100 transition-opacity"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
      {hasData ? (
        <div
          className="h-40 relative bg-background rounded"
          ref={containerRef}
          onMouseLeave={handleMouseLeave}
        >
          {/* Custom tooltip */}
          <div
            ref={tooltipRef}
            className="fixed z-[9999] max-w-[360px] bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl border border-border pointer-events-none"
            style={{ display: "none" }}
          />
          {isRefetching && !isPlaceholderData && (
            <Loader2 className="absolute bottom-0.5 left-0.5 h-3 w-3 animate-spin text-muted-foreground" />
          )}
        </div>
      ) : (
        <div className="h-40 flex items-center justify-center text-muted-foreground text-xs rounded">
          {isFetching
            ? "Loading..."
            : step === null || step === undefined
              ? "Select a step to view histogram"
              : "No data"}
        </div>
      )}
    </div>
  )
}

// Distribution Over Time Chart Component (Heatmap-style visualization)
interface DistributionOverTimeChartProps {
  runPath: string
  metricType: string
  label: string
  showZeroLine?: boolean
  unit?: string
  isTokenMetric?: boolean
  currentStep?: number | null
  shouldPoll: boolean
  onRemove: () => void
  selectedStepAtom?: AtomWithState<number | null>
  maxSelectableStep?: number | null
  scrollRoot?: Element | null
  headerPrefix?: React.ReactNode
}

// Helper: aggregate distribution data to ~100 time bins
const TARGET_COMPACT_BINS = 100

function aggregateDistributionData(
  steps: number[],
  counts: number[][],
  binEdges: number[],
): { steps: number[]; counts: number[][] } {
  const numSteps = steps.length
  if (numSteps <= TARGET_COMPACT_BINS) {
    return { steps, counts }
  }

  const stepsPerBin = Math.ceil(numSteps / TARGET_COMPACT_BINS)
  const newSteps: number[] = []
  const newCounts: number[][] = []
  const numValueBins = binEdges.length - 1

  for (let i = 0; i < numSteps; i += stepsPerBin) {
    const endIdx = Math.min(i + stepsPerBin, numSteps)
    // Use the middle step of the group as the representative
    const midIdx = Math.floor((i + endIdx - 1) / 2)
    newSteps.push(steps[midIdx])

    // Aggregate counts across all steps in this group
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

function DistributionOverTimeChart({
  runPath,
  metricType,
  label,
  showZeroLine,
  isTokenMetric,
  currentStep,
  shouldPoll,
  onRemove,
  selectedStepAtom = rolloutsSelectedStepAtom,
  maxSelectableStep,
  scrollRoot = null,
  headerPrefix,
}: DistributionOverTimeChartProps) {
  const darkMode = useAtomValue(darkModeAtom)
  const { isFullscreen, toggleFullscreen, fullscreenPortal } = useChartFullscreen()
  const visibilityRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const hoverLineRef = useRef<HTMLDivElement>(null)
  const setSelectedStep = useSetAtom(selectedStepAtom)
  const [resizeTick, setResizeTick] = useState(0)

  const isOnScreen = useOnScreen(visibilityRef, {
    root: scrollRoot,
    threshold: 0,
  })
  const isVisible = isOnScreen || isFullscreen

  // Fetch distribution over time data (only when visible)
  const { data, isFetching, isRefetching, isPlaceholderData } =
    useStepDistributionOverTime(
      runPath,
      metricType,
      isVisible && !!runPath && !!metricType,
      shouldPoll,
    )

  const hasData = data && data.steps.length > 0 && data.bin_edges.length > 0

  const isCompact = hasData && data.steps.length > TARGET_COMPACT_BINS

  // Compute aggregated data when in compact mode
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
    const height = container.clientHeight || 160
    const tickLabelColor = darkMode ? "rgba(255, 255, 255, 0.65)" : "rgba(100, 100, 100, 0.9)"
    const gridColor = darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(128, 128, 128, 0.15)"
    const axisFont = "10px system-ui, sans-serif"

    // Set canvas size
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    ctx.scale(dpr, dpr)
    ctx.font = axisFont

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    const padding = getDistributionPadding()
    const chartWidth = width - padding.left - padding.right
    const chartHeight = height - padding.top - padding.bottom

    // Use displayData (possibly aggregated) for rendering
    const steps = displayData.steps
    const binEdges = data.bin_edges
    const counts = displayData.counts
    const numSteps = steps.length
    const numBins = binEdges.length - 1

    if (numSteps === 0 || numBins === 0) return

    // Calculate max count for color scaling
    let maxCount = 0
    for (const stepCounts of counts) {
      for (const count of stepCounts) {
        if (count > maxCount) maxCount = count
      }
    }

    const xLabelCount = Math.min(5, numSteps)
    const yLabelCount = 5

    // Calculate scales
    const bandWidth = chartWidth / Math.max(numSteps, 1)
    const xScale = (stepIdx: number) =>
      padding.left + (stepIdx + 0.5) * bandWidth
    const yScale = (value: number) => {
      const range = (data.global_max ?? 1) - (data.global_min ?? 0)
      const normalized = (value - (data.global_min ?? 0)) / (range || 1)
      return padding.top + chartHeight - normalized * chartHeight
    }

    // Color function - blue gradient based on count
    const getColor = (count: number) => {
      if (count === 0) return "transparent"
      const intensity = Math.sqrt(count / maxCount) // Square root for better visibility
      const alpha = 0.15 + intensity * 0.75
      return `rgba(59, 130, 246, ${alpha})`
    }

    // Draw grid lines (match uPlot style)
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

    // Draw bars for each step
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
      ctx.strokeStyle = darkMode ? "rgba(255, 255, 255, 0.35)" : "rgba(128, 128, 128, 0.5)"
      ctx.lineWidth = 1
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(padding.left, zeroY)
      ctx.lineTo(width - padding.right, zeroY)
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Draw current step indicator - find closest step in displayData
    if (currentStep !== null && currentStep !== undefined) {
      // Find the closest step index in displayData
      let closestIdx = -1
      let minDiff = Infinity
      for (let i = 0; i < steps.length; i++) {
        const diff = Math.abs(steps[i] - currentStep)
        if (diff < minDiff) {
          minDiff = diff
          closestIdx = i
        }
      }
      if (closestIdx >= 0) {
        const x = xScale(closestIdx)
        ctx.strokeStyle = darkMode ? "rgba(255, 255, 255, 0.2)" : "rgba(128, 128, 128, 0.25)"
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(x, padding.top)
        ctx.lineTo(x, height - padding.bottom)
        ctx.stroke()
      }
    }

    // Draw axes
    ctx.strokeStyle = tickLabelColor
    ctx.lineWidth = 1

    // X axis
    ctx.beginPath()
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(width - padding.right, height - padding.bottom)
    ctx.stroke()

    // Y axis
    ctx.beginPath()
    ctx.moveTo(padding.left, padding.top)
    ctx.lineTo(padding.left, height - padding.bottom)
    ctx.stroke()

    // X axis labels (steps)
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

    // Y axis labels (values)
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
    currentStep,
    formatYAxisLabel,
    getDistributionPadding,
    darkMode,
    resizeTick,
    isFullscreen,
  ])

  // Handle resize — trigger full canvas redraw
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    const resizeObserver = new ResizeObserver(() => {
      setResizeTick((t) => t + 1)
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
    }
  }, [])

  // Handle click to select step
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!data || !displayData || !canvasRef.current || !containerRef.current)
        return

      const rect = canvasRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const width = containerRef.current.clientWidth
      const padding = getDistributionPadding()
      const chartWidth = width - padding.left - padding.right
      if (chartWidth <= 0) return

      // Calculate which step was clicked (using displayData steps)
      const relativeX = (x - padding.left) / chartWidth
      const stepIdx = Math.round(relativeX * (displayData.steps.length - 1))

      if (stepIdx >= 0 && stepIdx < displayData.steps.length) {
        // Use the step from displayData (which could be aggregated)
        let step = displayData.steps[stepIdx]
        // Clamp to max selectable step if the clicked step exceeds the run's max
        if (maxSelectableStep != null && step > maxSelectableStep) {
          step = maxSelectableStep
        }
        setSelectedStep(step)
      }
    },
    [
      data,
      displayData,
      getDistributionPadding,
      setSelectedStep,
      maxSelectableStep,
    ],
  )

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
      const height = containerRef.current.clientHeight || 160
      const padding = getDistributionPadding()
      const chartWidth = width - padding.left - padding.right
      const chartHeight = height - padding.top - padding.bottom
      if (chartWidth <= 0 || chartHeight <= 0) {
        if (hoverLineRef.current) hoverLineRef.current.style.display = "none"
        return
      }

      // Check if within chart area
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

      // Calculate step index (using displayData)
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

      // Calculate which bin the mouse is in based on Y position
      const relativeY = 1 - (y - padding.top) / chartHeight
      const numBins = data.bin_edges.length - 1
      const binIdx = Math.min(
        numBins - 1,
        Math.max(0, Math.floor(relativeY * numBins)),
      )

      // Get the bin range and count (from displayData)
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

      // Show aggregated info if in compact mode
      const isAggregated = isCompact && data.steps.length > TARGET_COMPACT_BINS
      const aggregationNote = isAggregated ? " (aggregated)" : ""

      tooltipRef.current.innerHTML = `
        <div class="font-medium mb-1">Step ${step.toLocaleString()}${aggregationNote}</div>
        <div class="text-muted-foreground">Range: ${formatValue(
          binMin,
        )} to ${formatValue(binMax)}</div>
        <div class="text-muted-foreground">Count: ${count.toLocaleString()} (${percentage}%)</div>
      `
      tooltipRef.current.style.display = "block"

      // Detect fixed positioning offset (backdrop-filter ancestors shift the coordinate system)
      tooltipRef.current.style.left = "0px"
      tooltipRef.current.style.top = "0px"
      const fixedOrigin = tooltipRef.current.getBoundingClientRect()

      const cursorViewportX = e.clientX

      // Horizontal: anchor to cursor; flip left only if overflowing.
      let tooltipX = cursorViewportX
      if (tooltipX + fixedOrigin.width + 20 > window.innerWidth) {
        tooltipX = cursorViewportX - fixedOrigin.width
      }
      const maxX = Math.max(4, window.innerWidth - fixedOrigin.width - 4)
      tooltipX = Math.min(Math.max(4, tooltipX), maxX)

      const containerRect = containerRef.current!.getBoundingClientRect()

      // Vertical: prefer overlapping top of chart, flip to bottom if overflowing viewport
      let tooltipY = containerRect.top - fixedOrigin.height + 20
      if (tooltipY < 4) {
        tooltipY = containerRect.top + 8
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

  const showLoadingOpacity = isFetching && (!isRefetching || isPlaceholderData)

  return fullscreenPortal(
    <div
      ref={visibilityRef}
      className={cn(
        "group/card bg-background",
        isFullscreen
          ? "fixed inset-0 left-56 z-50 p-6 flex flex-col"
          : "rounded-lg border border-border p-3 transition-opacity",
        !isFullscreen && showLoadingOpacity && "opacity-50",
      )}
    >
      <div className="flex items-center justify-between mb-2 shrink-0">
        {!isFullscreen && headerPrefix}
        <div className="flex-1 min-w-0">
          <h4
            className={cn("font-medium leading-snug line-clamp-2 break-words", isFullscreen ? "text-sm" : "text-xs")}
            title={label}
          >
            {label}
          </h4>
        </div>
        <div className="flex items-center gap-2">
          <FullscreenButton isFullscreen={isFullscreen} onClick={toggleFullscreen} />
          {!isFullscreen && (
            <button
              onClick={onRemove}
              className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-muted/50 opacity-0 group-hover/card:opacity-100 transition-opacity"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
      {hasData ? (
        <div className={cn("relative bg-background rounded", isFullscreen ? "flex-1 min-h-0" : "h-40")} ref={containerRef}>
          <canvas
            className="block w-full h-full max-w-full cursor-pointer"
            ref={canvasRef}
            onClick={handleClick}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />
          <div
            ref={hoverLineRef}
            className="absolute z-[2] border-l border-dashed border-border pointer-events-none"
            style={{ display: "none" }}
          />
          {/* Custom tooltip */}
          <div
            ref={tooltipRef}
            className="fixed z-[9999] max-w-[360px] bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl border border-border pointer-events-none"
            style={{ display: "none" }}
          />
          {isRefetching && !isPlaceholderData && (
            <Loader2 className="absolute bottom-0.5 left-0.5 h-3 w-3 animate-spin text-muted-foreground" />
          )}
        </div>
      ) : (
        <div className={cn("flex items-center justify-center text-muted-foreground text-xs rounded", isFullscreen ? "flex-1 min-h-0" : "h-40")}>
          {isFetching ? "Loading..." : "No data"}
        </div>
      )}
    </div>
  )
}
