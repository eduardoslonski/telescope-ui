import { atom } from "jotai"
import { atomWithStorage } from "jotai/utils"
import type {
  RolloutsDisplaySample,
  RolloutMetric,
  GoldenAnswer,
  CustomPlotItem,
} from "@/lib/types"

// Whether the server has a W&B API key configured (boolean flag only, not the secret).
// Regular atom (not localStorage) — set optimistically on config submit, server response is source of truth.
export const wandbApiKeyAtom = atom<boolean>(false)
export const wandbConfigDialogOpenAtom = atom<boolean>(false)

// Shared dialog state so the empty-state component can open the Known Projects dialog
export const knownProjectsDialogOpenAtom = atom<boolean>(false)

// Currently selected run path (the "active" run for detail views)
export const selectedRunPathAtom = atomWithStorage<string | null>(
  "selected-run-path",
  null
)

// Visible runs for plots (checked runs that will show in charts)
export const visibleRunsAtom = atomWithStorage<string[]>(
  "visible-runs",
  []
)

// Connection/tracking state
export const isTrackingAtom = atom<boolean>(false)
export const isSyncingAtom = atom<boolean>(false)

function createPerRunAtom<T>(defaultValue: T) {
  const byRunAtom = atom<Record<string, T>>({})
  return atom(
    (get) => {
      const runPath = get(selectedRunPathAtom)
      if (!runPath) return defaultValue
      const byRun = get(byRunAtom)
      return Object.prototype.hasOwnProperty.call(byRun, runPath)
        ? byRun[runPath]
        : defaultValue
    },
    (get, set, update: T | ((prev: T) => T)) => {
      const runPath = get(selectedRunPathAtom)
      if (!runPath) return
      const byRun = get(byRunAtom)
      const previous = Object.prototype.hasOwnProperty.call(byRun, runPath)
        ? byRun[runPath]
        : defaultValue
      const next =
        typeof update === "function"
          ? (update as (prev: T) => T)(previous)
          : update
      set(byRunAtom, { ...byRun, [runPath]: next })
    }
  )
}

// ============================================================================
// Page-specific UI state (persists across navigation)
// ============================================================================

// Overview Page
export const overviewShowCodeViewAtom = atom<boolean>(false)
export const overviewShowEmaAtom = atomWithStorage<boolean>("overview-show-ema", false)
export const overviewEmaSpanAtom = atomWithStorage<number>("overview-ema-span", 10)
export const overviewShowAllRunsAtom = atomWithStorage<boolean>("overview-show-all-runs", false)
export const DEFAULT_OVERVIEW_PLOTS: CustomPlotItem[] = [
  { id: "default-reward", metricKey: "reward_sum_mean", label: "Reward", plotType: "step_metric" },
  { id: "default-timing", metricKey: "timing_step_total", label: "Time per Step", plotType: "step_metric" },
  { id: "default-grad-norm", metricKey: "grad_norm", label: "Grad Norm", plotType: "step_metric" },
]
export const overviewPlotsAtom = atomWithStorage<CustomPlotItem[]>(
  "overview-plots",
  DEFAULT_OVERVIEW_PLOTS
)

// Metrics Page
export const metricsShowEmaAtom = atom<boolean>(false)
export const metricsEmaSpanAtom = atom<number>(10)
export type MetricsXAxisMode = "step" | "time"
export const metricsXAxisModeAtom = atom<MetricsXAxisMode>("step")
export const metricsMaxLimitEnabledAtom = atom<boolean>(false)
export const metricsMaxStepAtom = atom<number>(100)
export const metricsMaxTimeAtom = atom<number>(3600) // in seconds
export const metricsScrollTopAtom = atom<number>(0)
export const metricsViewModeAtom = atom<"all" | "custom">("all")
export interface MetricsChartFilterState {
  ignoreOutliers: boolean
  ignoreFirstStep: boolean
  minY: number | null
  maxY: number | null
}
export const metricsChartFiltersAtom = atom<
  Record<string, MetricsChartFilterState>
>({})

// Timeline Page
export const timelinePageAtom = createPerRunAtom<number>(0)
export const timelineIntervalAtom = atom<number>(30)

// Rollouts Page
export const rolloutsSelectedStepAtom = createPerRunAtom<number | null>(null)

// Rollouts Discarded Page
export const rolloutsDiscardedSelectedStepAtom =
  createPerRunAtom<number | null>(null)


// Combined Timeline Chart UI state
export const inferenceServerPageAtom = atom<number>(0)
export const trainerGpuPageAtom = atom<number>(0)
export const inferenceHighlightDiscardedAtom = atom<boolean>(true)
export const inferenceShowWeightUpdateAtom = atom<boolean>(true)
export const inferenceShowComputeRewardAtom = atom<boolean>(false)
export const inferenceLaneHeightAtom = atom<number>(14)
export const inferenceLanePageAtom = atom<number>(0)
export const inferenceMaxLanesAtom = atomWithStorage<number>("inference-max-lanes", 128)

// Trainer section selected event for drill-down
export interface SelectedTrainerEvent {
  eventType: string
  rank: number
  step: number
}
export const selectedTrainerEventAtom = createPerRunAtom<SelectedTrainerEvent | null>(null)

// Inference section selected request for highlighting related requests
export interface SelectedInferenceRequest {
  sampleId: number
  groupId: number
  isEval?: boolean
}
export const selectedInferenceRequestAtom = createPerRunAtom<SelectedInferenceRequest | null>(null)

// Rollouts Page - Selected metrics to display
export const rolloutsSelectedMetricsAtom = atomWithStorage<string[]>(
  "rollouts-selected-metrics",
  []
)

// Rollouts Discarded Page - Selected metrics to display
export const rolloutsDiscardedSelectedMetricsAtom = atomWithStorage<string[]>(
  "rollouts-discarded-selected-metrics",
  []
)

// Rollouts Page - Metrics sidebar open state
export const rolloutsMetricsSidebarOpenAtom = atomWithStorage<boolean>(
  "rollouts-metrics-sidebar-open",
  true
)

// Rollouts Discarded Page - Metrics sidebar open state
export const rolloutsDiscardedMetricsSidebarOpenAtom =
  atomWithStorage<boolean>(
    "rollouts-discarded-metrics-sidebar-open",
    true
  )

// Rollouts Page - Sample picker sidebar open state
export const rolloutsSamplePickerSidebarOpenAtom = atomWithStorage<boolean>(
  "rollouts-sample-picker-sidebar-open",
  true
)

// Rollouts Discarded Page - Sample picker sidebar open state
export const rolloutsDiscardedSamplePickerSidebarOpenAtom =
  atomWithStorage<boolean>(
    "rollouts-discarded-sample-picker-sidebar-open",
    true
  )

// Rollouts Page - Selected group in sample picker (null = show all groups)
export const rolloutsSelectedGroupIdAtom =
  createPerRunAtom<number | null>(null)

// Rollouts Discarded Page - Selected group in sample picker (null = show all groups)
export const rolloutsDiscardedSelectedGroupIdAtom =
  createPerRunAtom<number | null>(null)

// Rollouts Page - Selected sample index (null = show all samples in group/all groups)
export const rolloutsSelectedSampleIdxAtom =
  createPerRunAtom<number | null>(null)

// Rollouts Discarded Page - Selected sample index (null = show all samples in group/all groups)
export const rolloutsDiscardedSelectedSampleIdxAtom =
  createPerRunAtom<number | null>(null)

export interface RolloutsLastValidSampleState {
  sample: RolloutsDisplaySample
  rolloutMetrics: RolloutMetric[]
  goldenAnswers: GoldenAnswer[]
}

export const rolloutsLastValidSampleAtom =
  createPerRunAtom<RolloutsLastValidSampleState | null>(null)

export const rolloutsDiscardedLastValidSampleAtom =
  createPerRunAtom<RolloutsLastValidSampleState | null>(null)

// Rollouts Page - Sample picker view mode (groups or samples)
export const rolloutsSamplePickerViewModeAtom = atom<"groups" | "samples">("groups")

// Rollouts - Sample picker display metrics (shared across rollouts and discarded)
export const rolloutsSamplePickerDisplayMetricsAtom = atomWithStorage<string[]>(
  "rollouts-sample-picker-display-metrics",
  ["gen_length_avg", "reward"]
)

// Rollouts - Sample picker tooltips enabled
export const rolloutsSamplePickerTooltipsAtom = atomWithStorage<boolean>(
  "rollouts-sample-picker-tooltips",
  true
)

// Rollouts - Sample view selected metrics filter (empty = show all)
export const rolloutsSampleViewMetricsAtom = atomWithStorage<string[]>(
  "rollouts-sample-view-metrics",
  []
)

// Rollouts Discarded Page - Sample picker view mode (groups or samples)
export const rolloutsDiscardedSamplePickerViewModeAtom = atom<
  "groups" | "samples"
>("groups")

// Evals Page
export const evalsSelectedStepAtom = createPerRunAtom<number | null>(null)
export const evalsSelectedEvalNameAtom = createPerRunAtom<string | null>(null)
export const evalsSelectedGroupIdAtom = createPerRunAtom<number | null>(null)
export const evalsSelectedSampleIdxAtom = createPerRunAtom<number | null>(null)
export const evalsLastValidSampleAtom =
  createPerRunAtom<RolloutsLastValidSampleState | null>(null)
export const evalsSamplePickerSidebarOpenAtom = atomWithStorage<boolean>(
  "evals-sample-picker-sidebar-open",
  true
)
export const evalsSamplePickerViewModeAtom = atom<"groups" | "samples">("groups")
export const evalsMetricsSidebarOpenAtom = atomWithStorage<boolean>(
  "evals-metrics-sidebar-open",
  true
)
export const evalsSelectedMetricsAtom = atomWithStorage<string[]>(
  "evals-selected-metrics",
  []
)

// Infra Page
export type InfraViewTab = "metrics" | "topology" | "model"
export const infraViewTabAtom = atom<InfraViewTab>("metrics")
export const infraPageAtom = createPerRunAtom<number>(0)
export const infraIntervalAtom = atom<number>(60)
export const infraLiveAtom = atom<boolean>(true)
export const infraAggregateEnabledAtom = atom<boolean>(false)
export const infraAggregateWindowAtom = atom<number>(1) // seconds
export const infraScrollTopAtom = createPerRunAtom<number>(0)
export type InfraRoleMode = "combined" | "separated"
export type InfraNodeMode = "combined" | "separated"
export const infraRoleModeAtom = createPerRunAtom<InfraRoleMode>("separated")
export const infraSystemMetricsOpenAtom = createPerRunAtom<boolean>(true)
export const infraCpuMetricsOpenAtom = createPerRunAtom<boolean>(true)
export const infraTrainerSectionOpenAtom = createPerRunAtom<boolean>(true)
export const infraInferenceSectionOpenAtom = createPerRunAtom<boolean>(true)
export const infraTrainerNodeModeAtom = createPerRunAtom<InfraNodeMode>("combined")
export const infraInferenceNodeModeAtom = createPerRunAtom<InfraNodeMode>("combined")
export const infraVllmMetricsOpenAtom = createPerRunAtom<boolean>(true)

// Hovered run in sidebar (for highlighting in charts)
export const hoveredRunIdAtom = atom<string | null>(null)

// Rollouts - Render options (markdown, latex, code) for sample visualizer
export type RolloutsRenderOption = "markdown" | "latex" | "code"
export const rolloutsRenderOptionsAtom = atomWithStorage<RolloutsRenderOption[]>(
  "rollouts-render-options",
  ["markdown", "latex", "code"]
)
export const rolloutsFormatThinkAtom = atomWithStorage<boolean>(
  "rollouts-format-think",
  true
)

// Synced cursor for CTRL+hover across all metrics charts
// Stores the current x-value (step or time) when CTRL is held
export interface SyncedCursorState {
  xValue: number
  sourceChartId: string // To prevent the source chart from re-rendering
}
export const syncedCursorAtom = atom<SyncedCursorState | null>(null)
