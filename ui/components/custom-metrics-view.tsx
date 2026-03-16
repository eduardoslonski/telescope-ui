
import { useState, useCallback, useMemo, useRef, useEffect } from "react"
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
  useSortable,
  verticalListSortingStrategy,
  rectSortingStrategy,
} from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"
import { useQueryClient } from "@tanstack/react-query"
import { Check, ChevronDown, Plus, X, GripVertical, Pencil, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { API_BASE } from "@/lib/constants"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
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
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useCustomMetricsLayout, useCustomMetricsTemplate, useStepTimes } from "@/hooks/use-run-data"
import { MetricChart, EvalMetricChart, DistributionOverTimeChart, InferencePerformanceChartCard } from "@/components/step-metrics-charts"
import type {
  CustomMetricsLayout,
  CustomSection,
  CustomGroup,
  CustomPlotItem,
} from "@/lib/types"

// ============================================================================
// Constants
// ============================================================================

const STAT_SUFFIXES = ["mean", "std", "min", "max"] as const
const SUFFIX_LABELS: Record<string, string> = {
  mean: "Mean",
  std: "Std",
  min: "Min",
  max: "Max",
}

function generateId(): string {
  return Math.random().toString(36).slice(2, 10)
}

export function formatMetricLabel(metricName: string): string {
  return metricName
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ")
}

// ============================================================================
// Plot catalog
// ============================================================================

export interface PlotCatalogItem {
  section: string
  group?: string
  metricKey: string
  label: string
  plotType: "step_metric" | "eval_metric" | "distribution_over_time" | "histogram" | "inference_performance"
  evalName?: string
  distMetricType?: string
  histogramMetricType?: string
  inferenceMetricType?: string
  simple?: boolean
}

export function buildPlotCatalog(
  customMetricSections: Record<string, Record<string, string[]>>,
  availableRewardNames: string[],
  evalsList: Array<{
    eval_name: string
    available_rollout_metric_names: string[]
  }>,
): PlotCatalogItem[] {
  const catalog: PlotCatalogItem[] = []

  for (const [sectionName, groups] of Object.entries(customMetricSections)) {
    const sectionLabel = formatMetricLabel(sectionName)
    for (const [groupName, metricNames] of Object.entries(groups)) {
      const groupLabel = groupName ? formatMetricLabel(groupName) : undefined
      for (const metricName of metricNames) {
        catalog.push({
          section: sectionLabel,
          group: groupLabel,
          metricKey: metricName,
          label: formatMetricLabel(metricName),
          plotType: "step_metric",
          simple: true,
        })
      }
    }
  }

  for (const suffix of STAT_SUFFIXES) {
    catalog.push({
      section: "Reward",
      metricKey: `reward_sum_${suffix}`,
      label: `Reward ${SUFFIX_LABELS[suffix]}`,
      plotType: "step_metric",
    })
  }
  catalog.push({
    section: "Reward",
    metricKey: "reward_gini_mean",
    label: "Reward Sparsity (Gini)",
    plotType: "step_metric",
    simple: true,
  })
  catalog.push({
    section: "Reward",
    metricKey: "reward_sum_distribution_over_time",
    label: "Reward Dist. Over Time",
    plotType: "distribution_over_time",
    distMetricType: "reward_sum",
  })

  for (const rewardName of availableRewardNames) {
    const rewardLabel = formatMetricLabel(rewardName)
    for (const suffix of STAT_SUFFIXES) {
      catalog.push({
        section: "Samples Metrics",
        group: rewardLabel,
        metricKey: `reward_${rewardName}_${suffix}`,
        label: `${rewardLabel} ${SUFFIX_LABELS[suffix]}`,
        plotType: "step_metric",
      })
    }
    catalog.push({
      section: "Samples Metrics",
      group: rewardLabel,
      metricKey: `reward_${rewardName}_gini_mean`,
      label: `${rewardLabel} Sparsity (Gini)`,
      plotType: "step_metric",
      simple: true,
    })
    catalog.push({
      section: "Samples Metrics",
      group: rewardLabel,
      metricKey: `reward_${rewardName}_distribution_over_time`,
      label: `${rewardLabel} Dist. Over Time`,
      plotType: "distribution_over_time",
      distMetricType: `reward_${rewardName}`,
    })
  }

  for (const suffix of STAT_SUFFIXES) {
    catalog.push({
      section: "Advantage",
      metricKey: `advantage_${suffix}`,
      label: `Advantage ${SUFFIX_LABELS[suffix]}`,
      plotType: "step_metric",
    })
  }

  catalog.push({
    section: "Advantage",
    metricKey: "advantage_distribution_over_time",
    label: "Advantage Dist. Over Time",
    plotType: "distribution_over_time",
    distMetricType: "advantage",
  })

  for (const evalEntry of evalsList) {
    const evalName = evalEntry.eval_name
    for (const metricName of evalEntry.available_rollout_metric_names) {
      const metricLabel = formatMetricLabel(metricName)
      for (const suffix of STAT_SUFFIXES) {
        catalog.push({
          section: "Evals",
          group: `${evalName} / ${metricLabel}`,
          metricKey: `reward_${metricName}_${suffix}`,
          label: `${evalName} ${metricLabel} ${SUFFIX_LABELS[suffix]}`,
          plotType: "eval_metric",
          evalName,
        })
      }
    }
    for (const tokenPrefix of [
      "length_completion",
      "length_prompt",
      "length_sum",
    ]) {
      const tokenLabel =
        tokenPrefix === "length_completion"
          ? "Tokens Completion"
          : tokenPrefix === "length_prompt"
            ? "Tokens Prompt"
            : "Tokens Total"
      for (const suffix of STAT_SUFFIXES) {
        catalog.push({
          section: "Evals",
          group: `${evalName} / ${tokenLabel}`,
          metricKey: `${tokenPrefix}_${suffix}`,
          label: `${evalName} ${tokenLabel} ${SUFFIX_LABELS[suffix]}`,
          plotType: "eval_metric",
          evalName,
        })
      }
    }
  }

  for (const tokenPrefix of [
    "length_prompt",
    "length_completion",
    "length_sum",
  ]) {
    const tokenLabel =
      tokenPrefix === "length_prompt"
        ? "Tokens Prompt"
        : tokenPrefix === "length_completion"
          ? "Tokens Completion"
          : "Tokens Total"
    for (const suffix of STAT_SUFFIXES) {
      catalog.push({
        section: "Rollouts",
        group: tokenLabel,
        metricKey: `${tokenPrefix}_${suffix}`,
        label: `${tokenLabel} ${SUFFIX_LABELS[suffix]}`,
        plotType: "step_metric",
      })
    }
    catalog.push({
      section: "Rollouts",
      group: tokenLabel,
      metricKey: `${tokenPrefix}_distribution_over_time`,
      label: `${tokenLabel} Dist. Over Time`,
      plotType: "distribution_over_time",
      distMetricType: tokenPrefix,
    })
  }
  for (const m of [
    { key: "stop_reason_length_pct", label: "% Stop Reason = Length" },
    { key: "group_length_gini_mean", label: "Group Completion Length Gini" },
    {
      key: "group_length_max_median_ratio_mean",
      label: "Group Completion Length Max/Median",
    },
  ]) {
    catalog.push({
      section: "Rollouts",
      group: "General",
      metricKey: m.key,
      label: m.label,
      plotType: "step_metric",
      simple: true,
    })
  }

  for (const m of [
    { key: "discarded_count", label: "Discarded Count" },
    { key: "discarded_zero_advantage_pct", label: "Zero Advantage %" },
    { key: "discarded_max_async_pct", label: "Max Async %" },
    {
      key: "discarded_stop_reason_length_pct",
      label: "% Stop Reason = Length",
    },
    {
      key: "discarded_group_length_gini_mean",
      label: "Group Completion Length Gini",
    },
    {
      key: "discarded_group_length_max_median_ratio_mean",
      label: "Group Completion Length Max/Median",
    },
    {
      key: "discarded_zero_advantage_all_zero_pct",
      label: "Zero Adv (All Reward = 0) %",
    },
    {
      key: "discarded_zero_advantage_all_positive_pct",
      label: "Zero Adv (All Reward > 0) %",
    },
    {
      key: "discarded_zero_advantage_mean_reward",
      label: "Zero Adv Mean Reward",
    },
  ]) {
    catalog.push({
      section: "Discarded Rollouts",
      group: "General",
      metricKey: m.key,
      label: m.label,
      plotType: "step_metric",
      simple: true,
    })
  }
  catalog.push({
    section: "Discarded Rollouts",
    group: "Canceled",
    metricKey: "canceled_count",
    label: "Canceled Count",
    plotType: "step_metric",
    simple: true,
  })
  for (const tokenPrefix of [
    "discarded_length_prompt",
    "discarded_length_completion",
    "discarded_length_sum",
  ]) {
    const tokenLabel = tokenPrefix.includes("prompt")
      ? "Discarded Tokens Prompt"
      : tokenPrefix.includes("completion")
        ? "Discarded Tokens Completion"
        : "Discarded Tokens Total"
    for (const suffix of STAT_SUFFIXES) {
      catalog.push({
        section: "Discarded Rollouts",
        group: tokenLabel,
        metricKey: `${tokenPrefix}_${suffix}`,
        label: `${tokenLabel} ${SUFFIX_LABELS[suffix]}`,
        plotType: "step_metric",
      })
    }
    catalog.push({
      section: "Discarded Rollouts",
      group: tokenLabel,
      metricKey: `${tokenPrefix}_distribution_over_time`,
      label: `${tokenLabel} Dist. Over Time`,
      plotType: "distribution_over_time",
      distMetricType: tokenPrefix,
    })
  }

  // Timeline Trainer
  for (const m of [
    { key: "timing_step_total", label: "Time per Step" },
    { key: "timing_step_active", label: "Time per Step Active" },
    { key: "timing_microbatch_count", label: "Microbatches per Step" },
    { key: "timing_forward_total", label: "Timing Forward" },
    { key: "timing_backward_total", label: "Timing Backward" },
    { key: "timing_loss_computation_total", label: "Timing Loss Computation" },
    { key: "timing_compute_kl_total", label: "Timing Compute KL" },
    { key: "timing_compute_entropy_total", label: "Timing Compute Entropy" },
    { key: "timing_data_to_device_total", label: "Timing Data to Device" },
    { key: "timing_prepare_tensors_total", label: "Timing Prepare Tensors" },
    { key: "timing_waiting_for_data", label: "Timing Waiting for Data" },
    {
      key: "timing_weight_sync_trainer_total",
      label: "Timing Weight Broadcast (Trainer)",
    },
    {
      key: "timing_weight_sync_inference_total",
      label: "Timing Weight Broadcast (Inference)",
    },
  ]) {
    catalog.push({
      section: "Timeline Trainer",
      group: "Full Step (Total Time)",
      metricKey: m.key,
      label: m.label,
      plotType: "step_metric",
      simple: true,
    })
  }
  for (const m of [
    {
      key: "timing_forward_microbatch_mean",
      label: "Microbatch Timing Forward",
    },
    {
      key: "timing_backward_microbatch_mean",
      label: "Microbatch Timing Backward",
    },
    {
      key: "timing_loss_computation_microbatch_mean",
      label: "Microbatch Timing Loss Computation",
    },
    {
      key: "timing_compute_kl_microbatch_mean",
      label: "Microbatch Timing Compute KL",
    },
    {
      key: "timing_compute_entropy_microbatch_mean",
      label: "Microbatch Timing Compute Entropy",
    },
    {
      key: "timing_data_to_device_microbatch_mean",
      label: "Microbatch Timing Data to Device",
    },
    {
      key: "timing_prepare_tensors_microbatch_mean",
      label: "Microbatch Timing Prepare Tensors",
    },
  ]) {
    catalog.push({
      section: "Timeline Trainer",
      group: "Microbatch (Mean Time)",
      metricKey: m.key,
      label: m.label,
      plotType: "step_metric",
      simple: true,
    })
  }

  // Timeline Inference
  for (const m of [
    { key: "timing_save_batch_total", label: "Batch Completion (Save Batch)" },
    { key: "timing_avg_inference_time", label: "Avg Generation Time" },
    { key: "timing_avg_compute_reward_time", label: "Avg Compute Reward Time" },
  ]) {
    catalog.push({
      section: "Timeline Inference",
      group: "Batch & Averages",
      metricKey: m.key,
      label: m.label,
      plotType: "step_metric",
      simple: true,
    })
  }
  for (const m of [
    { key: "timing_generation_normal_pct", label: "Generation Normal" },
    { key: "timing_generation_discarded_pct", label: "Generation Discarded" },
    { key: "timing_generation_canceled_pct", label: "Generation Canceled" },
    { key: "timing_generation_all_pct", label: "Generation All" },
    { key: "timing_compute_reward_normal_pct", label: "Compute Reward Normal" },
    { key: "timing_compute_reward_discarded_pct", label: "Compute Reward Discarded" },
    { key: "timing_compute_reward_canceled_pct", label: "Compute Reward Canceled" },
    { key: "timing_compute_reward_all_pct", label: "Compute Reward All" },
    { key: "timing_idle_pct", label: "Idle Time" },
  ]) {
    catalog.push({
      section: "Timeline Inference",
      group: "Time Breakdown (% of Step Time)",
      metricKey: m.key,
      label: m.label,
      plotType: "step_metric",
      simple: true,
    })
  }

  // Inference Performance
  for (const m of [
    { key: "inference_calls", label: "Inference Calls", inferenceMetricType: "inference_calls" },
    { key: "requests_done", label: "Requests Done", inferenceMetricType: "requests_done" },
    { key: "rollouts_group_done", label: "Rollouts Group Done", inferenceMetricType: "rollouts_group_done" },
    { key: "rollouts_group_done_kept", label: "Rollouts Group Done Kept", inferenceMetricType: "rollouts_group_done_kept" },
    { key: "rollouts_group_done_discarded", label: "Rollouts Group Done Discarded", inferenceMetricType: "rollouts_group_done_discarded" },
    { key: "rollouts_group_done_canceled", label: "Rollouts Group Done Canceled", inferenceMetricType: "rollouts_group_done_canceled" },
  ]) {
    catalog.push({
      section: "Inference Performance",
      metricKey: m.key,
      label: m.label,
      plotType: "inference_performance",
      inferenceMetricType: m.inferenceMetricType,
    })
  }

  return catalog
}

// ============================================================================
// Save layout helper
// ============================================================================

async function saveLayout(layout: CustomMetricsLayout) {
  await fetch(`${API_BASE}/custom-metrics-layout`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ layout }),
  })
}

async function saveTemplateLayout(templateId: string, layout: CustomMetricsLayout) {
  await fetch(`${API_BASE}/custom-metrics-templates/${templateId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ layout }),
  })
}

// ============================================================================
// Name dialog (for create + rename)
// ============================================================================

function NameDialog({
  open,
  onOpenChange,
  title,
  existingNames,
  initialValue = "",
  onSubmit,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  existingNames: string[]
  initialValue?: string
  onSubmit: (name: string) => void
}) {
  const [value, setValue] = useState(initialValue)
  const [error, setError] = useState("")
  const inputRef = useRef<HTMLInputElement>(null)

  const handleOpenChange = (nextOpen: boolean) => {
    if (nextOpen) {
      setValue(initialValue)
      setError("")
      setTimeout(() => inputRef.current?.focus(), 50)
    }
    onOpenChange(nextOpen)
  }

  const handleSubmit = () => {
    const trimmed = value.trim()
    if (!trimmed) {
      setError("Name cannot be empty")
      return
    }
    if (
      existingNames.some((n) => n.toLowerCase() === trimmed.toLowerCase()) &&
      trimmed.toLowerCase() !== initialValue.toLowerCase()
    ) {
      setError("Name already exists")
      return
    }
    onSubmit(trimmed)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[340px]">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <div className="space-y-2">
          <Input
            ref={inputRef}
            value={value}
            onChange={(e) => {
              setValue(e.target.value)
              setError("")
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSubmit()
            }}
            placeholder="Name..."
          />
          {error && <p className="text-xs text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <Button size="sm" onClick={handleSubmit}>
            {initialValue ? "Rename" : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ============================================================================
// Plot select popover
// ============================================================================

export function plotItemKey(item: {
  plotType: string
  evalName?: string
  metricKey: string
}) {
  return `${item.plotType}|${item.evalName || ""}|${item.metricKey}`
}

export function PlotSelectPopover({
  catalog,
  onSelect,
  onDeselect,
  existingPlots,
  children,
  allowDuplicates = false,
}: {
  catalog: PlotCatalogItem[]
  onSelect: (item: PlotCatalogItem) => void
  onDeselect?: (item: PlotCatalogItem) => void
  existingPlots?: CustomPlotItem[]
  children: React.ReactNode
  allowDuplicates?: boolean
}) {
  const existingCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const p of existingPlots ?? []) {
      const key = plotItemKey(p)
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    return counts
  }, [existingPlots])
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState("")

  const grouped = useMemo(() => {
    const lower = search.toLowerCase()
    const filtered = lower
      ? catalog.filter(
          (item) =>
            item.label.toLowerCase().includes(lower) ||
            item.metricKey.toLowerCase().includes(lower) ||
            item.section.toLowerCase().includes(lower) ||
            (item.group?.toLowerCase().includes(lower) ?? false),
        )
      : catalog

    const sections: Array<{
      section: string
      groups: Array<{ group: string; items: PlotCatalogItem[] }>
    }> = []

    for (const item of filtered) {
      let sec = sections.find((s) => s.section === item.section)
      if (!sec) {
        sec = { section: item.section, groups: [] }
        sections.push(sec)
      }
      const groupKey = item.group || ""
      let grp = sec.groups.find((g) => g.group === groupKey)
      if (!grp) {
        grp = { group: groupKey, items: [] }
        sec.groups.push(grp)
      }
      grp.items.push(item)
    }

    return sections
  }, [catalog, search])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[320px] p-0"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-2 border-b">
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search metrics..."
            className="h-7 text-xs"
          />
        </div>
        <div className="max-h-[50vh] overflow-y-auto p-1.5">
          {grouped.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-4">
              No metrics found
            </p>
          )}
          {grouped.map((sec) => (
            <div key={sec.section} className="mb-2">
              <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider px-1.5 py-1">
                {sec.section}
              </div>
              {sec.groups.map((grp) => (
                <div key={`${sec.section}-${grp.group}`}>
                  {grp.group && (
                    <div className="text-[10px] font-medium text-muted-foreground/70 px-3 py-0.5">
                      {grp.group}
                    </div>
                  )}
                  {grp.items.map((item) => {
                    const count = existingCounts.get(plotItemKey(item)) ?? 0
                    const isSelected = count > 0
                    return (
                      <button
                        key={`${item.plotType}-${item.evalName || ""}-${item.metricKey}`}
                        type="button"
                        className={cn(
                          "flex w-full items-center gap-1.5 rounded px-3 py-0.5 text-xs text-left transition-colors hover:bg-accent hover:text-accent-foreground",
                          !allowDuplicates && isSelected && "text-accent-foreground",
                        )}
                        onClick={() => {
                          if (!allowDuplicates && isSelected && onDeselect) {
                            onDeselect(item)
                          } else {
                            onSelect(item)
                          }
                          if (allowDuplicates) {
                            setOpen(false)
                            setSearch("")
                          }
                        }}
                      >
                        <span className="flex-1">{item.label}</span>
                        {allowDuplicates
                          ? count > 0 && (
                              <span className="text-[10px] text-muted-foreground tabular-nums">
                                {count}
                              </span>
                            )
                          : isSelected && (
                              <Check className="h-3 w-3 text-muted-foreground shrink-0" />
                            )}
                      </button>
                    )
                  })}
                </div>
              ))}
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  )
}

// ============================================================================
// Confirm delete dialog
// ============================================================================

function ConfirmDeleteDialog({
  open,
  onOpenChange,
  title,
  description,
  onConfirm,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description: string
  onConfirm: () => void
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={onConfirm}>
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

// ============================================================================
// Shared chart props type
// ============================================================================

export interface SharedChartProps {
  runs: Array<{
    runPath: string
    runName?: string | null
    color: string
    isSelected: boolean
  }>
  shouldPoll: boolean
  showEma: boolean
  emaSpan: number
  hoveredRunId: string | null
  xAxisMode: "step" | "time"
  scrollRoot: Element | null
  maxStep?: number
  maxTime?: number
  stepTimesByRun?: Map<string, Map<number, number>>
  firstStepTimesByRun?: Map<string, number>
  isStepTimesFetching?: boolean
  isStepTimesRefetching?: boolean
  availableSampleTags?: Record<string, string[]>
}

// ============================================================================
// Sortable plot card (renders actual chart)
// ============================================================================

export function SortablePlotCard({
  plot,
  onRemove,
  chartProps,
}: {
  plot: CustomPlotItem
  onRemove: () => void
  chartProps: SharedChartProps
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: plot.id })

  const style: React.CSSProperties = {
    transform: transform
      ? `translate3d(${Math.round(transform.x)}px, ${Math.round(transform.y)}px, 0)`
      : undefined,
    transition,
    opacity: isDragging ? 0.4 : 1,
    position: "relative" as const,
    zIndex: isDragging ? 50 : undefined,
  }

  const label = plot.label || formatMetricLabel(plot.metricKey)

  const dragHandle = (
    <button
      {...attributes}
      {...listeners}
      className="-ml-1.5 cursor-grab active:cursor-grabbing p-0.5 rounded hover:bg-muted transition-all shrink-0 opacity-0 group-hover/chart:opacity-100"
    >
      <GripVertical className="h-3 w-3 text-muted-foreground" />
    </button>
  )

  const deleteButton = (
    <button
      onClick={onRemove}
      className="-mr-1 h-5 px-1.5 rounded border border-border hover:bg-muted flex items-center gap-1 transition-all opacity-0 group-hover/chart:opacity-100"
    >
      <X className="h-3 w-3" />
    </button>
  )

  return (
    <div ref={setNodeRef} style={style}>
      {plot.plotType === "distribution_over_time" && plot.distMetricType ? (
        (() => {
          const selectedRun = chartProps.runs.find((r) => r.isSelected) ?? chartProps.runs[0]
          return selectedRun ? (
            <DistributionOverTimeChart
              runPath={selectedRun.runPath}
              metricType={plot.distMetricType!}
              label={label}
              shouldPoll={chartProps.shouldPoll}
              scrollRoot={chartProps.scrollRoot}
              headerPrefix={dragHandle}
              headerSuffix={deleteButton}
            />
          ) : null
        })()
      ) : plot.plotType === "inference_performance" && plot.inferenceMetricType ? (
        (() => {
          const selectedRun = chartProps.runs.find((r) => r.isSelected) ?? chartProps.runs[0]
          return selectedRun ? (
            <InferencePerformanceChartCard
              runPath={selectedRun.runPath}
              shouldPoll={chartProps.shouldPoll}
              scrollRoot={chartProps.scrollRoot}
              inferenceMetricType={plot.inferenceMetricType!}
              label={label}
              headerPrefix={dragHandle}
              headerSuffix={deleteButton}
            />
          ) : null
        })()
      ) : plot.plotType === "eval_metric" && plot.evalName ? (
        <EvalMetricChart
          runs={chartProps.runs}
          shouldPoll={chartProps.shouldPoll}
          evalName={plot.evalName}
          metricName={plot.metricKey}
          label={label}
          showEma={chartProps.showEma}
          emaSpan={chartProps.emaSpan}
          hoveredRunId={chartProps.hoveredRunId}
          xAxisMode={chartProps.xAxisMode}
          stepTimesByRun={chartProps.stepTimesByRun}
          firstStepTimesByRun={chartProps.firstStepTimesByRun}
          isStepTimesFetching={chartProps.isStepTimesFetching}
          isStepTimesRefetching={chartProps.isStepTimesRefetching}
          scrollRoot={chartProps.scrollRoot}
          maxStepLimit={chartProps.maxStep}
          maxTimeLimit={chartProps.maxTime}
          headerPrefix={dragHandle}
          headerSuffix={deleteButton}
          filterKey={plot.id}
        />
      ) : (
        <MetricChart
          runs={chartProps.runs}
          shouldPoll={chartProps.shouldPoll}
          metricName={plot.metricKey}
          label={label}
          showEma={chartProps.showEma}
          emaSpan={chartProps.emaSpan}
          hoveredRunId={chartProps.hoveredRunId}
          xAxisMode={chartProps.xAxisMode}
          stepTimesByRun={chartProps.stepTimesByRun}
          firstStepTimesByRun={chartProps.firstStepTimesByRun}
          isStepTimesFetching={chartProps.isStepTimesFetching}
          isStepTimesRefetching={chartProps.isStepTimesRefetching}
          scrollRoot={chartProps.scrollRoot}
          maxStepLimit={chartProps.maxStep}
          maxTimeLimit={chartProps.maxTime}
          availableSampleTags={chartProps.availableSampleTags}
          headerPrefix={dragHandle}
          headerSuffix={deleteButton}
          filterKey={plot.id}
        />
      )}
    </div>
  )
}

// ============================================================================
// Sortable group
// ============================================================================

function SortableGroup({
  group,
  catalog,
  groupNames,
  onUpdate,
  onRemove,
  chartProps,
}: {
  group: CustomGroup
  catalog: PlotCatalogItem[]
  groupNames: string[]
  onUpdate: (updated: CustomGroup) => void
  onRemove: () => void
  chartProps: SharedChartProps
}) {
  const [isOpen, setIsOpen] = useState(true)
  const [titleHovered, setTitleHovered] = useState(false)
  const [renameOpen, setRenameOpen] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: group.id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  }

  const handleAddPlot = (item: PlotCatalogItem) => {
    const newPlot: CustomPlotItem = {
      id: generateId(),
      metricKey: item.metricKey,
      label: item.label,
      plotType: item.plotType,
      evalName: item.evalName,
      distMetricType: item.distMetricType,
      inferenceMetricType: item.inferenceMetricType,
    }
    onUpdate({ ...group, plots: [...group.plots, newPlot] })
  }

  const handleRemovePlot = (plotId: string) => {
    onUpdate({ ...group, plots: group.plots.filter((p) => p.id !== plotId) })
  }

  const handleRenameSave = (newName: string) => {
    onUpdate({ ...group, name: newName })
  }

  const plotSensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  )

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    if (!over || active.id === over.id) return
    const oldIdx = group.plots.findIndex((p) => p.id === active.id)
    const newIdx = group.plots.findIndex((p) => p.id === over.id)
    if (oldIdx === -1 || newIdx === -1) return
    onUpdate({ ...group, plots: arrayMove(group.plots, oldIdx, newIdx) })
  }

  return (
    <div ref={setNodeRef} style={style}>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <div
          className="flex items-center gap-1 h-8"
          onMouseEnter={() => setTitleHovered(true)}
          onMouseLeave={() => setTitleHovered(false)}
        >
          <CollapsibleTrigger asChild>
            <div className="flex items-center gap-1.5 py-1 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors shrink-0">
              <ChevronDown
                className={cn(
                  "h-3.5 w-3.5 text-muted-foreground transition-transform shrink-0",
                  !isOpen && "-rotate-90",
                )}
              />
              <h4 className="text-sm font-medium text-muted-foreground whitespace-nowrap">
                {group.name}
              </h4>
            </div>
          </CollapsibleTrigger>
          <button
            {...attributes}
            {...listeners}
            className={cn(
              "cursor-grab active:cursor-grabbing p-0.5 rounded hover:bg-muted transition-all shrink-0",
              titleHovered ? "opacity-100" : "opacity-0 pointer-events-none",
            )}
          >
            <GripVertical className="h-3.5 w-3.5 text-muted-foreground" />
          </button>
          <div
            className={cn(
              "flex items-center gap-1 shrink-0 transition-opacity",
              titleHovered ? "opacity-100" : "opacity-0 pointer-events-none",
            )}
          >
            <PlotSelectPopover
              catalog={catalog}
              onSelect={handleAddPlot}
              existingPlots={group.plots}
              allowDuplicates
            >
              <button className="flex items-center gap-0.5 h-6 px-2 text-[10px] rounded-md border border-input hover:bg-accent transition-colors">
                <Plus className="h-3 w-3" />
                New Plot
              </button>
            </PlotSelectPopover>
            <button
              onClick={() => setRenameOpen(true)}
              className="h-6 px-1.5 rounded-md border border-input hover:bg-accent flex items-center transition-colors"
              title="Rename group"
            >
              <Pencil className="h-3 w-3 text-muted-foreground" />
            </button>
            <button
              onClick={() => setConfirmDelete(true)}
              className="h-6 px-1.5 rounded-md border border-input hover:bg-accent flex items-center transition-colors"
            >
              <X className="h-3 w-3 text-muted-foreground" />
            </button>
          </div>
        </div>
        <CollapsibleContent>
          <DndContext
            sensors={plotSensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={group.plots.map((p) => p.id)}
              strategy={rectSortingStrategy}
            >
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-3 mb-4">
                {group.plots.map((plot) => (
                  <SortablePlotCard
                    key={plot.id}
                    plot={plot}
                    onRemove={() => handleRemovePlot(plot.id)}
                    chartProps={chartProps}
                  />
                ))}
                {group.plots.length === 0 && (
                  <PlotSelectPopover
                    catalog={catalog}
                    onSelect={handleAddPlot}
                    existingPlots={group.plots}
                    allowDuplicates
                  >
                    <button className="h-32 rounded-lg border-2 border-dashed border-gray-200 hover:border-gray-300 hover:bg-muted/50 transition-colors flex items-center justify-center text-xs text-muted-foreground gap-1">
                      <Plus className="h-3.5 w-3.5" />
                      Add a plot
                    </button>
                  </PlotSelectPopover>
                )}
              </div>
            </SortableContext>
          </DndContext>
        </CollapsibleContent>
      </Collapsible>
      <NameDialog
        open={renameOpen}
        onOpenChange={setRenameOpen}
        title="Rename Group"
        existingNames={groupNames.filter((n) => n !== group.name)}
        onSubmit={handleRenameSave}
        initialValue={group.name}
      />
      <ConfirmDeleteDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title="Delete Group"
        description={`Delete "${group.name}" and all its plots?`}
        onConfirm={onRemove}
      />
    </div>
  )
}

// ============================================================================
// Sortable section
// ============================================================================

function SortableSection({
  section,
  catalog,
  allSectionNames,
  onUpdate,
  onRemove,
  chartProps,
}: {
  section: CustomSection
  catalog: PlotCatalogItem[]
  allSectionNames: string[]
  onUpdate: (updated: CustomSection) => void
  onRemove: () => void
  chartProps: SharedChartProps
}) {
  const [isOpen, setIsOpen] = useState(true)
  const [titleHovered, setTitleHovered] = useState(false)
  const [newGroupOpen, setNewGroupOpen] = useState(false)
  const [renameOpen, setRenameOpen] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: section.id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  }

  const groupNames = section.groups.map((g) => g.name)

  const handleAddGroup = (name: string) => {
    const newGroup: CustomGroup = { id: generateId(), name, plots: [] }
    onUpdate({ ...section, groups: [...section.groups, newGroup] })
  }

  const handleUpdateGroup = (groupId: string, updated: CustomGroup) => {
    onUpdate({
      ...section,
      groups: section.groups.map((g) => (g.id === groupId ? updated : g)),
    })
  }

  const handleRemoveGroup = (groupId: string) => {
    onUpdate({
      ...section,
      groups: section.groups.filter((g) => g.id !== groupId),
    })
  }

  const handleAddPlot = (item: PlotCatalogItem) => {
    const newPlot: CustomPlotItem = {
      id: generateId(),
      metricKey: item.metricKey,
      label: item.label,
      plotType: item.plotType,
      evalName: item.evalName,
      distMetricType: item.distMetricType,
      inferenceMetricType: item.inferenceMetricType,
    }
    onUpdate({ ...section, plots: [...section.plots, newPlot] })
  }

  const handleRemovePlot = (plotId: string) => {
    onUpdate({
      ...section,
      plots: section.plots.filter((p) => p.id !== plotId),
    })
  }

  const handleRenameSave = (newName: string) => {
    onUpdate({ ...section, name: newName })
  }

  const groupSensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  )

  const handleGroupDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    if (!over || active.id === over.id) return
    const oldIdx = section.groups.findIndex((g) => g.id === active.id)
    const newIdx = section.groups.findIndex((g) => g.id === over.id)
    if (oldIdx === -1 || newIdx === -1) return
    onUpdate({ ...section, groups: arrayMove(section.groups, oldIdx, newIdx) })
  }

  const plotSensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  )

  const handlePlotDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    if (!over || active.id === over.id) return
    const oldIdx = section.plots.findIndex((p) => p.id === active.id)
    const newIdx = section.plots.findIndex((p) => p.id === over.id)
    if (oldIdx === -1 || newIdx === -1) return
    onUpdate({ ...section, plots: arrayMove(section.plots, oldIdx, newIdx) })
  }

  return (
    <div ref={setNodeRef} style={style}>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <div
          className="flex items-center gap-1.5 h-9"
          onMouseEnter={() => setTitleHovered(true)}
          onMouseLeave={() => setTitleHovered(false)}
        >
          <CollapsibleTrigger asChild>
            <div className="flex items-center gap-1.5 py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform shrink-0",
                  !isOpen && "-rotate-90",
                )}
              />
              <h3 className="text-sm font-semibold">{section.name}</h3>
            </div>
          </CollapsibleTrigger>
          <button
            {...attributes}
            {...listeners}
            className={cn(
              "cursor-grab active:cursor-grabbing p-0.5 rounded hover:bg-muted transition-all shrink-0",
              titleHovered ? "opacity-100" : "opacity-0 pointer-events-none",
            )}
          >
            <GripVertical className="h-4 w-4 text-muted-foreground" />
          </button>
          <div
            className={cn(
              "flex items-center gap-1 shrink-0 transition-opacity",
              titleHovered ? "opacity-100" : "opacity-0 pointer-events-none",
            )}
          >
            <button
              onClick={() => setNewGroupOpen(true)}
              className="flex items-center gap-0.5 h-7 px-2 text-xs rounded-md border border-input hover:bg-accent transition-colors"
            >
              <Plus className="h-3 w-3" />
              New Group
            </button>
            <PlotSelectPopover
              catalog={catalog}
              onSelect={handleAddPlot}
              existingPlots={section.plots}
              allowDuplicates
            >
              <button className="flex items-center gap-0.5 h-7 px-2 text-xs rounded-md border border-input hover:bg-accent transition-colors">
                <Plus className="h-3 w-3" />
                New Plot
              </button>
            </PlotSelectPopover>
            <button
              onClick={() => setRenameOpen(true)}
              className="h-7 px-1.5 rounded-md border border-input hover:bg-accent flex items-center transition-colors"
              title="Rename section"
            >
              <Pencil className="h-3.5 w-3.5 text-muted-foreground" />
            </button>
            <button
              onClick={() => setConfirmDelete(true)}
              className="h-7 px-1.5 rounded-md border border-input hover:bg-accent flex items-center transition-colors"
            >
              <X className="h-3.5 w-3.5 text-muted-foreground" />
            </button>
          </div>
        </div>
        <CollapsibleContent>
          {/* Ungrouped plots */}
          {section.plots.length > 0 && (
            <DndContext
              sensors={plotSensors}
              collisionDetection={closestCenter}
              onDragEnd={handlePlotDragEnd}
            >
              <SortableContext
                items={section.plots.map((p) => p.id)}
                strategy={rectSortingStrategy}
              >
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-3 mb-4">
                  {section.plots.map((plot) => (
                    <SortablePlotCard
                      key={plot.id}
                      plot={plot}
                      onRemove={() => handleRemovePlot(plot.id)}
                      chartProps={chartProps}
                    />
                  ))}
                </div>
              </SortableContext>
            </DndContext>
          )}

          {/* Groups */}
          <DndContext
            sensors={groupSensors}
            collisionDetection={closestCenter}
            onDragEnd={handleGroupDragEnd}
          >
            <SortableContext
              items={section.groups.map((g) => g.id)}
              strategy={verticalListSortingStrategy}
            >
              <div className="space-y-2">
                {section.groups.map((group) => (
                  <SortableGroup
                    key={group.id}
                    group={group}
                    catalog={catalog}
                    groupNames={groupNames}
                    onUpdate={(updated) => handleUpdateGroup(group.id, updated)}
                    onRemove={() => handleRemoveGroup(group.id)}
                    chartProps={chartProps}
                  />
                ))}
              </div>
            </SortableContext>
          </DndContext>

          {section.plots.length === 0 && section.groups.length === 0 && (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <p className="text-xs text-muted-foreground mb-3">
                No plots yet. Add a group or a plot.
              </p>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setNewGroupOpen(true)}
                >
                  <Plus className="h-3.5 w-3.5 mr-1" />
                  New Group
                </Button>
                <PlotSelectPopover
                  catalog={catalog}
                  onSelect={handleAddPlot}
                  existingPlots={section.plots}
                  allowDuplicates
                >
                  <Button size="sm" variant="outline">
                    <Plus className="h-3.5 w-3.5 mr-1" />
                    New Plot
                  </Button>
                </PlotSelectPopover>
              </div>
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>

      <NameDialog
        open={newGroupOpen}
        onOpenChange={setNewGroupOpen}
        title="New Group"
        existingNames={groupNames}
        onSubmit={handleAddGroup}
      />
      <NameDialog
        open={renameOpen}
        onOpenChange={setRenameOpen}
        title="Rename Section"
        existingNames={allSectionNames.filter((n) => n !== section.name)}
        onSubmit={handleRenameSave}
        initialValue={section.name}
      />
      <ConfirmDeleteDialog
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title="Delete Section"
        description={`Delete "${section.name}" and all its groups and plots?`}
        onConfirm={onRemove}
      />
    </div>
  )
}

// ============================================================================
// CustomMetricsView — main component
// ============================================================================

export function CustomMetricsView({
  runs,
  shouldPoll,
  showEma,
  emaSpan,
  hoveredRunId,
  availableRewardNames,
  customMetricSections,
  xAxisMode,
  scrollRoot,
  maxStep,
  maxTime,
  evalsList,
  newSectionTrigger,
  availableSampleTags,
  activeTemplateId,
  layoutSnapshotTrigger,
  onLayoutSnapshot,
}: {
  runs: Array<{
    runPath: string
    runName?: string | null
    color: string
    isSelected: boolean
  }>
  shouldPoll: boolean
  showEma: boolean
  emaSpan: number
  hoveredRunId: string | null
  availableRewardNames: string[]
  customMetricSections: Record<string, Record<string, string[]>>
  xAxisMode: "step" | "time"
  scrollRoot: Element | null
  maxStep?: number
  maxTime?: number
  evalsList: Array<{
    eval_name: string
    available_rollout_metric_names: string[]
  }>
  newSectionTrigger: number
  availableSampleTags?: Record<string, string[]>
  activeTemplateId: string | null
  layoutSnapshotTrigger: number
  onLayoutSnapshot: (layout: CustomMetricsLayout) => void
}) {
  const queryClient = useQueryClient()
  const { data: layoutData, isLoading: layoutLoading } = useCustomMetricsLayout()
  const { data: templateData } = useCustomMetricsTemplate(activeTemplateId)
  const [layout, setLayout] = useState<CustomMetricsLayout>({ sections: [] })
  const [newSectionOpen, setNewSectionOpen] = useState(false)
  const [layoutSynced, setLayoutSynced] = useState(false)

  // Track which template was last synced to detect template switches
  const [syncedTemplateId, setSyncedTemplateId] = useState<string | null | undefined>(undefined)

  // Sync from server on first load or when switching templates (render-time state adjustment)
  if (activeTemplateId !== syncedTemplateId) {
    // Template changed — need to resync
    if (activeTemplateId === null) {
      // Switching to default
      if (layoutData) {
        setSyncedTemplateId(null)
        setLayout(layoutData.layout ?? { sections: [] })
      }
    } else {
      // Switching to a named template
      if (templateData && templateData.id === activeTemplateId) {
        setSyncedTemplateId(activeTemplateId)
        setLayout(templateData.layout ?? { sections: [] })
      }
    }
  } else if (!layoutSynced && activeTemplateId === null && layoutData) {
    // Initial sync for default layout
    setLayoutSynced(true)
    setSyncedTemplateId(null)
    if (layoutData.layout) {
      setLayout(layoutData.layout)
    }
  }

  // Open "New Section" dialog when header button triggers (render-time state adjustment)
  const [prevTrigger, setPrevTrigger] = useState(newSectionTrigger)
  if (newSectionTrigger !== prevTrigger) {
    setPrevTrigger(newSectionTrigger)
    setNewSectionOpen(true)
  }

  // Provide layout snapshot when requested by parent (for save-as-template)
  const [prevSnapshotTrigger, setPrevSnapshotTrigger] = useState(layoutSnapshotTrigger)
  if (layoutSnapshotTrigger !== prevSnapshotTrigger) {
    setPrevSnapshotTrigger(layoutSnapshotTrigger)
    onLayoutSnapshot(layout)
  }

  // Debounced auto-save
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const pendingLayoutRef = useRef<CustomMetricsLayout | null>(null)
  // Keep a ref to activeTemplateId so the debounced callback uses the latest value
  const activeTemplateIdRef = useRef(activeTemplateId)
  activeTemplateIdRef.current = activeTemplateId

  const persistLayout = useCallback(
    (newLayout: CustomMetricsLayout) => {
      setLayout(newLayout)
      const tid = activeTemplateIdRef.current
      if (tid === null) {
        // Optimistically update the React Query cache for default layout
        queryClient.setQueryData(["custom-metrics-layout"], { layout: newLayout })
      } else {
        // Optimistically update the React Query cache for the template
        queryClient.setQueryData(
          ["custom-metrics-template", tid],
          (old: { id: string; name: string; layout: CustomMetricsLayout; updated_at: string | null } | undefined) =>
            old ? { ...old, layout: newLayout } : old,
        )
      }
      pendingLayoutRef.current = newLayout
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current)
      saveTimeoutRef.current = setTimeout(async () => {
        if (pendingLayoutRef.current) {
          const layoutToSave = pendingLayoutRef.current
          pendingLayoutRef.current = null
          const currentTid = activeTemplateIdRef.current
          if (currentTid === null) {
            await saveLayout(layoutToSave)
          } else {
            await saveTemplateLayout(currentTid, layoutToSave)
          }
        }
      }, 500)
    },
    [queryClient],
  )

  // Flush any pending layout save on unmount so changes aren't lost
  // (e.g. when the user navigates away within the 500ms debounce window).
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current)
      }
      if (pendingLayoutRef.current) {
        const layoutToSave = pendingLayoutRef.current
        pendingLayoutRef.current = null
        const currentTid = activeTemplateIdRef.current
        if (currentTid === null) {
          saveLayout(layoutToSave)
        } else {
          saveTemplateLayout(currentTid, layoutToSave)
        }
      }
    }
  }, [])

  const catalog = useMemo(
    () =>
      buildPlotCatalog(customMetricSections, availableRewardNames, evalsList),
    [customMetricSections, availableRewardNames, evalsList],
  )

  const shouldFetchStepTimes = xAxisMode === "time"
  const st0 = useStepTimes(
    runs[0]?.runPath || "",
    shouldFetchStepTimes && !!runs[0],
    shouldPoll,
  )
  const st1 = useStepTimes(
    runs[1]?.runPath || "",
    shouldFetchStepTimes && !!runs[1],
    shouldPoll,
  )
  const st2 = useStepTimes(
    runs[2]?.runPath || "",
    shouldFetchStepTimes && !!runs[2],
    shouldPoll,
  )
  const st3 = useStepTimes(
    runs[3]?.runPath || "",
    shouldFetchStepTimes && !!runs[3],
    shouldPoll,
  )
  const st4 = useStepTimes(
    runs[4]?.runPath || "",
    shouldFetchStepTimes && !!runs[4],
    shouldPoll,
  )
  const st5 = useStepTimes(
    runs[5]?.runPath || "",
    shouldFetchStepTimes && !!runs[5],
    shouldPoll,
  )
  const st6 = useStepTimes(
    runs[6]?.runPath || "",
    shouldFetchStepTimes && !!runs[6],
    shouldPoll,
  )
  const st7 = useStepTimes(
    runs[7]?.runPath || "",
    shouldFetchStepTimes && !!runs[7],
    shouldPoll,
  )
  const st8 = useStepTimes(
    runs[8]?.runPath || "",
    shouldFetchStepTimes && !!runs[8],
    shouldPoll,
  )
  const st9 = useStepTimes(
    runs[9]?.runPath || "",
    shouldFetchStepTimes && !!runs[9],
    shouldPoll,
  )
  const st10 = useStepTimes(
    runs[10]?.runPath || "",
    shouldFetchStepTimes && !!runs[10],
    shouldPoll,
  )
  const st11 = useStepTimes(
    runs[11]?.runPath || "",
    shouldFetchStepTimes && !!runs[11],
    shouldPoll,
  )
  const st12 = useStepTimes(
    runs[12]?.runPath || "",
    shouldFetchStepTimes && !!runs[12],
    shouldPoll,
  )
  const st13 = useStepTimes(
    runs[13]?.runPath || "",
    shouldFetchStepTimes && !!runs[13],
    shouldPoll,
  )
  const st14 = useStepTimes(
    runs[14]?.runPath || "",
    shouldFetchStepTimes && !!runs[14],
    shouldPoll,
  )
  const st15 = useStepTimes(
    runs[15]?.runPath || "",
    shouldFetchStepTimes && !!runs[15],
    shouldPoll,
  )
  const st16 = useStepTimes(
    runs[16]?.runPath || "",
    shouldFetchStepTimes && !!runs[16],
    shouldPoll,
  )
  const st17 = useStepTimes(
    runs[17]?.runPath || "",
    shouldFetchStepTimes && !!runs[17],
    shouldPoll,
  )
  const st18 = useStepTimes(
    runs[18]?.runPath || "",
    shouldFetchStepTimes && !!runs[18],
    shouldPoll,
  )
  const st19 = useStepTimes(
    runs[19]?.runPath || "",
    shouldFetchStepTimes && !!runs[19],
    shouldPoll,
  )

  const stQueries = useMemo(
    () => [st0, st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11, st12, st13, st14, st15, st16, st17, st18, st19],
    [st0, st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11, st12, st13, st14, st15, st16, st17, st18, st19],
  )

  const { isStepTimesFetching, isStepTimesRefetching } = useMemo(() => {
    if (!shouldFetchStepTimes)
      return { isStepTimesFetching: false, isStepTimesRefetching: false }
    const fetching = stQueries.some((q) => q.isFetching)
    const fetchingQueries = stQueries.filter((q) => q.isFetching)
    const refetching =
      fetching &&
      fetchingQueries.length > 0 &&
      fetchingQueries.every((q) => !!q.data)
    return { isStepTimesFetching: fetching, isStepTimesRefetching: refetching }
  }, [shouldFetchStepTimes, stQueries])

  const { stepTimesByRun, firstStepTimesByRun } = useMemo(() => {
    if (!shouldFetchStepTimes) {
      return {
        stepTimesByRun: new Map<string, Map<number, number>>(),
        firstStepTimesByRun: new Map<string, number>(),
      }
    }
    const timesByRun = new Map<string, Map<number, number>>()
    const firstTimesByRun = new Map<string, number>()
    runs.forEach((run, index) => {
      if (index >= 20) return
      const query = stQueries[index]
      const data = query.data
      if (
        !data ||
        data.first_step_time === null ||
        data.first_step_time === undefined
      )
        return
      if (!data.step_times || data.step_times.length === 0) return
      const stepTimeMap = new Map<number, number>()
      data.step_times.forEach((entry: { step: number; time: number }) => {
        if (typeof entry.time === "number")
          stepTimeMap.set(entry.step, entry.time)
      })
      if (stepTimeMap.size > 0) {
        timesByRun.set(run.runPath, stepTimeMap)
        firstTimesByRun.set(run.runPath, data.first_step_time)
      }
    })
    return { stepTimesByRun: timesByRun, firstStepTimesByRun: firstTimesByRun }
  }, [runs, shouldFetchStepTimes, stQueries])

  const chartProps: SharedChartProps = useMemo(
    () => ({
      runs,
      shouldPoll,
      showEma,
      emaSpan,
      hoveredRunId,
      xAxisMode,
      scrollRoot,
      maxStep,
      maxTime,
      stepTimesByRun,
      firstStepTimesByRun,
      isStepTimesFetching,
      isStepTimesRefetching,
      availableSampleTags,
    }),
    [
      runs,
      shouldPoll,
      showEma,
      emaSpan,
      hoveredRunId,
      xAxisMode,
      scrollRoot,
      maxStep,
      maxTime,
      stepTimesByRun,
      firstStepTimesByRun,
      isStepTimesFetching,
      isStepTimesRefetching,
      availableSampleTags,
    ],
  )

  const sectionNames = layout.sections.map((s) => s.name)

  const handleAddSection = (name: string) => {
    const newSection: CustomSection = {
      id: generateId(),
      name,
      groups: [],
      plots: [],
    }
    persistLayout({ sections: [...layout.sections, newSection] })
  }

  const handleUpdateSection = (sectionId: string, updated: CustomSection) => {
    persistLayout({
      sections: layout.sections.map((s) => (s.id === sectionId ? updated : s)),
    })
  }

  const handleRemoveSection = (sectionId: string) => {
    persistLayout({
      sections: layout.sections.filter((s) => s.id !== sectionId),
    })
  }

  const sectionSensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  )

  const handleSectionDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    if (!over || active.id === over.id) return
    const oldIdx = layout.sections.findIndex((s) => s.id === active.id)
    const newIdx = layout.sections.findIndex((s) => s.id === over.id)
    if (oldIdx === -1 || newIdx === -1) return
    persistLayout({ sections: arrayMove(layout.sections, oldIdx, newIdx) })
  }

  return (
    <div className="-mt-3">
      <DndContext
        sensors={sectionSensors}
        collisionDetection={closestCenter}
        onDragEnd={handleSectionDragEnd}
      >
        <SortableContext
          items={layout.sections.map((s) => s.id)}
          strategy={verticalListSortingStrategy}
        >
          {layout.sections.map((section, idx) => (
            <div key={section.id}>
              {idx > 0 && <div className="border-t border-gray-200 my-3" />}
              <SortableSection
                section={section}
                catalog={catalog}
                allSectionNames={sectionNames}
                onUpdate={(updated) => handleUpdateSection(section.id, updated)}
                onRemove={() => handleRemoveSection(section.id)}
                chartProps={chartProps}
              />
            </div>
          ))}
        </SortableContext>
      </DndContext>

      {layout.sections.length === 0 && layoutLoading && (
        <div className="flex items-center justify-center py-16 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin mr-2" />
          <span className="text-sm">Loading...</span>
        </div>
      )}

      {layout.sections.length === 0 && !layoutLoading && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <p className="text-sm text-muted-foreground mb-3">
            No custom sections yet. Create one to start building your dashboard.
          </p>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setNewSectionOpen(true)}
          >
            <Plus className="h-3.5 w-3.5 mr-1" />
            New Section
          </Button>
        </div>
      )}

      <NameDialog
        open={newSectionOpen}
        onOpenChange={setNewSectionOpen}
        title="New Section"
        existingNames={sectionNames}
        onSubmit={handleAddSection}
      />
    </div>
  )
}
