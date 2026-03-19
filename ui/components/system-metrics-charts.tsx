
import { useState, useMemo, useCallback, useEffect, useRef } from "react"
import { useAtomValue } from "jotai"
import { darkModeAtom } from "@/lib/atoms"
import { ChevronDown, X, SlidersHorizontal, Loader2 } from "lucide-react"
import uPlot from "uplot"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { GpuMetricChart, SYSTEM_METRIC_INFO, type SeriesRoleMap } from "@/components/gpu-metric-chart"
import { cn } from "@/lib/utils"
import {
  formatClockTimeAdaptive,
  formatSecondsCompact,
  formatSecondsHuman,
  formatValueSmart,
} from "@/lib/format"
import type { GpuMetric, CpuMetric, VllmMetric } from "@/lib/types"

// ============================================================================
// Constants
// ============================================================================

const MULTI_SERIES_COLORS = [
  "#3b82f6",
  "#ef4444",
  "#22c55e",
  "#f97316",
  "#8b5cf6",
  "#06b6d4",
  "#eab308",
  "#ec4899",
]

// Ordered list of GPU metric names to display
const GPU_SYSTEM_METRICS = [
  "gpu_utilization_percent",
  "gpu_memory_used_percent",
  "gpu_memory_bandwidth_utilization_percent",
  "gpu_memory_used_gb",
  "gpu_memory_free_gb",
  "gpu_memory_total_gb",
  "gpu_temperature_c",
  "gpu_power_w",
  "gpu_power_limit_w",
  "gpu_clock_sm_mhz",
  "gpu_clock_mem_mhz",
  "gpu_fan_speed_percent",
]

const GPU_TORCH_METRICS = [
  "torch_allocated_gb",
  "torch_reserved_gb",
  "torch_max_allocated_gb",
]

// Aggregate CPU metrics (skip per-core)
const CPU_AGGREGATE_METRIC_NAMES = [
  "cpu_utilization_percent",
  "system_memory_used_gb",
  "system_memory_available_gb",
  "system_memory_total_gb",
  "system_memory_percent",
]

// Ordered list of vLLM metric names to display
const VLLM_METRIC_NAMES = [
  "requests_running",
  "requests_waiting",
  "cache_usage",
  "cache_hit_rate",
  "requests_total",
  "prompt_tokens_total",
  "rollout_tokens_total",
  "preemptions_total",
  "cache_hits_total",
  "cache_queries_total",
  "ttft_mean",
  "e2e_latency_mean",
  "itl_mean",
]

// ============================================================================
// Types
// ============================================================================

export interface GpuIdentifier {
  node_id: number
  gpu_index: number
}

export interface SystemMetricsChartsProps {
  gpuMetrics: GpuMetric[]
  cpuMetrics: CpuMetric[]
  vllmMetrics?: VllmMetric[]
  availableGpuMetrics: string[]
  availableCpuMetrics?: string[]
  availableVllmMetrics?: string[]
  trainerGpus: GpuIdentifier[]
  inferenceGpus: GpuIdentifier[]
  isLoading?: boolean
  isRefetching?: boolean
  systemMetricsOpen?: boolean
  onSystemMetricsOpenChange?: (open: boolean) => void
  cpuMetricsOpen?: boolean
  onCpuMetricsOpenChange?: (open: boolean) => void
  vllmMetricsOpen?: boolean
  onVllmMetricsOpenChange?: (open: boolean) => void
  roleMode?: "combined" | "separated"
  onRoleModeChange?: (mode: "combined" | "separated") => void
  trainerSectionOpen?: boolean
  onTrainerSectionOpenChange?: (open: boolean) => void
  inferenceSectionOpen?: boolean
  onInferenceSectionOpenChange?: (open: boolean) => void
  trainerNodeMode?: "combined" | "separated"
  onTrainerNodeModeChange?: (mode: "combined" | "separated") => void
  inferenceNodeMode?: "combined" | "separated"
  onInferenceNodeModeChange?: (mode: "combined" | "separated") => void
  /** When provided, forces all charts to use this time range (e.g. for Live mode). */
  overrideTimeRange?: { start: number; end: number } | null
  /** Offset in seconds from run start to the current window start, for global x-axis labels */
  xOffset?: number
}

// ============================================================================
// Helper: compute time range from metrics
// ============================================================================

function computeTimeRange(metrics: Array<{ timestamp: number }>): {
  start: number
  end: number
} {
  if (metrics.length === 0) return { start: 0, end: 1 }
  let min = Infinity
  let max = -Infinity
  for (const m of metrics) {
    if (m.timestamp < min) min = m.timestamp
    if (m.timestamp > max) max = m.timestamp
  }
  return { start: min, end: max }
}

// ============================================================================
// Helper: compute GPU series info for labels
// ============================================================================

interface GpuSeriesLabel {
  seriesKey: string
  node_id: number | null
  gpu_index: number
  label: string
  color: string
}

function buildGpuSeriesKey(
  nodeId: number | null,
  gpuIndex: number,
  multiNode: boolean
): string {
  return multiNode ? `${nodeId ?? "?"}:${gpuIndex}` : `gpu:${gpuIndex}`
}

function withAlpha(color: string, alpha: number): string {
  if (color.startsWith("#")) {
    const hex = color.slice(1)
    const normalized =
      hex.length === 3
        ? hex
            .split("")
            .map((ch) => ch + ch)
            .join("")
        : hex
    if (normalized.length === 6) {
      const r = Number.parseInt(normalized.slice(0, 2), 16)
      const g = Number.parseInt(normalized.slice(2, 4), 16)
      const b = Number.parseInt(normalized.slice(4, 6), 16)
      return `rgba(${r}, ${g}, ${b}, ${alpha})`
    }
  }
  return color
}

function computeGpuLabels(
  data: GpuMetric[],
  multiNode: boolean
): GpuSeriesLabel[] {
  const seen = new Map<string, { node_id: number | null; gpu_index: number }>()
  const sorted = [...data].sort((a, b) => {
    const nodeA =
      typeof a.node_id === "number" && Number.isFinite(a.node_id)
        ? a.node_id
        : Number.MAX_SAFE_INTEGER
    const nodeB =
      typeof b.node_id === "number" && Number.isFinite(b.node_id)
        ? b.node_id
        : Number.MAX_SAFE_INTEGER
    if (nodeA !== nodeB) return nodeA - nodeB
    return a.gpu_index - b.gpu_index
  })
  for (const m of sorted) {
    const nodeId =
      typeof m.node_id === "number" && Number.isFinite(m.node_id)
        ? m.node_id
        : null
    const key = `${nodeId ?? "?"}:${m.gpu_index}`
    if (!seen.has(key)) {
      seen.set(key, { node_id: nodeId, gpu_index: m.gpu_index })
    }
  }
  return Array.from(seen.values()).map((entry, idx) => ({
    seriesKey: buildGpuSeriesKey(entry.node_id, entry.gpu_index, multiNode),
    ...entry,
    label: multiNode
      ? `Node ${entry.node_id ?? "?"} GPU ${entry.gpu_index}`
      : `GPU ${entry.gpu_index}`,
    // Match GpuMetricChart color assignment:
    // - multi-node ("node_gpu") uses sequential series colors
    // - single-node ("gpu") uses gpu_index-based colors
    color: multiNode
      ? MULTI_SERIES_COLORS[idx % MULTI_SERIES_COLORS.length]
      : MULTI_SERIES_COLORS[entry.gpu_index % MULTI_SERIES_COLORS.length],
  }))
}

function computeGpuLabelsFromIdentifiers(
  gpus: GpuIdentifier[],
  multiNode: boolean
): GpuSeriesLabel[] {
  const dedup = new Map<string, { node_id: number; gpu_index: number }>()
  for (const gpu of gpus) {
    const key = `${gpu.node_id}:${gpu.gpu_index}`
    if (!dedup.has(key)) {
      dedup.set(key, { node_id: gpu.node_id, gpu_index: gpu.gpu_index })
    }
  }
  const ordered = Array.from(dedup.values()).sort(
    (a, b) => a.node_id - b.node_id || a.gpu_index - b.gpu_index
  )
  return ordered.map((entry, idx) => ({
    seriesKey: buildGpuSeriesKey(entry.node_id, entry.gpu_index, multiNode),
    node_id: entry.node_id,
    gpu_index: entry.gpu_index,
    label: multiNode ? `Node ${entry.node_id} GPU ${entry.gpu_index}` : `GPU ${entry.gpu_index}`,
    color: multiNode
      ? MULTI_SERIES_COLORS[idx % MULTI_SERIES_COLORS.length]
      : MULTI_SERIES_COLORS[entry.gpu_index % MULTI_SERIES_COLORS.length],
  }))
}

// ============================================================================
// GPU Labels Components
// ============================================================================

function GpuLabels({
  series,
  activeSeriesKeys,
  onSeriesHover,
  onSeriesToggle,
}: {
  series: GpuSeriesLabel[]
  activeSeriesKeys?: Set<string> | null
  onSeriesHover?: (seriesKey: string | null) => void
  onSeriesToggle?: (seriesKey: string) => void
}) {
  if (series.length === 0) return null
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 mb-3">
      {series.map((s) => {
        const isActive = !!activeSeriesKeys?.has(s.seriesKey)
        const isDimmed = !!activeSeriesKeys && activeSeriesKeys.size > 0 && !isActive
        return (
          <div
            key={s.seriesKey}
            className={cn(
              "flex items-center gap-1.5 select-none transition-opacity duration-150",
              onSeriesToggle && "cursor-pointer",
              isActive && "ring-1 ring-border rounded-full px-1.5 py-0.5 -mx-1.5 -my-0.5"
            )}
            style={{ opacity: isDimmed ? 0.3 : 1 }}
            onPointerEnter={
              onSeriesHover ? () => onSeriesHover(s.seriesKey) : undefined
            }
            onPointerLeave={onSeriesHover ? () => onSeriesHover(null) : undefined}
            onClick={onSeriesToggle ? () => onSeriesToggle(s.seriesKey) : undefined}
          >
            <div
              className="w-2.5 h-2.5 rounded-sm"
              style={{
                backgroundColor: isDimmed ? withAlpha(s.color, 0.25) : s.color,
              }}
            />
            <span className="text-xs text-muted-foreground">{s.label}</span>
          </div>
        )
      })}
    </div>
  )
}

/** Labels grouped by role (Trainer / Inference) */
function RoleGroupedGpuLabels({
  trainerLabels,
  inferenceLabels,
  activeSeriesKeys,
  onSeriesHover,
  onSeriesToggle,
}: {
  trainerLabels: GpuSeriesLabel[]
  inferenceLabels: GpuSeriesLabel[]
  activeSeriesKeys?: Set<string> | null
  onSeriesHover?: (seriesKey: string | null) => void
  onSeriesToggle?: (seriesKey: string) => void
}) {
  if (trainerLabels.length === 0 && inferenceLabels.length === 0) return null
  return (
    <div className="space-y-2 mb-3">
      {trainerLabels.length > 0 && (
        <div>
          <div className="text-[11px] font-semibold text-muted-foreground mb-1">Trainer</div>
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {trainerLabels.map((s) => {
              const isActive = !!activeSeriesKeys?.has(s.seriesKey)
              const isDimmed = !!activeSeriesKeys && activeSeriesKeys.size > 0 && !isActive
              return (
                <div
                  key={`trainer-${s.seriesKey}`}
                  className={cn(
                    "flex items-center gap-1.5 select-none transition-opacity duration-150",
                    onSeriesToggle && "cursor-pointer",
                    isActive &&
                      "ring-1 ring-border rounded-full px-1.5 py-0.5 -mx-1.5 -my-0.5"
                  )}
                  style={{ opacity: isDimmed ? 0.3 : 1 }}
                  onPointerEnter={
                    onSeriesHover ? () => onSeriesHover(s.seriesKey) : undefined
                  }
                  onPointerLeave={onSeriesHover ? () => onSeriesHover(null) : undefined}
                  onClick={onSeriesToggle ? () => onSeriesToggle(s.seriesKey) : undefined}
                >
                  <div
                    className="w-2.5 h-2.5 rounded-sm"
                    style={{
                      backgroundColor: isDimmed ? withAlpha(s.color, 0.25) : s.color,
                    }}
                  />
                  <span className="text-xs text-muted-foreground">{s.label}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
      {inferenceLabels.length > 0 && (
        <div>
          <div className="text-[11px] font-semibold text-muted-foreground mb-1">Inference</div>
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {inferenceLabels.map((s) => {
              const isActive = !!activeSeriesKeys?.has(s.seriesKey)
              const isDimmed = !!activeSeriesKeys && activeSeriesKeys.size > 0 && !isActive
              return (
                <div
                  key={`inference-${s.seriesKey}`}
                  className={cn(
                    "flex items-center gap-1.5 select-none transition-opacity duration-150",
                    onSeriesToggle && "cursor-pointer",
                    isActive &&
                      "ring-1 ring-border rounded-full px-1.5 py-0.5 -mx-1.5 -my-0.5"
                  )}
                  style={{ opacity: isDimmed ? 0.3 : 1 }}
                  onPointerEnter={
                    onSeriesHover ? () => onSeriesHover(s.seriesKey) : undefined
                  }
                  onPointerLeave={onSeriesHover ? () => onSeriesHover(null) : undefined}
                  onClick={onSeriesToggle ? () => onSeriesToggle(s.seriesKey) : undefined}
                >
                  <div
                    className="w-2.5 h-2.5 rounded-sm"
                    style={{
                      backgroundColor: isDimmed ? withAlpha(s.color, 0.25) : s.color,
                    }}
                  />
                  <span className="text-xs text-muted-foreground">{s.label}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Formatting helpers (matching step-metrics-charts)
// ============================================================================

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

// ============================================================================
// Nearest-point lookup for sparse data (matching step-metrics-charts)
// ============================================================================

const TIME_TOOLTIP_MAX_DISTANCE_PX = 28

function findNearestDefinedIndex(
  series: ArrayLike<number | null | undefined> | undefined,
  xValues: ArrayLike<number>,
  targetIdx: number,
  maxDistanceX?: number | null
): number | null {
  if (!series || series.length === 0 || targetIdx < 0 || targetIdx >= series.length) {
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

// ============================================================================
// IQR outlier detection (matching step-metrics-charts)
// ============================================================================

function computeIQRBounds(values: number[]): { lower: number; upper: number } | null {
  if (values.length < 4) return null

  const sorted = [...values].sort((a, b) => a - b)
  const n = sorted.length

  const q1Index = (n - 1) * 0.25
  const q3Index = (n - 1) * 0.75

  const q1 = sorted[Math.floor(q1Index)] +
    (q1Index % 1) * (sorted[Math.ceil(q1Index)] - sorted[Math.floor(q1Index)])
  const q3 = sorted[Math.floor(q3Index)] +
    (q3Index % 1) * (sorted[Math.ceil(q3Index)] - sorted[Math.floor(q3Index)])

  const iqr = q3 - q1
  const multiplier = 6.0

  return {
    lower: q1 - multiplier * iqr,
    upper: q3 + multiplier * iqr,
  }
}

// ============================================================================
// Filter badge (matching step-metrics-charts)
// ============================================================================

function FilterBadge({
  label,
  onRemove,
}: {
  label: string
  onRemove: () => void
}) {
  return (
    <span
      className="group inline-flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] bg-background border border-border text-muted-foreground rounded-full cursor-pointer hover:bg-muted transition-colors"
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

// ============================================================================
// CPU Metric Chart Component (matching MetricChart style from /metrics)
// ============================================================================

function CpuMetricChart({
  metricName,
  data,
  isLoading,
  isRefetching,
  overrideTimeRange,
  xOffset = 0,
}: {
  metricName: string
  data: CpuMetric[]
  isLoading?: boolean
  isRefetching?: boolean
  overrideTimeRange?: { start: number; end: number } | null
  /** Offset in seconds from run start to window start, for global x-axis labels */
  xOffset?: number
}) {
  const darkMode = useAtomValue(darkModeAtom)
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<uPlot | null>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const [ignoreOutliers, setIgnoreOutliers] = useState(false)
  const [minY, setMinY] = useState<number | null>(null)
  const [maxY, setMaxY] = useState<number | null>(null)

  const metricInfo = SYSTEM_METRIC_INFO[metricName] || {
    label: metricName.replace(/_/g, " "),
    unit: "",
  }

  // Group by node_id
  const { uplotData, seriesConfig, hasData, nodeIds, timeRange, outlierBounds } =
    useMemo(() => {
      if (data.length === 0) {
        const fallbackRange = overrideTimeRange ?? { start: 0, end: 1 }
        return {
          uplotData: null,
          seriesConfig: [] as uPlot.Series[],
          hasData: false,
          nodeIds: [] as number[],
          timeRange: fallbackRange,
          outlierBounds: null,
        }
      }

      // Collect node IDs
      const nodeIdSet = new Set<number>()
      let minTime = Infinity
      let maxTime = -Infinity
      const allValues: number[] = []
      for (const m of data) {
        const nodeId =
          typeof m.node_id === "number" && Number.isFinite(m.node_id)
            ? m.node_id
            : 0
        nodeIdSet.add(nodeId)
        if (m.timestamp < minTime) minTime = m.timestamp
        if (m.timestamp > maxTime) maxTime = m.timestamp
        allValues.push(m.value)
      }
      const sortedNodeIds = Array.from(nodeIdSet).sort((a, b) => a - b)
      const tr = overrideTimeRange ?? { start: minTime, end: maxTime }

      // Build time-indexed data
      const byTimestamp: Record<number, Record<number, number>> = {}
      const timestamps = new Set<number>()
      for (const m of data) {
        const nodeId =
          typeof m.node_id === "number" && Number.isFinite(m.node_id)
            ? m.node_id
            : 0
        timestamps.add(m.timestamp)
        if (!byTimestamp[m.timestamp]) byTimestamp[m.timestamp] = {}
        byTimestamp[m.timestamp][nodeId] = m.value
      }

      const sortedTimes = Array.from(timestamps).sort((a, b) => a - b)
      if (sortedTimes.length === 0) {
        return {
          uplotData: null,
          seriesConfig: [] as uPlot.Series[],
          hasData: false,
          nodeIds: sortedNodeIds,
          timeRange: tr,
          outlierBounds: null,
        }
      }

      const xData = sortedTimes.map((t) => t - tr.start)
      const series: uPlot.Series[] = [{ label: "Time" }]
      const dataArrays: (number | null)[][] = [xData]

      for (let i = 0; i < sortedNodeIds.length; i++) {
        const nodeId = sortedNodeIds[i]
        const values: (number | null)[] = sortedTimes.map(
          (t) => byTimestamp[t]?.[nodeId] ?? null
        )
        dataArrays.push(values)
        series.push({
          label: `Node ${nodeId}`,
          stroke: MULTI_SERIES_COLORS[i % MULTI_SERIES_COLORS.length],
          width: 1.5,
          spanGaps: true,
          points: { show: false },
        })
      }

      return {
        uplotData: dataArrays as uPlot.AlignedData,
        seriesConfig: series,
        hasData: true,
        nodeIds: sortedNodeIds,
        timeRange: tr,
        outlierBounds: computeIQRBounds(allValues),
      }
    }, [data, overrideTimeRange])

  const showLegend = nodeIds.length > 1

  // Format y-axis values smartly
  const formatYAxisTick = useCallback(
    (v: number): string => {
      if (Math.abs(v) >= 1000) return `${parseFloat((v / 1000).toFixed(1))}k`
      if (Number.isInteger(v)) return v.toLocaleString()
      if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1)
      return String(parseFloat(v.toFixed(2)))
    },
    []
  )

  useEffect(() => {
    if (!containerRef.current || !uplotData || !hasData) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = 200

    let minVal = Number.POSITIVE_INFINITY
    let maxVal = Number.NEGATIVE_INFINITY
    for (let i = 1; i < uplotData.length; i += 1) {
      const values = uplotData[i]
      for (let j = 0; j < values.length; j += 1) {
        const value = values[j]
        if (value !== null && value !== undefined) {
          if (ignoreOutliers && outlierBounds) {
            if (value >= outlierBounds.lower && value <= outlierBounds.upper) {
              if (value < minVal) minVal = value
              if (value > maxVal) maxVal = value
            }
          } else {
            if (value < minVal) minVal = value
            if (value > maxVal) maxVal = value
          }
        }
      }
    }

    // If all values were outliers, fall back to using the bounds
    if (minVal === Number.POSITIVE_INFINITY && outlierBounds) {
      minVal = outlierBounds.lower
      maxVal = outlierBounds.upper
    }

    const range = maxVal - minVal
    const absMax = Math.max(Math.abs(maxVal), Math.abs(minVal))
    const minPadding = Math.max(absMax * 0.05, 1e-10)
    const pad = Math.max(range * 0.1, minPadding)
    const yDomain: [number, number] = [
      minY !== null ? minY : Math.max(0, minVal - pad),
      maxY !== null ? maxY : maxVal + pad,
    ]
    const gridColor = darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(128, 128, 128, 0.15)"
    const tickLabelColor = darkMode ? "rgba(255, 255, 255, 0.65)" : "rgba(100, 100, 100, 0.9)"

    // Dynamic Y axis size calculation based on tick label width
    const calcYAxisSize = (_u: uPlot, values: string[]) => {
      if (!values || values.length === 0) return 40
      const maxLen = Math.max(...values.map(v => v.length))
      return Math.max(40, maxLen * 7 + 14)
    }

    const opts: uPlot.Options = {
      width,
      height,
      // Reserve horizontal margins so start/end clock labels are not clipped.
      padding: [4, 24, 0, 4],
      cursor: {
        show: true,
        x: true,
        y: false,
        points: { show: false },
        drag: { x: true, y: false },
      },
      legend: { show: showLegend },
      select: { show: true, left: 0, top: 0, width: 0, height: 0 },
      scales: {
        x: {
          time: false,
          range: (_u: uPlot, dataMin: number, dataMax: number) => {
            const windowEnd = timeRange.end - timeRange.start
            if (windowEnd > 0) {
              // Keep viewport pinned to the selected interval.
              return [0, windowEnd]
            }
            return [dataMin, dataMax]
          },
        },
        y: { range: yDomain },
      },
      axes: [
        {
          stroke: tickLabelColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: 24,
              splits: (
                _u: uPlot,
                _axisIdx: number,
                min: number,
                max: number
              ) => [min, max],
              values: (_u, vals) =>
                vals.map((v) =>
                  formatClockTimeAdaptive(
                    timeRange.start + Number(v),
                    timeRange.end - timeRange.start
                  )
                ),
        },
        {
          stroke: tickLabelColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: calcYAxisSize,
          values: (_, vals) => vals.map(formatYAxisTick),
        },
      ],
      series: seriesConfig,
      hooks: {
        setCursor: [
          (u) => {
            if (!tooltipRef.current || !containerRef.current) return
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

            const relTime = uplotData[0][idx]
            if (relTime === undefined) {
              tooltipRef.current.style.display = "none"
              return
            }

            // Compute max distance for nearest-point snapping
            const xSeries = uplotData[0] as ArrayLike<number>
            const maxDistanceX = (() => {
              const xScale = u.scales.x
              const minX = Number(xScale.min)
              const maxX = Number(xScale.max)
              if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return null
              const xRange = Math.abs(maxX - minX)
              if (xRange <= 0) return null
              const plotWidthPx = Math.max(1, u.bbox.width)
              return (xRange * TIME_TOOLTIP_MAX_DISTANCE_PX) / plotWidthPx
            })()

            const elapsedStr = formatTooltipElapsedTime(Number(relTime) + xOffset)
            const absTimestamp = timeRange.start + Number(relTime)
            const clockStr = formatClockTimeAdaptive(
              absTimestamp,
              timeRange.end - timeRange.start
            )

            let html = `<div class="font-medium mb-1">${clockStr}</div>`
            html += `<div class="text-[10px] text-muted-foreground font-sans">Time Run: ${elapsedStr}</div>`
            html += `<div class="text-[10px] text-muted-foreground font-sans mb-1">Timestamp: ${absTimestamp.toFixed(3)}</div>`
            let hasAnyValue = false

            for (let i = 0; i < nodeIds.length; i += 1) {
              const valueSeries = uplotData[i + 1] as
                | ArrayLike<number | null | undefined>
                | undefined

              const valuePointIdx = findNearestDefinedIndex(
                valueSeries,
                xSeries,
                idx,
                maxDistanceX
              )
              if (valuePointIdx === null) continue

              const value = valueSeries?.[valuePointIdx]
              if (value === null || value === undefined) continue

              hasAnyValue = true
              const formatted = formatValueSmart(value)

              html += `
                <div class="flex items-center gap-2 mt-0.5">
                  <div class="w-2 h-2 rounded-full shrink-0" style="background-color: ${MULTI_SERIES_COLORS[i % MULTI_SERIES_COLORS.length]}"></div>
                  <span class="text-muted-foreground">Node ${nodeIds[i]}:</span>
                  <span class="font-medium">${formatted} ${metricInfo.unit}</span>
                </div>
              `
            }

            if (!hasAnyValue) {
              tooltipRef.current.style.display = "none"
              return
            }

            tooltipRef.current.innerHTML = html
            tooltipRef.current.style.display = "block"

            const containerRect = containerRef.current.getBoundingClientRect()
            const tooltipRect = tooltipRef.current.getBoundingClientRect()

            let tooltipX = containerRect.left + left + 15
            if (tooltipX + tooltipRect.width + 20 > window.innerWidth) {
              tooltipX = containerRect.left + left - tooltipRect.width - 15 + u.bbox.left
            }
            tooltipX = Math.max(4, tooltipX)

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
        const { width: newWidth } = entry.contentRect
        if (chart && newWidth > 0) {
          chart.setSize({ width: newWidth, height })
        }
      }
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
      chart.destroy()
      chartRef.current = null
    }
  }, [uplotData, hasData, seriesConfig, metricName, timeRange, xOffset, showLegend, metricInfo.unit, nodeIds, formatYAxisTick, ignoreOutliers, outlierBounds, minY, maxY, darkMode])

  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) tooltipRef.current.style.display = "none"
  }, [])

  return (
    <div
      className={cn(
        "group/chart rounded-lg border border-border p-3 transition-opacity bg-background",
        isLoading && "opacity-60"
      )}
    >
      <div className="flex items-center justify-between mb-2 gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <h4 className="text-xs font-medium truncate" title={metricName}>{metricInfo.label}</h4>
          {ignoreOutliers && (
            <FilterBadge label="Ignore Outliers" onRemove={() => setIgnoreOutliers(false)} />
          )}
          {minY !== null && (
            <FilterBadge label={`Min Y: ${minY}`} onRemove={() => setMinY(null)} />
          )}
          {maxY !== null && (
            <FilterBadge label={`Max Y: ${maxY}`} onRemove={() => setMaxY(null)} />
          )}
        </div>
        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                className="h-5 px-1.5 text-[10px] rounded border border-border hover:bg-muted flex items-center gap-1 transition-all opacity-0 group-hover/chart:opacity-100"
              >
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
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      {hasData ? (
        <div
          className="h-[200px] relative bg-background rounded"
          ref={containerRef}
          onMouseLeave={handleMouseLeave}
        >
          {isRefetching && (
            <Loader2 className="absolute bottom-0.5 left-0.5 h-3 w-3 animate-spin text-muted-foreground" />
          )}
          <div
            ref={tooltipRef}
            className="fixed z-[9999] max-w-[360px] bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl border border-border pointer-events-none"
            style={{ display: "none" }}
          />
        </div>
      ) : (
        <div className="h-[200px] flex items-center justify-center text-muted-foreground text-xs rounded">
          {isLoading ? "Loading..." : `No data for ${metricInfo.label}`}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// GPU Metrics Grid
// ============================================================================

function GpuMetricsGrid({
  data,
  metricNames,
  intervalStart,
  intervalEnd,
  gpuIndices,
  seriesMode,
  isLoading,
  isRefetching,
  seriesRoles,
  activeSeriesKeys,
  xOffset = 0,
}: {
  data: GpuMetric[]
  metricNames: string[]
  intervalStart: number
  intervalEnd: number
  gpuIndices: number[]
  seriesMode?: "gpu" | "node_gpu"
  isLoading?: boolean
  isRefetching?: boolean
  seriesRoles?: SeriesRoleMap
  activeSeriesKeys?: Set<string> | null
  xOffset?: number
}) {
  // Only show metrics that have data
  const metricsWithData = useMemo(() => {
    const metricDataCounts = new Map<string, number>()
    for (const m of data) {
      metricDataCounts.set(
        m.metric_name,
        (metricDataCounts.get(m.metric_name) ?? 0) + 1
      )
    }
    return metricNames.filter((name) => (metricDataCounts.get(name) ?? 0) > 0)
  }, [data, metricNames])

  const metricsToRender = useMemo(() => {
    // Keep cards mounted while fetching, even if the current payload is empty.
    if (isLoading && metricNames.length > 0) return metricNames
    if (isLoading && metricNames.length === 0) return GPU_SYSTEM_METRICS
    return metricsWithData
  }, [isLoading, metricNames, metricsWithData])

  if (metricsToRender.length === 0 && !isLoading) {
    return (
      <div className="h-24 flex items-center justify-center text-muted-foreground text-xs rounded-lg border border-border bg-background">
        No GPU metrics data available
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
      {metricsToRender.map((metricName) => {
        const metricData = data.filter((m) => m.metric_name === metricName)
        return (
          <GpuMetricChart
            key={metricName}
            metricName={metricName}
            data={metricData}
            gpuIndices={gpuIndices}
            intervalStart={intervalStart}
            intervalEnd={intervalEnd}
            seriesMode={seriesMode}
            showLegend={false}
            isLoading={isLoading}
            isRefetching={isRefetching}
            seriesRoles={seriesRoles}
            activeSeriesKeys={activeSeriesKeys}
            xOffset={xOffset}
            xAxisMode="clock"
            xAxisTicks="bounds"
          />
        )
      })}
    </div>
  )
}

// ============================================================================
// CPU Metrics Grid
// ============================================================================

function CpuMetricsGrid({
  data,
  availableMetricNames,
  isLoading,
  isRefetching,
  overrideTimeRange,
  xOffset = 0,
}: {
  data: CpuMetric[]
  availableMetricNames?: string[]
  isLoading?: boolean
  isRefetching?: boolean
  overrideTimeRange?: { start: number; end: number } | null
  xOffset?: number
}) {
  // Get unique metric names (only aggregates, skip per-core)
  const metricNames = useMemo(() => {
    const names = new Set<string>()
    for (const m of data) {
      // Skip per-core metrics
      if (m.metric_name.startsWith("cpu_core_")) continue
      names.add(m.metric_name)
    }
    for (const name of availableMetricNames ?? []) {
      if (name.startsWith("cpu_core_")) continue
      names.add(name)
    }
    // Sort to show in a nice order
    const ordered = CPU_AGGREGATE_METRIC_NAMES.filter((n) => names.has(n))
    // Add any other non-core metrics
    for (const n of names) {
      if (!ordered.includes(n)) ordered.push(n)
    }
    return ordered
  }, [data, availableMetricNames])

  const metricsToRender = useMemo(() => {
    if (metricNames.length > 0) return metricNames
    if (isLoading) return CPU_AGGREGATE_METRIC_NAMES
    return metricNames
  }, [metricNames, isLoading])

  if (metricsToRender.length === 0 && !isLoading) {
    return (
      <div className="h-24 flex items-center justify-center text-muted-foreground text-xs rounded-lg border border-border bg-background">
        No CPU metrics data available
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
      {metricsToRender.map((metricName) => {
        const metricData = data.filter((m) => m.metric_name === metricName)
        return (
          <CpuMetricChart
            key={metricName}
            metricName={metricName}
            data={metricData}
            isLoading={isLoading}
            isRefetching={isRefetching}
            overrideTimeRange={overrideTimeRange}
            xOffset={xOffset}
          />
        )
      })}
    </div>
  )
}

// ============================================================================
// vLLM Metric Chart Component (per-server, matching CpuMetricChart style)
// ============================================================================

function VllmMetricChart({
  metricName,
  data,
  isLoading,
  isRefetching,
  activeServerIds,
  overrideTimeRange,
  xOffset = 0,
}: {
  metricName: string
  data: VllmMetric[]
  isLoading?: boolean
  isRefetching?: boolean
  activeServerIds?: Set<number> | null
  overrideTimeRange?: { start: number; end: number } | null
  xOffset?: number
}) {
  const darkMode = useAtomValue(darkModeAtom)
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<uPlot | null>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const [ignoreOutliers, setIgnoreOutliers] = useState(false)
  const [minY, setMinY] = useState<number | null>(null)
  const [maxY, setMaxY] = useState<number | null>(null)

  const metricInfo = SYSTEM_METRIC_INFO[metricName] || {
    label: metricName.replace(/_/g, " "),
    unit: "",
  }

  // Group by server
  const { uplotData, seriesConfig, hasData, serverIds, timeRange, outlierBounds } =
    useMemo(() => {
      if (data.length === 0) {
        const fallbackRange = overrideTimeRange ?? { start: 0, end: 1 }
        return {
          uplotData: null,
          seriesConfig: [] as uPlot.Series[],
          hasData: false,
          serverIds: [] as number[],
          timeRange: fallbackRange,
          outlierBounds: null,
        }
      }

      // Collect server IDs
      const serverIdSet = new Set<number>()
      let minTime = Infinity
      let maxTime = -Infinity
      const allValues: number[] = []
      for (const m of data) {
        serverIdSet.add(m.server)
        if (m.timestamp < minTime) minTime = m.timestamp
        if (m.timestamp > maxTime) maxTime = m.timestamp
        allValues.push(m.value)
      }
      const sortedServerIds = Array.from(serverIdSet).sort((a, b) => a - b)
      const tr = overrideTimeRange ?? { start: minTime, end: maxTime }

      // Build time-indexed data
      const byTimestamp: Record<number, Record<number, number>> = {}
      const timestamps = new Set<number>()
      for (const m of data) {
        timestamps.add(m.timestamp)
        if (!byTimestamp[m.timestamp]) byTimestamp[m.timestamp] = {}
        byTimestamp[m.timestamp][m.server] = m.value
      }

      const sortedTimes = Array.from(timestamps).sort((a, b) => a - b)
      if (sortedTimes.length === 0) {
        return {
          uplotData: null,
          seriesConfig: [] as uPlot.Series[],
          hasData: false,
          serverIds: sortedServerIds,
          timeRange: tr,
          outlierBounds: null,
        }
      }

      const xData = sortedTimes.map((t) => t - tr.start)
      const series: uPlot.Series[] = [{ label: "Time" }]
      const dataArrays: (number | null)[][] = [xData]

      for (let i = 0; i < sortedServerIds.length; i++) {
        const serverId = sortedServerIds[i]
        const values: (number | null)[] = sortedTimes.map(
          (t) => byTimestamp[t]?.[serverId] ?? null
        )
        const isDimmed =
          !!activeServerIds &&
          activeServerIds.size > 0 &&
          !activeServerIds.has(serverId)
        dataArrays.push(values)
        series.push({
          label: `Server ${serverId}`,
          stroke: isDimmed
            ? withAlpha(MULTI_SERIES_COLORS[i % MULTI_SERIES_COLORS.length], 0.25)
            : MULTI_SERIES_COLORS[i % MULTI_SERIES_COLORS.length],
          width: isDimmed ? 1 : 1.5,
          spanGaps: true,
          points: { show: false },
        })
      }

      return {
        uplotData: dataArrays as uPlot.AlignedData,
        seriesConfig: series,
        hasData: true,
        serverIds: sortedServerIds,
        timeRange: tr,
        outlierBounds: computeIQRBounds(allValues),
      }
    }, [data, overrideTimeRange, activeServerIds])

  const formatYAxisTick = useCallback(
    (v: number): string => {
      if (Math.abs(v) >= 1000) return `${parseFloat((v / 1000).toFixed(1))}k`
      if (Number.isInteger(v)) return v.toLocaleString()
      if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1)
      return String(parseFloat(v.toFixed(2)))
    },
    []
  )

  useEffect(() => {
    if (!containerRef.current || !uplotData || !hasData) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = 200

    let minVal = Number.POSITIVE_INFINITY
    let maxVal = Number.NEGATIVE_INFINITY
    for (let i = 1; i < uplotData.length; i += 1) {
      const values = uplotData[i]
      for (let j = 0; j < values.length; j += 1) {
        const value = values[j]
        if (value !== null && value !== undefined) {
          if (ignoreOutliers && outlierBounds) {
            if (value >= outlierBounds.lower && value <= outlierBounds.upper) {
              if (value < minVal) minVal = value
              if (value > maxVal) maxVal = value
            }
          } else {
            if (value < minVal) minVal = value
            if (value > maxVal) maxVal = value
          }
        }
      }
    }

    if (minVal === Number.POSITIVE_INFINITY && outlierBounds) {
      minVal = outlierBounds.lower
      maxVal = outlierBounds.upper
    }

    const range = maxVal - minVal
    const absMax = Math.max(Math.abs(maxVal), Math.abs(minVal))
    const minPadding = Math.max(absMax * 0.05, 1e-10)
    const pad = Math.max(range * 0.1, minPadding)
    const yDomain: [number, number] = [
      minY !== null ? minY : Math.max(0, minVal - pad),
      maxY !== null ? maxY : maxVal + pad,
    ]
    const gridColor = darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(128, 128, 128, 0.15)"
    const tickLabelColor = darkMode ? "rgba(255, 255, 255, 0.65)" : "rgba(100, 100, 100, 0.9)"

    const calcYAxisSize = (_u: uPlot, values: string[]) => {
      if (!values || values.length === 0) return 40
      const maxLen = Math.max(...values.map(v => v.length))
      return Math.max(40, maxLen * 7 + 14)
    }

    const opts: uPlot.Options = {
      width,
      height,
      padding: [4, 24, 0, 4],
      cursor: {
        show: true,
        x: true,
        y: false,
        points: { show: false },
        drag: { x: true, y: false },
      },
      legend: { show: false },
      select: { show: true, left: 0, top: 0, width: 0, height: 0 },
      scales: {
        x: {
          time: false,
          range: (_u: uPlot, dataMin: number, dataMax: number) => {
            const windowEnd = timeRange.end - timeRange.start
            if (windowEnd > 0) return [0, windowEnd]
            return [dataMin, dataMax]
          },
        },
        y: { range: yDomain },
      },
      axes: [
        {
          stroke: tickLabelColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: 24,
          splits: (
            _u: uPlot,
            _axisIdx: number,
            min: number,
            max: number
          ) => [min, max],
          values: (_u, vals) =>
            vals.map((v) =>
              formatClockTimeAdaptive(
                timeRange.start + Number(v),
                timeRange.end - timeRange.start
              )
            ),
        },
        {
          stroke: tickLabelColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "10px system-ui, sans-serif",
          labelFont: "10px system-ui, sans-serif",
          size: calcYAxisSize,
          values: (_, vals) => vals.map(formatYAxisTick),
        },
      ],
      series: seriesConfig,
      hooks: {
        setCursor: [
          (u) => {
            if (!tooltipRef.current || !containerRef.current) return
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

            const relTime = uplotData[0][idx]
            if (relTime === undefined) {
              tooltipRef.current.style.display = "none"
              return
            }

            const xSeries = uplotData[0] as ArrayLike<number>
            const maxDistanceX = (() => {
              const xScale = u.scales.x
              const minX = Number(xScale.min)
              const maxX = Number(xScale.max)
              if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return null
              const xRange = Math.abs(maxX - minX)
              if (xRange <= 0) return null
              const plotWidthPx = Math.max(1, u.bbox.width)
              return (xRange * TIME_TOOLTIP_MAX_DISTANCE_PX) / plotWidthPx
            })()

            const elapsedStr = formatTooltipElapsedTime(Number(relTime) + xOffset)
            const absTimestamp = timeRange.start + Number(relTime)
            const clockStr = formatClockTimeAdaptive(
              absTimestamp,
              timeRange.end - timeRange.start
            )

            let html = `<div class="font-medium mb-1">${clockStr}</div>`
            html += `<div class="text-[10px] text-muted-foreground font-sans">Time Run: ${elapsedStr}</div>`
            html += `<div class="text-[10px] text-muted-foreground font-sans mb-1">Timestamp: ${absTimestamp.toFixed(3)}</div>`
            let hasAnyValue = false

            for (let i = 0; i < serverIds.length; i += 1) {
              const valueSeries = uplotData[i + 1] as
                | ArrayLike<number | null | undefined>
                | undefined

              const valuePointIdx = findNearestDefinedIndex(
                valueSeries,
                xSeries,
                idx,
                maxDistanceX
              )
              if (valuePointIdx === null) continue

              const value = valueSeries?.[valuePointIdx]
              if (value === null || value === undefined) continue

              hasAnyValue = true
              const formatted = formatValueSmart(value)

              html += `
                <div class="flex items-center gap-2 mt-0.5">
                  <div class="w-2 h-2 rounded-full shrink-0" style="background-color: ${MULTI_SERIES_COLORS[i % MULTI_SERIES_COLORS.length]}"></div>
                  <span class="text-muted-foreground">Server ${serverIds[i]}:</span>
                  <span class="font-medium">${formatted} ${metricInfo.unit}</span>
                </div>
              `
            }

            if (!hasAnyValue) {
              tooltipRef.current.style.display = "none"
              return
            }

            tooltipRef.current.innerHTML = html
            tooltipRef.current.style.display = "block"

            const containerRect = containerRef.current.getBoundingClientRect()
            const tooltipRect = tooltipRef.current.getBoundingClientRect()

            let tooltipX = containerRect.left + left + 15
            if (tooltipX + tooltipRect.width + 20 > window.innerWidth) {
              tooltipX = containerRect.left + left - tooltipRect.width - 15 + u.bbox.left
            }
            tooltipX = Math.max(4, tooltipX)

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
        const { width: newWidth } = entry.contentRect
        if (chart && newWidth > 0) {
          chart.setSize({ width: newWidth, height })
        }
      }
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
      chart.destroy()
      chartRef.current = null
    }
  }, [uplotData, hasData, seriesConfig, metricName, timeRange, xOffset, metricInfo.unit, serverIds, formatYAxisTick, ignoreOutliers, outlierBounds, minY, maxY, darkMode])

  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) tooltipRef.current.style.display = "none"
  }, [])

  return (
    <div
      className={cn(
        "group/chart rounded-lg border border-border p-3 transition-opacity bg-background",
        isLoading && "opacity-60"
      )}
    >
      <div className="flex items-center justify-between mb-2 gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <h4 className="text-xs font-medium truncate" title={metricName}>{metricInfo.label}</h4>
          {ignoreOutliers && (
            <FilterBadge label="Ignore Outliers" onRemove={() => setIgnoreOutliers(false)} />
          )}
          {minY !== null && (
            <FilterBadge label={`Min Y: ${minY}`} onRemove={() => setMinY(null)} />
          )}
          {maxY !== null && (
            <FilterBadge label={`Max Y: ${maxY}`} onRemove={() => setMaxY(null)} />
          )}
        </div>
        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                className="h-5 px-1.5 text-[10px] rounded border border-border hover:bg-muted flex items-center gap-1 transition-all opacity-0 group-hover/chart:opacity-100"
              >
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
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      {hasData ? (
        <div
          className="h-[200px] relative bg-background rounded"
          ref={containerRef}
          onMouseLeave={handleMouseLeave}
        >
          {isRefetching && (
            <Loader2 className="absolute bottom-0.5 left-0.5 h-3 w-3 animate-spin text-muted-foreground" />
          )}
          <div
            ref={tooltipRef}
            className="fixed z-[9999] max-w-[360px] bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl border border-border pointer-events-none"
            style={{ display: "none" }}
          />
        </div>
      ) : (
        <div className="h-[200px] flex items-center justify-center text-muted-foreground text-xs rounded">
          {isLoading ? "Loading..." : `No data for ${metricInfo.label}`}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// vLLM Server Labels (above charts, matching GpuLabels pattern)
// ============================================================================

function VllmServerLabels({
  data,
  activeServerIds,
  onServerHover,
  onServerToggle,
}: {
  data: VllmMetric[]
  activeServerIds?: Set<number> | null
  onServerHover?: (serverId: number | null) => void
  onServerToggle?: (serverId: number) => void
}) {
  const labels = useMemo(() => {
    const serverIdSet = new Set<number>()
    for (const m of data) serverIdSet.add(m.server)
    const sorted = Array.from(serverIdSet).sort((a, b) => a - b)
    return sorted.map((serverId, idx) => ({
      serverId,
      label: `Server ${serverId}`,
      color: MULTI_SERIES_COLORS[idx % MULTI_SERIES_COLORS.length],
    }))
  }, [data])

  if (labels.length <= 1) return null
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 mb-3">
      {labels.map((s) => {
        const isActive = !!activeServerIds?.has(s.serverId)
        const isDimmed = !!activeServerIds && activeServerIds.size > 0 && !isActive
        return (
          <div
            key={s.label}
            className={cn(
              "flex items-center gap-1.5 select-none transition-opacity duration-150",
              onServerToggle && "cursor-pointer",
              isActive && "ring-1 ring-border rounded-full px-1.5 py-0.5 -mx-1.5 -my-0.5"
            )}
            style={{ opacity: isDimmed ? 0.3 : 1 }}
            onPointerEnter={
              onServerHover ? () => onServerHover(s.serverId) : undefined
            }
            onPointerLeave={onServerHover ? () => onServerHover(null) : undefined}
            onClick={onServerToggle ? () => onServerToggle(s.serverId) : undefined}
          >
            <div
              className="w-2.5 h-2.5 rounded-sm"
              style={{
                backgroundColor: isDimmed ? withAlpha(s.color, 0.25) : s.color,
              }}
            />
            <span className="text-xs text-muted-foreground">{s.label}</span>
          </div>
        )
      })}
    </div>
  )
}

// ============================================================================
// vLLM Metrics Grid
// ============================================================================

function VllmMetricsGrid({
  data,
  availableMetricNames,
  isLoading,
  isRefetching,
  activeServerIds,
  overrideTimeRange,
  xOffset = 0,
}: {
  data: VllmMetric[]
  availableMetricNames?: string[]
  isLoading?: boolean
  isRefetching?: boolean
  activeServerIds?: Set<number> | null
  overrideTimeRange?: { start: number; end: number } | null
  xOffset?: number
}) {
  const metricNames = useMemo(() => {
    const names = new Set<string>()
    for (const m of data) names.add(m.metric_name)
    for (const name of availableMetricNames ?? []) names.add(name)
    // Sort to show in preferred order
    const ordered = VLLM_METRIC_NAMES.filter((n) => names.has(n))
    // Add any extra metrics not in our hardcoded list
    for (const n of names) {
      if (!ordered.includes(n)) ordered.push(n)
    }
    return ordered
  }, [data, availableMetricNames])

  const metricsToRender = useMemo(() => {
    if (metricNames.length > 0) return metricNames
    if (isLoading) return VLLM_METRIC_NAMES
    return metricNames
  }, [metricNames, isLoading])

  if (metricsToRender.length === 0 && !isLoading) {
    return (
      <div className="h-24 flex items-center justify-center text-muted-foreground text-xs rounded-lg border border-border bg-background">
        No vLLM metrics data available
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
      {metricsToRender.map((metricName) => {
        const metricData = data.filter((m) => m.metric_name === metricName)
        return (
          <VllmMetricChart
            key={metricName}
            metricName={metricName}
            data={metricData}
            isLoading={isLoading}
            isRefetching={isRefetching}
            activeServerIds={activeServerIds}
            overrideTimeRange={overrideTimeRange}
            xOffset={xOffset}
          />
        )
      })}
    </div>
  )
}

// ============================================================================
// Node Group Collapsible
// ============================================================================

function NodeGroupCollapsible({
  nodeId,
  gpuCount,
  children,
}: {
  nodeId: number | null
  gpuCount: number
  children: React.ReactNode
}) {
  const [isOpen, setIsOpen] = useState(true)
  const nodeLabel = nodeId === null ? "Unknown Node" : `Node ${nodeId}`

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
          <div className="flex items-center gap-1.5">
            <ChevronDown
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform",
                !isOpen && "-rotate-90"
              )}
            />
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {nodeLabel}
            </span>
            <span className="text-[11px] text-muted-foreground/80">
              {gpuCount} GPU{gpuCount !== 1 ? "s" : ""}
            </span>
          </div>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-3">{children}</div>
      </CollapsibleContent>
    </Collapsible>
  )
}

// ============================================================================
// Role Section (Trainer or Inference)
// ============================================================================

function RoleSection({
  title,
  gpuData,
  gpuMetricNames,
  timeRange,
  nodeGpuMap,
  isLoading,
  isRefetching,
  open,
  onOpenChange,
  nodeMode,
  onNodeModeChange,
  xOffset = 0,
}: {
  title: string
  gpuData: GpuMetric[]
  gpuMetricNames: string[]
  timeRange: { start: number; end: number }
  nodeGpuMap: Map<number, number[]> // node_id → gpu_indices
  isLoading?: boolean
  isRefetching?: boolean
  open?: boolean
  onOpenChange?: (open: boolean) => void
  nodeMode?: "combined" | "separated"
  onNodeModeChange?: (mode: "combined" | "separated") => void
  xOffset?: number
}) {
  const [localIsOpen, setLocalIsOpen] = useState(true)
  const [localNodeMode, setLocalNodeMode] = useState<"combined" | "separated">(
    "combined"
  )
  const isOpen = open ?? localIsOpen
  const setIsOpen = onOpenChange ?? setLocalIsOpen
  const effectiveNodeMode = nodeMode ?? localNodeMode
  const setNodeMode = onNodeModeChange ?? setLocalNodeMode
  const [hoveredSeriesKey, setHoveredSeriesKey] = useState<string | null>(null)
  const [selectedSeriesKeys, setSelectedSeriesKeys] = useState<string[]>([])

  const hasMultipleNodes = nodeGpuMap.size > 1

  const fallbackCombinedLabels = useMemo(() => {
    const entries: Array<{ nodeId: number; gpuIndex: number }> = []
    const sortedNodeIds = Array.from(nodeGpuMap.keys()).sort((a, b) => a - b)
    for (const nodeId of sortedNodeIds) {
      const gpuIndices = [...(nodeGpuMap.get(nodeId) ?? [])].sort((a, b) => a - b)
      for (const gpuIndex of gpuIndices) {
        entries.push({ nodeId, gpuIndex })
      }
    }
    return entries.map((entry, idx) => ({
      seriesKey: buildGpuSeriesKey(entry.nodeId, entry.gpuIndex, hasMultipleNodes),
      node_id: entry.nodeId,
      gpu_index: entry.gpuIndex,
      label: hasMultipleNodes
        ? `Node ${entry.nodeId} GPU ${entry.gpuIndex}`
        : `GPU ${entry.gpuIndex}`,
      color: hasMultipleNodes
        ? MULTI_SERIES_COLORS[idx % MULTI_SERIES_COLORS.length]
        : MULTI_SERIES_COLORS[entry.gpuIndex % MULTI_SERIES_COLORS.length],
    }))
  }, [nodeGpuMap, hasMultipleNodes])

  // Labels for combined mode
  const combinedLabels = useMemo(
    () => {
      const labels = computeGpuLabels(gpuData, hasMultipleNodes)
      return labels.length > 0 ? labels : fallbackCombinedLabels
    },
    [gpuData, hasMultipleNodes, fallbackCombinedLabels]
  )

  // All GPU indices for combined mode
  const allGpuIndices = useMemo(() => {
    const indices = new Set<number>()
    for (const m of gpuData) indices.add(m.gpu_index)
    return Array.from(indices).sort((a, b) => a - b)
  }, [gpuData])

  // Grouped data by node
  const groupedByNode = useMemo(() => {
    const groups: Array<{
      nodeId: number
      gpuIndices: number[]
      data: GpuMetric[]
      labels: GpuSeriesLabel[]
    }> = []
    const sortedNodeIds = Array.from(nodeGpuMap.keys()).sort((a, b) => a - b)
    for (const nodeId of sortedNodeIds) {
      const gpuIndices = nodeGpuMap.get(nodeId)!
      const nodeData = gpuData.filter(
        (m) =>
          (typeof m.node_id === "number" ? m.node_id : null) === nodeId &&
          gpuIndices.includes(m.gpu_index)
      )
      const labelsFromData = computeGpuLabels(nodeData, false)
      const labels =
        labelsFromData.length > 0
          ? labelsFromData
          : gpuIndices.map((gpuIndex) => ({
              seriesKey: buildGpuSeriesKey(nodeId, gpuIndex, false),
              node_id: nodeId,
              gpu_index: gpuIndex,
              label: `GPU ${gpuIndex}`,
              color: MULTI_SERIES_COLORS[gpuIndex % MULTI_SERIES_COLORS.length],
            }))
      groups.push({ nodeId, gpuIndices, data: nodeData, labels })
    }
    return groups
  }, [gpuData, nodeGpuMap])

  const visibleSeriesKeySet = useMemo(() => {
    const keys = new Set<string>()
    if (effectiveNodeMode === "combined" || !hasMultipleNodes) {
      combinedLabels.forEach((label) => keys.add(label.seriesKey))
      return keys
    }
    groupedByNode.forEach((group) => {
      group.labels.forEach((label) => keys.add(label.seriesKey))
    })
    return keys
  }, [effectiveNodeMode, hasMultipleNodes, combinedLabels, groupedByNode])

  const toggleSeriesSelection = useCallback(
    (seriesKey: string) => {
      setSelectedSeriesKeys((prev) => {
        const hidden = prev.filter((key) => !visibleSeriesKeySet.has(key))
        const visibleSelected = prev.filter((key) => visibleSeriesKeySet.has(key))
        const alreadySelected = visibleSelected.includes(seriesKey)
        const nextVisible = alreadySelected
          ? visibleSelected.filter((key) => key !== seriesKey)
          : [...visibleSelected, seriesKey]
        // Keep "all selected" equivalent to "none selected" for this visible label set.
        const collapsedVisible =
          visibleSeriesKeySet.size > 0 &&
          nextVisible.length === visibleSeriesKeySet.size
            ? []
            : nextVisible
        return [...hidden, ...collapsedVisible]
      })
    },
    [visibleSeriesKeySet]
  )

  const effectiveSelectedSeriesKeys = useMemo(() => {
    if (visibleSeriesKeySet.size === 0) return selectedSeriesKeys
    return selectedSeriesKeys.filter((key) => visibleSeriesKeySet.has(key))
  }, [selectedSeriesKeys, visibleSeriesKeySet])

  const effectiveHoveredSeriesKey = useMemo(() => {
    if (!hoveredSeriesKey) return null
    if (visibleSeriesKeySet.size === 0) return hoveredSeriesKey
    return visibleSeriesKeySet.has(hoveredSeriesKey) ? hoveredSeriesKey : null
  }, [hoveredSeriesKey, visibleSeriesKeySet])

  const activeSeriesKeys = useMemo(() => {
    if (effectiveSelectedSeriesKeys.length > 0) {
      return new Set(effectiveSelectedSeriesKeys)
    }
    if (effectiveHoveredSeriesKey) {
      return new Set([effectiveHoveredSeriesKey])
    }
    return null
  }, [effectiveSelectedSeriesKeys, effectiveHoveredSeriesKey])

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="flex items-center gap-2 mb-3">
        <CollapsibleTrigger asChild>
          <div className="flex items-center gap-1.5 py-1.5 px-2 -ml-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <ChevronDown
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform",
                !isOpen && "-rotate-90"
              )}
            />
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              {title}
            </h3>
          </div>
        </CollapsibleTrigger>
        {isOpen && hasMultipleNodes && (
          <ToggleGroup
            type="single"
            variant="selecting"
            size="sm"
            value={effectiveNodeMode}
            onValueChange={(value) => {
              if (value) setNodeMode(value as "combined" | "separated")
            }}
          >
            <ToggleGroupItem
              value="combined"
              className="text-xs px-2 py-1 h-7"
            >
              Combined
            </ToggleGroupItem>
            <ToggleGroupItem
              value="separated"
              className="text-xs px-2 py-1 h-7"
            >
              Separate Nodes
            </ToggleGroupItem>
          </ToggleGroup>
        )}
      </div>
      <CollapsibleContent>
        <div className="space-y-6">
          {effectiveNodeMode === "combined" || !hasMultipleNodes ? (
            <>
              <GpuLabels
                series={combinedLabels}
                activeSeriesKeys={activeSeriesKeys}
                onSeriesHover={setHoveredSeriesKey}
                onSeriesToggle={toggleSeriesSelection}
              />
              <GpuMetricsGrid
                data={gpuData}
                metricNames={gpuMetricNames}
                intervalStart={timeRange.start}
                intervalEnd={timeRange.end}
                gpuIndices={allGpuIndices}
                seriesMode={hasMultipleNodes ? "node_gpu" : "gpu"}
                isLoading={isLoading}
                isRefetching={isRefetching}
                activeSeriesKeys={activeSeriesKeys}
                xOffset={xOffset}
              />
            </>
          ) : (
            groupedByNode.map((group) => (
              <NodeGroupCollapsible
                key={group.nodeId}
                nodeId={group.nodeId}
                gpuCount={group.gpuIndices.length}
              >
                <GpuLabels
                  series={group.labels}
                  activeSeriesKeys={activeSeriesKeys}
                  onSeriesHover={setHoveredSeriesKey}
                  onSeriesToggle={toggleSeriesSelection}
                />
                <GpuMetricsGrid
                  data={group.data}
                  metricNames={gpuMetricNames}
                  intervalStart={timeRange.start}
                  intervalEnd={timeRange.end}
                  gpuIndices={group.gpuIndices}
                  seriesMode="gpu"
                  isLoading={isLoading}
                  isRefetching={isRefetching}
                  activeSeriesKeys={activeSeriesKeys}
                  xOffset={xOffset}
                />
              </NodeGroupCollapsible>
            ))
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

// ============================================================================
// Main Component
// ============================================================================

export function SystemMetricsCharts({
  gpuMetrics,
  cpuMetrics,
  vllmMetrics,
  availableGpuMetrics,
  availableCpuMetrics,
  availableVllmMetrics,
  trainerGpus,
  inferenceGpus,
  isLoading,
  isRefetching,
  systemMetricsOpen: systemMetricsOpenProp,
  onSystemMetricsOpenChange,
  cpuMetricsOpen: cpuMetricsOpenProp,
  onCpuMetricsOpenChange,
  vllmMetricsOpen: vllmMetricsOpenProp,
  onVllmMetricsOpenChange,
  roleMode: roleModeProp,
  onRoleModeChange,
  trainerSectionOpen,
  onTrainerSectionOpenChange,
  inferenceSectionOpen,
  onInferenceSectionOpenChange,
  trainerNodeMode,
  onTrainerNodeModeChange,
  inferenceNodeMode,
  onInferenceNodeModeChange,
  overrideTimeRange,
  xOffset = 0,
}: SystemMetricsChartsProps) {
  const [localSystemMetricsOpen, setLocalSystemMetricsOpen] = useState(true)
  const [localCpuMetricsOpen, setLocalCpuMetricsOpen] = useState(true)
  const [localVllmMetricsOpen, setLocalVllmMetricsOpen] = useState(true)
  const [localRoleMode, setLocalRoleMode] = useState<"combined" | "separated">(
    "separated"
  )
  const systemMetricsOpen = systemMetricsOpenProp ?? localSystemMetricsOpen
  const setSystemMetricsOpen =
    onSystemMetricsOpenChange ?? setLocalSystemMetricsOpen
  const cpuMetricsOpen = cpuMetricsOpenProp ?? localCpuMetricsOpen
  const setCpuMetricsOpen = onCpuMetricsOpenChange ?? setLocalCpuMetricsOpen
  const vllmMetricsOpen = vllmMetricsOpenProp ?? localVllmMetricsOpen
  const setVllmMetricsOpen = onVllmMetricsOpenChange ?? setLocalVllmMetricsOpen
  const roleMode = roleModeProp ?? localRoleMode
  const setRoleMode = onRoleModeChange ?? setLocalRoleMode
  const [hoveredCombinedGpuSeriesKey, setHoveredCombinedGpuSeriesKey] = useState<string | null>(null)
  const [selectedCombinedGpuSeriesKeys, setSelectedCombinedGpuSeriesKeys] = useState<string[]>([])
  const [hoveredVllmServerId, setHoveredVllmServerId] = useState<number | null>(null)
  const [selectedVllmServerIds, setSelectedVllmServerIds] = useState<number[]>([])

  // Compute global time range from all GPU metrics (or use override)
  const gpuTimeRange = useMemo(() => {
    if (overrideTimeRange) return overrideTimeRange
    return computeTimeRange(gpuMetrics)
  }, [gpuMetrics, overrideTimeRange])

  // Determine which metric names are available
  const gpuMetricNames = useMemo(() => {
    const available = new Set(availableGpuMetrics)
    // Return metrics in preferred order, filtered by what's available
    const systemAvailable = GPU_SYSTEM_METRICS.filter((n) => available.has(n))
    const torchAvailable = GPU_TORCH_METRICS.filter((n) => available.has(n))
    // Add any extra metrics not in our hardcoded list
    const allKnown = new Set([...GPU_SYSTEM_METRICS, ...GPU_TORCH_METRICS])
    const extra = availableGpuMetrics.filter((n) => !allKnown.has(n)).sort()
    return [...systemAvailable, ...torchAvailable, ...extra]
  }, [availableGpuMetrics])

  // Build GPU lookup sets for role filtering
  const trainerGpuSet = useMemo(
    () => new Set(trainerGpus.map((g) => `${g.node_id}:${g.gpu_index}`)),
    [trainerGpus]
  )
  const inferenceGpuSet = useMemo(
    () => new Set(inferenceGpus.map((g) => `${g.node_id}:${g.gpu_index}`)),
    [inferenceGpus]
  )

  // Filter GPU metrics by role
  const trainerGpuMetrics = useMemo(() => {
    if (trainerGpuSet.size === 0) return []
    return gpuMetrics.filter((m) => {
      const nodeId =
        typeof m.node_id === "number" && Number.isFinite(m.node_id)
          ? m.node_id
          : null
      if (nodeId === null) return false
      return trainerGpuSet.has(`${nodeId}:${m.gpu_index}`)
    })
  }, [gpuMetrics, trainerGpuSet])

  const inferenceGpuMetrics = useMemo(() => {
    if (inferenceGpuSet.size === 0) return []
    return gpuMetrics.filter((m) => {
      const nodeId =
        typeof m.node_id === "number" && Number.isFinite(m.node_id)
          ? m.node_id
          : null
      if (nodeId === null) return false
      return inferenceGpuSet.has(`${nodeId}:${m.gpu_index}`)
    })
  }, [gpuMetrics, inferenceGpuSet])

  // Build node → gpu_index maps for each role
  const trainerNodeGpuMap = useMemo(() => {
    const map = new Map<number, number[]>()
    for (const g of trainerGpus) {
      const existing = map.get(g.node_id) ?? []
      if (!existing.includes(g.gpu_index)) existing.push(g.gpu_index)
      map.set(g.node_id, existing)
    }
    // Sort gpu indices within each node
    for (const [key, indices] of map) {
      map.set(
        key,
        indices.sort((a, b) => a - b)
      )
    }
    return map
  }, [trainerGpus])

  const inferenceNodeGpuMap = useMemo(() => {
    const map = new Map<number, number[]>()
    for (const g of inferenceGpus) {
      const existing = map.get(g.node_id) ?? []
      if (!existing.includes(g.gpu_index)) existing.push(g.gpu_index)
      map.set(g.node_id, existing)
    }
    for (const [key, indices] of map) {
      map.set(
        key,
        indices.sort((a, b) => a - b)
      )
    }
    return map
  }, [inferenceGpus])

  // All GPU indices for combined mode
  const allGpuIndices = useMemo(() => {
    const indices = new Set<number>()
    for (const m of gpuMetrics) indices.add(m.gpu_index)
    return Array.from(indices).sort((a, b) => a - b)
  }, [gpuMetrics])

  // Check if multiple nodes exist in the data
  const hasMultipleNodes = useMemo(() => {
    const nodeIds = new Set<number | null>()
    for (const m of gpuMetrics) {
      const nodeId =
        typeof m.node_id === "number" && Number.isFinite(m.node_id)
          ? m.node_id
          : null
      nodeIds.add(nodeId)
    }
    return nodeIds.size > 1
  }, [gpuMetrics])

  // Labels for combined mode
  const combinedLabels = useMemo(
    () => computeGpuLabels(gpuMetrics, hasMultipleNodes),
    [gpuMetrics, hasMultipleNodes]
  )

  // Role-grouped labels for combined mode
  const fallbackCombinedLabels = useMemo(() => {
    if (trainerGpus.length === 0 && inferenceGpus.length === 0) return []
    const allRoleGpus = [...trainerGpus, ...inferenceGpus]
    const multiNode = new Set(allRoleGpus.map((g) => g.node_id)).size > 1
    return computeGpuLabelsFromIdentifiers(allRoleGpus, multiNode)
  }, [trainerGpus, inferenceGpus])

  const trainerCombinedLabels = useMemo(
    () => {
      const sourceLabels =
        combinedLabels.length > 0 ? combinedLabels : fallbackCombinedLabels
      const trainerGpuSet = new Set(trainerGpus.map((g) => `${g.node_id}:${g.gpu_index}`))
      return sourceLabels.filter((l) => {
        const key = `${l.node_id ?? "?"}:${l.gpu_index}`
        // Also try matching by node_id number
        return trainerGpuSet.has(key) || trainerGpuSet.has(`${l.node_id}:${l.gpu_index}`)
      })
    },
    [combinedLabels, fallbackCombinedLabels, trainerGpus]
  )

  const inferenceCombinedLabels = useMemo(
    () => {
      const sourceLabels =
        combinedLabels.length > 0 ? combinedLabels : fallbackCombinedLabels
      const inferenceGpuSet = new Set(inferenceGpus.map((g) => `${g.node_id}:${g.gpu_index}`))
      return sourceLabels.filter((l) => {
        const key = `${l.node_id ?? "?"}:${l.gpu_index}`
        return inferenceGpuSet.has(key) || inferenceGpuSet.has(`${l.node_id}:${l.gpu_index}`)
      })
    },
    [combinedLabels, fallbackCombinedLabels, inferenceGpus]
  )

  // Build seriesRoles map for combined mode tooltip grouping
  const combinedSeriesRoles = useMemo<SeriesRoleMap>(() => {
    const roles: SeriesRoleMap = {}
    const trainerGpuSet = new Set(trainerGpus.map((g) => `${g.node_id}:${g.gpu_index}`))
    const inferenceGpuSet = new Set(inferenceGpus.map((g) => `${g.node_id}:${g.gpu_index}`))

    // Build keys matching how GpuMetricChart builds them
    if (hasMultipleNodes) {
      // seriesMode === "node_gpu": key = "${nodeId ?? "?"}:${gpu_index}"
      for (const m of gpuMetrics) {
        const nodeId = typeof m.node_id === "number" && Number.isFinite(m.node_id)
          ? m.node_id
          : null
        const seriesKey = `${nodeId ?? "?"}:${m.gpu_index}`
        const lookupKey = `${nodeId}:${m.gpu_index}`
        if (trainerGpuSet.has(lookupKey) || trainerGpuSet.has(seriesKey)) {
          roles[seriesKey] = "Trainer"
        } else if (inferenceGpuSet.has(lookupKey) || inferenceGpuSet.has(seriesKey)) {
          roles[seriesKey] = "Inference"
        }
      }
    } else {
      // seriesMode === "gpu": key = "gpu:${gpu_index}"
      // Need to map gpu_index → role; when single node, node_id from setup data is used
      for (const g of trainerGpus) {
        roles[`gpu:${g.gpu_index}`] = "Trainer"
      }
      for (const g of inferenceGpus) {
        roles[`gpu:${g.gpu_index}`] = "Inference"
      }
    }
    return roles
  }, [gpuMetrics, trainerGpus, inferenceGpus, hasMultipleNodes])

  const hasRoleInfo = trainerGpus.length > 0 || inferenceGpus.length > 0
  const showTrainerSection =
    trainerGpuMetrics.length > 0 || (!!isLoading && trainerGpus.length > 0)
  const showInferenceSection =
    inferenceGpuMetrics.length > 0 || (!!isLoading && inferenceGpus.length > 0)

  const visibleCombinedGpuLabels = useMemo(() => {
    if (!hasRoleInfo) return combinedLabels
    const dedup = new Map<string, GpuSeriesLabel>()
    for (const label of trainerCombinedLabels) dedup.set(label.seriesKey, label)
    for (const label of inferenceCombinedLabels) dedup.set(label.seriesKey, label)
    return Array.from(dedup.values())
  }, [hasRoleInfo, combinedLabels, trainerCombinedLabels, inferenceCombinedLabels])

  const visibleCombinedGpuSeriesKeySet = useMemo(
    () => new Set(visibleCombinedGpuLabels.map((label) => label.seriesKey)),
    [visibleCombinedGpuLabels]
  )

  const effectiveSelectedCombinedGpuSeriesKeys = useMemo(() => {
    if (visibleCombinedGpuSeriesKeySet.size === 0) return selectedCombinedGpuSeriesKeys
    return selectedCombinedGpuSeriesKeys.filter((key) =>
      visibleCombinedGpuSeriesKeySet.has(key)
    )
  }, [selectedCombinedGpuSeriesKeys, visibleCombinedGpuSeriesKeySet])

  const toggleCombinedGpuSeriesSelection = useCallback(
    (seriesKey: string) => {
      setSelectedCombinedGpuSeriesKeys((prev) => {
        const hidden = prev.filter((key) => !visibleCombinedGpuSeriesKeySet.has(key))
        const visibleSelected = prev.filter((key) =>
          visibleCombinedGpuSeriesKeySet.has(key)
        )
        const alreadySelected = visibleSelected.includes(seriesKey)
        const nextVisible = alreadySelected
          ? visibleSelected.filter((key) => key !== seriesKey)
          : [...visibleSelected, seriesKey]
        // Keep "all selected" equivalent to "none selected" for visible labels.
        const collapsedVisible =
          visibleCombinedGpuSeriesKeySet.size > 0 &&
          nextVisible.length === visibleCombinedGpuSeriesKeySet.size
            ? []
            : nextVisible
        return [...hidden, ...collapsedVisible]
      })
    },
    [visibleCombinedGpuSeriesKeySet]
  )

  const effectiveHoveredCombinedGpuSeriesKey = useMemo(() => {
    if (!hoveredCombinedGpuSeriesKey) return null
    if (visibleCombinedGpuSeriesKeySet.size === 0) return hoveredCombinedGpuSeriesKey
    return visibleCombinedGpuSeriesKeySet.has(hoveredCombinedGpuSeriesKey)
      ? hoveredCombinedGpuSeriesKey
      : null
  }, [hoveredCombinedGpuSeriesKey, visibleCombinedGpuSeriesKeySet])

  const combinedActiveSeriesKeys = useMemo(() => {
    if (effectiveSelectedCombinedGpuSeriesKeys.length > 0) {
      if (
        visibleCombinedGpuSeriesKeySet.size > 0 &&
        effectiveSelectedCombinedGpuSeriesKeys.length ===
          visibleCombinedGpuSeriesKeySet.size
      ) {
        return null
      }
      return new Set(effectiveSelectedCombinedGpuSeriesKeys)
    }
    if (effectiveHoveredCombinedGpuSeriesKey) {
      return new Set([effectiveHoveredCombinedGpuSeriesKey])
    }
    return null
  }, [
    effectiveSelectedCombinedGpuSeriesKeys,
    effectiveHoveredCombinedGpuSeriesKey,
    visibleCombinedGpuSeriesKeySet,
  ])

  const vllmServerIdSet = useMemo(() => {
    const ids = new Set<number>()
    for (const metric of vllmMetrics ?? []) {
      ids.add(metric.server)
    }
    return ids
  }, [vllmMetrics])

  const effectiveSelectedVllmServerIds = useMemo(() => {
    if (vllmServerIdSet.size === 0) return selectedVllmServerIds
    return selectedVllmServerIds.filter((id) => vllmServerIdSet.has(id))
  }, [selectedVllmServerIds, vllmServerIdSet])

  const toggleVllmServerSelection = useCallback(
    (serverId: number) => {
      setSelectedVllmServerIds((prev) => {
        const hidden = prev.filter((id) => !vllmServerIdSet.has(id))
        const visibleSelected = prev.filter((id) => vllmServerIdSet.has(id))
        const alreadySelected = visibleSelected.includes(serverId)
        const nextVisible = alreadySelected
          ? visibleSelected.filter((id) => id !== serverId)
          : [...visibleSelected, serverId]
        // Keep "all selected" equivalent to "none selected" for visible labels.
        const collapsedVisible =
          vllmServerIdSet.size > 0 && nextVisible.length === vllmServerIdSet.size
            ? []
            : nextVisible
        return [...hidden, ...collapsedVisible]
      })
    },
    [vllmServerIdSet]
  )

  const effectiveHoveredVllmServerId = useMemo(() => {
    if (hoveredVllmServerId === null) return null
    if (vllmServerIdSet.size === 0) return hoveredVllmServerId
    return vllmServerIdSet.has(hoveredVllmServerId)
      ? hoveredVllmServerId
      : null
  }, [hoveredVllmServerId, vllmServerIdSet])

  const activeVllmServerIds = useMemo(() => {
    if (effectiveSelectedVllmServerIds.length > 0) {
      if (
        vllmServerIdSet.size > 0 &&
        effectiveSelectedVllmServerIds.length === vllmServerIdSet.size
      ) {
        return null
      }
      return new Set(effectiveSelectedVllmServerIds)
    }
    if (effectiveHoveredVllmServerId !== null) {
      return new Set([effectiveHoveredVllmServerId])
    }
    return null
  }, [
    effectiveSelectedVllmServerIds,
    effectiveHoveredVllmServerId,
    vllmServerIdSet,
  ])

  return (
    <div className="space-y-4">
      {/* GPU System Metrics Section */}
      <Collapsible
        open={systemMetricsOpen}
        onOpenChange={setSystemMetricsOpen}
      >
        <div className="flex items-center gap-2 mb-3">
          <CollapsibleTrigger asChild>
            <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
              <div className="flex items-center gap-1.5">
                <ChevronDown
                  className={cn(
                    "h-4 w-4 text-muted-foreground transition-transform",
                    !systemMetricsOpen && "-rotate-90"
                  )}
                />
                <h3 className="text-sm font-semibold">GPU Metrics</h3>
              </div>
            </div>
          </CollapsibleTrigger>
          {systemMetricsOpen && hasRoleInfo && (
            <ToggleGroup
              type="single"
              variant="selecting"
              size="sm"
              value={roleMode}
              onValueChange={(value) => {
                if (value) setRoleMode(value as "combined" | "separated")
              }}
            >
              <ToggleGroupItem
                value="separated"
                className="text-xs px-2 py-1 h-7"
              >
                Separate Trainer/Inference
              </ToggleGroupItem>
              <ToggleGroupItem
                value="combined"
                className="text-xs px-2 py-1 h-7"
              >
                Combined
              </ToggleGroupItem>
            </ToggleGroup>
          )}
        </div>
        <CollapsibleContent>
          <div className="space-y-6 mt-3 mb-4">
            {roleMode === "combined" || !hasRoleInfo ? (
              <>
                {hasRoleInfo ? (
                  <RoleGroupedGpuLabels
                    trainerLabels={trainerCombinedLabels}
                    inferenceLabels={inferenceCombinedLabels}
                    activeSeriesKeys={combinedActiveSeriesKeys}
                    onSeriesHover={setHoveredCombinedGpuSeriesKey}
                    onSeriesToggle={toggleCombinedGpuSeriesSelection}
                  />
                ) : (
                  <GpuLabels
                    series={combinedLabels}
                    activeSeriesKeys={combinedActiveSeriesKeys}
                    onSeriesHover={setHoveredCombinedGpuSeriesKey}
                    onSeriesToggle={toggleCombinedGpuSeriesSelection}
                  />
                )}
                <GpuMetricsGrid
                  data={gpuMetrics}
                  metricNames={gpuMetricNames}
                  intervalStart={gpuTimeRange.start}
                  intervalEnd={gpuTimeRange.end}
                  gpuIndices={allGpuIndices}
                  seriesMode={hasMultipleNodes ? "node_gpu" : "gpu"}
                  isLoading={isLoading}
                  isRefetching={isRefetching}
                  seriesRoles={hasRoleInfo ? combinedSeriesRoles : undefined}
                  activeSeriesKeys={combinedActiveSeriesKeys}
                  xOffset={xOffset}
                />
              </>
            ) : (
              <>
                {showTrainerSection && (
                  <RoleSection
                    title="Trainer"
                    gpuData={trainerGpuMetrics}
                    gpuMetricNames={gpuMetricNames}
                    timeRange={gpuTimeRange}
                    nodeGpuMap={trainerNodeGpuMap}
                    isLoading={isLoading}
                    isRefetching={isRefetching}
                    open={trainerSectionOpen}
                    onOpenChange={onTrainerSectionOpenChange}
                    nodeMode={trainerNodeMode}
                    onNodeModeChange={onTrainerNodeModeChange}
                    xOffset={xOffset}
                  />
                )}

                {showInferenceSection && (
                  <>
                    {showTrainerSection && (
                      <div className="border-t border-border my-3" />
                    )}
                    <RoleSection
                      title="Inference"
                      gpuData={inferenceGpuMetrics}
                      gpuMetricNames={gpuMetricNames}
                      timeRange={gpuTimeRange}
                      nodeGpuMap={inferenceNodeGpuMap}
                      isLoading={isLoading}
                      isRefetching={isRefetching}
                      open={inferenceSectionOpen}
                      onOpenChange={onInferenceSectionOpenChange}
                      nodeMode={inferenceNodeMode}
                      onNodeModeChange={onInferenceNodeModeChange}
                      xOffset={xOffset}
                    />
                  </>
                )}

                {!isLoading &&
                  trainerGpuMetrics.length === 0 &&
                  inferenceGpuMetrics.length === 0 && (
                    <div className="h-24 flex items-center justify-center text-muted-foreground text-xs rounded-lg border border-border bg-background">
                      No GPU data matches the trainer/inference role configuration
                    </div>
                  )}
              </>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>

      <div className="border-t border-border my-3" />

      {/* CPU Metrics Section */}
      <Collapsible open={cpuMetricsOpen} onOpenChange={setCpuMetricsOpen}>
        <CollapsibleTrigger asChild>
          <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <div className="flex items-center gap-1.5">
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  !cpuMetricsOpen && "-rotate-90"
                )}
              />
              <h3 className="text-sm font-semibold">CPU Metrics</h3>
            </div>
          </div>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-3 mb-4">
            <CpuMetricsGrid
              data={cpuMetrics}
              availableMetricNames={availableCpuMetrics}
              isLoading={isLoading}
              isRefetching={isRefetching}
              overrideTimeRange={overrideTimeRange}
              xOffset={xOffset}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* vLLM Metrics Section */}
      {(vllmMetrics && vllmMetrics.length > 0) || (availableVllmMetrics && availableVllmMetrics.length > 0) || isLoading ? (
        <>
          <div className="border-t border-border my-3" />
          <Collapsible open={vllmMetricsOpen} onOpenChange={setVllmMetricsOpen}>
            <CollapsibleTrigger asChild>
              <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
                <div className="flex items-center gap-1.5">
                  <ChevronDown
                    className={cn(
                      "h-4 w-4 text-muted-foreground transition-transform",
                      !vllmMetricsOpen && "-rotate-90"
                    )}
                  />
                  <h3 className="text-sm font-semibold">vLLM Metrics</h3>
                </div>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="space-y-6 mt-3 mb-4">
                <VllmServerLabels
                  data={vllmMetrics ?? []}
                  activeServerIds={activeVllmServerIds}
                  onServerHover={setHoveredVllmServerId}
                  onServerToggle={toggleVllmServerSelection}
                />
                <VllmMetricsGrid
                  data={vllmMetrics ?? []}
                  availableMetricNames={availableVllmMetrics}
                  isLoading={isLoading}
                  isRefetching={isRefetching}
                  activeServerIds={activeVllmServerIds}
                  overrideTimeRange={overrideTimeRange}
                  xOffset={xOffset}
                />
              </div>
            </CollapsibleContent>
          </Collapsible>
        </>
      ) : null}
    </div>
  )
}

