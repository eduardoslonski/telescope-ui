
import { useState, useCallback, useEffect, useMemo, useRef } from "react"
import uPlot from "uplot"
import { useAtomValue } from "jotai"
import { X, SlidersHorizontal, Loader2 } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"
import { darkModeAtom } from "@/lib/atoms"
import {
  formatClockTimeAdaptive,
  formatSecondsCompact,
  formatSecondsHuman,
  formatValueSmart,
} from "@/lib/format"
import type { GpuMetric } from "@/lib/types"

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

export const SYSTEM_METRIC_INFO: Record<string, { label: string; unit: string }> = {
  gpu_memory_used_gb: { label: "Memory Used", unit: "GB" },
  gpu_memory_total_gb: { label: "Memory Total", unit: "GB" },
  gpu_memory_free_gb: { label: "Memory Free", unit: "GB" },
  gpu_temperature_c: { label: "Temperature", unit: "degC" },
  gpu_power_w: { label: "Power", unit: "W" },
  gpu_power_limit_w: { label: "Power Limit", unit: "W" },
  gpu_utilization_percent: { label: "GPU Utilization", unit: "%" },
  gpu_memory_used_percent: { label: "Memory Capacity Used", unit: "%" },
  gpu_memory_bandwidth_utilization_percent: { label: "Memory Bandwidth Utilization", unit: "%" },
  gpu_clock_sm_mhz: { label: "SM Clock", unit: "MHz" },
  gpu_clock_mem_mhz: { label: "Memory Clock", unit: "MHz" },
  gpu_fan_speed_percent: { label: "Fan Speed", unit: "%" },
  torch_allocated_gb: { label: "Torch Memory Allocated", unit: "GB" },
  torch_reserved_gb: { label: "Torch Memory Reserved", unit: "GB" },
  torch_max_allocated_gb: { label: "Torch Max Memory Allocated", unit: "GB" },
  cpu_utilization_percent: { label: "CPU Utilization", unit: "%" },
  system_memory_total_gb: { label: "Memory Total", unit: "GB" },
  system_memory_used_gb: { label: "Memory Used", unit: "GB" },
  system_memory_available_gb: { label: "Memory Available", unit: "GB" },
  system_memory_percent: { label: "Memory Usage", unit: "%" },
  // vLLM metrics
  requests_running: { label: "Requests Running", unit: "" },
  requests_waiting: { label: "Requests Waiting", unit: "" },
  cache_usage: { label: "KV Cache Usage", unit: "%" },
  cache_hit_rate: { label: "KV Cache Hit Rate", unit: "%" },
  requests_total: { label: "Requests Total", unit: "" },
  prompt_tokens_total: { label: "Prompt Tokens Total", unit: "" },
  rollout_tokens_total: { label: "Rollout Tokens Total", unit: "" },
  preemptions_total: { label: "Preemptions Total", unit: "" },
  cache_hits_total: { label: "Cache Hits Total", unit: "" },
  cache_queries_total: { label: "Cache Queries Total", unit: "" },
  ttft_mean: { label: "TTFT (Mean)", unit: "s" },
  e2e_latency_mean: { label: "E2E Latency (Mean)", unit: "s" },
  itl_mean: { label: "ITL (Mean)", unit: "s" },
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
// Helpers
// ============================================================================

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
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
  const rgbMatch = color.match(/^rgb\(([^)]+)\)$/i)
  if (rgbMatch) return `rgba(${rgbMatch[1]}, ${alpha})`
  const rgbaMatch = color.match(/^rgba\(([^)]+)\)$/i)
  if (rgbaMatch) {
    const channels = rgbaMatch[1].split(",").slice(0, 3).join(",")
    return `rgba(${channels}, ${alpha})`
  }
  return color
}

interface GpuSeriesEntry {
  key: string
  label: string
  stroke: string
  gpuIndex: number
  nodeId: number | null
}

/** Optional map from series key → role name for tooltip grouping */
export type SeriesRoleMap = Record<string, string>

// ============================================================================
// GpuMetricChart
// ============================================================================

export function GpuMetricChart({
  metricName,
  data,
  gpuIndices,
  intervalStart,
  intervalEnd,
  variant = "default",
  strokeColor,
  gpuDisplayNames,
  isLoading,
  seriesMode = "gpu",
  showLegend,
  seriesRoles,
  activeSeriesKeys = null,
  xOffset = 0,
  xAxisMode = "elapsed",
  xAxisTicks = "auto",
  isRefetching = false,
}: {
  metricName: string
  data: GpuMetric[]
  gpuIndices: number[]
  intervalStart: number
  intervalEnd: number
  variant?: "default" | "timeline"
  strokeColor?: string
  gpuDisplayNames?: Record<number, string>
  isLoading?: boolean
  seriesMode?: "gpu" | "node_gpu"
  showLegend?: boolean
  seriesRoles?: SeriesRoleMap
  /** When set, series not in this set render dimmed. */
  activeSeriesKeys?: Set<string> | null
  /** Offset in seconds from run start to intervalStart, for global x-axis labels */
  xOffset?: number
  /** Axis label mode: elapsed from run start, or absolute wall clock */
  xAxisMode?: "elapsed" | "clock"
  /** X tick strategy: auto ticks or only window bounds */
  xAxisTicks?: "auto" | "bounds"
  /** Background refetch indicator (keep plot visible, show spinner) */
  isRefetching?: boolean
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

  const isTimeline = variant === "timeline"
  const useClockBoundsTicks = xAxisMode === "clock" && xAxisTicks === "bounds"

  const seriesEntries = useMemo<GpuSeriesEntry[]>(() => {
    if (seriesMode === "node_gpu") {
      const dedup = new Map<string, GpuSeriesEntry>()
      const sorted = [...data].sort((a, b) => {
        const nodeA = isFiniteNumber(a.node_id) ? a.node_id : Number.MAX_SAFE_INTEGER
        const nodeB = isFiniteNumber(b.node_id) ? b.node_id : Number.MAX_SAFE_INTEGER
        if (nodeA !== nodeB) return nodeA - nodeB
        return a.gpu_index - b.gpu_index
      })
      for (const metric of sorted) {
        const nodeId = isFiniteNumber(metric.node_id) ? metric.node_id : null
        const key = `${nodeId ?? "?"}:${metric.gpu_index}`
        if (dedup.has(key)) continue
        const nodeLabel = nodeId === null ? "Node ?" : `Node ${nodeId}`
        const index = dedup.size
        dedup.set(key, {
          key,
          label: `${nodeLabel} GPU ${metric.gpu_index}`,
          stroke: strokeColor || MULTI_SERIES_COLORS[index % MULTI_SERIES_COLORS.length],
          gpuIndex: metric.gpu_index,
          nodeId,
        })
      }
      return Array.from(dedup.values())
    }

    const sourceIndices =
      gpuIndices.length > 0
        ? Array.from(new Set(gpuIndices)).sort((a, b) => a - b)
        : Array.from(new Set(data.map((metric) => metric.gpu_index))).sort((a, b) => a - b)

    return sourceIndices.map((gpuIndex) => ({
      key: `gpu:${gpuIndex}`,
      label: gpuDisplayNames?.[gpuIndex] ?? `GPU ${gpuIndex}`,
      stroke: strokeColor || MULTI_SERIES_COLORS[gpuIndex % MULTI_SERIES_COLORS.length],
      gpuIndex,
      nodeId: null,
    }))
  }, [seriesMode, data, gpuIndices, gpuDisplayNames, strokeColor])

  const seriesKeySet = useMemo(
    () => new Set(seriesEntries.map((entry) => entry.key)),
    [seriesEntries]
  )

  const { uplotData, seriesConfig, hasData, outlierBounds } = useMemo(() => {
    if (data.length === 0 || seriesEntries.length === 0) {
      return { uplotData: null, seriesConfig: [] as uPlot.Series[], hasData: false, outlierBounds: null }
    }

    const byTimestamp: Record<number, Record<string, number>> = {}
    const timestamps = new Set<number>()
    const allValues: number[] = []

    for (const metric of data) {
      const key =
        seriesMode === "node_gpu"
          ? `${isFiniteNumber(metric.node_id) ? metric.node_id : "?"}:${metric.gpu_index}`
          : `gpu:${metric.gpu_index}`

      if (!seriesKeySet.has(key)) continue
      timestamps.add(metric.timestamp)
      if (!byTimestamp[metric.timestamp]) byTimestamp[metric.timestamp] = {}
      byTimestamp[metric.timestamp][key] = metric.value
      allValues.push(metric.value)
    }

    const sortedTimes = Array.from(timestamps).sort((a, b) => a - b)
    if (sortedTimes.length === 0) {
      return { uplotData: null, seriesConfig: [] as uPlot.Series[], hasData: false, outlierBounds: null }
    }

    const xData = sortedTimes.map((timestamp) => timestamp - intervalStart)
    const series: uPlot.Series[] = [{ label: "Time" }]
    const dataArrays: (number | null)[][] = [xData]

    for (const entry of seriesEntries) {
      const values: (number | null)[] = sortedTimes.map(
        (timestamp) => byTimestamp[timestamp]?.[entry.key] ?? null
      )
      const isDimmed =
        activeSeriesKeys !== null &&
        activeSeriesKeys.size > 0 &&
        !activeSeriesKeys.has(entry.key)
      dataArrays.push(values)
      series.push({
        label: entry.label,
        stroke: isDimmed ? withAlpha(entry.stroke, 0.25) : entry.stroke,
        width: isDimmed ? 1 : 1.5,
        spanGaps: true,
        points: { show: false },
      })
    }

    return {
      uplotData: dataArrays as uPlot.AlignedData,
      seriesConfig: series,
      hasData: true,
      outlierBounds: computeIQRBounds(allValues),
    }
  }, [data, intervalStart, seriesEntries, seriesMode, seriesKeySet, activeSeriesKeys])

  const shouldShowLegend = showLegend ?? (!isTimeline && seriesEntries.length > 1)

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
    const height = 192

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
    const padding = Math.max(range * 0.1, minPadding)
    const yDomain: [number, number] = [
      minY !== null ? minY : Math.max(0, minVal - padding),
      maxY !== null ? maxY : maxVal + padding,
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
      padding: isTimeline
        ? [0, 0, 0, 0]
        : useClockBoundsTicks
          ? [4, 24, 0, 4]
          : [4, 8, 0, 0],
      cursor: {
        show: true,
        x: true,
        y: false,
        points: { show: false },
        drag: { x: !isTimeline, y: false },
      },
      legend: {
        show: shouldShowLegend,
      },
      ...(isTimeline ? {} : { select: { show: true, left: 0, top: 0, width: 0, height: 0 } }),
      scales: {
        x: {
          time: false,
          range: (_u: uPlot, dataMin: number, dataMax: number) => {
            const windowEnd = intervalEnd - intervalStart
            if (windowEnd > 0) {
              // Keep viewport pinned to the selected interval.
              return [0, windowEnd]
            }
            return [dataMin, dataMax]
          },
        },
        y: { range: yDomain },
      },
      axes: isTimeline
        ? [
            { show: false, size: 0 },
            { show: false, size: 0 },
          ]
        : [
            {
              stroke: tickLabelColor,
              grid: { stroke: gridColor, width: 1 },
              ticks: { stroke: gridColor, width: 1 },
              font: "10px system-ui, sans-serif",
              labelFont: "10px system-ui, sans-serif",
              size: 24,
              splits:
                xAxisTicks === "bounds"
                  ? (
                      _u: uPlot,
                      _axisIdx: number,
                      min: number,
                      max: number
                    ) => [min, max]
                  : undefined,
              values: (_u, vals) =>
                vals.map((v) => {
                  if (xAxisMode === "clock") {
                    return formatClockTimeAdaptive(
                      intervalStart + Number(v),
                      intervalEnd - intervalStart
                    )
                  }
                  return formatSecondsCompact(Number(v) + xOffset)
                }),
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
            if (idx === null || idx === undefined || left === undefined || left < 0) {
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
            const absTimestamp = intervalStart + Number(relTime)
            const clockStr = formatClockTimeAdaptive(
              absTimestamp,
              intervalEnd - intervalStart
            )

            let html = ""
            if (!isTimeline) {
              html += `<div class="font-medium mb-1">${clockStr}</div>`
              html += `<div class="text-[10px] text-muted-foreground font-sans">Time Run: ${elapsedStr}</div>`
              html += `<div class="text-[10px] text-muted-foreground font-sans mb-1">Timestamp: ${absTimestamp.toFixed(3)}</div>`
            }

            // Collect values per series
            const tooltipEntries: Array<{
              entry: GpuSeriesEntry
              formatted: string
              role: string | null
            }> = []

            for (let i = 0; i < seriesEntries.length; i += 1) {
              const entry = seriesEntries[i]
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

              tooltipEntries.push({
                entry,
                formatted: formatValueSmart(value),
                role: seriesRoles?.[entry.key] ?? null,
              })
            }

            const hasAnyValue = tooltipEntries.length > 0

            // Group by role for display
            const hasRoles = tooltipEntries.some((e) => e.role !== null)
            if (hasRoles && !isTimeline) {
              // Role order: Trainer first, then Inference, then anything else
              const ROLE_PRIORITY: Record<string, number> = { Trainer: 0, Inference: 1 }
              const roleOrder: string[] = []
              for (const te of tooltipEntries) {
                const r = te.role ?? "Other"
                if (!roleOrder.includes(r)) roleOrder.push(r)
              }
              roleOrder.sort((a, b) => (ROLE_PRIORITY[a] ?? 99) - (ROLE_PRIORITY[b] ?? 99))
              for (const role of roleOrder) {
                html += `<div class="text-[10px] font-medium text-muted-foreground mt-1 mb-0.5">${role}</div>`
                for (const te of tooltipEntries) {
                  if ((te.role ?? "Other") !== role) continue
                  html += `
                    <div class="flex items-center gap-2 mt-0.5">
                      <div class="w-2 h-2 rounded-full shrink-0" style="background-color: ${te.entry.stroke}"></div>
                      <span class="text-muted-foreground">${te.entry.label}:</span>
                      <span class="font-medium">${te.formatted} ${metricInfo.unit}</span>
                    </div>
                  `
                }
              }
            } else {
              for (const te of tooltipEntries) {
                if (isTimeline) {
                  html += `
                    <div class="flex items-center gap-2">
                      <span class="text-muted-foreground">${te.entry.label}:</span>
                      <span class="font-medium">${te.formatted} ${metricInfo.unit}</span>
                    </div>
                  `
                } else {
                  html += `
                    <div class="flex items-center gap-2 mt-0.5">
                      <div class="w-2 h-2 rounded-full shrink-0" style="background-color: ${te.entry.stroke}"></div>
                      <span class="text-muted-foreground">${te.entry.label}:</span>
                      <span class="font-medium">${te.formatted} ${metricInfo.unit}</span>
                    </div>
                  `
                }
              }
            }

            if (!hasAnyValue) {
              tooltipRef.current.style.display = "none"
              return
            }

            if (isTimeline) {
              html += `<div class="text-muted-foreground mt-1">Time: ${elapsedStr}</div>`
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
  }, [
    uplotData,
    hasData,
    seriesConfig,
    metricName,
    intervalStart,
    intervalEnd,
    xOffset,
    xAxisMode,
    xAxisTicks,
    useClockBoundsTicks,
    isTimeline,
    metricInfo.unit,
    shouldShowLegend,
    seriesEntries,
    seriesRoles,
    formatYAxisTick,
    ignoreOutliers,
    outlierBounds,
    minY,
    maxY,
    darkMode,
  ])

  const handleMouseLeave = useCallback(() => {
    if (tooltipRef.current) tooltipRef.current.style.display = "none"
  }, [])

  if (!hasData && isTimeline) {
    return (
      <div className="h-48 bg-muted/30 rounded-lg border border-border/50 opacity-50 transition-opacity duration-200" />
    )
  }

  return (
    <div
      className={cn(
        "rounded-lg border transition-opacity",
        isTimeline
          ? "overflow-hidden"
          : "group/chart border-border p-3 bg-background",
        isLoading && "opacity-60"
      )}
    >
      {!isTimeline && (
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
      )}
      {hasData ? (
        <div className="h-48 relative bg-background rounded" ref={containerRef} onMouseLeave={handleMouseLeave}>
        {!isTimeline && isRefetching && (
          <Loader2 className="absolute bottom-0.5 left-0.5 h-3 w-3 animate-spin text-muted-foreground" />
        )}
        <div
          ref={tooltipRef}
          className="fixed z-[9999] max-w-[360px] bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl border border-border pointer-events-none"
          style={{ display: "none" }}
        />
      </div>
      ) : (
        <div className="h-48 flex items-center justify-center text-muted-foreground text-xs rounded">
          {isLoading ? "Loading..." : `No data for ${metricInfo.label}`}
        </div>
      )}
    </div>
  )
}
