import { useEffect, useMemo, useRef, useState } from "react"
import { useAtom, useAtomValue } from "jotai"
import { ChevronDown } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { NoRunSelectedState } from "@/components/no-run-selected-state"
import { ToggleWithInput } from "@/components/ui/toggle-with-input"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { PaginationControls } from "@/components/pagination-controls"
import {
  selectedRunPathAtom,
  infraViewTabAtom,
  infraPageAtom,
  infraIntervalAtom,
  infraLiveAtom,
  infraAggregateEnabledAtom,
  infraAggregateWindowAtom,
  infraScrollTopAtom,
  infraRoleModeAtom,
  infraSystemMetricsOpenAtom,
  infraCpuMetricsOpenAtom,
  infraVllmMetricsOpenAtom,
  infraThreadPoolMetricsOpenAtom,
  infraTrainerSectionOpenAtom,
  infraInferenceSectionOpenAtom,
  infraTrainerNodeModeAtom,
  infraInferenceNodeModeAtom,
} from "@/lib/atoms"
import type { InfraViewTab } from "@/lib/atoms"
import {
  useRunSummary,
  usePaginatedGpuMetrics,
  usePaginatedCpuMetrics,
  usePaginatedVllmMetrics,
  usePaginatedThreadPoolMetrics,
} from "@/hooks/use-run-data"
import {
  SystemMetricsCharts,
  type GpuIdentifier,
} from "@/components/system-metrics-charts"
import { TopologyViewer, parseTopology, parseSetupJson, asObject, asNumber } from "@/components/topology-viewer"
import { ModelArchitectureViewer } from "@/components/model-architecture-viewer"
import { formatClockTimeAdaptive, formatDurationHms } from "@/lib/format"
import type { GpuMetric, CpuMetric, VllmMetric, ThreadPoolMetric } from "@/lib/types"


function getSetupObject(
  summary: Record<string, unknown> | undefined,
): Record<string, unknown> | null {
  return parseSetupJson(summary?.setup)
}

/** Extract trainer GPU identifiers from v6.0 setup */
function extractTrainerGpusFromSetup(
  summary: Record<string, unknown> | undefined,
): GpuIdentifier[] {
  const setup = getSetupObject(summary)
  const result: GpuIdentifier[] = []

  const trainer = asObject(setup?.trainer)
  const trainerNodes = Array.isArray(trainer?.nodes) ? trainer.nodes : []
  for (const nodeEntry of trainerNodes) {
    const node = asObject(nodeEntry)
    if (!node) continue
    const nodeId = asNumber(node.node_id)
    if (nodeId === null) continue
    const gpus = Array.isArray(node.gpus) ? node.gpus : []
    for (const gpuEntry of gpus) {
      const gpu = asObject(gpuEntry)
      if (!gpu) continue
      const gpuIndex = asNumber(gpu.gpu_index)
      if (gpuIndex === null) continue
      result.push({ node_id: nodeId, gpu_index: gpuIndex })
    }
  }

  return result
}

/** Extract inference GPU identifiers from v6.0 setup */
function extractInferenceGpusFromSetup(
  summary: Record<string, unknown> | undefined,
): GpuIdentifier[] {
  const setup = getSetupObject(summary)
  const result: GpuIdentifier[] = []

  const inference = asObject(setup?.inference)
  const inferenceNodes = Array.isArray(inference?.nodes) ? inference.nodes : []
  for (const nodeEntry of inferenceNodes) {
    const node = asObject(nodeEntry)
    if (!node) continue
    const nodeId = asNumber(node.node_id)
    if (nodeId === null) continue
    const gpus = Array.isArray(node.gpus) ? node.gpus : []
    for (const gpuEntry of gpus) {
      const gpu = asObject(gpuEntry)
      if (!gpu) continue
      const gpuIndex = asNumber(gpu.gpu_index)
      if (gpuIndex === null) continue
      result.push({ node_id: nodeId, gpu_index: gpuIndex })
    }
  }

  return result
}

// Aggregate CPU metric names to fetch (skip per-core)
const CPU_METRIC_NAMES = [
  "cpu_utilization_percent",
  "system_memory_used_gb",
  "system_memory_available_gb",
  "system_memory_total_gb",
  "system_memory_percent",
]

// ============================================================================
// Duration helpers (matching timeline page)
// ============================================================================

const INTERVAL_PRESETS = [1, 5, 10, 30, 60, 120, 300]

const formatDuration = formatDurationHms

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

// ============================================================================
// Aggregation Helpers
// ============================================================================

/** Format aggregate window for display: 0.1 -> "0.1s", 1 -> "1s", 60 -> "1m" etc. */
function formatAggWindow(seconds: number): string {
  if (seconds < 1) return `${seconds}s`
  if (seconds < 60) return `${Math.round(seconds)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  if (s === 0) return `${m}m`
  return `${m}m ${s}s`
}

/**
 * Parse an aggregate window input.
 * Accepts: "0.1s", "1s", "5s", "1m", "500ms", or a plain number (seconds).
 */
function parseAggWindow(input: string): number | null {
  const trimmed = input.trim().toLowerCase()
  if (!trimmed) return null

  // "500ms" -> 0.5s
  const msMatch = trimmed.match(/^(\d+(?:\.\d+)?)\s*ms$/i)
  if (msMatch) {
    const ms = parseFloat(msMatch[1])
    return ms > 0 ? ms / 1000 : null
  }

  // Try compound duration "1m 30s", "2m", "30s", etc.
  const compoundRe = /^(?:(\d+)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*s)?$/i
  const compoundMatch = trimmed.match(compoundRe)
  if (compoundMatch && (compoundMatch[1] || compoundMatch[2])) {
    const m = parseInt(compoundMatch[1] || "0", 10)
    const s = parseFloat(compoundMatch[2] || "0")
    const total = m * 60 + s
    return total > 0 ? total : null
  }

  // Plain number -> seconds
  const num = parseFloat(trimmed)
  if (Number.isFinite(num) && num > 0) return num

  return null
}

/** Aggregate GPU metrics by bucketing timestamps into windows and averaging values. */
function aggregateGpuMetrics(
  metrics: GpuMetric[],
  windowSeconds: number,
): GpuMetric[] {
  if (windowSeconds <= 0 || metrics.length === 0) return metrics

  // Group by (metric_name, gpu_index, node_id)
  const groups = new Map<
    string,
    {
      key: {
        metric_name: string
        gpu_index: number
        node_id?: number | null
        source?: string | null
        rank?: number | null
        local_rank?: number | null
      }
      entries: GpuMetric[]
    }
  >()

  for (const m of metrics) {
    const groupKey = `${m.metric_name}|${m.gpu_index}|${m.node_id ?? ""}`
    let group = groups.get(groupKey)
    if (!group) {
      group = {
        key: {
          metric_name: m.metric_name,
          gpu_index: m.gpu_index,
          node_id: m.node_id,
          source: m.source,
          rank: m.rank,
          local_rank: m.local_rank,
        },
        entries: [],
      }
      groups.set(groupKey, group)
    }
    group.entries.push(m)
  }

  const result: GpuMetric[] = []
  for (const group of groups.values()) {
    // Sort by timestamp
    group.entries.sort((a, b) => a.timestamp - b.timestamp)

    // Bucket
    const buckets = new Map<
      number,
      { sum: number; count: number; ts: number }
    >()
    for (const m of group.entries) {
      const bucketKey = Math.floor(m.timestamp / windowSeconds)
      let bucket = buckets.get(bucketKey)
      if (!bucket) {
        bucket = { sum: 0, count: 0, ts: 0 }
        buckets.set(bucketKey, bucket)
      }
      bucket.sum += m.value
      bucket.count += 1
      bucket.ts += m.timestamp
    }

    for (const bucket of buckets.values()) {
      result.push({
        ...group.key,
        timestamp: bucket.ts / bucket.count, // use mean timestamp
        value: bucket.sum / bucket.count,
      })
    }
  }

  return result
}

/** Aggregate CPU metrics by bucketing timestamps into windows and averaging values. */
function aggregateCpuMetrics(
  metrics: CpuMetric[],
  windowSeconds: number,
): CpuMetric[] {
  if (windowSeconds <= 0 || metrics.length === 0) return metrics

  // Group by (metric_name, node_id)
  const groups = new Map<
    string,
    {
      key: {
        metric_name: string
        node_id?: number | null
        source?: string | null
      }
      entries: CpuMetric[]
    }
  >()

  for (const m of metrics) {
    const groupKey = `${m.metric_name}|${m.node_id ?? ""}`
    let group = groups.get(groupKey)
    if (!group) {
      group = {
        key: {
          metric_name: m.metric_name,
          node_id: m.node_id,
          source: m.source,
        },
        entries: [],
      }
      groups.set(groupKey, group)
    }
    group.entries.push(m)
  }

  const result: CpuMetric[] = []
  for (const group of groups.values()) {
    group.entries.sort((a, b) => a.timestamp - b.timestamp)

    const buckets = new Map<
      number,
      { sum: number; count: number; ts: number }
    >()
    for (const m of group.entries) {
      const bucketKey = Math.floor(m.timestamp / windowSeconds)
      let bucket = buckets.get(bucketKey)
      if (!bucket) {
        bucket = { sum: 0, count: 0, ts: 0 }
        buckets.set(bucketKey, bucket)
      }
      bucket.sum += m.value
      bucket.count += 1
      bucket.ts += m.timestamp
    }

    for (const bucket of buckets.values()) {
      result.push({
        ...group.key,
        timestamp: bucket.ts / bucket.count,
        value: bucket.sum / bucket.count,
      })
    }
  }

  return result
}

/** Aggregate vLLM metrics by bucketing timestamps into windows and averaging values. */
function aggregateVllmMetrics(
  metrics: VllmMetric[],
  windowSeconds: number,
): VllmMetric[] {
  if (windowSeconds <= 0 || metrics.length === 0) return metrics

  // Group by (metric_name, server)
  const groups = new Map<
    string,
    {
      key: {
        metric_name: string
        server: number
        node_id?: number | null
        tp_group_id?: number | null
        tp_size?: number | null
      }
      entries: VllmMetric[]
    }
  >()

  for (const m of metrics) {
    const groupKey = `${m.metric_name}|${m.server}`
    let group = groups.get(groupKey)
    if (!group) {
      group = {
        key: {
          metric_name: m.metric_name,
          server: m.server,
          node_id: m.node_id,
          tp_group_id: m.tp_group_id,
          tp_size: m.tp_size,
        },
        entries: [],
      }
      groups.set(groupKey, group)
    }
    group.entries.push(m)
  }

  const result: VllmMetric[] = []
  for (const group of groups.values()) {
    group.entries.sort((a, b) => a.timestamp - b.timestamp)

    const buckets = new Map<
      number,
      { sum: number; count: number; ts: number }
    >()
    for (const m of group.entries) {
      const bucketKey = Math.floor(m.timestamp / windowSeconds)
      let bucket = buckets.get(bucketKey)
      if (!bucket) {
        bucket = { sum: 0, count: 0, ts: 0 }
        buckets.set(bucketKey, bucket)
      }
      bucket.sum += m.value
      bucket.count += 1
      bucket.ts += m.timestamp
    }

    for (const bucket of buckets.values()) {
      result.push({
        ...group.key,
        timestamp: bucket.ts / bucket.count,
        value: bucket.sum / bucket.count,
      })
    }
  }

  return result
}

// ============================================================================
// IntervalPicker (matching timeline page)
// ============================================================================

function IntervalPicker({
  value,
  onChange,
}: {
  value: number
  onChange: (v: number) => void
}) {
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState("")
  const inputRef = useRef<HTMLInputElement>(null)

  const handleOpenChange = (newOpen: boolean) => {
    if (newOpen) {
      setDraft(formatDuration(value))
      setTimeout(() => inputRef.current?.select(), 0)
    }
    setOpen(newOpen)
  }

  const commitDraft = () => {
    const parsed = parseDuration(draft)
    if (parsed !== null) {
      onChange(parsed)
      setOpen(false)
    }
  }

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <button className="inline-flex items-center gap-1 rounded-lg border border-input bg-transparent px-2 h-7 text-xs hover:bg-accent transition-colors select-none whitespace-nowrap">
          {formatDuration(value)}
          <ChevronDown className="size-3 text-muted-foreground" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-40 p-1.5 gap-1">
        <div className="flex items-center gap-1">
          <input
            ref={inputRef}
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitDraft()
            }}
            className="h-7 w-full rounded-md border border-input bg-transparent px-2 text-xs outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[2px]"
            placeholder="e.g. 30s, 2m, 1h"
          />
          <button
            onClick={commitDraft}
            className="shrink-0 h-7 px-2 rounded-md text-xs font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            Go
          </button>
        </div>
        <div className="flex flex-col">
          {INTERVAL_PRESETS.map((preset) => (
            <button
              key={preset}
              onClick={() => {
                onChange(preset)
                setOpen(false)
              }}
              className={`text-left rounded-md px-2 py-1 text-xs transition-colors ${
                preset === value
                  ? "bg-accent text-accent-foreground font-medium"
                  : "hover:bg-accent hover:text-accent-foreground"
              }`}
            >
              {formatDuration(preset)}
            </button>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  )
}

// ============================================================================
// Infra Page
// ============================================================================

export default function InfraPage() {
  const [scrollRoot, setScrollRoot] = useState<HTMLDivElement | null>(null)
  const scrollTopRef = useRef(0)
  const selectedRunPath = useAtomValue(selectedRunPathAtom)
  const shouldPoll = true

  // View tab: Metrics vs Topology
  const [viewTab, setViewTab] = useAtom(infraViewTabAtom)

  // Pagination & interval (persisted in atoms)
  const [page, setPage] = useAtom(infraPageAtom)
  const [intervalSeconds, setIntervalSeconds] = useAtom(infraIntervalAtom)
  const [isLive, setIsLive] = useAtom(infraLiveAtom)

  // Aggregate settings
  const [aggregateEnabled, setAggregateEnabled] = useAtom(
    infraAggregateEnabledAtom,
  )
  const [aggregateWindow, setAggregateWindow] = useAtom(
    infraAggregateWindowAtom,
  )
  const [savedScrollTop, setSavedScrollTop] = useAtom(infraScrollTopAtom)
  const [roleMode, setRoleMode] = useAtom(infraRoleModeAtom)
  const [systemMetricsOpen, setSystemMetricsOpen] = useAtom(
    infraSystemMetricsOpenAtom,
  )
  const [cpuMetricsOpen, setCpuMetricsOpen] = useAtom(infraCpuMetricsOpenAtom)
  const [vllmMetricsOpen, setVllmMetricsOpen] = useAtom(
    infraVllmMetricsOpenAtom,
  )
  const [threadPoolMetricsOpen, setThreadPoolMetricsOpen] = useAtom(
    infraThreadPoolMetricsOpenAtom,
  )
  const [trainerSectionOpen, setTrainerSectionOpen] = useAtom(
    infraTrainerSectionOpenAtom,
  )
  const [inferenceSectionOpen, setInferenceSectionOpen] = useAtom(
    infraInferenceSectionOpenAtom,
  )
  const [trainerNodeMode, setTrainerNodeMode] = useAtom(
    infraTrainerNodeModeAtom,
  )
  const [inferenceNodeMode, setInferenceNodeMode] = useAtom(
    infraInferenceNodeModeAtom,
  )
  const [aggWindowInput, setAggWindowInput] = useState(
    formatAggWindow(aggregateWindow),
  )

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

  const { data: summaryData, error } = useRunSummary(
    selectedRunPath || "",
    !!selectedRunPath,
    shouldPoll,
  )

  // Topology data (derived from the summary)
  const topology = useMemo(
    () => parseTopology(summaryData?.summary),
    [summaryData?.summary],
  )

  // Fetch paginated GPU metrics
  const {
    data: gpuData,
    isFetching: gpuFetching,
    isPlaceholderData: gpuIsPlaceholder,
  } = usePaginatedGpuMetrics(
    selectedRunPath || "",
    page,
    intervalSeconds,
    isLive,
    !!selectedRunPath,
    shouldPoll,
  )

  // Wait for GPU to resolve once so CPU can use a stable anchor and avoid double-fetch flicker.
  const cpuQueryEnabled = !!selectedRunPath && gpuData !== undefined

  // Fetch paginated CPU metrics
  const {
    data: cpuData,
    isFetching: cpuFetching,
    isPlaceholderData: cpuIsPlaceholder,
  } = usePaginatedCpuMetrics(
    selectedRunPath || "",
    page,
    intervalSeconds,
    gpuData?.global_min_time ?? null,
    isLive,
    CPU_METRIC_NAMES,
    cpuQueryEnabled,
    shouldPoll,
  )

  // Fetch paginated vLLM metrics (anchor to GPU start time like CPU)
  const vllmQueryEnabled = !!selectedRunPath && gpuData !== undefined
  const {
    data: vllmData,
    isFetching: vllmFetching,
    isPlaceholderData: vllmIsPlaceholder,
  } = usePaginatedVllmMetrics(
    selectedRunPath || "",
    page,
    intervalSeconds,
    gpuData?.global_min_time ?? null,
    isLive,
    null, // fetch all metrics
    vllmQueryEnabled,
    shouldPoll,
  )

  // Fetch paginated thread pool metrics (anchor to GPU start time like CPU)
  const threadPoolQueryEnabled = !!selectedRunPath && gpuData !== undefined
  const {
    data: threadPoolData,
    isFetching: threadPoolFetching,
    isPlaceholderData: threadPoolIsPlaceholder,
  } = usePaginatedThreadPoolMetrics(
    selectedRunPath || "",
    page,
    intervalSeconds,
    gpuData?.global_min_time ?? null,
    isLive,
    null, // fetch all metrics
    threadPoolQueryEnabled,
    shouldPoll,
  )

  const knownTotalPages = gpuData?.total_pages ?? cpuData?.total_pages ?? null

  const displayedGpuData = gpuData
  const displayedCpuData = cpuData
  const displayedVllmData = vllmData
  const displayedThreadPoolData = threadPoolData

  // Keep metric card layout stable during page changes, even with no-cache queries.
  const [stickyGpuMetricNames, setStickyGpuMetricNames] = useState<string[]>([])
  const [stickyCpuMetricNames, setStickyCpuMetricNames] = useState<string[]>([])
  const [stickyVllmMetricNames, setStickyVllmMetricNames] = useState<string[]>(
    [],
  )
  const [stickyThreadPoolMetricNames, setStickyThreadPoolMetricNames] = useState<string[]>([])
  const [stickyThreadPoolNames, setStickyThreadPoolNames] = useState<string[]>([])

  useEffect(() => {
    if (
      displayedGpuData?.available_metrics &&
      displayedGpuData.available_metrics.length > 0
    ) {
      setStickyGpuMetricNames(displayedGpuData.available_metrics)
    }
  }, [displayedGpuData?.available_metrics])

  useEffect(() => {
    if (
      displayedCpuData?.available_metrics &&
      displayedCpuData.available_metrics.length > 0
    ) {
      setStickyCpuMetricNames(displayedCpuData.available_metrics)
    }
  }, [displayedCpuData?.available_metrics])

  useEffect(() => {
    if (
      displayedVllmData?.available_metrics &&
      displayedVllmData.available_metrics.length > 0
    ) {
      setStickyVllmMetricNames(displayedVllmData.available_metrics)
    }
  }, [displayedVllmData?.available_metrics])

  useEffect(() => {
    if (
      displayedThreadPoolData?.available_metrics &&
      displayedThreadPoolData.available_metrics.length > 0
    ) {
      setStickyThreadPoolMetricNames(displayedThreadPoolData.available_metrics)
    }
    if (
      displayedThreadPoolData?.available_pools &&
      displayedThreadPoolData.available_pools.length > 0
    ) {
      setStickyThreadPoolNames(displayedThreadPoolData.available_pools)
    }
  }, [displayedThreadPoolData?.available_metrics, displayedThreadPoolData?.available_pools])

  // Keep pagination/time metadata stable during in-flight fetches.
  const [stickyTotalPages, setStickyTotalPages] = useState(1)
  const [stickyGlobalMinTime, setStickyGlobalMinTime] = useState<number | null>(
    null,
  )
  const [stickyPageBounds, setStickyPageBounds] = useState<{
    start: number
    end: number
  } | null>(null)
  const [stickyTimeRangeOverride, setStickyTimeRangeOverride] = useState<{
    start: number
    end: number
  } | null>(null)

  useEffect(() => {
    if (knownTotalPages !== null && knownTotalPages > 0) {
      setStickyTotalPages(knownTotalPages)
    }
  }, [knownTotalPages])

  useEffect(() => {
    const nextGlobalMin =
      displayedGpuData?.global_min_time ??
      displayedCpuData?.global_min_time ??
      null
    if (nextGlobalMin !== null) {
      setStickyGlobalMinTime(nextGlobalMin)
    }
  }, [displayedGpuData?.global_min_time, displayedCpuData?.global_min_time])

  useEffect(() => {
    const start =
      displayedGpuData?.interval_start ??
      displayedCpuData?.interval_start ??
      null
    const end =
      displayedGpuData?.interval_end ?? displayedCpuData?.interval_end ?? null
    if (start !== null && end !== null) {
      setStickyPageBounds({ start, end })
    }
  }, [
    displayedGpuData?.interval_start,
    displayedGpuData?.interval_end,
    displayedCpuData?.interval_start,
    displayedCpuData?.interval_end,
  ])

  useEffect(() => {
    setStickyTotalPages(1)
    setStickyGlobalMinTime(null)
    setStickyPageBounds(null)
    setStickyTimeRangeOverride(null)
  }, [selectedRunPath])

  // Use GPU data for pagination info (total pages, time bounds)
  const totalPages = knownTotalPages ?? stickyTotalPages
  const displayedIntervalStart =
    displayedGpuData?.interval_start ?? displayedCpuData?.interval_start ?? null
  const displayedIntervalEnd =
    displayedGpuData?.interval_end ?? displayedCpuData?.interval_end ?? null
  const globalMinTime =
    displayedGpuData?.global_min_time ??
    displayedCpuData?.global_min_time ??
    stickyGlobalMinTime

  // Live mode: always go to last page when new data arrives
  useEffect(() => {
    // Avoid bouncing page while total_pages is unknown during in-flight fetches.
    if (isLive && knownTotalPages !== null && knownTotalPages > 0) {
      const lastPage = knownTotalPages - 1
      if (page !== lastPage) {
        setPage(lastPage)
      }
    }
  }, [isLive, knownTotalPages, page, setPage])

  // When user manually changes page, turn off live mode
  const handlePageChange = (newPage: number) => {
    setIsLive(false)
    setPage(newPage)
  }

  const handleIntervalChange = (newInterval: number) => {
    // Keep roughly the same time position when changing interval
    if (!isLive) {
      const newPage = Math.floor((page * intervalSeconds) / newInterval)
      setPage(newPage)
    }
    setIntervalSeconds(newInterval)
  }

  const handleLiveToggle = () => {
    if (!isLive) {
      // Turn on live: jump to last page
      setIsLive(true)
      if (knownTotalPages !== null && knownTotalPages > 0) {
        setPage(knownTotalPages - 1)
      }
    } else {
      setIsLive(false)
    }
  }

  // Keep input in sync with atom when it changes externally
  useEffect(() => {
    setAggWindowInput(formatAggWindow(aggregateWindow))
  }, [aggregateWindow])

  // Aggregate data when enabled
  const aggregatedGpuMetrics = useMemo(() => {
    const raw = displayedGpuData?.metrics ?? []
    if (!aggregateEnabled || aggregateWindow <= 0) return raw
    return aggregateGpuMetrics(raw, aggregateWindow)
  }, [displayedGpuData?.metrics, aggregateEnabled, aggregateWindow])

  const aggregatedCpuMetrics = useMemo(() => {
    const raw = displayedCpuData?.metrics ?? []
    if (!aggregateEnabled || aggregateWindow <= 0) return raw
    return aggregateCpuMetrics(raw, aggregateWindow)
  }, [displayedCpuData?.metrics, aggregateEnabled, aggregateWindow])

  const aggregatedVllmMetrics = useMemo(() => {
    const raw = displayedVllmData?.metrics ?? []
    if (!aggregateEnabled || aggregateWindow <= 0) return raw
    return aggregateVllmMetrics(raw, aggregateWindow)
  }, [displayedVllmData?.metrics, aggregateEnabled, aggregateWindow])

  // Compute the time range override from backend interval bounds
  // This ensures all plots share the same window (especially for Live mode)
  const computedTimeRangeOverride = useMemo<{
    start: number
    end: number
  } | null>(() => {
    if (isLive) {
      const liveEnds: number[] = []
      const liveStarts: number[] = []
      if (displayedGpuData?.interval_start != null)
        liveStarts.push(displayedGpuData.interval_start)
      if (displayedGpuData?.interval_end != null)
        liveEnds.push(displayedGpuData.interval_end)
      if (displayedCpuData?.interval_start != null)
        liveStarts.push(displayedCpuData.interval_start)
      if (displayedCpuData?.interval_end != null)
        liveEnds.push(displayedCpuData.interval_end)
      if (displayedVllmData?.interval_start != null)
        liveStarts.push(displayedVllmData.interval_start)
      if (displayedVllmData?.interval_end != null)
        liveEnds.push(displayedVllmData.interval_end)
      if (liveEnds.length === 0) return null
      const liveEnd = Math.min(...liveEnds)
      const requestedLiveStart = Math.max(
        globalMinTime ?? liveEnd,
        liveEnd - intervalSeconds,
      )
      // Clamp to the latest fetched start among sources so we don't render a
      // leading gap when one source lags and others fetch a newer live window.
      const fetchedLiveStart =
        liveStarts.length > 0 ? Math.max(...liveStarts) : requestedLiveStart
      const liveStart = Math.min(
        liveEnd,
        Math.max(requestedLiveStart, fetchedLiveStart),
      )
      return { start: liveStart, end: liveEnd }
    }
    const fallbackStart = stickyPageBounds?.start ?? null
    const fallbackEnd = stickyPageBounds?.end ?? null
    const start = displayedIntervalStart ?? fallbackStart
    const end = displayedIntervalEnd ?? fallbackEnd
    if (start === null || end === null) return null
    return { start, end }
  }, [
    isLive,
    displayedGpuData?.interval_end,
    displayedGpuData?.interval_start,
    displayedCpuData?.interval_end,
    displayedCpuData?.interval_start,
    displayedVllmData?.interval_end,
    displayedVllmData?.interval_start,
    globalMinTime,
    intervalSeconds,
    displayedIntervalStart,
    displayedIntervalEnd,
    stickyPageBounds?.start,
    stickyPageBounds?.end,
  ])

  useEffect(() => {
    if (computedTimeRangeOverride) {
      setStickyTimeRangeOverride(computedTimeRangeOverride)
    }
  }, [computedTimeRangeOverride])

  const timeRangeOverride = computedTimeRangeOverride ?? stickyTimeRangeOverride

  // Compute xOffset: offset from run start to the window start (for global x-axis labels)
  const xOffset = useMemo(() => {
    if (!timeRangeOverride || globalMinTime === null) return 0
    return timeRangeOverride.start - globalMinTime
  }, [timeRangeOverride, globalMinTime])

  const isTransitionLoading =
    (gpuFetching && gpuIsPlaceholder) ||
    (cpuFetching && cpuIsPlaceholder) ||
    (vllmFetching && vllmIsPlaceholder)
  const isInitialLoading =
    (gpuFetching && !displayedGpuData) ||
    (cpuQueryEnabled && cpuFetching && !displayedCpuData) ||
    (vllmQueryEnabled && vllmFetching && !displayedVllmData)
  const isBackgroundRefetching =
    !isTransitionLoading &&
    !isInitialLoading &&
    ((gpuFetching && !!displayedGpuData) ||
      (cpuFetching && !!displayedCpuData) ||
      (vllmFetching && !!displayedVllmData))
  const isChartsLoading = isTransitionLoading || isInitialLoading

  // Parse setup to get GPU role assignments
  const trainerGpus = useMemo(
    () => extractTrainerGpusFromSetup(summaryData?.summary),
    [summaryData?.summary],
  )

  const inferenceGpus = useMemo(
    () => extractInferenceGpusFromSetup(summaryData?.summary),
    [summaryData?.summary],
  )

  // No run selected
  if (!selectedRunPath) {
    return (
      <NoRunSelectedState description="Select a run from the sidebar to view infrastructure metrics." />
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="py-2.5 px-4">
          <div className="flex items-center gap-2 h-7">
            <ToggleGroup
              type="single"
              variant="selecting"
              size="sm"
              value={viewTab}
              onValueChange={(value) => {
                if (value) setViewTab(value as InfraViewTab)
              }}
            >
              <ToggleGroupItem
                value="metrics"
                className="text-xs px-2 py-1 h-7"
              >
                Metrics
              </ToggleGroupItem>
              <ToggleGroupItem
                value="topology"
                className="text-xs px-2 py-1 h-7"
              >
                Topology
              </ToggleGroupItem>
              <ToggleGroupItem value="model" className="text-xs px-2 py-1 h-7">
                Model
              </ToggleGroupItem>
            </ToggleGroup>
            {viewTab === "metrics" && (
              <>
                <PaginationControls
                  currentPage={page}
                  totalPages={totalPages}
                  onPageChange={handlePageChange}
                />
                <IntervalPicker
                  value={intervalSeconds}
                  onChange={handleIntervalChange}
                />
                {timeRangeOverride && (
                  <span className="text-xs text-muted-foreground whitespace-nowrap">
                    {formatClockTimeAdaptive(
                      timeRangeOverride.start,
                      timeRangeOverride.end - timeRangeOverride.start,
                    )}{" "}
                    –{" "}
                    {formatClockTimeAdaptive(
                      timeRangeOverride.end,
                      timeRangeOverride.end - timeRangeOverride.start,
                    )}
                  </span>
                )}
                <button
                  onClick={handleLiveToggle}
                  className={`ml-1 inline-flex items-center rounded-lg px-2.5 h-7 text-xs font-medium transition-all select-none whitespace-nowrap ${
                    isLive
                      ? "bg-emerald-500 text-white hover:bg-emerald-600"
                      : "bg-accent text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                >
                  Live
                </button>
                <ToggleWithInput
                  label="Aggregate"
                  variant="selecting"
                  size="sm"
                  inputType="text"
                  enabled={aggregateEnabled}
                  onEnabledChange={setAggregateEnabled}
                  value={aggWindowInput}
                  onValueChange={setAggWindowInput}
                  onValueCommit={(value) => {
                    const parsed = parseAggWindow(value)
                    if (parsed !== null) {
                      setAggregateWindow(parsed)
                      setAggWindowInput(formatAggWindow(parsed))
                    } else {
                      setAggWindowInput(formatAggWindow(aggregateWindow))
                    }
                  }}
                  inputWidth="w-10"
                />
              </>
            )}
          </div>
        </div>
      </header>

      {/* Content */}
      {viewTab === "metrics" && (
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

            <SystemMetricsCharts
              gpuMetrics={aggregatedGpuMetrics}
              cpuMetrics={aggregatedCpuMetrics}
              vllmMetrics={aggregatedVllmMetrics}
              threadPoolMetrics={displayedThreadPoolData?.metrics}
              availableGpuMetrics={
                displayedGpuData?.available_metrics ?? stickyGpuMetricNames
              }
              availableCpuMetrics={
                displayedCpuData?.available_metrics ?? stickyCpuMetricNames
              }
              availableVllmMetrics={
                displayedVllmData?.available_metrics ?? stickyVllmMetricNames
              }
              availableThreadPoolMetrics={
                displayedThreadPoolData?.available_metrics ?? stickyThreadPoolMetricNames
              }
              availableThreadPools={
                displayedThreadPoolData?.available_pools ?? stickyThreadPoolNames
              }
              trainerGpus={trainerGpus}
              inferenceGpus={inferenceGpus}
              isLoading={isChartsLoading}
              isRefetching={isBackgroundRefetching}
              roleMode={roleMode}
              onRoleModeChange={setRoleMode}
              systemMetricsOpen={systemMetricsOpen}
              onSystemMetricsOpenChange={setSystemMetricsOpen}
              cpuMetricsOpen={cpuMetricsOpen}
              onCpuMetricsOpenChange={setCpuMetricsOpen}
              vllmMetricsOpen={vllmMetricsOpen}
              onVllmMetricsOpenChange={setVllmMetricsOpen}
              threadPoolMetricsOpen={threadPoolMetricsOpen}
              onThreadPoolMetricsOpenChange={setThreadPoolMetricsOpen}
              trainerSectionOpen={trainerSectionOpen}
              onTrainerSectionOpenChange={setTrainerSectionOpen}
              inferenceSectionOpen={inferenceSectionOpen}
              onInferenceSectionOpenChange={setInferenceSectionOpen}
              trainerNodeMode={trainerNodeMode}
              onTrainerNodeModeChange={setTrainerNodeMode}
              inferenceNodeMode={inferenceNodeMode}
              onInferenceNodeModeChange={setInferenceNodeMode}
              overrideTimeRange={timeRangeOverride}
              xOffset={xOffset}
            />
          </div>
        </div>
      )}

      {viewTab === "topology" && (
        <div className="flex-1 overflow-hidden">
          {error ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="text-sm text-destructive mb-2">
                  Failed to load topology data
                </div>
                <div className="text-xs text-muted-foreground">
                  {error instanceof Error ? error.message : "Unknown error"}
                </div>
              </div>
            </div>
          ) : !summaryData ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-sm text-muted-foreground animate-pulse">
                Loading topology…
              </div>
            </div>
          ) : !topology || topology.nodes.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="text-sm text-muted-foreground mb-2">
                  No topology data available
                </div>
                <div className="text-xs text-muted-foreground/70">
                  The selected run does not have setup information in its W&B
                  summary.
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full w-full">
              <TopologyViewer topology={topology} />
            </div>
          )}
        </div>
      )}

      {viewTab === "model" && (
        <div className="flex-1 overflow-hidden">
          {error ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="text-sm text-destructive mb-2">
                  Failed to load model data
                </div>
                <div className="text-xs text-muted-foreground">
                  {error instanceof Error ? error.message : "Unknown error"}
                </div>
              </div>
            </div>
          ) : !summaryData ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-sm text-muted-foreground animate-pulse">
                Loading model architecture…
              </div>
            </div>
          ) : !summaryData.summary?.model_architecture ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="text-sm text-muted-foreground mb-2">
                  No model architecture available
                </div>
                <div className="text-xs text-muted-foreground/70">
                  The selected run does not have model_architecture in its W&B
                  summary.
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full w-full">
              <ModelArchitectureViewer
                modelRepr={summaryData.summary.model_architecture as string}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
