
import { useMemo, useRef, useState } from "react"
import { formatDurationHms } from "@/lib/format"
import { useAtom, useAtomValue } from "jotai"
import { ChevronDown, X } from "lucide-react"
import {
  Card,
  CardContent,
} from "@/components/ui/card"
import { NoRunSelectedState } from "@/components/no-run-selected-state"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { CombinedTimelineChart, GroupSampleTimeline, TrainerBreakdownContent } from "@/components/combined-timeline-chart"
import { PaginationControls } from "@/components/pagination-controls"
import {
  selectedRunPathAtom,
  selectedInferenceRequestAtom,
  selectedTrainerEventAtom,
  timelinePageAtom,
  timelineIntervalAtom,
  inferenceHighlightDiscardedAtom,
  inferenceShowComputeRewardAtom,
  darkModeAtom,
} from "@/lib/atoms"
import {
  formatTrainerEventTitle,
  getTrainerEventColor,
  INFERENCE_REQUEST_CANCELED_COLOR,
  INFERENCE_REQUEST_CANCELED_COLOR_DARK,
  INFERENCE_REQUEST_DISCARDED_COLOR,
  INFERENCE_REQUEST_DISCARDED_COLOR_DARK,
} from "@/lib/constants"
import {
  useGpuMetricsForTrainerRanks,
  useRolloutEventsByGroup,
  useTrainerBreakdownEvents,
  useTimelinePaginated,
  useInflightGenerations,
  useRunSummary,
  useSampleStatuses,
  useSampleDetails,
} from "@/hooks/use-run-data"
import type { GpuMetric, RolloutEvent, TrainerEvent, InflightSnapshot } from "@/lib/types"
import { parseSetupJson, asObject, asNumber } from "@/components/topology-viewer"

function parseDeviceList(value: unknown): number[] {
  if (value && typeof value === "object" && "value" in value) {
    return parseDeviceList((value as { value: unknown }).value)
  }
  if (Array.isArray(value)) {
    const numbers = value
      .map((item) => (typeof item === "string" ? item.trim() : item))
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item))
    return Array.from(new Set(numbers))
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return [value]
  }
  if (typeof value === "string") {
    const matches = value.match(/-?\d+/g) ?? []
    const numbers = matches
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item))
    return Array.from(new Set(numbers))
  }
  return []
}

const INTERVAL_PRESETS = [1, 5, 10, 30, 60, 120, 300]

const formatDuration = formatDurationHms
const formatHms = formatDurationHms

type TrainerGpuMetricsQueryData = {
  metrics?: GpuMetric[]
  available_metrics?: string[]
}

const DEFAULT_TRAINER_GPU_METRICS = ["torch_allocated_gb"]
const EMPTY_TRAINER_GPU_METRIC_SENTINEL = "__timeline_noop_metric__"

type TrainerRankInfo = {
  node_id: number | null
  local_rank: number | null
}


function getSetupObject(
  summary: Record<string, unknown> | undefined
): Record<string, unknown> | null {
  return parseSetupJson(summary?.setup)
}

function extractTrainerRankInfoFromSetup(
  summary: Record<string, unknown> | undefined
): Record<number, TrainerRankInfo> {
  const setup = getSetupObject(summary)
  const rankInfo: Record<number, TrainerRankInfo> = {}

  // v6.0: setup.trainer.nodes[].gpus[]
  const trainer = asObject(setup?.trainer)
  const trainerNodes = Array.isArray(trainer?.nodes) ? trainer.nodes : []
  for (const nodeEntry of trainerNodes) {
    const node = asObject(nodeEntry)
    if (!node) continue
    const nodeId = asNumber(node.node_id)
    const gpus = Array.isArray(node.gpus) ? node.gpus : []
    for (const gpuEntry of gpus) {
      const gpu = asObject(gpuEntry)
      if (!gpu) continue
      const rank = asNumber(gpu.rank)
      if (rank === null) continue
      rankInfo[rank] = {
        node_id: nodeId,
        local_rank: asNumber(gpu.local_rank),
      }
    }
  }

  return rankInfo
}

function extractNumInferenceServersFromSetup(
  summary: Record<string, unknown> | undefined
): number | undefined {
  const setup = getSetupObject(summary)
  // v6.0: setup.inference.num_servers
  const inference = asObject(setup?.inference)
  const value = asNumber(inference?.num_servers)
  return value ?? undefined
}

function extractInferenceServerNodeMapFromSetup(
  summary: Record<string, unknown> | undefined
): Record<number, number | null> {
  const setup = getSetupObject(summary)
  const map: Record<number, number | null> = {}

  // v6.0: setup.inference.nodes[].gpus[] — group by server_idx
  const inference = asObject(setup?.inference)
  const inferenceNodes = Array.isArray(inference?.nodes) ? inference.nodes : []
  for (const nodeEntry of inferenceNodes) {
    const node = asObject(nodeEntry)
    if (!node) continue
    const nodeId = asNumber(node.node_id)
    const gpus = Array.isArray(node.gpus) ? node.gpus : []
    for (const gpuEntry of gpus) {
      const gpu = asObject(gpuEntry)
      if (!gpu) continue
      const serverIdx = asNumber(gpu.server_idx)
      if (serverIdx === null) continue
      // Map server_idx → node_id (first occurrence wins; all GPUs of the same
      // server on this node share the same node_id)
      if (!(serverIdx in map)) {
        map[serverIdx] = nodeId
      }
    }
  }

  return map
}

function extractNumNodesFromSetup(
  summary: Record<string, unknown> | undefined
): number | undefined {
  const setup = getSetupObject(summary)
  // v6.0: setup.cluster.nodes[] — count from array length
  const cluster = asObject(setup?.cluster)
  const clusterNodes = Array.isArray(cluster?.nodes) ? cluster.nodes : []
  return clusterNodes.length > 0 ? clusterNodes.length : undefined
}

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
  if (compoundMatch && (compoundMatch[1] || compoundMatch[2] || compoundMatch[3])) {
    const h = parseInt(compoundMatch[1] || "0", 10)
    const m = parseInt(compoundMatch[2] || "0", 10)
    const s = parseInt(compoundMatch[3] || "0", 10)
    const total = h * 3600 + m * 60 + s
    return total > 0 ? total : null
  }

  // Plain number → default to seconds
  const num = parseFloat(trimmed)
  if (Number.isFinite(num) && num > 0) return Math.round(num)

  return null
}

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
        {/* Custom input */}
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
        {/* Preset options */}
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

export default function TimelinePage() {
  const selectedRunPath = useAtomValue(selectedRunPathAtom)

  // Pagination and interval (persisted in atoms)
  const [page, setPage] = useAtom(timelinePageAtom)
  const [intervalSeconds, setIntervalSeconds] = useAtom(timelineIntervalAtom)

  // Always poll when on this page - ensures data stays fresh
  const shouldPoll = true

  const {
    data: timelineData,
    isFetching,
    isPlaceholderData,
    error,
  } = useTimelinePaginated(
    selectedRunPath || "",
    page,
    intervalSeconds,
    !!selectedRunPath,
    shouldPoll
  )

  const { data: inflightData } = useInflightGenerations(
    selectedRunPath || "",
    !!selectedRunPath,
    shouldPoll
  )

  const { data: summaryData } = useRunSummary(
    selectedRunPath || "",
    !!selectedRunPath,
    shouldPoll
  )

  const trainerSystemGpuIndicesFromConfig = useMemo(() => {
    const configValue = summaryData?.config?.trainer_devices
    return parseDeviceList(configValue)
  }, [summaryData?.config?.trainer_devices])

  const intervalStart = timelineData?.interval_start ?? null
  const intervalEnd = timelineData?.interval_end ?? null
  const isTimelineTransitionLoading =
    isFetching && (isPlaceholderData || !timelineData)
  const [selectedTrainerGpuMetrics, setSelectedTrainerGpuMetrics] = useState<
    string[]
  >(() => [...DEFAULT_TRAINER_GPU_METRICS])

  const trainerGpuMetricNamesForQuery = useMemo<string[]>(() => {
    const normalized = Array.from(
      new Set(
        selectedTrainerGpuMetrics.filter(
          (metricName) =>
            typeof metricName === "string" && metricName.trim().length > 0
        )
      )
    ).sort((a, b) => a.localeCompare(b))

    // Keep querying metadata (available_metrics) while skipping metric rows.
    return normalized.length > 0
      ? normalized
      : [EMPTY_TRAINER_GPU_METRIC_SENTINEL]
  }, [selectedTrainerGpuMetrics])

  const trainerRankInfoFromSetup = useMemo(
    () => extractTrainerRankInfoFromSetup(summaryData?.summary),
    [summaryData?.summary]
  )

  const trainerRankInfoFromEvents = useMemo<
    Record<number, TrainerRankInfo>
  >(() => {
    const map: Record<number, TrainerRankInfo> = {}
    const events = timelineData?.trainer_events ?? []
    for (const event of events) {
      const rank = event.rank
      if (!Number.isFinite(rank)) continue
      const existing = map[rank] ?? { node_id: null, local_rank: null }
      const nextNodeId =
        existing.node_id !== null
          ? existing.node_id
          : typeof event.node_id === "number" && Number.isFinite(event.node_id)
            ? event.node_id
            : null
      const nextLocalRank =
        existing.local_rank !== null
          ? existing.local_rank
          : typeof event.local_rank === "number" &&
              Number.isFinite(event.local_rank)
            ? event.local_rank
            : null
      map[rank] = {
        node_id: nextNodeId,
        local_rank: nextLocalRank,
      }
    }
    return map
  }, [timelineData?.trainer_events])

  const trainerRankInfoByRank = useMemo<Record<number, TrainerRankInfo>>(() => {
    const merged: Record<number, TrainerRankInfo> = {}
    const allRanks = new Set<number>([
      ...Object.keys(trainerRankInfoFromSetup).map(Number),
      ...Object.keys(trainerRankInfoFromEvents).map(Number),
    ])
    allRanks.forEach((rank) => {
      const setupInfo = trainerRankInfoFromSetup[rank]
      const eventInfo = trainerRankInfoFromEvents[rank]
      merged[rank] = {
        node_id: setupInfo?.node_id ?? eventInfo?.node_id ?? null,
        local_rank: setupInfo?.local_rank ?? eventInfo?.local_rank ?? null,
      }
    })
    return merged
  }, [trainerRankInfoFromSetup, trainerRankInfoFromEvents])

  // Inference server → node_id mapping
  const inferenceServerNodeMapFromSetup = useMemo(
    () => extractInferenceServerNodeMapFromSetup(summaryData?.summary),
    [summaryData?.summary]
  )

  // In the new model, rollout_events don't carry node_id — server-to-node
  // mapping comes from setup only. Keep an empty fallback.
  const inferenceServerNodeMapFromEvents = useMemo<Record<number, number | null>>(() => {
    return {}
  }, [])

  const inferenceServerNodeMap = useMemo<Record<number, number | null>>(() => {
    const merged: Record<number, number | null> = {}
    const allServers = new Set<number>([
      ...Object.keys(inferenceServerNodeMapFromSetup).map(Number),
      ...Object.keys(inferenceServerNodeMapFromEvents).map(Number),
    ])
    allServers.forEach((server) => {
      merged[server] = inferenceServerNodeMapFromSetup[server] ?? inferenceServerNodeMapFromEvents[server] ?? null
    })
    return merged
  }, [inferenceServerNodeMapFromSetup, inferenceServerNodeMapFromEvents])

  const trainerRanksForMetrics = useMemo(() => {
    const fromEvents = (timelineData?.trainer_events ?? [])
      .map((event) => event.rank)
      .filter((rank) => Number.isFinite(rank))
    const fromRankInfo = Object.keys(trainerRankInfoByRank).map(Number)
    return Array.from(new Set([...fromEvents, ...fromRankInfo])).sort(
      (a, b) => a - b
    )
  }, [timelineData?.trainer_events, trainerRankInfoByRank])

  const trainerRankFilters = useMemo(() => {
    return trainerRanksForMetrics.map((rank) => ({
      rank,
      node_id: trainerRankInfoByRank[rank]?.node_id ?? null,
      local_rank: trainerRankInfoByRank[rank]?.local_rank ?? null,
    }))
  }, [trainerRanksForMetrics, trainerRankInfoByRank])

  const trainerSystemGpuIndices = useMemo(() => {
    const byRank: number[] = []
    let hasAnyLocalRank = false
    trainerRanksForMetrics.forEach((rank) => {
      const localRank = trainerRankInfoByRank[rank]?.local_rank
      if (localRank !== null && localRank !== undefined) {
        byRank[rank] = localRank
        hasAnyLocalRank = true
      }
    })
    if (hasAnyLocalRank) return byRank
    return trainerSystemGpuIndicesFromConfig
  }, [
    trainerRankInfoByRank,
    trainerRanksForMetrics,
    trainerSystemGpuIndicesFromConfig,
  ])

  const trainerGpuMetricsQueries = useGpuMetricsForTrainerRanks(
    selectedRunPath || "",
    trainerGpuMetricNamesForQuery,
    trainerRankFilters,
    intervalStart,
    intervalEnd,
    !!selectedRunPath &&
      intervalStart !== null &&
      intervalEnd !== null &&
      trainerRankFilters.length > 0,
    shouldPoll
  )

  const trainerGpuMetricsByRank = useMemo<Record<number, GpuMetric[]>>(() => {
    const byRank: Record<number, GpuMetric[]> = {}
    trainerRankFilters.forEach((filter, idx) => {
      const queryData = trainerGpuMetricsQueries[idx]
        ?.data as TrainerGpuMetricsQueryData | undefined
      byRank[filter.rank] = queryData?.metrics ?? []
    })
    return byRank
  }, [trainerRankFilters, trainerGpuMetricsQueries])

  const trainerAvailableGpuMetrics = useMemo<string[]>(() => {
    const fromApi: string[] = []
    trainerGpuMetricsQueries.forEach((query) => {
      const queryData = query.data as TrainerGpuMetricsQueryData | undefined
      if (!queryData?.available_metrics) return
      queryData.available_metrics.forEach((metricName) => {
        if (typeof metricName === "string") fromApi.push(metricName)
      })
    })

    const dedupedApiMetrics = Array.from(new Set(fromApi))
    if (dedupedApiMetrics.length > 0) return dedupedApiMetrics

    const fromData = new Set<string>()
    trainerGpuMetricsQueries.forEach((query) => {
      const queryData = query.data as TrainerGpuMetricsQueryData | undefined
      queryData?.metrics?.forEach((metric) => {
        if (typeof metric.metric_name === "string") {
          fromData.add(metric.metric_name)
        }
      })
    })

    return Array.from(fromData).sort((a, b) => a.localeCompare(b))
  }, [trainerGpuMetricsQueries])

  const trainerGpuMetricsIsLoading = trainerGpuMetricsQueries.some(
    (query) => query.isFetching
  )

  // No run selected
  if (!selectedRunPath) {
    return <NoRunSelectedState description="Select a run from the sidebar to view the timeline." />
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header - same height as /metrics */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="py-2.5 px-4">
          <div className="flex items-center gap-2 h-7">
            <PaginationControls
              currentPage={page}
              totalPages={timelineData?.total_pages ?? 1}
              onPageChange={setPage}
            />
            <IntervalPicker
              value={intervalSeconds}
              onChange={(newInterval) => {
                // Keep roughly the same time position when changing interval
                const newPage = Math.floor((page * intervalSeconds) / newInterval)
                setIntervalSeconds(newInterval)
                setPage(newPage)
              }}
            />
            {intervalStart !== null && intervalEnd !== null && timelineData?.global_min_time != null && (
              <span className="text-xs text-muted-foreground whitespace-nowrap">
                {formatHms(intervalStart - timelineData.global_min_time)} – {formatHms(intervalEnd - timelineData.global_min_time)}
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto">
          {error && (
            <Card className="border-destructive mb-6">
              <CardContent className="pt-6">
                <p className="text-destructive">
                  Error: {error instanceof Error ? error.message : "Unknown error"}
                </p>
              </CardContent>
            </Card>
          )}

          <CombinedTimelineChart
            orchestratorData={timelineData?.orchestrator_events}
            trainerData={timelineData?.trainer_events}
            rolloutEvents={timelineData?.rollout_events}
            infraEvents={timelineData?.infra_events}
            inflightSnapshot={inflightData}
            isLoading={isTimelineTransitionLoading}
            intervalStart={timelineData?.interval_start ?? 0}
            intervalEnd={timelineData?.interval_end ?? intervalSeconds}
            trainerGpuMetricsByRank={trainerGpuMetricsByRank}
            trainerGpuMetricsIsLoading={trainerGpuMetricsIsLoading}
            trainerAvailableGpuMetrics={trainerAvailableGpuMetrics}
            selectedTrainerGpuMetrics={selectedTrainerGpuMetrics}
            onSelectedTrainerGpuMetricsChange={setSelectedTrainerGpuMetrics}
            trainerSystemGpuIndices={trainerSystemGpuIndices}
            trainerRankInfoByRank={trainerRankInfoByRank}
            totalSetupNodes={extractNumNodesFromSetup(summaryData?.summary)}
            inferenceServerNodeMap={inferenceServerNodeMap}
            numInferenceServers={
              extractNumInferenceServersFromSetup(summaryData?.summary) ??
              (summaryData?.config?.num_inference_servers as number | undefined)
            }
            maxConcurrentPromptsPerServer={
              summaryData?.config?.max_concurrent_prompts_per_server as number | undefined
            }
            groupSize={summaryData?.config?.group_size as number | undefined}
            freeLaneAfterGeneration={
              summaryData?.config?.free_lane_after_generation as boolean | undefined
            }
          />
        </div>
      </div>

      {/* Footer - selected inference request / trainer breakdown */}
      <TimelineFooter
        runPath={selectedRunPath}
        trainerEvents={timelineData?.trainer_events}
        totalSetupNodes={extractNumNodesFromSetup(summaryData?.summary)}
      />
    </div>
  )
}

type FooterTab = "inference" | "trainer"

function TimelineFooter({
  runPath,
  trainerEvents,
  totalSetupNodes,
}: {
  runPath: string | null
  trainerEvents?: TrainerEvent[]
  totalSetupNodes?: number
}) {
  const darkMode = useAtomValue(darkModeAtom)
  const { data: inflightData } = useInflightGenerations(
    runPath || "",
    !!runPath,
    true,
  )
  const [selectedRequest, setSelectedRequest] = useAtom(selectedInferenceRequestAtom)
  const [selectedTrainerEvent, setSelectedTrainerEvent] = useAtom(selectedTrainerEventAtom)
  const [activeTab, setActiveTab] = useState<FooterTab>("inference")
  const [prevSelectedRequest, setPrevSelectedRequest] = useState(selectedRequest)
  const [prevSelectedTrainerEvent, setPrevSelectedTrainerEvent] = useState(selectedTrainerEvent)

  // Auto-switch tab when a selection changes
  if (selectedRequest !== prevSelectedRequest) {
    setPrevSelectedRequest(selectedRequest)
    if (selectedRequest) setActiveTab("inference")
  }
  if (selectedTrainerEvent !== prevSelectedTrainerEvent) {
    setPrevSelectedTrainerEvent(selectedTrainerEvent)
    if (selectedTrainerEvent) setActiveTab("trainer")
  }

  // ---- Inference data ----
  const { data: groupEventsData } = useRolloutEventsByGroup(
    runPath ?? "",
    selectedRequest?.groupId ?? null,
    !!selectedRequest && !!runPath
  )

  // Pivot rollout events from the group into spans, then group by sample_id
  const groupEventsBySampleId = useMemo(() => {
    if (!selectedRequest || !groupEventsData?.events) return null
    // Pivot start/end rows into spans
    const starts = new Map<string, RolloutEvent>()
    const spans: Array<{
      event_type: string
      start_time: number
      end_time: number
      sample_id: number
      group_id: number
      agent_id: number
      generation_idx: number
      tool_call_idx: number
      server_id: number
    }> = []
    for (const e of groupEventsData.events as RolloutEvent[]) {
      const key = `${e.event_type}:${e.sample_id ?? -1}:${e.generation_idx ?? -1}:${e.tool_call_idx ?? -1}`
      if (e.phase === "start") {
        starts.set(key, e)
      } else if (e.phase === "end") {
        const start = starts.get(key)
        if (start) {
          spans.push({
            event_type: e.event_type,
            start_time: start.timestamp,
            end_time: e.timestamp,
            sample_id: e.sample_id ?? -1,
            group_id: start.group_id ?? e.group_id ?? -1,
            agent_id: e.agent_id ?? 0,
            generation_idx: e.generation_idx ?? -1,
            tool_call_idx: e.tool_call_idx ?? -1,
            server_id: start.server_id ?? e.server_id ?? -1,
          })
          starts.delete(key)
        }
      }
    }
    // Only keep generation spans for the group timeline
    const generationSpans = spans.filter((s) => s.event_type === "generation")
    const eventsBySample: Record<number, typeof generationSpans> = {}
    for (const span of generationSpans) {
      if (span.sample_id >= 0) {
        if (!eventsBySample[span.sample_id]) {
          eventsBySample[span.sample_id] = []
        }
        eventsBySample[span.sample_id].push(span)
      }
    }
    return eventsBySample
  }, [selectedRequest, groupEventsData])

  // Build per-sample timing data from env_response and reward spans
  const groupTimingData = useMemo(() => {
    if (!groupEventsData?.events) return null
    // Pivot to get env_response and reward spans
    const starts = new Map<string, RolloutEvent>()
    const envSpans: Array<{ sample_id: number; generation_idx: number; start_time: number; end_time: number }> = []
    const rewardSpans: Array<{ sample_id: number; start_time: number; end_time: number }> = []
    for (const e of groupEventsData.events as RolloutEvent[]) {
      const key = `${e.event_type}:${e.sample_id ?? -1}:${e.generation_idx ?? -1}:${e.tool_call_idx ?? -1}`
      if (e.phase === "start") {
        starts.set(key, e)
      } else if (e.phase === "end") {
        const start = starts.get(key)
        if (start) {
          if (e.event_type === "env_response") {
            envSpans.push({
              sample_id: e.sample_id ?? -1,
              generation_idx: e.generation_idx ?? -1,
              start_time: start.timestamp,
              end_time: e.timestamp,
            })
          } else if (e.event_type === "reward") {
            rewardSpans.push({
              sample_id: e.sample_id ?? -1,
              start_time: start.timestamp,
              end_time: e.timestamp,
            })
          }
          starts.delete(key)
        }
      }
    }
    // Index env times by sample_id → sorted list of {turn_order, time}
    const envBySample: Record<number, Array<{ turn_order: number; time: number }>> = {}
    for (const es of envSpans) {
      if (!envBySample[es.sample_id]) envBySample[es.sample_id] = []
      envBySample[es.sample_id].push({
        turn_order: es.generation_idx,
        time: es.end_time - es.start_time,
      })
    }
    for (const arr of Object.values(envBySample)) {
      arr.sort((a, b) => a.turn_order - b.turn_order)
    }
    // Index reward times by sample_id
    const rewardBySample: Record<number, number> = {}
    for (const rs of rewardSpans) {
      rewardBySample[rs.sample_id] = rs.end_time - rs.start_time
    }
    return { envBySample, rewardBySample }
  }, [groupEventsData])

  const groupTimeBounds = useMemo(() => {
    if (!groupEventsBySampleId) return null
    let minTime = Infinity
    let maxTime = -Infinity
    for (const [sampleIdStr, events] of Object.entries(groupEventsBySampleId)) {
      for (const event of events) {
        minTime = Math.min(minTime, event.start_time)
        maxTime = Math.max(maxTime, event.end_time)
      }
      // Account for env response and compute reward durations from separate timing props
      if (groupTimingData) {
        const sampleId = Number(sampleIdStr)
        const sortedEvents = [...events].sort((a, b) => a.start_time - b.start_time)
        const envTimes = groupTimingData.envBySample[sampleId] ?? []
        let cursor = -Infinity
        for (let i = 0; i < sortedEvents.length; i++) {
          cursor = sortedEvents[i].end_time
          if (i < envTimes.length && envTimes[i].time > 0) {
            cursor += envTimes[i].time
          }
        }
        const rewardTime = groupTimingData.rewardBySample[sampleId] ?? 0
        if (rewardTime > 0) cursor += rewardTime
        maxTime = Math.max(maxTime, cursor)
      }
    }
    // Account for inflight items extending to snapshot timestamp
    if (inflightData?.timestamp && selectedRequest) {
      const snapshotTime = inflightData.timestamp
      const hasInflightReward = inflightData.inflight_rewards?.some(
        (r) => r.sample_id === selectedRequest.sampleId
      )
      const hasInflightEnv = inflightData.inflight_env_responses?.some(
        (r) => r.sample_id === selectedRequest.sampleId
      )
      if (hasInflightReward || hasInflightEnv) {
        maxTime = Math.max(maxTime, snapshotTime)
      }
    }
    if (minTime === Infinity) return null
    return { start: minTime, end: maxTime, duration: maxTime - minTime }
  }, [groupEventsBySampleId, groupTimingData, inflightData, selectedRequest])

  // ---- Trainer data ----
  const { data: trainerBreakdownData } = useTrainerBreakdownEvents(
    runPath ?? "",
    selectedTrainerEvent?.eventType ?? null,
    selectedTrainerEvent?.rank ?? null,
    selectedTrainerEvent?.step ?? null,
    !!runPath && !!selectedTrainerEvent
  )

  const trainerEventsForBreakdown = useMemo(() => {
    if (trainerBreakdownData?.events && trainerBreakdownData.events.length > 0) {
      return trainerBreakdownData.events
    }
    return trainerEvents ?? []
  }, [trainerBreakdownData, trainerEvents])

  const eventsByRank = useMemo(() => {
    if (trainerEventsForBreakdown.length === 0) {
      return {} as Record<number, TrainerEvent[]>
    }
    const map: Record<number, TrainerEvent[]> = {}
    for (const event of trainerEventsForBreakdown) {
      if (!map[event.rank]) map[event.rank] = []
      map[event.rank].push(event)
    }
    return map
  }, [trainerEventsForBreakdown])

  const parentEvent = useMemo(() => {
    if (!selectedTrainerEvent) return null
    const rankEvents = eventsByRank[selectedTrainerEvent.rank] || []
    return rankEvents.find(
      (e) =>
        e.event_type === selectedTrainerEvent.eventType &&
        e.step === selectedTrainerEvent.step &&
        e.rank === selectedTrainerEvent.rank &&
        !e.event_type.includes("/")
    ) ?? null
  }, [selectedTrainerEvent, eventsByRank])

  const childEvents = useMemo(() => {
    if (!selectedTrainerEvent) return []
    const rankEvents = eventsByRank[selectedTrainerEvent.rank] || []
    return rankEvents.filter(
      (e) =>
        e.event_type.startsWith(selectedTrainerEvent.eventType + "/") &&
        e.step === selectedTrainerEvent.step
    )
  }, [selectedTrainerEvent, eventsByRank])

  // ---- Discard status for the selected group ----
  const highlightDiscarded = useAtomValue(inferenceHighlightDiscardedAtom)
  const showComputeReward = useAtomValue(inferenceShowComputeRewardAtom)
  const footerSamples = useMemo(() => {
    if (!selectedRequest || !groupEventsBySampleId) return []
    return Object.keys(groupEventsBySampleId).map(Number).map((s) => ({
      group_id: selectedRequest.groupId,
      sample_idx: s,
    }))
  }, [selectedRequest, groupEventsBySampleId])
  const { data: footerSampleStatuses } = useSampleStatuses(
    runPath ?? "",
    footerSamples,
    !!runPath && footerSamples.length > 0 && highlightDiscarded,
    true
  )
  const footerDiscardStatusReady =
    !highlightDiscarded || footerSamples.length === 0 || !!footerSampleStatuses
  const isDiscardedGroup = useMemo(() => {
    if (!highlightDiscarded || !footerSampleStatuses?.statuses) return false
    return footerSampleStatuses.statuses.some(
      (s) => s.kind === "rollouts_discarded"
    )
  }, [highlightDiscarded, footerSampleStatuses])

  // RolloutSpan doesn't carry is_canceled — cancellation is not tracked
  const isCanceledGroup = false

  // Check if the selected group is inflight or pending status (not yet kept/discarded)
  const isInflightOrPendingGroup = useMemo(() => {
    if (isCanceledGroup || isDiscardedGroup) return false
    if (!highlightDiscarded || selectedRequest?.isEval) return false
    // If status not ready yet, it's pending
    if (!footerDiscardStatusReady) return true
    // If status is ready but no statuses found for this group, it's pending
    if (!footerSampleStatuses?.statuses?.length) return true
    // If none are categorized as kept ("rollouts"), it's pending
    return !footerSampleStatuses.statuses.some((s) => s.kind === "rollouts")
  }, [isCanceledGroup, isDiscardedGroup, highlightDiscarded, footerDiscardStatusReady, footerSampleStatuses])

  // ---- Sample details for discard reason + advantage summary ----
  const { data: sampleDetails } = useSampleDetails(
    runPath ?? "",
    selectedRequest?.groupId ?? null,
    selectedRequest?.sampleId ?? null,
    !!runPath && !!selectedRequest,
    selectedRequest?.isEval
  )

  const discardReason = useMemo(() => {
    if (!sampleDetails || sampleDetails.kind !== "rollouts_discarded") return null
    return sampleDetails.discard_reason
  }, [sampleDetails])

  const rewardSummary = useMemo(() => {
    if (!sampleDetails || sampleDetails.kind === null) return null
    const samples = sampleDetails.samples_data
    if (!samples || samples.length === 0) return null
    const rewards = samples
      .map((s) => s.reward)
      .filter((r): r is number => r !== null)
    if (rewards.length === 0) return null
    if (rewards.every((r) => r === 0)) return "all_zero"
    if (rewards.every((r) => r > 0)) return "all_positive"
    return null
  }, [sampleDetails])

  const formatDiscardReason = (reason: string) => {
    switch (reason) {
      case "max_async":
        return "Max Async"
      case "zero_advantage":
        return "Zero Advantage"
      default:
        return reason
          .split("_")
          .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
          .join(" ")
    }
  }

  const hasInference = !!selectedRequest
  const hasTrainer = !!selectedTrainerEvent && childEvents.length > 0 && !!parentEvent

  // When the active tab's data disappears, auto-resolve to the other tab if available
  const resolvedTab: FooterTab =
    activeTab === "inference" && !hasInference && hasTrainer
      ? "trainer"
      : activeTab === "trainer" && !hasTrainer && hasInference
        ? "inference"
        : activeTab

  // Nothing selected at all
  if (!hasInference && !hasTrainer) return null

  const handleClose = () => {
    if (resolvedTab === "inference") {
      setSelectedRequest(null)
      // Switch to trainer tab if it has data
      if (hasTrainer) setActiveTab("trainer")
    } else {
      setSelectedTrainerEvent(null)
      // Switch to inference tab if it has data
      if (hasInference) setActiveTab("inference")
    }
  }

  const trainerTitleInfo = parentEvent
    ? formatTrainerEventTitle(parentEvent.event_type, parentEvent.step, parentEvent.microbatch, parentEvent.minibatch)
    : null
  const trainerColor = parentEvent
    ? getTrainerEventColor(parentEvent.event_type)
    : undefined

  const showToggle = hasInference && hasTrainer

  return (
    <footer className="border-t bg-background shrink-0 h-[28vh] flex flex-col">
      {/* Footer header */}
      <div className="flex items-center gap-3 px-4 py-2 shrink-0">
        {/* Tab toggle - only when both are selected */}
        {showToggle && (
          <div className="flex items-center bg-muted rounded-md p-0.5 gap-0.5">
            <button
              onClick={() => setActiveTab("inference")}
              className={`text-[11px] font-medium px-2.5 py-1 rounded transition-colors ${
                resolvedTab === "inference"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Inference
            </button>
            <button
              onClick={() => setActiveTab("trainer")}
              className={`text-[11px] font-medium px-2.5 py-1 rounded transition-colors ${
                resolvedTab === "trainer"
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Trainer
            </button>
          </div>
        )}

        {/* Contextual info */}
        {resolvedTab === "inference" && selectedRequest && (
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <div
                className="w-2.5 h-2.5 rounded-sm"
                style={{ backgroundColor: isCanceledGroup ? (darkMode ? INFERENCE_REQUEST_CANCELED_COLOR_DARK : "#adadad") : isDiscardedGroup ? (darkMode ? "#9ca3af" : "#6b7280") : isInflightOrPendingGroup ? "rgba(161, 98, 7, 0.9)" : selectedRequest.isEval ? "#047857" : "#075985" }}
              />
              <span className="text-xs font-medium">Sample {selectedRequest.sampleId}</span>
            </div>
            <span className="text-muted-foreground">/</span>
            <div className="flex items-center gap-1.5">
              <div
                className={`w-2.5 h-2.5 rounded-sm${!footerDiscardStatusReady ? " animate-pulse bg-muted" : ""}`}
                style={footerDiscardStatusReady ? { backgroundColor: isCanceledGroup ? (darkMode ? INFERENCE_REQUEST_CANCELED_COLOR_DARK : INFERENCE_REQUEST_CANCELED_COLOR) : isDiscardedGroup ? (darkMode ? INFERENCE_REQUEST_DISCARDED_COLOR_DARK : INFERENCE_REQUEST_DISCARDED_COLOR) : isInflightOrPendingGroup ? "rgba(202, 138, 4, 0.8)" : selectedRequest.isEval ? "#10b981" : "#0369a1" } : undefined}
              />
              <span className="text-xs font-medium">Group {selectedRequest.groupId}</span>
            </div>
            {isCanceledGroup && (
              <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-muted text-muted-foreground border border-border">
                Canceled due to async policy
              </span>
            )}
            {isInflightOrPendingGroup && !isCanceledGroup && (
              <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-yellow-100 text-yellow-700 border border-yellow-200">
                Waiting for Group
              </span>
            )}
            {discardReason && (
              <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-red-100 text-red-700 border border-red-200">
                {formatDiscardReason(discardReason)}
              </span>
            )}
            {discardReason === "zero_advantage" && rewardSummary === "all_zero" && (
              <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 border border-amber-200">
                All Rewards = 0
              </span>
            )}
            {discardReason === "zero_advantage" && rewardSummary === "all_positive" && (
              <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 border border-emerald-200">
                All Rewards &gt; 0
              </span>
            )}
          </div>
        )}

        {resolvedTab === "trainer" && trainerTitleInfo && parentEvent && (
          <div className="flex items-center gap-2">
            <div
              className="w-2.5 h-2.5 rounded-sm"
              style={{ backgroundColor: trainerColor }}
            />
            <span className="text-xs font-medium">
              {trainerTitleInfo.primary} Breakdown
            </span>
            <span className="text-xs text-muted-foreground">
              GPU {parentEvent.rank}
            </span>
            <span className="text-xs text-muted-foreground">
              (Step {parentEvent.step})
            </span>
          </div>
        )}

        {/* Close button */}
        <button
          onClick={handleClose}
          className="ml-auto text-muted-foreground hover:text-foreground transition-colors p-1 rounded hover:bg-muted"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="overflow-auto flex-1 min-h-0">
        {resolvedTab === "inference" && selectedRequest && groupEventsBySampleId && groupTimeBounds && (
          <GroupSampleTimeline
            eventsBySampleId={groupEventsBySampleId}
            selectedSampleId={selectedRequest.sampleId}
            timeBounds={groupTimeBounds}
            groupId={selectedRequest.groupId}
            runPath={runPath ?? ""}
            envResponseTimesBySample={groupTimingData?.envBySample}
            computeRewardTimeBySample={showComputeReward ? groupTimingData?.rewardBySample : undefined}
            inflightSnapshot={inflightData}
            onSampleClick={(sampleId) => {
              setSelectedRequest({
                sampleId,
                groupId: selectedRequest.groupId,
                isEval: selectedRequest.isEval,
              })
            }}
            showNodeLabel={typeof totalSetupNodes === "number" && totalSetupNodes > 1}
            isEval={selectedRequest.isEval}
          />
        )}

        {resolvedTab === "trainer" && parentEvent && childEvents.length > 0 && (
          <TrainerBreakdownContent
            parentEvent={parentEvent}
            childEvents={childEvents}
          />
        )}
      </div>
    </footer>
  )
}

