import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react"
import { createPortal } from "react-dom"
import { useAtom, useAtomValue } from "jotai"
import { ChevronDown, ChevronUp, Minus, Plus, Settings } from "lucide-react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Spinner } from "@/components/ui/spinner"
import { Toggle } from "@/components/ui/toggle"
import {
  GpuMetricChart,
  SYSTEM_METRIC_INFO,
} from "@/components/gpu-metric-chart"
import { SampleDetailsDialog } from "@/components/sample-details-dialog"
import { cn } from "@/lib/utils"
import { formatTimeShort } from "@/lib/format"
import {
  INFERENCE_REQUEST_COLOR,
  INFERENCE_REQUEST_EVAL_COLOR,
  INFERENCE_REQUEST_DISCARDED_COLOR,
  INFERENCE_REQUEST_DISCARDED_COLOR_DARK,
  INFERENCE_REQUEST_CANCELED_COLOR,
  INFERENCE_REQUEST_CANCELED_COLOR_DARK,
  IDLE_COLOR,
  IDLE_COLOR_DARK,
  getOrchestratorEventColor,
  getTrainerEventColor,
  getInferenceEventColor,
  getEventDisplayName,
  formatTrainerEventTitle,
} from "@/lib/constants"
import {
  inferenceHighlightDiscardedAtom,
  inferenceShowWeightUpdateAtom,
  inferenceShowComputeRewardAtom,
  inferenceSeparateComputeRewardAtom,
  inferenceLaneHeightAtom,
  inferenceLanePageAtom,
  inferenceMaxLanesAtom,
  inferenceServerPageAtom,
  trainerGpuPageAtom,
  selectedRunPathAtom,
  selectedTrainerEventAtom,
  selectedInferenceRequestAtom,
  darkModeAtom,
  type SelectedTrainerEvent,
  type SelectedInferenceRequest,
} from "@/lib/atoms"
import { useSampleStatuses } from "@/hooks/use-run-data"
import type {
  OrchestratorEvent,
  TrainerEvent,
  InferenceEvent,
  GpuMetric,
  InflightSnapshot,
} from "@/lib/types"

// Threshold for considering events as "close" (5ms in seconds for 60s interval)
const CLOSE_EVENTS_THRESHOLD = 0.005

// Compute a dynamic close-events threshold based on the interval duration.
// Formula: intervalDuration / 12000, rounded to a "nice" value (1, 2, 5 × 10^n).
function computeCloseEventsThreshold(intervalDuration: number): {
  thresholdSeconds: number
  displayLabel: string
} {
  const rawMs = intervalDuration * 2 // (intervalDuration * 1000) * 2

  if (rawMs <= 0) return { thresholdSeconds: 0.005, displayLabel: "5ms" }

  const magnitude = Math.pow(10, Math.floor(Math.log10(rawMs)))
  const normalized = rawMs / magnitude

  let niceValue: number
  if (normalized < 1.5) niceValue = 1
  else if (normalized < 3.5) niceValue = 2
  else if (normalized < 7.5) niceValue = 5
  else niceValue = 10

  const roundedMs = niceValue * magnitude
  // Format: show clean number without trailing zeroes
  const displayLabel =
    roundedMs >= 1 ? `${roundedMs}ms` : `${Number(roundedMs.toPrecision(2))}ms`

  return { thresholdSeconds: roundedMs / 1000, displayLabel }
}

/** Format a duration in ms to a human-readable string, picking the best unit. */
export function formatDuration(ms: number): string {
  const abs = Math.abs(ms)
  if (abs < 1) return `${ms.toFixed(2)}ms`
  if (abs < 10) return `${ms.toFixed(1)}ms`
  if (abs < 1000) return `${Math.round(ms)}ms`
  const s = ms / 1000
  if (abs < 60_000) return `${s.toFixed(2)}s`
  const m = s / 60
  if (abs < 3_600_000) return `${m.toFixed(1)}min`
  const h = m / 60
  return `${h.toFixed(1)}h`
}

/** Format a number with thousands delimiter (e.g. 2334 → "2,334"). */
function formatNumberWithDelimiter(n: number): string {
  return n.toLocaleString("en-US")
}

/** Build the common tooltip detail rows for an inference request event. */
function buildInferenceRequestDetails(
  event: InferenceEvent,
  durationMs: number,
  extra: Array<{ label: string; value: string }> = [],
): Array<{ label: string; value: string }> {
  return [
    { label: "Duration", value: formatDuration(durationMs) },
    ...(event.prompt_tokens !== undefined
      ? [
          {
            label: "Prompt tokens",
            value: formatNumberWithDelimiter(event.prompt_tokens),
          },
        ]
      : []),
    ...(event.rollout_tokens !== undefined
      ? [
          {
            label: "Rollout tokens",
            value: formatNumberWithDelimiter(event.rollout_tokens),
          },
        ]
      : []),
    ...(event.max_tokens !== undefined && event.max_tokens !== null
      ? [
          {
            label: "Max tokens",
            value: formatNumberWithDelimiter(event.max_tokens),
          },
        ]
      : []),
    ...extra,
    ...(event.queue_time !== undefined && event.queue_time !== null
      ? [
          {
            label: "Queue time",
            value: formatDuration(event.queue_time * 1000),
          },
        ]
      : []),
    ...(event.time_to_first_token !== undefined &&
    event.time_to_first_token !== null
      ? [
          {
            label: "TTFT",
            value: formatDuration(event.time_to_first_token * 1000),
          },
        ]
      : []),
    ...(event.prefill_time !== undefined && event.prefill_time !== null
      ? [
          {
            label: "Prefill time",
            value: formatDuration(event.prefill_time * 1000),
          },
        ]
      : []),
    ...(event.decode_time !== undefined && event.decode_time !== null
      ? [
          {
            label: "Decode time",
            value: formatDuration(event.decode_time * 1000),
          },
        ]
      : []),
    ...(event.inference_time !== undefined && event.inference_time !== null
      ? [
          {
            label: "Inference time",
            value: formatDuration(event.inference_time * 1000),
          },
        ]
      : []),
    ...(event.e2e_latency !== undefined && event.e2e_latency !== null
      ? [
          {
            label: "E2E latency",
            value: formatDuration(event.e2e_latency * 1000),
          },
        ]
      : []),
    ...(event.off_policy_steps !== undefined && event.off_policy_steps !== null
      ? [
          {
            label: "Off-policy steps",
            value: String(event.off_policy_steps),
          },
        ]
      : []),
  ]
}

/** Lighten a hex color by mixing it with white. amount=0 returns original, amount=1 returns white. */
function lightenColor(hex: string, amount: number): string {
  const h = hex.replace("#", "")
  const r = parseInt(h.substring(0, 2), 16)
  const g = parseInt(h.substring(2, 4), 16)
  const b = parseInt(h.substring(4, 6), 16)
  const lr = Math.round(r + (255 - r) * amount)
  const lg = Math.round(g + (255 - g) * amount)
  const lb = Math.round(b + (255 - b) * amount)
  return `#${lr.toString(16).padStart(2, "0")}${lg.toString(16).padStart(2, "0")}${lb.toString(16).padStart(2, "0")}`
}

/** Darken a hex color by mixing it with black. amount=0 returns original, amount=1 returns black. */
function darkenColor(hex: string, amount: number): string {
  const h = hex.replace("#", "")
  const r = parseInt(h.substring(0, 2), 16)
  const g = parseInt(h.substring(2, 4), 16)
  const b = parseInt(h.substring(4, 6), 16)
  const dr = Math.round(r * (1 - amount))
  const dg = Math.round(g * (1 - amount))
  const db = Math.round(b * (1 - amount))
  return `#${dr.toString(16).padStart(2, "0")}${dg.toString(16).padStart(2, "0")}${db.toString(16).padStart(2, "0")}`
}

/** Dim a color: lighten in light mode, darken in dark mode. */
function dimColor(hex: string, amount: number, darkMode: boolean): string {
  return darkMode ? darkenColor(hex, amount) : lightenColor(hex, amount)
}

const EMPTY_CHILD_EVENT_KEYS = new Set<string>()

function getGpuMetricDisplayName(metricName: string): string {
  const knownLabel = SYSTEM_METRIC_INFO[metricName]?.label
  if (knownLabel) return knownLabel
  return metricName
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

type SortedOrchestratorEvent = {
  event: OrchestratorEvent
  originalIndex: number
}

type CloseEventWindow = {
  start: number
  end: number
  sortedIndex: number
}

// Precompute close-event windows for each event in O(n) after sorting.
function buildCloseEventWindows(
  events: OrchestratorEvent[],
  threshold: number = CLOSE_EVENTS_THRESHOLD,
): {
  sortedEvents: SortedOrchestratorEvent[]
  windowByOriginalIndex: Map<number, CloseEventWindow>
} {
  if (events.length === 0) {
    return { sortedEvents: [], windowByOriginalIndex: new Map() }
  }

  const sortedEvents = events
    .map((event, originalIndex) => ({ event, originalIndex }))
    .sort((a, b) => a.event.timestamp - b.event.timestamp)

  const startIdx: number[] = new Array(sortedEvents.length)
  const endIdx: number[] = new Array(sortedEvents.length)

  let start = 0
  let end = 0

  for (let i = 0; i < sortedEvents.length; i++) {
    const t = sortedEvents[i].event.timestamp

    while (t - sortedEvents[start].event.timestamp > threshold) {
      start++
    }

    if (end < i) end = i
    while (
      end + 1 < sortedEvents.length &&
      sortedEvents[end + 1].event.timestamp - t <= threshold
    ) {
      end++
    }

    startIdx[i] = start
    endIdx[i] = end
  }

  const windowByOriginalIndex = new Map<number, CloseEventWindow>()
  sortedEvents.forEach((item, sortedIndex) => {
    windowByOriginalIndex.set(item.originalIndex, {
      start: startIdx[sortedIndex],
      end: endIdx[sortedIndex],
      sortedIndex,
    })
  })

  return { sortedEvents, windowByOriginalIndex }
}

interface CombinedTimelineChartProps {
  orchestratorData?: OrchestratorEvent[]
  trainerData?: TrainerEvent[]
  inferenceData?: InferenceEvent[]
  isLoading?: boolean
  intervalStart: number
  intervalEnd: number
  trainerGpuMetricsByGpuIndex?: Record<number, GpuMetric[]>
  trainerGpuMetricsByRank?: Record<number, GpuMetric[]>
  trainerGpuMetricsIsLoading?: boolean
  trainerAvailableGpuMetrics?: string[]
  selectedTrainerGpuMetrics?: string[]
  onSelectedTrainerGpuMetricsChange?: (metricNames: string[]) => void
  trainerSystemGpuIndices?: number[]
  trainerRankInfoByRank?: Record<
    number,
    { node_id: number | null; local_rank: number | null }
  >
  totalSetupNodes?: number
  // Config for inference visualization
  inferenceServerNodeMap?: Record<number, number | null>
  numInferenceServers?: number
  maxConcurrentPrompts?: number
  groupSize?: number
  freeLaneAfterGeneration?: boolean
  inflightSnapshot?: InflightSnapshot
}

export function CombinedTimelineChart({
  orchestratorData,
  trainerData,
  inferenceData,
  inflightSnapshot,
  isLoading,
  intervalStart,
  intervalEnd,
  trainerGpuMetricsByGpuIndex,
  trainerGpuMetricsByRank,
  trainerGpuMetricsIsLoading = false,
  trainerAvailableGpuMetrics = [],
  selectedTrainerGpuMetrics = [],
  onSelectedTrainerGpuMetricsChange,
  trainerSystemGpuIndices,
  trainerRankInfoByRank,
  totalSetupNodes,
  inferenceServerNodeMap,
  numInferenceServers = 0,
  maxConcurrentPrompts = 12,
  groupSize = 1,
  freeLaneAfterGeneration = false,
}: CombinedTimelineChartProps) {
  const highlightDiscarded = useAtomValue(inferenceHighlightDiscardedAtom)

  const trainerSystemGpuIndicesForDisplay = useMemo(() => {
    const configured = (trainerSystemGpuIndices ?? []).filter((gpuIdx) =>
      Number.isFinite(gpuIdx),
    )
    if (configured.length > 0) {
      return configured
    }
    const inferred = trainerGpuMetricsByGpuIndex
      ? Object.keys(trainerGpuMetricsByGpuIndex)
          .map(Number)
          .filter((gpuIdx) => Number.isFinite(gpuIdx))
          .sort((a, b) => a - b)
      : []
    return inferred
  }, [trainerSystemGpuIndices, trainerGpuMetricsByGpuIndex])

  const intervalDuration = intervalEnd - intervalStart

  const filteredOrchestratorData = useMemo(() => {
    if (!orchestratorData || orchestratorData.length === 0) return []
    return orchestratorData.filter(
      (event) =>
        event.timestamp >= intervalStart && event.timestamp < intervalEnd,
    )
  }, [orchestratorData, intervalStart, intervalEnd])

  const filteredTrainerData = useMemo(() => {
    if (!trainerData || trainerData.length === 0) return []
    return trainerData.filter(
      (event) =>
        event.start_time < intervalEnd && event.end_time > intervalStart,
    )
  }, [trainerData, intervalStart, intervalEnd])

  // Merge inflight generations into inference events as synthetic "in-progress" bars.
  // For each running generation in the snapshot, if there's no matching completed event
  // (same sample_id with a real end_time), create a synthetic event extending to the snapshot time.
  // If the DB event exists but was canceled without an end_time, the synthetic event
  // inherits the canceled status so it renders as canceled instead of inflight.
  const inferenceWithInflight = useMemo(() => {
    const events = inferenceData ?? []
    if (!inflightSnapshot?.running?.length || !inflightSnapshot.snapshot_time) {
      return events
    }
    // Map sample_ids to their DB events
    const eventBySampleId = new Map<number, InferenceEvent>()
    for (const e of events) {
      if (e.sample_id != null) {
        eventBySampleId.set(e.sample_id, e)
      }
    }
    // Create synthetic events for inflight generations
    const syntheticEvents: InferenceEvent[] = []
    for (const gen of inflightSnapshot.running) {
      const existing = eventBySampleId.get(gen.sample_id)
      if (existing && existing.end_time !== existing.start_time) {
        // Real completed event with proper end_time — skip
        continue
      }
      // Either no DB event, or a canceled event with no real end_time (end_time == start_time fallback).
      // Create a synthetic event extending to snapshot_time.
      syntheticEvents.push({
        event_type: "request",
        server: gen.server,
        start_time: gen.start_time,
        end_time: inflightSnapshot.snapshot_time!,
        prompt_tokens: gen.prompt_tokens,
        sample_id: gen.sample_id,
        group_id: gen.group_id,
        lane: gen.server_lane,
        is_eval: gen.is_eval,
        // Inherit canceled status from the DB event if present
        is_canceled: existing?.is_canceled,
        phase: existing?.is_canceled ? undefined : "inflight",
      })
    }
    return [...events, ...syntheticEvents]
  }, [inferenceData, inflightSnapshot])

  // Page-scoped inference events (for display / discard checks).
  // Use the effective end time (including env response + compute reward)
  // so that events whose reward trace extends into this page are kept.
  const filteredInferenceData = useMemo(() => {
    if (!inferenceWithInflight || inferenceWithInflight.length === 0) return []
    return inferenceWithInflight.filter((event) => {
      const effectiveEnd =
        event.end_time +
        (event.environment_response_time ?? 0) +
        (event.compute_reward_time ?? 0)
      return event.start_time < intervalEnd && effectiveEnd > intervalStart
    })
  }, [inferenceWithInflight, intervalStart, intervalEnd])

  const { eventsByRank, ranks } = useMemo(() => {
    const byRank: Record<number, TrainerEvent[]> = {}
    filteredTrainerData.forEach((event) => {
      if (!byRank[event.rank]) {
        byRank[event.rank] = []
      }
      byRank[event.rank].push(event)
    })

    const rankList = Object.keys(byRank)
      .map(Number)
      .sort((a, b) => a - b)

    return { eventsByRank: byRank, ranks: rankList }
  }, [filteredTrainerData])

  const { eventsByServer, servers } = useMemo(() => {
    const byServer: Record<number, InferenceEvent[]> = {}
    filteredInferenceData.forEach((event) => {
      if (!byServer[event.server]) {
        byServer[event.server] = []
      }
      byServer[event.server].push(event)
    })

    const serverList = Object.keys(byServer)
      .map(Number)
      .sort((a, b) => a - b)

    return { eventsByServer: byServer, servers: serverList }
  }, [filteredInferenceData])

  const effectiveNumServers =
    numInferenceServers > 0 ? numInferenceServers : Math.max(1, servers.length)
  const lanesPerServer = Math.max(
    1,
    Math.floor((maxConcurrentPrompts / effectiveNumServers) * groupSize),
  )

  return (
    <div
      className={`transition-opacity duration-200 overflow-visible ${
        isLoading && !highlightDiscarded ? "opacity-50" : ""
      }`}
    >
      <div className="space-y-6">
        {/* Inference Servers Timeline Section */}
        {(servers.length > 0 || numInferenceServers > 0) && (
          <InferenceSection
            servers={
              servers.length > 0
                ? servers
                : Array.from({ length: numInferenceServers }, (_, i) => i)
            }
            eventsByServer={eventsByServer}
            intervalStart={intervalStart}
            intervalDuration={intervalDuration}
            lanesPerServer={lanesPerServer}
            inferenceServerNodeMap={inferenceServerNodeMap}
            totalSetupNodes={totalSetupNodes}
            isLoading={isLoading}
            freeLaneAfterGeneration={freeLaneAfterGeneration}
          />
        )}

        {/* Orchestrator Timeline Section */}
        <OrchestratorSection
          events={filteredOrchestratorData}
          intervalStart={intervalStart}
          intervalDuration={intervalDuration}
          isLoading={isLoading}
        />

        {/* Trainer GPU Timelines Section */}
        <TrainerSection
          ranks={ranks}
          eventsByRank={eventsByRank}
          intervalStart={intervalStart}
          intervalDuration={intervalDuration}
          intervalEnd={intervalEnd}
          trainerGpuMetricsByGpuIndex={trainerGpuMetricsByGpuIndex}
          trainerGpuMetricsByRank={trainerGpuMetricsByRank}
          trainerGpuMetricsIsLoading={trainerGpuMetricsIsLoading}
          trainerAvailableGpuMetrics={trainerAvailableGpuMetrics}
          selectedMetricNames={selectedTrainerGpuMetrics}
          onSelectedMetricNamesChange={onSelectedTrainerGpuMetricsChange}
          trainerSystemGpuIndices={trainerSystemGpuIndicesForDisplay}
          trainerRankInfoByRank={trainerRankInfoByRank}
          totalSetupNodes={totalSetupNodes}
          isLoading={isLoading}
        />
      </div>
    </div>
  )
}

// ============================================================================
// Orchestrator Section
// ============================================================================

function OrchestratorSection({
  events,
  intervalStart,
  intervalDuration,
  isLoading,
}: {
  events: OrchestratorEvent[]
  intervalStart: number
  intervalDuration: number
  isLoading?: boolean
}) {
  const [isOpen, setIsOpen] = useState(true)
  const [hoveredType, setHoveredType] = useState<string | null>(null)
  const [selectedType, setSelectedType] = useState<string | null>(null)

  const activeHighlight = selectedType ?? hoveredType

  const { thresholdSeconds, displayLabel: thresholdLabel } = useMemo(
    () => computeCloseEventsThreshold(intervalDuration),
    [intervalDuration],
  )

  const { sortedEvents, windowByOriginalIndex } = useMemo(
    () => buildCloseEventWindows(events, thresholdSeconds),
    [events, thresholdSeconds],
  )

  // Compute event counts by display name, preserving first-occurrence order
  const eventCountsByDisplayName = useMemo(() => {
    const result: Array<{ displayName: string; color: string; count: number }> =
      []
    const seen = new Map<string, number>()
    events.forEach((event) => {
      const displayName = getEventDisplayName(event.event_type)
      const existingIdx = seen.get(displayName)
      if (existingIdx !== undefined) {
        result[existingIdx].count++
      } else {
        seen.set(displayName, result.length)
        result.push({
          displayName,
          color: getOrchestratorEventColor(event.event_type),
          count: 1,
        })
      }
    })
    return result
  }, [events])

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <div className="py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors">
          <div className="flex items-center gap-1.5">
            <ChevronDown
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform",
                !isOpen && "-rotate-90",
              )}
            />
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              Orchestrator
            </h3>
            {isLoading && (
              <Spinner className="size-3.5 text-muted-foreground" />
            )}
          </div>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-3">
          {/* Event type legend */}
          {eventCountsByDisplayName.length > 0 && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mb-6 text-xs">
              {eventCountsByDisplayName.map(({ displayName, color, count }) => {
                const isActive = activeHighlight === displayName
                const isDimmed = activeHighlight !== null && !isActive
                return (
                  <div
                    key={displayName}
                    className={`flex items-center gap-1.5 cursor-pointer select-none transition-opacity duration-150 ${
                      isActive
                        ? "ring-1 ring-border rounded-full px-1.5 py-0.5 -mx-1.5 -my-0.5"
                        : ""
                    }`}
                    style={{ opacity: isDimmed ? 0.3 : 1 }}
                    onPointerEnter={() => setHoveredType(displayName)}
                    onPointerLeave={() => setHoveredType(null)}
                    onClick={() =>
                      setSelectedType((prev) =>
                        prev === displayName ? null : displayName,
                      )
                    }
                  >
                    <div
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: color }}
                    />
                    <span className="capitalize text-muted-foreground">
                      {displayName}
                    </span>
                    <span className="text-muted-foreground/50">{count}</span>
                  </div>
                )
              })}
            </div>
          )}

          <div className="relative">
            <div className="relative h-32 bg-background rounded-lg border border-border/50">
              {/* Time axis labels */}
              <div className="absolute bottom-0 left-0 right-0 h-8 flex items-center justify-between px-2 text-xs text-muted-foreground border-t border-border/50">
                <span>{formatTimeShort(0)}</span>
                <span>{formatTimeShort(intervalDuration / 2)}</span>
                <span>{formatTimeShort(intervalDuration)}</span>
              </div>

              {/* Event lines */}
              <div className="absolute top-0 left-0 right-0 h-24">
                {events.map((event, idx) => (
                  <OrchestratorEventLine
                    key={`${event.timestamp}-${idx}`}
                    event={event}
                    intervalStart={intervalStart}
                    intervalDuration={intervalDuration}
                    sortedEvents={sortedEvents}
                    closeEventWindow={windowByOriginalIndex.get(idx)}
                    thresholdLabel={thresholdLabel}
                    highlightedDisplayName={activeHighlight}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function OrchestratorEventLine({
  event,
  intervalStart,
  intervalDuration,
  sortedEvents,
  closeEventWindow,
  thresholdLabel,
  highlightedDisplayName,
}: {
  event: OrchestratorEvent
  intervalStart: number
  intervalDuration: number
  sortedEvents: SortedOrchestratorEvent[]
  closeEventWindow?: CloseEventWindow
  thresholdLabel: string
  highlightedDisplayName: string | null
}) {
  const relativeStart = event.timestamp - intervalStart
  const relativePosition = (relativeStart / intervalDuration) * 100

  if (relativePosition < 0 || relativePosition > 100) return null

  const darkMode = useAtomValue(darkModeAtom)
  const color = getOrchestratorEventColor(event.event_type)
  const closeEventCount = closeEventWindow
    ? Math.max(0, closeEventWindow.end - closeEventWindow.start)
    : 0
  const hasCloseEvents = closeEventCount > 0

  // Determine highlight state: null = no filter, true = matches, false = dimmed
  const displayName = getEventDisplayName(event.event_type)
  const isHighlighted =
    highlightedDisplayName === null
      ? null
      : displayName === highlightedDisplayName

  return (
    <HoverTooltipBlock
      className="absolute top-0 bottom-0 w-0.5 group cursor-pointer"
      style={{
        left: `${relativePosition}%`,
        backgroundColor:
          isHighlighted === false ? dimColor(color, 0.8, darkMode) : color,
        zIndex: isHighlighted === true ? 20 : 1,
      }}
      interactive={hasCloseEvents}
      tooltip={
        <OrchestratorEventTooltip
          event={event}
          sortedEvents={sortedEvents}
          closeEventWindow={closeEventWindow}
          thresholdLabel={thresholdLabel}
        />
      }
    ></HoverTooltipBlock>
  )
}

function OrchestratorEventTooltip({
  event,
  sortedEvents,
  closeEventWindow,
  thresholdLabel,
}: {
  event: OrchestratorEvent
  sortedEvents: SortedOrchestratorEvent[]
  closeEventWindow?: CloseEventWindow
  thresholdLabel: string
}) {
  const getDisplayName = (e: OrchestratorEvent) => {
    const base = getEventDisplayName(e.event_type)
      .split(" ")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ")
    return (e.event_type === "weight_update" ||
      e.event_type === "save_batch") &&
      e.step !== -1
      ? `${base} ${e.step}`
      : base
  }

  const allEvents = useMemo(() => {
    if (!closeEventWindow || sortedEvents.length === 0) return [event]
    const slice = sortedEvents
      .slice(closeEventWindow.start, closeEventWindow.end + 1)
      .map((item) => item.event)
    return slice
  }, [closeEventWindow, sortedEvents, event])

  const hasCloseEvents = allEvents.length > 1

  return (
    <div className="bg-popover text-popover-foreground text-xs rounded-lg shadow-xl whitespace-nowrap border border-border min-w-[220px] max-h-[60vh] overflow-y-auto">
      {hasCloseEvents && (
        <div className="flex items-center gap-1.5 px-3 pt-2 pb-2 border-b border-border/50 sticky top-0 bg-popover z-10">
          <div className="w-1 h-1 rounded-full bg-amber-500 flex-shrink-0" />
          <span className="text-amber-500 font-medium text-[10px]">
            {allEvents.length} events within {thresholdLabel}
          </span>
        </div>
      )}

      <div
        className={`px-3 ${hasCloseEvents ? "pt-2 pb-2 space-y-3" : "py-2"}`}
      >
        {allEvents.map((e, idx) => {
          const eColor = getOrchestratorEventColor(e.event_type)
          const eName = getDisplayName(e)

          return (
            <div
              key={`${e.timestamp}-${idx}`}
              className={
                hasCloseEvents && idx > 0
                  ? "pt-2 border-t border-border/30"
                  : ""
              }
            >
              <div className="font-semibold mb-1.5 flex items-center gap-2">
                <div
                  className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                  style={{ backgroundColor: eColor }}
                />
                <span>{eName}</span>
              </div>
              <div className="space-y-0.5 text-muted-foreground">
                {e.step !== -1 && <div>Step: {e.step}</div>}
                <div>
                  Time:{" "}
                  {new Date(e.timestamp * 1000).toISOString().slice(11, 23)}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ============================================================================
// Inference Section
// ============================================================================

interface AssignedInferenceEvent extends InferenceEvent {
  lane: number
}

function InferenceSection({
  servers,
  eventsByServer,
  intervalStart,
  intervalDuration,
  lanesPerServer,
  inferenceServerNodeMap,
  totalSetupNodes,
  isLoading,
  freeLaneAfterGeneration = false,
}: {
  servers: number[]
  eventsByServer: Record<number, InferenceEvent[]>
  intervalStart: number
  intervalDuration: number
  lanesPerServer: number
  inferenceServerNodeMap?: Record<number, number | null>
  totalSetupNodes?: number
  isLoading?: boolean
  freeLaneAfterGeneration?: boolean
}) {
  const DEFAULT_LANE_HEIGHT = 14
  const LANE_HEIGHT_STEP = 4
  const MIN_LANE_HEIGHT = 2
  const MAX_LANE_HEIGHT = 42

  const [isOpen, setIsOpen] = useState(true)
  const [laneHeight, setLaneHeight] = useAtom(inferenceLaneHeightAtom)
  const [highlightDiscarded, setHighlightDiscarded] = useAtom(
    inferenceHighlightDiscardedAtom,
  )
  const [showWeightUpdate, setShowWeightUpdate] = useAtom(
    inferenceShowWeightUpdateAtom,
  )
  const [showComputeReward, setShowComputeReward] = useAtom(
    inferenceShowComputeRewardAtom,
  )
  const [separateComputeReward, setSeparateComputeReward] = useAtom(
    inferenceSeparateComputeRewardAtom,
  )
  const darkMode = useAtomValue(darkModeAtom)
  const eventBorderMedium = darkMode
    ? "1px solid rgba(255, 255, 255, 0.15)"
    : "1px solid rgba(0, 0, 0, 0.2)"
  const [lanePage, setLanePage] = useAtom(inferenceLanePageAtom)
  const [maxLanes, setMaxLanes] = useAtom(inferenceMaxLanesAtom)
  const [serverPage, setServerPage] = useAtom(inferenceServerPageAtom)
  const [selectedRequest, setSelectedRequest] = useAtom(
    selectedInferenceRequestAtom,
  )
  const selectedRunPath = useAtomValue(selectedRunPathAtom)

  // Server page pagination (pages of 8 servers)
  const SERVER_PAGE_SIZE = 8
  const showServerPagination = servers.length > SERVER_PAGE_SIZE
  const serverPages = useMemo(() => {
    if (!showServerPagination) return []
    const pages: { label: string; servers: number[] }[] = []
    for (let i = 0; i < servers.length; i += SERVER_PAGE_SIZE) {
      const pageServers = servers.slice(i, i + SERVER_PAGE_SIZE)
      const start = i
      const end = Math.min(i + SERVER_PAGE_SIZE, servers.length) - 1
      pages.push({ label: `${start} – ${end}`, servers: pageServers })
    }
    return pages
  }, [servers, showServerPagination])
  const clampedServerPage = showServerPagination
    ? Math.min(serverPage, Math.max(0, serverPages.length - 1))
    : 0
  useEffect(() => {
    if (showServerPagination && serverPage !== clampedServerPage) {
      setServerPage(clampedServerPage)
    }
  }, [showServerPagination, serverPage, clampedServerPage, setServerPage])
  const paginatedServers = showServerPagination
    ? serverPages[clampedServerPage]?.servers ?? servers.slice(0, SERVER_PAGE_SIZE)
    : servers

  // Determine whether to show node labels on separated server titles
  const hasMultipleInferenceNodes = useMemo(() => {
    if (
      typeof totalSetupNodes === "number" &&
      Number.isFinite(totalSetupNodes)
    ) {
      return totalSetupNodes > 1
    }
    if (!inferenceServerNodeMap) return false
    const uniqueNodes = new Set(
      Object.values(inferenceServerNodeMap).filter((n) => n !== null),
    )
    return uniqueNodes.size > 1
  }, [inferenceServerNodeMap, totalSetupNodes])

  const requestSamples = useMemo(() => {
    const seen = new Set<string>()
    const samples: Array<{ group_id: number; sample_idx: number }> = []
    for (const events of Object.values(eventsByServer)) {
      for (const event of events) {
        if (
          event.event_type !== "request" ||
          event.sample_id == null ||
          event.group_id == null
        ) {
          continue
        }
        const key = `${event.group_id}:${event.sample_id}`
        if (seen.has(key)) continue
        seen.add(key)
        samples.push({ group_id: event.group_id, sample_idx: event.sample_id })
      }
    }
    return samples
  }, [eventsByServer])

  const { data: sampleStatuses } = useSampleStatuses(
    selectedRunPath ?? "",
    requestSamples,
    !!selectedRunPath && requestSamples.length > 0 && highlightDiscarded,
  )
  const discardStatusReady =
    !highlightDiscarded || requestSamples.length === 0 || !!sampleStatuses
  const readyToRenderRequests = !highlightDiscarded || discardStatusReady

  const sampleStatusByKey = useMemo(() => {
    const map = new Map<string, "rollouts" | "rollouts_discarded" | null>()
    sampleStatuses?.statuses.forEach((status) => {
      map.set(`${status.group_id}:${status.sample_idx}`, status.kind)
    })
    return map
  }, [sampleStatuses])

  // Compute actual lane count for pagination
  const actualMaxLanesPerServer = useMemo(() => {
    let maxPerServer = lanesPerServer
    for (const events of Object.values(eventsByServer)) {
      for (const event of events) {
        if (event.event_type !== "request") continue
        if (event.lane != null && event.lane + 1 > maxPerServer) {
          maxPerServer = event.lane + 1
        }
      }
    }
    return maxPerServer
  }, [eventsByServer, lanesPerServer])

  // Lane pagination
  const numServersForCalc = Math.max(1, servers.length)
  const lanesPerPage = Math.max(1, Math.floor(maxLanes / numServersForCalc))
  const totalActualLanes = actualMaxLanesPerServer
  const totalLanePages = Math.max(1, Math.ceil(totalActualLanes / lanesPerPage))
  const clampedLanePage = Math.min(lanePage, Math.max(0, totalLanePages - 1))

  // Reset lane page when it's out of range
  useEffect(() => {
    if (lanePage !== clampedLanePage) {
      setLanePage(clampedLanePage)
    }
  }, [lanePage, clampedLanePage, setLanePage])

  const childLaneStart = clampedLanePage * lanesPerPage
  const childMaxLanesToShow = lanesPerPage
  const displayLaneEnd = Math.min(
    childLaneStart + lanesPerPage - 1,
    totalActualLanes - 1,
  )

  const totalEventCount = useMemo(() => {
    let count = 0
    for (const events of Object.values(eventsByServer)) count += events.length
    return count
  }, [eventsByServer])

  const currentRenderData = useMemo(
    () => ({
      servers: paginatedServers,
      eventsByServer,
      intervalStart,
      intervalDuration,
      lanesPerServer,
    }),
    [paginatedServers, eventsByServer, intervalStart, intervalDuration, lanesPerServer],
  )
  const renderKey = useMemo(() => {
    return `${intervalStart}:${intervalDuration}:${paginatedServers.join(",")}:${
      totalEventCount
    }:${highlightDiscarded}`
  }, [
    intervalStart,
    intervalDuration,
    paginatedServers,
    totalEventCount,
    highlightDiscarded,
  ])

  const [activeRenderData, setActiveRenderData] = useState<
    typeof currentRenderData | null
  >(null)
  const [activeSampleStatusByKey, setActiveSampleStatusByKey] = useState<Map<
    string,
    "rollouts" | "rollouts_discarded" | null
  > | null>(null)
  const [activeRenderKey, setActiveRenderKey] = useState<string | null>(null)

  // Only update active data when fully ready (render-time state adjustment)
  if (
    readyToRenderRequests &&
    (activeRenderData !== currentRenderData ||
      activeSampleStatusByKey !== sampleStatusByKey ||
      activeRenderKey !== renderKey)
  ) {
    setActiveRenderData(currentRenderData)
    setActiveSampleStatusByKey(sampleStatusByKey)
    setActiveRenderKey(renderKey)
  }

  // We're transitioning if highlight is on and renderKey differs from activeRenderKey.
  // renderKey updates immediately (useMemo), while activeRenderKey only updates when data is ready.
  // This ensures opacity applies immediately on page change, not after an effect runs.
  const isTransitioning =
    highlightDiscarded &&
    activeRenderKey !== null &&
    renderKey !== activeRenderKey

  // Handle click on an inference event
  const handleEventClick = useCallback(
    (event: InferenceEvent) => {
      // Only handle requests, not weight_broadcast events
      if (event.event_type !== "request") return
      if (event.sample_id === undefined || event.group_id === undefined) return

      // Toggle off if clicking the same sample_id
      if (selectedRequest?.sampleId === event.sample_id) {
        setSelectedRequest(null)
        return
      }

      setSelectedRequest({
        sampleId: event.sample_id,
        groupId: event.group_id,
        isEval: event.is_eval,
      })
    },
    [selectedRequest, setSelectedRequest],
  )

  // Separate compute reward: extract reward items from all servers
  const computeRewardItems = useMemo(() => {
    if (!separateComputeReward) return []
    const items: Array<{
      startTime: number
      duration: number
      sourceEvent: InferenceEvent
      server: number
    }> = []
    for (const [serverStr, events] of Object.entries(eventsByServer)) {
      const server = Number(serverStr)
      for (const event of events) {
        if (event.event_type !== "request") continue
        const rewardTime = event.compute_reward_time
        if (rewardTime == null || rewardTime <= 0) continue
        const envTime = event.environment_response_time
        const startTime =
          event.end_time + (envTime != null && envTime > 0 ? envTime : 0)
        items.push({ startTime, duration: rewardTime, sourceEvent: event, server })
      }
    }
    items.sort((a, b) => a.startTime - b.startTime)
    return items
  }, [separateComputeReward, eventsByServer])

  // Greedy lane packing for separate compute reward section
  const { packedRewardItems, rewardLaneCount } = useMemo(() => {
    if (computeRewardItems.length === 0)
      return { packedRewardItems: [] as Array<(typeof computeRewardItems)[number] & { lane: number }>, rewardLaneCount: 0 }
    const laneEnds: number[] = []
    const packed: Array<(typeof computeRewardItems)[number] & { lane: number }> = []
    for (const item of computeRewardItems) {
      let placed = false
      for (let i = 0; i < laneEnds.length; i++) {
        if (laneEnds[i] <= item.startTime) {
          laneEnds[i] = item.startTime + item.duration
          packed.push({ ...item, lane: i })
          placed = true
          break
        }
      }
      if (!placed) {
        packed.push({ ...item, lane: laneEnds.length })
        laneEnds.push(item.startTime + item.duration)
      }
    }
    return { packedRewardItems: packed, rewardLaneCount: laneEnds.length }
  }, [computeRewardItems])

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="flex items-center gap-2 mb-3">
        <CollapsibleTrigger asChild>
          <div className="flex items-center gap-1.5 py-1.5 px-2 -ml-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <ChevronDown
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform",
                !isOpen && "-rotate-90",
              )}
            />
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              Inference Servers
            </h3>
            <div className="w-6 flex items-center justify-center shrink-0">
              {isLoading && (
                <Spinner className="size-3.5 text-muted-foreground" />
              )}
            </div>
          </div>
        </CollapsibleTrigger>
        {isOpen && (
          <>
            <div className="flex items-center rounded-md border border-input">
              <button
                className="flex items-center justify-center h-7 w-7 text-muted-foreground hover:bg-accent hover:text-accent-foreground rounded-l-md transition-colors disabled:opacity-50 disabled:pointer-events-none"
                onClick={() =>
                  setLaneHeight((h) =>
                    Math.max(MIN_LANE_HEIGHT, h - LANE_HEIGHT_STEP),
                  )
                }
                disabled={laneHeight <= MIN_LANE_HEIGHT}
              >
                <Minus className="h-3 w-3" />
              </button>
              <button
                className="flex items-center justify-center h-7 px-2 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground border-x border-input transition-colors"
                onClick={() => setLaneHeight(DEFAULT_LANE_HEIGHT)}
              >
                Reset
              </button>
              <button
                className="flex items-center justify-center h-7 w-7 text-muted-foreground hover:bg-accent hover:text-accent-foreground rounded-r-md transition-colors disabled:opacity-50 disabled:pointer-events-none"
                onClick={() =>
                  setLaneHeight((h) =>
                    Math.min(MAX_LANE_HEIGHT, h + LANE_HEIGHT_STEP),
                  )
                }
                disabled={laneHeight >= MAX_LANE_HEIGHT}
              >
                <Plus className="h-3 w-3" />
              </button>
            </div>
            {/* Lane page selector */}
            <div className="flex items-center rounded-md border border-input">
              <button
                className="flex items-center justify-center h-7 w-7 text-muted-foreground hover:bg-accent hover:text-accent-foreground rounded-l-md transition-colors disabled:opacity-50 disabled:pointer-events-none"
                onClick={() => setLanePage((p) => Math.max(0, p - 1))}
                disabled={clampedLanePage <= 0}
              >
                <ChevronUp className="h-3 w-3" />
              </button>
              <span className="flex items-center justify-center h-7 px-2 text-xs text-muted-foreground border-x border-input tabular-nums whitespace-nowrap select-none">
                {totalActualLanes === 0
                  ? "0"
                  : `${childLaneStart}–${displayLaneEnd}`}
              </span>
              <button
                className="flex items-center justify-center h-7 w-7 text-muted-foreground hover:bg-accent hover:text-accent-foreground rounded-r-md transition-colors disabled:opacity-50 disabled:pointer-events-none"
                onClick={() =>
                  setLanePage((p) => Math.min(totalLanePages - 1, p + 1))
                }
                disabled={clampedLanePage >= totalLanePages - 1}
              >
                <ChevronDown className="h-3 w-3" />
              </button>
            </div>
            {/* Max lanes gear selector */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="flex items-center justify-center h-7 w-7 text-muted-foreground hover:bg-accent hover:text-accent-foreground rounded-md border border-input transition-colors">
                  <Settings className="h-3.5 w-3.5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel>Max Lanes</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {[64, 128, 256, 512, 1024].map((n) => (
                  <DropdownMenuCheckboxItem
                    key={n}
                    checked={maxLanes === n}
                    onCheckedChange={() => {
                      setMaxLanes(n)
                      setLanePage(0)
                    }}
                  >
                    {n}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            {showServerPagination && (
              <Select
                value={String(clampedServerPage)}
                onValueChange={(v) => setServerPage(Number(v))}
              >
                <SelectTrigger size="sm" className="text-xs h-7 gap-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {serverPages.map((page, idx) => (
                    <SelectItem key={idx} value={String(idx)}>
                      Servers {page.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            <Toggle
              variant="selecting"
              size="sm"
              pressed={highlightDiscarded}
              onPressedChange={setHighlightDiscarded}
              className="text-xs px-2 h-7"
            >
              Highlight Discarded
            </Toggle>
            <Toggle
              variant="selecting"
              size="sm"
              pressed={showWeightUpdate}
              onPressedChange={setShowWeightUpdate}
              className="text-xs px-2 h-7"
            >
              Show Weight Update
            </Toggle>
            <div className="flex items-center">
              <Toggle
                variant="selecting"
                size="sm"
                pressed={showComputeReward}
                onPressedChange={setShowComputeReward}
                className="text-xs px-2 h-7 rounded-r-none"
              >
                Show Compute Reward
              </Toggle>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    className={cn(
                      "flex items-center justify-center h-7 w-6 rounded-r-[min(var(--radius-md),12px)] transition-colors",
                      showComputeReward
                        ? "bg-primary text-secondary hover:bg-primary/90"
                        : "bg-accent text-muted-foreground hover:text-accent-foreground hover:bg-muted",
                    )}
                  >
                    <ChevronDown className="h-3 w-3" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuCheckboxItem
                    checked={separateComputeReward}
                    onCheckedChange={setSeparateComputeReward}
                  >
                    Separate
                  </DropdownMenuCheckboxItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </>
        )}
      </div>

      <CollapsibleContent>
        <div className="space-y-4">
          {servers.length > 0 ? (
            (() => {
              const renderData =
                activeRenderData ??
                (readyToRenderRequests ? currentRenderData : null)
              const renderSampleStatusByKey =
                activeSampleStatusByKey ??
                (readyToRenderRequests ? sampleStatusByKey : null)
              if (!renderData || !renderSampleStatusByKey) return null
              const renderDiscardStatusReady = isTransitioning
                ? true
                : discardStatusReady
              return (
                <div
                  className={`transition-opacity ${
                    isTransitioning ? "opacity-60" : "opacity-100"
                  }`}
                >
                  <div className="space-y-3">
                    {renderData.servers.map((server) => (
                      <InferenceServerTimeline
                        key={server}
                        server={server}
                        nodeId={inferenceServerNodeMap?.[server] ?? null}
                        showNodeLabel={hasMultipleInferenceNodes}
                        events={renderData.eventsByServer[server] || []}
                        intervalStart={renderData.intervalStart}
                        intervalDuration={renderData.intervalDuration}
                        numLanes={renderData.lanesPerServer}
                        selectedRequest={selectedRequest}
                        sampleStatusByKey={renderSampleStatusByKey}
                        highlightDiscarded={highlightDiscarded}
                        discardStatusReady={renderDiscardStatusReady}
                        onEventClick={handleEventClick}
                        showWeightUpdate={showWeightUpdate}
                        showComputeReward={showComputeReward && !separateComputeReward}
                        laneHeight={laneHeight}
                        laneStart={childLaneStart}
                        maxLanesToShow={childMaxLanesToShow}
                        freeLaneAfterGeneration={freeLaneAfterGeneration}
                      />
                    ))}
                  </div>
                </div>
              )
            })()
          ) : (
            <div className="h-24 flex items-center justify-center text-muted-foreground bg-muted/30 rounded-lg border border-border/50">
              No inference events in this interval
            </div>
          )}
        </div>

        {/* Separate compute reward section */}
        {separateComputeReward && packedRewardItems.length > 0 && (
          <div className="mt-3">
            <div className="text-xs font-medium text-muted-foreground mb-0.5">
              Compute Reward
            </div>
            <div
              className="relative bg-muted/30 rounded-lg border border-border/50"
              style={{
                height:
                  rewardLaneCount * laneHeight +
                  Math.max(0, rewardLaneCount - 1) * 2,
              }}
            >
              {packedRewardItems.map((item, idx) => {
                const intervalEnd = intervalStart + intervalDuration
                const rewardEnd = item.startTime + item.duration
                if (
                  item.startTime >= intervalEnd ||
                  rewardEnd <= intervalStart
                )
                  return null

                const visibleStart = Math.max(item.startTime, intervalStart)
                const visibleEnd = Math.min(rewardEnd, intervalEnd)
                const leftPct = Math.max(
                  0,
                  ((visibleStart - intervalStart) / intervalDuration) * 100,
                )
                const widthPct = Math.max(
                  0.3,
                  ((visibleEnd - visibleStart) / intervalDuration) * 100,
                )
                const laneTop = item.lane * (laneHeight + 2)

                const rewardStyle = getTimingBarStyle(
                  item.sourceEvent,
                  "reward",
                  selectedRequest,
                  sampleStatusByKey,
                  highlightDiscarded,
                  discardStatusReady,
                  darkMode,
                )

                return (
                  <HoverTooltipBlock
                    key={`cr-${idx}`}
                    className="absolute group cursor-pointer"
                    style={{
                      left: `${leftPct}%`,
                      width: `${Math.min(widthPct, 100 - leftPct)}%`,
                      top: laneTop,
                      height: laneHeight,
                      backgroundColor: rewardStyle.color,
                      borderRadius: "1px",
                      border: eventBorderMedium,
                      boxSizing: "border-box",
                      opacity: rewardStyle.opacity,
                    }}
                    onClick={() => handleEventClick(item.sourceEvent)}
                    tooltip={
                      <EventTooltip
                        title={
                          item.sourceEvent.is_eval
                            ? "Compute Metrics"
                            : "Compute Reward"
                        }
                        titleSecondary={
                          item.sourceEvent.sample_id !== undefined
                            ? `Sample ${item.sourceEvent.sample_id}`
                            : undefined
                        }
                        color={rewardStyle.color}
                        details={[
                          {
                            label: "Duration",
                            value: formatDuration(item.duration * 1000),
                          },
                          ...(item.sourceEvent.group_id !== undefined
                            ? [
                                {
                                  label: "Group",
                                  value: String(item.sourceEvent.group_id),
                                },
                              ]
                            : []),
                          {
                            label: "Server",
                            value: String(item.server),
                          },
                          {
                            label: "Lane",
                            value: String(item.lane),
                          },
                        ]}
                      />
                    }
                  />
                )
              })}
            </div>
          </div>
        )}

        {/* Time axis at bottom */}
        <div className="flex justify-between text-xs text-muted-foreground mt-2">
          <span>{formatTimeShort(0)}</span>
          <span>{formatTimeShort(intervalDuration / 2)}</span>
          <span>{formatTimeShort(intervalDuration)}</span>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

// Selected = darker version of base color
const INFERENCE_REQUEST_SELECTED_COLOR = "#075985" // sky-800 (darker blue)
const INFERENCE_REQUEST_EVAL_SELECTED_COLOR = "#047857" // emerald-700 (darker green)
// Group timeline: non-selected samples
const GROUP_SAMPLE_COLOR = "#0284c7" // sky-600
const GROUP_SAMPLE_EVAL_COLOR = "#10b981" // emerald-500
const GROUP_SAMPLE_DISCARDED_COLOR_LIGHT = "#a1a1a1"
const GROUP_SAMPLE_DISCARDED_COLOR_DARK = "#555555"
const GROUP_SAMPLE_CANCELED_COLOR_LIGHT = "#b0b0b0"
const GROUP_SAMPLE_CANCELED_COLOR_DARK = "#4a4a4a"
// Group timeline: selected sample (darker)
const GROUP_SAMPLE_SELECTED_COLOR = "#0c4a6e" // sky-900
const GROUP_SAMPLE_EVAL_SELECTED_COLOR = "#065f46" // emerald-800
const GROUP_SAMPLE_CANCELED_SELECTED_COLOR_LIGHT = "#8a8a8a"
const GROUP_SAMPLE_CANCELED_SELECTED_COLOR_DARK = "#606060"
// Env response and compute reward base colors
const ENV_RESPONSE_COLOR = "#0ea5e9" // sky-400
const ENV_RESPONSE_DISCARDED_COLOR_LIGHT = "#ababab"
const ENV_RESPONSE_DISCARDED_COLOR_DARK = "#555555"
const COMPUTE_REWARD_COLOR = "#0ea5e9" // sky-400
const COMPUTE_REWARD_DISCARDED_COLOR_LIGHT = "#ababab"
const COMPUTE_REWARD_DISCARDED_COLOR_DARK = "#555555"
// Eval compute metrics colors (green variants)
const COMPUTE_METRICS_COLOR = "#34d399" // emerald-400
const COMPUTE_METRICS_DISCARDED_COLOR_LIGHT = "#ababab"
const COMPUTE_METRICS_DISCARDED_COLOR_DARK = "#555555"
// Env response / compute reward for selected sample (darker version of base)
const ENV_RESPONSE_SELECTED_COLOR = "#0369a1" // sky-700
const COMPUTE_REWARD_SELECTED_COLOR = "#0369a1" // sky-700
const COMPUTE_METRICS_SELECTED_COLOR = "#047857" // emerald-700
// Darker variants for same-group (non-selected) env response / compute reward
const ENV_RESPONSE_GROUP_COLOR = "#0284c7" // sky-600
const COMPUTE_REWARD_GROUP_COLOR = "#0284c7" // sky-600
const COMPUTE_METRICS_GROUP_COLOR = "#10b981" // emerald-500
const ENV_RESPONSE_GROUP_DISCARDED_COLOR_LIGHT = "#b3b3b3"
const ENV_RESPONSE_GROUP_DISCARDED_COLOR_DARK = "#505050"
const COMPUTE_REWARD_GROUP_DISCARDED_COLOR_LIGHT = "#b3b3b3"
const COMPUTE_REWARD_GROUP_DISCARDED_COLOR_DARK = "#505050"
const COMPUTE_METRICS_GROUP_DISCARDED_COLOR_LIGHT = "#b3b3b3"
const COMPUTE_METRICS_GROUP_DISCARDED_COLOR_DARK = "#505050"

// Mini timeline showing all samples in a group
export function GroupSampleTimeline({
  eventsBySampleId,
  selectedSampleId,
  timeBounds,
  groupId,
  runPath,
  envResponseTimesBySample,
  computeRewardTimeBySample,
  onSampleClick,
  showNodeLabel,
  isEval,
}: {
  eventsBySampleId: Record<number, InferenceEvent[]>
  selectedSampleId: number
  timeBounds: { start: number; end: number; duration: number }
  groupId: number
  runPath: string
  envResponseTimesBySample?: Record<
    number,
    Array<{ turn_order: number; time: number }>
  >
  computeRewardTimeBySample?: Record<number, number>
  onSampleClick?: (sampleId: number) => void
  showNodeLabel?: boolean
  isEval?: boolean
}) {
  const sampleIds = Object.keys(eventsBySampleId)
    .map(Number)
    .sort((a, b) => a - b)

  const highlightDiscarded = useAtomValue(inferenceHighlightDiscardedAtom)
  const darkMode = useAtomValue(darkModeAtom)
  const eventBorderLight = darkMode ? "1px solid rgba(255, 255, 255, 0.2)" : "1px solid rgba(0, 0, 0, 0.3)"
  const eventBorderLightSubtle = darkMode ? "1px solid rgba(255, 255, 255, 0.1)" : "1px solid rgba(0, 0, 0, 0.12)"
  const eventBorderMedium = darkMode ? "1px solid rgba(255, 255, 255, 0.15)" : "1px solid rgba(0, 0, 0, 0.2)"

  // Check discard status for all samples in this group
  const requestSamples = useMemo(
    () => sampleIds.map((s) => ({ group_id: groupId, sample_idx: s })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [groupId, sampleIds.join(",")],
  )
  const { data: sampleStatuses } = useSampleStatuses(
    runPath,
    requestSamples,
    !!runPath && requestSamples.length > 0 && highlightDiscarded,
  )
  const discardStatusReady =
    !highlightDiscarded || requestSamples.length === 0 || !!sampleStatuses
  const isDiscardedGroup = useMemo(() => {
    if (!highlightDiscarded || !sampleStatuses?.statuses) return false
    return sampleStatuses.statuses.some((s) => s.kind === "rollouts_discarded")
  }, [highlightDiscarded, sampleStatuses])

  const isEvalGroup = useMemo(() => {
    for (const events of Object.values(eventsBySampleId)) {
      if (events.some((e) => e.is_eval)) return true
    }
    return false
  }, [eventsBySampleId])

  const isCanceledGroup = useMemo(() => {
    for (const events of Object.values(eventsBySampleId)) {
      if (events.some((e) => e.is_canceled)) return true
    }
    return false
  }, [eventsBySampleId])

  // Check if group is inflight or pending status (not yet kept/discarded)
  const isInflightOrPendingGroup = useMemo(() => {
    if (isCanceledGroup || isDiscardedGroup) return false
    if (!highlightDiscarded) return false
    if (!discardStatusReady) return true
    if (!sampleStatuses?.statuses?.length) return true
    return !sampleStatuses.statuses.some((s) => s.kind === "rollouts")
  }, [isCanceledGroup, isDiscardedGroup, highlightDiscarded, discardStatusReady, sampleStatuses])

  const groupColor = isCanceledGroup
    ? (darkMode ? GROUP_SAMPLE_CANCELED_COLOR_DARK : GROUP_SAMPLE_CANCELED_COLOR_LIGHT)
    : isDiscardedGroup
      ? (darkMode ? GROUP_SAMPLE_DISCARDED_COLOR_DARK : GROUP_SAMPLE_DISCARDED_COLOR_LIGHT)
      : isInflightOrPendingGroup
        ? (darkMode ? "rgba(202, 138, 4, 0.7)" : "rgba(202, 138, 4, 0.8)")
        : isEvalGroup
          ? GROUP_SAMPLE_EVAL_COLOR
          : GROUP_SAMPLE_COLOR

  const rowHeight = 16
  const rowGap = 3
  const totalHeight =
    sampleIds.length * rowHeight + (sampleIds.length - 1) * rowGap
  const leftColumnWidth = 48

  // Don't render until discard statuses are loaded (avoids blue → gray flash)
  if (!discardStatusReady) {
    return (
      <div
        className="px-3 pb-2 flex items-center justify-center"
        style={{ height: totalHeight + 24 }}
      >
        <span className="text-xs text-muted-foreground">Loading…</span>
      </div>
    )
  }

  return (
    <div className="px-3 pb-2">
      <div className="flex items-start">
        {/* Sample ID labels */}
        <div
          className="flex flex-col pr-2"
          style={{ width: leftColumnWidth, gap: rowGap }}
        >
          {sampleIds.map((sampleId) => {
            const isSelected = sampleId === selectedSampleId
            return (
              <SampleDetailsDialog
                key={sampleId}
                runPath={runPath}
                groupId={groupId}
                sampleId={sampleId}
                isEval={isEval}
                isCanceled={isCanceledGroup}
              >
                <button
                  className={`text-[10px] font-semibold text-center rounded transition-colors flex items-center justify-center ${
                    isSelected
                      ? isCanceledGroup
                        ? "text-muted-foreground bg-gray-400/15 hover:bg-gray-400/25"
                        : isDiscardedGroup
                          ? "text-muted-foreground bg-gray-500/15 hover:bg-gray-500/25"
                          : isInflightOrPendingGroup
                            ? "text-yellow-800 bg-yellow-500/15 hover:bg-yellow-500/25"
                            : isEvalGroup
                              ? "text-emerald-800 bg-emerald-500/15 hover:bg-emerald-500/25"
                              : "text-sky-900 bg-sky-500/15 hover:bg-sky-500/25"
                      : isCanceledGroup
                        ? "text-muted-foreground bg-gray-400/10 hover:bg-gray-400/20"
                        : isDiscardedGroup
                          ? "text-muted-foreground bg-gray-500/10 hover:bg-gray-500/20"
                          : isInflightOrPendingGroup
                            ? "text-yellow-700 bg-yellow-500/10 hover:bg-yellow-500/20"
                            : isEvalGroup
                              ? "text-emerald-600 bg-emerald-500/10 hover:bg-emerald-500/20"
                              : "text-sky-700 bg-sky-500/10 hover:bg-sky-500/20"
                  }`}
                  style={{ height: rowHeight, width: 34 }}
                >
                  {sampleId}
                </button>
              </SampleDetailsDialog>
            )
          })}
        </div>

        {/* Timeline bars */}
        <div
          className="relative flex-1 bg-muted/30 rounded-lg border border-border/50"
          style={{ height: totalHeight }}
        >
          {sampleIds.map((sampleId, idx) => {
            const events = eventsBySampleId[sampleId]
            const isSelected = sampleId === selectedSampleId
            const rowTop = idx * (rowHeight + rowGap)
            const color = isSelected
              ? isCanceledGroup
                ? (darkMode ? GROUP_SAMPLE_CANCELED_SELECTED_COLOR_DARK : GROUP_SAMPLE_CANCELED_SELECTED_COLOR_LIGHT)
                : isDiscardedGroup
                  ? (darkMode ? "#9ca3af" : "#6b7280")
                  : isInflightOrPendingGroup
                    ? (darkMode ? "rgba(161, 98, 7, 0.8)" : "rgba(161, 98, 7, 0.9)")
                    : isEvalGroup
                      ? GROUP_SAMPLE_EVAL_SELECTED_COLOR
                      : GROUP_SAMPLE_SELECTED_COLOR
              : groupColor

            // Sort events by start_time for env response placement
            const sortedEvents = [...events].sort(
              (a, b) => a.start_time - b.start_time,
            )
            const envTimes = envResponseTimesBySample?.[sampleId] ?? []
            const rewardTime = computeRewardTimeBySample?.[sampleId] ?? 0

            // Build env response trace positions: after each inference event
            const envTraces: Array<{
              start: number
              duration: number
              idx: number
            }> = []
            for (
              let i = 0;
              i < sortedEvents.length && i < envTimes.length;
              i++
            ) {
              const envDur = envTimes[i].time
              if (envDur > 0) {
                envTraces.push({
                  start: sortedEvents[i].end_time,
                  duration: envDur,
                  idx: i,
                })
              }
            }

            // Compute reward trace: after the last event (or last env response)
            let rewardStart = 0
            if (rewardTime > 0 && sortedEvents.length > 0) {
              const lastEvent = sortedEvents[sortedEvents.length - 1]
              rewardStart = lastEvent.end_time
              // If there's an env response after the last event, start after it
              if (envTimes.length >= sortedEvents.length) {
                const lastEnv = envTimes[sortedEvents.length - 1]
                if (lastEnv && lastEnv.time > 0) {
                  rewardStart += lastEnv.time
                }
              }
            }

            return (
              <div
                key={sampleId}
                className="absolute left-0 right-0"
                style={{ top: rowTop, height: rowHeight }}
              >
                {/* Inference request bars */}
                {events.map((event, eventIdx) => {
                  const relativeStart = event.start_time - timeBounds.start
                  const leftPercent =
                    (relativeStart / timeBounds.duration) * 100
                  const duration = event.end_time - event.start_time
                  const durationMs = duration * 1000
                  const widthPercent = Math.max(
                    0.5,
                    (duration / timeBounds.duration) * 100,
                  )

                  return (
                    <HoverTooltipBlock
                      key={`req-${eventIdx}`}
                      className={`absolute top-px bottom-px group ${onSampleClick ? "cursor-pointer" : ""}`}
                      style={{
                        left: `${leftPercent}%`,
                        width: `${Math.min(widthPercent, 100 - leftPercent)}%`,
                        backgroundColor: color,
                        borderRadius: "1px",
                        border: isCanceledGroup
                          ? eventBorderLightSubtle
                          : eventBorderLight,
                        boxSizing: "border-box",
                      }}
                      onClick={
                        onSampleClick
                          ? () => onSampleClick(sampleId)
                          : undefined
                      }
                      tooltip={
                        <EventTooltip
                          title={
                            event.sample_id !== undefined
                              ? `Sample ${event.sample_id}`
                              : event.group_id !== undefined
                                ? `Group ${event.group_id}`
                                : "Request"
                          }
                          titleSecondary={
                            event.group_id !== undefined
                              ? `Group ${event.group_id}`
                              : null
                          }
                          color={color}
                          statusLabel={
                            isCanceledGroup
                              ? { text: "Canceled", className: "text-muted-foreground" }
                              : isDiscardedGroup
                                ? {
                                    text: "Discarded",
                                    className: "text-red-700",
                                  }
                                : null
                          }
                          details={buildInferenceRequestDetails(
                            event,
                            durationMs,
                            [
                              { label: "Server", value: String(event.server) },
                              ...(showNodeLabel &&
                              event.node_id !== null &&
                              event.node_id !== undefined
                                ? [
                                    {
                                      label: "Node",
                                      value: String(event.node_id),
                                    },
                                  ]
                                : []),
                            ],
                          )}
                          startTime={event.start_time}
                          endTime={event.end_time}
                        />
                      }
                    />
                  )
                })}

                {/* Environment response traces */}
                {envTraces.map((trace) => {
                  const relativeStart = trace.start - timeBounds.start
                  const leftPercent =
                    (relativeStart / timeBounds.duration) * 100
                  const widthPercent = Math.max(
                    0.3,
                    (trace.duration / timeBounds.duration) * 100,
                  )
                  const durationMs = trace.duration * 1000
                  const envColor = isSelected
                    ? isDiscardedGroup
                      ? (darkMode ? "#9ca3af" : "#6b7280")
                      : ENV_RESPONSE_SELECTED_COLOR
                    : isDiscardedGroup
                      ? (darkMode ? ENV_RESPONSE_GROUP_DISCARDED_COLOR_DARK : ENV_RESPONSE_GROUP_DISCARDED_COLOR_LIGHT)
                      : ENV_RESPONSE_GROUP_COLOR

                  return (
                    <HoverTooltipBlock
                      key={`env-${trace.idx}`}
                      className={`absolute top-px bottom-px group ${onSampleClick ? "cursor-pointer" : ""}`}
                      style={{
                        left: `${leftPercent}%`,
                        width: `${Math.min(widthPercent, 100 - leftPercent)}%`,
                        backgroundColor: envColor,
                        borderRadius: "1px",
                        border: eventBorderMedium,
                        boxSizing: "border-box",
                        opacity: 0.85,
                      }}
                      onClick={
                        onSampleClick
                          ? () => onSampleClick(sampleId)
                          : undefined
                      }
                      tooltip={
                        <EventTooltip
                          title="Env Response"
                          titleSecondary={`Turn ${trace.idx}`}
                          color={envColor}
                          details={[
                            {
                              label: "Duration",
                              value: formatDuration(durationMs),
                            },
                          ]}
                        />
                      }
                    />
                  )
                })}

                {/* Compute reward / metrics trace */}
                {rewardTime > 0 &&
                  sortedEvents.length > 0 &&
                  (() => {
                    const relativeStart = rewardStart - timeBounds.start
                    const leftPercent =
                      (relativeStart / timeBounds.duration) * 100
                    const widthPercent = Math.max(
                      0.3,
                      (rewardTime / timeBounds.duration) * 100,
                    )
                    const durationMs = rewardTime * 1000
                    const rwdColor = isSelected
                      ? isDiscardedGroup
                        ? (darkMode ? "#9ca3af" : "#6b7280")
                        : isEval
                          ? COMPUTE_METRICS_SELECTED_COLOR
                          : COMPUTE_REWARD_SELECTED_COLOR
                      : isDiscardedGroup
                        ? isEval
                          ? (darkMode ? COMPUTE_METRICS_GROUP_DISCARDED_COLOR_DARK : COMPUTE_METRICS_GROUP_DISCARDED_COLOR_LIGHT)
                          : (darkMode ? COMPUTE_REWARD_GROUP_DISCARDED_COLOR_DARK : COMPUTE_REWARD_GROUP_DISCARDED_COLOR_LIGHT)
                        : isEval
                          ? COMPUTE_METRICS_GROUP_COLOR
                          : COMPUTE_REWARD_GROUP_COLOR

                    return (
                      <HoverTooltipBlock
                        className={`absolute top-px bottom-px group ${onSampleClick ? "cursor-pointer" : ""}`}
                        style={{
                          left: `${leftPercent}%`,
                          width: `${Math.min(widthPercent, 100 - leftPercent)}%`,
                          backgroundColor: rwdColor,
                          borderRadius: "1px",
                          border: eventBorderMedium,
                          boxSizing: "border-box",
                          opacity: 0.85,
                        }}
                        onClick={
                          onSampleClick
                            ? () => onSampleClick(sampleId)
                            : undefined
                        }
                        tooltip={
                          <EventTooltip
                            title={isEval ? "Compute Metrics" : "Compute Reward"}
                            color={rwdColor}
                            details={[
                              {
                                label: "Duration",
                                value: formatDuration(durationMs),
                              },
                            ]}
                          />
                        }
                      />
                    )
                  })()}
              </div>
            )
          })}
        </div>
      </div>

      {/* Time axis */}
      <div
        className="flex justify-between text-[10px] text-muted-foreground mt-1"
        style={{ marginLeft: leftColumnWidth }}
      >
        <span>{formatDuration(0)}</span>
        <span>{formatDuration((timeBounds.duration / 2) * 1000)}</span>
        <span>{formatDuration(timeBounds.duration * 1000)}</span>
      </div>
    </div>
  )
}

function InferenceServerTimeline({
  server,
  nodeId,
  showNodeLabel,
  events,
  intervalStart,
  intervalDuration,
  numLanes,
  selectedRequest,
  sampleStatusByKey,
  highlightDiscarded,
  discardStatusReady,
  onEventClick,
  showWeightUpdate = true,
  showComputeReward = false,
  laneHeight: laneHeightProp = 14,
  laneStart = 0,
  maxLanesToShow,
  freeLaneAfterGeneration = false,
}: {
  server: number
  nodeId?: number | null
  showNodeLabel?: boolean
  events: InferenceEvent[]
  intervalStart: number
  intervalDuration: number
  numLanes: number
  selectedRequest: SelectedInferenceRequest | null
  sampleStatusByKey: Map<string, "rollouts" | "rollouts_discarded" | null>
  highlightDiscarded: boolean
  discardStatusReady: boolean
  onEventClick: (event: InferenceEvent) => void
  showWeightUpdate?: boolean
  showComputeReward?: boolean
  laneHeight?: number
  laneStart?: number
  maxLanesToShow?: number
  freeLaneAfterGeneration?: boolean
}) {
  const darkMode = useAtomValue(darkModeAtom)
  const eventBorderMedium = darkMode ? "1px solid rgba(255, 255, 255, 0.15)" : "1px solid rgba(0, 0, 0, 0.2)"
  const eventBorderWide = darkMode ? "1.5px solid rgba(255, 255, 255, 0.15)" : "1.5px solid rgba(0, 0, 0, 0.2)"

  // Only expose nodeId for tooltips when there are multiple nodes
  const displayNodeId =
    showNodeLabel && nodeId !== null && nodeId !== undefined
      ? nodeId
      : undefined

  const weightBroadcastEvents = useMemo(
    () => events.filter((e) => e.event_type === "weight_broadcast"),
    [events],
  )

  const requestEvents = useMemo(
    () => events.filter((e) => e.event_type === "request"),
    [events],
  )

  // Separated view uses server-provided per-server lane assignments directly.
  // Events without a lane (e.g. eval inference) get lanes via greedy assignment.
  const { assignedEvents, actualNumLanes } = useMemo(() => {
    const withLane: AssignedInferenceEvent[] = []
    const withoutLane: InferenceEvent[] = []
    for (const e of requestEvents) {
      if (e.lane != null && e.lane >= 0) {
        withLane.push({ ...e, lane: e.lane as number })
      } else {
        withoutLane.push(e)
      }
    }

    if (withoutLane.length > 0) {
      const sorted = [...withoutLane].sort(
        (a, b) => a.start_time - b.start_time,
      )
      const laneEnds: number[] = []
      for (const e of sorted) {
        // When freeLaneAfterGeneration, the lane is freed after generation
        // (end_time), so compute_reward doesn't block the lane.
        const eventLaneEnd = freeLaneAfterGeneration
          ? e.end_time + (e.environment_response_time ?? 0)
          : e.end_time +
            (e.environment_response_time ?? 0) +
            (e.compute_reward_time ?? 0)
        let placed = false
        for (let i = 0; i < laneEnds.length; i++) {
          if (laneEnds[i] <= e.start_time) {
            laneEnds[i] = eventLaneEnd
            withLane.push({ ...e, lane: i })
            placed = true
            break
          }
        }
        if (!placed) {
          const newLane = laneEnds.length
          laneEnds.push(eventLaneEnd)
          withLane.push({ ...e, lane: newLane })
        }
      }
    }

    const maxLane = withLane.reduce((max, e) => Math.max(max, e.lane), -1)
    return {
      assignedEvents: withLane,
      actualNumLanes: Math.max(numLanes, maxLane + 1),
    }
  }, [requestEvents, numLanes, freeLaneAfterGeneration])

  const intervalEnd = intervalStart + intervalDuration
  const laneHeight = laneHeightProp

  // Lane pagination: only display a subset of lanes
  const displayedEndLane =
    maxLanesToShow !== undefined
      ? Math.min(laneStart + maxLanesToShow, actualNumLanes)
      : actualNumLanes
  const displayedLaneCount = Math.max(0, displayedEndLane - laneStart)

  const totalHeight =
    displayedLaneCount * laneHeight + Math.max(0, displayedLaneCount - 1) * 2 // include gaps

  // All lane indices to render (simple range, no virtualization)
  const laneIndices = useMemo(() => {
    return Array.from({ length: displayedLaneCount }, (_, idx) => idx)
  }, [displayedLaneCount])

  // Precompute events + idle periods per lane in one pass
  const { laneEventsByIndex, idlePeriodsByLane } = useMemo(() => {
    const laneEvents: InferenceEvent[][] = Array.from(
      { length: actualNumLanes },
      () => [],
    )

    for (const event of assignedEvents) {
      const lane = event.lane
      if (laneEvents[lane]) {
        laneEvents[lane].push(event)
      }
    }

    const idleByLane = laneEvents.map((laneEventsForLane) => {
      const idlePeriods: Array<{
        start: number
        end: number
        duration: number
      }> = []

      if (laneEventsForLane.length > 0) {
        // Idle at beginning
        if (laneEventsForLane[0].start_time > intervalStart) {
          idlePeriods.push({
            start: intervalStart,
            end: laneEventsForLane[0].start_time,
            duration: laneEventsForLane[0].start_time - intervalStart,
          })
        }

        // Gaps between events
        for (let i = 0; i < laneEventsForLane.length - 1; i++) {
          const currentEnd = laneEventsForLane[i].end_time
          const nextStart = laneEventsForLane[i + 1].start_time
          if (currentEnd < nextStart) {
            idlePeriods.push({
              start: currentEnd,
              end: nextStart,
              duration: nextStart - currentEnd,
            })
          }
        }

        // Idle at end
        const lastEvent = laneEventsForLane[laneEventsForLane.length - 1]
        if (lastEvent.end_time < intervalEnd) {
          idlePeriods.push({
            start: lastEvent.end_time,
            end: intervalEnd,
            duration: intervalEnd - lastEvent.end_time,
          })
        }
      } else {
        // Entire lane is idle
        idlePeriods.push({
          start: intervalStart,
          end: intervalEnd,
          duration: intervalDuration,
        })
      }

      return idlePeriods
    })

    return {
      laneEventsByIndex: laneEvents,
      idlePeriodsByLane: idleByLane,
    }
  }, [
    assignedEvents,
    intervalStart,
    intervalEnd,
    intervalDuration,
    actualNumLanes,
  ])

  return (
    <div>
      <div className="text-xs font-medium text-muted-foreground mb-0.5 flex items-center gap-1.5">
        <span>Server {server}</span>
        {showNodeLabel && nodeId !== null && nodeId !== undefined && (
          <span className="text-[10px] text-muted-foreground/80">
            Node {nodeId}
          </span>
        )}
      </div>
      <div className="relative">
        <div
          className="relative bg-muted/30 rounded-lg border border-border/50"
          style={{ height: totalHeight }}
        >
          {/* Render lanes */}
          {laneIndices.map((displayIdx) => {
            const actualLaneIdx = displayIdx + laneStart
            const laneTop = displayIdx * (laneHeight + 2)
            const laneEvents = laneEventsByIndex[actualLaneIdx] || []
            const idlePeriods = idlePeriodsByLane[actualLaneIdx] || []

            return (
              <div
                key={actualLaneIdx}
                className="absolute left-0 right-0"
                style={{ top: laneTop, height: laneHeight }}
              >
                {/* Idle periods for this lane */}
                {idlePeriods.map((idle, idleIdx) => {
                  // Clamp idle period to the visible interval
                  const visibleStart = Math.max(idle.start, intervalStart)
                  const visibleEnd = Math.min(idle.end, intervalEnd)
                  const visibleDuration = visibleEnd - visibleStart
                  const leftPercent =
                    ((visibleStart - intervalStart) / intervalDuration) * 100
                  const widthPercent =
                    (visibleDuration / intervalDuration) * 100

                  if (
                    widthPercent < 0.5 ||
                    leftPercent >= 100 ||
                    visibleEnd <= visibleStart
                  )
                    return null

                  const idleColor = darkMode ? IDLE_COLOR_DARK : IDLE_COLOR
                  return (
                    <HoverTooltipBlock
                      key={`idle-${idleIdx}`}
                      className="absolute top-0 bottom-0 group"
                      style={{
                        left: `${leftPercent}%`,
                        width: `${Math.min(widthPercent, 100 - leftPercent)}%`,
                        backgroundColor: idleColor,
                        borderRadius: "1px",
                      }}
                      tooltip={
                        <EventTooltip
                          title="Idle"
                          color={idleColor}
                          details={[
                            {
                              label: "Duration",
                              value: formatDuration(idle.duration * 1000),
                            },
                            { label: "Server", value: String(server) },
                            ...(displayNodeId !== undefined
                              ? [
                                  {
                                    label: "Node",
                                    value: String(displayNodeId),
                                  },
                                ]
                              : []),
                            { label: "Lane", value: String(actualLaneIdx) },
                          ]}
                        />
                      }
                    />
                  )
                })}

                {/* Events for this lane */}
                {laneEvents.map((event, eventIdx) => {
                  const envTime = event.environment_response_time
                  const rewardTime = event.compute_reward_time
                  const envStyle = getTimingBarStyle(
                    event,
                    "env",
                    selectedRequest,
                    sampleStatusByKey,
                    highlightDiscarded,
                    discardStatusReady,
                    darkMode,
                  )
                  const rewardStyle = getTimingBarStyle(
                    event,
                    "reward",
                    selectedRequest,
                    sampleStatusByKey,
                    highlightDiscarded,
                    discardStatusReady,
                    darkMode,
                  )
                  return (
                    <Fragment key={`event-${eventIdx}`}>
                      <InferenceEventBlock
                        event={event}
                        server={server}
                        nodeId={displayNodeId}
                        lane={actualLaneIdx}
                        intervalStart={intervalStart}
                        intervalDuration={intervalDuration}
                        selectedRequest={selectedRequest}
                        sampleStatusByKey={sampleStatusByKey}
                        highlightDiscarded={highlightDiscarded}
                        discardStatusReady={discardStatusReady}
                        onClick={() => onEventClick(event)}
                      />
                      {/* Environment response bar (right after inference event) */}
                      {envTime != null &&
                        envTime > 0 &&
                        (() => {
                          const intervalEnd = intervalStart + intervalDuration
                          const envStart = event.end_time
                          const envEnd = envStart + envTime
                          if (
                            envStart >= intervalEnd ||
                            envEnd <= intervalStart
                          )
                            return null
                          const visibleStart = Math.max(envStart, intervalStart)
                          const visibleEnd = Math.min(envEnd, intervalEnd)
                          const envLeftPct = Math.max(
                            0,
                            ((visibleStart - intervalStart) /
                              intervalDuration) *
                              100,
                          )
                          const envWidthPct = Math.max(
                            0.3,
                            ((visibleEnd - visibleStart) / intervalDuration) *
                              100,
                          )
                          const envDurationMs = envTime * 1000
                          return (
                            <HoverTooltipBlock
                              className="absolute top-0 bottom-0 group cursor-pointer"
                              style={{
                                left: `${envLeftPct}%`,
                                width: `${Math.min(envWidthPct, 100 - envLeftPct)}%`,
                                backgroundColor: envStyle.color,
                                borderRadius: "1px",
                                border: eventBorderMedium,
                                boxSizing: "border-box",
                                opacity: envStyle.opacity,
                              }}
                              onClick={() => onEventClick(event)}
                              tooltip={
                                <EventTooltip
                                  title="Env Response"
                                  titleSecondary={
                                    event.sample_id !== undefined
                                      ? `Sample ${event.sample_id}`
                                      : undefined
                                  }
                                  color={envStyle.color}
                                  details={[
                                    {
                                      label: "Duration",
                                      value: formatDuration(envDurationMs),
                                    },
                                    ...(event.group_id !== undefined
                                      ? [
                                          {
                                            label: "Group",
                                            value: String(event.group_id),
                                          },
                                        ]
                                      : []),
                                    { label: "Server", value: String(server) },
                                    ...(displayNodeId !== undefined
                                      ? [
                                          {
                                            label: "Node",
                                            value: String(displayNodeId),
                                          },
                                        ]
                                      : []),
                                    {
                                      label: "Lane",
                                      value: String(actualLaneIdx),
                                    },
                                  ]}
                                />
                              }
                            />
                          )
                        })()}
                      {/* Compute reward bar (after last inference event for a sample) */}
                      {showComputeReward &&
                        rewardTime != null &&
                        rewardTime > 0 &&
                        (() => {
                          const intervalEnd = intervalStart + intervalDuration
                          const rewardStart =
                            event.end_time +
                            (envTime != null && envTime > 0 ? envTime : 0)
                          const rewardEnd = rewardStart + rewardTime
                          if (
                            rewardStart >= intervalEnd ||
                            rewardEnd <= intervalStart
                          )
                            return null
                          const visibleStart = Math.max(
                            rewardStart,
                            intervalStart,
                          )
                          const visibleEnd = Math.min(rewardEnd, intervalEnd)
                          const rewardLeftPct = Math.max(
                            0,
                            ((visibleStart - intervalStart) /
                              intervalDuration) *
                              100,
                          )
                          const rewardWidthPct = Math.max(
                            0.3,
                            ((visibleEnd - visibleStart) / intervalDuration) *
                              100,
                          )
                          const rewardDurationMs = rewardTime * 1000
                          // When freeLaneAfterGeneration, compute reward can overlap
                          // with next generation. Show it with smaller height,
                          // vertically centered, on top (higher z-index).
                          const rewardHeight = freeLaneAfterGeneration
                            ? Math.max(3, Math.round(laneHeight * 0.4))
                            : undefined
                          const rewardTop = freeLaneAfterGeneration
                            ? Math.round((laneHeight - rewardHeight!) / 2)
                            : undefined
                          return (
                            <HoverTooltipBlock
                              className="absolute group cursor-pointer"
                              style={{
                                left: `${rewardLeftPct}%`,
                                width: `${Math.min(rewardWidthPct, 100 - rewardLeftPct)}%`,
                                backgroundColor: rewardStyle.color,
                                borderRadius: "1px",
                                border: eventBorderMedium,
                                boxSizing: "border-box",
                                opacity: rewardStyle.opacity,
                                ...(freeLaneAfterGeneration
                                  ? { top: rewardTop, height: rewardHeight, zIndex: 2 }
                                  : { top: 0, bottom: 0 }),
                              }}
                              onClick={() => onEventClick(event)}
                              tooltip={
                                <EventTooltip
                                  title={event.is_eval ? "Compute Metrics" : "Compute Reward"}
                                  titleSecondary={
                                    event.sample_id !== undefined
                                      ? `Sample ${event.sample_id}`
                                      : undefined
                                  }
                                  color={rewardStyle.color}
                                  details={[
                                    {
                                      label: "Duration",
                                      value: formatDuration(rewardDurationMs),
                                    },
                                    ...(event.group_id !== undefined
                                      ? [
                                          {
                                            label: "Group",
                                            value: String(event.group_id),
                                          },
                                        ]
                                      : []),
                                    { label: "Server", value: String(server) },
                                    ...(displayNodeId !== undefined
                                      ? [
                                          {
                                            label: "Node",
                                            value: String(displayNodeId),
                                          },
                                        ]
                                      : []),
                                    {
                                      label: "Lane",
                                      value: String(actualLaneIdx),
                                    },
                                  ]}
                                />
                              }
                            />
                          )
                        })()}
                    </Fragment>
                  )
                })}
              </div>
            )
          })}

          {/* Weight broadcast events - span all lanes */}
          {showWeightUpdate &&
            weightBroadcastEvents.map((event, eventIdx) => {
              const relativeStart = event.start_time - intervalStart
              const leftPercent = Math.max(
                0,
                (relativeStart / intervalDuration) * 100,
              )
              const duration = event.end_time - event.start_time
              const durationMs = duration * 1000
              const visibleStart = Math.max(event.start_time, intervalStart)
              const visibleDuration = event.end_time - visibleStart
              const widthPercent = Math.max(
                0.5,
                (visibleDuration / intervalDuration) * 100,
              )
              const color = getInferenceEventColor(event.event_type)

              return (
                <HoverTooltipBlock
                  key={`wb-${eventIdx}`}
                  className="absolute group cursor-default"
                  style={{
                    top: 0,
                    left: `${leftPercent}%`,
                    width: `${Math.min(widthPercent, 100 - leftPercent)}%`,
                    height: totalHeight,
                    backgroundColor: color,
                    borderRadius: "2px",
                    border: eventBorderWide,
                    boxSizing: "border-box",
                    opacity: 0.6,
                  }}
                  tooltip={
                    <EventTooltip
                      title="Weight Broadcast"
                      color={color}
                      details={[
                        ...(event.step != null
                          ? [{ label: "Step", value: String(event.step) }]
                          : []),
                        {
                          label: "Duration",
                          value: formatDuration(durationMs),
                        },
                        { label: "Server", value: String(server) },
                        ...(displayNodeId !== undefined
                          ? [{ label: "Node", value: String(displayNodeId) }]
                          : []),
                      ]}
                      startTime={event.start_time}
                      endTime={event.end_time}
                    />
                  }
                />
              )
            })}
        </div>
      </div>
    </div>
  )
}

// Helper to get the base color for an inference event (eval = green, training = blue)
function getInferenceEventBaseColor(
  event: InferenceEvent,
  sampleStatusByKey?: Map<string, "rollouts" | "rollouts_discarded" | null>,
  highlightDiscarded: boolean = true,
  discardStatusReady: boolean = true,
  darkMode: boolean = false,
): string {
  if (event.event_type !== "request") {
    return getInferenceEventColor(event.event_type)
  }
  // Inflight (in-progress) events use yellow to distinguish from completed (blue)
  if (event.phase === "inflight") {
    return darkMode ? "rgba(250, 204, 21, 0.5)" : "rgba(234, 179, 8, 0.5)" // yellow-400/yellow-500 at 50% opacity
  }
  if (event.is_canceled) {
    return darkMode ? INFERENCE_REQUEST_CANCELED_COLOR_DARK : INFERENCE_REQUEST_CANCELED_COLOR
  }
  if (!highlightDiscarded) {
    return event.is_eval
      ? INFERENCE_REQUEST_EVAL_COLOR
      : INFERENCE_REQUEST_COLOR
  }
  // Status not loaded yet — show darker yellow ("done but unknown status")
  if (!discardStatusReady) {
    return darkMode ? "rgba(202, 138, 4, 0.6)" : "rgba(202, 138, 4, 0.7)" // yellow-600
  }
  if (!sampleStatusByKey || event.sample_id == null || event.group_id == null) {
    return event.is_eval
      ? INFERENCE_REQUEST_EVAL_COLOR
      : INFERENCE_REQUEST_COLOR
  }
  const key = `${event.group_id}:${event.sample_id}`
  const status = sampleStatusByKey.get(key)
  if (status === "rollouts_discarded") {
    return darkMode ? INFERENCE_REQUEST_DISCARDED_COLOR_DARK : INFERENCE_REQUEST_DISCARDED_COLOR
  }
  // Done but not yet categorized as kept or discarded — darker yellow
  if (status == null) {
    return darkMode ? "rgba(202, 138, 4, 0.6)" : "rgba(202, 138, 4, 0.7)" // yellow-600
  }
  return event.is_eval ? INFERENCE_REQUEST_EVAL_COLOR : INFERENCE_REQUEST_COLOR
}

function getInferenceEventSelectionColor(
  event: InferenceEvent,
  selectedRequest: SelectedInferenceRequest | null,
  sampleStatusByKey?: Map<string, "rollouts" | "rollouts_discarded" | null>,
  highlightDiscarded: boolean = true,
  discardStatusReady: boolean = true,
  darkMode: boolean = false,
): { color: string; opacity: number } {
  const defaultColor = getInferenceEventBaseColor(
    event,
    sampleStatusByKey,
    highlightDiscarded,
    discardStatusReady,
    darkMode,
  )

  if (!selectedRequest) {
    return { color: defaultColor, opacity: 1 }
  }

  const isDiscarded = defaultColor === INFERENCE_REQUEST_DISCARDED_COLOR || defaultColor === INFERENCE_REQUEST_DISCARDED_COLOR_DARK
  const isCanceled = defaultColor === INFERENCE_REQUEST_CANCELED_COLOR || defaultColor === INFERENCE_REQUEST_CANCELED_COLOR_DARK
  const isInflightOrPending = event.phase === "inflight" || defaultColor.includes("202, 138, 4") || defaultColor.includes("234, 179, 8") || defaultColor.includes("250, 204, 21")

  // Selected sample = darker version of base color
  if (event.sample_id === selectedRequest.sampleId) {
    if (isCanceled) {
      return { color: darkMode ? GROUP_SAMPLE_CANCELED_SELECTED_COLOR_DARK : GROUP_SAMPLE_CANCELED_SELECTED_COLOR_LIGHT, opacity: 1 }
    }
    if (isDiscarded) {
      return { color: darkMode ? "#9ca3af" : "#6b7280", opacity: 1 }
    }
    if (isInflightOrPending) {
      return { color: darkMode ? "rgba(161, 98, 7, 0.8)" : "rgba(161, 98, 7, 0.9)", opacity: 1 } // yellow-700
    }
    const selectedColor = event.is_eval
      ? INFERENCE_REQUEST_EVAL_SELECTED_COLOR
      : INFERENCE_REQUEST_SELECTED_COLOR
    return { color: selectedColor, opacity: 1 }
  }

  // Same group_id but different sample_id
  if (event.group_id === selectedRequest.groupId) {
    if (isCanceled) {
      return { color: darkMode ? GROUP_SAMPLE_CANCELED_COLOR_DARK : GROUP_SAMPLE_CANCELED_COLOR_LIGHT, opacity: 1 }
    }
    if (isDiscarded) {
      return { color: darkMode ? GROUP_SAMPLE_DISCARDED_COLOR_DARK : GROUP_SAMPLE_DISCARDED_COLOR_LIGHT, opacity: 1 }
    }
    if (isInflightOrPending) {
      return { color: darkMode ? "rgba(202, 138, 4, 0.7)" : "rgba(202, 138, 4, 0.8)", opacity: 1 } // yellow-600
    }
    return {
      color: event.is_eval ? GROUP_SAMPLE_EVAL_COLOR : GROUP_SAMPLE_COLOR,
      opacity: 1,
    }
  }

  // Other events when something is selected = slightly dimmed
  return { color: defaultColor, opacity: 0.55 }
}

/** Get color + opacity for env-response / compute-reward bars,
 *  taking selection state and discarded status into account. */
function getTimingBarStyle(
  event: InferenceEvent,
  kind: "env" | "reward",
  selectedRequest: SelectedInferenceRequest | null,
  sampleStatusByKey: Map<string, "rollouts" | "rollouts_discarded" | null>,
  highlightDiscarded: boolean,
  discardStatusReady: boolean,
  darkMode: boolean = false,
): { color: string; opacity: number } {
  const isEval = kind === "reward" && event.is_eval
  // Determine if the sample is discarded
  const isDiscarded = (() => {
    if (!highlightDiscarded) return false
    if (!discardStatusReady) return true // assume discarded until we know
    if (event.sample_id == null || event.group_id == null) return false
    const key = `${event.group_id}:${event.sample_id}`
    return sampleStatusByKey.get(key) === "rollouts_discarded"
  })()

  const baseColor = isDiscarded
    ? kind === "env"
      ? (darkMode ? ENV_RESPONSE_DISCARDED_COLOR_DARK : ENV_RESPONSE_DISCARDED_COLOR_LIGHT)
      : isEval
        ? (darkMode ? COMPUTE_METRICS_DISCARDED_COLOR_DARK : COMPUTE_METRICS_DISCARDED_COLOR_LIGHT)
        : (darkMode ? COMPUTE_REWARD_DISCARDED_COLOR_DARK : COMPUTE_REWARD_DISCARDED_COLOR_LIGHT)
    : kind === "env"
      ? ENV_RESPONSE_COLOR
      : isEval
        ? COMPUTE_METRICS_COLOR
        : COMPUTE_REWARD_COLOR

  if (!selectedRequest) return { color: baseColor, opacity: 0.85 }

  // Selected sample: darker version of base color (or darker gray if discarded)
  if (event.sample_id === selectedRequest.sampleId) {
    if (isDiscarded) {
      return { color: darkMode ? "#9ca3af" : "#6b7280", opacity: 1 }
    }
    const selectedColor =
      kind === "env"
        ? ENV_RESPONSE_SELECTED_COLOR
        : isEval
          ? COMPUTE_METRICS_SELECTED_COLOR
          : COMPUTE_REWARD_SELECTED_COLOR
    return { color: selectedColor, opacity: 1 }
  }

  // Same group: use darker color variants to match the group darkening
  if (event.group_id === selectedRequest.groupId) {
    const groupColor = isDiscarded
      ? kind === "env"
        ? (darkMode ? ENV_RESPONSE_GROUP_DISCARDED_COLOR_DARK : ENV_RESPONSE_GROUP_DISCARDED_COLOR_LIGHT)
        : isEval
          ? (darkMode ? COMPUTE_METRICS_GROUP_DISCARDED_COLOR_DARK : COMPUTE_METRICS_GROUP_DISCARDED_COLOR_LIGHT)
          : (darkMode ? COMPUTE_REWARD_GROUP_DISCARDED_COLOR_DARK : COMPUTE_REWARD_GROUP_DISCARDED_COLOR_LIGHT)
      : kind === "env"
        ? ENV_RESPONSE_GROUP_COLOR
        : isEval
          ? COMPUTE_METRICS_GROUP_COLOR
          : COMPUTE_REWARD_GROUP_COLOR
    return { color: groupColor, opacity: 1 }
  }

  // Other events when something is selected = dimmed
  return { color: baseColor, opacity: 0.35 }
}

function InferenceEventBlock({
  event,
  server,
  nodeId,
  lane,
  intervalStart,
  intervalDuration,
  selectedRequest,
  sampleStatusByKey,
  highlightDiscarded,
  discardStatusReady,
  onClick,
}: {
  event: InferenceEvent
  server: number
  nodeId?: number | null
  lane: number
  intervalStart: number
  intervalDuration: number
  selectedRequest: SelectedInferenceRequest | null
  sampleStatusByKey: Map<string, "rollouts" | "rollouts_discarded" | null>
  highlightDiscarded: boolean
  discardStatusReady: boolean
  onClick: () => void
}) {
  const darkMode = useAtomValue(darkModeAtom)
  const intervalEnd = intervalStart + intervalDuration

  if (event.end_time <= intervalStart || event.start_time >= intervalEnd)
    return null

  const relativeStart = event.start_time - intervalStart
  const leftPercent = Math.max(0, (relativeStart / intervalDuration) * 100)

  const duration = event.end_time - event.start_time
  const durationMs = duration * 1000

  // Calculate visible width
  const visibleStart = Math.max(event.start_time, intervalStart)
  const visibleDuration = event.end_time - visibleStart
  const widthPercent = (visibleDuration / intervalDuration) * 100

  const { color, opacity } = getInferenceEventSelectionColor(
    event,
    selectedRequest,
    sampleStatusByKey,
    highlightDiscarded,
    discardStatusReady,
    darkMode,
  )
  const isClickable =
    event.event_type === "request" && event.sample_id !== undefined

  // Determine status label for tooltip
  const statusLabel = (() => {
    if (event.event_type !== "request") return null
    if (event.phase === "inflight")
      return { text: "In Progress", className: "text-yellow-500" }
    if (event.is_canceled)
      return { text: "Canceled", className: "text-muted-foreground" }
    if (
      highlightDiscarded &&
      event.sample_id != null &&
      event.group_id != null
    ) {
      if (!discardStatusReady) {
        return { text: "Pending Status", className: "text-yellow-600" }
      }
      const key = `${event.group_id}:${event.sample_id}`
      const status = sampleStatusByKey.get(key)
      if (status === "rollouts_discarded") {
        return { text: "Discarded", className: "text-red-700" }
      }
      if (status == null) {
        return { text: "Finished (Waiting for Group)", className: "text-yellow-600" }
      }
    }
    return null
  })()

  return (
    <HoverTooltipBlock
      className={`absolute top-0 bottom-0 group ${
        isClickable ? "cursor-pointer" : "cursor-default"
      }`}
      style={{
        left: `${leftPercent}%`,
        width: `${Math.min(widthPercent, 100 - leftPercent)}%`,
        minWidth: "1px",
        backgroundColor: color,
        borderRadius: "1px",
        border: event.phase === "inflight"
          ? (darkMode ? "1.5px dashed rgba(250, 204, 21, 0.7)" : "1.5px dashed rgba(234, 179, 8, 0.7)")
          : event.is_canceled
            ? (darkMode ? "1.5px solid rgba(255, 255, 255, 0.15)" : "1.5px solid rgba(0, 0, 0, 0.2)")
            : (darkMode ? "1.5px solid rgba(255, 255, 255, 0.2)" : "1.5px solid rgba(0, 0, 0, 0.3)"),
        boxSizing: "border-box",
        opacity,
      }}
      onClick={isClickable ? onClick : undefined}
      tooltip={
        <EventTooltip
          title={
            event.event_type === "request"
              ? event.sample_id !== undefined
                ? `Sample ${event.sample_id}`
                : event.group_id !== undefined
                  ? `Group ${event.group_id}`
                  : "Request"
              : event.event_type
          }
          titleSecondary={
            event.event_type === "request" && event.group_id !== undefined
              ? `Group ${event.group_id}`
              : null
          }
          color={color}
          capitalize
          statusLabel={statusLabel}
          details={
            event.event_type === "request"
              ? buildInferenceRequestDetails(event, durationMs, [
                  { label: "Server", value: String(server) },
                  ...(nodeId !== null && nodeId !== undefined
                    ? [{ label: "Node", value: String(nodeId) }]
                    : []),
                  { label: "Lane", value: String(lane) },
                ])
              : [
                  { label: "Duration", value: formatDuration(durationMs) },
                  { label: "Server", value: String(server) },
                  ...(nodeId !== null && nodeId !== undefined
                    ? [{ label: "Node", value: String(nodeId) }]
                    : []),
                  { label: "Lane", value: String(lane) },
                ]
          }
          startTime={event.start_time}
          endTime={event.end_time}
        />
      }
    />
  )
}

// ============================================================================
// Trainer Section
// ============================================================================

function TrainerSection({
  ranks,
  eventsByRank,
  intervalStart,
  intervalDuration,
  intervalEnd,
  trainerGpuMetricsByGpuIndex,
  trainerGpuMetricsByRank,
  trainerGpuMetricsIsLoading,
  trainerAvailableGpuMetrics,
  selectedMetricNames,
  onSelectedMetricNamesChange,
  trainerSystemGpuIndices,
  trainerRankInfoByRank,
  totalSetupNodes,
  isLoading,
}: {
  ranks: number[]
  eventsByRank: Record<number, TrainerEvent[]>
  intervalStart: number
  intervalDuration: number
  intervalEnd: number
  trainerGpuMetricsByGpuIndex?: Record<number, GpuMetric[]>
  trainerGpuMetricsByRank?: Record<number, GpuMetric[]>
  trainerGpuMetricsIsLoading?: boolean
  trainerAvailableGpuMetrics?: string[]
  selectedMetricNames?: string[]
  onSelectedMetricNamesChange?: (metricNames: string[]) => void
  trainerSystemGpuIndices?: number[]
  trainerRankInfoByRank?: Record<
    number,
    { node_id: number | null; local_rank: number | null }
  >
  totalSetupNodes?: number
  isLoading?: boolean
}) {
  const [isOpen, setIsOpen] = useState(true)
  const [selectedEvent, setSelectedEvent] = useAtom(selectedTrainerEventAtom)
  const [gpuPage, setGpuPage] = useAtom(trainerGpuPageAtom)

  const availableTrainerGpuMetrics = useMemo(() => {
    const deduped = new Set<string>()
    ;(trainerAvailableGpuMetrics ?? []).forEach((metricName) => {
      if (typeof metricName === "string") deduped.add(metricName)
    })
    return Array.from(deduped)
  }, [trainerAvailableGpuMetrics])

  const [cachedTrainerGpuMetrics, setCachedTrainerGpuMetrics] = useState<
    string[]
  >([])

  const noGpuIndices = (trainerSystemGpuIndices?.length ?? 0) === 0
  if (noGpuIndices && cachedTrainerGpuMetrics.length > 0) {
    setCachedTrainerGpuMetrics([])
  } else if (
    !noGpuIndices &&
    availableTrainerGpuMetrics.length > 0 &&
    cachedTrainerGpuMetrics !== availableTrainerGpuMetrics
  ) {
    setCachedTrainerGpuMetrics(availableTrainerGpuMetrics)
  }

  const stableAvailableTrainerGpuMetrics = useMemo(() => {
    if (noGpuIndices) return []
    if (availableTrainerGpuMetrics.length > 0) return availableTrainerGpuMetrics
    return cachedTrainerGpuMetrics
  }, [noGpuIndices, availableTrainerGpuMetrics, cachedTrainerGpuMetrics])

  const normalizedSelectedMetricNames = useMemo(() => {
    return Array.from(
      new Set(
        (selectedMetricNames ?? []).filter(
          (metricName) =>
            typeof metricName === "string" && metricName.trim().length > 0,
        ),
      ),
    )
  }, [selectedMetricNames])

  const effectiveSelectedMetricNames = useMemo(() => {
    if (stableAvailableTrainerGpuMetrics.length === 0) {
      return normalizedSelectedMetricNames
    }
    return normalizedSelectedMetricNames.filter((metricName) =>
      stableAvailableTrainerGpuMetrics.includes(metricName),
    )
  }, [stableAvailableTrainerGpuMetrics, normalizedSelectedMetricNames])

  const configuredRanks = useMemo(() => {
    const configuredFromRankInfo = Object.keys(trainerRankInfoByRank ?? {})
      .map(Number)
      .filter((rank) => Number.isFinite(rank))
      .sort((a, b) => a - b)
    if (configuredFromRankInfo.length > 0) {
      return configuredFromRankInfo
    }
    if (!trainerSystemGpuIndices || trainerSystemGpuIndices.length === 0) {
      return []
    }
    return trainerSystemGpuIndices.map((_, idx) => idx)
  }, [trainerRankInfoByRank, trainerSystemGpuIndices])

  const displayRanks = useMemo(() => {
    const set = new Set<number>()
    ranks.forEach((r) => set.add(r))
    configuredRanks.forEach((r) => set.add(r))
    return Array.from(set).sort((a, b) => a - b)
  }, [ranks, configuredRanks])

  // GPU page pagination (pages of 8 GPUs)
  const GPU_PAGE_SIZE = 8
  const showGpuPagination = displayRanks.length > GPU_PAGE_SIZE
  const gpuPages = useMemo(() => {
    if (!showGpuPagination) return []
    const pages: { label: string; ranks: number[] }[] = []
    for (let i = 0; i < displayRanks.length; i += GPU_PAGE_SIZE) {
      const pageRanks = displayRanks.slice(i, i + GPU_PAGE_SIZE)
      const start = i + 1
      const end = Math.min(i + GPU_PAGE_SIZE, displayRanks.length)
      pages.push({ label: `${start} – ${end}`, ranks: pageRanks })
    }
    return pages
  }, [displayRanks, showGpuPagination])
  const clampedGpuPage = showGpuPagination
    ? Math.min(gpuPage, Math.max(0, gpuPages.length - 1))
    : 0
  useEffect(() => {
    if (showGpuPagination && gpuPage !== clampedGpuPage) {
      setGpuPage(clampedGpuPage)
    }
  }, [showGpuPagination, gpuPage, clampedGpuPage, setGpuPage])
  const paginatedRanks = showGpuPagination
    ? gpuPages[clampedGpuPage]?.ranks ?? displayRanks.slice(0, GPU_PAGE_SIZE)
    : displayRanks
  const paginatedRankSet = useMemo(
    () => new Set(paginatedRanks),
    [paginatedRanks],
  )

  const childEventKeysByRank = useMemo(() => {
    const map = new Map<number, Set<string>>()
    Object.entries(eventsByRank).forEach(([rankKey, rankEvents]) => {
      const set = new Set<string>()
      rankEvents.forEach((event) => {
        if (!event.event_type.includes("/")) return
        const parentType = event.event_type.split("/")[0]
        set.add(`${parentType}|${event.step}`)
      })
      map.set(Number(rankKey), set)
    })
    return map
  }, [eventsByRank])

  const groupedRanksByNode = useMemo(() => {
    const grouped = new Map<
      string,
      { nodeId: number | null; ranks: number[] }
    >()
    paginatedRanks.forEach((rank) => {
      const nodeId = trainerRankInfoByRank?.[rank]?.node_id ?? null
      const key = nodeId === null ? "__unknown__" : String(nodeId)
      if (!grouped.has(key)) {
        grouped.set(key, { nodeId, ranks: [] })
      }
      grouped.get(key)!.ranks.push(rank)
    })
    return Array.from(grouped.values())
      .map((group) => ({
        nodeId: group.nodeId,
        ranks: group.ranks.sort((a, b) => a - b),
      }))
      .sort((a, b) => {
        if (a.nodeId === null && b.nodeId === null) return 0
        if (a.nodeId === null) return 1
        if (b.nodeId === null) return -1
        return a.nodeId - b.nodeId
      })
  }, [paginatedRanks, trainerRankInfoByRank])

  const hasMultipleNodes = useMemo(() => {
    if (
      typeof totalSetupNodes === "number" &&
      Number.isFinite(totalSetupNodes)
    ) {
      return totalSetupNodes > 1
    }
    const knownNodes = groupedRanksByNode.filter(
      (group) => group.nodeId !== null,
    )
    return knownNodes.length > 1
  }, [groupedRanksByNode, totalSetupNodes])

  const handleEventClick = useCallback(
    (event: TrainerEvent) => {
      // Child events can't be expanded
      if (event.event_type.includes("/")) return

      // Check if this event has children (events with this type as prefix)
      const hasChildren =
        childEventKeysByRank
          .get(event.rank)
          ?.has(`${event.event_type}|${event.step}`) ?? false

      if (!hasChildren) return // Only expandable events with children can be clicked

      setSelectedEvent((prev) => {
        // Toggle off if clicking the same event
        if (
          prev?.eventType === event.event_type &&
          prev?.rank === event.rank &&
          prev?.step === event.step
        ) {
          return null
        }
        return {
          eventType: event.event_type,
          rank: event.rank,
          step: event.step,
        }
      })
    },
    [childEventKeysByRank, setSelectedEvent],
  )

  const toggleMetricSelection = useCallback(
    (metricName: string, checked: boolean | "indeterminate") => {
      const base = effectiveSelectedMetricNames
      const shouldSelect = checked === true
      const next = shouldSelect
        ? base.includes(metricName)
          ? base
          : [...base, metricName]
        : base.filter((name) => name !== metricName)
      onSelectedMetricNamesChange?.(next)
    },
    [effectiveSelectedMetricNames, onSelectedMetricNamesChange],
  )

  const handleSelectAllMetrics = useCallback(
    () => onSelectedMetricNamesChange?.([...stableAvailableTrainerGpuMetrics]),
    [stableAvailableTrainerGpuMetrics, onSelectedMetricNamesChange],
  )

  const handleUnselectAllMetrics = useCallback(
    () => onSelectedMetricNamesChange?.([]),
    [onSelectedMetricNamesChange],
  )

  const renderRankTimeline = useCallback(
    (rank: number) => {
      const rankInfo = trainerRankInfoByRank?.[rank]
      const systemGpuIndex =
        rankInfo?.local_rank ??
        (trainerSystemGpuIndices ? trainerSystemGpuIndices[rank] : undefined)
      const gpuMetrics =
        trainerGpuMetricsByRank?.[rank] ??
        (systemGpuIndex !== undefined && trainerGpuMetricsByGpuIndex
          ? (trainerGpuMetricsByGpuIndex[systemGpuIndex] ?? [])
          : undefined)

      return (
        <GpuRankTimeline
          key={rank}
          rank={rank}
          nodeId={rankInfo?.node_id ?? null}
          showNodeLabel={hasMultipleNodes}
          events={eventsByRank[rank] || []}
          intervalStart={intervalStart}
          intervalDuration={intervalDuration}
          intervalEnd={intervalEnd}
          selectedEvent={selectedEvent}
          onEventClick={handleEventClick}
          childEventKeys={
            childEventKeysByRank.get(rank) ?? EMPTY_CHILD_EVENT_KEYS
          }
          systemGpuIndex={systemGpuIndex}
          gpuMetrics={gpuMetrics}
          selectedMetricNames={effectiveSelectedMetricNames}
          gpuMetricsIsLoading={trainerGpuMetricsIsLoading ?? false}
        />
      )
    },
    [
      childEventKeysByRank,
      effectiveSelectedMetricNames,
      eventsByRank,
      handleEventClick,
      intervalDuration,
      intervalEnd,
      intervalStart,
      selectedEvent,
      trainerGpuMetricsByGpuIndex,
      trainerGpuMetricsByRank,
      trainerGpuMetricsIsLoading,
      trainerRankInfoByRank,
      trainerSystemGpuIndices,
      hasMultipleNodes,
    ],
  )

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="flex items-center gap-2 mb-3">
        <CollapsibleTrigger asChild>
          <div className="flex items-center gap-1.5 py-1.5 px-2 -ml-2 cursor-pointer hover:bg-muted rounded transition-colors">
            <ChevronDown
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform",
                !isOpen && "-rotate-90",
              )}
            />
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              Trainer GPUs
            </h3>
            <div className="w-6 flex items-center justify-center shrink-0">
              {isLoading && (
                <Spinner className="size-3.5 text-muted-foreground" />
              )}
            </div>
          </div>
        </CollapsibleTrigger>
        {isOpen && (
          <>
          {showGpuPagination && (
            <Select
              value={String(clampedGpuPage)}
              onValueChange={(v) => setGpuPage(Number(v))}
            >
              <SelectTrigger size="sm" className="text-xs h-7 gap-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {gpuPages.map((page, idx) => (
                  <SelectItem key={idx} value={String(idx)}>
                    GPUs {page.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded-md border border-input bg-transparent px-2 h-7 text-xs hover:bg-accent transition-colors"
              >
                Plots
                <span className="text-[10px] font-medium rounded bg-accent text-accent-foreground px-1.5 py-0.5">
                  {effectiveSelectedMetricNames.length}
                </span>
                <ChevronDown className="h-3 w-3 text-muted-foreground" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-64">
              <DropdownMenuLabel>Trainer GPU Plots</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onSelect={(event) => {
                  event.preventDefault()
                  handleSelectAllMetrics()
                }}
                className="text-xs"
              >
                Select All
              </DropdownMenuItem>
              <DropdownMenuItem
                onSelect={(event) => {
                  event.preventDefault()
                  handleUnselectAllMetrics()
                }}
                className="text-xs"
              >
                Unselect All
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              {stableAvailableTrainerGpuMetrics.length > 0 ? (
                stableAvailableTrainerGpuMetrics.map((metricName) => (
                  <DropdownMenuCheckboxItem
                    key={metricName}
                    checked={effectiveSelectedMetricNames.includes(metricName)}
                    onCheckedChange={(checked) =>
                      toggleMetricSelection(metricName, checked)
                    }
                    onSelect={(event) => event.preventDefault()}
                    className="text-xs"
                  >
                    {getGpuMetricDisplayName(metricName)}
                  </DropdownMenuCheckboxItem>
                ))
              ) : (
                <DropdownMenuItem
                  disabled
                  className="text-xs text-muted-foreground"
                >
                  {trainerGpuMetricsIsLoading
                    ? "Loading plots..."
                    : "No plots available"}
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
          </>
        )}
      </div>
      <CollapsibleContent>
        <div className="space-y-4">
          {paginatedRanks.length > 0 ? (
            hasMultipleNodes ? (
              groupedRanksByNode.map((group) => (
                <TrainerNodeGroup
                  key={
                    group.nodeId === null ? "__unknown__" : String(group.nodeId)
                  }
                  nodeId={group.nodeId}
                  rankCount={group.ranks.length}
                >
                  {group.ranks.map((rank) => renderRankTimeline(rank))}
                </TrainerNodeGroup>
              ))
            ) : (
              paginatedRanks.map((rank) => renderRankTimeline(rank))
            )
          ) : (
            <div className="h-24 flex items-center justify-center text-muted-foreground bg-muted/30 rounded-lg border border-border/50">
              No trainer events in this interval
            </div>
          )}
        </div>

        {/* Time axis at bottom of main timeline */}
        <div className="flex justify-between text-xs text-muted-foreground mt-2">
          <span>{formatTimeShort(0)}</span>
          <span>{formatTimeShort(intervalDuration / 2)}</span>
          <span>{formatTimeShort(intervalDuration)}</span>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function TrainerNodeGroup({
  nodeId,
  rankCount,
  children,
}: {
  nodeId: number | null
  rankCount: number
  children: ReactNode
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
                !isOpen && "-rotate-90",
              )}
            />
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {nodeLabel}
            </span>
            <span className="text-[11px] text-muted-foreground/80">
              {rankCount} GPU{rankCount !== 1 ? "s" : ""}
            </span>
          </div>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="space-y-4 mt-2">{children}</div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function GpuRankTimeline({
  rank,
  nodeId,
  showNodeLabel,
  events,
  intervalStart,
  intervalDuration,
  intervalEnd,
  selectedEvent,
  onEventClick,
  childEventKeys,
  systemGpuIndex,
  gpuMetrics,
  selectedMetricNames,
  gpuMetricsIsLoading,
}: {
  rank: number
  nodeId?: number | null
  showNodeLabel?: boolean
  events: TrainerEvent[]
  intervalStart: number
  intervalDuration: number
  intervalEnd: number
  selectedEvent: SelectedTrainerEvent | null
  onEventClick: (event: TrainerEvent) => void
  childEventKeys: Set<string>
  systemGpuIndex?: number
  gpuMetrics?: GpuMetric[]
  selectedMetricNames: string[]
  gpuMetricsIsLoading?: boolean
}) {
  const sortedEvents = useMemo(() => {
    const rootEvents = events.filter((e) => !e.event_type.includes("/"))
    return [...rootEvents].sort((a, b) => a.start_time - b.start_time)
  }, [events])

  const idlePeriods = useMemo(
    () => calculateIdlePeriods(sortedEvents, intervalStart, intervalEnd),
    [sortedEvents, intervalStart, intervalEnd],
  )

  const gpuMetricsByName = useMemo(() => {
    if (!gpuMetrics || gpuMetrics.length === 0) return {}
    const grouped: Record<string, GpuMetric[]> = {}
    gpuMetrics.forEach((metric) => {
      if (!grouped[metric.metric_name]) grouped[metric.metric_name] = []
      grouped[metric.metric_name].push(metric)
    })
    return grouped
  }, [gpuMetrics])

  const metricGpuIndices = useMemo(() => {
    const indices = new Set<number>()
    gpuMetrics?.forEach((metric) => {
      if (Number.isFinite(metric.gpu_index)) {
        indices.add(metric.gpu_index)
      }
    })
    if (indices.size > 0) {
      return Array.from(indices).sort((a, b) => a - b)
    }
    if (systemGpuIndex !== undefined) {
      return [systemGpuIndex]
    }
    return []
  }, [gpuMetrics, systemGpuIndex])

  const rankHeaderLabel =
    showNodeLabel && nodeId !== null && nodeId !== undefined
      ? `Rank ${rank} Node ${nodeId}`
      : `Rank ${rank}`

  const gpuDisplayNames = useMemo(() => {
    if (metricGpuIndices.length === 0) return undefined
    const labels: Record<number, string> = {}
    metricGpuIndices.forEach((idx) => {
      labels[idx] = rankHeaderLabel
    })
    return labels
  }, [metricGpuIndices, rankHeaderLabel])

  return (
    <div>
      <div className="text-xs font-medium text-muted-foreground mb-0.5 flex items-center gap-1.5">
        <span>Rank {rank}</span>
        {showNodeLabel && nodeId !== null && nodeId !== undefined && (
          <span className="text-[10px] text-muted-foreground/80">
            Node {nodeId}
          </span>
        )}
      </div>
      <div className="relative">
        <div className="relative h-12 bg-muted/30 rounded-lg border border-border/50">
          {/* Idle periods */}
          {idlePeriods.map((idle, idx) => (
            <IdleBlock
              key={`idle-${idx}`}
              idle={idle}
              rank={rank}
              intervalStart={intervalStart}
              intervalDuration={intervalDuration}
              dimmed={selectedEvent !== null}
            />
          ))}

          {/* Events */}
          {sortedEvents.map((event, idx) => {
            const isSelected =
              selectedEvent?.eventType === event.event_type &&
              selectedEvent?.rank === event.rank &&
              selectedEvent?.step === event.step
            const hasChildren = childEventKeys.has(
              `${event.event_type}|${event.step}`,
            )
            const isDimmed = selectedEvent !== null && !isSelected

            return (
              <TrainerEventBlock
                key={`${event.start_time}-${event.rank}-${idx}`}
                event={event}
                intervalStart={intervalStart}
                intervalDuration={intervalDuration}
                isSelected={isSelected}
                isDimmed={isDimmed}
                hasChildren={hasChildren}
                onClick={() => onEventClick(event)}
              />
            )
          })}
        </div>

        {/* Selected GPU metric charts */}
        {metricGpuIndices.length > 0 &&
          selectedMetricNames.map((metricName) => (
            <div
              key={metricName}
              className={`mt-2 transition-opacity duration-200 ${
                gpuMetricsIsLoading ? "opacity-50" : ""
              }`}
            >
              <div className="text-xs font-medium text-muted-foreground mb-0.5">
                {getGpuMetricDisplayName(metricName)}
              </div>
              <GpuMetricChart
                metricName={metricName}
                data={gpuMetricsByName[metricName] ?? []}
                gpuIndices={metricGpuIndices}
                intervalStart={intervalStart}
                intervalEnd={intervalEnd}
                variant="timeline"
                strokeColor="#ef4444"
                gpuDisplayNames={gpuDisplayNames}
                isLoading={gpuMetricsIsLoading}
              />
            </div>
          ))}
      </div>
    </div>
  )
}

function IdleBlock({
  idle,
  rank,
  intervalStart,
  intervalDuration,
  dimmed = false,
}: {
  idle: { start: number; end: number; duration: number }
  rank: number
  intervalStart: number
  intervalDuration: number
  dimmed?: boolean
}) {
  const darkMode = useAtomValue(darkModeAtom)
  const idleColor = darkMode ? IDLE_COLOR_DARK : IDLE_COLOR
  const relativeStart = idle.start - intervalStart
  const leftPercent = (relativeStart / intervalDuration) * 100
  const widthPercent = (idle.duration / intervalDuration) * 100

  if (widthPercent < 0.5 || leftPercent < 0 || leftPercent >= 100) return null

  return (
    <HoverTooltipBlock
      className="absolute top-1 bottom-1 group cursor-pointer transition-opacity duration-200"
      style={{
        left: `${leftPercent}%`,
        width: `${Math.min(widthPercent, 100 - leftPercent)}%`,
        backgroundColor: idleColor,
        borderRadius: "2px",
        opacity: dimmed ? 0.3 : 1,
      }}
      tooltip={
        <EventTooltip
          title="Idle"
          color={idleColor}
          details={[
            {
              label: "Duration",
              value: formatDuration(idle.duration * 1000),
            },
            { label: "Rank", value: String(rank) },
          ]}
          startTime={idle.start}
          endTime={idle.end}
        />
      }
    />
  )
}

function TrainerEventBlock({
  event,
  intervalStart,
  intervalDuration,
  isSelected = false,
  isDimmed = false,
  hasChildren = false,
  onClick,
}: {
  event: TrainerEvent
  intervalStart: number
  intervalDuration: number
  isSelected?: boolean
  isDimmed?: boolean
  hasChildren?: boolean
  onClick?: () => void
}) {
  const intervalEnd = intervalStart + intervalDuration

  // Skip if event doesn't overlap with interval at all
  if (event.end_time <= intervalStart || event.start_time >= intervalEnd)
    return null

  const relativeStart = event.start_time - intervalStart
  const leftPercent = Math.max(0, (relativeStart / intervalDuration) * 100)

  const duration = event.end_time - event.start_time
  const durationMs = duration * 1000

  // Calculate visible width (account for events starting before interval)
  const visibleStart = Math.max(event.start_time, intervalStart)
  const visibleDuration = event.end_time - visibleStart
  const widthPercent = (visibleDuration / intervalDuration) * 100

  const color = getTrainerEventColor(event.event_type)
  const titleInfo = formatTrainerEventTitle(
    event.event_type,
    event.step,
    event.microbatch,
    event.minibatch,
  )

  return (
    <HoverTooltipBlock
      className={`absolute top-1 bottom-1 group transition-all duration-200 ${
        hasChildren ? "cursor-pointer" : "cursor-default"
      }`}
      style={{
        left: `${leftPercent}%`,
        width: `${Math.min(widthPercent, 100 - leftPercent)}%`,
        minWidth: "1px",
        backgroundColor: color,
        borderRadius: "2px",
        opacity: isDimmed ? 0.3 : 1,
        boxShadow: isSelected
          ? `0 0 0 2px ${color}, 0 0 8px ${color}`
          : undefined,
        zIndex: isSelected ? 10 : 1,
      }}
      onClick={hasChildren ? onClick : undefined}
      tooltip={
        <EventTooltip
          title={titleInfo.primary}
          titleSecondary={titleInfo.secondary}
          color={color}
          details={[
            { label: "Duration", value: formatDuration(durationMs) },
            { label: "Rank", value: String(event.rank) },
            ...(hasChildren
              ? [{ label: "Click", value: "to expand children" }]
              : []),
          ]}
          startTime={event.start_time}
          endTime={event.end_time}
        />
      }
    />
  )
}

// ============================================================================
// Trainer Breakdown Content (for footer)
// ============================================================================

export function TrainerBreakdownContent({
  parentEvent,
  childEvents,
}: {
  parentEvent: TrainerEvent
  childEvents: TrainerEvent[]
}) {
  const darkMode = useAtomValue(darkModeAtom)
  const sortedChildren = [...childEvents].sort(
    (a, b) => a.start_time - b.start_time,
  )

  // Use the parent event's time range as the timeline bounds
  const intervalStart = parentEvent.start_time
  const intervalEnd = parentEvent.end_time
  const intervalDuration = intervalEnd - intervalStart

  // Calculate idle periods within the parent event
  const idlePeriods = calculateIdlePeriods(
    sortedChildren,
    intervalStart,
    intervalEnd,
  )

  return (
    <div className="px-3 pb-2">
      {/* Child events legend / badges */}
      <div className="flex flex-wrap gap-1.5 mb-2 text-xs">
        {sortedChildren.map((event) => {
          const color = getTrainerEventColor(event.event_type)
          const childName = event.event_type.includes("/")
            ? event.event_type.split("/").pop()
            : event.event_type
          const duration = event.end_time - event.start_time
          const pct =
            intervalDuration > 0
              ? ((duration / intervalDuration) * 100).toFixed(0)
              : "0"

          return (
            <div
              key={event.event_type}
              className="flex items-center gap-1.5 px-2 py-0.5 rounded-full"
              style={{ backgroundColor: `${color}20` }}
            >
              <div
                className="w-2 h-2 rounded-sm"
                style={{ backgroundColor: color }}
              />
              <span className="capitalize text-muted-foreground">
                {childName}
              </span>
              <span className="font-medium">
                {formatDuration(duration * 1000)}
              </span>
              <span className="text-muted-foreground">{pct}%</span>
            </div>
          )
        })}
      </div>

      {/* Sub-timeline bar */}
      <div className="relative">
        <div className="relative h-7 bg-muted/30 rounded-lg border border-border/50">
          {/* Idle periods within parent */}
          {idlePeriods.map((idle, idx) => {
            const relativeStart = idle.start - intervalStart
            const leftPercent = (relativeStart / intervalDuration) * 100
            const widthPercent = (idle.duration / intervalDuration) * 100

            if (widthPercent < 0.5 || leftPercent < 0 || leftPercent >= 100)
              return null

            return (
              <HoverTooltipBlock
                key={`child-idle-${idx}`}
                className="absolute top-px bottom-px group"
                style={{
                  left: `${leftPercent}%`,
                  width: `${Math.min(widthPercent, 100 - leftPercent)}%`,
                  backgroundColor: IDLE_COLOR,
                  borderRadius: "1px",
                  opacity: 0.5,
                }}
                tooltip={
                  <EventTooltip
                    title="Gap"
                    color={IDLE_COLOR}
                    details={[
                      {
                        label: "Duration",
                        value: formatDuration(idle.duration * 1000),
                      },
                    ]}
                  />
                }
              />
            )
          })}

          {/* Child events */}
          {sortedChildren.map((event, idx) => {
            const relativeStart = event.start_time - intervalStart
            const leftPercent = Math.max(
              0,
              (relativeStart / intervalDuration) * 100,
            )

            const duration = event.end_time - event.start_time
            const durationMs = duration * 1000
            const widthPercent = Math.max(
              0.5,
              (duration / intervalDuration) * 100,
            )

            const color = getTrainerEventColor(event.event_type)
            // Extract the child name (e.g., "loss/shift" -> "shift")
            const childName = event.event_type.includes("/")
              ? event.event_type.split("/").pop()
              : event.event_type

            return (
              <HoverTooltipBlock
                key={`child-${event.start_time}-${idx}`}
                className="absolute top-px bottom-px group cursor-default flex items-center justify-center"
                style={{
                  left: `${leftPercent}%`,
                  width: `${Math.min(widthPercent, 100 - leftPercent)}%`,
                  backgroundColor: color,
                  borderRadius: "1px",
                  border: darkMode ? "1px solid rgba(255, 255, 255, 0.2)" : "1px solid rgba(0, 0, 0, 0.3)",
                  boxSizing: "border-box",
                }}
                tooltip={
                  <EventTooltip
                    title={childName || event.event_type}
                    color={color}
                    capitalize
                    details={[
                      { label: "Duration", value: formatDuration(durationMs) },
                      {
                        label: "% of parent",
                        value: `${((duration / intervalDuration) * 100).toFixed(
                          1,
                        )}%`,
                      },
                    ]}
                    startTime={event.start_time}
                    endTime={event.end_time}
                  />
                }
              >
                {/* Show label if there's enough space */}
                {widthPercent > 8 && (
                  <span className="text-[9px] font-medium text-white truncate px-1 drop-shadow-sm pointer-events-none">
                    {childName}
                  </span>
                )}
              </HoverTooltipBlock>
            )
          })}
        </div>
      </div>

      {/* Time axis for sub-timeline */}
      <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
        <span>{formatDuration(0)}</span>
        <span>{formatDuration((intervalDuration / 2) * 1000)}</span>
        <span>{formatDuration(intervalDuration * 1000)}</span>
      </div>
    </div>
  )
}

// ============================================================================
// Event Tooltip
// ============================================================================

interface TooltipDetail {
  label: string
  value: string
}

function EventTooltip({
  title,
  titleSecondary,
  color,
  details,
  capitalize = false,
  timestamp,
  startTime,
  endTime,
  statusLabel,
}: {
  title: string
  titleSecondary?: string | null
  color: string
  details: TooltipDetail[]
  capitalize?: boolean
  timestamp?: number
  startTime?: number
  endTime?: number
  statusLabel?: { text: string; className: string } | null
}) {
  return (
    <div className="bg-popover text-popover-foreground text-xs py-2 px-3 rounded-lg shadow-xl whitespace-nowrap pointer-events-none border border-border min-w-[200px]">
      <div className="font-semibold mb-1.5 flex items-center gap-2">
        <div
          className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
          style={{ backgroundColor: color }}
        />
        <span className={capitalize ? "capitalize" : ""}>
          {title}
          {titleSecondary && (
            <span className="text-muted-foreground font-normal ml-1">
              {titleSecondary}
            </span>
          )}
        </span>
      </div>
      {statusLabel && (
        <div className={`font-semibold mb-1.5 ${statusLabel.className}`}>
          {statusLabel.text}
        </div>
      )}
      <div className="space-y-0.5 text-muted-foreground">
        {details.map((detail) => (
          <div key={detail.label}>
            {detail.label}: {detail.value}
          </div>
        ))}
        {timestamp !== undefined && (
          <div>
            Time: {new Date(timestamp * 1000).toISOString().slice(11, 23)}
          </div>
        )}
        {startTime !== undefined && (
          <div>
            Start: {new Date(startTime * 1000).toISOString().slice(11, 23)}
          </div>
        )}
        {endTime !== undefined && (
          <div>End: {new Date(endTime * 1000).toISOString().slice(11, 23)}</div>
        )}
      </div>
    </div>
  )
}

function HoverTooltipBlock({
  className,
  style,
  onClick,
  tooltip,
  children,
  interactive = false,
}: {
  className?: string
  style?: CSSProperties
  onClick?: () => void
  tooltip: ReactNode
  children?: ReactNode
  interactive?: boolean
}) {
  const [isHovered, setIsHovered] = useState(false)
  const [isPinned, setIsPinned] = useState(false)
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
  const [pinnedPos, setPinnedPos] = useState<{ x: number; y: number } | null>(
    null,
  )
  const tooltipRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLDivElement>(null)
  const isVisible = isHovered || isPinned
  // When pinned, use the frozen pinned position; otherwise follow cursor
  const effectiveMousePos = isPinned && pinnedPos ? pinnedPos : mousePos

  // After each render where position or visibility changes, measure tooltip and clamp to viewport
  useLayoutEffect(() => {
    const el = tooltipRef.current
    if (!isVisible || !el) return

    const rect = el.getBoundingClientRect()
    const pad = 8

    // Horizontal: centre on cursor, but clamp within viewport
    let left = effectiveMousePos.x - rect.width / 2
    if (left + rect.width + pad > window.innerWidth) {
      left = window.innerWidth - rect.width - pad
    }
    if (left < pad) left = pad

    // Vertical: prefer above cursor with a 12px gap
    let top = effectiveMousePos.y - rect.height - 12
    if (top < pad) {
      // Flip below cursor
      top = effectiveMousePos.y + 16
    }
    // If it still overflows at the bottom, clamp
    if (top + rect.height + pad > window.innerHeight) {
      top = window.innerHeight - rect.height - pad
    }

    el.style.left = `${left}px`
    el.style.top = `${top}px`
    el.style.visibility = "visible"
  }, [isVisible, effectiveMousePos])

  // Click-outside listener to unpin tooltip
  useEffect(() => {
    if (!isPinned) return
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node
      // If click is inside the tooltip portal or the trigger, ignore
      if (tooltipRef.current?.contains(target)) return
      if (triggerRef.current?.contains(target)) return
      setIsPinned(false)
      setPinnedPos(null)
    }
    // Use a timeout so the current click event doesn't immediately dismiss
    const timeoutId = setTimeout(() => {
      document.addEventListener("pointerdown", handleClickOutside)
    }, 0)
    return () => {
      clearTimeout(timeoutId)
      document.removeEventListener("pointerdown", handleClickOutside)
    }
  }, [isPinned])

  const handleClick = useCallback(() => {
    if (interactive) {
      // Toggle pin
      setIsPinned((prev) => {
        if (!prev) {
          // Pin at current mouse position
          setPinnedPos(mousePos)
          return true
        } else {
          setPinnedPos(null)
          return false
        }
      })
    }
    onClick?.()
  }, [interactive, onClick, mousePos])

  return (
    <div
      ref={triggerRef}
      className={className}
      style={style}
      onClick={handleClick}
      onPointerEnter={(e) => {
        setMousePos({ x: e.clientX, y: e.clientY })
        setIsHovered(true)
      }}
      onPointerLeave={() => {
        setIsHovered(false)
      }}
      onPointerMove={(e) => {
        if (!isPinned) {
          setMousePos({ x: e.clientX, y: e.clientY })
        }
      }}
    >
      {children}
      {isVisible
        ? createPortal(
            <div
              ref={tooltipRef}
              className="fixed"
              style={{
                left: effectiveMousePos.x,
                top: effectiveMousePos.y,
                zIndex: 99999,
                // Hidden until useLayoutEffect measures and adjusts position
                visibility: "hidden",
                pointerEvents: isPinned ? "auto" : "none",
              }}
            >
              {tooltip}
            </div>,
            document.body,
          )
        : null}
    </div>
  )
}

// ============================================================================
// Utility Functions
// ============================================================================

function calculateIdlePeriods(
  sortedEvents: TrainerEvent[],
  intervalStart: number,
  intervalEnd: number,
): Array<{ start: number; end: number; duration: number }> {
  const idlePeriods: Array<{ start: number; end: number; duration: number }> =
    []

  // Idle at the beginning
  if (sortedEvents.length > 0 && sortedEvents[0].start_time > intervalStart) {
    idlePeriods.push({
      start: intervalStart,
      end: sortedEvents[0].start_time,
      duration: sortedEvents[0].start_time - intervalStart,
    })
  }

  // Gaps between events
  for (let i = 0; i < sortedEvents.length - 1; i++) {
    const currentEnd = sortedEvents[i].end_time
    const nextStart = sortedEvents[i + 1].start_time
    if (currentEnd < nextStart) {
      idlePeriods.push({
        start: currentEnd,
        end: nextStart,
        duration: nextStart - currentEnd,
      })
    }
  }

  // Idle at the end
  if (sortedEvents.length > 0) {
    const lastEvent = sortedEvents[sortedEvents.length - 1]
    if (lastEvent.end_time < intervalEnd) {
      idlePeriods.push({
        start: lastEvent.end_time,
        end: intervalEnd,
        duration: intervalEnd - lastEvent.end_time,
      })
    }
  } else {
    // No events - entire interval is idle
    idlePeriods.push({
      start: intervalStart,
      end: intervalEnd,
      duration: intervalEnd - intervalStart,
    })
  }

  return idlePeriods
}
