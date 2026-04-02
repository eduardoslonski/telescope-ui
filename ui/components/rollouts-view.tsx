
import { useEffect, useMemo, useRef, useState } from "react"
import { useAtom } from "jotai"
import type { SetStateAction, WritableAtom } from "jotai"
import { useNavigate } from "react-router-dom"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Toggle } from "@/components/ui/toggle"
import { ScrollArea } from "@/components/ui/scroll-area"
import { CopyIcon, CheckIcon, ChevronDown, ChevronRight } from "lucide-react"
import ReactMarkdown from "react-markdown"
import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import katex from "katex"
import "katex/dist/katex.min.css"
import { cn } from "@/lib/utils"
import {
  rolloutsLastValidSampleAtom,
  rolloutsSampleViewMetricsAtom,
  type RolloutsLastValidSampleState,
  type RolloutsRenderOption,
} from "@/lib/atoms"
import type {
  Prompt,
  GenerationRow,
  EnvResponseRow,
  ToolCallRow,
  RolloutMetric,
  GoldenAnswer,
  InfoTurn,
  SampleData,
  RolloutsDisplaySample,
} from "@/lib/types"

/** Per-environment reward range from the run config */
export type EnvRewardRanges = Map<string, { min: number; max: number }>

/** Per-metric range with optional invert flag (when true, lower values are better) */
export type MetricRange = { min: number; max: number; invert?: boolean }

/** Per-environment, per-metric range from the run config's metrics_ranges */
export type EnvMetricRanges = Map<string, Map<string, MetricRange>>

/**
 * Normalize an environments config value which may be a JSON string, an array,
 * a single object, or a W&B config wrapper ({value: ...}) into an array of env objects.
 */
export function normalizeEnvItems(envs: unknown): unknown[] {
  let parsed = envs
  // Unwrap W&B config {value: ...} wrapper
  if (typeof parsed === "object" && parsed !== null && "value" in parsed) {
    parsed = (parsed as { value: unknown }).value
  }
  // Often stored as a JSON string
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed)
    } catch {
      return []
    }
  }
  return Array.isArray(parsed) ? parsed : parsed ? [parsed] : []
}

/**
 * Extract per-environment reward ranges from the run config.
 * Handles JSON string, array, and single-object formats for `env_details`.
 */
export function extractEnvRewardRanges(envs: unknown): EnvRewardRanges {
  const map: EnvRewardRanges = new Map()
  const items = normalizeEnvItems(envs)
  for (const env of items) {
    const e = env as Record<string, unknown>
    if (
      typeof e?.name === "string" &&
      typeof e?.reward_min === "number" &&
      typeof e?.reward_max === "number"
    ) {
      map.set(e.name, { min: e.reward_min, max: e.reward_max })
    }
  }
  return map
}

/**
 * Extract per-environment, per-metric ranges from the run config.
 * Each environment entry can have a `metrics_ranges` dict mapping
 * metric_name → { min, max }.
 */
export function extractEnvMetricRanges(envs: unknown): EnvMetricRanges {
  const map: EnvMetricRanges = new Map()
  const items = normalizeEnvItems(envs)
  for (const env of items) {
    const e = env as Record<string, unknown>
    if (typeof e?.name !== "string") continue
    const ranges = e.metrics_ranges
    if (ranges && typeof ranges === "object") {
      const metricMap = new Map<string, MetricRange>()
      for (const [metricName, range] of Object.entries(
        ranges as Record<string, unknown>,
      )) {
        const r = range as Record<string, unknown>
        if (typeof r?.min === "number" && typeof r?.max === "number") {
          const entry: MetricRange = { min: r.min, max: r.max }
          if (r.invert === true) entry.invert = true
          metricMap.set(metricName, entry)
        }
      }
      if (metricMap.size > 0) {
        map.set(e.name, metricMap)
      }
    }
  }
  return map
}

/**
 * Convert the server's data_metric_ranges (nested dict: env → metric → {min,max})
 * into the EnvMetricRanges Map format.
 */
export function parseDataMetricRanges(
  raw:
    | Record<
        string,
        Record<string, { min: number; max: number; invert?: boolean }>
      >
    | undefined,
): EnvMetricRanges {
  const map: EnvMetricRanges = new Map()
  if (!raw) return map
  for (const [env, metrics] of Object.entries(raw)) {
    const metricMap = new Map<string, MetricRange>()
    for (const [metricName, range] of Object.entries(metrics)) {
      if (typeof range?.min === "number" && typeof range?.max === "number") {
        const entry: MetricRange = { min: range.min, max: range.max }
        if (range.invert === true) entry.invert = true
        metricMap.set(metricName, entry)
      }
    }
    if (metricMap.size > 0) {
      map.set(env, metricMap)
    }
  }
  return map
}

/**
 * Merge two EnvMetricRanges maps. `primary` values take precedence;
 * `fallback` fills in any env/metric combinations missing from `primary`.
 */
export function mergeEnvMetricRanges(
  primary: EnvMetricRanges,
  fallback: EnvMetricRanges,
): EnvMetricRanges {
  const merged: EnvMetricRanges = new Map()
  // Copy all from fallback first
  for (const [env, metricMap] of fallback) {
    merged.set(env, new Map(metricMap))
  }
  // Override / add from primary
  for (const [env, metricMap] of primary) {
    const existing = merged.get(env) ?? new Map()
    for (const [metric, range] of metricMap) {
      existing.set(metric, range)
    }
    merged.set(env, existing)
  }
  return merged
}

interface RolloutsViewProps {
  prompts?: Prompt[]
  generations?: GenerationRow[]
  envResponses?: EnvResponseRow[]
  toolCalls?: ToolCallRow[]
  samplesData?: SampleData[]
  rolloutMetrics?: RolloutMetric[]
  goldenAnswers?: GoldenAnswer[]
  infoTurns?: InfoTurn[]
  isLoading?: boolean
  step?: number
  scrollToSampleId?: number | null
  showGoTo?: boolean
  /** Base path for "Go to" navigation (defaults to "/rollouts") */
  goToBasePath?: string
  /** Query param name for the step (defaults to "step") */
  goToStepParam?: string
  // Filter props for sample picker sidebar
  filterGroupId?: number | null
  filterSampleIdx?: number | null
  lastValidSampleAtom?: WritableAtom<
    RolloutsLastValidSampleState | null,
    [SetStateAction<RolloutsLastValidSampleState | null>],
    void
  >
  /** Increment to trigger collapse all sections */
  collapseAllSignal?: number
  /** Increment to trigger expand all sections */
  expandAllSignal?: number
  /** Default open state for sections when a new sample loads */
  defaultSectionsOpen?: boolean
  /** Per-environment reward range from the run config (env name → { min, max }) */
  envRewardRanges?: EnvRewardRanges
  /** Per-environment, per-metric ranges from the run config */
  envMetricRanges?: EnvMetricRanges
  /** Advantage range derived from group_size config */
  advantageRange?: { min: number; max: number } | null
  /** Render options for content (markdown, latex) */
  renderOptions?: RolloutsRenderOption[]
  /** Whether to render <think> blocks with special collapsible formatting */
  formatThinkBlocks?: boolean
}

// Grouped sample: a sample with its prompt and all its turns
type GroupedSample = RolloutsDisplaySample

export function RolloutsView({
  prompts,
  generations,
  envResponses,
  toolCalls,
  samplesData,
  rolloutMetrics,
  goldenAnswers,
  infoTurns,
  isLoading,
  step,
  scrollToSampleId = null,
  showGoTo = false,
  goToBasePath = "/rollouts",
  goToStepParam = "step",
  filterGroupId = null,
  filterSampleIdx = null,
  lastValidSampleAtom = rolloutsLastValidSampleAtom,
  collapseAllSignal = 0,
  expandAllSignal = 0,
  defaultSectionsOpen = true,
  envRewardRanges,
  envMetricRanges,
  advantageRange,
  renderOptions = [],
  formatThinkBlocks = true,
}: RolloutsViewProps) {
  // Group rollout metrics by sample_id for easy lookup
  const metricsBySample = useMemo(() => {
    if (!rolloutMetrics) return new Map<number, RolloutMetric[]>()
    const map = new Map<number, RolloutMetric[]>()
    for (const metric of rolloutMetrics) {
      const existing = map.get(metric.sample_id) || []
      existing.push(metric)
      map.set(metric.sample_id, existing)
    }
    return map
  }, [rolloutMetrics])

  // Derive all available metric names from rolloutMetrics
  const allAvailableMetricNames = useMemo(() => {
    if (!rolloutMetrics) return []
    const names = new Set<string>()
    for (const m of rolloutMetrics) names.add(m.metric_name)
    return [...names].sort()
  }, [rolloutMetrics])

  // Group golden answers by sample_id for easy lookup
  const goldenAnswersBySample = useMemo(() => {
    if (!goldenAnswers) return new Map<number, GoldenAnswer[]>()
    const map = new Map<number, GoldenAnswer[]>()
    for (const answer of goldenAnswers) {
      const existing = map.get(answer.sample_id) || []
      existing.push(answer)
      map.set(answer.sample_id, existing)
    }
    return map
  }, [goldenAnswers])

  // Group info turns by sample_id for easy lookup
  const infoTurnsBySample = useMemo(() => {
    if (!infoTurns) return new Map<number, InfoTurn[]>()
    const map = new Map<number, InfoTurn[]>()
    for (const it of infoTurns) {
      const existing = map.get(it.sample_id) || []
      existing.push(it)
      map.set(it.sample_id, existing)
    }
    return map
  }, [infoTurns])

  // Group prompts by group_id for easy lookup
  const promptsByGroup = useMemo(() => {
    if (!prompts) return new Map<number, Prompt>()
    const map = new Map<number, Prompt>()
    for (const p of prompts) {
      map.set(p.group_id, p)
    }
    return map
  }, [prompts])

  // Index samples data by sample_id for easy lookup
  const samplesDataBySample = useMemo(() => {
    if (!samplesData) return new Map<number, SampleData>()
    const map = new Map<number, SampleData>()
    for (const s of samplesData) {
      map.set(s.sample_id, s)
    }
    return map
  }, [samplesData])

  // Group generations, env_responses, tool_calls by sample_id and combine with prompts
  const groupedSamples = useMemo(() => {
    if (!generations) return []

    // Group generations by sample_id
    const gensBySample = new Map<number, GenerationRow[]>()
    const groupIdBySample = new Map<number, number>()

    for (const gen of generations) {
      const existing = gensBySample.get(gen.sample_id) || []
      existing.push(gen)
      gensBySample.set(gen.sample_id, existing)
      groupIdBySample.set(gen.sample_id, gen.group_id)
    }

    // Group env_responses by sample_id
    const envRespBySample = new Map<number, EnvResponseRow[]>()
    if (envResponses) {
      for (const er of envResponses) {
        const existing = envRespBySample.get(er.sample_id) || []
        existing.push(er)
        envRespBySample.set(er.sample_id, existing)
      }
    }

    // Group tool_calls by sample_id
    const toolCallsBySample = new Map<number, ToolCallRow[]>()
    if (toolCalls) {
      for (const tc of toolCalls) {
        const existing = toolCallsBySample.get(tc.sample_id) || []
        existing.push(tc)
        toolCallsBySample.set(tc.sample_id, existing)
      }
    }

    // Build grouped samples
    const samples: GroupedSample[] = []
    for (const [sample_id, gens] of gensBySample) {
      const sortedGens = [...gens].sort((a, b) => a.generation_idx - b.generation_idx)
      const group_id = groupIdBySample.get(sample_id) ?? -1
      const prompt = promptsByGroup.get(group_id) ?? null

      // Get sample-level data from samples_data
      const sampleData = samplesDataBySample.get(sample_id)

      samples.push({
        sample_id,
        group_id,
        prompt,
        generations: sortedGens,
        env_responses: envRespBySample.get(sample_id) ?? [],
        tool_calls: toolCallsBySample.get(sample_id) ?? [],
        reward: sampleData?.reward ?? null,
        advantage: sampleData?.advantage ?? null,
        total_tokens: sampleData?.total_tokens ?? null,
        raw_string: sampleData?.raw_string ?? null,
        stop_reason: sampleData?.stop_reason ?? null,
        num_generations: sampleData?.num_generations ?? null,
      })
    }

    // Sort by sample_id
    let result = samples.sort((a, b) => a.sample_id - b.sample_id)

    // Apply filters
    if (filterGroupId !== null) {
      result = result.filter((s) => s.group_id === filterGroupId)
    }
    if (filterSampleIdx !== null) {
      result = result.filter((s) => s.sample_id === filterSampleIdx)
    }

    return result
  }, [
    generations,
    envResponses,
    toolCalls,
    promptsByGroup,
    samplesDataBySample,
    filterGroupId,
    filterSampleIdx,
  ])

  // Keep track of last valid sample to show when changing steps or pages
  const [lastValidSample, setLastValidSample] = useAtom(lastValidSampleAtom)

  // Update the ref when we have valid filtered data
  const currentSample = groupedSamples[0]
  const currentSampleIdx = currentSample?.sample_id ?? null
  const currentMetrics = useMemo(
    () =>
      currentSampleIdx !== null
        ? (metricsBySample.get(currentSampleIdx) ?? [])
        : [],
    [metricsBySample, currentSampleIdx],
  )
  const currentGoldenAnswers = useMemo(
    () =>
      currentSampleIdx !== null
        ? (goldenAnswersBySample.get(currentSampleIdx) ?? [])
        : [],
    [goldenAnswersBySample, currentSampleIdx],
  )
  const currentInfoTurns = useMemo(
    () =>
      currentSampleIdx !== null
        ? (infoTurnsBySample.get(currentSampleIdx) ?? [])
        : [],
    [infoTurnsBySample, currentSampleIdx],
  )

  useEffect(() => {
    if (!currentSample || filterSampleIdx === null) return
    setLastValidSample({
      sample: currentSample,
      rolloutMetrics: currentMetrics,
      goldenAnswers: currentGoldenAnswers,
    })
  }, [
    currentSample,
    currentMetrics,
    currentGoldenAnswers,
    filterSampleIdx,
    setLastValidSample,
  ])

  const fallbackSample =
    filterSampleIdx !== null &&
    lastValidSample?.sample.sample_id === filterSampleIdx
      ? lastValidSample
      : null

  // Use last valid sample if current data doesn't have the selected sample
  const displaySample = currentSample || fallbackSample?.sample
  const displayMetrics = currentSample
    ? currentMetrics
    : fallbackSample?.rolloutMetrics || []
  const displayGoldenAnswers = currentSample
    ? currentGoldenAnswers
    : fallbackSample?.goldenAnswers || []
  const displayInfoTurns = currentSample ? currentInfoTurns : []

  const lastScrolledSampleRef = useRef<number | null>(null)

  useEffect(() => {
    if (scrollToSampleId == null) return
    if (lastScrolledSampleRef.current === scrollToSampleId) return
    const element = document.getElementById(`sample-${scrollToSampleId}`)
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" })
      lastScrolledSampleRef.current = scrollToSampleId
    }
  }, [scrollToSampleId, groupedSamples])

  const goToStep = step !== undefined ? step : generations?.[0]?.step

  // Show message when no sample is selected (require both group AND sample)
  if (filterSampleIdx === null) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
        <p>Select a sample from the sidebar to view</p>
      </div>
    )
  }

  // Show last valid sample if available, otherwise show empty state
  if (!displaySample && !isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        <p>No rollouts data available</p>
      </div>
    )
  }

  if (!displaySample) {
    return null
  }

  return (
    <div className="space-y-4 min-w-0">
      <SampleView
        key={displaySample.sample_id}
        sample={displaySample}
        rolloutMetrics={displayMetrics}
        goldenAnswers={displayGoldenAnswers}
        infoTurns={displayInfoTurns}
        availableMetricNames={allAvailableMetricNames}
        showGoTo={showGoTo}
        goToStep={goToStep}
        goToBasePath={goToBasePath}
        goToStepParam={goToStepParam}
        collapseAllSignal={collapseAllSignal}
        expandAllSignal={expandAllSignal}
        defaultSectionsOpen={defaultSectionsOpen}
        envRewardRanges={envRewardRanges}
        envMetricRanges={envMetricRanges}
        advantageRange={advantageRange}
        renderOptions={renderOptions}
        formatThinkBlocks={formatThinkBlocks}
      />
    </div>
  )
}

function SampleView({
  sample,
  rolloutMetrics,
  goldenAnswers,
  infoTurns,
  availableMetricNames,
  showGoTo,
  goToStep,
  goToBasePath = "/rollouts",
  goToStepParam = "step",
  collapseAllSignal = 0,
  expandAllSignal = 0,
  defaultSectionsOpen = true,
  envRewardRanges,
  envMetricRanges,
  advantageRange,
  renderOptions = [],
  formatThinkBlocks = true,
}: {
  sample: GroupedSample
  rolloutMetrics: RolloutMetric[]
  goldenAnswers: GoldenAnswer[]
  infoTurns?: InfoTurn[]
  availableMetricNames: string[]
  showGoTo: boolean
  goToStep?: number
  goToBasePath?: string
  goToStepParam?: string
  collapseAllSignal?: number
  expandAllSignal?: number
  defaultSectionsOpen?: boolean
  envRewardRanges?: EnvRewardRanges
  envMetricRanges?: EnvMetricRanges
  advantageRange?: { min: number; max: number } | null
  renderOptions?: RolloutsRenderOption[]
  formatThinkBlocks?: boolean
}) {
  // Sample Metrics selection (persisted)
  const [selectedSampleMetrics, setSelectedSampleMetrics] = useAtom(
    rolloutsSampleViewMetricsAtom,
  )

  const isAllSelected = selectedSampleMetrics.length === 0
  const isNoneSelected =
    selectedSampleMetrics.length === 1 &&
    selectedSampleMetrics[0] === "__none__"

  const isMetricVisible = (name: string) => {
    if (isAllSelected) return true
    if (isNoneSelected) return false
    return selectedSampleMetrics.includes(name)
  }

  const toggleSampleMetric = (name: string) => {
    setSelectedSampleMetrics((prev) => {
      if (prev.length === 0) {
        // Currently showing all → switch to explicit mode excluding this one
        return availableMetricNames.filter((n) => n !== name)
      }
      // Clean up sentinel if present (was in "none" mode, user is checking one)
      const clean = prev.filter((n) => n !== "__none__")
      if (clean.length === 0) return [name]
      if (clean.includes(name)) {
        const next = clean.filter((n) => n !== name)
        if (next.length === 0) return ["__none__"]
        return next
      }
      const next = [...clean, name]
      // If all are selected, reset to empty (= show all)
      if (
        availableMetricNames.length > 0 &&
        next.length >= availableMetricNames.length
      )
        return []
      return next
    })
  }

  // Sort metrics by name for consistent display
  const sortedMetrics = useMemo(() => {
    return [...rolloutMetrics].sort((a, b) =>
      a.metric_name.localeCompare(b.metric_name),
    )
  }, [rolloutMetrics])

  // Filter metrics based on selection
  const visibleMetrics = useMemo(() => {
    if (selectedSampleMetrics.length === 0) return sortedMetrics
    return sortedMetrics.filter((m) =>
      selectedSampleMetrics.includes(m.metric_name),
    )
  }, [sortedMetrics, selectedSampleMetrics])

  // Filter golden answers with meaningful values
  const visibleGoldenAnswers = useMemo(() => {
    return goldenAnswers.filter(
      (answer) => answer.value != null && answer.value !== "",
    )
  }, [goldenAnswers])

  const env = sample.prompt?.env ?? null
  const navigate = useNavigate()

  const handleGoTo = () => {
    if (goToStep == null) return
    navigate(
      `${goToBasePath}?${goToStepParam}=${goToStep}&group=${sample.group_id}&sample=${sample.sample_id}`,
    )
  }

  // Group info turns by generation_idx for efficient lookup
  const infoTurnsByGenerationIdx = useMemo(() => {
    if (!infoTurns || infoTurns.length === 0)
      return new Map<number, InfoTurn[]>()
    const map = new Map<number, InfoTurn[]>()
    for (const info of infoTurns) {
      const existing = map.get(info.generation_idx) || []
      existing.push(info)
      map.set(info.generation_idx, existing)
    }
    return map
  }, [infoTurns])

  // Collapsible state for each section
  const [goldenOpen, setGoldenOpen] = useState(false)
  const [systemOpen, setSystemOpen] = useState(false)
  const [promptOpen, setPromptOpen] = useState(defaultSectionsOpen)
  const [turnsOpen, setTurnsOpen] = useState<Record<number, boolean>>({})
  // No initialization effect needed – the component is keyed by sample_id so
  // it remounts on sample change, and the render falls back to defaultSectionsOpen
  // for any turn not explicitly set in turnsOpen.

  // Track previous signal values to detect changes during render
  const [prevCollapseSignal, setPrevCollapseSignal] = useState(collapseAllSignal)
  const [prevExpandSignal, setPrevExpandSignal] = useState(expandAllSignal)

  // Count total collapsible sections: each generation + each env_response that follows
  const turnsCount = sample.generations.length + sample.env_responses.length

  // Collapse all sections when signal changes (adjust state during render)
  if (collapseAllSignal !== prevCollapseSignal) {
    setPrevCollapseSignal(collapseAllSignal)
    setGoldenOpen(false)
    setSystemOpen(false)
    setPromptOpen(false)
    const collapsed: Record<number, boolean> = {}
    for (let i = 0; i < turnsCount; i++) {
      collapsed[i] = false
    }
    setTurnsOpen(collapsed)
  }

  // Expand all sections when signal changes (adjust state during render)
  if (expandAllSignal !== prevExpandSignal) {
    setPrevExpandSignal(expandAllSignal)
    setGoldenOpen(true)
    setSystemOpen(true)
    setPromptOpen(true)
    const expanded: Record<number, boolean> = {}
    for (let i = 0; i < turnsCount; i++) {
      expanded[i] = true
    }
    setTurnsOpen(expanded)
  }

  const toggleTurn = (idx: number) => {
    setTurnsOpen((prev) => ({ ...prev, [idx]: !prev[idx] }))
  }

  // Primary metrics (first line): env, turns, reward, advantage
  const rewardRange = env ? envRewardRanges?.get(env) : undefined
  const primaryItems: Array<{
    label: string
    value: string
    rangeColor?: string
  }> = []
  if (env) primaryItems.push({ label: "Env", value: env })
  const generationsCount = sample.generations.length
  if (generationsCount > 1)
    primaryItems.push({ label: "Turns", value: String(generationsCount) })
  if (sample.reward !== null) {
    const rewardColor = metricRangeColor(
      sample.reward,
      rewardRange?.min ?? null,
      rewardRange?.max ?? null,
    )
    primaryItems.push({
      label: "Reward",
      value: sample.reward.toFixed(2),
      rangeColor: rewardColor,
    })
  }
  if (sample.advantage !== null) {
    const advantageColor = metricRangeColor(
      sample.advantage,
      advantageRange?.min ?? null,
      advantageRange?.max ?? null,
    )
    primaryItems.push({
      label: "Advantage",
      value: sample.advantage.toFixed(2),
      rangeColor: advantageColor,
    })
  }

  // Secondary metrics (second line): rollout metrics (filtered by selection)
  const secondaryItems: Array<{
    label: string
    value: string
    highlight?: "positive" | "negative"
    rangeColor?: string
  }> = []
  for (const metric of visibleMetrics) {
    const metricEnv = metric.env ?? env
    const range = lookupMetricRange(
      envMetricRanges,
      metricEnv,
      metric.metric_name,
    )
    const color = metricRangeColor(
      metric.value,
      range?.min ?? null,
      range?.max ?? null,
      range?.invert,
    )
    secondaryItems.push({
      label: formatMetricName(metric.metric_name),
      value: formatMetricValue(metric.value),
      highlight: color
        ? undefined
        : metric.value === 1
          ? "positive"
          : metric.value === 0
            ? "negative"
            : undefined,
      rangeColor: color,
    })
  }

  return (
    <div id={`sample-${sample.sample_id}`}>
      {/* Metrics summary */}
      {(primaryItems.length > 0 || secondaryItems.length > 0 || showGoTo) && (
        <div className="mb-2 space-y-0.5">
          {/* First line: primary metrics + Sample Metrics selector */}
          {(primaryItems.length > 0 ||
            showGoTo ||
            availableMetricNames.length > 0) && (
            <div className="flex items-center gap-2 text-sm min-w-0 flex-wrap">
              {showGoTo && (
                <Button
                  variant="outline"
                  size="sm"
                  className="h-6 px-2 text-xs mr-1"
                  onClick={handleGoTo}
                  disabled={goToStep == null}
                >
                  Go to
                </Button>
              )}
              {primaryItems.map((item, i) => (
                <span
                  key={item.label}
                  className="inline-flex items-center gap-1.5"
                >
                  {i > 0 && <span className="text-foreground/20">·</span>}
                  <span className="text-muted-foreground">{item.label}</span>
                  <span
                    className="font-medium tabular-nums text-foreground/80"
                    style={
                      item.rangeColor ? { color: item.rangeColor } : undefined
                    }
                  >
                    {item.value}
                  </span>
                </span>
              ))}
              {availableMetricNames.length > 0 && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:underline cursor-pointer">
                      Sample Metrics
                      <ChevronDown className="h-3 w-3" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent
                    align="start"
                    className="w-max max-w-[400px]"
                  >
                    <DropdownMenuItem
                      onSelect={(e) => {
                        e.preventDefault()
                        setSelectedSampleMetrics([])
                      }}
                    >
                      Select All
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onSelect={(e) => {
                        e.preventDefault()
                        setSelectedSampleMetrics(["__none__"])
                      }}
                    >
                      Unselect All
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    {availableMetricNames.map((name) => (
                      <DropdownMenuCheckboxItem
                        key={name}
                        checked={isMetricVisible(name)}
                        onCheckedChange={() => toggleSampleMetric(name)}
                        onSelect={(e) => e.preventDefault()}
                      >
                        {formatMetricName(name)}
                      </DropdownMenuCheckboxItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </div>
          )}
          {/* Second line: rollout metrics */}
          {secondaryItems.length > 0 && (
            <div className="flex items-center gap-1.5 text-xs min-w-0 flex-wrap">
              {secondaryItems.map((item, i) => (
                <span
                  key={item.label}
                  className="inline-flex items-center gap-1.5"
                >
                  <span className="text-muted-foreground">{item.label}</span>
                  <span
                    className={cn(
                      "font-medium tabular-nums",
                      !item.rangeColor &&
                        item.highlight === "positive" &&
                        "text-emerald-600 dark:text-emerald-400",
                      !item.rangeColor &&
                        item.highlight === "negative" &&
                        "text-red-500 dark:text-red-400",
                      !item.rangeColor &&
                        !item.highlight &&
                        "text-foreground/80",
                    )}
                    style={
                      item.rangeColor ? { color: item.rangeColor } : undefined
                    }
                  >
                    {item.value}
                  </span>
                  {i < secondaryItems.length - 1 && (
                    <span className="text-foreground/20">·</span>
                  )}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Golden answers */}
      {visibleGoldenAnswers.length > 0 && (
        <Collapsible
          open={goldenOpen}
          onOpenChange={setGoldenOpen}
          className="mb-2"
        >
          <CollapsibleTrigger asChild>
            <div className="py-1.5 px-2 -mx-2 cursor-pointer bg-amber-500/[0.03] hover:bg-amber-500/[0.07] rounded transition-colors overflow-hidden">
              <div className="flex items-center gap-1.5 min-w-0">
                <span className="text-amber-600/80 dark:text-amber-400/70 font-medium text-[11px] uppercase tracking-wide shrink-0">
                  Golden
                </span>
                {!goldenOpen && (
                  <span className="text-xs text-muted-foreground font-normal truncate min-w-0">
                    {visibleGoldenAnswers
                      .map((a) => `${formatMetricName(a.key)}: ${a.value}`)
                      .join(" · ")}
                  </span>
                )}
                <div className="flex items-center gap-1.5 ml-auto shrink-0">
                  <ChevronDown
                    className={cn(
                      "h-4 w-4 text-muted-foreground transition-transform",
                      !goldenOpen && "-rotate-90",
                    )}
                  />
                </div>
              </div>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="py-2 px-2 -mx-2 bg-amber-500/[0.03] rounded-b">
              <div className="flex items-baseline gap-2 text-xs min-w-0 flex-wrap">
                {visibleGoldenAnswers.map((answer, i) => (
                  <span
                    key={answer.key}
                    className="inline-flex items-baseline gap-1.5"
                  >
                    {i > 0 && <span className="text-foreground/20">·</span>}
                    <span className="text-muted-foreground">
                      {formatMetricName(answer.key)}
                    </span>
                    <span className="font-medium text-foreground/80 whitespace-pre-wrap break-words">
                      {renderOptions.length > 0 && answer.value ? (
                        <RenderedContent
                          content={answer.value}
                          renderOptions={renderOptions}
                        />
                      ) : (
                        answer.value
                      )}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* System Prompt collapsible */}
      {sample.prompt?.system_prompt && (
        <Collapsible
          open={systemOpen}
          onOpenChange={setSystemOpen}
          className="mb-2"
        >
          <CollapsibleTrigger asChild>
            <div className="py-1.5 px-2 -mx-2 cursor-pointer bg-muted/50 hover:bg-muted rounded transition-colors">
              <div className="flex items-center gap-1.5 min-w-0">
                <h3 className="text-sm font-semibold shrink-0 w-[4.5rem]">
                  System
                </h3>
                {!systemOpen && (
                  <span className="text-xs text-muted-foreground font-normal truncate min-w-0">
                    {getContentPreview(
                      sample.prompt.system_prompt,
                      60,
                      formatThinkBlocks,
                    )}
                  </span>
                )}
                <div className="flex items-center gap-1.5 ml-auto shrink-0">
                  {sample.prompt.tokens_system_prompt != null && (
                    <span className="text-xs text-foreground/40 font-normal">
                      {sample.prompt.tokens_system_prompt.toLocaleString()}{" "}
                      tokens
                    </span>
                  )}
                  <InlineCopyButton text={sample.prompt.system_prompt} />
                  <ChevronDown
                    className={cn(
                      "h-4 w-4 text-muted-foreground transition-transform",
                      !systemOpen && "-rotate-90",
                    )}
                  />
                </div>
              </div>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="py-2">
              <ContentWithThinkBlocks
                content={sample.prompt.system_prompt}
                renderOptions={renderOptions}
                enableThinkBlocks={false}
              />
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* User Prompt collapsible */}
      {sample.prompt && (
        <Collapsible
          open={promptOpen}
          onOpenChange={setPromptOpen}
          className="mb-2"
        >
          <CollapsibleTrigger asChild>
            <div className="py-1.5 px-2 -mx-2 cursor-pointer bg-muted/50 hover:bg-muted rounded transition-colors">
              <div className="flex items-center gap-1.5 min-w-0">
                <h3 className="text-sm font-semibold shrink-0 w-[4.5rem]">
                  User
                </h3>
                {!promptOpen && (
                  <span className="text-xs text-muted-foreground font-normal truncate min-w-0">
                    {getContentPreview(
                      sample.prompt.prompt,
                      60,
                      formatThinkBlocks,
                    )}
                  </span>
                )}
                <div className="flex items-center gap-1.5 ml-auto shrink-0">
                  {sample.prompt.tokens_prompt != null && (
                    <span className="text-xs text-foreground/40 font-normal">
                      {sample.prompt.tokens_prompt.toLocaleString()} tokens
                    </span>
                  )}
                  <InlineCopyButton text={sample.prompt.prompt} />
                  <ChevronDown
                    className={cn(
                      "h-4 w-4 text-muted-foreground transition-transform",
                      !promptOpen && "-rotate-90",
                    )}
                  />
                </div>
              </div>
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="py-2">
              <ContentWithThinkBlocks
                content={sample.prompt.prompt}
                renderOptions={renderOptions}
                enableThinkBlocks={false}
              />
            </div>
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Generation turns with env_responses and tool_calls */}
      {sample.generations.map((gen, genIdx) => {
        // Each generation uses a sequential collapsible index
        const collapsibleIdx = genIdx * 2
        const isGenOpen = turnsOpen[collapsibleIdx] ?? defaultSectionsOpen

        // Find matching env_response for this generation_idx
        const matchingEnvResponse = sample.env_responses.find(
          (er) => er.generation_idx === gen.generation_idx,
        )

        // Find matching tool_calls for this generation_idx
        const matchingToolCalls = sample.tool_calls
          .filter((tc) => tc.generation_idx === gen.generation_idx)
          .sort((a, b) => a.tool_call_idx - b.tool_call_idx)

        const envCollapsibleIdx = collapsibleIdx + 1
        const isEnvOpen = turnsOpen[envCollapsibleIdx] ?? defaultSectionsOpen

        return (
          <div key={`${gen.sample_id}-gen-${gen.generation_idx}`}>
            {/* Assistant (generation) turn */}
            <Collapsible
              open={isGenOpen}
              onOpenChange={() => toggleTurn(collapsibleIdx)}
              className="mb-2"
            >
              <CollapsibleTrigger asChild>
                <div className="py-1.5 px-2 -mx-2 cursor-pointer bg-muted/50 hover:bg-muted rounded transition-colors">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <h3 className="text-sm font-semibold shrink-0 w-[4.5rem]">
                      Assistant
                    </h3>
                    {!isGenOpen && (
                      <span className="text-xs text-muted-foreground font-normal truncate min-w-0">
                        {getContentPreview(
                          gen.content,
                          60,
                          formatThinkBlocks,
                        )}
                      </span>
                    )}
                    <div className="flex items-center gap-1.5 ml-auto shrink-0">
                      {gen.tokens != null && (
                        <span className="text-xs text-foreground/40 font-normal">
                          {gen.tokens.toLocaleString()} tokens
                        </span>
                      )}
                      {generationsCount > 1 && (
                        <span className="text-xs text-foreground/40 font-normal">
                          Turn {gen.generation_idx}
                        </span>
                      )}
                      <InlineCopyButton text={gen.content} />
                      <ChevronDown
                        className={cn(
                          "h-4 w-4 text-muted-foreground transition-transform",
                          !isGenOpen && "-rotate-90",
                        )}
                      />
                    </div>
                  </div>
                </div>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="py-2">
                  <ContentWithThinkBlocks
                    content={gen.content}
                    renderOptions={renderOptions}
                    enableThinkBlocks={formatThinkBlocks}
                  />
                  {(() => {
                    const genInfos = infoTurnsByGenerationIdx.get(gen.generation_idx)
                    if (!genInfos || genInfos.length === 0) return null
                    return (
                      <div className="mt-2 space-y-1">
                        {genInfos.map((info, infoIdx) => (
                          <InfoTurnCollapsible
                            key={`${info.info_key}-${infoIdx}`}
                            info={info}
                            renderOptions={renderOptions}
                          />
                        ))}
                      </div>
                    )
                  })()}
                </div>
              </CollapsibleContent>
            </Collapsible>

            {/* Environment response (if exists for this generation) */}
            {matchingEnvResponse && (
              <Collapsible
                open={isEnvOpen}
                onOpenChange={() => toggleTurn(envCollapsibleIdx)}
                className="mb-2"
              >
                <CollapsibleTrigger asChild>
                  <div className="py-1.5 px-2 -mx-2 cursor-pointer bg-muted/50 hover:bg-muted rounded transition-colors">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <h3 className="text-sm font-semibold shrink-0 w-[4.5rem]">
                        Environment
                      </h3>
                      {!isEnvOpen && (
                        <span className="text-xs text-muted-foreground font-normal truncate min-w-0">
                          {getContentPreview(
                            matchingEnvResponse.content,
                            60,
                            false,
                          )}
                        </span>
                      )}
                      <div className="flex items-center gap-1.5 ml-auto shrink-0">
                        {matchingEnvResponse.tokens != null && (
                          <span className="text-xs text-foreground/40 font-normal">
                            {matchingEnvResponse.tokens.toLocaleString()} tokens
                          </span>
                        )}
                        {matchingToolCalls.length > 0 && (
                          <span className="text-xs text-foreground/40 font-normal">
                            {matchingToolCalls.length} tool{matchingToolCalls.length !== 1 ? "s" : ""}
                          </span>
                        )}
                        <InlineCopyButton text={matchingEnvResponse.content} />
                        <ChevronDown
                          className={cn(
                            "h-4 w-4 text-muted-foreground transition-transform",
                            !isEnvOpen && "-rotate-90",
                          )}
                        />
                      </div>
                    </div>
                  </div>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="py-2">
                    {/* Tool calls (collapsible within the env_response) */}
                    {matchingToolCalls.length > 0 && (
                      <div className="mb-2 space-y-1">
                        {matchingToolCalls.map((tc) => (
                          <ToolCallCollapsible
                            key={`tc-${tc.tool_call_idx}`}
                            toolCall={tc}
                            renderOptions={renderOptions}
                          />
                        ))}
                      </div>
                    )}
                    <ContentWithThinkBlocks
                      content={matchingEnvResponse.content}
                      renderOptions={renderOptions}
                      enableThinkBlocks={false}
                    />
                  </div>
                </CollapsibleContent>
              </Collapsible>
            )}
          </div>
        )
      })}
    </div>
  )
}

function InfoTurnCollapsible({
  info,
  renderOptions = [],
}: {
  info: InfoTurn
  renderOptions?: RolloutsRenderOption[]
}) {
  const [open, setOpen] = useState(false)
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger asChild>
        <div className="py-1 px-2 -mx-2 cursor-pointer bg-accent/40 hover:bg-accent/60 rounded transition-colors flex items-center gap-1.5">
          <ChevronDown
            className={cn(
              "h-3 w-3 text-muted-foreground transition-transform shrink-0",
              !open && "-rotate-90",
            )}
          />
          <span className="text-xs font-medium text-muted-foreground">
            {info.info_key}
          </span>
          {!open && info.info_value && (
            <span className="text-xs text-muted-foreground/50 truncate min-w-0">
              {info.info_value.slice(0, 80)}
              {info.info_value.length > 80 ? "…" : ""}
            </span>
          )}
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="py-1.5 px-2 text-xs bg-accent/20 rounded mt-0.5">
          {renderOptions.length > 0 && info.info_value ? (
            <RenderedContent
              content={info.info_value}
              renderOptions={renderOptions}
            />
          ) : (
            <pre className="whitespace-pre-wrap font-sans">
              {info.info_value}
            </pre>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function ToolCallCollapsible({
  toolCall,
  renderOptions = [],
}: {
  toolCall: ToolCallRow
  renderOptions?: RolloutsRenderOption[]
}) {
  const [open, setOpen] = useState(false)
  const statusColor = toolCall.success
    ? "text-emerald-600 dark:text-emerald-400"
    : "text-red-500 dark:text-red-400"
  const statusLabel = toolCall.success ? "success" : "error"
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger asChild>
        <div className="py-1 px-2 -mx-2 cursor-pointer bg-accent/40 hover:bg-accent/60 rounded transition-colors flex items-center gap-1.5">
          <ChevronDown
            className={cn(
              "h-3 w-3 text-muted-foreground transition-transform shrink-0",
              !open && "-rotate-90",
            )}
          />
          <span className="text-xs font-medium text-muted-foreground">
            {toolCall.tool_name}
          </span>
          <span className={cn("text-xs font-medium", statusColor)}>
            {statusLabel}
          </span>
          {!open && toolCall.arguments && (
            <span className="text-xs text-muted-foreground/50 truncate min-w-0">
              {toolCall.arguments.slice(0, 60)}
              {toolCall.arguments.length > 60 ? "…" : ""}
            </span>
          )}
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="py-1.5 px-2 text-xs bg-accent/20 rounded mt-0.5 space-y-2">
          {toolCall.arguments && (
            <div>
              <div className="text-muted-foreground font-medium mb-0.5">Arguments</div>
              <pre className="whitespace-pre-wrap font-sans break-words">
                {toolCall.arguments}
              </pre>
            </div>
          )}
          {toolCall.result != null && (
            <div>
              <div className="text-muted-foreground font-medium mb-0.5">Result</div>
              {renderOptions.length > 0 ? (
                <RenderedContent
                  content={toolCall.result}
                  renderOptions={renderOptions}
                />
              ) : (
                <pre className="whitespace-pre-wrap font-sans break-words">
                  {toolCall.result}
                </pre>
              )}
            </div>
          )}
          {toolCall.error != null && (
            <div>
              <div className="text-red-500 dark:text-red-400 font-medium mb-0.5">Error</div>
              <pre className="whitespace-pre-wrap font-sans break-words text-red-500 dark:text-red-400">
                {toolCall.error}
              </pre>
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}


/** Strip <think>...</think> blocks from content and trim leading whitespace */
function stripThinkTags(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>\s*/g, "").trimStart()
}

/** Get a truncated content preview; optionally strip <think> blocks first */
function getContentPreview(
  content: string | null | undefined,
  maxLength: number = 60,
  stripThinkBlocks: boolean = true,
): string {
  if (!content) return ""
  const previewContent = stripThinkBlocks ? stripThinkTags(content) : content
  const cleaned = previewContent.replace(/\s+/g, " ").trim()
  if (cleaned.length <= maxLength) return cleaned
  return cleaned.slice(0, maxLength) + "…"
}

/** Parse content into segments of plain text and <think>...</think> blocks */
function parseThinkBlocks(
  content: string,
): Array<{ type: "text"; value: string } | { type: "think"; value: string }> {
  const segments: Array<
    { type: "text"; value: string } | { type: "think"; value: string }
  > = []
  const regex = /<think>([\s\S]*?)<\/think>/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = regex.exec(content)) !== null) {
    // Text before this <think> block
    if (match.index > lastIndex) {
      segments.push({
        type: "text",
        value: content.slice(lastIndex, match.index),
      })
    }
    segments.push({ type: "think", value: match[1].replace(/^\n/, "") })
    lastIndex = regex.lastIndex
  }

  // Remaining text after last match
  if (lastIndex < content.length) {
    segments.push({ type: "text", value: content.slice(lastIndex) })
  }

  return segments
}

/** Get a truncated preview of think block content */
function getThinkPreview(content: string, maxLength: number = 80): string {
  const cleaned = content.replace(/\s+/g, " ").trim()
  if (cleaned.length <= maxLength) return cleaned
  return cleaned.slice(0, maxLength) + "…"
}

function CollapsibleThinkBlock({ content }: { content: string }) {
  const [isOpen, setIsOpen] = useState(true)

  return (
    <span className="block">
      <span
        className="cursor-pointer select-none inline-flex items-center gap-0.5 text-foreground/40 hover:text-foreground/60 transition-colors"
        onClick={() => setIsOpen((o) => !o)}
      >
        {"<think>"}
        {!isOpen && (
          <>
            <span className="text-foreground/40 italic text-xs ml-1">
              {getThinkPreview(content)}
            </span>
            <span>{"</think>"}</span>
            <ChevronRight className="h-3.5 w-3.5 inline-block shrink-0" />
          </>
        )}
        {isOpen && (
          <ChevronDown className="h-3.5 w-3.5 inline-block shrink-0" />
        )}
      </span>
      {isOpen && (
        <span className="block text-foreground/70">
          {content}
          <span
            className="cursor-pointer select-none inline-flex items-center gap-0.5 text-foreground/40 hover:text-foreground/60 transition-colors"
            onClick={() => setIsOpen(false)}
          >
            {"</think>"}
          </span>
        </span>
      )}
    </span>
  )
}

/**
 * Parse content into alternating text and math segments for LaTeX-only rendering.
 * Handles $$...$$, \[...\], $...$, and \(...\) delimiters.
 */
function parseLatexSegments(
  content: string,
): Array<{ type: "text" | "display" | "inline"; value: string }> {
  const segments: Array<{
    type: "text" | "display" | "inline"
    value: string
  }> = []
  // Match display math ($$...$$ or \[...\]) and inline math ($...$ or \(...\))
  // Order matters: $$ before $ to avoid partial matches
  const regex =
    /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\$(?!\$)(?:[^$\\]|\\.)+?\$|\\\((?:[^\\]|\\.)*?\\\))/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      segments.push({
        type: "text",
        value: content.slice(lastIndex, match.index),
      })
    }
    const raw = match[1]
    if (raw.startsWith("$$")) {
      segments.push({ type: "display", value: raw.slice(2, -2) })
    } else if (raw.startsWith("\\[")) {
      segments.push({ type: "display", value: raw.slice(2, -2) })
    } else if (raw.startsWith("\\(")) {
      segments.push({ type: "inline", value: raw.slice(2, -2) })
    } else {
      segments.push({ type: "inline", value: raw.slice(1, -1) })
    }
    lastIndex = regex.lastIndex
  }
  if (lastIndex < content.length) {
    segments.push({ type: "text", value: content.slice(lastIndex) })
  }
  return segments
}

/** Parse content into segments of plain text and fenced code blocks */
function parseCodeBlocks(
  content: string,
): Array<
  | { type: "text"; value: string }
  | { type: "code"; value: string; language?: string }
> {
  const segments: Array<
    | { type: "text"; value: string }
    | { type: "code"; value: string; language?: string }
  > = []
  const regex = /^```(\w*)\n([\s\S]*?)^```\s*$/gm
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      segments.push({
        type: "text",
        value: content.slice(lastIndex, match.index),
      })
    }
    segments.push({
      type: "code",
      value: match[2].replace(/\n$/, ""),
      language: match[1] || undefined,
    })
    lastIndex = regex.lastIndex
  }

  if (lastIndex < content.length) {
    segments.push({ type: "text", value: content.slice(lastIndex) })
  }

  if (segments.length === 0) {
    segments.push({ type: "text", value: content })
  }

  return segments
}

/** Renders a styled fenced code block */
function StyledCodeBlock({
  code,
  language,
}: {
  code: string
  language?: string
}) {
  const [copied, setCopied] = useState(false)
  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation()
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <div className="my-2 rounded-md border bg-muted/50 overflow-hidden">
      {language && (
        <div className="flex items-center justify-between px-3 py-1 text-xs text-muted-foreground border-b bg-muted/30">
          <span>{language}</span>
          <button
            onClick={handleCopy}
            className="hover:text-foreground transition-colors"
            title="Copy code"
          >
            {copied ? (
              <CheckIcon className="h-3 w-3" />
            ) : (
              <CopyIcon className="h-3 w-3" />
            )}
          </button>
        </div>
      )}
      <pre className="p-3 text-sm font-sans whitespace-pre-wrap break-words overflow-x-auto">
        <code>{code}</code>
      </pre>
    </div>
  )
}

/** Renders preformatted text with LaTeX math processing */
function LatexPreBlock({ content }: { content: string }) {
  const katexOverflowClass = "katex-scroll"
  const segments = parseLatexSegments(content)
  return (
    <div
      className={cn(
        "whitespace-pre-wrap break-words [&_.katex]:text-sm",
        katexOverflowClass,
      )}
    >
      {segments.map((seg, i) => {
        if (seg.type === "text") {
          return <span key={i}>{seg.value}</span>
        }
        try {
          const html = katex.renderToString(seg.value, {
            displayMode: seg.type === "display",
            throwOnError: false,
          })
          return <span key={i} dangerouslySetInnerHTML={{ __html: html }} />
        } catch {
          return (
            <span key={i}>
              {seg.type === "display" ? `$$${seg.value}$$` : `$${seg.value}$`}
            </span>
          )
        }
      })}
    </div>
  )
}

/** Render a text segment (non-code) with optional Markdown and/or LaTeX */
function RenderedTextSegment({
  content,
  renderMarkdown,
  renderLatex,
}: {
  content: string
  renderMarkdown: boolean
  renderLatex: boolean
}) {
  const katexOverflowClass = "katex-scroll"

  if (renderMarkdown) {
    const remarkPlugins: Array<typeof remarkMath> = []
    const rehypePlugins: Array<typeof rehypeKatex> = []
    if (renderLatex) {
      remarkPlugins.push(remarkMath)
      rehypePlugins.push(rehypeKatex)
    }
    return (
      <div
        className={cn(
          "prose prose-sm dark:prose-invert max-w-none break-words [&>*:first-child]:mt-0 [&>*:last-child]:mb-0 [&>*+*]:mt-[0.75em]",
          renderLatex && katexOverflowClass,
        )}
      >
        <ReactMarkdown
          remarkPlugins={remarkPlugins}
          rehypePlugins={rehypePlugins}
        >
          {content}
        </ReactMarkdown>
      </div>
    )
  }

  if (renderLatex) {
    return <LatexPreBlock content={content} />
  }

  return (
    <div className="whitespace-pre-wrap break-words">{content}</div>
  )
}

/** Renders text content with optional Markdown, LaTeX, and/or Code rendering */
function RenderedContent({
  content,
  renderOptions = [],
}: {
  content: string
  renderOptions?: RolloutsRenderOption[]
}) {
  const renderMarkdown = renderOptions.includes("markdown")
  const renderLatex = renderOptions.includes("latex")
  const renderCode = renderOptions.includes("code")

  if (!renderMarkdown && !renderLatex && !renderCode) {
    return <>{content}</>
  }

  // Code blocks are parsed FIRST so StyledCodeBlock always handles them,
  // regardless of whether markdown is also enabled.
  if (renderCode) {
    const codeSegments = parseCodeBlocks(content)
    if (codeSegments.some((s) => s.type === "code")) {
      return (
        <>
          {codeSegments.map((seg, i) => {
            if (seg.type === "code") {
              return (
                <StyledCodeBlock
                  key={i}
                  code={seg.value}
                  language={seg.language}
                />
              )
            }
            // Text segment between code blocks — render with markdown/latex/plain
            return (
              <RenderedTextSegment
                key={i}
                content={seg.value}
                renderMarkdown={renderMarkdown}
                renderLatex={renderLatex}
              />
            )
          })}
        </>
      )
    }
  }

  // No code blocks found (or code disabled) — render full content with markdown/latex
  return (
    <RenderedTextSegment
      content={content}
      renderMarkdown={renderMarkdown}
      renderLatex={renderLatex}
    />
  )
}

function ContentWithThinkBlocks({
  content,
  renderOptions = [],
  enableThinkBlocks = true,
}: {
  content: string
  renderOptions?: RolloutsRenderOption[]
  enableThinkBlocks?: boolean
}) {
  const segments = useMemo(
    () =>
      enableThinkBlocks
        ? parseThinkBlocks(content)
        : [{ type: "text" as const, value: content }],
    [content, enableThinkBlocks],
  )
  const hasRenderOptions = renderOptions.length > 0

  // If no think blocks found, render plain text (or rendered content)
  if (segments.length === 1 && segments[0].type === "text") {
    if (hasRenderOptions) {
      return (
        <div className="text-sm text-foreground/90">
          <RenderedContent content={content} renderOptions={renderOptions} />
        </div>
      )
    }
    return (
      <div className="text-sm whitespace-pre-wrap break-words text-foreground/90">
        {content}
      </div>
    )
  }

  if (hasRenderOptions) {
    return (
      <div className="text-sm text-foreground/90">
        {segments.map((seg, i) =>
          seg.type === "text" ? (
            <RenderedContent
              key={i}
              content={seg.value}
              renderOptions={renderOptions}
            />
          ) : (
            <CollapsibleThinkBlock key={i} content={seg.value} />
          ),
        )}
      </div>
    )
  }

  return (
    <div className="text-sm whitespace-pre-wrap break-words text-foreground/90">
      {segments.map((seg, i) =>
        seg.type === "text" ? (
          <span key={i}>{seg.value}</span>
        ) : (
          <CollapsibleThinkBlock key={i} content={seg.value} />
        ),
      )}
    </div>
  )
}

function InlineCopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation()
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button
      onClick={handleCopy}
      className="p-0.5 rounded hover:bg-muted-foreground/10 transition-colors"
      title={copied ? "Copied!" : "Copy to clipboard"}
    >
      {copied ? (
        <CheckIcon className="h-3.5 w-3.5 text-foreground/30" />
      ) : (
        <CopyIcon className="h-3.5 w-3.5 text-foreground/30" />
      )}
    </button>
  )
}

function formatMetricName(name: string): string {
  // Convert snake_case to Title Case and remove "_reward" suffix
  return name
    .replace(/_reward$/, "")
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ")
}

function formatMetricValue(value: number): string {
  // Format value: show as integer if it's whole, otherwise 2 decimal places
  if (Number.isInteger(value)) {
    return value.toString()
  }
  return value.toFixed(2)
}

function lookupMetricRange(
  envMetricRanges: EnvMetricRanges | undefined,
  metricEnv: string | null,
  metricKey: string,
): MetricRange | undefined {
  if (!envMetricRanges) return undefined
  if (metricEnv) {
    const exact = envMetricRanges.get(metricEnv)?.get(metricKey)
    if (exact) return exact
  }
  for (const metricMap of envMetricRanges.values()) {
    const found = metricMap.get(metricKey)
    if (found) return found
  }
  return undefined
}

/**
 * Compute an inline CSS color for a metric value given its [min, max] range.
 * Returns a hue-interpolated color from red (0°) to green (120°) via HSL,
 * or undefined when no range is available.
 * When invert is true, the color scale is flipped: min=green, max=red
 * (for metrics where lower is better, e.g. num_errors).
 */
function metricRangeColor(
  value: number,
  min: number | null | undefined,
  max: number | null | undefined,
  invert?: boolean,
): string | undefined {
  if (min == null || max == null) return undefined
  if (min >= max) return undefined
  // Clamp t to [0, 1]
  let t = Math.max(0, Math.min(1, (value - min) / (max - min)))
  // When inverted, flip the color scale so that min=green (good) and max=red (bad)
  if (invert) t = 1 - t
  // Hue: 0 (red) → 120 (green)
  const hue = t * 120
  return `hsl(${hue.toFixed(0)} 80% 38%)`
}

export function RawTextDialog({
  rawString,
  totalTokens,
  turns,
}: {
  rawString: string
  totalTokens?: number | null
  turns?: number
}) {
  const [copied, setCopied] = useState(false)
  const [showLineBreaks, setShowLineBreaks] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(rawString)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Render text with visible line break markers
  const renderTextWithLineBreaks = (text: string) => {
    const lines = text.split("\n")
    return lines.map((line, idx) => (
      <span key={idx}>
        {line}
        {idx < lines.length - 1 && (
          <>
            <span className="text-muted-foreground/50">\n</span>
            {"\n"}
          </>
        )}
      </span>
    ))
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="h-7 px-2 text-xs">
          Raw Text
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[75vw] max-h-[80vh]">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <DialogTitle>Raw Text</DialogTitle>
            <span className="text-xs text-muted-foreground">
              {totalTokens != null && (
                <>{totalTokens.toLocaleString()} tokens</>
              )}
              {totalTokens != null && turns != null && turns > 0 && " • "}
              {turns != null && turns > 0 && (
                <>
                  {turns} {turns === 1 ? "turn" : "turns"}
                </>
              )}
            </span>
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-xs gap-1"
              onClick={handleCopy}
            >
              {copied ? (
                <CheckIcon className="size-3" />
              ) : (
                <CopyIcon className="size-3" />
              )}
              {copied ? "Copied" : "Copy"}
            </Button>
            <Toggle
              variant="outline"
              size="sm"
              pressed={showLineBreaks}
              onPressedChange={setShowLineBreaks}
              className="text-xs"
            >
              Line breaks
            </Toggle>
          </div>
        </DialogHeader>
        <ScrollArea className="h-[60vh] w-full rounded-md border bg-background p-4">
          <pre className="text-xs whitespace-pre-wrap break-words font-sans">
            {showLineBreaks ? renderTextWithLineBreaks(rawString) : rawString}
          </pre>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}
