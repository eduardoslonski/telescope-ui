
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import type { ReactNode } from "react"
import { useAtom, useAtomValue } from "jotai"
import type { SetStateAction, WritableAtom } from "jotai"
import { Menu, PanelLeftClose, Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { PaginationControls } from "@/components/pagination-controls"
import { cn } from "@/lib/utils"
import {
  rolloutsSamplePickerSidebarOpenAtom,
  rolloutsSelectedGroupIdAtom,
  rolloutsSelectedSampleIdxAtom,
  rolloutsSamplePickerViewModeAtom,
  rolloutsSamplePickerDisplayMetricsAtom,
  rolloutsSamplePickerTooltipsAtom,
  selectedRunPathAtom,
  darkModeAtom,
} from "@/lib/atoms"
import { useRunSummary } from "@/hooks/use-run-data"
import {
  extractEnvRewardRanges,
  extractEnvMetricRanges,
  parseDataMetricRanges,
  mergeEnvMetricRanges,
  normalizeEnvItems,
  type EnvRewardRanges,
  type EnvMetricRanges,
  type MetricRange,
} from "@/components/rollouts-view"
import type { Prompt, Rollout, SampleData, RolloutMetric } from "@/lib/types"

// Grouped sample for picker
interface GroupedSample {
  sample_idx: number
  group_id: number
  prompt: Prompt | null
  responsePreview: string
  turnsCount: number
  turnsData: Rollout[]
  reward: number | null
  advantage: number | null
  assistantTokens: number | null // average tokens per turn (for color scale)
  assistantTokensTotal: number | null // total tokens (for display)
}

// Group with its samples
export interface Group {
  group_id: number
  prompt: Prompt | null
  samples: GroupedSample[]
  sampleCount: number
}

function truncateText(
  text: string | null | undefined,
  maxLength: number = 60,
): string {
  if (!text) return ""
  const cleaned = text.replace(/\s+/g, " ").trim()
  if (cleaned.length <= maxLength) return cleaned
  return cleaned.slice(0, maxLength) + "..."
}

function rewardToColor(reward: number | null): string | null {
  if (reward === null || Number.isNaN(reward)) return null
  const clamped = Math.max(0, Math.min(1, reward))
  const hue = clamped * 120
  return `hsl(${hue} 70% 38%)`
}

function rewardRangeColor(
  value: number,
  min: number | null | undefined,
  max: number | null | undefined,
  invert?: boolean,
): string | undefined {
  if (min == null || max == null) return undefined
  if (min >= max) return undefined
  let t = Math.max(0, Math.min(1, (value - min) / (max - min)))
  if (invert) t = 1 - t
  const hue = t * 120
  return `hsl(${hue.toFixed(0)} 80% 38%)`
}

function extractConfigNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null
  const rawValue =
    typeof value === "object" && value !== null && "value" in value
      ? (value as { value: unknown }).value
      : value

  if (typeof rawValue === "number" && Number.isFinite(rawValue)) return rawValue
  if (typeof rawValue === "string") {
    const parsed = Number(rawValue)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

function tokensToColor(
  tokens: number | null,
  maxTokens: number | null,
  maxLightness: number = 85,
  darkMode: boolean = false,
): string | null {
  if (tokens === null || maxTokens === null || maxTokens <= 0) return null
  const clamped = Math.max(0, Math.min(maxTokens, tokens))
  if (darkMode) {
    // Dark mode: higher tokens → lighter (more visible on dark bg)
    const minLightness = 30
    const maxDarkLightness = 90
    const lightness = minLightness + (clamped / maxTokens) * (maxDarkLightness - minLightness)
    return `hsl(0 0% ${lightness}%)`
  }
  // Light mode: higher tokens → darker
  const lightness = maxLightness - (clamped / maxTokens) * (maxLightness - 5)
  return `hsl(0 0% ${lightness}%)`
}

type AtomWithState<T> = WritableAtom<T, [SetStateAction<T>], void>

interface RolloutsSamplePickerSidebarProps {
  prompts?: Prompt[]
  data?: Rollout[]
  samplesData?: SampleData[]
  rolloutMetrics?: RolloutMetric[]
  availableMetricNames?: string[]
  isLoading?: boolean
  step?: number
  // Step navigation props (pagination style)
  currentStepIndex: number
  totalSteps: number
  onStepChange: (index: number) => void
  hasSteps: boolean
  stepValues?: number[]
  openAtom?: AtomWithState<boolean>
  selectedGroupIdAtom?: AtomWithState<number | null>
  selectedSampleIdxAtom?: AtomWithState<number | null>
  viewModeAtom?: AtomWithState<"groups" | "samples">
  getGroupBadge?: (group: Group) => ReactNode
  renderLeftControls?: (props: {
    viewMode: "groups" | "samples"
    setViewMode: (mode: "groups" | "samples") => void
    hasSelectedGroupInStep: boolean
    groups: Group[]
  }) => ReactNode
  onGroupClick?: (groupId: number, group: Group) => void
  hideRewardAdvantage?: boolean
  /** Override the env details used for reward/metric ranges (e.g. from summary.eval_env_details for evals). When unset, falls back to config.environments. */
  envDetailsOverride?: unknown
}

// Toggle button component to open the sidebar
export function RolloutsSamplePickerSidebarToggle({
  openAtom = rolloutsSamplePickerSidebarOpenAtom,
}: {
  openAtom?: AtomWithState<boolean>
} = {}) {
  const [isOpen, setIsOpen] = useAtom(openAtom)
  const selectedRunPath = useAtomValue(selectedRunPathAtom)

  if (!selectedRunPath || isOpen) return null

  return (
    <Button variant="outline" size="sm" onClick={() => setIsOpen(true)}>
      Samples
    </Button>
  )
}

const builtinMetricOrder = [
  "gen_length",
  "gen_length_sum",
  "gen_length_avg",
  "reward",
  "advantage",
]

export function RolloutsSamplePickerSidebar({
  prompts,
  data,
  samplesData,
  rolloutMetrics,
  availableMetricNames,
  isLoading,
  step,
  currentStepIndex,
  totalSteps,
  onStepChange,
  hasSteps,
  stepValues,
  openAtom = rolloutsSamplePickerSidebarOpenAtom,
  selectedGroupIdAtom = rolloutsSelectedGroupIdAtom,
  selectedSampleIdxAtom = rolloutsSelectedSampleIdxAtom,
  viewModeAtom = rolloutsSamplePickerViewModeAtom,
  getGroupBadge,
  renderLeftControls,
  onGroupClick: onGroupClickProp,
  hideRewardAdvantage = false,
  envDetailsOverride,
}: RolloutsSamplePickerSidebarProps) {
  const [isOpen, setIsOpen] = useAtom(openAtom)
  const [selectedGroupId, setSelectedGroupId] = useAtom(selectedGroupIdAtom)
  const [selectedSampleIdx, setSelectedSampleIdx] = useAtom(
    selectedSampleIdxAtom,
  )
  const [viewMode, setViewMode] = useAtom(viewModeAtom)
  const [displayMetrics, setDisplayMetrics] = useAtom(
    rolloutsSamplePickerDisplayMetricsAtom,
  )
  const [tooltipsEnabled, setTooltipsEnabled] = useAtom(
    rolloutsSamplePickerTooltipsAtom,
  )
  const selectedRunPath = useAtomValue(selectedRunPathAtom)
  const darkMode = useAtomValue(darkModeAtom)
  const { data: summaryData } = useRunSummary(
    selectedRunPath || "",
    !!selectedRunPath,
    true,
  )
  const rawMaxTokens = summaryData?.config?.max_tokens
  const maxTokensFromConfig = useMemo(
    () => extractConfigNumber(rawMaxTokens),
    [rawMaxTokens],
  )
  const rawGroupSize = summaryData?.config?.group_size
  const groupSizeFromConfig = useMemo(
    () => extractConfigNumber(rawGroupSize),
    [rawGroupSize],
  )
  // Advantage range: -sqrt(group_size - 1) to +sqrt(group_size - 1)
  const advantageRange = useMemo<{ min: number; max: number } | null>(() => {
    if (groupSizeFromConfig == null || groupSizeFromConfig <= 1) return null
    const bound = Math.sqrt(groupSizeFromConfig - 1)
    return { min: -bound, max: bound }
  }, [groupSizeFromConfig])
  const envDetailsValue =
    envDetailsOverride ?? summaryData?.config?.environments
  const envRewardRanges = useMemo(
    () => extractEnvRewardRanges(envDetailsValue),
    [envDetailsValue],
  )
  const envMetricRanges = useMemo(() => {
    const configRanges = extractEnvMetricRanges(envDetailsValue)
    const summaryRanges = extractEnvMetricRanges(summaryData?.summary?.env_details)
    const dataRanges = parseDataMetricRanges(summaryData?.data_metric_ranges)
    return mergeEnvMetricRanges(
      mergeEnvMetricRanges(configRanges, summaryRanges),
      dataRanges,
    )
  }, [envDetailsValue, summaryData?.summary?.env_details, summaryData?.data_metric_ranges])
  // Check if any environment is multi-turn to decide which gen length metrics to show.
  // Falls back to env_details in the wandb run summary when config.environments lacks is_multi_turn.
  const hasMultiTurn = useMemo(() => {
    const envSources = [envDetailsValue, summaryData?.summary?.env_details]
    for (const src of envSources) {
      const items = normalizeEnvItems(src)
      if (items.length === 0) continue
      return items.some(
        (e) => (e as Record<string, unknown>)?.is_multi_turn === true,
      )
    }
    return false
  }, [envDetailsValue, summaryData?.summary?.env_details])

  // Build metrics lookup: sample_idx -> metric_name -> { value, env }
  // Keep previous value during loading to prevent metric columns from disappearing
  type MetricsBySampleIdx = Map<number, Map<string, { value: number; env: string | null }>>
  const [cachedMetricsBySampleIdx, setCachedMetricsBySampleIdx] = useState<MetricsBySampleIdx>(new Map())
  const computedMetricsBySampleIdx: MetricsBySampleIdx | null = useMemo(() => {
    if (!rolloutMetrics) return null
    const map: MetricsBySampleIdx = new Map()
    for (const m of rolloutMetrics) {
      let sampleMap = map.get(m.sample_idx)
      if (!sampleMap) {
        sampleMap = new Map()
        map.set(m.sample_idx, sampleMap)
      }
      sampleMap.set(m.metric_name, { value: m.value, env: m.env })
    }
    return map
  }, [rolloutMetrics])
  if (computedMetricsBySampleIdx !== null && computedMetricsBySampleIdx !== cachedMetricsBySampleIdx) {
    setCachedMetricsBySampleIdx(computedMetricsBySampleIdx)
  }
  const metricsBySampleIdx = computedMetricsBySampleIdx ?? cachedMetricsBySampleIdx

  // Derive available metric names from data if not explicitly provided
  // Keep previous value during loading to prevent metric columns from disappearing
  const [cachedDerivedMetricNames, setCachedDerivedMetricNames] = useState<string[]>([])
  const computedDerivedMetricNames = useMemo(() => {
    if (availableMetricNames?.length) return availableMetricNames
    if (!rolloutMetrics) return null
    const names = new Set<string>()
    for (const m of rolloutMetrics) names.add(m.metric_name)
    return [...names].sort()
  }, [availableMetricNames, rolloutMetrics])
  if (computedDerivedMetricNames !== null && computedDerivedMetricNames !== cachedDerivedMetricNames) {
    setCachedDerivedMetricNames(computedDerivedMetricNames)
  }
  const derivedMetricNames = computedDerivedMetricNames ?? cachedDerivedMetricNames

  const toggleDisplayMetric = (
    key: string,
    checked: boolean | "indeterminate",
  ) => {
    setDisplayMetrics((prev) => {
      if (checked === true) {
        return prev.includes(key) ? prev : [...prev, key]
      }
      return prev.filter((k) => k !== key)
    })
  }

  // Prune stale keys from persisted displayMetrics that are no longer valid,
  // and map gen_length_avg/gen_length_sum ↔ gen_length based on hasMultiTurn.
  useEffect(() => {
    if (!derivedMetricNames.length && !data) return // Don't prune before data loads
    const hiddenBuiltins = new Set<string>()
    if (hideRewardAdvantage) { hiddenBuiltins.add("reward"); hiddenBuiltins.add("advantage") }
    if (hasMultiTurn) {
      hiddenBuiltins.add("gen_length")
    } else {
      hiddenBuiltins.add("gen_length_sum")
      hiddenBuiltins.add("gen_length_avg")
    }
    const validBuiltins = builtinMetricOrder.filter((k) => !hiddenBuiltins.has(k))
    const validKeys = new Set([...validBuiltins, ...derivedMetricNames])
    setDisplayMetrics((prev) => {
      let next = prev
      // Map between gen_length variants when switching single/multi-turn
      if (!hasMultiTurn && !prev.includes("gen_length") && (prev.includes("gen_length_avg") || prev.includes("gen_length_sum"))) {
        next = [...next.filter((k) => k !== "gen_length_avg" && k !== "gen_length_sum"), "gen_length"]
      } else if (hasMultiTurn && !prev.includes("gen_length_avg") && prev.includes("gen_length")) {
        next = [...next.filter((k) => k !== "gen_length"), "gen_length_avg"]
      }
      const filtered = next.filter((k) => validKeys.has(k))
      return filtered.length === prev.length && filtered.every((k, i) => k === prev[i]) ? prev : filtered
    })
  }, [derivedMetricNames, data, setDisplayMetrics, hideRewardAdvantage, hasMultiTurn])

  // Canonical order for display metrics: built-ins first, then rollout metrics in alpha order
  const orderedDisplayMetrics = useMemo(() => {
    const hidden = new Set<string>()
    if (hideRewardAdvantage) { hidden.add("reward"); hidden.add("advantage") }
    if (hasMultiTurn) { hidden.add("gen_length") } else { hidden.add("gen_length_sum"); hidden.add("gen_length_avg") }
    const ordered: string[] = []
    const seen = new Set<string>()
    // Built-ins first, in canonical order
    for (const key of builtinMetricOrder) {
      if (hidden.has(key)) continue
      if (displayMetrics.includes(key) && !seen.has(key)) {
        ordered.push(key)
        seen.add(key)
      }
    }
    // Then rollout metrics in the order they appear in derivedMetricNames
    for (const name of derivedMetricNames) {
      if (displayMetrics.includes(name) && !seen.has(name)) {
        ordered.push(name)
        seen.add(name)
      }
    }
    // Skip any stale keys that aren't in builtins or derived metrics
    return ordered
  }, [displayMetrics, derivedMetricNames, hideRewardAdvantage, hasMultiTurn])

  // Keep track of last known groups to prevent flickering during loading
  const [lastKnownGroups, setLastKnownGroups] = useState<Group[]>([])

  // Save scroll position for groups view when switching to samples
  const groupsScrollPositionRef = useRef<number>(0)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const lastStepRef = useRef<number | undefined>(step)

  // Continuously track viewport scroll position so we can restore after data-driven re-renders
  const liveScrollTopRef = useRef<number>(0)
  useEffect(() => {
    const viewport = scrollAreaRef.current?.querySelector(
      '[data-slot="scroll-area-viewport"]',
    )
    if (!viewport) return
    const handler = () => {
      liveScrollTopRef.current = (viewport as HTMLElement).scrollTop
    }
    viewport.addEventListener("scroll", handler, { passive: true })
    return () => viewport.removeEventListener("scroll", handler)
  }, [isOpen])

  // Reset scroll position when step changes
  useEffect(() => {
    if (step !== lastStepRef.current) {
      lastStepRef.current = step
      groupsScrollPositionRef.current = 0
      liveScrollTopRef.current = 0
      // Also scroll to top immediately
      const viewport = scrollAreaRef.current?.querySelector(
        '[data-slot="scroll-area-viewport"]',
      )
      if (viewport) {
        viewport.scrollTop = 0
      }
    }
  }, [step])

  // Group prompts by group_id
  const promptsByGroup = useMemo(() => {
    if (!prompts) return new Map<number, Prompt>()
    const map = new Map<number, Prompt>()
    for (const p of prompts) {
      map.set(p.group_id, p)
    }
    return map
  }, [prompts])

  // Index samples data by sample_idx
  const samplesDataBySample = useMemo(() => {
    if (!samplesData) return new Map<number, SampleData>()
    const map = new Map<number, SampleData>()
    for (const s of samplesData) {
      map.set(s.sample_idx, s)
    }
    return map
  }, [samplesData])

  // Build groups with their samples (returns null when data not yet loaded)
  const computedGroups = useMemo(() => {
    if (!data) return null

    // Group rollouts by sample_idx and get group_id
    const turnsBySample = new Map<number, Rollout[]>()
    const groupIdBySample = new Map<number, number>()

    for (const gen of data) {
      const existing = turnsBySample.get(gen.sample_idx) || []
      existing.push(gen)
      turnsBySample.set(gen.sample_idx, existing)
      groupIdBySample.set(gen.sample_idx, gen.group_id)
    }

    // Build grouped samples
    const samplesByGroup = new Map<number, GroupedSample[]>()

    for (const [sample_idx, turns] of turnsBySample) {
      const sortedTurns = [...turns].sort((a, b) => a.turn_order - b.turn_order)
      const group_id = groupIdBySample.get(sample_idx) ?? -1
      const prompt = promptsByGroup.get(group_id) ?? null
      const sampleData = samplesDataBySample.get(sample_idx)

      // Get the first model response as preview
      const firstModelTurn = sortedTurns.find((t) => t.turn_type === "model")
      const responsePreview = firstModelTurn?.content ?? ""
      let assistantTokens: number | null = null
      let assistantTokensTotal: number | null = null
      let assistantTokenSum = 0
      let assistantTurnCount = 0
      for (const turn of sortedTurns) {
        if (turn.turn_type !== "model") continue
        if (typeof turn.tokens !== "number") continue
        assistantTokenSum += turn.tokens
        assistantTurnCount += 1
      }
      if (assistantTurnCount > 0) {
        assistantTokens = assistantTokenSum / assistantTurnCount
        assistantTokensTotal = assistantTokenSum
      }

      const groupedSample: GroupedSample = {
        sample_idx,
        group_id,
        prompt,
        responsePreview,
        turnsCount: sortedTurns.length,
        turnsData: sortedTurns,
        reward: sampleData?.reward ?? null,
        advantage: sampleData?.advantage ?? null,
        assistantTokens,
        assistantTokensTotal,
      }

      const existingSamples = samplesByGroup.get(group_id) || []
      existingSamples.push(groupedSample)
      samplesByGroup.set(group_id, existingSamples)
    }

    // Build groups array
    const groupsArray: Group[] = []
    for (const [group_id, samples] of samplesByGroup) {
      const prompt = promptsByGroup.get(group_id) ?? null
      // Sort samples by sample_idx
      samples.sort((a, b) => a.sample_idx - b.sample_idx)
      groupsArray.push({
        group_id,
        prompt,
        samples,
        sampleCount: samples.length,
      })
    }

    // Sort groups by group_id
    const sortedGroups = groupsArray.sort((a, b) => a.group_id - b.group_id)

    return sortedGroups
  }, [data, promptsByGroup, samplesDataBySample])

  // Cache latest computed groups so stale data persists during loading
  if (computedGroups !== null && computedGroups !== lastKnownGroups) {
    setLastKnownGroups(computedGroups)
  }

  // Use fresh data when available, fall back to cached groups during loading
  const groups = computedGroups ?? lastKnownGroups

  // Restore scroll position after data-driven re-renders (prevents refetch polling from resetting scroll)
  useLayoutEffect(() => {
    const viewport = scrollAreaRef.current?.querySelector(
      '[data-slot="scroll-area-viewport"]',
    ) as HTMLElement | null
    if (!viewport) return
    if (
      liveScrollTopRef.current > 0 &&
      viewport.scrollTop !== liveScrollTopRef.current
    ) {
      viewport.scrollTop = liveScrollTopRef.current
    }
  }, [groups])

  // Get currently selected group data
  const selectedGroup = useMemo(() => {
    if (selectedGroupId === null) return null
    return groups.find((g) => g.group_id === selectedGroupId) ?? null
  }, [groups, selectedGroupId])

  const hasSelectedGroupInStep = selectedGroup !== null
  const maxAssistantTokensInData = useMemo(() => {
    let max = 0
    for (const group of groups) {
      for (const sample of group.samples) {
        if (sample.assistantTokens !== null && sample.assistantTokens > max) {
          max = sample.assistantTokens
        }
      }
    }
    return max > 0 ? max : null
  }, [groups])
  const tokenScaleMax = maxTokensFromConfig ?? maxAssistantTokensInData

  const maxAssistantTokensTotalInData = useMemo(() => {
    let max = 0
    for (const group of groups) {
      for (const sample of group.samples) {
        if (
          sample.assistantTokensTotal !== null &&
          sample.assistantTokensTotal > max
        ) {
          max = sample.assistantTokensTotal
        }
      }
    }
    return max > 0 ? max : null
  }, [groups])
  // For single-turn envs, sum == avg, so the scale should also incorporate
  // maxTokensFromConfig to match the avg scale and produce identical colors.
  // For multi-turn, Math.max ensures the scale still accommodates sums > config max.
  const tokenScaleSumMax =
    maxTokensFromConfig != null
      ? Math.max(maxTokensFromConfig, maxAssistantTokensTotalInData ?? 0) ||
        null
      : maxAssistantTokensTotalInData

  useEffect(() => {
    if (!hasSelectedGroupInStep && viewMode === "samples" && !isLoading) {
      setViewMode("groups")
    }
  }, [hasSelectedGroupInStep, viewMode, setViewMode, isLoading])

  const handleGroupClick = (groupId: number) => {
    const viewport = scrollAreaRef.current?.querySelector(
      '[data-slot="scroll-area-viewport"]',
    )
    if (viewport) {
      groupsScrollPositionRef.current = viewport.scrollTop
    }
    if (onGroupClickProp) {
      const group = groups.find((g) => g.group_id === groupId)
      if (group) onGroupClickProp(groupId, group)
      return
    }
    setSelectedGroupId(groupId)
    setViewMode("samples")
  }

  const handleSampleClick = (sampleIdx: number) => {
    setSelectedSampleIdx(sampleIdx)
  }

  const handleViewModeChange = (value: string) => {
    if (value === "samples" && !hasSelectedGroupInStep) {
      setViewMode("groups")
      return
    }
    if (value === "groups" || value === "samples") {
      // Save scroll position when leaving groups view
      if (viewMode === "groups" && value === "samples") {
        const viewport = scrollAreaRef.current?.querySelector(
          '[data-slot="scroll-area-viewport"]',
        )
        if (viewport) {
          groupsScrollPositionRef.current = viewport.scrollTop
        }
      }
      setViewMode(value)
    }
  }

  // Restore groups scroll position when switching back to groups view
  useEffect(() => {
    if (viewMode === "groups" && groupsScrollPositionRef.current > 0) {
      // Use requestAnimationFrame to ensure DOM has updated
      requestAnimationFrame(() => {
        const viewport = scrollAreaRef.current?.querySelector(
          '[data-slot="scroll-area-viewport"]',
        )
        if (viewport) {
          viewport.scrollTop = groupsScrollPositionRef.current
          liveScrollTopRef.current = groupsScrollPositionRef.current
        }
      })
    }
  }, [viewMode])

  // Scroll to selected sample when it changes (e.g. from "Go to" navigation)
  useEffect(() => {
    if (viewMode !== "samples" || selectedSampleIdx === null) return
    // Use requestAnimationFrame to ensure DOM has rendered
    requestAnimationFrame(() => {
      const viewport = scrollAreaRef.current?.querySelector(
        '[data-slot="scroll-area-viewport"]',
      ) as HTMLElement | null
      if (!viewport) return
      const sampleEl = viewport.querySelector(
        `[data-sample-idx="${selectedSampleIdx}"]`,
      )
      if (sampleEl) {
        sampleEl.scrollIntoView({ block: "nearest", behavior: "smooth" })
        // Update live ref after scroll completes (approximate)
        setTimeout(() => {
          liveScrollTopRef.current = viewport.scrollTop
        }, 350)
      }
    })
  }, [viewMode, selectedSampleIdx])

  // Don't render if no run selected or sidebar is closed
  if (!selectedRunPath || !isOpen) {
    return null
  }

  return (
    <div className="w-[260px] border-r border-border bg-card/50 backdrop-blur flex flex-col h-full min-h-0">
      {/* Header with step picker and view toggle - matching /metrics header height */}
      <div className="border-b border-sidebar-border shrink-0">
        {/* Step picker row */}
        <div className="py-2.5 px-1 flex items-center justify-center h-12">
          {hasSteps ? (
            <PaginationControls
              currentPage={currentStepIndex}
              totalPages={totalSteps}
              onPageChange={onStepChange}
              pageValues={stepValues}
            />
          ) : (
            <span className="text-xs text-muted-foreground">
              {isLoading ? "Loading..." : "No steps available"}
            </span>
          )}
        </div>

        {/* View toggle row */}
        <div className="py-1.5 px-3 flex items-center justify-between border-t border-sidebar-border">
          {renderLeftControls ? (
            renderLeftControls({
              viewMode,
              setViewMode,
              hasSelectedGroupInStep,
              groups,
            })
          ) : (
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={viewMode}
              onValueChange={handleViewModeChange}
            >
              <ToggleGroupItem value="groups" className="text-xs px-2 py-1 h-7">
                Groups
              </ToggleGroupItem>
              <ToggleGroupItem
                value="samples"
                className="text-xs px-2 py-1 h-7"
                disabled={!hasSelectedGroupInStep}
              >
                Samples
              </ToggleGroupItem>
            </ToggleGroup>
          )}
          <div className="flex items-center gap-0.5">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-7 w-7">
                  <Settings className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-[180px]">
                <DropdownMenuLabel>Settings</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuCheckboxItem
                  checked={tooltipsEnabled}
                  onCheckedChange={(checked) =>
                    setTooltipsEnabled(checked === true)
                  }
                  onSelect={(e) => e.preventDefault()}
                >
                  Show Tooltips
                </DropdownMenuCheckboxItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-7 w-7">
                  <Menu className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-[180px]">
                <DropdownMenuLabel>Display Metrics</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {hasMultiTurn ? (
                  <>
                    <DropdownMenuCheckboxItem
                      checked={displayMetrics.includes("gen_length_sum")}
                      onCheckedChange={(checked) =>
                        toggleDisplayMetric("gen_length_sum", checked)
                      }
                      onSelect={(e) => e.preventDefault()}
                    >
                      Gen. Length Sum
                    </DropdownMenuCheckboxItem>
                    <DropdownMenuCheckboxItem
                      checked={displayMetrics.includes("gen_length_avg")}
                      onCheckedChange={(checked) =>
                        toggleDisplayMetric("gen_length_avg", checked)
                      }
                      onSelect={(e) => e.preventDefault()}
                    >
                      Gen. Length Avg
                    </DropdownMenuCheckboxItem>
                  </>
                ) : (
                  <DropdownMenuCheckboxItem
                    checked={displayMetrics.includes("gen_length")}
                    onCheckedChange={(checked) =>
                      toggleDisplayMetric("gen_length", checked)
                    }
                    onSelect={(e) => e.preventDefault()}
                  >
                    Gen. Length
                  </DropdownMenuCheckboxItem>
                )}
                {!hideRewardAdvantage && (
                  <DropdownMenuCheckboxItem
                    checked={displayMetrics.includes("reward")}
                    onCheckedChange={(checked) =>
                      toggleDisplayMetric("reward", checked)
                    }
                    onSelect={(e) => e.preventDefault()}
                  >
                    Reward
                  </DropdownMenuCheckboxItem>
                )}
                {!hideRewardAdvantage && (
                  <DropdownMenuCheckboxItem
                    checked={displayMetrics.includes("advantage")}
                    onCheckedChange={(checked) =>
                      toggleDisplayMetric("advantage", checked)
                    }
                    onSelect={(e) => e.preventDefault()}
                  >
                    Advantage
                  </DropdownMenuCheckboxItem>
                )}
                {derivedMetricNames.length > 0 && <DropdownMenuSeparator />}
                {derivedMetricNames.map((name) => (
                  <DropdownMenuCheckboxItem
                    key={name}
                    checked={displayMetrics.includes(name)}
                    onCheckedChange={(checked) =>
                      toggleDisplayMetric(name, checked)
                    }
                    onSelect={(e) => e.preventDefault()}
                  >
                    {formatMetricName(name)}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => setIsOpen(false)}
            >
              <PanelLeftClose className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Content - opacity when loading but keep content visible */}
      <ScrollArea
        ref={scrollAreaRef}
        className={cn(
          "flex-1 min-h-0 transition-opacity",
          isLoading && "opacity-50",
        )}
      >
        {groups.length === 0 ? (
          <div className="p-4 text-center text-muted-foreground text-sm">
            {isLoading ? "Loading..." : "No samples available"}
          </div>
        ) : viewMode === "groups" || !selectedGroup ? (
          // Show groups list (also when in samples mode but selected group not in current step)
          <div className="p-2 space-y-1">
            {groups.map((group) => (
              <GroupItem
                key={group.group_id}
                group={group}
                onClick={() => handleGroupClick(group.group_id)}
                isSelected={selectedGroupId === group.group_id}
                tokenScaleMax={tokenScaleMax}
                tokenScaleSumMax={tokenScaleSumMax}
                envRewardRanges={envRewardRanges}
                envMetricRanges={envMetricRanges}
                advantageRange={advantageRange}
                badge={getGroupBadge ? getGroupBadge(group) : null}
                displayMetrics={orderedDisplayMetrics}
                metricsBySampleIdx={metricsBySampleIdx}
                tooltipsEnabled={tooltipsEnabled}
                darkMode={darkMode}
              />
            ))}
          </div>
        ) : (
          // Show samples in selected group
          <div className="flex flex-col min-h-0">
            {/* Group prompt info at top - with tooltip */}
            <Tooltip
              delayDuration={300}
              open={tooltipsEnabled ? undefined : false}
            >
              <TooltipTrigger asChild>
                <div className="p-3 border-b border-border bg-muted/30 shrink-0 cursor-default">
                  {selectedGroup.prompt?.system_prompt && (
                    <div className="mb-2">
                      <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">
                        System Prompt
                      </div>
                      <p className="text-xs text-foreground line-clamp-2 break-words overflow-hidden [overflow-wrap:anywhere]">
                        {truncateText(selectedGroup.prompt.system_prompt, 100)}
                      </p>
                    </div>
                  )}
                  <div>
                    <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">
                      Prompt
                    </div>
                    <p className="text-xs text-foreground line-clamp-3 break-words overflow-hidden [overflow-wrap:anywhere]">
                      {truncateText(selectedGroup.prompt?.prompt, 150)}
                    </p>
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent
                side="right"
                className="max-w-xl max-h-[70vh] overflow-y-auto p-3 space-y-2 bg-popover text-popover-foreground border border-border shadow-md"
              >
                {selectedGroup.prompt?.system_prompt && (
                  <div>
                    <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">
                      System Prompt
                    </div>
                    <p className="text-xs whitespace-pre-wrap break-words">
                      {selectedGroup.prompt.system_prompt}
                    </p>
                  </div>
                )}
                {selectedGroup.prompt?.prompt && (
                  <div>
                    <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">
                      Prompt
                    </div>
                    <p className="text-xs whitespace-pre-wrap break-words">
                      {selectedGroup.prompt.prompt}
                    </p>
                  </div>
                )}
              </TooltipContent>
            </Tooltip>

            {/* Samples list */}
            <div className="p-2 space-y-1">
              {selectedGroup.samples.map((sample) => (
                <div
                  key={sample.sample_idx}
                  data-sample-idx={sample.sample_idx}
                >
                  <SampleItem
                    sample={sample}
                    isSelected={selectedSampleIdx === sample.sample_idx}
                    onClick={() => handleSampleClick(sample.sample_idx)}
                    envRewardRanges={envRewardRanges}
                    envMetricRanges={envMetricRanges}
                    advantageRange={advantageRange}
                    displayMetrics={orderedDisplayMetrics}
                    metricsBySampleIdx={metricsBySampleIdx}
                    tokenScaleMax={tokenScaleMax}
                    tokenScaleSumMax={tokenScaleSumMax}
                    tooltipsEnabled={tooltipsEnabled}
                    darkMode={darkMode}
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </ScrollArea>
    </div>
  )
}

function GroupItem({
  group,
  onClick,
  isSelected,
  tokenScaleMax,
  tokenScaleSumMax,
  envRewardRanges,
  envMetricRanges,
  advantageRange,
  badge,
  displayMetrics,
  metricsBySampleIdx,
  tooltipsEnabled,
  darkMode,
}: {
  group: Group
  onClick: () => void
  isSelected: boolean
  tokenScaleMax: number | null
  tokenScaleSumMax: number | null
  envRewardRanges?: EnvRewardRanges
  envMetricRanges?: EnvMetricRanges
  advantageRange: { min: number; max: number } | null
  badge: ReactNode
  displayMetrics: string[]
  metricsBySampleIdx: Map<
    number,
    Map<string, { value: number; env: string | null }>
  >
  tooltipsEnabled: boolean
  darkMode: boolean
}) {
  return (
    <Tooltip delayDuration={300} open={tooltipsEnabled ? undefined : false}>
      <TooltipTrigger asChild>
        <button
          onClick={onClick}
          className={cn(
            "w-full text-left p-3 rounded-lg border transition-colors overflow-hidden",
            isSelected
              ? "border-primary/50 bg-primary/5"
              : "border-sidebar-border hover:border-border hover:bg-muted/50",
          )}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              {group.prompt?.system_prompt && (
                <p className="text-xs text-muted-foreground mb-1 line-clamp-1 break-words overflow-hidden [overflow-wrap:anywhere] break-all">
                  {truncateText(group.prompt.system_prompt, 50)}
                </p>
              )}
              <p className="text-xs text-foreground line-clamp-2 break-words overflow-hidden [overflow-wrap:anywhere] break-all">
                {truncateText(group.prompt?.prompt, 80)}
              </p>
              {badge && (
                <div className="mt-1 flex flex-wrap gap-1">{badge}</div>
              )}
            </div>
            {displayMetrics.length > 0 && (
              <div className="flex items-start gap-1 shrink-0 pt-0">
                {displayMetrics.map((metricKey) => (
                  <div
                    key={metricKey}
                    className="flex flex-col items-start gap-[0.5px]"
                  >
                    {group.samples.map((sample) => {
                      const display = getMetricDisplay(
                        sample,
                        metricKey,
                        metricsBySampleIdx,
                        tokenScaleMax,
                        tokenScaleSumMax,
                        envRewardRanges,
                        advantageRange,
                        envMetricRanges,
                        undefined,
                        darkMode,
                      )
                      return (
                        <div
                          key={sample.sample_idx}
                          title={display.label}
                          className={cn(
                            "h-1.5 w-1.5 rounded-none",
                            display.color ? "" : "bg-muted",
                          )}
                          style={
                            display.color
                              ? { backgroundColor: display.color }
                              : undefined
                          }
                        />
                      )
                    })}
                  </div>
                ))}
              </div>
            )}
          </div>
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="right"
        className="max-w-xl max-h-[70vh] overflow-y-auto p-3 space-y-2 bg-popover text-popover-foreground border border-border shadow-md"
      >
        {group.prompt?.system_prompt && (
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">
              System Prompt
            </div>
            <p className="text-xs whitespace-pre-wrap break-words">
              {group.prompt.system_prompt}
            </p>
          </div>
        )}
        {group.prompt?.prompt && (
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">
              Prompt
            </div>
            <p className="text-xs whitespace-pre-wrap break-words">
              {group.prompt.prompt}
            </p>
          </div>
        )}
      </TooltipContent>
    </Tooltip>
  )
}

function formatTurnType(turnType: string): string {
  if (turnType === "model") return "Assistant"
  if (turnType === "env" || turnType === "env_response") return "User"
  return turnType
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ")
}

function formatMetricName(name: string): string {
  return name
    .replace(/_reward$/, "")
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ")
}

function formatMetricValue(value: number): string {
  if (Number.isInteger(value)) return value.toString()
  return value.toFixed(2)
}

function formatDisplayValue(value: number, metricKey: string): string {
  switch (metricKey) {
    case "gen_length":
    case "gen_length_sum":
    case "gen_length_avg":
      return Math.round(value).toString()
    default:
      return formatMetricValue(value)
  }
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
  // Fallback: search across all envs (handles env name mismatches)
  for (const metricMap of envMetricRanges.values()) {
    const found = metricMap.get(metricKey)
    if (found) return found
  }
  return undefined
}

function getMetricDisplay(
  sample: GroupedSample,
  metricKey: string,
  metricsBySampleIdx: Map<
    number,
    Map<string, { value: number; env: string | null }>
  >,
  tokenScaleMax: number | null,
  tokenScaleSumMax: number | null,
  envRewardRanges?: EnvRewardRanges,
  advantageRange?: { min: number; max: number } | null,
  envMetricRanges?: EnvMetricRanges,
  tokenMaxLightness?: number,
  darkMode?: boolean,
): { value: number | null; color: string | null; label: string } {
  switch (metricKey) {
    case "gen_length":
      return {
        value: sample.assistantTokensTotal,
        color: tokensToColor(
          sample.assistantTokensTotal,
          tokenScaleSumMax,
          tokenMaxLightness,
          darkMode,
        ),
        label:
          sample.assistantTokensTotal !== null
            ? `Gen. Length: ${sample.assistantTokensTotal}`
            : "Gen. Length: n/a",
      }
    case "gen_length_sum":
      return {
        value: sample.assistantTokensTotal,
        color: tokensToColor(
          sample.assistantTokensTotal,
          tokenScaleSumMax,
          tokenMaxLightness,
          darkMode,
        ),
        label:
          sample.assistantTokensTotal !== null
            ? `Gen. Length Sum: ${sample.assistantTokensTotal}`
            : "Gen. Length Sum: n/a",
      }
    case "gen_length_avg":
      return {
        value:
          sample.assistantTokens !== null
            ? Math.round(sample.assistantTokens)
            : null,
        color: tokensToColor(
          sample.assistantTokens,
          tokenScaleMax,
          tokenMaxLightness,
          darkMode,
        ),
        label:
          sample.assistantTokens !== null
            ? `Gen. Length Avg: ${Math.round(sample.assistantTokens)}`
            : "Gen. Length Avg: n/a",
      }
    case "reward": {
      const env = sample.prompt?.env ?? null
      const rewardRange = env ? envRewardRanges?.get(env) : undefined
      return {
        value: sample.reward,
        color:
          sample.reward !== null
            ? (rewardRangeColor(
                sample.reward,
                rewardRange?.min ?? null,
                rewardRange?.max ?? null,
              ) ?? rewardToColor(sample.reward))
            : null,
        label:
          sample.reward !== null
            ? `Reward: ${sample.reward.toFixed(2)}`
            : "Reward: n/a",
      }
    }
    case "advantage": {
      return {
        value: sample.advantage,
        color:
          sample.advantage !== null
            ? (rewardRangeColor(
                sample.advantage,
                advantageRange?.min ?? null,
                advantageRange?.max ?? null,
              ) ?? null)
            : null,
        label:
          sample.advantage !== null
            ? `Advantage: ${sample.advantage.toFixed(2)}`
            : "Advantage: n/a",
      }
    }
    default: {
      const sampleMetrics = metricsBySampleIdx.get(sample.sample_idx)
      const metric = sampleMetrics?.get(metricKey)
      if (!metric)
        return {
          value: null,
          color: null,
          label: `${formatMetricName(metricKey)}: n/a`,
        }
      const metricEnv = metric.env ?? sample.prompt?.env ?? null
      const range = lookupMetricRange(envMetricRanges, metricEnv, metricKey)
      const color = rewardRangeColor(
        metric.value,
        range?.min ?? null,
        range?.max ?? null,
        range?.invert,
      )
      return {
        value: metric.value,
        color: color ?? null,
        label: `${formatMetricName(metricKey)}: ${formatMetricValue(metric.value)}`,
      }
    }
  }
}

function SampleItem({
  sample,
  isSelected,
  onClick,
  envRewardRanges,
  envMetricRanges,
  advantageRange,
  displayMetrics,
  metricsBySampleIdx,
  tokenScaleMax,
  tokenScaleSumMax,
  tooltipsEnabled,
  darkMode,
}: {
  sample: GroupedSample
  isSelected: boolean
  onClick: () => void
  envRewardRanges?: EnvRewardRanges
  envMetricRanges?: EnvMetricRanges
  advantageRange: { min: number; max: number } | null
  displayMetrics: string[]
  metricsBySampleIdx: Map<
    number,
    Map<string, { value: number; env: string | null }>
  >
  tokenScaleMax: number | null
  tokenScaleSumMax: number | null
  tooltipsEnabled: boolean
  darkMode: boolean
}) {
  return (
    <Tooltip delayDuration={300} open={tooltipsEnabled ? undefined : false}>
      <TooltipTrigger asChild>
        <button
          onClick={onClick}
          className={cn(
            "w-full text-left p-3 rounded-lg border transition-colors overflow-hidden",
            isSelected
              ? "border-primary/50 bg-primary/5"
              : "border-sidebar-border hover:border-border hover:bg-muted/50",
          )}
        >
          <div className="flex items-stretch justify-between gap-2 min-w-0">
            <p
              className="text-xs text-foreground break-words flex-1 min-w-0 overflow-hidden"
              style={{
                display: "-webkit-box",
                WebkitBoxOrient: "vertical",
                WebkitLineClamp:
                  displayMetrics.length > 0
                    ? Math.max(
                        2,
                        displayMetrics.filter((mk) => {
                          const d = getMetricDisplay(
                            sample,
                            mk,
                            metricsBySampleIdx,
                            tokenScaleMax,
                            tokenScaleSumMax,
                            envRewardRanges,
                            advantageRange,
                            envMetricRanges,
                            70,
                            darkMode,
                          )
                          return d.value !== null
                        }).length,
                      )
                    : 2,
                overflowWrap: "anywhere",
              }}
            >
              {truncateText(sample.responsePreview, 300) || "No response"}
            </p>
            {displayMetrics.length > 0 && (
              <div className="flex items-start gap-1 shrink-0">
                <div className="flex flex-col items-center gap-0.5">
                  {displayMetrics.map((metricKey) => {
                    const display = getMetricDisplay(
                      sample,
                      metricKey,
                      metricsBySampleIdx,
                      tokenScaleMax,
                      tokenScaleSumMax,
                      envRewardRanges,
                      advantageRange,
                      envMetricRanges,
                      70,
                      darkMode,
                    )
                    if (display.value === null) return null
                    return (
                      <span
                        key={metricKey}
                        className="text-[10px] font-semibold tabular-nums leading-none"
                        style={
                          display.color ? { color: display.color } : undefined
                        }
                        title={display.label}
                      >
                        {formatDisplayValue(display.value, metricKey)}
                      </span>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="right"
        className="max-w-xl max-h-[70vh] overflow-y-auto p-3 bg-popover text-popover-foreground border border-border shadow-md"
      >
        <div className="space-y-3">
          {sample.turnsData.map((turn) => (
            <div key={`${turn.sample_idx}-${turn.turn_order}`}>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5 flex items-center gap-2">
                <span>{formatTurnType(turn.turn_type)}</span>
                {turn.tokens != null && (
                  <span className="font-normal normal-case">
                    {turn.tokens.toLocaleString()} tokens
                  </span>
                )}
              </div>
              <p className="text-xs whitespace-pre-wrap break-words font-sans">
                {turn.content || "(empty)"}
              </p>
            </div>
          ))}
          {sample.turnsData.length === 0 && (
            <p className="text-xs text-muted-foreground">No turns available</p>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  )
}
