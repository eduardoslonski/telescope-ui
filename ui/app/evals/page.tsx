
import { useEffect, useMemo, useRef, useState, useCallback } from "react"
import { useAtom, useAtomValue } from "jotai"
import { useSearchParams } from "react-router-dom"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
} from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { NoRunSelectedState } from "@/components/no-run-selected-state"
import { RolloutsView, RawTextDialog, extractEnvRewardRanges, extractEnvMetricRanges, parseDataMetricRanges, mergeEnvMetricRanges, type EnvRewardRanges, type EnvMetricRanges } from "@/components/rollouts-view"
import {
  RolloutsMetricsPanel,
  RolloutsMetricsSidebarToggle,
} from "@/components/rollouts-metrics-panel"
import {
  RolloutsSamplePickerSidebar,
  RolloutsSamplePickerSidebarToggle,
} from "@/components/rollouts-sample-picker-sidebar"
import type { Group } from "@/components/rollouts-sample-picker-sidebar"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ChevronLeft } from "lucide-react"
import {
  selectedRunPathAtom,
  evalsSelectedStepAtom,
  evalsSelectedEvalNameAtom,
  evalsSelectedGroupIdAtom,
  evalsSelectedSampleIdxAtom,
  evalsLastValidSampleAtom,
  evalsSamplePickerSidebarOpenAtom,
  evalsSamplePickerViewModeAtom,
  evalsMetricsSidebarOpenAtom,
  evalsSelectedMetricsAtom,
  rolloutsRenderOptionsAtom,
  rolloutsFormatThinkAtom,
  type RolloutsRenderOption,
} from "@/lib/atoms"
import { useEvals, useRunSummary } from "@/hooks/use-run-data"
import type {
  Rollout,
  Prompt,
  RolloutMetric,
  GoldenAnswer,
  InfoTurn,
  SampleData,
} from "@/lib/types"

const COMPLETION_MULTIPLIER = 10000

function syntheticSampleIdx(sampleIdx: number, completionIdx: number): number {
  return sampleIdx * COMPLETION_MULTIPLIER + completionIdx
}

export default function EvalsPage() {
  const selectedRunPath = useAtomValue(selectedRunPathAtom)
  const [searchParams] = useSearchParams()

  const [selectedStep, setSelectedStep] = useAtom(evalsSelectedStepAtom)
  const [selectedEvalName, setSelectedEvalName] = useAtom(evalsSelectedEvalNameAtom)
  const [selectedGroupId, setSelectedGroupId] = useAtom(evalsSelectedGroupIdAtom)
  const [selectedSampleIdx, setSelectedSampleIdx] = useAtom(evalsSelectedSampleIdxAtom)
  const lastValidSample = useAtomValue(evalsLastValidSampleAtom)
  const [isSamplesSidebarOpen] = useAtom(evalsSamplePickerSidebarOpenAtom)
  const [, setViewMode] = useAtom(evalsSamplePickerViewModeAtom)
  const isMetricsSidebarOpen = useAtomValue(evalsMetricsSidebarOpenAtom)
  const [renderOptions, setRenderOptions] = useAtom(rolloutsRenderOptionsAtom)
  const [formatThinkBlocks, setFormatThinkBlocks] = useAtom(rolloutsFormatThinkAtom)
  const selectedOptionsCount = renderOptions.length + (formatThinkBlocks ? 1 : 0)

  const toggleRenderOption = (option: RolloutsRenderOption) => {
    setRenderOptions((prev) =>
      prev.includes(option)
        ? prev.filter((o) => o !== option)
        : [...prev, option]
    )
  }

  const [collapseAllSignal, setCollapseAllSignal] = useState(0)
  const [expandAllSignal, setExpandAllSignal] = useState(0)
  const [defaultSectionsOpen, setDefaultSectionsOpen] = useState(true)

  const shouldPoll = true

  const requestedStep = useMemo(() => {
    const raw = searchParams.get("step")
    if (!raw) return null
    const parsed = Number(raw)
    return Number.isFinite(parsed) ? parsed : null
  }, [searchParams])

  const lastRequestedStepRef = useRef<number | null>(null)

  useEffect(() => {
    if (!selectedRunPath) {
      lastRequestedStepRef.current = null
      return
    }
    if (requestedStep == null) {
      lastRequestedStepRef.current = null
      return
    }
    if (requestedStep === lastRequestedStepRef.current) return
    setSelectedStep(requestedStep)
    lastRequestedStepRef.current = requestedStep
  }, [requestedStep, selectedRunPath, setSelectedStep])

  const {
    data: evalsData,
    isLoading,
    error,
  } = useEvals(
    selectedRunPath || "",
    selectedStep,
    selectedEvalName,
    !!selectedRunPath,
    shouldPoll
  )

  const { data: summaryData } = useRunSummary(
    selectedRunPath || "",
    !!selectedRunPath,
    shouldPoll
  )

  const evalEnvDetails = summaryData?.summary?.eval_env_details ?? summaryData?.config?.eval_env_details
  const envRewardRanges = useMemo<EnvRewardRanges>(
    () => extractEnvRewardRanges(evalEnvDetails),
    [evalEnvDetails]
  )
  const envMetricRanges = useMemo<EnvMetricRanges>(() => {
    const configRanges = extractEnvMetricRanges(evalEnvDetails)
    const dataRanges = parseDataMetricRanges(summaryData?.data_metric_ranges)
    return mergeEnvMetricRanges(configRanges, dataRanges)
  }, [evalEnvDetails, summaryData?.data_metric_ranges])

  // Stabilize available_rollout_metric_names across step changes so the
  // metrics sidebar doesn't flash/reset while useEvals refetches for a new step.
  const [cachedRolloutMetricNames, setCachedRolloutMetricNames] = useState<string[]>([])
  const incomingRolloutNames = evalsData?.available_rollout_metric_names
  const stableRolloutMetricNames = incomingRolloutNames && incomingRolloutNames.length > 0
    ? incomingRolloutNames
    : cachedRolloutMetricNames
  if (incomingRolloutNames && incomingRolloutNames.length > 0 && incomingRolloutNames !== cachedRolloutMetricNames) {
    setCachedRolloutMetricNames(incomingRolloutNames)
  }

  // Build eval config for the metrics panel — prefix all metric keys with eval/{eval_name}/
  const evalConfig = useMemo(() => {
    if (!selectedEvalName) return undefined
    return {
      prefix: `eval/${selectedEvalName}/`,
      rewardNames: stableRolloutMetricNames,
      selectedSampleIdx: selectedGroupId,
    }
  }, [selectedEvalName, stableRolloutMetricNames, selectedGroupId])

  // Sync eval name from server response (auto-select first eval when none chosen)
  useEffect(() => {
    if (!evalsData) return
    const names = evalsData.available_eval_names
    if (!names.length) return
    if (selectedEvalName && names.includes(selectedEvalName)) return
    setSelectedEvalName(names[0])
  }, [evalsData?.available_eval_names, selectedEvalName, setSelectedEvalName, evalsData])

  // Compute initial step
  const initialStep = useMemo(() => {
    if (evalsData?.available_steps?.length && selectedStep === null) {
      return evalsData.available_steps[evalsData.available_steps.length - 1]
    }
    return null
  }, [evalsData?.available_steps, selectedStep])

  useEffect(() => {
    if (initialStep !== null && selectedStep === null && requestedStep == null) {
      setSelectedStep(initialStep)
    }
  }, [initialStep, requestedStep, selectedStep, setSelectedStep])

  const handleStepChange = (step: number) => {
    setSelectedStep(step)
  }

  const [lastKnownSteps, setLastKnownSteps] = useState<number[]>([])
  const [lastRunPath, setLastRunPath] = useState<string | null>(null)

  if (selectedRunPath !== lastRunPath) {
    setLastKnownSteps([])
    setLastRunPath(selectedRunPath)
  }

  const availableSteps = evalsData?.available_steps
  const sortedSteps = useMemo(() => {
    const steps = availableSteps ?? []
    const merged = new Set([...lastKnownSteps, ...steps])
    return [...merged].sort((a, b) => a - b)
  }, [availableSteps, lastKnownSteps])

  if (sortedSteps.length > lastKnownSteps.length) {
    setLastKnownSteps(sortedSteps)
  }

  const hasSteps = sortedSteps.length > 0
  const lastStep = hasSteps ? sortedSteps[sortedSteps.length - 1] : 0

  const currentStepIndex = useMemo(() => {
    if (selectedStep == null) return sortedSteps.length - 1
    const idx = sortedSteps.indexOf(selectedStep)
    return idx >= 0 ? idx : sortedSteps.length - 1
  }, [sortedSteps, selectedStep])

  const handleStepIndexChange = (index: number) => {
    if (index >= 0 && index < sortedSteps.length) {
      handleStepChange(sortedSteps[index])
    }
  }

  // Map eval data to rollouts format for shared components
  const mappedPrompts = useMemo<Prompt[] | undefined>(() => {
    if (!evalsData?.prompts) return undefined
    return evalsData.prompts.map((p) => ({
      step: p.step,
      group_id: p.sample_idx,
      env: p.env,
      system_prompt: p.system_prompt,
      tokens_system_prompt: p.tokens_system_prompt,
      prompt: p.prompt,
      tokens_prompt: p.tokens_prompt,
    }))
  }, [evalsData])

  const mappedRollouts = useMemo<Rollout[] | undefined>(() => {
    if (!evalsData?.rollouts) return undefined
    return evalsData.rollouts.map((r) => ({
      step: r.step,
      group_id: r.sample_idx,
      sample_idx: syntheticSampleIdx(r.sample_idx, r.completion_idx),
      turn_order: r.turn_order,
      turn_type: r.turn_type,
      content: r.content,
      tokens: r.tokens,
    }))
  }, [evalsData])

  const mappedSamplesData = useMemo<SampleData[] | undefined>(() => {
    if (!evalsData?.samples_data) return undefined
    return evalsData.samples_data.map((s) => ({
      step: s.step,
      group_id: s.sample_idx,
      sample_idx: syntheticSampleIdx(s.sample_idx, s.completion_idx),
      reward: null,
      advantage: null,
      turns: s.turns,
      total_tokens: null,
      raw_string: null,
    }))
  }, [evalsData])

  const mappedRolloutMetrics = useMemo<RolloutMetric[] | undefined>(() => {
    if (!evalsData?.rollout_metrics) return undefined
    return evalsData.rollout_metrics.map((m) => ({
      step: m.step,
      sample_idx: syntheticSampleIdx(m.sample_idx, m.completion_idx),
      env: m.env,
      metric_name: m.metric_name,
      value: m.value,
    }))
  }, [evalsData])

  const mappedGoldenAnswers = useMemo<GoldenAnswer[] | undefined>(() => {
    if (!evalsData?.golden_answers) return undefined
    return evalsData.golden_answers.map((a) => ({
      step: a.step,
      sample_idx: syntheticSampleIdx(a.sample_idx, a.completion_idx),
      env: a.env,
      key: a.key,
      value: a.value,
    }))
  }, [evalsData])

  const mappedInfoTurns = useMemo<InfoTurn[] | undefined>(() => {
    if (!evalsData?.info_turns) return undefined
    return evalsData.info_turns.map((it) => ({
      step: it.step,
      sample_idx: syntheticSampleIdx(it.sample_idx, it.completion_idx),
      turn_order: it.turn_order,
      env: it.env,
      info_key: it.info_key,
      info_value: it.info_value,
      info_type: it.info_type,
    }))
  }, [evalsData])

  // Compute completions count per eval sample_idx (for single vs pass@k detection)
  const completionsPerSample = useMemo(() => {
    if (!evalsData?.samples_data) return new Map<number, number>()
    const map = new Map<number, number>()
    for (const s of evalsData.samples_data) {
      map.set(s.sample_idx, (map.get(s.sample_idx) ?? 0) + 1)
    }
    return map
  }, [evalsData])

  // Custom group click handler for the sidebar
  const handleEvalGroupClick = useCallback(
    (groupId: number, group: Group) => {
      const numCompletions = completionsPerSample.get(groupId) ?? group.samples.length
      setSelectedGroupId(groupId)
      if (numCompletions <= 1) {
        // Single completion — select directly, stay in groups view
        const onlySample = group.samples[0]
        if (onlySample) {
          setSelectedSampleIdx(onlySample.sample_idx)
        }
      } else {
        // Multiple completions — show completions list
        setViewMode("samples")
      }
    },
    [completionsPerSample, setSelectedGroupId, setSelectedSampleIdx, setViewMode]
  )

  const handleBackToSamples = useCallback(() => {
    setViewMode("groups")
  }, [setViewMode])

  // Render the eval select + back button for the sidebar
  const renderLeftControls = useCallback(
    ({ viewMode }: { viewMode: "groups" | "samples" }) => (
      <div className="flex items-center gap-1 min-w-0 flex-1">
        {viewMode === "samples" && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={handleBackToSamples}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
        )}
        <Select
          value={selectedEvalName ?? ""}
          onValueChange={(val) => {
            setSelectedEvalName(val)
            setSelectedGroupId(null)
            setSelectedSampleIdx(null)
            setViewMode("groups")
          }}
        >
          <SelectTrigger className="h-7 text-xs min-w-0 flex-1">
            <SelectValue placeholder="Select eval..." />
          </SelectTrigger>
          <SelectContent>
            {(evalsData?.available_eval_names ?? []).map((name) => (
              <SelectItem key={name} value={name} className="text-xs">
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    ),
    [
      selectedEvalName,
      evalsData?.available_eval_names,
      setSelectedEvalName,
      setSelectedGroupId,
      setSelectedSampleIdx,
      setViewMode,
      handleBackToSamples,
    ]
  )

  // Header info display
  const headerInfo = useMemo(() => {
    if (!lastValidSample) return null
    const step = lastValidSample.sample.turns[0]?.step ?? selectedStep
    const evalSampleIdx = lastValidSample.sample.group_id
    const synthIdx = lastValidSample.sample.sample_idx
    const completionIdx = synthIdx % COMPLETION_MULTIPLIER
    const numCompletions = completionsPerSample.get(evalSampleIdx) ?? 1
    return { step, evalSampleIdx, completionIdx, numCompletions }
  }, [lastValidSample, selectedStep, completionsPerSample])

  if (!selectedRunPath) {
    return <NoRunSelectedState description="Select a run from the sidebar to view evals." />
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sample Picker Sidebar (Left) */}
      <RolloutsSamplePickerSidebar
        prompts={mappedPrompts}
        data={mappedRollouts}
        samplesData={mappedSamplesData}
        rolloutMetrics={mappedRolloutMetrics}
        availableMetricNames={stableRolloutMetricNames}
        isLoading={isLoading}
        step={selectedStep ?? undefined}
        currentStepIndex={currentStepIndex}
        totalSteps={sortedSteps.length}
        onStepChange={handleStepIndexChange}
        hasSteps={hasSteps}
        stepValues={sortedSteps}
        openAtom={evalsSamplePickerSidebarOpenAtom}
        selectedGroupIdAtom={evalsSelectedGroupIdAtom}
        selectedSampleIdxAtom={evalsSelectedSampleIdxAtom}
        viewModeAtom={evalsSamplePickerViewModeAtom}
        renderLeftControls={renderLeftControls}
        onGroupClick={handleEvalGroupClick}
        hideRewardAdvantage
        envDetailsOverride={evalEnvDetails}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
        {/* Header */}
        <header className="border-b bg-card/50 backdrop-blur-sm shrink-0">
          <div className="py-2.5 px-4">
            <div className="flex items-center justify-between h-7">
              <div className="flex items-center gap-3">
                <RolloutsSamplePickerSidebarToggle
                  openAtom={evalsSamplePickerSidebarOpenAtom}
                />
                {headerInfo && (
                  <div className="text-sm text-muted-foreground">
                    <span>Step {headerInfo.step}</span>
                    {selectedEvalName && (
                      <>
                        {" · "}
                        <span>{selectedEvalName}</span>
                      </>
                    )}
                    {" · "}
                    <span>Sample {headerInfo.evalSampleIdx}</span>
                    {headerInfo.numCompletions > 1 && (
                      <>
                        {" · "}
                        <span>Completion {headerInfo.completionIdx}</span>
                      </>
                    )}
                  </div>
                )}
                {lastValidSample?.sample.raw_string && (
                  <RawTextDialog
                    rawString={lastValidSample.sample.raw_string}
                    totalTokens={lastValidSample.sample.total_tokens}
                    turns={lastValidSample.sample.turns.length}
                  />
                )}
                {lastValidSample && (
                  <>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => { setCollapseAllSignal((s) => s + 1); setDefaultSectionsOpen(false) }}
                    >
                      Collapse All
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={() => { setExpandAllSignal((s) => s + 1); setDefaultSectionsOpen(true) }}
                    >
                      Expand All
                    </Button>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" size="sm" className="h-7 px-2 text-xs whitespace-nowrap shrink-0">
                          Render{selectedOptionsCount > 0 && ` (${selectedOptionsCount})`}
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="start" className="w-auto">
                        <DropdownMenuCheckboxItem
                          checked={formatThinkBlocks}
                          onCheckedChange={(checked) => setFormatThinkBlocks(Boolean(checked))}
                          onSelect={(e) => e.preventDefault()}
                        >
                          Format Think
                        </DropdownMenuCheckboxItem>
                        <DropdownMenuCheckboxItem
                          checked={renderOptions.includes("markdown")}
                          onCheckedChange={() => toggleRenderOption("markdown")}
                          onSelect={(e) => e.preventDefault()}
                        >
                          Markdown
                        </DropdownMenuCheckboxItem>
                        <DropdownMenuCheckboxItem
                          checked={renderOptions.includes("latex")}
                          onCheckedChange={() => toggleRenderOption("latex")}
                          onSelect={(e) => e.preventDefault()}
                        >
                          LaTeX
                        </DropdownMenuCheckboxItem>
                        <DropdownMenuCheckboxItem
                          checked={renderOptions.includes("code")}
                          onCheckedChange={() => toggleRenderOption("code")}
                          onSelect={(e) => e.preventDefault()}
                        >
                          Code
                        </DropdownMenuCheckboxItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </>
                )}
              </div>
              {evalConfig && (
                <RolloutsMetricsSidebarToggle
                  openAtom={evalsMetricsSidebarOpenAtom}
                />
              )}
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 min-h-0 overflow-auto p-3">
          <div className={isSamplesSidebarOpen || isMetricsSidebarOpen ? "max-w-5xl mx-auto" : ""}>
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

            <RolloutsView
              prompts={mappedPrompts}
              data={mappedRollouts}
              samplesData={mappedSamplesData}
              rolloutMetrics={mappedRolloutMetrics}
              goldenAnswers={mappedGoldenAnswers}
              infoTurns={mappedInfoTurns}
              isLoading={isLoading}
              step={selectedStep ?? undefined}
              filterGroupId={selectedGroupId}
              filterSampleIdx={selectedSampleIdx}
              lastValidSampleAtom={evalsLastValidSampleAtom}
              collapseAllSignal={collapseAllSignal}
              expandAllSignal={expandAllSignal}
              defaultSectionsOpen={defaultSectionsOpen}
              envRewardRanges={envRewardRanges}
              envMetricRanges={envMetricRanges}
              renderOptions={renderOptions}
              formatThinkBlocks={formatThinkBlocks}
            />
          </div>
        </div>
      </div>

      {/* Metrics Sidebar (Right) — only when eval data is available */}
      {evalConfig && (
        <RolloutsMetricsPanel
          currentStep={selectedStep}
          maxSelectableStep={lastStep}
          openAtom={evalsMetricsSidebarOpenAtom}
          selectedMetricsAtom={evalsSelectedMetricsAtom}
          selectedStepAtom={evalsSelectedStepAtom}
          evalConfig={evalConfig}
        />
      )}
    </div>
  )
}
