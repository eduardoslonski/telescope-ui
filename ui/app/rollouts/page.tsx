
import { useEffect, useMemo, useRef, useState } from "react"
import { useAtom, useAtomValue } from "jotai"
import { useSearchParams } from "react-router-dom"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
} from "@/components/ui/card"
import { NoRunSelectedState } from "@/components/no-run-selected-state"
import { RolloutsView, RawTextDialog } from "@/components/rollouts-view"
import {
  RolloutsMetricsPanel,
  RolloutsMetricsSidebarToggle,
} from "@/components/rollouts-metrics-panel"
import {
  RolloutsSamplePickerSidebar,
  RolloutsSamplePickerSidebarToggle,
} from "@/components/rollouts-sample-picker-sidebar"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  selectedRunPathAtom,
  rolloutsSelectedStepAtom,
  rolloutsSelectedGroupIdAtom,
  rolloutsSelectedSampleIdxAtom,
  rolloutsLastValidSampleAtom,
  rolloutsSamplePickerSidebarOpenAtom,
  rolloutsSamplePickerViewModeAtom,
  rolloutsMetricsSidebarOpenAtom,
  rolloutsRenderOptionsAtom,
  rolloutsFormatThinkAtom,
  type RolloutsRenderOption,
} from "@/lib/atoms"
import { useRollouts, useRunSummary } from "@/hooks/use-run-data"
import { extractEnvRewardRanges, extractEnvMetricRanges, parseDataMetricRanges, mergeEnvMetricRanges, type EnvRewardRanges, type EnvMetricRanges } from "@/components/rollouts-view"

export default function RolloutsPage() {
  const selectedRunPath = useAtomValue(selectedRunPathAtom)
  const [searchParams] = useSearchParams()

  // Selected step (persisted in atom)
  const [selectedStep, setSelectedStep] = useAtom(rolloutsSelectedStepAtom)

  // Sample picker filter state - keep selection even when changing steps
  const [selectedGroupId, setSelectedGroupId] = useAtom(rolloutsSelectedGroupIdAtom)
  const [selectedSampleIdx, setSelectedSampleIdx] = useAtom(rolloutsSelectedSampleIdxAtom)
  const lastValidSample = useAtomValue(rolloutsLastValidSampleAtom)
  const [isSamplesSidebarOpen, setIsSamplesSidebarOpen] = useAtom(rolloutsSamplePickerSidebarOpenAtom)
  const [, setViewMode] = useAtom(rolloutsSamplePickerViewModeAtom)
  const isMetricsSidebarOpen = useAtomValue(rolloutsMetricsSidebarOpenAtom)
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

  // Collapse / Expand all signals + default for new samples
  const [collapseAllSignal, setCollapseAllSignal] = useState(0)
  const [expandAllSignal, setExpandAllSignal] = useState(0)
  const [defaultSectionsOpen, setDefaultSectionsOpen] = useState(true)

  // Keep rollouts data fresh like /metrics
  const shouldPoll = true

  const requestedStep = useMemo(() => {
    const raw = searchParams.get("step")
    if (!raw) return null
    const parsed = Number(raw)
    return Number.isFinite(parsed) ? parsed : null
  }, [searchParams])

  const requestedSample = useMemo(() => {
    const raw = searchParams.get("sample")
    if (!raw) return null
    const parsed = Number(raw)
    return Number.isFinite(parsed) ? parsed : null
  }, [searchParams])

  const requestedGroup = useMemo(() => {
    const raw = searchParams.get("group")
    if (!raw) return null
    const parsed = Number(raw)
    return Number.isFinite(parsed) ? parsed : null
  }, [searchParams])

  const lastRequestedStepRef = useRef<number | null>(null)
  const lastRequestedSampleRef = useRef<{ group: number | null; sample: number | null }>({ group: null, sample: null })

  useEffect(() => {
    if (!selectedRunPath) {
      lastRequestedStepRef.current = null
      lastRequestedSampleRef.current = { group: null, sample: null }
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

  // Apply group + sample selection from URL params (e.g. from "Go to" button)
  useEffect(() => {
    if (!selectedRunPath) return
    if (requestedGroup == null || requestedSample == null) return
    if (
      lastRequestedSampleRef.current.group === requestedGroup &&
      lastRequestedSampleRef.current.sample === requestedSample
    ) return
    setSelectedGroupId(requestedGroup)
    setSelectedSampleIdx(requestedSample)
    setIsSamplesSidebarOpen(true)
    setViewMode("samples")
    lastRequestedSampleRef.current = { group: requestedGroup, sample: requestedSample }
  }, [requestedGroup, requestedSample, selectedRunPath, setSelectedGroupId, setSelectedSampleIdx, setIsSamplesSidebarOpen, setViewMode])

  const {
    data: rolloutsData,
    isLoading,
    error,
  } = useRollouts(
    selectedRunPath || "",
    selectedStep,
    !!selectedRunPath,
    shouldPoll
  )

  const { data: summaryData } = useRunSummary(
    selectedRunPath || "",
    !!selectedRunPath,
    shouldPoll
  )

  // Extract per-environment reward ranges from run config
  const envRewardRanges = useMemo<EnvRewardRanges>(() => {
    return extractEnvRewardRanges(summaryData?.config?.environments)
  }, [summaryData?.config?.environments])

  // Extract per-environment, per-metric ranges from run config + data fallback
  const envMetricRanges = useMemo<EnvMetricRanges>(() => {
    const configRanges = extractEnvMetricRanges(summaryData?.config?.environments)
    const dataRanges = parseDataMetricRanges(summaryData?.data_metric_ranges)
    return mergeEnvMetricRanges(configRanges, dataRanges)
  }, [summaryData?.config?.environments, summaryData?.data_metric_ranges])

  // Compute advantage range from group_size config
  const advantageRange = useMemo(() => {
    const raw = summaryData?.config?.group_size
    const gs = typeof raw === "object" && raw !== null && "value" in raw
      ? Number((raw as { value: unknown }).value)
      : Number(raw)
    if (!gs || isNaN(gs) || gs <= 1) return null
    const max = Math.sqrt(gs - 1)
    return { min: -max, max }
  }, [summaryData?.config?.group_size])

  // Compute initial step when available steps change and no step is selected
  const initialStep = useMemo(() => {
    if (rolloutsData?.available_steps?.length && selectedStep === null) {
      return rolloutsData.available_steps[
        rolloutsData.available_steps.length - 1
      ]
    }
    return null
  }, [rolloutsData, selectedStep])

  // Apply initial step only when needed
  useEffect(() => {
    if (initialStep !== null && selectedStep === null && requestedStep == null) {
      setSelectedStep(initialStep)
    }
  }, [initialStep, requestedStep, selectedStep, setSelectedStep])

  const handleStepChange = (step: number) => {
    setSelectedStep(step)
  }

  // Keep track of all known steps - merge to avoid losing steps when switching between cached responses
  const [lastKnownSteps, setLastKnownSteps] = useState<number[]>([])
  const [lastRunPath, setLastRunPath] = useState<string | null>(null)

  // Reset steps when run changes
  if (selectedRunPath !== lastRunPath) {
    setLastKnownSteps([])
    setLastRunPath(selectedRunPath)
  }

  const availableSteps = rolloutsData?.available_steps
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

  // Current step index for pagination controls
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

  // No run selected
  if (!selectedRunPath) {
    return <NoRunSelectedState description="Select a run from the sidebar to view rollouts." />
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sample Picker Sidebar (Left) */}
      <RolloutsSamplePickerSidebar
        prompts={rolloutsData?.prompts}
        data={rolloutsData?.rollouts}
        samplesData={rolloutsData?.samples_data}
        rolloutMetrics={rolloutsData?.rollout_metrics}
        availableMetricNames={rolloutsData?.available_rollout_metric_names}
        isLoading={isLoading}
        step={selectedStep ?? undefined}
        currentStepIndex={currentStepIndex}
        totalSteps={sortedSteps.length}
        onStepChange={handleStepIndexChange}
        hasSteps={hasSteps}
        stepValues={sortedSteps}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
        {/* Header - same height as /metrics */}
        <header className="border-b bg-card/50 backdrop-blur-sm shrink-0">
          <div className="py-2.5 px-4">
            <div className="flex items-center justify-between h-7">
              <div className="flex items-center gap-3">
                <RolloutsSamplePickerSidebarToggle />
                {/* Sample info: Step, Group ID, Sample ID */}
                {lastValidSample && (
                  <div className="text-sm text-muted-foreground">
                    <span>Step {lastValidSample.sample.turns[0]?.step ?? selectedStep}</span>
                    {" · "}
                    <span>Group {lastValidSample.sample.group_id}</span>
                    {" · "}
                    <span>Sample {lastValidSample.sample.sample_idx}</span>
                  </div>
                )}
                {/* Raw Text button */}
                {lastValidSample?.sample.raw_string && (
                  <RawTextDialog
                    rawString={lastValidSample.sample.raw_string}
                    totalTokens={lastValidSample.sample.total_tokens}
                    turns={lastValidSample.sample.turns.length}
                  />
                )}
                {/* Collapse / Expand All */}
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
              <RolloutsMetricsSidebarToggle />
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
              prompts={rolloutsData?.prompts}
              data={rolloutsData?.rollouts}
              samplesData={rolloutsData?.samples_data}
              rolloutMetrics={rolloutsData?.rollout_metrics}
              goldenAnswers={rolloutsData?.golden_answers}
              infoTurns={rolloutsData?.info_turns}
              isLoading={isLoading}
              step={selectedStep ?? undefined}
              scrollToSampleId={requestedSample}
              filterGroupId={selectedGroupId}
              filterSampleIdx={selectedSampleIdx}
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
        </div>
      </div>

      {/* Metrics Sidebar (Right) */}
      <RolloutsMetricsPanel currentStep={selectedStep} maxSelectableStep={lastStep} />
    </div>
  )
}
