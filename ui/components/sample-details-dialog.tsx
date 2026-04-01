
import { type ReactNode, useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAtom } from "jotai"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
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
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { RolloutsView, RawTextDialog, extractEnvRewardRanges, extractEnvMetricRanges, parseDataMetricRanges, mergeEnvMetricRanges, type EnvRewardRanges, type EnvMetricRanges } from "@/components/rollouts-view"
import { useSampleDetails, useRunSummary } from "@/hooks/use-run-data"
import { rolloutsFormatThinkAtom, rolloutsRenderOptionsAtom, type RolloutsRenderOption } from "@/lib/atoms"
import type {
  Prompt,
  GenerationRow,
  EnvResponseRow,
  ToolCallRow,
  SampleData,
  RolloutMetric,
  GoldenAnswer,
  InfoTurn,
  SampleDetailsDiscarded,
} from "@/lib/types"

function formatDiscardReason(reason: string) {
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

function getDiscardReasonVariant(reason: string): "destructive" | "secondary" | "outline" {
  switch (reason) {
    case "max_async":
      return "destructive"
    case "zero_advantage":
      return "destructive"
    default:
      return "outline"
  }
}

interface SampleDetailsDialogProps {
  runPath: string
  groupId: number
  sampleId: number
  isEval?: boolean
  isCanceled?: boolean
  children?: ReactNode
}

/** Map discarded data types to regular types so we can reuse RolloutsView */
function useDiscardedToRegularMapping(data: SampleDetailsDiscarded | null) {
  const mappedPrompts = useMemo<Prompt[] | undefined>(() => {
    if (!data?.prompts) return undefined
    return data.prompts.map((prompt) => ({
      step: prompt.trainer_step,
      group_id: prompt.group_id,
      env: prompt.env,
      system_prompt: prompt.system_prompt,
      tokens_system_prompt: prompt.tokens_system_prompt,
      prompt: prompt.prompt,
      tokens_prompt: prompt.tokens_prompt,
    }))
  }, [data])

  const mappedGenerations = useMemo<GenerationRow[] | undefined>(() => {
    if (!data?.generations) return undefined
    return data.generations.map((g) => ({
      ...g,
      step: g.trainer_step,
    } as unknown as GenerationRow))
  }, [data])

  const mappedEnvResponses = useMemo<EnvResponseRow[] | undefined>(() => {
    if (!data?.env_responses) return undefined
    return data.env_responses.map((e) => ({
      ...e,
      step: e.trainer_step,
    } as unknown as EnvResponseRow))
  }, [data])

  const mappedToolCalls = useMemo<ToolCallRow[] | undefined>(() => {
    if (!data?.tool_calls) return undefined
    return data.tool_calls.map((tc) => ({
      ...tc,
      step: tc.trainer_step,
    } as unknown as ToolCallRow))
  }, [data])

  const mappedSamplesData = useMemo<SampleData[] | undefined>(() => {
    if (!data?.samples_data) return undefined
    return data.samples_data.map((sample) => ({
      step: sample.trainer_step,
      group_id: sample.group_id,
      sample_id: sample.sample_id,
      reward: sample.reward,
      advantage: sample.advantage,
      num_generations: sample.num_generations,
      total_tokens: sample.total_tokens,
      raw_string: sample.raw_string,
      compute_reward_time: null,
      stop_reason: sample.stop_reason,
    }))
  }, [data])

  const mappedRolloutMetrics = useMemo<RolloutMetric[] | undefined>(() => {
    if (!data?.rollout_metrics) return undefined
    const step = data.trainer_step ?? 0
    return data.rollout_metrics.map((metric) => ({
      step,
      sample_id: metric.sample_id,
      env: metric.env,
      metric_name: metric.metric_name,
      value: metric.value,
    }))
  }, [data])

  const mappedGoldenAnswers = useMemo<GoldenAnswer[] | undefined>(() => {
    if (!data?.golden_answers) return undefined
    const step = data.trainer_step ?? 0
    return data.golden_answers.map((answer) => ({
      step,
      sample_id: answer.sample_id,
      env: answer.env,
      key: answer.key,
      value: answer.value,
    }))
  }, [data])

  const mappedInfoTurns = useMemo<InfoTurn[] | undefined>(() => {
    if (!data?.info_turns) return undefined
    const step = data.trainer_step ?? 0
    return data.info_turns.map((it) => ({
      step,
      sample_id: it.sample_id,
      agent_id: it.agent_id ?? 0,
      generation_idx: it.generation_idx,
      tool_call_idx: it.tool_call_idx ?? null,
      env: it.env,
      info_key: it.info_key,
      info_value: it.info_value,
      info_type: it.info_type,
    }))
  }, [data])

  return {
    mappedPrompts,
    mappedGenerations,
    mappedEnvResponses,
    mappedToolCalls,
    mappedSamplesData,
    mappedRolloutMetrics,
    mappedGoldenAnswers,
    mappedInfoTurns,
  }
}

export function SampleDetailsDialog({
  runPath,
  groupId,
  sampleId,
  isEval,
  isCanceled,
  children,
}: SampleDetailsDialogProps) {
  const [open, setOpen] = useState(false)
  const { data, isLoading, error } = useSampleDetails(
    runPath,
    groupId,
    sampleId,
    open,
    isEval
  )

  const { data: summaryData } = useRunSummary(runPath, open && !!runPath, false)

  const envConfigValue = isEval
    ? (summaryData?.summary?.eval_env_details ?? summaryData?.config?.eval_env_details)
    : summaryData?.config?.environments

  const envSummaryValue = isEval
    ? undefined
    : summaryData?.summary?.env_details

  const envRewardRanges = useMemo<EnvRewardRanges>(() => {
    return extractEnvRewardRanges(envConfigValue)
  }, [envConfigValue])

  const envMetricRanges = useMemo<EnvMetricRanges>(() => {
    const configRanges = extractEnvMetricRanges(envConfigValue)
    const summaryRanges = extractEnvMetricRanges(envSummaryValue)
    const dataRanges = parseDataMetricRanges(summaryData?.data_metric_ranges)
    return mergeEnvMetricRanges(
      mergeEnvMetricRanges(configRanges, summaryRanges),
      dataRanges,
    )
  }, [envConfigValue, envSummaryValue, summaryData?.data_metric_ranges])

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

  const discardedData =
    data?.kind === "rollouts_discarded" ? data : null

  const {
    mappedPrompts,
    mappedGenerations,
    mappedEnvResponses,
    mappedToolCalls,
    mappedSamplesData,
    mappedRolloutMetrics,
    mappedGoldenAnswers,
    mappedInfoTurns,
  } = useDiscardedToRegularMapping(discardedData)

  // For eval data the API returns eval-relative IDs (sample_idx as group_id,
  // completion_idx as sample_idx) which differ from the inference event IDs
  // passed via props. Extract actual IDs from the data so RolloutsView can
  // filter correctly.
  const evalFilterIds = useMemo(() => {
    if (!data || data.kind !== "eval") return null
    const firstSample = data.samples_data?.[0]
    if (!firstSample) return null
    return {
      groupId: firstSample.group_id,
      sampleIdx: firstSample.sample_id,
    }
  }, [data])

  const statusLabel =
    data?.kind === "rollouts_discarded"
      ? "Discarded"
      : data?.kind === "rollouts"
      ? "Kept"
      : data?.kind === "eval"
      ? "Eval"
      : null

  const navigate = useNavigate()

  const goToUrl = useMemo(() => {
    if (!data) return null
    if (data.kind === "rollouts") {
      return `/rollouts?step=${data.step}&group=${groupId}&sample=${sampleId}`
    }
    if (data.kind === "rollouts_discarded") {
      return `/rollouts-discarded?trainer_step=${data.trainer_step}&group=${groupId}&sample=${sampleId}`
    }
    if (data.kind === "eval") {
      return `/evals?step=${data.step}&eval_name=${encodeURIComponent(data.eval_name)}&group=${groupId}&sample=${sampleId}`
    }
    return null
  }, [data, groupId, sampleId])

  const handleGoTo = () => {
    if (!goToUrl) return
    setOpen(false)
    navigate(goToUrl)
  }

  // Toolbar state: collapse/expand, render options
  const [collapseAllSignal, setCollapseAllSignal] = useState(0)
  const [expandAllSignal, setExpandAllSignal] = useState(0)
  const [defaultSectionsOpen, setDefaultSectionsOpen] = useState(true)
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

  // Find the current sample's raw_string, total_tokens, turns for the Raw Text button
  const currentSampleData = useMemo(() => {
    if (!data) return null
    const samples =
      data.kind === "rollouts"
        ? data.samples_data
        : data.kind === "rollouts_discarded"
        ? mappedSamplesData
        : data.kind === "eval"
        ? data.samples_data
        : null
    if (!samples) return null
    const targetIdx = data.kind === "eval"
      ? (evalFilterIds?.sampleIdx ?? sampleId)
      : sampleId
    return samples.find((s) => s.sample_id === targetIdx) ?? null
  }, [data, mappedSamplesData, sampleId, evalFilterIds])

  const hasData = !isLoading && !error && data && data.kind !== null

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {children ?? (
          <Button
            variant="outline"
            size="sm"
            className="h-6 px-2 text-[10px]"
          >
            Check
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="sm:max-w-[85vw] max-h-[85vh] overflow-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 flex-wrap">
            <span>
              {data?.kind === "eval"
                ? `Completion ${sampleId} / Sample ${groupId}`
                : `Sample ${sampleId} / Group ${groupId}`}
            </span>
            {statusLabel && (
              <span className="text-sm font-normal text-muted-foreground">
                {statusLabel}
                {data?.kind === "eval" && data.eval_name && ` — ${data.eval_name}`}
              </span>
            )}
            {discardedData?.discard_reason && (
              <Badge
                variant={getDiscardReasonVariant(discardedData.discard_reason)}
                className="text-xs"
              >
                {formatDiscardReason(discardedData.discard_reason)}
              </Badge>
            )}
            {goToUrl && (
              <Button
                variant="outline"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={handleGoTo}
              >
                Go to
              </Button>
            )}
            {/* Toolbar options matching rollouts page */}
            {hasData && (
              <>
                {currentSampleData?.raw_string && (
                  <RawTextDialog
                    rawString={currentSampleData.raw_string}
                    totalTokens={currentSampleData.total_tokens}
                    turns={currentSampleData.num_generations ?? undefined}
                  />
                )}
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
          </DialogTitle>
        </DialogHeader>

        {error && (
          <Card className="border-destructive">
            <CardContent className="pt-6">
              <p className="text-sm text-destructive">
                Error: {error instanceof Error ? error.message : "Unknown error"}
              </p>
            </CardContent>
          </Card>
        )}

        {isLoading && !error && (
          <Card>
            <CardContent className="pt-6 space-y-3">
              <Skeleton className="h-5 w-40" />
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-24 w-full" />
              <div className="flex gap-2">
                <Skeleton className="h-6 w-20" />
                <Skeleton className="h-6 w-20" />
                <Skeleton className="h-6 w-20" />
              </div>
            </CardContent>
          </Card>
        )}

        {!isLoading && !error && data?.kind === "rollouts" && (
          <RolloutsView
            prompts={data.prompts}
            generations={data.generations}
            envResponses={data.env_responses}
            toolCalls={data.tool_calls}
            samplesData={data.samples_data}
            rolloutMetrics={data.rollout_metrics}
            goldenAnswers={data.golden_answers}
            infoTurns={data.info_turns}
            step={data.step}
            filterGroupId={groupId}
            filterSampleIdx={sampleId}
            envRewardRanges={envRewardRanges}
            envMetricRanges={envMetricRanges}
            advantageRange={advantageRange}
            collapseAllSignal={collapseAllSignal}
            expandAllSignal={expandAllSignal}
            defaultSectionsOpen={defaultSectionsOpen}
            renderOptions={renderOptions}
            formatThinkBlocks={formatThinkBlocks}
          />
        )}

        {!isLoading && !error && data?.kind === "rollouts_discarded" && (
          <RolloutsView
            prompts={mappedPrompts}
            generations={mappedGenerations}
            envResponses={mappedEnvResponses}
            toolCalls={mappedToolCalls}
            samplesData={mappedSamplesData}
            rolloutMetrics={mappedRolloutMetrics}
            goldenAnswers={mappedGoldenAnswers}
            infoTurns={mappedInfoTurns}
            step={data.trainer_step}
            goToBasePath="/rollouts-discarded"
            goToStepParam="trainer_step"
            filterGroupId={groupId}
            filterSampleIdx={sampleId}
            envRewardRanges={envRewardRanges}
            envMetricRanges={envMetricRanges}
            advantageRange={advantageRange}
            collapseAllSignal={collapseAllSignal}
            expandAllSignal={expandAllSignal}
            defaultSectionsOpen={defaultSectionsOpen}
            renderOptions={renderOptions}
            formatThinkBlocks={formatThinkBlocks}
          />
        )}

        {!isLoading && !error && data?.kind === "eval" && (
          <RolloutsView
            prompts={data.prompts}
            generations={data.generations}
            envResponses={data.env_responses}
            toolCalls={data.tool_calls}
            samplesData={data.samples_data}
            rolloutMetrics={data.rollout_metrics}
            goldenAnswers={data.golden_answers}
            infoTurns={data.info_turns}
            step={data.step}
            filterGroupId={evalFilterIds?.groupId ?? groupId}
            filterSampleIdx={evalFilterIds?.sampleIdx ?? sampleId}
            envRewardRanges={envRewardRanges}
            envMetricRanges={envMetricRanges}
            advantageRange={advantageRange}
            collapseAllSignal={collapseAllSignal}
            expandAllSignal={expandAllSignal}
            defaultSectionsOpen={defaultSectionsOpen}
            renderOptions={renderOptions}
            formatThinkBlocks={formatThinkBlocks}
          />
        )}

        {!isLoading && !error && data?.kind === null && (
          <Card>
            <CardContent className="pt-6">
              <p className="text-sm text-muted-foreground">
                {isCanceled
                  ? "Sample canceled due to async policy."
                  : "Sample not found in rollouts or discarded rollouts."}
              </p>
            </CardContent>
          </Card>
        )}
      </DialogContent>
    </Dialog>
  )
}
