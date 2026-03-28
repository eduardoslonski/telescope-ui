
import { useMemo, useRef } from "react"
import {
  useQuery,
  useQueryClient,
  keepPreviousData,
  useQueries,
} from "@tanstack/react-query"
import { API_BASE, POLL_INTERVAL } from "@/lib/constants"
import type {
  RolloutsResponse,
  RolloutsDiscardedResponse,
  EvalsResponse,
  SampleDetailsResponse,
  SampleStatusesResponse,
  TimelinePaginatedResponse,
  InferenceGroupEventsResponse,
  TrainerBreakdownEventsResponse,
  RunSummary,
  RunCodeTreeResponse,
  RunCodeFileResponse,
  RunCodeDiffSummaryResponse,
  StepMetricsMultiRunResponse,
  StepTimesResponse,
  StepMetricsResponse,
  StepHistogramResponse,
  StepDistributionOverTimeResponse,
  RunsResponse,
  RemovedRunsResponse,
  GpuMetric,
  CpuMetric,
  VllmMetric,
  CustomMetricsLayoutResponse,
  CustomMetricsTemplatesResponse,
  CustomMetricsTemplateResponse,
  InferencePerformanceResponse,
  TrainerPerformanceResponse,
  LogsResponse,
  LogsSummaryResponse,
  InflightSnapshot,
} from "@/lib/types"

// ============================================================================
// Runs Query
// ============================================================================

export function useRuns() {
  return useQuery<RunsResponse>({
    queryKey: ["runs"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/runs`)
      if (!response.ok) {
        throw new Error("Failed to fetch runs")
      }
      return response.json()
    },
    refetchInterval: (query) => {
      // Poll faster during W&B discovery so progress updates feel smooth
      const isDiscovering = query.state.data?.discovery?.status === "discovering"
      return isDiscovering ? 1500 : POLL_INTERVAL
    },
  })
}

export function useRemovedRuns(enabled = true) {
  return useQuery<RemovedRunsResponse>({
    queryKey: ["removed-runs"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/removed-runs`)
      if (!response.ok) {
        throw new Error("Failed to fetch removed runs")
      }
      return response.json()
    },
    enabled,
  })
}

export interface KnownProject {
  project: string
  source: "user" | "derived"
  added_at: string | null
}

export interface KnownProjectsResponse {
  projects: KnownProject[]
}

export function useKnownProjects(enabled = true) {
  return useQuery<KnownProjectsResponse>({
    queryKey: ["known-projects"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/known-projects`)
      if (!response.ok) {
        throw new Error("Failed to fetch known projects")
      }
      return response.json()
    },
    enabled,
  })
}

// ============================================================================
// Data Queries
// ============================================================================

export function useRollouts(
  runPath: string,
  selectedStep: number | null,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<RolloutsResponse>({
    queryKey: ["rollouts", runPath, selectedStep],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/rollouts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          step: selectedStep,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch rollouts")
      }
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
  })
}

export function useRolloutsDiscarded(
  runPath: string,
  selectedTrainerStep: number | null,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<RolloutsDiscardedResponse>({
    queryKey: ["rollouts-discarded", runPath, selectedTrainerStep],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/rollouts-discarded`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          trainer_step: selectedTrainerStep,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch discarded rollouts")
      }
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
  })
}

export function useEvals(
  runPath: string,
  selectedStep: number | null,
  evalName: string | null,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<EvalsResponse>({
    queryKey: ["evals", runPath, selectedStep, evalName],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/evals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          step: selectedStep,
          eval_name: evalName,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch evals")
      }
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
  })
}

export function useSampleDetails(
  runPath: string,
  groupId: number | null,
  sampleIdx: number | null,
  enabled: boolean,
  isEval?: boolean
) {
  return useQuery<SampleDetailsResponse>({
    queryKey: ["sample-details", runPath, groupId, sampleIdx, isEval ?? false],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/sample-details`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          group_id: groupId,
          sample_idx: sampleIdx,
          is_eval: isEval ?? false,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch sample details")
      }
      return response.json()
    },
    enabled: enabled && !!runPath && groupId != null && sampleIdx != null,
  })
}

export function useSampleStatuses(
  runPath: string,
  samples: Array<{ group_id: number; sample_idx: number }>,
  enabled: boolean,
  shouldPoll: boolean = false
) {
  const normalizedSamples = useMemo(() => {
    const seen = new Set<string>()
    const unique: Array<{ group_id: number; sample_idx: number }> = []
    for (const sample of samples) {
      const key = `${sample.group_id}:${sample.sample_idx}`
      if (seen.has(key)) continue
      seen.add(key)
      unique.push(sample)
    }
    return unique.sort(
      (a, b) => a.group_id - b.group_id || a.sample_idx - b.sample_idx
    )
  }, [samples])

  const sampleKey = useMemo(() => {
    return normalizedSamples
      .map((sample) => `${sample.group_id}:${sample.sample_idx}`)
      .join("|")
  }, [normalizedSamples])

  return useQuery<SampleStatusesResponse>({
    queryKey: ["sample-statuses", runPath, sampleKey],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/sample-statuses`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          samples: normalizedSamples,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch sample statuses")
      }
      return response.json()
    },
    enabled: enabled && !!runPath && normalizedSamples.length > 0,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
  })
}

export function useRunSummary(
  runPath: string,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<RunSummary>({
    queryKey: ["run-summary", runPath],
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE}/run-summary/${encodeURIComponent(runPath)}`
      )
      if (!response.ok) {
        throw new Error("Failed to fetch run summary")
      }
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
  })
}

/**
 * Fetch summaries for multiple runs and merge metric catalog fields.
 * Returns the union of custom_metric_sections, reward names, evals, tags, and envs.
 */
export function useRunSummaries(
  runPaths: string[],
  shouldPoll: boolean,
) {
  const queries = useQueries({
    queries: runPaths.map((runPath) => ({
      queryKey: ["run-summary", runPath],
      queryFn: async (): Promise<RunSummary> => {
        const response = await fetch(
          `${API_BASE}/run-summary/${encodeURIComponent(runPath)}`
        )
        if (!response.ok) {
          throw new Error("Failed to fetch run summary")
        }
        return response.json()
      },
      enabled: !!runPath,
      refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    })),
  })

  return useMemo(() => {
    const summaries = queries
      .map((q) => q.data)
      .filter((d): d is RunSummary => d != null)

    // Merge custom_metric_sections: union of all sections and their metric keys
    const mergedSections: Record<string, Record<string, string[]>> = {}
    for (const s of summaries) {
      const sections = s.step_metrics_info?.custom_metric_sections ?? {}
      for (const [sectionName, groups] of Object.entries(sections)) {
        if (!mergedSections[sectionName]) {
          mergedSections[sectionName] = {}
        }
        for (const [groupName, metrics] of Object.entries(groups)) {
          if (!mergedSections[sectionName][groupName]) {
            mergedSections[sectionName][groupName] = []
          }
          const existing = new Set(mergedSections[sectionName][groupName])
          for (const m of metrics) {
            if (!existing.has(m)) {
              mergedSections[sectionName][groupName].push(m)
            }
          }
        }
      }
    }

    // Merge reward names
    const rewardNameSet = new Set<string>()
    for (const s of summaries) {
      for (const name of s.available_rollout_metric_names ?? []) {
        rewardNameSet.add(name)
      }
    }

    // Merge evals (union by eval_name, merge their metric names)
    const evalsMap = new Map<string, Set<string>>()
    for (const s of summaries) {
      for (const e of s.eval_info?.evals ?? []) {
        if (!evalsMap.has(e.eval_name)) {
          evalsMap.set(e.eval_name, new Set())
        }
        const set = evalsMap.get(e.eval_name)!
        for (const m of e.available_rollout_metric_names) {
          set.add(m)
        }
      }
    }
    const mergedEvals = Array.from(evalsMap.entries()).map(([eval_name, metrics]) => ({
      eval_name,
      available_rollout_metric_names: Array.from(metrics),
    }))

    // Merge sample tags
    const mergedTags: Record<string, string[]> = {}
    for (const s of summaries) {
      for (const [key, values] of Object.entries(s.available_sample_tags ?? {})) {
        if (!mergedTags[key]) {
          mergedTags[key] = []
        }
        const existing = new Set(mergedTags[key])
        for (const v of values) {
          if (!existing.has(v)) {
            mergedTags[key].push(v)
          }
        }
      }
    }

    // Merge envs
    const envSet = new Set<string>()
    for (const s of summaries) {
      for (const env of s.available_envs ?? []) {
        envSet.add(env)
      }
    }

    // Total steps from the selected (first) run, or max across all
    const totalSteps = Math.max(0, ...summaries.map((s) => s.step_metrics_info?.local_steps ?? 0))

    const isLoading = queries.some((q) => q.isLoading)
    const error = queries.find((q) => q.error)?.error ?? null

    return {
      customMetricSections: mergedSections,
      availableRewardNames: Array.from(rewardNameSet),
      evalsList: mergedEvals,
      availableSampleTags: mergedTags,
      availableEnvs: Array.from(envSet),
      totalSteps,
      isLoading,
      error,
    }
  }, [queries])
}

// ============================================================================
// Logs Queries
// ============================================================================

export function useLogs(
  runPath: string,
  page: number,
  filters: {
    components?: string[]
    levels?: string[]
    sources?: string[]
    search?: string
  },
  enabled: boolean
) {
  return useQuery<LogsResponse>({
    queryKey: ["logs", runPath, page, filters],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/api/logs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          page,
          page_size: 500,
          ...filters,
        }),
      })
      if (!response.ok) throw new Error("Failed to fetch logs")
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: POLL_INTERVAL,
    placeholderData: keepPreviousData,
  })
}

export function useLogsSummary(runPath: string, enabled: boolean) {
  return useQuery<LogsSummaryResponse>({
    queryKey: ["logs-summary", runPath],
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE}/api/logs/summary/${encodeURIComponent(runPath)}`
      )
      if (!response.ok) throw new Error("Failed to fetch logs summary")
      return response.json()
    },
    enabled: enabled && !!runPath,
    staleTime: 10000,
  })
}

// ============================================================================
// Run Code Queries
// ============================================================================

export function useRunCodeTree(runPath: string, enabled: boolean) {
  return useQuery<RunCodeTreeResponse>({
    queryKey: ["run-code-tree", runPath],
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE}/run-code/tree/${encodeURIComponent(runPath)}`
      )
      if (!response.ok) {
        throw new Error("Failed to fetch run code tree")
      }
      return response.json()
    },
    enabled: enabled && !!runPath,
    staleTime: 60000,
  })
}

export function useRunCodeFile(
  runPath: string,
  filePath: string,
  enabled: boolean
) {
  return useQuery<RunCodeFileResponse>({
    queryKey: ["run-code-file", runPath, filePath],
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE}/run-code/file/${encodeURIComponent(runPath)}?file_path=${encodeURIComponent(filePath)}`
      )
      if (!response.ok) {
        throw new Error("Failed to fetch run code file")
      }
      return response.json()
    },
    enabled: enabled && !!runPath && !!filePath,
    staleTime: 60000,
    placeholderData: keepPreviousData,
  })
}

export function useRunCodeDiffSummary(
  leftRunPath: string,
  rightRunPath: string,
  enabled: boolean
) {
  return useQuery<RunCodeDiffSummaryResponse>({
    queryKey: ["run-code-diff-summary", leftRunPath, rightRunPath],
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE}/run-code/diff-summary/${encodeURIComponent(leftRunPath)}?right_run_path=${encodeURIComponent(rightRunPath)}`
      )
      if (!response.ok) {
        throw new Error("Failed to fetch run code diff summary")
      }
      return response.json()
    },
    enabled: enabled && !!leftRunPath && !!rightRunPath,
    staleTime: 60000,
    placeholderData: keepPreviousData,
  })
}

export function useTimelinePaginated(
  runPath: string,
  page: number,
  intervalSeconds: number,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<TimelinePaginatedResponse>({
    queryKey: ["timeline-paginated", runPath, page, intervalSeconds],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/events/timeline-paginated`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          page,
          interval_seconds: intervalSeconds,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch paginated timeline")
      }
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    placeholderData: keepPreviousData,
    gcTime: 0,
  })
}

export function useInflightGenerations(
  runPath: string,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<InflightSnapshot>({
    queryKey: ["inflight-generations", runPath],
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE}/events/inflight/${encodeURIComponent(runPath)}`
      )
      if (!response.ok) {
        throw new Error("Failed to fetch inflight generations")
      }
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    placeholderData: keepPreviousData,
  })
}

export function useInferenceEventsByGroup(
  runPath: string,
  groupId: number | null,
  enabled: boolean
) {
  return useQuery<InferenceGroupEventsResponse>({
    queryKey: ["inference-events-by-group", runPath, groupId],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/events/inference-by-group`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          group_id: groupId,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch inference events by group")
      }
      return response.json()
    },
    enabled: enabled && !!runPath && groupId !== null,
  })
}

export function useTrainerBreakdownEvents(
  runPath: string,
  parentEventType: string | null,
  rank: number | null,
  step: number | null,
  enabled: boolean
) {
  return useQuery<TrainerBreakdownEventsResponse>({
    queryKey: [
      "trainer-breakdown-events",
      runPath,
      parentEventType,
      rank,
      step,
    ],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/events/trainer-breakdown`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          parent_event_type: parentEventType,
          rank,
          step,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch trainer breakdown events")
      }
      return response.json()
    },
    enabled:
      enabled &&
      !!runPath &&
      !!parentEventType &&
      rank !== null &&
      step !== null,
  })
}

// ============================================================================
// Inference Performance Query
// ============================================================================

export function useInferencePerformance(
  runPath: string,
  enabled: boolean,
  shouldPoll: boolean,
  bucketSeconds: number = 60,
) {
  return useQuery<InferencePerformanceResponse>({
    queryKey: ["inference-performance", runPath, bucketSeconds],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/inference-performance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_path: runPath, bucket_seconds: bucketSeconds }),
      })
      if (!response.ok) throw new Error("Failed to fetch inference performance")
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    placeholderData: keepPreviousData,
  })
}

// ============================================================================
// Trainer Performance Query
// ============================================================================

export function useTrainerPerformance(
  runPath: string,
  enabled: boolean,
  shouldPoll: boolean,
  bucketSeconds: number = 60,
) {
  return useQuery<TrainerPerformanceResponse>({
    queryKey: ["trainer-performance", runPath, bucketSeconds],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/trainer-performance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_path: runPath, bucket_seconds: bucketSeconds }),
      })
      if (!response.ok) throw new Error("Failed to fetch trainer performance")
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    placeholderData: keepPreviousData,
  })
}

// ============================================================================
// System Metrics Queries
// ============================================================================

export interface TrainerGpuMetricsFilter {
  rank: number
  node_id?: number | null
  local_rank?: number | null
  source?: string | null
}

export function useGpuMetricsForTrainerRanks(
  runPath: string,
  metricNames: string[] | null,
  rankFilters: TrainerGpuMetricsFilter[],
  startTime: number | null,
  endTime: number | null,
  enabled: boolean,
  shouldPoll: boolean
) {
  const normalizedFilters = rankFilters.filter((filter) =>
    Number.isFinite(filter.rank)
  )

  return useQueries({
    queries: normalizedFilters.map((filter) => ({
      queryKey: [
        "gpu-metrics-rank",
        runPath,
        metricNames,
        filter.rank,
        filter.node_id ?? null,
        filter.local_rank ?? null,
        filter.source ?? null,
        startTime,
        endTime,
      ],
      queryFn: async () => {
        const response = await fetch(`${API_BASE}/system-metrics/gpu`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            run_path: runPath,
            metric_names: metricNames,
            rank: filter.rank,
            node_id: filter.node_id ?? null,
            local_rank: filter.local_rank ?? null,
            source: filter.source ?? null,
            include_associated_system_metrics: true,
            include_edge_points: true,
            start_time: startTime,
            end_time: endTime,
            limit: 50000,
          }),
        })
        if (!response.ok) {
          throw new Error("Failed to fetch rank-scoped GPU metrics")
        }
        return response.json()
      },
      enabled: enabled && !!runPath && Number.isFinite(filter.rank),
      refetchInterval: shouldPoll ? POLL_INTERVAL : false,
      placeholderData: keepPreviousData,
    })),
  })
}

// ============================================================================
// Paginated System Metrics Queries (for Infra page)
// ============================================================================

export interface PaginatedGpuMetricsResponse {
  metrics: GpuMetric[]
  total_pages: number
  current_page: number
  interval_start: number
  interval_end: number
  global_min_time: number | null
  global_max_time: number | null
  available_metrics: string[]
  available_gpus: number[]
}

export function usePaginatedGpuMetrics(
  runPath: string,
  page: number,
  intervalSeconds: number,
  alignToLatest: boolean,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<PaginatedGpuMetricsResponse>({
    queryKey: [
      "gpu-metrics-paginated",
      runPath,
      page,
      intervalSeconds,
      alignToLatest,
    ],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/system-metrics/gpu-paginated`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          page,
          interval_seconds: intervalSeconds,
          align_to_latest: alignToLatest,
        }),
      })
      if (!response.ok) throw new Error("Failed to fetch paginated GPU metrics")
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    placeholderData: keepPreviousData,
  })
}

export interface PaginatedCpuMetricsResponse {
  metrics: CpuMetric[]
  total_pages: number
  current_page: number
  interval_start: number
  interval_end: number
  global_min_time: number | null
  global_max_time: number | null
  available_metrics: string[]
}

export function usePaginatedCpuMetrics(
  runPath: string,
  page: number,
  intervalSeconds: number,
  anchorStartTime: number | null,
  alignToLatest: boolean,
  metricNames: string[] | null,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<PaginatedCpuMetricsResponse>({
    queryKey: [
      "cpu-metrics-paginated",
      runPath,
      page,
      intervalSeconds,
      anchorStartTime,
      alignToLatest,
      metricNames,
    ],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/system-metrics/cpu-paginated`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          page,
          interval_seconds: intervalSeconds,
          anchor_start_time: anchorStartTime,
          align_to_latest: alignToLatest,
          metric_names: metricNames,
        }),
      })
      if (!response.ok) throw new Error("Failed to fetch paginated CPU metrics")
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    placeholderData: keepPreviousData,
  })
}

// ============================================================================
// Paginated vLLM Metrics Queries (for Infra page)
// ============================================================================

export interface PaginatedVllmMetricsResponse {
  metrics: VllmMetric[]
  total_pages: number
  current_page: number
  interval_start: number
  interval_end: number
  global_min_time: number | null
  global_max_time: number | null
  available_metrics: string[]
  available_servers: number[]
}

export function usePaginatedVllmMetrics(
  runPath: string,
  page: number,
  intervalSeconds: number,
  anchorStartTime: number | null,
  alignToLatest: boolean,
  metricNames: string[] | null,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<PaginatedVllmMetricsResponse>({
    queryKey: [
      "vllm-metrics-paginated",
      runPath,
      page,
      intervalSeconds,
      anchorStartTime,
      alignToLatest,
      metricNames,
    ],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/vllm-metrics/paginated`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          page,
          interval_seconds: intervalSeconds,
          anchor_start_time: anchorStartTime,
          align_to_latest: alignToLatest,
          metric_names: metricNames,
        }),
      })
      if (!response.ok) throw new Error("Failed to fetch paginated vLLM metrics")
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    placeholderData: keepPreviousData,
  })
}

// ============================================================================
// Step Metrics Queries (Per-step training metrics)
// ============================================================================

export function useStepMetricsMultiRun(
  runPaths: string[],
  metricNames: string[] | null,
  enabled: boolean,
  shouldPoll: boolean,
  tagFilters?: Record<string, string[]>,
  envFilters?: string[]
) {
  const queryClient = useQueryClient()
  const prevRunPathsRef = useRef<string[]>([])

  // When runs are only removed (not added), pre-seed the new query key's cache
  // with filtered data from the old key. Combined with staleTime, this avoids
  // an unnecessary immediate re-fetch — the next poll cycle will fetch normally.
  const runPathsKey = runPaths.join("\0")
  const prevRunPathsKey = prevRunPathsRef.current.join("\0")
  if (runPathsKey !== prevRunPathsKey) {
    const prevPaths = prevRunPathsRef.current
    if (prevPaths.length > 0 && runPaths.length < prevPaths.length) {
      const prevSet = new Set(prevPaths)
      const isRemovalOnly = runPaths.every((p) => prevSet.has(p))
      if (isRemovalOnly) {
        const oldKey = ["step-metrics-multi", prevPaths, metricNames, tagFilters, envFilters]
        const oldData =
          queryClient.getQueryData<StepMetricsMultiRunResponse>(oldKey)
        if (oldData) {
          const currSet = new Set(runPaths)
          const newKey = ["step-metrics-multi", runPaths, metricNames, tagFilters, envFilters]
          queryClient.setQueryData<StepMetricsMultiRunResponse>(newKey, {
            runs: oldData.runs.filter((r) => currSet.has(r.run_path)),
          })
        }
      }
    }
    prevRunPathsRef.current = runPaths
  }

  return useQuery<StepMetricsMultiRunResponse>({
    queryKey: ["step-metrics-multi", runPaths, metricNames, tagFilters, envFilters],
    queryFn: async () => {
      const body: Record<string, unknown> = {
        run_paths: runPaths,
        metric_names: metricNames,
        limit: 100000,
      }
      if (tagFilters && Object.keys(tagFilters).length > 0) {
        body.tag_filters = tagFilters
      }
      if (envFilters && envFilters.length > 0) {
        body.env_filters = envFilters
      }
      const response = await fetch(`${API_BASE}/step-metrics/multi`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch step metrics (multi-run)")
      }
      return response.json()
    },
    enabled: enabled && runPaths.length > 0,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    staleTime: POLL_INTERVAL,
  })
}

export function useStepTimes(
  runPath: string,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<StepTimesResponse>({
    queryKey: ["step-times", runPath],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/step-times`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch step times")
      }
      return response.json()
    },
    enabled: enabled && !!runPath,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
  })
}

export function useStepMetricSingle(
  runPath: string,
  metricName: string,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<StepMetricsResponse>({
    queryKey: ["step-metric-single", runPath, metricName],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/step-metrics`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          metric_names: [metricName],
          limit: 100000,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch step metric")
      }
      return response.json()
    },
    enabled: enabled && !!runPath && !!metricName,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    placeholderData: keepPreviousData,
  })
}

export function useEvalStepMetricSingle(
  runPath: string,
  evalName: string,
  metricName: string,
  enabled: boolean,
  shouldPoll: boolean,
  sampleIdx?: number
) {
  return useQuery<StepMetricsResponse>({
    queryKey: [
      "eval-step-metric-single",
      runPath,
      evalName,
      metricName,
      sampleIdx ?? null,
    ],
    queryFn: async () => {
      const body: Record<string, unknown> = {
        run_path: runPath,
        eval_name: evalName,
        metric_names: [metricName],
        limit: 100000,
      }
      if (sampleIdx !== undefined) {
        body.sample_idx = sampleIdx
      }
      const response = await fetch(`${API_BASE}/eval-step-metrics`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch eval step metric")
      }
      return response.json()
    },
    enabled: enabled && !!runPath && !!evalName && !!metricName,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    placeholderData: keepPreviousData,
  })
}

export interface EvalStepMetricsMultiRunResponse {
  runs: Array<{
    run_path: string
    metrics: Array<{ step: number; value: number }>
  }>
}

export function useEvalStepMetricsMultiRun(
  runPaths: string[],
  evalName: string,
  metricNames: string[],
  enabled: boolean,
  shouldPoll: boolean,
  envFilters?: string[]
) {
  const queryClient = useQueryClient()
  const prevRunPathsRef = useRef<string[]>([])

  // Pre-seed cache on removal-only changes (same pattern as useStepMetricsMultiRun)
  const runPathsKey = runPaths.join("\0")
  const prevRunPathsKey = prevRunPathsRef.current.join("\0")
  if (runPathsKey !== prevRunPathsKey) {
    const prevPaths = prevRunPathsRef.current
    if (prevPaths.length > 0 && runPaths.length < prevPaths.length) {
      const prevSet = new Set(prevPaths)
      const isRemovalOnly = runPaths.every((p) => prevSet.has(p))
      if (isRemovalOnly) {
        const oldKey = [
          "eval-step-metrics-multi",
          prevPaths,
          evalName,
          metricNames,
          envFilters,
        ]
        const oldData =
          queryClient.getQueryData<EvalStepMetricsMultiRunResponse>(oldKey)
        if (oldData) {
          const currSet = new Set(runPaths)
          const newKey = [
            "eval-step-metrics-multi",
            runPaths,
            evalName,
            metricNames,
            envFilters,
          ]
          queryClient.setQueryData<EvalStepMetricsMultiRunResponse>(newKey, {
            runs: oldData.runs.filter((r) => currSet.has(r.run_path)),
          })
        }
      }
    }
    prevRunPathsRef.current = runPaths
  }

  return useQuery<EvalStepMetricsMultiRunResponse>({
    queryKey: ["eval-step-metrics-multi", runPaths, evalName, metricNames, envFilters],
    queryFn: async () => {
      const results = await Promise.all(
        runPaths.map(async (runPath) => {
          const reqBody: Record<string, unknown> = {
            run_path: runPath,
            eval_name: evalName,
            metric_names: metricNames,
            limit: 100000,
          }
          if (envFilters && envFilters.length > 0) {
            reqBody.env_filters = envFilters
          }
          const response = await fetch(`${API_BASE}/eval-step-metrics`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(reqBody),
          })
          if (!response.ok) {
            return { run_path: runPath, metrics: [] }
          }
          const data = await response.json()
          return { run_path: runPath, metrics: data.metrics ?? [] }
        })
      )
      return { runs: results }
    },
    enabled:
      enabled && runPaths.length > 0 && !!evalName && metricNames.length > 0,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    staleTime: POLL_INTERVAL,
  })
}


// ============================================================================
// Step Histogram Query (for distribution visualization)
// ============================================================================

export function useStepHistogram(
  runPath: string,
  step: number | null,
  metricType: string,
  enabled: boolean
) {
  return useQuery<StepHistogramResponse>({
    queryKey: ["step-histogram", runPath, step, metricType],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/step-histogram`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          step: step,
          metric_type: metricType,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch step histogram")
      }
      return response.json()
    },
    enabled: enabled && !!runPath && step !== null && !!metricType,
    staleTime: 60000, // Cache for 1 minute
    placeholderData: keepPreviousData,
  })
}

// ============================================================================
// Step Distribution Over Time Query (for heatmap visualization)
// ============================================================================

export function useStepDistributionOverTime(
  runPath: string,
  metricType: string,
  enabled: boolean,
  shouldPoll: boolean
) {
  return useQuery<StepDistributionOverTimeResponse>({
    queryKey: ["step-distribution-over-time", runPath, metricType],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/step-distribution-over-time`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_path: runPath,
          metric_type: metricType,
          num_bins: 10,
        }),
      })
      if (!response.ok) {
        throw new Error("Failed to fetch step distribution over time")
      }
      return response.json()
    },
    enabled: enabled && !!runPath && !!metricType,
    refetchInterval: shouldPoll ? POLL_INTERVAL : false,
    staleTime: 30000, // Cache for 30 seconds
    placeholderData: keepPreviousData,
  })
}

// ============================================================================
// Custom Metrics Layout
// ============================================================================

export function useCustomMetricsLayout() {
  return useQuery<CustomMetricsLayoutResponse>({
    queryKey: ["custom-metrics-layout"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/custom-metrics-layout`)
      if (!response.ok) {
        throw new Error("Failed to fetch custom metrics layout")
      }
      return response.json()
    },
    staleTime: 60000,
  })
}

export function useCustomMetricsTemplates() {
  return useQuery<CustomMetricsTemplatesResponse>({
    queryKey: ["custom-metrics-templates"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/custom-metrics-templates`)
      if (!response.ok) {
        throw new Error("Failed to fetch custom metrics templates")
      }
      return response.json()
    },
    staleTime: 60000,
  })
}

export function useCustomMetricsTemplate(templateId: string | null) {
  return useQuery<CustomMetricsTemplateResponse>({
    queryKey: ["custom-metrics-template", templateId],
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE}/custom-metrics-templates/${templateId}`
      )
      if (!response.ok) {
        throw new Error("Failed to fetch template")
      }
      return response.json()
    },
    enabled: !!templateId,
    staleTime: 60000,
  })
}
