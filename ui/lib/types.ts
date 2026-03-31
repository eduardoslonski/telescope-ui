// Shared TypeScript types for the telescope visualization

// ============================================================================
// Event Types
// ============================================================================

export interface OrchestratorEvent {
  timestamp: number
  event_type: string
  step: number
  node_id?: number | null
  group_id?: number | null
  sample_id?: number | null
}

export interface TrainerEvent {
  event_type: string
  step: number
  rank: number
  local_rank?: number | null
  node_id?: number | null
  gpu_index?: number | null
  microbatch: number | null
  minibatch: number | null
  start_time: number
  end_time: number
  parent: string | null
  depth: number
}

export interface InferenceEvent {
  event_type: string // "request" or "weight_broadcast"
  server: number // Server index (0, 1, ...)
  node_id?: number | null
  tp_group_id?: number | null
  tp_size?: number | null
  start_time: number
  end_time: number
  prompt_tokens?: number // Number of prompt tokens (for requests)
  rollout_tokens?: number // Number of generated tokens (for requests)
  sample_id?: number // Sample ID within the group
  group_id?: number // Group ID for the request
  // vLLM request metrics
  vllm_request_id?: string // e.g. "cmpl-a1b2c3d4", used as the join key
  queue_time?: number // time request waited in vLLM's queue before scheduling
  time_to_first_token?: number // TTFT from vLLM's perspective
  prefill_time?: number // time from scheduling to first token (model prefill)
  decode_time?: number // time from first token to last token
  inference_time?: number // prefill + decode (scheduling to last token)
  e2e_latency?: number // end-to-end latency inside vLLM
  max_tokens?: number // the max_tokens param for this request
  is_eval?: boolean // Whether this is an evaluation inference event
  is_canceled?: boolean // Whether this inference request was canceled
  off_policy_steps?: number | null // Number of weight updates that occurred while this rollout was in-flight
  step?: number | null // Training step associated with this event (e.g. weight_broadcast step)
  // Precomputed lane assignment (server-side, per-server)
  lane?: number | null // Lane index for per-server view
  // Timing data from rollouts/samples_data
  environment_response_time?: number | null // env response time after this turn (seconds)
  compute_reward_time?: number | null // compute reward time after the last turn (seconds, only on last request per sample)
  phase?: string // "start" or "end" (empty for backward compat)
}

export interface InflightGeneration {
  sample_id: number
  group_id: number
  server: number
  server_lane: number
  start_time: number
  is_eval: boolean
  prompt_tokens: number
}

export interface InflightSnapshot {
  snapshot_time: number | null
  running: InflightGeneration[]
  running_compute_reward?: InflightGeneration[]
  running_env_response?: InflightGeneration[]
}

// ============================================================================
// Rollout Types
// ============================================================================

export interface Prompt {
  step: number
  group_id: number
  env: string | null
  system_prompt: string | null
  tokens_system_prompt: number | null
  prompt: string
  tokens_prompt: number | null
}

export interface Rollout {
  step: number
  group_id: number
  sample_idx: number
  turn_order: number
  turn_type: string
  content: string
  tokens: number | null
}

export interface SampleData {
  step: number
  group_id: number
  sample_idx: number
  reward: number | null
  advantage: number | null
  turns: number | null
  total_tokens: number | null
  raw_string: string | null
}

export interface RolloutMetric {
  step: number
  sample_idx: number
  env: string | null
  metric_name: string
  value: number
}

export interface GoldenAnswer {
  step: number
  sample_idx: number
  env: string | null
  key: string
  value: string | null
}

export interface SampleTag {
  step?: number
  sample_idx: number
  env: string | null
  tag_name: string
  tag_value: string
}

export interface InfoTurn {
  step?: number
  sample_idx: number
  turn_order: number
  env: string | null
  info_key: string
  info_value: string
  info_type: string
}

export interface RolloutsDisplaySample {
  sample_idx: number
  group_id: number
  prompt: Prompt | null
  turns: Rollout[]
  reward: number | null
  advantage: number | null
  total_tokens: number | null
  raw_string: string | null
}

// ============================================================================
// Discarded Rollout Types
// ============================================================================

export interface PromptDiscarded {
  timestamp: number
  discard_reason: string
  trainer_step: number
  inference_step: number
  group_id: number
  env: string | null
  system_prompt: string | null
  tokens_system_prompt: number | null
  prompt: string
  tokens_prompt: number | null
}

export interface RolloutDiscarded {
  trainer_step: number
  inference_step: number
  group_id: number
  sample_idx: number
  turn_order: number
  turn_type: string
  content: string
  tokens: number | null
}

export interface SampleDataDiscarded {
  timestamp: number
  discard_reason: string
  trainer_step: number
  inference_step: number
  group_id: number
  sample_idx: number
  reward: number | null
  advantage: number | null
  turns: number | null
  total_tokens: number | null
  raw_string: string | null
}

export interface RolloutMetricDiscarded {
  sample_idx: number
  env: string | null
  metric_name: string
  value: number
  tail_idx?: number | null
}

export interface GoldenAnswerDiscarded {
  sample_idx: number
  env: string | null
  key: string
  value: string | null
  tail_idx?: number | null
}

export interface InfoTurnDiscarded {
  sample_idx: number
  turn_order: number
  env: string | null
  info_key: string
  info_value: string
  info_type: string
  tail_idx?: number | null
}

// ============================================================================
// Eval Types
// ============================================================================

export interface EvalPrompt {
  step: number
  eval_name: string
  model_step: number
  sample_idx: number
  env: string | null
  prompt: string
  tokens_prompt: number | null
  system_prompt: string | null
  tokens_system_prompt: number | null
}

export interface EvalRollout {
  step: number
  eval_name: string
  model_step: number
  sample_idx: number
  completion_idx: number
  turn_order: number
  turn_type: string
  content: string
  tokens: number | null
  stop_reason: string | null
  environment_response_time: number | null
}

export interface EvalSampleData {
  step: number
  eval_name: string
  model_step: number
  sample_idx: number
  completion_idx: number
  env: string | null
  turns: number | null
  compute_eval_metrics_time: number | null
}

export interface EvalRolloutMetric {
  step: number
  eval_name: string
  sample_idx: number
  completion_idx: number
  env: string | null
  metric_name: string
  value: number
}

export interface EvalGoldenAnswer {
  step: number
  eval_name: string
  sample_idx: number
  completion_idx: number
  env: string | null
  key: string
  value: string | null
}

export interface EvalInfoTurn {
  step: number
  eval_name: string
  sample_idx: number
  completion_idx: number
  turn_order: number
  env: string | null
  info_key: string
  info_value: string
  info_type: string
}

// ============================================================================
// API Response Types
// ============================================================================

export interface RolloutsResponse {
  prompts: Prompt[]
  rollouts: Rollout[]
  samples_data: SampleData[]
  rollout_metrics: RolloutMetric[]
  golden_answers: GoldenAnswer[]
  sample_tags: SampleTag[]
  info_turns: InfoTurn[]
  available_steps: number[]
  total_steps: number
  current_step: number | null
  available_rollout_metric_names: string[]
  available_envs: string[]
}

export interface RolloutsDiscardedResponse {
  prompts: PromptDiscarded[]
  rollouts: RolloutDiscarded[]
  samples_data: SampleDataDiscarded[]
  rollout_metrics: RolloutMetricDiscarded[]
  golden_answers: GoldenAnswerDiscarded[]
  sample_tags: SampleTag[]
  info_turns: InfoTurnDiscarded[]
  available_trainer_steps: number[]
  total_trainer_steps: number
  current_trainer_step: number | null
  available_discard_reasons: string[]
  available_envs: string[]
}

export interface EvalsResponse {
  prompts: EvalPrompt[]
  rollouts: EvalRollout[]
  samples_data: EvalSampleData[]
  rollout_metrics: EvalRolloutMetric[]
  golden_answers: EvalGoldenAnswer[]
  sample_tags: SampleTag[]
  info_turns: EvalInfoTurn[]
  available_steps: number[]
  available_eval_names: string[]
  current_step: number | null
  current_eval_name: string | null
  available_rollout_metric_names: string[]
}

export interface SampleDetailsRollouts {
  kind: "rollouts"
  step: number
  group_id: number
  sample_idx: number
  prompts: Prompt[]
  rollouts: Rollout[]
  samples_data: SampleData[]
  rollout_metrics: RolloutMetric[]
  golden_answers: GoldenAnswer[]
  info_turns: InfoTurn[]
}

export interface SampleDetailsDiscarded {
  kind: "rollouts_discarded"
  trainer_step: number
  inference_step: number
  discard_reason: string
  group_id: number
  sample_idx: number
  prompts: PromptDiscarded[]
  rollouts: RolloutDiscarded[]
  samples_data: SampleDataDiscarded[]
  rollout_metrics: RolloutMetricDiscarded[]
  golden_answers: GoldenAnswerDiscarded[]
  info_turns: InfoTurnDiscarded[]
}

export interface SampleDetailsEval {
  kind: "eval"
  step: number
  eval_name: string
  model_step: number
  group_id: number
  sample_idx: number
  prompts: Prompt[]
  rollouts: Rollout[]
  samples_data: SampleData[]
  rollout_metrics: RolloutMetric[]
  golden_answers: GoldenAnswer[]
  info_turns: InfoTurn[]
}

export interface SampleDetailsNotFound {
  kind: null
  group_id: number
  sample_idx: number
  prompts: []
  rollouts: []
  samples_data: []
  rollout_metrics: []
  golden_answers: []
  info_turns: []
}

export type SampleDetailsResponse =
  | SampleDetailsRollouts
  | SampleDetailsDiscarded
  | SampleDetailsEval
  | SampleDetailsNotFound

export interface SampleStatusItem {
  group_id: number
  sample_idx: number
  kind: "rollouts" | "rollouts_discarded" | null
}

export interface SampleStatusesResponse {
  statuses: SampleStatusItem[]
}

export interface EnvironmentResponseTime {
  sample_idx: number
  turn_order: number
  time: number
}

export interface ComputeRewardTime {
  sample_idx: number
  time: number
}

export interface InferenceGroupEventsResponse {
  events: InferenceEvent[]
  environment_response_times?: EnvironmentResponseTime[]
  compute_reward_times?: ComputeRewardTime[]
}

export interface TrainerBreakdownEventsResponse {
  events: TrainerEvent[]
}

export interface TimelinePaginatedResponse {
  orchestrator_events: OrchestratorEvent[]
  trainer_events: TrainerEvent[]
  inference_events: InferenceEvent[]
  total_pages: number
  current_page: number
  interval_start: number
  interval_end: number
  global_min_time: number | null
  global_max_time: number | null
}

export interface SyncStatus {
  status: string
  rollouts_fetched?: number
  events_fetched?: number
  error?: string
}

export interface RunSummary {
  summary: Record<string, unknown>
  config: Record<string, unknown>
  custom_config: Record<string, unknown> | null
  last_rollout_step: number
  local_rollout_count: number
  local_rollout_steps: number
  local_rollout_metrics_count: number
  available_rollout_metric_names: string[]
  available_envs: string[]
  local_orchestrator_event_count: number
  local_trainer_event_count: number
  local_inference_event_count: number
  local_gpu_metrics_count: number
  local_cpu_metrics_count: number
  local_vllm_metrics_count: number
  local_discarded_rollout_count: number
  local_discarded_rollout_metrics_count: number
  local_discarded_trainer_steps: number
  trainer_info: {
    last_training_step: number | null
  }
  event_info: {
    last_event_block_idx: number
    remote_num_finalized_blocks: number
    remote_current_block_idx: number
    tail_orchestrator_count: number
    tail_trainer_count: number
  }
  rollout_info: {
    remote_num_finalized_blocks: number
    remote_current_block_idx: number
    last_training_step: number
    block_live_rollout_count?: number
  }
  step_metrics_info: {
    local_steps: number
    custom_metric_sections: Record<string, Record<string, string[]>>
  }
  eval_info: {
    evals: Array<{
      eval_name: string
      available_rollout_metric_names: string[]
    }>
  }
  trainer_bucket_info: {
    step: number | null
    groups_done: number
  }
  waiting_buckets: number[]
  /** Per-(env, metric_name) min/max computed from actual rollouts_metrics data */
  data_metric_ranges: Record<string, Record<string, { min: number; max: number }>>
  available_sample_tags: Record<string, string[]>
  is_tracking: boolean
  is_syncing: boolean
  sync_status: SyncStatus | null
}

export interface RunCodeTreeNode {
  name: string
  path: string
  type: "directory" | "file"
  hash?: string | null
  children?: RunCodeTreeNode[]
}

export interface RunCodeTreeResponse {
  run_path: string
  available: boolean
  tree: RunCodeTreeNode[]
  truncated: boolean
  total_nodes: number
}

export interface RunCodeFileResponse {
  run_path: string
  file_path: string
  size_bytes: number
  truncated: boolean
  is_binary: boolean
  content: string
}

export interface RunCodeDiffSummaryResponse {
  left_run_path: string
  right_run_path: string
  available: boolean
  changed_files: number
  added_lines: number
  removed_lines: number
}

// ============================================================================
// Logs Types
// ============================================================================

export interface LogEntry {
  timestamp: number
  level: string
  component: string
  source: string
  message: string
}

export interface LogsResponse {
  logs: LogEntry[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface LogsSummaryResponse {
  components: string[]
  levels: string[]
  sources: string[]
  total_count: number
}

// ============================================================================
// System Metrics Types
// ============================================================================

export interface GpuMetric {
  timestamp: number
  node_id?: number | null
  rank?: number | null
  local_rank?: number | null
  gpu_index: number
  source?: string | null
  metric_name: string
  value: number
}

export interface CpuMetric {
  timestamp: number
  node_id?: number | null
  source?: string | null
  metric_name: string
  value: number
}

export interface VllmMetric {
  timestamp: number
  server: number
  node_id?: number | null
  tp_group_id?: number | null
  tp_size?: number | null
  metric_name: string
  value: number
}


// ============================================================================
// Step Metrics Types (Per-step training metrics)
// ============================================================================

export interface StepMetric {
  step: number
  metric_name: string
  value: number
}

export interface StepMetricsResponse {
  metrics: StepMetric[]
  total_returned: number
  available_metrics: string[]
  available_rollout_metric_names: string[]
  available_sample_tags: Record<string, string[]>
  available_custom_metrics: string[]
  custom_metric_sections: Record<string, Record<string, string[]>>
  min_step: number | null
  max_step: number | null
  total_steps: number | null
}

export interface StepMetricsRunResponse extends StepMetricsResponse {
  run_path: string
}

export interface StepMetricsMultiRunResponse {
  runs: StepMetricsRunResponse[]
}

export interface StepTime {
  step: number
  time: number
}

export interface StepTimesResponse {
  step_times: StepTime[]
  first_step_time: number | null
}

// ============================================================================
// Inference Performance Types
// ============================================================================

export interface InferencePerformanceBucket {
  time: number
  count: number
}

export interface InferencePerformanceAvgBucket {
  time: number
  value: number
}

export interface InferenceUtilizationBucket {
  time: number
  idle: number
  working: number
  generating: number
  weight_broadcast: number
  [key: string]: number
}

export interface InferencePerformanceResponse {
  inference_calls: InferencePerformanceBucket[]
  requests_done: InferencePerformanceBucket[]
  rollouts_group_done: InferencePerformanceBucket[]
  rollouts_group_done_kept: InferencePerformanceBucket[]
  rollouts_group_done_discarded: InferencePerformanceBucket[]
  rollouts_group_done_canceled: InferencePerformanceBucket[]
  avg_time_prefill: InferencePerformanceAvgBucket[]
  avg_time_decode: InferencePerformanceAvgBucket[]
  avg_time_compute_reward: InferencePerformanceAvgBucket[]
  avg_time_queue: InferencePerformanceAvgBucket[]
  avg_time_ttft: InferencePerformanceAvgBucket[]
  avg_time_inference: InferencePerformanceAvgBucket[]
  avg_time_e2e: InferencePerformanceAvgBucket[]
  avg_time_generation: InferencePerformanceAvgBucket[]
  avg_tokens_per_second_generation: InferencePerformanceAvgBucket[]
  tokens_per_second_throughput: InferencePerformanceAvgBucket[]
  vllm_requests_running_avg: InferencePerformanceAvgBucket[]
  vllm_requests_waiting_avg: InferencePerformanceAvgBucket[]
  vllm_preemptions: InferencePerformanceAvgBucket[]
  vllm_preemptions_per_request: InferencePerformanceAvgBucket[]
  vllm_ttft_avg: InferencePerformanceAvgBucket[]
  utilization_buckets: InferenceUtilizationBucket[]
  num_lanes: number
  step_times: StepTime[]
  first_time: number | null
  last_time: number | null
}

// ============================================================================
// Trainer Performance Types
// ============================================================================

export interface TrainerPerformanceBucket {
  time: number
  idle: number
  working: number
  working_except_weight_sync: number
  [event_type: string]: number
}

export interface TrainerPerformanceResponse {
  buckets: TrainerPerformanceBucket[]
  event_types: string[]
  first_time: number | null
  last_time: number | null
  step_times: StepTime[]
}

// ============================================================================
// Step Histogram Types
// ============================================================================

export interface StepHistogramResponse {
  values: number[]
  step: number
  metric_type: string
  count: number
}

export interface StepDistributionOverTimeResponse {
  steps: number[]
  bin_edges: number[]
  counts: number[][]  // 2D array: [step_idx][bin_idx]
  global_min: number | null
  global_max: number | null
}


// ============================================================================
// Run Types
// ============================================================================

export interface Run {
  run_id: string
  name: string | null
  created_at: string | null
  state: string | null
  entity: string | null
  project: string | null
  url: string | null
  last_rollout_step: number
  is_tracking: boolean
  is_syncing: boolean
  is_drained: boolean
  needs_update: boolean
  color: string
  notes: string | null
}

export interface DiscoveryStatus {
  status: "idle" | "discovering"
  runs_found?: number
  projects_scanned?: number
  total_projects?: number
}

export interface RunsResponse {
  runs: Run[]
  discovery: DiscoveryStatus
  has_wandb_key?: boolean
  wandb_key_source?: "netrc" | "custom" | "unconfigured"
  has_netrc_wandb_key?: boolean
  has_known_projects?: boolean
}

export interface RemovedRun {
  run_id: string
  name: string | null
  created_at: string | null
  state: string | null
  entity: string | null
  project: string | null
  url: string | null
  color: string | null
  removed_at: string | null
  config: Record<string, unknown> | null
}

export interface RemovedRunsResponse {
  runs: RemovedRun[]
}


// ============================================================================
// Custom Metrics Layout Types
// ============================================================================

export interface CustomPlotItem {
  id: string
  metricKey: string
  label: string
  plotType: "step_metric" | "eval_metric" | "distribution_over_time" | "histogram" | "inference_performance" | "inference_performance_area" | "inference_utilization_area" | "trainer_performance" | "trainer_performance_area"
  evalName?: string
  distMetricType?: string
  inferenceMetricType?: string
  inferenceAreaCategories?: string[]
  trainerMetricType?: string
  trainerAreaCategories?: string[]
}

export interface CustomGroup {
  id: string
  name: string
  plots: CustomPlotItem[]
}

export interface CustomSection {
  id: string
  name: string
  groups: CustomGroup[]
  plots: CustomPlotItem[]
}

export interface CustomMetricsLayout {
  sections: CustomSection[]
}

export interface CustomMetricsLayoutResponse {
  layout: CustomMetricsLayout | null
}

export interface CustomMetricsTemplateSummary {
  id: string
  name: string
  updated_at: string | null
}

export interface CustomMetricsTemplatesResponse {
  templates: CustomMetricsTemplateSummary[]
}

export interface CustomMetricsTemplateResponse {
  id: string
  name: string
  layout: CustomMetricsLayout
  updated_at: string | null
}
