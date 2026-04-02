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

export interface RolloutEvent {
  event_type: string // "generation", "tool_execution", "env_response", "reward", "eval_metrics"
  start_time: number
  end_time: number
  sample_id?: number | null
  group_id?: number | null
  agent_id?: number // 0 = main agent
  generation_idx?: number | null
  tool_call_idx?: number | null
  server_id?: number | null
  server_lane?: number | null // Per-server lane slot for timeline positioning
  // vLLM timing (from generations table, only for generation events)
  queue_time?: number | null
  time_to_first_token?: number | null
  prefill_time?: number | null
  decode_time?: number | null
  inference_time?: number | null
  e2e_latency?: number | null
  rollout_tokens?: number | null
  prompt_tokens?: number | null
  off_policy_steps?: number | null
}

export interface InfraEvent {
  timestamp: number
  event_type: string // "weight_sync", "sandbox"
  phase: string
  step?: number | null
  server_id?: number | null
  sandbox_id?: string | null
}

export interface InflightSnapshot {
  timestamp: number | null
  inflight_generations: { sample_id: number; generation_idx: number; server_id: number; server_lane: number; group_id: number; agent_id: number; start_time: number }[]
  inflight_tool_executions: { sample_id: number; generation_idx: number; tool_call_idx: number; tool_name: string; agent_id: number }[]
  inflight_env_responses: { sample_id: number; generation_idx: number; agent_id: number }[]
  inflight_sandbox_ops: { sandbox_id: string; phase: string }[]
  inflight_weight_syncs: { server_id: number; step: number }[]
  inflight_rewards: { sample_id: number; agent_id: number }[]
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

export interface GenerationRow {
  step: number
  group_id: number
  sample_id: number
  agent_id: number
  generation_idx: number
  content: string
  tokens: number | null
  prompt_tokens: number | null
  tool_call_count: number | null
  stop_reason: string | null
  queue_time: number | null
  ttft: number | null
  prefill_time: number | null
  decode_time: number | null
  inference_time: number | null
  e2e_latency: number | null
  server_id: number | null
  vllm_request_id: string | null
}

export interface EnvResponseRow {
  step: number
  group_id: number
  sample_id: number
  agent_id: number
  generation_idx: number
  content: string
  turn_type: string
  tokens: number | null
  response_time: number | null
}

export interface ToolCallRow {
  step: number
  group_id: number
  sample_id: number
  agent_id: number
  generation_idx: number
  tool_call_idx: number
  env_response_generation_idx: number
  tool_name: string
  arguments: string
  raw_text: string | null
  result: string | null
  success: boolean
  error: string | null
  exit_code: number | null
  truncated: boolean
  result_tokens: number | null
  sandbox_id: string | null
}

export interface SampleData {
  step: number
  group_id: number
  sample_id: number
  reward: number | null
  advantage: number | null
  num_generations: number | null
  total_tokens: number | null
  raw_string: string | null
  compute_reward_time: number | null
  stop_reason: string | null
}

export interface RolloutMetric {
  step: number
  sample_id: number
  env: string | null
  metric_name: string
  value: number
}

export interface GoldenAnswer {
  step: number
  sample_id: number
  env: string | null
  key: string
  value: string | null
}

export interface SampleTag {
  step?: number
  sample_id: number
  env: string | null
  tag_name: string
  tag_value: string
}

export interface InfoTurn {
  step?: number
  sample_id: number
  agent_id: number
  generation_idx: number
  tool_call_idx: number | null
  env: string | null
  info_key: string
  info_value: string
  info_type: string
}

export interface RolloutsDisplaySample {
  sample_id: number
  group_id: number
  prompt: Prompt | null
  generations: GenerationRow[]
  env_responses: EnvResponseRow[]
  tool_calls: ToolCallRow[]
  reward: number | null
  advantage: number | null
  total_tokens: number | null
  raw_string: string | null
  stop_reason: string | null
  num_generations: number | null
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

export interface GenerationDiscarded {
  trainer_step: number
  inference_step: number
  group_id: number
  sample_id: number
  agent_id: number
  generation_idx: number
  content: string
  tokens: number | null
  prompt_tokens: number | null
  tool_call_count: number | null
  stop_reason: string | null
}

export interface EnvResponseDiscarded {
  trainer_step: number
  inference_step: number
  group_id: number
  sample_id: number
  agent_id: number
  generation_idx: number
  content: string
  turn_type: string
  tokens: number | null
  response_time: number | null
}

export interface ToolCallDiscarded {
  trainer_step: number
  inference_step: number
  group_id: number
  sample_id: number
  agent_id: number
  generation_idx: number
  tool_call_idx: number
  env_response_generation_idx: number
  tool_name: string
  arguments: string
  result: string | null
  success: boolean
  error: string | null
}

export interface SampleDataDiscarded {
  timestamp: number
  discard_reason: string
  trainer_step: number
  inference_step: number
  group_id: number
  sample_id: number
  reward: number | null
  advantage: number | null
  num_generations: number | null
  total_tokens: number | null
  raw_string: string | null
  stop_reason: string | null
}

export interface RolloutMetricDiscarded {
  sample_id: number
  env: string | null
  metric_name: string
  value: number
  tail_idx?: number | null
}

export interface GoldenAnswerDiscarded {
  sample_id: number
  env: string | null
  key: string
  value: string | null
  tail_idx?: number | null
}

export interface InfoTurnDiscarded {
  sample_id: number
  agent_id: number
  generation_idx: number
  tool_call_idx: number | null
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

export interface EvalGeneration {
  step: number
  eval_name: string
  model_step: number
  sample_idx: number
  completion_idx: number
  agent_id: number
  generation_idx: number
  content: string
  tokens: number | null
  prompt_tokens: number | null
  tool_call_count: number | null
  stop_reason: string | null
}

export interface EvalEnvResponse {
  step: number
  eval_name: string
  model_step: number
  sample_idx: number
  completion_idx: number
  agent_id: number
  generation_idx: number
  content: string
  turn_type: string
  tokens: number | null
  response_time: number | null
}

export interface EvalToolCall {
  step: number
  eval_name: string
  model_step: number
  sample_idx: number
  completion_idx: number
  agent_id: number
  generation_idx: number
  tool_call_idx: number
  env_response_generation_idx: number
  tool_name: string
  arguments: string
  raw_text: string | null
  result: string | null
  success: boolean
  error: string | null
  exit_code: number | null
  truncated: boolean
  result_tokens: number | null
  sandbox_id: string | null
}

export interface EvalSampleData {
  step: number
  eval_name: string
  model_step: number
  sample_idx: number
  completion_idx: number
  env: string | null
  num_generations: number | null
  compute_eval_metrics_time: number | null
  stop_reason: string | null
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
  agent_id: number
  generation_idx: number
  tool_call_idx: number | null
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
  generations: GenerationRow[]
  env_responses: EnvResponseRow[]
  tool_calls: ToolCallRow[]
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
  generations: GenerationDiscarded[]
  env_responses: EnvResponseDiscarded[]
  tool_calls: ToolCallDiscarded[]
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
  generations: EvalGeneration[]
  env_responses: EvalEnvResponse[]
  tool_calls: EvalToolCall[]
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
  sample_id: number
  prompts: Prompt[]
  generations: GenerationRow[]
  env_responses: EnvResponseRow[]
  tool_calls: ToolCallRow[]
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
  sample_id: number
  prompts: PromptDiscarded[]
  generations: GenerationDiscarded[]
  env_responses: EnvResponseDiscarded[]
  tool_calls: ToolCallDiscarded[]
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
  sample_id: number
  prompts: Prompt[]
  generations: GenerationRow[]
  env_responses: EnvResponseRow[]
  tool_calls: ToolCallRow[]
  samples_data: SampleData[]
  rollout_metrics: RolloutMetric[]
  golden_answers: GoldenAnswer[]
  info_turns: InfoTurn[]
}

export interface SampleDetailsNotFound {
  kind: null
  group_id: number
  sample_id: number
  prompts: []
  generations: []
  env_responses: []
  tool_calls: []
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
  sample_id: number
  kind: "rollouts" | "rollouts_discarded" | "rollouts_cancelled" | "rollouts_eval" | null
}

export interface SampleStatusesResponse {
  statuses: SampleStatusItem[]
}

export interface TrainerBreakdownEventsResponse {
  events: TrainerEvent[]
}

export interface TimelinePaginatedResponse {
  orchestrator_events: OrchestratorEvent[]
  trainer_events: TrainerEvent[]
  rollout_events: RolloutEvent[]
  infra_events: InfraEvent[]
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
  local_generation_count: number
  local_generation_steps: number
  local_rollout_metrics_count: number
  available_rollout_metric_names: string[]
  available_envs: string[]
  local_orchestrator_event_count: number
  local_trainer_event_count: number
  local_rollout_event_count: number
  local_gpu_metrics_count: number
  local_cpu_metrics_count: number
  local_vllm_metrics_count: number
  local_discarded_generation_count: number
  local_discarded_generation_metrics_count: number
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
