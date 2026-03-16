// Shared constants for telescope visualization

export const API_BASE = import.meta.env.VITE_API_BASE ?? ""
export const POLL_INTERVAL = 5000

// ============================================================================
// Orchestrator Event Colors
// ============================================================================

export const ORCHESTRATOR_EVENT_COLORS: Record<string, string> = {
  inference_processes_start: "#3b82f6", // blue
  trainer_process_start: "#8b5cf6", // purple
  inference_servers_ready: "#22c55e", // green
  trainer_ready: "#10b981", // emerald
  weight_update: "#f59e0b", // amber
  save_batch: "#ef4444", // red
  inference_call: "#06b6d4", // cyan
  wandb_initialized: "#6366f1", // indigo
  training_loop_start: "#ec4899", // pink
  rollouts_group_done: "#4ccf7c", // green
  rollout_discarded_max_async: "#bdbdbd", // light gray
  rollout_discarded_zero_advantage: "#969696", // medium gray
  rollout_paused_max_async: "#ffabab", // light red
  rollout_resumed_max_async: "#3856ff", // dark ble
  checkpoint_save_start: "#328519", // dark green
  checkpoint_save_done: "#328519", // dark green
} as const

// ============================================================================
// Trainer Event Colors
// ============================================================================

export const TRAINER_EVENT_COLORS: Record<string, string> = {
  forward: "#3b82f6", // blue
  forward_backward: "#4f6edb", // blue-leaning purple
  loss: "#d6cd1c", // yellow - parent loss event
  loss_computation: "#d6cd1c", // yellow
  "loss/shift": "#fbbf24", // amber-400
  "loss/log_softmax": "#facc15", // yellow-400
  "loss/policy_loss": "#eab308", // yellow-500
  "loss/reduction": "#ca8a04", // yellow-600
  backward: "#ef4444", // red
  optimizer: "#22c55e", // green
  data_wait: "#6b7280", // gray
  weight_broadcast: "#f97316", // orange
  grad_clip: "#b342f5", // purple
  grad_norm: "#f9a8d4", // light pink (pink-300)
  // Micro batch events
  data_to_device: "#06b6d4", // cyan - data transfer
  prepare_tensors: "#14b8a6", // teal - tensor preparation
  compute_entropy: "#f97316", // orange - entropy computation
  compute_kl: "#ec4899", // pink - KL divergence computation
  finalize_model_grads: "#9041c4", // pink
  checkpoint: "#328519", // dark green
} as const

// ============================================================================
// Inference Event Colors
// ============================================================================

export const INFERENCE_REQUEST_COLOR = "#0284c7" // sky-600 (blue)
export const INFERENCE_REQUEST_EVAL_COLOR = "#10b981" // emerald-500 (green for eval)
export const INFERENCE_REQUEST_DISCARDED_COLOR = "#ababab" // gray for discarded
export const INFERENCE_REQUEST_DISCARDED_COLOR_DARK = "#555555" // darker gray for dark mode
export const INFERENCE_REQUEST_CANCELED_COLOR = "#c9c9c9" // lighter gray for canceled (gray-300)
export const INFERENCE_REQUEST_CANCELED_COLOR_DARK = "#4a4a4a" // darker gray for canceled in dark mode
export const INFERENCE_EVENT_COLORS: Record<string, string> = {
  request: INFERENCE_REQUEST_COLOR,
  weight_broadcast: "#f97316", // orange - same as trainer weight_broadcast
} as const

export const IDLE_COLOR = "#e6e9ed" // light gray
export const IDLE_COLOR_DARK = "#2a2d32" // dark gray
export const DEFAULT_EVENT_COLOR = "#6b7280" // gray

// ============================================================================
// Color Getters
// ============================================================================

/**
 * Get color for orchestrator event type (with prefix matching for numbered events)
 */
export function getOrchestratorEventColor(eventType: string): string {
  if (ORCHESTRATOR_EVENT_COLORS[eventType]) {
    return ORCHESTRATOR_EVENT_COLORS[eventType]
  }
  for (const [key, color] of Object.entries(ORCHESTRATOR_EVENT_COLORS)) {
    if (eventType.startsWith(key)) {
      return color
    }
  }
  return DEFAULT_EVENT_COLOR
}

/**
 * Get color for trainer event type (supports hierarchical events like "loss/shift").
 * Event types are clean operation names (e.g. "forward", "backward").
 * Microbatch index is a separate field, not part of event_type.
 */
export function getTrainerEventColor(eventType: string): string {
  // Direct match
  if (TRAINER_EVENT_COLORS[eventType]) {
    return TRAINER_EVENT_COLORS[eventType]
  }
  // Try parent event color for nested events (e.g., "loss/shift" -> "loss")
  if (eventType.includes("/")) {
    const parentType = eventType.split("/")[0]
    if (TRAINER_EVENT_COLORS[parentType]) {
      return TRAINER_EVENT_COLORS[parentType]
    }
  }
  return DEFAULT_EVENT_COLOR
}

/**
 * Get color for inference event type
 */
export function getInferenceEventColor(eventType: string): string {
  if (INFERENCE_EVENT_COLORS[eventType]) {
    return INFERENCE_EVENT_COLORS[eventType]
  }
  return DEFAULT_EVENT_COLOR
}

/**
 * Get display name for event type (replaces underscores with spaces)
 */
export function getEventDisplayName(eventType: string): string {
  return eventType.replace(/_step_\d+$/, "").replace(/_/g, " ")
}

/**
 * Format trainer event display name for tooltip.
 * Microbatch index is a separate field (not part of event_type).
 * Returns { primary: "Forward", secondary: "(Step 55 Micro Batch 2)" }
 * or { primary: "Optimizer", secondary: "(Step 55)" }
 */
export function formatTrainerEventTitle(
  eventType: string,
  step: number,
  microbatch?: number | null,
  minibatch?: number | null,
): { primary: string; secondary: string | null } {
  const formattedName = eventType
    .replace(/_/g, " ")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ")

  const parts: string[] = []
  if (step !== -1) {
    parts.push(`Step ${step}`)
  }
  if (minibatch != null && minibatch !== -1) {
    parts.push(`Mini Batch ${minibatch}`)
  }
  if (microbatch != null && microbatch !== -1) {
    parts.push(`Micro Batch ${microbatch}`)
  }

  return {
    primary: formattedName,
    secondary: parts.length > 0 ? `(${parts.join(" ")})` : null,
  }
}
