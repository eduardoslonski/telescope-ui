
import { useMemo, useState } from "react"
import {
  ChevronDown,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Spinner } from "@/components/ui/spinner"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { formatTimeAgo } from "@/lib/format"
import { MiddleEllipsisLabel } from "@/components/middle-ellipsis-label"
import { useRuns, useRunSummary } from "@/hooks/use-run-data"
import type { Run } from "@/lib/types"

// Type for configuration values
type ConfigValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | Record<string, unknown>
  | Array<unknown>

interface ConfigItem {
  key: string
  displayKey: string
  value: ConfigValue
}

interface ConfigCategory {
  key: string
  label: string
  items: ConfigItem[]
  subcategories?: ConfigCategory[]
}

function stableStringify(value: unknown): string {
  try {
    const json = JSON.stringify(value, (_key, val) => {
      if (val && typeof val === "object" && !Array.isArray(val)) {
        return Object.keys(val as Record<string, unknown>)
          .sort()
          .reduce((acc, key) => {
            acc[key] = (val as Record<string, unknown>)[key]
            return acc
          }, {} as Record<string, unknown>)
      }
      return val
    })
    return json ?? String(value)
  } catch {
    return String(value)
  }
}

// Extract the actual value from a wandb config structure
function extractValue(value: unknown): ConfigValue {
  if (value === null || value === undefined) return null
  const rawValue =
    typeof value === "object" && value !== null && "value" in value
      ? (value as { value: unknown }).value
      : value

  if (
    typeof rawValue === "string" ||
    typeof rawValue === "number" ||
    typeof rawValue === "boolean"
  ) {
    return rawValue
  }
  if (Array.isArray(rawValue)) return rawValue
  if (typeof rawValue === "object" && rawValue !== null) {
    return rawValue as Record<string, unknown>
  }
  return null
}

// Format a value for display
function formatValue(value: ConfigValue): string {
  if (value === null || value === undefined) return "—"
  if (typeof value === "boolean") return value ? "True" : "False"
  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return value.toLocaleString()
    }
    if (Math.abs(value) < 0.001 && value !== 0) {
      return value.toExponential(2)
    }
    return value.toLocaleString(undefined, { maximumFractionDigits: 6 })
  }
  if (typeof value === "object") {
    return stableStringify(value)
  }
  return String(value)
}

// Get a nice display key from a full key path
function getDisplayKey(key: string): string {
  const parts = key.split("/")
  const lastPart = parts[parts.length - 1]
  return lastPart.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

// Check if two values are equal
function valuesEqual(a: ConfigValue, b: ConfigValue): boolean {
  if (a === b) return true
  if (a === null || a === undefined) return b === null || b === undefined
  if (b === null || b === undefined) return false
  const aIsObject = typeof a === "object"
  const bIsObject = typeof b === "object"
  if (aIsObject || bIsObject) {
    return stableStringify(a) === stableStringify(b)
  }
  return String(a) === String(b)
}

// Categorize configuration keys (same order as run-config-panel.tsx)
function categorizeConfigs(
  config: Record<string, unknown>
): ConfigCategory[] {
  const categoryDefinitions: Array<{
    key: string
    label: string
    match: (key: string) => boolean
    order?: string[]
  }> = [
    {
      key: "general",
      label: "General",
      match: (k) => k === "debug",
      order: ["debug"],
    },
    {
      key: "model",
      label: "Model",
      match: (k) =>
        k === "model" || k === "model_dtype" || k === "mixed_precision_dtype",
      order: ["model", "model_dtype", "mixed_precision_dtype"],
    },
    {
      key: "ray_cluster",
      label: "Ray Cluster",
      match: (k) =>
        k.startsWith("ray_") && k !== "ray_torch_memory_drain_interval_seconds",
      order: [
        "ray_address", "ray_auto_start_local", "ray_namespace",
        "ray_log_to_driver", "ray_disable_runtime_env_hook",
        "ray_pin_py_executable", "ray_propagate_active_venv",
        "ray_propagate_run_dir", "ray_broadcast_init_timeout_s",
        "ray_broadcast_prefer_loopback_if_single_node",
        "ray_shutdown_on_exit",
        "ray_inference_cpus_per_worker", "ray_trainer_cpus_per_worker",
        "ray_inference_placement_strategy",
        "ray_trainer_placement_strategy", "ray_placement_timeout_s",
      ],
    },
    {
      key: "workers",
      label: "Workers",
      match: (k) =>
        [
          "inference_num_workers",
          "inference_tensor_parallel_size",
          "trainer_num_workers",
          "trainer_devices",
        ].includes(k),
      order: [
        "inference_num_workers", "inference_tensor_parallel_size",
        "trainer_num_workers", "trainer_devices",
      ],
    },
    {
      key: "orchestrator",
      label: "Orchestrator",
      match: (k) =>
        [
          "max_concurrent_prompts_per_server",
          "prompts_batch_size_for_trainer",
          "number_of_steps",
          "max_async_rollout",
          "max_off_policy_steps",
          "discard_group_zero_advantage",
          "enable_prompt_prefetch",
          "prompt_prefetch_buffer_size",
          "enable_individual_sample_lanes",
        ].includes(k),
      order: [
        "max_concurrent_prompts_per_server", "prompts_batch_size_for_trainer",
        "number_of_steps", "max_async_rollout", "max_off_policy_steps",
        "discard_group_zero_advantage", "enable_prompt_prefetch",
        "prompt_prefetch_buffer_size", "enable_individual_sample_lanes",
      ],
    },
    {
      key: "trainer",
      label: "Trainer",
      match: (k) =>
        [
          "learning_rate",
          "weight_decay",
          "warmup_steps",
          "grad_clip",
          "train_backend",
          "fsdp_activation_checkpointing",
          "jit",
        ].includes(k) || k.startsWith("megatron_"),
      order: [
        "learning_rate", "weight_decay", "warmup_steps", "grad_clip",
        "train_backend",
        "fsdp_activation_checkpointing", "jit",
        "megatron_tensor_parallel_size", "megatron_pipeline_parallel_size",
        "megatron_context_parallel_size", "megatron_expert_parallel_size",
        "megatron_disable_unified_memory_jit",
        "megatron_optimizer_cpu_offload", "megatron_optimizer_offload_fraction",
        "megatron_overlap_cpu_optimizer_d2h_h2d",
        "megatron_use_precision_aware_optimizer",
        "megatron_main_grads_dtype", "megatron_main_params_dtype",
        "megatron_exp_avg_dtype", "megatron_exp_avg_sq_dtype",
      ],
    },
    {
      key: "algorithm",
      label: "Algorithm",
      match: (k) =>
        [
          "algorithm",
          "number_of_minibatches",
          "use_ppo_clip",
          "ppo_clip_ref_logprobs",
          "clip_low",
          "clip_high",
          "sapo_tau_pos",
          "sapo_tau_neg",
          "advantage_norm",
          "use_tis",
          "tis_cap",
          "tis_logprob_clamp",
          "entropy_chunk_size",
        ].includes(k),
      order: [
        "algorithm", "number_of_minibatches", "use_ppo_clip",
        "ppo_clip_ref_logprobs", "clip_low", "clip_high",
        "sapo_tau_pos", "sapo_tau_neg",
        "advantage_norm", "use_tis", "tis_cap", "tis_logprob_clamp",
        "entropy_chunk_size",
      ],
    },
    {
      key: "weight_sync",
      label: "Weight Sync",
      match: (k) => k.startsWith("weight_broadcast_"),
      order: ["weight_broadcast_mode", "weight_broadcast_bucket_mb", "weight_broadcast_cpu_staging"],
    },
    {
      key: "sequence_packing",
      label: "Sequence Packing",
      match: (k) => k === "seq_len" || k === "pad_to_multiple_of",
      order: ["seq_len", "pad_to_multiple_of"],
    },
    {
      key: "inference",
      label: "Inference",
      match: (k) =>
        [
          "inference_host",
          "inference_devices",
          "inference_base_port",
          "num_inference_servers",
          "gpu_memory_utilization",
          "max_model_len",
          "vllm_scheduling_policy",
          "enable_thinking",
          "chat_template",
          "reasoning_parser",
          "enable_tool_call",
          "tool_call_parser",
        ].includes(k),
      order: [
        "inference_host", "inference_devices", "inference_base_port",
        "num_inference_servers", "gpu_memory_utilization", "max_model_len",
        "vllm_scheduling_policy", "enable_thinking", "chat_template", "reasoning_parser",
        "enable_tool_call", "tool_call_parser",
      ],
    },
    {
      key: "rollout",
      label: "Rollout / Sampling",
      match: (k) =>
        [
          "group_size",
          "temperature",
          "max_tokens",
          "top_p",
          "interleaved_rollouts",
        ].includes(k),
      order: ["group_size", "temperature", "max_tokens", "top_p", "interleaved_rollouts"],
    },
    {
      key: "checkpoint",
      label: "Checkpoint",
      match: (k) =>
        [
          "checkpoint_every",
          "checkpoint_save_training_state",
          "resume_from_checkpoint",
          "checkpoint_dir",
          "checkpoint_keep_last",
          "checkpoint_keep_every",
        ].includes(k),
      order: [
        "checkpoint_every", "checkpoint_save_training_state", "resume_from_checkpoint",
        "checkpoint_dir", "checkpoint_keep_last", "checkpoint_keep_every",
      ],
    },
    {
      key: "logging",
      label: "Logging",
      match: (k) =>
        k === "use_wandb" ||
        k.startsWith("wandb_") ||
        k === "_wandb" ||
        k.startsWith("event_") ||
        k.endsWith("_interval_seconds") ||
        k === "rollout_block_size" ||
        k === "track_gpu_events",
      order: [
        "use_wandb", "wandb_project", "wandb_run_name", "wandb_tags",
        "wandb_upload_code", "wandb_code_max_file_size_mb",
        "wandb_code_exclude_patterns",
        "system_metrics_collection_interval_seconds",
        "torch_memory_sample_interval_seconds",
        "event_tail_window_seconds", "event_block_duration_seconds",
        "event_upload_interval_seconds",
        "metrics_logger_interval_seconds",
        "ray_torch_memory_drain_interval_seconds",
        "rollout_block_size", "track_gpu_events",
      ],
    },
    {
      key: "evals",
      label: "Evals",
      match: (k) => k.startsWith("eval_"),
      order: [
        "eval_before_training", "eval_after_training",
        "eval_num_servers", "eval_start_end_use_all_servers",
      ],
    },
    {
      key: "vllm_tracing",
      label: "vLLM Tracing",
      match: (k) =>
        k === "enable_vllm_tracing" || k === "otlp_receiver_port",
      order: ["enable_vllm_tracing", "otlp_receiver_port"],
    },
    {
      key: "environments",
      label: "Environments",
      match: (k) => k === "environments" || k === "evals",
    },
    // Hardware setup categories
    {
      key: "setup/gpu",
      label: "GPU",
      match: (k) => k.startsWith("setup/gpu/"),
    },
    {
      key: "setup/cpu",
      label: "CPU",
      match: (k) => k.startsWith("setup/cpu/"),
    },
    {
      key: "setup/memory",
      label: "Memory",
      match: (k) => k.startsWith("setup/memory/"),
    },
    {
      key: "setup/disk",
      label: "Disk",
      match: (k) => k.startsWith("setup/disk/"),
    },
    {
      key: "setup/network",
      label: "Network",
      match: (k) => k.startsWith("setup/network/"),
    },
    {
      key: "setup/os",
      label: "Operating System",
      match: (k) => k.startsWith("setup/os/"),
    },
    {
      key: "setup/package_versions",
      label: "Package Versions",
      match: (k) => k.startsWith("setup/package_versions/"),
    },
  ]

  // Build a map for accumulation, preserving definition order
  const categoryMap = new Map<string, ConfigCategory>()
  for (const def of categoryDefinitions) {
    categoryMap.set(def.key, {
      key: def.key,
      label: def.label,
      items: [],
    })
  }

  // Process each config entry in original order
  for (const [key, rawValue] of Object.entries(config)) {
    if (key === "_wandb") continue

    const value = extractValue(rawValue)
    if (value === null) continue

    let matched = false
    for (const def of categoryDefinitions) {
      if (def.match(key)) {
        categoryMap.get(def.key)!.items.push({
          key,
          displayKey: getDisplayKey(key),
          value,
        })
        matched = true
        break
      }
    }

    if (!matched) {
      if (!categoryMap.has("other")) {
        categoryMap.set("other", {
          key: "other",
          label: "Other",
          items: [],
        })
      }
      categoryMap.get("other")!.items.push({
        key,
        displayKey: getDisplayKey(key),
        value,
      })
    }
  }

  // Sort items within each category according to defined order
  const defByKey = new Map(categoryDefinitions.map((d) => [d.key, d]))
  for (const [, cat] of categoryMap) {
    const def = defByKey.get(cat.key)
    if (def?.order && cat.items.length > 1) {
      const orderMap = new Map(def.order.map((k, i) => [k, i]))
      cat.items.sort((a, b) => {
        const ai = orderMap.get(a.key) ?? Infinity
        const bi = orderMap.get(b.key) ?? Infinity
        if (ai !== bi) return ai - bi
        return a.key.localeCompare(b.key)
      })
    }
  }

  // Group hardware setup categories under one parent, preserve order, remove empties
  const hardwareKeys = new Set([
    "setup/gpu",
    "setup/cpu",
    "setup/memory",
    "setup/disk",
    "setup/network",
    "setup/os",
    "setup/package_versions",
  ])

  const finalOrder = [
    "general",
    "model",
    "ray_cluster",
    "workers",
    "orchestrator",
    "trainer",
    "algorithm",
    "weight_sync",
    "sequence_packing",
    "inference",
    "rollout",
    "checkpoint",
    "logging",
    "evals",
    "vllm_tracing",
    "environments",
    "hardware",
    "other",
  ]

  const hardwareSubcategories: ConfigCategory[] = []
  for (const hk of hardwareKeys) {
    const cat = categoryMap.get(hk)
    if (cat && cat.items.length > 0) {
      hardwareSubcategories.push(cat)
    }
  }

  const result: ConfigCategory[] = []
  for (const key of finalOrder) {
    if (key === "hardware") {
      if (hardwareSubcategories.length > 0) {
        result.push({
          key: "hardware",
          label: "Hardware & Environment",
          items: [],
          subcategories: hardwareSubcategories,
        })
      }
    } else {
      const cat = categoryMap.get(key)
      if (cat && cat.items.length > 0) {
        result.push(cat)
      }
    }
  }

  return result
}

// Build a map of key -> value for quick lookup
function buildConfigMap(
  config: Record<string, unknown>
): Map<string, ConfigValue> {
  const map = new Map<string, ConfigValue>()
  for (const [key, rawValue] of Object.entries(config)) {
    if (key === "_wandb") continue
    const value = extractValue(rawValue)
    if (value !== null) {
      map.set(key, value)
    }
  }
  return map
}

// Check if a category has any differences
function categoryHasDifferences(
  category: ConfigCategory,
  config1Map: Map<string, ConfigValue>,
  config2Map: Map<string, ConfigValue>
): boolean {
  if (category.subcategories) {
    return category.subcategories.some((sub) =>
      categoryHasDifferences(sub, config1Map, config2Map)
    )
  }
  return category.items.some((item) => {
    const val1 = config1Map.get(item.key)
    const val2 = config2Map.get(item.key)
    return !valuesEqual(val1, val2)
  })
}

// Run selector component
function RunSelector({
  runs,
  currentRunId,
  onSelect,
}: {
  runs: Run[]
  currentRunId: string
  onSelect: (runId: string) => void
}) {
  return (
    <div className="flex flex-col gap-1">
      {runs.map((run) => {
        const isCurrentRun = run.run_id === currentRunId
        const displayName = run.name || run.run_id.split("/").pop() || run.run_id
        return (
          <button
            key={run.run_id}
            onClick={() => !isCurrentRun && onSelect(run.run_id)}
            disabled={isCurrentRun}
            className={cn(
              "flex w-full items-center gap-2 rounded-md px-3 py-2 text-left transition-colors",
              isCurrentRun
                ? "opacity-40 cursor-not-allowed"
                : "hover:bg-muted cursor-pointer"
            )}
          >
            <div
              className="w-2.5 h-2.5 rounded-full shrink-0"
              style={{ backgroundColor: run.color }}
            />
            <div className="flex-1 min-w-0">
              <div className="min-w-0">
                <MiddleEllipsisLabel text={displayName} className="text-sm font-medium" />
              </div>
              <div className="truncate text-xs text-muted-foreground">{run.run_id}</div>
            </div>
            {isCurrentRun && (
              <Badge variant="secondary" className="h-4 px-1.5 text-[10px] shrink-0">
                Current
              </Badge>
            )}
            {run.is_tracking && (
              <span
                className="h-2 w-2 rounded-full bg-green-500 shrink-0"
                title="Live"
              />
            )}
            {run.is_syncing && (
              <span
                className="h-2 w-2 rounded-full bg-blue-500 shrink-0"
                title="Syncing"
              />
            )}
            {run.created_at && (
              <span className="text-[10px] text-muted-foreground shrink-0">
                {formatTimeAgo(run.created_at)}
              </span>
            )}
          </button>
        )
      })}
      {runs.length === 0 && (
        <div className="text-sm text-muted-foreground text-center py-4">
          No other runs available
        </div>
      )}
    </div>
  )
}

// Value cell with tooltip for long values
function ValueCell({
  value,
  formatted,
  isDifferent,
}: {
  value: ConfigValue
  formatted: string
  isDifferent: boolean
}) {
  void value
  // Show tooltip for long values (more than 20 chars)
  const needsTooltip = formatted.length > 20

  const content = (
    <span
      className={cn(
        "text-sm font-sans truncate block max-w-full overflow-hidden text-ellipsis",
        isDifferent ? "font-medium" : "text-muted-foreground"
      )}
    >
      {formatted}
    </span>
  )

  if (needsTooltip) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{content}</TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs break-all font-sans text-xs">
          {formatted}
        </TooltipContent>
      </Tooltip>
    )
  }

  return content
}

// Compare config item (shows both values side by side)
function CompareConfigItem({
  displayKey,
  value1,
  value2,
  showOnlyDifferences,
}: {
  displayKey: string
  value1: ConfigValue
  value2: ConfigValue
  showOnlyDifferences: boolean
}) {
  const formatted1 = formatValue(value1)
  const formatted2 = formatValue(value2)
  const isDifferent = !valuesEqual(value1, value2)

  if (showOnlyDifferences && !isDifferent) {
    return null
  }

  return (
    <div
      className={cn(
        "grid grid-cols-[1fr_2fr_2fr] gap-3 py-1.5 rounded-md transition-colors min-w-0",
        isDifferent ? "hover:bg-muted/50" : "opacity-40 hover:opacity-60"
      )}
    >
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={cn(
              "text-sm truncate min-w-0",
              isDifferent ? "text-foreground" : "text-muted-foreground"
            )}
          >
            {displayKey}
          </span>
        </TooltipTrigger>
        <TooltipContent side="left" className="text-xs">
          {displayKey}
        </TooltipContent>
      </Tooltip>
      <div className="min-w-0 overflow-hidden">
        <ValueCell
          value={value1}
          formatted={formatted1}
          isDifferent={isDifferent}
        />
      </div>
      <div className="min-w-0 overflow-hidden">
        <ValueCell
          value={value2}
          formatted={formatted2}
          isDifferent={isDifferent}
        />
      </div>
    </div>
  )
}

// Chevron width (w-4 = 16px) + gap-1.5 (6px) = 22px
const LABEL_OFFSET = "ml-[22px]"
const NESTED_LABEL_OFFSET = "ml-[38px]"

// Compare category section
function CompareCategorySection({
  category,
  config1Map,
  config2Map,
  showOnlyDifferences,
  nested = false,
  defaultOpen,
}: {
  category: ConfigCategory
  config1Map: Map<string, ConfigValue>
  config2Map: Map<string, ConfigValue>
  showOnlyDifferences: boolean
  nested?: boolean
  defaultOpen?: boolean
}) {
  const hasDifferences = categoryHasDifferences(
    category,
    config1Map,
    config2Map
  )

  // Default open if has differences, unless overridden by defaultOpen prop
  const [isOpen, setIsOpen] = useState(defaultOpen ?? hasDifferences)

  // If showing only differences and no differences, hide this category
  if (showOnlyDifferences && !hasDifferences) {
    return null
  }

  const hasSubcategories =
    category.subcategories && category.subcategories.length > 0

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <div className={cn(
          "py-1.5 px-2 -mx-2 cursor-pointer hover:bg-muted rounded transition-colors",
          nested && "ml-4",
          !hasDifferences && "opacity-40"
        )}>
          <div className="flex items-center gap-1.5">
            <ChevronDown
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform",
                !isOpen && "-rotate-90"
              )}
            />
            <span className="text-sm font-semibold">{category.label}</span>
          </div>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        {hasSubcategories ? (
          <div className="mt-1">
            {category.subcategories!.map((subcat) => (
              <CompareCategorySection
                key={subcat.key}
                category={subcat}
                config1Map={config1Map}
                config2Map={config2Map}
                showOnlyDifferences={showOnlyDifferences}
                nested
              />
            ))}
          </div>
        ) : (
          <div className={cn("mt-1", nested ? NESTED_LABEL_OFFSET : LABEL_OFFSET)}>
            {category.items.map((item) => (
              <CompareConfigItem
                key={item.key}
                displayKey={item.displayKey}
                value1={config1Map.get(item.key)}
                value2={config2Map.get(item.key)}
                showOnlyDifferences={showOnlyDifferences}
              />
            ))}
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  )
}

// Main comparison view
function ConfigComparisonView({
  config1,
  config2,
  run1Name,
  run2Name,
  run1Path,
  run2Path,
  run1Color,
  run2Color,
}: {
  config1: Record<string, unknown>
  config2: Record<string, unknown>
  run1Name: string
  run2Name: string
  run1Path: string
  run2Path: string
  run1Color: string
  run2Color: string
}) {
  const [showOnlyDifferences, setShowOnlyDifferences] = useState(true)

  // Build maps and categorize
  const config1Map = useMemo(() => buildConfigMap(config1), [config1])
  const config2Map = useMemo(() => buildConfigMap(config2), [config2])

  // Merge configs to get all keys
  const mergedConfig = useMemo(() => {
    const merged: Record<string, unknown> = { ...config1 }
    for (const [key, value] of Object.entries(config2)) {
      if (!(key in merged)) {
        merged[key] = value
      }
    }
    return merged
  }, [config1, config2])

  const organizedCategories = useMemo(() => {
    return categorizeConfigs(mergedConfig)
  }, [mergedConfig])

  // Count differences
  const totalDifferences = useMemo(() => {
    let count = 0
    for (const [key, val1] of config1Map) {
      const val2 = config2Map.get(key)
      if (!valuesEqual(val1, val2)) count++
    }
    for (const [key] of config2Map) {
      if (!config1Map.has(key)) count++
    }
    return count
  }, [config1Map, config2Map])

  return (
    <div className="space-y-4">
      {/* Header with toggle */}
      <div className="flex items-center justify-between gap-4 pb-2 border-b">
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="text-xs">
            {totalDifferences} difference{totalDifferences !== 1 ? "s" : ""}
          </Badge>
        </div>
        <ToggleGroup
          type="single"
          value={showOnlyDifferences ? "differences" : "everything"}
          onValueChange={(value) => {
            if (value) setShowOnlyDifferences(value === "differences")
          }}
          variant="outline"
          size="sm"
        >
          <ToggleGroupItem value="everything" className="text-xs px-3">
            Everything
          </ToggleGroupItem>
          <ToggleGroupItem value="differences" className="text-xs px-3">
            Only differences
          </ToggleGroupItem>
        </ToggleGroup>
      </div>

      {/* Column headers */}
      <div className="grid grid-cols-[1fr_2fr_2fr] gap-3 pb-2 border-b text-sm font-medium">
        <span className="text-muted-foreground">Config</span>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: run1Color }}
            />
            <MiddleEllipsisLabel
              text={run1Name}
              className="text-sm font-medium min-w-0"
            />
          </div>
          <div className="ml-4">
            <MiddleEllipsisLabel
              text={run1Path}
              className="text-xs font-normal text-muted-foreground min-w-0"
            />
          </div>
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: run2Color }}
            />
            <MiddleEllipsisLabel
              text={run2Name}
              className="text-sm font-medium min-w-0"
            />
          </div>
          <div className="ml-4">
            <MiddleEllipsisLabel
              text={run2Path}
              className="text-xs font-normal text-muted-foreground min-w-0"
            />
          </div>
        </div>
      </div>

      {/* Categories */}
      <ScrollArea className="h-[72vh]">
        <div className="space-y-1 pr-4">
          {organizedCategories.map((category) => {
            // Hardware & Environment defaults closed unless it's the only
            // category with differences
            let defaultOpen: boolean | undefined
            if (category.key === "hardware") {
              const otherCategoriesWithDiffs = organizedCategories.filter(
                (c) =>
                  c.key !== "hardware" &&
                  categoryHasDifferences(c, config1Map, config2Map)
              )
              defaultOpen = otherCategoriesWithDiffs.length === 0
            }
            return (
              <CompareCategorySection
                key={category.key}
                category={category}
                config1Map={config1Map}
                config2Map={config2Map}
                showOnlyDifferences={showOnlyDifferences}
                defaultOpen={defaultOpen}
              />
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}

// Main dialog component
export function RunConfigCompareDialog({
  currentRunId,
  currentConfig,
}: {
  currentRunId: string
  currentConfig: Record<string, unknown>
}) {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedCompareRun, setSelectedCompareRun] = useState<string | null>(
    null
  )

  const { data: runsData, isLoading: isLoadingRuns } = useRuns()
  const runs = runsData?.runs || []

  const { data: compareSummary, isLoading: isLoadingCompare } = useRunSummary(
    selectedCompareRun || "",
    !!selectedCompareRun,
    false
  )

  // Get colors for runs (use persistent color from run data)
  const currentRun = runs.find((r) => r.run_id === currentRunId)
  const compareRun = runs.find((r) => r.run_id === selectedCompareRun)
  const currentRunColor = currentRun?.color ?? ""
  const compareRunColor = compareRun?.color ?? ""

  const handleRunSelect = (runId: string) => {
    setSelectedCompareRun(runId)
  }

  const handleBack = () => {
    setSelectedCompareRun(null)
  }

  const handleClose = () => {
    setIsOpen(false)
    // Reset state when dialog closes
    setTimeout(() => setSelectedCompareRun(null), 200)
  }

  const currentRunName = currentRun?.name || currentRunId.split("/").pop() || currentRunId
  const compareRunName = compareRun?.name || selectedCompareRun?.split("/").pop() || ""

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) handleClose()
        else setIsOpen(true)
      }}
    >
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="h-7 px-2 text-xs">
          Compare
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-4xl max-h-[95vh]">
        {!selectedCompareRun ? (
          // Run selection view
          <>
            <DialogHeader>
              <DialogTitle>Compare Configuration</DialogTitle>
              <DialogDescription>
                Select another run to compare its configuration with{" "}
                <span className="font-sans text-foreground">
                  {currentRunName}
                </span>
              </DialogDescription>
            </DialogHeader>
            <ScrollArea className="h-[500px] pr-4">
              {isLoadingRuns ? (
                <div className="flex items-center justify-center py-8">
                  <Spinner className="h-5 w-5" />
                </div>
              ) : (
                <RunSelector
                  runs={runs}
                  currentRunId={currentRunId}
                  onSelect={handleRunSelect}
                />
              )}
            </ScrollArea>
          </>
        ) : (
          // Comparison view
          <>
            <DialogHeader>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleBack}
                  className="h-7 px-2"
                >
                  ← Back
                </Button>
                <div>
                  <DialogTitle>Configuration Comparison</DialogTitle>
                </div>
              </div>
            </DialogHeader>
            {isLoadingCompare ? (
              <div className="flex items-center justify-center py-16">
                <Spinner className="h-6 w-6" />
                <span className="ml-3 text-muted-foreground">
                  Loading configuration...
                </span>
              </div>
            ) : compareSummary?.config ? (
              <ConfigComparisonView
                config1={compareSummary.config}
                config2={currentConfig}
                run1Name={compareRunName}
                run2Name={currentRunName}
                run1Path={selectedCompareRun || ""}
                run2Path={currentRunId}
                run1Color={compareRunColor}
                run2Color={currentRunColor}
              />
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No configuration data available for the selected run.
              </div>
            )}
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
