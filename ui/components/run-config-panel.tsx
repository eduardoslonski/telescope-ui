
import { useState, useMemo } from "react"
import {
  ChevronDown,
} from "lucide-react"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { cn } from "@/lib/utils"

// Type for configuration values
type ConfigValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | Record<string, unknown>
  | Array<unknown>

interface ConfigCategory {
  key: string
  label: string
  items: Array<{ key: string; displayKey: string; value: ConfigValue }>
  subcategories?: ConfigCategory[]
}

function stableStringify(value: unknown, space = 0): string {
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
    }, space)
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
    // Format numbers nicely
    if (Number.isInteger(value)) {
      return value.toLocaleString()
    }
    // For decimals, show appropriate precision
    if (Math.abs(value) < 0.001 && value !== 0) {
      return value.toExponential(2)
    }
    return value.toLocaleString(undefined, { maximumFractionDigits: 6 })
  }
  if (typeof value === "object") {
    return stableStringify(value, 2)
  }
  return String(value)
}

function maybeParseJsonString(value: ConfigValue): ConfigValue {
  if (typeof value !== "string") return value
  const trimmed = value.trim()
  if (!trimmed) return value
  const looksLikeJsonObject = trimmed.startsWith("{") && trimmed.endsWith("}")
  const looksLikeJsonArray = trimmed.startsWith("[") && trimmed.endsWith("]")
  if (!looksLikeJsonObject && !looksLikeJsonArray) return value

  try {
    const parsed = JSON.parse(trimmed)
    if (Array.isArray(parsed)) return parsed
    if (parsed && typeof parsed === "object") {
      return parsed as Record<string, unknown>
    }
  } catch {
    // Keep original value when string is not valid JSON.
  }

  return value
}

// Get a nice display key from a full key path
function getDisplayKey(key: string): string {
  // Remove common prefixes and format nicely
  const parts = key.split("/")
  const lastPart = parts[parts.length - 1]
  const trimmedPrefix = lastPart.replace(/^_+/, "")
  const normalized = trimmedPrefix.length > 0 ? trimmedPrefix : lastPart
  return normalized.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

// Categorize configuration keys
function categorizeConfigs(
  config: Record<string, unknown>
): ConfigCategory[] {
  // Define category matchers in desired display order
  // (matches sections in the default telescope config file)
  // The `order` arrays define the display order of keys within each category;
  // keys not listed are appended at the end alphabetically.
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
          "grad_clip",
          "lr_scheduler",
          "warmup_steps",
          "min_lr_ratio",
          "train_backend",
          "fsdp_activation_checkpointing",
          "jit",
        ].includes(k) || k.startsWith("megatron_"),
      order: [
        "learning_rate", "weight_decay", "grad_clip",
        "lr_scheduler", "warmup_steps", "min_lr_ratio",
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
          "dr_grpo_loss_agg_mode",
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
        "dr_grpo_loss_agg_mode", "advantage_norm", "use_tis", "tis_cap", "tis_logprob_clamp",
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
          "reasoning_parser",
          "enable_tool_call",
          "tool_call_parser",
        ].includes(k),
      order: [
        "inference_host", "inference_devices", "inference_base_port",
        "num_inference_servers", "gpu_memory_utilization", "max_model_len",
        "vllm_scheduling_policy", "enable_thinking", "reasoning_parser",
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
      match: (k) => k.startsWith("eval_") || k === "evals",
      order: [
        "eval_before_training", "eval_after_training",
        "eval_num_servers", "eval_start_end_use_all_servers",
        "evals",
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
      match: (k) => k === "environments",
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
    const extractedValue = extractValue(rawValue)
    const value =
      key === "environments"
        ? maybeParseJsonString(extractedValue)
        : extractedValue

    // Find matching category
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

    // If no match, add to "other"
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
    "hardware", // placeholder for grouped hardware
    "other",
  ]

  const hardwareSubcategories: ConfigCategory[] = []
  for (const hk of hardwareKeys) {
    const cat = categoryMap.get(hk)
    if (cat && cat.items.length > 0) {
      hardwareSubcategories.push(cat)
    }
  }

  // Build result in final order
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

// ConfigItem component
function ConfigItem({
  displayKey,
  value,
}: {
  displayKey: string
  value: ConfigValue
}) {
  const formattedValue = formatValue(value)
  const isStructuredValue = typeof value === "object" && value !== null

  if (isStructuredValue) {
    return (
      <div className="py-1 rounded-md hover:bg-muted/50 transition-colors group">
        <span className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">
          {displayKey}
        </span>
        <pre className="mt-1 text-xs font-sans font-medium rounded-md bg-muted/40 border border-border/40 p-2 whitespace-pre-wrap break-all max-h-72 overflow-auto">
          {formattedValue}
        </pre>
      </div>
    )
  }

  return (
    <div className="flex items-baseline gap-2 py-1 rounded-md hover:bg-muted/50 transition-colors group">
      <span className="text-sm text-muted-foreground group-hover:text-foreground transition-colors whitespace-nowrap">
        {displayKey}
      </span>
      <span className="text-sm font-sans font-medium break-all">
        {formattedValue}
      </span>
    </div>
  )
}

// Chevron width (w-4 = 16px) + gap-1.5 (6px) = 22px
// This offset aligns item text with the category label text
const LABEL_OFFSET = "ml-[22px]"
const NESTED_LABEL_OFFSET = "ml-[38px]"

// CategorySection component
function CategorySection({
  category,
  defaultOpen = false,
  nested = false,
  collapseAllSignal = 0,
  expandAllSignal = 0,
}: {
  category: ConfigCategory
  defaultOpen?: boolean
  nested?: boolean
  collapseAllSignal?: number
  expandAllSignal?: number
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  const [prevCollapseSignal, setPrevCollapseSignal] = useState(collapseAllSignal)
  const [prevExpandSignal, setPrevExpandSignal] = useState(expandAllSignal)

  // Collapse all when signal changes (adjust state during render)
  if (collapseAllSignal !== prevCollapseSignal) {
    setPrevCollapseSignal(collapseAllSignal)
    setIsOpen(false)
  }

  // Expand all when signal changes (adjust state during render)
  if (expandAllSignal !== prevExpandSignal) {
    setPrevExpandSignal(expandAllSignal)
    setIsOpen(true)
  }

  const hasSubcategories =
    category.subcategories && category.subcategories.length > 0

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <div className={cn(
          "py-1.5 px-2 -mx-2 cursor-pointer hover:bg-gray-50 rounded transition-colors",
          nested && "ml-4"
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
              <CategorySection
                key={subcat.key}
                category={subcat}
                nested
                collapseAllSignal={collapseAllSignal}
                expandAllSignal={expandAllSignal}
              />
            ))}
          </div>
        ) : (
          <div className={cn("mt-1", nested ? NESTED_LABEL_OFFSET : LABEL_OFFSET)}>
            {category.items.map((item) => (
              <ConfigItem
                key={item.key}
                displayKey={item.displayKey}
                value={item.value}
              />
            ))}
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  )
}

// Main component
export function RunConfigPanel({
  config,
  collapseAllSignal = 0,
  expandAllSignal = 0,
}: {
  config: Record<string, unknown>
  collapseAllSignal?: number
  expandAllSignal?: number
}) {
  const organizedCategories = useMemo(() => {
    return categorizeConfigs(config)
  }, [config])

  // Define default open categories (most important ones)
  const defaultOpenCategories = new Set(["model", "environments"])

  if (organizedCategories.length === 0) {
    return (
      <div className="text-sm text-muted-foreground text-center py-8">
        No configuration data available
      </div>
    )
  }

  return (
    <div className="space-y-1">
      {organizedCategories.map((category) => (
        <CategorySection
          key={category.key}
          category={category}
          defaultOpen={defaultOpenCategories.has(category.key)}
          collapseAllSignal={collapseAllSignal}
          expandAllSignal={expandAllSignal}
        />
      ))}
    </div>
  )
}
