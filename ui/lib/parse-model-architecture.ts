// ============================================================================
// PyTorch nn.Module.__repr__() Parser
// ============================================================================

export type LayerCategory =
  | "embedding"
  | "attention"
  | "feedforward"
  | "normalization"
  | "output"
  | "positional"
  | "activation"
  | "other"

export interface ModelNode {
  name: string
  type: string
  args: string // raw argument string from repr, e.g. "4096, 11008, bias=False"
  children: ModelNode[]
  depth: number
  repeatCount: number // >1 when consecutive identical children are collapsed
  estimatedParams: number
}

export interface ModelStats {
  modelName: string
  totalParams: number
  numLayers: number
  hiddenSize: number | null
  vocabSize: number | null
  intermediateSize: number | null
  numAttentionHeads: number | null
  numKvHeads: number | null
  layerTypeCounts: Record<string, number>
  paramsByCategory: Record<LayerCategory, number>
}

// ============================================================================
// Parser
// ============================================================================

/**
 * Parse a PyTorch nn.Module.__repr__() string into a tree of ModelNodes.
 *
 * The format looks like:
 *   ModelName(
 *     (child_name): ChildType(args)
 *     (parent): ParentModule(
 *       (sub): SubModule(args)
 *     )
 *   )
 */
export function parseModelArchitecture(repr: string): ModelNode | null {
  if (!repr || typeof repr !== "string") return null

  const lines = repr.split("\n")
  if (lines.length === 0) return null

  // Parse root line: "ModelName("
  const rootMatch = lines[0].match(/^(\w[\w.]*)(\(.*)?$/)
  if (!rootMatch) return null

  const root: ModelNode = {
    name: "",
    type: rootMatch[1],
    args: "",
    children: [],
    depth: 0,
    repeatCount: 1,
    estimatedParams: 0,
  }

  // If it's a single-line module (no children), extract args
  if (lines.length === 1) {
    const argMatch = lines[0].match(/^\w[\w.]*\((.+)\)\s*$/)
    root.args = argMatch ? argMatch[1] : ""
    root.estimatedParams = estimateParamsFromArgs(root.type, root.args)
    return root
  }

  // Stack-based parser for multi-line repr
  const stack: ModelNode[] = [root]

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trimStart()

    // Skip empty lines and closing parens
    if (!trimmed || trimmed === ")") {
      if (trimmed === ")" && stack.length > 1) {
        stack.pop()
      }
      continue
    }

    // Try compact repeated format first: "(0-35): 36 x TypeName(" or "(0-35): 36 x TypeName(args)"
    const repeatMatch = trimmed.match(
      /^\(([^)]*)\):\s*(\d+)\s*x\s+(\w[\w.]*)\((.*)$/
    )

    if (repeatMatch) {
      const childName = repeatMatch[1]
      const repeatCount = parseInt(repeatMatch[2], 10)
      const childType = repeatMatch[3]
      let argsOrOpen = repeatMatch[4]
      const parent = stack[stack.length - 1]

      if (argsOrOpen.endsWith(")")) {
        argsOrOpen = argsOrOpen.slice(0, -1)
        const child: ModelNode = {
          name: childName,
          type: childType,
          args: argsOrOpen,
          children: [],
          depth: stack.length,
          repeatCount: repeatCount,
          estimatedParams: 0,
        }
        child.estimatedParams = estimateParamsFromArgs(childType, argsOrOpen)
        parent.children.push(child)
      } else {
        const child: ModelNode = {
          name: childName,
          type: childType,
          args: "",
          children: [],
          depth: stack.length,
          repeatCount: repeatCount,
          estimatedParams: 0,
        }
        parent.children.push(child)
        stack.push(child)
      }
      continue
    }

    // Match "(name): Type(args)" or "(name): Type("
    const childMatch = trimmed.match(
      /^\(([^)]*)\):\s*(\w[\w.]*)\((.*)$/
    )
    if (!childMatch) continue

    const childName = childMatch[1]
    const childType = childMatch[2]
    let argsOrOpen = childMatch[3]

    const parent = stack[stack.length - 1]

    // Check if this opens a sub-block (ends with no closing paren) or is self-contained
    if (argsOrOpen.endsWith(")")) {
      // Self-contained: "(name): Type(args)"
      argsOrOpen = argsOrOpen.slice(0, -1)
      const child: ModelNode = {
        name: childName,
        type: childType,
        args: argsOrOpen,
        children: [],
        depth: stack.length,
        repeatCount: 1,
        estimatedParams: 0,
      }
      child.estimatedParams = estimateParamsFromArgs(childType, argsOrOpen)
      parent.children.push(child)
    } else {
      // Opens a sub-block: "(name): Type("
      const child: ModelNode = {
        name: childName,
        type: childType,
        args: "",
        children: [],
        depth: stack.length,
        repeatCount: 1,
        estimatedParams: 0,
      }
      parent.children.push(child)
      stack.push(child)
    }
  }

  // Collapse repeated identical children (e.g., layers.0 through layers.31)
  collapseRepeats(root)

  // Compute estimated params bottom-up
  computeParams(root)

  return root
}

/** Check if a node name looks like a numeric index (e.g. "0", "12", "0-35") */
function isNumericName(name: string): boolean {
  return /^\d+(-\d+)?$/.test(name)
}

/**
 * Collapse consecutive children with identical structure into a single node
 * with repeatCount > 1. Works recursively.
 */
function collapseRepeats(node: ModelNode): void {
  // First recurse into children
  for (const child of node.children) {
    collapseRepeats(child)
  }

  if (node.children.length < 2) return

  const collapsed: ModelNode[] = []
  let i = 0

  while (i < node.children.length) {
    const current = node.children[i]
    let count = 1

    // Count consecutive identical siblings (same type + same structure).
    // Only collapse nodes whose names look like numeric indices (e.g. "0", "1")
    // to avoid merging semantically distinct layers like k_proj and v_proj.
    while (
      i + count < node.children.length &&
      isNumericName(current.name) &&
      isNumericName(node.children[i + count].name) &&
      nodesStructurallyEqual(current, node.children[i + count])
    ) {
      count++
    }

    if (count > 1) {
      const merged = { ...current, repeatCount: count }
      collapsed.push(merged)
    } else {
      collapsed.push(current)
    }
    i += count
  }

  node.children = collapsed
}

function nodesStructurallyEqual(a: ModelNode, b: ModelNode): boolean {
  if (a.type !== b.type || a.args !== b.args) return false
  if (a.children.length !== b.children.length) return false
  for (let i = 0; i < a.children.length; i++) {
    if (!nodesStructurallyEqual(a.children[i], b.children[i])) return false
  }
  return true
}

function computeParams(node: ModelNode): number {
  if (node.children.length === 0) {
    // Leaf node: estimatedParams is per-instance (set from args)
    return node.estimatedParams * (node.repeatCount || 1)
  }

  let childSum = 0
  for (const child of node.children) {
    childSum += computeParams(child)
  }

  // childSum = total params for one instance of this composite
  node.estimatedParams = childSum
  return childSum * (node.repeatCount || 1)
}

// ============================================================================
// Parameter Estimation
// ============================================================================

function estimateParamsFromArgs(type: string, args: string): number {
  if (!args) return 0

  const nums = args.match(/\d+/g)?.map(Number) ?? []
  const hasBias = !args.includes("bias=False")
  const lowerType = type.toLowerCase()

  // Linear(in_features, out_features, bias=...)
  if (lowerType === "linear" && nums.length >= 2) {
    const params = nums[0] * nums[1]
    return hasBias ? params + nums[1] : params
  }

  // Embedding(num_embeddings, embedding_dim, ...)
  if (lowerType === "embedding" && nums.length >= 2) {
    return nums[0] * nums[1]
  }

  // LayerNorm / RMSNorm / GroupNorm — typically just the dimension
  if (
    (lowerType.includes("norm") || lowerType.includes("layernorm") || lowerType.includes("rmsnorm")) &&
    nums.length >= 1
  ) {
    // RMSNorm has just the dimension as weight, LayerNorm has weight + bias
    const dim = nums[0]
    if (lowerType.includes("rmsnorm")) return dim
    return hasBias ? dim * 2 : dim
  }

  // Conv1d / Conv2d — approximate
  if (lowerType.startsWith("conv") && nums.length >= 3) {
    const [inCh, outCh, kernel] = nums
    const params = inCh * outCh * kernel * (nums[3] ?? kernel)
    return hasBias ? params + outCh : params
  }

  return 0
}

// ============================================================================
// Stats Extraction
// ============================================================================

export function extractModelStats(root: ModelNode): ModelStats {
  const stats: ModelStats = {
    modelName: root.type,
    totalParams: 0,
    numLayers: 0,
    hiddenSize: null,
    vocabSize: null,
    intermediateSize: null,
    numAttentionHeads: null,
    numKvHeads: null,
    layerTypeCounts: {},
    paramsByCategory: {
      embedding: 0,
      attention: 0,
      feedforward: 0,
      normalization: 0,
      output: 0,
      positional: 0,
      activation: 0,
      other: 0,
    },
  }

  // Walk tree to collect stats
  walkTree(root, stats, 1)

  return stats
}

function walkTree(node: ModelNode, stats: ModelStats, multiplier: number): void {
  const count = (node.repeatCount || 1) * multiplier

  if (node.children.length === 0 && node.type) {
    // Leaf layer — count it
    const key = node.type
    stats.layerTypeCounts[key] = (stats.layerTypeCounts[key] || 0) + count
    stats.totalParams += node.estimatedParams * count

    const category = getLayerCategory(node)
    stats.paramsByCategory[category] += node.estimatedParams * count

    // Try to extract architecture dimensions from args
    const nums = node.args.match(/\d+/g)?.map(Number) ?? []
    const lowerType = node.type.toLowerCase()

    if (lowerType === "embedding" && nums.length >= 2) {
      if (!stats.vocabSize || nums[0] > stats.vocabSize) {
        stats.vocabSize = nums[0]
      }
      if (!stats.hiddenSize) {
        stats.hiddenSize = nums[1]
      }
    }

    if (lowerType === "linear" && nums.length >= 2) {
      // Intermediate size is typically the larger dimension in MLP layers
      const maxDim = Math.max(nums[0], nums[1])
      if (
        stats.hiddenSize &&
        maxDim > stats.hiddenSize &&
        (!stats.intermediateSize || maxDim > stats.intermediateSize)
      ) {
        stats.intermediateSize = maxDim
      }
    }
  } else if (node.children.length > 0) {
    // Composite node — check if it's a repeated layer block
    if (node.repeatCount > 1 && node.depth > 0) {
      stats.numLayers = Math.max(stats.numLayers, node.repeatCount)
    }

    for (const child of node.children) {
      walkTree(child, stats, count)
    }
  }

  // For composite nodes, try to detect attention head counts from children
  if (node.children.length > 0) {
    const lowerName = node.name.toLowerCase()
    if (lowerName.includes("attn") || lowerName.includes("attention")) {
      // Look for q_proj, k_proj linear layers to infer head counts
      for (const child of node.children) {
        if (child.type.toLowerCase() === "linear") {
          const nums = child.args.match(/\d+/g)?.map(Number) ?? []
          if (nums.length >= 2) {
            const lowerChildName = child.name.toLowerCase()
            if (lowerChildName.includes("q_proj") && !stats.numAttentionHeads && stats.hiddenSize) {
              // num heads = out_features / head_dim (commonly out == hidden)
            }
            if (lowerChildName.includes("k_proj") && !stats.numKvHeads && stats.hiddenSize) {
              const kvDim = nums[1]
              if (kvDim < (stats.hiddenSize ?? Infinity)) {
                // GQA: kv heads = kv_dim / head_dim
              }
            }
          }
        }
      }
    }
  }
}

// ============================================================================
// Layer Categorization
// ============================================================================

export function getLayerCategory(node: ModelNode): LayerCategory {
  const t = node.type.toLowerCase()
  const n = node.name.toLowerCase()

  if (t === "embedding" || n.includes("embed")) return "embedding"
  if (
    t.includes("attention") ||
    n.includes("attn") ||
    n.includes("attention") ||
    n.includes("q_proj") ||
    n.includes("k_proj") ||
    n.includes("v_proj") ||
    n.includes("o_proj")
  )
    return "attention"
  if (
    n.includes("mlp") ||
    n.includes("feed_forward") ||
    n.includes("ffn") ||
    n.includes("gate_proj") ||
    n.includes("up_proj") ||
    n.includes("down_proj") ||
    n.includes("fc1") ||
    n.includes("fc2")
  )
    return "feedforward"
  if (t.includes("norm") || n.includes("norm")) return "normalization"
  if (n.includes("lm_head") || n.includes("output") || n.includes("head"))
    return "output"
  if (
    t.includes("rotary") ||
    t.includes("rope") ||
    n.includes("rotary") ||
    n.includes("rope") ||
    n.includes("pos")
  )
    return "positional"
  if (
    t.includes("relu") ||
    t.includes("gelu") ||
    t.includes("silu") ||
    t.includes("swish") ||
    t.includes("activation") ||
    t === "silu" ||
    t === "gelu" ||
    t === "relu"
  )
    return "activation"

  return "other"
}

// ============================================================================
// Formatting
// ============================================================================

export function formatParamCount(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`
  if (n > 0) return `${n}`
  return "0"
}

/** Clean up verbose PyTorch args for compact display */
export function formatArgs(type: string, args: string): string {
  if (!args) return ""
  let s = args
  // Strip keyword prefixes
  s = s.replace(/in_features=/g, "")
  s = s.replace(/out_features=/g, "")
  s = s.replace(/num_embeddings=/g, "")
  s = s.replace(/embedding_dim=/g, "")
  // Remove bias=False (default, not interesting)
  s = s.replace(/,\s*bias=False/g, "")
  s = s.replace(/bias=False,?\s*/g, "")
  // Clean tuple notation (2048,) → 2048
  s = s.replace(/\((\d+),?\)/g, "$1")
  // Remove eps= for norms
  if (type.toLowerCase().includes("norm")) {
    s = s.replace(/,?\s*eps=[\d.eE+-]+/g, "")
  }
  // Clean up stray commas / whitespace
  s = s.replace(/,\s*$/, "").replace(/^\s*,\s*/, "").trim()
  return s
}

/** Category display color classes (Tailwind) */
export const CATEGORY_COLORS: Record<LayerCategory, { border: string; bg: string; text: string; bar: string }> = {
  embedding:     { border: "border-l-violet-500",  bg: "bg-violet-500/10",  text: "text-violet-600 dark:text-violet-400",  bar: "bg-violet-500" },
  attention:     { border: "border-l-blue-500",    bg: "bg-blue-500/10",    text: "text-blue-600 dark:text-blue-400",      bar: "bg-blue-500" },
  feedforward:   { border: "border-l-amber-500",   bg: "bg-amber-500/10",   text: "text-amber-600 dark:text-amber-400",    bar: "bg-amber-500" },
  normalization: { border: "border-l-emerald-500", bg: "bg-emerald-500/10", text: "text-emerald-600 dark:text-emerald-400", bar: "bg-emerald-500" },
  output:        { border: "border-l-rose-500",    bg: "bg-rose-500/10",    text: "text-rose-600 dark:text-rose-400",      bar: "bg-rose-500" },
  positional:    { border: "border-l-cyan-500",    bg: "bg-cyan-500/10",    text: "text-cyan-600 dark:text-cyan-400",      bar: "bg-cyan-500" },
  activation:    { border: "border-l-orange-500",  bg: "bg-orange-500/10",  text: "text-orange-600 dark:text-orange-400",  bar: "bg-orange-500" },
  other:         { border: "border-l-gray-500",    bg: "bg-gray-500/10",    text: "text-gray-600 dark:text-gray-400",      bar: "bg-gray-500" },
}
