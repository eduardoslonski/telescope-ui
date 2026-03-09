
import { useMemo, useState } from "react"
import { ChevronDown, ChevronRight, Layers } from "lucide-react"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  parseModelArchitecture,
  extractModelStats,
  formatParamCount,
  formatArgs,
  CATEGORY_COLORS,
  type ModelNode,
  type ModelStats,
  type LayerCategory,
} from "@/lib/parse-model-architecture"

// ============================================================================
// Name Formatting
// ============================================================================

const NAME_MAP: Record<string, string> = {
  self_attn: "Self Attention",
  q_proj: "Q Proj",
  k_proj: "K Proj",
  v_proj: "V Proj",
  o_proj: "O Proj",
  gate_proj: "Gate Proj",
  up_proj: "Up Proj",
  down_proj: "Down Proj",
  act_fn: "Activation",
  input_layernorm: "Input LayerNorm",
  post_attention_layernorm: "Post-Attention LayerNorm",
  embed_tokens: "Embed Tokens",
  rotary_emb: "Rotary Embedding",
  lm_head: "LM Head",
  mlp: "MLP",
  norm: "Norm",
  model: "Model",
  layers: "Layers",
}

function formatNodeName(name: string): string {
  if (!name) return ""
  if (NAME_MAP[name]) return NAME_MAP[name]
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

// ============================================================================
// Tree Flattening — unwrap transparent wrappers for a cleaner flow
// ============================================================================

function getFlowNodes(root: ModelNode): ModelNode[] {
  const result: ModelNode[] = []

  function shouldFlatten(node: ModelNode): boolean {
    if (node.name === "") return true
    if (node.name === "model" && node.children.length > 0) return true
    if (
      (node.type === "ModuleList" || node.type === "Sequential") &&
      node.children.length > 0
    )
      return true
    return false
  }

  function collect(node: ModelNode) {
    if (shouldFlatten(node)) {
      for (const child of node.children) {
        collect(child)
      }
    } else {
      result.push(node)
    }
  }

  collect(root)
  return result
}

// ============================================================================
// Connector
// ============================================================================

function Connector() {
  return <div className="w-px h-4 bg-border mx-auto" />
}

// ============================================================================
// Leaf Block — full-width card for standalone leaf nodes
// ============================================================================

function LeafBlock({ node }: { node: ModelNode }) {
  const displayName = formatNodeName(node.name) || node.type
  const cleanArgs = formatArgs(node.type, node.args)
  const typeStr = cleanArgs
    ? `${node.type}(${cleanArgs})`
    : node.type !== displayName
      ? node.type
      : ""
  const paramText =
    node.estimatedParams > 0 ? formatParamCount(node.estimatedParams) : null

  return (
    <div className="rounded-lg border bg-card px-3 py-2.5 w-full text-center">
      <div className="font-medium text-sm text-foreground">
        {displayName}
      </div>
      {typeStr && (
        <div className="text-xs text-muted-foreground mt-0.5 break-all">
          {typeStr}
        </div>
      )}
      {paramText && (
        <div className="text-xs text-muted-foreground mt-0.5">
          {paramText}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Child Pill — compact item for horizontal rows inside groups
// ============================================================================

function ChildPill({ node }: { node: ModelNode }) {
  const displayName = formatNodeName(node.name) || node.type
  const cleanArgs = formatArgs(node.type, node.args)
  const typeStr = cleanArgs
    ? `${node.type}(${cleanArgs})`
    : node.type !== displayName
      ? node.type
      : ""
  const paramText =
    node.estimatedParams > 0 ? formatParamCount(node.estimatedParams) : null

  return (
    <div className="rounded-md border bg-background px-2.5 py-2 flex-1 min-w-0 text-center">
      <div className="text-xs font-medium text-foreground">
        {displayName}
      </div>
      {typeStr && (
        <div className="text-[10px] text-muted-foreground mt-0.5 break-all">
          {typeStr}
        </div>
      )}
      {paramText && (
        <div className="text-[10px] text-muted-foreground mt-0.5">
          {paramText}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Group Block — composite with all-leaf children shown as label + row
// ============================================================================

function GroupBlock({ node }: { node: ModelNode }) {
  const displayName = formatNodeName(node.name) || node.type
  const typeStr = node.type !== displayName ? node.type : ""
  const paramText =
    node.estimatedParams > 0 ? formatParamCount(node.estimatedParams) : null

  return (
    <div className="rounded-lg border bg-card px-3 py-2.5 w-full text-center">
      <div className="font-medium text-sm text-foreground">
        {displayName}
      </div>
      {typeStr && (
        <div className="text-xs text-muted-foreground mt-0.5">
          {typeStr}
        </div>
      )}
      {paramText && (
        <div className="text-xs text-muted-foreground mt-0.5">
          {paramText}
        </div>
      )}
      <div className="flex gap-1.5 flex-wrap mt-2">
        {node.children.map((child, i) => (
          <ChildPill key={i} node={child} />
        ))}
      </div>
    </div>
  )
}

// ============================================================================
// Flow Node — dispatches to the right renderer
// ============================================================================

function FlowNode({ node }: { node: ModelNode }) {
  // Leaf node
  if (node.children.length === 0) {
    return <LeafBlock node={node} />
  }

  // Repeated composite (×N layers)
  if (node.repeatCount > 1) {
    return <RepeatedBlock node={node} />
  }

  // Composite with all-leaf children → group with horizontal row
  const allLeaves = node.children.every((c) => c.children.length === 0)
  if (allLeaves) {
    return <GroupBlock node={node} />
  }

  // Composite with mixed children → vertical flow
  return <MixedBlock node={node} />
}

// ============================================================================
// Repeated Block — dashed border with ×N label
// ============================================================================

function RepeatedBlock({ node }: { node: ModelNode }) {
  return (
    <div className="relative w-full">
      {/* Dashed container — full width */}
      <div className="rounded-lg border-2 border-dashed border-muted-foreground/25 p-3 w-full">
        <div className="flex flex-col items-center w-full">
          {node.children.map((child, i) => (
            <div key={i} className="flex flex-col items-center w-full">
              {i > 0 && <Connector />}
              <FlowNode node={child} />
            </div>
          ))}
        </div>
      </div>

      {/* ×N label — positioned outside to the right, does not affect box width */}
      <div className="absolute left-full top-1/2 -translate-y-1/2 pl-5 pointer-events-none select-none">
        <span className="text-4xl font-extralight text-muted-foreground/50 whitespace-nowrap leading-tight">
          ×{node.repeatCount}
          <br />
          <span className="text-2xl">layers</span>
        </span>
      </div>
    </div>
  )
}

// ============================================================================
// Mixed Block — composite with mixed children rendered vertically
// ============================================================================

function MixedBlock({ node }: { node: ModelNode }) {
  const displayName = formatNodeName(node.name) || node.type
  const typeStr = node.type !== displayName ? node.type : ""
  const paramText =
    node.estimatedParams > 0 ? formatParamCount(node.estimatedParams) : null

  return (
    <div className="rounded-lg border bg-card px-3 py-2.5 w-full">
      <div className="text-center">
        <div className="font-medium text-sm text-foreground">
          {displayName}
        </div>
        {typeStr && (
          <div className="text-xs text-muted-foreground mt-0.5">
            {typeStr}
          </div>
        )}
        {paramText && (
          <div className="text-xs text-muted-foreground mt-0.5">
            {paramText}
          </div>
        )}
      </div>
      <div className="flex flex-col items-center w-full mt-2">
        {node.children.map((child, i) => (
          <div key={i} className="flex flex-col items-center w-full">
            {i > 0 && <Connector />}
            <FlowNode node={child} />
          </div>
        ))}
      </div>
    </div>
  )
}

// ============================================================================
// Flow Diagram — top-level flow from Input to Output
// ============================================================================

function FlowDiagram({ root }: { root: ModelNode }) {
  const flowNodes = useMemo(() => getFlowNodes(root), [root])

  return (
    <div className="flex flex-col items-center py-8 px-6 max-w-2xl mx-auto">
      {/* Input pill */}
      <div className="rounded-full border bg-background px-5 py-1.5 text-xs text-muted-foreground">
        Input
      </div>

      {flowNodes.map((node, i) => (
        <div key={i} className="flex flex-col items-center w-full">
          <Connector />
          <FlowNode node={node} />
        </div>
      ))}

      <Connector />

      {/* Output pill */}
      <div className="rounded-full border bg-background px-5 py-1.5 text-xs text-muted-foreground">
        Output
      </div>
    </div>
  )
}

// ============================================================================
// Summary Sidebar
// ============================================================================

function SummarySidebar({
  stats,
  rawRepr,
}: {
  stats: ModelStats
  rawRepr: string
}) {
  const [rawOpen, setRawOpen] = useState(false)

  const totalParams = stats.totalParams
  const sortedCategories = useMemo(() => {
    return (
      Object.entries(stats.paramsByCategory) as [LayerCategory, number][]
    )
      .filter(([, v]) => v > 0)
      .sort((a, b) => b[1] - a[1])
  }, [stats.paramsByCategory])

  const sortedLayerTypes = useMemo(() => {
    return Object.entries(stats.layerTypeCounts).sort((a, b) => b[1] - a[1])
  }, [stats.layerTypeCounts])

  return (
    <div className="w-72 shrink-0 border-l overflow-y-auto p-4 space-y-5">
      {/* Model name + total params */}
      <div>
        <h2 className="text-base font-semibold">{stats.modelName}</h2>
        {totalParams > 0 && (
          <p className="text-sm text-muted-foreground">
            ~{formatParamCount(totalParams)} parameters
          </p>
        )}
      </div>

      {/* Architecture stats */}
      {(stats.numLayers > 0 || stats.hiddenSize || stats.vocabSize) && (
        <div>
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
            Architecture
          </h3>
          <div className="space-y-1 text-sm">
            {stats.numLayers > 0 && (
              <StatRow label="Layers" value={String(stats.numLayers)} />
            )}
            {stats.hiddenSize && (
              <StatRow
                label="Hidden size"
                value={stats.hiddenSize.toLocaleString()}
              />
            )}
            {stats.vocabSize && (
              <StatRow
                label="Vocab size"
                value={stats.vocabSize.toLocaleString()}
              />
            )}
            {stats.intermediateSize && (
              <StatRow
                label="Intermediate"
                value={stats.intermediateSize.toLocaleString()}
              />
            )}
          </div>
        </div>
      )}

      {/* Parameter breakdown bar */}
      {totalParams > 0 && sortedCategories.length > 0 && (
        <div>
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
            Parameters
          </h3>
          <div className="flex h-2 rounded-full overflow-hidden mb-2.5">
            {sortedCategories.map(([cat, count]) => (
              <div
                key={cat}
                className={`${CATEGORY_COLORS[cat].bar}`}
                style={{
                  width: `${(count / totalParams) * 100}%`,
                  minWidth: count > 0 ? "2px" : 0,
                }}
              />
            ))}
          </div>
          <div className="space-y-0.5">
            {sortedCategories.map(([cat, count]) => (
              <div key={cat} className="flex items-center gap-2 text-xs">
                <div
                  className={`w-2 h-2 rounded-sm shrink-0 ${CATEGORY_COLORS[cat].bar}`}
                />
                <span className="capitalize text-muted-foreground">
                  {cat}
                </span>
                <span className="ml-auto text-muted-foreground tabular-nums">
                  {formatParamCount(count)} ({((count / totalParams) * 100).toFixed(0)}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Layer type counts */}
      {sortedLayerTypes.length > 0 && (
        <div>
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
            Layer Types
          </h3>
          <div className="space-y-0.5">
            {sortedLayerTypes.map(([type, count]) => (
              <div key={type} className="flex items-center text-xs">
                <span className="text-foreground">{type}</span>
                <span className="ml-auto text-muted-foreground tabular-nums">
                  ×{count}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Raw repr */}
      <Collapsible open={rawOpen} onOpenChange={setRawOpen}>
        <CollapsibleTrigger className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wider hover:text-foreground transition-colors">
          {rawOpen ? (
            <ChevronDown className="size-3" />
          ) : (
            <ChevronRight className="size-3" />
          )}
          Raw repr
        </CollapsibleTrigger>
        <CollapsibleContent>
          <pre className="mt-2 text-[10px] leading-tight bg-muted/50 rounded-md p-2 overflow-x-auto max-h-96 overflow-y-auto whitespace-pre font-sans">
            {rawRepr}
          </pre>
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium tabular-nums">{value}</span>
    </div>
  )
}

// ============================================================================
// Main Component
// ============================================================================

export function ModelArchitectureViewer({
  modelRepr,
}: {
  modelRepr: string
}) {
  const parsed = useMemo(
    () => parseModelArchitecture(modelRepr),
    [modelRepr]
  )
  const stats = useMemo(
    () => (parsed ? extractModelStats(parsed) : null),
    [parsed]
  )

  if (!parsed || !stats) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Layers className="size-8 text-muted-foreground mx-auto mb-2" />
          <div className="text-sm text-muted-foreground">
            Could not parse model architecture
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full">
      {/* Flow diagram */}
      <div className="flex-1 overflow-auto">
        <FlowDiagram root={parsed} />
      </div>

      {/* Summary sidebar */}
      <SummarySidebar stats={stats} rawRepr={modelRepr} />
    </div>
  )
}
