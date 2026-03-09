
import { useEffect, useState, useMemo, useRef, useCallback } from "react"
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber"
import { OrbitControls, RoundedBox, Text, Html, Edges } from "@react-three/drei"
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib"
import * as THREE from "three"

// ============================================================================
// Types for parsed setup data
// ============================================================================

interface GpuHardware {
  name: string
  memory_total_gb: number
  pcie_gen: number
  pcie_width: number
  serial: string
  uuid: string
}

interface GpuRole {
  role: "trainer" | "inference" | "unassigned"
  gpu_index: number
  // Trainer-specific
  rank?: number
  dp_rank?: number
  tp_rank?: number
  pp_rank?: number
  // Inference-specific
  server_idx?: number
  tp_group_id?: number
  port?: number
  url?: string
}

interface CpuInfo {
  model: string
  cores: number
  logical_cores: number
  sockets: number
  architecture: string
  frequency_mhz: number
}

interface MemoryInfo {
  total_gb: number
}

interface GpuInterconnect {
  nvlink: boolean
  topology: string
  links_per_gpu: number
  speed_gbps: number
  connections: number
}

interface SoftwareInfo {
  cuda: string
  pytorch: string
  python: string
}

interface ParsedNode {
  node_id: number
  hostname: string
  ip: string
  is_driver: boolean
  cpu: CpuInfo
  memory: MemoryInfo
  gpus: Array<{
    hardware: GpuHardware
    role: GpuRole
    local_index: number
  }>
  interconnect: GpuInterconnect | null
  os: string
  container: { runtime: string; is_container: boolean } | null
  software: SoftwareInfo
}

interface ParsedTopology {
  model: string
  schema_version: string
  nodes: ParsedNode[]
  total_gpus: number
  trainer_total_gpus: number
  inference_total_gpus: number
  trainer_config: {
    data_parallel_size: number
    megatron_tensor_parallel_size: number
    megatron_pipeline_parallel_size: number
    backend: string
    world_size: number
  } | null
  inference_config: {
    num_servers: number
    tensor_parallel_size: number
    placement_strategy: string
  } | null
}

// ============================================================================
// Setup JSON Parser
// ============================================================================

export function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

export function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string") {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function asString(value: unknown): string {
  if (typeof value === "string") return value
  return ""
}

function asBool(value: unknown): boolean {
  return !!value
}

export function parseSetupJson(
  setupValue: unknown,
): Record<string, unknown> | null {
  if (typeof setupValue !== "string") return null
  let current: unknown = setupValue.trim()
  if (!current) return null
  for (let depth = 0; depth < 2; depth += 1) {
    if (typeof current !== "string") break
    try {
      current = JSON.parse(current)
    } catch {
      return null
    }
  }
  return asObject(current)
}

export function parseTopology(
  summary: Record<string, unknown> | undefined,
): ParsedTopology | null {
  if (!summary) return null
  const setup = parseSetupJson(summary.setup)
  if (!setup) return null

  const model = asString(setup.model)
  const schema_version = asString(setup.schema_version)

  const cluster = asObject(setup.cluster)
  if (!cluster) return null

  const clusterNodes = Array.isArray(cluster.nodes) ? cluster.nodes : []

  // Build maps from trainer/inference assignments
  const trainer = asObject(setup.trainer)
  const inference = asObject(setup.inference)

  // Map: `${node_id}-${gpu_index}` -> role info
  const gpuRoleMap = new Map<string, GpuRole>()

  const trainerNodes = Array.isArray(trainer?.nodes) ? trainer.nodes : []
  for (const tn of trainerNodes) {
    const tNode = asObject(tn)
    if (!tNode) continue
    const nodeId = asNumber(tNode.node_id)
    if (nodeId === null) continue
    const gpus = Array.isArray(tNode.gpus) ? tNode.gpus : []
    for (const g of gpus) {
      const gpu = asObject(g)
      if (!gpu) continue
      const gpuIndex = asNumber(gpu.gpu_index)
      if (gpuIndex === null) continue
      gpuRoleMap.set(`${nodeId}-${gpuIndex}`, {
        role: "trainer",
        gpu_index: gpuIndex,
        rank: asNumber(gpu.rank) ?? undefined,
        dp_rank: asNumber(gpu.dp_rank) ?? undefined,
        tp_rank: asNumber(gpu.tp_rank) ?? undefined,
        pp_rank: asNumber(gpu.pp_rank) ?? undefined,
      })
    }
  }

  const inferenceNodes = Array.isArray(inference?.nodes) ? inference.nodes : []
  for (const iNode of inferenceNodes) {
    const inNode = asObject(iNode)
    if (!inNode) continue
    const nodeId = asNumber(inNode.node_id)
    if (nodeId === null) continue
    const gpus = Array.isArray(inNode.gpus) ? inNode.gpus : []
    for (const g of gpus) {
      const gpu = asObject(g)
      if (!gpu) continue
      const gpuIndex = asNumber(gpu.gpu_index)
      if (gpuIndex === null) continue
      gpuRoleMap.set(`${nodeId}-${gpuIndex}`, {
        role: "inference",
        gpu_index: gpuIndex,
        server_idx: asNumber(gpu.server_idx) ?? undefined,
        tp_group_id: asNumber(gpu.tp_group_id) ?? undefined,
        port: asNumber(gpu.port) ?? undefined,
        url: asString(gpu.url) || undefined,
      })
    }
  }

  // Parse nodes
  const nodes: ParsedNode[] = []
  for (const cn of clusterNodes) {
    const node = asObject(cn)
    if (!node) continue
    const nodeId = asNumber(node.node_id) ?? 0
    const hw = asObject(node.hardware)

    const cpuRaw = asObject(hw?.cpu)
    const cpu: CpuInfo = {
      model: asString(cpuRaw?.model),
      cores: asNumber(cpuRaw?.cores) ?? 0,
      logical_cores: asNumber(cpuRaw?.logical_cores) ?? 0,
      sockets: asNumber(cpuRaw?.sockets) ?? 1,
      architecture: asString(cpuRaw?.architecture),
      frequency_mhz: asNumber(cpuRaw?.frequency_mhz) ?? 0,
    }

    const memRaw = asObject(hw?.memory)
    const memory: MemoryInfo = {
      total_gb: asNumber(memRaw?.total_gb) ?? 0,
    }

    const gpuSection = asObject(hw?.gpu)
    const gpuDevices = Array.isArray(gpuSection?.devices)
      ? gpuSection!.devices
      : []
    const gpus = gpuDevices.map((d: unknown, i: number) => {
      const dev = asObject(d)
      const gpuIndex = i
      const key = `${nodeId}-${gpuIndex}`
      const role = gpuRoleMap.get(key) ?? {
        role: "unassigned" as const,
        gpu_index: gpuIndex,
      }

      return {
        hardware: {
          name: asString(dev?.name),
          memory_total_gb: asNumber(dev?.memory_total_gb) ?? 0,
          pcie_gen: asNumber(dev?.pcie_gen) ?? 0,
          pcie_width: asNumber(dev?.pcie_width) ?? 0,
          serial: asString(dev?.serial),
          uuid: asString(dev?.uuid),
        },
        role,
        local_index: gpuIndex,
      }
    })

    const interconnectRaw = asObject(gpuSection?.interconnect)
    const interconnect: GpuInterconnect | null = interconnectRaw
      ? {
          nvlink: asBool(interconnectRaw.nvlink),
          topology: asString(interconnectRaw.topology),
          links_per_gpu: asNumber(interconnectRaw.links_per_gpu) ?? 0,
          speed_gbps: asNumber(interconnectRaw.speed_gbps) ?? 0,
          connections: asNumber(interconnectRaw.connections) ?? 0,
        }
      : null

    const sw = asObject(node.software)
    const osRaw = asObject(sw?.os)
    const containerRaw = asObject(sw?.container)
    const pkgs = asObject(sw?.packages)

    nodes.push({
      node_id: nodeId,
      hostname: asString(node.hostname),
      ip: asString(node.ip),
      is_driver: asBool(node.is_driver),
      cpu,
      memory,
      gpus,
      interconnect,
      os: asString(osRaw?.platform),
      container: containerRaw
        ? {
            runtime: asString(containerRaw.runtime),
            is_container: asBool(containerRaw.is_container),
          }
        : null,
      software: {
        cuda: asString(pkgs?.cuda),
        pytorch: asString(pkgs?.pytorch),
        python: asString(pkgs?.python),
      },
    })
  }

  return {
    model,
    schema_version,
    nodes,
    total_gpus: asNumber(cluster.total_gpus) ?? 0,
    trainer_total_gpus: asNumber(trainer?.total_gpus) ?? 0,
    inference_total_gpus: asNumber(inference?.total_gpus) ?? 0,
    trainer_config: trainer
      ? {
          data_parallel_size: asNumber(trainer.data_parallel_size) ?? 0,
          megatron_tensor_parallel_size: asNumber(trainer.megatron_tensor_parallel_size) ?? 0,
          megatron_pipeline_parallel_size: asNumber(trainer.megatron_pipeline_parallel_size) ?? 0,
          backend: asString(trainer.backend),
          world_size: asNumber(trainer.world_size) ?? 0,
        }
      : null,
    inference_config: inference
      ? {
          num_servers: asNumber(inference.num_servers) ?? 0,
          tensor_parallel_size: asNumber(inference.tensor_parallel_size) ?? 0,
          placement_strategy: asString(inference.placement_strategy),
        }
      : null,
  }
}

// ============================================================================
// Color Constants — Light theme
// ============================================================================

const COLORS = {
  // Keep only two accent colors; everything else is black/white.
  trainer: "#2563eb",
  inference: "#dc2626",
  unassigned: "#111111",
  nvlink: "#c7d5f0",
  nvlinkDot: "#f0f0f0",
  platform: "#ffffff",
  platformBorder: "#111111",
  gpuBody: "#ffffff",
  gpuBodyHover: "#ffffff",
  moduleBody: "#ffffff",
  moduleBodyHover: "#ffffff",
  chip: "#ffffff",
  chipBorder: "#ededed",
  textBlack: "#000000",
  textDark: "#000000",
  textMuted: "#000000",
  textLight: "#000000",
  background: "#ffffff",
  gridColor: "#ffffff",
}

// ============================================================================
// 3D Sub-Components
// ============================================================================

/** Parse GPU name into brand line + model line for the roof label */
function splitGpuName(name: string): { brand: string; model: string } {
  // e.g. "NVIDIA A100-SXM4-80GB" → brand: "NVIDIA", model: "A100 SXM4"
  const cleaned = name
    .replace(/-\d+GB$/i, "") // strip trailing memory like -80GB
    .replace(/-/g, " ") // dashes to spaces
  const parts = cleaned.split(/\s+/)
  if (parts.length <= 1) return { brand: "", model: cleaned }
  const brand = parts[0] // "NVIDIA", "AMD", etc.
  const model = parts.slice(1).join(" ")
  return { brand, model }
}

/** A single GPU card rendered as a 3D rounded box */
function GpuCard({
  position,
  gpu,
  nodeId,
  onHover,
  onUnhover,
  onClick,
  isHovered,
}: {
  position: [number, number, number]
  gpu: ParsedNode["gpus"][0]
  nodeId: number
  onHover: () => void
  onUnhover: () => void
  onClick: () => void
  isHovered: boolean
}) {
  const roleColor =
    gpu.role.role === "trainer"
      ? COLORS.trainer
      : gpu.role.role === "inference"
        ? COLORS.inference
        : COLORS.unassigned

  const roleLabel =
    gpu.role.role === "trainer"
      ? "Trainer"
      : gpu.role.role === "inference"
        ? "Inference"
        : ""

  const { brand, model } = splitGpuName(gpu.hardware.name)
  const roofY = 0.095 // just above the surface

  return (
    <group position={position}>
      {/* Main GPU body */}
      <RoundedBox
        args={[0.95, 0.18, 0.6]}
        radius={0}
        smoothness={4}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation()
          onHover()
        }}
        onPointerOut={onUnhover}
        onClick={(e: ThreeEvent<MouseEvent>) => {
          e.stopPropagation()
          onClick()
        }}
      >
        <meshBasicMaterial
          color={isHovered ? COLORS.gpuBodyHover : COLORS.gpuBody}
        />
        <Edges color={COLORS.platformBorder} threshold={1} scale={1.002} />
      </RoundedBox>

      {/* ═══ Roof labels — all rotated flat (XZ plane), readable from above ═══ */}

      {/* Role indicator — small colored square + label, top-left area of roof */}
      {gpu.role.role !== "unassigned" && (
        <>
          {/* Colored square dot */}
          <mesh
            position={[-0.34, roofY - 0.003, -0.2]}
            rotation={[-Math.PI / 2, 0, 0]}
          >
            <planeGeometry args={[0.055, 0.055]} />
            <meshStandardMaterial color={roleColor} side={THREE.DoubleSide} />
          </mesh>
          {/* Role text next to the dot */}
          <Text
            position={[-0.285, roofY, -0.2]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.04}
            color={COLORS.textBlack}
            anchorX="left"
            anchorY="middle"
          >
            {roleLabel}
          </Text>
        </>
      )}

      {/* Brand name (e.g. "NVIDIA") — centered upper area */}
      <Text
        position={[0, roofY, -0.04]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.055}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
        fontWeight={700}
      >
        {brand}
      </Text>

      {/* Model name (e.g. "A100 SXM4") — centered, below brand */}
      <Text
        position={[0, roofY, 0.07]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.07}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
        fontWeight={700}
      >
        {model}
      </Text>

      {/* GPU index — bottom-right corner of roof */}
      <Text
        position={[0.4, roofY, 0.2]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.032}
        color={COLORS.textBlack}
        anchorX="right"
        anchorY="middle"
      >
        {`GPU ${gpu.local_index}`}
      </Text>

      {/* Rank / Server index — bottom-left corner of roof */}
      {gpu.role.rank !== undefined && (
        <Text
          position={[-0.4, roofY, 0.2]}
          rotation={[-Math.PI / 2, 0, 0]}
          fontSize={0.032}
          color={COLORS.textBlack}
          anchorX="left"
          anchorY="middle"
        >
          {`Rank ${gpu.role.rank}`}
        </Text>
      )}
      {gpu.role.server_idx !== undefined && (
        <Text
          position={[-0.4, roofY, 0.2]}
          rotation={[-Math.PI / 2, 0, 0]}
          fontSize={0.032}
          color={COLORS.textBlack}
          anchorX="left"
          anchorY="middle"
        >
          {`Server ${gpu.role.server_idx}`}
        </Text>
      )}

      {/* Hover tooltip */}
      {isHovered && (
        <Html
          position={[0, 0.35, 0]}
          center
          distanceFactor={5}
          style={{ pointerEvents: "none" }}
        >
          <div className="bg-white border border-black rounded-none px-3 py-2 min-w-[180px]">
            <div className="text-xs font-semibold text-black mb-1">
              {gpu.hardware.name}
            </div>
            <div className="text-[10px] text-black space-y-0.5">
              <div className="flex justify-between gap-4">
                <span>VRAM</span>
                <span className="font-mono">
                  {gpu.hardware.memory_total_gb} GB
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>PCIe</span>
                <span className="font-mono">
                  Gen{gpu.hardware.pcie_gen} x{gpu.hardware.pcie_width}
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Role</span>
                <span className="font-semibold" style={{ color: roleColor }}>
                  {gpu.role.role.charAt(0).toUpperCase() +
                    gpu.role.role.slice(1)}
                  {gpu.role.rank !== undefined && ` (rank ${gpu.role.rank})`}
                  {gpu.role.server_idx !== undefined &&
                    ` (server ${gpu.role.server_idx})`}
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Node</span>
                <span className="font-mono">#{nodeId}</span>
              </div>
            </div>
          </div>
        </Html>
      )}
    </group>
  )
}

/** VRAM module — separate 3D object placed in front of a GPU */
function VramModule({
  position,
  memoryGb,
  isHovered,
  onHover,
  onUnhover,
}: {
  position: [number, number, number]
  memoryGb: number
  isHovered: boolean
  onHover: () => void
  onUnhover: () => void
}) {
  // VRAM box: same width as GPU column, shorter height, shallower depth
  const roofY = 0.063

  return (
    <group position={position}>
      <RoundedBox
        args={[0.85, 0.12, 0.35]}
        radius={0}
        smoothness={4}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation()
          onHover()
        }}
        onPointerOut={onUnhover}
      >
        <meshBasicMaterial
          color={isHovered ? COLORS.moduleBodyHover : COLORS.moduleBody}
        />
        <Edges color={COLORS.platformBorder} threshold={1} scale={1.002} />
      </RoundedBox>

      {/* "VRAM" label on roof — rotated flat */}
      <Text
        position={[0, roofY, -0.06]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.035}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
      >
        VRAM
      </Text>

      {/* Memory amount on roof — rotated flat */}
      <Text
        position={[0, roofY, 0.05]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.055}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
        fontWeight={700}
      >
        {`${memoryGb}GB`}
      </Text>
    </group>
  )
}

/** Shorten CPU model: "AMD EPYC 7513 32-Core Processor" → "EPYC 7513" */
function shortCpuModel(model: string): string {
  // Remove common prefixes/suffixes
  const cleaned = model
    .replace(/\b(AMD|Intel|ARM)\b/gi, "")
    .replace(/\d+-Core Processor/gi, "")
    .replace(/\bProcessor\b/gi, "")
    .replace(/\(R\)|\(TM\)/gi, "")
    .replace(/\s+/g, " ")
    .trim()
  return cleaned || model
}

/** CPU chip rendered as a square with pin-like details */
function CpuChip({
  position,
  cpu,
  isHovered,
  onHover,
  onUnhover,
  onClick,
}: {
  position: [number, number, number]
  cpu: CpuInfo
  isHovered: boolean
  onHover: () => void
  onUnhover: () => void
  onClick: () => void
}) {
  const modelLabel = shortCpuModel(cpu.model)

  return (
    <group position={position}>
      {/* CPU body */}
      <RoundedBox
        args={[0.6, 0.08, 0.6]}
        radius={0}
        smoothness={4}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation()
          onHover()
        }}
        onPointerOut={onUnhover}
        onClick={(e: ThreeEvent<MouseEvent>) => {
          e.stopPropagation()
          onClick()
        }}
      >
        <meshBasicMaterial
          color={isHovered ? COLORS.gpuBodyHover : COLORS.chip}
        />
        <Edges color={COLORS.platformBorder} threshold={1} scale={1.002} />
      </RoundedBox>

      {/* Heat spreader (inner square) */}
      <mesh position={[0, 0.041, 0]}>
        <boxGeometry args={[0.45, 0.005, 0.45]} />
        <meshBasicMaterial color={COLORS.platform} />
        <Edges color={COLORS.platformBorder} threshold={1} scale={1.002} />
      </mesh>

      {/* Model name — top line of roof */}
      <Text
        position={[0, 0.05, -0.12]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.032}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
      >
        {modelLabel}
      </Text>

      {/* "CPU" label — center of roof */}
      <Text
        position={[0, 0.05, 0.0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.045}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
      >
        CPU
      </Text>

      {/* Cores/threads — bottom of roof */}
      <Text
        position={[0, 0.05, 0.1]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.028}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
      >
        {`${cpu.cores}C / ${cpu.logical_cores}T`}
      </Text>

      {/* Hover tooltip */}
      {isHovered && (
        <Html
          position={[0, 0.3, 0]}
          center
          distanceFactor={5}
          style={{ pointerEvents: "none" }}
        >
          <div className="bg-white border border-black rounded-none px-3 py-2 min-w-[180px]">
            <div className="text-xs font-semibold text-black mb-1">
              {cpu.model}
            </div>
            <div className="text-[10px] text-black space-y-0.5">
              <div className="flex justify-between gap-4">
                <span>Cores</span>
                <span className="font-mono">{cpu.cores}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Threads</span>
                <span className="font-mono">{cpu.logical_cores}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Sockets</span>
                <span className="font-mono">{cpu.sockets}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Frequency</span>
                <span className="font-mono">
                  {(cpu.frequency_mhz / 1000).toFixed(2)} GHz
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Arch</span>
                <span className="font-mono">{cpu.architecture}</span>
              </div>
            </div>
          </div>
        </Html>
      )}
    </group>
  )
}

/** RAM module — same visual style as VRAM (rounded box with roof text) */
function RamModule({
  position,
  memory,
  isHovered,
  onHover,
  onUnhover,
  onClick,
}: {
  position: [number, number, number]
  memory: MemoryInfo
  isHovered: boolean
  onHover: () => void
  onUnhover: () => void
  onClick: () => void
}) {
  const roofY = 0.063

  return (
    <group position={position}>
      <RoundedBox
        args={[0.6, 0.12, 0.35]}
        radius={0}
        smoothness={4}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation()
          onHover()
        }}
        onPointerOut={onUnhover}
        onClick={(e: ThreeEvent<MouseEvent>) => {
          e.stopPropagation()
          onClick()
        }}
      >
        <meshBasicMaterial
          color={isHovered ? COLORS.moduleBodyHover : COLORS.moduleBody}
        />
        <Edges color={COLORS.platformBorder} threshold={1} scale={1.002} />
      </RoundedBox>

      {/* "RAM" label on roof — rotated flat */}
      <Text
        position={[0, roofY, -0.06]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.035}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
      >
        RAM
      </Text>

      {/* Memory amount on roof — rotated flat */}
      <Text
        position={[0, roofY, 0.05]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.045}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
        fontWeight={700}
      >
        {`${Math.round(memory.total_gb)}GB`}
      </Text>

      {isHovered && (
        <Html
          position={[0, 0.3, 0]}
          center
          distanceFactor={5}
          style={{ pointerEvents: "none" }}
        >
          <div className="bg-white border border-black rounded-none px-3 py-2 min-w-[140px]">
            <div className="text-xs font-semibold text-black mb-1">
              System Memory
            </div>
            <div className="text-[10px] text-black space-y-0.5">
              <div className="flex justify-between gap-4">
                <span>Total</span>
                <span className="font-mono">
                  {memory.total_gb.toFixed(0)} GB
                </span>
              </div>
            </div>
          </div>
        </Html>
      )}
    </group>
  )
}

/** NVLink connections — bus bar behind VRAMs with cable connectors */
function NvlinkConnections({
  vramPositions,
  vramDepth,
  interconnect,
}: {
  vramPositions: [number, number, number][]
  vramDepth: number
  interconnect: GpuInterconnect | null
}) {
  if (!interconnect?.nvlink || vramPositions.length < 2) return null

  // Sort by X to find leftmost/rightmost
  const sorted = [...vramPositions].sort((a, b) => a[0] - b[0])
  const leftX = sorted[0][0]
  const rightX = sorted[sorted.length - 1][0]
  const barCenterX = (leftX + rightX) / 2
  const barLength = rightX - leftX

  // Position the bus bar behind the VRAMs
  const vramBackEdge = vramPositions[0][2] - vramDepth / 2
  const barZ = vramBackEdge - 0.18
  const barY = 0.065

  return (
    <group>
      {/* Main NVLink bus bar */}
      <RoundedBox
        args={[barLength, 0.02, 0.035]}
        radius={0.002}
        smoothness={4}
        position={[barCenterX, barY, barZ]}
      >
        <meshBasicMaterial color={COLORS.nvlink} />
      </RoundedBox>

      {/* Connectors from each VRAM to the bus bar */}
      {vramPositions.map((pos, idx) => {
        const vramBack = pos[2] - vramDepth / 2
        const connLen = Math.abs(vramBack - barZ)
        const connCenterZ = (vramBack + barZ) / 2

        return (
          <group key={idx}>
            {/* Cable from VRAM to bus */}
            <RoundedBox
              args={[0.02, 0.02, connLen]}
              radius={0.001}
              smoothness={2}
              position={[pos[0], barY, connCenterZ]}
            >
              <meshBasicMaterial color={COLORS.nvlink} />
            </RoundedBox>
          </group>
        )
      })}

      {/* NVLink label — written on the ground behind the bar */}
      <Text
        position={[barCenterX, 0.035, barZ - 0.15]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.06}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
        fontWeight={700}
      >
        {`NVLink · ${interconnect.links_per_gpu} links/GPU · ${interconnect.speed_gbps} Gbps`}
      </Text>
    </group>
  )
}

/** A single node platform containing all components */
function NodePlatform({
  node,
  position,
  hoveredItem,
  setHoveredItem,
  setSelectedItem,
}: {
  node: ParsedNode
  position: [number, number, number]
  hoveredItem: string | null
  setHoveredItem: (id: string | null) => void
  setSelectedItem: (item: SelectedItem | null) => void
}) {
  // Layout (back → front): NVLink — VRAMs — GPUs — PCIe bar — CPU+RAM — software
  const gpuCount = node.gpus.length
  const gpuSpacing = 1.15
  const gpuRowWidth = gpuCount * gpuSpacing
  const platformWidth = Math.max(gpuRowWidth + 1.8, 3.5)
  const platformDepth = 3.2

  const gpuDepth = 0.6
  const vramDepth = 0.35
  const cpuSize = 0.6
  const rowGap = 0.06

  // Z positions (back → front): VRAMs — GPUs — PCIe bar — CPU
  const gpuZ = -0.15
  const vramZ = gpuZ - gpuDepth / 2 - rowGap - vramDepth / 2 // VRAMs behind GPUs
  const pcieBarZ = gpuZ + gpuDepth / 2 + 0.08 // PCIe bar just in front of GPUs
  const cpuZ = pcieBarZ + 0.4 + cpuSize / 2 // CPU in front of PCIe bar

  const gpuPositions: [number, number, number][] = node.gpus.map((_, i) => {
    const x = i * gpuSpacing - (gpuRowWidth - gpuSpacing) / 2
    return [x, 0.14, gpuZ] as [number, number, number]
  })
  const vramPositions: [number, number, number][] = node.gpus.map((_, i) => {
    const x = i * gpuSpacing - (gpuRowWidth - gpuSpacing) / 2
    return [x, 0.11, vramZ] as [number, number, number]
  })

  return (
    <group position={position}>
      {/* Platform base */}
      <RoundedBox
        args={[platformWidth, 0.06, platformDepth]}
        radius={0}
        smoothness={4}
      >
        <meshBasicMaterial color={COLORS.platform} />
        <Edges color={COLORS.platformBorder} threshold={1} scale={1.002} />
      </RoundedBox>

      {/* Node label — rotated flat, readable from above */}
      <Text
        position={[0, 0.04, -platformDepth / 2 + 0.15]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.09}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
      >
        {`Node ${node.node_id}`}
        {node.is_driver ? " (Driver)" : ""}
      </Text>
      <Text
        position={[0, 0.04, -platformDepth / 2 + 0.3]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.05}
        color={COLORS.textBlack}
        anchorX="center"
        anchorY="middle"
      >
        {`${node.hostname} · ${node.ip}`}
      </Text>

      {/* GPU Cards + VRAM Modules */}
      {node.gpus.map((gpu, i) => {
        const gp = gpuPositions[i]
        const wp: [number, number, number] = [
          position[0] + gp[0],
          position[1] + gp[1],
          position[2] + gp[2],
        ]
        return (
          <group key={i}>
            <GpuCard
              position={gp}
              gpu={gpu}
              nodeId={node.node_id}
              isHovered={
                hoveredItem === `gpu-${node.node_id}-${gpu.local_index}`
              }
              onHover={() =>
                setHoveredItem(`gpu-${node.node_id}-${gpu.local_index}`)
              }
              onUnhover={() => setHoveredItem(null)}
              onClick={() =>
                setSelectedItem({
                  type: "gpu",
                  nodeId: node.node_id,
                  gpuIndex: gpu.local_index,
                  data: gpu,
                  worldPosition: wp,
                })
              }
            />
            <VramModule
              position={vramPositions[i]}
              memoryGb={gpu.hardware.memory_total_gb}
              isHovered={
                hoveredItem === `vram-${node.node_id}-${gpu.local_index}`
              }
              onHover={() =>
                setHoveredItem(`vram-${node.node_id}-${gpu.local_index}`)
              }
              onUnhover={() => setHoveredItem(null)}
            />
          </group>
        )
      })}

      {/* NVLink Connections — behind VRAMs */}
      <NvlinkConnections
        vramPositions={vramPositions}
        vramDepth={vramDepth}
        interconnect={node.interconnect}
      />

      {/* CPU (centered) + RAM (next to it) — in front of PCIe bar */}
      {(() => {
        const cpuLocal: [number, number, number] = [0, 0.09, cpuZ]
        const ramLocal: [number, number, number] = [0.75, 0.11, cpuZ]
        const cpuWorld: [number, number, number] = [
          position[0] + cpuLocal[0],
          position[1] + cpuLocal[1],
          position[2] + cpuLocal[2],
        ]
        const ramWorld: [number, number, number] = [
          position[0] + ramLocal[0],
          position[1] + ramLocal[1],
          position[2] + ramLocal[2],
        ]
        return (
          <>
            <CpuChip
              position={cpuLocal}
              cpu={node.cpu}
              isHovered={hoveredItem === `cpu-${node.node_id}`}
              onHover={() => setHoveredItem(`cpu-${node.node_id}`)}
              onUnhover={() => setHoveredItem(null)}
              onClick={() =>
                setSelectedItem({
                  type: "cpu",
                  nodeId: node.node_id,
                  data: node.cpu,
                  worldPosition: cpuWorld,
                })
              }
            />
            <RamModule
              position={ramLocal}
              memory={node.memory}
              isHovered={hoveredItem === `mem-${node.node_id}`}
              onHover={() => setHoveredItem(`mem-${node.node_id}`)}
              onUnhover={() => setHoveredItem(null)}
              onClick={() =>
                setSelectedItem({
                  type: "ram",
                  nodeId: node.node_id,
                  data: node.memory,
                  worldPosition: ramWorld,
                })
              }
            />
          </>
        )
      })()}

      {/* PCIe bus bar — horizontal, in front of VRAMs, mirrors NVLink on opposite side */}
      {(() => {
        const pcieGen = node.gpus[0]?.hardware.pcie_gen ?? 0
        const pcieWidth = node.gpus[0]?.hardware.pcie_width ?? 0
        if (pcieGen === 0 || gpuCount < 1) return null

        const sorted = [...gpuPositions].sort((a, b) => a[0] - b[0])
        const leftX = sorted[0][0]
        const rightX = sorted[sorted.length - 1][0]
        const barCenterX = (leftX + rightX) / 2
        const barLength = rightX - leftX
        const barY = 0.065

        // Spine from PCIe bar forward to CPU
        const spineStartZ = pcieBarZ
        const spineEndZ = cpuZ - cpuSize / 2
        const spineLen = Math.abs(spineEndZ - spineStartZ)
        const spineMidZ = (spineStartZ + spineEndZ) / 2

        return (
          <group>
            {/* Horizontal PCIe bus bar — between GPUs and VRAMs */}
            <RoundedBox
              args={[barLength, 0.02, 0.035]}
              radius={0.002}
              smoothness={4}
              position={[barCenterX, barY, pcieBarZ]}
            >
              <meshBasicMaterial color={COLORS.chipBorder} />
            </RoundedBox>

            {/* Short connectors from the bar backward to each GPU */}
            {gpuPositions.map((pos, idx) => {
              const gpuFront = gpuZ + gpuDepth / 2
              const connLen = Math.abs(pcieBarZ - gpuFront)
              const connMidZ = (gpuFront + pcieBarZ) / 2

              return (
                <group key={idx}>
                  <RoundedBox
                    args={[0.02, 0.02, connLen]}
                    radius={0.001}
                    smoothness={2}
                    position={[pos[0], barY, connMidZ]}
                  >
                    <meshBasicMaterial color={COLORS.chipBorder} />
                  </RoundedBox>
                </group>
              )
            })}

            {/* Spine — from PCIe bar forward through VRAMs to CPU */}
            <RoundedBox
              args={[0.02, 0.02, spineLen]}
              radius={0.001}
              smoothness={2}
              position={[0, barY, spineMidZ]}
            >
              <meshBasicMaterial color={COLORS.chipBorder} />
            </RoundedBox>

            {/* PCIe label — rotated 90° along the spine, wraps if needed */}
            <Text
              position={[0.15, 0.035, spineMidZ]}
              rotation={[-Math.PI / 2, 0, -Math.PI / 2]}
              fontSize={0.06}
              maxWidth={spineLen - 0.1}
              color={COLORS.textBlack}
              anchorX="center"
              anchorY="middle"
              fontWeight={700}
              textAlign="center"
            >
              {`PCIe Gen${pcieGen} x${pcieWidth}`}
            </Text>
          </group>
        )
      })()}

      {/* Software stack — small text on the front edge of the platform */}
      {(() => {
        const parts: string[] = []
        if (node.software.cuda) parts.push(`CUDA ${node.software.cuda}`)
        if (node.software.pytorch)
          parts.push(`PyTorch ${node.software.pytorch}`)
        if (node.software.python) parts.push(`Python ${node.software.python}`)
        if (parts.length === 0) return null
        return (
          <Text
            position={[0, 0.035, platformDepth / 2 - 0.1]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.03}
            color={COLORS.textBlack}
            anchorX="center"
            anchorY="middle"
          >
            {parts.join(" · ")}
          </Text>
        )
      })()}
    </group>
  )
}

/** Floor */
function FloorGrid() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]}>
      <planeGeometry args={[50, 50]} />
      <meshBasicMaterial color={COLORS.background} />
    </mesh>
  )
}

// ============================================================================
// Floor Overview — NodeBox + FloorOverviewScene
// ============================================================================

/** A single node box for the floor overview */
function NodeBox({
  node,
  position,
  onClick,
}: {
  node: ParsedNode
  position: [number, number, number]
  onClick: () => void
}) {
  const [hovered, setHovered] = useState(false)
  const groupRef = useRef<THREE.Group>(null)

  // Reset cursor when component unmounts (e.g. clicking a node transitions views)
  useEffect(() => {
    return () => {
      document.body.style.cursor = "auto"
    }
  }, [])

  const trainerCount = node.gpus.filter((g) => g.role.role === "trainer").length
  const inferenceCount = node.gpus.filter(
    (g) => g.role.role === "inference",
  ).length
  const totalGpus = node.gpus.length

  const gpuSummary = (() => {
    const parts: string[] = [`${totalGpus} GPUs`]
    const roleParts: string[] = []
    if (trainerCount > 0) roleParts.push(`${trainerCount}T`)
    if (inferenceCount > 0) roleParts.push(`${inferenceCount}I`)
    if (roleParts.length > 0) parts.push(roleParts.join(" "))
    return parts.join(" · ")
  })()

  const boxWidth = 2.5
  const boxDepth = 1.5
  const platformHeight = 0.5
  const roofY = platformHeight / 2 + 0.005

  // Smooth hover elevation
  useFrame(() => {
    if (!groupRef.current) return
    const targetY = hovered ? position[1] + 0.06 : position[1]
    groupRef.current.position.y +=
      (targetY - groupRef.current.position.y) * 0.15
  })

  return (
    <group ref={groupRef} position={position}>
      {/* Platform base — thin tray like NodePlatform */}
      <RoundedBox
        args={[boxWidth, platformHeight, boxDepth]}
        radius={0}
        smoothness={4}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation()
          setHovered(true)
          document.body.style.cursor = "pointer"
        }}
        onPointerOut={() => {
          setHovered(false)
          document.body.style.cursor = "auto"
        }}
        onClick={(e: ThreeEvent<MouseEvent>) => {
          e.stopPropagation()
          document.body.style.cursor = "auto"
          onClick()
        }}
      >
        <meshBasicMaterial color={COLORS.platform} />
        <Edges
          color={hovered ? COLORS.trainer : COLORS.platformBorder}
          threshold={1}
          scale={1.002}
        />
      </RoundedBox>

      {/* Small GPU chip blocks sitting on the platform */}
      {node.gpus.map((gpu, i) => {
        const chipW = 0.18
        const chipGap = 0.06
        const totalChipW = totalGpus * chipW + (totalGpus - 1) * chipGap
        const chipX = -totalChipW / 2 + chipW / 2 + i * (chipW + chipGap)
        const roleColor =
          gpu.role.role === "trainer"
            ? COLORS.trainer
            : gpu.role.role === "inference"
              ? COLORS.inference
              : COLORS.unassigned
        return (
          <mesh key={i} position={[chipX, platformHeight / 2 + 0.025, -0.15]}>
            <boxGeometry args={[chipW, 0.04, 0.22]} />
            <meshBasicMaterial color={roleColor} />
          </mesh>
        )
      })}

      {/* Node ID + hostname */}
      <Text
        position={[-boxWidth / 2 + 0.15, roofY, -boxDepth / 2 + 0.15]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.12}
        color={COLORS.textBlack}
        anchorX="left"
        anchorY="middle"
        fontWeight={700}
      >
        {`Node ${node.node_id}`}
        {node.is_driver ? "  DRIVER" : ""}
      </Text>
      <Text
        position={[-boxWidth / 2 + 0.15, roofY, -boxDepth / 2 + 0.35]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.07}
        color={COLORS.textBlack}
        anchorX="left"
        anchorY="middle"
      >
        {node.hostname}
      </Text>

      {/* GPU summary */}
      <Text
        position={[-boxWidth / 2 + 0.15, roofY, 0.25]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.08}
        color={COLORS.textBlack}
        anchorX="left"
        anchorY="middle"
      >
        {gpuSummary}
      </Text>

      {/* CPU + RAM summary on the platform */}
      <Text
        position={[-boxWidth / 2 + 0.15, roofY, 0.45]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.06}
        color={COLORS.textBlack}
        anchorX="left"
        anchorY="middle"
      >
        {`${node.cpu.cores}C CPU · ${Math.round(node.memory.total_gb)}GB RAM`}
      </Text>
    </group>
  )
}

/** Floor overview scene — arranges NodeBoxes in a grid */
function FloorOverviewScene({
  topology,
  onSelectNode,
}: {
  topology: ParsedTopology
  onSelectNode: (nodeId: number) => void
}) {
  const nodeBoxW = 2.5
  const nodeBoxD = 1.5

  const nodePositions = useMemo(() => {
    const positions: [number, number, number][] = []
    const cols = Math.ceil(Math.sqrt(topology.nodes.length))
    const spacingX = 3.2
    const spacingZ = 2.2

    topology.nodes.forEach((_, i) => {
      const row = Math.floor(i / cols)
      const col = i % cols
      const totalCols = Math.min(cols, topology.nodes.length - row * cols)
      const offsetX = (-(totalCols - 1) * spacingX) / 2
      // ground top is at y=0.03; node box center = 0.03 + gap + platformHeight/2
      positions.push([col * spacingX + offsetX, 0.30, row * spacingZ])
    })
    return positions
  }, [topology.nodes])

  // Compute a ground platform that fits all node boxes with padding
  const groundDims = useMemo(() => {
    if (nodePositions.length === 0) return { w: 4, d: 3, cx: 0, cz: 0 }
    let minX = Infinity,
      maxX = -Infinity,
      minZ = Infinity,
      maxZ = -Infinity
    for (const [x, , z] of nodePositions) {
      minX = Math.min(minX, x - nodeBoxW / 2)
      maxX = Math.max(maxX, x + nodeBoxW / 2)
      minZ = Math.min(minZ, z - nodeBoxD / 2)
      maxZ = Math.max(maxZ, z + nodeBoxD / 2)
    }
    const pad = 0.6
    return {
      w: maxX - minX + pad * 2,
      d: maxZ - minZ + pad * 2,
      cx: (minX + maxX) / 2,
      cz: (minZ + maxZ) / 2,
    }
  }, [nodePositions])

  return (
    <>
      <color attach="background" args={[COLORS.background]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[8, 12, 5]} intensity={0.8} />
      <directionalLight position={[-5, 8, -5]} intensity={0.4} />
      <hemisphereLight color="#ffffff" groundColor="#ffffff" intensity={0.5} />

      <FloorGrid />
      <gridHelper
        args={[50, 50, COLORS.gridColor, COLORS.gridColor]}
        position={[0, -0.04, 0]}
      />

      {/* Ground platform — big white rectangle with black borders */}
      <RoundedBox
        args={[groundDims.w, 0.06, groundDims.d]}
        radius={0}
        smoothness={4}
        position={[groundDims.cx, 0, groundDims.cz]}
      >
        <meshBasicMaterial color={COLORS.platform} />
        <Edges color={COLORS.platformBorder} threshold={1} scale={1.002} />
      </RoundedBox>

      {topology.nodes.map((node, i) => (
        <NodeBox
          key={node.node_id}
          node={node}
          position={nodePositions[i]}
          onClick={() => onSelectNode(node.node_id)}
        />
      ))}
    </>
  )
}

// ============================================================================
// Selected Item Types
// ============================================================================

interface SelectedItem {
  type: "gpu" | "cpu" | "ram"
  nodeId: number
  gpuIndex?: number
  data: unknown
  worldPosition: [number, number, number]
}

// ============================================================================
// Camera Controller — smooth zoom to focused item
// ============================================================================

function CameraController({
  focusTarget,
  overviewPosition,
  overviewLookAt,
}: {
  focusTarget: { position: [number, number, number] } | null
  overviewPosition: [number, number, number]
  overviewLookAt: [number, number, number]
}) {
  const { camera } = useThree()
  const controlsRef = useThree(
    (state) => state.controls as unknown as OrbitControlsImpl | null,
  )
  const targetPos = useRef(new THREE.Vector3(...overviewPosition))
  const targetLookAt = useRef(new THREE.Vector3(...overviewLookAt))
  const animating = useRef(false)
  const prevFocusKey = useRef<string | null>(null)
  const prevOverviewKey = useRef(overviewPosition.join(","))

  // Detect overview position changes (view transition) → snap camera instantly
  const overviewKey = overviewPosition.join(",")
  useEffect(() => {
    prevOverviewKey.current = overviewKey
    if (!focusTarget && controlsRef) {
      camera.position.set(...overviewPosition)
      const ctrl = controlsRef as unknown as { target: THREE.Vector3 }
      ctrl.target.set(...overviewLookAt)
      controlsRef.update()
      animating.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overviewKey])

  // Detect focus changes → kick off animation
  const currentKey = focusTarget ? focusTarget.position.join(",") : null

  useEffect(() => {
    prevFocusKey.current = currentKey

    if (focusTarget) {
      // Focusing in → animate to the object
      const [fx, fy, fz] = focusTarget.position
      targetPos.current.set(fx + 0.8, fy + 1.2, fz + 1.4)
      targetLookAt.current.set(fx, fy, fz)
      animating.current = true
    } else {
      animating.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentKey])

  useFrame(() => {
    if (!controlsRef || !animating.current) return

    const speed = 0.06
    camera.position.lerp(targetPos.current, speed)
    const ctrl = controlsRef as unknown as { target: THREE.Vector3 }
    ctrl.target.lerp(targetLookAt.current, speed)
    controlsRef.update()

    // Stop once close enough — let OrbitControls take over
    const posDist = camera.position.distanceTo(targetPos.current)
    const lookDist = ctrl.target.distanceTo(targetLookAt.current)
    if (posDist < 0.01 && lookDist < 0.01) {
      camera.position.copy(targetPos.current)
      ctrl.target.copy(targetLookAt.current)
      controlsRef.update()
      animating.current = false
    }
  })

  return null
}

// ============================================================================
// Info Panel (2D overlay)
// ============================================================================

/** Row helper for detail cards */
function DetailRow({
  label,
  value,
  mono = true,
}: {
  label: string
  value: React.ReactNode
  mono?: boolean
}) {
  return (
    <div className="flex justify-between gap-4">
      <span>{label}</span>
      <span className={mono ? "font-mono" : ""}>{value}</span>
    </div>
  )
}

function InfoPanel({
  topology,
  focusedItem,
  onClose,
}: {
  topology: ParsedTopology
  focusedItem: SelectedItem | null
  onClose: () => void
}) {
  return (
    <div className="absolute top-4 right-4 w-72 max-h-[calc(100%-2rem)] overflow-auto">
      {/* When focused — show full detail card only */}
      {focusedItem ? (
        <div className="bg-white border border-black rounded-none p-4">
          {/* GPU detail */}
          {focusedItem.type === "gpu" &&
            (() => {
              const gpu = focusedItem.data as ParsedNode["gpus"][0]
              const roleColor =
                gpu.role.role === "trainer"
                  ? COLORS.trainer
                  : gpu.role.role === "inference"
                    ? COLORS.inference
                    : COLORS.unassigned
              return (
                <div className="space-y-1 text-[11px] text-black">
                  <div className="text-xs font-semibold text-black mb-2">
                    GPU Details
                  </div>
                  <div className="font-mono text-black text-xs mb-1">
                    {gpu.hardware.name}
                  </div>
                  <DetailRow
                    label="Node / Index"
                    value={`#${focusedItem.nodeId} / GPU ${gpu.local_index}`}
                  />
                  <DetailRow
                    label="Role"
                    value={
                      <span
                        className="font-semibold"
                        style={{ color: roleColor }}
                      >
                        {gpu.role.role.charAt(0).toUpperCase() +
                          gpu.role.role.slice(1)}
                      </span>
                    }
                    mono={false}
                  />
                  {gpu.role.rank !== undefined && (
                    <DetailRow label="Rank" value={gpu.role.rank} />
                  )}
                  {gpu.role.dp_rank !== undefined && (
                    <DetailRow
                      label="DP / TP / PP"
                      value={`${gpu.role.dp_rank} / ${gpu.role.tp_rank} / ${gpu.role.pp_rank}`}
                    />
                  )}
                  {gpu.role.server_idx !== undefined && (
                    <DetailRow label="Server" value={gpu.role.server_idx} />
                  )}
                  {gpu.role.url && (
                    <DetailRow
                      label="URL"
                      value={<span className="text-[9px]">{gpu.role.url}</span>}
                    />
                  )}
                  <div className="border-t border-black mt-1.5 pt-1.5" />
                  <DetailRow
                    label="VRAM"
                    value={`${gpu.hardware.memory_total_gb} GB`}
                  />
                  <DetailRow
                    label="PCIe"
                    value={`Gen${gpu.hardware.pcie_gen} x${gpu.hardware.pcie_width}`}
                  />
                  <DetailRow
                    label="Serial"
                    value={
                      <span className="text-[9px]">{gpu.hardware.serial}</span>
                    }
                  />
                  <DetailRow
                    label="UUID"
                    value={
                      <span className="text-[9px] break-all">
                        {gpu.hardware.uuid}
                      </span>
                    }
                  />
                </div>
              )
            })()}

          {/* CPU detail */}
          {focusedItem.type === "cpu" &&
            (() => {
              const cpu = focusedItem.data as CpuInfo
              return (
                <div className="space-y-1 text-[11px] text-black">
                  <div className="text-xs font-semibold text-black mb-2">
                    CPU Details
                  </div>
                  <div className="font-mono text-black text-xs mb-1">
                    {cpu.model}
                  </div>
                  <DetailRow label="Node" value={`#${focusedItem.nodeId}`} />
                  <DetailRow label="Architecture" value={cpu.architecture} />
                  <DetailRow label="Cores" value={cpu.cores} />
                  <DetailRow label="Logical Cores" value={cpu.logical_cores} />
                  <DetailRow label="Sockets" value={cpu.sockets} />
                  <DetailRow
                    label="Frequency"
                    value={`${(cpu.frequency_mhz / 1000).toFixed(2)} GHz`}
                  />
                </div>
              )
            })()}

          {/* RAM detail */}
          {focusedItem.type === "ram" &&
            (() => {
              const mem = focusedItem.data as MemoryInfo
              return (
                <div className="space-y-1 text-[11px] text-black">
                  <div className="text-xs font-semibold text-black mb-2">
                    System Memory
                  </div>
                  <DetailRow label="Node" value={`#${focusedItem.nodeId}`} />
                  <DetailRow
                    label="Total"
                    value={`${mem.total_gb.toFixed(0)} GB`}
                  />
                </div>
              )
            })()}

          {/* Close button */}
          <button
            onClick={onClose}
            className="mt-4 w-full py-1.5 text-xs font-medium text-black border border-black rounded-none hover:bg-black hover:text-white transition-colors"
          >
            Close
          </button>
        </div>
      ) : (
        <>
          {/* Summary card */}
          <div className="bg-white border border-black rounded-none p-4 mb-3">
            <div className="text-xs font-semibold text-black mb-2">
              Cluster Overview
            </div>
            <div className="space-y-1 text-[11px] text-black">
              <DetailRow
                label="Model"
                value={
                  <span className="text-black">{topology.model || "—"}</span>
                }
                mono={false}
              />
              <DetailRow label="Nodes" value={topology.nodes.length} />
              <DetailRow label="Total GPUs" value={topology.total_gpus} />
              {topology.trainer_config && (
                <>
                  <div
                    className="border-t border-black mt-2 pt-2 mb-1 text-[10px] font-semibold uppercase tracking-wider"
                    style={{ color: COLORS.trainer }}
                  >
                    Trainer
                  </div>
                  <DetailRow label="GPUs" value={topology.trainer_total_gpus} />
                  <DetailRow
                    label="Backend"
                    value={topology.trainer_config.backend}
                  />
                  <DetailRow
                    label="DP × TP × PP"
                    value={`${topology.trainer_config.data_parallel_size} × ${topology.trainer_config.megatron_tensor_parallel_size} × ${topology.trainer_config.megatron_pipeline_parallel_size}`}
                  />
                </>
              )}
              {topology.inference_config && (
                <>
                  <div
                    className="border-t border-black mt-2 pt-2 mb-1 text-[10px] font-semibold uppercase tracking-wider"
                    style={{ color: COLORS.inference }}
                  >
                    Inference
                  </div>
                  <DetailRow
                    label="GPUs"
                    value={topology.inference_total_gpus}
                  />
                  <DetailRow
                    label="Servers"
                    value={topology.inference_config.num_servers}
                  />
                  <DetailRow
                    label="TP Size"
                    value={topology.inference_config.tensor_parallel_size}
                  />
                  <DetailRow
                    label="Placement"
                    value={topology.inference_config.placement_strategy}
                  />
                </>
              )}
            </div>
          </div>

          {/* Legend */}
          <div className="bg-white border border-black rounded-none p-3">
            <div className="text-[10px] font-semibold text-black mb-1.5">
              Legend
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-[10px] text-black">
                <div
                  className="w-3 h-2"
                  style={{ background: COLORS.trainer }}
                />
                Trainer GPU
              </div>
              <div className="flex items-center gap-2 text-[10px] text-black">
                <div
                  className="w-3 h-2"
                  style={{ background: COLORS.inference }}
                />
                Inference GPU
              </div>
              <div className="flex items-center gap-2 text-[10px] text-black">
                <div
                  className="w-3 h-2"
                  style={{ background: COLORS.unassigned }}
                />
                Unassigned GPU
              </div>
              <div className="flex items-center gap-2 text-[10px] text-black">
                <div
                  className="w-3 h-1"
                  style={{ background: COLORS.nvlink }}
                />
                NVLink
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// ============================================================================
// Main 3D Scene
// ============================================================================

function Scene({
  topology,
  hoveredItem,
  setHoveredItem,
  setSelectedItem,
  focusedItem,
}: {
  topology: ParsedTopology
  hoveredItem: string | null
  setHoveredItem: (id: string | null) => void
  setSelectedItem: (item: SelectedItem | null) => void
  focusedItem: SelectedItem | null
}) {
  // Layout nodes in a grid
  const nodePositions = useMemo(() => {
    const positions: [number, number, number][] = []
    const cols = Math.ceil(Math.sqrt(topology.nodes.length))
    const spacingX = 6.5
    const spacingZ = 3.5

    topology.nodes.forEach((_, i) => {
      const row = Math.floor(i / cols)
      const col = i % cols
      const totalCols = Math.min(cols, topology.nodes.length - row * cols)
      const offsetX = (-(totalCols - 1) * spacingX) / 2
      positions.push([col * spacingX + offsetX, 0, row * spacingZ])
    })
    return positions
  }, [topology.nodes])

  return (
    <>
      {/* Scene background — ensures the world is light, not black */}
      <color attach="background" args={[COLORS.background]} />

      {/* Lighting — bright and even for white/light theme */}
      <ambientLight intensity={0.7} />
      <directionalLight position={[8, 12, 5]} intensity={0.8} />
      <directionalLight position={[-5, 8, -5]} intensity={0.4} />
      <hemisphereLight color="#ffffff" groundColor="#ffffff" intensity={0.5} />

      {/* Floor */}
      <FloorGrid />
      <gridHelper
        args={[50, 50, COLORS.gridColor, COLORS.gridColor]}
        position={[0, -0.04, 0]}
      />

      {/* Nodes */}
      {topology.nodes.map((node, i) => (
        <NodePlatform
          key={node.node_id}
          node={node}
          position={nodePositions[i]}
          hoveredItem={focusedItem ? null : hoveredItem}
          setHoveredItem={focusedItem ? () => {} : setHoveredItem}
          setSelectedItem={focusedItem ? () => {} : setSelectedItem}
        />
      ))}
    </>
  )
}

// ============================================================================
// Exported TopologyViewer Component
// ============================================================================

export function TopologyViewer({ topology }: { topology: ParsedTopology }) {
  const [hoveredItem, setHoveredItem] = useState<string | null>(null)
  const [focusedItem, setFocusedItem] = useState<SelectedItem | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null)

  const isMultiNode = topology.nodes.length > 1
  const showFloor = isMultiNode && selectedNodeId === null

  // When a node is selected (or single-node), create a filtered topology
  const detailTopology = useMemo<ParsedTopology>(() => {
    if (!isMultiNode) return topology
    if (selectedNodeId === null) return topology
    const node = topology.nodes.find((n) => n.node_id === selectedNodeId)
    if (!node) return topology
    return {
      ...topology,
      nodes: [node],
    }
  }, [topology, selectedNodeId, isMultiNode])

  // Compute camera position based on number of nodes
  const overviewPosition = useMemo<[number, number, number]>(() => {
    const t = showFloor ? topology : detailTopology
    const nodeCount = t.nodes.length
    const maxGpus = Math.max(...t.nodes.map((n) => n.gpus.length), 1)
    if (showFloor) {
      // Higher/further camera for floor overview
      const distance = Math.max(4, nodeCount * 1.5)
      return [distance * 0.6, distance * 0.9, distance * 0.6]
    }
    const distance = Math.max(3.5, nodeCount * 1.8, maxGpus * 0.8)
    return [distance * 0.8, distance * 0.7, distance * 0.8]
  }, [topology, detailTopology, showFloor])

  const overviewLookAt = useMemo<[number, number, number]>(() => [0, 0, 0], [])

  const handleSelect = useCallback((item: SelectedItem | null) => {
    setFocusedItem(item)
    if (item) setHoveredItem(null)
  }, [])

  const handleClose = useCallback(() => {
    setFocusedItem(null)
  }, [])

  const handleBackToCluster = useCallback(() => {
    setSelectedNodeId(null)
    setFocusedItem(null)
    setHoveredItem(null)
  }, [])

  const handleSelectNode = useCallback((nodeId: number) => {
    setSelectedNodeId(nodeId)
    setFocusedItem(null)
    setHoveredItem(null)
  }, [])

  const isFocused = focusedItem !== null

  return (
    <div className="relative w-full h-full bg-white">
      <Canvas
        camera={{
          position: overviewPosition,
          fov: 45,
          near: 0.1,
          far: 100,
        }}
        gl={{ antialias: true, alpha: false }}
        onCreated={({ gl }) => {
          gl.setClearColor(COLORS.background)
          gl.toneMapping = THREE.NoToneMapping
        }}
      >
        {showFloor ? (
          <FloorOverviewScene
            topology={topology}
            onSelectNode={handleSelectNode}
          />
        ) : (
          <Scene
            topology={detailTopology}
            hoveredItem={hoveredItem}
            setHoveredItem={setHoveredItem}
            setSelectedItem={handleSelect}
            focusedItem={focusedItem}
          />
        )}
        <CameraController
          focusTarget={
            !showFloor && focusedItem
              ? { position: focusedItem.worldPosition }
              : null
          }
          overviewPosition={overviewPosition}
          overviewLookAt={overviewLookAt}
        />
        <OrbitControls
          makeDefault
          enablePan
          enableZoom
          enableRotate
          minDistance={1}
          maxDistance={30}
          maxPolarAngle={Math.PI / 2 - 0.05}
          onStart={() => {
            if (focusedItem) handleClose()
          }}
        />
      </Canvas>

      {/* Back to cluster button — shown when viewing a single node in multi-node topology */}
      {isMultiNode && selectedNodeId !== null && (
        <button
          onClick={handleBackToCluster}
          className="absolute top-4 left-4 px-3 py-1.5 text-xs font-medium text-black bg-white border border-black rounded-none hover:bg-black hover:text-white transition-colors select-none"
        >
          ← Back to cluster
        </button>
      )}

      {/* 2D Overlay Panel */}
      <InfoPanel
        topology={showFloor ? topology : detailTopology}
        focusedItem={focusedItem}
        onClose={handleClose}
      />

      {/* Controls hint */}
      <div className="absolute bottom-4 left-4 text-[10px] text-black/60 select-none">
        {showFloor
          ? "Click a node to inspect · Drag to orbit · Scroll to zoom"
          : isFocused
            ? "Move to dismiss · Click Close to return"
            : "Drag to orbit · Scroll to zoom · Right-click to pan · Click an object to focus"}
      </div>
    </div>
  )
}
