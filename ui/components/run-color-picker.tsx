
import { useCallback, useEffect, useMemo, useState } from "react"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

type RgbColor = { r: number; g: number; b: number }
type HsvColor = { h: number; s: number; v: number }

const DEFAULT_COLOR = "#e67439"
const HUE_GRADIENT =
  "linear-gradient(to right, #ff0000 0%, #ffff00 17%, #00ff00 33%, #00ffff 50%, #0000ff 67%, #ff00ff 83%, #ff0000 100%)"
const PRESET_COLORS = [
  "#865ed6",
  "#f07fdd",
  "#e67439",
  "#87cec0",
  "#dc4cdc",
  "#ffb83e",
  "#5ac5db",
  "#239487",
  "#fab796",
  "#9bc750",
  "#ad6f51",
  "#c2337a",
  "#a2a9ad",
  "#538ae6",
  "#f1444f",
  "#479a60",
]

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)))
}

function rgbToHex({ r, g, b }: RgbColor): string {
  const toHex = (n: number) => n.toString(16).padStart(2, "0")
  return `#${toHex(clampByte(r))}${toHex(clampByte(g))}${toHex(clampByte(b))}`
}

function hexToRgb(hexColor: string): RgbColor | null {
  const normalized = hexColor.trim()
  const short = /^#([0-9a-f]{3})$/i.exec(normalized)
  if (short) {
    const [r, g, b] = short[1].split("")
    return {
      r: Number.parseInt(r + r, 16),
      g: Number.parseInt(g + g, 16),
      b: Number.parseInt(b + b, 16),
    }
  }

  const full = /^#([0-9a-f]{6})$/i.exec(normalized)
  if (!full) return null

  return {
    r: Number.parseInt(full[1].slice(0, 2), 16),
    g: Number.parseInt(full[1].slice(2, 4), 16),
    b: Number.parseInt(full[1].slice(4, 6), 16),
  }
}

function rgbToHsv({ r, g, b }: RgbColor): HsvColor {
  const rr = r / 255
  const gg = g / 255
  const bb = b / 255
  const max = Math.max(rr, gg, bb)
  const min = Math.min(rr, gg, bb)
  const delta = max - min

  let hue = 0
  if (delta !== 0) {
    if (max === rr) {
      hue = 60 * (((gg - bb) / delta) % 6)
    } else if (max === gg) {
      hue = 60 * ((bb - rr) / delta + 2)
    } else {
      hue = 60 * ((rr - gg) / delta + 4)
    }
  }

  if (hue < 0) hue += 360

  return {
    h: hue,
    s: max === 0 ? 0 : delta / max,
    v: max,
  }
}

function hsvToRgb({ h, s, v }: HsvColor): RgbColor {
  const hue = ((h % 360) + 360) % 360
  const sat = clamp01(s)
  const val = clamp01(v)

  const c = val * sat
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1))
  const m = val - c

  let rr = 0
  let gg = 0
  let bb = 0

  if (hue < 60) {
    rr = c
    gg = x
  } else if (hue < 120) {
    rr = x
    gg = c
  } else if (hue < 180) {
    gg = c
    bb = x
  } else if (hue < 240) {
    gg = x
    bb = c
  } else if (hue < 300) {
    rr = x
    bb = c
  } else {
    rr = c
    bb = x
  }

  return {
    r: clampByte((rr + m) * 255),
    g: clampByte((gg + m) * 255),
    b: clampByte((bb + m) * 255),
  }
}

function darkenHex(hexColor: string, amount: number): string {
  const rgb = hexToRgb(hexColor)
  if (!rgb) return "#000000"
  const factor = clamp01(1 - amount)
  return rgbToHex({
    r: rgb.r * factor,
    g: rgb.g * factor,
    b: rgb.b * factor,
  })
}

export function normalizeHexColor(color: string, fallback = DEFAULT_COLOR): string {
  const rgb = hexToRgb(color)
  if (!rgb) return fallback
  return rgbToHex(rgb)
}

type RunColorPickerProps = {
  value: string
  onChange: (color: string) => void
  className?: string
}

export function RunColorPicker({ value, onChange, className }: RunColorPickerProps) {
  const normalizedColor = useMemo(
    () => normalizeHexColor(value, DEFAULT_COLOR),
    [value]
  )
  const rgb = useMemo(() => hexToRgb(normalizedColor)!, [normalizedColor])
  const hsv = useMemo(() => rgbToHsv(rgb), [rgb])
  const [hexInput, setHexInput] = useState(normalizedColor.slice(1).toUpperCase())

  useEffect(() => {
    setHexInput(normalizedColor.slice(1).toUpperCase())
  }, [normalizedColor])

  const updateFromHsv = useCallback(
    (nextHsv: HsvColor) => {
      const color = rgbToHex(hsvToRgb(nextHsv))
      onChange(color)
    },
    [onChange]
  )

  const handleSvPointer = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const rect = event.currentTarget.getBoundingClientRect()
      const x = clamp01((event.clientX - rect.left) / rect.width)
      const y = clamp01((event.clientY - rect.top) / rect.height)
      updateFromHsv({ h: hsv.h, s: x, v: 1 - y })
    },
    [hsv.h, updateFromHsv]
  )

  const handleHuePointer = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const rect = event.currentTarget.getBoundingClientRect()
      const ratio = clamp01((event.clientX - rect.left) / rect.width)
      updateFromHsv({ h: ratio * 360, s: hsv.s, v: hsv.v })
    },
    [hsv.s, hsv.v, updateFromHsv]
  )

  const commitHexInput = useCallback(() => {
    const sanitized = hexInput.replace(/[^0-9a-f]/gi, "")
    const candidate = `#${sanitized}`
    const next = normalizeHexColor(candidate, normalizedColor)
    setHexInput(next.slice(1).toUpperCase())
    if (next !== normalizedColor) {
      onChange(next)
    }
  }, [hexInput, normalizedColor, onChange])

  const handleRgbChange = useCallback(
    (channel: keyof RgbColor, rawValue: string) => {
      const parsed = Number.parseInt(rawValue, 10)
      if (Number.isNaN(parsed)) return
      const nextRgb: RgbColor = { ...rgb, [channel]: clampByte(parsed) }
      onChange(rgbToHex(nextRgb))
    },
    [onChange, rgb]
  )

  return (
    <div className={cn("w-[196px] space-y-1.5 text-[10px]", className)}>
      <div
        className="relative h-28 w-full cursor-crosshair overflow-hidden rounded-md border border-border touch-none select-none"
        style={{ backgroundColor: `hsl(${hsv.h.toFixed(0)} 100% 50%)` }}
        onPointerDown={(event) => {
          event.preventDefault()
          event.currentTarget.setPointerCapture(event.pointerId)
          handleSvPointer(event)
        }}
        onPointerMove={(event) => {
          if (!event.currentTarget.hasPointerCapture(event.pointerId)) return
          handleSvPointer(event)
        }}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId)
          }
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent" />
        <div className="absolute inset-0 bg-gradient-to-t from-black to-transparent" />
        <div
          className="pointer-events-none absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,0.25)]"
          style={{ left: `${hsv.s * 100}%`, top: `${(1 - hsv.v) * 100}%` }}
        />
      </div>

      <div
        className="relative h-2 w-full cursor-ew-resize rounded-md border border-border touch-none select-none"
        style={{ background: HUE_GRADIENT }}
        onPointerDown={(event) => {
          event.preventDefault()
          event.currentTarget.setPointerCapture(event.pointerId)
          handleHuePointer(event)
        }}
        onPointerMove={(event) => {
          if (!event.currentTarget.hasPointerCapture(event.pointerId)) return
          handleHuePointer(event)
        }}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId)
          }
        }}
      >
        <div
          className="pointer-events-none absolute top-1/2 h-2.5 w-1.5 -translate-y-1/2 rounded border border-white shadow-[0_0_0_1px_rgba(0,0,0,0.35)]"
          style={{ left: `calc(${(hsv.h / 360) * 100}% - 3px)` }}
        />
      </div>

      <div className="grid grid-cols-[1fr_48px] items-end gap-1">
        <div className="space-y-0.5">
          <label className="text-[8px] uppercase tracking-wide text-muted-foreground">
            Hex
          </label>
          <Input
            value={hexInput}
            onChange={(event) => {
              setHexInput(event.target.value.replace(/[^0-9a-f]/gi, "").slice(0, 6).toUpperCase())
            }}
            onBlur={commitHexInput}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault()
                commitHexInput()
              }
            }}
            className="h-6 px-1.5 font-sans !text-[11px] md:!text-[11px] uppercase"
            aria-label="Hex color"
          />
        </div>
        <div
          className="h-6 rounded-md border border-border"
          style={{ backgroundColor: normalizedColor }}
          aria-label="Selected color preview"
        />
      </div>

      <div className="grid grid-cols-3 gap-1">
        {(
          [
            ["R", rgb.r, "r"],
            ["G", rgb.g, "g"],
            ["B", rgb.b, "b"],
          ] as const
        ).map(([label, val, channel]) => (
          <div key={label} className="space-y-0.5">
            <label className="text-[8px] uppercase tracking-wide text-muted-foreground">
              {label}
            </label>
            <Input
              type="number"
              min={0}
              max={255}
              value={val}
              onChange={(event) => handleRgbChange(channel, event.target.value)}
              className="h-6 px-1.5 !text-[11px] md:!text-[11px]"
              aria-label={`${label} channel`}
            />
          </div>
        ))}
      </div>

      <div className="grid grid-cols-8 gap-x-0.5 gap-y-1.5 pt-0.5">
        {PRESET_COLORS.map((presetColor) => (
          <button
            key={presetColor}
            type="button"
            className={cn(
              "h-4 w-4 rounded-[3px] border transition",
              normalizedColor === presetColor
                ? "ring-1 ring-foreground/70 ring-offset-0"
                : "hover:brightness-[0.97]"
            )}
            style={{
              backgroundColor: presetColor,
              borderColor: darkenHex(presetColor, 0.2),
            }}
            onClick={() => onChange(presetColor)}
            aria-label={`Use color ${presetColor}`}
          />
        ))}
      </div>
    </div>
  )
}

