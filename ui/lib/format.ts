// Formatting utilities for telescope visualization

/**
 * Format an absolute unix timestamp for chart axes, adapting precision to range.
 * - short windows (<10m): HH:MM:SS
 * - longer windows: HH:MM
 */
export function formatClockTimeAdaptive(
  timestampSeconds: number,
  rangeSeconds: number
): string {
  const showSeconds = Math.abs(rangeSeconds) < 600
  return new Date(timestampSeconds * 1000).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    ...(showSeconds ? { second: "2-digit" as const } : {}),
    hour12: false,
  })
}

/**
 * Format relative time in seconds (e.g., "30s")
 */
export function formatTimeShort(seconds: number): string {
  return `${seconds.toFixed(0)}s`
}

/**
 * Format a numeric value with smart precision.
 * Integers: no decimals (with locale formatting).
 * >= 1: 1 decimal.
 * >= 0.01: 2 decimals.
 * Very small: enough decimals to reveal the first significant digit.
 * e.g. 42 → "42", 130.1 → "130.1", 0.57 → "0.57", 0.0004 → "0.0004"
 */
export function formatValueSmart(v: number): string {
  if (Number.isInteger(v)) return v.toLocaleString()
  const abs = Math.abs(v)
  if (abs === 0) return "0.0"
  if (abs >= 1) return v.toFixed(1)
  if (abs >= 0.01) return v.toFixed(2)
  const decimals = Math.max(2, Math.ceil(-Math.log10(abs)) + 1)
  return v.toFixed(decimals)
}

/**
 * Format a seconds value with 1 decimal by default,
 * but extend precision for very small numbers so significant digits are visible.
 * e.g. 130.1 → "130.1", 0.0004 → "0.0004", 0.03 → "0.03"
 */
function formatSecondsSmart(seconds: number): string {
  const abs = Math.abs(seconds)
  if (abs === 0) return "0.00"
  if (abs >= 1) return seconds.toFixed(1)
  if (abs >= 0.01) return seconds.toFixed(2)
  // For very small numbers, show enough decimals to reveal the first significant digit
  const decimals = Math.max(2, Math.ceil(-Math.log10(abs)) + 1)
  return seconds.toFixed(decimals)
}

/**
 * Format seconds into human-readable duration with at most two parts.
 * - < 60s: "45.1s"
 * - >= 60s < 3600s: "2m 10s"
 * - >= 3600s: "1h 30m"
 */
export function formatSecondsHuman(seconds: number): string {
  if (seconds < 60) {
    return `${formatSecondsSmart(seconds)}s`
  }
  const totalSec = Math.floor(seconds)
  const h = Math.floor(totalSec / 3600)
  const m = Math.floor((totalSec % 3600) / 60)
  const s = totalSec % 60
  if (h > 0) {
    return `${h}h ${m}m`
  }
  return `${m}m ${s}s`
}

/**
 * Format seconds for tooltip display (HTML).
 * When >= 60s, shows human-readable format with raw seconds in gray parentheses.
 * e.g. "2m 10s <span class='...'>(130.1s)</span>"
 */
export function formatSecondsTooltipHtml(seconds: number): string {
  const rawStr = `${formatSecondsSmart(seconds)}s`
  if (seconds < 60) {
    return rawStr
  }
  const human = formatSecondsHuman(seconds)
  return `${human} <span class="text-muted-foreground font-normal">(${rawStr})</span>`
}

/**
 * Format seconds into a compact single-unit label for axis ticks.
 * Uses the largest appropriate unit: "2.2m", "1.5h", or "45s".
 */
export function formatSecondsCompact(seconds: number): string {
  const abs = Math.abs(seconds)
  if (abs >= 3600) {
    const h = seconds / 3600
    return `${parseFloat(h.toFixed(1))}h`
  }
  if (abs >= 60) {
    const m = seconds / 60
    return `${parseFloat(m.toFixed(1))}m`
  }
  if (abs < 0.1 && abs !== 0) {
    return `${parseFloat((seconds * 1000).toFixed(1))}ms`
  }
  return `${parseFloat(seconds.toFixed(1))}s`
}

/**
 * Format seconds into a compact human-readable duration (e.g. 90 → "1m 30s").
 * Shows up to three parts: hours, minutes, seconds — omitting zero components.
 */
export function formatDurationHms(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0 && m > 0 && s > 0) return `${h}h ${m}m ${s}s`
  if (h > 0 && m > 0) return `${h}h ${m}m`
  if (h > 0 && s > 0) return `${h}h ${s}s`
  if (h > 0) return `${h}h`
  if (m > 0 && s > 0) return `${m}m ${s}s`
  if (m > 0) return `${m}m`
  return `${s}s`
}

/**
 * Format elapsed time from a date string to a human-readable short format
 * e.g., "1m", "2h", "3d", "2mo", "1y"
 * Minimum is 1m (1 minute)
 */
export function formatTimeAgo(dateString: string | null): string {
  if (!dateString) return ""
  
  // If the ISO string has no timezone indicator, treat it as UTC by appending "Z"
  let normalized = dateString
  if (!/[Zz]|[+-]\d{2}(:\d{2})?$/.test(dateString)) {
    normalized = dateString + "Z"
  }
  const created = new Date(normalized)
  const now = new Date()
  const diffMs = now.getTime() - created.getTime()
  
  const minutes = Math.floor(diffMs / (1000 * 60))
  const hours = Math.floor(diffMs / (1000 * 60 * 60))
  const days = Math.floor(diffMs / (1000 * 60 * 60 * 24))
  const months = Math.floor(days / 30)
  const years = Math.floor(days / 365)
  
  if (years >= 1) return `${years}y`
  if (months >= 1) return `${months}mo`
  if (days >= 1) return `${days}d`
  if (hours >= 1) return `${hours}h`
  // Minimum is 1 minute
  return `${Math.max(1, minutes)}m`
}

