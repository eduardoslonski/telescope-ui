
import * as React from "react"
import { Toggle } from "@/components/ui/toggle"
import { cn } from "@/lib/utils"

type ToggleVariant = "default" | "outline" | "selecting"
type ToggleSize = "default" | "sm"

interface ToggleWithInputProps {
  label: string
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  value: string
  onValueChange: (value: string) => void
  onValueCommit: (value: string) => void
  inputType?: "number" | "text"
  inputMin?: number
  inputMax?: number
  inputWidth?: string
  icon?: React.ReactNode
  variant?: ToggleVariant
  size?: ToggleSize
  className?: string
}

export function ToggleWithInput({
  label,
  enabled,
  onEnabledChange,
  value,
  onValueChange,
  onValueCommit,
  inputType = "number",
  inputMin,
  inputMax,
  inputWidth = "w-12",
  icon,
  variant = "default",
  size = "default",
  className,
}: ToggleWithInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      if (!enabled) onEnabledChange(true)
      onValueCommit(value)
      e.currentTarget.blur()
    }
  }

  const handleBlur = () => {
    onValueCommit(value)
  }

  const isSelecting = variant === "selecting"

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-lg transition-colors",
        variant === "default" && "border border-input bg-transparent",
        variant === "default" && enabled && "bg-muted",
        variant === "outline" && "border border-input bg-transparent",
        variant === "outline" && enabled && "bg-muted",
        isSelecting && "bg-accent",
        isSelecting && enabled && "bg-primary",
        className
      )}
    >
      <Toggle
        variant={variant}
        size={size}
        pressed={enabled}
        onPressedChange={onEnabledChange}
        className={cn(
          "gap-1.5 border-0 rounded-r-none pr-1",
          size === "sm" && "text-xs px-2"
        )}
      >
        {icon}
        <span>{label}</span>
      </Toggle>
      <input
        type={inputType}
        min={inputMin}
        max={inputMax}
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        className={cn(
          "bg-transparent text-center font-medium outline-none rounded-r-lg transition-colors",
          inputWidth,
          size === "default" && "h-8 text-sm",
          size === "sm" && "h-7 text-xs",
          isSelecting
            ? enabled
              ? "text-secondary"
              : "text-muted-foreground/50"
            : enabled
              ? "text-foreground"
              : "text-muted-foreground/50"
        )}
      />
    </div>
  )
}

