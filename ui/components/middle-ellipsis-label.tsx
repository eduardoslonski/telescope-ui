import { useCallback, useLayoutEffect, useRef, useState } from "react"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

type MiddleEllipsisParts = {
  prefix: string
  suffix: string
  isTruncated: boolean
}

const MIDDLE_ELLIPSIS = "..."

function getMiddleEllipsisParts(text: string, maxChars: number): MiddleEllipsisParts {
  if (!text || maxChars <= 0 || text.length <= maxChars) {
    return { prefix: text, suffix: "", isTruncated: false }
  }

  const available = maxChars - MIDDLE_ELLIPSIS.length
  if (available < 2) {
    return { prefix: text.slice(0, Math.max(1, maxChars)), suffix: "", isTruncated: false }
  }

  const desiredSuffix = 4
  let suffixLength =
    available <= desiredSuffix + 1 ? Math.max(1, Math.floor(available / 2)) : desiredSuffix
  suffixLength = Math.min(suffixLength, text.length - 1)

  let prefixLength = available - suffixLength
  if (prefixLength < 1) {
    prefixLength = 1
    suffixLength = Math.max(1, available - prefixLength)
  }

  return {
    prefix: text.slice(0, prefixLength),
    suffix: text.slice(-suffixLength),
    isTruncated: true,
  }
}

export function MiddleEllipsisLabel({
  text,
  className,
  tooltipClassName,
  ellipsisClassName,
  ellipsisFadeClassName,
  tooltipDelay = 300,
}: {
  text: string
  className?: string
  tooltipClassName?: string
  ellipsisClassName?: string
  ellipsisFadeClassName?: string
  tooltipDelay?: number
}) {
  const containerRef = useRef<HTMLSpanElement | null>(null)
  const measureRef = useRef<HTMLSpanElement | null>(null)
  const [parts, setParts] = useState<MiddleEllipsisParts>({
    prefix: text,
    suffix: "",
    isTruncated: false,
  })

  const updateParts = useCallback(() => {
    const container = containerRef.current
    const measure = measureRef.current
    if (!container || !measure) return

    const containerWidth = container.getBoundingClientRect().width
    const textWidth = measure.getBoundingClientRect().width
    if (!containerWidth || !textWidth) return

    if (textWidth <= containerWidth) {
      setParts((prev) =>
        prev.isTruncated || prev.prefix !== text
          ? { prefix: text, suffix: "", isTruncated: false }
          : prev
      )
      return
    }

    const avgCharWidth = textWidth / Math.max(1, text.length)
    const maxChars = Math.max(1, Math.floor(containerWidth / avgCharWidth))
    const nextParts = getMiddleEllipsisParts(text, maxChars)
    setParts((prev) => {
      if (
        prev.prefix === nextParts.prefix &&
        prev.suffix === nextParts.suffix &&
        prev.isTruncated === nextParts.isTruncated
      ) {
        return prev
      }
      return nextParts
    })
  }, [text])

  useLayoutEffect(() => {
    updateParts() // eslint-disable-line react-hooks/set-state-in-effect -- DOM measurement requires sync setState in layout effect
    const container = containerRef.current
    if (!container || typeof ResizeObserver === "undefined") return

    const observer = new ResizeObserver(() => {
      updateParts()
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [updateParts])

  const label = (
    <span
      ref={containerRef}
      className={cn(
        "relative flex w-full min-w-0 items-center overflow-hidden whitespace-nowrap",
        className
      )}
    >
      <span
        ref={measureRef}
        className="pointer-events-none absolute -z-10 opacity-0"
        aria-hidden="true"
      >
        {text || "M"}
      </span>
      {parts.isTruncated ? (
        <>
          <span className="shrink-0">{parts.prefix}</span>
          <span
            className={cn(
              "relative inline-flex items-center rounded-[2px] px-0.5 text-muted-foreground",
              "before:absolute before:inset-y-0 before:-left-2 before:w-2 before:bg-gradient-to-r before:from-transparent before:content-['']",
              ellipsisClassName ?? "bg-background",
              ellipsisFadeClassName ?? "before:to-background"
            )}
          >
            {MIDDLE_ELLIPSIS}
          </span>
          <span className="shrink-0">{parts.suffix}</span>
        </>
      ) : (
        <span className="truncate">{text}</span>
      )}
    </span>
  )

  if (!parts.isTruncated) {
    return label
  }

  return (
    <Tooltip delayDuration={tooltipDelay}>
      <TooltipTrigger asChild>{label}</TooltipTrigger>
      <TooltipContent
        side="top"
        className={cn("max-w-xs break-all text-[11px]", tooltipClassName)}
      >
        {text}
      </TooltipContent>
    </Tooltip>
  )
}
