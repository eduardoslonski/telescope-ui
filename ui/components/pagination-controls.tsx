
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
} from "lucide-react"

interface PaginationControlsProps {
  currentPage: number
  totalPages: number
  onPageChange: (page: number) => void
  disabled?: boolean
  pageValues?: number[]
}

export function PaginationControls({
  currentPage,
  totalPages,
  onPageChange,
  disabled = false,
  pageValues,
}: PaginationControlsProps) {
  const currentDisplayValue = pageValues?.[currentPage] ?? currentPage
  const minValue = pageValues?.[0] ?? 0
  const maxValue =
    pageValues && pageValues.length > 0
      ? pageValues[pageValues.length - 1]
      : Math.max(0, totalPages - 1)

  const [inputValue, setInputValue] = useState(currentDisplayValue.toString())

  // Sync input value when currentPage changes externally
  useEffect(() => {
    setInputValue(currentDisplayValue.toString())
  }, [currentDisplayValue])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      const value = parseInt(inputValue, 10)
      if (isNaN(value)) {
        // Reset to current page if invalid
        setInputValue(currentDisplayValue.toString())
        return
      }

      if (pageValues && pageValues.length > 0) {
        const index = pageValues.indexOf(value)
        if (index >= 0) {
          onPageChange(index)
        } else {
          // Reset to current page if invalid
          setInputValue(currentDisplayValue.toString())
        }
        return
      }

      if (value >= 0 && value < totalPages) {
        onPageChange(value)
      } else {
        // Reset to current page if invalid
        setInputValue(currentDisplayValue.toString())
      }
    }
  }

  const handleBlur = () => {
    // Reset to current page on blur without committing
    setInputValue(currentDisplayValue.toString())
  }

  return (
    <div className="flex items-center gap-0.5">
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={() => onPageChange(0)}
        disabled={disabled || currentPage === 0}
        title="First page"
      >
        <ChevronsLeft className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={() => onPageChange(currentPage - 1)}
        disabled={disabled || currentPage === 0}
        title="Previous page"
      >
        <ChevronLeft className="h-3.5 w-3.5" />
      </Button>
      <div className="flex items-center gap-1 text-xs px-1">
        <Input
          type="number"
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onBlur={handleBlur}
          disabled={disabled}
          className="w-12 h-6 text-center px-1 py-0 text-xs"
          min={minValue}
          max={maxValue}
        />
        <span className="text-muted-foreground">
          / {maxValue}
        </span>
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={() => onPageChange(currentPage + 1)}
        disabled={disabled || currentPage >= totalPages - 1}
        title="Next page"
      >
        <ChevronRight className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={() => onPageChange(totalPages - 1)}
        disabled={disabled || currentPage >= totalPages - 1}
        title="Last page"
      >
        <ChevronsRight className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}

