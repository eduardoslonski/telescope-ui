import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ArrowLeft, ChevronLeft, ChevronRight, Search, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useLogs, useLogsSummary } from "@/hooks/use-run-data"
import { cn } from "@/lib/utils"

// ============================================================================
// Level colors
// ============================================================================

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: "text-zinc-500",
  INFO: "text-blue-400",
  WARNING: "text-yellow-400",
  ERROR: "text-red-400",
  CRITICAL: "text-red-500 font-bold",
}

const COMPONENT_COLORS: Record<string, string> = {
  orchestrator: "text-fuchsia-400",
  trainer: "text-green-400",
  inference: "text-blue-400",
}

// ============================================================================
// Filter pill component
// ============================================================================

function FilterPill({
  label,
  options,
  selected,
  onChange,
}: {
  label: string
  options: string[]
  selected: string[]
  onChange: (selected: string[]) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [])

  const toggle = (val: string) => {
    if (selected.includes(val)) {
      onChange(selected.filter((s) => s !== val))
    } else {
      onChange([...selected, val])
    }
  }

  const allSelected = selected.length === 0
  const displayLabel = allSelected ? label : `${label} (${selected.length})`

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "px-2 py-0.5 text-xs rounded border transition-colors",
          allSelected
            ? "border-zinc-700 text-zinc-400 hover:border-zinc-500"
            : "border-blue-600 text-blue-400 bg-blue-950/30"
        )}
      >
        {displayLabel}
      </button>
      {open && options.length > 0 && (
        <div className="absolute top-full left-0 mt-1 z-50 bg-zinc-900 border border-zinc-700 rounded shadow-lg py-1 min-w-[120px]">
          {options.map((opt) => (
            <button
              key={opt}
              onClick={() => toggle(opt)}
              className={cn(
                "block w-full text-left px-3 py-1 text-xs hover:bg-zinc-800 transition-colors",
                selected.includes(opt) ? "text-blue-400" : "text-zinc-400"
              )}
            >
              {selected.includes(opt) ? "* " : "  "}
              {opt}
            </button>
          ))}
          {selected.length > 0 && (
            <button
              onClick={() => onChange([])}
              className="block w-full text-left px-3 py-1 text-xs text-zinc-500 hover:bg-zinc-800 border-t border-zinc-700 mt-1 pt-1"
            >
              Clear
            </button>
          )}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Main component
// ============================================================================

interface LogsViewerProps {
  runPath: string
  onBack: () => void
}

export function LogsViewer({ runPath, onBack }: LogsViewerProps) {
  const [page, setPage] = useState(0)
  const [selectedComponents, setSelectedComponents] = useState<string[]>([])
  const [selectedLevels, setSelectedLevels] = useState<string[]>([])
  const [selectedSources, setSelectedSources] = useState<string[]>([])
  const [searchInput, setSearchInput] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null)

  const filters = useMemo(
    () => ({
      components: selectedComponents.length > 0 ? selectedComponents : undefined,
      levels: selectedLevels.length > 0 ? selectedLevels : undefined,
      sources: selectedSources.length > 0 ? selectedSources : undefined,
      search: searchQuery || undefined,
    }),
    [selectedComponents, selectedLevels, selectedSources, searchQuery]
  )

  const { data: summaryData } = useLogsSummary(runPath, true)
  const { data: logsData, isFetching } = useLogs(runPath, page, filters, true)

  // Reset page when filters change
  useEffect(() => {
    setPage(0)
  }, [selectedComponents, selectedLevels, selectedSources, searchQuery])

  // Debounced search
  const handleSearchChange = useCallback((value: string) => {
    setSearchInput(value)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => setSearchQuery(value), 300)
  }, [])

  const formatTimestamp = (ts: number) => {
    const d = new Date(ts * 1000)
    return d.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      fractionalSecondDigits: 3,
    } as Intl.DateTimeFormatOptions)
  }

  const logs = logsData?.logs ?? []
  const totalPages = logsData?.total_pages ?? 1
  const total = logsData?.total ?? 0

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-zinc-800 bg-zinc-950/50">
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2"
          onClick={onBack}
        >
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back
        </Button>
        <span className="text-sm font-medium text-zinc-300">Logs</span>
        <span className="text-xs text-zinc-500">
          {total.toLocaleString()} records
        </span>
        {isFetching && (
          <span className="text-xs text-zinc-600 animate-pulse">
            loading...
          </span>
        )}
      </div>

      {/* Filters bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-zinc-800 bg-zinc-950/30">
        <FilterPill
          label="Component"
          options={summaryData?.components ?? []}
          selected={selectedComponents}
          onChange={setSelectedComponents}
        />
        <FilterPill
          label="Level"
          options={summaryData?.levels ?? []}
          selected={selectedLevels}
          onChange={setSelectedLevels}
        />
        <FilterPill
          label="Source"
          options={summaryData?.sources ?? []}
          selected={selectedSources}
          onChange={setSelectedSources}
        />
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-zinc-500" />
          <input
            type="text"
            placeholder="Search messages..."
            value={searchInput}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="w-full pl-6 pr-6 py-1 text-xs bg-zinc-900 border border-zinc-700 rounded text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500"
          />
          {searchInput && (
            <button
              onClick={() => {
                setSearchInput("")
                setSearchQuery("")
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2"
            >
              <X className="h-3 w-3 text-zinc-500 hover:text-zinc-300" />
            </button>
          )}
        </div>
      </div>

      {/* Log content */}
      <div ref={scrollRef} className="flex-1 overflow-auto font-mono text-[11px] leading-[18px]">
        {logs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-zinc-500 text-sm">
            {total === 0 ? "No logs available" : "No logs match filters"}
          </div>
        ) : (
          <table className="w-full">
            <tbody>
              {logs.map((entry, i) => (
                <tr
                  key={`${page}-${i}`}
                  className="hover:bg-zinc-900/50 align-top"
                >
                  <td className="px-2 py-px text-zinc-600 whitespace-nowrap select-none">
                    {formatTimestamp(entry.timestamp)}
                  </td>
                  <td
                    className={cn(
                      "px-1 py-px whitespace-nowrap w-[60px] select-none",
                      LEVEL_COLORS[entry.level] ?? "text-zinc-400"
                    )}
                  >
                    {entry.level.substring(0, 5).padEnd(5)}
                  </td>
                  <td
                    className={cn(
                      "px-1 py-px whitespace-nowrap w-[90px] select-none",
                      COMPONENT_COLORS[entry.component] ?? "text-zinc-400"
                    )}
                  >
                    {entry.component}
                  </td>
                  <td className="px-1 py-px text-zinc-500 whitespace-nowrap w-[55px] select-none">
                    {entry.source}
                  </td>
                  <td className="px-2 py-px text-zinc-300 whitespace-pre-wrap break-all">
                    {entry.message}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-1.5 border-t border-zinc-800 bg-zinc-950/50">
          <span className="text-xs text-zinc-500">
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2"
              disabled={page === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
            >
              <ChevronLeft className="h-3 w-3" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2"
              disabled={page >= totalPages - 1}
              onClick={() => setPage((p) => p + 1)}
            >
              <ChevronRight className="h-3 w-3" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
