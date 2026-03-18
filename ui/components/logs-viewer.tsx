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
// Single-select dropdown
// ============================================================================

function SelectDropdown({
  label,
  options,
  value,
  onChange,
}: {
  label: string
  options: string[]
  value: string | null
  onChange: (value: string | null) => void
}) {
  return (
    <select
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value || null)}
      className="px-2 py-0.5 text-xs rounded border border-zinc-700 bg-zinc-900 text-zinc-300 focus:outline-none focus:border-zinc-500"
    >
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
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
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null)
  const [selectedSource, setSelectedSource] = useState<string | null>(null)
  const [selectedLevels, setSelectedLevels] = useState<string[]>([])
  const [searchInput, setSearchInput] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null)

  const { data: summaryData } = useLogsSummary(runPath, true)

  // Auto-select first component and source when summary loads
  useEffect(() => {
    if (summaryData) {
      if (selectedComponent === null && summaryData.components.length > 0) {
        setSelectedComponent(summaryData.components[0])
      }
      if (selectedSource === null && summaryData.sources.length > 0) {
        setSelectedSource(summaryData.sources[0])
      }
    }
  }, [summaryData, selectedComponent, selectedSource])

  const filters = useMemo(
    () => ({
      components: selectedComponent ? [selectedComponent] : undefined,
      levels: selectedLevels.length > 0 ? selectedLevels : undefined,
      sources: selectedSource ? [selectedSource] : undefined,
      search: searchQuery || undefined,
    }),
    [selectedComponent, selectedLevels, selectedSource, searchQuery]
  )

  const { data: logsData, isFetching } = useLogs(runPath, page, filters, true)

  // Reset page when filters change
  useEffect(() => {
    setPage(0)
  }, [selectedComponent, selectedLevels, selectedSource, searchQuery])

  // Debounced search
  const handleSearchChange = useCallback((value: string) => {
    setSearchInput(value)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => setSearchQuery(value), 300)
  }, [])

  const toggleLevel = (level: string) => {
    setSelectedLevels((prev) =>
      prev.includes(level) ? prev.filter((l) => l !== level) : [...prev, level]
    )
  }

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
      <div className="flex items-center gap-3 px-4 py-2 border-b border-zinc-800 bg-zinc-950/30">
        <SelectDropdown
          label="Component"
          options={summaryData?.components ?? []}
          value={selectedComponent}
          onChange={setSelectedComponent}
        />
        <SelectDropdown
          label="Source"
          options={summaryData?.sources ?? []}
          value={selectedSource}
          onChange={setSelectedSource}
        />
        <div className="flex items-center gap-1">
          {(summaryData?.levels ?? []).map((level) => (
            <button
              key={level}
              onClick={() => toggleLevel(level)}
              className={cn(
                "px-1.5 py-0.5 text-[10px] rounded border transition-colors",
                selectedLevels.includes(level)
                  ? "border-blue-600 bg-blue-950/30"
                  : selectedLevels.length === 0
                    ? "border-zinc-800 hover:border-zinc-600"
                    : "border-zinc-800 text-zinc-600 hover:border-zinc-600",
                LEVEL_COLORS[level] ?? "text-zinc-400"
              )}
            >
              {level}
            </button>
          ))}
        </div>
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
