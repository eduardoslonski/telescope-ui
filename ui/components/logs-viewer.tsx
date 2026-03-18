import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ArrowLeft, ChevronLeft, ChevronRight, Search, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useLogs, useLogsSummary } from "@/hooks/use-run-data"
import { cn } from "@/lib/utils"

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: "text-muted-foreground",
  INFO: "text-blue-600 dark:text-blue-400",
  WARNING: "text-yellow-600 dark:text-yellow-400",
  ERROR: "text-red-600 dark:text-red-400",
  CRITICAL: "text-red-700 dark:text-red-500 font-bold",
}

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
    <div className="flex h-full min-h-0 flex-col">
      {/* Header */}
      <div className="flex h-10 items-center gap-2 border-b px-2">
        <Button
          variant="outline"
          size="sm"
          className="h-7 px-2 text-xs"
          onClick={onBack}
        >
          <ArrowLeft className="mr-1 h-3.5 w-3.5" />
          Back
        </Button>
        <span className="text-xs font-medium">Logs</span>
        <span className="text-xs text-muted-foreground">
          {total.toLocaleString()} records
        </span>
        {isFetching && (
          <span className="text-xs text-muted-foreground animate-pulse">
            loading...
          </span>
        )}
      </div>

      {/* Filters bar */}
      <div className="flex h-10 items-center gap-2 border-b px-2">
        {(summaryData?.components.length ?? 0) > 0 && (
          <Select
            value={selectedComponent ?? ""}
            onValueChange={setSelectedComponent}
          >
            <SelectTrigger size="sm" className="text-xs h-7 gap-1">
              <SelectValue placeholder="Component" />
            </SelectTrigger>
            <SelectContent>
              {(summaryData?.components ?? []).map((c) => (
                <SelectItem key={c} value={c} className="text-xs">
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        {(summaryData?.sources.length ?? 0) > 0 && (
          <Select
            value={selectedSource ?? ""}
            onValueChange={setSelectedSource}
          >
            <SelectTrigger size="sm" className="text-xs h-7 gap-1">
              <SelectValue placeholder="Source" />
            </SelectTrigger>
            <SelectContent>
              {(summaryData?.sources ?? []).map((s) => (
                <SelectItem key={s} value={s} className="text-xs">
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        <div className="flex items-center gap-1">
          {(summaryData?.levels ?? []).map((level) => (
            <button
              key={level}
              onClick={() => toggleLevel(level)}
              className={cn(
                "px-1.5 py-0.5 text-[10px] rounded border transition-colors",
                selectedLevels.includes(level)
                  ? "border-blue-600 bg-blue-600/10"
                  : selectedLevels.length === 0
                    ? "border-border hover:border-ring"
                    : "border-border text-muted-foreground/50 hover:border-ring",
                LEVEL_COLORS[level] ?? "text-muted-foreground"
              )}
            >
              {level}
            </button>
          ))}
        </div>
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search messages..."
            value={searchInput}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="h-7 pl-7 pr-7 text-xs"
          />
          {searchInput && (
            <button
              onClick={() => {
                setSearchInput("")
                setSearchQuery("")
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2"
            >
              <X className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground" />
            </button>
          )}
        </div>
      </div>

      {/* Log content */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto font-mono text-[11px] leading-[18px]">
        {logs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
            {total === 0 ? "No logs available" : "No logs match filters"}
          </div>
        ) : (
          <table className="w-full">
            <tbody>
              {logs.map((entry, i) => (
                <tr
                  key={`${page}-${i}`}
                  className="hover:bg-muted/50 align-top"
                >
                  <td className="px-2 py-px text-muted-foreground whitespace-nowrap select-none">
                    {formatTimestamp(entry.timestamp)}
                  </td>
                  <td
                    className={cn(
                      "px-1 py-px whitespace-nowrap w-[60px] select-none",
                      LEVEL_COLORS[entry.level] ?? "text-muted-foreground"
                    )}
                  >
                    {entry.level.substring(0, 5).padEnd(5)}
                  </td>
                  <td className="px-2 py-px text-foreground whitespace-pre-wrap break-all">
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
        <div className="flex items-center justify-between px-2 border-t h-10">
          <span className="text-xs text-muted-foreground">
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex items-center gap-0.5">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              disabled={page === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              disabled={page >= totalPages - 1}
              onClick={() => setPage((p) => p + 1)}
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
