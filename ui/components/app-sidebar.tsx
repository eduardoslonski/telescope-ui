import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useLocation, Link } from "react-router-dom"
import { useAtom, useAtomValue, useSetAtom } from "jotai"
import { useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Check,
  Menu,
  Moon,
  Sun,
  Trash2,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Spinner } from "@/components/ui/spinner"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import {
  selectedRunPathAtom,
  visibleRunsAtom,
  wandbApiKeyAtom,
  wandbConfigDialogOpenAtom,
  knownProjectsDialogOpenAtom,
  hoveredRunIdAtom,
  isSyncingAtom,
  isTrackingAtom,
  overviewShowCodeViewAtom,
  overviewShowLogsViewAtom,
  darkModeAtom,
} from "@/lib/atoms"
import { useRemovedRuns, useRuns, useKnownProjects } from "@/hooks/use-run-data"
import { cn } from "@/lib/utils"
import { API_BASE } from "@/lib/constants"
import { formatTimeAgo } from "@/lib/format"
import {
  getSidebarRunNameParts,
  SIDEBAR_MAX_RUN_NAME_CHARS as MAX_RUN_NAME_CHARS,
} from "@/lib/run-name"
import { RunConfigPanel } from "@/components/run-config-panel"
import {
  RunColorPicker,
  normalizeHexColor,
} from "@/components/run-color-picker"
import { DatabaseDialog } from "@/components/database-dialog"
import type { RemovedRun, RunsResponse } from "@/lib/types"

type RemovedRunSortKey =
  | "name"
  | "entity"
  | "project"
  | "run_id"
  | "created_at"
  | "removed_at"
type SortDir = "asc" | "desc"

function RunNameLabel({
  name,
  className: extraClassName,
  maxChars = MAX_RUN_NAME_CHARS,
}: {
  name: string
  className?: string
  maxChars?: number
}) {
  const textClass = extraClassName ?? "text-xs font-sans"
  const { isTruncated, prefix, suffix } = getSidebarRunNameParts(name, maxChars)

  if (!isTruncated) {
    return <span className={cn("truncate", textClass)}>{prefix}</span>
  }

  return (
    <Tooltip delayDuration={300}>
      <TooltipTrigger asChild>
        <span
          className={cn(
            "flex w-full min-w-0 items-center overflow-hidden whitespace-nowrap",
            textClass,
          )}
        >
          <span className="shrink-0">{prefix}</span>
          <span
            className="relative shrink-0 inline-flex items-center px-0.5 text-muted-foreground/50"
            style={{ backgroundColor: "var(--run-item-bg)" }}
          >
            <span
              className="pointer-events-none absolute inset-y-0 -left-3 w-3"
              style={{
                background:
                  "linear-gradient(to right, transparent, var(--run-item-bg))",
              }}
              aria-hidden="true"
            />
            ...
          </span>
          <span className="shrink-0">{suffix}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        className="max-w-56 break-all text-xs font-sans"
      >
        {name}
      </TooltipContent>
    </Tooltip>
  )
}

export function AppSidebar() {
  const { pathname } = useLocation()
  const queryClient = useQueryClient()
  const [selectedRunPath, setSelectedRunPath] = useAtom(selectedRunPathAtom)
  const [visibleRuns, setVisibleRuns] = useAtom(visibleRunsAtom)
  const [apiKey, setApiKey] = useAtom(wandbApiKeyAtom)
  const [configDialogOpen, setConfigDialogOpen] = useAtom(
    wandbConfigDialogOpenAtom,
  )
  const [darkMode, setDarkMode] = useAtom(darkModeAtom)
  const setHoveredRunId = useSetAtom(hoveredRunIdAtom)
  const setOverviewShowCodeView = useSetAtom(overviewShowCodeViewAtom)
  const setOverviewShowLogsView = useSetAtom(overviewShowLogsViewAtom)
  const [addRunDialogOpen, setAddRunDialogOpen] = useState(false)
  const [projectsDialogOpen, setProjectsDialogOpen] = useAtom(
    knownProjectsDialogOpenAtom,
  )
  const [newProjectInput, setNewProjectInput] = useState("")
  const [isAddingProject, setIsAddingProject] = useState(false)
  const [addProjectError, setAddProjectError] = useState<string | null>(null)
  const [removingProject, setRemovingProject] = useState<string | null>(null)
  const [databaseDialogOpen, setDatabaseDialogOpen] = useState(false)
  const [configApiKey, setConfigApiKey] = useState("")
  const [useNetrcKey, setUseNetrcKey] = useState(false)
  const [isConfiguring, setIsConfiguring] = useState(false)
  const [isRemovingRun, setIsRemovingRun] = useState<string | null>(null)
  const [removeConfirmRunId, setRemoveConfirmRunId] = useState<string | null>(
    null,
  )
  const [removedSortKey, setRemovedSortKey] =
    useState<RemovedRunSortKey>("removed_at")
  const [removedSortDir, setRemovedSortDir] = useState<SortDir>("desc")
  const [configViewRun, setConfigViewRun] = useState<RemovedRun | null>(null)
  const [resyncConfirmRunId, setResyncConfirmRunId] = useState<string | null>(
    null,
  )
  const [isResyncingRun, setIsResyncingRun] = useState<string | null>(null)
  const [isSyncingEvalsRun, setIsSyncingEvalsRun] = useState<string | null>(
    null,
  )
  const [showOnlySelected, setShowOnlySelected] = useState(false)
  const [runSearchQuery, setRunSearchQuery] = useState("")
  const [configSearchMatchIds, setConfigSearchMatchIds] = useState<
    string[] | null
  >(null)
  const [isSearchingConfig, setIsSearchingConfig] = useState(false)
  const [openColorPickerRunId, setOpenColorPickerRunId] = useState<
    string | null
  >(null)
  const [runColorDrafts, setRunColorDrafts] = useState<Record<string, string>>(
    {},
  )

  const setIsSyncing = useSetAtom(isSyncingAtom)
  const setIsTracking = useSetAtom(isTrackingAtom)

  const { data: runsData, isLoading: isLoadingRuns } = useRuns()
  const { data: removedRunsData, isLoading: isLoadingRemovedRuns } =
    useRemovedRuns(addRunDialogOpen)
  const { data: knownProjectsData, refetch: refetchKnownProjects } =
    useKnownProjects(projectsDialogOpen)

  // ---------------------------------------------------------------------------
  // Detect when a run's sync/tracking completes and invalidate cached queries
  // so pages don't show stale empty data after sync finishes.
  // ---------------------------------------------------------------------------
  const prevRunStatesRef = useRef<
    Map<string, { is_syncing: boolean; is_tracking: boolean }>
  >(new Map())

  useEffect(() => {
    if (!runsData?.runs) return

    const prevStates = prevRunStatesRef.current
    const completedRunIds: string[] = []

    for (const run of runsData.runs) {
      const prev = prevStates.get(run.run_id)
      // Detect sync completion (was syncing → now not)
      if (prev?.is_syncing && !run.is_syncing) {
        completedRunIds.push(run.run_id)
      }
      // Detect tracking stop (was tracking → now not)
      if (prev?.is_tracking && !run.is_tracking) {
        completedRunIds.push(run.run_id)
      }
      prevStates.set(run.run_id, {
        is_syncing: run.is_syncing,
        is_tracking: run.is_tracking,
      })
    }

    // Invalidate all queries that reference a run whose sync/tracking just finished
    if (completedRunIds.length > 0) {
      const uniqueIds = [...new Set(completedRunIds)]
      for (const runId of uniqueIds) {
        queryClient.invalidateQueries({
          predicate: (query) => {
            const key = query.queryKey
            if (!Array.isArray(key)) return false
            return key.some((k) => {
              if (k === runId) return true
              if (Array.isArray(k) && k.includes(runId)) return true
              return false
            })
          },
        })
      }
      // Also refresh the global runs & database queries
      queryClient.invalidateQueries({ queryKey: ["runs"] })
    }
  }, [runsData?.runs, queryClient])

  // Keep the global sync/tracking atoms in sync with the selected run's
  // server-reported state.  This ensures pages that gate polling behind
  // `isTracking || isSyncing` (Overview) start polling correctly
  // even when not initiated from the Overview page.
  useEffect(() => {
    if (!runsData?.runs || !selectedRunPath) return
    const selectedRun = runsData.runs.find((r) => r.run_id === selectedRunPath)
    if (selectedRun) {
      setIsSyncing(selectedRun.is_syncing)
      setIsTracking(selectedRun.is_tracking)
    }
  }, [runsData?.runs, selectedRunPath, setIsSyncing, setIsTracking])

  // Keep Overview's inline Code/Logs views from sticking when navigating away.
  useEffect(() => {
    if (pathname !== "/") {
      setOverviewShowCodeView(false)
      setOverviewShowLogsView(false)
    }
  }, [pathname, setOverviewShowCodeView, setOverviewShowLogsView])

  // Keep persisted run selection state consistent with server data.
  // This prevents stale localStorage values from showing a phantom run
  // after DB resets or run removals.
  useEffect(() => {
    if (!runsData?.runs) return
    const knownRunIds = new Set(runsData.runs.map((run) => run.run_id))

    setVisibleRuns((prev) => {
      const filtered = prev.filter((runId) => knownRunIds.has(runId))
      return filtered.length === prev.length ? prev : filtered
    })

    if (selectedRunPath && !knownRunIds.has(selectedRunPath)) {
      setSelectedRunPath(null)
      setIsSyncing(false)
      setIsTracking(false)
    }
  }, [
    runsData?.runs,
    selectedRunPath,
    setSelectedRunPath,
    setVisibleRuns,
    setIsSyncing,
    setIsTracking,
  ])

  const sortedRemovedRuns = useMemo(() => {
    const runs = removedRunsData?.runs ?? []
    if (runs.length === 0) return runs
    return [...runs].sort((a, b) => {
      const key = removedSortKey
      let aVal: string | null
      let bVal: string | null
      if (key === "run_id") {
        aVal = a.run_id.split("/").pop() ?? a.run_id
        bVal = b.run_id.split("/").pop() ?? b.run_id
      } else {
        aVal = a[key]
        bVal = b[key]
      }
      // Nulls always last
      if (aVal == null && bVal == null) return 0
      if (aVal == null) return 1
      if (bVal == null) return -1
      const cmp = aVal.localeCompare(bVal)
      return removedSortDir === "asc" ? cmp : -cmp
    })
  }, [removedRunsData?.runs, removedSortKey, removedSortDir])

  const toggleRemovedSort = (key: RemovedRunSortKey) => {
    if (removedSortKey === key) {
      setRemovedSortDir((d) => (d === "asc" ? "desc" : "asc"))
    } else {
      setRemovedSortKey(key)
      setRemovedSortDir("asc")
    }
  }

  const handleConfigure = async (e: React.FormEvent) => {
    e.preventDefault()

    const body = useNetrcKey
      ? { use_netrc: true }
      : { api_key: configApiKey.trim(), use_netrc: false }

    if (!useNetrcKey && !configApiKey.trim()) return

    setIsConfiguring(true)
    try {
      const response = await fetch(`${API_BASE}/wandb-config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || "Failed to configure W&B")
      }

      setApiKey(true)
      queryClient.invalidateQueries({ queryKey: ["runs"] })
      setConfigDialogOpen(false)
      setConfigApiKey("")
    } catch (error) {
      console.error("Failed to configure W&B:", error)
    } finally {
      setIsConfiguring(false)
    }
  }

  const handleAddRun = (runPath: string) => {
    // Close dialog and select run immediately for a snappy UX
    setAddRunDialogOpen(false)
    setSelectedRunPath(runPath)

    // Fire API call in the background
    fetch(`${API_BASE}/add-run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_path: runPath }),
    })
      .then((response) => {
        if (!response.ok) {
          return response.json().then((error) => {
            throw new Error(error.detail || "Failed to add run")
          })
        }
      })
      .then(() => {
        queryClient.invalidateQueries({ queryKey: ["runs"] })
        queryClient.invalidateQueries({ queryKey: ["removed-runs"] })
      })
      .catch((error) => {
        console.error("Failed to add run:", error)
      })
  }

  const handleAddProject = async () => {
    const project = newProjectInput.trim()
    if (!project) return
    setIsAddingProject(true)
    setAddProjectError(null)
    try {
      const response = await fetch(`${API_BASE}/add-project`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project }),
      })
      if (!response.ok) {
        const error = await response.json()
        setAddProjectError(error.detail || "Failed to add project")
        return
      }
      setNewProjectInput("")
      setProjectsDialogOpen(false)
      refetchKnownProjects()
      queryClient.invalidateQueries({ queryKey: ["runs"] })
    } catch (error) {
      setAddProjectError("Failed to add project")
    } finally {
      setIsAddingProject(false)
    }
  }

  const handleRemoveProject = async (project: string) => {
    setRemovingProject(project)
    try {
      const response = await fetch(`${API_BASE}/remove-project`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project }),
      })
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || "Failed to remove project")
      }
      refetchKnownProjects()
    } catch (error) {
      console.error("Failed to remove project:", error)
    } finally {
      setRemovingProject(null)
    }
  }

  const handleRemoveRun = async (runId: string) => {
    if (!runId) return
    setIsRemovingRun(runId)
    try {
      const response = await fetch(`${API_BASE}/remove-run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_path: runId }),
      })
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || "Failed to remove run")
      }

      setVisibleRuns((prev) => prev.filter((id) => id !== runId))
      if (selectedRunPath === runId) {
        setSelectedRunPath(null)
      }
      queryClient.invalidateQueries({ queryKey: ["runs"] })
      queryClient.invalidateQueries({ queryKey: ["removed-runs"] })
    } catch (error) {
      console.error("Failed to remove run:", error)
    } finally {
      setIsRemovingRun(null)
      setRemoveConfirmRunId(null)
    }
  }

  const toggleRunVisibility = (runId: string) => {
    setVisibleRuns((prev) =>
      prev.includes(runId)
        ? prev.filter((id) => id !== runId)
        : [...prev, runId],
    )
  }

  const updateRunColorInCache = useCallback(
    (runId: string, color: string) => {
      queryClient.setQueryData<RunsResponse>(["runs"], (prev) => {
        if (!prev) return prev
        return {
          ...prev,
          runs: prev.runs.map((run) =>
            run.run_id === runId ? { ...run, color } : run,
          ),
        }
      })
    },
    [queryClient],
  )

  const persistRunColor = useCallback(
    async (runId: string, nextColor: string, previousColor: string) => {
      const normalizedNext = normalizeHexColor(nextColor, previousColor)
      if (normalizedNext.toLowerCase() === previousColor.toLowerCase()) {
        return
      }

      updateRunColorInCache(runId, normalizedNext)

      try {
        const response = await fetch(`${API_BASE}/set-run-color`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            run_path: runId,
            color: normalizedNext,
          }),
        })
        if (!response.ok) {
          let detail = "Failed to update run color"
          try {
            const errorPayload = await response.json()
            detail = errorPayload.detail ?? detail
          } catch {
            // Keep generic message if response body is unavailable.
          }
          throw new Error(detail)
        }
      } catch (error) {
        console.error("Failed to update run color:", error)
        updateRunColorInCache(runId, previousColor)
      } finally {
        queryClient.invalidateQueries({ queryKey: ["runs"] })
      }
    },
    [queryClient, updateRunColorInCache],
  )

  const handleRunColorPickerOpenChange = useCallback(
    (runId: string, open: boolean, persistedColor: string) => {
      if (open) {
        setOpenColorPickerRunId(runId)
        setRunColorDrafts((prev) =>
          prev[runId] === persistedColor
            ? prev
            : { ...prev, [runId]: persistedColor },
        )
        return
      }

      setOpenColorPickerRunId((prev) => (prev === runId ? null : prev))
      const draftColor = runColorDrafts[runId]
      if (draftColor) {
        void persistRunColor(runId, draftColor, persistedColor)
      }
      setRunColorDrafts((prev) => {
        if (!(runId in prev)) return prev
        const next = { ...prev }
        delete next[runId]
        return next
      })
    },
    [persistRunColor, runColorDrafts],
  )

  const handleForceFullResync = useCallback(
    async (runId: string) => {
      setIsResyncingRun(runId)
      try {
        // Find the run to check if it's tracking
        const run = runsData?.runs.find((r) => r.run_id === runId)
        // Stop tracking if active
        if (run?.is_tracking) {
          await fetch(`${API_BASE}/stop-tracking`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ run_path: runId }),
          })
        }
        // Delete all data
        const deleteResp = await fetch(`${API_BASE}/delete-run-data`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_path: runId }),
        })
        if (!deleteResp.ok) {
          const error = await deleteResp.json()
          throw new Error(error.detail || "Failed to delete run data")
        }
        // Start a fresh sync
        await fetch(`${API_BASE}/sync`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_path: runId }),
        })
        queryClient.invalidateQueries({ queryKey: ["run-summary", runId] })
        queryClient.invalidateQueries({ queryKey: ["runs"] })
      } catch (error) {
        console.error("Failed to force full resync:", error)
      } finally {
        setIsResyncingRun(null)
        setResyncConfirmRunId(null)
      }
    },
    [runsData?.runs, queryClient],
  )

  const handleSyncEvalsAfterTraining = useCallback(
    async (runId: string) => {
      setIsSyncingEvalsRun(runId)
      try {
        const resp = await fetch(`${API_BASE}/sync-evals-after-training`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_path: runId }),
        })
        if (!resp.ok) {
          const error = await resp.json()
          throw new Error(error.detail || "Failed to start evals sync")
        }
        queryClient.invalidateQueries({ queryKey: ["run-summary", runId] })
        queryClient.invalidateQueries({ queryKey: ["runs"] })
      } catch (error) {
        console.error("Failed to sync evals after training:", error)
      } finally {
        setIsSyncingEvalsRun(null)
      }
    },
    [queryClient],
  )

  const isConfigSearch = runSearchQuery
    .trimStart()
    .toLowerCase()
    .startsWith("config:")

  const handleConfigSearch = useCallback(async () => {
    const raw = runSearchQuery.trimStart()
    const filters = raw.slice("config:".length).trim()
    if (!filters) return
    setIsSearchingConfig(true)
    try {
      const resp = await fetch(`${API_BASE}/search-runs-by-config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filters }),
      })
      if (!resp.ok) {
        setConfigSearchMatchIds([])
        return
      }
      const data = await resp.json()
      setConfigSearchMatchIds(data.run_ids ?? [])
    } catch {
      setConfigSearchMatchIds([])
    } finally {
      setIsSearchingConfig(false)
    }
  }, [runSearchQuery])

  // Clear config search results when the user edits the query away from config:
  useEffect(() => {
    if (!isConfigSearch) {
      setConfigSearchMatchIds(null)
    }
  }, [isConfigSearch])

  const runs = runsData?.runs || []
  const discovery = runsData?.discovery
  const hasWandbKey = runsData?.has_wandb_key ?? apiKey
  const hasNetrcWandbKey = runsData?.has_netrc_wandb_key ?? false
  const wandbKeySource = runsData?.wandb_key_source ?? "unconfigured"

  // Sync useNetrcKey toggle from backend state when dialog opens.
  useEffect(() => {
    if (configDialogOpen) {
      setUseNetrcKey(wandbKeySource === "netrc")
    }
  }, [configDialogOpen, wandbKeySource])

  // Navigation items
  const navItems = [
    {
      title: "Overview",
      href: "/",
      isActive: pathname === "/",
    },
    {
      title: "Metrics",
      href: "/metrics",
      isActive: pathname === "/metrics",
    },
    {
      title: "Rollouts",
      href: "/rollouts",
      isActive: pathname === "/rollouts",
    },
    {
      title: "Rollouts Discarded",
      href: "/rollouts-discarded",
      isActive: pathname === "/rollouts-discarded",
    },
    {
      title: "Timeline",
      href: "/timeline",
      isActive: pathname === "/timeline",
    },
    {
      title: "Infra",
      href: "/infra",
      isActive: pathname === "/infra",
    },
    {
      title: "Evals",
      href: "/evals",
      isActive: pathname === "/evals",
    },
  ]

  return (
    <>
      <div className="flex flex-col h-screen w-56 shrink-0 border-r border-sidebar-border bg-sidebar text-sidebar-foreground fixed left-0 top-0">
        {/* Header */}
        <div className="py-2.5 px-4 border-b border-sidebar-border shrink-0 flex items-center justify-between">
          <Link
            to="/about"
            className="flex items-center h-7 hover:opacity-80 transition-opacity"
          >
            <img src={darkMode ? "/logo-full-dark.svg" : "/logo-full.svg"} alt="Telescope" className="h-5" />
          </Link>
          <button
            onClick={() => setDarkMode((prev) => !prev)}
            className="text-sidebar-foreground/60 hover:text-sidebar-foreground transition-colors"
            aria-label="Toggle dark mode"
          >
            {darkMode ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex flex-col gap-1 p-2 border-b border-sidebar-border shrink-0">
          {navItems.map((item) => (
            <Link
              key={item.href}
              to={item.href}
              onClick={() => {
                if (item.href === "/") {
                  setOverviewShowCodeView(false)
                  setOverviewShowLogsView(false)
                }
              }}
              className={cn(
                "rounded-md px-2 py-1.5 text-sm transition-colors",
                "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                item.isActive &&
                  "bg-sidebar-accent text-sidebar-accent-foreground font-medium",
              )}
            >
              {item.title}
            </Link>
          ))}
        </nav>

        {/* Runs section */}
        <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
          {/* Runs header */}
          <div className="px-2 py-2 shrink-0 flex items-center gap-1">
            <div className="relative flex-1 min-w-0">
              <Input
                type="text"
                placeholder="Search runs…"
                value={runSearchQuery}
                onChange={(e) => setRunSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && isConfigSearch) {
                    e.preventDefault()
                    handleConfigSearch()
                  }
                }}
                className={cn(
                  "h-6 text-[11px] w-full placeholder:text-muted-foreground/60 placeholder:text-[10px]",
                  isConfigSearch && "pr-10",
                )}
              />
              {isConfigSearch && (
                <Button
                  variant="default"
                  size="sm"
                  className="absolute right-0.5 top-1/2 -translate-y-1/2 h-4 px-1.5 text-[9px] rounded-sm"
                  onClick={handleConfigSearch}
                  disabled={isSearchingConfig}
                >
                  {isSearchingConfig ? (
                    <Spinner className="h-2.5 w-2.5" />
                  ) : (
                    "Go"
                  )}
                </Button>
              )}
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 text-muted-foreground hover:text-foreground"
                >
                  <Menu className="h-3.5 w-3.5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem
                  className="text-xs"
                  onClick={() => setProjectsDialogOpen(true)}
                >
                  Add Project
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="text-xs"
                  onClick={() => setAddRunDialogOpen(true)}
                >
                  Add Run
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="text-xs"
                  onClick={() => setConfigDialogOpen(true)}
                >
                  W&B Key{" "}
                  {hasWandbKey ? (
                    <span className="!text-green-500 ml-1">added</span>
                  ) : (
                    <span className="!text-red-500 ml-1">missing</span>
                  )}
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-xs"
                  onClick={() => {
                    setVisibleRuns([])
                    setSelectedRunPath(null)
                  }}
                >
                  Unselect All
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="text-xs flex items-center justify-between"
                  onClick={() => setShowOnlySelected((v) => !v)}
                >
                  Only Show Selected
                  <Check
                    className={cn(
                      "h-3.5 w-3.5",
                      showOnlySelected ? "opacity-100" : "opacity-0",
                    )}
                  />
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-xs flex items-center gap-1.5"
                  onClick={() => setDatabaseDialogOpen(true)}
                >
                  Database
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {/* Discovery progress banner — only while actively searching for runs */}
          {(() => {
            const isDiscovering = discovery?.status === "discovering"
            if (!isDiscovering) return null

            return (
              <div className="px-4 pb-2 shrink-0">
                <div className="flex flex-col gap-1 rounded-md bg-muted/50 px-2.5 py-2">
                  <span className="text-[11px] text-muted-foreground">
                    Fetching runs from W&B…
                  </span>
                  <span className="flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                    <Spinner className="h-3 w-3 shrink-0" />
                    {discovery?.runs_found ?? 0} found
                  </span>
                  <span className="text-[10px] text-muted-foreground/70 leading-tight">
                    Looking for telescope-tagged runs…
                  </span>
                </div>
              </div>
            )
          })()}

          {/* Runs list */}
          <ScrollArea className="flex-1 min-h-0 overflow-hidden">
            <div className="px-2 pb-2">
              {isLoadingRuns ? (
                <div className="flex items-center justify-center py-8">
                  <Spinner className="h-4 w-4" />
                </div>
              ) : (
                <div className="flex flex-col">
                  {runs
                    .filter((run) => {
                      if (showOnlySelected) {
                        if (
                          selectedRunPath !== run.run_id &&
                          !visibleRuns.includes(run.run_id)
                        )
                          return false
                      }
                      if (isConfigSearch) {
                        // In config search mode, filter by server results
                        if (
                          configSearchMatchIds !== null &&
                          !configSearchMatchIds.includes(run.run_id)
                        )
                          return false
                      } else if (runSearchQuery.trim()) {
                        const q = runSearchQuery.trim().toLowerCase()
                        const name = (run.name || "").toLowerCase()
                        const id = run.run_id.toLowerCase()
                        if (!name.includes(q) && !id.includes(q)) return false
                      }
                      return true
                    })
                    .map((run) => {
                      const isSelected = selectedRunPath === run.run_id
                      const isVisible =
                        isSelected || visibleRuns.includes(run.run_id)
                      const draftColor = runColorDrafts[run.run_id]
                      const runColor = normalizeHexColor(
                        draftColor ?? run.color,
                        run.color,
                      )
                      const isPendingSync =
                        run.last_rollout_step === -1 &&
                        !run.is_syncing &&
                        !run.is_tracking
                      const statusDotsCount =
                        Number(run.is_tracking) + Number(run.is_syncing)
                      const runNameMaxChars =
                        statusDotsCount >= 2
                          ? MAX_RUN_NAME_CHARS - 4
                          : statusDotsCount === 1
                            ? MAX_RUN_NAME_CHARS - 2
                            : MAX_RUN_NAME_CHARS - 1
                      const runLabel =
                        run.name || run.run_id.split("/").pop() || run.run_id
                      return (
                        <div
                          key={run.run_id}
                          className={cn(
                            "group flex items-center gap-2 rounded-md px-2 py-1.5 transition-colors w-full cursor-pointer",
                            !isSelected &&
                              "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground [--run-item-bg:var(--sidebar)] hover:[--run-item-bg:var(--sidebar-accent)]",
                            isSelected &&
                              "bg-[var(--sidebar-selected)] text-sidebar-accent-foreground [--run-item-bg:var(--sidebar-selected)]",
                          )}
                          onClick={() => setSelectedRunPath(run.run_id)}
                          onMouseEnter={() => {
                            // Only trigger hover effect if run is visible
                            if (isVisible) {
                              setHoveredRunId(run.run_id)
                            }
                          }}
                          onMouseLeave={() => setHoveredRunId(null)}
                        >
                          {/* Checkbox - locked for selected run */}
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              if (!isSelected) {
                                toggleRunVisibility(run.run_id)
                              }
                            }}
                            disabled={isSelected}
                            className={cn(
                              "w-3 h-3 rounded-sm border shrink-0 flex items-center justify-center transition-colors",
                              isVisible
                                ? "bg-foreground border-foreground"
                                : "border-muted-foreground/50",
                              isSelected && "opacity-50 cursor-not-allowed",
                            )}
                          >
                            {isVisible && (
                              <svg
                                width="8"
                                height="8"
                                viewBox="0 0 8 8"
                                fill="none"
                                className="text-background"
                              >
                                <path
                                  d="M1 4L3 6L7 2"
                                  stroke="currentColor"
                                  strokeWidth="1.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              </svg>
                            )}
                          </button>

                          {/* Color indicator */}
                          <Popover
                            open={openColorPickerRunId === run.run_id}
                            onOpenChange={(open) =>
                              handleRunColorPickerOpenChange(
                                run.run_id,
                                open,
                                run.color,
                              )
                            }
                          >
                            <PopoverTrigger asChild>
                              <button
                                type="button"
                                className="w-2.5 h-2.5 rounded-full shrink-0 focus:outline-none focus:ring-1 focus:ring-ring"
                                style={{ backgroundColor: runColor }}
                                onClick={(e) => e.stopPropagation()}
                                aria-label={`Change color for ${runLabel}`}
                              />
                            </PopoverTrigger>
                            <PopoverContent
                              side="right"
                              align="start"
                              className="w-auto p-1.5"
                              onClick={(e) => e.stopPropagation()}
                              onOpenAutoFocus={(event) =>
                                event.preventDefault()
                              }
                            >
                              <RunColorPicker
                                value={runColor}
                                onChange={(nextColor) => {
                                  setRunColorDrafts((prev) => ({
                                    ...prev,
                                    [run.run_id]: normalizeHexColor(
                                      nextColor,
                                      run.color,
                                    ),
                                  }))
                                }}
                              />
                            </PopoverContent>
                          </Popover>

                          {/* Run name */}
                          <div
                            className={cn(
                              "flex-1 min-w-0 text-left",
                              isPendingSync && "text-muted-foreground/60",
                            )}
                          >
                            <RunNameLabel
                              name={runLabel}
                              maxChars={runNameMaxChars}
                            />
                          </div>

                          {/* Status dots */}
                          {run.is_tracking && (
                            <span
                              className="w-2 h-2 rounded-full bg-green-500 shrink-0"
                              title="Live"
                            />
                          )}
                          {run.is_syncing && (
                            <span
                              className="w-2 h-2 rounded-full bg-blue-500 shrink-0"
                              title="Syncing"
                            />
                          )}

                          {/* Time ago / Menu container */}
                          <div className="ml-auto relative h-6 w-6 flex items-center justify-center">
                            {/* Elapsed time - visible by default, hidden on hover */}
                            <span className="absolute text-[10px] text-muted-foreground transition-opacity group-hover:opacity-0">
                              {formatTimeAgo(run.created_at)}
                            </span>
                            {/* Hamburger menu - hidden by default, visible on hover */}
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="absolute h-6 w-6 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-transparent"
                                  disabled={isRemovingRun === run.run_id}
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <Menu className="h-3.5 w-3.5" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent
                                align="end"
                                className="w-auto"
                              >
                                <DropdownMenuLabel className="text-xs font-normal">
                                  <span className="font-sans">
                                    {run.run_id}
                                  </span>
                                </DropdownMenuLabel>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem
                                  disabled={isResyncingRun === run.run_id}
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    setResyncConfirmRunId(run.run_id)
                                  }}
                                >
                                  {isResyncingRun === run.run_id
                                    ? "Resyncing..."
                                    : "Force Full Resync"}
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  disabled={isSyncingEvalsRun === run.run_id}
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleSyncEvalsAfterTraining(run.run_id)
                                  }}
                                >
                                  {isSyncingEvalsRun === run.run_id
                                    ? "Syncing evals..."
                                    : "Sync Evals After Training"}
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  className="text-destructive"
                                  disabled={isRemovingRun === run.run_id}
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    setRemoveConfirmRunId(run.run_id)
                                  }}
                                >
                                  Remove
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </div>
                        </div>
                      )
                    })}
                  {runs.length === 0 && (
                    <div className="text-center py-4 px-2">
                      {!hasWandbKey ? (
                        <div className="space-y-2">
                          <div className="text-xs text-muted-foreground">
                            No wandb key added yet
                          </div>
                          <Button
                            size="sm"
                            className="h-6 px-2 text-[11px] bg-green-600 hover:bg-green-700"
                            onClick={() => setConfigDialogOpen(true)}
                          >
                            Add wandb key
                          </Button>
                        </div>
                      ) : (
                        <div className="text-xs text-muted-foreground">
                          No runs yet
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      </div>

      {/* Config Dialog */}
      <Dialog open={configDialogOpen} onOpenChange={setConfigDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Configure W&B</DialogTitle>
            <DialogDescription>
              {useNetrcKey ? (
                <>
                  Use the API key saved by{" "}
                  <span className="font-sans">wandb login</span> to fetch all
                  runs tagged <span className="font-sans">telescope</span>.
                </>
              ) : (
                <>
                  Paste your W&B API key to fetch all runs tagged{" "}
                  <span className="font-sans">telescope</span>.
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleConfigure}>
            <div className="space-y-4 py-4">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={useNetrcKey}
                  onChange={(e) => setUseNetrcKey(e.target.checked)}
                  className="accent-primary h-3.5 w-3.5"
                />
                <span className="text-sm">
                  Use key from{" "}
                  <span className="font-sans text-xs">wandb login</span>
                </span>
                {useNetrcKey && !hasNetrcWandbKey && (
                  <span className="text-[11px] text-red-500 font-medium">
                    Not found — run{" "}
                    <code className="text-[10px]">wandb login</code>
                  </span>
                )}
                {useNetrcKey && hasNetrcWandbKey && (
                  <span className="text-[11px] text-green-500 font-medium">
                    Found
                  </span>
                )}
              </label>
              {!useNetrcKey && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="config-api-key">W&B API Key</Label>
                    {hasWandbKey && wandbKeySource === "custom" ? (
                      <span className="text-[11px] text-green-500 font-medium">
                        Key added
                      </span>
                    ) : (
                      <span className="text-[11px] text-red-500 font-medium">
                        No key configured
                      </span>
                    )}
                  </div>
                  <Input
                    id="config-api-key"
                    type="password"
                    placeholder="Enter API key"
                    value={configApiKey}
                    onChange={(e) => setConfigApiKey(e.target.value)}
                    required
                  />
                </div>
              )}
            </div>
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setConfigDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={
                  isConfiguring ||
                  (useNetrcKey ? !hasNetrcWandbKey : !configApiKey.trim())
                }
              >
                {isConfiguring ? (
                  <>
                    <Spinner className="mr-2 h-4 w-4" />
                    Saving...
                  </>
                ) : (
                  "Ok"
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Add Run Dialog */}
      <Dialog open={addRunDialogOpen} onOpenChange={setAddRunDialogOpen}>
        <DialogContent
          className="sm:max-w-3xl !grid-rows-[auto_1fr] !gap-0 max-h-[80vh] overflow-hidden"
          onOpenAutoFocus={(event) => {
            event.preventDefault()
            const dialogContent = event.currentTarget as HTMLElement | null
            dialogContent?.focus()
          }}
        >
          <DialogHeader className="pb-3">
            <DialogTitle>Add Run</DialogTitle>
            <DialogDescription>
              Select a removed run to add it back. Uses the stored W&B API key.
            </DialogDescription>
          </DialogHeader>
          <div className="overflow-y-auto -mx-6 px-6">
            {/* Column headers — sticky inside scroll so widths match rows */}
            <div className="sticky top-0 z-10 bg-background grid grid-cols-[12px_1fr_80px_100px_90px_76px_56px_44px] gap-2 items-center pb-1.5 border-b border-border px-3">
              {/* Spacer for color circle */}
              <div />
              {(
                [
                  { key: "name" as RemovedRunSortKey, label: "Name", cls: "" },
                  {
                    key: "entity" as RemovedRunSortKey,
                    label: "Entity",
                    cls: "",
                  },
                  {
                    key: "project" as RemovedRunSortKey,
                    label: "Project",
                    cls: "",
                  },
                  {
                    key: "run_id" as RemovedRunSortKey,
                    label: "Run ID",
                    cls: "",
                  },
                  {
                    key: "created_at" as RemovedRunSortKey,
                    label: "Created",
                    cls: "",
                  },
                  {
                    key: "removed_at" as RemovedRunSortKey,
                    label: "Removed",
                    cls: "justify-center",
                  },
                ] as const
              ).map((col) => (
                <button
                  key={col.key}
                  className={cn(
                    "flex items-center gap-0.5 text-[10px] font-medium text-muted-foreground hover:text-foreground transition-colors whitespace-nowrap select-none",
                    col.cls,
                  )}
                  onClick={() => toggleRemovedSort(col.key)}
                >
                  {col.label}
                  {removedSortKey === col.key &&
                    (removedSortDir === "asc" ? (
                      <ArrowUp className="h-2.5 w-2.5" />
                    ) : (
                      <ArrowDown className="h-2.5 w-2.5" />
                    ))}
                </button>
              ))}
              {/* Config header - no sort */}
              <span className="text-[10px] font-medium text-muted-foreground text-center">
                Config
              </span>
            </div>
            {/* Rows */}
            <div className="py-1">
              {isLoadingRemovedRuns ? (
                <div className="flex items-center justify-center py-6">
                  <Spinner className="h-4 w-4" />
                </div>
              ) : (
                <>
                  {sortedRemovedRuns.map((run) => {
                    const shortRunId = run.run_id.split("/").pop() ?? run.run_id
                    return (
                      <div
                        key={run.run_id}
                        className="w-full grid grid-cols-[12px_1fr_80px_100px_90px_76px_56px_44px] gap-2 items-center rounded-md px-3 py-1.5 text-left hover:bg-muted transition-colors group/row [--run-item-bg:var(--background)] hover:[--run-item-bg:var(--muted)]"
                      >
                        <div
                          className="w-2.5 h-2.5 rounded-full cursor-pointer"
                          style={{ backgroundColor: run.color ?? "#888" }}
                          onClick={() => handleAddRun(run.run_id)}
                        />
                        <span
                          className="min-w-0 cursor-pointer"
                          onClick={() => handleAddRun(run.run_id)}
                        >
                          <RunNameLabel name={run.name ?? "Unnamed"} />
                        </span>
                        <span
                          className="min-w-0 cursor-pointer"
                          onClick={() => handleAddRun(run.run_id)}
                        >
                          <RunNameLabel
                            name={run.entity ?? "—"}
                            className="text-xs font-sans text-muted-foreground"
                          />
                        </span>
                        <span
                          className="min-w-0 cursor-pointer"
                          onClick={() => handleAddRun(run.run_id)}
                        >
                          <RunNameLabel
                            name={run.project ?? "—"}
                            className="text-xs font-sans text-muted-foreground"
                          />
                        </span>
                        <span
                          className="text-xs font-sans text-muted-foreground truncate cursor-pointer"
                          onClick={() => handleAddRun(run.run_id)}
                        >
                          {shortRunId}
                        </span>
                        <span
                          className="text-[10px] text-muted-foreground whitespace-nowrap cursor-pointer"
                          onClick={() => handleAddRun(run.run_id)}
                        >
                          {run.created_at
                            ? new Date(run.created_at).toLocaleDateString(
                                "en-US",
                                {
                                  month: "2-digit",
                                  day: "2-digit",
                                  year: "numeric",
                                },
                              )
                            : "—"}
                        </span>
                        <span
                          className="text-[10px] text-red-400 whitespace-nowrap text-center cursor-pointer"
                          onClick={() => handleAddRun(run.run_id)}
                        >
                          {run.removed_at ? formatTimeAgo(run.removed_at) : "—"}
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-5 px-1.5 text-[10px] text-muted-foreground hover:text-foreground"
                          onClick={(e) => {
                            e.stopPropagation()
                            setConfigViewRun(run)
                          }}
                          disabled={!run.config}
                        >
                          View
                        </Button>
                      </div>
                    )
                  })}
                  {sortedRemovedRuns.length === 0 && (
                    <div className="text-xs text-muted-foreground text-center py-4">
                      No removed runs
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Known Projects Dialog */}
      <Dialog open={projectsDialogOpen} onOpenChange={setProjectsDialogOpen}>
        <DialogContent className="sm:max-w-lg max-h-[70vh] !grid-rows-[auto_1fr] !gap-0 overflow-hidden">
          <DialogHeader className="pb-3">
            <DialogTitle>Known Projects</DialogTitle>
            <DialogDescription>
              Projects polled for new telescope-tagged runs. Add an
              entity/project pair to start discovering runs from it.
            </DialogDescription>
          </DialogHeader>
          <div className="overflow-y-auto -mx-6 px-6">
            <div className="flex flex-col gap-1 mb-3">
              <div className="flex gap-2">
                <Input
                  className="h-7 text-xs flex-1"
                  placeholder="entity/project"
                  value={newProjectInput}
                  onChange={(e) => {
                    setNewProjectInput(e.target.value)
                    if (addProjectError) setAddProjectError(null)
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleAddProject()
                  }}
                />
                <Button
                  variant="default"
                  size="sm"
                  className="h-7 px-3 text-xs"
                  onClick={handleAddProject}
                  disabled={isAddingProject || !newProjectInput.trim()}
                >
                  {isAddingProject ? <Spinner className="h-3 w-3" /> : "Ok"}
                </Button>
              </div>
              {addProjectError && (
                <span className="text-[11px] text-red-500">
                  {addProjectError}
                </span>
              )}
            </div>
            <div className="space-y-1">
              {knownProjectsData?.projects.map((p) => (
                <div
                  key={p.project}
                  className="flex items-center justify-between rounded-md px-3 py-1.5 hover:bg-muted transition-colors group"
                >
                  <div className="flex flex-col min-w-0">
                    <span className="text-xs font-mono truncate">
                      {p.project}
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                      {p.source === "user"
                        ? "User-added"
                        : "From existing runs"}
                      {p.added_at &&
                        ` · ${new Date(p.added_at).toLocaleDateString()}`}
                    </span>
                  </div>
                  {p.source === "user" && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-5 w-5 p-0 text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={() => handleRemoveProject(p.project)}
                      disabled={removingProject === p.project}
                    >
                      {removingProject === p.project ? (
                        <Spinner className="h-3 w-3" />
                      ) : (
                        <Trash2 className="h-3 w-3" />
                      )}
                    </Button>
                  )}
                </div>
              ))}
              {knownProjectsData?.projects.length === 0 && (
                <div className="text-xs text-muted-foreground text-center py-4">
                  No known projects. Add a W&B API key and sync runs, or add a
                  project manually above.
                </div>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Config View Dialog for removed runs */}
      <Dialog
        open={!!configViewRun}
        onOpenChange={(open) => {
          if (!open) setConfigViewRun(null)
        }}
      >
        <DialogContent className="sm:max-w-2xl max-h-[85vh] !grid-rows-[auto_1fr] overflow-hidden">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {configViewRun?.color && (
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: configViewRun.color }}
                />
              )}
              Configuration —{" "}
              {configViewRun?.name ?? configViewRun?.run_id.split("/").pop()}
            </DialogTitle>
            <DialogDescription>{configViewRun?.run_id}</DialogDescription>
          </DialogHeader>
          <div className="overflow-y-auto -mx-6 px-6">
            {configViewRun?.config ? (
              <RunConfigPanel config={configViewRun.config} />
            ) : (
              <div className="text-sm text-muted-foreground text-center py-8">
                No configuration data available
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={!!removeConfirmRunId}
        onOpenChange={(open) => {
          if (!open) {
            setRemoveConfirmRunId(null)
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove run locally?</AlertDialogTitle>
            <AlertDialogDescription>
              All the data of this run will be removed from your local database.
              It will not be removed from W&B, your data is safe in W&B and you
              can add it again locally if you want.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (removeConfirmRunId) {
                  handleRemoveRun(removeConfirmRunId)
                }
              }}
            >
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={!!resyncConfirmRunId}
        onOpenChange={(open) => {
          if (!open) {
            setResyncConfirmRunId(null)
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogMedia className="bg-destructive/10">
              <AlertTriangle className="text-destructive" />
            </AlertDialogMedia>
            <AlertDialogTitle>Force Full Resync</AlertDialogTitle>
            <AlertDialogDescription>
              This will stop tracking (if active), delete all data for this run
              from the database, and then sync all data again from scratch. This
              action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                if (resyncConfirmRunId) {
                  handleForceFullResync(resyncConfirmRunId)
                }
              }}
            >
              Yes, Delete &amp; Resync
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <DatabaseDialog
        open={databaseDialogOpen}
        onOpenChange={setDatabaseDialogOpen}
      />
    </>
  )
}
