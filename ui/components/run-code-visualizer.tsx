
import { useEffect, useMemo, useRef, useState } from "react"
import { diffArrays } from "diff"
import {
  ArrowLeft,
  ArrowRight,
  CheckIcon,
  ChevronLeft,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  CopyIcon,
  FileCode2,
  Folder,
  FolderOpen,
  X,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { RunCodeCompareDialog } from "@/components/run-code-compare-dialog"
import {
  useRunCodeDiffSummary,
  useRunCodeFile,
  useRunCodeTree,
  useRuns,
} from "@/hooks/use-run-data"
import { formatTimeAgo } from "@/lib/format"
import type { RunCodeTreeNode } from "@/lib/types"
import { cn } from "@/lib/utils"

type HighlightToken = {
  text: string
  className?: string
}

type DiffKind = "context" | "add" | "remove"

type DiffRow = {
  kind: DiffKind
  text: string
  leftLine: number | null
  rightLine: number | null
}

type DiffBlock = {
  startRowIndex: number
  endRowIndex: number
}

type FrozenCompareDisplay = {
  targetPath: string
  leftContent: string
  rightContent: string
}

type TreeDiffStatus = "common" | "added" | "removed" | "modified"

type DiffTreeNode = {
  name: string
  path: string
  type: "directory" | "file"
  status: TreeDiffStatus
  children?: DiffTreeNode[]
}

const LANGUAGE_KEYWORDS: Record<string, string[]> = {
  python: [
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "True",
    "try",
    "while",
    "with",
    "yield",
  ],
  javascript: [
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "default",
    "delete",
    "do",
    "else",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "let",
    "new",
    "null",
    "return",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "var",
    "void",
    "while",
    "with",
    "yield",
  ],
  typescript: [
    "any",
    "as",
    "async",
    "await",
    "boolean",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "default",
    "declare",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "from",
    "function",
    "if",
    "implements",
    "import",
    "in",
    "interface",
    "let",
    "namespace",
    "new",
    "null",
    "private",
    "protected",
    "public",
    "readonly",
    "return",
    "static",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "type",
    "typeof",
    "undefined",
    "var",
    "while",
  ],
  json: ["true", "false", "null"],
  bash: [
    "if",
    "then",
    "else",
    "elif",
    "fi",
    "for",
    "while",
    "do",
    "done",
    "case",
    "esac",
    "function",
    "in",
    "local",
    "export",
    "return",
  ],
  sql: [
    "SELECT",
    "FROM",
    "WHERE",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "ON",
    "GROUP",
    "BY",
    "ORDER",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    "CREATE",
    "TABLE",
    "ALTER",
    "DROP",
    "AND",
    "OR",
    "NOT",
    "NULL",
    "AS",
    "DISTINCT",
    "UNION",
    "ALL",
  ],
}

const HASH_COMMENT_LANGUAGES = new Set(["python", "bash", "yaml", "toml"])
const DOUBLE_DASH_COMMENT_LANGUAGES = new Set(["sql"])

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

function getLanguageFromPath(filePath: string | null): string {
  if (!filePath) return "plaintext"
  const lowerPath = filePath.toLowerCase()
  const extension = lowerPath.split(".").pop() ?? ""

  if (extension === "py") return "python"
  if (extension === "ts" || extension === "tsx") return "typescript"
  if (extension === "js" || extension === "jsx" || extension === "mjs" || extension === "cjs") return "javascript"
  if (extension === "json") return "json"
  if (extension === "sh" || extension === "bash" || extension === "zsh") return "bash"
  if (extension === "sql") return "sql"
  if (extension === "yml" || extension === "yaml") return "yaml"
  if (extension === "toml") return "toml"
  if (extension === "md") return "markdown"
  if (extension === "go") return "go"
  if (extension === "rs") return "rust"
  if (extension === "java") return "java"
  if (extension === "php") return "php"
  if (extension === "rb") return "ruby"
  if (extension === "css") return "css"
  if (extension === "html") return "html"

  if (lowerPath.endsWith("/dockerfile")) return "dockerfile"
  return "plaintext"
}

function getCommentPattern(language: string): string | null {
  if (HASH_COMMENT_LANGUAGES.has(language)) return "#.*$"
  if (DOUBLE_DASH_COMMENT_LANGUAGES.has(language)) return "--.*$"
  if (language === "json") return null
  return "//.*$"
}

function tokenizeLine(line: string, language: string): HighlightToken[] {
  if (line.length === 0) return [{ text: " " }]

  const keywords = LANGUAGE_KEYWORDS[language] ?? []
  const keywordSet = new Set(keywords)

  const stringPattern = "\"(?:\\\\.|[^\"\\\\])*\"|'(?:\\\\.|[^'\\\\])*'|`(?:\\\\.|[^`\\\\])*`"
  const commentPattern = getCommentPattern(language)
  const keywordPattern =
    keywords.length > 0
      ? `\\b(?:${keywords.map((keyword) => escapeRegex(keyword)).join("|")})\\b`
      : null
  const numberPattern = "\\b\\d+(?:\\.\\d+)?\\b"

  const patternParts = [stringPattern]
  if (commentPattern) patternParts.push(commentPattern)
  if (keywordPattern) patternParts.push(keywordPattern)
  patternParts.push(numberPattern)
  const tokenRegex = new RegExp(patternParts.join("|"), "g")

  const tokens: HighlightToken[] = []
  let lastIndex = 0
  for (const match of line.matchAll(tokenRegex)) {
    const token = match[0] ?? ""
    const tokenStart = match.index
    if (tokenStart === undefined || token.length === 0) continue

    if (tokenStart > lastIndex) {
      tokens.push({ text: line.slice(lastIndex, tokenStart) })
    }

    let className: string | undefined
    if (token.startsWith('"') || token.startsWith("'") || token.startsWith("`")) {
      className = "text-emerald-600 dark:text-emerald-400"
    } else if (
      token.startsWith("#") ||
      token.startsWith("//") ||
      token.startsWith("--")
    ) {
      className = "text-muted-foreground"
    } else if (keywordSet.has(token)) {
      className = "text-violet-600 dark:text-violet-400"
    } else if (/^\d/.test(token)) {
      className = "text-amber-600 dark:text-amber-400"
    }

    tokens.push({ text: token, className })
    lastIndex = tokenStart + token.length
  }

  if (lastIndex < line.length) {
    tokens.push({ text: line.slice(lastIndex) })
  }

  return tokens
}

function findFirstTripleQuote(line: string): { index: number; quote: '"""' | "'''" } | null {
  const doubleIdx = line.indexOf('"""')
  const singleIdx = line.indexOf("'''")

  if (doubleIdx === -1 && singleIdx === -1) return null
  if (doubleIdx === -1) return { index: singleIdx, quote: "'''" }
  if (singleIdx === -1) return { index: doubleIdx, quote: '"""' }

  return doubleIdx <= singleIdx
    ? { index: doubleIdx, quote: '"""' }
    : { index: singleIdx, quote: "'''" }
}

function tokenizeCodeContent(content: string, language: string): HighlightToken[][] {
  const lines = content.split(/\r?\n/)
  if (language !== "python") {
    return lines.map((line) => tokenizeLine(line, language))
  }

  let openTriple: '"""' | "'''" | null = null
  const allLines: HighlightToken[][] = []

  for (const line of lines) {
    if (line.length === 0) {
      allLines.push([{ text: " " }])
      continue
    }

    if (openTriple) {
      const closeIdx = line.indexOf(openTriple)
      if (closeIdx === -1) {
        allLines.push([{ text: line, className: "text-muted-foreground" }])
        continue
      }

      const beforeAndClose = line.slice(0, closeIdx + 3)
      const afterClose = line.slice(closeIdx + 3)
      const tokens: HighlightToken[] = [
        { text: beforeAndClose, className: "text-muted-foreground" },
      ]
      openTriple = null
      if (afterClose.length > 0) {
        tokens.push(...tokenizeLine(afterClose, language))
      }
      allLines.push(tokens)
      continue
    }

    const tripleStart = findFirstTripleQuote(line)
    if (!tripleStart) {
      allLines.push(tokenizeLine(line, language))
      continue
    }

    const closeIdx = line.indexOf(tripleStart.quote, tripleStart.index + 3)
    const beforeTriple = line.slice(0, tripleStart.index)

    if (closeIdx === -1) {
      const tokens: HighlightToken[] = []
      if (beforeTriple.length > 0) {
        tokens.push(...tokenizeLine(beforeTriple, language))
      }
      tokens.push({
        text: line.slice(tripleStart.index),
        className: "text-muted-foreground",
      })
      allLines.push(tokens)
      openTriple = tripleStart.quote
      continue
    }

    const tokens: HighlightToken[] = []
    const tripleText = line.slice(tripleStart.index, closeIdx + 3)
    const afterTriple = line.slice(closeIdx + 3)

    if (beforeTriple.length > 0) {
      tokens.push(...tokenizeLine(beforeTriple, language))
    }
    tokens.push({ text: tripleText, className: "text-muted-foreground" })
    if (afterTriple.length > 0) {
      tokens.push(...tokenizeLine(afterTriple, language))
    }
    allLines.push(tokens)
  }

  return allLines
}

function contentToDiffLines(content: string): string[] {
  if (!content) return []
  const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n")
  const lines = normalized.split("\n")
  if (lines.length > 0 && lines[lines.length - 1] === "") {
    lines.pop()
  }
  return lines
}

function buildGitStyleDiff(leftContent: string, rightContent: string): DiffRow[] {
  const leftLines = contentToDiffLines(leftContent)
  const rightLines = contentToDiffLines(rightContent)

  const changes = diffArrays(leftLines, rightLines)
  const rows: DiffRow[] = []
  let leftNo = 1
  let rightNo = 1

  for (const change of changes) {
    for (const line of change.value) {
      if (change.added) {
        rows.push({ kind: "add", text: line, leftLine: null, rightLine: rightNo })
        rightNo += 1
      } else if (change.removed) {
        rows.push({ kind: "remove", text: line, leftLine: leftNo, rightLine: null })
        leftNo += 1
      } else {
        rows.push({ kind: "context", text: line, leftLine: leftNo, rightLine: rightNo })
        leftNo += 1
        rightNo += 1
      }
    }
  }

  return rows
}

function toCommonTree(nodes: RunCodeTreeNode[]): DiffTreeNode[] {
  return nodes.map((node) => ({
    name: node.name,
    path: node.path,
    type: node.type,
    status: "common",
    children:
      node.type === "directory" && node.children?.length
        ? toCommonTree(node.children)
        : undefined,
  }))
}

function mergeTreeNodes(
  leftNodes: RunCodeTreeNode[],
  rightNodes: RunCodeTreeNode[]
): DiffTreeNode[] {
  const leftMap = new Map(leftNodes.map((node) => [node.name, node]))
  const rightMap = new Map(rightNodes.map((node) => [node.name, node]))
  const names = Array.from(new Set([...leftMap.keys(), ...rightMap.keys()]))

  names.sort((a, b) => {
    const nodeA = leftMap.get(a) ?? rightMap.get(a)
    const nodeB = leftMap.get(b) ?? rightMap.get(b)
    if (!nodeA || !nodeB) return a.localeCompare(b)
    if (nodeA.type !== nodeB.type) {
      return nodeA.type === "directory" ? -1 : 1
    }
    return nodeA.name.localeCompare(nodeB.name)
  })

  const merged: DiffTreeNode[] = []
  for (const name of names) {
    const leftNode = leftMap.get(name)
    const rightNode = rightMap.get(name)
    if (!leftNode && !rightNode) continue

    const source = leftNode ?? rightNode
    if (!source) continue

    const baseStatus: TreeDiffStatus = leftNode
      ? rightNode
        ? "common"
        : "removed"
      : "added"

    if (source.type === "directory") {
      const children = mergeTreeNodes(
        leftNode?.children ?? [],
        rightNode?.children ?? []
      )
      const status: TreeDiffStatus =
        baseStatus === "common" && children.some((child) => child.status !== "common")
          ? "modified"
          : baseStatus
      merged.push({
        name: source.name,
        path: source.path,
        type: "directory",
        status,
        children,
      })
    } else {
      const status: TreeDiffStatus =
        baseStatus === "common" &&
        leftNode?.hash != null &&
        rightNode?.hash != null &&
        leftNode.hash !== rightNode.hash
          ? "modified"
          : baseStatus
      merged.push({
        name: source.name,
        path: source.path,
        type: "file",
        status,
      })
    }
  }

  return merged
}

function collectChangedFilePaths(nodes: DiffTreeNode[]): string[] {
  const out: string[] = []

  const walk = (items: DiffTreeNode[]) => {
    for (const node of items) {
      if (node.type === "file") {
        if (node.status !== "common") {
          out.push(node.path)
        }
      } else if (node.children?.length) {
        walk(node.children)
      }
    }
  }

  walk(nodes)
  return out
}

function getAncestorDirectoryPaths(filePath: string): string[] {
  const parts = filePath.split("/")
  const ancestors: string[] = []
  for (let i = 1; i < parts.length; i++) {
    ancestors.push(parts.slice(0, i).join("/"))
  }
  return ancestors
}

function buildChangeBlocks(diffRows: DiffRow[]): DiffBlock[] {
  const blocks: DiffBlock[] = []
  let openStart = -1

  for (let i = 0; i < diffRows.length; i++) {
    const isChange = diffRows[i].kind !== "context"
    if (isChange) {
      if (openStart === -1) openStart = i
    } else if (openStart !== -1) {
      blocks.push({ startRowIndex: openStart, endRowIndex: i - 1 })
      openStart = -1
    }
  }

  if (openStart !== -1) {
    blocks.push({ startRowIndex: openStart, endRowIndex: diffRows.length - 1 })
  }

  return blocks
}

function findFirstFilePath(nodes: DiffTreeNode[]): string | null {
  for (const node of nodes) {
    if (node.type === "file") return node.path
    if (node.type === "directory" && node.children?.length) {
      const nested = findFirstFilePath(node.children)
      if (nested) return nested
    }
  }
  return null
}

function treeContainsFilePath(nodes: RunCodeTreeNode[], filePath: string): boolean {
  for (const node of nodes) {
    if (node.type === "file" && node.path === filePath) return true
    if (node.type === "directory" && node.children?.length) {
      if (treeContainsFilePath(node.children, filePath)) return true
    }
  }
  return false
}

function treeContainsFilePathDiff(nodes: DiffTreeNode[], filePath: string): boolean {
  for (const node of nodes) {
    if (node.type === "file" && node.path === filePath) return true
    if (node.type === "directory" && node.children?.length) {
      if (treeContainsFilePathDiff(node.children, filePath)) return true
    }
  }
  return false
}

function collectTopLevelDirectories(nodes: DiffTreeNode[]): Set<string> {
  const out = new Set<string>()
  for (const node of nodes) {
    if (node.type === "directory") out.add(node.path)
  }
  return out
}

interface RunCodeVisualizerProps {
  runPath: string
  onBack: () => void
}

export function RunCodeVisualizer({ runPath, onBack }: RunCodeVisualizerProps) {
  const [expandedDirs, setExpandedDirs] = useState<Set<string> | null>(null)
  const [selectedFilePathState, setSelectedFilePathState] = useState<string | null>(null)
  const [compareRunId, setCompareRunId] = useState<string | null>(null)
  const [copiedFilePath, setCopiedFilePath] = useState<string | null>(null)
  const [activeChangeNavIndex, setActiveChangeNavIndex] = useState(0)
  const [frozenCompareDisplay, setFrozenCompareDisplay] =
    useState<FrozenCompareDisplay | null>(null)
  const diffRowRefs = useRef<Record<number, HTMLDivElement | null>>({})
  const fileItemRefs = useRef<Record<string, HTMLButtonElement | null>>({})
  const pendingJumpToFirstChangeRef = useRef(false)
  const pendingJumpTargetFileRef = useRef<string | null>(null)
  const pendingAutoSelectChangedFileRunRef = useRef<string | null>(null)

  const { data: runsData } = useRuns()
  const currentRun = runsData?.runs.find((run) => run.run_id === runPath) ?? null
  const currentRunLabel = currentRun?.name || runPath.split("/").pop() || runPath
  const compareRun = runsData?.runs.find((run) => run.run_id === compareRunId) ?? null
  const compareRunLabel =
    compareRun?.name || compareRunId?.split("/").pop() || compareRunId

  const {
    data: treeData,
    isLoading: isLoadingTree,
    error: treeError,
  } = useRunCodeTree(runPath, !!runPath)
  const {
    data: compareTreeData,
    isLoading: isLoadingCompareTree,
    error: compareTreeError,
  } = useRunCodeTree(compareRunId || "", !!compareRunId)
  const {
    data: globalDiffSummaryData,
    isFetching: isFetchingGlobalDiffSummary,
  } = useRunCodeDiffSummary(
    compareRunId || "",
    runPath,
    !!compareRunId &&
      !!treeData?.available &&
      !!compareTreeData?.available &&
      !isLoadingCompareTree
  )

  const currentTreeNodes = useMemo(
    () => (treeData?.available ? treeData.tree : []),
    [treeData]
  )
  const compareTreeNodes = useMemo(
    () => (compareTreeData?.available ? compareTreeData.tree : []),
    [compareTreeData]
  )
  const isCompareMode = compareRunId !== null
  const useMergedTree =
    isCompareMode && !isLoadingCompareTree && compareTreeData?.available === true

  const displayTreeNodes = useMemo(() => {
    if (useMergedTree) {
      // Compare direction is compare-run -> current-run, so removed/added
      // statuses match that orientation.
      return mergeTreeNodes(compareTreeNodes, currentTreeNodes)
    }
    return toCommonTree(currentTreeNodes)
  }, [useMergedTree, currentTreeNodes, compareTreeNodes])

  const defaultExpandedDirs = collectTopLevelDirectories(displayTreeNodes)
  const expandedDirSet = expandedDirs ?? defaultExpandedDirs
  const selectedFilePath =
    displayTreeNodes.length === 0
      ? null
      : selectedFilePathState &&
          treeContainsFilePathDiff(displayTreeNodes, selectedFilePathState)
        ? selectedFilePathState
        : findFirstFilePath(displayTreeNodes)
  const selectedLanguage = getLanguageFromPath(selectedFilePath)
  const currentHasSelectedFile =
    !!selectedFilePath && treeContainsFilePath(currentTreeNodes, selectedFilePath)
  const compareHasSelectedFile =
    !!selectedFilePath && treeContainsFilePath(compareTreeNodes, selectedFilePath)

  const {
    data: currentFileQueryData,
    isLoading: isLoadingCurrentFile,
    isFetching: isFetchingCurrentFile,
    error: currentFileError,
  } = useRunCodeFile(
    runPath,
    selectedFilePath ?? "",
    !!runPath &&
      !!selectedFilePath &&
      !!treeData?.available &&
      currentHasSelectedFile
  )
  const {
    data: compareFileQueryData,
    isLoading: isLoadingCompareFile,
    isFetching: isFetchingCompareFile,
    error: compareFileError,
  } = useRunCodeFile(
    compareRunId ?? "",
    selectedFilePath ?? "",
    !!compareRunId &&
      !!selectedFilePath &&
      !!compareTreeData?.available &&
      compareHasSelectedFile
  )

  const currentFileData = currentHasSelectedFile ? currentFileQueryData : null
  const compareFileData = compareHasSelectedFile ? compareFileQueryData : null
  const currentFileDataForSelected =
    selectedFilePath &&
    currentFileData?.file_path === selectedFilePath
      ? currentFileData
      : null
  const compareFileDataForSelected =
    selectedFilePath &&
    compareFileData?.file_path === selectedFilePath
      ? compareFileData
      : null
  const currentFileContent = currentFileData?.content ?? null
  const currentFileContentForCompare = currentFileDataForSelected?.content ?? null
  const compareFileContentForCompare = compareFileDataForSelected?.content ?? null
  const selectedCurrentContentForCompare = currentHasSelectedFile
    ? currentFileContentForCompare
    : ""
  const selectedCompareContentForCompare = compareHasSelectedFile
    ? compareFileContentForCompare
    : ""
  const frozenForSelected =
    selectedFilePath && frozenCompareDisplay?.targetPath === selectedFilePath
      ? frozenCompareDisplay
      : null
  const selectedCompareDataReady =
    selectedCurrentContentForCompare !== null &&
    selectedCompareContentForCompare !== null
  const useFrozenCompareDisplay =
    isCompareMode && !selectedCompareDataReady && frozenForSelected !== null
  const currentFileDisplayContent = useFrozenCompareDisplay
    ? frozenForSelected.rightContent
    : selectedCurrentContentForCompare
  const compareFileDisplayContent = useFrozenCompareDisplay
    ? frozenForSelected.leftContent
    : selectedCompareContentForCompare

  const toggleDirectory = (path: string) => {
    setExpandedDirs((prev) => {
      const next = new Set(prev ?? defaultExpandedDirs)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }

  const renderNodes = (nodes: DiffTreeNode[], depth: number) =>
    nodes.map((node) => {
      const statusTextClass =
        node.status === "added"
          ? "text-emerald-700 dark:text-emerald-400"
          : node.status === "removed"
            ? "text-[#f27c7e] line-through"
            : node.status === "modified"
              ? "text-[#ab6100]"
              : ""
      const statusIconClass =
        node.status === "added"
          ? "text-emerald-700 dark:text-emerald-400"
          : node.status === "removed"
            ? "text-[#f27c7e]"
          : node.status === "modified"
            ? "text-[#ab6100]"
            : "text-muted-foreground"

      if (node.type === "directory") {
        const isOpen = expandedDirSet.has(node.path)
        return (
          <div key={node.path}>
            <button
              type="button"
              onClick={() => toggleDirectory(node.path)}
              className="flex w-full items-center gap-1 rounded px-1.5 py-1 text-left text-xs hover:bg-muted"
              style={{ paddingLeft: `${depth * 14 + 6}px` }}
            >
              <span className={cn("inline-flex h-4 w-4 items-center justify-center", statusIconClass)}>
                {isOpen ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
              </span>
              {isOpen ? (
                <FolderOpen className={cn("h-3.5 w-3.5", statusIconClass)} />
              ) : (
                <Folder className={cn("h-3.5 w-3.5", statusIconClass)} />
              )}
              <span className={cn("truncate", statusTextClass)}>{node.name}</span>
            </button>
            {isOpen && node.children?.length
              ? renderNodes(node.children, depth + 1)
              : null}
          </div>
        )
      }

      const isSelected = selectedFilePath === node.path
      return (
        <button
          key={node.path}
          type="button"
          ref={(el) => {
            fileItemRefs.current[node.path] = el
          }}
          onClick={() => {
            freezeCompareForPath(node.path)
            setSelectedFilePathState(node.path)
            if (isCompareMode) {
              setActiveChangeNavIndex(0)
              pendingJumpToFirstChangeRef.current = true
              pendingJumpTargetFileRef.current = node.path
            }
          }}
          className={cn(
            "flex w-full items-center gap-1 rounded px-1.5 py-1 text-left text-xs hover:bg-muted",
            isSelected && "bg-muted font-medium"
          )}
          style={{ paddingLeft: `${depth * 14 + 26}px` }}
        >
          <FileCode2 className={cn("h-3.5 w-3.5", statusIconClass)} />
          <span className={cn("truncate", statusTextClass)}>{node.name}</span>
        </button>
      )
    })

  const highlightedLines = useMemo(
    () => tokenizeCodeContent(currentFileContent ?? "", selectedLanguage),
    [currentFileContent, selectedLanguage]
  )
  const leftDiffHighlightedLines = useMemo(
    () => tokenizeCodeContent(compareFileDisplayContent ?? "", selectedLanguage),
    [compareFileDisplayContent, selectedLanguage]
  )
  const rightDiffHighlightedLines = useMemo(
    () => tokenizeCodeContent(currentFileDisplayContent ?? "", selectedLanguage),
    [currentFileDisplayContent, selectedLanguage]
  )
  const diffRows = useMemo(() => {
    if (!isCompareMode || !selectedFilePath) return []
    return buildGitStyleDiff(
      compareFileDisplayContent ?? "",
      currentFileDisplayContent ?? ""
    )
  }, [
    isCompareMode,
    selectedFilePath,
    compareFileDisplayContent,
    currentFileDisplayContent,
  ])
  const isInitialCompareLoading =
    !!selectedFilePath &&
    isCompareMode &&
    !useFrozenCompareDisplay &&
    ((currentHasSelectedFile &&
      selectedCurrentContentForCompare === null &&
      !currentFileError) ||
      (compareHasSelectedFile &&
        selectedCompareContentForCompare === null &&
        !compareFileError))
  const isCompareFetching =
    !!selectedFilePath &&
    isCompareMode &&
    ((currentHasSelectedFile && isFetchingCurrentFile) ||
      (compareHasSelectedFile && isFetchingCompareFile))
  const isCurrentFileReadyForSelected =
    !currentHasSelectedFile ||
    (currentFileDataForSelected !== null &&
      !isLoadingCurrentFile &&
      !isFetchingCurrentFile)
  const isCompareFileReadyForSelected =
    !compareHasSelectedFile ||
    (compareFileDataForSelected !== null &&
      !isLoadingCompareFile &&
      !isFetchingCompareFile)
  const compareBinary = !!compareFileDataForSelected?.is_binary
  const currentBinary = !!currentFileData?.is_binary
  const currentBinaryForCompare = !!currentFileDataForSelected?.is_binary
  const copyableSelectedFileContent =
    isCompareMode
      ? currentHasSelectedFile
        ? currentFileDisplayContent
        : compareHasSelectedFile
          ? compareFileDisplayContent
          : null
      : currentFileContent
  const selectedCopySourceIsBinary = isCompareMode
    ? currentHasSelectedFile
      ? currentBinaryForCompare
      : compareHasSelectedFile
        ? compareBinary
        : false
    : currentBinary
  const canCopySelectedFile =
    !!selectedFilePath &&
    copyableSelectedFileContent !== null &&
    !selectedCopySourceIsBinary
  const changeBlocks = useMemo(() => buildChangeBlocks(diffRows), [diffRows])
  const changedFilePaths = useMemo(
    () => collectChangedFilePaths(displayTreeNodes),
    [displayTreeNodes]
  )
  const selectedChangedFileIndex = selectedFilePath
    ? changedFilePaths.indexOf(selectedFilePath)
    : -1
  const normalizedChangeIndex =
    changeBlocks.length > 0
      ? ((activeChangeNavIndex % changeBlocks.length) +
          changeBlocks.length) %
        changeBlocks.length
      : 0
  const displayedChangeIndex =
    changeBlocks.length > 0 ? normalizedChangeIndex + 1 : 0
  const displayedFileIndex =
    changedFilePaths.length > 0 && selectedChangedFileIndex >= 0
      ? selectedChangedFileIndex + 1
      : 0
  const fileDiffAddedCount = useMemo(
    () => diffRows.reduce((count, row) => count + (row.kind === "add" ? 1 : 0), 0),
    [diffRows]
  )
  const fileDiffRemovedCount = useMemo(
    () => diffRows.reduce((count, row) => count + (row.kind === "remove" ? 1 : 0), 0),
    [diffRows]
  )
  const globalDiffAddedCount = globalDiffSummaryData?.added_lines ?? 0
  const globalDiffRemovedCount = globalDiffSummaryData?.removed_lines ?? 0

  const didCopySelectedFile = copiedFilePath !== null && copiedFilePath === selectedFilePath

  const handleCopySelectedFile = async () => {
    if (!canCopySelectedFile || copyableSelectedFileContent === null) return
    await navigator.clipboard.writeText(copyableSelectedFileContent)
    setCopiedFilePath(selectedFilePath)
    setTimeout(() => setCopiedFilePath(null), 2000)
  }

  const freezeCompareForPath = (nextPath: string) => {
    if (!isCompareMode) return
    setFrozenCompareDisplay({
      targetPath: nextPath,
      leftContent: compareFileDisplayContent ?? "",
      rightContent: currentFileDisplayContent ?? "",
    })
  }

  const goToChange = (direction: 1 | -1) => {
    if (changeBlocks.length === 0) return
    const nextIndex =
      (normalizedChangeIndex + direction + changeBlocks.length) %
      changeBlocks.length
    setActiveChangeNavIndex(nextIndex)
    const targetRowIndex = changeBlocks[nextIndex].startRowIndex
    diffRowRefs.current[targetRowIndex]?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    })
  }

  const goToChangedFile = (direction: 1 | -1) => {
    if (changedFilePaths.length === 0) return
    const nextIndex =
      selectedChangedFileIndex < 0
        ? direction === 1
          ? 0
          : changedFilePaths.length - 1
        : (selectedChangedFileIndex + direction + changedFilePaths.length) %
          changedFilePaths.length
    if (nextIndex === selectedChangedFileIndex) return
    const nextPath = changedFilePaths[nextIndex]
    freezeCompareForPath(nextPath)
    setSelectedFilePathState(nextPath)
    setExpandedDirs((prev) => {
      const next = new Set(prev ?? defaultExpandedDirs)
      for (const ancestor of getAncestorDirectoryPaths(nextPath)) {
        next.add(ancestor)
      }
      return next
    })
    setActiveChangeNavIndex(0)
    pendingJumpToFirstChangeRef.current = true
    pendingJumpTargetFileRef.current = nextPath
    setTimeout(() => {
      fileItemRefs.current[nextPath]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      })
    }, 0)
  }

  const goToFirstChangedFile = () => {
    if (changedFilePaths.length === 0) return
    const firstPath = changedFilePaths[0]
    freezeCompareForPath(firstPath)
    setSelectedFilePathState(firstPath)
    setExpandedDirs((prev) => {
      const next = new Set(prev ?? defaultExpandedDirs)
      for (const ancestor of getAncestorDirectoryPaths(firstPath)) {
        next.add(ancestor)
      }
      return next
    })
    setActiveChangeNavIndex(0)
    pendingJumpToFirstChangeRef.current = true
    pendingJumpTargetFileRef.current = firstPath
    setTimeout(() => {
      fileItemRefs.current[firstPath]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      })
    }, 0)
  }

  useEffect(() => {
    pendingAutoSelectChangedFileRunRef.current = compareRunId
    if (compareRunId === null) {
      pendingJumpToFirstChangeRef.current = false
      pendingJumpTargetFileRef.current = null
    }
  }, [compareRunId])

  useEffect(() => {
    if (!isCompareMode || !compareRunId) return
    if (!useMergedTree) return
    if (pendingAutoSelectChangedFileRunRef.current !== compareRunId) return

    pendingAutoSelectChangedFileRunRef.current = null
    if (changedFilePaths.length === 0) return

    const firstPath = changedFilePaths[0]
    pendingJumpToFirstChangeRef.current = true
    pendingJumpTargetFileRef.current = firstPath
    const timer = window.setTimeout(() => {
      if (selectedFilePath !== firstPath) {
        setFrozenCompareDisplay({
          targetPath: firstPath,
          leftContent: compareFileDisplayContent ?? "",
          rightContent: currentFileDisplayContent ?? "",
        })
        setSelectedFilePathState(firstPath)
        setExpandedDirs((prev) => {
          const next = new Set(prev ?? collectTopLevelDirectories(displayTreeNodes))
          for (const ancestor of getAncestorDirectoryPaths(firstPath)) {
            next.add(ancestor)
          }
          return next
        })
      }
      setActiveChangeNavIndex(0)
      fileItemRefs.current[firstPath]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      })
    }, 0)
    return () => window.clearTimeout(timer)
  }, [
    isCompareMode,
    compareRunId,
    useMergedTree,
    changedFilePaths,
    selectedFilePath,
    compareFileDisplayContent,
    currentFileDisplayContent,
    displayTreeNodes,
  ])

  useEffect(() => {
    if (!isCompareMode || !selectedFilePath) return
    if (!pendingJumpToFirstChangeRef.current) return
    if (
      pendingJumpTargetFileRef.current !== null &&
      pendingJumpTargetFileRef.current !== selectedFilePath
    ) {
      return
    }
    if (!isCurrentFileReadyForSelected || !isCompareFileReadyForSelected) return

    const targetRowIndex = changeBlocks.length > 0 ? changeBlocks[0].startRowIndex : null
    const frame = requestAnimationFrame(() => {
      if (targetRowIndex !== null) {
        diffRowRefs.current[targetRowIndex]?.scrollIntoView({
          behavior: "smooth",
          block: "center",
        })
      }
      pendingJumpToFirstChangeRef.current = false
      pendingJumpTargetFileRef.current = null
    })

    return () => cancelAnimationFrame(frame)
  }, [
    isCompareMode,
    selectedFilePath,
    changeBlocks,
    isCurrentFileReadyForSelected,
    isCompareFileReadyForSelected,
  ])

  return (
    <div className="flex h-full min-h-0 flex-col">
      {isCompareMode && compareRunLabel && (
        <div className="flex h-9 items-center gap-2 border-b px-3 text-xs">
          <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
            <div className="flex min-w-0 items-center gap-1 overflow-hidden">
              <span className="truncate font-medium">{compareRunLabel}</span>
              {compareRun?.created_at && (
                <span className="shrink-0 text-[10px] text-muted-foreground">
                  {formatTimeAgo(compareRun.created_at)}
                </span>
              )}
            </div>
            <ArrowRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
            <div className="flex min-w-0 items-center gap-1 overflow-hidden">
              <span className="truncate font-medium">{currentRunLabel}</span>
              {currentRun?.created_at && (
                <span className="shrink-0 text-[10px] text-muted-foreground">
                  {formatTimeAgo(currentRun.created_at)}
                </span>
              )}
            </div>
          </div>
          <div className="ml-2 flex shrink-0 items-center gap-2 tabular-nums">
            <span
              className={cn(
                "text-emerald-700 transition-opacity dark:text-emerald-400",
                isFetchingGlobalDiffSummary && "opacity-70"
              )}
            >
              +{globalDiffAddedCount}
            </span>
            <span
              className={cn(
                "text-red-500 transition-opacity dark:text-red-300",
                isFetchingGlobalDiffSummary && "opacity-70"
              )}
            >
              -{globalDiffRemovedCount}
            </span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 shrink-0 p-0 text-muted-foreground/70 hover:text-muted-foreground"
            aria-label="Stop comparing code runs"
            onClick={() => {
              setCompareRunId(null)
              setFrozenCompareDisplay(null)
              setActiveChangeNavIndex(0)
              pendingJumpToFirstChangeRef.current = false
              pendingJumpTargetFileRef.current = null
            }}
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}

      <div className="min-h-0 flex-1">
        {isLoadingTree && !treeData ? (
          <div className="h-full min-h-0" />
        ) : treeError ? (
          <div className="flex h-full min-h-0 flex-col">
            <div className="flex h-10 items-center border-b px-2">
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={onBack}
              >
                <ArrowLeft className="mr-1 h-3.5 w-3.5" />
                Back
              </Button>
            </div>
            <div className="p-4 text-sm text-destructive">
              Failed to load code tree.
            </div>
          </div>
        ) : !treeData?.available ? (
          <div className="flex h-full min-h-0 flex-col">
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
              <RunCodeCompareDialog
                currentRunId={runPath}
                selectedCompareRunId={compareRunId}
                onSelectCompareRun={setCompareRunId}
                disabled={!selectedFilePath}
              />
            </div>
            <div className="p-4 text-sm text-muted-foreground">
              This run does not have a local `code/source.zip` extracted yet.
            </div>
          </div>
        ) : (
          <div className="grid h-full min-h-0 overflow-hidden grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)]">
            <div className="flex min-h-0 flex-col border-b lg:border-b-0 lg:border-r">
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
                <RunCodeCompareDialog
                  currentRunId={runPath}
                  selectedCompareRunId={compareRunId}
                  onSelectCompareRun={setCompareRunId}
                  disabled={!selectedFilePath}
                />
              </div>
              <div className="min-h-0 flex-1 overflow-auto p-2">
                {renderNodes(displayTreeNodes, 0)}
              </div>
            </div>

            <div className="flex min-h-0 min-w-0 flex-col">
              <div className="flex h-10 items-center gap-2 border-b px-3">
                <span className="min-w-0 flex-1 truncate text-xs font-medium">
                  {selectedFilePath ?? "Select a file"}
                </span>
                {selectedFilePath && (
                  <>
                    <button
                      type="button"
                      onClick={handleCopySelectedFile}
                      disabled={!canCopySelectedFile}
                      className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
                      title={didCopySelectedFile ? "Copied!" : "Copy code"}
                    >
                      {didCopySelectedFile ? (
                        <CheckIcon className="h-3.5 w-3.5" />
                      ) : (
                        <CopyIcon className="h-3.5 w-3.5" />
                      )}
                    </button>
                    <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                      {selectedLanguage}
                    </span>
                  </>
                )}
                {isCompareMode && selectedFilePath && (
                  <div className="ml-1 flex shrink-0 items-center gap-2 text-[11px] tabular-nums">
                    <span className="text-emerald-700 dark:text-emerald-400">
                      +{fileDiffAddedCount}
                    </span>
                    <span className="text-red-500 dark:text-red-300">
                      -{fileDiffRemovedCount}
                    </span>
                  </div>
                )}
              </div>

              <div className="relative min-h-0 min-w-0 flex-1">
                <div className={cn("h-full overflow-auto", isCompareMode && "pb-14")}>
                  {!selectedFilePath ? (
                    <div className="p-4 text-sm text-muted-foreground">
                      Select a file on the left to view its source.
                    </div>
                  ) : isCompareMode ? (
                    isLoadingCompareTree && !compareTreeData ? (
                      <div className="h-full" />
                    ) : compareTreeError ? (
                      <div className="p-4 text-sm text-destructive">
                        Failed to load compared run code.
                      </div>
                    ) : compareRunId &&
                      !isLoadingCompareTree &&
                      compareTreeData &&
                      !compareTreeData.available ? (
                      <div className="p-4 text-sm text-muted-foreground">
                        Compared run has no downloaded code snapshot.
                      </div>
                    ) : isInitialCompareLoading ? (
                      <div className="h-full" />
                    ) : (currentHasSelectedFile &&
                        currentFileError &&
                        !currentFileDataForSelected) ||
                      (compareHasSelectedFile &&
                        compareFileError &&
                        !compareFileDataForSelected) ? (
                      <div className="p-4 text-sm text-destructive">
                        Failed to load one side of the comparison.
                      </div>
                    ) : currentBinaryForCompare || compareBinary ? (
                      <div className="p-4 text-sm text-muted-foreground">
                        Binary file comparison is not supported.
                      </div>
                    ) : (
                      <div
                        className={cn(
                          "min-h-full min-w-max font-sans text-xs leading-5 transition-opacity",
                          isCompareFetching && "opacity-50"
                        )}
                      >
                        {diffRows.map((row, rowIdx) => {
                          const rowTokens =
                            row.kind === "add"
                              ? rightDiffHighlightedLines[(row.rightLine ?? 1) - 1] ??
                                tokenizeLine(
                                  row.text.length > 0 ? row.text : " ",
                                  selectedLanguage
                                )
                              : leftDiffHighlightedLines[(row.leftLine ?? 1) - 1] ??
                                tokenizeLine(
                                  row.text.length > 0 ? row.text : " ",
                                  selectedLanguage
                                )

                          return (
                            <div
                              key={`diff-line-${rowIdx}`}
                              ref={(el) => {
                                if (row.kind !== "context") {
                                  diffRowRefs.current[rowIdx] = el
                                }
                              }}
                              className={cn(
                                "flex",
                                row.kind === "add" && "bg-emerald-500/18",
                                row.kind === "remove" && "bg-red-400/10"
                              )}
                            >
                              <div
                                className={cn(
                                  "sticky left-0 z-10 flex shrink-0 bg-background",
                                  row.kind === "add" &&
                                    "border-l-2 border-emerald-500/60",
                                  row.kind === "remove" &&
                                    "border-l-2 border-red-400/35"
                                )}
                              >
                                <span
                                  className={cn(
                                    "select-none w-6 px-1.5 py-0.5 text-right text-[11px]",
                                    row.kind === "add" &&
                                      "bg-emerald-500/18 text-emerald-700 dark:text-emerald-400",
                                    row.kind === "remove" &&
                                      "bg-red-400/10 text-red-500 dark:text-red-300",
                                    row.kind === "context" && "text-muted-foreground"
                                  )}
                                >
                                  {row.kind === "add" ? "+" : row.kind === "remove" ? "-" : " "}
                                </span>
                                <span
                                  className={cn(
                                    "select-none w-[3.25rem] border-r px-1.5 py-0.5 text-right text-[11px] text-muted-foreground",
                                    row.kind === "add" && "bg-emerald-500/18",
                                    row.kind === "remove" && "bg-red-400/10"
                                  )}
                                >
                                  {row.leftLine ?? ""}
                                </span>
                                <span
                                  className={cn(
                                    "select-none w-[3.25rem] border-r px-1.5 py-0.5 text-right text-[11px] text-muted-foreground",
                                    row.kind === "add" && "bg-emerald-500/18",
                                    row.kind === "remove" && "bg-red-400/10"
                                  )}
                                >
                                  {row.rightLine ?? ""}
                                </span>
                              </div>
                              <span className="whitespace-pre px-3 py-0.5">
                                {rowTokens.map((token, tokenIdx) => (
                                  <span
                                    key={`diff-line-${rowIdx}-token-${tokenIdx}`}
                                    className={cn(token.className)}
                                  >
                                    {token.text}
                                  </span>
                                ))}
                              </span>
                            </div>
                          )
                        })}
                      </div>
                    )
                  ) : isLoadingCurrentFile && !currentFileData ? (
                    <div className="h-full" />
                  ) : currentFileError && !currentFileData ? (
                    <div className="p-4 text-sm text-destructive">
                      Failed to load file content.
                    </div>
                  ) : currentBinary ? (
                    <div
                      className={cn(
                        "p-4 text-sm text-muted-foreground transition-opacity",
                        isFetchingCurrentFile && "opacity-50"
                      )}
                    >
                      Binary file preview is not supported.
                    </div>
                  ) : (
                    <div
                      className={cn(
                        "min-h-full min-w-max font-sans text-xs leading-5 transition-opacity",
                        isFetchingCurrentFile && "opacity-50"
                      )}
                    >
                      {currentFileData?.truncated && (
                        <div className="border-b px-3 py-1 text-[11px] text-muted-foreground">
                          Showing first 2MB of this file.
                        </div>
                      )}
                      {highlightedLines.map((lineTokens, lineIdx) => (
                        <div
                          key={`line-${lineIdx}`}
                          className="flex"
                        >
                          <div className="sticky left-0 z-10 flex shrink-0 bg-background">
                            <span className="select-none border-r bg-muted/35 w-[3.25rem] px-2 py-0.5 text-right text-[11px] text-muted-foreground">
                              {lineIdx + 1}
                            </span>
                          </div>
                          <span className="whitespace-pre px-3 py-0.5">
                            {lineTokens.map((token, tokenIdx) => (
                              <span
                                key={`line-${lineIdx}-token-${tokenIdx}`}
                                className={cn(token.className)}
                              >
                                {token.text}
                              </span>
                            ))}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {isCompareMode && selectedFilePath && (
                  <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center">
                    {changeBlocks.length === 0 ? (
                      <div className="pointer-events-auto rounded-full border border-border/75 bg-background/90 px-2 py-1 backdrop-blur-sm">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2 text-xs"
                          onClick={goToFirstChangedFile}
                          disabled={changedFilePaths.length === 0}
                        >
                          <span className="inline-flex items-center gap-1">
                            <span>See Changes</span>
                            <ChevronRight className="h-3.5 w-3.5" />
                          </span>
                        </Button>
                      </div>
                    ) : (
                      <div className="pointer-events-auto flex items-center gap-3 rounded-full border border-border/75 bg-background/90 px-3 py-1 backdrop-blur-sm">
                        <div className="flex items-center gap-1 text-xs">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => goToChange(-1)}
                            disabled={changeBlocks.length === 0}
                          >
                            <ChevronUp className="h-3.5 w-3.5" />
                          </Button>
                          <span className="min-w-[4rem] text-center tabular-nums">
                            {displayedChangeIndex} / {changeBlocks.length}
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => goToChange(1)}
                            disabled={changeBlocks.length === 0}
                          >
                            <ChevronDown className="h-3.5 w-3.5" />
                          </Button>
                        </div>

                        <div className="h-4 w-px bg-border" />

                        <div className="flex items-center gap-1 text-xs">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => goToChangedFile(-1)}
                            disabled={changedFilePaths.length === 0}
                          >
                            <ChevronLeft className="h-3.5 w-3.5" />
                          </Button>
                          <span className="min-w-[6rem] text-center tabular-nums">
                            {displayedFileIndex} / {changedFilePaths.length} files
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => goToChangedFile(1)}
                            disabled={changedFilePaths.length === 0}
                          >
                            <ChevronRight className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

