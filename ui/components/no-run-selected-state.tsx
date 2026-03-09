import { useAtomValue, useSetAtom } from "jotai"
import { useRuns } from "@/hooks/use-run-data"
import {
  wandbApiKeyAtom,
  wandbConfigDialogOpenAtom,
  knownProjectsDialogOpenAtom,
} from "@/lib/atoms"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

interface NoRunSelectedStateProps {
  description: string
}

export function NoRunSelectedState({ description }: NoRunSelectedStateProps) {
  const apiKeyOptimistic = useAtomValue(wandbApiKeyAtom)
  const setConfigDialogOpen = useSetAtom(wandbConfigDialogOpenAtom)
  const setProjectsDialogOpen = useSetAtom(knownProjectsDialogOpenAtom)
  const { data: runsData, isLoading: isLoadingRuns } = useRuns()

  const runs = runsData?.runs ?? []
  const discovery = runsData?.discovery
  const hasApiKey = runsData?.has_wandb_key ?? apiKeyOptimistic
  const hasRuns = runs.length > 0
  const hasKnownProjects = runsData?.has_known_projects ?? false
  const isDiscovering = discovery?.status === "discovering"
  const showAddWandbKeyPrompt =
    !isLoadingRuns && !isDiscovering && !hasApiKey && !hasRuns
  const showAddProjectPrompt =
    !isLoadingRuns &&
    !isDiscovering &&
    !showAddWandbKeyPrompt &&
    hasApiKey &&
    !hasRuns &&
    !hasKnownProjects
  const showNoRunsYetState =
    !showAddWandbKeyPrompt && !showAddProjectPrompt && !hasRuns

  if (isDiscovering) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <Card className="max-w-md w-full border-dashed">
          <CardHeader className="items-center text-center">
            <CardTitle>Fetching runs from W&B…</CardTitle>
            <CardDescription className="flex items-center gap-1.5 font-medium">
              <Spinner className="h-3 w-3 shrink-0" />
              {discovery?.runs_found ?? 0} found
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    )
  }

  if (showNoRunsYetState) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <div className="text-xs text-muted-foreground">No runs yet</div>
      </div>
    )
  }

  return (
    <div className="flex h-full items-center justify-center p-8">
      <Card className="max-w-md w-full border-dashed">
        <CardHeader className="text-center items-center">
          {showAddWandbKeyPrompt ? (
            <>
              <CardTitle>No wandb key added yet</CardTitle>
              <CardDescription>
                Add your W&amp;B API key to discover and sync runs.
              </CardDescription>
              <Button
                className="mt-4 bg-green-600 hover:bg-green-700"
                onClick={() => setConfigDialogOpen(true)}
              >
                Add wandb key
              </Button>
            </>
          ) : showAddProjectPrompt ? (
            <>
              <CardTitle>Add a project</CardTitle>
              <CardDescription>
                Add a W&amp;B project to start discovering telescope-tagged runs.
              </CardDescription>
              <Button
                className="mt-4"
                onClick={() => setProjectsDialogOpen(true)}
              >
                Add project
              </Button>
            </>
          ) : (
            <>
              <CardTitle>No Run Selected</CardTitle>
              <CardDescription>{description}</CardDescription>
            </>
          )}
        </CardHeader>
      </Card>
    </div>
  )
}
