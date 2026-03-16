
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useState, useEffect } from "react"
import { Provider as JotaiProvider, useAtomValue } from "jotai"
import { TooltipProvider } from "@/components/ui/tooltip"
import { darkModeAtom } from "@/lib/atoms"

function DarkModeSync() {
  const darkMode = useAtomValue(darkModeAtom)
  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode)
  }, [darkMode])
  return null
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            refetchOnWindowFocus: false,
          },
        },
      })
  )

  return (
    <JotaiProvider>
      <DarkModeSync />
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>{children}</TooltipProvider>
      </QueryClientProvider>
    </JotaiProvider>
  )
}
