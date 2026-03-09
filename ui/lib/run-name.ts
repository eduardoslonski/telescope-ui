export const SIDEBAR_MAX_RUN_NAME_CHARS = 16

const SIDEBAR_MIN_RUN_NAME_CHARS = 6
const SIDEBAR_SUFFIX_LENGTH = 4
const SIDEBAR_MIN_PREFIX_LENGTH = 2

export interface SidebarRunNameParts {
  prefix: string
  suffix: string
  isTruncated: boolean
}

/**
 * Mirror the run-name truncation used in the left sidebar:
 * keep a fixed suffix and collapse the middle into "...".
 */
export function getSidebarRunNameParts(
  name: string,
  maxChars: number = SIDEBAR_MAX_RUN_NAME_CHARS
): SidebarRunNameParts {
  const safeMaxChars = Math.max(SIDEBAR_MIN_RUN_NAME_CHARS, maxChars)
  if (!name || name.length <= safeMaxChars) {
    return {
      prefix: name,
      suffix: "",
      isTruncated: false,
    }
  }

  const prefixLen = Math.max(
    SIDEBAR_MIN_PREFIX_LENGTH,
    safeMaxChars - SIDEBAR_SUFFIX_LENGTH
  )

  return {
    prefix: name.slice(0, prefixLen),
    suffix: name.slice(-SIDEBAR_SUFFIX_LENGTH),
    isTruncated: true,
  }
}

