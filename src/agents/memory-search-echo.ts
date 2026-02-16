type MemorySearchToolMeta = { toolName?: string };

const MEMORY_SEARCH_TOOL = "memory_search";
const MEMORY_SEARCH_ECHO_KEYS = new Set([
  "results",
  "provider",
  "model",
  "fallback",
  "citations",
  "disabled",
  "error",
]);

function unwrapJsonFence(text: string): string {
  const match = text.trim().match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  return match?.[1] ?? text;
}

function parseMemorySearchEcho(
  text: string,
): { results: unknown[]; error?: string; disabled?: boolean } | null {
  const candidate = unwrapJsonFence(text).trim();
  if (!candidate.startsWith("{") || !candidate.endsWith("}")) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(candidate);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return null;
  }
  const record = parsed as Record<string, unknown>;
  const keys = Object.keys(record);
  if (!keys.includes("results")) {
    return null;
  }
  if (!keys.every((key) => MEMORY_SEARCH_ECHO_KEYS.has(key))) {
    return null;
  }
  const results = record.results;
  if (!Array.isArray(results)) {
    return null;
  }
  const error = typeof record.error === "string" ? record.error.trim() : undefined;
  return {
    results,
    error: error && error.length > 0 ? error : undefined,
    disabled: record.disabled === true,
  };
}

function summarizeMemorySearchEcho(params: {
  results: unknown[];
  error?: string;
  disabled?: boolean;
}): string {
  if (params.results.length === 0) {
    if (params.error) {
      return `No relevant memory was found (${params.error}).`;
    }
    if (params.disabled) {
      return "Memory recall is currently disabled.";
    }
    return "No relevant memory was found.";
  }

  const previews: string[] = [];
  for (const entry of params.results.slice(0, 3)) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const record = entry as Record<string, unknown>;
    const snippetRaw = typeof record.snippet === "string" ? record.snippet : "";
    const snippet = snippetRaw.replace(/\s+/g, " ").trim();
    const compactSnippet = snippet.length > 200 ? `${snippet.slice(0, 197).trimEnd()}...` : snippet;
    const path = typeof record.path === "string" ? record.path : undefined;
    const startLine = typeof record.startLine === "number" ? record.startLine : undefined;
    const source =
      path && startLine !== undefined ? ` (${path}#L${startLine})` : path ? ` (${path})` : "";
    if (compactSnippet) {
      previews.push(`- ${compactSnippet}${source}`);
    }
  }
  if (previews.length === 0) {
    return `I found ${params.results.length} memory result(s), but could not extract readable snippets.`;
  }
  return `I found ${params.results.length} relevant memory snippet(s):\n${previews.join("\n")}`;
}

export function didUseMemorySearch(toolMetas: readonly MemorySearchToolMeta[]): boolean {
  return toolMetas.some((entry) => {
    if (typeof entry.toolName !== "string") {
      return false;
    }
    return entry.toolName.trim().toLowerCase() === MEMORY_SEARCH_TOOL;
  });
}

export function normalizeMemorySearchEchoText(params: {
  text: string;
  memorySearchUsed: boolean;
}): string {
  if (!params.memorySearchUsed) {
    return params.text;
  }
  const echo = parseMemorySearchEcho(params.text);
  if (!echo) {
    return params.text;
  }
  return summarizeMemorySearchEcho(echo);
}
