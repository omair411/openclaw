import type { StreamFn } from "@mariozechner/pi-agent-core";
import type {
  AssistantMessage,
  AssistantMessageEventStream,
  Context,
  ImageContent,
  Message,
  SimpleStreamOptions,
  TextContent,
  Tool,
  ToolCall,
} from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";
import crypto from "node:crypto";
import { log } from "./pi-embedded-runner/logger.js";

type OllamaToolCall = {
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string | Record<string, unknown> };
};

type OllamaMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  images?: string[];
  tool_calls?: OllamaToolCall[];
  tool_call_id?: string;
  name?: string;
};

type OllamaChatChunk = {
  message?: { role?: string; content?: string; tool_calls?: OllamaToolCall[] };
  done?: boolean;
  done_reason?: string;
  prompt_eval_count?: number;
  eval_count?: number;
};

function stripDataPrefix(data: string): string {
  const match = /^data:[^,]+,/.exec(data);
  return match ? data.slice(match[0].length) : data;
}

function flattenContent(parts: (TextContent | ImageContent)[]): { text: string; images: string[] } {
  const textParts: string[] = [];
  const images: string[] = [];
  for (const part of parts) {
    if (part.type === "text") {
      if (part.text) {
        textParts.push(part.text);
      }
    } else if (part.type === "image") {
      images.push(stripDataPrefix(part.data));
    }
  }
  return {
    text: textParts.join(""),
    images,
  };
}

function resolveSystemPrompt(context: Context): string | undefined {
  const systemPrompt = (context as { systemPrompt?: string; system?: string }).systemPrompt;
  if (typeof systemPrompt === "string" && systemPrompt.trim()) {
    return systemPrompt;
  }
  const system = (context as { system?: string }).system;
  if (typeof system === "string" && system.trim()) {
    return system;
  }
  return undefined;
}

function buildOllamaMessages(context: Context): OllamaMessage[] {
  const messages: OllamaMessage[] = [];
  const systemPrompt = resolveSystemPrompt(context);
  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }

  for (const msg of context.messages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        messages.push({ role: "user", content: msg.content });
      } else {
        const { text, images } = flattenContent(msg.content);
        const entry: OllamaMessage = { role: "user", content: text };
        if (images.length > 0) {
          entry.images = images;
        }
        messages.push(entry);
      }
      continue;
    }

    if (msg.role === "assistant") {
      let text = "";
      const toolCalls: OllamaToolCall[] = [];
      for (const block of msg.content) {
        if (block.type === "text") {
          text += block.text;
        } else if (block.type === "thinking") {
          text += `\n<thinking>\n${block.thinking}\n</thinking>\n`;
        } else if (block.type === "toolCall") {
          toolCalls.push({
            id: block.id,
            type: "function",
            function: {
              name: block.name,
              arguments: normalizeRequestToolArguments(block.arguments),
            },
          });
        }
      }
      const entry: OllamaMessage = { role: "assistant", content: text };
      if (toolCalls.length > 0) {
        entry.tool_calls = toolCalls;
      }
      messages.push(entry);
      continue;
    }

    if (msg.role === "toolResult") {
      const text = msg.content
        .filter((part): part is TextContent => part.type === "text")
        .map((part) => part.text)
        .join("");
      messages.push({
        role: "tool",
        content: text,
        tool_call_id: msg.toolCallId,
        name: msg.toolName,
      });
    }
  }

  return messages;
}

function buildOllamaTools(tools: Tool[] | undefined): Array<Record<string, unknown>> | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }));
}

function resolveOllamaBaseUrl(modelBaseUrl: string | undefined): string {
  const raw =
    modelBaseUrl?.trim() ??
    process.env.OLLAMA_API_BASE_URL?.trim() ??
    process.env.OLLAMA_BASE_URL?.trim() ??
    "http://127.0.0.1:11434";
  const withoutTrailing = raw.replace(/\/+$/, "");
  return withoutTrailing.endsWith("/v1") ? withoutTrailing.slice(0, -3) : withoutTrailing;
}

function ensureToolCallArguments(
  value: string | Record<string, unknown> | undefined,
): Record<string, unknown> {
  if (!value) {
    return {};
  }
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object") {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return {};
    }
  }
  return value as Record<string, unknown>;
}

function normalizeRequestToolArguments(value: unknown): Record<string, unknown> {
  if (!value) {
    return {};
  }
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object") {
        return parsed as Record<string, unknown>;
      }
    } catch {
      if (process.env.OPENCLAW_OLLAMA_DEBUG === "1") {
        log.debug("[ollama-native] toolCall arguments not valid JSON; sending empty object");
      }
      return {};
    }
  }
  if (typeof value === "object") {
    return value as Record<string, unknown>;
  }
  return {};
}

function mapDoneReason(
  reason: string | undefined,
  hasToolCalls: boolean,
): "stop" | "length" | "toolUse" {
  if (hasToolCalls) {
    return "toolUse";
  }
  if (reason === "length") {
    return "length";
  }
  return "stop";
}

function buildAssistantMessage(model: {
  api: string;
  provider: string;
  id: string;
}): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "stop",
    timestamp: Date.now(),
  };
}

export const streamOllamaNative: StreamFn = (model, context, options) => {
  const stream = createAssistantMessageEventStream();
  (async () => {
    const output = buildAssistantMessage(model);
    let currentTextBlock: { type: "text"; text: string } | null = null;
    let sawToolCalls = false;

    const finishTextBlock = () => {
      if (currentTextBlock) {
        const index = output.content.length - 1;
        stream.push({
          type: "text_end",
          contentIndex: index,
          content: currentTextBlock.text,
          partial: output,
        });
        currentTextBlock = null;
      }
    };

    try {
      const messages = buildOllamaMessages(context as Context);
      const tools = buildOllamaTools(context.tools);
      const baseOptions: Record<string, unknown> = {};
      if (options?.temperature !== undefined) {
        baseOptions.temperature = options.temperature;
      }
      if (options?.maxTokens !== undefined) {
        baseOptions.num_predict = options.maxTokens;
      }

      const payload: Record<string, unknown> = {
        model: model.id,
        messages,
        stream: true,
      };
      if (tools) {
        payload.tools = tools;
      }
      if (Object.keys(baseOptions).length > 0) {
        payload.options = baseOptions;
      }

      options?.onPayload?.(payload);

      if (process.env.OPENCLAW_OLLAMA_DEBUG === "1") {
        log.debug(
          `[ollama-native] payload summary: ${JSON.stringify({
            model: model.id,
            messageCount: messages.length,
            roles: messages.map((m) => m.role),
            toolCount: tools?.length ?? 0,
            options: payload.options ?? {},
          })}`,
        );
      }

      const baseUrl = resolveOllamaBaseUrl(model.baseUrl);
      const res = await fetch(`${baseUrl}/api/chat`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...model.headers,
          ...options?.headers,
        },
        body: JSON.stringify(payload),
        signal: options?.signal,
      });

      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => "");
        if (process.env.OPENCLAW_OLLAMA_DEBUG === "1") {
          log.warn(`[ollama-native] Ollama API error ${res.status}: ${text || "(empty body)"}`);
        }
        throw new Error(`Ollama API error (${res.status}): ${text}`);
      }

      stream.push({ type: "start", partial: output });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const handleChunk = (chunk: OllamaChatChunk) => {
        if (chunk.message?.content) {
          if (!currentTextBlock) {
            currentTextBlock = { type: "text", text: "" };
            output.content.push(currentTextBlock);
            stream.push({
              type: "text_start",
              contentIndex: output.content.length - 1,
              partial: output,
            });
          }
          currentTextBlock.text += chunk.message.content;
          stream.push({
            type: "text_delta",
            contentIndex: output.content.length - 1,
            delta: chunk.message.content,
            partial: output,
          });
        }

        if (chunk.message?.tool_calls && chunk.message.tool_calls.length > 0) {
          finishTextBlock();
          for (const toolCall of chunk.message.tool_calls) {
            sawToolCalls = true;
            const toolCallBlock: ToolCall = {
              type: "toolCall",
              id: toolCall.id ?? crypto.randomUUID(),
              name: toolCall.function?.name ?? "unknown",
              arguments: ensureToolCallArguments(toolCall.function?.arguments),
            };
            output.content.push(toolCallBlock);
            const index = output.content.length - 1;
            stream.push({ type: "toolcall_start", contentIndex: index, partial: output });
            stream.push({
              type: "toolcall_end",
              contentIndex: index,
              toolCall: toolCallBlock,
              partial: output,
            });
          }
        }

        if (chunk.done) {
          finishTextBlock();
          const input = chunk.prompt_eval_count ?? 0;
          const outputTokens = chunk.eval_count ?? 0;
          output.usage = {
            input,
            output: outputTokens,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: input + outputTokens,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          };
          output.stopReason = mapDoneReason(chunk.done_reason, sawToolCalls);
          stream.push({ type: "done", reason: output.stopReason, message: output });
          stream.end();
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        let lineBreak = buffer.indexOf("\n");
        while (lineBreak >= 0) {
          const line = buffer.slice(0, lineBreak).trim();
          buffer = buffer.slice(lineBreak + 1);
          if (line) {
            const chunk = JSON.parse(line) as OllamaChatChunk;
            handleChunk(chunk);
          }
          lineBreak = buffer.indexOf("\n");
        }
      }

      const leftover = buffer.trim();
      if (leftover) {
        const chunk = JSON.parse(leftover) as OllamaChatChunk;
        handleChunk(chunk);
      }
    } catch (error) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();

  return stream as AssistantMessageEventStream;
};
