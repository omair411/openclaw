import type { AssistantMessage } from "@mariozechner/pi-ai";
import { describe, expect, it, vi } from "vitest";
import { subscribeEmbeddedPiSession } from "./pi-embedded-subscribe.js";

type StubSession = {
  subscribe: (fn: (evt: unknown) => void) => () => void;
};

const MEMORY_ECHO =
  '{"results":[],"provider":"local","model":"hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf","citations":"auto"}';
const NORMALIZED_MESSAGE = "No relevant memory was found.";

function setup(params?: { blockReplyBreak?: "text_end" | "message_end" }) {
  let handler: ((evt: unknown) => void) | undefined;
  const session: StubSession = {
    subscribe: (fn) => {
      handler = fn;
      return () => {};
    },
  };
  const onBlockReply = vi.fn();
  const subscription = subscribeEmbeddedPiSession({
    session: session as unknown as Parameters<typeof subscribeEmbeddedPiSession>[0]["session"],
    runId: "run",
    onBlockReply,
    blockReplyBreak: params?.blockReplyBreak ?? "text_end",
  });
  return { handler, onBlockReply, subscription };
}

describe("subscribeEmbeddedPiSession", () => {
  it("normalizes echoed memory_search JSON for text_end block replies", () => {
    const { handler, onBlockReply, subscription } = setup({ blockReplyBreak: "text_end" });

    handler?.({
      type: "tool_execution_end",
      toolName: "memory_search",
      toolCallId: "tool-1",
      isError: false,
      result: { results: [] },
    });

    handler?.({ type: "message_start", message: { role: "assistant" } });
    handler?.({
      type: "message_update",
      message: { role: "assistant" },
      assistantMessageEvent: {
        type: "text_delta",
        delta: MEMORY_ECHO,
      },
    });
    handler?.({
      type: "message_update",
      message: { role: "assistant" },
      assistantMessageEvent: { type: "text_end" },
    });

    expect(onBlockReply).toHaveBeenCalledTimes(1);
    expect(onBlockReply.mock.calls[0]?.[0]?.text).toBe(NORMALIZED_MESSAGE);
    expect(subscription.assistantTexts).toEqual([NORMALIZED_MESSAGE]);
  });

  it("normalizes echoed memory_search JSON for message_end block replies", () => {
    const { handler, onBlockReply, subscription } = setup({ blockReplyBreak: "message_end" });

    handler?.({
      type: "tool_execution_end",
      toolName: "memory_search",
      toolCallId: "tool-1",
      isError: false,
      result: { results: [] },
    });

    const assistantMessage = {
      role: "assistant",
      content: [{ type: "text", text: MEMORY_ECHO }],
    } as AssistantMessage;
    handler?.({ type: "message_end", message: assistantMessage });

    expect(onBlockReply).toHaveBeenCalledTimes(1);
    expect(onBlockReply.mock.calls[0]?.[0]?.text).toBe(NORMALIZED_MESSAGE);
    expect(subscription.assistantTexts).toEqual([NORMALIZED_MESSAGE]);
  });

  it("does not rewrite memory-like JSON when memory_search was not used", () => {
    const { handler, onBlockReply } = setup({ blockReplyBreak: "message_end" });

    const assistantMessage = {
      role: "assistant",
      content: [{ type: "text", text: MEMORY_ECHO }],
    } as AssistantMessage;
    handler?.({ type: "message_end", message: assistantMessage });

    expect(onBlockReply).toHaveBeenCalledTimes(1);
    expect(onBlockReply.mock.calls[0]?.[0]?.text).toBe(MEMORY_ECHO);
  });
});
