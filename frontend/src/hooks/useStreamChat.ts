import { useState, useCallback, useRef } from "react";
import type { Message, TeachingPlan, Diagram } from "../types";

interface UseStreamChatOptions {
  onPlan?: (plan: TeachingPlan) => void;
  onDiagram?: (diagram: Diagram) => void;
}

export function useStreamChat(options: UseStreamChatOptions = {}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const threadIdRef = useRef<string>(`thread-${Date.now()}`);

  const sendMessage = useCallback(
    async (content: string) => {
      if (isStreaming || !content.trim()) return;

      // Add user message
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        role: "user",
        content: content.trim(),
      };

      // Add placeholder assistant message
      const assistantId = `assistant-${Date.now()}`;
      const assistantMessage: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsStreaming(true);

      // Create abort controller
      abortControllerRef.current = new AbortController();

      try {
        const response = await fetch("/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: content.trim(),
            thread_id: threadIdRef.current,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";
        let accumulatedContent = "";
        let accumulatedThinking = "";
        let plan: TeachingPlan | undefined;
        const diagrams: Map<string, Diagram> = new Map();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Parse SSE events
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; // Keep incomplete line in buffer

          let eventType = "";
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7);
            } else if (line.startsWith("data: ") && eventType) {
              const data = line.slice(6);
              try {
                if (eventType === "thinking") {
                  const token = JSON.parse(data) as string;
                  accumulatedThinking += token;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId
                        ? { ...msg, thinking: accumulatedThinking }
                        : msg
                    )
                  );
                } else if (eventType === "token") {
                  const token = JSON.parse(data) as string;
                  accumulatedContent += token;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    )
                  );
                } else if (eventType === "plan") {
                  plan = JSON.parse(data) as TeachingPlan;
                  options.onPlan?.(plan);
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId ? { ...msg, plan } : msg
                    )
                  );
                } else if (eventType === "diagram") {
                  const diagram = JSON.parse(data) as Diagram;
                  diagrams.set(diagram.id, diagram);
                  options.onDiagram?.(diagram);
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId
                        ? { ...msg, diagrams: Array.from(diagrams.values()) }
                        : msg
                    )
                  );
                } else if (eventType === "done") {
                  const doneData = JSON.parse(data) as { diagrams: Diagram[] };
                  // Update with final diagrams (only referenced ones)
                  const finalDiagrams = doneData.diagrams;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantId
                        ? {
                            ...msg,
                            diagrams: finalDiagrams,
                            isStreaming: false,
                          }
                        : msg
                    )
                  );
                } else if (eventType === "error") {
                  const errorData = JSON.parse(data) as { error: string };
                  throw new Error(errorData.error);
                }
              } catch (e) {
                if (e instanceof SyntaxError) {
                  console.warn("Failed to parse SSE data:", data);
                } else {
                  throw e;
                }
              }
              eventType = "";
            }
          }
        }
      } catch (error) {
        if ((error as Error).name === "AbortError") {
          console.log("Request aborted");
          // Mark message as stopped
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, isStreaming: false, wasStopped: true }
                : msg
            )
          );
        } else {
          console.error("Stream error:", error);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    content:
                      msg.content ||
                      "Sorry, an error occurred. Please try again.",
                    isStreaming: false,
                  }
                : msg
            )
          );
        }
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [isStreaming, options]
  );

  const stopStreaming = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    threadIdRef.current = `thread-${Date.now()}`;
  }, []);

  return {
    messages,
    isStreaming,
    sendMessage,
    stopStreaming,
    clearMessages,
  };
}
