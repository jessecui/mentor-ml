import { useRef, useEffect } from "react";
import { Message } from "./Message";
import type { Message as MessageType } from "../types";
import { GraduationCap } from "lucide-react";

interface MessageListProps {
  messages: MessageType[];
  onFillSuggestion?: (message: string) => void;
}

export function MessageList({ messages, onFillSuggestion }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center p-8 text-center">
        <div className="mb-4 rounded-full bg-primary/10 p-4">
          <GraduationCap className="h-12 w-12 text-primary" />
        </div>
        <h2 className="mb-2 text-2xl font-semibold">Welcome to MentorML</h2>
        <p className="mb-6 max-w-md text-muted-foreground">
          Your AI tutor for machine learning concepts. Ask me anything about
          neural networks, transformers, training algorithms, and more!
        </p>
        <div className="grid gap-2 text-sm">
          <SuggestionButton text="What is the attention mechanism?" onFill={onFillSuggestion} />
          <SuggestionButton text="Explain backpropagation step by step" onFill={onFillSuggestion} />
          <SuggestionButton text="How do transformers work?" onFill={onFillSuggestion} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl">
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function SuggestionButton({ text, onFill }: { text: string; onFill?: (message: string) => void }) {
  return (
    <button
      type="button"
      className="rounded-lg border border-border bg-white px-4 py-2 text-left text-muted-foreground transition-colors hover:border-primary hover:text-foreground"
      onClick={() => onFill?.(text)}
    >
      {text}
    </button>
  );
}
