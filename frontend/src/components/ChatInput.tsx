import { useState, useRef, useEffect } from "react";
import { Send, Square, Trash2 } from "lucide-react";
import { cn } from "../lib/utils";

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop: () => void;
  onClear: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

export function ChatInput({
  onSend,
  onStop,
  onClear,
  isStreaming,
  disabled,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isStreaming && !disabled) {
      onSend(input);
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t border-border bg-white p-4">
      <div className="mx-auto flex max-w-3xl items-end gap-2">
        <button
          type="button"
          onClick={onClear}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          title="Clear conversation"
        >
          <Trash2 className="h-5 w-5" />
        </button>
        <div className="relative flex-1">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about ML concepts..."
            disabled={disabled}
            rows={1}
            className={cn(
              "w-full resize-none rounded-xl border border-border bg-white px-4 py-3 pr-12 text-base",
              "placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary",
              "disabled:cursor-not-allowed disabled:opacity-50"
            )}
          />
        </div>
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-red-500 text-white transition-colors hover:bg-red-600"
            title="Stop generating"
          >
            <Square className="h-4 w-4 fill-current" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim() || disabled}
            className={cn(
              "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg transition-colors",
              input.trim() && !disabled
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "bg-muted text-muted-foreground cursor-not-allowed"
            )}
            title="Send message"
          >
            <Send className="h-5 w-5" />
          </button>
        )}
      </div>
    </form>
  );
}
