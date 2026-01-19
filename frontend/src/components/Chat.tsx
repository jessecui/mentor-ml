import { useStreamChat } from "../hooks/useStreamChat";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { GraduationCap } from "lucide-react";

export function Chat() {
  const { messages, isStreaming, sendMessage, stopStreaming, clearMessages } =
    useStreamChat();

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-border bg-white px-4 py-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <GraduationCap className="h-5 w-5" />
        </div>
        <div>
          <h1 className="font-semibold">MentorML</h1>
          <p className="text-xs text-muted-foreground">
            AI-powered ML tutoring with visual explanations
          </p>
        </div>
      </header>

      {/* Messages */}
      <MessageList messages={messages} />

      {/* Input */}
      <ChatInput
        onSend={sendMessage}
        onStop={stopStreaming}
        onClear={clearMessages}
        isStreaming={isStreaming}
      />
    </div>
  );
}
