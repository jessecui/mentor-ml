import ReactMarkdown from "react-markdown";
import type { Message as MessageType, Diagram } from "../types";
import { DiagramCard } from "./DiagramCard";
import { cn } from "../lib/utils";
import { User, Bot, Loader2, ChevronDown, ChevronRight, Brain, StopCircle } from "lucide-react";
import { useState, type ReactNode } from "react";

interface MessageProps {
  message: MessageType;
}

// Component to show the thinking/planning process
function ThinkingSection({ thinking }: { thinking: string }) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Parse the JSON from the thinking content (it's wrapped in ```json ... ```)
  const parseThinking = (raw: string) => {
    try {
      const jsonMatch = raw.match(/```json\s*([\s\S]*?)\s*```/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[1]);
        return parsed;
      }
      // Try parsing directly
      return JSON.parse(raw);
    } catch {
      return null;
    }
  };

  const parsed = parseThinking(thinking);

  return (
    <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium text-amber-800 hover:bg-amber-100 cursor-pointer flex-wrap"
      >
        <span className="flex items-center gap-2">
          {isExpanded ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          <Brain className="h-4 w-4" />
          <span>Teaching Plan</span>
        </span>
        {parsed?.topic && (
          <span className="font-normal text-amber-600 basis-full sm:basis-auto sm:ml-1">— {parsed.topic}</span>
        )}
      </button>
      {isExpanded && parsed && (
        <div className="border-t border-amber-200 px-4 py-3 text-sm">
          {parsed.steps && (
            <div className="mb-3">
              <div className="mb-1 font-medium text-amber-900">Steps:</div>
              <ol className="ml-4 list-decimal space-y-1 text-amber-800">
                {parsed.steps.map((step: string, i: number) => (
                  <li key={i}>{step}</li>
                ))}
              </ol>
            </div>
          )}
          {parsed.diagrams_needed && (
            <div>
              <div className="mb-1 font-medium text-amber-900">Diagrams to find:</div>
              <ul className="ml-4 list-disc space-y-1 text-amber-800">
                {parsed.diagrams_needed.map((d: string, i: number) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === "user";

  // Build a map of diagrams by ID for quick lookup
  const diagramMap = new Map<string, Diagram>();
  message.diagrams?.forEach((d) => diagramMap.set(d.id, d));

  // Custom renderer to replace [diagram: diagram_xxx] with actual diagrams
  const renderContent = (content: string) => {
    // Split by diagram references
    const parts = content.split(/\[diagram:\s*(diagram_\d+)\]/g);
    const result: ReactNode[] = [];

    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 0) {
        // Text part
        if (parts[i]) {
          result.push(
            <ReactMarkdown
              key={`text-${i}`}
              components={{
                h1: ({ children }) => (
                  <h1 className="mb-4 mt-6 text-2xl font-bold first:mt-0">
                    {children}
                  </h1>
                ),
                h2: ({ children }) => (
                  <h2 className="mb-3 mt-5 text-xl font-semibold first:mt-0">
                    {children}
                  </h2>
                ),
                h3: ({ children }) => (
                  <h3 className="mb-2 mt-4 text-lg font-semibold first:mt-0">
                    {children}
                  </h3>
                ),
                p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
                ul: ({ children }) => (
                  <ul className="mb-3 ml-6 list-disc space-y-1">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="mb-3 ml-6 list-decimal space-y-1">{children}</ol>
                ),
                li: ({ children }) => <li>{children}</li>,
                strong: ({ children }) => (
                  <strong className="font-semibold">{children}</strong>
                ),
                em: ({ children }) => <em className="italic">{children}</em>,
                code: ({ children, className }) => {
                  const isBlock = className?.includes("language-");
                  return isBlock ? (
                    <pre className="my-3 overflow-x-auto rounded-lg bg-slate-900 p-4 text-sm text-slate-100">
                      <code>{children}</code>
                    </pre>
                  ) : (
                    <code className="rounded bg-muted px-1.5 py-0.5 text-sm font-mono">
                      {children}
                    </code>
                  );
                },
                blockquote: ({ children }) => (
                  <blockquote className="my-3 border-l-4 border-primary/30 pl-4 italic text-muted-foreground">
                    {children}
                  </blockquote>
                ),
              }}
            >
              {parts[i]}
            </ReactMarkdown>
          );
        }
      } else {
        // Diagram reference
        const diagramId = parts[i];
        const diagram = diagramMap.get(diagramId);
        if (diagram) {
          result.push(<DiagramCard key={`diagram-${diagramId}`} diagram={diagram} />);
        }
      }
    }

    return result;
  };

  return (
    <div
      className={cn(
        "flex gap-4 px-4 py-6",
        isUser ? "bg-white" : "bg-muted/30"
      )}
    >
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-primary text-primary-foreground" : "bg-slate-700 text-white"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 text-sm font-medium">
          {isUser ? "You" : "MentorML"}
        </div>
        
        {/* Show thinking section if available */}
        {!isUser && message.thinking && (
          <ThinkingSection thinking={message.thinking} />
        )}
        
        <div className="prose prose-slate max-w-none">
          {message.content ? (
            renderContent(message.content)
          ) : message.isStreaming ? (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>{message.thinking ? "Retrieving diagrams..." : "Planning..."}</span>
            </div>
          ) : null}
          
          {/* Show stopped indicator */}
          {message.wasStopped && (
            <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
              <StopCircle className="h-4 w-4" />
              <span>Response stopped</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
