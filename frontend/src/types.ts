export interface TeachingPlan {
  topic: string;
  steps: string[];
  diagrams_needed: string[];
}

export interface Diagram {
  id: string;
  score: number;
  query: string;
  description: string;
  vision_description: string;
  vision_latency_s: number;
  post_url: string;
  post_title?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  plan?: TeachingPlan;
  diagrams?: Diagram[];
  isStreaming?: boolean;
}

export interface StreamEvent {
  event: "token" | "plan" | "diagram" | "done" | "error";
  data: string;
}
