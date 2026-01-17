"""
MentorML FastAPI Server

Provides:
- POST /chat: Chat with the LangGraph agent (requires thread_id for conversation state)
- GET /diagrams/{diagram_id}: Serve diagram images
- GET /: Health check
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DIAGRAMS_DIR = PROJECT_ROOT / "benchmark" / "corpus" / "images" / "diagrams"

# Global for Redis context manager cleanup
_redis_cm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    
    Loads SigLIP scorer, creates Redis checkpointer, and initializes agent at startup.
    This ensures the model is loaded once and reused across requests.
    """
    global _redis_cm
    
    print("🚀 Starting MentorML server...")
    
    # Import here to avoid loading heavy ML libraries until needed
    from model.scorer import SigLIPScorer
    from agent.graph import create_agent
    from langgraph.checkpoint.redis import RedisSaver
    
    print("📦 Loading SigLIP scorer...")
    scorer = SigLIPScorer()
    app.state.scorer = scorer
    
    # Setup Redis checkpointer with 1 day TTL
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"🔗 Connecting to Redis at {redis_url}...")
    ttl_config = {
        "default_ttl": 1440,     # 24 hours in minutes
        "refresh_on_read": True,  # Reset TTL when conversation is accessed
    }
    _redis_cm = RedisSaver.from_conn_string(redis_url, ttl=ttl_config)
    checkpointer = _redis_cm.__enter__()
    
    # Initialize Redis indices (required for first use)
    print("📝 Setting up Redis checkpoint indices...")
    checkpointer.setup()
    
    app.state.checkpointer = checkpointer
    
    # Check if vision review is enabled (adds ~5-10s latency per diagram)
    enable_vision = os.getenv("ENABLE_VISION", "true").lower() in ("true", "1", "yes")
    
    print("🤖 Creating LangGraph agent...")
    agent = create_agent(scorer, checkpointer=checkpointer, enable_vision=enable_vision)
    app.state.agent = agent
    
    print("✅ MentorML ready!")
    yield
    
    print("👋 Shutting down MentorML server...")
    if _redis_cm:
        _redis_cm.__exit__(None, None, None)


app = FastAPI(
    title="MentorML API",
    description="A multimodal AI/ML teaching assistant with diagram retrieval",
    version="2.0.0",
    lifespan=lifespan,
)


# --- Request/Response Models ---

class ChatRequest(BaseModel):
    """Chat request with message and thread ID for conversation state."""
    message: str
    thread_id: str
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Explain the attention mechanism in transformers",
                    "thread_id": "user-123-session-1"
                }
            ]
        }
    }


class DiagramReference(BaseModel):
    """Reference to a retrieved diagram."""
    id: str
    score: float
    query: str  # The query the agent used to retrieve this diagram
    description: str  # Pre-generated description of the diagram
    vision_description: str  # Contextual description from agent's vision review
    vision_latency_s: float  # How long vision review took (seconds)
    post_url: str


class TeachingPlanResponse(BaseModel):
    """The agent's teaching plan (Chain-of-Thought reasoning)."""
    topic: str
    steps: list[str]
    diagrams_needed: list[str]


class ChatResponse(BaseModel):
    """Chat response with agent's message and any retrieved diagrams."""
    response: str
    diagrams: list[DiagramReference] = []
    plan: TeachingPlanResponse | None = None  # Agent's CoT reasoning


# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse, summary="Chat with MentorML")
async def chat(request: ChatRequest):
    """
    Send a message to the MentorML agent.
    
    The agent can retrieve relevant diagrams and provide explanations.
    Use the same thread_id across requests to maintain conversation context.
    """
    from langchain_core.messages import HumanMessage
    
    agent = app.state.agent
    
    try:
        # Invoke agent with thread_id for conversation state
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.thread_id}}
        )
        
        # Extract the final response
        messages = result.get("messages", [])
        
        if not messages:
            return ChatResponse(response="No response generated.", diagrams=[])
        
        # Find the last AI message with actual text content (not just tool calls)
        response_text = ""
        for msg in reversed(messages):
            if msg.type != "ai":
                continue
            
            # Skip if this message only has tool calls and no text
            if getattr(msg, "tool_calls", None) and not msg.content:
                continue
            
            content = msg.content
            
            # Handle Gemini 3's list content format
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                extracted = "".join(text_parts)
                if extracted.strip():
                    response_text = extracted
                    break
            elif isinstance(content, str) and content.strip():
                response_text = content
                break
        
        # Fallback: try .text attribute (Gemini 3 convenience method)
        if not response_text:
            for msg in reversed(messages):
                if msg.type == "ai":
                    text_attr = getattr(msg, 'text', None)
                    if text_attr and text_attr.strip():
                        response_text = text_attr
                        break
        
        if not response_text:
            return ChatResponse(response="No response generated.", diagrams=[])
        
        # Find diagram IDs actually referenced in the response (e.g., [diagram: diagram_075])
        import re
        referenced_ids = set(re.findall(r'\[diagram:\s*(diagram_\d+)\]', response_text))
        
        # Extract diagram metadata from tool calls (only for referenced diagrams)
        diagrams_by_id: dict[str, DiagramReference] = {}
        for msg in messages:
            # Check for tool messages (responses)
            if msg.type == "tool" and msg.name == "retrieve_diagram":
                try:
                    # Tool content is the returned dict as string
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(tool_result, dict) and "id" in tool_result:
                        diagram_id = tool_result["id"]
                        # Only include diagrams actually referenced in the response
                        if diagram_id in referenced_ids:
                            # Keep highest scoring version if duplicate
                            if diagram_id not in diagrams_by_id or tool_result.get("score", 0) > diagrams_by_id[diagram_id].score:
                                diagrams_by_id[diagram_id] = DiagramReference(
                                    id=diagram_id,
                                    score=tool_result.get("score", 0.0),
                                    query=tool_result.get("query", ""),
                                    description=tool_result.get("description", ""),
                                    vision_description=tool_result.get("vision_description", ""),
                                    vision_latency_s=tool_result.get("vision_latency_s", 0.0),
                                    post_url=tool_result.get("post_url", ""),
                                )
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Sort by score descending
        diagrams = sorted(diagrams_by_id.values(), key=lambda d: d.score, reverse=True)
        
        # Extract the teaching plan from result state
        plan_response = None
        plan = result.get("plan")
        if plan:
            plan_response = TeachingPlanResponse(
                topic=plan.topic,
                steps=plan.steps,
                diagrams_needed=plan.diagrams_needed
            )
        
        return ChatResponse(response=response_text, diagrams=diagrams, plan=plan_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/diagrams/{diagram_id}", summary="Get a diagram image")
async def get_diagram(diagram_id: str):
    """
    Retrieve a diagram image by ID.
    
    The diagram_id should be in format "diagram_XXX" (e.g., "diagram_042").
    Returns the PNG image file.
    """
    # Validate diagram_id format
    if not diagram_id.startswith("diagram_"):
        raise HTTPException(status_code=400, detail="Invalid diagram ID format")
    
    # Construct path
    filename = f"{diagram_id}.png"
    filepath = DIAGRAMS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Diagram not found: {diagram_id}")
    
    return FileResponse(filepath, media_type="image/png")


@app.get("/", summary="Health check")
async def health_check():
    """Check if the MentorML API is running."""
    return {
        "status": "healthy",
        "service": "MentorML API",
        "version": "2.0.0",
    }


# --- Local Development ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
