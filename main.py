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
    
    print("📦 Loading SigLIP scorer (this may take a moment on first run)...")
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
    
    print("🤖 Creating LangGraph agent...")
    agent = create_agent(scorer, checkpointer=checkpointer)
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
    context: str
    post_url: str


class ChatResponse(BaseModel):
    """Chat response with agent's message and any retrieved diagrams."""
    response: str
    diagrams: list[DiagramReference] = []


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
        
        # Get the last AI message content
        last_content = messages[-1].content
        
        # Handle both string and list content formats (Gemini returns list)
        if isinstance(last_content, str):
            response_text = last_content
        elif isinstance(last_content, list):
            # Extract text from content parts
            text_parts = []
            for part in last_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            response_text = "".join(text_parts)
        else:
            response_text = str(last_content)
        
        # Extract diagram references from tool calls
        diagrams = []
        for msg in messages:
            if hasattr(msg, "tool_calls"):
                for tool_call in msg.tool_calls:
                    if tool_call.get("name") == "retrieve_diagram":
                        # Find the corresponding tool response
                        pass
            # Check for tool messages (responses)
            if msg.type == "tool" and msg.name == "retrieve_diagram":
                try:
                    # Tool content is the returned dict as string
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(tool_result, dict) and "id" in tool_result:
                        diagrams.append(DiagramReference(
                            id=tool_result["id"],
                            score=tool_result.get("score", 0.0),
                            context=tool_result.get("context", ""),
                            post_url=tool_result.get("post_url", ""),
                        ))
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return ChatResponse(response=response_text, diagrams=diagrams)
        
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
