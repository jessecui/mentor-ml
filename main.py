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
from fastapi import FastAPI, HTTPException, Cookie, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DIAGRAMS_DIR = PROJECT_ROOT / "benchmark" / "corpus" / "images" / "diagrams"

# Global for Redis context manager cleanup
_redis_cm = None
_async_redis_cm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    
    Loads SigLIP scorer, creates Redis checkpointer, and initializes agent at startup.
    This ensures the model is loaded once and reused across requests.
    """
    global _redis_cm, _async_redis_cm
    
    print("🚀 Starting MentorML server...")
    
    # Import here to avoid loading heavy ML libraries until needed
    from model.scorer import SigLIPScorer
    from agent.graph import create_agent
    from langgraph.checkpoint.redis import RedisSaver
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    
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
    
    # Sync checkpointer for /chat endpoint
    _redis_cm = RedisSaver.from_conn_string(redis_url, ttl=ttl_config)
    checkpointer = _redis_cm.__enter__()
    
    # Initialize Redis indices (required for first use)
    print("📝 Setting up Redis checkpoint indices...")
    checkpointer.setup()
    
    app.state.checkpointer = checkpointer
    
    # Async checkpointer for /chat/stream endpoint
    _async_redis_cm = AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl_config)
    async_checkpointer = await _async_redis_cm.__aenter__()
    await async_checkpointer.asetup()
    app.state.async_checkpointer = async_checkpointer
    
    # Check if vision review is enabled (adds ~5-10s latency per diagram)
    enable_vision = os.getenv("ENABLE_VISION", "true").lower() in ("true", "1", "yes")
    app.state.enable_vision = enable_vision
    
    print("🤖 Creating LangGraph agents...")
    # Sync agent for /chat
    agent = create_agent(scorer, checkpointer=checkpointer, enable_vision=enable_vision)
    app.state.agent = agent
    
    # Async agent for /chat/stream
    async_agent = create_agent(scorer, checkpointer=async_checkpointer, enable_vision=enable_vision)
    app.state.async_agent = async_agent
    
    print("✅ MentorML ready!")
    yield
    
    print("👋 Shutting down MentorML server...")
    if _redis_cm:
        _redis_cm.__exit__(None, None, None)
    if _async_redis_cm:
        await _async_redis_cm.__aexit__(None, None, None)



app = FastAPI(
    title="MentorML API",
    description="A multimodal AI/ML teaching assistant with diagram retrieval",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None,      # Disable /docs
    redoc_url=None,     # Disable /redoc  
    openapi_url=None,   # Disable /openapi.json
)

# Mount static files for diagram images
from fastapi.staticfiles import StaticFiles
IMAGES_DIR = PROJECT_ROOT / "benchmark" / "corpus" / "images"
if IMAGES_DIR.exists():
    app.mount("/benchmark/corpus/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Serve frontend static files in production (after `npm run build`)
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    # Serve static assets (JS, CSS, etc.)
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend-assets")


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


class ValidatePasswordRequest(BaseModel):
    """Password validation request."""
    password: str


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


# --- Auth Dependency ---

async def require_auth(mentor_auth: str | None = Cookie(default=None)):
    """Require valid auth cookie to access protected endpoints."""
    if mentor_auth != "true":
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse, summary="Chat with MentorML", dependencies=[Depends(require_auth)])
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


# --- Streaming Endpoint ---

@app.post("/chat/stream", summary="Stream chat with MentorML (SSE)", dependencies=[Depends(require_auth)])
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response using Server-Sent Events.
    
    Events emitted:
    - plan: The agent's teaching plan (CoT reasoning)
    - diagram: A retrieved diagram with metadata
    - token: A text token from the response
    - done: Final signal with aggregated diagrams
    - error: Any error that occurred
    """
    from fastapi.responses import StreamingResponse
    from langchain_core.messages import HumanMessage
    
    # Use async agent for streaming
    agent = app.state.async_agent
    
    async def event_generator():
        """Generate SSE events from agent stream."""
        import re
        
        diagrams_collected: dict[str, dict] = {}
        text_buffer = ""
        plan_emitted = False
        current_node = ""
        
        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=request.message)]},
                config={"configurable": {"thread_id": request.thread_id}},
                version="v2",
            ):
                event_type = event.get("event")
                
                # Track which node we're in
                if event_type == "on_chain_start":
                    node_name = event.get("name", "")
                    if node_name in ("planner", "executor", "tools"):
                        current_node = node_name
                
                # Plan complete - emit teaching plan
                if event_type == "on_chain_end" and not plan_emitted:
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "plan" in output:
                        plan = output["plan"]
                        if plan:
                            plan_data = {
                                "topic": plan.topic,
                                "steps": plan.steps,
                                "diagrams_needed": plan.diagrams_needed
                            }
                            yield f"event: plan\ndata: {json.dumps(plan_data)}\n\n"
                            plan_emitted = True
                
                # Tool result - diagram retrieved
                if event_type == "on_tool_end":
                    tool_name = event.get("name", "")
                    if tool_name == "retrieve_diagram":
                        tool_output = event.get("data", {}).get("output")
                        # tool_output is a ToolMessage object
                        if tool_output and hasattr(tool_output, 'content'):
                            try:
                                diagram_data = json.loads(tool_output.content) if isinstance(tool_output.content, str) else tool_output.content
                                if isinstance(diagram_data, dict) and "id" in diagram_data:
                                    diagram_id = diagram_data["id"]
                                    diagrams_collected[diagram_id] = diagram_data
                                    yield f"event: diagram\ndata: {json.dumps(diagram_data)}\n\n"
                            except (json.JSONDecodeError, TypeError) as e:
                                print(f"❌ Diagram parse error: {e}")
                
                # LLM streaming tokens
                if event_type == "on_chat_model_stream":
                    # Get metadata to check which node this is from
                    metadata = event.get("metadata", {})
                    langgraph_node = metadata.get("langgraph_node", "")
                    
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        content = chunk.content
                        text = ""
                        # Handle Gemini 3's list content format
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text += part.get("text", "")
                        elif isinstance(content, str):
                            text = content
                        
                        if text:
                            if langgraph_node == "plan":
                                # Stream planning tokens as "thinking" event
                                yield f"event: thinking\ndata: {json.dumps(text)}\n\n"
                            elif langgraph_node == "execute":
                                # Stream executor tokens as main content
                                text_buffer += text
                                yield f"event: token\ndata: {json.dumps(text)}\n\n"
                            # Skip 'tools' node tokens (vision descriptions)
            
            # Find referenced diagrams in final text
            referenced_ids = set(re.findall(r'\[diagram:\s*(diagram_\d+)\]', text_buffer))
            final_diagrams = [
                diagrams_collected[did] for did in referenced_ids 
                if did in diagrams_collected
            ]
            
            # Done event with final state
            yield f"event: done\ndata: {json.dumps({'diagrams': final_diagrams})}\n\n"
            
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


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


@app.post("/validate-password", summary="Validate app password")
async def validate_password(request: ValidatePasswordRequest):
    """
    Validate the password for accessing the app.
    
    Password is stored in APP_PASSWORD environment variable.
    Comparison is case-insensitive with all whitespace removed.
    """
    app_password = os.getenv("APP_PASSWORD", "").strip().replace(" ", "").lower()
    user_password = request.password.strip().replace(" ", "").lower()
    
    if not app_password:
        raise HTTPException(
            status_code=500, 
            detail="APP_PASSWORD environment variable not configured"
        )
    
    return {"valid": user_password == app_password}


@app.get("/", summary="Health check / Frontend")
async def root():
    """Serve frontend index.html in production, health check in dev."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    # Fallback: health check for dev/API-only mode
    return {
        "status": "healthy",
        "service": "MentorML API",
        "version": "2.0.0",
    }


# Catch-all for SPA routing (must be last)
@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str):
    """Serve index.html for any unmatched routes (SPA client-side routing)."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Not found")


# --- Local Development ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
