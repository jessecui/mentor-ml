"""
MCP server exposing diagram retrieval as the `retrieve_diagram` tool.

Owns the SigLIPScorer + precomputed embeddings, runs in a separate subprocess,
and is reached over stdio by the agent's MCP client. Returns the same dict
shape that the in-process tool returned previously, so the agent contract is
unchanged.

Run standalone:
    python -m mcp_server.diagram_server
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp.server.fastmcp import FastMCP

from model.scorer import SigLIPScorer

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_METADATA = PROJECT_ROOT / "benchmark" / "corpus" / "metadata" / "corpus_with_queries.json"
DIAGRAMS_DIR = PROJECT_ROOT / "benchmark" / "corpus" / "images" / "diagrams"
EMBEDDINGS_FILE = PROJECT_ROOT / "benchmark" / "corpus" / "embeddings" / "siglip_embeddings.npz"

VISION_REVIEW_PROMPT = """You searched for: "{query}"

Look at this diagram carefully. In 1-2 sentences, describe what this specific diagram shows
and how it helps explain the concept. Be specific about what visual elements are present
(arrows, boxes, flow direction, labels, colors, etc.).

Start with "This diagram shows..." and focus on what's visually depicted."""


def _log(msg: str) -> None:
    # MCP uses stdout for protocol; logs must go to stderr.
    print(msg, file=sys.stderr, flush=True)


_log("📦 Loading SigLIP scorer in MCP server...")
_scorer = SigLIPScorer()

_log(f"📊 Loading pre-computed embeddings from {EMBEDDINGS_FILE}...")
if not EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(
        f"Pre-computed embeddings not found: {EMBEDDINGS_FILE}\n"
        "Run: python scripts/precompute_embeddings.py"
    )
_emb_data = np.load(EMBEDDINGS_FILE)
_image_embeddings: np.ndarray = _emb_data["embeddings"]
_temperature = float(_emb_data["temperature"])
_bias = float(_emb_data["bias"])
_log(f"   ✅ {_image_embeddings.shape[0]} embeddings (dim={_image_embeddings.shape[1]})")

with open(CORPUS_METADATA) as f:
    _corpus: list[dict] = json.load(f)
_log(f"   ✅ {len(_corpus)} corpus entries loaded")

_ENABLE_VISION = os.getenv("ENABLE_VISION", "true").lower() in ("true", "1", "yes")
_log(f"   👁️ Vision review: {'enabled' if _ENABLE_VISION else 'disabled (global override)'}")


def _vision_review_diagram(diagram_id: str, query: str) -> tuple[str, float]:
    """Use Gemini vision to describe how a diagram relates to the query."""
    image_path = DIAGRAMS_DIR / f"{diagram_id}.png"
    if not image_path.exists():
        return "", 0.0

    start_time = time.time()
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = VISION_REVIEW_PROMPT.format(query=query)
        vision_message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ])

        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=1.0, thinking_level="low")
        response = llm.invoke([vision_message])

        latency = time.time() - start_time
        description = getattr(response, 'text', None) or (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        _log(f"   👁️ Vision review for {diagram_id}: {latency:.2f}s")
        return description, latency
    except Exception as e:
        latency = time.time() - start_time
        _log(f"   ❌ Vision review failed for {diagram_id}: {e} ({latency:.2f}s)")
        return "", latency


mcp = FastMCP("mentorml-diagrams")


@mcp.tool()
def retrieve_diagram(query: str, with_vision: bool = True) -> dict:
    """
    Search for a technical diagram that best matches the query.

    Use this tool when you want to find a visual explanation of an AI/ML concept.
    The tool searches through a corpus of diagrams from Jay Alammar's
    Illustrated ML posts (Transformer, BERT, GPT-2, Word2vec, Stable Diffusion).

    Args:
        query: A description of the diagram you're looking for, e.g.,
               "attention mechanism visualization" or "encoder-decoder architecture"
        with_vision: If True (default), performs a vision review of the diagram to get a
                    contextual description of what it shows. This adds ~5 seconds
                    but provides richer detail about the visual elements. Set to False
                    for faster retrieval without visual analysis.

    Returns:
        A dictionary with the top matching diagram:
        - id: Diagram identifier (e.g., "diagram_042")
        - score: Similarity score (0-1, higher is better)
        - description: AI-generated description of what the diagram shows
        - vision_description: Contextual description from visual analysis
        - post_url: URL of the source blog post
    """
    text_emb = _scorer.get_text_embedding(query)

    similarities = np.dot(_image_embeddings, text_emb.T).squeeze()
    scores = 1 / (1 + np.exp(-(_temperature * similarities + _bias)))

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_item = _corpus[best_idx]
    diagram_id = best_item["id"]

    if with_vision and _ENABLE_VISION:
        vision_description, vision_latency = _vision_review_diagram(diagram_id, query)
    else:
        vision_description, vision_latency = "", 0.0

    return {
        "id": diagram_id,
        "score": best_score,
        "query": query,
        "description": best_item.get("query", ""),
        "vision_description": vision_description,
        "vision_latency_s": vision_latency,
        "post_url": best_item.get("post_url", ""),
        "post_title": best_item.get("post_title", ""),
    }


if __name__ == "__main__":
    _log("🚀 Starting MentorML diagram MCP server (stdio)...")
    mcp.run(transport="stdio")
