"""
LangChain tools for the MentorML agent.

Provides diagram retrieval using SigLIP embeddings.
Image embeddings are loaded from pre-computed file for instant startup.

To regenerate embeddings after adding diagrams:
    python scripts/precompute_embeddings.py
"""

import base64
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

if TYPE_CHECKING:
    from model.scorer import SigLIPScorer

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_METADATA = PROJECT_ROOT / "benchmark" / "corpus" / "metadata" / "corpus_with_queries.json"
DIAGRAMS_DIR = PROJECT_ROOT / "benchmark" / "corpus" / "images" / "diagrams"
EMBEDDINGS_FILE = PROJECT_ROOT / "benchmark" / "corpus" / "embeddings" / "siglip_embeddings.npz"

# Module-level state (set by create_retrieval_tool)
_scorer: "SigLIPScorer | None" = None
_corpus: list[dict] | None = None
_image_embeddings: np.ndarray | None = None  # Pre-computed [N, embed_dim]
_temperature: float = 10.0
_bias: float = -10.0
_enable_vision: bool = True  # Whether to run vision review (adds latency)

# Vision review prompt
VISION_REVIEW_PROMPT = """You searched for: "{query}"

Look at this diagram carefully. In 1-2 sentences, describe what this specific diagram shows 
and how it helps explain the concept. Be specific about what visual elements are present 
(arrows, boxes, flow direction, labels, colors, etc.).

Start with "This diagram shows..." and focus on what's visually depicted."""


def _load_corpus() -> list[dict]:
    """Load diagram corpus metadata."""
    global _corpus
    if _corpus is None:
        with open(CORPUS_METADATA) as f:
            _corpus = json.load(f)
    assert _corpus is not None
    return _corpus


def _load_embeddings() -> tuple[np.ndarray, float, float]:
    """Load pre-computed embeddings from disk."""
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Pre-computed embeddings not found: {EMBEDDINGS_FILE}\n"
            "Run: python scripts/precompute_embeddings.py"
        )
    
    data = np.load(EMBEDDINGS_FILE)
    return data["embeddings"], float(data["temperature"]), float(data["bias"])


def _vision_review_diagram(diagram_id: str, query: str) -> tuple[str, float]:
    """
    Use Gemini vision to describe how a diagram relates to the query.
    
    Args:
        diagram_id: The diagram identifier (e.g., "diagram_042")
        query: The search query that was used to find this diagram
        
    Returns:
        Tuple of (contextual description, latency in seconds).
        Returns ("", 0.0) on failure.
    """
    image_path = DIAGRAMS_DIR / f"{diagram_id}.png"
    if not image_path.exists():
        return "", 0.0
    
    start_time = time.time()
    try:
        # Load and encode image
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Build multimodal message
        prompt = VISION_REVIEW_PROMPT.format(query=query)
        vision_message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ])
        
        # Quick vision call
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        response = llm.invoke([vision_message])
        
        latency = time.time() - start_time
        description = response.content if isinstance(response.content, str) else str(response.content)
        print(f"   👁️ Vision review for {diagram_id}: {latency:.2f}s")
        return description, latency
    except Exception as e:
        latency = time.time() - start_time
        print(f"   ❌ Vision review failed for {diagram_id}: {e} ({latency:.2f}s)")
        return "", latency


def create_retrieval_tool(scorer: "SigLIPScorer", enable_vision: bool = True):
    """
    Create a retrieval tool bound to a SigLIP scorer instance.
    
    Loads pre-computed image embeddings from disk for instant startup.
    
    Args:
        scorer: Pre-initialized SigLIPScorer instance (used for text encoding)
        enable_vision: Whether to run vision review on retrieved diagrams.
                      Adds ~5-10s latency but provides contextual descriptions.
        
    Returns:
        A LangChain tool function for diagram retrieval
    """
    global _scorer, _image_embeddings, _temperature, _bias, _enable_vision
    _scorer = scorer
    _enable_vision = enable_vision
    
    # Load pre-computed embeddings
    print(f"📊 Loading pre-computed embeddings from {EMBEDDINGS_FILE}...")
    _image_embeddings, _temperature, _bias = _load_embeddings()
    print(f"   ✅ Loaded {_image_embeddings.shape[0]} embeddings (dim={_image_embeddings.shape[1]})")
    print(f"   👁️ Vision review: {'enabled' if _enable_vision else 'disabled (global override)'}")
    
    @tool
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
        if _scorer is None or _image_embeddings is None:
            return {"error": "Scorer not initialized"}
        
        # Get text embedding [1, embed_dim]
        text_emb = _scorer.get_text_embedding(query)
        
        # Compute similarities via dot product (embeddings are normalized)
        # SigLIP uses sigmoid(dot * temp + bias) for final score
        similarities = np.dot(_image_embeddings, text_emb.T).squeeze()  # [N]
        
        # Apply SigLIP's learned temperature and bias for proper scoring
        scores = 1 / (1 + np.exp(-(_temperature * similarities + _bias)))
        
        # Find top-1 match
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        
        # Get metadata
        corpus = _load_corpus()
        best_item = corpus[best_idx]
        diagram_id = best_item["id"]
        
        # Vision review: agent "sees" the diagram and describes it contextually
        # Only runs if agent requests it AND global flag allows it
        if with_vision and _enable_vision:
            vision_description, vision_latency = _vision_review_diagram(diagram_id, query)
        else:
            vision_description, vision_latency = "", 0.0
        
        return {
            "id": diagram_id,
            "score": best_score,
            "query": query,  # The query used to retrieve this diagram
            "description": best_item.get("query", ""),  # Pre-generated description
            "vision_description": vision_description,  # Live contextual description
            "vision_latency_s": vision_latency,  # How long vision review took
            "post_url": best_item.get("post_url", ""),
            "post_title": best_item.get("post_title", ""),
        }
    
    return retrieve_diagram
