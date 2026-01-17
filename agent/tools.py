"""
LangChain tools for the MentorML agent.

Provides diagram retrieval using SigLIP embeddings.
Image embeddings are loaded from pre-computed file for instant startup.

To regenerate embeddings after adding diagrams:
    python scripts/precompute_embeddings.py
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.tools import tool

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


def create_retrieval_tool(scorer: "SigLIPScorer"):
    """
    Create a retrieval tool bound to a SigLIP scorer instance.
    
    Loads pre-computed image embeddings from disk for instant startup.
    
    Args:
        scorer: Pre-initialized SigLIPScorer instance (used for text encoding)
        
    Returns:
        A LangChain tool function for diagram retrieval
    """
    global _scorer, _image_embeddings, _temperature, _bias
    _scorer = scorer
    
    # Load pre-computed embeddings
    print(f"📊 Loading pre-computed embeddings from {EMBEDDINGS_FILE}...")
    _image_embeddings, _temperature, _bias = _load_embeddings()
    print(f"   ✅ Loaded {_image_embeddings.shape[0]} embeddings (dim={_image_embeddings.shape[1]})")
    
    @tool
    def retrieve_diagram(query: str) -> dict:
        """
        Search for a technical diagram that best matches the query.
        
        Use this tool when you want to find a visual explanation of an AI/ML concept.
        The tool searches through a corpus of diagrams from Jay Alammar's 
        Illustrated ML posts (Transformer, BERT, GPT-2, Word2vec, Stable Diffusion).
        
        Args:
            query: A description of the diagram you're looking for, e.g.,
                   "attention mechanism visualization" or "encoder-decoder architecture"
        
        Returns:
            A dictionary with the top matching diagram:
            - id: Diagram identifier (e.g., "diagram_042")
            - score: Similarity score (0-1, higher is better)
            - context: Original context from the blog post
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
        
        return {
            "id": best_item["id"],
            "score": best_score,
            "context": best_item.get("context", ""),
            "post_url": best_item.get("post_url", ""),
            "post_title": best_item.get("post_title", ""),
        }
    
    return retrieve_diagram
