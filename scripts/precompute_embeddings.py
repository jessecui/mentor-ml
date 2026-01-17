#!/usr/bin/env python3
"""
Pre-compute SigLIP embeddings for all diagrams in the corpus.

Run this once after adding new diagrams:
    python scripts/precompute_embeddings.py

Saves embeddings to: benchmark/corpus/embeddings/siglip_embeddings.npz
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.scorer import SigLIPScorer

# Paths
CORPUS_METADATA = PROJECT_ROOT / "benchmark" / "corpus" / "metadata" / "corpus_with_queries.json"
DIAGRAMS_DIR = PROJECT_ROOT / "benchmark" / "corpus" / "images" / "diagrams"
EMBEDDINGS_DIR = PROJECT_ROOT / "benchmark" / "corpus" / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "siglip_embeddings.npz"


def main():
    print("🚀 Pre-computing SigLIP embeddings for diagram corpus...")
    
    # Load corpus metadata
    with open(CORPUS_METADATA) as f:
        corpus = json.load(f)
    
    print(f"📊 Found {len(corpus)} diagrams in corpus")
    
    # Initialize scorer
    print("📦 Loading SigLIP scorer...")
    scorer = SigLIPScorer()
    
    # Compute embeddings
    embeddings = []
    filenames = []
    
    print(f"🔄 Computing embeddings...")
    for i, item in enumerate(corpus):
        image_path = DIAGRAMS_DIR / item["filename"]
        emb = scorer.get_image_embedding(str(image_path), use_cache=False)
        embeddings.append(emb.squeeze())  # [embed_dim]
        filenames.append(item["filename"])
        
        if (i + 1) % 20 == 0 or (i + 1) == len(corpus):
            print(f"   {i + 1}/{len(corpus)} images embedded")
    
    # Stack into array
    embeddings_array = np.vstack(embeddings)  # [N, embed_dim]
    
    # Save to disk
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        EMBEDDINGS_FILE,
        embeddings=embeddings_array,
        filenames=filenames,
        temperature=scorer.temperature,
        bias=scorer.bias,
    )
    
    print(f"\n✅ Saved {len(corpus)} embeddings to {EMBEDDINGS_FILE}")
    print(f"   Shape: {embeddings_array.shape}")
    print(f"   Temperature: {scorer.temperature}")
    print(f"   Bias: {scorer.bias}")


if __name__ == "__main__":
    main()
