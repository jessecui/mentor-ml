"""
Benchmark evaluation: SigLIP 2 vs CLIP on diagram retrieval.

Computes Top-1 Accuracy for both methods:
- SigLIP 2 (JAX/Flax): State-of-the-art contrastive model using sigmoid scoring
- CLIP (ViT-L/14): Industry baseline using cosine similarity

Usage:
    python benchmark/scripts/evaluate_siglip.py
    
    # Skip CLIP baseline
    python benchmark/scripts/evaluate_siglip.py --skip-clip
    
    # Limit to first N queries (for testing)
    python benchmark/scripts/evaluate_siglip.py --limit 10

Input:  benchmark/queries/benchmark_queries.json
Output: benchmark/results/siglip_evaluation_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# =============================================================================
# PATH SETUP
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CORPUS_DIR = PROJECT_ROOT / "benchmark" / "corpus"
IMAGES_DIR = CORPUS_DIR / "images"
QUERIES_FILE = PROJECT_ROOT / "benchmark" / "queries" / "benchmark_queries.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"

# Add project root to path for model imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# SIGLIP SCORER
# =============================================================================

def load_siglip_scorer():
    """Load the SigLIP 2 scorer."""
    print("Loading SigLIP 2 So400m/14@384px scorer...")
    from model.scorer import SigLIPScorer
    scorer = SigLIPScorer()
    print(f"  ✅ SigLIP 2 loaded (temp={scorer.temperature:.1f}, bias={scorer.bias:.1f})")
    return scorer


# =============================================================================
# CLIP BASELINE
# =============================================================================

CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"

def load_clip_model():
    """Load CLIP ViT-L/14@336px model from HuggingFace."""
    print(f"Loading CLIP ViT-L/14@336px...")
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print("  ❌ transformers not installed. Run: pip install transformers torch")
        return None, None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: CLIPModel = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)  # type: ignore[assignment]
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    print(f"  ✅ CLIP ViT-L/14@336px loaded on {device}")
    return model, processor, device


def compute_clip_image_embeddings(model, processor, device, image_paths: list[Path]) -> dict[str, np.ndarray]:
    """Pre-compute CLIP embeddings for all images."""
    import torch
    
    embeddings = {}
    print(f"Computing CLIP image embeddings for {len(image_paths)} images...")
    
    for i, path in enumerate(image_paths):
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings[path.name] = emb.cpu().numpy().flatten()
        except Exception as e:
            print(f"  ⚠️  Failed to embed {path.name}: {e}")
        
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(image_paths)} images embedded")
    
    print(f"  ✅ {len(embeddings)} image embeddings computed")
    return embeddings


def compute_clip_text_embedding(model, processor, device, query: str) -> np.ndarray:
    """Compute CLIP embedding for a text query."""
    import torch
    
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


def score_clip(image_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
    """Compute cosine similarity between CLIP embeddings."""
    return float(np.dot(image_embedding, text_embedding))


# =============================================================================
# EVALUATION
# =============================================================================

def get_image_path(image_id: str, corpus: list[dict]) -> Path | None:
    """Get the file path for an image ID."""
    for item in corpus:
        if item["id"] == image_id:
            source = item.get("source", "")
            filename = item.get("filename", "")
            if source and filename:
                return IMAGES_DIR / source / filename
    return None


def evaluate(
    queries: list[dict],
    corpus: list[dict],
    siglip_scorer,
    clip_model=None,
    clip_processor=None,
    clip_device=None,
    clip_embeddings: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Run evaluation on all queries.
    
    Returns:
        Dict with results for SigLIP and CLIP
    """
    # Get all image paths
    image_paths = []
    image_ids = []
    for item in corpus:
        path = get_image_path(item["id"], corpus)
        if path and path.exists():
            image_paths.append(path)
            image_ids.append(item["id"])
    
    print(f"\nEvaluating on {len(queries)} queries across {len(image_paths)} images")
    print("=" * 60)
    
    # Pre-compute CLIP embeddings if needed
    if clip_model is not None and clip_embeddings is None:
        clip_embeddings = compute_clip_image_embeddings(
            clip_model, clip_processor, clip_device, image_paths
        )
    
    # Pre-compute SigLIP embeddings for all images
    siglip_embeddings = {}
    if siglip_scorer is not None:
        print(f"\nComputing SigLIP image embeddings for {len(image_paths)} images...")
        for i, (img_id, img_path) in enumerate(zip(image_ids, image_paths)):
            siglip_embeddings[img_id] = siglip_scorer.get_image_embedding(str(img_path))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(image_paths)} images embedded")
        print(f"  ✅ {len(siglip_embeddings)} SigLIP embeddings computed")
    
    siglip_correct = 0
    clip_correct = 0
    results_detail = []
    
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    
    for i, q in enumerate(queries):
        query_text = q["query"]
        target_id = q["relevant_image_id"]
        
        print(f"\n[Query {i+1}/{len(queries)}] {query_text[:60]}{'...' if len(query_text) > 60 else ''}")
        
        # --- SigLIP scoring ---
        siglip_hit = None
        siglip_rank = None
        if siglip_scorer is not None:
            txt_emb = siglip_scorer.get_text_embedding(query_text)
            
            siglip_scores = {}
            for img_id in image_ids:
                if img_id in siglip_embeddings:
                    img_emb = siglip_embeddings[img_id]
                    score = siglip_scorer.score_fn(img_emb, txt_emb, siglip_scorer.temperature, siglip_scorer.bias)
                    siglip_scores[img_id] = float(score[0])
            
            siglip_ranking = sorted(siglip_scores.items(), key=lambda x: -x[1])
            siglip_top1 = siglip_ranking[0][0]
            siglip_hit = (siglip_top1 == target_id)
            if siglip_hit:
                siglip_correct += 1
            
            siglip_rank = next(
                (j + 1 for j, (img_id, _) in enumerate(siglip_ranking) if img_id == target_id),
                -1
            )
            
            status = "✅" if siglip_hit else "❌"
            print(f"  → SigLIP:  {status} Target rank: {siglip_rank}")
        
        # --- CLIP scoring ---
        clip_hit = None
        clip_rank = None
        if clip_model is not None and clip_embeddings:
            text_emb = compute_clip_text_embedding(clip_model, clip_processor, clip_device, query_text)
            
            clip_scores = {}
            for img_id, img_path in zip(image_ids, image_paths):
                if img_path.name in clip_embeddings:
                    clip_scores[img_id] = score_clip(clip_embeddings[img_path.name], text_emb)
            
            clip_ranking = sorted(clip_scores.items(), key=lambda x: -x[1])
            clip_top1 = clip_ranking[0][0]
            clip_hit = (clip_top1 == target_id)
            if clip_hit:
                clip_correct += 1
            
            clip_rank = next(
                (j + 1 for j, (img_id, _) in enumerate(clip_ranking) if img_id == target_id),
                -1
            )
            
            status = "✅" if clip_hit else "❌"
            print(f"  → CLIP:    {status} Target rank: {clip_rank}")
        
        results_detail.append({
            "query": query_text,
            "target_id": target_id,
            "siglip_rank": siglip_rank,
            "siglip_hit": siglip_hit,
            "clip_rank": clip_rank,
            "clip_hit": clip_hit,
        })
    
    # Compute final metrics
    n_queries = len(queries)
    siglip_acc = siglip_correct / n_queries if n_queries > 0 else 0
    clip_acc = clip_correct / n_queries if n_queries > 0 else 0
    
    return {
        "n_queries": n_queries,
        "n_images": len(image_paths),
        "siglip": {
            "correct": siglip_correct,
            "accuracy": siglip_acc,
        },
        "clip": {
            "correct": clip_correct,
            "accuracy": clip_acc,
        },
        "improvement": (siglip_acc - clip_acc) / clip_acc * 100 if clip_acc > 0 else None,
        "details": results_detail,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate SigLIP 2 vs CLIP on diagram retrieval")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP baseline")
    parser.add_argument("--skip-siglip", action="store_true", help="Skip SigLIP (CLIP only)")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit to first N queries")
    parser.add_argument("--queries", "-q", type=str, default=None, help="Comma-separated query indices")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Benchmark Evaluation: SigLIP 2 vs CLIP")
    print("=" * 60)
    print()
    
    # Load queries
    if not QUERIES_FILE.exists():
        print(f"❌ Queries not found: {QUERIES_FILE}")
        print("   Run: python benchmark/scripts/generate_queries.py")
        return
    
    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    
    # Load corpus
    corpus_file = CORPUS_DIR / "metadata" / "corpus_with_queries.json"
    if not corpus_file.exists():
        corpus_file = CORPUS_DIR / "metadata" / "corpus.json"
    if not corpus_file.exists():
        print(f"❌ Corpus not found: {corpus_file}")
        return
    
    with open(corpus_file) as f:
        corpus = json.load(f)
    
    # Filter queries
    if args.queries:
        query_indices = [int(i.strip()) for i in args.queries.split(",")]
        queries = [queries[i] for i in query_indices if i < len(queries)]
        print(f"Loaded {len(queries)} queries (indices: {query_indices})")
    elif args.limit:
        queries = queries[:args.limit]
        corpus = corpus[:args.limit]
        print(f"Loaded {len(queries)} queries (limited to {args.limit})")
    else:
        print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(corpus)} diagrams")
    
    # Load models
    siglip_scorer = None
    if not args.skip_siglip:
        siglip_scorer = load_siglip_scorer()
    
    clip_model, clip_processor, clip_device = None, None, None
    if not args.skip_clip:
        clip_model, clip_processor, clip_device = load_clip_model()
    
    # Run evaluation
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    results = evaluate(
        queries=queries,
        corpus=corpus,
        siglip_scorer=siglip_scorer,
        clip_model=clip_model,
        clip_processor=clip_processor,
        clip_device=clip_device,
    )
    elapsed = time.time() - start_time
    
    # Print summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Queries: {results['n_queries']}")
    print(f"Images:  {results['n_images']}")
    print()
    print(f"SigLIP 2 Top-1 Accuracy:  {results['siglip']['accuracy']:.1%} ({results['siglip']['correct']}/{results['n_queries']})")
    if results['clip']['accuracy'] > 0:
        print(f"CLIP Top-1 Accuracy:      {results['clip']['accuracy']:.1%} ({results['clip']['correct']}/{results['n_queries']})")
        print()
        abs_diff = (results['siglip']['accuracy'] - results['clip']['accuracy']) * 100
        if results['improvement'] is not None and results['improvement'] > 0:
            print(f"🚀 Improvement: +{abs_diff:.1f} percentage points (+{results['improvement']:.1f}% relative)")
        elif results['improvement'] is not None:
            print(f"📉 Difference: {results['improvement']:.1f}%")
    print()
    print(f"Time: {elapsed:.1f}s")
    
    # Save results
    output_file = RESULTS_DIR / "siglip_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary text file
    summary_file = RESULTS_DIR / "siglip_evaluation_summary.txt"
    abs_diff = (results['siglip']['accuracy'] - results['clip']['accuracy']) * 100
    summary_lines = [
        "SigLIP 2 vs CLIP Benchmark Results",
        "=" * 40,
        "",
        f"Queries: {results['n_queries']}",
        f"Images:  {results['n_images']}",
        "",
        f"SigLIP 2 Top-1 Accuracy:  {results['siglip']['accuracy']:.1%} ({results['siglip']['correct']}/{results['n_queries']})",
        f"CLIP Top-1 Accuracy:      {results['clip']['accuracy']:.1%} ({results['clip']['correct']}/{results['n_queries']})",
        "",
        f"Improvement: +{abs_diff:.1f} percentage points (+{results['improvement']:.1f}% relative)",
        "",
        "Models:",
        "  - SigLIP 2 So400m/14 @ 384px (JAX/Flax)",
        "  - CLIP ViT-L/14 @ 336px (HuggingFace)",
    ]
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))
    
    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
