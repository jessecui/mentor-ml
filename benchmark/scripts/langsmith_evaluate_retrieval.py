"""
LangSmith-backed retrieval eval: SigLIP 2 vs CLIP top-1 accuracy.

Runs each model as a target against the LangSmith dataset uploaded by
upload_to_langsmith.py. Equivalent to the original benchmark/scripts/evaluate.py,
but with traces, versioned datasets, and a side-by-side comparison UI.

Usage:
    export LANGSMITH_API_KEY=...
    python benchmark/scripts/langsmith_evaluate_retrieval.py
    python benchmark/scripts/langsmith_evaluate_retrieval.py --skip-clip
    python benchmark/scripts/langsmith_evaluate_retrieval.py --limit 10
    python benchmark/scripts/langsmith_evaluate_retrieval.py --dataset-name custom-name
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORPUS_DIR = PROJECT_ROOT / "benchmark" / "corpus"
IMAGES_DIR = CORPUS_DIR / "images"
EMBEDDINGS_FILE = CORPUS_DIR / "embeddings" / "siglip_embeddings.npz"

DEFAULT_DATASET_NAME = "mentorml-diagram-retrieval"


# =============================================================================
# CORPUS HELPERS (shared by both targets)
# =============================================================================

def load_corpus() -> tuple[list[dict], list[Path], list[str]]:
    """Load corpus metadata and resolve all image paths.

    Returns: (corpus, image_paths, image_ids) — index-aligned.
    """
    corpus_file = CORPUS_DIR / "metadata" / "corpus_with_queries.json"
    if not corpus_file.exists():
        corpus_file = CORPUS_DIR / "metadata" / "corpus.json"
    corpus = json.loads(corpus_file.read_text())

    image_paths: list[Path] = []
    image_ids: list[str] = []
    for item in corpus:
        path = IMAGES_DIR / item.get("source", "") / item.get("filename", "")
        if path.exists():
            image_paths.append(path)
            image_ids.append(item["id"])
    return corpus, image_paths, image_ids


# =============================================================================
# SIGLIP TARGET
# =============================================================================

def make_siglip_target():
    """Build a target function that returns top-1 diagram_id for SigLIP."""
    print("Loading SigLIP 2 scorer...")
    from model.scorer import SigLIPScorer
    scorer = SigLIPScorer()

    print(f"Loading precomputed SigLIP embeddings from {EMBEDDINGS_FILE.name}...")
    data = np.load(EMBEDDINGS_FILE)
    image_embeddings: np.ndarray = data["embeddings"]
    temperature = float(data["temperature"])
    bias = float(data["bias"])

    _, _, image_ids = load_corpus()
    if len(image_ids) != image_embeddings.shape[0]:
        raise RuntimeError(
            f"Corpus size ({len(image_ids)}) != precomputed embeddings ({image_embeddings.shape[0]}). "
            "Re-run scripts/precompute_embeddings.py."
        )

    print(f"  ✅ SigLIP ready ({image_embeddings.shape[0]} embeddings)")

    def target(inputs: dict) -> dict:
        text_emb = scorer.get_text_embedding(inputs["query"])
        similarities = np.dot(image_embeddings, text_emb.T).squeeze()
        scores = 1 / (1 + np.exp(-(temperature * similarities + bias)))
        best_idx = int(np.argmax(scores))
        return {
            "diagram_id": image_ids[best_idx],
            "score": float(scores[best_idx]),
        }

    return target


# =============================================================================
# CLIP TARGET
# =============================================================================

CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"


def make_clip_target():
    """Build a target function that returns top-1 diagram_id for CLIP baseline."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    print(f"Loading CLIP {CLIP_MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: CLIPModel = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)  # type: ignore[assignment]
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()

    _, image_paths, image_ids = load_corpus()

    print(f"Computing CLIP image embeddings for {len(image_paths)} images...")
    image_embs: list[np.ndarray] = []
    for i, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)  # type: ignore[call-arg]
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        image_embs.append(emb.cpu().numpy().flatten())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(image_paths)} embedded")
    image_embeddings = np.stack(image_embs)
    print(f"  ✅ CLIP ready ({image_embeddings.shape[0]} embeddings)")

    def target(inputs: dict) -> dict:
        proc = processor(text=[inputs["query"]], return_tensors="pt", padding=True, truncation=True).to(device)  # type: ignore[call-arg]
        with torch.no_grad():
            txt_emb = model.get_text_features(**proc)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        txt_emb_np = txt_emb.cpu().numpy().flatten()
        scores = image_embeddings @ txt_emb_np  # cosine similarity (normalized)
        best_idx = int(np.argmax(scores))
        return {
            "diagram_id": image_ids[best_idx],
            "score": float(scores[best_idx]),
        }

    return target


# =============================================================================
# EVALUATOR
# =============================================================================

def top_1_accuracy(run, example) -> dict:
    """1 if predicted diagram_id matches reference, else 0."""
    predicted = (run.outputs or {}).get("diagram_id")
    expected = (example.outputs or {}).get("relevant_image_id")
    return {
        "key": "top_1_accuracy",
        "score": 1 if predicted == expected else 0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--skip-siglip", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Only evaluate first N examples (useful for smoke tests).")
    args = parser.parse_args()

    from langsmith import Client, evaluate

    client = Client()

    # Resolve dataset (and optionally subset to first N examples)
    if args.limit is not None:
        all_examples = list(client.list_examples(dataset_name=args.dataset_name))
        data = all_examples[:args.limit]
        print(f"Running against {len(data)}/{len(all_examples)} examples (--limit {args.limit})")
    else:
        data = args.dataset_name
        existing = list(client.list_datasets(dataset_name=args.dataset_name))
        if not existing:
            print(f"❌ Dataset '{args.dataset_name}' not found. Run upload_to_langsmith.py first.")
            sys.exit(1)
        print(f"Running against full dataset '{args.dataset_name}'")

    print()

    if not args.skip_siglip:
        print("=" * 60)
        print("Evaluating SigLIP 2")
        print("=" * 60)
        siglip_target = make_siglip_target()
        siglip_results = evaluate(
            siglip_target,
            data=data,
            evaluators=[top_1_accuracy],
            experiment_prefix="siglip-v2",
        )
        print(f"\n✅ SigLIP eval complete: {siglip_results.experiment_name}")
        print()

    if not args.skip_clip:
        print("=" * 60)
        print("Evaluating CLIP baseline")
        print("=" * 60)
        clip_target = make_clip_target()
        clip_results = evaluate(
            clip_target,
            data=data,
            evaluators=[top_1_accuracy],
            experiment_prefix="clip-vit-l14",
        )
        print(f"\n✅ CLIP eval complete: {clip_results.experiment_name}")
        print()

    print("View results & side-by-side comparison in the LangSmith UI under your dataset.")


if __name__ == "__main__":
    main()
