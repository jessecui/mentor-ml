"""
Stage 3: Generate benchmark queries for classified diagrams.

Uses Gemini Vision to generate natural language queries that each diagram
could answer, creating ground-truth query-image pairs for benchmarking.

Usage:
    python benchmark/scripts/generate_queries.py
    
    # Process specific number of images
    python benchmark/scripts/generate_queries.py --limit 10
    
    # Resume from where you left off
    python benchmark/scripts/generate_queries.py --resume
    
    # Preview prompts without calling API
    python benchmark/scripts/generate_queries.py --dry-run

Input:  benchmark/corpus/metadata/corpus_classified.json
Output: benchmark/corpus/metadata/corpus_with_queries.json
        benchmark/queries/benchmark_queries.json
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Load .env file from project root
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CORPUS_DIR = PROJECT_ROOT / "benchmark" / "corpus"
IMAGES_DIR = CORPUS_DIR / "images"
METADATA_DIR = CORPUS_DIR / "metadata"
QUERIES_DIR = PROJECT_ROOT / "benchmark" / "queries"

# Use corpus.json directly (no classification step needed)
INPUT_FILE = METADATA_DIR / "corpus.json"
OUTPUT_FILE = METADATA_DIR / "corpus_with_queries.json"
QUERIES_FILE = QUERIES_DIR / "benchmark_queries.json"

# Model selection
# gemini-3-pro-preview for best vision capabilities
DEFAULT_MODEL = "gemini-3-pro-preview"

# Rate limiting (set to 0 to disable)
REQUEST_DELAY = 0

# Query generation prompt - Leverage text reading and diagram understanding
QUERY_GENERATION_PROMPT = """Look at this ML/AI diagram and generate a search query that requires READING and UNDERSTANDING the diagram content.

Write a query (10-25 words) that includes:
1. SPECIFIC TEXT LABELS or SYMBOLS visible in the diagram (e.g., "Q, K, V", "softmax", "Nx", "encoder")
2. RELATIONSHIPS or PROCESSES shown (e.g., "input flows to", "connects to", "produces output")
3. TECHNICAL CONCEPTS the diagram explains

Imagine there are 5 other similar diagrams. Include the ONE detail (text or connection) that makes this specific diagram unique.

The query should be something you could ONLY answer by reading the diagram, not just looking at it.

Examples of GOOD queries:
- "diagram showing Q K V matrices feeding into scaled dot-product attention with softmax"
- "encoder stack with Nx layers containing self-attention and feed-forward sublayers"  
- "positional encoding added to input embeddings before transformer layers"
- "multi-head attention concatenating h parallel attention outputs with linear projection"
- "masked attention in decoder preventing positions from attending to future tokens"

Examples of BAD queries:
- "transformer architecture" (too vague)
- "green and pink boxes with arrows" (visual description, not content-based)
- "Diagram showing a left-to-right flow where..." (describing appearance, not meaning)

Return ONLY the query (10-25 words), nothing else:"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_image_path(item: dict) -> Path | None:
    """Get the full path to an image from its metadata."""
    source = item.get("source", "")
    filename = item.get("filename", "")
    
    if not source or not filename:
        return None
    
    path = IMAGES_DIR / source / filename
    return path if path.exists() else None


def get_mime_type(filepath: Path) -> str:
    """Get MIME type from file extension."""
    ext = filepath.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/png")


def generate_query(client: genai.Client, model_name: str, filepath: Path) -> str | None:
    """
    Generate a query for an image using Gemini Vision.
    
    Returns:
        Query string or None on error
    """
    try:
        # Read image bytes and create Part
        image_bytes = filepath.read_bytes()
        mime_type = get_mime_type(filepath)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        
        # Send to Gemini using new Client API
        response = client.models.generate_content(
            model=model_name,
            contents=[QUERY_GENERATION_PROMPT, image_part]
        )
        
        # Extract text from parts directly to avoid SDK warning about thought_signature
        text_parts = []
        for candidate in response.candidates or []:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
        text = "".join(text_parts).strip()
        
        # Remove any numbering or bullets
        query = text.lstrip("0123456789.-)*• ").strip()
        
        if query and len(query) > 10:
            return query
        return None
        
    except Exception as e:
        print(f"    Error: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark queries for classified diagrams"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without calling API"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of images to process"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Starting index (0-based) for processing"
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated diagram IDs to process (e.g., 'diagram_056,diagram_058')"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip images that already have queries"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stage 3: Query Generation")
    print("=" * 60)
    print()
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        print("❌ GEMINI_API_KEY or GOOGLE_API_KEY not set")
        return
    
    # Load input
    if not INPUT_FILE.exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("   Run scraper first: python benchmark/scripts/scrape_ai_ml_diagrams.py")
        return
    
    with open(INPUT_FILE) as f:
        corpus = json.load(f)
    
    # Use all diagrams from the curated posts (no classification needed)
    diagrams = corpus
    
    print(f"Input: {len(diagrams)} diagrams")
    
    # Load existing if resuming
    existing = {}
    if args.resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing_list = json.load(f)
            existing = {item["id"]: item for item in existing_list}
        print(f"Resuming: {len(existing)} already have queries")
    
    # Filter by --only if specified
    if args.only:
        only_ids = set(id.strip() for id in args.only.split(","))
        diagrams = [d for d in diagrams if d["id"] in only_ids]
        print(f"Filtering to specific diagrams: {only_ids}")
    
    # Filter to unprocessed
    to_process = []
    for item in diagrams:
        if args.resume and item["id"] in existing:
            if existing[item["id"]].get("query"):
                continue
        to_process.append(item)
    
    # Apply start index
    if args.start > 0:
        to_process = to_process[args.start:]
        print(f"Starting from index {args.start}")
    
    if args.limit:
        to_process = to_process[:args.limit]
    
    print(f"To process: {len(to_process)} diagrams")
    print()
    
    if args.dry_run:
        print("Query generation prompt:")
        print("-" * 40)
        print(QUERY_GENERATION_PROMPT)
        print("-" * 40)
        print()
        print("(Dry run - no API calls made)")
        return
    
    # Initialize Gemini client with explicit API key
    client = genai.Client(api_key=api_key)
    print(f"Using model: {args.model}")
    print()
    
    # Ensure output directory exists
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing data (always, for --only to work correctly)
    existing_corpus: dict[str, dict] = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for item in json.load(f):
                existing_corpus[item["id"]] = item
    
    existing_queries: dict[str, dict] = {}
    if QUERIES_FILE.exists():
        with open(QUERIES_FILE) as f:
            for q in json.load(f):
                existing_queries[q["relevant_image_id"]] = q
    
    for i, item in enumerate(to_process):
        filepath = get_image_path(item)
        
        if filepath is None or filepath.suffix.lower() == ".svg":
            print(f"[{i+1}/{len(to_process)}] {item['id']}: SKIPPED")
            continue
        
        # Generate query
        query = generate_query(client, args.model, filepath)
        
        if query:
            # Update or add to corpus
            item["query"] = query
            existing_corpus[item["id"]] = item
            
            # Update or add to queries
            existing_queries[item["id"]] = {
                "query": query,
                "relevant_image_id": item["id"],
                "source_post": item.get("post_title", ""),
            }
            
            print(f"[{i+1}/{len(to_process)}] {item['id']}: {query}")
        else:
            print(f"[{i+1}/{len(to_process)}] {item['id']}: NO QUERY GENERATED")
        
        # Save progress after each item (preserving order by sorting)
        results = sorted(existing_corpus.values(), key=lambda x: x["id"])
        all_queries = sorted(existing_queries.values(), key=lambda x: x["relevant_image_id"])
        
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        with open(QUERIES_FILE, "w") as f:
            json.dump(all_queries, f, indent=2)
        
        # Rate limiting
        if i < len(to_process) - 1:
            time.sleep(REQUEST_DELAY)
    
    # Summary
    print()
    print("=" * 60)
    print("Query Generation Complete")
    print("=" * 60)
    print(f"  📊 Diagrams processed: {len(to_process)}")
    print(f"  ❓ Total queries: {len(existing_queries)}")
    print()
    print(f"Output files:")
    print(f"  - {OUTPUT_FILE.name} (corpus with queries)")
    print(f"  - {QUERIES_FILE.name} (benchmark queries)")


if __name__ == "__main__":
    main()
