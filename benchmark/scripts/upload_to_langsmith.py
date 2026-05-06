"""
One-time uploader: benchmark/queries/benchmark_queries.json -> LangSmith Dataset.

Each query becomes one example with:
    inputs  = {"query": "<query string>"}
    outputs = {"relevant_image_id": "diagram_XXX"}

Usage:
    export LANGSMITH_API_KEY=...
    python benchmark/scripts/upload_to_langsmith.py
    python benchmark/scripts/upload_to_langsmith.py --dataset-name custom-name
    python benchmark/scripts/upload_to_langsmith.py --dry-run
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
QUERIES_FILE = PROJECT_ROOT / "benchmark" / "queries" / "benchmark_queries.json"

DEFAULT_DATASET_NAME = "mentorml-diagram-retrieval"
DATASET_DESCRIPTION = (
    "Diagram retrieval benchmark for MentorML. Each example pairs a natural-language "
    "query with the diagram_id of the relevant diagram in the corpus. Used to evaluate "
    "embedding-based retrieval models (SigLIP 2 vs CLIP)."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded; do not call LangSmith.")
    args = parser.parse_args()

    queries = json.loads(QUERIES_FILE.read_text())
    print(f"Loaded {len(queries)} queries from {QUERIES_FILE.relative_to(PROJECT_ROOT)}")

    inputs = [{"query": q["query"]} for q in queries]
    outputs = [{"relevant_image_id": q["relevant_image_id"]} for q in queries]

    if args.dry_run:
        print("\nDRY RUN — first 3 examples that would be uploaded:")
        for i in range(min(3, len(queries))):
            print(f"  inputs:  {inputs[i]}")
            print(f"  outputs: {outputs[i]}")
            print()
        print(f"Dataset name: {args.dataset_name}")
        return

    client = Client()

    existing = list(client.list_datasets(dataset_name=args.dataset_name))
    if existing:
        dataset = existing[0]
        print(f"⚠️  Dataset '{args.dataset_name}' already exists (id={dataset.id})")
        existing_count = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
        print(f"   It has {existing_count} examples; skipping upload to avoid duplicates.")
        print(f"   Delete it in the LangSmith UI first if you want a fresh upload.")
        return

    print(f"Creating dataset '{args.dataset_name}'...")
    dataset = client.create_dataset(
        dataset_name=args.dataset_name,
        description=DATASET_DESCRIPTION,
    )

    print(f"Uploading {len(queries)} examples...")
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id,
    )

    print(f"✅ Uploaded {len(queries)} examples to dataset '{args.dataset_name}' (id={dataset.id})")


if __name__ == "__main__":
    main()
