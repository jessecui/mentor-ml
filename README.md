# MentorML

A multimodal learning companion for AI/ML concepts, featuring a JAX/Flax bi-encoder using SigLIP 2 for query-image retrieval of technical diagrams scraped from Jay Alammar's [Illustrated ML](https://jalammar.github.io/) posts.

## Quick Start

```bash
# Clone repo with big_vision dependency
git clone https://github.com/jessecui/mentor-ml.git
cd mentor-ml
git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision big_vision_repo

# Create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download Gemma tokenizer (used by SigLIP 2)
curl -L -o model/gemma_tokenizer.model https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

# Test the scorer (auto-downloads SigLIP weights ~1.5GB on first run)
python model/scorer.py
```

## SigLIP Scorer

The scorer uses [SigLIP 2](https://arxiv.org/abs/2502.14786) (So400m/14 @ 384px), Google's state-of-the-art contrastive vision-language model optimized for retrieval.

### Usage

```python
from model.scorer import SigLIPScorer

scorer = SigLIPScorer()

# Score single image-query pair
score = scorer.score("diagram.png", "transformer attention mechanism")

# Batch scoring (efficient - encodes query once)
scores = scorer.score_batch(["img1.png", "img2.png"], "self-attention layer")
```

### Model Details

| Component | Specification |
|-----------|---------------|
| **Model** | SigLIP 2 So400m/14 |
| **Image Size** | 384×384 |
| **Parameters** | ~400M |
| **Checkpoint** | ~1.5GB (auto-downloaded) |
| **Tokenizer** | Gemma (256k vocab) |
| **Framework** | JAX/Flax (big_vision) |

The checkpoint downloads automatically from Google Cloud Storage on first run:
```
https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m14_384.npz
```

## Benchmark

Evaluate SigLIP vs CLIP on technical ML diagram retrieval using 92 diagrams from Jay Alammar's Illustrated series.

### Results

| Model | Top-1 Accuracy |
|-------|----------------|
| **SigLIP 2** | **76.1%** (70/92) |
| CLIP ViT-L/14 | 46.7% (43/92) |

**+29.3 percentage points improvement** (+62.8% relative)

### Run Benchmark

```bash
# 1. Scrape diagrams from Jay Alammar's blog
python benchmark/scripts/scrape_ai_ml_diagrams.py

# 2. Generate queries (requires GEMINI_API_KEY in .env)
python benchmark/scripts/generate_queries.py

# 3. Evaluate SigLIP vs CLIP
python benchmark/scripts/evaluate.py
```

### Diagram Sources

The benchmark uses diagrams from the top 5 Illustrated posts:

1. **The Illustrated Transformer**
2. **The Illustrated BERT**
3. **The Illustrated GPT-2**
4. **The Illustrated Word2vec**
5. **The Illustrated Stable Diffusion**

### Directory Structure

```
benchmark/
├── corpus/
│   ├── images/diagrams/          # Downloaded ML diagrams
│   └── metadata/
│       ├── corpus.json           # Scraped metadata
│       └── corpus_with_queries.json
├── queries/
│   └── benchmark_queries.json    # Query-image ground truth
├── results/
│   ├── siglip_evaluation_results.json
│   └── siglip_evaluation_summary.txt
└── scripts/
    ├── scrape_ai_ml_diagrams.py
    ├── generate_queries.py
    └── evaluate.py

model/
├── scorer.py                     # SigLIP scorer
├── gemma_tokenizer.model         # Tokenizer (~4MB)
└── siglip2_so400m14_384.npz      # Checkpoint (~1.5GB, gitignored)
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required for query generation
GEMINI_API_KEY=your_gemini_api_key
```

## Requirements

- Python 3.10+
- macOS or Linux (CPU or GPU)
- ~4GB disk space (checkpoint + images)