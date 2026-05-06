# MentorML

A multimodal AI/ML teaching assistant that retrieves and explains technical diagrams using a Plan-and-Execute agent architecture. Features SigLIP 2 for semantic image retrieval and Gemini 3 Flash for vision-augmented explanations.

## Features

- 🤖 **LangGraph Agent**: Plan-and-Execute architecture with CoT planning and ReAct execution
- 🖼️ **Multimodal Vision**: Agent sees retrieved diagrams and provides contextual descriptions
- 🔍 **SigLIP 2 Retrieval**: State-of-the-art bi-encoder (76% top-1 accuracy on ML diagrams)
- 🔌 **MCP Tool Server**: Diagram retrieval runs as a separate MCP subprocess (stdio); the agent calls it like any LangChain tool
- 📊 **LangSmith Eval**: Traced retrieval (SigLIP vs CLIP) and end-to-end agent eval against a versioned dataset
- 💬 **Conversation Memory**: Redis-backed checkpointing with 24h TTL
- 🎨 **92 Diagrams**: Curated from Jay Alammar's [Illustrated ML](https://jalammar.github.io/) posts
- ⚡ **SSE Streaming**: Real-time token streaming with thinking/planning visibility
- 🖥️ **React Frontend**: Chat UI with inline diagrams and collapsible teaching plans

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

# Set up environment variables
cp .env.example .env  # Then edit with your keys

# Start Redis (required for conversation memory)
brew services start redis  # or: docker run -p 6379:6379 redis

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Development (two servers)

```bash
# Terminal 1: Backend
uvicorn main:app --reload --port 8080

# Terminal 2: Frontend (with hot reload)
cd frontend && npm run dev
```
Open **http://localhost:5173**

### Production (single server)

```bash
# Build frontend
cd frontend && npm run build && cd ..

# Run server (serves API + frontend)
uvicorn main:app --port 8080
```
Open **http://localhost:8080**

## API Usage

### Streaming (Recommended)

```bash
# Stream chat responses with SSE
curl -N -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "How do transformers work?", "thread_id": "user-123"}'
```

SSE Events:
- `thinking` - Planning tokens (JSON teaching plan)
- `diagram` - Retrieved diagram metadata
- `token` - Response text tokens
- `plan` - Parsed teaching plan object
- `done` - Final state with referenced diagrams

### Non-Streaming

```bash
# Chat with the agent (blocking)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do transformers work?", "thread_id": "user-123"}'
```

### Response Format

```json
{
  "response": "Transformers use self-attention to process sequences...",
  "diagrams": [
    {
      "id": "diagram_042",
      "score": 0.00045,
      "query": "transformer self-attention mechanism",
      "description": "Diagram showing Q, K, V matrices...",
      "vision_description": "This diagram illustrates the scaled dot-product attention...",
      "vision_latency_s": 2.5,
      "post_url": "https://jalammar.github.io/illustrated-transformer/"
    }
  ],
  "plan": {
    "topic": "Transformers",
    "steps": ["Explain self-attention", "Describe Q, K, V matrices", "..."],
    "diagrams_needed": ["attention mechanism", "encoder-decoder"]
  }
}
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

# 3. Evaluate SigLIP vs CLIP (offline, self-contained)
python benchmark/scripts/evaluate.py
```

### LangSmith Eval

The retrieval benchmark is also runnable via LangSmith for traced runs, a versioned dataset, and a side-by-side comparison UI. End-to-end agent quality (tool-call correctness + LLM-judge on explanation quality) ships as a separate script.

```bash
# Requires LANGSMITH_API_KEY in .env

# One-time: upload benchmark queries as a LangSmith dataset
python benchmark/scripts/upload_to_langsmith.py

# Retrieval eval (SigLIP + CLIP as evaluate() targets)
python benchmark/scripts/langsmith_evaluate_retrieval.py

# Full agent eval (LangGraph end-to-end; ~30 min, ~$1-3 in Gemini calls)
python benchmark/scripts/langsmith_evaluate_agent.py --limit 5   # smoke test
python benchmark/scripts/langsmith_evaluate_agent.py             # full run
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
    ├── evaluate.py                  # Offline SigLIP-vs-CLIP baseline
    ├── upload_to_langsmith.py       # Upload dataset to LangSmith
    ├── langsmith_evaluate_retrieval.py        # Retrieval eval via LangSmith
    └── langsmith_evaluate_agent.py  # End-to-end agent eval via LangSmith

model/
├── scorer.py                     # SigLIP scorer
├── gemma_tokenizer.model         # Tokenizer (~4MB)
└── siglip2_so400m14_384.npz      # Checkpoint (~1.5GB, gitignored)

agent/
├── graph.py                      # LangGraph agent (Plan-and-Execute)
└── tools.py                      # MCP client: persistent stdio session, wrapped tools

mcp_server/
└── diagram_server.py             # MCP server: SigLIP scorer + retrieve_diagram tool
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional
REDIS_URL=redis://localhost:6379  # Default
ENABLE_VISION=true                # Enable vision review (default: true)

# Optional - LangSmith tracing & eval
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true            # Auto-trace live agent runs
LANGSMITH_PROJECT=mentorml-prod   # Default project: "default"
```

## Architecture

```
User Query → Plan Node (CoT) → Execute Node (ReAct) ⇄ Tools → Response
                                      ↓
                              retrieve_diagram  ←── MCP stdio ──→  diagram_server.py
                                                                   (SigLIP scorer +
                                                                    Gemini vision review)
```

### Frontend

```
frontend/src/
├── components/
│   ├── Chat.tsx         # Main container
│   ├── ChatInput.tsx    # Input with send/stop/clear
│   ├── DiagramCard.tsx  # Diagram display with source link
│   ├── Message.tsx      # Message bubble + ThinkingSection
│   └── MessageList.tsx  # Message list + empty state
├── hooks/
│   └── useStreamChat.ts # SSE streaming hook
└── types.ts             # TypeScript interfaces
```

## Requirements

- Python 3.10+
- Node.js 18+ (for frontend)
- Redis (for conversation memory)
- ~4GB disk space (SigLIP checkpoint + diagrams)
- Gemini API key