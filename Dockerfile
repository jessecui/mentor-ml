# =============================================================================
# Stage 1: Build Frontend
# =============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies
COPY frontend/package*.json ./
RUN npm ci

# Build production bundle
COPY frontend/ ./
RUN npm run build

# =============================================================================
# Stage 2: Python Runtime
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone big_vision (required for SigLIP)
RUN git clone --quiet --branch=main --depth=1 \
    https://github.com/google-research/big_vision big_vision_repo

# Download model files
RUN mkdir -p model && \
    curl -L -o model/gemma_tokenizer.model \
    https://storage.googleapis.com/big_vision/paligemma_tokenizer.model && \
    curl -L -o model/siglip2_so400m14_384.npz \
    https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m14_384.npz

# Copy application code
COPY main.py .
COPY agent/ agent/
COPY model/*.py model/
COPY benchmark/ benchmark/

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Railway uses PORT env var (default 8080 for local testing)
ENV PORT=8080
EXPOSE ${PORT}

# Start server (Railway sets PORT automatically)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}