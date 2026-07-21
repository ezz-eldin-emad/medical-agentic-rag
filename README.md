# Medical RAG Pipeline

A modular and clean pipeline designed for scraping, cleaning, chunking, and indexing medical knowledge and clinic operational data into Qdrant database using BGE-M3 embeddings.

## Project Structure

```
T1/
├── modal/               # Modal GPU serverless deployment app for BGE-M3
├── src/
│   ├── ingestion/       # Data scraping & cleaning (Regex + LLM)
│   ├── chunking/        # Semantic markdown section-based chunker
│   ├── embeddings/      # BGE-M3 embedder backends (Modal GPU API, Local FlagEmbedding)
│   ├── vectordb/        # Qdrant indexing logic (Cloud / Local)
│   ├── llm/             # LiteLLM client with API key rotation support
│   └── utils/           # Shared utility helpers
├── data/                # Scraped raw, processed, and chunked files (gitignored)
└── requirements.txt     # Project dependencies
```

---

## Getting Started

### 1. Installation
Install the project dependencies in your environment:
```bash
pip install -r requirements.txt
```

### 2. Deployment (Modal GPU Embedding Backend)
Deploy the serverless BGE-M3 embedding endpoint to Modal GPU (T4):
```bash
modal setup
modal deploy modal/app.py
```
*After deployment, copy the generated FastAPI endpoint URL to your `.env` file.*

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
# API Keys (multiple keys can be separated by commas for rotation)
GEMINI_API_KEY=your-api-key-1,your-api-key-2
HF_TOKEN=your-hf-token

# Qdrant Database Configuration
QDRANT_URL=https://your-qdrant-instance.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Embeddings Configuration
EMBEDDER_BACKEND=modal
MODAL_EMBED_URL=https://your-modal-endpoint-url.modal.run
```

---

## Pipeline Workflow

### Step 1: Ingestion & Scraping
Gather medical articles from various online sources:
```bash
python -m src.ingestion.loader
```

### Step 2: Text Cleaning
Apply regex-based boilerplate removal:
```bash
python -m src.ingestion.cleaner
```

Then structure and optimize content using Gemini:
```bash
python -m src.ingestion.llm_cleaner
```

### Step 3: Semantic Chunking
Segment documents by markdown section headers:
```bash
python -m src.chunking.chunker
```
*Output chunk file is saved to `data/chunks/chunks.json`.*

### Step 4: Vector DB Indexing
Index chunks with BGE-M3 (dense + sparse) into Qdrant:

- **Primary Cloud API Indexing** (runs embeddings via Modal GPU, saves to Qdrant Cloud):
  ```bash
  python -m src.vectordb.vector_store
  ```
- **Local Indexing** (runs embeddings locally using FlagEmbedding, saves to local Qdrant by default):
  ```bash
  python -m src.vectordb.vector_store --local
  ```
