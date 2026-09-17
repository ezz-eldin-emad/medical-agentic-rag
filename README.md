# Medical Agentic RAG

Educational Telegram assistant combining agentic routing, evidence-grounded medical retrieval, and clinic services. Portfolio demonstration only—not a diagnostic or clinical-care system.

**Try the Telegram bot:** [@MedicalRag_bot](https://t.me/MedicalRag_bot)

## Architecture

```text
Telegram → FastAPI webhook (Vercel) → Agent Orchestrator
                                      ├─ Input Guardrail / Sanitizer
                                      ├─ Query Classifier Agent (LLM)
                                      │    ├─ Emergency → Safety Agent
                                      │    ├─ Clinic → Clinic Agent (retrieve + answer)
                                      │    └─ Medical → Documentation Agent
                                      │                         ├─ Query Rewriter Agent (LLM)
                                      │                         ├─ HF Embedding API
                                      │                         ├─ Qdrant medical_kb retrieval
                                      │                         ├─ Evidence / relevance gate
                                      │                         └─ Generator Agent (LLM)
                                      ├─ Inquiry Agent (LLM extraction + follow-up loop)
                                      ├─ Output Guardrail
                                      └─ Safe Telegram response
```

The supported runtime uses Qdrant Cloud, Hugging Face Inference API with `intfloat/multilingual-e5-large`, and Groq through LiteLLM.

## Features

- Arabic/English medical-information questions with evidence-gated answers.
- Clinic information, availability, booking, cancellation, and medical routes.
- LLM-assisted clarification loop with structured context extraction.
- Input sanitization, emergency routing, output guardrails, and safe rendering.
- Persistent sessions, bookings, and Telegram update idempotency in Qdrant.

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), a Telegram bot token, and an LLM key.

```bash
uv venv --python 3.12 .venv
uv sync --extra ingestion
cp .env.example .env
./.venv/bin/python main.py --check
./.venv/bin/python main.py --bot
```

The runtime requires the cloud environment variables described below.

## Indexing

```bash
./.venv/bin/python -m src.ingestion.loader
./.venv/bin/python -m src.ingestion.cleaner
./.venv/bin/python -m src.chunking.chunker
./.venv/bin/python -m src.vectordb.vector_store --mode clean
```

Cloud rebuild:

```bash
export EMBEDDER_BACKEND=huggingface
export HF_EMBED_MODEL=intfloat/multilingual-e5-large
./.venv/bin/python -m src.vectordb.vector_store --mode clean
```

Use the same embedding model for indexing and querying.

## Vercel deployment

Entrypoint: `src.web.app:app`. Add these variables in Vercel; never commit their values:

```env
TELEGRAM_TRANSPORT=webhook
PUBLIC_BASE_URL=https://<project>.vercel.app
TELEGRAM_WEBHOOK_SECRET=<random-secret>
TELEGRAM_BOT_TOKEN=<telegram-token>
GROQ_API_KEY=<groq-key>
EMBEDDER_BACKEND=huggingface
HF_TOKEN=<huggingface-token>
HF_EMBED_MODEL=intfloat/multilingual-e5-large
QDRANT_URL=https://<cluster>.qdrant.io
QDRANT_API_KEY=<qdrant-key>
QDRANT_STATE_COLLECTION=medical_app_state
PHOENIX_ENABLED=false
```

After deployment, verify `/healthz`, then register the webhook:

```bash
curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
  -d "url=$PUBLIC_BASE_URL/telegram/webhook" \
  -d "secret_token=$TELEGRAM_WEBHOOK_SECRET" \
  -d 'drop_pending_updates=false' \
  -d 'allowed_updates=["message"]'
```

Vercel may cold-start. This free-tier setup is for a low-traffic portfolio demo, not production clinical workloads.

## Prompts and configuration

Typed defaults live in `src/config/settings.py`. All editable prompt templates live under `prompts/`, separately from agent logic.

## Tests

```bash
./.venv/bin/python -m pytest -q
```

## Project layout

```text
main.py                 Local CLI/runtime
src/web                 FastAPI webhook
src/adapters            Telegram integration
src/agents              Orchestrator and domain agents
src/guardrails          Sanitization and routing
src/rag                 Retrieval and generation
src/embeddings          Embedding backends
src/state, src/memory   Persistent and local state
src/vectordb             Qdrant indexing
prompts                 All prompt templates
data                    Generated local artifacts (gitignored)
```

## Safety and privacy

Do not use real patient identifiers or medical records. Never expose API keys in logs, screenshots, commits, or issue reports. Responses are educational and do not replace a qualified healthcare professional.
