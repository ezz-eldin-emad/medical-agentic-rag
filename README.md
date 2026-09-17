# Medical Agentic RAG

A medical-information RAG demo with a local polling mode and a no-card Render webhook deployment. It is educational only and must not be used for diagnosis or clinical decisions.

Telegram and source scraping still require network access. AWS Lambda, API Gateway, and SQS are not part of the runtime.

## Architecture

```text
Telegram Bot API
        │ polling (local) or HTTPS webhook (Render)
        ▼
local Python process
   ├── Modal BGE-M3 (Render) or local BGE-M3 (development)
   ├── Qdrant Cloud (Render) or local Qdrant (development)
   └── Groq/Gemini/etc. LLM API through LiteLLM
```

| Component | Default | Supported alternative |
| --- | --- | --- |
| Telegram adapter | Local polling process | None; Telegram is external |
| Qdrant | Local on-disk database | Local Qdrant server or Qdrant Cloud |
| BGE-M3 | Local FlagEmbedding | Modal GPU endpoint |
| Rewriter/generator | Groq/Gemini API | Any LiteLLM-compatible provider |

Medical responses use one structured internal result with separate renderers:
the Telegram user receives only the safe patient response, while validated
evidence IDs, claims, retrieval details, prompts, scores, and evaluations stay
internal for audit and Phoenix tracing. Citations are optional presentation
data and are never generated from every retrieved chunk automatically.

## Install

Python 3.10+ is required.

```bash
uv venv --python 3.11 .venv
uv sync --extra local-ml --extra ingestion
cp .env.example .env
```

Run the application with the project interpreter. The dependency is named
`python-telegram-bot`; do not install the unrelated package named `telegram`.

```bash
./.venv/bin/python -c "import sys, telegram; print(sys.executable); print(telegram.__version__)"
./.venv/bin/python main.py --check --local
```

If your shell prompt shows `(.venv)` but `python` still resolves to
`/usr/bin/python`, it is using an old environment activation. From this
project directory, run:

```bash
deactivate 2>/dev/null || true
source .venv/bin/activate
hash -r
python -c "import sys; print(sys.executable)"
python main.py --check --local
```

The printed interpreter must be inside this project's `.venv`. The explicit
`./.venv/bin/python ...` form always bypasses stale shell activation. Avoid
`uv run` for normal startup unless the environment is already synchronized;
otherwise it may resolve and download the full ML dependency graph, including
Torch.

Set the Telegram token and the API key for the selected LLM provider. The example uses Groq:

```env
TELEGRAM_BOT_TOKEN=...
GROQ_API_KEY=...
```

Application defaults—including model assignments, agent parameters, retrieval
limits, paths, and timeouts—are centralized in `src/config/settings.py`.
`.env` is reserved for credentials and deployment-specific endpoints/switches.
To change a model or agent policy, edit the typed defaults in that file; do not
add behavioral settings to `.env`.

## Local-first configuration

The simplest setup uses Qdrant's on-disk mode and local BGE-M3. It does not require Docker or a separate Qdrant process:

```env
QDRANT_PATH=data/qdrant
EMBEDDER_BACKEND=local
```

The BGE-M3 model and FP16 defaults are configured centrally in
`src/config/settings.py`.

The first embedding run downloads BGE-M3 and loads it into the Python process. It needs several GB of disk/RAM and is slower on CPU. Use `BGE_USE_FP16=true` only with suitable GPU support.

On-disk Qdrant is single-process storage. Index before starting the bot. If multiple processes must access Qdrant simultaneously, use the local server option:

```bash
docker run --name medical-qdrant -p 6333:6333 qdrant/qdrant
```

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

`--local` on the commands below applies in-memory settings overrides for
on-disk Qdrant and local BGE-M3. It does not change the LLM provider.

## Build the knowledge base

Ingestion and indexing are separate from the Telegram runtime:

```bash
./.venv/bin/python -m src.ingestion.loader
./.venv/bin/python -m src.ingestion.cleaner
./.venv/bin/python -m src.ingestion.llm_cleaner
./.venv/bin/python -m src.chunking.chunker
./.venv/bin/python -m src.vectordb.vector_store --local
```

The scrapers need internet access. The LLM cleaning stage uses the configured provider API. The chunk file is written to `data/chunks/chunks.json`.

The indexer also supports mixed mode when `--local` is omitted: it follows `QDRANT_URL`, `QDRANT_PATH`, and `EMBEDDER_BACKEND` from `.env`.

## Run and verify

Check the local dependencies without loading BGE-M3 or calling the LLM:

```bash
./.venv/bin/python main.py --check --local
```

Start the Telegram bot:

```bash
./.venv/bin/python main.py --bot --local
```

Keep the process running. In Telegram, try `/start`, `/status`, `/help`, and then a medical question. Stop it with `Ctrl-C`.

You can also use the Makefile:

```bash
make install
make index
make check
make bot
```

## Optional mixed embedding mode

If local BGE-M3 is too slow or the machine lacks enough memory, keep the Python bot and Qdrant choice local while using a deployed Modal embedder:

```env
EMBEDDER_BACKEND=modal
MODAL_EMBED_URL=https://your-endpoint.modal.run
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

For Qdrant Cloud, replace `QDRANT_URL` and set `QDRANT_API_KEY`.

## Render + Qdrant Cloud deployment

The committed `render.yaml` starts FastAPI with `uv sync --locked --no-dev`.
Set these variables manually in Render (never commit their values):

```env
TELEGRAM_TRANSPORT=webhook
PUBLIC_BASE_URL=https://<service>.onrender.com
TELEGRAM_WEBHOOK_SECRET=<random-secret>
STATE_BACKEND=qdrant
QDRANT_URL=https://<cluster>.qdrant.io
QDRANT_API_KEY=...
QDRANT_STATE_COLLECTION=medical_app_state
EMBEDDER_BACKEND=modal
MODAL_EMBED_URL=https://<modal-endpoint>
MODAL_EMBED_TOKEN=...
GROQ_API_KEY=...
TELEGRAM_BOT_TOKEN=...
PHOENIX_ENABLED=false
```

Create the Telegram webhook after the Render service is healthy:

```bash
curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
  -d "url=$PUBLIC_BASE_URL/telegram/webhook" \
  -d "secret_token=$TELEGRAM_WEBHOOK_SECRET" \
  -d 'drop_pending_updates=false' \
  -d 'allowed_updates=["message"]'
```

Render Free sleeps after inactivity and has an ephemeral filesystem; persistent
patient-session, booking, and Telegram idempotency state therefore uses the
`medical_app_state` Qdrant collection. This is suitable for a portfolio demo,
not an always-on or transaction-safe clinical service.

The Modal deployment is optional:

```bash
./.venv/bin/python -m pip install -e '.[modal]'
modal setup
modal deploy modal/app.py
```

## Troubleshooting

- **Missing `medical_kb`:** run the indexer and confirm that the collection exists.
- **BGE-M3 is too slow or runs out of memory:** use a machine with more RAM/GPU, re-index with a smaller compatible model, or use Modal.
- **LLM authentication errors:** match the model prefix to its key (`groq/...` with `GROQ_API_KEY`, `gemini/...` with `GEMINI_API_KEY`, and so on).
- **No Telegram response:** keep the local polling process running and allow outbound access to the Telegram Bot API.
- **`No module named 'telegram'` or wrong Telegram package:** activate/use `.venv`, uninstall `telegram`, then install `python-telegram-bot` with `uv pip --python .venv/bin/python`. Do not launch with system `/usr/bin/python`.
- **Stale cloud settings:** use `--local` or remove old `QDRANT_URL`, `MODAL_EMBED_URL`, and `EMBEDDER_BACKEND` values from `.env`.

## Project structure

```text
medical-agentic-rag/
├── main.py                 # Local Telegram runtime and startup check
├── modal/app.py            # Optional Modal BGE-M3 deployment
├── prompts/                # Cleaning and RAG prompts
├── src/
│   ├── adapters/           # Telegram polling/webhook adapter
│   ├── agents/             # Orchestrator, clinic, safety, documentation, and tracing
│   ├── config/             # Central typed application settings
│   ├── chunking/           # Semantic chunking
│   ├── embeddings/         # Local and optional Modal BGE-M3 backends
│   ├── ingestion/          # Scrapers and cleaning stages
│   ├── llm/                # LiteLLM client and configuration
│   ├── guardrails/         # Input sanitization and query classification
│   ├── memory/             # Local or Qdrant-backed sanitized context
│   ├── state/              # Qdrant persistent app state
│   ├── web/                # FastAPI Render webhook entrypoint
│   ├── observability/      # Phoenix/OpenTelemetry tracing seam
│   ├── rag/                # Retrieval, relevance gate, provenance, generation
│   ├── response/           # Patient and internal response renderers
│   ├── utils/              # Environment, logging, and shared helpers
│   └── vectordb/           # Qdrant indexing
└── data/                   # Generated data and local Qdrant storage (gitignored)
```
