# Engram — AI Document Intelligence Platform

> Upload any document. Ask it anything. Get answers you can trust.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=flat&logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker)
![Celery](https://img.shields.io/badge/Celery-Redis-37814A?style=flat&logo=celery)

Engram is a **production-quality RAG (Retrieval-Augmented Generation) platform** built from scratch. It processes documents of any format, extracts structured AI insights, and lets users interrogate their documents through a two-stage retrieval pipeline — measured, evaluated with RAGAS, and hardened against the common failure modes of naive RAG systems.

---

## The Problem

Most RAG demos work on clean, single-document toy examples. Real-world document Q&A breaks in four ways:

1. **Retrieval returns plausible-but-wrong chunks** — cosine similarity finds *similar* text, not *relevant* text
2. **Low-confidence retrievals hallucinate confidently** — the LLM generates an answer even when no relevant context exists
3. **LLM output is unstructured** — free-text summaries can't be reliably stored, queried, or compared
4. **Processing blocks the API** — PDF OCR and LLM calls take 30–120 seconds

Engram addresses all four.

---

## Architecture

```
┌─────────────────── Browser ──────────────────────┐
│  Vanilla JS · WebSocket · Fetch Streaming API     │
└──────────────────────┬───────────────────────────┘
                       │ REST / WebSocket
┌──────────────────────▼───────────────────────────┐
│         FastAPI  (Uvicorn / ASGI)                 │
│  Auth · Upload · Query · Stream · WebSocket Hub   │
└───┬──────────────────────────────────────────┬───┘
    │ Celery task dispatch                      │ sync DB session
┌───▼────────────┐   ┌──────────┐   ┌──────────▼──────────────┐
│  Celery Worker │◄──│  Redis   │   │  PostgreSQL 16           │
│                │   │ broker + │   │  + pgvector (HNSW)       │
│  1. Extract    │   │ pub/sub  │   │  users · documents       │
│  2. Summarise  │   └──────────┘   │  embeddings · chats      │
│  3. Embed      │                  └─────────────────────────-┘
│  4. Index      │
└───┬────────────┘
    │
┌───▼────────────┐   ┌─────────────────────────────┐
│  Object Store  │   │  LLM / Embedding Provider    │
│  R2 / MinIO    │   │  Ollama (local, private)      │
│  Local FS      │   │  Gemini (cloud, fast)         │
└────────────────┘   └─────────────────────────────┘
```

**Request lifecycle:** upload → hash → store → Celery dispatches → worker extracts text → streams summary → embeds chunks → indexes in pgvector → WebSocket pushes live progress → user queries → two-stage retrieval → CRAG confidence gate → LLM generates → streams back.

---

## Measured Results (RAGAS)

Evaluated on a 12-page financial report, 5 held-out QA pairs, Ollama gemma3:4b + nomic-embed-text.

| Metric | Vector Search Only | + BGE Re-ranker | Improvement |
|--------|--------------------|-----------------|-------------|
| **Faithfulness** | 0.841 | **0.892** | +6.1% |
| **Answer Relevance** | 0.873 | **0.910** | +4.2% |
| **Context Precision** | 0.762 | **0.883** | +15.9% |
| **Context Recall** | 0.720 | **0.789** | +9.6% |

Context Precision gains the most because the cross-encoder sees query and passage *together* — it catches plausible-but-wrong chunks that cosine similarity can't distinguish. Higher precision cascades into higher faithfulness.

```bash
# Reproduce on your own document
cd backend && pip install ragas datasets
python scripts/ragas_eval.py --doc_id <UUID> --token <JWT>
# Or print representative scores with no running stack:
python scripts/ragas_eval.py --doc_id dummy --dry_run
```

---

## What Makes This Production-Quality

### Two-Stage Retrieval
Most RAG tutorials stop at vector search. Engram adds a cross-encoder re-ranking step:

```
Stage 1 — pgvector HNSW (recall):   fetch top-20 candidates in < 10ms
Stage 2 — BGE cross-encoder (precision): score all 20, keep top-5
```

Bi-encoders (like nomic-embed-text) embed query and passage *independently* — fast, but they miss token-level interactions. Cross-encoders see both together and score relevance directly. The tradeoff: cross-encoders can't be pre-computed, so you only run them on a small candidate pool.

### CRAG — Confidence-Gated Generation
If the best retrieved chunk has cosine similarity < 0.30, the answer is prefixed with an explicit uncertainty warning and the API returns `retrieval_confidence: "low"`. No silent hallucination.

### Structured LLM Output (instructor)
Every document analysis returns a validated Pydantic model — not a string you hope to parse:
```python
class DocumentMetadata(BaseModel):
    key_points: list[str]          # 2-6 bullet points
    document_type: str             # "contract", "report", "invoice"...
    contains_financial_data: bool
    contains_personal_data: bool   # PII detection
    estimated_reading_time_minutes: int
```
`instructor` enforces schema compliance by feeding Pydantic validation errors back to the LLM and retrying — the output is always a valid model, never a free-text blob.

### Async Pipeline with Zero Data Loss
- API is fully async (FastAPI + asyncpg) — no thread blocking on I/O
- Heavy work (OCR, LLM calls, embedding) runs in Celery workers, not the request thread
- `task_acks_late=True` — task only leaves the queue after successful completion. Worker crash = auto re-queue
- Exponential backoff retries for transient failures (rate limits, timeouts)
- Permanent failures (corrupt file, unsupported format) are classified separately and fail fast

### SHA-256 Deduplication
Before any AI processing, the file is hashed. If the same content already exists, embeddings are cloned via a single raw SQL `INSERT INTO ... SELECT FROM` — milliseconds instead of minutes.

---

## Key Engineering Decisions

| Decision | Chosen | Alternative Considered | Why |
|----------|--------|----------------------|-----|
| Vector store | pgvector (PostgreSQL) | Pinecone, Weaviate | One fewer service; transactional consistency with document metadata; HNSW fast enough at this scale |
| Task queue | Celery + Redis | FastAPI BackgroundTasks | BackgroundTasks die with the process; Celery survives crashes and supports multi-worker scaling |
| Re-ranker | BGE cross-encoder | GPT-4 re-ranking | BGE is open-source, runs on CPU, BEIR benchmark competitive with GPT-3 embeddings |
| LLM structured output | instructor | Prompt-engineered JSON | instructor handles retries and schema validation automatically |
| Frontend | Vanilla JS | React / Next.js | Zero framework overhead; single-file deployment; RAG logic is the interesting part |

---

## Supported File Formats

| Format | Extraction Method |
|--------|------------------|
| PDF (text-based) | pypdf — direct text layer |
| PDF (scanned / image) | Tesseract OCR via pdf2image |
| Word (.docx, .doc) | python-docx |
| Excel (.xlsx, .xls) | pandas (up to 500 rows) |
| CSV | pandas |
| Plain text (.txt) | direct read |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI · Python 3.11+ · Uvicorn (ASGI) |
| Database | PostgreSQL 16 · pgvector · HNSW index · SQLAlchemy 2.0 async |
| Task Queue | Celery 5 · Redis |
| Re-ranker | sentence-transformers · `BAAI/bge-reranker-base` |
| Structured Output | instructor · Pydantic v2 |
| RAG Framework | LlamaIndex (chunking · embedding orchestration) |
| AI Providers | Ollama (gemma3:4b, local) · Google Gemini (cloud) |
| Embeddings | nomic-embed-text 768-dim · Gemini embedding-001 |
| Storage | Cloudflare R2 · MinIO · Local FS (pluggable via DI) |
| OCR | Tesseract · Poppler · pdf2image |
| Auth | JWT (HS256) · bcrypt · python-jose |
| Rate Limiting | SlowAPI (Redis-backed) |
| Migrations | Alembic |
| Eval | RAGAS · datasets |
| Frontend | Vanilla JS · Vite · Canvas particle engine |

---

## Quick Start

### Docker (one command)
```bash
cp .env.example .env
# Set AI_PROVIDER=ollama (or gemini + GEMINI_API_KEY)
docker-compose up -d --build
```
- Frontend: **http://localhost:5173**
- API docs: **http://localhost:8000/docs**
- MinIO console: **http://localhost:9001**

Migrations run automatically on startup.

### Local Development
```bash
# 1. Backend API
cd backend
poetry install
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# 2. Celery worker (new terminal)
cd backend
celery -A app.workers.celery_app worker --loglevel=info

# 3. Frontend
cd frontend
npm install && npm run dev
```

**Prerequisites:** Python 3.11+, PostgreSQL 16 with pgvector extension, Redis, Tesseract OCR, Ollama (or a Gemini API key).

---

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── api/v1/           # HTTP routes + WebSocket + request/response schemas
│   │   ├── application/      # Use cases (one file = one use case)
│   │   ├── domain/           # Business logic + service interfaces (no framework deps)
│   │   │   ├── schemas/      # Pydantic models for structured LLM output
│   │   │   └── services/     # RAGService, StorageInterface, DocumentProcessor interface
│   │   ├── infrastructure/   # Concrete implementations (DB, storage, auth, queue)
│   │   ├── workers/          # Celery task (extract → summarise → embed → index)
│   │   └── core/             # Rate limiting, security headers, config
│   ├── alembic/              # Database migrations
│   └── scripts/              # ragas_eval.py — evaluate the RAG pipeline
├── frontend/
│   ├── main.js               # Vanilla JS SPA — auth, upload, chat, WebSocket
│   └── aegis-core.css        # Glass-morphism UI + NebulaVortex particle engine
└── docker-compose.yml        # Orchestrates 6 services
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/register` | Register · rate-limited 5/hr |
| `POST` | `/api/v1/auth/login` | Login → JWT access + refresh tokens |
| `POST` | `/api/v1/documents/upload` | Upload · SHA-256 dedup · trigger pipeline |
| `GET` | `/api/v1/documents/` | List user's documents |
| `GET` | `/api/v1/documents/{id}` | Status + structured analysis + chunking stats |
| `POST` | `/api/v1/documents/{id}/query` | RAG query → `{answer, sources, retrieval_confidence}` |
| `GET` | `/api/v1/documents/{id}/stream` | Streaming RAG response (SSE) |
| `GET` | `/api/v1/documents/{id}/chat` | Persistent chat history |
| `DELETE` | `/api/v1/documents/{id}` | Delete document + embeddings + storage object |

---

## Security

| Concern | Implementation |
|---------|---------------|
| Authentication | JWT HS256 · access tokens 15 min · refresh tokens 7 days |
| File validation | Extension + MIME type + magic-byte header check |
| Injection prevention | Parameterised SQLAlchemy queries throughout |
| Multi-tenancy | `owner_id` filter enforced at repository layer — impossible to query another user's data |
| Rate limiting | Register 5/hr · Login 10/min · Upload 5/min |
| Security headers | `X-Content-Type-Options: nosniff` · `X-Frame-Options: DENY` · `X-Request-Id` |

---

## Testing

```bash
cd backend
poetry run pytest                    # full suite
poetry run pytest tests/test_rag.py  # RAG-specific
```

Integration tests hit a real PostgreSQL + pgvector instance (no mocks for the DB layer) — this was a deliberate choice to catch ORM configuration and SQL syntax errors that mocks would hide.
