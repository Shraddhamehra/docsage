# DocSage — ask questions to your study PDFs

I built this to solve my own problem: during placement prep I have dozens of course PDFs (DBMS notes, ML unit PDFs, past papers) and finding "where did we cover normalization vs denormalization?" means scrolling through hundreds of pages. DocSage lets me upload my PDFs and ask questions in plain English — it answers **with citations to the exact page**, so I can trust and verify every answer.

## How it works

```
PDF upload → text extraction (pypdf) → chunking → embeddings (fastembed, MiniLM ONNX)
    → PostgreSQL + pgvector (cosine similarity search)
    → top chunks + question → LLM (Groq) → answer with page citations
```

- **Backend**: Flask REST API (Python)
- **Vector store**: PostgreSQL + pgvector — I already know Postgres, and it means one database for both metadata and vectors instead of adding a separate vector DB
- **Embeddings**: fastembed (ONNX MiniLM) — runs on CPU, no GPU or paid API needed for embeddings
- **LLM**: LLaMA via Groq API — fast inference, free tier
- **Frontend**: React (Vite)

## Why not just keyword search?

The app includes a keyword-search baseline so you can compare side by side. Semantic search finds "how do I make my model stop memorizing the training data?" → overfitting/regularization chunks, which keyword search misses entirely. Measured comparison in [eval/](eval/).

## Evaluation

Retrieval and generation are evaluated separately (numbers updated as the eval set grows):

- **Retrieval**: hit-rate@k on a labeled set of question → correct-chunk pairs, vs the keyword baseline. Current run (small starter corpus): semantic 10/10, keyword baseline 2/10 — note: with a corpus this small hit@4 flatters the semantic side (k ≈ corpus size), so treat the keyword number as the real signal until the corpus grows
- **Generation**: faithfulness spot-checks — does the answer only use facts present in the retrieved chunks?
- **Guardrail**: if no chunk clears the similarity threshold, DocSage says it doesn't know instead of guessing

## Run it locally

```bash
# 1. Postgres with pgvector, then:
createdb docsage && psql -d docsage -c "CREATE EXTENSION vector;"

# 2. Backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY + DATABASE_URL
.venv/bin/python -m api.app

# 3. Frontend
cd web && npm install && npm run dev
```

## Status

- [x] Project scaffold, database with pgvector
- [x] PDF ingestion + chunking
- [x] Embeddings + vector search
- [x] Ask endpoint (RAG pipeline + guardrail)
- [x] Keyword baseline + eval set
- [x] React frontend
- [ ] Deploy (Render + Vercel + Supabase)
