# DocSage — Technical Report

**A retrieval-augmented question-answering system over personal PDFs, with page-level citations, a refusal guardrail, and a measured evaluation.**

Author: Shraddha Mehra · Repository: [github.com/Shraddhamehra/docsage](https://github.com/Shraddhamehra/docsage)

---

## 1. Executive summary

During placement preparation I accumulated dozens of course PDFs — lecture notes, database chapters, past papers — and finding a specific concept meant scrolling through hundreds of pages. Large language models can answer questions fluently, but they have never seen my notes and cannot cite a source, so their answers are unverifiable and frequently invented.

**DocSage solves this by retrieving the relevant passages from my own documents first, then requiring the model to answer only from those passages and cite the exact page.** If nothing relevant is found, it refuses to answer rather than guessing.

**Measured result:** on a 497-chunk corpus (Stanford CS229 machine-learning notes + database chapters), across 20 deliberately paraphrased questions, semantic retrieval found the correct passage in the top 4 results **15 times out of 20 (75%)**, while a keyword-search baseline scored **0 out of 20**.

---

## 2. The problem, stated precisely

An LLM's knowledge lives in its weights — lossy statistics, frozen at training time. Three consequences follow, and each one is a requirement:

| Gap | What it means | Requirement it creates |
|---|---|---|
| **Privacy** | My notes were never in any model's training data, and never will be | The system must supply the knowledge, not rely on the model's memory |
| **Recency** | Training data ends on a date; a PDF added today does not exist for the model | Adding a document must take seconds, not a training run |
| **Verification** | Even when correct, the model cannot say where a fact came from | Every claim must carry a source the user can open and check |
| **No abstain instinct** | The model must always emit text, so with no grounding it produces plausible fiction | The system must be able to say "I don't know" |

---

## 3. Why RAG (and not the alternatives)

Three architectures could close those gaps. I evaluated all three.

**Fine-tuning** — continue training the model on my documents.
*Rejected.* Fine-tuning reliably changes how a model *behaves* (tone, format, domain idiom), but it is unreliable at teaching facts you need to retrieve verbatim. More decisively: once a fact is dissolved into the weights, **its provenance is destroyed** — the model cannot cite a page. Citation was a hard requirement, so this door was closed regardless of cost. And every new PDF would require a training run.

**Stuffing the whole corpus into the prompt** — paste everything in, every time.
*Rejected, but not dismissed.* For a very small corpus this is genuinely the correct engineering choice, and building retrieval would be over-engineering. It breaks on four walls: you pay for every token on **every** request; the model must process the entire prompt before producing the first word; the corpus eventually exceeds any context window; and — most subtly — **quality degrades**, because models use facts buried in the middle of a long context measurably worse than facts in a short one. My corpus is ~130,000 tokens from four documents; a full semester would be many times that.

**Retrieval-augmented generation** — find the few relevant passages per question, and hand only those to the model.
*Chosen.* It closes all four gaps: the documents live in my store (privacy), a new PDF is searchable in seconds (recency), I know exactly which passages I passed in so I can label them and demand the labels back (verification), and — because retrieval yields a *similarity score* — I can threshold it and refuse before the model is ever called (abstain).

> **The one-line justification:** *fine-tuning changes how a model behaves; RAG changes what it can see right now.* Fresh, private, citable knowledge is a retrieval problem.

**The honest cost of RAG:** more moving parts (chunker, embedding model, vector store, index, threshold) and a real tuning surface. I accepted that because it converts an unmeasurable question ("is the model's memory trustworthy?") into a measurable one ("did retrieval find the right chunk?").

---

## 4. Architecture

DocSage is **two pipelines joined by a store**. They run at different times, at different frequencies, with different performance requirements.

```
╔══ INGESTION (offline · once per document) ═══════════════════════════╗
║                                                                       ║
║  PDF ──▶ pypdf: extract text per page ──▶ chunk (200 words / 40 overlap)
║                                                    │                  ║
║                                                    ▼                  ║
║                          fastembed · MiniLM ──▶ 384-dim vector each   ║
║                                                    │                  ║
╚════════════════════════════════════════════════════┼══════════════════╝
                                                     ▼
                        ┌──────────────────────────────────────────┐
                        │  PostgreSQL + pgvector                   │
                        │  documents · chunks (text, page, vector) │
                        │  HNSW index for similarity search        │
                        └──────────────────────────────────────────┘
                                                     ▲
╔══ QUERY (online · once per question) ═══════════════┼══════════════════╗
║                                                     │                  ║
║  question ──▶ MiniLM ──▶ 384-dim vector ────────────┘                  ║
║                              │                                         ║
║                              ▼  ORDER BY embedding <=> q  LIMIT 4      ║
║                        top-4 chunks + similarity scores                ║
║                              │                                         ║
║                              ▼                                         ║
║           ┌─── GUARDRAIL: best similarity < 0.25? ──▶ REFUSE ──────┐   ║
║           │    (the LLM is never called — costs nothing)           │   ║
║           ▼                                                        │   ║
║   prompt = [rules] + [4 excerpts, each labelled "file, p.N"] + [Q] │   ║
║                              │                                     │   ║
║                              ▼  HTTPS                              │   ║
║                    Groq · LLaMA 3.3 70B · temperature 0            │   ║
║                              │                                     │   ║
║                              ▼                                     ▼   ║
║                    answer with [file, p.N] citations    "I couldn't    ║
║                                                          find this."   ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### The local/remote split — a deliberate decision

**Embeddings run locally; generation runs remotely.** This is the architectural choice I am asked about most, and it has a clean justification:

| | Embedding (MiniLM) | Generation (LLaMA 70B) |
|---|---|---|
| Size | ~90 MB | ~140 GB |
| Runs on | This laptop's CPU, ~10ms | Specialised hardware, behind an API |
| Cost | Free, forever | Free tier (Groq) |
| Privacy | **Documents never leave the machine during ingestion** | Only the 4 retrieved excerpts (~1,200 tokens) are sent |

The cheap model runs where the data is; the expensive model runs where the hardware is. A consequence worth stating: **with the network disconnected, uploading and indexing a PDF still works perfectly — only asking a question fails.**

### What Groq is, and is not, doing

Groq is the **generator, and nothing else**. It is called from exactly one place (`api/ask.py`), *after* retrieval has already selected the passages.

- It does **not** perform the search — that is pgvector + MiniLM, entirely local. Groq never sees the corpus.
- It is **not involved in ingestion** — parsing, chunking and embedding are 100% local.
- It is **not called when the guardrail fires** — a refusal costs nothing and leaks nothing.
- It receives **only** the four retrieved excerpts and the question — never the documents.

Alternatives considered: a local model via Ollama (free and fully private, but materially worse answers, and it cannot run on the free hosting tier), or a paid API such as OpenAI (same architecture, one line changed in `ask.py`, but a per-call cost where Groq's free tier has none).

---

## 5. Component decisions (and the alternatives rejected)

| Component | Chosen | Why | Rejected alternative |
|---|---|---|---|
| **PDF parsing** | `pypdf` | Pure Python; one capability needed | PyMuPDF (faster, heavier); OCR (only needed for *scanned* PDFs, which have no text layer at all) |
| **Chunking** | Fixed 200-word windows, 40-word overlap | Predictable, debuggable, no dependencies, and I can defend every line. Overlap ensures a fact at a boundary survives intact in at least one chunk | Recursive/semantic chunking — the upgrade path, triggered by evidence that split ideas are hurting retrieval |
| **Embeddings** | `fastembed` (MiniLM, 384-dim, ONNX) | Runs on CPU with no GPU and no PyTorch; free; documents never leave the machine | sentence-transformers (drags in ~2GB of PyTorch); an embedding API (per-chunk cost, and the data leaves) |
| **Vector store** | **PostgreSQL + pgvector** | One database for metadata *and* vectors: one backup, one connection, and — critically — a **SQL JOIN from a chunk to its document, which is what gives citations their filename**. Impossible if the vectors lived in a separate service | Pinecone / Chroma / Weaviate — a second service for zero benefit at this scale. I'd revisit at tens of millions of vectors |
| **Index** | HNSW | Builds incrementally as rows arrive; strong recall. *Approximate by design* — a greedy graph walk can miss a marginally closer vector, which is the correct trade at any real scale | IVFFlat — requires a training step over existing data, awkward on a table that starts empty and grows |
| **Similarity** | Cosine (`<=>`) | Meaning lives in *direction*, not magnitude. Vectors are normalised, so it yields an interpretable −1..1 score — which is exactly what the guardrail thresholds on | Euclidean — ranks identically for unit vectors, but gives a less interpretable number |
| **Backend** | Flask | Minimal, matches the rest of my stack, and sync is fine at this load | FastAPI — genuinely better (typed validation, async, auto-docs) and the natural upgrade; Flask's simplicity won at this size |
| **LLM** | LLaMA 3.3 70B via Groq | Fast inference, free tier, and a 70B model cannot run on a laptop | Local Ollama (worse quality, can't run on free hosting); OpenAI (per-call cost) |
| **Frontend** | React (Vite) | Six pieces of state, one page — a framework would be resume-driven engineering | Next.js — deliberately deferred to a separate project, where routing and SSR will actually have a reason to exist |

---

## 6. Evaluation

This is the section most portfolio RAG projects do not have, and it is the one I consider most important. *"It retrieves and generates"* is table stakes; **the question is how you know it works.**

### Method

Retrieval and generation fail in completely different ways and have completely different fixes, so they are measured **separately**:

- **Retrieval** — a labelled set of 20 questions, each paired with a distinctive string from the passage that truly answers it. Metric: **hit-rate@4** — did the correct passage appear in the top 4 retrieved chunks? Run against **both** semantic search and a keyword-search baseline.
- **Generation** — faithfulness: is every claim in the answer supported by the retrieved excerpts, and is the citation label correct? Currently a manual protocol.

**The questions are deliberately paraphrased** so that no lexical shortcut can rescue them. Example: *"which algorithm alternates between assigning points to clusters and recomputing centroids?"* — a question that shares almost no words with the passage about k-means that answers it.

### Results (corpus: 497 chunks — CS229 ML notes, 2 DBMS chapters, a CV)

| Method | hit-rate@4 |
|---|---|
| **Semantic search** (MiniLM + pgvector) | **15 / 20 — 75%** |
| Keyword baseline (PostgreSQL full-text search) | **0 / 20 — 0%** |

The keyword baseline scoring *zero* is not a bug in the baseline — PostgreSQL full-text search is a serious implementation with stemming and ranking. It scores zero because it matches **words**, and humans ask questions in **meaning**. That gap is the entire justification for the embedding-based approach, quantified.

### ⭐ The mistake I caught, and what it taught me

**My first evaluation scored 10/10 for semantic search.** A perfect score.

It was worthless. At that point the corpus contained one document — my CV — which produced exactly **3 chunks**, and retrieval returns the top **4**. Semantic search was returning *every chunk in the database, on every query*. **It was structurally incapable of missing.** A completely broken retriever would also have scored 10/10.

The fix was to the *experiment*, not the code: load a real corpus (497 chunks) and write questions that could actually fail. The honest numbers above are the result — and two questions that had "passed" on the tiny corpus now fail, because with 497 chunks there is real competition for the top four slots.

> **The generalisable law: an evaluation is only meaningful if failure is possible. Before trusting any metric, ask what a deliberately broken system would have scored. If it isn't near zero, you are measuring nothing.**

The five current misses are not an embarrassment — they are a documented roadmap (see §9).

---

## 7. Hallucination defences

Reliability is engineered **around** the model, not requested **from** it. Six standard defences exist; DocSage implements four and a half.

| # | Defence | Status | How |
|---|---|---|---|
| 1 | **Grounding** | ✅ | Retrieved passages are placed in the context, so the most probable continuation *is* their content |
| 2 | **Refusal** | ✅ | If the best similarity < 0.25, code returns "I couldn't find this in your documents" — **the LLM is never called.** An `if` statement over a measured number, not a prompt |
| 3 | **Prompt contract** | ✅ | System rules: answer only from the excerpts; cite each claim as `[file, p.N]`; use this exact sentence if the answer isn't there |
| 4 | **Low temperature** | ✅ | `temperature=0` — the model's most confident reading of the source, and reproducible for evaluation |
| 5 | **Post-hoc verification** | ⬜ offline | Faithfulness is measured in the eval harness; promoting it to a runtime check is the next feature |
| 6 | **Human oversight** | ✅ | The UI shows source chips — filename, page, similarity % — so verification is one click away, and a low score visibly signals suspicion |

**Citations are an architectural property, not a model capability.** The model can only cite labels that I actually placed in its context. A bare LLM asked to cite will fabricate authentic-looking references — a failure that has led to real lawyers being sanctioned for filing invented case citations.

**Two failures survive this design**, and naming them precisely matters: **unfaithfulness** (over-claiming beyond the excerpts) and **misattribution** (right fact, wrong page label). Both are generation-side — which is exactly why evaluation measures retrieval and generation separately.

---

## 8. What broke, and what it taught me

**PostgreSQL rejected the DBMS slides.** Ingestion crashed with `PostgreSQL text fields cannot contain NUL (0x00) bytes`. The PDF's text layer contained NUL bytes — something no clean test file ever contains. Fixed at the extraction boundary by normalising the text as it enters the system, so every downstream consumer receives one canonical form. *Lesson: every data pipeline eventually meets bytes it did not expect, and the fix belongs at the boundary, not scattered downstream.*

**The evaluation that could not fail** (§6). *Lesson: make failure possible, or you are measuring nothing.*

**A deploy bug in the sibling project, caught before it shipped:** the analyst dashboard read a cleaned CSV that was git-ignored, and the hosting platform has no database — so it would have crashed on boot. It now rebuilds its own dataset from the committed raw data. *Lesson: reproducibility (`raw + code = derived`) is not ceremony; it is what lets software run on a machine that has never seen it.*

---

## 9. Limitations & future work

**Known limitations, stated plainly:**
- Retrieval misses 5 of 20 evaluation questions. Each is documented.
- Faithfulness is verified offline, not enforced at runtime.
- Single-turn only — follow-up questions ("and what about the second one?") would retrieve noise without **query rewriting**.
- Scanned PDFs yield nothing (no text layer); OCR is not implemented — a deliberate scope cut, returning an explicit `422` rather than silently storing an empty document.
- Embeddings are weak at negation and exact identifiers (`E1042` vs `E1043` look nearly identical to the model).

**Next steps, in priority order — each justified by the evaluation harness rather than by taste:**
1. **Hybrid search** — fuse the keyword baseline with semantic search. They fail in *different directions*: dense search misses exact identifiers; sparse search misses paraphrases.
2. **Reranking** — a cross-encoder over the top ~50 candidates. Usually the single largest quality gain available in RAG, because the retriever's bi-encoder never lets the query and the chunk "look at" each other.
3. **Runtime faithfulness check** — flag unsupported claims before they reach the user.
4. **Streaming responses** — ~99% of the perceived wait is the LLM generating tokens one at a time; streaming makes the answer appear as it is written.

---

## 10. Running it

```bash
# Database (Postgres with pgvector)
createdb docsage && psql -d docsage -c "CREATE EXTENSION vector;"

# Backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY and DATABASE_URL
.venv/bin/python -m api.app   # http://localhost:5001

# Frontend
cd web && npm install && npm run dev   # http://localhost:4173

# Evaluation
.venv/bin/python -m eval.run_eval
```

**Repository layout** — the structure mirrors the architecture, and dependencies point one way (`app → ask → store → {ingest, embeddings, db}`), never back:

```
api/
  app.py         HTTP boundary — routes, validation, status codes
  ask.py         the RAG pipeline — guardrail, prompt, Groq
  store.py       persistence + search (vector and keyword)
  ingest.py      PDF parsing + chunking      (knows nothing about the DB)
  embeddings.py  text → vectors               (knows nothing about the DB)
  db.py          connections + schema
web/             React frontend (a separate world)
eval/            the measurement harness      (imports api/; api/ never imports eval/)
schema.sql       the database's shape, in the database's own language
render.yaml      deployment blueprint
```

---

## 11. What this project demonstrates

- **Full-stack**: React → Flask REST API → PostgreSQL, with clean, defensible boundaries.
- **Applied AI engineering**: retrieval, embeddings, vector search, prompt design, guardrails — building *with* pretrained models rather than training them, which is what the market actually hires for.
- **Database depth**: schema design, foreign keys, transactions, indexes, and a vector extension.
- **Engineering judgment**: every dependency chosen against a named alternative, with the threshold at which I would change my mind.
- **Measurement over vibes**: an evaluation harness, a baseline, honest numbers — and a methodological error I found in my own work and fixed.
