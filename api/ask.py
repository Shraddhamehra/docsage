"""The RAG pipeline: retrieve relevant chunks, then ask the LLM to answer from them only."""
import os

from groq import Groq

from api.store import vector_search

LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4

# Guardrail: if even the BEST chunk is below this similarity, the documents
# probably don't contain the answer — say so instead of letting the LLM guess.
SIMILARITY_THRESHOLD = 0.25

SYSTEM_PROMPT = """You answer questions using ONLY the provided document excerpts.
Rules:
- Every claim must come from the excerpts. Do not add outside knowledge.
- Cite the source after each claim like [filename, p.3].
- If the excerpts don't contain the answer, say exactly: "I couldn't find this in your documents."
"""


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['filename']}, p.{c['page']}]\n{c['content']}" for c in chunks
    )


def ask(question: str) -> dict:
    chunks = vector_search(question, k=TOP_K)

    if not chunks or chunks[0]["similarity"] < SIMILARITY_THRESHOLD:
        return {
            "answer": "I couldn't find this in your documents.",
            "sources": [],
            "guardrail_triggered": True,
        }

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,  # deterministic-as-possible answers for factual Q&A
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Document excerpts:\n\n{build_context(chunks)}\n\nQuestion: {question}",
            },
        ],
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {"filename": c["filename"], "page": c["page"], "similarity": round(c["similarity"], 3)}
            for c in chunks
        ],
        "guardrail_triggered": False,
    }
