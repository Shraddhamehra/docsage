"""Store PDFs in the database and search them (semantic + keyword baseline)."""
from pgvector.psycopg import register_vector

from api.db import get_conn
from api.embeddings import embed_query, embed_texts
from api.ingest import pdf_to_chunks


def save_document(pdf_path: str, filename: str) -> int:
    """Ingest one PDF: chunk it, embed every chunk, store everything. Returns document id."""
    page_chunks = pdf_to_chunks(pdf_path)
    if not page_chunks:
        raise ValueError("no extractable text in PDF")
    embeddings = embed_texts([chunk for _, chunk in page_chunks])
    page_count = max(page for page, _ in page_chunks)

    with get_conn() as conn:
        register_vector(conn)
        doc_id = conn.execute(
            "INSERT INTO documents (filename, page_count) VALUES (%s, %s) RETURNING id",
            (filename, page_count),
        ).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, page, content, embedding) VALUES (%s, %s, %s, %s)",
                [
                    (doc_id, page, chunk, embedding)
                    for (page, chunk), embedding in zip(page_chunks, embeddings)
                ],
            )
    return doc_id


def vector_search(question: str, k: int = 4) -> list[dict]:
    """Semantic search: chunks closest to the question in embedding space.

    Returns similarity as 1 - cosine_distance (1.0 = identical meaning).
    """
    query_embedding = embed_query(question)
    with get_conn() as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT c.id, d.filename, c.page, c.content,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, k),
        ).fetchall()
    return [
        {"chunk_id": r[0], "filename": r[1], "page": r[2], "content": r[3], "similarity": float(r[4])}
        for r in rows
    ]


def keyword_search(question: str, k: int = 4) -> list[dict]:
    """Baseline: Postgres full-text search. Only matches literal words, not meaning."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id, d.filename, c.page, c.content,
                   ts_rank(to_tsvector('english', c.content),
                           plainto_tsquery('english', %s)) AS rank
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (question, question, k),
        ).fetchall()
    return [
        {"chunk_id": r[0], "filename": r[1], "page": r[2], "content": r[3], "rank": float(r[4])}
        for r in rows
    ]
