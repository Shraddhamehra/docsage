"""Text → vector embeddings using fastembed (MiniLM, 384 dims, runs on CPU)."""
from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Loading the model takes a few seconds, so keep one instance for the whole app.
_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks (for storing)."""
    return [e.tolist() for e in get_model().embed(texts)]


def embed_query(question: str) -> list[float]:
    """Embed a user question (for searching)."""
    return next(get_model().query_embed(question)).tolist()
