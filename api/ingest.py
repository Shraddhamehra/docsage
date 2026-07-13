"""PDF ingestion: extract text per page, split into overlapping chunks."""
from pypdf import PdfReader

# ~200 words per chunk keeps each chunk about one topic; 40 words of overlap
# so a sentence cut at a boundary still appears whole in the next chunk.
CHUNK_WORDS = 200
OVERLAP_WORDS = 40


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page that has text."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())  # collapse whitespace/newlines
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_words - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def pdf_to_chunks(pdf_path: str) -> list[tuple[int, str]]:
    """Full ingestion for one PDF: returns (page_number, chunk_text) pairs."""
    result = []
    for page_no, text in extract_pages(pdf_path):
        for chunk in chunk_text(text):
            result.append((page_no, chunk))
    return result


if __name__ == "__main__":
    import sys

    for page_no, chunk in pdf_to_chunks(sys.argv[1]):
        print(f"--- page {page_no} ({len(chunk.split())} words)")
        print(chunk[:120], "…")
