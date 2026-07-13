"""Flask REST API for DocSage."""
import os
import tempfile

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

from api.ask import ask
from api.db import get_conn, init_schema
from api.store import save_document

app = Flask(__name__)
CORS(app)

MAX_UPLOAD_MB = 20
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.post("/api/documents")
def upload_document():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "attach a PDF as 'file'"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "only PDF files are supported"}), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        file.save(tmp.name)
        try:
            doc_id = save_document(tmp.name, file.filename)
        except ValueError as e:
            return jsonify({"error": str(e)}), 422
    return jsonify({"id": doc_id, "filename": file.filename}), 201


@app.get("/api/documents")
def list_documents():
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.filename, d.page_count, d.uploaded_at, count(c.id) AS chunks
            FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id ORDER BY d.uploaded_at DESC
            """
        ).fetchall()
    return jsonify(
        [
            {"id": r[0], "filename": r[1], "pages": r[2], "uploaded_at": r[3].isoformat(), "chunks": r[4]}
            for r in rows
        ]
    )


@app.post("/api/ask")
def ask_question():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "body must include a non-empty 'question'"}), 400
    return jsonify(ask(question))


if __name__ == "__main__":
    init_schema()
    app.run(port=int(os.environ.get("PORT", 5001)), debug=True)
