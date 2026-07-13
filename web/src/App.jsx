import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:5001";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const fileInput = useRef(null);

  async function loadDocuments() {
    try {
      const res = await fetch(`${API}/api/documents`);
      setDocuments(await res.json());
    } catch {
      setError("Can't reach the API — is the backend running?");
    }
  }

  useEffect(() => {
    loadDocuments();
  }, []);

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/api/documents`, { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error);
      await loadDocuments();
    } catch (err) {
      setError(err.message || "upload failed");
    } finally {
      setUploading(false);
      fileInput.current.value = "";
    }
  }

  async function handleAsk(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setAsking(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error);
      setResult(body);
    } catch (err) {
      setError(err.message || "something went wrong");
    } finally {
      setAsking(false);
    }
  }

  return (
    <main className="app">
      <header>
        <h1>DocSage</h1>
        <p>Ask questions to your study PDFs — answers cite the exact page.</p>
      </header>

      <section className="panel">
        <div className="panel-head">
          <h2>Your documents</h2>
          <label className="upload-btn">
            {uploading ? "Uploading…" : "+ Add PDF"}
            <input ref={fileInput} type="file" accept=".pdf" onChange={handleUpload} hidden />
          </label>
        </div>
        {documents.length === 0 ? (
          <p className="muted">No documents yet — upload a PDF to get started.</p>
        ) : (
          <ul className="doc-list">
            {documents.map((d) => (
              <li key={d.id}>
                <span className="doc-name">{d.filename}</span>
                <span className="muted">{d.pages} pages · {d.chunks} chunks</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel">
        <h2>Ask</h2>
        <form onSubmit={handleAsk} className="ask-form">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. what is normalization and where did we cover it?"
          />
          <button disabled={asking || !question.trim()}>{asking ? "Thinking…" : "Ask"}</button>
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className={`answer ${result.guardrail_triggered ? "not-found" : ""}`}>
            <p>{result.answer}</p>
            {result.sources.length > 0 && (
              <div className="sources">
                {result.sources.map((s, i) => (
                  <span key={i} className="source-chip">
                    {s.filename} · p.{s.page} · {(s.similarity * 100).toFixed(0)}%
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </section>
    </main>
  );
}
