"""Local-only FastAPI application. Start with python -m rag serve."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from html import escape
from urllib.parse import quote
from pydantic import BaseModel, Field
from .agent import ask
from .config import PDFS, DATA
from .store import connect, get_chunk, settings_hash

app = FastAPI(title="AI Dentist", docs_url=None, redoc_url=None)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    audience: str = "clinician"


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/static/{name}")
def static(name: str):
    if name not in ("style.css", "app.js"): raise HTTPException(404)
    return FileResponse(Path(__file__).parent / "static" / name,
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
@app.get("/health")
def health():
    db = connect()
    books = db.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    current = db.execute("SELECT COUNT(*) FROM books WHERE settings IN (?,?)",
                         (settings_hash(False), settings_hash(True))).fetchone()[0]
    db.close()
    status = "ready" if books == 9 and current == books and chunks else "index_outdated" if chunks else "index_missing"
    return {"status":status, "books": books, "chunks": chunks,
            "generation_configured": bool(__import__("os").getenv("DEEPSEEK_API_KEY"))}


@app.post("/api/ask")
def api_ask(body: AskRequest):
    if health()["status"] != "ready":
        raise HTTPException(503, "Index needs rebuilding: run rag/.venv/bin/python -m rag ingest")
    try: return ask(body.question, body.audience)
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc


@app.get("/api/source/{chunk_id}")
def source(chunk_id: str):
    db = connect(); row = get_chunk(db, chunk_id); db.close()
    if not row: raise HTTPException(404, "Source not found")
    row["pdf_url"] = f"/api/pdf/{row['book']}#page={row['page_start']}" if row["page_start"] else None
    return row


@app.get("/source/{chunk_id}", response_class=HTMLResponse)
def source_view(chunk_id: str):
    db = connect(); row = get_chunk(db, chunk_id); db.close()
    if not row: raise HTTPException(404, "Source not found")
    link = f"/api/pdf/{quote(row['book'])}#page={row['page_start'] or 1}"
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Cited passage · AI Dentist</title><link rel='stylesheet' href='/static/style.css'>"
            "<main style='max-width:800px;margin:4rem auto;padding:0 1.5rem'>"
            f"<a href='/'>← AI Dentist</a><h1>{escape(row['book'])}</h1>"
            f"<p>{escape(row['chapter'] or '')} · PDF page {row['page_start']}</p>"
            f"<p><a href='{link}' target='_blank' rel='noopener noreferrer'>Open PDF page ↗</a></p>"
            f"<div style='white-space:pre-wrap;line-height:1.7;background:white;padding:1.5rem;border-radius:12px'>{escape(row['text'])}</div>"
            f"<p>Source records: {escape(', '.join(row['source_ids']))}</p></main></html>")


@app.get("/api/pdf/{book}")
def pdf(book: str):
    # Exact corpus stem allowlist; FileResponse never receives a raw path.
    db = connect(); exists = db.execute("SELECT 1 FROM books WHERE book=?", (book,)).fetchone(); db.close()
    path = PDFS / (book + ".pdf")
    if not exists or not path.is_file(): raise HTTPException(404, "PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name, content_disposition_type="inline")
