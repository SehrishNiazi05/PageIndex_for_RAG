"""Resumable SQLite registry, local Qdrant vectors, and bm25s index."""
import hashlib
import json
import sqlite3
from pathlib import Path
from .config import DATA, CHUNK_VERSION, EMBED_MODEL, RERANK_MODEL, COLLECTION
from .chunker import chunk_book
from .config import PROCESSED


def connect():
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA / "rag.sqlite", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS books(book TEXT PRIMARY KEY, source_sha TEXT, settings TEXT, chunks INTEGER, ingested_at TEXT);
    CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY, book TEXT, chapter TEXT, heading TEXT,
      parent_id TEXT, source_ids TEXT, page_start INTEGER, page_end INTEGER, label TEXT, text TEXT,
      n_tokens INTEGER, reference_path INTEGER, version TEXT);
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, heading, text, tokenize='unicode61');
    CREATE TABLE IF NOT EXISTS traces(trace_id TEXT PRIMARY KEY, ts TEXT, audience TEXT, question TEXT,
      status TEXT, answer TEXT, evidence_ids TEXT, stages TEXT, latency_ms INTEGER);
    """)
    columns = {r[1] for r in db.execute("PRAGMA table_info(chunks)")}
    if "source_spans" not in columns:
        db.execute("ALTER TABLE chunks ADD COLUMN source_spans TEXT")
    return db


def settings_hash(dense=True):
    value = json.dumps({"chunk": CHUNK_VERSION, "embed": EMBED_MODEL, "rerank": RERANK_MODEL,
                        "dense": dense}, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def ingest(book=None, dense=True, force=False):
    db = connect()
    paths = [PROCESSED / (book + ".jsonl")] if book else sorted(PROCESSED.glob("*.jsonl"))
    if any(not p.exists() for p in paths): raise FileNotFoundError(book)
    encoder = qdrant = None
    if dense:
        try:
            from sentence_transformers import SentenceTransformer
            from qdrant_client import QdrantClient, models
        except ImportError as e:
            raise RuntimeError("Install rag/requirements.txt, or use --no-dense for a lexical baseline") from e
        encoder = SentenceTransformer(EMBED_MODEL)
        qdrant = QdrantClient(path=str(DATA / "qdrant"))
        if not qdrant.collection_exists(COLLECTION):
            qdrant.create_collection(COLLECTION, vectors_config=models.VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE))
    results = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        old = db.execute("SELECT source_sha,settings FROM books WHERE book=?", (path.stem,)).fetchone()
        if old and old[0] == digest and old[1] == settings_hash(dense) and not force:
            results[path.stem] = "skipped"; continue
        # Clear the completed marker before touching vectors. An interrupted
        # upsert is then retried rather than mistaken for a completed book.
        with db:
            db.execute("DELETE FROM books WHERE book=?", (path.stem,))
        chunks = list(chunk_book(path))
        if qdrant:
            from qdrant_client import models
            # Each book is replaced independently; a failed batch leaves the
            # SQLite book marker absent so rerunning continues safely.
            qdrant.delete(COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="book", match=models.MatchValue(value=path.stem))])))
            for start in range(0, len(chunks), 64):
                batch = chunks[start:start+64]
                vectors = encoder.encode([c["heading"] + "\n" + c["text"] for c in batch],
                                         batch_size=16, normalize_embeddings=True, show_progress_bar=False)
                qdrant.upsert(COLLECTION, points=[models.PointStruct(
                    id=int(c["chunk_id"][:15], 16), vector=v.tolist(), payload={"chunk_id": c["chunk_id"], "book": c["book"]})
                    for c, v in zip(batch, vectors)])
        with db:
            db.execute("DELETE FROM chunks WHERE book=?", (path.stem,))
            db.executemany("INSERT INTO chunks(chunk_id,book,chapter,heading,parent_id,source_ids,page_start,page_end,label,text,n_tokens,reference_path,version,source_spans) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
                (c["chunk_id"], c["book"], c["chapter"], c["heading"], c["parent_id"], json.dumps(c["source_ids"]),
                 c["page_start"], c["page_end"], c["label"], c["text"], c["n_tokens"], int(c["reference_path"]), c["version"], json.dumps(c["source_spans"]))
                for c in chunks])
            db.execute("INSERT OR REPLACE INTO books VALUES (?,?,?, ?,datetime('now'))",
                       (path.stem, digest, settings_hash(dense), len(chunks)))
        results[path.stem] = len(chunks)
    # FTS is a derived index. Rebuild once after all book transactions; this
    # also repairs an interrupted prior run where book data committed first.
    with db:
        db.execute("DELETE FROM chunks_fts")
        db.execute("INSERT INTO chunks_fts(chunk_id,heading,text) SELECT chunk_id,heading,text FROM chunks")
    db.close()
    return results


def get_chunk(db, chunk_id):
    row = db.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
    if not row: return None
    out = dict(row)
    out["source_ids"] = json.loads(out["source_ids"])
    out["source_spans"] = json.loads(out["source_spans"] or "[]")
    return out
