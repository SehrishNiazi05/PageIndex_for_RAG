"""Hybrid retrieval: Qdrant dense + bm25s lexical, RRF, local reranker."""
import re
from functools import lru_cache
from .config import DATA, COLLECTION, EMBED_MODEL, RERANK_MODEL, CONTEXT_TOKENS
from .store import connect, get_chunk, settings_hash


def _rrf(lists, k=60):
    scores = {}
    for hits in lists:
        for rank, chunk_id in enumerate(hits, 1): scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True), scores


class Retriever:
    def __init__(self):
        self.db = connect()
        self.qdrant = self.encoder = self.reranker = None
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer, CrossEncoder
            total_books = self.db.execute("SELECT COUNT(*) FROM books").fetchone()[0]
            dense_books = self.db.execute("SELECT COUNT(*) FROM books WHERE settings=?", (settings_hash(True),)).fetchone()[0]
            if total_books == 9 and dense_books == total_books and (DATA / "qdrant").exists():
                self.qdrant = QdrantClient(path=str(DATA / "qdrant"))
                if self.qdrant.collection_exists(COLLECTION):
                    self.encoder = SentenceTransformer(EMBED_MODEL)
                    self.reranker = CrossEncoder(RERANK_MODEL)
        except (ImportError, RuntimeError, ValueError):
            pass
        self.bm25 = self.bm25_ids = None
        try:
            import bm25s
            rows = self.db.execute("SELECT chunk_id,heading,text FROM chunks ORDER BY chunk_id").fetchall()
            self.bm25_ids = [r[0] for r in rows]
            if rows:
                self.bm25 = bm25s.BM25()
                self.bm25.index(bm25s.tokenize([r[1] + " " + r[2] for r in rows], stopwords="en"))
        except ImportError:
            pass

    def search(self, query, book=None, limit=8):
        stages = {}
        lexical = []
        if self.bm25:
            import bm25s
            hits, _ = self.bm25.retrieve(bm25s.tokenize([query], stopwords="en"), k=min(50, len(self.bm25_ids)))
            lexical = [self.bm25_ids[int(i)] for i in hits[0]]
        else:
            terms = re.findall(r"[\w]+", query)
            if terms:
                match = " OR ".join('"' + t.replace('"', '') + '"' for t in terms[:16])
                lexical = [r[0] for r in self.db.execute(
                    "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 50", (match,))]
        if book:
            lexical = [i for i in lexical if (self.db.execute("SELECT book FROM chunks WHERE chunk_id=?", (i,)).fetchone() or [None])[0] == book]
        stages["lexical"] = lexical
        dense = []
        if self.encoder and self.qdrant:
            from qdrant_client import models
            vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
            flt = models.Filter(must=[models.FieldCondition(key="book", match=models.MatchValue(value=book))]) if book else None
            hits = self.qdrant.query_points(COLLECTION, query=vector, query_filter=flt, limit=50).points
            dense = [h.payload["chunk_id"] for h in hits]
        stages["dense"] = dense
        fused, scores = _rrf([lexical, dense])
        stages["fused"] = fused[:25]
        chunks = [get_chunk(self.db, i) for i in fused[:25]]
        chunks = [c for c in chunks if c]
        if self.reranker and chunks:
            vals = self.reranker.predict([(query, c["heading"] + "\n" + c["text"]) for c in chunks])
            for c, val in zip(chunks, vals): c["rerank_score"] = float(val)
            chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        else:
            for c in chunks: c["rerank_score"] = None
        # Reference headings sometimes leak into subsequent body text. Retain
        # them for bibliographic questions; downweight otherwise.
        if not re.search(r"\b(reference|citation|bibliography)\b", query, re.I):
            chunks.sort(key=lambda c: (c["rerank_score"] if c["rerank_score"] is not None else scores[c["chunk_id"]])
                        - ((0.1 if self.reranker else 0.003) if c["reference_path"] else 0), reverse=True)
        stages["reranked"] = [c["chunk_id"] for c in chunks]
        selected, used = [], 0
        for c in chunks:
            if len(selected) >= limit: break
            if used + c["n_tokens"] > CONTEXT_TOKENS: continue
            selected.append(c); used += c["n_tokens"]
        stages["selected"] = [c["chunk_id"] for c in selected]
        return selected, stages


@lru_cache(maxsize=1)
def get_retriever():
    return Retriever()
