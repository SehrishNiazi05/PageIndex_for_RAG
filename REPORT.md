# Project Report — Dental Assistant: Agentic Vectorless RAG over Textbooks

> A complete record of what has been built, the problems hit, and the plan for a
> production UI + backend + database. Use this to build the demo presentation.

---

## 1. The Big Picture (What & Why)

I am building an AI assistant that answers clinical questions from **9 dental
textbooks**. Instead of reading a 1,000-page book for every question, the system
first **organizes each book into a hierarchical table of contents (a "tree")** —
chapters → sections → subsections, each with a page range. At query time the
assistant **reasons down the tree** to find the right section, extracts only those
pages, and answers from that text.

> **One-liner:** *I turn textbooks into a searchable table of contents so the AI
> jumps straight to the right section instead of scanning the whole book.*

**Why "vectorless"?** Classic RAG embeds fixed-size chunks into a vector database
and retrieves by *similarity*. But similarity ≠ relevance. This project instead
uses **reasoning-based retrieval**: an LLM navigates a structured tree the way a
human flips to the right section of a long book.

| | Classic Vector RAG | This project (PageIndex-style) |
|---|---|---|
| Index | vector index of chunks | hierarchical tree index |
| Unit | fixed-size chunks | natural sections (chapter/section) |
| Retrieval | semantic similarity | LLM reasoning over the tree |
| Traceability | opaque ("vibe") | explicit page-level references |
| Cost driver | embedding + vector DB | LLM calls on the selected pages only |

---

## 2. The Books (Raw Material)

9 dental textbooks in `textbooks/`:

| # | Book | Pages |
|---|---|---|
| 1 | Newman & Carranza's Clinical Periodontology (13e) | ~1000 |
| 2 | Contemporary Orthodontics | 812 |
| 3 | McCracken's Removable Partial Prosthodontics (12e) | 390 |
| 4 | McDonald & Avery's Dentistry for the Child and Adolescent (10e) | 711 |
| 5 | Oral Medicine and Radiology (Kulkarni) | 261 |
| 6 | Scully's Medical Problems in Dentistry (7e) | 816 |
| 7 | Shafer's Textbook of Oral Pathology | 1001 |
| 8 | Sturdevant's Art and Science of Operative Dentistry | 1020 |
| 9 | White & Pharoah's Oral Radiology | 1608 |

---

## 3. What Has Been Built (Current State)

### 3.1 The Indexing Pipeline

Three scripts produce a tree (TOC) for each PDF:

| Script | Role | Cost |
|---|---|---|
| `batch_index.py` | Runs PageIndex (`flash` or `standard`) on all PDFs | flash ≈ free, standard ≈ expensive |
| `hybrid_index.py` | **Flash-first, quality-gated** batch indexer with validation, checkpointing, retry/backoff, per-book logging | same as above |
| `manual_toc.py` | **Custom fallback**: free heading/Contents detection + one LLM call to build the tree | ~$0.03/book |

**Final pipeline flow** (in `hybrid_index.py`):

```
flash (fast heuristic, near-free)
   │
   ├─ passes validation ──────────────► done
   │
   └─ fails (empty structure) ───────► manual (free detection + ONE LLM call)
                                          │
                                          └─ standard is now opt-in only
                                             (--force-standard)
```

Validation is **local and free** (no LLM): checks top-level node count, empty
titles, page-range sanity, catch-all "leaf spanning hundreds of pages" detection,
and page-coverage vs the real PDF length (via `pdfinfo`).

### 3.2 The Query Loop (`query_cli.py`)

A CLI used to prove the knowledge source works, per question:

1. **Book routing** — `deepseek-v4-flash` picks the relevant book(s) from short descriptions.
2. **Tree search** — reasons over the book's tree (titles + summaries) to pick the best section(s).
3. **Text extraction** — reads the chosen section's pages as plain text via `pdftotext`.
4. **Answer** — `deepseek-v4-flash` answers grounded strictly in that extracted text.

### 3.3 Result — all 9 books indexed

| Book | Method used | Tree nodes |
|---|---|---|
| Carranza's Periodontology | PageIndex (existing) | 1818 |
| Contemporary Orthodontics | PageIndex (existing) | 231 |
| McCracken's Prosthodontics | PageIndex (existing) | 376 |
| Oral Medicine & Radiology | flash (bookmarks) | 260 |
| **McDonald & Avery's** | **manual** | 41 (5 parts, 31 chapters) |
| **Scully's Medical Problems** | **manual** | 301 (chapters → sections) |
| **Shafer's Oral Pathology** | **manual** | 33 (6 sections, 21 chapters) |
| **Sturdevant's Operative** | **manual** | 21 chapters |
| **White & Pharoah's Radiology** | **manual** | 38 (4 parts, 33 chapters) |

**Cost to index the 5 hard books: ~$0.16 total.**

### 3.4 The Tree (Knowledge Source) Format

Each book is a JSON document in `trees/<book>_pageindex.json`:

```jsonc
{
  "doc_name": "Shafer_s Textbook of Oral Pathology .pdf",
  "toc_source": "manual",
  "structure": [
    {
      "title": "SECTION I DISTURBANCES OF DEVELOPMENT AND GROWTH",
      "node_id": "0001",
      "start_index": 28,
      "end_index": 341,
      "summary": "",
      "nodes": [
        {
          "title": "1. Developmental Disturbances of Oral and Paraoral Structures",
          "node_id": "0002",
          "start_index": 28,
          "end_index": 105,
          "summary": "",
          "nodes": []
        }
      ]
    }
  ]
}
```

Every node has `title`, `node_id`, `start_index`, `end_index`, `summary`,
and nested `nodes` (chapters → sections → subsections). Page numbers are
**1-based physical PDF page indexes**, so they map directly to `pdftotext`.

---

## 4. Issues / Challenges Faced

This is the most valuable part for the demo — it shows real engineering.

### 4.1 Flash mode fails on bookmark-less books
- PageIndex **flash** builds the tree from the PDF's *embedded bookmarks* + layout
  statistics. 5 of the 9 books have **zero embedded bookmarks**.
- For those, flash hits its "empty outline" gate and returns an empty structure
  → `ValueError: could not extract a structure`.
- *Fix:* detect bookmark absence and route to a fallback.

### 4.2 Standard mode is catastrophically expensive & unreliable
- Escalating to **standard** made ~1 LLM call **per section title** to verify
  start pages (e.g. `check_title_appearance`).
- **McDonald & Avery's (711 pages):** standard mode ran **4+ hours**, made
  **1092 LLM calls**, burned **$3.16**, then **crashed with no output**.
- The auto-retry logic then tried to re-run the 4-hour job *twice more*.
- *Fix:* made standard **opt-in only**, removed retry-on-standard, and built
  `manual_toc.py` (one LLM call per book) as the default fallback.

### 4.3 Heterogeneous book structures
The 9 books each structure themselves differently, so one heuristic can't fit all:

| Book | Structure convention |
|---|---|
| McDonald / Sturdevant | uppercase `CHAPTER N <title>` at chapter start |
| Shafer | `SECTION I–VI` + a printed Contents page |
| White & Pharoah | `Part I–IV` (Roman) + a full "Table of Contents" |
| Scully | **no** contents page, **no** "Chapter" keyword — only running headers |
| Oral Medicine | clean embedded bookmarks (flash just works) |

*Fix:* `manual_toc.py` detects each signal and lets one LLM call assemble the
final tree.

### 4.4 Printed-page vs physical-page offset
- Printed Contents pages use the *book's* page numbers, but the tree must use
  *physical PDF* page indexes. The front-matter offset varies per book.
- A naive offset estimate from running headers returned **1061** for White &
  Pharoah (correct answer: 8) because headers contained chapter/figure numbers.
- *Fix:* estimate the offset by **title-matching** — find a Contents title's
  physical heading in the body and compute `physical − printed`.

### 4.5 Flaky/empty LLM responses
- `deepseek-v4-flash` is a reasoning model; occasionally it burned its token
  budget on `thinking` and returned empty content (or a truncated JSON).
- *Fix:* added retry-with-backoff and empty-content handling in `manual_toc.py`.

### 4.6 Text-only limitation (known ceiling)
- Extraction uses `pdftotext` → **no images, radiographs, tables, or charts**.
  Dental content is heavily visual, so this is the biggest accuracy ceiling.

---

## 5. What Still Needs to Be Done (Future Work)

### 5.1 Retrieval quality (short term)
- [ ] Actually use **multi-node** retrieval — `query_cli.py` currently answers
      from only the top-1 node (`nodes[0]`) despite fetching top-2.
- [ ] Fix the **silent page cap** (`MAX_PAGES_PER_QUERY=4`) so long sections
      aren't truncated; expand pages adaptively.
- [ ] Add **node summaries** to the manual trees (cheap: one LLM call per book).
- [ ] Add a **fallback retriever** (BM25 keyword search over cached page text)
      for queries the tree search misses.

### 5.2 Multi-modal (medium term) — closes the biggest gap
- [ ] Add **vision** for radiographs / tables / clinical photos (e.g. a vision
      model pass on the selected pages, or OCR for scanned pages).

### 5.3 Productionization (medium term)
- [ ] Move trees from flat JSON files into a **database** (see §7).
- [ ] Build a **backend API** over the query loop (see §8).
- [ ] Build a **UI** (see §6).
- [ ] Auth, rate-limiting, caching, usage tracking, CI/CD.

### 5.4 Quality & evaluation
- [ ] Build a small **golden Q&A set** (e.g. 50 questions with known page refs)
      to measure retrieval accuracy and regressions.
- [ ] Compare against a classic vector-RAG baseline to quantify the win.

---

## 6. How to Interact With the Data — UI Options

The tree JSON + query loop can be exposed through a UI. Options, simplest first:

| Option | Stack | Effort | Best for |
|---|---|---|---|
| **Streamlit** | Python only, no JS | Lowest | Fast internal demo, quick prototyping |
| **Gradio** | Python only | Low | Demo with chat + citations UI built-in |
| **React + FastAPI** | TS + Python | Higher | Polished production product |
| **Chainlit** | Python | Low | Agentic/chat UIs with streaming |

**Recommended for the demo: Streamlit or Chainlit** (both talk to the same Python
backend, zero extra language). A minimal UI shows:

1. A **book selector / search box** for the question.
2. The **answer** with **citations** (book + section + page range).
3. A **tree browser** panel (collapsible TOC per book).
4. Optional **source highlight** (the extracted page text).

Sketch:

```
┌─────────────────────────────────────────────┐
│  Ask: "What are the radiological signs of... "│
├─────────────────────────────────────────────┤
│  Answer (grounded in: Shafer, p28–105)       │
│  ┌─ Sources ────────────────────────────────┐│
│  │ ▸ Shafer — SECTION I, Ch.1 (p28–105)    ││
│  │ ▸ White & Pharoah — Part III, Ch.23      ││
│  └──────────────────────────────────────────┘│
│  [Tree browser]  [Raw text]                  │
└─────────────────────────────────────────────┘
```

---

## 7. Where the Data (Trees) Should Be Stored — Database

Today the trees are flat JSON files in `trees/`. For a real product, move them
into a database. The trees are **hierarchical JSON**, which is the key factor.

| Database | Why / When to use |
|---|---|
| **MongoDB** (recommended) | Trees are nested JSON → store them **as-is** (no transformation). Flexible schema, easy to add metadata (book title, edition, upload date, model used, summaries). Good querying by `book`, `title`, `page` range. |
| **PostgreSQL + JSONB** | If you already have relational data (users, auth, billing). `JSONB` stores the tree and supports indexing; a bit more setup than Mongo for deep-nested data. |
| **Elasticsearch / OpenSearch** | If you later add **full-text/BM25** fallback search over page text. |
| **A vector DB** (Pinecone / Weaviate / Qdrant / Chroma) | Only if you later go **hybrid** (tree navigation + semantic search). Not needed for the current vectorless design. |

**Recommended architecture:**

```
PostgreSQL (or Mongo)  ←  app metadata: users, books, jobs, versions
      │
MongoDB (or JSONB)     ←  the tree documents (1 doc per book)
      │
Object storage (S3 / MinIO)  ←  the original PDFs (optional, for re-extraction)
      │
Optional: full-text index  ←  cached page text for BM25 fallback
```

**Suggested MongoDB document shape:**

```jsonc
{
  "_id": "shafer",
  "title": "Shafer's Textbook of Oral Pathology",
  "edition": 7,
  "source_pdf": "textbooks/Shafer_s Textbook of Oral Pathology .pdf",
  "page_count": 1001,
  "index_method": "manual",
  "indexed_at": "2026-08-29T01:57:00Z",
  "tree": { "structure": [ /* the tree JSON from §3.4 */ ] }
}
```

Storing the tree in the DB (instead of files) lets the backend serve it to the UI,
version it, and rebuild it per book independently.

---

## 8. Backend Architecture

Expose the existing query logic as a small API. Recommended stack: **FastAPI**
(Python — reuses `query_cli.py` / `manual_toc.py` directly, async, auto OpenAPI docs).

### 8.1 Endpoints (minimal)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/ask` | Ask a question → answer + citations (book, section, pages) |
| `GET` | `/api/books` | List indexed books |
| `GET` | `/api/books/{id}/tree` | Return a book's TOC tree (for the tree-browser UI) |
| `POST` | `/api/books/{id}/reindex` | Rebuild a book's tree (admin) |
| `GET` | `/api/books/{id}/page?from=&to=` | Raw text of a page range (for source view) |

### 8.2 Components

```
Browser (Streamlit / React)
        │  HTTP / WebSocket
        ▼
FastAPI backend  ──────►  MongoDB (trees + metadata)
        │                     │
        ├─ Book routing ─────► deepseek-v4-flash (LLM)
        ├─ Tree search  ─────► deepseek-v4-flash (LLM)
        ├─ Page extraction ──► pdftotext (poppler)
        └─ Answer generation ► deepseek-v4-flash (LLM)
```

### 8.3 Notes
- Keep the **LLM calls server-side** (API key never reaches the browser).
- Add **response caching** (same question + tree version → cached answer) to cut cost.
- Use the DB for **trees** and **job/version** tracking; keep `pdftotext` for
  on-demand page extraction (don't pre-extract all pages to the DB unless needed).

---

## 9. Cost Summary

| Activity | Cost |
|---|---|
| Index 5 hard books (manual fallback) | **~$0.16** |
| One wasted standard-mode run (McDonald) | **$3.16** (and 4 hrs, no output) |
| Remaining balance after all indexing | ~$15.78 |

Lesson: the tree-navigation design keeps query cost low (only the selected pages
are read), but the *indexing* method must be chosen carefully — flash/manual are
cheap; standard mode is not.

---

## 10. Proposed Roadmap

1. **Now** — finish retrieval fixes (multi-node, page cap, summaries, BM25 fallback).
2. **Demo** — Streamlit/Chainlit UI over the existing CLI logic.
3. **Next** — FastAPI backend + MongoDB persistence of trees.
4. **Then** — multi-modal (vision for radiographs/tables), golden evaluation set.
5. **Scale** — caching, auth, hybrid semantic fallback if needed.