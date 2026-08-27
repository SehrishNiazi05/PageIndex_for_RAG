# Dental Assistant — Agentic Text RAG (PageIndex + DeepSeek)

Agentic, vectorless RAG pipeline over 9 dental textbooks. Uses PageIndex to build a
hierarchical tree index of each PDF (no vector DB, no chunking), and DeepSeek models
to reason over the tree and answer questions — reading only the plain text of the
selected pages (no images, tables, charts, or diagrams).

## How it works

1. **Index once per book** (`batch_index.py`): PageIndex parses each PDF's structure
   into a JSON tree (chapters -> sections -> subsections, each with a page range and
   an LLM-written summary).
2. **Query time** (`query_cli.py`), per question:
   - **Book routing**: `deepseek-v4-flash` picks which of the 9 books are relevant.
   - **Tree search**: `deepseek-v4-flash` reasons over that book's tree (titles +
     summaries) and picks the most relevant section(s).
   - **Text extraction**: the chosen section's pages are read as plain text via
     poppler's `pdftotext` (no image rendering).
   - **Text answer**: `deepseek-v4-flash` answers the question grounded in that
     extracted page text.

This keeps cost down — the text model is only called on the few pages that
tree search actually selects, not the whole corpus.

## Project layout

```
dental-rag/
├── .env.example         # copy to .env and fill in your DeepSeek key
├── pyproject.toml       # uv project (deps + pageindex git source)
├── uv.lock              # pinned dependency graph (reproducible installs)
├── .python-version      # uv-pinned Python (3.12)
├── batch_index.py       # Step 1: build tree index for all PDFs
├── hybrid_index.py      # Step 1 (alt): flash-first, quality-gated indexer
├── query_cli.py         # Step 2: interactive Q&A loop
├── PageIndex/           # cloned dependency (required at runtime)
├── textbooks/           # put your 9 PDFs here
└── trees/               # generated *_pageindex.json tree files
```

## Setup (macOS)

Prerequisites: `uv`, `git`, and poppler (for `pdftotext`/`pdfinfo`).

```bash
brew install uv poppler

# 1. Clone PageIndex alongside this repo (batch_index.py shells out to it)
git clone https://github.com/VectifyAI/PageIndex.git

# 2. Create the environment + install deps
uv python pin 3.12
uv sync

# 3. Put your 9 PDFs in ./textbooks/
cp /path/to/*.pdf textbooks/

# 4. Add your DeepSeek key
cp .env.example .env
# set DEEPSEEK_API_KEY=sk-...
```

Then (no `source activate` needed — `uv run` resolves the environment):

```bash
# Build the tree index for all 9 books (run once; re-run only if a PDF changes)
uv run hybrid_index.py
# or: uv run batch_index.py

# Ask questions
uv run query_cli.py
```

## Notes

- `batch_index.py --mode flash` (default) is fast/cheap and heuristic-based.
  Use `--mode standard` for LLM-verified structure if flash mis-splits a book's
  chapters (check the generated JSON against the book's real table of contents).
- `query_cli.py --summary-only` skips PDF text extraction and answers only from
  node summaries — useful for quick, cheap sanity checks.
- File names matter: PageIndex derives the tree file name from the PDF's file
  stem (e.g. `carranza.pdf` -> `carranza_pageindex.json`). Keep the same stem
  so `query_cli.py` can find the source PDF when it needs to extract page text.
- Text extraction relies on `pdftotext` (poppler).
  Pages that are scanned/image-only will return no text and are reported as such.
