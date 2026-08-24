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
├── .env.example        # copy to .env and fill in your DeepSeek key
├── requirements.txt
├── setup.sh             # one-time macOS setup
├── batch_index.py        # Step 1: build tree index for all PDFs
├── query_cli.py           # Step 2: interactive Q&A loop
├── PageIndex/            # cloned dependency (created by setup.sh)
├── textbooks/            # put your 9 PDFs here
└── trees/                # generated *_pageindex.json tree files
```

## Setup (macOS)

```bash
bash setup.sh
```

This installs Homebrew dependencies (poppler), creates a virtualenv, clones
PageIndex, and installs Python packages.

Then:

```bash
# 1. Put your 9 PDFs in ./textbooks/
cp /path/to/*.pdf textbooks/

# 2. Add your DeepSeek key
open .env   # or nano .env
# set DEEPSEEK_API_KEY=sk-...

# 3. Activate the environment
source venv/bin/activate

# 4. Build the tree index for all 9 books (run once; re-run only if a PDF changes)
python3 batch_index.py

# 5. Ask questions
python3 query_cli.py
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
- Text extraction relies on `pdftotext` (poppler), installed by `setup.sh`.
  Pages that are scanned/image-only will return no text and are reported as such.
