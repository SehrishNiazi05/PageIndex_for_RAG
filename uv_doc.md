# Setting up `PageIndex_for_RAG` with uv

Setup guide for [SehrishNiazi05/PageIndex_for_RAG](https://github.com/SehrishNiazi05/PageIndex_for_RAG)
— an agentic, vectorless RAG pipeline over dental textbooks (PageIndex tree index +
DeepSeek models).

The repo ships a `setup.sh` that builds a `venv` and installs with `pip`. This guide
replaces that with `uv`, which manages the Python interpreter, the virtualenv, and a
lockfile in one tool. **You still need the `PageIndex` repo cloned locally** (Step 4) —
that part isn't a Python-packaging concern and `uv` doesn't remove it.

Verified on macOS (Darwin 25.5, Apple Silicon) with `uv 0.8.4` and Python 3.11.9.

---

## 1. Prerequisites

| Tool | Why | Install |
|---|---|---|
| `uv` | environment + dependency management | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (or `brew install uv`) |
| `git` | clone this repo and PageIndex | Xcode CLI tools |
| `poppler` | provides `pdftotext`, used to extract page text | `brew install poppler` |
| DeepSeek API key | indexing + all three LLM calls per query | https://platform.deepseek.com |

Confirm poppler landed — `query_cli.py` shells out to `pdftotext` by name, so it must be
on `PATH`:

```bash
brew install poppler
which pdftotext     # expect /opt/homebrew/bin/pdftotext
```

You do **not** need a system Python 3.10+. `uv` downloads and manages its own
interpreter. This matters here: macOS ships Python 3.9 in many setups, and both
`batch_index.py` (uses `Path | None` syntax) and the `pageindex` package
(`requires-python >=3.10`) will fail on it.

---

## 2. Clone the project

```bash
git clone https://github.com/SehrishNiazi05/PageIndex_for_RAG.git
cd PageIndex_for_RAG
```

---

## 3. Create the environment with uv

The repo has no `pyproject.toml`, only a `requirements.txt`. Pick one of the two paths
below — **Option A** is recommended, since it gives you a lockfile and reproducible
installs.

### Option A — project-managed (recommended)

Create `pyproject.toml` in the repo root:

```toml
[project]
name = "dental-rag"
version = "0.1.0"
description = "Agentic vectorless RAG over dental textbooks (PageIndex + DeepSeek)"
requires-python = ">=3.11"
dependencies = [
    "pageindex",
    "openai>=1.40.0",
    "python-dotenv>=1.0.1",
    "tqdm>=4.66.0",
]

[tool.uv.sources]
pageindex = { git = "https://github.com/VectifyAI/PageIndex.git" }
```

Then:

```bash
uv python pin 3.11    # writes .python-version
uv sync               # creates .venv, resolves, writes uv.lock
```

`pageindex` isn't installed from PyPI here — the repo's `requirements.txt` pins the git
source (`pageindex @ git+https://github.com/VectifyAI/PageIndex.git`), and
`[tool.uv.sources]` is how uv expresses that. It pulls in `litellm`, `openai`,
`pypdfium2`, `PyPDF2`, `openai-agents`, and friends transitively.

If you'd rather generate the same file by command instead of pasting it:

```bash
uv init --python 3.11 --bare
uv add "pageindex @ git+https://github.com/VectifyAI/PageIndex.git" \
       openai python-dotenv tqdm
```

That writes an equivalent `pyproject.toml` (uv normalizes the git URL into
`[tool.uv.sources]` for you) — just note it resolves to whatever `openai` is current
rather than the repo's `>=1.40.0` floor.

### Option B — use the existing requirements.txt as-is

No new files, no lockfile:

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

This works — uv handles the git dependency in `requirements.txt` directly — but you get
no `uv.lock`, so installs aren't reproducible across machines. Use it for a quick trial,
Option A for real work.

---

## 4. Clone PageIndex into the project (required)

`batch_index.py` does not import the `pageindex` package. It shells out to
`./PageIndex/run_pageindex.py` as a subprocess and hard-fails if that file is missing.
Installing the package in Step 3 is **not** a substitute:

```bash
git clone https://github.com/VectifyAI/PageIndex.git
```

The directory must be named `PageIndex` and sit next to `batch_index.py`. It's already
in `.gitignore`, so it won't be committed.

You don't need to install PageIndex's own `requirements.txt` separately (as `setup.sh`
does). `batch_index.py` launches the script with `sys.executable`, which is your uv
venv's Python — and Step 3 already installed the same dependency set into it.

Sanity check:

```bash
uv run python PageIndex/run_pageindex.py --help
```

---

## 5. Folders and configuration

```bash
mkdir -p textbooks trees
cp .env.example .env
```

Put your textbook PDFs in `./textbooks/`, then edit `.env`:

```ini
DEEPSEEK_API_KEY=sk-...                          # required
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_TEXT_MODEL=deepseek-v4-flash            # indexing + routing + answering
PDF_DIR=./textbooks
TREE_DIR=./trees
MAX_PAGES_PER_QUERY=4                            # caps per-query text sent to the model
```

Two things worth knowing about filenames and the repo's existing state:

- **PDF stems must match the tree filenames.** PageIndex derives the tree name from the
  PDF stem: `carranza.pdf` → `carranza_pageindex.json`. `query_cli.py` reverses that to
  find the source PDF when extracting page text. Rename a PDF and text extraction breaks
  for that book.
- **`trees/` is not empty in the repo.** Three tree JSONs are already committed
  (Newman & Carranza, Contemporary Orthodontics, McCracken's). `query_cli.py` will load
  them immediately — but text extraction only works for books whose matching PDF is
  present in `textbooks/`. Use `--summary-only` if you want to query without the PDFs.

---

## 6. Run it

With uv there's nothing to activate — `uv run` resolves the environment per invocation.

```bash
# Step 1: build tree indexes (run once per PDF; re-run only if a PDF changes)
uv run batch_index.py

# Higher-quality, slower/pricier structure detection
uv run batch_index.py --mode standard

# Retry or add specific books only
uv run batch_index.py --only carranza.pdf shafer.pdf

# Step 2: interactive Q&A
uv run query_cli.py

# Cheap mode — answers from node summaries, skips PDF text extraction entirely
uv run query_cli.py --summary-only
```

Both scripts accept `--help`, which is the fastest way to confirm the environment is
sound before spending API credits.

If you prefer a conventional activated shell, that still works:

```bash
source .venv/bin/activate
python batch_index.py
```

---

## 7. Verify the setup

```bash
uv run python -c "import sys; print(sys.version)"          # expect 3.11.x
uv run python -c "import pageindex, openai, dotenv, tqdm"   # silent = OK
uv run batch_index.py --help
uv run query_cli.py --help
which pdftotext
```

`batch_index.py --help` works without an API key. `query_cli.py --help` does **not** —
see the first troubleshooting entry.

---

## Troubleshooting

**`openai.OpenAIError: Missing credentials` when running `query_cli.py`**
The `OpenAI()` client is constructed at module import time (`query_cli.py:41`), before
`argparse` runs — so even `--help` fails without a key. Confirm `.env` exists in the
directory you're running from and that `DEEPSEEK_API_KEY` is set to a real value, not
the `your_deepseek_api_key_here` placeholder.

**`Could not find .../PageIndex/run_pageindex.py`**
Step 4 was skipped, or the clone landed under a different directory name. `uv sync`
installing the `pageindex` package does not create this file.

**`pdftotext: command not found` / pages return no text**
Either poppler isn't installed (`brew install poppler`), or the pages are
scanned/image-only. The pipeline reads plain text only — no OCR, no images, tables, or
diagrams — and reports empty pages as such.

**`SyntaxError` on `Path | None`, or pageindex refuses to install**
You're on Python 3.9 or older. `uv python pin 3.11 && uv sync` fixes it; verify with
`uv run python -V`.

**Tree JSON doesn't match the book's real table of contents**
`--mode flash` (the default) is heuristic. Re-index that book with
`uv run batch_index.py --mode standard --only <file>.pdf` for LLM-verified structure.

**`WARNING: could not locate output JSON`**
`run_pageindex.py` writes `<stem>_structure.json` into `./results` relative to its own
directory; `batch_index.py` then moves it to `trees/<stem>_pageindex.json`. If PageIndex
upstream changes that output path, this search fails — check `PageIndex/results/` by hand.

---

## Notes on committing this

`.gitignore` already covers `.venv/`, `venv/`, `.env`, and `PageIndex/`. If you adopt
Option A, commit `pyproject.toml`, `uv.lock`, and `.python-version` — the lockfile is the
whole point. Keep `requirements.txt` around or delete it, but don't maintain both as
sources of truth; if you keep it, regenerate it with
`uv export --no-hashes -o requirements.txt` rather than editing it by hand.
