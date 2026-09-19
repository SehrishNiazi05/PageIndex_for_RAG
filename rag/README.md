# AI Dentist local reference

This app uses the nine existing `docling_extraction/processed/*.jsonl` books. It does not ingest the unprocessed oral surgery PDF or store patient history. Questions and selected textbook excerpts are sent to DeepSeek when a key is configured.

## Setup

From the repository root:

```sh
python3 -m venv rag/.venv
rag/.venv/bin/pip install -r rag/requirements.txt
python3 -m rag audit
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 rag/.venv/bin/python -m rag ingest
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 rag/.venv/bin/python -m rag serve
```

Open `http://127.0.0.1:8000`. The server command rejects non-local host names. Model downloads for BGE embedding and reranking require network access on the first run. Full dense ingestion may take substantial time and disk space; reruns skip books whose source and settings are unchanged. The generated index lives in `rag/data/`.
The two BGE models were downloaded into `rag/data/hf` during setup on this computer, so the shown offline commands use those cached files. Remove the two offline environment flags if the cache is absent on another computer.

Put your DeepSeek key in `rag/.env.local` as `DEEPSEEK_API_KEY=...`. That file is ignored by git, read locally at startup, and never written to traces. Do not paste the key in chat or commit it.

For a dependency-free lexical baseline, use `python3 -m rag ingest --no-dense`. This mode uses SQLite FTS5 if `bm25s` is absent; with `bm25s` installed it uses that package. A later `python3 -m rag ingest` rebuilds books with BGE embeddings and Qdrant vectors.

## Commands

```sh
python3 -m rag retrieve "What resists distal extension base rotation?"
python3 -m rag ask "Why is a rubber dam used?" --audience clinician
python3 -m rag eval
rag/.venv/bin/python -m rag eval --generate
python3 -m unittest discover -s rag/tests
```

Routes: `POST /api/ask` with `{ "question": "...", "audience": "clinician" | "patient" }`, `GET /api/health`, `GET /api/source/{chunk_id}`, `GET /source/{chunk_id}`, and `GET /api/pdf/{book}`. Citation pages are **PDF pages**, not printed page numbers. The web view opens PDFs with `#page=N`.

The app refuses unsupported answers and personal patient symptom questions. It is a textbook reference, not a diagnostic service. The current evaluation set is exploratory; clinical review is required before patient-facing use.

The latest measured audit and evaluation reports are `rag/data/corpus_audit.json` and `rag/data/evaluation.md`.
