# AI Dentist v1 — implementation and measured status

Updated 2026-09-17. The prior document was a proposed design. This document describes the implementation under `rag/` and records what has actually run.

## Scope

- Nine processed Docling books are the only answer corpus. The unprocessed surgery PDF is excluded.
- A local FastAPI app serves clinician/student reference and patient general education at `127.0.0.1`.
- Questions and selected textbook excerpts go to DeepSeek when `DEEPSEEK_API_KEY` is configured. No patient history is stored; trace records omit raw questions and answer text.
- The app is a reference for review against original PDF pages. Patient mode declines first-person questions.

## Pipeline

1. `audit.py` reports size outliers, merged-list page issues, repeated headings, reference-path body fraction, and table samples in `rag/data/corpus_audit.json`.
2. `chunker.py` reads processed records and raw Docling provenance without changing either. It reconstructs individual merged list item IDs and PDF pages, skips the contents/preface leak before PDF page 38 in White and Pharoah, uses actual heading occurrences as parent IDs, splits oversize content, and joins adjacent table/caption records when within the budget. Chunks are capped at 512 tokens. When the BGE tokenizer is cached, it counts BGE tokens; offline it uses a conservative UTF-8 bound. Every chunk carries source IDs, exact source text spans, and a PDF page.
3. `store.py` versions each book by source SHA and chunk/model settings, and writes a SQLite chunk registry with FTS5. Full ingestion creates local BGE embeddings in Qdrant; `--no-dense` builds a lexical baseline without model downloads. Rerunning skips unchanged books.
4. `retrieve.py` fuses `bm25s` and Qdrant top results with reciprocal rank fusion, then applies local BGE reranking. If optional packages are absent, SQLite FTS5 provides a lexical baseline. Reference-path material is downweighted for ordinary questions, preserving potentially useful text. Parent expansion is disabled pending measured benefit. The context budget is 7,000 tokens.
5. `agent.py` performs retrieval, at most two targeted searches, answer synthesis through `deepseek-flash`, and an independent support check. Complex synthesis enables thinking mode. The model returns claim/evidence ID/verbatim quote triples. Code accepts only evidence IDs from retrieved chunks and quotes that exactly appear within one source record span; it constructs the displayed citation page and source ID from that span. Any rejected claim or failed support check makes the answer abstain.
6. `app.py` exposes `POST /api/ask`, source JSON and HTML routes, PDF files, and a health route. The UI displays citations beside claims and opens the cited PDF page.

## Measured baseline

The read-only audit covered **106,800 processed records**. It found **882 records longer than 2,000 characters**, **12,349 list items whose recovered PDF page differs from the first item in their merged record**, and **24 table records without a markdown grid**. McDonald has **32.7%** of body characters under a reference-like heading path; this remains an indexing risk. The earlier 68% estimate was not reproduced by this audit's definition. Audit samples are in `rag/data/corpus_audit.json`.

The prior offline lexical build completed across all nine books with **59,606 chunks**. With the project virtual environment's `bm25s`, results from the exploratory 15-case set (13 answerable, two refusal probes) were:

| Metric | Result |
|---|---:|
| Hit@5 | 76.9% (10/13) |
| Hit@10 | 76.9% (10/13) |
| MRR | 0.490 |
| Citation accuracy | Not run |
| Reviewed clinical errors | Not run |
| Patient safety review | Not run |

These are small-sample retrieval diagnostics, not release thresholds. The set spans all nine books, a dosage table, cross-book synthesis, a false-premise correction, both audience modes, and two refusal probes. The cross-book case missed at top 10; retrieval needs improvement. It needs expansion and independently reviewed answers. The FastAPI health, home, source, PDF range, and patient-boundary routes passed local HTTP checks. The BGE embedding and reranking models loaded locally, but the full dense ingest was stopped at the user's request. The latest chunking code reserves room for model special tokens and **requires a fresh ingest** before serving; the health route reports `index_outdated` until then. No DeepSeek key was available for generation or clinical review. The release gate remains **zero fabricated citations and zero unsafe patient-specific advice in a reviewed set**; it has not been met or claimed.

## Setup and next verification

See `rag/README.md`. Install `rag/requirements.txt`, run `python -m rag ingest` to build dense vectors, configure `DEEPSEEK_API_KEY`, and run `python -m rag eval` plus reviewed answer cases. Compare lexical and hybrid retrieval before enabling parent expansion or setting an abstention cutoff. The prior two-day estimate is provisional.

DeepSeek's current model alias is `deepseek-flash`; thinking mode uses the API `thinking` parameter. See [DeepSeek release notes](https://api-docs.deepseek.com/news/news260910/) and [thinking-mode documentation](https://api-docs.deepseek.com/guides/thinking_mode/). The patient boundary follows the need for careful health AI validation described by [WHO](https://www.who.int/news/item/16-05-2023-who-calls-for-safe-and-ethical-ai-for-health) and the independent-review concern in [FDA clinical decision support guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software).
