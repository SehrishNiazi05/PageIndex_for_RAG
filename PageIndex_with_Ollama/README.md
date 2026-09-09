# Local PageIndex: Deterministic & LLM-Hybrid Textbook Indexer

A zero-cost, local-first engine designed to extract structural document trees (Chapters, Sections, Subsections) and node summaries from massive PDF textbooks (1,000+ pages) without relying on existing Tables of Contents, printed page numbers, or paid cloud APIs.

By pairing C-accelerated deterministic PDF layout parsing (`PyMuPDF`) with a local LLM (`Qwen 2.5` via Ollama), this pipeline resolves layout hierarchies in minutes while keeping 100% of data on your machine.

---

## 🌟 Key Features

- **TOC-Independent**: Extracts natural hierarchies directly from document typography, spatial bounding boxes, and inline header anchors.
- **100% Free & Local**: No OpenAI/Anthropic cloud API keys required. Runs fully offline on local GPU hardware.
- **Apple Silicon Optimized**: Tested on a MacBook Air M4 (16GB unified memory), leveraging MPS acceleration for local LLM inference.
- **Tested Models**: Benchmarked with both `Qwen2.5-14B` and `Qwen2.5-7B` via Ollama — the 7B variant offers faster inference with a smaller memory footprint, useful on the 16GB unified memory tier.
- **High-Throughput Batching**: Processes 1,000+ page textbooks in ~8–12 minutes per book (14B) / ~4–7 minutes per book (7B).
- **PageIndex JSON Schema**: Generates clean, vectorless JSON document maps complete with titles, page spans, and local node summaries for RAG pipelines.

---

## 🏗 System Architecture

```text
       [ PDF Textbook (1000+ pgs) ]
                    │
                    ▼
     Stage 1: Deterministic Layout Engine (PyMuPDF)
    ──────────────────────────────────────────────
    • Global Font Histogram Sampling (Find Body Size)
    • Strip Headers, Footers & Page Noise (BBox Filters)
    • Match Regex Anchors (e.g., "Chapter 1", "3.2.1")
                    │
            Ambiguous Header?
            ├── NO ──> Append to Structural Tree
            └── YES ─> Stage 2: Local LLM Disambiguation (Ollama)
                       ─────────────────────────────────────────
                       Pass 10-line text window to Qwen2.5
                       (14B or 7B) for fast structured JSON evaluation
                    │
                    ▼
       Stage 3: Leaf-Node Summarization (Ollama)
      ─────────────────────────────────────────
      Pass text from identified page spans to LLM 
      to synthesize 2-3 sentence section summaries
                    │
                    ▼
       [ Final Output: PageIndex Tree JSON ]
```
