#!/usr/bin/env python3
"""
hybrid_index.py

Cost-optimized replacement for batch_index.py.

Core idea:
    PageIndex "flash" mode is cheap/heuristic (little to no LLM cost).
    PageIndex "standard" mode is accurate but pays an LLM call per node —
    this is what cost $10 on a single 231-node book.

    Instead of picking one mode for all 9 books, this script:
      1. Runs "flash" on every book first (near-free).
      2. VALIDATES the resulting tree with cheap, local (non-LLM) heuristics:
         - enough top-level nodes to be a real TOC (not 1 giant blob)
         - page ranges are monotonic / non-overlapping / cover the doc
         - no absurdly long "catch-all" sections
         - titles aren't empty/duplicate junk
      3. Only books that FAIL validation get re-run in "standard" mode —
         so you pay the expensive per-node LLM cost only where flash
         actually failed, not on all 9 books.
      4. Results are cached: a book already indexed + validated is skipped
         on re-run (checkpointing), so a crash or interruption doesn't
         cost you a re-index of already-good books.
      5. API calls get retry/backoff so a transient failure doesn't kill
         the whole batch or force a full re-run.

Usage:
    python3 hybrid_index.py                      # index all PDFs in PDF_DIR
    python3 hybrid_index.py --only a.pdf b.pdf    # only these files
    python3 hybrid_index.py --force-standard      # skip flash, go straight
                                                    # to standard for --only books
    python3 hybrid_index.py --revalidate-only     # don't re-index; just
                                                    # re-run validation + report
                                                    # on existing trees/*.json

Requires the same setup as batch_index.py (PageIndex cloned alongside this
script, .env with DEEPSEEK_API_KEY etc.)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

PDF_DIR = Path(os.getenv("PDF_DIR", "./textbooks")).resolve()
TREE_DIR = Path(os.getenv("TREE_DIR", "./trees")).resolve()
LOGS_DIR = Path(os.getenv("LOG_DIR", "./logs")).resolve()
PAGEINDEX_REPO = Path(__file__).parent / "PageIndex"
RUN_SCRIPT = PAGEINDEX_REPO / "run_pageindex.py"
RESULTS_DIR = PAGEINDEX_REPO / "results"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TEXT_MODEL = os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-v4-flash")

# --- Validation thresholds (tune these to your books) ----------------------
MIN_TOP_NODES = 3                 # a real TOC almost never has fewer than this
MAX_SINGLE_NODE_PAGE_SPAN = 150   # absolute cap: one section spanning >150 pages
                                  # is likely a mis-detected "catch-all" chapter
MAX_SINGLE_NODE_SPAN_RATIO = 0.15  # ...but also flag a node spanning >15% of the
                                   # whole book (books vary 261-1608 pages)
MIN_PAGE_COVERAGE_RATIO = 0.85    # tree should account for most of the PDF's pages
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5


@dataclass
class ValidationResult:
    ok: bool
    reasons: list = field(default_factory=list)  # populated when ok=False


# ---------------------------------------------------------------------------
# Setup checks
# ---------------------------------------------------------------------------

def check_setup():
    problems = []
    if not DEEPSEEK_API_KEY:
        problems.append("DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in.")
    if not RUN_SCRIPT.exists():
        problems.append(
            f"Could not find {RUN_SCRIPT}. Clone PageIndex next to this script:\n"
            f"    git clone https://github.com/VectifyAI/PageIndex.git"
        )
    if not PDF_DIR.exists():
        problems.append(f"PDF_DIR does not exist: {PDF_DIR}.")
    if shutil.which("pdfinfo") is None:
        problems.append(
            "pdfinfo not found on PATH (poppler). It is required for page-count/"
            "coverage validation. Install with: brew install poppler"
        )
    if shutil.which("pdftotext") is None:
        problems.append(
            "pdftotext not found on PATH (poppler). It is required at query time "
            "for page-text extraction. Install with: brew install poppler"
        )
    if problems:
        print("Setup problems found:\n")
        for p in problems:
            print(f"  - {p}\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# PDF page count (for coverage validation) — uses pdfinfo (poppler), no LLM
# ---------------------------------------------------------------------------

def get_pdf_page_count(pdf_path: Path) -> int | None:
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Tree validation — all local/heuristic, zero API cost
# ---------------------------------------------------------------------------

def flatten(nodes):
    flat = []
    for n in nodes:
        flat.append(n)
        children = n.get("nodes", [])
        if children:
            flat.extend(flatten(children))
    return flat


def validate_tree(tree_json: dict, pdf_path: Path) -> ValidationResult:
    reasons = []
    structure = tree_json.get("structure", tree_json) if isinstance(tree_json, dict) else tree_json
    top_nodes = structure if isinstance(structure, list) else structure.get("nodes", [])

    if not top_nodes:
        return ValidationResult(False, ["empty structure — flash mode found no tree at all"])

    if len(top_nodes) < MIN_TOP_NODES:
        reasons.append(f"only {len(top_nodes)} top-level node(s), expected >= {MIN_TOP_NODES}")

    all_nodes = flatten(top_nodes)

    # empty / missing titles
    untitled = [n for n in all_nodes if not (n.get("title") or "").strip()]
    if untitled:
        reasons.append(f"{len(untitled)} node(s) have empty titles")

    # total PDF length is needed for both the adaptive span cap and the
    # coverage check — resolve it once, up front.
    total_pages = get_pdf_page_count(pdf_path)
    if total_pages is None:
        reasons.append("could not read PDF page count (pdfinfo) — coverage/spans unchecked")

    # page span sanity + collect ranges for coverage check.
    # Only LEAF nodes are span-checked: parent nodes (Parts/Chapters) legitimately
    # span hundreds of pages because they contain many child sections, so a wide
    # parent range is expected. A wide LEAF is the "catch-all" mis-detection.
    span_limit = MAX_SINGLE_NODE_PAGE_SPAN
    if total_pages:
        span_limit = max(span_limit, int(total_pages * MAX_SINGLE_NODE_SPAN_RATIO))
    spans = []
    for n in all_nodes:
        s, e = n.get("start_index"), n.get("end_index")
        if s is None or e is None:
            continue
        try:
            s, e = int(s), int(e)
        except (TypeError, ValueError):
            continue
        if e < s:
            reasons.append(f"node '{n.get('title')}' has end_index < start_index")
            continue
        span = e - s + 1
        is_leaf = not n.get("nodes")
        if is_leaf and span > span_limit:
            reasons.append(
                f"leaf node '{n.get('title')}' spans {span} pages (> {span_limit}) "
                f"— likely a mis-detected catch-all section"
            )
        spans.append((s, e))

    # page coverage vs actual PDF length (top-level nodes only, since children
    # nest inside parent ranges)
    if total_pages and top_nodes:
        top_spans = []
        for n in top_nodes:
            s, e = n.get("start_index"), n.get("end_index")
            if s is not None and e is not None:
                try:
                    top_spans.append((int(s), int(e)))
                except (TypeError, ValueError):
                    pass
        if top_spans:
            covered = max(e for _, e in top_spans) - min(s for s, _ in top_spans) + 1
            coverage_ratio = covered / total_pages
            if coverage_ratio < MIN_PAGE_COVERAGE_RATIO:
                reasons.append(
                    f"tree covers only {coverage_ratio:.0%} of the PDF's {total_pages} pages "
                    f"(< {MIN_PAGE_COVERAGE_RATIO:.0%}) — likely missed content at start/end"
                )

    return ValidationResult(ok=(len(reasons) == 0), reasons=reasons)


# ---------------------------------------------------------------------------
# Running PageIndex itself, with retry/backoff
# ---------------------------------------------------------------------------

def write_log(stem: str, text: str):
    """Append diagnostic output for a book to LOGS_DIR/<stem>.log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / f"{stem}.log", "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def run_pageindex(pdf_path: Path, mode: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
    env["OPENAI_BASE_URL"] = DEEPSEEK_BASE_URL

    cmd = [
        sys.executable, str(RUN_SCRIPT),
        "--pdf_path", str(pdf_path),
        "--mode", mode,
        "--index-model", TEXT_MODEL,
    ]

    last_result = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        last_result = subprocess.run(cmd, cwd=str(PAGEINDEX_REPO), env=env,
                                      capture_output=True, text=True)
        if last_result.returncode == 0:
            return last_result
        if attempt < RETRY_ATTEMPTS:
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"    attempt {attempt} failed, retrying in {wait}s...")
            time.sleep(wait)
    return last_result


def locate_output_json(pdf_path: Path) -> Path | None:
    # run_pageindex.py always writes <stem>_structure.json into ./results
    # (its cwd = PAGEINDEX_REPO). Read that exact path instead of globbing,
    # so a stale file from an earlier run can't be picked up by accident.
    src = RESULTS_DIR / f"{pdf_path.stem}_structure.json"
    return src if src.exists() else None


# ---------------------------------------------------------------------------
# Per-book pipeline: flash -> validate -> escalate to standard if needed
# ---------------------------------------------------------------------------

def index_one_pdf(pdf_path: Path, force_standard: bool = False, no_escalate: bool = False) -> dict:
    dst = TREE_DIR / f"{pdf_path.stem}_pageindex.json"
    log_prefix = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {pdf_path.name}\n"

    # Checkpoint: skip books already indexed AND passing validation
    if dst.exists():
        try:
            existing = json.loads(dst.read_text())
            v = validate_tree(existing, pdf_path)
            if v.ok:
                return {"status": "cached_ok", "mode": None, "path": str(dst)}
        except Exception:
            pass  # fall through and re-index if the cached file is corrupt

    # flash (cheap) -> manual (one LLM call) -> standard (opt-in only: expensive)
    modes_to_try = ["standard"] if force_standard else ["flash"]
    if not no_escalate:
        modes_to_try.append("manual")
    last_validation = None

    for mode in modes_to_try:
        print(f"\n=== {pdf_path.name}: trying mode={mode} ===")

        if mode == "manual":
            # In-process manual TOC build (free detection + one LLM call) — no
            # PageIndex subprocess. Used for bookmark-less books where flash
            # returns an empty structure and standard is too expensive.
            try:
                import manual_toc
                from openai import OpenAI
                client = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
                tree_json = manual_toc.build_tree(pdf_path, client)
            except Exception as e:
                write_log(pdf_path.stem, log_prefix + f"mode=manual\nERROR: {e}\n")
                print(f"    manual build FAILED: {e}")
                continue
            TREE_DIR.mkdir(parents=True, exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(tree_json, f, indent=2, ensure_ascii=False)
            validation = validate_tree(tree_json, pdf_path)
            last_validation = validation
            if validation.ok:
                print(f"    PASSED validation on {mode} mode -> {dst}")
                return {"status": "ok", "mode": mode, "path": str(dst)}
            print(f"    Validation FAILED on {mode} mode:")
            for r in validation.reasons:
                print(f"      - {r}")
            continue

        # clear PageIndex's results dir first so locate_output_json can't
        # match a stale file left over from a previous run of this book
        if RESULTS_DIR.exists():
            for stale in RESULTS_DIR.glob("*.json"):
                stale.unlink()

        result = run_pageindex(pdf_path, mode)
        write_log(pdf_path.stem, log_prefix + f"mode={mode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n")
        if result.returncode != 0:
            print(f"    PageIndex run FAILED in {mode} mode")
            print(result.stderr[-1500:])
            continue

        src = locate_output_json(pdf_path)
        if not src:
            print(f"    WARNING: no output JSON found for {mode} mode")
            continue

        tree_json = json.loads(src.read_text())
        toc_source = tree_json.get("toc_source", "?") if isinstance(tree_json, dict) else "?"
        validation = validate_tree(tree_json, pdf_path)
        last_validation = validation
        print(f"    toc_source={toc_source}")

        if validation.ok:
            TREE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"    PASSED validation on {mode} mode -> {dst}")
            return {"status": "ok", "mode": mode, "path": str(dst)}
        else:
            print(f"    Validation FAILED on {mode} mode:")
            for r in validation.reasons:
                print(f"      - {r}")
            if mode == "flash" and "manual" in modes_to_try:
                print("    Escalating to manual mode (one cheap LLM call)...")
            # keep the failed src around under a debug name for inspection
            debug_dst = TREE_DIR / f"{pdf_path.stem}_{mode}_FAILED.json"
            TREE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(debug_dst))

    return {
        "status": "failed",
        "mode": None,
        "path": None,
        "reasons": last_validation.reasons if last_validation else ["no output produced"],
    }


# ---------------------------------------------------------------------------
# Revalidate existing trees without re-indexing (free — pure local check)
# ---------------------------------------------------------------------------

def revalidate_existing():
    for tree_path in sorted(TREE_DIR.glob("*_pageindex.json")):
        stem = tree_path.stem.replace("_pageindex", "")
        pdf_path = PDF_DIR / f"{stem}.pdf"
        tree_json = json.loads(tree_path.read_text())
        v = validate_tree(tree_json, pdf_path)
        status = "OK" if v.ok else "NEEDS REVIEW"
        print(f"{stem}: {status}")
        for r in v.reasons:
            print(f"    - {r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hybrid flash-first, quality-gated PageIndex batch indexer.")
    parser.add_argument("--only", nargs="*", default=None,
                         help="Only these PDF filenames (space separated).")
    parser.add_argument("--force-standard", action="store_true",
                         help="Skip flash/manual, run standard mode directly (for books you already know need it).")
    parser.add_argument("--no-escalate", action="store_true",
                         help="Run flash only; do NOT escalate failed books to manual/standard mode "
                              "(useful to review flash results cheaply before spending more).")
    parser.add_argument("--revalidate-only", action="store_true",
                         help="Don't index anything; just re-check existing trees/*.json against validation rules.")
    args = parser.parse_args()

    if args.revalidate_only:
        revalidate_existing()
        return

    check_setup()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if p.name in args.only]
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s):")
    for p in pdfs:
        print(f"  - {p.name}")

    results = {}
    for pdf_path in tqdm(pdfs, desc="Indexing textbooks"):
        results[pdf_path.name] = index_one_pdf(pdf_path, force_standard=args.force_standard,
                                               no_escalate=args.no_escalate)

    print("\n=== Summary ===")
    cached = [n for n, r in results.items() if r["status"] == "cached_ok"]
    flash_ok = [n for n, r in results.items() if r["status"] == "ok" and r.get("mode") == "flash"]
    manual_ok = [n for n, r in results.items() if r["status"] == "ok" and r.get("mode") == "manual"]
    standard_ok = [n for n, r in results.items() if r["status"] == "ok" and r.get("mode") == "standard"]
    failed = [n for n, r in results.items() if r["status"] == "failed"]

    for name, r in results.items():
        print(f"  {name}: {r['status']} (mode={r.get('mode')})")

    print(f"\n{len(cached)} book(s) skipped (already valid).")
    print(f"{len(flash_ok)} book(s) done cheaply on flash mode.")
    print(f"{len(manual_ok)} book(s) done via manual TOC (one LLM call): {manual_ok}")
    print(f"{len(standard_ok)} book(s) needed standard mode (higher cost): {standard_ok}")
    if failed:
        print(f"{len(failed)} book(s) FAILED — check logs/*.log and trees/*_FAILED.json, then re-run: {failed}")


if __name__ == "__main__":
    main()