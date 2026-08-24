#!/usr/bin/env python3
"""
batch_index.py

Runs PageIndex tree-structure generation on every PDF in PDF_DIR and
stores the resulting JSON tree files in TREE_DIR.

Usage:
    python3 batch_index.py
    python3 batch_index.py --mode standard   # higher quality, slower/pricier
    python3 batch_index.py --only carranza.pdf shafer.pdf   # index specific books only

Requires:
    - The PageIndex repo cloned next to this script (see setup.sh / README),
      i.e. ./PageIndex/run_pageindex.py must exist.
    - .env configured with DEEPSEEK_API_KEY etc. (see .env.example)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

PDF_DIR = Path(os.getenv("PDF_DIR", "./textbooks")).resolve()
TREE_DIR = Path(os.getenv("TREE_DIR", "./trees")).resolve()
PAGEINDEX_REPO = Path(__file__).parent / "PageIndex"
RUN_SCRIPT = PAGEINDEX_REPO / "run_pageindex.py"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TEXT_MODEL = os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-v4-flash")


def check_setup():
    problems = []
    if not DEEPSEEK_API_KEY:
        problems.append("DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in.")
    if not RUN_SCRIPT.exists():
        problems.append(
            f"Could not find {RUN_SCRIPT}. Run setup.sh first, or:\n"
            f"    git clone https://github.com/VectifyAI/PageIndex.git\n"
            f"    (cloned inside this project folder as ./PageIndex)"
        )
    if not PDF_DIR.exists():
        problems.append(f"PDF_DIR does not exist: {PDF_DIR}. Create it and put your 9 textbook PDFs inside.")
    if problems:
        print("Setup problems found:\n")
        for p in problems:
            print(f"  - {p}\n")
        sys.exit(1)


def index_one_pdf(pdf_path: Path, mode: str) -> Path | None:
    """Runs PageIndex's run_pageindex.py on a single PDF and moves the
    resulting JSON into TREE_DIR. Returns the final JSON path, or None on failure."""

    print(f"\n=== Indexing: {pdf_path.name} (mode={mode}) ===")

    env = os.environ.copy()
    # PageIndex's local runner reads OPENAI_API_KEY / OPENAI_BASE_URL style vars
    # via LiteLLM under the hood — point them at DeepSeek's OpenAI-compatible endpoint.
    env["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
    env["OPENAI_BASE_URL"] = DEEPSEEK_BASE_URL

    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--pdf_path", str(pdf_path),
        "--mode", mode,
        "--index-model", TEXT_MODEL,
    ]

    result = subprocess.run(cmd, cwd=str(PAGEINDEX_REPO), env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAILED: {pdf_path.name}")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return None

    # run_pageindex.py writes <pdf_stem>_structure.json under PageIndex/results/
    # (its cwd) — search for it, then move it into TREE_DIR as <stem>_pageindex.json.
    structure_name = f"{pdf_path.stem}_structure.json"
    candidates = (
        list((PAGEINDEX_REPO / "results").glob(structure_name))
        + list(PAGEINDEX_REPO.glob(f"**/{structure_name}"))
        + list(pdf_path.parent.glob(structure_name))
    )

    if not candidates:
        print(f"  WARNING: could not locate output JSON for {pdf_path.name}. Check PageIndex output location.")
        return None

    src = candidates[0]
    TREE_DIR.mkdir(parents=True, exist_ok=True)
    dst = TREE_DIR / f"{pdf_path.stem}_pageindex.json"
    shutil.move(str(src), str(dst))
    print(f"  OK -> {dst}")
    return dst


def main():
    parser = argparse.ArgumentParser(description="Batch-generate PageIndex trees for all textbooks.")
    parser.add_argument("--mode", default="flash", choices=["flash", "standard"],
                         help="flash = fast heuristic (default). standard = LLM-verified structure, higher quality/cost.")
    parser.add_argument("--only", nargs="*", default=None,
                         help="Only index these specific PDF filenames (space separated), skip the rest.")
    args = parser.parse_args()

    check_setup()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if p.name in args.only]

    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) to index:")
    for p in pdfs:
        print(f"  - {p.name}")

    results = {}
    for pdf_path in tqdm(pdfs, desc="Indexing textbooks"):
        out = index_one_pdf(pdf_path, args.mode)
        results[pdf_path.name] = str(out) if out else "FAILED"

    print("\n=== Summary ===")
    for name, status in results.items():
        print(f"  {name}: {status}")

    failed = [n for n, s in results.items() if s == "FAILED"]
    if failed:
        print(f"\n{len(failed)} book(s) failed to index. Re-run with --only to retry them.")
        sys.exit(1)
    else:
        print(f"\nAll {len(pdfs)} textbooks indexed successfully. Trees saved in {TREE_DIR}")


if __name__ == "__main__":
    main()
