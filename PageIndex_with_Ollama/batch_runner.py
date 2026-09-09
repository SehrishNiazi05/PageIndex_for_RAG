"""
Batch runner: index every PDF inside the textbooks/ folder sequentially.
"""

import sys
import time
from pathlib import Path

from tree_builder import build, MODEL_NAME


def _pdf_dirs():
    for d in ("textbooks", "data/input_textbooks"):
        p = Path(d)
        if p.is_dir():
            yield p


def main() -> None:
    limit_pages = int(sys.argv[1]) if len(sys.argv) > 1 else None
    no_llm = "--no-llm" in sys.argv
    out_dir = Path("extracted_trees")

    pdfs: list[Path] = []
    for d in _pdf_dirs():
        pdfs.extend(sorted(d.glob("*.pdf")))
    pdfs = sorted(set(pdfs), key=lambda p: p.name)

    if not pdfs:
        print("No PDFs found in textbooks/ or data/input_textbooks/")
        return

    print(f"Found {len(pdfs)} book(s). Starting batch run...")
    t_start = time.time()

    for i, pdf in enumerate(pdfs, 1):
        print(f"\n=== [{i}/{len(pdfs)}] {pdf.name} ===")
        out = out_dir / (pdf.stem + "_pageindex.json")
        try:
            build(
                str(pdf),
                out_path=str(out),
                model=MODEL_NAME,
                no_llm=no_llm,
                quiet=False,
                limit_pages=limit_pages,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"      [ERROR] {exc}")

    print(f"\nBatch complete in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
