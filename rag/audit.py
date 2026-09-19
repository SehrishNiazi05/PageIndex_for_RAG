"""Read-only checks of the nine processed books. Run: python -m rag.audit."""
import json
import re
from collections import Counter
from pathlib import Path
from .config import PROCESSED, DATA
from .chunker import records_with_pages

REF = re.compile(r"\b(reference|references|bibliography)\b", re.I)


def audit_book(path: Path):
    counts = Counter()
    headings = Counter()
    oversize = []
    examples = []
    ref_chars = 0
    body_chars = 0
    last_page = None
    page_backtracks = 0
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        counts[r["label"]] += 1
        text = r.get("text", "")
        if len(text) > 2000 and r["label"] != "section_header":
            counts["oversized_records"] += 1
        if r["label"] == "section_header":
            headings[text.casefold()] += 1
        else:
            body_chars += len(text)
            if REF.search(r.get("heading_path_str", "")):
                ref_chars += len(text)
            if len(text) > 2000 and len(oversize) < 25:
                oversize.append({"source_id": r["id"], "characters": len(text), "label": r["label"], "pdf_page": r.get("page_no")})
            if r["label"] == "table" and len(examples) < 3:
                examples.append({"source_id": r["id"], "pdf_page": r.get("page_no"), "preview": text[:450]})
            if r["label"] == "table" and ("|" not in text or "---" not in text):
                counts["tables_without_markdown_grid"] += 1
        page = r.get("page_no")
        if page is None:
            counts["missing_page"] += 1
        elif last_page is not None and page < last_page:
            page_backtracks += 1
        if page is not None:
            last_page = page
        if r["label"] == "list" and "\n" in text:
            counts["merged_lists"] += 1
    cross_page_items = 0
    first_pages = {}
    for r in records_with_pages(path):
        base = r.get("merged_source_id")
        if base:
            if base not in first_pages: first_pages[base] = r["page_no"]
            elif first_pages[base] != r["page_no"]: cross_page_items += 1
    return {"book": path.stem, "records": sum(counts.get(k,0) for k in ("section_header","text","list","table","caption","footnote","formula")),
            "labels": dict(counts), "oversized_samples": oversize,
            "repeated_headings": headings.most_common(10), "page_backtracks": page_backtracks,
            "recovered_cross_page_list_items": cross_page_items,
            "reference_path_body_fraction": round(ref_chars / max(body_chars, 1), 4),
            "sample_tables": examples}


def run():
    report = {"books": [audit_book(p) for p in sorted(PROCESSED.glob("*.jsonl"))]}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "corpus_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    for b in run()["books"]:
        print(f"{b['book']}: {b['records']} records, reference-path {b['reference_path_body_fraction']:.1%}, "
              f"cross-page list items {b['recovered_cross_page_list_items']}, oversized records {b['labels'].get('oversized_records',0)}")
