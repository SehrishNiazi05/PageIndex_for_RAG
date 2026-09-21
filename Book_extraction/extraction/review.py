"""Build a page-level manual review queue from completed extractions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("output"))
    args = parser.parse_args()
    root = args.output.resolve()
    rows = []
    statuses = Counter()

    for report_path in sorted(root.glob("*/*/pages-*/quality.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads((report_path.parent / "manifest.json").read_text(encoding="utf-8"))
        statuses[report["status"]] += 1
        reasons_by_page = {}
        for flag in report["flagged_pages"]:
            reasons_by_page.setdefault(str(flag["page"]), []).append(flag["reason"])
        for reason in reasons_by_page.get("None", []):
            rows.append({
                "book": Path(manifest["source"]).name,
                "profile": report_path.parent.parent.name,
                "pdf_page": "batch",
                "reasons": reason,
                "source_text_chars": "",
                "extracted_text_chars": "",
                "tables": "",
                "pictures": "",
                "batch_folder": str(report_path.parent),
            })
        for page, record in report["pages"].items():
            reasons = reasons_by_page.get(page, [])
            if record["tables"] and "table_manual_review_required" not in reasons:
                reasons.append("table_spot_check")
            if record["pictures"]:
                reasons.append("figure_spot_check")
            if not reasons:
                continue
            rows.append({
                "book": Path(manifest["source"]).name,
                "profile": report_path.parent.parent.name,
                "pdf_page": page,
                "reasons": ";".join(reasons),
                "source_text_chars": record.get("source_text_chars", ""),
                "extracted_text_chars": record["text_chars"],
                "tables": record["tables"],
                "pictures": record["pictures"],
                "batch_folder": str(report_path.parent),
            })

    root.mkdir(parents=True, exist_ok=True)
    queue = root / "review_queue.csv"
    fields = ["book", "profile", "pdf_page", "reasons", "source_text_chars",
              "extracted_text_chars", "tables", "pictures", "batch_folder"]
    with queue.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Batches: {sum(statuses.values())} ({dict(statuses)})")
    print(f"Review queue: {len(rows)} pages -> {queue}")


if __name__ == "__main__":
    main()
