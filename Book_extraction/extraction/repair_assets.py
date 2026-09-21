"""Repair portable image references in existing batches without rerunning Docling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assets import normalize_references, prepare_references
from extract import Settings, quality_report, sha256_file, source_char_counts, write_json


def repair_batch(folder: Path, dry_run: bool = False) -> dict[str, int]:
    manifest_file = folder / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    checksums = manifest.get("output_sha256", {})
    for name in ("document.json", "document.md"):
        if checksums.get(name) != sha256_file(folder / name):
            raise ValueError(f"Checksum mismatch before repair: {folder / name}")
    if dry_run:
        _, _, counts = prepare_references(folder)
        return counts
    counts = normalize_references(folder)
    document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
    old_quality = json.loads((folder / "quality.json").read_text(encoding="utf-8"))
    raw_settings = manifest["settings"]
    settings = Settings(**{
        **raw_settings,
        "input_dir": Path(raw_settings["input_dir"]),
        "output_dir": Path(raw_settings["output_dir"]),
    })
    start, end = manifest["pdf_page_range"]
    quality = quality_report(
        document, source_char_counts(Path(manifest["source"]), start, end),
        start, end, settings,
    )
    for key in ("figures_detected", "figures_exported", "tables_detected",
                "tables_exported", "image_placeholders", "image_warning",
                "table_warning", "image_placeholder_warning"):
        if key in old_quality:
            quality[key] = old_quality[key]
    if any(key in quality for key in ("image_warning", "table_warning", "image_placeholder_warning")):
        quality["status"] = "review"
    quality["asset_references"] = counts
    manifest["output_sha256"] = {
        name: sha256_file(folder / name) for name in ("document.json", "document.md")
    }
    manifest["asset_references"] = counts
    manifest["status"] = quality["status"]
    write_json(manifest_file, manifest)
    write_json(folder / "quality.json", quality)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument("--book", help="Exact source PDF filename; defaults to every extracted book")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    folders = sorted(args.output.glob("*/*/pages-*/manifest.json"))
    count = 0
    try:
        for manifest_file in folders:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if args.book and Path(manifest["source"]).name.casefold() != args.book.casefold():
                continue
            counts = repair_batch(manifest_file.parent, args.dry_run)
            count += 1
            print(f"{manifest_file.parent}: {counts}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    if not count:
        parser.exit(1, "ERROR: No matching extracted batches found\n")
    print(f"{'Checked' if args.dry_run else 'Repaired'} {count} batches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
