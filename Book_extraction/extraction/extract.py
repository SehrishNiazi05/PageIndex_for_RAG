"""Resumable Docling extraction with source-linked quality reports.

The script launches a bounded Docling worker. It never modifies a source PDF.
Each page batch is written to a temporary directory and published only after
both structured JSON and Markdown have passed basic validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from assets import normalize_references, prepare_references


@dataclass(frozen=True)
class Settings:
    input_dir: Path
    output_dir: Path
    batch_size: int
    threads: int
    document_timeout_seconds: int
    ocr_mode: str
    table_mode: str
    image_scale: float
    min_source_chars: int
    min_retained_fraction: float


def load_settings(path: Path, book_name: str | None = None) -> Settings:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    base = path.resolve().parent
    overrides = raw.get("books", {}).get(book_name, {}) if book_name else {}
    allowed = {"batch_size", "threads", "document_timeout_seconds", "ocr_mode",
               "table_mode", "image_scale", "min_source_chars", "min_retained_fraction"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown settings for {book_name}: {', '.join(sorted(unknown))}")

    def option(key: str) -> Any:
        return overrides.get(key, raw[key])

    def resolved(key: str) -> Path:
        return (base / raw[key]).resolve()

    settings = Settings(
        input_dir=resolved("input_dir"),
        output_dir=resolved("output_dir"),
        batch_size=int(option("batch_size")),
        threads=int(option("threads")),
        document_timeout_seconds=int(option("document_timeout_seconds")),
        ocr_mode=str(option("ocr_mode")),
        table_mode=str(option("table_mode")),
        image_scale=float(option("image_scale")),
        min_source_chars=int(option("min_source_chars")),
        min_retained_fraction=float(option("min_retained_fraction")),
    )
    if settings.batch_size < 1 or settings.threads < 1:
        raise ValueError("batch_size and threads must be positive")
    if settings.document_timeout_seconds < 1:
        raise ValueError("document_timeout_seconds must be positive")
    if settings.ocr_mode not in {"off", "auto", "full_page"}:
        raise ValueError("ocr_mode must be off, auto, or full_page")
    if settings.table_mode not in {"accurate", "fast"}:
        raise ValueError("table_mode must be accurate or fast")
    if not 0.5 <= settings.image_scale <= 4:
        raise ValueError("image_scale must be between 0.5 and 4")
    if settings.min_source_chars < 0 or not 0 <= settings.min_retained_fraction <= 1:
        raise ValueError("invalid quality thresholds")
    return settings


def pdf_page_count(pdf: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(pdf)], capture_output=True, text=True, check=True
    )
    match = re.search(r"^Pages:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not read page count: {pdf}")
    return int(match.group(1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def book_id(pdf: Path) -> str:
    normalized = unicodedata.normalize("NFKD", pdf.stem)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:65]
    name_hash = hashlib.sha256(pdf.name.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'book'}-{name_hash}"


def profile_id(settings: Settings) -> str:
    options = {
        "batch_size": settings.batch_size,
        "threads": settings.threads,
        "ocr_mode": settings.ocr_mode,
        "table_mode": settings.table_mode,
        "image_scale": settings.image_scale,
    }
    fingerprint = hashlib.sha256(
        json.dumps(options, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return f"{settings.ocr_mode}-{fingerprint}"


def docling_command(
    pdf: Path, output: Path, start: int, end: int, settings: Settings
) -> list[str]:
    return [
        sys.executable, str(Path(__file__).with_name("worker.py")),
        "--pdf", str(pdf), "--output", str(output),
        "--start", str(start), "--end", str(end),
        "--ocr-mode", settings.ocr_mode,
        "--table-mode", settings.table_mode,
        "--image-scale", str(settings.image_scale),
        "--threads", str(settings.threads),
        "--timeout", str(settings.document_timeout_seconds),
    ]


def source_char_counts(pdf: Path, start: int, end: int) -> dict[int, int] | None:
    if not shutil.which("pdftotext"):
        return None
    result = subprocess.run(
        ["pdftotext", "-f", str(start), "-l", str(end), "-enc", "UTF-8", str(pdf), "-"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return None
    pages = result.stdout.decode("utf-8", errors="replace").split("\f")
    return {number: len(pages[number - start].strip())
            for number in range(start, end + 1) if number - start < len(pages)}


def provenance_pages(item: dict[str, Any]) -> set[int]:
    pages = set()
    for prov in item.get("prov", []):
        if isinstance(prov, dict) and isinstance(prov.get("page_no"), int):
            pages.add(prov["page_no"])
    return pages


def item_text(item: dict[str, Any], kind: str) -> str:
    if kind == "tables":
        cells = item.get("data", {}).get("table_cells", [])
        if cells:
            return " ".join(str(cell.get("text", "")) for cell in cells)
    return str(item.get("text", ""))


def quality_report(
    document: dict[str, Any], source_counts: dict[int, int] | None,
    start: int, end: int, settings: Settings,
) -> dict[str, Any]:
    counts = {number: {"text_chars": 0, "texts": 0, "tables": 0, "pictures": 0,
                       "encoding_artifacts": 0, "watermark_hits": 0}
              for number in range(start, end + 1)}
    observed = {
        page for kind in ("texts", "tables", "pictures")
        for item in document.get(kind, []) if isinstance(item, dict)
        for page in provenance_pages(item)
    }
    relative_pages = (
        start > 1 and observed and 1 in observed
        and observed.issubset(set(range(1, end - start + 2)))
    )
    unmapped = 0
    empty_formula_pages: set[int] = set()
    for kind in ("texts", "tables", "pictures"):
        for item in document.get(kind, []):
            if not isinstance(item, dict):
                continue
            pages = {page + start - 1 if relative_pages else page
                     for page in provenance_pages(item)}
            if not pages or any(page not in counts for page in pages):
                unmapped += 1
                continue
            for page in pages:
                counts[page][kind] += 1
                content = item_text(item, kind)
                if kind == "texts" and item.get("label") == "formula" and not content.strip():
                    empty_formula_pages.add(page)
                counts[page]["text_chars"] += len(content)
                counts[page]["encoding_artifacts"] += content.count("\ufffd") + content.lower().count("glyph<")
                counts[page]["watermark_hits"] += content.lower().count("t.me/dental_books_lib")

    flagged = []
    if len(document.get("pages", {})) != end - start + 1:
        flagged.append({"page": None, "reason": "page_count_mismatch"})
    if source_counts is None:
        flagged.append({"page": None, "reason": "source_text_comparison_unavailable"})
    for page, record in counts.items():
        if page in empty_formula_pages:
            flagged.append({"page": page, "reason": "empty_formula_text"})
        if record["tables"]:
            flagged.append({"page": page, "reason": "table_manual_review_required"})
        if source_counts is not None:
            record["source_text_chars"] = source_counts.get(page)
            source = source_counts.get(page, 0)
            if source >= settings.min_source_chars and record["text_chars"] < source * settings.min_retained_fraction:
                flagged.append({"page": page, "reason": "low_text_retention"})
            if source < 30 and record["text_chars"] < 30 and record["pictures"]:
                flagged.append({"page": page, "reason": "image_text_review"})
        if record["encoding_artifacts"]:
            flagged.append({"page": page, "reason": "encoding_artifact"})
        if sum(record[kind] for kind in ("texts", "tables", "pictures")) == 0:
            flagged.append({"page": page, "reason": "no_mapped_elements"})

    return {
        "pdf_pages": [start, end],
        "status": "review" if flagged or unmapped else "pass",
        "flagged_pages": flagged,
        "unmapped_elements": unmapped,
        "provenance_page_mode": "relative" if relative_pages else "absolute",
        "element_totals": {kind: sum(page[kind] for page in counts.values())
                           for kind in ("texts", "tables", "pictures")},
        "pages": {str(page): record for page, record in counts.items()},
        "note": "Table pages require manual alignment review. Automated checks do not verify reading order, captions, or image completeness; inspect against the PDF before RAG ingestion.",
    }


def validate_outputs(folder: Path) -> tuple[Path, Path, dict[str, Any]]:
    json_file = folder / "document.json"
    md_file = folder / "document.md"
    if not json_file.exists() or not md_file.exists():
        raise ValueError(f"Expected Docling JSON and Markdown in {folder}")
    document = json.loads(json_file.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("pages"), dict):
        raise ValueError("Docling JSON has no page map")
    if not md_file.read_text(encoding="utf-8").strip():
        raise ValueError("Docling Markdown is empty")
    portable_md, portable_doc, _ = prepare_references(folder)
    if portable_md != md_file.read_text(encoding="utf-8") or portable_doc != document:
        raise ValueError("Image references are not portable; run repair_assets.py")
    return json_file, md_file, document


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_batch(
    pdf: Path, start: int, end: int, settings: Settings,
    source_sha256: str, docling_version: str, dry_run: bool,
) -> str:
    parent = settings.output_dir / book_id(pdf) / profile_id(settings)
    destination = parent / f"pages-{start:05d}-{end:05d}"
    expected = {
        "source": str(pdf.resolve()),
        "source_sha256": source_sha256,
        "pdf_page_range": [start, end],
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in asdict(settings).items()},
        "docling_version": docling_version,
    }
    if destination.exists():
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if all(old.get(key) == value for key, value in expected.items()):
                try:
                    json_file, md_file, _ = validate_outputs(destination)
                    checksums = old.get("output_sha256", {})
                    if all((destination / name).exists() for name in
                           ("quality.json", "figure_index.json", "table_index.json")) and checksums == {
                               "document.json": sha256_file(json_file),
                               "document.md": sha256_file(md_file),
                           }:
                        if dry_run:
                            print(f"SKIP {destination}")
                        return "skipped"
                except ValueError as exc:
                    if "Image references are not portable" in str(exc):
                        raise
                except (OSError, json.JSONDecodeError):
                    pass
        raise FileExistsError(f"Output exists but does not match this run: {destination}")

    if dry_run:
        print(" ".join(docling_command(pdf, destination, start, end, settings)))
        return "planned"

    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".working-", dir=parent) as temporary:
        working = Path(temporary)
        command = docling_command(pdf, working, start, end, settings)
        failed_log = parent / f"pages-{start:05d}-{end:05d}.failed.log"
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False,
                timeout=settings.document_timeout_seconds + 120,
            )
        except subprocess.TimeoutExpired as exc:
            failed_log.write_text(f"TIMEOUT\nCOMMAND: {command!r}\n", encoding="utf-8")
            raise RuntimeError(f"Docling timed out on {pdf.name}, pages {start}-{end}") from exc
        (working / "docling.log").write_text(
            f"COMMAND: {command!r}\nEXIT: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
            encoding="utf-8",
        )
        if result.returncode:
            shutil.copy2(working / "docling.log", failed_log)
            raise RuntimeError(f"Docling failed on {pdf.name}, pages {start}-{end}: {result.stderr[-1500:]}")
        try:
            asset_counts = normalize_references(working)
            json_file, md_file, document = validate_outputs(working)
        except (OSError, ValueError, json.JSONDecodeError):
            shutil.copy2(working / "docling.log", failed_log)
            raise
        report = quality_report(document, source_char_counts(pdf, start, end), start, end, settings)
        figure_index = json.loads((working / "figure_index.json").read_text(encoding="utf-8"))
        table_index = json.loads((working / "table_index.json").read_text(encoding="utf-8"))
        missing_images = [entry for entry in figure_index if not entry["file"]
                          or not (working / entry["file"]).is_file()]
        missing_tables = [entry for entry in table_index if not entry["file"]
                          or not (working / entry["file"]).is_file()]
        report["figures_detected"] = len(figure_index)
        report["figures_exported"] = len(figure_index) - len(missing_images)
        report["tables_detected"] = len(table_index)
        report["tables_exported"] = len(table_index) - len(missing_tables)
        report["asset_references"] = asset_counts
        markdown_text = md_file.read_text(encoding="utf-8")
        report["image_placeholders"] = markdown_text.count("<!-- image -->") + markdown_text.count("Image not available")
        if report["image_placeholders"]:
            report["status"] = "review"
            report["image_placeholder_warning"] = "Markdown contains unavailable image placeholders"
        if missing_images:
            report["status"] = "review"
            report["image_warning"] = f"{len(missing_images)} detected figures had no crop"
        if missing_tables:
            report["status"] = "review"
            report["table_warning"] = f"{len(missing_tables)} detected tables had no HTML export"
        write_json(working / "quality.json", report)
        expected.update({
            "status": report["status"],
            "json_file": json_file.name,
            "markdown_file": md_file.name,
            "output_sha256": {
                "document.json": sha256_file(json_file),
                "document.md": sha256_file(md_file),
            },
            "command": command,
        })
        write_json(working / "manifest.json", expected)
        working.rename(destination)
    return report["status"]


def select_books(settings: Settings, requested: list[str]) -> list[Path]:
    books = sorted(path.resolve() for path in settings.input_dir.glob("*.pdf"))
    if requested:
        wanted = {name.casefold() for name in requested}
        books = [book for book in books if book.name.casefold() in wanted]
        missing = wanted - {book.name.casefold() for book in books}
        if missing:
            raise ValueError(f"Book filename(s) not found: {', '.join(sorted(missing))}")
    if not books:
        raise ValueError(f"No PDF books found in {settings.input_dir}")
    return books


class BookProgress:
    """Display committed PDF pages, including pages in resumed batches."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.terminal = sys.stdout.isatty()
        self.previous_width = 0

    def show(self, label: str) -> None:
        if not self.terminal:
            return
        width = 24
        filled = width * self.completed // self.total
        line = (f"  [{'#' * filled}{'-' * (width - filled)}] "
                f"{self.completed}/{self.total} pages | {label}")
        sys.stdout.write("\r" + line.ljust(self.previous_width))
        sys.stdout.flush()
        self.previous_width = len(line)

    def advance(self, pages: int, label: str) -> None:
        self.completed += pages
        if self.completed > self.total:
            raise ValueError("Progress exceeds selected page count")
        if self.terminal:
            self.show(label)
            if self.completed == self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()
        else:
            width = 24
            filled = width * self.completed // self.total
            print(f"  [{'#' * filled}{'-' * (width - filled)}] "
                  f"{self.completed}/{self.total} pages | {label}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--book", action="append", default=[], help="Exact PDF filename; repeatable")
    parser.add_argument("--start-page", type=int)
    parser.add_argument("--end-page", type=int)
    parser.add_argument("--batch-size", type=int,
                        help="Override config batch size; use 1 for exact page-by-page progress")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without extracting")
    args = parser.parse_args(argv)

    try:
        base_settings = load_settings(args.config)
        if args.batch_size is not None and args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        books = select_books(base_settings, args.book)
        if (args.start_page is not None or args.end_page is not None) and len(books) != 1:
            raise ValueError("Page selection requires exactly one --book")
        if not shutil.which("pdfinfo") or not shutil.which("pdftotext"):
            raise RuntimeError("pdfinfo and pdftotext are required; install Poppler first")
        if not args.dry_run and importlib.util.find_spec("docling") is None:
            raise RuntimeError("Docling is missing; run 'uv sync' in extraction/")
        try:
            version = importlib.metadata.version("docling")
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed" if args.dry_run else "unknown"

        for pdf in books:
            settings = load_settings(args.config, pdf.name)
            if args.batch_size is not None:
                settings = replace(settings, batch_size=args.batch_size)
            total = pdf_page_count(pdf)
            first = args.start_page if args.start_page is not None else 1
            last = args.end_page if args.end_page is not None else total
            if first < 1 or last < first or last > total:
                raise ValueError(f"Invalid page range {first}-{last} for {pdf.name} ({total} pages)")
            digest = sha256_file(pdf)
            print(f"BOOK {pdf.name} | {total} pages | sha256 {digest[:12]}", flush=True)
            progress = BookProgress(last - first + 1)
            for start in range(first, last + 1, settings.batch_size):
                end = min(start + settings.batch_size - 1, last)
                if not args.dry_run:
                    progress.show(f"processing PDF pages {start}-{end}")
                result = extract_batch(pdf, start, end, settings, digest, version, args.dry_run)
                if args.dry_run:
                    print(f"  {start}-{end}: {result}", flush=True)
                else:
                    progress.advance(end - start + 1, f"PDF pages {start}-{end}: {result}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
