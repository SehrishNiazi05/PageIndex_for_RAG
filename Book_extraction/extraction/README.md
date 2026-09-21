# Dental textbook extraction

This folder contains a local Docling pipeline for the ten PDFs in `../textbooks`.
It extracts structured text, tables, and detected figures in batches while
keeping PDF page numbers and source hashes. Source PDFs are read only. Nothing
is uploaded by this pipeline.

## Setup

From this folder in your own terminal:

```bash
cd extraction
uv sync
```

Docling downloads its layout and table models on first use. `pdfinfo` and
`pdftotext` from Poppler are needed for PDF page counting and source-text
comparison. On macOS, install Poppler with `brew install poppler` if absent.
The `.python-version` file asks uv for Python 3.12.

## Run

Inspect and adjust `config.toml`, then start with a small book or a few pages:

```bash
uv run python extract.py --book "Oral Medicine and Radiology.pdf" --start-page 20 --end-page 25
```

To process the entire collection:

```bash
uv run python extract.py
```

To completely extract just one book (all its PDF pages):

```bash
uv run python extract.py --book "Oral Medicine and Radiology.pdf"
```

The terminal shows a separate progress bar for each book. It counts pages
whose output was saved and validated, including previously completed batches
when resuming. With the default 40-page batches, the bar advances by up to 40
pages after each batch finishes and shows the current PDF page range while it
runs. Docling does not provide a page-completion callback within a batch.

For **exact one-page-at-a-time updates** on a single book, run:

```bash
uv run python extract.py --book "Oral Medicine and Radiology.pdf" --batch-size 1
```

One-page batches take longer and may split tables or headings across page
boundaries. The batch-size override creates a separate output profile, so it
does not mix results with the default 40-page run. Use the default command for
the main extraction when continuity across pages matters.

To see the planned worker commands without parsing:

```bash
uv run python extract.py --book "Oral Medicine and Radiology.pdf" --start-page 20 --end-page 25 --dry-run
```

Completed matching batches are skipped, so rerun the same command to resume.
To run a different configuration, edit `config.toml`: the output profile name
changes for OCR mode, table mode, image scale, threads, or batch size. If only
quality thresholds or input/output paths change, select a new output directory
instead of overwriting existing results. Failed jobs leave a `.failed.log` in
the book profile directory.

### Existing outputs with broken image links

Older batches may contain image links to a removed `.working-*` directory.
Repair them without running Docling again:

```bash
uv run python repair_assets.py --book "Oral Medicine and Radiology.pdf"
```

The repair checks the saved files against their manifest checksums, converts
image references in JSON and Markdown to paths relative to each batch, checks
that each image exists, and updates the checksums. For a read-only preview add
`--dry-run`. New batches are checked and normalized before publication.

### Different textbook layouts

The defaults are a general starting profile, not a guarantee that every book
will parse equally well. Docling detects layout and tables per page, but books
can differ in column order, scanned versus native text, tables, equations, and
figure design. Add exact-filename overrides in `config.toml` when inspection
shows a particular book needs different OCR mode, table mode, image scale,
batch size, threads, or timeout:

```toml
[books."Example scanned textbook.pdf"]
ocr_mode = "auto"
batch_size = 20
```

The command-line `--batch-size` takes precedence over a book override. Review
a sample of prose, multi-column pages, tables, equations, and figures from each
book before indexing. Smaller batches limit memory but can cut across tables
or headings; OCR should be enabled only for pages or books that need it.

The default `ocr_mode = "off"` uses the native text already present in these
books. For image-only or badly encoded pages, run a **separate** pass with
`ocr_mode = "auto"`; use `full_page` only after reviewing a page that needs it.
An OCR pass is not an automatic correction of the original extraction.

## Output

Each `output/<book-id>/<profile>/pages-xxxxx-yyyyy/` folder contains:

| File | Purpose |
| --- | --- |
| `document.json` | Canonical Docling structure, including element labels, table cells, and page/bounding-box provenance |
| `document.md` | Reading-order Markdown for inspection and later chunking |
| `tables/*.html`, `table_index.json` | Individually exported tables with source page and Docling reference |
| `figures/*.png`, `figure_index.json` | Detected figure crops with source page and Docling reference |
| `quality.json` | Page-level source/extracted text counts and review flags |
| `manifest.json` | PDF hash, exact configuration, Docling version, page range, and worker command |
| `docling.log` | Worker stdout and stderr |

Docling may also create a `document_artifacts/` folder for images referenced
from Markdown and JSON. Links in these exports are relative to the batch
folder; keep the folder beside those files so links resolve.

After processing, make a review queue:

```bash
uv run python review.py
```

`output/review_queue.csv` lists flagged pages plus every page containing a
detected table or picture. Open each listed PDF page beside its `document.md`,
table HTML, and figure PNG. Specifically check column reading order, table
row/column alignment, captions, medication doses, measurements, and figure
labels. Table pages are marked for manual review because a populated table
can still have shifted or merged cells. The automatic `pass` status only
checks basic completeness; it does not certify clinical accuracy. Only
reviewed output should enter the RAG index.

The pipeline deliberately retains raw extraction. It does not silently rewrite
medical terms, remove repeated headers/watermarks, or generate figure
descriptions. Those operations need book-specific review because they can
remove or alter clinical content. A figure PNG by itself is not searchable
text; a separate, reviewed image-description stage is needed if your agent
must answer from diagrams or radiographs.

## Limits and recovery

The 40-page batch size limits memory use on the largest books. A heading or
table can cross a batch boundary, so check the first and last page of each
batch before making RAG chunks. PDF page numbers, rather than printed page
numbers, are recorded; printed numbering often differs because of front
matter. Some figures are vector artwork or complex composites and may be
missed or cropped imperfectly. Compare figure-heavy pages with the original
PDF and treat missing crops as a review item.
