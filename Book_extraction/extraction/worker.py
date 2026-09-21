"""Run one bounded Docling conversion; called by extract.py in a subprocess."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--ocr-mode", choices=["off", "auto", "full_page"], required=True)
    parser.add_argument("--table-mode", choices=["accurate", "fast"], required=True)
    parser.add_argument("--image-scale", type=float, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        HeadingHierarchyOptions,
        OcrMode,
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import ImageRefMode

    options = PdfPipelineOptions(
        do_ocr=args.ocr_mode != "off",
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE if args.table_mode == "accurate" else TableFormerMode.FAST,
        ),
        generate_picture_images=True,
        generate_page_images=False,
        generate_parsed_pages=True,
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
        images_scale=args.image_scale,
        accelerator_options=AcceleratorOptions(num_threads=args.threads),
        document_timeout=args.timeout,
        enable_remote_services=False,
        do_picture_description=False,
    )
    if args.ocr_mode != "off":
        ocr_mode = (
            OcrMode.PDF_AWARE_LAYOUT_REGIONS if args.ocr_mode == "auto"
            else OcrMode.FULL_PAGE
        )
        options.ocr_options = EasyOcrOptions(mode=ocr_mode)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    result = converter.convert(args.pdf, page_range=(args.start, args.end))
    if result.status != ConversionStatus.SUCCESS:
        raise RuntimeError(f"Docling returned {result.status}: {result.errors}")

    args.output.mkdir(parents=True, exist_ok=True)
    document = result.document
    document.save_as_json(args.output / "document.json", image_mode=ImageRefMode.REFERENCED)
    document.save_as_markdown(args.output / "document.md", image_mode=ImageRefMode.REFERENCED)

    figures_dir = args.output / "figures"
    figure_index = []
    for number, picture in enumerate(document.pictures, start=1):
        pages = sorted({prov.page_no for prov in picture.prov})
        filename = None
        try:
            image = picture.get_image(document)
            if image is not None:
                figures_dir.mkdir(exist_ok=True)
                candidate = f"figures/figure_{number:05d}_page_{pages[0] if pages else 0:05d}.png"
                image.save(args.output / candidate)
                filename = candidate
        except (OSError, ValueError, RuntimeError):
            # A missing crop is reported for manual review by the orchestrator.
            pass
        figure_index.append({
            "self_ref": picture.self_ref,
            "pages": pages,
            "file": filename,
        })
    (args.output / "figure_index.json").write_text(
        json.dumps(figure_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tables_dir = args.output / "tables"
    table_index = []
    for number, table in enumerate(document.tables, start=1):
        pages = sorted({prov.page_no for prov in table.prov})
        filename = None
        try:
            html = table.export_to_html(doc=document)
            if html.strip():
                tables_dir.mkdir(exist_ok=True)
                candidate = f"tables/table_{number:05d}_page_{pages[0] if pages else 0:05d}.html"
                (args.output / candidate).write_text(html, encoding="utf-8")
                filename = candidate
        except (OSError, ValueError, RuntimeError):
            pass
        try:
            caption = table.caption_text(document)
        except (OSError, ValueError, RuntimeError):
            caption = ""
        table_index.append({
            "self_ref": table.self_ref,
            "pages": pages,
            "file": filename,
            "caption": caption,
        })
    (args.output / "table_index.json").write_text(
        json.dumps(table_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
