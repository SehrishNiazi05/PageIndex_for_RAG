import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from assets import normalize_references, prepare_references
from extract import Settings, book_id, docling_command, extract_batch, load_settings, main, quality_report, sha256_file
from repair_assets import repair_batch


class ExtractionConfigurationTests(unittest.TestCase):
    def settings(self) -> Settings:
        return Settings(
            input_dir=Path("/tmp/books"), output_dir=Path("/tmp/out"),
            batch_size=40, threads=4, document_timeout_seconds=1800,
            ocr_mode="off", table_mode="accurate", image_scale=2.0,
            min_source_chars=300, min_retained_fraction=0.45,
        )

    def test_command_keeps_exact_page_range_and_modes(self) -> None:
        command = docling_command(Path("/tmp/books/Book.pdf"), Path("/tmp/out"), 41, 80, self.settings())
        self.assertEqual(command[command.index("--start") + 1], "41")
        self.assertEqual(command[command.index("--end") + 1], "80")
        self.assertEqual(command[command.index("--ocr-mode") + 1], "off")
        self.assertEqual(command[command.index("--table-mode") + 1], "accurate")

    def test_quality_flags_missing_text_and_tracks_tables(self) -> None:
        document = {
            "texts": [{"text": "x" * 30, "prov": [{"page_no": 1}]}],
            "tables": [{"data": {"table_cells": [{"text": "dose 5 mg"}]}, "prov": [{"page_no": 2}]}],
            "pictures": [],
        }
        report = quality_report(document, {1: 500, 2: 20}, 1, 2, self.settings())
        self.assertEqual(report["status"], "review")
        self.assertIn({"page": 1, "reason": "low_text_retention"}, report["flagged_pages"])
        self.assertEqual(report["pages"]["2"]["tables"], 1)
        self.assertIn({"page": 2, "reason": "table_manual_review_required"}, report["flagged_pages"])

    def test_empty_formula_is_flagged(self) -> None:
        document = {"pages": {"1": {}}, "texts": [
            {"label": "formula", "text": "", "orig": "x = 2", "prov": [{"page_no": 1}]}
        ], "tables": [], "pictures": []}
        report = quality_report(document, {1: 5}, 1, 1, self.settings())
        self.assertIn({"page": 1, "reason": "empty_formula_text"}, report["flagged_pages"])

    def test_book_override_changes_only_selected_book(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.toml"
            config.write_text('''input_dir = "books"
output_dir = "output"
batch_size = 40
threads = 4
document_timeout_seconds = 1800
ocr_mode = "off"
table_mode = "accurate"
image_scale = 2.0
min_source_chars = 300
min_retained_fraction = 0.45
[books."Scan.pdf"]
ocr_mode = "auto"
batch_size = 20
''', encoding="utf-8")
            self.assertEqual(load_settings(config, "Scan.pdf").ocr_mode, "auto")
            self.assertEqual(load_settings(config, "Scan.pdf").batch_size, 20)
            self.assertEqual(load_settings(config, "Other.pdf").ocr_mode, "off")
            self.assertEqual(load_settings(config, "Other.pdf").batch_size, 40)

    def test_image_references_become_portable_and_missing_targets_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            assets = folder / "document_artifacts"
            assets.mkdir()
            (assets / "image 1.png").write_bytes(b"image")
            old = "/old/.working-123/document_artifacts/image 1.png"
            (folder / "document.md").write_text(
                "![Image](/old/.working-123/document_artifacts/image%201.png)", encoding="utf-8")
            (folder / "document.json").write_text(json.dumps({
                "pages": {"1": {}}, "pictures": [{"image": {"uri": old}}]
            }), encoding="utf-8")
            self.assertEqual(normalize_references(folder),
                             {"markdown_images": 1, "json_images": 1})
            self.assertIn("document_artifacts/image%201.png",
                          (folder / "document.md").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((folder / "document.json").read_text())
                             ["pictures"][0]["image"]["uri"],
                             "document_artifacts/image 1.png")
            (assets / "image 1.png").unlink()
            with self.assertRaisesRegex(ValueError, "Broken image references"):
                prepare_references(folder)

    def test_existing_batch_repair_updates_checksums_and_review_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "document_artifacts").mkdir()
            (folder / "document_artifacts" / "image.png").write_bytes(b"image")
            (folder / "document.md").write_text(
                "![Image](/old/.working-1/document_artifacts/image.png)", encoding="utf-8")
            document = {"pages": {"1": {}}, "texts": [],
                        "tables": [{"data": {"table_cells": [{"text": "A"}]},
                                    "prov": [{"page_no": 1}]}],
                        "pictures": [{"image": {"uri":
                            "/old/.working-1/document_artifacts/image.png"},
                            "prov": [{"page_no": 1}]}]}
            (folder / "document.json").write_text(json.dumps(document), encoding="utf-8")
            settings = self.settings()
            manifest = {"source": str(folder / "Book.pdf"), "pdf_page_range": [1, 1],
                        "settings": {**settings.__dict__, "input_dir": str(settings.input_dir),
                                     "output_dir": str(settings.output_dir)},
                        "output_sha256": {name: sha256_file(folder / name)
                                          for name in ("document.json", "document.md")}}
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (folder / "quality.json").write_text(json.dumps({}), encoding="utf-8")
            with patch("repair_assets.source_char_counts", return_value={1: 1}):
                repair_batch(folder)
            repaired = json.loads((folder / "manifest.json").read_text())
            report = json.loads((folder / "quality.json").read_text())
            self.assertEqual(repaired["output_sha256"]["document.md"],
                             sha256_file(folder / "document.md"))
            self.assertEqual(repaired["status"], "review")
            self.assertIn({"page": 1, "reason": "table_manual_review_required"},
                          report["flagged_pages"])

    def test_relative_page_provenance_is_mapped(self) -> None:
        document = {"texts": [{"text": "content", "prov": [{"page_no": 1}]}],
                    "tables": [], "pictures": []}
        report = quality_report(document, {41: 7}, 41, 41, self.settings())
        self.assertEqual(report["provenance_page_mode"], "relative")
        self.assertEqual(report["pages"]["41"]["texts"], 1)

    def test_book_id_is_stable_and_safe(self) -> None:
        self.assertEqual(book_id(Path("McCracken’s.pdf")), book_id(Path("McCracken’s.pdf")))
        self.assertNotIn("’", book_id(Path("McCracken’s.pdf")))

    def test_successful_batch_is_published_and_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "Book.pdf"
            pdf.write_bytes(b"test source")
            settings = self.settings()
            settings = Settings(**{**settings.__dict__, "output_dir": root / "output"})

            def fake_worker(command, **_kwargs):
                output = Path(command[command.index("--output") + 1])
                (output / "document.json").write_text(json.dumps({
                    "pages": {"1": {}},
                    "texts": [{"text": "test content", "prov": [{"page_no": 1}]}],
                    "tables": [], "pictures": [],
                }), encoding="utf-8")
                (output / "document.md").write_text("test content", encoding="utf-8")
                (output / "figure_index.json").write_text("[]", encoding="utf-8")
                (output / "table_index.json").write_text("[]", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with patch("extract.subprocess.run", side_effect=fake_worker) as run, \
                 patch("extract.source_char_counts", return_value={1: 12}):
                result = extract_batch(pdf, 1, 1, settings, sha256_file(pdf), "mock", False)
                resumed = extract_batch(pdf, 1, 1, settings, sha256_file(pdf), "mock", False)
            self.assertEqual((result, resumed), ("pass", "skipped"))
            self.assertEqual(run.call_count, 1)

    def test_single_page_progress_covers_every_page_of_one_book(self) -> None:
        book = Path("/tmp/books/Book.pdf")
        output = io.StringIO()
        with patch("extract.load_settings", return_value=self.settings()), \
             patch("extract.select_books", return_value=[book]), \
             patch("extract.shutil.which", return_value="/usr/bin/tool"), \
             patch("extract.importlib.util.find_spec", return_value=object()), \
             patch("extract.importlib.metadata.version", return_value="test"), \
             patch("extract.pdf_page_count", return_value=3), \
             patch("extract.sha256_file", return_value="abc123"), \
             patch("extract.extract_batch", side_effect=["pass", "skipped", "pass"]) as batches, \
             patch("sys.stdout", output):
            result = main(["--book", "Book.pdf", "--batch-size", "1"])
        self.assertEqual(result, 0)
        self.assertEqual([(call.args[1], call.args[2]) for call in batches.call_args_list],
                         [(1, 1), (2, 2), (3, 3)])
        self.assertTrue(all(call.args[3].batch_size == 1 for call in batches.call_args_list))
        self.assertIn("3/3 pages", output.getvalue())
        self.assertIn("2/3 pages | PDF pages 2-2: skipped", output.getvalue())


if __name__ == "__main__":
    unittest.main()
