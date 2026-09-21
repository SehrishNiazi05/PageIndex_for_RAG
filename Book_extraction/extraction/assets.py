"""Make Docling image references portable and verify their targets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


IMAGE_LINK = re.compile(r"(!\[[^\]]*\]\()([^)]*)(\))")
ASSET_DIRS = {"document_artifacts", "figures"}


def asset_relative_path(uri: str) -> Path | None:
    """Locate a local exported asset, including one under an old working path."""
    decoded = unquote(uri)
    if urlsplit(decoded).scheme in {"http", "https", "data"}:
        return None
    parts = Path(decoded).parts
    for index, part in enumerate(parts):
        if part in ASSET_DIRS and index + 1 < len(parts):
            remainder = parts[index:]
            if ".." not in remainder:
                return Path(*remainder)
    return None


def prepare_references(folder: Path) -> tuple[str, dict, dict[str, int]]:
    """Return portable exports after checking every local image target exists."""
    md_file = folder / "document.md"
    json_file = folder / "document.json"
    markdown = md_file.read_text(encoding="utf-8")
    document = json.loads(json_file.read_text(encoding="utf-8"))
    counts = {"markdown_images": 0, "json_images": 0}
    missing: list[str] = []

    def target(uri: str, kind: str) -> Path | None:
        relative = asset_relative_path(uri)
        if relative is None:
            if urlsplit(uri).scheme not in {"http", "https", "data"}:
                missing.append(f"{kind}: unsupported local image path {uri}")
            return None
        if not (folder / relative).is_file():
            missing.append(f"{kind}: missing {relative}")
        return relative

    def image_link(match: re.Match[str]) -> str:
        counts["markdown_images"] += 1
        relative = target(match.group(2), "Markdown")
        uri = quote(relative.as_posix(), safe="/") if relative else match.group(2)
        return match.group(1) + uri + match.group(3)

    markdown = IMAGE_LINK.sub(image_link, markdown)

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "uri" and isinstance(value, str) and "document_artifacts" in value:
                    counts["json_images"] += 1
                    relative = target(value, "JSON")
                    if relative:
                        node[key] = relative.as_posix()
                else:
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(document)
    if missing:
        raise ValueError("Broken image references: " + "; ".join(missing[:5]) +
                         (f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""))
    return markdown, document, counts


def normalize_references(folder: Path) -> dict[str, int]:
    markdown, document, counts = prepare_references(folder)
    md_file = folder / "document.md"
    json_file = folder / "document.json"
    if markdown != md_file.read_text(encoding="utf-8"):
        md_file.write_text(markdown, encoding="utf-8")
    original = json.loads(json_file.read_text(encoding="utf-8"))
    if document != original:
        json_file.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    return counts
