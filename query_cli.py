#!/usr/bin/env python3
"""
query_cli.py

Interactive agentic RAG query loop over your 9 indexed dental textbooks.

Pipeline per question:
  1. Book selection   -- cheap text model picks which book(s) are relevant
                          from short book descriptions.
  2. Tree search       -- cheap text model reasons over that book's tree
                          (titles + summaries) and picks the best node(s).
  3. Text extraction   -- the chosen node's page range is read as plain text
                          (no images, charts, or diagrams).
  4. Text answer       -- a DeepSeek text model answers the question grounded
                          in that extracted page text.

Usage:
    python3 query_cli.py
    python3 query_cli.py --summary-only   # text-only mode (cheaper, answers
                                          # from node summaries only, no PDF text)
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PDF_DIR = Path(os.getenv("PDF_DIR", "./textbooks")).resolve()
TREE_DIR = Path(os.getenv("TREE_DIR", "./trees")).resolve()

TEXT_MODEL = os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-v4-flash")

MAX_PAGES_PER_QUERY = int(os.getenv("MAX_PAGES_PER_QUERY", "4"))

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)


# ---------------------------------------------------------------------------
# Load all book trees + build short descriptions for book-level routing
# ---------------------------------------------------------------------------

def load_registry():
    registry = {}
    for tree_path in sorted(TREE_DIR.glob("*_pageindex.json")):
        book_key = tree_path.stem.replace("_pageindex", "")
        with open(tree_path, "r") as f:
            tree = json.load(f)

        # PageIndex trees may be a dict with "structure"/"nodes" or a list — handle both.
        nodes = tree.get("structure", tree) if isinstance(tree, dict) else tree
        if isinstance(nodes, dict) and "nodes" in nodes:
            top_nodes = nodes["nodes"]
        elif isinstance(nodes, list):
            top_nodes = nodes
        else:
            top_nodes = []

        description = tree.get("doc_description") if isinstance(tree, dict) else None
        if not description and top_nodes:
            # fall back: concatenate first few top-level titles as a rough description
            titles = [n.get("title", "") for n in top_nodes[:5]]
            description = "Chapters include: " + "; ".join(t for t in titles if t)

        pdf_path = PDF_DIR / f"{book_key}.pdf"

        registry[book_key] = {
            "tree_path": tree_path,
            "pdf_path": pdf_path,
            "top_nodes": top_nodes,
            "description": description or book_key,
        }
    return registry


def _to_int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def flatten_nodes(nodes, path=""):
    """Flatten a nested tree into a flat list of (path, node) for reasoning."""
    flat = []
    for n in nodes:
        label = f"{path} > {n.get('title', '')}" if path else n.get("title", "")
        flat.append({
            "node_id": n.get("node_id"),
            "title": label,
            "summary": n.get("summary", ""),
            "start_index": _to_int(n.get("start_index")),
            "end_index": _to_int(n.get("end_index")),
        })
        children = n.get("nodes", [])
        if children:
            flat.extend(flatten_nodes(children, label))
    return flat


# ---------------------------------------------------------------------------
# Step 1: pick relevant book(s)
# ---------------------------------------------------------------------------

def pick_books(question, registry, top_k=2):
    listing = "\n".join(f'- "{k}": {v["description"][:300]}' for k, v in registry.items())
    prompt = f"""You are routing a dental clinical question to the right textbook(s).

Available textbooks:
{listing}

Question: {question}

Return ONLY a JSON list of the {top_k} most relevant textbook key(s) (the quoted keys above),
ordered by relevance. Example: ["shafer", "scully"]"""

    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"} if False else None,
    )
    text = resp.choices[0].message.content.strip()
    try:
        start, end = text.index("["), text.rindex("]") + 1
        keys = json.loads(text[start:end])
        return [k for k in keys if k in registry][:top_k]
    except Exception:
        # fallback: just search everything if parsing fails
        return list(registry.keys())


# ---------------------------------------------------------------------------
# Step 2: tree search within a book
# ---------------------------------------------------------------------------

def pick_nodes(question, book_key, registry, top_k=2):
    flat = flatten_nodes(registry[book_key]["top_nodes"])
    listing = "\n".join(
        f'- node_id={n["node_id"]} | {n["title"]} | pages {n["start_index"]}-{n["end_index"]} | {n["summary"][:200]}'
        for n in flat
    )
    prompt = f"""You are searching the tree index of "{book_key}" to answer a dental question.

Tree nodes:
{listing}

Question: {question}

Return ONLY a JSON list of the {top_k} most relevant node_id value(s), ordered by relevance.
Example: ["0007", "0012"]"""

    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    try:
        start, end = text.index("["), text.rindex("]") + 1
        ids = json.loads(text[start:end])
    except Exception:
        ids = [n["node_id"] for n in flat[:top_k]]

    return [n for n in flat if n["node_id"] in ids]


# ---------------------------------------------------------------------------
# Step 3+4: extract page text, ask a DeepSeek text model
# ---------------------------------------------------------------------------

def extract_text(pdf_path, start_page, end_page, max_pages=MAX_PAGES_PER_QUERY):
    """Extract plain text (no images/charts/diagrams) from a PDF page range
    using poppler's pdftotext. Page numbers are 1-based, matching the tree's
    start_index/end_index fields."""
    end_page = min(end_page, start_page + max_pages - 1)  # cap pages per query
    if end_page < start_page:
        return ""

    cmd = [
        "pdftotext",
        "-layout",
        "-enc", "UTF-8",
        "-f", str(start_page),
        "-l", str(end_page),
        str(pdf_path),
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def answer_with_pages(question, book_key, node, registry):
    pdf_path = registry[book_key]["pdf_path"]
    if not pdf_path.exists():
        return f"[Cannot read pages: {pdf_path.name} not found in {PDF_DIR}]"

    start = node["start_index"] or 1
    end = node["end_index"] or start
    page_text = extract_text(pdf_path, start, end)

    if not page_text:
        return (f"[No extractable text on pages {start}-{end} of '{book_key}'. "
                f"These pages may be scanned/image-only.]")

    prompt = (
        f"You are a dental clinical reference assistant. Answer the question strictly "
        f"based on the extracted text below from '{book_key}' (pages {start}-{end}, "
        f"section: {node['title']}). Rely only on the running prose — ignore any "
        f"table, chart, or figure content. If the text doesn't contain the answer, say so.\n\n"
        f"--- Extracted text ---\n{page_text}\n--- end text ---\n\n"
        f"Question: {question}"
    )

    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def answer_with_summary_only(question, book_key, node):
    prompt = f"""Answer this dental question using ONLY the section summary below.
If the summary is insufficient, say what's missing.

Book: {book_key}
Section: {node['title']} (pages {node['start_index']}-{node['end_index']})
Summary: {node['summary']}

Question: {question}"""
    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true",
                         help="Skip PDF text extraction; answer from node summaries only (cheaper).")
    args = parser.parse_args()

    registry = load_registry()
    if not registry:
        print(f"No tree files found in {TREE_DIR}. Run batch_index.py first.")
        return

    print(f"Loaded {len(registry)} textbook(s): {', '.join(registry.keys())}")
    print("Type your dental question, or 'quit' to exit.\n")

    while True:
        question = input("Q> ").strip()
        if not question or question.lower() in ("quit", "exit"):
            break

        books = pick_books(question, registry)
        print(f"  [routing] candidate books: {books}")

        for book_key in books:
            nodes = pick_nodes(question, book_key, registry)
            if not nodes:
                continue
            best_node = nodes[0]
            print(f"  [tree search] {book_key} -> {best_node['title']} "
                  f"(pages {best_node['start_index']}-{best_node['end_index']})")

            if args.summary_only:
                answer = answer_with_summary_only(question, book_key, best_node)
            else:
                answer = answer_with_pages(question, book_key, best_node)

            print(f"\n--- Answer (source: {book_key}, pages {best_node['start_index']}-{best_node['end_index']}) ---")
            print(answer)
            print()


if __name__ == "__main__":
    main()
