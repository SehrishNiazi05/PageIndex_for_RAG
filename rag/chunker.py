"""Heading occurrence aware, page preserving chunks from Docling records."""
import hashlib
import json
import math
import re
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from .config import PROCESSED, RAW, CHUNK_VERSION, MAX_TOKENS, SOFT_TOKENS

_TOKENIZER = None


def tokens(text):
    """Use the BGE tokenizer when cached; conservative UTF-8 fallback offline."""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            from .config import EMBED_MODEL
            _TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL, local_files_only=True)
            _TOKENIZER.model_max_length = 10**9  # counting long source text before splitting
        except Exception:
            _TOKENIZER = False
    if _TOKENIZER:
        return len(_TOKENIZER.encode(text, add_special_tokens=True))
    return math.ceil(len(text.encode("utf-8")) / 2) + 2


def split_text(text, budget):
    """Split paragraphs, list lines, sentences, then words. Never exceed budget."""
    units = re.split(r"(?<=\n)|(?<=[.!?])\s+(?=[A-Z0-9])", text)
    out, buf = [], ""
    for unit in units:
        candidates = [unit] if tokens(unit) <= budget else re.findall(r"\S+\s*", unit)
        for part in candidates:
            if tokens(part) > budget:
                if buf:
                    out.append(buf.strip()); buf = ""
                step = max(1, budget * 2 // 3)
                while part:
                    lo, hi = 1, min(len(part), step * 2)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if tokens(part[:mid]) <= budget: lo = mid
                        else: hi = mid - 1
                    out.append(part[:lo].strip()); part = part[lo:]
            elif tokens(buf + part) > budget:
                if buf: out.append(buf.strip())
                buf = part
            else:
                buf += part
    if buf.strip(): out.append(buf.strip())
    return [x for x in out if x]


def _raw_list_pages(book):
    """Recover each merged list item's actual PDF page from its raw Docling item."""
    path = RAW / f"{book}.json"
    if not path.exists(): return []
    import importlib.util
    spec = importlib.util.spec_from_file_location("docling_postprocess", PROCESSED.parent / "2_postprocess.py")
    # Its config.py is a sibling; import via a temporary sys.path entry.
    import sys
    sys.path.insert(0, str(PROCESSED.parent))
    try:
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        doc = json.loads(path.read_text(encoding="utf-8"))
        return [(module.clean_text(str(it.get("text") or "")), (it.get("prov") or [{}])[0].get("page_no"))
                for it in module.iter_items(doc.get("body") or {}, doc) if it.get("label") == "list_item"]
    finally:
        sys.path.pop(0)


def records_with_pages(path):
    items = _raw_list_pages(path.stem)
    positions = defaultdict(list)
    for index, (text, _) in enumerate(items): positions[text].append(index)
    pos = 0
    with path.open(encoding="utf-8") as source:
      for line in source:
        r = json.loads(line)
        if r.get("front_matter"): continue
        # The source processor's first-content cut includes the table of
        # contents in this volume. The first real PART I occurrence is p.38.
        if path.stem == "White and Pharoah_s Oral Radiology (1)" and (r.get("page_no") or 0) < 38:
            continue
        if r["label"] != "list" or "\n" not in r["text"]:
            r["page_end"] = r.get("page_no")
            yield r
            continue
        lines = [x.removeprefix("- ").strip() for x in r["text"].splitlines() if x.strip()]
        pages = []
        for line_text in lines:
            matches = positions.get(line_text, [])
            at = bisect_left(matches, pos)
            if at < len(matches):
                pos = matches[at] + 1
                pages.append(items[pos - 1][1])
            else:
                pages.append(r.get("page_no"))
        # Keep a source record ID for each original list item, even though the
        # processed file merged them under the first ID.
        first = int(r["id"].rsplit("#", 1)[1])
        for i, (line_text, page) in enumerate(zip(lines, pages)):
            item = dict(r, id=f"{r['book']}#{first+i}", text="- " + line_text,
                        page_no=page, page_end=page, merged_source_id=r["id"])
            yield item


def chunk_book(path):
    stack = []
    pending = []
    special = []
    count = 0
    book = path.stem

    def emit(parts):
        nonlocal count
        if not parts: return None
        count += 1
        body = "\n\n".join(p["text"] for p in parts)
        source_spans = []
        offset = 0
        for part in parts:
            source_spans.append({"source_id":part["id"], "page":part.get("page_no"),
                                 "start":offset, "end":offset + len(part["text"])})
            offset += len(part["text"]) + 2
        pages = [p.get("page_no") for p in parts if isinstance(p.get("page_no"), int)]
        pages += [p.get("page_end") for p in parts if isinstance(p.get("page_end"), int)]
        heading = parts[0].get("heading_path_str", "")
        parent = stack[-1]["id"] if stack else None
        source_ids = list(dict.fromkeys(p["id"] for p in parts))
        raw_id = f"{CHUNK_VERSION}|{book}|{source_ids[0]}|{count}"
        chunk_id = hashlib.sha1(raw_id.encode()).hexdigest()[:20]
        return {"chunk_id": chunk_id, "book": book, "chapter": parts[0].get("chapter") or "",
                "heading": heading, "parent_id": parent, "source_ids": source_ids,
                "source_spans": source_spans,
                "page_start": min(pages) if pages else None, "page_end": max(pages) if pages else None,
                "label": parts[0]["label"], "text": body, "n_tokens": tokens(heading + "\n" + body),
                "reference_path": bool(re.search(r"\b(?:references?|bibliography)\b", heading, re.I)),
                "version": CHUNK_VERSION}

    def flush():
        nonlocal pending
        result = emit(pending)
        pending = []
        return result

    def flush_special():
        nonlocal special
        result = emit(special)
        special = []
        return result

    for r in records_with_pages(path):
        if r["label"] == "section_header":
            x = flush()
            if x: yield x
            x = flush_special()
            if x: yield x
            level = r.get("level") or 6
            while stack and stack[-1]["level"] >= level: stack.pop()
            stack.append({"level": level, "id": r["id"]})
            continue
        heading = r.get("heading_path_str", "")
        budget = MAX_TOKENS - tokens(heading + "\n")
        if budget < 64: heading = heading[-100:]; budget = MAX_TOKENS - tokens(heading + "\n")
        # Tables and captions remain isolated. Large tables are split by row;
        # every fragment keeps the table's source ID and PDF page.
        if r["label"] in ("table", "caption"):
            x = flush()
            if x: yield x
            parts = split_text(r["text"], budget)
            if len(parts) != 1:
                x = flush_special()
                if x: yield x
                for part in parts:
                    x = emit([dict(r, text=part)])
                    if x: yield x
                continue
            if special:
                prior = special[0]
                paired = {prior["label"], r["label"]} == {"table", "caption"}
                same_page = prior.get("page_no") == r.get("page_no")
                within = tokens(heading + "\n" + prior["text"] + "\n\n" + r["text"]) <= MAX_TOKENS
                if paired and same_page and within:
                    special.append(r)
                    x = flush_special()
                    if x: yield x
                    continue
                x = flush_special()
                if x: yield x
            special.append(dict(r, text=parts[0]))
            continue
        x = flush_special()
        if x: yield x
        for part in split_text(r["text"], budget):
            q = dict(r, text=part)
            if pending and (pending[-1].get("page_no") != q.get("page_no")
                            or tokens(heading + "\n" + "\n\n".join(x["text"] for x in pending) + "\n\n" + part) > SOFT_TOKENS
                            or pending[0].get("heading_path_str") != heading):
                x = flush()
                if x: yield x
            pending.append(q)
            if tokens(heading + "\n" + "\n\n".join(x["text"] for x in pending)) >= SOFT_TOKENS:
                x = flush()
                if x: yield x
    x = flush()
    if x: yield x
    x = flush_special()
    if x: yield x


def all_chunks():
    for path in sorted(PROCESSED.glob("*.jsonl")):
        yield from chunk_book(path)
