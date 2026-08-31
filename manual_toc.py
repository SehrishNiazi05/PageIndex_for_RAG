#!/usr/bin/env python3
"""
manual_toc.py

Build a PageIndex-format tree JSON for a bookmark-less PDF WITHOUT PageIndex's
expensive "standard" mode (which makes ~1 LLM call per section title and can
burn $3+/book over hours).

Pipeline:
  1. FREE detection (no LLM): scan pages for chapter/part start markers and any
     printed Contents/outline text.
  2. ONE cheap LLM call: turn the detected headings into a nested tree JSON.
  3. Post-process: assign node_id / end_index / summary="" and write the JSON in
     the same shape PageIndex writes (title/node_id/start_index/end_index/nodes).

Usage:
    python3 manual_toc.py --pdf "textbooks/Some Book.pdf"
    python3 manual_toc.py --pdf "textbooks/Some Book.pdf" --out "trees/Some Book_pageindex.json"
    python3 manual_toc.py --pdf "..." --detect-only   # print detection, skip LLM
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pypdfium2 as pdfium
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TEXT_MODEL = os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-v4-flash")

CHAPTER_NUM_RE = re.compile(r"CHAPTER\s+(\d+)\b")  # case-sensitive: start-page marker
PART_RE = re.compile(r"PART\s+(\d+|[IVX]+)\s*:?\s*([^\r\n]{1,60})", re.IGNORECASE)
SECTION_RE = re.compile(r"SECTION\s+([IVX]+|\d+)\s*:?\s*([^\r\n]{1,60})", re.IGNORECASE)
CONTENTS_LINE_RE = re.compile(r"^\s*(chapter|part|\d+[\.\)]?\s+)\b.*[\s.]+\d+\s*$", re.IGNORECASE)
TOC_LINE_RE = re.compile(r"[\s.]+\d{1,4}\s*$")  # any line ending in a page number

_TITLE_JUNK = ("\u25a0", "\u25b2", "\u25c6", "\u2022", "*")


def _clean(s):
    return re.sub(r"\s+", " ", s).strip()


def _valid_div_title(t):
    """Real PART/SECTION titles are short UPPERCASE noun phrases; citations and
    prose fragments (e.g. 'Part 1, J Clin...', 'Part II . Treatment') are not."""
    t = t.strip()
    if not t:
        return True  # allow "SECTION I" / "PART 1" with no trailing title
    if t.isdigit():
        return False  # running-header page numbers like "SECTION I 5"
    if t[0] in ".,;:()":
        return False
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.islower() for c in letters) > len(letters) * 0.3:
        return False
    return True


def _title_after(txt, pos):
    """Extract the title lines immediately following a 'CHAPTER N' match."""
    seg = txt[pos:pos + 200]
    parts = []
    for l in re.split(r"\r\n|\n|\r", seg):
        l = l.strip()
        if not l:
            if parts:
                break
            continue
        if l[0] in _TITLE_JUNK:
            break
        if re.search(r"\bOUTLINE\b", l, re.IGNORECASE):
            break
        if re.match(r"^(CHAPTER|PART|SECTION)\s+", l):
            break
        parts.append(l)
    title = " ".join(parts)
    if not title or len(title) > 120 or not title[0].isupper():
        return None
    return title


def _detect_contents(doc, total):
    """Find the printed Table of Contents and return its text (joined pages).

    Detect TOC pages by content (many "title ... page-number" lines), group
    consecutive such pages into runs, and keep the longest run. This is robust
    to the "Contents" heading appearing only on later TOC pages (or as a running
    header), and to books that have both a brief front-matter TOC and a full one.
    """
    runs = []
    current = []
    for i in range(min(total, 45)):
        txt = doc[i].get_textpage().get_text_range()
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        title_lines = sum(1 for l in lines if TOC_LINE_RE.search(l))
        if title_lines >= 5:
            current.append(i)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)

    if not runs:
        return []
    longest = max(runs, key=len)
    text = "\n".join(doc[j].get_textpage().get_text_range() for j in longest)
    return [(longest[0] + 1, text)]


def detect(pdf_path: Path):
    """Scan a PDF and return free (no-LLM) structural hints."""
    doc = pdfium.PdfDocument(str(pdf_path))
    total = len(doc)
    divisions = {}   # label -> (page, title)  (first occurrence wins)
    chapters = {}    # number  -> (page, title)
    running_headers = []

    for i in range(total):
        txt = doc[i].get_textpage().get_text_range()
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if lines:
            running_headers.append((i + 1, _clean(lines[0])[:70]))

        # Uppercase "CHAPTER N" is a chapter start marker; running headers /
        # cross-references use title case. Dedup by number -> first occurrence.
        for m in CHAPTER_NUM_RE.finditer(txt):
            num = m.group(1)
            if num in chapters:
                continue
            title = _title_after(txt, m.end())
            if title:
                chapters[num] = (i + 1, title)

        # PART / SECTION division headers (e.g. "PART 1 DIAGNOSES", "SECTION I ...")
        for m in PART_RE.finditer(txt):
            if not _valid_div_title(m.group(2)):
                continue
            label = ("PART " + " ".join(x for x in m.groups() if x)).strip()
            if len(label) <= 45:
                divisions.setdefault(label, (i + 1, label))
        for m in SECTION_RE.finditer(txt):
            if not _valid_div_title(m.group(2)):
                continue
            label = ("SECTION " + " ".join(x for x in m.groups() if x)).strip()
            if len(label) <= 55:
                divisions.setdefault(label, (i + 1, label))

    contents_pages = _detect_contents(doc, total)

    offset = None
    if contents_pages:
        offset = _offset_via_contents(doc, total, contents_pages[0][1])
    if offset is None:
        offset = _estimate_offset(running_headers)

    section_headers = _uppercase_headers(doc, total) if not chapters and not contents_pages else []

    # sort by page
    chapters = [(p, n, t) for n, (p, t) in sorted(chapters.items(), key=lambda kv: kv[1][0])]
    divisions = sorted(divisions.values(), key=lambda v: v[0])

    return {
        "total_pages": total,
        "parts": divisions,
        "chapters": chapters,
        "contents": contents_pages,
        "running_headers": running_headers,
        "section_headers": section_headers,
        "offset": offset,
    }


def compress_headers(headers):
    """Collapse running headers to the pages where they change."""
    out = []
    prev = None
    for page, h in headers:
        if h != prev:
            out.append((page, h))
            prev = h
    return out


def _uppercase_headers(doc, total):
    """Extract UPPERCASE running headers (chapter/section titles) with their
    first-appearance pages. For books with no TOC and no 'CHAPTER N' markers."""
    seen = {}
    for i in range(total):
        lines = [l.strip() for l in doc[i].get_textpage().get_text_range().splitlines() if l.strip()]
        for l in lines[:3]:
            c = re.sub(r"^\d{1,4}\s+|\s+\d{1,4}$", "", l).strip()
            if len(c) < 4 or len(c) > 80:
                continue
            letters = [ch for ch in c if ch.isalpha()]
            if not letters or sum(ch.isupper() for ch in letters) < len(letters) * 0.6:
                continue
            if c not in seen:
                seen[c] = i + 1
    return sorted([(p, t) for t, p in seen.items()], key=lambda x: x[0])


def _estimate_offset(running_headers):
    """Estimate printed-page -> physical-page offset from running headers.

    Running headers usually embed the printed page number (e.g. "SECTION I 5"),
    so offset = physical_page - printed_page. Returns the median offset, or None
    if it can't be estimated.
    """
    diffs = []
    for phys, h in running_headers:
        m = re.search(r"(\d{1,4})\s*$", h)
        if m and phys > int(m.group(1)):
            diffs.append(phys - int(m.group(1)))
    if not diffs:
        return None
    diffs.sort()
    return diffs[len(diffs) // 2]


def _offset_via_contents(doc, total, contents_text):
    """Robust offset: match the first few Contents titles to their physical
    headings in the body. Returns the median (physical - printed) or None."""
    entries = re.findall(
        r"^(?:Chapter\s+\d+|Part\s+[IVX\d]+|\d+[\.\)]?)\s+([A-Z][^\r\n]{2,60}?)[\.\s]+(\d{1,4})\s*$",
        contents_text, re.IGNORECASE | re.MULTILINE,
    )
    diffs = []
    for title, printed in entries[:8]:
        key = re.sub(r"[^\w\s]", "", title).strip().lower()
        if not key:
            continue
        for i in range(total):
            lines = doc[i].get_textpage().get_text_range().splitlines()
            head = " ".join(re.sub(r"[^\w\s]", "", l).strip() for l in lines[:4]).lower()
            if key in head:
                diffs.append((i + 1) - int(printed))
                break
    if not diffs:
        return None
    diffs.sort()
    return diffs[len(diffs) // 2]


def build_prompt(pdf_name, info):
    parts = info["parts"]
    chapters = info["chapters"]
    contents = info["contents"]
    total = info["total_pages"]

    sections = []

    if contents:
        text = "\n".join(t for _, t in contents)
        sections.append(
            "The book has a printed Table of Contents (front matter). Its text is:\n"
            f"<<<\n{text}\n>>>"
        )
        offset = info.get("offset")
        if offset:
            sections.append(
                f"NOTE: the Contents lists PRINTED page numbers. Convert them to "
                f"physical pages with: physical page = printed page + {offset} "
                f"(the front matter is about {offset} pages)."
            )
    if parts and (chapters or contents):
        listing = "\n".join(f'- physical page {p}: "{t}"' for p, t in parts)
        sections.append(f"Detected PART/SECTION headings:\n{listing}")
    if chapters:
        listing = "\n".join(
            f'- physical page {p}: "Chapter {n} {t}"' for p, n, t in chapters
        )
        sections.append(f"Detected CHAPTER headings (their start pages):\n{listing}")
    if not chapters and not contents:
        headers = info.get("section_headers") or compress_headers(info["running_headers"])
        listing = "\n".join(f'- page {p}: "{h}"' for p, h in headers)
        sections.append(
            "No chapter/part headings or Contents page were detected. Below are the "
            "UPPERCASE page/section headers with the physical page where each first "
            "appears. The chapter names are the broad recurring headers; the section "
            "names are the narrower ones that change frequently. Build a 2-level tree "
            f"(chapters -> sections) from them:\n{listing}"
        )

    body = "\n\n".join(sections)

    return (
        "You are building a table-of-contents (TOC) tree for a dental textbook PDF.\n\n"
        f"The PDF is named \"{pdf_name}\" and has {total} physical pages. "
        "All page numbers you output must be 1-based physical PDF page indexes.\n\n"
        f"{body}\n\n"
        "Produce a hierarchical tree of the book's real content structure. Rules:\n"
        "- Top level = the book's main divisions (Parts/Sections if present, otherwise chapters).\n"
        "- Chapters nest under their Part/Section.\n"
        "- Output at MOST TWO levels: divisions at the top, chapters nested under them. "
        "Do NOT create nodes for subsections, 'Abstract', 'Bibliography', 'References', "
        "or any leaf headings within a chapter.\n"
        "- Include ONLY real structural headings. Drop: Cover, Title Page, Copyright, "
        "Contributors, Foreword, Preface, Acknowledgments, running headers, and "
        "cross-references like \"Chapter 5).\" or \"Part 1, Compend...\". Keep Index/"
        "References/Appendices only if they are real top-level sections.\n"
        "- Each node needs \"title\" and \"start_index\" (the physical page where it starts). "
        "Do NOT include end_index (I compute it from the next sibling).\n"
        "- Use titles verbatim from the listing; fix only obvious extraction glitches and "
        "strip any trailing author names.\n\n"
        'Return ONLY a JSON object of the form: {"structure": [ {"title": "...", '
        '"start_index": <int>, "nodes": [...] } ]}'
    )


def _finalize(nodes, total_pages):
    counter = [0]

    def rec(ns, parent_end):
        for idx, n in enumerate(ns):
            counter[0] += 1
            n["node_id"] = f"{counter[0]:04d}"
            try:
                n["start_index"] = int(n.get("start_index", 1))
            except (TypeError, ValueError):
                n["start_index"] = 1
            children = n.get("nodes") or []
            if idx + 1 < len(ns):
                nxt = ns[idx + 1]
                try:
                    nxt_start = int(nxt.get("start_index", parent_end))
                except (TypeError, ValueError):
                    nxt_start = parent_end
                end = nxt_start - 1
            else:
                end = parent_end
            n["end_index"] = max(n["start_index"], end)
            n.setdefault("summary", "")
            if children:
                rec(children, n["end_index"])

    rec(nodes, total_pages)
    return nodes


def build_tree(pdf_path: Path, client: OpenAI) -> dict:
    info = detect(pdf_path)
    prompt = build_prompt(pdf_path.name, info)

    import time as _time
    last_err = None
    for attempt in range(1, 4):
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        try:
            data = json.loads(raw)
            structure = data.get("structure", [])
            if not isinstance(structure, list) or not structure:
                raise ValueError("LLM returned no 'structure' list")
            structure = _finalize(structure, info["total_pages"])
            return {
                "doc_name": pdf_path.name,
                "structure": structure,
                "toc_source": "manual",
            }
        except Exception as e:  # noqa: BLE001 - retry empty/truncated LLM output
            last_err = e
            if attempt < 3:
                _time.sleep(3 * attempt)

    raise RuntimeError(f"LLM failed to return a valid tree after retries: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Manual (cheap) TOC builder for bookmark-less PDFs.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF.")
    parser.add_argument("--out", default=None, help="Output JSON path (default: trees/<stem>_pageindex.json).")
    parser.add_argument("--detect-only", action="store_true", help="Print detection results and exit (no LLM).")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    if args.detect_only:
        info = detect(pdf_path)
        print(f"total_pages: {info['total_pages']}")
        print(f"parts: {len(info['parts'])}")
        for p, t in info["parts"]:
            print(f"  page {p}: {t}")
        print(f"chapters: {len(info['chapters'])}")
        for p, n, t in info["chapters"]:
            print(f"  page {p}: Chapter {n} {t}")
        print(f"contents pages: {[p for p, _ in info['contents']]}")
        return

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY not set. Copy .env.example to .env.")
        sys.exit(1)
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    tree = build_tree(pdf_path, client)

    out = Path(args.out) if args.out else (Path("trees") / f"{pdf_path.stem}_pageindex.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"Tree saved to: {out}")


if __name__ == "__main__":
    main()
