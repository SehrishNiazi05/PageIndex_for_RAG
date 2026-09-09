"""
Local PageIndex: a vectorless RAG indexer powered entirely by a local Ollama model.

A self-contained Python port of the reference "PageIndex_Ollama" pipeline
(https://github.com/Meldron09/PageIndex_Ollama) that builds a hierarchical
document tree (structure -> title -> physical page span) and generates a summary
for every node, using a local LLM via Ollama so indexing costs nothing and stays
on-machine.

The indexer routes each book through the reference pipeline's three modes:

  1. process_toc_with_page_numbers   - a front-matter Table of Contents exists
                                        and carries printed page numbers.
  2. process_toc_no_page_numbers     - a TOC exists but lacks printed numbers.
  3. process_no_toc                  - no usable TOC; the LLM infers the tree
                                        from the page text itself.

On top of the reference behaviour, this version adds two pragmatic fallbacks
used only when the primary mode verifies poorly:

  - CHAPTER OUTLINE seeding: some textbooks (e.g. McCracken, Sturdevant) print a
    per-chapter "CHAPTER OUTLINE" block (section list + page numbers) instead of
    a global TOC; those are harvested to seed the tree.
  - deterministic typography parser: builds the tree from font-size / bold /
    numbering-anchor signals (PyMuPDF) when no TOC or outline is available, and
    acts as a sanity check for the LLM-driven tree.

Output schema mirrors the reference:

    {"doc_name": "...", "structure": [{"title", "node_id", "summary",
       "start_index", "end_index", "nodes": [...]}]}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
import requests
import tiktoken

# ---------------------------------------------------------------------------
# Configuration (mirrors the reference config/config.yaml)
# ---------------------------------------------------------------------------

MODEL_NAME = "qwen2.5:7b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

TOC_CHECK_PAGE_NUM = 20           # pages to scan for a table of contents
MAX_PAGE_NUM_EACH_NODE = 10       # beyond this a node is split recursively
MAX_TOKEN_NUM_EACH_NODE = 20000   # token budget per node

# Output toggles
IF_ADD_NODE_ID = "yes"
IF_ADD_NODE_SUMMARY = "yes"
IF_ADD_DOC_DESCRIPTION = "no"
IF_ADD_NODE_TEXT = "no"

# Behaviour toggles for the extra (non-reference) fallbacks
VERIFY_ACCURACY_MIN = 0.6         # fall through to next strategy below this
MAX_VERIFY_SAMPLE = 25            # cap on entries sampled during verification
USE_OUTLINE_SEEDING = True        # harvest per-chapter "CHAPTER OUTLINE" pages
USE_TYPOGRAPHY_FALLBACK = True    # deterministic parser fallback / sanity check

# Deterministic (typography) parser thresholds
BODY_FONT_DELTA = 1.5
HEADER_FOOTER_MARGIN = 0.05

HEADING_PATTERNS = [
    re.compile(r"^\s*(chapter|part|appendix)\s+(?P<num>[IVXLCDM0-9]+)\b", re.IGNORECASE),
    re.compile(r"^\s*(?P<num>\d+(?:\.\d+){0,3})\.?\s+(?P<title>[A-Z][^\n]{1,120})$"),
    re.compile(r"^\s*(?P<num>[IVXLCDM]+)\s+(?P<title>[A-Z][^\n]{1,120})$"),
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: Optional[str] = None) -> int:
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model or "gpt-4o")
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def extract_json(content: Any) -> Any:
    """Best-effort JSON extraction from an LLM response (handles fenced code
    blocks, prose, trailing commas, and `null` literals)."""
    if not isinstance(content, str):
        return content
    text = content
    # strip reasoning-model thinking markers
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)

    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        # find first { or [ ... last } or ]
        start = min([i for i in (text.find("{"), text.find("[")) if i != -1] or [-1])
        if start != -1:
            end = max(text.rfind("}"), text.rfind("]"))
            if end > start:
                text = text[start:end + 1]

    text = text.replace("None", "null").replace("True", "true").replace("False", "false")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # trailing comma / single-quote cleanup pass
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}


class OllamaClient:
    """Thin Ollama wrapper with drop-in OpenAI-style call signatures."""

    def __init__(self, model: str = MODEL_NAME, url: str = OLLAMA_URL):
        self.model = model
        self.url = url
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = requests.post(
                self.url,
                json={"model": self.model, "prompt": "ping", "stream": False},
                timeout=15,
            )
            self._available = r.status_code == 200
        except requests.RequestException:
            self._available = False
        return self._available

    def call(self, prompt: str, max_retries: int = 10, timeout: int = 600) -> str:
        for i in range(max_retries):
            try:
                r = requests.post(
                    self.url,
                    json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
                    timeout=timeout,
                )
                r.raise_for_status()
                return r.json()["response"]
            except Exception as e:  # noqa: BLE001
                print(f"      [ollama] retry {i + 1}/{max_retries}: {e}")
                if i < max_retries - 1:
                    time.sleep(1)
                else:
                    return "Error"

    def call_json(self, prompt: str) -> Any:
        return extract_json(self.call(prompt))


# ---------------------------------------------------------------------------
# PDF I/O
# ---------------------------------------------------------------------------

class PdfDocument:
    def __init__(self, path: Path, limit_pages: Optional[int] = None):
        self.path = path
        self.doc = fitz.open(str(path))
        total = len(self.doc)
        if limit_pages:
            total = min(total, limit_pages)
        self.num_pages = total

    def page_text(self, index: int) -> str:
        """index is 0-based."""
        if 0 <= index < len(self.doc):
            return self.doc[index].get_text("text")
        return ""

    def build_page_list(self) -> list[tuple[str, int]]:
        return [(self.page_text(i), count_tokens(self.page_text(i))) for i in range(self.num_pages)]

    def full_page_text(self, index: int) -> str:
        return self.page_text(index)

    def close(self) -> None:
        self.doc.close()


# ---------------------------------------------------------------------------
# Deterministic typography parser (fallback)
# ---------------------------------------------------------------------------

@dataclass
class TypoHeading:
    title: str
    page: int          # 0-based
    size: float
    level: int


def _typography_tree(pdf: PdfDocument) -> list[dict]:
    """Build a flat [{structure, title, physical_index}] list on typography alone."""
    doc = pdf.doc
    hist: dict[float, int] = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    n = len(span["text"].strip())
                    if n:
                        size = round(span["size"], 1)
                        hist[size] = hist.get(size, 0) + n
    if not hist:
        return []
    body_size = max(hist, key=lambda k: hist[k])

    candidates: list[TypoHeading] = []
    for pno in range(pdf.num_pages):
        page = doc[pno]
        ph = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text or len(text) > 160:
                    continue
                bbox = fitz.Rect(line["bbox"])
                if bbox.y1 < ph * HEADER_FOOTER_MARGIN or bbox.y0 > ph * (1 - HEADER_FOOTER_MARGIN):
                    continue
                size = max(s["size"] for s in spans)

                anchor = None
                for pat in HEADING_PATTERNS:
                    m = pat.match(text)
                    if m:
                        anchor = m.groupdict().get("num", "x")
                        break

                def infer(num: str) -> int:
                    n = num.count(".")
                    return 1 if n == 0 else n + 1

                if anchor is not None:
                    candidates.append(TypoHeading(title=text, page=pno, size=size, level=infer(anchor)))
                else:
                    title_like = size > body_size and len(text.split()) <= 14 and not text.endswith((".", ";", ":,", ","))
                    if size >= body_size + BODY_FONT_DELTA and title_like:
                        lvl = 1 if size - body_size >= 6 else (2 if size - body_size >= 3 else 3)
                        candidates.append(TypoHeading(title=text, page=pno, size=size, level=lvl))

    out: list[dict] = []
    for i, c in enumerate(candidates):
        out.append({
            "structure": str(i + 1),
            "title": c.title,
            "physical_index": c.page + 1,
        })
    return out


# ---------------------------------------------------------------------------
# Reference pipeline port
# ---------------------------------------------------------------------------

def _tagged(page_text: str, physical_index: int) -> str:
    return f"<physical_index_{physical_index}>\n{page_text}\n<physical_index_{physical_index}>\n\n"


class PageIndexBuilder:
    def __init__(self, pdf: PdfDocument, opt, verbose: bool = True):
        self.pdf = pdf
        self.opt = opt
        self.verbose = verbose
        self.ollama = OllamaClient(opt.model)
        self.log: list[dict] = []

    def log_info(self, message):
        self.log.append({"message": message})
        if self.verbose:
            print(f"      {message}", flush=True)

    def _parallel(self, fn, items, max_workers=6):
        """Map fn over items concurrently, preserving order."""
        if not items:
            return []
        results: list = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception:  # noqa: BLE001
                    results[i] = None
        return results

    # -- low-level prompts (each maps to a reference function) ---------------

    def toc_detector_single_page(self, content: str) -> str:
        prompt = (
            'Your job is to detect if there is a table of content provided in the given text.\n\n'
            f'Given text: {content}\n\n'
            'return the following JSON format:\n'
            '{"thinking": "...", "toc_detected": "yes" or "no"}\n\n'
            'Directly return the final JSON structure. Do not output anything else.\n'
            'Please note: abstract, summary, notation list, figure list, table list are not table of contents.'
        )
        return self.ollama.call_json(prompt).get("toc_detected", "no")

    def detect_page_index(self, toc_content: str) -> str:
        prompt = (
            'You will be given a table of contents. Detect if page numbers/indices are given within it.\n\n'
            f'Given text: {toc_content}\n\n'
            'Reply format: {"thinking": "...", "page_index_given_in_toc": "yes" or "no"}\n\n'
            'Directly return the final JSON structure. Do not output anything else.'
        )
        return self.ollama.call_json(prompt).get("page_index_given_in_toc", "no")

    def toc_transformer(self, toc_content: str) -> list[dict]:
        prompt = (
            'You are given a table of contents. Transform it into JSON.\n'
            'structure is the numeric hierarchical index (first section = "1", first subsection = "1.1", etc).\n'
            'The response should be:\n'
            '{"table_of_contents": [ {"structure": "x.x.x" or None, "title": "...", "page": <page number or null>}, ... ]}\n\n'
            'Directly return the final JSON structure, do not output anything else.\n\n'
            f'Given table of contents:\n{toc_content}'
        )
        res = self.ollama.call_json(prompt)
        if isinstance(res, list):
            items = res
        elif isinstance(res, dict) and isinstance(res.get("table_of_contents"), list):
            items = res["table_of_contents"]
        else:
            items = []
        for it in items:
            if it.get("page") is not None:
                try:
                    it["page"] = int(re.sub(r"[^0-9]", "", str(it["page"])) or 0)
                except ValueError:
                    it["page"] = None
        return items

    def toc_index_extractor(self, toc: list[dict], content: str) -> list[dict]:
        prompt = (
            'You are given a table of contents (JSON) and tagged document pages. '
            'Add "physical_index" (keep the <physical_index_X> format) to each section that starts in the given pages.\n\n'
            'structure is the numeric hierarchical index.\n'
            'Response format: [ {"structure": "x", "title": "...", "physical_index": "<physical_index_X>"}, ... ]\n'
            'Only add physical_index to sections present in the pages; otherwise omit it.\n'
            'Directly return the final JSON. Do not output anything else.\n\n'
            f'Table of contents:\n{json.dumps(toc)}\n\nDocument pages:\n{content}'
        )
        res = self.ollama.call_json(prompt)
        return res if isinstance(res, list) else []

    def check_title_appearance(self, title: str, page_text: str) -> str:
        prompt = (
            'Check if the given section appears or starts in the given page_text (fuzzy match, ignore spaces).\n\n'
            f'The given section title is {title}.\nThe given page_text is {page_text}.\n\n'
            'Reply format: {"thinking": "...", "answer": "yes" or "no"}\n'
            'Directly return the final JSON. Do not output anything else.'
        )
        return self.ollama.call_json(prompt).get("answer", "no")

    def check_title_appearance_in_start(self, title: str, page_text: str) -> str:
        prompt = (
            'Check if the given section starts at the beginning of the page_text '
            '(no other content before the title).\n\n'
            f'Section title: {title}\npage_text: {page_text}\n\n'
            'Reply format: {"thinking": "...", "start_begin": "yes" or "no"}\n'
            'Directly return the final JSON. Do not output anything else.'
        )
        return self.ollama.call_json(prompt).get("start_begin", "no")

    def single_toc_item_index_fixer(self, title: str, content: str) -> int:
        prompt = (
            'Given a section title and tagged pages, find the physical index of the start page.\n\n'
            f'Section Title: {title}\nDocument pages: {content}\n\n'
            'Reply JSON: {"thinking": "...", "physical_index": "<physical_index_X>"}\n'
            'Directly return the final JSON. Do not output anything else.'
        )
        res = self.ollama.call_json(prompt)
        val = res.get("physical_index")
        if isinstance(val, str):
            m = re.search(r"(\d+)", val)
            return int(m.group(1)) if m else 0
        return int(val) if isinstance(val, int) else 0

    def summarize_node(self, text: str) -> str:
        return self.ollama.call(
            'You are given a part of a document. Describe the main points covered.\n\n'
            f'Partial Document Text: {text}\n\n'
            'Directly return the description, do not include any other text.'
        )

    # -- mode implementations ------------------------------------------------

    def find_toc_pages(self, page_list: list[tuple[str, int]], model) -> list[int]:
        limit = min(self.opt.toc_check_page_num, len(page_list))
        cands = [i for i in range(limit) if len(page_list[i][0].strip()) >= 5]

        def detect(i):
            return i, self.toc_detector_single_page(page_list[i][0])

        results = self._parallel(detect, cands, max_workers=6)

        flags = {}
        for res in results:
            if res is not None:
                flags[res[0]] = res[1]

        toc_pages: list[int] = []
        last_yes = False
        for i in range(limit):
            if i not in flags:
                pass
            detected = flags.get(i, "no")
            if detected == "yes":
                toc_pages.append(i)
                last_yes = True
            elif last_yes:
                # stop a couple pages after TOC ends
                break
            if i >= limit - 1:
                break
        return toc_pages

    def process_no_toc(self, page_list, model):
        # group pages into token chunks
        groups = self._group_pages(page_list, MAX_TOKEN_NUM_EACH_NODE)
        log = f"process_no_toc groups={len(groups)}"
        self.log_info(log)

        prompt_head = (
            'You are an expert in extracting hierarchical tree structure from a document.\n'
            'The structure variable is the numeric hierarchical index (first section = "1", first subsection = "1.1").\n'
            'Extract the original title (fix only space inconsistency).\n'
            'The text has <physical_index_X> tags marking page boundaries.\n'
            'Extract the physical_index of the section start (keep the <physical_index_X> format).\n\n'
            'Response format: [ {"structure": "x.x.x", "title": "...", "physical_index": "<physical_index_X>"}, ... ]\n'
            'Directly return the final JSON. Do not output anything else.\n\n'
        )
        first = self.ollama.call_json(prompt_head + f"Given text:\n{groups[0]}")
        toc = first if isinstance(first, list) else []

        for group in groups[1:]:
            cont = self.ollama.call_json(
                prompt_head
                + "Continue the tree structure from the previous part.\n"
                + f"Previous tree structure:\n{json.dumps(toc)}\nGiven text:\n{group}"
            )
            if isinstance(cont, list):
                toc.extend(cont)

        return self._convert_physical(toc)

    def process_outline_seed(self, page_list, model):
        """Harvest per-chapter CHAPTER OUTLINE blocks into a seeded tree."""
        seed = self._extract_chapter_outlines(page_list)
        if not seed:
            return []
        self.log_info(f"process_outline_seed entries={len(seed)}")
        return seed

    def _extract_chapter_outlines(self, page_list):
        items: list[dict] = []
        outline_re = re.compile(r"\bCHAPTER\s+OUTLINE\b|\bCHAPTER\s*(?P<ch>[0-9IVXLCDM]+)\b", re.IGNORECASE)
        for idx, (text, _tokens) in enumerate(page_list):
            if not outline_re.search(text):
                continue
            # only trust pages that look like outline listings (many trailing numbers)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            numbered = [l for l in lines if re.search(r"\d{1,4}$", l)]
            if len(numbered) < 5:
                continue
            for i, l in enumerate(lines):
                m = re.match(r"^\s*([IVXLCDM]+|\d+(?:\.\d+)*)[\).]?\s+(.+?)[,;.]?\s+(\d{1,4})$", l)
                if not m:
                    continue
                structure, title, page = m.group(1), m.group(2).strip(), int(m.group(3))
                # strip author names that trail titles (heuristic: title before comma)
                title = re.split(r",\s+[A-Z][a-z]+", title)[0].strip()
                items.append({"structure": self._outline_structure(structure), "title": title, "physical_index": page})
                if len(items) > 400:
                    break
            if len(items) > 30:
                break
        return items

    @staticmethod
    def _outline_structure(s: str) -> str:
        if s.isalpha():
            roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
            return str(roman.get(s, 1))
        return s

    @staticmethod
    def _convert_physical(toc: list[dict]) -> list[dict]:
        for it in toc:
            val = it.get("physical_index")
            if isinstance(val, str):
                m = re.search(r"(\d+)", val)
                it["physical_index"] = int(m.group(1)) if m else None
        return toc

    @staticmethod
    def _group_pages(page_list: list[tuple[str, int]], max_tokens: int) -> list[str]:
        contents = []
        token_lengths = []
        for i, (text, tok) in enumerate(page_list):
            contents.append(_tagged(text, i + 1))
            token_lengths.append(count_tokens(_tagged(text, i + 1)))

        total = sum(token_lengths)
        if total <= max_tokens:
            return ["".join(contents)]

        groups: list[str] = []
        cur: list[str] = []
        cur_tokens = 0
        expected = math.ceil(total / max_tokens)
        avg = math.ceil((total / expected + max_tokens) / 2)
        for content, tok in zip(contents, token_lengths):
            if cur_tokens + tok > avg and cur:
                groups.append("".join(cur))
                cur = []
                cur_tokens = 0
            cur.append(content)
            cur_tokens += tok
        if cur:
            groups.append("".join(cur))
        return groups

    # -- tree assembly -------------------------------------------------------

    @staticmethod
    def list_to_tree(data: list[dict]) -> list[dict]:
        def parent_of(structure):
            if structure is None:
                return None
            parts = str(structure).split(".")
            return ".".join(parts[:-1]) if len(parts) > 1 else None

        nodes: dict[str, dict] = {}
        roots: list[dict] = []
        for item in data:
            structure = item.get("structure")
            node = {
                "title": item.get("title"),
                "start_index": item.get("start_index"),
                "end_index": item.get("end_index"),
                "nodes": [],
            }
            nodes[str(structure)] = node
            parent = parent_of(structure)
            if parent is not None and parent in nodes:
                nodes[parent]["nodes"].append(node)
            else:
                roots.append(node)

        def clean(node):
            if not node["nodes"]:
                del node["nodes"]
            else:
                for child in node["nodes"]:
                    clean(child)
            return node

        return [clean(n) for n in roots]

    def post_processing(self, structure: list[dict], end_physical_index: int) -> list[dict]:
        for i, item in enumerate(structure):
            item["start_index"] = item.get("physical_index")
            if i < len(structure) - 1:
                nxt = structure[i + 1]
                item["end_index"] = nxt["physical_index"] - 1 if nxt.get("appear_start") == "yes" else nxt["physical_index"]
            else:
                item["end_index"] = end_physical_index
        tree = self.list_to_tree(structure)
        if tree:
            return tree
        for node in structure:
            node.pop("appear_start", None)
            node.pop("physical_index", None)
        return structure

    def verify_toc(self, page_list, list_result, model) -> tuple[float, list]:
        last = None
        for item in reversed(list_result):
            if item.get("physical_index") is not None:
                last = item["physical_index"]
                break
        if last is None or last < len(page_list) / 2:
            return 0.0, []

        indices = list(range(len(list_result)))
        if len(indices) > MAX_VERIFY_SAMPLE:
            indices = random.sample(indices, MAX_VERIFY_SAMPLE)

        n_pages = len(page_list)

        def check(idx):
            item = list_result[idx]
            if item.get("physical_index") is None:
                return None
            pnum = item["physical_index"]
            if not (0 < pnum <= n_pages):
                return None
            ans = self.check_title_appearance(item.get("title", ""), page_list[pnum - 1][0])
            return (idx, item.get("title", ""), pnum, ans)

        correct = 0
        incorrect = []
        for res in self._parallel(check, indices):
            if res is None:
                continue
            idx, title, pnum, ans = res
            if ans == "yes":
                correct += 1
            else:
                incorrect.append({"list_index": idx, "title": title, "physical_index": pnum, "answer": ans})
        checked = sum(1 for idx in indices if list_result[idx].get("physical_index") is not None)
        checked = max(1, checked)
        accuracy = correct / checked
        self.log_info(f"verify_toc accuracy={accuracy:.2f} incorrect={len(incorrect)}")
        return accuracy, incorrect

    def fix_incorrect_toc(self, toc, page_list, incorrect, model):
        end_index = len(page_list)
        incorrect_indices = {r["list_index"] for r in incorrect}
        for r in incorrect:
            li = r["list_index"]
            if toc[li].get("physical_index") is None:
                continue
            prev = 0
            for j in range(li - 1, -1, -1):
                if j not in incorrect_indices and toc[j].get("physical_index") is not None:
                    prev = toc[j]["physical_index"]
                    break
            nxt = end_index
            for j in range(li + 1, len(toc)):
                if j not in incorrect_indices and toc[j].get("physical_index") is not None:
                    nxt = toc[j]["physical_index"]
                    break
            # bound the search window to keep prompts small
            window = [p for p in range(prev, min(nxt, end_index) + 1) if 0 < p <= end_index]
            window = window[:60]
            content = "".join(_tagged(page_list[p - 1][0], p) for p in window)
            fixed = self.single_toc_item_index_fixer(r.get("title", ""), content)
            if 0 < fixed <= end_index:
                toc[li]["physical_index"] = fixed
        return toc

    def _index_titles(self, page_list) -> list[str]:
        # one normalized text blob per page, cached
        if getattr(self, "_title_index", None) is None:
            self._title_index = [re.sub(r"\s+", " ", t.lower()) for t, _ in page_list]
        return self._title_index

    def _compute_page_offset(self, toc, page_list) -> int:
        """Deterministic printed->physical offset: match titles to pages."""
        blobs = self._index_titles(page_list)
        diffs = []
        for it in toc[:80]:
            page_no = it.get("page")
            title = re.sub(r"\s+", " ", (it.get("title") or "")).lower().strip()
            if not page_no or not title or len(title) < 3:
                continue
            # search for the title across pages (bounded window around printed page)
            lo = max(0, page_no - 6)
            hi = min(len(blobs), page_no + 6)
            for pi in range(lo, hi):
                if title[:25] in blobs[pi]:
                    diffs.append(pi + 1 - page_no)
                    break
        if not diffs:
            return 0
        from collections import Counter
        return Counter(diffs).most_common(1)[0][0]

    def process_toc_with_page_numbers(self, toc_content, page_list, model):
        toc = self.toc_transformer(toc_content)
        self.log_info(f"toc_transformer entries={len(toc)}")

        offset = self._compute_page_offset(toc, page_list)
        self.log_info(f"printed<->physical offset={offset}")
        for it in toc:
            page_no = it.get("page")
            if page_no:
                it["physical_index"] = page_no + offset
            else:
                it["physical_index"] = None

        toc = [it for it in toc if it.get("physical_index") is not None]
        self.log_info(f"toc entries with physical_index={len(toc)}")
        return toc

    def process_toc_no_page_numbers(self, toc_content, page_list, model):
        toc = self.toc_transformer(toc_content)
        # resolve missing page numbers by locating each title in the document
        for it in toc:
            it["physical_index"] = self._locate_title(it.get("title", ""), page_list)
        return [it for it in toc if it.get("physical_index") is not None]

    def _locate_title(self, title: str, page_list) -> Optional[int]:
        if not title:
            return None
        norm = re.sub(r"\s+", " ", title.strip().lower())
        for i, (text, _t) in enumerate(page_list):
            if norm and norm[:20] in re.sub(r"\s+", " ", text.lower()):
                return i + 1
        return None

    def meta_processor(self, page_list, mode, toc_content=None, model=None) -> list[dict]:
        if mode == "process_toc_with_page_numbers":
            return self.process_toc_with_page_numbers(toc_content, page_list, model)
        if mode == "process_toc_no_page_numbers":
            return self.process_toc_no_page_numbers(toc_content, page_list, model)
        return self.process_no_toc(page_list, model)

    # -- top-level -----------------------------------------------------------

    def run(self, no_llm: bool = False) -> dict:
        page_list = self.pdf.build_page_list()
        self.log_info(f"pages={self.pdf.num_pages} tokens={sum(t for _, t in page_list)}")

        llm_ready = (not no_llm) and self.ollama.is_available()
        if not llm_ready:
            print("      [!] Deterministic-only mode (no LLM).", flush=True)
            toc_items = _typography_tree(self.pdf)
        else:
            toc_items = self._route(page_list)

        toc_items = self._convert_physical([it for it in toc_items if it.get("physical_index")])

        # annotate appear_start via LLM (or skip when no LLM)
        if llm_ready:
            def annotate(it):
                pnum = it.get("physical_index")
                if pnum and 0 < pnum - 1 < len(page_list):
                    it["appear_start"] = self.check_title_appearance_in_start(it["title"], page_list[pnum - 1][0])
                else:
                    it["appear_start"] = "no"
                return it
            self._parallel(annotate, toc_items)
        else:
            for it in toc_items:
                it["appear_start"] = "yes"

        valid = [it for it in toc_items if it.get("physical_index") is not None]
        tree = self.post_processing(valid, len(page_list))

        if self.opt.if_add_node_id == "yes":
            self._write_node_id(tree)
        if self.opt.if_add_node_summary == "yes":
            self._add_summaries(tree, page_list, llm_ready)

        return {"doc_name": self.pdf.path.name, "structure": tree}

    def _route(self, page_list) -> list[dict]:
        toc_pages = self.find_toc_pages(page_list, self.opt.model)
        if not toc_pages:
            self.log_info("check_toc: no TOC found -> outline/typography/no-toc")
            items = []
            if USE_OUTLINE_SEEDING:
                items = self.process_outline_seed(page_list, self.opt.model)
            if not items:
                if USE_TYPOGRAPHY_FALLBACK:
                    items = _typography_tree(self.pdf)
            if not items:
                items = self.process_no_toc(page_list, self.opt.model)
            return items

        toc_content = "".join(page_list[i][0] for i in toc_pages)
        toc_content = re.sub(r"\.{5,}", ": ", toc_content)
        has_index = self.detect_page_index(toc_content)
        self.log_info(f"check_toc: found TOC pages={toc_pages} page_index_in_toc={has_index}")

        if has_index == "yes":
            items = self.meta_processor(page_list, "process_toc_with_page_numbers", toc_content=toc_content, model=self.opt.model)
        else:
            items = self.meta_processor(page_list, "process_toc_no_page_numbers", toc_content=toc_content, model=self.opt.model)

        accuracy, incorrect = self.verify_toc(page_list, items, self.opt.model)
        if len(items) == 0:
            self.log_info("TOC path produced no entries; falling back to typography parser.")
            fallback = _typography_tree(self.pdf)
            if fallback:
                return fallback
        return items

    # -- outputs -------------------------------------------------------------

    def _write_node_id(self, data, node_id=0):
        stack = list(data)
        counter = 0
        while stack:
            node = stack.pop(0)
            counter += 1
            node["node_id"] = str(counter).zfill(4)
            stack.extend(node.get("nodes", []))
        return data

    def _add_summaries(self, tree, page_list, llm_ready):
        all_nodes: list[dict] = []

        def collect(nodes):
            for node in nodes:
                all_nodes.append(node)
                if node.get("nodes"):
                    collect(node["nodes"])

        collect(tree)

        def summ(node):
            start = node.get("start_index")
            end = node.get("end_index")
            if start and end and llm_ready:
                lo, hi = start - 1, min(end, len(page_list))
                text = " ".join(page_list[p][0] for p in range(lo, hi) if p < len(page_list))
                if text.strip():
                    node["summary"] = self.summarize_node(text[:MAX_TOKEN_NUM_EACH_NODE * 4])
            return node

        self._parallel(summ, all_nodes, max_workers=4)

    def save(self, result: dict, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"      wrote {out_path}")


class Options:
    def __init__(self, model, toc_check_page_num, max_page_num_each_node, max_token_num_each_node,
                 if_add_node_id, if_add_node_summary, if_add_doc_description, if_add_node_text):
        self.model = model
        self.toc_check_page_num = toc_check_page_num
        self.max_page_num_each_node = max_page_num_each_node
        self.max_token_num_each_node = max_token_num_each_node
        self.if_add_node_id = if_add_node_id
        self.if_add_node_summary = if_add_node_summary
        self.if_add_doc_description = if_add_doc_description
        self.if_add_node_text = if_add_node_text


def build(pdf_path: str, out_path: Optional[str] = None, model: str = MODEL_NAME,
          no_llm: bool = False, quiet: bool = False, limit_pages: Optional[int] = None,
          toc_check_pages: Optional[int] = None) -> dict:
    pdf = PdfDocument(Path(pdf_path), limit_pages=limit_pages)
    opt = Options(
        model=model,
        toc_check_page_num=toc_check_pages or TOC_CHECK_PAGE_NUM,
        max_page_num_each_node=MAX_PAGE_NUM_EACH_NODE,
        max_token_num_each_node=MAX_TOKEN_NUM_EACH_NODE,
        if_add_node_id=IF_ADD_NODE_ID,
        if_add_node_summary="no" if no_llm else IF_ADD_NODE_SUMMARY,
        if_add_doc_description=IF_ADD_DOC_DESCRIPTION,
        if_add_node_text=IF_ADD_NODE_TEXT,
    )
    builder = PageIndexBuilder(pdf, opt, verbose=not quiet)
    result = builder.run(no_llm=no_llm)

    if out_path:
        builder.save(result, Path(out_path))
    pdf.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Local PageIndex tree builder (Ollama)")
    parser.add_argument("pdf", nargs="?", help="Path to input PDF")
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model name")
    parser.add_argument("--out", help="Output JSON path (default extracted_trees/<name>_pageindex.json)")
    parser.add_argument("--no-llm", action="store_true", help="Deterministic-only, skip all LLM calls")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--limit-pages", type=int, help="Only process the first N pages")
    parser.add_argument("--toc-check-pages", type=int, help="Pages scanned for a table of contents")
    args = parser.parse_args()

    pdf_path = Path(args.pdf) if args.pdf else None
    if pdf_path is None:
        inputs = sorted(Path("data/input_textbooks").glob("*.pdf"))
        if not inputs:
            parser.error("No PDF provided and none found in data/input_textbooks/")
        pdf_path = inputs[0]
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    out = Path(args.out) if args.out else Path("extracted_trees") / (pdf_path.stem + "_pageindex.json")
    build(
        str(pdf_path),
        out_path=str(out),
        model=args.model,
        no_llm=args.no_llm,
        quiet=args.quiet,
        limit_pages=args.limit_pages,
        toc_check_pages=args.toc_check_pages,
    )


if __name__ == "__main__":
    main()
