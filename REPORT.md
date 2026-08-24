# Project Report — Building a Knowledge Source for an Agentic RAG App

## 1. The Big Picture (What and Why)

I am building an AI assistant that can answer questions about dental textbooks.

For an AI to answer questions from books, it needs to know where information lives
inside those books. Instead of reading a whole 1000-page book every time, I first
**organize each book into a table of contents (TOC)** — a structured outline of
chapters, sections, and subsections, each with a page range.

That outline (the "tree") becomes the **knowledge source** for my agentic RAG app,
which I will later pass to **Copilot Studio RAG** so it can find and read the right
pages when answering a question.

> In one line: **I turn textbooks into a searchable TOC so the AI can jump
> straight to the right section instead of scanning the whole book.**

## 2. Main Objective

- Use **PageIndex** to automatically create the **TOC (structure)** of my books.
- Store each book's structure as a JSON file (the "tree index").
- These trees are my **knowledge source**.
- Later, this knowledge source will be handed to **Copilot Studio RAG** for the
  agentic RAG app.

## 3. What Has Been Built So Far

### 3.1 The Books (the raw material)

- 9 dental textbooks are placed in the `textbooks/` folder. Examples:
  - Carranza's Clinical Periodontology
  - Shafer's Textbook of Oral Pathology
  - Scully's Medical Problems in Dentistry
  - White and Pharoah's Oral Radiology
  - and 5 more.

### 3.2 The Indexer (`batch_index.py`)

This is the first step. It:

- Goes through every PDF in `textbooks/`.
- Calls **PageIndex** (an open-source tool) to read the book and build its TOC.
- Saves the result as a JSON file in the `trees/` folder, for example:
  `trees/<bookname>_pageindex.json`.

The JSON contains chapters -> sections -> subsections, each with:
- a title,
- a start page and end page,
- a short AI-written summary of what that section is about.

### 3.3 The Query Loop (`query_cli.py`) — for testing

This is a small command-line app I use to test that the knowledge source works
before moving to Copilot Studio. For each question it:

1. Picks the most relevant book(s).
2. Searches that book's TOC to find the most relevant section(s).
3. Extracts the **plain text** of those pages (no images, tables, or charts).
4. Asks a **DeepSeek** text model to answer using only that text.

This proves the TOC (knowledge source) can actually locate the right content.

## 4. How It Fits Together

```
textbooks/*.pdf  -->  batch_index.py (PageIndex)  -->  trees/*.json  (the TOC / knowledge source)
                                                              |
                                                              v
                                         query_cli.py (DeepSeek)  <-- testing now
                                                              |
                                                              v
                                         Copilot Studio RAG  (next step)
```

## 5. Current Status

- PageIndex + DeepSeek are set up and running.
- The indexing script works; most books have been turned into TOC JSON files.
- A few books had no readable table of contents in the PDF, so they failed in
  "flash" (fast) mode. Fix: re-run them in "standard" mode, which uses the AI
  model to build the structure.
- The query loop works and answers from the extracted text only.

## 6. What Is Done vs What Is Next

| Step | Status |
| --- | --- |
| Collect 9 textbooks | Done |
| Build TOC for each book with PageIndex | Mostly done (a few need standard mode) |
| Test knowledge source with DeepSeek | Done / working |
| Hand the knowledge source to Copilot Studio RAG | Next step |

## 7. Key Idea to Remember

The important output of this whole stage is the **TOC JSON files in `trees/`**.
They are the structured knowledge source. Everything else (the query CLI) is a
testing tool to confirm the knowledge source is correct before it goes into
**Copilot Studio RAG**.
