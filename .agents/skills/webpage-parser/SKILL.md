---
name: webpage-parser
description: Guidance for adding or updating webpage-to-EPUB parsers in the book-management project. Use when Codex needs to support a new novel/book website, inspect webpage HTML, implement a parser under booklib/parsers, register domain auto-detection, convert Traditional Chinese to Simplified Chinese when needed, remove boilerplate/review/footer text, handle pagination/concurrency safely, or patch generated EPUB output.
---

# Webpage Parser

## Purpose

Use this skill to add or update a parser for a webpage source in this project. The goal is a maintainable parser that plugs into the single `book-to-epub` entry point and produces clean EPUB output.

## Core Workflow

1. Inspect the target page HTML with `curl` or the browser-backed path if the site blocks raw requests.
2. Identify stable selectors for title, author, intro, cover, table of contents, chapter links, and chapter body.
3. Add or update a parser module in `booklib/parsers/<site>.py`; do not add root-level entry scripts.
4. Reuse shared models and writer: `Chapter`, `Volume`, and `write_epub` from `booklib`.
5. Register the parser in `booklib/parser_registry.py` with a unique name, supported domains, and runner function.
6. Add cleanup rules for site boilerplate: review sections, editor comments, navigation junk, FC2 footer markers, duplicated intro chapters, and ad text.
7. Check for multi-page chapter lists and chapter body pagination; implement pagination before judging chapter counts or gaps.
8. For large books, add concurrency only with request pacing, progress logging, and serial fallback for failed chapters so faster crawls cannot skip content.
9. Validate with saved HTML when possible, then run `uv run book-to-epub --list-parsers` and `uv run python -m py_compile ...`.
10. If an EPUB was already generated with bad content, patch the EPUB zip safely and create a `.bak` backup.

## Project Reference

Read `references/book-management-parser.md` when implementing or modifying a parser in this repository. It contains the expected parser shape, registry pattern, pagination/concurrency patterns, cleanup heuristics, validation checklist, and EPUB patching guidance.

## Rules

- Keep `book_to_epub.py` as the only user-facing entry point.
- Keep site-specific behavior inside `booklib/parsers/`.
- Prefer robust text/DOM heuristics over one-off hard-coded line numbers.
- Convert Traditional Chinese with `OpenCC("t2s")` before EPUB writing when the source site uses Traditional Chinese.
- Never commit generated EPUBs from `epub/`.
