---
name: webpage-parser
description: Guidance for adding or updating webpage-to-EPUB providers in the epub-creator-from-web project. Use when Codex needs to support a new novel/book website, inspect webpage HTML, implement provider parser/search modules, register domain auto-detection, add search/preview support, convert Traditional Chinese to Simplified Chinese when needed, remove boilerplate/review/footer text, handle pagination/concurrency safely, or patch generated EPUB output.
---

# Webpage Parser

## Purpose

Use this skill to add or update a provider for a webpage source in this project. A provider owns both parsing and optional search/preview behavior for one site. The goal is maintainable provider code that plugs into the single `book-to-epub` entry point, supports `--search` where possible, and produces clean EPUB output.

## Provider Layout

Each supported source should live in its own provider package:

```text
src/providers/<provider>/
  __init__.py
  parser.py
  search.py
```

- `parser.py` contains site-specific crawling, parsing, EPUB-building helpers, `_resolve_book_url()`, and the provider's command-line `main()` if needed.
- `search.py` contains `search_books(query, *, limit)` and `preview_book(result)`.
- Keep orchestration shared: `src/providers/registry.py` for parser dispatch, `src/search/orchestrator.py` for search models/ranking, `src/search/engines.py` for reusable external site search, and `src/fetch/browser.py` for Chromium-compatible browser discovery.

## Core Workflow

1. Inspect the target page HTML with `curl`, Playwright, or the browser-backed path if the site blocks raw requests.
2. Identify stable selectors for title, author, intro, cover, table of contents, chapter links, and chapter body.
3. Add or update `src/providers/<provider>/parser.py`; do not add root-level entry scripts.
4. Reuse shared models and writer: `Chapter`, `Volume`, and `write_epub` from `src`.
5. Register the parser in `src/providers/registry.py` with a unique name, supported domains, and runner function.
6. Add `src/providers/<provider>/search.py` with lightweight discovery and preview support where possible.
7. Add cleanup rules for site boilerplate: review sections, editor comments, navigation junk, FC2 footer markers, duplicated intro chapters, and ad text.
8. Check for multi-page chapter lists and chapter body pagination; implement pagination before judging chapter counts or gaps.
9. For large books, add concurrency only with request pacing, progress logging, and serial fallback for failed chapters so faster crawls cannot skip content.
10. Validate with saved HTML when possible, then run `uv run book-to-epub --list-parsers`, `uv run book-to-epub --search "<known title>" --parser <provider>`, and `uv run python -m py_compile ...`.
11. If an EPUB was already generated with bad content, patch the EPUB zip safely and create a `.bak` backup.

## Search and Preview

Implement search/preview as a lightweight metadata layer. Do not download chapter bodies during preview unless the site has no table of contents and there is no cheaper option.

Expected `search.py` API:

```python
from src.search import BookPreview, SearchResult

def search_books(query: str, *, limit: int = 10) -> list[SearchResult]:
    ...

def preview_book(result: SearchResult) -> BookPreview:
    ...
```

Search strategy preference:

1. Native site search page or endpoint.
2. Local provider index, if one exists.
3. Shared external site search through `src.search.engines.site_search()`.
4. No search support, but keep `preview_book()` working for direct URLs if useful.

Preview should return title, author, canonical book URL, status when available, chapter count, first two chapter titles, last two chapter titles, and a short intro. Reuse parser helpers like `parse_info()`, `parse_toc()`, `parse_list_page()`, and `_resolve_book_url()` instead of duplicating selectors.

Ranking is centralized in `src/search/orchestrator.py`: query match level first, then chapter count, then provider priority. Provider `search.py` should return clean candidate metadata and avoid its own complex ranking.

Use `src.search.engines.site_search()` for reusable DuckDuckGo/Google fallback:

```python
from src.search.engines import site_search

items = site_search(
    query,
    site="example.com",
    path_prefix="/book/",
    limit=limit,
)
```

Always filter external search results back to canonical provider URLs before returning `SearchResult`.

For providers where raw result pages throttle or block requests, optionally expose
`search_books_with_browser(query, *, limit, browser)` and reuse the same external
search strategy through Chromium. Keep browser search provider-specific when the
result parsing or canonicalization depends on that site's URL shapes.

## Project Reference

Read `references/book-management-parser.md` when implementing or modifying a parser in this repository. It contains the expected parser shape, registry pattern, pagination/concurrency patterns, cleanup heuristics, validation checklist, and EPUB patching guidance.

## Rules

- Keep `uv run book-to-epub` as the only user-facing entry point.
- Keep site-specific behavior inside `src/providers/<provider>/`.
- Keep search previews lightweight; avoid fetching full chapter bodies before the user chooses a result.
- For browser-backed providers, use `src.fetch.browser.resolve_browser_executable()` instead of hard-coding Chrome or Chromium paths. Users can override discovery with `BOOKLIB_BROWSER_PATH`.
- Prefer robust text/DOM heuristics over one-off hard-coded line numbers.
- Convert Traditional Chinese with `OpenCC("t2s")` before EPUB writing when the source site uses Traditional Chinese.
- Never commit generated EPUBs from `epub/`.
