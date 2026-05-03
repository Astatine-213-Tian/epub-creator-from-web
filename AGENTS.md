# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python scripts for scraping web novels and building EPUB files:

- `src/cli/main.py` is the unified entry point; it detects supported domains and dispatches to the correct parser.
- `src/cli/validate_epub_chapters.py` validates generated EPUB chapter numbering and table-of-contents consistency.
- `src/core/` contains shared dataclasses and EPUB writing helpers.
- `src/fetch/` contains browser and parallel-fetching helpers.
- `src/runtime/` contains environment loading and progress reporting.
- `src/providers/registry.py` contains parser dispatch.
- `src/providers/<provider>/parser.py` contains site-specific parsing, crawling, and EPUB-building helpers.
- `src/providers/<provider>/search.py` contains provider search and lightweight preview helpers.
- `src/search/orchestrator.py` contains shared search models, ranking, preview orchestration, and interactive selection.
- `src/search/engines.py` contains reusable external site-search helpers for providers without reliable native search.
- `epub/` stores generated EPUB outputs. Do not treat generated books as source code.

## Build, Test, and Development Commands

Create and sync the local environment with `uv`:

```bash
uv sync
```

Run the unified entry point:

```bash
uv run book-to-epub "https://www.mangguoshufang.com/1/2574/info.html" -o epub/book.epub
uv run book-to-epub "http://jrkywsy.blog.fc2.com/blog-entry-938.html" -o epub/book.epub
uv run book-to-epub 2574 --parser mgsf -o epub/book.epub
```

Run the search-and-preview pipeline:

```bash
uv run book-to-epub --search "全球高考"
uv run book-to-epub --search "斗破苍穹" --parser quanben
uv run book-to-epub --search "全球高考" --first -o epub/book.epub
```

Use `uv run book-to-epub --list-parsers` to inspect supported sites. Browser-backed providers such as xfxs and pili45 use `src.fetch.browser.resolve_browser_executable()` to find a Chromium-compatible browser. Set `BOOKLIB_BROWSER_PATH` to force a specific executable; otherwise discovery checks Playwright-managed Chromium, common executables on `PATH`, and common macOS app bundle paths.

## Coding Style & Naming Conventions

Target modern Python 3 with `from __future__ import annotations`. Use 4-space indentation, type hints for data models and helpers, and `dataclass` for structured records. Keep constants in `UPPER_SNAKE_CASE`, classes in `PascalCase`, and functions or variables in `snake_case`. Prefer small parser/fetcher/build functions over large monolithic changes. Preserve the existing section-divider comment style for readability.

For new providers, create `src/providers/<provider>/parser.py` and `src/providers/<provider>/search.py`. Keep provider-specific selectors, URL normalization, boilerplate cleanup, and browser work inside the provider package. Register parser dispatch in `src/providers/registry.py`; keep ranking and interactive selection in `src/search/orchestrator.py`.

For browser-backed providers, do not hard-code Chrome or Chromium paths. Use `resolve_browser_executable()` from `src.fetch.browser`.

Provider `search.py` modules should expose:

```python
def search_books(query: str, *, limit: int = 10) -> list[SearchResult]: ...
def preview_book(result: SearchResult) -> BookPreview: ...
```

Prefer native site search. If unavailable, use `src.search.engines.site_search()` and filter results back to canonical provider book URLs. The shared helper supports optional API-backed search through `TAVILY_API_KEY`, then falls back to raw Bing/Google result pages. Preview should be lightweight: parse metadata and table-of-contents pages, but do not fetch all chapter bodies before the user chooses a result.

## Testing Guidelines

There is no formal test suite yet. For parser changes, add lightweight tests only if introducing a test framework is explicitly requested. Validate manually with a small known book or saved HTML fixture when possible. For search changes, verify `uv run book-to-epub --search "known title" --parser <provider>` shows sensible previews without immediately downloading the whole book. For EPUB output, open the generated file and confirm metadata, table of contents, chapter order, and cover handling.

## Commit & Pull Request Guidelines

This checkout has no Git history, so use clear, imperative commit messages such as `Fix mangguoshufang chapter title parsing` or `Add xfxs cover fallback`. Pull requests should describe the target site, commands used for validation, generated output path, and any manual steps such as Cloudflare verification.

## Security & Configuration Tips

Do not commit credentials, browser profiles, temporary downloads, or copyrighted source text. Keep generated EPUBs in `epub/` and avoid hard-coded absolute paths.
