# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python scripts for scraping web novels and building EPUB files:

- `src/cli/main.py` is the unified entry point; it detects supported domains and dispatches to the correct parser.
- `src/cli/validate_epub_chapters.py` validates generated EPUB chapter numbering and table-of-contents consistency.
- `src/core/` contains shared dataclasses and EPUB writing helpers.
- `src/crawl/` contains the reusable crawl snapshot schema and helpers.
- `src/fetch/` contains browser and parallel-fetching helpers.
- `src/runtime/` contains environment loading and progress reporting.
- `src/providers/registry.py` contains parser dispatch.
- `src/providers/<provider>/parser.py` contains site-specific parsing, crawling, and EPUB-building helpers.
- `src/providers/<provider>/search.py` contains provider search and lightweight preview helpers.
- `src/search/orchestrator.py` contains shared search models, ranking, preview orchestration, and interactive selection.
- `src/search/engines.py` contains reusable external site-search helpers for providers without reliable native search.
- `src/translation/` contains the reusable crawl-snapshot-to-bilingual-EPUB translation pipeline: comment ranking, glossary loading, prompt generation, content-plan author-style transfer, Codex CLI execution, semantic validation, and bilingual EPUB building.
- `book_specs/` stores maintained per-book specs. Each book should have a slug folder, for example `book_specs/eternal_gate/config.json` and `book_specs/eternal_gate/glossary.json`.
- `generated/` stores production artifacts such as crawl snapshots, translation runs, and minimized author-style bundles.
- `books/` stores final reader-facing book outputs, currently EPUB files grouped by author. Do not treat generated books as source code.
- `tests/` contains focused production-pipeline contract tests.
- `research/` is an independent nested `uv` project for local corpora, author-style benchmarks, experiment scripts, reports, and generated research evidence. Production code must not import it.

## Build, Test, and Development Commands

Create and sync the local environment with `uv`:

```bash
uv sync
```

Run the unified entry point:

```bash
uv run book-to-epub "https://www.mangguoshufang.com/1/2574/info.html" -o books/book.epub
uv run book-to-epub "http://jrkywsy.blog.fc2.com/blog-entry-938.html" -o books/book.epub
uv run book-to-epub 2574 --parser mgsf -o books/book.epub
```

Run the search-and-preview pipeline:

```bash
uv run book-to-epub --search "全球高考"
uv run book-to-epub --search "斗破苍穹" --parser quanben
uv run book-to-epub --search "全球高考" --first -o books/book.epub
```

Use `uv run book-to-epub --list-parsers` to inspect supported sites. Browser-backed providers such as xfxs and pili45 use `src.fetch.browser.resolve_browser_executable()` to find a Chromium-compatible browser. Set `BOOKLIB_BROWSER_PATH` to force a specific executable; otherwise discovery checks Playwright-managed Chromium, common executables on `PATH`, and common macOS app bundle paths.

Run the reusable crawl-then-translate workflow as separate tasks:

```bash
uv run book-crawl "https://www.patreon.com/collection/2218551?view=condensed" \
  --provider patreon \
  --title "永恒之门" \
  --author "顾雪柔" \
  --output generated/crawls/eternal_gate

uv run book-translate prepare generated/crawls/eternal_gate \
  --config book_specs/eternal_gate/config.json

uv run book-translate run generated/translation_runs/eternal_gate \
  --config book_specs/eternal_gate/config.json
uv run book-translate validate generated/translation_runs/eternal_gate \
  --config book_specs/eternal_gate/config.json
uv run book-translate transfer-style generated/translation_runs/eternal_gate \
  --config book_specs/eternal_gate/config.json \
  --run-codex
uv run book-translate validate \
  generated/translation_runs/eternal_gate/author_style_transfer \
  --config book_specs/eternal_gate/config.json
uv run book-translate build-epub generated/crawls/eternal_gate \
  --run-dir generated/translation_runs/eternal_gate/author_style_transfer \
  --config book_specs/eternal_gate/config.json \
  -o books/顾雪柔/永恒之门.bilingual.epub
```

Dataset TXT output and manifests live under `research/datasets/`. The production
`book-ingest` and `book-dataset` commands write there by default so crawling and
EPUB generation remain centralized while research owns the resulting corpus.
Run author-style experiments from `research/` with its own environment:

```bash
cd research
uv sync
uv run author-style-research verify
```

## Coding Style & Naming Conventions

Target modern Python 3 with `from __future__ import annotations`. Use 4-space indentation, type hints for data models and helpers, and `dataclass` for structured records. Keep constants in `UPPER_SNAKE_CASE`, classes in `PascalCase`, and functions or variables in `snake_case`. Prefer small parser/fetcher/build functions over large monolithic changes. Preserve the existing section-divider comment style for readability.

For new providers, create `src/providers/<provider>/parser.py` and `src/providers/<provider>/search.py`. Keep provider-specific selectors, URL normalization, boilerplate cleanup, and browser work inside the provider package. Register parser dispatch in `src/providers/registry.py`; keep ranking and interactive selection in `src/search/orchestrator.py`.

For crawl snapshots, keep site-specific crawling in `src/providers/<provider>/` and write the normalized reusable snapshot format consumed by `src/translation/`. Do not put book-specific translation choices in provider code; use `book_specs/<book_slug>/config.json` and `book_specs/<book_slug>/glossary.json`.

For browser-backed providers, do not hard-code Chrome or Chromium paths. Use `resolve_browser_executable()` from `src.fetch.browser`.

Provider `search.py` modules should expose:

```python
def search_books(query: str, *, limit: int = 10) -> list[SearchResult]: ...
def preview_book(result: SearchResult) -> BookPreview: ...
```

Prefer native site search. If unavailable, use `src.search.engines.site_search()` and filter results back to canonical provider book URLs. The shared helper uses no-key DuckDuckGo and raw Google result-page fallbacks. Preview should be lightweight: parse metadata and table-of-contents pages, but do not fetch all chapter bodies before the user chooses a result.

## Testing Guidelines

Focused production contract tests live under `tests/` and run with
`uv run python -m unittest discover -s tests -p 'test_*.py'`. For parser
changes, also validate manually with a small known book or saved HTML fixture
when possible. For search changes, verify `uv run book-to-epub --search "known
title" --parser <provider>` shows sensible previews without immediately
downloading the whole book. For EPUB output, open the generated file and confirm
metadata, table of contents, chapter order, and cover handling.

For translation-pipeline changes, at minimum run `uv run python -m py_compile ...`, `uv run book-crawl --help`, `uv run book-translate --help`, the focused production unit tests, and a small fixture through `book-translate prepare`, `validate --config`, `transfer-style`, and `build-epub`.

For generated bilingual EPUBs, the main Codex run retries failed chunks with narrower context and paragraph fallbacks. Empty `zh` values are allowed only after retry when a paragraph still cannot be translated and must produce no placeholder/refusal text. In that case validate the run with `book-translate validate --allow-missing --config ...`, then scan the final EPUB for placeholder strings and confirm only the intended paragraphs are English-only.

## Commit & Pull Request Guidelines

Use Conventional Commits for commit messages, such as `fix(xfxs): repair preview metadata` or `docs: update provider search notes`. Pull requests should describe the target site, commands used for validation, generated output path, and any manual steps such as Cloudflare verification.

## Security & Configuration Tips

Do not commit credentials, browser profiles, temporary downloads, generated crawl snapshots, generated translation runs, generated EPUBs, or copyrighted source text. Keep final book outputs in `books/`, production crawl/translation artifacts under `generated/`, research corpora and experiment outputs under `research/datasets/` and `research/generated/`, and maintained per-book specs under `book_specs/`. Avoid hard-coded absolute paths.
