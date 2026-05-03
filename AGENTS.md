# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python scripts for scraping web novels and building EPUB files:

- `book_to_epub.py` is the unified entry point; it detects supported domains and dispatches to the correct parser.
- `booklib/parsers/` contains site-specific implementations: `towasakata.py`, `jrkywsy.py`, `xfxs.py`, and `mgsf.py`.
- `booklib/` contains shared dataclasses, EPUB writing, and parser registry code.
- `_patch_lswz_titles.py` is a one-off EPUB post-processing helper.
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

Use `uv run book-to-epub --list-parsers` to inspect supported sites. The xfxs parser expects Google Chrome at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`.

## Coding Style & Naming Conventions

Target modern Python 3 with `from __future__ import annotations`. Use 4-space indentation, type hints for data models and helpers, and `dataclass` for structured records. Keep constants in `UPPER_SNAKE_CASE`, classes in `PascalCase`, and functions or variables in `snake_case`. Prefer small parser/fetcher/build functions over large monolithic changes. Preserve the existing section-divider comment style for readability.

## Testing Guidelines

There is no formal test suite yet. For parser changes, add lightweight tests only if introducing a test framework is explicitly requested. Validate manually with a small known book or saved HTML fixture when possible. For EPUB output, open the generated file and confirm metadata, table of contents, chapter order, and cover handling.

## Commit & Pull Request Guidelines

This checkout has no Git history, so use clear, imperative commit messages such as `Fix mangguoshufang chapter title parsing` or `Add xfxs cover fallback`. Pull requests should describe the target site, commands used for validation, generated output path, and any manual steps such as Cloudflare verification.

## Security & Configuration Tips

Do not commit credentials, browser profiles, temporary downloads, or copyrighted source text. Keep generated EPUBs in `epub/` and avoid hard-coded absolute paths unless the script is intentionally local-only, as in `_patch_lswz_titles.py`.
