# EPUB Creator From Web

Scrape supported web novel sites and build EPUB files through one command-line entry point.

## Setup

```bash
uv sync
```

Browser-backed providers require a Chromium-compatible browser. Install Chromium,
Google Chrome, or Playwright-managed Chromium; set `BOOKLIB_BROWSER_PATH` if the
browser is not discoverable automatically.

## Search And Choose A Version

Use `--search` when you know the title but not the best source URL:

```bash
uv run book-to-epub --search "全球高考"
```

The pipeline:

1. Searches supported providers.
2. Builds lightweight previews without downloading full chapter bodies.
3. Ranks results by query match, chapter count, then provider preference.
4. Shows each candidate with title, author, chapter count, first two chapters, and last two chapters.
5. Prompts you to choose one.
6. Runs the existing parser for the selected provider and writes the EPUB.

Limit search to one provider:

```bash
uv run book-to-epub --search "斗破苍穹" --parser quanben
uv run book-to-epub --search "魔道祖师" --parser mgsf --limit 3
```

Automatically choose the top ranked result:

```bash
uv run book-to-epub --search "全球高考" --first -o epub/book.epub
```

## Parse A Known Book URL

Pass a supported book URL directly:

```bash
uv run book-to-epub "https://www.mangguoshufang.com/1/2574/info.html" -o epub/book.epub
uv run book-to-epub "http://jrkywsy.blog.fc2.com/blog-entry-938.html" -o epub/book.epub
```

List supported providers when you need to choose one explicitly:

```bash
uv run book-to-epub --list-parsers
```

For provider-specific IDs, force the provider:

```bash
uv run book-to-epub 2574 --parser mgsf -o epub/book.epub
uv run book-to-epub doupocangqiong --parser quanben -o epub/book.epub
```

## Ranking

Search results are ranked simply:

1. Query match level.
2. More chapters.
3. Provider preference: `pili45`, then `towasakata`, then `xfxs`, then the rest.

This helps surface fuller versions when the same book exists on multiple sites.

## Provider Notes

- `pili45` and `xfxs` use browser-backed fetching through a Chromium-compatible browser.
- Browser discovery checks `BOOKLIB_BROWSER_PATH`, Playwright-managed Chromium, common `chromium` / `google-chrome` executables on `PATH`, then common macOS app bundle paths.
- To force a browser path:

```bash
BOOKLIB_BROWSER_PATH="/path/to/chromium" uv run book-to-epub --search "全球高考" --parser pili45
```

- Browser-backed providers may pause on Cloudflare verification.
- `xfxs` native search currently returns a 404 page, so its search uses external site-search fallback when available.
`src.search.engines.site_search()` tries DuckDuckGo and raw Google result-page fallbacks. Browser-backed providers can also use the same third-party engines through Chromium when raw search pages throttle. All results are still filtered back to the provider's canonical URL pattern.
- Generated EPUB files belong in `epub/` and should not be treated as source code.

## Development Layout

Provider-specific code lives under:

```text
src/providers/<provider>/
  parser.py
  search.py
```

Shared orchestration lives in:

```text
src/cli/main.py
src/core/
src/fetch/
src/providers/registry.py
src/runtime/
src/search/orchestrator.py
src/search/engines.py
```

## Validation

Useful checks after parser or search changes:

```bash
uv run book-to-epub --list-parsers
uv run book-to-epub --search "known title" --parser provider_name
python3 -m py_compile src/*.py src/*/*.py src/providers/*/*.py
```
