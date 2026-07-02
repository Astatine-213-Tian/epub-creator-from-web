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
uv run book-to-epub --search "锦衣卫" --parser xfxs --author 非天夜翔
```

Automatically choose the top ranked result:

```bash
uv run book-to-epub --search "全球高考" --first -o books/book.epub
```

## Parse A Known Book URL

Pass a supported book URL directly:

```bash
uv run book-to-epub "https://www.mangguoshufang.com/1/2574/info.html" -o books/book.epub
uv run book-to-epub "http://jrkywsy.blog.fc2.com/blog-entry-938.html" -o books/book.epub
```

When `-o/--output` is omitted, the generated file is written under the author
folder, for example `books/非天夜翔/书名.epub`. Existing author folders are reused.
An explicit `-o/--output` path still overrides this default.

List supported providers when you need to choose one explicitly:

```bash
uv run book-to-epub --list-parsers
```

For provider-specific IDs, force the provider:

```bash
uv run book-to-epub 2574 --parser mgsf -o books/book.epub
uv run book-to-epub doupocangqiong --parser quanben -o books/book.epub
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
- `xfxs` native search currently returns a 404 page. When `--author <name>` is supplied, xfxs searches the fixed author page `/a/<GBK-encoded-author>.html` first and skips external site-search for that provider. Without an author hint, xfxs uses external site-search fallback when available.
`src.search.engines.site_search()` tries DuckDuckGo and raw Google result-page fallbacks. Browser-backed providers can also use the same third-party engines through Chromium when raw search pages throttle. All results are still filtered back to the provider's canonical URL pattern.
- Generated EPUB files belong in `books/<author>/` by default and should not be treated as source code.

## Crawl Then Translate

For bilingual workflows, crawling and translation are separate commands. The
crawler writes a reusable snapshot with raw source data, normalized chapter
blocks, comments, and assets. The translation command can then reuse that
snapshot with any compatible glossary and vector-reference config.

Maintained per-book settings live under `book_specs/<book_slug>/`. For Eternal
Gate, `book_specs/eternal_gate/config.json` holds crawl/translation settings and
`book_specs/eternal_gate/glossary.json` holds the reusable glossary.

Create an authenticated Patreon crawl snapshot:

```bash
uv run book-crawl "https://www.patreon.com/collection/2218551?view=condensed" \
  --provider patreon \
  --title "永恒之门" \
  --author "顾雪柔" \
  --output generated/crawls/eternal_gate
```

Build or refresh the configured local vector index:

```bash
uv run --with sentence-transformers --with torch --with numpy \
  book-translate index --config book_specs/eternal_gate/config.json
```

Prepare translation prompts from the snapshot:

```bash
uv run --with sentence-transformers --with torch --with numpy \
  book-translate prepare generated/crawls/eternal_gate \
  --config book_specs/eternal_gate/config.json
```

Run prepared prompts through Codex, validate outputs, and build the bilingual
EPUB:

```bash
uv run book-translate run generated/translation_runs/eternal_gate \
  --config book_specs/eternal_gate/config.json \
  --retry-empty
uv run book-translate validate generated/translation_runs/eternal_gate \
  --config book_specs/eternal_gate/config.json \
  --allow-missing
uv run book-translate build-epub generated/crawls/eternal_gate \
  --run-dir generated/translation_runs/eternal_gate \
  --config book_specs/eternal_gate/config.json \
  -o books/顾雪柔/永恒之门.bilingual.epub
```

The same translation pipeline can be reused for other books by creating another
config under `book_specs/` and pointing it at a different crawl
snapshot, glossary, and reference-book set.

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
src/crawl/
src/translation/
```

## Validation

Useful checks after parser or search changes:

```bash
uv run book-to-epub --list-parsers
uv run book-to-epub --search "known title" --parser provider_name
python3 -m py_compile src/*.py src/*/*.py src/providers/*/*.py
```
