# Crawler Development

## Expected File Layout

- Entry point: `src/cli/main.py`, exposed as `uv run book-to-epub`.
- Registry: `src/crawler/registry.py`.
- Parser modules: `src/crawler/providers/<site>/parser.py`.
- Search modules: `src/crawler/providers/<site>/search.py`.
- Prepared source adapter: `src/content/prepare.py`.
- CMS draft upload: `src/notion/upload.py`.
- EPUB presentation: `src/epub/`.
- Shared models: `src/content/models.py`.
- Sources and checkpoints: `generated/`; optional reader outputs: `books/`.

## Parser Module Contract

A parser module should expose these pieces where practical:

```python
def crawl_book(url: str, ...) -> tuple[BookMeta, list[Volume]]: ...
```

Use local `BookMeta` when site metadata differs, but reuse:

```python
from src.content.models import Chapter, Volume
```

The registry collector returns `CrawledBook`. `src/workflows/ingest.py` owns
source preparation and the explicitly selected Notion, EPUB and/or TXT outputs. Providers only collect source material.

## Registry Pattern

Add a lazy collector in `src/crawler/registry.py`:

```python
def run_example(target: str, options: CrawlOptions) -> CrawledBook:
    from src.crawler.providers.example import parser

    meta, volumes = parser.crawl_book(target)
    return CrawledBook(
        title=meta.title, author=meta.author, volumes=volumes,
        source_url=target, intro_paragraphs=meta.intro_paragraphs,
    )
```

Register a `ParserSpec` with `name`, `domains`, `description`, and `crawl`.
Lazy imports keep discovery independent of individual provider dependencies.

## Search and Preview

Implement search/preview as a lightweight metadata layer. Do not download chapter bodies during preview unless the site has no table of contents and there is no cheaper option.

Expected `search.py` API:

```python
from src.crawler.search.models import BookPreview, SearchResult

def search_books(query: str, *, limit: int = 10) -> list[SearchResult]:
    ...

def preview_book(result: SearchResult) -> BookPreview:
    ...
```

Search strategy preference:

1. Native site search page or endpoint.
2. Local provider index, if one exists.
3. Shared external site search through `src.crawler.search.engines.site_search()`.
4. No search support, but keep `preview_book()` working for direct URLs if useful.

Preview should return title, author, canonical book URL, status when available, chapter count, first two chapter titles, last two chapter titles, and a short intro. Reuse parser helpers like `parse_info()`, `parse_toc()`, `parse_list_page()`, and `_resolve_book_url()` instead of duplicating selectors.

Ranking is centralized in `src/crawler/search/orchestrator.py`: query match level first, then chapter count, then provider priority. Provider `search.py` should return clean candidate metadata and avoid its own complex ranking.

Use `src.crawler.search.engines.site_search()` for reusable DuckDuckGo/Google fallback:

```python
from src.crawler.search.engines import site_search

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

## HTML Inspection

Inspect a saved HTML fixture first. For live inspection, follow the repository
browser/session rules; browser-backed collection remains inside the provider.
When an ordinary HTTP fetch is appropriate:

```bash
curl -sSL --compressed -A 'Mozilla/5.0 ...' '<url>' -o /tmp/site.html
uv run python - <<'PY'
from pathlib import Path
from bs4 import BeautifulSoup
html = Path('/tmp/site.html').read_text(encoding='utf-8', errors='replace')
soup = BeautifulSoup(html, 'lxml')
print(soup.title.get_text(strip=True) if soup.title else '')
for selector in ['div.main', 'article', '.intro', '#chaptercontent']:
    nodes = soup.select(selector)
    if nodes:
        print(selector, nodes[0].get_text(' ', strip=True)[:300])
PY
```

For browser-backed sites, follow the existing `xfxs`/`pili45` collection pattern
and use `src.crawler.fetch.browser.resolve_browser_executable()`;
`BOOKLIB_BROWSER_PATH` overrides browser discovery.

## Pagination and Completeness

Many novel sites paginate both the chapter list and individual long chapters. Treat both as first-class parser behavior:

- Chapter list pagination: look for `下一页`, numbered TOC pages, or API offsets. Crawl all TOC pages before building `ChapterRef`s. Deduplicate by stable chapter id, not title, because source sites can duplicate titles or split one title into `(1)` and `(2)`.
- Chapter body pagination: follow same-chapter next-page links such as `<chapter>_2.html`; stop only when the next link points to a different chapter id, a non-chapter URL, or no link exists.
- Never infer completeness from the first TOC page. Books with more than 100 chapters often hide later entries on `/2/<book_id>/2/`, `/3/`, etc.
- Preserve source order, but validate after generation so source numbering problems are visible instead of silently renumbered.

For long books, make a temporary subset test around a TOC page boundary and a chapter body pagination boundary. This catches skipped chapters without waiting for the whole book.

## Concurrency Pattern

Concurrency is useful for large books, but parser correctness is more important than speed. Use this pattern:

- Add a CLI option such as `--concurrency`, defaulting conservatively.
- Keep results ordered by the original `ChapterRef` index even when tasks finish out of order.
- Add progress logging on completion, not only task start: `done`, `failed`, `remaining`, and current chapter title.
- Add a global request interval or rate limiter across all workers. Multi-page chapters can multiply requests and trigger `429` if each worker fetches subpages aggressively.
- If the fast concurrent path fails for a chapter, collect the failed indexes and retry those chapters serially with the most reliable navigation path before failing the whole book.
- Do not let failed concurrent tasks produce a partial EPUB. Raise if fallback also fails.

For browser-backed sites:

- One browser with multiple tabs can work better than multiple browser processes, but still test it. Multiple tabs may be blocked by origin/cookie state unless each worker tab is initialized on the site origin.
- Multiple browser processes are heavier and may still hit the same IP/site rate limit; try request pacing and fallback before adding them.
- Browser `response.text()` can decode legacy encodings incorrectly. For GBK/Big5 pages, fetch `arrayBuffer()`, return base64 to Python, and decode bytes explicitly.
- If multi-tab setup times out, degrade gracefully to a shared-tab or serial path instead of aborting before chapter crawling.

## Text Cleanup Heuristics

Implement cleanup during source preparation, before either output. Common rules used in this project:

- Stop intro at review headings such as `作品简评`, `编辑评价`, or image-alt-prefixed headings like `金.gif 作品简评`.
- Stop FC2 post bodies at footer markers like `FC2拍手标签从这里开始` or Traditional variants.
- Remove embedded navigation lines, page-break prompts, duplicated chapter headings, ad text, and empty paragraphs.
- Drop chapter zero when it mostly repeats intro text. Normalize punctuation/whitespace and compare paragraph overlap rather than exact raw strings.
- Preserve meaningful intro metadata like `内容标签` and `搜索关键字` unless the user asks otherwise.

## Traditional Chinese Conversion

For Traditional Chinese sources, convert title, author, intro, volume titles, chapter titles, and paragraphs with:

```python
from opencc import OpenCC
cc = OpenCC('t2s')
```

Convert after parsing and before source preparation.

## Validation Checklist

Run at least:

```bash
uv run book-to-epub --list-parsers
uv run python -m compileall -q src
```

For a new parser, validate with a saved page fixture or a small live crawl:

- Title and author are correct.
- Intro does not contain reviews/editor comments/footers.
- Chapter count and first/last chapter titles are plausible.
- Chapter list pagination includes entries after page 1 when present.
- Chapter body pagination joins all parts of a long chapter.
- Concurrent runs do not skip chapters; compare requested refs, fetched chapters, empty bodies, and final nav/toc counts.
- Traditional Chinese sources are Simplified in output.
- EPUB nav/toc/spine do not reference removed chapters.

## Repairing Reader Output

When a parser defect also affects an existing book, correct the parser first.
For an explicitly requested repair of that book, follow [formatting.md](formatting.md)
to keep its source, chapter titles and navigation consistent.
