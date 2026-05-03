# Book Management Parser Reference

## Expected File Layout

- Entry point: `book_to_epub.py` only.
- Registry: `booklib/parser_registry.py`.
- Parser modules: `booklib/parsers/<site>.py`.
- Shared EPUB writer: `booklib/epub_writer.py`.
- Shared models: `booklib/models.py`.
- Generated outputs: `epub/`, ignored by Git.

## Parser Module Contract

A parser module should expose these pieces where practical:

```python
def crawl_book(url: str, ...) -> tuple[BookMeta, list[Volume]]: ...
def build_epub(meta: BookMeta, volumes: list[Volume], out_path: Path) -> None: ...
def main(argv: list[str] | None = None) -> int: ...
```

Use local `BookMeta` when site metadata differs, but reuse:

```python
from booklib import Chapter, Volume, write_epub
```

`build_epub()` should delegate to `write_epub(...)` instead of creating EbookLib objects directly.

## Registry Pattern

Add a lazy import runner in `booklib/parser_registry.py`:

```python
def run_example(target: str, options: ParserOptions) -> Path:
    from booklib.parsers import example as parser

    meta, volumes = parser.crawl_book(target)
    out_path = resolve_output(options, meta.title)
    parser.build_epub(meta, volumes, out_path)
    return out_path
```

Then add a `ParserSpec` with `name`, `domains`, `description`, and `run`. Lazy imports keep `--list-parsers` working even when optional site dependencies are missing.

## HTML Inspection

Start with raw HTML:

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

If the site requires JavaScript or Cloudflare, follow the existing `xfxs` pattern using `zendriver`.

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

Implement cleanup before EPUB writing. Common rules used in this project:

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

Convert after parsing and before `write_epub()`.

## Validation Checklist

Run at least:

```bash
uv run book-to-epub --list-parsers
uv run python -m py_compile book_to_epub.py booklib/*.py booklib/parsers/*.py
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

## Patching Existing EPUBs

When fixing an already-generated EPUB, treat it as a zip archive:

1. Copy `book.epub` to `book.epub.bak`.
2. Rewrite into `book.epub.tmp`.
3. Remove or edit affected XHTML plus `content.opf`, `nav.xhtml`, and `toc.ncx` if deleting files.
4. Move tmp over original only after the patch succeeds.
5. Reopen the EPUB zip and assert bad markers are gone.

Never patch generated EPUBs instead of fixing the parser; do both when the user asks to repair existing output.
