from __future__ import annotations

from urllib.parse import quote, urljoin

from bs4 import BeautifulSoup

from src.fetch.browser import wait_for_page_ready
from src.runtime.progress import ProgressLogger
from src.search import BookPreview, SearchResult

from . import parser


PROGRESS = ProgressLogger()


def _card_text(card, slot: str) -> str:
    node = card.select_one(f"[slot='{slot}']")
    return parser._clean_text(node.get_text(" ", strip=True)) if node else ""


def parse_search_page(html: str, *, limit: int = 10) -> list[SearchResult]:
    soup = BeautifulSoup(html, "lxml")
    results: list[SearchResult] = []
    for card in soup.select("z-bookcard[href]"):
        if card.find_parent(attrs={"hidden": True}) is not None:
            continue
        if parser._attr(card, "deleted") == "1" or parser._attr(card, "extension").lower() != "epub":
            continue
        title = _card_text(card, "title")
        author = _card_text(card, "author")
        if not title:
            continue
        publisher = parser._clean_text(parser._attr(card, "publisher"))
        year = parser._clean_text(parser._attr(card, "year"))
        file_size = parser._clean_text(parser._attr(card, "filesize"))
        quality = parser._clean_text(parser._attr(card, "quality"))
        snippet = " · ".join(
            value
            for value in (publisher, year, "EPUB", file_size, f"quality {quality}" if quality else "")
            if value
        )
        publisher_score = 20.0 if publisher == "MoyuStudio" else 0.0
        quality_score = float(quality) if quality.replace(".", "", 1).isdigit() else 0.0
        results.append(
            SearchResult(
                parser="zlibrary",
                title=title,
                author=author,
                url=urljoin(parser.HOST, parser._attr(card, "href")),
                source="Z-Library native search",
                snippet=snippet,
                raw_score=100.0 + publisher_score + quality_score,
            )
        )
        if len(results) >= limit:
            break
    return sorted(results, key=lambda result: result.raw_score, reverse=True)


def search_books(
    query: str,
    *,
    limit: int = 10,
    author: str | None = None,
) -> list[SearchResult]:
    del query, limit, author
    raise RuntimeError("Z-Library search requires a browser")


async def search_books_with_browser(
    query: str,
    *,
    limit: int = 10,
    browser,
    author: str | None = None,
) -> list[SearchResult]:
    search_query = f"{query} {author}".strip() if author else query
    url = f"{parser.HOST}/s/{quote(search_query)}"
    PROGRESS.provider_detail("zlibrary", f"browser fetching {url}")
    tab = await browser.get(url, new_tab=True)
    await wait_for_page_ready(tab, ready_selector="z-bookcard", settle_delay=0.5)
    results = parse_search_page(await tab.get_content(), limit=limit)
    if author:
        author_key = parser._clean_text(author).casefold()
        exact_author_results = [
            result
            for result in results
            if parser._clean_text(result.author).casefold() == author_key
        ]
        if exact_author_results:
            return exact_author_results
    return results


def _preview(result: SearchResult, meta: parser.BookMeta | None = None) -> BookPreview:
    if meta:
        title = meta.title
        author = meta.author
        intro = meta.description
        status = " · ".join(
            value
            for value in (meta.publisher, meta.year, meta.extension.upper(), meta.file_size)
            if value
        )
    else:
        title = result.title
        author = result.author
        intro = result.snippet
        status = "EPUB"
    return BookPreview(
        parser="zlibrary",
        title=title,
        author=author,
        url=result.url,
        chapter_count=None,
        first_chapters=(),
        last_chapters=(),
        intro=intro,
        status=status,
        source=result.source,
    )


def preview_book(result: SearchResult) -> BookPreview:
    return _preview(result)


async def preview_book_with_browser(result: SearchResult, *, browser) -> BookPreview:
    detail_url = parser._resolve_book_url(result.url)
    tab = await browser.get(detail_url, new_tab=True)
    await wait_for_page_ready(tab, ready_selector="h1.book-title", settle_delay=0.5)
    meta = parser.parse_detail_page(await tab.get_content(), detail_url)
    return _preview(result, meta)
