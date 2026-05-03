from __future__ import annotations

import asyncio
from urllib.parse import quote, urljoin, urlparse

from bs4 import BeautifulSoup

from src.fetch.browser import wait_for_page_ready
from src.runtime.progress import ProgressLogger
from src.search import BookPreview, SearchResult
from . import parser


PROGRESS = ProgressLogger()


def search_books(query: str, *, limit: int = 10) -> list[SearchResult]:
    return asyncio.run(_search_books(query, limit=limit))


async def _search_books(query: str, *, limit: int = 10) -> list[SearchResult]:
    fetcher = parser.Fetcher(headless=False, delay=0.1)
    PROGRESS.provider_detail("pili45", "starting browser for search")
    await fetcher.start()
    try:
        url = f"{parser.HOST}/search/0/{quote(query)}/1.html"
        PROGRESS.provider_detail("pili45", f"fetching search page {url}")
        return _parse_search_html(await fetcher.get_html(url), limit=limit)
    finally:
        PROGRESS.provider_detail("pili45", "closing search browser")
        await fetcher.stop()


async def search_books_with_browser(query: str, *, limit: int = 10, browser) -> list[SearchResult]:
    url = f"{parser.HOST}/search/0/{quote(query)}/1.html"
    PROGRESS.provider_detail("pili45", f"fetching search page {url}")
    tab = await browser.get(url, new_tab=True)
    await _wait_for_search_page(tab)
    return _parse_search_html(await tab.get_content(), limit=limit)


async def _wait_for_search_page(tab) -> None:
    await wait_for_page_ready(
        tab,
        ready_selector='a[href*="info.html"], .works-intro-title, .works-chapter-list',
    )


def _parse_search_html(html: str, *, limit: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "lxml")
    results: list[SearchResult] = []
    for a in soup.find_all("a", href=True):
        path = urlparse(a["href"]).path
        if not parser.INFO_RE.match(path):
            continue
        item = a.find_parent("li") or a.find_parent("div") or a
        title = parser._clean_text(a.get_text())
        title = title.replace("已完结", "").strip()
        if not title:
            continue
        snippet = parser._clean_text(item.get_text(" ")) if item else ""
        results.append(
            SearchResult(
                parser="pili45",
                title=title,
                author="",
                url=urljoin(parser.HOST, a["href"]),
                source="pili45 native search",
                snippet=snippet,
            )
        )
        if len(results) >= limit:
            break
    return results


def preview_book(result: SearchResult) -> BookPreview:
    return asyncio.run(_preview_book(result))


async def _preview_book(result: SearchResult) -> BookPreview:
    book_url = parser._resolve_book_url(result.url)
    parsed = urlparse(book_url)
    m = parser.INFO_RE.match(parsed.path)
    if not m:
        raise ValueError(f"not a pili45 info URL: {book_url}")
    cat_id, book_id = m.group(1), m.group(2)

    fetcher = parser.Fetcher(headless=False, delay=0.1)
    PROGRESS.provider_detail("pili45", "starting browser for preview")
    await fetcher.start()
    try:
        meta, refs = await _fetch_preview_data(
            book_url,
            cat_id,
            book_id,
            get_html=fetcher.get_html,
        )
    finally:
        PROGRESS.provider_detail("pili45", "closing preview browser")
        await fetcher.stop()

    return _make_preview(result, book_url, meta, refs)


async def preview_book_with_browser(result: SearchResult, *, browser) -> BookPreview:
    book_url = parser._resolve_book_url(result.url)
    parsed = urlparse(book_url)
    m = parser.INFO_RE.match(parsed.path)
    if not m:
        raise ValueError(f"not a pili45 info URL: {book_url}")
    cat_id, book_id = m.group(1), m.group(2)
    tab = await browser.get(book_url, new_tab=True)
    await _wait_for_info_page(tab)
    meta, refs = await _fetch_preview_data(
        book_url,
        cat_id,
        book_id,
        get_html=lambda url: _get_html_with_tab(tab, url),
        first_html=await tab.get_content(),
    )
    return _make_preview(result, book_url, meta, refs)


async def _wait_for_info_page(tab) -> None:
    await wait_for_page_ready(
        tab,
        ready_selector=".works-intro-title, .works-cover, .works-intro-short",
    )


async def _fetch_preview_data(
    book_url: str,
    cat_id: str,
    book_id: str,
    *,
    get_html,
    first_html: str | None = None,
) -> tuple[parser.BookMeta, list[parser.ChapterRef]]:
    PROGRESS.provider_detail("pili45", f"fetching info {book_url}")
    meta = parser.parse_info(first_html or await get_html(book_url), cat_id, book_id)
    refs: list[parser.ChapterRef] = []
    menu_url: str | None = urljoin(parser.HOST, f"/{cat_id}/{book_id}/menu/1.html")
    while menu_url:
        PROGRESS.provider_detail("pili45", f"fetching menu {menu_url}")
        page_refs, next_url = parser.parse_toc(
            await get_html(menu_url), cat_id, book_id
        )
        refs.extend(page_refs)
        menu_url = next_url
    return meta, refs


async def _get_html_with_tab(tab, url: str) -> str:
    await tab.get(url)
    await _wait_for_search_page(tab)
    return await tab.get_content()


def _make_preview(
    result: SearchResult,
    book_url: str,
    meta: parser.BookMeta,
    refs: list[parser.ChapterRef],
) -> BookPreview:
    titles = [ref.title for ref in refs]
    title = meta.title if meta.title and meta.title != "未命名" else result.title
    author = meta.author or result.author
    return BookPreview(
        parser="pili45",
        title=title,
        author=author,
        url=book_url,
        chapter_count=len(titles),
        first_chapters=tuple(titles[:2]),
        last_chapters=tuple(titles[-2:]),
        intro="\n".join(meta.intro_paragraphs[:2]),
        status=meta.status,
        source=result.source,
    )
