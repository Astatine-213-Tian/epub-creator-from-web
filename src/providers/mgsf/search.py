from __future__ import annotations

import asyncio
from urllib.parse import quote, urljoin, urlparse

from bs4 import BeautifulSoup

from src.fetch.browser import start_browser, wait_for_page_ready
from src.runtime.progress import ProgressLogger
from src.search import BookPreview, SearchResult
from . import parser


PROGRESS = ProgressLogger()


def search_books(query: str, *, limit: int = 10) -> list[SearchResult]:
    url = f"{parser.HOST}/search/1/{quote(query)}/1.html"
    return _parse_search_html(_get_search_html(url), limit=limit)


async def search_books_with_browser(query: str, *, limit: int = 10, browser) -> list[SearchResult]:
    url = f"{parser.HOST}/search/1/{quote(query)}/1.html"
    PROGRESS.provider_detail("mgsf", f"browser fetching {url}")
    tab = await browser.get(url, new_tab=True)
    html = await _wait_for_tab_html(tab)
    return _parse_search_html(html, limit=limit)


def _parse_search_html(html: str, *, limit: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "lxml")
    results: list[SearchResult] = []
    for a in soup.find_all("a", href=True):
        path = urlparse(a["href"]).path
        if not parser.INFO_RE.match(path):
            continue
        item = a.find_parent("li") or a.find_parent("div") or a
        title_node = item.find(["h3", "h4", "h5"]) if item else None
        title = parser._clean((title_node or a).get_text())
        if not title:
            continue
        snippet = parser._clean(item.get_text(" ")) if item else ""
        results.append(
            SearchResult(
                parser="mgsf",
                title=title,
                author="",
                url=urljoin(parser.HOST, a["href"]),
                source="mgsf native search",
                snippet=snippet,
            )
        )
        if len(results) >= limit:
            break
    return results


def _get_search_html(url: str) -> str:
    fetcher = parser.Fetcher(delay=0.1)
    try:
        PROGRESS.provider_detail("mgsf", f"fetching search page {url}")
        return fetcher.get_html(url)
    except Exception:
        PROGRESS.provider_detail("mgsf", "raw search blocked; starting browser fallback")
        return asyncio.run(_get_html_with_browser(url))


async def _get_html_with_browser(url: str) -> str:
    PROGRESS.provider_detail("mgsf", "starting browser for search fallback")
    browser = await start_browser()
    try:
        PROGRESS.provider_detail("mgsf", f"browser fetching {url}")
        page = await browser.get(url)
        return await _wait_for_tab_html(page)
    finally:
        PROGRESS.provider_detail("mgsf", "closing search fallback browser")
        await browser.stop()


async def _wait_for_tab_html(tab) -> str:
    await wait_for_page_ready(tab)
    return await tab.get_content()


def preview_book(result: SearchResult) -> BookPreview:
    book_url = parser._resolve_book_url(result.url)
    parsed = urlparse(book_url)
    m = parser.INFO_RE.match(parsed.path)
    if not m:
        raise ValueError(f"not a mgsf info URL: {book_url}")
    cat_id, book_id = m.group(1), m.group(2)

    fetcher = parser.Fetcher(delay=0.1)
    PROGRESS.provider_detail("mgsf", f"fetching info {book_url}")
    meta = parser.parse_info(fetcher.get_html(book_url), cat_id, book_id)
    refs: list[parser.ChapterRef] = []
    menu_url: str | None = urljoin(parser.HOST, f"/{cat_id}/{book_id}/menu/1.html")
    while menu_url:
        PROGRESS.provider_detail("mgsf", f"fetching menu {menu_url}")
        page_refs, next_url = parser.parse_menu(fetcher.get_html(menu_url), cat_id, book_id)
        refs.extend(page_refs)
        menu_url = next_url

    titles = [ref.raw_title for ref in refs]
    return BookPreview(
        parser="mgsf",
        title=meta.title,
        author=meta.author,
        url=book_url,
        chapter_count=len(titles),
        first_chapters=tuple(titles[:2]),
        last_chapters=tuple(titles[-2:]),
        intro="\n".join(meta.intro_paragraphs[:2]),
        source=result.source,
    )
