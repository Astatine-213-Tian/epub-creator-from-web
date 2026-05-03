from __future__ import annotations

import asyncio
import html
import re
from urllib.parse import urljoin, urlparse

from src.fetch.browser import wait_for_page_ready
from src.runtime.progress import ProgressLogger
from src.search.engines import site_search
from src.search import BookPreview, SearchResult
from . import parser


PROGRESS = ProgressLogger()


def search_books(query: str, *, limit: int = 10) -> list[SearchResult]:
    # The live xfxs search form currently posts to /search/ but returns a 404
    # page. Use a site-scoped external search and keep only canonical book URLs.
    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    PROGRESS.provider_detail("xfxs", "native search is unavailable; using external site search")
    for item in site_search(
        query,
        site="xfxs1.com",
        path_prefix="/goreadbook/",
        limit=limit,
    ):
        title, author = _parse_result_title(item.title)
        book_url = _canonical_book_url(item.url)
        if book_url is None:
            continue
        if book_url in seen_urls:
            continue
        seen_urls.add(book_url)
        results.append(
            SearchResult(
                parser="xfxs",
                title=title,
                author=author,
                url=book_url,
                source=f"xfxs {item.engine} site search",
                snippet=item.description,
                raw_score=float(limit - len(results)),
            )
        )
    return results


def _canonical_book_url(url: str) -> str | None:
    parsed = urlparse(url)
    m = parser.BOOK_INDEX_RE.search(parsed.path.rstrip("/") + "/")
    if m:
        return f"{parser.HOST}/goreadbook/{m.group(1)}/"
    m = parser.CHAPTER_RE.search(parsed.path)
    if m:
        return f"{parser.HOST}/goreadbook/{m.group(1)}/"
    return f"{parser.HOST}/goreadbook/{m.group(1)}/"


def _parse_result_title(title: str) -> tuple[str, str]:
    title = re.sub(r"\s+", " ", html.unescape(title)).strip()
    title = re.sub(r"\s*[-_]\s*先锋小说网\s*$", "", title)
    title = re.sub(r"最新章节.*$|全文阅读.*$|免费全文阅读.*$", "", title).strip()
    author = ""
    m = re.search(r"[（(]([^()（）]+)[)）]", title)
    if m:
        author = m.group(1).strip()
        title = (title[: m.start()] + title[m.end() :]).strip()
    title = title.strip(" _-：:")
    return title or "未命名", author


def preview_book(result: SearchResult) -> BookPreview:
    return asyncio.run(_preview_book(result))


async def _preview_book(result: SearchResult) -> BookPreview:
    book_url = parser._resolve_book_url(result.url)
    parsed = urlparse(book_url)
    m = parser.BOOK_INDEX_RE.search(parsed.path.rstrip("/") + "/")
    if not m:
        raise ValueError(f"not an xfxs book index URL: {book_url}")
    book_id = m.group(1)

    fetcher = parser.Fetcher(headless=False, delay=0.1)
    PROGRESS.provider_detail("xfxs", "starting browser for preview")
    await fetcher.start()
    try:
        meta, refs = await _fetch_preview_data(book_id, get_html=fetcher.get_html)
    finally:
        PROGRESS.provider_detail("xfxs", "closing preview browser")
        await fetcher.stop()

    return _make_preview(result, book_url, meta, refs)


async def preview_book_with_browser(result: SearchResult, *, browser) -> BookPreview:
    book_url = parser._resolve_book_url(result.url)
    parsed = urlparse(book_url)
    m = parser.BOOK_INDEX_RE.search(parsed.path.rstrip("/") + "/")
    if not m:
        raise ValueError(f"not an xfxs book index URL: {book_url}")
    book_id = m.group(1)
    tab = await browser.get(book_url, new_tab=True)
    meta, refs = await _fetch_preview_data(
        book_id,
        get_html=lambda url: _get_html_with_tab(tab, url),
        first_html=await tab.get_content(),
    )
    return _make_preview(result, book_url, meta, refs)


async def _fetch_preview_data(
    book_id: str,
    *,
    get_html,
    first_html: str | None = None,
) -> tuple[parser.BookMeta, list[parser.ChapterRef]]:
    index_url = urljoin(parser.HOST, f"/goreadbook/{book_id}/")
    PROGRESS.provider_detail("xfxs", f"fetching index {index_url}")
    meta = parser.parse_book_index(
        first_html or await get_html(index_url),
        book_id,
    )
    refs: list[parser.ChapterRef] = []
    toc_url: str | None = urljoin(parser.HOST, f"/2/{book_id}/")
    seen: set[str] = set()
    while toc_url:
        PROGRESS.provider_detail("xfxs", f"fetching toc {toc_url}")
        page_refs, next_url = parser.parse_toc(await get_html(toc_url), book_id)
        for ref in page_refs:
            if ref.chapter_id in seen:
                continue
            seen.add(ref.chapter_id)
            refs.append(ref)
        toc_url = next_url
    return meta, refs


async def _get_html_with_tab(tab, url: str) -> str:
    await tab.get(url)
    await wait_for_page_ready(tab, settle_delay=0.1)
    return await tab.get_content()


def _make_preview(
    result: SearchResult,
    book_url: str,
    meta: parser.BookMeta,
    refs: list[parser.ChapterRef],
) -> BookPreview:
    titles = [ref.title for ref in refs]
    return BookPreview(
        parser="xfxs",
        title=meta.title,
        author=meta.author,
        url=book_url,
        chapter_count=len(titles),
        first_chapters=tuple(titles[:2]),
        last_chapters=tuple(titles[-2:]),
        intro="\n".join(meta.intro_paragraphs[:2]),
        status=meta.status,
        source=result.source,
    )
