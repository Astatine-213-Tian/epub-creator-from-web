from __future__ import annotations

import asyncio
import html
import re
from urllib.parse import quote_from_bytes, quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.fetch.browser import wait_for_page_ready
from src.runtime.progress import ProgressLogger
from src.search.engines import UA, WebSearchResult, site_search, web_search
from src.search import BookPreview, SearchResult
from . import parser


PROGRESS = ProgressLogger()
AUTHOR_PAGE_TIMEOUT = 20
DUCKDUCKGO_SEARCH_URL = "https://duckduckgo.com/"
GOOGLE_SEARCH_URL = "https://www.google.com/search"


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
    if len(results) < limit:
        for result in _search_author_pages(query, limit=limit - len(results)):
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            results.append(result)
    return results


async def search_books_with_browser(query: str, *, limit: int = 10, browser) -> list[SearchResult]:
    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for item in await _browser_external_search(query, browser=browser, limit=limit):
        title, author = _parse_result_title(item.title)
        book_url = _canonical_book_url(item.url)
        if book_url is None or book_url in seen_urls:
            continue
        seen_urls.add(book_url)
        results.append(
            SearchResult(
                parser="xfxs",
                title=title,
                author=author,
                url=book_url,
                source=f"xfxs {item.engine} browser search",
                snippet=item.description,
                raw_score=float(limit - len(results)),
            )
        )
        if len(results) >= limit:
            return results

    if len(results) < limit:
        for result in _search_author_pages(query, limit=limit - len(results)):
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            results.append(result)
    return results


async def _browser_external_search(query: str, *, browser, limit: int) -> list[WebSearchResult]:
    items: list[WebSearchResult] = []
    seen: set[str] = set()
    for engine, url in (
        ("duckduckgo", f"{DUCKDUCKGO_SEARCH_URL}?q={quote_plus(query + ' xfxs1')}"),
        ("google", f"{GOOGLE_SEARCH_URL}?q={quote_plus(query + ' xfxs1')}&hl=zh-CN&num={limit}"),
    ):
        tab = await browser.get(url, new_tab=True)
        await wait_for_page_ready(tab, settle_delay=2.0)
        html_text = await tab.get_content()
        for item in _external_items_from_html(html_text, engine=engine):
            if item.url in seen:
                continue
            seen.add(item.url)
            items.append(item)
            if len(items) >= limit:
                return items
    return items


def _external_items_from_html(html_text: str, *, engine: str) -> list[WebSearchResult]:
    soup = BeautifulSoup(html_text, "lxml")
    items: list[WebSearchResult] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        title = a.get_text(" ", strip=True)
        if not href.startswith("http") or not title:
            continue
        if "xfxs1.com" not in href and "xfxs1.com" not in title:
            continue
        items.append(WebSearchResult(title=title, url=href, description="", engine=engine))
    return items


def _canonical_book_url(url: str) -> str | None:
    parsed = urlparse(url)
    m = parser.BOOK_INDEX_RE.search(parsed.path.rstrip("/") + "/")
    if m:
        return f"{parser.HOST}/goreadbook/{m.group(1)}/"
    m = parser.CHAPTER_RE.search(parsed.path)
    if m:
        return f"{parser.HOST}/goreadbook/{m.group(1)}/"
    m = re.search(r"/2/(\d+)/?$", parsed.path.rstrip("/") + "/")
    if m:
        return f"{parser.HOST}/goreadbook/{m.group(1)}/"
    return None


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


def _search_author_pages(query: str, *, limit: int) -> list[SearchResult]:
    authors = _discover_author_candidates(query)
    if not authors:
        return []

    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for author in authors:
        for title, url in _books_from_author_page(author):
            if url in seen_urls:
                continue
            if not _title_matches_query(title, query):
                continue
            seen_urls.add(url)
            results.append(
                SearchResult(
                    parser="xfxs",
                    title=title,
                    author=author,
                    url=url,
                    source="xfxs author page",
                    snippet=f"Matched from xfxs author page for {author}",
                    raw_score=float(limit - len(results)),
                )
            )
            if len(results) >= limit:
                return results
    return results


def _discover_author_candidates(query: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    search_queries = (
        f"{query} 作者",
        f"{query} 小说 作者",
        f"{query} 小说",
    )
    for search_query in search_queries:
        for item in web_search(search_query, limit=8, engines=("duckduckgo", "google")):
            for author in _authors_from_search_item(item, query):
                if author in seen:
                    continue
                seen.add(author)
                candidates.append(author)
                if len(candidates) >= 5:
                    return candidates
    return candidates


def _authors_from_search_item(item: WebSearchResult, query: str) -> list[str]:
    text = " ".join((item.title, item.description))
    patterns = (
        r"[（(]([^()（）]{1,20})[)）]",
        r"作者\s*[:：]\s*([^\s,，。|｜_《》()（）-]{1,20})",
        r"by\s*([^\s,，。|｜_《》()（）-]{1,20})",
        r"_([^_\s]{1,20})小说",
        r"《[^》]+》\s*([^_\s|｜-]{1,20})",
    )
    authors: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            author = _clean_author(match.group(1), query)
            if author:
                authors.append(author)
    return authors


def _clean_author(author: str, query: str) -> str:
    author = re.sub(r"\s+", "", html.unescape(author)).strip(" _-：:|｜")
    if not author or author == query:
        return ""
    bad_terms = ("最新章节", "全文阅读", "免费阅读", "小说", "在线阅读", "先锋")
    if any(term in author for term in bad_terms):
        return ""
    if len(author) > 12:
        return ""
    return author


def _books_from_author_page(author: str) -> list[tuple[str, str]]:
    try:
        encoded = quote_from_bytes(author.encode("gbk"))
    except UnicodeEncodeError:
        return []
    url = f"{parser.HOST}/a/{encoded}.html"
    try:
        response = requests.get(url, headers={"User-Agent": UA}, timeout=AUTHOR_PAGE_TIMEOUT)
        response.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(response.content.decode("gbk", errors="replace"), "lxml")
    books: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        book_url = _canonical_book_url(urljoin(parser.HOST, a["href"]))
        if not book_url or book_url in seen:
            continue
        title = parser._clean_text(a.get_text())
        if not title:
            continue
        seen.add(book_url)
        books.append((title, book_url))
    return books


def _title_matches_query(title: str, query: str) -> bool:
    title_norm = _normalize_title(title)
    query_norm = _normalize_title(query)
    return query_norm in title_norm or title_norm in query_norm


def _normalize_title(text: str) -> str:
    return re.sub(r"[\s　\xa0《》〈〉“”\"'‘’【】\[\]（）()·._\-—:：|｜]+", "", text)


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
