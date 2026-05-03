from __future__ import annotations

import re
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from opencc import OpenCC

from src.search import BookPreview, SearchResult
from . import parser


CATALOG_TITLE_RE = re.compile(r"^(目錄|目录)(?:\b|[-（(]|$)", flags=re.I)


def search_books(query: str, *, limit: int = 10) -> list[SearchResult]:
    session = requests.Session()
    search_query = OpenCC("s2t").convert(query)
    url = f"http://{parser.BLOG_HOST}/?q={quote(search_query)}"
    soup = BeautifulSoup(parser.fetch(url, session), "lxml")
    results: list[SearchResult] = []
    for li in soup.find_all("li"):
        links = [
            a for a in li.find_all("a", href=True)
            if parser.ENTRY_RE.search(a["href"]) or parser.NO_RE.search(a["href"])
        ]
        if not links:
            continue
        a = links[0]
        title = a.get_text(" ", strip=True)
        if not title:
            h = li.find(["h2", "h3"])
            title = h.get_text(" ", strip=True) if h else ""
        if not title:
            continue
        snippet = re.sub(r"\s+", " ", li.get_text(" ", strip=True))
        if _is_catalog_result(title, snippet, a["href"]):
            continue
        author = _author_from_title(title)
        if not _is_relevant_result(query, title, author):
            continue
        results.append(
            SearchResult(
                parser="towasakata",
                title=title,
                author=author,
                url=parser.normalize_url(a["href"]),
                source="towasakata FC2 search",
                snippet=snippet,
            )
        )
        if len(results) >= limit:
            break
    return results


def preview_book(result: SearchResult) -> BookPreview:
    session = requests.Session()
    cc = OpenCC("t2s")
    first_url = parser.normalize_url(result.url)
    first = parser.parse_page(first_url, parser.fetch(first_url, session))
    urls = first.intro_links or [first.url]
    pages: list[parser.VolumePage] = []
    seen_ids: set[str] = set()
    for url in urls:
        eid = parser.entry_id(url) or url
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        if eid == first.entry_id:
            pages.append(first)
        else:
            time.sleep(0.1)
            pages.append(parser.parse_page(url, parser.fetch(url, session)))

    title, author = parser.guess_title_author(first.page_title, cc)
    volumes = parser.split_into_volumes(pages)
    chapter_titles = [
        cc.convert(chapter.title)
        for volume in volumes
        for chapter in volume.chapters
    ]
    return BookPreview(
        parser="towasakata",
        title=title or cc.convert(result.title),
        author=author,
        url=first.url,
        chapter_count=len(chapter_titles),
        first_chapters=tuple(chapter_titles[:2]),
        last_chapters=tuple(chapter_titles[-2:]),
        intro=BeautifulSoup(parser.render_intro_html(first.intro_html, cc), "lxml").get_text(" "),
        source=result.source,
    )


def _author_from_title(title: str) -> str:
    m = re.search(r"\s+by\s+(.+?)\s*$", title, flags=re.I)
    return m.group(1).strip() if m else ""


def _is_catalog_result(title: str, snippet: str, url: str) -> bool:
    normalized_title = title.strip()
    if CATALOG_TITLE_RE.search(normalized_title):
        return True
    entry_id = parser.entry_id(url)
    if entry_id == "49":
        return True
    normalized_snippet = snippet.casefold()
    return (
        normalized_title.casefold().startswith(("updated to", "目錄", "目录"))
        or "updated to" in normalized_snippet and "blog-entry-49" in url
    )


def _is_relevant_result(query: str, title: str, author: str) -> bool:
    normalized_query = _normalize_search_text(query)
    if not normalized_query:
        return True
    return (
        normalized_query in _normalize_search_text(title)
        or normalized_query in _normalize_search_text(author)
    )


def _normalize_search_text(value: str) -> str:
    try:
        value = OpenCC("t2s").convert(value)
    except Exception:
        pass
    return re.sub(r"[\s\xa0　\W_]+", "", value).casefold()
