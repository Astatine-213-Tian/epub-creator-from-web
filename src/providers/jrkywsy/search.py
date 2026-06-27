from __future__ import annotations

import re
from urllib.parse import quote, urljoin

import requests
from opencc import OpenCC

from src.search import BookPreview, SearchResult
from . import parser


BLOG_HOST = "jrkywsy.blog.fc2.com"
HOST = f"http://{BLOG_HOST}"
CATALOG_TITLE_RE = re.compile(r"^(目錄|目录)(?:\b|[-（(]|$)", flags=re.I)


def search_books(query: str, *, limit: int = 10, author: str | None = None) -> list[SearchResult]:
    session = requests.Session()
    search_query = OpenCC("s2t").convert(query)
    url = f"{HOST}/?q={quote(search_query)}"
    soup = parser.BeautifulSoup(parser.fetch(url), "lxml")
    results: list[SearchResult] = []
    for li in soup.find_all("li"):
        links = [
            a for a in li.find_all("a", href=True)
            if re.search(r"blog-entry-\d+\.html", a["href"])
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
        parsed_title, author = parser.guess_title_author(title)
        snippet = re.sub(r"\s+", " ", li.get_text(" ", strip=True))
        if _is_catalog_result(parsed_title, title, snippet):
            continue
        if not _is_relevant_result(query, parsed_title, author):
            continue
        results.append(
            SearchResult(
                parser="jrkywsy",
                title=parsed_title,
                author=author,
                url=urljoin(HOST, a["href"]),
                source="jrkywsy FC2 search",
                snippet=snippet,
            )
        )
        if len(results) >= limit:
            break
    return results


def _is_catalog_result(parsed_title: str, raw_title: str, snippet: str) -> bool:
    title = parsed_title.strip() or raw_title.strip()
    if CATALOG_TITLE_RE.search(title):
        return True
    return "目錄" in raw_title or "目录" in raw_title or "目錄" in snippet or "目录" in snippet


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


def preview_book(result: SearchResult) -> BookPreview:
    meta, volumes = parser.crawl_book(result.url)
    chapter_titles = [chapter.title for volume in volumes for chapter in volume.chapters]
    title = _base_title(meta.title)
    return BookPreview(
        parser="jrkywsy",
        title=title,
        author=meta.author,
        url=result.url,
        chapter_count=len(chapter_titles),
        first_chapters=tuple(chapter_titles[:2]),
        last_chapters=tuple(chapter_titles[-2:]),
        intro="\n".join(meta.intro_paragraphs[:2]),
        source=result.source,
    )


def _base_title(title: str) -> str:
    return re.sub(r"\s*[（(][上中下][)）]\s*$", "", title).strip() or title
