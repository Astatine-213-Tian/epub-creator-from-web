from __future__ import annotations

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.search import BookPreview, SearchResult

from . import parser


def search_books(query: str, *, limit: int = 10, author: str | None = None) -> list[SearchResult]:
    fetcher = parser.Fetcher(delay=0.1)
    url = f"{parser.HOST}/search?name={query}&type=3"
    soup = BeautifulSoup(fetcher.get_html(url), "lxml")
    results: list[SearchResult] = []
    for a in soup.select("a[href^='/novel/']"):
        href = urljoin(parser.HOST, a["href"])
        if not parser.DETAIL_RE.match(urlparse(href).path):
            continue
        title = parser._clean_title(a.get_text(" "))
        if not title:
            continue
        results.append(
            SearchResult(
                parser="ibusread",
                title=title,
                author="",
                url=href,
                source="ibusread search",
            )
        )
        if len(results) >= limit:
            break
    return results


def preview_book(result: SearchResult) -> BookPreview:
    book_url = parser._resolve_book_url(result.url)
    fetcher = parser.Fetcher(delay=0.1)
    if "/novel/chapter/" in book_url:
        html = fetcher.get_html(book_url)
        soup = BeautifulSoup(html, "lxml")
        page = soup.select_one(".js_page_novel_chapter")
        if not page:
            raise ValueError("chapter metadata not found")
        book_id = page.get("data-id", "")
        title = parser._clean_title(page.get("data-name", ""))
        author = parser._clean_text(page.get("data-auth", ""))
    else:
        book_id = urlparse(book_url).path.rsplit("/", 1)[-1]
        title, author, _category, _first = parser._parse_detail(fetcher.get_html(book_url))
    refs, total = parser.parse_catalogs(fetcher, book_id)
    titles = [ref.title for ref in refs]
    return BookPreview(
        parser="ibusread",
        title=title,
        author=author,
        url=book_url,
        chapter_count=total or len(refs),
        first_chapters=tuple(titles[:2]),
        last_chapters=tuple(titles[-2:]),
        source=result.source,
    )
