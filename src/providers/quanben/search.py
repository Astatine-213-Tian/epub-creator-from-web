from __future__ import annotations

from urllib.parse import quote, urljoin, urlparse

from bs4 import BeautifulSoup

from src.search import BookPreview, SearchResult
from . import parser


def search_books(query: str, *, limit: int = 10, author: str | None = None) -> list[SearchResult]:
    fetcher = parser.Fetcher(delay=0.1)
    url = f"{parser.HOST}/index.php?c=book&a=search&keywords={quote(query)}"
    soup = BeautifulSoup(fetcher.get_html(url), "lxml")
    results: list[SearchResult] = []
    for a in soup.select("h3 a[href], .list2 h3 a[href]"):
        path = urlparse(a["href"]).path
        if not parser.BOOK_RE.match(path):
            continue
        item = a.find_parent("div")
        title = parser._clean_text(a.get_text())
        author = ""
        snippet = ""
        if item:
            text = item.get_text(" ")
            snippet = parser._clean_text(text)
            for p in item.find_all("p"):
                p_text = parser._clean_text(p.get_text())
                if p_text.startswith("作者"):
                    author = p_text.split(":", 1)[-1].split("：", 1)[-1].strip()
                    break
        results.append(
            SearchResult(
                parser="quanben",
                title=title,
                author=author,
                url=urljoin(parser.HOST, a["href"]),
                source="quanben native search",
                snippet=snippet,
            )
        )
        if len(results) >= limit:
            break
    return results


def preview_book(result: SearchResult) -> BookPreview:
    book_url = parser._resolve_book_url(result.url)
    parsed = urlparse(book_url)
    m = parser.LIST_RE.match(parsed.path) or parser.BOOK_RE.match(parsed.path)
    if not m:
        raise ValueError(f"not a quanben book URL: {book_url}")
    slug = m.group(1)
    fetcher = parser.Fetcher(delay=0.1)
    list_url = urljoin(parser.AMP_HOST, f"/amp/n/{slug}/list.html")
    meta, refs = parser.parse_list_page(fetcher.get_html(list_url), slug)
    titles = [ref.title for ref in refs]
    return BookPreview(
        parser="quanben",
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

