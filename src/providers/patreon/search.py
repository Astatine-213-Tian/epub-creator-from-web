from __future__ import annotations

from urllib.parse import urlparse

from src.providers.patreon.parser import (
    DEFAULT_AUTHOR,
    DEFAULT_TITLE,
    _collection_id_from_url,
    _resolve_book_url,
)
from src.search.engines import site_search
from src.search.orchestrator import BookPreview, SearchResult


def search_books(query: str, *, limit: int = 10) -> list[SearchResult]:
    results: list[SearchResult] = []
    seen: set[str] = set()
    for item in site_search(
        query,
        site="patreon.com",
        path_prefix="/collection/",
        limit=limit,
    ):
        try:
            url = _resolve_book_url(item.url)
        except ValueError:
            continue
        if url in seen:
            continue
        seen.add(url)
        results.append(
            SearchResult(
                parser="patreon",
                title=item.title or DEFAULT_TITLE,
                author=DEFAULT_AUTHOR,
                url=url,
                source=f"patreon {item.engine} site search",
                snippet=item.description,
                raw_score=float(limit - len(results)),
            )
        )
    return results


def preview_book(result: SearchResult) -> BookPreview:
    collection_id = _collection_id_from_url(result.url)
    parsed = urlparse(result.url)
    return BookPreview(
        parser="patreon",
        title=DEFAULT_TITLE,
        author=DEFAULT_AUTHOR,
        url=f"{parsed.scheme}://{parsed.netloc}/collection/{collection_id}?view=condensed",
        chapter_count=None,
        first_chapters=(),
        last_chapters=(),
        intro="Patreon collection; authenticated crawl is required for member-only chapters.",
        status="authenticated browser required",
        source=result.source,
        match_level=1,
    )
