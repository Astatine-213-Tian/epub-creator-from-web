"""Search result and lightweight book preview contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    parser: str
    title: str
    author: str
    url: str
    source: str
    snippet: str = ""
    raw_score: float = 0.0


@dataclass(frozen=True)
class BookPreview:
    parser: str
    title: str
    author: str
    url: str
    chapter_count: int | None
    first_chapters: tuple[str, ...]
    last_chapters: tuple[str, ...]
    intro: str = ""
    status: str = ""
    source: str = ""
    match_level: int = 0
