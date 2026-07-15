"""Shared helpers for book scraping scripts."""

from .core.models import Chapter, Volume

__all__ = ["Chapter", "Volume", "write_epub", "write_txt"]


def __getattr__(name: str):
    if name == "write_epub":
        from .core.epub_writer import write_epub

        return write_epub
    if name == "write_txt":
        from .core.text_writer import write_txt

        return write_txt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
