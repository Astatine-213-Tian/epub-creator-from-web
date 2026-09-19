"""Content and formatting comparison shared by source adapters and exporters."""

from __future__ import annotations


def content_signature(blocks: list[dict]) -> list[tuple]:
    """Compare editable presentation, excluding local archive bookkeeping."""
    # Markdown moves whitespace outside emphasis. Its styling has no visible
    # effect, while every character and non-whitespace style must be preserved.
    return [
        (
            b["kind"],
            b.get("level", 3) if b["kind"] == "heading" else None,
            b.get("alignment", "left"),
            tuple(
                (char, tuple(sorted(r["styles"])) if not char.isspace() else ())
                for r in b["runs"]
                for char in r["text"]
            ),
        )
        for b in blocks
    ]
