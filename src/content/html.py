"""Content-aware cleanup of crawler paragraphs before source publication."""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from src.content.models import Chapter

COMMA_SPACE_RE = re.compile(r"([，,])[\t ]+")
COMMA_PARAGRAPH_BREAK_RE = re.compile(r"([，,])\s*</p>\s*<p>\s*")
SCENE_BREAK_RE = re.compile(r"^(?:\*\s*){3,}$")


def escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def normalize_comma_spacing(text: str) -> str:
    return COMMA_SPACE_RE.sub(r"\1", text)


def normalize_comma_html_breaks(value: str) -> str:
    return COMMA_PARAGRAPH_BREAK_RE.sub(r"\1", normalize_comma_spacing(value))


def is_scene_break_text(text: str) -> bool:
    return bool(SCENE_BREAK_RE.fullmatch(text.strip()))


def render_scene_break() -> str:
    return '<p class="scene-break">***</p>'


def normalize_paragraphs(paragraphs: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for paragraph in paragraphs:
        paragraph = normalize_comma_spacing(paragraph)
        if is_scene_break_text(paragraph):
            normalized.append("***")
        elif (
            normalized
            and not is_scene_break_text(normalized[-1])
            and normalized[-1].rstrip().endswith(("，", ","))
        ):
            normalized[-1] = normalized[-1].rstrip() + paragraph.lstrip()
        else:
            normalized.append(paragraph)
    return normalized


def render_paragraph(paragraph: str) -> str:
    if is_scene_break_text(paragraph):
        return render_scene_break()
    return f"<p>{escape_text(paragraph)}</p>"


def render_paragraphs(paragraphs: Iterable[str]) -> str:
    return "\n".join(
        render_paragraph(paragraph) for paragraph in normalize_paragraphs(paragraphs)
    )


def render_chapter_body(chapter: Chapter) -> str:
    if chapter.html_blocks is not None:
        return "\n".join(chapter.html_blocks)
    return render_paragraphs(chapter.paragraphs)
