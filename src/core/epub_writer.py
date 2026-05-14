from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

from ebooklib import epub

from .models import Volume


HTML_HEAD = (
    '<html xmlns="http://www.w3.org/1999/xhtml" lang="zh-CN">'
    "<head><meta charset='utf-8'/><title>{title}</title>"
    "<link rel='stylesheet' type='text/css' href='style/main.css'/></head><body>"
)
HTML_TAIL = "</body></html>"

CSS = """
body { font-family: serif; line-height: 1.7; margin: 1em; }
h1 { font-size: 1.6em; text-align: center; margin: 1em 0 0.6em; }
h2 { font-size: 1.3em; text-align: center; margin: 1em 0 0.4em; }
h3 { font-size: 1.1em; margin: 0.8em 0 0.3em; }
p  { text-indent: 2em; margin: 0.3em 0; }
.intro p { text-indent: 0; }
"""

COMMA_SPACE_RE = re.compile(r"([，,])[\t ]+")
COMMA_PARAGRAPH_BREAK_RE = re.compile(r"([，,])\s*</p>\s*<p>\s*")


def escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def normalize_comma_spacing(text: str) -> str:
    return COMMA_SPACE_RE.sub(r"\1", text)


def normalize_comma_html_breaks(value: str) -> str:
    return COMMA_PARAGRAPH_BREAK_RE.sub(r"\1", normalize_comma_spacing(value))


def normalize_paragraphs(paragraphs: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for paragraph in paragraphs:
        paragraph = normalize_comma_spacing(paragraph)
        if normalized and normalized[-1].rstrip().endswith(("，", ",")):
            normalized[-1] = normalized[-1].rstrip() + paragraph.lstrip()
        else:
            normalized.append(paragraph)
    return normalized


def render_paragraphs(paragraphs: Iterable[str]) -> str:
    return "\n".join(f"<p>{escape_text(paragraph)}</p>" for paragraph in normalize_paragraphs(paragraphs))


def cover_extension(cover_mime: str) -> str:
    return {
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }.get(cover_mime, ".jpg")


def write_epub(
    *,
    identifier: str,
    title: str,
    author: str,
    volumes: list[Volume],
    out_path: Path,
    intro_html: str = "",
    intro_paragraphs: Iterable[str] | None = None,
    cover_bytes: bytes | None = None,
    cover_mime: str = "image/jpeg",
    emit_single_volume_cover: bool = True,
) -> None:
    book = epub.EpubBook()
    book.set_identifier(identifier)
    book.set_title(title)
    book.set_language("zh-CN")
    if author:
        book.add_author(author)

    css = epub.EpubItem(
        uid="style_main",
        file_name="style/main.css",
        media_type="text/css",
        content=CSS,
    )
    book.add_item(css)

    if cover_bytes:
        book.set_cover(f"cover{cover_extension(cover_mime)}", cover_bytes)

    if intro_paragraphs is not None:
        intro_html = render_paragraphs(intro_paragraphs)
    else:
        intro_html = normalize_comma_html_breaks(intro_html)

    intro_body = (
        HTML_HEAD.format(title="简介")
        + "<div class='intro'><h1>简介</h1>"
        + intro_html
        + "</div>"
        + HTML_TAIL
    )
    intro_item = epub.EpubHtml(
        title="简介", file_name="intro.xhtml", lang="zh-CN", content=intro_body
    )
    intro_item.add_item(css)
    book.add_item(intro_item)

    spine: list = ["nav", intro_item]
    toc: list = [intro_item]

    for vi, vol in enumerate(volumes, 1):
        chapter_items: list[epub.EpubHtml] = []
        for ci, chapter in enumerate(vol.chapters, 1):
            file_name = f"chap_{vi:02d}_{ci:03d}.xhtml"
            content = (
                HTML_HEAD.format(title=escape_text(chapter.title))
                + f"<h2>{escape_text(chapter.title)}</h2>"
                + render_paragraphs(chapter.paragraphs)
                + HTML_TAIL
            )
            chapter_item = epub.EpubHtml(
                title=chapter.title,
                file_name=file_name,
                lang="zh-CN",
                content=content,
            )
            chapter_item.add_item(css)
            book.add_item(chapter_item)
            spine.append(chapter_item)
            chapter_items.append(chapter_item)

        should_emit_volume = (
            vol.title
            and chapter_items
            and (emit_single_volume_cover or len(volumes) > 1)
        )
        if should_emit_volume:
            vol_file = f"vol_{vi:02d}.xhtml"
            vol_content = (
                HTML_HEAD.format(title=escape_text(vol.title))
                + f"<h1>{escape_text(vol.title)}</h1>"
                + HTML_TAIL
            )
            vol_item = epub.EpubHtml(
                title=vol.title, file_name=vol_file, lang="zh-CN", content=vol_content
            )
            vol_item.add_item(css)
            book.add_item(vol_item)
            spine.insert(spine.index(chapter_items[0]), vol_item)
            toc.append((epub.Section(vol.title, href=vol_file), tuple(chapter_items)))
        else:
            toc.extend(chapter_items)

    book.toc = tuple(toc)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    epub.write_epub(str(out_path), book, {})
