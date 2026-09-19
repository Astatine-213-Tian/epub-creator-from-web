"""Write prepared bilingual chapters; source and translation files stay outside the renderer."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from ebooklib import epub

from src.content.models import Chapter
from src.content.styles import CSS

BILINGUAL_CSS = (
    CSS
    + "\n.scene-break { text-indent: 0; text-align: center; margin: 1em 0; }\n"
    + """
.zh-translation { color: #333; }
hr { border: 0; border-top: 1px solid #999; margin: 1.2em 20%; }
"""
)
HTML_HEAD = (
    '<html xmlns="http://www.w3.org/1999/xhtml" lang="zh-CN">'
    "<head><meta charset='utf-8'/><title>{title}</title>"
    "<link rel='stylesheet' type='text/css' href='style/main.css'/></head><body>"
)
HTML_TAIL = "</body></html>"


def create_bilingual_book(
    *,
    identifier: str,
    title: str,
    author: str,
    output: Path,
    intro_html: str,
    chapters: list[Chapter],
    volume_title: str = "",
    cover: bytes | None = None,
    cover_mime: str = "image/jpeg",
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
        content=BILINGUAL_CSS,
    )
    book.add_item(css)

    if cover:
        extension = {"image/png": "png", "image/gif": "gif", "image/webp": "webp"}.get(
            cover_mime, "jpg"
        )
        book.set_cover(f"cover.{extension}", cover, create_page=False)

    intro_body = (
        HTML_HEAD.format(title="简介")
        + "<div class='intro'><h1>简介</h1>"
        + intro_html
        + "</div>"
        + HTML_TAIL
    )
    intro_item = epub.EpubHtml(
        title="简介",
        file_name="intro.xhtml",
        lang="zh-CN",
        content=intro_body,
    )
    intro_item.add_item(css)
    book.add_item(intro_item)

    spine: list[Any] = ["nav", intro_item]
    toc: list[Any] = [intro_item]
    chapter_items: list[epub.EpubHtml] = []
    for order, chapter in enumerate(chapters, 1):
        chapter_title = chapter.title
        body_parts = [
            f"<h2>{html.escape(chapter_title, quote=False)}</h2>",
            *(chapter.html_blocks or []),
        ]
        content = (
            HTML_HEAD.format(title=html.escape(chapter_title, quote=False))
            + "\n".join(body_parts)
            + HTML_TAIL
        )
        chapter_item = epub.EpubHtml(
            title=chapter_title,
            file_name=f"chap_01_{order:03d}.xhtml",
            lang="zh-CN",
            content=content,
        )
        chapter_item.add_item(css)
        book.add_item(chapter_item)
        chapter_items.append(chapter_item)
        spine.append(chapter_item)

    if volume_title and chapter_items:
        toc.append((epub.Section(volume_title), tuple(chapter_items)))
    else:
        toc.extend(chapter_items)

    book.toc = tuple(toc)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    output.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(output), book, {})
