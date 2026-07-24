from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ebooklib import epub

from src.core.epub_normalizer import normalize_new_epub
from src.core.epub_writer import cover_extension, render_paragraphs
from src.core.output import resolve_output_path
from src.crawl.snapshot import load_chapter, load_manifest, snapshot_chapter_ids

BILINGUAL_CSS = """
body { font-family: serif; line-height: 1.7; margin: 1em; }
h1 { font-size: 1.6em; text-align: center; margin: 1em 0 0.6em; }
h2 { font-size: 1.3em; text-align: center; margin: 1em 0 0.4em; }
h3 { font-size: 1.1em; margin: 0.8em 0 0.3em; }
p  { text-indent: 2em; margin: 0.3em 0; }
.intro p { text-indent: 0; }
.scene-break { text-indent: 0; text-align: center; margin: 1em 0; }
.zh-translation { color: #333; }
hr { border: 0; border-top: 1px solid #999; margin: 1.2em 20%; }
"""

HTML_HEAD = (
    '<html xmlns="http://www.w3.org/1999/xhtml" lang="zh-CN">'
    "<head><meta charset='utf-8'/><title>{title}</title>"
    "<link rel='stylesheet' type='text/css' href='style/main.css'/></head><body>"
)
HTML_TAIL = "</body></html>"


def _translation_paragraph(text: str) -> str:
    return (
        '<p class="zh-translation" xml:lang="zh-CN">'
        + html.escape(text, quote=False)
        + "</p>"
    )


def _chapter_blocks(chapter: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = chapter.get("blocks")
    if isinstance(blocks, list) and blocks:
        return blocks
    output = []
    for item in chapter.get("paragraphs") or []:
        text = str(item.get("english") or "")
        output.append(
            {
                "type": "content",
                "index": int(item["index"]),
                "english": text,
                "html": f"<p>{html.escape(text, quote=False)}</p>",
            }
        )
    return output


def _load_translations(translations_dir: Path, chapter_id: str) -> dict[int, str]:
    path = translations_dir / f"{chapter_id}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(item["index"]): str(item.get("zh") or "").strip()
        for item in data.get("translations", [])
        if str(item.get("zh") or "").strip()
    }


def build_bilingual_epub(
    *,
    snapshot_dir: Path,
    translations_dir: Path,
    output: Path | None,
    title: str | None = None,
    author: str | None = None,
) -> Path:
    manifest = load_manifest(snapshot_dir)
    book_title = title or str(manifest.get("title") or "book")
    book_author = author or str(manifest.get("author") or "")
    out_path = resolve_output_path(output, book_title, book_author)

    book = epub.EpubBook()
    book.set_identifier(f"bilingual-{manifest.get('source', {}).get('provider', 'crawl')}-{book_title}")
    book.set_title(book_title)
    book.set_language("zh-CN")
    if book_author:
        book.add_author(book_author)

    css = epub.EpubItem(
        uid="style_main",
        file_name="style/main.css",
        media_type="text/css",
        content=BILINGUAL_CSS,
    )
    book.add_item(css)

    cover = manifest.get("cover") or {}
    cover_path = cover.get("path")
    if cover_path:
        asset_path = snapshot_dir / str(cover_path)
        if asset_path.exists():
            cover_mime = str(cover.get("mime") or "image/jpeg")
            book.set_cover(f"cover{cover_extension(cover_mime)}", asset_path.read_bytes())

    intro_html = render_paragraphs(manifest.get("intro_paragraphs") or [])
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
    for order, chapter_id in enumerate(snapshot_chapter_ids(manifest), 1):
        chapter = load_chapter(snapshot_dir, manifest, chapter_id)
        chapter_title = str(chapter.get("title") or f"Chapter {order:03d}")
        translations = _load_translations(translations_dir, chapter_id)
        body_parts = [f"<h2>{html.escape(chapter_title, quote=False)}</h2>"]
        for block in _chapter_blocks(chapter):
            block_html = str(block.get("html") or "")
            if block_html:
                body_parts.append(block_html)
            if block.get("type") == "content":
                zh = translations.get(int(block["index"]))
                if zh:
                    body_parts.append(_translation_paragraph(zh))
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

    volume_title = str(manifest.get("volume_title") or "")
    if volume_title and chapter_items:
        toc.append((epub.Section(volume_title), tuple(chapter_items)))
    else:
        toc.extend(chapter_items)

    book.toc = tuple(toc)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    epub.write_epub(str(out_path), book, {})
    normalize_new_epub(out_path)
    return out_path
