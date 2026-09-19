"""Create a new EPUB from prepared content, without interpreting its prose."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from ebooklib import epub
from lxml import etree as ET

from src.content.outline import ordered_members
from src.content.styles import CSS
from src.epub.archive import X, install_archive, validate_archive, validate_navigation
from src.epub.xhtml import render_blocks
from src.runtime.files import digest


def create_book(
    book: dict,
    output: Path,
    *,
    cover: bytes | None = None,
    cover_name: str = "cover.jpg",
) -> None:
    metadata = book["metadata"]
    package = epub.EpubBook()
    package.set_identifier(book["identifier"])
    package.set_title(metadata["title"])
    package.set_language(metadata["language"] or "zh-CN")
    package.add_author(metadata["creator"])
    for key in ("date", "source", "description"):
        if metadata.get(key):
            package.add_metadata("DC", key, metadata[key])
    for subject in metadata.get("subjects", []):
        package.add_metadata("DC", "subject", subject)
    if metadata.get("series"):
        package.add_metadata(
            None,
            "meta",
            metadata["series"],
            {"property": "belongs-to-collection", "id": "series"},
        )
        package.add_metadata(
            None,
            "meta",
            "series",
            {"property": "collection-type", "refines": "#series"},
        )
        if metadata.get("series_position") not in (None, ""):
            package.add_metadata(
                None,
                "meta",
                str(metadata["series_position"]),
                {"property": "group-position", "refines": "#series"},
            )
    css = epub.EpubItem(
        uid="style_main", file_name="style/main.css", media_type="text/css", content=CSS
    )
    package.add_item(css)
    if cover:
        package.set_cover(cover_name, cover, create_page=False)
    items = {}
    for member, chapter in book["chapters"].items():
        root = ET.Element(f"{{{X}}}html", nsmap={None: X})
        head = ET.SubElement(root, f"{{{X}}}head")
        ET.SubElement(head, f"{{{X}}}title").text = chapter["title"]
        ET.SubElement(
            head,
            f"{{{X}}}link",
            rel="stylesheet",
            href="style/main.css",
            type="text/css",
        )
        body = ET.SubElement(root, f"{{{X}}}body")
        if chapter.get("role") == "intro":
            body.set("class", "intro")
        ET.SubElement(body, f"{{{X}}}h2").text = chapter["title"]
        render_blocks(body, chapter["blocks"])
        item = epub.EpubHtml(
            title=chapter["title"],
            file_name=member.removeprefix("EPUB/"),
            content=ET.tostring(root),
            lang=metadata["language"] or "zh-CN",
        )
        item.add_item(css)
        package.add_item(item)
        items[member] = item

    def toc(nodes):
        return tuple(
            (
                epub.Section(
                    n["title"], href=items[ordered_members(n["children"])[0]].file_name
                ),
                toc(n["children"]),
            )
            if "children" in n
            else items[n["member"]]
            for n in nodes
        )

    package.toc = toc(book["sections"])
    package.spine = ["nav"] + [items[m] for m in ordered_members(book["sections"])]
    package.add_item(epub.EpubNcx())
    package.add_item(epub.EpubNav())
    output.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(output), package, {})
    validate_archive(output)
    validate_navigation(output, "EPUB/nav.xhtml", "EPUB/toc.ncx", "EPUB/content.opf")


def export_local(
    source: dict,
    output: Path,
    *,
    cover: bytes | None = None,
    cover_mime: str = "image/jpeg",
    prevent_overwrite: bool = True,
) -> Path:
    """Render the prepared crawl, including extras, without any Notion access."""
    if output.exists() and prevent_overwrite:
        raise ValueError("Output EPUB exists; choose another path or use --overwrite")
    original = digest(output.read_bytes()) if output.exists() else None
    book = copy.deepcopy(source)
    extras = []
    for index, story in enumerate(book.get("extras", []), 1):
        member = f"EPUB/extra_{index:03d}.xhtml"
        book["chapters"][member] = story
        extras.append({"member": member})
    if extras:
        book["sections"].append({"title": "番外", "children": extras})
    suffix = {"image/png": "png", "image/gif": "gif", "image/webp": "webp"}.get(
        cover_mime, "jpg"
    )
    with tempfile.TemporaryDirectory(prefix="book-export-") as temporary:
        candidate = Path(temporary) / "book.epub"
        create_book(book, candidate, cover=cover, cover_name=f"cover.{suffix}")
        install_archive(candidate, output, original)
    return output
