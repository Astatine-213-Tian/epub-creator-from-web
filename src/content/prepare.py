"""Adapt crawler output to the editable book source before publishing it."""

from __future__ import annotations

from html import escape
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import requests
from lxml import etree as ET

from src.content.formatting import prepare_blocks
from src.content.html import render_chapter_body, render_paragraphs
from src.content.metadata import enrich_source
from src.content.models import Volume
from src.content.normalization import NormalizationReport, normalize_member
from src.content.styles import CSS
from src.content.xhtml import read_xhtml

SOURCE_CSS = CSS + "\n.scene-break { text-align: center; text-indent: 0; }\n"


def prepare_crawl(
    *,
    title: str,
    author: str,
    volumes: list[Volume],
    source_url: str,
    intro_paragraphs: list[str] | None = None,
    intro_html: str = "",
) -> dict:
    book = {
        "version": 2,
        "metadata": {
            "title": title,
            "creator": author,
            "language": "zh-CN",
            "date": "",
            "source": source_url,
            "description": "",
            "subjects": [],
            "series": "",
            "series_position": "",
        },
        "sections": [],
        "chapters": {},
        "extras": [],
    }
    report = NormalizationReport(path=Path("crawl-source"), applied=True)

    def chapter(name: str, body: str, member: str, role: str = "chapter") -> dict:
        data = (
            f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>{escape(name)}</title>'
            f"<style>{SOURCE_CSS}</style></head><body><h2>{escape(name)}</h2>{body}</body></html>"
        ).encode()
        data = normalize_member(member, data, report, grouped_fanwai_titles=set())
        name = ET.fromstring(data).findtext("{*}head/{*}title")
        blocks = read_xhtml(data, title=name)
        return {
            "title": name,
            "blocks": prepare_blocks(data, member, {}, blocks),
            "role": role,
        }

    intro = (
        render_paragraphs(intro_paragraphs)
        if intro_paragraphs is not None
        else intro_html
    )
    if intro:
        member = "EPUB/intro.xhtml"
        book["chapters"][member] = chapter("简介", intro, member, "intro")
        book["sections"].append({"member": member})
        book["metadata"]["description"] = "\n".join(
            "".join(run["text"] for run in block["runs"])
            for block in book["chapters"][member]["blocks"]
        ).strip()
    for vi, volume in enumerate(volumes, 1):
        children = []
        for ci, source in enumerate(volume.chapters, 1):
            member = f"EPUB/chap_{vi:02d}_{ci:03d}.xhtml"
            value = chapter(source.title, render_chapter_body(source), member)
            if volume.title.strip() == "番外":
                book["extras"].append(value)
            else:
                book["chapters"][member] = value
                children.append({"member": member})
        if children:
            if volume.title and (len(volumes) > 1 or volume.title != "正文"):
                book["sections"].append({"title": volume.title, "children": children})
            else:
                book["sections"].extend(children)
    if not book["chapters"]:
        raise ValueError("Crawl produced no readable chapters")
    return book


def prepare_source(
    *,
    title: str,
    author: str,
    volumes: list[Volume],
    source_url: str,
    intro_paragraphs: list[str] | None = None,
    intro_html: str = "",
) -> dict:
    """Prepare editable content and enrich metadata before choosing a destination."""
    book = prepare_crawl(
        title=title,
        author=author,
        volumes=volumes,
        source_url=source_url,
        intro_paragraphs=intro_paragraphs,
        intro_html=intro_html,
    )
    book["identifier"] = str(uuid5(NAMESPACE_URL, source_url))
    book["source_format"] = "crawl"
    try:
        book["metadata_report"] = enrich_source(book)
    except (OSError, ValueError, requests.RequestException):
        book["metadata_report"] = {"status": "unavailable"}
        print("Official metadata unavailable; keeping collected metadata")
    return book
