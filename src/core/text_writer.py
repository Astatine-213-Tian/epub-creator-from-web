from __future__ import annotations

import re
from html import unescape
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

from src.core.models import Chapter, Volume


SPACE_RE = re.compile(r"[ \t\u3000]+")
BLANK_RE = re.compile(r"\n{3,}")
DOMAIN_PLACEHOLDER_RE = re.compile(r"^[()\[\]（）【】\s•·.。ｃｏｍcomCOM]+$")
CHAPTER_AD_TAIL_RE = re.compile(r"吗[?？]请记住.*$")
SHORT_PROMO_RE = re.compile(r"^[^\u4e00-\u9fffA-Za-z0-9]{0,3}想看.{0,40}《[^》]+》$")
MIRROR_AD_MARKERS = (
    "提醒您最全",
    "最新章 节",
    "最新章节",
    "全网首发更新",
    "想看",
)


def _is_mirror_boilerplate_line(value: str) -> bool:
    if DOMAIN_PLACEHOLDER_RE.fullmatch(value):
        return True
    if SHORT_PROMO_RE.fullmatch(value):
        return True
    return "域名" in value and "《" in value and any(marker in value for marker in MIRROR_AD_MARKERS)


def clean_text_line(value: str) -> str:
    text = SPACE_RE.sub(" ", unescape(value)).strip()
    text = CHAPTER_AD_TAIL_RE.sub("", text).strip()
    if _is_mirror_boilerplate_line(text):
        return ""
    return text


def html_fragment_to_lines(fragment: str) -> list[str]:
    soup = BeautifulSoup(fragment or "", "html.parser")
    for tag in soup.find_all(["br", "p", "div", "section", "h1", "h2", "h3", "li"]):
        tag.append("\n")
    lines = [clean_text_line(line) for line in soup.get_text("\n").splitlines()]
    return [line for line in lines if line]


def chapter_lines(chapter: Chapter) -> list[str]:
    if chapter.paragraphs:
        return [line for line in (clean_text_line(p) for p in chapter.paragraphs) if line]
    if chapter.html_blocks:
        lines: list[str] = []
        for block in chapter.html_blocks:
            lines.extend(html_fragment_to_lines(block))
        return lines
    return []


def intro_lines(
    *,
    intro_paragraphs: Iterable[str] | None = None,
    intro_html: str = "",
) -> list[str]:
    if intro_paragraphs is not None:
        return [line for line in (clean_text_line(p) for p in intro_paragraphs) if line]
    return html_fragment_to_lines(intro_html)


def render_book_txt(
    *,
    title: str,
    author: str,
    volumes: list[Volume],
    intro_paragraphs: Iterable[str] | None = None,
    intro_html: str = "",
) -> str:
    lines: list[str] = [clean_text_line(title)]
    if author.strip():
        lines.append(f"作者：{clean_text_line(author)}")
    lines.append("")

    intro = intro_lines(intro_paragraphs=intro_paragraphs, intro_html=intro_html)
    if intro:
        lines.extend(["简介", ""])
        lines.extend(intro)
        lines.append("")

    for volume in volumes:
        if volume.title.strip():
            lines.extend([clean_text_line(volume.title), ""])
        for chapter in volume.chapters:
            lines.extend([clean_text_line(chapter.title), ""])
            lines.extend(chapter_lines(chapter))
            lines.append("")

    text = "\n".join(lines).strip() + "\n"
    return BLANK_RE.sub("\n\n", text)


def write_txt(
    *,
    title: str,
    author: str,
    volumes: list[Volume],
    out_path: Path,
    intro_paragraphs: Iterable[str] | None = None,
    intro_html: str = "",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_book_txt(
            title=title,
            author=author,
            volumes=volumes,
            intro_paragraphs=intro_paragraphs,
            intro_html=intro_html,
        ),
        encoding="utf-8",
    )
