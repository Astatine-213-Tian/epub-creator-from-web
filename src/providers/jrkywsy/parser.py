#!/usr/bin/env python3
"""Scrape a single-post book from jrkywsy.blog.fc2.com and build an EPUB.

Usage:
    python jrkywsy_to_epub.py <url> [-o output.epub]

The source page is Traditional Chinese; output is converted to Simplified
Chinese before EPUB generation.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from opencc import OpenCC

from src import Chapter, Volume, write_epub
from src.core.output import resolve_output_path

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
ENTRY_RE = re.compile(r"blog-entry-\d+\.html")
CHAPTER_RE = re.compile(
    r"^\s*(?:.+?\s+)?(第\s*[一二三四五六七八九十百零兩两\d]+\s*回(?:\s+.*)?)\s*$"
)
FOOTER_STOP_MARKERS = (
    "FC2拍手标签从这里开始",
    "FC2拍手標籤從這裡開始",
)


@dataclass
class BookMeta:
    title: str
    author: str
    intro_paragraphs: list[str]


def fetch(url: str, retries: int = 3) -> str:
    last_err: Exception | None = None
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    for i in range(retries):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except Exception as exc:
            last_err = exc
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def guess_title_author(raw_title: str) -> tuple[str, str]:
    raw = raw_title.strip()
    author = ""
    m = re.search(r"\s+by\s+(.+?)\s*$", raw, flags=re.I)
    if m:
        author = m.group(1).strip()
        raw = raw[: m.start()].strip()
    return raw or raw_title, author


def _text_lines_from_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    lines: list[str] = []
    for line in soup.get_text("\n").splitlines():
        line = line.strip(" \t\u3000\xa0")
        if line:
            lines.append(line)
    return lines


def _split_intro_and_body(main: Tag) -> tuple[list[str], list[str]]:
    more_anchor = main.find("a", attrs={"name": "more"}) or main.find(id="more")
    intro_parts: list[str] = []
    body_parts: list[str] = []

    if more_anchor:
        for child in main.children:
            if child is more_anchor:
                break
            intro_parts.append(str(child))
        for sibling in more_anchor.next_siblings:
            body_parts.append(str(sibling))
    else:
        body_parts.append(str(main))

    return (
        _text_lines_from_html("".join(intro_parts)),
        _text_lines_from_html("".join(body_parts)),
    )


def _chapter_title(line: str, book_title: str) -> str | None:
    if line.startswith(("（", "(", "【")):
        return None
    m = CHAPTER_RE.match(line)
    if not m:
        return None
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    if book_title and title.startswith(book_title):
        title = title[len(book_title):].strip()
    return title


def split_into_volume(lines: list[str], book_title: str) -> Volume:
    volume = Volume(title="")
    current: Chapter | None = None

    def ensure_chapter() -> Chapter:
        nonlocal current
        if current is None:
            current = Chapter(title=book_title or "正文")
            volume.chapters.append(current)
        return current

    for line in lines:
        if any(marker in line for marker in FOOTER_STOP_MARKERS):
            break
        title = _chapter_title(line, book_title)
        if title:
            current = Chapter(title=title)
            volume.chapters.append(current)
            continue
        ensure_chapter().paragraphs.append(line)

    volume.chapters = [chapter for chapter in volume.chapters if chapter.paragraphs]
    return volume


def parse_page(html: str, url: str) -> tuple[BookMeta, list[Volume]]:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.select_one("h2.title") or soup.select_one("h1")
    raw_title = title_tag.get_text(" ", strip=True) if title_tag else url
    title, author = guess_title_author(raw_title)

    main = soup.select_one("div.main")
    if main is None:
        raise RuntimeError("could not locate <div class='main'> in page")

    intro_paragraphs, body_lines = _split_intro_and_body(main)
    volume = split_into_volume(body_lines, title)
    return BookMeta(title=title, author=author, intro_paragraphs=intro_paragraphs), [volume]


def collect_related_entry_urls(html: str, url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one("div.main") or soup
    urls: list[str] = []
    seen: set[str] = set()
    for a in main.find_all("a", href=True):
        href = a["href"]
        if not ENTRY_RE.search(href):
            continue
        linked = urljoin(url, href)
        if linked in seen:
            continue
        seen.add(linked)
        urls.append(linked)
    return urls


def convert_book(meta: BookMeta, volumes: list[Volume], cc: OpenCC) -> None:
    meta.title = cc.convert(meta.title)
    meta.author = cc.convert(meta.author)
    meta.intro_paragraphs = [cc.convert(paragraph) for paragraph in meta.intro_paragraphs]
    for volume in volumes:
        volume.title = cc.convert(volume.title)
        for chapter in volume.chapters:
            chapter.title = cc.convert(chapter.title)
            chapter.paragraphs = [cc.convert(paragraph) for paragraph in chapter.paragraphs]


def crawl_book(url: str) -> tuple[BookMeta, list[Volume]]:
    html = fetch(url)
    related_urls = collect_related_entry_urls(html, url) or [url]
    meta: BookMeta | None = None
    volumes: list[Volume] = []
    for index, page_url in enumerate(related_urls):
        page_html = html if index == 0 and page_url == url else fetch(page_url)
        page_meta, page_volumes = parse_page(page_html, page_url)
        if meta is None:
            meta = page_meta
        volumes.extend(page_volumes)
    assert meta is not None
    convert_book(meta, volumes, OpenCC("t2s"))
    return meta, volumes


def build_epub(meta: BookMeta, volumes: list[Volume], out_path: Path) -> None:
    write_epub(
        identifier=f"jrkywsy-{int(time.time())}",
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        out_path=out_path,
        intro_paragraphs=meta.intro_paragraphs,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url", help="Book post URL on jrkywsy.blog.fc2.com")
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args(argv)

    print(f"[+] fetching {args.url}", file=sys.stderr)
    meta, volumes = crawl_book(args.url)
    n_chap = sum(len(volume.chapters) for volume in volumes)
    print(f"[+] book: {meta.title} / {meta.author}", file=sys.stderr)
    print(f"[+] {n_chap} chapter(s)", file=sys.stderr)

    out_path = resolve_output_path(args.output, meta.title, meta.author)

    build_epub(meta, volumes, out_path)
    print(f"[+] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
