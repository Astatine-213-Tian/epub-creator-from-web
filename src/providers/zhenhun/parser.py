#!/usr/bin/env python3
"""Scrape books from zhenhunxiaoshuo.com and build an EPUB.

The site stores some books as WordPress category pages whose posts are only
coarse containers ("第1节", "第2节", ...).  For those books, the real reading
order is encoded by numeric section markers inside the article body, so chapter
splitting and ordering are based on the body markers rather than post titles.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src import Chapter, Volume, write_epub
from src.core.output import resolve_output_path
from src.fetch.parallel import crawl_items, retry_async


HOST = "https://www.zhenhunxiaoshuo.com"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)

BOOK_RE = re.compile(r"^/([^/]+)/?$")
POST_RE = re.compile(r"^/(\d+)\.html$")
SECTION_MARK_RE = re.compile(r"^\s*(\d{1,4})\s*$")


class Fetcher:
    def __init__(self, *, delay: float = 0.4):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})

    def get_html(self, url: str) -> str:
        for i in range(4):
            try:
                r = self.session.get(url, timeout=30)
                r.raise_for_status()
                r.encoding = r.apparent_encoding or "utf-8"
                time.sleep(self.delay)
                return r.text
            except Exception:
                if i == 3:
                    raise
                time.sleep(1.0 * (i + 1))
        raise RuntimeError("unreachable")


@dataclass
class BookMeta:
    slug: str
    title: str
    author: str
    intro_paragraphs: list[str]


@dataclass
class ChapterRef:
    title: str
    url: str
    post_id: str
    order: int


@dataclass
class Section:
    number: int
    paragraphs: list[str]
    source_title: str
    source_url: str


def _clean_text(s: str) -> str:
    return re.sub(r"[\s\xa0　]+", " ", s).strip()


def _paragraph_lines(tag) -> list[str]:
    for child in tag.find_all(["script", "style", "iframe", "ins"]):
        child.decompose()
    for br in tag.find_all("br"):
        br.replace_with("\n")
    lines: list[str] = []
    for chunk in tag.get_text("\n").splitlines():
        line = _clean_text(chunk)
        if line:
            lines.append(line)
    return lines


def _author_from_page(soup: BeautifulSoup) -> str:
    title = _clean_text(soup.title.string) if soup.title and soup.title.string else ""
    m = re.search(r"\(([^()]+)\)", title)
    if m:
        return _clean_text(m.group(1))

    intro = soup.select_one(".focusbox-text .text") or soup.select_one(".focusbox-text")
    intro_text = _clean_text(intro.get_text(" ")) if intro else ""
    m = re.search(r"\bBY\s*([^\s，,。；;、]+)", intro_text, flags=re.IGNORECASE)
    if m:
        author = _clean_text(m.group(1))
        if author == "南康":
            return "南康白起"
        return author
    return ""


def parse_book_page(html: str, slug: str) -> tuple[BookMeta, list[ChapterRef]]:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    h1 = soup.select_one("h1.focusbox-title") or soup.find("h1")
    if h1:
        title = _clean_text(h1.get_text())
    if not title and soup.title and soup.title.string:
        title = _clean_text(re.split(r"\(|-|_", soup.title.string)[0])

    intro_paragraphs: list[str] = []
    intro = soup.select_one(".focusbox-text .text") or soup.select_one(".focusbox-text")
    if intro:
        intro_paragraphs = _paragraph_lines(intro)

    refs: list[ChapterRef] = []
    seen: set[str] = set()
    container = soup.select_one(".excerpts") or soup
    for a in container.select("article.excerpt a[href], a[href]"):
        path = urlparse(a["href"]).path
        m = POST_RE.match(path)
        if not m:
            continue
        post_id = m.group(1)
        if post_id in seen:
            continue
        seen.add(post_id)
        title_text = _clean_text(a.get("title") or a.get_text())
        if not title_text:
            continue
        refs.append(
            ChapterRef(
                title=title_text,
                url=urljoin(HOST, a["href"]),
                post_id=post_id,
                order=len(refs),
            )
        )

    return (
        BookMeta(
            slug=slug,
            title=title or "未命名",
            author=_author_from_page(soup),
            intro_paragraphs=intro_paragraphs,
        ),
        refs,
    )


def parse_chapter_page(html: str, ref: ChapterRef) -> list[Section]:
    soup = BeautifulSoup(html, "lxml")
    article = soup.select_one("article.article-content")
    if not article:
        return []

    lines = _paragraph_lines(article)
    sections: list[Section] = []
    current_number: int | None = None
    current_paragraphs: list[str] = []

    for line in lines:
        marker = SECTION_MARK_RE.match(line)
        if marker:
            if current_number is not None and current_paragraphs:
                sections.append(Section(current_number, current_paragraphs, ref.title, ref.url))
            current_number = int(marker.group(1))
            current_paragraphs = []
            continue

        if current_number is None:
            continue
        if set(line) <= {"-"}:
            continue
        current_paragraphs.append(line)

    if current_number is not None and current_paragraphs:
        sections.append(Section(current_number, current_paragraphs, ref.title, ref.url))

    if sections:
        return sections

    paragraphs = [line for line in lines if line != ref.title and set(line) > {"-"}]
    return [Section(ref.order + 1, paragraphs, ref.title, ref.url)]


async def crawl_post(ref: ChapterRef, *, delay: float = 0.4) -> list[Section]:
    def fetch() -> list[Section]:
        fetcher = Fetcher(delay=delay)
        sections = parse_chapter_page(fetcher.get_html(ref.url), ref)
        if not sections:
            raise RuntimeError(f"empty chapter body: {ref.url}")
        return sections

    return await retry_async(ref.title, lambda: asyncio.to_thread(fetch))


async def crawl_posts(
    refs: list[ChapterRef],
    *,
    delay: float = 0.4,
    concurrency: int = 4,
) -> list[list[Section]]:
    return await crawl_items(
        refs,
        lambda ref: crawl_post(ref, delay=delay),
        concurrency=concurrency,
        item_name="chapter",
    )


def _sections_to_chapters(section_groups: list[list[Section]]) -> list[Chapter]:
    sections = [section for group in section_groups for section in group]
    sections.sort(key=lambda section: section.number)

    seen: set[int] = set()
    chapters: list[Chapter] = []
    for section in sections:
        if section.number in seen:
            print(
                f"[!] duplicate content section {section.number} in {section.source_url}; keeping first",
                file=sys.stderr,
            )
            continue
        seen.add(section.number)
        chapters.append(Chapter(title=f"第{section.number}节", paragraphs=section.paragraphs))

    if chapters:
        numbers = [section.number for section in sections]
        missing = sorted(set(range(min(numbers), max(numbers) + 1)) - seen)
        if missing:
            print(f"[!] missing content section number(s): {', '.join(map(str, missing))}", file=sys.stderr)

    return chapters


def crawl_book(
    book_url: str,
    *,
    delay: float = 0.4,
    concurrency: int = 4,
) -> tuple[BookMeta, list[Volume]]:
    parsed = urlparse(book_url)
    m = BOOK_RE.match(parsed.path)
    if not m:
        raise ValueError(f"not a zhenhunxiaoshuo category URL: {book_url}")
    slug = m.group(1)

    fetcher = Fetcher(delay=delay)
    print(f"[+] fetching list {book_url}", file=sys.stderr)
    meta, refs = parse_book_page(fetcher.get_html(book_url), slug)
    print(f"[+] book: {meta.title} / {meta.author}", file=sys.stderr)
    print(f"[+] {len(refs)} source chapter page(s) discovered", file=sys.stderr)
    if not refs:
        raise ValueError(f"no chapters found on {book_url}")

    section_groups = asyncio.run(crawl_posts(refs, delay=delay, concurrency=concurrency))
    chapters = _sections_to_chapters(section_groups)
    return meta, [Volume(title="", chapters=chapters)]


def build_epub(meta: BookMeta, volumes: list[Volume], out_path: Path) -> None:
    write_epub(
        identifier=f"zhenhun-{meta.slug}-{int(time.time())}",
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        out_path=out_path,
        intro_paragraphs=meta.intro_paragraphs,
        emit_single_volume_cover=False,
    )


def _resolve_book_url(arg: str) -> str:
    if arg.startswith("http"):
        return arg
    slug = arg.strip("/")
    return f"{HOST}/{slug}/"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("book", help="Book URL on zhenhunxiaoshuo.com or just the category slug")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--delay", type=float, default=0.4)
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum source chapter pages to fetch concurrently",
    )
    args = p.parse_args(argv)

    book_url = _resolve_book_url(args.book)
    meta, volumes = crawl_book(book_url, delay=args.delay, concurrency=args.concurrency)
    n_chap = sum(len(v.chapters) for v in volumes)
    print(f"[+] {n_chap} content section(s)", file=sys.stderr)

    out_path = resolve_output_path(args.output, meta.title, meta.author)
    build_epub(meta, volumes, out_path)
    print(f"[+] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
