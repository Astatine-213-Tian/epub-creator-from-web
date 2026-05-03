#!/usr/bin/env python3
"""Scrape a book from quanben.io and build an EPUB.

Usage:
    python -m booklib.parsers.quanben <book_url_or_slug> [-o output.epub]

Examples:
    python -m booklib.parsers.quanben https://quanben.io/n/yaoer/list.html
    python -m booklib.parsers.quanben yaoer

The canonical list page hides middle chapters behind JSONP, but the AMP list
page exposes the complete table of contents in plain HTML.
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

from booklib import Chapter, Volume, write_epub
from booklib.parallel_fetch import crawl_items, retry_async


HOST = "https://quanben.io"
AMP_HOST = "https://quanben.io/amp"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)

LIST_RE = re.compile(r"^/(?:amp/)?n/([^/]+)/list\.html$")
BOOK_RE = re.compile(r"^/(?:amp/)?n/([^/]+)/?$")
CHAPTER_RE = re.compile(r"^/(?:amp/)?n/([^/]+)/(\d+)\.html$")


# ---------------------------------------------------------------------------
# HTTP


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

    def get_bytes(self, url: str) -> bytes:
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        return r.content


# ---------------------------------------------------------------------------
# Models


@dataclass
class BookMeta:
    slug: str
    title: str
    author: str
    intro_paragraphs: list[str]
    category: str = ""
    status: str = ""
    cover_url: str | None = None
    cover_bytes: bytes | None = None
    cover_mime: str = "image/jpeg"


@dataclass
class ChapterRef:
    title: str
    url: str
    chapter_id: str


# ---------------------------------------------------------------------------
# Parsers


def _clean_text(s: str) -> str:
    return re.sub(r"[\s\xa0　]+", " ", s).strip()


def parse_list_page(html: str, slug: str) -> tuple[BookMeta, list[ChapterRef]]:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    h1 = soup.find("h1")
    if h1:
        title = _clean_text(h1.get_text())
    name = soup.find(itemprop="name") or soup.select_one(".list2 h3")
    if name:
        title = _clean_text(name.get_text()) or title

    author = ""
    category = ""
    status = ""
    list2 = soup.find(class_="list2")
    if list2:
        for p in list2.find_all("p"):
            text = _clean_text(p.get_text())
            value = p.find("span")
            value_text = _clean_text(value.get_text()) if value else ""
            if text.startswith("作者"):
                author = value_text or text.split(":", 1)[-1].strip()
            elif text.startswith("类别"):
                category = value_text
            elif text.startswith("状态"):
                status = value_text

    cover_url: str | None = None
    cover = soup.select_one(".list2 img, .list2 amp-img")
    if cover and cover.get("src"):
        cover_url = urljoin(HOST, cover["src"])

    intro_paragraphs: list[str] = []
    desc = soup.find(class_="description")
    if desc:
        for tag in desc.find_all(["script", "style", "iframe", "ins"]):
            tag.decompose()
        for br in desc.find_all("br"):
            br.replace_with("\n")
        for chunk in re.split(r"\n+", desc.get_text("\n")):
            chunk = chunk.strip(" \t\xa0　")
            if chunk:
                intro_paragraphs.append(chunk)

    refs: list[ChapterRef] = []
    seen: set[str] = set()
    for a in soup.select("ul.list3 a[href]"):
        path = urlparse(a["href"]).path
        m = CHAPTER_RE.match(path)
        if not m or m.group(1) != slug:
            continue
        chapter_id = m.group(2)
        if chapter_id in seen:
            continue
        seen.add(chapter_id)
        title_text = _clean_text(a.get_text())
        if not title_text:
            continue
        refs.append(
            ChapterRef(
                title=title_text,
                url=urljoin(AMP_HOST, f"/amp/n/{slug}/{chapter_id}.html"),
                chapter_id=chapter_id,
            )
        )

    return (
        BookMeta(
            slug=slug,
            title=title or "未命名",
            author=author,
            intro_paragraphs=intro_paragraphs,
            category=category,
            status=status,
            cover_url=cover_url,
        ),
        refs,
    )


def parse_chapter_html(html: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    h = soup.find("h1", class_="headline") or soup.find("h1")
    if h:
        title = _clean_text(h.get_text())
    if not title and soup.title and soup.title.string:
        title = _clean_text(re.split(r"\s+-\s+|_", soup.title.string)[0])

    content = (
        soup.find(id="content")
        or soup.find(itemprop="articleBody")
        or soup.find(class_="articlebody")
    )
    paragraphs: list[str] = []
    if content:
        for tag in content.find_all(["script", "style", "iframe", "ins"]):
            tag.decompose()
        for p in content.find_all("p"):
            text = _clean_text(p.get_text())
            if not text or text in {"上一页", "下一页", "目录"}:
                continue
            paragraphs.append(text)

    return title, paragraphs


# ---------------------------------------------------------------------------
# Crawl


async def crawl_chapter(ref: ChapterRef, *, delay: float = 0.4) -> Chapter:
    def fetch() -> Chapter:
        fetcher = Fetcher(delay=delay)
        page_title, paragraphs = parse_chapter_html(fetcher.get_html(ref.url))
        if not paragraphs:
            raise RuntimeError(f"empty chapter body: {ref.url}")
        return Chapter(title=page_title or ref.title, paragraphs=paragraphs)

    return await retry_async(ref.title, lambda: asyncio.to_thread(fetch))


async def crawl_chapters(
    refs: list[ChapterRef],
    *,
    delay: float = 0.4,
    concurrency: int = 4,
) -> list[Chapter]:
    return await crawl_items(
        refs,
        lambda ref: crawl_chapter(ref, delay=delay),
        concurrency=concurrency,
        item_name="chapter",
    )


def crawl_book(
    book_url: str,
    *,
    delay: float = 0.4,
    concurrency: int = 4,
) -> tuple[BookMeta, list[Volume]]:
    parsed = urlparse(book_url)
    m = LIST_RE.match(parsed.path) or BOOK_RE.match(parsed.path)
    if not m:
        raise ValueError(f"not a quanben /n/<slug>/list.html URL: {book_url}")
    slug = m.group(1)

    fetcher = Fetcher(delay=delay)
    list_url = urljoin(AMP_HOST, f"/amp/n/{slug}/list.html")
    print(f"[+] fetching list {list_url}", file=sys.stderr)
    meta, refs = parse_list_page(fetcher.get_html(list_url), slug)
    print(f"[+] book: {meta.title} / {meta.author}", file=sys.stderr)
    print(f"[+] {len(refs)} chapters discovered", file=sys.stderr)
    if not refs:
        raise ValueError(f"no chapters found on {list_url}")

    if meta.cover_url:
        try:
            meta.cover_bytes = fetcher.get_bytes(meta.cover_url)
            ext = Path(urlparse(meta.cover_url).path).suffix.lower()
            meta.cover_mime = (
                "image/png" if ext == ".png" else
                "image/gif" if ext == ".gif" else
                "image/webp" if ext == ".webp" else
                "image/jpeg"
            )
            print(f"[+] cover {len(meta.cover_bytes)} bytes", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[!] cover fetch failed: {e}", file=sys.stderr)

    chapters = asyncio.run(
        crawl_chapters(refs, delay=delay, concurrency=concurrency)
    )

    return meta, [Volume(title="", chapters=chapters)]


def build_epub(meta: BookMeta, volumes: list[Volume], out_path: Path) -> None:
    write_epub(
        identifier=f"quanben-{meta.slug}-{int(time.time())}",
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        out_path=out_path,
        intro_paragraphs=meta.intro_paragraphs,
        cover_bytes=meta.cover_bytes,
        cover_mime=meta.cover_mime,
        emit_single_volume_cover=False,
    )


# ---------------------------------------------------------------------------
# Main


def _resolve_book_url(arg: str) -> str:
    if arg.startswith("http"):
        return arg
    slug = arg.strip("/")
    if slug.startswith("n/"):
        slug = slug.split("/", 1)[1]
    if slug.endswith("/list.html"):
        slug = slug[: -len("/list.html")]
    return f"{HOST}/n/{slug}/list.html"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("book", help="Book URL on quanben.io or just the slug")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--delay", type=float, default=0.4)
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of chapter pages to fetch concurrently",
    )
    args = p.parse_args(argv)

    book_url = _resolve_book_url(args.book)
    meta, volumes = crawl_book(
        book_url,
        delay=args.delay,
        concurrency=args.concurrency,
    )

    n_chap = sum(len(v.chapters) for v in volumes)
    print(f"[+] {n_chap} chapter(s)", file=sys.stderr)

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(__file__).resolve().parents[2] / "epub"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{meta.title}.epub"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_epub(meta, volumes, out_path)
    print(f"[+] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
