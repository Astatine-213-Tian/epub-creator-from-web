#!/usr/bin/env python3
"""Scrape a book from ibusread.com and build an EPUB."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src import Chapter, Volume, write_epub
from src.core.output import resolve_output_path
from src.runtime.progress import ProgressLogger


HOST = "https://www.ibusread.com"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)

CHAPTER_RE = re.compile(
    r"^/novel/chapter/(?P<category>\d+)_(?P<book_id>[0-9a-f]+)_(?P<chapter_id>[0-9a-f]+)\.html$"
)
DETAIL_RE = re.compile(r"^/novel/(?P<book_id>[0-9a-f]+)$")
CATALOG_RE = re.compile(r"^/novel/catalog/(?P<book_id>[0-9a-f]+)(?:_(?P<section>\d+))?$")
EXTRA_SPLIT_MARKERS = (
    "小东西在洗澡，等下陪他去看上次耽误的失恋",
    "每次去上海接机的时候",
    "http://weibo.com/lin2xin",
)


@dataclass
class BookMeta:
    book_id: str
    category: str
    title: str
    author: str
    intro_paragraphs: list[str]


@dataclass(frozen=True)
class ChapterRef:
    title: str
    url: str
    api_id: str
    order: int


@dataclass(frozen=True)
class ChapterChunk:
    ref: ChapterRef
    lines: list[str]


class Fetcher:
    def __init__(self, *, delay: float = 0.25):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": UA,
                "X-Requested-With": "XMLHttpRequest",
                "Origin": HOST,
            }
        )

    def get_html(self, url: str) -> str:
        for attempt in range(4):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                response.encoding = response.apparent_encoding or "utf-8"
                time.sleep(self.delay)
                return response.text
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError("unreachable")

    def post_json(self, url: str, data: dict[str, str], *, referer: str) -> dict:
        headers = {"Referer": referer}
        for attempt in range(4):
            try:
                response = self.session.post(url, data=data, headers=headers, timeout=30)
                response.raise_for_status()
                time.sleep(self.delay)
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("JSON response is not an object")
                return payload
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError("unreachable")


def _clean_text(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = value.replace("丨", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n　")


def _clean_title(value: str) -> str:
    value = _clean_text(value)
    value = re.sub(r"\[BL\]$", "", value).strip()
    if value == "我们的十二年一个轮回这是传奇":
        return "我们的十二年 一个轮回 这是传奇"
    return value


def _clean_chapter_title(value: str) -> str:
    value = _clean_text(value)
    match = re.fullmatch(r"第([0-9０-９一二三四五六七八九十百千两]+)节", value)
    if match:
        return f"第{match.group(1)}章"
    return value


def _resolve_chapter_url(target: str) -> str:
    if target.startswith("http"):
        parsed = urlparse(target)
        match = CHAPTER_RE.match(parsed.path)
        if match:
            return urljoin(HOST, parsed.path)
        detail = DETAIL_RE.match(parsed.path)
        if detail:
            return f"{HOST}/novel/{detail.group('book_id')}"
        raise ValueError(f"not an ibusread novel/chapter URL: {target}")
    if re.fullmatch(r"[0-9a-f]+", target.strip()):
        return f"{HOST}/novel/{target.strip()}"
    raise ValueError(f"not an ibusread book id or URL: {target}")


def _parse_chapter_url(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    match = CHAPTER_RE.match(parsed.path)
    if not match:
        raise ValueError(f"not an ibusread chapter URL: {url}")
    return match.group("category"), match.group("book_id"), match.group("chapter_id")


def _parse_detail(html: str) -> tuple[str, str, str, str]:
    soup = BeautifulSoup(html, "lxml")
    name = ""
    author = ""
    category = ""
    first_chapter = ""

    page = soup.select_one(".page-novel")
    if page:
        h1 = page.find("h1")
        if h1:
            text = _clean_text(h1.get_text(""))
            name = re.split(r"_第\d+章|_", text, maxsplit=1)[0]
        auth = page.select_one(".auth")
        if auth:
            author = _clean_text(auth.get_text())
        start = page.select_one("a.js_chapter_history[href]")
        if start:
            first_chapter = urljoin(HOST, start["href"])
            category, _book, _chapter = _parse_chapter_url(first_chapter)

    if not name:
        title = soup.title.string if soup.title and soup.title.string else ""
        name = re.split(r"[_-]", title, maxsplit=1)[0]

    return _clean_title(name), author, category, first_chapter


def _parse_catalog(html: str) -> tuple[str, str, list[ChapterRef], int]:
    soup = BeautifulSoup(html, "lxml")
    title = ""
    author = ""
    total = 0
    header = soup.select_one(".page-novel-top h1")
    if header:
        text = _clean_text(header.get_text(""))
        match = re.search(r"《(.+?)》共(\d+)节", text)
        if match:
            title = _clean_title(match.group(1))
            total = int(match.group(2))

    refs: list[ChapterRef] = []
    for a in soup.select(".section-list a[href]"):
        pass
    for a in soup.select(".page-novel-catalog .list a[href]"):
        href = urljoin(HOST, a["href"])
        category, book_id, chapter_id = _parse_chapter_url(href)
        refs.append(
            ChapterRef(
                title=_clean_chapter_title(a.get_text()),
                url=href,
                api_id=f"{category}_{book_id}_{chapter_id}",
                order=len(refs) + 1,
            )
        )
    return title, author, refs, total


def _catalog_url(book_id: str, section: int = 1) -> str:
    return f"{HOST}/novel/catalog/{book_id}_{section}?cur=1&page=1"


def parse_catalogs(fetcher: Fetcher, book_id: str) -> tuple[list[ChapterRef], int]:
    refs: list[ChapterRef] = []
    total = 0
    seen: set[str] = set()

    section = 1
    while True:
        url = _catalog_url(book_id, section)
        title, author, page_refs, page_total = _parse_catalog(fetcher.get_html(url))
        del title, author
        total = max(total, page_total)
        for ref in page_refs:
            if ref.api_id in seen:
                continue
            seen.add(ref.api_id)
            refs.append(
                ChapterRef(
                    title=ref.title,
                    url=ref.url,
                    api_id=ref.api_id,
                    order=len(refs) + 1,
                )
            )
        if total and len(refs) >= total:
            break
        if not page_refs:
            break
        section += 1
    return refs, total


def _chapter_lines_from_api(payload: dict) -> list[str]:
    if payload.get("code") != 0:
        raise RuntimeError(f"chapter API failed: {payload}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"chapter API missing data: {payload}")
    raw = data.get("content_data", "[]")
    lines = json.loads(raw)
    if not isinstance(lines, list):
        raise RuntimeError("chapter content_data is not a list")
    return [_clean_text(str(line)) for line in lines if _clean_text(str(line))]


def crawl_chunk(ref: ChapterRef, *, delay: float) -> ChapterChunk:
    fetcher = Fetcher(delay=delay)
    payload = fetcher.post_json(
        f"{HOST}/api/chapter/detail",
        {"id": ref.api_id},
        referer=ref.url,
    )
    lines = _chapter_lines_from_api(payload)
    if not lines:
        raise RuntimeError(f"empty chapter body: {ref.url}")
    return ChapterChunk(ref=ref, lines=lines)


def _split_content_volumes(chunks: list[ChapterChunk]) -> list[Volume]:
    main_chapters: list[Chapter] = []
    extra_chapters: list[Chapter] = []
    active_chapters = main_chapters
    title = "序"
    paragraphs: list[str] = []
    current_number = 0

    def flush() -> None:
        nonlocal paragraphs
        if paragraphs:
            active_chapters.append(Chapter(title=title, paragraphs=paragraphs))
            paragraphs = []

    for chunk in chunks:
        for line in chunk.lines:
            if line == "那些之后的事":
                flush()
                active_chapters = extra_chapters
                current_number = 0
                title = ""
                continue
            if active_chapters is extra_chapters and any(marker in line for marker in EXTRA_SPLIT_MARKERS):
                flush()
                current_number += 1
                title = str(current_number)
            marker = re.fullmatch(r"\d{1,4}", line)
            if marker:
                number = int(marker.group(0))
                if number > current_number and number - current_number <= 20:
                    flush()
                    current_number = number
                    title = str(number)
                    continue
            paragraphs.append(line)
    flush()
    volumes = [Volume(title="", chapters=main_chapters)]
    if extra_chapters:
        volumes.append(Volume(title="那些之后的事", chapters=extra_chapters))
    return volumes


def crawl_volumes(
    refs: list[ChapterRef],
    *,
    delay: float,
    concurrency: int,
) -> list[Volume]:
    progress = ProgressLogger()
    total = len(refs)
    counter = progress.counter("Fetch", total, "chapter(s)")
    counter.start()
    completed = 0
    results: list[ChapterChunk | None] = [None] * total
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = {
                executor.submit(crawl_chunk, ref, delay=delay): index
                for index, ref in enumerate(refs)
            }
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                completed += 1
                counter.update(completed, detail=refs[index].title)
        counter.finish(f"Fetch complete: {completed}/{total} chapter(s)")
    finally:
        counter.close()
    chunks = [chunk for chunk in results if chunk is not None]
    return _split_content_volumes(chunks)


def crawl_book(
    book_url: str,
    *,
    delay: float = 0.25,
    concurrency: int = 4,
) -> tuple[BookMeta, list[Volume]]:
    fetcher = Fetcher(delay=delay)
    url = _resolve_chapter_url(book_url)
    if "/novel/chapter/" in url:
        print(f"[+] fetching first chapter page {url}", file=sys.stderr)
        first_html = fetcher.get_html(url)
        soup = BeautifulSoup(first_html, "lxml")
        page = soup.select_one(".js_page_novel_chapter")
        if not page:
            raise ValueError("chapter metadata not found")
        book_id = page.get("data-id", "")
        category = page.get("data-category", "")
        title = _clean_title(page.get("data-name", ""))
        author = _clean_text(page.get("data-auth", ""))
    else:
        print(f"[+] fetching book detail {url}", file=sys.stderr)
        book_id = urlparse(url).path.rsplit("/", 1)[-1]
        title, author, category, first_chapter = _parse_detail(fetcher.get_html(url))
        if not category and first_chapter:
            category, _book_id, _chapter_id = _parse_chapter_url(first_chapter)

    if not book_id:
        raise ValueError(f"book id not found from {book_url}")

    refs, total = parse_catalogs(fetcher, book_id)
    if not refs:
        raise ValueError(f"no chapters found for ibusread book {book_id}")
    if not category:
        category, _book_id, _chapter_id = _parse_chapter_url(refs[0].url)
    print(f"[+] book: {title} / {author}", file=sys.stderr)
    print(f"[+] {len(refs)} chapter(s) discovered" + (f" of {total}" if total else ""), file=sys.stderr)

    volumes = crawl_volumes(refs, delay=delay, concurrency=concurrency)
    meta = BookMeta(
        book_id=book_id,
        category=category,
        title=title or "未命名",
        author=author,
        intro_paragraphs=[],
    )
    return meta, volumes


def build_epub(meta: BookMeta, volumes: list[Volume], out_path: Path) -> None:
    write_epub(
        identifier=f"ibusread-{meta.book_id}-{int(time.time())}",
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        out_path=out_path,
        intro_paragraphs=meta.intro_paragraphs,
        emit_single_volume_cover=False,
    )


def _resolve_book_url(arg: str) -> str:
    return _resolve_chapter_url(arg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", help="Book URL on ibusread.com or just the book id")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    meta, volumes = crawl_book(
        _resolve_book_url(args.book),
        delay=args.delay,
        concurrency=args.concurrency,
    )
    out_path = resolve_output_path(args.output, meta.title, meta.author)
    build_epub(meta, volumes, out_path)
    print(f"[+] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
