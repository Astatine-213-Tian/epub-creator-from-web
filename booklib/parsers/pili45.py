#!/usr/bin/env python3
"""Scrape a book from pili45.com (霹雳书屋) and build an EPUB.

Usage:
    python -m booklib.parsers.pili45 <book_url_or_id> [-o output.epub] [--concurrency N]

Examples:
    python -m booklib.parsers.pili45 https://www.pili45.com/5/2965/info.html
    python -m booklib.parsers.pili45 2965          # defaults to category 5
    python -m booklib.parsers.pili45 5/2965        # explicit category

The site returns a Cloudflare JavaScript challenge to raw HTTP clients, so this
parser follows the existing browser-backed approach used by xfxs.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import zendriver as zd

from booklib import Chapter, Volume, write_epub


HOST = "https://www.pili45.com"
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

INFO_RE = re.compile(r"^/(\d+)/(\d+)/info\.html$")
CHAPTER_RE = re.compile(r"^/(\d+)/(\d+)/read/(\d+)\.html$")


# ---------------------------------------------------------------------------
# Browser session


class Fetcher:
    def __init__(self, *, headless: bool = False, delay: float = 0.4):
        self.headless = headless
        self.delay = delay
        self.browser: zd.Browser | None = None

    async def start(self) -> None:
        config = zd.Config(
            headless=self.headless,
            browser_executable_path=CHROME_PATH,
            sandbox=False,
            browser_connection_timeout=1.0,
            browser_connection_max_tries=30,
        )
        self.browser = await zd.start(config)

    async def stop(self) -> None:
        if self.browser:
            await self.browser.stop()

    async def get_html(self, url: str) -> str:
        assert self.browser is not None
        page = await self.browser.get(url)
        for _ in range(60):
            title = await page.evaluate("document.title")
            if title and "moment" not in title.lower() and "稍候" not in title:
                break
            await asyncio.sleep(1)
        for _ in range(60):
            ready = await page.evaluate(
                """
                Boolean(document.querySelector(
                  '.works-intro-title, .works-chapter-list, .read-content, .j_readContent'
                ))
                """
            )
            if ready:
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(self.delay)
        return await page.get_content()

    async def get_bytes(self, url: str) -> bytes:
        assert self.browser is not None
        page = self.browser.main_tab
        if page is None:
            page = await self.browser.get("about:blank")
        b64 = await page.evaluate(
            f"""
            (async () => {{
              const r = await fetch({url!r}, {{credentials: 'include'}});
              const buf = new Uint8Array(await r.arrayBuffer());
              let s = '';
              for (const b of buf) s += String.fromCharCode(b);
              return btoa(s);
            }})()
            """,
            await_promise=True,
        )
        return base64.b64decode(b64)

    async def fetch_html(self, url: str) -> str:
        assert self.browser is not None
        page = self.browser.main_tab
        if page is None:
            page = await self.browser.get("about:blank")
        return await page.evaluate(
            f"""
            (async () => {{
              const r = await fetch({url!r}, {{credentials: 'include'}});
              if (!r.ok) throw new Error(`HTTP ${{r.status}} for {url}`);
              return await r.text();
            }})()
            """,
            await_promise=True,
        )


# ---------------------------------------------------------------------------
# Models


@dataclass
class BookMeta:
    cat_id: str
    book_id: str
    title: str
    author: str
    intro_paragraphs: list[str]
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


def parse_info(html: str, cat_id: str, book_id: str) -> BookMeta:
    soup = BeautifulSoup(html, "lxml")
    title = ""
    author = ""
    status = ""
    cover_url: str | None = None

    h = soup.find("h2", class_="works-intro-title")
    if h:
        strong = h.find("strong")
        if strong:
            title = _clean_text(strong.get_text())
        text = _clean_text(h.get_text())
        m = re.search(r"作者[：:]\s*([^）)]+)", text)
        if m:
            author = _clean_text(m.group(1))

    cover_div = soup.find("div", class_="works-cover")
    if cover_div:
        status_span = cover_div.find("span")
        if status_span:
            status = _clean_text(status_span.get_text())
        img = cover_div.find("img")
        if img and img.get("src") and "nocover" not in img["src"]:
            cover_url = urljoin(HOST, img["src"])

    intro = soup.find("p", class_="works-intro-short")
    intro_paragraphs: list[str] = []
    if intro:
        for tag in intro.find_all(["script", "style", "iframe", "ins"]):
            tag.decompose()
        for br in intro.find_all("br"):
            br.replace_with("\n")
        for chunk in re.split(r"\n+", intro.get_text("\n")):
            chunk = chunk.strip(" \t\xa0　")
            if chunk:
                intro_paragraphs.append(chunk)

    if not title:
        page_title = soup.title.string if soup.title and soup.title.string else ""
        m = re.match(r"^《([^》]+)》.*?_([^_]+)_", page_title)
        if m:
            title = m.group(1).strip()
            author = author or m.group(2).strip()

    return BookMeta(
        cat_id=cat_id,
        book_id=book_id,
        title=title or "未命名",
        author=author,
        intro_paragraphs=intro_paragraphs,
        status=status,
        cover_url=cover_url,
    )


def parse_toc(html: str, cat_id: str, book_id: str) -> tuple[list[ChapterRef], str | None]:
    soup = BeautifulSoup(html, "lxml")
    container = (
        soup.find(class_="works-chapter-list-con")
        or soup.find(class_="works-chapter-item")
        or soup.find(class_="works-chapter-list")
        or soup
    )
    refs: list[ChapterRef] = []
    seen: set[str] = set()
    for a in container.find_all("a", href=True):
        m = CHAPTER_RE.match(urlparse(a["href"]).path)
        if not m or m.group(1) != cat_id or m.group(2) != book_id:
            continue
        chapter_id = m.group(3)
        if chapter_id in seen:
            continue
        seen.add(chapter_id)
        title = _clean_text(a.get("title") or a.get_text())
        if not title:
            continue
        refs.append(
            ChapterRef(
                title=title,
                url=urljoin(HOST, a["href"]),
                chapter_id=chapter_id,
            )
        )

    next_url: str | None = None
    for a in soup.find_all("a", href=True):
        if "下一页" in _clean_text(a.get_text()) and "/menu/" in a["href"]:
            next_url = urljoin(HOST, a["href"])
            break
    return refs, next_url


_BOILERPLATE_RE = re.compile(
    r"^霹雳书屋|本站所有小说|所有内容版权|最新网址|手机用户请到|加入书签|推荐本书|返回目录$"
)


def parse_chapter(html: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "lxml")
    title = ""
    h = soup.find(class_="j_chapterName") or soup.find("div", class_="text-head")
    if h:
        title = _clean_text(h.get_text())
    if not title:
        page_title = soup.title.string if soup.title and soup.title.string else ""
        m = re.match(r"^《[^》]+》([^_]+)在线阅读", page_title)
        if m:
            title = _clean_text(m.group(1))

    content = soup.find("div", class_="read-content") or soup.find(class_="j_readContent")
    paragraphs: list[str] = []
    if content:
        for tag in content.find_all(["script", "style", "iframe", "ins"]):
            tag.decompose()
        for p in content.find_all("p"):
            text = _clean_text(p.get_text())
            if not text or text == "0" or _BOILERPLATE_RE.search(text):
                continue
            paragraphs.append(text)
        if not paragraphs:
            for br in content.find_all("br"):
                br.replace_with("\n")
            for chunk in re.split(r"\n+", content.get_text("\n")):
                text = _clean_text(chunk)
                if text and not _BOILERPLATE_RE.search(text):
                    paragraphs.append(text)
    return title, paragraphs


# ---------------------------------------------------------------------------
# Crawl


async def crawl_book(
    book_url: str,
    *,
    headless: bool = False,
    delay: float = 0.4,
    concurrency: int = 4,
) -> tuple[BookMeta, list[Volume]]:
    parsed = urlparse(book_url)
    m = INFO_RE.match(parsed.path)
    if not m:
        raise ValueError(f"not a /<cat>/<id>/info.html URL: {book_url}")
    cat_id, book_id = m.group(1), m.group(2)

    fetcher = Fetcher(headless=headless, delay=delay)
    await fetcher.start()
    try:
        info_url = urljoin(HOST, f"/{cat_id}/{book_id}/info.html")
        print(f"[+] fetching info {info_url}", file=sys.stderr)
        info_html = await fetcher.get_html(info_url)
        meta = parse_info(info_html, cat_id, book_id)
        print(f"[+] book: {meta.title} / {meta.author}", file=sys.stderr)

        if meta.cover_url:
            try:
                meta.cover_bytes = await fetcher.get_bytes(meta.cover_url)
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

        refs: list[ChapterRef] = []
        menu_url: str | None = urljoin(HOST, f"/{cat_id}/{book_id}/menu/1.html")
        while menu_url:
            print(f"[+] fetching menu {menu_url}", file=sys.stderr)
            page_refs, next_url = parse_toc(
                await fetcher.get_html(menu_url), cat_id, book_id
            )
            refs.extend(page_refs)
            menu_url = next_url
        print(f"[+] {len(refs)} chapters discovered", file=sys.stderr)

        chapters = await crawl_chapters(fetcher, refs, concurrency=concurrency)
        return meta, [Volume(title="", chapters=chapters)]
    finally:
        await fetcher.stop()


async def crawl_chapter(fetcher: Fetcher, ref: ChapterRef) -> Chapter | None:
    for attempt in range(5):
        try:
            title, paragraphs = parse_chapter(await fetcher.fetch_html(ref.url))
            if not paragraphs:
                return None
            return Chapter(title=title or ref.title, paragraphs=paragraphs)
        except Exception as exc:
            if attempt == 4:
                raise
            print(
                f"[!] fetch failed for {ref.title}: {exc}; retrying ({attempt + 2}/5)",
                file=sys.stderr,
            )
            await asyncio.sleep(1.0 * (attempt + 1))
    return None


async def crawl_chapter_with_navigation(fetcher: Fetcher, ref: ChapterRef) -> Chapter | None:
    for attempt in range(5):
        try:
            title, paragraphs = parse_chapter(await fetcher.get_html(ref.url))
            if not paragraphs:
                return None
            return Chapter(title=title or ref.title, paragraphs=paragraphs)
        except Exception as exc:
            if attempt == 4:
                raise
            print(
                f"[!] navigation failed for {ref.title}: {exc}; retrying ({attempt + 2}/5)",
                file=sys.stderr,
            )
            await asyncio.sleep(1.0 * (attempt + 1))
    return None


async def crawl_chapters(
    fetcher: Fetcher,
    refs: list[ChapterRef],
    *,
    concurrency: int = 4,
) -> list[Chapter]:
    if not refs:
        return []

    limit = max(1, concurrency)
    total = len(refs)
    if limit == 1:
        chapters: list[Chapter] = []
        for index, ref in enumerate(refs):
            print(f"[+] [{index + 1}/{total}] {ref.title}", file=sys.stderr)
            chapter = await crawl_chapter_with_navigation(fetcher, ref)
            if chapter:
                chapters.append(chapter)
        return chapters

    semaphore = asyncio.Semaphore(limit)

    async def fetch_one(index: int, ref: ChapterRef) -> Chapter | None:
        async with semaphore:
            print(f"[+] [{index + 1}/{total}] {ref.title}", file=sys.stderr)
            return await crawl_chapter(fetcher, ref)

    chapters = await asyncio.gather(
        *(fetch_one(index, ref) for index, ref in enumerate(refs))
    )
    return [chapter for chapter in chapters if chapter is not None]


def build_epub(meta: BookMeta, volumes: list[Volume], out_path: Path) -> None:
    write_epub(
        identifier=f"pili45-{meta.book_id}-{int(time.time())}",
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
    if "/" in arg:
        cat, bid = arg.split("/", 1)
        return f"{HOST}/{cat}/{bid}/info.html"
    return f"{HOST}/5/{arg}/info.html"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("book", help="Book URL on pili45.com or just the book id")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--delay", type=float, default=0.4)
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of chapter pages to fetch concurrently",
    )
    args = p.parse_args(argv)

    book_url = _resolve_book_url(args.book)
    meta, volumes = asyncio.run(
        crawl_book(
            book_url,
            headless=args.headless,
            delay=args.delay,
            concurrency=args.concurrency,
        )
    )

    n_chap = sum(len(v.chapters) for v in volumes)
    print(f"[+] crawled {n_chap} chapter(s)", file=sys.stderr)

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
