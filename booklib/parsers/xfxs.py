#!/usr/bin/env python3
"""Scrape a book from xfxs1.com (先锋小说网) and build an EPUB.

Usage:
    python xfxs_to_epub.py <book_url_or_id> [-o output.epub]

Examples:
    python xfxs_to_epub.py https://www.xfxs1.com/goreadbook/2287/
    python xfxs_to_epub.py 2287

The site is fronted by Cloudflare with an interactive Turnstile challenge.
We use ``zendriver`` to bring up a real Chrome instance, solve the challenge
once per run, then crawl all chapter pages within that browser session.
A visible window is required for the first request: Cloudflare may demand a
manual click; subsequent navigations reuse the same cookies automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import zendriver as zd

from booklib import Chapter, Volume, write_epub


HOST = "https://www.xfxs1.com"
BOOK_INDEX_RE = re.compile(r"/goreadbook/(\d+)/?$")
CHAPTER_RE = re.compile(r"/goreadbook/(\d+)/(\d+)(?:_(\d+))?\.html$")
BAD_TOC_TITLE_RE = re.compile(r"[”」』]|。")

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
INTRO_REVIEW_HEADING_RE = re.compile(
    r"^(?:[^，。！？；\n]{0,40})?(?:(?:作品)?简评|编辑评价|强推奖章)\s*[:：】）)]?$"
)

# Volume + chapter markers (kept for forward-compat — this book has none).
VOL_RE = re.compile(r"^\s*第\s*[一二三四五六七八九十百零兩两\d]+\s*卷\s*(.*)$")


# ---------------------------------------------------------------------------
# Browser session


class Fetcher:
    """Wraps a single zendriver browser; bypass Cloudflare once, reuse for all."""

    def __init__(
        self,
        *,
        headless: bool = False,
        delay: float = 1.0,
        request_interval: float = 0.0,
    ):
        self.headless = headless
        self.delay = delay
        self.request_interval = request_interval
        self.browser: zd.Browser | None = None
        self._cf_passed = False
        self._shared_fetch_page: zd.Tab | None = None
        self._fetch_pages: list[zd.Tab] = []
        self._fetch_page_queue: asyncio.Queue[zd.Tab] | None = None
        self._request_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def _wait_for_request_slot(self) -> None:
        if self.request_interval <= 0:
            return
        async with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_at
            if elapsed < self.request_interval:
                await asyncio.sleep(self.request_interval - elapsed)
            self._last_request_at = time.monotonic()

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
        for page in self._fetch_pages:
            try:
                await page.close()
            except Exception as exc:  # noqa: BLE001
                logging.debug("closing fetch tab failed: %s", exc)
        self._fetch_pages.clear()
        self._fetch_page_queue = None
        if self.browser:
            await self.browser.stop()

    async def prepare_fetch_pages(self, count: int) -> None:
        """Create a small tab pool for concurrent JS fetch calls."""
        assert self.browser is not None
        wanted = max(1, count)
        try:
            while len(self._fetch_pages) < wanted:
                page = await self.browser.get("about:blank", new_tab=True)
                await page.get(HOST + "/")
                self._fetch_pages.append(page)
            self._fetch_page_queue = asyncio.Queue()
            for page in self._fetch_pages[:wanted]:
                self._fetch_page_queue.put_nowait(page)
        except Exception as exc:
            print(
                f"[!] multi-tab setup failed ({exc}); using shared-tab fetch",
                file=sys.stderr,
            )
            self._fetch_page_queue = None

    async def get_bytes(self, url: str) -> bytes:
        """Fetch a binary resource (e.g. cover image) reusing the CF cookie."""
        assert self.browser is not None
        # zendriver doesn't expose a raw fetch — drive `fetch()` from a JS page.
        page = self.browser.main_tab
        if page is None:
            page = await self.browser.get("about:blank")
        b64 = await page.evaluate(
            f"""
            (async () => {{
              const r = await fetch({url!r}, {{credentials: 'include'}});
              const buf = new Uint8Array(await r.arrayBuffer());
              let s=''; for (const b of buf) s += String.fromCharCode(b);
              return btoa(s);
            }})()
            """,
            await_promise=True,
        )
        return base64.b64decode(b64)

    async def fetch_html(self, url: str) -> str:
        """Fetch HTML bytes without navigating the shared browser tab.

        xfxs1 pages are served as GBK. Browser ``response.text()`` can produce
        replacement-character mojibake for some chapter pages, so decode the raw
        response bytes on the Python side where the EPUB writer expects text.
        """
        assert self.browser is not None
        await self._wait_for_request_slot()
        page = None
        if self._fetch_page_queue is not None:
            page = await self._fetch_page_queue.get()
        else:
            page = self._shared_fetch_page
            if page is None:
                page = await self.browser.get(HOST + "/")
                self._shared_fetch_page = page
        try:
            b64 = await page.evaluate(
                f"""
                (async () => {{
                  const r = await fetch({url!r}, {{credentials: 'include'}});
                  if (!r.ok) throw new Error(`HTTP ${{r.status}} for {url}`);
                  const buf = new Uint8Array(await r.arrayBuffer());
                  let s=''; for (const b of buf) s += String.fromCharCode(b);
                  return btoa(s);
                }})()
                """,
                await_promise=True,
            )
        finally:
            if self._fetch_page_queue is not None:
                self._fetch_page_queue.put_nowait(page)
        await asyncio.sleep(self.delay)
        data = base64.b64decode(b64)
        return data.decode("gbk", errors="replace")

    async def get_html(self, url: str) -> str:
        assert self.browser is not None
        page = await self.browser.get(url)
        self._shared_fetch_page = page
        # Only attempt CF verification on the first navigation; subsequent
        # ones reuse the cookie. ``verify_cf`` raises on no challenge — that
        # is fine, swallow it.
        if not self._cf_passed:
            try:
                await page.verify_cf()
            except Exception as e:  # noqa: BLE001 — verify_cf raises broad
                logging.debug("verify_cf: %s", e)
            self._cf_passed = True
        # Wait for "moment"/"稍候" to disappear from <title>.
        for _ in range(60):
            t = await page.evaluate("document.title")
            if t and "moment" not in t.lower() and "稍候" not in t:
                break
            await asyncio.sleep(1)
        await asyncio.sleep(self.delay)
        return await page.get_content()


# ---------------------------------------------------------------------------
# Parsers


@dataclass
class BookMeta:
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


def _clean_text(s: str) -> str:
    return re.sub(r"[　\xa0\s]+", " ", s).strip()


def _is_intro_review_heading(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    if not compact:
        return False
    if not INTRO_REVIEW_HEADING_RE.match(compact):
        return False
    return (
        "作品简评" in compact
        or "编辑评价" in compact
        or "强推奖章" in compact
        or compact.startswith("简评")
        or ".gif" in compact.lower()
        or ".jpg" in compact.lower()
        or ".png" in compact.lower()
    )


def _clean_intro_paragraphs(text: str) -> list[str]:
    text = re.sub(r"^\s*小说简介\s*[:：]?\s*", "", text)
    paragraphs: list[str] = []
    for chunk in re.split(r"\n+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if _is_intro_review_heading(chunk):
            break
        paragraphs.append(chunk)
    return paragraphs


def _normalized_text_parts(paragraphs: list[str]) -> set[str]:
    parts: set[str] = set()
    for paragraph in paragraphs:
        text = re.sub(r"\s+", "", paragraph)
        text = re.sub(r"[，。！？、；：:,.!?;《》〈〉“”\"'‘’\[\]【】（）()\-—_·┃|]", "", text)
        if len(text) >= 8:
            parts.add(text)
    return parts


def _chapter_repeats_intro(chapter: Chapter, intro_paragraphs: list[str]) -> bool:
    if not chapter.title.strip().startswith("第0章"):
        return False
    metadata_markers = ("文案", "内容标签", "搜索关键字", "一句话简介", "立意")
    marker_hits = sum(
        1
        for paragraph in chapter.paragraphs
        for marker in metadata_markers
        if marker in paragraph
    )
    if marker_hits >= 3:
        return True
    intro_parts = _normalized_text_parts(intro_paragraphs)
    chapter_parts = _normalized_text_parts(chapter.paragraphs)
    if not intro_parts or not chapter_parts:
        return False
    repeated = sum(1 for part in intro_parts if part in chapter_parts)
    return repeated / len(intro_parts) >= 0.6


def _drop_repeated_intro_chapter(volumes: list[Volume], meta: BookMeta) -> None:
    for volume in volumes:
        if not volume.chapters:
            continue
        first = volume.chapters[0]
        if _chapter_repeats_intro(first, meta.intro_paragraphs):
            print(f"[+] dropping duplicated intro chapter: {first.title}", file=sys.stderr)
            del volume.chapters[0]
        return


def parse_book_index(html: str, book_id: str) -> BookMeta:
    soup = BeautifulSoup(html, "lxml")
    book_div = soup.find("div", class_="book")
    title = ""
    author = ""
    status = ""
    cover_url: str | None = None

    if book_div:
        # Title lives in <div class="right"><h1><a>title</a></h1>
        right = book_div.find("div", class_="right")
        if right:
            h1 = right.find("h1")
            if h1:
                title = _clean_text(h1.get_text())
            # Author: <span><i>作者：</i><a>非天夜翔</a></span>
            for span in right.find_all("span"):
                label = span.find("i")
                if label and "作者" in label.get_text():
                    a = span.find("a")
                    author = _clean_text(a.get_text()) if a else _clean_text(
                        span.get_text().replace("作者：", "").replace("作者:", "")
                    )
                    break
        # Status: <div class="cover"><span>言情 / 已完成</span>
        cover_div = book_div.find("div", class_="cover")
        if cover_div:
            sp = cover_div.find("span")
            if sp:
                txt = _clean_text(sp.get_text())
                status = txt.split("/")[-1].strip() if "/" in txt else txt
            img = cover_div.find("img")
            if img and img.get("src"):
                cover_url = urljoin(HOST, img["src"])
        if not cover_url:
            # Fall back to the mobile cover image.
            mobile = book_div.find("img", class_="backcover")
            if mobile and mobile.get("src"):
                cover_url = urljoin(HOST, mobile["src"])

    if not title:
        # Fall back to <title>: "王子病的春天(非天夜翔)最新章节,全文阅读 - 先锋小说网"
        page_title = soup.title.string if soup.title else ""
        m = re.match(r"^([^()（）]+)\s*[（(]([^()（）]+)[)）]", page_title or "")
        if m:
            title = m.group(1).strip()
            if not author:
                author = m.group(2).strip()

    intro_div = soup.find("div", class_="intro")
    intro_paragraphs: list[str] = []
    if intro_div:
        text = intro_div.get_text("\n")
        intro_paragraphs = _clean_intro_paragraphs(text)

    return BookMeta(
        book_id=book_id,
        title=title or "未命名",
        author=author,
        intro_paragraphs=intro_paragraphs,
        status=status,
        cover_url=cover_url,
    )


def parse_toc(html: str, book_id: str) -> tuple[list[ChapterRef], str | None]:
    soup = BeautifulSoup(html, "lxml")
    list_chapter = soup.find("div", class_="list-chapter") or soup
    refs: list[ChapterRef] = []
    seen: set[str] = set()
    for a in list_chapter.find_all("a", href=True):
        m = CHAPTER_RE.search(a["href"])
        if not m or m.group(1) != book_id:
            continue
        if m.group(3):  # this is a paginated _N url, skip — we follow next-links
            continue
        cid = m.group(2)
        if cid in seen:
            continue
        seen.add(cid)
        title = _clean_text(a.get_text())
        if not title:
            continue
        refs.append(
            ChapterRef(
                title=title,
                url=urljoin(HOST, a["href"]),
                chapter_id=cid,
            )
        )
    _repair_bad_toc_titles(refs)

    next_url: str | None = None
    for a in soup.find_all("a", href=True):
        if "下一页" not in _clean_text(a.get_text()):
            continue
        href = a["href"]
        if re.search(rf"/2/{re.escape(book_id)}/\d+/?$", href):
            next_url = urljoin(HOST, href)
            break
    return refs, next_url


def _repair_bad_toc_titles(refs: list[ChapterRef]) -> None:
    previous_title = ""
    for ref in refs:
        if BAD_TOC_TITLE_RE.search(ref.title) and previous_title:
            ref.title = f"{previous_title}（续）"
        else:
            previous_title = ref.title


_NAV_RE = re.compile(
    r"上[\d一二三四五六七八九十]?一?章|下[\d一二三四五六七八九十]?一?页|目录|加书签"
)
# End-of-page advert / continuation strings the site stamps into the body
# itself (so they survive HTML structure-based filtering).
_PAGE_BREAK_RE = re.compile(
    r"本章未完[，,].{0,30}(下一页|继续阅读)"   # 本章未完，点击下一页继续阅读
    r"|(请)?点击下一页继续阅读"
    r"|^[（(]?本章完[)）]?$"
    r"|为您提供.{0,30}的小说"
    r"|继续阅读请点击下一?页"
)


def _extract_chapter_paragraphs(html: str) -> tuple[list[str], str | None, str | None]:
    """Return (paragraphs, page_title, next_page_url_if_same_chapter)."""
    soup = BeautifulSoup(html, "lxml")
    page_title = soup.title.string.strip() if soup.title and soup.title.string else None
    content = soup.find(id="chaptercontent")
    paragraphs: list[str] = []
    if content:
        # The site uses raw text with leading 4-space / fullwidth indent and
        # newline separators — split on blank lines to get paragraphs.
        for br in content.find_all("br"):
            br.replace_with("\n")
        text = content.get_text("\n")
        for chunk in re.split(r"\n+", text):
            chunk = chunk.strip("　 \t\xa0")
            # Strip the in-content nav garbage some pages embed.
            if not chunk:
                continue
            if _NAV_RE.fullmatch(chunk):
                continue
            if _PAGE_BREAK_RE.search(chunk):
                continue
            paragraphs.append(chunk)

    # Find next-page link (next within same chapter — `<id>_<n>.html`).
    next_url: str | None = None
    next_a = soup.find("a", id="next_url")
    if next_a and next_a.get("href"):
        next_url = next_a["href"]
    return paragraphs, page_title, next_url


# ---------------------------------------------------------------------------
# Crawl


async def crawl_chapter(
    fetcher: Fetcher,
    ref: ChapterRef,
    *,
    use_navigation: bool = False,
) -> Chapter:
    """Fetch a chapter, following pagination until we hit a different chapter id."""
    paragraphs: list[str] = []
    url = ref.url
    visited: set[str] = set()
    chapter = Chapter(title=ref.title)

    while url and url not in visited:
        visited.add(url)
        paras: list[str] = []
        page_title: str | None = None
        next_url: str | None = None
        for attempt in range(5):
            try:
                if use_navigation:
                    html = await fetcher.get_html(url)
                else:
                    html = await fetcher.fetch_html(url)
                paras, page_title, next_url = _extract_chapter_paragraphs(html)
                if paras:
                    break
                if attempt == 4:
                    raise RuntimeError(f"empty chapter body: {url}")
                print(f"[!] empty chapter body, retrying: {url}", file=sys.stderr)
            except Exception as exc:
                if attempt == 4:
                    raise
                message = str(exc)
                retry_delay = 5.0 * (attempt + 1) if (
                    "HTTP 429" in message or "HTTP 520" in message
                ) else 1.0 * (attempt + 1)
                print(
                    f"[!] fetch failed for {ref.title}: {exc}; "
                    f"retrying ({attempt + 2}/5) in {retry_delay:.1f}s",
                    file=sys.stderr,
                )
                await asyncio.sleep(retry_delay)
                continue
            await asyncio.sleep(1.0 * (attempt + 1))
        # First page may include the chapter heading inline; trim it if so.
        if not paragraphs and paras:
            heading = paras[0]
            if ref.title in heading or heading.startswith(("第", "Chapter")):
                # Replace heading with a cleaner version we already have.
                paras = paras[1:]
        paragraphs.extend(paras)

        if not next_url:
            break
        # Stop when the next link is no longer this chapter's pagination.
        m_cur = CHAPTER_RE.search(urlparse(url).path)
        m_next = CHAPTER_RE.search(urlparse(next_url).path)
        if not m_next:
            break
        if m_cur and m_cur.group(2) != m_next.group(2):
            break  # next link is the next chapter, stop here
        url = urljoin(HOST, next_url)

    chapter.paragraphs = paragraphs
    return chapter


async def crawl_chapters(
    fetcher: Fetcher,
    refs: list[ChapterRef],
    *,
    concurrency: int = 4,
) -> list[Chapter]:
    if not refs:
        return []

    total = len(refs)
    limit = max(1, concurrency)
    if limit == 1:
        chapters: list[Chapter] = []
        for index, ref in enumerate(refs):
            print(f"[+] [{index + 1}/{total}] {ref.title}", file=sys.stderr)
            chapter = await crawl_chapter(fetcher, ref, use_navigation=True)
            chapters.append(chapter)
            remaining = total - len(chapters)
            print(
                f"[=] progress: done={len(chapters)}/{total}, "
                f"remaining={remaining}, current={ref.title}",
                file=sys.stderr,
            )
        return chapters

    semaphore = asyncio.Semaphore(limit)
    progress_lock = asyncio.Lock()
    completed = 0
    failed = 0

    async def log_progress(label: str, ref: ChapterRef) -> None:
        remaining = total - completed - failed
        print(
            f"[=] progress: done={completed}/{total}, "
            f"failed={failed}, remaining={remaining}, "
            f"{label}={ref.title}",
            file=sys.stderr,
        )

    async def fetch_one(index: int, ref: ChapterRef) -> Chapter:
        nonlocal completed, failed
        async with semaphore:
            print(f"[+] [{index + 1}/{total}] {ref.title}", file=sys.stderr)
            try:
                chapter = await crawl_chapter(fetcher, ref)
            except Exception:
                async with progress_lock:
                    failed += 1
                    await log_progress("failed_chapter", ref)
                raise
            async with progress_lock:
                completed += 1
                await log_progress("current", ref)
            return chapter

    results: list[Chapter | BaseException] = await asyncio.gather(
        *(fetch_one(index, ref) for index, ref in enumerate(refs)),
        return_exceptions=True,
    )
    failed_indexes = [
        index
        for index, result in enumerate(results)
        if isinstance(result, BaseException)
    ]
    if failed_indexes:
        print(
            f"[!] fast path failed for {len(failed_indexes)} chapter(s); "
            "retrying serial navigation fallback",
            file=sys.stderr,
        )
    for index in failed_indexes:
        ref = refs[index]
        print(f"[>] fallback [{index + 1}/{total}] {ref.title}", file=sys.stderr)
        try:
            chapter = await crawl_chapter(fetcher, ref, use_navigation=True)
        except Exception as exc:
            print(f"[!] fallback failed for {ref.title}: {exc}", file=sys.stderr)
            raise RuntimeError(
                f"chapter fetch failed after fallback: [{index + 1}/{total}] {ref.title}"
            ) from exc
        results[index] = chapter
        async with progress_lock:
            completed += 1
            failed -= 1
            await log_progress("fallback_recovered", ref)

    return [result for result in results if isinstance(result, Chapter)]


async def crawl_book(
    book_url: str,
    *,
    headless: bool = False,
    delay: float = 0.4,
    concurrency: int = 2,
    request_interval: float = 0.0,
) -> tuple[BookMeta, list[Volume]]:
    parsed = urlparse(book_url)
    m = BOOK_INDEX_RE.search(parsed.path.rstrip("/") + "/")
    if not m:
        raise ValueError(f"not a book index URL: {book_url}")
    book_id = m.group(1)
    index_url = urljoin(HOST, f"/goreadbook/{book_id}/")
    toc_url = urljoin(HOST, f"/2/{book_id}/")

    fetcher = Fetcher(
        headless=headless,
        delay=delay,
        request_interval=request_interval,
    )
    await fetcher.start()
    try:
        meta: BookMeta | None = None
        for attempt in range(3):
            print(f"[+] fetching index {index_url}", file=sys.stderr)
            idx_html = await fetcher.get_html(index_url)
            meta = parse_book_index(idx_html, book_id)
            if meta.title != "未命名" and meta.author:
                break
            if attempt == 2:
                raise RuntimeError(f"book index did not load metadata: {index_url}")
            print("[!] book metadata incomplete, retrying index", file=sys.stderr)
            await asyncio.sleep(3.0)
        assert meta is not None
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
                print(f"[+] cover {len(meta.cover_bytes)} bytes ({meta.cover_url})",
                      file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"[!] cover fetch failed: {e}", file=sys.stderr)

        refs: list[ChapterRef] = []
        seen_ref_ids: set[str] = set()
        while toc_url:
            print(f"[+] fetching toc {toc_url}", file=sys.stderr)
            toc_html = await fetcher.get_html(toc_url)
            page_refs, next_url = parse_toc(toc_html, book_id)
            for ref in page_refs:
                if ref.chapter_id in seen_ref_ids:
                    continue
                seen_ref_ids.add(ref.chapter_id)
                refs.append(ref)
            toc_url = next_url
        print(f"[+] {len(refs)} chapters discovered", file=sys.stderr)

        volumes = [Volume(title="", chapters=[])]
        if concurrency > 2:
            await fetcher.prepare_fetch_pages(concurrency)
        volumes[0].chapters = await crawl_chapters(
            fetcher,
            refs,
            concurrency=concurrency,
        )
        _drop_repeated_intro_chapter(volumes, meta)
        return meta, volumes
    finally:
        await fetcher.stop()


# ---------------------------------------------------------------------------
def build_epub(meta: BookMeta, volumes: list[Volume], out_path: Path) -> None:
    write_epub(
        identifier=f"xfxs1-{meta.book_id}-{int(time.time())}",
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        out_path=out_path,
        intro_paragraphs=meta.intro_paragraphs,
        cover_bytes=meta.cover_bytes,
        cover_mime=meta.cover_mime,
    )


# ---------------------------------------------------------------------------
# Main


def _resolve_book_url(arg: str) -> str:
    if arg.isdigit():
        return f"{HOST}/goreadbook/{arg}/"
    return arg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("book", help="Book URL on xfxs1.com or just the book id")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--headless", action="store_true",
                   help="Run Chrome headless (CF challenge often fails — use only "
                        "if you've previously cached a session)")
    p.add_argument("--delay", type=float, default=0.4)
    p.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between chapter page fetches across all workers",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Maximum number of chapter pages to fetch concurrently",
    )
    args = p.parse_args(argv)

    book_url = _resolve_book_url(args.book)
    meta, volumes = asyncio.run(crawl_book(book_url, headless=args.headless,
                                            delay=args.delay,
                                            concurrency=args.concurrency,
                                            request_interval=args.request_interval))

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
