#!/usr/bin/env python3
"""Scrape a book from mangguoshufang.com (芒果书坊) and build an EPUB.

Usage:
    python mgsf_to_epub.py <book_url_or_id> [-o output.epub]

Examples:
    python mgsf_to_epub.py https://www.mangguoshufang.com/1/2574/info.html
    python mgsf_to_epub.py 2574          # defaults to category 1
    python mgsf_to_epub.py 1/2574        # explicit category

This site is plain HTTP — no Cloudflare or auth. Volumes are encoded inside
chapter titles like ``第1章 卷一 摸鱼儿`` so we extract the volume name from
the per-chapter heading.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src import Chapter, Volume, write_epub


HOST = "https://www.mangguoshufang.com"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)

INFO_RE = re.compile(r"^/(\d+)/(\d+)/info\.html$")
CHAPTER_RE = re.compile(r"^/(\d+)/(\d+)/read/(\d+)\.html$")

# Chapter heading split. The menu version has no spaces:
#   "第1章卷一摸鱼儿"           -> chap="第1章" vol_num="一" vol_name="摸鱼儿"
#   "第282章卷五八声甘州"        -> chap="第282章" vol_num="五" vol_name="八声甘州"
#   "第299章2022中秋番外：千星"  -> chap="第299章" vol_num=""  vol_name=""
# The chapter-page title is space-separated ("第1章 卷一 摸鱼儿") and parsing
# becomes trivial once we strip whitespace, so the same regex works for both.
# Chinese ordinal up to ~99 inclusive: 一..十, 十一..十九, 二十..九十九, 百.
_CN_ORDINAL = r"(?:[一二三四五六七八九]?十[一二三四五六七八九]?|百|[一二三四五六七八九])"
CHAPTER_TITLE_PARTS_RE = re.compile(
    r"^\s*(?P<chap>第\s*[一二三四五六七八九十百零兩两\d]+\s*章)\s*"
    r"(?:"
        r"卷\s*(?P<vol_num>" + _CN_ORDINAL + r"|\d+)\s*(?P<vol_name>.*)"
        r"|"
        r"(?P<rest>.*)"
    r")\s*$"
)


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
    cat_id: str
    book_id: str
    title: str
    author: str
    intro_paragraphs: list[str]
    cover_url: str | None = None
    cover_bytes: bytes | None = None
    cover_mime: str = "image/jpeg"


@dataclass
class ChapterRef:
    raw_title: str          # original text from menu link, e.g. "第1章卷一摸鱼儿"
    chap_label: str         # "第1章"
    vol_label: str          # "卷一 摸鱼儿" (may be empty)
    chap_name: str          # part after volume, can be empty (just "第1章")
    url: str
    chapter_id: str


# ---------------------------------------------------------------------------
# Parse: info page


def _clean(s: str) -> str:
    return re.sub(r"[\s 　]+", " ", s).strip()


def parse_info(html: str, cat_id: str, book_id: str) -> BookMeta:
    soup = BeautifulSoup(html, "lxml")
    detail = soup.find("div", class_="works-intro-detail")
    title = ""
    author = ""
    cover_url: str | None = None

    if detail:
        h = detail.find("h2", class_="works-intro-title")
        if h:
            strong = h.find("strong")
            title = _clean((strong or h).get_text())
        digi = detail.find("p", class_="works-intro-digi")
        if digi:
            em = digi.find("em")
            if em:
                author = _clean(em.get_text())

    # Intro paragraph: drop ads/scripts first.
    intro_p = soup.find("p", class_="works-intro-short")
    intro_paragraphs: list[str] = []
    if intro_p:
        for tag in intro_p.find_all(["script", "ins", "iframe", "style"]):
            tag.decompose()
        for c in intro_p.find_all(string=lambda t: t and isinstance(t, type(intro_p.string))):
            pass  # no-op, just keep BeautifulSoup type happy
        for br in intro_p.find_all("br"):
            br.replace_with("\n")
        text = intro_p.get_text("\n")
        for chunk in re.split(r"\n+", text):
            chunk = chunk.strip("　 \t\xa0")
            if chunk and chunk not in {"展示开始", "芒果-展示"}:
                intro_paragraphs.append(chunk)

    # Cover image — typically /files/works/cover/{book_id}.jpg ; placeholder
    # is /files/images/nocover.jpg which we skip.
    cover_div = soup.find("div", class_="works-cover")
    if cover_div:
        img = cover_div.find("img")
        if img and img.get("src"):
            src = img["src"]
            if "nocover" not in src:
                cover_url = urljoin(HOST, src)

    return BookMeta(
        cat_id=cat_id, book_id=book_id,
        title=title or "未命名", author=author,
        intro_paragraphs=intro_paragraphs,
        cover_url=cover_url,
    )


# ---------------------------------------------------------------------------
# Parse: menu page (chapter list)


def parse_menu(html: str, cat_id: str, book_id: str) -> tuple[list[ChapterRef], str | None]:
    """Return (chapter refs, next_menu_url_if_any)."""
    soup = BeautifulSoup(html, "lxml")
    container = soup.find("div", class_="works-chapter-list-con") or soup
    refs: list[ChapterRef] = []
    seen: set[str] = set()
    for a in container.find_all("a", href=True):
        m = CHAPTER_RE.match(urlparse(a["href"]).path)
        if not m:
            continue
        if m.group(1) != cat_id or m.group(2) != book_id:
            continue
        cid = m.group(3)
        if cid in seen:
            continue
        seen.add(cid)
        raw = _clean(a.get_text())
        parts = _split_chapter_title(raw)
        refs.append(
            ChapterRef(
                raw_title=raw,
                chap_label=parts[0],
                vol_label=parts[1],
                chap_name=parts[2],
                url=urljoin(HOST, a["href"]),
                chapter_id=cid,
            )
        )

    # Detect next menu page (in case of pagination — most books fit on one).
    next_url: str | None = None
    for a in soup.find_all("a", href=True):
        if "下一页" in a.get_text(strip=True) and "/menu/" in a["href"]:
            next_url = urljoin(HOST, a["href"])
            break
    return refs, next_url


def _split_chapter_title(raw: str) -> tuple[str, str, str]:
    """Return (chapter label, volume label, chapter-specific name).

    Examples:
      "第1章卷一摸鱼儿"          -> ("第1章", "卷一 摸鱼儿", "")
      "第1章 卷一 摸鱼儿"         -> ("第1章", "卷一 摸鱼儿", "")
      "第299章2022中秋番外：千星" -> ("第299章", "",          "2022中秋番外：千星")
    """
    m = CHAPTER_TITLE_PARTS_RE.match(raw)
    if not m:
        return (raw, "", "")
    chap = _clean(m.group("chap"))
    vol_num = (m.group("vol_num") or "").strip()
    vol_name = _clean(m.group("vol_name") or "")
    rest = _clean(m.group("rest") or "")
    if vol_num:
        vol = f"卷{vol_num}" + (f" {vol_name}" if vol_name else "")
        return chap, vol, ""
    return chap, "", rest


# ---------------------------------------------------------------------------
# Parse: chapter page


def parse_chapter_html(html: str) -> tuple[str, list[str]]:
    """Return (proper chapter title from page, paragraphs)."""
    soup = BeautifulSoup(html, "lxml")
    page_title = soup.title.string if soup.title else ""
    # Title format: "乱世为王-第1章 卷一 摸鱼儿 -芒果书坊"
    chap_title = ""
    if page_title:
        parts = re.split(r"[-—–]\s*", page_title)
        # Pull the segment that contains "第" 章
        for part in parts:
            if "章" in part or "番外" in part:
                chap_title = _clean(part)
                break
    if not chap_title:
        # Fall back to the in-page header if available.
        h = soup.find(class_="text-head") or soup.find("h1")
        if h:
            chap_title = _clean(h.get_text())

    content = soup.find("div", class_="read-content")
    paragraphs: list[str] = []
    if content:
        for tag in content.find_all(["script", "ins", "iframe", "style"]):
            tag.decompose()
        for p in content.find_all("p"):
            wrap = p.find("span", class_="content-wrap")
            text = _clean(wrap.get_text()) if wrap else _clean(p.get_text())
            if not text or text == "0":
                continue
            paragraphs.append(text)
    return chap_title, paragraphs


# ---------------------------------------------------------------------------
# Volume bucketing


def split_into_volumes(refs: list[ChapterRef], chapters: list[Chapter]) -> list[Volume]:
    assert len(refs) == len(chapters)
    volumes: list[Volume] = []
    cur: Volume | None = None
    for ref, ch in zip(refs, chapters):
        vol_title = ref.vol_label
        if cur is None or cur.title != vol_title:
            cur = Volume(title=vol_title)
            volumes.append(cur)
        cur.chapters.append(ch)
    # If only a single unnamed volume exists, keep it (chapters become flat).
    return volumes


# ---------------------------------------------------------------------------
# Crawl


def crawl_book(book_url: str, *, delay: float = 0.4) -> tuple[BookMeta, list[Volume]]:
    parsed = urlparse(book_url)
    m = INFO_RE.match(parsed.path)
    if not m:
        raise ValueError(f"not a /<cat>/<id>/info.html URL: {book_url}")
    cat_id, book_id = m.group(1), m.group(2)

    fetcher = Fetcher(delay=delay)

    info_url = urljoin(HOST, f"/{cat_id}/{book_id}/info.html")
    print(f"[+] fetching info {info_url}", file=sys.stderr)
    info_html = fetcher.get_html(info_url)
    meta = parse_info(info_html, cat_id, book_id)
    print(f"[+] book: {meta.title} / {meta.author}", file=sys.stderr)

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

    # Walk the menu pages.
    menu_url: str | None = urljoin(HOST, f"/{cat_id}/{book_id}/menu/1.html")
    refs: list[ChapterRef] = []
    while menu_url:
        print(f"[+] fetching menu {menu_url}", file=sys.stderr)
        page_refs, next_url = parse_menu(fetcher.get_html(menu_url), cat_id, book_id)
        refs.extend(page_refs)
        menu_url = next_url
    print(f"[+] {len(refs)} chapters discovered", file=sys.stderr)

    # Crawl each chapter.
    chapters: list[Chapter] = []
    for i, ref in enumerate(refs, 1):
        title_for_log = ref.raw_title[:40]
        print(f"[+] [{i}/{len(refs)}] {title_for_log}", file=sys.stderr)
        _chap_title_from_page, paragraphs = parse_chapter_html(fetcher.get_html(ref.url))
        # Build a clean title that doesn't repeat the volume name (the
        # volume already shows in the TOC parent).
        if ref.vol_label:
            # In a named volume, just keep "第N章".
            title = ref.chap_label or ref.raw_title
        elif ref.chap_name:
            # Extras / chapters with their own name.
            title = f"{ref.chap_label} {ref.chap_name}".strip()
        else:
            title = ref.chap_label or ref.raw_title
        chapters.append(Chapter(title=title, paragraphs=paragraphs))

    volumes = split_into_volumes(refs, chapters)
    return meta, volumes


def build_epub(meta: BookMeta, volumes: list[Volume], out_path: Path) -> None:
    write_epub(
        identifier=f"mgsf-{meta.book_id}-{int(time.time())}",
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
    return f"{HOST}/1/{arg}/info.html"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("book", help="Book URL on mangguoshufang.com or just the book id")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--delay", type=float, default=0.4)
    args = p.parse_args(argv)

    book_url = _resolve_book_url(args.book)
    meta, volumes = crawl_book(book_url, delay=args.delay)

    n_chap = sum(len(v.chapters) for v in volumes)
    n_vol = sum(1 for v in volumes if v.title)
    print(f"[+] {n_vol} named volume(s), {n_chap} chapter(s)", file=sys.stderr)

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(__file__).resolve().parents[3] / "epub"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{meta.title}.epub"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_epub(meta, volumes, out_path)
    print(f"[+] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
