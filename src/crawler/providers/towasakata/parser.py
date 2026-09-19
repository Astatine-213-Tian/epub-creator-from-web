#!/usr/bin/env python3
"""Collect and parse towasakata source material for the shared ingest workflow."""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from opencc import OpenCC

from src.content.html import escape_text
from src.content.models import Chapter, Volume

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
BLOG_HOST = "towasakata.blog.fc2.com"
ENTRY_RE = re.compile(r"blog-entry-(\d+)\.html")
NO_RE = re.compile(r"[?&]no=(\d+)")

# Chapter / volume markers inside the post body.
VOL_RE = re.compile(r"^\s*第\s*([一二三四五六七八九十百零兩两\d]+)\s*卷\s*(.*)$")
CHAP_RE_CN = re.compile(r"^\s*第\s*([一二三四五六七八九十百零兩两\d]+)\s*章\s*(.*)$")
CHAP_RE_EN = re.compile(r"^\s*Chapter\s*(\d+)\s*[:．\.]?\s*(.*)$", re.IGNORECASE)
# Markers we want to drop entirely (end-of-volume tags, 上/中/下 part labels).
SKIP_RE = re.compile(
    r"^\s*("
    r"[（(]?[上中下][)）]?"  # bare 上/中/下
    r"|第[一二三四五六七八九十百零兩两\d]+卷[^\n]{0,20}[完終终]"
    r"|卷[一二三四五六七八九十\d]*[結结][束尾]"
    r"|本卷[終终完]"
    r")\s*$"
)


# ---------------------------------------------------------------------------
# HTTP helpers


def normalize_url(u: str) -> str:
    """Rewrite all known mirror domains to the canonical fc2.com host and
    convert ``?no=N`` to ``blog-entry-N.html``."""
    parsed = urlparse(u)
    host = parsed.netloc.lower()
    # towasakata.blog.fc2blog.us -> towasakata.blog.fc2.com
    if host.endswith("fc2blog.us"):
        host = BLOG_HOST
    if not host:
        host = BLOG_HOST
    path = parsed.path or "/"
    qs = parse_qs(parsed.query)
    if "no" in qs:
        return f"http://{host}/blog-entry-{qs['no'][0]}.html"
    return f"http://{host}{path}"


def entry_id(u: str) -> str | None:
    """Return the numeric blog-entry id for canonicalisation/dedup."""
    m = ENTRY_RE.search(u) or NO_RE.search(u)
    return m.group(1) if m else None


def fetch(url: str, session: requests.Session, retries: int = 3) -> str:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            r = session.get(url, timeout=30, headers={"User-Agent": UA})
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


# ---------------------------------------------------------------------------
# Page parsing


@dataclass
class VolumePage:
    url: str
    entry_id: str
    page_title: str  # raw post title (e.g. "...第一卷 貓將軍...")
    intro_html: str  # html of the introduction (only for first page)
    intro_links: list[str]  # canonicalised volume urls in document order
    body_html: str  # html after the <a name="more"> cut


def _extract_main(html: str) -> Tag:
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one("div.main")
    if main is None:
        raise RuntimeError("could not locate <div class='main'> in page")
    return main


def _collect_intro_and_links(main: Tag) -> tuple[str, list[str], Tag | None]:
    """Walk the main div top-to-bottom collecting intro html and volume links.

    Stops at the ``<a name='more'>`` anchor — content after that is the
    actual chapter body. We separate the prose ("intro_html") from the
    list of volume links sitting underneath it.
    """
    more_anchor: Tag | None = main.find("a", attrs={"name": "more"})

    intro_parts: list[str] = []
    link_urls: list[str] = []
    seen_a_link = False  # once we hit a volume <a>, prose is over

    for child in list(main.children):
        if more_anchor is not None and child is more_anchor:
            break
        if isinstance(child, Tag) and child.name == "a" and child.has_attr("href"):
            href = child["href"]
            if ENTRY_RE.search(href) or NO_RE.search(href):
                link_urls.append(normalize_url(href))
                seen_a_link = True
                continue
        # treat <br> after the link block as part of link block (drop)
        if seen_a_link and isinstance(child, Tag) and child.name == "br":
            continue
        if (
            seen_a_link
            and isinstance(child, NavigableString)
            and not str(child).strip()
        ):
            continue
        if seen_a_link:
            # stray text after links — usually empty; ignore
            continue
        intro_parts.append(str(child))

    intro_html = "".join(intro_parts).strip()
    # Dedup links preserving order.
    seen: set[str] = set()
    uniq_links: list[str] = []
    for u in link_urls:
        eid = entry_id(u) or u
        if eid in seen:
            continue
        seen.add(eid)
        uniq_links.append(u)
    return intro_html, uniq_links, more_anchor


def _collect_body(main: Tag, more_anchor: Tag | None) -> str:
    if more_anchor is None:
        return ""
    parts: list[str] = []
    for sib in more_anchor.next_siblings:
        parts.append(str(sib))
    return "".join(parts).strip()


def parse_page(url: str, html: str) -> VolumePage:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.select_one("h2.title a") or soup.select_one("h2.title")
    page_title = title_tag.get_text(strip=True) if title_tag else url

    main = _extract_main(html)
    intro_html, links, more_anchor = _collect_intro_and_links(main)
    body_html = _collect_body(main, more_anchor)

    eid = entry_id(url) or url
    return VolumePage(
        url=normalize_url(url),
        entry_id=eid,
        page_title=page_title,
        intro_html=intro_html,
        intro_links=links,
        body_html=body_html,
    )


# ---------------------------------------------------------------------------
# Chapter splitting


def _html_to_lines(body_html: str) -> list[str]:
    """Split body html into clean text lines.

    The post bodies are essentially "<text><br><text><br>..." with the
    occasional <p> wrapper, so we replace <br> with newlines and strip.
    """
    soup = BeautifulSoup(body_html, "lxml")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text()
    # Collapse non-breaking / fullwidth whitespace at line edges, keep inner.
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip("　 \t ")  # fullwidth space, nbsp, tab, space
        if not line:
            continue
        out.append(line)
    return out


def _is_skip(line: str) -> bool:
    return bool(SKIP_RE.match(line))


def _vol_match(line: str) -> str | None:
    if len(line) > 40:
        return None
    m = VOL_RE.match(line)
    if not m:
        return None
    # Reject end-of-volume markers like "第一卷 ·萬里長城·完——".
    if re.search(r"[完終终]", line):
        return None
    return line.strip()


def _chap_match(line: str) -> str | None:
    if len(line) > 60:
        return None
    if CHAP_RE_CN.match(line) or CHAP_RE_EN.match(line):
        return line.strip()
    return None


_PAGE_VOL_RE = re.compile(r"第\s*[一二三四五六七八九十百零兩两\d]+\s*卷[^()（）by]*")


def _page_volume_name(page_title: str) -> str | None:
    """Pull a `第X卷 名字` fragment out of a page heading, if present."""
    m = _PAGE_VOL_RE.search(page_title)
    if not m:
        return None
    name = m.group(0).strip(" 　")
    # Strip trailing series/category tags like "(靈異向盜墓文)".
    name = re.sub(r"[（(][^()（）]*[)）]\s*$", "", name).strip()
    return name or None


def split_into_volumes(pages: list[VolumePage]) -> list[Volume]:
    """Walk every page's body and produce a hierarchical Volume/Chapter list.

    Volume detection looks at two places:
      1. The page heading (``第X卷 名字`` in the post title) — used when each
         volume lives on its own entry page (e.g. 靈魂深處鬧革命).
      2. Inline ``第X卷`` markers inside the body text (e.g. 奪夢 出書版 上).

    Lines that are pure 上/中/下 part labels or end-of-volume tags are
    discarded. If no volume markers ever appear we synthesise a single
    unnamed volume so chapters become first-level TOC entries.
    """
    volumes: list[Volume] = []
    current_vol: Volume | None = None
    current_chap: Chapter | None = None

    def open_volume(title: str) -> None:
        nonlocal current_vol, current_chap
        current_vol = Volume(title=title)
        volumes.append(current_vol)
        current_chap = None

    def ensure_vol() -> Volume:
        nonlocal current_vol
        if current_vol is None:
            current_vol = Volume(title="")
            volumes.append(current_vol)
        return current_vol

    def ensure_chap() -> Chapter:
        nonlocal current_chap
        if current_chap is None:
            current_chap = Chapter(title="正文")
            ensure_vol().chapters.append(current_chap)
        return current_chap

    for page in pages:
        page_vol = _page_volume_name(page.page_title)
        if page_vol and (current_vol is None or current_vol.title != page_vol):
            open_volume(page_vol)

        for line in _html_to_lines(page.body_html):
            if _is_skip(line):
                continue
            vol = _vol_match(line)
            if vol:
                open_volume(vol)
                continue
            chap = _chap_match(line)
            if chap:
                current_chap = Chapter(title=chap)
                ensure_vol().chapters.append(current_chap)
                continue
            ensure_chap().paragraphs.append(line)

    # Drop empty volumes/chapters that only had skip lines.
    cleaned: list[Volume] = []
    for v in volumes:
        v.chapters = [c for c in v.chapters if c.paragraphs]
        if v.chapters:
            cleaned.append(v)
    return cleaned


# ---------------------------------------------------------------------------
# EPUB intro rendering


def render_intro_html(intro_html: str, cc: OpenCC) -> str:
    """Convert the original intro HTML to simplified Chinese plain paragraphs."""
    soup = BeautifulSoup(intro_html, "lxml")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text()
    paragraphs: list[str] = []
    for chunk in re.split(r"\n{2,}", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk = "<br/>".join(escape_text(part) for part in chunk.splitlines())
        paragraphs.append(f"<p>{cc.convert(chunk)}</p>")
    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Title / author guessing


def guess_title_author(first_page_title: str, cc: OpenCC) -> tuple[str, str]:
    """Try to extract a clean book title + author from the post heading.

    Heading example: ``靈魂深處鬧革命 第一卷 貓將軍 by 非天夜翔(靈異向盜墓文)``
    """
    raw = first_page_title.strip()
    # Strip trailing parenthesised tags.
    raw = re.sub(r"[（(][^()（）]*[)）]\s*$", "", raw).strip()
    author = ""
    m = re.search(r"\bby\s+([^()（）]+?)\s*$", raw, flags=re.I)
    if m:
        author = m.group(1).strip()
        raw = raw[: m.start()].strip()
    # Drop "第X卷 ..." / "(上/中/下)" / "出書版" markers.
    raw = re.sub(r"第[一二三四五六七八九十百零\d]+卷[^()（）]*$", "", raw).strip()
    raw = re.sub(r"-\s*出[書书]版.*$", "", raw).strip()
    raw = re.sub(r"[（(]?[上中下][)）]?\s*$", "", raw).strip()
    title = raw if raw else first_page_title
    return cc.convert(title), cc.convert(author)


# ---------------------------------------------------------------------------
# Main


def crawl(start_url: str, *, delay: float = 1.0) -> tuple[VolumePage, list[VolumePage]]:
    session = requests.Session()
    start = normalize_url(start_url)
    first_html = fetch(start, session)
    first = parse_page(start, first_html)

    # Determine the full volume URL list. The intro section usually lists
    # every volume page (including itself); fall back to the start url
    # alone if no links were found.
    urls = first.intro_links or [first.url]
    pages: list[VolumePage] = []
    seen_ids: set[str] = set()
    for u in urls:
        eid = entry_id(u) or u
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        if eid == first.entry_id:
            pages.append(first)
            continue
        time.sleep(delay)
        html = fetch(u, session)
        pages.append(parse_page(u, html))
    return first, pages
