#!/usr/bin/env python3
"""Download an existing EPUB from a Z-Library book detail page."""

from __future__ import annotations

import asyncio
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import zendriver as zd
from bs4 import BeautifulSoup

from src.crawler.fetch.browser import resolve_browser_executable, wait_for_page_ready

HOST = "https://zh.1lib.sk"
BOOK_RE = re.compile(r"^/book/(?P<book_id>[^/]+)(?:/[^/]+)?/?$")


@dataclass(frozen=True)
class BookMeta:
    book_id: str
    title: str
    author: str
    detail_url: str
    download_url: str
    extension: str
    file_size: str
    description: str = ""
    publisher: str = ""
    year: str = ""
    language: str = ""
    isbn: str = ""


def _clean_text(value: str) -> str:
    return re.sub(r"[\s\xa0　]+", " ", value).strip()


def _attr(node, name: str) -> str:
    value = node.get(name, "")
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _property_value(soup: BeautifulSoup, class_name: str) -> str:
    node = soup.select_one(f".{class_name} .property_value")
    return _clean_text(node.get_text(" ", strip=True)) if node else ""


def parse_detail_page(html: str, detail_url: str) -> BookMeta:
    soup = BeautifulSoup(html, "lxml")
    parsed = urlparse(detail_url)
    match = BOOK_RE.match(parsed.path)
    if not match:
        raise ValueError(f"not a Z-Library book URL: {detail_url}")

    title_node = soup.select_one("h1.book-title")
    author_node = soup.select_one(".authors a, .authors")
    download = soup.select_one("a.addDownloadedBook[href^='/dl/']")
    if not title_node or not author_node or not download:
        raise ValueError(f"Z-Library book metadata did not load: {detail_url}")

    title = _clean_text(title_node.get_text(" ", strip=True))
    author = _clean_text(author_node.get_text(" ", strip=True))
    download_text = _clean_text(download.get_text(" ", strip=True))
    extension_match = re.match(r"([A-Za-z0-9]+)", download_text)
    extension = extension_match.group(1).lower() if extension_match else ""
    file_size = (
        download_text[len(extension_match.group(0)) :].lstrip(" ,")
        if extension_match
        else ""
    )
    description_node = soup.select_one("meta[name='description']")

    return BookMeta(
        book_id=match.group("book_id"),
        title=title,
        author=author,
        detail_url=detail_url,
        download_url=urljoin(HOST, _attr(download, "href")),
        extension=extension,
        file_size=file_size,
        description=_clean_text(_attr(description_node, "content"))
        if description_node
        else "",
        publisher=_property_value(soup, "property_publisher"),
        year=_property_value(soup, "property_year"),
        language=_property_value(soup, "property_language"),
        isbn=_property_value(soup, "property_isbn"),
    )


async def _wait_for_download(page, directory: Path, *, timeout: float = 90.0) -> Path:
    deadline = asyncio.get_running_loop().time() + timeout
    next_page_check = 0.0
    while asyncio.get_running_loop().time() < deadline:
        partials = list(directory.glob("*.crdownload"))
        files = [
            path
            for path in directory.iterdir()
            if path.is_file() and not path.name.endswith(".crdownload")
        ]
        for path in files:
            if path.stat().st_size > 0 and zipfile.is_zipfile(path):
                return path
        if not partials and files:
            names = ", ".join(path.name for path in files)
            raise RuntimeError(f"download completed but was not an EPUB ZIP: {names}")
        now = asyncio.get_running_loop().time()
        if now >= next_page_check:
            next_page_check = now + 1.0
            try:
                page_text = await page.evaluate("document.body?.innerText || ''")
            except Exception:
                page_text = ""
            if "每日限额已用完" in page_text or "daily limit" in page_text.lower():
                raise RuntimeError(
                    "Z-Library daily download limit is exhausted; retry later or use an authenticated browser"
                )
        await asyncio.sleep(0.25)
    raise TimeoutError(f"timed out waiting for EPUB download in {directory}")


async def download_epub(
    book_url: str,
    *,
    headless: bool = False,
    delay: float = 0.5,
) -> tuple[BookMeta, bytes]:
    detail_url = _resolve_book_url(book_url)
    browser = await zd.start(
        zd.Config(
            headless=headless,
            browser_executable_path=resolve_browser_executable(),
            sandbox=False,
            browser_connection_timeout=1.0,
            browser_connection_max_tries=30,
        )
    )
    try:
        page = await browser.get(detail_url)
        await wait_for_page_ready(
            page, ready_selector="h1.book-title", settle_delay=delay
        )
        meta = parse_detail_page(await page.get_content(), detail_url)
        if meta.extension != "epub":
            raise ValueError(
                f"selected Z-Library edition is {meta.extension or 'unknown format'}, not EPUB: {detail_url}"
            )

        with tempfile.TemporaryDirectory(prefix="booklib-zlibrary-") as tmp:
            download_dir = Path(tmp)
            await page.set_download_path(download_dir)
            clicked = await page.evaluate(
                """
                (() => {
                  const link = document.querySelector("a.addDownloadedBook[href^='/dl/']");
                  if (!link) return false;
                  link.click();
                  return true;
                })()
                """
            )
            if not clicked:
                raise RuntimeError(f"download link not found on {detail_url}")
            downloaded = await _wait_for_download(page, download_dir)
            data = downloaded.read_bytes()

        with tempfile.NamedTemporaryFile(suffix=".epub") as probe:
            probe.write(data)
            probe.flush()
            with zipfile.ZipFile(probe.name) as archive:
                bad_member = archive.testzip()
                if bad_member:
                    raise RuntimeError(
                        f"downloaded EPUB has a corrupt ZIP member: {bad_member}"
                    )
                if "mimetype" not in archive.namelist():
                    raise RuntimeError(
                        "downloaded ZIP is missing the EPUB mimetype member"
                    )
        return meta, data
    finally:
        await browser.stop()


def _resolve_book_url(arg: str) -> str:
    if arg.startswith("http"):
        parsed = urlparse(arg)
        match = BOOK_RE.match(parsed.path)
        if not match:
            raise ValueError(f"not a Z-Library book URL: {arg}")
        return urljoin(HOST, parsed.path)
    book_id = arg.strip("/")
    if not re.fullmatch(r"[A-Za-z0-9]+", book_id):
        raise ValueError(f"not a Z-Library book id: {arg}")
    return f"{HOST}/book/{book_id}"
