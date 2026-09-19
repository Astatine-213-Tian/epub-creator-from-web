"""Detect a source and collect a book; output policy belongs to workflows."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

from src.crawler.models import CrawledBook, CrawlOptions, DownloadedEdition


@dataclass(frozen=True)
class ParserSpec:
    name: str
    domains: tuple[str, ...]
    description: str
    crawl: Callable[[str, CrawlOptions], CrawledBook | DownloadedEdition]
    downloads_edition: bool = False

    def matches(self, target: str) -> bool:
        host = urlparse(target).netloc.lower().removeprefix("www.")
        return any(
            host == domain or host.endswith(f".{domain}") for domain in self.domains
        )


def run_towasakata(target: str, options: CrawlOptions) -> CrawledBook:
    from opencc import OpenCC

    from src.crawler.providers.towasakata import parser

    delay = options.delay if options.delay is not None else 1.0
    cc = OpenCC("t2s")

    print(f"[+] fetching {target}", file=sys.stderr)
    first, pages = parser.crawl(target, delay=delay)
    print(f"[+] got {len(pages)} volume page(s)", file=sys.stderr)

    title, author = parser.guess_title_author(first.page_title, cc)
    intro_simplified = parser.render_intro_html(first.intro_html, cc)
    volumes = parser.split_into_volumes(pages)

    for volume in volumes:
        volume.title = cc.convert(volume.title)
        for chapter in volume.chapters:
            chapter.title = cc.convert(chapter.title)
            chapter.paragraphs = [cc.convert(line) for line in chapter.paragraphs]

    return CrawledBook(
        title=title or "未命名",
        author=author,
        volumes=volumes,
        source_url=target,
        intro_html=intro_simplified,
    )


def run_jrkywsy(target: str, options: CrawlOptions) -> CrawledBook:
    from src.crawler.providers.jrkywsy import parser

    meta, volumes = parser.crawl_book(target)
    return CrawledBook(
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        source_url=target,
        cover_bytes=getattr(meta, "cover_bytes", None),
        cover_mime=getattr(meta, "cover_mime", "image/jpeg"),
        intro_paragraphs=meta.intro_paragraphs,
    )


def run_mgsf(target: str, options: CrawlOptions) -> CrawledBook:
    from src.crawler.providers.mgsf import parser

    delay = options.delay if options.delay is not None else 0.4
    book_url = parser._resolve_book_url(target)
    meta, volumes = parser.crawl_book(book_url, delay=delay)

    return CrawledBook(
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        source_url=book_url,
        cover_bytes=getattr(meta, "cover_bytes", None),
        cover_mime=getattr(meta, "cover_mime", "image/jpeg"),
        intro_paragraphs=meta.intro_paragraphs,
    )


def run_xfxs(target: str, options: CrawlOptions) -> CrawledBook:
    from src.crawler.providers.xfxs import parser

    delay = options.delay if options.delay is not None else 0.4
    concurrency = options.concurrency if options.concurrency is not None else 2
    book_url = parser._resolve_book_url(target)
    meta, volumes = asyncio.run(
        parser.crawl_book(
            book_url,
            headless=options.headless,
            delay=delay,
            concurrency=concurrency,
            request_interval=options.request_interval,
        )
    )

    return CrawledBook(
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        source_url=book_url,
        cover_bytes=getattr(meta, "cover_bytes", None),
        cover_mime=getattr(meta, "cover_mime", "image/jpeg"),
        intro_paragraphs=meta.intro_paragraphs,
    )


def run_pili45(target: str, options: CrawlOptions) -> CrawledBook:
    from src.crawler.providers.pili45 import parser

    delay = options.delay if options.delay is not None else 0.4
    concurrency = options.concurrency if options.concurrency is not None else 4
    book_url = parser._resolve_book_url(target)
    meta, volumes = asyncio.run(
        parser.crawl_book(
            book_url,
            headless=options.headless,
            delay=delay,
            concurrency=concurrency,
        )
    )

    return CrawledBook(
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        source_url=book_url,
        cover_bytes=getattr(meta, "cover_bytes", None),
        cover_mime=getattr(meta, "cover_mime", "image/jpeg"),
        intro_paragraphs=meta.intro_paragraphs,
    )


def run_quanben(target: str, options: CrawlOptions) -> CrawledBook:
    from src.crawler.providers.quanben import parser

    delay = options.delay if options.delay is not None else 0.4
    concurrency = options.concurrency if options.concurrency is not None else 4
    book_url = parser._resolve_book_url(target)
    meta, volumes = parser.crawl_book(
        book_url,
        delay=delay,
        concurrency=concurrency,
    )

    return CrawledBook(
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        source_url=book_url,
        cover_bytes=getattr(meta, "cover_bytes", None),
        cover_mime=getattr(meta, "cover_mime", "image/jpeg"),
        intro_paragraphs=meta.intro_paragraphs,
    )


def run_zhenhun(target: str, options: CrawlOptions) -> CrawledBook:
    from src.crawler.providers.zhenhun import parser

    delay = options.delay if options.delay is not None else 0.4
    concurrency = options.concurrency if options.concurrency is not None else 4
    book_url = parser._resolve_book_url(target)
    meta, volumes = parser.crawl_book(
        book_url,
        delay=delay,
        concurrency=concurrency,
    )

    return CrawledBook(
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        source_url=book_url,
        cover_bytes=getattr(meta, "cover_bytes", None),
        cover_mime=getattr(meta, "cover_mime", "image/jpeg"),
        intro_paragraphs=meta.intro_paragraphs,
    )


def run_patreon(target: str, options: CrawlOptions) -> CrawledBook:
    from src.crawler.providers.patreon import parser

    delay = options.delay if options.delay is not None else 0.4
    concurrency = options.concurrency if options.concurrency is not None else 2
    book_url = parser._resolve_book_url(target)
    meta, volumes = asyncio.run(
        parser.crawl_book(
            book_url,
            headless=options.headless,
            delay=delay,
            concurrency=concurrency,
        )
    )

    return CrawledBook(
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        source_url=book_url,
        cover_bytes=getattr(meta, "cover_bytes", None),
        cover_mime=getattr(meta, "cover_mime", "image/jpeg"),
        intro_paragraphs=meta.intro_paragraphs,
    )


def run_zlibrary(target: str, options: CrawlOptions) -> DownloadedEdition:
    from src.crawler.providers.zlibrary import parser

    delay = options.delay if options.delay is not None else 0.5
    book_url = parser._resolve_book_url(target)
    meta, data = asyncio.run(
        parser.download_epub(book_url, headless=options.headless, delay=delay)
    )
    return DownloadedEdition(meta.title, meta.author, book_url, data)


PARSERS: tuple[ParserSpec, ...] = (
    ParserSpec(
        name="towasakata",
        domains=("towasakata.blog.fc2.com", "towasakata.blog.fc2blog.us"),
        description="towasakata.blog.fc2.com FC2 blog novels",
        crawl=run_towasakata,
    ),
    ParserSpec(
        name="jrkywsy",
        domains=("jrkywsy.blog.fc2.com",),
        description="jrkywsy.blog.fc2.com single-post novels",
        crawl=run_jrkywsy,
    ),
    ParserSpec(
        name="mgsf",
        domains=("mangguoshufang.com",),
        description="mangguoshufang.com novels",
        crawl=run_mgsf,
    ),
    ParserSpec(
        name="xfxs",
        domains=("xfxs1.com",),
        description="xfxs1.com novels through browser-backed zendriver",
        crawl=run_xfxs,
    ),
    ParserSpec(
        name="pili45",
        domains=("pili45.com", "pilishuwu.com"),
        description="pili45.com / pilishuwu.com novels through browser-backed zendriver",
        crawl=run_pili45,
    ),
    ParserSpec(
        name="quanben",
        domains=("quanben.io",),
        description="quanben.io novels",
        crawl=run_quanben,
    ),
    ParserSpec(
        name="zhenhun",
        domains=("zhenhunxiaoshuo.com",),
        description="zhenhunxiaoshuo.com WordPress category novels",
        crawl=run_zhenhun,
    ),
    ParserSpec(
        name="zlibrary",
        domains=("1lib.sk", "z-lib.sk", "z-library.sk"),
        description="Z-Library browser-backed EPUB editions",
        crawl=run_zlibrary,
        downloads_edition=True,
    ),
    ParserSpec(
        name="patreon",
        domains=("patreon.com",),
        description="patreon.com collections through authenticated browser profile",
        crawl=run_patreon,
    ),
)


def find_parser(target: str, parser_name: str | None = None) -> ParserSpec:
    if parser_name:
        for parser in PARSERS:
            if parser.name == parser_name:
                return parser
        choices = ", ".join(parser.name for parser in PARSERS)
        raise ValueError(f"unknown parser {parser_name!r}; choose one of: {choices}")

    for parser in PARSERS:
        if parser.matches(target):
            return parser

    domains = ", ".join(domain for parser in PARSERS for domain in parser.domains)
    raise ValueError(
        f"could not detect parser for {target!r}; supported domains: {domains}. "
        "Use --parser when passing a site-specific book id instead of a URL."
    )
