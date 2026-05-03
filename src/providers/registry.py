from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse


ParserRunner = Callable[[str, "ParserOptions"], Path]


@dataclass(frozen=True)
class ParserOptions:
    output: Path | None = None
    delay: float | None = None
    headless: bool = False
    concurrency: int | None = None


@dataclass(frozen=True)
class ParserSpec:
    name: str
    domains: tuple[str, ...]
    description: str
    run: ParserRunner

    def matches(self, target: str) -> bool:
        host = urlparse(target).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return any(host == domain or host.endswith(f".{domain}") for domain in self.domains)


def default_output_path(title: str) -> Path:
    out_dir = Path(__file__).resolve().parent.parent / "epub"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{title or 'book'}.epub"


def resolve_output(options: ParserOptions, title: str) -> Path:
    out_path = options.output or default_output_path(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def run_towasakata(target: str, options: ParserOptions) -> Path:
    from opencc import OpenCC

    from src.providers.towasakata import parser

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

    out_path = resolve_output(options, title or "book")
    parser.build_epub(title or "未命名", author, volumes, intro_simplified, out_path)
    return out_path


def run_jrkywsy(target: str, options: ParserOptions) -> Path:
    from src.providers.jrkywsy import parser

    meta, volumes = parser.crawl_book(target)
    out_path = resolve_output(options, meta.title)
    parser.build_epub(meta, volumes, out_path)
    return out_path


def run_mgsf(target: str, options: ParserOptions) -> Path:
    from src.providers.mgsf import parser

    delay = options.delay if options.delay is not None else 0.4
    book_url = parser._resolve_book_url(target)
    meta, volumes = parser.crawl_book(book_url, delay=delay)

    out_path = resolve_output(options, meta.title)
    parser.build_epub(meta, volumes, out_path)
    return out_path


def run_xfxs(target: str, options: ParserOptions) -> Path:
    from src.providers.xfxs import parser

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

    out_path = resolve_output(options, meta.title)
    parser.build_epub(meta, volumes, out_path)
    return out_path


def run_pili45(target: str, options: ParserOptions) -> Path:
    from src.providers.pili45 import parser

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

    out_path = resolve_output(options, meta.title)
    parser.build_epub(meta, volumes, out_path)
    return out_path


def run_quanben(target: str, options: ParserOptions) -> Path:
    from src.providers.quanben import parser

    delay = options.delay if options.delay is not None else 0.4
    concurrency = options.concurrency if options.concurrency is not None else 4
    book_url = parser._resolve_book_url(target)
    meta, volumes = parser.crawl_book(
        book_url,
        delay=delay,
        concurrency=concurrency,
    )

    out_path = resolve_output(options, meta.title)
    parser.build_epub(meta, volumes, out_path)
    return out_path


PARSERS: tuple[ParserSpec, ...] = (
    ParserSpec(
        name="towasakata",
        domains=("towasakata.blog.fc2.com", "towasakata.blog.fc2blog.us"),
        description="towasakata.blog.fc2.com FC2 blog novels",
        run=run_towasakata,
    ),
    ParserSpec(
        name="jrkywsy",
        domains=("jrkywsy.blog.fc2.com",),
        description="jrkywsy.blog.fc2.com single-post novels",
        run=run_jrkywsy,
    ),
    ParserSpec(
        name="mgsf",
        domains=("mangguoshufang.com",),
        description="mangguoshufang.com novels",
        run=run_mgsf,
    ),
    ParserSpec(
        name="xfxs",
        domains=("xfxs1.com",),
        description="xfxs1.com novels through browser-backed zendriver",
        run=run_xfxs,
    ),
    ParserSpec(
        name="pili45",
        domains=("pili45.com",),
        description="pili45.com novels through browser-backed zendriver",
        run=run_pili45,
    ),
    ParserSpec(
        name="quanben",
        domains=("quanben.io",),
        description="quanben.io novels",
        run=run_quanben,
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
