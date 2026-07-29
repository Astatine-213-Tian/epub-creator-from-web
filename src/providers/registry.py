from __future__ import annotations

import asyncio
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from src.core.models import Volume
from src.core.output import default_output_path as core_default_output_path
from src.core.output import dataset_txt_output_path
from src.core.output import resolve_output_path
from src.core.output import resolve_txt_output_path
from src.core.text_writer import write_txt

ParserRunner = Callable[[str, "ParserOptions"], Path]
BuildEpub = Callable[[Path], None]


@dataclass(frozen=True)
class ParserOptions:
    output: Path | None = None
    txt_output: Path | None = None
    output_formats: tuple[str, ...] = ("epub",)
    dataset_root: Path | None = None
    prevent_overwrite: bool = True
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


def default_output_path(title: str, author: str = "") -> Path:
    return core_default_output_path(title, author)


def resolve_output(options: ParserOptions, title: str, author: str = "") -> Path:
    return resolve_output_path(options.output, title, author)


def requested_formats(options: ParserOptions) -> tuple[str, ...]:
    formats: list[str] = []
    for output_format in options.output_formats or ("epub",):
        normalized = output_format.strip().lower()
        if normalized == "both":
            normalized_formats = ("epub", "txt")
        else:
            normalized_formats = (normalized,)
        for item in normalized_formats:
            if item not in {"epub", "txt"}:
                raise ValueError(f"unsupported output format: {output_format}")
            if item not in formats:
                formats.append(item)
    return tuple(formats or ["epub"])


def resolve_requested_txt_output(
    options: ParserOptions,
    *,
    title: str,
    author: str,
    intro_paragraphs: list[str] | None = None,
    intro_html: str = "",
) -> Path:
    _ = (intro_paragraphs, intro_html)
    if options.dataset_root:
        return dataset_txt_output_path(
            options.dataset_root,
            title=title,
            author=author,
            time_area="",
            genre="",
        )
    return resolve_txt_output_path(options.txt_output, title, author)


def emit_requested_outputs(
    *,
    options: ParserOptions,
    title: str,
    author: str,
    volumes: list[Volume],
    build_epub: BuildEpub,
    intro_paragraphs: list[str] | None = None,
    intro_html: str = "",
) -> Path:
    formats = requested_formats(options)
    epub_path = resolve_output(options, title, author) if "epub" in formats else None
    txt_path = (
        resolve_requested_txt_output(
            options,
            title=title,
            author=author,
            intro_paragraphs=intro_paragraphs,
            intro_html=intro_html,
        )
        if "txt" in formats
        else None
    )

    if epub_path and (not options.prevent_overwrite or not epub_path.exists()):
        build_epub(epub_path)
    if txt_path and (not options.prevent_overwrite or not txt_path.exists()):
        write_txt(
            title=title,
            author=author,
            volumes=volumes,
            out_path=txt_path,
            intro_paragraphs=intro_paragraphs,
            intro_html=intro_html,
        )

    if epub_path:
        return epub_path
    if txt_path:
        return txt_path
    raise ValueError("no output formats requested")


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

    return emit_requested_outputs(
        options=options,
        title=title or "未命名",
        author=author,
        volumes=volumes,
        build_epub=lambda out_path: parser.build_epub(
            title or "未命名",
            author,
            volumes,
            intro_simplified,
            out_path,
        ),
        intro_html=intro_simplified,
    )


def run_jrkywsy(target: str, options: ParserOptions) -> Path:
    from src.providers.jrkywsy import parser

    meta, volumes = parser.crawl_book(target)
    return emit_requested_outputs(
        options=options,
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        build_epub=lambda out_path: parser.build_epub(meta, volumes, out_path),
        intro_paragraphs=meta.intro_paragraphs,
    )


def run_mgsf(target: str, options: ParserOptions) -> Path:
    from src.providers.mgsf import parser

    delay = options.delay if options.delay is not None else 0.4
    book_url = parser._resolve_book_url(target)
    meta, volumes = parser.crawl_book(book_url, delay=delay)

    return emit_requested_outputs(
        options=options,
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        build_epub=lambda out_path: parser.build_epub(meta, volumes, out_path),
        intro_paragraphs=meta.intro_paragraphs,
    )


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

    return emit_requested_outputs(
        options=options,
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        build_epub=lambda out_path: parser.build_epub(meta, volumes, out_path),
        intro_paragraphs=meta.intro_paragraphs,
    )


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

    return emit_requested_outputs(
        options=options,
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        build_epub=lambda out_path: parser.build_epub(meta, volumes, out_path),
        intro_paragraphs=meta.intro_paragraphs,
    )


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

    return emit_requested_outputs(
        options=options,
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        build_epub=lambda out_path: parser.build_epub(meta, volumes, out_path),
        intro_paragraphs=meta.intro_paragraphs,
    )


def run_zhenhun(target: str, options: ParserOptions) -> Path:
    from src.providers.zhenhun import parser

    delay = options.delay if options.delay is not None else 0.4
    concurrency = options.concurrency if options.concurrency is not None else 4
    book_url = parser._resolve_book_url(target)
    meta, volumes = parser.crawl_book(
        book_url,
        delay=delay,
        concurrency=concurrency,
    )

    return emit_requested_outputs(
        options=options,
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        build_epub=lambda out_path: parser.build_epub(meta, volumes, out_path),
        intro_paragraphs=meta.intro_paragraphs,
    )


def run_zlibrary(target: str, options: ParserOptions) -> Path:
    from src.cli.dataset import extract_epub_text
    from src.core.epub_normalizer import normalize_new_epub
    from src.providers.zlibrary import parser

    delay = options.delay if options.delay is not None else 0.5
    meta, epub_bytes = asyncio.run(
        parser.download_epub(
            parser._resolve_book_url(target),
            headless=options.headless,
            delay=delay,
        )
    )
    formats = requested_formats(options)
    epub_path = resolve_output(options, meta.title, meta.author) if "epub" in formats else None
    txt_path = (
        resolve_requested_txt_output(options, title=meta.title, author=meta.author)
        if "txt" in formats
        else None
    )

    with tempfile.TemporaryDirectory(prefix="booklib-zlibrary-output-") as tmp:
        source_path = Path(tmp) / f"{meta.book_id}.epub"
        source_path.write_bytes(epub_bytes)
        extraction_path = source_path
        if epub_path and (not options.prevent_overwrite or not epub_path.exists()):
            epub_path.parent.mkdir(parents=True, exist_ok=True)
            staging_path = epub_path.with_suffix(f"{epub_path.suffix}.tmp")
            staging_path.write_bytes(epub_bytes)
            staging_path.replace(epub_path)
            normalize_new_epub(epub_path)
            extraction_path = epub_path
        elif epub_path:
            extraction_path = epub_path

        if txt_path and (not options.prevent_overwrite or not txt_path.exists()):
            _title, _author, text = extract_epub_text(extraction_path)
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(
                f"{meta.title}\n作者：{meta.author}\n\n{text.strip()}\n",
                encoding="utf-8",
            )

    if epub_path:
        return epub_path
    if txt_path:
        return txt_path
    raise ValueError("no output formats requested")


def run_patreon(target: str, options: ParserOptions) -> Path:
    from src.providers.patreon import parser

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

    return emit_requested_outputs(
        options=options,
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        build_epub=lambda out_path: parser.build_epub(meta, volumes, out_path),
        intro_paragraphs=meta.intro_paragraphs,
    )


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
        domains=("pili45.com", "pilishuwu.com"),
        description="pili45.com / pilishuwu.com novels through browser-backed zendriver",
        run=run_pili45,
    ),
    ParserSpec(
        name="quanben",
        domains=("quanben.io",),
        description="quanben.io novels",
        run=run_quanben,
    ),
    ParserSpec(
        name="zhenhun",
        domains=("zhenhunxiaoshuo.com",),
        description="zhenhunxiaoshuo.com WordPress category novels",
        run=run_zhenhun,
    ),
    ParserSpec(
        name="zlibrary",
        domains=("1lib.sk", "z-lib.sk", "z-library.sk"),
        description="Z-Library browser-backed EPUB editions",
        run=run_zlibrary,
    ),
    ParserSpec(
        name="patreon",
        domains=("patreon.com",),
        description="patreon.com collections through authenticated browser profile",
        run=run_patreon,
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
