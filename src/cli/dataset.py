#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.core.output import dataset_txt_output_path
from src.core.text_writer import BLANK_RE, clean_text_line
from src.metadata.classifier import classify_many_with_codex, classify_with_codex, metadata_to_cache
from src.metadata.jjwxc import GENRES, TIME_AREAS, JjwxcTypeMetadata, classify_from_text, fetch_jjwxc_type


CONTAINER_NS = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
OPF_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}


@dataclass
class ExportEntry:
    source_epub: str
    txt_path: str
    title: str
    author: str
    time_area: str
    genre: str
    metadata_source: str
    article_type: str
    jjwxc_url: str
    char_count: int
    status: str
    error: str = ""


AUTHOR_ALIASES = {
    "顾雪柔": "非天夜翔",
}
REVERSE_AUTHOR_ALIASES: dict[str, tuple[str, ...]] = {}
for alias, canonical in AUTHOR_ALIASES.items():
    REVERSE_AUTHOR_ALIASES.setdefault(canonical, tuple())
    REVERSE_AUTHOR_ALIASES[canonical] = (*REVERSE_AUTHOR_ALIASES[canonical], alias)


def _metadata_key(title: str, author: str) -> str:
    return f"local::{author}::{title}"


def _canonical_author(author: str) -> str:
    return AUTHOR_ALIASES.get(author.strip(), author.strip())


def _metadata_cache_keys(title: str, author: str) -> list[str]:
    authors = [author]
    authors.extend(REVERSE_AUTHOR_ALIASES.get(author, ()))
    keys: list[str] = []
    for item in authors:
        key = _metadata_key(title, item)
        if key not in keys:
            keys.append(key)
    return keys


def _normalize_text_author_header(text: str, author: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines[:8]):
        if line.startswith("作者："):
            lines[index] = f"作者：{author}"
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    return text


def _collapse_adjacent_duplicate_lines(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    last_nonblank = ""
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == last_nonblank:
            continue
        out.append(line)
        if stripped:
            last_nonblank = stripped
        elif out and out[-1] == "":
            last_nonblank = ""
    return "\n".join(out)



def _decode_html(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _html_to_lines(html: str) -> list[str]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["head", "script", "style", "nav"]):
        tag.decompose()
    content = soup.body or soup
    for tag in content.find_all(["br", "p", "div", "section", "h1", "h2", "h3", "li"]):
        tag.append("\n")
    lines = [clean_text_line(line) for line in content.get_text("\n").splitlines()]
    deduped: list[str] = []
    for line in lines:
        if line and (not deduped or deduped[-1] != line):
            deduped.append(line)
    return deduped


def _container_rootfile(zf: zipfile.ZipFile) -> str:
    root = ET.fromstring(zf.read("META-INF/container.xml"))
    rootfile = root.find(".//container:rootfile", CONTAINER_NS)
    if rootfile is None:
        raise ValueError("missing EPUB rootfile")
    path = rootfile.attrib.get("full-path", "")
    if not path:
        raise ValueError("empty EPUB rootfile path")
    return path


def _metadata_text(root: ET.Element, tag: str) -> str:
    node = root.find(f".//dc:{tag}", OPF_NS)
    return clean_text_line(node.text or "") if node is not None else ""


def _is_content_spine_item(
    *,
    item_id: str,
    href: str,
    media_type: str,
    properties: str,
) -> bool:
    lower_id = item_id.lower()
    lower_href = href.lower()
    basename = posixpath.basename(lower_href)
    property_tokens = set(properties.lower().split())
    if "html" not in media_type and not lower_href.endswith((".xhtml", ".html", ".htm")):
        return False
    if "nav" in property_tokens:
        return False
    if lower_id in {"nav", "toc", "ncx", "cover", "cover-page", "titlepage", "title-page"}:
        return False
    if basename in {
        "nav.xhtml",
        "nav.html",
        "toc.xhtml",
        "toc.html",
        "cover.xhtml",
        "cover.html",
        "titlepage.xhtml",
        "title-page.xhtml",
    }:
        return False
    return True


def extract_epub_text(epub_path: Path) -> tuple[str, str, str]:
    with zipfile.ZipFile(epub_path) as zf:
        opf_path = _container_rootfile(zf)
        opf_root = ET.fromstring(zf.read(opf_path))
        opf_dir = posixpath.dirname(opf_path)
        title = _metadata_text(opf_root, "title") or epub_path.stem
        author = _metadata_text(opf_root, "creator") or epub_path.parent.name

        manifest: dict[str, tuple[str, str, str]] = {}
        for item in opf_root.findall(".//opf:manifest/opf:item", OPF_NS):
            item_id = item.attrib.get("id", "")
            href = item.attrib.get("href", "")
            media_type = item.attrib.get("media-type", "")
            properties = item.attrib.get("properties", "")
            if item_id and href:
                manifest[item_id] = (href, media_type, properties)

        text_sections: list[str] = [title]
        if author:
            text_sections.append(f"作者：{author}")

        for itemref in opf_root.findall(".//opf:spine/opf:itemref", OPF_NS):
            idref = itemref.attrib.get("idref", "")
            href_media = manifest.get(idref)
            if not href_media:
                continue
            href, media_type, properties = href_media
            if not _is_content_spine_item(
                item_id=idref,
                href=href,
                media_type=media_type,
                properties=properties,
            ):
                continue
            member_path = posixpath.normpath(posixpath.join(opf_dir, href))
            if member_path not in zf.namelist():
                continue
            lines = _html_to_lines(_decode_html(zf.read(member_path)))
            if lines:
                text_sections.append("\n".join(lines))

    text = "\n\n".join(section.strip() for section in text_sections if section.strip())
    text = _collapse_adjacent_duplicate_lines(BLANK_RE.sub("\n\n", text).strip())
    return title, author, text + "\n"


def load_jjwxc_manifest(manifest_path: Path | None, top50_path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if not manifest_path or not manifest_path.exists():
        return {}
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranking_types: dict[int, str] = {}
    if top50_path and top50_path.exists():
        for item in json.loads(top50_path.read_text(encoding="utf-8")):
            ranking_types[int(item.get("rank", 0))] = item.get("genre", "")
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        title = row.get("jjwxc_title", "")
        author = _canonical_author(row.get("jjwxc_author", ""))
        if not title or not author:
            continue
        by_key[(title, author)] = {
            "jjwxc_url": row.get("jjwxc_url", ""),
            "article_type": ranking_types.get(int(row.get("rank", 0)), ""),
        }
    return by_key


def load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prefill_codex_cache(
    *,
    epub_paths: list[Path],
    jjwxc_index: dict[tuple[str, str], dict[str, str]],
    cache: dict[str, dict[str, str]],
    exclude_titles: set[str],
    exclude_metadata_keys: set[str],
    batch_size: int = 20,
) -> None:
    pending: list[dict[str, str]] = []
    for epub_path in epub_paths:
        try:
            title, author, text = extract_epub_text(epub_path)
        except Exception:
            continue
        author = _canonical_author(author)
        text = _normalize_text_author_header(text, author)
        if title in exclude_titles or _metadata_key(title, author) in exclude_metadata_keys:
            continue
        if jjwxc_index.get((title, author)):
            continue
        cache_key = _metadata_key(title, author)
        if cache_key in cache:
            continue
        pending.append(
            {
                "key": cache_key,
                "title": title,
                "author": author,
                "sample": text[:1800],
            }
        )
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        for key, metadata in classify_many_with_codex(batch).items():
            cache[key] = metadata_to_cache(metadata)


def metadata_for_book(
    *,
    title: str,
    author: str,
    text: str,
    jjwxc_index: dict[tuple[str, str], dict[str, str]],
    cache: dict[str, dict[str, str]],
    fetch_jjwxc: bool,
    codex_classify: bool,
) -> tuple[JjwxcTypeMetadata, str]:
    jjwxc = jjwxc_index.get((title, author), {})
    jjwxc_url = jjwxc.get("jjwxc_url", "")
    if fetch_jjwxc and jjwxc_url:
        cached = cache.get(jjwxc_url)
        if cached:
            return JjwxcTypeMetadata(**cached), jjwxc_url
        try:
            fetched = fetch_jjwxc_type(jjwxc_url)
        except Exception:
            fetched = None
        if fetched:
            cache[jjwxc_url] = asdict(fetched)
            return fetched, jjwxc_url

    article_type = jjwxc.get("article_type", "")
    if article_type:
        from src.metadata.jjwxc import parse_article_type

        parsed = parse_article_type(article_type, source="jjwxc-ranking")
        if parsed:
            return parsed, jjwxc_url

    local_cache_key = _metadata_key(title, author)
    for cache_key in _metadata_cache_keys(title, author):
        cached = cache.get(cache_key)
        if cached:
            if cache_key != local_cache_key:
                cache[local_cache_key] = cached
            return JjwxcTypeMetadata(**cached), jjwxc_url
    if codex_classify:
        codex_metadata = classify_with_codex(title=title, author=author, sample=text[:6000])
        if codex_metadata:
            cache[local_cache_key] = metadata_to_cache(codex_metadata)
            return codex_metadata, jjwxc_url

    return classify_from_text(title, author, sample=text[:5000]), jjwxc_url


def infer_txt_title_author(
    txt_path: Path,
    *,
    title: str | None = None,
    author: str | None = None,
) -> tuple[str, str]:
    inferred_title = (title or "").strip()
    inferred_author = (author or "").strip()
    if inferred_title and inferred_author:
        return inferred_title, inferred_author

    for line in txt_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not inferred_title:
            inferred_title = stripped
            continue
        if not inferred_author and stripped.startswith("作者："):
            inferred_author = stripped.removeprefix("作者：").strip()
        break

    if not inferred_title:
        inferred_title = txt_path.stem
    if not inferred_author:
        inferred_author = txt_path.parent.name
    return inferred_title, _canonical_author(inferred_author)


def load_manifest(output_root: Path) -> list[dict[str, object]]:
    manifest_path = output_root / "dataset_manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(output_root: Path, entries: list[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def manifest_path(path: Path | None, *, output_root: Path) -> str:
    """Store portable paths relative to the dataset manifest directory."""
    if path is None:
        return ""
    return Path(
        os.path.relpath(path.expanduser().resolve(), output_root.expanduser().resolve())
    ).as_posix()


def upsert_manifest_entry(output_root: Path, entry: ExportEntry) -> None:
    entries = load_manifest(output_root)
    entry_dict = asdict(entry)
    target_key = (entry.author, entry.title)
    for index, existing in enumerate(entries):
        if (existing.get("author"), existing.get("title")) == target_key:
            entries[index] = entry_dict
            break
    else:
        entries.append(entry_dict)
    save_manifest(output_root, entries)


def build_dataset_entry_from_text(
    *,
    text: str,
    title: str,
    author: str,
    source_epub: Path | None,
    txt_path: Path,
    output_root: Path,
    jjwxc_manifest: Path | None,
    jjwxc_top50: Path | None,
    fetch_jjwxc: bool,
    codex_classify: bool,
    time_area: str | None = None,
    genre: str | None = None,
) -> ExportEntry:
    author = _canonical_author(author)
    text = _normalize_text_author_header(text, author)
    cache_path = output_root / "metadata_cache.json"
    cache = load_cache(cache_path)
    jjwxc_index = load_jjwxc_manifest(jjwxc_manifest, jjwxc_top50)

    if time_area and genre:
        metadata = JjwxcTypeMetadata(
            article_type="manual override",
            time_area=time_area,
            genre=genre,
            source="manual",
        )
        jjwxc_url = ""
        cache[_metadata_key(title, author)] = metadata_to_cache(metadata)
    else:
        metadata, jjwxc_url = metadata_for_book(
            title=title,
            author=author,
            text=text,
            jjwxc_index=jjwxc_index,
            cache=cache,
            fetch_jjwxc=fetch_jjwxc,
            codex_classify=codex_classify,
        )

    save_cache(cache_path, cache)
    return ExportEntry(
        source_epub=manifest_path(source_epub, output_root=output_root),
        txt_path=manifest_path(txt_path, output_root=output_root),
        title=title,
        author=author,
        time_area=metadata.time_area,
        genre=metadata.genre,
        metadata_source=metadata.source,
        article_type=metadata.article_type,
        jjwxc_url=jjwxc_url,
        char_count=len(re.sub(r"\s+", "", text)),
        status="upserted",
    )


def upsert_txt_dataset_entry(
    *,
    txt_path: Path,
    output_root: Path,
    jjwxc_manifest: Path | None,
    jjwxc_top50: Path | None,
    fetch_jjwxc: bool,
    codex_classify: bool,
    title: str | None = None,
    author: str | None = None,
    source_epub: Path | None = None,
    time_area: str | None = None,
    genre: str | None = None,
) -> ExportEntry:
    title, author = infer_txt_title_author(txt_path, title=title, author=author)
    text = txt_path.read_text(encoding="utf-8")
    entry = build_dataset_entry_from_text(
        text=text,
        title=title,
        author=author,
        source_epub=source_epub,
        txt_path=txt_path,
        output_root=output_root,
        jjwxc_manifest=jjwxc_manifest,
        jjwxc_top50=jjwxc_top50,
        fetch_jjwxc=fetch_jjwxc,
        codex_classify=codex_classify,
        time_area=time_area,
        genre=genre,
    )
    upsert_manifest_entry(output_root, entry)
    return entry


def export_single_epub_txt(
    *,
    epub_path: Path,
    output_root: Path,
    jjwxc_manifest: Path | None,
    jjwxc_top50: Path | None,
    overwrite: bool,
    fetch_jjwxc: bool,
    codex_classify: bool,
) -> ExportEntry:
    title, author, text = extract_epub_text(epub_path)
    author = _canonical_author(author)
    text = _normalize_text_author_header(text, author)
    txt_path = dataset_txt_output_path(
        output_root,
        title=title,
        author=author,
        time_area="",
        genre="",
    )
    status = "skipped_existing"
    if overwrite or not txt_path.exists():
        txt_path.write_text(text, encoding="utf-8")
        status = "written"

    entry = build_dataset_entry_from_text(
        text=text,
        title=title,
        author=author,
        source_epub=epub_path,
        txt_path=txt_path,
        output_root=output_root,
        jjwxc_manifest=jjwxc_manifest,
        jjwxc_top50=jjwxc_top50,
        fetch_jjwxc=fetch_jjwxc,
        codex_classify=codex_classify,
    )
    entry.status = status
    upsert_manifest_entry(output_root, entry)
    return entry


def export_txt_dataset(
    *,
    books_root: Path,
    output_root: Path,
    jjwxc_manifest: Path | None,
    jjwxc_top50: Path | None,
    overwrite: bool,
    fetch_jjwxc: bool,
    codex_classify: bool,
    exclude_titles: set[str],
    exclude_metadata_keys: set[str],
) -> list[ExportEntry]:
    epub_paths = sorted(books_root.glob("**/*.epub"))
    cache_path = output_root / "metadata_cache.json"
    cache = load_cache(cache_path)
    jjwxc_index = load_jjwxc_manifest(jjwxc_manifest, jjwxc_top50)
    if codex_classify:
        prefill_codex_cache(
            epub_paths=epub_paths,
            jjwxc_index=jjwxc_index,
            cache=cache,
            exclude_titles=exclude_titles,
            exclude_metadata_keys=exclude_metadata_keys,
        )
    entries: list[ExportEntry] = []
    seen_txt_paths: set[Path] = set()

    for epub_path in epub_paths:
        try:
            title, author, text = extract_epub_text(epub_path)
            author = _canonical_author(author)
            text = _normalize_text_author_header(text, author)
            if title in exclude_titles or _metadata_key(title, author) in exclude_metadata_keys:
                continue
            metadata, jjwxc_url = metadata_for_book(
                title=title,
                author=author,
                text=text,
                jjwxc_index=jjwxc_index,
                cache=cache,
                fetch_jjwxc=fetch_jjwxc,
                codex_classify=codex_classify,
            )
            txt_path = dataset_txt_output_path(
                output_root,
                title=title,
                author=author,
                time_area=metadata.time_area,
                genre=metadata.genre,
            )
            if txt_path in seen_txt_paths:
                txt_path = dataset_txt_output_path(
                    output_root,
                    title=epub_path.stem,
                    author=author,
                    time_area=metadata.time_area,
                    genre=metadata.genre,
                )
                counter = 2
                while txt_path in seen_txt_paths:
                    txt_path = dataset_txt_output_path(
                        output_root,
                        title=f"{epub_path.stem}.{counter}",
                        author=author,
                        time_area=metadata.time_area,
                        genre=metadata.genre,
                    )
                    counter += 1
            seen_txt_paths.add(txt_path)
            status = "skipped_existing"
            if overwrite or not txt_path.exists():
                txt_path.write_text(text, encoding="utf-8")
                status = "written"
            entries.append(
                ExportEntry(
                    source_epub=manifest_path(epub_path, output_root=output_root),
                    txt_path=manifest_path(txt_path, output_root=output_root),
                    title=title,
                    author=author,
                    time_area=metadata.time_area,
                    genre=metadata.genre,
                    metadata_source=metadata.source,
                    article_type=metadata.article_type,
                    jjwxc_url=jjwxc_url,
                    char_count=len(re.sub(r"\s+", "", text)),
                    status=status,
                )
            )
        except Exception as exc:
            entries.append(
                ExportEntry(
                    source_epub=manifest_path(epub_path, output_root=output_root),
                    txt_path="",
                    title=epub_path.stem,
                    author=_canonical_author(epub_path.parent.name),
                    time_area="近代现代",
                    genre="爱情",
                    metadata_source="error",
                    article_type="",
                    jjwxc_url="",
                    char_count=0,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    save_cache(cache_path, cache)
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(entry) for entry in entries], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and maintain targeted TXT dataset entries.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export-txt",
        help="Export TXT from an explicit EPUB or explicit EPUB tree.",
    )
    export_source = export_parser.add_mutually_exclusive_group(required=True)
    export_source.add_argument("--epub", type=Path, help="Single EPUB to export and upsert")
    export_source.add_argument("--books-root", type=Path, help="Explicit EPUB directory for a bulk export")
    export_parser.add_argument(
        "--output-root", type=Path, default=Path("research/datasets")
    )
    export_parser.add_argument(
        "--jjwxc-manifest",
        type=Path,
        default=Path("research/generated/corpus_acquisition/ranking/manifest.json"),
    )
    export_parser.add_argument(
        "--jjwxc-top50",
        type=Path,
        default=Path("research/generated/corpus_acquisition/ranking/top50.json"),
    )
    export_parser.add_argument("--overwrite", action="store_true")
    export_parser.add_argument("--no-fetch-jjwxc", action="store_true")
    export_parser.add_argument("--no-codex-classify", action="store_true")
    export_parser.add_argument(
        "--exclude-title",
        action="append",
        default=[],
        help="Bulk only: skip EPUBs whose metadata title exactly matches this value; repeatable",
    )
    export_parser.add_argument(
        "--exclude-book",
        action="append",
        default=[],
        metavar="AUTHOR::TITLE",
        help="Bulk only: skip EPUBs whose metadata author/title match this value; repeatable",
    )

    upsert_parser = subparsers.add_parser(
        "upsert",
        help="Upsert one existing TXT file into dataset_manifest.json.",
    )
    upsert_parser.add_argument("--txt", type=Path, required=True)
    upsert_parser.add_argument("--title")
    upsert_parser.add_argument("--author")
    upsert_parser.add_argument("--source-epub", type=Path)
    upsert_parser.add_argument(
        "--output-root", type=Path, default=Path("research/datasets")
    )
    upsert_parser.add_argument(
        "--jjwxc-manifest",
        type=Path,
        default=Path("research/generated/corpus_acquisition/ranking/manifest.json"),
    )
    upsert_parser.add_argument(
        "--jjwxc-top50",
        type=Path,
        default=Path("research/generated/corpus_acquisition/ranking/top50.json"),
    )
    upsert_parser.add_argument("--time-area", choices=TIME_AREAS)
    upsert_parser.add_argument("--genre", choices=GENRES)
    upsert_parser.add_argument("--no-fetch-jjwxc", action="store_true")
    upsert_parser.add_argument("--no-codex-classify", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "upsert":
        entry = upsert_txt_dataset_entry(
            txt_path=args.txt,
            output_root=args.output_root,
            jjwxc_manifest=args.jjwxc_manifest,
            jjwxc_top50=args.jjwxc_top50,
            fetch_jjwxc=not args.no_fetch_jjwxc,
            codex_classify=not args.no_codex_classify,
            title=args.title,
            author=args.author,
            source_epub=args.source_epub,
            time_area=args.time_area,
            genre=args.genre,
        )
        print(f"upserted {entry.author}::{entry.title}")
        print(f"txt: {entry.txt_path}")
        print(f"manifest: {args.output_root / 'dataset_manifest.json'}")
        return 0

    if args.epub:
        entry = export_single_epub_txt(
            epub_path=args.epub,
            output_root=args.output_root,
            jjwxc_manifest=args.jjwxc_manifest,
            jjwxc_top50=args.jjwxc_top50,
            overwrite=args.overwrite,
            fetch_jjwxc=not args.no_fetch_jjwxc,
            codex_classify=not args.no_codex_classify,
        )
        print(
            f"exported 1 EPUB: written={1 if entry.status == 'written' else 0}, "
            f"skipped_existing={1 if entry.status == 'skipped_existing' else 0}, failed=0"
        )
        print(f"txt: {entry.txt_path}")
        print(f"manifest: {args.output_root / 'dataset_manifest.json'}")
        return 0

    entries = export_txt_dataset(
        books_root=args.books_root,
        output_root=args.output_root,
        jjwxc_manifest=args.jjwxc_manifest,
        jjwxc_top50=args.jjwxc_top50,
        overwrite=args.overwrite,
        fetch_jjwxc=not args.no_fetch_jjwxc,
        codex_classify=not args.no_codex_classify,
        exclude_titles=set(args.exclude_title),
        exclude_metadata_keys={f"local::{item}" for item in args.exclude_book},
    )
    written = sum(1 for entry in entries if entry.status == "written")
    skipped = sum(1 for entry in entries if entry.status == "skipped_existing")
    failed = sum(1 for entry in entries if entry.status == "failed")
    print(
        f"exported {len(entries)} EPUBs: written={written}, "
        f"skipped_existing={skipped}, failed={failed}"
    )
    print(f"manifest: {args.output_root / 'dataset_manifest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
