#!/usr/bin/env python3
"""Validate chapter numbering consistency in generated EPUB files."""
from __future__ import annotations

import argparse
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile


CHAPTER_RE = re.compile(r"^\s*第\s*([0-9]+|[零〇一二两三四五六七八九十百千]+)\s*([章回])")
SPACE_RE = re.compile(r"\s+")
CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000}


@dataclass(frozen=True)
class ChapterRef:
    source: str
    href: str
    title: str
    number_text: str
    number: int
    unit: str


def normalize_text(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip()


def normalize_href(base: str, href: str) -> str:
    path = href.split("#", 1)[0]
    return posixpath.normpath(posixpath.join(posixpath.dirname(base), path))


def chinese_to_int(value: str) -> int:
    total = 0
    current = 0
    for char in value:
        if char in CHINESE_DIGITS:
            current = CHINESE_DIGITS[char]
        elif char in CHINESE_UNITS:
            unit = CHINESE_UNITS[char]
            total += (current or 1) * unit
            current = 0
        else:
            raise ValueError(f"unsupported Chinese numeral: {value}")
    return total + current


def parse_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    return chinese_to_int(value)


def chapter_from_title(source: str, href: str, title: str) -> ChapterRef | None:
    match = CHAPTER_RE.search(title)
    if not match:
        return None
    number_text, unit = match.groups()
    return ChapterRef(
        source=source,
        href=href,
        title=normalize_text(title),
        number_text=number_text,
        number=parse_number(number_text),
        unit=unit,
    )


def parse_xml(data: bytes, source: str) -> ET.Element:
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"{source}: XML parse error: {exc}") from exc


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def text_content(element: ET.Element) -> str:
    return normalize_text("".join(element.itertext()))


def find_nav_entries(epub: ZipFile) -> list[ChapterRef]:
    entries: list[ChapterRef] = []
    for name in epub.namelist():
        if not name.endswith("nav.xhtml"):
            continue
        root = parse_xml(epub.read(name), name)
        for element in root.iter():
            if local_name(element.tag) != "a":
                continue
            href = element.attrib.get("href")
            if not href:
                continue
            chapter = chapter_from_title("nav", normalize_href(name, href), text_content(element))
            if chapter:
                entries.append(chapter)
    return entries


def find_ncx_entries(epub: ZipFile) -> list[ChapterRef]:
    entries: list[ChapterRef] = []
    for name in epub.namelist():
        if not name.endswith("toc.ncx"):
            continue
        root = parse_xml(epub.read(name), name)
        for nav_point in root.iter():
            if local_name(nav_point.tag) != "navPoint":
                continue
            href = None
            label = None
            for child in nav_point.iter():
                child_name = local_name(child.tag)
                if child_name == "content":
                    href = child.attrib.get("src")
                elif child_name == "text" and label is None:
                    label = text_content(child)
            if href and label:
                chapter = chapter_from_title("ncx", normalize_href(name, href), label)
                if chapter:
                    entries.append(chapter)
    return entries


def find_document_chapter(epub: ZipFile, href: str) -> ChapterRef | None:
    try:
        data = epub.read(href)
    except KeyError:
        return None
    try:
        root = parse_xml(data, href)
    except ValueError:
        return None
    for element in root.iter():
        if local_name(element.tag) in {"title", "h1", "h2"}:
            chapter = chapter_from_title("document", href, text_content(element))
            if chapter:
                return chapter
    return None


def format_ref(ref: ChapterRef) -> str:
    return f"{ref.href}: {ref.title}"


def validate_sequence(entries: list[ChapterRef]) -> list[str]:
    issues: list[str] = []
    by_unit: dict[str, list[ChapterRef]] = {}
    for entry in entries:
        by_unit.setdefault(entry.unit, []).append(entry)

    for unit, unit_entries in by_unit.items():
        seen: dict[int, list[ChapterRef]] = {}
        for entry in unit_entries:
            seen.setdefault(entry.number, []).append(entry)
        for number, duplicates in seen.items():
            if len(duplicates) > 1:
                refs = "; ".join(format_ref(entry) for entry in duplicates)
                issues.append(f"duplicate 第{number}{unit}: {refs}")

        previous = unit_entries[0]
        for current in unit_entries[1:]:
            if current.number == previous.number:
                previous = current
                continue
            expected = previous.number + 1
            if current.number != expected:
                issues.append(
                    f"number gap/order issue before {format_ref(current)}: "
                    f"previous 第{previous.number}{unit}, expected 第{expected}{unit}"
                )
            previous = current
    return issues


def validate_cross_references(
    epub: ZipFile,
    nav_entries: list[ChapterRef],
    ncx_entries: list[ChapterRef],
) -> list[str]:
    issues: list[str] = []
    ncx_by_href: dict[str, ChapterRef] = {entry.href: entry for entry in ncx_entries}

    if len(nav_entries) != len(ncx_entries):
        issues.append(f"nav/ncx numbered entry count mismatch: nav={len(nav_entries)} ncx={len(ncx_entries)}")

    for nav_entry in nav_entries:
        ncx_entry = ncx_by_href.get(nav_entry.href)
        if not ncx_entry:
            issues.append(f"nav entry missing from ncx: {format_ref(nav_entry)}")
        elif (nav_entry.number, nav_entry.unit) != (ncx_entry.number, ncx_entry.unit):
            issues.append(
                f"nav/ncx number mismatch for {nav_entry.href}: "
                f"nav 第{nav_entry.number}{nav_entry.unit}, "
                f"ncx 第{ncx_entry.number}{ncx_entry.unit}"
            )

        document_entry = find_document_chapter(epub, nav_entry.href)
        if not document_entry:
            issues.append(f"chapter document missing or lacks numbered title: {format_ref(nav_entry)}")
        elif (nav_entry.number, nav_entry.unit) != (document_entry.number, document_entry.unit):
            issues.append(
                f"nav/document number mismatch for {nav_entry.href}: "
                f"nav 第{nav_entry.number}{nav_entry.unit}, "
                f"document 第{document_entry.number}{document_entry.unit}"
            )

    nav_hrefs = {entry.href for entry in nav_entries}
    for ncx_entry in ncx_entries:
        if ncx_entry.href not in nav_hrefs:
            issues.append(f"ncx entry missing from nav: {format_ref(ncx_entry)}")

    return issues


def validate_epub(path: Path) -> tuple[list[str], int]:
    issues: list[str] = []
    try:
        with ZipFile(path, "r") as epub:
            corrupt_member = epub.testzip()
            if corrupt_member:
                issues.append(f"zip integrity failure at member: {corrupt_member}")

            nav_entries = find_nav_entries(epub)
            ncx_entries = find_ncx_entries(epub)
            if not nav_entries:
                issues.append("no numbered chapter entries found in nav.xhtml")
            else:
                issues.extend(validate_sequence(nav_entries))
            issues.extend(validate_cross_references(epub, nav_entries, ncx_entries))
            return issues, len(nav_entries)
    except (BadZipFile, ValueError) as exc:
        return [str(exc)], 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("epub")],
        help="EPUB files or directories containing EPUB files",
    )
    args = parser.parse_args(argv)

    epub_paths: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            epub_paths.extend(sorted(path.glob("*.epub")))
        else:
            epub_paths.append(path)

    if not epub_paths:
        print("No EPUB files found.", file=sys.stderr)
        return 2

    failed = 0
    for path in epub_paths:
        issues, chapter_count = validate_epub(path)
        if issues:
            failed += 1
            print(f"FAIL {path} ({chapter_count} numbered chapter entries)")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"OK   {path} ({chapter_count} numbered chapter entries)")

    print(f"\nValidated {len(epub_paths)} EPUB file(s); {failed} file(s) with issues.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
