"""Validate EPUB chapter numbering and navigation."""

from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

CHINESE_CHAPTER_RE = re.compile(
    r"^\s*第\s*([0-9]+|[零〇一二两三四五六七八九十百千]+)\s*([章回])"
)


ENGLISH_CHAPTER_RE = re.compile(r"^\s*Chapter\s+([0-9]+)\b", re.IGNORECASE)


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
    fanwai_before: int = 0


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


def chapter_from_title(
    source: str,
    href: str,
    title: str,
    *,
    fanwai_before: int = 0,
) -> ChapterRef | None:
    match = CHINESE_CHAPTER_RE.search(title)
    if match:
        number_text, unit = match.groups()
    else:
        english_match = ENGLISH_CHAPTER_RE.search(title)
        if not english_match:
            return None
        number_text = english_match.group(1)
        unit = "Chapter"
    return ChapterRef(
        source=source,
        href=href,
        title=normalize_text(title),
        number_text=number_text,
        number=parse_number(number_text),
        unit=unit,
        fanwai_before=fanwai_before,
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


def is_managed_fanwai_document(epub: ZipFile, href: str) -> bool:
    """Recognize synced extras even when their titles omit the word 番外."""
    try:
        root = parse_xml(epub.read(href), href)
    except (KeyError, ValueError):
        return False
    return any(
        local_name(element.tag) == "meta"
        and element.get("name") == "notion-source"
        and bool(element.get("content", "").strip())
        for element in root.iter()
    )


def find_nav_entries(epub: ZipFile) -> list[ChapterRef]:
    entries: list[ChapterRef] = []
    managed_fanwai: dict[str, bool] = {}
    for name in epub.namelist():
        if not name.endswith("nav.xhtml"):
            continue
        root = parse_xml(epub.read(name), name)
        fanwai_anchors: set[int] = set()
        for item in root.iter():
            if local_name(item.tag) != "li":
                continue
            label = next(
                (
                    text_content(child)
                    for child in item
                    if local_name(child.tag) in {"a", "span"}
                ),
                "",
            )
            if "番外" not in label:
                continue
            fanwai_anchors.update(
                id(descendant)
                for descendant in item.iter()
                if local_name(descendant.tag) == "a"
            )

        pending_fanwai_hrefs: set[str] = set()
        for element in root.iter():
            if local_name(element.tag) != "a":
                continue
            href = element.attrib.get("href")
            if not href:
                continue
            normalized_href = normalize_href(name, href)
            title = text_content(element)
            chapter = chapter_from_title(
                "nav",
                normalized_href,
                title,
                fanwai_before=len(pending_fanwai_hrefs),
            )
            if chapter:
                entries.append(chapter)
                pending_fanwai_hrefs.clear()
            else:
                if normalized_href not in managed_fanwai:
                    managed_fanwai[normalized_href] = is_managed_fanwai_document(
                        epub, normalized_href
                    )
                if (
                    "番外" in title
                    or id(element) in fanwai_anchors
                    or managed_fanwai[normalized_href]
                ):
                    pending_fanwai_hrefs.add(normalized_href)
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


def format_number(ref: ChapterRef) -> str:
    if ref.unit == "Chapter":
        return f"Chapter {ref.number_text}"
    return f"第{ref.number}{ref.unit}"


def format_expected_number(unit: str, number: int) -> str:
    if unit == "Chapter":
        return f"Chapter {number}"
    return f"第{number}{unit}"


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
                label = format_number(duplicates[0])
                issues.append(f"duplicate {label}: {refs}")

        previous = unit_entries[0]
        for current in unit_entries[1:]:
            if current.number == previous.number:
                previous = current
                continue
            expected = previous.number + 1
            if current.number != expected:
                missing = current.number - expected
                if missing > 0 and current.fanwai_before >= missing:
                    previous = current
                    continue
                issues.append(
                    f"number gap/order issue before {format_ref(current)}: "
                    f"previous {format_number(previous)}, expected {format_expected_number(unit, expected)}"
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
        issues.append(
            f"nav/ncx numbered entry count mismatch: nav={len(nav_entries)} ncx={len(ncx_entries)}"
        )

    for nav_entry in nav_entries:
        ncx_entry = ncx_by_href.get(nav_entry.href)
        if not ncx_entry:
            issues.append(f"nav entry missing from ncx: {format_ref(nav_entry)}")
        elif (nav_entry.number, nav_entry.unit) != (ncx_entry.number, ncx_entry.unit):
            issues.append(
                f"nav/ncx number mismatch for {nav_entry.href}: "
                f"nav {format_number(nav_entry)}, "
                f"ncx {format_number(ncx_entry)}"
            )

        document_entry = find_document_chapter(epub, nav_entry.href)
        if not document_entry:
            issues.append(
                f"chapter document missing or lacks numbered title: {format_ref(nav_entry)}"
            )
        elif (nav_entry.number, nav_entry.unit) != (
            document_entry.number,
            document_entry.unit,
        ):
            issues.append(
                f"nav/document number mismatch for {nav_entry.href}: "
                f"nav {format_number(nav_entry)}, "
                f"document {format_number(document_entry)}"
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
