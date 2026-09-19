"""Repair and inspect existing EPUB archives using the shared source rules."""

from __future__ import annotations

import html
import os
import posixpath
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from src.content.normalization import (
    CHAPTER_LIKE_TITLE_RE,
    CHAPTER_NUMBER_PREFIX_RE,
    NormalizationIssue,
    NormalizationReport,
    _center_paragraph_opening,
    _grouped_fanwai_title_labels,
    _local_name,
    _scan_member_issues,
    _tag_pattern,
    _visible_fragment_text,
    normalize_member,
)


def _scan_epub_structure(path: Path, report: NormalizationReport) -> None:
    from src.epub.validate import validate_epub

    issues, _ = validate_epub(path)
    for issue in issues:
        report.issues.append(
            NormalizationIssue(
                kind="epub_structure",
                member="EPUB/nav.xhtml + EPUB/toc.ncx + chapter documents",
                message=issue,
                excerpt="",
                recommended_action=(
                    "Let Codex inspect nav.xhtml, toc.ncx, content.opf, and the "
                    "referenced chapter before changing structure or numbering."
                ),
            )
        )


def _backup_target(path: Path, backup_dir: Path) -> Path:
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve() / "books")
    except ValueError:
        relative = Path(path.name)
    return backup_dir / relative


def _replace_exact_surface_text(
    text: str,
    *,
    tags: set[str],
    before: str,
    after: str,
) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        opening, fragment, closing = match.groups()
        if _visible_fragment_text(fragment).strip() != before:
            return match.group(0)
        count += 1
        return opening + html.escape(after, quote=False) + closing

    return _tag_pattern(tags).sub(replace, text), count


def _center_exact_heading(
    text: str,
    *,
    tag: str,
    title: str,
) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        opening, fragment, closing = match.groups()
        if _visible_fragment_text(fragment).strip() != title:
            return match.group(0)
        centered = _center_paragraph_opening(opening)
        if centered == opening:
            return match.group(0)
        count += 1
        return centered + fragment + closing

    return _tag_pattern({tag}).sub(replace, text), count


def _repair_epub2_missing_visible_toc(
    members: dict[str, bytes],
    report: NormalizationReport,
) -> dict[str, bytes]:
    """Repair a narrow EPUB 2 pattern with NCX-only, bare chapter navigation."""

    staged_report = NormalizationReport(path=report.path, applied=report.applied)
    opf_member = next(
        (name for name in members if name.lower().endswith(".opf")),
        None,
    )
    if opf_member is None:
        return members
    try:
        opf_root = ET.fromstring(members[opf_member])
    except ET.ParseError:
        return members
    if not opf_root.attrib.get("version", "").startswith("2"):
        return members

    opf_dir = posixpath.dirname(opf_member)
    manifest = next(
        (element for element in opf_root if _local_name(element.tag) == "manifest"),
        None,
    )
    spine = next(
        (element for element in opf_root if _local_name(element.tag) == "spine"),
        None,
    )
    if manifest is None or spine is None:
        return members

    manifest_items: list[tuple[str, str, str, str]] = []
    for item in manifest:
        if _local_name(item.tag) != "item":
            continue
        item_id = item.attrib.get("id", "")
        href = item.attrib.get("href", "")
        media_type = item.attrib.get("media-type", "")
        if not item_id or not href:
            continue
        archive_member = posixpath.normpath(
            posixpath.join(opf_dir, href.split("#", 1)[0])
        )
        manifest_items.append((item_id, href, media_type, archive_member))

    if any(
        posixpath.basename(archive_member).lower() == "nav.xhtml"
        for _, _, _, archive_member in manifest_items
    ):
        return members
    ncx_item = next(
        (item for item in manifest_items if item[2] == "application/x-dtbncx+xml"),
        None,
    )
    if ncx_item is None:
        ncx_item = next(
            (
                item
                for item in manifest_items
                if posixpath.basename(item[3]).lower() == "toc.ncx"
            ),
            None,
        )
    if ncx_item is None or ncx_item[3] not in members:
        return members
    ncx_member = ncx_item[3]
    try:
        ncx_root = ET.fromstring(members[ncx_member])
    except ET.ParseError:
        return members

    nav_map = next(
        (
            element
            for element in ncx_root.iter()
            if _local_name(element.tag) == "navMap"
        ),
        None,
    )
    if nav_map is None:
        return members
    direct_nav_points = [
        element for element in nav_map if _local_name(element.tag) == "navPoint"
    ]
    all_nav_points = [
        element for element in nav_map.iter() if _local_name(element.tag) == "navPoint"
    ]
    if len(direct_nav_points) != len(all_nav_points):
        return members

    entries: list[dict[str, Any]] = []
    for nav_point in direct_nav_points:
        label_element = next(
            (
                element
                for element in nav_point.iter()
                if _local_name(element.tag) == "text"
            ),
            None,
        )
        content_element = next(
            (
                element
                for element in nav_point.iter()
                if _local_name(element.tag) == "content"
            ),
            None,
        )
        if label_element is None or content_element is None:
            return members
        label = "".join(label_element.itertext()).strip()
        source = content_element.attrib.get("src", "")
        if not label or not source:
            return members
        source_path, separator, fragment = source.partition("#")
        archive_member = posixpath.normpath(
            posixpath.join(posixpath.dirname(ncx_member), source_path)
        )
        entries.append(
            {
                "label": label,
                "label_element": label_element,
                "source": source,
                "member": archive_member,
                "fragment": f"#{fragment}" if separator else "",
            }
        )

    prologue_index = next(
        (
            index
            for index, entry in enumerate(entries)
            if entry["label"] in {"序章", "序言"}
        ),
        None,
    )
    if prologue_index is None:
        return members
    main_end = len(entries)
    for index in range(prologue_index + 1, len(entries)):
        if re.match(
            r"^(?:番外|后记|尾声|终章|终卷|附录|跋|末页)",
            entries[index]["label"],
        ):
            main_end = index
            break
    main_entries = entries[prologue_index + 1 : main_end]
    if len(main_entries) < 2:
        return members

    numbered = [
        bool(CHAPTER_NUMBER_PREFIX_RE.match(entry["label"])) for entry in main_entries
    ]
    if all(numbered):
        add_numbers = False
    elif not any(numbered) and all(
        len(entry["label"]) <= 40 and not CHAPTER_LIKE_TITLE_RE.match(entry["label"])
        for entry in main_entries
    ):
        add_numbers = True
    else:
        return members

    chapter_entries = [entries[prologue_index], *main_entries]
    chapter_surfaces: list[tuple[dict[str, Any], str, str]] = []
    for index, entry in enumerate(chapter_entries):
        archive_member = entry["member"]
        if archive_member not in members or not archive_member.lower().endswith(
            (".html", ".xhtml")
        ):
            return members
        try:
            chapter_root = ET.fromstring(members[archive_member])
        except ET.ParseError:
            return members
        title_element = next(
            (
                element
                for element in chapter_root.iter()
                if _local_name(element.tag) == "title"
            ),
            None,
        )
        heading_element = next(
            (
                element
                for element in chapter_root.iter()
                if _local_name(element.tag) in {"h1", "h2", "h3"}
            ),
            None,
        )
        label = entry["label"]
        if (
            title_element is None
            or heading_element is None
            or len(title_element)
            or len(heading_element)
            or (title_element.text or "").strip() != label
            or (heading_element.text or "").strip() != label
        ):
            return members
        new_label = f"第{index}章 {label}" if add_numbers and index > 0 else label
        chapter_surfaces.append((entry, _local_name(heading_element.tag), new_label))

    rewritten = dict(members)
    for entry, heading_tag, new_label in chapter_surfaces:
        archive_member = entry["member"]
        old_label = entry["label"]
        chapter_text = rewritten[archive_member].decode("utf-8")
        if new_label != old_label:
            chapter_text, title_count = _replace_exact_surface_text(
                chapter_text,
                tags={"title"},
                before=old_label,
                after=new_label,
            )
            chapter_text, heading_count = _replace_exact_surface_text(
                chapter_text,
                tags={heading_tag},
                before=old_label,
                after=new_label,
            )
            staged_report.record_change(
                "chapter_number_prefix_added",
                archive_member,
                title_count + heading_count,
                old_label,
                new_label,
            )
            entry["label_element"].text = new_label
            staged_report.record_change(
                "chapter_number_prefix_added",
                ncx_member,
                1,
                old_label,
                new_label,
            )
            entry["label"] = new_label
        chapter_text, centered_count = _center_exact_heading(
            chapter_text,
            tag=heading_tag,
            title=new_label,
        )
        staged_report.record_change(
            "chapter_heading_centered",
            archive_member,
            centered_count,
            new_label,
            new_label,
        )
        rewritten[archive_member] = chapter_text.encode("utf-8")

    ncx_namespace = (
        ncx_root.tag[1:].split("}", 1)[0] if ncx_root.tag.startswith("{") else ""
    )
    if ncx_namespace:
        ET.register_namespace("", ncx_namespace)
    rewritten[ncx_member] = ET.tostring(
        ncx_root,
        encoding="utf-8",
        xml_declaration=True,
    )

    nav_member = posixpath.normpath(posixpath.join(opf_dir, "nav.xhtml"))
    nav_dir = posixpath.dirname(nav_member)
    toc_entries = entries[prologue_index:]
    links: list[str] = []
    for entry in toc_entries:
        relative = posixpath.relpath(entry["member"], nav_dir or ".")
        href = relative + entry["fragment"]
        links.append(
            '      <li><a href="'
            + html.escape(href, quote=True)
            + '">'
            + html.escape(entry["label"], quote=False)
            + "</a></li>"
        )
    css_item = next(
        (item for item in manifest_items if item[2] == "text/css"),
        None,
    )
    css_link = ""
    if css_item is not None:
        css_href = posixpath.relpath(css_item[3], nav_dir or ".")
        css_link = (
            '    <link rel="stylesheet" type="text/css" href="'
            + html.escape(css_href, quote=True)
            + '"/>\n'
        )
    nav_text = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops">\n'
        "  <head>\n"
        "    <title>目录</title>\n"
        f"{css_link}"
        "  </head>\n"
        '  <body class="article">\n'
        '    <nav epub:type="toc" id="toc">\n'
        '      <h1 style="text-align: center; text-indent: 0;">目录</h1>\n'
        "      <ol>\n" + "\n".join(links) + "\n"
        "      </ol>\n"
        "    </nav>\n"
        "  </body>\n"
        "</html>\n"
    )
    ET.fromstring(nav_text.encode("utf-8"))
    rewritten[nav_member] = nav_text.encode("utf-8")
    staged_report.record_change(
        "visible_toc_page_added",
        nav_member,
        1,
        "",
        "目录",
    )

    existing_ids = {item[0] for item in manifest_items}
    toc_id = "reader_toc"
    suffix = 2
    while toc_id in existing_ids:
        toc_id = f"reader_toc_{suffix}"
        suffix += 1
    prologue_member = entries[prologue_index]["member"]
    prologue_id = next(
        (
            item_id
            for item_id, _, _, archive_member in manifest_items
            if archive_member == prologue_member
        ),
        None,
    )
    if prologue_id is None:
        return members

    opf_text = rewritten[opf_member].decode("utf-8")
    manifest_close = re.search(
        r"(?P<indent>^[ \t]*)</(?:[A-Za-z_][\w.-]*:)?manifest\s*>",
        opf_text,
        re.MULTILINE,
    )
    if manifest_close is None:
        return members
    indent = manifest_close.group("indent")
    manifest_entry = (
        f'{indent}  <item id="{toc_id}" href="nav.xhtml" '
        'media-type="application/xhtml+xml"/>\n'
    )
    opf_text = (
        opf_text[: manifest_close.start()]
        + manifest_entry
        + opf_text[manifest_close.start() :]
    )

    prologue_itemref = re.search(
        rf"<(?:[A-Za-z_][\w.-]*:)?itemref\b"
        rf"(?=[^>]*\bidref=[\"']{re.escape(prologue_id)}[\"'])[^>]*/?>",
        opf_text,
    )
    if prologue_itemref is None:
        return members
    line_start = opf_text.rfind("\n", 0, prologue_itemref.start()) + 1
    itemref_indent = opf_text[line_start : prologue_itemref.start()]
    spine_entry = f'{itemref_indent}<itemref idref="{toc_id}"/>\n'
    opf_text = opf_text[:line_start] + spine_entry + opf_text[line_start:]

    guide_close = re.search(
        r"(?P<indent>^[ \t]*)</(?:[A-Za-z_][\w.-]*:)?guide\s*>",
        opf_text,
        re.MULTILINE,
    )
    guide_entry = '<reference type="toc" href="nav.xhtml" title="目录"/>'
    if guide_close is not None:
        guide_indent = guide_close.group("indent")
        insertion = f"{guide_indent}  {guide_entry}\n"
        opf_text = (
            opf_text[: guide_close.start()]
            + insertion
            + opf_text[guide_close.start() :]
        )
    else:
        package_close = re.search(
            r"(?P<indent>^[ \t]*)</(?:[A-Za-z_][\w.-]*:)?package\s*>",
            opf_text,
            re.MULTILINE,
        )
        if package_close is None:
            return members
        package_indent = package_close.group("indent")
        insertion = (
            f"{package_indent}  <guide>\n"
            f"{package_indent}    {guide_entry}\n"
            f"{package_indent}  </guide>\n"
        )
        opf_text = (
            opf_text[: package_close.start()]
            + insertion
            + opf_text[package_close.start() :]
        )
    ET.fromstring(opf_text.encode("utf-8"))
    rewritten[opf_member] = opf_text.encode("utf-8")
    staged_report.record_change(
        "visible_toc_registered",
        opf_member,
        3,
        "",
        "manifest + spine + guide",
    )
    report.change_counts.update(staged_report.change_counts)
    for member, counts in staged_report.member_changes.items():
        report.member_changes[member].update(counts)
    for kind, samples in staged_report.samples.items():
        remaining = max(0, 5 - len(report.samples[kind]))
        report.samples[kind].extend(samples[:remaining])
    return rewritten


def normalize_epub(
    path: Path,
    *,
    apply: bool = True,
    backup_dir: Path | None = None,
    overwrite_backup: bool = False,
) -> NormalizationReport:
    path = Path(path)
    report = NormalizationReport(path=path, applied=apply)
    rewritten: dict[str, bytes] = {}
    with zipfile.ZipFile(path, "r") as source:
        infos = source.infolist()
        original = {info.filename: source.read(info.filename) for info in infos}
    structured = _repair_epub2_missing_visible_toc(original, report)
    grouped_fanwai_titles = _grouped_fanwai_title_labels(structured)
    for info in infos:
        data = structured[info.filename]
        normalized = normalize_member(
            info.filename,
            data,
            report,
            grouped_fanwai_titles=grouped_fanwai_titles,
        )
        rewritten[info.filename] = normalized
    for member, data in structured.items():
        if member in rewritten:
            continue
        rewritten[member] = normalize_member(
            member,
            data,
            report,
            grouped_fanwai_titles=grouped_fanwai_titles,
        )

    for member, data in rewritten.items():
        _scan_member_issues(member, data, report)

    if not apply or not report.total_changes:
        _scan_epub_structure(path, report)
        return report

    if backup_dir is not None:
        backup = _backup_target(path, backup_dir)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if overwrite_backup or not backup.exists():
            backup.write_bytes(path.read_bytes())

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}-normalize-",
        suffix=".epub",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        with (
            zipfile.ZipFile(path, "r") as source,
            zipfile.ZipFile(temporary, "w") as target,
        ):
            for info in source.infolist():
                target.writestr(info, rewritten[info.filename])
            for member, data in rewritten.items():
                if member not in original:
                    target.writestr(member, data)
        with zipfile.ZipFile(temporary, "r") as check:
            bad_member = check.testzip()
            if bad_member:
                raise RuntimeError(f"CRC failure after normalization: {bad_member}")
            for name in check.namelist():
                if name.endswith((".html", ".xhtml", ".ncx", ".opf")):
                    ET.fromstring(check.read(name))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _scan_epub_structure(path, report)
    return report
