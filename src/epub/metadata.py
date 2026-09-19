"""Read and update EPUB package metadata."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree

from src.metadata.catalog import (
    DEFAULT_PRIMARY_SUBJECT,
    MetadataEnrichmentReport,
    MetadataLookup,
    PackageMetadata,
    SourceMetadata,
)
from src.runtime.files import DEFAULT_BACKUP_DIR

OPF_NS = "http://www.idpf.org/2007/opf"

DC_NS = "http://purl.org/dc/elements/1.1/"

CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"

OPF_META = f"{{{OPF_NS}}}meta"

TARGET_DC_TAGS = {
    f"{{{DC_NS}}}title",
    f"{{{DC_NS}}}creator",
    f"{{{DC_NS}}}language",
    f"{{{DC_NS}}}date",
    f"{{{DC_NS}}}source",
    f"{{{DC_NS}}}description",
    f"{{{DC_NS}}}subject",
}


def _find_opf_member(archive: zipfile.ZipFile) -> str:
    if "EPUB/content.opf" in archive.namelist():
        return "EPUB/content.opf"
    try:
        container = etree.fromstring(archive.read("META-INF/container.xml"))
    except KeyError as error:
        raise ValueError("EPUB has no META-INF/container.xml") from error
    rootfile = container.find(f".//{{{CONTAINER_NS}}}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("EPUB container has no package rootfile")
    return str(rootfile.get("full-path"))


def _dc_texts(metadata: etree._Element, local_name: str) -> list[str]:
    return [
        (element.text or "").strip()
        for element in metadata.findall(f"{{{DC_NS}}}{local_name}")
    ]


def _package_metadata(opf_data: bytes) -> PackageMetadata:
    root = etree.fromstring(
        opf_data,
        etree.XMLParser(remove_blank_text=False, resolve_entities=False),
    )
    metadata = root.find(f"{{{OPF_NS}}}metadata")
    if metadata is None:
        raise ValueError("EPUB package has no metadata element")
    series = next(
        (
            (element.text or "").strip()
            for element in metadata.findall(OPF_META)
            if element.get("property") == "belongs-to-collection"
        ),
        "",
    )
    collection_ids = {
        element.get("id")
        for element in metadata.findall(OPF_META)
        if element.get("property") == "belongs-to-collection" and element.get("id")
    }
    position = next(
        (
            (element.text or "").strip()
            for element in metadata.findall(OPF_META)
            if element.get("property") == "group-position"
            and element.get("refines", "").removeprefix("#") in collection_ids
        ),
        "",
    )

    def first(name: str, default: str = "") -> str:
        values = _dc_texts(metadata, name)
        return values[0] if values else default

    return PackageMetadata(
        title=first("title"),
        author=first("creator"),
        language=first("language", "zh-CN"),
        date=first("date"),
        source=first("source"),
        description=first("description"),
        subjects=tuple(_dc_texts(metadata, "subject")),
        series=series or None,
        series_position=position or None,
    )


def _desired_package(
    current: PackageMetadata,
    source: SourceMetadata,
) -> PackageMetadata:
    replace_series = source.series_verified
    return PackageMetadata(
        title=source.title or current.title,
        author=source.author or current.author,
        language=source.language or current.language or "zh-CN",
        date=source.date,
        source=source.source,
        description=source.description or current.description,
        subjects=source.subjects or current.subjects,
        series=source.series if replace_series else current.series,
        series_position=(
            str(source.series_position)
            if replace_series and source.series_position is not None
            else (None if replace_series else current.series_position)
        ),
    )


def _changed_fields(
    before: PackageMetadata,
    after: PackageMetadata,
) -> list[str]:
    return [
        name
        for name in (
            "title",
            "author",
            "language",
            "date",
            "source",
            "description",
            "subjects",
            "series",
            "series_position",
        )
        if getattr(before, name) != getattr(after, name)
    ]


def _make_dc(
    local_name: str,
    text: str,
    **attributes: str,
) -> etree._Element:
    element = etree.Element(f"{{{DC_NS}}}{local_name}", **attributes)
    element.text = text
    return element


def _updated_opf(
    opf_data: bytes,
    desired: PackageMetadata,
    *,
    replace_series: bool,
    modified: str,
) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    root = etree.fromstring(opf_data, parser)
    metadata = root.find(f"{{{OPF_NS}}}metadata")
    if metadata is None:
        raise ValueError("EPUB package has no metadata element")
    collection_ids = {
        child.get("id")
        for child in metadata.findall(OPF_META)
        if child.get("property") == "belongs-to-collection" and child.get("id")
    }
    for child in list(metadata):
        if child.tag in TARGET_DC_TAGS:
            metadata.remove(child)
            continue
        if child.tag != OPF_META or not replace_series:
            continue
        if child.get("property") == "belongs-to-collection":
            metadata.remove(child)
            continue
        refines = child.get("refines", "").removeprefix("#")
        if refines in collection_ids:
            metadata.remove(child)

    modified_meta = next(
        (
            child
            for child in metadata.findall(OPF_META)
            if child.get("property") == "dcterms:modified"
        ),
        None,
    )
    if modified_meta is None:
        modified_meta = etree.Element(OPF_META, property="dcterms:modified")
        metadata.insert(0, modified_meta)
    modified_meta.text = modified

    identifier_index = next(
        (
            index
            for index, child in enumerate(metadata)
            if child.tag == f"{{{DC_NS}}}identifier"
        ),
        len(metadata) - 1,
    )
    insert_at = identifier_index + 1
    dc_elements = [
        _make_dc("title", desired.title),
        _make_dc("creator", desired.author, id="creator"),
        _make_dc("language", desired.language),
        _make_dc("date", desired.date),
        _make_dc("source", desired.source),
    ]
    if desired.description:
        dc_elements.append(_make_dc("description", desired.description))
    dc_elements.extend(_make_dc("subject", subject) for subject in desired.subjects)
    for element in dc_elements:
        metadata.insert(insert_at, element)
        insert_at += 1

    if replace_series and desired.series:
        existing_ids = {
            element.get("id")
            for element in metadata.findall(OPF_META)
            if element.get("id")
        }
        collection_id = "c01"
        suffix = 1
        while collection_id in existing_ids:
            suffix += 1
            collection_id = f"c{suffix:02d}"
        collection = etree.Element(
            OPF_META,
            property="belongs-to-collection",
            id=collection_id,
        )
        collection.text = desired.series
        metadata.append(collection)
        collection_type = etree.Element(
            OPF_META,
            refines=f"#{collection_id}",
            property="collection-type",
        )
        collection_type.text = "series"
        metadata.append(collection_type)
        group_position = etree.Element(
            OPF_META,
            refines=f"#{collection_id}",
            property="group-position",
        )
        group_position.text = desired.series_position or "1"
        metadata.append(group_position)

    etree.indent(metadata, space="  ", level=1)
    output = etree.tostring(
        root.getroottree(),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    )
    if opf_data.endswith(b"\n"):
        output += b"\n"
    return output


def _backup_target(path: Path, backup_dir: Path) -> Path:
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve() / "books")
    except ValueError:
        relative = Path(path.name)
    return backup_dir / relative


def _rewrite_epub_member(
    path: Path,
    member: str,
    data: bytes,
) -> None:
    with zipfile.ZipFile(path, "r") as source:
        infos = source.infolist()
        members = {info.filename: source.read(info.filename) for info in infos}
    members[member] = data
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}-metadata-",
        suffix=".epub",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        with zipfile.ZipFile(temporary, "w") as destination:
            for info in infos:
                destination.writestr(info, members[info.filename])
        with zipfile.ZipFile(temporary, "r") as check:
            bad_member = check.testzip()
            if bad_member:
                raise ValueError(f"Corrupt rewritten EPUB member: {bad_member}")
            etree.fromstring(check.read(member))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def enrich_epub_metadata(
    path: Path,
    *,
    apply: bool = True,
    backup_dir: Path | None = DEFAULT_BACKUP_DIR,
    overwrite_backup: bool = False,
    primary_subject: str = DEFAULT_PRIMARY_SUBJECT,
    lookup: MetadataLookup | None = None,
) -> MetadataEnrichmentReport:
    path = Path(path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            opf_member = _find_opf_member(archive)
            opf_data = archive.read(opf_member)
        current = _package_metadata(opf_data)
        if not current.title or not current.author:
            return MetadataEnrichmentReport(
                path=path,
                applied=apply,
                status="unmatched",
                opf_member=opf_member,
                error="EPUB metadata must contain dc:title and dc:creator",
            )
        source = (lookup or MetadataLookup()).find(
            current,
            primary_subject=primary_subject,
        )
        if source is None:
            return MetadataEnrichmentReport(
                path=path,
                applied=apply,
                status="unmatched",
                opf_member=opf_member,
                before=current.to_dict(),
                after=current.to_dict(),
            )
        desired = _desired_package(current, source)
        changes = _changed_fields(current, desired)
        status = "enriched" if changes else "already_complete"
        report = MetadataEnrichmentReport(
            path=path,
            applied=apply,
            status=status,
            provider=source.provider,
            opf_member=opf_member,
            changed_fields=changes,
            before=current.to_dict(),
            after=desired.to_dict(),
            source_url=source.source,
            publication_date=source.date,
        )
        if not changes or not apply:
            return report
        if backup_dir is not None:
            backup = _backup_target(path, backup_dir)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if overwrite_backup or not backup.exists():
                shutil.copy2(path, backup)
            report.backup = backup
        modified = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        updated = _updated_opf(
            opf_data,
            desired,
            replace_series=source.series_verified,
            modified=modified,
        )
        _rewrite_epub_member(path, opf_member, updated)
        return report
    except Exception as error:
        return MetadataEnrichmentReport(
            path=path,
            applied=apply,
            status="error",
            error=f"{type(error).__name__}: {error}",
        )
